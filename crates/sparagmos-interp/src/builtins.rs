//! Runtime support for calls whose bodies exist only in std/core,
//! plus the `core::fmt` formatting engine.
//!
//! The interpreter executes guest functions from converted bodies.
//! Some callees have no body anywhere in the guest program — they
//! *are* the standard library (`std::io::_print`, the `fmt::Arguments`
//! constructors, iterator adapters over std types, ...). Those calls
//! are converted to [`crate::ir::TerminatorKind::CallBuiltin`] and
//! executed here, against the same value model as everything else.
//! This is the main path for printing, not a fallback: there is no
//! other `_print` to fall back to.
//!
//! Design notes:
//! * A `core::fmt::rt::Argument` is a value + formatter function
//!   pointer pair in core. The pointer's target is std code with no
//!   converted body, so it cannot be represented; the pair
//!   (formatting trait, pointee type tag) is stored instead
//!   ([`Value::FmtArg`]). That pair is exactly what core's formatter
//!   dispatch keys on, and the pointee is read through the stored
//!   reference when the arguments are formatted — as the real
//!   `format_args!` reads through it at print time.
//! * `fmt::Arguments` in its placeholder form is an encoded byte
//!   template plus an argument array (core `fmt::Arguments::new`).
//!   The template encoding and the write loop are mirrored from
//!   Rust 1.100.0 `core::fmt` (see `format_template`), and the
//!   padding rules mirror `Formatter::pad` / `pad_integral`.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;

use sparagmos_mir as mir;

use crate::error::{InterpError, InterpResult};
use crate::interp::{
    box_alloc_pointer, channel_handle_id, eval_binary, eval_unary, float_to_int,
    float_to_uint, impl_qualified_self_type, rc_handle_id, round_to_f16,
    synthetic_address, FmtBuilderKind, FmtBuilderState, FmtCapture, Frame,
    Interpreter, Resolved,
};
use crate::ir::{BinOp, Body, FloatTy, IntTy, Operand, Ty, UintTy, UnOp};
use crate::value::{FmtArguments, FmtTrait, PanicPayload, RefProj, RefTarget, Value};

// ---------------------------------------------------------------------------
// Callee names
// ---------------------------------------------------------------------------

/// The canonical form of a callee path: `::<...>` groups that contain
/// only lifetimes (`<'a>`, `<'_>`, `<'a, 'b>`) are removed, so the
/// same function has one name regardless of how the driver rendered
/// its lifetime arguments. Groups with type content (`<Idx>`,
/// `<T, E>`) are kept — dispatch never keys on them, but diagnostics
/// stay precise.
pub fn canonical_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut rest = name;
    while let Some(open) = rest.find('<') {
        // Find the matching '>' (groups may nest).
        let mut depth = 0usize;
        let mut close = None;
        for (i, c) in rest[open..].char_indices() {
            match c {
                '<' => depth += 1,
                '>' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(close) = close else {
            break;
        };
        let inner = &rest[open + 1..close];
        if is_lifetime_list(inner) {
            out.push_str(&rest[..open]);
            if out.ends_with("::") {
                out.truncate(out.len() - 2);
            }
            rest = &rest[close + 1..];
        } else {
            out.push_str(&rest[..=open]);
            rest = &rest[open + 1..];
        }
    }
    out.push_str(rest);
    out
}

fn is_lifetime_list(inner: &str) -> bool {
    !inner.is_empty()
        && inner.split(',').all(|part| {
            let part = part.trim();
            part == "_" || (part.starts_with('\'') && part.len() > 1)
        })
}

/// Render a `sparagmos-mir` type as the source-like tag the builtin
/// dispatch keys on, e.g. `i32`, `&str`, `std::ops::Range<usize>`,
/// `[i32; 4]`. ADT names come from the driver's name table.
pub fn ty_tag(ty: &mir::Ty, names: &HashMap<mir::DefId, String>) -> String {
    use mir::Ty as T;
    match ty {
        T::Bool => "bool".to_string(),
        T::Char => "char".to_string(),
        T::Int(it) => int_ty_name(*it).to_string(),
        T::Uint(ut) => uint_ty_name(*ut).to_string(),
        T::Float(ft) => float_ty_name(*ft).to_string(),
        T::Str => "str".to_string(),
        T::Never => "!".to_string(),
        T::Tuple(ts) => {
            if ts.is_empty() {
                "()".to_string()
            } else {
                let parts: Vec<String> = ts.iter().map(|t| ty_tag(t, names)).collect();
                if parts.len() == 1 {
                    format!("({},)", parts[0])
                } else {
                    format!("({})", parts.join(", "))
                }
            }
        }
        T::Array(elem, len) => {
            let n = match len {
                mir::ArrayLen::Known(n) => n.to_string(),
                _ => "_".to_string(),
            };
            format!("[{}; {n}]", ty_tag(elem, names))
        }
        T::Slice(elem) => format!("[{}]", ty_tag(elem, names)),
        T::Ref(_, inner, mutability) => {
            let m = if matches!(mutability, mir::Mutability::Mut) {
                "mut "
            } else {
                ""
            };
            format!("&{m}{}", ty_tag(inner, names))
        }
        T::RawPtr(inner, _) => format!("*const {}", ty_tag(inner, names)),
        T::Adt(def, args) | T::Closure(def, args) => {
            // A closure type is identified by its definition's
            // name plus its generic arguments (the upvars tuple
            // among them), the same rendering as an ADT instance:
            // two distinct closure types must not share a tag,
            // because the layout table is keyed by it (`size_of` /
            // `size_of_val` of a closure value is a per-type fact).
            let base = names
                .get(def)
                .cloned()
                .unwrap_or_else(|| format!("{def:?}"));
            if args.is_empty() {
                base
            } else {
                let parts: Vec<String> = args.iter().map(|t| ty_tag(t, names)).collect();
                format!("{base}<{}>", parts.join(", "))
            }
        }
        T::FnDef(def, _) => names
            .get(def)
            .cloned()
            .unwrap_or_else(|| format!("{def:?}")),
        T::Coroutine(def, args) => {
            // A coroutine type is named by its definition (the
            // async fn / coroutine literal it was lowered from),
            // with its generic arguments rendered like an ADT's, so
            // the tag identifies one monomorphic coroutine instance.
            // The driver records real layouts under this tag (a
            // coroutine type has no ADT entry), which is what lets
            // the `size_of_val` / `align_of_val` builtins answer
            // for coroutine values.
            let base = names
                .get(def)
                .cloned()
                .unwrap_or_else(|| "Coroutine".to_string());
            if args.is_empty() {
                base
            } else {
                let parts: Vec<String> = args.iter().map(|t| ty_tag(t, names)).collect();
                format!("{base}<{}>", parts.join(", "))
            }
        }
        T::Param(_, name) => name.clone(),
        T::Dynamic(bounds, ..) => {
            // The rendering the driver records in instance
            // names (`dyn std::marker::Send`): the bounds are
            // already trait path strings.
            if bounds.is_empty() {
                "dyn".to_string()
            } else {
                format!("dyn {}", bounds.join(" + "))
            }
        }
        _ => ty.kind_name().to_string(),
    }
}

fn int_ty_name(ty: mir::IntTy) -> &'static str {
    match ty {
        mir::IntTy::I8 => "i8",
        mir::IntTy::I16 => "i16",
        mir::IntTy::I32 => "i32",
        mir::IntTy::I64 => "i64",
        mir::IntTy::I128 => "i128",
        mir::IntTy::Isize => "isize",
    }
}

fn uint_ty_name(ty: mir::UintTy) -> &'static str {
    match ty {
        mir::UintTy::U8 => "u8",
        mir::UintTy::U16 => "u16",
        mir::UintTy::U32 => "u32",
        mir::UintTy::U64 => "u64",
        mir::UintTy::U128 => "u128",
        mir::UintTy::Usize => "usize",
    }
}

fn float_ty_name(ty: mir::FloatTy) -> &'static str {
    match ty {
        mir::FloatTy::F16 => "f16",
        mir::FloatTy::F32 => "f32",
        mir::FloatTy::F64 => "f64",
        mir::FloatTy::F128 => "f128",
    }
}

// ---------------------------------------------------------------------------
// The builtin registry
// ---------------------------------------------------------------------------

/// A runtime-supported callee. Conversion consults [`lookup`]; a
/// callee with no converted body and no entry here stays an explicit
/// unsupported-call error naming the function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Builtin {
    /// `std::io::_print(Arguments)`: format and append to guest stdout.
    Print,
    /// `std::io::_eprint(Arguments)`: format and append to guest stderr.
    Eprint,
    /// `std::env::args() -> Args`: the run's argument iterator
    /// (argv[0] plus the configured guest arguments).
    EnvArgs,
    /// `std::env::args_os() -> ArgsOs`: the run's argument iterator
    /// over `OsString` values, which are the string values
    /// themselves in this model (guest arguments are text).
    EnvArgsOs,
    /// `std::env::current_exe() -> io::Result<PathBuf>`: the
    /// host executable's path (the driver binary's own path —
    /// the guest observes a real, existing path, as it would
    /// from the OS).
    EnvCurrentExe,
    /// `std::env::var::inner(&key) -> Result<String, VarError>`:
    /// the non-generic leaf under `std::env::var`, over the host
    /// environment (see `EnvGetenv`).
    EnvVarInner,
    /// `std::env::split_paths(&OsStr) -> SplitPaths`: the
    /// parts split on the platform path separator (`:` on
    /// unix, `slice::Split` semantics -- empty segments are
    /// kept, including a trailing one), snapshotted as an
    /// owned-iterator state of `PathBuf` values.
    EnvSplitPaths,
    /// `std::env::join_paths(iterable) -> Result<OsString,
    /// JoinPathsError>`: concatenates the items' bytes with
    /// the platform path separator (`:` on unix), failing if
    /// any item contains the separator. The native body
    /// composes iterator adapters whose inlined slice-iterator
    /// machinery reconstructs slices from raw pointers, which
    /// this value model cannot dereference; the builtin steps
    /// the modelled iterator state directly (the pattern
    /// `Iterator::count` uses) and extracts each item's bytes
    /// through the `OsStr` argument funnel.
    EnvJoinPaths,
    /// `std::sys::env::unix::getenv(&key) -> Option<OsString>`:
    /// the host process environment, read live (a run-make
    /// recipe's tool invocations are configured through it).
    EnvGetenv,
    /// `std::sys::env::unix::setenv(&key, &value) ->
    /// io::Result<()>`: mutate the host process environment, as
    /// the real call does.
    EnvSetenv,
    /// `std::sys::env::unix::unsetenv(&key) -> io::Result<()>`:
    /// remove a variable from the host process environment.
    EnvUnsetenv,
    /// `std::env::current_dir()` (and the
    /// `std::sys::paths::unix::getcwd` leaf under it): the host
    /// process's working directory. A recipe runs in a real host
    /// directory, so the host value is the faithful one.
    EnvCurrentDir,
    /// `std::env::temp_dir()` (and its `std::sys::paths` leaf):
    /// the host's temporary directory.
    EnvTempDir,
    /// `std::env::home_dir()` (and its `std::sys::paths` leaf):
    /// the host's home directory, when it has one.
    EnvHomeDir,
    /// `std::sys::paths::unix::chdir(&path) -> io::Result<()>`:
    /// change the host process's working directory (what
    /// `std::env::set_current_dir` bottoms out in).
    SysChdir,
    /// `std::fs::read::inner(&path) -> io::Result<Vec<u8>>`: the
    /// non-generic leaf under `std::fs::read`, performed on the
    /// host filesystem (the fs sys layer's model).
    FsReadInner,
    /// `std::fs::read_to_string::inner(&Path) ->
    /// io::Result<String>`: the host read plus the UTF-8
    /// validation (an invalid file is `InvalidData`, exactly
    /// the error the native `read_to_string` reports).
    FsReadToStringInner,
    /// `String::from_utf8(Vec<u8>) -> Result<String,
    /// FromUtf8Error>`: the host validation over the vector's
    /// elements. (The native body's construction path walks
    /// the `String`/`Vec` field layout, which the element-list
    /// value model does not have.)
    StringFromUtf8,
    /// `std::fs::write::inner(&path, &[u8]) -> io::Result<()>`:
    /// the non-generic leaf under `std::fs::write`.
    FsWriteInner,
    /// `std::path::Path::parent(&self) -> Option<&Path>`: the
    /// lexical parent, a prefix of the path's own bytes.
    PathParent,
    /// `std::path::Path::file_name(&self) -> Option<&OsStr>`:
    /// the final component when it is a normal one.
    PathFileName,
    /// `std::path::Path::file_stem(&self) -> Option<&OsStr>`.
    PathFileStem,
    /// `std::path::Path::extension(&self) -> Option<&OsStr>`.
    PathExtension,
    /// `std::path::Path::_starts_with(&self, &Path) -> bool`:
    /// the component-wise prefix test under `Path::starts_with`.
    PathStartsWith,
    /// `std::path::Path::_ends_with(&self, &Path) -> bool`:
    /// the component-wise suffix test under `Path::ends_with`.
    PathEndsWith,
    /// `std::path::Path::is_absolute(&self) -> bool`: the path
    /// has a root component.
    PathIsAbsolute,
    /// `std::path::Path::_join(&self, &Path) -> PathBuf`: the
    /// owned path with the argument pushed onto it (the leaf
    /// under `Path::join`).
    PathJoin,
    /// `std::fs::OpenOptions::_open(&options, path)`: the fs
    /// sys layer's entry point. A path containing a NUL byte is
    /// rejected with `InvalidInput`, exactly as std's own
    /// CString conversion rejects it; opening a real file needs
    /// the File value model (the sys File layout and its method
    /// bodies), which is the recorded next stage, so a
    /// NUL-free open is an explicit unsupported error.
    FsOpen,
    /// `std::fs::DirBuilder::_create(&builder, path)`: create
    /// one directory on the host filesystem (std's
    /// `create_dir_all` body iterates this per component).
    FsDirCreate,
    /// `std::string::String::from_utf8_lossy(&[u8]) ->
    /// Cow<str>`: the host conversion, verbatim (borrowed when
    /// the bytes are valid UTF-8, owned with replacement
    /// characters otherwise; in this model both sides carry the
    /// string data itself).
    StringFromUtf8Lossy,
    /// `std::str::from_utf8(&[u8]) -> Result<&str, Utf8Error>`:
    /// the host validation, verbatim; the error is the plain
    /// struct value (valid-up-to count, optional error length).
    StrFromUtf8,
    /// `to_string_lossy` on an `OsString`, `OsStr` or `Path`
    /// receiver: the lossy conversion of the receiver's bytes
    /// as a `Cow<str>` (the same conversion
    /// `StringFromUtf8Lossy` performs, over bytes read from the
    /// receiver instead of a byte slice).
    ToStringLossy,
    /// `std::fs::DirEntry::path(&entry) -> PathBuf`: the full
    /// path the modelled entry was snapshotted with.
    DirEntryPath,
    /// `std::fs::DirEntry::file_name(&entry) -> OsString`: the
    /// entry's final component.
    DirEntryFileName,
    /// `std::fs::DirEntry::metadata(&entry) ->
    /// io::Result<Metadata>`: the host metadata of the entry's
    /// path, following symlinks as std's does.
    DirEntryMetadata,
    /// A `std::sys::fs` leaf operation, performed on the host
    /// filesystem: the arguments and results are plain values
    /// (paths as byte strings, `io::Error` in std's own shape),
    /// so the converted public `std::fs` bodies above them run
    /// unchanged.
    FsSys(FsSysOp),
    /// `std::process::exit(code)`: end the run with the given
    /// exit code (see `InterpError::ProcessExit`).
    ProcessExit,
    /// `std::io::stdout() -> Stdout`: the standard output handle.
    /// The handle value is a marker in this model: writes through
    /// it are modelled by the `Write` builtins, which dispatch on
    /// the receiver type, not on the handle's contents.
    Stdout,
    /// `std::io::Write::write_fmt(&mut self, Arguments)` for the
    /// modelled receivers (the std handles and `Vec<u8>`).
    WriteFmt,
    /// `core::fmt::write(&mut dyn Write, Arguments)`: format the
    /// arguments (the same model `format_arguments` implements)
    /// and deliver the text through the output's own `write_str`.
    FmtWrite,
    /// `std::fmt::format(Arguments) -> String`.
    Format,
    /// `std::fmt::Arguments::from_str(&str)`.
    ArgumentsFromStr,
    /// `std::fmt::Arguments::new(&[u8; N], &[Argument; M])`.
    ArgumentsNew,
    /// `core::fmt::rt::Argument::new_*(ref)`.
    ArgumentNew(FmtTrait),
    /// A `core::fmt` trait method itself (`Display::fmt`,
    /// `Debug::fmt`, ...) called by name from a guest body
    /// (a UFCS call) whose `Self` instance has no converted
    /// body: the structural fmt engine formats the receiver
    /// into the captured `Formatter` argument. Deliberately
    /// absent from `lookup` (see [`fmt_trait_method`]): a
    /// collected impl body always wins over this model.
    FmtMethod(FmtTrait),
    /// `core::fmt::rt::Argument::from_usize(&usize)`.
    ArgumentFromUsize,
    /// `std::fmt::Formatter::write_str(&mut self, &str)`: append
    /// the string to the captured formatter's buffer.
    FormatterWriteStr,
    /// `std::fmt::Formatter::write_char(&mut self, char)`.
    FormatterWriteChar,
    /// `std::fmt::Formatter::pad(&mut self, &str)`: core's
    /// padding write into the capture (precision truncates to
    /// that many chars, width pads with the fill).
    FormatterPad,
    /// `std::fmt::Formatter::write_fmt(&mut self, Arguments)`:
    /// format the arguments into the captured buffer.
    FormatterWriteFmt,
    /// `std::fmt::Formatter::alternate(&self) -> bool`.
    FormatterAlternate,
    /// `std::fmt::Formatter::width(&self) -> Option<usize>`.
    FormatterWidth,
    /// `std::fmt::Formatter::precision(&self) -> Option<usize>`.
    FormatterPrecision,
    /// `std::fmt::Formatter::fill(&self) -> char`.
    FormatterFill,
    /// `std::fmt::Formatter::sign_plus(&self) -> bool`.
    FormatterSignPlus,
    /// `std::fmt::Formatter::sign_aware_zero_pad(&self) -> bool`.
    FormatterSignAwareZeroPad,
    /// `Formatter::debug_struct_fieldN_finish` (derive's fast
    /// path for structs of at most five fields): render the
    /// whole struct value from the name and the interleaved
    /// field-name / `&dyn Debug` arguments into the capture.
    DebugStructFinish { fields: usize },
    /// `Formatter::debug_struct_fields_finish` (the same for
    /// larger structs, with parallel name/value slices).
    DebugStructFieldsFinish,
    DebugTupleFieldsFinish,
    /// `Formatter::debug_tuple_fieldN_finish`: the tuple-struct
    /// counterpart of [`Builtin::DebugStructFinish`].
    DebugTupleFinish { fields: usize },
    /// `Formatter::debug_list` / `debug_set` / `debug_map` /
    /// `debug_struct` / `debug_tuple`: open a builder over the
    /// captured formatter.
    FmtBuilderNew(FmtBuilderKind),
    /// `DebugList::entry` / `DebugSet::entry`: append one entry
    /// to the builder.
    FmtBuilderEntry,
    /// `DebugMap::entry`: append one key/value entry.
    FmtBuilderMapEntry,
    /// `DebugStruct::field`: append one named field.
    FmtBuilderStructField,
    /// `DebugTuple::field`: append one field.
    FmtBuilderTupleField,
    /// A builder's `finish` (and `DebugStruct`'s
    /// `finish_non_exhaustive`): close the rendered value.
    FmtBuilderFinish { exhaustive: bool },
    /// `std::ops::RangeInclusive::new(start, end)`.
    RangeInclusiveNew,
    /// `std::iter::IntoIterator::into_iter(value)`.
    IntoIter,
    /// `std::iter::Iterator::next(&mut iter)`.
    IteratorNext,
    /// `std::iter::DoubleEndedIterator::next_back(&mut iter)`.
    IteratorNextBack,
    /// `std::iter::Iterator::size_hint(&iter)`.
    IteratorSizeHint,
    /// `std::iter::Iterator::count(iter)`: drive the modelled
    /// `next` state transition to exhaustion and count the items.
    /// (Core's own default body is a `fold`, and several iterator
    /// types override `count` with pointer machinery over the
    /// remaining items; the observable result is the same count.)
    IteratorCount,
    /// `std::iter::Iterator::fold(iter, init, f)`: the core
    /// default body is a `next` loop applying `f` per item;
    /// modelled by driving the `next` state transition and
    /// invoking `f` synchronously per item.
    IteratorFold,
    /// `std::iter::Iterator::for_each(iter, f)`: a `next` loop
    /// invoking `f` per item, discarding the results.
    IteratorForEach,
    IteratorCollect,
    FromIteratorFromIter,
    /// A primitive arithmetic trait method served by the
    /// run-time name fallback (`<i32 as std::ops::Add>::add` and
    /// kin): the computation is the interpreter's own binary
    /// evaluation at the Self type the name carries.
    PrimitiveBinOp(BinOp),
    /// The unary counterpart (`<i32 as std::ops::Neg>::neg`,
    /// `<bool as std::ops::Not>::not`).
    PrimitiveUnOp(UnOp),
    /// `std::convert::From::from(value)` (modelled conversions only).
    FromFrom,
    /// `std::hint::must_use(value)`: identity.
    MustUse,
    /// `std::sys::thread_local::destructors::linux_like::register`:
    /// record the registration (the storage reference and the
    /// destructor callable); the interpreter runs the recorded
    /// destructors when the registering thread finishes. The
    /// entry thread's registrations never run, matching std's
    /// documented caveat that TLS destructors do not run for the
    /// main thread on Unix.
    TlsRegisterDtor,
    /// `std::thread::local::panic_access_error(AccessError)`: the
    /// panic raised when a destroyed thread-local is accessed. It
    /// cannot fire in this model (thread-locals are never
    /// destroyed); the builtin exists so the name has the exact
    /// native behaviour if reached.
    PanicAccessError,
    /// `std::ops::Deref::deref(&value)` (modelled types only).
    Deref,
    /// `str::to_uppercase(&str) -> String`.
    ToUppercase,
    /// `str::to_lowercase(&str) -> String`.
    ToLowercase,
    /// `String::new()`: the empty string. (`String` is the string
    /// data itself in this value model, as for every method here:
    /// the collected std bodies project the buffer/length/capacity
    /// fields of the real struct, which the abstract value does not
    /// have, so the methods are modelled here instead.)
    StringNew,
    /// `String::with_capacity(cap)`: the empty string. The
    /// collected body inlines down to
    /// `RawVecInner::with_capacity_in`, which has no converted
    /// body, so the constructor is modelled here like the other
    /// `String` constructors. The requested capacity is not
    /// represented (the same convention as `Vec::with_capacity`,
    /// including not modelling the capacity-overflow panic);
    /// nothing a caller can observe changes, because `capacity`
    /// itself is uniformly the explicit `StringCapacity` error.
    StringWithCapacity,
    /// `String::len(&String)`: the byte length of the string data.
    StringLen,
    /// `String::is_empty(&String)`.
    StringIsEmpty,
    /// `String::capacity(&String)`: allocation capacity is not
    /// tracked by the abstract string model, so the call stays an
    /// explicit unsupported error naming the method rather than a
    /// projection failure inside the collected body.
    StringCapacity,
    /// `String::as_str(&String) -> &str`: the string data itself
    /// (the same value `Deref` yields for a `String`).
    StringAsStr,
    /// `String::as_bytes(&String) -> &[u8]`: a byte view of the
    /// string data.
    StringAsBytes,
    /// `String::into_bytes(String) -> Vec<u8>`: the string data as
    /// a vector of bytes (the model's element aggregate).
    StringIntoBytes,
    /// `String::into_boxed_str(String) -> Box<str>`: the string
    /// data in a fresh box allocation. The collected body moves
    /// the `vec` field out of the string value and rebuilds the
    /// box from the vector's raw parts, neither of which the
    /// string model can serve; the box over the string value
    /// itself is the same observable value.
    StringIntoBoxedStr,
    /// `Vec::into_boxed_slice(Vec<T>) -> Box<[T]>`: the element
    /// aggregate in a fresh box allocation, for the same reason
    /// as `String::into_boxed_str`.
    VecIntoBoxedSlice,
    /// `AsRef::as_ref(&Self) -> &Target`: a view of the receiver's
    /// data as the target type. The collected std bodies build
    /// the view from the receiver's raw parts (the string's
    /// pointer and length read field by field), which the value
    /// model cannot serve; the view itself is a direct function
    /// of the modelled data. Modelled receiver types are served
    /// here; anything else is an explicit unsupported error.
    AsRef,
    /// `AsMut::as_mut(&mut Self) -> &mut Target`: the mutable
    /// counterpart of `AsRef`; for the modelled receivers the
    /// receiver place itself is the viewed storage, so the
    /// argument reference is returned unchanged.
    AsMut,
    /// `alloc::__rust_no_alloc_shim_is_unstable_v2()`: the alloc
    /// shim's link marker. Allocation in the model never goes
    /// through the shim symbols (the allocation store is the
    /// allocator), so the marker call is a no-op.
    NoAllocShimMarker,
    /// `String::push_str(&mut String, &str)`.
    StringPushStr,
    /// `String::push(&mut String, char)`.
    StringPush,
    /// `str::len(&str)`: the byte length of the string data.
    StrLen,
    /// `str::is_empty(&str)`.
    StrIsEmpty,
    /// `str::as_bytes(&str) -> &[u8]`: a byte view of the string
    /// data.
    StrAsBytes,
    /// `str::chars(&str)`: a char iterator state (the char values
    /// plus the next index), the shape `Iterator::next` consumes
    /// for `std::str::Chars`.
    StrChars,
    /// `str::bytes(&str)`: a byte iterator state, the shape
    /// `Iterator::next` consumes for `std::str::Bytes`.
    StrBytes,
    /// `str::make_ascii_uppercase(&mut str)` /
    /// `str::make_ascii_lowercase(&mut str)`: in-place ASCII case
    /// conversion of the string data. The collected bodies go
    /// through `as_bytes_mut` and write byte by byte, which the
    /// string model cannot serve (a string has no byte slots);
    /// the conversion itself is exact on the model's string.
    StrMakeAsciiCase(bool),
    /// `str::trim` / `trim_start` / `trim_end` (`&str -> &str`):
    /// strip `char::is_whitespace` from the selected side(s); the
    /// result is the stripped string data.
    StrTrim(TrimSide),
    /// `core::panicking::panic(&str)`: raise a guest panic.
    PanicStr,
    /// `std::rt::panic_fmt(Arguments)` / `core::panicking::panic_fmt`:
    /// raise a guest panic with the formatted message.
    PanicFmt,
    /// `core::panicking::assert_failed(kind, &left, &right, args)`:
    /// raise a guest panic reporting the failed comparison.
    AssertFailed,
    /// `core::panicking::panic_nounwind_fmt(Arguments, bool)`:
    /// raise a guest panic with the formatted message. The flag
    /// only controls native backtrace capture, which no guest
    /// program can observe.
    PanicNounwindFmt,
    /// `core::option::unwrap_failed()`: raise a guest panic with
    /// core's fixed `Option::unwrap` message.
    OptionUnwrapFailed,
    /// `core::result::unwrap_failed(&str, &dyn Debug)`: raise a
    /// guest panic `"{msg}: {error:?}"`, core's message.
    ResultUnwrapFailed,
    /// `core::num::from_ascii_bytes_radix_panic(radix)`: raise a
    /// guest panic naming the out-of-range radix.
    FromAsciiRadixPanic,
    /// `core::slice::index::slice_index_fail(start, end, len)`:
    /// raise a guest panic with the message core's branch order
    /// selects for the failing range.
    SliceIndexFail,
    /// `core::slice::memchr::memchr(needle, haystack) ->
    /// Option<usize>` (and `memrchr`, searching from the
    /// back): a linear scan over the haystack's elements.
    /// The native word-at-a-time implementation walks raw
    /// pointers whose distances this value model cannot
    /// express for value-held slices; the scan computes the
    /// same result.
    Memchr { reverse: bool },
    /// `core::slice::index::RangeError::report(self, len)`: core
    /// forwards to `slice_index_fail`; the same builtin shape.
    RangeErrorReport,
    /// `Vec::remove`'s nested `assert_failed(index, len)`: raise
    /// a guest panic with alloc's removal-index message.
    VecRemoveAssertFailed,
    /// `std::thread::spawn(f)` / `spawn_unchecked(f)`: capture the
    /// callable as a pending thread (cooperative model, see
    /// `interp.rs`) and return its join handle.
    ThreadSpawn,
    /// `std::thread::Builder::spawn(builder, f)` /
    /// `spawn_unchecked(builder, f)`: `ThreadSpawn` with a builder
    /// (the builder's name and stack size are not observable in the
    /// value model beyond the thread's name, which is not tracked).
    ThreadBuilderSpawn,
    /// `std::thread::JoinHandle::join(handle)`: run the thread if
    /// it is still pending and return its outcome as a `Result`.
    ThreadJoin,
    /// `std::thread::ScopedJoinHandle::join(handle)`.
    ScopedThreadJoin,
    /// `std::thread::Scope::spawn(&scope, f)`: `ThreadSpawn` whose
    /// handle the enclosing `std::thread::scope` joins at scope
    /// exit.
    ThreadScopedSpawn,
    /// `std::thread::scope(f)`: call `f` with a fresh scope value,
    /// then run every pending thread (the scope's threads must
    /// finish before the scope returns).
    ThreadScope,
    /// `std::thread::current()`: the executing thread's value.
    ThreadCurrent,
    /// `std::thread::Thread::id(&thread)`: the thread's id value.
    ThreadIdOf,
    /// `std::thread::sleep(dur)` / `sleep_ms(ms)`: a cooperative
    /// scheduling point (pending threads run to completion); the
    /// passage of time itself is not observable in the value model.
    ThreadSleep,
    /// `std::thread::yield_now()`: a cooperative scheduling point.
    ThreadYield,
    /// `std::sync::mpsc::channel()`: a new unbounded channel.
    MpscChannel,
    /// `std::sync::mpsc::sync_channel(bound)`: a new bounded channel.
    MpscSyncChannel,
    /// `Sender::send(&self, value)` / `SyncSender::send(&self, value)`.
    MpscSend,
    /// `SyncSender::try_send(&self, value)`.
    MpscTrySend,
    /// `Receiver::recv(&self)`.
    MpscRecv,
    /// `Receiver::try_recv(&self)`.
    MpscTryRecv,
    /// `Receiver::recv_timeout(&self, dur)`.
    MpscRecvTimeout,
    /// `Receiver::iter(&self)`: an iterator over the channel's
    /// incoming values.
    MpscIter,
    /// `std::thread::Builder::new()`: a builder value in the real
    /// field shape (`name`, `stack_size`, both unset).
    ThreadBuilderNew,
    /// `std::thread::Builder::name(builder, name)`: the builder
    /// with its name field set.
    ThreadBuilderName,
    /// `std::thread::Builder::stack_size(builder, size)`: the
    /// builder with its stack-size field set.
    ThreadBuilderStackSize,
    /// `std::time::Duration::new(secs, nanos)`: a duration value
    /// in the real field shape (`secs`, `nanos`, normalized).
    DurationNew,
    /// `std::time::Duration::from_secs(n)`.
    DurationFromSecs,
    /// `std::time::Duration::from_millis(n)`.
    DurationFromMillis,
    /// `std::time::Duration::from_micros(n)`.
    DurationFromMicros,
    /// `std::time::Duration::from_nanos(n)`.
    DurationFromNanos,
    /// `std::intrinsics::caller_location()`: the caller location
    /// of the current function (the received stamp for a
    /// `#[track_caller]` function, the stamped call site
    /// otherwise).
    CallerLocation,
    /// `std::panic::set_hook(hook)`: install the panic hook.
    PanicSetHook,
    /// `std::panic::take_hook()`: take the panic hook, leaving
    /// the default in its place.
    PanicTakeHook,
    /// `test::test_main_env_args(&[TestDescAndFn]) -> ExitCode`:
    /// the libtest entry point. The compiler-generated harness
    /// main calls a crate-local wrapper of this name, so the
    /// registry matches the name's suffix.
    TestMainEnvArgs,
    /// `std::time::Instant::now()`: the host monotonic clock.
    InstantNow,
    /// `std::time::SystemTime::now()`: the host wall clock.
    SystemTimeNow,
    /// `std::sys::random::linux::hashmap_random_keys()`: the two
    /// SipHash keys a fresh `std::hash::RandomState` is seeded
    /// with. Std draws them from OS entropy; the model draws them
    /// from the host clock and a per-process counter (see
    /// `fresh_hash_keys`).
    HashmapRandomKeys,
    /// `std::sys::process::unix::common::Command::new(&OsStr)`:
    /// begin a modelled process command. The returned value is
    /// a handle carrying the id of the builder state in the
    /// interpreter's command table (see [`CommandState`]); the
    /// sys `Command` methods below record into that state. No
    /// child process is ever started (see `ProcessSpawn`).
    ProcessCommandNew,
    /// `std::process::Command::new(program)`: the public
    /// constructor. The public `Command` is a one-field struct
    /// over the sys command, so the value is the sys handle
    /// wrapped in one aggregate layer. (The public method
    /// bodies are not available to conversion either, so the
    /// public builder surface is modelled here directly,
    /// alongside the sys layer other std code calls.)
    ProcessCommandNewPublic,
    /// `std::process::Command::arg(&mut self, arg) -> &mut
    /// Command`: record one argument; returns the receiver.
    ProcessCommandArgPublic,
    /// `std::process::Command::current_dir(&mut self, dir) ->
    /// &mut Command`: record the child's working directory;
    /// returns the receiver.
    ProcessCommandCwdPublic,
    /// `std::process::Command::env(&mut self, key, val) ->
    /// &mut Command`: record an environment entry.
    ProcessCommandEnvPublic,
    /// `std::process::Command::env_remove(&mut self, key) ->
    /// &mut Command`: record an environment removal.
    ProcessCommandEnvRemovePublic,
    /// `std::process::Command::env_clear(&mut self) -> &mut
    /// Command`: record that the environment was cleared.
    ProcessCommandEnvClearPublic,
    /// `std::process::Command::stdin` / `stdout` / `stderr`
    /// `(&mut self, cfg) -> &mut Command`: accepted and dropped
    /// like the sys layer's; returns the receiver.
    ProcessCommandStdioPublic,
    /// `std::process::Command::output(&mut self) ->
    /// io::Result<Output>` and `std::sys::process::output`:
    /// the resolution failure a spawn would report; a program
    /// that would start is a named unsupported error.
    ProcessOutput,
    /// `std::sys::process::unix::common::Command::arg(&mut self,
    /// &OsStr)`: record one argument on the command.
    ProcessCommandArg,
    /// `std::sys::process::unix::common::Command::cwd(&mut self,
    /// &OsStr)`: record the child's working directory.
    ProcessCommandCwd,
    /// `std::sys::process::unix::common::Command::env_mut(&mut
    /// self) -> &mut CommandEnv`: the returned reference denotes
    /// a handle value carrying the command's id, which the
    /// `CommandEnv` builtins read back.
    ProcessCommandEnvMut,
    /// `std::sys::process::env::CommandEnv::set(&mut self, &OsStr,
    /// &OsStr)`: record an environment entry on the command.
    ProcessCommandEnvSet,
    /// `std::sys::process::env::CommandEnv::remove(&mut self,
    /// &OsStr)`: record an environment removal on the command.
    ProcessCommandEnvRemove,
    /// `std::sys::process::env::CommandEnv::clear(&mut self)`:
    /// record that the command's environment was cleared.
    ProcessCommandEnvClear,
    /// `std::sys::process::unix::common::Command::stdin` /
    /// `stdout` / `stderr` `(&mut self, Stdio)`: the stdio
    /// configuration is not observed by the process model (no
    /// child is ever started, so no descriptor is ever wired),
    /// so it is accepted and dropped.
    ProcessCommandStdio,
    /// `std::sys::process::unix::unix::Command::spawn(&mut self,
    /// Stdio, bool) -> io::Result<(Process, StdioPipes)>`:
    /// resolve the program against the command's state; a
    /// resolution failure is the `io::Error` std produces for it,
    /// and a program that would start is a named unsupported
    /// error, because the model does not start child processes.
    ProcessSpawn,
    /// `std::env::current_exe() -> io::Result<PathBuf>`: the
    /// path of the executable running the guest program, which
    /// is the driver host process's own executable path.
    CurrentExe,
    /// `std::process::id() -> u32`: the id of the process
    /// running the guest program (the driver host process).
    ProcessId,
    /// `core::io::error::os_functions::get_os_functions()`:
    /// the OS error function table std registers for `core`'s
    /// `io::Error` machinery. In a native run std installs its
    /// table before any error is decoded (the guest's own
    /// registration path, `set_functions`, has no converted
    /// body); the model answers with the table directly, its
    /// entries naming the model functions below.
    IoGetOsFunctions,
    /// `core::io::error::os_functions::set_functions(...)` and
    /// its inner helper: registering the OS error table. A
    /// no-op in the model: `get_os_functions` always answers
    /// the model's table, which carries the same entries std
    /// would register.
    IoSetOsFunctions,
    /// `std::sys::io::error::unix::decode_error_kind(errno)`:
    /// std's unix errno -> `ErrorKind` table (the entry the OS
    /// error function table names).
    IoDecodeErrorKind,
    /// `std::sys::io::error::unix::format_error(errno, fmt)`:
    /// write the OS error message into the formatter. The text
    /// is the platform's own message for the code (see
    /// `os_error_message`), exactly what std's native
    /// implementation writes through `strerror_r`.
    IoFormatOsError,
    /// `std::sys::io::error::unix::is_interrupted(errno)`:
    /// the error is `EINTR`.
    IoIsInterrupted,
    /// `Instant::checked_add` / `SystemTime::checked_add`
    /// (payload: true for `SystemTime`).
    TimeCheckedAdd(bool),
    /// `Instant::checked_sub` / `SystemTime::checked_sub`
    /// (payload: true for `SystemTime`).
    TimeCheckedSub(bool),
    /// `std::time::Instant::duration_since(earlier)`.
    InstantDurationSince,
    /// `std::time::Instant::checked_duration_since(earlier)`.
    InstantCheckedDurationSince,
    /// `std::time::Instant::saturating_duration_since(earlier)`.
    InstantSaturatingDurationSince,
    /// `std::time::Instant::elapsed()`.
    InstantElapsed,
    /// `std::time::SystemTime::duration_since(earlier)`.
    SystemTimeDurationSince,
    /// `std::time::SystemTime::elapsed()`.
    SystemTimeElapsed,
    /// `std::intrinsics::catch_unwind(try_fn, data, catch_fn) -> bool`:
    /// run the try function to completion; on a guest panic, roll
    /// the frame stack back to the call and run the catch function
    /// with the panic's raw payload, reporting `true`.
    CatchUnwind,
    /// `std::panicking::catch_unwind::cleanup(payload)`: box a raw
    /// panic payload into a `Box<dyn Any + Send>` (a `DynRef` in
    /// this value model). The native cleanup also decrements the
    /// panic count; no guest program can observe that counter
    /// through the modelled surface (`std::panicking()` is not
    /// modelled), so there is nothing further to do.
    PanicCleanup,
    /// `std::panicking::begin_panic(payload)` /
    /// `std::rt::begin_panic(payload)`: raise a guest panic
    /// carrying the argument as its payload. This is the sink
    /// `panic_any` lowers to; modelling it here (rather than
    /// executing std's converted body) keeps the payload intact —
    /// the body routes it through the panic-hook machinery, which
    /// has no observable effect beyond the panic itself.
    PanicAny,
    /// `std::panic::resume_unwind(box)` /
    /// `std::panicking::resume_unwind(box)`: re-raise a caught
    /// payload as a guest panic carrying the same payload.
    ResumeUnwind,
    /// `<(dyn Any + ...)>::downcast_ref::<T>(&self) -> Option<&T>`:
    /// `Some` exactly when the payload's concrete type is `T`.
    DynAnyDowncastRef,
    /// `<(dyn Any + ...)>::downcast::<T>(self) -> Result<T, Box>`
    /// (and `Box<dyn Any + ...>::downcast`, the inherent method
    /// guest code actually calls): the payload value on a type
    /// match, the box back otherwise. The payload is the generic
    /// argument at the given index: 0 for the dyn method itself,
    /// 1 for the `Box` impl method, whose first generic argument
    /// is the allocator.
    DynAnyDowncast(usize),
    /// `<(dyn Any + ...)>::is::<T>(&self) -> bool`: whether the
    /// payload's concrete type is `T`.
    DynAnyIs,
    /// `PartialEq::eq(&a, &b)`: semantic (derived) value equality.
    PartialEqEq,
    /// `PartialEq::ne(&a, &b)`.
    PartialEqNe,
    /// `PartialOrd::partial_cmp(&a, &b) -> Option<Ordering>`.
    PartialCmp,
    /// `Ord::cmp(&a, &b) -> Ordering`.
    OrdCmp,
    /// `PartialOrd::lt/le/gt/ge(&a, &b) -> bool`.
    PartialOrdRel(RelOp),
    /// `Clone::clone(&value)`: a copy of the pointee value.
    CloneClone,
    /// `Default::default()`: the default value of the type.
    DefaultDefault,
    /// `ToString::to_string(&value)` for the modelled `Display` types.
    ToStringToString,
    /// `Box::new(value)`: the box is an ADT aggregate whose single
    /// field is the pointer to the allocation holding its pointee
    /// (see `Interpreter::new_box`).
    BoxNew,
    /// `Box::new_uninit()`: a box whose pointee is uninitialized,
    /// with the shape its type fixes (an array pointee is a tuple of
    /// uninitialized elements), allocated like `Box::new`.
    BoxNewUninit,
    /// `box_assume_init_into_vec_unsafe(box)`: the `vec!` expansion's
    /// final step — the box's array pointee becomes a `Vec` value
    /// (an ADT aggregate of its elements).
    BoxAssumeInitIntoVec,
    /// `Vec::new()`: an empty vector (ADT aggregate of its elements).
    VecNew,
    /// `Vec::push(&mut vec, value)`.
    VecPush,
    /// `Vec::extend_from_slice(&mut self, &[T])`: appends a
    /// clone of the slice's elements. The native body walks
    /// raw buffer pointers through `ptr::copy_nonoverlapping`,
    /// whose unsafe-precondition checks order pointers within
    /// one allocation -- an order the synthetic-address model
    /// does not define -- so the append is computed directly
    /// on the element list, exactly as `Vec::push` is.
    VecExtendFromSlice,
    /// `Vec::len(&vec)`.
    VecLen,
    /// `Vec::is_empty(&vec)`.
    VecIsEmpty,
    /// `Vec::with_capacity(cap)` / `Vec::with_capacity_in(cap, alloc)`:
    /// an empty vector. The allocation is not observable in this
    /// value model, which does not represent capacity (the same
    /// reason `Vec::capacity` is not modelled).
    VecWithCapacity,
    /// `std::vec::from_elem(elem, n)`: a vector of `n` copies of
    /// `elem` (the `vec![x; n]` form). Values are immutable in this
    /// model, so the copies repeat the one value and each drops
    /// independently. That is exact for `Copy` elements (the
    /// specialization the real body selects for them performs no
    /// `clone` calls) and for elements whose `Clone` is the plain
    /// field-wise one; a `Clone` impl with side effects or one that
    /// produces a different value is not run per copy.
    VecFromElem,
    /// `Vec::reserve(&mut vec, additional)` /
    /// `Vec::reserve_exact(&mut vec, additional)`: no-ops. Growing
    /// the allocation is not observable in this value model, which
    /// does not represent capacity (the same convention as
    /// `Vec::with_capacity`).
    VecReserve,
    /// `Vec::try_reserve(&mut vec, additional)` /
    /// `Vec::try_reserve_exact(&mut vec, additional)`:
    /// `Result<(), TryReserveError>`. The collected body bottoms
    /// out in `RawVecInner::try_reserve`, which has no converted
    /// body. In the abstract model allocation never fails, so the
    /// only error the real body can produce is
    /// `TryReserveError::CapacityOverflow`, decided by the exact
    /// std rule: the required element count `len + additional`
    /// must fit `isize::MAX / size_of::<T>()` elements (any count
    /// up to `usize::MAX` for a zero-sized element). The two
    /// spellings differ only in the capacity they would reach,
    /// which the value model does not represent. An element type
    /// whose layout the model does not know stays an explicit
    /// unsupported error rather than a guessed verdict.
    VecTryReserve,
    /// `Vec::shrink_to_fit(&mut vec)` / `Vec::shrink_to(&mut vec,
    /// min_capacity)`: no-ops. Shrinking changes only the
    /// allocation's capacity, which the value model does not
    /// represent and no modelled operation can observe
    /// (`Vec::capacity` is the explicit `VecCapacity` error,
    /// growth is the `VecReserve` no-op). The collected bodies
    /// project the RawVec struct's fields off the abstract
    /// aggregate and fail, so the entry points are modelled here.
    VecShrink,
    /// `Vec::set_len(&mut vec, new_len)`: resize the element
    /// list to `new_len`, padding with uninitialized elements
    /// when growing. The collected body first reads the
    /// capacity through the `RawVec` fields for its UB
    /// precondition check, and capacity is not represented in
    /// this value model (the same reason `Vec::capacity` is
    /// the explicit `VecCapacity` error), so the entry point
    /// is modelled here. Shrinking forgets the tail elements
    /// without dropping them, exactly as the real `set_len`
    /// contract specifies.
    VecSetLen,
    /// `Vec::extend_trusted(&mut vec, iter)`: append every item
    /// the iterator yields. The collected body is the TrustedLen
    /// fast path (`SpecFromIter::from_iter` reaches it for every
    /// `TrustedLen` iterator): it drives the RawVec allocation
    /// machinery directly, projecting the real buffer fields off
    /// the abstract aggregate, so it cannot run in this model.
    /// The observable semantics are the appends themselves,
    /// performed here by stepping the iterator with the same
    /// machinery `Iterator::next` uses.
    VecExtendTrusted,
    /// `SpecFromIterNested::from_iter(iter) -> Vec<T>`: build a
    /// vector from every item the iterator yields. The collected
    /// body has `Vec::with_capacity` inlined into it and drives
    /// `RawVecInner` directly, which the value model does not
    /// serve; the observable semantics are the collected items in
    /// order, produced here by stepping the iterator with the
    /// same machinery `Iterator::next` uses. This is the entry
    /// every `Iterator::collect::<Vec<_>>` reaches (the
    /// `SpecFromIter` instance delegates to it).
    VecFromIterNested,
    /// `slice::to_vec` / `slice::to_vec_in`: a vector holding a
    /// clone of every element of the slice, in order. The
    /// collected bodies (the `ConvertVec` fast path) drive the
    /// RawVec allocation machinery directly -- with capacity,
    /// `copy_nonoverlapping`, `set_len` -- and assemble the
    /// vector in its native struct layout, which is not the
    /// value model's vector representation (the element
    /// aggregate every `Vec` consumer here reads), so they are
    /// modelled here like the other RawVec-driving entry
    /// points. The elements are cloned with the model's value
    /// clone (`clone_owned_value`), the same treatment
    /// `Clone::clone` receives.
    SliceToVec,
    /// `Vec::capacity(&vec)`: not modelled. Capacity has no
    /// representation in the value model, and returning the length
    /// would be observably wrong, so the call fails with an
    /// explicit error naming the function (rather than projecting
    /// the abstract value as if it had the std layout).
    VecCapacity,
    /// `core::slice::<impl [T]>::iter(&slice)`: a slice iterator state
    /// (element snapshot + next index), the shape `Iterator::next`
    /// consumes for `std::slice::Iter`.
    SliceIter,
    /// `core::slice::<impl [T]>::windows(size) ->
    /// Windows<'_, T>`: the sliding-window iterator's state
    /// in this model is [slice, window size, next index];
    /// `Iterator::next` for it is modelled in
    /// `iterator_next_state`. (The native struct stores the
    /// shrinking tail slice instead; no converted body reads
    /// its fields -- `next` is the only operation the string
    /// search machinery needs.)
    SliceWindows,
    /// `str::contains(&str) -> bool` (and `starts_with` /
    /// `ends_with`): the pattern search. The native
    /// `TwoWaySearcher` execution produces wrong answers in
    /// this interpreter for multi-byte needles (it reports
    /// no match even for a needle at position 0), so the
    /// search is computed on the resolved strings directly,
    /// with the host's own `str` semantics -- the same
    /// semantics the guest's `str` operations denote.
    StrContains,
    /// See `StrContains`.
    StrStartsWith,
    /// See `StrContains`.
    StrEndsWith,
    /// `std::iter::empty()`: an empty iterator.
    IterEmpty,
    /// `Index::index(&container, idx) -> &elem`.
    IndexIndex,
    /// `IndexMut::index_mut(&mut container, idx) -> &mut elem`
    /// (containers whose elements are addressable places only).
    IndexMut,
    /// `std::ops::DerefMut::deref_mut(&mut value)` (modelled types).
    DerefMut,
    /// `Cell::new(value)`: a cell is the natural two-level
    /// aggregate `Cell { UnsafeCell { value } }` - the shape
    /// collected bodies and decoded constants produce.
    CellNew,
    /// `Cell::get(&cell)`.
    CellGet,
    /// `Cell::set(&cell, value)`.
    CellSet,
    /// `Cell::replace(&cell, value) -> old value`.
    CellReplace,
    /// `Cell::into_inner(cell) -> value`.
    CellIntoInner,
    /// `RefCell::new(value)`: the natural `RefCell` aggregate
    /// (borrow-flag `Cell` + `UnsafeCell` value).
    RefCellNew,
    /// `RefCell::borrow(&cell) -> Ref<T>`.
    RefCellBorrow,
    /// `RefCell::borrow_mut(&cell) -> RefMut<T>`.
    RefCellBorrowMut,
    /// `RefCell::try_borrow(&cell) -> Result<Ref<T>, BorrowError>`.
    RefCellTryBorrow,
    /// `RefCell::try_borrow_mut(&cell) -> Result<RefMut<T>, BorrowMutError>`.
    RefCellTryBorrowMut,
    /// `RefCell::get_mut(&mut cell) -> &mut T`.
    RefCellGetMut,
    /// `RefCell::replace(&cell, value) -> old value`.
    RefCellReplace,
    /// `RefCell::as_ptr(&cell) -> *mut T`.
    RefCellAsPtr,
    /// `Option::unwrap(option)`.
    OptionUnwrap,
    /// `Result::unwrap(result)`.
    ResultUnwrap,
    /// `Option::is_some(&option)`.
    OptionIsSome,
    /// `Option::is_none(&option)`.
    OptionIsNone,
    /// `Result::is_ok(&result)`.
    ResultIsOk,
    /// `Result::is_err(&result)`.
    ResultIsErr,
    /// `Option::take(&mut option) -> old value`.
    OptionTake,
    /// `std::mem::drop(value)`: the value is consumed. When
    /// conversion could account for the dropped type's drop glue,
    /// the call carries it (`CallBuiltin::drop_glue`) and the
    /// interpreter runs it before this dispatch is ever reached;
    /// this arm is the glue-free remainder (scalars, constants),
    /// where dropping has no guest-observable effect.
    MemDrop,
    /// `std::mem::swap(&mut a, &mut b)`.
    MemSwap,
    /// `std::ptr::swap_nonoverlapping(x, y, count)` and its
    /// intrinsic form `typed_swap_nonoverlapping`: the two
    /// ranges' elements exchanged pairwise. The collected body
    /// swaps large values byte by byte through chunk buffers,
    /// which the value model cannot follow (a byte-displaced
    /// read of a structured value is not a value); the swap
    /// itself is a whole-value operation, so it is served
    /// directly over the elements the pointers denote.
    SwapNonoverlapping,
    /// `std::mem::size_of::<T>()` (types with a known layout only).
    MemSizeOf,
    /// `std::mem::align_of::<T>()` (types with a known layout only).
    MemAlignOf,
    /// `std::mem::size_of_val(&value)`.
    MemSizeOfVal,
    /// `std::mem::align_of_val(&value)`.
    MemAlignOfVal,
    /// A `*_fast` float intrinsic (`fadd_fast`, `fsub_fast`,
    /// `fmul_fast`, `fdiv_fast`, `frem_fast`): the ordinary
    /// floating-point operation. The `fast` contract only
    /// licenses the compiler to reassociate operations and to
    /// assume the guest has no NaNs or infinities; neither
    /// changes the value of a single operation computed here.
    FastFloat(FastFloatOp),
    /// `std::intrinsics::float_to_int_unchecked(value)`: a
    /// truncating float-to-integer conversion. Out-of-range and
    /// NaN inputs are undefined behaviour in the guest; the model
    /// produces the saturating result the `as` cast defines,
    /// which agrees on every input where the intrinsic is
    /// defined.
    FloatToIntUnchecked,
    /// `std::mem::needs_drop::<T>()` (types with a known answer only).
    MemNeedsDrop,
    /// `ManuallyDrop::new(value)`: an ADT aggregate of one field.
    ManuallyDropNew,
    /// `ToOwned::to_owned(&value)` for the modelled owned types.
    ToOwned,
    /// `std::convert::Into::into(value)`: shares the `From::from`
    /// modelling (the blanket `Into` impl is `From` by definition).
    IntoInto,
    /// A primitive `f32` / `f64` inherent method (`is_nan`, `sqrt`,
    /// `powi`, ...). The float width is part of the callee name
    /// (`core::f64::<impl f64>::sqrt`), so it is carried here and the
    /// operation runs at that width.
    FloatMethod { f32: bool, op: FloatOp },
    /// A primitive `f128` inherent method. `f128` values are raw
    /// bit patterns (`Value::Float128`) and the operations run in
    /// the exact software binary128 module ([`crate::f128`]).
    Float128Method { op: FloatOp },
    /// A primitive `f16` inherent method. `f16` values ride in
    /// `Value::Float` as their exact `f64` widening; the operation
    /// computes in `f64` and rounds back to `f16`, the same model
    /// the interpreter's `f16` arithmetic uses.
    Float16Method { op: FloatOp },
    /// `std::intrinsics::black_box(value)`: the identity function.
    /// The value passes through unchanged; the hint exists only to
    /// block optimiser assumptions the interpreter does not make.
    BlackBox,
    /// `Rc::new(value)`: a fresh reference-counted allocation
    /// in the interpreter's allocation store (an `RcInner`
    /// aggregate holding the strong/weak counts and the value);
    /// the result is the handle naming it (see
    /// `Interpreter::rc_new_handle`).
    RcNew,
    /// `Arc::new(value)`: the `Rc` shape, with an `ArcInner`.
    ArcNew,
    /// `Rc::from_raw(ptr)`: rebuild the strong handle naming
    /// the allocation the raw data pointer denotes (the
    /// reverse of the data-field pointer `into_raw` produces;
    /// see the evaluation arm). No count moves: `into_raw`
    /// forgot a live handle, and `from_raw` reconstitutes it.
    RcFromRaw,
    /// `Arc::from_raw(ptr)`: the `Rc` shape, with an `ArcInner`.
    ArcFromRaw,
    /// `Weak::from_raw(ptr)` (either the `Rc` or the `Arc`
    /// weak handle): the weak handle naming the same
    /// allocation, with no count movement, as for the strong
    /// reconstruction.
    WeakFromRaw,
    /// An integer bit intrinsic (`ctpop`, `ctlz`, `cttz`,
    /// `bitreverse`, `bswap`): a unary operation on the operand
    /// type's bit pattern, at the type's own width.
    IntBitOp(BitOp),
    /// `std::intrinsics::saturating_add` / `saturating_sub`:
    /// integer arithmetic clamped to the operand type's range.
    SaturatingArith(SatOp),
    /// `std::intrinsics::atomic_load(ptr)`: in this
    /// single-threaded interpreter an atomic load is a plain read
    /// through the pointer; the ordering only constrains
    /// concurrent access, which cannot occur.
    AtomicLoad,
    /// `std::intrinsics::atomic_store(ptr, value)`: a plain
    /// write through the pointer, as for `atomic_load`.
    AtomicStore,
    /// `std::intrinsics::atomic_cxchg(ptr, old, new) ->
    /// (previous, success)`: store `new` iff the pointee equals
    /// `old`. `atomic_cxchgweak` is the same builtin: its only
    /// difference is that the exchange may fail spuriously, which
    /// a single-threaded interpreter has no reason to simulate.
    AtomicCxchg,
    /// `std::intrinsics::atomic_xchg(ptr, new) -> previous`: an
    /// unconditional exchange through the pointer.
    AtomicXchg,
    /// `std::intrinsics::atomic_xadd(ptr, value) -> previous`:
    /// a wrapping add through the pointer, in this
    /// single-threaded interpreter a read, an add and a write.
    AtomicXadd,
    /// `std::intrinsics::atomic_xsub(ptr, value) -> previous`:
    /// a wrapping subtract through the pointer, as for
    /// `AtomicXadd`.
    AtomicXsub,
    /// The atomic fetch family other than add/sub
    /// (`atomic_and`, `atomic_or`, `atomic_xor`, `atomic_nand`,
    /// `atomic_max`, `atomic_min`, `atomic_umax`, `atomic_umin`):
    /// a read-modify-write through the pointer that returns the
    /// previous value. The interpreter is single-threaded, so the
    /// update is indivisible by construction; the arithmetic
    /// wraps at the operand type's width, as the corresponding
    /// hardware operations do. (`atomic_xadd` / `atomic_xsub`
    /// keep their own variants above.)
    AtomicFetchOp(FetchOp),
    /// `std::intrinsics::atomic_fence()`: orders other threads'
    /// accesses; there are none, so it is a no-op.
    AtomicFence,
    /// `std::intrinsics::assert_inhabited` / `assert_zero_valid` /
    /// `assert_mem_uninitialized_valid`: validity assertions
    /// about the generic type — that it is inhabited, that its
    /// all-zero bit pattern is a valid value, that it may be left
    /// uninitialized. Code generation keeps each as a no-op for
    /// exactly the types that satisfy the property, and a
    /// violation is undefined behaviour in the guest, so a
    /// converted program that reaches one has nothing observable
    /// left for the interpreter to check.
    AssertValidity,
    /// `std::intrinsics::write_bytes(ptr, val, count)` and
    /// `volatile_set_memory`: fill `count` elements through the
    /// pointer with the value whose bytes are all `val`, read at
    /// the element type's own width. A volatile fill is a plain
    /// fill here: volatility only constrains concurrent or
    /// signal-observable access, which cannot occur.
    WriteBytes,
    /// `std::intrinsics::arith_offset(ptr, count)`: wrapping
    /// pointer arithmetic — what `wrapping_offset` / `wrapping_add`
    /// / `wrapping_sub` lower to — and `std::intrinsics::offset`,
    /// the checked form `BinaryOp::Offset` computes. Both evaluate
    /// through the interpreter's pointer-offset machinery and
    /// differ only in the null-base case (see
    /// `Interpreter::ptr_offset_value`).
    ArithOffset { wrapping: bool },
    /// `std::intrinsics::ptr_offset_from(x, y)` /
    /// `ptr_offset_from_unsigned(x, y)`: the number of elements
    /// between two pointers into the same allocation. The
    /// unsigned form requires `x >= y` in the guest (a negative
    /// distance is undefined behaviour), so the model keeps it a
    /// named unsupported there.
    PtrOffsetFrom { unsigned: bool },
    /// `std::mem::zeroed()`: the all-zero-bits value of a scalar
    /// type, or of a tuple/array of scalar types.
    MemZeroed,
    /// `std::ptr::null()` / `std::ptr::null_mut()`: the null
    /// pointer, a dedicated `RefTarget::Null` denoting no place
    /// and no value.
    PtrNull,
    /// `std::ptr::copy_nonoverlapping(src, dst, count)` and
    /// `std::ptr::copy(src, dst, count)`: the element copy the
    /// `copy_nonoverlapping` intrinsic performs, arriving as an
    /// ordinary call to the `std::ptr` wrapper (a direct
    /// `core::intrinsics::copy_nonoverlapping` call arrives as a
    /// `StatementKind::Intrinsic` instead; both share one
    /// implementation, whose source snapshot makes it a correct
    /// memmove for the overlapping `copy` as well).
    PtrCopy,
    /// `std::intrinsics::compare_bytes(x, y, count) -> c_int`:
    /// the memcmp of two byte ranges (the inlined slice / `str`
    /// ordering core).
    CompareBytes,
    /// One `std::intrinsics::simd::*` operation: lane-wise
    /// arithmetic, comparison, shuffling, casting, gathering,
    /// scattering and reduction over the vector values the
    /// interpreter models as lane tuples (possibly inside the
    /// one-field wrapper a SIMD type is).
    SimdOp(SimdOp),
    /// `std::path::Path::parent` / `file_name` / `file_stem` /
    /// `extension`: the path-parsing accessors. These are
    /// non-inline library methods, so no MIR body is available
    /// for them; the parsing is a pure function of the path's
    /// byte string and is answered with the host's own
    /// `std::path` (the same rules — the model's paths are unix
    /// paths, and a path value is its byte string, see
    /// `path_bytes_of`).
    PathPart(PathPart),
    /// `std::path::Path::components(&self) -> Components`: a
    /// non-inline library method, so no MIR body is available
    /// for it. The iterator state the model builds is the
    /// path's byte string plus the numbers of components
    /// consumed from the front and from the back; the
    /// `Iterator::next` / `DoubleEndedIterator::next_back`
    /// steps for `Components` replay the host's own
    /// `std::path` parsing from that state (see
    /// `components_step`).
    PathComponents,
    /// `std::path::Path::to_path_buf(&self) -> PathBuf`: a
    /// non-inline library method, so no MIR body is available
    /// for it; the result is the path's own byte string in a
    /// fresh buffer (the shape `pathbuf_value` builds).
    PathToPathBuf,
    /// `std::path::PathBuf::_push(&mut self, path: &Path)`:
    /// the mutation `PathBuf::push` forwards to; no MIR body
    /// is available for it. Answered with the host's
    /// `std::path::PathBuf` over the model's byte string (the
    /// shape `pathbuf_value` builds).
    PathBufPush,
    /// `std::path::PathBuf::pop(&mut self) -> bool`: no MIR
    /// body is available; answered with the host's
    /// `std::path::PathBuf` over the model's byte string, as
    /// for `_push`.
    PathBufPop,
    /// `std::path::PathBuf::_set_extension(&mut self,
    /// extension: &OsStr) -> bool`: the mutation
    /// `PathBuf::set_extension` forwards to; no MIR body is
    /// available for it. Answered with the host's
    /// `std::path::PathBuf` over the model's byte string, as
    /// for `_push`.
    PathBufSetExtension,
    /// `std::path::Path::_with_extension(&self, extension:
    /// &OsStr) -> PathBuf`: the leaf `Path::with_extension`
    /// forwards to; no MIR body is available for it.
    /// Answered with the host's `std::path::PathBuf` over
    /// the model's byte string.
    PathWithExtension,
    /// `std::path::PathBuf::new() -> PathBuf` and
    /// `PathBuf::with_capacity(n)`: the empty buffer in the
    /// model's shape (`pathbuf_value` over no bytes; a
    /// capacity is unobservable in the model). Modelling the
    /// constructors keeps the values in the model's shape —
    /// the converted library bodies would build the real
    /// `Vec` value (a raw pointer / capacity / length triple)
    /// whose bytes the path readers do not take.
    PathBufNew,
    /// `std::str::from_utf8(&[u8]) -> Result<&str, Utf8Error>`:
    /// a non-inline library method, so no MIR body is available;
    /// the validation is a pure function of the bytes, answered
    /// with the host's own `std::str::from_utf8` (the same
    /// rules). The error value is std's shape: the
    /// `Utf8Error` struct's `valid_up_to` and `error_len`
    /// fields.
    FromUtf8,
    /// The C library `strlen`, reached as
    /// `core::ffi::c_str::strlen::runtime::strlen`: the
    /// runtime branch of core's `const_eval_select` C-string
    /// length helper is an `extern "C"` declaration, so no MIR
    /// body exists for it. The length is a pure function of
    /// the bytes behind the pointer, scanned through the
    /// pointer model to the first NUL.
    LibcStrlen,
    /// `std::ffi::CStr::from_bytes_with_nul(&[u8]) ->
    /// Result<&CStr, FromBytesWithNulError>`: a non-inline
    /// library method, so no MIR body is available; the
    /// validation is a pure function of the bytes (the first
    /// NUL must be the slice's last byte). The `Ok` value is
    /// the argument itself, exactly what the converted
    /// `from_bytes_with_nul_unchecked` cast produces.
    CStrFromBytesWithNul,
    /// `std::ffi::CStr::from_bytes_until_nul(&[u8]) ->
    /// Result<&CStr, FromBytesUntilNulError>`: a non-inline
    /// library method, so no MIR body is available; the `Ok`
    /// value is the argument fat reference shortened to end
    /// at (and include) its first NUL byte.
    CStrFromBytesUntilNul,
    /// `std::ffi::CStr::to_str(&self) -> Result<&str,
    /// Utf8Error>`: a non-inline library method, so no MIR
    /// body is available; it is `str::from_utf8` over the
    /// C string's bytes (without the terminator), answered
    /// with the host's own validation, like
    /// [`Builtin::StrFromUtf8`].
    CStrToStr,
    /// `CStr::to_string_lossy(&self) -> Cow<'_, str>`: a
    /// non-inline library method, so no MIR body is
    /// available; the lossy conversion of the C string's
    /// bytes (without the terminator), the same model as
    /// [`Builtin::StringFromUtf8Lossy`] over those bytes.
    CStrToStringLossy,
    /// `CString::new`'s specialization dispatch: the
    /// specialized instances for `&str` / `&[u8]` have no
    /// MIR body, so the whole entry point is modelled:
    /// extract the argument's bytes (the `Into<Vec<u8>>`
    /// the default instance performs), reject an interior
    /// NUL with `NulError(position, bytes)`, else build the
    /// owned C string.
    CStringSpecNewImpl,
    /// `CString::_from_vec_unchecked` /
    /// `CString::from_vec_unchecked` (neither has a MIR
    /// body): append the trailing NUL and box the bytes.
    CStringFromVecUnchecked,
    /// `CString::from_raw` (no MIR body): the length is
    /// recomputed with the runtime `strlen`, and the
    /// pointer is reconstituted as the owned boxed slice.
    CStringFromRaw,
    /// `CString::into_string` (no MIR body): the UTF-8
    /// validation of the owned bytes; on failure the
    /// error carries the string back (`IntoStringError {
    /// inner, error }`).
    CStringIntoString,
    /// `alloc::alloc::__rust_alloc(size, align) -> *mut u8` and
    /// `__rust_alloc_zeroed`: a fresh block in the allocation
    /// store, named by a new allocation id. The uninitialised
    /// form's storage is `Value::Uninit` (the buffer-fill write
    /// paths materialise its elements, exactly as for a
    /// `Box::new_uninit` pointee); the zeroed form's storage is
    /// zero bytes. A request too large to materialise answers
    /// the null pointer, which is the shim's own failure signal
    /// (the caller then runs the allocation-error path).
    RustAlloc { zeroed: bool },
    /// `alloc::alloc::__rust_dealloc(ptr, size, align)`: removes
    /// the pointer's block from the allocation store. A pointer
    /// that names no live block (a fabricated or dangling one)
    /// frees nothing.
    RustDealloc,
    /// `alloc::alloc::__rust_realloc(ptr, old_size, align,
    /// new_size) -> *mut u8`: a fresh block holding the old
    /// block's contents; the old block is freed.
    RustRealloc,
    /// `libc::unix::malloc(size) -> *mut u8`: the bottom of the
    /// system allocator's call chain — a fresh uninitialised
    /// block of `size` bytes in the allocation store, the same
    /// storage the `__rust_alloc` model hands out.
    LibcMalloc,
    /// `libc::unix::calloc(nmemb, size) -> *mut u8`: a fresh
    /// zero-filled block of `nmemb * size` bytes.
    LibcCalloc,
    /// `libc::unix::free(ptr)`: removes the pointer's block
    /// from the allocation store.
    LibcFree,
    /// `libc::unix::realloc(ptr, new_size) -> *mut u8`: a fresh
    /// block holding the old block's contents; the old block is
    /// freed.
    LibcRealloc,
    /// `libc::unix::posix_memalign(memptr, alignment, size)
    /// -> c_int`: allocates a block, stores its pointer through
    /// `memptr`, and answers 0. The model's blocks carry no
    /// machine address whose alignment could be observed, so
    /// the alignment argument only documents the request.
    LibcPosixMemalign,
    /// `core::num::imp::overflow_panic::pow(base, exp)`: the
    /// exponentiation behind the primitive `pow` methods,
    /// panicking on overflow exactly like the checked
    /// multiplications it is built from when overflow checks
    /// are on, and wrapping when they are off.
    IntPow,
    /// `str::FromStr::from_str` for the primitive `Self` types
    /// core implements (integers, floats, `bool`, `char`): the
    /// host's parser is the same dec2flt / integer algorithm
    /// core ships, so results and error kinds match exactly.
    /// Collected impl bodies for other types win over this
    /// model through the usual impl-resolution path.
    FromStrParse,
    /// `std::intrinsics::volatile_load(ptr)`: a plain load;
    /// volatility only orders accesses, which a sequential
    /// interpreter already does.
    VolatileLoad,
    /// `std::intrinsics::volatile_store(ptr, val)`: a plain
    /// store, as with `VolatileLoad`.
    VolatileStore,
    /// `std::intrinsics::volatile_copy_memory(dst, src, count)`
    /// and its non-overlapping twin: note the destination
    /// comes first, unlike the `copy` intrinsics.
    VolatileCopy,
    /// `std::intrinsics::volatile_set_memory(dst, val, count)`:
    /// store `val` at each of `count` consecutive elements.
    VolatileSet,
    /// A `rust_test_helpers` identity function
    /// (`rust_dbg_extern_identity_*`, `rust_int8_to_int32`, the
    /// i128 helper `identity`): return the single argument
    /// unchanged. The value model stores integers sign-extended
    /// and floats widened, so the C function's own widening is
    /// already the stored shape. Served only for callees with
    /// no converted body (see [`foreign_helper_lookup`]).
    FfiIdentity,
    /// `rust_dbg_extern_return_Two{U8s,U16s,U32s,U64s}`: the
    /// fixed two-field struct `{10, 20}`.
    FfiReturnTwo,
    /// `rust_dbg_extern_return_FiveU16s`: the fixed struct
    /// `{10, 20, 30, 40, 50}`.
    FfiReturnFive,
    /// `rust_dbg_call(cb, data) -> cb(data)`: invoke the guest
    /// callback with the data argument and return its result.
    FfiDbgCall,
    /// `rust_dbg_extern_empty_struct(v1, e, v2)`: the C body
    /// asserts each field of `v1` equals the matching field of
    /// `v2` plus one; modelled as the same checks (a failed
    /// check is the C `assert` failure: a guest abort).
    FfiEmptyStruct,
    /// `rust_get_test_int() -> 1`.
    FfiGetTestInt,
    /// `rust_get_null_ptr() -> NULL`.
    FfiGetNullPtr,
    /// `rust_dbg_abi_1(quad) -> {c + 1, d - 1, a + 1, b - 1}`
    /// (u64 wrapping).
    FfiDbgAbi1,
    /// `rust_dbg_abi_2(floats) -> {c + 1.0, 0xff, a - 1.0}`.
    FfiDbgAbi2,
    /// `rust_dbg_abi_3(char_char_double) -> {a + 1, b - 1,
    /// c + 1.0}` (u8 wrapping).
    FfiDbgAbi3,
    /// `rust_dbg_abi_4(char_char_float) -> {a + 1, b - 1,
    /// c + 1.0f}` (u8 wrapping, f32 arithmetic).
    FfiDbgAbi4,
    /// A helper returning one field of one argument: `get_x` /
    /// `get_y` / `get_z` (field 0/1/2 of argument 0),
    /// `get_c_many_params` (field 2 of argument 4),
    /// `get_c_exhaust_sysv64_ints` (field 2 of argument 7).
    FfiArgField { arg: usize, field: usize },
    /// `rust_interesting_average(n, ...)`: the variadic tail
    /// is `n` pairs of `(i64, f64)`; the result is the pair
    /// sum divided by `n` (NaN for `n == 0`, as in C).
    FfiInterestingAverage,
    /// `rust_valist_interesting_average(n, ap)`: the same pair
    /// sum as [`Builtin::FfiInterestingAverage`], read from the
    /// va list `ap` at its current cursor, which advances past
    /// the `n` pairs read.
    FfiValistInterestingAverage,
    /// `rust_va_list_next_i32(ap)`: `va_arg(*ap, int32_t)` —
    /// the next `i32` of the va list behind the pointer, whose
    /// cursor advances.
    FfiVaListNextI32,
    /// `std::intrinsics::va_arg(ap) -> T`: the next argument of
    /// the va list `ap` points to, whose cursor advances. The
    /// argument was stored in the promoted shape the caller
    /// passed it in (see [`crate::interp::VaListState`]), which
    /// is the shape `T` reads back.
    VaArg,
    /// The i128 helper `square(x) -> x * x` (wrapping).
    FfiSquare128,
    /// The i128 helper `sub(a, b) -> a - b` (wrapping).
    FfiSub128,
    /// `increment_all_parts(LARGE_INTEGER)`: each 32-bit part
    /// is incremented twice (once per struct view of the
    /// union) and the whole 64-bit value once more.
    FfiIncrementAllParts,
    /// `issue_97463_leak_uninit_data(a, b, c) -> (b & 0xFFFF)
    /// as u16`.
    FfiLeakUninitData,
    /// `rust_dbg_new_some_u64(x)` /
    /// `rust_dbg_new_some_u64u64(a, b)`: the `Some` variant
    /// (index 1) of the `#[repr(C, u8)]` tagged-option enums,
    /// holding the arguments as its payload fields.
    FfiNewSome,
    /// `rust_dbg_new_none_u64()` /
    /// `rust_dbg_new_none_u64u64()`: the `None` variant
    /// (index 0, no payload).
    FfiNewNone,
    /// `rust_dbg_unpack_option_u64(o, into)` /
    /// `rust_dbg_unpack_option_u64u64(o, a, b)`: for a `Some`
    /// value, write the payload fields through the pointer
    /// arguments and return 1; for `None`, return 0. The C
    /// body asserts the pointers are non-null.
    FfiUnpackOption,
    /// `proc_macro::is_available()`: false for code running
    /// as an ordinary program; the true answer exists only
    /// inside a proc-macro expansion, which is compile-time.
    ProcMacroIsAvailable,

}

/// The `std::sys::fs` leaf operations the interpreter performs
/// on the host filesystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsSysOp {
    Metadata,
    SymlinkMetadata,
    Canonicalize,
    Copy,
    HardLink,
    ReadDir,
    ReadLink,
    RemoveDir,
    RemoveDirAll,
    RemoveFile,
    Rename,
    SetPermissions,
    Symlink,
}

/// A relational operator selected by a `PartialOrd` method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelOp {
    Lt,
    Le,
    Gt,
    Ge,
}

/// Which end(s) of a string a `trim*` method strips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimSide {
    Both,
    Start,
    End,
}

/// One primitive float method operation, selected by the method name
/// of an `f32` / `f64` inherent impl call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatOp {
    IsNan,
    IsInfinite,
    IsFinite,
    IsSignPositive,
    IsSignNegative,
    Abs,
    Sqrt,
    Floor,
    Ceil,
    Round,
    RoundTiesEven,
    Trunc,
    Fract,
    Recip,
    Signum,
    ToDegrees,
    ToRadians,
    Powi,
    Powf,
    Min,
    Max,
    ToBits,
    FromBits,
}

/// One integer bit intrinsic operation, selected by the
/// intrinsic's name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitOp {
    CtPop,
    Ctlz,
    Cttz,
    BitReverse,
    Bswap,
}

/// One atomic read-modify-write operation, selected by the
/// intrinsic's name. `Max` / `Min` compare in the signed domain
/// and `UMax` / `UMin` in the unsigned domain, matching the
/// intrinsic names (the operand type agrees in practice).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Nand,
    Max,
    Min,
    UMax,
    UMin,
}

/// One `*_fast` float operation, selected by the intrinsic's
/// name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastFloatOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

/// Which part of a path a `Builtin::PathPart` extracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathPart {
    Parent,
    FileName,
    FileStem,
    Extension,
}

/// A saturating arithmetic operation, selected by the
/// intrinsic's name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SatOp {
    Add,
    Sub,
}

/// One SIMD intrinsic operation, selected by the intrinsic's
/// name. The variants follow the argument shapes of
/// `core::intrinsics::simd` (1.100.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Neg,
    Fabs,
    Ceil,
    Floor,
    Trunc,
    Round,
    RoundTiesEven,
    Fsqrt,
    Fexp,
    Fexp2,
    Flog,
    Flog2,
    Flog10,
    Fsin,
    Fcos,
    Bswap,
    BitReverse,
    Ctlz,
    Ctpop,
    Cttz,
    SaturatingAdd,
    SaturatingSub,
    CarrylessMul,
    FunnelShl,
    FunnelShr,
    MaximumNumber,
    MinimumNumber,
    Splat,
    Extract,
    Insert,
    Shuffle,
    Cast,
    Bitmask,
    Select,
    SelectBitmask,
    Gather,
    Scatter,
    MaskedLoad,
    MaskedStore,
    ArithOffset,
    ReduceAddOrdered,
    ReduceAddUnordered,
    ReduceMulOrdered,
    ReduceMulUnordered,
    ReduceMax,
    ReduceMin,
    ReduceAnd,
    ReduceOr,
    ReduceXor,
    ReduceAll,
    ReduceAny,
    Fma,
    CastPtr,
    ExposeProvenance,
    WithExposedProvenance,
}

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

/// A successful `core::fmt::Result` value (`Ok(())`).
fn fmt_result_ok() -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Unit],
    }
}

/// Look a canonical callee name up in the registry, spelled
/// exactly as given; `None` when the interpreter has no model
/// for it.
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
        "std::process::exit" => Builtin::ProcessExit,
        "std::io::stdout" => Builtin::Stdout,
        "std::io::Write::write_fmt" => Builtin::WriteFmt,
        "std::fmt::write" | "core::fmt::write" => Builtin::FmtWrite,
        "std::fmt::format" => Builtin::Format,
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
        "std::iter::Iterator::count" => Builtin::IteratorCount,
        "std::iter::Iterator::fold" => Builtin::IteratorFold,
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
        "core::str::<impl str>::len" => Builtin::StrLen,
        "core::str::<impl str>::is_empty" => Builtin::StrIsEmpty,
        "core::str::<impl str>::as_bytes" => Builtin::StrAsBytes,
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
            && canonical.ends_with(">::extend_trusted") =>
        {
            Builtin::VecExtendTrusted
        }
        "std::vec::spec_from_iter_nested::SpecFromIterNested::from_iter" => {
            Builtin::VecFromIterNested
        }
        _ if canonical.starts_with("std::vec::Vec") && canonical.ends_with(">::capacity") => {
            Builtin::VecCapacity
        }
        _ if canonical.starts_with("std::vec::Vec") && canonical.ends_with(">::remove::assert_failed") => Builtin::VecRemoveAssertFailed,
        _ if canonical.starts_with("core::slice::<impl") && canonical.ends_with(">::iter") => {
            Builtin::SliceIter
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
        _ => return float_method(canonical),
    })
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

/// Resolve a primitive float method callee
/// (`core::f64::<impl f64>::sqrt`, `std::f32::<impl f32>::floor`,
/// ...) to its builtin. The methods live partly in `core` and partly
/// in `std`; both prefixes name the same inherent impls, so either is
/// accepted for every modelled method. A float method this registry
/// does not model (e.g. `sin_cos`, which returns a tuple) stays an
/// explicit unsupported call.
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

/// Execute one primitive `f128` method on raw bit patterns.
fn exec_float128_method(
    name: &str,
    op: FloatOp,
    vals: &[Value],
) -> InterpResult<Value> {
    use crate::f128;
    let arity = |n: usize| -> InterpResult<()> {
        if vals.len() == n {
            Ok(())
        } else {
            Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                format!("{n} argument(s)"),
                format!("{} argument(s)", vals.len()),
            ))
        }
    };
    let as_bits = |v: &Value| -> InterpResult<u128> {
        match v {
            Value::Float128(bits) => Ok(*bits),
            other => Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "f128",
                other.kind_name(),
            )),
        }
    };
    use FloatOp as O;
    if op == O::FromBits {
        arity(1)?;
        let bits = match vals[0] {
            Value::Uint(v) => v,
            Value::Int(v) => v as u128,
            ref other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "integer",
                    other.kind_name(),
                ));
            }
        };
        return Ok(Value::Float128(bits));
    }
    arity(if matches!(op, O::Powi | O::Min | O::Max) {
        2
    } else {
        1
    })?;
    let x = as_bits(&vals[0])?;
    Ok(match op {
        O::IsNan => Value::Bool(f128::is_nan(x)),
        O::IsInfinite => Value::Bool(f128::is_infinite(x)),
        O::IsFinite => Value::Bool(f128::is_finite(x)),
        O::IsSignPositive => Value::Bool(f128::is_sign_positive(x)),
        O::IsSignNegative => Value::Bool(f128::is_sign_negative(x)),
        O::Abs => Value::Float128(f128::abs(x)),
        O::Sqrt => Value::Float128(f128::sqrt(x)),
        O::Floor => Value::Float128(f128::floor(x)),
        O::Ceil => Value::Float128(f128::ceil(x)),
        O::Round => Value::Float128(f128::round_half_away(x)),
        O::RoundTiesEven => Value::Float128(f128::round_ties_even(x)),
        O::Trunc => Value::Float128(f128::trunc(x)),
        O::Fract => Value::Float128(f128::fract(x)),
        O::Recip => Value::Float128(f128::recip(x)),
        O::Signum => Value::Float128(f128::signum(x)),
        O::ToBits => Value::Uint(x),
        O::Powi => {
            let n = match vals[1] {
                Value::Int(v) => v as i32,
                Value::Uint(v) => v as i32,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "i32",
                        other.kind_name(),
                    ));
                }
            };
            Value::Float128(f128::powi(x, n))
        }
        O::Min => Value::Float128(f128::min(x, as_bits(&vals[1])?)),
        O::Max => Value::Float128(f128::max(x, as_bits(&vals[1])?)),
        O::Powf | O::ToDegrees | O::ToRadians => {
            return Err(unsupported(format!(
                "f128 method {name} is not runtime-supported"
            )));
        }
        O::FromBits => unreachable!("handled above"),
    })
}

/// Execute one primitive `f16` method. The receiver rides as its
/// exact `f64` widening; the operation computes in `f64` and the
/// result rounds back to `f16` (exact for the operations modelled
/// here, by the same double-rounding argument as `f32`).
fn exec_float16_method(
    name: &str,
    op: FloatOp,
    vals: &[Value],
) -> InterpResult<Value> {
    use crate::f128::{f16_bits_to_f64, f64_to_f16_bits};
    let arity = |n: usize| -> InterpResult<()> {
        if vals.len() == n {
            Ok(())
        } else {
            Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                format!("{n} argument(s)"),
                format!("{} argument(s)", vals.len()),
            ))
        }
    };
    let as_float = |v: &Value| -> InterpResult<f64> {
        match v {
            Value::Float(x) => Ok(*x),
            other => Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "f16",
                other.kind_name(),
            )),
        }
    };
    use FloatOp as O;
    if op == O::FromBits {
        arity(1)?;
        let bits = match vals[0] {
            Value::Uint(v) => v,
            Value::Int(v) => v as u128,
            ref other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "integer",
                    other.kind_name(),
                ));
            }
        };
        return Ok(Value::Float(f16_bits_to_f64(bits as u16)));
    }
    arity(if matches!(op, O::Powi | O::Powf | O::Min | O::Max) {
        2
    } else {
        1
    })?;
    let x = as_float(&vals[0])?;
    let round = |r: f64| Value::Float(f16_bits_to_f64(f64_to_f16_bits(r)));
    Ok(match op {
        O::IsNan => Value::Bool(x.is_nan()),
        O::IsInfinite => Value::Bool(x.is_infinite()),
        O::IsFinite => Value::Bool(x.is_finite()),
        O::IsSignPositive => Value::Bool(x.is_sign_positive()),
        O::IsSignNegative => Value::Bool(x.is_sign_negative()),
        O::Abs => round(x.abs()),
        O::Sqrt => round(x.sqrt()),
        O::Floor => round(x.floor()),
        O::Ceil => round(x.ceil()),
        O::Round => round(x.round()),
        O::RoundTiesEven => round(x.round_ties_even()),
        O::Trunc => round(x.trunc()),
        O::Fract => round(x.fract()),
        O::Recip => round(x.recip()),
        O::Signum => round(x.signum()),
        O::ToDegrees => round(x.to_degrees()),
        O::ToRadians => round(x.to_radians()),
        O::ToBits => Value::Uint(f64_to_f16_bits(x) as u128),
        O::Powi => {
            let n = match vals[1] {
                Value::Int(v) => v as i32,
                Value::Uint(v) => v as i32,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "i32",
                        other.kind_name(),
                    ));
                }
            };
            round(x.powi(n))
        }
        O::Powf => round(x.powf(as_float(&vals[1])?)),
        O::Min => round(x.min(as_float(&vals[1])?)),
        O::Max => round(x.max(as_float(&vals[1])?)),
        O::FromBits => unreachable!("handled above"),
    })
}

/// Execute one primitive float method. The operation runs at the
/// method's own width — an `f32` method computes on the `f32` value
/// and the result widens back (widening is exact) — so rounding, NaN
/// propagation and signed-zero behaviour are the native ones by
/// construction: the host `f32` / `f64` methods are the definitions
/// core and std themselves use.
fn exec_float_method(
    name: &str,
    f32_width: bool,
    op: FloatOp,
    vals: &[Value],
) -> InterpResult<Value> {
    let arity = |n: usize| -> InterpResult<()> {
        if vals.len() == n {
            Ok(())
        } else {
            Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                format!("{n} argument(s)"),
                format!("{} argument(s)", vals.len()),
            ))
        }
    };
    let as_float = |v: &Value| -> InterpResult<f64> {
        match v {
            Value::Float(x) => Ok(*x),
            other => Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "float",
                other.kind_name(),
            )),
        }
    };
    use FloatOp as O;
    // `from_bits` is an associated function: no receiver, the single
    // argument is the bit pattern.
    if op == O::FromBits {
        arity(1)?;
        let bits = match vals[0] {
            Value::Uint(v) => v,
            Value::Int(v) => v as u128,
            ref other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "integer",
                    other.kind_name(),
                ));
            }
        };
        return Ok(Value::Float(if f32_width {
            f32::from_bits(bits as u32) as f64
        } else {
            f64::from_bits(bits as u64)
        }));
    }
    arity(if matches!(op, O::Powi | O::Powf | O::Min | O::Max) {
        2
    } else {
        1
    })?;
    let x = as_float(&vals[0])?;
    macro_rules! unary {
        ($method:ident) => {
            if f32_width {
                (x as f32).$method() as f64
            } else {
                x.$method()
            }
        };
    }
    macro_rules! predicate {
        ($method:ident) => {
            if f32_width {
                (x as f32).$method()
            } else {
                x.$method()
            }
        };
    }
    Ok(match op {
        O::IsNan => Value::Bool(predicate!(is_nan)),
        O::IsInfinite => Value::Bool(predicate!(is_infinite)),
        O::IsFinite => Value::Bool(predicate!(is_finite)),
        O::IsSignPositive => Value::Bool(predicate!(is_sign_positive)),
        O::IsSignNegative => Value::Bool(predicate!(is_sign_negative)),
        O::Abs => Value::Float(unary!(abs)),
        O::Sqrt => Value::Float(unary!(sqrt)),
        O::Floor => Value::Float(unary!(floor)),
        O::Ceil => Value::Float(unary!(ceil)),
        O::Round => Value::Float(unary!(round)),
        O::RoundTiesEven => Value::Float(unary!(round_ties_even)),
        O::Trunc => Value::Float(unary!(trunc)),
        O::Fract => Value::Float(unary!(fract)),
        O::Recip => Value::Float(unary!(recip)),
        O::Signum => Value::Float(unary!(signum)),
        O::ToDegrees => Value::Float(unary!(to_degrees)),
        O::ToRadians => Value::Float(unary!(to_radians)),
        O::ToBits => Value::Uint(if f32_width {
            (x as f32).to_bits() as u128
        } else {
            x.to_bits() as u128
        }),
        O::Powi => {
            let n = match vals[1] {
                Value::Int(v) => v as i32,
                Value::Uint(v) => v as i32,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "i32",
                        other.kind_name(),
                    ));
                }
            };
            Value::Float(if f32_width {
                (x as f32).powi(n) as f64
            } else {
                x.powi(n)
            })
        }
        O::Powf => {
            let y = as_float(&vals[1])?;
            Value::Float(if f32_width {
                (x as f32).powf(y as f32) as f64
            } else {
                x.powf(y)
            })
        }
        O::Min => {
            let y = as_float(&vals[1])?;
            Value::Float(if f32_width {
                (x as f32).min(y as f32) as f64
            } else {
                x.min(y)
            })
        }
        O::Max => {
            let y = as_float(&vals[1])?;
            Value::Float(if f32_width {
                (x as f32).max(y as f32) as f64
            } else {
                x.max(y)
            })
        }
        O::FromBits => unreachable!("handled above"),
    })
}

fn unsupported(what: impl Into<String>) -> InterpError {
    InterpError::UnsupportedTerminator {
        kind: format!("runtime support: {}", what.into()),
    }
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// Escape a string for the harness's JSON output formats:
/// the quote, the backslash, and the control characters (the
/// short escapes for newline, carriage return, and tab, the
/// `\uXXXX` form for the rest), as libtest's own JSON writer
/// emits them.
fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// A panic report as it appears in a passing should-panic
/// test's captured output: the `thread '...' panicked at ...`
/// head and the message, without the backtrace note the
/// failure account adds.
/// The full panic report of a failing test's captured output:
/// the report head plus the backtrace note line, as libtest
/// prints it ahead of a failure's own account (for example the
/// `panic did not contain expected string` note).
fn panic_report_full(
    test: &TestDesc,
    thread_id: u64,
    location: Option<(String, u64, u64)>,
    message: &str,
) -> String {
    format!(
        "{}note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n",
        panic_report_head(test, thread_id, location, message)
    )
}

fn panic_report_head(
    test: &TestDesc,
    thread_id: u64,
    location: Option<(String, u64, u64)>,
    message: &str,
) -> String {
    match location {
        Some((file, line, col)) => format!(
            "\nthread '{}' ({}) panicked at {}:{}:{}:\n{}\n",
            test.name, thread_id, file, line, col, message
        ),
        None => format!(
            "\nthread '{}' ({}) panicked:\n{}\n",
            test.name, thread_id, message
        ),
    }
}

/// The `ExitCode` value the harness main returns: the byte
/// wrapped in the two single-field enums of its model shape.
fn test_exit_code(byte: u128) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Enum {
            variant: 0,
            fields: vec![Value::Uint(byte)],
        }],
    }
}

/// The libtest command line, read from the guest's arguments:
/// positional filters (a test is selected when it matches ANY
/// of them, as a substring, or by equality with `--exact`),
/// `--skip` exclusions, `--list`, `--nocapture`, the
/// ignored-test mode (`--ignored` / `--include-ignored`), and
/// the output format (`--quiet` is libtest's terse format;
/// `--format json` is the JSON Lines event stream, accepted
/// only together with `-Z unstable-options`). Flags the model
/// does not act on (thread counts, colors, log files) are
/// accepted and ignored: they cannot change a sequential
/// run's output. The terse run format is only honored for
/// `--list`.
/// How a run treats `#[ignore]`d tests (libtest's
/// `RunIgnored`): by default they are skipped and reported as
/// ignored; `--include-ignored` runs them with the rest;
/// `--ignored` runs only them, the rest counting as filtered
/// out. When both flags are given the later one wins, as in
/// libtest's own parser.
#[derive(Clone, Copy, PartialEq)]
enum RunIgnored {
    No,
    Yes,
    Only,
}

struct HarnessOpts {
    filters: Vec<String>,
    exact: bool,
    skips: Vec<String>,
    list: bool,
    nocapture: bool,
    terse: bool,
    json: bool,
    show_output: bool,
    run_ignored: RunIgnored,
}

impl HarnessOpts {
    fn parse(args: &[String]) -> HarnessOpts {
        let mut opts = HarnessOpts {
            filters: Vec::new(),
            exact: false,
            skips: Vec::new(),
            list: false,
            nocapture: false,
            terse: false,
            json: false,
            show_output: false,
            run_ignored: RunIgnored::No,
        };
        let mut i = 0;
        while i < args.len() {
            let arg = &args[i];
            match arg.as_str() {
                "--list" => opts.list = true,
                "--nocapture" => opts.nocapture = true,
                "--exact" => opts.exact = true,
                "--quiet" => opts.terse = true,
                // A `-Z` flag's value is an unstable option
                // (the gating in the harness arm judges it),
                // never a positional filter.
                "-Z" => i += 1,
                "--include-ignored" => opts.run_ignored = RunIgnored::Yes,
                "--ignored" => opts.run_ignored = RunIgnored::Only,
                "--show-output" => opts.show_output = true,
                "--format" => {
                    i += 1;
                    match args.get(i).map(String::as_str) {
                        Some("terse") => opts.terse = true,
                        Some("json") => opts.json = true,
                        _ => {}
                    }
                }
                "--skip" => {
                    i += 1;
                    if let Some(v) = args.get(i) {
                        opts.skips.push(v.clone());
                    }
                }
                _ => {
                    if let Some(v) = arg.strip_prefix("--format=") {
                        if v == "terse" {
                            opts.terse = true;
                        } else if v == "json" {
                            opts.json = true;
                        }
                    } else if let Some(v) = arg.strip_prefix("--skip=") {
                        opts.skips.push(v.to_string());
                    } else if !arg.starts_with('-') {
                        opts.filters.push(arg.clone());
                    }
                }
            }
            i += 1;
        }
        opts
    }

    /// Whether ignored tests run in this run (with the rest,
    /// or as the only tests).
    fn runs_ignored(&self) -> bool {
        self.run_ignored != RunIgnored::No
    }

    fn selected(&self, test: &TestDesc) -> bool {
        if self.run_ignored == RunIgnored::Only && !test.ignore {
            return false;
        }
        if self.skips.iter().any(|s| test.name.contains(s.as_str())) {
            return false;
        }
        if self.filters.is_empty() {
            return true;
        }
        self.filters.iter().any(|f| {
            if self.exact {
                test.name == *f
            } else {
                test.name.contains(f.as_str())
            }
        })
    }
}

/// One entry of the libtest harness's test list, read by
/// `test_main_env_args` (see `Interpreter::read_test_descs`).
struct TestDesc {
    name: String,
    ignore: bool,
    ignore_message: Option<String>,
    should_panic: ShouldPanic,
    /// The test function's definition site (the descriptor's
    /// source file and the line/column of its name), which the
    /// harness names in its did-not-panic note.
    source_file: String,
    start_line: u64,
    start_col: u64,
    end_line: u64,
    end_col: u64,
    callable: Value,
}

/// The descriptor's should-panic mode: libtest's `ShouldPanic`
/// enum, whose message-carrying variant holds the expected
/// substring a panicking test's payload must contain.
#[derive(Clone, PartialEq)]
enum ShouldPanic {
    No,
    Yes,
    YesWithMessage(String),
}

/// A point in time as the model stores it: seconds and
/// nanoseconds (0 <= nsec < 1e9) since the clock's epoch (the
/// monotonic clock's unspecified epoch for `Instant`, the Unix
/// epoch for `SystemTime`).
type TimePair = (i128, i128);

/// The host monotonic clock as a [`TimePair`]: the elapsed time
/// since this process's first reading, so the values are
/// non-negative and strictly follow the host clock's order (the
/// epoch itself is unobservable for `Instant`, as natively).
fn monotonic_now() -> TimePair {
    static BASE: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let base = BASE.get_or_init(std::time::Instant::now);
    let d = std::time::Instant::now().duration_since(*base);
    (d.as_secs() as i128, d.subsec_nanos() as i128)
}

/// The host wall clock as a [`TimePair`] (seconds since the Unix
/// epoch; negative for earlier times, as `SystemTime` allows).
fn wall_now() -> TimePair {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => (d.as_secs() as i128, d.subsec_nanos() as i128),
        Err(e) => {
            let d = e.duration();
            (-(d.as_secs() as i128), -(d.subsec_nanos() as i128))
        }
    }
}

/// The stored value of an `Instant` / `SystemTime`: the nesting
/// the constant decoder produces for `UNIX_EPOCH`
/// (`{ t: { t: { sec, { nsec } } } }`), so modelled values and
/// decoded constants have the same shape and compare field-wise.
fn instant_value(pair: TimePair) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Enum {
            variant: 0,
            fields: vec![Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Int(pair.0),
                    Value::Enum {
                        variant: 0,
                        fields: vec![Value::Int(pair.1)],
                    },
                ],
            }],
        }],
    }
}

/// Read a [`TimePair`] out of a stored time value, unwrapping
/// the single-field wrapper layers (and the nanoseconds
/// wrapper), so both modelled values and constants decoded from
/// allocations are accepted.
fn time_pair(value: &Value) -> Option<TimePair> {
    fn as_int(value: &Value) -> Option<i128> {
        match value {
            Value::Int(v) => Some(*v),
            Value::Uint(v) => Some(*v as i128),
            Value::Enum { fields, .. } | Value::Tuple(fields) if fields.len() == 1 => {
                as_int(&fields[0])
            }
            _ => None,
        }
    }
    match value {
        Value::Enum { fields, .. } | Value::Tuple(fields) => {
            if fields.len() == 1 {
                return time_pair(&fields[0]);
            }
            if fields.len() == 2 {
                return Some((as_int(&fields[0])?, as_int(&fields[1])?));
            }
            None
        }
        _ => None,
    }
}

/// Read a `Duration` value as a (secs, nanos) pair in the same
/// form as a [`TimePair`]. The fields are unsigned (`u64` secs,
/// `u32` nanos, the nanos possibly behind its wrapper struct in
/// a decoded constant), so an integer field is read as the bit
/// pattern of the unsigned value.
fn duration_pair(value: &Value) -> Option<TimePair> {
    fn as_unsigned(value: &Value) -> Option<i128> {
        match value {
            Value::Uint(v) => Some(*v as i128),
            Value::Int(v) => Some(*v as u64 as i128),
            Value::Enum { fields, .. } | Value::Tuple(fields) if fields.len() == 1 => {
                as_unsigned(&fields[0])
            }
            _ => None,
        }
    }
    match value {
        Value::Enum { fields, .. } | Value::Tuple(fields) if fields.len() == 2 => {
            Some((as_unsigned(&fields[0])?, as_unsigned(&fields[1])?))
        }
        _ => None,
    }
}

/// `base + dur` on time pairs, normalized; `None` on overflow of
/// the seconds range. For `Instant` (system = false) the result
/// must stay in the monotonic clock's representable range, which
/// this model takes as the i64 seconds range; `SystemTime` and
/// `Duration` arithmetic shares the same bound here.
fn checked_time_add(base: TimePair, dur: TimePair, _system: bool) -> Option<TimePair> {
    let mut secs = base.0.checked_add(dur.0)?;
    let mut nanos = base.1 + dur.1;
    if nanos >= 1_000_000_000 {
        nanos -= 1_000_000_000;
        secs = secs.checked_add(1)?;
    }
    if secs > i64::MAX as i128 {
        return None;
    }
    Some((secs, nanos))
}

/// `base - dur` on time pairs, normalized; `None` when the
/// result would be negative (for a duration-since computation)
/// or underflow the seconds range.
fn checked_time_sub(base: TimePair, dur: TimePair, _system: bool) -> Option<TimePair> {
    let mut secs = base.0.checked_sub(dur.0)?;
    let mut nanos = base.1 - dur.1;
    if nanos < 0 {
        nanos += 1_000_000_000;
        secs = secs.checked_sub(1)?;
    }
    if secs < 0 {
        return None;
    }
    Some((secs, nanos))
}

impl<'a> Interpreter<'a> {
    /// Execute one `CallBuiltin` terminator. `name` is canonical;
    /// `generic_args` are the callee's rendered generic type tags.
    pub(crate) fn exec_builtin(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        name: &str,
        generic_args: &[String],
        args: &[Operand],
        cmp_discriminants: Option<&[i128]>,
    ) -> InterpResult<Value> {
        // `Add` / `Sub` on a modelled time type: conversion emits
        // a builtin node only for those calls (see
        // `time_add_sub_terminator` in mir_convert). The names are
        // deliberately absent from `lookup`, so the driver's reach
        // pass still collects the other instances' bodies (the
        // generic shims, guest impls) as usual.
        if name == "std::ops::Add::add" || name == "std::ops::Sub::sub" {
            let mut vals = Vec::with_capacity(args.len());
            for arg in args {
                vals.push(self.eval_operand(frames, body, arg)?);
            }
            // The time types' `Add` / `Sub` are `#[track_caller]`,
            // so conversion appended the caller-location stamp;
            // note it for the overflow panic's location, exactly
            // as the panic builtins do with theirs.
            if vals.len() == 3 {
                let stamp = vals.pop().expect("three values checked");
                self.note_panic_stamp(stamp);
            }
            let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
            return self.time_add_sub(name, self_ty, name.ends_with("Sub::sub"), &vals);
        }
        let kind = lookup(name)
            .or_else(|| foreign_helper_lookup(name))
            .or_else(|| fmt_trait_method(name))
            .ok_or_else(|| unsupported(format!("call to {name} is not runtime-supported")))?;
        // The closure-taking consumers evaluate their callable
        // operand themselves: the generic evaluation below moves
        // the closure out of its place, destroying the environment
        // reference the calls must alias for `FnMut` semantics
        // (see `call_callable_operand_to_completion`).
        if matches!(kind, Builtin::IteratorFold | Builtin::IteratorForEach) {
            return self.exec_iterator_fold_like(frames, body, kind, generic_args, args);
        }
        let mut vals = Vec::with_capacity(args.len());
        for arg in args {
            vals.push(self.eval_operand(frames, body, arg)?);
        }
        // The panic entry points are `#[track_caller]`: the
        // driver appends the caller-location stamp as a trailing
        // argument, which the arms below do not consume; it is
        // the panic's location, recorded for the panic hook.
        if matches!(
            kind,
            Builtin::PanicStr | Builtin::PanicFmt | Builtin::PanicAny | Builtin::PanicAccessError
        ) && vals.len() == 2
        {
            let stamp = vals.pop().expect("two arguments");
            self.note_panic_stamp(stamp);
        }
        self.exec_builtin_values(frames, body, args, kind, name, generic_args, vals, cmp_discriminants)
    }

    /// Execute a builtin whose arguments are already evaluated
    /// values: the dispatch half of [`Self::exec_builtin`], split
    /// out for the interpreter's run-time name fallback, whose
    /// call sites hold evaluated values, not operands.
    pub(crate) fn exec_builtin_values(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        args: &[Operand],
        kind: Builtin,
        name: &str,
        generic_args: &[String],
        vals: Vec<Value>,
        cmp_discriminants: Option<&[i128]>,
    ) -> InterpResult<Value> {
        let arity = |n: usize| -> InterpResult<()> {
            if vals.len() == n {
                Ok(())
            } else {
                Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    format!("{n} argument(s)"),
                    format!("{} argument(s)", vals.len()),
                ))
            }
        };
        match kind {
            Builtin::Print => {
                arity(1)?;
        self.note_stdout_use(frames)?;
        let text = self.format_arguments(frames, &self.coerce_arguments(frames, &vals[0], name)?, 0)?;
                self.push_output(&text);
                Ok(Value::Unit)
            }
            Builtin::Eprint => {
                arity(1)?;
                let text = self.format_arguments(frames, &self.coerce_arguments(frames, &vals[0], name)?, 0)?;
                self.push_error_output(&text);
                Ok(Value::Unit)
            }
            Builtin::FfiIdentity => {
                arity(1)?;
                Ok(vals[0].clone())
            }
            Builtin::FfiReturnTwo => {
                arity(0)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Uint(10), Value::Uint(20)],
                })
            }
            Builtin::FfiReturnFive => {
                arity(0)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![
                        Value::Uint(10),
                        Value::Uint(20),
                        Value::Uint(30),
                        Value::Uint(40),
                        Value::Uint(50),
                    ],
                })
            }
            Builtin::FfiDbgCall => {
                arity(2)?;
                match &vals[0] {
                    Value::FnPtr(target) => {
                        let target = target.clone();
                        let data = vals[1].clone();
                        self.call_sync(frames, &target, vec![data])
                    }
                    other => Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "function pointer callback",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::FfiEmptyStruct => {
                arity(3)?;
                let v1 = aggregate_fields(&vals[0]).ok_or_else(|| {
                    unsupported(format!("call to {name}: first argument is not a struct value"))
                })?;
                let v2 = aggregate_fields(&vals[2]).ok_or_else(|| {
                    unsupported(format!("call to {name}: third argument is not a struct value"))
                })?;
                if v1.len() != 6 || v2.len() != 6 {
                    return Err(unsupported(format!(
                        "call to {name}: ManyInts shape not recognised"
                    )));
                }
                // The C asserts: each scalar field of v1 is the
                // matching field of v2 plus one, and likewise
                // for the nested TwoU8s (field 5).
                let mut holds = true;
                for i in 0..5 {
                    holds &= as_i128(&v1[i], name)? == as_i128(&v2[i], name)? + 1;
                }
                let n1 = aggregate_fields(&v1[5]).ok_or_else(|| {
                    unsupported(format!("call to {name}: nested struct shape not recognised"))
                })?;
                let n2 = aggregate_fields(&v2[5]).ok_or_else(|| {
                    unsupported(format!("call to {name}: nested struct shape not recognised"))
                })?;
                if n1.len() != 2 || n2.len() != 2 {
                    return Err(unsupported(format!(
                        "call to {name}: nested struct shape not recognised"
                    )));
                }
                for i in 0..2 {
                    holds &= as_u128(&n1[i], name)? == as_u128(&n2[i], name)? + 1;
                }
                if !holds {
                    return Err(InterpError::Abort {
                        message: format!(
                            "{name}: assertion `v1 == v2 + 1` failed (C assert in rust_test_helpers)"
                        ),
                    });
                }
                Ok(Value::Unit)
            }
            Builtin::FfiGetTestInt => {
                arity(0)?;
                Ok(Value::Int(1))
            }
            Builtin::FfiGetNullPtr => {
                arity(0)?;
                Ok(Value::Ref(RefTarget::Null))
            }
            Builtin::FfiDbgAbi1 => {
                arity(1)?;
                let f = aggregate_fields(&vals[0]).ok_or_else(|| {
                    unsupported(format!("call to {name}: argument is not a struct value"))
                })?;
                if f.len() != 4 {
                    return Err(unsupported(format!(
                        "call to {name}: quad shape not recognised"
                    )));
                }
                let a = as_u128(&f[0], name)? as u64;
                let b = as_u128(&f[1], name)? as u64;
                let c = as_u128(&f[2], name)? as u64;
                let d = as_u128(&f[3], name)? as u64;
                Ok(aggregate_like(
                    &vals[0],
                    vec![
                        Value::Uint(c.wrapping_add(1) as u128),
                        Value::Uint(d.wrapping_sub(1) as u128),
                        Value::Uint(a.wrapping_add(1) as u128),
                        Value::Uint(b.wrapping_sub(1) as u128),
                    ],
                ))
            }
            Builtin::FfiDbgAbi2 => {
                arity(1)?;
                let f = aggregate_fields(&vals[0]).ok_or_else(|| {
                    unsupported(format!("call to {name}: argument is not a struct value"))
                })?;
                if f.len() != 3 {
                    return Err(unsupported(format!(
                        "call to {name}: floats shape not recognised"
                    )));
                }
                let a = as_f64(&f[0], name)?;
                let c = as_f64(&f[2], name)?;
                Ok(aggregate_like(
                    &vals[0],
                    vec![
                        Value::Float(c + 1.0),
                        Value::Uint(0xff),
                        Value::Float(a - 1.0),
                    ],
                ))
            }
            Builtin::FfiDbgAbi3 | Builtin::FfiDbgAbi4 => {
                arity(1)?;
                let f = aggregate_fields(&vals[0]).ok_or_else(|| {
                    unsupported(format!("call to {name}: argument is not a struct value"))
                })?;
                if f.len() != 3 {
                    return Err(unsupported(format!(
                        "call to {name}: char_char shape not recognised"
                    )));
                }
                let a = as_u128(&f[0], name)? as u8;
                let b = as_u128(&f[1], name)? as u8;
                let c = as_f64(&f[2], name)?;
                // abi_4's third field is an f32: the C adds in
                // float precision; abi_3's is an f64.
                let c = if kind == Builtin::FfiDbgAbi4 {
                    (c as f32 + 1.0f32) as f64
                } else {
                    c + 1.0
                };
                Ok(aggregate_like(
                    &vals[0],
                    vec![
                        Value::Uint(a.wrapping_add(1) as u128),
                        Value::Uint(b.wrapping_sub(1) as u128),
                        Value::Float(c),
                    ],
                ))
            }
            Builtin::FfiArgField { arg, field } => {
                let Some(target) = vals.get(arg) else {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        format!("at least {} argument(s)", arg + 1),
                        format!("{} argument(s)", vals.len()),
                    ));
                };
                let f = aggregate_fields(target).ok_or_else(|| {
                    unsupported(format!(
                        "call to {name}: argument {arg} is not a struct value"
                    ))
                })?;
                let Some(value) = f.get(field) else {
                    return Err(unsupported(format!(
                        "call to {name}: argument {arg} has no field {field}"
                    )));
                };
                Ok(value.clone())
            }
            Builtin::FfiInterestingAverage => {
                if vals.is_empty() {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "at least 1 argument(s)",
                        "0 argument(s)",
                    ));
                }
                let n = as_u128(&vals[0], name)?;
                let rest = &vals[1..];
                if rest.len() as u128 != 2 * n {
                    return Err(unsupported(format!(
                        "call to {name}: expected {n} (i64, f64) pairs, got {} values",
                        rest.len()
                    )));
                }
                let mut sum = 0.0f64;
                for pair in rest.chunks(2) {
                    sum += as_i128(&pair[0], name)? as f64;
                    sum += as_f64(&pair[1], name)?;
                }
                Ok(Value::Float(sum / n as f64))
            }
            Builtin::FfiValistInterestingAverage => {
                arity(2)?;
                let n = as_u128(&vals[0], name)?;
                let id = self.va_list_handle(frames, &vals[1])?;
                let mut sum = 0.0f64;
                for _ in 0..n {
                    let int = self.va_list_next(id)?;
                    sum += as_i128(&int, name)? as f64;
                    let float = self.va_list_next(id)?;
                    sum += as_f64(&float, name)?;
                }
                Ok(Value::Float(sum / n as f64))
            }
            Builtin::FfiVaListNextI32 => {
                arity(1)?;
                let id = self.va_list_handle(frames, &vals[0])?;
                let value = self.va_list_next(id)?;
                Ok(Value::Int(as_i128(&value, name)? as i32 as i128))
            }
            Builtin::VaArg => {
                arity(1)?;
                let id = self.va_list_handle(frames, &vals[0])?;
                self.va_list_next(id)
            }
            Builtin::FfiSquare128 => {
                arity(1)?;
                let v = as_i128(&vals[0], name)?;
                Ok(Value::Int(v.wrapping_mul(v)))
            }
            Builtin::FfiSub128 => {
                arity(2)?;
                let a = as_i128(&vals[0], name)?;
                let b = as_i128(&vals[1], name)?;
                Ok(Value::Int(a.wrapping_sub(b)))
            }
            Builtin::FfiIncrementAllParts => {
                arity(1)?;
                let Some(bits) = union_u64_bits(&vals[0]) else {
                    return Err(unsupported(format!(
                        "call to {name}: argument is not a 64-bit union value"
                    )));
                };
                let lo = (bits as u32).wrapping_add(2) as u64;
                let hi = ((bits >> 32) as u32).wrapping_add(2) as u64;
                let combined = (hi << 32) | lo;
                Ok(union_with_u64_bits(&vals[0], combined.wrapping_add(1)))
            }
            Builtin::FfiLeakUninitData => {
                arity(3)?;
                let b = as_u128(&vals[1], name)?;
                Ok(Value::Uint(b & 0xFFFF))
            }
            Builtin::FfiNewSome => {
                if vals.is_empty() || vals.len() > 2 {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "1 or 2 argument(s)",
                        format!("{} argument(s)", vals.len()),
                    ));
                }
                Ok(Value::Enum {
                    variant: 1,
                    fields: vals.clone(),
                })
            }
            Builtin::FfiNewNone => {
                arity(0)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                })
            }
            Builtin::FfiUnpackOption => {
                if vals.len() < 2 || vals.len() > 3 {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "2 or 3 argument(s)",
                        format!("{} argument(s)", vals.len()),
                    ));
                }
                let (variant, fields) = match &vals[0] {
                    Value::Enum { variant, fields } => (*variant, fields.clone()),
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "tagged option enum value",
                            other.kind_name(),
                        ))
                    }
                };
                if variant == 0 {
                    return Ok(Value::Int(0));
                }
                if variant != 1 || fields.len() != vals.len() - 1 {
                    return Err(unsupported(format!(
                        "call to {name}: option shape not recognised"
                    )));
                }
                for (ptr, value) in vals[1..].iter().zip(fields) {
                    if matches!(ptr, Value::Ref(RefTarget::Null)) {
                        return Err(InterpError::Abort {
                            message: format!(
                                "{name}: assertion `into` failed (C assert in rust_test_helpers)"
                            ),
                        });
                    }
                    let target = self.deref_value(frames, ptr)?;
                    self.write_resolved(frames, &target, value)?;
                }
                Ok(Value::Int(1))
            }
            Builtin::ProcMacroIsAvailable => {
                arity(0)?;
                Ok(Value::Bool(false))
            }
            Builtin::EnvArgs | Builtin::EnvArgsOs => {
                arity(0)?;
                // argv[0] is the program path the run was invoked
                // with, then the configured guest arguments, in
                // order. The wrapper shape matches the real std
                // layout — `Args { inner: ArgsOs }`, `ArgsOs {
                // inner: sys::args::common::Args }`, and the common
                // `Args { inner: vec::IntoIter }` — because std
                // bodies that navigate the wrappers field by field
                // (the `ExactSizeIterator::len` one-liners) are
                // collected and run against this value: a shallower
                // shape makes their second field step land on the
                // iteration state's element snapshot. The innermost
                // value is the `vec::IntoIter` in this model's
                // iteration-state shape ([element snapshot, next
                // index]).
                let mut items: Vec<Value> = Vec::new();
                if !self.program_path().is_empty() {
                    items.push(Value::Str(self.program_path().to_string()));
                }
                for arg in self.guest_args() {
                    items.push(Value::Str(arg.clone()));
                }
                let common_args = Value::Enum {
                    variant: 0,
                    fields: vec![owned_iter_state(items)],
                };
                let args_os = Value::Enum {
                    variant: 0,
                    fields: vec![common_args],
                };
                if kind == Builtin::EnvArgsOs {
                    Ok(args_os)
                } else {
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![args_os],
                    })
                }
            }
            Builtin::ProcessExit => {
                arity(1)?;
                let code = as_i128(&vals[0], name)? as i32;
                Err(InterpError::ProcessExit { code })
            }
            Builtin::Stdout => {
                arity(0)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Unit],
                })
            }
            Builtin::WriteFmt => {
                arity(2)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                let self_ty = self_ty
                    .strip_prefix("&mut ")
                    .or_else(|| self_ty.strip_prefix("&"))
                    .unwrap_or(self_ty);
                let text =
                    self.format_arguments(frames, &self.coerce_arguments(frames, &vals[1], name)?, 0)?;
                match self_ty {
                    "std::io::Stdout" | "std::io::StdoutLock" => {
                        self.push_output(&text);
                    }
                    "std::io::Stderr" | "std::io::StderrLock" => {
                        self.push_error_output(&text);
                    }
                    _ if self_ty.starts_with("std::vec::Vec<u8") => {
                        // `Write for Vec<u8>` appends the formatted
                        // bytes to the vector.
                        let target = self.deref_value(frames, &vals[0])?;
                        let state = self.read_resolved(frames, &target)?;
                        match state {
                            Value::Enum {
                                variant,
                                mut fields,
                            } => {
                                fields.extend(
                                    text.bytes().map(|b| Value::Uint(b as u128)),
                                );
                                self.write_resolved(
                                    frames,
                                    &target,
                                    Value::Enum { variant, fields },
                                )?;
                            }
                            ref other => {
                                return Err(InterpError::type_mismatch(
                                    "Write::write_fmt for Vec<u8>",
                                    "vector value",
                                    other.kind_name(),
                                ))
                            }
                        }
                    }
                    _ => {
                        return Err(unsupported(format!(
                            "Write::write_fmt for {self_ty} is not runtime-supported"
                        )))
                    }
                }
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Unit],
                })
            }
            Builtin::FmtWrite => {
                arity(2)?;
                let text =
                    self.format_arguments(frames, &self.coerce_arguments(frames, &vals[1], name)?, 0)?;
                // The output is a `&mut dyn Write`: resolve it to
                // the dyn value, then deliver the formatted text
                // through the concrete type's own `write_str`. The
                // real `fmt::write` body walks the template and
                // calls `write_str` piece by piece; one `write_str`
                // of the whole formatted text is the same
                // observable output, and matches the fast path the
                // body itself takes for a plain string.
                let dyn_value = match &vals[0] {
                    Value::DynRef { .. } => vals[0].clone(),
                    Value::Ref(_) => {
                        let target = self.deref_value(frames, &vals[0])?;
                        self.read_resolved(frames, &target)?
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "&mut dyn Write value",
                            other.kind_name(),
                        ))
                    }
                };
                let Value::DynRef {
                    base, concrete, ..
                } = dyn_value
                else {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "&mut dyn Write value",
                        dyn_value.kind_name(),
                    ));
                };
                let Some(impl_name) = self.scan_virtual_impl(
                    &concrete,
                    "core::fmt::Write",
                    "write_str",
                ) else {
                    return Err(unsupported(format!(
                        "fmt::write output type {concrete} has no collected write_str impl"
                    )));
                };
                let receiver = match &*base {
                    Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. } => {
                        (*base).clone()
                    }
                    _ => {
                        return Err(unsupported(format!(
                            "fmt::write output type {concrete} is not held behind a reference"
                        )))
                    }
                };
                self.call_sync(frames, &impl_name, vec![receiver, Value::Str(text)])
            }
            Builtin::Format => {
                arity(1)?;
                let text = self.format_arguments(frames, &self.coerce_arguments(frames, &vals[0], name)?, 0)?;
                Ok(Value::Str(text))
            }
            Builtin::ArgumentsFromStr => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                Ok(Value::FmtArguments(FmtArguments::Plain(s)))
            }
            Builtin::ArgumentsNew => {
                arity(2)?;
                let template = self.resolve_bytes(frames, &vals[0])?;
                let args = self.resolve_arg_array(frames, &vals[1])?;
                Ok(Value::FmtArguments(FmtArguments::Template {
                    template,
                    args,
                }))
            }
            Builtin::ArgumentNew(trait_) => {
                arity(1)?;
                let ty = generic_args.first().cloned().unwrap_or_default();
                Ok(Value::FmtArg {
                    target: Box::new(vals.into_iter().next().expect("arity checked")),
                    trait_,
                    ty,
                })
            }
            Builtin::FmtMethod(trait_) => {
                // A UFCS `fmt` trait call whose `Self` instance
                // has no converted body: format the receiver
                // with the structural engine, under the captured
                // formatter's own options, and append the text
                // to its buffer — the same output the type's
                // `Argument` formatting produces for the same
                // value and options.
                arity(2)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                let pointee = self.resolve_value(frames, &vals[0])?;
                let id = self.formatter_id_of(frames, &vals[1])?;
                let capture = self.capture_of(id)?;
                let options = FmtOptions {
                    flags: capture.flags,
                    width: capture.width,
                    precision: capture.precision,
                };
                let text = self.format_value(
                    frames,
                    &pointee,
                    trait_,
                    self_ty,
                    &options,
                    capture.indent,
                )?;
                self.formatter_push(id, &text);
                Ok(fmt_result_ok())
            }
            Builtin::FormatterWriteStr => {
                arity(2)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                let s = self.resolve_string(frames, &vals[1])?;
                self.formatter_push(id, &s);
                Ok(fmt_result_ok())
            }
            Builtin::FormatterWriteChar => {
                arity(2)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                let c = match self.resolve_value(frames, &vals[1])? {
                    Value::Uint(v) => as_char(v)?,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "Formatter::write_char",
                            "char value",
                            other.kind_name(),
                        ))
                    }
                };
                self.formatter_push(id, &c.to_string());
                Ok(fmt_result_ok())
            }
            Builtin::FormatterPad => {
                arity(2)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                let s = self.resolve_string(frames, &vals[1])?;
                let options = self.capture_of(id)?.options();
                let text = pad_str(&s, &options);
                self.formatter_push(id, &text);
                Ok(fmt_result_ok())
            }
            Builtin::FormatterWriteFmt => {
                arity(2)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                let indent = self.capture_of(id)?.indent;
                let text = self.format_arguments(
                    frames,
                    &self.coerce_arguments(frames, &vals[1], name)?,
                    indent,
                )?;
                self.formatter_push(id, &text);
                Ok(fmt_result_ok())
            }
            Builtin::FormatterAlternate => {
                arity(1)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                Ok(Value::Bool(self.capture_of(id)?.options().alternate()))
            }
            Builtin::FormatterWidth => {
                arity(1)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                Ok(match self.capture_of(id)?.options().width() {
                    Some(w) => Value::Enum {
                        variant: 1,
                        fields: vec![Value::Uint(w as u128)],
                    },
                    None => Value::Enum {
                        variant: 0,
                        fields: vec![],
                    },
                })
            }
            Builtin::FormatterPrecision => {
                arity(1)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                Ok(match self.capture_of(id)?.options().precision() {
                    Some(p) => Value::Enum {
                        variant: 1,
                        fields: vec![Value::Uint(p as u128)],
                    },
                    None => Value::Enum {
                        variant: 0,
                        fields: vec![],
                    },
                })
            }
            Builtin::FormatterFill => {
                arity(1)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                Ok(Value::Uint(self.capture_of(id)?.options().fill() as u32 as u128))
            }
            Builtin::FormatterSignPlus => {
                arity(1)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                Ok(Value::Bool(self.capture_of(id)?.options().sign_plus()))
            }
            Builtin::FormatterSignAwareZeroPad => {
                arity(1)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                Ok(Value::Bool(self.capture_of(id)?.options().zero_pad()))
            }
            Builtin::DebugStructFinish { fields } => {
                arity(2 + 2 * fields)?;
                self.exec_debug_struct_finish(frames, &vals, fields)
            }
            Builtin::DebugStructFieldsFinish => {
                arity(4)?;
                self.exec_debug_struct_fields_finish(frames, &vals)
            }
            Builtin::DebugTupleFieldsFinish => {
                arity(3)?;
                self.exec_debug_tuple_fields_finish(frames, &vals)
            }
            Builtin::DebugTupleFinish { fields } => {
                arity(2 + fields)?;
                self.exec_debug_tuple_finish(frames, &vals, fields)
            }
            Builtin::FmtBuilderNew(kind) => {
                let (fmt_arg, builder_name) = match kind {
                    FmtBuilderKind::Struct | FmtBuilderKind::Tuple => {
                        arity(2)?;
                        let name = self.resolve_string(frames, &vals[1])?;
                        (vals[0].clone(), Some(name))
                    }
                    _ => {
                        arity(1)?;
                        (vals[0].clone(), None)
                    }
                };
                let formatter = self.formatter_id_of(frames, &fmt_arg)?;
                let id = self.new_fmt_builder(FmtBuilderState {
                    formatter,
                    kind,
                    name: builder_name,
                    has_fields: false,
                    result_ok: true,
                });
                Ok(Value::FmtBuilder(id))
            }
            Builtin::FmtBuilderEntry => {
                arity(2)?;
                let id = self.builder_id_of(frames, &vals[0])?;
                self.builder_entry(frames, id, &vals[1])?;
                Ok(Value::Ref(RefTarget::value(Value::FmtBuilder(id))))
            }
            Builtin::FmtBuilderMapEntry => {
                arity(3)?;
                let id = self.builder_id_of(frames, &vals[0])?;
                self.builder_map_entry(frames, id, &vals[1], &vals[2])?;
                Ok(Value::Ref(RefTarget::value(Value::FmtBuilder(id))))
            }
            Builtin::FmtBuilderStructField => {
                arity(3)?;
                let id = self.builder_id_of(frames, &vals[0])?;
                let field_name = self.resolve_string(frames, &vals[1])?;
                self.builder_struct_field(frames, id, &field_name, &vals[2])?;
                Ok(Value::Ref(RefTarget::value(Value::FmtBuilder(id))))
            }
            Builtin::FmtBuilderTupleField => {
                arity(2)?;
                let id = self.builder_id_of(frames, &vals[0])?;
                self.builder_tuple_field(frames, id, &vals[1])?;
                Ok(Value::Ref(RefTarget::value(Value::FmtBuilder(id))))
            }
            Builtin::FmtBuilderFinish { exhaustive } => {
                arity(1)?;
                let id = self.builder_id_of(frames, &vals[0])?;
                self.builder_finish(id, exhaustive)?;
                Ok(fmt_result_ok())
            }
            Builtin::ArgumentFromUsize => {
                arity(1)?;
                // Current core passes the count by reference
                // (`from_usize(&usize)`: the format-args capture
                // tuple holds a reference to the count), so read
                // through one reference level; a bare integer is
                // accepted as well. core truncates the count to
                // u16.
                let arg = self.read_ref_once(frames, &vals[0])?;
                let n = match arg {
                    Value::Uint(v) => v as u16,
                    Value::Int(v) => v as u16,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            "Argument::from_usize",
                            "integer",
                            other.kind_name(),
                        ))
                    }
                };
                Ok(Value::FmtCount(n))
            }
            Builtin::RangeInclusiveNew => {
                arity(2)?;
                let mut it = vals.into_iter();
                let start = it.next().expect("arity checked");
                let end = it.next().expect("arity checked");
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![start, end, Value::Bool(false)],
                })
            }
            Builtin::ThreadSpawn => {
                arity(1)?;
                let callable = vals.into_iter().next().expect("arity checked");
                self.spawn_callable(callable)
            }
            Builtin::ThreadBuilderSpawn => {
                // `(builder, f)`: the builder carries nothing the
                // model observes beyond what its fields hold; the
                // callable is the second argument. `Builder::spawn`
                // returns `io::Result<JoinHandle>`: the model's
                // spawn cannot fail, so the handle is the `Ok`.
                arity(2)?;
                let callable = vals.into_iter().nth(1).expect("arity checked");
                Ok(ok_result(self.spawn_callable(callable)?))
            }
            Builtin::ThreadScopedSpawn => {
                // `(&scope, f)`: the scope's identity is not
                // observed; the callable is the second argument.
                arity(2)?;
                let callable = vals.into_iter().nth(1).expect("arity checked");
                self.spawn_callable(callable)
            }
            Builtin::ThreadJoin | Builtin::ScopedThreadJoin => {
                arity(1)?;
                let Some(id) = join_handle_id(&vals[0]) else {
                    return Err(InterpError::type_mismatch(
                        "JoinHandle::join",
                        "a join handle value",
                        vals[0].kind_name(),
                    ));
                };
                match self.join_thread(frames, id)? {
                    Ok(value) => Ok(ok_result(value)),
                    // The error is the thread's panic payload as
                    // the `Box<dyn Any + Send>` the guest
                    // downcasts, in the same shape the
                    // `catch_unwind` cleanup produces.
                    Err(payload) => Ok(err_result(Value::DynRef {
                        base: Box::new(Value::Enum {
                            variant: 0,
                            fields: vec![(*payload.value).clone()],
                        }),
                        concrete: payload.ty.clone(),
                        traits: vec![
                            "std::any::Any".to_string(),
                            "std::marker::Send".to_string(),
                        ],
                        drop_glue: None,
                    })),
                }
            }
            Builtin::ThreadScope => {
                arity(1)?;
                let callable = vals.into_iter().next().expect("arity checked");
                let scope_ref = Value::Ref(RefTarget::value(Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                }));
                let result =
                    self.call_callable_to_completion(frames, &callable, vec![scope_ref])?;
                // Every thread spawned inside the scope must
                // finish before the scope returns; the cooperative
                // model runs all pending threads here.
                self.drain_threads(frames)?;
                Ok(result)
            }
            Builtin::ThreadCurrent => {
                arity(0)?;
                Ok(id_handle(self.current_thread_id()))
            }
            Builtin::ThreadIdOf => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                let thread = self.read_resolved(frames, &target)?;
                match channel_handle_id(&thread) {
                    Some(id) => Ok(id_handle(id)),
                    None => Err(InterpError::type_mismatch(
                        "Thread::id",
                        "a thread value",
                        thread.kind_name(),
                    )),
                }
            }
            Builtin::ThreadSleep => {
                arity(1)?;
                self.drain_threads(frames)?;
                Ok(Value::Unit)
            }
            Builtin::ThreadYield => {
                arity(0)?;
                self.drain_threads(frames)?;
                Ok(Value::Unit)
            }
            Builtin::ThreadBuilderNew => {
                arity(0)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![none_value(), none_value()],
                })
            }
            Builtin::ThreadBuilderName | Builtin::ThreadBuilderStackSize => {
                arity(2)?;
                let field = if kind == Builtin::ThreadBuilderName { 0 } else { 1 };
                let mut vals = vals.into_iter();
                let builder = vals.next().expect("arity checked");
                let setting = vals.next().expect("arity checked");
                match builder {
                    Value::Enum { variant: 0, mut fields } if fields.len() == 2 => {
                        fields[field] = some_value(setting);
                        Ok(Value::Enum { variant: 0, fields })
                    }
                    other => Err(InterpError::type_mismatch(
                        "thread Builder",
                        "a builder value",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::DurationNew => {
                arity(2)?;
                let secs = uint_arg_bits(&vals[0], name)?;
                let nanos = uint_arg_bits(&vals[1], name)?;
                self.duration_value(secs, nanos)
            }
            Builtin::DurationFromSecs => {
                arity(1)?;
                self.duration_value(uint_arg_bits(&vals[0], name)?, 0)
            }
            Builtin::DurationFromMillis => {
                arity(1)?;
                let ms = uint_arg_bits(&vals[0], name)?;
                self.duration_value(ms / 1_000, (ms % 1_000) * 1_000_000)
            }
            Builtin::DurationFromMicros => {
                arity(1)?;
                let us = uint_arg_bits(&vals[0], name)?;
                self.duration_value(us / 1_000_000, (us % 1_000_000) * 1_000)
            }
            Builtin::DurationFromNanos => {
                arity(1)?;
                let ns = uint_arg_bits(&vals[0], name)?;
                self.duration_value(ns / 1_000_000_000, ns % 1_000_000_000)
            }
            Builtin::TestMainEnvArgs => {
                arity(1)?;
                let tests = self.read_test_descs(frames, &vals[0])?;
                // A `-Z` option is rejected before anything else
                // happens: the harness this models is a
                // stable-channel one, and libtest's own gate is
                // `is_nightly()` -- a nightly-built harness, or
                // `RUSTC_BOOTSTRAP` present in the environment.
                // The modelled harness is not nightly-built, so
                // the environment decides (the run's environment
                // is the guest's, as for `RUST_TEST_THREADS`
                // below): with no bootstrap, every `-Z` use is
                // refused, `-Z unstable-options` included. With
                // bootstrap, the one exception is `-Z
                // unstable-options` unlocking the json format, the
                // pair libtest's own JSON consumers pass; any
                // other `-Z` use, and unstable-options without a
                // json request, is refused as before. The json
                // format without the unlocking option is likewise
                // refused before listing or running.
                let raw_args: Vec<String> = self.guest_args().to_vec();
                let opts = HarnessOpts::parse(&raw_args);
                let bootstrap = std::env::var("RUSTC_BOOTSTRAP").is_ok();
                let mut unstable_options = false;
                let mut other_z = false;
                {
                    let mut j = 0;
                    while j < raw_args.len() {
                        let a = &raw_args[j];
                        if a == "-Zunstable-options" {
                            unstable_options = true;
                        } else if a == "-Z" {
                            if raw_args.get(j + 1).map(String::as_str)
                                == Some("unstable-options")
                            {
                                unstable_options = true;
                                j += 1;
                            } else {
                                other_z = true;
                            }
                        } else if a.starts_with("-Z") {
                            other_z = true;
                        }
                        j += 1;
                    }
                }
                if other_z || (unstable_options && (!bootstrap || !opts.json)) {
                    self.push_error_output(
                        "error: the option `Z` is only accepted on the nightly compiler\n",
                    );
                    return Ok(test_exit_code(101));
                }
                if opts.json && !unstable_options {
                    self.push_error_output(
                        "error: The \"json\" format is only accepted on the nightly \
                         compiler with -Z unstable-options\n",
                    );
                    return Ok(test_exit_code(101));
                }
                let mut selected: Vec<&TestDesc> = Vec::new();
                let mut filtered_out = 0usize;
                for test in &tests {
                    if opts.selected(test) {
                        selected.push(test);
                    } else {
                        filtered_out += 1;
                    }
                }
                if opts.list {
                    if opts.json {
                        // `--list` in the json format is the
                        // discovery event stream: a suite event,
                        // one `discovered` event per selected
                        // test carrying its descriptor's span,
                        // and the closing suite event with the
                        // counts.
                        self.push_output(
                            "{ \"type\": \"suite\", \"event\": \"discovery\" }\n",
                        );
                        let mut discovered_ignored = 0usize;
                        for test in &selected {
                            if test.ignore {
                                discovered_ignored += 1;
                            }
                            self.push_output(&format!(
                                "{{ \"type\": \"test\", \"event\": \"discovered\", \
                                 \"name\": \"{}\", \"ignore\": {}, \
                                 \"ignore_message\": \"{}\", \"source_path\": \"{}\", \
                                 \"start_line\": {}, \"start_col\": {}, \
                                 \"end_line\": {}, \"end_col\": {} }}\n",
                                json_escape(&test.name),
                                test.ignore,
                                json_escape(
                                    test.ignore_message.as_deref().unwrap_or("")
                                ),
                                json_escape(&test.source_file),
                                test.start_line,
                                test.start_col,
                                test.end_line,
                                test.end_col
                            ));
                        }
                        self.push_output(&format!(
                            "{{ \"type\": \"suite\", \"event\": \"completed\", \
                             \"tests\": {}, \"benchmarks\": 0, \"total\": {}, \
                             \"ignored\": {} }}\n",
                            selected.len(),
                            selected.len(),
                            discovered_ignored
                        ));
                        return Ok(test_exit_code(0));
                    }
                    // `--list`: one `{name}: test` line per
                    // selected test; the default format adds the
                    // count footer, the terse format does not.
                    for test in &selected {
                        self.push_output(&format!("{}: test\n", test.name));
                    }
                    if !opts.terse {
                        self.push_output(&format!(
                            "\n{} tests, 0 benchmarks\n",
                            selected.len()
                        ));
                    }
                    return Ok(test_exit_code(0));
                }
                let mut passed = 0usize;
                let mut failed = 0usize;
                let mut ignored = 0usize;
                // libtest's concurrency setup reads
                // RUST_TEST_THREADS before any test runs and
                // panics on a value that is not a positive
                // integer; the panic escapes the harness main.
                if let Ok(v) = std::env::var("RUST_TEST_THREADS") {
                    if !matches!(v.parse::<usize>(), Ok(n) if n > 0) {
                        return Err(InterpError::panic(format!(
                            "RUST_TEST_THREADS is `{v}`, should be a positive integer."
                        )));
                    }
                }
                let mut failure_names: Vec<String> = Vec::new();
                let mut failure_sections: Vec<String> = Vec::new();
                if opts.json {
                    self.push_output(&format!(
                        "{{ \"type\": \"suite\", \"event\": \"started\", \
                         \"test_count\": {} }}\n",
                        selected.len()
                    ));
                } else {
                    self.push_output(&format!(
                        "\nrunning {} {}\n",
                        selected.len(),
                        if selected.len() == 1 { "test" } else { "tests" }
                    ));
                }
                for test in selected {
                    let annotation = match test.should_panic {
                        ShouldPanic::No => "",
                        _ => " - should panic",
                    };
                    if test.ignore && !opts.runs_ignored() {
                        ignored += 1;
                        if opts.json {
                            self.push_output(&format!(
                                "{{ \"type\": \"test\", \"event\": \"started\", \
                                 \"name\": \"{}\" }}\n",
                                json_escape(&test.name)
                            ));
                            let message = match &test.ignore_message {
                                Some(msg) => format!(
                                    ", \"message\": \"{}\"",
                                    json_escape(msg)
                                ),
                                None => String::new(),
                            };
                            self.push_output(&format!(
                                "{{ \"type\": \"test\", \"name\": \"{}\", \
                                 \"event\": \"ignored\"{} }}\n",
                                json_escape(&test.name),
                                message
                            ));
                        } else {
                            let note = match &test.ignore_message {
                                Some(msg) => format!(", {msg}"),
                                None => String::new(),
                            };
                            self.push_output(&format!(
                                "test {}{} ... ignored{}\n",
                                test.name, annotation, note
                            ));
                        }
                        continue;
                    }
                    if opts.json {
                        self.push_output(&format!(
                            "{{ \"type\": \"test\", \"event\": \"started\", \
                             \"name\": \"{}\" }}\n",
                            json_escape(&test.name)
                        ));
                    } else if opts.nocapture {
                        self.push_output(&format!(
                            "test {}{} ... ",
                            test.name, annotation
                        ));
                    }
                    let out_mark = self.output_len();
                    let err_mark = self.error_output_len();
                    self.clear_panic_stamp();
                    let result = self.call_callable_to_completion(
                        frames,
                        &test.callable,
                        Vec::new(),
                    );
                    // The test's own output since the marks: in
                    // capture mode it is drained here (kept only
                    // for a failure's section); in nocapture mode
                    // it already flowed into the output live.
                    let (cap_out, cap_err) = if opts.nocapture {
                        (String::new(), String::new())
                    } else {
                        (
                            self.drain_output_since(out_mark),
                            self.drain_error_output_since(err_mark),
                        )
                    };
                    // `Some(body)` is the failure's section body:
                    // the captured output followed by the
                    // failure's account. A bare panic in
                    // nocapture mode has no section (its report
                    // went to the guest's stderr as it ran).
                    // For a test that panics as it was expected
                    // to, the panic report without the backtrace
                    // note: the json format's `--show-output`
                    // account of a passing should-panic test is
                    // its captured output plus this report.
                    let mut pass_note: Option<String> = None;
                    let failure_body: Option<Option<String>> = match result {
                        Ok(_) => match test.should_panic {
                            ShouldPanic::No => None,
                            _ => Some(Some(format!(
                                "{cap_out}{cap_err}note: test did not panic as expected at {}:{}:{}",
                                test.source_file, test.start_line, test.start_col
                            ))),
                        },
                        Err(InterpError::Panic { message, payload }) => {
                            self.note_panic_consumed();
                            match &test.should_panic {
                                ShouldPanic::No => {
                                    if opts.nocapture {
                                        Some(None)
                                    } else {
                                        let report = match self.panic_location() {
                                            Some((file, line, col)) => format!(
                                                "\nthread '{}' ({}) panicked at {}:{}:{}:\n{}\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n",
                                                test.name,
                                                self.current_thread_id(),
                                                file,
                                                line,
                                                col,
                                                message
                                            ),
                                            None => format!(
                                                "\nthread '{}' ({}) panicked:\n{}\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n",
                                                test.name,
                                                self.current_thread_id(),
                                                message
                                            ),
                                        };
                                        Some(Some(format!(
                                            "{cap_out}{cap_err}{report}"
                                        )))
                                    }
                                }
                                ShouldPanic::Yes => {
                                    pass_note = Some(panic_report_head(
                                        test,
                                        self.current_thread_id(),
                                        self.panic_location(),
                                        &message,
                                    ));
                                    None
                                }
                                ShouldPanic::YesWithMessage(expected) => {
                                    let text = payload
                                        .as_ref()
                                        .and_then(|p| {
                                            self.resolve_string(frames, &p.value).ok()
                                        })
                                        .unwrap_or_else(|| message.clone());
                                    let is_string = match &payload {
                                        None => true,
                                        Some(p) => {
                                            p.ty == "&str"
                                                || p.ty == "std::string::String"
                                                || self
                                                    .resolve_string(frames, &p.value)
                                                    .is_ok()
                                        }
                                    };
                                    if !is_string {
                                        let ty = payload
                                            .as_ref()
                                            .map(|p| p.ty.clone())
                                            .unwrap_or_default();
                                        let report = panic_report_full(
                                            test,
                                            self.current_thread_id(),
                                            self.panic_location(),
                                            &message,
                                        );
                                        Some(Some(format!(
                                            "{cap_out}{cap_err}{report}note: expected panic with string value,\n found non-string value: `{ty}`\n     expected substring: \"{expected}\""
                                        )))
                                    } else if text.contains(expected.as_str()) {
                                        pass_note = Some(panic_report_head(
                                            test,
                                            self.current_thread_id(),
                                            self.panic_location(),
                                            &message,
                                        ));
                                        None
                                    } else {
                                        let report = panic_report_full(
                                            test,
                                            self.current_thread_id(),
                                            self.panic_location(),
                                            &message,
                                        );
                                        Some(Some(format!(
                                            "{cap_out}{cap_err}{report}note: panic did not contain expected string\n      panic message: \"{text}\"\n expected substring: \"{expected}\""
                                        )))
                                    }
                                }
                            }
                        }
                        Err(e) => return Err(e),
                    };
                    match failure_body {
                        None => {
                            passed += 1;
                            if opts.json {
                                // With `--show-output`, a passing
                                // test's event carries its
                                // captured output (plus the report
                                // of a panic it was expected to
                                // raise); without it, or under
                                // `--nocapture`, the field is
                                // omitted.
                                let captured = format!(
                                    "{cap_out}{cap_err}{}",
                                    pass_note.as_deref().unwrap_or("")
                                );
                                let stdout = if opts.show_output
                                    && !opts.nocapture
                                    && !captured.is_empty()
                                {
                                    format!(
                                        ", \"stdout\": \"{}\"",
                                        json_escape(&captured)
                                    )
                                } else {
                                    String::new()
                                };
                                self.push_output(&format!(
                                    "{{ \"type\": \"test\", \"name\": \"{}\", \
                                     \"event\": \"ok\"{} }}\n",
                                    json_escape(&test.name),
                                    stdout
                                ));
                            } else if opts.nocapture {
                                self.push_output("ok\n");
                            } else {
                                self.push_output(&format!(
                                    "test {}{} ... ok\n",
                                    test.name, annotation
                                ));
                            }
                        }
                        Some(body) => {
                            failed += 1;
                            failure_names.push(test.name.clone());
                            if let Some(body) = &body {
                                failure_sections.push(format!(
                                    "\n---- {} stdout ----\n{}",
                                    test.name, body
                                ));
                            }
                            if opts.json {
                                // A failed test's event carries
                                // its failure account as `stdout`
                                // (the captured output followed by
                                // the account itself); under
                                // `--nocapture` nothing was
                                // captured, so the field is omitted.
                                let stdout = if opts.nocapture {
                                    String::new()
                                } else {
                                    format!(
                                        ", \"stdout\": \"{}\"",
                                        json_escape(
                                            body.as_deref().unwrap_or("")
                                        )
                                    )
                                };
                                self.push_output(&format!(
                                    "{{ \"type\": \"test\", \"name\": \"{}\", \
                                     \"event\": \"failed\"{} }}\n",
                                    json_escape(&test.name),
                                    stdout
                                ));
                            } else if opts.nocapture {
                                self.push_output("FAILED\n");
                            } else {
                                self.push_output(&format!(
                                    "test {}{} ... FAILED\n",
                                    test.name, annotation
                                ));
                            }
                        }
                    }
                }
                if opts.json {
                    let event = if failed == 0 { "ok" } else { "failed" };
                    self.push_output(&format!(
                        "{{ \"type\": \"suite\", \"event\": \"{}\", \
                         \"passed\": {}, \"failed\": {}, \"ignored\": {}, \
                         \"measured\": 0, \"filtered_out\": {}, \
                         \"exec_time\": \"0.00\" }}\n",
                        event, passed, failed, ignored, filtered_out
                    ));
                } else {
                    if !failure_names.is_empty() {
                        self.push_output("\nfailures:\n");
                        for section in &failure_sections {
                            self.push_output(section);
                        }
                        if !failure_sections.is_empty() {
                            self.push_output("\n");
                        }
                        self.push_output("\nfailures:\n");
                        for name in &failure_names {
                            self.push_output(&format!("    {name}\n"));
                        }
                    }
                    let verdict = if failed == 0 { "ok" } else { "FAILED" };
                    self.push_output(&format!(
                        "\ntest result: {verdict}. {passed} passed; {failed} failed; \
                         {ignored} ignored; 0 measured; {filtered_out} filtered out; \
                         finished in 0.00s\n\n"
                    ));
                }
                // `ExitCode`: success is 0, any failure is 101,
                // as in libtest.
                let byte = if failed == 0 { 0 } else { 101 };
                Ok(test_exit_code(byte))
            }
            Builtin::PanicSetHook => {
                arity(1)?;
                self.set_panic_hook(vals.into_iter().next().expect("one argument"));
                Ok(Value::Unit)
            }
            Builtin::PanicTakeHook => {
                arity(0)?;
                Ok(self.take_panic_hook())
            }
            Builtin::CallerLocation => {
                // A `#[track_caller]` function's location is the
                // one it received (bound to its synthetic local);
                // any other function's is the stamp appended to
                // this call.
                if let Some(idx) = body.caller_location_local {
                    let frame = frames.last().expect("frame");
                    let received = frame.locals.get(idx).cloned().unwrap_or(Value::Uninit);
                    if !matches!(received, Value::Uninit) {
                        return Ok(received);
                    }
                }
                arity(1)?;
                Ok(vals.into_iter().next().expect("one argument"))
            }
            Builtin::InstantNow => {
                arity(0)?;
                Ok(instant_value(monotonic_now()))
            }
            Builtin::SystemTimeNow => {
                arity(0)?;
                Ok(instant_value(wall_now()))
            }
            Builtin::HashmapRandomKeys => {
                arity(0)?;
                let (k0, k1) = fresh_hash_keys();
                Ok(Value::Tuple(vec![
                    Value::Uint(u128::from(k0)),
                    Value::Uint(u128::from(k1)),
                ]))
            }
            Builtin::ProcessCommandNew => {
                arity(1)?;
                let program = self.os_str_bytes(frames, &vals[0])?;
                Ok(id_handle(self.new_command(command_state_new(program))))
            }
            Builtin::ProcessCommandArg => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let arg = self.os_str_bytes(frames, &vals[1])?;
                self.with_command(id, |state| {
                    state.saw_nul |= arg.contains(&0);
                    state.args.push(arg);
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(Value::Unit)
            }
            Builtin::ProcessCommandCwd => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let dir = self.os_str_bytes(frames, &vals[1])?;
                self.with_command(id, |state| {
                    state.saw_nul |= dir.contains(&0);
                    state.cwd = Some(dir);
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(Value::Unit)
            }
            Builtin::ProcessCommandEnvMut => {
                arity(1)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                // The environment editor is a handle value
                // carrying the command's id, in a reference the
                // `CommandEnv` builtins resolve back.
                Ok(Value::Ref(RefTarget::value(id_handle(id))))
            }
            Builtin::ProcessCommandEnvSet => {
                arity(3)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let key = self.os_str_bytes(frames, &vals[1])?;
                let val = self.os_str_bytes(frames, &vals[2])?;
                self.with_command(id, |state| {
                    state.saw_nul |= key.contains(&0) || val.contains(&0);
                    state.envs.push((key, Some(val)));
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(Value::Unit)
            }
            Builtin::ProcessCommandEnvRemove => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let key = self.os_str_bytes(frames, &vals[1])?;
                self.with_command(id, |state| {
                    state.envs.push((key, None));
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(Value::Unit)
            }
            Builtin::ProcessCommandEnvClear => {
                arity(1)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                self.with_command(id, |state| {
                    state.env_clear = true;
                    state.envs.clear();
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(Value::Unit)
            }
            Builtin::ProcessCommandStdio => {
                arity(2)?;
                // The receiver must still denote a live command;
                // the configuration itself is not observed.
                let _ = self.command_id_of(frames, &vals[0], name)?;
                Ok(Value::Unit)
            }
            Builtin::ProcessSpawn => {
                // The public entry points (`Command::spawn`,
                // `Command::status`) pass only the receiver; the
                // sys entry point also passes the default stdio
                // and a flag. The outcome is the same.
                if vals.len() != 1 && vals.len() != 3 {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "1 or 3 argument(s)",
                        format!("{} argument(s)", vals.len()),
                    ));
                }
                let id = self.command_id_of(frames, &vals[0], name)?;
                let state = self
                    .command_snapshot(id)
                    .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                match process_spawn_error(&state) {
                    Some(error) => Ok(err_result(error)),
                    None => {
                        // `Command::status` on a configured tool
                        // program runs the tool and reports its
                        // exit status. A `Child` handle for a
                        // spawned tool cannot be fabricated, so
                        // `spawn` itself stays unsupported.
                        if name == "std::process::Command::status" {
                            if let Some(outcome) =
                                tool_outcome(self.tool_interception(), &state)?
                            {
                                return Ok(ok_result(exit_status_value(
                                    outcome.exit_code,
                                )));
                            }
                        }
                        Err(unsupported(format!(
                            "spawning program `{}` is not runtime-supported (the process model does not start child processes)",
                            String::from_utf8_lossy(&state.program)
                        )))
                    }
                }
            }
            Builtin::ProcessCommandNewPublic => {
                arity(1)?;
                let program = self.os_str_bytes(frames, &vals[0])?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![id_handle(self.new_command(command_state_new(program)))],
                })
            }
            Builtin::ProcessCommandArgPublic => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let arg = self.os_str_bytes(frames, &vals[1])?;
                self.with_command(id, |state| {
                    state.saw_nul |= arg.contains(&0);
                    state.args.push(arg);
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::ProcessCommandCwdPublic => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let dir = self.os_str_bytes(frames, &vals[1])?;
                self.with_command(id, |state| {
                    state.saw_nul |= dir.contains(&0);
                    state.cwd = Some(dir);
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::ProcessCommandEnvPublic => {
                arity(3)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let key = self.os_str_bytes(frames, &vals[1])?;
                let val = self.os_str_bytes(frames, &vals[2])?;
                self.with_command(id, |state| {
                    state.saw_nul |= key.contains(&0) || val.contains(&0);
                    state.envs.push((key, Some(val)));
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::ProcessCommandEnvRemovePublic => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let key = self.os_str_bytes(frames, &vals[1])?;
                self.with_command(id, |state| {
                    state.envs.push((key, None));
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::ProcessCommandEnvClearPublic => {
                arity(1)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                self.with_command(id, |state| {
                    state.env_clear = true;
                    state.envs.clear();
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::ProcessCommandStdioPublic => {
                arity(2)?;
                let _ = self.command_id_of(frames, &vals[0], name)?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::ProcessOutput => {
                arity(1)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let state = self
                    .command_snapshot(id)
                    .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                match process_spawn_error(&state) {
                    Some(error) => Ok(err_result(error)),
                    None => match tool_outcome(self.tool_interception(), &state)? {
                        Some(outcome) => {
                            Ok(ok_result(process_output_value(&outcome)))
                        }
                        None => Err(unsupported(format!(
                            "running program `{}` to collect its output is not runtime-supported (the process model does not start child processes)",
                            String::from_utf8_lossy(&state.program)
                        ))),
                    },
                }
            }
            Builtin::CurrentExe => {
                arity(0)?;
                let path = std::env::current_exe().map_err(|e| {
                    unsupported(format!("current_exe is not available in this run: {e}"))
                })?;
                Ok(ok_result(path_buf_value(&path)))
            }
            Builtin::ProcessId => {
                arity(0)?;
                Ok(Value::Uint(u128::from(std::process::id())))
            }
            Builtin::IoGetOsFunctions => {
                arity(0)?;
                Ok(Value::Ref(RefTarget::value(os_functions_table())))
            }
            Builtin::IoSetOsFunctions => {
                arity(1)?;
                Ok(Value::Unit)
            }
            Builtin::IoDecodeErrorKind => {
                arity(1)?;
                let code = match vals[0] {
                    Value::Int(v) => v as i32,
                    Value::Uint(v) => v as i32,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "an errno integer",
                            other.kind_name(),
                        ))
                    }
                };
                Ok(Value::Enum {
                    variant: decode_error_kind_unix(code),
                    fields: Vec::new(),
                })
            }
            Builtin::IoFormatOsError => {
                arity(2)?;
                let code = match vals[0] {
                    Value::Int(v) => v as i32,
                    Value::Uint(v) => v as i32,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "an errno integer",
                            other.kind_name(),
                        ))
                    }
                };
                let id = self.formatter_id_of(frames, &vals[1])?;
                self.formatter_push(id, &os_error_message(code));
                Ok(fmt_result_ok())
            }
            Builtin::IoIsInterrupted => {
                arity(1)?;
                let code = match vals[0] {
                    Value::Int(v) => v as i32,
                    Value::Uint(v) => v as i32,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "an errno integer",
                            other.kind_name(),
                        ))
                    }
                };
                Ok(Value::Bool(code == 4))
            }
            Builtin::TimeCheckedAdd(system) => {
                arity(2)?;
                let base = time_pair(&self.resolve_value(frames, &vals[0])?).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[0].kind_name())
                })?;
                let dur = duration_pair(&vals[1]).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a Duration value", vals[1].kind_name())
                })?;
                Ok(match checked_time_add(base, dur, system) {
                    Some(pair) => some_value(instant_value(pair)),
                    None => none_value(),
                })
            }
            Builtin::TimeCheckedSub(system) => {
                arity(2)?;
                let base = time_pair(&self.resolve_value(frames, &vals[0])?).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[0].kind_name())
                })?;
                let dur = duration_pair(&vals[1]).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a Duration value", vals[1].kind_name())
                })?;
                Ok(match checked_time_sub(base, dur, system) {
                    Some(pair) => some_value(instant_value(pair)),
                    None => none_value(),
                })
            }
            Builtin::InstantDurationSince => {
                arity(2)?;
                let a = time_pair(&self.resolve_value(frames, &vals[0])?).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[0].kind_name())
                })?;
                let b = time_pair(&vals[1]).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[1].kind_name())
                })?;
                match checked_time_sub(a, b, false) {
                    Some(pair) => self.duration_value(pair.0 as u64, pair.1 as u64),
                    None => Err(InterpError::panic_str_payload(
                        "supplied instant is later than self",
                    )),
                }
            }
            Builtin::InstantCheckedDurationSince => {
                arity(2)?;
                let a = time_pair(&self.resolve_value(frames, &vals[0])?).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[0].kind_name())
                })?;
                let b = time_pair(&vals[1]).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[1].kind_name())
                })?;
                match checked_time_sub(a, b, false) {
                    Some(pair) => Ok(some_value(
                        self.duration_value(pair.0 as u64, pair.1 as u64)?,
                    )),
                    None => Ok(none_value()),
                }
            }
            Builtin::InstantSaturatingDurationSince => {
                arity(2)?;
                let a = time_pair(&self.resolve_value(frames, &vals[0])?).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[0].kind_name())
                })?;
                let b = time_pair(&vals[1]).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[1].kind_name())
                })?;
                match checked_time_sub(a, b, false) {
                    Some(pair) => self.duration_value(pair.0 as u64, pair.1 as u64),
                    None => self.duration_value(0, 0),
                }
            }
            Builtin::InstantElapsed => {
                arity(1)?;
                let a = time_pair(&self.resolve_value(frames, &vals[0])?).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[0].kind_name())
                })?;
                match checked_time_sub(monotonic_now(), a, false) {
                    Some(pair) => self.duration_value(pair.0 as u64, pair.1 as u64),
                    None => Err(InterpError::panic_str_payload(
                        "supplied instant is later than self",
                    )),
                }
            }
            Builtin::SystemTimeDurationSince => {
                arity(2)?;
                let a = time_pair(&self.resolve_value(frames, &vals[0])?).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[0].kind_name())
                })?;
                let b = time_pair(&vals[1]).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[1].kind_name())
                })?;
                match checked_time_sub(a, b, true) {
                    Some(pair) => Ok(Value::Enum {
                        variant: 0,
                        fields: vec![self.duration_value(pair.0 as u64, pair.1 as u64)?],
                    }),
                    None => {
                        // `Err(SystemTimeError(duration))`: the
                        // error wraps the positive difference.
                        let back = checked_time_sub(b, a, true)
                            .expect("one direction is non-negative");
                        Ok(Value::Enum {
                            variant: 1,
                            fields: vec![Value::Enum {
                                variant: 0,
                                fields: vec![
                                    self.duration_value(back.0 as u64, back.1 as u64)?,
                                ],
                            }],
                        })
                    }
                }
            }
            Builtin::SystemTimeElapsed => {
                arity(1)?;
                let a = time_pair(&self.resolve_value(frames, &vals[0])?).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[0].kind_name())
                })?;
                match checked_time_sub(wall_now(), a, true) {
                    Some(pair) => Ok(Value::Enum {
                        variant: 0,
                        fields: vec![self.duration_value(pair.0 as u64, pair.1 as u64)?],
                    }),
                    None => {
                        let back = checked_time_sub(a, wall_now(), true)
                            .expect("one direction is non-negative");
                        Ok(Value::Enum {
                            variant: 1,
                            fields: vec![Value::Enum {
                                variant: 0,
                                fields: vec![
                                    self.duration_value(back.0 as u64, back.1 as u64)?,
                                ],
                            }],
                        })
                    }
                }
            }
            Builtin::MpscChannel => {
                arity(0)?;
                let id = self.new_channel(None);
                Ok(Value::Tuple(vec![id_handle(id), id_handle(id)]))
            }
            Builtin::MpscSyncChannel => {
                arity(1)?;
                let bound = match &vals[0] {
                    Value::Uint(n) => *n as usize,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "sync_channel",
                            "a bound",
                            other.kind_name(),
                        ))
                    }
                };
                let id = self.new_channel(Some(bound));
                Ok(Value::Tuple(vec![id_handle(id), id_handle(id)]))
            }
            Builtin::MpscSend => {
                arity(2)?;
                let id = self.handle_id_of_ref(frames, &vals[0], name)?;
                let value = vals.into_iter().nth(1).expect("arity checked");
                self.mpsc_send(frames, id, value, name)
            }
            Builtin::MpscTrySend => {
                arity(2)?;
                let id = self.handle_id_of_ref(frames, &vals[0], name)?;
                let value = vals.into_iter().nth(1).expect("arity checked");
                match self.with_channel(id, |st| {
                    if st.receivers == 0 {
                        Err(1) // TrySendError::Disconnected(value)
                    } else if st.bound.is_some_and(|b| st.queue.len() >= b) {
                        Err(0) // TrySendError::Full(value)
                    } else {
                        st.queue.push_back(value.clone());
                        Ok(())
                    }
                }) {
                    Some(Ok(())) => Ok(ok_result(Value::Unit)),
                    Some(Err(variant)) => Ok(err_result(Value::Enum {
                        variant,
                        fields: vec![value],
                    })),
                    None => Err(unsupported(format!("{name} on an unknown channel"))),
                }
            }
            Builtin::MpscRecv => {
                arity(1)?;
                let id = self.handle_id_of_ref(frames, &vals[0], name)?;
                match self.mpsc_recv(frames, id, name)? {
                    Ok(value) => Ok(ok_result(value)),
                    // `RecvError` is a fieldless one-variant struct.
                    Err(()) => Ok(err_result(Value::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    })),
                }
            }
            Builtin::MpscTryRecv => {
                arity(1)?;
                let id = self.handle_id_of_ref(frames, &vals[0], name)?;
                match self.with_channel(id, |st| match st.queue.pop_front() {
                    Some(v) => Ok(v),
                    None => Err(if st.senders == 0 { 1 } else { 0 }),
                }) {
                    Some(Ok(value)) => Ok(ok_result(value)),
                    // `TryRecvError`: Empty = 0, Disconnected = 1,
                    // both fieldless.
                    Some(Err(variant)) => Ok(err_result(Value::Enum {
                        variant,
                        fields: Vec::new(),
                    })),
                    None => Err(unsupported(format!("{name} on an unknown channel"))),
                }
            }
            Builtin::MpscRecvTimeout => {
                arity(2)?;
                let id = self.handle_id_of_ref(frames, &vals[0], name)?;
                // The model cannot wait out a duration: a value
                // that no pending thread produces is a timeout,
                // exactly what a native run whose producers have
                // all finished would observe.
                match self.mpsc_recv_or(frames, id, name)? {
                    Ok(value) => Ok(ok_result(value)),
                    Err(variant) => Ok(err_result(Value::Enum {
                        variant,
                        fields: Vec::new(),
                    })),
                }
            }
            Builtin::MpscIter => {
                arity(1)?;
                let id = self.handle_id_of_ref(frames, &vals[0], name)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![id_handle(id)],
                })
            }
            Builtin::IntoIter => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                let value = vals.into_iter().next().expect("arity checked");
                if self_ty.starts_with("std::ops::Range<")
                    || self_ty.starts_with("std::ops::RangeInclusive<")
                    || self_ty.starts_with("std::iter::Rev<")
                    || self_ty.starts_with("std::slice::Iter<")
                    || self_ty.starts_with("std::slice::IterMut<")
                    || self_ty.starts_with("std::array::IntoIter<")
                    || self_ty.starts_with("std::vec::IntoIter<")
                    || self_ty.starts_with("alloc::vec::IntoIter<")
                    || self_ty.starts_with("std::option::IntoIter<")
                    || self_ty.starts_with("std::option::Iter<")
                    || self_ty.starts_with("std::option::IterMut<")
                    || self_ty.starts_with("std::iter::Map<")
                    || self_ty.starts_with("std::iter::Enumerate<")
                    || self_ty.starts_with("std::iter::Copied<")
                    || self_ty.starts_with("std::iter::Take<")
                    || self_ty.starts_with("std::iter::Skip<")
                    || self_ty.starts_with("std::iter::Chain<")
                    || self_ty.starts_with("std::iter::SkipWhile<")
                    || self_ty.starts_with("std::iter::Once<")
                    || self_ty.starts_with("std::path::Components")
                    || self_ty.starts_with("std::path::Ancestors")
                    || self_ty_is(self_ty, "std::fs::ReadDir")
                    || self_ty.starts_with("std::iter::Empty<")
                    || self_ty.starts_with("std::iter::Repeat<")
                    || self_ty_is(self_ty, "std::str::Chars")
                    || self_ty_is(self_ty, "core::str::Chars")
                    || self_ty_is(self_ty, "std::str::Bytes")
                    || self_ty_is(self_ty, "core::str::Bytes")
                {
                    // These types are their own iterator: the
                    // blanket `IntoIterator for I: Iterator` impl
                    // is the identity, and the value is already in
                    // the shape `Iterator::next` consumes for the
                    // type — a range is its own state, a slice /
                    // char / byte iterator was built by the
                    // corresponding runtime support, an adapter or
                    // source iterator by its constructor's
                    // converted body, and the owning iterators by
                    // the arms below.
                    Ok(value)
                } else if self_ty.starts_with('[') {
                    // Array by value: iteration state is the element
                    // tuple plus the next index (our layout for
                    // `std::array::IntoIter`, whose `next` below is
                    // the only consumer).
                    match value {
                        Value::Tuple(items) => Ok(Value::Enum {
                            variant: 0,
                            fields: vec![Value::Tuple(items), Value::Uint(0)],
                        }),
                        ref other => Err(InterpError::type_mismatch(
                            "IntoIterator::into_iter for an array",
                            "array value",
                            other.kind_name(),
                        )),
                    }
                } else if self_ty.starts_with("&std::vec::Vec<")
                    || self_ty.starts_with("&alloc::vec::Vec<")
                    || self_ty.starts_with("&[")
                {
                    // `&Vec<T>` and `&[T]` iterate as `slice::Iter`,
                    // in the real field layout `slice_iter_value`
                    // builds and `Iterator::next` consumes.
                    let (elems, _) = self.slice_elements(frames, &value)?;
                    Ok(slice_iter_value(elems))
                } else if self_ty_is(self_ty, "std::env::Args")
                    || self_ty_is(self_ty, "std::env::ArgsOs")
                    || self_ty_is(self_ty, "std::env::SplitPaths")
                    || self_ty.starts_with("std::slice::Windows<")
                {
                    // The env argument iterators are their own
                    // `IntoIterator` (the state `env::args` built).
                    Ok(value)
                } else if self_ty.starts_with("std::vec::Vec<")
                    || self_ty.starts_with("alloc::vec::Vec<")
                {
                    // A by-value `Vec` iterates as `vec::IntoIter`:
                    // the element snapshot plus the next index (the
                    // vector value's fields are its elements).
                    match value {
                        Value::Enum { fields, .. } => Ok(owned_iter_state(fields)),
                        ref other => Err(InterpError::type_mismatch(
                            "IntoIterator::into_iter for a Vec",
                            "vector value",
                            other.kind_name(),
                        )),
                    }
                } else if self_ty.starts_with("std::sync::mpsc::Iter<") {
                    // `mpsc::Iter` is its own `IntoIterator`.
                    Ok(value)
                } else if self_ty.starts_with("std::sync::mpsc::Receiver<") {
                    // The channel iterator holds the receiver
                    // handle as its one field (see `iterator_next`).
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![value],
                    })
                } else if self_ty.starts_with("std::option::Option<")
                    || self_ty.starts_with("core::option::Option<")
                {
                    // An `Option<T>` by value iterates as
                    // `option::IntoIter` over zero or one element,
                    // in the same owning-iterator state shape.
                    let elems = match &value {
                        Value::Enum {
                            variant: 1,
                            fields,
                        } => fields.clone(),
                        Value::Enum { variant: 0, .. } => Vec::new(),
                        other => {
                            return Err(InterpError::type_mismatch(
                                "IntoIterator::into_iter for an Option",
                                "option value",
                                other.kind_name(),
                            ))
                        }
                    };
                    Ok(owned_iter_state(elems))
                } else if let Some(inner) = self_ty.strip_prefix("&mut ") {
                    if inner.starts_with("std::vec::Vec<")
                        || inner.starts_with("alloc::vec::Vec<")
                        || inner.starts_with('[')
                    {
                        // `&mut Vec<T>`, `&mut [T; N]` and `&mut [T]`
                        // iterate as `slice::IterMut`: the state is
                        // the container reference, the next index
                        // and the length. The elements stay in the
                        // container's own storage; each step hands
                        // out a reference to the element place (see
                        // `Iterator::next` for `slice::IterMut`).
                        let len = match &value {
                            Value::FatRef { len, .. } => *len,
                            _ => {
                                let target = self.deref_value(frames, &value)?;
                                let pointee = self.read_resolved(frames, &target)?;
                                value_elements(&pointee)?.len() as u128
                            }
                        };
                        Ok(Value::Enum {
                            variant: 0,
                            fields: vec![value, Value::Uint(0), Value::Uint(len)],
                        })
                    } else if inner.starts_with("std::option::Option<")
                        || inner.starts_with("core::option::Option<")
                    {
                        self.option_ref_into_iter(frames, &value, true)
                    } else {
                        Err(unsupported(format!(
                            "IntoIterator::into_iter for {self_ty} is not runtime-supported"
                        )))
                    }
                } else if self_ty.starts_with("&std::option::Option<")
                    || self_ty.starts_with("&core::option::Option<")
                {
                    self.option_ref_into_iter(frames, &value, false)
                } else {
                    Err(unsupported(format!(
                        "IntoIterator::into_iter for {self_ty} is not runtime-supported"
                    )))
                }
            }
            Builtin::IteratorNext => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                self.iterator_next(frames, self_ty, &vals[0])
            }
            Builtin::IteratorNextBack => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                self.iterator_next_back(frames, self_ty, &vals[0])
            }
            Builtin::IteratorSizeHint => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                self.iterator_size_hint(frames, self_ty, &vals[0])
            }
            Builtin::IteratorFold | Builtin::IteratorForEach => {
                unreachable!("fold-like builtins are dispatched before argument evaluation")
            }
            Builtin::IteratorCount => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                let mut state = vals.into_iter().next().expect("arity checked");
                let mut count: u128 = 0;
                loop {
                    self.tick_step()?;
                    let (next_state, item) =
                        self.iterator_next_state(frames, self_ty, state)?;
                    state = next_state;
                    match option_payload(item)? {
                        None => break,
                        Some(_) => count += 1,
                    }
                }
                Ok(Value::Uint(count))
            }
            Builtin::IteratorCollect => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                let target = generic_args.get(1).map(String::as_str).unwrap_or("");
                let state = vals.into_iter().next().expect("arity checked");
                self.exec_collect(frames, self_ty, target, state)
            }
            Builtin::FromIteratorFromIter => {
                arity(1)?;
                let target = generic_args.first().map(String::as_str).unwrap_or("");
                let self_ty = generic_args.get(2).map(String::as_str).unwrap_or("");
                let state = vals.into_iter().next().expect("arity checked");
                self.exec_collect(frames, self_ty, target, state)
            }
            Builtin::PrimitiveBinOp(op) => {
                arity(2)?;
                let mut vals = vals.into_iter();
                let lhs = vals.next().expect("arity checked");
                let rhs = vals.next().expect("arity checked");
                self.exec_primitive_binop(name, op, &lhs, &rhs)
            }
            Builtin::PrimitiveUnOp(op) => {
                arity(1)?;
                let value = vals.into_iter().next().expect("arity checked");
                self.exec_primitive_unop(name, op, &value)
            }
            Builtin::EnvCurrentExe => {
                arity(0)?;
                Ok(match std::env::current_exe() {
                    Ok(path) => io_result_ok(pathbuf_value(&host_path_bytes(&path))),
                    Err(e) => io_result_err(host_io_error(&e)),
                })
            }
            Builtin::FsOpen => {
                arity(2)?;
                let mut vals = vals.into_iter();
                let _options = vals.next().expect("arity checked");
                let path = vals.next().expect("arity checked");
                let bytes = self.path_bytes(frames, &path)?;
                if bytes.contains(&0) {
                    return Ok(io_result_err(io_simple_error_value(ERROR_KIND_INVALID_INPUT)));
                }
                Err(unsupported(
                    "opening a real file: the File value model is not built yet"
                        .to_string(),
                ))
            }
            Builtin::FsDirCreate => {
                arity(2)?;
                let mut vals = vals.into_iter();
                let builder = vals.next().expect("arity checked");
                let path = vals.next().expect("arity checked");
                let bytes = self.path_bytes(frames, &path)?;
                if bytes.contains(&0) {
                    return Ok(io_result_err(io_simple_error_value(ERROR_KIND_INVALID_INPUT)));
                }
                // The builder's `recursive` flag (its second
                // field) selects between std's two operations:
                // with it, `_create` builds the whole ancestor
                // chain; without it, one directory. The chain is
                // built on the host like the rest of the fs layer:
                // the converted `create_dir_all` body is never
                // reached (this builtin serves the `_create`
                // call), and its `Ancestors` iteration does not
                // carry its state faithfully in this model.
                let recursive = match self.resolve_value(frames, &builder) {
                    Ok(Value::Enum { fields, .. }) if fields.len() >= 2 => {
                        matches!(fields[1], Value::Bool(true))
                    }
                    _ => false,
                };
                Ok(io_result_from_host(if recursive {
                    std::fs::create_dir_all(host_path(&bytes))
                        .map(|()| Value::Unit)
                } else {
                    std::fs::create_dir(host_path(&bytes)).map(|()| Value::Unit)
                }))
            }
            Builtin::EnvJoinPaths => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                let value = vals.into_iter().next().expect("arity checked");
                // The argument is an `IntoIterator`; the forms
                // whose conversion is not the identity are the
                // by-value and by-reference containers, converted
                // here to their iterator states (the same states
                // the `IntoIterator::into_iter` builtin builds).
                let (state, iter_ty): (Value, String) = if iterator_next_modelled(self_ty) {
                    (value, self_ty.to_string())
                } else if self_ty.starts_with("std::vec::Vec<")
                    || self_ty.starts_with("alloc::vec::Vec<")
                {
                    match value {
                        Value::Enum { fields, .. } => (
                            owned_iter_state(fields),
                            self_ty.replacen("Vec<", "IntoIter<", 1),
                        ),
                        other => {
                            return Err(InterpError::type_mismatch(
                                "join_paths argument",
                                "vector value",
                                other.kind_name(),
                            ))
                        }
                    }
                } else if self_ty.starts_with("&std::vec::Vec<")
                    || self_ty.starts_with("&alloc::vec::Vec<")
                    || self_ty.starts_with("&[")
                {
                    let (elems, _) = self.slice_elements(frames, &value)?;
                    (slice_iter_value(elems), "std::slice::Iter<>".to_string())
                } else {
                    return Err(unsupported(format!(
                        "join_paths over {self_ty} is not runtime-supported"
                    )));
                };
                let mut state = state;
                let mut joined: Vec<u8> = Vec::new();
                let mut first = true;
                loop {
                    self.tick_step()?;
                    let (next_state, item) =
                        self.iterator_next_state(frames, &iter_ty, state)?;
                    state = next_state;
                    let Some(item) = option_payload(item)? else {
                        break;
                    };
                    let bytes = self.os_str_bytes(frames, &item)?;
                    if bytes.contains(&b':') {
                        // `JoinPathsError` is a zero-sized
                        // struct: the empty tuple shape.
                        return Ok(err_result(Value::Tuple(Vec::new())));
                    }
                    if !first {
                        joined.push(b':');
                    }
                    first = false;
                    joined.extend_from_slice(&bytes);
                }
                Ok(ok_result(os_string_value(&joined)))
            }
            Builtin::EnvSplitPaths => {
                arity(1)?;
                let bytes = self.os_str_bytes(frames, &vals[0])?;
                let parts: Vec<Value> = bytes
                    .split(|b| *b == b':')
                    .map(pathbuf_value)
                    .collect();
                Ok(owned_iter_state(parts))
            }
            Builtin::EnvVarInner => {
                arity(1)?;
                let key = self.os_str_bytes(frames, &vals[0])?;
                use std::os::unix::ffi::OsStrExt as _;
                let value = if key.contains(&0) {
                    None
                } else {
                    std::env::var_os(std::ffi::OsStr::from_bytes(&key))
                };
                Ok(match value {
                    None => err_result(Value::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    }),
                    Some(v) => {
                        let bytes = v.as_bytes();
                        match std::str::from_utf8(bytes) {
                            Ok(s) => ok_result(Value::Str(s.to_string())),
                            Err(_) => err_result(Value::Enum {
                                variant: 1,
                                fields: vec![os_string_value(bytes)],
                            }),
                        }
                    }
                })
            }
            Builtin::EnvGetenv => {
                arity(1)?;
                let key = self.os_str_bytes(frames, &vals[0])?;
                if key.contains(&0) {
                    return Ok(none_value());
                }
                use std::os::unix::ffi::OsStrExt as _;
                let value = std::env::var_os(std::ffi::OsStr::from_bytes(&key));
                Ok(match value {
                    Some(v) => some_value(os_string_value(v.as_bytes())),
                    None => none_value(),
                })
            }
            Builtin::EnvSetenv => {
                arity(2)?;
                let key = self.os_str_bytes(frames, &vals[0])?;
                let value = self.os_str_bytes(frames, &vals[1])?;
                if key.is_empty()
                    || key.contains(&b'=')
                    || key.contains(&0)
                    || value.contains(&0)
                {
                    return Ok(io_result_err(io_simple_error_value(20)));
                }
                use std::os::unix::ffi::OsStrExt as _;
                std::env::set_var(
                    std::ffi::OsStr::from_bytes(&key),
                    std::ffi::OsStr::from_bytes(&value),
                );
                Ok(io_result_ok(Value::Unit))
            }
            Builtin::EnvUnsetenv => {
                arity(1)?;
                let key = self.os_str_bytes(frames, &vals[0])?;
                if key.is_empty() || key.contains(&b'=') || key.contains(&0) {
                    return Ok(io_result_err(io_simple_error_value(20)));
                }
                use std::os::unix::ffi::OsStrExt as _;
                std::env::remove_var(std::ffi::OsStr::from_bytes(&key));
                Ok(io_result_ok(Value::Unit))
            }
            Builtin::EnvCurrentDir => {
                arity(0)?;
                Ok(io_result_from_host(
                    std::env::current_dir()
                        .map(|dir| path_buf_value(&dir)),
                ))
            }
            Builtin::EnvTempDir => {
                arity(0)?;
                Ok(path_buf_value(&std::env::temp_dir()))
            }
            Builtin::EnvHomeDir => {
                arity(0)?;
                Ok(match std::env::home_dir() {
                    Some(dir) => {
                        some_value(path_buf_value(&dir))
                    }
                    None => none_value(),
                })
            }
            Builtin::SysChdir => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                if bytes.contains(&0) {
                    return Ok(io_result_err(io_simple_error_value(20)));
                }
                Ok(io_result_from_host(
                    std::env::set_current_dir(host_path(&bytes))
                        .map(|()| Value::Unit),
                ))
            }
            Builtin::StringFromUtf8 => {
                arity(1)?;
                let resolved = self.resolve_value(frames, &vals[0])?;
                let bytes = match &resolved {
                    Value::Enum { fields, .. } => fields
                        .iter()
                        .map(|f| match f {
                            Value::Uint(v) => Ok(*v as u8),
                            Value::Int(v) => Ok(*v as u8),
                            other => Err(InterpError::type_mismatch(
                                "String::from_utf8 argument",
                                "byte elements",
                                other.kind_name(),
                            )),
                        })
                        .collect::<InterpResult<Vec<u8>>>()?,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "String::from_utf8 argument",
                            "byte vector value",
                            other.kind_name(),
                        ))
                    }
                };
                Ok(match String::from_utf8(bytes.clone()) {
                    Ok(text) => ok_result(Value::Str(text)),
                    Err(e) => {
                        let ue = e.utf8_error();
                        let error_len = match ue.error_len() {
                            Some(n) => Value::Enum {
                                variant: 1,
                                fields: vec![Value::Uint(n as u128)],
                            },
                            None => Value::Enum {
                                variant: 0,
                                fields: Vec::new(),
                            },
                        };
                        let utf8_error = Value::Tuple(vec![
                            Value::Uint(ue.valid_up_to() as u128),
                            error_len,
                        ]);
                        err_result(Value::Enum {
                            variant: 0,
                            fields: vec![byte_vec_value(&bytes), utf8_error],
                        })
                    }
                })
            }
            Builtin::FsReadToStringInner => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                if bytes.contains(&0) {
                    return Ok(io_result_err(io_simple_error_value(20)));
                }
                Ok(match std::fs::read(host_path(&bytes)) {
                    Ok(data) => match String::from_utf8(data) {
                        Ok(text) => io_result_ok(Value::Str(text)),
                        Err(_) => io_result_err(io_simple_error_value(std::io::ErrorKind::InvalidData as u32)),
                    },
                    Err(e) => io_result_from_host(Err::<Value, _>(e)),
                })
            }
            Builtin::FsReadInner => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                if bytes.contains(&0) {
                    return Ok(io_result_err(io_simple_error_value(20)));
                }
                Ok(io_result_from_host(
                    std::fs::read(host_path(&bytes)).map(|data| byte_vec_value(&data)),
                ))
            }
            Builtin::FsWriteInner => {
                arity(2)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                if bytes.contains(&0) {
                    return Ok(io_result_err(io_simple_error_value(20)));
                }
                let contents = self.arg_bytes(frames, &vals[1])?;
                Ok(io_result_from_host(
                    std::fs::write(host_path(&bytes), &contents)
                        .map(|()| Value::Unit),
                ))
            }
            Builtin::PathToPathBuf => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                Ok(pathbuf_value(&bytes))
            }
            Builtin::PathIsAbsolute => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                Ok(Value::Bool(bytes.first() == Some(&b'/')))
            }
            Builtin::PathJoin => {
                arity(2)?;
                let base = self.path_bytes(frames, &vals[0])?;
                let path = self.path_bytes(frames, &vals[1])?;
                Ok(pathbuf_value(&path_push_bytes(&base, &path)))
            }
            Builtin::PathParent => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                Ok(match path_parent_len(&bytes) {
                    Some(end) => some_value(sub_path_value(&bytes[..end])),
                    None => none_value(),
                })
            }
            Builtin::PathFileName => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                Ok(match path_file_name_range(&bytes) {
                    Some((start, end)) => {
                        some_value(sub_path_value(&bytes[start..end]))
                    }
                    None => none_value(),
                })
            }
            Builtin::PathFileStem => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                Ok(match path_file_name_range(&bytes) {
                    Some((start, end)) => {
                        let (before, after) =
                            rsplit_file_at_dot(&bytes[start..end]);
                        let stem = before
                            .or(after)
                            .expect("a file name always has a stem side");
                        some_value(sub_path_value(stem))
                    }
                    None => none_value(),
                })
            }
            Builtin::PathExtension => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                Ok(match path_file_name_range(&bytes) {
                    Some((start, end)) => {
                        let (before, after) =
                            rsplit_file_at_dot(&bytes[start..end]);
                        match before.and(after) {
                            Some(ext) => some_value(sub_path_value(ext)),
                            None => none_value(),
                        }
                    }
                    None => none_value(),
                })
            }
            Builtin::StringFromUtf8Lossy => {
                arity(1)?;
                let bytes = self.arg_bytes(frames, &vals[0])?;
                Ok(cow_str_value(&bytes))
            }
            Builtin::StrFromUtf8 => {
                arity(1)?;
                let bytes = self.arg_bytes(frames, &vals[0])?;
                Ok(match std::str::from_utf8(&bytes) {
                    Ok(s) => ok_result(Value::Str(s.to_string())),
                    Err(e) => err_result(utf8_error_value(&e)),
                })
            }
            Builtin::LibcStrlen => {
                arity(1)?;
                let bytes = self.read_c_str(frames, &vals[0])?;
                Ok(Value::Uint(bytes.len() as u128))
            }
            Builtin::CStrFromBytesWithNul => {
                arity(1)?;
                let bytes = self.arg_bytes(frames, &vals[0])?;
                Ok(match bytes.iter().position(|b| *b == 0) {
                    Some(position) if position + 1 == bytes.len() => {
                        ok_result(vals[0].clone())
                    }
                    Some(position) => err_result(Value::Enum {
                        variant: 0,
                        fields: vec![Value::Uint(position as u128)],
                    }),
                    None => err_result(Value::Enum {
                        variant: 1,
                        fields: Vec::new(),
                    }),
                })
            }
            Builtin::CStrFromBytesUntilNul => {
                arity(1)?;
                let bytes = self.arg_bytes(frames, &vals[0])?;
                Ok(match bytes.iter().position(|b| *b == 0) {
                    Some(position) => {
                        ok_result(self.fat_ref_prefix(frames, &vals[0], position + 1)?)
                    }
                    None => err_result(Value::Enum {
                        variant: 0,
                        fields: vec![Value::Unit],
                    }),
                })
            }
            Builtin::CStrToStr => {
                arity(1)?;
                let bytes = self.c_str_bytes(frames, &vals[0])?;
                Ok(match std::str::from_utf8(&bytes) {
                    Ok(s) => ok_result(Value::Str(s.to_string())),
                    Err(e) => err_result(utf8_error_value(&e)),
                })
            }
            Builtin::CStrToStringLossy => {
                arity(1)?;
                let bytes = self.c_str_bytes(frames, &vals[0])?;
                Ok(cow_str_value(&bytes))
            }
            Builtin::CStringSpecNewImpl => {
                arity(1)?;
                let bytes = self.arg_bytes(frames, &vals[0])?;
                Ok(match bytes.iter().position(|b| *b == 0) {
                    Some(i) => err_result(Value::Enum {
                        variant: 0,
                        fields: vec![
                            Value::Uint(i as u128),
                            byte_vec_value(&bytes),
                        ],
                    }),
                    None => ok_result(self.c_string_value(bytes)),
                })
            }
            Builtin::CStringFromVecUnchecked => {
                arity(1)?;
                let bytes = self.arg_bytes(frames, &vals[0])?;
                Ok(self.c_string_value(bytes))
            }
            Builtin::CStringIntoString => {
                arity(1)?;
                let receiver = Value::Ref(RefTarget::Value(
                    Box::new(vals[0].clone()),
                    Vec::new(),
                ));
                let cstr = self.c_string_deref(frames, &receiver)?;
                let bytes = self.c_str_bytes(frames, &cstr)?;
                Ok(match std::str::from_utf8(&bytes) {
                    Ok(s) => ok_result(Value::Str(s.to_string())),
                    Err(e) => err_result(Value::Enum {
                        variant: 0,
                        fields: vec![
                            self.c_string_value(bytes),
                            utf8_error_value(&e),
                        ],
                    }),
                })
            }
            Builtin::CStringFromRaw => {
                arity(1)?;
                // The real body recomputes the length with
                // the runtime `strlen` before reconstituting
                // the box; the read itself is the model's
                // check that the pointer denotes a C string.
                let _bytes = self.read_c_str(frames, &vals[0])?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Enum {
                        variant: 0,
                        fields: vec![vals[0].clone()],
                    }],
                })
            }
            Builtin::ToStringLossy => {
                arity(1)?;
                let bytes = if name == "std::ffi::OsString::to_string_lossy" {
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    os_string_bytes(&pointee).ok_or_else(|| {
                        InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "an OS string value",
                            pointee.kind_name(),
                        )
                    })?
                } else {
                    self.path_bytes(frames, &vals[0])?
                };
                Ok(cow_str_value(&bytes))
            }
            Builtin::DirEntryPath | Builtin::DirEntryFileName => {
                arity(1)?;
                let entry = self.resolve_value(frames, &vals[0])?;
                let (path, name) = dir_entry_parts(&entry).ok_or_else(|| {
                    InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "a directory entry value",
                        entry.kind_name(),
                    )
                })?;
                Ok(if matches!(kind, Builtin::DirEntryPath) {
                    path
                } else {
                    name
                })
            }
            Builtin::DirEntryMetadata => {
                arity(1)?;
                let entry = self.resolve_value(frames, &vals[0])?;
                let (path, _) = dir_entry_parts(&entry).ok_or_else(|| {
                    InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "a directory entry value",
                        entry.kind_name(),
                    )
                })?;
                let bytes = path_buf_bytes(&path).ok_or_else(|| {
                    InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "a directory entry holding a path buffer",
                        path.kind_name(),
                    )
                })?;
                if bytes.contains(&0) {
                    return Ok(io_result_err(io_simple_error_value(20)));
                }
                Ok(io_result_from_host(
                    std::fs::metadata(host_path(&bytes)).map(|md| {
                        // The public `Metadata` wraps the sys
                        // value `metadata_value` builds.
                        Value::Enum {
                            variant: 0,
                            fields: vec![metadata_value(&md)],
                        }
                    }),
                ))
            }
            Builtin::PathStartsWith => {
                arity(2)?;
                let path = self.path_bytes(frames, &vals[0])?;
                let child = self.path_bytes(frames, &vals[1])?;
                Ok(Value::Bool(path_starts_with(&path, &child)))
            }
            Builtin::PathEndsWith => {
                arity(2)?;
                let path = self.path_bytes(frames, &vals[0])?;
                let child = self.path_bytes(frames, &vals[1])?;
                Ok(Value::Bool(path_ends_with(&path, &child)))
            }
            Builtin::FsSys(op) => self.exec_fs_sys(frames, op, vals),
            Builtin::FromFrom | Builtin::IntoInto => {
                arity(1)?;
                // `From::from` renders the callee's generic
                // arguments as [target, source]; `Into::into`
                // renders [source, target] (its `Self` is the
                // source). The blanket `Into` is `From` by
                // definition, so resolve both to the same
                // (target, source) pair before dispatching on
                // them.
                let (target, source) = if matches!(kind, Builtin::IntoInto) {
                    (
                        generic_args.get(1).map(String::as_str).unwrap_or(""),
                        generic_args.first().map(String::as_str).unwrap_or(""),
                    )
                } else {
                    (
                        generic_args.first().map(String::as_str).unwrap_or(""),
                        generic_args.get(1).map(String::as_str).unwrap_or(""),
                    )
                };
                let value = vals.into_iter().next().expect("arity checked");
                // In this value model `String` and `str` data are the
                // same value; the conversion is the identity on it.
                if matches!(value, Value::Str(_))
                    && (target == "std::string::String" || target == "str" || target == "&str")
                {
                    Ok(value)
                } else if matches!(value, Value::Int(_) | Value::Uint(_)) && is_integer_tag(target)
                {
                    // The `From` impls between integer types are the
                    // value-preserving ones; the stored value already
                    // denotes the result (width/sign are applied when
                    // the destination place is written).
                    Ok(value)
                } else if target == "std::process::ExitCode"
                    && matches!(value, Value::Uint(_) | Value::Int(_))
                {
                    // `ExitCode::from(u8)`: the code byte inside
                    // the platform exit-code wrapper, two one-field
                    // aggregates in this model (the shape the
                    // driver's entry-return handling reads back).
                    let byte = match value {
                        Value::Uint(v) => v,
                        Value::Int(v) => v as u128,
                        _ => unreachable!(),
                    };
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![Value::Enum {
                            variant: 0,
                            fields: vec![Value::Uint(byte)],
                        }],
                    })
                } else if target == "run_make_support::CompletedProcess"
                    && source == "std::process::Output"
                {
                    // run-make-support's `CompletedProcess` is a
                    // one-field wrapper around the process
                    // `Output`; the conversion moves it in.
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![value],
                    })
                } else if target == "std::path::PathBuf"
                    && (source == "&str" || source == "str")
                {
                    // `PathBuf: From<&str>` copies the text into
                    // the path's buffer.
                    match self.resolve_value(frames, &value)? {
                        Value::Str(s) => Ok(pathbuf_value(s.as_bytes())),
                        other => Err(InterpError::type_mismatch(
                            "From::from to PathBuf",
                            "string value",
                            other.kind_name(),
                        )),
                    }
                } else if target == "std::ffi::OsString"
                    && (source == "&str" || source == "str")
                {
                    // `OsString: From<&str>` likewise.
                    match self.resolve_value(frames, &value)? {
                        Value::Str(s) => Ok(os_string_value(s.as_bytes())),
                        other => Err(InterpError::type_mismatch(
                            "From::from to OsString",
                            "string value",
                            other.kind_name(),
                        )),
                    }
                } else if target == "std::path::PathBuf" && source == "std::string::String" {
                    // `PathBuf: From<String>` moves the string's
                    // bytes into the path.
                    match &value {
                        Value::Str(s) => Ok(pathbuf_value(s.as_bytes())),
                        other => Err(InterpError::type_mismatch(
                            "From::from to PathBuf",
                            "string value",
                            other.kind_name(),
                        )),
                    }
                } else if target == "std::ffi::OsString" && source == "std::string::String" {
                    // `OsString: From<String>` likewise.
                    match &value {
                        Value::Str(s) => Ok(os_string_value(s.as_bytes())),
                        other => Err(InterpError::type_mismatch(
                            "From::from to OsString",
                            "string value",
                            other.kind_name(),
                        )),
                    }
                } else if target == "std::ffi::OsString"
                    && (source == "&std::ffi::OsStr" || source == "std::ffi::OsStr")
                {
                    // `OsString: From<&OsStr>` clones the byte
                    // string into an owned buffer.
                    let bytes = self.os_str_bytes(frames, &value)?;
                    Ok(os_string_value(&bytes))
                } else if target == "std::path::PathBuf"
                    && (source == "&std::path::Path" || source == "std::path::Path")
                {
                    // `PathBuf: From<&Path>` clones the path.
                    let bytes = self.path_bytes(frames, &value)?;
                    Ok(pathbuf_value(&bytes))
                } else if target.starts_with("std::vec::Vec<") {
                    // `Vec::from([T; N])`: the array's elements become
                    // the vector's.
                    match value {
                        Value::Tuple(items) => Ok(Value::Enum {
                            variant: 0,
                            fields: items,
                        }),
                        ref other => Err(InterpError::type_mismatch(
                            "From::from to Vec",
                            "array value",
                            other.kind_name(),
                        )),
                    }
                } else if target.starts_with("std::boxed::Box<") {
                    // `Box::from(value)` is `Box::new` by definition.
                    Ok(self.new_box(value))
                } else if target.starts_with("std::cell::Cell<") {
                    // `From<T> for Cell<T>` is `Cell::new` by
                    // definition (core::cell), so the result is the
                    // natural cell shape.
                    Ok(cell_value(value))
                } else if target.starts_with("std::cell::RefCell<") {
                    // `From<T> for RefCell<T>` is `RefCell::new` by
                    // definition (core::cell).
                    Ok(refcell_value(value))
                } else if (target == "std::task::Waker"
                    && (source.starts_with("std::sync::Arc<")
                        || source.starts_with("alloc::sync::Arc<")))
                    || (target == "std::task::LocalWaker"
                        && (source.starts_with("std::rc::Rc<")
                            || source.starts_with("alloc::rc::Rc<")))
                {
                    // `Waker::from(Arc<T: Wake>)` (and the
                    // `LocalWaker` impl, which takes `Rc<T: LocalWake>`):
                    // std builds a `RawWaker` whose data pointer is
                    // the smart pointer's allocation and whose vtable
                    // forwards wake/clone/drop to that allocation.
                    // In this value model the waker is the source
                    // value carried in a one-field aggregate: the
                    // data the vtable would act on *is* the Arc/Rc
                    // value, and the reference count the vtable
                    // would maintain is unobservable (as it is for
                    // the modelled `Arc` itself). Carrying the
                    // waker, handing it to `Context::from_waker`
                    // and cloning it (a value copy, exactly how the
                    // modelled `Arc` clones) are faithful. Calls
                    // that would go through the vtable -
                    // `Waker::wake` / `wake_by_ref` - are not
                    // modelled and stay unsupported at their own
                    // call sites.
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![value],
                    })
                } else if target == "std::io::Error" && source == "std::io::ErrorKind" {
                    // `Error::from(kind)` is std's `const_error!`:
                    // the bare kind in the bitpacked error
                    // representation (the same shape the fs layer
                    // and the process model produce).
                    let disc = match value {
                        Value::Enum { variant, .. } => variant,
                        Value::Uint(v) => v as u32,
                        Value::Int(v) => v as u32,
                        ref other => {
                            return Err(InterpError::type_mismatch(
                                "From::from to io::Error",
                                "an ErrorKind value",
                                other.kind_name(),
                            ));
                        }
                    };
                    Ok(io_simple_error_value(disc))
                } else if target.starts_with("std::collections::VecDeque<") {
                    // `VecDeque: From<Vec<T>> / From<[T; N]> /
                    // From<&[T]>`: the deque has no modelled shape
                    // of its own in this value model (its value is
                    // the real aggregate, built and consumed by
                    // the converted std bodies), so the conversion
                    // assembles that aggregate from the source's
                    // elements (see `vec_deque_from_value`).
                    Ok(self.vec_deque_from_value(frames, value)?)
                } else if (target.starts_with("std::rc::Rc<")
                    || target.starts_with("std::sync::Arc<"))
                    && (source.starts_with("std::vec::Vec<")
                        || source.starts_with("&[")
                        || source.starts_with("&mut ["))
                {
                    // `Rc<[T]>: From<Vec<T>>` / `From<&[T]>` (and
                    // the `Arc` twins): a fresh counted handle
                    // over the element sequence. The slice
                    // pointee lives in the allocation store as
                    // the element tuple, the shape every slice
                    // consumer of a reference-counted handle
                    // derives the length from (see the handle
                    // arm of `unsize_to_slice`).
                    let resolved = self.resolve_value(frames, &value)?;
                    let elements = match resolved {
                        Value::Enum { fields, .. } => fields,
                        Value::Tuple(items) => items,
                        Value::FatRef { base, .. } => match *base {
                            Value::Tuple(items) => items,
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "From::from to a counted slice",
                                    "slice value over an element tuple",
                                    other.kind_name(),
                                ));
                            }
                        },
                        ref other => {
                            return Err(InterpError::type_mismatch(
                                "From::from to a counted slice",
                                "vector or slice value",
                                other.kind_name(),
                            ));
                        }
                    };
                    Ok(self.rc_new_handle(Value::Tuple(elements)))
                } else if target == "std::path::PathBuf" {
                    // `PathBuf: From<&str> / From<&Path> /
                    // From<&OsStr> / From<String> / From<OsString>`:
                    // every one of them is the source's byte
                    // string in a fresh buffer (the shape
                    // `pathbuf_value` builds). An owned
                    // string-like source arrives as its wrapper
                    // aggregate; a borrowed one as the byte
                    // string itself or a reference to it.
                    let bytes = match string_aggregate_bytes(&value) {
                        Some(bytes) => bytes,
                        None => self.path_bytes_of(frames, &value)?.0,
                    };
                    Ok(pathbuf_value(&bytes))
                } else {
                    Err(unsupported(format!(
                        "From::from from {source} to {target} is not runtime-supported"
                    )))
                }
            }
            Builtin::MustUse => {
                arity(1)?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::TlsRegisterDtor => {
                arity(2)?;
                let mut vals = vals.into_iter();
                let data = vals.next().expect("arity checked");
                let dtor = vals.next().expect("arity checked");
                self.register_tls_dtor(data, dtor);
                Ok(Value::Unit)
            }
            Builtin::PanicAccessError => {
                // `#[track_caller]`: the caller-location stamp was
                // popped before the dispatch, like the other panic
                // entry points.
                arity(1)?;
                Err(InterpError::panic_str_payload(
                    "cannot access a Thread Local Storage value during or after destruction: AccessError",
                ))
            }
            Builtin::Deref => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                if self_ty == "std::string::String" {
                    // `String::deref -> &str`: the string data itself.
                    let s = self.resolve_string(frames, &vals[0])?;
                    Ok(Value::Str(s))
                } else if self_ty == "std::path::PathBuf" {
                    // `PathBuf::deref -> &Path`: the path the
                    // buffer holds, in the shape a path over the
                    // same bytes denotes (see `sub_path_value`).
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    let bytes = path_buf_bytes(&pointee).ok_or_else(|| {
                        InterpError::type_mismatch(
                            "Deref::deref for PathBuf",
                            "a path buffer value",
                            pointee.kind_name(),
                        )
                    })?;
                    Ok(sub_path_value(&bytes))
                } else if self_ty == "std::ffi::OsString" {
                    // `OsString::deref -> &OsStr`: the string's
                    // bytes as the slice value, the same shape
                    // `PathBuf`'s deref produces.
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    let bytes = os_string_bytes(&pointee).ok_or_else(|| {
                        InterpError::type_mismatch(
                            "Deref::deref for OsString",
                            "an OS string value",
                            pointee.kind_name(),
                        )
                    })?;
                    Ok(sub_path_value(&bytes))
                } else if self_ty == "std::ffi::CString" {
                    // `CString::deref -> &CStr`: the fat
                    // reference over the owned buffer.
                    self.c_string_deref(frames, &vals[0])
                } else if self_ty.starts_with("std::borrow::Cow<") {
                    // `Cow<B>::deref -> &B::Target`: the borrowed
                    // variant's field already IS the target value
                    // (a `&str` / `&[u8]` reference value); the
                    // owned variant's field is the owned value
                    // (`String` / `Vec<_>`), whose deref is the
                    // same data read as the target — for a string
                    // target that is the string data itself.
                    let cow = self.resolve_value(frames, &vals[0])?;
                    match cow {
                        Value::Enum { variant, mut fields, .. }
                            if fields.len() == 1 =>
                        {
                            let field = fields.pop().expect("len checked");
                            if variant == 0 {
                                Ok(field)
                            } else if self_ty.contains("str") {
                                let s = self.resolve_string(frames, &field)?;
                                Ok(Value::Str(s))
                            } else {
                                Ok(field)
                            }
                        }
                        ref other => Err(InterpError::type_mismatch(
                            "Deref::deref for Cow",
                            "cow aggregate",
                            other.kind_name(),
                        )),
                    }
                } else if self_ty.starts_with("std::vec::Vec<") {
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    let len = pointee.len().ok_or_else(|| {
                        InterpError::type_mismatch(
                            "Deref::deref for Vec",
                            "array value",
                            pointee.kind_name(),
                        )
                    })?;
                    Ok(Value::FatRef {
                        base: Box::new(vals.into_iter().next().expect("arity checked")),
                        len,
                    })
                } else if self_ty.starts_with("std::boxed::Box<") {
                    // A built box's single field is its allocation
                    // pointer, and `deref` yields that very
                    // reference: it names the pointee's storage in
                    // the allocation store, live for as long as the
                    // allocation is. (A box value of the older
                    // aggregate shape — one built by a converted
                    // body or a downcast — holds its pointee as the
                    // field value itself; `deref` then yields a
                    // reference to that value, as it does for
                    // constant aggregates.) A `Box<dyn Trait>` is
                    // the fat pointer itself, and dereferencing it
                    // yields that pointer — the `&dyn` the trait
                    // methods dispatch through.
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    if let Value::DynRef { .. } = pointee {
                        return Ok(pointee);
                    }
                    match pointee {
                        Value::Enum { fields, .. } if fields.len() == 1 => {
                            let inner = fields.into_iter().next().expect("len checked");
                            if let Value::Ref(RefTarget::Alloc { .. }) = inner {
                                return Ok(inner);
                            }
                            // An aggregate-shaped box holds its
                            // pointee as the field value itself.
                            // When the box lives in a place, the
                            // deref denotes that place's field 0,
                            // so writes through the result land in
                            // the box's storage; a box reached
                            // only as a value keeps the snapshot
                            // form, as for constant aggregates.
                            // (A field that is itself a one-field
                            // wrapper around a reference is another
                            // pointer layer, not the pointee: its
                            // snapshot is kept, as for the built
                            // box's allocation pointer above.)
                            let wrapped_pointer = matches!(
                                &inner,
                                Value::Enum { fields, .. }
                                    if fields.len() == 1
                                        && matches!(fields[0], Value::Ref(_))
                            );
                            if !wrapped_pointer {
                                if let Some(place_ref) = self.place_projected_ref(
                                    frames,
                                    &vals[0],
                                    RefProj::Field(0),
                                )? {
                                    return Ok(place_ref);
                                }
                            }
                            Ok(Value::Ref(RefTarget::value(inner)))
                        }
                        ref other => Err(InterpError::type_mismatch(
                            "Deref::deref for Box",
                            "boxed value",
                            other.kind_name(),
                        )),
                    }
                } else if self_ty.starts_with("std::rc::Rc<")
                    || self_ty.starts_with("std::sync::Arc<")
                {
                    // The handle names an allocation in the store:
                    // `deref` yields a reference to the stored
                    // value field — a live place, so reads and
                    // writes reach the storage every handle shares.
                    // An `Rc<dyn Trait>` / `Arc<dyn Trait>`
                    // dereferences to the fat pointer itself, as
                    // `Box<dyn Trait>` does.
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    if let Value::DynRef { .. } = pointee {
                        return Ok(pointee);
                    }
                    match rc_handle_id(&pointee) {
                        Some(id) => Ok(Value::Ref(RefTarget::Alloc {
                            id,
                            projection: vec![RefProj::Field(2)],
                        })),
                        None => Err(InterpError::type_mismatch(
                            "Deref::deref for Rc/Arc",
                            "a reference-counted handle",
                            pointee.kind_name(),
                        )),
                    }
                } else if self_ty_is(self_ty, "std::cell::Ref")
                    || self_ty_is(self_ty, "std::cell::RefMut")
                {
                    // A borrow guard: field 0 is the reference to the
                    // cell's value stored by the borrow builtins.
                    let guard = self.resolve_value(frames, &vals[0])?;
                    match guard {
                        Value::Enum { fields, .. } if fields.len() == 2 => {
                            Ok(fields.into_iter().next().expect("len checked"))
                        }
                        ref other => Err(InterpError::type_mismatch(
                            "RefCell borrow guard",
                            "Ref/RefMut guard aggregate",
                            other.kind_name(),
                        )),
                    }
                } else if self_ty.starts_with('&') {
                    // A reference's own `Deref` returns the inner
                    // reference itself: `<&T as Deref>::deref` and
                    // `<&mut T as Deref>::deref` take `&self`, so
                    // the argument names the place holding the
                    // inner reference, and reading it exactly once
                    // yields that reference. Generic code over
                    // `P: Deref` instantiated with a reference (the
                    // `Pin<&mut T>` machinery, `ManuallyDrop`
                    // access, ...) reaches this arm.
                    Ok(self.read_ref_once(frames, &vals[0])?)
                } else if self_ty.starts_with("std::pin::Pin<") {
                    // `Pin<P>` derefs exactly as the pointer it
                    // wraps: the wrapper is a one-field aggregate
                    // holding the pointer value.
                    let wrapper = self.resolve_value(frames, &vals[0])?;
                    let pointer = one_field_value(wrapper, "Deref::deref for Pin")?;
                    deref_pointer_value(pointer)
                } else if self_ty.starts_with("std::mem::ManuallyDrop<") {
                    // A one-field newtype wrapper: the result is a
                    // reference to the held value.
                    self.field_zero_ref(frames, &vals[0], "Deref::deref for ManuallyDrop")
                } else if self_ty.starts_with("std::panic::AssertUnwindSafe<") {
                    // A one-field newtype wrapper: the result is a
                    // reference to the held value.
                    self.field_zero_ref(frames, &vals[0], "Deref::deref for AssertUnwindSafe")
                } else if self_ty == "std::path::PathBuf" {
                    // `PathBuf::deref -> &Path`: the model's path
                    // values hold the byte string itself at the
                    // bottom of a chain of one-field wrappers (the
                    // `OsString`, its buffer, the byte vector's own
                    // wrapper), and a `Path` is that same byte
                    // string under the same kind of wrappers (the
                    // constructors reach it through raw pointer
                    // casts). Descend the chain and return a
                    // reference to the byte data: a live reference
                    // when the buffer lives in a place, and a
                    // self-contained fat pointer when the data is a
                    // byte vector's element aggregate.
                    let resolved = self.deref_value(frames, &vals[0])?;
                    let stored = self.read_resolved(frames, &resolved)?;
                    let mut depth = 0usize;
                    let mut bottom = &stored;
                    while let Value::Enum { fields, .. } = bottom {
                        // The byte vector itself is an aggregate
                        // of `Uint` elements: stop there, even
                        // when it holds a single byte (a one-byte
                        // path's vector is a one-field aggregate,
                        // indistinguishable in shape from a
                        // wrapper without this check).
                        if fields
                            .iter()
                            .all(|f| matches!(f, Value::Uint(_)))
                        {
                            break;
                        }
                        if fields.len() != 1 {
                            break;
                        }
                        depth += 1;
                        bottom = &fields[0];
                    }
                    match bottom {
                        Value::Str(_) | Value::Bytes(_) => {
                            let steps =
                                vec![RefProj::Field(0); depth];
                            match resolved {
                                Resolved::Place {
                                    frame_pos,
                                    local,
                                    mut projection,
                                } => {
                                    projection.extend(steps);
                                    Ok(Value::Ref(RefTarget::Place {
                                        frame: frames[frame_pos].id,
                                        local,
                                        projection,
                                    }))
                                }
                                Resolved::Static {
                                    name,
                                    mut projection,
                                } => {
                                    projection.extend(steps);
                                    Ok(Value::Ref(RefTarget::Static {
                                        name,
                                        projection,
                                    }))
                                }
                                Resolved::Alloc {
                                    id,
                                    mut projection,
                                } => {
                                    projection.extend(steps);
                                    Ok(Value::Ref(RefTarget::Alloc {
                                        id,
                                        projection,
                                    }))
                                }
                                Resolved::Value(base, mut projection) => {
                                    projection.extend(steps);
                                    Ok(Value::Ref(RefTarget::Value(
                                        Box::new(base),
                                        projection,
                                    )))
                                }
                            }
                        }
                        Value::FatRef { .. } => Ok(bottom.clone()),
                        Value::Enum { fields, .. }
                            if fields
                                .iter()
                                .all(|f| matches!(f, Value::Uint(_))) =>
                        {
                            let bytes: Vec<u8> = fields
                                .iter()
                                .map(|f| match f {
                                    Value::Uint(b) => *b as u8,
                                    _ => unreachable!("shape checked"),
                                })
                                .collect();
                            let len = bytes.len() as u128;
                            Ok(Value::FatRef {
                                base: Box::new(Value::Bytes(bytes)),
                                len,
                            })
                        }
                        other => Err(InterpError::type_mismatch(
                            "Deref::deref for PathBuf",
                            "path buffer holding byte data",
                            other.kind_name(),
                        )),
                    }
                } else {
                    Err(unsupported(format!(
                        "Deref::deref for {self_ty} is not runtime-supported"
                    )))
                }
            }
            Builtin::PathPart(part) => {
                arity(1)?;
                let (bytes, was_str) = self.path_bytes_of(frames, &vals[0])?;
                let part = {
                    use std::os::unix::ffi::OsStrExt;
                    let path = std::path::Path::new(
                        std::ffi::OsStr::from_bytes(&bytes),
                    );
                    let part: Option<&[u8]> = match part {
                        PathPart::Parent => {
                            path.parent().map(|p| p.as_os_str().as_bytes())
                        }
                        PathPart::FileName => {
                            path.file_name().map(|s| s.as_bytes())
                        }
                        PathPart::FileStem => {
                            path.file_stem().map(|s| s.as_bytes())
                        }
                        PathPart::Extension => {
                            path.extension().map(|s| s.as_bytes())
                        }
                    };
                    part.map(<[u8]>::to_vec)
                };
                Ok(match part {
                    Some(bytes) => some_value(if was_str {
                        // The parts of a text path are text: the
                        // parse only cuts at ASCII separators.
                        match String::from_utf8(bytes) {
                            Ok(s) => Value::Str(s),
                            Err(e) => Value::Bytes(e.into_bytes()),
                        }
                    } else {
                        Value::Bytes(bytes)
                    }),
                    None => none_value(),
                })
            }
            Builtin::PathComponents => {
                arity(1)?;
                let (bytes, _) = self.path_bytes_of(frames, &vals[0])?;
                Ok(components_state(&bytes, 0, 0))
            }
            Builtin::PathToPathBuf => {
                arity(1)?;
                let (bytes, _) = self.path_bytes_of(frames, &vals[0])?;
                Ok(pathbuf_value(&bytes))
            }
            Builtin::PathBufPush => {
                arity(2)?;
                let (addition, _) = self.path_bytes_of(frames, &vals[1])?;
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_resolved(frames, &target, |slot| {
                    let current = string_aggregate_bytes(slot).ok_or_else(|| {
                        InterpError::type_mismatch(
                            "PathBuf::push",
                            "path buffer holding byte data",
                            slot.kind_name(),
                        )
                    })?;
                    let mut buf = host_path(&current);
                    buf.push(host_path(&addition));
                    *slot = pathbuf_value(&host_path_bytes(&buf));
                    Ok(Value::Unit)
                })
            }
            Builtin::PathBufPop => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_resolved(frames, &target, |slot| {
                    let current = string_aggregate_bytes(slot).ok_or_else(|| {
                        InterpError::type_mismatch(
                            "PathBuf::pop",
                            "path buffer holding byte data",
                            slot.kind_name(),
                        )
                    })?;
                    let mut buf = host_path(&current);
                    let popped = buf.pop();
                    *slot = pathbuf_value(&host_path_bytes(&buf));
                    Ok(Value::Bool(popped))
                })
            }
            Builtin::PathBufSetExtension => {
                arity(2)?;
                let (extension, _) = self.path_bytes_of(frames, &vals[1])?;
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_resolved(frames, &target, |slot| {
                    let current = string_aggregate_bytes(slot).ok_or_else(|| {
                        InterpError::type_mismatch(
                            "PathBuf::set_extension",
                            "path buffer holding byte data",
                            slot.kind_name(),
                        )
                    })?;
                    use std::os::unix::ffi::OsStrExt;
                    let mut buf = host_path(&current);
                    let set =
                        buf.set_extension(std::ffi::OsStr::from_bytes(&extension));
                    *slot = pathbuf_value(&host_path_bytes(&buf));
                    Ok(Value::Bool(set))
                })
            }
            Builtin::PathWithExtension => {
                arity(2)?;
                let (bytes, _) = self.path_bytes_of(frames, &vals[0])?;
                let (extension, _) = self.path_bytes_of(frames, &vals[1])?;
                use std::os::unix::ffi::OsStrExt;
                let buf = host_path(&bytes)
                    .with_extension(std::ffi::OsStr::from_bytes(&extension));
                Ok(pathbuf_value(&host_path_bytes(&buf)))
            }
            Builtin::PathBufNew => Ok(pathbuf_value(&[])),
            Builtin::FromUtf8 => {
                arity(1)?;
                let (bytes, _) = self.path_bytes_of(frames, &vals[0])?;
                Ok(match std::str::from_utf8(&bytes) {
                    Ok(_) => Value::Enum {
                        variant: 0,
                        fields: vec![Value::Str(
                            // The bytes are valid UTF-8 (just
                            // checked) and the model's string is
                            // the same data.
                            String::from_utf8(bytes)
                                .expect("validity checked above"),
                        )],
                    },
                    Err(e) => Value::Enum {
                        variant: 1,
                        fields: vec![Value::Enum {
                            variant: 0,
                            fields: vec![
                                Value::Uint(e.valid_up_to() as u128),
                                match e.error_len() {
                                    Some(n) => some_value(Value::Uint(n as u128)),
                                    None => none_value(),
                                },
                            ],
                        }],
                    },
                })
            }
            Builtin::ToUppercase => {
                arity(1)?;
                // The argument is the `str` data itself or a
                // reference to it, depending on the MIR shape.
                let s = match &vals[0] {
                    Value::Str(s) => s.clone(),
                    _ => self.resolve_string(frames, &vals[0])?,
                };
                Ok(Value::Str(s.to_uppercase()))
            }
            Builtin::ToLowercase => {
                arity(1)?;
                let s = match &vals[0] {
                    Value::Str(s) => s.clone(),
                    _ => self.resolve_string(frames, &vals[0])?,
                };
                Ok(Value::Str(s.to_lowercase()))
            }
            Builtin::StringNew => {
                arity(0)?;
                Ok(Value::Str(String::new()))
            }
            Builtin::StringWithCapacity => {
                // The capacity argument is not represented, exactly
                // as for `Vec::with_capacity`: the result is the
                // empty string. A guest allocator still observes
                // the buffer allocation the native call makes
                // (the abstract string never reads the buffer;
                // the matching deallocation runs through the
                // string's drop glue as usual).
                arity(1)?;
                let cap = match &vals[0] {
                    Value::Uint(n) => *n,
                    Value::Int(n) if *n >= 0 => *n as u128,
                    _ => 0,
                };
                if cap > 0 {
                    if let Some(ptr) = self.call_guest_alloc(frames, cap, 1)? {
                        *self.pending_string_buffer.borrow_mut() =
                            Some((ptr, cap));
                    }
                }
                Ok(Value::Str(String::new()))
            }
            Builtin::StringLen | Builtin::StrLen => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                Ok(Value::Uint(s.len() as u128))
            }
            Builtin::StringIsEmpty | Builtin::StrIsEmpty => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                Ok(Value::Bool(s.is_empty()))
            }
            Builtin::StringCapacity => {
                arity(1)?;
                Err(unsupported(format!(
                    "{name}: allocation capacity is not tracked by \
                     the abstract string model"
                )))
            }
            Builtin::StringAsStr => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                Ok(Value::Str(s))
            }
            Builtin::StringAsBytes | Builtin::StrAsBytes => {
                arity(1)?;
                self.string_bytes_ref(frames, &vals[0])
            }
            Builtin::StringIntoBytes => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: s
                        .into_bytes()
                        .into_iter()
                        .map(|b| Value::Uint(b as u128))
                        .collect(),
                })
            }
            Builtin::StringIntoBoxedStr => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                Ok(self.new_box(Value::Str(s)))
            }
            Builtin::NoAllocShimMarker => {
                arity(0)?;
                Ok(Value::Unit)
            }
            Builtin::AsRef => {
                arity(1)?;
                self.as_ref_view(frames, generic_args, &vals[0], false)
            }
            Builtin::AsMut => {
                arity(1)?;
                self.as_ref_view(frames, generic_args, &vals[0], true)
            }
            Builtin::VecIntoBoxedSlice => {
                arity(1)?;
                match vals.into_iter().next().expect("arity checked") {
                    Value::Enum { fields, .. } => {
                        Ok(self.new_box(Value::Tuple(fields)))
                    }
                    Value::Tuple(items) => Ok(self.new_box(Value::Tuple(items))),
                    other => Err(InterpError::type_mismatch(
                        "Vec::into_boxed_slice",
                        "vector value",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::StringPushStr => {
                arity(2)?;
                let suffix = self.resolve_string(frames, &vals[1])?;
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_resolved(frames, &target, |slot| match slot {
                    Value::Str(s) => {
                        s.push_str(&suffix);
                        Ok(Value::Unit)
                    }
                    other => Err(InterpError::type_mismatch(
                        "String::push_str",
                        "string value",
                        other.kind_name(),
                    )),
                })
            }
            Builtin::StringPush => {
                arity(2)?;
                let c = match vals[1] {
                    Value::Uint(v) => u32::try_from(v)
                        .ok()
                        .and_then(char::from_u32)
                        .ok_or_else(|| {
                            InterpError::type_mismatch(
                                "String::push",
                                "char",
                                "invalid code point",
                            )
                        })?,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            "String::push",
                            "char",
                            other.kind_name(),
                        ))
                    }
                };
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_resolved(frames, &target, |slot| match slot {
                    Value::Str(s) => {
                        s.push(c);
                        Ok(Value::Unit)
                    }
                    other => Err(InterpError::type_mismatch(
                        "String::push",
                        "string value",
                        other.kind_name(),
                    )),
                })
            }
            Builtin::StrChars => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                let items = s
                    .chars()
                    .map(|c| Value::Uint(c as u32 as u128))
                    .collect();
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Tuple(items), Value::Uint(0)],
                })
            }
            Builtin::StrMakeAsciiCase(upper) => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_resolved(frames, &target, |slot| match slot {
                    Value::Str(s) => {
                        if upper {
                            s.make_ascii_uppercase();
                        } else {
                            s.make_ascii_lowercase();
                        }
                        Ok(Value::Unit)
                    }
                    other => Err(InterpError::type_mismatch(
                        "str::make_ascii_uppercase/lowercase",
                        "string value",
                        other.kind_name(),
                    )),
                })
            }
            Builtin::StrBytes => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                let items = s
                    .as_bytes()
                    .iter()
                    .map(|b| Value::Uint(*b as u128))
                    .collect();
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Tuple(items), Value::Uint(0)],
                })
            }
            Builtin::StrTrim(side) => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                let out = match side {
                    TrimSide::Both => s.trim(),
                    TrimSide::Start => s.trim_start(),
                    TrimSide::End => s.trim_end(),
                };
                Ok(Value::Str(out.to_string()))
            }
            Builtin::PanicStr => {
                arity(1)?;
                // `core::panicking::panic` takes the message as a
                // string; formatted-panic shapes hand over a built
                // `fmt::Arguments` instead. The payload mirrors
                // core: a literal message is a `&str` payload, a
                // formatted one a `String`.
                match &vals[0] {
                    Value::FmtArguments(FmtArguments::Plain(text)) => {
                        Err(InterpError::panic_str_payload(text.clone()))
                    }
                    Value::FmtArguments(_) => {
                        let message = self.format_arguments(
                            frames,
                            &self.coerce_arguments(frames, &vals[0], name)?,
                            0,
                        )?;
                        Err(InterpError::panic_string_payload(message))
                    }
                    _ => {
                        let message = self.resolve_string(frames, &vals[0])?;
                        Err(InterpError::panic_str_payload(message))
                    }
                }
            }
            Builtin::PanicFmt => {
                arity(1)?;
                // A verbatim `Arguments` (a literal message) is a
                // `&str` payload; a formatted one is a `String`.
                match self.coerce_arguments(frames, &vals[0], name)? {
                    FmtArguments::Plain(text) => {
                        Err(InterpError::panic_str_payload(text))
                    }
                    args => {
                        let message = self.format_arguments(frames, &args, 0)?;
                        Err(InterpError::panic_string_payload(message))
                    }
                }
            }
            Builtin::AssertFailed => {
                arity(4)?;
                // (kind, &left, &right, Option<fmt::Arguments>): the
                // message mirrors core's: the comparison, the custom
                // message when one was given, and both operands.
                let kind_variant = match self.resolve_value(frames, &vals[0])? {
                    Value::Enum { variant, .. } => variant,
                    _ => u32::MAX,
                };
                let mut message = match kind_variant {
                    0 => "assertion `left == right` failed".to_string(),
                    1 => "assertion `left != right` failed".to_string(),
                    2 => "assertion `left matches right` failed".to_string(),
                    _ => "assertion failed".to_string(),
                };
                if let Value::Enum {
                    variant: 1,
                    fields,
                } = self.resolve_value(frames, &vals[3])?
                {
                    if let Some(arg) = fields.into_iter().next() {
                        if let Ok(coerced) = self.coerce_arguments(frames, &arg, name) {
                            if let Ok(text) = self.format_arguments(frames, &coerced, 0)
                            {
                                message = format!("{message}: {text}");
                            }
                        }
                    }
                }
                let left = self.resolve_value(frames, &vals[1])?;
                let right = self.resolve_value(frames, &vals[2])?;
                message = format!("{message}\n  left: {left}\n right: {right}");
                Err(InterpError::panic_string_payload(message))
            }
            Builtin::PanicNounwindFmt => {
                // `panic_nounwind_fmt(args, force_no_backtrace)`:
                // the flag only controls backtrace capture in the
                // native runtime, which no guest program can
                // observe; the panic carries the formatted message,
                // exactly as `panic_fmt`'s does.
                if vals.is_empty() || vals.len() > 2 {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "1 or 2 argument(s)",
                        format!("{} argument(s)", vals.len()),
                    ));
                }
                match self.coerce_arguments(frames, &vals[0], name)? {
                    FmtArguments::Plain(text) => {
                        Err(InterpError::panic_str_payload(text))
                    }
                    args => {
                        let message = self.format_arguments(frames, &args, 0)?;
                        Err(InterpError::panic_string_payload(message))
                    }
                }
            }
            Builtin::OptionUnwrapFailed => {
                arity(0)?;
                Err(InterpError::panic_str_payload(
                    "called `Option::unwrap()` on a `None` value",
                ))
            }
            Builtin::ResultUnwrapFailed => {
                arity(2)?;
                // `unwrap_failed(msg, error: &dyn Debug)` panics
                // with "{msg}: {error:?}"; the Debug rendering is
                // the fmt engine's, the same one `panic!` uses.
                let msg = self.resolve_string(frames, &vals[0])?;
                let error = match &vals[1] {
                    Value::DynRef { .. } => {
                        let target = self.deref_value(frames, &vals[1])?;
                        self.read_resolved(frames, &target)?
                    }
                    other => self.resolve_value(frames, other)?,
                };
                let rendered = self.format_value(
                    frames,
                    &error,
                    FmtTrait::Debug,
                    "",
                    &FmtOptions::new(),
                    0,
                )?;
                Err(InterpError::panic_string_payload(format!(
                    "{msg}: {rendered}"
                )))
            }
            Builtin::FromAsciiRadixPanic => {
                arity(1)?;
                let radix = as_u128(&vals[0], name)?;
                Err(InterpError::panic_string_payload(format!(
                    "from_ascii_bytes_radix: radix must lie in the range `[2, 36]` - found {radix}"
                )))
            }
            Builtin::Memchr { reverse } => {
                arity(2)?;
                let needle = match self.resolve_value(frames, &vals[0])? {
                    Value::Uint(v) => v as u8,
                    Value::Int(v) => v as u8,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "memchr needle",
                            "byte value",
                            other.kind_name(),
                        ))
                    }
                };
                let (elems, _) = self.slice_elements(frames, &vals[1])?;
                let bytes: Vec<u8> = elems
                    .iter()
                    .map(|e| match e {
                        Value::Uint(v) => Ok(*v as u8),
                        Value::Int(v) => Ok(*v as u8),
                        other => Err(InterpError::type_mismatch(
                            "memchr haystack",
                            "byte elements",
                            other.kind_name(),
                        )),
                    })
                    .collect::<InterpResult<Vec<u8>>>()?;
                let found = if reverse {
                    bytes.iter().rposition(|b| *b == needle)
                } else {
                    bytes.iter().position(|b| *b == needle)
                };
                Ok(match found {
                    Some(i) => some_value(Value::Uint(i as u128)),
                    None => none_value(),
                })
            }
            Builtin::SliceIndexFail => {
                arity(3)?;
                let start = as_u128(&vals[0], name)? as usize;
                let end = as_u128(&vals[1], name)? as usize;
                let len = as_u128(&vals[2], name)? as usize;
                Err(InterpError::panic_string_payload(
                    slice_index_fail_message(start, end, len),
                ))
            }
            Builtin::RangeErrorReport => {
                arity(2)?;
                // `RangeError::report(self, len)` forwards to
                // `slice_index_fail`: with a zero start for
                // `EndPastLen`, with the stored pair for
                // `StartPastEnd`.
                let len = as_u128(&vals[1], name)? as usize;
                let (start, end) = match self.resolve_value(frames, &vals[0])? {
                    Value::Enum { variant: 0, fields } if fields.len() == 1 => {
                        (0, as_u128(&fields[0], name)? as usize)
                    }
                    Value::Enum { variant: 1, fields } if fields.len() == 2 => (
                        as_u128(&fields[0], name)? as usize,
                        as_u128(&fields[1], name)? as usize,
                    ),
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "a RangeError value",
                            other.kind_name(),
                        ))
                    }
                };
                Err(InterpError::panic_string_payload(
                    slice_index_fail_message(start, end, len),
                ))
            }
            Builtin::VecRemoveAssertFailed => {
                arity(2)?;
                let index = as_u128(&vals[0], name)?;
                let len = as_u128(&vals[1], name)?;
                Err(InterpError::panic_string_payload(format!(
                    "removal index (is {index}) should be < len (is {len})"
                )))
            }
            Builtin::CatchUnwind => {
                arity(3)?;
                let try_fn = expect_fn_ptr(&vals[0], name)?;
                let catch_fn = expect_fn_ptr(&vals[2], name)?;
                let panicked =
                    self.exec_catch_unwind(frames, &try_fn, vals[1].clone(), &catch_fn)?;
                Ok(Value::Bool(panicked))
            }
            Builtin::PanicCleanup => {
                arity(1)?;
                match &vals[0] {
                    Value::PanicPayload(payload) => Ok(Value::DynRef {
                        // The payload travels in the aggregate
                        // box shape a sized box value has, so the
                        // data pointer has the shape `Value::DynRef`
                        // documents for a `Box` coercion.
                        base: Box::new(Value::Enum {
                            variant: 0,
                            fields: vec![payload.value.as_ref().clone()],
                        }),
                        concrete: payload.ty.clone(),
                        traits: vec![
                            "std::any::Any".to_string(),
                            "std::marker::Send".to_string(),
                        ],
                        // Fabricated by the interpreter, not formed
                        // by an unsizing cast: no accounted glue
                        // travels with it, and its drop is judged
                        // by shape (see `DropPlan::Dyn`).
                        drop_glue: None,
                    }),
                    other => Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "a raw panic payload",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::PanicAny => {
                arity(1)?;
                let ty = generic_args.first().cloned().unwrap_or_default();
                let value = vals.into_iter().next().expect("arity checked");
                // The message the default hook reports for the
                // payload: a string payload reports as itself,
                // anything else as the opaque box (std's
                // `payload_as_str`).
                let message = match &value {
                    Value::Str(text) => text.clone(),
                    _ if ty == "&str" => self
                        .resolve_string(frames, &value)
                        .unwrap_or_else(|_| "Box<dyn Any>".to_string()),
                    _ => "Box<dyn Any>".to_string(),
                };
                Err(InterpError::panic_with_payload(
                    message,
                    PanicPayload {
                        ty,
                        value: Box::new(value),
                    },
                ))
            }
            Builtin::ResumeUnwind => {
                arity(1)?;
                // A resumed panic is not a new panic: the hook
                // (already fired at the original raise) must not
                // fire again, which the unwinder's resumed flag
                // records.
                self.note_resumed_panic();
                let (concrete, value, _) = resolve_dyn_box(self, frames, &vals[0], name)?;
                // The re-raised payload is the pointee the dyn
                // value's base denotes (the box's contents), not
                // the box itself: the cleanup that later turns a
                // payload back into a dyn value wraps it in a box
                // again, so handing it the box would double it.
                let value = dyn_box_pointee(self, frames, &value)?;
                let message = match &value {
                    Value::Str(text) => text.clone(),
                    _ => "Box<dyn Any>".to_string(),
                };
                Err(InterpError::panic_with_payload(
                    message,
                    PanicPayload {
                        ty: concrete,
                        value: Box::new(value),
                    },
                ))
            }
            Builtin::DynAnyDowncastRef => {
                arity(1)?;
                let want = expect_generic_ty(generic_args, name)?;
                let (concrete, value, _) = resolve_dyn_box(self, frames, &vals[0], name)?;
                if concrete == want {
                    // The result denotes the payload the dyn
                    // value's base points at (the box's pointee),
                    // not the box itself.
                    let value = dyn_box_pointee(self, frames, &value)?;
                    Ok(Value::Enum {
                        variant: 1,
                        fields: vec![Value::Ref(RefTarget::value(value))],
                    })
                } else {
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![],
                    })
                }
            }
            Builtin::DynAnyDowncast(t_index) => {
                arity(1)?;
                let want = expect_generic_ty_at(generic_args, name, t_index)?;
                let (concrete, value, original) =
                    resolve_dyn_box(self, frames, &vals[0], name)?;
                if concrete == want {
                    // `Ok` holds a `Box<T>`: the dyn value's base
                    // itself, which is the box in the shape the
                    // sized value had (see `Value::DynRef`) — an
                    // allocation-backed box from an unsizing cast,
                    // or the aggregate box a fabricated payload
                    // travels in. Wrapping it once more would hand
                    // back a box whose field is another box, and
                    // writes through the result's `Deref` could not
                    // reach the payload.
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![value],
                    })
                } else {
                    Ok(Value::Enum {
                        variant: 1,
                        fields: vec![original],
                    })
                }
            }
            Builtin::DynAnyIs => {
                arity(1)?;
                let want = expect_generic_ty(generic_args, name)?;
                let (concrete, _, _) = resolve_dyn_box(self, frames, &vals[0], name)?;
                Ok(Value::Bool(concrete == want))
            }
            Builtin::PartialEqEq | Builtin::PartialEqNe => {
                arity(2)?;
                // std defines `Path` / `PathBuf` equality over
                // the parsed components (repeated and trailing
                // separators and interior `.` components are
                // normalised away), not over the raw bytes the
                // structural comparison below would use. The
                // callee name carries no impl subject in this
                // model, so the path types are recognised by the
                // call's own generic arguments (the `Self` and
                // `Rhs` type tags).
                let path_cmp = self.comparison_is_path(frames, generic_args, &vals);
                let eq = if path_cmp {
                    let lhs = self.path_cmp_bytes(frames, &vals[0])?;
                    let rhs = self.path_cmp_bytes(frames, &vals[1])?;
                    path_components(&lhs) == path_components(&rhs)
                } else {
                    self.values_equal(frames, &vals[0], &vals[1])?
                };
                Ok(Value::Bool(if kind == Builtin::PartialEqNe {
                    !eq
                } else {
                    eq
                }))
            }
            Builtin::PartialCmp => {
                arity(2)?;
                let ord = if self.comparison_is_path(frames, generic_args, &vals) {
                    let lhs = self.path_cmp_bytes(frames, &vals[0])?;
                    let rhs = self.path_cmp_bytes(frames, &vals[1])?;
                    Some(path_ordering(&lhs, &rhs))
                } else {
                    self.values_cmp(frames, &vals[0], &vals[1], cmp_discriminants)?
                };
                Ok(match ord {
                    Some(ord) => Value::Enum {
                        variant: 1,
                        fields: vec![ordering_value(ord)],
                    },
                    None => Value::Enum {
                        variant: 0,
                        fields: vec![],
                    },
                })
            }
            Builtin::OrdCmp => {
                arity(2)?;
                let ord = if self.comparison_is_path(frames, generic_args, &vals) {
                    let lhs = self.path_cmp_bytes(frames, &vals[0])?;
                    let rhs = self.path_cmp_bytes(frames, &vals[1])?;
                    Some(path_ordering(&lhs, &rhs))
                } else {
                    self.values_cmp(frames, &vals[0], &vals[1], cmp_discriminants)?
                };
                match ord {
                    Some(ord) => Ok(ordering_value(ord)),
                    // `Ord` is a total ordering: an unordered pair
                    // (a NaN-carrying value) cannot reach a genuine
                    // `Ord` implementation, so this stays explicit.
                    None => Err(InterpError::type_mismatch(
                        "Ord::cmp",
                        "a total ordering",
                        "an unordered value (NaN comparison)",
                    )),
                }
            }
            Builtin::PartialOrdRel(op) => {
                arity(2)?;
                let ord = if self.comparison_is_path(frames, generic_args, &vals) {
                    let lhs = self.path_cmp_bytes(frames, &vals[0])?;
                    let rhs = self.path_cmp_bytes(frames, &vals[1])?;
                    Some(path_ordering(&lhs, &rhs))
                } else {
                    self.values_cmp(frames, &vals[0], &vals[1], cmp_discriminants)?
                };
                // An unordered pair makes every relational operator
                // false, as `partial_cmp` returning `None` does.
                Ok(Value::Bool(match (ord, op) {
                    (Some(Ordering::Less), RelOp::Lt) => true,
                    (Some(Ordering::Greater), RelOp::Gt) => true,
                    (Some(Ordering::Less | Ordering::Equal), RelOp::Le) => true,
                    (Some(Ordering::Greater | Ordering::Equal), RelOp::Ge) => true,
                    _ => false,
                }))
            }
            Builtin::CloneClone => {
                arity(1)?;
                // Cloning an mpsc handle creates another live
                // handle: the channel's handle count must move
                // with it, or disconnection would be observed
                // while a clone still lives.
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                let chan_kind = if self_ty.starts_with("std::sync::mpsc::Sender<")
                    || self_ty.starts_with("std::sync::mpsc::SyncSender<")
                {
                    Some("sender")
                } else if self_ty.starts_with("std::sync::mpsc::Receiver<") {
                    Some("receiver")
                } else {
                    None
                };
                match &vals[0] {
                    // `&self`: the clone is a copy of the pointee
                    // value (which may itself be a reference value
                    // when `Self` is a reference type).
                    Value::Ref(_) => {
                        let target = self.deref_value(frames, &vals[0])?;
                        let value = self.read_resolved(frames, &target)?;
                        // A `VaList` clone is C `va_copy`: a second
                        // list over the same arguments with its own
                        // cursor, not another handle on the same
                        // cursor.
                        if let Value::VaList(id) = value {
                            return self.va_list_duplicate(id);
                        }
                        // A `Waker` / `LocalWaker` clone is not
                        // a value copy: a waker built by
                        // `from_raw` clones through its
                        // `RawWakerVTable` clone entry (which is
                        // where the reference count moves), and
                        // the modelled `From` waker clones the
                        // handle it carries.
                        if self_ty == "std::task::Waker"
                            || self_ty == "std::task::LocalWaker"
                        {
                            return self.clone_waker_value(frames, &value, name);
                        }
                        // A `Box` clone owns a fresh allocation
                        // holding a clone of the pointee: copying
                        // the box value itself would alias the
                        // source allocation, so writes through the
                        // clone would rewrite the original (and the
                        // two boxes would free the same storage).
                        // The pointee is cloned by
                        // `clone_owned_value`, the same value-copy
                        // reading this builtin applies everywhere,
                        // with owned boxes re-allocated at every
                        // level.
                        if self_ty.starts_with("std::boxed::Box<") {
                            if let Some((id, projection)) = box_alloc_pointer(&value) {
                                let pointee = self.read_resolved(
                                    frames,
                                    &Resolved::Alloc { id, projection },
                                )?;
                                let cloned = self.clone_owned_value(frames, &pointee)?;
                                return Ok(self.new_box(cloned));
                            }
                        }
                        // A cloned container owns its elements: a
                        // `Vec` value's fields are its elements in
                        // this model, and a tuple's / array's fields
                        // likewise, so the elements go through the
                        // same owned clone (a `Vec<Box<T>>` clone
                        // must not alias the source's boxes).
                        let value = if self_ty.starts_with("std::vec::Vec<")
                            || self_ty.starts_with("alloc::vec::Vec<")
                        {
                            match value {
                                Value::Enum { variant, fields } => Value::Enum {
                                    variant,
                                    fields: fields
                                        .iter()
                                        .map(|f| self.clone_owned_value(frames, f))
                                        .collect::<InterpResult<Vec<_>>>()?,
                                },
                                // A sparse vector's stored
                                // elements clone the same way;
                                // its unmaterialised tail has
                                // nothing to clone.
                                Value::SparseVec { elems, len } => {
                                    Value::SparseVec {
                                        elems: elems
                                            .iter()
                                            .map(|f| {
                                                self.clone_owned_value(frames, f)
                                            })
                                            .collect::<InterpResult<Vec<_>>>()?,
                                        len,
                                    }
                                }
                                other => other,
                            }
                        } else if (self_ty.starts_with('(') || self_ty.starts_with('['))
                            && matches!(value, Value::Tuple(_))
                        {
                            match value {
                                Value::Tuple(items) => Value::Tuple(
                                    items
                                        .iter()
                                        .map(|f| self.clone_owned_value(frames, f))
                                        .collect::<InterpResult<Vec<_>>>()?,
                                ),
                                other => other,
                            }
                        } else {
                            value
                        };
                        if let (Some(kind), Some(id)) =
                            (chan_kind, channel_handle_id(&value))
                        {
                            self.channel_handle_cloned(id, kind);
                        }
                        // Cloning a reference-counted handle makes
                        // another live handle on the same
                        // allocation: the count the converted
                        // bodies read must move with it, or
                        // uniqueness (`Rc::get_mut`, the counts,
                        // drop) would be observed wrong.
                        if self_ty.starts_with("std::rc::Rc<")
                            || self_ty.starts_with("std::sync::Arc<")
                        {
                            if let Some(id) = rc_handle_id(&value) {
                                self.rc_inc_strong(id)?;
                            }
                        } else if self_ty.starts_with("std::rc::Weak<")
                            || self_ty.starts_with("std::sync::Weak<")
                        {
                            if let Some(id) = rc_handle_id(&value) {
                                self.rc_inc_weak(id)?;
                            }
                        }
                        Ok(value)
                    }
                    // A fat pointer clones as the pointer itself
                    // (slices are `Clone` through their reference).
                    other => Ok(other.clone()),
                }
            }
            Builtin::DefaultDefault => {
                arity(0)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                match self.default_value(tag) {
                    Some(value) => Ok(value),
                    None => Err(unsupported(format!(
                        "Default::default for {tag} is not runtime-supported"
                    ))),
                }
            }
            Builtin::ToStringToString => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                match self.resolve_value(frames, &vals[0])? {
                    Value::Str(s) => Ok(Value::Str(s)),
                    Value::Bool(b) => Ok(Value::Str(b.to_string())),
                    Value::Int(v) => Ok(Value::Str(v.to_string())),
                    Value::Uint(v) if self_ty == "char" => {
                        match u32::try_from(v).ok().and_then(char::from_u32) {
                            Some(c) => Ok(Value::Str(c.to_string())),
                            None => Err(unsupported(
                                "ToString::to_string for an invalid char value",
                            )),
                        }
                    }
                    Value::Uint(v) => Ok(Value::Str(v.to_string())),
                    // `Arguments` implements `Display` (and so
                    // `ToString`) by formatting itself — exactly
                    // the `format_arguments` model.
                    Value::FmtArguments(ref args) => {
                        Ok(Value::Str(self.format_arguments(frames, args, 0)?))
                    }
                    ref other => Err(unsupported(format!(
                        "ToString::to_string for {self_ty} (a {}) is not runtime-supported",
                        other.kind_name()
                    ))),
                }
            }
            Builtin::BoxNew => {
                arity(1)?;
                // A box's pointee lives in the allocation store;
                // the box value is the allocation pointer (see
                // `Interpreter::new_box`). With a guest allocator
                // the storage comes from its shim, as the native
                // `__rust_alloc` call inside `Box::new` binds to
                // it; a zero-sized pointee allocates nothing in
                // either world.
                let v = vals.into_iter().next().expect("arity checked");
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                if let Some((size, align)) = self.layout_of(tag) {
                    if size > 0 {
                        if let Some(ptr) =
                            self.call_guest_alloc(frames, size, align)?
                        {
                            if matches!(ptr, Value::Ref(RefTarget::Null)) {
                                return Err(unsupported(
                                    "guest allocator returned null for Box::new",
                                ));
                            }
                            let target = self.deref_value(frames, &ptr)?;
                            self.write_resolved(frames, &target, v)?;
                            let boxed = Value::Enum {
                                variant: 0,
                                fields: vec![ptr],
                            };
                            self.record_box_layout(&boxed, size, align);
                            return Ok(boxed);
                        }
                    }
                }
                Ok(self.new_box(v))
            }
            Builtin::ManuallyDropNew => {
                arity(1)?;
                // A one-field ADT aggregate holding its value.
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![vals.into_iter().next().expect("arity checked")],
                })
            }
            Builtin::RcNew | Builtin::ArcNew => {
                arity(1)?;
                // A fresh reference-counted allocation in the
                // store: the handle is the `NonNull` pointer to the
                // stored `RcInner` / `ArcInner`, so the converted
                // std bodies (`get_mut`, `make_mut`, the count
                // accessors, `Weak`) operate on the real layout.
                let v = vals.into_iter().next().expect("arity checked");
                Ok(self.rc_new_handle(v))
            }
            Builtin::RcFromRaw | Builtin::ArcFromRaw | Builtin::WeakFromRaw => {
                arity(1)?;
                // Rebuild the handle naming the allocation the
                // raw pointer denotes. The pointer `into_raw`
                // produces (and the only shape this reconstruction
                // accepts) is the stored inner aggregate's data
                // pointer: a reference to the allocation carrying
                // the data field's projection (`Field(2)`, the
                // position `Deref::deref` reads through), or the
                // bare allocation reference itself. The handle is
                // the allocation's canonical handle value; no
                // count moves, as `into_raw` forgot a live
                // handle rather than releasing one. Any other
                // pointer shape denotes no allocation this model
                // can name, and is an explicit error.
                // The argument is the pointer value itself (a
                // reference into the allocation store), not a
                // place denoting it: it is matched as it stands.
                let ptr = vals.into_iter().next().expect("arity checked");
                let id = match &ptr {
                    Value::Ref(RefTarget::Alloc { id, projection })
                        if projection.is_empty()
                            || matches!(
                                projection.as_slice(),
                                [RefProj::Field(2)]
                            ) =>
                    {
                        Some(*id)
                    }
                    _ => None,
                };
                match id {
                    Some(id) => Ok(self.rc_handle(id)),
                    None => Err(InterpError::type_mismatch(
                        "Rc::from_raw / Arc::from_raw / Weak::from_raw",
                        "a raw pointer to a reference-counted allocation's data field",
                        ptr.kind_name(),
                    )),
                }
            }
            Builtin::BoxNewUninit => {
                arity(0)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                match uninit_value_for_tag(tag) {
                    Some(pointee) => {
                        if let Some((size, align)) = self.layout_of(tag) {
                            if size > 0 {
                                if let Some(ptr) = self
                                    .call_guest_alloc(frames, size, align)?
                                {
                                    if matches!(
                                        ptr,
                                        Value::Ref(RefTarget::Null)
                                    ) {
                                        return Err(unsupported(
                                            "guest allocator returned null \
                                             for Box::new_uninit",
                                        ));
                                    }
                                    let target =
                                        self.deref_value(frames, &ptr)?;
                                    self.write_resolved(
                                        frames,
                                        &target,
                                        pointee,
                                    )?;
                                    let boxed = Value::Enum {
                                        variant: 0,
                                        fields: vec![ptr],
                                    };
                                    self.record_box_layout(
                                        &boxed, size, align,
                                    );
                                    return Ok(boxed);
                                }
                            }
                        }
                        Ok(self.new_box(pointee))
                    }
                    None => Err(unsupported(format!(
                        "Box::new_uninit for {tag} is not runtime-supported"
                    ))),
                }
            }
            Builtin::BoxAssumeInitIntoVec => {
                arity(1)?;
                let arg = vals.into_iter().next().expect("arity checked");
                let boxed = match &arg {
                    Value::Ref(_) | Value::FatRef { .. } => {
                        let target = self.deref_value(frames, &arg)?;
                        self.read_resolved(frames, &target)?
                    }
                    _ => arg,
                };
                match boxed {
                    Value::Enum { fields, .. } if fields.len() == 1 => {
                        let pointee = fields.into_iter().next().expect("one field");
                        // A built box's field is its allocation
                        // pointer: the array lives behind it in
                        // the allocation store.
                        let pointee = match &pointee {
                            Value::Ref(RefTarget::Alloc { .. }) => {
                                let target = self.deref_value(frames, &pointee)?;
                                self.read_resolved(frames, &target)?
                            }
                            _ => pointee,
                        };
                        match assume_init_elements(pointee) {
                            Some(elems) => Ok(Value::Enum {
                                variant: 0,
                                fields: elems,
                            }),
                            None => Err(InterpError::type_mismatch(
                                "box_assume_init_into_vec_unsafe",
                                "box of an array value",
                                "box of a non-array value",
                            )),
                        }
                    }
                    ref other => Err(InterpError::type_mismatch(
                        "box_assume_init_into_vec_unsafe",
                        "box value",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::VecNew => {
                arity(0)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                })
            }
            Builtin::VecExtendFromSlice => {
                arity(2)?;
                let target = self.deref_value(frames, &vals[0])?;
                let (elems, _) = self.slice_elements(frames, &vals[1])?;
                self.modify_resolved(frames, &target, |slot| match slot {
                    Value::Enum { fields, .. } => {
                        fields.extend(elems.iter().cloned());
                        Ok(Value::Unit)
                    }
                    // The appended elements would land past the
                    // unmaterialised region, as for `Vec::push`.
                    Value::SparseVec { .. } => Err(unsupported(
                        "Vec::extend_from_slice on a vector whose elements are not materialised"
                            .to_string(),
                    )),
                    other => Err(InterpError::type_mismatch(
                        "Vec::extend_from_slice",
                        "vector value",
                        other.kind_name(),
                    )),
                })
            }
            Builtin::VecPush => {
                arity(2)?;
                let target = self.deref_value(frames, &vals[0])?;
                let mut vals = vals.into_iter();
                let _ = vals.next();
                let elem = vals.next().expect("arity checked");
                self.modify_resolved(frames, &target, |slot| match slot {
                    Value::Enum { fields, .. } => {
                        fields.push(elem);
                        Ok(Value::Unit)
                    }
                    // The pushed element would land past the
                    // unmaterialised region, which only
                    // materialising that region could store.
                    Value::SparseVec { .. } => Err(unsupported(
                        "Vec::push on a vector whose elements are not materialised"
                            .to_string(),
                    )),
                    other => Err(InterpError::type_mismatch(
                        "Vec::push",
                        "vector value",
                        other.kind_name(),
                    )),
                })
            }
            Builtin::VecLen | Builtin::VecIsEmpty => {
                arity(1)?;
                // Only the length is read: resolve by borrowing,
                // so the query does not clone every element of the
                // vector the way the owned resolution did.
                let value = self.resolve_ref(frames, &vals[0])?;
                let len = value.len().ok_or_else(|| {
                    InterpError::type_mismatch(
                        "Vec length",
                        "vector value",
                        value.kind_name(),
                    )
                })?;
                if kind == Builtin::VecLen {
                    Ok(Value::Uint(len))
                } else {
                    Ok(Value::Bool(len == 0))
                }
            }
            Builtin::VecWithCapacity => {
                // One argument (`with_capacity`) or two
                // (`with_capacity_in`: capacity and the allocator,
                // a zero-sized handle here). The requested capacity
                // has no representation in the value model, so the
                // result is the empty vector, as for `Vec::new`.
                if vals.len() != 1 && vals.len() != 2 {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "1 or 2 argument(s)",
                        format!("{} argument(s)", vals.len()),
                    ));
                }
                Ok(Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                })
            }
            Builtin::VecFromElem => {
                arity(2)?;
                let mut vals = vals.into_iter();
                let elem = vals.next().expect("arity checked");
                let count = match vals.next().expect("arity checked") {
                    Value::Uint(n) => n,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            "std::vec::from_elem",
                            "element count (usize)",
                            other.kind_name(),
                        ));
                    }
                };
                // Mirrors the repeat-aggregate resource bound in
                // mir_convert: past it the vector is not
                // materialised, so a huge count fails explicitly
                // instead of exhausting the host.
                const MAX_FROM_ELEM: u128 = 1_000_000;
                if count > MAX_FROM_ELEM {
                    return Err(unsupported(format!(
                        "std::vec::from_elem with {count} elements exceeds the model's {MAX_FROM_ELEM}-element bound"
                    )));
                }
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![elem; count as usize],
                })
            }
            Builtin::VecReserve => {
                arity(2)?;
                Ok(Value::Unit)
            }
            Builtin::VecTryReserve => {
                arity(2)?;
                let additional = match vals[1] {
                    Value::Uint(n) => n,
                    // A `usize` argument whose high bit is set (e.g.
                    // `usize::MAX`) materialises as a sign-extended
                    // signed value; its bit pattern is the count.
                    Value::Int(v) => (v as u64) as u128,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "additional capacity (usize)",
                            other.kind_name(),
                        ));
                    }
                };
                // Only the element count is read: borrow the
                // receiver instead of cloning every element.
                let resolved = self.resolve_ref(frames, &vals[0])?;
                let len = value_elements(&resolved)?.len() as u128;
                let elem_tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (elem_size, _) = type_layout(elem_tag).ok_or_else(|| {
                    unsupported(format!(
                        "{name}: element type {elem_tag} has no layout known to the                          value model, so the capacity-overflow check cannot be decided"
                    ))
                })?;
                // The std rule (RawVec's capacity-overflow check):
                // the byte size `required * elem_size` must not
                // exceed `isize::MAX`; a zero-sized element is
                // bounded only by `usize::MAX` elements. Allocation
                // failure, the other `TryReserveError` source, has
                // no abstract counterpart.
                let required = len + additional;
                let overflow = if elem_size == 0 {
                    required > usize::MAX as u128
                } else {
                    required > (isize::MAX as u128) / elem_size
                };
                if overflow {
                    // Err(TryReserveError::CapacityOverflow)
                    Ok(Value::Enum {
                        variant: 1,
                        fields: vec![Value::Enum {
                            variant: 0,
                            fields: Vec::new(),
                        }],
                    })
                } else {
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![Value::Unit],
                    })
                }
            }
            Builtin::VecShrink => {
                // `shrink_to_fit` passes one argument, `shrink_to`
                // two; neither changes anything the value model can
                // observe (see the variant's documentation).
                if vals.len() != 1 && vals.len() != 2 {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "1 or 2 argument(s)",
                        format!("{} argument(s)", vals.len()),
                    ));
                }
                Ok(Value::Unit)
            }
            Builtin::VecSetLen => {
                arity(2)?;
                let new_len = match vals[1] {
                    Value::Uint(n) => n as usize,
                    Value::Int(n) if n >= 0 => n as usize,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            "Vec::set_len",
                            "unsigned length",
                            other.kind_name(),
                        ));
                    }
                };
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_resolved(frames, &target, |slot| {
                    // The vector's value at the new length (see
                    // `vec_with_len`): materialised up to the
                    // materialisation limit, sparse past it, so
                    // an astronomical length (the
                    // `Vec::with_capacity(1_000_000_000)` +
                    // `set_len` idiom) declares its length
                    // instead of exhausting the host. Shrinking
                    // forgets the tail elements without dropping
                    // them, exactly as the real `set_len`
                    // contract specifies.
                    let elems = match std::mem::replace(slot, Value::Uninit) {
                        Value::Enum { fields, .. } => fields,
                        Value::SparseVec { elems, .. } => elems,
                        other => {
                            return Err(InterpError::type_mismatch(
                                "Vec::set_len",
                                "vector value",
                                other.kind_name(),
                            ));
                        }
                    };
                    *slot = vec_with_len(elems, new_len);
                    Ok(Value::Unit)
                })
            }
            Builtin::VecExtendTrusted => {
                arity(2)?;
                // The callee's generic arguments are the Vec's own
                // parameters followed by the iterator type, which
                // is the stepping dispatch key.
                let iter_ty = generic_args.last().map(String::as_str).unwrap_or("");
                let target = self.deref_value(frames, &vals[0])?;
                let mut fields = match self.read_resolved(frames, &target)? {
                    Value::Enum { fields, .. } => fields,
                    Value::SparseVec { .. } => {
                        return Err(unsupported(
                            "Vec::extend_trusted on a vector whose elements are not materialised"
                                .to_string(),
                        ));
                    }
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            "Vec::extend_trusted",
                            "vector value",
                            other.kind_name(),
                        ))
                    }
                };
                let items = self.drain_iterator(
                    frames,
                    iter_ty,
                    vals.into_iter().nth(1).expect("arity checked"),
                )?;
                fields.extend(items);
                self.write_resolved(
                    frames,
                    &target,
                    Value::Enum {
                        variant: 0,
                        fields,
                    },
                )?;
                Ok(Value::Unit)
            }
            Builtin::VecFromIterNested => {
                arity(1)?;
                // The trait's generic arguments are the element
                // type followed by the iterator type, which is
                // the stepping dispatch key.
                let iter_ty = generic_args.last().map(String::as_str).unwrap_or("");
                let items = self.drain_iterator(
                    frames,
                    iter_ty,
                    vals.into_iter().next().expect("arity checked"),
                )?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: items,
                })
            }
            Builtin::VecCapacity => Err(unsupported(
                "Vec::capacity is not runtime-supported (capacity is not represented in the value model)"
                    .to_string(),
            )),
            Builtin::SliceToVec => {
                // `to_vec` takes the slice alone; `to_vec_in`
                // takes the allocator handle after it (a
                // zero-sized value in this model).
                if vals.is_empty() || vals.len() > 2 {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "1 or 2 argument(s)",
                        format!("{} argument(s)", vals.len()),
                    ));
                }
                let (elems, _) = self.slice_elements(frames, &vals[0])?;
                let fields = elems
                    .iter()
                    .map(|e| self.clone_owned_value(frames, e))
                    .collect::<InterpResult<Vec<_>>>()?;
                Ok(Value::Enum {
                    variant: 0,
                    fields,
                })
            }
            Builtin::SliceIter => {
                arity(1)?;
                let (elems, _) = self.slice_elements(frames, &vals[0])?;
                Ok(slice_iter_value(elems))
            }
            Builtin::StrContains | Builtin::StrStartsWith | Builtin::StrEndsWith => {
                arity(2)?;
                let text_of = |v: &Value| -> InterpResult<String> {
                    match self.resolve_value(frames, v)? {
                        Value::Str(s) => Ok(s),
                        Value::Bytes(b) => Ok(String::from_utf8_lossy(&b).into_owned()),
                        other => Err(InterpError::type_mismatch(
                            "str pattern argument",
                            "string value",
                            other.kind_name(),
                        )),
                    }
                };
                let haystack = text_of(&vals[0])?;
                let needle = text_of(&vals[1])?;
                Ok(Value::Bool(match kind {
                    Builtin::StrContains => haystack.contains(&needle),
                    Builtin::StrStartsWith => haystack.starts_with(&needle),
                    _ => haystack.ends_with(&needle),
                }))
            }
            Builtin::SliceWindows => {
                arity(2)?;
                let size = match self.resolve_value(frames, &vals[1])? {
                    Value::Uint(v) => v,
                    Value::Enum { fields, .. } if fields.len() == 1 => match fields[0] {
                        Value::Uint(v) => v,
                        ref other => {
                            return Err(InterpError::type_mismatch(
                                "windows size",
                                "non-zero integer",
                                other.kind_name(),
                            ))
                        }
                    },
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            "windows size",
                            "non-zero integer",
                            other.kind_name(),
                        ))
                    }
                };
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![vals[0].clone(), Value::Uint(size), Value::Uint(0)],
                })
            }
            Builtin::IterEmpty => {
                arity(0)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                })
            }
            Builtin::IndexIndex => {
                arity(2)?;
                // A range index (`slice[1..3]`, `slice[..]`, ...)
                // yields a sub-slice; the index type tag names the
                // range kind. Anything else is a plain integer index.
                let idx_tag = generic_args.get(1).map(String::as_str).unwrap_or("");
                if idx_tag.contains("ops::Range") {
                    return self.index_range(frames, name, &vals[0], &vals[1], idx_tag);
                }
                let idx = expect_index(&vals[1], name)?;
                // A receiver backed by a live place yields a live
                // element reference, exactly as `IndexMut` does:
                // interior-mutability writes through the shared
                // reference (`v[i].set(..)` on a `Vec<Cell<_>>`)
                // must land in the container's storage, which a
                // snapshot of the element cannot do.
                let fat_len = match &vals[0] {
                    Value::FatRef { len, .. } => Some(*len),
                    _ => None,
                };
                if let Some(r) = self.index_place_ref(frames, &vals[0], idx, fat_len)? {
                    return Ok(r);
                }
                let (elems, len) = self.slice_elements_ref(frames, &vals[0])?;
                if idx as u128 >= len {
                    return Err(InterpError::panic(format!(
                        "index out of bounds: the len is {len} but the index is {idx}"
                    )));
                }
                // A receiver with no live place behind it (a
                // constant aggregate, a detached value): reads
                // resolve against the element value itself, as for
                // constant aggregates. Only the one element is
                // cloned out of the borrowed element storage.
                let elem = elems.at(idx);
                Ok(Value::Ref(RefTarget::value(elem)))
            }
            Builtin::IndexMut => {
                arity(2)?;
                // A range index (`slice[1..3]`, `slice[..]`, ...)
                // yields a mutable sub-slice; the index type tag
                // names the range kind, as for the shared `Index`.
                let idx_tag = generic_args.get(1).map(String::as_str).unwrap_or("");
                if idx_tag.contains("ops::Range") {
                    return self.index_range_mut(frames, name, &vals[0], &vals[1], idx_tag);
                }
                let idx = expect_index(&vals[1], name)?;
                // A fat receiver's bounds are its carried length,
                // not the length of the base it points into (a
                // sub-slice of a longer array must panic at its
                // own length).
                let fat_len = match &vals[0] {
                    Value::FatRef { len, .. } => Some(*len),
                    _ => None,
                };
                let target = match &vals[0] {
                    Value::Ref(_) | Value::FatRef { .. } => {
                        self.deref_value(frames, &vals[0])?
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            "IndexMut::index_mut",
                            "mutable reference",
                            other.kind_name(),
                        ))
                    }
                };
                match target {
                    Resolved::Place {
                        frame_pos,
                        local,
                        projection,
                    } => {
                        let base = self.read_resolved(frames, &Resolved::Place {
                            frame_pos,
                            local,
                            projection: projection.clone(),
                        })?;
                        let len = match fat_len {
                            Some(len) => len,
                            None => base.len().ok_or_else(|| {
                                InterpError::type_mismatch(
                                    "IndexMut::index_mut",
                                    "indexable value",
                                    base.kind_name(),
                                )
                            })?,
                        };
                        if idx as u128 >= len {
                            return Err(InterpError::panic(format!(
                                "index out of bounds: the len is {len} but the index is {idx}"
                            )));
                        }
                        match base {
                            // Array (tuple) elements and vector
                            // (ADT aggregate) elements are both
                            // addressable through an index projection:
                            // the projection layer reads and writes
                            // `RefProj::Index` on both shapes.
                            Value::Tuple(_) | Value::Enum { .. } => {
                                let mut projection = projection;
                                projection.push(RefProj::Index(idx));
                                let frame_id = frames[frame_pos].id;
                                Ok(Value::Ref(RefTarget::Place {
                                    frame: frame_id,
                                    local,
                                    projection,
                                }))
                            }
                            ref other => Err(unsupported(format!(
                                "IndexMut::index_mut on a {} is not runtime-supported",
                                other.kind_name()
                            ))),
                        }
                    }
                    Resolved::Static { name, projection } => {
                        let base = self.read_resolved(frames, &Resolved::Static {
                            name: name.clone(),
                            projection: projection.clone(),
                        })?;
                        let len = base.len().ok_or_else(|| {
                            InterpError::type_mismatch(
                                "IndexMut::index_mut",
                                "indexable value",
                                base.kind_name(),
                            )
                        })?;
                        if idx as u128 >= len {
                            return Err(InterpError::panic(format!(
                                "index out of bounds: the len is {len} but the index is {idx}"
                            )));
                        }
                        match base {
                            Value::Tuple(_) | Value::Enum { .. } => {
                                let mut projection = projection;
                                projection.push(RefProj::Index(idx));
                                Ok(Value::Ref(RefTarget::Static { name, projection }))
                            }
                            ref other => Err(unsupported(format!(
                                "IndexMut::index_mut on a {} is not runtime-supported",
                                other.kind_name()
                            ))),
                        }
                    }
                    Resolved::Alloc { id, projection } => {
                        let base = self.read_resolved(frames, &Resolved::Alloc {
                            id,
                            projection: projection.clone(),
                        })?;
                        let len = base.len().ok_or_else(|| {
                            InterpError::type_mismatch(
                                "IndexMut::index_mut",
                                "indexable value",
                                base.kind_name(),
                            )
                        })?;
                        if idx as u128 >= len {
                            return Err(InterpError::panic(format!(
                                "index out of bounds: the len is {len} but the index is {idx}"
                            )));
                        }
                        match base {
                            Value::Tuple(_) => {
                                let mut projection = projection;
                                projection.push(RefProj::Index(idx));
                                Ok(Value::Ref(RefTarget::Alloc { id, projection }))
                            }
                            ref other => Err(unsupported(format!(
                                "IndexMut::index_mut on a {} is not runtime-supported",
                                other.kind_name()
                            ))),
                        }
                    }
                    Resolved::Value(..) => Err(unsupported(
                        "IndexMut::index_mut on a constant aggregate is not runtime-supported",
                    )),
                }
            }
            Builtin::DerefMut => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                if self_ty == "std::string::String" {
                    // `&mut str` over the `String` place itself: reads
                    // see the string data, writes replace it.
                    Ok(vals.into_iter().next().expect("arity checked"))
                } else if self_ty.starts_with("std::vec::Vec<") {
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    let len = pointee.len().ok_or_else(|| {
                        InterpError::type_mismatch(
                            "DerefMut::deref_mut for Vec",
                            "array value",
                            pointee.kind_name(),
                        )
                    })?;
                    Ok(Value::FatRef {
                        base: Box::new(vals.into_iter().next().expect("arity checked")),
                        len,
                    })
                } else if self_ty == "std::ffi::CString" {
                    // `CString::deref_mut -> &mut CStr`: the
                    // same fat reference `deref` produces.
                    self.c_string_deref(frames, &vals[0])
                } else if self_ty.starts_with("std::boxed::Box<") {
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    // As in `Deref`: a `Box<dyn Trait>` dereferences
                    // to the fat pointer itself, and a built box
                    // dereferences to its allocation pointer — a
                    // live reference into the allocation store, so
                    // writes through it land in the pointee's
                    // storage and are observed through every copy
                    // of the box.
                    if let Value::DynRef { .. } = pointee {
                        return Ok(pointee);
                    }
                    match pointee {
                        Value::Enum { fields, .. } if fields.len() == 1 => {
                            let inner = fields.into_iter().next().expect("len checked");
                            if let Value::Ref(RefTarget::Alloc { .. }) = inner {
                                return Ok(inner);
                            }
                            // As in `Deref`: an aggregate-shaped
                            // box in a live place yields a
                            // reference to that place's field 0,
                            // so mutable writes through the
                            // result land in the box; a box
                            // reached only as a value keeps the
                            // snapshot form.
                            // (A field that is itself a one-field
                            // wrapper around a reference is another
                            // pointer layer, not the pointee: its
                            // snapshot is kept, as for the built
                            // box's allocation pointer above.)
                            let wrapped_pointer = matches!(
                                &inner,
                                Value::Enum { fields, .. }
                                    if fields.len() == 1
                                        && matches!(fields[0], Value::Ref(_))
                            );
                            if !wrapped_pointer {
                                if let Some(place_ref) = self.place_projected_ref(
                                    frames,
                                    &vals[0],
                                    RefProj::Field(0),
                                )? {
                                    return Ok(place_ref);
                                }
                            }
                            Ok(Value::Ref(RefTarget::value(inner)))
                        }
                        ref other => Err(InterpError::type_mismatch(
                            "DerefMut::deref_mut for Box",
                            "boxed value",
                            other.kind_name(),
                        )),
                    }
                } else if self_ty.starts_with("std::mem::ManuallyDrop<") {
                    // A one-field newtype wrapper: the result is a
                    // mutable reference to the held value. As for
                    // `AssertUnwindSafe` below, a place-backed
                    // argument yields a live place reference, so
                    // writes through the result land in the wrapper
                    // (a resolved-value snapshot would silently
                    // drop them).
                    self.field_zero_ref(
                        frames,
                        &vals[0],
                        "DerefMut::deref_mut for ManuallyDrop",
                    )
                } else if self_ty.starts_with("std::rc::Rc<")
                    || self_ty.starts_with("std::sync::Arc<")
                {
                    // As in `Deref`: a reference to the stored
                    // value field, a live place in the allocation
                    // store, so a write through it is observed by
                    // every handle on the allocation. A dyn pointee
                    // yields the fat pointer itself.
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    if let Value::DynRef { .. } = pointee {
                        return Ok(pointee);
                    }
                    match rc_handle_id(&pointee) {
                        Some(id) => Ok(Value::Ref(RefTarget::Alloc {
                            id,
                            projection: vec![RefProj::Field(2)],
                        })),
                        None => Err(InterpError::type_mismatch(
                            "DerefMut::deref_mut for Rc/Arc",
                            "a reference-counted handle",
                            pointee.kind_name(),
                        )),
                    }
                } else if self_ty_is(self_ty, "std::cell::RefMut") {
                    // A mutable borrow guard: field 0 is the
                    // reference to the cell's value stored by the
                    // borrow builtins.
                    let guard = self.resolve_value(frames, &vals[0])?;
                    match guard {
                        Value::Enum { fields, .. } if fields.len() == 2 => {
                            Ok(fields.into_iter().next().expect("len checked"))
                        }
                        ref other => Err(InterpError::type_mismatch(
                            "RefCell borrow guard",
                            "RefMut guard aggregate",
                            other.kind_name(),
                        )),
                    }
                } else if self_ty.starts_with("&mut ") {
                    // A mutable reference's own `DerefMut` returns
                    // the inner `&mut T` itself: the argument is
                    // `&mut self`, naming the place that holds the
                    // inner reference, so reading it exactly once
                    // yields that reference. The coroutine
                    // runtime paths reach this arm constantly:
                    // `Pin<&mut C>` is built by dereferencing the
                    // `&mut C` it wraps, and async drop glue
                    // dereferences `&mut ManuallyDrop<C>` /
                    // `&mut Pin<..>` the same way.
                    Ok(self.read_ref_once(frames, &vals[0])?)
                } else if self_ty.starts_with("std::pin::Pin<") {
                    // `Pin<P>` derefs exactly as the pointer it
                    // wraps: the wrapper is a one-field aggregate
                    // holding the pointer value.
                    let wrapper = self.resolve_value(frames, &vals[0])?;
                    let pointer =
                        one_field_value(wrapper, "DerefMut::deref_mut for Pin")?;
                    deref_pointer_value(pointer)
                } else if self_ty.starts_with("std::panic::AssertUnwindSafe<") {
                    // A one-field newtype wrapper: the result is a
                    // mutable reference to the held value.
                    self.field_zero_ref(
                        frames,
                        &vals[0],
                        "DerefMut::deref_mut for AssertUnwindSafe",
                    )
                } else {
                    Err(unsupported(format!(
                        "DerefMut::deref_mut for {self_ty} is not runtime-supported"
                    )))
                }
            }
            Builtin::CellNew => {
                arity(1)?;
                Ok(cell_value(vals.into_iter().next().expect("arity checked")))
            }
            Builtin::CellGet | Builtin::CellIntoInner => {
                arity(1)?;
                let state = self.resolve_value(frames, &vals[0])?;
                match cell_held(&state).cloned() {
                    Some(value) => Ok(value),
                    None => Err(InterpError::type_mismatch(
                        "Cell value",
                        "cell aggregate",
                        state.kind_name(),
                    )),
                }
            }
            Builtin::CellSet | Builtin::CellReplace => {
                arity(2)?;
                let target = self.deref_value(frames, &vals[0])?;
                let state = self.read_resolved(frames, &target)?;
                let old = match cell_held(&state).cloned() {
                    Some(value) => value,
                    None => {
                        return Err(InterpError::type_mismatch(
                            "Cell::set",
                            "cell aggregate",
                            state.kind_name(),
                        ))
                    }
                };
                self.write_resolved(
                    frames,
                    &target,
                    cell_value(vals.into_iter().nth(1).expect("arity checked")),
                )?;
                if kind == Builtin::CellReplace {
                    Ok(old)
                } else {
                    Ok(Value::Unit)
                }
            }
            Builtin::RefCellNew => {
                arity(1)?;
                let value = vals.into_iter().next().expect("arity checked");
                Ok(refcell_value(value))
            }
            Builtin::RefCellBorrow | Builtin::RefCellBorrowMut => {
                arity(1)?;
                let mutable = kind == Builtin::RefCellBorrowMut;
                let cell = self.refcell_place(&vals[0], name)?;
                let flag_target =
                    crate::interp::refcell_sub_target(&cell, &[RefProj::Field(0)]);
                let flag = self.refcell_flag(frames, &flag_target)?;
                if mutable {
                    if flag != 0 {
                        return Err(InterpError::panic("RefCell already borrowed"));
                    }
                    self.refcell_set_flag(frames, &flag_target, -1)?;
                } else {
                    if flag < 0 {
                        return Err(InterpError::panic(
                            "RefCell already mutably borrowed",
                        ));
                    }
                    self.refcell_set_flag(frames, &flag_target, flag + 1)?;
                }
                Ok(refcell_guard(&cell))
            }
            Builtin::RefCellTryBorrow | Builtin::RefCellTryBorrowMut => {
                arity(1)?;
                let mutable = kind == Builtin::RefCellTryBorrowMut;
                let cell = self.refcell_place(&vals[0], name)?;
                let flag_target =
                    crate::interp::refcell_sub_target(&cell, &[RefProj::Field(0)]);
                let flag = self.refcell_flag(frames, &flag_target)?;
                let conflict = if mutable { flag != 0 } else { flag < 0 };
                if conflict {
                    // Err(BorrowError / BorrowMutError): unit-struct
                    // aggregates in this value model.
                    return Ok(Value::Enum {
                        variant: 1,
                        fields: vec![Value::Enum {
                            variant: 0,
                            fields: Vec::new(),
                        }],
                    });
                }
                self.refcell_set_flag(
                    frames,
                    &flag_target,
                    if mutable { -1 } else { flag + 1 },
                )?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![refcell_guard(&cell)],
                })
            }
            Builtin::RefCellGetMut | Builtin::RefCellAsPtr => {
                arity(1)?;
                let cell = self.refcell_place(&vals[0], name)?;
                Ok(Value::Ref(crate::interp::refcell_sub_target(
                    &cell,
                    &[RefProj::Field(1), RefProj::Field(0)],
                )))
            }
            Builtin::RefCellReplace => {
                arity(2)?;
                let mut vals = vals.into_iter();
                let cell_arg = vals.next().expect("arity checked");
                let new_value = vals.next().expect("arity checked");
                let cell = self.refcell_place(&cell_arg, name)?;
                let flag_target =
                    crate::interp::refcell_sub_target(&cell, &[RefProj::Field(0)]);
                let flag = self.refcell_flag(frames, &flag_target)?;
                if flag != 0 {
                    return Err(InterpError::panic("RefCell already borrowed"));
                }
                let value_target = crate::interp::refcell_sub_target(
                    &cell,
                    &[RefProj::Field(1), RefProj::Field(0)],
                );
                let resolved = self.deref_value(frames, &Value::Ref(value_target))?;
                let old = self.read_resolved(frames, &resolved)?;
                self.write_resolved(frames, &resolved, new_value)?;
                Ok(old)
            }
            Builtin::OptionUnwrap => {
                arity(1)?;
                match self.resolve_value(frames, &vals[0])? {
                    Value::Enum { variant: 1, fields } if fields.len() == 1 => {
                        Ok(fields.into_iter().next().expect("len checked"))
                    }
                    Value::Enum { variant: 0, .. } => Err(InterpError::panic(
                        "called `Option::unwrap()` on a `None` value",
                    )),
                    ref other => Err(InterpError::type_mismatch(
                        "Option::unwrap",
                        "Option value",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::ResultUnwrap => {
                arity(1)?;
                match self.resolve_value(frames, &vals[0])? {
                    Value::Enum { variant: 0, fields } if fields.len() == 1 => {
                        Ok(fields.into_iter().next().expect("len checked"))
                    }
                    Value::Enum { variant: 1, fields } => {
                        let err = fields
                            .into_iter()
                            .next()
                            .unwrap_or(Value::Unit);
                        Err(InterpError::panic(format!(
                            "called `Result::unwrap()` on an `Err` value: {err}"
                        )))
                    }
                    ref other => Err(InterpError::type_mismatch(
                        "Result::unwrap",
                        "Result value",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::OptionIsSome
            | Builtin::OptionIsNone
            | Builtin::ResultIsOk
            | Builtin::ResultIsErr => {
                arity(1)?;
                match self.resolve_value(frames, &vals[0])? {
                    Value::Enum { variant, .. } => Ok(Value::Bool(match kind {
                        Builtin::OptionIsSome => variant == 1,
                        Builtin::OptionIsNone => variant == 0,
                        Builtin::ResultIsOk => variant == 0,
                        _ => variant == 1,
                    })),
                    ref other => Err(InterpError::type_mismatch(
                        "Option/Result variant check",
                        "enum value",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::OptionTake => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                let old = self.read_resolved(frames, &target)?;
                match old {
                    Value::Enum { .. } => {
                        self.write_resolved(frames, &target, none_value())?;
                        Ok(old)
                    }
                    ref other => Err(InterpError::type_mismatch(
                        "Option::take",
                        "Option value",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::MemDrop => {
                arity(1)?;
                // A `mem::drop` whose argument's glue was trivial
                // runs no glue session, so a dropped box's storage
                // free is observed here instead (see
                // `Interpreter::note_box_free`).
                if self.program().alloc_shims.contains_key("__rust_dealloc") {
                    let arg = vals.into_iter().next().expect("arity checked");
                    if crate::interp::box_alloc_pointer(&arg).is_some() {
                        self.note_box_free(frames, &arg);
                        if let Some((id, _)) =
                            crate::interp::box_alloc_pointer(&arg)
                        {
                            self.free_alloc(id);
                        }
                    }
                }
                Ok(Value::Unit)
            }
            Builtin::MemSwap => {
                arity(2)?;
                let a = self.deref_value(frames, &vals[0])?;
                let b = self.deref_value(frames, &vals[1])?;
                let va = self.read_resolved(frames, &a)?;
                let vb = self.read_resolved(frames, &b)?;
                self.write_resolved(frames, &a, vb)?;
                self.write_resolved(frames, &b, va)?;
                Ok(Value::Unit)
            }
            Builtin::SwapNonoverlapping => {
                // `typed_swap_nonoverlapping` swaps a single pair
                // (two arguments); `swap_nonoverlapping` carries
                // the element count as a third.
                let n = match vals.len() {
                    2 => 1,
                    3 => match vals[2] {
                        Value::Uint(n) => usize::try_from(n).map_err(|_| {
                            InterpError::type_mismatch(
                                "swap_nonoverlapping count",
                                "count that fits in usize",
                                format!("{n}"),
                            )
                        })?,
                        Value::Int(n) if n >= 0 => n as usize,
                        ref other => {
                            return Err(InterpError::type_mismatch(
                                "swap_nonoverlapping count",
                                "unsigned integer",
                                other.kind_name(),
                            ))
                        }
                    },
                    _ => {
                        return Err(InterpError::type_mismatch(
                            "swap_nonoverlapping",
                            "two or three arguments",
                            format!("{} arguments", vals.len()),
                        ))
                    }
                };
                if n == 0 {
                    // A zero-count swap touches no memory: the
                    // pointers are never dereferenced.
                    return Ok(Value::Unit);
                }
                let a = self.deref_value(frames, &vals[0])?;
                let b = self.deref_value(frames, &vals[1])?;
                if n == 1 {
                    // A single element is the whole value each
                    // target denotes: the sequence access would
                    // index INTO a structured value (an index
                    // step descends into an aggregate's fields)
                    // instead of naming the element, so the
                    // exchange goes through the whole-value
                    // access, exactly like the one-element copy.
                    let va = self.read_resolved(frames, &a)?;
                    let vb = self.read_resolved(frames, &b)?;
                    self.write_resolved(frames, &a, vb)?;
                    self.write_resolved(frames, &b, va)?;
                    return Ok(Value::Unit);
                }
                // Snapshot both ranges before writing: the
                // ranges are promised non-overlapping, but the
                // snapshot keeps the exchange exact regardless
                // (the same discipline as the element copy).
                let mut va = Vec::with_capacity(n);
                let mut vb = Vec::with_capacity(n);
                for i in 0..n {
                    va.push(self.read_seq_at(frames, &a, i)?);
                    vb.push(self.read_seq_at(frames, &b, i)?);
                }
                for i in 0..n {
                    self.write_seq_at(frames, &a, i, vb[i].clone())?;
                    self.write_seq_at(frames, &b, i, va[i].clone())?;
                }
                Ok(Value::Unit)
            }
            Builtin::MemSizeOf | Builtin::MemAlignOf => {
                arity(0)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                match self.layout_of(tag) {
                    Some((size, align)) => Ok(Value::Uint(if kind == Builtin::MemSizeOf {
                        size
                    } else {
                        align
                    })),
                    None => Err(unsupported(format!(
                        "size_of/align_of for {tag} is not runtime-supported"
                    ))),
                }
            }
            Builtin::MemSizeOfVal => {
                arity(1)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                if tag == "str" {
                    let s = self.resolve_string(frames, &vals[0])?;
                    Ok(Value::Uint(s.len() as u128))
                } else if tag.starts_with('[') && tag.ends_with(']') && !tag.contains(';') {
                    // A slice: element count × element size.
                    let elem = &tag[1..tag.len() - 1];
                    let (elem_size, _) = self.layout_of(elem).ok_or_else(|| {
                        unsupported(format!(
                            "size_of_val for {tag} is not runtime-supported"
                        ))
                    })?;
                    let len = match &vals[0] {
                        Value::FatRef { len, .. } => *len,
                        _ => {
                            let value = self.resolve_value(frames, &vals[0])?;
                            value.len().ok_or_else(|| {
                                InterpError::type_mismatch(
                                    "size_of_val",
                                    "slice value",
                                    value.kind_name(),
                                )
                            })?
                        }
                    };
                    Ok(Value::Uint(len * elem_size))
                } else {
                    match self.layout_of(tag) {
                        Some((size, _)) => Ok(Value::Uint(size)),
                        None => Err(unsupported(format!(
                            "size_of_val for {tag} is not runtime-supported"
                        ))),
                    }
                }
            }
            Builtin::MemAlignOfVal => {
                arity(1)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                if tag == "str" {
                    Ok(Value::Uint(1))
                } else if tag.starts_with('[') && tag.ends_with(']') && !tag.contains(';') {
                    let elem = &tag[1..tag.len() - 1];
                    let (_, align) = self.layout_of(elem).ok_or_else(|| {
                        unsupported(format!(
                            "align_of_val for {tag} is not runtime-supported"
                        ))
                    })?;
                    Ok(Value::Uint(align))
                } else {
                    match self.layout_of(tag) {
                        Some((_, align)) => Ok(Value::Uint(align)),
                        None => Err(unsupported(format!(
                            "align_of_val for {tag} is not runtime-supported"
                        ))),
                    }
                }
            }
            Builtin::FastFloat(op) => {
                arity(2)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (a, b) = match (&vals[0], &vals[1]) {
                    (Value::Float(a), Value::Float(b)) => (*a, *b),
                    _ => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "two float operands",
                            format!(
                                "{} and {}",
                                vals[0].kind_name(),
                                vals[1].kind_name()
                            ),
                        ));
                    }
                };
                let r = match op {
                    FastFloatOp::Add => a + b,
                    FastFloatOp::Sub => a - b,
                    FastFloatOp::Mul => a * b,
                    FastFloatOp::Div => a / b,
                    FastFloatOp::Rem => a % b,
                };
                // A narrower float operation's result rounds to
                // the operation's width, as in the interpreter's
                // own float arithmetic.
                Ok(Value::Float(match tag {
                    "f16" => round_to_f16(r),
                    "f32" => (r as f32) as f64,
                    _ => r,
                }))
            }
            Builtin::FloatToIntUnchecked => {
                arity(1)?;
                let tag = generic_args.get(1).map(String::as_str).unwrap_or("");
                let (bits, signed) = int_tag(tag).ok_or_else(|| {
                    unsupported(format!(
                        "float_to_int_unchecked to {tag} is not runtime-supported"
                    ))
                })?;
                match vals[0] {
                    Value::Float(x) => Ok(if signed {
                        Value::Int(float_to_int(x, bits))
                    } else {
                        Value::Uint(float_to_uint(x, bits))
                    }),
                    ref other => Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "float operand",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::MemNeedsDrop => {
                arity(0)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                match needs_drop_for_tag(tag) {
                    Some(needs) => Ok(Value::Bool(needs)),
                    None => Err(unsupported(format!(
                        "needs_drop for {tag} is not runtime-supported"
                    ))),
                }
            }
            Builtin::ToOwned => {
                arity(1)?;
                match &vals[0] {
                    // A slice: the owned form is a vector of the
                    // elements.
                    Value::FatRef { .. } => {
                        let (elems, _) = self.slice_elements(frames, &vals[0])?;
                        Ok(Value::Enum {
                            variant: 0,
                            fields: elems,
                        })
                    }
                    Value::Ref(_) => {
                        let target = self.deref_value(frames, &vals[0])?;
                        match self.read_resolved(frames, &target)? {
                            Value::Tuple(items) => Ok(Value::Enum {
                                variant: 0,
                                fields: items,
                            }),
                            other => Ok(other),
                        }
                    }
                    other => Ok(other.clone()),
                }
            }
            Builtin::BlackBox => {
                arity(1)?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::IntPow => {
                // One call form in core's own `pow` bodies names
                // this function with no arguments and no return
                // target: it is the cold overflow exit of the
                // checked path, reached only when the
                // exponentiation has already overflowed.
                if vals.is_empty() {
                    return Err(InterpError::panic(
                        "attempt to exponentiate with overflow",
                    ));
                }
                arity(2)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (bits, signed) = int_tag(tag).ok_or_else(|| {
                    unsupported(format!("pow for type {tag} is not runtime-supported"))
                })?;
                let mask = bit_mask(bits);
                let base = (match &vals[0] {
                    Value::Int(i) => *i as u128,
                    Value::Uint(u) => *u,
                    _ => 0,
                }) & mask;
                let mut exp = match &vals[1] {
                    Value::Uint(u) => *u,
                    Value::Int(i) if *i >= 0 => *i as u128,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "pow exponent",
                            "unsigned integer",
                            other.kind_name(),
                        ))
                    }
                };
                // Checked multiplication at the type's width;
                // the wrapped product is kept for the
                // overflow-checks-off path.
                let mul = |x: u128, y: u128| -> (u128, bool) {
                    if signed {
                        let xi = sign_extend(x, bits);
                        let yi = sign_extend(y, bits);
                        let wrapped = (xi.wrapping_mul(yi) as u128) & mask;
                        let in_range = if bits == 128 {
                            xi.checked_mul(yi).is_some()
                        } else {
                            let max = (1i128 << (bits - 1)) - 1;
                            let min = -(1i128 << (bits - 1));
                            xi.checked_mul(yi)
                                .is_some_and(|v| (min..=max).contains(&v))
                        };
                        (wrapped, !in_range)
                    } else {
                        let wrapped = x.wrapping_mul(y) & mask;
                        let over = match x.checked_mul(y) {
                            Some(v) => v > mask,
                            None => true,
                        };
                        (wrapped, over)
                    }
                };
                let mut result: u128 = 1;
                let mut b = base;
                let mut overflowed = false;
                while exp > 0 {
                    if exp & 1 == 1 {
                        let (v, over) = mul(result, b);
                        result = v;
                        overflowed |= over;
                    }
                    exp >>= 1;
                    if exp > 0 {
                        let (v, over) = mul(b, b);
                        b = v;
                        overflowed |= over;
                    }
                }
                if overflowed && self.overflow_checks() {
                    return Err(InterpError::panic(
                        "attempt to exponentiate with overflow",
                    ));
                }
                Ok(if signed {
                    Value::Int(sign_extend(result, bits))
                } else {
                    Value::Uint(result)
                })
            }
            Builtin::FromStrParse => {
                arity(1)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let text = self.resolve_string(frames, &vals[0])?;
                return self.exec_from_str(tag, &text, name);
            }
            Builtin::VolatileLoad => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                // As with the atomic load: a wider-than-byte
                // access into byte storage composes the bytes.
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                if let Some((bits, signed)) = int_tag(tag) {
                    if let Some(value) =
                        self.read_punned_int(frames, &target, bits, signed)?
                    {
                        return Ok(value);
                    }
                }
                // The pointee value itself, composite types
                // included: no wrapper unwrapping (that is the
                // atomic operations' UnsafeCell model, not a
                // property of a plain load).
                Ok(self.read_resolved(frames, &target)?)
            }
            Builtin::VolatileStore => {
                arity(2)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                if self.layout_of(tag).is_some_and(|(size, _)| size == 0) {
                    return Ok(Value::Unit);
                }
                let target = self.deref_value(frames, &vals[0])?;
                // The store matching the punned load above.
                if let Some((bits, _)) = int_tag(tag) {
                    if self.write_punned_int(frames, &target, bits, &vals[1])? {
                        return Ok(Value::Unit);
                    }
                }
                self.write_resolved(frames, &target, vals[1].clone())?;
                Ok(Value::Unit)
            }
            Builtin::VolatileSet => {
                arity(3)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (size, _) = self.layout_of(tag).ok_or_else(|| {
                    unsupported(format!(
                        "volatile_set_memory element type {tag} has no recorded layout"
                    ))
                })?;
                let count = match &vals[2] {
                    Value::Uint(n) => *n,
                    Value::Int(n) if *n >= 0 => *n as u128,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "volatile_set_memory count",
                            "unsigned integer",
                            other.kind_name(),
                        ))
                    }
                };
                for i in 0..count {
                    let p = self.ptr_offset_value(
                        frames,
                        vals[0].clone(),
                        i as i128,
                        Some(size as u64),
                        true,
                    )?;
                    let target = self.deref_value(frames, &p)?;
                    self.write_resolved(frames, &target, vals[1].clone())?;
                }
                Ok(Value::Unit)
            }
            Builtin::VolatileCopy => {
                arity(3)?;
                let mut vals = vals.into_iter();
                let dst = vals.next().expect("arity checked");
                let src_v = vals.next().expect("arity checked");
                let count = vals.next().expect("arity checked");
                // The element type, from the caller's declared
                // pointer types, as on the intrinsic statement
                // path (this builtin's argument order is dst, src).
                let elem = args
                    .get(1)
                    .and_then(|a| crate::interp::operand_pointee_ty(body, a))
                    .or_else(|| {
                        args.first()
                            .and_then(|a| crate::interp::operand_pointee_ty(body, a))
                    });
                self.exec_copy_nonoverlapping(frames, src_v, dst, count, elem.as_ref())?;
                Ok(Value::Unit)
            }
            Builtin::PtrNull => {
                arity(0)?;
                // The null pointer: a dedicated reference target
                // denoting no place and no value. It is equal only
                // to itself (see `values_equal`); dereferencing it
                // is an explicit error, as a null dereference has
                // no meaning in the value model.
                Ok(Value::Ref(RefTarget::Null))
            }
            Builtin::PtrOffsetFrom { unsigned } => {
                arity(2)?;
                // Two address-only pointers have no pointee to
                // dereference in this model; their distance is a
                // byte distance over the synthetic addresses.
                let addr_pair = match (&vals[0], &vals[1]) {
                    (
                        Value::Ref(RefTarget::Address(a)),
                        Value::Ref(RefTarget::Address(b)),
                    ) => Some((*a, *b)),
                    _ => None,
                };
                let d = match addr_pair {
                    Some((a, b)) => {
                        let tag = generic_args.first().map(String::as_str).unwrap_or("");
                        let size = self
                            .layout_of(tag)
                            .map(|(size, _)| size)
                            .filter(|size| *size > 0)
                            .ok_or_else(|| {
                                unsupported(format!(
                                    "ptr_offset_from with element type `{tag}` is not \
                                     runtime-supported"
                                ))
                            })? as i128;
                        let bytes = (a as i128).wrapping_sub(b as i128);
                        if bytes % size != 0 {
                            None
                        } else {
                            Some(bytes / size)
                        }
                    }
                    _ => {
                        let x = self.deref_value(frames, &vals[0])?;
                        let y = self.deref_value(frames, &vals[1])?;
                        self.pointer_distance(&x, &y, generic_args)?
                    }
                };
                match d {
                    Some(d) if unsigned && d < 0 => Err(unsupported(
                        "ptr_offset_from_unsigned with a negative distance is not \
                         runtime-supported"
                            .to_string(),
                    )),
                    Some(d) => Ok(if unsigned {
                        Value::Uint(d as u128)
                    } else {
                        Value::Int(d)
                    }),
                    None => Err(unsupported(
                        "ptr_offset_from between unrelated pointers is not runtime-supported"
                            .to_string(),
                    )),
                }
            }
            Builtin::MemZeroed => {
                arity(0)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                match zeroed_value_for_tag(tag) {
                    Some(value) => Ok(value),
                    None => Err(unsupported(format!(
                        "mem::zeroed for {tag} is not runtime-supported"
                    ))),
                }
            }
            Builtin::IntBitOp(op) => {
                arity(1)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (bits, signed) = int_tag(tag).ok_or_else(|| {
                    unsupported(format!("bit intrinsic for {tag} is not runtime-supported"))
                })?;
                let pattern = match vals[0] {
                    Value::Int(v) => (v as u128) & bit_mask(bits),
                    Value::Uint(v) => v & bit_mask(bits),
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "integer",
                            other.kind_name(),
                        ));
                    }
                };
                let result = match op {
                    BitOp::CtPop => pattern.count_ones() as u128,
                    BitOp::Ctlz => (pattern.leading_zeros() - (128 - bits)) as u128,
                    BitOp::Cttz => {
                        if pattern == 0 {
                            bits as u128
                        } else {
                            pattern.trailing_zeros() as u128
                        }
                    }
                    BitOp::BitReverse => pattern.reverse_bits() >> (128 - bits),
                    BitOp::Bswap => pattern.swap_bytes() >> (128 - bits),
                };
                Ok(if signed {
                    Value::Int(sign_extend(result, bits))
                } else {
                    Value::Uint(result)
                })
            }
            Builtin::SaturatingArith(op) => {
                arity(2)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (bits, signed) = int_tag(tag).ok_or_else(|| {
                    unsupported(format!(
                        "saturating arithmetic for {tag} is not runtime-supported"
                    ))
                })?;
                if signed {
                    let (a, b) = (as_i128(&vals[0], name)?, as_i128(&vals[1], name)?);
                    let max = sign_extend(bit_mask(bits) >> 1, bits);
                    let min = -max - 1;
                    let r = match op {
                        SatOp::Add => {
                            if b > 0 && a > max - b {
                                max
                            } else if b < 0 && a < min - b {
                                min
                            } else {
                                a + b
                            }
                        }
                        SatOp::Sub => {
                            if b < 0 && a > max + b {
                                max
                            } else if b > 0 && a < min + b {
                                min
                            } else {
                                a - b
                            }
                        }
                    };
                    Ok(Value::Int(r))
                } else {
                    let (a, b) = (as_u128(&vals[0], name)?, as_u128(&vals[1], name)?);
                    let max = bit_mask(bits);
                    let r = match op {
                        SatOp::Add => a.checked_add(b).map_or(max, |sum| sum.min(max)),
                        SatOp::Sub => a.saturating_sub(b),
                    };
                    Ok(Value::Uint(r))
                }
            }
            Builtin::ArithOffset { wrapping } => {
                arity(2)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (size, _) = self.layout_of(tag).ok_or_else(|| {
                    unsupported(format!(
                        "pointer offset for {tag} is not runtime-supported"
                    ))
                })?;
                let count = match &vals[1] {
                    Value::Int(v) => *v,
                    Value::Uint(v) => i128::try_from(*v).map_err(|_| {
                        unsupported("pointer offset count does not fit in an isize")
                    })?,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "pointer offset count",
                            "integer count",
                            other.kind_name(),
                        ))
                    }
                };
                self.ptr_offset_value(frames, vals[0].clone(), count, Some(size as u64), wrapping)
            }
            Builtin::AtomicLoad => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                // A wider-than-byte access through a pointer into
                // byte storage (an unaligned / type-punned load)
                // composes the stored bytes.
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                if let Some((bits, signed)) = int_tag(tag) {
                    if let Some(value) =
                        self.read_punned_int(frames, &target, bits, signed)?
                    {
                        return Ok(value);
                    }
                }
                let current = self.read_resolved(frames, &target)?;
                Ok(unwrap_wrapper(&current).clone())
            }
            Builtin::AtomicStore => {
                arity(2)?;
                let target = self.deref_value(frames, &vals[0])?;
                let value = vals.into_iter().nth(1).expect("arity checked");
                // The store matching the punned load above.
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                if let Some((bits, _)) = int_tag(tag) {
                    if self.write_punned_int(frames, &target, bits, &value)? {
                        return Ok(Value::Unit);
                    }
                }
                self.atomic_write(frames, &target, value)?;
                Ok(Value::Unit)
            }
            Builtin::AtomicXadd | Builtin::AtomicXsub => {
                arity(2)?;
                let sub = kind == Builtin::AtomicXsub;
                let target = self.deref_value(frames, &vals[0])?;
                let current = self.read_resolved(frames, &target)?;
                let old = unwrap_wrapper(&current).clone();
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (bits, signed) = int_tag(tag).ok_or_else(|| {
                    unsupported(format!("{name} on `{tag}` is not runtime-supported"))
                })?;
                let new = match (&old, &vals[1]) {
                    (Value::Uint(a), Value::Uint(b)) => {
                        let mask = if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 };
                        Value::Uint(if sub {
                            a.wrapping_sub(*b) & mask
                        } else {
                            a.wrapping_add(*b) & mask
                        })
                    }
                    (Value::Int(a), Value::Int(b)) => {
                        let raw = if sub {
                            a.wrapping_sub(*b)
                        } else {
                            a.wrapping_add(*b)
                        };
                        // Re-wrap at the operand's width.
                        let shift = 128 - bits;
                        Value::Int((raw << shift) >> shift)
                    }
                    _ => {
                        return Err(InterpError::type_mismatch(
                            name,
                            "matching integer operands",
                            format!(
                                "{} and {}",
                                old.kind_name(),
                                vals[1].kind_name()
                            ),
                        ))
                    }
                };
                let _ = signed;
                self.atomic_write(frames, &target, new)?;
                Ok(old)
            }
            Builtin::AtomicFetchOp(op) => {
                arity(2)?;
                let target = self.deref_value(frames, &vals[0])?;
                let current = self.read_resolved(frames, &target)?;
                let old = unwrap_wrapper(&current).clone();
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (bits, signed) = int_tag(tag).ok_or_else(|| {
                    unsupported(format!("{name} on `{tag}` is not runtime-supported"))
                })?;
                let mask = bit_mask(bits);
                let cur = as_u128(&old, name)? & mask;
                let operand = as_u128(&vals[1], name)? & mask;
                let new = match op {
                    FetchOp::Add => cur.wrapping_add(operand) & mask,
                    FetchOp::Sub => cur.wrapping_sub(operand) & mask,
                    FetchOp::And => cur & operand,
                    FetchOp::Or => cur | operand,
                    FetchOp::Xor => cur ^ operand,
                    FetchOp::Nand => !(cur & operand) & mask,
                    FetchOp::Max => {
                        if sign_extend(cur, bits) >= sign_extend(operand, bits) {
                            cur
                        } else {
                            operand
                        }
                    }
                    FetchOp::Min => {
                        if sign_extend(cur, bits) <= sign_extend(operand, bits) {
                            cur
                        } else {
                            operand
                        }
                    }
                    FetchOp::UMax => cur.max(operand),
                    FetchOp::UMin => cur.min(operand),
                };
                let new_value = if signed {
                    Value::Int(sign_extend(new, bits))
                } else {
                    Value::Uint(new)
                };
                self.atomic_write(frames, &target, new_value)?;
                Ok(old)
            }
            Builtin::PtrCopy => {
                arity(3)?;
                let mut vals = vals.into_iter();
                let src = vals.next().expect("arity checked");
                let dst = vals.next().expect("arity checked");
                let count = vals.next().expect("arity checked");
                // The element type, from the caller's declared
                // pointer types, as on the intrinsic statement
                // path.
                let elem = args
                    .first()
                    .and_then(|a| crate::interp::operand_pointee_ty(body, a))
                    .or_else(|| {
                        args.get(1)
                            .and_then(|a| crate::interp::operand_pointee_ty(body, a))
                    });
                self.exec_copy_nonoverlapping(frames, src, dst, count, elem.as_ref())?;
                Ok(Value::Unit)
            }
            Builtin::CompareBytes => {
                arity(3)?;
                let n = match &vals[2] {
                    Value::Uint(n) => usize::try_from(*n).map_err(|_| {
                        InterpError::type_mismatch(
                            "compare_bytes count",
                            "count that fits in usize",
                            format!("{n}"),
                        )
                    })?,
                    Value::Int(n) if *n >= 0 => *n as usize,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "compare_bytes count",
                            "unsigned integer",
                            other.kind_name(),
                        ));
                    }
                };
                let a = self.read_byte_range(frames, &vals[0], n)?;
                let b = self.read_byte_range(frames, &vals[1], n)?;
                let ord = a.cmp(&b);
                Ok(Value::Int(match ord {
                    std::cmp::Ordering::Less => -1,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                }))
            }
            Builtin::SimdOp(op) => self.exec_simd(op, frames, generic_args, &vals),
            Builtin::RustAlloc { zeroed } => {
                arity(2)?;
                let size = match &vals[0] {
                    Value::Uint(n) => *n,
                    Value::Int(n) if *n >= 0 => *n as u128,
                    other => {
                        return Err(InterpError::type_mismatch(
                            name,
                            "allocation size",
                            other.kind_name(),
                        ))
                    }
                };
                // The uninitialised form stores nothing per byte,
                // so only the zeroed form has a materialisation
                // limit; past it the shim's failure signal (the
                // null pointer) is the faithful answer.
                if zeroed && size > (1 << 28) {
                    return Ok(Value::Ref(RefTarget::Null));
                }
                let content = if zeroed {
                    Value::Bytes(vec![0u8; size as usize])
                } else {
                    // An element tuple that grows on write (see
                    // the allocation arm of write_resolved): the
                    // blocks elements have no value-level count
                    // until they are written, and unwritten slots
                    // read back as uninitialized.
                    Value::Tuple(Vec::new())
                };
                let id = self.alloc_insert(content);
                self.mark_elem_block(id);
                Ok(Value::Ref(RefTarget::Alloc {
                    id,
                    projection: Vec::new(),
                }))
            }
            Builtin::RustDealloc => {
                arity(3)?;
                let ptr = unwrap_wrapper(&vals[0]);
                if let Value::Ref(RefTarget::Alloc { id, projection }) = ptr {
                    if crate::interp::projection_is_base(projection) {
                        self.free_alloc(*id);
                    }
                }
                Ok(Value::Unit)
            }
            Builtin::RustRealloc => {
                arity(4)?;
                let new_size = match &vals[3] {
                    Value::Uint(n) => *n,
                    Value::Int(n) if *n >= 0 => *n as u128,
                    other => {
                        return Err(InterpError::type_mismatch(
                            name,
                            "allocation size",
                            other.kind_name(),
                        ))
                    }
                };
                let ptr = unwrap_wrapper(&vals[0]).clone();
                let content = match &ptr {
                    Value::Ref(RefTarget::Alloc { id, projection })
                        if crate::interp::projection_is_base(projection) =>
                    {
                        let old = self.alloc_base(*id)?;
                        self.free_alloc(*id);
                        old
                    }
                    _ => {
                        let _ = new_size;
                        Value::Uninit
                    }
                };
                let id = self.alloc_insert(content);
                self.mark_elem_block(id);
                Ok(Value::Ref(RefTarget::Alloc {
                    id,
                    projection: Vec::new(),
                }))
            }
            Builtin::LibcMalloc => {
                arity(1)?;
                let size = alloc_size_arg(name, &vals[0])?;
                let id = self.alloc_insert(Value::Tuple(Vec::new()));
                self.mark_elem_block(id);
                Ok(Value::Ref(RefTarget::Alloc {
                    id,
                    projection: Vec::new(),
                }))
            }
            Builtin::LibcCalloc => {
                arity(2)?;
                let count = alloc_size_arg(name, &vals[0])?;
                let size = alloc_size_arg(name, &vals[1])?;
                let total = count.saturating_mul(size);
                if total > (1 << 28) {
                    return Ok(Value::Ref(RefTarget::Null));
                }
                let id = self.alloc_insert(Value::Bytes(vec![0u8; total as usize]));
                self.mark_elem_block(id);
                Ok(Value::Ref(RefTarget::Alloc {
                    id,
                    projection: Vec::new(),
                }))
            }
            Builtin::LibcFree => {
                arity(1)?;
                let ptr = unwrap_wrapper(&vals[0]);
                if let Value::Ref(RefTarget::Alloc { id, projection }) = ptr {
                    if crate::interp::projection_is_base(projection) {
                        self.free_alloc(*id);
                    }
                }
                Ok(Value::Unit)
            }
            Builtin::LibcRealloc => {
                arity(2)?;
                let new_size = alloc_size_arg(name, &vals[1])?;
                let ptr = unwrap_wrapper(&vals[0]).clone();
                let content = match &ptr {
                    Value::Ref(RefTarget::Alloc { id, projection })
                        if crate::interp::projection_is_base(projection) =>
                    {
                        let old = self.alloc_base(*id)?;
                        self.free_alloc(*id);
                        old
                    }
                    _ => {
                        let _ = new_size;
                        Value::Uninit
                    }
                };
                let id = self.alloc_insert(content);
                self.mark_elem_block(id);
                Ok(Value::Ref(RefTarget::Alloc {
                    id,
                    projection: Vec::new(),
                }))
            }
            Builtin::LibcPosixMemalign => {
                arity(3)?;
                let size = alloc_size_arg(name, &vals[2])?;
                let _ = size;
                let id = self.alloc_insert(Value::Tuple(Vec::new()));
                self.mark_elem_block(id);
                let block = Value::Ref(RefTarget::Alloc {
                    id,
                    projection: Vec::new(),
                });
                let target = self.deref_value(frames, &vals[0])?;
                self.write_resolved(frames, &target, block)?;
                Ok(Value::Int(0))
            }
            Builtin::AtomicCxchg => {
                arity(3)?;
                // Single-threaded: nothing can intervene between
                // the read and the conditional write, and a weak
                // exchange's spurious failure is not simulated, so
                // `atomic_cxchgweak` is the same operation.
                let target = self.deref_value(frames, &vals[0])?;
                let current = self.read_resolved(frames, &target)?;
                let success = self.atomic_value_eq(frames, &current, &vals[1])?;
                if success {
                    self.atomic_write(frames, &target, vals[2].clone())?;
                }
                // The exchanged-out value the caller receives is the
                // cell's scalar, not the wrapper aggregate the cell
                // place holds.
                let previous = unwrap_wrapper(&current).clone();
                Ok(Value::Tuple(vec![previous, Value::Bool(success)]))
            }
            Builtin::AtomicXchg => {
                arity(2)?;
                let target = self.deref_value(frames, &vals[0])?;
                let current = self.read_resolved(frames, &target)?;
                let previous = unwrap_wrapper(&current).clone();
                self.atomic_write(frames, &target, vals[1].clone())?;
                Ok(previous)
            }
            Builtin::AtomicFence => {
                arity(0)?;
                // A fence orders other threads' accesses; there
                // are none.
                Ok(Value::Unit)
            }
            Builtin::AssertValidity => {
                arity(0)?;
                // The asserted property is a fact about the type
                // the conversion already produced; nothing about
                // the running program can still violate it.
                Ok(Value::Unit)
            }
            Builtin::WriteBytes => {
                arity(3)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                // The fill argument is a `u8`; take its low eight
                // bits, which are the byte either way it is stored.
                let byte = match &vals[1] {
                    Value::Uint(v) => *v as u8,
                    Value::Int(v) => *v as u8,
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "fill byte (u8)",
                            other.kind_name(),
                        ));
                    }
                };
                let fill = byte_fill_value(tag, byte)
                    .or_else(|| self.wrapped_byte_fill(tag, byte, 0))
                    .ok_or_else(|| {
                        unsupported(format!(
                            "write_bytes for {tag} is not runtime-supported"
                        ))
                    })?;
                let n = element_count(&vals[2], name)?;
                if n == 0 {
                    // A zero-count fill touches no memory, so the
                    // pointer is never dereferenced.
                    return Ok(Value::Unit);
                }
                let target = self.deref_value(frames, &vals[0])?;
                // Filling not-yet-initialized memory is the
                // intrinsic's canonical use (a `MaybeUninit`
                // buffer being initialized): the region has no
                // prior value to patch element by element, so
                // materialize exactly the filled region in one
                // write. The pointer denotes the start of the
                // uninitialized region in that shape (a fresh
                // buffer's base pointer).
                // A target ending in an index projection denotes
                // one element of an existing aggregate (the shape
                // an array-to-element pointer cast produces), not
                // the start of an unmaterialized region: filling it
                // goes through the sequence access below, which
                // writes the elements at their own positions. (A
                // single-element fill writes the denoted element
                // itself either way, so it keeps this fast path.)
                let ends_with_index = match &target {
                    Resolved::Place { projection, .. }
                    | Resolved::Static { projection, .. }
                    | Resolved::Alloc { projection, .. } => {
                        matches!(projection.last(), Some(RefProj::Index(_)))
                    }
                    Resolved::Value(_, projection) => {
                        matches!(projection.last(), Some(RefProj::Index(_)))
                    }
                };
                if matches!(
                    self.read_resolved(frames, &target),
                    Err(InterpError::UninitializedRead { .. })
                ) && (n == 1 || !ends_with_index)
                {
                    if n == 1 {
                        self.write_resolved(frames, &target, fill)?;
                    } else {
                        self.write_resolved(frames, &target, Value::Tuple(vec![fill; n]))?;
                    }
                    return Ok(Value::Unit);
                }
                // As in the element copy: an unreadable destination
                // is a buffer being filled, which the indexed path
                // can create elements in.
                let dst = self.read_resolved(frames, &target);
                // A one-element fill denotes the target whole when
                // the target is not a sequence (a scalar or wrapper
                // place: the fill is its new value), and also when
                // the target is an array whose scalar-leaf count
                // the fill's type exactly matches — the intrinsic's
                // type is then the target's own type (an array
                // filled in one operation, the shape `mem::zeroed`
                // takes under optimisation), and writing the fill
                // as element 0 of a sequence would nest it inside
                // its own first element. Any other fill denotes one
                // element of the target sequence.
                let whole = n == 1
                    && match &dst {
                        Ok(v @ Value::Tuple(_)) => {
                            value_leaf_count(v).is_some()
                                && value_leaf_count(v) == tag_leaf_count(tag)
                        }
                        Ok(Value::FatRef { .. } | Value::Bytes(_) | Value::Str(_)) => {
                            false
                        }
                        Ok(_) => true,
                        Err(_) => false,
                    };
                if whole {
                    self.write_resolved(frames, &target, fill)?;
                } else {
                    for i in 0..n {
                        self.write_seq_at(frames, &target, i, fill.clone())?;
                    }
                }
                Ok(Value::Unit)
            }
            Builtin::FloatMethod { f32, op } => exec_float_method(name, f32, op, &vals),
            Builtin::Float128Method { op } => exec_float128_method(name, op, &vals),
            Builtin::Float16Method { op } => exec_float16_method(name, op, &vals),
        }
    }

    /// The fill value of a `write_bytes` over a pointee type
    /// [`byte_fill_value`] does not cover, resolved through the
    /// wrappers the byte pattern passes through unchanged:
    ///
    /// * `MaybeUninit<T>`: a byte fill over the union view writes
    ///   `T`'s bytes; the region's canonical value is `T`'s fill
    ///   (the readers of such a region read it as `T` — a raw
    ///   table's control bytes are allocated as
    ///   `MaybeUninit<Tag>` and read as `Tag`).
    /// * a one-field struct (per the program's recorded shape of
    ///   the instance): the struct value holding the field's
    ///   fill — the shape an aggregate construction of the
    ///   wrapper produces (`Tag` over its `u8`).
    ///
    /// Anything else stays unsupported at the call site.
    fn wrapped_byte_fill(&self, tag: &str, byte: u8, depth: u32) -> Option<Value> {
        if depth > 4 {
            return None;
        }
        let tag = tag.trim();
        if let Some(inner) = generic_first_arg(tag, "std::mem::MaybeUninit<")
            .or_else(|| generic_first_arg(tag, "core::mem::MaybeUninit<"))
        {
            return byte_fill_value(inner, byte)
                .or_else(|| self.wrapped_byte_fill(inner, byte, depth + 1));
        }
        let shape = self.program().adt_debug_shapes.get(tag)?;
        if shape.kind != mir::AdtShapeKind::Struct || shape.variants.len() != 1 {
            return None;
        }
        let variant = &shape.variants[0];
        if variant.fields.len() != 1 {
            return None;
        }
        let field_ty = variant.fields[0].ty.as_str();
        let field_fill = byte_fill_value(field_ty, byte)
            .or_else(|| self.wrapped_byte_fill(field_ty, byte, depth + 1))?;
        Some(Value::Enum {
            variant: 0,
            fields: vec![field_fill],
        })
    }

    /// Write an atomic cell's new scalar, preserving the wrapper
    /// aggregates the cell place holds (`UnsafeCell` inside the
    /// atomic struct): the pointer names the wrapper place, so a
    /// bare scalar write would replace the aggregate and later
    /// whole-place reads of the cell would see the wrong shape.
    /// When the current contents cannot be read (an uninitialised
    /// cell), the value is written as-is.
    /// Evaluate `FromStr::from_str` for a primitive `Self`
    /// type. The result is the `Result` aggregate: variant 0
    /// (`Ok`) with the parsed value, variant 1 (`Err`) with
    /// the error value core constructs — a one-field struct
    /// holding the error-kind enum.
    fn exec_from_str(
        &self,
        tag: &str,
        text: &str,
        name: &str,
    ) -> InterpResult<Value> {
        fn ok(v: Value) -> Value {
            Value::Enum {
                variant: 0,
                fields: vec![v],
            }
        }
        fn err(kind: u32) -> Value {
            Value::Enum {
                variant: 1,
                fields: vec![Value::Enum {
                    variant: 0,
                    fields: vec![Value::Enum {
                        variant: kind,
                        fields: vec![],
                    }],
                }],
            }
        }
        if let Some((bits, signed)) = int_tag(tag) {
            use std::num::IntErrorKind as K;
            let parsed: Result<u128, u32> = if signed {
                macro_rules! go {
                    ($t:ty) => {
                        match text.parse::<$t>() {
                            Ok(v) => Ok(v as i128 as u128),
                            Err(e) => Err(match e.kind() {
                                K::Empty => 0,
                                K::InvalidDigit => 1,
                                K::PosOverflow => 2,
                                K::NegOverflow => 3,
                                K::Zero => 4,
                                _ => 1,
                            }),
                        }
                    };
                }
                match bits {
                    8 => go!(i8),
                    16 => go!(i16),
                    32 => go!(i32),
                    64 => go!(i64),
                    _ => go!(i128),
                }
            } else {
                macro_rules! go {
                    ($t:ty) => {
                        match text.parse::<$t>() {
                            Ok(v) => Ok(v as u128),
                            Err(e) => Err(match e.kind() {
                                K::Empty => 0,
                                K::InvalidDigit => 1,
                                K::PosOverflow => 2,
                                K::NegOverflow => 3,
                                K::Zero => 4,
                                _ => 1,
                            }),
                        }
                    };
                }
                match bits {
                    8 => go!(u8),
                    16 => go!(u16),
                    32 => go!(u32),
                    64 => go!(u64),
                    _ => go!(u128),
                }
            };
            return Ok(match parsed {
                Ok(p) => ok(if signed {
                    Value::Int(sign_extend(p, bits))
                } else {
                    Value::Uint(p)
                }),
                Err(kind) => err(kind),
            });
        }
        match tag {
            "f32" | "f64" => {
                let parsed = if tag == "f32" {
                    text.parse::<f32>().map(|v| Value::Float(f64::from(v)))
                } else {
                    text.parse::<f64>().map(Value::Float)
                };
                Ok(match parsed {
                    Ok(v) => ok(v),
                    // `ParseFloatError` distinguishes only the
                    // empty input from everything else.
                    Err(_) => err(if text.is_empty() { 0 } else { 1 }),
                })
            }
            "bool" => Ok(match text.parse::<bool>() {
                Ok(v) => ok(Value::Bool(v)),
                // `ParseBoolError` is a fieldless struct.
                Err(_) => Value::Enum {
                    variant: 1,
                    fields: vec![Value::Enum {
                        variant: 0,
                        fields: vec![],
                    }],
                },
            }),
            "char" => Ok(match text.parse::<char>() {
                Ok(v) => ok(Value::Uint(u128::from(v as u32))),
                // `ParseCharError` distinguishes only the empty
                // input from too-many-chars.
                Err(_) => err(if text.is_empty() { 0 } else { 1 }),
            }),
            _ => Err(unsupported(format!(
                "{name}: from_str for {tag} is not runtime-supported"
            ))),
        }
    }

    /// The equality an atomic compare-exchange performs: the
    /// pointee's current value against the expected one.
    /// Pointer-shaped values compare by identity — the synthetic
    /// address, the same identity `PtrToInt` exposes — and every
    /// other value by the semantic (derived) equality.
    fn atomic_value_eq(
        &self,
        frames: &[Frame],
        current: &Value,
        expected: &Value,
    ) -> InterpResult<bool> {
        let current = unwrap_wrapper(current);
        let expected = unwrap_wrapper(expected);
        match (synthetic_address(current), synthetic_address(expected)) {
            (Some(a), Some(b)) => Ok(a == b),
            _ => self.values_equal(frames, current, expected),
        }
    }
    /// Write an atomic cell's new scalar, preserving the wrapper
    /// aggregates the cell place holds (`UnsafeCell` inside the
    /// atomic struct): the pointer names the wrapper place, so a
    /// bare scalar write would replace the aggregate and later
    /// whole-place reads of the cell would see the wrong shape.
    /// When the current contents cannot be read (an uninitialised
    /// cell), the value is written as-is.
    fn atomic_write(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        value: Value,
    ) -> InterpResult<()> {
        let value = match self.read_resolved(frames, target) {
            Ok(current) => rewrap_like(&current, value),
            Err(_) => value,
        };
        self.write_resolved(frames, target, value)
    }
    /// The signed distance, in elements of the pointee type the
    /// generic arguments name, between two resolved pointer
    /// targets, when both denote positions in the same place,
    /// static or allocation. Positions are compared through their
    /// projections: the interpreter's offset machinery leaves an
    /// element-aligned position as a trailing `Index` step, or —
    /// when the position cannot be resolved structurally — as a
    /// trailing byte displacement, so a trailing step of either
    /// shape is read back as the position's element index.
    /// `None` when the two targets are unrelated (their distance
    /// is then meaningless in the guest too).
/// A structural fingerprint of a value (an FNV-1a walk over
/// its shape and scalar leaves), used only as an identity key
/// for pointer-distance purposes: two pointers whose base
/// values share a fingerprint are treated as pointing into the
/// same allocation.
fn value_fingerprint(value: &Value) -> u64 {
    fn mix(h: &mut u64, x: u64) {
        *h ^= x;
        *h = h.wrapping_mul(0x100_0000_01b3);
    }
    fn walk(h: &mut u64, value: &Value) {
        match value {
            Value::Unit => mix(h, 1),
            Value::Bool(b) => {
                mix(h, 2);
                mix(h, *b as u64);
            }
            Value::Int(v) => {
                mix(h, 3);
                mix(h, *v as u64);
            }
            Value::Uint(v) => {
                mix(h, 4);
                mix(h, *v as u64);
            }
            Value::Float(v) => {
                mix(h, 5);
                mix(h, v.to_bits());
            }
            Value::Str(s) => {
                mix(h, 6);
                mix(h, s.len() as u64);
                for b in s.as_bytes() {
                    mix(h, *b as u64);
                }
            }
            Value::Bytes(b) => {
                mix(h, 7);
                mix(h, b.len() as u64);
                for byte in b {
                    mix(h, *byte as u64);
                }
            }
            Value::Tuple(items) => {
                mix(h, 8);
                mix(h, items.len() as u64);
                for item in items {
                    walk(h, item);
                }
            }
            Value::Enum { variant, fields } => {
                mix(h, 9);
                mix(h, *variant as u64);
                mix(h, fields.len() as u64);
                for field in fields {
                    walk(h, field);
                }
            }
            Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. } => {
                mix(h, 11);
                if let Some(addr) = crate::interp::synthetic_address(value) {
                    mix(h, addr as u64);
                    mix(h, (addr >> 64) as u64);
                }
            }
            Value::FnPtr(name) => {
                mix(h, 12);
                for b in name.as_bytes() {
                    mix(h, *b as u64);
                }
            }
            Value::Closure { def, .. } => {
                mix(h, 13);
                for b in def.as_bytes() {
                    mix(h, *b as u64);
                }
            }
            other => {
                // The remaining shapes (uninitialized slots,
                // formatting arguments, SIMD vectors, ...) do
                // not underlie slice pointers in practice; a
                // Debug-shape hash keeps the key deterministic.
                mix(h, 14);
                let text = format!("{other:?}");
                mix(h, text.len() as u64);
                for b in text.as_bytes() {
                    mix(h, *b as u64);
                }
            }
        }
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    walk(&mut h, value);
    h
}

    fn pointer_distance(
        &self,
        x: &Resolved,
        y: &Resolved,
        generic_args: &[String],
    ) -> InterpResult<Option<i128>> {
        let tag = generic_args.first().map(String::as_str).unwrap_or("");
        let elem_size = self.layout_of(tag).map(|(size, _)| size);
        // Split a target into its base identity and the element
        // index its trailing projection step denotes, if any.
        let split = |target: &Resolved| -> Option<(String, i128)> {
            let (base, projection): (String, &[RefProj]) = match target {
                Resolved::Place {
                    frame_pos,
                    local,
                    projection,
                } => (format!("place:{frame_pos}:{local}"), projection),
                Resolved::Static { name, projection } => (format!("static:{name}"), projection),
                Resolved::Alloc { id, projection } => (format!("alloc:{id}"), projection),
                // Two pointers into the same value tree (a
                // cloned copy of one base value -- the shape a
                // slice iterator's start/end pair takes, both
                // derived from one fat reference) are the same
                // allocation for distance purposes: the base
                // value's structural fingerprint is the identity.
                // Pointers into different allocations that happen
                // to hold equal values would be conflated, but a
                // cross-allocation `offset_from` is undefined
                // behaviour in the guest, so no defined program
                // can observe the difference.
                Resolved::Value(base, projection) => {
                    (format!("value:{:016x}", Self::value_fingerprint(base)), projection)
                }
            };
            let key = |steps: &[RefProj]| {
                steps
                    .iter()
                    .map(|p| format!("{p:?}"))
                    .collect::<Vec<_>>()
                    .join("/")
            };
            match projection.split_last() {
                Some((RefProj::Index(i), rest)) => {
                    Some((format!("{base}|{}", key(rest)), *i as i128))
                }
                Some((RefProj::ByteOffset(d), rest)) => {
                    let size = elem_size.filter(|sz| *sz > 0)? as i128;
                    if d % size != 0 {
                        return None;
                    }
                    Some((format!("{base}|{}", key(rest)), d / size))
                }
                _ => Some((format!("{base}|{}", key(projection)), 0)),
            }
        };
        Ok(match (split(x), split(y)) {
            (Some((bx, ix)), Some((by, iy))) if bx == by => Some(ix - iy),
            _ => None,
        })
    }
    /// The lane shape of a vector type: (lane width in bits,
    /// signed lanes, float width in bits when the lanes are
    /// floats). Taken from the element type in the tag when it
    /// names one, else from the vector's layout and lane count,
    /// else from the lanes' own representation.
    fn simd_lane_info(&self, tag: &str, lanes: &[Value]) -> (u32, bool, Option<u32>) {
        if let Some(elem) = vector_elem_tag(tag) {
            if let Some((bits, signed)) = int_tag(elem) {
                return (bits, signed, None);
            }
            match elem {
                "f16" => return (16, true, Some(16)),
                "f32" => return (32, true, Some(32)),
                "f64" => return (64, true, Some(64)),
                "bool" => return (8, false, None),
                _ => {}
            }
        }
        if let Some((size, _)) = self.layout_of(tag) {
            if !lanes.is_empty() && size % lanes.len() as u128 == 0 {
                let bits = ((size / lanes.len() as u128 * 8) as u32).min(128);
                return match lanes.first() {
                    Some(Value::Float(_)) => (bits, true, Some(bits)),
                    Some(Value::Int(_)) => (bits, true, None),
                    _ => (bits, false, None),
                };
            }
        }
        match lanes.first() {
            Some(Value::Float(_)) => (64, true, Some(64)),
            Some(Value::Int(_)) => (128, true, None),
            _ => (128, false, None),
        }
    }

    /// One `std::intrinsics::simd::*` operation over the lanes
    /// of its vector arguments. Integer lanes wrap at the lane
    /// width; a shift amount is taken modulo the lane width (the
    /// hardware behaviour); a zero integer divisor lane fails the
    /// guest the way a scalar division by zero does.
    fn exec_simd(
        &self,
        op: SimdOp,
        frames: &mut Vec<Frame>,
        generic_args: &[String],
        vals: &[Value],
    ) -> InterpResult<Value> {
        use SimdOp as S;
        let tag0 = generic_args.first().map(String::as_str).unwrap_or("");
        match op {
            S::Add
            | S::Sub
            | S::Mul
            | S::Div
            | S::Rem
            | S::And
            | S::Or
            | S::Xor
            | S::Shl
            | S::Shr
            | S::SaturatingAdd
            | S::SaturatingSub
            | S::CarrylessMul
            | S::FunnelShl
            | S::FunnelShr
            | S::MaximumNumber
            | S::MinimumNumber => {
                let a = simd_lanes(&vals[0])?;
                let b = simd_lanes(&vals[1])?;
                let info = self.simd_lane_info(tag0, &a);
                let mut out = Vec::with_capacity(a.len());
                for (x, y) in a.iter().zip(b.iter()) {
                    out.push(simd_binop(op, x, y, info)?);
                }
                simd_wrap(&vals[0], out)
            }
            S::Eq | S::Ne | S::Lt | S::Le | S::Gt | S::Ge => {
                let a = simd_lanes(&vals[0])?;
                let b = simd_lanes(&vals[1])?;
                let info = self.simd_lane_info(tag0, &a);
                let mut out = Vec::with_capacity(a.len());
                for (x, y) in a.iter().zip(b.iter()) {
                    let yes = simd_compare(op, x, y, info)?;
                    out.push(Value::Int(if yes { -1 } else { 0 }));
                }
                simd_wrap(&vals[0], out)
            }
            S::Neg
            | S::Fabs
            | S::Ceil
            | S::Floor
            | S::Trunc
            | S::Round
            | S::RoundTiesEven
            | S::Fsqrt
            | S::Fexp
            | S::Fexp2
            | S::Flog
            | S::Flog2
            | S::Flog10
            | S::Fsin
            | S::Fcos
            | S::Bswap
            | S::BitReverse
            | S::Ctlz
            | S::Ctpop
            | S::Cttz => {
                let a = simd_lanes(&vals[0])?;
                let info = self.simd_lane_info(tag0, &a);
                let mut out = Vec::with_capacity(a.len());
                for x in a.iter() {
                    out.push(simd_unop(op, x, info)?);
                }
                simd_wrap(&vals[0], out)
            }
            S::Fma => {
                let a = simd_lanes(&vals[0])?;
                let b = simd_lanes(&vals[1])?;
                let c = simd_lanes(&vals[2])?;
                let info = self.simd_lane_info(tag0, &a);
                let mut out = Vec::with_capacity(a.len());
                for ((x, y), z) in a.iter().zip(b.iter()).zip(c.iter()) {
                    let (x, y, z) = (
                        simd_as_f64(x, "simd lane")?,
                        simd_as_f64(y, "simd lane")?,
                        simd_as_f64(z, "simd lane")?,
                    );
                    let r = if info.2 == Some(32) {
                        (x as f32).mul_add(y as f32, z as f32) as f64
                    } else {
                        x.mul_add(y, z)
                    };
                    out.push(Value::Float(simd_round_float(r, info.2)));
                }
                simd_wrap(&vals[0], out)
            }
            S::Splat => {
                // The generic arguments name the element type and
                // the vector type, in one order or the other; the
                // scalar one is the element. Raw pointers are
                // scalar elements too (a vector of pointers is a
                // supported `simd_splat` target).
                let scalar = generic_args
                    .iter()
                    .find(|t| {
                        let t = t.as_str();
                        int_tag(t).is_some()
                            || matches!(t, "f16" | "f32" | "f64" | "bool" | "char")
                            || t.starts_with("*const ")
                            || t.starts_with("*mut ")
                    })
                    .ok_or_else(|| {
                        unsupported("simd_splat without an element type is not runtime-supported")
                    })?;
                let vector = generic_args
                    .iter()
                    .find(|t| t.as_str() != scalar.as_str())
                    .ok_or_else(|| {
                        unsupported("simd_splat without a vector type is not runtime-supported")
                    })?;
                let count = match simd_tag_lane_count(vector) {
                    Some(n) => n,
                    None => {
                        let (vsize, _) = self.layout_of(vector).ok_or_else(|| {
                            unsupported("simd_splat vector size is not runtime-supported")
                        })?;
                        let (esize, _) = self.layout_of(scalar).ok_or_else(|| {
                            unsupported("simd_splat element size is not runtime-supported")
                        })?;
                        if esize == 0 || vsize % esize != 0 {
                            return Err(unsupported(
                                "simd_splat lane count is not runtime-supported",
                            ));
                        }
                        (vsize / esize) as usize
                    }
                };
                let lanes = vec![vals[0].clone(); count];
                if vector.trim_start().starts_with('[') {
                    Ok(Value::Tuple(lanes))
                } else {
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![Value::Tuple(lanes)],
                    })
                }
            }
            S::Extract => {
                let lanes = simd_lanes(&vals[0])?;
                let idx = simd_index(&vals[1])?;
                lanes.get(idx).cloned().ok_or_else(|| InterpError::Panic {
                    message: format!("simd lane index {idx} out of range"),
                    payload: None,
                })
            }
            S::Insert => {
                let mut lanes = simd_lanes(&vals[0])?;
                let idx = simd_index(&vals[1])?;
                if idx >= lanes.len() {
                    return Err(InterpError::Panic {
                        message: format!("simd lane index {idx} out of range"),
                        payload: None,
                    });
                }
                lanes[idx] = vals[2].clone();
                simd_wrap(&vals[0], lanes)
            }
            S::Shuffle => {
                let a = simd_lanes(&vals[0])?;
                let b = simd_lanes(&vals[1])?;
                let idx = simd_lanes(&vals[2])?;
                let mut out = Vec::with_capacity(idx.len());
                for i in idx.iter() {
                    let i = simd_index(i)?;
                    let lane = if i < a.len() {
                        a.get(i)
                    } else {
                        b.get(i - a.len())
                    };
                    out.push(lane.cloned().ok_or_else(|| InterpError::Panic {
                        message: format!("simd shuffle index {i} out of range"),
                        payload: None,
                    })?);
                }
                simd_wrap(&vals[0], out)
            }
            S::Cast => {
                let src = simd_lanes(&vals[0])?;
                let src_info = self.simd_lane_info(tag0, &src);
                let dst_tag = generic_args.get(1).map(String::as_str).unwrap_or("");
                let dst_elem = vector_elem_tag(dst_tag).ok_or_else(|| {
                    unsupported("simd cast target element type is not runtime-supported")
                })?;
                let mut out = Vec::with_capacity(src.len());
                for lane in src.iter() {
                    out.push(simd_cast_lane(lane, src_info, dst_elem)?);
                }
                simd_wrap(&vals[0], out)
            }
            S::Bitmask => {
                let lanes = simd_lanes(&vals[0])?;
                let mut bits: u128 = 0;
                for (i, lane) in lanes.iter().enumerate() {
                    if simd_lane_active(lane)? {
                        bits |= 1u128 << i;
                    }
                }
                // The result type is the call's second generic
                // argument, and `simd_bitmask` is declared over
                // two shapes for it: an integer, or a byte array
                // holding the mask's little-endian byte
                // decomposition.
                if let Some(bytes) = generic_args
                    .get(1)
                    .and_then(|tag| tag.trim().strip_prefix("[u8;"))
                    .and_then(|rest| rest.trim().strip_suffix(']'))
                    .and_then(|n| n.trim().parse::<usize>().ok())
                {
                    let out: Vec<Value> = (0..bytes)
                        .map(|i| Value::Uint((bits >> (8 * i)) & 0xff))
                        .collect();
                    return Ok(Value::Tuple(out));
                }
                Ok(Value::Uint(bits))
            }
            S::Select => {
                let mask = simd_lanes(&vals[0])?;
                let yes = simd_lanes(&vals[1])?;
                let no = simd_lanes(&vals[2])?;
                let mut out = Vec::with_capacity(yes.len());
                for (i, lane) in yes.iter().enumerate() {
                    out.push(if simd_lane_active(&mask[i])? {
                        lane.clone()
                    } else {
                        no[i].clone()
                    });
                }
                simd_wrap(&vals[1], out)
            }
            S::SelectBitmask => {
                // The mask arrives as an integer or in the byte
                // array form `simd_bitmask` also produces; the
                // array's bytes compose little-endian.
                let bits = match &vals[0] {
                    Value::Tuple(items) => {
                        let mut bits: u128 = 0;
                        for (i, item) in items.iter().enumerate() {
                            bits |=
                                as_u128(item, "simd select bitmask")? << (8 * i);
                        }
                        bits
                    }
                    Value::Bytes(bytes) => bytes.iter().enumerate().fold(
                        0u128,
                        |acc, (i, b)| acc | ((*b as u128) << (8 * i)),
                    ),
                    other => as_u128(other, "simd select bitmask")?,
                };
                let yes = simd_lanes(&vals[1])?;
                let no = simd_lanes(&vals[2])?;
                let mut out = Vec::with_capacity(yes.len());
                for (i, lane) in yes.iter().enumerate() {
                    out.push(if (bits >> i) & 1 == 1 {
                        lane.clone()
                    } else {
                        no[i].clone()
                    });
                }
                simd_wrap(&vals[1], out)
            }
            S::Gather => {
                let defaults = simd_lanes(&vals[0])?;
                let ptrs = simd_lanes(&vals[1])?;
                let mask = simd_lanes(&vals[2])?;
                let mut out = Vec::with_capacity(defaults.len());
                for (i, default) in defaults.iter().enumerate() {
                    if simd_lane_active(&mask[i])? {
                        let target = self.deref_value(frames, &ptrs[i])?;
                        out.push(self.read_resolved(frames, &target)?);
                    } else {
                        out.push(default.clone());
                    }
                }
                simd_wrap(&vals[0], out)
            }
            S::Scatter => {
                let values = simd_lanes(&vals[0])?;
                let ptrs = simd_lanes(&vals[1])?;
                let mask = simd_lanes(&vals[2])?;
                for (i, value) in values.iter().enumerate() {
                    if simd_lane_active(&mask[i])? {
                        let target = self.deref_value(frames, &ptrs[i])?;
                        self.write_resolved(frames, &target, value.clone())?;
                    }
                }
                Ok(Value::Unit)
            }
            S::MaskedLoad => {
                let mask = simd_lanes(&vals[0])?;
                let defaults = simd_lanes(&vals[2])?;
                let target = self.deref_value(frames, &vals[1])?;
                let aggregate = matches!(
                    self.read_resolved(frames, &target),
                    Ok(Value::Tuple(_)) | Ok(Value::Enum { .. })
                );
                let mut out = Vec::with_capacity(defaults.len());
                for (i, default) in defaults.iter().enumerate() {
                    if simd_lane_active(&mask[i])? {
                        let value = if aggregate || i > 0 {
                            // The sequence accessor, not the plain
                            // indexed one: the pointer may denote
                            // a single element (an element pointer
                            // into an array), whose lanes are its
                            // siblings in the enclosing aggregate,
                            // not indices into the element itself.
                            self.read_seq_at(frames, &target, i)?
                        } else {
                            self.read_resolved(frames, &target)?
                        };
                        out.push(value);
                    } else {
                        out.push(default.clone());
                    }
                }
                simd_wrap(&vals[2], out)
            }
            S::MaskedStore => {
                let mask = simd_lanes(&vals[0])?;
                let values = simd_lanes(&vals[2])?;
                let target = self.deref_value(frames, &vals[1])?;
                let mut aggregate = false;
                let mut uninit = false;
                match self.read_resolved(frames, &target) {
                    Ok(Value::Tuple(_)) | Ok(Value::Enum { .. }) => aggregate = true,
                    Err(InterpError::UninitializedRead { .. }) => uninit = true,
                    _ => {}
                }
                if !aggregate
                    && uninit
                    && mask.iter().any(|m| simd_lane_active(m).unwrap_or(false))
                {
                    // Materialize the region uninitialized so the
                    // masked lanes can land; the others stay
                    // uninitialized, as they should.
                    self.write_resolved(
                        frames,
                        &target,
                        Value::Tuple(vec![Value::Uninit; values.len()]),
                    )?;
                    aggregate = true;
                }
                for (i, value) in values.iter().enumerate() {
                    if simd_lane_active(&mask[i])? {
                        if aggregate || i > 0 {
                            // Sequence access, as in MaskedLoad:
                            // an element pointer's lanes are its
                            // siblings, not indices into it.
                            self.write_seq_at(frames, &target, i, value.clone())?;
                        } else {
                            self.write_resolved(frames, &target, value.clone())?;
                        }
                    }
                }
                Ok(Value::Unit)
            }
            S::ArithOffset => {
                let ptrs = simd_lanes(&vals[0])?;
                let offsets = simd_lanes(&vals[1])?;
                // The element type of the pointer vector names
                // the pointee; the interpreter's own offset
                // machinery applies the per-lane offsets, wrapping
                // as the intrinsic's name promises.
                let elem_size = vector_elem_tag(tag0)
                    .and_then(|elem| {
                        let elem = elem.trim();
                        elem.strip_prefix("*const ")
                            .or_else(|| elem.strip_prefix("*mut "))
                    })
                    .and_then(|pointee| self.layout_of(pointee.trim()))
                    .map(|(size, _)| size as u64);
                let mut out = Vec::with_capacity(ptrs.len());
                for (ptr, off) in ptrs.iter().zip(offsets.iter()) {
                    out.push(self.ptr_offset_value(
                        frames,
                        ptr.clone(),
                        as_i128(off, "simd lane")?,
                        elem_size,
                        true,
                    )?);
                }
                simd_wrap(&vals[0], out)
            }
            S::ReduceAddOrdered
            | S::ReduceAddUnordered
            | S::ReduceMulOrdered
            | S::ReduceMulUnordered
            | S::ReduceMax
            | S::ReduceMin
            | S::ReduceAnd
            | S::ReduceOr
            | S::ReduceXor => {
                let lanes = simd_lanes(&vals[0])?;
                let info = self.simd_lane_info(tag0, &lanes);
                simd_reduce(op, &lanes, vals.get(1), info)
            }
            S::ReduceAll | S::ReduceAny => {
                let lanes = simd_lanes(&vals[0])?;
                let mut result = op == S::ReduceAll;
                for lane in lanes.iter() {
                    let active = simd_lane_active(lane)?;
                    if op == S::ReduceAll {
                        result &= active;
                    } else {
                        result |= active;
                    }
                }
                Ok(Value::Bool(result))
            }
            S::CastPtr => Ok(vals[0].clone()),
            S::ExposeProvenance => {
                let lanes = simd_lanes(&vals[0])?;
                let mut out = Vec::with_capacity(lanes.len());
                for lane in lanes.iter() {
                    let addr = synthetic_address(lane).ok_or_else(|| {
                        unsupported("simd_expose_provenance of this lane is not runtime-supported")
                    })?;
                    out.push(Value::Uint(addr));
                }
                simd_wrap(&vals[0], out)
            }
            S::WithExposedProvenance => {
                let lanes = simd_lanes(&vals[0])?;
                let mut out = Vec::with_capacity(lanes.len());
                for lane in lanes.iter() {
                    let addr = as_u128(lane, "simd lane")?;
                    let addr = u64::try_from(addr).map_err(|_| {
                        unsupported("simd_with_exposed_provenance address is not runtime-supported")
                    })?;
                    out.push(Value::Ref(RefTarget::Address(addr)));
                }
                simd_wrap(&vals[0], out)
            }
        }
    }
    /// `Iterator::next`: mutate the iterator in place through the
    /// `&mut` argument and return the `Option` value.
    /// A `Duration` value in the real field shape (`secs`,
    /// `nanos`), with the normalization `Duration::new` performs
    /// (a nanosecond count at or past one second carries into the
    /// seconds; the carry is checked, as in std).


    /// Clone an owned value the way this module's `Clone::clone`
    /// model does: a value copy, except that a built box (the
    /// one-field aggregate holding an allocation pointer — the
    /// shape `box_alloc_pointer` documents as exact) is
    /// re-allocated around a recursively cloned pointee, so the
    /// clone owns its storage at every level. References, fat
    /// pointers and trait objects copy as the pointers they are,
    /// exactly as a field-wise `Clone` of a struct containing
    /// them behaves.
    fn clone_owned_value(&self, frames: &[Frame], value: &Value) -> InterpResult<Value> {
        if let Some((id, projection)) = box_alloc_pointer(value) {
            let pointee =
                self.read_resolved(frames, &Resolved::Alloc { id, projection })?;
            let cloned = self.clone_owned_value(frames, &pointee)?;
            return Ok(self.new_box(cloned));
        }
        Ok(match value {
            Value::Enum { variant, fields } => Value::Enum {
                variant: *variant,
                fields: fields
                    .iter()
                    .map(|f| self.clone_owned_value(frames, f))
                    .collect::<InterpResult<Vec<_>>>()?,
            },
            Value::Tuple(items) => Value::Tuple(
                items
                    .iter()
                    .map(|f| self.clone_owned_value(frames, f))
                    .collect::<InterpResult<Vec<_>>>()?,
            ),
            other => other.clone(),
        })
    }

    /// Clone a `Waker` / `LocalWaker` value the way std's manual
    /// `Clone` impls do. The modelled `From<Arc<T: Wake>>` /
    /// `From<Rc<T: LocalWake>>` waker is a one-field aggregate
    /// carrying the handle itself: the clone is another live
    /// handle on the same allocation, so the strong count moves
    /// and the value copies. A `from_raw` waker holds a real
    /// `RawWaker` (as the waker value itself or as its one
    /// non-zero-sized field): the clone calls the vtable's clone
    /// entry with the data pointer and wraps the `RawWaker` it
    /// returns in the same shape. Any other shape is an explicit
    /// error.
    fn clone_waker_value(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        name: &str,
    ) -> InterpResult<Value> {
        if let Value::Enum { fields, .. } = value {
            if fields.len() == 1 {
                if let Some(id) = rc_handle_id(&fields[0]) {
                    self.rc_inc_strong(id)?;
                    return Ok(value.clone());
                }
            }
        }
        let (variant, wrapped, raw) = match value {
            Value::Enum { variant, fields } if fields.len() == 1 => {
                (*variant, true, fields[0].clone())
            }
            Value::Enum { .. } => (0, false, value.clone()),
            other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "a Waker value",
                    other.kind_name(),
                ));
            }
        };
        let Value::Enum { fields: raw_fields, .. } = &raw else {
            return Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "a RawWaker value",
                raw.kind_name(),
            ));
        };
        if raw_fields.len() != 2 {
            return Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "a RawWaker value (data pointer and vtable)",
                raw.kind_name(),
            ));
        }
        let data = raw_fields[0].clone();
        let resolved = self.deref_value(frames, &raw_fields[1])?;
        let vtable = self.read_resolved(frames, &resolved)?;
        let Value::Enum { fields: vtable_fields, .. } = &vtable else {
            return Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "a RawWakerVTable value",
                vtable.kind_name(),
            ));
        };
        let Some(Value::FnPtr(clone_name)) = vtable_fields.first() else {
            return Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "a RawWakerVTable with a function-pointer clone entry",
                vtable.kind_name(),
            ));
        };
        let clone_name = clone_name.clone();
        let new_raw = self.call_sync(frames, &clone_name, vec![data])?;
        Ok(if wrapped {
            Value::Enum { variant, fields: vec![new_raw] }
        } else {
            new_raw
        })
    }

    /// The tests of a `&[TestDescAndFn]` slice, read out of the
    /// value the harness main hands to `test_main_env_args`.
    /// The descriptor layout below was measured against the
    /// descriptors this toolchain's `--test` harness emits
    /// (twelve fields): 0 the `TestName`, 1 the ignore flag,
    /// 2 the ignore message (`Option`: variant 1 carries the
    /// string), 3 the source file, 4..7 the definition span
    /// (start line/column, end line/column, the start naming
    /// the function's name), 8 the `ShouldPanic` mode (variant
    /// 0 `No`, 1 `Yes`, 2 `YesWithMessage` with the expected
    /// substring as its payload). The second field of the
    /// element is the `TestFn` enum whose payload is the
    /// callable.
    fn read_test_descs(
        &self,
        frames: &[Frame],
        slice: &Value,
    ) -> InterpResult<Vec<TestDesc>> {
        let target = self.deref_value(frames, slice)?;
        let array = self.read_resolved(frames, &target)?;
        let elements = match &array {
            Value::Tuple(items) | Value::Enum { fields: items, .. } => items.clone(),
            other => {
                return Err(InterpError::type_mismatch(
                    "test::test_main_env_args",
                    "a slice of test descriptors",
                    other.kind_name(),
                ))
            }
        };
        let mut out = Vec::with_capacity(elements.len());
        for element in &elements {
            let element = self.resolve_value(frames, element)?;
            let pair = match &element {
                Value::Tuple(items) | Value::Enum { fields: items, .. } if items.len() == 2 => {
                    items
                }
                other => {
                    return Err(InterpError::type_mismatch(
                        "test::test_main_env_args",
                        "a TestDescAndFn value",
                        other.kind_name(),
                    ))
                }
            };
            let desc_value = self.resolve_value(frames, &pair[0])?;
            let desc = match &desc_value {
                Value::Tuple(items) | Value::Enum { fields: items, .. } if items.len() >= 3 => {
                    items
                }
                other => {
                    return Err(InterpError::type_mismatch(
                        "test::test_main_env_args",
                        "a TestDesc value",
                        other.kind_name(),
                    ))
                }
            };
            let name = match &desc[0] {
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    self.resolve_string(frames, &fields[0])?
                }
                other => self.resolve_string(frames, other)?,
            };
            let ignore = matches!(desc.get(1), Some(Value::Bool(true)));
            // A descriptor string field, read through the
            // one-field enum a string constant may decode to.
            let desc_string = |v: &Value| -> Option<String> {
                match v {
                    Value::Enum { fields, .. } if fields.len() == 1 => {
                        self.resolve_string(frames, &fields[0]).ok()
                    }
                    other => self.resolve_string(frames, other).ok(),
                }
            };
            let ignore_message = match desc.get(2) {
                Some(Value::Enum { variant: 1, fields }) if !fields.is_empty() => {
                    desc_string(&fields[0])
                }
                _ => None,
            };
            let source_file = desc.get(3).and_then(desc_string).unwrap_or_default();
            let desc_uint = |i: usize| -> u64 {
                match desc.get(i) {
                    Some(Value::Uint(n)) => *n as u64,
                    _ => 0,
                }
            };
            let should_panic = match desc.get(8) {
                Some(Value::Enum { variant: 1, .. }) => ShouldPanic::Yes,
                Some(Value::Enum { variant: 2, fields }) if !fields.is_empty() => {
                    match desc_string(&fields[0]) {
                        Some(msg) => ShouldPanic::YesWithMessage(msg),
                        None => ShouldPanic::Yes,
                    }
                }
                _ => ShouldPanic::No,
            };
            let callable = match &pair[1] {
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    self.resolve_value(frames, &fields[0])?
                }
                other => self.resolve_value(frames, other)?,
            };
            out.push(TestDesc {
                name,
                ignore,
                ignore_message,
                should_panic,
                source_file,
                start_line: desc_uint(4),
                start_col: desc_uint(5),
                end_line: desc_uint(6),
                end_col: desc_uint(7),
                callable,
            });
        }
        Ok(out)
    }

    /// `Add::add` / `Sub::sub` for the modelled time types,
    /// completed by the interpreter when the demanded instance
    /// has no collected body (the `std::time` instances are not
    /// available from the compiler). Instances with bodies — a
    /// guest impl, a generic instantiation's shim — always take
    /// the normal call path and never reach here, so the
    /// collected-body priority is untouched. Returns the result
    /// value, or an error for a Self type outside the model.
    pub(crate) fn time_add_sub(
        &self,
        name: &str,
        self_ty: &str,
        sub: bool,
        vals: &[Value],
    ) -> InterpResult<Value> {
        if vals.len() != 2 {
            return Err(InterpError::type_mismatch(
                name,
                "2 argument(s)",
                format!("{} argument(s)", vals.len()),
            ));
        }
        match self_ty {
                "std::time::Instant" | "std::time::SystemTime" => {
                    let system = self_ty == "std::time::SystemTime";
                    let base = time_pair(&vals[0]).ok_or_else(|| {
                        InterpError::type_mismatch(
                            name,
                            "a time value",
                            vals[0].kind_name(),
                        )
                    })?;
                    let dur = duration_pair(&vals[1]).ok_or_else(|| {
                        InterpError::type_mismatch(
                            name,
                            "a Duration value",
                            vals[1].kind_name(),
                        )
                    })?;
                    let pair = if sub {
                        checked_time_sub(base, dur, system)
                    } else {
                        checked_time_add(base, dur, system)
                    };
                    match pair {
                        Some(pair) => Ok(instant_value(pair)),
                        None => Err(InterpError::panic_str_payload(
                            match (system, sub) {
                                (false, false) => {
                                    "overflow when adding duration to instant"
                                }
                                (false, true) => {
                                    "overflow when subtracting duration from instant"
                                }
                                (true, false) => {
                                    "overflow when adding duration to system time"
                                }
                                (true, true) => {
                                    "overflow when subtracting duration from system time"
                                }
                            },
                        )),
                    }
                }
                "std::time::Duration" => {
                    let a = duration_pair(&vals[0]).ok_or_else(|| {
                        InterpError::type_mismatch(
                            name,
                            "a Duration value",
                            vals[0].kind_name(),
                        )
                    })?;
                    let b = duration_pair(&vals[1]).ok_or_else(|| {
                        InterpError::type_mismatch(
                            name,
                            "a Duration value",
                            vals[1].kind_name(),
                        )
                    })?;
                    if sub {
                        match checked_time_sub(a, b, true) {
                            Some(pair) => {
                                self.duration_value(pair.0 as u64, pair.1 as u64)
                            }
                            None => Err(InterpError::panic_str_payload(
                                "overflow when subtracting duration",
                            )),
                        }
                    } else {
                        match checked_time_add(a, b, true) {
                            Some(pair) => {
                                self.duration_value(pair.0 as u64, pair.1 as u64)
                            }
                            None => Err(InterpError::panic_str_payload(
                                "overflow when adding duration",
                            )),
                        }
                    }
                }
                _ => Err(unsupported(format!(
                    "{} for {self_ty} is not runtime-supported",
                    if sub { "Sub::sub" } else { "Add::add" }
                ))),
            }
    }

    fn duration_value(&self, secs: u64, nanos: u64) -> InterpResult<Value> {
        let (secs, nanos) = if nanos >= 1_000_000_000 {
            let carry = nanos / 1_000_000_000;
            let secs = secs
                .checked_add(carry)
                .ok_or_else(|| InterpError::panic("overflow in Duration::new"))?;
            (secs, nanos % 1_000_000_000)
        } else {
            (secs, nanos)
        };
        Ok(Value::Enum {
            variant: 0,
            fields: vec![Value::Uint(u128::from(secs)), Value::Uint(u128::from(nanos))],
        })
    }

    /// Register a thread spawn: the callable must be a closure or
    /// a function pointer (a `Box<dyn FnOnce>` has no body the
    /// model can run and stays an explicit error). Returns the
    /// join handle value.
    fn spawn_callable(&mut self, callable: Value) -> InterpResult<Value> {
        match callable {
            Value::Closure { .. } | Value::FnPtr(_) => {
                let id = self.spawn_thread(callable);
                Ok(join_handle_value(id))
            }
            other => Err(unsupported(format!(
                "thread spawn of a {} value is not runtime-supported",
                other.kind_name()
            ))),
        }
    }

    /// The channel id of the handle an mpsc method's `&self`
    /// argument points at.
    fn handle_id_of_ref(
        &self,
        frames: &[Frame],
        arg: &Value,
        name: &str,
    ) -> InterpResult<u64> {
        let target = self.deref_value(frames, arg)?;
        let value = self.read_resolved(frames, &target)?;
        channel_handle_id(&value).ok_or_else(|| {
            InterpError::type_mismatch(
                format!("builtin {name}"),
                "an mpsc channel handle",
                value.kind_name(),
            )
        })
    }

    /// Why a blocking channel operation cannot proceed: either
    /// every pending thread has already run (or none exists), or —
    /// when the block happens inside a spawned thread's run — the
    /// thread that must act next is the very thread that spawned
    /// this one and is itself blocked waiting for it: a cyclic
    /// wait no run-to-completion schedule can serve.
    fn block_forever_message(&self, name: &str, situation: &str) -> String {
        if self.current_thread_id() != 1 {
            format!(
                "{name} would block forever: {situation}; the thread that \
                 must act next is the blocked spawner of the running thread \
                 (a cyclic wait between threads is not servable by the \
                 single-threaded model)"
            )
        } else {
            format!(
                "{name} would block forever: {situation} (single-threaded model)"
            )
        }
    }

    /// `Sender::send` / `SyncSender::send`: enqueue the value,
    /// running pending threads while a bounded channel is full
    /// (the sender would block; the threads it waits for run
    /// first). Disconnection hands the value back inside a
    /// `SendError`.
    fn mpsc_send(
        &mut self,
        frames: &mut Vec<Frame>,
        id: u64,
        value: Value,
        name: &str,
    ) -> InterpResult<Value> {
        enum Step {
            Sent,
            Disconnected,
            Full,
            Unknown,
        }
        loop {
            let step = match self.with_channel(id, |st| {
                if st.receivers == 0 {
                    Step::Disconnected
                } else if st.bound.is_none_or(|b| st.queue.len() < b) {
                    st.queue.push_back(value.clone());
                    Step::Sent
                } else {
                    Step::Full
                }
            }) {
                Some(step) => step,
                None => Step::Unknown,
            };
            match step {
                Step::Sent => return Ok(ok_result(Value::Unit)),
                Step::Disconnected => {
                    return Ok(err_result(Value::Enum {
                        variant: 0,
                        fields: vec![value],
                    }))
                }
                Step::Full => {
                    if !self.drain_threads(frames)? {
                        return Err(unsupported(self.block_forever_message(
                            name,
                            "the channel is full and no pending thread can receive",
                        )));
                    }
                }
                Step::Unknown => {
                    return Err(unsupported(format!("{name} on an unknown channel")))
                }
            }
        }
    }

    /// `Receiver::recv`: dequeue the oldest value, running
    /// pending threads while the channel is empty but senders
    /// live. `Err(())` is disconnection (queue empty and the last
    /// sender gone).
    fn mpsc_recv(
        &mut self,
        frames: &mut Vec<Frame>,
        id: u64,
        name: &str,
    ) -> InterpResult<Result<Value, ()>> {
        loop {
            let step = self.with_channel(id, |st| (st.queue.pop_front(), st.senders));
            let Some((popped, senders)) = step else {
                return Err(unsupported(format!("{name} on an unknown channel")));
            };
            if let Some(value) = popped {
                return Ok(Ok(value));
            }
            if senders == 0 {
                return Ok(Err(()));
            }
            if !self.drain_threads(frames)? {
                return Err(unsupported(self.block_forever_message(
                    name,
                    "the channel is empty and no pending thread can send",
                )));
            }
        }
    }

    /// `Receiver::recv_timeout`: like `mpsc_recv`, except that
    /// when no pending thread can produce a value the outcome is
    /// a timeout (`Err(0)`) rather than an error — the model
    /// cannot wait, and a native run whose producers have all
    /// finished observes exactly that. Disconnection is `Err(1)`.
    fn mpsc_recv_or(
        &mut self,
        frames: &mut Vec<Frame>,
        id: u64,
        name: &str,
    ) -> InterpResult<Result<Value, u32>> {
        loop {
            let step = self.with_channel(id, |st| (st.queue.pop_front(), st.senders));
            let Some((popped, senders)) = step else {
                return Err(unsupported(format!("{name} on an unknown channel")));
            };
            if let Some(value) = popped {
                return Ok(Ok(value));
            }
            if senders == 0 {
                return Ok(Err(1));
            }
            if !self.drain_threads(frames)? {
                return Ok(Err(0));
            }
        }
    }

    fn option_ref_into_iter(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        mutable: bool,
    ) -> InterpResult<Value> {
        let target = self.deref_value(frames, value)?;
        let option = self.read_resolved(frames, &target)?;
        let elems = match &option {
            Value::Enum { variant: 1, .. } => {
                let elem = match &target {
                    Resolved::Place {
                        frame_pos,
                        local,
                        projection,
                    } => {
                        let mut projection = projection.clone();
                        projection.push(RefProj::Field(0));
                        Value::Ref(RefTarget::place(
                            frames[*frame_pos].id,
                            *local,
                            projection,
                        ))
                    }
                    Resolved::Static { name, projection } => {
                        let mut projection = projection.clone();
                        projection.push(RefProj::Field(0));
                        Value::Ref(RefTarget::Static {
                            name: name.clone(),
                            projection,
                        })
                    }
                    Resolved::Alloc { id, projection } => {
                        let mut projection = projection.clone();
                        projection.push(RefProj::Field(0));
                        Value::Ref(RefTarget::Alloc {
                            id: *id,
                            projection,
                        })
                    }
                    Resolved::Value(base, projection) => {
                        let v = self.value_target_value(base, projection)?;
                        if mutable {
                            return Err(unsupported(
                                "IntoIterator::into_iter for &mut Option over a constant aggregate is not runtime-supported",
                            ));
                        }
                        match v {
                            Value::Enum { fields, .. } if !fields.is_empty() => {
                                Value::Ref(RefTarget::value(fields[0].clone()))
                            }
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "IntoIterator::into_iter for &Option",
                                    "option value",
                                    other.kind_name(),
                                ))
                            }
                        }
                    }
                };
                vec![elem]
            }
            Value::Enum { variant: 0, .. } => Vec::new(),
            other => {
                return Err(InterpError::type_mismatch(
                    "IntoIterator::into_iter for &Option",
                    "option value",
                    other.kind_name(),
                ))
            }
        };
        Ok(Value::Enum {
            variant: 0,
            fields: vec![Value::Tuple(elems), Value::Uint(0)],
        })
    }

    /// `Iterator::next`: mutate the iterator in place through the
    /// `&mut` argument and return the `Option` value.
    fn iterator_next(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        ref_arg: &Value,
    ) -> InterpResult<Value> {
        let target = self.deref_value(frames, ref_arg)?;
        self.iterator_next_at(frames, self_ty, &target)
    }

    /// `Iterator::next` on the iterator stored at `target`:
    /// mutate it in place and return the `Option` value.
    fn iterator_next_at(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &Resolved,
    ) -> InterpResult<Value> {
        // A `Map` on a place-based state is stepped in place
        // (its closure's environment reference must name the
        // closure field's own storage); `map_next` declines a
        // state that does not live in a place, and the
        // value-level step below then applies.
        if self_ty.starts_with("std::iter::Map<") {
            if let Some(result) = self.map_next(frames, self_ty, target)? {
                return Ok(result);
            }
        }
        if self_ty.starts_with("std::iter::SkipWhile<") {
            if let Some(result) = self.skip_while_next(frames, self_ty, target)? {
                return Ok(result);
            }
        }
        let state = self.read_resolved(frames, target)?;
        let (new_state, result) = self.iterator_next_state(frames, self_ty, state)?;
        self.write_resolved(frames, target, new_state)?;
        Ok(result)
    }

    /// `DoubleEndedIterator::next_back`: mutate the iterator in
    /// place through the `&mut` argument and return the `Option`
    /// value — the mirror of [`Self::iterator_next`].
    fn iterator_next_back(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        ref_arg: &Value,
    ) -> InterpResult<Value> {
        let target = self.deref_value(frames, ref_arg)?;
        self.iterator_next_back_at(frames, self_ty, &target)
    }

    /// `DoubleEndedIterator::next_back` on the iterator stored
    /// at `target`: mutate it in place and return the `Option`
    /// value.
    fn iterator_next_back_at(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &Resolved,
    ) -> InterpResult<Value> {
        // A `Map` on a place-based state is stepped in place,
        // exactly as for `next` (see `iterator_next_at`): its
        // closure's environment reference must name the closure
        // field's own storage.
        if self_ty.starts_with("std::iter::Map<") {
            if let Some(result) = self.map_next_back(frames, self_ty, target)? {
                return Ok(result);
            }
        }
        let state = self.read_resolved(frames, target)?;
        let (new_state, result) = self.next_back_on_state(frames, self_ty, state)?;
        self.write_resolved(frames, target, new_state)?;
        Ok(result)
    }

    /// The place-based back step of a `Map` (see `map_next`):
    /// step the inner iterator in place from the back, then apply
    /// the closure to the yielded item. Declines a state that
    /// does not live in a place, like `map_next` does.
    fn map_next_back(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &Resolved,
    ) -> InterpResult<Option<Value>> {
        let Some(inner_target) = extend_target(target, RefProj::Field(0)) else {
            return Ok(None);
        };
        let Some(closure_target) = extend_target(target, RefProj::Field(1)) else {
            return Ok(None);
        };
        let inner_ty = first_generic_arg(self_ty).unwrap_or("");
        let item = self.iterator_next_back_at(frames, inner_ty, &inner_target)?;
        let Some(item) = option_payload_opt(item) else {
            return Ok(Some(none_value()));
        };
        let callable = self.read_resolved(frames, &closure_target)?;
        let env_ref = self.target_ref_value(frames, &closure_target)?;
        let mapped = self.call_callable_sync(frames, &callable, Some(env_ref), vec![item])?;
        Ok(Some(some_value(mapped)))
    }

    /// One `DoubleEndedIterator::next_back` step on an iterator
    /// state value: the new state and the step's `Option` value.
    /// The leaf shapes are the pure state transition
    /// (`iterator_next_back_state`) that `Rev::next` already
    /// runs; the arms here are the wrappers whose back step is a
    /// composition of other steps: a `Rev`'s back step is its
    /// inner iterator's front step, an `Enumerate`'s is the
    /// inner back step indexed from the end, a `Copied`'s is the
    /// inner back step with the yielded reference followed, a
    /// `Map`'s is the inner back step with the closure applied,
    /// and an `IterMut`'s hands out the last remaining element
    /// of the container.
    fn next_back_on_state(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        state: Value,
    ) -> InterpResult<(Value, Value)> {
        if self_ty.starts_with("std::path::Components") {
            // The back step of the state `Builtin::PathComponents`
            // builds, replaying the host's parsing (see
            // `components_step`).
            return self.components_step(state, true);
        }
        if self_ty.starts_with("std::iter::Rev<") {
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Rev state",
                    "[inner iterator]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = self_ty
                .strip_prefix("std::iter::Rev<")
                .and_then(|rest| rest.strip_suffix('>'))
                .unwrap_or("");
            let (new_inner, result) =
                self.iterator_next_state(frames, inner_ty, fields[0].clone())?;
            return Ok((rebuild_state(&state, vec![new_inner]), result));
        }
        if self_ty.starts_with("std::iter::Enumerate<") {
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Enumerate state",
                    "[iterator, count]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let count = match fields[1] {
                Value::Uint(v) => v,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "Enumerate count",
                        "unsigned integer",
                        other.kind_name(),
                    ))
                }
            };
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (new_inner, item) =
                self.next_back_on_state(frames, inner_ty, fields[0].clone())?;
            let Some(x) = option_payload_opt(item) else {
                return Ok((
                    rebuild_state(&state, vec![new_inner, Value::Uint(count)]),
                    none_value(),
                ));
            };
            // The back item's index is the front count plus the
            // items that remain after it; std computes it from
            // the inner iterator's exact length, so an inexact
            // size hint is an explicit unsupported error here
            // (the step itself already happened on the state
            // value, but the run fails before anything observes
            // the stepped state).
            let inner_ref = Value::Ref(RefTarget::value(fields[0].clone()));
            let hint = self.iterator_size_hint(frames, inner_ty, &inner_ref)?;
            let remaining = match &hint {
                Value::Tuple(parts) if parts.len() == 2 => match &parts[0] {
                    Value::Uint(lower) => match option_payload_opt(parts[1].clone()) {
                        Some(Value::Uint(upper)) if upper == *lower => Some(*lower),
                        _ => None,
                    },
                    _ => None,
                },
                _ => None,
            };
            let Some(remaining) = remaining else {
                return Err(unsupported(format!(
                    "DoubleEndedIterator::next_back for {self_ty} needs the inner \
                     iterator's exact length, which its size hint does not give"
                )));
            };
            if remaining == 0 {
                return Err(unsupported(format!(
                    "DoubleEndedIterator::next_back for {self_ty} yielded an item \
                     from an inner iterator reporting zero remaining items"
                )));
            }
            let index = count + remaining - 1;
            return Ok((
                rebuild_state(&state, vec![new_inner, Value::Uint(count)]),
                some_value(Value::Tuple(vec![Value::Uint(index), x])),
            ));
        }
        if self_ty.starts_with("std::iter::Copied<") {
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Copied state",
                    "[iterator]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (new_inner, item) =
                self.next_back_on_state(frames, inner_ty, fields[0].clone())?;
            return match option_payload_opt(item) {
                None => Ok((rebuild_state(&state, vec![new_inner]), none_value())),
                Some(x) => {
                    let copied = self.resolve_value(frames, &x)?;
                    Ok((rebuild_state(&state, vec![new_inner]), some_value(copied)))
                }
            };
        }
        if self_ty.starts_with("std::iter::Map<") {
            // The value-level step, used when the Map does not
            // live in a place (see `map_next_back`): the closure
            // is applied with a reference to the value itself,
            // exactly as in the front step's fallback.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Map state",
                    "[iterator, closure]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (new_inner, item) =
                self.next_back_on_state(frames, inner_ty, fields[0].clone())?;
            return match option_payload_opt(item) {
                None => Ok((
                    rebuild_state(&state, vec![new_inner, fields[1].clone()]),
                    none_value(),
                )),
                Some(x) => {
                    let mapped =
                        self.call_callable_sync(frames, &fields[1], None, vec![x])?;
                    Ok((
                        rebuild_state(&state, vec![new_inner, fields[1].clone()]),
                        some_value(mapped),
                    ))
                }
            };
        }
        if self_ty.starts_with("std::option::Iter<")
            || self_ty.starts_with("std::option::IterMut<")
        {
            // The two state shapes of the front step (see
            // `next_on_state`): the [elements, index] state, whose
            // back step drops the last remaining element, and the
            // nested one-payload shape, whose only element is
            // taken exactly as the front step takes it.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() == 2 {
                if let (Value::Tuple(items), Value::Uint(idx)) = (&fields[0], &fields[1]) {
                    let idx = *idx as usize;
                    if idx < items.len() {
                        let item = items[items.len() - 1].clone();
                        let kept = items[..items.len() - 1].to_vec();
                        let new_state = rebuild_state(
                            &state,
                            vec![Value::Tuple(kept), Value::Uint(idx as u128)],
                        );
                        return Ok((new_state, some_value(item)));
                    }
                    return Ok((state, none_value()));
                }
            }
            return match take_option_payload(&state) {
                Ok(Some((new_state, item))) => Ok((new_state, some_value(item))),
                Ok(None) => Ok((state, none_value())),
                Err(()) => Err(InterpError::type_mismatch(
                    format!("iterator state of {self_ty}"),
                    "an Option value in one-field wrappers",
                    state.kind_name(),
                )),
            };
        }
        if self_ty.starts_with("std::iter::Once<") {
            // `Once`'s single payload is taken on the first step
            // from either end; the step is the front step's.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Once state",
                    "[option]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            return match take_option_payload(&fields[0]) {
                Ok(Some((new_inner, item))) => {
                    Ok((rebuild_state(&state, vec![new_inner]), some_value(item)))
                }
                Ok(None) => Ok((state, none_value())),
                Err(()) => Err(InterpError::type_mismatch(
                    "Once state",
                    "an Option value in one-field wrappers",
                    fields[0].kind_name(),
                )),
            };
        }
        if self_ty.starts_with("std::slice::IterMut<") {
            // The state `IntoIterator` builds for `&mut Vec` /
            // `&mut [T; N]` / `&mut [T]` (see the front step in
            // `next_on_state`): the back step hands out a
            // reference to the last remaining element in the
            // container's own storage and shrinks the length.
            // Any other state shape — the pointer-pair layout
            // the real `iter_mut` body builds, which the front
            // step does not serve either — falls through to the
            // pure transition below and is reported unsupported.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() == 3 {
                if let (Value::Uint(idx), Value::Uint(len)) = (&fields[1], &fields[2]) {
                    let (idx, len) = (*idx, *len);
                    if idx < len {
                        let elem =
                            self.iter_mut_element_ref(frames, &fields[0], (len - 1) as usize)?;
                        return Ok((
                            rebuild_state(
                                &state,
                                vec![
                                    fields[0].clone(),
                                    Value::Uint(idx),
                                    Value::Uint(len - 1),
                                ],
                            ),
                            some_value(elem),
                        ));
                    }
                    return Ok((state, none_value()));
                }
            }
        }
        let (new_state, item) = iterator_next_back_state(&state, self_ty)?;
        Ok((
            new_state,
            match item {
                Some(value) => some_value(value),
                None => none_value(),
            },
        ))
    }

    /// The state transition of `Iterator::next` for the modelled
    /// iterator shapes, as a function of the iterator state value
    /// alone: returns the successor state and the `Option` result.
    /// `iterator_next_at` applies it to the state stored behind a
    /// reference; the consumer builtins (`count`, `fold`,
    /// `collect`, `for_each`) drive it directly over the owned
    /// iterator value they receive, performing exactly the
    /// sequence of `next` calls the core default bodies perform.
    pub(crate) fn iterator_next_state(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        state: Value,
    ) -> InterpResult<(Value, Value)> {
        if self_ty.starts_with("std::sync::mpsc::Iter<") {
            // The state holds the receiver handle; iteration is
            // `recv` until the channel disconnects.
            let handle = match &state {
                Value::Enum { fields, .. } if fields.len() == 1 => fields[0].clone(),
                other => {
                    return Err(InterpError::type_mismatch(
                        "mpsc Iter state",
                        "an iterator holding a receiver handle",
                        other.kind_name(),
                    ))
                }
            };
            let Some(id) = channel_handle_id(&handle) else {
                return Err(InterpError::type_mismatch(
                    "mpsc Iter state",
                    "a receiver handle",
                    handle.kind_name(),
                ));
            };
            return match self.mpsc_recv(frames, id, "Iterator::next for mpsc::Iter")? {
                Ok(value) => Ok((state, some_value(value))),
                Err(()) => Ok((state, none_value())),
            };
        }
        if self_ty.starts_with("std::sync::mpsc::IntoIter<") {
            // The state holds the receiver handle as its one field.
            let handle = match &state {
                Value::Enum { fields, .. } if fields.len() == 1 => fields[0].clone(),
                other => {
                    return Err(InterpError::type_mismatch(
                        "mpsc IntoIter state",
                        "an iterator holding a receiver handle",
                        other.kind_name(),
                    ))
                }
            };
            let Some(id) = channel_handle_id(&handle) else {
                return Err(InterpError::type_mismatch(
                    "mpsc IntoIter state",
                    "a receiver handle",
                    handle.kind_name(),
                ));
            };
            return match self.mpsc_recv(frames, id, "Iterator::next for mpsc::IntoIter")? {
                Ok(value) => Ok((state, some_value(value))),
                Err(()) => Ok((state, none_value())),
            };
        }
        if self_ty_is(self_ty, "std::env::Args") || self_ty_is(self_ty, "std::env::ArgsOs") {
            // `Args { inner: ArgsOs }` / `ArgsOs { inner:
            // sys::args::common::Args }` / the common `Args {
            // inner: vec::IntoIter }`: step the innermost owned-iterator
            // state and rebuild the wrappers. The yielded
            // `OsString` is the string value itself in this model,
            // so `Args::next`'s OsString-to-String conversion is
            // the identity (the arguments are always valid text
            // here, so the native panic on invalid Unicode cannot
            // arise).
            let depth = if self_ty_is(self_ty, "std::env::Args") { 3 } else { 2 };
            let mut inner = &state;
            for _ in 0..depth {
                inner = match inner {
                    Value::Enum { fields, .. } if fields.len() == 1 => &fields[0],
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("{self_ty} state"),
                            "nested one-field iterator wrappers",
                            other.kind_name(),
                        ))
                    }
                };
            }
            let (new_inner, item) = step_owned_iter(inner, self_ty)?;
            let mut rebuilt = new_inner;
            for _ in 0..depth {
                rebuilt = Value::Enum {
                    variant: 0,
                    fields: vec![rebuilt],
                };
            }
            return Ok((
                rebuilt,
                match item {
                    Some(value) => some_value(value),
                    None => none_value(),
                },
            ));
        }
        if self_ty.starts_with("std::path::Components") {
            // The state is the path's byte string and the
            // numbers of components consumed from each end (see
            // `Builtin::PathComponents`); the step replays the
            // host's own parsing from it.
            return self.components_step(state, false);
        }
        if self_ty.starts_with("std::path::Ancestors") {
            // The state is the struct the converted
            // `Path::ancestors` body built: one field, the
            // `Option<&Path>` holding the next ancestor to
            // yield (see `ancestors_step`).
            return self.ancestors_step(frames, state);
        }
        if self_ty.starts_with("std::slice::Windows<") {
            // Two state shapes reach here: the native struct
            // {tail slice, window size} (built by a struct
            // literal in core's search machinery), stepped by
            // shrinking the tail exactly as the native `next`
            // does, and the `windows` builtin's [slice, size,
            // next index] snapshot.
            let fields = iterator_fields(&state, self_ty)?;
            let tail_slice = |elems: &[Value]| {
                Value::FatRef {
                    base: Box::new(Value::Ref(RefTarget::Value(
                        Box::new(Value::Tuple(elems.to_vec())),
                        vec![RefProj::Index(0)],
                    ))),
                    len: elems.len() as u128,
                }
            };
            let size_of = |f: &Value| -> InterpResult<usize> {
                // `NonZero<usize>` arrives as nested one-field
                // wrappers around the integer.
                let mut cur = f;
                loop {
                    match cur {
                        Value::Uint(v) => return Ok(*v as usize),
                        Value::Enum { fields, .. } if fields.len() == 1 => {
                            cur = &fields[0];
                        }
                        Value::Tuple(fields) if fields.len() == 1 => {
                            cur = &fields[0];
                        }
                        other => {
                            return Err(InterpError::type_mismatch(
                                "Windows size",
                                "non-zero integer",
                                other.kind_name(),
                            ))
                        }
                    }
                }
            };
            if fields.len() == 2 {
                let (elems, _) = self.slice_elements(frames, &fields[0])?;
                let size = size_of(&fields[1])?;
                if size == 0 || elems.len() < size {
                    return Ok((state, none_value()));
                }
                let item = tail_slice(&elems[..size]);
                let new_state = Value::Enum {
                    variant: 0,
                    fields: vec![tail_slice(&elems[1..]), fields[1].clone()],
                };
                return Ok((new_state, some_value(item)));
            }
            if fields.len() != 3 {
                return Err(InterpError::type_mismatch(
                    "Windows state",
                    "[slice, window size] or [slice, window size, next index]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let (elems, _) = self.slice_elements(frames, &fields[0])?;
            let size = size_of(&fields[1])?;
            let index = match fields[2] {
                Value::Uint(v) => v as usize,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "Windows index",
                        "integer",
                        other.kind_name(),
                    ))
                }
            };
            if size == 0 || index + size > elems.len() {
                return Ok((state, none_value()));
            }
            let item = tail_slice(&elems[index..index + size]);
            let new_state = Value::Enum {
                variant: 0,
                fields: vec![
                    fields[0].clone(),
                    fields[1].clone(),
                    Value::Uint((index + 1) as u128),
                ],
            };
            return Ok((new_state, some_value(item)));
        }
        if self_ty_is(self_ty, "std::env::SplitPaths") {
            // The state is the owned-iterator state the
            // `split_paths` builtin snapshots ([parts, next
            // index]); the items are the parts' `PathBuf`
            // values themselves.
            let (new_state, item) = step_owned_iter(&state, self_ty)?;
            return Ok((
                new_state,
                match item {
                    Some(part) => some_value(part),
                    None => none_value(),
                },
            ));
        }
        if self_ty_is(self_ty, "std::fs::ReadDir") {
            // The state is the one-field wrapper around the
            // owned-iterator state ([entries, next index]) the
            // fs sys layer's snapshot builds; the items are the
            // entries' `io::Result<DirEntry>` values themselves.
            let inner = match &state {
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    fields[0].clone()
                }
                other => {
                    return Err(InterpError::type_mismatch(
                        "ReadDir state",
                        "a wrapper around [entries, next index]",
                        other.kind_name(),
                    ))
                }
            };
            let (new_inner, item) = step_owned_iter(&inner, self_ty)?;
            let rebuilt = Value::Enum {
                variant: 0,
                fields: vec![new_inner],
            };
            return Ok((
                rebuilt,
                match item {
                    Some(result) => some_value(result),
                    None => none_value(),
                },
            ));
        }
        let (new_state, result) = self.next_on_state(frames, self_ty, &state)?;
        Ok((new_state, result))
    }

    /// One step of a `std::path::Components` state, wrapping
    /// the item in its `Option` (see `components_step_state`).
    fn components_step(
        &self,
        state: Value,
        from_back: bool,
    ) -> InterpResult<(Value, Value)> {
        let (new_state, item) = components_step_state(&state, from_back)?;
        Ok((
            new_state,
            match item {
                Some(value) => some_value(value),
                None => none_value(),
            },
        ))
    }

    /// One step of a `std::path::Ancestors` state: the one
    /// field is the `Option<&Path>` the iterator yields
    /// next; the step returns it and stores its parent in
    /// its place (`None` once the parent chain ends) —
    /// exactly the native `Ancestors::next`. The parent is
    /// taken with the host's own `std::path`, as for the
    /// other path-parsing models, and the yielded value is
    /// a `&Path` in the model's usual form (`sub_path_value`).
    fn ancestors_step(
        &self,
        frames: &[Frame],
        state: Value,
    ) -> InterpResult<(Value, Value)> {
        let fields = iterator_fields(&state, "std::path::Ancestors")?;
        if fields.len() != 1 {
            return Err(InterpError::type_mismatch(
                "Ancestors state",
                "one field (the next ancestor)",
                format!("{} field(s)", fields.len()),
            ));
        }
        let field = fields.into_iter().next().unwrap();
        if matches!(&field, Value::Enum { variant: 0, .. }) {
            // The `next` field is `None`: the iterator is
            // exhausted and stays exhausted.
            return Ok((state, none_value()));
        }
        let Some(payload) = option_payload_opt(field) else {
            return Err(InterpError::type_mismatch(
                "Ancestors state",
                "an Option<&Path> field",
                "a non-Option field",
            ));
        };
        let bytes = self.path_bytes_of(frames, &payload)?.0;
        let next = {
            use std::os::unix::ffi::OsStrExt;
            let path =
                std::path::Path::new(std::ffi::OsStr::from_bytes(&bytes));
            match path.parent() {
                Some(parent) => {
                    some_value(sub_path_value(parent.as_os_str().as_bytes()))
                }
                None => none_value(),
            }
        };
        Ok((
            rebuild_state(&state, vec![next]),
            some_value(sub_path_value(&bytes)),
        ))
    }

    fn map_next(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &Resolved,
    ) -> InterpResult<Option<Value>> {
        let Some(inner_target) = extend_target(target, RefProj::Field(0)) else {
            return Ok(None);
        };
        let Some(closure_target) = extend_target(target, RefProj::Field(1)) else {
            return Ok(None);
        };
        let inner_ty = first_generic_arg(self_ty).unwrap_or("");
        let item = self.iterator_next_at(frames, inner_ty, &inner_target)?;
        let Some(item) = option_payload_opt(item) else {
            return Ok(Some(none_value()));
        };
        let callable = self.read_resolved(frames, &closure_target)?;
        let env_ref = match &closure_target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => Value::Ref(RefTarget::place(
                frames[*frame_pos].id,
                *local,
                projection.clone(),
            )),
            Resolved::Static { name, projection } => Value::Ref(RefTarget::Static {
                name: name.clone(),
                projection: projection.clone(),
            }),
            Resolved::Alloc { id, projection } => Value::Ref(RefTarget::Alloc {
                id: *id,
                projection: projection.clone(),
            }),
            Resolved::Value(..) => unreachable!("closure target is a place"),
        };
        let mapped = self.call_callable_sync(frames, &callable, Some(env_ref), vec![item])?;
        Ok(Some(some_value(mapped)))
    }

    fn target_ref_value(&self, frames: &[Frame], target: &Resolved) -> InterpResult<Value> {
        match target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => Ok(Value::Ref(RefTarget::place(
                frames[*frame_pos].id,
                *local,
                projection.clone(),
            ))),
            Resolved::Static { name, projection } => Ok(Value::Ref(RefTarget::Static {
                name: name.clone(),
                projection: projection.clone(),
            })),
            Resolved::Alloc { id, projection } => Ok(Value::Ref(RefTarget::Alloc {
                id: *id,
                projection: projection.clone(),
            })),
            Resolved::Value(..) => Err(unsupported(
                "a closure environment reference for a value with no storage".to_string(),
            )),
        }
    }

    fn skip_while_next(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &Resolved,
    ) -> InterpResult<Option<Value>> {
        let Some(inner_target) = extend_target(target, RefProj::Field(0)) else {
            return Ok(None);
        };
        let Some(flag_target) = extend_target(target, RefProj::Field(1)) else {
            return Ok(None);
        };
        let Some(closure_target) = extend_target(target, RefProj::Field(2)) else {
            return Ok(None);
        };
        let inner_ty = first_generic_arg(self_ty).unwrap_or("");
        loop {
            let state = self.read_resolved(frames, target)?;
            let fields = iterator_fields(&state, self_ty)?;
            let done = matches!(fields.get(1), Some(Value::Bool(true)));
            let item = self.iterator_next_at(frames, inner_ty, &inner_target)?;
            let Some(item) = option_payload_opt(item) else {
                return Ok(Some(none_value()));
            };
            if done {
                return Ok(Some(some_value(item)));
            }
            let callable = self.read_resolved(frames, &closure_target)?;
            let env_ref = self.target_ref_value(frames, &closure_target)?;
            let arg = Value::Ref(RefTarget::value(item.clone()));
            let keep = self.call_callable_sync(frames, &callable, Some(env_ref), vec![arg])?;
            match keep {
                Value::Bool(true) => continue,
                Value::Bool(false) => {
                    self.write_resolved(frames, &flag_target, Value::Bool(true))?;
                    return Ok(Some(some_value(item)));
                }
                other => {
                    return Err(InterpError::type_mismatch(
                        "SkipWhile predicate result",
                        "a boolean",
                        other.kind_name(),
                    ))
                }
            }
        }
    }

    fn iter_mut_element_ref(
        &self,
        frames: &[Frame],
        container: &Value,
        idx: usize,
    ) -> InterpResult<Value> {
        match container {
            Value::Ref(RefTarget::Place {
                frame,
                local,
                projection,
            }) => {
                let target = Resolved::Place {
                    frame_pos: self.frame_pos(frames, *frame)?,
                    local: *local,
                    projection: projection.clone(),
                };
                let base = self.read_resolved(frames, &target)?;
                let mut projection = projection.clone();
                match base {
                    Value::Tuple(_) => projection.push(RefProj::Index(idx)),
                    Value::Enum { .. } => projection.push(RefProj::Field(idx)),
                    ref other => {
                        return Err(unsupported(format!(
                            "Iterator::next for slice::IterMut over a {} is not runtime-supported",
                            other.kind_name()
                        )))
                    }
                }
                Ok(Value::Ref(RefTarget::Place {
                    frame: *frame,
                    local: *local,
                    projection,
                }))
            }
            Value::Ref(RefTarget::Static { name, projection }) => {
                let target = Resolved::Static {
                    name: name.clone(),
                    projection: projection.clone(),
                };
                let base = self.read_resolved(frames, &target)?;
                let mut projection = projection.clone();
                match base {
                    Value::Tuple(_) => projection.push(RefProj::Index(idx)),
                    Value::Enum { .. } => projection.push(RefProj::Field(idx)),
                    ref other => {
                        return Err(unsupported(format!(
                            "Iterator::next for slice::IterMut over a {} is not runtime-supported",
                            other.kind_name()
                        )))
                    }
                }
                Ok(Value::Ref(RefTarget::Static {
                    name: name.clone(),
                    projection,
                }))
            }
            Value::FatRef { base, .. } => self.iter_mut_element_ref(frames, base, idx),
            other => Err(unsupported(format!(
                "Iterator::next for slice::IterMut over {} is not runtime-supported",
                other.kind_name()
            ))),
        }
    }

    /// One `Iterator::next` step on an iterator state value:
    /// the new state and the step's `Option` value. Most
    /// iterators are stepped here, on their state as a value;
    /// a `Map` whose state lives in a place is stepped in
    /// `map_next` instead, and this value-level step is the
    /// fallback for a `Map` behind a constant aggregate.
    fn next_on_state(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        state: &Value,
    ) -> InterpResult<(Value, Value)> {
        if self_ty.is_empty() {
            // No type tag: the caller stayed in generic form (a
            // trait method taken as a function value, an
            // impl-qualified name conversion could not resolve),
            // so the state value is all there is. The one state
            // shape whose stepping does not depend on the tag is
            // the owned-sequence state [items, next index] the
            // model builds for the array / Vec / Option
            // into-iterators: all three step it identically in
            // their tagged arm below, including marking the taken
            // slot uninitialized. Any other shape falls through to
            // the tag dispatch, which names it unsupported.
            if let Ok(fields) = iterator_fields(state, self_ty) {
                if fields.len() == 2 {
                    if let (Value::Tuple(items), Value::Uint(idx)) =
                        (&fields[0], &fields[1])
                    {
                        let idx = *idx as usize;
                        if idx < items.len() {
                            let item = items[idx].clone();
                            let mut items = items.clone();
                            items[idx] = Value::Uninit;
                            let new_state = rebuild_state(
                                state,
                                vec![
                                    Value::Tuple(items),
                                    Value::Uint((idx + 1) as u128),
                                ],
                            );
                            return Ok((new_state, some_value(item)));
                        }
                        return Ok((state.clone(), none_value()));
                    }
                }
            }
        }
        if self_ty.starts_with("std::option::Iter<")
            || self_ty.starts_with("std::option::IterMut<")
        {
            // Two state shapes reach here: the [elements, index]
            // state `IntoIterator` builds for `&Option` /
            // `&mut Option` (the element is the payload
            // reference itself, yielded as it is), and the shape
            // `Option::iter`'s own converted body builds — the
            // option nested inside one-field wrappers
            // (`Iter { inner: Item { .. } }`).
            let fields = iterator_fields(state, self_ty)?;
            if fields.len() == 2 {
                if let (Value::Tuple(items), Value::Uint(idx)) = (&fields[0], &fields[1]) {
                    let idx = *idx as usize;
                    if idx < items.len() {
                        let item = items[idx].clone();
                        let new_state = rebuild_state(
                            state,
                            vec![
                                Value::Tuple(items.clone()),
                                Value::Uint((idx + 1) as u128),
                            ],
                        );
                        return Ok((new_state, some_value(item)));
                    }
                    return Ok((state.clone(), none_value()));
                }
            }
            return match take_option_payload(state) {
                Ok(Some((new_state, item))) => Ok((new_state, some_value(item))),
                Ok(None) => Ok((state.clone(), none_value())),
                Err(()) => Err(InterpError::type_mismatch(
                    format!("iterator state of {self_ty}"),
                    "an Option value in one-field wrappers",
                    state.kind_name(),
                )),
            };
        }
        let fields = iterator_fields(state, self_ty)?;
        let (new_fields, result) = if self_ty.starts_with("std::ops::RangeInclusive<") {
            if fields.len() != 3 {
                return Err(InterpError::type_mismatch(
                    "RangeInclusive state",
                    "[start, end, exhausted]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let (start, end, exhausted) = (
                fields[0].clone(),
                fields[1].clone(),
                fields[2].clone(),
            );
            if exhausted == Value::Bool(true) {
                (fields, none_value())
            } else {
                match compare_values(&start, &end)? {
                    Ordering::Less => (
                        vec![increment(&start)?, end, Value::Bool(false)],
                        some_value(start),
                    ),
                    Ordering::Equal => (
                        vec![start.clone(), end, Value::Bool(true)],
                        some_value(start),
                    ),
                    Ordering::Greater => (fields, none_value()),
                }
            }
        } else if self_ty.starts_with("std::ops::Range<") {
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Range state",
                    "[start, end]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let (start, end) = (fields[0].clone(), fields[1].clone());
            match compare_values(&start, &end)? {
                Ordering::Less => (vec![increment(&start)?, end], some_value(start)),
                _ => (fields, none_value()),
            }
        } else if self_ty.starts_with("std::iter::Empty<") {
            // An empty iterator yields `None` forever.
            (fields, none_value())
        } else if self_ty.starts_with("std::slice::Iter<") {
            // The state is the real field layout (see
            // `slice_iter_value`): (ptr, end, marker) with element
            // pointers into the element snapshot. `next` hands out
            // the current pointer itself as the `&T` item and
            // advances it by one element.
            if fields.len() != 3 {
                return Err(InterpError::type_mismatch(
                    "slice::Iter state",
                    "[ptr, end, marker]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let (ptr_field, end) = (fields[0].clone(), fields[1].clone());
            let (Some(cur), Some(last)) = (iter_ptr_index(&ptr_field), iter_ptr_index(&end))
            else {
                return Err(InterpError::type_mismatch(
                    "slice::Iter state",
                    "element pointers into the element snapshot",
                    format!("{} / {}", ptr_field.kind_name(), end.kind_name()),
                ));
            };
            if cur < last {
                let advanced =
                    iter_ptr_with_index(&ptr_field, cur + 1).expect("index step checked");
                let item = iter_ptr_value(&ptr_field).expect("pointer field checked");
                (vec![advanced, end, fields[2].clone()], some_value(item))
            } else {
                (fields, none_value())
            }
        } else if self_ty.starts_with("std::array::IntoIter<")
            || self_ty.starts_with("std::vec::IntoIter<")
            || self_ty.starts_with("alloc::vec::IntoIter<")
            || self_ty.starts_with("std::option::IntoIter<")
            || self_ty_is(self_ty, "std::str::Chars")
            || self_ty_is(self_ty, "core::str::Chars")
            || self_ty_is(self_ty, "std::str::Bytes")
            || self_ty_is(self_ty, "core::str::Bytes")
        {
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "array::IntoIter state",
                    "[items, next index]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let items = match &fields[0] {
                Value::Tuple(items) => items.clone(),
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "array::IntoIter items",
                        "array value",
                        other.kind_name(),
                    ))
                }
            };
            let idx = match fields[1] {
                Value::Uint(v) => v as usize,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "array::IntoIter index",
                        "unsigned integer",
                        other.kind_name(),
                    ))
                }
            };
            if idx < items.len() {
                let item = items[idx].clone();
                // An owning iterator's yielded element has moved
                // out of the iterator's storage: mark the slot
                // uninitialized in the new state, so the state's
                // drop glue (which walks the element tuple) does
                // not drop it a second time. `Chars` / `Bytes`
                // elements carry no glue, so their snapshots keep
                // the elements. (`slice::Iter` is stepped by its
                // own real-layout arm above.)
                let items = if self_ty.starts_with("std::array::IntoIter<")
                    || self_ty.starts_with("std::vec::IntoIter<")
                    || self_ty.starts_with("alloc::vec::IntoIter<")
                    || self_ty.starts_with("std::option::IntoIter<")
                {
                    let mut items = items;
                    items[idx] = Value::Uninit;
                    items
                } else {
                    items
                };
                (
                    vec![Value::Tuple(items), Value::Uint((idx + 1) as u128)],
                    some_value(item),
                )
            } else {
                (fields, none_value())
            }
        } else if self_ty.starts_with("std::iter::Rev<") {
            // `Rev::next` is the inner iterator's `next_back`,
            // computed on the inner state value (see
            // `iterator_next_back_state`); the Rev state is the
            // one-field wrapper around it.
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Rev state",
                    "[inner iterator]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = self_ty
                .strip_prefix("std::iter::Rev<")
                .and_then(|rest| rest.strip_suffix('>'))
                .unwrap_or("");
            let (new_inner, item) = iterator_next_back_state(&fields[0], inner_ty)?;
            (
                vec![new_inner],
                match item {
                    Some(value) => some_value(value),
                    None => none_value(),
                },
            )
        } else if self_ty.starts_with("std::slice::IterMut<") {
            // The state `IntoIterator` builds for `&mut Vec` /
            // `&mut [T; N]` / `&mut [T]`: the container
            // reference, the next index and the length. The
            // element handed out is a reference to the element's
            // place in the container's own storage.
            if fields.len() != 3 {
                return Err(InterpError::type_mismatch(
                    "slice::IterMut state",
                    "[container, next index, length]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let idx = match fields[1] {
                Value::Uint(v) => v,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "slice::IterMut index",
                        "unsigned integer",
                        other.kind_name(),
                    ))
                }
            };
            let len = match fields[2] {
                Value::Uint(v) => v,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "slice::IterMut length",
                        "unsigned integer",
                        other.kind_name(),
                    ))
                }
            };
            if idx < len {
                let elem = self.iter_mut_element_ref(frames, &fields[0], idx as usize)?;
                (
                    vec![fields[0].clone(), Value::Uint(idx + 1), Value::Uint(len)],
                    some_value(elem),
                )
            } else {
                (fields, none_value())
            }
        } else if self_ty.starts_with("std::iter::Once<") {
            // `Once` is a one-field struct wrapping an `Option`:
            // the payload is taken on the first step.
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Once state",
                    "[option]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            // The field wraps the option in the same one-field
            // chain `option::IntoIter` is built from
            // (`Once { inner: IntoIter { inner: Item { .. } } }`).
            match take_option_payload(&fields[0]) {
                Ok(Some((new_inner, item))) => (vec![new_inner], some_value(item)),
                Ok(None) => (fields, none_value()),
                Err(()) => {
                    return Err(InterpError::type_mismatch(
                        "Once state",
                        "an Option value in one-field wrappers",
                        fields[0].kind_name(),
                    ))
                }
            }
        } else if self_ty.starts_with("std::iter::Repeat<") {
            // `Repeat` yields a clone of its one field forever;
            // in the value model the clone is the value itself.
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Repeat state",
                    "[element]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let elem = fields[0].clone();
            (fields, some_value(elem))
        } else if self_ty.starts_with("std::iter::Enumerate<") {
            // `Enumerate` is (inner iterator, count): the step
            // is the inner step paired with the current count.
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Enumerate state",
                    "[iterator, count]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let count = match fields[1] {
                Value::Uint(v) => v,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "Enumerate count",
                        "unsigned integer",
                        other.kind_name(),
                    ))
                }
            };
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (new_inner, item) = self.next_on_state(frames, inner_ty, &fields[0])?;
            match option_payload_opt(item) {
                None => (vec![new_inner, Value::Uint(count)], none_value()),
                Some(x) => (
                    vec![new_inner, Value::Uint(count + 1)],
                    some_value(Value::Tuple(vec![Value::Uint(count), x])),
                ),
            }
        } else if self_ty.starts_with("std::iter::Copied<") {
            // `Copied` is (inner iterator): the step is the
            // inner step with the yielded reference followed to
            // the element it denotes.
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Copied state",
                    "[iterator]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (new_inner, item) = self.next_on_state(frames, inner_ty, &fields[0])?;
            match option_payload_opt(item) {
                None => (vec![new_inner], none_value()),
                Some(x) => {
                    let copied = self.resolve_value(frames, &x)?;
                    (vec![new_inner], some_value(copied))
                }
            }
        } else if self_ty.starts_with("std::iter::SkipWhile<") {
            // `SkipWhile` is (inner iterator, done flag,
            // predicate) — core's own field layout: the
            // predicate is consulted only until its first
            // failure, which sets the flag. The predicate takes
            // the item by reference and is handed a reference
            // to the pulled item.
            if fields.len() != 3 {
                return Err(InterpError::type_mismatch(
                    "SkipWhile state",
                    "[iterator, done, predicate]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let mut inner = fields[0].clone();
            let predicate = fields[2].clone();
            let mut done = matches!(fields.get(1), Some(Value::Bool(true)));
            let mut result = none_value();
            loop {
                let (new_inner, item) = self.next_on_state(frames, inner_ty, &inner)?;
                inner = new_inner;
                let Some(x) = option_payload_opt(item) else {
                    break;
                };
                if done {
                    result = some_value(x);
                    break;
                }
                let arg = Value::Ref(RefTarget::value(x.clone()));
                let keep = self.call_callable_sync(frames, &predicate, None, vec![arg])?;
                match keep {
                    Value::Bool(true) => continue,
                    Value::Bool(false) => {
                        done = true;
                        result = some_value(x);
                        break;
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            "SkipWhile predicate result",
                            "a boolean",
                            other.kind_name(),
                        ))
                    }
                }
            }
            (vec![inner, Value::Bool(done), predicate], result)
        } else if self_ty.starts_with("std::iter::Take<") {
            // `Take` is (inner iterator, remaining count): at
            // zero it yields `None` without touching the inner
            // iterator.
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Take state",
                    "[iterator, remaining]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let remaining = match fields[1] {
                Value::Uint(v) => v,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "Take remaining count",
                        "unsigned integer",
                        other.kind_name(),
                    ))
                }
            };
            if remaining == 0 {
                (fields, none_value())
            } else {
                let inner_ty = first_generic_arg(self_ty).unwrap_or("");
                let (new_inner, item) = self.next_on_state(frames, inner_ty, &fields[0])?;
                match option_payload_opt(item) {
                    None => (vec![new_inner, Value::Uint(remaining)], none_value()),
                    Some(x) => (
                        vec![new_inner, Value::Uint(remaining - 1)],
                        some_value(x),
                    ),
                }
            }
        } else if self_ty.starts_with("std::iter::Map<") {
            // `Map` is (inner iterator, closure): the step is
            // the inner step with the closure applied to the
            // element. This is the value-level step, used when
            // the Map does not live in a place (see `map_next`):
            // the closure is applied with a reference to the
            // value itself, so capture reads resolve and writes
            // to by-value captures are the explicit
            // constant-aggregate error inside the call.
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Map state",
                    "[iterator, closure]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (new_inner, item) = self.next_on_state(frames, inner_ty, &fields[0])?;
            match option_payload_opt(item) {
                None => (vec![new_inner, fields[1].clone()], none_value()),
                Some(x) => {
                    let mapped =
                        self.call_callable_sync(frames, &fields[1], None, vec![x])?;
                    (
                        vec![new_inner, fields[1].clone()],
                        some_value(mapped),
                    )
                }
            }
        } else if self_ty.starts_with("std::iter::Skip<") {
            // `Skip` is (inner iterator, remaining skip count):
            // the first step discards up to `count` items and
            // yields the one after them, and the count is spent
            // by that step whether or not the inner iterator had
            // enough items -- core's `next` delegates the first
            // step to `nth`, which has exactly this contract --
            // so every later step is a plain inner step.
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Skip state",
                    "[iterator, remaining]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let remaining = match fields[1] {
                Value::Uint(v) => v,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "Skip remaining count",
                        "unsigned integer",
                        other.kind_name(),
                    ))
                }
            };
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let mut inner = fields[0].clone();
            let mut to_skip = remaining;
            let mut result = none_value();
            loop {
                let (new_inner, item) =
                    self.iterator_next_state(frames, inner_ty, inner)?;
                inner = new_inner;
                let Some(x) = option_payload_opt(item) else {
                    break;
                };
                if to_skip > 0 {
                    to_skip -= 1;
                    continue;
                }
                result = some_value(x);
                break;
            }
            (vec![inner, Value::Uint(0)], result)
        } else if self_ty.starts_with("std::iter::Chain<") {
            // `Chain` is (first as an `Option`, second as an
            // `Option`): the first iterator is stepped while it
            // is present; the step that exhausts it drops it to
            // `None` and pulls from the second in the same call.
            // An exhausted second stays in place -- its own
            // state keeps answering `None` -- exactly as core
            // leaves it.
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Chain state",
                    "[Option<iterator>, Option<iterator>]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let type_args = generic_args_of(self_ty);
            let first_ty = type_args.first().map(String::as_str).unwrap_or("");
            let second_ty = type_args.get(1).map(String::as_str).unwrap_or("");
            let mut first = fields[0].clone();
            let mut second = fields[1].clone();
            let mut result = none_value();
            let mut yielded = false;
            match first.clone() {
                Value::Enum {
                    variant: 1,
                    fields: payload,
                } if payload.len() == 1 => {
                    let (new_first, item) = self.iterator_next_state(
                        frames,
                        first_ty,
                        payload.into_iter().next().expect("length checked"),
                    )?;
                    match option_payload_opt(item) {
                        Some(x) => {
                            first = some_value(new_first);
                            result = some_value(x);
                            yielded = true;
                        }
                        None => first = none_value(),
                    }
                }
                Value::Enum { variant: 0, .. } => {}
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "Chain first state",
                        "an Option holding the first iterator",
                        other.kind_name(),
                    ))
                }
            }
            if !yielded {
                match second.clone() {
                    Value::Enum {
                        variant: 1,
                        fields: payload,
                    } if payload.len() == 1 => {
                        let (new_second, item) = self.iterator_next_state(
                            frames,
                            second_ty,
                            payload.into_iter().next().expect("length checked"),
                        )?;
                        second = some_value(new_second);
                        result = item;
                    }
                    Value::Enum { variant: 0, .. } => {}
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            "Chain second state",
                            "an Option holding the second iterator",
                            other.kind_name(),
                        ))
                    }
                }
            }
            (vec![first, second], result)
        } else {
            return Err(unsupported(format!(
                "Iterator::next for {self_ty} is not runtime-supported"
            )));
        };
        Ok((rebuild_state(state, new_fields), result))

    }

    /// Step an iterator value to exhaustion with the
    /// [`Self::iterator_next`] machinery, returning the items in
    /// order. The iterator arrives by value; it is parked in a
    /// fresh allocation so the place-based machinery can step it.
    /// Callers consume the iterator, so the parked state is never
    /// read back.
    fn drain_iterator(
        &mut self,
        frames: &mut Vec<Frame>,
        iter_ty: &str,
        iter_value: Value,
    ) -> InterpResult<Vec<Value>> {
        let id = self.alloc_insert(iter_value);
        let iter_ref = Value::Ref(RefTarget::Alloc {
            id,
            projection: Vec::new(),
        });
        let mut items = Vec::new();
        loop {
            match self.iterator_next(frames, iter_ty, &iter_ref)? {
                Value::Enum {
                    variant: 1,
                    fields: item,
                } if item.len() == 1 => {
                    items.push(item.into_iter().next().expect("length checked"));
                }
                Value::Enum { variant: 0, .. } => break,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "iterator drain",
                        "an Option value from Iterator::next",
                        other.kind_name(),
                    ))
                }
            }
        }
        Ok(items)
    }

    /// The `(lower, upper)` bounds of an inner iterator
    /// state's `size_hint`, for the adapter arms of
    /// [`Self::iterator_size_hint`]: the state is parked as a
    /// value reference and the hint the modelled step reports
    /// for it is taken apart.
    fn size_hint_parts(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        state: Value,
    ) -> InterpResult<(u128, Option<u128>)> {
        let inner_ref = Value::Ref(RefTarget::value(state));
        match self.iterator_size_hint(frames, self_ty, &inner_ref)? {
            Value::Tuple(pair) if pair.len() == 2 => {
                let lower = match pair[0] {
                    Value::Uint(v) => v,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            format!("size_hint lower bound of {self_ty}"),
                            "unsigned integer",
                            other.kind_name(),
                        ))
                    }
                };
                let upper = match option_payload_opt(pair[1].clone()) {
                    Some(Value::Uint(v)) => Some(v),
                    Some(ref other) => {
                        return Err(InterpError::type_mismatch(
                            format!("size_hint upper bound of {self_ty}"),
                            "unsigned integer",
                            other.kind_name(),
                        ))
                    }
                    None => None,
                };
                Ok((lower, upper))
            }
            other => Err(InterpError::type_mismatch(
                format!("size_hint of {self_ty}"),
                "(usize, Option<usize>)",
                other.kind_name(),
            )),
        }
    }

    /// `Iterator::size_hint` for the modelled iterators: the
    /// `(lower, Option<upper>)` pair the standard library reports
    /// for each, computed from the iterator's state the same way
    /// [`Self::iterator_next`] steps it.
    ///
    /// Registering the name stops the driver collecting the real
    /// `size_hint` bodies. Those bodies cannot run against the
    /// modelled states: `env::Args::size_hint` — the caller that
    /// motivated this — delegates through `ArgsOs` to
    /// `vec::IntoIter::size_hint`, whose body reads the real
    /// struct layout (the buffer pointer and end fields) off the
    /// model's snapshot state and fails on an uninitialized deep
    /// field; the range and string iterators' bodies project
    /// their real layouts the same way. The values below are the
    /// exact std answers for each modelled state: an exact count
    /// where std knows one, `(0, None)` where std reports the
    /// trait default (the channel iterators do not override
    /// `size_hint`). Iterators the model does not serve fall
    /// back to that same trait default: `(0, None)` is valid
    /// for every iterator, so the call succeeds where the
    /// collected default body would have run before this
    /// builtin was registered — at the price of returning
    /// the default rather than the exact value for an
    /// unmodelled iterator that overrides `size_hint`.
    fn iterator_size_hint(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        ref_arg: &Value,
    ) -> InterpResult<Value> {
        let target = self.deref_value(frames, ref_arg)?;
        let state = self.read_resolved(frames, &target)?;
        fn hint(lower: u128, upper: Option<u128>) -> Value {
            Value::Tuple(vec![
                Value::Uint(lower),
                match upper {
                    Some(v) => some_value(Value::Uint(v)),
                    None => none_value(),
                },
            ])
        }
        if self_ty.starts_with("std::sync::mpsc::Iter<")
            || self_ty.starts_with("std::sync::mpsc::IntoIter<")
        {
            // Not overridden in std: the trait default.
            return Ok(hint(0, None));
        }
        if self_ty_is(self_ty, "std::env::Args") || self_ty_is(self_ty, "std::env::ArgsOs") {
            // Delegates to the inner `vec::IntoIter`'s exact count.
            let depth = if self_ty_is(self_ty, "std::env::Args") { 3 } else { 2 };
            let mut inner = &state;
            for _ in 0..depth {
                inner = match inner {
                    Value::Enum { fields, .. } if fields.len() == 1 => &fields[0],
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("{self_ty} state"),
                            "nested one-field iterator wrappers",
                            other.kind_name(),
                        ))
                    }
                };
            }
            let (items, idx) = owned_iter_parts(inner, self_ty)?;
            let n = items.len().saturating_sub(idx) as u128;
            return Ok(hint(n, Some(n)));
        }
        if self_ty.starts_with("std::iter::Rev<") {
            // `Rev::size_hint` is its inner iterator's.
            let inner_ty = self_ty
                .strip_prefix("std::iter::Rev<")
                .and_then(|rest| rest.strip_suffix('>'))
                .unwrap_or("");
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Rev state",
                    "[inner iterator]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ref = Value::Ref(RefTarget::value(fields[0].clone()));
            return self.iterator_size_hint(frames, inner_ty, &inner_ref);
        }
        if self_ty.starts_with("std::iter::Skip<") {
            // `Skip::size_hint` is the inner hint with the
            // remaining skip count saturated off both bounds.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Skip state",
                    "[iterator, remaining]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let remaining = match fields[1] {
                Value::Uint(v) => v,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "Skip remaining count",
                        "unsigned integer",
                        other.kind_name(),
                    ))
                }
            };
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (lower, upper) =
                self.size_hint_parts(frames, inner_ty, fields[0].clone())?;
            return Ok(hint(
                lower.saturating_sub(remaining),
                upper.map(|u| u.saturating_sub(remaining)),
            ));
        }
        if self_ty.starts_with("std::iter::Chain<") {
            // `Chain::size_hint` adds the two sides' hints; a
            // side already dropped to `None` counts as the
            // exact `(0, Some(0))`, and the upper bound is
            // known only when both sides' are.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Chain state",
                    "[Option<iterator>, Option<iterator>]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let type_args = generic_args_of(self_ty);
            let mut lower = 0u128;
            let mut upper = Some(0u128);
            for (i, side) in fields.iter().enumerate() {
                let side_ty = type_args.get(i).map(String::as_str).unwrap_or("");
                let (l, u) = match side {
                    Value::Enum {
                        variant: 1,
                        fields: payload,
                    } if payload.len() == 1 => {
                        self.size_hint_parts(frames, side_ty, payload[0].clone())?
                    }
                    Value::Enum { variant: 0, .. } => (0, Some(0)),
                    other => {
                        return Err(InterpError::type_mismatch(
                            "Chain state",
                            "an Option holding an iterator",
                            other.kind_name(),
                        ))
                    }
                };
                lower = lower.saturating_add(l);
                upper = match (upper, u) {
                    (Some(a), Some(b)) => a.checked_add(b),
                    _ => None,
                };
            }
            return Ok(hint(lower, upper));
        }
        let fields = iterator_fields(&state, self_ty)?;
        if self_ty.starts_with("std::ops::RangeInclusive<") {
            if fields.len() != 3 {
                return Err(InterpError::type_mismatch(
                    "RangeInclusive state",
                    "[start, end, exhausted]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            if fields[2] == Value::Bool(true) {
                return Ok(hint(0, Some(0)));
            }
            return Ok(match range_count(
                &fields[0],
                &fields[1],
                true,
                range_elem_ty(self_ty),
            )? {
                Some(n) => hint(n, Some(n)),
                None => hint(u128::from(u64::MAX), None),
            });
        }
        if self_ty.starts_with("std::ops::Range<") {
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Range state",
                    "[start, end]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            return Ok(match range_count(
                &fields[0],
                &fields[1],
                false,
                range_elem_ty(self_ty),
            )? {
                Some(n) => hint(n, Some(n)),
                None => hint(u128::from(u64::MAX), None),
            });
        }
        if self_ty.starts_with("std::iter::Empty<") {
            return Ok(hint(0, Some(0)));
        }
        if self_ty.starts_with("std::slice::Iter<") {
            // The real layout's pointer pair: the remaining count
            // is the distance between the two element pointers.
            let (cur, end) = slice_iter_bounds(&state, self_ty)?;
            let n = end.saturating_sub(cur) as u128;
            return Ok(hint(n, Some(n)));
        }
        if self_ty.starts_with("std::array::IntoIter<")
            || self_ty.starts_with("std::vec::IntoIter<")
            || self_ty.starts_with("alloc::vec::IntoIter<")
            || self_ty_is(self_ty, "std::str::Chars")
            || self_ty_is(self_ty, "core::str::Chars")
            || self_ty_is(self_ty, "std::str::Bytes")
            || self_ty_is(self_ty, "core::str::Bytes")
        {
            let (items, idx) = owned_iter_parts(&state, self_ty)?;
            let remaining = &items[idx.min(items.len())..];
            if self_ty_is(self_ty, "std::str::Chars")
                || self_ty_is(self_ty, "core::str::Chars")
            {
                // std's `Chars::size_hint`: the remaining byte
                // count bounds the char count from above, and a
                // char is at most 4 bytes, so the lower bound is
                // the byte count divided by 4 rounded up. The
                // state holds the decoded code points, so the
                // remaining byte count is exact.
                let mut bytes: u128 = 0;
                for item in remaining {
                    bytes += match item {
                        Value::Uint(cp) => char::from_u32(*cp as u32)
                            .map(|c| c.len_utf8() as u128)
                            .ok_or_else(|| {
                                InterpError::type_mismatch(
                                    "Chars state",
                                    "char code points",
                                    format!("code point {cp}"),
                                )
                            })?,
                        Value::Str(text) => text.len() as u128,
                        other => {
                            return Err(InterpError::type_mismatch(
                                "Chars state",
                                "char values",
                                other.kind_name(),
                            ))
                        }
                    };
                }
                return Ok(hint(bytes.div_ceil(4), Some(bytes)));
            }
            let n = remaining.len() as u128;
            return Ok(hint(n, Some(n)));
        }
        // Any other iterator: the trait default, `(0, None)`.
        // That pair is a valid `size_hint` answer for every
        // iterator — a lower bound of zero and no upper bound
        // satisfy the contract unconditionally — and it is the
        // exact answer for iterators that do not override
        // `size_hint` (a plain user iterator behind adapters
        // such as `Fuse` is the common case). For an iterator
        // that DOES override `size_hint` but whose state the
        // model does not serve, this returns the contract-valid
        // default rather than the implementation's exact value;
        // callers may then under-reserve or check bounds less
        // tightly, but never observe an invalid bound. (Before
        // this fallback existed, registering the builtin name
        // removed the default body from collection, so such
        // calls failed outright even when the default was the
        // whole story.)
        Ok(hint(0, None))
    }

    /// Execute a primitive arithmetic trait method served by
    /// the run-time name fallback: the name carries the Self
    /// type, whose tag selects the operation's width and
    /// signedness, and the computation is the interpreter's own
    /// binary evaluation (division by zero panics there, as the
    /// operation itself requires). In an overflow-checked build
    /// the core impls panic on overflow — the same panic the
    /// `Assert` after a checked MIR operation raises — so the
    /// checked form's flag is consulted here; otherwise the
    /// wrapped result stands.
    fn exec_primitive_binop(
        &self,
        name: &str,
        op: BinOp,
        lhs: &Value,
        rhs: &Value,
    ) -> InterpResult<Value> {
        let ty = primitive_self_ty(name)?;
        let (result, overflowed) = eval_binary(op, lhs, rhs, Some(&ty), true)?;
        if overflowed && self.overflow_checks() {
            return Err(InterpError::panic_str_payload(
                primitive_overflow_message(op),
            ));
        }
        Ok(result)
    }

    /// The unary counterpart: `Neg` on the Self type's minimum
    /// is the one overflowing case (the checked form of the
    /// unary evaluation wraps it, leaving the panic to the
    /// following `Assert` in MIR; here the call is the whole
    /// operation, so the panic is raised directly).
    fn exec_primitive_unop(
        &self,
        name: &str,
        op: UnOp,
        value: &Value,
    ) -> InterpResult<Value> {
        let ty = primitive_self_ty(name)?;
        if op == UnOp::Neg && self.overflow_checks() {
            if let (Value::Int(a), Ty::Int(it)) = (value, &ty) {
                let bits = it.bits();
                let min = if bits >= 128 {
                    i128::MIN
                } else {
                    -(1i128 << (bits - 1))
                };
                if *a == min {
                    return Err(InterpError::panic_str_payload(
                        "attempt to negate with overflow",
                    ));
                }
            }
        }
        eval_unary(op, value, Some(&ty))
    }

    /// The bytes of a `&[u8]` argument.
    fn arg_bytes(&self, frames: &[Frame], value: &Value) -> InterpResult<Vec<u8>> {
        let (items, _) = self.slice_elements(frames, value)?;
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            match item {
                Value::Uint(b) => out.push(b as u8),
                other => {
                    return Err(InterpError::type_mismatch(
                        "byte slice argument",
                        "byte elements",
                        other.kind_name(),
                    ))
                }
            }
        }
        Ok(out)
    }

    /// A `CString` value over `bytes` (given without the
    /// trailing NUL): the struct's single field is the
    /// owned boxed byte slice, built the way
    /// `Vec::into_boxed_slice` builds it (an allocation
    /// holding the byte elements, terminator appended).
    fn c_string_value(&self, mut bytes: Vec<u8>) -> Value {
        bytes.push(0);
        let elems: Vec<Value> = bytes
            .iter()
            .map(|b| Value::Uint(u128::from(*b)))
            .collect();
        Value::Enum {
            variant: 0,
            fields: vec![self.new_box(Value::Tuple(elems))],
        }
    }

    /// The `&CStr` a `CString` dereferences to: the fat
    /// reference over the owned buffer, terminator
    /// included. The base is the inner box's allocation
    /// pointer and the length is the buffer's byte count.
    fn c_string_deref(
        &self,
        frames: &[Frame],
        receiver: &Value,
    ) -> InterpResult<Value> {
        let pointee = self.resolve_value(frames, receiver)?;
        match pointee {
            Value::Enum { fields, .. } if fields.len() == 1 => match &fields[0] {
                Value::FatRef { .. } => Ok(fields[0].clone()),
                Value::Enum {
                    fields: box_fields,
                    ..
                } if box_fields.len() == 1 => {
                    let base = box_fields[0].clone();
                    let buf = self.resolve_value(frames, &base)?;
                    let len = buf.len().ok_or_else(|| {
                        InterpError::type_mismatch(
                            "Deref for CString",
                            "a byte buffer value",
                            buf.kind_name(),
                        )
                    })?;
                    Ok(Value::FatRef {
                        base: Box::new(base),
                        len,
                    })
                }
                other => Err(InterpError::type_mismatch(
                    "Deref for CString",
                    "a boxed byte slice",
                    other.kind_name(),
                )),
            },
            ref other => Err(InterpError::type_mismatch(
                "Deref for CString",
                "cstring aggregate",
                other.kind_name(),
            )),
        }
    }

    /// The bytes a `&CStr` receiver denotes, without the
    /// terminating NUL: the receiver is the fat byte slice
    /// the C string was built from, terminator included, so
    /// the last byte (always a NUL in a well-formed value)
    /// is dropped.
    fn c_str_bytes(&self, frames: &[Frame], value: &Value) -> InterpResult<Vec<u8>> {
        let mut bytes = self.arg_bytes(frames, value)?;
        if bytes.last() == Some(&0) {
            bytes.pop();
        }
        Ok(bytes)
    }

    /// The fat reference `value` shortened to its first
    /// `len` elements, the sub-slice view
    /// `CStr::from_bytes_until_nul` returns: a snapshot of
    /// the prefix elements under the shortened length.
    fn fat_ref_prefix(
        &self,
        frames: &[Frame],
        value: &Value,
        len: usize,
    ) -> InterpResult<Value> {
        // The prefix is materialised as a snapshot of its
        // elements: a fat reference that keeps the source's
        // base would have its length re-derived from that
        // base (the full slice) when the value is written
        // to a typed place, losing the shortening.
        let (items, _) = self.slice_elements(frames, value)?;
        if items.len() < len {
            return Err(InterpError::type_mismatch(
                "sub-slice prefix",
                "a slice of at least the requested length",
                format!("{} elements", items.len()),
            ));
        }
        Ok(Value::FatRef {
            base: Box::new(Value::Ref(RefTarget::Value(
                Box::new(Value::Tuple(items.into_iter().take(len).collect())),
                Vec::new(),
            ))),
            len: len as u128,
        })
    }

    /// Whether a comparison call compares paths: the
    /// call's generic arguments name a path type
    /// (`path_cmp_subject`), or — for a call the converter
    /// kept in generic form, which carries no type tags —
    /// both operands are owned `PathBuf` values by shape
    /// (`is_pathbuf_value`).
    fn comparison_is_path(
        &self,
        frames: &[Frame],
        generic_args: &[String],
        vals: &[Value],
    ) -> bool {
        if path_cmp_subject(generic_args) {
            return true;
        }
        if !generic_args.is_empty() || vals.len() != 2 {
            return false;
        }
        vals.iter().all(|v| {
            self.resolve_value(frames, v)
                .is_ok_and(|resolved| is_pathbuf_value(&resolved))
        })
    }

    /// The bytes of a path argument: a path value is a fat
    /// pointer over the path's bytes (or a reference to one).
    /// The byte string a `PartialEq` operand for a path type
    /// denotes: the fat-pointer form reads through
    /// [`Self::path_bytes`]; an owned `PathBuf` value (a
    /// one-field aggregate chain around the string) reads
    /// through [`Self::os_str_bytes`].
    fn path_cmp_bytes(&self, frames: &[Frame], value: &Value) -> InterpResult<Vec<u8>> {
        if let Ok(bytes) = self.path_bytes(frames, value) {
            return Ok(bytes);
        }
        // An owned path value may be stored in the aggregate
        // shapes `string_aggregate_bytes` recognises (a wrapper
        // chain around a byte-element aggregate, as the
        // `current_exe` model's `PathBuf` is stored).
        if let Ok(resolved) = self.resolve_value(frames, value) {
            if let Some(bytes) = string_aggregate_bytes(&resolved) {
                return Ok(bytes);
            }
        }
        self.os_str_bytes(frames, value)
    }

    fn path_bytes(&self, frames: &[Frame], value: &Value) -> InterpResult<Vec<u8>> {
        match value {
            Value::Str(text) => Ok(text.clone().into_bytes()),
            Value::Bytes(bytes) => Ok(bytes.clone()),
            Value::FatRef { .. } => {
                let (items, _) = self.slice_elements(frames, value)?;
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        Value::Uint(b) => out.push(b as u8),
                        other => {
                            return Err(InterpError::type_mismatch(
                                "path bytes",
                                "byte elements",
                                other.kind_name(),
                            ))
                        }
                    }
                }
                Ok(out)
            }
            Value::Ref(_) => {
                let target = self.deref_value(frames, value)?;
                let stored = self.read_resolved(frames, &target)?;
                self.path_bytes(frames, &stored)
            }
            other => Err(InterpError::type_mismatch(
                "path argument",
                "path (fat pointer) value",
                other.kind_name(),
            )),
        }
    }

    /// Execute a `std::sys::fs` leaf operation on the host
    /// filesystem. Every operation first rejects a path
    /// containing a NUL byte with `InvalidInput`, exactly as
    /// std's own CString conversion does before any syscall;
    /// the rest are the host operations themselves, with the
    /// host's errno returned in std's `Os` error shape (the
    /// guest's own `Error::kind` body maps it from there).
    fn exec_fs_sys(
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

    /// Execute `Iterator::collect` / `FromIterator::from_iter`:
    /// drain the modelled `next` state transition into the
    /// collection the call's target generic names. The value
    /// model has a native shape for two targets: a `Vec` is its
    /// element list, and a `String` is accumulated text (chars
    /// arrive as code-point integers, string items as string
    /// values or references to them). Every other target is
    /// named in an explicit unsupported error rather than
    /// approximated.
    fn exec_collect(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &str,
        mut state: Value,
    ) -> InterpResult<Value> {
        let mut items = Vec::new();
        loop {
            self.tick_step()?;
            let (next_state, item) = self.iterator_next_state(frames, self_ty, state)?;
            state = next_state;
            match option_payload(item)? {
                None => break,
                Some(item) => items.push(item),
            }
        }
        if target.starts_with("std::vec::Vec<") || target.starts_with("alloc::vec::Vec<") {
            return Ok(Value::Enum {
                variant: 0,
                fields: items,
            });
        }
        if target == "std::string::String" || target == "alloc::string::String" {
            let mut out = String::new();
            for item in items {
                match item {
                    Value::Uint(cp) => {
                        let c = char::from_u32(cp as u32).ok_or_else(|| {
                            InterpError::type_mismatch(
                                "collect into String",
                                "valid char code point",
                                format!("{cp}"),
                            )
                        })?;
                        out.push(c);
                    }
                    Value::Str(s) => out.push_str(&s),
                    Value::Ref(_) => {
                        let s = self.resolve_string(frames, &item)?;
                        out.push_str(&s);
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            "collect into String",
                            "char or string item",
                            other.kind_name(),
                        ));
                    }
                }
            }
            return Ok(Value::Str(out));
        }
        Err(unsupported(format!(
            "collect into {target} is not runtime-supported"
        )))
    }

    /// Execute `Iterator::fold` / `Iterator::for_each`: drive
    /// the modelled `next` state transition over the owned
    /// iterator value and invoke the closure operand per item —
    /// exactly the loop the core default bodies spell out. The
    /// closure is invoked through its operand (not a
    /// pre-evaluated value) so its environment aliases its
    /// place, as through the real `FnMut` shim, and mutations
    /// of its captures persist from one item to the next.
    fn exec_iterator_fold_like(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        kind: Builtin,
        generic_args: &[String],
        args: &[Operand],
    ) -> InterpResult<Value> {
        let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
        let (mut state, mut accum, f) = match kind {
            Builtin::IteratorFold => {
                if args.len() != 3 {
                    return Err(InterpError::type_mismatch(
                        "builtin Iterator::fold",
                        "3 argument(s)",
                        format!("{} argument(s)", args.len()),
                    ));
                }
                let state = self.eval_operand(frames, body, &args[0])?;
                let accum = self.eval_operand(frames, body, &args[1])?;
                (state, accum, &args[2])
            }
            Builtin::IteratorForEach => {
                if args.len() != 2 {
                    return Err(InterpError::type_mismatch(
                        "builtin Iterator::for_each",
                        "2 argument(s)",
                        format!("{} argument(s)", args.len()),
                    ));
                }
                let state = self.eval_operand(frames, body, &args[0])?;
                (state, Value::Unit, &args[1])
            }
            _ => unreachable!("fold-like dispatch covers fold and for_each only"),
        };
        loop {
            self.tick_step()?;
            let (next_state, item) = self.iterator_next_state(frames, self_ty, state)?;
            state = next_state;
            let Some(item) = option_payload(item)? else {
                break;
            };
            let call_args = match kind {
                Builtin::IteratorFold => {
                    vec![std::mem::replace(&mut accum, Value::Unit), item]
                }
                _ => vec![item],
            };
            let result = self.call_callable_operand_to_completion(frames, body, f, call_args)?;
            if kind == Builtin::IteratorFold {
                accum = result;
            }
        }
        Ok(accum)
    }

    /// Semantic (derived `PartialEq`) equality of two values:
    /// references are followed to the data they denote and aggregates
    /// compare element-wise. This is the equality of the *data*, not
    /// of the representation — exactly what the derived `PartialEq`
    /// impls of the modelled types compute.
    /// The logical element sequence of a `VecDeque` value (the
    /// real aggregate: wrapped head index, length, and the ring
    /// buffer), in iteration order, when the value has exactly
    /// that shape; `None` for any other value, and for a deque
    /// whose buffer slots cannot be read back (a physical slot
    /// of the logical range was never written -- not a state a
    /// well-formed deque reaches). The buffer's physical layout
    /// is not part of the sequence.
    fn vec_deque_elements(
        &self,
        value: &Value,
    ) -> InterpResult<Option<Vec<Value>>> {
        let Value::Enum { fields, .. } = value else {
            return Ok(None);
        };
        if fields.len() != 3 {
            return Ok(None);
        }
        let Value::Enum {
            fields: head_fields, ..
        } = &fields[0]
        else {
            return Ok(None);
        };
        let [Value::Uint(head)] = head_fields.as_slice() else {
            return Ok(None);
        };
        let Value::Uint(len) = fields[1] else {
            return Ok(None);
        };
        let Value::Enum {
            fields: buf_fields, ..
        } = &fields[2]
        else {
            return Ok(None);
        };
        if buf_fields.len() != 2 {
            return Ok(None);
        }
        let Value::Enum {
            fields: inner_fields, ..
        } = &buf_fields[0]
        else {
            return Ok(None);
        };
        if inner_fields.len() != 3 {
            return Ok(None);
        }
        let Value::Enum {
            fields: ptr_fields, ..
        } = &inner_fields[0]
        else {
            return Ok(None);
        };
        let [ptr] = ptr_fields.as_slice() else {
            return Ok(None);
        };
        let Value::Enum {
            fields: cap_fields, ..
        } = &inner_fields[1]
        else {
            return Ok(None);
        };
        let [Value::Uint(cap)] = cap_fields.as_slice() else {
            return Ok(None);
        };
        let (head, len, cap) = (*head as usize, len as usize, *cap as usize);
        if len == 0 {
            return Ok(Some(Vec::new()));
        }
        if cap == 0 {
            return Ok(None);
        }
        // The buffer's element block: the allocation the
        // `RawVec` pointer denotes, whose stored value is the
        // element tuple.
        let Value::Ref(RefTarget::Alloc { id, .. }) = ptr else {
            return Ok(None);
        };
        let Ok(buffer) = self.alloc_base(*id) else {
            return Ok(None);
        };
        let Value::Tuple(slots) = buffer else {
            return Ok(None);
        };
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let physical = (head + i) % cap;
            let Some(elem) = slots.get(physical) else {
                return Ok(None);
            };
            if *elem == Value::Uninit {
                return Ok(None);
            }
            out.push(elem.clone());
        }
        Ok(Some(out))
    }

    pub(crate) fn values_equal(
        &self,
        frames: &[Frame],
        a: &Value,
        b: &Value,
    ) -> InterpResult<bool> {
        if matches!(a, Value::FatRef { .. }) || matches!(b, Value::FatRef { .. }) {
            let (ea, la) = self.slice_elements_ref(frames, a)?;
            let (eb, lb) = self.slice_elements_ref(frames, b)?;
            if la != lb {
                return Ok(false);
            }
            return self.elems_equal(frames, &ea, &eb);
        }
        // An address-only pointer (the value
        // `simd_with_exposed_provenance` and pointer arithmetic
        // produce) denotes no place in the current frames, so
        // there is no pointee to resolve to: it compares by
        // address against the other side's synthetic address,
        // the identity every pointer value carries in this
        // model.
        if matches!(a, Value::Ref(RefTarget::Address(_)))
            || matches!(b, Value::Ref(RefTarget::Address(_)))
        {
            if let (Some(x), Some(y)) = (synthetic_address(a), synthetic_address(b)) {
                return Ok(x == y);
            }
        }
        let a = self.resolve_ref(frames, a)?;
        let b = self.resolve_ref(frames, b)?;
        // A `VecDeque` value (the real aggregate: wrapped head,
        // length, ring buffer) compares by its logical element
        // sequence, as the type's own `PartialEq` does: the
        // physical buffer layout -- where the head sits, the
        // capacity, the buffer pointer's projection form -- is
        // not part of the value's identity, so two deques with
        // the same elements in the same order are equal however
        // their buffers happen to be arranged.
        if let (Some(ea), Some(eb)) = (
            self.vec_deque_elements(a.as_ref())?,
            self.vec_deque_elements(b.as_ref())?,
        ) {
            return Ok(ea.len() == eb.len() && self.seq_equal(frames, &ea, &eb)?);
        }
        // A `dyn Trait` value denotes its concrete pointee: two
        // trait objects compare equal when their concrete types
        // agree and the pointees compare equal, and unequal when
        // the concrete types differ -- the semantics a hand-written
        // `PartialEq for dyn Trait` computes. For the primitive
        // `PartialEq` supertrait comparisons (the trait-upcasting
        // tests, whose concrete type is `i32`) the structural
        // comparison of the pointee is the concrete type's own
        // `PartialEq`; for a user-defined `PartialEq` the builtin's
        // structural evaluation is the same policy the model applies
        // to direct calls. A dyn side against a sized side compares
        // the pointee against that side.
        if matches!(&*a, Value::DynRef { .. }) || matches!(&*b, Value::DynRef { .. }) {
            if let (
                Value::DynRef { concrete: ca, .. },
                Value::DynRef { concrete: cb, .. },
            ) = (&*a, &*b)
            {
                if ca != cb {
                    return Ok(false);
                }
            }
            let pa = self.dyn_pointee(frames, &a)?;
            let pb = self.dyn_pointee(frames, &b)?;
            return self.values_equal(frames, &pa, &pb);
        }
        // A side that resolves to a self-contained fat value (a
        // slice constant or materialised sub-slice held in a local)
        // compares as a slice against the other side's elements:
        // the other side may resolve to the bare array value, whose
        // elements are exactly its own.
        if matches!(&*a, Value::FatRef { .. }) || matches!(&*b, Value::FatRef { .. }) {
            let (ea, la) = self.slice_elements_ref(frames, &a)?;
            let (eb, lb) = self.slice_elements_ref(frames, &b)?;
            if la != lb {
                return Ok(false);
            }
            return match (&ea, &eb) {
                (SliceElems::Values(x), SliceElems::Values(y)) => {
                    self.seq_equal(frames, x, y)
                }
                _ => self.elems_equal(frames, &ea, &eb),
            };
        }
        Ok(match (&*a, &*b) {
            (Value::Ref(RefTarget::Null), Value::Ref(RefTarget::Null)) => true,
            // Address-only pointers compare by address; address 0
            // is the null address.
            (Value::Ref(RefTarget::Address(x)), Value::Ref(RefTarget::Address(y))) => {
                x == y
            }
            (Value::Ref(RefTarget::Address(0)), Value::Ref(RefTarget::Null))
            | (Value::Ref(RefTarget::Null), Value::Ref(RefTarget::Address(0))) => true,
            (Value::Ref(RefTarget::Null), _) | (_, Value::Ref(RefTarget::Null)) => false,
            (Value::Unit, Value::Unit) => true,
            (Value::Bool(x), Value::Bool(y)) => x == y,
            (Value::Int(x), Value::Int(y)) => x == y,
            (Value::Uint(x), Value::Uint(y)) => x == y,
            (Value::Int(x), Value::Uint(y)) => int_uint_eq(*x, *y),
            (Value::Uint(x), Value::Int(y)) => int_uint_eq(*y, *x),
            (Value::Str(x), Value::Str(y)) => x == y,
            (Value::Bytes(x), Value::Bytes(y)) => x == y,
            // Float equality is IEEE equality, as derived
            // `PartialEq` on a float field performs it (NaN compares
            // unequal to itself).
            (Value::Float(x), Value::Float(y)) => x == y,
            (Value::Float128(x), Value::Float128(y)) => crate::f128::eq(*x, *y),
            (Value::FnPtr(x), Value::FnPtr(y)) => x == y,
            (Value::Tuple(x), Value::Tuple(y)) => {
                x.len() == y.len() && self.seq_equal(frames, x, y)?
            }
            // A byte string and an array of byte values are the
            // same data in two of the model's shapes (a `[u8; N]`
            // constant decodes to bytes; the same array built by an
            // aggregate is a tuple).
            (Value::Bytes(x), Value::Tuple(y)) | (Value::Tuple(y), Value::Bytes(x)) => {
                x.len() == y.len()
                    && x.iter().zip(y.iter()).all(|(byte, elem)| {
                        matches!(elem, Value::Uint(v) if *v == u128::from(*byte))
                            || matches!(elem, Value::Int(v) if *v == i128::from(*byte))
                    })
            }
            // A byte string against an ADT value whose single
            // field is a sequence: the holder is the sequence it
            // holds -- a vector read through an array-typed
            // pointer (`simd.as_array()`) against a byte-array
            // literal is the array against the bytes, not a
            // one-element tuple of it.
            (Value::Enum { variant: 0, fields: x }, Value::Bytes(_))
            | (Value::Bytes(_), Value::Enum { variant: 0, fields: x })
                if x.len() == 1
                    && matches!(x[0], Value::Tuple(_) | Value::Bytes(_)) =>
            {
                let field = x[0].clone();
                let bytes_side = if matches!(&*a, Value::Bytes(_)) {
                    &*a
                } else {
                    &*b
                };
                return self.values_equal(frames, &field, bytes_side);
            }
            // The same holder against a tuple, when the tuple
            // cannot be the holder's field list (its length
            // differs from the field count): it is the held
            // sequence's own tuple form.
            (Value::Enum { variant: 0, fields: x }, Value::Tuple(y))
            | (Value::Tuple(y), Value::Enum { variant: 0, fields: x })
                if x.len() == 1
                    && y.len() != 1
                    && matches!(x[0], Value::Tuple(_) | Value::Bytes(_)) =>
            {
                let field = x[0].clone();
                let tuple_side = if matches!(&*a, Value::Tuple(_)) {
                    &*a
                } else {
                    &*b
                };
                return self.values_equal(frames, &field, tuple_side);
            }
            // A single-variant ADT value (a tuple struct) and a
            // plain tuple of the same fields: the constant decoder
            // and aggregate construction produce the two shapes for
            // the same source value.
            (
                Value::Enum { variant: 0, fields: x },
                Value::Tuple(y),
            )
            | (
                Value::Tuple(y),
                Value::Enum { variant: 0, fields: x },
            ) => x.len() == y.len() && self.seq_equal(frames, x, y)?,
            (
                Value::Enum {
                    variant: va,
                    fields: fa,
                },
                Value::Enum {
                    variant: vb,
                    fields: fb,
                },
            ) => va == vb && fa.len() == fb.len() && self.seq_equal(frames, fa, fb)?,
            _ => {
                return Err(InterpError::type_mismatch(
                    "value equality",
                    "comparable values",
                    format!("{} and {}", a.kind_name(), b.kind_name()),
                ))
            }
        })
    }

    /// The concrete value a `dyn Trait` value denotes: the
    /// pointee of its data pointer. A box-shaped base (the
    /// one-field handle aggregate a `Box` / `Rc` / `Arc` coercion
    /// records, the same judgment the dereference of a dyn fat
    /// pointer makes) holds the pointee as its only field; a
    /// reference or slice base resolves to the value it points at.
    /// A non-dyn value denotes itself. An upcast chain (a box
    /// whose field is itself a `DynRef`) yields that `DynRef`,
    /// which the caller's recursion unwraps one layer at a time.
    fn dyn_pointee(&self, frames: &[Frame], value: &Value) -> InterpResult<Value> {
        match value {
            Value::DynRef { base, .. } => match &**base {
                Value::Enum { fields, .. } if fields.len() == 1 => Ok(fields[0].clone()),
                _ => self.resolve_value(frames, base),
            },
            other => Ok(other.clone()),
        }
    }

    fn seq_equal(&self, frames: &[Frame], a: &[Value], b: &[Value]) -> InterpResult<bool> {
        for (x, y) in a.iter().zip(b.iter()) {
            if !self.values_equal(frames, x, y)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Element-wise equality of two borrowed element lists: the
    /// slice arm of `values_equal` without materialising either
    /// side's elements, so a comparison that decides early does not
    /// pay for a snapshot of the rest.
    fn elems_equal(
        &self,
        frames: &[Frame],
        a: &SliceElems<'_>,
        b: &SliceElems<'_>,
    ) -> InterpResult<bool> {
        for i in 0..a.len().min(b.len()) {
            let (x, y) = (a.get(i), b.get(i));
            if !self.values_equal(frames, &x, &y)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Semantic (derived `Ord`) comparison of two values: integers
    /// and booleans by value, strings byte-wise (as `str`'s `Ord`
    /// is), aggregates lexicographically with enum variants ordered
    /// by declaration order — the derived semantics.
/// The discriminant value a variant index stands for in a
/// structural comparison: the baked table's entry when the
/// comparison's `Self` type has one, else the index itself (the
/// sequential default, where the two agree).
fn cmp_discriminant(discriminants: Option<&[i128]>, variant: u32) -> i128 {
    discriminants
        .and_then(|table| table.get(variant as usize).copied())
        .unwrap_or(variant as i128)
}

    /// The comparison behind the `PartialOrd` / `Ord` builtins,
    /// with `partial_cmp` semantics: the result is `None` when the
    /// values are unordered (a float comparison involving NaN,
    /// propagated out of the first field or element whose comparison
    /// is unordered, exactly as a derived `partial_cmp` returns the
    /// first non-`Equal` field result). Callers that need a total
    /// ordering treat `None` as an error at their own site.
    pub(crate) fn values_cmp(
        &self,
        frames: &[Frame],
        a: &Value,
        b: &Value,
        discriminants: Option<&[i128]>,
    ) -> InterpResult<Option<Ordering>> {
        if matches!(a, Value::FatRef { .. }) || matches!(b, Value::FatRef { .. }) {
            let (ea, la) = self.slice_elements_ref(frames, a)?;
            let (eb, lb) = self.slice_elements_ref(frames, b)?;
            match self.elems_cmp(frames, &ea, &eb)? {
                Some(Ordering::Equal) => {}
                other => return Ok(other),
            }
            return Ok(Some(la.cmp(&lb)));
        }
        let a = self.resolve_ref(frames, a)?;
        let b = self.resolve_ref(frames, b)?;
        Ok(match (&*a, &*b) {
            (Value::Unit, Value::Unit) => Some(Ordering::Equal),
            (Value::Bool(x), Value::Bool(y)) => Some(x.cmp(y)),
            (Value::Int(x), Value::Int(y)) => Some(x.cmp(y)),
            (Value::Uint(x), Value::Uint(y)) => Some(x.cmp(y)),
            (Value::Int(x), Value::Uint(y)) => Some(int_uint_cmp(*x, *y)),
            (Value::Uint(x), Value::Int(y)) => Some(int_uint_cmp(*y, *x).reverse()),
            (Value::Float(x), Value::Float(y)) => x.partial_cmp(y),
            (Value::Str(x), Value::Str(y)) => Some(x.as_bytes().cmp(y.as_bytes())),
            (Value::Bytes(x), Value::Bytes(y)) => Some(x.cmp(y)),
            (Value::Tuple(x), Value::Tuple(y)) => self.seq_cmp(frames, x, y)?,
            (
                Value::Enum {
                    variant: va,
                    fields: fa,
                },
                Value::Enum {
                    variant: vb,
                    fields: fb,
                },
            ) => match Self::cmp_discriminant(discriminants, *va)
                .cmp(&Self::cmp_discriminant(discriminants, *vb))
            {
                Ordering::Equal => self.seq_cmp(frames, fa, fb)?,
                other => Some(other),
            },
            _ => {
                return Err(InterpError::type_mismatch(
                    "value comparison",
                    "comparable values",
                    format!("{} and {}", a.kind_name(), b.kind_name()),
                ))
            }
        })
    }

    fn seq_cmp(
        &self,
        frames: &[Frame],
        a: &[Value],
        b: &[Value],
    ) -> InterpResult<Option<Ordering>> {
        for (x, y) in a.iter().zip(b.iter()) {
            match self.values_cmp(frames, x, y, None)? {
                Some(Ordering::Equal) => {}
                other => return Ok(other),
            }
        }
        Ok(Some(a.len().cmp(&b.len())))
    }

    /// Element-wise comparison of two borrowed element lists: the
    /// slice arm of `values_cmp` without materialising either
    /// side's elements. The result is the first non-`Equal` element
    /// result, or `Equal` when the common prefix compares equal
    /// (the caller applies the length tiebreak).
    fn elems_cmp(
        &self,
        frames: &[Frame],
        a: &SliceElems<'_>,
        b: &SliceElems<'_>,
    ) -> InterpResult<Option<Ordering>> {
        for i in 0..a.len().min(b.len()) {
            let (x, y) = (a.get(i), b.get(i));
            match self.values_cmp(frames, &x, &y, None)? {
                Some(Ordering::Equal) => {}
                other => return Ok(other),
            }
        }
        Ok(Some(Ordering::Equal))
    }

    /// The elements a slice-like value denotes, with its length: a
    /// fat reference contributes its base's elements truncated to the
    /// carried length; any other value contributes its own elements.
    fn slice_elements(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<(Vec<Value>, u128)> {
        let (elems, len) = self.slice_elements_ref(frames, value)?;
        // An owned copy of a sparse sequence's elements would
        // materialise its uninitialised tail: past the
        // materialisation limit that stays an explicit error
        // rather than an allocation that exhausts the host.
        // (Borrowing consumers use `slice_elements_ref` and read
        // the tail as the uninitialised slots they are.)
        if elems.stored_len() < elems.len()
            && elems.len() > MAX_VEC_SET_LEN_ELEMENTS
        {
            return Err(unsupported(format!(
                "a slice of {len} elements is beyond the value model's materialisation limit"
            )));
        }
        Ok((elems.into_vec(), len))
    }

    /// The borrowing form of [`slice_elements`]: the same elements
    /// and length, but the element storage is lent out of the
    /// resolved value wherever it is already materialised, so a
    /// consumer that needs one element (or a comparison that stops
    /// early) does not pay for a snapshot of all of them.
    /// The live element reference for an index into a
    /// place-backed receiver (the shared-`Index` counterpart of
    /// the resolution `IndexMut::index_mut` performs): the
    /// receiver's own place with an `Index` step appended, so
    /// reads and interior-mutability writes reach the container's
    /// storage. `Ok(None)` when the receiver is not backed by a
    /// live place holding an indexable aggregate — the caller
    /// falls back to a snapshot of the element.
    fn index_place_ref(
        &self,
        frames: &[Frame],
        receiver: &Value,
        idx: usize,
        fat_len: Option<u128>,
    ) -> InterpResult<Option<Value>> {
        if !matches!(receiver, Value::Ref(_) | Value::FatRef { .. }) {
            return Ok(None);
        }
        let target = self.deref_value(frames, receiver)?;
        if matches!(target, Resolved::Value(..)) {
            return Ok(None);
        }
        // The base is only inspected (its length and form): read
        // it by borrowing, so indexing a vector does not clone
        // every element before returning one element's reference.
        let base = self.read_resolved_ref(frames, &target)?;
        let len = match fat_len {
            Some(len) => len,
            None => match base.len() {
                Some(len) => len,
                None => return Ok(None),
            },
        };
        if idx as u128 >= len {
            return Err(InterpError::panic(format!(
                "index out of bounds: the len is {len} but the index is {idx}"
            )));
        }
        if !matches!(
            &*base,
            Value::Tuple(_) | Value::Enum { .. } | Value::SparseVec { .. }
        ) {
            return Ok(None);
        }
        let refr = match target {
            Resolved::Place {
                frame_pos,
                local,
                mut projection,
            } => {
                projection.push(RefProj::Index(idx));
                Value::Ref(RefTarget::Place {
                    frame: frames[frame_pos].id,
                    local,
                    projection,
                })
            }
            Resolved::Static {
                name,
                mut projection,
            } => {
                projection.push(RefProj::Index(idx));
                Value::Ref(RefTarget::Static { name, projection })
            }
            Resolved::Alloc { id, mut projection } => {
                projection.push(RefProj::Index(idx));
                Value::Ref(RefTarget::Alloc { id, projection })
            }
            Resolved::Value(..) => unreachable!("returned above"),
        };
        Ok(Some(refr))
    }

    /// Whether the `len` elements a fat pointer denotes can be
    /// the elements of the container its base target sits in,
    /// starting at the target's trailing element index. A fat
    /// pointer formed by dereferencing a container value that
    /// itself sits at an element position (a `Vec` nested inside
    /// another container, bound by a deref pattern) denotes that
    /// value's own elements, whose count the container's
    /// remaining span can fall short of; such a span cannot be
    /// the container's elements, so the answer is no and the
    /// caller reads the pointee's own elements instead. Any
    /// shape this cannot judge answers yes, keeping the sibling
    /// reading.
    fn sibling_span_fits(&self, frames: &[Frame], target: &Resolved, len: u128) -> bool {
        let projection = match target {
            Resolved::Place { projection, .. }
            | Resolved::Static { projection, .. }
            | Resolved::Alloc { projection, .. }
            | Resolved::Value(_, projection) => projection,
        };
        let Some((&RefProj::Index(start), head)) = projection.split_last() else {
            return true;
        };
        let parent = match target {
            Resolved::Place {
                frame_pos,
                local,
                ..
            } => Resolved::Place {
                frame_pos: *frame_pos,
                local: *local,
                projection: head.to_vec(),
            },
            Resolved::Static { name, .. } => Resolved::Static {
                name: name.clone(),
                projection: head.to_vec(),
            },
            Resolved::Alloc { id, .. } => Resolved::Alloc {
                id: *id,
                projection: head.to_vec(),
            },
            Resolved::Value(value, _) => Resolved::Value(value.clone(), head.to_vec()),
        };
        match self.read_resolved(frames, &parent) {
            Ok(container) => match container.len() {
                Some(count) => (start as u128) + len <= count,
                None => true,
            },
            Err(_) => true,
        }
    }

    fn slice_elements_ref<'x>(
        &self,
        frames: &'x [Frame],
        value: &'x Value,
    ) -> InterpResult<(SliceElems<'x>, u128)> {
        match value {
            Value::FatRef { base, len } => {
                // A base pointer denoting one element of a
                // container (a raw-parts slice formed over an
                // offset element pointer) denotes the elements
                // from that position on, not the pointed element
                // itself: read them through the sequence accessor,
                // which continues to the element's siblings in the
                // container (and reads within the element when the
                // element is itself an aggregate).
                if let Ok(target) = self.deref_value(frames, base) {
                    // A sparse pointee's elements past its stored
                    // prefix are read through the ordinary path
                    // below (which reports them as the
                    // uninitialised slots they are); the sibling
                    // walk would materialise them one by one.
                    let sparse_target = matches!(
                        self.read_resolved(frames, &target),
                        Ok(Value::SparseVec { .. })
                    );
                    if !sparse_target
                        && target_ends_at_element(&target)
                        && self.sibling_span_fits(frames, &target, *len)
                    {
                        let mut elems = Vec::with_capacity(*len as usize);
                        for i in 0..(*len as usize) {
                            elems.push(self.read_seq_at(frames, &target, i)?);
                        }
                        return Ok((SliceElems::Values(Cow::Owned(elems)), *len));
                    }
                }
                let base = self.resolve_ref(frames, base)?;
                let elems = elements_of(base)?.truncated(*len as usize);
                Ok((elems, *len))
            }
            _ => {
                let resolved = self.resolve_ref(frames, value)?;
                let elems = elements_of(resolved)?;
                let len = elems.len() as u128;
                Ok((elems, len))
            }
        }
    }

    /// `Index::index` with a range argument: the sub-slice of the
    /// receiver between the range's bounds, with core's own panic
    /// messages for out-of-range bounds. The result is a fat
    /// reference over a materialised copy of the sub-slice's
    /// elements (a shared slice: reads through it are exact; the
    /// model has no offsettable place to alias the original
    /// storage) — or, when the range runs past a sparse
    /// receiver's stored elements, over the sparse form of the
    /// range (see `Value::SparseVec`). A `str` receiver yields
    /// the sub-string, with char-boundary checks as in core.
    fn index_range(
        &self,
        frames: &[Frame],
        name: &str,
        receiver: &Value,
        index: &Value,
        idx_tag: &str,
    ) -> InterpResult<Value> {
        // The bounds as stored in the range value: `Range` and
        // `RangeInclusive` carry [start, end], `RangeFrom` [start],
        // `RangeTo` / `RangeToInclusive` [end], `RangeFull` none
        // (it arrives as the unit value).
        let fields: &[Value] = match index {
            Value::Unit => &[],
            Value::Enum { fields, .. } => fields,
            Value::Tuple(fields) => fields,
            other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "range index value",
                    other.kind_name(),
                ))
            }
        };
        let bound = |i: usize| -> InterpResult<usize> {
            match fields.get(i) {
                Some(v) => expect_index(v, name),
                None => Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "range index value with bounds",
                    index.kind_name(),
                )),
            }
        };
        let kind = if idx_tag.contains("RangeFull") {
            RangeKindForIndex::Full
        } else if idx_tag.contains("RangeToInclusive") {
            RangeKindForIndex::ToInclusive
        } else if idx_tag.contains("RangeInclusive") {
            RangeKindForIndex::Inclusive
        } else if idx_tag.contains("RangeFrom") {
            RangeKindForIndex::From
        } else if idx_tag.contains("RangeTo") {
            RangeKindForIndex::To
        } else {
            RangeKindForIndex::Range
        };
        // Resolve the receiver's length first: for a `str` the
        // length is in bytes, as range indices on strings are byte
        // indices.
        let resolved = self.resolve_value(frames, receiver)?;
        if let Value::Str(text) = &resolved {
            let len = text.len();
            let (start, end) = range_bounds(&kind, &bound, len)?;
            return match text.get(start..end) {
                Some(sub) => Ok(Value::Str(sub.to_string())),
                None => Err(InterpError::panic(str_slice_error(text, start, end))),
            };
        }
        let (elems, len) = self.slice_elements_ref(frames, receiver)?;
        let len = usize::try_from(len).unwrap_or(usize::MAX);
        let (start, end) = range_bounds(&kind, &bound, len)?;
        Ok(Value::FatRef {
            base: Box::new(elems.sub_snapshot(start, end)),
            len: (end - start) as u128,
        })
    }

    /// `IndexMut::index_mut` with a range argument: the mutable
    /// sub-slice of the receiver between the range's bounds, with
    /// core's own panic messages for out-of-range bounds (via
    /// [`range_bounds`], as for the shared `Index`). The result is
    /// a fat reference aliasing the receiver's storage: its base
    /// is the receiver reference with a resolved
    /// `RefProj::Subslice` step appended (the step carries the
    /// range's offset, which is what makes a range starting above
    /// zero expressible), over the receiver reference itself for a
    /// place/static receiver and over the same base for a fat
    /// receiver. A fat receiver whose base is a materialised copy
    /// has no storage to alias and stays explicitly unsupported.
    fn index_range_mut(
        &self,
        frames: &[Frame],
        name: &str,
        receiver: &Value,
        index: &Value,
        idx_tag: &str,
    ) -> InterpResult<Value> {
        // The bounds as stored in the range value, in the same
        // shapes as `index_range`.
        let fields: &[Value] = match index {
            Value::Unit => &[],
            Value::Enum { fields, .. } => fields,
            Value::Tuple(fields) => fields,
            other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "range index value",
                    other.kind_name(),
                ))
            }
        };
        let bound = |i: usize| -> InterpResult<usize> {
            match fields.get(i) {
                Some(v) => expect_index(v, name),
                None => Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "range index value with bounds",
                    index.kind_name(),
                )),
            }
        };
        let kind = if idx_tag.contains("RangeFull") {
            RangeKindForIndex::Full
        } else if idx_tag.contains("RangeToInclusive") {
            RangeKindForIndex::ToInclusive
        } else if idx_tag.contains("RangeInclusive") {
            RangeKindForIndex::Inclusive
        } else if idx_tag.contains("RangeFrom") {
            RangeKindForIndex::From
        } else if idx_tag.contains("RangeTo") {
            RangeKindForIndex::To
        } else {
            RangeKindForIndex::Range
        };
        // The receiver's effective length: a fat receiver's
        // carried length, otherwise the pointed-to value's length.
        let len: usize = match receiver {
            Value::FatRef { len, .. } => usize::try_from(*len).unwrap_or(usize::MAX),
            Value::Ref(_) => {
                let resolved = self.resolve_value(frames, receiver)?;
                if matches!(resolved, Value::Str(_)) {
                    return Err(unsupported(
                        "IndexMut::index_mut with a range on a str is not runtime-supported"
                            .to_string(),
                    ));
                }
                let (_, len) = self.slice_elements(frames, receiver)?;
                usize::try_from(len).unwrap_or(usize::MAX)
            }
            other => {
                return Err(InterpError::type_mismatch(
                    "IndexMut::index_mut",
                    "mutable reference",
                    other.kind_name(),
                ))
            }
        };
        let (start, end) = range_bounds(&kind, &bound, len)?;
        let sub_len = end - start;
        match receiver {
            Value::FatRef { base, .. } => match &**base {
                Value::Ref(_) => {
                    let base = subslice_ref(base, start, sub_len, name)?;
                    Ok(Value::FatRef {
                        base: Box::new(base),
                        len: sub_len as u128,
                    })
                }
                other => Err(unsupported(format!(
                    "IndexMut::index_mut with a range over a {} base is not runtime-supported (a materialised copy has no storage to alias)",
                    other.kind_name()
                ))),
            },
            Value::Ref(_) => {
                let base = subslice_ref(receiver, start, sub_len, name)?;
                Ok(Value::FatRef {
                    base: Box::new(base),
                    len: sub_len as u128,
                })
            }
            _ => unreachable!("receiver shapes checked above"),
        }
    }

    /// `AsRef::as_ref` / `AsMut::as_mut`: the receiver viewed as
    /// the target type, computed from the modelled data (see the
    /// `Builtin::AsRef` note). `generic_args` are the trait's
    /// `[Self, Target]` type arguments. The blanket impls for
    /// `&T` / `&mut T` forward to `T`'s impl, so leading
    /// references are stripped from the self type and the value
    /// resolution below follows them. For the mutable form the
    /// receiver place itself is the viewed storage for the
    /// modelled containers, so the argument reference is returned
    /// unchanged; only `Box` needs its allocation pointer.
    fn as_ref_view(
        &self,
        frames: &[Frame],
        generic_args: &[String],
        arg: &Value,
        mutable: bool,
    ) -> InterpResult<Value> {
        let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
        let target_ty = generic_args.get(1).map(String::as_str).unwrap_or("");
        let mut base_ty = self_ty;
        while let Some(rest) = base_ty
            .strip_prefix("&mut ")
            .or_else(|| base_ty.strip_prefix('&'))
        {
            base_ty = rest.trim();
        }
        let what = if mutable { "AsMut::as_mut" } else { "AsRef::as_ref" };
        if mutable {
            match base_ty {
                t if is_string_like_ty(t)
                    || t.starts_with("std::vec::Vec<")
                    || t.starts_with("alloc::vec::Vec<")
                    || t.starts_with('[') =>
                {
                    return Ok(arg.clone());
                }
                t if t.starts_with("std::boxed::Box<") => {
                    let pointee = self.resolve_value(frames, arg)?;
                    if let Value::DynRef { .. } = pointee {
                        return Ok(pointee);
                    }
                    return match pointee {
                        Value::Enum { fields, .. } if fields.len() == 1 => {
                            let inner =
                                fields.into_iter().next().expect("len checked");
                            Ok(match inner {
                                Value::Ref(RefTarget::Alloc { .. }) => inner,
                                other => Value::Ref(RefTarget::value(other)),
                            })
                        }
                        other => Err(InterpError::type_mismatch(
                            what,
                            "boxed value",
                            other.kind_name(),
                        )),
                    };
                }
                _ => {
                    return Err(unsupported(format!(
                        "{what} for {self_ty} is not runtime-supported"
                    )));
                }
            }
        }
        let value = self.resolve_value(frames, arg)?;
        let view = self.view_value_as(frames, value, base_ty, target_ty, what)?;
        // `as_ref` returns a reference: a view that came out as the
        // bare pointee data (an identity view of an aggregate —
        // `Box<T> -> &T`, `Vec<T> -> &Vec<T>`, a self view) must be
        // re-wrapped as a reference to that data, or storing the
        // result fails the destination's `Ref` type check. The fat
        // and string views are already reference-shaped.
        Ok(match view {
            Value::Ref(_)
            | Value::FatRef { .. }
            | Value::DynRef { .. }
            | Value::Str(_)
            | Value::Bytes(_)
            | Value::FnPtr(_)
            | Value::Closure { .. } => view,
            other => Value::Ref(RefTarget::value(other)),
        })
    }

    /// The value-level view behind `as_ref_view`: `value` is the
    /// receiver's data (references already followed), `base_ty`
    /// its type, `target_ty` the view's type.
    fn view_value_as(
        &self,
        frames: &[Frame],
        value: Value,
        base_ty: &str,
        target_ty: &str,
        what: &str,
    ) -> InterpResult<Value> {
        // A value still wrapped in a reference (a blanket
        // `&T` layer the type stripping already accounted for):
        // follow it and view the pointee.
        if matches!(
            value,
            Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. }
        ) {
            let inner = self.resolve_value(frames, &value)?;
            if inner != value {
                return self.view_value_as(frames, inner, base_ty, target_ty, what);
            }
        }
        if is_string_like_ty(base_ty) {
            if target_ty == "[u8]" {
                return self.string_bytes_ref(frames, &value);
            }
            if target_ty == "str"
                || target_ty == "std::ffi::OsStr"
                || target_ty == "std::path::Path"
                || target_ty == base_ty
            {
                return match value {
                    Value::Str(_) | Value::Bytes(_) => Ok(value),
                    Value::Enum { .. } => {
                        // An owned string-like value in aggregate
                        // form (a `PathBuf` around an `OsString`
                        // around the buffer around the byte
                        // vector, the shape `pathbuf_value`
                        // builds): the view is the byte string at
                        // the bottom of the wrapper chain.
                        match string_aggregate_bytes(&value) {
                            Some(bytes) => Ok(Value::Bytes(bytes)),
                            None => Err(InterpError::type_mismatch(
                                what,
                                "string data",
                                value.kind_name(),
                            )),
                        }
                    }
                    other => Err(InterpError::type_mismatch(
                        what,
                        "string data",
                        other.kind_name(),
                    )),
                };
            }
        }
        if base_ty.starts_with("std::vec::Vec<")
            || base_ty.starts_with("alloc::vec::Vec<")
        {
            if target_ty.starts_with('[') {
                let len = value.len().ok_or_else(|| {
                    InterpError::type_mismatch(
                        what,
                        "array value",
                        value.kind_name(),
                    )
                })?;
                return Ok(Value::FatRef {
                    base: Box::new(value),
                    len,
                });
            }
            if target_ty.starts_with("std::vec::Vec<")
                || target_ty.starts_with("alloc::vec::Vec<")
            {
                return Ok(value);
            }
        }
        if base_ty.starts_with('[') && target_ty.starts_with('[') {
            if base_ty.contains(';') && !target_ty.contains(';') {
                // An array viewed as a slice.
                let len = value.len().ok_or_else(|| {
                    InterpError::type_mismatch(
                        what,
                        "array value",
                        value.kind_name(),
                    )
                })?;
                return Ok(Value::FatRef {
                    base: Box::new(value),
                    len,
                });
            }
            return Ok(value);
        }
        if base_ty.starts_with("std::boxed::Box<") {
            if let Value::DynRef { .. } = value {
                return Ok(value);
            }
            let inner_ty = first_generic_param(base_ty).ok_or_else(|| {
                unsupported(format!("{what} for {base_ty}: no inner type"))
            })?;
            return match value {
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    let field = fields.into_iter().next().expect("len checked");
                    if target_ty == inner_ty {
                        // The view is the pointee itself: the
                        // result is a reference to the box's
                        // storage, not the pointee value. The box
                        // field already is that reference (the
                        // allocation pointer); a non-reference
                        // field is resolved and re-wrapped as a
                        // value reference.
                        return Ok(match field {
                            Value::Ref(_) | Value::DynRef { .. } => field,
                            other => {
                                let pointee = self.resolve_value(frames, &other)?;
                                Value::Ref(RefTarget::value(pointee))
                            }
                        });
                    }
                    let pointee = self.resolve_value(frames, &field)?;
                    self.view_value_as(frames, pointee, &inner_ty, target_ty, what)
                }
                other => Err(InterpError::type_mismatch(
                    what,
                    "boxed value",
                    other.kind_name(),
                )),
            };
        }
        if base_ty.starts_with("std::rc::Rc<")
            || base_ty.starts_with("std::sync::Arc<")
        {
            if let Value::DynRef { .. } = value {
                return Ok(value);
            }
            let inner_ty = first_generic_param(base_ty).ok_or_else(|| {
                unsupported(format!("{what} for {base_ty}: no inner type"))
            })?;
            return match rc_handle_id(&value) {
                Some(id) => {
                    let pointee = self.read_resolved(
                        frames,
                        &Resolved::Alloc {
                            id,
                            projection: vec![RefProj::Field(2)],
                        },
                    )?;
                    self.view_value_as(frames, pointee, &inner_ty, target_ty, what)
                }
                None => Err(InterpError::type_mismatch(
                    what,
                    "a reference-counted handle",
                    value.kind_name(),
                )),
            };
        }
        if base_ty.starts_with("std::borrow::Cow<") {
            let inner_ty = last_generic_param(base_ty).ok_or_else(|| {
                unsupported(format!("{what} for {base_ty}: no inner type"))
            })?;
            return match value {
                Value::Enum { fields, .. } if !fields.is_empty() => {
                    let data = fields.into_iter().next().expect("checked non-empty");
                    self.view_value_as(frames, data, &inner_ty, target_ty, what)
                }
                other => Err(InterpError::type_mismatch(
                    what,
                    "a Cow value",
                    other.kind_name(),
                )),
            };
        }
        if base_ty == target_ty {
            return Ok(value);
        }
        Err(unsupported(format!(
            "{what} for {base_ty} -> {target_ty} is not runtime-supported"
        )))
    }

    /// The reference `step` inside the place a reference value
    /// denotes, when the reference denotes a live place: a frame
    /// place, a static's storage or an allocation, with `step`
    /// appended to its projection. Returns `None` when the
    /// reference denotes no place (a value snapshot), where the
    /// caller falls back to resolving against the value itself.
    ///
    /// Builtins that hand out references to a part of a value they
    /// were given a reference to (the pointee of an
    /// aggregate-shaped box) use this to keep the result denoting
    /// the same storage the part lives in, so writes through the
    /// result \u2014 interior mutation, `SetDiscriminant`, a
    /// `Cell::set` write-back \u2014 land in that storage instead of
    /// failing on an immutable snapshot of it.
    fn place_projected_ref(
        &self,
        frames: &[Frame],
        arg: &Value,
        step: RefProj,
    ) -> InterpResult<Option<Value>> {
        Ok(match self.deref_value(frames, arg)? {
            Resolved::Place {
                frame_pos,
                local,
                mut projection,
            } => {
                projection.push(step);
                Some(Value::Ref(RefTarget::Place {
                    frame: frames[frame_pos].id,
                    local,
                    projection,
                }))
            }
            Resolved::Static {
                name,
                mut projection,
            } => {
                projection.push(step);
                Some(Value::Ref(RefTarget::Static { name, projection }))
            }
            Resolved::Alloc {
                id,
                mut projection,
            } => {
                projection.push(step);
                Some(Value::Ref(RefTarget::Alloc { id, projection }))
            }
            Resolved::Value(..) => None,
        })
    }

    /// A reference to field 0 of the aggregate `arg` points at:
    /// the dereference result of a one-field newtype wrapper
    /// (`AssertUnwindSafe`). A place-backed argument yields a live
    /// place reference, so writes through the result land in the
    /// wrapper; a detached aggregate value yields a reference to a
    /// copy of the field.
    pub(crate) fn field_zero_ref(
        &self,
        frames: &[Frame],
        arg: &Value,
        what: &str,
    ) -> InterpResult<Value> {
        Ok(match self.deref_value(frames, arg)? {
            Resolved::Place {
                frame_pos,
                local,
                mut projection,
            } => {
                projection.push(RefProj::Field(0));
                Value::Ref(RefTarget::Place {
                    frame: frames[frame_pos].id,
                    local,
                    projection,
                })
            }
            Resolved::Static {
                name,
                mut projection,
            } => {
                projection.push(RefProj::Field(0));
                Value::Ref(RefTarget::Static { name, projection })
            }
            Resolved::Alloc { id, mut projection } => {
                projection.push(RefProj::Field(0));
                Value::Ref(RefTarget::Alloc { id, projection })
            }
            Resolved::Value(base, projection) => {
                let value = self.value_target_value(&base, &projection)?;
                Value::Ref(RefTarget::value(one_field_value(value, what)?))
            }
        })
    }

    /// Read a reference value exactly one level: the value it
    /// names, without following that value further if it is itself
    /// a reference. The `Deref` / `DerefMut` impls for references
    /// have exactly this shape: `deref(&&T) -> &T` reads the
    /// argument place once and returns the inner reference itself.
    /// (`resolve_value` follows the chain to the final pointee,
    /// which hands the caller the pointee where the inner
    /// reference was asked for.)
    pub(crate) fn read_ref_once(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Value> {
        match value {
            // The null and address-only pointers denote no data;
            // one more level of reading is the value itself.
            Value::Ref(RefTarget::Null) | Value::Ref(RefTarget::Address(_)) => {
                Ok(value.clone())
            }
            Value::Ref(_) => {
                let target = self.deref_value(frames, value)?;
                self.read_resolved(frames, &target)
            }
            _ => Ok(value.clone()),
        }
    }

    /// Follow references in `value` until a non-reference value.
    pub(crate) fn resolve_value(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Value> {
        Ok(self.resolve_ref(frames, value)?.into_owned())
    }

    /// Follow references in `value` until a non-reference value,
    /// borrowing the denoted data wherever it is already
    /// materialised: the borrowing form of [`resolve_value`], with
    /// the same results and errors. The shapes that denote no data
    /// (null and address-only pointers) and a fat pointer over a
    /// self-contained element tuple (a slice constant, a
    /// materialised sub-slice — dereferencing it would yield
    /// itself, so following it would spin) resolve to themselves.
    pub(crate) fn resolve_ref<'x>(
        &self,
        frames: &'x [Frame],
        value: &'x Value,
    ) -> InterpResult<Cow<'x, Value>> {
        let mut cur: Cow<'x, Value> = Cow::Borrowed(value);
        loop {
            let follows = match &*cur {
                Value::Ref(RefTarget::Null) | Value::Ref(RefTarget::Address(_)) => false,
                Value::FatRef { base, .. } if matches!(&**base, Value::Tuple(_)) => false,
                Value::Ref(_) | Value::FatRef { .. } => true,
                _ => false,
            };
            if !follows {
                return Ok(cur);
            }
            let target = self.deref_value(frames, &cur)?;
            cur = self.read_resolved_ref(frames, &target)?;
        }
    }

    /// The bytes behind a raw C string pointer, up to (not
    /// including) the first NUL byte: the scan the C library
    /// `strlen` performs. String and byte data are scanned
    /// directly; any other aggregate is read element by
    /// element, as [`Self::read_byte_range`] reads a fixed
    /// range. Storage with no NUL in it is an explicit error,
    /// never a silently truncated length.
    fn read_c_str(&self, frames: &[Frame], ptr: &Value) -> InterpResult<Vec<u8>> {
        let target = self.deref_value(frames, ptr)?;
        let base = self.read_resolved(frames, &target)?;
        let direct: Option<&[u8]> = match &base {
            Value::Str(s) => Some(s.as_bytes()),
            Value::Bytes(b) => Some(b.as_slice()),
            _ => None,
        };
        if let Some(bytes) = direct {
            return match bytes.iter().position(|b| *b == 0) {
                Some(end) => Ok(bytes[..end].to_vec()),
                None => Err(unsupported(
                    "strlen of a string whose storage has no NUL terminator",
                )),
            };
        }
        let mut out = Vec::new();
        loop {
            match self.read_seq_at(frames, &target, out.len())? {
                Value::Uint(0) => return Ok(out),
                Value::Uint(b) => out.push(b as u8),
                Value::Int(b) if (0..=255).contains(&b) => {
                    if b == 0 {
                        return Ok(out);
                    }
                    out.push(b as u8);
                }
                other => {
                    return Err(InterpError::type_mismatch(
                        "strlen element",
                        "byte",
                        other.kind_name(),
                    ));
                }
            }
        }
    }

    /// The `&[u8]` value denoting a string value's bytes: a fat
    /// reference over the receiver itself when the receiver is a
    /// reference (element reads resolve through it to the string
    /// data, as for a `Vec` deref), over the receiver's own base
    /// when it is already a fat reference, or over a snapshot of
    /// the bytes when the receiver is the string data itself (the
    /// shape of a byte-slice constant).
    /// The `n` bytes starting at a raw byte pointer, for
    /// `compare_bytes`: string and byte data read directly, any
    /// other aggregate element by element.
    fn read_byte_range(
        &self,
        frames: &[Frame],
        ptr: &Value,
        n: usize,
    ) -> InterpResult<Vec<u8>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let target = self.deref_value(frames, ptr)?;
        let base = self.read_resolved(frames, &target)?;
        let direct: Option<&[u8]> = match &base {
            Value::Str(s) => Some(s.as_bytes()),
            Value::Bytes(b) => Some(b.as_slice()),
            _ => None,
        };
        if let Some(bytes) = direct {
            if bytes.len() < n {
                return Err(InterpError::type_mismatch(
                    "compare_bytes range",
                    "byte range of the requested length",
                    format!("{} bytes", bytes.len()),
                ));
            }
            return Ok(bytes[..n].to_vec());
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            match self.read_seq_at(frames, &target, i)? {
                Value::Uint(b) => out.push(b as u8),
                Value::Int(b) if (0..=255).contains(&b) => out.push(b as u8),
                other => {
                    return Err(InterpError::type_mismatch(
                        "compare_bytes element",
                        "byte",
                        other.kind_name(),
                    ));
                }
            }
        }
        Ok(out)
    }

    fn string_bytes_ref(&self, frames: &[Frame], receiver: &Value) -> InterpResult<Value> {
        let s = self.resolve_string(frames, receiver)?;
        let len = s.len() as u128;
        let base = match receiver {
            Value::Ref(_) => receiver.clone(),
            Value::FatRef { base, .. } => (**base).clone(),
            _ => Value::Tuple(
                s.as_bytes()
                    .iter()
                    .map(|b| Value::Uint(*b as u128))
                    .collect(),
            ),
        };
        Ok(Value::FatRef {
            base: Box::new(base),
            len,
        })
    }

    /// The string data a value denotes: the value itself, or the
    /// pointee of a reference to string data.
    /// The bytes an `&OsStr` argument denotes (a process
    /// program, argument, environment entry or directory): in
    /// this model an `OsStr` is its byte string.
    fn os_str_bytes(&self, frames: &[Frame], value: &Value) -> InterpResult<Vec<u8>> {
        let resolved0 = self.resolve_value(frames, value)?;
        // An owned byte-string aggregate in buffer form (an
        // `OsString` or `PathBuf` wrapping a `Buf` wrapping the
        // byte vector): the unwrap loop below cannot descend
        // into the byte list itself, so read the bytes off the
        // aggregate's shape first. Where both readings apply
        // they agree (the string form's data is the same bytes).
        if let Some(bytes) =
            os_string_bytes(&resolved0).or_else(|| path_buf_bytes(&resolved0))
        {
            return Ok(bytes);
        }
        let mut resolved = resolved0;
        // The path and OS-string values of this model wrap the
        // string data in one-field aggregates (`PathBuf`
        // around `OsString` around the string itself, an owned
        // `Cow<OsStr>` around its `OsString`): unwrap those
        // layers to the data they hold.
        loop {
            match resolved {
                Value::Str(s) => return Ok(s.into_bytes()),
                Value::Bytes(b) => return Ok(b),
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    resolved = fields.into_iter().next().expect("length checked");
                }
                other => {
                    return Err(InterpError::type_mismatch(
                        "OsStr argument",
                        "string or byte data",
                        other.kind_name(),
                    ))
                }
            }
        }
    }

    /// The command id a process-command receiver denotes: the
    /// receiver is a reference to the handle value the
    /// `Command::new` builtin returned (possibly reached through
    /// the public `Command`'s `inner` field projection).
    fn command_id_of(
        &self,
        frames: &[Frame],
        value: &Value,
        name: &str,
    ) -> InterpResult<u64> {
        let resolved = self.resolve_value(frames, value)?;
        command_handle_id(&resolved).ok_or_else(|| {
            InterpError::type_mismatch(
                format!("builtin {name}"),
                "a process command handle",
                resolved.kind_name(),
            )
        })
    }

    /// The byte string a path argument denotes, and whether it
    /// arrived as text: a `Path` / `OsStr` value in this model
    /// is the byte string itself under its wrapper chain (see
    /// the field projection pun in `project_value`), held as a
    /// `Str` for text data or a `Bytes` for arbitrary bytes, or
    /// as a fat pointer over the bytes.
    /// Assemble a `VecDeque` value from a `From` / `Into`
    /// source: a `Vec` (modelled, or in its real aggregate
    /// form), an array, or a slice. The deque's value is the
    /// real aggregate the converted std bodies build and
    /// consume (see `vec_deque_value`): a source `Vec` in its
    /// real aggregate form hands its buffer over unchanged, as
    /// the std conversion does; every other source's elements
    /// are copied into a fresh element block at exact-fit
    /// capacity, with the head at index zero.
    fn vec_deque_from_value(
        &self,
        frames: &[Frame],
        value: Value,
    ) -> InterpResult<Value> {
        let resolved = self.resolve_value(frames, &value)?;
        if let Some((buf, len)) = vec_real_aggregate_parts(&resolved) {
            return Ok(vec_deque_value(buf, len));
        }
        let elements = match resolved {
            Value::Enum { fields, .. } => fields,
            Value::Tuple(items) => items,
            Value::FatRef { base, .. } => match *base {
                Value::Tuple(items) => items,
                other => {
                    return Err(InterpError::type_mismatch(
                        "From::from to VecDeque",
                        "slice value over an element tuple",
                        other.kind_name(),
                    ));
                }
            },
            other => {
                return Err(InterpError::type_mismatch(
                    "From::from to VecDeque",
                    "vector, array or slice value",
                    other.kind_name(),
                ));
            }
        };
        let len = elements.len();
        let id = self.alloc_insert(Value::Tuple(elements));
        self.mark_elem_block(id);
        let buf = raw_vec_value(
            Value::Ref(RefTarget::Alloc {
                id,
                projection: Vec::new(),
            }),
            len,
        );
        Ok(vec_deque_value(buf, len))
    }

    fn path_bytes_of(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<(Vec<u8>, bool)> {
        match value {
            Value::Str(text) => Ok((text.clone().into_bytes(), true)),
            Value::Bytes(b) => Ok((b.clone(), false)),
            Value::FatRef { .. } => {
                let (items, _) = self.slice_elements(frames, value)?;
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        Value::Uint(b) => out.push(b as u8),
                        other => {
                            return Err(InterpError::type_mismatch(
                                "path argument",
                                "byte elements",
                                other.kind_name(),
                            ))
                        }
                    }
                }
                Ok((out, false))
            }
            Value::Ref(_) => {
                let target = self.deref_value(frames, value)?;
                let stored = self.read_resolved(frames, &target)?;
                self.path_bytes_of(frames, &stored)
            }
            other => Err(InterpError::type_mismatch(
                "path argument",
                "path byte string",
                other.kind_name(),
            )),
        }
    }

    fn resolve_string(&self, frames: &[Frame], value: &Value) -> InterpResult<String> {
        match self.resolve_value(frames, value)? {
            Value::Str(s) => Ok(s),
            other => Err(InterpError::type_mismatch(
                "string argument",
                "string data",
                other.kind_name(),
            )),
        }
    }

    /// The byte data a value denotes (an fmt pieces template).
    /// The `fmt::Arguments` a builtin argument denotes. The
    /// modelled value comes from the `Arguments` constructor
    /// builtins; a body that builds the struct itself (an
    /// inlined `format_args!` expansion inside a library body —
    /// the shape edition-2015 `unreachable!` reaches through
    /// `core::panicking::unreachable_display`) hands over the
    /// raw aggregate instead: two wrapper fields holding the
    /// encoded template bytes and a reference to the argument
    /// array. Both forms denote the same value.
    fn coerce_arguments(
        &self,
        frames: &[Frame],
        value: &Value,
        name: &str,
    ) -> InterpResult<FmtArguments> {
        let resolved = self.resolve_value(frames, value)?;
        match resolved {
            Value::FmtArguments(args) => Ok(args),
            Value::Enum { fields, .. } if fields.len() == 2 => {
                let mut fields = fields.into_iter();
                let template_field = fields.next().expect("two fields");
                let args_field = fields.next().expect("two fields");
                let template =
                    self.resolve_bytes(frames, &unwrap_newtype(template_field))?;
                let args =
                    self.resolve_arg_array(frames, &unwrap_newtype(args_field))?;
                Ok(FmtArguments::Template { template, args })
            }
            other => Err(InterpError::type_mismatch(
                format!("builtin {name}"),
                "fmt::Arguments value",
                other.kind_name(),
            )),
        }
    }

    fn resolve_bytes(&self, frames: &[Frame], value: &Value) -> InterpResult<Vec<u8>> {
        match self.resolve_value(frames, value)? {
            Value::Bytes(b) => Ok(b),
            other => Err(InterpError::type_mismatch(
                "fmt pieces template",
                "byte data",
                other.kind_name(),
            )),
        }
    }

    /// Snapshot the elements of an fmt argument array.
    fn resolve_arg_array(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Vec<Value>> {
        match self.resolve_value(frames, value)? {
            Value::Tuple(items) => Ok(items),
            other => Err(InterpError::type_mismatch(
                "fmt argument array",
                "array of fmt arguments",
                other.kind_name(),
            )),
        }
    }
}

/// The function name a function-pointer argument denotes (the
/// `catch_unwind` intrinsic's try/catch functions are fn items).
fn expect_fn_ptr(value: &Value, name: &str) -> InterpResult<String> {
    match value {
        Value::FnPtr(func) => Ok(func.clone()),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "a function pointer",
            other.kind_name(),
        )),
    }
}

/// The first generic argument of a builtin call (the `T` of a
/// `dyn Any` downcast), rendered as a type tag.
fn expect_generic_ty(generic_args: &[String], name: &str) -> InterpResult<String> {
    expect_generic_ty_at(generic_args, name, 0)
}

/// The generic argument at `index` of a builtin call, rendered
/// as a type tag.
fn expect_generic_ty_at(
    generic_args: &[String],
    name: &str,
    index: usize,
) -> InterpResult<String> {
    generic_args.get(index).cloned().ok_or_else(|| {
        InterpError::type_mismatch(
            format!("builtin {name}"),
            "a generic type argument",
            "no generic arguments",
        )
    })
}

/// Resolve a `Box<dyn Any ...>` value (or a reference to one) to
/// the payload it carries: the concrete type tag, the payload
/// value, and the box value itself (handed back to the caller on
/// a failed by-value downcast).
fn resolve_dyn_box(
    interp: &mut Interpreter,
    frames: &mut Vec<Frame>,
    value: &Value,
    name: &str,
) -> InterpResult<(String, Value, Value)> {
    // Resolve references down to the box value itself (that value
    // is what a failed by-value downcast hands back).
    let mut current = value.clone();
    while matches!(current, Value::Ref(_) | Value::FatRef { .. }) {
        let target = interp.deref_value(frames, &current)?;
        current = interp.read_resolved(frames, &target)?;
    }
    let original = current.clone();
    // Then unwrap the box-shaped wrappers (the one-field `Box` /
    // `Unique` / `NonNull` aggregate chain) down to the `DynRef`,
    // and read the payload out of it.
    loop {
        match current {
            Value::DynRef {
                base, concrete, ..
            } => return Ok((concrete, (*base).clone(), original)),
            Value::Enum { fields, .. } if fields.len() == 1 => {
                let field = fields.into_iter().next().expect("one field");
                // A built box's field is its allocation pointer:
                // unwrapping the box reads the pointee out of
                // the allocation store.
                if matches!(field, Value::Ref(RefTarget::Alloc { .. })) {
                    let target = interp.deref_value(frames, &field)?;
                    current = interp.read_resolved(frames, &target)?;
                } else {
                    current = field;
                }
            }
            Value::Ref(_) | Value::FatRef { .. } => {
                let target = interp.deref_value(frames, &current)?;
                current = interp.read_resolved(frames, &target)?;
            }
            other => {
                return Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    "a `Box<dyn Any>` value",
                    other.kind_name(),
                ));
            }
        }
    }
}

/// The pointee a box-shaped value denotes: the payload a
/// `DynRef` base in a box shape carries. An allocation-backed
/// box (a one-field aggregate chain bottoming out in a pointer)
/// denotes the value the pointer points at; an aggregate-shaped
/// box denotes its field value itself. Any other value denotes
/// itself.
fn dyn_box_pointee(
    interp: &mut Interpreter,
    frames: &mut Vec<Frame>,
    base: &Value,
) -> InterpResult<Value> {
    let mut current = base.clone();
    let mut first_field: Option<Value> = None;
    loop {
        match current {
            Value::Enum { fields, .. } if fields.len() == 1 => {
                let field = fields.into_iter().next().expect("one field");
                if first_field.is_none() {
                    first_field = Some(field.clone());
                }
                current = field;
            }
            Value::Ref(_) | Value::FatRef { .. } => {
                let target = interp.deref_value(frames, &current)?;
                return interp.read_resolved(frames, &target);
            }
            other => return Ok(first_field.unwrap_or(other)),
        }
    }
}

/// The modelled fmt argument a raw `core::fmt::rt::Argument`
/// aggregate denotes, when `value` has that shape: single-field
/// wrappers around the triple of the erased value reference,
/// the formatter function (whose name states the trait and the
/// Self type), and a unit-shaped marker field. Any other shape
/// is not a raw argument.
fn coerce_fmt_arg(value: &Value) -> Option<Value> {
    let mut current = value.clone();
    loop {
        match current {
            Value::Enum { fields, .. } if fields.len() == 1 => {
                current = fields.into_iter().next().expect("one field");
            }
            Value::Enum { fields, .. } => {
                let formatter = fields.iter().find_map(|f| match f {
                    Value::FnPtr(name) => Some(name.clone()),
                    _ => None,
                })?;
                let target = fields.iter().find_map(|f| match f {
                    Value::FnPtr(_) => None,
                    Value::Enum { fields, .. } if fields.is_empty() => None,
                    other => Some(unwrap_newtype(other.clone())),
                })?;
                let trait_ = [
                    ("Display", FmtTrait::Display),
                    ("Debug", FmtTrait::Debug),
                    ("Octal", FmtTrait::Octal),
                    ("LowerHex", FmtTrait::LowerHex),
                    ("UpperHex", FmtTrait::UpperHex),
                    ("Pointer", FmtTrait::Pointer),
                    ("Binary", FmtTrait::Binary),
                    ("LowerExp", FmtTrait::LowerExp),
                    ("UpperExp", FmtTrait::UpperExp),
                ]
                .iter()
                .find(|(name, _)| formatter.contains(&format!("{name}>::fmt")))
                .map(|(_, t)| *t)?;
                let ty = formatter
                    .strip_prefix('<')
                    .and_then(|rest| rest.split(" as ").next())
                    .unwrap_or("")
                    .to_string();
                return Some(Value::FmtArg {
                    target: Box::new(target),
                    trait_,
                    ty,
                });
            }
            _ => return None,
        }
    }
}

/// Unwrap one newtype layer: a single-field aggregate denotes
/// its field (the wrappers core puts around the two halves of a
/// built `fmt::Arguments`).
fn unwrap_newtype(value: Value) -> Value {
    match value {
        Value::Enum { fields, .. } if fields.len() == 1 => {
            fields.into_iter().next().expect("one field")
        }
        other => other,
    }
}

/// Whether a rendered Self type tag denotes the ADT `name`: either
/// the bare name (a type whose generic arguments are all lifetimes
/// renders without them, e.g. `std::str::Chars`) or `name<...>`
/// with type arguments.
/// The abstract `RefCell` value, in the real struct's shape: field 0
/// is the borrow flag as a `Cell` aggregate (`Cell { UnsafeCell {
/// isize } }`, starting at 0 = unused), field 1 is the `UnsafeCell`
/// aggregate holding the value. Building the natural shape keeps
/// collected bodies that project `RefCell` fields (e.g. `into_inner`)
/// working on builtin-constructed cells.
/// A `Cell` value of the natural shape `Cell { UnsafeCell { value
/// } }`: the shape a collected `Cell` aggregate and a decoded
/// constant produce, which the `Cell` builtins and the
/// `UnsafeCell::get` pointer path must agree on.
/// The parts of a `Vec` value in its real aggregate form
/// (`[buf, len]`): the buffer value and the length, when the
/// value has exactly that shape. A modelled vector (its elements
/// as the fields) does not qualify: its second field is an
/// element, never a bare length beside a `RawVec`-shaped first
/// field.
fn vec_real_aggregate_parts(value: &Value) -> Option<(Value, usize)> {
    let Value::Enum { fields, .. } = value else {
        return None;
    };
    if fields.len() != 2 {
        return None;
    }
    let Value::Uint(len) = fields[1] else {
        return None;
    };
    let Value::Enum {
        fields: buf_fields, ..
    } = &fields[0]
    else {
        return None;
    };
    if buf_fields.len() != 2 {
        return None;
    }
    let Value::Enum {
        fields: inner_fields, ..
    } = &buf_fields[0]
    else {
        return None;
    };
    if inner_fields.len() != 3 {
        return None;
    }
    Some((fields[0].clone(), len as usize))
}

/// A `RawVec` buffer value over the element block pointer
/// `ptr`, with capacity `cap`: the `RawVecInner` triple
/// (`NonNull` pointer, `UsizeNoHighBit` capacity, the global
/// allocator marker) plus the trailing `PhantomData`.
fn raw_vec_value(ptr: Value, cap: usize) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![
            Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Enum {
                        variant: 0,
                        fields: vec![ptr],
                    },
                    Value::Enum {
                        variant: 0,
                        fields: vec![Value::Uint(cap as u128)],
                    },
                    Value::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    },
                ],
            },
            Value::Enum {
                variant: 0,
                fields: Vec::new(),
            },
        ],
    }
}

/// A `VecDeque` value in its real aggregate form over buffer
/// `buf` holding `len` elements from index zero: the wrapped
/// head index, the length, and the buffer.
fn vec_deque_value(buf: Value, len: usize) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![
            Value::Enum {
                variant: 0,
                fields: vec![Value::Uint(0)],
            },
            Value::Uint(len as u128),
            buf,
        ],
    }
}

fn cell_value(value: Value) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Enum {
            variant: 0,
            fields: vec![value],
        }],
    }
}

/// The value a `Cell` aggregate of the natural shape holds;
/// `None` for any other value.
/// The single field of a one-field aggregate value: the shape of
/// the std newtype wrappers (`Pin`, `ManuallyDrop`,
/// `AssertUnwindSafe`).
/// The types whose data the value model stores as a string:
/// the string, byte-string and path types and `str` itself.
fn is_string_like_ty(ty: &str) -> bool {
    matches!(
        ty,
        "str"
            | "std::string::String"
            | "alloc::string::String"
            | "std::ffi::OsString"
            | "std::ffi::OsStr"
            | "std::path::PathBuf"
            | "std::path::Path"
    )
}

/// The generic arguments of a rendered type, split at top-level
/// commas: `std::boxed::Box<T, A>` yields `["T", "A"]`.
fn generic_args_of(ty: &str) -> Vec<String> {
    let Some(open) = ty.find('<') else {
        return Vec::new();
    };
    let mut depth = 0usize;
    let mut close = None;
    for (i, c) in ty[open..].char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(close) = close else {
        return Vec::new();
    };
    let inner = &ty[open + 1..close];
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, c) in inner.char_indices() {
        match c {
            '<' | '[' | '(' => depth += 1,
            '>' | ']' | ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(inner[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(inner[start..].trim().to_string());
    parts.retain(|p| !p.is_empty());
    parts
}

/// The first generic argument of a rendered type, if any.
fn first_generic_param(ty: &str) -> Option<String> {
    generic_args_of(ty).into_iter().next()
}

/// The last generic argument of a rendered type, if any (the
/// borrowed type of a `Cow<'_, B>` is its last parameter).
fn last_generic_param(ty: &str) -> Option<String> {
    generic_args_of(ty).into_iter().last()
}

fn one_field_value(value: Value, what: &str) -> InterpResult<Value> {
    match value {
        Value::Enum { fields, .. } if fields.len() == 1 => {
            Ok(fields.into_iter().next().expect("one field"))
        }
        other => Err(InterpError::type_mismatch(
            what,
            "one-field aggregate",
            other.kind_name(),
        )),
    }
}

/// The value produced by dereferencing a pointer-shaped value: a
/// reference, fat reference or dyn reference is its own
/// dereference; a built `Box` yields the pointer to its allocation
/// (an aggregate-built box a reference to the held pointee value);
/// an `Rc` / `Arc` handle yields a reference to the stored value
/// field. This is the shared tail of the `Box` and `Rc` arms of
/// the `Deref` / `DerefMut` builtins, factored out for `Pin<P>`,
/// which derefs as the pointer it wraps.
fn deref_pointer_value(pointer: Value) -> InterpResult<Value> {
    if let Some(id) = rc_handle_id(&pointer) {
        return Ok(Value::Ref(RefTarget::Alloc {
            id,
            projection: vec![RefProj::Field(2)],
        }));
    }
    match pointer {
        Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. } => Ok(pointer),
        Value::Enum { fields, .. } if fields.len() == 1 => {
            let inner = fields.into_iter().next().expect("one field");
            match inner {
                Value::Ref(RefTarget::Alloc { .. }) => Ok(inner),
                other => Ok(Value::Ref(RefTarget::value(other))),
            }
        }
        other => Err(InterpError::type_mismatch(
            "dereference of a wrapped pointer",
            "reference, box or handle value",
            other.kind_name(),
        )),
    }
}

fn cell_held(state: &Value) -> Option<&Value> {
    match state {
        Value::Enum { fields, .. } if fields.len() == 1 => match &fields[0] {
            Value::Enum { fields, .. } if fields.len() == 1 => Some(&fields[0]),
            _ => None,
        },
        _ => None,
    }
}

fn refcell_value(value: Value) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![
            Value::Enum {
                variant: 0,
                fields: vec![Value::Enum {
                    variant: 0,
                    fields: vec![Value::Int(0)],
                }],
            },
            Value::Enum {
                variant: 0,
                fields: vec![value],
            },
        ],
    }
}

/// A `Ref` / `RefMut` guard value in this model: field 0 is a
/// reference to the cell's value (through the `UnsafeCell` wrapper),
/// field 1 is a reference to the cell's borrow-flag `Cell`, which is
/// what the guard's drop glue (`DropPlan::RefCellRelease`) releases
/// through.
fn refcell_guard(cell: &RefTarget) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![
            Value::Ref(crate::interp::refcell_sub_target(
                cell,
                &[RefProj::Field(1), RefProj::Field(0)],
            )),
            Value::Ref(crate::interp::refcell_sub_target(
                cell,
                &[RefProj::Field(0)],
            )),
        ],
    }
}

/// Whether the modelled `Iterator::next` state transition
/// (`iterator_next_state`) serves the iterator type the tag
/// names: the shapes whose values the builtin iterator
/// machinery itself produces — ranges, the `[items, index]`
/// states of `slice::Iter` / `array::IntoIter` / `Chars` /
/// `Bytes`, `Empty`, and the mpsc channel iterators.
/// Whether this type tag names a hash-collection iterator
/// (`hash_map` / `hash_set` view types, or hashbrown's raw
/// iterators): their `next` walks a raw table's control bytes,
/// which the modelled state machines do not reproduce — the
/// collected impl body is the faithful execution.
pub fn hash_iterator_tag(self_ty: &str) -> bool {
    // `Iterator` is also implemented for `&mut I`, whose
    // blanket body forwards to the inner iterator: the tag
    // may carry the reference.
    let self_ty = self_ty
        .strip_prefix("&mut ")
        .or_else(|| self_ty.strip_prefix('&'))
        .unwrap_or(self_ty);
    self_ty.starts_with("std::collections::hash_map::")
        || self_ty.starts_with("std::collections::hash_set::")
        || self_ty.starts_with("hashbrown::map::")
        || self_ty.starts_with("hashbrown::set::")
        || self_ty.starts_with("hashbrown::raw::RawIter")
        || self_ty.starts_with("hashbrown::raw::RawIntoIter")
        || self_ty.starts_with("hashbrown::raw::RawDrain")
        || self_ty.starts_with("hashbrown::raw::RawExtractIf")
}

/// Whether this `IntoIterator::into_iter` `Self` tag names a
/// hash collection (by value or by reference) or a reference
/// to a hash-collection iterator (the `&mut I` blanket impl):
/// the real body assembles the raw-table iterator state,
/// which the builtin's modelled constructions do not cover.
pub fn hash_collection_tag(self_ty: &str) -> bool {
    let inner = self_ty
        .strip_prefix("&mut ")
        .or_else(|| self_ty.strip_prefix('&'))
        .unwrap_or(self_ty);
    inner.starts_with("std::collections::HashMap<")
        || inner.starts_with("std::collections::HashSet<")
        || hash_iterator_tag(inner)
        || hash_iterator_tag(self_ty)
}

pub fn iterator_next_modelled(self_ty: &str) -> bool {
    self_ty.starts_with("std::ops::Range<")
        || self_ty.starts_with("std::ops::RangeInclusive<")
        || self_ty.starts_with("std::slice::Iter<")
        || self_ty.starts_with("std::array::IntoIter<")
        || self_ty.starts_with("std::vec::IntoIter<")
        || self_ty.starts_with("alloc::vec::IntoIter<")
        || self_ty_is(self_ty, "std::env::Args")
        || self_ty_is(self_ty, "std::env::ArgsOs")
        || self_ty_is(self_ty, "std::fs::ReadDir")
        || self_ty_is(self_ty, "std::env::SplitPaths")
        || self_ty.starts_with("std::slice::Windows<")
        || self_ty.starts_with("std::iter::Empty<")
        || self_ty.starts_with("std::sync::mpsc::Iter<")
        || self_ty.starts_with("std::sync::mpsc::IntoIter<")
        || self_ty_is(self_ty, "std::str::Chars")
        || self_ty_is(self_ty, "core::str::Chars")
        || self_ty_is(self_ty, "std::str::Bytes")
        || self_ty_is(self_ty, "core::str::Bytes")
}

/// Whether the modelled `DoubleEndedIterator::next_back` step
/// transitions this iterator type's state directly: the leaf
/// shapes of `iterator_next_back_state` plus the `IterMut`
/// state the back step serves in `next_back_on_state`. Adapter
/// types are absent on purpose — their collected bodies
/// compose the modelled steps and stay the primary execution
/// (see `consumer_instance_needs_body` in the driver).
pub fn iterator_next_back_modelled(self_ty: &str) -> bool {
    if self_ty.starts_with("std::ops::Range<")
        || self_ty.starts_with("std::ops::RangeInclusive<")
        || self_ty.starts_with("std::slice::Iter<")
        || self_ty.starts_with("std::slice::IterMut<")
        || self_ty.starts_with("std::array::IntoIter<")
        || self_ty.starts_with("std::vec::IntoIter<")
        || self_ty.starts_with("alloc::vec::IntoIter<")
        || self_ty.starts_with("std::option::IntoIter<")
        || self_ty.starts_with("std::option::Iter<")
        || self_ty.starts_with("std::option::IterMut<")
        || self_ty_is(self_ty, "std::env::Args")
        || self_ty_is(self_ty, "std::env::ArgsOs")
        || self_ty.starts_with("std::iter::Empty<")
        || self_ty.starts_with("std::path::Components")
        || self_ty.starts_with("std::iter::Once<")
        || self_ty_is(self_ty, "std::str::Chars")
        || self_ty_is(self_ty, "core::str::Chars")
        || self_ty_is(self_ty, "std::str::Bytes")
        || self_ty_is(self_ty, "core::str::Bytes")
    {
        return true;
    }
    // A `Rev`'s back step is its inner iterator's front step.
    if self_ty.starts_with("std::iter::Rev<") {
        return first_generic_arg(self_ty).is_some_and(iterator_next_modelled);
    }
    // These adapters' back steps compose the inner back step.
    if self_ty.starts_with("std::iter::Enumerate<")
        || self_ty.starts_with("std::iter::Copied<")
        || self_ty.starts_with("std::iter::Map<")
    {
        return first_generic_arg(self_ty).is_some_and(iterator_next_back_modelled);
    }
    false
}

fn self_ty_is(self_ty: &str, name: &str) -> bool {
    self_ty == name
        || self_ty
            .strip_prefix(name)
            .is_some_and(|rest| rest.starts_with('<'))
}

/// The fields of an iterator state value (ADT aggregate or tuple).
/// One step of an owned-sequence iterator state (see
/// [`owned_iter_state`]): returns the new state and the yielded
/// item, or `None` when the sequence is exhausted (the state is
/// then returned unchanged by the caller).
fn step_owned_iter(state: &Value, self_ty: &str) -> InterpResult<(Value, Option<Value>)> {
    let fields = iterator_fields(state, self_ty)?;
    if fields.len() != 2 {
        return Err(InterpError::type_mismatch(
            format!("{self_ty} state"),
            "[items, next index]",
            format!("{} field(s)", fields.len()),
        ));
    }
    let items = match &fields[0] {
        Value::Tuple(items) => items.clone(),
        ref other => {
            return Err(InterpError::type_mismatch(
                format!("{self_ty} items"),
                "tuple of elements",
                other.kind_name(),
            ))
        }
    };
    let idx = match fields[1] {
        Value::Uint(v) => v as usize,
        ref other => {
            return Err(InterpError::type_mismatch(
                format!("{self_ty} index"),
                "unsigned integer",
                other.kind_name(),
            ))
        }
    };
    if idx < items.len() {
        let item = items[idx].clone();
        let new_state = match state {
            Value::Enum { variant, .. } => Value::Enum {
                variant: *variant,
                fields: vec![Value::Tuple(items), Value::Uint((idx + 1) as u128)],
            },
            _ => Value::Tuple(vec![Value::Tuple(items), Value::Uint((idx + 1) as u128)]),
        };
        Ok((new_state, Some(item)))
    } else {
        Ok((state.clone(), None))
    }
}

/// The iteration state of an owned-sequence iterator in this
/// model (`vec::IntoIter`, `array::IntoIter`, the env argument
/// iterators): the element snapshot plus the next index.
fn owned_iter_state(items: Vec<Value>) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Tuple(items), Value::Uint(0)],
    }
}

/// The (items, next index) of an owned-sequence iterator state,
/// with the shape checks [`step_owned_iter`] performs.
fn owned_iter_parts(state: &Value, self_ty: &str) -> InterpResult<(Vec<Value>, usize)> {
    let fields = iterator_fields(state, self_ty)?;
    if fields.len() != 2 {
        return Err(InterpError::type_mismatch(
            format!("{self_ty} state"),
            "[items, next index]",
            format!("{} field(s)", fields.len()),
        ));
    }
    let items = match &fields[0] {
        Value::Tuple(items) => items.clone(),
        ref other => {
            return Err(InterpError::type_mismatch(
                format!("{self_ty} items"),
                "tuple of elements",
                other.kind_name(),
            ))
        }
    };
    let idx = match fields[1] {
        Value::Uint(v) => v as usize,
        ref other => {
            return Err(InterpError::type_mismatch(
                format!("{self_ty} index"),
                "unsigned integer",
                other.kind_name(),
            ))
        }
    };
    Ok((items, idx))
}

/// An iterator state value with `state`'s shape (ADT aggregate
/// or tuple) and new fields: the rebuild `iterator_next`
/// performs on a stepped state.
fn iterator_state_like(state: &Value, fields: Vec<Value>) -> Value {
    match state {
        Value::Enum { variant, .. } => Value::Enum {
            variant: *variant,
            fields,
        },
        _ => Value::Tuple(fields),
    }
}

/// One back step of an owned-sequence iterator state: yields
/// the last remaining item and drops it from the snapshot. The
/// front index is untouched, so front and back steps compose
/// exactly as the real double-ended iterators' do.
fn step_owned_iter_back(
    state: &Value,
    self_ty: &str,
) -> InterpResult<(Value, Option<Value>)> {
    let (items, idx) = owned_iter_parts(state, self_ty)?;
    if items.len() > idx {
        let item = items[items.len() - 1].clone();
        let kept = items[..items.len() - 1].to_vec();
        Ok((
            iterator_state_like(state, vec![Value::Tuple(kept), Value::Uint(idx as u128)]),
            Some(item),
        ))
    } else {
        Ok((state.clone(), None))
    }
}

/// One back step (`DoubleEndedIterator::next_back`) of an
/// iterator state, computed on the state value itself: returns
/// the new state and the yielded item. This is what `Rev<I>`'s
/// `next` runs. The shapes served are the ones `iterator_next`
/// serves whose back step is a state transition of the same
/// kind; anything else stays an explicit unsupported error.
fn iterator_next_back_state(
    state: &Value,
    self_ty: &str,
) -> InterpResult<(Value, Option<Value>)> {
    if self_ty.starts_with("std::path::Components") {
        return components_step_state(state, true);
    }
    if self_ty_is(self_ty, "std::env::Args") || self_ty_is(self_ty, "std::env::ArgsOs") {
        // Step the innermost owned-iterator state from the back
        // and rebuild the wrappers, mirroring the front step in
        // `iterator_next`.
        let depth = if self_ty_is(self_ty, "std::env::Args") { 3 } else { 2 };
        let mut inner = state;
        for _ in 0..depth {
            inner = match inner {
                Value::Enum { fields, .. } if fields.len() == 1 => &fields[0],
                other => {
                    return Err(InterpError::type_mismatch(
                        format!("{self_ty} state"),
                        "nested one-field iterator wrappers",
                        other.kind_name(),
                    ))
                }
            };
        }
        let (new_inner, item) = step_owned_iter_back(inner, self_ty)?;
        let mut rebuilt = new_inner;
        for _ in 0..depth {
            rebuilt = Value::Enum {
                variant: 0,
                fields: vec![rebuilt],
            };
        }
        return Ok((rebuilt, item));
    }
    let fields = iterator_fields(state, self_ty)?;
    if self_ty.starts_with("std::ops::RangeInclusive<") {
        if fields.len() != 3 {
            return Err(InterpError::type_mismatch(
                "RangeInclusive state",
                "[start, end, exhausted]",
                format!("{} field(s)", fields.len()),
            ));
        }
        let (start, end, exhausted) = (
            fields[0].clone(),
            fields[1].clone(),
            fields[2].clone(),
        );
        if exhausted == Value::Bool(true) {
            return Ok((state.clone(), None));
        }
        return Ok(match compare_values(&start, &end)? {
            Ordering::Less => (
                iterator_state_like(
                    state,
                    vec![start, decrement(&end)?, Value::Bool(false)],
                ),
                Some(end),
            ),
            Ordering::Equal => (
                iterator_state_like(state, vec![start, end.clone(), Value::Bool(true)]),
                Some(end),
            ),
            Ordering::Greater => (state.clone(), None),
        });
    }
    if self_ty.starts_with("std::ops::Range<") {
        if fields.len() != 2 {
            return Err(InterpError::type_mismatch(
                "Range state",
                "[start, end]",
                format!("{} field(s)", fields.len()),
            ));
        }
        let (start, end) = (fields[0].clone(), fields[1].clone());
        return Ok(match compare_values(&start, &end)? {
            Ordering::Less => {
                let new_end = decrement(&end)?;
                (
                    iterator_state_like(state, vec![start, new_end.clone()]),
                    Some(new_end),
                )
            }
            _ => (state.clone(), None),
        });
    }
    if self_ty.starts_with("std::slice::Iter<") {
        return step_slice_iter_back(state, self_ty);
    }
    if self_ty.starts_with("std::array::IntoIter<")
        || self_ty.starts_with("std::vec::IntoIter<")
        || self_ty.starts_with("alloc::vec::IntoIter<")
        || self_ty.starts_with("std::option::IntoIter<")
        || self_ty_is(self_ty, "std::str::Chars")
        || self_ty_is(self_ty, "core::str::Chars")
        || self_ty_is(self_ty, "std::str::Bytes")
        || self_ty_is(self_ty, "core::str::Bytes")
    {
        return step_owned_iter_back(state, self_ty);
    }
    if self_ty.starts_with("std::iter::Empty<") {
        return Ok((state.clone(), None));
    }
    Err(unsupported(format!(
        "DoubleEndedIterator::next_back for {self_ty} is not runtime-supported"
    )))
}

/// The `slice::Iter` value over an element snapshot, in the
/// struct's real field layout: `ptr` and `end_or_len` are element
/// pointers into the snapshot — value targets carrying an
/// `Index` projection — and the third field is the
/// `PhantomData` marker. The layout matters beyond the builtins:
/// core's converted bodies for the iterator's other methods
/// (`fold`, `count`, `nth`, `size_hint`, ...) project these very
/// fields, so a fabricated (snapshot, cursor) pair in their place
/// reads back as a bogus pointer pair and the run dies on the
/// first field copy.
fn slice_iter_value(elems: Vec<Value>) -> Value {
    let len = elems.len();
    let snapshot = Value::Tuple(elems);
    let ptr = Value::Ref(RefTarget::Value(
        Box::new(snapshot.clone()),
        vec![RefProj::Index(0)],
    ));
    let end = Value::Ref(RefTarget::Value(
        Box::new(snapshot),
        vec![RefProj::Index(len)],
    ));
    Value::Enum {
        variant: 0,
        fields: vec![
            // `ptr: NonNull<T>` — the one-field wrapper around
            // the raw element pointer.
            Value::Enum {
                variant: 0,
                fields: vec![ptr],
            },
            end,
            // `_marker: PhantomData<&T>` — a zero-sized ADT
            // value.
            Value::Enum {
                variant: 0,
                fields: Vec::new(),
            },
        ],
    }
}

/// The element pointer a slice-iterator `ptr` field holds: the
/// field's declared type is `NonNull<T>`, a one-field wrapper
/// around the raw pointer, so the pointer is the wrapper's only
/// field when the value is wrapped, and the value itself when it
/// is not.
fn iter_ptr_value(field: &Value) -> Option<Value> {
    match field {
        Value::Enum { fields, .. } if fields.len() == 1 => Some(fields[0].clone()),
        Value::Ref(_) => Some(field.clone()),
        _ => None,
    }
}

/// The element index a slice-iterator pointer denotes: the
/// trailing `Index` step of its target's projection, if it has
/// one.
fn iter_ptr_index(value: &Value) -> Option<usize> {
    if let Value::Enum { fields, .. } = value {
        if fields.len() == 1 {
            return iter_ptr_index(&fields[0]);
        }
    }
    let projection = match value {
        Value::Ref(RefTarget::Place { projection, .. }) => projection,
        Value::Ref(RefTarget::Static { projection, .. }) => projection,
        Value::Ref(RefTarget::Alloc { projection, .. }) => projection,
        Value::Ref(RefTarget::Value(_, projection)) => projection,
        _ => return None,
    };
    match projection.last() {
        Some(RefProj::Index(i)) => Some(*i),
        _ => None,
    }
}

/// The same element pointer with its trailing `Index` step set
/// to `idx`.
fn iter_ptr_with_index(value: &Value, idx: usize) -> Option<Value> {
    if let Value::Enum { variant, fields } = value {
        if fields.len() == 1 {
            let inner = iter_ptr_with_index(&fields[0], idx)?;
            return Some(Value::Enum {
                variant: *variant,
                fields: vec![inner],
            });
        }
    }
    fn with_index(projection: &[RefProj], idx: usize) -> Option<Vec<RefProj>> {
        let mut projection = projection.to_vec();
        match projection.last_mut() {
            Some(RefProj::Index(i)) => {
                *i = idx;
                Some(projection)
            }
            _ => None,
        }
    }
    match value {
        Value::Ref(RefTarget::Place {
            frame,
            local,
            projection,
        }) => Some(Value::Ref(RefTarget::place(
            *frame,
            *local,
            with_index(projection, idx)?,
        ))),
        Value::Ref(RefTarget::Static { name, projection }) => {
            Some(Value::Ref(RefTarget::Static {
                name: name.clone(),
                projection: with_index(projection, idx)?,
            }))
        }
        Value::Ref(RefTarget::Alloc { id, projection }) => Some(Value::Ref(RefTarget::Alloc {
            id: *id,
            projection: with_index(projection, idx)?,
        })),
        Value::Ref(RefTarget::Value(v, projection)) => Some(Value::Ref(RefTarget::Value(
            v.clone(),
            with_index(projection, idx)?,
        ))),
        _ => None,
    }
}

/// The (current, end) element indices of a real-layout
/// `slice::Iter` state value, with the shape checks
/// `iterator_next` performs for it.
fn slice_iter_bounds(state: &Value, self_ty: &str) -> InterpResult<(usize, usize)> {
    let fields = iterator_fields(state, self_ty)?;
    if fields.len() != 3 {
        return Err(InterpError::type_mismatch(
            "slice::Iter state",
            "[ptr, end, marker]",
            format!("{} field(s)", fields.len()),
        ));
    }
    let (Some(cur), Some(end)) = (iter_ptr_index(&fields[0]), iter_ptr_index(&fields[1]))
    else {
        return Err(InterpError::type_mismatch(
            "slice::Iter state",
            "element pointers into the element snapshot",
            format!(
                "{} / {}",
                fields[0].kind_name(),
                fields[1].kind_name()
            ),
        ));
    };
    Ok((cur, end))
}

/// One back step of a real-layout `slice::Iter` state: yields
/// the last remaining element reference and pulls the end
/// pointer back by one element. The front pointer is untouched,
/// so front and back steps compose exactly as the real
/// double-ended iterator's do.
fn step_slice_iter_back(
    state: &Value,
    self_ty: &str,
) -> InterpResult<(Value, Option<Value>)> {
    let fields = iterator_fields(state, self_ty)?;
    if fields.len() != 3 {
        return Err(InterpError::type_mismatch(
            "slice::Iter state",
            "[ptr, end, marker]",
            format!("{} field(s)", fields.len()),
        ));
    }
    let (cur, end) = slice_iter_bounds(state, self_ty)?;
    if cur < end {
        let new_end =
            iter_ptr_with_index(&fields[1], end - 1).expect("index step checked");
        let item_field =
            iter_ptr_with_index(&fields[0], end - 1).expect("index step checked");
        let item = iter_ptr_value(&item_field).expect("pointer field checked");
        Ok((
            iterator_state_like(
                state,
                vec![fields[0].clone(), new_end, fields[2].clone()],
            ),
            Some(item),
        ))
    } else {
        Ok((state.clone(), None))
    }
}

fn iterator_fields(state: &Value, self_ty: &str) -> InterpResult<Vec<Value>> {
    match state {
        Value::Enum { fields, .. } => Ok(fields.clone()),
        Value::Tuple(items) => Ok(items.clone()),
        other => Err(InterpError::type_mismatch(
            format!("iterator state of {self_ty}"),
            "ADT value",
            other.kind_name(),
        )),
    }
}

fn rebuild_state(state: &Value, new_fields: Vec<Value>) -> Value {
    match state {
        Value::Enum { variant, .. } => Value::Enum {
            variant: *variant,
            fields: new_fields,
        },
        _ => Value::Tuple(new_fields),
    }
}

fn option_payload_opt(value: Value) -> Option<Value> {
    match value {
        Value::Enum {
            variant: 1,
            mut fields,
        } if fields.len() == 1 => Some(fields.remove(0)),
        _ => None,
    }
}

fn extend_target(target: &Resolved, step: RefProj) -> Option<Resolved> {
    match target {
        Resolved::Place {
            frame_pos,
            local,
            projection,
        } => {
            let mut projection = projection.clone();
            projection.push(step);
            Some(Resolved::Place {
                frame_pos: *frame_pos,
                local: *local,
                projection,
            })
        }
        Resolved::Static { name, projection } => {
            let mut projection = projection.clone();
            projection.push(step);
            Some(Resolved::Static {
                name: name.clone(),
                projection,
            })
        }
        Resolved::Alloc { id, projection } => {
            let mut projection = projection.clone();
            projection.push(step);
            Some(Resolved::Alloc {
                id: *id,
                projection,
            })
        }
        Resolved::Value(..) => None,
    }
}

fn take_option_payload(state: &Value) -> Result<Option<(Value, Value)>, ()> {
    match state {
        Value::Enum {
            variant: 1,
            fields,
        } if fields.len() == 1 => Ok(Some((
            Value::Enum {
                variant: 0,
                fields: Vec::new(),
            },
            fields[0].clone(),
        ))),
        Value::Enum { variant: 0, fields } if fields.is_empty() => Ok(None),
        Value::Enum { variant: 0, fields } if fields.len() == 1 => {
            match take_option_payload(&fields[0])? {
                Some((new_inner, item)) => Ok(Some((
                    Value::Enum {
                        variant: 0,
                        fields: vec![new_inner],
                    },
                    item,
                ))),
                None => Ok(None),
            }
        }
        Value::Tuple(items) if items.len() == 1 => match take_option_payload(&items[0])? {
            Some((new_inner, item)) => {
                Ok(Some((Value::Tuple(vec![new_inner]), item)))
            }
            None => Ok(None),
        },
        _ => Err(()),
    }
}

/// The panic message `core::slice::index::slice_index_fail`
/// selects for `(start, end, len)`, in core's own branch order
/// (the runtime form of its `const_panic!` messages).
fn slice_index_fail_message(start: usize, end: usize, len: usize) -> String {
    if start > len {
        format!("range start index {start} out of range for slice of length {len}")
    } else if end > len {
        format!("range end index {end} out of range for slice of length {len}")
    } else if start > end {
        format!("slice index starts at {start} but ends at {end}")
    } else {
        format!("range end index {end} out of range for slice of length {len}")
    }
}

/// Peel one-field wrapper aggregates off a value: reading through
/// a raw pointer into a transparent wrapper (`UnsafeCell<T>`, the
/// `AtomicUsize`-family structs, any one-field struct) yields the
/// wrapper's aggregate, whose only field is the scalar the pointer
/// denotes. Anything else is returned unchanged.
pub(crate) fn unwrap_wrapper(value: &Value) -> &Value {
    let mut current = value;
    while let Value::Enum {
        variant: 0,
        fields,
    } = current
    {
        if fields.len() != 1 {
            break;
        }
        current = &fields[0];
    }
    current
}

/// Rebuild the wrapper chain `current` carries around a new core
/// value: the mirror of [`unwrap_wrapper`] for writes, so a cell
/// place keeps its aggregate shape with the new scalar inside.
pub(crate) fn rewrap_like(current: &Value, core: Value) -> Value {
    match current {
        Value::Enum {
            variant: 0,
            fields,
        } if fields.len() == 1 => Value::Enum {
            variant: 0,
            fields: vec![rewrap_like(&fields[0], core)],
        },
        _ => core,
    }
}

fn compare_values(a: &Value, b: &Value) -> InterpResult<Ordering> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(x.cmp(y)),
        (Value::Uint(x), Value::Uint(y)) => Ok(x.cmp(y)),
        _ => Err(InterpError::type_mismatch(
            "iterator bounds",
            "matching integer bounds",
            format!("{} and {}", a.kind_name(), b.kind_name()),
        )),
    }
}

fn increment(value: &Value) -> InterpResult<Value> {
    match value {
        Value::Int(v) => Ok(Value::Int(v + 1)),
        Value::Uint(v) => Ok(Value::Uint(v + 1)),
        other => Err(InterpError::type_mismatch(
            "iterator bound increment",
            "integer",
            other.kind_name(),
        )),
    }
}

fn decrement(value: &Value) -> InterpResult<Value> {
    match value {
        Value::Int(v) => Ok(Value::Int(v - 1)),
        Value::Uint(v) => Ok(Value::Uint(v - 1)),
        other => Err(InterpError::type_mismatch(
            "iterator bound decrement",
            "integer",
            other.kind_name(),
        )),
    }
}

/// The element type tag of a `Range<elem>` / `RangeInclusive<elem>`
/// type tag.
fn range_elem_ty(self_ty: &str) -> &str {
    self_ty
        .strip_prefix("std::ops::RangeInclusive<")
        .or_else(|| self_ty.strip_prefix("std::ops::Range<"))
        .and_then(|rest| rest.strip_suffix('>'))
        .unwrap_or("")
}

/// The element count of a range, as `size_hint` reports it:
/// `end - start` (plus one for an inclusive range), zero when
/// the range is empty, with the surrogate gap [0xD800, 0xE000)
/// removed for `char` ranges (std's char range iterators skip
/// the gap, and their `size_hint` accounts for it). `None` when
/// the count exceeds `usize::MAX` — std then reports
/// `(usize::MAX, None)`.
///
/// The bounds are read in the element type's own arithmetic:
/// for an unsigned element type a bound is its bit pattern (a
/// constant with the high bit set, `u64::MAX`, materialises as
/// a signed value in this model — see `uint_arg_bits` — and an
/// unassigned temporary's fields keep that form), for a signed
/// element type its value.
fn range_count(
    start: &Value,
    end: &Value,
    inclusive: bool,
    elem: &str,
) -> InterpResult<Option<u128>> {
    fn bits(value: &Value) -> InterpResult<u128> {
        match value {
            Value::Uint(v) => Ok(*v),
            Value::Int(v) => Ok(*v as u128),
            other => Err(InterpError::type_mismatch(
                "range bounds",
                "integer bounds",
                other.kind_name(),
            )),
        }
    }
    fn signed(value: &Value) -> InterpResult<i128> {
        match value {
            Value::Int(v) => Ok(*v),
            Value::Uint(v) => i128::try_from(*v).map_err(|_| {
                InterpError::type_mismatch(
                    "range bounds",
                    "signed integer bounds",
                    "an out-of-range unsigned integer",
                )
            }),
            other => Err(InterpError::type_mismatch(
                "range bounds",
                "integer bounds",
                other.kind_name(),
            )),
        }
    }
    let span: Option<u128> = match elem {
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "char" => {
            let (a, b) = (bits(start)?, bits(end)?);
            if b < a {
                Some(0)
            } else {
                (b - a).checked_add(u128::from(inclusive))
            }
        }
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => {
            let (a, b) = (signed(start)?, signed(end)?);
            if b < a {
                Some(0)
            } else {
                b.checked_sub(a)
                    .and_then(|d| d.checked_add(i128::from(inclusive)))
                    .map(|v| v as u128)
            }
        }
        _ => {
            return Err(InterpError::type_mismatch(
                "range bounds",
                "an integer element type",
                elem,
            ))
        }
    };
    let Some(mut span) = span else {
        return Ok(None);
    };
    if elem == "char" {
        let (a, b) = (bits(start)?, bits(end)?);
        let hi = if inclusive { b + 1 } else { b };
        let g_lo = a.max(0xD800);
        let g_hi = hi.min(0xE000);
        if g_hi > g_lo {
            span -= g_hi - g_lo;
        }
    }
    if span > u128::from(u64::MAX) {
        Ok(None)
    } else {
        Ok(Some(span))
    }
}


/// A `u64` argument read as its bit pattern: a constant with
/// the high bit set (e.g. `u64::MAX`) materialises as a signed
/// value in this model, and a time constructor's argument is the
/// unsigned value with that pattern.
fn uint_arg_bits(value: &Value, name: &str) -> InterpResult<u64> {
    match value {
        Value::Uint(n) => Ok(*n as u64),
        Value::Int(n) => Ok(*n as u64),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "an unsigned integer",
            other.kind_name(),
        )),
    }
}

/// A `u64`/`u32` argument of a time constructor.
fn uint_arg(value: &Value, name: &str) -> InterpResult<u64> {
    match value {
        Value::Uint(n) => Ok(*n as u64),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "an unsigned integer",
            other.kind_name(),
        )),
    }
}

/// `Ok(value)` in the value model.
/// The initial builder state for a command over `program`.
fn command_state_new(program: Vec<u8>) -> crate::interp::CommandState {
    crate::interp::CommandState {
        saw_nul: program.contains(&0),
        program,
        args: Vec::new(),
        argv0: None,
        envs: Vec::new(),
        env_clear: false,
        cwd: None,
    }
}

/// The command id a handle value carries, at whichever nesting
/// the value arrives with (the bare handle, or the handle inside
/// the public `Command` aggregate).
fn command_handle_id(value: &Value) -> Option<u64> {
    let mut cur = value;
    loop {
        match cur {
            Value::Uint(id) => return Some(*id as u64),
            Value::Enum { fields, .. } if fields.len() == 1 => {
                cur = &fields[0];
            }
            _ => return None,
        }
    }
}

/// The byte string at the bottom of an owned string-like
/// aggregate: one-field wrappers (a `PathBuf` around an
/// `OsString` around its buffer) peel to their field, and the
/// innermost element list (a `Vec<u8>` value) is the byte
/// string itself. `None` for any other shape.
/// One parsed path component, in std's `components` parsing
/// on Unix: a root directory, a leading current-directory
/// marker, a parent reference, or a normal name.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PathComponent<'a> {
    RootDir,
    CurDir,
    ParentDir,
    Normal(&'a [u8]),
}

/// Parse path bytes into the component sequence std's
/// `Path::components` yields (and its `PartialEq` compares):
/// a leading run of separators is one root component,
/// repeated and trailing separators are dropped, an interior
/// `.` component is dropped, and a leading `.` component of
/// a relative path is the current-directory component.
fn path_components(bytes: &[u8]) -> Vec<PathComponent<'_>> {
    let mut comps = Vec::new();
    let mut rest = bytes;
    if rest.first() == Some(&b'/') {
        comps.push(PathComponent::RootDir);
        let run = rest.iter().take_while(|b| **b == b'/').count();
        rest = &rest[run..];
    }
    let has_root = !comps.is_empty();
    let mut first = true;
    for piece in rest.split(|b| *b == b'/') {
        match piece {
            [] => {}
            [b'.'] if first && !has_root => comps.push(PathComponent::CurDir),
            [b'.'] => {}
            [b'.', b'.'] => comps.push(PathComponent::ParentDir),
            name => comps.push(PathComponent::Normal(name)),
        }
        first = false;
    }
    comps
}

/// Whether a comparison call's generic arguments name a
/// path type (`Path` / `PathBuf`) as the `Self` or `Rhs`
/// type. The callee name carries no impl subject in this
/// model, so the path comparisons are recognised by the
/// call's own generic arguments (the type tags, with any
/// reference marker stripped).
fn path_cmp_subject(generic_args: &[String]) -> bool {
    generic_args.iter().take(2).any(|tag| {
        let tag = tag
            .strip_prefix("&mut ")
            .or_else(|| tag.strip_prefix('&'))
            .unwrap_or(tag);
        matches!(tag, "std::path::Path" | "std::path::PathBuf")
    })
}

/// Whether a comparison call compares paths. The primary
/// signal is the call's own generic arguments
/// (`path_cmp_subject`). A call the converter kept in
/// generic form (a comparison inside a generic library
/// body, reached through the run-time name fallback)
/// carries no type tags at all; there the owned operands
/// themselves are the only signal, and an owned `PathBuf`
/// value is recognised by its exact shape — three
/// one-field wrappers (`PathBuf` over `OsString` over its
/// buffer) around the byte-element list. No other modelled
/// string-like type has that shape: an `OsString` has one
/// wrapper fewer, a `String` two fewer, and a `Vec<u8>`
/// none, so their byte-order semantics are untouched.
fn is_pathbuf_value(value: &Value) -> bool {
    let mut cur = value;
    let mut wrappers = 0usize;
    loop {
        match cur {
            Value::Enum { fields, .. }
                if fields.len() == 1
                    && matches!(fields[0], Value::Enum { .. }) =>
            {
                wrappers += 1;
                cur = &fields[0];
            }
            Value::Enum { fields, .. } => {
                return wrappers == 3
                    && fields
                        .iter()
                        .all(|f| matches!(f, Value::Uint(_) | Value::Int(_)));
            }
            _ => return false,
        }
    }
}

/// std's ordering of two paths: `Path::cmp` compares the
/// parsed component sequences lexicographically, and a
/// component orders first by kind — `Component`'s
/// declaration order (root directory, current directory,
/// parent directory, then normal names) — and only then by
/// the name's bytes. This differs from the raw byte order
/// wherever a name contains a byte below '/' (notably '.'):
/// "a/c" < "a.b/c" here, the opposite bytewise.
fn path_ordering(lhs: &[u8], rhs: &[u8]) -> Ordering {
    fn key<'a>(component: &PathComponent<'a>) -> (u8, &'a [u8]) {
        match component {
            PathComponent::RootDir => (1, &[]),
            PathComponent::CurDir => (2, &[]),
            PathComponent::ParentDir => (3, &[]),
            PathComponent::Normal(name) => (4, name),
        }
    }
    path_components(lhs)
        .iter()
        .map(key)
        .cmp(path_components(rhs).iter().map(key))
}

fn string_aggregate_bytes(value: &Value) -> Option<Vec<u8>> {
    match value {
        Value::Str(s) => Some(s.clone().into_bytes()),
        Value::Bytes(b) => Some(b.clone()),
        Value::Enum { fields, .. } => {
            if fields
                .iter()
                .all(|f| matches!(f, Value::Uint(_) | Value::Int(_)))
            {
                // The element list of the byte vector (an empty
                // list is the empty string).
                let mut out = Vec::with_capacity(fields.len());
                for f in fields {
                    match f {
                        Value::Uint(b) => out.push(*b as u8),
                        Value::Int(b) if (0..=255).contains(b) => {
                            out.push(*b as u8)
                        }
                        _ => return None,
                    }
                }
                Some(out)
            } else if fields.len() == 1 {
                string_aggregate_bytes(&fields[0])
            } else {
                None
            }
        }
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
fn io_os_error_value(code: i32) -> Value {
    io_error_value(((code as u32 as u128) << 32) | 0b10)
}

/// An `std::io::Error` holding the bare `ErrorKind` with the
/// given discriminant (the shape of std's `const_error!`).
fn io_simple_error_value(kind_discriminant: u32) -> Value {
    io_error_value(((kind_discriminant as u128) << 32) | 0b11)
}

/// The tagged word an `std::io::Error` value in the bitpacked
/// representation carries: the address-only reference at the
/// bottom of the `Error { repr: Repr(NonNull) }` nesting. The
/// nesting's other leaves are zero-sized markers (the
/// `PhantomData` field a guest-built value carries), so the
/// word is the value tree's unique address leaf; a tree with
/// zero or several address leaves is not this representation.
fn io_error_tagged_word(value: &Value) -> Option<u64> {
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

/// The platform's message text for an OS error code: the text
/// the C library's `strerror` gives for it, which is exactly
/// what std's native `format_os_error` writes. Read through the
/// host's own `io::Error` (whose `Display` is built from that
/// text), with the ` (os error {code})` suffix std's `Display`
/// adds removed.
fn os_error_message(code: i32) -> String {
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
fn error_kind_name(discriminant: u32) -> Option<&'static str> {
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
fn error_kind_message(discriminant: u32) -> Option<&'static str> {
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
fn is_count_cell(value: &Value) -> bool {
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

/// The `ErrorKind::InvalidInput` discriminant.
/// The largest element count `Vec::set_len` will materialise
/// (see the `VecSetLen` builtin): the element list is stored
/// value by value, so the count is bounded well clear of the
/// memory a pathological length would need.
pub(crate) const MAX_VEC_SET_LEN_ELEMENTS: usize = 1 << 22;

/// A vector value holding `elems` at declared length `len`:
/// the plain element aggregate when the whole length is
/// materialised — a growth into the materialisation limit pads
/// with uninitialised slots, as `Vec::set_len` does — and the
/// sparse form (see `Value::SparseVec`) past it, which declares
/// the length the stored elements do not cover.
fn vec_with_len(mut elems: Vec<Value>, len: usize) -> Value {
    if len <= elems.len() {
        elems.truncate(len);
        Value::Enum {
            variant: 0,
            fields: elems,
        }
    } else if len <= MAX_VEC_SET_LEN_ELEMENTS {
        elems.resize(len, Value::Uninit);
        Value::Enum {
            variant: 0,
            fields: elems,
        }
    } else {
        Value::SparseVec {
            elems,
            len: len as u128,
        }
    }
}

const ERROR_KIND_INVALID_INPUT: u32 = 20;
/// The `errno` for a missing file or directory.
const ERRNO_NOT_FOUND: i32 = 2;
/// The `errno` for a permission failure.
const ERRNO_PERMISSION_DENIED: i32 = 13;
/// The `errno` for a path component that is not a directory.
const ERRNO_NOT_A_DIRECTORY: i32 = 20;

/// Whether `candidate` could be executed as a program: it must
/// exist, not be a directory, and carry an execute bit. The
/// error is the `errno` the native spawn reports for it.
fn check_executable(candidate: &std::path::Path) -> Result<(), i32> {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::metadata(candidate) {
        Ok(m) if m.is_dir() => Err(ERRNO_PERMISSION_DENIED),
        Ok(m) if m.permissions().mode() & 0o111 == 0 => Err(ERRNO_PERMISSION_DENIED),
        Ok(_) => Ok(()),
        Err(_) => Err(ERRNO_NOT_FOUND),
    }
}

/// Resolve a command's program the way `execvp` does at spawn:
/// a program containing a path separator is used as-is
/// (relative to the child's working directory when one is set);
/// any other program is searched along the effective `PATH`
/// (the command's own `PATH` entry wins over the inherited one,
/// as in std's environment capture). `Ok` means the program
/// would start; `Err` is the `errno` the failed spawn reports.
fn resolve_program(state: &crate::interp::CommandState) -> Result<(), i32> {
    use std::os::unix::ffi::OsStrExt;
    let program = std::ffi::OsStr::from_bytes(&state.program);
    if state.program.contains(&b'/') {
        let candidate = if state.program.starts_with(b"/") {
            std::path::PathBuf::from(program)
        } else {
            match &state.cwd {
                Some(cwd) => {
                    std::path::Path::new(std::ffi::OsStr::from_bytes(cwd)).join(program)
                }
                None => std::path::PathBuf::from(program),
            }
        };
        return check_executable(&candidate);
    }
    // The effective PATH: the inherited environment with the
    // command's entries applied in order.
    let mut path_var: Option<Vec<u8>> =
        std::env::var_os("PATH").map(|v| v.as_bytes().to_vec());
    if state.env_clear {
        path_var = None;
    }
    for (key, value) in &state.envs {
        if key == b"PATH" {
            path_var = value.clone();
        }
    }
    let search: Vec<u8> = path_var.unwrap_or_else(|| b"/bin:/usr/bin".to_vec());
    let mut denied = false;
    for dir in std::env::split_paths(std::ffi::OsStr::from_bytes(&search)) {
        let candidate = dir.join(program);
        match check_executable(&candidate) {
            Ok(()) => return Ok(()),
            Err(ERRNO_PERMISSION_DENIED) => denied = true,
            Err(_) => {}
        }
    }
    Err(if denied {
        ERRNO_PERMISSION_DENIED
    } else {
        ERRNO_NOT_FOUND
    })
}

/// The `io::Error` a spawn of this command fails with, if it
/// fails before any program starts: a NUL byte in the program
/// or an argument is std's `InvalidInput` const error, a bad
/// working directory fails the spawn's `chdir`, and an
/// unresolvable program fails the `exec`. `None` means the
/// program would start — which the process model does not do,
/// so the caller reports it as a named unsupported error.
fn process_spawn_error(state: &crate::interp::CommandState) -> Option<Value> {
    use std::os::unix::ffi::OsStrExt;
    if state.saw_nul {
        return Some(io_simple_error_value(ERROR_KIND_INVALID_INPUT));
    }
    if let Some(cwd) = &state.cwd {
        let dir = std::path::Path::new(std::ffi::OsStr::from_bytes(cwd));
        match std::fs::metadata(dir) {
            Ok(m) if m.is_dir() => {}
            Ok(_) => return Some(io_os_error_value(ERRNO_NOT_A_DIRECTORY)),
            Err(_) => return Some(io_os_error_value(ERRNO_NOT_FOUND)),
        }
    }
    match resolve_program(state) {
        Ok(()) => None,
        Err(code) => Some(io_os_error_value(code)),
    }
}

/// A `PathBuf` value holding `path`: the path is the string
/// data itself, wrapped in the `OsString` field.
fn path_buf_value(path: &std::path::Path) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Enum {
            variant: 0,
            fields: vec![Value::Str(path.to_string_lossy().into_owned())],
        }],
    }
}

/// The OS error function table the model answers for
/// `get_os_functions`: std's unix entries, by name (all three
/// entries are the builtins above).
fn os_functions_table() -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![
            Value::FnPtr("std::sys::io::error::unix::format_error".to_string()),
            Value::FnPtr("std::sys::io::error::unix::decode_error_kind".to_string()),
            Value::FnPtr("std::sys::io::error::unix::is_interrupted".to_string()),
        ],
    }
}

/// std's unix `decode_error_kind`: the `ErrorKind` discriminant
/// for an `errno` (Linux values). Mirrors the table in
/// `std::sys::io::error::unix` entry for entry; anything not
/// listed is `Uncategorized` (43).
fn decode_error_kind_unix(errno: i32) -> u32 {
    match errno {
        7 => 34,   // E2BIG -> ArgumentListTooLong
        98 => 8,   // EADDRINUSE -> AddrInUse
        99 => 9,   // EADDRNOTAVAIL -> AddrNotAvailable
        16 => 28,  // EBUSY -> ResourceBusy
        103 => 6,  // ECONNABORTED -> ConnectionAborted
        111 => 2,  // ECONNREFUSED -> ConnectionRefused
        104 => 3,  // ECONNRESET -> ConnectionReset
        35 => 30,  // EDEADLK -> Deadlock
        122 => 26, // EDQUOT -> QuotaExceeded
        17 => 12,  // EEXIST -> AlreadyExists
        27 => 27,  // EFBIG -> FileTooLarge
        113 => 4,  // EHOSTUNREACH -> HostUnreachable
        4 => 35,   // EINTR -> Interrupted
        22 => 20,  // EINVAL -> InvalidInput
        21 => 15,  // EISDIR -> IsADirectory
        40 => 18,  // ELOOP -> FilesystemLoop
        2 => 0,    // ENOENT -> NotFound
        12 => 38,  // ENOMEM -> OutOfMemory
        28 => 24,  // ENOSPC -> StorageFull
        31 => 32,  // EMLINK -> TooManyLinks
        36 => 33,  // ENAMETOOLONG -> InvalidFilename
        100 => 10, // ENETDOWN -> NetworkDown
        101 => 5,  // ENETUNREACH -> NetworkUnreachable
        107 => 7,  // ENOTCONN -> NotConnected
        20 => 14,  // ENOTDIR -> NotADirectory
        39 => 16,  // ENOTEMPTY -> DirectoryNotEmpty
        32 => 11,  // EPIPE -> BrokenPipe
        30 => 17,  // EROFS -> ReadOnlyFilesystem
        29 => 25,  // ESPIPE -> NotSeekable
        116 => 19, // ESTALE -> StaleNetworkFileHandle
        110 => 22, // ETIMEDOUT -> TimedOut
        26 => 29,  // ETXTBSY -> ExecutableFileBusy
        18 => 31,  // EXDEV -> CrossesDevices
        115 => 39, // EINPROGRESS -> InProgress
        24 | 23 => 40, // EMFILE | ENFILE -> TooManyOpenFiles
        5 => 41,   // EIO -> InputOutputError
        13 | 1 => 1, // EACCES | EPERM -> PermissionDenied
        38 => 36,  // ENOSYS -> Unsupported
        95 => 36,  // EOPNOTSUPP / ENOTSUP -> Unsupported
        11 => 13,  // EAGAIN / EWOULDBLOCK -> WouldBlock
        _ => 43,   // Uncategorized
    }
}

fn ok_result(value: Value) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![value],
    }
}

/// `Err(value)` in the value model.
fn err_result(value: Value) -> Value {
    Value::Enum {
        variant: 1,
        fields: vec![value],
    }
}

/// The `str::Utf8Error` value for a host validation failure:
/// the struct's `valid_up_to` and `error_len` fields, in the
/// model's aggregate shape.
fn utf8_error_value(error: &std::str::Utf8Error) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![
            Value::Uint(u128::from(error.valid_up_to() as u64)),
            match error.error_len() {
                Some(n) => Value::Enum {
                    variant: 1,
                    fields: vec![Value::Uint(n as u128)],
                },
                None => Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                },
            },
        ],
    }
}

/// A one-field handle value carrying an id: the shape of the
/// thread model's `JoinHandle` / `Thread` / `ThreadId` values and
/// of the mpsc `Sender` / `Receiver` handle values.
fn id_handle(id: u64) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Uint(u128::from(id))],
    }
}

/// A join handle value: `JoinHandle { inner: JoinInner }`, the
/// inner value being the one-field handle of the thread id.
/// Optimizing MIR moves field 0 (the `JoinInner`) out of the
/// handle and calls `JoinInner::join` on it directly, so the
/// nesting is observable: the inner value must have the
/// `JoinInner` type's shape, not the bare id.
fn join_handle_value(id: u64) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![id_handle(id)],
    }
}

/// The thread id a join handle value carries, at whichever
/// nesting the value arrives with: the `JoinHandle` aggregate,
/// the `JoinInner` moved out of it, or the bare id.
fn join_handle_id(value: &Value) -> Option<u64> {
    let mut cur = value;
    loop {
        match cur {
            Value::Uint(id) => return Some(*id as u64),
            Value::Enum { fields, .. } if fields.len() == 1 => {
                cur = &fields[0];
            }
            _ => return None,
        }
    }
}

fn some_value(value: Value) -> Value {
    Value::Enum {
        variant: 1,
        fields: vec![value],
    }
}

fn none_value() -> Value {
    Value::Enum {
        variant: 0,
        fields: Vec::new(),
    }
}

/// Split the `Option` value an `Iterator::next` transition
/// returns into its payload; `None` for the `None` variant.
fn option_payload(value: Value) -> InterpResult<Option<Value>> {
    match value {
        Value::Enum { variant: 0, .. } => Ok(None),
        Value::Enum {
            variant: 1,
            mut fields,
        } if fields.len() == 1 => Ok(Some(fields.pop().expect("len checked"))),
        other => Err(InterpError::type_mismatch(
            "Iterator::next result",
            "an Option value",
            other.kind_name(),
        )),
    }
}

/// A `core::cmp::Ordering` as a value: the variants in declaration
/// order (Less, Equal, Greater).
fn ordering_value(ord: Ordering) -> Value {
    Value::Enum {
        variant: match ord {
            Ordering::Less => 0,
            Ordering::Equal => 1,
            Ordering::Greater => 2,
        },
        fields: Vec::new(),
    }
}

/// The elements a slice-like value denotes, borrowed where the
/// element storage is already materialised: aggregate elements
/// lend their storage, while byte/string elements are the bytes
/// themselves (each byte is one element, exactly what
/// [`value_elements`] materialises as an integer).
pub(crate) enum SliceElems<'a> {
    Values(Cow<'a, [Value]>),
    Bytes(Cow<'a, [u8]>),
    /// The stored prefix of a sparse sequence (see
    /// `Value::SparseVec`) together with its declared length:
    /// indices past the prefix read as the uninitialised slot.
    Sparse { stored: Cow<'a, [Value]>, len: usize },
}

impl<'a> SliceElems<'a> {
    /// The number of elements.
    pub(crate) fn len(&self) -> usize {
        match self {
            SliceElems::Values(v) => v.len(),
            SliceElems::Bytes(b) => b.len(),
            SliceElems::Sparse { len, .. } => *len,
        }
    }

    /// The number of elements actually stored: a sparse
    /// sequence's declared length runs past its stored prefix
    /// (see `Value::SparseVec`); every other form stores them
    /// all.
    pub(crate) fn stored_len(&self) -> usize {
        match self {
            SliceElems::Values(v) => v.len(),
            SliceElems::Bytes(b) => b.len(),
            SliceElems::Sparse { stored, .. } => stored.len(),
        }
    }

    /// The element at `idx`, cloned out of the element storage (a
    /// byte element is constructed as an integer).
    pub(crate) fn at(&self, idx: usize) -> Value {
        match self {
            SliceElems::Values(v) => v[idx].clone(),
            SliceElems::Bytes(b) => Value::Uint(b[idx] as u128),
            SliceElems::Sparse { stored, .. } => stored
                .get(idx)
                .cloned()
                .unwrap_or(Value::Uninit),
        }
    }

    /// The element at `idx`, borrowed when it is stored element
    /// storage (a byte element is a constructed temporary).
    pub(crate) fn get(&self, idx: usize) -> Cow<'_, Value> {
        match self {
            SliceElems::Values(v) => Cow::Borrowed(&v[idx]),
            SliceElems::Bytes(b) => Cow::Owned(Value::Uint(b[idx] as u128)),
            SliceElems::Sparse { stored, .. } => match stored.get(idx) {
                Some(v) => Cow::Borrowed(v),
                None => Cow::Owned(Value::Uninit),
            },
        }
    }

    /// Truncate to at most `len` elements (a fat reference's
    /// carried length applied to its base's full element list, as
    /// `Vec::truncate` applies it in `slice_elements`).
    fn truncated(self, len: usize) -> Self {
        match self {
            SliceElems::Values(Cow::Borrowed(s)) => {
                SliceElems::Values(Cow::Borrowed(&s[..len.min(s.len())]))
            }
            SliceElems::Values(Cow::Owned(mut v)) => {
                v.truncate(len);
                SliceElems::Values(Cow::Owned(v))
            }
            SliceElems::Bytes(Cow::Borrowed(b)) => {
                SliceElems::Bytes(Cow::Borrowed(&b[..len.min(b.len())]))
            }
            SliceElems::Bytes(Cow::Owned(mut v)) => {
                v.truncate(len);
                SliceElems::Bytes(Cow::Owned(v))
            }
            SliceElems::Sparse { stored, len: declared } => SliceElems::Sparse {
                stored: match stored {
                    Cow::Borrowed(s) => {
                        Cow::Borrowed(&s[..len.min(s.len())])
                    }
                    Cow::Owned(mut v) => {
                        v.truncate(len);
                        Cow::Owned(v)
                    }
                },
                len: declared.min(len),
            },
        }
    }

    /// The elements as an owned vector (what `slice_elements`
    /// returns): byte elements materialise as integers.
    pub(crate) fn into_vec(self) -> Vec<Value> {
        match self {
            SliceElems::Values(v) => v.into_owned(),
            SliceElems::Bytes(b) => {
                b.iter().map(|byte| Value::Uint(*byte as u128)).collect()
            }
            // Only reached at or under the materialisation
            // limit (see the guard in `slice_elements`): the
            // tail materialises as the uninitialised slots it
            // denotes.
            SliceElems::Sparse { stored, len } => {
                let mut v = stored.into_owned();
                v.resize(len, Value::Uninit);
                v
            }
        }
    }

    /// The snapshot value a range index over these elements
    /// denotes (what `index_range` stores as the sub-slice's
    /// base): the materialised elements of the range — or, when
    /// the range runs past a sparse sequence's stored elements,
    /// the sparse form over the stored part of the range, its
    /// remaining slots uninitialised (see `Value::SparseVec`).
    pub(crate) fn sub_snapshot(&self, start: usize, end: usize) -> Value {
        match self {
            SliceElems::Sparse { stored, .. } if end > stored.len() => {
                Value::SparseVec {
                    elems: stored[start.min(stored.len())..].to_vec(),
                    len: (end - start) as u128,
                }
            }
            _ => Value::Tuple((start..end).map(|i| self.at(i)).collect()),
        }
    }
}

// `io::Error` values are built only in the bitpacked `Repr`
// layout (`io_error_value` / `io_os_error_value` /
// `io_simple_error_value` above): that is the representation
// the guest's own converted `Error::kind` / `raw_os_error`
// bodies read, so every producer - the fs layer here and the
// process model alike - goes through those constructors.

/// A host `io::Error` as a guest `io::Error` value: the errno
/// form when the host error carries one (the faithful case for
/// filesystem operations), otherwise the kind-only form.
fn host_io_error(error: &std::io::Error) -> Value {
    match error.raw_os_error() {
        Some(code) => io_os_error_value(code),
        None => io_simple_error_value(42),
    }
}

fn io_result_ok(value: Value) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![value],
    }
}

fn io_result_err(error: Value) -> Value {
    Value::Enum {
        variant: 1,
        fields: vec![error],
    }
}

fn io_result_from_host(result: Result<Value, std::io::Error>) -> Value {
    match result {
        Ok(value) => io_result_ok(value),
        Err(error) => io_result_err(host_io_error(&error)),
    }
}

/// Run the configured tool runner for this command, when the
/// command's program is one of the intercepted tool programs;
/// `None` when no interception applies, in which case the
/// caller keeps the process model's standing answer. A runner
/// error becomes the run's unsupported error: the tool ran the
/// invocation and reported it cannot be carried out.
fn tool_outcome(
    interception: Option<&crate::interp::ToolInterception>,
    state: &crate::interp::CommandState,
) -> InterpResult<Option<crate::interp::ToolRunOutcome>> {
    let Some(interception) = interception else {
        return Ok(None);
    };
    if !interception.programs.iter().any(|p| p == &state.program) {
        return Ok(None);
    }
    let request = crate::interp::ToolRunRequest {
        program: state.program.clone(),
        args: state.args.clone(),
        argv0: state.argv0.clone(),
        envs: state.envs.clone(),
        env_clear: state.env_clear,
        cwd: state.cwd.clone(),
    };
    match (interception.runner.0)(&request) {
        Ok(outcome) => Ok(Some(outcome)),
        Err(reason) => Err(unsupported(reason)),
    }
}

/// A `Cow<str>` value over the given bytes: borrowed when
/// they are valid UTF-8, owned with replacement characters
/// otherwise (the host `from_utf8_lossy` semantics); in this
/// model both variants carry the string data itself.
fn cow_str_value(bytes: &[u8]) -> Value {
    match std::str::from_utf8(bytes) {
        Ok(text) => Value::Enum {
            variant: 0,
            fields: vec![Value::Str(text.to_string())],
        },
        Err(_) => Value::Enum {
            variant: 1,
            fields: vec![Value::Str(
                String::from_utf8_lossy(bytes).into_owned(),
            )],
        },
    }
}

/// A `Vec<u8>` value: the element-list aggregate the
/// interpreter uses for vectors.
fn byte_vec_value(bytes: &[u8]) -> Value {
    Value::Enum {
        variant: 0,
        fields: bytes.iter().map(|b| Value::Uint(u128::from(*b))).collect(),
    }
}

/// An `OsString` value in std's shape: `OsString { inner: Buf {
/// inner: Vec<u8> } }`, the same nesting `pathbuf_value` builds
/// for the `OsString` inside a `PathBuf`.
fn os_string_value(bytes: &[u8]) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Enum {
            variant: 0,
            fields: vec![byte_vec_value(bytes)],
        }],
    }
}

/// A `std::process::ExitStatus` value: the platform wait
/// status in std's own layout (the outer struct wraps the sys
/// status struct, which wraps the raw `c_int`). An exited
/// process's raw status carries its code in the high byte; a
/// process ended by a signal carries the signal number in the
/// low byte, as `waitpid` reports it. The guest's own
/// `ExitStatus::code` body decodes it from there.
fn exit_status_value(exit_code: i32) -> Value {
    let raw: i32 = if exit_code >= 0 {
        exit_code << 8
    } else {
        -exit_code
    };
    Value::Enum {
        variant: 0,
        fields: vec![Value::Enum {
            variant: 0,
            fields: vec![Value::Int(i128::from(raw))],
        }],
    }
}

/// A `std::process::Output` value: the exit status and the two
/// captured byte vectors, in declaration order.
fn process_output_value(outcome: &crate::interp::ToolRunOutcome) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![
            exit_status_value(outcome.exit_code),
            byte_vec_value(&outcome.stdout),
            byte_vec_value(&outcome.stderr),
        ],
    }
}

/// A `&Path` / `&OsStr` value over the given bytes. When the
/// bytes are UTF-8 the value is the string data itself, which
/// is what a path over string data denotes everywhere else in
/// this model (a `Path::new` over a `&str` keeps the string
/// value); non-UTF-8 bytes become a fat pointer to a fresh
/// aggregate of the byte elements, the shape the interpreter's
/// own slice operations produce for a materialised sub-slice.
fn sub_path_value(bytes: &[u8]) -> Value {
    match std::str::from_utf8(bytes) {
        Ok(text) => Value::Str(text.to_string()),
        Err(_) => Value::FatRef {
            base: Box::new(Value::Tuple(
                bytes.iter().map(|b| Value::Uint(u128::from(*b))).collect(),
            )),
            len: bytes.len() as u128,
        },
    }
}

/// The byte length of a path's parent, following std's
/// components-based `Path::parent` on unix separators: trailing
/// separators and trailing `.` components take no part (as
/// `Components` normalizes them away), the last remaining
/// component is removed, and a root or empty path has no
/// parent. The parent is always a prefix of the path's bytes.
fn path_parent_len(bytes: &[u8]) -> Option<usize> {
    let mut end = bytes.len();
    loop {
        if end > 1 && bytes[end - 1] == b'/' {
            end -= 1;
        } else if end >= 2 && bytes[end - 2] == b'/' && bytes[end - 1] == b'.' {
            end -= 2;
        } else {
            break;
        }
    }
    if end == 0 || bytes[..end].iter().all(|b| *b == b'/') {
        return None;
    }
    match bytes[..end].iter().rposition(|b| *b == b'/') {
        None => Some(0),
        Some(0) => Some(1),
        Some(mut i) => {
            while i > 1 && bytes[i - 1] == b'/' {
                i -= 1;
            }
            Some(i)
        }
    }
}

/// The byte range of a path's file name, following std's
/// `Path::file_name`: the final component when it is a normal
/// one (a trailing `.` component names the previous one, as
/// `Components` drops it; `..` and the root have no file name).
fn path_file_name_range(bytes: &[u8]) -> Option<(usize, usize)> {
    let mut end = bytes.len();
    while end > 1 && bytes[end - 1] == b'/' {
        end -= 1;
    }
    if end == 0 || bytes[..end].iter().all(|b| *b == b'/') {
        return None;
    }
    let start = bytes[..end]
        .iter()
        .rposition(|b| *b == b'/')
        .map_or(0, |i| i + 1);
    let seg = &bytes[start..end];
    if seg == b".." {
        return None;
    }
    if seg == b"." {
        if start == 0 {
            return None;
        }
        return path_file_name_range(&bytes[..start - 1]);
    }
    Some((start, end))
}

/// One path component in std's `Components` normalization
/// (unix separators): a leading `.` is its own component, `.`
/// elsewhere and empty segments are dropped, and a leading
/// run of separators is the root.
/// A path's components, in `Components` order.
#[derive(PartialEq, Eq)]
enum PathComp<'a> {
    Root,
    Cur,
    Parent,
    Normal(&'a [u8]),
}

/// `PathBuf::push` on raw bytes (unix): an absolute argument
/// replaces the base; otherwise the argument is appended, with
/// a separator inserted unless the base is empty or already
/// ends with one.
fn path_push_bytes(base: &[u8], path: &[u8]) -> Vec<u8> {
    if path.first() == Some(&b'/') {
        return path.to_vec();
    }
    if base.is_empty() || path.is_empty() {
        let mut out = base.to_vec();
        out.extend_from_slice(path);
        return out;
    }
    let mut out = base.to_vec();
    if !out.ends_with(b"/") {
        out.push(b'/');
    }
    out.extend_from_slice(path);
    out
}

fn path_comps(bytes: &[u8]) -> Vec<PathComp<'_>> {
    let mut comps = Vec::new();
    let mut rest = bytes;
    if rest.first() == Some(&b'/') {
        comps.push(PathComp::Root);
        while rest.first() == Some(&b'/') {
            rest = &rest[1..];
        }
    } else if rest == b"." || rest.starts_with(b"./") {
        comps.push(PathComp::Cur);
        rest = if rest.len() > 1 { &rest[2..] } else { &rest[1..] };
        while rest.first() == Some(&b'/') {
            rest = &rest[1..];
        }
    }
    for seg in rest.split(|b| *b == b'/') {
        match seg {
            b"" | b"." => {}
            b".." => comps.push(PathComp::Parent),
            s => comps.push(PathComp::Normal(s)),
        }
    }
    comps
}

/// std's `Path::starts_with`: the child's components are a
/// prefix of the path's components.
fn path_starts_with(path: &[u8], child: &[u8]) -> bool {
    let comps = path_comps(path);
    let child = path_comps(child);
    comps.len() >= child.len() && comps[..child.len()] == child[..]
}

/// std's `Path::ends_with`: the child's components are a
/// suffix of the path's components.
fn path_ends_with(path: &[u8], child: &[u8]) -> bool {
    let comps = path_comps(path);
    let child = path_comps(child);
    comps.len() >= child.len() && comps[comps.len() - child.len()..] == child[..]
}

/// std's `rsplit_file_at_dot` on a file name's bytes: the
/// split at the last dot, as a (before, after) pair, with std's
/// special cases (a `..` name splits as itself with no
/// extension side; a dot in the first position does not start
/// an extension).
fn rsplit_file_at_dot(name: &[u8]) -> (Option<&[u8]>, Option<&[u8]>) {
    if name == b".." {
        return (Some(name), None);
    }
    match name.iter().rposition(|b| *b == b'.') {
        None => (None, Some(name)),
        Some(0) => (Some(name), None),
        Some(i) => (Some(&name[..i]), Some(&name[i + 1..])),
    }
}

/// The bytes of an `OsString` value, in either of the model's
/// two shapes: the string form (the inner data is the string
/// itself) and the buffer form (an `OsString` wrapping a `Buf`
/// wrapping the byte vector).
fn os_string_bytes(value: &Value) -> Option<Vec<u8>> {
    let Value::Enum { fields, .. } = value else {
        return None;
    };
    match fields.as_slice() {
        [Value::Str(text)] => Some(text.clone().into_bytes()),
        [buf] => {
            let Value::Enum { fields, .. } = buf else {
                return None;
            };
            let [vec] = fields.as_slice() else {
                return None;
            };
            byte_list_value(vec)
        }
        _ => None,
    }
}

/// The bytes of a byte-vector value (the element-list
/// aggregate): `None` when any element is not a byte.
fn byte_list_value(value: &Value) -> Option<Vec<u8>> {
    let Value::Enum { fields, .. } = value else {
        return None;
    };
    let mut out = Vec::with_capacity(fields.len());
    for field in fields {
        match field {
            Value::Uint(b) => out.push(*b as u8),
            _ => return None,
        }
    }
    Some(out)
}

/// The bytes of a `PathBuf` value: its one field is the
/// buffer's `OsString`.
fn path_buf_bytes(value: &Value) -> Option<Vec<u8>> {
    let Value::Enum { fields, .. } = value else {
        return None;
    };
    let [os_string] = fields.as_slice() else {
        return None;
    };
    os_string_bytes(os_string)
}

/// The sys-level `std::fs::ReadDir` value: the struct whose
/// fields are the owned-iterator state ([entry results, next
/// index]) the fs sys layer's snapshot builds. The public
/// `std::fs::read_dir` body wraps it in the public one-field
/// `ReadDir` aggregate itself.
fn read_dir_value(items: Vec<Value>) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Tuple(items), Value::Uint(0)],
    }
}

/// A `std::fs::DirEntry` value: the public one-field wrapper
/// around the pair of the entry's full path (as a `PathBuf`)
/// and its file name (as an `OsString`) — the two facts the
/// entry's modelled methods read.
fn dir_entry_value(entry: &std::fs::DirEntry) -> Value {
    use std::os::unix::ffi::OsStrExt as _;
    Value::Enum {
        variant: 0,
        fields: vec![Value::Tuple(vec![
            pathbuf_value(&host_path_bytes(&entry.path())),
            os_string_value(entry.file_name().as_bytes()),
        ])],
    }
}

/// The (full path, file name) values of a modelled `DirEntry`.
fn dir_entry_parts(value: &Value) -> Option<(Value, Value)> {
    let Value::Enum { fields, .. } = value else {
        return None;
    };
    let [inner] = fields.as_slice() else {
        return None;
    };
    let Value::Tuple(pair) = inner else {
        return None;
    };
    let [path, name] = pair.as_slice() else {
        return None;
    };
    Some((path.clone(), name.clone()))
}

/// A host path from raw path bytes (guest paths are byte
/// strings, not necessarily UTF-8).
fn host_path(bytes: &[u8]) -> std::path::PathBuf {
    use std::os::unix::ffi::OsStrExt;
    std::ffi::OsStr::from_bytes(bytes).to_os_string().into()
}

fn host_path_bytes(path: &std::path::Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_os_str().as_bytes().to_vec()
}

/// One `Iterator::next` / `DoubleEndedIterator::next_back`
/// step of a `std::path::Components` state (see
/// `Builtin::PathComponents`): rebuild the host iterator
/// over the state's byte string, replay the components
/// already consumed from each end, and take one more, so
/// the parsing rules (root, `.` and `..` handling,
/// repeated separators) are the host's own. The counts in
/// the returned state advance on the side the step came
/// from; an exhausted iterator returns its state
/// unchanged, so it stays exhausted.
fn components_step_state(
    state: &Value,
    from_back: bool,
) -> InterpResult<(Value, Option<Value>)> {
    let fields = iterator_fields(state, "std::path::Components")?;
    if fields.len() != 3 {
        return Err(InterpError::type_mismatch(
            "Components state",
            "[path bytes, front count, back count]",
            format!("{} field(s)", fields.len()),
        ));
    }
    let bytes = match &fields[0] {
        Value::Bytes(b) => b.clone(),
        Value::Str(s) => s.clone().into_bytes(),
        other => {
            return Err(InterpError::type_mismatch(
                "Components state",
                "path byte string",
                other.kind_name(),
            ))
        }
    };
    let (front, back) = match (&fields[1], &fields[2]) {
        (Value::Uint(f), Value::Uint(b)) => (*f, *b),
        _ => {
            return Err(InterpError::type_mismatch(
                "Components state",
                "front and back counts",
                "non-integer count(s)",
            ))
        }
    };
    use std::os::unix::ffi::OsStrExt;
    let path = std::path::Path::new(std::ffi::OsStr::from_bytes(&bytes));
    let mut iter = path.components();
    for _ in 0..front {
        let _ = iter.next();
    }
    for _ in 0..back {
        let _ = iter.next_back();
    }
    let item = if from_back { iter.next_back() } else { iter.next() };
    match item {
        Some(component) => {
            let (front, back) = if from_back {
                (front, back + 1)
            } else {
                (front + 1, back)
            };
            Ok((
                components_state(&bytes, front, back),
                Some(component_value(component)),
            ))
        }
        None => Ok((state.clone(), None)),
    }
}

/// The `std::path::Components` iterator state in this
/// model: the path's byte string and the numbers of
/// components consumed from the front and from the back.
/// The parse itself is replayed from this state with the
/// host's `std::path` on every step (see `components_step`),
/// so no parse state has to be represented here.
fn components_state(bytes: &[u8], front: u128, back: u128) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![
            Value::Bytes(bytes.to_vec()),
            Value::Uint(front),
            Value::Uint(back),
        ],
    }
}

/// A `std::path::Component` value in std's shape: the
/// variants in declaration order (`Prefix`, `RootDir`,
/// `CurDir`, `ParentDir`, `Normal`); the `Normal` payload is
/// the component's byte string (the model's `&OsStr`, the
/// same form `Builtin::PathPart` returns for `&Path` /
/// `&OsStr` results).
fn component_value(component: std::path::Component) -> Value {
    use std::os::unix::ffi::OsStrExt;
    match component {
        std::path::Component::Prefix(prefix) => Value::Enum {
            variant: 0,
            fields: vec![prefix_component_value(prefix)],
        },
        std::path::Component::RootDir => Value::Enum {
            variant: 1,
            fields: Vec::new(),
        },
        std::path::Component::CurDir => Value::Enum {
            variant: 2,
            fields: Vec::new(),
        },
        std::path::Component::ParentDir => Value::Enum {
            variant: 3,
            fields: Vec::new(),
        },
        std::path::Component::Normal(part) => Value::Enum {
            variant: 4,
            fields: vec![Value::Bytes(part.as_bytes().to_vec())],
        },
    }
}

/// A `std::path::PrefixComponent` value in std's shape,
/// `{ kind, raw }`: the `Prefix` enum in declaration order
/// (`Verbatim`, `VerbatimUNC`, `VerbatimDisk`, `DeviceNS`,
/// `UNC`, `Disk`) and the raw prefix string. Prefixes do
/// not occur on Unix; the shape is built for completeness.
fn prefix_component_value(prefix: std::path::PrefixComponent) -> Value {
    use std::os::unix::ffi::OsStrExt;
    let os = |s: &std::ffi::OsStr| Value::Bytes(s.as_bytes().to_vec());
    let kind = match prefix.kind() {
        std::path::Prefix::Verbatim(s) => Value::Enum {
            variant: 0,
            fields: vec![os(s)],
        },
        std::path::Prefix::VerbatimUNC(a, b) => Value::Enum {
            variant: 1,
            fields: vec![os(a), os(b)],
        },
        std::path::Prefix::VerbatimDisk(d) => Value::Enum {
            variant: 2,
            fields: vec![Value::Uint(d as u128)],
        },
        std::path::Prefix::DeviceNS(s) => Value::Enum {
            variant: 3,
            fields: vec![os(s)],
        },
        std::path::Prefix::UNC(a, b) => Value::Enum {
            variant: 4,
            fields: vec![os(a), os(b)],
        },
        std::path::Prefix::Disk(d) => Value::Enum {
            variant: 5,
            fields: vec![Value::Uint(d as u128)],
        },
    };
    Value::Enum {
        variant: 0,
        fields: vec![kind, os(prefix.as_os_str())],
    }
}

/// A `PathBuf` value in std's shape: `PathBuf { inner:
/// OsString { inner: Buf { inner: Vec<u8> } } }`, the Vec being
/// the element-list value.
fn pathbuf_value(bytes: &[u8]) -> Value {
    let vec = Value::Enum {
        variant: 0,
        fields: bytes.iter().map(|b| Value::Uint(*b as u128)).collect(),
    };
    let buf = Value::Enum {
        variant: 0,
        fields: vec![vec],
    };
    let os_string = Value::Enum {
        variant: 0,
        fields: vec![buf],
    };
    Value::Enum {
        variant: 0,
        fields: vec![os_string],
    }
}

/// A `std::sys::fs::Metadata` value in std's shape: `{ stat }`
/// with the stat fields in libc's order (the mode is the
/// fourth field, which is what `permissions()` reads).
fn metadata_value(md: &std::fs::Metadata) -> Value {
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
    Value::Enum {
        variant: 0,
        fields: vec![stat],
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

/// The IR type of a primitive tag (`i32`, `usize`, `f64`,
/// `bool`): the Self types the arithmetic trait models compute
/// on. `None` for anything else.
fn primitive_ty(tag: &str) -> Option<Ty> {
    Some(match tag {
        "i8" => Ty::Int(IntTy::I8),
        "i16" => Ty::Int(IntTy::I16),
        "i32" => Ty::Int(IntTy::I32),
        "i64" => Ty::Int(IntTy::I64),
        "i128" => Ty::Int(IntTy::I128),
        "isize" => Ty::Int(IntTy::Isize),
        "u8" => Ty::Uint(UintTy::U8),
        "u16" => Ty::Uint(UintTy::U16),
        "u32" => Ty::Uint(UintTy::U32),
        "u64" => Ty::Uint(UintTy::U64),
        "u128" => Ty::Uint(UintTy::U128),
        "usize" => Ty::Uint(UintTy::Usize),
        "f32" => Ty::Float(FloatTy::F32),
        "f64" => Ty::Float(FloatTy::F64),
        "bool" => Ty::Bool,
        _ => return None,
    })
}

/// The Self type of an impl-qualified arithmetic trait call, as
/// an IR type: the only width / signedness source the model
/// has, so a name that does not carry a primitive Self type
/// does not qualify.
fn primitive_self_ty(name: &str) -> InterpResult<Ty> {
    let self_tag = impl_qualified_self_type(name).ok_or_else(|| {
        unsupported(format!(
            "arithmetic trait call {name} does not name its Self type"
        ))
    })?;
    primitive_ty(&self_tag).ok_or_else(|| {
        unsupported(format!(
            "arithmetic trait call for {self_tag} is not runtime-supported"
        ))
    })
}

/// Core's overflow panic messages for the arithmetic traits,
/// verbatim (they are the messages the checked impls raise).
fn primitive_overflow_message(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "attempt to add with overflow",
        BinOp::Sub => "attempt to subtract with overflow",
        BinOp::Mul => "attempt to multiply with overflow",
        BinOp::Div => "attempt to divide with overflow",
        BinOp::Rem => "attempt to calculate the remainder with overflow",
        BinOp::Shl => "attempt to shift left with overflow",
        BinOp::Shr => "attempt to shift right with overflow",
        _ => "attempt to perform an arithmetic operation with overflow",
    }
}

/// Whether a resolved target denotes one element of a container
/// (its projection ends at an index step): a fat pointer over
/// such a target denotes the elements from that position on, not
/// the element itself.
fn target_ends_at_element(target: &Resolved) -> bool {
    let projection = match target {
        Resolved::Place { projection, .. }
        | Resolved::Static { projection, .. }
        | Resolved::Alloc { projection, .. }
        | Resolved::Value(_, projection) => projection,
    };
    matches!(projection.last(), Some(RefProj::Index(_)))
}

/// The elements of an already-resolved value, borrowed where
/// possible: the borrowing form of [`value_elements`], with the
/// same results and errors.
fn elements_of<'a>(value: Cow<'a, Value>) -> InterpResult<SliceElems<'a>> {
    match value {
        Cow::Borrowed(v) => Ok(match v {
            Value::Tuple(items) => SliceElems::Values(Cow::Borrowed(items)),
            Value::Enum { fields, .. } => SliceElems::Values(Cow::Borrowed(fields)),
            Value::SparseVec { elems, len } => SliceElems::Sparse {
                stored: Cow::Borrowed(elems),
                len: usize::try_from(*len).unwrap_or(usize::MAX),
            },
            Value::Bytes(b) => SliceElems::Bytes(Cow::Borrowed(b)),
            Value::Str(text) => SliceElems::Bytes(Cow::Borrowed(text.as_bytes())),
            other => {
                return Err(InterpError::type_mismatch(
                    "slice elements",
                    "aggregate, byte or string value",
                    other.kind_name(),
                ))
            }
        }),
        Cow::Owned(v) => Ok(match v {
            Value::Tuple(items) => SliceElems::Values(Cow::Owned(items)),
            Value::Enum { fields, .. } => SliceElems::Values(Cow::Owned(fields)),
            Value::SparseVec { elems, len } => SliceElems::Sparse {
                stored: Cow::Owned(elems),
                len: usize::try_from(len).unwrap_or(usize::MAX),
            },
            Value::Bytes(b) => SliceElems::Bytes(Cow::Owned(b)),
            Value::Str(text) => SliceElems::Bytes(Cow::Owned(text.into_bytes())),
            other => {
                return Err(InterpError::type_mismatch(
                    "slice elements",
                    "aggregate, byte or string value",
                    other.kind_name(),
                ))
            }
        }),
    }
}

/// The element values of an aggregate-like value: tuple items, ADT
/// fields, or the bytes of a byte/string value as integers.
fn value_elements(value: &Value) -> InterpResult<Vec<Value>> {
    Ok(match value {
        Value::Tuple(items) => items.clone(),
        Value::Enum { fields, .. } => fields.clone(),
        Value::Bytes(b) => b.iter().map(|byte| Value::Uint(*byte as u128)).collect(),
        Value::Str(s) => s
            .as_bytes()
            .iter()
            .map(|byte| Value::Uint(*byte as u128))
            .collect(),
        other => {
            return Err(InterpError::type_mismatch(
                "slice elements",
                "aggregate, byte or string value",
                other.kind_name(),
            ))
        }
    })
}

fn int_uint_eq(x: i128, y: u128) -> bool {
    x >= 0 && (x as u128) == y
}

fn int_uint_cmp(x: i128, y: u128) -> Ordering {
    if x < 0 {
        Ordering::Less
    } else {
        (x as u128).cmp(&y)
    }
}

/// An integer index argument as a `usize`.
/// The (start, end) element bounds of a range index against a
/// receiver of length `len`, applying core's checks in core's
/// order with core's panic messages. Inclusive ends are converted
/// to exclusive ones first (an inclusive end of `usize::MAX`
/// overflows that conversion, which core reports separately).
/// A reference value one sub-slice deeper: `value` must be a
/// `Value::Ref`; the result names the same target with a
/// resolved `RefProj::Subslice` step appended, so reads and
/// writes through it alias the original storage at the
/// sub-slice's offset. A `Value` target (a constant aggregate)
/// carries the step in its projection, like the other targets:
/// reads through it stay exact, and writes through it are
/// refused by the write path, as for any constant.
fn subslice_ref(value: &Value, offset: usize, len: usize, _name: &str) -> InterpResult<Value> {
    let step = RefProj::Subslice { offset, len };
    match value {
        Value::Ref(RefTarget::Place {
            frame,
            local,
            projection,
        }) => {
            let mut projection = projection.clone();
            projection.push(step);
            Ok(Value::Ref(RefTarget::place(*frame, *local, projection)))
        }
        Value::Ref(RefTarget::Static {
            name: static_name,
            projection,
        }) => {
            let mut projection = projection.clone();
            projection.push(step);
            Ok(Value::Ref(RefTarget::Static {
                name: static_name.clone(),
                projection,
            }))
        }
        Value::Ref(RefTarget::Alloc { id, projection }) => {
            let mut projection = projection.clone();
            projection.push(step);
            Ok(Value::Ref(RefTarget::Alloc {
                id: *id,
                projection,
            }))
        }
        Value::Ref(RefTarget::Value(v, projection)) => {
            let mut projection = projection.clone();
            projection.push(step);
            Ok(Value::Ref(RefTarget::Value(v.clone(), projection)))
        }
        other => Err(InterpError::type_mismatch(
            "IndexMut::index_mut",
            "reference value",
            other.kind_name(),
        )),
    }
}

fn range_bounds(
    kind: &RangeKindForIndex,
    bound: &dyn Fn(usize) -> InterpResult<usize>,
    len: usize,
) -> InterpResult<(usize, usize)> {
    use RangeKindForIndex::*;
    let (start, end) = match kind {
        Full => (0, len),
        Range => (bound(0)?, bound(1)?),
        From => (bound(0)?, len),
        To => (0, bound(0)?),
        Inclusive => {
            let start = bound(0)?;
            let end = bound(1)?;
            if end == usize::MAX {
                return Err(InterpError::panic("range end index overflow".to_string()));
            }
            (start, end + 1)
        }
        ToInclusive => {
            let end = bound(0)?;
            if end == usize::MAX {
                return Err(InterpError::panic("range end index overflow".to_string()));
            }
            (0, end + 1)
        }
    };
    if start > end {
        return Err(InterpError::panic(format!(
            "slice index starts at {start} but ends at {end}"
        )));
    }
    if end > len {
        return Err(InterpError::panic(format!(
            "range end index {end} out of range for slice of length {len}"
        )));
    }
    if start > len {
        return Err(InterpError::panic(format!(
            "range start index {start} out of range for slice of length {len}"
        )));
    }
    Ok((start, end))
}

enum RangeKindForIndex {
    Full,
    Range,
    From,
    To,
    Inclusive,
    ToInclusive,
}

/// core's message for a string slice index that is not a char
/// boundary (`str::slice_error_fail`).
fn str_slice_error(text: &str, start: usize, end: usize) -> String {
    for idx in [start, end] {
        if idx != 0 && idx != text.len() && !text.is_char_boundary(idx) {
            let lo = text[..idx].chars().next_back().map_or(0, |c| idx - c.len_utf8());
            let ch = text[lo..].chars().next().expect("char at boundary");
            let hi = lo + ch.len_utf8();
            return format!(
                "byte index {idx} is not a char boundary; it is inside {ch:?} (bytes {lo}..{hi}) of `{text}`"
            );
        }
    }
    format!("byte index {start} is not a char boundary")
}

fn expect_index(value: &Value, name: &str) -> InterpResult<usize> {
    match value {
        Value::Uint(v) => usize::try_from(*v).map_err(|_| {
            InterpError::panic(format!("index {v} does not fit in usize"))
        }),
        Value::Int(v) if *v >= 0 => usize::try_from(*v).map_err(|_| {
            InterpError::panic(format!("index {v} does not fit in usize"))
        }),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "integer index",
            other.kind_name(),
        )),
    }
}

/// Whether a rendered type tag names an integer type.
fn is_integer_tag(tag: &str) -> bool {
    matches!(
        tag,
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64" | "u128"
            | "usize"
    )
}

/// Split a rendered type list on the commas that sit at nesting
/// depth zero (for tuple/array type tags).
fn split_top_level(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    let tail = s[start..].trim();
    if !tail.is_empty() {
        parts.push(tail);
    }
    parts.into_iter().filter(|p| !p.is_empty()).collect()
}

/// The uninitialized value of the shape a rendered type tag fixes,
/// for `Box::new_uninit`: an array type is a tuple of uninitialized
/// elements (the count is part of the type), a `MaybeUninit<T>`
/// wrapper is transparent, and any other type is a single
/// uninitialized value. `None` for an array too large to materialize.
fn uninit_value_for_tag(tag: &str) -> Option<Value> {
    let tag = tag.trim();
    if let Some(inner) = tag
        .strip_prefix("std::mem::MaybeUninit<")
        .and_then(|t| t.strip_suffix('>'))
    {
        return uninit_value_for_tag(inner);
    }
    if tag.starts_with('[') && tag.ends_with(']') {
        let inner = &tag[1..tag.len() - 1];
        if let Some((_, n)) = inner.rsplit_once(';') {
            if let Ok(n) = n.trim().parse::<usize>() {
                if n > 4096 {
                    return None;
                }
                // The pointee is a `MaybeUninit<[T; N]>`, and the
                // `vec!` expansion writes the elements through the
                // place `(*ptr).1.0`: the union's value field (index
                // 1), then `ManuallyDrop`'s only field. The value is
                // shaped to that layout -- field 0 is the unit
                // `uninit` field, field 1 wraps the uninitialized
                // element array -- so the projected write lands on
                // the array.
                return Some(Value::Enum {
                    variant: 0,
                    fields: vec![
                        Value::Unit,
                        Value::Enum {
                            variant: 0,
                            fields: vec![Value::Tuple(vec![Value::Uninit; n])],
                        },
                    ],
                });
            }
        }
    }
    Some(Value::Uninit)
}

/// The element values inside a `box_assume_init_into_vec_unsafe`
/// argument's pointee: an array value is its element tuple; a
/// `MaybeUninit` wrapper value wraps the array in one of its
/// fields. Anything else is not an array pointee.
fn assume_init_elements(pointee: Value) -> Option<Vec<Value>> {
    match pointee {
        Value::Tuple(elems) => Some(elems),
        // The pointee is a `MaybeUninit<[T; N]>`: either a
        // single-field wrapper (the array is the only field) or the
        // layout-shaped value `Box::new_uninit` produces (the array
        // sits in the union's value field, inside `ManuallyDrop`).
        // Search the wrapper fields for the element array.
        Value::Enum { fields, .. } if fields.len() <= 2 => {
            fields.into_iter().find_map(assume_init_elements)
        }
        _ => None,
    }
}

/// The `Default` value for a rendered type tag, for the types whose
/// default is fixed by the type alone in this value model.
/// The (bit width, signedness) of a primitive integer type tag
/// as rendered by `ty_tag` (`i32`, `usize`, ...). `isize` /
/// `usize` are 64-bit on every platform this runs on.
/// An element count argument (of a memory intrinsic), as a
/// `usize`. Mirrors the count handling of the interpreter's
/// element copy.
fn element_count(value: &Value, name: &str) -> InterpResult<usize> {
    match value {
        Value::Uint(n) => usize::try_from(*n).map_err(|_| {
            InterpError::type_mismatch(
                format!("builtin {name}"),
                "count that fits in usize",
                format!("{n}"),
            )
        }),
        Value::Int(n) if *n >= 0 => Ok(*n as usize),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "unsigned integer count",
            other.kind_name(),
        )),
    }
}

/// The most scalar leaves a materialised fill value (a
/// `mem::zeroed` result, a `write_bytes` fill) may hold. The
/// interpreter's values are trees, so a fill of a large array
/// type is a tree with one value per element: the bound admits
/// a `mem::zeroed::<[u8; 1 << 20]>()` — the big-buffer shape
/// real programs use, the same order as the conversion bound
/// on `Rvalue::Repeat` aggregates — while keeping an absurd
/// type from exhausting memory.
const MAX_FILL_LEAVES: usize = 1 << 21;

/// The number of scalar leaves in the value of the type `tag`
/// names, for the fill functions' materialisation budget: a
/// scalar is one leaf, a tuple sums its parts, an array
/// multiplies. `None` for a tag outside the fill grammar.
fn tag_leaf_count(tag: &str) -> Option<usize> {
    let tag = tag.trim();
    match tag {
        "bool" | "char" | "f16" | "f32" | "f64" | "f128" | "()" => Some(1),
        _ if int_tag(tag).is_some() => Some(1),
        _ if tag.starts_with('(') && tag.ends_with(')') => {
            let inner = &tag[1..tag.len() - 1];
            let mut total: usize = 0;
            if !inner.trim().is_empty() {
                for part in split_top_level(inner) {
                    total = total.checked_add(tag_leaf_count(part)?)?;
                }
            }
            Some(total)
        }
        _ if tag.starts_with('[') && tag.ends_with(']') => {
            let inner = &tag[1..tag.len() - 1];
            let (elem, n) = inner.rsplit_once(';')?;
            let n: usize = n.trim().parse().ok()?;
            n.checked_mul(tag_leaf_count(elem)?)
        }
        _ => None,
    }
}

/// The number of scalar leaves in a materialised `value`,
/// mirroring [`tag_leaf_count`]: aggregates sum their parts,
/// anything else is one leaf -- including an uninitialised leaf,
/// which still occupies its slot in the aggregate's shape (a
/// materialised-but-uninitialised array has the same leaf count
/// as the filled value `write_bytes` would write into it, and
/// the caller compares exactly those two counts). Byte blocks
/// and strings keep the sequence path (their fills are
/// byte-wise), and a fat reference's pointee count is not a fact
/// of the value alone: those are `None` here.
fn value_leaf_count(value: &Value) -> Option<usize> {
    match value {
        Value::Tuple(items) => items.iter().map(value_leaf_count).sum(),
        Value::Enum { fields, .. } => fields.iter().map(value_leaf_count).sum(),
        Value::Coroutine { fields, saved, .. } => {
            let mut total = 0usize;
            for value in fields.iter().chain(saved.iter().flatten()) {
                total += value_leaf_count(value)?;
            }
            Some(total)
        }
        Value::Closure { env, .. } => env.iter().map(value_leaf_count).sum(),
        Value::Bytes(_)
        | Value::Str(_)
        | Value::FatRef { .. }
        | Value::DynRef { .. } => None,
        _ => Some(1),
    }
}

/// The value whose byte pattern is `byte` repeated, for the
/// element type `tag` names — the value `write_bytes` fills with.
/// Only types with a fixed byte interpretation are filled: for
/// any other type an arbitrary byte pattern need not be a valid
/// value, so the fill stays unsupported there. An array of a
/// fillable element type is filled element by element (the
/// shape `mem::zeroed` takes under optimisation, where the
/// intrinsic's type is the whole array), within the
/// materialisation budget.
fn byte_fill_value(tag: &str, byte: u8) -> Option<Value> {
    let tag = tag.trim();
    if let Some((bits, signed)) = int_tag(tag) {
        let mut pattern: u128 = 0;
        for _ in 0..bits / 8 {
            pattern = (pattern << 8) | u128::from(byte);
        }
        return Some(if signed {
            Value::Int(sign_extend(pattern, bits))
        } else {
            Value::Uint(pattern)
        });
    }
    Some(match tag {
        "bool" => match byte {
            0 => Value::Bool(false),
            1 => Value::Bool(true),
            _ => return None,
        },
        // A `char` is stored as its code point; the filled code
        // point is the byte repeated four times, valid only when
        // it names a character.
        "char" => Value::Uint(u128::from(char::from_u32(u32::from(byte) * 0x0101_0101)? as u32)),
        "f16" => Value::Float(crate::mir_convert::f16_to_f64(
            u16::from(byte) * 0x0101,
        )),
        "f32" => Value::Float(f64::from(f32::from_bits(u32::from(byte) * 0x0101_0101))),
        "f64" => Value::Float(f64::from_bits(u64::from(byte) * 0x0101_0101_0101_0101)),
        _ if tag.starts_with('[') && tag.ends_with(']') => {
            let inner = &tag[1..tag.len() - 1];
            let (elem, n) = inner.rsplit_once(';')?;
            let n: usize = n.trim().parse().ok()?;
            if tag_leaf_count(tag)? > MAX_FILL_LEAVES {
                return None;
            }
            let elem = byte_fill_value(elem, byte)?;
            Value::Tuple(vec![elem; n])
        }
        _ => return None,
    })
}

/// An allocation-size argument (`usize`, or a non-negative
/// `isize`-shaped integer) as a byte count.
fn alloc_size_arg(name: &str, v: &Value) -> InterpResult<u128> {
    match v {
        Value::Uint(n) => Ok(*n),
        Value::Int(n) if *n >= 0 => Ok(*n as u128),
        other => Err(InterpError::type_mismatch(
            name,
            "allocation size",
            other.kind_name(),
        )),
    }
}

fn int_tag(tag: &str) -> Option<(u32, bool)> {
    Some(match tag {
        "i8" => (8, true),
        "i16" => (16, true),
        "i32" => (32, true),
        "i64" => (64, true),
        "i128" => (128, true),
        "isize" => (64, true),
        "u8" => (8, false),
        "u16" => (16, false),
        "u32" => (32, false),
        "u64" => (64, false),
        "u128" => (128, false),
        "usize" => (64, false),
        _ => return None,
    })
}

/// The element type tag of a vector type tag: the element of
/// an array tag `[E; N]`, or the first top-level generic
/// argument of `...<E, ...>`.
fn vector_elem_tag(tag: &str) -> Option<&str> {
    let tag = tag.trim();
    if let Some(inner) = tag.strip_prefix('[') {
        return inner.split(';').next().map(str::trim);
    }
    let open = tag.find('<')?;
    let close = tag.rfind('>')?;
    if close <= open {
        return None;
    }
    let parts = split_top_level(&tag[open + 1..close]);
    parts.into_iter().next().map(str::trim)
}

/// A float lane's value.
fn simd_as_f64(value: &Value, ctx: &str) -> InterpResult<f64> {
    match value {
        Value::Float(x) => Ok(*x),
        other => Err(InterpError::type_mismatch(ctx, "float lane", other.kind_name())),
    }
}

/// The lane count named by a vector type tag: the length of
/// an array tag `[E; N]`, or a trailing const argument
/// (`Simd<i32, 4>`).
fn simd_tag_lane_count(tag: &str) -> Option<usize> {
    let tag = tag.trim();
    if let Some(inner) = tag.strip_prefix('[') {
        let (_, n) = inner.rsplit_once(';')?;
        return n.trim().trim_end_matches(']').trim().parse().ok();
    }
    let open = tag.find('<')?;
    let close = tag.rfind('>')?;
    if close <= open {
        return None;
    }
    let parts = split_top_level(&tag[open + 1..close]);
    parts.last().and_then(|p| p.trim().parse().ok())
}

/// The lanes of a vector value: the elements of its lane
/// array, reached through the one-field wrapper a SIMD type
/// is. A bare tuple is the lanes directly.
fn simd_lanes(value: &Value) -> InterpResult<Vec<Value>> {
    match value {
        Value::Tuple(lanes) => Ok(lanes.clone()),
        Value::Enum { fields, .. } if fields.len() == 1 => simd_lanes(&fields[0]),
        _ => Err(InterpError::type_mismatch(
            "simd operation",
            "vector value",
            value.kind_name(),
        )),
    }
}

/// Rebuild a vector value with `lanes` in the shape of
/// `template` (the value the lanes were taken from).
fn simd_wrap(template: &Value, lanes: Vec<Value>) -> InterpResult<Value> {
    match template {
        Value::Tuple(_) => Ok(Value::Tuple(lanes)),
        Value::Enum { variant, fields } if fields.len() == 1 => Ok(Value::Enum {
            variant: *variant,
            fields: vec![simd_wrap(&fields[0], lanes)?],
        }),
        _ => Err(InterpError::type_mismatch(
            "simd operation",
            "vector value",
            template.kind_name(),
        )),
    }
}

/// Whether a mask lane selects its element.
fn simd_lane_active(lane: &Value) -> InterpResult<bool> {
    match lane {
        Value::Int(v) => Ok(*v != 0),
        Value::Uint(v) => Ok(*v != 0),
        Value::Bool(b) => Ok(*b),
        Value::Float(f) => Ok(*f != 0.0),
        _ => Err(InterpError::type_mismatch(
            "simd mask lane",
            "integer or boolean lane",
            lane.kind_name(),
        )),
    }
}

/// A lane index argument (`u32` in the guest).
fn simd_index(value: &Value) -> InterpResult<usize> {
    let raw = as_u128(value, "simd lane index")?;
    usize::try_from(raw).map_err(|_| InterpError::Panic {
        message: format!("simd lane index {raw} out of range"),
        payload: None,
    })
}

/// Round a float result to a SIMD lane's width.
fn simd_round_float(x: f64, float_bits: Option<u32>) -> f64 {
    match float_bits {
        Some(16) => round_to_f16(x),
        Some(32) => (x as f32) as f64,
        _ => x,
    }
}

/// IEEE 754 maximumNumber: a NaN operand yields the other
/// operand, and +0 is greater than -0.
fn maximum_number(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else if a == b {
        if a.is_sign_positive() {
            a
        } else {
            b
        }
    } else {
        a.max(b)
    }
}

/// IEEE 754 minimumNumber: a NaN operand yields the other
/// operand, and -0 is less than +0.
fn minimum_number(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else if a == b {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else {
        a.min(b)
    }
}

/// One element-wise binary SIMD operation on two lanes of the
/// shape `info` = (bits, signed, float width).
fn simd_binop(
    op: SimdOp,
    a: &Value,
    b: &Value,
    info: (u32, bool, Option<u32>),
) -> InterpResult<Value> {
    use SimdOp as S;
    let (bits, signed, float_bits) = info;
    if float_bits.is_some() {
        let x = simd_as_f64(a, "simd lane")?;
        let y = simd_as_f64(b, "simd lane")?;
        let r = match op {
            S::Add => x + y,
            S::Sub => x - y,
            S::Mul => x * y,
            S::Div => x / y,
            S::Rem => x % y,
            S::MaximumNumber => maximum_number(x, y),
            S::MinimumNumber => minimum_number(x, y),
            _ => {
                return Err(unsupported(
                    "this simd operation on float lanes is not runtime-supported",
                ))
            }
        };
        return Ok(Value::Float(simd_round_float(r, float_bits)));
    }
    if matches!(op, S::SaturatingAdd | S::SaturatingSub) {
        let mask = bit_mask(bits);
        let pa = as_u128(a, "simd lane")? & mask;
        let pb = as_u128(b, "simd lane")? & mask;
        let pattern = if signed {
            let (sa, sb) = (sign_extend(pa, bits), sign_extend(pb, bits));
            let max = sign_extend(mask >> 1, bits);
            let min = -max - 1;
            let r = if op == S::SaturatingAdd {
                sa.saturating_add(sb)
            } else {
                sa.saturating_sub(sb)
            };
            (r.clamp(min, max) as u128) & mask
        } else {
            let r = if op == S::SaturatingAdd {
                pa.saturating_add(pb)
            } else {
                pa.saturating_sub(pb)
            };
            r.min(mask)
        };
        return Ok(wrap_simd_int(pattern, bits, signed));
    }
    let mask = bit_mask(bits);
    let pa = as_u128(a, "simd lane")? & mask;
    let pb = as_u128(b, "simd lane")? & mask;
    let pattern = match op {
        S::Add => pa.wrapping_add(pb),
        S::Sub => pa.wrapping_sub(pb),
        S::Mul => pa.wrapping_mul(pb),
        S::Div | S::Rem => {
            if pb == 0 {
                return Err(InterpError::Panic {
                    message: "attempt to divide by zero in a SIMD lane".to_string(),
                    payload: None,
                });
            }
            if signed {
                let sa = sign_extend(pa, bits);
                let sb = sign_extend(pb, bits);
                if op == S::Div {
                    sa.checked_div(sb).map(|v| v as u128).unwrap_or(pa)
                } else {
                    sa.checked_rem(sb).map(|v| v as u128).unwrap_or(0)
                }
            } else if op == S::Div {
                pa / pb
            } else {
                pa % pb
            }
        }
        S::And => pa & pb,
        S::Or => pa | pb,
        S::Xor => pa ^ pb,
        S::Shl => pa << ((pb % u128::from(bits)) as u32),
        S::Shr => {
            let s = (pb % u128::from(bits)) as u32;
            if signed {
                (sign_extend(pa, bits) >> s) as u128
            } else {
                pa >> s
            }
        }
        S::CarrylessMul => {
            let mut result: u128 = 0;
            let mut aa = pa;
            let mut bb = pb;
            while bb != 0 {
                if bb & 1 == 1 {
                    result ^= aa;
                }
                aa = aa.wrapping_shl(1);
                bb >>= 1;
            }
            result
        }
        S::FunnelShl => {
            let s = (pb % u128::from(bits)) as u32;
            if s == 0 {
                pa
            } else {
                (pa << s) | (pb >> (bits - s))
            }
        }
        S::FunnelShr => {
            let s = (pb % u128::from(bits)) as u32;
            if s == 0 {
                pb
            } else {
                (pa << (bits - s)) | (pb >> s)
            }
        }
        _ => {
            return Err(unsupported(
                "this simd operation on integer lanes is not runtime-supported",
            ))
        }
    };
    Ok(wrap_simd_int(pattern, bits, signed))
}

/// Wrap a bit pattern to a SIMD integer lane of `bits` width.
fn wrap_simd_int(pattern: u128, bits: u32, signed: bool) -> Value {
    let pattern = pattern & bit_mask(bits);
    if signed {
        Value::Int(sign_extend(pattern, bits))
    } else {
        Value::Uint(pattern)
    }
}

/// One SIMD comparison of two lanes.
fn simd_compare(
    op: SimdOp,
    a: &Value,
    b: &Value,
    info: (u32, bool, Option<u32>),
) -> InterpResult<bool> {
    use SimdOp as S;
    let (bits, signed, float_bits) = info;
    let ordering = if float_bits.is_some() {
        let x = simd_as_f64(a, "simd lane")?;
        let y = simd_as_f64(b, "simd lane")?;
        return Ok(match op {
            S::Eq => x == y,
            S::Ne => x != y,
            S::Lt => x < y,
            S::Le => x <= y,
            S::Gt => x > y,
            S::Ge => x >= y,
            _ => unreachable!(),
        });
    } else if signed {
        sign_extend(as_u128(a, "simd lane")? & bit_mask(bits), bits)
            .cmp(&sign_extend(as_u128(b, "simd lane")? & bit_mask(bits), bits))
    } else {
        (as_u128(a, "simd lane")? & bit_mask(bits))
            .cmp(&(as_u128(b, "simd lane")? & bit_mask(bits)))
    };
    Ok(match op {
        S::Eq => ordering == std::cmp::Ordering::Equal,
        S::Ne => ordering != std::cmp::Ordering::Equal,
        S::Lt => ordering == std::cmp::Ordering::Less,
        S::Le => ordering != std::cmp::Ordering::Greater,
        S::Gt => ordering == std::cmp::Ordering::Greater,
        S::Ge => ordering != std::cmp::Ordering::Less,
        _ => unreachable!(),
    })
}

/// One element-wise unary SIMD operation on a lane.
fn simd_unop(op: SimdOp, a: &Value, info: (u32, bool, Option<u32>)) -> InterpResult<Value> {
    use SimdOp as S;
    let (bits, signed, float_bits) = info;
    if float_bits.is_some() {
        let x = simd_as_f64(a, "simd lane")?;
        let r = match op {
            S::Neg => -x,
            S::Fabs => x.abs(),
            S::Ceil => x.ceil(),
            S::Floor => x.floor(),
            S::Trunc => x.trunc(),
            S::Round => x.round(),
            S::RoundTiesEven => x.round_ties_even(),
            S::Fsqrt => x.sqrt(),
            S::Fexp => x.exp(),
            S::Fexp2 => x.exp2(),
            S::Flog => x.ln(),
            S::Flog2 => x.log2(),
            S::Flog10 => x.log10(),
            S::Fsin => x.sin(),
            S::Fcos => x.cos(),
            _ => {
                return Err(unsupported(
                    "this simd operation on float lanes is not runtime-supported",
                ))
            }
        };
        return Ok(Value::Float(simd_round_float(r, float_bits)));
    }
    let mask = bit_mask(bits);
    let p = as_u128(a, "simd lane")? & mask;
    let pattern = match op {
        S::Neg => 0u128.wrapping_sub(p),
        S::Bswap => p.swap_bytes() >> (128 - bits),
        S::BitReverse => p.reverse_bits() >> (128 - bits),
        S::Ctlz => u128::from(p.leading_zeros() - (128 - bits)),
        S::Ctpop => u128::from(p.count_ones()),
        S::Cttz => u128::from(if p == 0 { bits } else { p.trailing_zeros() }),
        _ => {
            return Err(unsupported(
                "this simd operation on integer lanes is not runtime-supported",
            ))
        }
    };
    Ok(wrap_simd_int(pattern, bits, signed))
}

/// One lane of a `simd_cast` / `simd_as`: the numeric cast to
/// the element type `dst_elem`, with `as` semantics.
fn simd_cast_lane(
    lane: &Value,
    src_info: (u32, bool, Option<u32>),
    dst_elem: &str,
) -> InterpResult<Value> {
    let (src_bits, src_signed, src_float) = src_info;
    if let Some((dst_bits, dst_signed)) = int_tag(dst_elem) {
        return Ok(match lane {
            Value::Float(f) => {
                if dst_signed {
                    Value::Int(float_to_int(*f, dst_bits))
                } else {
                    Value::Uint(float_to_uint(*f, dst_bits))
                }
            }
            Value::Bool(b) => {
                if dst_signed {
                    Value::Int(i128::from(*b))
                } else {
                    Value::Uint(u128::from(*b))
                }
            }
            _ => {
                let masked = as_u128(lane, "simd lane")? & bit_mask(src_bits);
                // A signed source sign-extends from its own width
                // before the value is wrapped to the destination
                // width (the extension the float destination below
                // applies to an integer lane).
                let pattern = if src_signed {
                    sign_extend(masked, src_bits) as u128
                } else {
                    masked
                };
                wrap_simd_int(pattern, dst_bits, dst_signed)
            }
        });
    }
    match dst_elem {
        "f16" | "f32" | "f64" => {
            let dst_float = match dst_elem {
                "f16" => Some(16),
                "f32" => Some(32),
                _ => Some(64),
            };
            let x = match lane {
                Value::Float(f) => *f,
                Value::Bool(b) => f64::from(*b),
                Value::Int(_) => {
                    sign_extend(as_u128(lane, "simd lane")? & bit_mask(src_bits), src_bits) as f64
                }
                Value::Uint(_) => (as_u128(lane, "simd lane")? & bit_mask(src_bits)) as f64,
                _ => {
                    return Err(InterpError::type_mismatch(
                        "simd cast lane",
                        "numeric lane",
                        lane.kind_name(),
                    ))
                }
            };
            let _ = src_float;
            Ok(Value::Float(simd_round_float(x, dst_float)))
        }
        "bool" => Ok(Value::Bool(simd_lane_active(lane)?)),
        _ => Err(unsupported(format!(
            "simd cast to element type {dst_elem} is not runtime-supported"
        ))),
    }
}

/// A SIMD reduction over `lanes`; `init` is the extra operand
/// of the ordered add/mul forms.
fn simd_reduce(
    op: SimdOp,
    lanes: &[Value],
    init: Option<&Value>,
    info: (u32, bool, Option<u32>),
) -> InterpResult<Value> {
    use SimdOp as S;
    let (bits, signed, float_bits) = info;
    match op {
        S::ReduceAddOrdered | S::ReduceAddUnordered | S::ReduceMulOrdered
        | S::ReduceMulUnordered => {
            let add = matches!(op, S::ReduceAddOrdered | S::ReduceAddUnordered);
            let binop = if add { S::Add } else { S::Mul };
            let mut acc = match init {
                Some(v) => v.clone(),
                None => {
                    if float_bits.is_some() {
                        Value::Float(if add { 0.0 } else { 1.0 })
                    } else if add {
                        wrap_simd_int(0, bits, signed)
                    } else {
                        wrap_simd_int(1, bits, signed)
                    }
                }
            };
            for lane in lanes {
                acc = simd_binop(binop, &acc, lane, info)?;
            }
            Ok(acc)
        }
        S::ReduceMax | S::ReduceMin => {
            let mut iter = lanes.iter();
            let mut acc = iter
                .next()
                .cloned()
                .ok_or_else(|| unsupported("simd reduction of an empty vector"))?;
            for lane in iter {
                let take = if float_bits.is_some() {
                    let (x, y) = (simd_as_f64(&acc, "simd lane")?, simd_as_f64(lane, "simd lane")?);
                    if x.is_nan() || y.is_nan() {
                        acc = Value::Float(f64::NAN);
                        continue;
                    }
                    if op == S::ReduceMax {
                        y > x
                    } else {
                        y < x
                    }
                } else {
                    let ord = if signed {
                        sign_extend(as_u128(lane, "simd lane")? & bit_mask(bits), bits).cmp(
                            &sign_extend(as_u128(&acc, "simd lane")? & bit_mask(bits), bits),
                        )
                    } else {
                        (as_u128(lane, "simd lane")? & bit_mask(bits))
                            .cmp(&(as_u128(&acc, "simd lane")? & bit_mask(bits)))
                    };
                    if op == S::ReduceMax {
                        ord == std::cmp::Ordering::Greater
                    } else {
                        ord == std::cmp::Ordering::Less
                    }
                };
                if take {
                    acc = lane.clone();
                }
            }
            Ok(acc)
        }
        S::ReduceAnd | S::ReduceOr | S::ReduceXor => {
            let mask = bit_mask(bits);
            let mut acc: u128 = if op == S::ReduceAnd { mask } else { 0 };
            for lane in lanes {
                let p = as_u128(lane, "simd lane")? & mask;
                match op {
                    S::ReduceAnd => acc &= p,
                    S::ReduceOr => acc |= p,
                    _ => acc ^= p,
                }
            }
            Ok(wrap_simd_int(acc, bits, signed))
        }
        _ => Err(unsupported("this simd reduction is not runtime-supported")),
    }
}

/// A mask of the low `bits` bits.
fn bit_mask(bits: u32) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

/// Sign-extend the low `bits` bits of `pattern` to an `i128`.
fn sign_extend(pattern: u128, bits: u32) -> i128 {
    if bits >= 128 {
        pattern as i128
    } else {
        let shift = 128 - bits;
        ((pattern << shift) as i128) >> shift
    }
}

/// The field values of an aggregate value: a struct rides
/// as a `Value::Enum` (variant 0), a plain aggregate as a
/// `Value::Tuple`.
fn aggregate_fields(value: &Value) -> Option<&[Value]> {
    match value {
        Value::Enum { fields, .. } => Some(fields),
        Value::Tuple(items) => Some(items),
        _ => None,
    }
}

/// Rebuild an aggregate with the shape of `template` and new
/// field values.
fn aggregate_like(template: &Value, fields: Vec<Value>) -> Value {
    match template {
        Value::Enum { variant, .. } => Value::Enum {
            variant: *variant,
            fields,
        },
        _ => Value::Tuple(fields),
    }
}

fn as_f64(value: &Value, name: &str) -> InterpResult<f64> {
    match value {
        Value::Float(f) => Ok(*f),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "float",
            other.kind_name(),
        )),
    }
}

/// The 64-bit pattern a `LARGE_INTEGER` union value holds.
/// A union value is an `Enum` holding one live field (the
/// last written; the other slots are `Uninit`): here either
/// the `QuadPart` scalar or the two-`u32` struct view, whose
/// parts combine low-then-high. A bare scalar (the shape a
/// constant aggregate can arrive in) reads as itself.
fn union_u64_bits(value: &Value) -> Option<u64> {
    match value {
        Value::Uint(v) => Some(*v as u64),
        Value::Int(v) => Some(*v as u64),
        Value::Enum { fields, .. } => {
            let live: Vec<&Value> = fields
                .iter()
                .filter(|f| **f != Value::Uninit)
                .collect();
            match live.as_slice() {
                [only] => match only {
                    Value::Enum { fields: parts, .. } if parts.len() == 2 => {
                        let lo = match &parts[0] {
                            Value::Uint(v) => *v as u64,
                            _ => return None,
                        };
                        let hi = match &parts[1] {
                            Value::Uint(v) => *v as u64,
                            _ => return None,
                        };
                        Some(lo | (hi << 32))
                    }
                    other => union_u64_bits(other),
                },
                _ => None,
            }
        }
        Value::Tuple(items) if items.len() == 1 => union_u64_bits(&items[0]),
        _ => None,
    }
}

/// A union value with `template`'s shape holding `bits` as
/// its 64-bit pattern: the live slot of the template (the
/// scalar, or the struct view as its two 32-bit parts) is
/// rewritten, every other slot stays `Uninit`, so the guest's
/// union field reads reinterpret the new pattern exactly as
/// they did the old one.
fn union_with_u64_bits(template: &Value, bits: u64) -> Value {
    match template {
        Value::Enum { variant, fields } => {
            let live_idx = fields.iter().position(|f| *f != Value::Uninit);
            match live_idx {
                Some(idx) => {
                    let mut out: Vec<Value> =
                        fields.iter().map(|_| Value::Uninit).collect();
                    out[idx] = match &fields[idx] {
                        Value::Enum { variant: v2, fields: parts } if parts.len() == 2 => {
                            Value::Enum {
                                variant: *v2,
                                fields: vec![
                                    Value::Uint((bits & 0xFFFF_FFFF) as u128),
                                    Value::Uint((bits >> 32) as u128),
                                ],
                            }
                        }
                        _ => Value::Uint(bits as u128),
                    };
                    Value::Enum {
                        variant: *variant,
                        fields: out,
                    }
                }
                None => Value::Uint(bits as u128),
            }
        }
        Value::Tuple(items) if items.len() == 1 => {
            Value::Tuple(vec![union_with_u64_bits(&items[0], bits)])
        }
        _ => Value::Uint(bits as u128),
    }
}

fn as_i128(value: &Value, name: &str) -> InterpResult<i128> {
    match value {
        Value::Int(v) => Ok(*v),
        Value::Uint(v) => Ok(*v as i128),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "integer",
            other.kind_name(),
        )),
    }
}

fn as_u128(value: &Value, name: &str) -> InterpResult<u128> {
    match value {
        Value::Uint(v) => Ok(*v),
        Value::Int(v) => Ok(*v as u128),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "integer",
            other.kind_name(),
        )),
    }
}

/// The all-zero-bits value of the type `tag` names, for
/// `mem::zeroed`: scalars, and tuples/arrays built from scalars,
/// within the materialisation budget (see [`MAX_FILL_LEAVES`]):
/// a large array of scalars is a big tree of zero leaves, which
/// the budget admits up to millions of elements. Anything else
/// (an ADT whose zero bit pattern need not be a valid value, a
/// reference, ...) stays unsupported.
fn zeroed_value_for_tag(tag: &str) -> Option<Value> {
    if tag_leaf_count(tag)? > MAX_FILL_LEAVES {
        return None;
    }
    let tag = tag.trim();
    Some(match tag {
        "bool" => Value::Bool(false),
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => Value::Int(0),
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "char" => Value::Uint(0),
        "f32" | "f64" => Value::Float(0.0),
        "f128" => Value::Float128(0),
        "()" => Value::Unit,
        _ if tag.starts_with('(') && tag.ends_with(')') => {
            let inner = &tag[1..tag.len() - 1];
            let mut vals = Vec::new();
            if !inner.trim().is_empty() {
                for part in split_top_level(inner) {
                    vals.push(zeroed_value_for_tag(part)?);
                }
            }
            Value::Tuple(vals)
        }
        _ if tag.starts_with('[') && tag.ends_with(']') => {
            let inner = &tag[1..tag.len() - 1];
            let (elem, n) = inner.rsplit_once(';')?;
            let n: usize = n.trim().parse().ok()?;
            let elem = zeroed_value_for_tag(elem)?;
            Value::Tuple(vec![elem; n])
        }
        _ => return None,
    })
}


/// The first generic argument of a rendered generic type tag
/// (`std::cell::RefCell<std::vec::Vec<u32>>` -> the `Vec` tag), or
/// `None` when the tag does not start with `prefix`.
fn generic_first_arg<'a>(tag: &'a str, prefix: &str) -> Option<&'a str> {
    let inner = tag.strip_prefix(prefix)?.strip_suffix('>')?;
    split_top_level(inner).into_iter().next()
}

/// Two fresh SipHash keys for a new `std::hash::RandomState`,
/// drawn the way the modelled time builtins draw from the host:
/// the wall clock's nanoseconds, a per-process counter and the
/// process id, mixed through splitmix64. Successive calls in a
/// run differ, as successive `RandomState`s do natively.
fn fresh_hash_keys() -> (u64, u64) {
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut state = nanos
        ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ u64::from(std::process::id()).rotate_left(32);
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    (next(), next())
}

fn default_value_for_tag(tag: &str) -> Option<Value> {
    let tag = tag.trim();
    Some(match tag {
        "bool" => Value::Bool(false),
        "f32" | "f64" => Value::Float(0.0),
        "&str" => Value::Str(String::new()),
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => Value::Int(0),
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "char" => Value::Uint(0),
        "()" => Value::Unit,
        "std::string::String" => Value::Str(String::new()),
        _ if tag.starts_with("std::vec::Vec<") => Value::Enum {
            variant: 0,
            fields: Vec::new(),
        },
        // The cell wrappers whose `Default` impl wraps the inner
        // type's default (`RefCell::new`, `Cell::new` of
        // `T::default()`): the shapes are the ones the
        // corresponding `new` builtins build. The allocating
        // wrappers (`Box`, `Rc`, `Arc`) are handled in
        // `Interpreter::default_value`, whose shapes only the
        // interpreter can mint.
        _ if generic_first_arg(tag, "std::cell::RefCell<").is_some() => {
            let inner =
                generic_first_arg(tag, "std::cell::RefCell<").expect("guard checked");
            refcell_value(default_value_for_tag(inner)?)
        }
        _ if generic_first_arg(tag, "std::cell::Cell<").is_some() => {
            let inner = generic_first_arg(tag, "std::cell::Cell<").expect("guard checked");
            cell_value(default_value_for_tag(inner)?)
        }
        _ if tag.starts_with("std::marker::PhantomData<") => Value::Enum {
            variant: 0,
            fields: Vec::new(),
        },
        _ if tag.contains("option::Option<") => Value::Enum {
            variant: 0,
            fields: Vec::new(),
        },
        _ if tag.starts_with('(') && tag.ends_with(')') => {
            let inner = &tag[1..tag.len() - 1];
            let mut vals = Vec::new();
            for part in split_top_level(inner) {
                vals.push(default_value_for_tag(part)?);
            }
            Value::Tuple(vals)
        }
        _ if tag.starts_with('[') && tag.ends_with(']') => {
            // `[T; N]`: an array of the element default.
            let inner = &tag[1..tag.len() - 1];
            let (elem, n) = inner.rsplit_once(';')?;
            let n: usize = n.trim().parse().ok()?;
            if n > 4096 {
                return None;
            }
            let elem_default = default_value_for_tag(elem.trim())?;
            Value::Tuple(vec![elem_default; n])
        }
        _ => return None,
    })
}

/// The `(size, align)` of a rendered type tag on this platform
/// (x86_64), for the types whose layout is fixed and known.
impl<'a> Interpreter<'a> {
    /// The `(size, align)` of a rendered type tag: the static table
    /// for the primitive and abstract-model types, then the
    /// program's ADT layout table (compiler-computed layouts of the
    /// monomorphic ADT instances, recorded by the driver) for ADT
    /// tags. `None` stays an explicit unsupported error at the call
    /// site.
    /// `Default::default()` for a rendered type tag: the pure
    /// shapes come from `default_value_for_tag`; the allocating
    /// wrappers wrap the inner type's default in the value their
    /// `new` builtin builds (an allocation for `Box`, a counted
    /// handle for `Rc` / `Arc`), which only the interpreter can
    /// mint.
    fn default_value(&self, tag: &str) -> Option<Value> {
        let tag = tag.trim();
        if let Some(inner) = generic_first_arg(tag, "std::boxed::Box<") {
            let pointee = if inner.starts_with('[') && inner.ends_with(']') && !inner.contains(';')
            {
                // `Box<[T]>`: its `Default` is an empty boxed
                // slice -- the pointee an empty
                // `Vec::into_boxed_slice` builds.
                Value::Tuple(Vec::new())
            } else {
                self.default_value(inner)?
            };
            return Some(self.new_box(pointee));
        }
        if let Some(inner) = generic_first_arg(tag, "std::rc::Rc<")
            .or_else(|| generic_first_arg(tag, "std::sync::Arc<"))
        {
            let pointee = self.default_value(inner)?;
            return Some(self.rc_new_handle(pointee));
        }
        if tag == "std::hash::RandomState" {
            // `RandomState::default()` is `RandomState::new()`:
            // the struct value holding two fresh hash keys, the
            // shape the collected `new` body builds.
            let (k0, k1) = fresh_hash_keys();
            return Some(Value::Enum {
                variant: 0,
                fields: vec![Value::Uint(u128::from(k0)), Value::Uint(u128::from(k1))],
            });
        }
        default_value_for_tag(tag)
    }

    pub(crate) fn layout_of(&self, tag: &str) -> Option<(u128, u128)> {
        let tag = tag.trim();
        if let Some(layout) = type_layout(tag) {
            return Some(layout);
        }
        // Arrays: the layout is fixed by the element layout alone.
        // Elements sit at a stride of the element's size (a type's
        // size is always a multiple of its alignment, for the
        // primitive table, for the compiler-recorded ADT layouts
        // and for the composites computed here alike), so an array
        // is `n` element sizes at the element's alignment.
        if tag.starts_with('[') && tag.ends_with(']') {
            let inner = &tag[1..tag.len() - 1];
            if let Some((elem, n)) = inner.rsplit_once(';') {
                let n: u128 = n.trim().parse().ok()?;
                let (elem_size, elem_align) = self.layout_of(elem.trim())?;
                return Some((elem_size.checked_mul(n)?, elem_align));
            }
            // A slice is unsized: it has no static layout.
            return None;
        }
        // Tuples: rustc's `repr(Rust)` aggregate layout orders the
        // fields by decreasing alignment before assigning offsets.
        // Because every field's size is a multiple of its own
        // alignment, no padding is inserted between fields in that
        // order, so the size is the sum of the field sizes rounded
        // up to the largest field alignment -- a function of the
        // field layouts alone, independent of the written order.
        if tag.len() > 2 && tag.starts_with('(') && tag.ends_with(')') {
            let inner = &tag[1..tag.len() - 1];
            let mut size: u128 = 0;
            let mut align: u128 = 1;
            for part in split_top_level(inner) {
                let (field_size, field_align) = self.layout_of(part)?;
                size = size.checked_add(field_size)?;
                align = align.max(field_align);
            }
            let size = size.checked_add(align - 1)? & !(align - 1);
            return Some((size, align));
        }
        self.program()
            .adt_layouts
            .get(tag)
            .map(|&(size, align)| (u128::from(size), u128::from(align)))
    }
}

fn type_layout(tag: &str) -> Option<(u128, u128)> {
    let tag = tag.trim();
    Some(match tag {
        "bool" | "u8" | "i8" => (1, 1),
        "u16" | "i16" => (2, 2),
        "u32" | "i32" | "f32" | "char" => (4, 4),
        "u64" | "i64" | "f64" | "usize" | "isize" => (8, 8),
        "u128" | "i128" => (16, 16),
        "()" => (0, 1),
        "std::string::String" => (24, 8),
        _ if tag.starts_with("std::vec::Vec<") => (24, 8),
        _ if tag.starts_with("std::boxed::Box<") => (8, 8),
        _ if tag.starts_with('&') => {
            let inner = tag[1..].trim_start();
            let inner = inner.strip_prefix("mut ").unwrap_or(inner);
            if inner == "str" || inner.starts_with('[') {
                (16, 8)
            } else {
                (8, 8)
            }
        }
        _ if tag.starts_with("*const ") || tag.starts_with("*mut ") => {
            let inner = tag.split_once(' ')?.1.trim();
            if inner == "str" || inner.starts_with('[') {
                (16, 8)
            } else {
                (8, 8)
            }
        }
        _ => return None,
    })
}

/// Whether values of a rendered type tag need drop glue, for the
/// types with a known answer: plain data never does; the standard
/// owning containers do.
fn needs_drop_for_tag(tag: &str) -> Option<bool> {
    let tag = tag.trim();
    if is_integer_tag(tag)
        || matches!(tag, "bool" | "char" | "()" | "str")
        || tag.starts_with('&')
        || tag.starts_with("*const ")
        || tag.starts_with("*mut ")
    {
        return Some(false);
    }
    const OWNING: &[&str] = &[
        "std::string::String",
        "std::vec::Vec<",
        "std::boxed::Box<",
        "std::collections::HashMap<",
        "std::collections::BTreeMap<",
        "std::collections::VecDeque<",
        "std::cell::RefCell<",
        "std::sync::Mutex<",
        "std::sync::RwLock<",
        "std::sync::Arc<",
        "std::rc::Rc<",
        "std::ffi::CString",
    ];
    if OWNING.iter().any(|prefix| tag.starts_with(prefix)) {
        return Some(true);
    }
    None
}

// ---------------------------------------------------------------------------
// The fmt engine
//
// Mirrors Rust 1.100.0 `core::fmt`: `Arguments::write` walks the
// encoded template; each placeholder pulls one argument (an explicit
// index replaces the running index, which still advances afterwards),
// and the argument is formatted through its trait with the decoded
// `FormattingOptions`. Padding mirrors `Formatter::pad` (strings:
// default align Left, precision truncates chars) and
// `Formatter::pad_integral` (numbers: sign and alternate prefix are
// counted in the width, zero-padding is sign-aware, default align
// Right, and a precision — minimum digits — suppresses zero-padding).
// ---------------------------------------------------------------------------

/// Flag word layout (`core::fmt::FormattingOptions` private flags).
const SIGN_PLUS: u32 = 1 << 21;
const ALTERNATE: u32 = 1 << 23;
const SIGN_AWARE_ZERO_PAD: u32 = 1 << 24;
const DEBUG_LOWER_HEX: u32 = 1 << 25;
const DEBUG_UPPER_HEX: u32 = 1 << 26;
const WIDTH_FLAG: u32 = 1 << 27;
const PRECISION_FLAG: u32 = 1 << 28;
const ALIGN_MASK: u32 = 0b11 << 29;
const ALIGN_LEFT: u32 = 0;
const ALIGN_RIGHT: u32 = 1 << 29;
const ALIGN_CENTER: u32 = 2 << 29;
const ALIGN_UNKNOWN: u32 = 3 << 29;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Align {
    Left,
    Right,
    Center,
}

#[derive(Debug, Clone, Copy)]
struct FmtOptions {
    flags: u32,
    width: u16,
    precision: u16,
}

impl FmtOptions {
    /// core's default: fill `' '`, alignment unknown.
    fn new() -> Self {
        FmtOptions {
            flags: (' ' as u32) | ALIGN_UNKNOWN,
            width: 0,
            precision: 0,
        }
    }
    fn width(&self) -> Option<u16> {
        if self.flags & WIDTH_FLAG != 0 {
            Some(self.width)
        } else {
            None
        }
    }
    fn precision(&self) -> Option<u16> {
        if self.flags & PRECISION_FLAG != 0 {
            Some(self.precision)
        } else {
            None
        }
    }
    fn fill(&self) -> char {
        char::from_u32(self.flags & ((1 << 21) - 1)).unwrap_or(' ')
    }
    fn align(&self) -> Option<Align> {
        match self.flags & ALIGN_MASK {
            ALIGN_LEFT => Some(Align::Left),
            ALIGN_RIGHT => Some(Align::Right),
            ALIGN_CENTER => Some(Align::Center),
            _ => None,
        }
    }
    fn sign_plus(&self) -> bool {
        self.flags & SIGN_PLUS != 0
    }
    fn alternate(&self) -> bool {
        self.flags & ALTERNATE != 0
    }
    fn zero_pad(&self) -> bool {
        self.flags & SIGN_AWARE_ZERO_PAD != 0
    }
}

impl FmtCapture {
    /// The placeholder options this capture was seeded with,
    /// in the fmt engine's terms.
    fn options(&self) -> FmtOptions {
        FmtOptions {
            flags: self.flags,
            width: self.width,
            precision: self.precision,
        }
    }
}

fn template_error(what: &str) -> InterpError {
    InterpError::type_mismatch("fmt pieces template", "well-formed template", what)
}

impl<'a> Interpreter<'a> {
    /// Format a built `fmt::Arguments` value to a string.
    pub(crate) fn format_arguments(
        &mut self,
        frames: &mut Vec<Frame>,
        args: &FmtArguments,
        indent: usize,
    ) -> InterpResult<String> {
        match args {
            FmtArguments::Plain(s) => Ok(s.clone()),
            FmtArguments::Template { template, args } => {
                self.format_template(frames, template, args, indent)
            }
        }
    }

    fn format_template(
        &mut self,
        frames: &mut Vec<Frame>,
        template: &[u8],
        args: &[Value],
        indent: usize,
    ) -> InterpResult<String> {
        let mut out = String::new();
        let mut arg_index = 0usize;
        let mut i = 0usize;
        while i < template.len() {
            let b = template[i];
            if b == 0 {
                break;
            }
            if b < 0x80 {
                // Short literal piece: one length byte, then the bytes.
                let len = b as usize;
                out.push_str(template_piece(template, i + 1, len)?);
                i += 1 + len;
            } else if b == 0x80 {
                // Long literal piece: 0x80, u16 LE length, then bytes.
                let len = read_u16(template, i + 1)? as usize;
                out.push_str(template_piece(template, i + 3, len)?);
                i += 3 + len;
            } else if b == 0xC0 {
                // Placeholder with entirely default options.
                let arg = template_arg(args, arg_index)?;
                out.push_str(&self.format_arg(frames, arg, &FmtOptions::new(), indent)?);
                arg_index += 1;
                i += 1;
            } else if b & 0xC0 == 0xC0 {
                // General placeholder: field-presence bits in the
                // first byte, then fields in the order flags, width,
                // precision, arg_index.
                let mut options = FmtOptions::new();
                let mut j = i + 1;
                if b & 0x01 != 0 {
                    options.flags = read_u32(template, j)?;
                    j += 4;
                }
                if b & 0x02 != 0 {
                    let raw = read_u16(template, j)?;
                    j += 2;
                    options.width = if b & 0x10 != 0 {
                        self.count_arg(args, raw as usize)?
                    } else {
                        raw
                    };
                    options.flags |= WIDTH_FLAG;
                }
                if b & 0x04 != 0 {
                    let raw = read_u16(template, j)?;
                    j += 2;
                    options.precision = if b & 0x20 != 0 {
                        self.count_arg(args, raw as usize)?
                    } else {
                        raw
                    };
                    options.flags |= PRECISION_FLAG;
                }
                let mut this_arg = arg_index;
                if b & 0x08 != 0 {
                    this_arg = read_u16(template, j)? as usize;
                    j += 2;
                }
                let arg = template_arg(args, this_arg)?;
                out.push_str(&self.format_arg(frames, arg, &options, indent)?);
                arg_index += 1;
                i = j;
            } else {
                return Err(template_error("unexpected byte at a part boundary"));
            }
        }
        Ok(out)
    }

    /// An indirect width/precision: the argument at `idx` must be a
    /// count argument (`Argument::from_usize`).
    fn count_arg(&self, args: &[Value], idx: usize) -> InterpResult<u16> {
        match args.get(idx) {
            Some(Value::FmtCount(n)) => Ok(*n),
            _ => Err(template_error(
                "indirect width/precision does not name a count argument",
            )),
        }
    }

    fn format_arg(
        &mut self,
        frames: &mut Vec<Frame>,
        arg: &Value,
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<String> {
        // A body that builds `core::fmt::rt::Argument` itself
        // (an inlined `format_args!` expansion inside a library
        // body) hands over the raw aggregate instead of the
        // modelled value the `Argument::new_*` builtins produce;
        // both denote the same argument.
        if !matches!(arg, Value::FmtArg { .. }) {
            if let Some(coerced) = coerce_fmt_arg(arg) {
                return self.format_arg(frames, &coerced, options, indent);
            }
        }
        match arg {
            Value::FmtArg {
                target,
                trait_,
                ty,
            } => {
                // A reference to a fat pointer whose base denotes
                // an element of a container: the stored fat pointer
                // denotes the slice starting at that element, so
                // format it as the slice it is rather than resolving
                // through to the element.
                if let Ok(resolved) = self.deref_value(frames, target) {
                    if let Ok(stored) = self.read_resolved(frames, &resolved) {
                        if let Value::FatRef { base, .. } = &stored {
                            if let Ok(bd) = self.deref_value(frames, base) {
                                if target_ends_at_element(&bd) {
                                    return self.format_value(
                                        frames, &stored, *trait_, ty, options, 0,
                                    );
                                }
                            }
                        }
                    }
                }
                // A fat pointer whose base denotes an element of a
                // container denotes the slice starting at that
                // element: resolving it would read the element
                // itself and lose the slice shape, so hand the fat
                // pointer itself to the formatter.
                if let Value::FatRef { base, .. } = &**target {
                    if let Ok(resolved) = self.deref_value(frames, base) {
                        if target_ends_at_element(&resolved) {
                            return self.format_value(
                                frames, target, *trait_, ty, options, 0,
                            );
                        }
                    }
                }
                // A scalar argument read through a
                // byte-displaced pointer: the argument's type
                // tag is the load's width, and the byte view of
                // the base value's storage composes it (the
                // resolved read has no byte-level view of that
                // storage).
                let pointee = match self.byte_view_fmt_arg(frames, target, ty) {
                    Some(view) => view?,
                    None => self.resolve_value(frames, target)?,
                };
                self.format_value(frames, &pointee, *trait_, ty, options, indent)
            }
            Value::FmtCount(_) => Err(template_error(
                "count argument used as a formatting argument",
            )),
            other => self.format_value(frames, other, FmtTrait::Display, "", options, indent),
        }
    }

    /// The captured formatter a `Formatter` argument value
    /// names: the handle itself, or the handle behind the
    /// reference a guest body passes.
    fn formatter_id_of(&self, frames: &[Frame], value: &Value) -> InterpResult<u64> {
        match self.resolve_value(frames, value)? {
            Value::Formatter(id) => Ok(id),
            other => Err(InterpError::type_mismatch(
                "Formatter method",
                "a captured Formatter",
                other.kind_name(),
            )),
        }
    }

    /// The builder a builder-method argument value names: the
    /// handle itself, or the handle behind the reference.
    fn builder_id_of(&self, frames: &[Frame], value: &Value) -> InterpResult<u64> {
        match self.resolve_value(frames, value)? {
            Value::FmtBuilder(id) => Ok(id),
            other => Err(InterpError::type_mismatch(
                "fmt builder method",
                "a fmt builder",
                other.kind_name(),
            )),
        }
    }

    /// The current state of a captured formatter.
    fn capture_of(&self, id: u64) -> InterpResult<FmtCapture> {
        self.formatter_snapshot(id).ok_or_else(|| {
            unsupported("a Formatter method on a capture that already finished".to_string())
        })
    }

    /// Append text to a captured formatter's buffer.
    fn formatter_push(&self, id: u64, text: &str) {
        self.with_formatter(id, |capture| capture.buffer.push_str(text));
    }

    /// Run a guest `fmt` implementation body for `value`, if
    /// the program collected one: the body executes against a
    /// captured `Formatter` seeded with the placeholder's
    /// options, and the captured text is the formatted output.
    /// Returns `None` when no collected body serves this type:
    /// core/std/alloc types and the shapes the structural
    /// engine models keep their structural formatting (their
    /// impls are core's, written against a real `Formatter`
    /// value layout the value model does not reproduce).
    fn format_with_body(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        trait_: FmtTrait,
        ty: &str,
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<Option<String>> {
        if matches!(trait_, FmtTrait::DebugNoop | FmtTrait::Pointer) {
            return Ok(None);
        }
        if !matches!(value, Value::Enum { .. }) {
            return Ok(None);
        }
        let Some(call) = self.fmt_body_call(ty, trait_) else {
            return Ok(None);
        };
        let id = self.new_formatter(FmtCapture {
            buffer: String::new(),
            flags: options.flags,
            width: options.width,
            precision: options.precision,
            indent,
            self_tag: pointee_tag(ty).to_string(),
        });
        let formatter = Value::Ref(RefTarget::value(Value::Formatter(id)));
        let receiver = Value::Ref(RefTarget::value(value.clone()));
        let result = self.call_sync(frames, &call, vec![receiver, formatter]);
        let capture = self.take_formatter(id);
        let result = result?;
        match result {
            Value::Enum { variant: 0, .. } => {}
            _ => {
                return Err(unsupported(format!(
                    "the {trait_:?} implementation for {ty} returned an error"
                )))
            }
        }
        Ok(Some(capture.map(|c| c.buffer).unwrap_or_default()))
    }

    /// The program name of the `fmt` method serving `ty`
    /// through `trait_`, when a body for it was collected. The
    /// recorded instance name is tried first (the impl-def
    /// spelling, for bodies the item loop and the reachability
    /// collection recorded), then the trait-name spelling the
    /// instance-demand pass records (`Trait::fmt [<impl>::fmt]`
    /// for instances it converted).
    fn fmt_body_call(&self, ty: &str, trait_: FmtTrait) -> Option<String> {
        let trait_name = match trait_ {
            FmtTrait::Display => "Display",
            FmtTrait::Debug => "Debug",
            FmtTrait::Octal => "Octal",
            FmtTrait::LowerHex => "LowerHex",
            FmtTrait::UpperHex => "UpperHex",
            FmtTrait::Binary => "Binary",
            FmtTrait::LowerExp => "LowerExp",
            FmtTrait::UpperExp => "UpperExp",
            FmtTrait::DebugNoop | FmtTrait::Pointer => return None,
        };
        // The value the dispatch runs on is the pointee value
        // (references are unwrapped before this point), so the
        // only `Self` whose body can execute on it is the
        // pointee type's -- a reference type's own impl (core's
        // `&T` forwarding impls included) takes the reference
        // itself as its value and must not be selected here.
        let plain = ty.trim();
        let selves: Vec<&str> = vec![pointee_tag(plain)];
        for self_ty in selves.iter().copied() {
            if !guest_fmt_tag(self_ty) {
                continue;
            }
            for spelling in ["std", "core"] {
                let direct =
                    format!("<{self_ty} as {spelling}::fmt::{trait_name}>::fmt");
                if self.program().get(&direct).is_some() {
                    return Some(direct);
                }
                for inner in ["std", "core"] {
                    let suffixed = format!(
                        "{spelling}::fmt::{trait_name}::fmt [<{self_ty} as {inner}::fmt::{trait_name}>::fmt]"
                    );
                    if self.program().get(&suffixed).is_some() {
                        return Some(suffixed);
                    }
                }
            }
        }
        // The compiler's instance name and the converter's
        // type-tag rendering differ in const generic
        // arguments (the tag rendering carries type arguments
        // only), so an exact candidate can miss an instance
        // the driver recorded under its instance name. Scan
        // the program's instance names for one whose Self
        // type is this tag modulo const generic arguments.
        let normalized: Vec<String> = selves
            .iter()
            .filter(|s| guest_fmt_tag(s))
            .map(|s| normalize_fmt_tag(s))
            .collect();
        if !normalized.is_empty() {
            for key in self.program().functions.keys() {
                let Some(rest) = key.strip_prefix('<') else {
                    continue;
                };
                let Some(rest) = rest.strip_suffix(">::fmt") else {
                    continue;
                };
                for spelling in ["std", "core"] {
                    let suffix = format!(" as {spelling}::fmt::{trait_name}");
                    if let Some(self_part) = rest.strip_suffix(&suffix) {
                        if normalized.iter().any(|n| *n == normalize_fmt_tag(self_part)) {
                            return Some(key.clone());
                        }
                    }
                }
            }
        }
        None
    }

    /// The recorded field type tags of the variant of the
    /// capture's `self_tag` type that the finisher name
    /// identifies (a struct's type name, an enum variant's
    /// name), in declaration order; `None` when no recorded
    /// shape fits, in which case the caller formats fields
    /// with no tag, as a `dyn` dispatch would. The shape is
    /// the only place the full field types survive: the
    /// finisher's `&dyn Debug` field values carry their type
    /// by definition name only.
    fn capture_field_tags(&self, id: u64, name: &str) -> Option<Vec<String>> {
        let capture = self.capture_of(id).ok()?;
        let shape = self
            .program()
            .adt_debug_shapes
            .get(pointee_tag(&capture.self_tag))?;
        let variant = shape
            .variants
            .iter()
            .find(|v| v.name == name)
            .or_else(|| shape.variants.iter().next().filter(|_| shape.variants.len() == 1))?;
        Some(variant.fields.iter().map(|f| f.ty.clone()).collect())
    }

    /// Format one finisher/builder field: a `&dyn Debug`
    /// value whose full type tag `tag` comes from the
    /// containing type's recorded shape. The dyn value is
    /// unwrapped here (with the same receiver shaping the
    /// `dyn` formatting path applies) rather than left to
    /// that path, because the recorded concrete type names
    /// an ADT by its definition name only -- the element
    /// types the structural forms parse out of the tag
    /// (`Vec<T>`'s `T`, a box's pointee) would be lost.
    /// A `Box` / `Rc` / `Arc` field formats its pointee
    /// with the element tag, as the structural `Box` form
    /// does.
    fn format_finisher_field(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        tag: &str,
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<String> {
        if let Value::DynRef { base, .. } = value {
            let resolved = self.resolve_value(frames, base)?;
            let shaped = if let Some(id) = rc_handle_id(&resolved) {
                Value::Ref(RefTarget::Alloc {
                    id,
                    projection: vec![RefProj::Field(2)],
                })
            } else if let Value::Enum { fields, .. } = &resolved {
                if fields.len() == 1
                    && (matches!(base.as_ref(), Value::Enum { .. })
                        || matches!(
                            tag_base_name(pointee_tag(tag)),
                            "Box" | "Rc" | "Arc"
                        ))
                {
                    fields[0].clone()
                } else {
                    resolved
                }
            } else {
                resolved
            };
            let tag = match tag_base_name(pointee_tag(tag)) {
                "Box" | "Rc" | "Arc" => {
                    first_generic_arg(pointee_tag(tag)).unwrap_or(tag)
                }
                _ => tag,
            };
            return self.format_value(frames, &shaped, FmtTrait::Debug, tag, options, indent);
        }
        self.format_value(frames, value, FmtTrait::Debug, tag, options, indent)
    }

    /// `Formatter::debug_struct_fieldN_finish`: render the
    /// struct from its name and the interleaved field-name /
    /// `&dyn Debug` arguments into the capture.
    fn exec_debug_struct_finish(
        &mut self,
        frames: &mut Vec<Frame>,
        vals: &[Value],
        count: usize,
    ) -> InterpResult<Value> {
        let id = self.formatter_id_of(frames, &vals[0])?;
        let capture = self.capture_of(id)?;
        let options = capture.options();
        let name = self.resolve_string(frames, &vals[1])?;
        let field_indent = if options.alternate() {
            capture.indent + 4
        } else {
            capture.indent
        };
        let tags = self.capture_field_tags(id, &name);
        let mut fields = Vec::with_capacity(count);
        for i in 0..count {
            let field_name = self.resolve_string(frames, &vals[2 + 2 * i])?;
            let tag = tags
                .as_ref()
                .and_then(|tags| tags.get(i))
                .map(String::as_str)
                .unwrap_or("");
            let text = self.format_finisher_field(
                frames,
                &vals[3 + 2 * i],
                tag,
                &options,
                field_indent,
            )?;
            fields.push((field_name, text));
        }
        let text = render_debug_struct(&name, &fields, &options, capture.indent);
        self.formatter_push(id, &text);
        Ok(fmt_result_ok())
    }

    /// `Formatter::debug_struct_fields_finish`: the same
    /// rendering with parallel field-name / value slices.
    fn exec_debug_struct_fields_finish(
        &mut self,
        frames: &mut Vec<Frame>,
        vals: &[Value],
    ) -> InterpResult<Value> {
        let id = self.formatter_id_of(frames, &vals[0])?;
        let capture = self.capture_of(id)?;
        let options = capture.options();
        let name = self.resolve_string(frames, &vals[1])?;
        let (names, _) = self.slice_elements(frames, &vals[2])?;
        let (values, _) = self.slice_elements(frames, &vals[3])?;
        if names.len() != values.len() {
            return Err(InterpError::type_mismatch(
                "debug_struct_fields_finish",
                "field-name and value slices of equal length",
                "slices of different lengths",
            ));
        }
        let field_indent = if options.alternate() {
            capture.indent + 4
        } else {
            capture.indent
        };
        let tags = self.capture_field_tags(id, &name);
        let mut fields = Vec::with_capacity(names.len());
        for (i, (name_value, field_value)) in names.iter().zip(values.iter()).enumerate() {
            let field_name = self.resolve_string(frames, name_value)?;
            let tag = tags
                .as_ref()
                .and_then(|tags| tags.get(i))
                .map(String::as_str)
                .unwrap_or("");
            let text = self.format_finisher_field(
                frames,
                field_value,
                tag,
                &options,
                field_indent,
            )?;
            fields.push((field_name, text));
        }
        let text = render_debug_struct(&name, &fields, &options, capture.indent);
        self.formatter_push(id, &text);
        Ok(fmt_result_ok())
    }

    /// `Formatter::debug_tuple_fieldN_finish`: render the
    /// tuple struct from its name and `&dyn Debug` fields.
    fn exec_debug_tuple_finish(
        &mut self,
        frames: &mut Vec<Frame>,
        vals: &[Value],
        count: usize,
    ) -> InterpResult<Value> {
        let id = self.formatter_id_of(frames, &vals[0])?;
        let capture = self.capture_of(id)?;
        let options = capture.options();
        let name = self.resolve_string(frames, &vals[1])?;
        let field_indent = if options.alternate() {
            capture.indent + 4
        } else {
            capture.indent
        };
        let tags = self.capture_field_tags(id, &name);
        let mut fields = Vec::with_capacity(count);
        for i in 0..count {
            let tag = tags
                .as_ref()
                .and_then(|tags| tags.get(i))
                .map(String::as_str)
                .unwrap_or("");
            let text = self.format_finisher_field(
                frames,
                &vals[2 + i],
                tag,
                &options,
                field_indent,
            )?;
            fields.push(text);
        }
        let text = render_debug_tuple(&name, &fields, &options, capture.indent);
        self.formatter_push(id, &text);
        Ok(fmt_result_ok())
    }

    /// `Formatter::debug_tuple_fields_finish`: the tuple
    /// rendering with the fields as one `&[&dyn Debug]` slice
    /// (the form derived tuple structs with more than five
    /// fields use).
    fn exec_debug_tuple_fields_finish(
        &mut self,
        frames: &mut Vec<Frame>,
        vals: &[Value],
    ) -> InterpResult<Value> {
        let id = self.formatter_id_of(frames, &vals[0])?;
        let capture = self.capture_of(id)?;
        let options = capture.options();
        let name = self.resolve_string(frames, &vals[1])?;
        let (values, _) = self.slice_elements(frames, &vals[2])?;
        let field_indent = if options.alternate() {
            capture.indent + 4
        } else {
            capture.indent
        };
        let tags = self.capture_field_tags(id, &name);
        let mut fields = Vec::with_capacity(values.len());
        for (i, field_value) in values.iter().enumerate() {
            let tag = tags
                .as_ref()
                .and_then(|tags| tags.get(i))
                .map(String::as_str)
                .unwrap_or("");
            let text = self.format_finisher_field(
                frames,
                field_value,
                tag,
                &options,
                field_indent,
            )?;
            fields.push(text);
        }
        let text = render_debug_tuple(&name, &fields, &options, capture.indent);
        self.formatter_push(id, &text);
        Ok(fmt_result_ok())
    }

    /// Append one entry's rendered text to a builder, with
    /// the separator and pretty indentation core's builders
    /// produce, and mark the builder non-empty. A builder
    /// whose writes already failed writes nothing (core's
    /// `result` short-circuit).
    fn builder_write_entry(&mut self, builder: u64, text: &str) -> InterpResult<()> {
        let state = self
            .fmt_builder_snapshot(builder)
            .ok_or_else(|| unsupported("a fmt builder on a finished capture".to_string()))?;
        if !state.result_ok {
            return Ok(());
        }
        let capture = self.capture_of(state.formatter)?;
        let options = capture.options();
        let indent = capture.indent;
        let mut out = String::new();
        match state.kind {
            FmtBuilderKind::Struct => {
                if state.has_fields {
                    if !options.alternate() {
                        out.push_str(", ");
                    }
                } else {
                    out.push_str(state.name.as_deref().unwrap_or(""));
                    out.push_str(if options.alternate() { " {\n" } else { " { " });
                }
                if options.alternate() {
                    out.push_str(&" ".repeat(indent + 4));
                }
            }
            kind => {
                let (open, _) = builder_brackets(kind, state.name.as_deref());
                if state.has_fields {
                    if !options.alternate() {
                        out.push_str(", ");
                    }
                } else {
                    out.push_str(&open);
                    if options.alternate() {
                        out.push('\n');
                    }
                }
                if options.alternate() {
                    out.push_str(&" ".repeat(indent + 4));
                }
            }
        }
        out.push_str(text);
        if options.alternate() {
            out.push_str(",\n");
        }
        self.formatter_push(state.formatter, &out);
        self.with_fmt_builder(builder, |st| st.has_fields = true);
        Ok(())
    }

    /// The entry text's formatting shared by the builder
    /// entry/field builtins: the value's own `Debug` under
    /// the builder formatter's options.
    fn builder_entry_text(
        &mut self,
        frames: &mut Vec<Frame>,
        builder: u64,
        entry: &Value,
    ) -> InterpResult<String> {
        let state = self
            .fmt_builder_snapshot(builder)
            .ok_or_else(|| unsupported("a fmt builder on a finished capture".to_string()))?;
        let capture = self.capture_of(state.formatter)?;
        let options = capture.options();
        let field_indent = if options.alternate() {
            capture.indent + 4
        } else {
            capture.indent
        };
        self.format_value(frames, entry, FmtTrait::Debug, "", &options, field_indent)
    }

    /// `DebugList::entry` / `DebugSet::entry` /
    /// `DebugTuple::field`: append one entry.
    fn builder_entry(
        &mut self,
        frames: &mut Vec<Frame>,
        builder: u64,
        entry: &Value,
    ) -> InterpResult<()> {
        let text = self.builder_entry_text(frames, builder, entry)?;
        self.builder_write_entry(builder, &text)
    }

    /// `DebugTuple::field` is the tuple builder's `entry`.
    fn builder_tuple_field(
        &mut self,
        frames: &mut Vec<Frame>,
        builder: u64,
        entry: &Value,
    ) -> InterpResult<()> {
        self.builder_entry(frames, builder, entry)
    }

    /// `DebugMap::entry`: append one `key: value` entry.
    fn builder_map_entry(
        &mut self,
        frames: &mut Vec<Frame>,
        builder: u64,
        key: &Value,
        value: &Value,
    ) -> InterpResult<()> {
        let key_text = self.builder_entry_text(frames, builder, key)?;
        let value_text = self.builder_entry_text(frames, builder, value)?;
        self.builder_write_entry(builder, &format!("{key_text}: {value_text}"))
    }

    /// `DebugStruct::field`: append one `name: value` field.
    fn builder_struct_field(
        &mut self,
        frames: &mut Vec<Frame>,
        builder: u64,
        field_name: &str,
        value: &Value,
    ) -> InterpResult<()> {
        // The field's type tag comes from the recorded shape
        // of the type whose body is running, matched by field
        // name, when the builder is for that type (the usual
        // `debug_struct` in a handwritten impl).
        let tag = self
            .fmt_builder_snapshot(builder)
            .and_then(|state| {
                let variant_name = state.name.clone()?;
                self.capture_field_tags(state.formatter, &variant_name)
                    .and_then(|tags| {
                        let shape = self
                            .program()
                            .adt_debug_shapes
                            .get(pointee_tag(
                                &self.capture_of(state.formatter).ok()?.self_tag,
                            ))?
                            .clone();
                        let variant = shape
                            .variants
                            .iter()
                            .find(|v| v.name == variant_name)?;
                        let idx = variant
                            .fields
                            .iter()
                            .position(|f| f.name.as_deref() == Some(field_name))?;
                        tags.get(idx).cloned()
                    })
            })
            .unwrap_or_default();
        let state = self
            .fmt_builder_snapshot(builder)
            .ok_or_else(|| unsupported("a fmt builder on a finished capture".to_string()))?;
        let capture = self.capture_of(state.formatter)?;
        let options = capture.options();
        let field_indent = if options.alternate() {
            capture.indent + 4
        } else {
            capture.indent
        };
        let value_text = self.format_finisher_field(
            frames,
            value,
            &tag,
            &options,
            field_indent,
        )?;
        self.builder_write_entry(builder, &format!("{field_name}: {value_text}"))
    }

    /// A builder's `finish`: write the closing delimiter (or
    /// the whole empty form) to the builder's formatter.
    fn builder_finish(&mut self, builder: u64, exhaustive: bool) -> InterpResult<()> {
        let state = self
            .fmt_builder_snapshot(builder)
            .ok_or_else(|| unsupported("a fmt builder on a finished capture".to_string()))?;
        let capture = self.capture_of(state.formatter)?;
        let options = capture.options();
        let indent = capture.indent;
        let name = state.name.as_deref().unwrap_or("");
        let text = if !state.has_fields {
            match state.kind {
                FmtBuilderKind::List => "[]".to_string(),
                FmtBuilderKind::Set | FmtBuilderKind::Map => "{}".to_string(),
                FmtBuilderKind::Struct => {
                    if exhaustive {
                        name.to_string()
                    } else {
                        format!("{name} {{ .. }}")
                    }
                }
                FmtBuilderKind::Tuple => format!("{name}()"),
            }
        } else if options.alternate() {
            let mut out = String::new();
            if state.kind == FmtBuilderKind::Struct && !exhaustive {
                out.push_str(&" ".repeat(indent + 4));
                out.push_str("..,\n");
            }
            out.push_str(&" ".repeat(indent));
            out.push(match state.kind {
                FmtBuilderKind::List => ']',
                FmtBuilderKind::Set | FmtBuilderKind::Map | FmtBuilderKind::Struct => '}',
                FmtBuilderKind::Tuple => ')',
            });
            out
        } else {
            match state.kind {
                FmtBuilderKind::List => "]".to_string(),
                FmtBuilderKind::Set | FmtBuilderKind::Map => "}".to_string(),
                FmtBuilderKind::Struct => {
                    if exhaustive {
                        " }".to_string()
                    } else {
                        ", .. }".to_string()
                    }
                }
                FmtBuilderKind::Tuple => ")".to_string(),
            }
        };
        self.formatter_push(state.formatter, &text);
        Ok(())
    }

    /// Format one argument value through a formatting trait.
    /// `indent` is the current pretty-print (`{:#?}`) indent level.
    fn format_value(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        trait_: FmtTrait,
        ty: &str,
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<String> {
        // A one-field wrapper aggregate around string data,
        // formatted as a `String`: the storage form a
        // `MaybeUninit` payload slot leaves in raw storage (a
        // `Box::new_in` pointee read back through its
        // reference), where the model's `String` value is the
        // data itself. The wrapper is transparent to the
        // type's own formatting, exactly as it is to a typed
        // place's store (see `normalize_for_ty`).
        if let Value::Enum {
            variant: 0,
            fields,
        } = value
        {
            if fields.len() == 1 && matches!(fields[0], Value::Str(_)) {
                let plain = pointee_tag(ty);
                if plain == "std::string::String" || plain == "alloc::string::String" {
                    let field = fields[0].clone();
                    return self.format_value(frames, &field, trait_, ty, options, indent);
                }
            }
        }
        // References format as their pointee for every trait here.
        if matches!(value, Value::Ref(_)) {
            let pointee = self.resolve_value(frames, value)?;
            return self.format_value(frames, &pointee, trait_, ty, options, indent);
        }
        if let Value::FatRef { base, .. } = value {
            // A slice value: only Debug is defined for it (a list).
            if trait_ != FmtTrait::Debug {
                return Err(unsupported(format!(
                    "{trait_:?} formatting for a slice value"
                )));
            }
            let elem = slice_elem_tag(pointee_tag(ty));
            // A base denoting an element of a container denotes
            // the slice from that element on: keep the fat pointer
            // itself so the sequence formatter reads the elements
            // from that position (resolving it would read just the
            // element).
            if let Ok(resolved) = self.deref_value(frames, base) {
                if target_ends_at_element(&resolved) {
                    return self.format_debug_seq(
                        frames, value, options, indent, '[', ']', elem,
                    );
                }
            }
            let pointee = self.resolve_value(frames, value)?;
            return self.format_debug_seq(frames, &pointee, options, indent, '[', ']', elem);
        }
        if let Value::DynRef {
            base,
            concrete,
            traits,
            ..
        } = value
        {
            // A trait object's fmt traits are its concrete
            // type's (that is the vtable entry, for the principal
            // trait and its supertraits alike), so format the
            // pointee with the concrete type as the tag.
            // `Pointer` keeps the existing behaviour: the value
            // model has no address to print for the data pointer.
            if !matches!(trait_, FmtTrait::Pointer | FmtTrait::DebugNoop) {
                // Core's `Debug` impls for the `dyn Any` family
                // ignore the concrete value and print `Any { .. }`:
                // the vtable entry for such an object is that impl,
                // not the concrete type's `Debug`.
                if trait_ == FmtTrait::Debug {
                    let is_any = traits.iter().any(|t| t.trim().ends_with("::any::Any"));
                    let only_auto = traits.iter().all(|t| {
                        let t = t.trim();
                        t.ends_with("::any::Any")
                            || t.ends_with("::marker::Send")
                            || t.ends_with("::marker::Sync")
                            || t.ends_with("::marker::Unpin")
                            || t.ends_with("::panic::UnwindSafe")
                            || t.ends_with("::panic::RefUnwindSafe")
                    });
                    if is_any && only_auto {
                        return Ok(pad_width("Any { .. }", options));
                    }
                }
                let pointee = self.resolve_value(frames, base)?;
                // The base is the data pointer in the shape the
                // sized value had: a `Box<dyn Trait>` base is the
                // box aggregate (a one-field wrapper around the
                // pointee reference) and an `Rc`/`Arc` base is the
                // handle, so shape the value the way method
                // dispatch shapes its receiver (see
                // `ref_shaped_receiver` in the interpreter).
                let pointee = if let Some(id) = rc_handle_id(&pointee) {
                    Value::Ref(RefTarget::Alloc {
                        id,
                        projection: vec![RefProj::Field(2)],
                    })
                } else if let Value::Enum { fields, .. } = &pointee {
                    if fields.len() == 1 && matches!(base.as_ref(), Value::Enum { .. }) {
                        // The data pointer is the box aggregate
                        // itself, whose single field denotes the
                        // pointee: the pointee reference of an
                        // allocation-backed box (which the `Ref`
                        // arm above then resolves), or the pointee
                        // value itself of an aggregate-shaped box.
                        // A one-field payload reached through a
                        // reference is the payload itself, not a
                        // box, and keeps its shape.
                        fields[0].clone()
                    } else {
                        pointee
                    }
                } else {
                    pointee
                };
                return self.format_value(frames, &pointee, trait_, concrete, options, indent);
            }
        }
        // `std::io::Error` values in the bitpacked representation
        // the value model owns, and `ErrorKind` values: their
        // `Display` / `Debug` text is a fixed function of the
        // decoded representation (and, for OS errors, of the
        // platform's message text), produced here like the other
        // std shapes the structural engine models.
        if let Some(text) = self.format_io_error(value, trait_, ty, options) {
            return Ok(text);
        }
        // The standard pointer wrappers (`Box`, `Rc`, `Arc`)
        // forward their `Display` and `Debug` to the pointee:
        // format the pointee with the element type as the tag.
        if matches!(trait_, FmtTrait::Display | FmtTrait::Debug) {
            if let Some(text) = self.format_pointer_wrapper(
                frames, value, trait_, ty, options, indent,
            )? {
                return Ok(text);
            }
        }
        // `Cow<B>` forwards every fmt trait to the value it
        // holds (std defines the impls as the borrowed type's
        // own), so format the held value under the borrowed
        // type's tag.
        if let Some(text) = self.format_cow(frames, value, trait_, ty, options, indent)? {
            return Ok(text);
        }
        // A guest `fmt` implementation body, when the program
        // collected one for this type, is the faithful
        // formatting: run it against a captured `Formatter`
        // instead of reconstructing the output structurally.
        if let Some(text) =
            self.format_with_body(frames, value, trait_, ty, options, indent)?
        {
            return Ok(text);
        }
        match trait_ {
            FmtTrait::DebugNoop => Ok(String::new()),
            FmtTrait::Pointer => Err(unsupported(
                "pointer formatting: the value model has no addresses",
            )),
            FmtTrait::Display => self.format_display(frames, value, ty, options),
            FmtTrait::Debug => self.format_debug(frames, value, ty, options, indent),
            FmtTrait::Octal => self.format_radix(value, ty, 8, "0o", options),
            FmtTrait::LowerHex => self.format_radix(value, ty, 16, "0x", options),
            FmtTrait::UpperHex => self.format_radix(value, ty, 16, "0X", options),
            FmtTrait::Binary => self.format_radix(value, ty, 2, "0b", options),
            FmtTrait::LowerExp => self.format_exp(value, ty, false, options),
            FmtTrait::UpperExp => self.format_exp(value, ty, true, options),
        }
    }

    /// Structural `Display` / `Debug` for `std::io::Error` and
    /// `ErrorKind` values, mirroring core's own implementations
    /// (`core::io::error`): an OS error displays as
    /// `{message} (os error {code})` and debugs as
    /// `Os { code, kind, message }`; a kind-only (simple) error
    /// displays as the kind's message and debugs as
    /// `Kind({kind})`; a kind displays as its message. The
    /// `Custom` and `SimpleMessage` representations hold a
    /// pointer the value model does not resolve here, so those
    /// keep the existing behaviour (`None`).
    fn format_io_error(
        &self,
        value: &Value,
        trait_: FmtTrait,
        ty: &str,
        options: &FmtOptions,
    ) -> Option<String> {
        let plain = pointee_tag(ty);
        if plain == "std::io::ErrorKind" || plain == "core::io::ErrorKind" {
            if trait_ != FmtTrait::Display {
                return None;
            }
            let Value::Enum { variant, .. } = value else {
                return None;
            };
            return error_kind_message(*variant).map(|m| pad_str(m, options));
        }
        if plain != "std::io::Error" && plain != "core::io::Error" {
            return None;
        }
        let word = io_error_tagged_word(value)?;
        let payload = (word >> 32) as u32;
        match (trait_, word & 0b11) {
            (FmtTrait::Display, 0b10) => {
                let code = payload as i32;
                Some(pad_width(
                    &format!("{} (os error {code})", os_error_message(code)),
                    options,
                ))
            }
            (FmtTrait::Debug, 0b10) => {
                let code = payload as i32;
                let kind = error_kind_name(decode_error_kind_unix(code))?;
                Some(pad_width(
                    &format!(
                        "Os {{ code: {code}, kind: {kind}, message: \"{}\" }}",
                        os_error_message(code)
                    ),
                    options,
                ))
            }
            (FmtTrait::Display, 0b11) => {
                error_kind_message(payload).map(|m| pad_str(m, options))
            }
            (FmtTrait::Debug, 0b11) => error_kind_name(payload)
                .map(|name| pad_width(&format!("Kind({name})"), options)),
            _ => None,
        }
    }

    /// `Display` / `Debug` for the standard pointer wrappers
    /// (`Box`, `Rc`, `Arc`), which std defines as forwarding to
    /// the pointee: shape the pointee value the way the `dyn`
    /// formatting path shapes its base, then format it with the
    /// element type as the tag. `None` when the type is not one
    /// of the wrappers or the value does not have the wrapper's
    /// shape, in which case the existing behaviour stands.
    fn format_pointer_wrapper(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        trait_: FmtTrait,
        ty: &str,
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<Option<String>> {
        let tag = pointee_tag(ty);
        let head = tag.split('<').next().unwrap_or(tag).trim();
        if !matches!(
            head,
            "std::boxed::Box"
                | "alloc::boxed::Box"
                | "std::rc::Rc"
                | "alloc::rc::Rc"
                | "std::sync::Arc"
                | "alloc::sync::Arc"
        ) {
            return Ok(None);
        }
        let Some(elem) = first_generic_arg(tag) else {
            return Ok(None);
        };
        let pointee: Value = match tag_base_name(tag) {
            "Box" => {
                // A box value is the one-field wrapper around the
                // pointee reference (or, for an aggregate-shaped
                // box, around the pointee value itself) -- the
                // same shaping `format_debug_adt`'s `Box` arm and
                // the `dyn` path apply.
                let Value::Enum { fields, .. } = value else {
                    return Ok(None);
                };
                if fields.len() != 1 {
                    return Ok(None);
                }
                fields[0].clone()
            }
            _ => {
                // `Rc` / `Arc`: the handle names the inner
                // allocation, whose field 2 is the value. A bare
                // inner aggregate (the handle already resolved
                // through its allocation) carries the value as
                // its own field 2, behind the two count cells.
                if let Some(id) = rc_handle_id(value) {
                    Value::Ref(RefTarget::Alloc {
                        id,
                        projection: vec![RefProj::Field(2)],
                    })
                } else if let Value::Enum { fields, .. } = value {
                    if fields.len() == 3
                        && is_count_cell(&fields[0])
                        && is_count_cell(&fields[1])
                    {
                        fields[2].clone()
                    } else {
                        return Ok(None);
                    }
                } else {
                    return Ok(None);
                }
            }
        };
        Ok(Some(self.format_value(
            frames, &pointee, trait_, elem, options, indent,
        )?))
    }

    /// `Display` / `Debug` (and the remaining fmt traits) for
    /// a `Cow<'_, B>` value: std's impls forward to the held
    /// value — the borrowed variant's reference or the owned
    /// variant's value — formatted as `B` itself. `None` when
    /// the type is not a `Cow` or the value does not have the
    /// one-field enum shape, in which case the existing
    /// behaviour stands.
    fn format_cow(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        trait_: FmtTrait,
        ty: &str,
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<Option<String>> {
        let plain = pointee_tag(ty);
        if !plain.starts_with("std::borrow::Cow<")
            && !plain.starts_with("alloc::borrow::Cow<")
        {
            return Ok(None);
        }
        let Some(inner_ty) = last_generic_param(plain) else {
            return Ok(None);
        };
        let Value::Enum { fields, .. } = value else {
            return Ok(None);
        };
        if fields.len() != 1 {
            return Ok(None);
        }
        let field = fields[0].clone();
        // The borrowed variant's field is a fat reference when
        // the borrowed type is unsized: a fat reference whose
        // pointee is string data denotes the `str` itself (the
        // shape a `&str` field materialises in), while a fat
        // reference over elements stays the slice it is.
        let field = if let Value::FatRef { .. } = field {
            match self.resolve_value(frames, &field) {
                Ok(resolved @ Value::Str(_)) => resolved,
                _ => field,
            }
        } else {
            field
        };
        Ok(Some(self.format_value(
            frames, &field, trait_, &inner_ty, options, indent,
        )?))
    }

    fn format_display(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        ty: &str,
        options: &FmtOptions,
    ) -> InterpResult<String> {
        match value {
            Value::Int(v) => Ok(format_int_radix(*v, None, 10, "", options)),
            Value::Uint(v) => {
                if ty == "char" {
                    let c = as_char(*v)?;
                    Ok(pad_str(&c.to_string(), options))
                } else {
                    Ok(format_uint_radix(*v, None, 10, "", options))
                }
            }
            Value::Bool(b) => Ok(pad_str(bool_str(*b), options)),
            Value::Float(v) => Ok(format_float(*v, ty, options, false)),
            Value::Float128(_) => Err(unsupported(
                "Display formatting for an f128 value (decimal conversion is not modelled)",
            )),
            Value::Str(s) => Ok(pad_str(s, options)),
            Value::FmtArguments(inner) => {
                let text = self.format_arguments(frames, inner, 0)?;
                Ok(pad_width(&text, options))
            }
            Value::Tuple(_) => Err(unsupported(
                "Display formatting for a tuple value (core has no such impl)",
            )),
            Value::Enum { .. } => Err(unsupported(
                "Display formatting for an ADT value (its impl lives in unconverted code)",
            )),
            other => Err(InterpError::type_mismatch(
                "Display formatting",
                "integer, bool, float, string or char value",
                other.kind_name(),
            )),
        }
    }

    fn format_debug(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        ty: &str,
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<String> {
        match value {
            // The unit value's Debug (and Display) text is `()`.
            Value::Unit => Ok("()".to_string()),
            Value::Int(v) => {
                if options.flags & DEBUG_LOWER_HEX != 0 {
                    Ok(format_int_radix(*v, Some(ty), 16, "0x", options))
                } else if options.flags & DEBUG_UPPER_HEX != 0 {
                    Ok(format_int_radix(*v, Some(ty), 16, "0X", options))
                } else {
                    Ok(format_int_radix(*v, Some(ty), 10, "", options))
                }
            }
            Value::Uint(v) => {
                if ty == "char" {
                    let c = as_char(*v)?;
                    return Ok(pad_str(&format!("{c:?}"), options));
                }
                if options.flags & DEBUG_LOWER_HEX != 0 {
                    Ok(format_uint_radix(*v, Some(ty), 16, "0x", options))
                } else if options.flags & DEBUG_UPPER_HEX != 0 {
                    Ok(format_uint_radix(*v, Some(ty), 16, "0X", options))
                } else {
                    Ok(format_uint_radix(*v, Some(ty), 10, "", options))
                }
            }
            Value::Bool(b) => Ok(pad_str(bool_str(*b), options)),
            Value::Float(v) => Ok(format_float(*v, ty, options, true)),
            Value::Str(s) => {
                // Precision truncates the string (in chars) before
                // it is escaped and quoted.
                let truncated: String = match options.precision() {
                    Some(p) => s.chars().take(p as usize).collect(),
                    None => s.clone(),
                };
                Ok(pad_width(&format!("{truncated:?}"), options))
            }
            // Arrays ride as tuple values; the type tag is what
            // distinguishes `[T; N]` / `[T]` from a tuple, and core's
            // Debug for an array uses brackets, not parentheses.
            // The tag names the type the fmt argument was built
            // over, which may sit behind references (`&mut [i32; 4]`
            // for `println!("{:#?}", mref)`), so strip those first.
            Value::Tuple(items) if pointee_tag(ty).starts_with('[') => {
                let elem = slice_elem_tag(pointee_tag(ty));
                let tys: Vec<&str> = items.iter().map(|_| elem).collect();
                self.format_debug_seq_items(frames, items, options, indent, '[', ']', &tys)
            }
            Value::Tuple(items) => {
                let tys = tuple_elem_tags(pointee_tag(ty), items.len());
                self.format_debug_seq_items(frames, items, options, indent, '(', ')', &tys)
            }
            Value::FmtArguments(inner) => {
                let text = self.format_arguments(frames, inner, indent)?;
                Ok(pad_width(&text, options))
            }
            Value::Enum { variant, fields } => {
                self.format_debug_adt(frames, *variant, fields, ty, options, indent)
            }
            other => Err(InterpError::type_mismatch(
                "Debug formatting",
                "integer, bool, float, string, char or tuple value",
                other.kind_name(),
            )),
        }
    }

    /// Octal/hex/binary formatting of an integer value.
    fn format_radix(
        &self,
        value: &Value,
        ty: &str,
        radix: u32,
        prefix: &str,
        options: &FmtOptions,
    ) -> InterpResult<String> {
        match value {
            Value::Int(v) => Ok(format_int_radix(*v, Some(ty), radix, prefix, options)),
            Value::Uint(v) => Ok(format_uint_radix(*v, Some(ty), radix, prefix, options)),
            other => Err(InterpError::type_mismatch(
                "radix formatting",
                "integer value",
                other.kind_name(),
            )),
        }
    }

    /// Exponential formatting of an integer or float value
    /// (`{:e}` / `{:E}`).
    fn format_exp(
        &self,
        value: &Value,
        ty: &str,
        upper: bool,
        options: &FmtOptions,
    ) -> InterpResult<String> {
        if let Value::Float(v) = value {
            let body = if ty == "f32" {
                let x = *v as f32;
                match (options.precision(), upper) {
                    (Some(p), false) => format!("{:.*e}", p as usize, x),
                    (Some(p), true) => format!("{:.*E}", p as usize, x),
                    (None, false) => format!("{x:e}"),
                    (None, true) => format!("{x:E}"),
                }
            } else {
                match (options.precision(), upper) {
                    (Some(p), false) => format!("{:.*e}", p as usize, v),
                    (Some(p), true) => format!("{:.*E}", p as usize, v),
                    (None, false) => format!("{v:e}"),
                    (None, true) => format!("{v:E}"),
                }
            };
            let (negative, digits) = match body.strip_prefix('-') {
                Some(rest) => (true, rest.to_string()),
                None => (false, body),
            };
            return Ok(pad_integral(negative, "", &digits, options));
        }
        let (negative, digits) = match value {
            Value::Int(v) => (*v < 0, v.unsigned_abs().to_string()),
            Value::Uint(v) => (false, v.to_string()),
            other => {
                return Err(InterpError::type_mismatch(
                    "exponential formatting",
                    "integer or float value",
                    other.kind_name(),
                ))
            }
        };
        let body = exp_body(&digits, upper);
        Ok(pad_integral(negative, "", &body, options))
    }

    /// Format one operand of an `ops::Range`-family value with
    /// the field's recorded type tag.
    fn format_range_operand(
        &mut self,
        frames: &mut Vec<Frame>,
        variant_shape: &sparagmos_mir::AdtDebugVariant,
        index: usize,
        fields: &[Value],
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<String> {
        self.format_value(
            frames,
            &fields[index],
            FmtTrait::Debug,
            &variant_shape.fields[index].ty,
            options,
            indent,
        )
    }

    /// Derived-`Debug` formatting of an ADT value (`{:?}` /
    /// `{:#?}` of a struct or enum): the shape the driver recorded
    /// for the value's type tag supplies the variant name, the
    /// field names and each field's own type tag, so nested values
    /// format with their types known. This is exactly the output
    /// the compiler's `#[derive(Debug)]` expansion produces; a
    /// handwritten `Debug` impl is a body like any other and never
    /// reaches this formatter unless it itself formats a field
    /// with `{:?}`.
    ///
    /// Two std shapes the value model does not store in declared
    /// shape are recognised by tag first: a `Vec` value's fields
    /// are its elements (it prints as the list it is), and a `Box`
    /// value's one field is the pointee reference (its `Debug`
    /// forwards to the pointee). An ADT with no recorded shape
    /// stays an explicit unsupported error.
    fn format_debug_adt(
        &mut self,
        frames: &mut Vec<Frame>,
        variant: u32,
        fields: &[Value],
        ty: &str,
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<String> {
        let tag = pointee_tag(ty);
        match tag_base_name(tag) {
            "Vec" => {
                let elem = first_generic_arg(tag).unwrap_or("");
                let tys: Vec<&str> = fields.iter().map(|_| elem).collect();
                return self.format_debug_seq_items(
                    frames, fields, options, indent, '[', ']', &tys,
                );
            }
            "Box" if fields.len() == 1 => {
                let elem = first_generic_arg(tag).unwrap_or("");
                return self.format_value(
                    frames,
                    &fields[0],
                    FmtTrait::Debug,
                    elem,
                    options,
                    indent,
                );
            }
            _ => {}
        }
        // The shape is cloned out of the program: formatting
        // the fields runs guest `fmt` bodies, which needs
        // `&mut self`, so no program borrow may live across
        // the field formatting below.
        let Some(shape) = self.program().adt_debug_shapes.get(tag).cloned() else {
            return Err(unsupported(
                "Debug formatting for an ADT value (no derived-Debug shape was recorded for its type)",
            ));
        };
        let Some(variant_shape) = shape.variants.get(variant as usize) else {
            return Err(unsupported(format!(
                "Debug formatting for an ADT value (variant {variant} of {tag} has no recorded shape)"
            )));
        };
        if variant_shape.fields.len() != fields.len() {
            return Err(InterpError::type_mismatch(
                "Debug formatting",
                "an ADT value whose fields match the recorded shape",
                "an ADT value with a different field count",
            ));
        }
        // Three `core` wrapper types whose handwritten `Debug`
        // impls are a fixed function of the stored value, produced
        // here like the `ops::Range` family below. The match is on
        // the tag's head (the path before any generic arguments),
        // so a user type that merely shares the base name keeps
        // its derived shape.
        let head = tag.split('<').next().unwrap_or(tag).trim();
        match head {
            "core::cell::UnsafeCell" | "std::cell::UnsafeCell" => {
                // The impl is non-exhaustive by design.
                return Ok(pad_width("UnsafeCell { .. }", options));
            }
            "core::marker::PhantomData" | "std::marker::PhantomData" => {
                // The impl prints the type parameter it carries.
                let arg = first_generic_arg(tag).unwrap_or("");
                return Ok(pad_width(&format!("PhantomData<{arg}>"), options));
            }
            "core::cell::RefCell" | "std::cell::RefCell" => {
                // The impl shows only the `value` field, as the
                // inner value behind its `UnsafeCell`, or as
                // `<borrowed>` while a borrow is outstanding (the
                // `borrow` field's flag is nonzero then). The
                // fields are located by their recorded names.
                let named_field = |want: &str| -> Option<Value> {
                    variant_shape
                        .fields
                        .iter()
                        .position(|f| f.name.as_deref() == Some(want))
                        .and_then(|i| fields.get(i).cloned())
                };
                let unwrap_one = |v: &Value| -> Option<Value> {
                    match v {
                        Value::Enum { fields: inner, .. } if inner.len() == 1 => {
                            Some(inner[0].clone())
                        }
                        Value::Tuple(inner) if inner.len() == 1 => Some(inner[0].clone()),
                        _ => None,
                    }
                };
                let borrowed = named_field("borrow")
                    .and_then(|b| unwrap_one(&b))
                    .and_then(|b| unwrap_one(&b))
                    .map(|flag| !matches!(flag, Value::Int(0) | Value::Uint(0)));
                if let (Some(borrowed), Some(stored)) = (borrowed, named_field("value")) {
                    let stored = if matches!(stored, Value::Ref(_)) {
                        self.resolve_value(frames, &stored)?
                    } else {
                        stored
                    };
                    if borrowed {
                        return Ok(pad_width("RefCell { value: <borrowed> }", options));
                    }
                    if let Some(inner) = unwrap_one(&stored) {
                        let elem = first_generic_arg(tag).unwrap_or("");
                        if options.alternate() {
                            let inner_indent = indent + 4;
                            let text = self.format_value(
                                frames,
                                &inner,
                                FmtTrait::Debug,
                                elem,
                                options,
                                inner_indent,
                            )?;
                            return Ok(format!(
                                "RefCell {{\n{}value: {},\n{}}}",
                                " ".repeat(inner_indent),
                                text,
                                " ".repeat(indent)
                            ));
                        }
                        let text = self.format_value(
                            frames,
                            &inner,
                            FmtTrait::Debug,
                            elem,
                            &FmtOptions::new(),
                            indent,
                        )?;
                        return Ok(pad_width(
                            &format!("RefCell {{ value: {text} }}"),
                            options,
                        ));
                    }
                }
            }
            "core::cell::Cell" | "std::cell::Cell" if fields.len() == 1 => {
                // The impl shows the inner value as the `value`
                // field, not the `UnsafeCell` wrapper the value is
                // stored in: unwrap one layer and format the inner
                // value with the wrapper's own type argument.
                let stored = if matches!(fields[0], Value::Ref(_)) {
                    self.resolve_value(frames, &fields[0])?
                } else {
                    fields[0].clone()
                };
                let inner = match &stored {
                    Value::Enum { fields: inner, .. } if inner.len() == 1 => {
                        Some(inner[0].clone())
                    }
                    Value::Tuple(inner) if inner.len() == 1 => Some(inner[0].clone()),
                    _ => None,
                };
                if let Some(inner) = inner {
                    let elem = first_generic_arg(tag).unwrap_or("");
                    if options.alternate() {
                        let inner_indent = indent + 4;
                        let text = self.format_value(
                            frames,
                            &inner,
                            FmtTrait::Debug,
                            elem,
                            options,
                            inner_indent,
                        )?;
                        return Ok(format!(
                            "Cell {{\n{}value: {},\n{}}}",
                            " ".repeat(inner_indent),
                            text,
                            " ".repeat(indent)
                        ));
                    }
                    let text = self.format_value(
                        frames,
                        &inner,
                        FmtTrait::Debug,
                        elem,
                        &FmtOptions::new(),
                        indent,
                    )?;
                    return Ok(pad_width(&format!("Cell {{ value: {text} }}"), options));
                }
            }
            _ => {}
        }
        // The `ops::Range` family's `Debug` impls are handwritten
        // in core, but their output is a fixed function of the
        // fields (`start..end` and its siblings), so it is produced
        // here from the recorded shape like the derived form. The
        // match is on the tag's head (the path before any generic
        // arguments): a user type that merely shares the base name
        // renders without the `core::ops` path and keeps its
        // derived shape.
        let tag_head = tag.split('<').next().unwrap_or(tag).trim();
        let range_text = match tag_head {
            "core::ops::Range" | "std::ops::Range" if fields.len() == 2 => {
                let start = self.format_range_operand(frames, variant_shape, 0, fields, options, indent)?;
                let end = self.format_range_operand(frames, variant_shape, 1, fields, options, indent)?;
                Some(format!("{start}..{end}"))
            }
            "core::ops::RangeInclusive" | "std::ops::RangeInclusive" if fields.len() == 3 => {
                // Fields are `start`, `end`, `exhausted`; the
                // printed form uses the first two.
                let start = self.format_range_operand(frames, variant_shape, 0, fields, options, indent)?;
                let end = self.format_range_operand(frames, variant_shape, 1, fields, options, indent)?;
                Some(format!("{start}..={end}"))
            }
            "core::ops::RangeFrom" | "std::ops::RangeFrom" if fields.len() == 1 => {
                let start = self.format_range_operand(frames, variant_shape, 0, fields, options, indent)?;
                Some(format!("{start}.."))
            }
            "core::ops::RangeTo" | "std::ops::RangeTo" if fields.len() == 1 => {
                let end = self.format_range_operand(frames, variant_shape, 0, fields, options, indent)?;
                Some(format!("..{end}"))
            }
            "core::ops::RangeToInclusive" | "std::ops::RangeToInclusive" if fields.len() == 1 => {
                let end = self.format_range_operand(frames, variant_shape, 0, fields, options, indent)?;
                Some(format!("..={end}"))
            }
            "core::ops::RangeFull" | "std::ops::RangeFull" if fields.is_empty() => {
                Some("..".to_string())
            }
            _ => None,
        };
        if let Some(text) = range_text {
            return Ok(pad_width(&text, options));
        }
        let name = variant_shape.name.as_str();
        if fields.is_empty() {
            return Ok(pad_width(name, options));
        }
        let named = variant_shape.fields.iter().all(|f| f.name.is_some());
        if options.alternate() {
            let inner = indent + 4;
            let pad = " ".repeat(inner);
            let mut out = String::new();
            out.push_str(name);
            out.push_str(if named { " {\n" } else { "(\n" });
            for (field, value) in variant_shape.fields.iter().zip(fields) {
                out.push_str(&pad);
                if let Some(field_name) = &field.name {
                    out.push_str(field_name);
                    out.push_str(": ");
                }
                out.push_str(&self.format_value(
                    frames,
                    value,
                    FmtTrait::Debug,
                    &field.ty,
                    options,
                    inner,
                )?);
                out.push_str(",\n");
            }
            out.push_str(&" ".repeat(indent));
            out.push(if named { '}' } else { ')' });
            return Ok(out);
        }
        let mut parts = Vec::with_capacity(fields.len());
        for (field, value) in variant_shape.fields.iter().zip(fields) {
            let text = self.format_value(
                frames,
                value,
                FmtTrait::Debug,
                &field.ty,
                &FmtOptions::new(),
                indent,
            )?;
            match &field.name {
                Some(field_name) => parts.push(format!("{field_name}: {text}")),
                None => parts.push(text),
            }
        }
        let body = if named {
            format!("{name} {{ {} }}", parts.join(", "))
        } else {
            format!("{name}({})", parts.join(", "))
        };
        Ok(pad_width(&body, options))
    }

    /// Debug formatting of a tuple or slice value: `(a, b)` / `[a, b]`,
    /// or the multi-line form under `{:#?}`.
    fn format_debug_seq(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        options: &FmtOptions,
        indent: usize,
        open: char,
        close: char,
        elem_ty: &str,
    ) -> InterpResult<String> {
        let items = match value {
            Value::Tuple(items) => items,
            // A self-contained fat pointer (a slice constant, a
            // materialised sub-slice) resolves to itself, so it can
            // reach this formatter directly: its elements are the
            // base's elements truncated to the carried length.
            Value::FatRef { .. } => {
                let (items, _) = self.slice_elements(frames, value)?;
                let tys: Vec<&str> = items.iter().map(|_| elem_ty).collect();
                return self.format_debug_seq_items(
                    frames,
                    &items,
                    options,
                    indent,
                    open,
                    close,
                    &tys,
                );
            }
            Value::Bytes(b) => {
                let items: Vec<Value> = b.iter().map(|byte| Value::Uint(*byte as u128)).collect();
                let tys: Vec<&str> = items.iter().map(|_| "u8").collect();
                return self.format_debug_seq_items(frames, &items, options, indent, open, close, &tys);
            }
            other => {
                return Err(InterpError::type_mismatch(
                    "Debug sequence formatting",
                    "tuple or byte value",
                    other.kind_name(),
                ))
            }
        };
        let tys: Vec<&str> = items.iter().map(|_| elem_ty).collect();
        self.format_debug_seq_items(frames, items, options, indent, open, close, &tys)
    }

    fn format_debug_seq_items(
        &mut self,
        frames: &mut Vec<Frame>,
        items: &[Value],
        options: &FmtOptions,
        indent: usize,
        open: char,
        close: char,
        elem_tys: &[&str],
    ) -> InterpResult<String> {
        // Element type tags are not tracked in values; the caller
        // supplies the ones it can recover from the container's own
        // tag (an empty tag formats the element from its value kind
        // alone, as before).
        if options.alternate() {
            if items.is_empty() {
                return Ok(format!("{open}{close}"));
            }
            let inner = indent + 4;
            let mut out = String::new();
            out.push(open);
            out.push('\n');
            for (i, item) in items.iter().enumerate() {
                out.push_str(&" ".repeat(inner));
                out.push_str(&self.format_value(
                    frames,
                    item,
                    FmtTrait::Debug,
                    elem_tys.get(i).copied().unwrap_or(""),
                    options,
                    inner,
                )?);
                out.push_str(",\n");
            }
            out.push_str(&" ".repeat(indent));
            out.push(close);
            return Ok(out);
        }
        let mut parts = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            parts.push(self.format_value(
                frames,
                item,
                FmtTrait::Debug,
                elem_tys.get(i).copied().unwrap_or(""),
                &FmtOptions::new(),
                indent,
            )?);
        }
        let body = if open == '(' && parts.len() == 1 {
            format!("({},)", parts[0])
        } else {
            format!("{open}{}{close}", parts.join(", "))
        };
        Ok(pad_width(&body, options))
    }
}

fn bool_str(b: bool) -> &'static str {
    if b {
        "true"
    } else {
        "false"
    }
}

fn as_char(v: u128) -> InterpResult<char> {
    let n = u32::try_from(v).map_err(|_| {
        InterpError::type_mismatch("char formatting", "char value", "integer out of range")
    })?;
    char::from_u32(n).ok_or_else(|| {
        InterpError::type_mismatch("char formatting", "char value", "invalid code point")
    })
}

fn template_piece(template: &[u8], start: usize, len: usize) -> InterpResult<&str> {
    let bytes = template
        .get(start..start + len)
        .ok_or_else(|| template_error("literal piece overruns the template"))?;
    std::str::from_utf8(bytes).map_err(|_| template_error("literal piece is not UTF-8"))
}

fn read_u16(template: &[u8], at: usize) -> InterpResult<u16> {
    let b = template
        .get(at..at + 2)
        .ok_or_else(|| template_error("truncated u16 field"))?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

fn read_u32(template: &[u8], at: usize) -> InterpResult<u32> {
    let b = template
        .get(at..at + 4)
        .ok_or_else(|| template_error("truncated u32 field"))?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn template_arg<'v>(args: &'v [Value], idx: usize) -> InterpResult<&'v Value> {
    args.get(idx).ok_or_else(|| {
        InterpError::type_mismatch(
            "fmt pieces template",
            "argument for every placeholder",
            format!("missing argument {idx}"),
        )
    })
}

/// `(pre, post)` padding strings of total length `n`.
fn padding(n: usize, align: Align, fill: char) -> (String, String) {
    let one = |k: usize| fill.to_string().repeat(k);
    match align {
        Align::Left => (String::new(), one(n)),
        Align::Right => (one(n), String::new()),
        Align::Center => {
            let pre = n / 2;
            (one(pre), one(n - pre))
        }
    }
}

/// core `Formatter::pad`: precision truncates to that many chars,
/// width pads with the fill (default align Left).
fn pad_str(s: &str, options: &FmtOptions) -> String {
    let body: String = match options.precision() {
        Some(p) => s.chars().take(p as usize).collect(),
        None => s.to_string(),
    };
    pad_width(&body, options)
}

/// Width padding only (no precision truncation).
/// Whether a rendered type tag names a guest-crate type —
/// the only types whose `fmt` implementation bodies the fmt
/// engine runs. Crate-qualified, and not a core/std/alloc
/// path (those impls are core's; the structural engine
/// models them).
pub(crate) fn guest_fmt_tag(tag: &str) -> bool {
    let head = tag.trim().trim_start_matches('&').trim_start();
    (head.starts_with(|c: char| c.is_alphabetic() || c == '_'))
        && head.contains("::")
        && !head.starts_with("core::")
        && !head.starts_with("std::")
        && !head.starts_with("alloc::")
}

/// Split a rendered type at its top-level commas (commas
/// not nested inside `<>`, `()` or `[]`).
fn split_top_commas(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

/// Whether a rendered generic argument is a lifetime
/// (`'a`, `'_`, `'static`): char literals also start with a
/// quote but close it, lifetimes never do.
fn is_lifetime_arg(arg: &str) -> bool {
    let a = arg.trim();
    a.len() > 1
        && a.starts_with('\'')
        && !a.ends_with('\'')
        && a[1..].chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Whether a rendered generic argument is a const literal
/// (an integer, a boolean or a char) rather than a type.
fn is_const_literal(arg: &str) -> bool {
    let a = arg.trim();
    !a.is_empty()
        && (a == "true"
            || a == "false"
            || a.chars().all(|c| c.is_ascii_digit())
            || (a.len() > 1
                && a.starts_with('-')
                && a[1..].chars().all(|c| c.is_ascii_digit()))
            || (a.starts_with('\'') && a.ends_with('\'') && a.chars().count() <= 4))
}

/// Normalize a rendered type for instance-name matching:
/// drop const generic arguments (see [`is_const_literal`])
/// and lifetime arguments (see [`is_lifetime_arg`]) at every
/// nesting level and remove whitespace, so the converter's
/// tag rendering (type arguments only) and the compiler's
/// instance name (all arguments) compare equal for the
/// same type.
fn normalize_fmt_tag(tag: &str) -> String {
    fn norm(s: &str, out: &mut String) {
        let bytes = s.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i] == b'<' {
                let mut depth = 0i32;
                let mut j = i;
                while j < bytes.len() {
                    match bytes[j] {
                        b'<' => depth += 1,
                        b'>' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    j += 1;
                }
                if j >= bytes.len() {
                    out.push_str(&s[i..]);
                    return;
                }
                let inner = &s[i + 1..j];
                let kept: Vec<String> = split_top_commas(inner)
                    .iter()
                    .filter(|a| !is_const_literal(a) && !is_lifetime_arg(a))
                    .map(|a| {
                        let mut t = String::new();
                        norm(a, &mut t);
                        t
                    })
                    .collect();
                if !kept.is_empty() {
                    out.push('<');
                    out.push_str(&kept.join(","));
                    out.push('>');
                }
                i = j + 1;
            } else {
                let ch_len = s[i..].chars().next().map_or(1, |c| c.len_utf8());
                out.push_str(&s[i..i + ch_len]);
                i += ch_len;
            }
        }
    }
    let mut out = String::new();
    norm(tag, &mut out);
    out.chars().filter(|c| !c.is_whitespace()).collect()
}

/// The `(open, close)` delimiters of a builder kind; the
/// struct kind is rendered by its own entry/finish arms and
/// does not use this.
fn builder_brackets(kind: FmtBuilderKind, name: Option<&str>) -> (String, String) {
    match kind {
        FmtBuilderKind::List => ("[".to_string(), "]".to_string()),
        FmtBuilderKind::Set | FmtBuilderKind::Map => ("{".to_string(), "}".to_string()),
        FmtBuilderKind::Tuple => (format!("{}(", name.unwrap_or("")), ")".to_string()),
        FmtBuilderKind::Struct => (format!("{} {{", name.unwrap_or("")), "}".to_string()),
    }
}

/// Render a derived-`Debug` struct value the way core's
/// `debug_struct_fieldN_finish` does: `Name { a: 1, b: 2 }`,
/// or the multi-line form under `{:#}` with the fields at
/// `indent + 4` and the closing brace at `indent`.
fn render_debug_struct(
    name: &str,
    fields: &[(String, String)],
    options: &FmtOptions,
    indent: usize,
) -> String {
    if options.alternate() {
        let pad = " ".repeat(indent + 4);
        let mut out = String::new();
        out.push_str(name);
        out.push_str(" {\n");
        for (field_name, text) in fields {
            out.push_str(&pad);
            out.push_str(field_name);
            out.push_str(": ");
            out.push_str(text);
            out.push_str(",\n");
        }
        out.push_str(&" ".repeat(indent));
        out.push('}');
        out
    } else {
        let parts: Vec<String> = fields
            .iter()
            .map(|(field_name, text)| format!("{field_name}: {text}"))
            .collect();
        format!("{name} {{ {} }}", parts.join(", "))
    }
}

/// Render a derived-`Debug` tuple-struct value the way
/// core's `debug_tuple_fieldN_finish` does: `Name(1, 2)`, or
/// the multi-line form under `{:#}`.
fn render_debug_tuple(
    name: &str,
    fields: &[String],
    options: &FmtOptions,
    indent: usize,
) -> String {
    if options.alternate() {
        let pad = " ".repeat(indent + 4);
        let mut out = String::new();
        out.push_str(name);
        out.push_str("(\n");
        for text in fields {
            out.push_str(&pad);
            out.push_str(text);
            out.push_str(",\n");
        }
        out.push_str(&" ".repeat(indent));
        out.push(')');
        out
    } else {
        format!("{name}({})", fields.join(", "))
    }
}

fn pad_width(s: &str, options: &FmtOptions) -> String {
    let width = options.width().unwrap_or(0) as usize;
    let len = s.chars().count();
    if len >= width {
        return s.to_string();
    }
    let (pre, post) = padding(
        width - len,
        options.align().unwrap_or(Align::Left),
        options.fill(),
    );
    format!("{pre}{s}{post}")
}

/// core `Formatter::pad_integral`: the sign and (alternate) prefix
/// count toward the width; zero-padding is sign-aware; otherwise the
/// default alignment is Right.
fn pad_integral(negative: bool, prefix: &str, digits: &str, options: &FmtOptions) -> String {
    let sign = if negative {
        "-"
    } else if options.sign_plus() {
        "+"
    } else {
        ""
    };
    let prefix = if options.alternate() { prefix } else { "" };
    let width = options.width().unwrap_or(0) as usize;
    let total = sign.len() + prefix.len() + digits.len();
    if total >= width {
        return format!("{sign}{prefix}{digits}");
    }
    let side = width - total;
    if options.zero_pad() && options.align() != Some(Align::Left) {
        return format!("{sign}{prefix}{}{digits}", "0".repeat(side));
    }
    let (pre, post) = padding(
        side,
        options.align().unwrap_or(Align::Right),
        options.fill(),
    );
    format!("{pre}{sign}{prefix}{digits}{post}")
}

/// The bit width a type tag names, for two's-complement radix
/// formatting of negative signed values.
fn tag_bits(ty: &str) -> u32 {
    match ty {
        "i8" | "u8" => 8,
        "i16" | "u16" => 16,
        "i32" | "u32" => 32,
        "i64" | "u64" | "isize" | "usize" => 64,
        _ => 128,
    }
}

fn digits_in_radix(mut n: u128, radix: u32) -> String {
    if n == 0 {
        return "0".to_string();
    }
    let mut out = Vec::new();
    while n > 0 {
        let d = (n % radix as u128) as u32;
        out.push(char::from_digit(d, radix).expect("digit below radix"));
        n /= radix as u128;
    }
    out.iter().rev().collect()
}

/// Integer formatting through `pad_integral`. Radix 10 formats the
/// magnitude with a sign; other radices format the two's-complement
/// pattern at the type's width, as core does.
fn format_int_radix(v: i128, ty: Option<&str>, radix: u32, prefix: &str, options: &FmtOptions) -> String {
    let (negative, digits) = if radix == 10 {
        (v < 0, v.unsigned_abs().to_string())
    } else {
        let bits = tag_bits(ty.unwrap_or(""));
        let mask = if bits >= 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        };
        (false, digits_in_radix((v as u128) & mask, radix))
    };
    finish_int(negative, digits, prefix, options)
}

fn format_uint_radix(v: u128, ty: Option<&str>, radix: u32, prefix: &str, options: &FmtOptions) -> String {
    let digits = if radix == 10 {
        v.to_string()
    } else {
        let bits = tag_bits(ty.unwrap_or(""));
        let mask = if bits >= 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        };
        digits_in_radix(v & mask, radix)
    };
    finish_int(false, digits, prefix, options)
}

/// Precision (minimum digits, zero-extended) then `pad_integral`;
/// a specified precision suppresses sign-aware zero-padding.
fn finish_int(negative: bool, mut digits: String, prefix: &str, options: &FmtOptions) -> String {
    // A `0X` prefix marks the upper-case hex form: core prints
    // its digits upper-cased and the alternate prefix itself
    // stays `0x`.
    let prefix = if prefix == "0X" {
        digits = digits.to_ascii_uppercase();
        "0x"
    } else {
        prefix
    };
    let mut options = *options;
    if let Some(p) = options.precision() {
        if digits.len() < p as usize {
            digits = format!("{}{digits}", "0".repeat(p as usize - digits.len()));
        }
        options.flags &= !SIGN_AWARE_ZERO_PAD;
    }
    pad_integral(negative, prefix, &digits, &options)
}

/// A type tag with any reference / raw-pointer layers stripped:
/// the tag of the value a fmt argument ultimately formats.
fn pointee_tag(ty: &str) -> &str {
    let mut ty = ty;
    loop {
        ty = ty.trim_start();
        if let Some(rest) = ty.strip_prefix("&mut ") {
            ty = rest;
        } else if let Some(rest) = ty.strip_prefix('&') {
            ty = rest;
        } else if let Some(rest) = ty.strip_prefix("*const ") {
            ty = rest;
        } else if let Some(rest) = ty.strip_prefix("*mut ") {
            ty = rest;
        } else {
            return ty;
        }
    }
}

/// The base name of a rendered type tag: the last path segment
/// of the head before any generic argument list or tuple body
/// (`std::vec::Vec<i32>` -> `Vec`, `Pair<i32, u8>` -> `Pair`).
fn tag_base_name(tag: &str) -> &str {
    let head = tag.split(['<', '(']).next().unwrap_or(tag).trim();
    head.rsplit("::").next().unwrap_or(head).trim()
}

/// Split a rendered tag list on its top-level commas: commas
/// inside `<...>`, `(...)` and `[...]` groups do not split.
fn split_tag_list(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    let tail = s[start..].trim();
    if !tail.is_empty() {
        parts.push(tail);
    }
    parts
}

/// The inside of the bracket group that opens at byte `open` in a
/// rendered tag, honouring nested groups of every bracket kind.
fn tag_group_inner(tag: &str, open: usize) -> Option<&str> {
    let bytes = tag.as_bytes();
    let mut depth = 0i32;
    let mut i = open;
    while i < bytes.len() {
        match bytes[i] {
            b'<' | b'(' | b'[' => depth += 1,
            b'>' | b')' | b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&tag[open + 1..i]);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// The first generic argument of a rendered tag
/// (`Vec<Point, Global>` -> `Point`), if it has a `<...>` list.
fn first_generic_arg(tag: &str) -> Option<&str> {
    let open = tag.find('<')?;
    let inner = tag_group_inner(tag, open)?;
    split_tag_list(inner).into_iter().next()
}

/// The element tag of a rendered array or slice tag
/// (`[Point; 2]` / `[Point]` -> `Point`); empty when the tag is
/// not in that form.
fn slice_elem_tag(tag: &str) -> &str {
    let Some(rest) = tag.strip_prefix('[') else {
        return "";
    };
    let Some(inner) = rest.strip_suffix(']') else {
        return "";
    };
    // An array length sits behind a top-level `;`.
    let mut depth = 0i32;
    for (i, c) in inner.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ';' if depth == 0 => return inner[..i].trim(),
            _ => {}
        }
    }
    inner.trim()
}

/// The per-element tags of a rendered tuple tag, one per element
/// (`(Point, i32)` -> [`Point`, `i32`]); an element whose tag
/// cannot be recovered gets the empty tag (it then formats from
/// its value kind alone).
fn tuple_elem_tags(tag: &str, count: usize) -> Vec<&str> {
    let empty: Vec<&str> = vec![""; count];
    let Some(open) = tag.find('(') else {
        return empty;
    };
    let Some(inner) = tag_group_inner(tag, open) else {
        return empty;
    };
    let parts = split_tag_list(inner);
    if parts.len() == count {
        parts
    } else {
        empty
    }
}

/// Display/Debug formatting of a float value. The host formatter is
/// the same std code the native binary runs, so the
/// shortest-round-trip rendering matches exactly; `ty` selects the
/// width because an `f32` value is stored widened to `f64` and must
/// be narrowed back first (`0.1f32` prints as `0.1`, not as the
/// widened `0.10000000149011612`). NaN ignores the sign flag and
/// precision in core's float formatting; width still applies.
fn format_float(v: f64, ty: &str, options: &FmtOptions, debug: bool) -> String {
    if v.is_nan() {
        let mut options = *options;
        options.flags &= !(SIGN_PLUS | SIGN_AWARE_ZERO_PAD);
        return pad_integral(false, "", "NaN", &options);
    }
    let body = if ty == "f32" {
        let x = v as f32;
        match (options.precision(), debug) {
            (Some(p), false) => format!("{:.*}", p as usize, x),
            (Some(p), true) => format!("{:.*?}", p as usize, x),
            (None, false) => format!("{x}"),
            (None, true) => format!("{x:?}"),
        }
    } else {
        match (options.precision(), debug) {
            (Some(p), false) => format!("{:.*}", p as usize, v),
            (Some(p), true) => format!("{:.*?}", p as usize, v),
            (None, false) => format!("{v}"),
            (None, true) => format!("{v:?}"),
        }
    };
    let (negative, digits) = match body.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, body.as_str()),
    };
    pad_integral(negative, "", digits, options)
}

/// The body of an exponential rendering: `1.23e2`, `0e0`, ...
fn exp_body(digits: &str, upper: bool) -> String {
    let trimmed = digits.trim_start_matches('0');
    if trimmed.is_empty() {
        return if upper { "0E0".to_string() } else { "0e0".to_string() };
    }
    let exp = trimmed.len() - 1;
    let mantissa = trimmed.trim_end_matches('0');
    let mantissa = if mantissa.len() == 1 {
        mantissa.to_string()
    } else {
        format!("{}.{}", &mantissa[..1], &mantissa[1..])
    };
    format!("{mantissa}{}{exp}", if upper { 'E' } else { 'e' })
}
