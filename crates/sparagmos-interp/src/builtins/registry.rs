//! The builtin registry: the Builtin kind enum and its operation enums.



use crate::interp::FmtBuilderKind;
use crate::ir::{BinOp, UnOp};
use crate::value::FmtTrait;


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
    /// over `OsString` values (the stored elements are the
    /// argument strings; the iterator step presents each in
    /// the standard `OsString` buffer form).
    EnvArgsOs,
    /// `std::env::vars() -> Vars`: the environment iterator
    /// over the host environment, snapshotted at the call
    /// (the wrapper-shape documentation is on the dispatch
    /// arm, next to `EnvArgs`).
    EnvVars,
    /// `std::env::vars_os() -> VarsOs`: the environment
    /// iterator over `OsString` keys and values, snapshotted
    /// from the host environment at the call.
    EnvVarsOs,
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
    /// `std::path::Path::is_file`: host metadata query (follows
    /// symlinks, as `std::fs::metadata` does); `false` when the
    /// metadata cannot be read, as in std.
    PathIsFile,
    /// `std::path::Path::is_dir`: as `PathIsFile`, for a
    /// directory.
    PathIsDir,
    /// `std::path::Path::_join(&self, &Path) -> PathBuf`: the
    /// owned path with the argument pushed onto it (the leaf
    /// under `Path::join`).
    PathJoin,
    /// `std::fs::OpenOptions::_open(&options, path)`: the fs
    /// sys layer's entry point. A path containing a NUL byte is
    /// rejected with `InvalidInput`, exactly as std's own
    /// CString conversion rejects it; otherwise the guest
    /// options are applied to a host open and the resulting
    /// file enters the interpreter's file table (see the `File`
    /// value model in `exec_file_io`).
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
    /// A `std::sys::net::connection::socket` leaf operation,
    /// performed on the interpreter's host socket table: the
    /// arguments and results are plain values (addresses in
    /// std's `SocketAddr` shape, `io::Error` in std's own
    /// shape), so the converted public `std::net` bodies above
    /// them run unchanged.
    NetSys(NetSysOp),
    /// A `core::unicode::unicode_data` skiplist property
    /// lookup (`<module>::lookup_slow(char) -> bool`): the
    /// generated tables' only reader has no MIR body in the
    /// shipped core (it is `#[inline(never)]` and non-generic,
    /// so the rmeta carries no body for it), so the runtime
    /// replays core's `skip_search` over the module's own
    /// table statics, whose seeds the driver records in the
    /// program (see `exec_unicode_lookup_slow` in
    /// `builtins::unicode`).
    UnicodeLookupSlow(UnicodeProperty),
    /// `std::process::exit(code)`: end the run with the given
    /// exit code (see `InterpError::ProcessExit`).
    ProcessExit,
    /// `core::intrinsics::abort()` / `std::process::abort()`:
    /// end the run as an abort (see `InterpError::Abort`), the
    /// guest-side equivalent of dying on `SIGABRT`.
    IntrinsicsAbort,
    /// `core::intrinsics::unreachable()`: the point is
    /// unreachable by construction, so executing it is
    /// undefined behaviour; the run ends as an abort (see
    /// `InterpError::Abort`), the same crash class as
    /// executing a MIR `Unreachable` terminator.
    IntrinsicsUnreachable,
    /// `std::io::stdout() -> Stdout`: the standard output handle.
    /// The handle value is a marker in this model: writes through
    /// it are modelled by the `Write` builtins, which dispatch on
    /// the receiver type, not on the handle's contents.
    Stdout,
    /// `std::io::stdin() -> Stdin`: the standard input handle.
    /// The handle value is the file value of the run's own
    /// stdin in the file table (see `Interpreter::stdin_file_id`
    /// in `interp.rs`), so reads through it -- and through its
    /// lock -- are the file model's reads.
    Stdin,
    /// `std::io::Stdout::lock` / `std::io::Stderr::lock`: the
    /// locked handle. The lock value is the handle itself in this
    /// model: writes through it are modelled by the `Write`
    /// builtins, which dispatch on the receiver type, and the
    /// guard's unlock on drop has no observable effect.
    StdStreamLock,
    /// `std::io::Write::write_fmt(&mut self, Arguments)` for the
    /// modelled receivers (the std handles and `Vec<u8>`).
    WriteFmt,
    /// `core::fmt::write(&mut dyn Write, Arguments)`: format the
    /// arguments (the same model `format_arguments` implements)
    /// and deliver the text through the output's own `write_str`.
    FmtWrite,
    /// `std::fmt::format(Arguments) -> String`. The nested
    /// `format_inner` helper inside `format` shares this model:
    /// it is neither generic nor `#[inline]`, so its MIR body is
    /// not encoded in the distributed alloc rlib, yet the
    /// `#[inline]` parent's body — inlined into optimized guest
    /// code — calls it directly. Its observable behaviour is
    /// exactly this builtin's (the `with_capacity` it performs
    /// first is only a size hint; the `write_fmt` text is the
    /// result).
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
    /// `std::iter::ExactSizeIterator::len(&iter)` for the
    /// slice-state iterators (see `iterator_len_modelled`):
    /// core's own body reads the remaining count out of the
    /// end pointer's address — a zero-sized element's count
    /// is the end pointer's own bits in a snapshot state,
    /// whose address word is the snapshot's, and a stepped
    /// state's front index is invisible to it — which the
    /// value model does not reproduce. The model reads the
    /// same unified bounds the modelled steps maintain (see
    /// `iter_zst_bounds`).
    IteratorLen,
    /// `slice::Iter::as_slice` / `slice::IterMut::as_slice`,
    /// and `IterMut::into_slice` / `as_mut_slice` (the same
    /// `make_slice` construction): the remaining slice as a
    /// fat reference over the state's current front pointer
    /// with the remaining count the unified bounds give (see
    /// `slice_iter_as_slice`).
    SliceIterAsSlice,
    /// `std::iter::Iterator::count(iter)`: drive the modelled
    /// `next` state transition to exhaustion and count the items.
    /// (Core's own default body is a `fold`, and several iterator
    /// types override `count` with pointer machinery over the
    /// remaining items; the observable result is the same count.)
    IteratorCount,
    /// `std::iter::Iterator::nth(&mut iter, n)`: skip `n` items
    /// through the modelled `next` state transition and take the
    /// next one. (Several iterator types override `nth` or
    /// `advance_by` with pointer machinery over the remaining
    /// items — `slice::Iter`'s zero-sized-element path reads the
    /// end pointer's own bits as the remaining count, a
    /// representation the value model does not reproduce; the
    /// observable result is the same item.)
    IteratorNth,
    /// `std::iter::Iterator::fold(iter, init, f)`: the core
    /// default body is a `next` loop applying `f` per item;
    /// modelled by driving the `next` state transition and
    /// invoking `f` synchronously per item.
    IteratorFold,
    /// `std::iter::Iterator::try_fold(iter, init, f)`: the
    /// same `next` loop as `fold`, with the `Try` branching of
    /// `f`'s result deciding whether the loop continues. The
    /// specialised bodies several iterator types override
    /// `try_fold` with (a `vec::IntoIter`'s, notably) loop
    /// over the iterator's raw pointer fields instead of
    /// calling `next`, a representation the value model does
    /// not reproduce; the observable result is the loop's.
    IteratorTryFold,
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
    /// `std::task::Waker::wake(self)` / `std::task::LocalWaker::wake(self)`:
    /// dispatch to the waker's wake implementation - the `Wake` /
    /// `LocalWake` impl method of the handle a modelled `From`
    /// conversion built the waker from, or the `RawWakerVTable`
    /// wake entry of a `from_raw` waker (see
    /// `Interpreter::exec_waker_wake`).
    WakerWake,
    /// `std::task::Waker::wake_by_ref(&self)` /
    /// `std::task::LocalWaker::wake_by_ref(&self)`: the
    /// by-reference counterpart of [`Builtin::WakerWake`].
    WakerWakeByRef,
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
    /// represented (and the capacity-overflow panic is not
    /// modelled); nothing a caller can observe changes, because
    /// `capacity` itself is uniformly the explicit
    /// `StringCapacity` error.
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
    /// `String::as_mut_vec(&mut String) -> &mut Vec<u8>`: the
    /// string's storage viewed as its byte vector. The abstract
    /// string has no byte slots a vector operation could write
    /// through, so the call rewrites the receiver's storage in
    /// place into the byte-vector element form (the form
    /// `String::into_bytes` produces) and returns the receiver
    /// reference unchanged: the place itself is the viewed
    /// storage, exactly as for `AsMut`. Vector operations then
    /// mutate the element form, and the string readers coerce
    /// that form back to text (see `string_aggregate_bytes`), so
    /// a write through the vector is a write to the string.
    StringAsMutVec,
    /// `String::leak(String) -> &'static mut str`: the string
    /// data moved into a fresh allocation nothing owns, denoted
    /// by a fat reference to it. The collected body rebuilds
    /// the slice from the vector's raw parts (a pointer transmute
    /// the conversion cannot serve); the allocation store entry
    /// is the leaked storage, and with no owning box its entry
    /// is never removed, as for a `Box::into_raw` leak.
    StringLeak,
    /// `String::into_boxed_str(String) -> Box<str>`: the string
    /// data in a fresh box allocation. The collected body moves
    /// the `vec` field out of the string value and rebuilds the
    /// box from the vector's raw parts, neither of which the
    /// string model can serve; the box over the string value
    /// itself is the same observable value.
    StringIntoBoxedStr,
    /// `OsString::into_boxed_os_str(OsString) -> Box<OsStr>`:
    /// the OS string's bytes in a fresh box allocation, the
    /// counterpart of `StringIntoBoxedStr`. The collected body
    /// rebuilds the box from the buffer's raw parts, which the
    /// OS-string model cannot serve; the box over the byte
    /// value itself is the same observable value (in this
    /// model an `OsStr` is its byte string).
    OsStringIntoBoxedOsStr,
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
    /// `String::insert(&mut String, usize, char)`: insert a
    /// char at a byte position. The collected body walks the
    /// vector's raw parts (reserve, pointer copies, `set_len`),
    /// which the string model cannot serve; the insertion
    /// itself is exact on the model's string, through the same
    /// coercions as `String::push_str` (see `modify_string`).
    StringInsert,
    /// `String::insert_str(&mut String, usize, &str)`: the
    /// string-slice counterpart of `StringInsert`.
    StringInsertStr,
    /// `String::remove(&mut String, usize) -> char`: remove
    /// and return the char at a byte position.
    StringRemove,
    /// `String::truncate(&mut String, usize)`: shorten to a
    /// byte length (a no-op past the current length).
    StringTruncate,
    /// `String::pop(&mut String) -> Option<char>`: remove and
    /// return the last char.
    StringPop,
    /// `String::clear(&mut String)`: empty the string.
    StringClear,
    /// `String::retain(&mut String, F)`: keep the chars the
    /// predicate accepts, calling it once per char in order.
    /// Dispatched before the generic argument evaluation (the
    /// predicate operand is evaluated in place, like the
    /// iterator consumers'), see `exec_string_retain`.
    StringRetain,
    /// `String::replace_range(&mut String, R, &str)`: replace
    /// a byte range with a string slice.
    StringReplaceRange,
    /// `String::split_off(&mut String, usize) -> String`:
    /// split at a byte position; the tail is the result.
    StringSplitOff,
    /// `String::reserve(&mut String, usize)` /
    /// `String::reserve_exact(&mut String, usize)`: capacity
    /// is not tracked by the abstract string model (see
    /// `StringCapacity`), so reserving is a no-op, exactly as
    /// `VecReserve` treats a vector's growth.
    StringReserve,
    /// `String::try_reserve(&mut String, usize)` /
    /// `String::try_reserve_exact(&mut String, usize)`:
    /// the fallible counterparts of `StringReserve`; with no
    /// capacity to allocate they always succeed.
    StringTryReserve,
    /// `str::len(&str)`: the byte length of the string data.
    StrLen,
    /// `str::is_empty(&str)`.
    StrIsEmpty,
    /// `str::as_bytes(&str) -> &[u8]`: a byte view of the string
    /// data.
    StrAsBytes,
    /// `str::as_bytes_mut(&mut str) -> &mut [u8]`: a mutable
    /// byte view of the string data. As for
    /// `String::as_mut_vec`, the call rewrites the denoted
    /// storage in place into the byte element form and returns
    /// a fat reference over the same storage, so byte writes
    /// land in the element form and the string readers coerce
    /// it back (a write leaving the bytes invalid UTF-8 makes
    /// the next string read fail its coercion, where native
    /// semantics leave the value undefined).
    StrAsBytesMut,
    /// `str::chars(&str)`: a `std::str::Chars` value in its
    /// real shape (see `str_chars_value`), the shape
    /// `Iterator::next` consumes for `std::str::Chars`.
    StrChars,
    /// `str::bytes(&str)`: a `std::str::Bytes` value in its
    /// real shape (see `str_bytes_value`), the shape
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
    /// `core::num::imp::int_log10::panic_for_nonpositive_argument()`:
    /// the `ilog` family's guard for a non-positive argument;
    /// raise a guest panic with core's message.
    PanicForNonpositiveArgument,
    /// `core::slice::sort::shared::smallsort::
    /// panic_on_ord_violation()`: raised when a sort's comparison
    /// function provably does not implement a total order; raise
    /// a guest panic with core's message.
    PanicOnOrdViolation,
    /// `alloc::sync::panic_arc_overflow()`: the `Arc` counter
    /// overflow guard; raise a guest panic with alloc's message.
    PanicArcOverflow,
    /// `std::cell::panic_already_borrowed(_: BorrowMutError)`:
    /// the `RefCell` double-borrow guard; raise a guest panic
    /// with core's message.
    PanicAlreadyBorrowed,
    /// `core::str::slice_error_fail(s, begin, end)`: raise a
    /// guest panic with the message core's branch order selects
    /// for the failing string range.
    SliceErrorFail,
    /// `core::slice::index::slice_index_fail(start, end, len)`:
    /// raise a guest panic with the message core's branch order
    /// selects for the failing range.
    SliceIndexFail,
    /// `core::slice::copy_from_slice_impl::len_mismatch_fail(
    /// dst_len, src_len)`: raise a guest panic with core's
    /// message (a `String` payload, as `panic!` produces).
    LenMismatchFail,
    /// `core::slice::memchr::memchr(needle, haystack) ->
    /// Option<usize>` (and `memrchr`, searching from the
    /// back): a linear scan over the haystack's elements.
    /// The native word-at-a-time implementation walks raw
    /// pointers whose distances this value model cannot
    /// express for value-held slices; the scan computes the
    /// same result.
    Memchr { reverse: bool },
    /// `std::str::pattern::StrSearcher::new(haystack, needle)`:
    /// the `&str` pattern's searcher constructor. Core defines
    /// it without `#[inline]`, so its MIR is not encoded in
    /// core's rlib and no converted body can be collected for
    /// it (the `TwoWaySearcher::new` it calls is in the same
    /// position); the model builds the searcher value in core's
    /// real field shape -- the `Empty` and `Byte` payloads
    /// exactly as core fills them, and the `TwoWay` payload
    /// with the critical-factorization preprocessing core's
    /// own constructor computes (see `str_searcher_value`) --
    /// so the converted `Searcher` bodies and the modelled
    /// matches / split steps both read a genuine state.
    StrSearcherNew,
    /// `core::slice::index::RangeError::report(self, len)`: core
    /// forwards to `slice_index_fail`; the same builtin shape.
    RangeErrorReport,
    /// `Vec::remove`'s nested `assert_failed(index, len)`: raise
    /// a guest panic with alloc's removal-index message.
    VecRemoveAssertFailed,
    /// `Vec::insert_mut`'s nested `assert_failed(index, len)`:
    /// raise a guest panic with alloc's insertion-index message.
    VecInsertAssertFailed,
    /// `std::thread::spawn(f)` / `spawn_unchecked(f)`: capture the
    /// callable as a pending thread (cooperative model, see
    /// `interp.rs`) and return its join handle.
    ThreadSpawn,
    /// `std::sync::Barrier::new(n)`: register a barrier in
    /// the interpreter's barrier table; the handle value
    /// carries its id (see `barrier_value`).
    BarrierNew,
    /// `std::sync::Barrier::wait(&self)`: arrive at the
    /// barrier; the last party releases the generation, and
    /// an earlier party's wait is the cooperative scheduling
    /// point where the remaining parties run (see
    /// `Interpreter::drain_threads`).
    BarrierWait,
    /// `std::sys::sync::mutex::futex::Mutex::lock_contended`:
    /// the sys mutex's contended acquisition: wait on the
    /// cooperative scheduler until the holder releases the
    /// lock, then take it (see
    /// `Interpreter::sys_mutex_lock_contended`).
    SysMutexLockContended,
    /// `std::sys::sync::mutex::futex::Mutex::wake`: the wake
    /// half of the sys mutex's unlock. The caller's swap has
    /// already published the unlocked word; in the cooperative
    /// model a waiter observes the word at its next activation,
    /// so there is no wait queue to notify and the call does
    /// nothing.
    SysMutexWake,
    /// `std::sys::sync::rwlock::futex::RwLock::read_contended`:
    /// the sys rwlock's contended read acquisition (see
    /// `Interpreter::sys_rwlock_read_contended`).
    SysRwLockReadContended,
    /// `std::sys::sync::rwlock::futex::RwLock::write_contended`:
    /// the sys rwlock's contended write acquisition (see
    /// `Interpreter::sys_rwlock_write_contended`).
    SysRwLockWriteContended,
    /// `std::sys::sync::rwlock::futex::RwLock::wake_writer_or_readers`:
    /// the sys rwlock's post-unlock wake routing: clear the
    /// waiting bits the caller left, handing the lock to a
    /// waiting writer (through the writer-notify word) or to
    /// the waiting readers (see
    /// `Interpreter::sys_rwlock_wake_writer_or_readers`).
    SysRwLockWakeWriterOrReaders,
    /// `std::sys::sync::futex::unix::futex_wait`: the futex
    /// wait the sys locks, `Once`, the condvar and the thread
    /// parker sleep on: wait until the word differs from the
    /// expected value (see `Interpreter::futex_wait`).
    FutexWait,
    /// `std::sys::sync::futex::unix::futex_wake`: wake one
    /// futex waiter. A waiter in this model observes the word
    /// at its next activation, so there is no wait queue to
    /// notify; the result reports that none was woken, which
    /// some platforms always report as well.
    FutexWake,
    /// `std::sys::sync::futex::unix::futex_wake_all`: wake
    /// every futex waiter (see `Builtin::FutexWake`).
    FutexWakeAll,
    /// `std::backtrace::Backtrace::capture()`: a backtrace of
    /// the current call stack when the run's backtrace
    /// settings enable capture, a disabled one otherwise
    /// (see `backtrace_value`).
    BacktraceCapture,
    /// `std::backtrace::Backtrace::force_capture()`: a
    /// backtrace of the current call stack, always captured.
    BacktraceForceCapture,
    /// `std::backtrace::Backtrace::status(&self)`: the
    /// capture status recorded in the backtrace value.
    BacktraceStatus,
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
    /// `std::thread::Thread::name(&thread)`: the thread's name
    /// (the main thread is `main`; a spawned thread carries the
    /// name its `Builder` set, or none).
    ThreadName,
    /// `std::thread::sleep(dur)` / `sleep_ms(ms)`: a cooperative
    /// scheduling point (pending threads run to completion); the
    /// passage of time itself is not observable in the value model.
    ThreadSleep,
    /// `std::thread::yield_now()`: a cooperative scheduling point.
    ThreadYield,
    /// `std::thread::park()`: block the executing thread until
    /// its park token is set by an `unpark` (see
    /// [`Builtin::ThreadUnpark`]), consuming the token. The
    /// function has no MIR body in the pinned std (the dump
    /// records it as "no MIR body available"), so the wait is
    /// modelled on the interpreter's per-thread token set.
    ThreadPark,
    /// `std::thread::park_timeout(dur)`: `park` with an upper
    /// bound on the wait. The cooperative model has no clock
    /// for the bound: the wait ends when the token is set, or
    /// when no pending thread can still set it (the point past
    /// which only the passage of time could, which the value
    /// model does not observe — the same convention as
    /// [`Builtin::ThreadSleep`]). A zero duration only polls
    /// the token.
    ThreadParkTimeout,
    /// `std::thread::Thread::unpark(&thread)`: set the target
    /// thread's park token. A converted body exists for this
    /// method, but it walks the real `Thread` struct's Parker
    /// word, which the model's id-handle thread values do not
    /// carry (the body fails over them), so the token lives in
    /// the interpreter's per-thread set instead, where
    /// [`Builtin::ThreadPark`] observes it.
    ThreadUnpark,
    /// `core::core_arch::x86::sse2::pause` (what
    /// `std::hint::spin_loop` lowers to): a spin-wait hint with
    /// no architectural effect, so the model is a no-op (in
    /// particular it is not a scheduling point: native code may
    /// spin on it without the holder of a lock making progress
    /// being observable).
    SpinPause,
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
    /// `std::panic::set_backtrace_style(style)`: record the
    /// backtrace style the guest configured; the default panic
    /// report's style resolution reads it (see
    /// `Interpreter::report_backtrace`).
    PanicSetBacktraceStyle,
    /// `std::panic::always_abort()`: from now on every panic
    /// aborts without the hook running.
    PanicAlwaysAbort,
    /// `std::panic::take_hook()`: take the panic hook, leaving
    /// the default in its place.
    PanicTakeHook,
    /// `std::panic::update_hook(f)`: replace the panic hook
    /// with a chain link that runs `f` with the previous hook
    /// and the panic info (see
    /// `Interpreter::update_panic_hook`).
    PanicUpdateHook,
    /// `test::test_main_env_args(&[TestDescAndFn]) -> ExitCode`:
    /// the libtest entry point. The compiler-generated harness
    /// main calls a crate-local wrapper of this name, so the
    /// registry matches the name's suffix.
    TestMainEnvArgs,
    /// `test::test_main_env_args_abort(&[TestDescAndFn]) ->
    /// ExitCode`: the libtest entry point of a `panic=abort`
    /// test binary, whose harness runs each test in a spawned
    /// subprocess (see the dispatch arm). Matched by suffix
    /// like the unwinding entry.
    TestMainEnvArgsAbort,
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
    /// the sys `Command` aggregate (see `command_value`), which
    /// carries the id of the builder state in the interpreter's
    /// command table (see [`CommandState`]); the sys `Command`
    /// methods below record into that state. No child process
    /// is ever started (see `ProcessSpawn`).
    ProcessCommandNew,
    /// `std::process::Command::new(program)`: the public
    /// constructor. The public `Command` is a one-field struct
    /// over the sys command, so the value is the sys command
    /// aggregate wrapped in one aggregate layer. (The public method
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
    /// `std::os::unix::process::CommandExt::uid(&mut self,
    /// u32)`: record the child user id on the command; the
    /// process host applies it at spawn.
    ProcessCommandUid,
    /// `std::os::unix::process::CommandExt::gid(&mut self,
    /// u32)`: record the child group id on the command.
    ProcessCommandGid,
    /// `std::os::unix::process::CommandExt::groups(&mut
    /// self, &[u32])`: record the child's supplementary
    /// groups on the command.
    ProcessCommandGroups,
    /// `std::os::unix::process::CommandExt::arg0(&mut self,
    /// arg) -> &mut Self` and the sys setter it forwards to,
    /// `std::sys::process::unix::common::Command::set_arg_0
    /// (&mut self, &OsStr)`: record the child's argv[0] on
    /// the command; the spawn request already carries it to
    /// the process host.
    ProcessCommandArg0,
    /// `std::os::fd::FromRawFd::from_raw_fd(i32) -> Self`:
    /// build the handle value of the `Self` type (a
    /// `File`, a `Stdio`) around a guest descriptor
    /// number: in the descriptor model the number the
    /// guest holds IS the handle's identity, so the
    /// value is the file-handle chain over the number
    /// (see `exec_from_raw_fd` in `builtins::process`).
    FromRawFd,
    /// `std::sys::process::unix::unix::Command::spawn(&mut self,
    /// Stdio, bool) -> io::Result<(Process, StdioPipes)>`:
    /// resolve the program against the command's state; a
    /// resolution failure is the `io::Error` std produces for it,
    /// and a program that would start is a named unsupported
    /// error, because the model does not start child processes.
    ProcessSpawn,
    /// `std::process::Child::wait(&mut self) ->
    /// io::Result<ExitStatus>`: wait for the child the process
    /// host started and report its exit status (see
    /// `ProcessSpawn` for how a `Child` handle comes to be).
    ProcessChildWait,
    /// `std::os::unix::process::CommandExt::exec(&mut self)
    /// -> io::Error` and the sys entry it forwards to,
    /// `std::sys::process::unix::unix::Command::exec(&mut
    /// self, Stdio) -> io::Error`: replace the guest process
    /// with the command's program. On failure the `io::Error`
    /// is returned to the guest; on success the program runs
    /// through the process host with the guest's streams and
    /// the run ends with the program's exit code, as the
    /// replaced process's own end would be reported.
    ProcessCommandExec,
    /// `std::process::Child::try_wait(&mut self) ->
    /// io::Result<Option<ExitStatus>>`: the child's exit status
    /// if it has already ended, `None` while it runs.
    ProcessChildTryWait,
    /// `std::process::Child::kill(&mut self) -> io::Result<()>`:
    /// kill the child the process host started.
    ProcessChildKill,
    /// `std::process::Child::wait_with_output(self) ->
    /// io::Result<Output>`: wait for the child and collect the
    /// streams its invocation piped.
    ProcessChildWaitWithOutput,
    /// `std::process::Child::id(&self) -> u32`: the child's
    /// process id.
    ProcessChildId,
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
    /// `std::net::ToSocketAddrs::to_socket_addrs`: name
    /// resolution into the modelled address iterator (see
    /// [`net_trait_method`]).
    ToSocketAddrs,
    /// `alloc::io::error::custom_owner_from_box(kind, error)`:
    /// build the `Custom` payload of an `io::Error::new` (its
    /// kind, the boxed trait object, and the two drop functions
    /// std stores beside them) in a fresh allocation and hand
    /// back its owner. The function's body is not exposed
    /// through Stable MIR (it assembles raw pointers and
    /// function pointers), so the model performs the same
    /// assembly: the allocation is the box the packed
    /// representation's word will name by synthetic address.
    IoCustomOwnerFromBox,
    /// `core::io::error::repr::Repr::new_custom(owner)`: pack
    /// the owner's payload into the bitpacked representation
    /// word (the allocation's synthetic address plus the
    /// `Custom` tag), registering the address -> allocation
    /// reverse lookup every later access goes through. The
    /// body is not exposed through Stable MIR.
    IoReprNewCustom,
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
    /// panic count; in this model the count is the interpreter's
    /// in-flight count (see `Interpreter::panics_in_flight`) and
    /// the catch protocol decrements it when the panic is
    /// consumed (see `Interpreter::exec_catch_unwind`), so there
    /// is nothing further to do here.
    PanicCleanup,
    /// `std::panicking::panicking()` / `std::thread::panicking()`:
    /// whether the calling thread is currently unwinding a
    /// panic. Served from the interpreter's per-thread in-flight
    /// panic count (see `Interpreter::panics_in_flight`) — the
    /// count std's runtime maintains through `__rust_start_panic`
    /// and the catch cleanup, which the interpreter's raise /
    /// catch protocol mirrors. This is the query the sync poison
    /// flags consult when a lock guard drops during unwinding.
    Panicking,
    /// `std::rt::panic_count::count_is_zero()` and
    /// `std::rt::panic_count::is_zero_slow_path()`: the runtime's
    /// own spellings of the same query — the check the collected
    /// `panicking()` body is built on in this std build, and the
    /// fallback its inlined forms call when the global count is
    /// nonzero (the global word itself is kept current by the
    /// interpreter; see `Interpreter::sync_global_panic_count`).
    PanicCountIsZero,
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
    /// the empty vector holding the requested capacity as
    /// uninitialised storage (the storage extent is the model's
    /// capacity; see the execution arm for the materialisation
    /// limit).
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
    /// `Vec::reserve_exact(&mut vec, additional)`: grow the
    /// element storage to the capacity RawVec would reach for
    /// the requirement -- amortized for `reserve`, the exact
    /// requirement for `reserve_exact` (see
    /// `vec_ensure_capacity`). Capacity in this model is the
    /// storage extent (see `vec_stored_len`), so the growth is
    /// the padding of the storage with uninitialised slots:
    /// observable exactly where the real allocation's growth is
    /// (the `capacity` and `spare_capacity_mut` answers).
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
    /// up to `usize::MAX` for a zero-sized element). On success
    /// the storage grows as the corresponding `reserve`
    /// spelling's does (see `vec_ensure_capacity`). An element type
    /// whose layout the model does not know stays an explicit
    /// unsupported error rather than a guessed verdict.
    VecTryReserve,
    /// `Vec::shrink_to_fit(&mut vec)` / `Vec::shrink_to(&mut vec,
    /// min_capacity)`: drop the spare storage past the target
    /// capacity (the length itself for `shrink_to_fit`, the
    /// larger of the length and `min_capacity` for `shrink_to`).
    /// Capacity in this model is the extent of the element
    /// storage a vector value holds (see `vec_stored_len`), so
    /// shrinking is the one capacity change the model performs;
    /// growth stays the `VecReserve` no-op. The collected bodies
    /// project the RawVec struct's fields off the abstract
    /// aggregate and fail, so the entry points are modelled here.
    VecShrink,
    /// `Vec::set_len(&mut vec, new_len)`: declare the vector's
    /// length to be `new_len`. The collected body first reads
    /// the capacity through the `RawVec` fields for its UB
    /// precondition check, a chain the value model answers in
    /// place resolution instead, so the entry point is modelled
    /// here. The storage change is `vec_with_len`'s: growing
    /// pads with the element type's uninitialized form (sparsely past the
    /// materialisation limit), and shrinking keeps the tail
    /// elements in storage as spare capacity without dropping
    /// them, exactly as the real `set_len` contract specifies —
    /// the slots stay reachable through the buffer pointer and
    /// count toward the capacity the model reports.
    VecSetLen,
    /// `Vec::as_ptr(&vec)` / `Vec::as_mut_ptr(&mut vec)`: the
    /// receiver reference retargeted at the first element. The
    /// collected bodies walk the `RawVec` / `Unique` / `NonNull`
    /// wrapper chain, which the value model does not hold; the
    /// pointer they produce is the buffer pointer, which in
    /// this model is a reference to the vector place's first
    /// element — the same target the buffer chain in place
    /// resolution denotes.
    VecAsPtr,
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
    /// `SpecExtend::spec_extend(&mut vec, iter)` for the
    /// `vec::IntoIter` instance: append every remaining item
    /// of the source iterator. The collected specialization
    /// body projects the iterator's real buffer fields
    /// (`as_slice`, inlined into it, reads the `ptr` / `end`
    /// pair), which the builtin-produced iterator state form
    /// does not carry, so it cannot run against that state;
    /// the observable semantics are the appends themselves,
    /// performed here by draining the source with the same
    /// machinery `Iterator::next` uses -- the native body
    /// moves the elements out and forgets the rest
    /// dealloc-only, which a full drain reproduces. Only this
    /// instance is served: conversion yields every other
    /// instance to its collected body (see
    /// `consumer_prefers_body` in mir_convert).
    VecSpecExtend,
    /// `SpecFromIterNested::from_iter(iter) -> Vec<T>`: build a
    /// vector from every item the iterator yields. The collected
    /// body has `Vec::with_capacity` inlined into it and drives
    /// `RawVecInner` directly, which the value model does not
    /// serve; the observable semantics are the collected items in
    /// order, produced here by stepping the iterator with the
    /// same machinery `Iterator::next` uses. This is the entry
    /// every `Iterator::collect::<Vec<_>>` reaches (the
    /// `SpecFromIter` instance delegates to it). The registry
    /// also maps `in_place_collect::from_iter_in_place` -- the
    /// specialization's in-place entry, whose body drives the
    /// source vector's raw buffer -- to this model; its
    /// observable semantics are the same items in the same
    /// order.
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
    /// `Vec::capacity(&vec)`: the extent of the element
    /// storage the value holds (see `vec_stored_len`) — the
    /// allocation this model can name. A zero-sized element
    /// type is the exception: its allocation is the dangling
    /// pointer, and std's capacity for it is `usize::MAX` by
    /// definition, not an extent. The collected body
    /// projects the `RawVec` capacity field off the abstract
    /// aggregate and fails, so the entry point is modelled
    /// here; the buffer chain's capacity slot in place
    /// resolution gives the same answer.
    VecCapacity,
    /// `std::ops::FnPtr::addr`: the function pointer's
    /// address as an integer. `addr` has a default body in
    /// core (`self.as_ptr().addr().get()`), but its callees
    /// are the lang-item methods below, which have no bodies
    /// at all, so the entry point is modelled directly with
    /// the same meaning: the synthetic address every other
    /// exposure of the same function value agrees on.
    FnPtrAddr,
    /// `std::ops::FnPtr::as_ptr` (`#[lang = "fn_ptr_as_ptr"]`):
    /// the function pointer as a raw code pointer. The
    /// compiler provides this method; it has no MIR body.
    /// The result is the address-only reference the
    /// `FnPtrToPtr` cast produces, in the `NonNull` scalar
    /// shape (a bare pointer in this model).
    FnPtrAsPtr,
    /// `std::ops::FnPtr::from_ptr`
    /// (`#[lang = "fn_ptr_from_ptr"]`): rebuild the function
    /// pointer value a code pointer denotes. The compiler
    /// provides this method; it has no MIR body. The
    /// pointer's address must be one a function pointer was
    /// exposed at in this run -- the round trip the method's
    /// safety contract requires.
    FnPtrFromPtr,
    /// `core::slice::<impl [T]>::iter(&slice)`: a slice iterator state
    /// (element snapshot + next index), the shape `Iterator::next`
    /// consumes for `std::slice::Iter`.
    SliceIter,
    /// `std::collections::BTreeSet::iter` / `BTreeMap::iter`: the
    /// borrowed iterator over an element snapshot walked from the
    /// tree's node storage, in the `slice::Iter` state layout that
    /// `Iterator::next` consumes for the `btree_set::Iter` /
    /// `btree_map::Iter` tags.
    BTreeIter,
    /// `std::collections::BTreeMap::keys`: like `BTreeIter`, over
    /// the key snapshot only (`btree_map::Keys`).
    BTreeKeys,
    /// `std::collections::BTreeMap::values`: like `BTreeIter`, over
    /// the value snapshot only (`btree_map::Values`).
    BTreeValues,
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
    /// `str::trim_matches` / `trim_start_matches` /
    /// `trim_end_matches` (`(&str, pattern) -> &str`): the
    /// pattern trim, served by the `StrContains` dispatch arm
    /// (the pattern resolution is shared). The collected
    /// bodies run core's searcher machinery, whose
    /// per-character stepping hands the nested search a
    /// scalar where the string data belongs, so the trim is
    /// computed on the resolved strings directly, like the
    /// searches beside it.
    StrTrimMatches(TrimSide),
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
    /// `core::intrinsics::vtable_size(ptr)`: the size
    /// stored in the vtable the pointer denotes -- the
    /// size of the concrete type the vtable is for.
    VtableSize,
    /// `core::intrinsics::vtable_align(ptr)`: the
    /// alignment stored in the vtable the pointer denotes.
    VtableAlign,
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
    /// `std::mem::needs_drop::<T>()` (the compiler's verdict
    /// recorded at conversion, else the types the static tag
    /// table knows).
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
    /// `intrinsics::exact_div(x, y)`: the exact quotient. The
    /// caller guarantees the division is exact and the divisor
    /// nonzero, so the model divides directly; a zero divisor
    /// still raises the usual divide-by-zero guest panic.
    ExactDiv,
    /// `core::slice::sort::stable::drift::sqrt_approx(n)`: the
    /// driftsort run-length heuristic, a pure integer formula
    /// over the input length; the model computes it directly
    /// (the native body's general sort machinery around it is
    /// what conversion cannot demand as a standalone callee).
    SqrtApprox,
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
    /// exactly the types that satisfy the property and answers
    /// the rest with a `panic_nounwind` carrying the check's
    /// message; conversion bakes that verdict per call site
    /// (see `TerminatorKind::CallBuiltin::validity_panic`), and
    /// the builtin raises it, or serves the no-op when no
    /// verdict was baked.
    AssertValidity,
    /// `std::intrinsics::overflow_checks`: whether the
    /// *instantiating* crate was compiled with overflow checks.
    /// The intrinsic's fallback body is `cfg!(debug_assertions)`
    /// as compiled into core -- a constant of the library's own
    /// build -- while the intrinsic's meaning is delayed until
    /// monomorphization, so the value is resolved from the call
    /// stack at run time instead (see
    /// `Interpreter::overflow_checks_at`).
    OverflowChecks,
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
    /// `intrinsics::raw_eq(a, b) -> bool`: byte equality of the
    /// two referenced values of the generic type, over the same
    /// byte ranges `compare_bytes` reads.
    RawEq,
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
    /// `alloc::alloc::handle_alloc_error(layout) -> !`: the
    /// allocation-failure endpoint every allocator null result
    /// funnels into. It has no Stable MIR body anywhere (the
    /// shipped alloc compiles it behind cfg flags Stable MIR
    /// does not expose), so the runtime models its dispatch:
    /// a guest `#[alloc_error_handler]`'s generated
    /// `__rust_alloc_error_handler` shim when the program
    /// defines one, std's default hook plus abort when the
    /// program links std, and alloc's `__rdl_alloc_error_handler`
    /// nounwind panic (to the guest panic handler, when the
    /// program defines one) otherwise. See
    /// `builtins::alloc_error`.
    HandleAllocError,
    /// `alloc::raw_vec::handle_error(TryReserveError) -> !`:
    /// RawVec's reserve-failure endpoint, likewise body-less
    /// in Stable MIR. `CapacityOverflow` is the
    /// `capacity overflow` panic; `AllocError { layout }`
    /// forwards to `handle_alloc_error`. See
    /// `builtins::alloc_error`.
    RawVecHandleError,
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
    /// `libc::sigaction(signum, act, oldact) -> c_int`:
    /// install the action `act` names for `signum` in the
    /// interpreter's signal table (a null `act` only
    /// queries), write the previously installed action
    /// through `oldact` when it is non-null, and answer 0.
    /// The recorded action keeps the installed structure
    /// verbatim; delivery consults only its handler field
    /// and its `SA_SIGINFO` flag (see the signal model in
    /// `interp`).
    LibcSigaction,
    /// `libc::signal(signum, handler) -> sighandler_t`:
    /// install `handler` as the disposition of `signum`
    /// in the interpreter's signal table and answer the
    /// handler it replaces (`SIG_DFL` when none was
    /// installed), or `SIG_ERR` for a signal number the
    /// host call refuses. The platform's `signal` is the
    /// BSD form: the recorded action carries the
    /// `SA_RESTART` flag and an empty mask (see
    /// `exec_libc_signal` in `builtins::signals`).
    LibcSignal,
    /// `libc::raise(sig) -> c_int`: deliver the signal to
    /// the guest itself, synchronously — the installed
    /// handler runs to completion before the call returns,
    /// an ignored signal does nothing, and a signal whose
    /// default disposition terminates the process aborts
    /// the run (see `deliver_signal` in `interp`).
    LibcRaise,
    /// `libc::atexit(func) -> c_int`: record `func` as an
    /// exit handler; the entry's exit runs the recorded
    /// handlers in reverse registration order (see
    /// `run_exit_handlers` in `interp`).
    LibcAtexit,
    /// `libc::read(fd, buf, count) -> ssize_t`: a read from
    /// a guest descriptor (see `exec_libc_read` in
    /// `builtins::libc_io`): the descriptor names an entry
    /// of the interpreter's one descriptor space (the file
    /// and socket tables), the read runs on the host object
    /// behind it, and a descriptor naming no entry -- the
    /// shape a cloexec descriptor has in a re-executed
    /// child, whose tables start empty -- fails with
    /// `EBADF` recorded in the errno cell, as the host
    /// call's failure does.
    LibcRead,
    /// `libc::write(fd, buf, count) -> ssize_t`: a write
    /// to a guest descriptor (see `exec_libc_write` in
    /// `builtins::libc_io`), the twin of `LibcRead`: the
    /// write runs on the host object behind the
    /// descriptor's table entry (descriptors 1 and 2 are
    /// the guest's captured stdout and stderr), and a
    /// descriptor naming no entry fails with `EBADF`
    /// recorded in the errno cell, as the host call's
    /// failure does.
    LibcWrite,
    /// `std::sys::io::error::unix::errno_location()
    /// -> *mut c_int` (and the libc crate's own
    /// `__errno_location`): the address of the guest's
    /// errno cell (see `exec_errno_location` in
    /// `builtins::libc_io`), the cell
    /// `std::io::Error::last_os_error` reads and the
    /// failing descriptor operations write.
    ErrnoLocation,
    /// `libc::sysconf(name) -> c_long`: the host call
    /// itself (see `exec_libc_sysconf` in
    /// `builtins::libc_sys`): the guest runs in-process,
    /// so the host's configuration values are the values
    /// the guest observes, and a failing query records
    /// the host errno in the errno cell.
    LibcSysconf,
    /// `libc::getuid() -> uid_t`: the host process's real
    /// user id (see `exec_libc_getuid` in
    /// `builtins::libc_sys`): the guest runs in-process,
    /// so the identity calls answer with this process's
    /// own credentials, as `std::process::id` answers
    /// with its own process id.
    LibcGetuid,
    /// `libc::getgid() -> gid_t`: the twin of
    /// `LibcGetuid` for the real group id.
    LibcGetgid,
    /// `libc::getpid() -> pid_t`: this process's own id,
    /// the same value `std::process::id` answers (the
    /// process host starts the guest's children from this
    /// process, so their parent id is this value).
    LibcGetpid,
    /// `libc::clearenv() -> c_int`: clear the host
    /// environment (see `exec_libc_clearenv` in
    /// `builtins::libc_sys`): the guest's environment
    /// model reads and mutates the host environment
    /// directly (see `Builtin::EnvVars`), so clearing it
    /// is the faithful execution.
    LibcClearenv,
    /// `libc::prctl(option, ...) -> c_int`: the host call
    /// itself with the variadic tail zero-padded (see
    /// `exec_libc_prctl` in `builtins::libc_sys`).
    LibcPrctl,
    /// `libc::dlopen(filename, flags) -> *mut c_void`:
    /// only the null filename (the main program) is
    /// modelled, answering a sentinel handle `dlsym`
    /// resolves against the program's own exported
    /// symbols (see `exec_libc_dlopen` in
    /// `builtins::libc_sys`); a named library is an
    /// explicit unsupported call.
    LibcDlopen,
    /// `libc::dlsym(handle, symbol) -> *mut c_void`: a
    /// lookup in the program's link exports for the
    /// main-program handle (or `RTLD_DEFAULT`), answering
    /// the exported body's function pointer, or null when
    /// the program exports no such symbol (see
    /// `exec_libc_dlsym` in `builtins::libc_sys`).
    LibcDlsym,
    /// `libc::dup(oldfd) -> c_int`: duplicate a guest
    /// descriptor in the interpreter's descriptor routing
    /// (see `exec_libc_dup` in `builtins::libc_sys`): the
    /// new descriptor denotes the same open file, socket
    /// or standard stream as the old one.
    LibcDup,
    /// `libc::dup2(oldfd, newfd) -> c_int`: reroute the
    /// guest descriptor `newfd` onto the open file,
    /// socket or standard stream `oldfd` denotes, closing
    /// what `newfd` denoted first (see `exec_libc_dup2`
    /// in `builtins::libc_sys`).
    LibcDup2,
    /// `libc::close(fd) -> c_int`: close a guest
    /// descriptor in the descriptor routing (see
    /// `exec_libc_close` in `builtins::libc_sys`): the
    /// table entry it denotes is dropped, or the standard
    /// stream it names is marked closed.
    LibcClose,
    /// `libc::pipe(fds) -> c_int`: create a host pipe,
    /// register both ends in the file table, and write
    /// the two descriptor numbers through `fds` (see
    /// `exec_libc_pipe` in `builtins::libc_sys`).
    LibcPipe,
    /// `libc::fcntl(fd, cmd, ...) -> c_int`: only
    /// `F_GETFD` is modelled (0 for an open descriptor,
    /// -1 with `EBADF` for a closed one); any other
    /// command is an explicit unsupported call (see
    /// `exec_libc_fcntl` in `builtins::libc_sys`).
    LibcFcntl,
    /// The `nftw` of glibc 2.2.5, reached through a
    /// versioned-symbol extern declaration
    /// (`#[link_name = "nftw@GLIBC_2.2.5"]`): walk the
    /// host directory tree, invoking the guest callback
    /// per entry; this version does not validate the
    /// flags word (see `exec_libc_nftw` in
    /// `builtins::libc_sys`).
    LibcNftwLegacy,
    /// The `nftw` of glibc 2.3.3 and later
    /// (`#[link_name = "nftw@GLIBC_2.3.3"]`): the twin
    /// of `LibcNftwLegacy`, except an unknown flag bit
    /// fails the call with `EINVAL` before any walking.
    LibcNftw,
    /// `core::num::imp::overflow_panic::pow(base, exp)`: the
    /// exponentiation behind the primitive `pow` methods,
    /// panicking on overflow exactly like the checked
    /// multiplications it is built from when overflow checks
    /// are on, and wrapping when they are off.
    IntPow,
    /// The primitive `funnel_shl` / `funnel_shr` methods:
    /// a funnel shift over the `(self, right)` pair, panicking
    /// when the shift amount reaches the type's width exactly
    /// when the instantiating crate was compiled with overflow
    /// checks (the methods are
    /// `#[rustc_inherit_overflow_checks]`, and the collected
    /// bodies have that predicate already folded to a constant
    /// of core's own build, so only the model resolves it from
    /// the call stack -- see
    /// `Interpreter::overflow_checks_at`).
    FunnelShift { left: bool },
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
    /// The C library's `atol(s) -> c_long`: parse the
    /// initial numeric prefix of the C string `s`, delegated
    /// to the host's libc function of the same name (see
    /// `host_atol`). Served only for callees with no
    /// converted body (see [`foreign_helper_lookup`]).
    LibcAtol,
    /// The C library's `atoll(s) -> c_longlong`, delegated
    /// like [`Builtin::LibcAtol`].
    LibcAtoll,
    /// The C library's `atoi(s) -> c_int`, delegated like
    /// [`Builtin::LibcAtol`].
    LibcAtoi,
    /// The C library's `strtol(s, endptr, base) -> c_long`,
    /// delegated to the host's libc function of the same
    /// name when `endptr` is NULL. A non-NULL `endptr` is
    /// an explicit error — writing a mid-string pointer
    /// back into guest storage is not modelled — never a
    /// silently skipped write. Served only for callees
    /// with no converted body (see
    /// [`foreign_helper_lookup`]).
    LibcStrtol,
    /// The C library's `strtoll(s, endptr, base) ->
    /// c_longlong`, delegated like [`Builtin::LibcStrtol`].
    LibcStrtoll,
    /// The C library's `strtoul(s, endptr, base) ->
    /// c_ulong`, delegated like [`Builtin::LibcStrtol`].
    LibcStrtoul,
    /// The C library's `strtoull(s, endptr, base) ->
    /// c_ulonglong`, delegated like [`Builtin::LibcStrtol`].
    LibcStrtoull,
    /// `isatty(fd)` (the libc function, reached as a body-less
    /// extern declaration): 1 when the guest descriptor names a
    /// terminal. Guest descriptors 0/1/2 are the host process's
    /// own standard streams (the guest inherits them, as a
    /// native child would); descriptors issued by the file
    /// model are answered from the host file table.
    LibcIsatty,
    /// The C library's `lgamma_r(x, signp) -> f64`: the
    /// natural log of the gamma function, with the sign of
    /// the gamma value written through `signp`. Delegated to
    /// the host's libm function of the same name: the guest's
    /// declaration names that very symbol, so the host
    /// function is the function's own semantics. Served only
    /// for callees with no converted body (see
    /// [`foreign_helper_lookup`]).
    LgammaR,
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

/// The `core::unicode::unicode_data` property modules whose
/// membership test is a skiplist lookup (`skip_search` over
/// the module's `SHORT_OFFSET_RUNS` / `OFFSETS` statics): the
/// modules core's generated `unicode_data.rs` defines in that
/// shape. A module not listed here keeps the explicit
/// no-converted-body error rather than a guessed answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnicodeProperty {
    Alphabetic,
    CaseIgnorable,
    Cf,
    CnPlanes03,
    DefaultIgnorableCodePoint,
    GraphemeExtend,
    Lt,
    N,
}

impl UnicodeProperty {
    /// The property module's full name in core: the statics
    /// the lookup reads are its `SHORT_OFFSET_RUNS` and
    /// `OFFSETS` items.
    pub fn module_path(self) -> &'static str {
        match self {
            UnicodeProperty::Alphabetic => "core::unicode::unicode_data::alphabetic",
            UnicodeProperty::CaseIgnorable => "core::unicode::unicode_data::case_ignorable",
            UnicodeProperty::Cf => "core::unicode::unicode_data::cf",
            UnicodeProperty::CnPlanes03 => "core::unicode::unicode_data::cn_planes_0_3",
            UnicodeProperty::DefaultIgnorableCodePoint => {
                "core::unicode::unicode_data::default_ignorable_code_point"
            }
            UnicodeProperty::GraphemeExtend => "core::unicode::unicode_data::grapheme_extend",
            UnicodeProperty::Lt => "core::unicode::unicode_data::lt",
            UnicodeProperty::N => "core::unicode::unicode_data::n",
        }
    }
}

/// The `std::sys::net::connection::socket` leaf operations
/// the interpreter performs on its host socket table (see
/// `exec_net_sys` in `builtins::net`). Blocking operations
/// (`Accept`, `StreamRead`, `UdpRecv`, ...) run through the
/// cooperative scheduler exactly like the mpsc waits: the
/// host sockets are nonblocking, and a would-block outcome
/// drives the other guest threads until one can proceed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetSysOp {
    ListenerBind,
    ListenerAccept,
    ListenerSocketAddr,
    ListenerDuplicate,
    ListenerSetTtl,
    ListenerTtl,
    ListenerSetOnlyV6,
    ListenerOnlyV6,
    ListenerTakeError,
    ListenerSetNonblocking,
    StreamConnect,
    StreamConnectTimeout,
    StreamSocketAddr,
    StreamPeerAddr,
    StreamRead,
    StreamReadVectored,
    StreamWrite,
    StreamWriteVectored,
    StreamPeek,
    StreamDuplicate,
    StreamShutdown,
    StreamSetNodelay,
    StreamNodelay,
    StreamSetTtl,
    StreamTtl,
    StreamTakeError,
    StreamSetNonblocking,
    StreamSetReadTimeout,
    StreamReadTimeout,
    StreamSetWriteTimeout,
    StreamWriteTimeout,
    UdpBind,
    UdpSocketAddr,
    UdpDuplicate,
    UdpRecvFrom,
    UdpSendTo,
    UdpConnect,
    UdpRecv,
    UdpSend,
    UdpSetBroadcast,
    UdpBroadcast,
    UdpSetTtl,
    UdpTtl,
    UdpTakeError,
    UdpSetNonblocking,
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
    IsNormal,
    Classify,
    ToBeBytes,
    ToLeBytes,
    ToNeBytes,
    FromBeBytes,
    FromLeBytes,
    FromNeBytes,
    Exp,
    Exp2,
    ExpM1,
    Ln,
    Ln1p,
    Log,
    Log2,
    Log10,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    SinCos,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Cbrt,
    Hypot,
    MulAdd,
    DivEuclid,
    RemEuclid,
    CopySign,
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
