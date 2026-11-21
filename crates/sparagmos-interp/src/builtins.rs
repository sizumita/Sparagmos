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
    box_alloc_pointer, channel_handle_id, rc_handle_id, synthetic_address, Frame, Interpreter,
    Resolved,
};
use crate::ir::{Body, Operand};
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
        T::Adt(def, args) => {
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
    /// `std::fmt::format(Arguments) -> String`.
    Format,
    /// `std::fmt::Arguments::from_str(&str)`.
    ArgumentsFromStr,
    /// `std::fmt::Arguments::new(&[u8; N], &[Argument; M])`.
    ArgumentsNew,
    /// `core::fmt::rt::Argument::new_*(ref)`.
    ArgumentNew(FmtTrait),
    /// `core::fmt::rt::Argument::from_usize(usize)`.
    ArgumentFromUsize,
    /// `std::ops::RangeInclusive::new(start, end)`.
    RangeInclusiveNew,
    /// `std::iter::IntoIterator::into_iter(value)`.
    IntoIter,
    /// `std::iter::Iterator::next(&mut iter)`.
    IteratorNext,
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
    /// `std::convert::From::from(value)` (modelled conversions only).
    FromFrom,
    /// `std::hint::must_use(value)`: identity.
    MustUse,
    /// `std::sys::thread_local::destructors::linux_like::register`:
    /// thread-local values are never destroyed in this model (the
    /// per-thread stores live for the whole run), so registering a
    /// destructor is a no-op -- the same observable behaviour as
    /// std's documented caveat that TLS destructors do not run for
    /// the main thread on Unix.
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
    /// `std::mem::size_of::<T>()` (types with a known layout only).
    MemSizeOf,
    /// `std::mem::align_of::<T>()` (types with a known layout only).
    MemAlignOf,
    /// `std::mem::size_of_val(&value)`.
    MemSizeOfVal,
    /// `std::mem::align_of_val(&value)`.
    MemAlignOfVal,
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
    /// `std::intrinsics::atomic_fence()`: orders other threads'
    /// accesses; there are none, so it is a no-op.
    AtomicFence,
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

/// A saturating arithmetic operation, selected by the
/// intrinsic's name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SatOp {
    Add,
    Sub,
}

/// Look a canonical callee name up in the registry.
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

pub fn lookup(canonical: &str) -> Option<Builtin> {
    use FmtTrait as F;
    Some(match canonical {
        "std::io::_print" => Builtin::Print,
        "std::io::_eprint" => Builtin::Eprint,
        "std::env::args" => Builtin::EnvArgs,
        "std::env::args_os" => Builtin::EnvArgsOs,
        "std::process::exit" => Builtin::ProcessExit,
        "std::io::stdout" => Builtin::Stdout,
        "std::io::Write::write_fmt" => Builtin::WriteFmt,
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
        "std::iter::IntoIterator::into_iter" => Builtin::IntoIter,
        "std::iter::Iterator::next" => Builtin::IteratorNext,
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
        "std::string::String::push_str" | "alloc::string::String::push_str" => {
            Builtin::StringPushStr
        }
        "std::string::String::push" | "alloc::string::String::push" => Builtin::StringPush,
        "core::str::<impl str>::len" => Builtin::StrLen,
        "core::str::<impl str>::is_empty" => Builtin::StrIsEmpty,
        "core::str::<impl str>::as_bytes" => Builtin::StrAsBytes,
        "core::str::<impl str>::chars" => Builtin::StrChars,
        "core::str::<impl str>::bytes" => Builtin::StrBytes,
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
        "std::mem::needs_drop" => Builtin::MemNeedsDrop,
        _ if canonical.starts_with("std::mem::ManuallyDrop") && canonical.ends_with(">::new") => {
            Builtin::ManuallyDropNew
        }
        "std::intrinsics::black_box" => Builtin::BlackBox,
        "std::intrinsics::ctpop" => Builtin::IntBitOp(BitOp::CtPop),
        "std::intrinsics::ctlz" => Builtin::IntBitOp(BitOp::Ctlz),
        "std::intrinsics::cttz" => Builtin::IntBitOp(BitOp::Cttz),
        "std::intrinsics::bitreverse" => Builtin::IntBitOp(BitOp::BitReverse),
        "std::intrinsics::bswap" => Builtin::IntBitOp(BitOp::Bswap),
        "std::intrinsics::saturating_add" => Builtin::SaturatingArith(SatOp::Add),
        "std::intrinsics::saturating_sub" => Builtin::SaturatingArith(SatOp::Sub),
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
        "std::intrinsics::atomic_fence" | "core::intrinsics::atomic_fence" => {
            Builtin::AtomicFence
        }
        "std::mem::zeroed" => Builtin::MemZeroed,
        "std::ptr::null" | "std::ptr::null_mut" => Builtin::PtrNull,
        "std::ptr::copy_nonoverlapping" | "std::ptr::copy" => Builtin::PtrCopy,
        "std::intrinsics::compare_bytes" | "core::intrinsics::compare_bytes" => {
            Builtin::CompareBytes
        }
        _ if canonical.starts_with("std::rc::Rc") && canonical.ends_with(">::new") => {
            Builtin::RcNew
        }
        _ if canonical.starts_with("std::sync::Arc") && canonical.ends_with(">::new") => {
            Builtin::ArcNew
        }
        _ => return float_method(canonical),
    })
}

/// Resolve a primitive float method callee
/// (`core::f64::<impl f64>::sqrt`, `std::f32::<impl f32>::floor`,
/// ...) to its builtin. The methods live partly in `core` and partly
/// in `std`; both prefixes name the same inherent impls, so either is
/// accepted for every modelled method. A float method this registry
/// does not model (e.g. `sin_cos`, which returns a tuple) stays an
/// explicit unsupported call.
fn float_method(canonical: &str) -> Option<Builtin> {
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
    Some(Builtin::FloatMethod { f32, op })
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
/// `--skip` exclusions, `--list`, `--nocapture`,
/// `--include-ignored`, and the output format (`--quiet` is
/// libtest's terse format). Flags the model does not act on
/// (thread counts, colors, log files) are accepted and
/// ignored: they cannot change a sequential run's output.
/// `--ignored` (run only the ignored tests) is not modeled,
/// and the terse run format is only honored for `--list`.
struct HarnessOpts {
    filters: Vec<String>,
    exact: bool,
    skips: Vec<String>,
    list: bool,
    nocapture: bool,
    terse: bool,
    include_ignored: bool,
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
            include_ignored: false,
        };
        let mut i = 0;
        while i < args.len() {
            let arg = &args[i];
            match arg.as_str() {
                "--list" => opts.list = true,
                "--nocapture" => opts.nocapture = true,
                "--exact" => opts.exact = true,
                "--quiet" => opts.terse = true,
                "--include-ignored" => opts.include_ignored = true,
                "--format" => {
                    i += 1;
                    if args.get(i).map(String::as_str) == Some("terse") {
                        opts.terse = true;
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

    fn selected(&self, test: &TestDesc) -> bool {
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
                let text = self.format_arguments(frames, expect_arguments(&vals[0], name)?)?;
                self.push_output(&text);
                Ok(Value::Unit)
            }
            Builtin::Eprint => {
                arity(1)?;
                let text = self.format_arguments(frames, expect_arguments(&vals[0], name)?)?;
                self.push_error_output(&text);
                Ok(Value::Unit)
            }
            Builtin::EnvArgs | Builtin::EnvArgsOs => {
                arity(0)?;
                // argv[0] is the program path the run was invoked
                // with, then the configured guest arguments, in
                // order. `Args` wraps `ArgsOs`, which wraps a
                // `vec::IntoIter` in this model's iteration-state
                // shape ([element snapshot, next index]).
                let mut items: Vec<Value> = Vec::new();
                if !self.program_path().is_empty() {
                    items.push(Value::Str(self.program_path().to_string()));
                }
                for arg in self.guest_args() {
                    items.push(Value::Str(arg.clone()));
                }
                let args_os = Value::Enum {
                    variant: 0,
                    fields: vec![owned_iter_state(items)],
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
                    self.format_arguments(frames, expect_arguments(&vals[1], name)?)?;
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
            Builtin::Format => {
                arity(1)?;
                let text = self.format_arguments(frames, expect_arguments(&vals[0], name)?)?;
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
            Builtin::ArgumentFromUsize => {
                arity(1)?;
                // core truncates the count to u16.
                let n = match vals[0] {
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
                let Some(id) = channel_handle_id(&vals[0]) else {
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
                        base: Box::new((*payload.value).clone()),
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
                let scope_ref = Value::Ref(RefTarget::Value(Box::new(Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                })));
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
                // stable-channel one (the toolchain's own harness
                // refuses `-Z` without a nightly channel). The
                // json format is likewise an unstable option and
                // is refused before listing or running.
                let raw_args: Vec<String> = self.guest_args().to_vec();
                if raw_args.iter().any(|a| a == "-Z" || a.starts_with("-Z")) {
                    self.push_error_output(
                        "error: the option `Z` is only accepted on the nightly compiler\n",
                    );
                    return Ok(test_exit_code(101));
                }
                let format_json = raw_args.iter().enumerate().any(|(j, a)| {
                    a == "--format=json"
                        || (a == "--format"
                            && raw_args.get(j + 1).map(String::as_str) == Some("json"))
                });
                if format_json {
                    self.push_error_output(
                        "error: The \"json\" format is only accepted on the nightly \
                         compiler with -Z unstable-options\n",
                    );
                    return Ok(test_exit_code(101));
                }
                let opts = HarnessOpts::parse(self.guest_args());
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
                self.push_output(&format!(
                    "\nrunning {} {}\n",
                    selected.len(),
                    if selected.len() == 1 { "test" } else { "tests" }
                ));
                for test in selected {
                    let annotation = match test.should_panic {
                        ShouldPanic::No => "",
                        _ => " - should panic",
                    };
                    if test.ignore && !opts.include_ignored {
                        ignored += 1;
                        let note = match &test.ignore_message {
                            Some(msg) => format!(", {msg}"),
                            None => String::new(),
                        };
                        self.push_output(&format!(
                            "test {}{} ... ignored{}\n",
                            test.name, annotation, note
                        ));
                        continue;
                    }
                    if opts.nocapture {
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
                                ShouldPanic::Yes => None,
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
                                        Some(Some(format!(
                                            "{cap_out}{cap_err}note: expected panic with string value,\n found non-string value: `{ty}`\n     expected substring: \"{expected}\""
                                        )))
                                    } else if text.contains(expected.as_str()) {
                                        None
                                    } else {
                                        Some(Some(format!(
                                            "{cap_out}{cap_err}note: panic did not contain expected string\n      panic message: \"{text}\"\n expected substring: \"{expected}\""
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
                            if opts.nocapture {
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
                            if let Some(body) = body {
                                failure_sections.push(format!(
                                    "\n---- {} stdout ----\n{}",
                                    test.name, body
                                ));
                            }
                            if opts.nocapture {
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
                    || self_ty.starts_with("std::iter::SkipWhile<")
                    || self_ty.starts_with("std::iter::Once<")
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
                    // `&Vec<T>` and `&[T]` iterate as `slice::Iter`:
                    // the state is the element snapshot plus the
                    // next index, the shape the `slice::iter`
                    // runtime support builds and `Iterator::next`
                    // consumes for `std::slice::Iter`.
                    let (elems, _) = self.slice_elements(frames, &value)?;
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![Value::Tuple(elems), Value::Uint(0)],
                    })
                } else if self_ty_is(self_ty, "std::env::Args")
                    || self_ty_is(self_ty, "std::env::ArgsOs")
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
                            Ok(Value::Ref(RefTarget::Value(Box::new(inner))))
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
                } else {
                    Err(unsupported(format!(
                        "Deref::deref for {self_ty} is not runtime-supported"
                    )))
                }
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
                // empty string.
                arity(1)?;
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
                            expect_arguments(&vals[0], name)?,
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
                match expect_arguments(&vals[0], name)? {
                    FmtArguments::Plain(text) => {
                        Err(InterpError::panic_str_payload(text.clone()))
                    }
                    _ => {
                        let message = self.format_arguments(
                            frames,
                            expect_arguments(&vals[0], name)?,
                        )?;
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
                        if let Ok(text) =
                            self.format_arguments(frames, expect_arguments(&arg, name)?)
                        {
                            message = format!("{message}: {text}");
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
                match expect_arguments(&vals[0], name)? {
                    FmtArguments::Plain(text) => {
                        Err(InterpError::panic_str_payload(text.clone()))
                    }
                    _ => {
                        let message = self.format_arguments(
                            frames,
                            expect_arguments(&vals[0], name)?,
                        )?;
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
                        base: Box::new(payload.value.as_ref().clone()),
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
                    Ok(Value::Enum {
                        variant: 1,
                        fields: vec![Value::Ref(RefTarget::Value(Box::new(value)))],
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
                    // `Ok` holds a `Box<T>`: the box aggregate
                    // around the payload value (the same shape
                    // `Box::new` builds), not the bare payload.
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![Value::Enum {
                            variant: 0,
                            fields: vec![value],
                        }],
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
                let eq = self.values_equal(frames, &vals[0], &vals[1])?;
                Ok(Value::Bool(if kind == Builtin::PartialEqNe {
                    !eq
                } else {
                    eq
                }))
            }
            Builtin::PartialCmp => {
                arity(2)?;
                let ord = self.values_cmp(frames, &vals[0], &vals[1], cmp_discriminants)?;
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
                let ord = self.values_cmp(frames, &vals[0], &vals[1], cmp_discriminants)?;
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
                let ord = self.values_cmp(frames, &vals[0], &vals[1], cmp_discriminants)?;
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
                match default_value_for_tag(tag) {
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
                // `Interpreter::new_box`).
                Ok(self.new_box(vals.into_iter().next().expect("arity checked")))
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
            Builtin::BoxNewUninit => {
                arity(0)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                match uninit_value_for_tag(tag) {
                    Some(pointee) => Ok(self.new_box(pointee)),
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
                    other => Err(InterpError::type_mismatch(
                        "Vec::push",
                        "vector value",
                        other.kind_name(),
                    )),
                })
            }
            Builtin::VecLen | Builtin::VecIsEmpty => {
                arity(1)?;
                let value = self.resolve_value(frames, &vals[0])?;
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
                let resolved = self.resolve_value(frames, &vals[0])?;
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
            Builtin::VecExtendTrusted => {
                arity(2)?;
                // The callee's generic arguments are the Vec's own
                // parameters followed by the iterator type, which
                // is the stepping dispatch key.
                let iter_ty = generic_args.last().map(String::as_str).unwrap_or("");
                let target = self.deref_value(frames, &vals[0])?;
                let mut fields = match self.read_resolved(frames, &target)? {
                    Value::Enum { fields, .. } => fields,
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
            Builtin::SliceIter => {
                arity(1)?;
                let (elems, _) = self.slice_elements(frames, &vals[0])?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Tuple(elems), Value::Uint(0)],
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
                let (elems, len) = self.slice_elements_ref(frames, &vals[0])?;
                if idx as u128 >= len {
                    return Err(InterpError::panic(format!(
                        "index out of bounds: the len is {len} but the index is {idx}"
                    )));
                }
                // A shared element reference: reads resolve against
                // the element value itself (the model has no separate
                // heap place for it), as for constant aggregates.
                // Only the one element is cloned out of the borrowed
                // element storage.
                let elem = elems.at(idx);
                Ok(Value::Ref(RefTarget::Value(Box::new(elem))))
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
                    Resolved::Value(_) => Err(unsupported(
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
                            Ok(Value::Ref(RefTarget::Value(Box::new(inner))))
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
            Builtin::PtrNull => {
                arity(0)?;
                // The null pointer: a dedicated reference target
                // denoting no place and no value. It is equal only
                // to itself (see `values_equal`); dereferencing it
                // is an explicit error, as a null dereference has
                // no meaning in the value model.
                Ok(Value::Ref(RefTarget::Null))
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
            Builtin::AtomicLoad => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                let current = self.read_resolved(frames, &target)?;
                Ok(unwrap_wrapper(&current).clone())
            }
            Builtin::AtomicStore => {
                arity(2)?;
                let target = self.deref_value(frames, &vals[0])?;
                let value = vals.into_iter().nth(1).expect("arity checked");
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
            Builtin::PtrCopy => {
                arity(3)?;
                let mut vals = vals.into_iter();
                let src = vals.next().expect("arity checked");
                let dst = vals.next().expect("arity checked");
                let count = vals.next().expect("arity checked");
                self.exec_copy_nonoverlapping(frames, src, dst, count)?;
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
            Builtin::FloatMethod { f32, op } => exec_float_method(name, f32, op, &vals),
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
                Ok(id_handle(id))
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
                    Resolved::Value(v) => {
                        if mutable {
                            return Err(unsupported(
                                "IntoIterator::into_iter for &mut Option over a constant aggregate is not runtime-supported",
                            ));
                        }
                        match v {
                            Value::Enum { fields, .. } if !fields.is_empty() => {
                                Value::Ref(RefTarget::Value(Box::new(fields[0].clone())))
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
            // vec::IntoIter }`: step the innermost owned-iterator
            // state and rebuild the wrappers. The yielded
            // `OsString` is the string value itself in this model,
            // so `Args::next`'s OsString-to-String conversion is
            // the identity (the arguments are always valid text
            // here, so the native panic on invalid Unicode cannot
            // arise).
            let depth = if self_ty_is(self_ty, "std::env::Args") { 2 } else { 1 };
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
        let (new_state, result) = self.next_on_state(frames, self_ty, &state)?;
        self.write_resolved(frames, target, new_state)?;
        Ok(result)
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
        let Some(item) = option_payload(item) else {
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
            Resolved::Value(_) => unreachable!("closure target is a place"),
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
            Resolved::Value(_) => Err(unsupported(
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
            let Some(item) = option_payload(item) else {
                return Ok(Some(none_value()));
            };
            if done {
                return Ok(Some(some_value(item)));
            }
            let callable = self.read_resolved(frames, &closure_target)?;
            let env_ref = self.target_ref_value(frames, &closure_target)?;
            let arg = Value::Ref(RefTarget::Value(Box::new(item.clone())));
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
        } else if self_ty.starts_with("std::array::IntoIter<")
            || self_ty.starts_with("std::vec::IntoIter<")
            || self_ty.starts_with("alloc::vec::IntoIter<")
            || self_ty.starts_with("std::option::IntoIter<")
            || self_ty.starts_with("std::slice::Iter<")
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
                // `slice::Iter` yields `&T`: the item is a shared
                // reference to the element value, the shape
                // `Index::index` produces for a shared element
                // reference (reads resolve against the element
                // itself). The other iterators sharing this state
                // shape (`array::IntoIter`, `Chars`, `Bytes`) yield
                // owned values.
                let item = if self_ty.starts_with("std::slice::Iter<") {
                    Value::Ref(RefTarget::Value(Box::new(item)))
                } else {
                    item
                };
                // An owning iterator's yielded element has moved
                // out of the iterator's storage: mark the slot
                // uninitialized in the new state, so the state's
                // drop glue (which walks the element tuple) does
                // not drop it a second time. A `slice::Iter`'s
                // elements stay owned by the slice, and `Chars` /
                // `Bytes` elements carry no glue, so their
                // snapshots keep the elements.
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
            match option_payload(item) {
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
            match option_payload(item) {
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
                let Some(x) = option_payload(item) else {
                    break;
                };
                if done {
                    result = some_value(x);
                    break;
                }
                let arg = Value::Ref(RefTarget::Value(Box::new(x.clone())));
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
                match option_payload(item) {
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
            match option_payload(item) {
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
            let depth = if self_ty_is(self_ty, "std::env::Args") { 2 } else { 1 };
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
            let inner_ref = Value::Ref(RefTarget::Value(Box::new(fields[0].clone())));
            return self.iterator_size_hint(frames, inner_ty, &inner_ref);
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
        if self_ty.starts_with("std::array::IntoIter<")
            || self_ty.starts_with("std::vec::IntoIter<")
            || self_ty.starts_with("alloc::vec::IntoIter<")
            || self_ty.starts_with("std::slice::Iter<")
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
        let a = self.resolve_ref(frames, a)?;
        let b = self.resolve_ref(frames, b)?;
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
        Ok((elems.into_vec(), len))
    }

    /// The borrowing form of [`slice_elements`]: the same elements
    /// and length, but the element storage is lent out of the
    /// resolved value wherever it is already materialised, so a
    /// consumer that needs one element (or a comparison that stops
    /// early) does not pay for a snapshot of all of them.
    fn slice_elements_ref<'x>(
        &self,
        frames: &'x [Frame],
        value: &'x Value,
    ) -> InterpResult<(SliceElems<'x>, u128)> {
        match value {
            Value::FatRef { base, len } => {
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
    /// storage). A `str` receiver yields the sub-string, with
    /// char-boundary checks as in core.
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
        let (elems, len) = self.slice_elements(frames, receiver)?;
        let len = usize::try_from(len).unwrap_or(usize::MAX);
        let (start, end) = range_bounds(&kind, &bound, len)?;
        let sub: Vec<Value> = elems[start..end].to_vec();
        let sub_len = sub.len() as u128;
        Ok(Value::FatRef {
            base: Box::new(Value::Tuple(sub)),
            len: sub_len,
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
            Resolved::Value(value) => {
                Value::Ref(RefTarget::Value(Box::new(one_field_value(value, what)?)))
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
            match self.read_resolved_at(frames, &target, i)? {
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

fn expect_arguments<'v>(value: &'v Value, name: &str) -> InterpResult<&'v FmtArguments> {
    match value {
        Value::FmtArguments(args) => Ok(args),
        other => Err(InterpError::type_mismatch(
            format!("builtin {name}"),
            "fmt::Arguments value",
            other.kind_name(),
        )),
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
                other => Ok(Value::Ref(RefTarget::Value(Box::new(other)))),
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
pub fn iterator_next_modelled(self_ty: &str) -> bool {
    self_ty.starts_with("std::ops::Range<")
        || self_ty.starts_with("std::ops::RangeInclusive<")
        || self_ty.starts_with("std::slice::Iter<")
        || self_ty.starts_with("std::array::IntoIter<")
        || self_ty.starts_with("std::vec::IntoIter<")
        || self_ty.starts_with("alloc::vec::IntoIter<")
        || self_ty_is(self_ty, "std::env::Args")
        || self_ty_is(self_ty, "std::env::ArgsOs")
        || self_ty.starts_with("std::iter::Empty<")
        || self_ty.starts_with("std::sync::mpsc::Iter<")
        || self_ty.starts_with("std::sync::mpsc::IntoIter<")
        || self_ty_is(self_ty, "std::str::Chars")
        || self_ty_is(self_ty, "core::str::Chars")
        || self_ty_is(self_ty, "std::str::Bytes")
        || self_ty_is(self_ty, "core::str::Bytes")
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
        // `slice::Iter`'s back step yields `&T`, like its front
        // step (see `iterator_next`).
        let item = if self_ty.starts_with("std::slice::Iter<") {
            Value::Ref(RefTarget::Value(Box::new(item)))
        } else {
            item
        };
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
    if self_ty_is(self_ty, "std::env::Args") || self_ty_is(self_ty, "std::env::ArgsOs") {
        // Step the innermost owned-iterator state from the back
        // and rebuild the wrappers, mirroring the front step in
        // `iterator_next`.
        let depth = if self_ty_is(self_ty, "std::env::Args") { 2 } else { 1 };
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
    if self_ty.starts_with("std::array::IntoIter<")
        || self_ty.starts_with("std::vec::IntoIter<")
        || self_ty.starts_with("alloc::vec::IntoIter<")
        || self_ty.starts_with("std::slice::Iter<")
        || self_ty_is(self_ty, "std::str::Chars")
        || self_ty_is(self_ty, "core::str::Chars")
        || self_ty_is(self_ty, "std::str::Bytes")
        || self_ty_is(self_ty, "core::str::Bytes")
    {
        return step_owned_iter_back(state, self_ty);
    }
    Err(unsupported(format!(
        "DoubleEndedIterator::next_back for {self_ty} is not runtime-supported"
    )))
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

fn option_payload(value: Value) -> Option<Value> {
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
        Resolved::Value(_) => None,
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
fn rewrap_like(current: &Value, core: Value) -> Value {
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

/// A one-field handle value carrying an id: the shape of the
/// thread model's `JoinHandle` / `Thread` / `ThreadId` values and
/// of the mpsc `Sender` / `Receiver` handle values.
fn id_handle(id: u64) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Uint(u128::from(id))],
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
}

impl<'a> SliceElems<'a> {
    /// The number of elements.
    pub(crate) fn len(&self) -> usize {
        match self {
            SliceElems::Values(v) => v.len(),
            SliceElems::Bytes(b) => b.len(),
        }
    }

    /// The element at `idx`, cloned out of the element storage (a
    /// byte element is constructed as an integer).
    pub(crate) fn at(&self, idx: usize) -> Value {
        match self {
            SliceElems::Values(v) => v[idx].clone(),
            SliceElems::Bytes(b) => Value::Uint(b[idx] as u128),
        }
    }

    /// The element at `idx`, borrowed when it is stored element
    /// storage (a byte element is a constructed temporary).
    pub(crate) fn get(&self, idx: usize) -> Cow<'_, Value> {
        match self {
            SliceElems::Values(v) => Cow::Borrowed(&v[idx]),
            SliceElems::Bytes(b) => Cow::Owned(Value::Uint(b[idx] as u128)),
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
        }
    }
}

/// The elements of an already-resolved value, borrowed where
/// possible: the borrowing form of [`value_elements`], with the
/// same results and errors.
fn elements_of<'a>(value: Cow<'a, Value>) -> InterpResult<SliceElems<'a>> {
    match value {
        Cow::Borrowed(v) => Ok(match v {
            Value::Tuple(items) => SliceElems::Values(Cow::Borrowed(items)),
            Value::Enum { fields, .. } => SliceElems::Values(Cow::Borrowed(fields)),
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
/// is projected eagerly: reads through it stay exact, and
/// writes through it are refused by the write path, as for any
/// constant.
fn subslice_ref(value: &Value, offset: usize, len: usize, name: &str) -> InterpResult<Value> {
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
        Value::Ref(RefTarget::Value(v)) => {
            let projected =
                crate::interp::project_value((**v).clone(), &[step], name, "index_mut receiver")?;
            Ok(Value::Ref(RefTarget::Value(Box::new(projected))))
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
/// `mem::zeroed`: scalars, and tuples/arrays built from scalars.
/// Anything else (an ADT whose zero bit pattern need not be a
/// valid value, a reference, ...) stays unsupported.
fn zeroed_value_for_tag(tag: &str) -> Option<Value> {
    let tag = tag.trim();
    Some(match tag {
        "bool" => Value::Bool(false),
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => Value::Int(0),
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "char" => Value::Uint(0),
        "f32" | "f64" => Value::Float(0.0),
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
            if n > 4096 {
                return None;
            }
            let elem = zeroed_value_for_tag(elem)?;
            Value::Tuple(vec![elem; n])
        }
        _ => return None,
    })
}


fn default_value_for_tag(tag: &str) -> Option<Value> {
    let tag = tag.trim();
    Some(match tag {
        "bool" => Value::Bool(false),
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => Value::Int(0),
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "char" => Value::Uint(0),
        "()" => Value::Unit,
        "std::string::String" => Value::Str(String::new()),
        _ if tag.starts_with("std::vec::Vec<") => Value::Enum {
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
    fn layout_of(&self, tag: &str) -> Option<(u128, u128)> {
        type_layout(tag).or_else(|| {
            self.program()
                .adt_layouts
                .get(tag)
                .map(|&(size, align)| (u128::from(size), u128::from(align)))
        })
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

fn template_error(what: &str) -> InterpError {
    InterpError::type_mismatch("fmt pieces template", "well-formed template", what)
}

impl<'a> Interpreter<'a> {
    /// Format a built `fmt::Arguments` value to a string.
    pub(crate) fn format_arguments(
        &self,
        frames: &[Frame],
        args: &FmtArguments,
    ) -> InterpResult<String> {
        match args {
            FmtArguments::Plain(s) => Ok(s.clone()),
            FmtArguments::Template { template, args } => {
                self.format_template(frames, template, args)
            }
        }
    }

    fn format_template(
        &self,
        frames: &[Frame],
        template: &[u8],
        args: &[Value],
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
                out.push_str(&self.format_arg(frames, arg, &FmtOptions::new())?);
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
                out.push_str(&self.format_arg(frames, arg, &options)?);
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
        &self,
        frames: &[Frame],
        arg: &Value,
        options: &FmtOptions,
    ) -> InterpResult<String> {
        match arg {
            Value::FmtArg {
                target,
                trait_,
                ty,
            } => {
                let pointee = self.resolve_value(frames, target)?;
                self.format_value(frames, &pointee, *trait_, ty, options, 0)
            }
            Value::FmtCount(_) => Err(template_error(
                "count argument used as a formatting argument",
            )),
            other => self.format_value(frames, other, FmtTrait::Display, "", options, 0),
        }
    }

    /// Format one argument value through a formatting trait.
    /// `indent` is the current pretty-print (`{:#?}`) indent level.
    fn format_value(
        &self,
        frames: &[Frame],
        value: &Value,
        trait_: FmtTrait,
        ty: &str,
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<String> {
        // References format as their pointee for every trait here.
        if matches!(value, Value::Ref(_)) {
            let pointee = self.resolve_value(frames, value)?;
            return self.format_value(frames, &pointee, trait_, ty, options, indent);
        }
        if matches!(value, Value::FatRef { .. }) {
            // A slice value: only Debug is defined for it (a list).
            if trait_ != FmtTrait::Debug {
                return Err(unsupported(format!(
                    "{trait_:?} formatting for a slice value"
                )));
            }
            let pointee = self.resolve_value(frames, value)?;
            let elem = slice_elem_tag(pointee_tag(ty));
            return self.format_debug_seq(frames, &pointee, options, indent, '[', ']', elem);
        }
        if let Value::DynRef {
            base, concrete, ..
        } = value
        {
            // A trait object's `Debug` is its concrete type's
            // `Debug` (that is the vtable entry), so format the
            // pointee with the concrete type as the tag. The other
            // traits keep their existing behaviour: their impls
            // are handwritten bodies, not the derived shape this
            // formatter models.
            if trait_ == FmtTrait::Debug {
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
                    if fields.len() == 1
                        && matches!(
                            fields[0],
                            Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. }
                        )
                    {
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

    fn format_display(
        &self,
        frames: &[Frame],
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
            Value::Str(s) => Ok(pad_str(s, options)),
            Value::FmtArguments(inner) => {
                let text = self.format_arguments(frames, inner)?;
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
        &self,
        frames: &[Frame],
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
                let text = self.format_arguments(frames, inner)?;
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
        &self,
        frames: &[Frame],
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
        &self,
        frames: &[Frame],
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
        let Some(shape) = self.program().adt_debug_shapes.get(tag) else {
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
        &self,
        frames: &[Frame],
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
        &self,
        frames: &[Frame],
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
