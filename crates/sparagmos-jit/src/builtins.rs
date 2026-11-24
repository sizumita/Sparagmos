//! The JIT's builtin layer: name resolution and host-side semantics
//! for calls whose bodies exist only in std/core.
//!
//! The interpreter executes guest functions from converted bodies.
//! Some callees have no body anywhere in the guest program — they
//! *are* the standard library (`std::io::_print`, the `fmt::Arguments`
//! constructors, iterator entry points over std types, primitive
//! float methods, ...). The interpreter serves those calls from its
//! builtin registry (`sparagmos-interp`'s `builtins` module); this
//! module is the JIT path's equivalent of that registry — the main
//! path for these callees, not a fallback: there is no other `_print`
//! to fall back to.
//!
//! # How a builtin call executes
//!
//! A call to a function that is not part of the program is resolved
//! by name ([`canonical_name`] + [`lookup`], the same resolution the
//! interpreter's conversion performs). The MIR front end then lowers
//! the call in one of two ways:
//!
//! * Constructors and identity conversions whose value fits the
//!   slot model (`RangeInclusive::new`, `IntoIterator::into_iter`
//!   over the range types) are lowered inline, as the aggregate
//!   construction / copy they are.
//! * Everything else lowers to a call to a *host function* in this
//!   module (the backend's `HostCall` instruction): a plain
//!   `extern "C"` function that receives the program's
//!   [`BuiltinCtx`] plus the call's argument words and returns up to
//!   two result words. Host functions are how the generated code
//!   reaches semantics that live naturally on the host side: the
//!   `core::fmt` formatting engine and the primitive float methods
//!   (the host's own `std` float methods and float formatting are
//!   the very code the guest's std ships, so results match exactly,
//!   as they do in the interpreter, which computes them the same
//!   way).
//!
//! # Value representation
//!
//! The opaque values the builtin layer owns are single words in the
//! slot model:
//!
//! * A `core::fmt::rt::Argument` and a `std::fmt::Arguments` are
//!   *handles* into the context's store (an index + 1, so 0 is never
//!   a valid handle). The store entry for an argument records the
//!   descriptor of its pointee type (interned at compile time, see
//!   [`FmtDescriptor`]) and the pointee's address: the argument is
//!   read through that address when the arguments are formatted —
//!   as the real `format_args!` reads through it at print time.
//!   The layout the front end assigns these two ADT types is the
//!   one-leaf handle shape, so arrays of arguments and by-value
//!   moves work with the ordinary slot machinery.
//! * A `std::ops::Range` / `RangeInclusive` value is its real field
//!   layout (`[start, end]` / `[start, end, exhausted]`), the shape
//!   the interpreter's iterator state uses; `Iterator::next` is a
//!   host function that steps that state in guest memory through
//!   the `&mut` pointer it is handed.
//! * A `std::string::String` value is its real layout — the
//!   (pointer, capacity, length) triple — with the pointer
//!   naming a real host allocation: the `__rust_alloc` family is
//!   served by the host allocator, and the string-producing
//!   builtins (`From::from` from a `str`, `fmt::format`, the
//!   case conversions) allocate the buffer and write the triple
//!   through the destination's address. The guest's own
//!   converted drop bodies read the triple's fields and free
//!   the buffer through `__rust_dealloc`, so construction and
//!   destruction agree by construction.
//!
//! Anything the layer does not model — a callee name with no
//! registry entry, an iterator over a type whose state is not the
//! value itself, a formatting argument of a type the formatter
//! cannot read exactly — stays an explicit unsupported error naming
//! the construct, exactly as on the interpreter path. A host-side
//! failure discovered only at run time (a malformed fmt template,
//! which rustc's own templates never are) raises a guest panic
//! through the program's panic path rather than producing output
//! that might be wrong.

// ---------------------------------------------------------------------------
// Callee names
// ---------------------------------------------------------------------------

/// The canonical form of a callee path: `::<...>` groups that contain
/// only lifetimes (`<'a>`, `<'_>`, `<'a, 'b>`) are removed, so the
/// same function has one name regardless of how the driver rendered
/// its lifetime arguments. Groups with type content (`<Idx>`,
/// `<T, E>`) are kept — dispatch never keys on them, but diagnostics
/// stay precise. (The interpreter's registry canonicalises the same
/// way; the two must agree, because the driver renders one name
/// space for both paths.)
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

// ---------------------------------------------------------------------------
// The builtin registry
// ---------------------------------------------------------------------------

/// Which formatting trait an `rt::Argument` was built for; mirrors
/// the `new_*` constructor family in `core::fmt::rt` (and the
/// interpreter's `FmtTrait` of the same name).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FmtTrait {
    Display,
    Debug,
    /// `new_debug_noop`: formats as the empty string.
    DebugNoop,
    Octal,
    LowerHex,
    UpperHex,
    Pointer,
    Binary,
    LowerExp,
    UpperExp,
}

/// One primitive float method operation, selected by the method name
/// of an `f32` / `f64` inherent impl call (the interpreter's
/// `FloatOp` table, restricted to the two widths the JIT computes).
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

impl FloatOp {
    /// The code the front end bakes into a float-method host call.
    pub fn code(self) -> i64 {
        self as i64
    }

    fn from_code(code: i64) -> Option<FloatOp> {
        use FloatOp as O;
        Some(match code {
            0 => O::IsNan,
            1 => O::IsInfinite,
            2 => O::IsFinite,
            3 => O::IsSignPositive,
            4 => O::IsSignNegative,
            5 => O::Abs,
            6 => O::Sqrt,
            7 => O::Floor,
            8 => O::Ceil,
            9 => O::Round,
            10 => O::RoundTiesEven,
            11 => O::Trunc,
            12 => O::Fract,
            13 => O::Recip,
            14 => O::Signum,
            15 => O::ToDegrees,
            16 => O::ToRadians,
            17 => O::Powi,
            18 => O::Powf,
            19 => O::Min,
            20 => O::Max,
            21 => O::ToBits,
            22 => O::FromBits,
            23 => O::Exp,
            24 => O::Exp2,
            25 => O::ExpM1,
            26 => O::Ln,
            27 => O::Ln1p,
            28 => O::Log,
            29 => O::Log2,
            30 => O::Log10,
            31 => O::Sin,
            32 => O::Cos,
            33 => O::Tan,
            34 => O::Asin,
            35 => O::Acos,
            36 => O::Atan,
            37 => O::Atan2,
            38 => O::SinCos,
            39 => O::Sinh,
            40 => O::Cosh,
            41 => O::Tanh,
            42 => O::Asinh,
            43 => O::Acosh,
            44 => O::Atanh,
            45 => O::Cbrt,
            46 => O::Hypot,
            47 => O::MulAdd,
            48 => O::DivEuclid,
            49 => O::RemEuclid,
            50 => O::CopySign,
            _ => return None,
        })
    }
}

/// A callee the JIT's builtin layer serves. The front end consults
/// [`lookup`] for every call whose target is not part of the
/// program; a callee with no entry here stays an explicit
/// unsupported-call error naming the function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Builtin {
    /// `core::fmt::rt::Argument::new_*(ref)`.
    ArgumentNew(FmtTrait),
    /// `core::fmt::rt::Argument::from_usize(&usize)`.
    ArgumentFromUsize,
    /// `std::fmt::Arguments::new(&[u8; N], &[Argument; M])`.
    ArgumentsNew,
    /// `std::fmt::Arguments::from_str(&str)`.
    ArgumentsFromStr,
    /// `std::io::_print(Arguments)`: format and append to guest stdout.
    Print,
    /// `std::io::_eprint(Arguments)`: format and append to guest stderr.
    Eprint,
    /// `std::ops::RangeInclusive::new(start, end)`.
    RangeInclusiveNew,
    /// `std::iter::IntoIterator::into_iter(value)` (range types only:
    /// the types that are their own iterator state).
    IntoIter,
    /// `std::iter::Iterator::next(&mut iter)` (range types only).
    IteratorNext,
    /// A primitive `f32` / `f64` inherent method.
    FloatMethod { f32: bool, op: FloatOp },
    /// `PartialEq::eq(&a, &b)`: semantic (derived) value equality,
    /// as the interpreter's builtin of the same name computes it.
    PartialEqEq,
    /// `PartialEq::ne(&a, &b)`.
    PartialEqNe,
    /// `alloc::alloc::__rust_alloc(size, align) -> *mut u8`: a
    /// real host allocation; the guest's own drop path frees it
    /// through `__rust_dealloc` with the same layout.
    RustAlloc,
    /// `alloc::alloc::__rust_alloc_zeroed(size, align)`.
    RustAllocZeroed,
    /// `alloc::alloc::__rust_dealloc(ptr, size, align)`.
    RustDealloc,
    /// `alloc::alloc::__rust_realloc(ptr, old_size, align, new_size)`.
    RustRealloc,
    /// `std::fmt::format(Arguments) -> String`.
    Format,
    /// `str::to_uppercase(&str) -> String`.
    StrToUppercase,
    /// `str::to_lowercase(&str) -> String`.
    StrToLowercase,
    /// `std::convert::From::from(value)` (modelled conversions:
    /// string construction from a `str` source, and the identity
    /// conversions the interpreter's builtin also serves).
    FromFrom,
    /// `std::convert::Into::into(value)`: `From::from` with the
    /// generic arguments in the other order.
    IntoInto,
    /// `std::ops::Deref::deref(&value)` for the buffer types
    /// (`String`, `Vec<T>`): the (pointer, length) pair read out
    /// of the value's own triple.
    Deref,
    /// `std::hint::must_use(value)`: the identity.
    MustUse,
    /// An integer bit intrinsic (`ctpop`, `ctlz`, `cttz`,
    /// `bitreverse`, `bswap`); the payload is the operation code
    /// the host function dispatches on (see `host_bit_op`).
    IntBitOp(u8),
    /// `core::panicking::panic_nounwind_fmt(Arguments, bool)`:
    /// format the arguments and raise the guest panic with that
    /// message (the call never returns in core either).
    PanicNounwindFmt,
    /// `String::new() -> String`: the empty triple.
    StringNew,
    /// `String::len(&String) -> usize`: the triple's length word.
    StringLen,
    /// `String::is_empty(&String) -> bool`.
    StringIsEmpty,
    /// `String::capacity(&String) -> usize`: the triple's
    /// capacity word.
    StringCapacity,
    /// `String::push_str(&mut String, &str)`.
    StringPushStr,
    /// `String::push(&mut String, char)`.
    StringPush,
    /// `str::len(&str) -> usize`: the pair's length word.
    StrLen,
    /// `str::is_empty(&str) -> bool`.
    StrIsEmpty,
    /// A `str` pattern search (`starts_with`, `ends_with`,
    /// `contains` with a `&str` pattern); the payload is the
    /// operation code the host function dispatches on (see
    /// `host_str_op`).
    StrOp(u8),
}

/// Look a canonical callee name up in the registry.
pub fn lookup(canonical: &str) -> Option<Builtin> {
    use FmtTrait as F;
    Some(match canonical {
        "std::io::_print" => Builtin::Print,
        "std::io::_eprint" => Builtin::Eprint,
        "std::fmt::Arguments::from_str" | "std::fmt::Arguments::from_str_nonconst" => {
            Builtin::ArgumentsFromStr
        }
        "std::fmt::Arguments::new" => Builtin::ArgumentsNew,
        "alloc::alloc::__rust_alloc" => Builtin::RustAlloc,
        "alloc::alloc::__rust_alloc_zeroed" => Builtin::RustAllocZeroed,
        "alloc::alloc::__rust_dealloc" => Builtin::RustDealloc,
        "alloc::alloc::__rust_realloc" => Builtin::RustRealloc,
        "std::fmt::format" => Builtin::Format,
        "std::convert::From::from" => Builtin::FromFrom,
        "std::convert::Into::into" => Builtin::IntoInto,
        "std::ops::Deref::deref" => Builtin::Deref,
        "std::hint::must_use" => Builtin::MustUse,
        "core::panicking::panic_nounwind_fmt" => Builtin::PanicNounwindFmt,
        "std::intrinsics::ctpop" | "core::intrinsics::ctpop" => Builtin::IntBitOp(0),
        "std::intrinsics::ctlz" | "core::intrinsics::ctlz" => Builtin::IntBitOp(1),
        "std::intrinsics::cttz" | "core::intrinsics::cttz" => Builtin::IntBitOp(2),
        "std::intrinsics::ctlz_nonzero" | "core::intrinsics::ctlz_nonzero" => {
            Builtin::IntBitOp(1)
        }
        "std::intrinsics::cttz_nonzero" | "core::intrinsics::cttz_nonzero" => {
            Builtin::IntBitOp(2)
        }
        "std::intrinsics::bitreverse" | "core::intrinsics::bitreverse" => {
            Builtin::IntBitOp(3)
        }
        "std::intrinsics::bswap" | "core::intrinsics::bswap" => Builtin::IntBitOp(4),
        "std::string::String::new" | "alloc::string::String::new" => Builtin::StringNew,
        "std::string::String::len" | "alloc::string::String::len" => Builtin::StringLen,
        "std::string::String::is_empty" | "alloc::string::String::is_empty" => {
            Builtin::StringIsEmpty
        }
        "std::string::String::capacity" | "alloc::string::String::capacity" => {
            Builtin::StringCapacity
        }
        "std::string::String::push_str" | "alloc::string::String::push_str" => {
            Builtin::StringPushStr
        }
        "std::string::String::push" | "alloc::string::String::push" => {
            Builtin::StringPush
        }
        // `String::as_str` / `String::as_bytes` are the
        // dereference to the buffer's pair, the `Deref` model's
        // own operation.
        "std::string::String::as_str" | "alloc::string::String::as_str" => {
            Builtin::Deref
        }
        "std::string::String::as_bytes" | "alloc::string::String::as_bytes" => {
            Builtin::Deref
        }
        "core::str::<impl str>::len" => Builtin::StrLen,
        "core::str::<impl str>::is_empty" => Builtin::StrIsEmpty,
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
        _ if canonical.contains("RangeInclusive") && canonical.ends_with("::new") => {
            Builtin::RangeInclusiveNew
        }
        _ if canonical.ends_with("::PartialEq::eq") => Builtin::PartialEqEq,
        _ if canonical.ends_with("::PartialEq::ne") => Builtin::PartialEqNe,
        _ if canonical.ends_with("::to_uppercase") => Builtin::StrToUppercase,
        _ if canonical.ends_with("::to_lowercase") => Builtin::StrToLowercase,
        _ if canonical.starts_with("core::str::<impl") && canonical.ends_with(">::starts_with") => {
            Builtin::StrOp(0)
        }
        _ if canonical.starts_with("core::str::<impl") && canonical.ends_with(">::ends_with") => {
            Builtin::StrOp(1)
        }
        _ if canonical.starts_with("core::str::<impl") && canonical.ends_with(">::contains") => {
            Builtin::StrOp(2)
        }
        _ => return float_method(canonical),
    })
}

/// Resolve a primitive float method callee
/// (`core::f64::<impl f64>::sqrt`, `std::f32::<impl f32>::floor`,
/// ...) to its builtin. The methods live partly in `core` and partly
/// in `std`; both prefixes name the same inherent impls, so either
/// is accepted for every modelled method. A float method this
/// registry does not model stays an explicit unsupported call.
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
        _ => return None,
    };
    Some(op)
}

// ---------------------------------------------------------------------------
// Compile-time tables
// ---------------------------------------------------------------------------

/// The host-side shape of a fmt argument's pointee type: what the
/// formatter needs to read the value out of guest memory and render
/// it exactly. The front end builds one of these from the MIR type
/// of a `new_*` constructor's generic argument; a pointee type with
/// no entry here (an ADT, a tuple, a slice, ...) cannot be formatted
/// by this layer, and the front end rejects the descriptor at
/// compile time with an explicit error rather than letting the call
/// through to produce a guessed rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FmtTy {
    /// A signed integer of this bit width. The slot word holds the
    /// canonical (sign-extended) value.
    Int { bits: u32 },
    /// An unsigned integer of this bit width (also `usize` /
    /// `isize` at 64). The slot word holds the canonical
    /// (zero-extended) value.
    Uint { bits: u32 },
    Bool,
    Char,
    F32,
    F64,
    /// A `str` value: a (pointer, length) pair, stored at the
    /// pointee address (first word the pointer, second the length)
    /// or carried directly as a fat constructor argument.
    Str,
    /// A `std::string::String` value: the (pointer, capacity,
    /// length) triple at the pointee address, in the aggregate's
    /// leaf order; the formatted content is the `length` bytes at
    /// `pointer` — the same bytes `String`'s own `Display` / `Debug`
    /// impls present.
    OwnedStr,
    /// A reference to another formattable value: the word at the
    /// pointee address is the inner value's address (or, for a fat
    /// inner type, the inner pair itself).
    Ref(Box<FmtTy>),
    /// The pointee is never read (`new_debug_noop`).
    Ignored,
}

/// One interned fmt argument descriptor: the formatting trait a
/// `new_*` constructor was called for, plus the pointee's shape.
/// The front end interns descriptors per program and bakes the
/// index into the host call, the same treatment the panic message
/// table gets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FmtDescriptor {
    pub trait_: FmtTrait,
    pub ty: FmtTy,
}

/// Everything the front end interns for the builtin layer while
/// lowering a program, handed to the program layer at compile time:
/// fmt descriptors and literal strings (`Arguments::from_str`
/// constants). Static byte data (fmt template constants) is not
/// part of these tables: it goes through the program's statics
/// table as a [`crate::program::StaticBlob::GuestBytes`] blob, the
/// one interning mechanism for static data.
#[derive(Debug, Default, Clone)]
pub struct BuiltinTables {
    pub descriptors: Vec<FmtDescriptor>,
    pub strings: Vec<String>,
}

// ---------------------------------------------------------------------------
// Host-side state
// ---------------------------------------------------------------------------

/// One value in the builtin store (see the module docs): a built
/// fmt argument, a count argument, or a built `fmt::Arguments`.
enum StoreEntry {
    FmtArg {
        /// Index into the context's descriptor table.
        desc: u32,
        /// The pointee's address (a thin constructor argument), or
        /// the string data address for a fat `str` argument.
        ptr: i64,
        /// The second argument word: the length of a fat `str`
        /// argument, unused (0) otherwise.
        second: i64,
        /// The constructor argument was a fat reference (the value
        /// is the pair `(ptr, second)` itself, not a pair stored at
        /// `ptr`).
        fat: bool,
    },
    /// An `rt::Argument` count (`Argument::from_usize`).
    FmtCount(u16),
    Arguments(ArgumentsData),
}

/// A built `std::fmt::Arguments` value (the interpreter's
/// `FmtArguments`, with argument handles in place of values).
#[derive(Clone)]
enum ArgumentsData {
    /// `Arguments::from_str`: printed verbatim.
    Plain(String),
    /// `Arguments::new`: the encoded pieces template (see the
    /// parser below) plus the argument handles, snapshotted from
    /// the argument array when `new` runs, as the real call
    /// snapshots the array reference's elements.
    Template { template: Vec<u8>, args: Vec<i64> },
}

/// The two result words of a host call, in the pair-return
/// registers (RAX, then RDX) — the `#[repr(C)]` two-`i64` shape
/// classifies as two INTEGER eightbytes under System V, exactly
/// like the program layer's own result pair.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HostRet {
    pub first: i64,
    pub second: i64,
}

impl HostRet {
    fn one(first: i64) -> Self {
        HostRet { first, second: 0 }
    }
    fn pair(first: i64, second: i64) -> Self {
        HostRet { first, second }
    }
}

/// The host-side state generated code reaches through host calls:
/// the builtin store, the guest's captured stdout / stderr, and the
/// compile-time tables. Owned by the compiled program (boxed, so
/// its address — which the generated code receives on every host
/// call — is stable), and reset at every Rust-side entry call, like
/// the program's panic context.
///
/// The two pointer fields are wired by the program layer after the
/// context is boxed: they name the program's panic context and
/// panic scratch buffer, so a host function that discovers it
/// cannot serve a call (a malformed fmt template) can raise a guest
/// panic through the usual dynamic-message path instead of
/// guessing. They are null only in unit tests that drive the host
/// functions directly.
pub struct BuiltinCtx {
    descriptors: Vec<FmtDescriptor>,
    strings: Vec<String>,
    store: Vec<StoreEntry>,
    stdout: String,
    stderr: String,
    panic_flag: *mut u64,
    scratch: *mut u8,
}

impl BuiltinCtx {
    pub fn new(tables: BuiltinTables) -> Self {
        BuiltinCtx {
            descriptors: tables.descriptors,
            strings: tables.strings,
            store: Vec::new(),
            stdout: String::new(),
            stderr: String::new(),
            panic_flag: std::ptr::null_mut(),
            scratch: std::ptr::null_mut(),
        }
    }

    /// Wire the panic path (see the type docs). Called by the
    /// program layer once both contexts are boxed and their
    /// addresses are final.
    pub fn wire_panic(&mut self, panic_flag: *mut u64, scratch: *mut u8) {
        self.panic_flag = panic_flag;
        self.scratch = scratch;
    }

    /// Clear the store and the captured output (a fresh run).
    pub fn reset(&mut self) {
        self.store.clear();
        self.stdout.clear();
        self.stderr.clear();
    }

    /// The guest's captured (stdout, stderr) so far.
    pub fn output(&self) -> (String, String) {
        (self.stdout.clone(), self.stderr.clone())
    }

    /// Record a new store entry and return its handle.
    fn push(&mut self, entry: StoreEntry) -> i64 {
        self.store.push(entry);
        self.store.len() as i64
    }

    /// The store entry a handle names, if it is a live handle.
    fn entry(&self, handle: i64) -> Option<&StoreEntry> {
        if handle <= 0 {
            return None;
        }
        self.store.get((handle - 1) as usize)
    }

    /// Raise a guest panic with a run-time message, through the
    /// program's dynamic-message panic path (the scratch buffer
    /// contract the program layer's own formatters use), and report
    /// dummy result words; the `CheckPanic` the front end emits
    /// after every host call propagates the panic before they are
    /// read. Without a wired panic path (unit tests) the message is
    /// simply dropped and the dummy words returned.
    fn fail(&mut self, msg: &str) -> HostRet {
        if !self.panic_flag.is_null() {
            let bytes = msg.as_bytes();
            let n = bytes.len().min(120);
            // Safety: `scratch` is the program's 128-byte scratch
            // buffer (8 length bytes + message), wired by the
            // program layer; `panic_flag` is the program's panic
            // context (flag word, then message-index word). Both
            // outlive every host call of the owning program.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (n as u64).to_le_bytes().as_ptr(),
                    self.scratch,
                    8,
                );
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.scratch.add(8), n);
                sparagmos_runtime::ffi::sparagmos_panic(self.scratch.add(8), n);
                *self.panic_flag = 1;
                *self.panic_flag.add(1) = u64::MAX;
            }
        }
        HostRet::one(0)
    }
}

// ---------------------------------------------------------------------------
// Host functions
// ---------------------------------------------------------------------------
//
// Every host function has the same signature: the builtin context,
// then five argument words (unused ones are passed as 0), and a
// two-word result. The front end's host-call lowering guarantees at
// most five argument words per builtin; the indices below are the
// host table the backend calls through.

/// Host table index of [`host_argument_new`].
pub const HOST_ARGUMENT_NEW: u32 = 0;
/// Host table index of [`host_argument_from_usize`].
pub const HOST_ARGUMENT_FROM_USIZE: u32 = 1;
/// Host table index of [`host_arguments_new`].
pub const HOST_ARGUMENTS_NEW: u32 = 2;
/// Host table index of [`host_arguments_from_str_const`].
pub const HOST_ARGUMENTS_FROM_STR_CONST: u32 = 3;
/// Host table index of [`host_arguments_from_str_rt`].
pub const HOST_ARGUMENTS_FROM_STR_RT: u32 = 4;
/// Host table index of [`host_print`].
pub const HOST_PRINT: u32 = 5;
/// Host table index of [`host_next_range`].
pub const HOST_NEXT_RANGE: u32 = 6;
/// Host table index of [`host_next_range_inclusive`].
pub const HOST_NEXT_RANGE_INCLUSIVE: u32 = 7;
/// Host table index of [`host_float_method`].
pub const HOST_FLOAT_METHOD: u32 = 8;
/// Host table index of [`host_rust_alloc`].
pub const HOST_RUST_ALLOC: u32 = 9;
/// Host table index of [`host_rust_alloc_zeroed`].
pub const HOST_RUST_ALLOC_ZEROED: u32 = 10;
/// Host table index of [`host_rust_dealloc`].
pub const HOST_RUST_DEALLOC: u32 = 11;
/// Host table index of [`host_rust_realloc`].
pub const HOST_RUST_REALLOC: u32 = 12;
/// Host table index of [`host_string_from_str`].
pub const HOST_STRING_FROM_STR: u32 = 13;
/// Host table index of [`host_string_format`].
pub const HOST_STRING_FORMAT: u32 = 14;
/// Host table index of [`host_str_to_uppercase`].
pub const HOST_STR_TO_UPPERCASE: u32 = 15;
/// Host table index of [`host_str_to_lowercase`].
pub const HOST_STR_TO_LOWERCASE: u32 = 16;
/// Host table index of [`host_deref_buffer`].
pub const HOST_DEREF_BUFFER: u32 = 17;
/// Host table index of [`host_bit_op`].
pub const HOST_BIT_OP: u32 = 18;
/// Host table index of [`host_panic_nounwind_fmt`].
pub const HOST_PANIC_NOUNWIND_FMT: u32 = 19;
/// Host table index of [`host_arguments_from_struct`].
pub const HOST_ARGUMENTS_FROM_STRUCT: u32 = 20;
/// Host table index of [`host_string_len`].
pub const HOST_STRING_LEN: u32 = 21;
/// Host table index of [`host_string_capacity`].
pub const HOST_STRING_CAPACITY: u32 = 22;
/// Host table index of [`host_string_push_str`].
pub const HOST_STRING_PUSH_STR: u32 = 23;
/// Host table index of [`host_string_push`].
pub const HOST_STRING_PUSH: u32 = 24;
/// Host table index of [`host_str_op`].
pub const HOST_STR_OP: u32 = 25;
/// Host table index of [`host_str_eq`].
pub const HOST_STR_EQ: u32 = 26;

/// The host function table, in [`HOST_*`] index order. The program
/// layer copies these addresses into the backend's program context.
pub fn host_fns() -> Vec<usize> {
    vec![
        host_argument_new as usize,
        host_argument_from_usize as usize,
        host_arguments_new as usize,
        host_arguments_from_str_const as usize,
        host_arguments_from_str_rt as usize,
        host_print as usize,
        host_next_range as usize,
        host_next_range_inclusive as usize,
        host_float_method as usize,
        host_rust_alloc as usize,
        host_rust_alloc_zeroed as usize,
        host_rust_dealloc as usize,
        host_rust_realloc as usize,
        host_string_from_str as usize,
        host_string_format as usize,
        host_str_to_uppercase as usize,
        host_str_to_lowercase as usize,
        host_deref_buffer as usize,
        host_bit_op as usize,
        host_panic_nounwind_fmt as usize,
        host_arguments_from_struct as usize,
        host_string_len as usize,
        host_string_capacity as usize,
        host_string_push_str as usize,
        host_string_push as usize,
        host_str_op as usize,
        host_str_eq as usize,
    ]
}

/// Read the word at guest address `addr`. Guest addresses are real
/// host addresses in this execution model (a pointer is the address
/// of a frame slot or of memory the guest itself addressed), and
/// the front end only ever hands a host function an address the
/// guest program itself formed for the value being read.
///
/// # Safety
/// `addr` must name a readable 8-byte word of live guest storage.
unsafe fn load_word(addr: i64) -> i64 {
    unsafe { *(addr as *const i64) }
}

/// Write `v` as the word at guest address `addr` (see
/// [`load_word`] for the addressing model).
///
/// # Safety
/// `addr` must name a writable 8-byte word of live guest storage.
unsafe fn store_word(addr: i64, v: i64) {
    unsafe { *(addr as *mut i64) = v }
}

/// `Argument::new_*(ref) -> Argument`: record the argument's
/// descriptor and pointee address as a store entry and return its
/// handle. Arguments: (pointee word, second word, descriptor
/// index, fat flag, _). The pointee is *not* read here — as in
/// core, the value is read when the arguments are formatted.
extern "C" fn host_argument_new(
    ctx: *mut BuiltinCtx,
    ptr: i64,
    second: i64,
    desc: i64,
    fat: i64,
    _pad: i64,
) -> HostRet {
    // Safety: the generated code passes the owning program's
    // context, which outlives the call.
    let ctx = unsafe { &mut *ctx };
    if desc < 0 || desc as usize >= ctx.descriptors.len() {
        return ctx.fail("fmt argument descriptor out of range");
    }
    let handle = ctx.push(StoreEntry::FmtArg {
        desc: desc as u32,
        ptr,
        second,
        fat: fat != 0,
    });
    HostRet::one(handle)
}

/// `Argument::from_usize(&usize) -> Argument`: the count argument.
/// The argument word is the reference; the count is the word it
/// points at, truncated to `u16` as core truncates it. Arguments:
/// (count reference, _, _, _, _).
extern "C" fn host_argument_from_usize(
    ctx: *mut BuiltinCtx,
    arg: i64,
    _a: i64,
    _b: i64,
    _c: i64,
    _d: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    // Safety: the argument is the guest's own `&usize`.
    let n = unsafe { load_word(arg) } as u64 as u16;
    let handle = ctx.push(StoreEntry::FmtCount(n));
    HostRet::one(handle)
}

/// `Arguments::new(&[u8; N], &[Argument; M]) -> Arguments`:
/// snapshot the template bytes and the argument handles. The
/// template address names word-per-byte storage (a guest array's
/// layout, which the program's statics table mirrors for constants):
/// byte `k` is the low byte of the word `8 * k` below the address.
/// The argument array is one handle word per element, at the same
/// descending stride. Arguments: (template address, template
/// length, args address, args count, _).
extern "C" fn host_arguments_new(
    ctx: *mut BuiltinCtx,
    tmpl_addr: i64,
    tmpl_len: i64,
    args_addr: i64,
    args_len: i64,
    _pad: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    if tmpl_len < 0 || args_len < 0 || tmpl_len > (1 << 24) || args_len > (1 << 16) {
        return ctx.fail("fmt::Arguments::new with an out-of-range length");
    }
    // Safety: both addresses are the guest's own references to the
    // template / argument arrays, live for the call.
    let mut template = Vec::with_capacity(tmpl_len as usize);
    for k in 0..tmpl_len {
        template.push((unsafe { load_word(tmpl_addr - 8 * k) } & 0xFF) as u8);
    }
    let mut args = Vec::with_capacity(args_len as usize);
    for k in 0..args_len {
        args.push(unsafe { load_word(args_addr - 8 * k) });
    }
    let handle = ctx.push(StoreEntry::Arguments(ArgumentsData::Template {
        template,
        args,
    }));
    HostRet::one(handle)
}

/// `Arguments::from_str(s) -> Arguments` where `s` is a literal the
/// front end interned: the value is the string itself, verbatim.
/// Arguments: (string table index, _, _, _, _).
extern "C" fn host_arguments_from_str_const(
    ctx: *mut BuiltinCtx,
    index: i64,
    _a: i64,
    _b: i64,
    _c: i64,
    _d: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    let Some(s) = ctx.strings.get(index as usize).cloned() else {
        return ctx.fail("fmt::Arguments::from_str string index out of range");
    };
    let handle = ctx.push(StoreEntry::Arguments(ArgumentsData::Plain(s)));
    HostRet::one(handle)
}

/// `Arguments::from_str(s) -> Arguments` where `s` is a runtime
/// `&str`: copy its bytes out of guest memory. Arguments:
/// (data pointer, length, _, _, _).
extern "C" fn host_arguments_from_str_rt(
    ctx: *mut BuiltinCtx,
    ptr: i64,
    len: i64,
    _a: i64,
    _b: i64,
    _c: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    let s = match read_str(ptr, len) {
        Ok(s) => s,
        Err(m) => return ctx.fail(&m),
    };
    let handle = ctx.push(StoreEntry::Arguments(ArgumentsData::Plain(s)));
    HostRet::one(handle)
}

/// Read the string of `len` bytes at guest address `ptr`.
fn read_str(ptr: i64, len: i64) -> Result<String, String> {
    if len < 0 || len > (1 << 31) {
        return Err("string length out of range".to_string());
    }
    // Safety: the caller hands over a guest `&str`'s own pointer
    // and length; the bytes are the string's data.
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };
    String::from_utf8(bytes.to_vec()).map_err(|_| "string is not valid UTF-8".to_string())
}

/// Read `len` raw bytes at guest address `ptr` (no UTF-8
/// validation: the caller models a producer whose input is
/// already a `str`).
fn read_bytes(ptr: i64, len: i64) -> Result<Vec<u8>, String> {
    if len < 0 || len > (1 << 31) {
        return Err("string length out of range".to_string());
    }
    // Safety: the caller hands over a guest `&str`'s own pointer
    // and length; the bytes are the string's data.
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };
    Ok(bytes.to_vec())
}

/// Write a `String` value for `bytes` at guest address `dst`:
/// the (pointer, capacity, length) triple, capacity equal to
/// length, at the aggregate's descending leaf stride. The buffer
/// is a real host allocation of exactly `bytes.len()` bytes, so
/// the guest's own drop path (`__rust_dealloc` with the layout
/// its converted bodies compute from the triple) frees exactly
/// this allocation. The empty string's pointer is the dangling
/// address an empty `Vec` carries (the alignment of `u8`), and
/// no allocation backs it — the drop path frees nothing for a
/// zero capacity, as in core.
fn write_string_at(ctx: &mut BuiltinCtx, dst: i64, bytes: &[u8]) -> HostRet {
    let len = bytes.len();
    let ptr = if len == 0 {
        1
    } else {
        let layout = std::alloc::Layout::from_size_align(len, 1)
            .expect("a byte count is a valid u8 layout");
        // Safety: the layout is non-zero; the result is checked.
        let p = unsafe { std::alloc::alloc(layout) };
        if p.is_null() {
            return ctx.fail("memory allocation failed");
        }
        // Safety: `p` owns `len` fresh bytes; the source is a
        // live host string of the same length.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, len) };
        p as i64
    };
    // Safety: `dst` is the destination value's own first-word
    // address (the front end hands over the address of the
    // destination's slot run or of its memory place).
    unsafe {
        store_word(dst, ptr);
        store_word(dst - 8, len as i64);
        store_word(dst - 16, len as i64);
    }
    HostRet::one(0)
}

/// The host allocation layout a `__rust_*` call names, when it
/// is well-formed.
fn guest_layout(size: i64, align: i64) -> Option<std::alloc::Layout> {
    if size < 0 || align <= 0 || !(align as u64).is_power_of_two() {
        return None;
    }
    std::alloc::Layout::from_size_align(size as usize, align as usize).ok()
}

/// `__rust_alloc(size, align) -> *mut u8`: a real host
/// allocation, null on failure (the contract core's allocator
/// shim exposes; the guest's allocation path checks for null).
/// Arguments: (size, align, _, _, _).
extern "C" fn host_rust_alloc(
    _ctx: *mut BuiltinCtx,
    size: i64,
    align: i64,
    _a: i64,
    _b: i64,
    _c: i64,
) -> HostRet {
    let Some(layout) = guest_layout(size, align) else {
        return HostRet::one(0);
    };
    if layout.size() == 0 {
        // A zero-size allocation is never dereferenced; the
        // aligned dangling address is its pointer, as in core.
        return HostRet::one(align);
    }
    // Safety: the layout is non-zero and well-formed.
    HostRet::one(unsafe { std::alloc::alloc(layout) } as i64)
}

/// `__rust_alloc_zeroed(size, align) -> *mut u8`: as
/// [`host_rust_alloc`], with the allocation zero-filled.
/// Arguments: (size, align, _, _, _).
extern "C" fn host_rust_alloc_zeroed(
    _ctx: *mut BuiltinCtx,
    size: i64,
    align: i64,
    _a: i64,
    _b: i64,
    _c: i64,
) -> HostRet {
    let Some(layout) = guest_layout(size, align) else {
        return HostRet::one(0);
    };
    if layout.size() == 0 {
        return HostRet::one(align);
    }
    // Safety: the layout is non-zero and well-formed.
    HostRet::one(unsafe { std::alloc::alloc_zeroed(layout) } as i64)
}

/// `__rust_dealloc(ptr, size, align)`: free an allocation a
/// `__rust_alloc*` call (or a modelled string construction)
/// produced, with the layout the guest's drop path computed for
/// it. A null pointer or a zero size frees nothing (the empty
/// buffer's dangling pointer reaches here in no correct
/// program, but the guard keeps the model total). Arguments:
/// (pointer, size, align, _, _).
extern "C" fn host_rust_dealloc(
    _ctx: *mut BuiltinCtx,
    ptr: i64,
    size: i64,
    align: i64,
    _a: i64,
    _b: i64,
) -> HostRet {
    if ptr == 0 || size == 0 {
        return HostRet::one(0);
    }
    let Some(layout) = guest_layout(size, align) else {
        return HostRet::one(0);
    };
    // Safety: the guest hands back an allocation of exactly
    // this layout (its own drop glue computed it from the
    // value's recorded pointer and capacity).
    unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
    HostRet::one(0)
}

/// `__rust_realloc(ptr, old_size, align, new_size) -> *mut u8`:
/// resize a host allocation, null on failure (the old block is
/// left untouched then, as the contract requires). Arguments:
/// (pointer, old size, align, new size, _).
extern "C" fn host_rust_realloc(
    _ctx: *mut BuiltinCtx,
    ptr: i64,
    old_size: i64,
    align: i64,
    new_size: i64,
    _a: i64,
) -> HostRet {
    if ptr == 0 {
        return host_rust_alloc(_ctx, new_size, align, 0, 0, 0);
    }
    let Some(layout) = guest_layout(old_size, align) else {
        return HostRet::one(0);
    };
    if new_size < 0 {
        return HostRet::one(0);
    }
    // Safety: `ptr` is a live allocation of `layout`; the
    // result is checked before use, as core checks it.
    let p = unsafe { std::alloc::realloc(ptr as *mut u8, layout, new_size as usize) };
    HostRet::one(p as i64)
}

/// `String::from(&str)` (the string case of `From::from`):
/// copy the source's bytes into a fresh buffer and write the
/// `String` triple at the destination address. Arguments:
/// (destination address, data pointer, length, _, _).
extern "C" fn host_string_from_str(
    ctx: *mut BuiltinCtx,
    dst: i64,
    ptr: i64,
    len: i64,
    _a: i64,
    _b: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    let bytes = match read_bytes(ptr, len) {
        Ok(b) => b,
        Err(m) => return ctx.fail(&m),
    };
    write_string_at(ctx, dst, &bytes)
}

/// `std::fmt::format(Arguments) -> String`: format the
/// arguments with the fmt engine and write the resulting text
/// as a `String` at the destination address. Arguments:
/// (destination address, arguments handle, _, _, _).
extern "C" fn host_string_format(
    ctx: *mut BuiltinCtx,
    dst: i64,
    handle: i64,
    _a: i64,
    _b: i64,
    _c: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    let Some(StoreEntry::Arguments(data)) = ctx.entry(handle) else {
        return ctx.fail("fmt::format of a value that is not fmt::Arguments");
    };
    let data = data.clone();
    let text = match format_arguments(ctx, &data) {
        Ok(t) => t,
        Err(m) => return ctx.fail(&m),
    };
    write_string_at(ctx, dst, text.as_bytes())
}

/// `str::to_uppercase(&str) -> String`: the host's own
/// `str::to_uppercase` (the very code the guest's std ships, so
/// the result matches exactly), written as a `String` at the
/// destination address. Arguments: (destination address, data
/// pointer, length, _, _).
extern "C" fn host_str_to_uppercase(
    ctx: *mut BuiltinCtx,
    dst: i64,
    ptr: i64,
    len: i64,
    _a: i64,
    _b: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    let s = match read_str(ptr, len) {
        Ok(s) => s,
        Err(m) => return ctx.fail(&m),
    };
    write_string_at(ctx, dst, s.to_uppercase().as_bytes())
}

/// `str::to_lowercase(&str) -> String`: as
/// [`host_str_to_uppercase`], with `to_lowercase`. Arguments:
/// (destination address, data pointer, length, _, _).
extern "C" fn host_str_to_lowercase(
    ctx: *mut BuiltinCtx,
    dst: i64,
    ptr: i64,
    len: i64,
    _a: i64,
    _b: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    let s = match read_str(ptr, len) {
        Ok(s) => s,
        Err(m) => return ctx.fail(&m),
    };
    write_string_at(ctx, dst, s.to_lowercase().as_bytes())
}

/// `Deref::deref` for a buffer value (`String`, `Vec<T>`): the
/// dereferenced slice is the (pointer, length) pair the value's
/// own triple starts and ends with. Arguments: (value address,
/// _, _, _, _).
extern "C" fn host_deref_buffer(
    _ctx: *mut BuiltinCtx,
    addr: i64,
    _a: i64,
    _b: i64,
    _c: i64,
    _d: i64,
) -> HostRet {
    // Safety: the argument is the guest's own reference to the
    // buffer value; its triple is live for the call.
    let ptr = unsafe { load_word(addr) };
    let len = unsafe { load_word(addr - 16) };
    HostRet::pair(ptr, len)
}

/// An integer bit intrinsic (`ctpop`, `ctlz`, `cttz`,
/// `bitreverse`, `bswap` — operation codes 0..=4): the
/// interpreter's builtin computes on the operand's bit pattern
/// masked to the type's width; so does this. Arguments:
/// (operation, low word, high word, bit width, _). The result
/// is the value as a (low, high) pair (counts are one word;
/// `bitreverse` / `bswap` of a 128-bit operand are two).
extern "C" fn host_bit_op(
    _ctx: *mut BuiltinCtx,
    op: i64,
    lo: i64,
    hi: i64,
    bits: i64,
    _a: i64,
) -> HostRet {
    if !(1..=128).contains(&bits) {
        return HostRet::one(0);
    }
    let bits = bits as u32;
    let pattern = ((hi as u64 as u128) << 64) | (lo as u64 as u128);
    let pattern = if bits == 128 {
        pattern
    } else {
        pattern & ((1u128 << bits) - 1)
    };
    let result: u128 = match op {
        0 => pattern.count_ones() as u128,
        1 => (pattern.leading_zeros() - (128 - bits)) as u128,
        2 => {
            if pattern == 0 {
                bits as u128
            } else {
                pattern.trailing_zeros() as u128
            }
        }
        3 => pattern.reverse_bits() >> (128 - bits),
        4 => pattern.swap_bytes() >> (128 - bits),
        _ => 0,
    };
    HostRet::pair(result as u64 as i64, (result >> 64) as u64 as i64)
}

/// A structural `fmt::Arguments` value, as core's own
/// constructors build it (the aggregate in a panic path, not
/// an `Arguments::new` call): two words — the template's data
/// pointer and a second word. In the tagged form (low bit
/// set) the second word packs the template's byte length
/// above the tag and there is no argument array: the value
/// is the plain string itself (the shape of the static
/// panic messages). Arguments: (data pointer, second word,
/// _, _, _).
extern "C" fn host_arguments_from_struct(
    ctx: *mut BuiltinCtx,
    ptr: i64,
    second: i64,
    _a: i64,
    _b: i64,
    _c: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    if second & 1 == 1 {
        let len = ((second as u64) >> 1) as i64;
        let s = match read_str(ptr, len) {
            Ok(s) => s,
            Err(m) => return ctx.fail(&m),
        };
        let handle = ctx.push(StoreEntry::Arguments(ArgumentsData::Plain(s)));
        return HostRet::one(handle);
    }
    ctx.fail("structural fmt::Arguments with an argument array")
}

/// `String::len(&String)`: the triple's length word.
/// Arguments: (value address, _, _, _, _).
extern "C" fn host_string_len(
    _ctx: *mut BuiltinCtx,
    addr: i64,
    _a: i64,
    _b: i64,
    _c: i64,
    _d: i64,
) -> HostRet {
    // Safety: the argument is the guest's own reference to the
    // string value; its triple is live for the call.
    HostRet::one(unsafe { load_word(addr - 16) })
}

/// `String::capacity(&String)`: the triple's capacity word.
/// Arguments: (value address, _, _, _, _).
extern "C" fn host_string_capacity(
    _ctx: *mut BuiltinCtx,
    addr: i64,
    _a: i64,
    _b: i64,
    _c: i64,
    _d: i64,
) -> HostRet {
    // Safety: as `host_string_len`.
    HostRet::one(unsafe { load_word(addr - 8) })
}

/// Ensure the string at `addr` can hold `additional` more
/// bytes, growing its buffer the way core's `RawVec` growth
/// does: the new capacity is the largest of the required
/// capacity, twice the old capacity, and the minimum
/// non-zero capacity of a byte buffer (8).
/// Returns false when the allocation failed.
fn string_grow(_ctx: &mut BuiltinCtx, addr: i64, additional: i64) -> bool {
    // Safety: `addr` is the guest's own `&mut String` target;
    // its triple is live and exclusively borrowed for the call.
    let (ptr, cap, len) = unsafe {
        (
            load_word(addr),
            load_word(addr - 8),
            load_word(addr - 16),
        )
    };
    let required = len + additional;
    if additional < 0 || required <= cap {
        return additional >= 0;
    }
    let doubled = if cap > (1i64 << 62) { required } else { cap * 2 };
    let new_cap = required.max(doubled).max(8);
    let layout_new = match guest_layout(new_cap, 1) {
        Some(l) => l,
        None => return false,
    };
    let new_ptr = if cap == 0 {
        // Safety: the layout is non-zero and well-formed.
        unsafe { std::alloc::alloc(layout_new) }
    } else {
        let layout_old = match guest_layout(cap, 1) {
            Some(l) => l,
            None => return false,
        };
        // Safety: `ptr` is a live allocation of `layout_old`.
        unsafe { std::alloc::realloc(ptr as *mut u8, layout_old, new_cap as usize) }
    };
    if new_ptr.is_null() {
        return false;
    }
    // Safety: as above; the triple's words are the value's own.
    unsafe {
        store_word(addr, new_ptr as i64);
        store_word(addr - 8, new_cap);
    }
    true
}

/// Append `bytes` to the string at `addr` (growing first).
fn string_append(ctx: &mut BuiltinCtx, addr: i64, bytes: &[u8]) -> HostRet {
    if !string_grow(ctx, addr, bytes.len() as i64) {
        return ctx.fail("memory allocation failed");
    }
    // Safety: the buffer now has room; the triple is the
    // value's own.
    let (ptr, len) = unsafe { (load_word(addr), load_word(addr - 16)) };
    if !bytes.is_empty() {
        // Safety: `ptr + len` is inside the (grown) allocation
        // with `bytes.len()` bytes free.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (ptr as *mut u8).add(len as usize),
                bytes.len(),
            );
        }
    }
    // Safety: as above.
    unsafe { store_word(addr - 16, len + bytes.len() as i64) };
    HostRet::one(0)
}

/// `String::push_str(&mut String, &str)`: append the source's
/// bytes. Arguments: (string address, source pointer, source
/// length, _, _).
extern "C" fn host_string_push_str(
    ctx: *mut BuiltinCtx,
    addr: i64,
    src_ptr: i64,
    src_len: i64,
    _a: i64,
    _b: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    let bytes = match read_bytes(src_ptr, src_len) {
        Ok(b) => b,
        Err(m) => return ctx.fail(&m),
    };
    string_append(ctx, addr, &bytes)
}

/// `String::push(&mut String, char)`: append the character's
/// UTF-8 encoding. Arguments: (string address, char word, _,
/// _, _).
extern "C" fn host_string_push(
    ctx: *mut BuiltinCtx,
    addr: i64,
    ch: i64,
    _a: i64,
    _b: i64,
    _c: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    let Some(c) = char::from_u32(ch as u64 as u32) else {
        return ctx.fail("String::push of an invalid char value");
    };
    let mut buf = [0u8; 4];
    let s = c.encode_utf8(&mut buf);
    string_append(ctx, addr, s.as_bytes())
}

/// A `str` pattern search (`starts_with` = 0, `ends_with` = 1,
/// `contains` = 2) with a `&str` pattern: the host's own `str`
/// methods, so the answer matches the guest's std exactly.
/// Arguments: (operation, data pointer, length, pattern
/// pointer, pattern length).
extern "C" fn host_str_op(
    ctx: *mut BuiltinCtx,
    op: i64,
    ptr: i64,
    len: i64,
    pat_ptr: i64,
    pat_len: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    let s = match read_str(ptr, len) {
        Ok(s) => s,
        Err(m) => return ctx.fail(&m),
    };
    let pat = match read_str(pat_ptr, pat_len) {
        Ok(s) => s,
        Err(m) => return ctx.fail(&m),
    };
    let r = match op {
        0 => s.starts_with(&pat),
        1 => s.ends_with(&pat),
        2 => s.contains(&pat),
        _ => return ctx.fail("str search with an unknown operation"),
    };
    HostRet::one(r as i64)
}

/// String equality: the byte comparison of the two contents
/// (the equality `str`'s `PartialEq` defines, which the
/// `String` impls share). Arguments: (pointer, length,
/// pointer, length, _).
extern "C" fn host_str_eq(
    ctx: *mut BuiltinCtx,
    p1: i64,
    l1: i64,
    p2: i64,
    l2: i64,
    _a: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    if l1 != l2 {
        return HostRet::one(0);
    }
    let a = match read_bytes(p1, l1) {
        Ok(b) => b,
        Err(m) => return ctx.fail(&m),
    };
    let b = match read_bytes(p2, l2) {
        Ok(b) => b,
        Err(m) => return ctx.fail(&m),
    };
    HostRet::one((a == b) as i64)
}

/// `core::panicking::panic_nounwind_fmt(Arguments, bool)`:
/// the message is the formatted arguments; the call raises the
/// guest panic with it and never returns a value. Arguments:
/// (arguments handle, _, _, _, _).
extern "C" fn host_panic_nounwind_fmt(
    ctx: *mut BuiltinCtx,
    handle: i64,
    _a: i64,
    _b: i64,
    _c: i64,
    _d: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    let Some(StoreEntry::Arguments(data)) = ctx.entry(handle) else {
        return ctx.fail("panic_nounwind_fmt of a value that is not fmt::Arguments");
    };
    let data = data.clone();
    match format_arguments(ctx, &data) {
        Ok(text) => ctx.fail(&text),
        Err(m) => ctx.fail(&m),
    }
}

/// `std::io::_print(Arguments)` / `_eprint`: format the arguments
/// and append the text to the guest's captured stdout (or stderr).
/// Arguments: (arguments handle, stderr flag, _, _, _).
extern "C" fn host_print(
    ctx: *mut BuiltinCtx,
    handle: i64,
    to_stderr: i64,
    _a: i64,
    _b: i64,
    _c: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    let Some(StoreEntry::Arguments(data)) = ctx.entry(handle) else {
        return ctx.fail("print of a value that is not fmt::Arguments");
    };
    let data = data.clone();
    let text = match format_arguments(ctx, &data) {
        Ok(t) => t,
        Err(m) => return ctx.fail(&m),
    };
    if to_stderr != 0 {
        ctx.stderr.push_str(&text);
    } else {
        ctx.stdout.push_str(&text);
    }
    HostRet::one(0)
}

/// `Iterator::next(&mut Range) -> Option<Item>`: step the range
/// state in guest memory. The state is `[start, end]` at the
/// pointer (each leaf `8` bytes below the previous). Arguments:
/// (state pointer, signed flag, None discriminant, Some
/// discriminant, _). The result is the (discriminant, item) pair.
/// Comparison and increment run on the canonical words: a signed
/// type's words are sign-extended, an unsigned type's
/// zero-extended, so a plain `i64` / `u64` comparison is the
/// type's own; the increment of a `start` known to be below `end`
/// cannot leave the type's range, so it stays canonical.
extern "C" fn host_next_range(
    ctx: *mut BuiltinCtx,
    ptr: i64,
    signed: i64,
    none_disc: i64,
    some_disc: i64,
    _pad: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    // Safety: the pointer is the guest's own `&mut Range`.
    let (start, end) = unsafe { (load_word(ptr), load_word(ptr - 8)) };
    let less = if signed != 0 {
        start < end
    } else {
        (start as u64) < (end as u64)
    };
    if less {
        unsafe { store_word(ptr, start.wrapping_add(1)) };
        HostRet::pair(some_disc, start)
    } else {
        let _ = ctx;
        HostRet::pair(none_disc, 0)
    }
}

/// `Iterator::next(&mut RangeInclusive) -> Option<Item>`: step the
/// `[start, end, exhausted]` state in guest memory — the
/// interpreter's transition: an exhausted range yields `None`; a
/// `start` below `end` yields and increments; a `start` equal to
/// `end` yields once and marks the range exhausted; a `start`
/// above `end` yields `None`. Arguments and result as in
/// [`host_next_range`].
extern "C" fn host_next_range_inclusive(
    ctx: *mut BuiltinCtx,
    ptr: i64,
    signed: i64,
    none_disc: i64,
    some_disc: i64,
    _pad: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    // Safety: the pointer is the guest's own `&mut RangeInclusive`.
    let (start, end, exhausted) =
        unsafe { (load_word(ptr), load_word(ptr - 8), load_word(ptr - 16)) };
    if exhausted != 0 {
        return HostRet::pair(none_disc, 0);
    }
    let (less, equal) = if signed != 0 {
        (start < end, start == end)
    } else {
        ((start as u64) < (end as u64), start == end)
    };
    if less {
        unsafe { store_word(ptr, start.wrapping_add(1)) };
        HostRet::pair(some_disc, start)
    } else if equal {
        unsafe { store_word(ptr - 16, 1) };
        HostRet::pair(some_disc, start)
    } else {
        let _ = ctx;
        HostRet::pair(none_disc, 0)
    }
}

/// A primitive float method: compute the operation on the host,
/// whose `f32` / `f64` methods are the same std code the guest's
/// methods are (the interpreter computes them the same way, for the
/// same reason). Arguments: (receiver bits, second argument bits —
/// the other operand of `min` / `max` / `powf`, or the `i32`
/// exponent of `powi` —, op code, f32 flag, third argument bits —
/// the addend of `mul_add`). The result word is the result's
/// canonical slot form: a float's bit pattern (an `f32` pattern in
/// the low 32 bits), 0 / 1 for the predicates, the integer bit
/// pattern for `to_bits`. `sin_cos` is the one two-word result:
/// the pair (sin bits, cos bits).
extern "C" fn host_float_method(
    ctx: *mut BuiltinCtx,
    x: i64,
    y: i64,
    op_code: i64,
    f32_flag: i64,
    z: i64,
) -> HostRet {
    let ctx = unsafe { &mut *ctx };
    let Some(op) = FloatOp::from_code(op_code) else {
        return ctx.fail("float method op out of range");
    };
    if op == FloatOp::SinCos {
        let (s, c) = if f32_flag != 0 {
            let (s, c) = f32::from_bits(x as u32).sin_cos();
            (s.to_bits() as i64, c.to_bits() as i64)
        } else {
            let (s, c) = f64::from_bits(x as u64).sin_cos();
            (s.to_bits() as i64, c.to_bits() as i64)
        };
        return HostRet::pair(s, c);
    }
    let result = if f32_flag != 0 {
        float_method_f32(op, x, y, z)
    } else {
        float_method_f64(op, x, y, z)
    };
    HostRet::one(result)
}

fn float_method_f64(op: FloatOp, x: i64, y: i64, z: i64) -> i64 {
    use FloatOp as O;
    let v = f64::from_bits(x as u64);
    let bits = |f: f64| f.to_bits() as i64;
    match op {
        O::IsNan => v.is_nan() as i64,
        O::IsInfinite => v.is_infinite() as i64,
        O::IsFinite => v.is_finite() as i64,
        O::IsSignPositive => v.is_sign_positive() as i64,
        O::IsSignNegative => v.is_sign_negative() as i64,
        O::Abs => bits(v.abs()),
        O::Sqrt => bits(v.sqrt()),
        O::Floor => bits(v.floor()),
        O::Ceil => bits(v.ceil()),
        O::Round => bits(v.round()),
        O::RoundTiesEven => bits(v.round_ties_even()),
        O::Trunc => bits(v.trunc()),
        O::Fract => bits(v.fract()),
        O::Recip => bits(v.recip()),
        O::Signum => bits(v.signum()),
        O::ToDegrees => bits(v.to_degrees()),
        O::ToRadians => bits(v.to_radians()),
        O::Powi => bits(v.powi(y as i32)),
        O::Powf => bits(v.powf(f64::from_bits(y as u64))),
        O::Min => bits(v.min(f64::from_bits(y as u64))),
        O::Max => bits(v.max(f64::from_bits(y as u64))),
        O::ToBits => v.to_bits() as i64,
        O::FromBits => x,
        O::Exp => bits(v.exp()),
        O::Exp2 => bits(v.exp2()),
        O::ExpM1 => bits(v.exp_m1()),
        O::Ln => bits(v.ln()),
        O::Ln1p => bits(v.ln_1p()),
        O::Log => bits(v.log(f64::from_bits(y as u64))),
        O::Log2 => bits(v.log2()),
        O::Log10 => bits(v.log10()),
        O::Sin => bits(v.sin()),
        O::Cos => bits(v.cos()),
        O::Tan => bits(v.tan()),
        O::Asin => bits(v.asin()),
        O::Acos => bits(v.acos()),
        O::Atan => bits(v.atan()),
        O::Atan2 => bits(v.atan2(f64::from_bits(y as u64))),
        O::SinCos => unreachable!("sin_cos returns a pair; host_float_method serves it"),
        O::Sinh => bits(v.sinh()),
        O::Cosh => bits(v.cosh()),
        O::Tanh => bits(v.tanh()),
        O::Asinh => bits(v.asinh()),
        O::Acosh => bits(v.acosh()),
        O::Atanh => bits(v.atanh()),
        O::Cbrt => bits(v.cbrt()),
        O::Hypot => bits(v.hypot(f64::from_bits(y as u64))),
        O::MulAdd => bits(v.mul_add(
            f64::from_bits(y as u64),
            f64::from_bits(z as u64),
        )),
        O::DivEuclid => bits(v.div_euclid(f64::from_bits(y as u64))),
        O::RemEuclid => bits(v.rem_euclid(f64::from_bits(y as u64))),
        O::CopySign => (x & !(1i64 << 63)) | (y & (1i64 << 63)),
    }
}

fn float_method_f32(op: FloatOp, x: i64, y: i64, z: i64) -> i64 {
    use FloatOp as O;
    let v = f32::from_bits(x as u32);
    let bits = |f: f32| f.to_bits() as i64;
    match op {
        O::IsNan => v.is_nan() as i64,
        O::IsInfinite => v.is_infinite() as i64,
        O::IsFinite => v.is_finite() as i64,
        O::IsSignPositive => v.is_sign_positive() as i64,
        O::IsSignNegative => v.is_sign_negative() as i64,
        O::Abs => bits(v.abs()),
        O::Sqrt => bits(v.sqrt()),
        O::Floor => bits(v.floor()),
        O::Ceil => bits(v.ceil()),
        O::Round => bits(v.round()),
        O::RoundTiesEven => bits(v.round_ties_even()),
        O::Trunc => bits(v.trunc()),
        O::Fract => bits(v.fract()),
        O::Recip => bits(v.recip()),
        O::Signum => bits(v.signum()),
        O::ToDegrees => bits(v.to_degrees()),
        O::ToRadians => bits(v.to_radians()),
        O::Powi => bits(v.powi(y as i32)),
        O::Powf => bits(v.powf(f32::from_bits(y as u32))),
        O::Min => bits(v.min(f32::from_bits(y as u32))),
        O::Max => bits(v.max(f32::from_bits(y as u32))),
        O::ToBits => v.to_bits() as i64,
        O::FromBits => x & 0xFFFF_FFFF,
        O::Exp => bits(v.exp()),
        O::Exp2 => bits(v.exp2()),
        O::ExpM1 => bits(v.exp_m1()),
        O::Ln => bits(v.ln()),
        O::Ln1p => bits(v.ln_1p()),
        O::Log => bits(v.log(f32::from_bits(y as u32))),
        O::Log2 => bits(v.log2()),
        O::Log10 => bits(v.log10()),
        O::Sin => bits(v.sin()),
        O::Cos => bits(v.cos()),
        O::Tan => bits(v.tan()),
        O::Asin => bits(v.asin()),
        O::Acos => bits(v.acos()),
        O::Atan => bits(v.atan()),
        O::Atan2 => bits(v.atan2(f32::from_bits(y as u32))),
        O::SinCos => unreachable!("sin_cos returns a pair; host_float_method serves it"),
        O::Sinh => bits(v.sinh()),
        O::Cosh => bits(v.cosh()),
        O::Tanh => bits(v.tanh()),
        O::Asinh => bits(v.asinh()),
        O::Acosh => bits(v.acosh()),
        O::Atanh => bits(v.atanh()),
        O::Cbrt => bits(v.cbrt()),
        O::Hypot => bits(v.hypot(f32::from_bits(y as u32))),
        O::MulAdd => bits(v.mul_add(
            f32::from_bits(y as u32),
            f32::from_bits(z as u32),
        )),
        O::DivEuclid => bits(v.div_euclid(f32::from_bits(y as u32))),
        O::RemEuclid => bits(v.rem_euclid(f32::from_bits(y as u32))),
        O::CopySign => (((x as u32) & 0x7fff_ffff) | ((y as u32) & 0x8000_0000)) as i64,
    }
}

// ---------------------------------------------------------------------------
// The fmt engine
// ---------------------------------------------------------------------------
//
// The template encoding and the write loop mirror Rust 1.100.0
// `core::fmt` (as the interpreter's engine does), and the padding
// rules mirror `Formatter::pad` / `pad_integral`. The scalar
// renderings themselves — decimal digits, the float
// shortest-round-trip rendering — come from the host's own std
// formatting, which is the same code the guest's std runs.

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

/// A pointee value read out of guest memory for formatting.
enum HostVal {
    /// A signed integer value, with its type's bit width (the
    /// width two's-complement radix forms are taken at).
    Int(i128, u32),
    /// An unsigned integer value, with its type's bit width.
    Uint(u128, u32),
    Bool(bool),
    Char(char),
    /// A float value: an `f32` value widened exactly to `f64`
    /// (the flag remembers the width, which the rendering must
    /// narrow back to, exactly as the interpreter's model does).
    Float(f64, bool),
    Str(String),
}

/// Format a built `fmt::Arguments` value to a string.
fn format_arguments(ctx: &BuiltinCtx, args: &ArgumentsData) -> Result<String, String> {
    match args {
        ArgumentsData::Plain(s) => Ok(s.clone()),
        ArgumentsData::Template { template, args } => format_template(ctx, template, args),
    }
}

fn format_template(
    ctx: &BuiltinCtx,
    template: &[u8],
    args: &[i64],
) -> Result<String, String> {
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
            out.push_str(&format_arg(ctx, arg, &FmtOptions::new())?);
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
                    count_arg(ctx, args, raw as usize)?
                } else {
                    raw
                };
                options.flags |= WIDTH_FLAG;
            }
            if b & 0x04 != 0 {
                let raw = read_u16(template, j)?;
                j += 2;
                options.precision = if b & 0x20 != 0 {
                    count_arg(ctx, args, raw as usize)?
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
            out.push_str(&format_arg(ctx, arg, &options)?);
            arg_index += 1;
            i = j;
        } else {
            return Err("unexpected byte at a fmt template part boundary".to_string());
        }
    }
    Ok(out)
}

fn template_piece(template: &[u8], start: usize, len: usize) -> Result<&str, String> {
    let bytes = template
        .get(start..start + len)
        .ok_or("fmt template literal piece overruns the template")?;
    std::str::from_utf8(bytes).map_err(|_| "fmt template literal piece is not UTF-8".to_string())
}

fn read_u16(template: &[u8], at: usize) -> Result<u16, String> {
    let b = template
        .get(at..at + 2)
        .ok_or("fmt template truncated u16 field")?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

fn read_u32(template: &[u8], at: usize) -> Result<u32, String> {
    let b = template
        .get(at..at + 4)
        .ok_or("fmt template truncated u32 field")?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn template_arg(args: &[i64], idx: usize) -> Result<i64, String> {
    args.get(idx)
        .copied()
        .ok_or_else(|| format!("fmt template placeholder has no argument {idx}"))
}

/// An indirect width/precision: the argument at `idx` must be a
/// count argument (`Argument::from_usize`).
fn count_arg(ctx: &BuiltinCtx, args: &[i64], idx: usize) -> Result<u16, String> {
    match args.get(idx).and_then(|h| ctx.entry(*h)) {
        Some(StoreEntry::FmtCount(n)) => Ok(*n),
        _ => Err("fmt template indirect width/precision does not name a count argument".into()),
    }
}

fn format_arg(ctx: &BuiltinCtx, handle: i64, options: &FmtOptions) -> Result<String, String> {
    let Some(entry) = ctx.entry(handle) else {
        return Err(format!("fmt template argument handle {handle} is not live"));
    };
    match entry {
        StoreEntry::FmtCount(_) => {
            Err("count argument used as a formatting argument".to_string())
        }
        StoreEntry::Arguments(_) => {
            Err("fmt::Arguments used as a formatting argument".to_string())
        }
        StoreEntry::FmtArg {
            desc,
            ptr,
            second,
            fat,
        } => {
            let (desc_idx, ptr, second, fat) = (*desc, *ptr, *second, *fat);
            let Some(descriptor) = ctx.descriptors.get(desc_idx as usize) else {
                return Err("fmt argument descriptor out of range".to_string());
            };
            if descriptor.trait_ == FmtTrait::DebugNoop {
                return Ok(String::new());
            }
            let value = read_arg_value(descriptor, ptr, second, fat)?;
            format_value(&value, descriptor.trait_, options)
        }
    }
}

/// Read the pointee a fmt argument denotes, through the address
/// its constructor received.
fn read_arg_value(
    desc: &FmtDescriptor,
    ptr: i64,
    second: i64,
    fat: bool,
) -> Result<HostVal, String> {
    if fat {
        // The argument words are the value itself; only a `str`
        // is ever passed fat.
        if desc.ty == FmtTy::Str {
            return Ok(HostVal::Str(read_str(ptr, second)?));
        }
        return Err("fat fmt argument of a non-str type".to_string());
    }
    read_value_at(&desc.ty, ptr)
}

/// Read the value of shape `ty` stored at guest address `addr`:
/// a scalar is the word at `addr`; a `str` is the pair at `addr`
/// (pointer word, then the length word below it); a reference is
/// the inner value's address — or, for a fat inner type, the pair
/// stored at `addr` itself.
fn read_value_at(ty: &FmtTy, addr: i64) -> Result<HostVal, String> {
    // Safety: `addr` is the guest's own reference target for a
    // value of this shape, live while the arguments are formatted
    // (the formatting happens inside the `_print` call, with every
    // captured frame still on the stack).
    unsafe {
        Ok(match ty {
            FmtTy::Int { bits } => {
                if *bits == 128 {
                    // A 128-bit value's words are the low word at
                    // `addr` and the high word below it (the slot
                    // layout: leaf `k` at `addr - 8 * k`).
                    let lo = load_word(addr) as u64 as u128;
                    let hi = load_word(addr - 8) as u64 as u128;
                    HostVal::Int(((hi << 64) | lo) as i128, *bits)
                } else {
                    HostVal::Int(load_word(addr) as i128, *bits)
                }
            }
            FmtTy::Uint { bits } => {
                if *bits == 128 {
                    let lo = load_word(addr) as u64 as u128;
                    let hi = load_word(addr - 8) as u64 as u128;
                    HostVal::Uint((hi << 64) | lo, *bits)
                } else {
                    HostVal::Uint(load_word(addr) as u64 as u128, *bits)
                }
            }
            FmtTy::Bool => HostVal::Bool(load_word(addr) != 0),
            FmtTy::Char => {
                let n = load_word(addr) as u64 as u32;
                HostVal::Char(
                    char::from_u32(n).ok_or("invalid char value in fmt argument")?,
                )
            }
            FmtTy::F32 => HostVal::Float(f32::from_bits(load_word(addr) as u32) as f64, true),
            FmtTy::F64 => HostVal::Float(f64::from_bits(load_word(addr) as u64), false),
            FmtTy::Str => HostVal::Str(read_str(load_word(addr), load_word(addr - 8))?),
            FmtTy::OwnedStr => {
                HostVal::Str(read_str(load_word(addr), load_word(addr - 16))?)
            }
            FmtTy::Ref(inner) => {
                if matches!(&**inner, FmtTy::Str) {
                    return read_value_at(inner, addr);
                }
                let inner_addr = load_word(addr);
                return read_value_at(inner, inner_addr);
            }
            FmtTy::Ignored => {
                return Err("fmt argument read through a noop descriptor".to_string())
            }
        })
    }
}

/// Format one argument value through a formatting trait.
fn format_value(
    value: &HostVal,
    trait_: FmtTrait,
    options: &FmtOptions,
) -> Result<String, String> {
    match trait_ {
        FmtTrait::DebugNoop => Ok(String::new()),
        FmtTrait::Pointer => Err("pointer formatting is not modelled".to_string()),
        FmtTrait::Display => format_display(value, options),
        FmtTrait::Debug => format_debug(value, options),
        FmtTrait::Octal => format_radix(value, 8, "0o", options),
        FmtTrait::LowerHex => format_radix(value, 16, "0x", options),
        FmtTrait::UpperHex => format_radix(value, 16, "0X", options),
        FmtTrait::Binary => format_radix(value, 2, "0b", options),
        FmtTrait::LowerExp => format_exp(value, false, options),
        FmtTrait::UpperExp => format_exp(value, true, options),
    }
}

fn bool_str(b: bool) -> &'static str {
    if b {
        "true"
    } else {
        "false"
    }
}

fn format_display(value: &HostVal, options: &FmtOptions) -> Result<String, String> {
    Ok(match value {
        HostVal::Int(v, bits) => format_int_radix(*v, *bits, 10, "", options),
        HostVal::Uint(v, bits) => format_uint_radix(*v, *bits, 10, "", options),
        HostVal::Bool(b) => pad_str(bool_str(*b), options),
        HostVal::Char(c) => pad_str(&c.to_string(), options),
        HostVal::Float(v, f32) => format_float(*v, *f32, options, false),
        HostVal::Str(s) => pad_str(s, options),
    })
}

fn format_debug(value: &HostVal, options: &FmtOptions) -> Result<String, String> {
    Ok(match value {
        HostVal::Int(v, bits) => {
            if options.flags & DEBUG_LOWER_HEX != 0 {
                format_int_radix(*v, *bits, 16, "0x", options)
            } else if options.flags & DEBUG_UPPER_HEX != 0 {
                format_int_radix(*v, *bits, 16, "0X", options)
            } else {
                format_int_radix(*v, *bits, 10, "", options)
            }
        }
        HostVal::Uint(v, bits) => {
            if options.flags & DEBUG_LOWER_HEX != 0 {
                format_uint_radix(*v, *bits, 16, "0x", options)
            } else if options.flags & DEBUG_UPPER_HEX != 0 {
                format_uint_radix(*v, *bits, 16, "0X", options)
            } else {
                format_uint_radix(*v, *bits, 10, "", options)
            }
        }
        HostVal::Bool(b) => pad_str(bool_str(*b), options),
        HostVal::Char(c) => pad_str(&format!("{c:?}"), options),
        HostVal::Float(v, f32) => format_float(*v, *f32, options, true),
        HostVal::Str(s) => {
            // Precision truncates the string (in chars) before
            // it is escaped and quoted.
            let truncated: String = match options.precision() {
                Some(p) => s.chars().take(p as usize).collect(),
                None => s.clone(),
            };
            pad_width(&format!("{truncated:?}"), options)
        }
    })
}

/// Octal/hex/binary formatting of an integer value.
fn format_radix(
    value: &HostVal,
    radix: u32,
    prefix: &str,
    options: &FmtOptions,
) -> Result<String, String> {
    Ok(match value {
        HostVal::Int(v, bits) => format_int_radix(*v, *bits, radix, prefix, options),
        HostVal::Uint(v, bits) => format_uint_radix(*v, *bits, radix, prefix, options),
        _ => return Err("radix formatting of a non-integer value".to_string()),
    })
}

/// Exponential formatting of an integer or float value
/// (`{:e}` / `{:E}`).
fn format_exp(value: &HostVal, upper: bool, options: &FmtOptions) -> Result<String, String> {
    if let HostVal::Float(v, f32) = value {
        let body = if *f32 {
            let x = *v as f32;
            match (options.precision(), upper) {
                (Some(p), false) => format!("{:.*e}", p as usize, x),
                (Some(p), true) => format!("{:.*E}", p as usize, x),
                (None, false) => format!("{x:e}"),
                (None, true) => format!("{x:E}"),
            }
        } else {
            match (options.precision(), upper) {
                (Some(p), false) => format!("{:.*e}", p as usize, *v),
                (Some(p), true) => format!("{:.*E}", p as usize, *v),
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
        HostVal::Int(v, _) => (*v < 0, v.unsigned_abs().to_string()),
        HostVal::Uint(v, _) => (false, v.to_string()),
        _ => return Err("exponential formatting of a non-numeric value".to_string()),
    };
    let body = exp_body(&digits, upper);
    Ok(pad_integral(negative, "", &body, options))
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
/// count toward the width; zero-padding is sign-aware; otherwise
/// the default alignment is Right.
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

/// Integer formatting through `pad_integral`. Radix 10 formats
/// the magnitude with a sign; other radices format the
/// two's-complement pattern at the type's width, as core does.
fn format_int_radix(v: i128, bits: u32, radix: u32, prefix: &str, options: &FmtOptions) -> String {
    let (negative, digits) = if radix == 10 {
        (v < 0, v.unsigned_abs().to_string())
    } else {
        let mask = if bits >= 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        };
        (false, digits_in_radix((v as u128) & mask, radix))
    };
    finish_int(negative, digits, prefix, options)
}

fn format_uint_radix(v: u128, bits: u32, radix: u32, prefix: &str, options: &FmtOptions) -> String {
    let digits = if radix == 10 {
        v.to_string()
    } else {
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

/// Display/Debug formatting of a float value. The host formatter
/// is the same std code the native binary runs, so the
/// shortest-round-trip rendering matches exactly; the width flag
/// selects because an `f32` value is stored widened to `f64` and
/// must be narrowed back first (`0.1f32` prints as `0.1`, not as
/// the widened `0.10000000149011612`). NaN ignores the sign flag
/// and precision in core's float formatting; width still applies.
fn format_float(v: f64, f32: bool, options: &FmtOptions, debug: bool) -> String {
    if v.is_nan() {
        let mut options = *options;
        options.flags &= !(SIGN_PLUS | SIGN_AWARE_ZERO_PAD);
        return pad_integral(false, "", "NaN", &options);
    }
    let body = if f32 {
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
        return if upper {
            "0E0".to_string()
        } else {
            "0e0".to_string()
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_with(descriptors: Vec<FmtDescriptor>, strings: Vec<String>) -> BuiltinCtx {
        BuiltinCtx::new(BuiltinTables {
            descriptors,
            strings,
        })
    }

    /// Word-per-byte descending storage for `bytes`, plus the base
    /// address (the address of element 0's word) — the layout a
    /// guest array has in this model, which the host functions read.
    fn byte_storage(bytes: &[u8]) -> (Vec<i64>, i64) {
        let n = bytes.len();
        let mut words = vec![0i64; n.max(1)];
        for (k, b) in bytes.iter().enumerate() {
            words[n - 1 - k] = *b as i64;
        }
        let addr = words.as_ptr() as i64 + 8 * (n as i64 - 1);
        (words, addr)
    }

    /// Handle-word descending storage, as for `byte_storage`.
    fn handle_storage(handles: &[i64]) -> (Vec<i64>, i64) {
        let n = handles.len();
        let mut words = vec![0i64; n.max(1)];
        for (k, h) in handles.iter().enumerate() {
            words[n - 1 - k] = *h;
        }
        let addr = words.as_ptr() as i64 + 8 * (n as i64 - 1);
        (words, addr)
    }

    #[test]
    fn canonical_names_and_lookup() {
        assert_eq!(
            canonical_name("core::fmt::rt::Argument::<'_>::new_display"),
            "core::fmt::rt::Argument::new_display"
        );
        assert_eq!(
            canonical_name("std::fmt::Arguments::<'a>::from_str"),
            "std::fmt::Arguments::from_str"
        );
        assert_eq!(
            canonical_name("std::ops::RangeInclusive::<Idx>::new"),
            "std::ops::RangeInclusive::<Idx>::new"
        );
        assert_eq!(
            lookup("core::fmt::rt::Argument::new_display"),
            Some(Builtin::ArgumentNew(FmtTrait::Display))
        );
        assert_eq!(
            lookup("std::ops::RangeInclusive::<Idx>::new"),
            Some(Builtin::RangeInclusiveNew)
        );
        assert_eq!(
            lookup("core::f64::<impl f64>::is_nan"),
            Some(Builtin::FloatMethod {
                f32: false,
                op: FloatOp::IsNan
            })
        );
        assert_eq!(
            lookup("std::f32::<impl f32>::sqrt"),
            Some(Builtin::FloatMethod {
                f32: true,
                op: FloatOp::Sqrt
            })
        );
        assert_eq!(lookup("std::thread::spawn"), None);
        assert_eq!(
            lookup("core::f64::<impl f64>::sin_cos"),
            Some(Builtin::FloatMethod {
                f32: false,
                op: FloatOp::SinCos
            })
        );
        assert_eq!(
            lookup("std::f32::<impl f32>::exp"),
            Some(Builtin::FloatMethod {
                f32: true,
                op: FloatOp::Exp
            })
        );
        assert_eq!(
            lookup("core::f64::<impl f64>::mul_add"),
            Some(Builtin::FloatMethod {
                f32: false,
                op: FloatOp::MulAdd
            })
        );
        // Still unstable in Rust 1.100.0, hence unmodelled.
        assert_eq!(lookup("core::f64::<impl f64>::maximum"), None);
    }

    #[test]
    fn print_formats_through_the_host_layer() {
        let mut ctx = ctx_with(
            vec![
                FmtDescriptor {
                    trait_: FmtTrait::Display,
                    ty: FmtTy::Int { bits: 32 },
                },
                FmtDescriptor {
                    trait_: FmtTrait::Display,
                    ty: FmtTy::Bool,
                },
            ],
            Vec::new(),
        );
        let cp = &mut ctx as *mut BuiltinCtx;
        let a: i64 = -42;
        let b: i64 = 1;
        // Safety: the host functions receive the test's own
        // context and the addresses of the test's own locals, the
        // same contract the generated code runs under.
        unsafe {
            let h1 = host_argument_new(cp, &a as *const i64 as i64, 0, 0, 0, 0).first;
            let h2 = host_argument_new(cp, &b as *const i64 as i64, 0, 1, 0, 0).first;
            // "{} {}\n" in the 1.100 template encoding.
            let (_tw, taddr) = byte_storage(&[192, 1, b' ', 192, 1, b'\n', 0]);
            let (_aw, aaddr) = handle_storage(&[h1, h2]);
            let hargs = host_arguments_new(cp, taddr, 7, aaddr, 2, 0).first;
            host_print(cp, hargs, 0, 0, 0, 0);
        }
        assert_eq!(ctx.output().0, "-42 true\n");
        assert_eq!(ctx.output().1, "");
    }

    #[test]
    fn print_applies_precision_and_str_arguments() {
        let mut ctx = ctx_with(
            vec![
                FmtDescriptor {
                    trait_: FmtTrait::Display,
                    ty: FmtTy::F64,
                },
                FmtDescriptor {
                    trait_: FmtTrait::Display,
                    ty: FmtTy::Str,
                },
            ],
            Vec::new(),
        );
        let cp = &mut ctx as *mut BuiltinCtx;
        let x: i64 = 3.14159f64.to_bits() as i64;
        let text = "mid";
        unsafe {
            let h1 = host_argument_new(cp, &x as *const i64 as i64, 0, 0, 0, 0).first;
            // A fat `str` argument: the pair is the value.
            let h2 = host_argument_new(
                cp,
                text.as_ptr() as i64,
                text.len() as i64,
                1,
                1,
                0,
            )
            .first;
            // "{:.2} {}\n": general placeholder (precision field),
            // then a default placeholder, with literal pieces.
            let template = [0xC4, 2, 0, 1, b' ', 192, 1, b'\n', 0];
            let (_tw, taddr) = byte_storage(&template);
            let (_aw, aaddr) = handle_storage(&[h1, h2]);
            let hargs = host_arguments_new(cp, taddr, template.len() as i64, aaddr, 2, 0).first;
            host_print(cp, hargs, 0, 0, 0, 0);
        }
        assert_eq!(ctx.output().0, "3.14 mid\n");
    }

    #[test]
    fn from_str_const_prints_verbatim() {
        let mut ctx = ctx_with(Vec::new(), vec!["hello from corpus\n".to_string()]);
        let cp = &mut ctx as *mut BuiltinCtx;
        unsafe {
            let h = host_arguments_from_str_const(cp, 0, 0, 0, 0, 0).first;
            host_print(cp, h, 0, 0, 0, 0);
        }
        assert_eq!(ctx.output().0, "hello from corpus\n");
    }

    #[test]
    fn next_range_steps_state_in_memory() {
        let mut ctx = ctx_with(Vec::new(), Vec::new());
        let cp = &mut ctx as *mut BuiltinCtx;
        // State [start=3, end=5]: start at the pointer, end below.
        let mut state = vec![5i64, 3i64];
        let ptr = state.as_mut_ptr() as i64 + 8;
        unsafe {
            let r = host_next_range(cp, ptr, 0, 0, 1, 0);
            assert_eq!((r.first, r.second), (1, 3));
            let r = host_next_range(cp, ptr, 0, 0, 1, 0);
            assert_eq!((r.first, r.second), (1, 4));
            let r = host_next_range(cp, ptr, 0, 0, 1, 0);
            assert_eq!((r.first, r.second), (0, 0));
        }
        assert_eq!(state[1], 5);
    }

    #[test]
    fn next_range_inclusive_marks_exhaustion() {
        let mut ctx = ctx_with(Vec::new(), Vec::new());
        let cp = &mut ctx as *mut BuiltinCtx;
        // State [start=4, end=4, exhausted=0], start at the pointer.
        let mut state = vec![0i64, 4i64, 4i64];
        let ptr = state.as_mut_ptr() as i64 + 16;
        unsafe {
            let r = host_next_range_inclusive(cp, ptr, 0, 0, 1, 0);
            assert_eq!((r.first, r.second), (1, 4));
            assert_eq!(state[0], 1);
            let r = host_next_range_inclusive(cp, ptr, 0, 0, 1, 0);
            assert_eq!((r.first, r.second), (0, 0));
        }
    }

    #[test]
    fn float_methods_compute_on_the_host() {
        let mut ctx = ctx_with(Vec::new(), Vec::new());
        let cp = &mut ctx as *mut BuiltinCtx;
        unsafe {
            let nan = f64::NAN.to_bits() as i64;
            let r = host_float_method(cp, nan, 0, FloatOp::IsNan.code(), 0, 0);
            assert_eq!(r.first, 1);
            let r = host_float_method(cp, 4.0f64.to_bits() as i64, 0, FloatOp::Sqrt.code(), 0, 0);
            assert_eq!(r.first, 2.0f64.to_bits() as i64);
            let r = host_float_method(
                cp,
                (-2.5f32).to_bits() as i64,
                0,
                FloatOp::Abs.code(),
                1,
                0,
            );
            assert_eq!(r.first, 2.5f32.to_bits() as i64);
            let r = host_float_method(
                cp,
                1.5f64.to_bits() as i64,
                3,
                FloatOp::Powi.code(),
                0,
                0,
            );
            assert_eq!(r.first, 3.375f64.to_bits() as i64);
            let r = host_float_method(
                cp,
                1.25f64.to_bits() as i64,
                0,
                FloatOp::Exp.code(),
                0,
                0,
            );
            assert_eq!(r.first, 1.25f64.exp().to_bits() as i64);
            // mul_add reads its addend from the third argument slot.
            let r = host_float_method(
                cp,
                1.5f64.to_bits() as i64,
                2.0f64.to_bits() as i64,
                FloatOp::MulAdd.code(),
                0,
                0.5f64.to_bits() as i64,
            );
            assert_eq!(r.first, 3.5f64.to_bits() as i64);
            // sin_cos returns the (sin, cos) pair.
            let r = host_float_method(
                cp,
                0.7f64.to_bits() as i64,
                0,
                FloatOp::SinCos.code(),
                0,
                0,
            );
            let (s, c) = 0.7f64.sin_cos();
            assert_eq!(r.first, s.to_bits() as i64);
            assert_eq!(r.second, c.to_bits() as i64);
            let r = host_float_method(
                cp,
                0.7f32.to_bits() as i64,
                0,
                FloatOp::SinCos.code(),
                1,
                0,
            );
            let (s, c) = 0.7f32.sin_cos();
            assert_eq!(r.first, s.to_bits() as i64);
            assert_eq!(r.second, c.to_bits() as i64);
        }
    }
}
