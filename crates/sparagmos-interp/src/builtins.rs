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

use std::cmp::Ordering;
use std::collections::HashMap;

use sparagmos_mir as mir;

use crate::error::{InterpError, InterpResult};
use crate::interp::{Frame, Interpreter, Resolved};
use crate::ir::{Body, Operand};
use crate::value::{FmtArguments, FmtTrait, RefProj, RefTarget, Value};

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
    /// `std::convert::From::from(value)` (modelled conversions only).
    FromFrom,
    /// `std::hint::must_use(value)`: identity.
    MustUse,
    /// `std::ops::Deref::deref(&value)` (modelled types only).
    Deref,
    /// `str::to_uppercase(&str) -> String`.
    ToUppercase,
    /// `str::to_lowercase(&str) -> String`.
    ToLowercase,
    /// `core::panicking::panic(&str)`: raise a guest panic.
    PanicStr,
    /// `std::rt::panic_fmt(Arguments)` / `core::panicking::panic_fmt`:
    /// raise a guest panic with the formatted message.
    PanicFmt,
    /// `core::panicking::assert_failed(kind, &left, &right, args)`:
    /// raise a guest panic reporting the failed comparison.
    AssertFailed,
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
    /// `Box::new(value)`: the box is an ADT aggregate holding its
    /// pointee in this value model.
    BoxNew,
    /// `Vec::new()`: an empty vector (ADT aggregate of its elements).
    VecNew,
    /// `Vec::push(&mut vec, value)`.
    VecPush,
    /// `Vec::len(&vec)`.
    VecLen,
    /// `Vec::is_empty(&vec)`.
    VecIsEmpty,
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
    /// `Cell::new(value)`: a cell is an ADT aggregate of one field.
    CellNew,
    /// `Cell::get(&cell)`.
    CellGet,
    /// `Cell::set(&cell, value)`.
    CellSet,
    /// `Cell::replace(&cell, value) -> old value`.
    CellReplace,
    /// `Cell::into_inner(cell) -> value`.
    CellIntoInner,
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
    /// `std::string::String::new()`: an empty string.
    StringNew,
    /// `Rc::new(value)`: a one-field ADT aggregate holding the
    /// value, the same shape as `Box::new` in this model. The
    /// reference count is not observable through the modelled
    /// operations (`new`, `Deref`, `Clone` as a value copy).
    RcNew,
    /// `Arc::new(value)`: the `Rc` shape; the interpreter is
    /// single-threaded, so the atomic count is unobservable too.
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
    /// `std::mem::zeroed()`: the all-zero-bits value of a scalar
    /// type, or of a tuple/array of scalar types.
    MemZeroed,
    /// `std::ptr::null()` / `std::ptr::null_mut()`: the null
    /// pointer, a dedicated `RefTarget::Null` denoting no place
    /// and no value.
    PtrNull,
}

/// A relational operator selected by a `PartialOrd` method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelOp {
    Lt,
    Le,
    Gt,
    Ge,
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
pub fn lookup(canonical: &str) -> Option<Builtin> {
    use FmtTrait as F;
    Some(match canonical {
        "std::io::_print" => Builtin::Print,
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
        "std::convert::Into::into" => Builtin::IntoInto,
        "std::borrow::ToOwned::to_owned" => Builtin::ToOwned,
        _ if canonical.starts_with("std::boxed::Box") && canonical.ends_with(">::new") => {
            Builtin::BoxNew
        }
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
        "std::mem::needs_drop" => Builtin::MemNeedsDrop,
        _ if canonical.starts_with("std::mem::ManuallyDrop") && canonical.ends_with(">::new") => {
            Builtin::ManuallyDropNew
        }
        "std::intrinsics::black_box" => Builtin::BlackBox,
        "std::string::String::new" => Builtin::StringNew,
        "std::intrinsics::ctpop" => Builtin::IntBitOp(BitOp::CtPop),
        "std::intrinsics::ctlz" => Builtin::IntBitOp(BitOp::Ctlz),
        "std::intrinsics::cttz" => Builtin::IntBitOp(BitOp::Cttz),
        "std::intrinsics::bitreverse" => Builtin::IntBitOp(BitOp::BitReverse),
        "std::intrinsics::bswap" => Builtin::IntBitOp(BitOp::Bswap),
        "std::intrinsics::saturating_add" => Builtin::SaturatingArith(SatOp::Add),
        "std::intrinsics::saturating_sub" => Builtin::SaturatingArith(SatOp::Sub),
        "std::intrinsics::atomic_load" => Builtin::AtomicLoad,
        "std::intrinsics::atomic_store" => Builtin::AtomicStore,
        "std::mem::zeroed" => Builtin::MemZeroed,
        "std::ptr::null" | "std::ptr::null_mut" => Builtin::PtrNull,
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
    ) -> InterpResult<Value> {
        let kind = lookup(name)
            .ok_or_else(|| unsupported(format!("call to {name} is not runtime-supported")))?;
        let mut vals = Vec::with_capacity(args.len());
        for arg in args {
            vals.push(self.eval_operand(frames, body, arg)?);
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
            Builtin::IntoIter => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                let value = vals.into_iter().next().expect("arity checked");
                if self_ty.starts_with("std::ops::Range<")
                    || self_ty.starts_with("std::ops::RangeInclusive<")
                    || self_ty.starts_with("std::slice::Iter<")
                {
                    // A range is its own iterator in this model, and a
                    // slice iterator is already in `next`-ready shape
                    // (built by the `slice::iter` runtime support).
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
            Builtin::FromFrom | Builtin::IntoInto => {
                arity(1)?;
                let target = generic_args.first().map(String::as_str).unwrap_or("");
                let source = generic_args.get(1).map(String::as_str).unwrap_or("");
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
                    // A box in this model is an ADT aggregate holding
                    // its pointee; `deref` yields a reference to that
                    // pointee value (reads resolve against the value,
                    // as they do for constant aggregates).
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    match pointee {
                        Value::Enum { fields, .. } if fields.len() == 1 => {
                            let inner = fields.into_iter().next().expect("len checked");
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
                    // The same one-field aggregate as `Box::new`
                    // in this model: `deref` yields a reference to
                    // the held value.
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    match pointee {
                        Value::Enum { fields, .. } if fields.len() == 1 => {
                            let inner = fields.into_iter().next().expect("len checked");
                            Ok(Value::Ref(RefTarget::Value(Box::new(inner))))
                        }
                        ref other => Err(InterpError::type_mismatch(
                            "Deref::deref for Rc/Arc",
                            "one-field aggregate",
                            other.kind_name(),
                        )),
                    }
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
            Builtin::PanicStr => {
                arity(1)?;
                // `core::panicking::panic` takes the message as a
                // string; formatted-panic shapes hand over a built
                // `fmt::Arguments` instead.
                let message = match &vals[0] {
                    Value::FmtArguments(_) => {
                        self.format_arguments(frames, expect_arguments(&vals[0], name)?)?
                    }
                    _ => self.resolve_string(frames, &vals[0])?,
                };
                Err(InterpError::panic(message))
            }
            Builtin::PanicFmt => {
                arity(1)?;
                let message =
                    self.format_arguments(frames, expect_arguments(&vals[0], name)?)?;
                Err(InterpError::panic(message))
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
                Err(InterpError::panic(message))
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
                let ord = self.values_cmp(frames, &vals[0], &vals[1])?;
                Ok(Value::Enum {
                    variant: 1,
                    fields: vec![ordering_value(ord)],
                })
            }
            Builtin::OrdCmp => {
                arity(2)?;
                let ord = self.values_cmp(frames, &vals[0], &vals[1])?;
                Ok(ordering_value(ord))
            }
            Builtin::PartialOrdRel(op) => {
                arity(2)?;
                let ord = self.values_cmp(frames, &vals[0], &vals[1])?;
                Ok(Value::Bool(match op {
                    RelOp::Lt => ord == Ordering::Less,
                    RelOp::Le => ord != Ordering::Greater,
                    RelOp::Gt => ord == Ordering::Greater,
                    RelOp::Ge => ord != Ordering::Less,
                }))
            }
            Builtin::CloneClone => {
                arity(1)?;
                match &vals[0] {
                    // `&self`: the clone is a copy of the pointee
                    // value (which may itself be a reference value
                    // when `Self` is a reference type).
                    Value::Ref(_) => {
                        let target = self.deref_value(frames, &vals[0])?;
                        self.read_resolved(frames, &target)
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
            Builtin::BoxNew | Builtin::ManuallyDropNew | Builtin::RcNew | Builtin::ArcNew => {
                arity(1)?;
                // Both are one-field ADT aggregates in this model:
                // a box holds its pointee, a `ManuallyDrop` its value.
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![vals.into_iter().next().expect("arity checked")],
                })
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
                let state = self.read_resolved(frames, &target)?;
                match state {
                    Value::Enum {
                        variant,
                        mut fields,
                    } => {
                        fields.push(vals.into_iter().nth(1).expect("arity checked"));
                        self.write_resolved(
                            frames,
                            &target,
                            Value::Enum { variant, fields },
                        )?;
                        Ok(Value::Unit)
                    }
                    ref other => Err(InterpError::type_mismatch(
                        "Vec::push",
                        "vector value",
                        other.kind_name(),
                    )),
                }
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
                let idx = expect_index(&vals[1], name)?;
                let (elems, len) = self.slice_elements(frames, &vals[0])?;
                if idx as u128 >= len {
                    return Err(InterpError::panic(format!(
                        "index out of bounds: the len is {len} but the index is {idx}"
                    )));
                }
                // A shared element reference: reads resolve against
                // the element value itself (the model has no separate
                // heap place for it), as for constant aggregates.
                let elem = elems.into_iter().nth(idx).expect("bounds checked");
                Ok(Value::Ref(RefTarget::Value(Box::new(elem))))
            }
            Builtin::IndexMut => {
                arity(2)?;
                let idx = expect_index(&vals[1], name)?;
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
                            // Only array (tuple) elements are
                            // addressable places in this model; a
                            // vector's elements live inside its
                            // aggregate value and cannot be pointed
                            // into, so those stay unsupported.
                            Value::Tuple(_) => {
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
                    match pointee {
                        Value::Enum { fields, .. } if fields.len() == 1 => {
                            let inner = fields.into_iter().next().expect("len checked");
                            Ok(Value::Ref(RefTarget::Value(Box::new(inner))))
                        }
                        ref other => Err(InterpError::type_mismatch(
                            "DerefMut::deref_mut for Box",
                            "boxed value",
                            other.kind_name(),
                        )),
                    }
                } else {
                    Err(unsupported(format!(
                        "DerefMut::deref_mut for {self_ty} is not runtime-supported"
                    )))
                }
            }
            Builtin::CellNew => {
                arity(1)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![vals.into_iter().next().expect("arity checked")],
                })
            }
            Builtin::CellGet | Builtin::CellIntoInner => {
                arity(1)?;
                match self.resolve_value(frames, &vals[0])? {
                    Value::Enum { fields, .. } if fields.len() == 1 => {
                        Ok(fields.into_iter().next().expect("len checked"))
                    }
                    ref other => Err(InterpError::type_mismatch(
                        "Cell value",
                        "cell aggregate",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::CellSet | Builtin::CellReplace => {
                arity(2)?;
                let target = self.deref_value(frames, &vals[0])?;
                let state = self.read_resolved(frames, &target)?;
                match state {
                    Value::Enum { variant, fields } if fields.len() == 1 => {
                        let old = fields.into_iter().next().expect("len checked");
                        self.write_resolved(
                            frames,
                            &target,
                            Value::Enum {
                                variant,
                                fields: vec![
                                    vals.into_iter().nth(1).expect("arity checked"),
                                ],
                            },
                        )?;
                        if kind == Builtin::CellReplace {
                            Ok(old)
                        } else {
                            Ok(Value::Unit)
                        }
                    }
                    ref other => Err(InterpError::type_mismatch(
                        "Cell::set",
                        "cell aggregate",
                        other.kind_name(),
                    )),
                }
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
                match type_layout(tag) {
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
                    let (elem_size, _) = type_layout(elem).ok_or_else(|| {
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
                    match type_layout(tag) {
                        Some((size, _)) => Ok(Value::Uint(size)),
                        None => Err(unsupported(format!(
                            "size_of_val for {tag} is not runtime-supported"
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
            Builtin::StringNew => {
                arity(0)?;
                Ok(Value::Str(String::new()))
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
                self.read_resolved(frames, &target)
            }
            Builtin::AtomicStore => {
                arity(2)?;
                let target = self.deref_value(frames, &vals[0])?;
                let value = vals.into_iter().nth(1).expect("arity checked");
                self.write_resolved(frames, &target, value)?;
                Ok(Value::Unit)
            }
            Builtin::FloatMethod { f32, op } => exec_float_method(name, f32, op, &vals),
        }
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
        let state = self.read_resolved(frames, &target)?;
        let fields = iterator_fields(&state, self_ty)?;
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
            || self_ty.starts_with("std::slice::Iter<")
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
                (
                    vec![Value::Tuple(items), Value::Uint((idx + 1) as u128)],
                    some_value(item),
                )
            } else {
                (fields, none_value())
            }
        } else {
            return Err(unsupported(format!(
                "Iterator::next for {self_ty} is not runtime-supported"
            )));
        };
        let new_state = match state {
            Value::Enum { variant, .. } => Value::Enum {
                variant,
                fields: new_fields,
            },
            _ => Value::Tuple(new_fields),
        };
        self.write_resolved(frames, &target, new_state)?;
        Ok(result)
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
            let (ea, la) = self.slice_elements(frames, a)?;
            let (eb, lb) = self.slice_elements(frames, b)?;
            if la != lb {
                return Ok(false);
            }
            for (x, y) in ea.iter().zip(eb.iter()) {
                if !self.values_equal(frames, x, y)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        let a = self.resolve_value(frames, a)?;
        let b = self.resolve_value(frames, b)?;
        Ok(match (&a, &b) {
            (Value::Ref(RefTarget::Null), Value::Ref(RefTarget::Null)) => true,
            (Value::Ref(RefTarget::Null), _) | (_, Value::Ref(RefTarget::Null)) => false,
            (Value::Unit, Value::Unit) => true,
            (Value::Bool(x), Value::Bool(y)) => x == y,
            (Value::Int(x), Value::Int(y)) => x == y,
            (Value::Uint(x), Value::Uint(y)) => x == y,
            (Value::Int(x), Value::Uint(y)) => int_uint_eq(*x, *y),
            (Value::Uint(x), Value::Int(y)) => int_uint_eq(*y, *x),
            (Value::Str(x), Value::Str(y)) => x == y,
            (Value::Bytes(x), Value::Bytes(y)) => x == y,
            (Value::Tuple(x), Value::Tuple(y)) => {
                x.len() == y.len() && self.seq_equal(frames, x, y)?
            }
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

    fn seq_equal(&self, frames: &[Frame], a: &[Value], b: &[Value]) -> InterpResult<bool> {
        for (x, y) in a.iter().zip(b.iter()) {
            if !self.values_equal(frames, x, y)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Semantic (derived `Ord`) comparison of two values: integers
    /// and booleans by value, strings byte-wise (as `str`'s `Ord`
    /// is), aggregates lexicographically with enum variants ordered
    /// by declaration order — the derived semantics.
    pub(crate) fn values_cmp(
        &self,
        frames: &[Frame],
        a: &Value,
        b: &Value,
    ) -> InterpResult<Ordering> {
        if matches!(a, Value::FatRef { .. }) || matches!(b, Value::FatRef { .. }) {
            let (ea, la) = self.slice_elements(frames, a)?;
            let (eb, lb) = self.slice_elements(frames, b)?;
            for (x, y) in ea.iter().zip(eb.iter()) {
                match self.values_cmp(frames, x, y)? {
                    Ordering::Equal => {}
                    other => return Ok(other),
                }
            }
            return Ok(la.cmp(&lb));
        }
        let a = self.resolve_value(frames, a)?;
        let b = self.resolve_value(frames, b)?;
        Ok(match (&a, &b) {
            (Value::Unit, Value::Unit) => Ordering::Equal,
            (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
            (Value::Int(x), Value::Int(y)) => x.cmp(y),
            (Value::Uint(x), Value::Uint(y)) => x.cmp(y),
            (Value::Int(x), Value::Uint(y)) => int_uint_cmp(*x, *y),
            (Value::Uint(x), Value::Int(y)) => int_uint_cmp(*y, *x).reverse(),
            (Value::Str(x), Value::Str(y)) => x.as_bytes().cmp(y.as_bytes()),
            (Value::Bytes(x), Value::Bytes(y)) => x.cmp(y),
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
            ) => match va.cmp(vb) {
                Ordering::Equal => self.seq_cmp(frames, fa, fb)?,
                other => other,
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

    fn seq_cmp(&self, frames: &[Frame], a: &[Value], b: &[Value]) -> InterpResult<Ordering> {
        for (x, y) in a.iter().zip(b.iter()) {
            match self.values_cmp(frames, x, y)? {
                Ordering::Equal => {}
                other => return Ok(other),
            }
        }
        Ok(a.len().cmp(&b.len()))
    }

    /// The elements a slice-like value denotes, with its length: a
    /// fat reference contributes its base's elements truncated to the
    /// carried length; any other value contributes its own elements.
    fn slice_elements(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<(Vec<Value>, u128)> {
        match value {
            Value::FatRef { base, len } => {
                let base = self.resolve_value(frames, base)?;
                let mut elems = value_elements(&base)?;
                elems.truncate(*len as usize);
                Ok((elems, *len))
            }
            _ => {
                let resolved = self.resolve_value(frames, value)?;
                let elems = value_elements(&resolved)?;
                let len = elems.len() as u128;
                Ok((elems, len))
            }
        }
    }

    /// Follow references in `value` until a non-reference value.
    pub(crate) fn resolve_value(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Value> {
        let mut value = value.clone();
        loop {
            match &value {
                // The null pointer denotes no data; resolving it
                // yields itself so equality can see it.
                Value::Ref(RefTarget::Null) => return Ok(value),
                Value::Ref(_) | Value::FatRef { .. } => {
                    let target = self.deref_value(frames, &value)?;
                    value = self.read_resolved(frames, &target)?;
                }
                _ => return Ok(value),
            }
        }
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

/// The fields of an iterator state value (ADT aggregate or tuple).
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
            return self.format_debug_seq(frames, &pointee, options, indent, '[', ']');
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
            Value::Tuple(_) if pointee_tag(ty).starts_with('[') => {
                self.format_debug_seq(frames, value, options, indent, '[', ']')
            }
            Value::Tuple(_) => self.format_debug_seq(frames, value, options, indent, '(', ')'),
            Value::FmtArguments(inner) => {
                let text = self.format_arguments(frames, inner)?;
                Ok(pad_width(&text, options))
            }
            Value::Enum { .. } => Err(unsupported(
                "Debug formatting for an ADT value (variant names are not tracked in values)",
            )),
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
    ) -> InterpResult<String> {
        let items = match value {
            Value::Tuple(items) => items,
            Value::Bytes(b) => {
                let items: Vec<Value> = b.iter().map(|byte| Value::Uint(*byte as u128)).collect();
                return self.format_debug_seq_items(frames, &items, options, indent, open, close);
            }
            other => {
                return Err(InterpError::type_mismatch(
                    "Debug sequence formatting",
                    "tuple or byte value",
                    other.kind_name(),
                ))
            }
        };
        self.format_debug_seq_items(frames, items, options, indent, open, close)
    }

    fn format_debug_seq_items(
        &self,
        frames: &[Frame],
        items: &[Value],
        options: &FmtOptions,
        indent: usize,
        open: char,
        close: char,
    ) -> InterpResult<String> {
        // Element type tags are not tracked in values, so elements
        // are formatted from their value kind alone.
        if options.alternate() {
            if items.is_empty() {
                return Ok(format!("{open}{close}"));
            }
            let inner = indent + 4;
            let mut out = String::new();
            out.push(open);
            out.push('\n');
            for item in items {
                out.push_str(&" ".repeat(inner));
                out.push_str(&self.format_value(
                    frames,
                    item,
                    FmtTrait::Debug,
                    "",
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
        for item in items {
            parts.push(self.format_value(
                frames,
                item,
                FmtTrait::Debug,
                "",
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
