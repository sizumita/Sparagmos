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
use crate::interp::{Frame, Interpreter};
use crate::ir::{Body, Operand};
use crate::value::{FmtArguments, FmtTrait, Value};

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
        _ => return None,
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
                {
                    // A range is its own iterator in this model.
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
            Builtin::FromFrom => {
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
        } else if self_ty.starts_with("std::array::IntoIter<") {
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

    /// Follow references in `value` until a non-reference value.
    pub(crate) fn resolve_value(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Value> {
        let mut value = value.clone();
        loop {
            match &value {
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
            FmtTrait::LowerExp => self.format_exp(value, false, options),
            FmtTrait::UpperExp => self.format_exp(value, true, options),
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
                "integer, bool, string or char value",
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
            Value::Str(s) => {
                // Precision truncates the string (in chars) before
                // it is escaped and quoted.
                let truncated: String = match options.precision() {
                    Some(p) => s.chars().take(p as usize).collect(),
                    None => s.clone(),
                };
                Ok(pad_width(&format!("{truncated:?}"), options))
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
                "integer, bool, string, char or tuple value",
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

    /// Exponential formatting of an integer value (`{:e}` / `{:E}`).
    fn format_exp(
        &self,
        value: &Value,
        upper: bool,
        options: &FmtOptions,
    ) -> InterpResult<String> {
        let (negative, digits) = match value {
            Value::Int(v) => (*v < 0, v.unsigned_abs().to_string()),
            Value::Uint(v) => (false, v.to_string()),
            other => {
                return Err(InterpError::type_mismatch(
                    "exponential formatting",
                    "integer value",
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
