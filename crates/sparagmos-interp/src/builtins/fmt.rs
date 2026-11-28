//! The core::fmt formatting engine and its argument helpers.



use crate::error::{InterpError, InterpResult};
use crate::interp::{
    rc_handle_id,
    FmtBuilderKind, FmtCapture,
    Frame, Interpreter, PanicOrigin,
};
use crate::value::{FmtArguments, FmtTrait, PanicPayload, RefProj, RefTarget, Value};

use super::*;

/// The function name a function-pointer argument denotes (the
/// `catch_unwind` intrinsic's try/catch functions are fn items).
pub(super) fn expect_fn_ptr(value: &Value, name: &str) -> InterpResult<String> {
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
pub(super) fn expect_generic_ty(generic_args: &[String], name: &str) -> InterpResult<String> {
    expect_generic_ty_at(generic_args, name, 0)
}

/// The generic argument at `index` of a builtin call, rendered
/// as a type tag.
pub(super) fn expect_generic_ty_at(
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
/// the payload it carries: the concrete type tag, the concrete
/// type's `TypeId` segments when the value carries them, the
/// payload value, and the box value itself (handed back to the
/// caller on a failed by-value downcast).
pub(super) fn resolve_dyn_box(
    interp: &mut Interpreter,
    frames: &mut Vec<Frame>,
    value: &Value,
    name: &str,
) -> InterpResult<(String, Option<(u64, u64)>, Value, Value)> {
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
                base,
                concrete,
                type_id,
                ..
            } => return Ok((concrete, type_id, (*base).clone(), original)),
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

/// Whether a `dyn Any` value's concrete type is the type a
/// downcast asks for: the rendered tags decide, unless both
/// sides carry `TypeId` segments, in which case the segments
/// decide -- the tags erase the regions the segments
/// distinguish, so two region variants of one type share a tag
/// but never segments.
pub(super) fn dyn_any_matches(
    concrete: &str,
    concrete_id: Option<(u64, u64)>,
    want: &str,
    want_id: Option<(u64, u64)>,
) -> bool {
    match (concrete_id, want_id) {
        (Some(c), Some(w)) => c == w,
        _ => concrete == want,
    }
}

/// The pointee a box-shaped value denotes: the payload a
/// `DynRef` base in a box shape carries. An allocation-backed
/// box (a one-field aggregate chain bottoming out in a pointer)
/// denotes the value the pointer points at; an aggregate-shaped
/// box denotes its field value itself. Any other value denotes
/// itself.
pub(super) fn dyn_box_pointee(
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
/// Self type), and a unit-shaped marker field; a count argument
/// (`ArgumentType::Count`) is the variant-1 single-field shape
/// holding the count itself. Any other shape is not a raw
/// argument.
fn coerce_fmt_arg(value: &Value) -> Option<Value> {
    let mut current = value.clone();
    loop {
        // `ArgumentType::Count(n)`: checked at every level of
        // the wrapper descent, because the `Argument` struct
        // wraps the `ArgumentType` enum in one more
        // single-field layer.
        if let Value::Enum {
            variant: 1,
            fields,
        } = &current
        {
            if let [Value::Uint(n)] = fields.as_slice() {
                if let Ok(n) = u16::try_from(*n) {
                    return Some(Value::FmtCount(n));
                }
            }
        }
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
                // The impl-segment form states the trait and
                // the Self type in the name itself.
                if let Some((trait_, ty)) = impl_segment_fmt(&formatter) {
                    return Some(Value::FmtArg {
                        target: Box::new(target),
                        trait_,
                        ty,
                    });
                }
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

/// The (formatter name, erased value reference) pair a raw
/// `core::fmt::rt::Argument` aggregate carries, when `value`
/// has that shape: the same wrapper descent [`coerce_fmt_arg`]
/// performs, without the instance-name interpretation of the
/// formatter (which is what states the trait and the Self type
/// there). `None` for any other shape.
fn raw_argument_parts(value: &Value) -> Option<(String, Value)> {
    let mut current = value;
    loop {
        match current {
            Value::Enum { fields, .. } if fields.len() == 1 => {
                current = &fields[0];
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
                return Some((formatter, target));
            }
            _ => return None,
        }
    }
}

/// The fmt trait a trait *name* denotes, among the nine
/// traits a `format_args!` placeholder can name.
fn fmt_trait_named(name: &str) -> Option<FmtTrait> {
    Some(match name {
        "Display" => FmtTrait::Display,
        "Debug" => FmtTrait::Debug,
        "Octal" => FmtTrait::Octal,
        "LowerHex" => FmtTrait::LowerHex,
        "UpperHex" => FmtTrait::UpperHex,
        "Pointer" => FmtTrait::Pointer,
        "Binary" => FmtTrait::Binary,
        "LowerExp" => FmtTrait::LowerExp,
        "UpperExp" => FmtTrait::UpperExp,
        _ => return None,
    })
}

/// The (trait, Self type) a formatter name in the
/// *impl-segment* form states:
/// `std::array::<impl std::fmt::Debug for [i32; 3]>::fmt` —
/// the name a raw argument's formatter carries for an impl
/// declared on an array / slice / primitive type inside a
/// module, where the instance is named through its impl block
/// rather than as `<Self as Trait>::fmt`. The segment states
/// both the trait and the Self type, so no recovery is
/// needed. `None` for other name forms.
fn impl_segment_fmt(formatter: &str) -> Option<(FmtTrait, String)> {
    let start = formatter.find("<impl ")?;
    // The segment runs to its balancing `>`: the Self type
    // may itself carry generic arguments.
    let bytes = formatter.as_bytes();
    let mut depth = 0i32;
    let mut end = None;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        match b {
            b'<' => depth += 1,
            // The `>` of a `->` arrow (a function-pointer Self
            // type, `fn(i32) -> i32`) closes no segment.
            b'>' if i > 0 && bytes[i - 1] == b'-' => {}
            b'>' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end?;
    if &formatter[end + 1..] != "::fmt" {
        return None;
    }
    let inner = &formatter[start + "<impl ".len()..end];
    let (trait_path, self_ty) = inner.split_once(" for ")?;
    let trait_ = fmt_trait_named(trait_path.rsplit("::").next()?)?;
    Some((trait_, self_ty.to_string()))
}

/// The fmt trait a *bare trait-item* formatter name denotes
/// (`std::fmt::Display::fmt` and its `core::` spelling, and
/// the eight other traits a `format_args!` placeholder can
/// name). A body-less trait item's `FnDef` constant carries
/// this plain name when no collected body names the instance
/// that serves it — a library type's impl lives in the
/// distributed rlib without a MIR body, so conversion cannot
/// resolve the constant to the instance name an impl-qualified
/// spelling would state. Unlike an instance name it states no
/// Self type; the caller recovers that from the argument's
/// target place.
fn bare_fmt_trait(formatter: &str) -> Option<FmtTrait> {
    let name = formatter
        .strip_prefix("std::fmt::")
        .or_else(|| formatter.strip_prefix("core::fmt::"))?
        .strip_suffix("::fmt")?;
    fmt_trait_named(name)
}

/// Unwrap one newtype layer: a single-field aggregate denotes
/// its field (the wrappers core puts around the two halves of a
/// built `fmt::Arguments`).
pub(super) fn unwrap_newtype(value: Value) -> Value {
    match value {
        Value::Enum { fields, .. } if fields.len() == 1 => {
            fields.into_iter().next().expect("one field")
        }
        other => other,
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
pub(super) struct FmtOptions {
    pub(super) flags: u32,
    pub(super) width: u16,
    pub(super) precision: u16,
}

impl FmtOptions {
    /// core's default: fill `' '`, alignment unknown.
    pub(super) fn new() -> Self {
        FmtOptions {
            flags: (' ' as u32) | ALIGN_UNKNOWN,
            width: 0,
            precision: 0,
        }
    }
    pub(super) fn width(&self) -> Option<u16> {
        if self.flags & WIDTH_FLAG != 0 {
            Some(self.width)
        } else {
            None
        }
    }
    pub(super) fn precision(&self) -> Option<u16> {
        if self.flags & PRECISION_FLAG != 0 {
            Some(self.precision)
        } else {
            None
        }
    }
    pub(super) fn fill(&self) -> char {
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
    pub(super) fn sign_plus(&self) -> bool {
        self.flags & SIGN_PLUS != 0
    }
    pub(super) fn alternate(&self) -> bool {
        self.flags & ALTERNATE != 0
    }
    pub(super) fn zero_pad(&self) -> bool {
        self.flags & SIGN_AWARE_ZERO_PAD != 0
    }
}

impl FmtCapture {
    /// The placeholder options this capture was seeded with,
    /// in the fmt engine's terms.
    pub(super) fn options(&self) -> FmtOptions {
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
    /// Format a panic entry point's message arguments, with
    /// the in-hook mark held: natively the formatting happens
    /// inside the hook (the payload's lazy `fill` runs when
    /// the default hook reads the message), so a panic the
    /// formatting raises is a panic while processing a panic
    /// (see `Interpreter::fire_panic_hook`).
    pub(crate) fn format_panic_message(
        &mut self,
        frames: &mut Vec<Frame>,
        args: &FmtArguments,
    ) -> InterpResult<String> {
        let previous = self.in_panic_hook();
        self.set_in_panic_hook(true);
        let result = self.format_arguments(frames, args, 0);
        self.set_in_panic_hook(previous);
        result
    }

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

    /// The parts of `args` in template order (see
    /// `format_template_parts`): a plain string is one part.
    pub(crate) fn format_arguments_parts(
        &mut self,
        frames: &mut Vec<Frame>,
        args: &FmtArguments,
        indent: usize,
    ) -> (Vec<String>, Option<InterpError>) {
        match args {
            FmtArguments::Plain(s) => (vec![s.clone()], None),
            FmtArguments::Template { template, args } => {
                self.format_template_parts(frames, template, args, indent)
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
        let (parts, failure) = self.format_template_parts(frames, template, args, indent);
        match failure {
            Some(error) => Err(error),
            None => Ok(parts.join("")),
        }
    }

    /// The parts of a template in order -- literal pieces and
    /// formatted arguments as separate strings -- stopping at
    /// the first failure: the parts composed before it are
    /// returned alongside the error, so a streaming consumer
    /// (core's `fmt::write`, see `Builtin::FmtWrite`) can
    /// deliver them before reporting the failure, in the
    /// order the body itself composes and writes them.
    fn format_template_parts(
        &mut self,
        frames: &mut Vec<Frame>,
        template: &[u8],
        args: &[Value],
        indent: usize,
    ) -> (Vec<String>, Option<InterpError>) {
        let mut parts: Vec<String> = Vec::new();
        let mut arg_index = 0usize;
        let mut i = 0usize;
        macro_rules! attempt {
            ($e:expr) => {
                match $e {
                    Ok(value) => value,
                    Err(error) => return (parts, Some(error)),
                }
            };
        }
        while i < template.len() {
            let b = template[i];
            if b == 0 {
                break;
            }
            if b < 0x80 {
                // Short literal piece: one length byte, then the bytes.
                let len = b as usize;
                let piece = attempt!(template_piece(template, i + 1, len));
                parts.push(piece.to_string());
                i += 1 + len;
            } else if b == 0x80 {
                // Long literal piece: 0x80, u16 LE length, then bytes.
                let len = attempt!(read_u16(template, i + 1)) as usize;
                let piece = attempt!(template_piece(template, i + 3, len));
                parts.push(piece.to_string());
                i += 3 + len;
            } else if b == 0xC0 {
                // Placeholder with entirely default options.
                let arg = attempt!(template_arg(args, arg_index));
                let text = attempt!(self.format_arg(frames, arg, &FmtOptions::new(), indent));
                parts.push(text);
                arg_index += 1;
                i += 1;
            } else if b & 0xC0 == 0xC0 {
                // General placeholder: field-presence bits in the
                // first byte, then fields in the order flags, width,
                // precision, arg_index.
                let mut options = FmtOptions::new();
                let mut j = i + 1;
                if b & 0x01 != 0 {
                    options.flags = attempt!(read_u32(template, j));
                    j += 4;
                }
                if b & 0x02 != 0 {
                    let raw = attempt!(read_u16(template, j));
                    j += 2;
                    options.width = if b & 0x10 != 0 {
                        attempt!(self.count_arg(args, raw as usize))
                    } else {
                        raw
                    };
                    options.flags |= WIDTH_FLAG;
                }
                if b & 0x04 != 0 {
                    let raw = attempt!(read_u16(template, j));
                    j += 2;
                    options.precision = if b & 0x20 != 0 {
                        attempt!(self.count_arg(args, raw as usize))
                    } else {
                        raw
                    };
                    options.flags |= PRECISION_FLAG;
                }
                let mut this_arg = arg_index;
                if b & 0x08 != 0 {
                    this_arg = attempt!(read_u16(template, j)) as usize;
                    j += 2;
                }
                let arg = attempt!(template_arg(args, this_arg));
                let text = attempt!(self.format_arg(frames, arg, &options, indent));
                parts.push(text);
                arg_index += 1;
                i = j;
            } else {
                return (
                    parts,
                    Some(template_error("unexpected byte at a part boundary")),
                );
            }
        }
        (parts, None)
    }

    /// An indirect width/precision: the argument at `idx` must be a
    /// count argument (`Argument::from_usize`).
    fn count_arg(&self, args: &[Value], idx: usize) -> InterpResult<u16> {
        match args.get(idx) {
            Some(Value::FmtCount(n)) => Ok(*n),
            // A raw `ArgumentType::Count` aggregate in an
            // argument array a body built itself denotes the
            // same count (see `coerce_fmt_arg`); the placeholder
            // path coerces its arguments on use, and the count
            // path does the same here.
            Some(arg) => match coerce_fmt_arg(arg) {
                Some(Value::FmtCount(n)) => Ok(n),
                _ => Err(template_error(
                    "indirect width/precision does not name a count argument",
                )),
            },
            None => Err(template_error(
                "indirect width/precision does not name a count argument",
            )),
        }
    }

    /// [`coerce_fmt_arg`] with the Self-type recovery a bare
    /// trait-item formatter name needs. The formatter of a raw
    /// argument whose impl body was never collected names only
    /// the trait (`std::fmt::Display::fmt`), so the Self type
    /// comes from the place the argument's erased reference
    /// denotes: the frame body's declared type for that local,
    /// refined by the variable table conversion recorded from
    /// the body's debug info, which states the exact type
    /// (generic arguments and `char` included) where the IR
    /// type cannot — the IR renders `char` as `u32`. `None`
    /// when the type cannot be stated exactly (a value-backed
    /// target, a projection the declared types do not
    /// determine, or a `u32` local no variable entry
    /// disambiguates from `char`): the argument then keeps the
    /// existing explicit error instead of a guessed rendering.
    fn coerce_raw_fmt_arg(&self, frames: &[Frame], value: &Value) -> Option<Value> {
        if let Some(coerced) = coerce_fmt_arg(value) {
            return Some(coerced);
        }
        let (formatter, target) = raw_argument_parts(value)?;
        let trait_ = bare_fmt_trait(&formatter)?;
        let ty = self.fmt_arg_target_ty(frames, &target)?;
        Some(Value::FmtArg {
            target: Box::new(target),
            trait_,
            ty,
        })
    }

    /// The declared type a runtime projection denotes over a
    /// place of declared type `root`, walked the way the
    /// resolved read walks the value: a structural step first
    /// follows the reference layers the target's formation
    /// absorbed (a MIR place spells those as `Deref` steps; the
    /// runtime projection drops them, so the step applies to
    /// the pointee), then steps through the field, index or
    /// sub-slice it names. Returns the resulting type and the
    /// (ADT instance, variant, field) of the last ADT field
    /// step, whose recorded debug shape states the field's
    /// exact type tag where the IR type cannot. `None` when a
    /// step names a position the declared types do not
    /// determine.
    fn ref_projection_result_ty(
        &self,
        root: &crate::ir::Ty,
        projection: &[RefProj],
    ) -> Option<(crate::ir::Ty, Option<(String, usize, usize)>)> {
        use crate::ir::Ty;
        // The pointee type a structural step applies to.
        fn deref_ty(ty: &Ty) -> Option<Ty> {
            match ty {
                Ty::Ref(inner) => Some((**inner).clone()),
                _ => None,
            }
        }
        let mut ty = root.clone();
        let mut variant = 0usize;
        let mut last_field: Option<(String, usize, usize)> = None;
        for step in projection {
            match step {
                RefProj::Downcast(v) => {
                    variant = *v as usize;
                    last_field = None;
                }
                RefProj::View(t) => {
                    ty = t.clone();
                    variant = 0;
                    last_field = None;
                }
                RefProj::Field(i) => {
                    while let Some(inner) = deref_ty(&ty) {
                        ty = inner;
                    }
                    match &ty {
                        Ty::Adt(name) => {
                            let name = name.clone();
                            ty = self.adt_field_ty(&ty, variant, *i)?;
                            last_field = Some((name, variant, *i));
                        }
                        Ty::Tuple(_) => {
                            ty = self.adt_field_ty(&ty, variant, *i)?;
                            last_field = None;
                        }
                        _ => return None,
                    }
                    variant = 0;
                }
                RefProj::Index(_) => {
                    while let Some(inner) = deref_ty(&ty) {
                        ty = inner;
                    }
                    match &ty {
                        Ty::Array(elem, _) | Ty::Slice(elem) => {
                            ty = (**elem).clone();
                        }
                        _ => return None,
                    }
                    last_field = None;
                }
                RefProj::Subslice { len, .. } => {
                    while let Some(inner) = deref_ty(&ty) {
                        ty = inner;
                    }
                    match &ty {
                        Ty::Array(elem, _) => {
                            ty = Ty::Array(elem.clone(), u64::try_from(*len).ok()?);
                        }
                        Ty::Slice(elem) => {
                            ty = Ty::Slice(elem.clone());
                        }
                        _ => return None,
                    }
                    last_field = None;
                }
                _ => return None,
            }
        }
        Some((ty, last_field))
    }

    /// The exact type tag at the end of a field projection
    /// over the local `local` of `body`, read from the debug
    /// shapes of the ADT instances the projection steps
    /// through: the local's variable-table entry names its
    /// exact instance (generic arguments included), and the
    /// shape recorded for an instance states each field's
    /// type in that instance — the channel that states a
    /// field whose type is a generic parameter, which the
    /// declared-type walk cannot. Only field and downcast
    /// steps compose in this domain; any other step keeps the
    /// recovery unstated.
    fn projected_shape_tag(
        &self,
        body: &crate::ir::Body,
        local: usize,
        projection: &[RefProj],
    ) -> Option<String> {
        let mut tag: Option<String> = None;
        for entry in &body.debug_vars {
            if let crate::ir::DebugVar::Place(l, proj) = &entry.source {
                if *l == local && proj.is_empty() {
                    if let Some(ty) = &entry.ty {
                        tag = Some(ty.clone());
                        break;
                    }
                }
            }
        }
        let mut tag = tag?;
        let mut variant = 0usize;
        for step in projection {
            match step {
                RefProj::Downcast(v) => {
                    variant = *v as usize;
                }
                RefProj::Field(i) => {
                    tag = self.adt_field_tag(&tag, variant, *i)?;
                    variant = 0;
                }
                _ => return None,
            }
        }
        Some(tag)
    }

    /// The declared type tag of the place a raw fmt argument's
    /// erased reference denotes (see
    /// [`Self::coerce_raw_fmt_arg`]).
    fn fmt_arg_target_ty(&self, frames: &[Frame], target: &Value) -> Option<String> {
        match target {
            Value::Ref(RefTarget::Place {
                frame,
                local,
                projection,
            }) => {
                let frame = frames.iter().find(|f| f.id == *frame)?;
                let body = self.program().get(&frame.func)?;
                if !projection.is_empty() {
                    // A projected place (a field of a local, an
                    // element of an array): the Self type is the
                    // local's declared type walked over the
                    // projection, the same walk the resolved
                    // read performs over the value.
                    let root = body.local_tys.get(*local)?;
                    if let Some((ty, last_field)) =
                        self.ref_projection_result_ty(root, projection)
                    {
                        return match ir_ty_tag(&ty) {
                            Some(tag) => Some(tag),
                            None => {
                                // The walk's one inexact
                                // rendering is a `char` / `u32`
                                // the IR type spells alike
                                // (also nested inside a field's
                                // own type): the field's recorded
                                // debug shape states its exact
                                // tag.
                                let (name, variant, field) = last_field?;
                                self.adt_field_tag(&name, variant, field)
                            }
                        };
                    }
                    // The declared-type walk cannot state a
                    // field whose type is one of the ADT's
                    // generic parameters (the field-type table
                    // records no type for it). The base local's
                    // variable-table entry names the exact
                    // instance, and the debug shape recorded
                    // for that instance states each field's
                    // type in it.
                    return self.projected_shape_tag(body, *local, projection);
                }
                for entry in &body.debug_vars {
                    if let crate::ir::DebugVar::Place(l, proj) = &entry.source {
                        if *l == *local && proj.is_empty() {
                            if let Some(ty) = &entry.ty {
                                return Some(ty.clone());
                            }
                            if entry.is_char {
                                return Some("char".to_string());
                            }
                        }
                    }
                }
                let ty = body.local_tys.get(*local)?;
                // The IR type renders `char` as `Uint(U32)`, so
                // that one case is decided by the char table
                // conversion recorded from the MIR types (see
                // `ir::Body::char_locals`); a body without the
                // table (hand-built) keeps the honest refusal
                // `ir_ty_tag` makes for the ambiguous case.
                if matches!(ty, crate::ir::Ty::Uint(crate::ir::UintTy::U32)) {
                    return match body.char_locals.get(*local) {
                        Some(true) => Some("char".to_string()),
                        Some(false) => Some("u32".to_string()),
                        None => None,
                    };
                }
                ir_ty_tag(ty)
            }
            Value::Ref(RefTarget::Static { name, projection }) => {
                if !projection.is_empty() {
                    return None;
                }
                // The declared type conversion rendered from
                // the static's MIR type (exact where `char`
                // and generic arguments are concerned). The
                // reference's spelling is resolved to the
                // registered key first: a reference path is
                // spelled from the referencing crate, so an
                // `extern crate .. as ..` alias segment or a
                // `#DefId` disambiguator can separate the two
                // (see `resolve_static_key`).
                let program = self.program();
                if let Some(key) = resolve_static_key(program.static_debug_tags.keys(), name) {
                    return Some(program.static_debug_tags[key].clone());
                }
                let key = resolve_static_key(program.static_tys.keys(), name)?;
                let ty = program.static_tys.get(key)?;
                ir_ty_tag(ty)
            }
            _ => None,
        }
    }

    /// `ToSocketAddrs::to_socket_addrs` (see
    /// [`net_trait_method`]): resolve the receiver into the
    /// modelled `vec::IntoIter<SocketAddr>` inside the
    /// `io::Result` the trait returns.
    pub(super) fn exec_to_socket_addrs(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        receiver: &Value,
    ) -> InterpResult<Value> {
        let value = self.resolve_value(frames, receiver)?;
        match self.resolve_socket_addrs(frames, self_ty, &value)? {
            Ok(addrs) => Ok(io_result_ok(owned_iter_state(addrs))),
            Err(()) => Ok(io_result_err(io_simple_error_value(
                ERROR_KIND_INVALID_INPUT,
            ))),
        }
    }

    /// The addresses a `ToSocketAddrs` receiver denotes:
    /// `Ok(addrs)` on success, `Err(())` for the outcomes
    /// std reports as an `io::Error` (an unparseable address,
    /// a name the resolver cannot resolve). A receiver shape
    /// the model does not know is an interpreter error, never
    /// a silent resolution failure. The receiver forms are
    /// std's own impls: a string (`str` / `String`, reached
    /// by reference or by value), a single address
    /// (`SocketAddr` / `SocketAddrV4` / `SocketAddrV6`), an
    /// `(address, port)` pair, and a slice of addresses.
    pub(super) fn resolve_socket_addrs(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        value: &Value,
    ) -> InterpResult<Result<Vec<Value>, ()>> {
        let shape_error = |found: &Value| {
            InterpError::type_mismatch(
                "ToSocketAddrs::to_socket_addrs",
                "string, socket address, (address, port) pair or address slice receiver",
                found.kind_name(),
            )
        };
        // A single address value, in any of its forms.
        if let Some(addr) = socket_addr_value(value) {
            return Ok(Ok(vec![addr]));
        }
        // A string receiver, in whichever form the model
        // holds it (a `str` value, a `String` aggregate).
        if let Ok(text) = self.resolve_string(frames, value) {
            return Ok(parse_socket_addr_str(&text));
        }
        match value {
            Value::FatRef { .. } => {
                let resolved = self.resolve_value(frames, value)?;
                let items = value_elements(&resolved)?;
                let mut addrs = Vec::with_capacity(items.len());
                for item in &items {
                    match socket_addr_value(item) {
                        Some(addr) => addrs.push(addr),
                        None => return Err(shape_error(item)),
                    }
                }
                Ok(Ok(addrs))
            }
            Value::Tuple(items) if items.len() == 2 => {
                let Some(port) = net_port(&items[1]) else {
                    return Err(shape_error(&items[1]));
                };
                if let Ok(text) = self.resolve_string(frames, &items[0]) {
                    return Ok(parse_socket_host(&text, port));
                }
                match ip_addr_with_port(&items[0], port) {
                    Some(addr) => Ok(Ok(vec![addr])),
                    None => Err(shape_error(&items[0])),
                }
            }
            _ => Err(InterpError::type_mismatch(
                format!("ToSocketAddrs::to_socket_addrs for {self_ty}"),
                "string, socket address, (address, port) pair or address slice receiver",
                value.kind_name(),
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
            if let Some(coerced) = self.coerce_raw_fmt_arg(frames.as_slice(), arg) {
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
                // A stored fat pointer is the argument's value
                // itself (a slice): resolving it would read the
                // stored aggregate and lose a regrouping cast's
                // element shape, so hand the fat pointer itself
                // to the formatter, as the element-base cases
                // above do (its own arm resolves it when the
                // view does not regroup it).
                if *trait_ == FmtTrait::Debug && pointee_tag(ty).starts_with('[') {
                    if let Ok(resolved) = self.deref_value(frames, target) {
                        if let Ok(stored) = self.read_resolved(frames, &resolved) {
                            if let Some(fat) = self.denoted_fat_ref(frames, &stored) {
                                return self.format_value(
                                    frames, &fat, *trait_, ty, options, indent,
                                );
                            }
                        }
                    }
                }
                // A scalar argument read through a
                // byte-displaced pointer: the argument's type
                // tag is the load's width, and the byte view of
                // the base value's storage composes it (the
                // resolved read has no byte-level view of that
                // storage). The `Pointer` trait is excepted:
                // it asks for the argument's address, not its
                // pointee, so it keeps the plain path.
                let pointee = match if *trait_ == FmtTrait::Pointer {
                    None
                } else {
                    self.byte_view_fmt_arg(frames, target, ty)
                } {
                    Some(view) => view?,
                    None => match self.regrouped_fmt_arg(frames, target, ty) {
                        Some(view) => view?,
                        None => self.resolve_value(frames, target)?,
                    },
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
    pub(super) fn formatter_id_of(&self, frames: &[Frame], value: &Value) -> InterpResult<u64> {
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
    pub(super) fn builder_id_of(&self, frames: &[Frame], value: &Value) -> InterpResult<u64> {
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
    pub(super) fn capture_of(&self, id: u64) -> InterpResult<FmtCapture> {
        self.formatter_snapshot(id).ok_or_else(|| {
            unsupported("a Formatter method on a capture that already finished".to_string())
        })
    }

    /// Append text to a captured formatter: into its buffer, or
    /// -- for a streaming capture (a `Termination` report
    /// rendering's own formatter, see [`FmtCapture::stream`]) --
    /// straight into the captured stderr, at the position the
    /// write happens, interleaved with the body's own stderr
    /// writes as natively.
    pub(super) fn formatter_push(&mut self, id: u64, text: &str) {
        if self
            .formatter_snapshot(id)
            .map(|capture| capture.stream)
            .unwrap_or(false)
        {
            self.push_error_output(text);
        } else {
            self.with_formatter(id, |capture| capture.buffer.push_str(text));
        }
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
        // A capture created under the stream arm streams (see
        // [`FmtCapture::stream`]): the arm is one-shot, so bodies
        // dispatched inside this one -- a nested value's body, an
        // inner `format!` -- buffer and compose as usual.
        let stream = self.fmt_stream_arm;
        self.fmt_stream_arm = false;
        let id = self.new_formatter(FmtCapture {
            buffer: String::new(),
            flags: options.flags,
            width: options.width,
            precision: options.precision,
            indent,
            self_tag: pointee_tag(ty).to_string(),
            stream,
        });
        let formatter = Value::Ref(RefTarget::value(Value::Formatter(id)));
        let receiver = Value::Ref(RefTarget::value(value.clone()));
        let result = self.call_sync(frames, &call, vec![receiver, formatter]);
        let capture = self.take_formatter(id);
        let result = result?;
        match result {
            Value::Enum { variant: 0, .. } => {}
            _ => {
                return Err(InterpError::FmtReturnedError {
                    trait_: format!("{trait_:?}"),
                    ty: ty.to_string(),
                })
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

    /// The fat pointer a trait-object data pointer denotes,
    /// when it denotes one directly or through the reference
    /// to the place holding it (a `&dyn Debug` built over a
    /// `&&[T]` field): the slice value a regrouping cast's
    /// element shape rides on, which resolving the pointer
    /// would collapse into the stored aggregate.
    fn denoted_fat_ref(&self, frames: &[Frame], base: &Value) -> Option<Value> {
        // The pointer may denote the fat pointer through a
        // short reference chain (the place holding it, a
        // reference to the value a temporary read out of the
        // field): follow the denotation to the slice value.
        let mut current = base.clone();
        for _ in 0..4 {
            if matches!(current, Value::FatRef { .. }) {
                return Some(current);
            }
            if !matches!(current, Value::Ref(_)) {
                return None;
            }
            let target = self.deref_value(frames, &current).ok()?;
            current = self.read_resolved(frames, &target).ok()?;
        }
        if matches!(current, Value::FatRef { .. }) {
            return Some(current);
        }
        None
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
            // A slice field's data pointer is the fat pointer
            // itself: resolving it would read the stored
            // aggregate and lose a regrouping cast's element
            // shape, so the fat pointer formats as the slice
            // it denotes.
            if pointee_tag(tag).starts_with('[') {
                if let Some(fat) = self.denoted_fat_ref(frames, base) {
                    return self.format_value(
                        frames, &fat, FmtTrait::Debug, tag, options, indent,
                    );
                }
            }
            let resolved = self.resolve_value(frames, base)?;
            // As in `format_value`'s `DynRef` arm, the handle
            // shaping needs the allocation's contents to license
            // it (`rc_inner_id`): the resolved value's shape
            // alone cannot tell a reference-counted handle from
            // a plain aggregate that shares it (a `CString`).
            let shaped = if let Some(id) = self.rc_inner_id(&resolved) {
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
    pub(super) fn exec_debug_struct_finish(
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
    pub(super) fn exec_debug_struct_fields_finish(
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
    pub(super) fn exec_debug_tuple_finish(
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
    pub(super) fn exec_debug_tuple_fields_finish(
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
    pub(super) fn builder_entry(
        &mut self,
        frames: &mut Vec<Frame>,
        builder: u64,
        entry: &Value,
    ) -> InterpResult<()> {
        let text = self.builder_entry_text(frames, builder, entry)?;
        self.builder_write_entry(builder, &text)
    }

    /// `DebugTuple::field` is the tuple builder's `entry`.
    pub(super) fn builder_tuple_field(
        &mut self,
        frames: &mut Vec<Frame>,
        builder: u64,
        entry: &Value,
    ) -> InterpResult<()> {
        self.builder_entry(frames, builder, entry)
    }

    /// `DebugMap::entry`: append one `key: value` entry.
    pub(super) fn builder_map_entry(
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
    pub(super) fn builder_struct_field(
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
    pub(super) fn builder_finish(&mut self, builder: u64, exhaustive: bool) -> InterpResult<()> {
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

    /// The panic core's `default_write_fmt` raises when a
    /// formatting trait implementation returns `Err(fmt::Error)`
    /// and the sink itself did not fail: core's own message, at
    /// its fixed position in `core::io::write` (pinned to the
    /// toolchain build, as the runtime's other fixed library
    /// positions are). The stamp, origin and backtrace notes
    /// are recorded here, as for the detached rendering in
    /// `format_debug_detached`, which raises the same panic
    /// for the report's sink; the caller returns the error
    /// into the drive loop, which runs the unwind (and fires
    /// the panic hook, which the detached path fires itself).
    pub(crate) fn fmt_returned_error_panic(&mut self) -> InterpError {
        let message = "a formatting trait implementation returned an error \
             when the underlying stream did not";
        self.note_panic_stamp(Value::Ref(RefTarget::value(Value::Enum {
            variant: 0,
            fields: vec![
                Value::Str(
                    "/rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library/core/src/io/write.rs"
                        .to_string(),
                ),
                Value::Uint(411),
                Value::Uint(17),
            ],
        })));
        self.note_panic_origin(PanicOrigin::PanicFmt);
        self.note_panic_no_backtrace(false);
        InterpError::panic_str_payload(message)
    }

    /// Render `value` through the `Debug` formatting engine
    /// under the type tag `ty`, outside any running frame: the
    /// rendering `Termination::report` performs for a `Result`
    /// entry's `Err` payload, applied by the interpreter that
    /// executed the entry after the entry body has returned and
    /// its frames are gone, while its allocations and statics
    /// still live. A payload that reaches into the finished run's
    /// frames does not render here and yields the engine's error,
    /// exactly as a live formatting of an unresolvable value does.
    ///
    /// The rendering streams when it goes through a guest `Debug`
    /// body for the payload itself: natively the report's
    /// formatter writes straight to stderr, so the body's writes
    /// land in the captured stderr as they happen, interleaved
    /// with the body's own stderr writes, and text written before
    /// a panic or a `process::exit` in the body stands. In that
    /// case the result is `None` -- the text is already in the
    /// captured stderr, and only the report line's ending is the
    /// caller's to write. A rendering the structural engine
    /// composes (no guest body for the payload itself) yields the
    /// composed text as `Some`, for the caller to write.
    ///
    /// A payload `fmt` body that returns `Err(fmt::Error)` is
    /// the one rendering failure this does not pass through as
    /// an ordinary error: the report's sink is a `write_fmt` to
    /// stderr whose own writes do not fail, and core's
    /// `write_fmt` panics on a formatting error there, so the
    /// rendering's outcome is that panic (see
    /// [`InterpError::FmtReturnedError`]).
    pub fn format_debug_detached(
        &mut self,
        value: &Value,
        ty: &str,
    ) -> InterpResult<Option<String>> {
        let saved = self.fmt_stream_arm;
        self.fmt_stream_arm = true;
        let result = self.format_value(
            &mut Vec::new(),
            value,
            FmtTrait::Debug,
            ty,
            &FmtOptions::new(),
            0,
        );
        // The arm survives only when no capture took it: that is
        // the structural case, whose composed text is the result.
        // A capture that took it streamed the whole rendering
        // (its buffer stays empty), so the composed text is empty
        // there and `None` reports the streaming.
        let streamed = !self.fmt_stream_arm;
        self.fmt_stream_arm = saved;
        match result {
            Ok(text) => Ok(if streamed { None } else { Some(text) }),
            // The report renders through a `write_fmt` to stderr,
            // whose own writes do not fail here; a payload `fmt`
            // implementation that returns `Err` therefore panics
            // natively, out of core's `write_fmt` -- it does not
            // fall back to the payload's structural form, and the
            // report does not complete. Raise that panic (with
            // core's own message) as the rendering's outcome, so
            // the caller runs the payload's drop as the panic's
            // unwind cleanup, as it does for a panic the body
            // itself raises.
            Err(InterpError::FmtReturnedError { .. }) => {
                // The panic is a real one (core's `write_fmt`
                // raises it at a fixed position in
                // `core::io::write`, pinned to the toolchain
                // build, as the runtime's own nounwind positions
                // are): fire the hook for it here, since no
                // drive loop will see it — the detached rendering
                // is where it was raised.
                let message = "a formatting trait implementation returned an error \
                     when the underlying stream did not";
                self.note_panic_stamp(Value::Ref(RefTarget::value(Value::Enum {
                    variant: 0,
                    fields: vec![
                        Value::Str(
                            "/rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library/core/src/io/write.rs"
                                .to_string(),
                        ),
                        Value::Uint(411),
                        Value::Uint(17),
                    ],
                })));
                self.note_panic_origin(PanicOrigin::PanicFmt);
                self.note_panic_no_backtrace(false);
                let payload = Some(PanicPayload {
                    ty: "&str".to_string(),
                    value: Box::new(Value::Str(message.to_string())),
                    type_id: None,
                });
                self.fire_panic_hook(&mut Vec::new(), 0, message, &payload);
                Err(InterpError::panic_str_payload(message))
            }
            Err(e) => Err(e),
        }
    }

    /// Format one argument value through a formatting trait.
    /// `indent` is the current pretty-print (`{:#?}`) indent level.
    pub(super) fn format_value(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        trait_: FmtTrait,
        ty: &str,
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<String> {
        // `CStr` / `CString` `Debug` is a fixed function of the
        // string's bytes (core prints the quoted, ASCII-escaped
        // text), and the value model stores those bytes behind
        // several shapes, so it is handled here, before the
        // shape-driven dispatch below.
        if trait_ == FmtTrait::Debug {
            if let Some(text) = self.format_c_string(frames, value, ty)? {
                return Ok(text);
            }
        }
        // A `Backtrace` value is the status and the rendered
        // trace the capture builtin recorded (see
        // `backtrace_value`); both `Display` and `Debug`
        // print that rendering, as std's do.
        if pointee_tag(ty) == "std::backtrace::Backtrace"
            && matches!(trait_, FmtTrait::Display | FmtTrait::Debug)
        {
            return Ok(backtrace_display(value));
        }
        // std's `Debug` for `process::ExitStatus` is a manual
        // impl: it names the raw wait status the value holds
        // (`ExitStatus(unix_wait_status(N))` on unix), not the
        // wrapper's fields.
        if trait_ == FmtTrait::Debug && pointee_tag(ty) == "std::process::ExitStatus" {
            if let Value::Enum { fields, .. } = value {
                if let [Value::Enum {
                    fields: inner, ..
                }] = fields.as_slice()
                {
                    if let [Value::Int(raw)] = inner.as_slice() {
                        return Ok(format!("ExitStatus(unix_wait_status({raw}))"));
                    }
                }
            }
        }
        // std's `Display` for `process::ExitStatus` is a manual
        // impl too: it reports the outcome the raw wait status
        // encodes -- the exit code, or the signal that ended
        // the process (with the core-dump note, when the status
        // carries it) -- in the sys impl's shape.
        if trait_ == FmtTrait::Display && pointee_tag(ty) == "std::process::ExitStatus" {
            if let Value::Enum { fields, .. } = value {
                if let [Value::Enum {
                    fields: inner, ..
                }] = fields.as_slice()
                {
                    if let [Value::Int(raw)] = inner.as_slice() {
                        return Ok(exit_status_display(*raw));
                    }
                }
            }
        }
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
        // A one-field wrapper aggregate around a scalar, tuple
        // or array payload, formatted as the payload type: the
        // storage form raw storage leaves behind (a
        // `MaybeUninit` / `ManuallyDrop` payload slot, an
        // `UnsafeCell`), where the model's value for the type
        // is the payload itself -- a vector element written
        // through `spare_capacity_mut` and read back after
        // `set_len` is the canonical case. The wrapper is
        // transparent to the type's own formatting, exactly as
        // it is to a typed place's store (see
        // `normalize_for_ty`); the payload is the value at its
        // own declared width, so formatting it under the tag
        // reinterprets nothing. ADT tags are excluded: a
        // one-field aggregate may be the ADT's own value (a
        // newtype, a one-variant enum).
        if let Value::Enum {
            variant: 0,
            fields,
        } = value
        {
            if fields.len() == 1 {
                if let Some(payload) = storage_wrapper_payload(&fields[0], pointee_tag(ty)) {
                    return self.format_value(frames, &payload, trait_, ty, options, indent);
                }
            }
        }
        // A non-data pointer (null, or a pointer known only by
        // its address) denotes no value to resolve to: the
        // generic reference arm below would resolve it to itself
        // and recurse without bound. The pointer value itself is
        // its only formatting: `Pointer` prints it, and `Debug`
        // for a raw pointer is defined as the same address text.
        // Every other trait is an explicit error (raw pointers
        // implement no `Display` / radix formatting).
        if let Value::Ref(target) = value {
            let address = match target {
                RefTarget::Null => Some(0),
                RefTarget::Address(addr) => Some(*addr),
                _ => None,
            };
            if let Some(address) = address {
                // A reference to a zero-sized pointee at a
                // storage-less address (the dangling pointer
                // std serves a zero-sized allocation with)
                // denotes the pointee's single value, which
                // is what its formatting runs on. A raw
                // pointer keeps the address forms below: its
                // own `Debug` is the address text, and it has
                // no other fmt impl.
                let raw_pointer = {
                    let t = ty.trim_start();
                    t.starts_with("*const ") || t.starts_with("*mut ")
                };
                let storage_less = match target {
                    RefTarget::Null => true,
                    RefTarget::Address(addr) => self.address_owner(*addr).is_none(),
                    _ => false,
                };
                if !raw_pointer && storage_less {
                    if let Some(single) = ty_from_tag(pointee_tag(ty))
                        .and_then(|t| self.zst_single_value(&t))
                    {
                        return self.format_value(
                            frames,
                            &single,
                            trait_,
                            pointee_tag(ty),
                            options,
                            indent,
                        );
                    }
                }
                return match trait_ {
                    FmtTrait::Pointer | FmtTrait::Debug => Ok(format_pointer(address, options)),
                    FmtTrait::DebugNoop => Ok(String::new()),
                    _ => Err(unsupported(format!(
                        "{trait_:?} formatting for a pointer that denotes no data"
                    ))),
                };
            }
        }
        // A function pointer's only formatting is its address:
        // `Debug` for a function pointer prints the address it
        // denotes, and `Pointer` is the same text. The model
        // has no machine addresses; the synthetic address of
        // the named function is the deterministic stand-in the
        // pointer-identity model already assigns it, so two
        // formattings of the same function agree and distinct
        // functions differ, which is all a guest can observe.
        if let Value::FnPtr(_) = value {
            return match trait_ {
                FmtTrait::Debug | FmtTrait::Pointer => {
                    let address = crate::interp::synthetic_address(value).unwrap_or(0);
                    Ok(format_pointer(address as u64, options))
                }
                FmtTrait::DebugNoop => Ok(String::new()),
                _ => Err(unsupported(format!(
                    "{trait_:?} formatting for a function pointer"
                ))),
            };
        }
        // References format as their pointee for every trait here.
        if matches!(value, Value::Ref(_)) {
            // A reference denoting the element of a same-width
            // punned scalar view (a `&[u32]` over an `f32`
            // place, Issue #479) denotes the storage's bits
            // under the element kind, not the backing scalar:
            // a consumer that resolves the reference through
            // this arm (a sequence or aggregate element, above
            // all) formats the punned reading. The direct
            // argument path judges the same shape through the
            // byte view; this arm covers the consumers that
            // arrive with only the value and its tag.
            if let Ok(target) = self.deref_value(frames, value) {
                if let Some(punned) =
                    self.punned_scalar_element(frames, &target, pointee_tag(ty))
                {
                    return self.format_value(
                        frames,
                        &punned,
                        trait_,
                        pointee_tag(ty),
                        options,
                        indent,
                    );
                }
            }
            let pointee = self.resolve_value(frames, value)?;
            return self.format_value(frames, &pointee, trait_, ty, options, indent);
        }
        // A `String` in its real aggregate form (see
        // `string_from_real_aggregate`) formats as the string it
        // denotes, for every trait `String` / `str` formatting
        // defines. The same denotation covers a `str` pointee
        // over such storage, and the bare element form (and the
        // sparse form a growth of it leaves) that
        // `String::as_mut_vec` / `str::as_bytes_mut` leave in
        // the string's storage, which denotes the string
        // directly, with no wrapper around it.
        if matches!(trait_, FmtTrait::Display | FmtTrait::Debug) {
            let plain = pointee_tag(ty);
            if plain == "std::string::String"
                || plain == "alloc::string::String"
                || plain == "str"
            {
                let text = match value {
                    Value::Enum {
                        variant: 0,
                        fields,
                    } => match self.string_from_real_aggregate(frames, value)? {
                        Some(text) => Some(text),
                        None => match fields.first() {
                            Some(vec) => self.string_from_element_form(vec),
                            None => None,
                        }
                        .or_else(|| self.string_from_element_form(value)),
                    },
                    Value::SparseVec { .. } => self.string_from_element_form(value),
                    // The array form, too: a `str` tail field of
                    // a DST struct over a sized byte array (the
                    // sized twin's storage) reaches the fmt
                    // machinery as that array's value -- a tuple
                    // of byte elements -- which denotes the
                    // string exactly as the element form does.
                    // Any other shape declines through
                    // `string_from_element_form` itself.
                    _ => self.string_from_element_form(value),
                };
                if let Some(text) = text {
                    let decoded = Value::Str(text);
                    return self.format_value(
                        frames, &decoded, trait_, ty, options, indent,
                    );
                }
            }
        }
        // `ThinBox<T>`'s `Display` and `Debug` impls forward
        // to the pointee through `Deref` (they are the type's
        // only fmt impls): format the dereferenced value with
        // the pointee's tag, exactly as the forwarding body
        // would.
        if matches!(trait_, FmtTrait::Display | FmtTrait::Debug) {
            let plain = pointee_tag(ty);
            if let Some(pointee_ty) = plain
                .strip_prefix("std::boxed::ThinBox<")
                .and_then(|rest| rest.strip_suffix('>'))
            {
                if matches!(value, Value::Enum { .. }) {
                    let derefed =
                        self.thin_box_deref(frames, value, plain, "formatting a ThinBox")?;
                    return self.format_value(
                        frames,
                        &derefed,
                        trait_,
                        pointee_ty,
                        options,
                        indent,
                    );
                }
            }
        }
        // A fat pointer whose tag names one of the std
        // byte-string wrapper types denotes that wrapper's
        // content, not a slice (an `&OsStr` value is a fat
        // pointer over its bytes): the string-content
        // rendering judges those types for every other value
        // form, and the slice arm below would otherwise
        // pre-empt it for this one form.
        if matches!(value, Value::FatRef { .. }) {
            if let Some(text) =
                self.format_string_content(frames, value, trait_, ty, options)?
            {
                return Ok(text);
            }
        }
        if let Value::FatRef { base, .. } = value {
            // A slice value: only Debug is defined for it (a list).
            if trait_ != FmtTrait::Debug {
                return Err(unsupported(format!(
                    "{trait_:?} formatting for a slice value"
                )));
            }
            let elem = slice_elem_tag(pointee_tag(ty));
            // A fat pointer whose formation regrouped its
            // storage's elements under a different element
            // shape (a pointer cast: the `[[u8; 3]; 2]` storage
            // ranged over as `[[u8; 2]]`): the stored elements
            // are the birth shape's, so the byte view rebuilds
            // them under the tag's element type, as it does for
            // slice comparisons, and the rebuilt elements format
            // as the sequence they are.
            if let Some(elem_ty) = ty_from_tag(elem) {
                if let Some(items) =
                    self.foreign_slice_elements(frames, value, &elem_ty)
                {
                    return self.format_debug_seq(
                        frames,
                        &Value::Tuple(items),
                        options,
                        indent,
                        '[',
                        ']',
                        elem,
                    );
                }
            }
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
                // A slice trait object's data pointer is the
                // fat pointer itself: resolving it would read
                // the stored aggregate and lose a regrouping
                // cast's element shape, so the fat pointer
                // formats as the slice it denotes.
                if pointee_tag(concrete).starts_with('[') {
                    if let Some(fat) = self.denoted_fat_ref(frames, base) {
                        return self.format_value(
                            frames, &fat, trait_, concrete, options, indent,
                        );
                    }
                }
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
                // An aggregate-built box base (a `Box::from_raw`
                // product, or the sized box a downcast rebuilds)
                // holds its `Unique` at field 0 beside the
                // allocator field, so the resolved base is the box
                // aggregate itself, not the pointee: name the
                // pointee through the data-pointer extraction,
                // the same extraction `deref_value` applies to
                // this base shape.
                let pointee = if matches!(
                    base.as_ref(),
                    Value::Enum { fields, .. } if fields.len() > 1
                ) {
                    self.dyn_data_pointer(base)
                } else {
                    pointee
                };
                // The base is the data pointer in the shape the
                // sized value had: a `Box<dyn Trait>` base is the
                // box aggregate (a one-field wrapper around the
                // pointee reference) and an `Rc`/`Arc` base is the
                // handle, so shape the value the way method
                // dispatch shapes its receiver (see
                // `ref_shaped_receiver` in the interpreter). The
                // handle shaping is licensed by the allocation's
                // contents, not by the value's shape alone
                // (`rc_inner_id`): a plain value can share the
                // handle's shape -- a `CString` is a one-field
                // aggregate over a one-field box aggregate over
                // an allocation pointer -- and reading such a
                // value as a handle would name a byte inside its
                // buffer instead of the value itself.
                let pointee = if let Some(id) = self.rc_inner_id(&pointee) {
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
        if let Some(text) =
            self.format_io_error_custom(frames, value, trait_, ty, options, indent)?
        {
            return Ok(text);
        }
        if let Some(text) = self.format_io_error(value, trait_, ty, options, indent) {
            return Ok(text);
        }
        // `std::net` address values (`Ipv4Addr`, `IpAddr`,
        // `SocketAddrV4`, `SocketAddr`): their `Display` and
        // `Debug` are a fixed function of the stored octets
        // (and port), produced here like the other std shapes
        // the structural engine models. `Debug` for these
        // types is defined as the `Display` text in core.
        if let Some(text) = format_net_addr(value, trait_, ty, options) {
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
        // The `core::num` wrappers (`Wrapping<T>`,
        // `Saturating<T>`) forward the fmt traits they
        // implement to the inner value: format the held value
        // under `T`'s tag.
        if let Some(text) = self.format_num_wrapper(frames, value, trait_, ty, options, indent)? {
            return Ok(text);
        }
        // The std byte-string wrapper types (`PathBuf`, `Path`,
        // `OsString`, `OsStr`) define their `Debug` over the
        // string content they hold, and the `std::path::Display`
        // helper defines its `Display` as the lossy content:
        // render the extracted bytes.
        if let Some(text) = self.format_string_content(frames, value, trait_, ty, options)? {
            return Ok(text);
        }
        // The escape iterators' `Display` is the drain of the
        // iterator itself: core defines the `str` types' as
        // `clone().try_for_each(|c| f.write_char(c))` and the
        // `char` types' as the inner buffer state's `Display`,
        // which writes the remaining chars in order. Every one
        // of these types' steps is modelled, so the formatting
        // is the drain of the value through the modelled step
        // (on the formatting argument's own snapshot — the
        // guest's value is not consumed).
        if trait_ == FmtTrait::Display {
            if let Some(text) = self.format_escape_display(frames, value, ty, options)? {
                return Ok(text);
            }
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
        // A deref-forwarding guard value (`std::cell::Ref` /
        // `RefMut`): core's `Display` and `Debug` impls for the
        // guard types format the guarded value, and no guest
        // body serves a core Self type, so the structural
        // engine forwards here -- the aggregate's first field
        // is the reference to the guarded value; resolve it
        // and format the pointee under the guard's element
        // type. Any other trait keeps the existing behaviour
        // (the guard types implement no other fmt trait), as
        // does a guard-shaped value whose first field is not
        // a reference.
        if matches!(trait_, FmtTrait::Display | FmtTrait::Debug) {
            if let Some(elem_ty) = guard_elem_tag(ty) {
                if let Value::Enum { fields, .. } = value {
                    // The reference can sit inside one-field
                    // wrapper aggregates (the storage form a
                    // guard built by a converted body -- a
                    // `Ref::map` result -- leaves in its first
                    // field); the tag names the guard type
                    // exactly, so the wrappers are transparent,
                    // as they are for the storage forms above.
                    let mut first = fields.first();
                    for _ in 0..4 {
                        match first {
                            Some(Value::Enum { fields: inner, .. })
                                if inner.len() == 1 =>
                            {
                                first = inner.first();
                            }
                            _ => break,
                        }
                    }
                    if let Some(first) = first {
                        if matches!(
                            first,
                            Value::Ref(_)
                                | Value::FatRef { .. }
                                | Value::DynRef { .. }
                        ) {
                            let pointee = self.resolve_value(frames, first)?;
                            return self.format_value(
                                frames,
                                &pointee,
                                trait_,
                                &elem_ty,
                                options,
                                indent,
                            );
                        }
                    }
                }
            }
        }
        // A lock guard value (`MutexGuard`, `RwLockReadGuard`,
        // `RwLockWriteGuard`): std's `Display` and `Debug` impls
        // for the guard types format the guarded value, exactly
        // as the `RefCell` guards above, and no guest body
        // serves a std Self type, so the structural engine
        // forwards here. The pointee is reached through the
        // guard's first field: for `MutexGuard` /
        // `RwLockWriteGuard` it is the reference to the lock,
        // and the guarded value is the lock aggregate's data
        // field (field 2, the `UnsafeCell`'s field 0); for
        // `RwLockReadGuard` it is the `NonNull` data pointer
        // the guard's constructor took from the lock. For a
        // mapped guard it is likewise a `NonNull` pointer, but
        // one that denotes the mapped component itself. A
        // guard-shaped value whose first field does not resolve
        // this way keeps the existing behaviour.
        if matches!(trait_, FmtTrait::Display | FmtTrait::Debug) {
            if let Some((elem_ty, guard_kind)) = sync_guard_elem_tag(ty) {
                if let Value::Enum { fields, .. } = value {
                    let pointee = if guard_kind != SyncGuardKind::LockRef {
                        let mut first = fields.first();
                        for _ in 0..4 {
                            match first {
                                Some(Value::Enum {
                                    fields: inner, ..
                                }) if inner.len() == 1 => {
                                    first = inner.first();
                                }
                                _ => break,
                            }
                        }
                        match first {
                            Some(
                                first @ (Value::Ref(_)
                                | Value::FatRef { .. }
                                | Value::DynRef { .. }),
                            ) => {
                                // A read guard's pointer denotes
                                // the lock's `UnsafeCell` place,
                                // so the pointee is its field 0;
                                // a mapped guard's pointer denotes
                                // the mapped component itself.
                                let pointee = if guard_kind
                                    == SyncGuardKind::ReadPtr
                                {
                                    super::sync::project_pointee_cell(first.clone())
                                        .ok()
                                } else {
                                    Some(first.clone())
                                };
                                pointee.and_then(|p| {
                                    self.resolve_value(frames, &p).ok()
                                })
                            }
                            _ => None,
                        }
                    } else {
                        match fields.first() {
                            Some(Value::Ref(target)) => target
                                .projected(RefProj::Field(2))
                                .and_then(|t| t.projected(RefProj::Field(0)))
                                .and_then(|data| {
                                    self.resolve_value(frames, &Value::Ref(data)).ok()
                                }),
                            _ => None,
                        }
                    };
                    if let Some(pointee) = pointee {
                        return self.format_value(
                            frames,
                            &pointee,
                            trait_,
                            &elem_ty,
                            options,
                            indent,
                        );
                    }
                }
            }
        }
        // A std lock value (`Mutex` / `RwLock`): std's `Debug`
        // impls for the lock types are handwritten -- they
        // acquire the lock, show the guarded value (or the
        // `<locked>` placeholder while a guard is outstanding),
        // then the poison flag, and finish non-exhaustively --
        // and no guest body serves a std Self type, so the
        // structural engine composes the form here from the
        // lock aggregate's fields, as it does for the lock
        // guards above. A lock-shaped value whose state cannot
        // be read keeps the existing behaviour.
        if trait_ == FmtTrait::Debug {
            if let Some(text) =
                self.format_sync_lock(frames, value, ty, options, indent)?
            {
                return Ok(text);
            }
        }
        // `core::num::NonZero<T>`: every fmt impl core
        // gives the type (`Debug`, `Display` and the
        // radix / exponent traits) delegates to the
        // primitive's own impl over the stored scalar, so
        // the formatting is the scalar's, under the
        // argument's type tag. The value arrives as nested
        // one-field wrappers -- the `NonZeroInner`
        // associated type hides the primitive one layer
        // down, behind a type alias the debug-shape table
        // cannot resolve -- so the unwrap follows the
        // value's own one-field layers until a scalar
        // bottoms out (the reading the debugger's `NonZero`
        // rendering makes as well). The tag gate is what
        // licenses the delegation: an arbitrary one-field
        // ADT is not transparent to its payload's
        // formatting (see the storage-wrapper arms above),
        // but this type's impls are defined as the
        // delegation. A value that does not unwrap to a
        // scalar keeps the behaviour below.
        if !matches!(trait_, FmtTrait::Pointer | FmtTrait::DebugNoop)
            && tag_base_name(pointee_tag(ty)) == "NonZero"
        {
            if let Some(inner_ty) = first_generic_arg(pointee_tag(ty)) {
                let mut cur = value;
                let scalar = loop {
                    match cur {
                        Value::Enum { fields, .. } if fields.len() == 1 => {
                            cur = &fields[0];
                        }
                        Value::Tuple(fields) if fields.len() == 1 => {
                            cur = &fields[0];
                        }
                        Value::Int(_) | Value::Uint(_) => {
                            break Some(cur.clone());
                        }
                        _ => break None,
                    }
                };
                if let Some(scalar) = scalar {
                    return self.format_value(
                        frames,
                        &scalar,
                        trait_,
                        inner_ty,
                        options,
                        indent,
                    );
                }
            }
        }
        // A byte-form scalar: a scalar read out of byte-form
        // storage (a constant allocation's pointee, above all)
        // reaches formatting as the allocation's bytes from the
        // scalar on, while the tag still names the scalar type.
        // The scalar those bytes denote is the value formatting
        // runs on -- the reading the equality, ordering and
        // `SwitchInt` points make of the same form -- so
        // denominate it here, where every trait route below
        // sees the scalar (and every recursive subvalue, which
        // re-enters this function, inherits the reading). A
        // byte string whose tag names no scalar (a slice, an
        // array, a byte-string wrapper type) keeps the byte
        // form the routes below serve.
        let denominated;
        let value = match denominate_bytes_scalar(value, pointee_tag(ty)) {
            Some(scalar) => {
                denominated = scalar;
                &denominated
            }
            None => value,
        };
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

    /// `Debug` for a std lock value (`Mutex` / `RwLock`):
    /// std's handwritten form, composed from the lock
    /// aggregate, whose fields are the sys lock, the poison
    /// flag and the `UnsafeCell` holding the guarded value.
    /// Whether the acquisition std performs would succeed
    /// is read from the sys lock's state word: a `Mutex`
    /// is held while its futex word is nonzero, and an
    /// `RwLock` refuses a reader only while its state
    /// word's counter bits hold the write-locked sentinel
    /// (`(1 << 30) - 1`). The poison flag is the atomic
    /// bool the flag wraps; a flag with no storage (the
    /// `panic = "abort"` form, whose `get` is constantly
    /// `false`) reads as unset. Returns `None` when the
    /// value does not read as a lock, so the caller keeps
    /// the existing behaviour.
    fn format_sync_lock(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        ty: &str,
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<Option<String>> {
        let Some((name, elem_ty, rw_lock)) = sync_lock_tag(ty) else {
            return Ok(None);
        };
        let resolved;
        let value = match value {
            Value::Ref(_) => {
                resolved = self.resolve_value(frames, value)?;
                &resolved
            }
            other => other,
        };
        let Value::Enum { fields, .. } = value else {
            return Ok(None);
        };
        if fields.len() != 3 {
            return Ok(None);
        }
        let Some(state) = lock_state_word(&fields[0]) else {
            return Ok(None);
        };
        let locked = if rw_lock {
            state & 0x3fff_ffff == 0x3fff_ffff
        } else {
            state != 0
        };
        let poisoned = match scalar_word(&fields[1]) {
            Some(word) => word != 0,
            None => match &fields[1] {
                Value::Enum { fields: inner, .. } if inner.is_empty() => false,
                Value::Tuple(inner) if inner.is_empty() => false,
                _ => return Ok(None),
            },
        };
        // The guarded value is the `UnsafeCell`'s payload.
        let data = match &fields[2] {
            Value::Enum { fields: inner, .. } if inner.len() == 1 => inner[0].clone(),
            Value::Tuple(inner) if inner.len() == 1 => inner[0].clone(),
            other => other.clone(),
        };
        let data = if matches!(data, Value::Ref(_)) {
            self.resolve_value(frames, &data)?
        } else {
            data
        };
        let data_text = if locked {
            // The `Mutex` form shows the `&str` placeholder
            // (quoted, as a string's `Debug` is); the
            // `RwLock` form shows the bare formatted text
            // its `format_args!` placeholder renders as.
            if rw_lock {
                "<locked>".to_string()
            } else {
                "\"<locked>\"".to_string()
            }
        } else if options.alternate() {
            self.format_value(
                frames,
                &data,
                FmtTrait::Debug,
                &elem_ty,
                options,
                indent + 4,
            )?
        } else {
            self.format_value(
                frames,
                &data,
                FmtTrait::Debug,
                &elem_ty,
                &FmtOptions::new(),
                indent,
            )?
        };
        if options.alternate() {
            let pad = " ".repeat(indent + 4);
            return Ok(Some(format!(
                "{name} {{\n{pad}data: {data_text},\n{pad}poisoned: {poisoned},\n{pad}..\n{}}}",
                " ".repeat(indent)
            )));
        }
        Ok(Some(pad_width(
            &format!("{name} {{ data: {data_text}, poisoned: {poisoned}, .. }}"),
            options,
        )))
    }

    /// `Display` / `Debug` for an `io::Error` in the `Custom`
    /// representation: the payload the packed word names by
    /// registered synthetic address. `Display` is the payload's
    /// own `Display` (std's `Error` `Display` forwards to it);
    /// `Debug` is the `Custom { kind, error }` struct core's
    /// `Debug for Custom` writes, over the payload's `Debug`.
    /// Any other representation, and a word whose address the
    /// registry does not know, keeps the existing behaviour
    /// (`None`).
    fn format_io_error_custom(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        trait_: FmtTrait,
        ty: &str,
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<Option<String>> {
        // The `Custom { .. }` text is composed around the
        // payload's own formatted text, so the payload's body
        // buffers into the composition as usual: suppress the
        // stream arm for the duration, as `format_debug` does.
        let saved = self.fmt_stream_arm;
        self.fmt_stream_arm = false;
        let result =
            self.format_io_error_custom_composed(frames, value, trait_, ty, options, indent);
        self.fmt_stream_arm = saved;
        result
    }

    fn format_io_error_custom_composed(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        trait_: FmtTrait,
        ty: &str,
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<Option<String>> {
        let plain = pointee_tag(ty);
        if plain != "std::io::Error" && plain != "core::io::Error" {
            return Ok(None);
        }
        let Some(word) = io_error_tagged_word(value) else {
            return Ok(None);
        };
        if word & 0b11 != 0b01 {
            return Ok(None);
        }
        let Some(id) = self.address_owner(word.wrapping_sub(1)) else {
            return Ok(None);
        };
        // The payload: the `Custom` value in the allocation,
        // whose fields are (kind, error, error_drop, outer_drop).
        let custom = self.resolve_value(
            frames,
            &Value::Ref(RefTarget::Alloc {
                id,
                projection: Vec::new(),
            }),
        )?;
        let Value::Enum { fields, .. } = &custom else {
            return Ok(None);
        };
        if fields.len() < 2 {
            return Ok(None);
        }
        let payload = fields[1].clone();
        let payload_ty = match &payload {
            Value::DynRef { concrete, .. } => concrete.clone(),
            _ => String::new(),
        };
        Ok(match trait_ {
            FmtTrait::Display => {
                // A trait object's `Display` is its concrete
                // type's `Display` (the vtable entry), so format
                // the pointee under the concrete tag, shaping the
                // box base the way the trait-object arm of
                // `format_value` shapes it for `Debug`.
                let Value::DynRef { base, concrete, .. } = &payload else {
                    return Ok(None);
                };
                let pointee = self.resolve_value(frames, base)?;
                let pointee = match &pointee {
                    Value::Enum { fields, .. }
                        if fields.len() == 1
                            && matches!(
                                fields[0],
                                Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. }
                            ) =>
                    {
                        fields[0].clone()
                    }
                    _ => pointee,
                };
                let text =
                    self.format_value(frames, &pointee, trait_, concrete, options, indent)?;
                // `io::Error`'s `Display` forwards the formatter
                // to the payload's own `Display`, so the text
                // `format_value` produced under those options is
                // the final output: applying string padding
                // semantics on top would truncate or pad payloads
                // whose `Display` ignores precision / width, which
                // native leaves untouched.
                Some(text)
            }
            FmtTrait::Debug => {
                // The kind formats as an `ErrorKind` value (its
                // own `Debug` names the variant), not by index:
                // the variant index and the discriminant differ
                // for the kinds std added out of order. The
                // rendering goes through `debug_struct_text`, like
                // the other `io::Error` representations, so under
                // `{:#?}` the struct takes the multi-line layout;
                // the fields are formatted at the inner indent
                // then, so a nested payload's own multi-line
                // `Debug` lines up under its field.
                let field_indent = if options.alternate() {
                    indent + 4
                } else {
                    indent
                };
                let kind = self.format_value(
                    frames,
                    &fields[0],
                    trait_,
                    "std::io::ErrorKind",
                    options,
                    field_indent,
                )?;
                let error = self.format_value(
                    frames,
                    &payload,
                    trait_,
                    &payload_ty,
                    options,
                    field_indent,
                )?;
                Some(debug_struct_text(
                    "Custom",
                    &[("kind", kind), ("error", error)],
                    options,
                    indent,
                ))
            }
            _ => None,
        })
    }

    /// `Debug` text of a `CStr` / `CString` value: core's impl
    /// prints the string's bytes (without the terminating NUL)
    /// as a quoted literal, each byte escaped with
    /// `u8::escape_ascii`, written fragment by fragment with
    /// no padding. `None` when the type tag is not a C string
    /// type or the value's bytes cannot be read, in which case
    /// the existing behaviour stands.
    fn format_c_string(
        &self,
        frames: &[Frame],
        value: &Value,
        ty: &str,
    ) -> InterpResult<Option<String>> {
        // Only the standard library's own C string types take
        // this path: the tag must be one of their qualified
        // paths (the defining crate's or the `std` re-export).
        // A user-defined type that merely shares the base name
        // keeps its own `Debug` behaviour.
        let plain = pointee_tag(ty);
        if !matches!(
            plain,
            "std::ffi::CStr"
                | "core::ffi::CStr"
                | "alloc::ffi::CStr"
                | "std::ffi::CString"
                | "alloc::ffi::CString"
        ) {
            return Ok(None);
        }
        let Some(bytes) = self.c_string_value_bytes(frames, value)? else {
            return Ok(None);
        };
        let content = match bytes.last() {
            Some(0) => &bytes[..bytes.len() - 1],
            _ => &bytes[..],
        };
        let mut out = String::with_capacity(content.len() + 2);
        out.push('"');
        out.extend(content.iter().flat_map(|b| b.escape_ascii()).map(char::from));
        out.push('"');
        Ok(Some(out))
    }

    /// The bytes of a C string value, in whichever shape the
    /// value model stored them: a byte block (a `CStr`
    /// constant), a snapshot tuple of byte values (the buffer
    /// an `as_c_str` fat reference denotes), a fat reference
    /// over the buffer, or the owned aggregate (a `CString`'s
    /// wrapper chain down to its boxed slice). `None` for any
    /// other shape.
    fn c_string_value_bytes(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Option<Vec<u8>>> {
        match value {
            Value::Bytes(b) => Ok(Some(b.clone())),
            Value::Str(s) => Ok(Some(s.as_bytes().to_vec())),
            Value::Tuple(items) => Ok(byte_values(items)),
            Value::SparseVec { .. } | Value::FatRef { .. } => {
                let (elems, _) = self.slice_elements(frames, value)?;
                Ok(byte_values(&elems))
            }
            Value::Ref(_) | Value::DynRef { .. } => {
                let resolved = self.resolve_value(frames, value)?;
                if matches!(resolved, Value::Ref(_) | Value::DynRef { .. }) {
                    return Ok(None);
                }
                self.c_string_value_bytes(frames, &resolved)
            }
            Value::Enum { fields, .. } if fields.len() == 1 => {
                self.c_string_value_bytes(frames, &fields[0])
            }
            _ => Ok(None),
        }
    }

    /// Structural `Display` / `Debug` for `std::io::Error` and
    /// `ErrorKind` values, mirroring core's own implementations
    /// (`core::io::error`): an OS error displays as
    /// `{message} (os error {code})` and debugs as
    /// `Os { code, kind, message }`; a kind-only (simple) error
    /// displays as the kind's message and debugs as
    /// `Kind({kind})`; a kind displays as its message; a
    /// `SimpleMessage` error displays as its message and debugs
    /// as `Error { kind, message }`, decoded from the preserved
    /// constant block the guest stored through the erased slot.
    /// The `Custom` representation is resolved through the
    /// address registry by `format_io_error_custom`, which is
    /// consulted before this function, so it does not reach
    /// here. Under `{:#?}` the `Debug` forms take the
    /// multi-line layout of core's `debug_struct` /
    /// `debug_tuple` rendering, with fields at `indent + 4` and
    /// the closing delimiter at `indent`, like the derived form
    /// in `format_debug_adt`.
    fn format_io_error(
        &self,
        value: &Value,
        trait_: FmtTrait,
        ty: &str,
        options: &FmtOptions,
        indent: usize,
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
        if let Some((kind, message)) = io_error_simple_message(value) {
            return match trait_ {
                FmtTrait::Display => Some(pad_str(&message, options)),
                FmtTrait::Debug => {
                    let name = error_kind_name(kind)?;
                    Some(debug_struct_text(
                        "Error",
                        &[
                            ("kind", name.to_string()),
                            ("message", format!("{message:?}")),
                        ],
                        options,
                        indent,
                    ))
                }
                _ => None,
            };
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
                Some(debug_struct_text(
                    "Os",
                    &[
                        ("code", code.to_string()),
                        ("kind", kind.to_string()),
                        ("message", format!("\"{}\"", os_error_message(code))),
                    ],
                    options,
                    indent,
                ))
            }
            (FmtTrait::Display, 0b11) => {
                error_kind_message(payload).map(|m| pad_str(m, options))
            }
            (FmtTrait::Debug, 0b11) => error_kind_name(payload)
                .map(|name| debug_tuple_text("Kind", &[name.to_string()], options, indent)),
            _ => None,
        }
    }

    /// `Display` / `Debug` for the standard pointer wrappers
    /// (`Box`, `Rc`, `Arc`), which std defines as forwarding to
    /// the pointee: shape the pointee value the way the `dyn`
    /// formatting path shapes its base, then format it with the
    /// element type as the tag. A value that does not have the
    /// wrapper's stored shape is the pointee itself in the
    /// storage form a move leaves behind (the extracted
    /// snapshot the copy machinery stores in the slot in place
    /// of the wrapper aggregate): it formats under the element
    /// tag directly, the same licence `format_num_wrapper`
    /// takes for a scalar under a wrapper tag. `None` when the
    /// type is not one of the wrappers, in which case the
    /// existing behaviour stands.
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
                // the `dyn` path apply. An aggregate-built box
                // (`Box::from_raw`'s product) instead holds its
                // `Unique` at field 0, beside the allocator field:
                // its pointee reference is the box's data pointer.
                // A value in any other form is the pointee itself
                // in storage form (see the doc comment above).
                let Value::Enum { fields, .. } = value else {
                    return Ok(Some(self.format_value(
                        frames, value, trait_, elem, options, indent,
                    )?));
                };
                if fields.len() == 1 {
                    fields[0].clone()
                } else if let Some(fat) = box_fat_pointer(value) {
                    // A fat box: the box's own pointer is the
                    // fat one at the end of the field-0 chain,
                    // whose metadata the thin extraction below
                    // would drop.
                    fat
                } else {
                    self.dyn_data_pointer(value)
                }
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
                    // The pointee itself in storage form, as in
                    // the `Box` arm above.
                    return Ok(Some(self.format_value(
                        frames, value, trait_, elem, options, indent,
                    )?));
                }
            }
        };
        Ok(Some(self.format_value(
            frames, &pointee, trait_, elem, options, indent,
        )?))
    }

    /// The fmt traits `core` implements for the numeric
    /// wrappers `std::num::Wrapping<T>` and
    /// `std::num::Saturating<T>` (`Debug`, `Display`, `Binary`,
    /// `Octal`, `LowerHex` and `UpperHex`) all forward to the
    /// inner value's own impl of the same trait: the wrapper's
    /// name never appears and the formatting options pass
    /// through unchanged, so `Wrapping(5)` debugs as `5` and a
    /// pretty-printed `Saturating<Vec<_>>` is the inner list's
    /// pretty form at the same indent. `None` when the type is
    /// not one of these wrappers, the trait is one they do not
    /// implement (neither has a `LowerExp` / `UpperExp` /
    /// `Pointer` impl), or the value does not have the
    /// wrapper's one-field shape, in which case the existing
    /// behaviour stands.
    fn format_num_wrapper(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        trait_: FmtTrait,
        ty: &str,
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<Option<String>> {
        if !matches!(
            trait_,
            FmtTrait::Display
                | FmtTrait::Debug
                | FmtTrait::Binary
                | FmtTrait::Octal
                | FmtTrait::LowerHex
                | FmtTrait::UpperHex
        ) {
            return Ok(None);
        }
        let plain = pointee_tag(ty);
        let head = plain.split('<').next().unwrap_or(plain).trim();
        if head != "core::num::Wrapping"
            && head != "std::num::Wrapping"
            && head != "core::num::Saturating"
            && head != "std::num::Saturating"
        {
            return Ok(None);
        }
        let Some(inner_ty) = first_generic_arg(plain) else {
            return Ok(None);
        };
        let inner_ty = inner_ty.to_string();
        let inner = match value {
            Value::Enum { fields, .. } if fields.len() == 1 => fields[0].clone(),
            Value::Tuple(items) if items.len() == 1 => items[0].clone(),
            // A scalar under a wrapper tag is the payload
            // itself in storage form: reformatting it under the
            // inner type's tag gives it its own width.
            Value::Int(_) | Value::Uint(_) | Value::Float(_) | Value::Bool(_) => value.clone(),
            _ => return Ok(None),
        };
        Ok(Some(self.format_value(
            frames, &inner, trait_, &inner_ty, options, indent,
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

    /// Formatting for the std byte-string wrapper types whose
    /// fmt impls std defines directly over the held bytes. A
    /// transparent wrapper around string data — `PathBuf` around
    /// `OsString` around the byte buffer, `Path` / `OsStr` as the
    /// unsized forms — debugs as the content itself: quoted, with
    /// non-UTF-8 bytes escaped, exactly the rendering the host's
    /// `OsStr` `Debug` produces (std's impls write the content
    /// directly and do not consult width or precision). The
    /// `Display<'_>` helpers that `Path::display` and
    /// `OsStr::display` return display as the lossy content
    /// and debug as the content's own `Debug`. `None` when
    /// the type is not one of these or the value holds no
    /// extractable bytes, in which case the existing behaviour
    /// stands.
    fn format_string_content(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        trait_: FmtTrait,
        ty: &str,
        options: &FmtOptions,
    ) -> InterpResult<Option<String>> {
        let plain = pointee_tag(ty);
        let head = plain.split('<').next().unwrap_or(plain).trim();
        match head {
            "std::path::PathBuf"
            | "std::path::Path"
            | "std::ffi::OsString"
            | "std::ffi::OsStr" => {
                if trait_ != FmtTrait::Debug {
                    return Ok(None);
                }
                let Some(bytes) = self.string_content_bytes(frames, value) else {
                    return Ok(None);
                };
                let Some(os) = crate::platform::os_str_from_bytes(&bytes)
                else {
                    return Ok(None);
                };
                Ok(Some(format!("{:?}", os)))
            }
            "std::path::Display" | "std::ffi::os_str::Display" => {
                // The `Display` helpers `Path::display` and
                // `OsStr::display` return: `Display` is the
                // lossy content of the bytes the inner value
                // holds, and `Debug` forwards to the inner
                // value's own `Debug` -- the byte content's
                // `OsStr` form, which consults no options
                // either.
                let Value::Enum { fields, .. } = value else {
                    return Ok(None);
                };
                let [field] = fields.as_slice() else {
                    return Ok(None);
                };
                let field = field.clone();
                let Some(bytes) = self.string_content_bytes(frames, &field) else {
                    return Ok(None);
                };
                match trait_ {
                    FmtTrait::Display => Ok(Some(pad_str(
                        &String::from_utf8_lossy(&bytes),
                        options,
                    ))),
                    FmtTrait::Debug => {
                        let Some(os) =
                            crate::platform::os_str_from_bytes(&bytes)
                        else {
                            return Ok(None);
                        };
                        Ok(Some(format!("{:?}", os)))
                    }
                    _ => Ok(None),
                }
            }
            _ => Ok(None),
        }
    }

    /// The bytes a path / OS-string value holds, through
    /// whichever representation the value model gave it: an
    /// owned wrapper aggregate, a string value, or a fat
    /// reference over byte storage.
    fn string_content_bytes(&self, frames: &[Frame], value: &Value) -> Option<Vec<u8>> {
        if let Ok(bytes) = self.os_str_bytes(frames, value) {
            return Some(bytes);
        }
        if let Ok(bytes) = self.path_bytes(frames, value) {
            return Some(bytes);
        }
        // A reference to the data may itself sit inside
        // one-field wrappers (the shape the field of a
        // `std::path::Display` takes): unwrap those layers,
        // resolving references as they surface, and retry the
        // extraction at each step.
        let mut current = value.clone();
        for _ in 0..8 {
            match current {
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    current = fields.into_iter().next().expect("length checked");
                }
                Value::Ref(_) | Value::FatRef { .. } => {
                    current = self.resolve_value(frames, &current).ok()?;
                }
                _ => return None,
            }
            if let Ok(bytes) = self.os_str_bytes(frames, &current) {
                return Some(bytes);
            }
            if let Ok(bytes) = self.path_bytes(frames, &current) {
                return Some(bytes);
            }
        }
        None
    }

    /// `Display` for an escape iterator value (see
    /// [`escape_display_tag`]): the concatenation of the
    /// chars the value's own modelled step still yields, in
    /// order — the text core's `Display` definitions for
    /// these types produce. `None` for any other type.
    /// The formatting options do not apply: core's impls
    /// write the chars one by one and never pad or
    /// truncate, so a width, fill or precision in the
    /// format spec leaves the text untouched (unlike the
    /// primitive impls, which pad through the formatter).
    fn format_escape_display(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        ty: &str,
        _options: &FmtOptions,
    ) -> InterpResult<Option<String>> {
        let plain = pointee_tag(ty);
        if !escape_display_tag(plain) {
            return Ok(None);
        }
        let (items, _) = self.drain_iterator(frames, plain, value.clone())?;
        let mut text = String::new();
        for item in items {
            match item {
                Value::Uint(v) => text.push(as_char(v)?),
                other => {
                    return Err(InterpError::type_mismatch(
                        format!("Display of {plain}"),
                        "char items from the modelled step",
                        other.kind_name(),
                    ))
                }
            }
        }
        Ok(Some(text))
    }

    /// The `Display` rendering of a `core::panic::PanicInfo` /
    /// `core::panic::PanicMessage` value, or `None` when `ty`
    /// names neither. The info's fields are
    /// `(message, location, can_unwind, force_no_backtrace)` and
    /// the message's single field is its `fmt::Arguments`; both
    /// are stored behind references in the aggregate.
    fn format_panic_display(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        ty: &str,
    ) -> InterpResult<Option<String>> {
        let tag = pointee_tag(ty);
        let head = tag.split('<').next().unwrap_or(tag).trim();
        let fields = match value {
            Value::Enum { fields, .. } | Value::Tuple(fields) => fields,
            _ => return Ok(None),
        };
        if head == "core::panic::PanicMessage" {
            let Some(message) = fields.first() else {
                return Ok(None);
            };
            let message = self.resolve_value(frames, message)?;
            return match message {
                Value::FmtArguments(args) => {
                    Ok(Some(self.format_arguments(frames, &args, 0)?))
                }
                _ => Ok(None),
            };
        }
        if head != "core::panic::PanicInfo" || fields.len() != 4 {
            return Ok(None);
        }
        let message = self.resolve_value(frames, &fields[0])?;
        let Value::FmtArguments(args) = message else {
            return Ok(None);
        };
        let message = self.format_arguments(frames, &args, 0)?;
        let location = self.resolve_value(frames, &fields[1])?;
        let loc_fields = match &location {
            Value::Enum { fields, .. } | Value::Tuple(fields) => fields.clone(),
            _ => return Ok(None),
        };
        if loc_fields.len() != 3 {
            return Ok(None);
        }
        let file = self.resolve_value(frames, &loc_fields[0])?;
        let line = self.resolve_value(frames, &loc_fields[1])?;
        let col = self.resolve_value(frames, &loc_fields[2])?;
        let (Value::Str(file), Value::Uint(line), Value::Uint(col)) =
            (file, line, col)
        else {
            return Ok(None);
        };
        Ok(Some(format!(
            "panicked at {file}:{line}:{col}:\n{message}"
        )))
    }

    fn format_display(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        ty: &str,
        options: &FmtOptions,
    ) -> InterpResult<String> {
        // `core::panic::PanicInfo` and `core::panic::PanicMessage`
        // have fixed, small `Display` impls in core (the info
        // prints `panicked at {location}:\n{message}`, the message
        // formats its `fmt::Arguments`) and no Stable MIR body, so
        // the structural engine renders them from the value's
        // fields like the other core types it models. A panic
        // handler in a `no_std` program reaches them through
        // `PanicInfo::to_string` / its message accessor.
        if let Some(text) = self.format_panic_display(frames, value, ty)? {
            return Ok(pad_str(&text, options));
        }
        match value {
            Value::Int(v) => Ok(format_int_radix(*v, None, 10, "", options)),
            Value::Uint(v) => {
                if pointee_tag(ty) == "char" {
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
            Value::Enum { .. } => {
                Err(unsupported(
                    "Display formatting for an ADT value (its impl lives in unconverted code)",
                ))
            }
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
        // Structural composition: the text is composed bottom-up
        // from the subvalues' own formatted texts, so a guest
        // body dispatched for a subvalue buffers into the
        // composition as usual -- the stream arm (see
        // [`FmtCapture::stream`]) is suppressed for the duration,
        // so only a `Termination` report rendering's own payload
        // body can take it.
        let saved = self.fmt_stream_arm;
        self.fmt_stream_arm = false;
        let result = self.format_debug_composed(frames, value, ty, options, indent);
        self.fmt_stream_arm = saved;
        result
    }

    fn format_debug_composed(
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
                if pointee_tag(ty) == "char" {
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
            // A byte block: the storage form of a byte slice or
            // array value (and of a byte-string constant). The
            // tag names the element type; only its signedness
            // matters (`i8` / `c_char` elements print signed).
            Value::Bytes(b) if pointee_tag(ty).starts_with('[') => {
                let elem = slice_elem_tag(pointee_tag(ty));
                let items = byte_seq_items(b, elem);
                let tys: Vec<&str> = items.iter().map(|_| elem).collect();
                self.format_debug_seq_items(frames, &items, options, indent, '[', ']', &tys)
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
            Value::SparseVec { elems, len, .. } => {
                // A vector value in sparse form (see
                // `Value::SparseVec`) formats as the list of its
                // declared elements, exactly as the plain
                // aggregate form does through `format_debug_adt`:
                // the stored elements up to the length, with an
                // unmaterialised tail padded as the uninitialised
                // slots it denotes (bounded by the same
                // materialisation limit as `Vec::set_len`; past
                // it the list cannot be produced and stays an
                // explicit error). The spare-capacity tail past
                // the length is not part of the list.
                let declared = usize::try_from(*len).unwrap_or(usize::MAX);
                if declared > elems.len() && declared > MAX_VEC_SET_LEN_ELEMENTS {
                    return Err(unsupported(
                        "Debug formatting of a vector whose elements are not materialised"
                            .to_string(),
                    ));
                }
                let mut fields: Vec<Value> =
                    elems.iter().take(declared).cloned().collect();
                fields.resize(declared, Value::Uninit);
                self.format_debug_adt(frames, 0, &fields, ty, options, indent)
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

    /// The elements of a `Vec` value in its real aggregate
    /// form ([RawVec, len]), read back through the buffer
    /// pointer with the element type the container's tag
    /// names: the typed read rebuilds each element from the
    /// buffer's storage, where a bulk snapshot of the storage
    /// hands back an aggregate element's fields flattened
    /// into the element list instead. `None` when the value
    /// is not in the real form, when its declared length is
    /// past the materialisation limit, or when its buffer
    /// cannot be read back.
    fn real_vec_elements(
        &self,
        frames: &[Frame],
        value: &Value,
        elem_tag: &str,
    ) -> Option<Vec<Value>> {
        let (_, len) = vec_real_aggregate_parts(value)?;
        if len > MAX_VEC_SET_LEN_ELEMENTS {
            return None;
        }
        // An empty vector's elements are no elements, whatever
        // its buffer pointer denotes (a dangling address-only
        // pointer an allocator model hands out has no pointee
        // to read).
        if len == 0 {
            return Some(Vec::new());
        }
        let ptr = Self::real_vec_buffer_ptr(value)?;
        let target = self.deref_value(frames, &ptr).ok()?;
        let elem_ty = ty_from_tag(elem_tag);
        let mut elements = Vec::with_capacity(len);
        for i in 0..len {
            elements.push(
                self.read_seq_at_with_elem(frames, &target, i, elem_ty.as_ref())
                    .ok()?,
            );
        }
        Some(elements)
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
    /// Three std shapes the value model does not store in declared
    /// shape are recognised by tag first: a `Vec` value's fields
    /// are its elements (it prints as the list it is), a `VecDeque`
    /// value is the ring-buffer aggregate whose logical element
    /// sequence prints as the same list, and a `Box` value's one
    /// field is the pointee reference (its `Debug` forwards to the
    /// pointee). An ADT with no recorded shape stays an explicit
    /// unsupported error.
    /// Debug formatting of a `std::process::Command` value.
    ///
    /// The `Debug` impls of the public `Command` and of the
    /// sys `Command` it forwards to are handwritten in std:
    /// the plain form renders a shell-like invocation (an
    /// optional `cd ... &&`, the environment changes as an
    /// `env` prefix, then the quoted program and arguments),
    /// and the alternate form renders a `Command` debug
    /// struct over the fields that are set. Neither is the
    /// derived form of the sys struct's storage, and the
    /// value model does not hold that storage at all: a
    /// command value is the handle aggregate the process
    /// model builds (see `process::command_value`), whose
    /// state lives in the command table. The text is
    /// therefore composed here from the table snapshot,
    /// following std's own rendering rules -- the
    /// environment changes fold into key order, the way
    /// std's `BTreeMap` storage iterates them, and a
    /// removal recorded after `env_clear` deletes the entry
    /// instead of naming a removal (the model's own log is
    /// emptied by the clear, so only the post-clear calls
    /// fold here). `None` when the type is not a command
    /// type or the value carries no live command handle, in
    /// which case the existing behaviour stands.
    fn format_debug_command(
        &mut self,
        variant: u32,
        fields: &[Value],
        tag: &str,
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<Option<String>> {
        let head = tag.split('<').next().unwrap_or(tag).trim();
        if head != "std::process::Command"
            && head != "std::sys::process::unix::common::Command"
        {
            return Ok(None);
        }
        let aggregate = Value::Enum {
            variant,
            fields: fields.to_vec(),
        };
        let Some(id) = super::process::command_handle_id(&aggregate) else {
            return Ok(None);
        };
        let Some(state) = self.command_snapshot(id) else {
            return Ok(None);
        };
        let mut envs: std::collections::BTreeMap<Vec<u8>, Option<Vec<u8>>> =
            std::collections::BTreeMap::new();
        for (key, value) in &state.envs {
            match value {
                Some(v) => {
                    envs.insert(key.clone(), Some(v.clone()));
                }
                None => {
                    if state.env_clear {
                        envs.remove(key);
                    } else {
                        envs.insert(key.clone(), None);
                    }
                }
            }
        }
        let os_debug = |bytes: &[u8]| -> Option<String> {
            crate::platform::os_str_from_bytes(bytes).map(|os| format!("{os:?}"))
        };
        let argv0: &[u8] = state.argv0.as_deref().unwrap_or(&state.program);
        if !options.alternate() {
            let mut out = String::new();
            if let Some(cwd) = &state.cwd {
                let Some(cwd) = os_debug(cwd) else {
                    return Ok(None);
                };
                out += &format!("cd {cwd} && ");
            }
            if state.env_clear {
                out += "env -i ";
            } else {
                let mut any_removed = false;
                for (key, value) in &envs {
                    if value.is_none() {
                        if !any_removed {
                            out += "env ";
                            any_removed = true;
                        }
                        out += &format!("-u {} ", String::from_utf8_lossy(key));
                    }
                }
            }
            for (key, value) in &envs {
                if let Some(value) = value {
                    let Some(value) = os_debug(value) else {
                        return Ok(None);
                    };
                    out += &format!("{}={value} ", String::from_utf8_lossy(key));
                }
            }
            if state.program != argv0 {
                let Some(program) = os_debug(&state.program) else {
                    return Ok(None);
                };
                out += &format!("[{program}] ");
            }
            let Some(text) = os_debug(argv0) else {
                return Ok(None);
            };
            out += &text;
            for arg in &state.args {
                let Some(arg) = os_debug(arg) else {
                    return Ok(None);
                };
                out += &format!(" {arg}");
            }
            return Ok(Some(pad_width(&out, options)));
        }
        // The alternate form: std's `debug_struct` over the
        // fields that are set, in std's field order. A
        // `Stdio` field the model recorded but cannot decode
        // keeps the existing behaviour rather than a guessed
        // rendering.
        let pretty_list = |items: &[String], at: usize| -> String {
            if items.is_empty() {
                return "[]".to_string();
            }
            let inner = at + 4;
            let mut out = String::from("[\n");
            for item in items {
                out += &" ".repeat(inner);
                out += item;
                out += ",\n";
            }
            out += &" ".repeat(at);
            out += "]";
            out
        };
        let pretty_some = |text: String, at: usize| -> String {
            let inner = at + 4;
            format!("Some(\n{}{},\n{})", " ".repeat(inner), text, " ".repeat(at))
        };
        let inner = indent + 4;
        let mut out = String::from("Command {\n");
        let mut field = |out: &mut String, name: &str, text: &str| {
            out.push_str(&" ".repeat(inner));
            out.push_str(name);
            out.push_str(": ");
            out.push_str(text);
            out.push_str(",\n");
        };
        let Some(program) = os_debug(&state.program) else {
            return Ok(None);
        };
        field(&mut out, "program", &program);
        let mut arg_texts = Vec::with_capacity(state.args.len() + 1);
        let Some(text) = os_debug(argv0) else {
            return Ok(None);
        };
        arg_texts.push(text);
        for arg in &state.args {
            let Some(arg) = os_debug(arg) else {
                return Ok(None);
            };
            arg_texts.push(arg);
        }
        let args_text = pretty_list(&arg_texts, inner);
        field(&mut out, "args", &args_text);
        if state.env_clear || !envs.is_empty() {
            let vars_inner = inner + 8;
            let mut vars = String::new();
            if envs.is_empty() {
                vars += "{}";
            } else {
                vars += "{\n";
                for (key, value) in &envs {
                    let Some(key) = os_debug(key) else {
                        return Ok(None);
                    };
                    vars += &" ".repeat(vars_inner);
                    vars += &key;
                    vars += ": ";
                    match value {
                        Some(v) => {
                            let Some(v) = os_debug(v) else {
                                return Ok(None);
                            };
                            vars += &pretty_some(v, vars_inner);
                        }
                        None => vars += "None",
                    }
                    vars += ",\n";
                }
                vars += &" ".repeat(inner + 4);
                vars += "}";
            }
            let env_text = format!(
                "CommandEnv {{\n{}clear: {},\n{}vars: {},\n{}}}",
                " ".repeat(inner + 4),
                state.env_clear,
                " ".repeat(inner + 4),
                vars,
                " ".repeat(inner)
            );
            field(&mut out, "env", &env_text);
        }
        if let Some(cwd) = &state.cwd {
            let Some(cwd) = os_debug(cwd) else {
                return Ok(None);
            };
            field(&mut out, "cwd", &pretty_some(cwd, inner));
        }
        if let Some(uid) = state.uid {
            field(&mut out, "uid", &pretty_some(uid.to_string(), inner));
        }
        if let Some(gid) = state.gid {
            field(&mut out, "gid", &pretty_some(gid.to_string(), inner));
        }
        if let Some(groups) = &state.groups {
            let items: Vec<String> =
                groups.iter().map(|g| g.to_string()).collect();
            let text = pretty_list(&items, inner + 4);
            field(&mut out, "groups", &pretty_some(text, inner));
        }
        for (name, config) in [
            ("stdin", state.stdin_stdio),
            ("stdout", state.stdout_stdio),
            ("stderr", state.stderr_stdio),
        ] {
            if let Some(config) = config {
                // The alternate form shows the sys `Stdio`
                // enum's derived `Debug`: the three routings
                // with no descriptor to name. A descriptor
                // form carries a host fd number the model
                // does not hold, so it keeps the existing
                // behaviour rather than a guessed rendering.
                let text = match config {
                    crate::interp::StdioConfig::Inherit => "Inherit",
                    crate::interp::StdioConfig::Null => "Null",
                    crate::interp::StdioConfig::Piped => "MakePipe",
                    crate::interp::StdioConfig::File(_)
                    | crate::interp::StdioConfig::Unmodelled => {
                        return Ok(None);
                    }
                };
                field(&mut out, name, &pretty_some(text.to_string(), inner));
            }
        }
        field(&mut out, "create_pidfd", "false");
        out += &" ".repeat(indent);
        out += "}";
        Ok(Some(out))
    }

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
        // The std map/set collections' `Debug` impls are
        // handwritten: a map prints as the `debug_map` of its
        // entries (`{k: v, ...}`), a set as the `debug_set` of
        // its elements (`{a, b, ...}`) -- neither is the derived
        // form of the storage aggregate the value model holds
        // (a hash collection's raw table, a B-tree's root and
        // length), and no guest body serves a std Self type, so
        // the structural engine composes the form from the
        // entries the storage walks read back (see
        // `format_debug_collection`). A value whose entries
        // cannot be read back keeps the recorded-shape
        // rendering below.
        if let Some(text) =
            self.format_debug_collection(frames, variant, fields, tag, options, indent)?
        {
            return Ok(text);
        }
        if let Some(text) =
            self.format_debug_command(variant, fields, tag, options, indent)?
        {
            return Ok(text);
        }
        match tag_base_name(tag) {
            "Vec" => {
                let elem = first_generic_arg(tag).unwrap_or("");
                // A vector in its real aggregate form ([RawVec,
                // len]) -- the shape inlined construction builds
                // at higher opt levels -- holds its elements
                // behind the buffer pointer, not as the fields:
                // read them back through the pointer and format
                // those. A buffer the model cannot read back
                // keeps the field-list rendering below.
                let aggregate = Value::Enum {
                    variant,
                    fields: fields.to_vec(),
                };
                if let Some(elements) =
                    self.real_vec_elements(frames, &aggregate, elem)
                {
                    let tys: Vec<&str> =
                        elements.iter().map(|_| elem).collect();
                    return self.format_debug_seq_items(
                        frames, &elements, options, indent, '[', ']', &tys,
                    );
                }
                let tys: Vec<&str> = fields.iter().map(|_| elem).collect();
                return self.format_debug_seq_items(
                    frames, fields, options, indent, '[', ']', &tys,
                );
            }
            "VecDeque" => {
                // A `VecDeque` value is the real aggregate
                // (wrapped head, length, ring buffer), not an
                // element list; its `Debug` is the `debug_list`
                // of its logical element sequence, so render that
                // sequence through the same sequence formatter a
                // `Vec` uses. A value whose buffer slots cannot
                // be read back is not a well-formed deque state
                // and keeps the recorded-shape rendering below.
                let aggregate = Value::Enum {
                    variant,
                    fields: fields.to_vec(),
                };
                if let Some(elements) = self.vec_deque_elements(&aggregate)? {
                    let elem = first_generic_arg(tag).unwrap_or("");
                    let tys: Vec<&str> = elements.iter().map(|_| elem).collect();
                    return self.format_debug_seq_items(
                        frames, &elements, options, indent, '[', ']', &tys,
                    );
                }
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
            "core::asserting::Capture" => {
                // The handwritten `Debug` of the `assert!`
                // capture wrapper is a fixed function of the
                // marker type argument: the `TryCaptureWithoutDebug`
                // marker always prints `N/A`; the
                // `TryCaptureWithDebug` marker prints the
                // captured element's own `Debug` (the `elem`
                // field's `Some` payload), or `N/A` when no
                // element was captured.
                let marker = second_generic_arg(tag).unwrap_or("");
                let marker_head = marker.split('<').next().unwrap_or("").trim();
                if marker_head == "core::asserting::TryCaptureWithoutDebug" {
                    return Ok(pad_width("N/A", options));
                }
                if marker_head == "core::asserting::TryCaptureWithDebug" {
                    match fields.first() {
                        Some(Value::Enum {
                            variant: 1,
                            fields: payload,
                        }) if payload.len() == 1 => {
                            let elem = first_generic_arg(tag).unwrap_or("");
                            let text = self.format_value(
                                frames,
                                &payload[0],
                                FmtTrait::Debug,
                                elem,
                                options,
                                indent,
                            )?;
                            return Ok(text);
                        }
                        Some(Value::Enum { variant: 0, .. }) => {
                            return Ok(pad_width("N/A", options));
                        }
                        _ => {}
                    }
                }
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

    /// `Debug` for a std map/set collection value, composed
    /// from the entries its storage walk reads back: a map's
    /// entries as `{k: v, ...}`, a set's elements as
    /// `{a, b, ...}` (the forms std's handwritten impls produce
    /// through `debug_map` / `debug_set`). The hash collections'
    /// entries come from the raw-table walk, the B-tree
    /// collections' from the node walk the iteration builtins
    /// share. `None` when the tag names no such collection, or
    /// the value's entries cannot be read back, so the caller
    /// keeps the recorded-shape rendering.
    fn format_debug_collection(
        &mut self,
        frames: &mut Vec<Frame>,
        variant: u32,
        fields: &[Value],
        tag: &str,
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<Option<String>> {
        let head = tag.split('<').next().unwrap_or(tag).trim();
        let aggregate = Value::Enum {
            variant,
            fields: fields.to_vec(),
        };
        match head {
            "std::collections::HashMap" | "alloc::collections::HashMap" => {
                let Some(entries) = self.hash_map_entries(&aggregate, false) else {
                    return Ok(None);
                };
                let key_ty = first_generic_arg(tag).unwrap_or("");
                let val_ty = second_generic_arg(tag).unwrap_or("");
                Ok(Some(self.format_debug_map_entries(
                    frames, &entries, key_ty, val_ty, options, indent,
                )?))
            }
            "std::collections::HashSet" | "alloc::collections::HashSet" => {
                let Some(entries) = self.hash_map_entries(&aggregate, true) else {
                    return Ok(None);
                };
                let elem = first_generic_arg(tag).unwrap_or("");
                let keys: Vec<Value> =
                    entries.into_iter().map(|(key, _)| key).collect();
                let tys: Vec<&str> = keys.iter().map(|_| elem).collect();
                Ok(Some(self.format_debug_seq_items(
                    frames, &keys, options, indent, '{', '}', &tys,
                )?))
            }
            "std::collections::BTreeMap" | "alloc::collections::BTreeMap" => {
                let Ok(entries) = self.btree_map_pairs(frames, &aggregate) else {
                    return Ok(None);
                };
                let key_ty = first_generic_arg(tag).unwrap_or("");
                let val_ty = second_generic_arg(tag).unwrap_or("");
                Ok(Some(self.format_debug_map_entries(
                    frames, &entries, key_ty, val_ty, options, indent,
                )?))
            }
            "std::collections::BTreeSet" | "alloc::collections::BTreeSet" => {
                let Ok(elements) = self.btree_set_elements(frames, &aggregate) else {
                    return Ok(None);
                };
                let elem = first_generic_arg(tag).unwrap_or("");
                let tys: Vec<&str> = elements.iter().map(|_| elem).collect();
                Ok(Some(self.format_debug_seq_items(
                    frames, &elements, options, indent, '{', '}', &tys,
                )?))
            }
            _ => Ok(None),
        }
    }

    /// Debug formatting of a map's entries: `{k: v, ...}`, or
    /// the multi-line form under `{:#?}` -- the `DebugMap`
    /// builder's output the std map `Debug` impls produce.
    fn format_debug_map_entries(
        &mut self,
        frames: &mut Vec<Frame>,
        entries: &[(Value, Value)],
        key_ty: &str,
        val_ty: &str,
        options: &FmtOptions,
        indent: usize,
    ) -> InterpResult<String> {
        if options.alternate() {
            if entries.is_empty() {
                return Ok("{}".to_string());
            }
            let inner = indent + 4;
            let mut out = String::new();
            out.push('{');
            out.push('\n');
            for (key, value) in entries {
                out.push_str(&" ".repeat(inner));
                let key_text =
                    self.format_value(frames, key, FmtTrait::Debug, key_ty, options, inner)?;
                let value_text =
                    self.format_value(frames, value, FmtTrait::Debug, val_ty, options, inner)?;
                out.push_str(&format!("{key_text}: {value_text},\n"));
            }
            out.push_str(&" ".repeat(indent));
            out.push('}');
            return Ok(out);
        }
        let mut parts = Vec::with_capacity(entries.len());
        for (key, value) in entries {
            let key_text = self.format_value(
                frames,
                key,
                FmtTrait::Debug,
                key_ty,
                &FmtOptions::new(),
                indent,
            )?;
            let value_text = self.format_value(
                frames,
                value,
                FmtTrait::Debug,
                val_ty,
                &FmtOptions::new(),
                indent,
            )?;
            parts.push(format!("{key_text}: {value_text}"));
        }
        Ok(pad_width(&format!("{{{}}}", parts.join(", ")), options))
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
        // A sequence composition reached straight from a slice
        // value: suppress the stream arm for the composition, as
        // `format_debug` does.
        let saved = self.fmt_stream_arm;
        self.fmt_stream_arm = false;
        let result =
            self.format_debug_seq_composed(frames, value, options, indent, open, close, elem_ty);
        self.fmt_stream_arm = saved;
        result
    }

    fn format_debug_seq_composed(
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
                // A regrouped fat pointer (see `format_value`):
                // the byte view's elements under the element
                // type, when it judges the view.
                if let Some(elem) = ty_from_tag(elem_ty) {
                    if let Some(items) =
                        self.foreign_slice_elements(frames, value, &elem)
                    {
                        let tys: Vec<&str> =
                            items.iter().map(|_| elem_ty).collect();
                        return self.format_debug_seq_items(
                            frames, &items, options, indent, open, close, &tys,
                        );
                    }
                }
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
                let items = byte_seq_items(b, elem_ty);
                let tys: Vec<&str> = items.iter().map(|_| elem_ty).collect();
                return self.format_debug_seq_items(frames, &items, options, indent, open, close, &tys);
            }
            // A string value read as the byte sequence it
            // denotes (a `str` value a transmute or cast
            // re-denominated as a byte slice): its elements
            // are its UTF-8 bytes, exactly as for the byte
            // form above.
            Value::Str(text) => {
                let items = byte_seq_items(text.as_bytes(), elem_ty);
                let tys: Vec<&str> = items.iter().map(|_| elem_ty).collect();
                return self.format_debug_seq_items(frames, &items, options, indent, open, close, &tys);
            }
            // A sequence container's own aggregate form: the
            // fat pointer a `Deref` produces (a `Vec`'s, an owned
            // `Cow`'s) denotes the container itself, so resolving
            // it yields the container value rather than an
            // element snapshot. The slice's elements are the
            // container's, read by the same element readers the
            // slice machinery uses: a real aggregate form's
            // elements sit behind its buffer pointer (see
            // `real_vec_elements`); the plain and sparse forms
            // are the element reader's own shapes.
            Value::Enum { .. } | Value::SparseVec { .. } => {
                if let Some(items) = self.real_vec_elements(frames, value, elem_ty) {
                    let tys: Vec<&str> = items.iter().map(|_| elem_ty).collect();
                    return self.format_debug_seq_items(
                        frames, &items, options, indent, open, close, &tys,
                    );
                }
                let (items, _) = self.slice_elements(frames, value)?;
                let tys: Vec<&str> = items.iter().map(|_| elem_ty).collect();
                return self.format_debug_seq_items(
                    frames, &items, options, indent, open, close, &tys,
                );
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


/// The element values of a byte block formatted as a byte
/// sequence with element tag `elem`: bytes read as signed
/// values when the element type is a signed byte type (`i8`,
/// `c_char`), as unsigned values otherwise.
fn byte_seq_items(bytes: &[u8], elem: &str) -> Vec<Value> {
    if elem == "i8" || tag_base_name(elem) == "c_char" {
        bytes.iter().map(|b| Value::Int(*b as i8 as i128)).collect()
    } else {
        bytes.iter().map(|b| Value::Uint(*b as u128)).collect()
    }
}

/// The scalar a byte-form value denotes under a scalar type
/// tag, if the tag names one: the string's leading bytes read
/// at the tag's width -- unsigned for the unsigned types and
/// for `char` (whose renderers read the code point off the
/// unsigned form), sign-extended for the signed types, a
/// truth value for `bool`. The string is the scalar's own
/// storage or an allocation snapshot that starts at the
/// scalar, so the scalar's bytes lead it in both shapes. A
/// string shorter than the width denotes no scalar and
/// answers `None`, as does a tag naming no scalar type.
fn denominate_bytes_scalar(value: &Value, tag: &str) -> Option<Value> {
    let Value::Bytes(_) = value else {
        return None;
    };
    let (width, signed) = match tag {
        "u8" => (1, false),
        "u16" => (2, false),
        "u32" => (4, false),
        "u64" | "usize" => (8, false),
        "u128" => (16, false),
        "i8" => (1, true),
        "i16" => (2, true),
        "i32" => (4, true),
        "i64" | "isize" => (8, true),
        "i128" => (16, true),
        "bool" => {
            return value
                .bytes_prefix_as_uint(1)
                .map(|b| Value::Bool(b != 0))
        }
        "char" => return value.bytes_prefix_as_uint(4).map(Value::Uint),
        _ => return None,
    };
    if signed {
        value.bytes_prefix_as_int(width).map(Value::Int)
    } else {
        value.bytes_prefix_as_uint(width).map(Value::Uint)
    }
}

/// The byte list of a sequence of byte values, when every
/// element is one (`None` otherwise).
fn byte_values(items: &[Value]) -> Option<Vec<u8>> {
    items
        .iter()
        .map(|item| match item {
            Value::Uint(b) => u8::try_from(*b).ok(),
            Value::Int(b) => u8::try_from(*b).ok(),
            _ => None,
        })
        .collect()
}

fn bool_str(b: bool) -> &'static str {
    if b {
        "true"
    } else {
        "false"
    }
}

pub(super) fn as_char(v: u128) -> InterpResult<char> {
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

/// `Debug` rendering of a named struct whose field texts are
/// already rendered: the compact `Name { a: x, b: y }` form
/// (width padding applied), or, under `{:#?}`, the multi-line
/// layout the derived form in `format_debug_adt` produces —
/// fields at `indent + 4`, the closing brace at `indent`, no
/// width padding. Handwritten core `Debug` impls built on
/// `debug_struct` (the `std::io::Error` representations) render
/// through this so the two modes never disagree on the fields.
fn debug_struct_text(
    name: &str,
    fields: &[(&str, String)],
    options: &FmtOptions,
    indent: usize,
) -> String {
    if options.alternate() {
        let inner = indent + 4;
        let pad = " ".repeat(inner);
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
        return out;
    }
    let parts: Vec<String> = fields
        .iter()
        .map(|(field_name, text)| format!("{field_name}: {text}"))
        .collect();
    pad_width(&format!("{name} {{ {} }}", parts.join(", ")), options)
}

/// `Debug` rendering of a tuple struct with pre-rendered item
/// texts: `Name(x, y)` compactly, or the multi-line form under
/// `{:#?}`, mirroring `debug_struct_text`.
fn debug_tuple_text(name: &str, items: &[String], options: &FmtOptions, indent: usize) -> String {
    if options.alternate() {
        let inner = indent + 4;
        let pad = " ".repeat(inner);
        let mut out = String::new();
        out.push_str(name);
        out.push_str("(\n");
        for text in items {
            out.push_str(&pad);
            out.push_str(text);
            out.push_str(",\n");
        }
        out.push_str(&" ".repeat(indent));
        out.push(')');
        return out;
    }
    pad_width(&format!("{name}({})", items.join(", ")), options)
}

/// The octets an `Ipv4Addr` value holds, through whichever
/// representation the value model gave it: the declared shape
/// (a one-field aggregate over the `[u8; 4]` array), the array
/// or a four-field aggregate directly, a byte string, or the
/// big-endian `u32` encoding. `None` for any other shape.
pub(super) fn ipv4_octets(value: &Value) -> Option<[u8; 4]> {
    fn byte(value: &Value) -> Option<u8> {
        match value {
            Value::Uint(v) => u8::try_from(*v).ok(),
            Value::Int(v) => u8::try_from(*v).ok(),
            Value::Enum { fields, .. } if fields.len() == 1 => byte(&fields[0]),
            Value::Tuple(items) if items.len() == 1 => byte(&items[0]),
            _ => None,
        }
    }
    match value {
        Value::Enum { fields, .. } if fields.len() == 1 => ipv4_octets(&fields[0]),
        Value::Enum { fields, .. } if fields.len() == 4 => Some([
            byte(&fields[0])?,
            byte(&fields[1])?,
            byte(&fields[2])?,
            byte(&fields[3])?,
        ]),
        Value::Tuple(items) if items.len() == 1 => ipv4_octets(&items[0]),
        Value::Tuple(items) if items.len() == 4 => Some([
            byte(&items[0])?,
            byte(&items[1])?,
            byte(&items[2])?,
            byte(&items[3])?,
        ]),
        Value::Bytes(bytes) if bytes.len() == 4 => Some([bytes[0], bytes[1], bytes[2], bytes[3]]),
        Value::Uint(v) => u32::try_from(*v).ok().map(u32::to_be_bytes),
        Value::Int(v) => u32::try_from(*v).ok().map(u32::to_be_bytes),
        _ => None,
    }
}

/// An `Ipv4Addr` value in its declared shape: a one-field
/// aggregate over the octet array (the shape `Ipv4Addr::new`
/// and the address constants materialise in this model).
pub(super) fn ipv4_value(octets: [u8; 4]) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Tuple(
            octets.iter().map(|b| Value::Uint(u128::from(*b))).collect(),
        )],
    }
}

/// An `Ipv6Addr` value in the model's shape: the address
/// aggregate over a sixteen-byte tuple, mirroring
/// [`ipv4_value`].
pub(super) fn ipv6_value(octets: [u8; 16]) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Tuple(
            octets.iter().map(|b| Value::Uint(u128::from(*b))).collect(),
        )],
    }
}

/// A `SocketAddr` value wrapping a V4 address, in the shape
/// [`format_net_addr`] reads: the enum over the
/// `SocketAddrV4` aggregate of (address, port).
pub(super) fn socket_addr_v4_value(octets: [u8; 4], port: u16) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Enum {
            variant: 0,
            fields: vec![ipv4_value(octets), Value::Uint(u128::from(port))],
        }],
    }
}

/// A `SocketAddr` value wrapping a V6 address: the enum
/// over the `SocketAddrV6` aggregate of (address, port,
/// flowinfo, scope id), std's field order.
pub(super) fn socket_addr_v6_value(octets: [u8; 16], port: u16, flowinfo: u32, scope_id: u32) -> Value {
    Value::Enum {
        variant: 1,
        fields: vec![Value::Enum {
            variant: 0,
            fields: vec![
                ipv6_value(octets),
                Value::Uint(u128::from(port)),
                Value::Uint(u128::from(flowinfo)),
                Value::Uint(u128::from(scope_id)),
            ],
        }],
    }
}

/// The sixteen octets an `Ipv6Addr` value holds, behind the
/// same wrappers [`ipv4_octets`] looks through.
fn ipv6_octets(value: &Value) -> Option<[u8; 16]> {
    fn byte(value: &Value) -> Option<u8> {
        match value {
            Value::Uint(v) => u8::try_from(*v).ok(),
            Value::Int(v) => u8::try_from(*v).ok(),
            Value::Enum { fields, .. } if fields.len() == 1 => byte(&fields[0]),
            Value::Tuple(items) if items.len() == 1 => byte(&items[0]),
            _ => None,
        }
    }
    fn collect_bytes(items: &[Value]) -> Option<[u8; 16]> {
        if items.len() != 16 {
            return None;
        }
        let mut out = [0u8; 16];
        for (slot, item) in out.iter_mut().zip(items) {
            *slot = byte(item)?;
        }
        Some(out)
    }
    match value {
        Value::Enum { fields, .. } if fields.len() == 1 => ipv6_octets(&fields[0]),
        Value::Enum { fields, .. } if fields.len() == 16 => collect_bytes(fields),
        Value::Tuple(items) if items.len() == 1 => ipv6_octets(&items[0]),
        Value::Tuple(items) if items.len() == 16 => collect_bytes(items),
        Value::Bytes(b) if b.len() == 16 => {
            let mut out = [0u8; 16];
            out.copy_from_slice(b);
            Some(out)
        }
        _ => None,
    }
}

/// A `u32` field of a net aggregate (a V6 flowinfo or scope
/// id), behind the same wrappers [`net_port`] looks through.
fn net_u32(value: &Value) -> Option<u32> {
    match value {
        Value::Uint(v) => u32::try_from(*v).ok(),
        Value::Int(v) => u32::try_from(*v).ok(),
        Value::Enum { fields, .. } if fields.len() == 1 => net_u32(&fields[0]),
        Value::Tuple(items) if items.len() == 1 => net_u32(&items[0]),
        _ => None,
    }
}

/// The (octets, port) a `SocketAddrV4` aggregate holds.
fn socket_v4_fields(value: &Value) -> Option<([u8; 4], u16)> {
    let Value::Enum { fields, .. } = value else {
        return None;
    };
    if fields.len() != 2 {
        return None;
    }
    Some((ipv4_octets(&fields[0])?, net_port(&fields[1])?))
}

/// The (octets, port, flowinfo, scope id) a `SocketAddrV6`
/// aggregate holds.
fn socket_v6_fields(value: &Value) -> Option<([u8; 16], u16, u32, u32)> {
    let Value::Enum { fields, .. } = value else {
        return None;
    };
    if fields.len() != 4 {
        return None;
    }
    Some((
        ipv6_octets(&fields[0])?,
        net_port(&fields[1])?,
        net_u32(&fields[2])?,
        net_u32(&fields[3])?,
    ))
}

/// A single-address value (`SocketAddr`, `SocketAddrV4` or
/// `SocketAddrV6`) normalized to the canonical `SocketAddr`
/// form the modelled iterator yields.
pub(super) fn socket_addr_value(value: &Value) -> Option<Value> {
    if let Value::Enum { variant, fields } = value {
        if fields.len() == 1 {
            match variant {
                0 if socket_v4_fields(&fields[0]).is_some() => return Some(value.clone()),
                1 if socket_v6_fields(&fields[0]).is_some() => return Some(value.clone()),
                _ => {}
            }
        }
    }
    if let Some((octets, port)) = socket_v4_fields(value) {
        return Some(socket_addr_v4_value(octets, port));
    }
    if let Some((octets, port, flowinfo, scope_id)) = socket_v6_fields(value) {
        return Some(socket_addr_v6_value(octets, port, flowinfo, scope_id));
    }
    None
}

/// The host `SocketAddr` a guest address value denotes (any
/// of the single-address forms [`socket_addr_value`]
/// normalizes), for the socket table's host operations.
pub(super) fn host_socket_addr(value: &Value) -> Option<std::net::SocketAddr> {
    let normalized = socket_addr_value(value)?;
    let Value::Enum { variant, fields } = &normalized else {
        return None;
    };
    let inner = fields.first()?;
    match variant {
        0 => {
            let (octets, port) = socket_v4_fields(inner)?;
            Some(std::net::SocketAddr::from((octets, port)))
        }
        1 => {
            let (octets, port, flowinfo, scope_id) = socket_v6_fields(inner)?;
            Some(std::net::SocketAddr::V6(std::net::SocketAddrV6::new(
                std::net::Ipv6Addr::from(octets),
                port,
                flowinfo,
                scope_id,
            )))
        }
        _ => None,
    }
}

/// A guest `SocketAddr` value for a host address (the
/// canonical shape [`socket_addr_v4_value`] /
/// [`socket_addr_v6_value`] build), for the socket table's
/// host operation results.
pub(super) fn socket_addr_from_host(addr: std::net::SocketAddr) -> Value {
    match addr {
        std::net::SocketAddr::V4(a) => {
            socket_addr_v4_value(a.ip().octets(), a.port())
        }
        std::net::SocketAddr::V6(a) => socket_addr_v6_value(
            a.ip().octets(),
            a.port(),
            a.flowinfo(),
            a.scope_id(),
        ),
    }
}

/// An `IpAddr` (or a bare `Ipv4Addr` / `Ipv6Addr`) value
/// combined with a port into a `SocketAddr`.
pub(super) fn ip_addr_with_port(value: &Value, port: u16) -> Option<Value> {
    if let Value::Enum {
        variant: 1,
        fields,
    } = value
    {
        if fields.len() == 1 {
            let octets = ipv6_octets(&fields[0])?;
            return Some(socket_addr_v6_value(octets, port, 0, 0));
        }
        return None;
    }
    if let Some(octets) = ipv6_octets(value) {
        return Some(socket_addr_v6_value(octets, port, 0, 0));
    }
    let octets = ipv4_octets(value)?;
    Some(socket_addr_v4_value(octets, port))
}

/// std's `str::to_socket_addrs` order: a literal socket
/// address first, then `host:port` with the host resolved
/// by [`parse_socket_host`]. `Err(())` is every failure —
/// the `io::Error` std reports — never an invented address.
pub(super) fn parse_socket_addr_str(text: &str) -> Result<Vec<Value>, ()> {
    if let Ok(addr) = text.parse::<std::net::SocketAddrV4>() {
        return Ok(vec![socket_addr_v4_value(
            addr.ip().octets(),
            addr.port(),
        )]);
    }
    if let Ok(addr) = text.parse::<std::net::SocketAddrV6>() {
        return Ok(vec![socket_addr_v6_value(
            addr.ip().octets(),
            addr.port(),
            addr.flowinfo(),
            addr.scope_id(),
        )]);
    }
    let (host, port) = split_host_port(text).ok_or(())?;
    parse_socket_host(host, port)
}

/// Resolve the host part of a `host:port` pair (or the
/// address part of an `(address, port)` receiver) at a
/// known port: a literal IP, or the one name the model
/// resolves without a resolver — `localhost`, the loopback
/// name every host table carries. Any other name is the
/// lookup failure std's resolver reports.
fn parse_socket_host(host: &str, port: u16) -> Result<Vec<Value>, ()> {
    if let Ok(ip) = host.parse::<std::net::Ipv4Addr>() {
        return Ok(vec![socket_addr_v4_value(ip.octets(), port)]);
    }
    if let Ok(ip) = host.parse::<std::net::Ipv6Addr>() {
        return Ok(vec![socket_addr_v6_value(ip.octets(), port, 0, 0)]);
    }
    if host == "localhost" {
        return Ok(vec![socket_addr_v4_value([127, 0, 0, 1], port)]);
    }
    Err(())
}

/// Split `host:port`, honouring the bracketed IPv6 form
/// (`[addr]:port`). A non-numeric port (a service name)
/// does not split: the model keeps no services database,
/// and std's failure for an unknown service is the same
/// resolution error the caller reports.
fn split_host_port(text: &str) -> Option<(&str, u16)> {
    if let Some(rest) = text.strip_prefix('[') {
        let (host, after) = rest.split_once(']')?;
        let port = after.strip_prefix(':')?.parse().ok()?;
        return Some((host, port));
    }
    let (host, port) = text.rsplit_once(':')?;
    Some((host, port.parse().ok()?))
}

/// Resolve a static reference's spelling to the program's
/// registered key for the static, among `keys`: exact match
/// first, then the unique key equal modulo the `#DefId`
/// disambiguator a shared name carries and one
/// `extern crate .. as ..` alias segment (a reference path
/// is spelled from the referencing crate, so the alias it
/// names the crate by appears as an extra middle segment
/// the definition's own key lacks). Ambiguity — or no
/// candidate at all — resolves nothing.
pub(super) fn resolve_static_key<'a>(
    keys: impl Iterator<Item = &'a String>,
    name: &str,
) -> Option<&'a str> {
    let keys: Vec<&String> = keys.collect();
    if let Some(key) = keys.iter().find(|k| k.as_str() == name) {
        return Some(key.as_str());
    }
    let base = name.split('#').next().unwrap_or(name);
    let mut found: Option<&str> = None;
    for key in keys {
        let kbase = key.split('#').next().unwrap_or(key.as_str());
        if kbase == base
            || path_matches_modulo_alias(kbase, base)
            || path_matches_modulo_alias(base, kbase)
        {
            if found.is_some() {
                return None;
            }
            found = Some(key.as_str());
        }
    }
    found
}

/// Whether `longer` is `shorter` with exactly one extra
/// middle path segment (the first and last segments — the
/// crate and the item — must agree).
fn path_matches_modulo_alias(longer: &str, shorter: &str) -> bool {
    let l: Vec<&str> = longer.split("::").collect();
    let sh: Vec<&str> = shorter.split("::").collect();
    if l.len() != sh.len() + 1 || l.len() < 3 {
        return false;
    }
    if l[0] != sh[0] || l[l.len() - 1] != sh[sh.len() - 1] {
        return false;
    }
    (1..l.len() - 1).any(|i| {
        let mut v = l.clone();
        v.remove(i);
        v == sh
    })
}

/// The port a `SocketAddrV4` field holds: a bare `u16` scalar,
/// possibly behind one-field wrappers.
fn net_port(value: &Value) -> Option<u16> {
    match value {
        Value::Uint(v) => u16::try_from(*v).ok(),
        Value::Int(v) => u16::try_from(*v).ok(),
        Value::Enum { fields, .. } if fields.len() == 1 => net_port(&fields[0]),
        Value::Tuple(items) if items.len() == 1 => net_port(&items[0]),
        _ => None,
    }
}

/// `Display` / `Debug` text for the `std::net` address types
/// whose rendering is a fixed function of the stored value:
/// `Ipv4Addr` in dotted-decimal, `IpAddr` forwarding to the
/// address it holds, and `SocketAddrV4` / `SocketAddr` as
/// `{ip}:{port}`. Core defines `Debug` for these types as the
/// `Display` text, and both go through `Formatter::pad`, so
/// width / alignment / fill / precision apply to the composed
/// text (`pad_str`). IPv6 variants are not modelled here and
/// keep the existing behaviour (`None`).
fn format_net_addr(
    value: &Value,
    trait_: FmtTrait,
    ty: &str,
    options: &FmtOptions,
) -> Option<String> {
    if !matches!(trait_, FmtTrait::Display | FmtTrait::Debug) {
        return None;
    }
    let dotted = |octets: [u8; 4]| {
        format!("{}.{}.{}.{}", octets[0], octets[1], octets[2], octets[3])
    };
    let socket_v4 = |fields: &[Value]| -> Option<String> {
        if fields.len() != 2 {
            return None;
        }
        let ip = dotted(ipv4_octets(&fields[0])?);
        let port = net_port(&fields[1])?;
        Some(format!("{ip}:{port}"))
    };
    let plain = pointee_tag(ty);
    let text = match plain {
        "std::net::Ipv4Addr" | "core::net::Ipv4Addr" => dotted(ipv4_octets(value)?),
        "std::net::IpAddr" | "core::net::IpAddr" => match value {
            Value::Enum {
                variant: 0,
                fields,
            } if fields.len() == 1 => dotted(ipv4_octets(&fields[0])?),
            _ => return None,
        },
        "std::net::SocketAddrV4" | "core::net::SocketAddrV4" => match value {
            Value::Enum { fields, .. } => socket_v4(fields)?,
            _ => return None,
        },
        "std::net::SocketAddr" | "core::net::SocketAddr" => match value {
            Value::Enum {
                variant: 0,
                fields,
            } if fields.len() == 1 => match &fields[0] {
                Value::Enum { fields, .. } => socket_v4(fields)?,
                _ => return None,
            },
            _ => return None,
        },
        _ => return None,
    };
    Some(pad_str(&text, options))
}

/// core `Formatter::pad`: precision truncates to that many chars,
/// width pads with the fill (default align Left).
pub(super) fn pad_str(s: &str, options: &FmtOptions) -> String {
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

/// core's `Pointer` formatting: the address as lower-case hex
/// behind an unconditional `0x` prefix, padded by the same rules
/// as `pad_integral` -- the sign and prefix count toward the
/// width, the default alignment is Right -- except that
/// zero-padding (the `0` flag, or the alternate form) inserts
/// zeros between the prefix and the digits regardless of the
/// requested alignment or fill, and the alternate form defaults
/// the width to the full pointer width (the prefix plus two hex
/// digits per address byte) when no width is given. Precision
/// plays no part, as for integers.
fn format_pointer(address: u64, options: &FmtOptions) -> String {
    let sign = if options.sign_plus() { "+" } else { "" };
    let digits = format!("{address:x}");
    let width = match options.width() {
        Some(w) => w as usize,
        None if options.alternate() => 2 + 2 * size_of::<usize>(),
        None => 0,
    };
    let total = sign.len() + 2 + digits.len();
    if total >= width {
        return format!("{sign}0x{digits}");
    }
    let side = width - total;
    if options.zero_pad() || options.alternate() {
        return format!("{sign}0x{}{digits}", "0".repeat(side));
    }
    let (pre, post) = padding(
        side,
        options.align().unwrap_or(Align::Right),
        options.fill(),
    );
    format!("{pre}{sign}0x{digits}{post}")
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
/// `std::process::ExitStatus`'s `Display` rendering of a raw
/// (unix) wait status, as the value model stores it (see
/// `exit_status_value`): an exited process carries its code in
/// the high byte; a process ended by a signal carries the
/// signal number in the low seven bits and the core-dump flag
/// in the next bit.
fn exit_status_display(raw: i128) -> String {
    let low = raw & 0x7f;
    if low == 0 {
        return format!("exit code: {}", (raw >> 8) & 0xff);
    }
    if low == 0x7f {
        // A stopped/continued status: the process model never
        // produces one; report it as the raw status, as std's
        // fallback for an unclassifiable status does.
        return format!("exit code: {raw}");
    }
    let core = if raw & 0x80 != 0 { " (core dumped)" } else { "" };
    match signal_display_name(low as i32) {
        Some(name) => format!("signal: {low} ({name}){core}"),
        None => format!("signal: {low}{core}"),
    }
}

/// The conventional name of a signal number, as the platform
/// names it in an `ExitStatus` report (`signal: 6 (SIGABRT)`).
fn signal_display_name(signal: i32) -> Option<&'static str> {
    Some(match signal {
        1 => "SIGHUP",
        2 => "SIGINT",
        3 => "SIGQUIT",
        4 => "SIGILL",
        5 => "SIGTRAP",
        6 => "SIGABRT",
        7 => "SIGBUS",
        8 => "SIGFPE",
        9 => "SIGKILL",
        10 => "SIGUSR1",
        11 => "SIGSEGV",
        12 => "SIGUSR2",
        13 => "SIGPIPE",
        14 => "SIGALRM",
        15 => "SIGTERM",
        16 => "SIGSTKFLT",
        17 => "SIGCHLD",
        18 => "SIGCONT",
        19 => "SIGSTOP",
        20 => "SIGTSTP",
        21 => "SIGTTIN",
        22 => "SIGTTOU",
        23 => "SIGURG",
        24 => "SIGXCPU",
        25 => "SIGXFSZ",
        26 => "SIGVTALRM",
        27 => "SIGPROF",
        28 => "SIGWINCH",
        29 => "SIGIO",
        30 => "SIGPWR",
        31 => "SIGSYS",
        _ => return None,
    })
}

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
/// The payload a one-field storage wrapper denotes when the
/// wrapped value is formatted as the type `tag`, if the type's
/// own value form is never a one-field aggregate: a scalar's
/// value is the scalar, a tuple's the tuple of its arity, an
/// array's the element tuple (or the byte form of a `u8`
/// array). The payload kind must be the type's own kind -- a
/// wrapper around anything else is not this type's storage
/// form and stays the aggregate it is.
fn storage_wrapper_payload(field: &Value, tag: &str) -> Option<Value> {
    let payload = match field {
        Value::Bool(_) if tag == "bool" => field,
        Value::Int(_)
            if matches!(tag, "i8" | "i16" | "i32" | "i64" | "i128" | "isize") =>
        {
            field
        }
        Value::Uint(_)
            if matches!(
                tag,
                "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "char"
            ) =>
        {
            field
        }
        Value::Float(_) if matches!(tag, "f16" | "f32" | "f64" | "f128") => field,
        Value::Str(_) if tag == "str" => field,
        Value::Tuple(items)
            if tag.starts_with('(')
                && tuple_elem_tags(tag, items.len())
                    .iter()
                    .all(|part| !part.is_empty()) =>
        {
            field
        }
        Value::Tuple(items) if array_tag_len(tag) == Some(items.len()) => field,
        Value::Bytes(bytes)
            if array_tag_len(tag) == Some(bytes.len()) && slice_elem_tag(tag) == "u8" =>
        {
            field
        }
        _ => return None,
    };
    Some(payload.clone())
}

/// The element count of an array type tag (`[i32; 3]` -> 3);
/// `None` for a slice tag or a non-array tag.
fn array_tag_len(tag: &str) -> Option<usize> {
    let inner = tag.strip_prefix('[')?.strip_suffix(']')?;
    // An array length sits behind a top-level `;`.
    let mut depth = 0i32;
    for (i, c) in inner.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ';' if depth == 0 => {
                return inner[i + 1..].trim().parse().ok();
            }
            _ => {}
        }
    }
    None
}

/// Whether the type tag names one of the escape
/// iterator types, whose core `Display` is the drain of
/// the iterator (see `format_escape_display`): the
/// `char` types' impls forward to the inner buffer
/// state's own drain-shaped `Display`, and the `str`
/// types' define it directly as
/// `clone().try_for_each(write_char)`. Every one of
/// these types' steps is modelled, so the drain serves
/// them all.
pub(super) fn escape_display_tag(plain: &str) -> bool {
    self_ty_is(plain, "core::char::EscapeDebug")
        || self_ty_is(plain, "std::char::EscapeDebug")
        || self_ty_is(plain, "core::char::EscapeDefault")
        || self_ty_is(plain, "std::char::EscapeDefault")
        || self_ty_is(plain, "core::char::EscapeUnicode")
        || self_ty_is(plain, "std::char::EscapeUnicode")
        || self_ty_is(plain, "core::str::EscapeDebug")
        || self_ty_is(plain, "std::str::EscapeDebug")
        || self_ty_is(plain, "core::str::EscapeDefault")
        || self_ty_is(plain, "std::str::EscapeDefault")
        || self_ty_is(plain, "core::str::EscapeUnicode")
        || self_ty_is(plain, "std::str::EscapeUnicode")
}

/// The fat pointer at the end of a box aggregate's
/// field-0 chain, when the chain ends in one: following
/// first fields through the `Unique` / `NonNull` wrappers
/// reaches the pointer the box carries, and for a fat box
/// that pointer is the fat one, metadata included. `None`
/// when the chain ends in anything else.
fn box_fat_pointer(value: &Value) -> Option<Value> {
    let mut cur = value;
    loop {
        match cur {
            Value::Enum { fields, .. } if !fields.is_empty() => {
                cur = &fields[0];
            }
            Value::Tuple(items) if !items.is_empty() => {
                cur = &items[0];
            }
            fat @ Value::FatRef { .. } => return Some(fat.clone()),
            _ => return None,
        }
    }
}

pub(super) fn pointee_tag(ty: &str) -> &str {
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

/// The element type a deref-forwarding guard tag names
/// (`std::cell::Ref<'_, i32>` -> `i32`): the first generic
/// argument that is not a lifetime. `None` for every other
/// tag. The guard types are core's `Ref` / `RefMut`, whose
/// `Display` and `Debug` impls format the guarded value;
/// both spellings of the defining crate are accepted, as
/// elsewhere. The head must be followed directly by the
/// argument list, so a longer name sharing the prefix
/// (`RefCell`) does not match.
/// How a lock guard value reaches its guarded value (see
/// `sync_guard_elem_tag`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SyncGuardKind {
    /// `MutexGuard` / `RwLockWriteGuard`: the first field is
    /// the reference to the lock itself.
    LockRef,
    /// `RwLockReadGuard`: the first field is the `NonNull`
    /// data pointer its constructor took from the lock, which
    /// denotes the lock's `UnsafeCell` place.
    ReadPtr,
    /// A mapped guard (`MappedMutexGuard`,
    /// `MappedRwLockReadGuard`, `MappedRwLockWriteGuard`):
    /// the first field is the `NonNull` data pointer `map`
    /// built from the mapping closure's returned reference,
    /// which denotes the mapped component itself.
    MappedPtr,
}

/// The element type tag of a std lock guard type tag
/// (`std::sync::MutexGuard<T>`, `std::sync::RwLockReadGuard<T>`,
/// `std::sync::RwLockWriteGuard<T>`, the mapped guards
/// `std::sync::MappedMutexGuard<U>` /
/// `std::sync::MappedRwLockReadGuard<U>` /
/// `std::sync::MappedRwLockWriteGuard<U>`, and the `nonpoison`
/// spellings), paired with how the guard reaches its guarded
/// value (see `SyncGuardKind`). For a mapped guard the element
/// type is the mapped component's type. The head must be
/// followed directly by the argument list, as in
/// `guard_elem_tag`.
pub(super) fn sync_guard_elem_tag(ty: &str) -> Option<(String, SyncGuardKind)> {
    use SyncGuardKind::*;
    let plain = pointee_tag(ty);
    let mut read_guard = None;
    for (head, kind) in [
        ("std::sync::RwLockReadGuard", ReadPtr),
        ("std::sync::nonpoison::RwLockReadGuard", ReadPtr),
        ("std::sync::MappedMutexGuard", MappedPtr),
        ("std::sync::MappedRwLockReadGuard", MappedPtr),
        ("std::sync::MappedRwLockWriteGuard", MappedPtr),
        ("std::sync::nonpoison::MappedMutexGuard", MappedPtr),
        ("std::sync::nonpoison::MappedRwLockReadGuard", MappedPtr),
        ("std::sync::nonpoison::MappedRwLockWriteGuard", MappedPtr),
        ("std::sync::MutexGuard", LockRef),
        ("std::sync::RwLockWriteGuard", LockRef),
        ("std::sync::nonpoison::MutexGuard", LockRef),
        ("std::sync::nonpoison::RwLockWriteGuard", LockRef),
    ] {
        if plain
            .strip_prefix(head)
            .is_some_and(|rest| rest.starts_with('<'))
        {
            read_guard = Some(kind);
            break;
        }
    }
    let read_guard = read_guard?;
    let open = plain.find('<')?;
    let inner = tag_group_inner(plain, open)?;
    split_tag_list(inner)
        .into_iter()
        .find(|arg| !is_lifetime_arg(arg))
        .map(|elem| (elem.to_string(), read_guard))
}

/// The printed name and element type tag of a std lock type
/// tag (`std::sync::Mutex<T>` / `std::sync::RwLock<T>`),
/// paired with whether the lock is a read-write lock: std's
/// `Debug` for an `RwLock` acquires a read guard, so only a
/// writer blocks it, and its `<locked>` placeholder is the
/// bare formatted text where the `Mutex`'s is a quoted
/// string. The head must be followed directly by the
/// argument list, as in `guard_elem_tag`, so the guard
/// types (`MutexGuard`, ...) do not match.
pub(super) fn sync_lock_tag(ty: &str) -> Option<(&'static str, String, bool)> {
    let plain = pointee_tag(ty);
    for (head, name, rw_lock) in [
        ("std::sync::Mutex", "Mutex", false),
        ("std::sync::RwLock", "RwLock", true),
    ] {
        if plain
            .strip_prefix(head)
            .is_some_and(|rest| rest.starts_with('<'))
        {
            let open = plain.find('<')?;
            let inner = tag_group_inner(plain, open)?;
            return split_tag_list(inner)
                .into_iter()
                .find(|arg| !is_lifetime_arg(arg))
                .map(|elem| (name, elem.to_string(), rw_lock));
        }
    }
    None
}

/// The scalar a lock field's storage bottoms out in: the
/// word an atomic wrapper chain (an `AtomicU32` over an
/// `UnsafeCell`, and the like) holds, read through any
/// number of one-field wrappers.
fn scalar_word(value: &Value) -> Option<u128> {
    match value {
        Value::Uint(word) => Some(*word),
        Value::Int(word) => Some(*word as u128),
        Value::Bool(word) => Some(*word as u128),
        Value::Enum { fields, .. } if fields.len() == 1 => scalar_word(&fields[0]),
        Value::Tuple(fields) if fields.len() == 1 => scalar_word(&fields[0]),
        _ => None,
    }
}

/// The state word of a sys lock value: the state futex is
/// the lock's first field, and the futex itself is an
/// atomic wrapper over the word.
fn lock_state_word(sys: &Value) -> Option<u128> {
    match sys {
        Value::Enum { fields, .. } | Value::Tuple(fields) if !fields.is_empty() => {
            scalar_word(&fields[0])
        }
        other => scalar_word(other),
    }
}

pub(super) fn guard_elem_tag(ty: &str) -> Option<String> {
    let plain = pointee_tag(ty);
    let is_guard = [
        "std::cell::Ref",
        "core::cell::Ref",
        "std::cell::RefMut",
        "core::cell::RefMut",
    ]
    .iter()
    .any(|head| {
        plain
            .strip_prefix(head)
            .is_some_and(|rest| rest.starts_with('<'))
    });
    if !is_guard {
        return None;
    }
    let open = plain.find('<')?;
    let inner = tag_group_inner(plain, open)?;
    split_tag_list(inner)
        .into_iter()
        .find(|arg| !is_lifetime_arg(arg))
        .map(str::to_string)
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
    let mut prev = '\0';
    for (i, c) in s.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
            // The `>` of a function pointer signature's `->`
            // closes nothing (see `split_top_level`).
            '>' if prev == '-' => {}
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
        prev = c;
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
            // The `>` of a function pointer signature's `->`
            // closes nothing (see `split_top_level`).
            b'>' if i > 0 && bytes[i - 1] == b'-' => {}
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
/// The element tag of the reference a reference-yielding
/// iterator's tag names as its item: `std::slice::Iter<T>`
/// and `std::option::Iter<T>` spell it directly, and the
/// adapters that pass the item through unchanged (`Filter`,
/// `Rev`, `Take`, `Skip`, `StepBy`, `Fuse`, `Peekable`, and
/// `Chain`'s first side) forward their inner iterator's.
/// Empty when the tag does not spell it -- a `Map`'s item
/// is its closure's return type, which the tag does not
/// carry.
pub(super) fn ref_item_element_ty(inner_ty: &str) -> String {
    let mut ty = inner_ty;
    loop {
        if ty.starts_with("std::slice::Iter<") || ty.starts_with("std::option::Iter<") {
            return first_generic_arg(ty).unwrap_or("").to_string();
        }
        if ty.starts_with("std::iter::Filter<")
            || ty.starts_with("std::iter::Rev<")
            || ty.starts_with("std::iter::Take<")
            || ty.starts_with("std::iter::Skip<")
            || ty.starts_with("std::iter::StepBy<")
            || ty.starts_with("std::iter::Fuse<")
            || ty.starts_with("std::iter::Peekable<")
            || ty.starts_with("std::iter::Chain<")
        {
            match first_generic_arg(ty) {
                Some(inner) => ty = inner,
                None => return String::new(),
            }
            continue;
        }
        return String::new();
    }
}

/// Whether a rendered type tag is a slice type `[T]`, as opposed
/// to an array type `[T; N]`: the tag is bracketed and carries no
/// semicolon at its own bracket level. An element type may itself
/// be an array (`[[i32; 2]]` is a slice of arrays), whose
/// semicolon sits one level deeper and does not make the outer
/// type an array.
pub(super) fn is_slice_tag(tag: &str) -> bool {
    let Some(rest) = tag.strip_prefix('[') else {
        return false;
    };
    let Some(inner) = rest.strip_suffix(']') else {
        return false;
    };
    let mut depth = 0i32;
    let mut prev = '\0';
    for c in inner.chars() {
        match c {
            '<' | '(' | '[' => depth += 1,
            // The `>` of a function pointer signature's `->`
            // closes nothing (see `split_top_level`).
            '>' if prev == '-' => {}
            '>' | ')' | ']' => depth -= 1,
            ';' if depth == 0 => return false,
            _ => {}
        }
        prev = c;
    }
    true
}

pub(super) fn first_generic_arg(tag: &str) -> Option<&str> {
    let open = tag.find('<')?;
    let inner = tag_group_inner(tag, open)?;
    split_tag_list(inner).into_iter().next()
}

/// The second generic argument of a rendered type, if any: the
/// callable type of an iterator adapter tag (`Map<I, F>` and its
/// siblings all carry the callable second).
pub(super) fn second_generic_arg(tag: &str) -> Option<&str> {
    let open = tag.find('<')?;
    let inner = tag_group_inner(tag, open)?;
    split_tag_list(inner).into_iter().nth(1)
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

/// Registration fragment of the builtin name registry: the `core::fmt` surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn fmt_lookup(canonical: &str) -> Option<Builtin> {
    use crate::interp::FmtBuilderKind;
    use crate::value::FmtTrait as F;
    Some(match canonical {
        "std::fmt::write" | "core::fmt::write" => Builtin::FmtWrite,
        "std::fmt::format" | "std::fmt::format::format_inner"
        | "alloc::fmt::format::format_inner" => Builtin::Format,
        // The integer `Display` digit writer (see
        // `Builtin::FmtNumFmt`): an inherent method on each
        // unsigned integer type, in the `imp` submodule except
        // for `u128`, whose impl sits at the `num` level.
        "core::fmt::num::imp::<impl u8>::_fmt"
        | "core::fmt::num::imp::<impl u16>::_fmt"
        | "core::fmt::num::imp::<impl u32>::_fmt"
        | "core::fmt::num::imp::<impl u64>::_fmt"
        | "core::fmt::num::imp::<impl usize>::_fmt"
        | "core::fmt::num::<impl u128>::_fmt" => Builtin::FmtNumFmt,
        // The public `NumBuffer` entry of the same digit
        // writer (see `Builtin::FmtNumFormatInto`): the
        // `imp` submodule holds every type except the
        // 128-bit pair, whose impls sit at the `num` level.
        // The capacity is the digit count of the type's
        // maximum, plus a sign slot for a signed type.
        "core::fmt::num::imp::<impl i8>::format_into" => Builtin::FmtNumFormatInto {
            signed: true,
            buf_len: 4,
        },
        "core::fmt::num::imp::<impl i16>::format_into" => Builtin::FmtNumFormatInto {
            signed: true,
            buf_len: 6,
        },
        "core::fmt::num::imp::<impl i32>::format_into" => Builtin::FmtNumFormatInto {
            signed: true,
            buf_len: 11,
        },
        "core::fmt::num::imp::<impl i64>::format_into" => Builtin::FmtNumFormatInto {
            signed: true,
            buf_len: 20,
        },
        "core::fmt::num::imp::<impl isize>::format_into" => Builtin::FmtNumFormatInto {
            signed: true,
            buf_len: 20,
        },
        "core::fmt::num::<impl i128>::format_into" => Builtin::FmtNumFormatInto {
            signed: true,
            buf_len: 40,
        },
        "core::fmt::num::imp::<impl u8>::format_into" => Builtin::FmtNumFormatInto {
            signed: false,
            buf_len: 3,
        },
        "core::fmt::num::imp::<impl u16>::format_into" => Builtin::FmtNumFormatInto {
            signed: false,
            buf_len: 5,
        },
        "core::fmt::num::imp::<impl u32>::format_into" => Builtin::FmtNumFormatInto {
            signed: false,
            buf_len: 10,
        },
        "core::fmt::num::imp::<impl u64>::format_into" => Builtin::FmtNumFormatInto {
            signed: false,
            buf_len: 20,
        },
        "core::fmt::num::imp::<impl usize>::format_into" => Builtin::FmtNumFormatInto {
            signed: false,
            buf_len: 20,
        },
        "core::fmt::num::<impl u128>::format_into" => Builtin::FmtNumFormatInto {
            signed: false,
            buf_len: 39,
        },
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
        _ => return None,
    })
}
