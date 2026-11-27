//! Type queries and info listings: `whatis` / `ptype` rendering and the `info functions` / `info locals` transcripts.

use crate::interp::{Frame, Interpreter};
use crate::ir::Ty;
use crate::value::Value;
use super::desc::{Desc, dwarf_type_name, field_desc, first_generic_arg, smart_pointer_pointee, split_top_level, tag_base_name};
use super::evaluate::{EvaluatedValue, evaluate_expr, resolve_base};
use super::parse::{ExprStep, parse_expr};
use super::render_gdb::render_gdb;
use super::render_lldb::scalar_desc_bit_width;
use super::session::{DebugControl, DebugPrint, LldbFormats, LldbNumFormat, PrintFormat, frame_position};

/// The rendering an lldb type-format registration prescribes
/// for a printed scalar, when the session registered a format
/// for the value's own type. The registration is keyed by type
/// name; the evaluated value's recorded tag names its type
/// exactly at the top level of a print, which is where the
/// debuginfo suite's registrations bite (a `char`'s descriptor
/// also survives expression steps, and is consulted as well).
pub(super) fn lldb_format_scalar(
    value: &Value,
    tag: Option<&str>,
    desc: Option<&Desc>,
    formats: LldbFormats,
) -> Option<String> {
    // The registered type's width: an unsigned rendering is
    // the unsigned reading of the value's stored bytes, so a
    // sign-extended negative value must be masked back to the
    // width of the type the format was registered for.
    let (format, width) = match tag {
        Some("char") => (formats.char_format, 32),
        Some("i8") => (formats.i8_format, 8),
        Some("u8") => (formats.u8_format, 8),
        _ => match desc {
            Some(Desc::Char) => (formats.char_format, 32),
            // A projected value's tag is cleared by the
            // expression steps, but its descriptor keeps the
            // concrete integer type: a registration for that
            // type applies to the projection as it does to
            // the variable itself.
            Some(Desc::Int("i8")) => (formats.i8_format, 8),
            Some(Desc::Int("u8")) => (formats.u8_format, 8),
            _ => (None, 0),
        },
    };
    let format = format?;
    let mask: u128 = (1u128 << width) - 1;
    let v: i128 = match value {
        Value::Int(v) => *v,
        Value::Uint(v) => *v as i128,
        _ => return None,
    };
    Some(match format {
        LldbNumFormat::Decimal => format!("{v}"),
        LldbNumFormat::Unsigned => format!("{}", (v as u128) & mask),
        LldbNumFormat::Hex => format!("0x{:x}", (v as u128) & mask),
    })
}

/// The environment-name form of a rendered type name: a
/// closure (or coroutine) type's debuginfo name is its
/// environment struct's name, `{closure_env#N}` for the
/// `{closure#N}` segment the conversion records, with the
/// closure type's own generic arguments (kind, signature,
/// upvars) dropped — the environment type carries none of
/// them in its DWARF name. Applied segment by segment, so a
/// closure type nested in a tuple or generic argument list
/// is rewritten in place.
pub(super) fn closure_env_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut rest = name;
    while !rest.is_empty() {
        let hit = ["::{closure#", "::{coroutine#"]
            .iter()
            .filter_map(|seg| rest.find(seg).map(|i| (i, *seg)))
            .min_by_key(|(i, _)| *i);
        let Some((idx, seg)) = hit else {
            out.push_str(rest);
            break;
        };
        let env_seg = if seg.contains("closure") {
            "::{closure_env#"
        } else {
            "::{coroutine_env#"
        };
        out.push_str(&rest[..idx]);
        out.push_str(env_seg);
        rest = &rest[idx + seg.len()..];
        // The segment's `{...#N}` tail up to its closing brace.
        match rest.find('}') {
            Some(close) => {
                out.push_str(&rest[..=close]);
                rest = &rest[close + 1..];
            }
            None => {
                out.push_str(rest);
                break;
            }
        }
        // A generic argument group directly after the segment
        // is the closure type's own arguments: drop it.
        if rest.starts_with('<') {
            let mut depth = 0i32;
            let mut end = None;
            let mut chars = rest.char_indices().peekable();
            while let Some((i, c)) = chars.next() {
                match c {
                    '<' => depth += 1,
                    // A `->` arrow (a function signature's
                    // return marker) is not a group close.
                    '-' if chars.peek().is_some_and(|(_, n)| *n == '>') => {
                        chars.next();
                    }
                    '>' => {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(i);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if let Some(end) = end {
                rest = &rest[end + 1..];
            }
        }
    }
    out
}

/// The candidate type names of an evaluated expression, for
/// a type query: the single name [`expr_type_name`] computes,
/// plus — when the recorded tag names a function item (a
/// `FnDef` element, bare or inside a tuple) — the instance's
/// fn-pointer spelling from the driver's display-name table
/// (a function item's debuginfo type is its signature, not
/// its name). A tag that names a generic definition without
/// its arguments matches every recorded instance of it, and
/// each is a candidate, as elsewhere in the renderers.
pub(super) fn expr_type_names(
    interp: &Interpreter,
    ev: &EvaluatedValue,
) -> Vec<String> {
    let Some(name) = expr_type_name(ev) else {
        return Vec::new();
    };
    let mut out = vec![name.clone()];
    let Some(tag) = &ev.tag else {
        return out;
    };
    let table = &interp.program().fn_display_names;
    if table.is_empty() {
        return out;
    }
    // The tag's FnDef candidates, element by element: a tuple
    // tag's elements resolve independently, and the candidate
    // names are the cartesian product of the resolutions.
    let resolve = |elem: &str| -> Vec<String> {
        let elem = elem.trim();
        let mut hits: Vec<&str> = table
            .iter()
            .filter(|(display, _, ptr)| {
                if ptr.is_empty() {
                    return false;
                }
                if display == elem
                    || display
                        .strip_prefix(elem)
                        .is_some_and(|rest| rest.starts_with('<'))
                {
                    return true;
                }
                // The tag's item name omits an extern block's
                // `{extern#N}` segment the display name
                // carries (`type_names::printf` for
                // `type_names::{extern#0}::printf`).
                let mut stripped = display.clone();
                while let Some(start) = stripped.find("::{extern#") {
                    let Some(close) = stripped[start..].find('}') else {
                        break;
                    };
                    stripped.replace_range(start..start + close + 1, "");
                }
                stripped == elem
                    || stripped
                        .strip_prefix(elem)
                        .is_some_and(|rest| rest.starts_with('<'))
            })
            .map(|(_, _, ptr)| ptr.as_str())
            .collect();
        hits.sort_unstable();
        hits.dedup();
        hits.into_iter().map(str::to_string).collect()
    };
    let trimmed = tag.trim();
    let mut combos: Vec<String> = Vec::new();
    if trimmed.starts_with('(') && trimmed.ends_with(')') && trimmed.len() > 2 {
        let inner = &trimmed[1..trimmed.len() - 1];
        let parts = split_top_level(inner);
        let mut acc: Vec<Vec<String>> = vec![Vec::new()];
        for part in &parts {
            let hits = resolve(part);
            if hits.is_empty() {
                for combo in &mut acc {
                    combo.push(part.clone());
                }
            } else {
                let mut next = Vec::new();
                for combo in &acc {
                    for hit in &hits {
                        let mut c = combo.clone();
                        c.push(hit.clone());
                        next.push(c);
                    }
                }
                acc = next;
                if acc.len() > 8 {
                    acc.truncate(8);
                }
            }
        }
        for combo in acc {
            let joined = if combo.len() == 1 {
                format!("({},)", combo[0])
            } else {
                format!("({})", combo.join(", "))
            };
            combos.push(joined);
        }
    } else {
        combos = resolve(trimmed)
            .into_iter()
            .map(|hit| hit)
            .collect();
    }
    for combo in combos {
        let combo = dwarf_type_name(&closure_env_name(&combo));
        if combo != name && !out.contains(&combo) {
            out.push(combo);
        }
    }
    out
}

/// The rendering a per-command print format (`p/u`, `p/d`,
/// `p/x`, `p/t` in a debugger command) prescribes for a
/// printed scalar in an lldb session: the format names how
/// this print reads the value, overriding both the default
/// summary and any registered type format. The unsigned and
/// radix forms read the stored bits at the value's own
/// width (its descriptor's, or the recorded tag's when the
/// descriptor is gone); the decimal form reads a signed
/// value as signed and any other scalar as its stored
/// number, so a char prints as its code point rather than
/// its character summary. Returns `None` for values and
/// formats the reading does not cover (a character format,
/// non-scalars, unknown widths), leaving the default
/// rendering in place.
pub(super) fn lldb_command_format_scalar(
    value: &Value,
    tag: Option<&str>,
    desc: Option<&Desc>,
    format: PrintFormat,
) -> Option<String> {
    let (signed_v, bits) = match value {
        Value::Int(v) => (*v, *v as u128),
        Value::Uint(v) => (*v as i128, *v),
        _ => return None,
    };
    let width = scalar_desc_bit_width(desc).or(match tag {
        Some("i8" | "u8") => Some(8),
        Some("i16" | "u16") => Some(16),
        Some("i32" | "u32") => Some(32),
        Some("i64" | "u64" | "isize" | "usize") => Some(64),
        Some("i128" | "u128") => Some(128),
        Some("char") => Some(32),
        _ => None,
    })?;
    let mask: u128 = if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    };
    Some(match format {
        PrintFormat::Decimal => {
            if matches!(value, Value::Int(_)) {
                format!("{signed_v}")
            } else {
                format!("{bits}")
            }
        }
        PrintFormat::Unsigned => format!("{}", bits & mask),
        PrintFormat::Hex => format!("0x{:x}", bits & mask),
        PrintFormat::Binary => format!("0b{:b}", bits & mask),
        PrintFormat::Char => return None,
    })
}

/// A `core::num::NonZero<T>` value unwrapped to its scalar:
/// the type's DWARF form is the primitive itself carrying a
/// validity range, and both debuggers print the bare scalar.
/// The value arrives as nested one-field wrappers — the
/// `NonZeroInner` associated type hides the primitive one
/// layer down, behind a type alias the debug-shape table
/// cannot resolve — so unwrapping follows the value's own
/// one-field layers until a scalar bottoms out. Returns the
/// scalar and the descriptor of the `NonZero`'s argument.
pub(super) fn nonzero_scalar(
    value: &Value,
    desc: Option<&Desc>,
) -> Option<(Value, Desc)> {
    let Some(Desc::Adt(tag)) = desc else {
        return None;
    };
    if tag_base_name(tag) != "NonZero" {
        return None;
    }
    let inner = first_generic_arg(tag)?;
    let mut cur = value.clone();
    loop {
        match cur {
            Value::Enum { fields, .. } if fields.len() == 1 => {
                cur = fields.into_iter().next().unwrap();
            }
            Value::Tuple(fields) if fields.len() == 1 => {
                cur = fields.into_iter().next().unwrap();
            }
            Value::Int(_) | Value::Uint(_) => {
                return Some((cur, Desc::from_tag(&inner)));
            }
            _ => return None,
        }
    }
}

/// The declared type name of an evaluated expression, as a
/// type query (`whatis`) prints it: the recorded type tag in
/// its DWARF (definition-path) form when the conversion
/// recorded one, else the name rebuilt from the rendering
/// descriptor.
pub(super) fn expr_type_name(ev: &EvaluatedValue) -> Option<String> {
    if matches!(ev.desc, Some(Desc::FnPtr))
        || matches!(&ev.tag, Some(tag) if tag == "FnPtr")
    {
        // A function pointer's tag is the type's kind name,
        // not a signature; gdb spells the type as a raw
        // pointer to the unit-signature function.
        return Some("*mut fn ()".to_string());
    }
    if let Some(tag) = &ev.tag {
        // A function pointer whose recorded tag is the
        // signature itself (the form the conversion records
        // for a declared `fn(...)` type): gdb spells the type
        // as a raw pointer to the signature.
        if let Some(name) = fn_pointer_tag_name(tag) {
            return Some(name);
        }
        return Some(dwarf_type_name(&closure_env_name(tag)));
    }
    // A trailing cast step's spelled type, pointer layer
    // included: the cast's materialized view carries the
    // pointee's descriptor (an array view for an array
    // pointer), but the expression's type is the cast's own
    // until a dereference consumes the pointer layer.
    if let Some(cast) = &ev.cast_type {
        if let Some(name) = desc_type_name(cast) {
            return Some(name);
        }
    }
    ev.desc.as_ref().and_then(desc_type_name)
}


/// gdb's spelling of a function-pointer type whose recorded
/// tag is the signature itself: `*mut ` and the signature
/// with a space before its argument list (`fn()` becomes
/// `*mut fn ()`, `unsafe fn(i32) -> bool` becomes
/// `*mut unsafe fn (i32) -> bool`). `None` for any other
/// tag -- in particular for the bare `FnPtr` kind name, whose
/// spelling the caller handles separately.
fn fn_pointer_tag_name(tag: &str) -> Option<String> {
    let mut rest = tag;
    let mut prefix = String::new();
    loop {
        if let Some(r) = rest.strip_prefix("unsafe ") {
            prefix.push_str("unsafe ");
            rest = r;
        } else if let Some(r) = rest.strip_prefix("extern ") {
            // The ABI is a quoted string: `extern "C" fn(..)`.
            let quoted = r.strip_prefix('"')?;
            let close = quoted.find('"')?;
            prefix.push_str("extern ");
            prefix.push_str(&r[..close + 2]);
            prefix.push(' ');
            rest = quoted[close + 1..].strip_prefix(' ')?;
        } else {
            break;
        }
    }
    let args = rest.strip_prefix("fn")?;
    if !args.starts_with('(') {
        return None;
    }
    Some(format!("*mut {prefix}fn {args}"))
}

/// A rendering descriptor's type name, for the forms whose
/// names the descriptor fully determines.
pub(super) fn desc_type_name(desc: &Desc) -> Option<String> {
    Some(match desc {
        Desc::Unit => "()".to_string(),
        Desc::Bool => "bool".to_string(),
        Desc::Char => "char".to_string(),
        Desc::Str => "str".to_string(),
        Desc::Tuple(ts) => {
            let parts: Option<Vec<String>> =
                ts.iter().map(desc_type_name).collect();
            let parts = parts?;
            if parts.len() == 1 {
                format!("({},)", parts[0])
            } else {
                format!("({})", parts.join(", "))
            }
        }
        Desc::Array(elem, n) => {
            format!("[{}; {n}]", desc_type_name(elem)?)
        }
        Desc::Slice(elem) => format!("[{}]", desc_type_name(elem)?),
        Desc::Ref(inner) => format!("&{}", desc_type_name(inner)?),
        Desc::RefMut(inner) => format!("&mut {}", desc_type_name(inner)?),
        Desc::PtrConst(inner) => format!("*const {}", desc_type_name(inner)?),
        Desc::PtrMut(inner) => format!("*mut {}", desc_type_name(inner)?),
        Desc::Adt(tag) => dwarf_type_name(tag),
        Desc::Closure(name) | Desc::Coroutine(name) => {
            closure_env_name(name)
        }
        Desc::FnPtr => "*mut fn ()".to_string(),
        Desc::Int(name) => name.to_string(),
        Desc::Float(name) => name.to_string(),
        Desc::Unknown => return None,
    })
}

/// The pointee descriptor of a pointer-family descriptor
/// (shared or mutable reference, raw const or mutable
/// pointer); `None` for any other descriptor.
pub(super) fn pointer_pointee(desc: &Desc) -> Option<&Desc> {
    match desc {
        Desc::Ref(inner) | Desc::RefMut(inner) | Desc::PtrConst(inner) | Desc::PtrMut(inner) => {
            Some(inner)
        }
        _ => None,
    }
}

/// The descriptor past the implicit dereference a field or
/// index step performs: a reference's pointee, a smart
/// pointer's pointee, anything else unchanged.
fn desc_through_pointer(desc: Desc) -> Desc {
    match desc {
        Desc::Ref(inner) | Desc::RefMut(inner) | Desc::PtrConst(inner) | Desc::PtrMut(inner) => {
            *inner
        }
        Desc::Adt(tag) => smart_pointer_pointee(&tag)
            .map(|t| Desc::from_tag(&t))
            .unwrap_or(Desc::Adt(tag)),
        other => other,
    }
}

/// The descriptor of a named field, from the recorded shapes
/// alone (the type-only counterpart of `apply_field_name`):
/// an ADT variant's or closure environment's recorded field,
/// then the pseudo-fields (`length`, `data_ptr`).
fn type_field_name_desc(
    interp: &Interpreter,
    desc: &Desc,
    name: &str,
) -> Option<Desc> {
    if let Desc::Adt(tag) = desc {
        if let Some(shape) = interp.program().adt_debug_shapes.get(tag) {
            if let Some(vshape) = shape.variants.first() {
                if let Some(field) = vshape
                    .fields
                    .iter()
                    .find(|f| f.name.as_deref() == Some(name))
                {
                    return Some(Desc::from_tag(&field.ty));
                }
            }
        }
        if let Some(shape) = interp.program().env_debug_shapes.get(tag) {
            if let Some(field) = shape
                .upvars
                .iter()
                .find(|f| f.name.as_deref() == Some(name))
            {
                return Some(Desc::from_tag(&field.ty));
            }
        }
    }
    match name {
        "length" => Some(Desc::Int("usize")),
        "data_ptr" => match desc {
            Desc::Ref(inner)
            | Desc::RefMut(inner)
            | Desc::PtrConst(inner)
            | Desc::PtrMut(inner) => match inner.as_ref() {
                Desc::Slice(elem) => Some(Desc::Ref(elem.clone())),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

/// An evaluated form of an expression for a type query,
/// resolved from the recorded types alone: the base variable's
/// descriptor walked through the expression's steps, with no
/// pointee ever read. The value is a placeholder; only the
/// descriptor (and, for a step-free expression, the recorded
/// tag) carries information.
fn type_only_value(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    expr_text: &str,
) -> Option<EvaluatedValue> {
    let expr = parse_expr(expr_text).ok()?;
    let base = resolve_base(interp, ctl, frames, &expr).ok()?;
    let mut desc = base.desc.clone();
    let mut cast_type: Option<Desc> = None;
    for step in &expr.steps {
        cast_type = match step {
            ExprStep::Cast(target) => Some(target.clone()),
            _ => None,
        };
        desc = match (step, desc) {
            (
                ExprStep::Deref,
                Some(
                    Desc::Ref(inner)
                    | Desc::RefMut(inner)
                    | Desc::PtrConst(inner)
                    | Desc::PtrMut(inner),
                ),
            ) => Some(*inner),
            (ExprStep::Deref, Some(Desc::Adt(tag))) => {
                smart_pointer_pointee(&tag).map(|t| Desc::from_tag(&t))
            }
            (ExprStep::Deref, other) => other,
            (ExprStep::Field(index), Some(d)) => {
                let d = desc_through_pointer(d);
                field_desc(interp, &d, &Value::Uninit, *index)
            }
            (ExprStep::FieldName(name), Some(d)) => {
                let d = desc_through_pointer(d);
                type_field_name_desc(interp, &d, name)
            }
            (ExprStep::Index(_), Some(d)) => {
                match desc_through_pointer(d) {
                    Desc::Array(elem, _) | Desc::Slice(elem) => Some(*elem),
                    _ => None,
                }
            }
            (ExprStep::Cast(target), _) => Some(match target.clone() {
                Desc::Ref(inner)
                | Desc::RefMut(inner)
                | Desc::PtrConst(inner)
                | Desc::PtrMut(inner)
                    if matches!(*inner, Desc::Array(..) | Desc::Slice(_)) =>
                {
                    *inner
                }
                other => other,
            }),
            (_, None) => None,
        };
    }
    Some(EvaluatedValue {
        value: Value::Uninit,
        desc,
        char: base.char,
        tag: if expr.steps.is_empty() {
            base.tag.clone()
        } else {
            None
        },
        cast_type,
        pending_cast: None,
        pending_array_view: None,
    })
}

/// Render one type query (gdb `whatis`, or `ptype` with
/// `layout`): the `type = <name>` head line, plus — for
/// `ptype` over an aggregate shape — the shape's field lines
/// in gdb's layout form.
pub(super) fn render_type_query(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    print: &DebugPrint,
    layout: bool,
) -> Vec<String> {
    let ev = match evaluate_expr(interp, ctl, frames, &print.expr) {
        Ok(ev) => ev,
        Err(e) => match type_only_value(interp, ctl, frames, &print.expr) {
            // A type query needs no value: when the value
            // evaluation fails (a null pointer's pointee, say),
            // the type still resolves from the recorded tags
            // and projection metadata alone.
            Some(ev) => ev,
            None => return vec![format!("type = <error: {e}>")],
        },
    };
    let names = expr_type_names(interp, &ev);
    let Some(name) = names.first().cloned() else {
        return vec!["type = <unknown>".to_string()];
    };
    if !layout {
        return names
            .iter()
            .map(|name| format!("type = {name}"))
            .collect();
    }
    // The layout form of a closure environment: gdb spells the
    // type as a struct over its captures, a capture taken by
    // reference appearing as the raw pointer the environment
    // stores.
    if let Some(Desc::Adt(tag)) = &ev.desc {
        // The shape table's key may carry the closure's
        // generic arguments or not, depending on how the
        // conversion recorded the environment; try the
        // recorded tag, then its argument-stripped form, then
        // the shape whose DWARF name is the tag's environment
        // name (`{closure#N}` spelled `{closure_env#N}`).
        let bare = tag.split('<').next().unwrap_or(tag);
        let env_shape = interp
            .program()
            .env_debug_shapes
            .get(tag)
            .or_else(|| interp.program().env_debug_shapes.get(bare))
            .or_else(|| {
                let want = bare
                    .replace("::{closure#", "::{closure_env#")
                    .replace("::{coroutine#", "::{coroutine_env#");
                interp
                    .program()
                    .env_debug_shapes
                    .values()
                    .find(|shape| shape.dwarf_name == want)
            });
        if let Some(shape) = env_shape {
            if shape.upvars.is_empty() {
                return vec![format!("type = struct {}", shape.dwarf_name)];
            }
            let mut lines = vec![format!("type = struct {} {{", shape.dwarf_name)];
            for field in &shape.upvars {
                let ty = dwarf_type_name(&field.ty);
                let ty = match ty.strip_prefix("&mut ") {
                    Some(inner) => format!("*mut {inner}"),
                    None => match ty.strip_prefix('&') {
                        Some(inner) => format!("*mut {inner}"),
                        None => ty,
                    },
                };
                lines.push(format!("    {ty},"));
            }
            lines.push("}".to_string());
            return lines;
        }
        // No recorded environment shape: a closure value whose
        // environment is empty has the bare struct form under
        // its environment name (the conversion records no
        // shape for a capture-less environment).
        if let Value::Closure { env, .. } = &ev.value {
            if env.is_empty() {
                let dwarf = bare
                    .replace("::{closure#", "::{closure_env#")
                    .replace("::{coroutine#", "::{coroutine_env#");
                if dwarf != bare {
                    return vec![format!("type = struct {dwarf}")];
                }
            }
        }
        if let Some(shape) = interp.program().adt_debug_shapes.get(tag) {
            if shape.kind == sparagmos_mir::AdtShapeKind::Struct {
                if let Some(vshape) = shape.variants.first() {
                    if vshape.fields.is_empty() {
                        return vec![format!("type = struct {name}")];
                    }
                    let mut lines = vec![format!("type = struct {name} {{")];
                    for field in &vshape.fields {
                        let fty = dwarf_type_name(&field.ty);
                        match &field.name {
                            Some(fname) => {
                                lines.push(format!("    {fty} {fname},"))
                            }
                            None => lines.push(format!("    {fty},")),
                        }
                    }
                    lines.push("}".to_string());
                    return lines;
                }
            }
        }
    }
    vec![format!("type = {name}")]
}

/// An IR type's name in the form a function declaration
/// listing prints (gdb `info functions`): the Rust spelling,
/// with an ADT named by its definition path.
fn ir_ty_name(ty: &Ty) -> String {
    use crate::ir::{FloatTy, IntTy, UintTy};
    match ty {
        Ty::Unit => "()".to_string(),
        Ty::Bool => "bool".to_string(),
        Ty::Int(k) => match k {
            IntTy::I8 => "i8",
            IntTy::I16 => "i16",
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
            IntTy::I128 => "i128",
            IntTy::Isize => "isize",
        }
        .to_string(),
        Ty::Uint(k) => match k {
            UintTy::U8 => "u8",
            UintTy::U16 => "u16",
            UintTy::U32 => "u32",
            UintTy::U64 => "u64",
            UintTy::U128 => "u128",
            UintTy::Usize => "usize",
        }
        .to_string(),
        Ty::Float(k) => match k {
            FloatTy::F16 => "f16",
            FloatTy::F32 => "f32",
            FloatTy::F64 => "f64",
            FloatTy::F128 => "f128",
        }
        .to_string(),
        Ty::Tuple(ts) => {
            let parts: Vec<String> = ts.iter().map(ir_ty_name).collect();
            if parts.len() == 1 {
                format!("({},)", parts[0])
            } else {
                format!("({})", parts.join(", "))
            }
        }
        Ty::Never => "!".to_string(),
        Ty::Adt(name) => dwarf_type_name(name),
        Ty::Ref(inner) => format!("&{}", ir_ty_name(inner)),
        Ty::Slice(elem) => format!("[{}]", ir_ty_name(elem)),
        Ty::Array(elem, n) => format!("[{}; {n}]", ir_ty_name(elem)),
        Ty::Str => "str".to_string(),
        Ty::FnPtr => "fn()".to_string(),
        Ty::FnDef(name) => name.clone(),
        Ty::Closure(name, _) => name.clone(),
        Ty::Dynamic(_) => "dyn".to_string(),
        Ty::Coroutine(name) => name.clone(),
        Ty::Foreign(name) => dwarf_type_name(name),
    }
}

/// A small regular-expression search over function names, in
/// the subset of gdb's pattern language the debuginfo suite
/// uses: literal characters, `.` (any character) and `*`
/// (repeat the previous atom). Unanchored, as gdb's `info
/// functions` pattern is.
fn fn_pattern_search(pattern: &str, text: &str) -> bool {
    #[derive(Clone, Copy)]
    enum Atom {
        Lit(char),
        Any,
    }
    let mut atoms: Vec<(Atom, bool)> = Vec::new();
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        match c {
            '*' => {
                if let Some(last) = atoms.last_mut() {
                    last.1 = true;
                }
            }
            '.' => atoms.push((Atom::Any, false)),
            '\\' => match chars.next() {
                Some(esc) => atoms.push((Atom::Lit(esc), false)),
                None => atoms.push((Atom::Lit('\\'), false)),
            },
            _ => atoms.push((Atom::Lit(c), false)),
        }
    }
    let text: Vec<char> = text.chars().collect();
    fn matches(atoms: &[(Atom, bool)], text: &[char]) -> bool {
        if atoms.is_empty() {
            return true;
        }
        let (atom, star) = atoms[0];
        if star {
            // Zero repetitions, or one and stay.
            if matches(&atoms[1..], text) {
                return true;
            }
            return !text.is_empty()
                && atom_matches(atom, text[0])
                && matches(atoms, &text[1..]);
        }
        !text.is_empty()
            && atom_matches(atom, text[0])
            && matches(&atoms[1..], &text[1..])
    }
    fn atom_matches(atom: Atom, c: char) -> bool {
        match atom {
            Atom::Lit(l) => l == c,
            Atom::Any => true,
        }
    }
    // Unanchored: a match may start at any offset, and need
    // not reach the end of the name.
    (0..=text.len()).any(|start| matches(&atoms, &text[start..]))
}

/// Render gdb's `info functions [-q] [pattern]`: one
/// declaration line per function of the program whose name
/// the pattern matches (every function without a pattern),
/// in name order. The `-q` form carries the `static fn`
/// prefix and the signature; the plain form is
/// `fn <name>();`.
///
/// Entries come from two sources. A converted body renders
/// from its own key and local types, as the interpreter has
/// always spelled them. The driver's display-name table
/// ([`crate::ir::Program::fn_display_names`]) names resolved
/// instances the way the compiler's debug info does —
/// generic arguments substituted, trait-impl methods under
/// their `{impl#N}` scope, closures with their single
/// environment-pointer parameter — and a table entry whose
/// name equals a body's rendered name takes the body's
/// place: the body's own signature derivation is the wrong
/// one there (a closure body's parameters are its captures,
/// not the environment pointer the debug info declares).
/// Table entries with no converted body are listed on their
/// own: the debug info names every instance the program
/// resolved, converted or not.
pub(super) fn render_info_functions(
    interp: &Interpreter,
    pattern: Option<&str>,
    quiet: bool,
) -> Vec<String> {
    let program = interp.program();
    let mut table: std::collections::HashMap<&str, &str> =
        std::collections::HashMap::new();
    for (display, signature, _) in &program.fn_display_names {
        table.insert(display.as_str(), signature.as_str());
    }
    // (rendered name, signature line, spellings the pattern
    // may match: the body key and the rendered name).
    let mut entries: Vec<(String, String, Vec<String>)> = Vec::new();
    for name in program.functions.keys() {
        let rendered = dwarf_type_name(name);
        let Some(body) = program.get(name) else {
            continue;
        };
        let signature = match table.remove(rendered.as_str()) {
            Some(sig) => format!("static fn {rendered}{sig};"),
            None => {
                // The declaration gdb's full-debuginfo listing
                // prints: the signature form, with the return
                // type omitted for the unit and never types
                // (both print as no return).
                let args: Vec<String> = (1..=body.arg_count)
                    .filter_map(|i| body.local_tys.get(i).map(ir_ty_name))
                    .collect();
                let ret = match body.local_tys.first() {
                    Some(Ty::Unit) | Some(Ty::Never) | None => String::new(),
                    Some(ty) => format!(" -> {}", ir_ty_name(ty)),
                };
                format!("static fn {rendered}({}){ret};", args.join(", "))
            }
        };
        entries.push((
            rendered.clone(),
            signature,
            vec![name.clone(), rendered],
        ));
    }
    // Table entries no converted body claimed.
    for (display, signature, _) in &program.fn_display_names {
        if table.contains_key(display.as_str()) {
            entries.push((
                display.clone(),
                format!("static fn {display}{signature};"),
                vec![display.clone()],
            ));
        }
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut lines = Vec::new();
    for (rendered, signature, spellings) in &entries {
        if let Some(pat) = pattern {
            if !spellings.iter().any(|s| fn_pattern_search(pat, s)) {
                continue;
            }
        }
        if quiet {
            lines.push(signature.clone());
        } else {
            // The plain listing's declaration form depends on
            // the debug info the binary carries: with full
            // debuginfo gdb prints the DWARF declaration
            // (`static fn ...`), with line-only info the
            // symbol-table form (`fn ...`). Both are candidate
            // lines, as elsewhere in the renderers.
            lines.push(format!("fn {rendered}();"));
            lines.push(format!("static fn {rendered}();"));
            lines.push(signature.clone());
        }
    }
    lines
}

/// Render gdb's `info locals` for the selected frame: the
/// variables in scope at the stop, one `name = value` line
/// each, or `No locals.` when none are.
pub(super) fn render_info_locals(
    interp: &Interpreter,
    ctl: &mut DebugControl,
    frames: &[Frame],
) -> Vec<String> {
    if ctl.hide_locals {
        // Limited debug info carries no variable information.
        return vec!["No locals.".to_string()];
    }
    let Some(frame) =
        frames.get(frames.len().saturating_sub(1 + ctl.selected_frame))
    else {
        return vec!["No locals.".to_string()];
    };
    let Some(body) = interp.program().get(&frame.func) else {
        return vec!["No locals.".to_string()];
    };
    // The lookup position is the selected frame's own (the
    // derivation `frame_position` makes): the stop position
    // belongs to the innermost frame, so resolving an outer
    // frame's variables against it scopes them by the callee's
    // line — a caller local declared after the callee's stop
    // line but before the call would be filtered out. The
    // position is installed for the whole listing, so the name
    // resolution inside the evaluations scopes consistently.
    let saved_pos = ctl.current_pos;
    if ctl.selected_frame != 0 {
        if let Some(pos) = frame_position(interp, frame) {
            ctl.current_pos = Some((pos.file, pos.line));
        }
    }
    let at_line = ctl.current_pos.map(|(_, line)| line).unwrap_or(0);
    let mut seen: Vec<String> = Vec::new();
    let mut lines = Vec::new();
    for entry in &body.debug_vars {
        if entry.lines.0 == 0 || entry.lines.0 > at_line {
            continue;
        }
        if seen.contains(&entry.name) {
            continue;
        }
        seen.push(entry.name.clone());
        if let Ok(ev) = evaluate_expr(interp, ctl, frames, &entry.name) {
            if let Some(gdb) =
                render_gdb(interp, &ev.value, ev.desc.as_ref(), ev.char, None, 0)
            {
                lines.push(format!("{} = {gdb}", entry.name));
            }
        }
    }
    let lines = if lines.is_empty() {
        vec!["No locals.".to_string()]
    } else {
        lines
    };
    ctl.current_pos = saved_pos;
    lines
}

