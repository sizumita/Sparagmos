//! lldb-form rendering: the compact summary form, expanded child lines and the environment (closure / coroutine) forms.

use crate::interp::{Frame, Interpreter};
use crate::ir::Ty;
use crate::value::Value;
use super::desc::{Desc, dwarf_type_name, path_text, render_vec_lldb, tag_base_name};
use super::evaluate::desc_scalar_width;
use super::queries::nonzero_scalar;
use super::render_gdb::{
    MAX_RENDER_DEPTH, UNION_FIELD_UNREADABLE, fat_base_text, nonzero_lldb_expanded,
    render_elements, render_gdb, str_place_desc, union_field_from_bytes, union_field_values,
};
use super::session::{DebugDialect, PrintFormat};

/// Render a value in lldb's compact form (the summary its
/// `frame variable` output shows): scalars as [`render_scalar`]
/// renders them, aggregates in brace form with no space around
/// the colon (`{x:10, y:23.5}`), tuples and arrays in tuple form
/// (`(0, 1, 2)`), an enum as its variant — named at the top
/// level, nameless when nested (`{a:1, b:2}`) — and a reference
/// as its address.
pub(super) fn render_lldb_compact(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
    top_char: bool,
    nested: bool,
    depth: usize,
) -> Option<String> {
    if depth > MAX_RENDER_DEPTH {
        return None;
    }
    if let Some((scalar, inner)) = nonzero_scalar(value, desc) {
        return render_lldb_compact(
            interp,
            &scalar,
            Some(&inner),
            top_char,
            nested,
            depth,
        );
    }
    match value {
        Value::Bool(_)
        | Value::Unit
        | Value::Int(_)
        | Value::Uint(_)
        | Value::Float(_)
        | Value::Float128(_) => render_scalar(
            value,
            None,
            top_char || matches!(desc, Some(Desc::Char)),
            None,
            scalar_desc_bit_width(desc),
            DebugDialect::Lldb,
        ),
        Value::Tuple(fields) => {
            let (descs, shared): (Vec<Desc>, Option<&Desc>) = match desc {
                Some(Desc::Tuple(ts)) => (ts.clone(), None),
                Some(Desc::Array(elem, _)) => (Vec::new(), Some(elem.as_ref())),
                _ => (Vec::new(), None),
            };
            let parts = render_elements(
                interp,
                fields,
                &descs,
                shared,
                DebugDialect::Lldb,
                depth,
                &[],
                false,
                200,
            )?;
            Some(format!("({})", parts.join(", ")))
        }
        Value::Enum { variant, fields } => {
            let Some(Desc::Adt(tag)) = desc else {
                return None;
            };
            // The std model's own aggregates, which the shape
            // table (the library's declared layout) cannot
            // describe: a `Vec` is its element list, and the
            // path family wraps its text in single-field layers.
            match tag_base_name(tag) {
                "Vec" => {
                    return render_vec_lldb(interp, tag, fields, depth);
                }
                "PathBuf" | "Path" | "OsString" | "OsStr" => {
                    return path_text(value)
                        .map(|text| format!("\"{}\"", escape_str(&text)));
                }
                _ => {}
            }
            let shape = interp.program().adt_debug_shapes.get(tag)?;
            render_adt_lldb_compact(interp, shape, *variant, fields, nested, depth)
        }
        Value::Ref(_) | Value::DynRef { .. } | Value::FnPtr(_) => {
            let addr = crate::interp::synthetic_address(value)?;
            Some(format!("0x{addr:x}"))
        }
        Value::FatRef { base, .. } => {
            // A fat reference to `str` whose base is a value
            // snapshot of the text renders as the text, as a
            // `str` value itself does below; any other fat
            // reference keeps the address form.
            if str_place_desc(desc) {
                if let Some(text) = fat_base_text(base) {
                    return Some(format!("\"{}\"", escape_str(&text)));
                }
            }
            let addr = crate::interp::synthetic_address(value)?;
            Some(format!("0x{addr:x}"))
        }
        Value::Closure { .. } | Value::Coroutine { .. } => {
            render_env_lldb_compact(interp, None, value, desc, depth)
        }
        Value::Str(s) => Some(format!("\"{}\"", escape_str(s))),
        Value::Uninit => match desc {
            Some(Desc::Int(_)) | Some(Desc::Float(_)) => Some("0".to_string()),
            Some(Desc::Bool) => Some("false".to_string()),
            _ => None,
        },
        Value::UnionBytes { bytes } => {
            // A union in its byte form, as in the gdb form:
            // every declared field is read out of the storage
            // bytes (see `union_field_from_bytes`).
            let Some(Desc::Adt(tag)) = desc else {
                return None;
            };
            let shape = interp.program().adt_debug_shapes.get(tag)?;
            if !matches!(shape.kind, sparagmos_mir::AdtShapeKind::Union) {
                return None;
            }
            let vshape = shape.variants.first()?;
            let parts = union_parts_lldb(interp, vshape, depth, |i| {
                union_field_from_bytes(&vshape.fields, bytes, i)
            })?;
            Some(format!("{{{}}}", parts.join(", ")))
        }
        _ => None,
    }
}

/// The `name:value` parts of a union's fields in lldb's
/// compact form, each field's value produced by `value_of`
/// (the storage reinterpretation behind the aggregate and
/// byte forms); a field with no producible or renderable
/// value degrades in place, as in the gdb form (see
/// [`UNION_FIELD_UNREADABLE`]).
fn union_parts_lldb(
    interp: &Interpreter,
    vshape: &sparagmos_mir::AdtDebugVariant,
    depth: usize,
    value_of: impl Fn(usize) -> Option<Value>,
) -> Option<Vec<String>> {
    let mut parts = Vec::with_capacity(vshape.fields.len());
    for (i, field) in vshape.fields.iter().enumerate() {
        let name = field.name.as_deref()?;
        let d = Desc::from_tag(&field.ty);
        let r = value_of(i)
            .and_then(|v| render_lldb_compact(interp, &v, Some(&d), false, true, depth + 1))
            .unwrap_or_else(|| UNION_FIELD_UNREADABLE.to_string());
        parts.push(format!("{name}:{r}"));
    }
    Some(parts)
}

/// Render one ADT value in lldb's compact form (see
/// [`render_lldb_compact`]).
fn render_adt_lldb_compact(
    interp: &Interpreter,
    shape: &sparagmos_mir::AdtDebugShape,
    variant: u32,
    fields: &[Value],
    nested: bool,
    depth: usize,
) -> Option<String> {
    let vshape = shape.variants.get(variant as usize)?;
    if matches!(shape.kind, sparagmos_mir::AdtShapeKind::Union) {
        let values = union_field_values(&vshape.fields, fields);
        let parts = union_parts_lldb(interp, vshape, depth, |i| {
            values.get(i).cloned().flatten()
        })?;
        return Some(format!("{{{}}}", parts.join(", ")));
    }
    if vshape.fields.len() != fields.len() {
        return None;
    }
    let descs: Vec<Desc> = vshape
        .fields
        .iter()
        .map(|f| Desc::from_tag(&f.ty))
        .collect();
    let named = vshape.fields.iter().all(|f| f.name.is_some());
    let brace_parts = |interp: &Interpreter| -> Option<Vec<String>> {
        let mut parts = Vec::with_capacity(fields.len());
        for (i, field) in fields.iter().enumerate() {
            let v = render_lldb_compact(
                interp,
                field,
                Some(&descs[i]),
                false,
                true,
                depth + 1,
            )?;
            if named {
                let name = vshape.fields[i].name.as_deref()?;
                parts.push(format!("{name}:{v}"));
            } else {
                parts.push(format!("{i}:{v}"));
            }
        }
        Some(parts)
    };
    match shape.kind {
        sparagmos_mir::AdtShapeKind::Struct => {
            let parts = brace_parts(interp)?;
            Some(format!("{{{}}}", parts.join(", ")))
        }
        sparagmos_mir::AdtShapeKind::Enum => {
            if fields.is_empty() {
                return Some(vshape.name.clone());
            }
            if named {
                let parts = brace_parts(interp)?;
                let braces = format!("{{{}}}", parts.join(", "));
                if nested {
                    Some(braces)
                } else {
                    Some(format!("{}{}", vshape.name, braces))
                }
            } else {
                let parts = render_elements(
                    interp,
                    fields,
                    &descs,
                    None,
                    DebugDialect::Lldb,
                    depth,
                    &[],
                    false,
                    200,
                )?;
                if nested {
                    Some(format!("({})", parts.join(", ")))
                } else {
                    Some(format!("{}({})", vshape.name, parts.join(", ")))
                }
            }
        }
        sparagmos_mir::AdtShapeKind::Union => None,
    }
}

/// Render a field value inside lldb's expanded children block:
/// a nested aggregate expands in turn when it has fields of its
/// own, and anything else takes the compact form.
fn render_lldb_sub(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
    depth: usize,
) -> Option<String> {
    if matches!(value, Value::Tuple(_) | Value::Enum { .. }) {
        render_lldb_expanded(interp, value, desc, depth)
            .or_else(|| render_lldb_compact(interp, value, desc, false, true, depth))
    } else {
        render_lldb_compact(interp, value, desc, false, true, depth)
    }
}

/// Render an aggregate value in lldb's expanded children form:
/// `{ f = v f = v }`, each field's value as [`render_lldb_sub`]
/// renders it. Tuple and array elements are named by their
/// index (`0 = v`, `[0] = v`). `None` for a value with no fields
/// to expand.
fn render_lldb_expanded(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
    depth: usize,
) -> Option<String> {
    lldb_expanded_pairs(interp, value, desc, depth)
        .map(|pairs| format!("{{ {} }}", pairs.join(" ")))
}

/// The `name = value` pairs of an aggregate's expanded
/// children form (see [`render_lldb_expanded`]), unjoined:
/// the parenthesised comma-joined form lldb prints for a
/// sequence's children is built from the same pairs.
fn lldb_expanded_pairs(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
    depth: usize,
) -> Option<Vec<String>> {
    if depth > MAX_RENDER_DEPTH {
        return None;
    }
    match value {
        Value::Enum { variant, fields } => {
            let Some(Desc::Adt(tag)) = desc else {
                return None;
            };
            let shape = interp.program().adt_debug_shapes.get(tag)?;
            let vshape = shape.variants.get(*variant as usize)?;
            if vshape.fields.len() != fields.len() || fields.is_empty() {
                return None;
            }
            let mut pairs = Vec::with_capacity(fields.len());
            for (i, field) in fields.iter().enumerate() {
                let name = match &vshape.fields[i].name {
                    Some(name) => name.clone(),
                    None => i.to_string(),
                };
                let d = Desc::from_tag(&vshape.fields[i].ty);
                let v = render_lldb_sub(interp, field, Some(&d), depth + 1)?;
                pairs.push(format!("{name} = {v}"));
            }
            Some(pairs)
        }
        Value::Tuple(fields) => {
            if fields.is_empty() {
                return None;
            }
            let is_array = matches!(
                desc,
                Some(Desc::Array(..)) | Some(Desc::Slice(..))
            );
            let mut pairs = Vec::with_capacity(fields.len());
            for (i, field) in fields.iter().enumerate() {
                let d = match desc {
                    Some(Desc::Tuple(ts)) => ts.get(i),
                    Some(Desc::Array(elem, _)) | Some(Desc::Slice(elem)) => {
                        Some(elem.as_ref())
                    }
                    _ => None,
                };
                let name = if is_array {
                    format!("[{i}]")
                } else {
                    i.to_string()
                };
                let v = render_lldb_sub(interp, field, d, depth + 1)?;
                pairs.push(format!("{name} = {v}"));
            }
            Some(pairs)
        }
        _ => None,
    }
}

/// The (head name, fields) of a named-field aggregate value:
/// the head is the struct's base name or the enum's variant
/// name, and each field carries its gdb-form and lldb-compact
/// renderings. `None` when the value is not a named-field
/// aggregate.
fn named_field_values(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
) -> Option<(String, Vec<(String, String, String)>)> {
    let Value::Enum { variant, fields } = value else {
        return None;
    };
    let Some(Desc::Adt(tag)) = desc else {
        return None;
    };
    let shape = interp.program().adt_debug_shapes.get(tag)?;
    let vshape = shape.variants.get(*variant as usize)?;
    if vshape.fields.len() != fields.len() || fields.is_empty() {
        return None;
    }
    if !vshape.fields.iter().all(|f| f.name.is_some()) {
        return None;
    }
    let head = match shape.kind {
        sparagmos_mir::AdtShapeKind::Struct => tag
            .split('<')
            .next()?
            .rsplit("::")
            .next()?
            .to_string(),
        sparagmos_mir::AdtShapeKind::Enum => vshape.name.clone(),
        sparagmos_mir::AdtShapeKind::Union => return None,
    };
    let mut pairs = Vec::with_capacity(fields.len());
    for (i, field) in fields.iter().enumerate() {
        let d = Desc::from_tag(&vshape.fields[i].ty);
        let gdb = render_gdb(interp, field, Some(&d), false, None, 1)?;
        let compact = render_lldb_compact(interp, field, Some(&d), false, true, 1)?;
        pairs.push((vshape.fields[i].name.clone()?, gdb, compact));
    }
    Some((head, pairs))
}

/// The (field, value) pairs a closure or coroutine value
/// renders with, in printed order, with the instance's recorded
/// environment shape: for a closure, its captures in environment
/// order; for a coroutine, the current state's saved fields
/// followed by its captures (the order the compiler's debuginfo
/// lays a state out in). `None` when the value's instance has no
/// recorded shape or a field's storage is absent.
fn env_render_fields<'i, 'v>(
    interp: &'i Interpreter,
    value: &'v Value,
    desc: Option<&Desc>,
) -> Option<(
    &'i sparagmos_mir::EnvDebugShape,
    Vec<(&'i sparagmos_mir::AdtDebugField, &'v Value)>,
)> {
    let Some(Desc::Adt(tag)) = desc else {
        return None;
    };
    let shape: &'i sparagmos_mir::EnvDebugShape =
        interp.program().env_debug_shapes.get(tag)?;
    let mut pairs = Vec::new();
    match value {
        Value::Closure { env, .. } => {
            if env.len() != shape.upvars.len() {
                return None;
            }
            pairs.extend(shape.upvars.iter().zip(env.iter()));
        }
        Value::Coroutine {
            variant,
            fields,
            saved,
        } => {
            let vshape = shape.variants.get(*variant as usize)?;
            if vshape.fields.len() != vshape.slots.len() {
                return None;
            }
            for (field, slot) in vshape.fields.iter().zip(vshape.slots.iter()) {
                let stored = if shape.slotted {
                    fields.get(*slot as usize)?
                } else {
                    saved.get(*variant as usize)?.get(*slot as usize)?
                };
                pairs.push((field, stored));
            }
            if fields.len() < shape.upvars.len() {
                return None;
            }
            pairs.extend(shape.upvars.iter().zip(fields.iter()));
        }
        _ => return None,
    }
    Some((shape, pairs))
}

/// Render one closure or coroutine value in gdb form:
/// `path::{closure_env#0} {name: value, ...}` for a closure,
/// `path::{coroutine_env#0}::State{name: value, ...}` for a
/// coroutine (see [`env_render_fields`] for the field order).
pub(super) fn render_env_gdb(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
    depth: usize,
) -> Option<String> {
    let (shape, pairs) = env_render_fields(interp, value, desc)?;
    let mut parts = Vec::with_capacity(pairs.len());
    for (field, stored) in &pairs {
        let name = field.name.as_deref()?;
        let d = Desc::from_tag(&field.ty);
        let rendered = render_gdb(interp, stored, Some(&d), false, None, depth + 1)?;
        parts.push(format!("{name}: {rendered}"));
    }
    match value {
        Value::Closure { .. } => {
            if parts.is_empty() {
                Some(shape.dwarf_name.clone())
            } else {
                Some(format!("{} {{{}}}", shape.dwarf_name, parts.join(", ")))
            }
        }
        Value::Coroutine { variant, .. } => {
            let vname = &shape.variants.get(*variant as usize)?.name;
            let prefix = format!("{}::{vname}", shape.dwarf_name);
            if parts.is_empty() {
                Some(prefix)
            } else {
                Some(format!("{prefix}{{{}}}", parts.join(", ")))
            }
        }
        _ => None,
    }
}

/// Render one closure or coroutine value in lldb's compact
/// form: `{name:value, ...}` over the same fields as
/// [`render_env_gdb`]. At the top level of a print (`frames`
/// given), a captured field that is a reference to an aggregate
/// renders as its pointee - the form lldb's own display gives a
/// reference member - while a reference to a scalar or to
/// another reference stays the address; nested renders
/// (`frames` absent) keep the address form throughout.
fn render_env_lldb_compact(
    interp: &Interpreter,
    frames: Option<&[Frame]>,
    value: &Value,
    desc: Option<&Desc>,
    depth: usize,
) -> Option<String> {
    let (_shape, pairs) = env_render_fields(interp, value, desc)?;
    let mut parts = Vec::with_capacity(pairs.len());
    for (field, stored) in &pairs {
        let name = field.name.as_deref()?;
        let d = Desc::from_tag(&field.ty);
        let rendered = render_env_lldb_field(interp, frames, stored, &d, depth + 1)?;
        parts.push(format!("{name}:{rendered}"));
    }
    Some(format!("{{{}}}", parts.join(", ")))
}

/// Render one environment field's value in lldb's compact form
/// (see [`render_env_lldb_compact`] for the reference rule).
fn render_env_lldb_field(
    interp: &Interpreter,
    frames: Option<&[Frame]>,
    value: &Value,
    desc: &Desc,
    depth: usize,
) -> Option<String> {
    if let (Some(frames), Value::Ref(_), Desc::Ref(inner) | Desc::RefMut(inner)) =
        (frames, value, desc)
    {
        if matches!(inner.as_ref(), Desc::Adt(_)) {
            if let Ok(resolved) = interp.deref_value(frames, value) {
                if let Ok(pointee) = interp.read_resolved(frames, &resolved) {
                    return render_lldb_compact(
                        interp,
                        &pointee,
                        Some(inner.as_ref()),
                        false,
                        true,
                        depth,
                    );
                }
            }
        }
    }
    render_lldb_compact(interp, value, Some(desc), false, true, depth)
}

/// The transcript lines of one lldb print of a closure or
/// coroutine value: the single `(Type) name = <compact>` line,
/// with the environment type's DWARF name.
pub(super) fn render_env_lldb_lines(
    interp: &Interpreter,
    frames: &[Frame],
    value: &Value,
    desc: Option<&Desc>,
    expr: &str,
) -> Vec<String> {
    let Some(compact) = render_env_lldb_compact(interp, Some(frames), value, desc, 0)
    else {
        return Vec::new();
    };
    let Some(Desc::Adt(tag)) = desc else {
        return vec![format!("= {compact}")];
    };
    let Some(shape) = interp.program().env_debug_shapes.get(tag) else {
        return vec![format!("= {compact}")];
    };
    vec![format!(
        "({}) {expr} = {compact}",
        lldb_type_name(&shape.dwarf_name)
    )]
}

/// The transcript lines of one lldb print. A scalar keeps the
/// session's established form (`= <value>`). An aggregate value
/// gets the lines lldb's own output forms suggest, in order: the
/// `(Type) name = <compact>` summary line; the same line with
/// the expanded children block appended; and, for a named-field
/// aggregate, the `Name { f: v, ... }` line and the
/// comma-separated `{ f = v, ... }` line. Check lines match
/// these as fragments in order, so the several forms coexist:
/// each print contributes every form its value has.
pub(super) fn render_lldb_lines(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
    top_char: bool,
    expr: &str,
) -> Vec<String> {
    let Some(compact) = render_lldb_compact(interp, value, desc, top_char, false, 0)
    else {
        return Vec::new();
    };
    if !matches!(value, Value::Tuple(_) | Value::Enum { .. }) {
        return vec![format!("= {compact}")];
    }
    let mut lines = Vec::new();
    if let Some(Desc::Adt(tag)) = desc {
        let first = format!(
            "({}) {expr} = {compact}",
            lldb_type_name(&dwarf_type_name(tag))
        );
        lines.push(first.clone());
        if let Some(expanded) = nonzero_lldb_expanded(interp, value, desc) {
            lines.push(format!("{first} {expanded}"));
        }
        if let Some(expanded) = render_lldb_expanded(interp, value, desc, 0) {
            lines.push(format!("{first} {expanded}"));
        }
        if let Some(pairs) = lldb_expanded_pairs(interp, value, desc, 0) {
            lines.push(format!("{first} ({})", pairs.join(", ")));
        }
        if let Some((head, pairs)) = named_field_values(interp, value, desc) {
            let colon = pairs
                .iter()
                .map(|(name, gdb, _)| format!("{name}: {gdb}"))
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("{head} {{ {colon} }}"));
            let eq = pairs
                .iter()
                .map(|(name, _, compact)| format!("{name} = {compact}"))
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("{{ {eq} }}"));
        }
    } else {
        lines.push(format!("= {compact}"));
        if let Some(expanded) = render_lldb_expanded(interp, value, desc, 0) {
            lines.push(format!("= {compact} {expanded}"));
        }
        if let Some(pairs) = lldb_expanded_pairs(interp, value, desc, 0) {
            lines.push(format!("= {compact} ({})", pairs.join(", ")));
        }
    }
    lines
}

/// An ADT instance tag as lldb renders the type in its
/// `(Type) name = value` output: scalar generic arguments take
/// the C names of the scalar types (`i32` as `int`, `f64` as
/// `double`), nested closing brackets are separated by a space
/// (`> >`), and tuple arguments keep their Rust spelling.
pub(super) fn lldb_type_name(tag: &str) -> String {
    fn scalar_name(word: &str) -> &str {
        match word {
            "i8" => "signed char",
            "i16" => "short",
            "i32" => "int",
            "i64" => "long",
            "isize" => "long",
            "u8" => "unsigned char",
            "u16" => "unsigned short",
            "u32" => "unsigned int",
            "u64" => "unsigned long",
            "usize" => "unsigned long",
            "f32" => "float",
            "f64" => "double",
            _ => word,
        }
    }
    let bytes = tag.as_bytes();
    let mut out = String::with_capacity(tag.len());
    let mut angle = 0usize;
    let mut paren = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => {
                angle += 1;
                out.push('<');
                i += 1;
            }
            b'>' => {
                angle = angle.saturating_sub(1);
                if out.ends_with('>') {
                    out.push(' ');
                }
                out.push('>');
                i += 1;
            }
            b'(' => {
                paren += 1;
                out.push('(');
                i += 1;
            }
            b')' => {
                paren = paren.saturating_sub(1);
                out.push(')');
                i += 1;
            }
            c if angle > 0 && paren == 0 && (c.is_ascii_alphabetic() || c == b'_') => {
                let start = i;
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
                {
                    i += 1;
                }
                out.push_str(scalar_name(&tag[start..i]));
            }
            c => {
                out.push(c as char);
                i += 1;
            }
        }
    }
    out
}

/// Escape a string value for lldb's quoted string summary.
pub(super) fn escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out
}

/// Render a scalar value the way gdb prints it: integers in
/// decimal (signedness from the declared type when the value's
/// own form does not carry it), booleans as `true`/`false`,
/// floats with Rust's `{}` (which agrees with gdb's rendering of
/// the simple decimals the checks use), a char as
/// `<codepoint> '<char>'`, unit as `()`.
/// The bit width a scalar descriptor's layout occupies,
/// for the width-sensitive print formats (`print/u` reads a
/// value at its own width).
pub(super) fn scalar_desc_bit_width(desc: Option<&Desc>) -> Option<u32> {
    desc.and_then(desc_scalar_width)
        .map(|bytes| (bytes * 8) as u32)
}

pub(super) fn render_scalar(
    value: &Value,
    ty: Option<&Ty>,
    char_hint: bool,
    format: Option<PrintFormat>,
    width: Option<u32>,
    dialect: DebugDialect,
) -> Option<String> {
    match value {
        Value::Bool(b) => Some(b.to_string()),
        Value::Unit => Some("()".to_string()),
        Value::Int(v) => Some(format_int(*v as u128, true, format, width)),
        Value::Uint(v) => {
            if char_hint && format.is_none() {
                let c = char::from_u32(*v as u32)?;
                return Some(match dialect {
                    DebugDialect::Gdb => format!("{} '{}'", v, escape_char(c)),
                    // lldb's char summary: the code point in
                    // `U+0xXXXXXXXX` form, then the character
                    // itself.
                    DebugDialect::Lldb => {
                        format!("U+0x{:08X} U'{}'", v, escape_char(c))
                    }
                });
            }
            let signed = matches!(format, Some(PrintFormat::Decimal));
            Some(format_int(*v, signed, format, width))
        }
        Value::Float(v) => Some(format_float(*v)),
        Value::Float128(bits) => {
            // Rendered through f64 when the value is exactly
            // representable there (the simple decimals the
            // checks use are); a value that is not keeps the
            // f64 rounding of it, which a check written against
            // the full binary128 rendering will not match.
            let v = f64::from_bits(crate::f128::to_f64_bits(*bits));
            Some(format_float(v))
        }
        _ => None,
    }
}

fn format_int(bits: u128, signed: bool, format: Option<PrintFormat>, width: Option<u32>) -> String {
    match format {
        Some(PrintFormat::Hex) => format!("{bits:#x}"),
        Some(PrintFormat::Binary) => format!("{bits:#b}"),
        Some(PrintFormat::Unsigned) => {
            // The unsigned reading of the stored bits at the
            // value's own width: a sign-extended negative
            // value masks back to its two's-complement form.
            let masked = match width {
                Some(w) if w < 128 => bits & ((1u128 << w) - 1),
                _ => bits,
            };
            format!("{masked}")
        }
        _ => {
            if signed {
                // Values are stored sign-extended to 128 bits, so
                // the i128 reading is the value.
                format!("{}", bits as i128)
            } else {
                format!("{bits}")
            }
        }
    }
}

fn format_float(v: f64) -> String {
    // Rust's Display prints the shortest form that round-trips,
    // which for the simple decimals the checks use is the same
    // text gdb prints (1.5, 2.5, -10.5; an integral float prints
    // without a fraction in both).
    format!("{v}")
}

/// gdb's rendering of one character inside quotes: the printable
/// character itself, the standard escapes for the control
/// characters it names.
pub(super) fn escape_char(c: char) -> String {
    match c {
        '\n' => "\\n".to_string(),
        '\t' => "\\t".to_string(),
        '\r' => "\\r".to_string(),
        '\\' => "\\\\".to_string(),
        '\'' => "\\'".to_string(),
        c if c.is_control() => format!("\\u{{{:x}}}", c as u32),
        c => c.to_string(),
    }
}
