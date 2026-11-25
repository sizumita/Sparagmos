//! Print-expression evaluation: resolving an expression against the stopped frame's scopes to a value and its descriptor.

use crate::interp::{Frame, Interpreter};
use crate::value::{RefProj, Value};
use super::desc::{Desc, field_desc, path_text, smart_pointer_pointee, tag_base_name};
use super::parse::{ExprStep, ParsedExpr, parse_expr};
use super::printers::{render_lldb_provider_line, render_pretty_gdb};
use super::queries::{desc_type_name, expr_type_name, lldb_command_format_scalar, lldb_format_scalar, pointer_pointee};
use super::render_gdb::{render_gdb, union_projected_field};
use super::render_lldb::{lldb_type_name, render_env_lldb_lines, render_lldb_lines};
use super::session::{DebugControl, DebugDialect, DebugPrint};

/// Render one print command against the selected frame, in the
/// session dialect's form: gdb's single `$N = <value>` line, or
/// lldb's line(s) for the value (see [`render_lldb_lines`]). A
/// value a dialect cannot render stays an explicit error line in
/// that dialect.
pub(super) fn render_print(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    print: &DebugPrint,
) -> Vec<String> {
    let err_line = |e: &str| match ctl.dialect {
        DebugDialect::Gdb => {
            vec![format!("${} = <error: {e}>", ctl.print_count + 1)]
        }
        DebugDialect::Lldb => vec![format!("= <error: {e}>")],
    };
    match evaluate_print(interp, ctl, frames, print) {
        Ok(evaluated) => match ctl.dialect {
            DebugDialect::Gdb => match evaluated.gdb {
                Some(gdb) => vec![format!("${} = {gdb}", ctl.print_count + 1)],
                None => err_line(&format!(
                    "`{}` could not be rendered",
                    print.expr
                )),
            },
            DebugDialect::Lldb => {
                if evaluated.lldb.is_empty() {
                    err_line(&format!("`{}` could not be rendered", print.expr))
                } else {
                    evaluated.lldb
                }
            }
        },
        Err(e) => err_line(&e),
    }
}

/// The outcome of evaluating one print expression: the value's
/// gdb-form rendering and its lldb-form transcript lines. Either
/// may be unavailable — a form the session cannot render stays
/// an explicit error in that dialect only.
pub(super) struct EvaluatedPrint {
    pub(super) gdb: Option<String>,
    pub(super) lldb: Vec<String>,
}

/// A print expression evaluated to a value: the value itself,
/// its rendering descriptor, its `char` fact, and its declared
/// type tag when the conversion recorded one (the form the
/// pretty printers key on and the type queries print).
pub(super) struct EvaluatedValue {
    pub(super) value: Value,
    pub(super) desc: Option<Desc>,
    pub(super) char: bool,
    pub(super) tag: Option<String>,
    /// The type a trailing cast step spelled, pointer layer
    /// included, when the cast's materialized view carries a
    /// descriptor of a different form: a cast to an array
    /// pointer views the pointee's elements under the array
    /// descriptor, but the expression's type keeps the pointer
    /// layer until a dereference consumes it.
    pub(super) cast_type: Option<Desc>,
    /// A scalar pointee reinterpretation a cast step recorded
    /// but has not yet applied: the cast changed a pointer's
    /// pointee type while the target storage still holds the
    /// source type's bytes, and the dereference that consumes
    /// the pointer re-reads those bytes through the target
    /// layout. `(source, target)`.
    pub(super) pending_cast: Option<(Desc, Desc)>,
    /// An array/slice view a cast to an array or slice
    /// pointer spelled but has not yet materialized: the
    /// cast keeps the pointer value (printing the cast
    /// renders the pointer), and the step that consumes the
    /// pointer -- a dereference or a projection --
    /// materializes the view first. The cast's spelled
    /// pointer descriptor, and the value's descriptor
    /// before the cast: an element-changing cast reinterprets
    /// the storage through the source element type, which
    /// the spelled target no longer records.
    pub(super) pending_array_view: Option<(Desc, Option<Desc>)>,
}

/// Resolve a print expression's base name: a variable of the
/// selected frame's body (as recorded in its debug variable
/// table), or — for a name no local carries, and for any
/// path-qualified name — a static of the program.
pub(super) fn resolve_base(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    expr: &ParsedExpr,
) -> Result<EvaluatedValue, String> {
    // A path-qualified name can only name a static: debug
    // variable entries carry plain identifiers.
    if !expr.base.contains("::") && !frames.is_empty() {
        if let Some(local) = resolve_local_base(interp, ctl, frames, expr)? {
            return Ok(local);
        }
    }
    if let Some(statik) = resolve_static_base(interp, ctl, frames, &expr.base) {
        return Ok(statik);
    }
    Err(format!("no variable `{}` in scope", expr.base))
}

/// Resolve a print expression's base name against the selected
/// frame's debug variable table. `Ok(None)` when the frame's
/// body records no variable of that name (the caller then
/// tries the statics).
fn resolve_local_base(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    expr: &ParsedExpr,
) -> Result<Option<EvaluatedValue>, String> {
    let frame = frames
        .get(frames.len().saturating_sub(1 + ctl.selected_frame))
        .ok_or_else(|| format!("frame {} is not live", ctl.selected_frame))?;
    let body = interp
        .program()
        .get(&frame.func)
        .ok_or_else(|| format!("no body for frame `{}`", frame.func))?;
    // Resolve the name the way a debugger does: among the
    // body's entries of that name, prefer the declaration whose
    // recorded span contains the current line (the smallest such
    // extent, as the innermost scope); the compiler records most
    // entries with a point span at the declaration, so the
    // general rule is the latest declaration at or before the
    // current line — the declaration in effect at the stop. With
    // neither, the last declaration stands.
    let entry = {
        let named = body
            .debug_vars
            .iter()
            .filter(|e| e.name == expr.base)
            .collect::<Vec<_>>();
        let at_line = ctl.current_pos.map(|(_, line)| line).unwrap_or(0);
        let containing = named
            .iter()
            .filter(|e| e.lines.0 != 0 && e.lines.0 <= at_line && at_line <= e.lines.1)
            .min_by_key(|e| e.lines.1 - e.lines.0);
        let declared = named
            .iter()
            .filter(|e| e.lines.0 != 0 && e.lines.0 <= at_line)
            .max_by_key(|e| e.lines.0);
        containing
            .or_else(|| declared)
            .copied()
            .or_else(|| named.last().copied())
    };
    let Some(entry) = entry else {
        return Ok(None);
    };
    let (source, char_hint) = (&entry.source, entry.is_char);
    // The value's type, as a rendering descriptor: the variable
    // entry's recorded type tag when the conversion supplied one
    // (it carries the full instance name, generic arguments
    // included), else — for a plain local — the frame body's
    // local type, which names an ADT by its definition only.
    let (value, desc): (Value, Option<Desc>) = match source {
        crate::ir::DebugVar::Place(local, proj) => {
            let value = frame
                .locals
                .get(*local)
                .cloned()
                .ok_or_else(|| format!("variable `{}` has no storage", expr.base))?;
            let desc = entry
                .ty
                .as_ref()
                .map(|tag| Desc::from_tag(tag))
                .or_else(|| {
                    if proj.is_empty() {
                        body.local_tys.get(*local).map(Desc::from_ir)
                    } else {
                        None
                    }
                });
            if proj.is_empty() {
                (value, desc)
            } else {
                (
                    crate::interp::project_value(value, proj, "debugger", &expr.base)
                        .map_err(|e| e.to_string())?,
                    desc,
                )
            }
        }
        crate::ir::DebugVar::Constant(constant) => (
            interp.eval_constant(constant).map_err(|e| e.to_string())?,
            entry.ty.as_ref().map(|tag| Desc::from_tag(tag)),
        ),
    };
    Ok(Some(EvaluatedValue {
        value,
        desc,
        // Whether the value (or, for a reference variable, its
        // pointee) is a `char` at the source level: the IR type
        // system has no char (it is a u32), so the conversion
        // recorded it in the debug variable table.
        char: char_hint,
        tag: entry.ty.clone(),
        cast_type: None,
        pending_cast: None,
        pending_array_view: None,
    }))
}

/// Resolve a print expression's base name against the
/// program's statics: the name is a static key (a full item
/// path), or the last segment of one (a plain name, as gdb
/// resolves a static from any scope). The value is the
/// static's current storage content — evaluating its
/// `StaticRef` constant seeds the storage from the
/// initializer on first touch, so a print before the program
/// has touched the static reads the initial value, and a
/// print after a write reads the written one.
fn resolve_static_base(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    base: &str,
) -> Option<EvaluatedValue> {
    let program = interp.program();
    let mut exact: Option<&String> = None;
    let mut by_segment: Option<&String> = None;
    for key in program.static_debug_tags.keys() {
        let stripped = key.split('#').next().unwrap_or(key);
        if stripped == base {
            exact = Some(key);
            break;
        }
        if !base.contains("::") && stripped.rsplit("::").next() == Some(base) {
            // Prefer the shortest key among last-segment
            // matches (the least-qualified reading).
            match by_segment {
                Some(cur) if cur.len() <= key.len() => {}
                _ => by_segment = Some(key),
            }
        }
    }
    let key = exact.or(by_segment)?;
    if !ctl.static_seeds.contains_key(key) {
        return None;
    }
    let reference = interp
        .eval_constant(&crate::ir::Constant::StaticRef {
            name: key.clone(),
            projection: Vec::new(),
            len: None,
        })
        .ok()?;
    let resolved = interp.deref_value(frames, &reference).ok()?;
    let value = interp.read_resolved(frames, &resolved).ok()?;
    let tag = program.static_debug_tags.get(key).cloned();
    let desc = tag
        .as_deref()
        .map(Desc::from_tag)
        .or_else(|| program.static_tys.get(key).map(Desc::from_ir));
    let char = matches!(desc, Some(Desc::Char));
    Some(EvaluatedValue {
        value,
        desc,
        char,
        tag,
        cast_type: None,
        pending_cast: None,
        pending_array_view: None,
    })
}

/// Follow a reference or smart-pointer value to its pointee:
/// the (pointee, pointee descriptor) when the value is one, the
/// value itself otherwise. Field and index steps apply this
/// first, as a debugger's field access dereferences through
/// references and smart pointers implicitly.
fn deref_if_pointer(
    interp: &Interpreter,
    frames: &[Frame],
    value: Value,
    desc: Option<Desc>,
) -> Result<(Value, Option<Desc>), String> {
    let is_pointer = matches!(
        value,
        Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. }
    ) || matches!(&desc, Some(Desc::Adt(tag)) if smart_pointer_pointee(tag).is_some());
    if !is_pointer {
        return Ok((value, desc));
    }
    let resolved = interp
        .deref_value(frames, &value)
        .map_err(|e| e.to_string())?;
    let pointee = interp
        .read_resolved(frames, &resolved)
        .map_err(|e| e.to_string())?;
    let next = match desc {
        Some(
            Desc::Ref(inner) | Desc::RefMut(inner) | Desc::PtrConst(inner) | Desc::PtrMut(inner),
        ) => {
            // A pointer whose pointee the value model holds as
            // a sequence aggregate (a slice's base pointer: its
            // deref yields the whole element tuple) describes
            // the sequence, not the single element its type
            // names -- the descriptor follows the value, so an
            // index step through it reads the element descriptor
            // from the sequence form.
            if matches!(pointee, Value::Tuple(_))
                && !matches!(
                    inner.as_ref(),
                    Desc::Slice(_) | Desc::Array(..) | Desc::Tuple(_)
                )
            {
                Some(Desc::Slice(inner))
            } else {
                Some(*inner)
            }
        }
        Some(Desc::Adt(tag)) => {
            smart_pointer_pointee(&tag).map(|t| Desc::from_tag(&t))
        }
        other => other,
    };
    Ok((pointee, next))
}

/// The length a `.length` step reads: a fat pointer's element
/// count, a string's byte count, a stored sequence's element
/// count. gdb spells a slice's and a `String`'s length this
/// way (the fat pointer's and the vector's length field).
fn sequence_length(
    interp: &Interpreter,
    frames: &[Frame],
    value: &Value,
    desc: Option<&Desc>,
) -> Result<u128, String> {
    match value {
        Value::FatRef { len, .. } => Ok(*len),
        Value::Str(s) => Ok(s.len() as u128),
        Value::Bytes(b) => Ok(b.len() as u128),
        Value::SparseVec { len, .. } => Ok(*len),
        Value::Tuple(items) => Ok(items.len() as u128),
        Value::Ref(_) | Value::DynRef { .. } => {
            let resolved = interp
                .deref_value(frames, value)
                .map_err(|e| e.to_string())?;
            let pointee = interp
                .read_resolved(frames, &resolved)
                .map_err(|e| e.to_string())?;
            sequence_length(interp, frames, &pointee, None)
        }
        Value::Enum { fields, .. } => {
            // A `Vec` value in this model is its element
            // aggregate; a `String` value wraps its text (a
            // plain string value is handled above, so what
            // remains is the wrapped form `path_text` reads).
            if let Some(tag) = desc.and_then(|d| match d {
                Desc::Adt(tag) => Some(tag),
                _ => None,
            }) {
                match tag_base_name(tag) {
                    "Vec" => return Ok(fields.len() as u128),
                    "String" | "OsString" => {
                        if let Some(text) = path_text(value) {
                            return Ok(text.len() as u128);
                        }
                    }
                    _ => {}
                }
            }
            Err("value has no length".to_string())
        }
        _ => Err("value has no length".to_string()),
    }
}

/// Whether the value's recorded shape — after the implicit
/// dereference a field step performs — has a field of `name`:
/// the ADT shape's active variant, or a closure environment's
/// captures. A recorded field wins over a pseudo-field of the
/// same name (a struct may perfectly well declare a field
/// called `length`).
fn has_recorded_field(
    interp: &Interpreter,
    frames: &[Frame],
    value: &Value,
    desc: &Option<Desc>,
    name: &str,
) -> bool {
    let Ok((value, desc)) =
        deref_if_pointer(interp, frames, value.clone(), desc.clone())
    else {
        return false;
    };
    let Some(Desc::Adt(tag)) = &desc else {
        return false;
    };
    if let Some(shape) = interp.program().adt_debug_shapes.get(tag) {
        let variant = match &value {
            Value::Enum { variant, .. } => *variant as usize,
            _ => 0,
        };
        if let Some(vshape) = shape.variants.get(variant) {
            if vshape.fields.iter().any(|f| f.name.as_deref() == Some(name)) {
                return true;
            }
        }
    }
    if matches!(value, Value::Closure { .. }) {
        if let Some(shape) = interp.program().env_debug_shapes.get(tag) {
            if shape.upvars.iter().any(|f| f.name.as_deref() == Some(name)) {
                return true;
            }
        }
    }
    false
}

/// Apply one named-field step (`.name`) to a value: the fat
/// pointer pseudo-fields (`length`, `data_ptr`), then — after
/// an implicit dereference — the active variant's named field
/// from the value's debug shape, or a closure environment's
/// capture of that name.
fn apply_field_name(
    interp: &Interpreter,
    frames: &[Frame],
    value: Value,
    desc: Option<Desc>,
    name: &str,
    expr_text: &str,
) -> Result<(Value, Option<Desc>), String> {
    if name == "length" && !has_recorded_field(interp, frames, &value, &desc, name) {
        let len = sequence_length(interp, frames, &value, desc.as_ref())?;
        return Ok((Value::Uint(len), Some(Desc::Int("usize"))));
    }
    if name == "data_ptr" {
        if let Value::FatRef { base, .. } = &value {
            let elem = match &desc {
                Some(Desc::Ref(inner) | Desc::RefMut(inner)) => {
                    match inner.as_ref() {
                        Desc::Slice(elem) => Some((**elem).clone()),
                        _ => None,
                    }
                }
                _ => None,
            };
            // The base of a fat reference built from a constant
            // (a slice constant's element tuple) is the aggregate
            // itself, not a reference to it; a `data_ptr` is a
            // pointer, so the aggregate is wrapped the way the
            // runtime's own slice values carry it.
            let base = (**base).clone();
            let base = match base {
                Value::Ref(_)
                | Value::FatRef { .. }
                | Value::DynRef { .. }
                | Value::FnPtr(_) => base,
                other => Value::Ref(crate::value::RefTarget::value(other)),
            };
            return Ok((base, elem.map(|e| Desc::Ref(Box::new(e)))));
        }
    }
    let (value, desc) = deref_if_pointer(interp, frames, value, desc)?;
    if let Some(Desc::Adt(tag)) = &desc {
        if let Some(shape) = interp.program().adt_debug_shapes.get(tag) {
            let variant = match &value {
                Value::Enum { variant, .. } => *variant as usize,
                _ => 0,
            };
            if let Some(vshape) = shape.variants.get(variant) {
                if let Some(index) = vshape
                    .fields
                    .iter()
                    .position(|f| f.name.as_deref() == Some(name))
                {
                    let next = Desc::from_tag(&vshape.fields[index].ty);
                    // A union stores only the written field;
                    // reading another field reinterprets that
                    // storage through the requested field's
                    // type, the reading whole-union printing
                    // applies.
                    if let Some(reinterpreted) =
                        union_projected_field(interp, desc.as_ref(), &value, index)
                    {
                        return Ok((reinterpreted, Some(next)));
                    }
                    let projected = crate::interp::project_value(
                        value,
                        &[RefProj::Field(index)],
                        "debugger",
                        expr_text,
                    )
                    .map_err(|e| e.to_string())?;
                    return Ok((projected, Some(next)));
                }
            }
        }
        if let Some(shape) = interp.program().env_debug_shapes.get(tag) {
            if let Value::Closure { env, .. } = &value {
                if let Some(index) = shape
                    .upvars
                    .iter()
                    .position(|f| f.name.as_deref() == Some(name))
                {
                    if let Some(stored) = env.get(index) {
                        let next = Desc::from_tag(&shape.upvars[index].ty);
                        return Ok((stored.clone(), Some(next)));
                    }
                }
            }
        }
    }
    Err(format!("no field `{name}` in `{expr_text}`"))
}

/// The element aggregate a cast to an array/slice type views:
/// unwrap fat pointers and references until the pointee value
/// itself. A sub-slice's data pointer can denote a reference
/// to a materialised fat-pointer value (the sub-slice built as
/// a standalone value, then referenced), so the unwrapping
/// recurses through that layer too.
fn cast_pointee_elements(
    interp: &Interpreter,
    frames: &[Frame],
    value: Value,
) -> Result<Value, String> {
    match value {
        Value::FatRef { base, .. } => {
            cast_pointee_elements(interp, frames, (*base).clone())
        }
        Value::Ref(_) | Value::DynRef { .. } => {
            let resolved = interp
                .deref_value(frames, &value)
                .map_err(|e| e.to_string())?;
            let read = interp
                .read_resolved(frames, &resolved)
                .map_err(|e| e.to_string())?;
            cast_pointee_elements(interp, frames, read)
        }
        other => Ok(other),
    }
}

/// The byte width of a scalar descriptor's layout, for the
/// scalar families a byte reinterpretation can express.
/// Half- and quad-precision floats are excluded: their values
/// do not survive this runtime's `f64` transport bit-exactly,
/// so a reinterpretation through them is rejected instead.
pub(super) fn desc_scalar_width(desc: &Desc) -> Option<usize> {
    Some(match desc {
        Desc::Int(name) => match *name {
            "i8" | "u8" => 1,
            "i16" | "u16" => 2,
            "i32" | "u32" => 4,
            "i64" | "u64" | "isize" | "usize" => 8,
            "i128" | "u128" => 16,
            _ => return None,
        },
        Desc::Float("f32") => 4,
        Desc::Float("f64") => 8,
        Desc::Bool => 1,
        Desc::Char => 4,
        _ => return None,
    })
}

/// A scalar value's little-endian storage bytes, at the
/// width its descriptor declares.
fn scalar_value_bytes(value: &Value, desc: &Desc) -> Option<Vec<u8>> {
    let width = desc_scalar_width(desc)?;
    Some(match (value, desc) {
        (Value::Int(v), Desc::Int(_)) => v.to_le_bytes()[..width].to_vec(),
        (Value::Uint(v), Desc::Int(_)) => v.to_le_bytes()[..width].to_vec(),
        (Value::Uint(v), Desc::Char) => (*v as u32).to_le_bytes().to_vec(),
        (Value::Float(v), Desc::Float("f32")) => {
            (*v as f32).to_bits().to_le_bytes().to_vec()
        }
        (Value::Float(v), Desc::Float("f64")) => {
            v.to_bits().to_le_bytes().to_vec()
        }
        (Value::Bool(b), Desc::Bool) => vec![*b as u8],
        _ => return None,
    })
}

/// A scalar value read back from its little-endian storage
/// bytes under a target descriptor.
fn scalar_value_from_bytes(bytes: &[u8], desc: &Desc) -> Option<Value> {
    match desc {
        Desc::Int(name) => {
            let mut buf = [0u8; 16];
            buf[..bytes.len()].copy_from_slice(bytes);
            let raw = u128::from_le_bytes(buf);
            if name.starts_with('i') {
                let shift = 128 - (bytes.len() as u32) * 8;
                let v = raw as i128;
                Some(Value::Int((v << shift) >> shift))
            } else {
                Some(Value::Uint(raw))
            }
        }
        Desc::Float("f32") => {
            let bits = u32::from_le_bytes(bytes.try_into().ok()?);
            Some(Value::Float(f32::from_bits(bits) as f64))
        }
        Desc::Float("f64") => {
            let bits = u64::from_le_bytes(bytes.try_into().ok()?);
            Some(Value::Float(f64::from_bits(bits)))
        }
        Desc::Bool => Some(Value::Bool(bytes[0] != 0)),
        Desc::Char => {
            let raw = u32::from_le_bytes(bytes.try_into().ok()?);
            char::from_u32(raw).map(|_| Value::Uint(raw as u128))
        }
        _ => None,
    }
}

/// The element descriptor a cast's source value carries: the
/// pointee array/slice element of the source descriptor, the
/// scalar a pointer-to-scalar descriptor names when the
/// backing value is an element aggregate, or `u8` for a
/// byte-form pointee.
fn cast_source_elem(pointee: &Value, source: Option<&Desc>) -> Option<Desc> {
    if matches!(pointee, Value::Bytes(_)) {
        return Some(Desc::Int("u8"));
    }
    let desc = source?;
    let inner = pointer_pointee(desc).unwrap_or(desc);
    match inner {
        Desc::Array(elem, _) | Desc::Slice(elem) => Some((**elem).clone()),
        // A pointer to a scalar over an element aggregate --
        // the form a fat pointer's `data_ptr` projection
        // produces -- names the storage's element type
        // directly: the scalar is what each stored element
        // holds. A scalar backing value is a one-element
        // storage of that same type.
        scalar
            if is_scalar_desc(scalar)
                && (matches!(pointee, Value::Tuple(_)) || is_scalar_value(pointee)) =>
        {
            Some(scalar.clone())
        }
        _ => None,
    }
}

/// Reinterpret a cast pointee's elements through a different
/// target element type, comparing byte counts rather than
/// element counts. Returns `None` when no reinterpretation
/// applies (the element type is unchanged, or the source
/// element type is not recorded), and `Some` of the
/// reinterpreted value -- or of an explicit error for a
/// change the scalar layouts cannot express -- otherwise.
/// `count` is the target array's length; a slice target
/// reinterprets the whole storage.
fn reinterpret_cast(
    pointee: &Value,
    source: Option<&Desc>,
    target_elem: &Desc,
    count: Option<u64>,
) -> Option<Result<Value, String>> {
    let source_elem = cast_source_elem(pointee, source)?;
    // A scalar backing value is one element of storage: the
    // view still has to materialize (its bytes read as the
    // target's elements), where an aggregate already holding
    // the target's element type needs no reinterpretation.
    let scalar_backing = is_scalar_value(pointee);
    if source_elem == *target_elem && !scalar_backing {
        return None;
    }
    let items: Vec<Value> = match pointee {
        Value::Tuple(items) => items.clone(),
        Value::Bytes(bytes) => bytes
            .iter()
            .map(|b| Value::Uint(*b as u128))
            .collect(),
        scalar if scalar_backing => vec![scalar.clone()],
        _ => return None,
    };
    let unsupported = || {
        format!(
            "cast changing element type from {} to {} is not supported",
            desc_type_name(&source_elem).unwrap_or_else(|| "?".into()),
            desc_type_name(target_elem).unwrap_or_else(|| "?".into()),
        )
    };
    let mut bytes = Vec::new();
    for item in &items {
        match scalar_value_bytes(item, &source_elem) {
            Some(bs) => bytes.extend(bs),
            None => return Some(Err(unsupported())),
        }
    }
    let Some(width) = desc_scalar_width(target_elem) else {
        return Some(Err(unsupported()));
    };
    let take = match count {
        Some(n) => {
            if bytes.len() < (n as usize) * width {
                return Some(Err(format!(
                    "cast to an array of {n} elements over {} bytes of backing storage",
                    bytes.len()
                )));
            }
            n as usize
        }
        None => {
            if bytes.len() % width != 0 {
                return Some(Err(format!(
                    "cast of {} bytes of backing storage to a slice of {}-byte elements",
                    bytes.len(),
                    width
                )));
            }
            bytes.len() / width
        }
    };
    let mut out = Vec::with_capacity(take);
    for i in 0..take {
        let chunk = &bytes[i * width..(i + 1) * width];
        match scalar_value_from_bytes(chunk, target_elem) {
            Some(v) => out.push(v),
            None => return Some(Err(unsupported())),
        }
    }
    Some(Ok(Value::Tuple(out)))
}

/// The scalar type a descriptor's storage starts with: its
/// pointee (or the descriptor itself) when scalar, or an
/// array/slice pointee's element when scalar -- the type
/// whose bytes a scalar dereference of the storage re-reads.
fn storage_scalar_desc(desc: &Desc) -> Option<Desc> {
    let inner = pointer_pointee(desc).unwrap_or(desc);
    match inner {
        Desc::Array(elem, _) | Desc::Slice(elem) => match &**elem {
            scalar if is_scalar_desc(scalar) => Some(scalar.clone()),
            _ => None,
        },
        scalar if is_scalar_desc(scalar) => Some(scalar.clone()),
        _ => None,
    }
}

/// Whether a value is a scalar: the storage form a single
/// arithmetic, boolean or character value takes, as opposed
/// to an aggregate of elements.
fn is_scalar_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Bool(_) | Value::Int(_) | Value::Uint(_) | Value::Float(_)
    )
}

/// Whether a descriptor names a scalar type, the family a
/// byte reinterpretation can express.
fn is_scalar_desc(desc: &Desc) -> bool {
    matches!(
        desc,
        Desc::Int(_) | Desc::Float(_) | Desc::Bool | Desc::Char
    )
}

/// Re-read a scalar pointee value's storage bytes through a
/// different scalar layout: the dereference side of a pointer
/// cast that changed the pointee type, whose target storage
/// still holds the source type's bytes. A change the scalar
/// layouts cannot express, and a target wider than the
/// storage the source type occupies, are explicit errors
/// rather than wrong values.
fn reinterpret_scalar_pointee(
    value: &Value,
    source: &Desc,
    target: &Desc,
) -> Result<Value, String> {
    // A cast to the pointee's own type changes no bytes, so
    // there is nothing to reinterpret -- but the dereference
    // still reads a single scalar: the storage a scalar
    // pointer reaches (a slice's `data_ptr`, above all) is
    // the whole element aggregate, whose first element is
    // the pointee. Any other value form already is the
    // scalar the pointer names (or a storage form the byte
    // path does not model) and passes through unchanged.
    if source == target {
        return Ok(match value {
            Value::Tuple(items) => match items.first() {
                Some(first) if is_scalar_value(first) => first.clone(),
                _ => value.clone(),
            },
            _ => value.clone(),
        });
    }
    let unsupported = || {
        format!(
            "cast changing pointee type from {} to {} is not supported",
            desc_type_name(source).unwrap_or_else(|| "?".into()),
            desc_type_name(target).unwrap_or_else(|| "?".into()),
        )
    };
    // The pointee value may arrive as an element aggregate
    // (a cast that reached the storage through an array
    // view): its items are consecutive storage, serialized
    // under the source type the way `reinterpret_cast`
    // serializes them, and the target reads the leading
    // bytes. An item the source layout cannot express is
    // the same explicit error as an unexpressible scalar.
    let bytes = match value {
        Value::Tuple(items) => {
            let mut bytes = Vec::new();
            for item in items {
                match scalar_value_bytes(item, source) {
                    Some(bs) => bytes.extend(bs),
                    None => return Err(unsupported()),
                }
            }
            bytes
        }
        _ => match scalar_value_bytes(value, source) {
            Some(bytes) => bytes,
            None => return Err(unsupported()),
        },
    };
    let Some(width) = desc_scalar_width(target) else {
        return Err(unsupported());
    };
    if bytes.len() < width {
        return Err(format!(
            "cast to a {width}-byte pointee over {} bytes of backing storage",
            bytes.len()
        ));
    }
    match scalar_value_from_bytes(&bytes[..width], target) {
        Some(value) => Ok(value),
        None => Err(unsupported()),
    }
}

/// Read one element of a cast pointer's storage through
/// the target layout: the indexing counterpart of
/// `reinterpret_scalar_pointee`. The storage's bytes are the
/// source type's, serialized the way `reinterpret_cast`
/// serializes them; the element at `index` occupies the
/// target-width slot at that offset, and an index past the
/// storage the source occupies is an explicit error rather
/// than a read of neighbouring memory.
fn reinterpret_index(
    storage: &Value,
    source: &Desc,
    target: &Desc,
    index: usize,
) -> Result<Value, String> {
    let unsupported = || {
        format!(
            "cast changing pointee type from {} to {} is not supported",
            desc_type_name(source).unwrap_or_else(|| "?".into()),
            desc_type_name(target).unwrap_or_else(|| "?".into()),
        )
    };
    let bytes = match storage {
        Value::Tuple(items) => {
            let mut bytes = Vec::new();
            for item in items {
                match scalar_value_bytes(item, source) {
                    Some(bs) => bytes.extend(bs),
                    None => return Err(unsupported()),
                }
            }
            bytes
        }
        Value::Bytes(bytes) => bytes.clone(),
        _ => match scalar_value_bytes(storage, source) {
            Some(bytes) => bytes,
            None => return Err(unsupported()),
        },
    };
    let Some(width) = desc_scalar_width(target) else {
        return Err(unsupported());
    };
    let offset = index * width;
    if bytes.len() < offset + width {
        return Err(format!(
            "index {index} of a {width}-byte element over {} bytes of backing storage",
            bytes.len()
        ));
    }
    match scalar_value_from_bytes(&bytes[offset..offset + width], target) {
        Some(value) => Ok(value),
        None => Err(unsupported()),
    }
}

/// Whether a descriptor is a pointer (a reference or a raw
/// pointer) to an array or a slice: the cast target form
/// whose view of the pointee a consuming step materializes.
fn is_array_pointer_desc(desc: &Desc) -> bool {
    match desc {
        Desc::Ref(inner) | Desc::RefMut(inner) | Desc::PtrConst(inner) | Desc::PtrMut(inner) => {
            matches!(**inner, Desc::Array(..) | Desc::Slice(_))
        }
        _ => false,
    }
}

/// Materialize the array/slice view a cast to an array or
/// slice pointer spelled (see
/// `EvaluatedValue::pending_array_view`): read the pointee's
/// elements through the cast, the reading a dereference or
/// projection of the cast pointer applies. A no-op when no
/// view is pending.
fn materialize_array_view(
    interp: &Interpreter,
    frames: &[Frame],
    ev: &mut EvaluatedValue,
) -> Result<(), String> {
    let Some((spelled, source)) = ev.pending_array_view.take() else {
        return Ok(());
    };
    let (value, desc) = apply_cast(interp, frames, ev.value.clone(), source.as_ref(), spelled)?;
    ev.value = value;
    ev.desc = desc;
    Ok(())
}

/// Apply a cast step (`as *const [T; N]` and kin): a pointer
/// cast retargets the descriptor; an array cast views the
/// pointee's elements as an array value (the form the slice
/// tests print through a fat pointer's `data_ptr`).
fn apply_cast(
    interp: &Interpreter,
    frames: &[Frame],
    value: Value,
    source: Option<&Desc>,
    target: Desc,
) -> Result<(Value, Option<Desc>), String> {
    // A cast spelled as a pointer to an array/slice (`as *const
    // [T; N]`) is the array cast below: the pointer layer is
    // the spelling, not a value layer.
    let target = match target {
        Desc::Ref(inner) | Desc::RefMut(inner) | Desc::PtrConst(inner) | Desc::PtrMut(inner)
            if matches!(*inner, Desc::Array(..) | Desc::Slice(_)) =>
        {
            *inner
        }
        other => other,
    };
    match target {
        Desc::Array(target_elem, n) => {
            let pointee = cast_pointee_elements(interp, frames, value)?;
            // A cast that changes the element type reinterprets
            // the backing storage through the target element
            // layout (the reading a native debugger applies);
            // a change the scalar layouts cannot express is an
            // explicit error rather than a wrong value.
            if let Some(reinterpreted) = reinterpret_cast(&pointee, source, &target_elem, Some(n)) {
                return Ok((reinterpreted?, Some(Desc::Array(target_elem, n))));
            }
            // The cast views the backing elements through the
            // target type: a shorter array sees only its first
            // elements (and a later index step bounds against
            // them, as it projects this value), and a target
            // longer than the backing holds is an error rather
            // than a short read.
            let pointee = match pointee {
                Value::Tuple(mut items) => {
                    if (items.len() as u64) < n {
                        return Err(format!(
                            "cast to an array of {n} elements over a value holding {}",
                            items.len()
                        ));
                    }
                    items.truncate(n as usize);
                    Value::Tuple(items)
                }
                other => other,
            };
            Ok((pointee, Some(Desc::Array(target_elem, n))))
        }
        Desc::Slice(target_elem) => {
            let pointee = cast_pointee_elements(interp, frames, value)?;
            if let Some(reinterpreted) = reinterpret_cast(&pointee, source, &target_elem, None) {
                return Ok((reinterpreted?, Some(Desc::Slice(target_elem))));
            }
            Ok((pointee, Some(Desc::Slice(target_elem))))
        }
        other => Ok((value, Some(other))),
    }
}

/// Evaluate a print expression: resolve its base name, then
/// apply its dereference, field, index and cast steps.
pub(super) fn evaluate_expr(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    expr_text: &str,
) -> Result<EvaluatedValue, String> {
    let expr = parse_expr(expr_text)?;
    let mut ev = resolve_base(interp, ctl, frames, &expr)?;
    for step in &expr.steps {
        match step {
            ExprStep::Deref => {
                // A cast to an array/slice pointer kept the
                // pointer value; this dereference consumes it,
                // so the spelled view materializes first, after
                // which the dereference is the identity on the
                // elements, as for a cast to an array type that
                // already replaced the value with the pointee's
                // elements (see `apply_cast`).
                materialize_array_view(interp, frames, &mut ev)?;
                if matches!(ev.desc, Some(Desc::Array(..) | Desc::Slice(_)))
                    && matches!(ev.value, Value::Tuple(_))
                {
                    // The dereference consumes the pointer
                    // layer a cast spelled, and the spelled
                    // type with it (the loop's step bookkeeping
                    // below does not run on this path).
                    ev.cast_type = None;
                    continue;
                }
                let resolved = interp
                    .deref_value(frames, &ev.value)
                    .map_err(|e| e.to_string())?;
                ev.value = interp
                    .read_resolved(frames, &resolved)
                    .map_err(|e| e.to_string())?;
                ev.desc = match ev.desc {
                    Some(
                        Desc::Ref(inner)
                        | Desc::RefMut(inner)
                        | Desc::PtrConst(inner)
                        | Desc::PtrMut(inner),
                    ) => Some(*inner),
                    Some(Desc::Adt(tag)) => {
                        smart_pointer_pointee(&tag).map(|t| Desc::from_tag(&t))
                    }
                    _ => None,
                };
                // A cast step that changed the pointee's
                // scalar type recorded a reinterpretation for
                // this dereference: the storage still holds
                // the source type's bytes, and the value is
                // re-read through the target layout (the
                // reading a native debugger applies) rather
                // than rendered under the target descriptor
                // unchanged.
                if let Some((source, target)) = ev.pending_cast.take() {
                    ev.value = reinterpret_scalar_pointee(&ev.value, &source, &target)?;
                }
                // The char hint is recorded for the variable as
                // declared: a `char` local, or a reference to a
                // char (whose pointee the hint then describes).
                // A dereference step therefore keeps it.
            }
            ExprStep::Field(index) => {
                materialize_array_view(interp, frames, &mut ev)?;
                let (v, d) = deref_if_pointer(interp, frames, ev.value, ev.desc)?;
                ev.value = v;
                ev.desc = d;
                let next = ev
                    .desc
                    .as_ref()
                    .and_then(|d| field_desc(interp, d, &ev.value, *index));
                // A union field read reinterprets the stored
                // field's storage (see `apply_field_name`);
                // any other value projects directly.
                ev.value = match union_projected_field(
                    interp,
                    ev.desc.as_ref(),
                    &ev.value,
                    *index,
                ) {
                    Some(reinterpreted) => reinterpreted,
                    None => crate::interp::project_value(
                        ev.value,
                        &[RefProj::Field(*index)],
                        "debugger",
                        expr_text,
                    )
                    .map_err(|e| e.to_string())?,
                };
                ev.desc = next;
                ev.char = false;
            }
            ExprStep::FieldName(name) => {
                materialize_array_view(interp, frames, &mut ev)?;
                let (v, d) = apply_field_name(
                    interp,
                    frames,
                    ev.value,
                    ev.desc,
                    name,
                    expr_text,
                )?;
                ev.value = v;
                ev.desc = d;
                ev.char = false;
            }
            ExprStep::Index(index) => {
                materialize_array_view(interp, frames, &mut ev)?;
                // A cast that changed the pointer's scalar
                // pointee type recorded a reinterpretation for
                // the dereference that consumes the pointer;
                // indexing dereferences it too, and reads the
                // element through the target layout at the
                // target's width rather than projecting the
                // storage's own elements.
                // A cast to the pointee's own type records
                // the pair as well, but indexing it reads the
                // storage's own elements: the ordinary
                // projection below already consumes the right
                // one, so only a type-changing pair takes the
                // reinterpretation branch.
                let reinterpret = match ev.pending_cast.take() {
                    Some((source, target)) if source != target => {
                        Some((source, target))
                    }
                    _ => None,
                };
                if let Some((source, target)) = reinterpret {
                    let (v, _) = deref_if_pointer(interp, frames, ev.value, ev.desc)?;
                    ev.value = reinterpret_index(&v, &source, &target, *index)?;
                    ev.desc = Some(target);
                } else {
                    let (v, d) = deref_if_pointer(interp, frames, ev.value, ev.desc)?;
                    ev.value = v;
                    ev.desc = d;
                    let next = ev.desc.as_ref().and_then(|d| match d {
                        Desc::Array(elem, _) | Desc::Slice(elem) => {
                            Some((**elem).clone())
                        }
                        _ => None,
                    });
                    ev.value = crate::interp::project_value(
                        ev.value,
                        &[RefProj::Index(*index)],
                        "debugger",
                        expr_text,
                    )
                    .map_err(|e| e.to_string())?;
                    ev.desc = next;
                }
                ev.char = false;
            }
            ExprStep::Cast(target) => {
                // A cast that changes a pointer's scalar
                // pointee type leaves the target storage
                // holding the source type's bytes: record the
                // reinterpretation for the dereference that
                // consumes the pointer (see the `Deref` arm).
                // A chain of such casts composes back to the
                // first cast's source, the type the storage
                // actually holds.
                let source_pointee = ev.desc.as_ref().and_then(pointer_pointee).cloned();
                let target_pointee = pointer_pointee(target).cloned();
                // When this cast supersedes a pending array
                // view, the descriptor the view recorded names
                // the storage's actual type: the spelled
                // descriptor is an array pointer whose pointee
                // is not the scalar the storage holds, so the
                // reinterpretation a scalar target composes
                // starts from the recorded descriptor's scalar
                // rather than discarding it with the view.
                let view_source_scalar = ev
                    .pending_array_view
                    .as_ref()
                    .and_then(|(_, recorded)| recorded.as_ref())
                    .and_then(storage_scalar_desc);
                // A pointer whose pointee is an array or slice
                // of scalars holds its storage in the elements:
                // a scalar target reinterprets the leading
                // element's bytes, so the composition starts
                // from the element type the descriptor records.
                // (With a view pending, the recorded source --
                // not this cast's spelling -- names the
                // storage, as above.)
                let desc_source_scalar = if ev.pending_array_view.is_none() {
                    ev.desc.as_ref().and_then(storage_scalar_desc)
                } else {
                    None
                };
                let effective_source = match &source_pointee {
                    Some(s) if is_scalar_desc(s) => source_pointee.clone(),
                    _ => view_source_scalar
                        .or(desc_source_scalar)
                        .or(source_pointee.clone()),
                };
                // A cast to an array or slice pointer keeps
                // the pointer value: only a step that consumes
                // the pointer materializes the spelled view
                // (see `pending_array_view`), so printing the
                // cast itself renders the pointer, not the
                // pointee. Any other cast supersedes a pending
                // view.
                let (v, d) = if is_array_pointer_desc(target) {
                    // A further array-pointer cast before the
                    // view materializes re-spells the view but
                    // does not reinterpret the storage: the
                    // pending view keeps the descriptor the
                    // first cast recorded, the type the stored
                    // bytes actually hold, so materialization
                    // reinterprets from that type rather than
                    // from the intermediate cast's.
                    let source = match &ev.pending_array_view {
                        Some((_, recorded)) => recorded.clone(),
                        // A pending scalar reinterpretation
                        // names the storage's actual type: the
                        // descriptor this cast replaces is the
                        // intermediate cast's, whose pointee
                        // the storage does not hold.
                        None => match &ev.pending_cast {
                            Some((first, _)) => Some(first.clone()),
                            None => ev.desc.clone(),
                        },
                    };
                    ev.pending_array_view = Some((target.clone(), source));
                    (ev.value.clone(), Some(target.clone()))
                } else {
                    ev.pending_array_view = None;
                    apply_cast(
                        interp,
                        frames,
                        ev.value,
                        ev.desc.as_ref(),
                        target.clone(),
                    )?
                };
                ev.value = v;
                ev.desc = d;
                ev.char = false;
                let new_pending = match (effective_source, target_pointee) {
                    (Some(source), Some(target_inner))
                        if is_scalar_desc(&source) && is_scalar_desc(&target_inner) =>
                    {
                        let source = match &ev.pending_cast {
                            Some((first, mid)) if *mid == source => first.clone(),
                            _ => source,
                        };
                        // The pair is recorded even when the
                        // cast changes nothing: the storage a
                        // scalar pointer reaches is the whole
                        // element aggregate, and the dereference
                        // that consumes the pointer must still
                        // read a single scalar from it (see
                        // `reinterpret_scalar_pointee`).
                        Some((source, target_inner))
                    }
                    _ => None,
                };
                ev.pending_cast = new_pending;
            }
        }
        // The recorded tag describes the base value's declared
        // type; after a step it no longer describes the value.
        ev.tag = None;
        // A cast step's spelled type and pending pointee
        // reinterpretation describe the expression only until
        // a later step consumes them: a dereference applies
        // (and clears) the reinterpretation, and any step
        // other than a cast drops the spelled type.
        if !matches!(step, ExprStep::Cast(_) | ExprStep::Deref) {
            ev.pending_cast = None;
        }
        ev.cast_type = match step {
            ExprStep::Cast(target) => Some(target.clone()),
            _ => None,
        };
    }
    Ok(ev)
}

/// Evaluate a print expression in the selected frame and
/// render the resulting value in both dialects' forms.
pub(super) fn evaluate_print(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    print: &DebugPrint,
) -> Result<EvaluatedPrint, String> {
    let ev = evaluate_expr(interp, ctl, frames, &print.expr)?;
    let tyname = expr_type_name(&ev);
    let (value, desc, char) = (ev.value, ev.desc, ev.char);
    // The lldb synthetic-child providers' summary lines for
    // the sequence and counted-handle types (the forms the
    // sysroot's `lldb_providers.py` produces), appended to the
    // dialect's own lines when the printed value's type has
    // one.
    let mut lldb_lines_extra: Vec<String> = Vec::new();
    if parse_expr(&print.expr).map(|e| e.steps.is_empty()).unwrap_or(false) {
        if let Some(line) = render_lldb_provider_line(
            interp,
            frames,
            &value,
            desc.as_ref(),
            ev.tag.as_deref(),
            &print.expr,
        ) {
            lldb_lines_extra.push(line);
        }
    }
    // lldb's summary for a reference to a path is the path text
    // itself, not the address: render the pointee.
    let mut lldb = if matches!(value, Value::Closure { .. } | Value::Coroutine { .. }) {
        // A closure or coroutine value renders through its
        // recorded environment shape; the lldb form needs the
        // frames to read through a captured reference field.
        render_env_lldb_lines(interp, frames, &value, desc.as_ref(), &print.expr)
    } else {
        match &desc {
            Some(Desc::Ref(inner) | Desc::RefMut(inner))
                if parse_expr(&print.expr).map(|e| e.steps.is_empty()).unwrap_or(false)
                    && matches!(
                        inner.as_ref(),
                        Desc::Adt(tag)
                            if matches!(tag_base_name(tag), "Path" | "OsStr")
                    ) =>
            {
                let pointee = interp
                    .deref_value(frames, &value)
                    .and_then(|resolved| interp.read_resolved(frames, &resolved));
                match pointee {
                    Ok(target) => render_lldb_lines(
                        interp,
                        &target,
                        Some(inner.as_ref()),
                        char,
                        &print.expr,
                    ),
                    Err(_) => render_lldb_lines(
                        interp,
                        &value,
                        desc.as_ref(),
                        char,
                        &print.expr,
                    ),
                }
            }
            _ => render_lldb_lines(interp, &value, desc.as_ref(), char, &print.expr),
        }
    };
    lldb.extend(lldb_lines_extra);
    // An lldb `type format add` registration for the printed
    // value's own type replaces the value's rendering with the
    // registered numeric format (see [`lldb_format_scalar`]).
    if let Some(formatted) =
        lldb_format_scalar(&value, ev.tag.as_deref(), desc.as_ref(), ctl.lldb_formats)
    {
        if let Some(first) = lldb.first_mut() {
            if first.starts_with("= ") {
                *first = format!("= {formatted}");
            }
        }
    }
    // A per-command format (`p/u b`) prescribes this print's
    // own reading of the value, overriding the registered
    // type format as it does in lldb itself.
    if let Some(format) = print.format {
        if let Some(formatted) =
            lldb_command_format_scalar(&value, ev.tag.as_deref(), desc.as_ref(), format)
        {
            if let Some(first) = lldb.first_mut() {
                if first.starts_with("= ") {
                    *first = format!("= {formatted}");
                }
            }
        }
    }
    // lldb's `(Type) name = value` head line, for a print whose
    // rendering is the bare summary form: checks that name the
    // printed variable — statics printed by path, in
    // particular — anchor on the name, which the bare form
    // omits.
    if parse_expr(&print.expr)
        .map(|e| e.steps.is_empty())
        .unwrap_or(false)
    {
        if let Some(first) = lldb.first().cloned() {
            if let Some(compact) = first.strip_prefix("= ") {
                if let Some(tyname) = &tyname {
                    lldb.push(format!(
                        "({}) {} = {}",
                        lldb_type_name(tyname),
                        print.expr,
                        compact
                    ));
                }
            }
        }
    }
    Ok(EvaluatedPrint {
        gdb: if ctl.pretty_printers && print.format.is_none() {
            render_pretty_gdb(
                interp,
                frames,
                &value,
                desc.as_ref(),
                ev.tag.as_deref(),
                0,
            )
            .or_else(|| render_gdb(interp, &value, desc.as_ref(), char, print.format, 0))
        } else {
            render_gdb(interp, &value, desc.as_ref(), char, print.format, 0)
        },
        lldb,
    })
}

