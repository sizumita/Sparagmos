//! gdb-form rendering: aggregates, layouts and the union byte forms behind the raw `$N = ...` print.

use crate::interp::{Frame, Interpreter};
use crate::value::Value;
use super::desc::{Desc, dwarf_type_name, split_top_level};
use super::printers::render_gdb_any;
use super::queries::{desc_type_name, nonzero_scalar};
use super::render_lldb::{
    escape_str, render_env_gdb, render_lldb_compact, render_scalar, scalar_desc_bit_width,
};
use super::session::{DebugDialect, PrintFormat};

/// The deepest aggregate nesting the renderers unfold. A value
/// graph in this model is finite (recursive types pass through
/// references, which render as addresses), so the cap is a
/// safeguard, not a shaping rule.
pub(super) const MAX_RENDER_DEPTH: usize = 24;

/// The inline form of a union field a debugger cannot read:
/// gdb reports such a field in place (`<error reading
/// variable: ...>`) and still prints the union's other
/// fields, so the whole-union rendering degrades the field
/// instead of failing with it.
pub(super) const UNION_FIELD_UNREADABLE: &str = "<error reading variable>";

/// Render a sequence of element values, in one dialect's element
/// form: per-element descriptors from `descs`, or one shared
/// element descriptor.
pub(super) fn render_elements(
    interp: &Interpreter,
    fields: &[Value],
    descs: &[Desc],
    shared: Option<&Desc>,
    dialect: DebugDialect,
    depth: usize,
    frames: &[Frame],
    pretty: bool,
    elements: usize,
) -> Option<Vec<String>> {
    let mut parts = Vec::with_capacity(fields.len());
    for (i, field) in fields.iter().enumerate() {
        let d = shared.or_else(|| descs.get(i));
        parts.push(match dialect {
            DebugDialect::Gdb => {
                if pretty {
                    render_gdb_any(interp, frames, field, d, depth + 1, elements)?
                } else {
                    render_gdb_mode(
                        interp,
                        frames,
                        field,
                        d,
                        false,
                        None,
                        depth + 1,
                        false,
                        elements,
                    )?
                }
            }
            DebugDialect::Lldb => {
                render_lldb_compact(interp, field, d, false, true, depth + 1)?
            }
        });
    }
    Some(parts)
}

/// gdb's array-element repeat compression: a run of more than
/// ten identical elements is printed once, followed by
/// `<repeats N times>` with the run's full length.
pub(super) fn compress_repeats(rendered: &[String]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < rendered.len() {
        let mut j = i + 1;
        while j < rendered.len() && rendered[j] == rendered[i] {
            j += 1;
        }
        let run = j - i;
        if run > 10 {
            out.push(format!("{} <repeats {} times>", rendered[i], run));
        } else {
            out.extend_from_slice(&rendered[i..j]);
        }
        i = j;
    }
    out.join(", ")
}

/// Render a value the way gdb (with its Rust pretty printers
/// disabled, as the debuginfo suite runs it) prints it: scalars
/// as [`render_scalar`] renders them, tuples as `(a, b)`, arrays
/// as `[a, b]` with gdb's repeat compression, and ADT values
/// through the debug-shape table — `Tag {f: v, ...}` for a
/// named-field struct, `Tag (v, ...)` for a tuple struct,
/// `Tag::Variant{...}` / `Tag::Variant(...)` / `Tag::Variant`
/// for an enum. A reference renders as its address. Anything the
/// shape table does not describe stays unrendered (`None`).
pub(super) fn render_gdb(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
    top_char: bool,
    format: Option<PrintFormat>,
    depth: usize,
) -> Option<String> {
    render_gdb_mode(
        interp,
        &[],
        value,
        desc,
        top_char,
        format,
        depth,
        false,
        200,
    )
}

/// Render a value in gdb form in a session with the pretty
/// printers enabled: the raw aggregate form of [`render_gdb`],
/// except that a nested value whose type has a provider
/// renders in the provider's form, the way gdb applies the
/// printers at every level of a printed value (an
/// `Option<String>`'s payload prints as a quoted string, not
/// as the string's raw aggregate). `elements` is the
/// session's `print elements` setting, for the provider
/// forms' truncation.
pub(super) fn render_gdb_with_printers(
    interp: &Interpreter,
    frames: &[Frame],
    value: &Value,
    desc: Option<&Desc>,
    top_char: bool,
    format: Option<PrintFormat>,
    depth: usize,
    elements: usize,
) -> Option<String> {
    render_gdb_mode(
        interp, frames, value, desc, top_char, format, depth, true, elements,
    )
}

/// The shared body of [`render_gdb`] and
/// [`render_gdb_with_printers`]; `pretty` selects the
/// field-rendering mode (see the latter).
#[allow(clippy::too_many_arguments)]
fn render_gdb_mode(
    interp: &Interpreter,
    frames: &[Frame],
    value: &Value,
    desc: Option<&Desc>,
    top_char: bool,
    format: Option<PrintFormat>,
    depth: usize,
    pretty: bool,
    elements: usize,
) -> Option<String> {
    if depth > MAX_RENDER_DEPTH {
        return None;
    }
    if let Some((scalar, inner)) = nonzero_scalar(value, desc) {
        return render_gdb_mode(
            interp,
            frames,
            &scalar,
            Some(&inner),
            top_char,
            format,
            depth,
            pretty,
            elements,
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
            format,
            scalar_desc_bit_width(desc),
            DebugDialect::Gdb,
        ),
        Value::Tuple(fields) => {
            // A tuple-shaped value with an array descriptor is an
            // array; with a slice descriptor it is an array whose
            // length did not survive conversion (the length was
            // named by a const item) — a genuine slice value is a
            // fat reference, never a tuple.
            let array_elem = match desc {
                Some(Desc::Array(elem, _)) | Some(Desc::Slice(elem)) => {
                    Some(elem.as_ref())
                }
                _ => None,
            };
            if let Some(elem) = array_elem {
                let mut rendered = Vec::with_capacity(fields.len());
                for field in fields {
                    rendered.push(if pretty {
                        render_gdb_any(
                            interp,
                            frames,
                            field,
                            Some(elem),
                            depth + 1,
                            elements,
                        )?
                    } else {
                        render_gdb_mode(
                            interp,
                            frames,
                            field,
                            Some(elem),
                            false,
                            None,
                            depth + 1,
                            false,
                            elements,
                        )?
                    });
                }
                Some(format!("[{}]", compress_repeats(&rendered)))
            } else {
                let descs: Vec<Desc> = match desc {
                    Some(Desc::Tuple(ts)) => ts.clone(),
                    _ => Vec::new(),
                };
                let parts = render_elements(
                    interp,
                    fields,
                    &descs,
                    None,
                    DebugDialect::Gdb,
                    depth,
                    frames,
                    pretty,
                    elements,
                )?;
                if parts.len() == 1 {
                    Some(format!("({},)", parts[0]))
                } else {
                    Some(format!("({})", parts.join(", ")))
                }
            }
        }
        Value::Enum { variant, fields } => {
            let Some(Desc::Adt(tag)) = desc else {
                return None;
            };
            let shape = interp.program().adt_debug_shapes.get(tag)?;
            render_adt_gdb(
                interp, tag, shape, *variant, fields, depth, frames, pretty,
                elements,
            )
        }
        Value::Str(s) => {
            // gdb prints a `str` value -- a `&str` place, or a
            // str place a dereference reached -- as its text.
            // A `String`/`OsString` place holds the same value
            // form, but gdb prints those as their struct
            // expansion; the descriptor tells the two apart, and
            // the expansion is not this arm's to synthesize.
            if str_place_desc(desc) {
                Some(format!("\"{}\"", escape_str(s)))
            } else {
                None
            }
        }
        Value::FatRef { base, len } => {
            render_fat_ref_gdb(value, base, *len, desc)
        }
        Value::DynRef {
            concrete, traits, ..
        } => render_dyn_ref_gdb(value, concrete, traits, desc),
        Value::Ref(_) | Value::FnPtr(_) => {
            let addr = crate::interp::synthetic_address(value)?;
            // A pointer to the unit type prints with its type
            // in cast form (`(*mut ()) 0x...`): the unit type
            // has no value form for the address to stand for.
            if matches!(
                desc,
                Some(
                    Desc::Ref(inner)
                    | Desc::RefMut(inner)
                    | Desc::PtrConst(inner)
                    | Desc::PtrMut(inner)
                ) if matches!(inner.as_ref(), Desc::Unit)
            ) {
                return Some(format!("(*mut ()) 0x{addr:x}"));
            }
            Some(format!("0x{addr:x}"))
        }
        Value::Closure { .. } | Value::Coroutine { .. } => {
            render_env_gdb(interp, value, desc, depth)
        }
        Value::Uninit => match desc {
            Some(Desc::Int(_)) | Some(Desc::Float(_)) => Some("0".to_string()),
            Some(Desc::Bool) => Some("false".to_string()),
            _ => None,
        },
        Value::UnionBytes { bytes } => {
            // A union in its byte form: the storage bytes are
            // the value, and every declared field is read out
            // of them (see `union_field_from_bytes`).
            let Some(Desc::Adt(tag)) = desc else {
                return None;
            };
            let shape = interp.program().adt_debug_shapes.get(tag)?;
            if !matches!(shape.kind, sparagmos_mir::AdtShapeKind::Union) {
                return None;
            }
            let vshape = shape.variants.first()?;
            let parts = union_parts_gdb(interp, frames, vshape, depth, pretty, elements, |i| {
                union_field_from_bytes(&vshape.fields, bytes, i)
            })?;
            Some(format!(
                "{} {{{}}}",
                dwarf_type_name(tag),
                parts.join(", ")
            ))
        }
        _ => None,
    }
}

/// Whether a descriptor names a `str` place itself or a
/// pointer-like place (a reference or a raw pointer) whose
/// pointee is `str`: the places gdb prints as the string text.
pub(super) fn str_place_desc(desc: Option<&Desc>) -> bool {
    match desc {
        Some(Desc::Str) => true,
        Some(
            Desc::Ref(inner)
            | Desc::RefMut(inner)
            | Desc::PtrConst(inner)
            | Desc::PtrMut(inner),
        ) => matches!(inner.as_ref(), Desc::Str),
        _ => false,
    }
}

/// The text a fat pointer's base carries when the base is a
/// value snapshot of the text itself (a string constant's
/// storage in this model): the one base form whose bytes the
/// renderers can read without the frame stack. Any other base
/// (a live place, an allocation) stays unreadable here.
pub(super) fn fat_base_text(base: &Value) -> Option<String> {
    match base {
        Value::Str(s) => Some(s.clone()),
        Value::Bytes(b) => Some(String::from_utf8_lossy(b).into_owned()),
        Value::Ref(crate::value::RefTarget::Value(snapshot, projection))
            if projection.is_empty() =>
        {
            fat_base_text(snapshot)
        }
        _ => None,
    }
}

/// Render a fat pointer in gdb's form. gdb's Rust support
/// prints a fat pointer as the struct its DWARF describes:
/// the pointer type's name, then the metadata fields --
/// `data_ptr` and `length`. A fat pointer to `str` is the
/// exception: gdb prints the string's text, which the base
/// snapshot carries when it carries it at all; a base the
/// renderer cannot read keeps the bare-address form. The
/// struct form needs no read, so it is unconditional.
fn render_fat_ref_gdb(
    value: &Value,
    base: &Value,
    len: u128,
    desc: Option<&Desc>,
) -> Option<String> {
    let addr = crate::interp::synthetic_address(value)?;
    if str_place_desc(desc) {
        return match fat_base_text(base) {
            Some(text) => Some(format!("\"{}\"", escape_str(&text))),
            None => Some(format!("0x{addr:x}")),
        };
    }
    let name = match desc {
        Some(Desc::Adt(tag)) => Some(dwarf_type_name(tag)),
        Some(_) => desc_type_name(desc?),
        None => None,
    };
    match name {
        Some(name) => Some(format!("{name} {{data_ptr: 0x{addr:x}, length: {len}}}")),
        None => Some(format!("0x{addr:x}")),
    }
}

/// Render a trait-object fat pointer in gdb's form: the
/// pointer type's name, then the `pointer` and `vtable`
/// fields. The vtable word is the model's own identity for
/// the (concrete type, trait bounds) pair, in the same
/// synthetic address space as every other printed address.
/// A descriptor that recovers no type name (a bare `&dyn`
/// whose pointee descriptor was not recovered) keeps the
/// bare-address form.
fn render_dyn_ref_gdb(
    value: &Value,
    concrete: &str,
    traits: &[String],
    desc: Option<&Desc>,
) -> Option<String> {
    let addr = crate::interp::synthetic_address(value)?;
    let name = match desc {
        Some(Desc::Adt(tag)) => Some(dwarf_type_name(tag)),
        Some(_) => desc_type_name(desc?),
        None => None,
    };
    match name {
        Some(name) => {
            let vtable = crate::interp::vtable_word(concrete, traits);
            Some(format!(
                "{name} {{pointer: 0x{addr:x}, vtable: 0x{vtable:x}}}"
            ))
        }
        None => Some(format!("0x{addr:x}")),
    }
}

/// The byte width of an integer type tag, when the tag names
/// one.
fn int_ty_width(ty: &str) -> Option<usize> {
    match ty {
        "i8" | "u8" => Some(1),
        "i16" | "u16" => Some(2),
        "i32" | "u32" => Some(4),
        "i64" | "u64" | "isize" | "usize" => Some(8),
        "i128" | "u128" => Some(16),
        _ => None,
    }
}

fn align_up(v: usize, align: usize) -> usize {
    (v + align - 1) / align.max(1) * align.max(1)
}

/// Whether a pointee type tag is one a pointer to it is fat
/// for: a slice, `str`, or a `dyn` tail.
fn fat_pointee(pointee: &str) -> bool {
    let pointee = pointee.trim();
    pointee == "str"
        || pointee.starts_with("dyn ")
        || (pointee.starts_with('[') && pointee.ends_with(']') && !pointee.contains(';'))
}

/// The pointee tag of a pointer type tag (a raw pointer or a
/// reference), when the tag names a pointer at all.
fn pointer_pointee(ty: &str) -> Option<&str> {
    let ty = ty.trim();
    if let Some(rest) = ty.strip_prefix("*const ") {
        Some(rest)
    } else if let Some(rest) = ty.strip_prefix("*mut ") {
        Some(rest)
    } else if let Some(rest) = ty.strip_prefix("&mut ") {
        Some(rest)
    } else {
        ty.strip_prefix('&')
    }
}

/// Whether a debugger field type tag names a thin pointer:
/// a raw pointer or reference to a sized pointee, or a
/// function pointer -- one address word of storage. A pointee
/// named by an ADT tag is not judged here: the tag alone does
/// not say whether that ADT is sized, so such a pointer is
/// left to the stored-field identity and the byte forms of
/// the fields around it.
fn thin_pointer_tag(ty: &str) -> bool {
    let ty = ty.trim();
    // A function pointer's recorded tag is the bare kind name
    // (the type tagger has no signature rendering for it).
    if ty.starts_with("fn(") || ty == "FnPtr" {
        return true;
    }
    let Some(pointee) = pointer_pointee(ty) else {
        return false;
    };
    !fat_pointee(pointee) && !matches!(Desc::from_tag(pointee), Desc::Adt(_))
}

/// The fat-pointer kind a debugger field type tag names, when
/// it names one: a two-word pointer (a data word plus a
/// length or vtable word) to a slice, a `str`, or a `dyn`
/// tail.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FatPointerKind {
    Slice,
    Str,
    Dyn,
}

fn fat_pointer_kind(ty: &str) -> Option<FatPointerKind> {
    let pointee = pointer_pointee(ty)?.trim();
    if pointee == "str" {
        Some(FatPointerKind::Str)
    } else if pointee.starts_with("dyn ") {
        Some(FatPointerKind::Dyn)
    } else if pointee.starts_with('[') && pointee.ends_with(']') && !pointee.contains(';') {
        Some(FatPointerKind::Slice)
    } else {
        None
    }
}

/// The (size, alignment) a debugger field type tag describes,
/// for the scalar, tuple, array and pointer types a union's fields take:
/// the layout the byte reinterpretation of a union value's
/// storage follows (a tuple's fields sit at the compiler's
/// own offsets — see [`tuple_field_layout`]).
fn tag_layout(ty: &str) -> Option<(usize, usize)> {
    let ty = ty.trim();
    let scalar = match ty {
        "bool" | "i8" | "u8" => Some((1, 1)),
        "char" => Some((4, 4)),
        "i16" | "u16" | "f16" => Some((2, 2)),
        "i32" | "u32" | "f32" => Some((4, 4)),
        "i64" | "u64" | "f64" | "isize" | "usize" => Some((8, 8)),
        "i128" | "u128" | "f128" => Some((16, 16)),
        _ => None,
    };
    if let Some(s) = scalar {
        return Some(s);
    }
    if thin_pointer_tag(ty) {
        return Some((8, 8));
    }
    if fat_pointer_kind(ty).is_some() {
        return Some((16, 8));
    }
    if ty.starts_with('(') && ty.ends_with(')') {
        let inner = &ty[1..ty.len() - 1];
        if inner.trim().is_empty() {
            return Some((0, 1));
        }
        let parts = split_top_level(inner);
        let (_, size, align) = tuple_field_layout(&parts)?;
        return Some((size, align));
    }
    if ty.starts_with('[') && ty.ends_with(']') {
        let inner = &ty[1..ty.len() - 1];
        let (elem, len) = inner.rsplit_once(';')?;
        let n: usize = len.trim().parse().ok()?;
        let (esize, ealign) = tag_layout(elem)?;
        let stride = align_up(esize, ealign);
        return Some((stride * n, ealign));
    }
    None
}

/// The byte layout of a tuple type's fields: each field's
/// offset (in declaration order), the total size and the
/// alignment. This is the compiler's own layout for a tuple —
/// the `rustc_abi` univariant layout under the tuple's
/// `MaybeUnsized` kind: the last field stays last (it is the
/// unsizing tail), and the remaining fields are stably sorted
/// by a group key, `trailing_zeros(max(align, size))`, capped
/// by the largest head alignment when one of them carries a
/// niche, with the larger niche preferred inside a group.
/// Verified against the pinned toolchain's `offset_of!`
/// layouts over scalar, array and nested-tuple shapes,
/// niches included.
fn tuple_field_layout(parts: &[String]) -> Option<(Vec<usize>, usize, usize)> {
    let layouts: Vec<(usize, usize)> = parts
        .iter()
        .map(|p| tag_layout(p))
        .collect::<Option<Vec<_>>>()?;
    let n = parts.len();
    if n == 0 {
        return Some((Vec::new(), 0, 1));
    }
    let niches: Vec<Option<(u64, usize)>> =
        parts.iter().map(|p| tag_niche(p)).collect();
    let head = n - 1;
    let largest_niche = niches[..head]
        .iter()
        .flatten()
        .map(|(available, _)| *available)
        .max()
        .unwrap_or(0);
    let max_head_align = layouts[..head]
        .iter()
        .map(|(_, align)| *align)
        .max()
        .unwrap_or(1);
    let group_key = |i: usize| -> u64 {
        let (size, align) = layouts[i];
        let key = u64::from(align.max(size).trailing_zeros());
        if largest_niche > 0 {
            key.min(u64::from(max_head_align.trailing_zeros()))
        } else {
            key
        }
    };
    let mut order: Vec<usize> = (0..head).collect();
    order.sort_by_key(|&i| {
        let (niche_size, niche_offset) = niches[i].unwrap_or((0, 0));
        (std::cmp::Reverse(group_key(i)), !niche_size, niche_offset)
    });
    order.push(head);
    let mut offsets = vec![0usize; n];
    let mut offset = 0usize;
    let mut align = 1usize;
    for i in order {
        let (size, field_align) = layouts[i];
        offset = align_up(offset, field_align);
        offsets[i] = offset;
        offset += size;
        align = align.max(field_align);
    }
    Some((offsets, align_up(offset, align), align))
}

/// The largest niche a debugger field type's values leave
/// unused, as (the number of unused values, the niche's offset
/// within the type), for the types whose layout ordering a
/// niche steers: `bool` and `char`, and aggregates containing
/// them. Anything else has no niche here.
fn tag_niche(ty: &str) -> Option<(u64, usize)> {
    let ty = ty.trim();
    match ty {
        // A `bool` leaves every byte but 0 and 1 unused.
        "bool" => return Some((254, 0)),
        // A `char` leaves the `u32` values above the last
        // scalar value unused, plus the surrogate range.
        "char" => return Some(((1u64 << 32) - (0x11_0000 - 0x800), 0)),
        _ => {}
    }
    if ty.starts_with('(') && ty.ends_with(')') {
        let inner = &ty[1..ty.len() - 1];
        if inner.trim().is_empty() {
            return None;
        }
        let parts = split_top_level(inner);
        let (offsets, _, _) = tuple_field_layout(&parts)?;
        let mut best: Option<(u64, usize)> = None;
        for (i, part) in parts.iter().enumerate() {
            if let Some((available, inner_offset)) = tag_niche(part) {
                if best.is_none_or(|(b, _)| available > b) {
                    best = Some((available, offsets[i] + inner_offset));
                }
            }
        }
        return best;
    }
    if ty.starts_with('[') && ty.ends_with(']') {
        let inner = &ty[1..ty.len() - 1];
        let (elem, len) = inner.rsplit_once(';')?;
        if len.trim().parse::<usize>().ok()? == 0 {
            return None;
        }
        // An array's niche is its first element's.
        return tag_niche(elem).map(|(available, _)| (available, 0));
    }
    None
}

/// Serialize a value as the little-endian bytes of its type
/// (see [`tag_layout`]): the storage a union field of that type
/// occupies. A thin pointer serializes as its address word; a
/// fat pointer as its data word and its length or vtable word
/// (the data word in the interpreter-free reading,
/// [`crate::interp::data_word`]). Types with no byte form here
/// (ADT fields, strings) make the whole reinterpretation fail,
/// and the union keeps the unrenderable form.
fn value_to_union_bytes(value: &Value, ty: &str) -> Option<Vec<u8>> {
    let ty = ty.trim();
    match ty {
        "bool" => match value {
            Value::Bool(b) => Some(vec![*b as u8]),
            _ => None,
        },
        "char" => match value {
            Value::Uint(v) => Some((*v as u32).to_le_bytes().to_vec()),
            Value::Int(v) => Some((*v as u32).to_le_bytes().to_vec()),
            _ => None,
        },
        "f32" => match value {
            Value::Float(f) => Some((*f as f32).to_bits().to_le_bytes().to_vec()),
            _ => None,
        },
        "f64" => match value {
            Value::Float(f) => Some(f.to_bits().to_le_bytes().to_vec()),
            _ => None,
        },
        "f128" => match value {
            Value::Float128(bits) => Some(bits.to_le_bytes().to_vec()),
            _ => None,
        },
        _ if int_ty_width(ty).is_some() => {
            let width = int_ty_width(ty)?;
            let bits = match value {
                Value::Int(v) => *v as u128,
                Value::Uint(v) => *v,
                _ => return None,
            };
            Some(bits.to_le_bytes()[..width].to_vec())
        }
        _ if ty.starts_with('(') && ty.ends_with(')') => {
            let Value::Tuple(items) = value else {
                return None;
            };
            let inner = &ty[1..ty.len() - 1];
            if inner.trim().is_empty() {
                return Some(Vec::new());
            }
            let parts = split_top_level(inner);
            if parts.len() != items.len() {
                return None;
            }
            // The fields sit at the compiler's tuple offsets,
            // which are not the declaration order (see
            // `tuple_field_layout`).
            let (offsets, size, _) = tuple_field_layout(&parts)?;
            let mut out = vec![0u8; size];
            for (i, (item, part)) in items.iter().zip(parts.iter()).enumerate() {
                let bytes = value_to_union_bytes(item, part)?;
                out[offsets[i]..offsets[i] + bytes.len()].copy_from_slice(&bytes);
            }
            Some(out)
        }
        _ if ty.starts_with('[') && ty.ends_with(']') => {
            let Value::Tuple(items) = value else {
                return None;
            };
            let inner = &ty[1..ty.len() - 1];
            let (elem, _) = inner.rsplit_once(';')?;
            let (esize, ealign) = tag_layout(elem)?;
            let stride = align_up(esize, ealign);
            let mut out = Vec::new();
            for item in items {
                let bytes = value_to_union_bytes(item, elem)?;
                out.extend_from_slice(&bytes);
                out.resize(out.len() + (stride - bytes.len()), 0);
            }
            Some(out)
        }
        _ if thin_pointer_tag(ty) => match value {
            Value::Ref(_) | Value::FnPtr(_) => {
                let addr = crate::interp::synthetic_address(value)?;
                Some((addr as u64).to_le_bytes().to_vec())
            }
            _ => None,
        },
        _ if fat_pointer_kind(ty).is_some() => match value {
            Value::FatRef { base, len } => {
                let mut out =
                    (crate::interp::data_word(base) as u64).to_le_bytes().to_vec();
                out.extend_from_slice(&(*len as u64).to_le_bytes());
                Some(out)
            }
            Value::DynRef {
                base,
                concrete,
                traits,
                ..
            } => {
                let mut out =
                    (crate::interp::data_word(base) as u64).to_le_bytes().to_vec();
                out.extend_from_slice(
                    &(crate::interp::vtable_word(concrete, traits) as u64).to_le_bytes(),
                );
                Some(out)
            }
            _ => None,
        },
        _ => None,
    }
}

/// Read a value of a union field's type back from the field's
/// little-endian storage bytes (see [`value_to_union_bytes`]).
fn value_from_union_bytes(bytes: &[u8], ty: &str) -> Option<Value> {
    let ty = ty.trim();
    match ty {
        "bool" => Some(Value::Bool(bytes.first().copied()? != 0)),
        "char" => {
            let v = u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?);
            Some(Value::Uint(u128::from(v)))
        }
        "f32" => {
            let bits = u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?);
            Some(Value::Float(f32::from_bits(bits) as f64))
        }
        "f64" => {
            let bits = u64::from_le_bytes(bytes.get(..8)?.try_into().ok()?);
            Some(Value::Float(f64::from_bits(bits)))
        }
        "f128" => {
            let bits = u128::from_le_bytes(bytes.get(..16)?.try_into().ok()?);
            Some(Value::Float128(bits))
        }
        _ if int_ty_width(ty).is_some() => {
            let width = int_ty_width(ty)?;
            let raw = bytes.get(..width)?;
            let mut buf = [0u8; 16];
            buf[..width].copy_from_slice(raw);
            let bits = u128::from_le_bytes(buf);
            if ty.starts_with('i') {
                let shift = 128 - width * 8;
                Some(Value::Int(((bits << shift) as i128) >> shift))
            } else {
                Some(Value::Uint(bits))
            }
        }
        _ if ty.starts_with('(') && ty.ends_with(')') => {
            let inner = &ty[1..ty.len() - 1];
            if inner.trim().is_empty() {
                return Some(Value::Tuple(Vec::new()));
            }
            let parts = split_top_level(inner);
            let (offsets, _, _) = tuple_field_layout(&parts)?;
            let mut items = Vec::with_capacity(parts.len());
            for (i, part) in parts.iter().enumerate() {
                let (esize, _) = tag_layout(part)?;
                items.push(value_from_union_bytes(
                    bytes.get(offsets[i]..offsets[i] + esize)?,
                    part,
                )?);
            }
            Some(Value::Tuple(items))
        }
        _ if ty.starts_with('[') && ty.ends_with(']') => {
            let inner = &ty[1..ty.len() - 1];
            let (elem, len) = inner.rsplit_once(';')?;
            let n: usize = len.trim().parse().ok()?;
            let (esize, ealign) = tag_layout(elem)?;
            let stride = align_up(esize, ealign);
            let mut items = Vec::with_capacity(n);
            for i in 0..n {
                items.push(value_from_union_bytes(
                    bytes.get(i * stride..i * stride + esize)?,
                    elem,
                )?);
            }
            Some(Value::Tuple(items))
        }
        _ if thin_pointer_tag(ty) => {
            let raw = bytes.get(..8)?;
            let bits = u64::from_le_bytes(raw.try_into().ok()?);
            Some(Value::Ref(crate::value::RefTarget::Address(bits)))
        }
        _ => None,
    }
}

/// The storage bytes of a union value in its aggregate form:
/// the stored field serialized (see [`value_to_union_bytes`]),
/// zero-extended to the widest field layout the shape
/// declares (a field whose layout the byte forms cannot
/// describe does not hold the others back). `None` when the
/// stored field itself has no byte form.
fn union_storage_bytes(
    shape_fields: &[sparagmos_mir::AdtDebugField],
    fields: &[Value],
    stored: usize,
) -> Option<Vec<u8>> {
    let stored_ty = &shape_fields.get(stored)?.ty;
    let mut bytes = value_to_union_bytes(&fields[stored], stored_ty)?;
    let size = shape_fields
        .iter()
        .filter_map(|f| tag_layout(&f.ty).map(|(s, _)| s))
        .max()
        .unwrap_or(0);
    bytes.resize(size, 0);
    Some(bytes)
}

/// A fat pointer value read back from a union's storage
/// words: the data word becomes an address-only pointer and
/// the second word the length. A `dyn` field is not read this
/// way: the model's vtable is an identity (the concrete type
/// the table is for), not a word the storage carries, so a
/// trait object read out of another field's storage has no
/// honest value here.
fn fat_value_from_words(bytes: &[u8], kind: FatPointerKind) -> Option<Value> {
    let data = u64::from_le_bytes(bytes.get(..8)?.try_into().ok()?);
    let word = u64::from_le_bytes(bytes.get(8..16)?.try_into().ok()?);
    match kind {
        FatPointerKind::Slice | FatPointerKind::Str => Some(Value::FatRef {
            base: Box::new(Value::Ref(crate::value::RefTarget::Address(data))),
            len: u128::from(word),
        }),
        FatPointerKind::Dyn => None,
    }
}

/// One field of a union value, produced from the union's
/// storage independently of its sibling fields: the stored
/// field as it stands, any other field reinterpreted through
/// the storage bytes (see [`union_storage_bytes`] and
/// [`fat_value_from_words`]). `None` for a field the storage
/// cannot produce -- an ADT field whose layout the byte forms
/// do not describe, a trait object read out of foreign
/// storage -- which the whole-union rendering degrades in
/// place rather than failing on: a real debugger reads every
/// field from the same storage and reports the fields it
/// cannot read inline.
fn union_field_from(
    shape_fields: &[sparagmos_mir::AdtDebugField],
    fields: &[Value],
    stored: usize,
    index: usize,
) -> Option<Value> {
    if index == stored {
        return Some(fields[stored].clone());
    }
    let requested = shape_fields.get(index)?;
    // A fat pointer read where the stored field is itself a
    // fat pointer is the stored pointer under the requested
    // field's type: the two fields share both storage words,
    // and a debugger shows the one pointer for both.
    if fat_pointer_kind(&requested.ty).is_some()
        && matches!(fields[stored], Value::FatRef { .. } | Value::DynRef { .. })
    {
        return Some(fields[stored].clone());
    }
    let bytes = union_storage_bytes(shape_fields, fields, stored)?;
    if let Some(value) = value_from_union_bytes(&bytes, &requested.ty) {
        return Some(value);
    }
    fat_pointer_kind(&requested.ty).and_then(|kind| fat_value_from_words(&bytes, kind))
}

/// The values of every declared field of a union value, one
/// entry per field: the value model stores only the written
/// field (the slots before it uninitialised, the slots after
/// it absent), while a debugger reads every field from the
/// same storage, so each field is produced from the stored
/// field's storage independently (see [`union_field_from`]).
/// A field the storage cannot produce is `None`, not a
/// failure of the whole.
pub(super) fn union_field_values(
    shape_fields: &[sparagmos_mir::AdtDebugField],
    fields: &[Value],
) -> Vec<Option<Value>> {
    let Some(stored) = fields.iter().rposition(|f| !matches!(f, Value::Uninit)) else {
        return shape_fields.iter().map(|_| None).collect();
    };
    (0..shape_fields.len())
        .map(|i| union_field_from(shape_fields, fields, stored, i))
        .collect()
}

/// One field of a union value in its byte form
/// (`Value::UnionBytes`): the storage bytes are the value, so
/// the field is read straight out of them -- the same
/// reinterpretation [`union_field_from`] applies to the
/// aggregate form's serialized storage, with a byte no write
/// covered reading as zero (the fixed reading of storage past
/// a write, as in the interpreter's own union reads).
pub(super) fn union_field_from_bytes(
    shape_fields: &[sparagmos_mir::AdtDebugField],
    bytes: &[Option<u8>],
    index: usize,
) -> Option<Value> {
    let requested = shape_fields.get(index)?;
    // The byte form covers only the written extent; storage
    // past it reads as zero, as in the aggregate form's
    // zero-extension to the widest declared field.
    let size = shape_fields
        .iter()
        .filter_map(|f| tag_layout(&f.ty).map(|(s, _)| s))
        .max()
        .unwrap_or(0);
    let mut concrete: Vec<u8> = bytes.iter().map(|b| b.unwrap_or(0)).collect();
    concrete.resize(size.max(concrete.len()), 0);
    if let Some(value) = value_from_union_bytes(&concrete, &requested.ty) {
        return Some(value);
    }
    fat_pointer_kind(&requested.ty).and_then(|kind| fat_value_from_words(&concrete, kind))
}

/// One field of a union value: the field read through the
/// storage reinterpretation (see [`union_field_from`] and
/// [`union_field_from_bytes`]), independently of the union's
/// other fields, so a sibling the storage cannot produce does
/// not prevent reading a field it can.
fn union_field_value(
    shape_fields: &[sparagmos_mir::AdtDebugField],
    value: &Value,
    index: usize,
) -> Option<Value> {
    if let Value::UnionBytes { bytes } = value {
        return union_field_from_bytes(shape_fields, bytes, index);
    }
    let fields: &[Value] = match value {
        Value::Enum { fields, .. } => fields,
        Value::Tuple(fields) => fields,
        _ => return None,
    };
    let stored = fields.iter().rposition(|f| !matches!(f, Value::Uninit))?;
    union_field_from(shape_fields, fields, stored, index)
}

/// The value of one field of the ADT value a descriptor
/// names, when that ADT is a union: the field read through
/// the storage reinterpretation above, so inspecting a
/// field agrees with printing its containing union. `None`
/// for non-ADT descriptors, non-union shapes, and storage
/// the reinterpretation cannot read (the caller then
/// projects directly).
pub(super) fn union_projected_field(
    interp: &Interpreter,
    desc: Option<&Desc>,
    value: &Value,
    index: usize,
) -> Option<Value> {
    let Desc::Adt(tag) = desc? else {
        return None;
    };
    let shape = interp.program().adt_debug_shapes.get(tag)?;
    if !matches!(shape.kind, sparagmos_mir::AdtShapeKind::Union) {
        return None;
    }
    let variant = match value {
        Value::Enum { variant, .. } => *variant as usize,
        _ => 0,
    };
    let vshape = shape.variants.get(variant)?;
    union_field_value(&vshape.fields, value, index)
}

/// lldb's expanded-children form of a `NonZero` value: the
/// scalar summary is the value itself, and the children are the
/// two one-field wrappers the DWARF describes (the struct's
/// field 0 holding the inner validity wrapper holding the
/// scalar), the inner one in compact form: `{ 0 = {0:11} }`.
/// The recorded debug shape does not describe those wrappers
/// (the value arrives already unwrapped for the summary), so
/// the form is synthesized from the wrapper structure itself.
pub(super) fn nonzero_lldb_expanded(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
) -> Option<String> {
    let (scalar, inner) = nonzero_scalar(value, desc)?;
    let wrapper = match value {
        Value::Enum { fields, .. } if fields.len() == 1 => &fields[0],
        Value::Tuple(fields) if fields.len() == 1 => &fields[0],
        _ => return None,
    };
    let inner_fields: &[Value] = match wrapper {
        Value::Enum { fields, .. } => fields,
        Value::Tuple(fields) => fields,
        _ => return None,
    };
    if inner_fields.len() != 1 || inner_fields[0] != scalar {
        return None;
    }
    let text = render_lldb_compact(interp, &scalar, Some(&inner), false, true, 1)?;
    Some(format!("{{ 0 = {{0:{text}}} }}"))
}

/// The `name: value` parts of a union's fields in gdb form,
/// each field's value produced by `value_of` (the storage
/// reinterpretation behind the aggregate and byte forms). A
/// field with no producible value, or a value this renderer
/// cannot shape, degrades in place (see
/// [`UNION_FIELD_UNREADABLE`]): gdb reports such a field
/// inline and still prints the rest of the union. A field
/// renders in the caller's mode (pretty printers applied
/// when the session enables them), like the fields of any
/// other aggregate.
fn union_parts_gdb(
    interp: &Interpreter,
    frames: &[Frame],
    vshape: &sparagmos_mir::AdtDebugVariant,
    depth: usize,
    pretty: bool,
    elements: usize,
    value_of: impl Fn(usize) -> Option<Value>,
) -> Option<Vec<String>> {
    let mut parts = Vec::with_capacity(vshape.fields.len());
    for (i, field) in vshape.fields.iter().enumerate() {
        let name = field.name.as_deref()?;
        let d = Desc::from_tag(&field.ty);
        let r = value_of(i)
            .and_then(|v| {
                if pretty {
                    render_gdb_any(interp, frames, &v, Some(&d), depth + 1, elements)
                } else {
                    render_gdb_mode(
                        interp,
                        frames,
                        &v,
                        Some(&d),
                        false,
                        None,
                        depth + 1,
                        false,
                        elements,
                    )
                }
            })
            .unwrap_or_else(|| UNION_FIELD_UNREADABLE.to_string());
        parts.push(format!("{name}: {r}"));
    }
    Some(parts)
}

/// Render one ADT value in gdb form (see [`render_gdb`]).
#[allow(clippy::too_many_arguments)]
fn render_adt_gdb(
    interp: &Interpreter,
    tag: &str,
    shape: &sparagmos_mir::AdtDebugShape,
    variant: u32,
    fields: &[Value],
    depth: usize,
    frames: &[Frame],
    pretty: bool,
    elements: usize,
) -> Option<String> {
    // The printed name is the DWARF (definition-path) form of
    // the recorded tag; the shape lookup above used the tag
    // itself, which is the table's key.
    let tag = &dwarf_type_name(tag);
    let vshape = shape.variants.get(variant as usize)?;
    if matches!(shape.kind, sparagmos_mir::AdtShapeKind::Union) {
        // A union value stores only the written field; every
        // declared field is read back from the same storage
        // (see `union_field_values`).
        let values = union_field_values(&vshape.fields, fields);
        let parts = union_parts_gdb(interp, frames, vshape, depth, pretty, elements, |i| {
            values.get(i).cloned().flatten()
        })?;
        return Some(format!("{tag} {{{}}}", parts.join(", ")));
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
    let named_parts = |interp: &Interpreter| -> Option<Vec<String>> {
        let mut parts = Vec::with_capacity(fields.len());
        for (i, field) in fields.iter().enumerate() {
            let name = vshape.fields[i].name.as_deref()?;
            let v = if pretty {
                render_gdb_any(
                    interp,
                    frames,
                    field,
                    Some(&descs[i]),
                    depth + 1,
                    elements,
                )?
            } else {
                render_gdb_mode(
                    interp,
                    frames,
                    field,
                    Some(&descs[i]),
                    false,
                    None,
                    depth + 1,
                    false,
                    elements,
                )?
            };
            parts.push(format!("{name}: {v}"));
        }
        Some(parts)
    };
    match shape.kind {
        sparagmos_mir::AdtShapeKind::Struct => {
            if fields.is_empty() {
                return Some(tag.to_string());
            }
            if named {
                let parts = named_parts(interp)?;
                Some(format!("{tag} {{{}}}", parts.join(", ")))
            } else {
                let parts = render_elements(
                    interp,
                    fields,
                    &descs,
                    None,
                    DebugDialect::Gdb,
                    depth,
                    frames,
                    pretty,
                    elements,
                )?;
                Some(format!("{tag} ({})", parts.join(", ")))
            }
        }
        sparagmos_mir::AdtShapeKind::Enum => {
            let prefix = format!("{tag}::{}", vshape.name);
            if fields.is_empty() {
                return Some(prefix);
            }
            if named {
                let parts = named_parts(interp)?;
                Some(format!("{prefix}{{{}}}", parts.join(", ")))
            } else {
                let parts = render_elements(
                    interp,
                    fields,
                    &descs,
                    None,
                    DebugDialect::Gdb,
                    depth,
                    frames,
                    pretty,
                    elements,
                )?;
                Some(format!("{prefix}({})", parts.join(", ")))
            }
        }
        sparagmos_mir::AdtShapeKind::Union => None,
    }
}

