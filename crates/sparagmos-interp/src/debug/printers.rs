//! The sysroot pretty printers: provider classification and the gdb / lldb provider forms for the standard types.

use crate::interp::{Frame, Interpreter};
use crate::value::{RefProj, RefTarget, Value};
use super::desc::{Desc, dwarf_type_name, first_generic_arg, path_text, split_top_level};
use super::render_gdb::{MAX_RENDER_DEPTH, compress_repeats, render_gdb};
use super::render_lldb::{escape_char, escape_str, lldb_type_name, render_lldb_compact};

// ---------------------------------------------------------------------------
// Pretty printers
//
// When a session enables the pretty printers (compiletest's
// default; see `DebugControl::pretty_printers`), a print of a
// value whose type the sysroot registers a provider for renders
// in the provider's form instead of the raw aggregate form. The
// classification mirrors `rust_types.py`'s type-kind table,
// matched on the DWARF (definition-path) name of the value's
// type, and each form mirrors the corresponding provider in
// `gdb_providers.py` / `lldb_providers.py`. A value with no
// provider — or a provider value whose shape the value model
// does not produce — keeps the raw form.
// ---------------------------------------------------------------------------

/// The sysroot pretty-printer kinds this renderer implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrinterKind {
    String,
    OsString,
    Vec,
    VecDeque,
    BTreeSet,
    BTreeMap,
    HashMap,
    HashSet,
    Rc,
    Arc,
}

/// The printer kind of a DWARF type name, if any: the type's
/// base name and its parent segment decide, the way
/// `rust_types.py`'s anchored path patterns do (the crate and
/// module spellings of a std type vary between its definition
/// path and its re-export paths; the last two segments are
/// stable across them).
fn printer_kind(dwarf_name: &str) -> Option<PrinterKind> {
    let base = dwarf_name.split('<').next()?.trim();
    if !(base.starts_with("alloc::")
        || base.starts_with("std::")
        || base.starts_with("core::"))
    {
        return None;
    }
    let mut segs = base.rsplit("::");
    let name = segs.next()?;
    let parent = segs.next().unwrap_or("");
    Some(match (name, parent) {
        ("String", "string") => PrinterKind::String,
        ("OsString", "os_str") => PrinterKind::OsString,
        ("Vec", "vec") => PrinterKind::Vec,
        ("VecDeque", "vec_deque") | ("VecDeque", "collections") => {
            PrinterKind::VecDeque
        }
        ("BTreeSet", "set") | ("BTreeSet", "collections") => {
            PrinterKind::BTreeSet
        }
        ("BTreeMap", "map") | ("BTreeMap", "collections") => {
            PrinterKind::BTreeMap
        }
        ("HashMap", "map") | ("HashMap", "collections") => {
            PrinterKind::HashMap
        }
        ("HashSet", "set") | ("HashSet", "collections") => {
            PrinterKind::HashSet
        }
        ("Rc", "rc") => PrinterKind::Rc,
        ("Arc", "sync") => PrinterKind::Arc,
        _ => return None,
    })
}

/// The elements of a sequence value in this model: an element
/// tuple (an array value or a materialised slice), an ADT
/// element aggregate (a vector's storage form), or a byte block
/// (each byte an element).
fn sequence_values(value: &Value) -> Option<Vec<Value>> {
    match value {
        Value::Tuple(items) => Some(items.clone()),
        Value::Enum { fields, .. } => Some(fields.clone()),
        // A sparse sequence's stored elements (its declared
        // length runs past them; see `Value::SparseVec`).
        Value::SparseVec { elems, .. } => Some(elems.clone()),
        Value::Bytes(bytes) => Some(
            bytes
                .iter()
                .map(|b| Value::Uint(u128::from(*b)))
                .collect(),
        ),
        Value::Str(text) => Some(
            text.bytes()
                .map(|b| Value::Uint(u128::from(b)))
                .collect(),
        ),
        _ => None,
    }
}

/// gdb's `print elements` limit for a printer's children.
const GDB_PRINT_ELEMENTS: usize = 200;

/// Render a value in gdb form with the pretty printers applied
/// where a provider exists (the raw form otherwise).
fn render_gdb_any(
    interp: &Interpreter,
    frames: &[Frame],
    value: &Value,
    desc: Option<&Desc>,
    depth: usize,
) -> Option<String> {
    render_pretty_gdb(interp, frames, value, desc, None, depth)
        .or_else(|| render_gdb(interp, value, desc, false, None, depth))
}

/// Resolve a printed slice value to the value holding its
/// elements: follow the fat pointer, and keep following while
/// the pointee is itself a fat pointer — a slice formed over
/// another slice value (a vector's `&vec[..]`, whose base is
/// the vector's own slice value) denotes the same elements
/// through every layer. The peeling is bounded; a pointee
/// still fat past the bound is left as it stands (and the
/// caller's sequence reading then declines it, as it did
/// before peeling existed).
fn slice_pointee(
    interp: &Interpreter,
    frames: &[Frame],
    value: &Value,
) -> Option<Value> {
    let mut pointee = interp
        .read_resolved(frames, &interp.deref_value(frames, value).ok()?)
        .ok()?;
    for _ in 0..8 {
        match &pointee {
            Value::FatRef { base, .. } => match &**base {
                // The elements are the base aggregate itself:
                // the fat pointer is self-contained over it.
                Value::Tuple(_) | Value::SparseVec { .. } => {
                    return Some((**base).clone());
                }
                // A reference base: follow the fat pointer to
                // what it denotes and judge that value.
                Value::Ref(_) => {
                    pointee = interp
                        .read_resolved(
                            frames,
                            &interp.deref_value(frames, &pointee).ok()?,
                        )
                        .ok()?;
                }
                _ => return Some(pointee),
            },
            _ => return Some(pointee),
        }
    }
    Some(pointee)
}

/// The elements a sequence rendering walks: the stored
/// elements, extended with uninitialised slots up to the
/// declared length when the sequence is sparse (see
/// `Value::SparseVec`) — bounded just past the print-elements
/// limit, since the renderer truncates there anyway. A fully
/// stored sequence passes through unchanged.
fn render_sequence_elems(stored: &[Value], declared: u128) -> Vec<Value> {
    let mut elems = stored.to_vec();
    let want = usize::try_from(declared)
        .unwrap_or(usize::MAX)
        .min(GDB_PRINT_ELEMENTS + 1);
    if elems.len() < want {
        elems.resize(want, Value::Uninit);
    }
    elems
}

/// Render a printer's array-hint children block (`{a, b, c}`),
/// with gdb's repeat compression and its truncation marker
/// past the `print elements` limit.
fn render_printer_children(
    interp: &Interpreter,
    frames: &[Frame],
    elems: &[Value],
    elem_desc: Option<&Desc>,
    depth: usize,
) -> Option<String> {
    let shown = elems.len().min(GDB_PRINT_ELEMENTS);
    let mut rendered = Vec::with_capacity(shown);
    for elem in &elems[..shown] {
        rendered.push(render_gdb_any(interp, frames, elem, elem_desc, depth + 1)?);
    }
    let mut body = compress_repeats(&rendered);
    if elems.len() > shown {
        if !body.is_empty() {
            body.push_str(", ");
        }
        body.push_str("...");
    }
    Some(format!("{{{body}}}"))
}

/// Render one printed value in the sysroot gdb pretty
/// printers' form, when its type has a provider this renderer
/// implements. `tag_hint` is the printed variable's declared
/// type tag, for the forms that name the type (a slice's
/// `&[T]` / `&mut [T]` head).
pub(super) fn render_pretty_gdb(
    interp: &Interpreter,
    frames: &[Frame],
    value: &Value,
    desc: Option<&Desc>,
    tag_hint: Option<&str>,
    depth: usize,
) -> Option<String> {
    if depth > MAX_RENDER_DEPTH {
        return None;
    }
    match desc {
        Some(Desc::Ref(inner) | Desc::RefMut(inner)) => match inner.as_ref() {
            Desc::Slice(elem) => {
                let pointee = slice_pointee(interp, frames, value)?;
                // A sparse pointee declares its length past its
                // stored elements (see `Value::SparseVec`): the
                // size is the declared length, and the children
                // render from the stored elements plus the
                // uninitialised tail.
                let (elems, declared) = match &pointee {
                    Value::SparseVec { elems, len, .. } => {
                        (elems.clone(), *len)
                    }
                    _ => {
                        let elems = sequence_values(&pointee)?;
                        let len = elems.len() as u128;
                        (elems, len)
                    }
                };
                let ty = match tag_hint {
                    Some(tag) if tag.trim_start().starts_with('&') => {
                        tag.trim().to_string()
                    }
                    _ => return None,
                };
                let head = format!("{ty}(size={declared})");
                if declared == 0 {
                    Some(head)
                } else {
                    let elems = render_sequence_elems(&elems, declared);
                    let block = render_printer_children(
                        interp,
                        frames,
                        &elems,
                        Some(elem),
                        depth,
                    )?;
                    Some(format!("{head} = {block}"))
                }
            }
            Desc::Str => {
                let resolved = interp.deref_value(frames, value).ok()?;
                let pointee = interp.read_resolved(frames, &resolved).ok()?;
                let text = match &pointee {
                    Value::Str(s) => s.clone(),
                    Value::Bytes(b) => {
                        String::from_utf8_lossy(b).into_owned()
                    }
                    _ => return None,
                };
                Some(format!("\"{}\"", escape_str(&text)))
            }
            _ => None,
        },
        Some(Desc::Adt(tag)) => {
            let kind = printer_kind(&dwarf_type_name(tag))?;
            render_printer_adt(interp, frames, kind, tag, value, depth)
        }
        _ => None,
    }
}

/// Render an ADT value in its gdb provider's form.
fn render_printer_adt(
    interp: &Interpreter,
    frames: &[Frame],
    kind: PrinterKind,
    tag: &str,
    value: &Value,
    depth: usize,
) -> Option<String> {
    match kind {
        PrinterKind::String | PrinterKind::OsString => {
            let text = match value {
                Value::Str(s) => Some(s.clone()),
                _ => path_text(value),
            }?;
            Some(format!("\"{}\"", escape_str(&text)))
        }
        PrinterKind::Vec => {
            let elem =
                first_generic_arg(tag).map(|t| Desc::from_tag(&t));
            match value {
                // A sparse vector (see `Value::SparseVec`): the
                // size is the declared length, and the children
                // render from the stored elements plus the
                // uninitialised tail, bounded like any long
                // vector's.
                Value::SparseVec { elems, len, .. } => {
                    let head = format!("Vec(size={len})");
                    if *len == 0 {
                        Some(head)
                    } else {
                        let elems = render_sequence_elems(elems, *len);
                        let block = render_printer_children(
                            interp,
                            frames,
                            &elems,
                            elem.as_ref(),
                            depth,
                        )?;
                        Some(format!("{head} = {block}"))
                    }
                }
                Value::Enum { fields, .. } => {
                    let head = format!("Vec(size={})", fields.len());
                    if fields.is_empty() {
                        Some(head)
                    } else {
                        let block = render_printer_children(
                            interp,
                            frames,
                            fields,
                            elem.as_ref(),
                            depth,
                        )?;
                        Some(format!("{head} = {block}"))
                    }
                }
                // A vector place the program has not written
                // yet (the variable is declared but not yet
                // assigned at the stop): the provider reads the
                // uninitialised buffer's length as whatever the
                // memory holds, and the suite's check for this
                // case accepts any size; the empty form is the
                // one honest rendering available here.
                Value::Uninit => Some("Vec(size=0)".to_string()),
                _ => None,
            }
        }
        PrinterKind::Rc | PrinterKind::Arc => {
            let (strong, weak, pointee, elem) = rc_parts(interp, tag, value)?;
            let v = render_gdb_any(
                interp,
                frames,
                &pointee,
                elem.as_ref(),
                depth + 1,
            )?;
            let name = if kind == PrinterKind::Rc {
                "Rc"
            } else {
                "Arc"
            };
            Some(format!("{name}(strong={strong}, weak={weak}) = {{value = {v}, strong = {strong}, weak = {weak}}}"))
        }
        PrinterKind::VecDeque => {
            let elem =
                first_generic_arg(tag).map(|t| Desc::from_tag(&t));
            let elems = interp.vec_deque_elements(value).ok()??;
            let head = format!("VecDeque(size={})", elems.len());
            if elems.is_empty() {
                Some(head)
            } else {
                let block = render_printer_children(
                    interp,
                    frames,
                    &elems,
                    elem.as_ref(),
                    depth,
                )?;
                Some(format!("{head} = {block}"))
            }
        }
        PrinterKind::BTreeSet => {
            let (entries, size) = btree_entries(interp, value, true)?;
            let elem =
                first_generic_arg(tag).map(|t| Desc::from_tag(&t));
            let head = format!("BTreeSet(size={size})");
            if entries.is_empty() {
                Some(head)
            } else {
                let keys: Vec<Value> =
                    entries.into_iter().map(|(k, _)| k).collect();
                let block = render_printer_children(
                    interp,
                    frames,
                    &keys,
                    elem.as_ref(),
                    depth,
                )?;
                Some(format!("{head} = {block}"))
            }
        }
        PrinterKind::BTreeMap => {
            let (entries, size) = btree_entries(interp, value, false)?;
            let args = generic_args(tag);
            let key_desc = args
                .first()
                .map(|t| Desc::from_tag(t));
            let val_desc = args
                .get(1)
                .map(|t| Desc::from_tag(t));
            let head = format!("BTreeMap(size={size})");
            if entries.is_empty() {
                Some(head)
            } else {
                let block = render_printer_map_children(
                    interp,
                    frames,
                    &entries,
                    key_desc.as_ref(),
                    val_desc.as_ref(),
                    depth,
                )?;
                Some(format!("{head} = {block}"))
            }
        }
        PrinterKind::HashMap => {
            let (entries, size) = hash_entries(interp, value, false)?;
            let args = generic_args(tag);
            let key_desc = args
                .first()
                .map(|t| Desc::from_tag(t));
            let val_desc = args
                .get(1)
                .map(|t| Desc::from_tag(t));
            let head = format!("HashMap(size={size})");
            if entries.is_empty() {
                Some(head)
            } else {
                let block = render_printer_map_children(
                    interp,
                    frames,
                    &entries,
                    key_desc.as_ref(),
                    val_desc.as_ref(),
                    depth,
                )?;
                Some(format!("{head} = {block}"))
            }
        }
        PrinterKind::HashSet => {
            let (entries, size) = hash_entries(interp, value, true)?;
            let elem =
                first_generic_arg(tag).map(|t| Desc::from_tag(&t));
            let head = format!("HashSet(size={size})");
            if entries.is_empty() {
                Some(head)
            } else {
                let keys: Vec<Value> =
                    entries.into_iter().map(|(k, _)| k).collect();
                let block = render_printer_children(
                    interp,
                    frames,
                    &keys,
                    elem.as_ref(),
                    depth,
                )?;
                Some(format!("{head} = {block}"))
            }
        }
    }
}

/// All generic arguments of a rendered instance tag.
fn generic_args(tag: &str) -> Vec<String> {
    let Some(open) = tag.find('<') else {
        return Vec::new();
    };
    let Some(inner) = tag.get(open + 1..tag.len().saturating_sub(1)) else {
        return Vec::new();
    };
    split_top_level(inner)
        .into_iter()
        .filter(|arg| !arg.is_empty())
        .collect()
}

/// The fields of an aggregate value (an ADT value in this
/// model), when it is one.
fn agg_fields(value: &Value) -> Option<&Vec<Value>> {
    match value {
        Value::Enum { fields, .. } => Some(fields),
        _ => None,
    }
}

/// The integer an aggregate field holds, when it is one.
fn agg_uint(value: &Value) -> Option<u128> {
    match value {
        Value::Uint(n) => Some(*n),
        Value::Int(n) => Some(*n as u128),
        _ => None,
    }
}

/// The value under one single-field aggregate layer (the
/// `MaybeUninit` shell a collection element is stored in);
/// any other value stands as it is.
fn unwrap_element(value: &Value) -> Value {
    match value {
        Value::Enum { fields, .. } if fields.len() == 1 => {
            fields[0].clone()
        }
        _ => value.clone(),
    }
}

/// The allocation a pointer-shaped aggregate denotes: the
/// chain of single-field wrappers (`NonNull`, `Unique`, a
/// box shell) unwrapped down to the reference they hold.
/// Returns the allocation id and the reference's projection.
fn wrapped_alloc_ref(value: &Value) -> Option<(u64, Vec<RefProj>)> {
    let mut cur = value;
    loop {
        match cur {
            Value::Ref(RefTarget::Alloc { id, projection }) => {
                return Some((*id, projection.clone()));
            }
            Value::Enum { fields, .. } if fields.len() == 1 => {
                cur = &fields[0];
            }
            _ => return None,
        }
    }
}

/// Render a map provider's children block (`{[k] = v, ...}`),
/// with the same repeat compression and truncation as the
/// array-hint children.
fn render_printer_map_children(
    interp: &Interpreter,
    frames: &[Frame],
    entries: &[(Value, Value)],
    key_desc: Option<&Desc>,
    val_desc: Option<&Desc>,
    depth: usize,
) -> Option<String> {
    let shown = entries.len().min(GDB_PRINT_ELEMENTS);
    let mut rendered = Vec::with_capacity(shown);
    for (k, v) in &entries[..shown] {
        let ks = render_gdb_any(interp, frames, k, key_desc, depth + 1)?;
        let vs = render_gdb_any(interp, frames, v, val_desc, depth + 1)?;
        rendered.push(format!("[{ks}] = {vs}"));
    }
    let mut body = compress_repeats(&rendered);
    if entries.len() > shown {
        if !body.is_empty() {
            body.push_str(", ");
        }
        body.push_str("...");
    }
    Some(format!("{{{body}}}"))
}

/// The `(key, value)` entries of a B-tree collection value
/// (a `BTreeMap`, or a `BTreeSet` through its inner map), in
/// the tree's own order, with the collection's length. The
/// walk reads the nodes the way the tree's own code lays
/// them out: a node allocation's element is the node
/// aggregate under its single-field shells; a leaf holds
/// `len` keys and values in its last two fields; an internal
/// node interleaves `len + 1` child edges with its keys, the
/// edges held in the node's second field (the flattened
/// storage form the cast between the leaf and internal node
/// types produces) or after the data aggregate (the nested
/// form). Elements sit in their `MaybeUninit` shells in
/// storage; the shells are not part of the values.
fn btree_entries(
    interp: &Interpreter,
    value: &Value,
    is_set: bool,
) -> Option<(Vec<(Value, Value)>, u128)> {
    let map = if is_set {
        agg_fields(value)?.first()?
    } else {
        value
    };
    let map_fields = agg_fields(map)?;
    let root = map_fields.first()?;
    let size = agg_uint(map_fields.get(1)?)?;
    let root_fields = agg_fields(root)?;
    if root_fields.is_empty() {
        // `None` root: the empty tree.
        return Some((Vec::new(), size));
    }
    let noderef = root_fields.first()?;
    let noderef_fields = agg_fields(noderef)?;
    let height = usize::try_from(agg_uint(noderef_fields.first()?)?)
        .ok()?;
    let mut entries = Vec::new();
    btree_walk(
        interp,
        noderef_fields.get(1)?,
        height,
        size,
        &mut entries,
    )?;
    if entries.len() as u128 != size {
        return None;
    }
    Some((entries, size))
}

/// One node of the B-tree walk: append the subtree's entries
/// to `out`, in order. `node_ptr` is the pointer-shaped value
/// naming the node; `height` is its height (0 = leaf).
fn btree_walk(
    interp: &Interpreter,
    node_ptr: &Value,
    height: usize,
    total: u128,
    out: &mut Vec<(Value, Value)>,
) -> Option<()> {
    let (id, _) = wrapped_alloc_ref(node_ptr)?;
    let base = interp.alloc_base(id).ok()?;
    let node = btree_node_value(&base)?;
    let fields = agg_fields(&node)?;
    let (len, keys, vals, edges) = if fields.len() == 5 {
        let len = usize::try_from(agg_uint(fields.get(2)?)?).ok()?;
        let edges = if height > 0 {
            Some(fields.get(1)?.clone())
        } else {
            None
        };
        (
            len,
            fields.get(3)?.clone(),
            fields.get(4)?.clone(),
            edges,
        )
    } else if fields.len() == 2 {
        let data = btree_node_value(fields.first()?)?;
        let data_fields = agg_fields(&data)?;
        if data_fields.len() != 5 {
            return None;
        }
        let len =
            usize::try_from(agg_uint(data_fields.get(2)?)?).ok()?;
        (
            len,
            data_fields.get(3)?.clone(),
            data_fields.get(4)?.clone(),
            Some(fields.get(1)?.clone()),
        )
    } else {
        return None;
    };
    let keys = match &keys {
        Value::Tuple(items) => items,
        _ => return None,
    };
    let vals = match &vals {
        Value::Tuple(items) => items,
        _ => return None,
    };
    // A slot within the length that was never stored is an
    // elided zero-sized element (the representation the
    // execution walker reconstructs with `btree_fill_elided`
    // in builtins.rs): the stored array is shorter than the
    // declared length, and every present slot holds the
    // element type's sole value.
    let keys = btree_fill_elided_slots(keys, len)?;
    let vals = btree_fill_elided_slots(vals, len)?;
    let pairs: Vec<(Value, Value)> = (0..len)
        .map(|i| (unwrap_element(&keys[i]), unwrap_element(&vals[i])))
        .collect();
    if height == 0 {
        out.extend(pairs);
        return Some(());
    }
    let edges = edges?;
    let edge_items = match &edges {
        Value::Tuple(items) => items.clone(),
        _ => return None,
    };
    if edge_items.len() < len + 1 {
        return None;
    }
    for (i, pair) in pairs.into_iter().enumerate() {
        btree_walk(interp, &edge_items[i], height - 1, total, out)?;
        out.push(pair);
        if out.len() as u128 > total {
            return None;
        }
    }
    btree_walk(interp, &edge_items[len], height - 1, total, out)?;
    Some(())
}

/// Whether a stored slot's value is zero-sized in form: a
/// unit, an empty aggregate, or a wrapper over one (the same
/// judgment as `btree_zst_shaped` in builtins.rs). A zero-sized
/// element type has exactly one value, and every stored slot
/// of it has this shape.
fn btree_zst_shaped(value: &Value) -> bool {
    match value {
        Value::Unit => true,
        Value::Tuple(items) => items.iter().all(btree_zst_shaped),
        Value::Enum { fields, .. } => fields.iter().all(btree_zst_shaped),
        _ => false,
    }
}

/// Fill a B-tree node's key/value array out to the declared
/// length, as the execution walker does (`btree_fill_elided`
/// in builtins.rs): a slot within the length that was never
/// stored is an elided zero-sized element -- the write
/// machinery stores nothing for a value with no leaves, so a
/// zero-sized element type (a set's marker value, a unit key)
/// leaves the array shorter than the length, and every
/// present slot holds the type's sole value in the same form
/// the missing slots hold. An array with no stored slots at
/// all holds only elided elements; their sole value is the
/// unit the model stores for it. A short array whose present
/// slots are not all zero-sized is a genuine hole in the node
/// and stays unrenderable.
fn btree_fill_elided_slots(slots: &[Value], len: usize) -> Option<Vec<Value>> {
    if slots.len() >= len {
        return Some(slots.to_vec());
    }
    if !slots.iter().all(btree_zst_shaped) {
        return None;
    }
    let sole = slots.first().cloned().unwrap_or(Value::Unit);
    let mut filled = slots.to_vec();
    filled.resize(len, sole);
    Some(filled)
}

/// The node aggregate an allocation (or a data field) holds:
/// the element value under its single-field shells, down to
/// the multi-field node aggregate.
fn btree_node_value(stored: &Value) -> Option<Value> {
    let mut cur = match stored {
        Value::Tuple(items) => items.first()?.clone(),
        other => other.clone(),
    };
    for _ in 0..8 {
        match &cur {
            Value::Enum { fields, .. } if fields.len() == 1 => {
                cur = fields[0].clone();
            }
            _ => break,
        }
    }
    Some(cur)
}

/// The `(key, value)` entries of a hash collection value (a
/// `HashMap`, or a `HashSet` through its inner map), in
/// bucket order, with the table's item count. The walk
/// reads the raw table the way hashbrown lays it out in
/// this model: one allocation holds the buckets and the
/// control bytes; the control pointer's projection names
/// the slot where the control bytes begin, and the buckets
/// sit in the slots before it, addressed from the control
/// pointer the way the table's own pointer arithmetic
/// addresses them: bucket `i`'s pair ends one bucket width
/// further from the control bytes than bucket `i + 1`'s.
/// The region is padded before its first bucket up to the
/// control region's alignment, so the control offset only
/// bounds the bucket width: the width is the region's even
/// split when there is no padding, and otherwise one of
/// the narrower widths whose padded region ends at the
/// same offset. A candidate width is accepted only when
/// every pair-shaped slot in the region sits at one of
/// its bucket starts, so a deleted bucket's stale pair
/// cannot be read as a live entry at a wider candidate's
/// offset. A bucket is live when its slot holds a
/// pair and its control byte is a hash byte (below the
/// empty/deleted sentinels).
fn hash_entries(
    interp: &Interpreter,
    value: &Value,
    is_set: bool,
) -> Option<(Vec<(Value, Value)>, u128)> {
    let base = agg_fields(value)?.first()?;
    let map = if is_set {
        agg_fields(base)?.first()?
    } else {
        base
    };
    let table = agg_fields(map)?.get(1)?;
    let inner = agg_fields(table)?.first()?;
    let inner_fields = agg_fields(inner)?;
    if inner_fields.len() != 4 {
        return None;
    }
    let mask = agg_uint(inner_fields.first()?)?;
    let size = agg_uint(inner_fields.get(3)?)?;
    if size == 0 {
        return Some((Vec::new(), 0));
    }
    let (id, projection) = wrapped_alloc_ref(inner_fields.get(1)?)?;
    let ctrl_index = projection.iter().rev().find_map(|step| {
        match step {
            RefProj::Index(i) => Some(*i as usize),
            _ => None,
        }
    })?;
    let num_buckets = usize::try_from(mask + 1).ok()?;
    if num_buckets == 0 {
        return None;
    }
    let stored = interp.alloc_base(id).ok()?;
    let slots = match &stored {
        Value::Tuple(items) => items,
        _ => return None,
    };
    // One walk at a candidate bucket width: the entries in
    // bucket order, when every live control byte's bucket
    // holds a pair and the live count is the item count.
    let walk = |width: usize| -> Option<Vec<(Value, Value)>> {
        let mut entries = Vec::new();
        for i in 0..num_buckets {
            let ctrl = slots.get(ctrl_index + i)?;
            let ctrl_byte = match unwrap_element(ctrl) {
                Value::Uint(b) => b,
                _ => return None,
            };
            if ctrl_byte >= 0x80 {
                continue;
            }
            let pos = ctrl_index.checked_sub((i + 1) * width)?;
            let pair = match slots.get(pos)? {
                Value::Tuple(items) if items.len() == 2 => items,
                _ => return None,
            };
            entries.push((pair[0].clone(), pair[1].clone()));
        }
        if entries.len() as u128 != size {
            return None;
        }
        // The width must also account for every pair-shaped
        // slot in the region: a pair this partition does not
        // place at a bucket start belongs to another width's
        // buckets — a deleted bucket's stale pair sits at its
        // own bucket start, and a wider candidate reads it as
        // a nearer bucket's entry, passing the shape and
        // count checks above with a removed entry in place of
        // a live one. Only the true width places every pair
        // in the region at one of its bucket starts.
        let region_start = ctrl_index - num_buckets * width;
        for j in region_start..ctrl_index {
            let is_pair = matches!(
                slots.get(j),
                Some(Value::Tuple(items)) if items.len() == 2
            );
            if is_pair && (ctrl_index - j) % width != 0 {
                return None;
            }
        }
        Some(entries)
    };
    // The even split first (exact for unpadded tables), then
    // the remaining widths whose region — padded before its
    // first bucket by less than the control alignment, at
    // most 64 slots — ends at the same control offset,
    // widest first.
    let mut widths: Vec<usize> = Vec::new();
    if ctrl_index % num_buckets == 0 && ctrl_index / num_buckets > 0 {
        widths.push(ctrl_index / num_buckets);
    }
    let widest = ctrl_index / num_buckets;
    let narrowest = if ctrl_index >= 64 {
        (ctrl_index - 64) / num_buckets + 1
    } else {
        1
    };
    for w in (narrowest..=widest).rev() {
        if !widths.contains(&w) {
            widths.push(w);
        }
    }
    for width in widths {
        if let Some(entries) = walk(width) {
            return Some((entries, size));
        }
    }
    None
}

/// The counts and pointee of a reference-counted handle value
/// (`Rc` / `Arc`): the strong count, the weak count as the
/// providers display it (the stored count less the
/// allocation's own implicit weak reference), the pointee
/// value, and the pointee's descriptor from the instance tag.
fn rc_parts(
    interp: &Interpreter,
    tag: &str,
    value: &Value,
) -> Option<(u128, u128, Value, Option<Desc>)> {
    let id = crate::interp::rc_handle_id(value)?;
    let inner = interp.alloc_base(id).ok()?;
    let strong = rc_count(&inner, 0)?;
    let weak = rc_count(&inner, 1)?.saturating_sub(1);
    let pointee = match &inner {
        Value::Enum { fields, .. } => fields.get(2)?.clone(),
        _ => return None,
    };
    let elem = first_generic_arg(tag).map(|t| Desc::from_tag(&t));
    Some((strong, weak, pointee, elem))
}

/// One reference count of an `RcInner` / `ArcInner` aggregate:
/// field `field` (0 = strong, 1 = weak) is a `Cell<usize>` /
/// `AtomicUsize` wrapper around an `UnsafeCell<usize>` holding
/// the count (the same reading the interpreter's own count
/// adjustments make).
fn rc_count(inner: &Value, field: usize) -> Option<u128> {
    fn unwrap1(v: &Value) -> Option<&Value> {
        match v {
            Value::Enum { fields, .. } if fields.len() == 1 => {
                Some(&fields[0])
            }
            _ => None,
        }
    }
    let fields = match inner {
        Value::Enum { fields, .. } => fields,
        _ => return None,
    };
    let leaf = unwrap1(unwrap1(fields.get(field)?)?)?;
    match leaf {
        Value::Uint(n) => Some(*n),
        _ => None,
    }
}

/// The lldb provider summary line for one printed value, when
/// its type has a synthetic-child provider with a summary form
/// this renderer implements (a slice, a string slice, a counted
/// handle): the `(Type) name = <summary> { <children> }` line
/// lldb prints for it.
pub(super) fn render_lldb_provider_line(
    interp: &Interpreter,
    frames: &[Frame],
    value: &Value,
    desc: Option<&Desc>,
    tag_hint: Option<&str>,
    expr: &str,
) -> Option<String> {
    match desc {
        Some(Desc::Ref(inner) | Desc::RefMut(inner)) => match inner.as_ref() {
            Desc::Slice(elem) => {
                let pointee = slice_pointee(interp, frames, value)?;
                let elems = sequence_values(&pointee)?;
                // A sparse pointee's size is its declared
                // length, past the stored elements the children
                // render from (see `Value::SparseVec`).
                let size = match &pointee {
                    Value::SparseVec { len, .. } => *len,
                    _ => elems.len() as u128,
                };
                let ty = tag_hint?.trim();
                let mut parts = Vec::with_capacity(elems.len());
                for (i, e) in elems.iter().enumerate() {
                    let v = render_lldb_compact(
                        interp,
                        e,
                        Some(elem),
                        false,
                        true,
                        1,
                    )?;
                    parts.push(format!("[{i}] = {v}"));
                }
                if parts.is_empty() {
                    Some(format!("({ty}) {expr} = size={size}"))
                } else {
                    let children = format!("{{ {} }}", parts.join(" "));
                    Some(format!(
                        "({ty}) {expr} = size={} {children}",
                        size
                    ))
                }
            }
            Desc::Str => {
                let resolved = interp.deref_value(frames, value).ok()?;
                let pointee = interp.read_resolved(frames, &resolved).ok()?;
                let text = match &pointee {
                    Value::Str(s) => s.clone(),
                    Value::Bytes(b) => {
                        String::from_utf8_lossy(b).into_owned()
                    }
                    _ => return None,
                };
                let ty = tag_hint?.trim();
                // The slice provider formats each element as
                // a character (`eFormatChar`): the plain
                // quoted character, not the scalar char form.
                let mut parts = Vec::with_capacity(text.len());
                for (i, c) in text.chars().enumerate() {
                    parts.push(format!("[{i}] = '{}'", escape_char(c)));
                }
                let children = if parts.is_empty() {
                    "{}".to_string()
                } else {
                    format!("{{ {} }}", parts.join(" "))
                };
                Some(format!(
                    "({ty}) {expr} = \"{}\" {children}",
                    escape_str(&text)
                ))
            }
            _ => None,
        },
        Some(Desc::Adt(tag)) => {
            let kind = printer_kind(&dwarf_type_name(tag))?;
            match kind {
                PrinterKind::Rc | PrinterKind::Arc => {
                    let (strong, weak, pointee, elem) =
                        rc_parts(interp, tag, value)?;
                    let v = render_lldb_compact(
                        interp,
                        &pointee,
                        elem.as_ref(),
                        false,
                        true,
                        1,
                    )?;
                    // The `Rc` provider's child is named
                    // `value`, the `Arc` provider's `data`.
                    let child = if kind == PrinterKind::Rc {
                        "value"
                    } else {
                        "data"
                    };
                    let ty = lldb_type_name(&dwarf_type_name(tag));
                    Some(format!("({ty}) {expr} = strong={strong}, weak={weak} {{ {child} = {v} }}"))
                }
                PrinterKind::VecDeque => {
                    let elem = first_generic_arg(tag)
                        .map(|t| Desc::from_tag(&t));
                    let elems =
                        interp.vec_deque_elements(value).ok()??;
                    let ty = lldb_type_name(&dwarf_type_name(tag));
                    let children = lldb_children(interp, elems.iter().map(|e| {
                        render_lldb_compact(
                            interp,
                            e,
                            elem.as_ref(),
                            false,
                            true,
                            1,
                        )
                    }))?;
                    Some(format!(
                        "({ty}) {expr} = size={} {children}",
                        elems.len()
                    ))
                }
                PrinterKind::HashMap => {
                    let (entries, size) =
                        hash_entries(interp, value, false)?;
                    let args = generic_args(tag);
                    let pair_desc = Desc::Tuple(vec![
                        args.first().map(|t| Desc::from_tag(t))?,
                        args.get(1).map(|t| Desc::from_tag(t))?,
                    ]);
                    let ty = lldb_type_name(&dwarf_type_name(tag));
                    let children = lldb_children(interp, entries.iter().map(|(k, v)| {
                        render_lldb_compact(
                            interp,
                            &Value::Tuple(vec![k.clone(), v.clone()]),
                            Some(&pair_desc),
                            false,
                            true,
                            1,
                        )
                    }))?;
                    Some(format!(
                        "({ty}) {expr} = size={size} {children}"
                    ))
                }
                PrinterKind::HashSet => {
                    let (entries, size) =
                        hash_entries(interp, value, true)?;
                    let elem = first_generic_arg(tag)
                        .map(|t| Desc::from_tag(&t));
                    let ty = lldb_type_name(&dwarf_type_name(tag));
                    let children = lldb_children(interp, entries.iter().map(|(k, _)| {
                        render_lldb_compact(
                            interp,
                            k,
                            elem.as_ref(),
                            false,
                            true,
                            1,
                        )
                    }))?;
                    Some(format!(
                        "({ty}) {expr} = size={size} {children}"
                    ))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// The children block of an lldb provider line: the
/// `[i] = value` parts in braces, or the empty block.
fn lldb_children(
    _interp: &Interpreter,
    parts: impl Iterator<Item = Option<String>>,
) -> Option<String> {
    let mut rendered = Vec::new();
    for part in parts {
        rendered.push(part?);
    }
    if rendered.is_empty() {
        Some("{}".to_string())
    } else {
        let parts: Vec<String> = rendered
            .into_iter()
            .enumerate()
            .map(|(i, v)| format!("[{i}] = {v}"))
            .collect();
        Some(format!("{{ {} }}", parts.join(" ")))
    }
}


#[cfg(test)]
mod btree_elided_tests {
    use super::*;
    use crate::ir::Program;

    fn leaf_node(len: u128, keys: Vec<Value>, vals: Vec<Value>) -> Value {
        Value::Enum {
            variant: 0,
            fields: vec![
                Value::Uninit,
                Value::Uint(0),
                Value::Uint(len),
                Value::Tuple(keys),
                Value::Tuple(vals),
            ],
        }
    }

    fn map_with_node(node: Value, size: u128) -> (Interpreter<'static>, Value) {
        let program: &'static Program = Box::leak(Box::new(Program::default()));
        let interp = Interpreter::new(program);
        let id = interp.alloc_insert(node);
        let noderef = Value::Enum {
            variant: 0,
            fields: vec![
                Value::Uint(0),
                Value::Enum {
                    variant: 0,
                    fields: vec![Value::Ref(RefTarget::Alloc {
                        id,
                        projection: Vec::new(),
                    })],
                },
            ],
        };
        let root = Value::Enum {
            variant: 1,
            fields: vec![noderef],
        };
        let map = Value::Enum {
            variant: 0,
            fields: vec![root, Value::Uint(size)],
        };
        (interp, map)
    }

    /// A map's unit values are never stored: the node's value
    /// array is empty at the declared length, and the entries
    /// still read as the pairs the tree holds.
    #[test]
    fn elided_value_slots_are_filled() {
        let (interp, map) = map_with_node(
            leaf_node(2, vec![Value::Int(7), Value::Int(8)], Vec::new()),
            2,
        );
        let (entries, size) = btree_entries(&interp, &map, false).expect("entries");
        assert_eq!(size, 2);
        assert_eq!(
            entries,
            vec![
                (Value::Int(7), Value::Unit),
                (Value::Int(8), Value::Unit),
            ]
        );
    }

    /// A set's marker values are elided the same way (the set
    /// walk reads its inner map's pairs).
    #[test]
    fn elided_set_markers_are_filled() {
        let (interp, map) = map_with_node(
            leaf_node(2, vec![Value::Int(1), Value::Int(2)], Vec::new()),
            2,
        );
        let set = Value::Enum {
            variant: 0,
            fields: vec![map],
        };
        let (entries, size) = btree_entries(&interp, &set, true).expect("entries");
        assert_eq!(size, 2);
        assert_eq!(
            entries,
            vec![
                (Value::Int(1), Value::Unit),
                (Value::Int(2), Value::Unit),
            ]
        );
    }

    /// Unit keys elide too: the key array is the short one.
    #[test]
    fn elided_key_slots_are_filled() {
        let (interp, map) = map_with_node(leaf_node(1, Vec::new(), vec![Value::Int(1)]), 1);
        let (entries, size) = btree_entries(&interp, &map, false).expect("entries");
        assert_eq!(size, 1);
        assert_eq!(entries, vec![(Value::Unit, Value::Int(1))]);
    }

    /// A short array whose present slots are not zero-sized is
    /// a genuine hole in the node: the walk still refuses it.
    #[test]
    fn genuine_hole_stays_unrenderable() {
        let (interp, map) = map_with_node(
            leaf_node(2, vec![Value::Int(7)], vec![Value::Int(1), Value::Int(2)]),
            2,
        );
        assert!(btree_entries(&interp, &map, false).is_none());
    }

    /// Fully stored arrays walk as before.
    #[test]
    fn stored_slots_walk_unchanged() {
        let (interp, map) = map_with_node(
            leaf_node(
                2,
                vec![Value::Int(7), Value::Int(8)],
                vec![Value::Int(1), Value::Int(2)],
            ),
            2,
        );
        let (entries, size) = btree_entries(&interp, &map, false).expect("entries");
        assert_eq!(size, 2);
        assert_eq!(
            entries,
            vec![
                (Value::Int(7), Value::Int(1)),
                (Value::Int(8), Value::Int(2)),
            ]
        );
    }
}

#[cfg(test)]
mod hash_stride_tests {
    use super::*;
    use crate::ir::Program;

    /// A `HashMap` value over a hand-laid raw table
    /// allocation: `width` slots per bucket, the control
    /// bytes at `ctrl_index`, the `live` buckets holding the
    /// given pairs (every other bucket empty), and the
    /// table's declared item count.
    fn hash_map(
        num_buckets: usize,
        width: usize,
        ctrl_index: usize,
        declared: u128,
        live: &[(usize, Value, Value)],
    ) -> (Interpreter<'static>, Value) {
        let program: &'static Program = Box::leak(Box::new(Program::default()));
        let interp = Interpreter::new(program);
        let mut slots = vec![Value::Uninit; ctrl_index + num_buckets + 16];
        for i in 0..num_buckets {
            slots[ctrl_index + i] = Value::Uint(0xFF);
        }
        for (i, k, v) in live {
            slots[ctrl_index - (i + 1) * width] = Value::Tuple(vec![k.clone(), v.clone()]);
            slots[ctrl_index + i] = Value::Uint(0x2A);
        }
        let id = interp.alloc_insert(Value::Tuple(slots));
        let ctrl = Value::Enum {
            variant: 0,
            fields: vec![Value::Ref(RefTarget::Alloc {
                id,
                projection: vec![RefProj::Index(ctrl_index)],
            })],
        };
        let inner = Value::Enum {
            variant: 0,
            fields: vec![
                Value::Uint((num_buckets - 1) as u128),
                ctrl,
                Value::Uint(0),
                Value::Uint(declared),
            ],
        };
        let table = Value::Enum {
            variant: 0,
            fields: vec![inner],
        };
        let map = Value::Enum {
            variant: 0,
            fields: vec![Value::Unit, table],
        };
        let value = Value::Enum {
            variant: 0,
            fields: vec![map],
        };
        (interp, value)
    }

    /// Eight 3-slot buckets fill 24 slots ending at the
    /// control bytes, which begin at 32 after the region's
    /// front padding: the walk must use width 3, not the
    /// even split 32 / 8 = 4.
    #[test]
    fn padded_region_walks_at_the_true_width() {
        let (interp, map) = hash_map(
            8,
            3,
            32,
            3,
            &[
                (1, Value::Uint(1), Value::Uint(10)),
                (3, Value::Uint(3), Value::Uint(30)),
                (6, Value::Uint(6), Value::Uint(60)),
            ],
        );
        let (entries, size) = hash_entries(&interp, &map, false).expect("entries");
        assert_eq!(size, 3);
        assert_eq!(
            entries,
            vec![
                (Value::Uint(1), Value::Uint(10)),
                (Value::Uint(3), Value::Uint(30)),
                (Value::Uint(6), Value::Uint(60)),
            ]
        );
    }

    /// An unpadded region (eight 2-slot buckets, control at
    /// 16) walks at the even split, as before.
    #[test]
    fn unpadded_region_walks_at_the_even_split() {
        let (interp, map) = hash_map(
            8,
            2,
            16,
            3,
            &[
                (0, Value::Uint(1), Value::Uint(10)),
                (2, Value::Uint(2), Value::Uint(20)),
                (7, Value::Uint(7), Value::Uint(70)),
            ],
        );
        let (entries, size) = hash_entries(&interp, &map, false).expect("entries");
        assert_eq!(size, 3);
        assert_eq!(
            entries,
            vec![
                (Value::Uint(1), Value::Uint(10)),
                (Value::Uint(2), Value::Uint(20)),
                (Value::Uint(7), Value::Uint(70)),
            ]
        );
    }

    /// A table whose live buckets do not add up to the
    /// declared item count stays unrenderable at every
    /// candidate width.
    #[test]
    fn count_mismatch_stays_unrenderable() {
        let (interp, map) = hash_map(
            8,
            3,
            32,
            4,
            &[
                (1, Value::Uint(1), Value::Uint(10)),
                (3, Value::Uint(3), Value::Uint(30)),
                (6, Value::Uint(6), Value::Uint(60)),
            ],
        );
        assert!(hash_entries(&interp, &map, false).is_none());
    }
    /// A deleted bucket retains its stale pair at its own
    /// bucket start; the wider even-split candidate (16 / 4
    /// = 4) would read that pair as bucket 0's entry and
    /// pass the shape and count checks with the removed
    /// entry in place of the live one. The walk must reject
    /// width 4 — slot 14's live pair sits at no bucket start
    /// of its partition — and walk at the true width 2.
    #[test]
    fn deleted_bucket_stale_pair_does_not_win_a_wider_width() {
        let program: &'static Program = Box::leak(Box::new(Program::default()));
        let interp = Interpreter::new(program);
        let num_buckets = 4usize;
        let ctrl_index = 16usize;
        let mut slots = vec![Value::Uninit; ctrl_index + num_buckets + 16];
        for i in 0..num_buckets {
            slots[ctrl_index + i] = Value::Uint(0xFF);
        }
        slots[14] = Value::Tuple(vec![Value::Uint(1), Value::Uint(10)]);
        slots[ctrl_index] = Value::Uint(0x2A);
        slots[12] = Value::Tuple(vec![Value::Uint(2), Value::Uint(20)]);
        slots[ctrl_index + 1] = Value::Uint(0x80);
        let id = interp.alloc_insert(Value::Tuple(slots));
        let ctrl = Value::Enum {
            variant: 0,
            fields: vec![Value::Ref(RefTarget::Alloc {
                id,
                projection: vec![RefProj::Index(ctrl_index)],
            })],
        };
        let inner = Value::Enum {
            variant: 0,
            fields: vec![
                Value::Uint((num_buckets - 1) as u128),
                ctrl,
                Value::Uint(0),
                Value::Uint(1),
            ],
        };
        let table = Value::Enum {
            variant: 0,
            fields: vec![inner],
        };
        let map = Value::Enum {
            variant: 0,
            fields: vec![Value::Unit, table],
        };
        let value = Value::Enum {
            variant: 0,
            fields: vec![map],
        };
        let (entries, size) = hash_entries(&interp, &value, false).expect("entries");
        assert_eq!(size, 1);
        assert_eq!(entries, vec![(Value::Uint(1), Value::Uint(10))]);
    }
}
