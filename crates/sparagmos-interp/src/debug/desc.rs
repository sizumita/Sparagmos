//! The rendering descriptor: the type shape recovered for a printed value, which drives every dialect's rendering.

use crate::interp::Interpreter;
use crate::ir::{FloatTy, IntTy, Ty, UintTy};
use crate::value::Value;
use super::render_lldb::render_lldb_compact;

// ---------------------------------------------------------------------------
// Value rendering
//
// A print expression's value is rendered from the value itself
// plus a type descriptor recovered for it: the variable entry's
// recorded type tag (the full instance name, generic arguments
// included), refined by the expression's steps. The descriptor
// drives everything the value alone does not say — whether a
// tuple-shaped value is a tuple or an array, an ADT value's
// variant and field names (through the program's debug-shape
// table), a field's element type, and a `char`'s identity.
// ---------------------------------------------------------------------------

/// A rendering descriptor for a value being printed.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Desc {
    Unit,
    Bool,
    Char,
    /// An integer type, by its canonical name (`i32`, `u8`,
    /// ...): the concrete name, not just the signedness, so a
    /// type query over a projected value (a field, an element, a
    /// pointee) can print the primitive type it was projected
    /// from.
    Int(&'static str),
    /// A floating-point type, by its canonical name.
    Float(&'static str),
    Str,
    Tuple(Vec<Desc>),
    Array(Box<Desc>, u64),
    Slice(Box<Desc>),
    Ref(Box<Desc>),
    /// A mutable reference (`&mut T`): the qualifier is part
    /// of the recorded type, so it is carried separately from
    /// a shared reference.
    RefMut(Box<Desc>),
    /// A raw const pointer (`*const T`).
    PtrConst(Box<Desc>),
    /// A raw mutable pointer (`*mut T`).
    PtrMut(Box<Desc>),
    /// An ADT instance, by its rendered type tag (the key of the
    /// program's debug-shape table).
    Adt(String),
    /// A closure type, by its body's name.
    Closure(String),
    /// A coroutine type, by its body's name.
    Coroutine(String),
    FnPtr,
    Unknown,
}

/// The canonical name of a signed integer type.
fn int_ty_name(ty: IntTy) -> &'static str {
    match ty {
        IntTy::I8 => "i8",
        IntTy::I16 => "i16",
        IntTy::I32 => "i32",
        IntTy::I64 => "i64",
        IntTy::I128 => "i128",
        IntTy::Isize => "isize",
    }
}

/// The canonical name of an unsigned integer type.
fn uint_ty_name(ty: UintTy) -> &'static str {
    match ty {
        UintTy::U8 => "u8",
        UintTy::U16 => "u16",
        UintTy::U32 => "u32",
        UintTy::U64 => "u64",
        UintTy::U128 => "u128",
        UintTy::Usize => "usize",
    }
}

/// The canonical name of a floating-point type.
fn float_ty_name(ty: FloatTy) -> &'static str {
    match ty {
        FloatTy::F16 => "f16",
        FloatTy::F32 => "f32",
        FloatTy::F64 => "f64",
        FloatTy::F128 => "f128",
    }
}

impl Desc {
    pub(super) fn from_ir(ty: &Ty) -> Desc {
        match ty {
            Ty::Unit => Desc::Unit,
            Ty::Bool => Desc::Bool,
            Ty::Int(t) => Desc::Int(int_ty_name(*t)),
            Ty::Uint(t) => Desc::Int(uint_ty_name(*t)),
            Ty::Float(t) => Desc::Float(float_ty_name(*t)),
            Ty::Tuple(ts) => Desc::Tuple(ts.iter().map(Desc::from_ir).collect()),
            Ty::Never => Desc::Unknown,
            Ty::Adt(name) => Desc::Adt(name.clone()),
            Ty::Ref(inner) => Desc::Ref(Box::new(Desc::from_ir(inner))),
            Ty::Slice(elem) => Desc::Slice(Box::new(Desc::from_ir(elem))),
            Ty::Array(elem, n) => Desc::Array(Box::new(Desc::from_ir(elem)), *n),
            Ty::Str => Desc::Str,
            Ty::FnPtr | Ty::FnDef(_) => Desc::FnPtr,
            Ty::Closure(name, _) => Desc::Closure(name.clone()),
            Ty::Dynamic(_) => Desc::Unknown,
            Ty::Coroutine(name) => Desc::Coroutine(name.clone()),
        }
    }

    /// Parse a rendered type tag (the form [`crate::builtins`]'s
    /// `ty_tag` produces, and the form the debug-shape table's
    /// field entries carry) back into a descriptor.
    pub(super) fn from_tag(tag: &str) -> Desc {
        let tag = tag.trim();
        match tag {
            "()" => return Desc::Unit,
            "bool" => return Desc::Bool,
            "char" => return Desc::Char,
            "str" => return Desc::Str,
            "i8" => return Desc::Int("i8"),
            "i16" => return Desc::Int("i16"),
            "i32" => return Desc::Int("i32"),
            "i64" => return Desc::Int("i64"),
            "i128" => return Desc::Int("i128"),
            "isize" => return Desc::Int("isize"),
            "u8" => return Desc::Int("u8"),
            "u16" => return Desc::Int("u16"),
            "u32" => return Desc::Int("u32"),
            "u64" => return Desc::Int("u64"),
            "u128" => return Desc::Int("u128"),
            "usize" => return Desc::Int("usize"),
            "f16" => return Desc::Float("f16"),
            "f32" => return Desc::Float("f32"),
            "f64" => return Desc::Float("f64"),
            "f128" => return Desc::Float("f128"),
            "!" => return Desc::Unknown,
            _ => {}
        }
        if let Some(rest) = tag.strip_prefix("&mut ") {
            return Desc::RefMut(Box::new(Desc::from_tag(rest)));
        }
        if let Some(rest) = tag.strip_prefix('&') {
            return Desc::Ref(Box::new(Desc::from_tag(rest)));
        }
        if let Some(rest) = tag.strip_prefix("*const ") {
            return Desc::PtrConst(Box::new(Desc::from_tag(rest)));
        }
        if let Some(rest) = tag.strip_prefix("*mut ") {
            return Desc::PtrMut(Box::new(Desc::from_tag(rest)));
        }
        if tag.starts_with('(') && tag.ends_with(')') {
            let inner = &tag[1..tag.len() - 1];
            if inner.trim().is_empty() {
                return Desc::Unit;
            }
            let descs = split_top_level(inner)
                .iter()
                .filter(|part| !part.is_empty())
                .map(|part| Desc::from_tag(part))
                .collect();
            return Desc::Tuple(descs);
        }
        if tag.starts_with('[') && tag.ends_with(']') {
            let inner = &tag[1..tag.len() - 1];
            if let Some((elem, len)) = inner.rsplit_once(';') {
                if let Ok(n) = len.trim().parse::<u64>() {
                    return Desc::Array(Box::new(Desc::from_tag(elem)), n);
                }
            }
            return Desc::Slice(Box::new(Desc::from_tag(inner)));
        }
        if tag.starts_with("dyn ") {
            return Desc::Unknown;
        }
        Desc::Adt(tag.to_string())
    }
}

/// The base name of a rendered type tag: the last path segment
/// of the part before any generic argument list.
pub(super) fn tag_base_name(tag: &str) -> &str {
    let head = tag.split('<').next().unwrap_or(tag).trim();
    head.rsplit("::").next().unwrap_or(head)
}

/// The text of a path-family value (`PathBuf`, `Path`,
/// `OsString`, `OsStr`) in this runtime's std model: nested
/// single-field aggregates wrapping a byte list (the `Vec`
/// element model), or a plain string value.
pub(super) fn path_text(value: &Value) -> Option<String> {
    let mut cur = value;
    for _ in 0..8 {
        match cur {
            Value::Str(s) => return Some(s.clone()),
            Value::Enum { fields, .. } if fields.len() == 1 => {
                cur = &fields[0];
            }
            Value::Enum { fields, .. }
                if fields
                    .iter()
                    .all(|f| matches!(f, Value::Uint(_))) =>
            {
                let bytes: Vec<u8> = fields
                    .iter()
                    .map(|f| match f {
                        Value::Uint(n) => *n as u8,
                        _ => 0,
                    })
                    .collect();
                return Some(String::from_utf8_lossy(&bytes).into_owned());
            }
            _ => return None,
        }
    }
    None
}

/// lldb's summary for a `Vec` value: in this runtime's std model
/// a vector is an ADT aggregate of its elements, not the
/// library's buffer/length layout, so the debug-shape table
/// cannot describe it; the formatter's own summary is the
/// element count with the elements by index.
pub(super) fn render_vec_lldb(
    interp: &Interpreter,
    tag: &str,
    fields: &[Value],
    depth: usize,
) -> Option<String> {
    let elem = first_generic_arg(tag).map(|t| Desc::from_tag(&t));
    let mut parts = Vec::with_capacity(fields.len());
    for (i, field) in fields.iter().enumerate() {
        let v = render_lldb_compact(
            interp,
            field,
            elem.as_ref(),
            false,
            true,
            depth + 1,
        )?;
        parts.push(format!("[{i}] = {v}"));
    }
    if parts.is_empty() {
        Some("size=0".to_string())
    } else {
        Some(format!("size={} {{ {} }}", fields.len(), parts.join(" ")))
    }
}

/// Split a rendered type list on its top-level commas (commas
/// inside `<...>`, `(...)`, `[...]` do not split).
pub(super) fn split_top_level(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in text.char_indices() {
        match c {
            '<' | '(' | '[' | '{' => depth += 1,
            '>' | ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(text[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(text[start..].trim().to_string());
    // A trailing comma -- the canonical one-element tuple
    // spelling `(u32,)` -- leaves an empty final component:
    // it is not a component.
    if parts.len() > 1 && parts.last().is_some_and(|part| part.is_empty()) {
        parts.pop();
    }
    parts
}

/// The first generic argument of a rendered instance tag.
pub(super) fn first_generic_arg(tag: &str) -> Option<String> {
    let open = tag.find('<')?;
    let inner = tag.get(open + 1..tag.len().checked_sub(1)?)?;
    split_top_level(inner)
        .into_iter()
        .next()
        .filter(|arg| !arg.is_empty())
}

/// The pointee type tag of a smart-pointer instance tag (`Box`,
/// `Rc`, `Arc`): the pointee a dereference step reaches. The
/// match is on the type's base name, the convention the format
/// engine's tag dispatch uses: the recorded names spell the
/// smart pointers through both their defining crate and the
/// `std` re-export (`alloc::boxed::Box`, `std::boxed::Box`).
pub(super) fn smart_pointer_pointee(tag: &str) -> Option<String> {
    let base = tag.split('<').next()?.trim();
    let name = base.rsplit("::").next()?;
    if matches!(name, "Box" | "Rc" | "Arc") {
        first_generic_arg(tag)
    } else {
        None
    }
}

/// The DWARF form of a rendered type tag. The debugger's type
/// names are definition paths, so a path that names a `std`
/// re-export of a `core` or `alloc` item takes the defining
/// crate's name: `std::option::Option` is
/// `core::option::Option`, `std::boxed::Box` is
/// `alloc::boxed::Box`, `std::alloc::Global` is
/// `alloc::alloc::Global`. The mapping is the standard library's
/// re-export structure: the whole modules `std` re-exports from
/// `core` or `alloc`, plus the individual items of the mixed
/// modules (`sync`, `collections`, `ffi`, `time`). Any other
/// path already is a definition path.
pub(super) fn dwarf_type_name(tag: &str) -> String {
    fn defining_crate(seg2: &str, seg3: &str) -> Option<&'static str> {
        Some(match (seg2, seg3) {
            (
                "option" | "result" | "fmt" | "marker" | "mem" | "ptr" | "ops"
                | "cell" | "num" | "slice" | "str" | "any" | "array" | "ascii"
                | "char" | "cmp" | "convert" | "default" | "hash" | "iter"
                | "clone" | "panic" | "pin" | "task" | "future" | "intrinsics",
                _,
            ) => "core",
            ("alloc" | "borrow" | "boxed" | "rc" | "string" | "vec", _) => {
                "alloc"
            }
            ("sync", "atomic") => "core",
            ("sync", "Arc" | "Weak") => "alloc",
            (
                "collections",
                "BTreeMap" | "BTreeSet" | "BinaryHeap" | "LinkedList"
                | "VecDeque",
            ) => "alloc",
            ("ffi", "CStr") => "core",
            ("ffi", "CString") => "alloc",
            ("time", "Duration") => "core",
            _ => return None,
        })
    }
    let bytes = tag.as_bytes();
    let mut out = String::with_capacity(tag.len());
    let mut i = 0;
    while i < bytes.len() {
        let at_path_start = i == 0
            || (!bytes[i - 1].is_ascii_alphanumeric()
                && bytes[i - 1] != b'_'
                && bytes[i - 1] != b':');
        if at_path_start && tag[i..].starts_with("std::") {
            let rest = &tag[i + 5..];
            let seg2_end = rest
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .unwrap_or(rest.len());
            let seg2 = &rest[..seg2_end];
            let seg3 = rest[seg2_end..]
                .strip_prefix("::")
                .map(|r| {
                    let e = r
                        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                        .unwrap_or(r.len());
                    &r[..e]
                })
                .unwrap_or("");
            if let Some(krate) = defining_crate(seg2, seg3) {
                out.push_str(krate);
                i += 3; // consume `std`, keep the `::` and the rest
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// The descriptor of a value's `index`-th field, under `desc`:
/// the element of a tuple, the element type of an array, or the
/// recorded field type of an ADT instance (the variant taken
/// from the value itself).
pub(super) fn field_desc(
    interp: &Interpreter,
    desc: &Desc,
    value: &Value,
    index: usize,
) -> Option<Desc> {
    match desc {
        Desc::Tuple(ts) => ts.get(index).cloned(),
        Desc::Array(elem, _) => Some((**elem).clone()),
        Desc::Adt(tag) => {
            let shape = interp.program().adt_debug_shapes.get(tag)?;
            let variant = match value {
                Value::Enum { variant, .. } => *variant as usize,
                _ => 0,
            };
            let field = shape.variants.get(variant)?.fields.get(index)?;
            Some(Desc::from_tag(&field.ty))
        }
        _ => None,
    }
}
