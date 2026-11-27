//! Name lookup for call resolution: parsing rendered
//! names and impl-instance keys, and mapping impl-path
//! trait methods and primitive operations to builtins.

/// The generic arguments of a rendered type name, split at the
/// top nesting level of the outermost angle brackets
/// (`std::vec::Vec<(), A>` gives `["()", "A"]`).
fn generic_args_of(name: &str) -> Option<Vec<&str>> {
    let open = name.find('<')?;
    let bytes = name.as_bytes();
    let mut depth = 0i32;
    let mut end = None;
    for i in open..bytes.len() {
        match bytes[i] {
            b'<' | b'(' | b'[' => depth += 1,
            // The `>` of a `->` arrow (a function-pointer
            // signature inside the name) closes no bracket.
            b'>' if i > open && bytes[i - 1] == b'-' => {}
            b'>' | b')' | b']' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let inner = &name[open + 1..end?];
    Some(split_top_level_str(inner))
}

/// Split a comma-separated list at the top nesting level
/// (angle brackets, parentheses and brackets all nest).
pub(crate) fn split_top_level_str(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut prev = '\0';
    for (i, c) in s.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
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
    parts.push(s[start..].trim());
    parts
}

/// The element type tag of the sequence a rendered type tag
/// denotes: an array's element, a pointer or reference position's
/// pointee sequence (an array pointee's element, any other
/// pointee itself), or an ADT instance's first generic argument
/// (the element type of the modelled collections — `Vec<T>` and
/// its kin — whose element positions a projection indexes
/// directly).
pub(crate) fn tag_sequence_elem(tag: &str) -> Option<&str> {
    let tag = tag.trim();
    for prefix in ["*const ", "*mut "] {
        if let Some(rest) = tag.strip_prefix(prefix) {
            let rest = rest.trim();
            if rest.starts_with('[') {
                return tag_sequence_elem(rest);
            }
            return Some(rest);
        }
    }
    if let Some(rest) = tag.strip_prefix('&') {
        let mut rest = rest.trim_start();
        if let Some(stripped) = rest.strip_prefix('\'') {
            // A lifetime: skip its name and the following space.
            rest = stripped
                .split_once(' ')
                .map(|(_, tail)| tail)
                .unwrap_or(stripped)
                .trim_start();
        }
        let rest = rest.strip_prefix("mut ").unwrap_or(rest).trim();
        if rest.starts_with('[') {
            return tag_sequence_elem(rest);
        }
        return Some(rest);
    }
    if tag.starts_with('[') {
        let inner = tag.strip_prefix('[')?.strip_suffix(']')?;
        if let Some((elem, _)) = inner.rsplit_once(';') {
            return Some(elem.trim());
        }
        return Some(inner.trim());
    }
    generic_args_of(tag)?.into_iter().next()
}

/// The Self type of an impl-qualified method name
/// (`<i32 as std::default::Default>::default` gives `i32`):
/// the text between the leading `<` and the ` as ` separator at
/// the same bracket depth. `None` for any other name shape.
pub(crate) fn impl_qualified_self_type(name: &str) -> Option<String> {
    let inner = name.strip_prefix('<')?;
    let bytes = inner.as_bytes();
    let mut depth = 1usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => depth += 1,
            b'>' if i > 0 && bytes[i - 1] == b'-' => {}
            b'>' => {
                depth -= 1;
                if depth == 0 {
                    return None;
                }
            }
            b' ' if depth == 1 && inner[i..].starts_with(" as ") => {
                return Some(inner[..i].trim().to_string());
            }
            _ => {}
        }
        i += 1;
    }
    None
}

pub(super) fn impl_path_trait_method(name: &str) -> Option<crate::builtins::Builtin> {
    use crate::builtins::Builtin;
    const METHODS: &[(&str, &str, Builtin)] = &[
        ("PartialEq", "eq", Builtin::PartialEqEq),
        ("PartialEq", "ne", Builtin::PartialEqNe),
        ("PartialOrd", "partial_cmp", Builtin::PartialCmp),
        ("PartialOrd", "lt", Builtin::PartialOrdRel(crate::builtins::RelOp::Lt)),
        ("PartialOrd", "le", Builtin::PartialOrdRel(crate::builtins::RelOp::Le)),
        ("PartialOrd", "gt", Builtin::PartialOrdRel(crate::builtins::RelOp::Gt)),
        ("PartialOrd", "ge", Builtin::PartialOrdRel(crate::builtins::RelOp::Ge)),
        ("Ord", "cmp", Builtin::OrdCmp),
        ("Clone", "clone", Builtin::CloneClone),
        ("ToOwned", "to_owned", Builtin::ToOwned),
    ];
    for (trait_seg, method, kind) in METHODS {
        if !name.ends_with(&format!(">::{method}")) {
            continue;
        }
        let qualified = format!("::{trait_seg}>::{method}");
        let impl_header = format!("::{trait_seg} for ");
        if name.contains(&qualified) || name.contains(&impl_header) {
            return Some(*kind);
        }
    }
    None
}

/// The primitive arithmetic trait method an impl-qualified
/// spelling of a call denotes (`<i32 as std::ops::Add>::add`,
/// `<i32 as std::ops::Add<i32>>::add`): the operator traits on
/// the primitive types, whose core impls are the operations
/// themselves. The Self type must be a primitive the value
/// model computes on — for any other Self type the impl body
/// is the only honest source, so the name does not qualify.
/// The plain trait path (`std::ops::Add::add`) does not
/// qualify either: it names no Self type, and the values carry
/// no width to compute with. `None` for any other shape.
pub(super) fn impl_path_primitive_op(name: &str) -> Option<crate::builtins::Builtin> {
    use crate::builtins::Builtin;
    use crate::ir::{BinOp, UnOp};
    const BINOPS: &[(&str, &str, BinOp)] = &[
        ("Add", "add", BinOp::Add),
        ("Sub", "sub", BinOp::Sub),
        ("Mul", "mul", BinOp::Mul),
        ("Div", "div", BinOp::Div),
        ("Rem", "rem", BinOp::Rem),
        ("BitAnd", "bitand", BinOp::BitAnd),
        ("BitOr", "bitor", BinOp::BitOr),
        ("BitXor", "bitxor", BinOp::BitXor),
        ("Shl", "shl", BinOp::Shl),
        ("Shr", "shr", BinOp::Shr),
    ];
    const UNOPS: &[(&str, &str, UnOp)] = &[
        ("Neg", "neg", UnOp::Neg),
        ("Not", "not", UnOp::Not),
    ];
    const PRIMITIVE_SELF: &[&str] = &[
        "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64",
        "u128", "usize", "f32", "f64", "bool",
    ];
    let self_ty = impl_qualified_self_type(name)?;
    if !PRIMITIVE_SELF.contains(&self_ty.as_str()) {
        return None;
    }
    // The trait segment: the last path segment before the
    // closing `>::`, generic arguments aside.
    let close = name.rfind(">::")?;
    let segment = name[..close].rsplit("::").next()?;
    let trait_name = segment.split('<').next()?;
    for (trait_seg, method, op) in BINOPS {
        if trait_name == *trait_seg && name.ends_with(&format!("::{method}")) {
            return Some(Builtin::PrimitiveBinOp(*op));
        }
    }
    for (trait_seg, method, op) in UNOPS {
        if trait_name == *trait_seg && name.ends_with(&format!("::{method}")) {
            return Some(Builtin::PrimitiveUnOp(*op));
        }
    }
    None
}

/// Parse an impl-instance key `<Self as Trait>::method` into
/// `(Self section, trait path without generic arguments, method)`,
/// mirroring `mir_convert`'s key parsing. `None` for other shapes.
/// Split a rendered generic-argument list's contents at its
/// top-level commas (commas inside nested `<...>`, `(...)`,
/// `[...]`, `{...}` groups do not split). Empty input yields an
/// empty list.
pub(crate) fn split_generic_args(inner: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let bytes = inner.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'-' if bytes.get(i + 1) == Some(&b'>') => {
                i += 1;
            }
            b'<' | b'(' | b'[' | b'{' => depth += 1,
            b'>' | b')' | b']' | b'}' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(inner[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    let tail = inner[start..].trim();
    if !tail.is_empty() {
        parts.push(tail.to_string());
    }
    parts
}

/// The `(plain path, arguments)` of a rendered instance path:
/// every top-level `::<...>` group is removed from the path and
/// its arguments collected in order. A path with no group yields
/// itself and an empty list; an unbalanced group yields `None`.
fn strip_generic_groups(path: &str) -> Option<(String, Vec<String>)> {
    let mut plain = String::new();
    let mut args = Vec::new();
    let bytes = path.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b':' && path[i..].starts_with("::<") {
            let mut depth = 0i32;
            let mut j = i + 2;
            let inner_start = j + 1;
            let mut inner_end = None;
            while j < bytes.len() {
                match bytes[j] {
                    b'-' if bytes.get(j + 1) == Some(&b'>') => {
                        j += 1;
                    }
                    b'<' => depth += 1,
                    b'>' => {
                        depth -= 1;
                        if depth == 0 {
                            inner_end = Some(j);
                            break;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            let inner_end = inner_end?;
            args.extend(split_generic_args(&path[inner_start..inner_end]));
            i = inner_end + 1;
        } else {
            let ch_len = path[i..].chars().next()?.len_utf8();
            plain.push_str(&path[i..i + ch_len]);
            i += ch_len;
        }
    }
    Some((plain, args))
}

/// The parts of a coroutine *type* tag as a `dyn` value carries
/// it: the enclosing definition's plain path, the closing
/// `{closure#N}` segment, and the tag's generic arguments (the
/// enclosing definition's arguments first, then the coroutine
/// signature's own). `None` when the tag is not of this shape.
pub(crate) fn parse_coroutine_tag(tag: &str) -> Option<(String, String, Vec<String>)> {
    let idx = tag.rfind("::{closure#")?;
    let parent = &tag[..idx];
    let rest = &tag[idx + 2..];
    let seg_end = rest.find('}')?;
    let seg = &rest[..=seg_end];
    let tail = &rest[seg_end + 1..];
    let args = if tail.is_empty() {
        Vec::new()
    } else {
        let inner = tail.strip_prefix('<')?.strip_suffix('>')?;
        split_generic_args(inner)
    };
    Some((parent.to_string(), seg.to_string(), args))
}

/// The parts of a coroutine body *key*: the enclosing
/// definition's plain path and its instance arguments (see
/// [`strip_generic_groups`]), plus the `{closure#N}` segment.
/// `None` when the key is not a coroutine body key of this
/// shape.
pub(crate) fn parse_coroutine_key(key: &str) -> Option<(String, String, Vec<String>)> {
    let idx = key.rfind("::{closure#")?;
    let rest = &key[idx + 2..];
    let seg_end = rest.find('}')?;
    if seg_end + 1 != rest.len() {
        return None;
    }
    let seg = &rest[..=seg_end];
    let (parent, args) = strip_generic_groups(&key[..idx])?;
    Some((parent, seg.to_string(), args))
}

/// Whether a recorded vtable method instance name (see
/// `Value::DynRef::vtable_methods`) is the entry a call to
/// `method` of the trait at `trait_path` dispatches through,
/// given the bounds the pointer currently answers to
/// (`traits`): the name must be an instance of that trait and
/// that method, and no current bound may name the same trait
/// under different generic arguments -- an upcast to another
/// instantiation of the trait answers to that instantiation's
/// vtable, not to the one the name was recorded from. A bound
/// naming the trait without arguments, and a name recorded
/// without arguments, are each compatible with the other's
/// spelling: neither distinguishes instantiations the other
/// names. The `std` / `core` spellings of one trait path are
/// interchangeable, as in the bounds step of the dispatch.
pub(super) fn vtable_entry_answers(
    name: &str,
    trait_path: &str,
    method: &str,
    traits: &[String],
) -> bool {
    let Some(rest) = name.strip_suffix(&format!(">::{method}")) else {
        return false;
    };
    // `rest` is `<{self} as {trait}`: the trait segment
    // follows the last ` as ` (a Self spelling can itself
    // contain one, inside a qualified path argument).
    let Some((_, seg)) = rest.rsplit_once(" as ") else {
        return false;
    };
    let (seg_path, seg_args) = match seg.split_once('<') {
        Some((path, args)) => (path, Some(args)),
        None => (seg, None),
    };
    let path_matches = seg_path == trait_path
        || trait_path
            .strip_prefix("std::")
            .is_some_and(|r| seg_path == format!("core::{r}"))
        || trait_path
            .strip_prefix("core::")
            .is_some_and(|r| seg_path == format!("std::{r}"));
    if !path_matches {
        return false;
    }
    for bound in traits {
        let (bound_path, bound_args) = match bound.split_once('<') {
            Some((path, args)) => (path, Some(args)),
            None => (bound.as_str(), None),
        };
        if bound_path != seg_path {
            continue;
        }
        if let (Some(a), Some(b)) = (seg_args, bound_args) {
            if a != b {
                return false;
            }
        }
    }
    true
}

/// The spellings of a rendered type with trailing generic
/// arguments dropped from its outermost argument list, longest
/// first: `a::W<X, Y, Z>` yields `a::W<X, Y>`, then `a::W<X>`,
/// then `a::W`. A name with no argument list (or an empty one)
/// yields nothing.
pub(super) fn elided_self_spellings(name: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = name.to_string();
    while let Some(elided) = drop_last_generic_arg(&current) {
        out.push(elided.clone());
        current = elided;
    }
    out
}

/// `name` with the last argument of its outermost generic
/// argument list removed (the bare path once the list is
/// empty); `None` when the name does not end in an argument
/// list. Both scans track nesting depth, so commas inside
/// nested arguments do not count, and the `>` of a `->` is not
/// a bracket.
fn drop_last_generic_arg(name: &str) -> Option<String> {
    let bytes = name.as_bytes();
    if bytes.last() != Some(&b'>') {
        return None;
    }
    // The '<' matching the final '>', scanning backwards.
    let mut depth = 0i32;
    let mut open = None;
    for i in (0..bytes.len()).rev() {
        match bytes[i] {
            b'>' if i > 0 && bytes[i - 1] == b'-' => {}
            b'>' => depth += 1,
            b'<' => {
                depth -= 1;
                if depth == 0 {
                    open = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let open = open?;
    let inner = &name[open + 1..name.len() - 1];
    if inner.is_empty() {
        return None;
    }
    // The last comma at depth 0 within the list.
    let mut depth = 0i32;
    let mut last_comma = None;
    for (i, &b) in inner.as_bytes().iter().enumerate() {
        match b {
            b'<' | b'(' | b'[' => depth += 1,
            b'>' | b')' | b']' => depth -= 1,
            b',' if depth == 0 => last_comma = Some(i),
            _ => {}
        }
    }
    match last_comma {
        Some(i) => Some(format!("{}<{}>", &name[..open], &inner[..i])),
        None => Some(name[..open].to_string()),
    }
}

pub(super) fn parse_impl_key(base: &str) -> Option<(String, String, String)> {
    let (head, method) = base.rsplit_once(">::")?;
    // An instance key spells the method's own generic arguments
    // as a trailing `::<...>` segment (`bar::<'_>` for a method
    // with lifetime parameters); the method a call names is the
    // segment-free name.
    let method = method.split("::<").next().unwrap_or(method);
    if method.is_empty() || method.contains("::") {
        return None;
    }
    let inner = head.strip_prefix('<')?;
    let (self_part, trait_part) = inner.split_once(" as ")?;
    if self_part.is_empty() || trait_part.is_empty() {
        return None;
    }
    let trait_base = trait_part.split('<').next().unwrap_or(trait_part);
    Some((
        self_part.to_string(),
        trait_base.to_string(),
        method.to_string(),
    ))
}

/// Parse an impl body key in the impl-block path form the
/// driver records for impls inside a module (foreign crates in
/// particular): `some::module::<impl Trait<..> for Self>::method`.
/// Returns the `Self` section, the trait path without generic
/// arguments, and the method name, as [`parse_impl_key`] does
/// for the `<Self as Trait>::method` form. Mirrors the
/// conversion-side parser of the same name in
/// `mir_convert::helpers`, which resolves the same keys when a
/// call is converted.
pub(super) fn parse_impl_block_key(base: &str) -> Option<(String, String, String)> {
    let (head, method) = base.rsplit_once(">::")?;
    // As in `parse_impl_key`: the method's own generic
    // arguments are a trailing `::<...>` segment, not part of
    // the name a call names.
    let method = method.split("::<").next().unwrap_or(method);
    if method.is_empty() || method.contains("::") {
        return None;
    }
    // The `>::` separator consumed the impl's closing `>`, so
    // `head` ends with the `Self` section itself.
    let marker = head.rfind("::<impl ")?;
    let inner = &head[marker + "::<impl ".len()..];
    let (trait_part, self_part) = inner.split_once(" for ")?;
    if self_part.is_empty() || trait_part.is_empty() {
        return None;
    }
    let trait_base = trait_part.split('<').next().unwrap_or(trait_part);
    Some((
        self_part.to_string(),
        trait_base.to_string(),
        method.to_string(),
    ))
}


/// The spelling of a type name with every function-item type
/// in it replaced by the item's own path. The compiler spells
/// a function item's type as its signature followed by the
/// definition path in braces (`fn() -> i32 {crate::f}`, or
/// `fn() {crate::g}` for a unit return), while a rendered
/// concrete type names the item directly (`crate::f` in place
/// of the whole signature). The two spellings denote the same
/// type in different positions of one `Self` section — a tuple
/// element, a generic argument — so canonicalising the
/// signature form away lets the spellings of one type compare
/// equal. A signature whose return type carries braces of its
/// own is left as it stands: the annotation cannot be told
/// apart from the return type there, and the spelling simply
/// stays unmatched, as before. Mirrors the copy of the same
/// name in `mir_convert::helpers`.
fn fn_item_spelling(name: &str) -> String {
    let bytes = name.as_bytes();
    let mut out = String::with_capacity(name.len());
    let mut rest = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let Some((item, end)) = fn_item_at(bytes, i) else {
            i += 1;
            continue;
        };
        out.push_str(&name[rest..i]);
        out.push_str(item);
        rest = end;
        i = end;
    }
    out.push_str(&name[rest..]);
    out
}

/// If a function-item type spelling starts at byte `start`
/// of `bytes`, return the item's path and the byte just past
/// the spelling. The spelling is the item's signature —
/// optional `unsafe` / `extern ".."` qualifiers, the argument
/// list, an optional return type — followed by the definition
/// path in braces.
fn fn_item_at(bytes: &[u8], start: usize) -> Option<(&str, usize)> {
    // A type can begin here only at a boundary of the
    // enclosing spelling (so the `fn` inside an identifier
    // such as `callfn` is not read as a signature).
    if start > 0 && !matches!(bytes[start - 1], b'(' | b',' | b'<' | b'[' | b' ') {
        return None;
    }
    let mut p = start;
    // The qualifiers, in either order (`unsafe extern "C"
    // fn` is the current spelling, `extern "C" unsafe fn`
    // the older one).
    for _ in 0..2 {
        if bytes[p..].starts_with(b"unsafe ") {
            p += "unsafe ".len();
        } else if bytes[p..].starts_with(b"extern ") {
            let quote = p + "extern ".len();
            if bytes.get(quote) != Some(&b'"') {
                return None;
            }
            let close = bytes[quote + 1..].iter().position(|&c| c == b'"')? + quote + 1;
            if bytes.get(close + 1) != Some(&b' ') {
                return None;
            }
            p = close + 2;
        } else {
            break;
        }
    }
    if !bytes[p..].starts_with(b"fn(") {
        return None;
    }
    // The argument list, balanced.
    let mut depth = 0usize;
    let mut q = p + 2;
    loop {
        match bytes.get(q)? {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            _ => {}
        }
        q += 1;
    }
    q += 1; // past ')'
    // The annotation: directly after the argument list, or
    // past a return type. A return type carrying braces of
    // its own ends the parse (see `fn_item_spelling`).
    let ann = if bytes[q..].starts_with(b" {") {
        q + 1
    } else if bytes[q..].starts_with(b" -> ") {
        let mut j = q + 4;
        let mut depth = 0usize;
        loop {
            match bytes.get(j)? {
                b'-' if bytes.get(j + 1) == Some(&b'>') => j += 1,
                b'(' | b'[' | b'<' => depth += 1,
                b')' | b']' => {
                    if depth == 0 {
                        return None;
                    }
                    depth -= 1;
                }
                b'>' => {
                    if depth == 0 {
                        return None;
                    }
                    depth -= 1;
                }
                b'{' => {
                    if bytes[j - 1] != b' ' {
                        return None;
                    }
                    break;
                }
                b',' | b';' | b'}' if depth == 0 => return None,
                _ => {}
            }
            j += 1;
        }
        j
    } else {
        return None;
    };
    // `bytes[ann]` is '{': the path runs to the closing '}'.
    let inner = &bytes[ann + 1..];
    let close = inner.iter().position(|&c| c == b'}')?;
    let path = &inner[..close];
    if path.is_empty()
        || !path.iter().all(|&c| {
            c.is_ascii_alphanumeric() || matches!(c, b'_' | b':' | b'.' | b'<' | b'>' | b'#')
        })
    {
        return None;
    }
    let path = std::str::from_utf8(path).ok()?;
    Some((path, ann + 1 + close + 1))
}

/// Whether an impl key's `Self` section denotes `name`: the name
/// itself, or the name with a generic argument list appended; the
/// crate prefix may be omitted on a side that carries none. A
/// function item inside the type may be spelled by its signature
/// on one side and by its path on the other (see
/// `fn_item_spelling`). When both sides spell a generic argument
/// list, the lists must agree as canonical spellings (see
/// `normalize_impl_self`): the base name alone never equates two
/// instantiations of one generic type.
pub(super) fn impl_self_matches(self_part: &str, name: &str) -> bool {
    // The section and the queried name can be the very same
    // rendering (a `Self` that is itself an impl-path type,
    // whose base the `<` split below cannot recover).
    if self_part == name {
        return true;
    }
    // The two sides can also differ only in how a function
    // item inside the type is spelled.
    if fn_item_spelling(self_part) == fn_item_spelling(name) {
        return true;
    }
    // When both sides spell a generic argument list, the
    // lists are part of the type's identity: two
    // instantiations of one generic type are different
    // types, and a body collected for one must never serve
    // a call monomorphised at another (a `Map` over one
    // guest adapter and a `Map` over another share the
    // base name and nothing else). The comparison is
    // therefore the canonical spelling (see
    // `normalize_impl_self`), which erases only the parts
    // the two sides legitimately spell differently,
    // compared argument by argument with the key's bare
    // parameter names standing for any instantiation (see
    // `canonical_self_matches`). The
    // base-name comparison below remains for a side that
    // carries no argument list at all.
    if self_part.contains('<') && name.contains('<') {
        return canonical_self_matches(
            &normalize_impl_self(self_part),
            &normalize_impl_self(name),
        );
    }
    let self_base = self_part.split('<').next().unwrap_or(self_part);
    // The queried name may itself carry a generic argument list
    // (a fully rendered concrete type); only its base takes part
    // in the comparison, as the key side's list is stripped above.
    let name = name.split('<').next().unwrap_or(name);
    if self_part != self_base && !self_part.ends_with('>') {
        return false;
    }
    if self_base == name {
        return true;
    }
    // A closure type is spelled two ways that share no text:
    // an impl key's `Self` section uses the compiler's span
    // spelling (`{closure@<file>:16:17: 16:23}`), while a
    // rendered type tag names the definition path's indexed
    // segment (`main::{closure#0}`). Neither spelling carries
    // the other's discriminant -- the span has no index and
    // the tag no span -- so the two are equated only under
    // the caller's uniqueness contract: the scans using this
    // match resolve nothing when two distinct candidates
    // match, so two closures holding impls of one trait
    // method in one program stay unresolved rather than
    // mis-dispatched.
    if self_part.starts_with("{closure@") && self_part.ends_with('}') {
        let segment = name.rsplit("::").next().unwrap_or(name);
        if segment.starts_with("{closure#") && segment.ends_with('}') {
            return true;
        }
    }
    if self_base.contains("::") && name.contains("::") {
        return false;
    }
    self_base.rsplit("::").next().unwrap_or(self_base)
        == name.rsplit("::").next().unwrap_or(name)
}

/// A type spelling with the lifetime of every reference in
/// it removed (`&'a isize` -> `&isize`, `&'a mut isize` ->
/// `&mut isize`): rendered type names drop reference
/// lifetimes entirely, while instance keys keep them, so
/// the two sides of an impl-`Self` comparison legitimately
/// differ in exactly this part. Lifetimes in other
/// positions (generic arguments) are not touched here;
/// `normalize_impl_self`'s list processing handles those.
fn strip_reference_lifetimes(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut chars = name.chars().peekable();
    while let Some(c) = chars.next() {
        out.push(c);
        if c == '&' && chars.peek() == Some(&'\'') {
            // Drop the lifetime token and the space after it.
            while let Some(&n) = chars.peek() {
                chars.next();
                if n == ' ' {
                    break;
                }
            }
        }
    }
    out
}

/// The spelling of an impl `Self` type (or of a queried
/// concrete type) with the parts the two sides legitimately
/// differ in removed, so spellings of one type compare
/// equal: lifetime arguments (`'_`, `'static`, named
/// lifetimes) and reference lifetimes (`&'a isize`), which
/// rendered type names drop and instance keys keep, and
/// the defaulted allocator argument
/// (`std::alloc::Global`), which rendered names spell out
/// and instance keys elide. Argument lists are processed at
/// every nesting level; a list emptied by the removals is
/// dropped with its brackets. A closure-type argument, in
/// either of its two spellings, renders as the bare segment
/// `{closure}` (see `canonical_arg`). Function-item types are first
/// canonicalised to the item's path (see `fn_item_spelling`),
/// which the two sides spell differently.
pub(super) fn normalize_impl_self(name: &str) -> String {
    let name = &strip_reference_lifetimes(&fn_item_spelling(name));
    fn list(chars: &[char], start: usize) -> (String, usize) {
        // `chars[start]` is '<': render the argument list
        // (possibly to nothing) and return the index just
        // past its closing '>'.
        let mut args: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut i = start + 1;
        let mut depth = 0usize;
        while i < chars.len() {
            let c = chars[i];
            if c == '-' && chars.get(i + 1) == Some(&'>') {
                current.push_str("->");
                i += 2;
            } else if c == '<' {
                let (rendered, next) = list(chars, i);
                current.push_str(&rendered);
                i = next;
            } else if c == '>' && depth == 0 {
                let arg = current.trim().to_string();
                if !arg.is_empty() {
                    args.push(canonical_arg(arg));
                }
                let kept: Vec<&str> = args
                    .iter()
                    .map(String::as_str)
                    .filter(|a| {
                        !a.starts_with('\'')
                            && *a != "std::alloc::Global"
                            && *a != "alloc::alloc::Global"
                    })
                    .collect();
                if kept.is_empty() {
                    return (String::new(), i + 1);
                }
                return (format!("<{}>", kept.join(", ")), i + 1);
            } else if c == '(' || c == '[' {
                depth += 1;
                current.push(c);
                i += 1;
            } else if c == ')' || c == ']' {
                depth = depth.saturating_sub(1);
                current.push(c);
                i += 1;
            } else if c == ',' && depth == 0 {
                let arg = current.trim().to_string();
                if !arg.is_empty() {
                    args.push(canonical_arg(arg));
                }
                current = String::new();
                i += 1;
            } else {
                current.push(c);
                i += 1;
            }
        }
        (current, i)
    }
    let chars: Vec<char> = name.chars().collect();
    let mut out = String::new();
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i] == '<' {
            let (rendered, next) = list(&chars, i);
            out.push_str(&rendered);
            i = next;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    canonical_arg(out.trim().to_string())
}

/// The canonical rendering of one generic argument of a
/// `Self` spelling (or of a whole `Self` spelling): an
/// argument that is a closure type, in either of its two
/// spellings — the span segment an instance key carries
/// (`{closure@<file>:16:17: 16:23}`, with no argument
/// list), the indexed path segment a rendered type
/// carries (`main::{closure#0}`, with the closure's own
/// generic arguments appended) — renders as the bare
/// segment `{closure}`. The two spellings name one type
/// and share no text, and neither carries the other's
/// discriminant (the span has no index, the path no
/// span), so the discriminant is erased here and the
/// scans' uniqueness contract (see `impl_self_matches`)
/// keeps two closures holding impls of one trait method
/// in one program unresolved rather than mis-dispatched.
/// Any other argument renders as itself.
fn canonical_arg(arg: String) -> String {
    let head = arg.split('<').next().unwrap_or(&arg);
    let segment = head
        .rsplit("::")
        .next()
        .unwrap_or(head)
        .trim_start_matches('&')
        .trim();
    if segment.starts_with("{closure@") || segment.starts_with("{closure#") {
        "{closure}".to_string()
    } else {
        arg
    }
}
/// Whether two canonical `Self` spellings (see
/// `normalize_impl_self`) denote the same type, where the
/// first spelling is a recorded key's: an argument of the
/// key that is a bare identifier is the generic
/// definition's parameter standing in that position — a
/// body recorded under its generic definition name
/// (`YCombinator<F, A, R>`, or a partially instantiated
/// `GroupBy<InGroup<...>, F, I>`) serves a call
/// monomorphised at any instantiation of that parameter —
/// and matches any one argument of the query. A bare
/// identifier that names a primitive type is not a
/// parameter: `Foo<i32>` and `Foo<u8>` are different
/// types. Every other part of the two spellings must
/// agree exactly, argument list by argument list.
/// Mirrors the copy of the same name in the other
/// `impl_self_matches` home.
fn canonical_self_matches(key: &str, query: &str) -> bool {
    // A spelling split into its head and, when it carries
    // one that runs to the spelling's end, the arguments
    // of its outermost list.
    fn split(spelling: &str) -> Option<(String, Option<Vec<String>>)> {
        let chars: Vec<char> = spelling.chars().collect();
        let mut i = 0usize;
        let mut head = String::new();
        while i < chars.len() && chars[i] != '<' {
            head.push(chars[i]);
            i += 1;
        }
        if i == chars.len() {
            return Some((head.trim().to_string(), None));
        }
        i += 1;
        let mut args = Vec::new();
        let mut current = String::new();
        let mut angle = 0usize;
        let mut paren = 0usize;
        while i < chars.len() {
            let c = chars[i];
            if c == '-' && chars.get(i + 1) == Some(&'>') {
                current.push_str("->");
                i += 2;
            } else if c == '<' {
                angle += 1;
                current.push(c);
                i += 1;
            } else if c == '>' && angle == 0 && paren == 0 {
                let arg = current.trim().to_string();
                if !arg.is_empty() {
                    args.push(arg);
                }
                i += 1;
                if i != chars.len() {
                    return None;
                }
                return Some((head.trim().to_string(), Some(args)));
            } else if c == '>' && angle == 0 {
                // The '>' of a fn-pointer return inside
                // parentheses, already shielded by the `->`
                // case; a bare one here is unbalanced.
                return None;
            } else if c == '>' {
                angle -= 1;
                current.push(c);
                i += 1;
            } else if c == '(' || c == '[' {
                paren += 1;
                current.push(c);
                i += 1;
            } else if c == ')' || c == ']' {
                paren = paren.saturating_sub(1);
                current.push(c);
                i += 1;
            } else if c == ',' && angle == 0 && paren == 0 {
                let arg = current.trim().to_string();
                if !arg.is_empty() {
                    args.push(arg);
                }
                current = String::new();
                i += 1;
            } else {
                current.push(c);
                i += 1;
            }
        }
        None
    }
    let Some((key_head, key_args)) = split(key) else {
        return false;
    };
    let Some((query_head, query_args)) = split(query) else {
        return false;
    };
    if key_head != query_head {
        // The crate prefix may be present on either side
        // or on neither; two different qualified heads are
        // different types.
        let key_short = key_head.rsplit("::").next().unwrap_or(&key_head);
        let query_short = query_head.rsplit("::").next().unwrap_or(&query_head);
        if (key_head.contains("::") && query_head.contains("::"))
            || key_short != query_short
        {
            return false;
        }
    }
    match (key_args, query_args) {
        (None, None) => true,
        (Some(key_args), Some(query_args)) => {
            key_args.len() == query_args.len()
                && key_args.iter().zip(&query_args).all(|(key_arg, query_arg)| {
                    is_definition_param(key_arg)
                        || canonical_self_matches(key_arg, query_arg)
                })
        }
        _ => false,
    }
}

/// Whether an impl key's `Self` section denotes `name`
/// once the regions in the key's spelling are erased: the
/// fallback for a receiver whose rendered concrete type
/// dropped every region (see `erase_regions`). The
/// comparison is the canonical one (see
/// `canonical_self_matches`) between the key's erased
/// canonical spelling and the queried name's canonical
/// spelling, with one structural exclusion: a key whose
/// canonical `Self` spelling carries a binder (a
/// `for<'a> fn(&'a u8)` argument) never matches by
/// erasure. The higher-ranked impl's binder survives in
/// its key, while an erased spelling has no binder to
/// offer, so the erased form cannot select it: among the
/// region variants of one erased shape, only the
/// binderless keys take part, and the scans' uniqueness
/// contract keeps the call unresolved when more than one
/// of them remains. Mirrors the copy of the same name in the other
/// `impl_self_matches` home.
pub(super) fn impl_self_matches_erased(self_part: &str, name: &str) -> bool {
    let key = normalize_impl_self(self_part);
    if key.contains("for<") {
        return false;
    }
    canonical_self_matches(&erase_regions(&key), &normalize_impl_self(name))
}

/// The spelling of a type with every lifetime token
/// removed: `'static`, `'a`, `'_` and the like, together
/// with the one space that follows the token in a
/// reference (`&'static u8` renders as `&u8`, `&'a mut u8`
/// as `&mut u8`). This is the erasure a rendered concrete
/// type applies to regions in the positions the canonical
/// spelling keeps -- inside a function pointer's
/// signature above all, where the instance key spells
/// `fn(&'static u8)` and the fat pointer's concrete tag
/// spells `fn(&u8)`. A quote that does not start a
/// lifetime (a character constant's closing quote, an
/// escape) is left as it stands. Mirrors the copy of the same name in the other
/// `impl_self_matches` home.
pub(super) fn erase_regions(spelling: &str) -> String {
    let chars: Vec<char> = spelling.chars().collect();
    let mut out = String::with_capacity(spelling.len());
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i] == '\'' {
            let mut j = i + 1;
            while j < chars.len()
                && (chars[j].is_ascii_alphanumeric() || chars[j] == '_')
            {
                j += 1;
            }
            // A lifetime token is a non-empty identifier not
            // closed by a second quote (a character constant
            // such as `'a'` keeps both of its quotes).
            if j > i + 1 && chars.get(j) != Some(&'\'') {
                i = j;
                if chars.get(i) == Some(&' ') {
                    i += 1;
                }
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Whether a canonical argument is a generic definition's
/// parameter name standing alone: a single identifier that
/// does not name a primitive type.
fn is_definition_param(arg: &str) -> bool {
    let arg = arg.trim();
    !arg.is_empty()
        && arg.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && arg.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !matches!(
            arg,
            "i8" | "i16"
                | "i32"
                | "i64"
                | "i128"
                | "isize"
                | "u8"
                | "u16"
                | "u32"
                | "u64"
                | "u128"
                | "usize"
                | "f16"
                | "f32"
                | "f64"
                | "f128"
                | "bool"
                | "char"
                | "str"
        )
}

