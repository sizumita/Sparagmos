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

/// Whether an impl key's `Self` section denotes `name`: the name
/// itself, or the name with a generic argument list appended; the
/// crate prefix may be omitted on a side that carries none.
pub(super) fn impl_self_matches(self_part: &str, name: &str) -> bool {
    // The section and the queried name can be the very same
    // rendering (a `Self` that is itself an impl-path type,
    // whose base the `<` split below cannot recover).
    if self_part == name {
        return true;
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
    if self_base.contains("::") && name.contains("::") {
        return false;
    }
    self_base.rsplit("::").next().unwrap_or(self_base)
        == name.rsplit("::").next().unwrap_or(name)
}

/// The spelling of an impl `Self` type (or of a queried
/// concrete type) with the parts the two sides legitimately
/// differ in removed, so spellings of one type compare
/// equal: lifetime arguments (`'_`, `'static`, named
/// lifetimes), which rendered type names drop and instance
/// keys keep, and the defaulted allocator argument
/// (`std::alloc::Global`), which rendered names spell out
/// and instance keys elide. Argument lists are processed at
/// every nesting level; a list emptied by the removals is
/// dropped with its brackets.
pub(super) fn normalize_impl_self(name: &str) -> String {
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
                    args.push(arg);
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
                    args.push(arg);
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
    out.trim().to_string()
}
