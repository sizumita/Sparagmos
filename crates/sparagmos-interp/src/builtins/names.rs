//! Callee names and type tags: canonical_name and ty_tag.

use std::collections::HashMap;

use sparagmos_mir as mir;



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
        T::RawPtr(inner, mutability) => {
            let m = if matches!(mutability, mir::Mutability::Mut) {
                "mut"
            } else {
                "const"
            };
            format!("*{m} {}", ty_tag(inner, names))
        }
        T::Adt(def, args) | T::Closure(def, args) => {
            // A closure type is identified by its definition's
            // name plus its generic arguments (the upvars tuple
            // among them), the same rendering as an ADT instance:
            // two distinct closure types must not share a tag,
            // because the layout table is keyed by it (`size_of` /
            // `size_of_val` of a closure value is a per-type fact).
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
        T::Coroutine(def, args) => {
            // A coroutine type is named by its definition (the
            // async fn / coroutine literal it was lowered from),
            // with its generic arguments rendered like an ADT's, so
            // the tag identifies one monomorphic coroutine instance.
            // The driver records real layouts under this tag (a
            // coroutine type has no ADT entry), which is what lets
            // the `size_of_val` / `align_of_val` builtins answer
            // for coroutine values.
            let base = names
                .get(def)
                .cloned()
                .unwrap_or_else(|| "Coroutine".to_string());
            if args.is_empty() {
                base
            } else {
                let parts: Vec<String> = args.iter().map(|t| ty_tag(t, names)).collect();
                format!("{base}<{}>", parts.join(", "))
            }
        }
        T::Param(_, name) => name.clone(),
        T::Dynamic(bounds, ..) => {
            // The rendering the driver records in instance
            // names (`dyn std::marker::Send`): the bounds are
            // already trait path strings.
            if bounds.is_empty() {
                "dyn".to_string()
            } else {
                format!("dyn {}", bounds.join(" + "))
            }
        }
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

/// A type tag rendered from an IR type, in the spellings
/// [`ty_tag`] gives the same types at the MIR level. `None`
/// for the one IR type that does not determine its tag on its
/// own: a `u32` local is a `u32` or a `char`, the two format
/// differently, and conversion renders both as the same IR
/// type (the caller disambiguates from the body's variable
/// table when it can).
pub(super) fn ir_ty_tag(ty: &crate::ir::Ty) -> Option<String> {
    use crate::ir::Ty as T;
    Some(match ty {
        T::Unit => "()".to_string(),
        T::Bool => "bool".to_string(),
        T::Int(t) => match t {
            crate::ir::IntTy::I8 => "i8",
            crate::ir::IntTy::I16 => "i16",
            crate::ir::IntTy::I32 => "i32",
            crate::ir::IntTy::I64 => "i64",
            crate::ir::IntTy::I128 => "i128",
            crate::ir::IntTy::Isize => "isize",
        }
        .to_string(),
        T::Uint(crate::ir::UintTy::U32) => return None,
        T::Uint(t) => match t {
            crate::ir::UintTy::U8 => "u8",
            crate::ir::UintTy::U16 => "u16",
            crate::ir::UintTy::U32 => "u32",
            crate::ir::UintTy::U64 => "u64",
            crate::ir::UintTy::U128 => "u128",
            crate::ir::UintTy::Usize => "usize",
        }
        .to_string(),
        T::Float(t) => match t {
            crate::ir::FloatTy::F16 => "f16",
            crate::ir::FloatTy::F32 => "f32",
            crate::ir::FloatTy::F64 => "f64",
            crate::ir::FloatTy::F128 => "f128",
        }
        .to_string(),
        T::Tuple(parts) => {
            let parts: Option<Vec<String>> = parts.iter().map(ir_ty_tag).collect();
            let parts = parts?;
            if parts.len() == 1 {
                format!("({},)", parts[0])
            } else {
                format!("({})", parts.join(", "))
            }
        }
        T::Adt(name) => name.clone(),
        T::Ref(inner) => format!("&{}", ir_ty_tag(inner)?),
        T::Slice(elem) => format!("[{}]", ir_ty_tag(elem)?),
        T::Array(elem, n) => format!("[{}; {n}]", ir_ty_tag(elem)?),
        T::Str => "str".to_string(),
        T::FnPtr => "FnPtr".to_string(),
        T::FnDef(name) => name.clone(),
        T::Closure(name, _) => name.clone(),
        T::Dynamic(bounds) => {
            if bounds.is_empty() {
                "dyn".to_string()
            } else {
                format!("dyn {}", bounds.join(" + "))
            }
        }
        T::Coroutine(name) => name.clone(),
        T::Never => "!".to_string(),
    })
}
