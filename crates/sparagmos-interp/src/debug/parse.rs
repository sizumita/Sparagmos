//! The print-expression parser: dereferences, a base variable, static path or history reference, and the field / index steps after it.

use super::desc::Desc;

/// A parsed print expression: dereferences first (they prefix
/// the whole expression in debugger syntax), then a base
/// variable — a plain name, a path naming a static, or a
/// history reference (`$1`) naming an earlier print's value —
/// then field/index steps.
pub(super) struct ParsedExpr {
    pub(super) base: String,
    pub(super) steps: Vec<ExprStep>,
}

pub(super) enum ExprStep {
    Deref,
    Field(usize),
    FieldName(String),
    Index(usize),
    /// A cast to the given type (`as *const [T; N]` and kin).
    Cast(Desc),
}

/// Parse the steps that follow a print expression's base (or a
/// parenthesised sub-expression): `.field` / `.name` and
/// `[index]` steps, in order.
fn parse_steps(text: &str, steps: &mut Vec<ExprStep>) -> Result<(), String> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'.' => {
                i += 1;
                let s = i;
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
                {
                    i += 1;
                }
                if s == i {
                    return Err(format!("cannot parse expression `{text}`"));
                }
                let field = &text[s..i];
                match field.parse::<usize>() {
                    Ok(index) => steps.push(ExprStep::Field(index)),
                    Err(_) => steps.push(ExprStep::FieldName(field.to_string())),
                }
            }
            b'[' => {
                i += 1;
                let s = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let index: usize = text[s..i]
                    .parse()
                    .map_err(|_| format!("cannot parse index in `{text}`"))?;
                if i >= bytes.len() || bytes[i] != b']' {
                    return Err(format!("cannot parse expression `{text}`"));
                }
                i += 1;
                steps.push(ExprStep::Index(index));
            }
            b' ' => {
                i += 1;
            }
            _ => return Err(format!("cannot parse expression `{text}`")),
        }
    }
    Ok(())
}

/// The byte index of the `)` matching the `(` at the start of
/// `text` (which must start with `(`), or None when unbalanced.
fn matching_paren(text: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in text.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Split a parenthesised expression's text at its top-level
/// cast separator: the ` as ` that belongs to the whole
/// expression, not one inside a nested parenthesised
/// subexpression -- a cast chain grouped as
/// `((r as *const u32) as *const f32)`, or an extra grouping
/// layer around a single cast, whose inner separator sits
/// one level down.
fn split_top_level_cast(inner: &str) -> Option<(&str, &str)> {
    let bytes = inner.as_bytes();
    let mut depth = 0i32;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            _ => {}
        }
        if depth == 0 && inner[i..].starts_with(" as ") {
            return Some((&inner[..i], &inner[i + 4..]));
        }
        i += 1;
    }
    None
}

/// The operand of a `sizeof(<expr>)` print expression, when
/// the whole text is that form: both dialects spell the size
/// operator this way, and the session answers it from the
/// operand's declared type (see `expr_size_of` in the
/// evaluator). The closing parenthesis must end the text, so
/// a longer expression that merely starts with a sizeof form
/// is not one.
pub(super) fn parse_sizeof_operand(text: &str) -> Option<&str> {
    let rest = text.trim().strip_prefix("sizeof")?.trim_start();
    let close = matching_paren(rest)?;
    if !rest[close + 1..].trim().is_empty() {
        return None;
    }
    Some(rest[1..close].trim())
}

pub(super) fn parse_expr(text: &str) -> Result<ParsedExpr, String> {
    let text = text.trim();
    // Leading dereferences prefix the whole expression.
    let mut derefs = 0usize;
    let mut rest = text;
    while let Some(r) = rest.strip_prefix('*') {
        derefs += 1;
        rest = r.trim_start();
    }
    if let Some(after_open) = rest.strip_prefix('(') {
        let close = matching_paren(rest)
            .ok_or_else(|| format!("cannot parse expression `{text}`"))?;
        let inner = &after_open[..close - 1];
        let mut parsed = if let Some((lhs, ty)) = split_top_level_cast(inner) {
            let mut parsed = parse_expr(lhs)?;
            parsed.steps.push(ExprStep::Cast(Desc::from_tag(ty.trim())));
            parsed
        } else {
            parse_expr(inner)?
        };
        // Trailing projections apply to the parenthesised
        // value first (`*(t).0` projects the tuple, then
        // dereferences the field -- the unparenthesised form
        // below orders its steps the same way); the leading
        // dereferences apply to the projected value as a
        // whole, after those steps.
        parse_steps(&rest[close + 1..], &mut parsed.steps)?;
        for _ in 0..derefs {
            parsed.steps.push(ExprStep::Deref);
        }
        return Ok(parsed);
    }
    // A base name: a plain identifier, or a path of identifiers
    // joined by `::` (the form that names a static).
    let bytes = rest.as_bytes();
    let mut i = 0;
    let ident = |i: &mut usize| -> Result<String, String> {
        let s = *i;
        while *i < bytes.len()
            && (bytes[*i].is_ascii_alphanumeric() || bytes[*i] == b'_')
        {
            *i += 1;
        }
        if s == *i {
            return Err(format!("cannot parse expression `{text}`"));
        }
        Ok(rest[s..*i].to_string())
    };
    // A history reference (`$1`, `$12`) is a base of its own:
    // it names the value an earlier print of the session
    // produced (see `resolve_base` in the evaluator).
    if rest.starts_with('$') {
        let mut i = 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == 1 {
            return Err(format!("cannot parse expression `{text}`"));
        }
        let mut steps: Vec<ExprStep> = Vec::new();
        parse_steps(&rest[i..], &mut steps)?;
        steps.extend((0..derefs).map(|_| ExprStep::Deref));
        return Ok(ParsedExpr {
            base: rest[..i].to_string(),
            steps,
        });
    }
    let mut base = ident(&mut i)?;
    while rest[i..].starts_with("::") {
        i += 2;
        base.push_str("::");
        base.push_str(&ident(&mut i)?);
    }
    // The leading dereferences apply to the projected value
    // as a whole, after its own steps -- `*t.0` reads the
    // pointee of the field, as in Rust and C -- matching the
    // parenthesised form above. Seeding the steps with the
    // dereferences would dereference the base value itself
    // (a tuple or an aggregate) before projecting it.
    let mut steps: Vec<ExprStep> = Vec::new();
    parse_steps(&rest[i..], &mut steps)?;
    steps.extend((0..derefs).map(|_| ExprStep::Deref));
    Ok(ParsedExpr { base, steps })
}

/// Split a print expression at its top-level comparison: the
/// `==` or `!=` outside any parenthesis or bracket, returning
/// (lhs, is_equal, rhs) with both sides trimmed, or None when
/// the expression carries no top-level comparison (or a side
/// is empty). The evaluator speaks exactly one comparison, at
/// the top level of the whole expression — the form the
/// debuginfo suite's history checks use (`print $1 == $2`); a
/// parenthesised comparison is outside the grammar. Each side
/// is a print expression of the ordinary grammar, steps and
/// casts included.
pub(super) fn split_comparison(text: &str) -> Option<(&str, bool, &str)> {
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b'=' | b'!' if depth == 0 && bytes.get(i + 1) == Some(&b'=') => {
                let lhs = text[..i].trim();
                let rhs = text[i + 2..].trim();
                if lhs.is_empty() || rhs.is_empty() {
                    return None;
                }
                return Some((lhs, bytes[i] == b'=', rhs));
            }
            _ => {}
        }
        i += 1;
    }
    None
}
