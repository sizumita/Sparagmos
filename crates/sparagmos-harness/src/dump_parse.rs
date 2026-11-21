//! Reader for the driver's dump format v0 (`sparagmos-mir-dump v0`,
//! see `sparagmos-driver`'s `dump` module): the text the driver
//! writes with `--dump`, reconstructed into the JIT's program input.
//!
//! The per-function payload of the format is the `Debug`
//! representation of `sparagmos_mir::Body`. This reader parses that
//! representation generically first (a small tree of atoms, calls,
//! structs, tuples and lists — the whole Debug grammar these types
//! use), then converts the tree into `sparagmos-mir` values with
//! explicit per-variant handling. A construct the conversion does
//! not handle is an error naming the construct, never a guess: the
//! caller (the harness's JIT runner) reports it as an unsupported
//! case with that reason.

use std::collections::HashMap;
use std::fmt;

use sparagmos_jit::MirProgramInput;
use sparagmos_mir as mir;

/// Why a dump could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DumpError(pub String);

impl fmt::Display for DumpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dump read error: {}", self.0)
    }
}
impl std::error::Error for DumpError {}

fn err<T>(msg: impl Into<String>) -> Result<T, DumpError> {
    Err(DumpError(msg.into()))
}

/// A parsed dump: the program for the JIT, plus the driver's own
/// list of functions it skipped during conversion (name, reasons).
#[derive(Debug)]
pub struct ParsedDump {
    pub program: MirProgramInput,
    pub skipped: Vec<(String, String)>,
}

/// Parse a whole dump file's text.
pub fn parse_dump(text: &str) -> Result<ParsedDump, DumpError> {
    let mut lines = text.lines();
    match lines.next() {
        Some("sparagmos-mir-dump v0") => {}
        other => return err(format!("not a v0 dump (first line: {other:?})")),
    }
    let mut def_names: HashMap<mir::DefId, String> = HashMap::new();
    let mut functions: Vec<sparagmos_jit::MirFunctionInput> = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();

    // Sections are separated by `=== ` header lines; everything
    // before the first one is the header (source / functions /
    // def-names).
    enum Section {
        Header,
        Fn(String),
        Skipped,
    }

    fn finish_section(
        section: &Section,
        text: &str,
        def_names: &mut HashMap<mir::DefId, String>,
        functions: &mut Vec<sparagmos_jit::MirFunctionInput>,
        skipped: &mut Vec<(String, String)>,
    ) -> Result<(), DumpError> {
        match section {
            Section::Header => {
                for line in text.lines() {
                    let line = line.trim();
                    if let Some(rest) = line.strip_prefix("DefId(0:") {
                        let Some((idx, name)) = rest.split_once(") = ") else {
                            return err(format!("malformed def-names line: {line}"));
                        };
                        let idx: u32 = idx
                            .parse()
                            .map_err(|_| DumpError(format!("bad DefId index in line: {line}")))?;
                        def_names.insert(mir::DefId { krate: 0, index: idx }, name.to_string());
                    }
                }
                Ok(())
            }
            Section::Skipped => {
                for line in text.lines() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    if let Some((name, reasons)) = line.split_once(": ") {
                        skipped.push((name.to_string(), reasons.to_string()));
                    }
                }
                Ok(())
            }
            Section::Fn(name) => {
                let mut parser = Parser::new(text)?;
                let node = parser.parse_node()?;
                parser.expect_end()?;
                let body = Conv::convert_body(&node)
                    .map_err(|e| DumpError(format!("in function `{name}`: {}", e.0)))?;
                let def_id = def_names.iter().find(|(_, n)| *n == name).map(|(d, _)| *d);
                functions.push(sparagmos_jit::MirFunctionInput {
                    name: name.clone(),
                    def_id,
                    body,
                });
                Ok(())
            }
        }
    }

    let mut section = Section::Header;
    let mut text = String::new();
    for line in lines {
        if let Some(rest) = line.strip_prefix("=== ") {
            finish_section(&section, &text, &mut def_names, &mut functions, &mut skipped)?;
            text = String::new();
            section = if rest == "skipped" {
                Section::Skipped
            } else if let Some(name) = rest.strip_prefix("fn ") {
                Section::Fn(name.split(" (args=").next().unwrap_or(name).to_string())
            } else {
                return err(format!("unknown dump section: {rest}"));
            };
        } else {
            text.push_str(line);
            text.push('\n');
        }
    }
    finish_section(&section, &text, &mut def_names, &mut functions, &mut skipped)?;

    Ok(ParsedDump {
        program: MirProgramInput {
            functions,
            names: def_names,
            adt_discriminants: Default::default(),
        },
        skipped,
    })
}

// ---------------------------------------------------------------------------
// Generic Debug-tree parsing
// ---------------------------------------------------------------------------

/// A node of the Debug representation.
#[derive(Debug, Clone, PartialEq)]
enum Node {
    /// A bare word: a unit variant, a number, `true` / `false`.
    Atom(String),
    /// A quoted string literal (unescaped).
    Str(String),
    /// A quoted char literal.
    Chr(char),
    /// `Name(arg, ...)` — tuple variants and tuple structs.
    Call(String, Vec<Node>),
    /// `Name { field: value, ... }` — structs and struct variants.
    Struct(String, Vec<(String, Node)>),
    /// `[value, ...]`
    List(Vec<Node>),
    /// `(value, ...)`; `()` is the empty tuple.
    Tuple(Vec<Node>),
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Num(String),
    Str(String),
    Chr(char),
    Punct(char),
}

fn lex(text: &str) -> Result<Vec<Tok>, DumpError> {
    let chars: Vec<char> = text.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    let mut read_escape = |i: &mut usize| -> Result<char, DumpError> {
        // chars[*i] is the backslash.
        *i += 1;
        let c = *chars.get(*i).ok_or_else(|| DumpError("truncated escape".into()))?;
        let out = match c {
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            '0' => '\0',
            '\\' => '\\',
            '"' => '"',
            '\'' => '\'',
            'u' => {
                // \u{XXXX}
                if chars.get(*i + 1) != Some(&'{') {
                    return err("malformed \\u escape");
                }
                let mut hex = String::new();
                let mut j = *i + 2;
                while let Some(&c) = chars.get(j) {
                    if c == '}' {
                        break;
                    }
                    hex.push(c);
                    j += 1;
                }
                let v = u32::from_str_radix(&hex, 16)
                    .map_err(|_| DumpError("bad \\u escape".into()))?;
                *i = j;
                char::from_u32(v).ok_or_else(|| DumpError("bad \\u code point".into()))?
            }
            other => return err(format!("unknown escape \\{other}")),
        };
        Ok(out)
    };
    while i < chars.len() {
        let c = chars[i];
        match c {
            c if c.is_whitespace() => i += 1,
            '"' => {
                i += 1;
                let mut s = String::new();
                loop {
                    let c = *chars
                        .get(i)
                        .ok_or_else(|| DumpError("unterminated string".into()))?;
                    match c {
                        '"' => {
                            i += 1;
                            break;
                        }
                        '\\' => {
                            s.push(read_escape(&mut i)?);
                            i += 1;
                        }
                        _ => {
                            s.push(c);
                            i += 1;
                        }
                    }
                }
                toks.push(Tok::Str(s));
            }
            '\'' => {
                i += 1;
                let c = *chars
                    .get(i)
                    .ok_or_else(|| DumpError("unterminated char".into()))?;
                let out = if c == '\\' {
                    let e = read_escape(&mut i)?;
                    i += 1;
                    e
                } else {
                    i += 1;
                    c
                };
                if chars.get(i) != Some(&'\'') {
                    return err("malformed char literal");
                }
                i += 1;
                toks.push(Tok::Chr(out));
            }
            c if c.is_alphabetic() || c == '_' => {
                let mut s = String::new();
                while let Some(&c) = chars.get(i) {
                    if c.is_alphanumeric() || c == '_' {
                        s.push(c);
                        i += 1;
                    } else {
                        break;
                    }
                }
                toks.push(Tok::Ident(s));
            }
            c if c.is_ascii_digit()
                || (c == '-' && chars.get(i + 1).is_some_and(|n| n.is_ascii_digit())) =>
            {
                let mut s = String::new();
                while let Some(&c) = chars.get(i) {
                    if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-'
                    {
                        s.push(c);
                        i += 1;
                    } else {
                        break;
                    }
                }
                toks.push(Tok::Num(s));
            }
            '(' | ')' | '{' | '}' | '[' | ']' | ':' | ',' => {
                toks.push(Tok::Punct(c));
                i += 1;
            }
            other => return err(format!("unexpected character {other:?} in dump body")),
        }
    }
    Ok(toks)
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn new(text: &str) -> Result<Self, DumpError> {
        Ok(Self {
            toks: lex(text)?,
            pos: 0,
        })
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Result<Tok, DumpError> {
        let t = self
            .toks
            .get(self.pos)
            .cloned()
            .ok_or_else(|| DumpError("unexpected end of dump body".into()))?;
        self.pos += 1;
        Ok(t)
    }

    fn expect_punct(&mut self, c: char) -> Result<(), DumpError> {
        match self.next()? {
            Tok::Punct(p) if p == c => Ok(()),
            other => err(format!("expected `{c}`, found {other:?}")),
        }
    }

    fn parse_node(&mut self) -> Result<Node, DumpError> {
        match self.next()? {
            Tok::Ident(name) => match self.peek() {
                Some(Tok::Punct('{')) => {
                    self.next()?;
                    let mut fields = Vec::new();
                    if self.peek() == Some(&Tok::Punct('}')) {
                        self.next()?;
                        return Ok(Node::Struct(name, fields));
                    }
                    loop {
                        let field = match self.next()? {
                            Tok::Ident(f) => f,
                            other => return err(format!("expected field name, found {other:?}")),
                        };
                        self.expect_punct(':')?;
                        let value = self.parse_node()?;
                        fields.push((field, value));
                        match self.next()? {
                            Tok::Punct(',') => {
                                if self.peek() == Some(&Tok::Punct('}')) {
                                    self.next()?;
                                    break;
                                }
                            }
                            Tok::Punct('}') => break,
                            other => return err(format!("expected `,` or `}}`, found {other:?}")),
                        }
                    }
                    Ok(Node::Struct(name, fields))
                }
                Some(Tok::Punct('(')) => {
                    self.next()?;
                    let args = self.parse_seq(')')?;
                    Ok(Node::Call(name, args))
                }
                _ => Ok(Node::Atom(name)),
            },
            Tok::Num(s) => Ok(Node::Atom(s)),
            Tok::Str(s) => Ok(Node::Str(s)),
            Tok::Chr(c) => Ok(Node::Chr(c)),
            Tok::Punct('(') => Ok(Node::Tuple(self.parse_seq(')')?)),
            Tok::Punct('[') => Ok(Node::List(self.parse_seq(']')?)),
            other => err(format!("unexpected token {other:?}")),
        }
    }

    /// Parse `node, node, ... <close>` after the opening bracket was
    /// already consumed.
    fn parse_seq(&mut self, close: char) -> Result<Vec<Node>, DumpError> {
        let mut out = Vec::new();
        if self.peek() == Some(&Tok::Punct(close)) {
            self.next()?;
            return Ok(out);
        }
        loop {
            out.push(self.parse_node()?);
            match self.next()? {
                Tok::Punct(',') => {
                    if self.peek() == Some(&Tok::Punct(close)) {
                        self.next()?;
                        break;
                    }
                }
                Tok::Punct(c) if c == close => break,
                other => return err(format!("expected `,` or `{close}`, found {other:?}")),
            }
        }
        Ok(out)
    }

    fn expect_end(&self) -> Result<(), DumpError> {
        if self.pos == self.toks.len() {
            Ok(())
        } else {
            err(format!(
                "trailing tokens in dump body ({} of {})",
                self.pos,
                self.toks.len()
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Tree -> sparagmos-mir conversion
// ---------------------------------------------------------------------------

enum Payload<'a> {
    Unit,
    Args(&'a [Node]),
    Fields(&'a [(String, Node)]),
}

fn variant<'a>(n: &'a Node) -> Result<(&'a str, Payload<'a>), DumpError> {
    match n {
        Node::Atom(name) => Ok((name, Payload::Unit)),
        Node::Call(name, args) => Ok((name, Payload::Args(args))),
        Node::Struct(name, fields) => Ok((name, Payload::Fields(fields))),
        other => err(format!("expected a variant, found {other:?}")),
    }
}

fn args<'a>(n: &'a Node, what: &str) -> Result<&'a [Node], DumpError> {
    match variant(n)? {
        (_, Payload::Args(a)) => Ok(a),
        (name, _) => err(format!("{what} variant {name} has an unexpected shape")),
    }
}

fn fields<'a>(n: &'a Node, what: &str) -> Result<&'a [(String, Node)], DumpError> {
    match variant(n)? {
        (_, Payload::Fields(f)) => Ok(f),
        (name, _) => err(format!("{what} variant {name} has an unexpected shape")),
    }
}

fn field<'a>(fields: &'a [(String, Node)], name: &str) -> Result<&'a Node, DumpError> {
    fields
        .iter()
        .find(|(f, _)| f == name)
        .map(|(_, v)| v)
        .ok_or_else(|| DumpError(format!("missing field `{name}`")))
}

fn atom(n: &Node) -> Result<&str, DumpError> {
    match n {
        Node::Atom(s) => Ok(s),
        other => err(format!("expected an atom, found {other:?}")),
    }
}

fn atom_parse<T: std::str::FromStr>(n: &Node, what: &str) -> Result<T, DumpError> {
    atom(n)?
        .parse()
        .map_err(|_| DumpError(format!("bad {what}: {:?}", atom(n).unwrap_or("?"))))
}

fn as_bool(n: &Node) -> Result<bool, DumpError> {
    match atom(n)? {
        "true" => Ok(true),
        "false" => Ok(false),
        other => err(format!("expected a bool, found {other}")),
    }
}

fn as_list(n: &Node) -> Result<&[Node], DumpError> {
    match n {
        Node::List(items) => Ok(items),
        other => err(format!("expected a list, found {other:?}")),
    }
}

fn as_opt(n: &Node) -> Result<Option<&Node>, DumpError> {
    match variant(n)? {
        ("Some", Payload::Args(a)) if a.len() == 1 => Ok(Some(&a[0])),
        ("None", Payload::Unit) => Ok(None),
        (name, _) => err(format!("expected an Option, found {name}")),
    }
}

fn unsupported<T>(what: &str, name: &str) -> Result<T, DumpError> {
    err(format!("unsupported {what} variant `{name}` in dump"))
}

struct Conv;

impl Conv {
    fn convert_body(n: &Node) -> Result<mir::Body, DumpError> {
        let f = fields(n, "Body")?;
        let mut basic_blocks = Vec::new();
        for b in as_list(field(f, "basic_blocks")?)? {
            basic_blocks.push(Self::basic_block(b)?);
        }
        let mut local_decls = Vec::new();
        for d in as_list(field(f, "local_decls")?)? {
            local_decls.push(Self::local_decl(d)?);
        }
        Ok(mir::Body {
            basic_blocks,
            local_decls,
            arg_count: atom_parse(field(f, "arg_count")?, "arg_count")?,
            var_debug_info: Vec::new(),
            span: mir::Span::DUMMY,
            required_consts: Vec::new(),
        })
    }

    fn local_decl(n: &Node) -> Result<mir::LocalDecl, DumpError> {
        let f = fields(n, "LocalDecl")?;
        Ok(mir::LocalDecl {
            ty: Self::ty(field(f, "ty")?)?,
            mutability: Self::mutability(field(f, "mutability")?)?,
        })
    }

    fn mutability(n: &Node) -> Result<mir::Mutability, DumpError> {
        match atom(n)? {
            "Not" => Ok(mir::Mutability::Not),
            "Mut" => Ok(mir::Mutability::Mut),
            other => unsupported("Mutability", other),
        }
    }

    fn def_id(n: &Node) -> Result<mir::DefId, DumpError> {
        let f = fields(n, "DefId")?;
        Ok(mir::DefId {
            krate: atom_parse(field(f, "krate")?, "krate")?,
            index: atom_parse(field(f, "index")?, "index")?,
        })
    }

    fn ty(n: &Node) -> Result<mir::Ty, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("Bool", Payload::Unit) => mir::Ty::Bool,
            ("Char", Payload::Unit) => mir::Ty::Char,
            ("Str", Payload::Unit) => mir::Ty::Str,
            ("Never", Payload::Unit) => mir::Ty::Never,
            ("Int", Payload::Args(a)) if a.len() == 1 => mir::Ty::Int(Self::int_ty(&a[0])?),
            ("Uint", Payload::Args(a)) if a.len() == 1 => mir::Ty::Uint(Self::uint_ty(&a[0])?),
            ("Float", Payload::Args(a)) if a.len() == 1 => mir::Ty::Float(Self::float_ty(&a[0])?),
            ("Tuple", Payload::Args(a)) if a.len() == 1 => {
                mir::Ty::Tuple(Self::ty_list(&a[0])?)
            }
            ("Array", Payload::Args(a)) if a.len() == 2 => {
                mir::Ty::Array(Box::new(Self::ty(&a[0])?), Self::array_len(&a[1])?)
            }
            ("Slice", Payload::Args(a)) if a.len() == 1 => {
                mir::Ty::Slice(Box::new(Self::ty(&a[0])?))
            }
            ("Adt", Payload::Args(a)) if a.len() == 2 => {
                mir::Ty::Adt(Self::def_id(&a[0])?, Self::ty_list(&a[1])?)
            }
            ("Foreign", Payload::Args(a)) if a.len() == 1 => mir::Ty::Foreign(Self::def_id(&a[0])?),
            ("RawPtr", Payload::Args(a)) if a.len() == 2 => {
                mir::Ty::RawPtr(Box::new(Self::ty(&a[0])?), Self::mutability(&a[1])?)
            }
            ("Ref", Payload::Args(a)) if a.len() == 3 => mir::Ty::Ref(
                Self::region(&a[0])?,
                Box::new(Self::ty(&a[1])?),
                Self::mutability(&a[2])?,
            ),
            ("FnDef", Payload::Args(a)) if a.len() == 2 => {
                mir::Ty::FnDef(Self::def_id(&a[0])?, Self::ty_list(&a[1])?)
            }
            (other, _) => return unsupported("Ty", other),
        })
    }

    fn ty_list(n: &Node) -> Result<Vec<mir::Ty>, DumpError> {
        as_list(n)?.iter().map(Self::ty).collect()
    }

    fn int_ty(n: &Node) -> Result<mir::IntTy, DumpError> {
        Ok(match atom(n)? {
            "I8" => mir::IntTy::I8,
            "I16" => mir::IntTy::I16,
            "I32" => mir::IntTy::I32,
            "I64" => mir::IntTy::I64,
            "I128" => mir::IntTy::I128,
            "Isize" => mir::IntTy::Isize,
            other => return unsupported("IntTy", other),
        })
    }

    fn uint_ty(n: &Node) -> Result<mir::UintTy, DumpError> {
        Ok(match atom(n)? {
            "U8" => mir::UintTy::U8,
            "U16" => mir::UintTy::U16,
            "U32" => mir::UintTy::U32,
            "U64" => mir::UintTy::U64,
            "U128" => mir::UintTy::U128,
            "Usize" => mir::UintTy::Usize,
            other => return unsupported("UintTy", other),
        })
    }

    fn float_ty(n: &Node) -> Result<mir::FloatTy, DumpError> {
        Ok(match atom(n)? {
            "F16" => mir::FloatTy::F16,
            "F32" => mir::FloatTy::F32,
            "F64" => mir::FloatTy::F64,
            "F128" => mir::FloatTy::F128,
            other => return unsupported("FloatTy", other),
        })
    }

    fn array_len(n: &Node) -> Result<mir::ArrayLen, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("Known", Payload::Args(a)) if a.len() == 1 => {
                mir::ArrayLen::Known(atom_parse(&a[0], "array length")?)
            }
            ("Param", Payload::Args(a)) if a.len() == 1 => match &a[0] {
                Node::Str(s) => mir::ArrayLen::Param(s.clone()),
                _ => return err("ArrayLen::Param without a string"),
            },
            ("Unevaluated", Payload::Args(a)) if a.len() == 1 => {
                mir::ArrayLen::Unevaluated(Self::def_id(&a[0])?)
            }
            (other, _) => return unsupported("ArrayLen", other),
        })
    }

    fn region(n: &Node) -> Result<mir::Region, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("Erased", Payload::Unit) => mir::Region::Erased,
            ("Static", Payload::Unit) => mir::Region::Static,
            ("Placeholder", Payload::Unit) => mir::Region::Placeholder,
            ("Var", Payload::Args(a)) if a.len() == 1 => {
                mir::Region::Var(atom_parse(&a[0], "region var")?)
            }
            ("Named", Payload::Args(a)) if a.len() == 1 => match &a[0] {
                Node::Str(s) => mir::Region::Named(s.clone()),
                _ => return err("Region::Named without a string"),
            },
            (other, _) => return unsupported("Region", other),
        })
    }

    fn basic_block(n: &Node) -> Result<mir::BasicBlockData, DumpError> {
        let f = fields(n, "BasicBlockData")?;
        let mut statements = Vec::new();
        for s in as_list(field(f, "statements")?)? {
            statements.push(Self::statement(s)?);
        }
        let terminator = match as_opt(field(f, "terminator")?)? {
            Some(t) => Some(Self::terminator(t)?),
            None => None,
        };
        Ok(mir::BasicBlockData {
            statements,
            terminator,
            is_cleanup: as_bool(field(f, "is_cleanup")?)?,
        })
    }

    fn statement(n: &Node) -> Result<mir::Statement, DumpError> {
        let f = fields(n, "Statement")?;
        Ok(mir::Statement {
            source_info: mir::SourceInfo {
                span: mir::Span::DUMMY,
                scope: 0,
            },
            kind: Self::statement_kind(field(f, "kind")?)?,
        })
    }

    fn statement_kind(n: &Node) -> Result<mir::StatementKind, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("Assign", Payload::Args(a)) if a.len() == 1 => {
                let Node::Tuple(pair) = &a[0] else {
                    return err("Assign without a (place, rvalue) pair");
                };
                if pair.len() != 2 {
                    return err("Assign pair of unexpected length");
                }
                mir::StatementKind::Assign(Box::new((
                    Self::place(&pair[0])?,
                    Self::rvalue(&pair[1])?,
                )))
            }
            ("FakeRead", Payload::Args(a)) if a.len() == 1 => {
                let Node::Tuple(pair) = &a[0] else {
                    return err("FakeRead without a (cause, place) pair");
                };
                mir::StatementKind::FakeRead(Box::new((
                    Self::fake_read_cause(&pair[0])?,
                    Self::place(&pair[1])?,
                )))
            }
            ("SetDiscriminant", Payload::Fields(f)) => mir::StatementKind::SetDiscriminant {
                place: Box::new(Self::place(field(f, "place")?)?),
                variant_index: atom_parse(field(f, "variant_index")?, "variant_index")?,
            },
            ("StorageLive", Payload::Args(a)) if a.len() == 1 => {
                mir::StatementKind::StorageLive(Self::local(&a[0])?)
            }
            ("StorageDead", Payload::Args(a)) if a.len() == 1 => {
                mir::StatementKind::StorageDead(Self::local(&a[0])?)
            }
            ("Retag", Payload::Args(a)) if a.len() == 2 => mir::StatementKind::Retag(
                Self::retag_kind(&a[0])?,
                Box::new(Self::place(&a[1])?),
            ),
            ("PlaceMention", Payload::Args(a)) if a.len() == 1 => {
                mir::StatementKind::PlaceMention(Box::new(Self::place(&a[0])?))
            }
            ("AscribeUserType", Payload::Args(a)) if a.len() == 2 => {
                let Node::Tuple(pair) = &a[0] else {
                    return err("AscribeUserType without a (place, projection) pair");
                };
                mir::StatementKind::AscribeUserType(
                    Box::new((Self::place(&pair[0])?, Self::user_type_projection(&pair[1])?)),
                    Self::variance(&a[1])?,
                )
            }
            ("Coverage", Payload::Unit) => mir::StatementKind::Coverage,
            ("Intrinsic", Payload::Args(a)) if a.len() == 1 => {
                mir::StatementKind::Intrinsic(Box::new(Self::intrinsic(&a[0])?))
            }
            ("ConstEvalCounter", Payload::Unit) => mir::StatementKind::ConstEvalCounter,
            ("Nop", Payload::Unit) => mir::StatementKind::Nop,
            (other, _) => return unsupported("StatementKind", other),
        })
    }

    fn fake_read_cause(n: &Node) -> Result<mir::FakeReadCause, DumpError> {
        Ok(match atom(n)? {
            "ForMatchedGuard" => mir::FakeReadCause::ForMatchedGuard,
            "ForGuardBinding" => mir::FakeReadCause::ForGuardBinding,
            "ForLet" => mir::FakeReadCause::ForLet,
            "ForIndexOf" => mir::FakeReadCause::ForIndexOf,
            "ForVarBinding" => mir::FakeReadCause::ForVarBinding,
            other => return unsupported("FakeReadCause", other),
        })
    }

    fn retag_kind(n: &Node) -> Result<mir::RetagKind, DumpError> {
        Ok(match atom(n)? {
            "FnEntry" => mir::RetagKind::FnEntry,
            "TwoPhase" => mir::RetagKind::TwoPhase,
            "Raw" => mir::RetagKind::Raw,
            "Default" => mir::RetagKind::Default,
            other => return unsupported("RetagKind", other),
        })
    }

    fn variance(n: &Node) -> Result<mir::Variance, DumpError> {
        Ok(match atom(n)? {
            "Covariant" => mir::Variance::Covariant,
            "Invariant" => mir::Variance::Invariant,
            "Contravariant" => mir::Variance::Contravariant,
            "Bivariant" => mir::Variance::Bivariant,
            other => return unsupported("Variance", other),
        })
    }

    fn user_type_projection(n: &Node) -> Result<mir::UserTypeProjection, DumpError> {
        let f = fields(n, "UserTypeProjection")?;
        let mut projections = Vec::new();
        for p in as_list(field(f, "projections")?)? {
            match p {
                Node::Str(s) => projections.push(s.clone()),
                _ => return err("UserTypeProjection with a non-string projection"),
            }
        }
        Ok(mir::UserTypeProjection {
            user_ty_index: atom_parse(field(f, "user_ty_index")?, "user_ty_index")?,
            projections,
        })
    }

    fn intrinsic(n: &Node) -> Result<mir::NonDivergingIntrinsic, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("CopyNonOverlapping", Payload::Fields(f)) => {
                mir::NonDivergingIntrinsic::CopyNonOverlapping {
                    src: Self::operand(field(f, "src")?)?,
                    dst: Self::operand(field(f, "dst")?)?,
                    count: Self::operand(field(f, "count")?)?,
                }
            }
            ("Assume", Payload::Args(a)) if a.len() == 1 => {
                mir::NonDivergingIntrinsic::Assume(Self::operand(&a[0])?)
            }
            (other, _) => return unsupported("NonDivergingIntrinsic", other),
        })
    }

    fn local(n: &Node) -> Result<mir::Local, DumpError> {
        let a = args(n, "Local")?;
        if a.len() != 1 {
            return err("Local with an unexpected shape");
        }
        Ok(mir::Local(atom_parse(&a[0], "local index")?))
    }

    fn place(n: &Node) -> Result<mir::Place, DumpError> {
        let f = fields(n, "Place")?;
        let mut projection = Vec::new();
        for e in as_list(field(f, "projection")?)? {
            projection.push(Self::place_elem(e)?);
        }
        Ok(mir::Place {
            local: Self::local(field(f, "local")?)?,
            projection,
        })
    }

    fn place_elem(n: &Node) -> Result<mir::PlaceElem, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("Deref", Payload::Unit) => mir::PlaceElem::Deref,
            ("Field", Payload::Args(a)) if a.len() == 2 => {
                mir::PlaceElem::Field(atom_parse(&a[0], "field index")?, Self::ty(&a[1])?)
            }
            ("Index", Payload::Args(a)) if a.len() == 1 => {
                mir::PlaceElem::Index(Self::local(&a[0])?)
            }
            ("ConstantIndex", Payload::Fields(f)) => mir::PlaceElem::ConstantIndex {
                offset: atom_parse(field(f, "offset")?, "offset")?,
                min_length: atom_parse(field(f, "min_length")?, "min_length")?,
                from_end: as_bool(field(f, "from_end")?)?,
            },
            ("Subslice", Payload::Fields(f)) => mir::PlaceElem::Subslice {
                from: atom_parse(field(f, "from")?, "from")?,
                to: atom_parse(field(f, "to")?, "to")?,
                from_end: as_bool(field(f, "from_end")?)?,
            },
            ("Downcast", Payload::Args(a)) if a.len() == 2 => {
                let name = match as_opt(&a[0])? {
                    Some(Node::Str(s)) => Some(s.clone()),
                    Some(_) => return err("Downcast name is not a string"),
                    None => None,
                };
                mir::PlaceElem::Downcast(name, atom_parse(&a[1], "variant index")?)
            }
            ("OpaqueCast", Payload::Args(a)) if a.len() == 1 => {
                mir::PlaceElem::OpaqueCast(Self::ty(&a[0])?)
            }
            ("Subtype", Payload::Args(a)) if a.len() == 1 => {
                mir::PlaceElem::Subtype(Self::ty(&a[0])?)
            }
            (other, _) => return unsupported("PlaceElem", other),
        })
    }

    fn operand(n: &Node) -> Result<mir::Operand, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("Copy", Payload::Args(a)) if a.len() == 1 => {
                mir::Operand::Copy(Self::place(&a[0])?)
            }
            ("Move", Payload::Args(a)) if a.len() == 1 => {
                mir::Operand::Move(Self::place(&a[0])?)
            }
            ("Constant", Payload::Args(a)) if a.len() == 1 => {
                mir::Operand::Constant(Box::new(Self::constant(&a[0])?))
            }
            (other, _) => return unsupported("Operand", other),
        })
    }

    fn constant(n: &Node) -> Result<mir::Constant, DumpError> {
        let f = fields(n, "Constant")?;
        let user_ty = match as_opt(field(f, "user_ty")?)? {
            Some(v) => Some(atom_parse(v, "user_ty")?),
            None => None,
        };
        Ok(mir::Constant {
            literal: Self::const_kind(field(f, "literal")?)?,
            user_ty,
        })
    }

    fn const_kind(n: &Node) -> Result<mir::ConstKind, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("Value", Payload::Args(a)) if a.len() == 1 => {
                mir::ConstKind::Value(Self::const_value(&a[0])?)
            }
            ("Ty", Payload::Args(a)) if a.len() == 1 => mir::ConstKind::Ty(Self::ty(&a[0])?),
            ("Unevaluated", Payload::Fields(f)) => mir::ConstKind::Unevaluated {
                def: Self::def_id(field(f, "def")?)?,
                args: Self::ty_list(field(f, "args")?)?,
            },
            (other, _) => return unsupported("ConstKind", other),
        })
    }

    fn const_value(n: &Node) -> Result<mir::ConstValue, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("Scalar", Payload::Args(a)) if a.len() == 1 => {
                mir::ConstValue::Scalar(Self::scalar(&a[0])?)
            }
            ("ZeroSized", Payload::Unit) => mir::ConstValue::ZeroSized,
            ("Bool", Payload::Args(a)) if a.len() == 1 => {
                mir::ConstValue::Bool(as_bool(&a[0])?)
            }
            ("Char", Payload::Args(a)) if a.len() == 1 => match &a[0] {
                Node::Chr(c) => mir::ConstValue::Char(*c),
                _ => return err("Char constant without a char"),
            },
            ("Str", Payload::Args(a)) if a.len() == 1 => match &a[0] {
                Node::Str(s) => mir::ConstValue::Str(s.clone()),
                _ => return err("Str constant without a string"),
            },
            ("Bytes", Payload::Args(a)) if a.len() == 1 => {
                let mut bytes = Vec::new();
                for b in as_list(&a[0])? {
                    bytes.push(atom_parse(b, "byte")?);
                }
                mir::ConstValue::Bytes(bytes)
            }
            ("Slice", Payload::Fields(f)) => {
                let mut data = Vec::new();
                for b in as_list(field(f, "data")?)? {
                    data.push(atom_parse(b, "byte")?);
                }
                mir::ConstValue::Slice {
                    data,
                    meta: atom_parse(field(f, "meta")?, "meta")?,
                }
            }
            ("FnDef", Payload::Args(a)) if a.len() == 1 => {
                mir::ConstValue::FnDef(Self::def_id(&a[0])?)
            }
            ("Closure", Payload::Args(a)) if a.len() == 1 => {
                mir::ConstValue::Closure(Self::def_id(&a[0])?)
            }
            ("IntPtr", Payload::Args(a)) if a.len() == 1 => {
                mir::ConstValue::IntPtr(atom_parse(&a[0], "address")?)
            }
            ("ByRef", Payload::Fields(f)) => mir::ConstValue::ByRef {
                alloc_id: atom_parse(field(f, "alloc_id")?, "alloc_id")?,
                offset: atom_parse(field(f, "offset")?, "offset")?,
            },
            (other, _) => return unsupported("ConstValue", other),
        })
    }

    fn scalar(n: &Node) -> Result<mir::Scalar, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("Int", Payload::Fields(f)) => mir::Scalar::Int {
                data: atom_parse(field(f, "data")?, "scalar data")?,
                size: atom_parse(field(f, "size")?, "scalar size")?,
            },
            ("Float", Payload::Fields(f)) => mir::Scalar::Float {
                bits: atom_parse(field(f, "bits")?, "float bits")?,
                ty: Self::float_ty(field(f, "ty")?)?,
            },
            ("Ptr", Payload::Fields(f)) => mir::Scalar::Ptr {
                alloc_id: atom_parse(field(f, "alloc_id")?, "alloc_id")?,
                offset: atom_parse(field(f, "offset")?, "offset")?,
            },
            (other, _) => return unsupported("Scalar", other),
        })
    }
}

impl Conv {
    fn rvalue(n: &Node) -> Result<mir::Rvalue, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("Use", Payload::Args(a)) if a.len() == 1 => mir::Rvalue::Use(Self::operand(&a[0])?),
            ("Repeat", Payload::Args(a)) if a.len() == 2 => mir::Rvalue::Repeat(
                Self::operand(&a[0])?,
                atom_parse(&a[1], "repeat count")?,
            ),
            ("Ref", Payload::Args(a)) if a.len() == 3 => mir::Rvalue::Ref(
                Self::region(&a[0])?,
                Self::borrow_kind(&a[1])?,
                Self::place(&a[2])?,
            ),
            ("ThreadLocalRef", Payload::Args(a)) if a.len() == 1 => {
                mir::Rvalue::ThreadLocalRef(Self::def_id(&a[0])?)
            }
            ("RawPtr", Payload::Args(a)) if a.len() == 2 => mir::Rvalue::RawPtr(
                Self::mutability(&a[0])?,
                Self::place(&a[1])?,
            ),
            ("Reborrow", Payload::Args(a)) if a.len() == 3 => mir::Rvalue::Reborrow(
                Self::ty(&a[0])?,
                Self::mutability(&a[1])?,
                Self::place(&a[2])?,
            ),
            ("Len", Payload::Args(a)) if a.len() == 1 => mir::Rvalue::Len(Self::place(&a[0])?),
            ("Cast", Payload::Args(a)) if a.len() == 3 => mir::Rvalue::Cast(
                Self::cast_kind(&a[0])?,
                Self::operand(&a[1])?,
                Self::ty(&a[2])?,
            ),
            ("BinaryOp", Payload::Args(a)) if a.len() == 2 => {
                let Node::Tuple(pair) = &a[1] else {
                    return err("BinaryOp without an operand pair");
                };
                mir::Rvalue::BinaryOp(
                    Self::binop(&a[0])?,
                    Box::new((Self::operand(&pair[0])?, Self::operand(&pair[1])?)),
                )
            }
            ("CheckedBinaryOp", Payload::Args(a)) if a.len() == 2 => {
                let Node::Tuple(pair) = &a[1] else {
                    return err("CheckedBinaryOp without an operand pair");
                };
                mir::Rvalue::CheckedBinaryOp(
                    Self::binop(&a[0])?,
                    Box::new((Self::operand(&pair[0])?, Self::operand(&pair[1])?)),
                )
            }
            ("NullaryOp", Payload::Args(a)) if a.len() == 2 => {
                mir::Rvalue::NullaryOp(Self::nullary_op(&a[0])?, Self::ty(&a[1])?)
            }
            ("UnaryOp", Payload::Args(a)) if a.len() == 2 => {
                mir::Rvalue::UnaryOp(Self::unop(&a[0])?, Self::operand(&a[1])?)
            }
            ("Discriminant", Payload::Args(a)) if a.len() == 1 => {
                mir::Rvalue::Discriminant(Self::place(&a[0])?)
            }
            ("Aggregate", Payload::Args(a)) if a.len() == 2 => {
                let mut ops = Vec::new();
                for o in as_list(&a[1])? {
                    ops.push(Self::operand(o)?);
                }
                mir::Rvalue::Aggregate(Box::new(Self::aggregate_kind(&a[0])?), ops)
            }
            ("ShallowInitBox", Payload::Args(a)) if a.len() == 2 => {
                mir::Rvalue::ShallowInitBox(Self::operand(&a[0])?, Self::ty(&a[1])?)
            }
            ("CopyForDeref", Payload::Args(a)) if a.len() == 1 => {
                mir::Rvalue::CopyForDeref(Self::place(&a[0])?)
            }
            (other, _) => return unsupported("Rvalue", other),
        })
    }

    fn borrow_kind(n: &Node) -> Result<mir::BorrowKind, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("Shared", Payload::Unit) => mir::BorrowKind::Shared,
            ("Fake", Payload::Args(a)) if a.len() == 1 => {
                mir::BorrowKind::Fake(Self::fake_borrow_kind(&a[0])?)
            }
            ("Mut", Payload::Fields(f)) => mir::BorrowKind::Mut {
                allow_two_phase_borrow: as_bool(field(f, "allow_two_phase_borrow")?)?,
            },
            (other, _) => return unsupported("BorrowKind", other),
        })
    }

    fn fake_borrow_kind(n: &Node) -> Result<mir::FakeBorrowKind, DumpError> {
        Ok(match atom(n)? {
            "Deep" => mir::FakeBorrowKind::Deep,
            "Shallow" => mir::FakeBorrowKind::Shallow,
            other => return unsupported("FakeBorrowKind", other),
        })
    }

    fn cast_kind(n: &Node) -> Result<mir::CastKind, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("PointerExposeProvenance", Payload::Unit) => mir::CastKind::PointerExposeProvenance,
            ("PointerWithExposedProvenance", Payload::Unit) => {
                mir::CastKind::PointerWithExposedProvenance
            }
            ("PointerCoercion", Payload::Args(a)) if a.len() == 1 => {
                mir::CastKind::PointerCoercion(Self::pointer_coercion(&a[0])?)
            }
            ("IntToInt", Payload::Unit) => mir::CastKind::IntToInt,
            ("FloatToInt", Payload::Unit) => mir::CastKind::FloatToInt,
            ("FloatToFloat", Payload::Unit) => mir::CastKind::FloatToFloat,
            ("IntToFloat", Payload::Unit) => mir::CastKind::IntToFloat,
            ("PtrToPtr", Payload::Unit) => mir::CastKind::PtrToPtr,
            ("BoxDerefTransmute", Payload::Unit) => mir::CastKind::BoxDerefTransmute,
            ("FnPtrToPtr", Payload::Unit) => mir::CastKind::FnPtrToPtr,
            ("Transmute", Payload::Unit) => mir::CastKind::Transmute,
            ("Subtype", Payload::Unit) => mir::CastKind::Subtype,
            (other, _) => return unsupported("CastKind", other),
        })
    }

    fn pointer_coercion(n: &Node) -> Result<mir::PointerCoercion, DumpError> {
        Ok(match atom(n)? {
            "ReifyFnPointer" => mir::PointerCoercion::ReifyFnPointer,
            "ClosureFnPointer" => mir::PointerCoercion::ClosureFnPointer,
            "UnsafeFnPointer" => mir::PointerCoercion::UnsafeFnPointer,
            "Unsize" => mir::PointerCoercion::Unsize,
            "MutToConstPointer" => mir::PointerCoercion::MutToConstPointer,
            "ArrayToPointer" => mir::PointerCoercion::ArrayToPointer,
            "DynStar" => mir::PointerCoercion::DynStar,
            other => return unsupported("PointerCoercion", other),
        })
    }

    fn nullary_op(n: &Node) -> Result<mir::NullaryOp, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("SizeOf", Payload::Unit) => mir::NullaryOp::SizeOf,
            ("AlignOf", Payload::Unit) => mir::NullaryOp::AlignOf,
            ("UbChecks", Payload::Unit) => mir::NullaryOp::UbChecks,
            ("OffsetOf", Payload::Args(a)) if a.len() == 1 => {
                let mut idx = Vec::new();
                for v in as_list(&a[0])? {
                    idx.push(atom_parse(v, "offset field")?);
                }
                mir::NullaryOp::OffsetOf(idx)
            }
            (other, _) => return unsupported("NullaryOp", other),
        })
    }

    fn aggregate_kind(n: &Node) -> Result<mir::AggregateKind, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("Tuple", Payload::Unit) => mir::AggregateKind::Tuple,
            ("Array", Payload::Args(a)) if a.len() == 1 => {
                mir::AggregateKind::Array(Self::ty(&a[0])?)
            }
            ("Adt", Payload::Fields(f)) => mir::AggregateKind::Adt {
                def: Self::def_id(field(f, "def")?)?,
                variant_index: atom_parse(field(f, "variant_index")?, "variant_index")?,
                generic_args: Self::ty_list(field(f, "generic_args")?)?,
                user_ty: match as_opt(field(f, "user_ty")?)? {
                    Some(v) => Some(atom_parse(v, "user_ty")?),
                    None => None,
                },
                field_index: match as_opt(field(f, "field_index")?)? {
                    Some(v) => Some(atom_parse(v, "field_index")?),
                    None => None,
                },
            },
            ("Closure", Payload::Fields(f)) => mir::AggregateKind::Closure {
                def: Self::def_id(field(f, "def")?)?,
                generic_args: Self::ty_list(field(f, "generic_args")?)?,
            },
            ("Coroutine", Payload::Fields(f)) => mir::AggregateKind::Coroutine {
                def: Self::def_id(field(f, "def")?)?,
                generic_args: Self::ty_list(field(f, "generic_args")?)?,
            },
            ("CoroutineClosure", Payload::Fields(f)) => mir::AggregateKind::CoroutineClosure {
                def: Self::def_id(field(f, "def")?)?,
                generic_args: Self::ty_list(field(f, "generic_args")?)?,
            },
            ("RawPtr", Payload::Fields(f)) => mir::AggregateKind::RawPtr {
                data_ty: Self::ty(field(f, "data_ty")?)?,
                mutability: Self::mutability(field(f, "mutability")?)?,
            },
            (other, _) => return unsupported("AggregateKind", other),
        })
    }

    fn binop(n: &Node) -> Result<mir::BinOp, DumpError> {
        use mir::BinOp as M;
        Ok(match atom(n)? {
            "Add" => M::Add,
            "AddWithOverflow" => M::AddWithOverflow,
            "AddUnchecked" => M::AddUnchecked,
            "Sub" => M::Sub,
            "SubWithOverflow" => M::SubWithOverflow,
            "SubUnchecked" => M::SubUnchecked,
            "Mul" => M::Mul,
            "MulWithOverflow" => M::MulWithOverflow,
            "MulUnchecked" => M::MulUnchecked,
            "Div" => M::Div,
            "Rem" => M::Rem,
            "BitXor" => M::BitXor,
            "BitAnd" => M::BitAnd,
            "BitOr" => M::BitOr,
            "Shl" => M::Shl,
            "ShlUnchecked" => M::ShlUnchecked,
            "Shr" => M::Shr,
            "ShrUnchecked" => M::ShrUnchecked,
            "Eq" => M::Eq,
            "Lt" => M::Lt,
            "Le" => M::Le,
            "Ne" => M::Ne,
            "Ge" => M::Ge,
            "Gt" => M::Gt,
            "Offset" => M::Offset,
            "Cmp" => M::Cmp,
            other => return unsupported("BinOp", other),
        })
    }

    fn unop(n: &Node) -> Result<mir::UnOp, DumpError> {
        Ok(match atom(n)? {
            "Not" => mir::UnOp::Not,
            "Neg" => mir::UnOp::Neg,
            "PtrMetadata" => mir::UnOp::PtrMetadata,
            other => return unsupported("UnOp", other),
        })
    }

    fn terminator(n: &Node) -> Result<mir::Terminator, DumpError> {
        let f = fields(n, "Terminator")?;
        Ok(mir::Terminator {
            source_info: mir::SourceInfo {
                span: mir::Span::DUMMY,
                scope: 0,
            },
            kind: Self::terminator_kind(field(f, "kind")?)?,
        })
    }

    fn basic_block_id(n: &Node) -> Result<mir::BasicBlock, DumpError> {
        let a = args(n, "BasicBlock")?;
        if a.len() != 1 {
            return err("BasicBlock with an unexpected shape");
        }
        Ok(mir::BasicBlock(atom_parse(&a[0], "basic block")?))
    }

    fn terminator_kind(n: &Node) -> Result<mir::TerminatorKind, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("Goto", Payload::Fields(f)) => mir::TerminatorKind::Goto {
                target: Self::basic_block_id(field(f, "target")?)?,
            },
            ("SwitchInt", Payload::Fields(f)) => mir::TerminatorKind::SwitchInt {
                discr: Self::operand(field(f, "discr")?)?,
                targets: Self::switch_targets(field(f, "targets")?)?,
            },
            ("Resume", Payload::Unit) => mir::TerminatorKind::Resume,
            ("Terminate", Payload::Args(a)) if a.len() == 1 => {
                mir::TerminatorKind::Terminate(Self::terminate_kind(&a[0])?)
            }
            ("Return", Payload::Unit) => mir::TerminatorKind::Return,
            ("Unreachable", Payload::Unit) => mir::TerminatorKind::Unreachable,
            ("Drop", Payload::Fields(f)) => mir::TerminatorKind::Drop {
                place: Self::place(field(f, "place")?)?,
                target: Self::basic_block_id(field(f, "target")?)?,
                unwind: Self::unwind_action(field(f, "unwind")?)?,
                replace: as_bool(field(f, "replace")?)?,
            },
            ("Call", Payload::Fields(f)) => mir::TerminatorKind::Call {
                func: Self::operand(field(f, "func")?)?,
                args: {
                    let mut v = Vec::new();
                    for a in as_list(field(f, "args")?)? {
                        v.push(Self::operand(a)?);
                    }
                    v
                },
                destination: Self::place(field(f, "destination")?)?,
                target: match as_opt(field(f, "target")?)? {
                    Some(t) => Some(Self::basic_block_id(t)?),
                    None => None,
                },
                unwind: Self::unwind_action(field(f, "unwind")?)?,
                call_source: Self::call_source(field(f, "call_source")?)?,
            },
            ("Assert", Payload::Fields(f)) => mir::TerminatorKind::Assert {
                cond: Self::operand(field(f, "cond")?)?,
                expected: as_bool(field(f, "expected")?)?,
                msg: Box::new(Self::assert_message(field(f, "msg")?)?),
                target: Self::basic_block_id(field(f, "target")?)?,
                unwind: Self::unwind_action(field(f, "unwind")?)?,
            },
            ("Yield", Payload::Fields(f)) => mir::TerminatorKind::Yield {
                value: Self::operand(field(f, "value")?)?,
                resume: Self::basic_block_id(field(f, "resume")?)?,
                resume_arg: Self::place(field(f, "resume_arg")?)?,
                drop: match as_opt(field(f, "drop")?)? {
                    Some(t) => Some(Self::basic_block_id(t)?),
                    None => None,
                },
            },
            ("CoroutineDrop", Payload::Unit) => mir::TerminatorKind::CoroutineDrop,
            ("FalseEdge", Payload::Fields(f)) => mir::TerminatorKind::FalseEdge {
                real_target: Self::basic_block_id(field(f, "real_target")?)?,
                imaginary_target: Self::basic_block_id(field(f, "imaginary_target")?)?,
            },
            ("FalseUnwind", Payload::Fields(f)) => mir::TerminatorKind::FalseUnwind {
                real_target: Self::basic_block_id(field(f, "real_target")?)?,
                unwind: Self::unwind_action(field(f, "unwind")?)?,
            },
            ("InlineAsm", Payload::Fields(f)) => mir::TerminatorKind::InlineAsm {
                template: match field(f, "template")? {
                    Node::Str(s) => s.clone(),
                    _ => return err("InlineAsm template is not a string"),
                },
                operands: {
                    let mut v = Vec::new();
                    for a in as_list(field(f, "operands")?)? {
                        v.push(Self::operand(a)?);
                    }
                    v
                },
                options: match field(f, "options")? {
                    Node::Str(s) => s.clone(),
                    _ => return err("InlineAsm options is not a string"),
                },
                destination: match as_opt(field(f, "destination")?)? {
                    Some(t) => Some(Self::basic_block_id(t)?),
                    None => None,
                },
                unwind: Self::unwind_action(field(f, "unwind")?)?,
            },
            ("TailCall", Payload::Fields(f)) => mir::TerminatorKind::TailCall {
                func: Self::operand(field(f, "func")?)?,
                args: {
                    let mut v = Vec::new();
                    for a in as_list(field(f, "args")?)? {
                        v.push(Self::operand(a)?);
                    }
                    v
                },
            },
            (other, _) => return unsupported("TerminatorKind", other),
        })
    }

    fn switch_targets(n: &Node) -> Result<mir::SwitchTargets, DumpError> {
        let f = fields(n, "SwitchTargets")?;
        let mut values = Vec::new();
        for v in as_list(field(f, "values")?)? {
            values.push(atom_parse(v, "switch value")?);
        }
        let mut targets = Vec::new();
        for t in as_list(field(f, "targets")?)? {
            targets.push(Self::basic_block_id(t)?);
        }
        Ok(mir::SwitchTargets {
            values,
            targets,
            otherwise: Self::basic_block_id(field(f, "otherwise")?)?,
        })
    }

    fn unwind_action(n: &Node) -> Result<mir::UnwindAction, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("Continue", Payload::Unit) => mir::UnwindAction::Continue,
            ("Unreachable", Payload::Unit) => mir::UnwindAction::Unreachable,
            ("Terminate", Payload::Args(a)) if a.len() == 1 => {
                mir::UnwindAction::Terminate(Self::unwind_terminate_reason(&a[0])?)
            }
            ("Cleanup", Payload::Args(a)) if a.len() == 1 => {
                mir::UnwindAction::Cleanup(Self::basic_block_id(&a[0])?)
            }
            (other, _) => return unsupported("UnwindAction", other),
        })
    }

    fn unwind_terminate_reason(n: &Node) -> Result<mir::UnwindTerminateReason, DumpError> {
        Ok(match atom(n)? {
            "Generic" => mir::UnwindTerminateReason::Generic,
            "InCleanup" => mir::UnwindTerminateReason::InCleanup,
            other => return unsupported("UnwindTerminateReason", other),
        })
    }

    fn terminate_kind(n: &Node) -> Result<mir::TerminateKind, DumpError> {
        Ok(match atom(n)? {
            "Resume" => mir::TerminateKind::Resume,
            "Abort" => mir::TerminateKind::Abort,
            other => return unsupported("TerminateKind", other),
        })
    }

    fn call_source(n: &Node) -> Result<mir::CallSource, DumpError> {
        Ok(match atom(n)? {
            "Normal" => mir::CallSource::Normal,
            "OverloadedOperator" => mir::CallSource::OverloadedOperator,
            "DivergingOperator" => mir::CallSource::DivergingOperator,
            other => return unsupported("CallSource", other),
        })
    }

    fn coroutine_kind(n: &Node) -> Result<mir::CoroutineKind, DumpError> {
        Ok(match atom(n)? {
            "Async" => mir::CoroutineKind::Async,
            "Gen" => mir::CoroutineKind::Gen,
            "Coroutine" => mir::CoroutineKind::Coroutine,
            other => return unsupported("CoroutineKind", other),
        })
    }

    fn assert_message(n: &Node) -> Result<mir::AssertMessage, DumpError> {
        let (name, payload) = variant(n)?;
        Ok(match (name, payload) {
            ("BoundsCheck", Payload::Fields(f)) => mir::AssertMessage::BoundsCheck {
                len: Box::new(Self::operand(field(f, "len")?)?),
                index: Box::new(Self::operand(field(f, "index")?)?),
            },
            ("Overflow", Payload::Args(a)) if a.len() == 3 => mir::AssertMessage::Overflow(
                Self::binop(&a[0])?,
                Box::new(Self::operand(&a[1])?),
                Box::new(Self::operand(&a[2])?),
            ),
            ("OverflowNeg", Payload::Args(a)) if a.len() == 1 => {
                mir::AssertMessage::OverflowNeg(Box::new(Self::operand(&a[0])?))
            }
            ("DivisionByZero", Payload::Args(a)) if a.len() == 1 => {
                mir::AssertMessage::DivisionByZero(Box::new(Self::operand(&a[0])?))
            }
            ("RemainderByZero", Payload::Args(a)) if a.len() == 1 => {
                mir::AssertMessage::RemainderByZero(Box::new(Self::operand(&a[0])?))
            }
            ("ResumedAfterReturn", Payload::Args(a)) if a.len() == 1 => {
                mir::AssertMessage::ResumedAfterReturn(Self::coroutine_kind(&a[0])?)
            }
            ("ResumedAfterPanic", Payload::Args(a)) if a.len() == 1 => {
                mir::AssertMessage::ResumedAfterPanic(Self::coroutine_kind(&a[0])?)
            }
            ("ResumedAfterDrop", Payload::Args(a)) if a.len() == 1 => {
                mir::AssertMessage::ResumedAfterDrop(Self::coroutine_kind(&a[0])?)
            }
            ("MisalignedPointerDereference", Payload::Fields(f)) => {
                mir::AssertMessage::MisalignedPointerDereference {
                    required: Box::new(Self::operand(field(f, "required")?)?),
                    found: Box::new(Self::operand(field(f, "found")?)?),
                }
            }
            ("NullPointerDereference", Payload::Unit) => {
                mir::AssertMessage::NullPointerDereference
            }
            ("InvalidEnumConstruction", Payload::Args(a)) if a.len() == 1 => {
                mir::AssertMessage::InvalidEnumConstruction(Box::new(Self::operand(&a[0])?))
            }
            ("Custom", Payload::Args(a)) if a.len() == 1 => match &a[0] {
                Node::Str(s) => mir::AssertMessage::Custom(s.clone()),
                _ => return err("Custom assert message is not a string"),
            },
            (other, _) => return unsupported("AssertMessage", other),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
sparagmos-mir-dump v0
source: sample.rs
functions: 1
def-names:
  DefId(0:0) = sample::main

=== fn sample::main (args=0, blocks=1, locals=2)
Body {
    basic_blocks: [
        BasicBlockData {
            statements: [
                Statement {
                    source_info: SourceInfo {
                        span: Span {
                            lo: 0,
                            hi: 0,
                        },
                        scope: 0,
                    },
                    kind: Assign(
                        (
                            Place {
                                local: Local(
                                    1,
                                ),
                                projection: [],
                            },
                            Use(
                                Constant(
                                    Constant {
                                        literal: Value(
                                            Scalar(
                                                Int {
                                                    data: 41,
                                                    size: 4,
                                                },
                                            ),
                                        ),
                                        user_ty: None,
                                    },
                                ),
                            ),
                        ),
                    ),
                },
            ],
            terminator: Some(
                Terminator {
                    source_info: SourceInfo {
                        span: Span {
                            lo: 0,
                            hi: 0,
                        },
                        scope: 0,
                    },
                    kind: Return,
                },
            ),
            is_cleanup: false,
        },
    ],
    local_decls: [
        LocalDecl {
            ty: Tuple(
                [],
            ),
            mutability: Not,
        },
        LocalDecl {
            ty: Int(
                I32,
            ),
            mutability: Not,
        },
    ],
    arg_count: 0,
    var_debug_info: [],
    span: Span {
        lo: 0,
        hi: 0,
    },
    required_consts: [],
}
";

    #[test]
    fn parses_a_v0_dump() {
        let parsed = parse_dump(SAMPLE).expect("sample dump parses");
        assert_eq!(parsed.program.functions.len(), 1);
        let f = &parsed.program.functions[0];
        assert_eq!(f.name, "sample::main");
        assert_eq!(f.def_id, Some(mir::DefId { krate: 0, index: 0 }));
        assert_eq!(f.body.local_decls.len(), 2);
        assert_eq!(f.body.basic_blocks.len(), 1);
        assert!(parsed.skipped.is_empty());
        assert_eq!(
            parsed.program.names.get(&mir::DefId { krate: 0, index: 0 }),
            Some(&"sample::main".to_string())
        );
    }

    #[test]
    fn rejects_a_non_dump() {
        assert!(parse_dump("hello\n").is_err());
    }
}
