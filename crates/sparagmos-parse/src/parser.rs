//! Hand-written recursive-descent parser for textual MIR.
//!
//! Grammar (first tranche — see lib docs for the honest scope):
//! file := body* ; body := "fn" path "(" args ")" ("->" ty)? "{" decls blocks "}"

use crate::ast::*;
use crate::error::{ParseError, ParseResult};
use crate::lexer::{tokenize, Token, TokenKind};

pub fn parse_mir(src: &str) -> ParseResult<MirFile> {
    let tokens = tokenize(src)?;
    let mut p = Parser {
        toks: tokens,
        pos: 0,
    };
    p.parse_file()
}

struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

fn token_text(k: &TokenKind) -> String {
    match k {
        TokenKind::Ident(s) => s.clone(),
        TokenKind::IntLit(s) => s.clone(),
        TokenKind::FloatLit(s) => s.clone(),
        TokenKind::StrLit(s) => format!("\"{s}\""),
        TokenKind::CharLit(c) => format!("'{c}'"),
        TokenKind::LParen => "(".into(),
        TokenKind::RParen => ")".into(),
        TokenKind::LBrace => "{".into(),
        TokenKind::RBrace => "}".into(),
        TokenKind::LBracket => "[".into(),
        TokenKind::RBracket => "]".into(),
        TokenKind::Comma => ",".into(),
        TokenKind::Semi => ";".into(),
        TokenKind::Colon => ":".into(),
        TokenKind::ColonColon => "::".into(),
        TokenKind::Dot => ".".into(),
        TokenKind::Eq => "=".into(),
        TokenKind::Arrow => "->".into(),
        TokenKind::FatArrow => "=>".into(),
        TokenKind::Bang => "!".into(),
        TokenKind::Amp => "&".into(),
        TokenKind::Star => "*".into(),
        TokenKind::Minus => "-".into(),
        TokenKind::Plus => "+".into(),
        TokenKind::Lt => "<".into(),
        TokenKind::Gt => ">".into(),
        TokenKind::Underscore => "_".into(),
        TokenKind::Eof => "<eof>".into(),
    }
}

fn local_index(name: &str) -> Option<u32> {
    name.strip_prefix('_')?.parse().ok()
}

fn block_index(name: &str) -> Option<u32> {
    name.strip_prefix("bb")?.parse().ok()
}

impl Parser {
    fn peek(&self) -> &Token {
        &self.toks[self.pos.min(self.toks.len() - 1)]
    }
    fn peek2(&self) -> &Token {
        &self.toks[(self.pos + 1).min(self.toks.len() - 1)]
    }
    fn bump(&mut self) -> Token {
        let t = self.peek().clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }
    fn err<T>(&self, msg: impl Into<String>) -> ParseResult<T> {
        let t = self.peek();
        Err(ParseError::new(t.line, t.col, msg))
    }
    fn at_eof(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Eof)
    }
    fn at_ident(&self, name: &str) -> bool {
        matches!(&self.peek().kind, TokenKind::Ident(s) if s == name)
    }
    fn eat_ident(&mut self, name: &str) -> bool {
        if self.at_ident(name) {
            self.bump();
            true
        } else {
            false
        }
    }
    fn expect(&mut self, kind: &TokenKind) -> ParseResult<Token> {
        if &self.peek().kind == kind {
            Ok(self.bump())
        } else {
            let t = self.peek().clone();
            Err(ParseError::new(
                t.line,
                t.col,
                format!(
                    "expected `{}`, found `{}`",
                    token_text(kind),
                    token_text(&t.kind)
                ),
            ))
        }
    }
    fn expect_ident(&mut self, name: &str) -> ParseResult<Token> {
        if self.at_ident(name) {
            Ok(self.bump())
        } else {
            let t = self.peek().clone();
            Err(ParseError::new(
                t.line,
                t.col,
                format!("expected `{name}`, found `{}`", token_text(&t.kind)),
            ))
        }
    }

    // ---------- file / body ----------

    fn parse_file(&mut self) -> ParseResult<MirFile> {
        let mut bodies = Vec::new();
        while !self.at_eof() {
            bodies.push(self.parse_body()?);
        }
        Ok(MirFile { bodies })
    }

    fn parse_body(&mut self) -> ParseResult<Body> {
        // optional qualifiers
        while self.at_ident("const") || self.at_ident("async") || self.at_ident("unsafe") {
            self.bump();
        }
        let fn_tok = self.expect_ident("fn")?;
        let name = self.parse_path_name()?;
        // optional generic params <...>
        if self.peek().kind == TokenKind::Lt {
            self.skip_balanced('<', '>')?;
        }
        self.expect(&TokenKind::LParen)?;
        let mut args = Vec::new();
        while self.peek().kind != TokenKind::RParen {
            if self.at_ident("mut") {
                self.bump();
            }
            let local = self.parse_local()?;
            self.expect(&TokenKind::Colon)?;
            let ty = self.parse_ty()?;
            args.push(LocalDecl {
                local,
                mutable: false,
                ty,
            });
            if self.peek().kind == TokenKind::Comma {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(&TokenKind::RParen)?;
        let return_ty = if self.peek().kind == TokenKind::Arrow {
            self.bump();
            self.parse_ty()?
        } else {
            Ty::Unit
        };
        self.expect(&TokenKind::LBrace)?;
        let mut body = Body {
            name,
            args,
            return_ty,
            locals: Vec::new(),
            debug_vars: Vec::new(),
            blocks: Vec::new(),
            line: fn_tok.line,
        };
        // declarations until first basic block
        loop {
            match &self.peek().kind {
                TokenKind::RBrace => {
                    self.bump();
                    return Ok(body);
                }
                TokenKind::Ident(s) if block_index(s).is_some() => break,
                TokenKind::Ident(s) if s == "let" || s == "debug" || s == "scope" => {
                    self.parse_decl_item(&mut body)?;
                }
                TokenKind::Eof => return self.err("unexpected end of file inside function body"),
                _ => {
                    let t = self.peek().clone();
                    return Err(ParseError::new(
                        t.line,
                        t.col,
                        format!(
                            "unsupported item in function body: `{}` (expected `let`, `debug`, `scope`, or a basic block)",
                            token_text(&t.kind)
                        ),
                    ));
                }
            }
        }
        // basic blocks
        loop {
            match &self.peek().kind {
                TokenKind::RBrace => {
                    self.bump();
                    return Ok(body);
                }
                TokenKind::Ident(s) if block_index(s).is_some() => {
                    let b = self.parse_block()?;
                    body.blocks.push(b);
                }
                TokenKind::Eof => return self.err("unexpected end of file inside function body"),
                _ => {
                    let t = self.peek().clone();
                    return Err(ParseError::new(
                        t.line,
                        t.col,
                        format!(
                            "unsupported item between basic blocks: `{}`",
                            token_text(&t.kind)
                        ),
                    ));
                }
            }
        }
    }

    fn parse_decl_item(&mut self, body: &mut Body) -> ParseResult<()> {
        if self.at_ident("let") {
            self.bump();
            let mutable = self.eat_ident("mut");
            let local = self.parse_local()?;
            self.expect(&TokenKind::Colon)?;
            let ty = self.parse_ty()?;
            if self.peek().kind == TokenKind::Eq {
                let t = self.peek().clone();
                return Err(ParseError::new(
                    t.line,
                    t.col,
                    "local declarations with an initializer are not supported yet",
                ));
            }
            self.expect(&TokenKind::Semi)?;
            body.locals.push(LocalDecl { local, mutable, ty });
            Ok(())
        } else if self.at_ident("debug") {
            self.bump();
            let name = match self.bump().kind {
                TokenKind::Ident(s) => s,
                k => {
                    return Err(ParseError::new(
                        self.peek().line,
                        self.peek().col,
                        format!("expected debug variable name, found `{}`", token_text(&k)),
                    ))
                }
            };
            self.expect(&TokenKind::FatArrow)?;
            let local = self.parse_local()?;
            self.expect(&TokenKind::Semi)?;
            body.debug_vars.push(DebugVar { name, local });
            Ok(())
        } else if self.at_ident("scope") {
            self.bump();
            // scope index
            match self.bump().kind {
                TokenKind::IntLit(_) => {}
                k => {
                    return Err(ParseError::new(
                        self.peek().line,
                        self.peek().col,
                        format!("expected scope index, found `{}`", token_text(&k)),
                    ))
                }
            }
            self.expect(&TokenKind::LBrace)?;
            while self.peek().kind != TokenKind::RBrace {
                if self.at_eof() {
                    return self.err("unexpected end of file inside scope");
                }
                self.parse_decl_item(body)?;
            }
            self.expect(&TokenKind::RBrace)?;
            Ok(())
        } else {
            self.err("expected `let`, `debug`, or `scope`")
        }
    }

    fn parse_path_name(&mut self) -> ParseResult<String> {
        let mut name = match self.bump().kind {
            TokenKind::Ident(s) => s,
            k => {
                return Err(ParseError::new(
                    self.peek().line,
                    self.peek().col,
                    format!("expected function name, found `{}`", token_text(&k)),
                ))
            }
        };
        while self.peek().kind == TokenKind::ColonColon {
            self.bump();
            match self.bump().kind {
                TokenKind::Ident(s) => {
                    name.push_str("::");
                    name.push_str(&s);
                }
                TokenKind::Lt => {
                    // `<Type as Trait>::method` style segment: keep raw
                    self.pos -= 1;
                    let raw = self.capture_balanced('<', '>')?;
                    name.push_str("::");
                    name.push_str(&raw);
                }
                k => {
                    return Err(ParseError::new(
                        self.peek().line,
                        self.peek().col,
                        format!("expected path segment, found `{}`", token_text(&k)),
                    ))
                }
            }
        }
        Ok(name)
    }

    // ---------- blocks / statements / terminators ----------

    fn parse_block(&mut self) -> ParseResult<BasicBlock> {
        let label = self.bump();
        let id = match &label.kind {
            TokenKind::Ident(s) => BlockId(block_index(s).unwrap()),
            _ => unreachable!(),
        };
        self.expect(&TokenKind::Colon)?;
        self.expect(&TokenKind::LBrace)?;
        let mut statements = Vec::new();
        loop {
            match &self.peek().kind {
                TokenKind::RBrace => {
                    let t = self.peek().clone();
                    return Err(ParseError::new(
                        t.line,
                        t.col,
                        format!("basic block {} has no terminator", id),
                    ));
                }
                TokenKind::Ident(s) if s == "return" => {
                    self.bump();
                    self.expect(&TokenKind::Semi)?;
                    self.expect(&TokenKind::RBrace)?;
                    return Ok(BasicBlock {
                        id,
                        statements,
                        terminator: Terminator::Return,
                        line: label.line,
                    });
                }
                TokenKind::Ident(s) if s == "unreachable" => {
                    self.bump();
                    self.expect(&TokenKind::Semi)?;
                    self.expect(&TokenKind::RBrace)?;
                    return Ok(BasicBlock {
                        id,
                        statements,
                        terminator: Terminator::Unreachable,
                        line: label.line,
                    });
                }
                TokenKind::Ident(s) if s == "goto" => {
                    self.bump();
                    self.expect(&TokenKind::Arrow)?;
                    let target = self.parse_block_id()?;
                    self.expect(&TokenKind::Semi)?;
                    self.expect(&TokenKind::RBrace)?;
                    return Ok(BasicBlock {
                        id,
                        statements,
                        terminator: Terminator::Goto { target },
                        line: label.line,
                    });
                }
                TokenKind::Ident(s) if s == "switchInt" => {
                    let term = self.parse_switch_int()?;
                    self.expect(&TokenKind::RBrace)?;
                    return Ok(BasicBlock {
                        id,
                        statements,
                        terminator: term,
                        line: label.line,
                    });
                }
                TokenKind::Ident(s) if s == "assert" => {
                    let term = self.parse_assert()?;
                    self.expect(&TokenKind::RBrace)?;
                    return Ok(BasicBlock {
                        id,
                        statements,
                        terminator: term,
                        line: label.line,
                    });
                }
                TokenKind::Ident(s) if s == "drop" => {
                    let term = self.parse_drop()?;
                    self.expect(&TokenKind::RBrace)?;
                    return Ok(BasicBlock {
                        id,
                        statements,
                        terminator: term,
                        line: label.line,
                    });
                }
                TokenKind::Ident(s) if s == "StorageLive" || s == "StorageDead" => {
                    let is_live = s == "StorageLive";
                    self.bump();
                    self.expect(&TokenKind::LParen)?;
                    let local = self.parse_local()?;
                    self.expect(&TokenKind::RParen)?;
                    self.expect(&TokenKind::Semi)?;
                    statements.push(if is_live {
                        Statement::StorageLive(local)
                    } else {
                        Statement::StorageDead(local)
                    });
                }
                TokenKind::Ident(s) if s == "nop" => {
                    self.bump();
                    self.expect(&TokenKind::Semi)?;
                    statements.push(Statement::Nop);
                }
                _ => {
                    // assignment statement or call terminator
                    match self.parse_assign_or_call()? {
                        AssignOrCall::Stmt(s) => statements.push(s),
                        AssignOrCall::Term(t) => {
                            self.expect(&TokenKind::RBrace)?;
                            return Ok(BasicBlock {
                                id,
                                statements,
                                terminator: t,
                                line: label.line,
                            });
                        }
                    }
                }
            }
        }
    }

    fn parse_switch_int(&mut self) -> ParseResult<Terminator> {
        self.expect_ident("switchInt")?;
        self.expect(&TokenKind::LParen)?;
        let discr = self.parse_operand()?;
        self.expect(&TokenKind::RParen)?;
        self.expect(&TokenKind::Arrow)?;
        self.expect(&TokenKind::LBracket)?;
        let mut targets = Vec::new();
        let mut otherwise = None;
        while self.peek().kind != TokenKind::RBracket {
            if self.at_ident("otherwise") {
                self.bump();
                self.expect(&TokenKind::Colon)?;
                otherwise = Some(self.parse_block_id()?);
            } else {
                let neg = if self.peek().kind == TokenKind::Minus {
                    self.bump();
                    true
                } else {
                    false
                };
                let v = match self.bump().kind {
                    TokenKind::IntLit(s) => {
                        let (num, _suffix) = split_int_suffix(&s);
                        let parsed: i128 = num.replace('_', "").parse().map_err(|_| {
                            ParseError::new(
                                self.peek().line,
                                self.peek().col,
                                format!("invalid switchInt value `{s}`"),
                            )
                        })?;
                        if neg {
                            (-parsed) as u128
                        } else {
                            parsed as u128
                        }
                    }
                    TokenKind::CharLit(c) => c as u128,
                    k => {
                        return Err(ParseError::new(
                            self.peek().line,
                            self.peek().col,
                            format!("unsupported switchInt value `{}`", token_text(&k)),
                        ))
                    }
                };
                self.expect(&TokenKind::Colon)?;
                let target = self.parse_block_id()?;
                targets.push((v, target));
            }
            if self.peek().kind == TokenKind::Comma {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(&TokenKind::RBracket)?;
        self.expect(&TokenKind::Semi)?;
        Ok(Terminator::SwitchInt {
            discr,
            targets,
            otherwise,
        })
    }

    /// Parses `key: bbN` / `key: continue` entries inside `[ ... ]` target lists.
    fn parse_target_list(
        &mut self,
    ) -> ParseResult<Vec<(String, Option<BlockId>, Option<UnwindAction>)>> {
        self.expect(&TokenKind::LBracket)?;
        let mut out = Vec::new();
        while self.peek().kind != TokenKind::RBracket {
            let key = match self.bump().kind {
                TokenKind::Ident(s) => s,
                k => {
                    return Err(ParseError::new(
                        self.peek().line,
                        self.peek().col,
                        format!("expected target key, found `{}`", token_text(&k)),
                    ))
                }
            };
            // `return: bb1` / `unwind: bb2` have a colon, but
            // `unwind continue` / `unwind unreachable` do not.
            if self.peek().kind == TokenKind::Colon {
                self.bump();
            }
            if matches!(&self.peek().kind, TokenKind::Ident(s) if block_index(s).is_some()) {
                let b = self.parse_block_id()?;
                out.push((key, Some(b), None));
            } else if self.at_ident("continue") {
                self.bump();
                out.push((key, None, Some(UnwindAction::Continue)));
            } else if self.at_ident("unreachable") {
                self.bump();
                out.push((key, None, Some(UnwindAction::Unreachable)));
            } else if self.at_ident("terminate") {
                self.bump();
                out.push((key, None, Some(UnwindAction::Terminate)));
            } else {
                let t = self.peek().clone();
                return Err(ParseError::new(
                    t.line,
                    t.col,
                    format!(
                        "unsupported call/assert target value `{}`",
                        token_text(&t.kind)
                    ),
                ));
            }
            if self.peek().kind == TokenKind::Comma {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(&TokenKind::RBracket)?;
        Ok(out)
    }

    fn parse_assert(&mut self) -> ParseResult<Terminator> {
        self.expect_ident("assert")?;
        self.expect(&TokenKind::LParen)?;
        // condition, possibly negated: `!move _2` / `!move (_3.1: bool)`
        let negated = if self.peek().kind == TokenKind::Bang {
            self.bump();
            true
        } else {
            false
        };
        let cond_operand = self.parse_operand()?;
        let cond = if negated {
            Box::new(Rvalue::UnaryOp(UnOp::Not, Box::new(cond_operand)))
        } else {
            Box::new(Rvalue::Use(cond_operand))
        };
        let mut msg = String::new();
        while self.peek().kind == TokenKind::Comma {
            self.bump();
            match &self.peek().kind {
                TokenKind::StrLit(s) => {
                    msg = s.clone();
                    self.bump();
                }
                _ => {
                    // format arguments: operands, skip them (they belong to the message)
                    let _ = self.parse_operand()?;
                }
            }
        }
        self.expect(&TokenKind::RParen)?;
        self.expect(&TokenKind::Arrow)?;
        let entries = self.parse_target_list()?;
        self.expect(&TokenKind::Semi)?;
        let mut target = None;
        let mut unwind = None;
        for (key, block, action) in entries {
            match key.as_str() {
                "success" => target = block,
                "unwind" => {
                    unwind = action.or_else(|| block.map(UnwindAction::Cleanup));
                }
                _ => {}
            }
        }
        Ok(Terminator::Assert {
            cond,
            expected: true,
            msg,
            target,
            unwind,
        })
    }

    fn parse_drop(&mut self) -> ParseResult<Terminator> {
        self.expect_ident("drop")?;
        self.expect(&TokenKind::LParen)?;
        let place = self.parse_place()?;
        self.expect(&TokenKind::RParen)?;
        self.expect(&TokenKind::Arrow)?;
        let entries = self.parse_target_list()?;
        self.expect(&TokenKind::Semi)?;
        let mut target = None;
        let mut unwind = None;
        for (key, block, action) in entries {
            match key.as_str() {
                "return" => target = block,
                "unwind" => unwind = action.or_else(|| block.map(UnwindAction::Cleanup)),
                _ => {}
            }
        }
        Ok(Terminator::Drop {
            place,
            target,
            unwind,
        })
    }

    fn parse_assign_or_call(&mut self) -> ParseResult<AssignOrCall> {
        // Detect a call terminator: an `->` at paren depth 0 before the `;`.
        let is_call = {
            let mut depth_paren = 0i32;
            let mut depth_brack = 0i32;
            let mut j = self.pos;
            let mut found = false;
            while j < self.toks.len() {
                match &self.toks[j].kind {
                    TokenKind::LParen => depth_paren += 1,
                    TokenKind::RParen => depth_paren -= 1,
                    TokenKind::LBracket => depth_brack += 1,
                    TokenKind::RBracket => depth_brack -= 1,
                    TokenKind::Arrow if depth_paren == 0 && depth_brack == 0 => {
                        found = true;
                        break;
                    }
                    TokenKind::Semi if depth_paren == 0 && depth_brack == 0 => break,
                    TokenKind::Eof => break,
                    _ => {}
                }
                j += 1;
            }
            found
        };
        let place = self.parse_place()?;
        self.expect(&TokenKind::Eq)?;
        if is_call {
            // func operand: a path (`answer`, `std::fmt::...`) or an operand
            let func = if matches!(&self.peek().kind, TokenKind::Ident(s) if local_index(s).is_none() && s != "copy" && s != "move" && s != "const")
            {
                let path = self.parse_path_name()?;
                // generic args on the callee, e.g. `foo::<i32>(...)`
                let path = if self.peek().kind == TokenKind::ColonColon
                    && matches!(&self.peek2().kind, TokenKind::Lt)
                {
                    self.bump();
                    let g = self.capture_balanced('<', '>')?;
                    format!("{path}::{g}")
                } else {
                    path
                };
                Operand::Constant(Const::Named(path))
            } else {
                self.parse_operand()?
            };
            self.expect(&TokenKind::LParen)?;
            let mut args = Vec::new();
            while self.peek().kind != TokenKind::RParen {
                args.push(self.parse_operand()?);
                if self.peek().kind == TokenKind::Comma {
                    self.bump();
                } else {
                    break;
                }
            }
            self.expect(&TokenKind::RParen)?;
            self.expect(&TokenKind::Arrow)?;
            let entries = self.parse_target_list()?;
            self.expect(&TokenKind::Semi)?;
            let mut target = None;
            let mut unwind = None;
            for (key, block, action) in entries {
                match key.as_str() {
                    "return" | "success" => target = block,
                    "unwind" => unwind = action.or_else(|| block.map(UnwindAction::Cleanup)),
                    _ => {}
                }
            }
            return Ok(AssignOrCall::Term(Terminator::Call {
                func,
                args,
                destination: Some(place),
                target,
                unwind,
            }));
        }
        let rvalue = self.parse_rvalue()?;
        self.expect(&TokenKind::Semi)?;
        Ok(AssignOrCall::Stmt(Statement::Assign(place, rvalue)))
    }

    // ---------- places / operands / rvalues ----------

    pub(crate) fn parse_local(&mut self) -> ParseResult<Local> {
        let t = self.bump();
        match &t.kind {
            TokenKind::Ident(s) => match local_index(s) {
                Some(i) => Ok(Local(i)),
                None => Err(ParseError::new(
                    t.line,
                    t.col,
                    format!("expected a local like `_0`, found `{s}`"),
                )),
            },
            k => Err(ParseError::new(
                t.line,
                t.col,
                format!("expected a local, found `{}`", token_text(k)),
            )),
        }
    }

    fn parse_block_id(&mut self) -> ParseResult<BlockId> {
        let t = self.bump();
        match &t.kind {
            TokenKind::Ident(s) => match block_index(s) {
                Some(i) => Ok(BlockId(i)),
                None => Err(ParseError::new(
                    t.line,
                    t.col,
                    format!("expected a basic block like `bb0`, found `{s}`"),
                )),
            },
            k => Err(ParseError::new(
                t.line,
                t.col,
                format!("expected a basic block, found `{}`", token_text(k)),
            )),
        }
    }

    fn parse_place(&mut self) -> ParseResult<Place> {
        // parenthesised place: `(*_1)`, `(_3.0: i32)`, `(_1)`
        if self.peek().kind == TokenKind::LParen {
            self.bump();
            let mut place = if self.peek().kind == TokenKind::Star {
                self.bump();
                let mut inner = self.parse_place()?;
                inner.projections.push(Projection::Deref);
                inner
            } else {
                self.parse_place()?
            };
            // optional type ascription `: Ty` applying to the last field projection
            if self.peek().kind == TokenKind::Colon {
                self.bump();
                let ty = self.parse_ty()?;
                if let Some(Projection::Field { ty: slot, .. }) = place.projections.last_mut() {
                    *slot = Some(ty);
                }
            }
            self.expect(&TokenKind::RParen)?;
            self.parse_projections(&mut place)?;
            return Ok(place);
        }
        let local = self.parse_local()?;
        let mut place = Place {
            local,
            projections: Vec::new(),
        };
        self.parse_projections(&mut place)?;
        Ok(place)
    }

    fn parse_projections(&mut self, place: &mut Place) -> ParseResult<()> {
        loop {
            match &self.peek().kind {
                TokenKind::Dot => {
                    self.bump();
                    match self.bump().kind {
                        TokenKind::IntLit(s) => {
                            let idx: u32 = s.parse().map_err(|_| {
                                ParseError::new(
                                    self.peek().line,
                                    self.peek().col,
                                    format!("invalid field index `{s}`"),
                                )
                            })?;
                            place.projections.push(Projection::Field {
                                index: idx,
                                ty: None,
                            });
                        }
                        TokenKind::Ident(name) => {
                            // named field: not structured in tranche 1; keep as
                            // a raw error rather than dropping it silently.
                            let t = self.peek().clone();
                            return Err(ParseError::new(
                                t.line,
                                t.col,
                                format!("named field projection `.{name}` is not supported yet"),
                            ));
                        }
                        k => {
                            return Err(ParseError::new(
                                self.peek().line,
                                self.peek().col,
                                format!(
                                    "expected field index after `.`, found `{}`",
                                    token_text(&k)
                                ),
                            ))
                        }
                    }
                }
                TokenKind::LBracket => {
                    self.bump();
                    let idx = self.parse_local()?;
                    self.expect(&TokenKind::RBracket)?;
                    place.projections.push(Projection::Index(idx));
                }
                _ => break,
            }
        }
        Ok(())
    }

    fn parse_operand(&mut self) -> ParseResult<Operand> {
        if self.at_ident("copy") {
            self.bump();
            return Ok(Operand::Copy(self.parse_place()?));
        }
        if self.at_ident("move") {
            self.bump();
            return Ok(Operand::Move(self.parse_place()?));
        }
        if self.at_ident("const") {
            self.bump();
            return Ok(Operand::Constant(self.parse_const()?));
        }
        if self.peek().kind == TokenKind::Minus
            || matches!(
                self.peek().kind,
                TokenKind::IntLit(_) | TokenKind::FloatLit(_)
            )
        {
            // bare literal operand (rare in MIR dumps, but harmless)
            return Ok(Operand::Constant(self.parse_const()?));
        }
        // bare place operand
        Ok(Operand::Copy(self.parse_place()?))
    }

    fn parse_const(&mut self) -> ParseResult<Const> {
        // negative numbers
        if self.peek().kind == TokenKind::Minus {
            self.bump();
            return match self.bump().kind {
                TokenKind::IntLit(s) => {
                    let (num, suffix) = split_int_suffix(&s);
                    let v: i128 = num.replace('_', "").parse().map_err(|_| {
                        ParseError::new(
                            self.peek().line,
                            self.peek().col,
                            format!("invalid integer constant `{s}`"),
                        )
                    })?;
                    Ok(Const::Int {
                        value: -v,
                        ty: suffix,
                    })
                }
                TokenKind::FloatLit(s) => {
                    let (num, suffix) = split_float_suffix(&s);
                    Ok(Const::Float {
                        text: format!("-{num}"),
                        ty: suffix,
                    })
                }
                k => Err(ParseError::new(
                    self.peek().line,
                    self.peek().col,
                    format!(
                        "expected a numeric constant after `-`, found `{}`",
                        token_text(&k)
                    ),
                )),
            };
        }
        let t = self.bump();
        match t.kind {
            TokenKind::IntLit(s) => {
                let (num, suffix) = split_int_suffix(&s);
                let clean = num.replace('_', "");
                if suffix.starts_with('u') || suffix == "usize" {
                    let v: u128 = clean.parse().map_err(|_| {
                        ParseError::new(t.line, t.col, format!("invalid integer constant `{s}`"))
                    })?;
                    Ok(Const::Uint {
                        value: v,
                        ty: suffix,
                    })
                } else {
                    let v: i128 = clean.parse().map_err(|_| {
                        ParseError::new(t.line, t.col, format!("invalid integer constant `{s}`"))
                    })?;
                    Ok(Const::Int {
                        value: v,
                        ty: suffix,
                    })
                }
            }
            TokenKind::FloatLit(s) => {
                let (num, suffix) = split_float_suffix(&s);
                Ok(Const::Float {
                    text: num,
                    ty: suffix,
                })
            }
            TokenKind::StrLit(s) => Ok(Const::Str(s)),
            TokenKind::CharLit(c) => Ok(Const::Char(c)),
            TokenKind::LParen => {
                self.expect(&TokenKind::RParen)?;
                Ok(Const::Unit)
            }
            TokenKind::LBrace => {
                self.pos -= 1;
                let raw = self.capture_balanced('{', '}')?;
                Ok(Const::Raw(raw))
            }
            TokenKind::Ident(s) => {
                if s == "true" {
                    return Ok(Const::Bool(true));
                }
                if s == "false" {
                    return Ok(Const::Bool(false));
                }
                // named / associated constant path, e.g. `i32::MIN`
                let mut path = s;
                while self.peek().kind == TokenKind::ColonColon {
                    self.bump();
                    match self.bump().kind {
                        TokenKind::Ident(seg) => {
                            path.push_str("::");
                            path.push_str(&seg);
                        }
                        k => {
                            return Err(ParseError::new(
                                self.peek().line,
                                self.peek().col,
                                format!(
                                    "expected constant path segment, found `{}`",
                                    token_text(&k)
                                ),
                            ))
                        }
                    }
                }
                Ok(Const::Named(path))
            }
            k => Err(ParseError::new(
                t.line,
                t.col,
                format!("unsupported constant starting with `{}`", token_text(&k)),
            )),
        }
    }

    fn parse_rvalue(&mut self) -> ParseResult<Rvalue> {
        // reference: `&place`, `&mut place`, `&raw const place`, `&raw mut place`
        if self.peek().kind == TokenKind::Amp {
            self.bump();
            let mut kind = String::new();
            if self.at_ident("mut") {
                self.bump();
                kind = "mut".into();
            } else if self.at_ident("raw") {
                self.bump();
                kind = "raw".into();
                if self.at_ident("const") || self.at_ident("mut") {
                    kind.push(' ');
                    kind.push_str(&token_text(&self.bump().kind));
                }
            }
            let place = self.parse_place()?;
            return Ok(Rvalue::Ref { kind, place });
        }
        if self.peek().kind == TokenKind::Minus {
            // `-operand` as unary negation (rare; MIR normally uses Neg(..))
            let save = self.pos;
            self.bump();
            if matches!(self.peek().kind, TokenKind::Ident(_)) {
                let op = self.parse_operand()?;
                return Ok(Rvalue::UnaryOp(UnOp::Neg, Box::new(op)));
            }
            self.pos = save;
        }
        if self.peek().kind == TokenKind::LParen {
            // `(_3.0: i32)` / `(*_1)` used as an operand, or a tuple aggregate
            let save = self.pos;
            self.bump();
            if self.peek().kind == TokenKind::Star
                || matches!(&self.peek().kind, TokenKind::Ident(s) if local_index(s).is_some())
            {
                self.pos = save;
                let op = self.parse_operand()?;
                return Ok(Rvalue::Use(op));
            }
            // tuple / raw parenthesised aggregate: preserve verbatim
            self.pos = save;
            let raw = self.capture_balanced('(', ')')?;
            return Ok(Rvalue::Raw(raw));
        }
        if self.at_ident("copy") || self.at_ident("move") || self.at_ident("const") {
            return Ok(Rvalue::Use(self.parse_operand()?));
        }
        if matches!(&self.peek().kind, TokenKind::Ident(s) if local_index(s).is_some()) {
            return Ok(Rvalue::Use(self.parse_operand()?));
        }
        if matches!(self.peek().kind, TokenKind::Ident(_)) {
            // `Name(args...)` — binary/unary ops or aggregate-call fallback
            let name_tok = self.bump();
            let name = token_text(&name_tok.kind);
            if self.peek().kind == TokenKind::LParen {
                self.bump();
                let mut args = Vec::new();
                while self.peek().kind != TokenKind::RParen {
                    args.push(self.parse_operand()?);
                    if self.peek().kind == TokenKind::Comma {
                        self.bump();
                    } else {
                        break;
                    }
                }
                self.expect(&TokenKind::RParen)?;
                if let Some(op) = binop_from_name(&name) {
                    if args.len() != 2 {
                        return Err(ParseError::new(
                            name_tok.line,
                            name_tok.col,
                            format!("binary op `{name}` needs 2 operands, got {}", args.len()),
                        ));
                    }
                    let b = args.pop().unwrap();
                    let a = args.pop().unwrap();
                    return Ok(Rvalue::BinaryOp(op, Box::new(a), Box::new(b)));
                }
                if let Some(op) = unop_from_name(&name) {
                    if args.len() != 1 {
                        return Err(ParseError::new(
                            name_tok.line,
                            name_tok.col,
                            format!("unary op `{name}` needs 1 operand, got {}", args.len()),
                        ));
                    }
                    let a = args.pop().unwrap();
                    return Ok(Rvalue::UnaryOp(op, Box::new(a)));
                }
                return Ok(Rvalue::AggregateCall { name, args });
            }
            // path-like rvalue (enum variant without args, fn item, …): preserve
            let mut full = name;
            while self.peek().kind == TokenKind::ColonColon {
                self.bump();
                match self.bump().kind {
                    TokenKind::Ident(seg) => {
                        full.push_str("::");
                        full.push_str(&seg);
                    }
                    k => {
                        return Err(ParseError::new(
                            self.peek().line,
                            self.peek().col,
                            format!("expected path segment, found `{}`", token_text(&k)),
                        ))
                    }
                }
            }
            if self.peek().kind == TokenKind::LParen {
                self.bump();
                let mut args = Vec::new();
                while self.peek().kind != TokenKind::RParen {
                    args.push(self.parse_operand()?);
                    if self.peek().kind == TokenKind::Comma {
                        self.bump();
                    } else {
                        break;
                    }
                }
                self.expect(&TokenKind::RParen)?;
                return Ok(Rvalue::AggregateCall { name: full, args });
            }
            return Ok(Rvalue::Use(Operand::Constant(Const::Named(full))));
        }
        let t = self.peek().clone();
        Err(ParseError::new(
            t.line,
            t.col,
            format!("unsupported rvalue starting with `{}`", token_text(&t.kind)),
        ))
    }

    // ---------- types ----------

    fn parse_ty(&mut self) -> ParseResult<Ty> {
        match &self.peek().kind {
            TokenKind::LParen => {
                self.bump();
                if self.peek().kind == TokenKind::RParen {
                    self.bump();
                    return Ok(Ty::Unit);
                }
                let mut tys = Vec::new();
                loop {
                    tys.push(self.parse_ty()?);
                    if self.peek().kind == TokenKind::Comma {
                        self.bump();
                        if self.peek().kind == TokenKind::RParen {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                self.expect(&TokenKind::RParen)?;
                if tys.len() == 1 {
                    Ok(tys.pop().unwrap())
                } else {
                    Ok(Ty::Tuple(tys))
                }
            }
            TokenKind::LBracket => {
                self.bump();
                let elem = self.parse_ty()?;
                if self.peek().kind == TokenKind::Semi {
                    self.bump();
                    // array length: capture raw until `]`
                    let mut len = String::new();
                    while self.peek().kind != TokenKind::RBracket {
                        if self.at_eof() {
                            return self.err("unexpected end of file in array type");
                        }
                        if !len.is_empty() {
                            len.push(' ');
                        }
                        len.push_str(&token_text(&self.bump().kind));
                    }
                    self.expect(&TokenKind::RBracket)?;
                    Ok(Ty::Array {
                        elem: Box::new(elem),
                        len,
                    })
                } else {
                    self.expect(&TokenKind::RBracket)?;
                    Ok(Ty::Slice(Box::new(elem)))
                }
            }
            TokenKind::Amp => {
                self.bump();
                // optional lifetime
                if matches!(&self.peek().kind, TokenKind::Ident(s) if s.starts_with('\'')) {
                    self.bump();
                }
                let mutable = self.eat_ident("mut");
                let inner = self.parse_ty()?;
                Ok(Ty::Ref {
                    mutable,
                    inner: Box::new(inner),
                })
            }
            TokenKind::Star => {
                self.bump();
                let mutable = if self.eat_ident("mut") {
                    true
                } else {
                    self.expect_ident("const")?;
                    false
                };
                let inner = self.parse_ty()?;
                Ok(Ty::RawPtr {
                    mutable,
                    inner: Box::new(inner),
                })
            }
            TokenKind::Ident(_) => {
                let first = match self.bump().kind {
                    TokenKind::Ident(s) => s,
                    _ => unreachable!(),
                };
                match first.as_str() {
                    "bool" => return Ok(Ty::Bool),
                    "char" => return Ok(Ty::Char),
                    "str" => return Ok(Ty::Str),
                    "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => return Ok(Ty::Int(first)),
                    "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => return Ok(Ty::Uint(first)),
                    "f16" | "f32" | "f64" | "f128" => return Ok(Ty::Float(first)),
                    _ => {}
                }
                // Named path, possibly with :: segments, generics, fn-ptr tail
                let mut name = first;
                if name == "fn" {
                    // fn pointer type: fn(A, B) -> C
                    if self.peek().kind == TokenKind::LParen {
                        let raw = self.capture_balanced('(', ')')?;
                        name.push_str(&raw);
                    }
                    if self.peek().kind == TokenKind::Arrow {
                        self.bump();
                        let ret = self.parse_ty()?;
                        name.push_str(" -> ");
                        name.push_str(&ret.to_string());
                    }
                    return Ok(Ty::Named(name));
                }
                loop {
                    if self.peek().kind == TokenKind::ColonColon {
                        // distinguish `::<generics>` from `::segment`
                        if matches!(self.peek2().kind, TokenKind::Lt) {
                            self.bump();
                            let g = self.capture_balanced('<', '>')?;
                            name.push_str("::");
                            name.push_str(&g);
                            continue;
                        }
                        self.bump();
                        match self.bump().kind {
                            TokenKind::Ident(seg) => {
                                name.push_str("::");
                                name.push_str(&seg);
                            }
                            TokenKind::Lt => {
                                self.pos -= 1;
                                let g = self.capture_balanced('<', '>')?;
                                name.push_str("::");
                                name.push_str(&g);
                            }
                            k => {
                                return Err(ParseError::new(
                                    self.peek().line,
                                    self.peek().col,
                                    format!(
                                        "expected type path segment, found `{}`",
                                        token_text(&k)
                                    ),
                                ))
                            }
                        }
                    } else if self.peek().kind == TokenKind::Lt {
                        let g = self.capture_balanced('<', '>')?;
                        name.push_str(&g);
                    } else {
                        break;
                    }
                }
                Ok(Ty::Named(name))
            }
            TokenKind::Bang => {
                self.bump();
                Ok(Ty::Named("!".into()))
            }
            _ => {
                let t = self.peek().clone();
                Err(ParseError::new(
                    t.line,
                    t.col,
                    format!("unsupported type starting with `{}`", token_text(&t.kind)),
                ))
            }
        }
    }

    // ---------- balanced capture helpers ----------

    fn capture_balanced(&mut self, open: char, close: char) -> ParseResult<String> {
        let (open_kind, close_kind) = match (open, close) {
            ('(', ')') => (TokenKind::LParen, TokenKind::RParen),
            ('[', ']') => (TokenKind::LBracket, TokenKind::RBracket),
            ('{', '}') => (TokenKind::LBrace, TokenKind::RBrace),
            ('<', '>') => (TokenKind::Lt, TokenKind::Gt),
            _ => return self.err("internal: bad balanced pair"),
        };
        let mut out = String::new();
        let mut depth = 0i32;
        loop {
            let t = self.bump();
            if t.kind == TokenKind::Eof {
                return Err(ParseError::new(
                    t.line,
                    t.col,
                    format!("unterminated `{open}...{close}` group"),
                ));
            }
            let is_open = t.kind == open_kind;
            let is_close = t.kind == close_kind;
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&token_text(&t.kind));
            if is_open {
                depth += 1;
            }
            if is_close {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
        }
        Ok(out)
    }

    fn skip_balanced(&mut self, open: char, close: char) -> ParseResult<()> {
        let _ = self.capture_balanced(open, close)?;
        Ok(())
    }
}

enum AssignOrCall {
    Stmt(Statement),
    Term(Terminator),
}

fn binop_from_name(name: &str) -> Option<BinOp> {
    Some(match name {
        "Add" => BinOp::Add,
        "Sub" => BinOp::Sub,
        "Mul" => BinOp::Mul,
        "Div" => BinOp::Div,
        "Rem" => BinOp::Rem,
        "BitXor" => BinOp::BitXor,
        "BitAnd" => BinOp::BitAnd,
        "BitOr" => BinOp::BitOr,
        "Shl" => BinOp::Shl,
        "Shr" => BinOp::Shr,
        "Eq" => BinOp::Eq,
        "Ne" => BinOp::Ne,
        "Lt" => BinOp::Lt,
        "Le" => BinOp::Le,
        "Gt" => BinOp::Gt,
        "Ge" => BinOp::Ge,
        "Offset" => BinOp::Offset,
        "AddWithOverflow" => BinOp::AddWithOverflow,
        "SubWithOverflow" => BinOp::SubWithOverflow,
        "MulWithOverflow" => BinOp::MulWithOverflow,
        "AddUnchecked" => BinOp::AddUnchecked,
        "SubUnchecked" => BinOp::SubUnchecked,
        "MulUnchecked" => BinOp::MulUnchecked,
        "ShlUnchecked" => BinOp::ShlUnchecked,
        "ShrUnchecked" => BinOp::ShrUnchecked,
        _ => return None,
    })
}

fn unop_from_name(name: &str) -> Option<UnOp> {
    Some(match name {
        "Not" => UnOp::Not,
        "Neg" => UnOp::Neg,
        "PtrMetadata" => UnOp::PtrMetadata,
        _ => return None,
    })
}

/// Split `123_i32` into ("123", "i32"); `123` into ("123", "").
fn split_int_suffix(s: &str) -> (String, String) {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'_') {
        i += 1;
    }
    // suffix starts with `_` followed by a type name, or directly with a letter
    if i < bytes.len()
        && bytes[i] == b'_'
        && i + 1 < bytes.len()
        && bytes[i + 1].is_ascii_alphabetic()
    {
        (s[..i].to_string(), s[i + 1..].to_string())
    } else if i < bytes.len() && bytes[i].is_ascii_alphabetic() {
        (s[..i].to_string(), s[i..].to_string())
    } else {
        (s.to_string(), String::new())
    }
}

fn split_float_suffix(s: &str) -> (String, String) {
    for suffix in ["f16", "f32", "f64", "f128"] {
        if let Some(stripped) = s.strip_suffix(suffix) {
            let stripped = stripped.strip_suffix('_').unwrap_or(stripped);
            return (stripped.to_string(), suffix.to_string());
        }
    }
    (s.to_string(), String::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_const_return() {
        let src = "fn answer() -> i32 {\n    let mut _0: i32;\n    bb0: {\n        _0 = const 1_i32;\n        return;\n    }\n}\n";
        let file = parse_mir(src).unwrap();
        assert_eq!(file.bodies.len(), 1);
        let b = &file.bodies[0];
        assert_eq!(b.name, "answer");
        assert_eq!(b.locals.len(), 1);
        assert_eq!(b.blocks.len(), 1);
        match &b.blocks[0].statements[0] {
            Statement::Assign(p, Rvalue::Use(Operand::Constant(Const::Int { value: 1, ty }))) => {
                assert_eq!(p.local, Local(0));
                assert_eq!(ty, "i32");
            }
            other => panic!("unexpected statement: {other:?}"),
        }
        assert!(matches!(b.blocks[0].terminator, Terminator::Return));
    }

    #[test]
    fn error_has_line_number() {
        let src = "fn f() -> i32 {\n    let mut _0: i32;\n    bb0: {\n        _0 = @bad;\n        return;\n    }\n}\n";
        let err = parse_mir(src).unwrap_err();
        assert_eq!(err.line, 4);
    }

    #[test]
    fn unsupported_rvalue_is_error_not_silence() {
        // A statement starting with an unexpected token must error.
        let src = "fn f() -> i32 {\n    let mut _0: i32;\n    bb0: {\n        42;\n        return;\n    }\n}\n";
        assert!(parse_mir(src).is_err());
    }
}
