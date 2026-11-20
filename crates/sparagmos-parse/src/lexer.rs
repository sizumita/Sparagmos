//! Hand-written lexer for textual MIR (`rustc --emit=mir` output).
//! No parser-generator / lexer-generator crates are used.

use crate::error::{ParseError, ParseResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    Ident(String),
    /// Integer literal text as written, e.g. `1`, `42`, including any suffix
    /// split off by the parser (suffix stays in the string).
    IntLit(String),
    FloatLit(String),
    StrLit(String),
    CharLit(char),
    // punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semi,
    Colon,
    ColonColon,
    Dot,
    Eq,
    Arrow,    // ->
    FatArrow, // =>
    Bang,     // !
    Amp,      // &
    Star,     // *
    Minus,
    Plus,
    Lt,
    Gt,
    Underscore,
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub line: usize,
    pub col: usize,
}

pub fn tokenize(src: &str) -> ParseResult<Vec<Token>> {
    let bytes = src.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0usize;
    let mut line = 1usize;
    let mut line_start = 0usize;
    while i < bytes.len() {
        let c = bytes[i] as char;
        // whitespace
        if c == '\n' {
            line += 1;
            i += 1;
            line_start = i;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        // line comment
        if c == '/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start_col = i - line_start + 1;
        let kind = match c {
            '(' => {
                i += 1;
                TokenKind::LParen
            }
            ')' => {
                i += 1;
                TokenKind::RParen
            }
            '{' => {
                i += 1;
                TokenKind::LBrace
            }
            '}' => {
                i += 1;
                TokenKind::RBrace
            }
            '[' => {
                i += 1;
                TokenKind::LBracket
            }
            ']' => {
                i += 1;
                TokenKind::RBracket
            }
            ',' => {
                i += 1;
                TokenKind::Comma
            }
            ';' => {
                i += 1;
                TokenKind::Semi
            }
            '.' => {
                i += 1;
                TokenKind::Dot
            }
            '=' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                    i += 2;
                    TokenKind::FatArrow
                } else {
                    i += 1;
                    TokenKind::Eq
                }
            }
            '-' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                    i += 2;
                    TokenKind::Arrow
                } else {
                    i += 1;
                    TokenKind::Minus
                }
            }
            '+' => {
                i += 1;
                TokenKind::Plus
            }
            '!' => {
                i += 1;
                TokenKind::Bang
            }
            '&' => {
                i += 1;
                TokenKind::Amp
            }
            '*' => {
                i += 1;
                TokenKind::Star
            }
            '<' => {
                i += 1;
                TokenKind::Lt
            }
            '>' => {
                i += 1;
                TokenKind::Gt
            }
            ':' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b':' {
                    i += 2;
                    TokenKind::ColonColon
                } else {
                    i += 1;
                    TokenKind::Colon
                }
            }
            '"' => {
                // string literal (handles backslash escapes minimally)
                let mut j = i + 1;
                let mut s = String::new();
                let mut closed = false;
                while j < bytes.len() {
                    match bytes[j] {
                        b'\\' if j + 1 < bytes.len() => {
                            let e = bytes[j + 1] as char;
                            s.push(match e {
                                'n' => '\n',
                                't' => '\t',
                                'r' => '\r',
                                '\\' => '\\',
                                '"' => '"',
                                '0' => '\0',
                                other => other,
                            });
                            j += 2;
                        }
                        b'"' => {
                            closed = true;
                            j += 1;
                            break;
                        }
                        b'\n' => {
                            line += 1;
                            s.push('\n');
                            j += 1;
                            line_start = j;
                        }
                        _ => {
                            // consume one UTF-8 char
                            let ch_len = utf8_len(bytes[j]);
                            s.push_str(&src[j..(j + ch_len).min(src.len())]);
                            j += ch_len;
                        }
                    }
                }
                if !closed {
                    return Err(ParseError::new(
                        line,
                        start_col,
                        "unterminated string literal",
                    ));
                }
                i = j;
                TokenKind::StrLit(s)
            }
            '\'' => {
                // char literal like 'a' or '\n'; lifetime like 'a is lexed as
                // CharLit only when it has a closing quote in char position,
                // otherwise fall back to an Ident prefixed with '.
                if i + 2 < bytes.len() && bytes[i + 2] == b'\'' {
                    let ch = bytes[i + 1] as char;
                    i += 3;
                    TokenKind::CharLit(ch)
                } else if i + 3 < bytes.len() && bytes[i + 1] == b'\\' && bytes[i + 3] == b'\'' {
                    let ch = match bytes[i + 2] as char {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '\\' => '\\',
                        '\'' => '\'',
                        '0' => '\0',
                        o => o,
                    };
                    i += 4;
                    TokenKind::CharLit(ch)
                } else {
                    // lifetime: 'static, 'a ...
                    let mut j = i + 1;
                    while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_')
                    {
                        j += 1;
                    }
                    let name = format!("'{}", &src[i + 1..j]);
                    i = j;
                    TokenKind::Ident(name)
                }
            }
            _ if c.is_ascii_digit() => {
                let mut j = i;
                let mut is_float = false;
                while j < bytes.len() {
                    let b = bytes[j];
                    if b.is_ascii_alphanumeric() || b == b'_' {
                        j += 1;
                    } else if b == b'.' && j + 1 < bytes.len() && bytes[j + 1].is_ascii_digit() {
                        is_float = true;
                        j += 1;
                    } else {
                        break;
                    }
                }
                let text = src[i..j].to_string();
                i = j;
                if is_float {
                    TokenKind::FloatLit(text)
                } else {
                    TokenKind::IntLit(text)
                }
            }
            _ if c.is_alphabetic() || c == '_' => {
                let mut j = i;
                while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                    j += 1;
                }
                // `_` alone (wildcard) vs `_0` local vs identifier
                let text = &src[i..j];
                i = j;
                if text == "_" {
                    TokenKind::Underscore
                } else {
                    TokenKind::Ident(text.to_string())
                }
            }
            _ => {
                return Err(ParseError::new(
                    line,
                    start_col,
                    format!("unexpected character `{c}`"),
                ));
            }
        };
        tokens.push(Token {
            kind,
            line,
            col: start_col,
        });
    }
    tokens.push(Token {
        kind: TokenKind::Eof,
        line,
        col: i - line_start + 1,
    });
    Ok(tokens)
}

fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lexes_locals_and_blocks() {
        let toks = tokenize("_0 = const 1_i32; bb0: { return; }").unwrap();
        assert!(matches!(&toks[0].kind, TokenKind::Ident(s) if s == "_0"));
        assert!(toks
            .iter()
            .any(|t| matches!(&t.kind, TokenKind::Ident(s) if s == "bb0")));
    }
    #[test]
    fn tracks_line_numbers() {
        let toks = tokenize("fn f() {\n_0 = const 1_i32;\n}").unwrap();
        let eq = toks.iter().find(|t| t.kind == TokenKind::Eq).unwrap();
        assert_eq!(eq.line, 2);
    }
    #[test]
    fn skips_comments() {
        let toks = tokenize("// hello\nreturn;").unwrap();
        assert!(matches!(&toks[0].kind, TokenKind::Ident(s) if s == "return"));
    }
}
