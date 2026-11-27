//! Parsing of debug-mode scripts.


/// A parsed debugger session script (see [`execute_debug`]).
pub(crate) struct ParsedDebugScript {
    /// Breakpoint lines in the input file.
    pub(crate) break_lines: Vec<u32>,
    /// Breakpoints in a named file: (file, line) pairs.
    pub(crate) break_files: Vec<(String, u32)>,
    /// Function breakpoints: body-name patterns.
    pub(crate) break_fns: Vec<String>,
    /// The session's commands, in script order.
    pub(crate) commands: Vec<sparagmos_interp::DebugCommand>,
    /// Whether gdb prints render through the pretty printers.
    pub(crate) pretty_printers: bool,
}

/// Parse a debugger session script (see [`execute_debug`]) into
/// its breakpoint lines and its command list.
pub(crate) fn parse_debug_script(text: &str) -> Result<ParsedDebugScript, String> {
    use sparagmos_interp::{DebugCommand, DebugPrint, PrintFormat};
    let mut breakpoints = Vec::new();
    let mut break_files = Vec::new();
    let mut break_fns = Vec::new();
    let mut commands = Vec::new();
    let mut pretty_printers = false;
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let bad = |what: &str| format!("line {}: {what}", lineno + 1);
        let (head, rest) = match line.split_once(char::is_whitespace) {
            Some((h, r)) => (h, r.trim()),
            None => (line, ""),
        };
        match head {
            "pretty-printers" => {
                if !rest.is_empty() {
                    return Err(bad("`pretty-printers` takes no argument"));
                }
                pretty_printers = true;
            }
            "print-elements" => {
                // gdb `set print elements N` (the runner
                // forwards it under this name): the session's
                // string and child truncation limit from this
                // point on. `unlimited` is gdb's 0.
                let n: usize = if rest == "unlimited" {
                    0
                } else {
                    rest.parse()
                        .map_err(|_| bad(&format!("bad print elements `{rest}`")))?
                };
                commands.push(DebugCommand::SetPrintElements(n));
            }
            "type-format" => {
                let (fmt, ty) = rest
                    .split_once(char::is_whitespace)
                    .ok_or_else(|| bad(&format!("bad type-format `{rest}`")))?;
                let format = match fmt {
                    "dec" => sparagmos_interp::LldbNumFormat::Decimal,
                    "unsigned" => sparagmos_interp::LldbNumFormat::Unsigned,
                    "hex" => sparagmos_interp::LldbNumFormat::Hex,
                    other => {
                        return Err(bad(&format!("unknown type format `{other}`")));
                    }
                };
                commands.push(DebugCommand::SetLldbFormat {
                    ty: ty.trim().to_string(),
                    format,
                });
            }
            "break" => {
                let n: u32 = rest
                    .parse()
                    .map_err(|_| bad(&format!("bad breakpoint line `{rest}`")))?;
                breakpoints.push(n);
            }
            "break-at" => {
                let (file, line) = rest
                    .rsplit_once(':')
                    .ok_or_else(|| bad(&format!("bad file breakpoint `{rest}`")))?;
                let n: u32 = line
                    .parse()
                    .map_err(|_| bad(&format!("bad breakpoint line `{rest}`")))?;
                break_files.push((file.to_string(), n));
            }
            "break-fn" => {
                if rest.is_empty() {
                    return Err(bad("`break-fn` takes a function path"));
                }
                break_fns.push(rest.to_string());
            }
            "run" => commands.push(DebugCommand::Run),
            "continue" => commands.push(DebugCommand::Continue),
            "next" => commands.push(DebugCommand::Next),
            "step" => commands.push(DebugCommand::Step),
            "finish" => commands.push(DebugCommand::Finish),
            "show-line" => commands.push(DebugCommand::ShowLine),
            "frame" => {
                let n: usize = rest
                    .parse()
                    .map_err(|_| bad(&format!("bad frame index `{rest}`")))?;
                commands.push(DebugCommand::FrameSelect(n));
            }
            "v" => commands.push(DebugCommand::Print(DebugPrint {
                expr: rest.to_string(),
                format: None,
            })),
            "print" => commands.push(DebugCommand::Print(DebugPrint {
                expr: rest.to_string(),
                format: None,
            })),
            "printf" => {
                // gdb `printf "<fmt>", <args>`: the format is
                // a C string literal (its escapes are decoded
                // here, where the script grammar lives); the
                // arguments are print expressions the session
                // evaluates at the stop.
                let (format, args) = parse_printf_args(rest)
                    .ok_or_else(|| bad(&format!("bad printf `{rest}`")))?;
                commands.push(DebugCommand::Printf { format, args });
            }
            "whatis" => commands.push(DebugCommand::WhatIs(DebugPrint {
                expr: rest.to_string(),
                format: None,
            })),
            "ptype" => commands.push(DebugCommand::Ptype(DebugPrint {
                expr: rest.to_string(),
                format: None,
            })),
            "repr" => commands.push(DebugCommand::Repr(rest.to_string())),
            "up" => commands.push(DebugCommand::FrameUp),
            "down" => commands.push(DebugCommand::FrameDown),
            "info-locals" => commands.push(DebugCommand::InfoLocals),
            "info-functions" | "info-functions-q" => {
                commands.push(DebugCommand::InfoFunctions {
                    pattern: if rest.is_empty() || rest == "-" {
                        None
                    } else {
                        Some(rest.to_string())
                    },
                    quiet: head == "info-functions-q",
                });
            }
            _ if head.starts_with("print/") => {
                let format = match &head["print/".len()..] {
                    "d" => Some(PrintFormat::Decimal),
                    "u" => Some(PrintFormat::Unsigned),
                    "x" => Some(PrintFormat::Hex),
                    "t" => Some(PrintFormat::Binary),
                    "c" => Some(PrintFormat::Char),
                    other => return Err(bad(&format!("unknown print format `{other}`"))),
                };
                commands.push(DebugCommand::Print(DebugPrint {
                    expr: rest.to_string(),
                    format,
                }));
            }
            _ => return Err(bad(&format!("unknown command `{head}`"))),
        }
    }
    Ok(ParsedDebugScript {
        break_lines: breakpoints,
        break_files,
        break_fns,
        commands,
        pretty_printers,
    })
}

/// Parse the argument text of a session `printf` line: a C
/// string literal (the format, with the usual backslash
/// escapes decoded) followed by an optional comma-separated
/// list of print expressions, split at top-level commas --
/// an argument's own parentheses and brackets nest.
fn parse_printf_args(rest: &str) -> Option<(String, Vec<String>)> {
    let bytes = rest.as_bytes();
    if bytes.first() != Some(&b'"') {
        return None;
    }
    let mut format = String::new();
    let mut i = 1;
    loop {
        match *bytes.get(i)? {
            b'"' => {
                i += 1;
                break;
            }
            b'\\' => {
                i += 1;
                let escaped = *bytes.get(i)? as char;
                i += 1;
                match escaped {
                    'n' => format.push('\n'),
                    't' => format.push('\t'),
                    'r' => format.push('\r'),
                    '0' => format.push('\0'),
                    other => format.push(other),
                }
            }
            byte => {
                // Copy the byte's whole UTF-8 sequence: the
                // format text outside escapes is ordinary
                // text (a directive's `%` included).
                let ch_len = utf8_len(byte);
                format.push_str(std::str::from_utf8(&bytes[i..i + ch_len]).ok()?);
                i += ch_len;
            }
        }
    }
    let rest = rest[i..].trim();
    if rest.is_empty() {
        return Some((format, Vec::new()));
    }
    let args_text = rest.strip_prefix(',')?;
    let mut args = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (j, c) in args_text.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                args.push(args_text[start..j].trim().to_string());
                start = j + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return None;
    }
    args.push(args_text[start..].trim().to_string());
    if args.iter().any(|a| a.is_empty()) {
        return None;
    }
    Some((format, args))
}

/// The length of the UTF-8 sequence a leading byte starts.
fn utf8_len(byte: u8) -> usize {
    match byte {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}
