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
