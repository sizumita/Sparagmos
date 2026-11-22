//! The interpreter's debugger interface: the machinery the
//! debuginfo test harness drives instead of a real debugger.
//!
//! A real debugger reads DWARF — type and line information out of
//! a compiled binary's debug sections — which this runtime never
//! produces (it executes MIR directly; there is no binary). What a
//! debugger session actually needs from the execution engine is
//! small, though, and the interpreter can provide it directly:
//! line breakpoints, run/continue, step/next/finish, and reading
//! the current frame's locals rendered the way the debugger would
//! print them. That is the interface here.
//!
//! A [`DebugSession`] wraps one converted program. The caller sets
//! breakpoints (by source file and line, from the test file's
//! `#break` markers, as compiletest does) and hands over the
//! session's source text (for rendering stop lines), then runs a
//! script of [`DebugCommand`]s. Execution is a single ordinary
//! interpreter run with a hook installed: the drive loop consults
//! the hook at every source-line change (see
//! [`crate::interp::Interpreter::run`]'s loop and `drive_hook`
//! below), and the hook consumes script commands at each stop —
//! printing values into the transcript — until a resuming command
//! sets the next stop plan. With no session installed the hook is
//! absent and ordinary runs are untouched.
//!
//! The transcript is the debugger's output, in the shape the
//! harness's check lines expect: a stop emits the source line it
//! stopped on; a `print` emits `$N = <value>` (gdb form) or
//! `= <value>` (lldb form, whose check lines are loose fragments).
//! Only scalar rendering is supported — integers, booleans,
//! floats, chars, unit — reached through locals, dereferences and
//! field/index projections. Aggregate rendering is the follow-up
//! scope (the D2 category of the reachability census).

use crate::interp::{Frame, Interpreter};
use crate::ir::{Body, Program, SourceLine, Ty};
use crate::value::{RefProj, Value};
use std::collections::{HashMap, HashSet, VecDeque};

/// One command of a debugger session script, in the vocabulary the
/// harness translates the test file's `//@ gdb-command:` /
/// `//@ lldb-command:` sections into.
#[derive(Debug, Clone, PartialEq)]
pub enum DebugCommand {
    /// Start the program (gdb/lldb `run`): the session's start;
    /// consumed before execution begins.
    Run,
    /// Resume until the next breakpoint.
    Continue,
    /// Step over: stop at the next line change at or above the
    /// current frame depth.
    Next,
    /// Step into: stop at the next line change anywhere.
    Step,
    /// Step out: stop at the next line change above the current
    /// frame depth (gdb `finish`).
    Finish,
    /// Print an expression's value (gdb `print`, lldb `v`).
    Print(DebugPrint),
    /// Select the frame expressions are evaluated in (0 =
    /// innermost), gdb `frame N` / lldb `frame select N`.
    FrameSelect(usize),
    /// Show the current frame's source line (gdb `frame`, lldb
    /// `frame select` without a number): emits the line the
    /// session is stopped on into the transcript again.
    ShowLine,
}

/// A print command: the expression text plus the number format
/// the command prescribed (`print/d`, `print/x`), if any.
#[derive(Debug, Clone, PartialEq)]
pub struct DebugPrint {
    pub expr: String,
    pub format: Option<PrintFormat>,
}

/// The value format of a `print/<fmt>` command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrintFormat {
    /// `print/d`: signed decimal (a char prints as its code point).
    Decimal,
    /// `print/x`: hexadecimal.
    Hex,
    /// `print/t`: binary.
    Binary,
    /// `print/c`: a character (an integer prints as a character).
    Char,
}

/// Which debugger's output form a session transcript follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugDialect {
    Gdb,
    Lldb,
}

/// The result of running one session script.
#[derive(Debug)]
pub struct DebugRun {
    /// The transcript lines, in emission order.
    pub transcript: Vec<String>,
    /// How the guest program ended: `Ok` carries a description
    /// ("program exited normally" / the value main returned is not
    /// modelled), `Err` the run failure's message (a guest panic
    /// or an interpreter error).
    pub outcome: Result<(), String>,
    /// The guest's captured stdout.
    pub stdout: String,
    /// The guest's captured stderr.
    pub stderr: String,
}

/// When the drive hook stops next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StopPlan {
    /// Only at a breakpoint line.
    Breakpoints,
    /// At the next line change, any depth.
    StepInto,
    /// At the next line change with frame depth <= the recorded one.
    StepOver(usize),
    /// At the next line change with frame depth < the recorded one.
    StepOut(usize),
    /// Never again (the script is exhausted; run to completion).
    Done,
}

/// The debugger state the drive loop consults. Installed on the
/// interpreter for the length of one session run.
pub(crate) struct DebugControl {
    /// Breakpoint positions as (file id, line).
    breakpoints: HashSet<(u32, u32)>,
    /// The commands not yet consumed.
    script: VecDeque<DebugCommand>,
    /// The current stop plan.
    plan: StopPlan,
    /// The last position executed anywhere; step plans fire on
    /// line *changes*, so consecutive statements on one line are
    /// a single position.
    last_pos: Option<(u32, u32)>,
    /// The last position executed in each live frame, by frame
    /// id. Breakpoint stops are judged per frame: a line
    /// breakpoint sits at the line's first instruction, so a
    /// frame that is already on the line — it called out from
    /// the line and came back to the line's continuation — does
    /// not stop again until it has been to another line (a loop
    /// iteration qualifies; a call's return does not).
    frame_lines: HashMap<u64, (u32, u32)>,
    /// The frame expressions evaluate in, counted from the
    /// innermost frame (0).
    selected_frame: usize,
    /// The position of the current stop, for resolving a variable
    /// name against the scopes in scope there.
    current_pos: Option<(u32, u32)>,
    /// The `$N` counter of gdb-form print output.
    print_count: u32,
    /// The transcript lines emitted so far.
    transcript: Vec<String>,
    /// Source text by file id, for rendering stop lines.
    source_text: HashMap<u32, Vec<String>>,
    /// The output dialect.
    dialect: DebugDialect,
}

/// A debugger session over one converted program.
pub struct DebugSession {
    program: Program,
    entry: String,
    args: Vec<Value>,
    config: crate::InterpConfig,
    breakpoints: HashSet<(u32, u32)>,
    source_text: HashMap<u32, Vec<String>>,
}

impl DebugSession {
    pub fn new(
        program: Program,
        entry: String,
        args: Vec<Value>,
        config: crate::InterpConfig,
    ) -> DebugSession {
        DebugSession {
            program,
            entry,
            args,
            config,
            breakpoints: HashSet::new(),
            source_text: HashMap::new(),
        }
    }

    /// The program's source-file table (the names the bodies'
    /// source lines index into).
    pub fn source_files(&self) -> &[String] {
        &self.program.source_files
    }

    /// Set a line breakpoint in the named source file. The name is
    /// matched against the program's source-file table exactly, or
    /// as a path suffix (the table holds the paths the compiler
    /// reported, the caller names the file it compiled).
    pub fn set_breakpoint(&mut self, file: &str, line: u32) {
        if let Some(id) = self.file_id(file) {
            self.breakpoints.insert((id, line));
        }
    }

    /// Hand the session the source text of a file, for rendering
    /// the lines it stops on.
    pub fn set_source_text(&mut self, file: &str, text: &str) {
        if let Some(id) = self.file_id(file) {
            self.source_text
                .insert(id, text.lines().map(str::to_string).collect());
        }
    }

    fn file_id(&self, file: &str) -> Option<u32> {
        if let Some(id) = self
            .program
            .source_files
            .iter()
            .position(|f| f == file)
        {
            return Some(id as u32);
        }
        self.program
            .source_files
            .iter()
            .position(|f| f.ends_with(file) || file.ends_with(f.as_str()))
            .map(|id| id as u32)
    }

    /// Run the session: execute the program, consuming `commands`
    /// at the stops they prescribe, in `dialect`'s output form.
    /// A leading [`DebugCommand::Run`] is the session start itself
    /// and is consumed before execution begins.
    pub fn run_script(
        &mut self,
        commands: Vec<DebugCommand>,
        dialect: DebugDialect,
    ) -> DebugRun {
        let mut script: VecDeque<DebugCommand> = commands.into();
        if matches!(script.front(), Some(DebugCommand::Run)) {
            script.pop_front();
        }
        let ctl = DebugControl {
            breakpoints: std::mem::take(&mut self.breakpoints),
            script,
            plan: StopPlan::Breakpoints,
            last_pos: None,
            current_pos: None,
            frame_lines: HashMap::new(),
            selected_frame: 0,
            print_count: 0,
            transcript: Vec::new(),
            source_text: std::mem::take(&mut self.source_text),
            dialect,
        };
        let mut interp = Interpreter::with_config(&self.program, self.config.clone());
        interp.debug_ctl = Some(ctl);
        let result = interp.run(&self.entry, std::mem::take(&mut self.args));
        let ctl = interp.debug_ctl.take().expect("session control installed");
        DebugRun {
            transcript: ctl.transcript,
            outcome: result.map(|_| ()).map_err(|e| e.to_string()),
            stdout: interp.output().to_string(),
            stderr: interp.error_output().to_string(),
        }
    }
}

/// The drive loop's debugger hook: consulted before the statement
/// or terminator at (`block`, `stmt`) of the current frame
/// executes, whenever a session is installed. Decides whether the
/// position is a stop under the current plan; at a stop, consumes
/// script commands (prints render into the transcript) until one
/// of them resumes execution with a new plan.
pub(crate) fn drive_hook(
    interp: &mut Interpreter,
    frames: &[Frame],
    body: &Body,
    block: usize,
    stmt: usize,
) {
    let Some(pos) = position(body, block, stmt) else {
        return;
    };
    let key = (pos.file, pos.line);
    let mut ctl = interp
        .debug_ctl
        .take()
        .expect("drive hook fires only with a session installed");
    let depth = frames.len();
    // A stop fires on a line *change*: the position differs from
    // the previously executed one. The tracker therefore updates
    // on every consultation, not only at stops — otherwise a
    // breakpoint line reached again later in the run (the same
    // line in a later call of the same function) would read as
    // "unchanged" forever after its first stop.
    let changed = ctl.last_pos != Some(key);
    ctl.last_pos = Some(key);
    let frame_id = frames.last().map(|f| f.id).unwrap_or(0);
    let frame_changed = ctl.frame_lines.get(&frame_id) != Some(&key);
    ctl.frame_lines.insert(frame_id, key);
    let stop = match ctl.plan {
        StopPlan::Breakpoints => {
            frame_changed && ctl.breakpoints.contains(&key)
        }
        StopPlan::StepInto => changed,
        StopPlan::StepOver(d) => changed && depth <= d,
        StopPlan::StepOut(d) => changed && depth < d,
        StopPlan::Done => false,
    };
    if stop {
        ctl.selected_frame = 0;
        ctl.current_pos = Some(key);
        if let Some(lines) = ctl.source_text.get(&pos.file) {
            if let Some(text) = lines.get((pos.line - 1) as usize) {
                ctl.transcript.push(text.trim().to_string());
            }
        }
        // Consume commands until one resumes execution.
        loop {
            match ctl.script.pop_front() {
                Some(DebugCommand::Print(print)) => {
                    let line = render_print(interp, &ctl, frames, &print);
                    ctl.transcript.push(line);
                    if ctl.dialect == DebugDialect::Gdb {
                        ctl.print_count += 1;
                    }
                }
                Some(DebugCommand::FrameSelect(n)) => {
                    ctl.selected_frame = n;
                }
                Some(DebugCommand::ShowLine) => {
                    if let Some(lines) = ctl.source_text.get(&pos.file) {
                        if let Some(text) = lines.get((pos.line - 1) as usize) {
                            ctl.transcript.push(text.trim().to_string());
                        }
                    }
                }
                Some(DebugCommand::Next) => {
                    ctl.plan = StopPlan::StepOver(depth);
                    break;
                }
                Some(DebugCommand::Step) => {
                    ctl.plan = StopPlan::StepInto;
                    break;
                }
                Some(DebugCommand::Finish) => {
                    ctl.plan = StopPlan::StepOut(depth);
                    break;
                }
                Some(DebugCommand::Continue) | Some(DebugCommand::Run) => {
                    ctl.plan = StopPlan::Breakpoints;
                    break;
                }
                None => {
                    ctl.plan = StopPlan::Done;
                    break;
                }
            }
        }
    }
    interp.debug_ctl = Some(ctl);
}

/// The source position of the statement (or terminator, when
/// `stmt` is past the block's statements) about to execute.
fn position(body: &Body, block: usize, stmt: usize) -> Option<SourceLine> {
    let lines = body.source_lines.get(block)?;
    let pos = if stmt < lines.statements.len() {
        lines.statements[stmt]
    } else {
        lines.terminator
    };
    if pos.is_known() {
        Some(pos)
    } else {
        None
    }
}

/// Render one print command against the selected frame, in the
/// session dialect's form.
fn render_print(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    print: &DebugPrint,
) -> String {
    let rendered = evaluate_print(interp, ctl, frames, print)
        .unwrap_or_else(|e| format!("<error: {e}>"));
    match ctl.dialect {
        DebugDialect::Gdb => format!("${} = {}", ctl.print_count + 1, rendered),
        DebugDialect::Lldb => format!("= {rendered}"),
    }
}

/// Evaluate a print expression in the selected frame: a variable
/// of the frame's body (as recorded in its debug variable table),
/// followed by dereferences, field and index steps. Returns the
/// gdb-form rendering of the resulting scalar.
fn evaluate_print(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    print: &DebugPrint,
) -> Result<String, String> {
    let frame = frames
        .get(frames.len().saturating_sub(1 + ctl.selected_frame))
        .ok_or_else(|| format!("frame {} is not live", ctl.selected_frame))?;
    let body = interp
        .program()
        .get(&frame.func)
        .ok_or_else(|| format!("no body for frame `{}`", frame.func))?;
    let expr = parse_expr(&print.expr)?;
    // Resolve the name the way a debugger does: among the
    // body's entries of that name, prefer the declaration whose
    // recorded span contains the current line (the smallest such
    // extent, as the innermost scope); the compiler records most
    // entries with a point span at the declaration, so the
    // general rule is the latest declaration at or before the
    // current line — the declaration in effect at the stop. With
    // neither, the last declaration stands.
    let entry = {
        let named = body
            .debug_vars
            .iter()
            .filter(|e| e.name == expr.base)
            .collect::<Vec<_>>();
        let at_line = ctl.current_pos.map(|(_, line)| line).unwrap_or(0);
        let containing = named
            .iter()
            .filter(|e| e.lines.0 != 0 && e.lines.0 <= at_line && at_line <= e.lines.1)
            .min_by_key(|e| e.lines.1 - e.lines.0);
        let declared = named
            .iter()
            .filter(|e| e.lines.0 != 0 && e.lines.0 <= at_line)
            .max_by_key(|e| e.lines.0);
        containing
            .or_else(|| declared)
            .copied()
            .or_else(|| named.last().copied())
    };
    let entry = entry.ok_or_else(|| format!("no variable `{}` in scope", expr.base))?;
    let (source, char_hint) = (&entry.source, entry.is_char);
    let (mut value, mut ty): (Value, Option<Ty>) = match source {
        crate::ir::DebugVar::Place(local, proj) => {
            let value = frame
                .locals
                .get(*local)
                .cloned()
                .ok_or_else(|| format!("variable `{}` has no storage", expr.base))?;
            let ty = body.local_tys.get(*local).cloned();
            if proj.is_empty() {
                (value, ty)
            } else {
                (
                    crate::interp::project_value(value, proj, "debugger", &expr.base)
                        .map_err(|e| e.to_string())?,
                    None,
                )
            }
        }
        crate::ir::DebugVar::Constant(constant) => (
            interp.eval_constant(constant).map_err(|e| e.to_string())?,
            None,
        ),
    };
    // Whether the value (or, for a reference variable, its
    // pointee) is a `char` at the source level: the IR type
    // system has no char (it is a u32), so the conversion
    // recorded it in the debug variable table.
    let mut char = char_hint;
    for step in &expr.steps {
        match step {
            ExprStep::Deref => {
                let resolved = interp
                    .deref_value(frames, &value)
                    .map_err(|e| e.to_string())?;
                value = interp
                    .read_resolved(frames, &resolved)
                    .map_err(|e| e.to_string())?;
                ty = match ty {
                    Some(Ty::Ref(inner)) => Some(*inner),
                    _ => None,
                };
                // The char hint is recorded for the variable as
                // declared: a `char` local, or a reference to a
                // char (whose pointee the hint then describes).
                // A dereference step therefore keeps it.
            }
            ExprStep::Field(index) => {
                value = crate::interp::project_value(
                    value,
                    &[RefProj::Field(*index)],
                    "debugger",
                    &print.expr,
                )
                .map_err(|e| e.to_string())?;
                ty = match (ty, index) {
                    (Some(Ty::Tuple(tys)), i) => tys.get(*i).cloned(),
                    (Some(Ty::Array(elem, _)), _) => Some(*elem),
                    _ => None,
                };
                char = false;
            }
            ExprStep::Index(index) => {
                value = crate::interp::project_value(
                    value,
                    &[RefProj::Index(*index)],
                    "debugger",
                    &print.expr,
                )
                .map_err(|e| e.to_string())?;
                ty = match ty {
                    Some(Ty::Array(elem, _)) => Some(*elem),
                    Some(Ty::Slice(elem)) => Some(*elem),
                    _ => None,
                };
                char = false;
            }
        }
    }
    render_scalar(&value, ty.as_ref(), char, print.format, ctl.dialect)
        .ok_or_else(|| format!("`{}` is not a scalar value", print.expr))
}

/// Render a scalar value the way gdb prints it: integers in
/// decimal (signedness from the declared type when the value's
/// own form does not carry it), booleans as `true`/`false`,
/// floats with Rust's `{}` (which agrees with gdb's rendering of
/// the simple decimals the checks use), a char as
/// `<codepoint> '<char>'`, unit as `()`.
fn render_scalar(
    value: &Value,
    ty: Option<&Ty>,
    char_hint: bool,
    format: Option<PrintFormat>,
    dialect: DebugDialect,
) -> Option<String> {
    match value {
        Value::Bool(b) => Some(b.to_string()),
        Value::Unit => Some("()".to_string()),
        Value::Int(v) => Some(format_int(*v as u128, true, format)),
        Value::Uint(v) => {
            if char_hint && format.is_none() {
                let c = char::from_u32(*v as u32)?;
                return Some(match dialect {
                    DebugDialect::Gdb => format!("{} '{}'", v, escape_char(c)),
                    // lldb's char summary: the code point in
                    // `U+XXXX` form, then the character itself.
                    DebugDialect::Lldb => {
                        format!("U+{:04X} U'{}'", v, escape_char(c))
                    }
                });
            }
            let signed = matches!(format, Some(PrintFormat::Decimal));
            Some(format_int(*v, signed, format))
        }
        Value::Float(v) => Some(format_float(*v)),
        Value::Float128(bits) => {
            // Rendered through f64 when the value is exactly
            // representable there (the simple decimals the
            // checks use are); a value that is not keeps the
            // f64 rounding of it, which a check written against
            // the full binary128 rendering will not match.
            let v = f64::from_bits(crate::f128::to_f64_bits(*bits));
            Some(format_float(v))
        }
        _ => None,
    }
}

fn format_int(bits: u128, signed: bool, format: Option<PrintFormat>) -> String {
    match format {
        Some(PrintFormat::Hex) => format!("{bits:#x}"),
        Some(PrintFormat::Binary) => format!("{bits:#b}"),
        _ => {
            if signed {
                // Values are stored sign-extended to 128 bits, so
                // the i128 reading is the value.
                format!("{}", bits as i128)
            } else {
                format!("{bits}")
            }
        }
    }
}

fn format_float(v: f64) -> String {
    // Rust's Display prints the shortest form that round-trips,
    // which for the simple decimals the checks use is the same
    // text gdb prints (1.5, 2.5, -10.5; an integral float prints
    // without a fraction in both).
    format!("{v}")
}

/// gdb's rendering of one character inside quotes: the printable
/// character itself, the standard escapes for the control
/// characters it names.
fn escape_char(c: char) -> String {
    match c {
        '\n' => "\\n".to_string(),
        '\t' => "\\t".to_string(),
        '\r' => "\\r".to_string(),
        '\\' => "\\\\".to_string(),
        '\'' => "\\'".to_string(),
        c if c.is_control() => format!("\\u{{{:x}}}", c as u32),
        c => c.to_string(),
    }
}

/// A parsed print expression: dereferences first (they prefix the
/// whole expression in debugger syntax), then a base variable,
/// then field/index steps.
struct ParsedExpr {
    base: String,
    steps: Vec<ExprStep>,
}

enum ExprStep {
    Deref,
    Field(usize),
    Index(usize),
}

fn parse_expr(text: &str) -> Result<ParsedExpr, String> {
    let text = text.trim();
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut steps = Vec::new();
    while i < bytes.len() && (bytes[i] == b'*' || bytes[i] == b' ') {
        if bytes[i] == b'*' {
            steps.push(ExprStep::Deref);
        }
        i += 1;
    }
    let start = i;
    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
        i += 1;
    }
    if start == i {
        return Err(format!("cannot parse expression `{text}`"));
    }
    let base = text[start..i].to_string();
    while i < bytes.len() {
        match bytes[i] {
            b'.' => {
                i += 1;
                let s = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let field = &text[s..i];
                let index: usize = field
                    .parse()
                    .map_err(|_| format!("named field `{field}` is not supported"))?;
                steps.push(ExprStep::Field(index));
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
    Ok(ParsedExpr { base, steps })
}
