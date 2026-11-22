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
//! Values render as the dialect prints them: scalars (integers,
//! booleans, floats, chars, unit) and aggregates — tuples, arrays
//! and ADT values, the last through the program's debug-shape
//! table — reached through locals, dereferences and field/index
//! projections.

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
                    let lines = render_print(interp, &ctl, frames, &print);
                    ctl.transcript.extend(lines);
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
/// session dialect's form: gdb's single `$N = <value>` line, or
/// lldb's line(s) for the value (see [`render_lldb_lines`]). A
/// value a dialect cannot render stays an explicit error line in
/// that dialect.
fn render_print(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    print: &DebugPrint,
) -> Vec<String> {
    let err_line = |e: &str| match ctl.dialect {
        DebugDialect::Gdb => {
            vec![format!("${} = <error: {e}>", ctl.print_count + 1)]
        }
        DebugDialect::Lldb => vec![format!("= <error: {e}>")],
    };
    match evaluate_print(interp, ctl, frames, print) {
        Ok(evaluated) => match ctl.dialect {
            DebugDialect::Gdb => match evaluated.gdb {
                Some(gdb) => vec![format!("${} = {gdb}", ctl.print_count + 1)],
                None => err_line(&format!(
                    "`{}` could not be rendered",
                    print.expr
                )),
            },
            DebugDialect::Lldb => {
                if evaluated.lldb.is_empty() {
                    err_line(&format!("`{}` could not be rendered", print.expr))
                } else {
                    evaluated.lldb
                }
            }
        },
        Err(e) => err_line(&e),
    }
}

/// The outcome of evaluating one print expression: the value's
/// gdb-form rendering and its lldb-form transcript lines. Either
/// may be unavailable — a form the session cannot render stays
/// an explicit error in that dialect only.
struct EvaluatedPrint {
    gdb: Option<String>,
    lldb: Vec<String>,
}

/// Evaluate a print expression in the selected frame: a variable
/// of the frame's body (as recorded in its debug variable table),
/// followed by dereferences, field and index steps, and render
/// the resulting value in both dialects' forms.
fn evaluate_print(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    print: &DebugPrint,
) -> Result<EvaluatedPrint, String> {
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
    // The value's type, as a rendering descriptor: the variable
    // entry's recorded type tag when the conversion supplied one
    // (it carries the full instance name, generic arguments
    // included), else — for a plain local — the frame body's
    // local type, which names an ADT by its definition only.
    let (mut value, mut desc): (Value, Option<Desc>) = match source {
        crate::ir::DebugVar::Place(local, proj) => {
            let value = frame
                .locals
                .get(*local)
                .cloned()
                .ok_or_else(|| format!("variable `{}` has no storage", expr.base))?;
            let desc = entry
                .ty
                .as_ref()
                .map(|tag| Desc::from_tag(tag))
                .or_else(|| {
                    if proj.is_empty() {
                        body.local_tys.get(*local).map(Desc::from_ir)
                    } else {
                        None
                    }
                });
            if proj.is_empty() {
                (value, desc)
            } else {
                (
                    crate::interp::project_value(value, proj, "debugger", &expr.base)
                        .map_err(|e| e.to_string())?,
                    desc,
                )
            }
        }
        crate::ir::DebugVar::Constant(constant) => (
            interp.eval_constant(constant).map_err(|e| e.to_string())?,
            entry.ty.as_ref().map(|tag| Desc::from_tag(tag)),
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
                desc = match desc {
                    Some(Desc::Ref(inner)) => Some(*inner),
                    Some(Desc::Adt(tag)) => {
                        smart_pointer_pointee(&tag).map(|t| Desc::from_tag(&t))
                    }
                    _ => None,
                };
                // The char hint is recorded for the variable as
                // declared: a `char` local, or a reference to a
                // char (whose pointee the hint then describes).
                // A dereference step therefore keeps it.
            }
            ExprStep::Field(index) => {
                let next =
                    desc.as_ref()
                        .and_then(|d| field_desc(interp, d, &value, *index));
                value = crate::interp::project_value(
                    value,
                    &[RefProj::Field(*index)],
                    "debugger",
                    &print.expr,
                )
                .map_err(|e| e.to_string())?;
                desc = next;
                char = false;
            }
            ExprStep::Index(index) => {
                let next = desc.as_ref().and_then(|d| match d {
                    Desc::Array(elem, _) | Desc::Slice(elem) => {
                        Some((**elem).clone())
                    }
                    _ => None,
                });
                value = crate::interp::project_value(
                    value,
                    &[RefProj::Index(*index)],
                    "debugger",
                    &print.expr,
                )
                .map_err(|e| e.to_string())?;
                desc = next;
                char = false;
            }
        }
    }
    // lldb's summary for a reference to a path is the path text
    // itself, not the address: render the pointee.
    let lldb = match &desc {
        Some(Desc::Ref(inner))
            if expr.steps.is_empty()
                && matches!(
                    inner.as_ref(),
                    Desc::Adt(tag)
                        if matches!(tag_base_name(tag), "Path" | "OsStr")
                ) =>
        {
            let pointee = interp
                .deref_value(frames, &value)
                .and_then(|resolved| interp.read_resolved(frames, &resolved));
            match pointee {
                Ok(target) => render_lldb_lines(
                    interp,
                    &target,
                    Some(inner.as_ref()),
                    char,
                    &print.expr,
                ),
                Err(_) => render_lldb_lines(
                    interp,
                    &value,
                    desc.as_ref(),
                    char,
                    &print.expr,
                ),
            }
        }
        _ => render_lldb_lines(interp, &value, desc.as_ref(), char, &print.expr),
    };
    Ok(EvaluatedPrint {
        gdb: render_gdb(interp, &value, desc.as_ref(), char, print.format, 0),
        lldb,
    })
}

// ---------------------------------------------------------------------------
// Value rendering
//
// A print expression's value is rendered from the value itself
// plus a type descriptor recovered for it: the variable entry's
// recorded type tag (the full instance name, generic arguments
// included), refined by the expression's steps. The descriptor
// drives everything the value alone does not say — whether a
// tuple-shaped value is a tuple or an array, an ADT value's
// variant and field names (through the program's debug-shape
// table), a field's element type, and a `char`'s identity.
// ---------------------------------------------------------------------------

/// A rendering descriptor for a value being printed.
#[derive(Debug, Clone, PartialEq)]
enum Desc {
    Unit,
    Bool,
    Char,
    /// An integer type; the flag is its signedness.
    Int(bool),
    Float,
    Str,
    Tuple(Vec<Desc>),
    Array(Box<Desc>, u64),
    Slice(Box<Desc>),
    Ref(Box<Desc>),
    /// An ADT instance, by its rendered type tag (the key of the
    /// program's debug-shape table).
    Adt(String),
    /// A closure type, by its body's name.
    Closure(String),
    /// A coroutine type, by its body's name.
    Coroutine(String),
    FnPtr,
    Unknown,
}

impl Desc {
    fn from_ir(ty: &Ty) -> Desc {
        match ty {
            Ty::Unit => Desc::Unit,
            Ty::Bool => Desc::Bool,
            Ty::Int(_) => Desc::Int(true),
            Ty::Uint(_) => Desc::Int(false),
            Ty::Float(_) => Desc::Float,
            Ty::Tuple(ts) => Desc::Tuple(ts.iter().map(Desc::from_ir).collect()),
            Ty::Never => Desc::Unknown,
            Ty::Adt(name) => Desc::Adt(name.clone()),
            Ty::Ref(inner) => Desc::Ref(Box::new(Desc::from_ir(inner))),
            Ty::Slice(elem) => Desc::Slice(Box::new(Desc::from_ir(elem))),
            Ty::Array(elem, n) => Desc::Array(Box::new(Desc::from_ir(elem)), *n),
            Ty::Str => Desc::Str,
            Ty::FnPtr | Ty::FnDef(_) => Desc::FnPtr,
            Ty::Closure(name, _) => Desc::Closure(name.clone()),
            Ty::Dynamic(_) => Desc::Unknown,
            Ty::Coroutine(name) => Desc::Coroutine(name.clone()),
        }
    }

    /// Parse a rendered type tag (the form [`crate::builtins`]'s
    /// `ty_tag` produces, and the form the debug-shape table's
    /// field entries carry) back into a descriptor.
    fn from_tag(tag: &str) -> Desc {
        let tag = tag.trim();
        match tag {
            "()" => return Desc::Unit,
            "bool" => return Desc::Bool,
            "char" => return Desc::Char,
            "str" => return Desc::Str,
            "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => {
                return Desc::Int(true)
            }
            "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => {
                return Desc::Int(false)
            }
            "f16" | "f32" | "f64" | "f128" => return Desc::Float,
            "!" => return Desc::Unknown,
            _ => {}
        }
        if let Some(rest) = tag.strip_prefix("&mut ") {
            return Desc::Ref(Box::new(Desc::from_tag(rest)));
        }
        if let Some(rest) = tag.strip_prefix('&') {
            return Desc::Ref(Box::new(Desc::from_tag(rest)));
        }
        if let Some(rest) = tag
            .strip_prefix("*const ")
            .or_else(|| tag.strip_prefix("*mut "))
        {
            return Desc::Ref(Box::new(Desc::from_tag(rest)));
        }
        if tag.starts_with('(') && tag.ends_with(')') {
            let inner = &tag[1..tag.len() - 1];
            if inner.trim().is_empty() {
                return Desc::Unit;
            }
            let descs = split_top_level(inner)
                .iter()
                .filter(|part| !part.is_empty())
                .map(|part| Desc::from_tag(part))
                .collect();
            return Desc::Tuple(descs);
        }
        if tag.starts_with('[') && tag.ends_with(']') {
            let inner = &tag[1..tag.len() - 1];
            if let Some((elem, len)) = inner.rsplit_once(';') {
                if let Ok(n) = len.trim().parse::<u64>() {
                    return Desc::Array(Box::new(Desc::from_tag(elem)), n);
                }
            }
            return Desc::Slice(Box::new(Desc::from_tag(inner)));
        }
        if tag.starts_with("dyn ") {
            return Desc::Unknown;
        }
        Desc::Adt(tag.to_string())
    }
}

/// The base name of a rendered type tag: the last path segment
/// of the part before any generic argument list.
fn tag_base_name(tag: &str) -> &str {
    let head = tag.split('<').next().unwrap_or(tag).trim();
    head.rsplit("::").next().unwrap_or(head)
}

/// The text of a path-family value (`PathBuf`, `Path`,
/// `OsString`, `OsStr`) in this runtime's std model: nested
/// single-field aggregates wrapping a byte list (the `Vec`
/// element model), or a plain string value.
fn path_text(value: &Value) -> Option<String> {
    let mut cur = value;
    for _ in 0..8 {
        match cur {
            Value::Str(s) => return Some(s.clone()),
            Value::Enum { fields, .. } if fields.len() == 1 => {
                cur = &fields[0];
            }
            Value::Enum { fields, .. }
                if fields
                    .iter()
                    .all(|f| matches!(f, Value::Uint(_))) =>
            {
                let bytes: Vec<u8> = fields
                    .iter()
                    .map(|f| match f {
                        Value::Uint(n) => *n as u8,
                        _ => 0,
                    })
                    .collect();
                return Some(String::from_utf8_lossy(&bytes).into_owned());
            }
            _ => return None,
        }
    }
    None
}

/// lldb's summary for a `Vec` value: in this runtime's std model
/// a vector is an ADT aggregate of its elements, not the
/// library's buffer/length layout, so the debug-shape table
/// cannot describe it; the formatter's own summary is the
/// element count with the elements by index.
fn render_vec_lldb(
    interp: &Interpreter,
    tag: &str,
    fields: &[Value],
    depth: usize,
) -> Option<String> {
    let elem = first_generic_arg(tag).map(|t| Desc::from_tag(&t));
    let mut parts = Vec::with_capacity(fields.len());
    for (i, field) in fields.iter().enumerate() {
        let v = render_lldb_compact(
            interp,
            field,
            elem.as_ref(),
            false,
            true,
            depth + 1,
        )?;
        parts.push(format!("[{i}] = {v}"));
    }
    if parts.is_empty() {
        Some("size=0 {}".to_string())
    } else {
        Some(format!("size={} {{ {} }}", fields.len(), parts.join(" ")))
    }
}

/// Split a rendered type list on its top-level commas (commas
/// inside `<...>`, `(...)`, `[...]` do not split).
fn split_top_level(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in text.char_indices() {
        match c {
            '<' | '(' | '[' | '{' => depth += 1,
            '>' | ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(text[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(text[start..].trim().to_string());
    parts
}

/// The first generic argument of a rendered instance tag.
fn first_generic_arg(tag: &str) -> Option<String> {
    let open = tag.find('<')?;
    let inner = tag.get(open + 1..tag.len().checked_sub(1)?)?;
    split_top_level(inner)
        .into_iter()
        .next()
        .filter(|arg| !arg.is_empty())
}

/// The pointee type tag of a smart-pointer instance tag (`Box`,
/// `Rc`, `Arc`): the pointee a dereference step reaches. The
/// match is on the type's base name, the convention the format
/// engine's tag dispatch uses: the recorded names spell the
/// smart pointers through both their defining crate and the
/// `std` re-export (`alloc::boxed::Box`, `std::boxed::Box`).
fn smart_pointer_pointee(tag: &str) -> Option<String> {
    let base = tag.split('<').next()?.trim();
    let name = base.rsplit("::").next()?;
    if matches!(name, "Box" | "Rc" | "Arc") {
        first_generic_arg(tag)
    } else {
        None
    }
}

/// The DWARF form of a rendered type tag. The debugger's type
/// names are definition paths, so a path that names a `std`
/// re-export of a `core` or `alloc` item takes the defining
/// crate's name: `std::option::Option` is
/// `core::option::Option`, `std::boxed::Box` is
/// `alloc::boxed::Box`, `std::alloc::Global` is
/// `alloc::alloc::Global`. The mapping is the standard library's
/// re-export structure: the whole modules `std` re-exports from
/// `core` or `alloc`, plus the individual items of the mixed
/// modules (`sync`, `collections`, `ffi`, `time`). Any other
/// path already is a definition path.
fn dwarf_type_name(tag: &str) -> String {
    fn defining_crate(seg2: &str, seg3: &str) -> Option<&'static str> {
        Some(match (seg2, seg3) {
            (
                "option" | "result" | "fmt" | "marker" | "mem" | "ptr" | "ops"
                | "cell" | "num" | "slice" | "str" | "any" | "array" | "ascii"
                | "char" | "cmp" | "convert" | "default" | "hash" | "iter"
                | "clone" | "panic" | "pin" | "task" | "future" | "intrinsics",
                _,
            ) => "core",
            ("alloc" | "borrow" | "boxed" | "rc" | "string" | "vec", _) => {
                "alloc"
            }
            ("sync", "atomic") => "core",
            ("sync", "Arc" | "Weak") => "alloc",
            (
                "collections",
                "BTreeMap" | "BTreeSet" | "BinaryHeap" | "LinkedList"
                | "VecDeque",
            ) => "alloc",
            ("ffi", "CStr") => "core",
            ("ffi", "CString") => "alloc",
            ("time", "Duration") => "core",
            _ => return None,
        })
    }
    let bytes = tag.as_bytes();
    let mut out = String::with_capacity(tag.len());
    let mut i = 0;
    while i < bytes.len() {
        let at_path_start = i == 0
            || (!bytes[i - 1].is_ascii_alphanumeric()
                && bytes[i - 1] != b'_'
                && bytes[i - 1] != b':');
        if at_path_start && tag[i..].starts_with("std::") {
            let rest = &tag[i + 5..];
            let seg2_end = rest
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .unwrap_or(rest.len());
            let seg2 = &rest[..seg2_end];
            let seg3 = rest[seg2_end..]
                .strip_prefix("::")
                .map(|r| {
                    let e = r
                        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                        .unwrap_or(r.len());
                    &r[..e]
                })
                .unwrap_or("");
            if let Some(krate) = defining_crate(seg2, seg3) {
                out.push_str(krate);
                i += 3; // consume `std`, keep the `::` and the rest
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// The descriptor of a value's `index`-th field, under `desc`:
/// the element of a tuple, the element type of an array, or the
/// recorded field type of an ADT instance (the variant taken
/// from the value itself).
fn field_desc(
    interp: &Interpreter,
    desc: &Desc,
    value: &Value,
    index: usize,
) -> Option<Desc> {
    match desc {
        Desc::Tuple(ts) => ts.get(index).cloned(),
        Desc::Array(elem, _) => Some((**elem).clone()),
        Desc::Adt(tag) => {
            let shape = interp.program().adt_debug_shapes.get(tag)?;
            let variant = match value {
                Value::Enum { variant, .. } => *variant as usize,
                _ => 0,
            };
            let field = shape.variants.get(variant)?.fields.get(index)?;
            Some(Desc::from_tag(&field.ty))
        }
        _ => None,
    }
}

/// The deepest aggregate nesting the renderers unfold. A value
/// graph in this model is finite (recursive types pass through
/// references, which render as addresses), so the cap is a
/// safeguard, not a shaping rule.
const MAX_RENDER_DEPTH: usize = 24;

/// Render a sequence of element values, in one dialect's element
/// form: per-element descriptors from `descs`, or one shared
/// element descriptor.
fn render_elements(
    interp: &Interpreter,
    fields: &[Value],
    descs: &[Desc],
    shared: Option<&Desc>,
    dialect: DebugDialect,
    depth: usize,
) -> Option<Vec<String>> {
    let mut parts = Vec::with_capacity(fields.len());
    for (i, field) in fields.iter().enumerate() {
        let d = shared.or_else(|| descs.get(i));
        parts.push(match dialect {
            DebugDialect::Gdb => {
                render_gdb(interp, field, d, false, None, depth + 1)?
            }
            DebugDialect::Lldb => {
                render_lldb_compact(interp, field, d, false, true, depth + 1)?
            }
        });
    }
    Some(parts)
}

/// gdb's array-element repeat compression: a run of more than
/// ten identical elements is printed once, followed by
/// `<repeats N times>` with the run's full length.
fn compress_repeats(rendered: &[String]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < rendered.len() {
        let mut j = i + 1;
        while j < rendered.len() && rendered[j] == rendered[i] {
            j += 1;
        }
        let run = j - i;
        if run > 10 {
            out.push(format!("{} <repeats {} times>", rendered[i], run));
        } else {
            out.extend_from_slice(&rendered[i..j]);
        }
        i = j;
    }
    out.join(", ")
}

/// Render a value the way gdb (with its Rust pretty printers
/// disabled, as the debuginfo suite runs it) prints it: scalars
/// as [`render_scalar`] renders them, tuples as `(a, b)`, arrays
/// as `[a, b]` with gdb's repeat compression, and ADT values
/// through the debug-shape table — `Tag {f: v, ...}` for a
/// named-field struct, `Tag (v, ...)` for a tuple struct,
/// `Tag::Variant{...}` / `Tag::Variant(...)` / `Tag::Variant`
/// for an enum. A reference renders as its address. Anything the
/// shape table does not describe stays unrendered (`None`).
fn render_gdb(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
    top_char: bool,
    format: Option<PrintFormat>,
    depth: usize,
) -> Option<String> {
    if depth > MAX_RENDER_DEPTH {
        return None;
    }
    match value {
        Value::Bool(_)
        | Value::Unit
        | Value::Int(_)
        | Value::Uint(_)
        | Value::Float(_)
        | Value::Float128(_) => render_scalar(
            value,
            None,
            top_char || matches!(desc, Some(Desc::Char)),
            format,
            DebugDialect::Gdb,
        ),
        Value::Tuple(fields) => {
            // A tuple-shaped value with an array descriptor is an
            // array; with a slice descriptor it is an array whose
            // length did not survive conversion (the length was
            // named by a const item) — a genuine slice value is a
            // fat reference, never a tuple.
            let array_elem = match desc {
                Some(Desc::Array(elem, _)) | Some(Desc::Slice(elem)) => {
                    Some(elem.as_ref())
                }
                _ => None,
            };
            if let Some(elem) = array_elem {
                let mut rendered = Vec::with_capacity(fields.len());
                for field in fields {
                    rendered.push(render_gdb(
                        interp,
                        field,
                        Some(elem),
                        false,
                        None,
                        depth + 1,
                    )?);
                }
                Some(format!("[{}]", compress_repeats(&rendered)))
            } else {
                let descs: Vec<Desc> = match desc {
                    Some(Desc::Tuple(ts)) => ts.clone(),
                    _ => Vec::new(),
                };
                let parts = render_elements(
                    interp,
                    fields,
                    &descs,
                    None,
                    DebugDialect::Gdb,
                    depth,
                )?;
                if parts.len() == 1 {
                    Some(format!("({},)", parts[0]))
                } else {
                    Some(format!("({})", parts.join(", ")))
                }
            }
        }
        Value::Enum { variant, fields } => {
            let Some(Desc::Adt(tag)) = desc else {
                return None;
            };
            let shape = interp.program().adt_debug_shapes.get(tag)?;
            render_adt_gdb(interp, tag, shape, *variant, fields, depth)
        }
        Value::Ref(_)
        | Value::FatRef { .. }
        | Value::DynRef { .. }
        | Value::FnPtr(_) => {
            let addr = crate::interp::synthetic_address(value)?;
            Some(format!("0x{addr:x}"))
        }
        Value::Uninit => match desc {
            Some(Desc::Int(_)) | Some(Desc::Float) => Some("0".to_string()),
            Some(Desc::Bool) => Some("false".to_string()),
            _ => None,
        },
        _ => None,
    }
}

/// Render one ADT value in gdb form (see [`render_gdb`]).
fn render_adt_gdb(
    interp: &Interpreter,
    tag: &str,
    shape: &sparagmos_mir::AdtDebugShape,
    variant: u32,
    fields: &[Value],
    depth: usize,
) -> Option<String> {
    // The printed name is the DWARF (definition-path) form of
    // the recorded tag; the shape lookup above used the tag
    // itself, which is the table's key.
    let tag = &dwarf_type_name(tag);
    let vshape = shape.variants.get(variant as usize)?;
    if vshape.fields.len() != fields.len() {
        return None;
    }
    let descs: Vec<Desc> = vshape
        .fields
        .iter()
        .map(|f| Desc::from_tag(&f.ty))
        .collect();
    let named = vshape.fields.iter().all(|f| f.name.is_some());
    let named_parts = |interp: &Interpreter| -> Option<Vec<String>> {
        let mut parts = Vec::with_capacity(fields.len());
        for (i, field) in fields.iter().enumerate() {
            let name = vshape.fields[i].name.as_deref()?;
            let v = render_gdb(
                interp,
                field,
                Some(&descs[i]),
                false,
                None,
                depth + 1,
            )?;
            parts.push(format!("{name}: {v}"));
        }
        Some(parts)
    };
    match shape.kind {
        sparagmos_mir::AdtShapeKind::Struct => {
            if fields.is_empty() {
                return Some(tag.to_string());
            }
            if named {
                let parts = named_parts(interp)?;
                Some(format!("{tag} {{{}}}", parts.join(", ")))
            } else {
                let parts = render_elements(
                    interp,
                    fields,
                    &descs,
                    None,
                    DebugDialect::Gdb,
                    depth,
                )?;
                Some(format!("{tag} ({})", parts.join(", ")))
            }
        }
        sparagmos_mir::AdtShapeKind::Enum => {
            let prefix = format!("{tag}::{}", vshape.name);
            if fields.is_empty() {
                return Some(prefix);
            }
            if named {
                let parts = named_parts(interp)?;
                Some(format!("{prefix}{{{}}}", parts.join(", ")))
            } else {
                let parts = render_elements(
                    interp,
                    fields,
                    &descs,
                    None,
                    DebugDialect::Gdb,
                    depth,
                )?;
                Some(format!("{prefix}({})", parts.join(", ")))
            }
        }
        sparagmos_mir::AdtShapeKind::Union => None,
    }
}

/// Render a value in lldb's compact form (the summary its
/// `frame variable` output shows): scalars as [`render_scalar`]
/// renders them, aggregates in brace form with no space around
/// the colon (`{x:10, y:23.5}`), tuples and arrays in tuple form
/// (`(0, 1, 2)`), an enum as its variant — named at the top
/// level, nameless when nested (`{a:1, b:2}`) — and a reference
/// as its address.
fn render_lldb_compact(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
    top_char: bool,
    nested: bool,
    depth: usize,
) -> Option<String> {
    if depth > MAX_RENDER_DEPTH {
        return None;
    }
    match value {
        Value::Bool(_)
        | Value::Unit
        | Value::Int(_)
        | Value::Uint(_)
        | Value::Float(_)
        | Value::Float128(_) => render_scalar(
            value,
            None,
            top_char || matches!(desc, Some(Desc::Char)),
            None,
            DebugDialect::Lldb,
        ),
        Value::Tuple(fields) => {
            let (descs, shared): (Vec<Desc>, Option<&Desc>) = match desc {
                Some(Desc::Tuple(ts)) => (ts.clone(), None),
                Some(Desc::Array(elem, _)) => (Vec::new(), Some(elem.as_ref())),
                _ => (Vec::new(), None),
            };
            let parts = render_elements(
                interp,
                fields,
                &descs,
                shared,
                DebugDialect::Lldb,
                depth,
            )?;
            Some(format!("({})", parts.join(", ")))
        }
        Value::Enum { variant, fields } => {
            let Some(Desc::Adt(tag)) = desc else {
                return None;
            };
            // The std model's own aggregates, which the shape
            // table (the library's declared layout) cannot
            // describe: a `Vec` is its element list, and the
            // path family wraps its text in single-field layers.
            match tag_base_name(tag) {
                "Vec" => {
                    return render_vec_lldb(interp, tag, fields, depth);
                }
                "PathBuf" | "Path" | "OsString" | "OsStr" => {
                    return path_text(value)
                        .map(|text| format!("\"{}\"", escape_str(&text)));
                }
                _ => {}
            }
            let shape = interp.program().adt_debug_shapes.get(tag)?;
            render_adt_lldb_compact(interp, shape, *variant, fields, nested, depth)
        }
        Value::Ref(_)
        | Value::FatRef { .. }
        | Value::DynRef { .. }
        | Value::FnPtr(_) => {
            let addr = crate::interp::synthetic_address(value)?;
            Some(format!("0x{addr:x}"))
        }
        Value::Str(s) => Some(format!("\"{}\"", escape_str(s))),
        Value::Uninit => match desc {
            Some(Desc::Int(_)) | Some(Desc::Float) => Some("0".to_string()),
            Some(Desc::Bool) => Some("false".to_string()),
            _ => None,
        },
        _ => None,
    }
}

/// Render one ADT value in lldb's compact form (see
/// [`render_lldb_compact`]).
fn render_adt_lldb_compact(
    interp: &Interpreter,
    shape: &sparagmos_mir::AdtDebugShape,
    variant: u32,
    fields: &[Value],
    nested: bool,
    depth: usize,
) -> Option<String> {
    let vshape = shape.variants.get(variant as usize)?;
    if vshape.fields.len() != fields.len() {
        return None;
    }
    let descs: Vec<Desc> = vshape
        .fields
        .iter()
        .map(|f| Desc::from_tag(&f.ty))
        .collect();
    let named = vshape.fields.iter().all(|f| f.name.is_some());
    let brace_parts = |interp: &Interpreter| -> Option<Vec<String>> {
        let mut parts = Vec::with_capacity(fields.len());
        for (i, field) in fields.iter().enumerate() {
            let v = render_lldb_compact(
                interp,
                field,
                Some(&descs[i]),
                false,
                true,
                depth + 1,
            )?;
            if named {
                let name = vshape.fields[i].name.as_deref()?;
                parts.push(format!("{name}:{v}"));
            } else {
                parts.push(format!("{i}:{v}"));
            }
        }
        Some(parts)
    };
    match shape.kind {
        sparagmos_mir::AdtShapeKind::Struct => {
            let parts = brace_parts(interp)?;
            Some(format!("{{{}}}", parts.join(", ")))
        }
        sparagmos_mir::AdtShapeKind::Enum => {
            if fields.is_empty() {
                return Some(vshape.name.clone());
            }
            if named {
                let parts = brace_parts(interp)?;
                let braces = format!("{{{}}}", parts.join(", "));
                if nested {
                    Some(braces)
                } else {
                    Some(format!("{}{}", vshape.name, braces))
                }
            } else {
                let parts = render_elements(
                    interp,
                    fields,
                    &descs,
                    None,
                    DebugDialect::Lldb,
                    depth,
                )?;
                if nested {
                    Some(format!("({})", parts.join(", ")))
                } else {
                    Some(format!("{}({})", vshape.name, parts.join(", ")))
                }
            }
        }
        sparagmos_mir::AdtShapeKind::Union => None,
    }
}

/// Render a field value inside lldb's expanded children block:
/// a nested aggregate expands in turn when it has fields of its
/// own, and anything else takes the compact form.
fn render_lldb_sub(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
    depth: usize,
) -> Option<String> {
    if matches!(value, Value::Tuple(_) | Value::Enum { .. }) {
        render_lldb_expanded(interp, value, desc, depth)
            .or_else(|| render_lldb_compact(interp, value, desc, false, true, depth))
    } else {
        render_lldb_compact(interp, value, desc, false, true, depth)
    }
}

/// Render an aggregate value in lldb's expanded children form:
/// `{ f = v f = v }`, each field's value as [`render_lldb_sub`]
/// renders it. Tuple and array elements are named by their
/// index (`0 = v`, `[0] = v`). `None` for a value with no fields
/// to expand.
fn render_lldb_expanded(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
    depth: usize,
) -> Option<String> {
    if depth > MAX_RENDER_DEPTH {
        return None;
    }
    match value {
        Value::Enum { variant, fields } => {
            let Some(Desc::Adt(tag)) = desc else {
                return None;
            };
            let shape = interp.program().adt_debug_shapes.get(tag)?;
            let vshape = shape.variants.get(*variant as usize)?;
            if vshape.fields.len() != fields.len() || fields.is_empty() {
                return None;
            }
            let mut pairs = Vec::with_capacity(fields.len());
            for (i, field) in fields.iter().enumerate() {
                let name = match &vshape.fields[i].name {
                    Some(name) => name.clone(),
                    None => i.to_string(),
                };
                let d = Desc::from_tag(&vshape.fields[i].ty);
                let v = render_lldb_sub(interp, field, Some(&d), depth + 1)?;
                pairs.push(format!("{name} = {v}"));
            }
            Some(format!("{{ {} }}", pairs.join(" ")))
        }
        Value::Tuple(fields) => {
            if fields.is_empty() {
                return None;
            }
            let is_array = matches!(
                desc,
                Some(Desc::Array(..)) | Some(Desc::Slice(..))
            );
            let mut pairs = Vec::with_capacity(fields.len());
            for (i, field) in fields.iter().enumerate() {
                let d = match desc {
                    Some(Desc::Tuple(ts)) => ts.get(i),
                    Some(Desc::Array(elem, _)) | Some(Desc::Slice(elem)) => {
                        Some(elem.as_ref())
                    }
                    _ => None,
                };
                let name = if is_array {
                    format!("[{i}]")
                } else {
                    i.to_string()
                };
                let v = render_lldb_sub(interp, field, d, depth + 1)?;
                pairs.push(format!("{name} = {v}"));
            }
            Some(format!("{{ {} }}", pairs.join(" ")))
        }
        _ => None,
    }
}

/// The (head name, fields) of a named-field aggregate value:
/// the head is the struct's base name or the enum's variant
/// name, and each field carries its gdb-form and lldb-compact
/// renderings. `None` when the value is not a named-field
/// aggregate.
fn named_field_values(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
) -> Option<(String, Vec<(String, String, String)>)> {
    let Value::Enum { variant, fields } = value else {
        return None;
    };
    let Some(Desc::Adt(tag)) = desc else {
        return None;
    };
    let shape = interp.program().adt_debug_shapes.get(tag)?;
    let vshape = shape.variants.get(*variant as usize)?;
    if vshape.fields.len() != fields.len() || fields.is_empty() {
        return None;
    }
    if !vshape.fields.iter().all(|f| f.name.is_some()) {
        return None;
    }
    let head = match shape.kind {
        sparagmos_mir::AdtShapeKind::Struct => tag
            .split('<')
            .next()?
            .rsplit("::")
            .next()?
            .to_string(),
        sparagmos_mir::AdtShapeKind::Enum => vshape.name.clone(),
        sparagmos_mir::AdtShapeKind::Union => return None,
    };
    let mut pairs = Vec::with_capacity(fields.len());
    for (i, field) in fields.iter().enumerate() {
        let d = Desc::from_tag(&vshape.fields[i].ty);
        let gdb = render_gdb(interp, field, Some(&d), false, None, 1)?;
        let compact = render_lldb_compact(interp, field, Some(&d), false, true, 1)?;
        pairs.push((vshape.fields[i].name.clone()?, gdb, compact));
    }
    Some((head, pairs))
}

/// The transcript lines of one lldb print. A scalar keeps the
/// session's established form (`= <value>`). An aggregate value
/// gets the lines lldb's own output forms suggest, in order: the
/// `(Type) name = <compact>` summary line; the same line with
/// the expanded children block appended; and, for a named-field
/// aggregate, the `Name { f: v, ... }` line and the
/// comma-separated `{ f = v, ... }` line. Check lines match
/// these as fragments in order, so the several forms coexist:
/// each print contributes every form its value has.
fn render_lldb_lines(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
    top_char: bool,
    expr: &str,
) -> Vec<String> {
    let Some(compact) = render_lldb_compact(interp, value, desc, top_char, false, 0)
    else {
        return Vec::new();
    };
    if !matches!(value, Value::Tuple(_) | Value::Enum { .. }) {
        return vec![format!("= {compact}")];
    }
    let mut lines = Vec::new();
    if let Some(Desc::Adt(tag)) = desc {
        let first = format!(
            "({}) {expr} = {compact}",
            lldb_type_name(&dwarf_type_name(tag))
        );
        lines.push(first.clone());
        if let Some(expanded) = render_lldb_expanded(interp, value, desc, 0) {
            lines.push(format!("{first} {expanded}"));
        }
        if let Some((head, pairs)) = named_field_values(interp, value, desc) {
            let colon = pairs
                .iter()
                .map(|(name, gdb, _)| format!("{name}: {gdb}"))
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("{head} {{ {colon} }}"));
            let eq = pairs
                .iter()
                .map(|(name, _, compact)| format!("{name} = {compact}"))
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("{{ {eq} }}"));
        }
    } else {
        lines.push(format!("= {compact}"));
        if let Some(expanded) = render_lldb_expanded(interp, value, desc, 0) {
            lines.push(format!("= {compact} {expanded}"));
        }
    }
    lines
}

/// An ADT instance tag as lldb renders the type in its
/// `(Type) name = value` output: scalar generic arguments take
/// the C names of the scalar types (`i32` as `int`, `f64` as
/// `double`), nested closing brackets are separated by a space
/// (`> >`), and tuple arguments keep their Rust spelling.
fn lldb_type_name(tag: &str) -> String {
    fn scalar_name(word: &str) -> &str {
        match word {
            "i8" => "signed char",
            "i16" => "short",
            "i32" => "int",
            "i64" => "long",
            "isize" => "long",
            "u8" => "unsigned char",
            "u16" => "unsigned short",
            "u32" => "unsigned int",
            "u64" => "unsigned long",
            "usize" => "unsigned long",
            "f32" => "float",
            "f64" => "double",
            _ => word,
        }
    }
    let bytes = tag.as_bytes();
    let mut out = String::with_capacity(tag.len());
    let mut angle = 0usize;
    let mut paren = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => {
                angle += 1;
                out.push('<');
                i += 1;
            }
            b'>' => {
                angle = angle.saturating_sub(1);
                if out.ends_with('>') {
                    out.push(' ');
                }
                out.push('>');
                i += 1;
            }
            b'(' => {
                paren += 1;
                out.push('(');
                i += 1;
            }
            b')' => {
                paren = paren.saturating_sub(1);
                out.push(')');
                i += 1;
            }
            c if angle > 0 && paren == 0 && (c.is_ascii_alphabetic() || c == b'_') => {
                let start = i;
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
                {
                    i += 1;
                }
                out.push_str(scalar_name(&tag[start..i]));
            }
            c => {
                out.push(c as char);
                i += 1;
            }
        }
    }
    out
}

/// Escape a string value for lldb's quoted string summary.
fn escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out
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
