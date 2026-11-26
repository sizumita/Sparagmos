//! Session state and the drive hook: the command vocabulary, the session and its control block, and the stop machinery.

use crate::interp::{Frame, Interpreter};
use crate::ir::{Body, Program, SourceLine};
use crate::value::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use super::evaluate::{evaluate_print, render_print};
use super::queries::{render_info_functions, render_info_locals, render_type_query};

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
    /// Print an expression's declared type (gdb `whatis`):
    /// emits `type = <type>` in gdb form.
    WhatIs(DebugPrint),
    /// Print an expression's declared type with its layout
    /// (gdb `ptype`): the `whatis` head line plus, for an
    /// aggregate type, the field lines of its shape.
    Ptype(DebugPrint),
    /// The debugger_tester `repr` pseudo-command: evaluate and
    /// render the named value; success emits `<name>: Ok` (the
    /// check line compiletest synthesises for the directive).
    Repr(String),
    /// Move the frame selection one frame toward the caller
    /// (gdb/lldb `up`): emits the newly selected frame's line.
    FrameUp,
    /// Move the frame selection one frame toward the innermost
    /// frame (gdb/lldb `down`): emits the newly selected
    /// frame's line.
    FrameDown,
    /// List the program's functions whose names match a
    /// pattern (gdb `info functions [-q] [pattern]`): one
    /// declaration line per matching body. `quiet` is gdb's
    /// `-q` form, whose lines carry the `static fn` prefix and
    /// the full signature; the plain form is `fn <name>();`.
    InfoFunctions {
        pattern: Option<String>,
        quiet: bool,
    },
    /// List the selected frame's locals (gdb `info locals`):
    /// `No locals.` when none are in scope at the stop.
    InfoLocals,
    /// Register an lldb type format (lldb `type format add`):
    /// takes effect at this point in the session, so a print
    /// before the command renders under the earlier
    /// registration and a print after it under the new one --
    /// the order the real debugger applies its commands in.
    SetLldbFormat { ty: String, format: LldbNumFormat },
    /// Set gdb's `print elements` limit (gdb `set print
    /// elements N`): takes effect at this point in the
    /// session, bounding the string contents and element
    /// children later prints render.
    SetPrintElements(usize),
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
    /// `print/u`: unsigned decimal, read at the value's own
    /// width (a negative value prints as its two's-complement
    /// reading at that width).
    Unsigned,
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

/// A numeric display format an lldb `type format add` command
/// registered for one of the char-family types (lldb renders a
/// value of a registered type through the format instead of
/// the type's own form).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LldbNumFormat {
    /// Signed decimal (`-f d` / `--format decimal`).
    Decimal,
    /// Unsigned decimal (`-f u`).
    Unsigned,
    /// Hexadecimal (`-f x` / `--format hex`).
    Hex,
}

/// The session's lldb type-format registrations, restricted to
/// the char family — the only types the debuginfo suite
/// registers formats for: Rust `char` (lldb's `char`), `i8`
/// (lldb's `signed char`), and `u8` (lldb's `unsigned char`).
/// A registration for any of a type's names sets the whole
/// slot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LldbFormats {
    pub char_format: Option<LldbNumFormat>,
    pub i8_format: Option<LldbNumFormat>,
    pub u8_format: Option<LldbNumFormat>,
}

impl LldbFormats {
    /// Register one format for the type named `ty` (any of the
    /// names the suite uses for a char-family type, Rust or C
    /// spelling). Names outside the char family are ignored:
    /// no suite registration names another type, and the
    /// renderers consult only these slots.
    pub fn register(&mut self, ty: &str, format: LldbNumFormat) {
        match ty {
            "char" => self.char_format = Some(format),
            "i8" | "signed char" => self.i8_format = Some(format),
            "u8" | "unsigned char" => self.u8_format = Some(format),
            _ => {}
        }
    }
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
    /// The last position the current step plan ranged over;
    /// step plans fire on line *changes* within that range, so
    /// consecutive statements on one line are a single position,
    /// and positions outside the range (a stepped-over call's
    /// body) leave the tracker untouched. Re-anchored at every
    /// stop.
    last_pos: Option<(u32, u32)>,
    /// Where each line breakpoint has bound, per frame: a
    /// line breakpoint sits at the line's first instruction —
    /// in this model, the first position executed on that line
    /// in that frame — so later positions on the same line (the
    /// continuation after a call, a merge block's transfer
    /// terminator) do not stop. The bound position itself stops
    /// again once execution has left it, which is what a loop
    /// iteration does. Keyed by (frame id, file id, line); the
    /// value is the bound (block, statement) position and
    /// whether the binding is armed, i.e. execution has left
    /// the bound position since the last stop there.
    bp_bindings: HashMap<(u64, u32, u32), ((usize, usize), bool)>,
    /// The frame expressions evaluate in, counted from the
    /// innermost frame (0).
    pub(super) selected_frame: usize,
    /// The position of the current stop, for resolving a variable
    /// name against the scopes in scope there.
    pub(super) current_pos: Option<(u32, u32)>,
    /// The `$N` counter of gdb-form print output.
    pub(super) print_count: u32,
    /// The transcript lines emitted so far.
    transcript: Vec<String>,
    /// Source text by file id, for rendering stop lines.
    source_text: HashMap<u32, Vec<String>>,
    /// The output dialect.
    pub(super) dialect: DebugDialect,
    /// Whether gdb prints render through the sysroot Rust
    /// pretty printers (compiletest's default; a test opts out
    /// with `disable-gdb-pretty-printers`). Only the gdb form
    /// consults it.
    pub(super) pretty_printers: bool,
    /// gdb's `print elements` limit as set by the session's
    /// `set print elements` command, if any: the bound later
    /// prints apply to string contents and element children.
    /// Unset keeps the renderers' own defaults.
    pub(super) print_elements: Option<usize>,
    /// Function breakpoints (gdb `break <fn-path>` in a command
    /// section): body names to stop at on first entry.
    fn_breakpoints: Vec<String>,
    /// Frame ids a function breakpoint has already stopped in:
    /// a function breakpoint stops once per frame instance, at
    /// the first position that frame executes.
    fn_bp_stopped: HashSet<u64>,
    /// The program's statics' initializer constants, by static
    /// key (see [`collect_static_seeds`]): the seeds a print of
    /// a static evaluates to reach its storage before the
    /// program itself has touched it.
    pub(super) static_seeds: HashMap<String, crate::ir::Constant>,
    /// The session's lldb type-format registrations.
    pub(super) lldb_formats: LldbFormats,
    /// Whether the program was compiled with limited debug
    /// info (`-C debuginfo=1` or lower): the binary then
    /// carries no variable information, and `info locals`
    /// reports none.
    pub(super) hide_locals: bool,
}

/// A debugger session over one converted program.
pub struct DebugSession {
    program: Program,
    entry: String,
    args: Vec<Value>,
    config: crate::InterpConfig,
    breakpoints: HashSet<(u32, u32)>,
    fn_breakpoints: Vec<String>,
    source_text: HashMap<u32, Vec<String>>,
    pretty_printers: bool,
    lldb_formats: LldbFormats,
    hide_locals: bool,
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
            fn_breakpoints: Vec::new(),
            source_text: HashMap::new(),
            pretty_printers: false,
            lldb_formats: LldbFormats::default(),
            hide_locals: false,
        }
    }

    /// Register an lldb type format for a char-family type
    /// (lldb `type format add`): values of the type render
    /// through the format for the rest of the session.
    pub fn set_lldb_format(&mut self, ty: &str, format: LldbNumFormat) {
        self.lldb_formats.register(ty, format);
    }

    /// Mark the program as compiled with limited debug info:
    /// `info locals` then reports no locals (see
    /// [`DebugControl::hide_locals`]).
    pub fn set_hide_locals(&mut self, hide: bool) {
        self.hide_locals = hide;
    }

    /// Set a function breakpoint (gdb `break <fn-path>`): the
    /// session stops at the first position executed by each
    /// frame of a body whose name matches `name` (exactly, or
    /// as a suffix at a `::` boundary).
    pub fn set_fn_breakpoint(&mut self, name: &str) {
        self.fn_breakpoints.push(name.to_string());
    }

    /// Set whether gdb prints in this session render through
    /// the sysroot Rust pretty printers (see
    /// [`DebugControl::pretty_printers`]). Sessions default to
    /// the raw form, as if the printers were disabled.
    pub fn set_pretty_printers(&mut self, on: bool) {
        self.pretty_printers = on;
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
        let script: VecDeque<DebugCommand> = commands.into();
        let mut ctl = DebugControl {
            breakpoints: std::mem::take(&mut self.breakpoints),
            fn_breakpoints: std::mem::take(&mut self.fn_breakpoints),
            fn_bp_stopped: HashSet::new(),
            static_seeds: collect_static_seeds(&self.program),
            script,
            plan: StopPlan::Breakpoints,
            last_pos: None,
            current_pos: None,
            bp_bindings: HashMap::new(),
            selected_frame: 0,
            print_count: 0,
            transcript: Vec::new(),
            source_text: std::mem::take(&mut self.source_text),
            dialect,
            pretty_printers: self.pretty_printers,
            print_elements: None,
            lldb_formats: self.lldb_formats,
            hide_locals: self.hide_locals,
        };
        let mut interp = Interpreter::with_config(&self.program, self.config.clone());
        // Commands ahead of the session's first `run` are
        // consumed against the loaded-but-not-started program,
        // as a real debugger consumes them: prints there read
        // the statics at their initial values (the only values
        // in scope before any frame exists). A leading `run`
        // itself is the session start and is consumed here.
        loop {
            match ctl.script.front() {
                Some(DebugCommand::Run) => {
                    ctl.script.pop_front();
                    break;
                }
                Some(
                    DebugCommand::Print(_)
                    | DebugCommand::WhatIs(_)
                    | DebugCommand::Ptype(_)
                    | DebugCommand::Repr(_)
                    | DebugCommand::InfoFunctions { .. }
                    | DebugCommand::SetLldbFormat { .. }
                    | DebugCommand::SetPrintElements(_),
                ) => {
                    let cmd = ctl.script.pop_front().expect("front checked");
                    exec_session_command(&mut ctl, &interp, &[], &cmd);
                }
                _ => break,
            }
        }
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

/// Execute one non-resuming session command against the current
/// stop (or, with no frames, against the not-yet-started
/// program): prints, type queries, the `repr` pseudo-command
/// and the `info` listings render into the transcript. Shared
/// by the pre-run phase and the drive hook's stop loop.
fn exec_session_command(
    ctl: &mut DebugControl,
    interp: &Interpreter,
    frames: &[Frame],
    cmd: &DebugCommand,
) {
    match cmd {
        DebugCommand::Print(print) => {
            let lines = render_print(interp, ctl, frames, print);
            ctl.transcript.extend(lines);
            if ctl.dialect == DebugDialect::Gdb {
                ctl.print_count += 1;
            }
        }
        DebugCommand::WhatIs(print) => {
            let lines = render_type_query(interp, ctl, frames, print, false);
            ctl.transcript.extend(lines);
        }
        DebugCommand::Ptype(print) => {
            let lines = render_type_query(interp, ctl, frames, print, true);
            ctl.transcript.extend(lines);
        }
        DebugCommand::Repr(name) => {
            let print = DebugPrint {
                expr: name.clone(),
                format: None,
            };
            let ok = match evaluate_print(interp, ctl, frames, &print) {
                Ok(ev) => match ctl.dialect {
                    DebugDialect::Gdb => ev.gdb.is_some(),
                    DebugDialect::Lldb => !ev.lldb.is_empty(),
                },
                Err(_) => false,
            };
            ctl.transcript.push(if ok {
                format!("{name}: Ok")
            } else {
                format!("{name}: Err")
            });
        }
        DebugCommand::InfoFunctions { pattern, quiet } => {
            let lines = render_info_functions(interp, pattern.as_deref(), *quiet);
            ctl.transcript.extend(lines);
        }
        DebugCommand::InfoLocals => {
            let lines = render_info_locals(interp, ctl, frames);
            ctl.transcript.extend(lines);
        }
        DebugCommand::SetLldbFormat { ty, format } => {
            ctl.lldb_formats.register(ty, *format);
        }
        DebugCommand::SetPrintElements(n) => {
            ctl.print_elements = Some(*n);
        }
        _ => {}
    }
}

/// The program's statics' initializer constants, by static
/// key: the seed table the conversion recorded in the program
/// (see `ir::Program::static_seeds`). A static no converted
/// reference names has no seed here; a print of it stays an
/// explicit error, as it did before statics could be printed
/// at all.
fn collect_static_seeds(
    program: &Program,
) -> HashMap<String, crate::ir::Constant> {
    program.static_seeds.clone()
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
    // Storage markers and nops generate no code, so native
    // debuggers have no line row for them: they are not
    // positions in the session's stream at all. (A macro block
    // argument's result temp, for instance, gets a StorageLive
    // whose span is the `$b` reference in the macro definition;
    // counting it would stop the session on a line no debugger
    // can stop on.)
    if let Some(stmt_ref) = body.blocks.get(block).and_then(|b| b.statements.get(stmt)) {
        if matches!(
            stmt_ref.kind,
            crate::ir::StatementKind::StorageLive(_)
                | crate::ir::StatementKind::StorageDead(_)
                | crate::ir::StatementKind::Nop
        ) {
            return;
        }
    }
    let Some(pos) = position(body, block, stmt) else {
        return;
    };
    if std::env::var_os("SPARAGMOS_DEBUG_TRACE").is_some() {
        let kind = body
            .blocks
            .get(block)
            .and_then(|b| b.statements.get(stmt))
            .map(|s| format!("{:?}", s.kind))
            .unwrap_or_else(|| {
                body.blocks
                    .get(block)
                    .map(|b| format!("Terminator {:?}", b.terminator))
                    .unwrap_or_else(|| "?".to_string())
            });
        eprintln!(
            "[dbgtrace] frame={} block={} stmt={} line={} kind={:.60}",
            frames.len(),
            block,
            stmt,
            pos.line,
            kind
        );
    }
    let key = (pos.file, pos.line);
    let mut ctl = interp
        .debug_ctl
        .take()
        .expect("drive hook fires only with a session installed");
    let depth = frames.len();
    let frame_id = frames.last().map(|f| f.id).unwrap_or(0);
    // Leaving a bound breakpoint position re-arms its
    // binding, whatever line execution moved to.
    for ((fid, _, _), binding) in ctl.bp_bindings.iter_mut() {
        if *fid == frame_id && binding.0 != (block, stmt) {
            binding.1 = true;
        }
    }
    // A step stop fires on a line *change*, but the change is
    // judged only over the positions the plan ranges over. A
    // step-over plan ranges over the stepped frame and its
    // callers: positions inside a stepped-over call must not
    // rewrite the tracker, or the return to the call line's
    // continuation reads as a fresh change and the session stops
    // on the call line a second time. (A real debugger's `next`
    // runs to the return address; the callee's lines are not
    // part of the stepped frame's line stream.) A step-out plan
    // ranges over the callers only, and its stop is the first
    // position there — the return landing — whatever its line.
    let eligible = match ctl.plan {
        StopPlan::StepInto => true,
        StopPlan::StepOver(d) => depth <= d,
        StopPlan::StepOut(d) => depth < d,
        StopPlan::Breakpoints | StopPlan::Done => false,
    };
    let changed = eligible && ctl.last_pos != Some(key);
    if eligible {
        ctl.last_pos = Some(key);
    }
    // A function breakpoint stops at the first position a
    // matching frame executes, under any plan that is still
    // running the program (gdb breakpoints interrupt `next`
    // too; only the exhausted-script plan runs free).
    let fn_stop = !matches!(ctl.plan, StopPlan::Done)
        && !ctl.fn_breakpoints.is_empty()
        && !ctl.fn_bp_stopped.contains(&frame_id)
        && ctl
            .fn_breakpoints
            .iter()
            .any(|pat| fn_name_matches(&body.name, pat));
    if fn_stop {
        ctl.fn_bp_stopped.insert(frame_id);
    }
    let stop = fn_stop
        || match ctl.plan {
        StopPlan::Breakpoints => {
            if !ctl.breakpoints.contains(&key) {
                false
            } else {
                let bkey = (frame_id, key.0, key.1);
                match ctl.bp_bindings.get(&bkey) {
                    // The breakpoint's first executed position
                    // in this frame: it binds here and stops.
                    None => {
                        ctl.bp_bindings.insert(bkey, ((block, stmt), false));
                        true
                    }
                    Some(&((bb, ss), armed)) => {
                        if (bb, ss) == (block, stmt) && armed {
                            ctl.bp_bindings.insert(bkey, ((bb, ss), false));
                            true
                        } else {
                            false
                        }
                    }
                }
            }
        }
        StopPlan::StepInto => changed,
        StopPlan::StepOver(_) => changed,
        StopPlan::StepOut(d) => depth < d,
        StopPlan::Done => false,
    };
    if stop {
        if std::env::var_os("SPARAGMOS_DEBUG_TRACE").is_some() {
            eprintln!("[dbgtrace] STOP depth={} line={}", depth, pos.line);
        }
        // Anchor the step tracker at the stop: the next step
        // command measures line changes from here, not from
        // whatever position an earlier plan last ranged over.
        ctl.last_pos = Some(key);
        ctl.selected_frame = 0;
        ctl.current_pos = Some(key);
        if let Some(lines) = ctl.source_text.get(&pos.file) {
            if let Some(text) = lines.get((pos.line - 1) as usize) {
                let rendered = stop_line_text(&ctl, pos.line, text);
                ctl.transcript.push(rendered);
            }
        }
        // Consume commands until one resumes execution.
        loop {
            match ctl.script.pop_front() {
                Some(
                    cmd @ (DebugCommand::Print(_)
                    | DebugCommand::WhatIs(_)
                    | DebugCommand::Ptype(_)
                    | DebugCommand::Repr(_)
                    | DebugCommand::InfoFunctions { .. }
                    | DebugCommand::InfoLocals
                    | DebugCommand::SetLldbFormat { .. }
                    | DebugCommand::SetPrintElements(_)),
                ) => {
                    exec_session_command(&mut ctl, interp, frames, &cmd);
                }
                Some(DebugCommand::FrameSelect(n)) => {
                    ctl.selected_frame = n;
                }
                Some(DebugCommand::FrameUp) => {
                    if ctl.selected_frame + 1 < frames.len() {
                        ctl.selected_frame += 1;
                    }
                    emit_selected_line(&mut ctl, interp, frames, pos);
                }
                Some(DebugCommand::FrameDown) => {
                    ctl.selected_frame = ctl.selected_frame.saturating_sub(1);
                    emit_selected_line(&mut ctl, interp, frames, pos);
                }
                Some(DebugCommand::ShowLine) => {
                    emit_selected_line(&mut ctl, interp, frames, pos);
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

/// A stop's source line as the dialect's debugger shows it:
/// gdb lists the line numbered, with the source text as
/// written (indentation intact — check lines are written
/// against that form), lldb's form here is the bare text.
fn stop_line_text(ctl: &DebugControl, line: u32, text: &str) -> String {
    match ctl.dialect {
        DebugDialect::Gdb => format!("{line}\t{text}"),
        DebugDialect::Lldb => text.trim().to_string(),
    }
}

/// Whether a body name matches a function-breakpoint pattern:
/// an exact match, or the pattern naming a suffix of the body
/// name at a `::` boundary (the pattern names the item from
/// its crate root; a body key may carry a longer prefix for
/// items the conversion names through their impl).
fn fn_name_matches(body_name: &str, pattern: &str) -> bool {
    // A body's later instances carry the driver's instance
    // suffix (`test::helper [test::helper::<i32>]`, see
    // `collect_local_instances` in the driver): the breakpoint
    // names the definition, so the suffix is stripped before
    // matching, or every instance after the first is missed.
    let body_name = match body_name.split_once(" [") {
        Some((head, _)) if body_name.ends_with(']') => head,
        _ => body_name,
    };
    if body_name == pattern {
        return true;
    }
    match body_name.len().checked_sub(pattern.len()) {
        Some(idx) if idx >= 2 => {
            body_name.ends_with(pattern) && body_name[..idx].ends_with("::")
        }
        _ => false,
    }
}

/// The source position a frame is suspended at: the position
/// of the statement (or terminator) it is about to execute —
/// for a caller frame, its suspended call line.
pub(super) fn frame_position(interp: &Interpreter, frame: &Frame) -> Option<SourceLine> {
    let body = interp.program().get(&frame.func)?;
    let (block, stmt) = frame.current_block_stmt();
    position(body, block, stmt)
}

/// Emit the selected frame's current source line into the
/// transcript (gdb `frame` / `up` / `down`, lldb `frame
/// select`): the stop line itself for the innermost frame,
/// the caller frame's suspended call line for an outer one.
fn emit_selected_line(
    ctl: &mut DebugControl,
    interp: &Interpreter,
    frames: &[Frame],
    stop_pos: SourceLine,
) {
    let mut pos = stop_pos;
    if ctl.selected_frame != 0 {
        if let Some(frame) =
            frames.get(frames.len().saturating_sub(1 + ctl.selected_frame))
        {
            if let Some(body) = interp.program().get(&frame.func) {
                let (block, stmt) = frame.current_block_stmt();
                if let Some(p) = position(body, block, stmt) {
                    pos = p;
                }
            }
        }
    }
    if let Some(lines) = ctl.source_text.get(&pos.file) {
        if let Some(text) = lines.get((pos.line - 1) as usize) {
            let rendered = stop_line_text(ctl, pos.line, text);
            ctl.transcript.push(rendered);
        }
    }
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

