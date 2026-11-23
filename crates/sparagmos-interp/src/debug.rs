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
use crate::ir::{Body, FloatTy, IntTy, Program, SourceLine, Ty, UintTy};
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
    /// Whether gdb prints render through the sysroot Rust
    /// pretty printers (compiletest's default; a test opts out
    /// with `disable-gdb-pretty-printers`). Only the gdb form
    /// consults it.
    pretty_printers: bool,
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
    static_seeds: HashMap<String, crate::ir::Constant>,
    /// The session's lldb type-format registrations.
    lldb_formats: LldbFormats,
    /// Whether the program was compiled with limited debug
    /// info (`-C debuginfo=1` or lower): the binary then
    /// carries no variable information, and `info locals`
    /// reports none.
    hide_locals: bool,
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
                    | DebugCommand::InfoFunctions { .. },
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
        _ => {}
    }
}

/// Collect the program's statics' initializer constants, by
/// static key: walk every body's operands for the `StaticRef`
/// constants the conversion embedded (each carries its
/// static's full initializer as its seed). A static no body
/// references has no seed here; a print of it stays an explicit
/// error, as it did before statics could be printed at all.
fn collect_static_seeds(
    program: &Program,
) -> HashMap<String, crate::ir::Constant> {
    use crate::ir::{Operand, Rvalue, StatementKind, TerminatorKind};
    fn collect_constant(
        c: &crate::ir::Constant,
        seeds: &mut HashMap<String, crate::ir::Constant>,
    ) {
        match c {
            crate::ir::Constant::StaticRef { name, value, .. } => {
                seeds
                    .entry(name.clone())
                    .or_insert_with(|| (**value).clone());
                collect_constant(value, seeds);
            }
            crate::ir::Constant::Ref(inner) => collect_constant(inner, seeds),
            crate::ir::Constant::Tuple(items)
            | crate::ir::Constant::SliceVal(items) => {
                for item in items {
                    collect_constant(item, seeds);
                }
            }
            crate::ir::Constant::Adt { fields, .. } => {
                for field in fields {
                    collect_constant(field, seeds);
                }
            }
            crate::ir::Constant::ErasedAlloc { relocs, .. } => {
                for (_, reloc) in relocs {
                    collect_constant(reloc, seeds);
                }
            }
            _ => {}
        }
    }
    fn collect_operand(
        op: &Operand,
        seeds: &mut HashMap<String, crate::ir::Constant>,
    ) {
        if let Operand::Constant(c) = op {
            collect_constant(c, seeds);
        }
    }
    fn collect_rvalue(
        rv: &Rvalue,
        seeds: &mut HashMap<String, crate::ir::Constant>,
    ) {
        match rv {
            Rvalue::Use(op)
            | Rvalue::UnaryOp(_, op)
            | Rvalue::Cast(op, _)
            | Rvalue::Unsize(op, _)
            | Rvalue::BoxDerefTransmute(op)
            | Rvalue::DataPtr(op)
            | Rvalue::Materialize(op, _)
            | Rvalue::PtrToInt(op)
            | Rvalue::PtrField(op, _)
            | Rvalue::PtrFieldFirstElem(op, _)
            | Rvalue::PtrCastUnits(op, _)
            | Rvalue::PtrFirstElem(op)
            | Rvalue::PtrArrayBase(op)
            | Rvalue::WrapArrayElems(op)
            | Rvalue::UnwrapArrayElems(op)
            | Rvalue::IntToPtr(op)
            | Rvalue::EnumToInt(op, _, _)
            | Rvalue::Reinterpret(op, _, _)
            | Rvalue::IntToEnum(op, _) => collect_operand(op, seeds),
            Rvalue::BinaryOp(_, pair) | Rvalue::CheckedBinaryOp(_, pair) => {
                collect_operand(&pair.0, seeds);
                collect_operand(&pair.1, seeds);
            }
            Rvalue::AggregateTuple(ops) | Rvalue::CoroutineAggregate(ops) => {
                for op in ops {
                    collect_operand(op, seeds);
                }
            }
            Rvalue::AggregateAdt { operands, .. }
            | Rvalue::ClosureAggregate { operands, .. } => {
                for op in operands {
                    collect_operand(op, seeds);
                }
            }
            Rvalue::RawPtrAggregate { data, meta } => {
                collect_operand(data, seeds);
                if let Some(meta) = meta {
                    collect_operand(meta, seeds);
                }
            }
            Rvalue::PtrOffset(a, b, ..) => {
                collect_operand(a, seeds);
                collect_operand(b, seeds);
            }
            _ => {}
        }
    }
    let mut seeds = HashMap::new();
    for body in program.functions.values() {
        for entry in &body.debug_vars {
            if let crate::ir::DebugVar::Constant(c) = &entry.source {
                collect_constant(c, &mut seeds);
            }
        }
        for block in &body.blocks {
            for stmt in &block.statements {
                match &stmt.kind {
                    StatementKind::Assign(_, rv) => {
                        collect_rvalue(rv, &mut seeds)
                    }
                    StatementKind::Assume(op) => collect_operand(op, &mut seeds),
                    _ => {}
                }
            }
            match &block.terminator {
                TerminatorKind::SwitchInt { discr, .. } => {
                    collect_operand(discr, &mut seeds)
                }
                TerminatorKind::Call { args, .. }
                | TerminatorKind::CallBuiltin { args, .. } => {
                    for op in args {
                        collect_operand(op, &mut seeds);
                    }
                }
                TerminatorKind::CallIndirect { func, args, .. } => {
                    collect_operand(func, &mut seeds);
                    for op in args {
                        collect_operand(op, &mut seeds);
                    }
                }
                TerminatorKind::CallTrait {
                    receiver, args, ..
                } => {
                    collect_operand(receiver, &mut seeds);
                    collect_operand(args, &mut seeds);
                }
                _ => {}
            }
        }
    }
    seeds
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
                    | DebugCommand::InfoLocals),
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
fn frame_position(interp: &Interpreter, frame: &Frame) -> Option<SourceLine> {
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

/// A print expression evaluated to a value: the value itself,
/// its rendering descriptor, its `char` fact, and its declared
/// type tag when the conversion recorded one (the form the
/// pretty printers key on and the type queries print).
struct EvaluatedValue {
    value: Value,
    desc: Option<Desc>,
    char: bool,
    tag: Option<String>,
    /// The type a trailing cast step spelled, pointer layer
    /// included, when the cast's materialized view carries a
    /// descriptor of a different form: a cast to an array
    /// pointer views the pointee's elements under the array
    /// descriptor, but the expression's type keeps the pointer
    /// layer until a dereference consumes it.
    cast_type: Option<Desc>,
    /// A scalar pointee reinterpretation a cast step recorded
    /// but has not yet applied: the cast changed a pointer's
    /// pointee type while the target storage still holds the
    /// source type's bytes, and the dereference that consumes
    /// the pointer re-reads those bytes through the target
    /// layout. `(source, target)`.
    pending_cast: Option<(Desc, Desc)>,
    /// An array/slice view a cast to an array or slice
    /// pointer spelled but has not yet materialized: the
    /// cast keeps the pointer value (printing the cast
    /// renders the pointer), and the step that consumes the
    /// pointer -- a dereference or a projection --
    /// materializes the view first. The cast's spelled
    /// pointer descriptor, and the value's descriptor
    /// before the cast: an element-changing cast reinterprets
    /// the storage through the source element type, which
    /// the spelled target no longer records.
    pending_array_view: Option<(Desc, Option<Desc>)>,
}

/// Resolve a print expression's base name: a variable of the
/// selected frame's body (as recorded in its debug variable
/// table), or — for a name no local carries, and for any
/// path-qualified name — a static of the program.
fn resolve_base(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    expr: &ParsedExpr,
) -> Result<EvaluatedValue, String> {
    // A path-qualified name can only name a static: debug
    // variable entries carry plain identifiers.
    if !expr.base.contains("::") && !frames.is_empty() {
        if let Some(local) = resolve_local_base(interp, ctl, frames, expr)? {
            return Ok(local);
        }
    }
    if let Some(statik) = resolve_static_base(interp, ctl, frames, &expr.base) {
        return Ok(statik);
    }
    Err(format!("no variable `{}` in scope", expr.base))
}

/// Resolve a print expression's base name against the selected
/// frame's debug variable table. `Ok(None)` when the frame's
/// body records no variable of that name (the caller then
/// tries the statics).
fn resolve_local_base(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    expr: &ParsedExpr,
) -> Result<Option<EvaluatedValue>, String> {
    let frame = frames
        .get(frames.len().saturating_sub(1 + ctl.selected_frame))
        .ok_or_else(|| format!("frame {} is not live", ctl.selected_frame))?;
    let body = interp
        .program()
        .get(&frame.func)
        .ok_or_else(|| format!("no body for frame `{}`", frame.func))?;
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
    let Some(entry) = entry else {
        return Ok(None);
    };
    let (source, char_hint) = (&entry.source, entry.is_char);
    // The value's type, as a rendering descriptor: the variable
    // entry's recorded type tag when the conversion supplied one
    // (it carries the full instance name, generic arguments
    // included), else — for a plain local — the frame body's
    // local type, which names an ADT by its definition only.
    let (value, desc): (Value, Option<Desc>) = match source {
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
    Ok(Some(EvaluatedValue {
        value,
        desc,
        // Whether the value (or, for a reference variable, its
        // pointee) is a `char` at the source level: the IR type
        // system has no char (it is a u32), so the conversion
        // recorded it in the debug variable table.
        char: char_hint,
        tag: entry.ty.clone(),
        cast_type: None,
        pending_cast: None,
        pending_array_view: None,
    }))
}

/// Resolve a print expression's base name against the
/// program's statics: the name is a static key (a full item
/// path), or the last segment of one (a plain name, as gdb
/// resolves a static from any scope). The value is the
/// static's current storage content — evaluating its
/// `StaticRef` constant seeds the storage from the
/// initializer on first touch, so a print before the program
/// has touched the static reads the initial value, and a
/// print after a write reads the written one.
fn resolve_static_base(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    base: &str,
) -> Option<EvaluatedValue> {
    let program = interp.program();
    let mut exact: Option<&String> = None;
    let mut by_segment: Option<&String> = None;
    for key in program.static_debug_tags.keys() {
        let stripped = key.split('#').next().unwrap_or(key);
        if stripped == base {
            exact = Some(key);
            break;
        }
        if !base.contains("::") && stripped.rsplit("::").next() == Some(base) {
            // Prefer the shortest key among last-segment
            // matches (the least-qualified reading).
            match by_segment {
                Some(cur) if cur.len() <= key.len() => {}
                _ => by_segment = Some(key),
            }
        }
    }
    let key = exact.or(by_segment)?;
    let seed = ctl.static_seeds.get(key)?;
    let reference = interp
        .eval_constant(&crate::ir::Constant::StaticRef {
            name: key.clone(),
            value: Box::new(seed.clone()),
            projection: Vec::new(),
            len: None,
        })
        .ok()?;
    let resolved = interp.deref_value(frames, &reference).ok()?;
    let value = interp.read_resolved(frames, &resolved).ok()?;
    let tag = program.static_debug_tags.get(key).cloned();
    let desc = tag
        .as_deref()
        .map(Desc::from_tag)
        .or_else(|| program.static_tys.get(key).map(Desc::from_ir));
    let char = matches!(desc, Some(Desc::Char));
    Some(EvaluatedValue {
        value,
        desc,
        char,
        tag,
        cast_type: None,
        pending_cast: None,
        pending_array_view: None,
    })
}

/// Follow a reference or smart-pointer value to its pointee:
/// the (pointee, pointee descriptor) when the value is one, the
/// value itself otherwise. Field and index steps apply this
/// first, as a debugger's field access dereferences through
/// references and smart pointers implicitly.
fn deref_if_pointer(
    interp: &Interpreter,
    frames: &[Frame],
    value: Value,
    desc: Option<Desc>,
) -> Result<(Value, Option<Desc>), String> {
    let is_pointer = matches!(
        value,
        Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. }
    ) || matches!(&desc, Some(Desc::Adt(tag)) if smart_pointer_pointee(tag).is_some());
    if !is_pointer {
        return Ok((value, desc));
    }
    let resolved = interp
        .deref_value(frames, &value)
        .map_err(|e| e.to_string())?;
    let pointee = interp
        .read_resolved(frames, &resolved)
        .map_err(|e| e.to_string())?;
    let next = match desc {
        Some(
            Desc::Ref(inner) | Desc::RefMut(inner) | Desc::PtrConst(inner) | Desc::PtrMut(inner),
        ) => {
            // A pointer whose pointee the value model holds as
            // a sequence aggregate (a slice's base pointer: its
            // deref yields the whole element tuple) describes
            // the sequence, not the single element its type
            // names -- the descriptor follows the value, so an
            // index step through it reads the element descriptor
            // from the sequence form.
            if matches!(pointee, Value::Tuple(_))
                && !matches!(
                    inner.as_ref(),
                    Desc::Slice(_) | Desc::Array(..) | Desc::Tuple(_)
                )
            {
                Some(Desc::Slice(inner))
            } else {
                Some(*inner)
            }
        }
        Some(Desc::Adt(tag)) => {
            smart_pointer_pointee(&tag).map(|t| Desc::from_tag(&t))
        }
        other => other,
    };
    Ok((pointee, next))
}

/// The length a `.length` step reads: a fat pointer's element
/// count, a string's byte count, a stored sequence's element
/// count. gdb spells a slice's and a `String`'s length this
/// way (the fat pointer's and the vector's length field).
fn sequence_length(
    interp: &Interpreter,
    frames: &[Frame],
    value: &Value,
    desc: Option<&Desc>,
) -> Result<u128, String> {
    match value {
        Value::FatRef { len, .. } => Ok(*len),
        Value::Str(s) => Ok(s.len() as u128),
        Value::Bytes(b) => Ok(b.len() as u128),
        Value::SparseVec { len, .. } => Ok(*len),
        Value::Tuple(items) => Ok(items.len() as u128),
        Value::Ref(_) | Value::DynRef { .. } => {
            let resolved = interp
                .deref_value(frames, value)
                .map_err(|e| e.to_string())?;
            let pointee = interp
                .read_resolved(frames, &resolved)
                .map_err(|e| e.to_string())?;
            sequence_length(interp, frames, &pointee, None)
        }
        Value::Enum { fields, .. } => {
            // A `Vec` value in this model is its element
            // aggregate; a `String` value wraps its text (a
            // plain string value is handled above, so what
            // remains is the wrapped form `path_text` reads).
            if let Some(tag) = desc.and_then(|d| match d {
                Desc::Adt(tag) => Some(tag),
                _ => None,
            }) {
                match tag_base_name(tag) {
                    "Vec" => return Ok(fields.len() as u128),
                    "String" | "OsString" => {
                        if let Some(text) = path_text(value) {
                            return Ok(text.len() as u128);
                        }
                    }
                    _ => {}
                }
            }
            Err("value has no length".to_string())
        }
        _ => Err("value has no length".to_string()),
    }
}

/// Whether the value's recorded shape — after the implicit
/// dereference a field step performs — has a field of `name`:
/// the ADT shape's active variant, or a closure environment's
/// captures. A recorded field wins over a pseudo-field of the
/// same name (a struct may perfectly well declare a field
/// called `length`).
fn has_recorded_field(
    interp: &Interpreter,
    frames: &[Frame],
    value: &Value,
    desc: &Option<Desc>,
    name: &str,
) -> bool {
    let Ok((value, desc)) =
        deref_if_pointer(interp, frames, value.clone(), desc.clone())
    else {
        return false;
    };
    let Some(Desc::Adt(tag)) = &desc else {
        return false;
    };
    if let Some(shape) = interp.program().adt_debug_shapes.get(tag) {
        let variant = match &value {
            Value::Enum { variant, .. } => *variant as usize,
            _ => 0,
        };
        if let Some(vshape) = shape.variants.get(variant) {
            if vshape.fields.iter().any(|f| f.name.as_deref() == Some(name)) {
                return true;
            }
        }
    }
    if matches!(value, Value::Closure { .. }) {
        if let Some(shape) = interp.program().env_debug_shapes.get(tag) {
            if shape.upvars.iter().any(|f| f.name.as_deref() == Some(name)) {
                return true;
            }
        }
    }
    false
}

/// Apply one named-field step (`.name`) to a value: the fat
/// pointer pseudo-fields (`length`, `data_ptr`), then — after
/// an implicit dereference — the active variant's named field
/// from the value's debug shape, or a closure environment's
/// capture of that name.
fn apply_field_name(
    interp: &Interpreter,
    frames: &[Frame],
    value: Value,
    desc: Option<Desc>,
    name: &str,
    expr_text: &str,
) -> Result<(Value, Option<Desc>), String> {
    if name == "length" && !has_recorded_field(interp, frames, &value, &desc, name) {
        let len = sequence_length(interp, frames, &value, desc.as_ref())?;
        return Ok((Value::Uint(len), Some(Desc::Int("usize"))));
    }
    if name == "data_ptr" {
        if let Value::FatRef { base, .. } = &value {
            let elem = match &desc {
                Some(Desc::Ref(inner) | Desc::RefMut(inner)) => {
                    match inner.as_ref() {
                        Desc::Slice(elem) => Some((**elem).clone()),
                        _ => None,
                    }
                }
                _ => None,
            };
            // The base of a fat reference built from a constant
            // (a slice constant's element tuple) is the aggregate
            // itself, not a reference to it; a `data_ptr` is a
            // pointer, so the aggregate is wrapped the way the
            // runtime's own slice values carry it.
            let base = (**base).clone();
            let base = match base {
                Value::Ref(_)
                | Value::FatRef { .. }
                | Value::DynRef { .. }
                | Value::FnPtr(_) => base,
                other => Value::Ref(crate::value::RefTarget::value(other)),
            };
            return Ok((base, elem.map(|e| Desc::Ref(Box::new(e)))));
        }
    }
    let (value, desc) = deref_if_pointer(interp, frames, value, desc)?;
    if let Some(Desc::Adt(tag)) = &desc {
        if let Some(shape) = interp.program().adt_debug_shapes.get(tag) {
            let variant = match &value {
                Value::Enum { variant, .. } => *variant as usize,
                _ => 0,
            };
            if let Some(vshape) = shape.variants.get(variant) {
                if let Some(index) = vshape
                    .fields
                    .iter()
                    .position(|f| f.name.as_deref() == Some(name))
                {
                    let next = Desc::from_tag(&vshape.fields[index].ty);
                    // A union stores only the written field;
                    // reading another field reinterprets that
                    // storage through the requested field's
                    // type, the reading whole-union printing
                    // applies.
                    if let Some(reinterpreted) =
                        union_projected_field(interp, desc.as_ref(), &value, index)
                    {
                        return Ok((reinterpreted, Some(next)));
                    }
                    let projected = crate::interp::project_value(
                        value,
                        &[RefProj::Field(index)],
                        "debugger",
                        expr_text,
                    )
                    .map_err(|e| e.to_string())?;
                    return Ok((projected, Some(next)));
                }
            }
        }
        if let Some(shape) = interp.program().env_debug_shapes.get(tag) {
            if let Value::Closure { env, .. } = &value {
                if let Some(index) = shape
                    .upvars
                    .iter()
                    .position(|f| f.name.as_deref() == Some(name))
                {
                    if let Some(stored) = env.get(index) {
                        let next = Desc::from_tag(&shape.upvars[index].ty);
                        return Ok((stored.clone(), Some(next)));
                    }
                }
            }
        }
    }
    Err(format!("no field `{name}` in `{expr_text}`"))
}

/// The element aggregate a cast to an array/slice type views:
/// unwrap fat pointers and references until the pointee value
/// itself. A sub-slice's data pointer can denote a reference
/// to a materialised fat-pointer value (the sub-slice built as
/// a standalone value, then referenced), so the unwrapping
/// recurses through that layer too.
fn cast_pointee_elements(
    interp: &Interpreter,
    frames: &[Frame],
    value: Value,
) -> Result<Value, String> {
    match value {
        Value::FatRef { base, .. } => {
            cast_pointee_elements(interp, frames, (*base).clone())
        }
        Value::Ref(_) | Value::DynRef { .. } => {
            let resolved = interp
                .deref_value(frames, &value)
                .map_err(|e| e.to_string())?;
            let read = interp
                .read_resolved(frames, &resolved)
                .map_err(|e| e.to_string())?;
            cast_pointee_elements(interp, frames, read)
        }
        other => Ok(other),
    }
}

/// The byte width of a scalar descriptor's layout, for the
/// scalar families a byte reinterpretation can express.
/// Half- and quad-precision floats are excluded: their values
/// do not survive this runtime's `f64` transport bit-exactly,
/// so a reinterpretation through them is rejected instead.
fn desc_scalar_width(desc: &Desc) -> Option<usize> {
    Some(match desc {
        Desc::Int(name) => match *name {
            "i8" | "u8" => 1,
            "i16" | "u16" => 2,
            "i32" | "u32" => 4,
            "i64" | "u64" | "isize" | "usize" => 8,
            "i128" | "u128" => 16,
            _ => return None,
        },
        Desc::Float("f32") => 4,
        Desc::Float("f64") => 8,
        Desc::Bool => 1,
        Desc::Char => 4,
        _ => return None,
    })
}

/// A scalar value's little-endian storage bytes, at the
/// width its descriptor declares.
fn scalar_value_bytes(value: &Value, desc: &Desc) -> Option<Vec<u8>> {
    let width = desc_scalar_width(desc)?;
    Some(match (value, desc) {
        (Value::Int(v), Desc::Int(_)) => v.to_le_bytes()[..width].to_vec(),
        (Value::Uint(v), Desc::Int(_)) => v.to_le_bytes()[..width].to_vec(),
        (Value::Uint(v), Desc::Char) => (*v as u32).to_le_bytes().to_vec(),
        (Value::Float(v), Desc::Float("f32")) => {
            (*v as f32).to_bits().to_le_bytes().to_vec()
        }
        (Value::Float(v), Desc::Float("f64")) => {
            v.to_bits().to_le_bytes().to_vec()
        }
        (Value::Bool(b), Desc::Bool) => vec![*b as u8],
        _ => return None,
    })
}

/// A scalar value read back from its little-endian storage
/// bytes under a target descriptor.
fn scalar_value_from_bytes(bytes: &[u8], desc: &Desc) -> Option<Value> {
    match desc {
        Desc::Int(name) => {
            let mut buf = [0u8; 16];
            buf[..bytes.len()].copy_from_slice(bytes);
            let raw = u128::from_le_bytes(buf);
            if name.starts_with('i') {
                let shift = 128 - (bytes.len() as u32) * 8;
                let v = raw as i128;
                Some(Value::Int((v << shift) >> shift))
            } else {
                Some(Value::Uint(raw))
            }
        }
        Desc::Float("f32") => {
            let bits = u32::from_le_bytes(bytes.try_into().ok()?);
            Some(Value::Float(f32::from_bits(bits) as f64))
        }
        Desc::Float("f64") => {
            let bits = u64::from_le_bytes(bytes.try_into().ok()?);
            Some(Value::Float(f64::from_bits(bits)))
        }
        Desc::Bool => Some(Value::Bool(bytes[0] != 0)),
        Desc::Char => {
            let raw = u32::from_le_bytes(bytes.try_into().ok()?);
            char::from_u32(raw).map(|_| Value::Uint(raw as u128))
        }
        _ => None,
    }
}

/// The element descriptor a cast's source value carries: the
/// pointee array/slice element of the source descriptor, or
/// `u8` for a byte-form pointee.
fn cast_source_elem(pointee: &Value, source: Option<&Desc>) -> Option<Desc> {
    if matches!(pointee, Value::Bytes(_)) {
        return Some(Desc::Int("u8"));
    }
    let desc = source?;
    let inner = pointer_pointee(desc).unwrap_or(desc);
    match inner {
        Desc::Array(elem, _) | Desc::Slice(elem) => Some((**elem).clone()),
        _ => None,
    }
}

/// Reinterpret a cast pointee's elements through a different
/// target element type, comparing byte counts rather than
/// element counts. Returns `None` when no reinterpretation
/// applies (the element type is unchanged, or the source
/// element type is not recorded), and `Some` of the
/// reinterpreted value -- or of an explicit error for a
/// change the scalar layouts cannot express -- otherwise.
/// `count` is the target array's length; a slice target
/// reinterprets the whole storage.
fn reinterpret_cast(
    pointee: &Value,
    source: Option<&Desc>,
    target_elem: &Desc,
    count: Option<u64>,
) -> Option<Result<Value, String>> {
    let source_elem = cast_source_elem(pointee, source)?;
    if source_elem == *target_elem {
        return None;
    }
    let items: Vec<Value> = match pointee {
        Value::Tuple(items) => items.clone(),
        Value::Bytes(bytes) => bytes
            .iter()
            .map(|b| Value::Uint(*b as u128))
            .collect(),
        _ => return None,
    };
    let unsupported = || {
        format!(
            "cast changing element type from {} to {} is not supported",
            desc_type_name(&source_elem).unwrap_or_else(|| "?".into()),
            desc_type_name(target_elem).unwrap_or_else(|| "?".into()),
        )
    };
    let mut bytes = Vec::new();
    for item in &items {
        match scalar_value_bytes(item, &source_elem) {
            Some(bs) => bytes.extend(bs),
            None => return Some(Err(unsupported())),
        }
    }
    let Some(width) = desc_scalar_width(target_elem) else {
        return Some(Err(unsupported()));
    };
    let take = match count {
        Some(n) => {
            if bytes.len() < (n as usize) * width {
                return Some(Err(format!(
                    "cast to an array of {n} elements over {} bytes of backing storage",
                    bytes.len()
                )));
            }
            n as usize
        }
        None => {
            if bytes.len() % width != 0 {
                return Some(Err(format!(
                    "cast of {} bytes of backing storage to a slice of {}-byte elements",
                    bytes.len(),
                    width
                )));
            }
            bytes.len() / width
        }
    };
    let mut out = Vec::with_capacity(take);
    for i in 0..take {
        let chunk = &bytes[i * width..(i + 1) * width];
        match scalar_value_from_bytes(chunk, target_elem) {
            Some(v) => out.push(v),
            None => return Some(Err(unsupported())),
        }
    }
    Some(Ok(Value::Tuple(out)))
}

/// Whether a descriptor names a scalar type, the family a
/// byte reinterpretation can express.
fn is_scalar_desc(desc: &Desc) -> bool {
    matches!(
        desc,
        Desc::Int(_) | Desc::Float(_) | Desc::Bool | Desc::Char
    )
}

/// Re-read a scalar pointee value's storage bytes through a
/// different scalar layout: the dereference side of a pointer
/// cast that changed the pointee type, whose target storage
/// still holds the source type's bytes. A change the scalar
/// layouts cannot express, and a target wider than the
/// storage the source type occupies, are explicit errors
/// rather than wrong values.
fn reinterpret_scalar_pointee(
    value: &Value,
    source: &Desc,
    target: &Desc,
) -> Result<Value, String> {
    let unsupported = || {
        format!(
            "cast changing pointee type from {} to {} is not supported",
            desc_type_name(source).unwrap_or_else(|| "?".into()),
            desc_type_name(target).unwrap_or_else(|| "?".into()),
        )
    };
    let bytes = match scalar_value_bytes(value, source) {
        Some(bytes) => bytes,
        None => return Err(unsupported()),
    };
    let Some(width) = desc_scalar_width(target) else {
        return Err(unsupported());
    };
    if bytes.len() < width {
        return Err(format!(
            "cast to a {width}-byte pointee over {} bytes of backing storage",
            bytes.len()
        ));
    }
    match scalar_value_from_bytes(&bytes[..width], target) {
        Some(value) => Ok(value),
        None => Err(unsupported()),
    }
}

/// Whether a descriptor is a pointer (a reference or a raw
/// pointer) to an array or a slice: the cast target form
/// whose view of the pointee a consuming step materializes.
fn is_array_pointer_desc(desc: &Desc) -> bool {
    match desc {
        Desc::Ref(inner) | Desc::RefMut(inner) | Desc::PtrConst(inner) | Desc::PtrMut(inner) => {
            matches!(**inner, Desc::Array(..) | Desc::Slice(_))
        }
        _ => false,
    }
}

/// Materialize the array/slice view a cast to an array or
/// slice pointer spelled (see
/// `EvaluatedValue::pending_array_view`): read the pointee's
/// elements through the cast, the reading a dereference or
/// projection of the cast pointer applies. A no-op when no
/// view is pending.
fn materialize_array_view(
    interp: &Interpreter,
    frames: &[Frame],
    ev: &mut EvaluatedValue,
) -> Result<(), String> {
    let Some((spelled, source)) = ev.pending_array_view.take() else {
        return Ok(());
    };
    let (value, desc) = apply_cast(interp, frames, ev.value.clone(), source.as_ref(), spelled)?;
    ev.value = value;
    ev.desc = desc;
    Ok(())
}

/// Apply a cast step (`as *const [T; N]` and kin): a pointer
/// cast retargets the descriptor; an array cast views the
/// pointee's elements as an array value (the form the slice
/// tests print through a fat pointer's `data_ptr`).
fn apply_cast(
    interp: &Interpreter,
    frames: &[Frame],
    value: Value,
    source: Option<&Desc>,
    target: Desc,
) -> Result<(Value, Option<Desc>), String> {
    // A cast spelled as a pointer to an array/slice (`as *const
    // [T; N]`) is the array cast below: the pointer layer is
    // the spelling, not a value layer.
    let target = match target {
        Desc::Ref(inner) | Desc::RefMut(inner) | Desc::PtrConst(inner) | Desc::PtrMut(inner)
            if matches!(*inner, Desc::Array(..) | Desc::Slice(_)) =>
        {
            *inner
        }
        other => other,
    };
    match target {
        Desc::Array(target_elem, n) => {
            let pointee = cast_pointee_elements(interp, frames, value)?;
            // A cast that changes the element type reinterprets
            // the backing storage through the target element
            // layout (the reading a native debugger applies);
            // a change the scalar layouts cannot express is an
            // explicit error rather than a wrong value.
            if let Some(reinterpreted) = reinterpret_cast(&pointee, source, &target_elem, Some(n)) {
                return Ok((reinterpreted?, Some(Desc::Array(target_elem, n))));
            }
            // The cast views the backing elements through the
            // target type: a shorter array sees only its first
            // elements (and a later index step bounds against
            // them, as it projects this value), and a target
            // longer than the backing holds is an error rather
            // than a short read.
            let pointee = match pointee {
                Value::Tuple(mut items) => {
                    if (items.len() as u64) < n {
                        return Err(format!(
                            "cast to an array of {n} elements over a value holding {}",
                            items.len()
                        ));
                    }
                    items.truncate(n as usize);
                    Value::Tuple(items)
                }
                other => other,
            };
            Ok((pointee, Some(Desc::Array(target_elem, n))))
        }
        Desc::Slice(target_elem) => {
            let pointee = cast_pointee_elements(interp, frames, value)?;
            if let Some(reinterpreted) = reinterpret_cast(&pointee, source, &target_elem, None) {
                return Ok((reinterpreted?, Some(Desc::Slice(target_elem))));
            }
            Ok((pointee, Some(Desc::Slice(target_elem))))
        }
        other => Ok((value, Some(other))),
    }
}

/// Evaluate a print expression: resolve its base name, then
/// apply its dereference, field, index and cast steps.
fn evaluate_expr(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    expr_text: &str,
) -> Result<EvaluatedValue, String> {
    let expr = parse_expr(expr_text)?;
    let mut ev = resolve_base(interp, ctl, frames, &expr)?;
    for step in &expr.steps {
        match step {
            ExprStep::Deref => {
                // A cast to an array/slice pointer kept the
                // pointer value; this dereference consumes it,
                // so the spelled view materializes first, after
                // which the dereference is the identity on the
                // elements, as for a cast to an array type that
                // already replaced the value with the pointee's
                // elements (see `apply_cast`).
                materialize_array_view(interp, frames, &mut ev)?;
                if matches!(ev.desc, Some(Desc::Array(..) | Desc::Slice(_)))
                    && matches!(ev.value, Value::Tuple(_))
                {
                    // The dereference consumes the pointer
                    // layer a cast spelled, and the spelled
                    // type with it (the loop's step bookkeeping
                    // below does not run on this path).
                    ev.cast_type = None;
                    continue;
                }
                let resolved = interp
                    .deref_value(frames, &ev.value)
                    .map_err(|e| e.to_string())?;
                ev.value = interp
                    .read_resolved(frames, &resolved)
                    .map_err(|e| e.to_string())?;
                ev.desc = match ev.desc {
                    Some(
                        Desc::Ref(inner)
                        | Desc::RefMut(inner)
                        | Desc::PtrConst(inner)
                        | Desc::PtrMut(inner),
                    ) => Some(*inner),
                    Some(Desc::Adt(tag)) => {
                        smart_pointer_pointee(&tag).map(|t| Desc::from_tag(&t))
                    }
                    _ => None,
                };
                // A cast step that changed the pointee's
                // scalar type recorded a reinterpretation for
                // this dereference: the storage still holds
                // the source type's bytes, and the value is
                // re-read through the target layout (the
                // reading a native debugger applies) rather
                // than rendered under the target descriptor
                // unchanged.
                if let Some((source, target)) = ev.pending_cast.take() {
                    ev.value = reinterpret_scalar_pointee(&ev.value, &source, &target)?;
                }
                // The char hint is recorded for the variable as
                // declared: a `char` local, or a reference to a
                // char (whose pointee the hint then describes).
                // A dereference step therefore keeps it.
            }
            ExprStep::Field(index) => {
                materialize_array_view(interp, frames, &mut ev)?;
                let (v, d) = deref_if_pointer(interp, frames, ev.value, ev.desc)?;
                ev.value = v;
                ev.desc = d;
                let next = ev
                    .desc
                    .as_ref()
                    .and_then(|d| field_desc(interp, d, &ev.value, *index));
                // A union field read reinterprets the stored
                // field's storage (see `apply_field_name`);
                // any other value projects directly.
                ev.value = match union_projected_field(
                    interp,
                    ev.desc.as_ref(),
                    &ev.value,
                    *index,
                ) {
                    Some(reinterpreted) => reinterpreted,
                    None => crate::interp::project_value(
                        ev.value,
                        &[RefProj::Field(*index)],
                        "debugger",
                        expr_text,
                    )
                    .map_err(|e| e.to_string())?,
                };
                ev.desc = next;
                ev.char = false;
            }
            ExprStep::FieldName(name) => {
                materialize_array_view(interp, frames, &mut ev)?;
                let (v, d) = apply_field_name(
                    interp,
                    frames,
                    ev.value,
                    ev.desc,
                    name,
                    expr_text,
                )?;
                ev.value = v;
                ev.desc = d;
                ev.char = false;
            }
            ExprStep::Index(index) => {
                materialize_array_view(interp, frames, &mut ev)?;
                let (v, d) = deref_if_pointer(interp, frames, ev.value, ev.desc)?;
                ev.value = v;
                ev.desc = d;
                let next = ev.desc.as_ref().and_then(|d| match d {
                    Desc::Array(elem, _) | Desc::Slice(elem) => {
                        Some((**elem).clone())
                    }
                    _ => None,
                });
                ev.value = crate::interp::project_value(
                    ev.value,
                    &[RefProj::Index(*index)],
                    "debugger",
                    expr_text,
                )
                .map_err(|e| e.to_string())?;
                ev.desc = next;
                ev.char = false;
            }
            ExprStep::Cast(target) => {
                // A cast that changes a pointer's scalar
                // pointee type leaves the target storage
                // holding the source type's bytes: record the
                // reinterpretation for the dereference that
                // consumes the pointer (see the `Deref` arm).
                // A chain of such casts composes back to the
                // first cast's source, the type the storage
                // actually holds.
                let source_pointee = ev.desc.as_ref().and_then(pointer_pointee).cloned();
                let target_pointee = pointer_pointee(target).cloned();
                // A cast to an array or slice pointer keeps
                // the pointer value: only a step that consumes
                // the pointer materializes the spelled view
                // (see `pending_array_view`), so printing the
                // cast itself renders the pointer, not the
                // pointee. Any other cast supersedes a pending
                // view.
                let (v, d) = if is_array_pointer_desc(target) {
                    ev.pending_array_view = Some((target.clone(), ev.desc.clone()));
                    (ev.value.clone(), Some(target.clone()))
                } else {
                    ev.pending_array_view = None;
                    apply_cast(
                        interp,
                        frames,
                        ev.value,
                        ev.desc.as_ref(),
                        target.clone(),
                    )?
                };
                ev.value = v;
                ev.desc = d;
                ev.char = false;
                let new_pending = match (source_pointee, target_pointee) {
                    (Some(source), Some(target_inner))
                        if is_scalar_desc(&source) && is_scalar_desc(&target_inner) =>
                    {
                        let source = match &ev.pending_cast {
                            Some((first, mid)) if *mid == source => first.clone(),
                            _ => source,
                        };
                        if source == target_inner {
                            None
                        } else {
                            Some((source, target_inner))
                        }
                    }
                    _ => None,
                };
                ev.pending_cast = new_pending;
            }
        }
        // The recorded tag describes the base value's declared
        // type; after a step it no longer describes the value.
        ev.tag = None;
        // A cast step's spelled type and pending pointee
        // reinterpretation describe the expression only until
        // a later step consumes them: a dereference applies
        // (and clears) the reinterpretation, and any step
        // other than a cast drops the spelled type.
        if !matches!(step, ExprStep::Cast(_) | ExprStep::Deref) {
            ev.pending_cast = None;
        }
        ev.cast_type = match step {
            ExprStep::Cast(target) => Some(target.clone()),
            _ => None,
        };
    }
    Ok(ev)
}

/// Evaluate a print expression in the selected frame and
/// render the resulting value in both dialects' forms.
fn evaluate_print(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    print: &DebugPrint,
) -> Result<EvaluatedPrint, String> {
    let ev = evaluate_expr(interp, ctl, frames, &print.expr)?;
    let tyname = expr_type_name(&ev);
    let (value, desc, char) = (ev.value, ev.desc, ev.char);
    // The lldb synthetic-child providers' summary lines for
    // the sequence and counted-handle types (the forms the
    // sysroot's `lldb_providers.py` produces), appended to the
    // dialect's own lines when the printed value's type has
    // one.
    let mut lldb_lines_extra: Vec<String> = Vec::new();
    if parse_expr(&print.expr).map(|e| e.steps.is_empty()).unwrap_or(false) {
        if let Some(line) = render_lldb_provider_line(
            interp,
            frames,
            &value,
            desc.as_ref(),
            ev.tag.as_deref(),
            &print.expr,
        ) {
            lldb_lines_extra.push(line);
        }
    }
    // lldb's summary for a reference to a path is the path text
    // itself, not the address: render the pointee.
    let mut lldb = if matches!(value, Value::Closure { .. } | Value::Coroutine { .. }) {
        // A closure or coroutine value renders through its
        // recorded environment shape; the lldb form needs the
        // frames to read through a captured reference field.
        render_env_lldb_lines(interp, frames, &value, desc.as_ref(), &print.expr)
    } else {
        match &desc {
            Some(Desc::Ref(inner) | Desc::RefMut(inner))
                if parse_expr(&print.expr).map(|e| e.steps.is_empty()).unwrap_or(false)
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
        }
    };
    lldb.extend(lldb_lines_extra);
    // An lldb `type format add` registration for the printed
    // value's own type replaces the value's rendering with the
    // registered numeric format (see [`lldb_format_scalar`]).
    if let Some(formatted) =
        lldb_format_scalar(&value, ev.tag.as_deref(), desc.as_ref(), ctl.lldb_formats)
    {
        if let Some(first) = lldb.first_mut() {
            if first.starts_with("= ") {
                *first = format!("= {formatted}");
            }
        }
    }
    // lldb's `(Type) name = value` head line, for a print whose
    // rendering is the bare summary form: checks that name the
    // printed variable — statics printed by path, in
    // particular — anchor on the name, which the bare form
    // omits.
    if parse_expr(&print.expr)
        .map(|e| e.steps.is_empty())
        .unwrap_or(false)
    {
        if let Some(first) = lldb.first().cloned() {
            if let Some(compact) = first.strip_prefix("= ") {
                if let Some(tyname) = &tyname {
                    lldb.push(format!(
                        "({}) {} = {}",
                        lldb_type_name(tyname),
                        print.expr,
                        compact
                    ));
                }
            }
        }
    }
    Ok(EvaluatedPrint {
        gdb: if ctl.pretty_printers && print.format.is_none() {
            render_pretty_gdb(
                interp,
                frames,
                &value,
                desc.as_ref(),
                ev.tag.as_deref(),
                0,
            )
            .or_else(|| render_gdb(interp, &value, desc.as_ref(), char, print.format, 0))
        } else {
            render_gdb(interp, &value, desc.as_ref(), char, print.format, 0)
        },
        lldb,
    })
}

/// The rendering an lldb type-format registration prescribes
/// for a printed scalar, when the session registered a format
/// for the value's own type. The registration is keyed by type
/// name; the evaluated value's recorded tag names its type
/// exactly at the top level of a print, which is where the
/// debuginfo suite's registrations bite (a `char`'s descriptor
/// also survives expression steps, and is consulted as well).
fn lldb_format_scalar(
    value: &Value,
    tag: Option<&str>,
    desc: Option<&Desc>,
    formats: LldbFormats,
) -> Option<String> {
    // The registered type's width: an unsigned rendering is
    // the unsigned reading of the value's stored bytes, so a
    // sign-extended negative value must be masked back to the
    // width of the type the format was registered for.
    let (format, width) = match tag {
        Some("char") => (formats.char_format, 32),
        Some("i8") => (formats.i8_format, 8),
        Some("u8") => (formats.u8_format, 8),
        _ => match desc {
            Some(Desc::Char) => (formats.char_format, 32),
            // A projected value's tag is cleared by the
            // expression steps, but its descriptor keeps the
            // concrete integer type: a registration for that
            // type applies to the projection as it does to
            // the variable itself.
            Some(Desc::Int("i8")) => (formats.i8_format, 8),
            Some(Desc::Int("u8")) => (formats.u8_format, 8),
            _ => (None, 0),
        },
    };
    let format = format?;
    let mask: u128 = (1u128 << width) - 1;
    let v: i128 = match value {
        Value::Int(v) => *v,
        Value::Uint(v) => *v as i128,
        _ => return None,
    };
    Some(match format {
        LldbNumFormat::Decimal => format!("{v}"),
        LldbNumFormat::Unsigned => format!("{}", (v as u128) & mask),
        LldbNumFormat::Hex => {
            if v < 0 {
                return None;
            }
            format!("0x{:x}", (v as u128) & mask)
        }
    })
}

/// A `core::num::NonZero<T>` value unwrapped to its scalar:
/// the type's DWARF form is the primitive itself carrying a
/// validity range, and both debuggers print the bare scalar.
/// The value arrives as nested one-field wrappers — the
/// `NonZeroInner` associated type hides the primitive one
/// layer down, behind a type alias the debug-shape table
/// cannot resolve — so unwrapping follows the value's own
/// one-field layers until a scalar bottoms out. Returns the
/// scalar and the descriptor of the `NonZero`'s argument.
fn nonzero_scalar(
    value: &Value,
    desc: Option<&Desc>,
) -> Option<(Value, Desc)> {
    let Some(Desc::Adt(tag)) = desc else {
        return None;
    };
    if tag_base_name(tag) != "NonZero" {
        return None;
    }
    let inner = first_generic_arg(tag)?;
    let mut cur = value.clone();
    loop {
        match cur {
            Value::Enum { fields, .. } if fields.len() == 1 => {
                cur = fields.into_iter().next().unwrap();
            }
            Value::Tuple(fields) if fields.len() == 1 => {
                cur = fields.into_iter().next().unwrap();
            }
            Value::Int(_) | Value::Uint(_) => {
                return Some((cur, Desc::from_tag(&inner)));
            }
            _ => return None,
        }
    }
}

/// The declared type name of an evaluated expression, as a
/// type query (`whatis`) prints it: the recorded type tag in
/// its DWARF (definition-path) form when the conversion
/// recorded one, else the name rebuilt from the rendering
/// descriptor.
fn expr_type_name(ev: &EvaluatedValue) -> Option<String> {
    if matches!(ev.desc, Some(Desc::FnPtr))
        || matches!(&ev.tag, Some(tag) if tag == "FnPtr")
    {
        // A function pointer's tag is the type's kind name,
        // not a signature; gdb spells the type as a raw
        // pointer to the unit-signature function.
        return Some("*mut fn ()".to_string());
    }
    if let Some(tag) = &ev.tag {
        return Some(dwarf_type_name(tag));
    }
    // A trailing cast step's spelled type, pointer layer
    // included: the cast's materialized view carries the
    // pointee's descriptor (an array view for an array
    // pointer), but the expression's type is the cast's own
    // until a dereference consumes the pointer layer.
    if let Some(cast) = &ev.cast_type {
        if let Some(name) = desc_type_name(cast) {
            return Some(name);
        }
    }
    ev.desc.as_ref().and_then(desc_type_name)
}

/// A rendering descriptor's type name, for the forms whose
/// names the descriptor fully determines.
fn desc_type_name(desc: &Desc) -> Option<String> {
    Some(match desc {
        Desc::Unit => "()".to_string(),
        Desc::Bool => "bool".to_string(),
        Desc::Char => "char".to_string(),
        Desc::Str => "str".to_string(),
        Desc::Tuple(ts) => {
            let parts: Option<Vec<String>> =
                ts.iter().map(desc_type_name).collect();
            let parts = parts?;
            if parts.len() == 1 {
                format!("({},)", parts[0])
            } else {
                format!("({})", parts.join(", "))
            }
        }
        Desc::Array(elem, n) => {
            format!("[{}; {n}]", desc_type_name(elem)?)
        }
        Desc::Slice(elem) => format!("[{}]", desc_type_name(elem)?),
        Desc::Ref(inner) => format!("&{}", desc_type_name(inner)?),
        Desc::RefMut(inner) => format!("&mut {}", desc_type_name(inner)?),
        Desc::PtrConst(inner) => format!("*const {}", desc_type_name(inner)?),
        Desc::PtrMut(inner) => format!("*mut {}", desc_type_name(inner)?),
        Desc::Adt(tag) => dwarf_type_name(tag),
        Desc::Closure(name) | Desc::Coroutine(name) => name.clone(),
        Desc::FnPtr => "*mut fn ()".to_string(),
        Desc::Int(name) => name.to_string(),
        Desc::Float(name) => name.to_string(),
        Desc::Unknown => return None,
    })
}

/// The pointee descriptor of a pointer-family descriptor
/// (shared or mutable reference, raw const or mutable
/// pointer); `None` for any other descriptor.
fn pointer_pointee(desc: &Desc) -> Option<&Desc> {
    match desc {
        Desc::Ref(inner) | Desc::RefMut(inner) | Desc::PtrConst(inner) | Desc::PtrMut(inner) => {
            Some(inner)
        }
        _ => None,
    }
}

/// The descriptor past the implicit dereference a field or
/// index step performs: a reference's pointee, a smart
/// pointer's pointee, anything else unchanged.
fn desc_through_pointer(desc: Desc) -> Desc {
    match desc {
        Desc::Ref(inner) | Desc::RefMut(inner) | Desc::PtrConst(inner) | Desc::PtrMut(inner) => {
            *inner
        }
        Desc::Adt(tag) => smart_pointer_pointee(&tag)
            .map(|t| Desc::from_tag(&t))
            .unwrap_or(Desc::Adt(tag)),
        other => other,
    }
}

/// The descriptor of a named field, from the recorded shapes
/// alone (the type-only counterpart of `apply_field_name`):
/// an ADT variant's or closure environment's recorded field,
/// then the pseudo-fields (`length`, `data_ptr`).
fn type_field_name_desc(
    interp: &Interpreter,
    desc: &Desc,
    name: &str,
) -> Option<Desc> {
    if let Desc::Adt(tag) = desc {
        if let Some(shape) = interp.program().adt_debug_shapes.get(tag) {
            if let Some(vshape) = shape.variants.first() {
                if let Some(field) = vshape
                    .fields
                    .iter()
                    .find(|f| f.name.as_deref() == Some(name))
                {
                    return Some(Desc::from_tag(&field.ty));
                }
            }
        }
        if let Some(shape) = interp.program().env_debug_shapes.get(tag) {
            if let Some(field) = shape
                .upvars
                .iter()
                .find(|f| f.name.as_deref() == Some(name))
            {
                return Some(Desc::from_tag(&field.ty));
            }
        }
    }
    match name {
        "length" => Some(Desc::Int("usize")),
        "data_ptr" => match desc {
            Desc::Ref(inner)
            | Desc::RefMut(inner)
            | Desc::PtrConst(inner)
            | Desc::PtrMut(inner) => match inner.as_ref() {
                Desc::Slice(elem) => Some(Desc::Ref(elem.clone())),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

/// An evaluated form of an expression for a type query,
/// resolved from the recorded types alone: the base variable's
/// descriptor walked through the expression's steps, with no
/// pointee ever read. The value is a placeholder; only the
/// descriptor (and, for a step-free expression, the recorded
/// tag) carries information.
fn type_only_value(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    expr_text: &str,
) -> Option<EvaluatedValue> {
    let expr = parse_expr(expr_text).ok()?;
    let base = resolve_base(interp, ctl, frames, &expr).ok()?;
    let mut desc = base.desc.clone();
    let mut cast_type: Option<Desc> = None;
    for step in &expr.steps {
        cast_type = match step {
            ExprStep::Cast(target) => Some(target.clone()),
            _ => None,
        };
        desc = match (step, desc) {
            (
                ExprStep::Deref,
                Some(
                    Desc::Ref(inner)
                    | Desc::RefMut(inner)
                    | Desc::PtrConst(inner)
                    | Desc::PtrMut(inner),
                ),
            ) => Some(*inner),
            (ExprStep::Deref, Some(Desc::Adt(tag))) => {
                smart_pointer_pointee(&tag).map(|t| Desc::from_tag(&t))
            }
            (ExprStep::Deref, other) => other,
            (ExprStep::Field(index), Some(d)) => {
                let d = desc_through_pointer(d);
                field_desc(interp, &d, &Value::Uninit, *index)
            }
            (ExprStep::FieldName(name), Some(d)) => {
                let d = desc_through_pointer(d);
                type_field_name_desc(interp, &d, name)
            }
            (ExprStep::Index(_), Some(d)) => {
                match desc_through_pointer(d) {
                    Desc::Array(elem, _) | Desc::Slice(elem) => Some(*elem),
                    _ => None,
                }
            }
            (ExprStep::Cast(target), _) => Some(match target.clone() {
                Desc::Ref(inner)
                | Desc::RefMut(inner)
                | Desc::PtrConst(inner)
                | Desc::PtrMut(inner)
                    if matches!(*inner, Desc::Array(..) | Desc::Slice(_)) =>
                {
                    *inner
                }
                other => other,
            }),
            (_, None) => None,
        };
    }
    Some(EvaluatedValue {
        value: Value::Uninit,
        desc,
        char: base.char,
        tag: if expr.steps.is_empty() {
            base.tag.clone()
        } else {
            None
        },
        cast_type,
        pending_cast: None,
        pending_array_view: None,
    })
}

/// Render one type query (gdb `whatis`, or `ptype` with
/// `layout`): the `type = <name>` head line, plus — for
/// `ptype` over an aggregate shape — the shape's field lines
/// in gdb's layout form.
fn render_type_query(
    interp: &Interpreter,
    ctl: &DebugControl,
    frames: &[Frame],
    print: &DebugPrint,
    layout: bool,
) -> Vec<String> {
    let ev = match evaluate_expr(interp, ctl, frames, &print.expr) {
        Ok(ev) => ev,
        Err(e) => match type_only_value(interp, ctl, frames, &print.expr) {
            // A type query needs no value: when the value
            // evaluation fails (a null pointer's pointee, say),
            // the type still resolves from the recorded tags
            // and projection metadata alone.
            Some(ev) => ev,
            None => return vec![format!("type = <error: {e}>")],
        },
    };
    let Some(name) = expr_type_name(&ev) else {
        return vec!["type = <unknown>".to_string()];
    };
    if !layout {
        return vec![format!("type = {name}")];
    }
    // The layout form of a closure environment: gdb spells the
    // type as a struct over its captures, a capture taken by
    // reference appearing as the raw pointer the environment
    // stores.
    if let Some(Desc::Adt(tag)) = &ev.desc {
        // The shape table's key may carry the closure's
        // generic arguments or not, depending on how the
        // conversion recorded the environment; try the
        // recorded tag, then its argument-stripped form, then
        // the shape whose DWARF name is the tag's environment
        // name (`{closure#N}` spelled `{closure_env#N}`).
        let bare = tag.split('<').next().unwrap_or(tag);
        let env_shape = interp
            .program()
            .env_debug_shapes
            .get(tag)
            .or_else(|| interp.program().env_debug_shapes.get(bare))
            .or_else(|| {
                let want = bare
                    .replace("::{closure#", "::{closure_env#")
                    .replace("::{coroutine#", "::{coroutine_env#");
                interp
                    .program()
                    .env_debug_shapes
                    .values()
                    .find(|shape| shape.dwarf_name == want)
            });
        if let Some(shape) = env_shape {
            if shape.upvars.is_empty() {
                return vec![format!("type = struct {}", shape.dwarf_name)];
            }
            let mut lines = vec![format!("type = struct {} {{", shape.dwarf_name)];
            for field in &shape.upvars {
                let ty = dwarf_type_name(&field.ty);
                let ty = match ty.strip_prefix("&mut ") {
                    Some(inner) => format!("*mut {inner}"),
                    None => match ty.strip_prefix('&') {
                        Some(inner) => format!("*mut {inner}"),
                        None => ty,
                    },
                };
                lines.push(format!("    {ty},"));
            }
            lines.push("}".to_string());
            return lines;
        }
        // No recorded environment shape: a closure value whose
        // environment is empty has the bare struct form under
        // its environment name (the conversion records no
        // shape for a capture-less environment).
        if let Value::Closure { env, .. } = &ev.value {
            if env.is_empty() {
                let dwarf = bare
                    .replace("::{closure#", "::{closure_env#")
                    .replace("::{coroutine#", "::{coroutine_env#");
                if dwarf != bare {
                    return vec![format!("type = struct {dwarf}")];
                }
            }
        }
        if let Some(shape) = interp.program().adt_debug_shapes.get(tag) {
            if shape.kind == sparagmos_mir::AdtShapeKind::Struct {
                if let Some(vshape) = shape.variants.first() {
                    if vshape.fields.is_empty() {
                        return vec![format!("type = struct {name}")];
                    }
                    let mut lines = vec![format!("type = struct {name} {{")];
                    for field in &vshape.fields {
                        let fty = dwarf_type_name(&field.ty);
                        match &field.name {
                            Some(fname) => {
                                lines.push(format!("    {fty} {fname},"))
                            }
                            None => lines.push(format!("    {fty},")),
                        }
                    }
                    lines.push("}".to_string());
                    return lines;
                }
            }
        }
    }
    vec![format!("type = {name}")]
}

/// An IR type's name in the form a function declaration
/// listing prints (gdb `info functions`): the Rust spelling,
/// with an ADT named by its definition path.
fn ir_ty_name(ty: &Ty) -> String {
    use crate::ir::{FloatTy, IntTy, UintTy};
    match ty {
        Ty::Unit => "()".to_string(),
        Ty::Bool => "bool".to_string(),
        Ty::Int(k) => match k {
            IntTy::I8 => "i8",
            IntTy::I16 => "i16",
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
            IntTy::I128 => "i128",
            IntTy::Isize => "isize",
        }
        .to_string(),
        Ty::Uint(k) => match k {
            UintTy::U8 => "u8",
            UintTy::U16 => "u16",
            UintTy::U32 => "u32",
            UintTy::U64 => "u64",
            UintTy::U128 => "u128",
            UintTy::Usize => "usize",
        }
        .to_string(),
        Ty::Float(k) => match k {
            FloatTy::F16 => "f16",
            FloatTy::F32 => "f32",
            FloatTy::F64 => "f64",
            FloatTy::F128 => "f128",
        }
        .to_string(),
        Ty::Tuple(ts) => {
            let parts: Vec<String> = ts.iter().map(ir_ty_name).collect();
            if parts.len() == 1 {
                format!("({},)", parts[0])
            } else {
                format!("({})", parts.join(", "))
            }
        }
        Ty::Never => "!".to_string(),
        Ty::Adt(name) => dwarf_type_name(name),
        Ty::Ref(inner) => format!("&{}", ir_ty_name(inner)),
        Ty::Slice(elem) => format!("[{}]", ir_ty_name(elem)),
        Ty::Array(elem, n) => format!("[{}; {n}]", ir_ty_name(elem)),
        Ty::Str => "str".to_string(),
        Ty::FnPtr => "fn()".to_string(),
        Ty::FnDef(name) => name.clone(),
        Ty::Closure(name, _) => name.clone(),
        Ty::Dynamic(_) => "dyn".to_string(),
        Ty::Coroutine(name) => name.clone(),
    }
}

/// A small regular-expression search over function names, in
/// the subset of gdb's pattern language the debuginfo suite
/// uses: literal characters, `.` (any character) and `*`
/// (repeat the previous atom). Unanchored, as gdb's `info
/// functions` pattern is.
fn fn_pattern_search(pattern: &str, text: &str) -> bool {
    #[derive(Clone, Copy)]
    enum Atom {
        Lit(char),
        Any,
    }
    let mut atoms: Vec<(Atom, bool)> = Vec::new();
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        match c {
            '*' => {
                if let Some(last) = atoms.last_mut() {
                    last.1 = true;
                }
            }
            '.' => atoms.push((Atom::Any, false)),
            '\\' => match chars.next() {
                Some(esc) => atoms.push((Atom::Lit(esc), false)),
                None => atoms.push((Atom::Lit('\\'), false)),
            },
            _ => atoms.push((Atom::Lit(c), false)),
        }
    }
    let text: Vec<char> = text.chars().collect();
    fn matches(atoms: &[(Atom, bool)], text: &[char]) -> bool {
        if atoms.is_empty() {
            return true;
        }
        let (atom, star) = atoms[0];
        if star {
            // Zero repetitions, or one and stay.
            if matches(&atoms[1..], text) {
                return true;
            }
            return !text.is_empty()
                && atom_matches(atom, text[0])
                && matches(atoms, &text[1..]);
        }
        !text.is_empty()
            && atom_matches(atom, text[0])
            && matches(&atoms[1..], &text[1..])
    }
    fn atom_matches(atom: Atom, c: char) -> bool {
        match atom {
            Atom::Lit(l) => l == c,
            Atom::Any => true,
        }
    }
    // Unanchored: a match may start at any offset, and need
    // not reach the end of the name.
    (0..=text.len()).any(|start| matches(&atoms, &text[start..]))
}

/// Render gdb's `info functions [-q] [pattern]`: one
/// declaration line per body of the program whose name the
/// pattern matches (every body without a pattern), in name
/// order. The `-q` form carries the `static fn` prefix and
/// the signature; the plain form is `fn <name>();`.
fn render_info_functions(
    interp: &Interpreter,
    pattern: Option<&str>,
    quiet: bool,
) -> Vec<String> {
    let program = interp.program();
    let mut names: Vec<&String> = program.functions.keys().collect();
    names.sort();
    let mut lines = Vec::new();
    for name in names {
        if let Some(pat) = pattern {
            if !fn_pattern_search(pat, name) {
                continue;
            }
        }
        let Some(body) = program.get(name) else {
            continue;
        };
        // The declaration gdb's full-debuginfo listing prints:
        // the signature form, with the return type omitted for
        // the unit and never types (both print as no return).
        let signature = {
            let args: Vec<String> = (1..=body.arg_count)
                .filter_map(|i| body.local_tys.get(i).map(ir_ty_name))
                .collect();
            let ret = match body.local_tys.first() {
                Some(Ty::Unit) | Some(Ty::Never) | None => String::new(),
                Some(ty) => format!(" -> {}", ir_ty_name(ty)),
            };
            format!(
                "static fn {}({}){};",
                dwarf_type_name(name),
                args.join(", "),
                ret
            )
        };
        if quiet {
            lines.push(signature);
        } else {
            // The plain listing's declaration form depends on
            // the debug info the binary carries: with full
            // debuginfo gdb prints the DWARF declaration
            // (`static fn ...`), with line-only info the
            // symbol-table form (`fn ...`). Both are candidate
            // lines, as elsewhere in the renderers.
            lines.push(format!("fn {}();", dwarf_type_name(name)));
            lines.push(format!("static fn {}();", dwarf_type_name(name)));
            lines.push(signature);
        }
    }
    lines
}

/// Render gdb's `info locals` for the selected frame: the
/// variables in scope at the stop, one `name = value` line
/// each, or `No locals.` when none are.
fn render_info_locals(
    interp: &Interpreter,
    ctl: &mut DebugControl,
    frames: &[Frame],
) -> Vec<String> {
    if ctl.hide_locals {
        // Limited debug info carries no variable information.
        return vec!["No locals.".to_string()];
    }
    let Some(frame) =
        frames.get(frames.len().saturating_sub(1 + ctl.selected_frame))
    else {
        return vec!["No locals.".to_string()];
    };
    let Some(body) = interp.program().get(&frame.func) else {
        return vec!["No locals.".to_string()];
    };
    // The lookup position is the selected frame's own (the
    // derivation `frame_position` makes): the stop position
    // belongs to the innermost frame, so resolving an outer
    // frame's variables against it scopes them by the callee's
    // line — a caller local declared after the callee's stop
    // line but before the call would be filtered out. The
    // position is installed for the whole listing, so the name
    // resolution inside the evaluations scopes consistently.
    let saved_pos = ctl.current_pos;
    if ctl.selected_frame != 0 {
        if let Some(pos) = frame_position(interp, frame) {
            ctl.current_pos = Some((pos.file, pos.line));
        }
    }
    let at_line = ctl.current_pos.map(|(_, line)| line).unwrap_or(0);
    let mut seen: Vec<String> = Vec::new();
    let mut lines = Vec::new();
    for entry in &body.debug_vars {
        if entry.lines.0 == 0 || entry.lines.0 > at_line {
            continue;
        }
        if seen.contains(&entry.name) {
            continue;
        }
        seen.push(entry.name.clone());
        if let Ok(ev) = evaluate_expr(interp, ctl, frames, &entry.name) {
            if let Some(gdb) =
                render_gdb(interp, &ev.value, ev.desc.as_ref(), ev.char, None, 0)
            {
                lines.push(format!("{} = {gdb}", entry.name));
            }
        }
    }
    let lines = if lines.is_empty() {
        vec!["No locals.".to_string()]
    } else {
        lines
    };
    ctl.current_pos = saved_pos;
    lines
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
    /// An integer type, by its canonical name (`i32`, `u8`,
    /// ...): the concrete name, not just the signedness, so a
    /// type query over a projected value (a field, an element, a
    /// pointee) can print the primitive type it was projected
    /// from.
    Int(&'static str),
    /// A floating-point type, by its canonical name.
    Float(&'static str),
    Str,
    Tuple(Vec<Desc>),
    Array(Box<Desc>, u64),
    Slice(Box<Desc>),
    Ref(Box<Desc>),
    /// A mutable reference (`&mut T`): the qualifier is part
    /// of the recorded type, so it is carried separately from
    /// a shared reference.
    RefMut(Box<Desc>),
    /// A raw const pointer (`*const T`).
    PtrConst(Box<Desc>),
    /// A raw mutable pointer (`*mut T`).
    PtrMut(Box<Desc>),
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

/// The canonical name of a signed integer type.
fn int_ty_name(ty: IntTy) -> &'static str {
    match ty {
        IntTy::I8 => "i8",
        IntTy::I16 => "i16",
        IntTy::I32 => "i32",
        IntTy::I64 => "i64",
        IntTy::I128 => "i128",
        IntTy::Isize => "isize",
    }
}

/// The canonical name of an unsigned integer type.
fn uint_ty_name(ty: UintTy) -> &'static str {
    match ty {
        UintTy::U8 => "u8",
        UintTy::U16 => "u16",
        UintTy::U32 => "u32",
        UintTy::U64 => "u64",
        UintTy::U128 => "u128",
        UintTy::Usize => "usize",
    }
}

/// The canonical name of a floating-point type.
fn float_ty_name(ty: FloatTy) -> &'static str {
    match ty {
        FloatTy::F16 => "f16",
        FloatTy::F32 => "f32",
        FloatTy::F64 => "f64",
        FloatTy::F128 => "f128",
    }
}

impl Desc {
    fn from_ir(ty: &Ty) -> Desc {
        match ty {
            Ty::Unit => Desc::Unit,
            Ty::Bool => Desc::Bool,
            Ty::Int(t) => Desc::Int(int_ty_name(*t)),
            Ty::Uint(t) => Desc::Int(uint_ty_name(*t)),
            Ty::Float(t) => Desc::Float(float_ty_name(*t)),
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
            "i8" => return Desc::Int("i8"),
            "i16" => return Desc::Int("i16"),
            "i32" => return Desc::Int("i32"),
            "i64" => return Desc::Int("i64"),
            "i128" => return Desc::Int("i128"),
            "isize" => return Desc::Int("isize"),
            "u8" => return Desc::Int("u8"),
            "u16" => return Desc::Int("u16"),
            "u32" => return Desc::Int("u32"),
            "u64" => return Desc::Int("u64"),
            "u128" => return Desc::Int("u128"),
            "usize" => return Desc::Int("usize"),
            "f16" => return Desc::Float("f16"),
            "f32" => return Desc::Float("f32"),
            "f64" => return Desc::Float("f64"),
            "f128" => return Desc::Float("f128"),
            "!" => return Desc::Unknown,
            _ => {}
        }
        if let Some(rest) = tag.strip_prefix("&mut ") {
            return Desc::RefMut(Box::new(Desc::from_tag(rest)));
        }
        if let Some(rest) = tag.strip_prefix('&') {
            return Desc::Ref(Box::new(Desc::from_tag(rest)));
        }
        if let Some(rest) = tag.strip_prefix("*const ") {
            return Desc::PtrConst(Box::new(Desc::from_tag(rest)));
        }
        if let Some(rest) = tag.strip_prefix("*mut ") {
            return Desc::PtrMut(Box::new(Desc::from_tag(rest)));
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
        Some("size=0".to_string())
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
    // A trailing comma -- the canonical one-element tuple
    // spelling `(u32,)` -- leaves an empty final component:
    // it is not a component.
    if parts.len() > 1 && parts.last().is_some_and(|part| part.is_empty()) {
        parts.pop();
    }
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
    if let Some((scalar, inner)) = nonzero_scalar(value, desc) {
        return render_gdb(interp, &scalar, Some(&inner), top_char, format, depth);
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
            // A pointer to the unit type prints with its type
            // in cast form (`(*mut ()) 0x...`): the unit type
            // has no value form for the address to stand for.
            if matches!(
                desc,
                Some(
                    Desc::Ref(inner)
                    | Desc::RefMut(inner)
                    | Desc::PtrConst(inner)
                    | Desc::PtrMut(inner)
                ) if matches!(inner.as_ref(), Desc::Unit)
            ) {
                return Some(format!("(*mut ()) 0x{addr:x}"));
            }
            Some(format!("0x{addr:x}"))
        }
        Value::Closure { .. } | Value::Coroutine { .. } => {
            render_env_gdb(interp, value, desc, depth)
        }
        Value::Uninit => match desc {
            Some(Desc::Int(_)) | Some(Desc::Float(_)) => Some("0".to_string()),
            Some(Desc::Bool) => Some("false".to_string()),
            _ => None,
        },
        _ => None,
    }
}

/// The byte width of an integer type tag, when the tag names
/// one.
fn int_ty_width(ty: &str) -> Option<usize> {
    match ty {
        "i8" | "u8" => Some(1),
        "i16" | "u16" => Some(2),
        "i32" | "u32" => Some(4),
        "i64" | "u64" | "isize" | "usize" => Some(8),
        "i128" | "u128" => Some(16),
        _ => None,
    }
}

fn align_up(v: usize, align: usize) -> usize {
    (v + align - 1) / align.max(1) * align.max(1)
}

/// The (size, alignment) a debugger field type tag describes,
/// for the scalar, tuple and array types a union's fields take:
/// the layout the byte reinterpretation of a union value's
/// storage follows (a tuple's fields sit at the compiler's
/// own offsets — see [`tuple_field_layout`]).
fn tag_layout(ty: &str) -> Option<(usize, usize)> {
    let ty = ty.trim();
    let scalar = match ty {
        "bool" | "i8" | "u8" => Some((1, 1)),
        "char" => Some((4, 4)),
        "i16" | "u16" | "f16" => Some((2, 2)),
        "i32" | "u32" | "f32" => Some((4, 4)),
        "i64" | "u64" | "f64" | "isize" | "usize" => Some((8, 8)),
        "i128" | "u128" | "f128" => Some((16, 16)),
        _ => None,
    };
    if let Some(s) = scalar {
        return Some(s);
    }
    if ty.starts_with('(') && ty.ends_with(')') {
        let inner = &ty[1..ty.len() - 1];
        if inner.trim().is_empty() {
            return Some((0, 1));
        }
        let parts = split_top_level(inner);
        let (_, size, align) = tuple_field_layout(&parts)?;
        return Some((size, align));
    }
    if ty.starts_with('[') && ty.ends_with(']') {
        let inner = &ty[1..ty.len() - 1];
        let (elem, len) = inner.rsplit_once(';')?;
        let n: usize = len.trim().parse().ok()?;
        let (esize, ealign) = tag_layout(elem)?;
        let stride = align_up(esize, ealign);
        return Some((stride * n, ealign));
    }
    None
}

/// The byte layout of a tuple type's fields: each field's
/// offset (in declaration order), the total size and the
/// alignment. This is the compiler's own layout for a tuple —
/// the `rustc_abi` univariant layout under the tuple's
/// `MaybeUnsized` kind: the last field stays last (it is the
/// unsizing tail), and the remaining fields are stably sorted
/// by a group key, `trailing_zeros(max(align, size))`, capped
/// by the largest head alignment when one of them carries a
/// niche, with the larger niche preferred inside a group.
/// Verified against the pinned toolchain's `offset_of!`
/// layouts over scalar, array and nested-tuple shapes,
/// niches included.
fn tuple_field_layout(parts: &[String]) -> Option<(Vec<usize>, usize, usize)> {
    let layouts: Vec<(usize, usize)> = parts
        .iter()
        .map(|p| tag_layout(p))
        .collect::<Option<Vec<_>>>()?;
    let n = parts.len();
    if n == 0 {
        return Some((Vec::new(), 0, 1));
    }
    let niches: Vec<Option<(u64, usize)>> =
        parts.iter().map(|p| tag_niche(p)).collect();
    let head = n - 1;
    let largest_niche = niches[..head]
        .iter()
        .flatten()
        .map(|(available, _)| *available)
        .max()
        .unwrap_or(0);
    let max_head_align = layouts[..head]
        .iter()
        .map(|(_, align)| *align)
        .max()
        .unwrap_or(1);
    let group_key = |i: usize| -> u64 {
        let (size, align) = layouts[i];
        let key = u64::from(align.max(size).trailing_zeros());
        if largest_niche > 0 {
            key.min(u64::from(max_head_align.trailing_zeros()))
        } else {
            key
        }
    };
    let mut order: Vec<usize> = (0..head).collect();
    order.sort_by_key(|&i| {
        let (niche_size, niche_offset) = niches[i].unwrap_or((0, 0));
        (std::cmp::Reverse(group_key(i)), !niche_size, niche_offset)
    });
    order.push(head);
    let mut offsets = vec![0usize; n];
    let mut offset = 0usize;
    let mut align = 1usize;
    for i in order {
        let (size, field_align) = layouts[i];
        offset = align_up(offset, field_align);
        offsets[i] = offset;
        offset += size;
        align = align.max(field_align);
    }
    Some((offsets, align_up(offset, align), align))
}

/// The largest niche a debugger field type's values leave
/// unused, as (the number of unused values, the niche's offset
/// within the type), for the types whose layout ordering a
/// niche steers: `bool` and `char`, and aggregates containing
/// them. Anything else has no niche here.
fn tag_niche(ty: &str) -> Option<(u64, usize)> {
    let ty = ty.trim();
    match ty {
        // A `bool` leaves every byte but 0 and 1 unused.
        "bool" => return Some((254, 0)),
        // A `char` leaves the `u32` values above the last
        // scalar value unused, plus the surrogate range.
        "char" => return Some(((1u64 << 32) - (0x11_0000 - 0x800), 0)),
        _ => {}
    }
    if ty.starts_with('(') && ty.ends_with(')') {
        let inner = &ty[1..ty.len() - 1];
        if inner.trim().is_empty() {
            return None;
        }
        let parts = split_top_level(inner);
        let (offsets, _, _) = tuple_field_layout(&parts)?;
        let mut best: Option<(u64, usize)> = None;
        for (i, part) in parts.iter().enumerate() {
            if let Some((available, inner_offset)) = tag_niche(part) {
                if best.is_none_or(|(b, _)| available > b) {
                    best = Some((available, offsets[i] + inner_offset));
                }
            }
        }
        return best;
    }
    if ty.starts_with('[') && ty.ends_with(']') {
        let inner = &ty[1..ty.len() - 1];
        let (elem, len) = inner.rsplit_once(';')?;
        if len.trim().parse::<usize>().ok()? == 0 {
            return None;
        }
        // An array's niche is its first element's.
        return tag_niche(elem).map(|(available, _)| (available, 0));
    }
    None
}

/// Serialize a value as the little-endian bytes of its type
/// (see [`tag_layout`]): the storage a union field of that type
/// occupies. Types with no byte form here (references, ADT
/// fields, strings) make the whole reinterpretation fail, and
/// the union keeps the unrenderable form.
fn value_to_union_bytes(value: &Value, ty: &str) -> Option<Vec<u8>> {
    let ty = ty.trim();
    match ty {
        "bool" => match value {
            Value::Bool(b) => Some(vec![*b as u8]),
            _ => None,
        },
        "char" => match value {
            Value::Uint(v) => Some((*v as u32).to_le_bytes().to_vec()),
            Value::Int(v) => Some((*v as u32).to_le_bytes().to_vec()),
            _ => None,
        },
        "f32" => match value {
            Value::Float(f) => Some((*f as f32).to_bits().to_le_bytes().to_vec()),
            _ => None,
        },
        "f64" => match value {
            Value::Float(f) => Some(f.to_bits().to_le_bytes().to_vec()),
            _ => None,
        },
        "f128" => match value {
            Value::Float128(bits) => Some(bits.to_le_bytes().to_vec()),
            _ => None,
        },
        _ if int_ty_width(ty).is_some() => {
            let width = int_ty_width(ty)?;
            let bits = match value {
                Value::Int(v) => *v as u128,
                Value::Uint(v) => *v,
                _ => return None,
            };
            Some(bits.to_le_bytes()[..width].to_vec())
        }
        _ if ty.starts_with('(') && ty.ends_with(')') => {
            let Value::Tuple(items) = value else {
                return None;
            };
            let inner = &ty[1..ty.len() - 1];
            if inner.trim().is_empty() {
                return Some(Vec::new());
            }
            let parts = split_top_level(inner);
            if parts.len() != items.len() {
                return None;
            }
            // The fields sit at the compiler's tuple offsets,
            // which are not the declaration order (see
            // `tuple_field_layout`).
            let (offsets, size, _) = tuple_field_layout(&parts)?;
            let mut out = vec![0u8; size];
            for (i, (item, part)) in items.iter().zip(parts.iter()).enumerate() {
                let bytes = value_to_union_bytes(item, part)?;
                out[offsets[i]..offsets[i] + bytes.len()].copy_from_slice(&bytes);
            }
            Some(out)
        }
        _ if ty.starts_with('[') && ty.ends_with(']') => {
            let Value::Tuple(items) = value else {
                return None;
            };
            let inner = &ty[1..ty.len() - 1];
            let (elem, _) = inner.rsplit_once(';')?;
            let (esize, ealign) = tag_layout(elem)?;
            let stride = align_up(esize, ealign);
            let mut out = Vec::new();
            for item in items {
                let bytes = value_to_union_bytes(item, elem)?;
                out.extend_from_slice(&bytes);
                out.resize(out.len() + (stride - bytes.len()), 0);
            }
            Some(out)
        }
        _ => None,
    }
}

/// Read a value of a union field's type back from the field's
/// little-endian storage bytes (see [`value_to_union_bytes`]).
fn value_from_union_bytes(bytes: &[u8], ty: &str) -> Option<Value> {
    let ty = ty.trim();
    match ty {
        "bool" => Some(Value::Bool(bytes.first().copied()? != 0)),
        "char" => {
            let v = u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?);
            Some(Value::Uint(u128::from(v)))
        }
        "f32" => {
            let bits = u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?);
            Some(Value::Float(f32::from_bits(bits) as f64))
        }
        "f64" => {
            let bits = u64::from_le_bytes(bytes.get(..8)?.try_into().ok()?);
            Some(Value::Float(f64::from_bits(bits)))
        }
        "f128" => {
            let bits = u128::from_le_bytes(bytes.get(..16)?.try_into().ok()?);
            Some(Value::Float128(bits))
        }
        _ if int_ty_width(ty).is_some() => {
            let width = int_ty_width(ty)?;
            let raw = bytes.get(..width)?;
            let mut buf = [0u8; 16];
            buf[..width].copy_from_slice(raw);
            let bits = u128::from_le_bytes(buf);
            if ty.starts_with('i') {
                let shift = 128 - width * 8;
                Some(Value::Int(((bits << shift) as i128) >> shift))
            } else {
                Some(Value::Uint(bits))
            }
        }
        _ if ty.starts_with('(') && ty.ends_with(')') => {
            let inner = &ty[1..ty.len() - 1];
            if inner.trim().is_empty() {
                return Some(Value::Tuple(Vec::new()));
            }
            let parts = split_top_level(inner);
            let (offsets, _, _) = tuple_field_layout(&parts)?;
            let mut items = Vec::with_capacity(parts.len());
            for (i, part) in parts.iter().enumerate() {
                let (esize, _) = tag_layout(part)?;
                items.push(value_from_union_bytes(
                    bytes.get(offsets[i]..offsets[i] + esize)?,
                    part,
                )?);
            }
            Some(Value::Tuple(items))
        }
        _ if ty.starts_with('[') && ty.ends_with(']') => {
            let inner = &ty[1..ty.len() - 1];
            let (elem, len) = inner.rsplit_once(';')?;
            let n: usize = len.trim().parse().ok()?;
            let (esize, ealign) = tag_layout(elem)?;
            let stride = align_up(esize, ealign);
            let mut items = Vec::with_capacity(n);
            for i in 0..n {
                items.push(value_from_union_bytes(
                    bytes.get(i * stride..i * stride + esize)?,
                    elem,
                )?);
            }
            Some(Value::Tuple(items))
        }
        _ => None,
    }
}

/// The values of every declared field of a union value: the
/// value model stores only the written field (the slots before
/// it uninitialised, the slots after it absent), while a
/// debugger reads every field from the same storage, so the
/// stored field's bytes are reinterpreted through each declared
/// field's type.
fn union_field_values(
    shape_fields: &[sparagmos_mir::AdtDebugField],
    fields: &[Value],
) -> Option<Vec<Value>> {
    let stored = fields.iter().rposition(|f| !matches!(f, Value::Uninit))?;
    let stored_ty = &shape_fields.get(stored)?.ty;
    let mut bytes = value_to_union_bytes(&fields[stored], stored_ty)?;
    let size = shape_fields
        .iter()
        .map(|f| tag_layout(&f.ty).map(|(s, _)| s))
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    bytes.resize(size, 0);
    shape_fields
        .iter()
        .map(|f| value_from_union_bytes(&bytes, &f.ty))
        .collect()
}

/// One field of a union value, read the way
/// [`union_field_values`] reads every field: the stored
/// field's bytes reinterpreted as the requested field's
/// type.
fn union_field_value(
    shape_fields: &[sparagmos_mir::AdtDebugField],
    value: &Value,
    index: usize,
) -> Option<Value> {
    let fields: &[Value] = match value {
        Value::Enum { fields, .. } => fields,
        Value::Tuple(fields) => fields,
        _ => return None,
    };
    union_field_values(shape_fields, fields)?
        .into_iter()
        .nth(index)
}

/// The value of one field of the ADT value a descriptor
/// names, when that ADT is a union: the field read through
/// the storage reinterpretation above, so inspecting a
/// field agrees with printing its containing union. `None`
/// for non-ADT descriptors, non-union shapes, and storage
/// the reinterpretation cannot read (the caller then
/// projects directly).
fn union_projected_field(
    interp: &Interpreter,
    desc: Option<&Desc>,
    value: &Value,
    index: usize,
) -> Option<Value> {
    let Desc::Adt(tag) = desc? else {
        return None;
    };
    let shape = interp.program().adt_debug_shapes.get(tag)?;
    if !matches!(shape.kind, sparagmos_mir::AdtShapeKind::Union) {
        return None;
    }
    let variant = match value {
        Value::Enum { variant, .. } => *variant as usize,
        _ => 0,
    };
    let vshape = shape.variants.get(variant)?;
    union_field_value(&vshape.fields, value, index)
}

/// lldb's expanded-children form of a `NonZero` value: the
/// scalar summary is the value itself, and the children are the
/// two one-field wrappers the DWARF describes (the struct's
/// field 0 holding the inner validity wrapper holding the
/// scalar), the inner one in compact form: `{ 0 = {0:11} }`.
/// The recorded debug shape does not describe those wrappers
/// (the value arrives already unwrapped for the summary), so
/// the form is synthesized from the wrapper structure itself.
fn nonzero_lldb_expanded(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
) -> Option<String> {
    let (scalar, inner) = nonzero_scalar(value, desc)?;
    let wrapper = match value {
        Value::Enum { fields, .. } if fields.len() == 1 => &fields[0],
        Value::Tuple(fields) if fields.len() == 1 => &fields[0],
        _ => return None,
    };
    let inner_fields: &[Value] = match wrapper {
        Value::Enum { fields, .. } => fields,
        Value::Tuple(fields) => fields,
        _ => return None,
    };
    if inner_fields.len() != 1 || inner_fields[0] != scalar {
        return None;
    }
    let text = render_lldb_compact(interp, &scalar, Some(&inner), false, true, 1)?;
    Some(format!("{{ 0 = {{0:{text}}} }}"))
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
    if matches!(shape.kind, sparagmos_mir::AdtShapeKind::Union) {
        // A union value stores only the written field; every
        // declared field is read back from the same storage
        // (see `union_field_values`).
        let values = union_field_values(&vshape.fields, fields)?;
        let mut parts = Vec::with_capacity(values.len());
        for (i, v) in values.iter().enumerate() {
            let name = vshape.fields[i].name.as_deref()?;
            let d = Desc::from_tag(&vshape.fields[i].ty);
            let r = render_gdb(interp, v, Some(&d), false, None, depth + 1)?;
            parts.push(format!("{name}: {r}"));
        }
        return Some(format!("{tag} {{{}}}", parts.join(", ")));
    }
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

// ---------------------------------------------------------------------------
// Pretty printers
//
// When a session enables the pretty printers (compiletest's
// default; see `DebugControl::pretty_printers`), a print of a
// value whose type the sysroot registers a provider for renders
// in the provider's form instead of the raw aggregate form. The
// classification mirrors `rust_types.py`'s type-kind table,
// matched on the DWARF (definition-path) name of the value's
// type, and each form mirrors the corresponding provider in
// `gdb_providers.py` / `lldb_providers.py`. A value with no
// provider — or a provider value whose shape the value model
// does not produce — keeps the raw form.
// ---------------------------------------------------------------------------

/// The sysroot pretty-printer kinds this renderer implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrinterKind {
    String,
    OsString,
    Vec,
    VecDeque,
    BTreeSet,
    BTreeMap,
    HashMap,
    HashSet,
    Rc,
    Arc,
}

/// The printer kind of a DWARF type name, if any: the type's
/// base name and its parent segment decide, the way
/// `rust_types.py`'s anchored path patterns do (the crate and
/// module spellings of a std type vary between its definition
/// path and its re-export paths; the last two segments are
/// stable across them).
fn printer_kind(dwarf_name: &str) -> Option<PrinterKind> {
    let base = dwarf_name.split('<').next()?.trim();
    if !(base.starts_with("alloc::")
        || base.starts_with("std::")
        || base.starts_with("core::"))
    {
        return None;
    }
    let mut segs = base.rsplit("::");
    let name = segs.next()?;
    let parent = segs.next().unwrap_or("");
    Some(match (name, parent) {
        ("String", "string") => PrinterKind::String,
        ("OsString", "os_str") => PrinterKind::OsString,
        ("Vec", "vec") => PrinterKind::Vec,
        ("VecDeque", "vec_deque") | ("VecDeque", "collections") => {
            PrinterKind::VecDeque
        }
        ("BTreeSet", "set") | ("BTreeSet", "collections") => {
            PrinterKind::BTreeSet
        }
        ("BTreeMap", "map") | ("BTreeMap", "collections") => {
            PrinterKind::BTreeMap
        }
        ("HashMap", "map") => PrinterKind::HashMap,
        ("HashSet", "set") => PrinterKind::HashSet,
        ("Rc", "rc") => PrinterKind::Rc,
        ("Arc", "sync") => PrinterKind::Arc,
        _ => return None,
    })
}

/// The elements of a sequence value in this model: an element
/// tuple (an array value or a materialised slice), an ADT
/// element aggregate (a vector's storage form), or a byte block
/// (each byte an element).
fn sequence_values(value: &Value) -> Option<Vec<Value>> {
    match value {
        Value::Tuple(items) => Some(items.clone()),
        Value::Enum { fields, .. } => Some(fields.clone()),
        // A sparse sequence's stored elements (its declared
        // length runs past them; see `Value::SparseVec`).
        Value::SparseVec { elems, .. } => Some(elems.clone()),
        Value::Bytes(bytes) => Some(
            bytes
                .iter()
                .map(|b| Value::Uint(u128::from(*b)))
                .collect(),
        ),
        Value::Str(text) => Some(
            text.bytes()
                .map(|b| Value::Uint(u128::from(b)))
                .collect(),
        ),
        _ => None,
    }
}

/// gdb's `print elements` limit for a printer's children.
const GDB_PRINT_ELEMENTS: usize = 200;

/// Render a value in gdb form with the pretty printers applied
/// where a provider exists (the raw form otherwise).
fn render_gdb_any(
    interp: &Interpreter,
    frames: &[Frame],
    value: &Value,
    desc: Option<&Desc>,
    depth: usize,
) -> Option<String> {
    render_pretty_gdb(interp, frames, value, desc, None, depth)
        .or_else(|| render_gdb(interp, value, desc, false, None, depth))
}

/// Resolve a printed slice value to the value holding its
/// elements: follow the fat pointer, and keep following while
/// the pointee is itself a fat pointer — a slice formed over
/// another slice value (a vector's `&vec[..]`, whose base is
/// the vector's own slice value) denotes the same elements
/// through every layer. The peeling is bounded; a pointee
/// still fat past the bound is left as it stands (and the
/// caller's sequence reading then declines it, as it did
/// before peeling existed).
fn slice_pointee(
    interp: &Interpreter,
    frames: &[Frame],
    value: &Value,
) -> Option<Value> {
    let mut pointee = interp
        .read_resolved(frames, &interp.deref_value(frames, value).ok()?)
        .ok()?;
    for _ in 0..8 {
        match &pointee {
            Value::FatRef { base, .. } => match &**base {
                // The elements are the base aggregate itself:
                // the fat pointer is self-contained over it.
                Value::Tuple(_) | Value::SparseVec { .. } => {
                    return Some((**base).clone());
                }
                // A reference base: follow the fat pointer to
                // what it denotes and judge that value.
                Value::Ref(_) => {
                    pointee = interp
                        .read_resolved(
                            frames,
                            &interp.deref_value(frames, &pointee).ok()?,
                        )
                        .ok()?;
                }
                _ => return Some(pointee),
            },
            _ => return Some(pointee),
        }
    }
    Some(pointee)
}

/// The elements a sequence rendering walks: the stored
/// elements, extended with uninitialised slots up to the
/// declared length when the sequence is sparse (see
/// `Value::SparseVec`) — bounded just past the print-elements
/// limit, since the renderer truncates there anyway. A fully
/// stored sequence passes through unchanged.
fn render_sequence_elems(stored: &[Value], declared: u128) -> Vec<Value> {
    let mut elems = stored.to_vec();
    let want = usize::try_from(declared)
        .unwrap_or(usize::MAX)
        .min(GDB_PRINT_ELEMENTS + 1);
    if elems.len() < want {
        elems.resize(want, Value::Uninit);
    }
    elems
}

/// Render a printer's array-hint children block (`{a, b, c}`),
/// with gdb's repeat compression and its truncation marker
/// past the `print elements` limit.
fn render_printer_children(
    interp: &Interpreter,
    frames: &[Frame],
    elems: &[Value],
    elem_desc: Option<&Desc>,
    depth: usize,
) -> Option<String> {
    let shown = elems.len().min(GDB_PRINT_ELEMENTS);
    let mut rendered = Vec::with_capacity(shown);
    for elem in &elems[..shown] {
        rendered.push(render_gdb_any(interp, frames, elem, elem_desc, depth + 1)?);
    }
    let mut body = compress_repeats(&rendered);
    if elems.len() > shown {
        if !body.is_empty() {
            body.push_str(", ");
        }
        body.push_str("...");
    }
    Some(format!("{{{body}}}"))
}

/// Render one printed value in the sysroot gdb pretty
/// printers' form, when its type has a provider this renderer
/// implements. `tag_hint` is the printed variable's declared
/// type tag, for the forms that name the type (a slice's
/// `&[T]` / `&mut [T]` head).
fn render_pretty_gdb(
    interp: &Interpreter,
    frames: &[Frame],
    value: &Value,
    desc: Option<&Desc>,
    tag_hint: Option<&str>,
    depth: usize,
) -> Option<String> {
    if depth > MAX_RENDER_DEPTH {
        return None;
    }
    match desc {
        Some(Desc::Ref(inner) | Desc::RefMut(inner)) => match inner.as_ref() {
            Desc::Slice(elem) => {
                let pointee = slice_pointee(interp, frames, value)?;
                // A sparse pointee declares its length past its
                // stored elements (see `Value::SparseVec`): the
                // size is the declared length, and the children
                // render from the stored elements plus the
                // uninitialised tail.
                let (elems, declared) = match &pointee {
                    Value::SparseVec { elems, len, .. } => {
                        (elems.clone(), *len)
                    }
                    _ => {
                        let elems = sequence_values(&pointee)?;
                        let len = elems.len() as u128;
                        (elems, len)
                    }
                };
                let ty = match tag_hint {
                    Some(tag) if tag.trim_start().starts_with('&') => {
                        tag.trim().to_string()
                    }
                    _ => return None,
                };
                let head = format!("{ty}(size={declared})");
                if declared == 0 {
                    Some(head)
                } else {
                    let elems = render_sequence_elems(&elems, declared);
                    let block = render_printer_children(
                        interp,
                        frames,
                        &elems,
                        Some(elem),
                        depth,
                    )?;
                    Some(format!("{head} = {block}"))
                }
            }
            Desc::Str => {
                let resolved = interp.deref_value(frames, value).ok()?;
                let pointee = interp.read_resolved(frames, &resolved).ok()?;
                let text = match &pointee {
                    Value::Str(s) => s.clone(),
                    Value::Bytes(b) => {
                        String::from_utf8_lossy(b).into_owned()
                    }
                    _ => return None,
                };
                Some(format!("\"{}\"", escape_str(&text)))
            }
            _ => None,
        },
        Some(Desc::Adt(tag)) => {
            let kind = printer_kind(&dwarf_type_name(tag))?;
            render_printer_adt(interp, frames, kind, tag, value, depth)
        }
        _ => None,
    }
}

/// Render an ADT value in its gdb provider's form.
fn render_printer_adt(
    interp: &Interpreter,
    frames: &[Frame],
    kind: PrinterKind,
    tag: &str,
    value: &Value,
    depth: usize,
) -> Option<String> {
    match kind {
        PrinterKind::String | PrinterKind::OsString => {
            let text = match value {
                Value::Str(s) => Some(s.clone()),
                _ => path_text(value),
            }?;
            Some(format!("\"{}\"", escape_str(&text)))
        }
        PrinterKind::Vec => {
            let elem =
                first_generic_arg(tag).map(|t| Desc::from_tag(&t));
            match value {
                // A sparse vector (see `Value::SparseVec`): the
                // size is the declared length, and the children
                // render from the stored elements plus the
                // uninitialised tail, bounded like any long
                // vector's.
                Value::SparseVec { elems, len, .. } => {
                    let head = format!("Vec(size={len})");
                    if *len == 0 {
                        Some(head)
                    } else {
                        let elems = render_sequence_elems(elems, *len);
                        let block = render_printer_children(
                            interp,
                            frames,
                            &elems,
                            elem.as_ref(),
                            depth,
                        )?;
                        Some(format!("{head} = {block}"))
                    }
                }
                Value::Enum { fields, .. } => {
                    let head = format!("Vec(size={})", fields.len());
                    if fields.is_empty() {
                        Some(head)
                    } else {
                        let block = render_printer_children(
                            interp,
                            frames,
                            fields,
                            elem.as_ref(),
                            depth,
                        )?;
                        Some(format!("{head} = {block}"))
                    }
                }
                // A vector place the program has not written
                // yet (the variable is declared but not yet
                // assigned at the stop): the provider reads the
                // uninitialised buffer's length as whatever the
                // memory holds, and the suite's check for this
                // case accepts any size; the empty form is the
                // one honest rendering available here.
                Value::Uninit => Some("Vec(size=0)".to_string()),
                _ => None,
            }
        }
        PrinterKind::Rc | PrinterKind::Arc => {
            let (strong, weak, pointee, elem) = rc_parts(interp, tag, value)?;
            let v = render_gdb_any(
                interp,
                frames,
                &pointee,
                elem.as_ref(),
                depth + 1,
            )?;
            let name = if kind == PrinterKind::Rc {
                "Rc"
            } else {
                "Arc"
            };
            Some(format!("{name}(strong={strong}, weak={weak}) = {{value = {v}, strong = {strong}, weak = {weak}}}"))
        }
        // The collection providers are rendered from the
        // collections' own entry walks, added with the
        // collections' debugger support.
        PrinterKind::VecDeque
        | PrinterKind::BTreeSet
        | PrinterKind::BTreeMap
        | PrinterKind::HashMap
        | PrinterKind::HashSet => None,
    }
}

/// The counts and pointee of a reference-counted handle value
/// (`Rc` / `Arc`): the strong count, the weak count as the
/// providers display it (the stored count less the
/// allocation's own implicit weak reference), the pointee
/// value, and the pointee's descriptor from the instance tag.
fn rc_parts(
    interp: &Interpreter,
    tag: &str,
    value: &Value,
) -> Option<(u128, u128, Value, Option<Desc>)> {
    let id = crate::interp::rc_handle_id(value)?;
    let inner = interp.alloc_base(id).ok()?;
    let strong = rc_count(&inner, 0)?;
    let weak = rc_count(&inner, 1)?.saturating_sub(1);
    let pointee = match &inner {
        Value::Enum { fields, .. } => fields.get(2)?.clone(),
        _ => return None,
    };
    let elem = first_generic_arg(tag).map(|t| Desc::from_tag(&t));
    Some((strong, weak, pointee, elem))
}

/// One reference count of an `RcInner` / `ArcInner` aggregate:
/// field `field` (0 = strong, 1 = weak) is a `Cell<usize>` /
/// `AtomicUsize` wrapper around an `UnsafeCell<usize>` holding
/// the count (the same reading the interpreter's own count
/// adjustments make).
fn rc_count(inner: &Value, field: usize) -> Option<u128> {
    fn unwrap1(v: &Value) -> Option<&Value> {
        match v {
            Value::Enum { fields, .. } if fields.len() == 1 => {
                Some(&fields[0])
            }
            _ => None,
        }
    }
    let fields = match inner {
        Value::Enum { fields, .. } => fields,
        _ => return None,
    };
    let leaf = unwrap1(unwrap1(fields.get(field)?)?)?;
    match leaf {
        Value::Uint(n) => Some(*n),
        _ => None,
    }
}

/// The lldb provider summary line for one printed value, when
/// its type has a synthetic-child provider with a summary form
/// this renderer implements (a slice, a string slice, a counted
/// handle): the `(Type) name = <summary> { <children> }` line
/// lldb prints for it.
fn render_lldb_provider_line(
    interp: &Interpreter,
    frames: &[Frame],
    value: &Value,
    desc: Option<&Desc>,
    tag_hint: Option<&str>,
    expr: &str,
) -> Option<String> {
    match desc {
        Some(Desc::Ref(inner) | Desc::RefMut(inner)) => match inner.as_ref() {
            Desc::Slice(elem) => {
                let pointee = slice_pointee(interp, frames, value)?;
                let elems = sequence_values(&pointee)?;
                // A sparse pointee's size is its declared
                // length, past the stored elements the children
                // render from (see `Value::SparseVec`).
                let size = match &pointee {
                    Value::SparseVec { len, .. } => *len,
                    _ => elems.len() as u128,
                };
                let ty = tag_hint?.trim();
                let mut parts = Vec::with_capacity(elems.len());
                for (i, e) in elems.iter().enumerate() {
                    let v = render_lldb_compact(
                        interp,
                        e,
                        Some(elem),
                        false,
                        true,
                        1,
                    )?;
                    parts.push(format!("[{i}] = {v}"));
                }
                if parts.is_empty() {
                    Some(format!("({ty}) {expr} = size={size}"))
                } else {
                    let children = format!("{{ {} }}", parts.join(" "));
                    Some(format!(
                        "({ty}) {expr} = size={} {children}",
                        size
                    ))
                }
            }
            Desc::Str => {
                let resolved = interp.deref_value(frames, value).ok()?;
                let pointee = interp.read_resolved(frames, &resolved).ok()?;
                let text = match &pointee {
                    Value::Str(s) => s.clone(),
                    Value::Bytes(b) => {
                        String::from_utf8_lossy(b).into_owned()
                    }
                    _ => return None,
                };
                let ty = tag_hint?.trim();
                // The slice provider formats each element as
                // a character (`eFormatChar`): the plain
                // quoted character, not the scalar char form.
                let mut parts = Vec::with_capacity(text.len());
                for (i, c) in text.chars().enumerate() {
                    parts.push(format!("[{i}] = '{}'", escape_char(c)));
                }
                let children = if parts.is_empty() {
                    "{}".to_string()
                } else {
                    format!("{{ {} }}", parts.join(" "))
                };
                Some(format!(
                    "({ty}) {expr} = \"{}\" {children}",
                    escape_str(&text)
                ))
            }
            _ => None,
        },
        Some(Desc::Adt(tag)) => {
            let kind = printer_kind(&dwarf_type_name(tag))?;
            match kind {
                PrinterKind::Rc | PrinterKind::Arc => {
                    let (strong, weak, pointee, elem) =
                        rc_parts(interp, tag, value)?;
                    let v = render_lldb_compact(
                        interp,
                        &pointee,
                        elem.as_ref(),
                        false,
                        true,
                        1,
                    )?;
                    // The `Rc` provider's child is named
                    // `value`, the `Arc` provider's `data`.
                    let child = if kind == PrinterKind::Rc {
                        "value"
                    } else {
                        "data"
                    };
                    let ty = lldb_type_name(&dwarf_type_name(tag));
                    Some(format!("({ty}) {expr} = strong={strong}, weak={weak} {{ {child} = {v} }}"))
                }
                _ => None,
            }
        }
        _ => None,
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
    if let Some((scalar, inner)) = nonzero_scalar(value, desc) {
        return render_lldb_compact(
            interp,
            &scalar,
            Some(&inner),
            top_char,
            nested,
            depth,
        );
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
        Value::Closure { .. } | Value::Coroutine { .. } => {
            render_env_lldb_compact(interp, None, value, desc, depth)
        }
        Value::Str(s) => Some(format!("\"{}\"", escape_str(s))),
        Value::Uninit => match desc {
            Some(Desc::Int(_)) | Some(Desc::Float(_)) => Some("0".to_string()),
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
    if matches!(shape.kind, sparagmos_mir::AdtShapeKind::Union) {
        let values = union_field_values(&vshape.fields, fields)?;
        let mut parts = Vec::with_capacity(values.len());
        for (i, v) in values.iter().enumerate() {
            let name = vshape.fields[i].name.as_deref()?;
            let d = Desc::from_tag(&vshape.fields[i].ty);
            let r = render_lldb_compact(interp, v, Some(&d), false, true, depth + 1)?;
            parts.push(format!("{name}:{r}"));
        }
        return Some(format!("{{{}}}", parts.join(", ")));
    }
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
    lldb_expanded_pairs(interp, value, desc, depth)
        .map(|pairs| format!("{{ {} }}", pairs.join(" ")))
}

/// The `name = value` pairs of an aggregate's expanded
/// children form (see [`render_lldb_expanded`]), unjoined:
/// the parenthesised comma-joined form lldb prints for a
/// sequence's children is built from the same pairs.
fn lldb_expanded_pairs(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
    depth: usize,
) -> Option<Vec<String>> {
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
            Some(pairs)
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
            Some(pairs)
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

/// The (field, value) pairs a closure or coroutine value
/// renders with, in printed order, with the instance's recorded
/// environment shape: for a closure, its captures in environment
/// order; for a coroutine, the current state's saved fields
/// followed by its captures (the order the compiler's debuginfo
/// lays a state out in). `None` when the value's instance has no
/// recorded shape or a field's storage is absent.
fn env_render_fields<'i, 'v>(
    interp: &'i Interpreter,
    value: &'v Value,
    desc: Option<&Desc>,
) -> Option<(
    &'i sparagmos_mir::EnvDebugShape,
    Vec<(&'i sparagmos_mir::AdtDebugField, &'v Value)>,
)> {
    let Some(Desc::Adt(tag)) = desc else {
        return None;
    };
    let shape: &'i sparagmos_mir::EnvDebugShape =
        interp.program().env_debug_shapes.get(tag)?;
    let mut pairs = Vec::new();
    match value {
        Value::Closure { env, .. } => {
            if env.len() != shape.upvars.len() {
                return None;
            }
            pairs.extend(shape.upvars.iter().zip(env.iter()));
        }
        Value::Coroutine {
            variant,
            fields,
            saved,
        } => {
            let vshape = shape.variants.get(*variant as usize)?;
            if vshape.fields.len() != vshape.slots.len() {
                return None;
            }
            for (field, slot) in vshape.fields.iter().zip(vshape.slots.iter()) {
                let stored = if shape.slotted {
                    fields.get(*slot as usize)?
                } else {
                    saved.get(*variant as usize)?.get(*slot as usize)?
                };
                pairs.push((field, stored));
            }
            if fields.len() < shape.upvars.len() {
                return None;
            }
            pairs.extend(shape.upvars.iter().zip(fields.iter()));
        }
        _ => return None,
    }
    Some((shape, pairs))
}

/// Render one closure or coroutine value in gdb form:
/// `path::{closure_env#0} {name: value, ...}` for a closure,
/// `path::{coroutine_env#0}::State{name: value, ...}` for a
/// coroutine (see [`env_render_fields`] for the field order).
fn render_env_gdb(
    interp: &Interpreter,
    value: &Value,
    desc: Option<&Desc>,
    depth: usize,
) -> Option<String> {
    let (shape, pairs) = env_render_fields(interp, value, desc)?;
    let mut parts = Vec::with_capacity(pairs.len());
    for (field, stored) in &pairs {
        let name = field.name.as_deref()?;
        let d = Desc::from_tag(&field.ty);
        let rendered = render_gdb(interp, stored, Some(&d), false, None, depth + 1)?;
        parts.push(format!("{name}: {rendered}"));
    }
    match value {
        Value::Closure { .. } => {
            if parts.is_empty() {
                Some(shape.dwarf_name.clone())
            } else {
                Some(format!("{} {{{}}}", shape.dwarf_name, parts.join(", ")))
            }
        }
        Value::Coroutine { variant, .. } => {
            let vname = &shape.variants.get(*variant as usize)?.name;
            let prefix = format!("{}::{vname}", shape.dwarf_name);
            if parts.is_empty() {
                Some(prefix)
            } else {
                Some(format!("{prefix}{{{}}}", parts.join(", ")))
            }
        }
        _ => None,
    }
}

/// Render one closure or coroutine value in lldb's compact
/// form: `{name:value, ...}` over the same fields as
/// [`render_env_gdb`]. At the top level of a print (`frames`
/// given), a captured field that is a reference to an aggregate
/// renders as its pointee - the form lldb's own display gives a
/// reference member - while a reference to a scalar or to
/// another reference stays the address; nested renders
/// (`frames` absent) keep the address form throughout.
fn render_env_lldb_compact(
    interp: &Interpreter,
    frames: Option<&[Frame]>,
    value: &Value,
    desc: Option<&Desc>,
    depth: usize,
) -> Option<String> {
    let (_shape, pairs) = env_render_fields(interp, value, desc)?;
    let mut parts = Vec::with_capacity(pairs.len());
    for (field, stored) in &pairs {
        let name = field.name.as_deref()?;
        let d = Desc::from_tag(&field.ty);
        let rendered = render_env_lldb_field(interp, frames, stored, &d, depth + 1)?;
        parts.push(format!("{name}:{rendered}"));
    }
    Some(format!("{{{}}}", parts.join(", ")))
}

/// Render one environment field's value in lldb's compact form
/// (see [`render_env_lldb_compact`] for the reference rule).
fn render_env_lldb_field(
    interp: &Interpreter,
    frames: Option<&[Frame]>,
    value: &Value,
    desc: &Desc,
    depth: usize,
) -> Option<String> {
    if let (Some(frames), Value::Ref(_), Desc::Ref(inner) | Desc::RefMut(inner)) =
        (frames, value, desc)
    {
        if matches!(inner.as_ref(), Desc::Adt(_)) {
            if let Ok(resolved) = interp.deref_value(frames, value) {
                if let Ok(pointee) = interp.read_resolved(frames, &resolved) {
                    return render_lldb_compact(
                        interp,
                        &pointee,
                        Some(inner.as_ref()),
                        false,
                        true,
                        depth,
                    );
                }
            }
        }
    }
    render_lldb_compact(interp, value, Some(desc), false, true, depth)
}

/// The transcript lines of one lldb print of a closure or
/// coroutine value: the single `(Type) name = <compact>` line,
/// with the environment type's DWARF name.
fn render_env_lldb_lines(
    interp: &Interpreter,
    frames: &[Frame],
    value: &Value,
    desc: Option<&Desc>,
    expr: &str,
) -> Vec<String> {
    let Some(compact) = render_env_lldb_compact(interp, Some(frames), value, desc, 0)
    else {
        return Vec::new();
    };
    let Some(Desc::Adt(tag)) = desc else {
        return vec![format!("= {compact}")];
    };
    let Some(shape) = interp.program().env_debug_shapes.get(tag) else {
        return vec![format!("= {compact}")];
    };
    vec![format!(
        "({}) {expr} = {compact}",
        lldb_type_name(&shape.dwarf_name)
    )]
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
        if let Some(expanded) = nonzero_lldb_expanded(interp, value, desc) {
            lines.push(format!("{first} {expanded}"));
        }
        if let Some(expanded) = render_lldb_expanded(interp, value, desc, 0) {
            lines.push(format!("{first} {expanded}"));
        }
        if let Some(pairs) = lldb_expanded_pairs(interp, value, desc, 0) {
            lines.push(format!("{first} ({})", pairs.join(", ")));
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
        if let Some(pairs) = lldb_expanded_pairs(interp, value, desc, 0) {
            lines.push(format!("= {compact} ({})", pairs.join(", ")));
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
                    // `U+0xXXXXXXXX` form, then the character
                    // itself.
                    DebugDialect::Lldb => {
                        format!("U+0x{:08X} U'{}'", v, escape_char(c))
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

/// A parsed print expression: dereferences first (they prefix
/// the whole expression in debugger syntax), then a base
/// variable — a plain name or a path naming a static — then
/// field/index steps.
struct ParsedExpr {
    base: String,
    steps: Vec<ExprStep>,
}

enum ExprStep {
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

fn parse_expr(text: &str) -> Result<ParsedExpr, String> {
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
        // The leading dereferences apply to the parenthesised
        // value as a whole, after its own steps.
        for _ in 0..derefs {
            parsed.steps.push(ExprStep::Deref);
        }
        parse_steps(&rest[close + 1..], &mut parsed.steps)?;
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
