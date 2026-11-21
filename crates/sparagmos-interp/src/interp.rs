//! The MIR interpreter: a straightforward, obviously-correct executor
//! intended as the correctness oracle for the JIT lane.
//!
//! Semantics (aligned with rustc MIR / Rust 1.100.0):
//! * `BinaryOp` arithmetic is *checked*: overflow, division by zero and
//!   invalid shifts are guest panics ([`InterpError::Panic`]).
//! * `CheckedBinaryOp` never panics on overflow; it yields
//!   `(wrapped_value, overflowed)` — exactly what a following MIR `Assert`
//!   (also implemented) turns into a panic, as rustc emits it.
//! * Integer width comes from the destination local's declared [`Ty`];
//!   without a declared width, values use the full `i128`/`u128` range.
//! * Places are resolved against the whole frame stack: a reference
//!   value names a frame by its unique id, so reads and writes through
//!   references work across calls in both directions while the target
//!   frame is live.
//! * Execution is bounded by a step limit and a call-depth limit so a
//!   looping program can never hang the oracle (or a test run).
//! * Host-panic freedom: every failure a guest program can cause is
//!   an [`InterpError`], never a host `panic!`. The `expect` /
//!   `unreachable!` sites in this crate are internal invariants only:
//!   the frame stack is non-empty by construction while the run loop
//!   executes, builtin arities are checked before argument slots are
//!   read, and the arithmetic helpers split the closed `BinOp`
//!   enumeration exhaustively. Input-shaped failures (unsupported
//!   kinds, type mismatches, uninitialized reads, resource limits)
//!   all return errors. Regression coverage lives in
//!   `tests/robustness_tests.rs`.

use crate::error::{InterpError, InterpResult};
use crate::ir::*;
use crate::value::{RefProj, RefTarget, Value};

/// Interpreter limits / configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterpConfig {
    /// Maximum number of statements + terminators executed in one run.
    pub max_steps: usize,
    /// Maximum number of simultaneously live call frames.
    pub max_call_depth: usize,
}

impl Default for InterpConfig {
    fn default() -> Self {
        InterpConfig {
            max_steps: 1_000_000,
            max_call_depth: 1024,
        }
    }
}

/// One call frame: a function's locals plus where to resume.
pub(crate) struct Frame {
    /// Unique id within the run; references name their target frame
    /// by this id, so a frame keeps its identity when other frames
    /// are pushed and popped around it.
    pub(crate) id: u64,
    pub(crate) func: String,
    pub(crate) locals: Vec<Value>,
    block: usize,
    stmt: usize,
    /// Where (in the caller) the return value goes; `None` for the entry frame.
    return_dest: Option<Place>,
    /// Block to resume in the caller after return.
    return_block: Option<BasicBlock>,
}

/// Drop glue suspended while a destructor body runs. A `Drop`
/// terminator whose glue calls a destructor pushes the destructor's
/// frame and one of these; when that frame returns, the interpreter
/// resumes the remaining glue work instead of delivering a return
/// value to a caller place.
struct DropContinuation {
    /// Frame id of the destructor call this work waits on.
    callee_frame: u64,
    /// The dropped place the remaining work applies to.
    root: RefTarget,
    /// Remaining (path from `root`, plan) items, a stack: the last
    /// item runs next.
    work: Vec<(Vec<RefProj>, crate::ir::DropPlan)>,
    /// ADT glue table of the originating `Drop` terminator.
    adts: Vec<crate::ir::DropPlan>,
    /// Block the owner frame jumps to when the work is done.
    target: BasicBlock,
}

/// A place fully resolved against the frame stack: either a concrete
/// slot (frame position + local + constant projection) or, when the
/// resolution passed through a constant aggregate, the value itself.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Resolved {
    Place {
        frame_pos: usize,
        local: usize,
        projection: Vec<RefProj>,
    },
    Value(Value),
}

/// The interpreter state for one program.
pub struct Interpreter<'a> {
    program: &'a Program,
    config: InterpConfig,
    steps: usize,
    next_frame_id: u64,
    /// Everything the guest wrote to its stdout (via the runtime
    /// print support) during the most recent run.
    output: String,
    /// Drop glue suspended on destructor calls (see
    /// [`DropContinuation`]); a stack, innermost last.
    drop_continuations: Vec<DropContinuation>,
}

/// The synthetic address of a pointer value (see
/// [`crate::ir::Rvalue::PtrToInt`]): an FNV-1a hash of the reference
/// target, mapped into a non-zero, 4096-aligned range. The same
/// target always yields the same address within a run, so equality,
/// null and alignment checks on transmuted pointers behave as they
/// do natively; distinct targets collide only on a hash collision.
fn synthetic_address(value: &Value) -> Option<u128> {
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
    let key = match value {
        Value::Ref(target) => format!("ref:{target:?}"),
        Value::FatRef { base, len } => format!("fat:{base:?}:{len}"),
        Value::FnPtr(name) => format!("fn:{name}"),
        _ => return None,
    };
    let h = fnv1a(key.as_bytes());
    Some(0x1_0000_0000u128 + u128::from(h % 0x10_0000) * 0x1000)
}

impl<'a> Interpreter<'a> {
    pub fn new(program: &'a Program) -> Self {
        Interpreter::with_config(program, InterpConfig::default())
    }

    pub fn with_config(program: &'a Program, config: InterpConfig) -> Self {
        Interpreter {
            program,
            config,
            steps: 0,
            next_frame_id: 0,
            output: String::new(),
            drop_continuations: Vec::new(),
        }
    }

    /// Steps consumed by the most recent (or current) run.
    pub fn steps_used(&self) -> usize {
        self.steps
    }

    /// The guest stdout captured during the most recent run.
    pub fn output(&self) -> &str {
        &self.output
    }

    pub(crate) fn push_output(&mut self, text: &str) {
        self.output.push_str(text);
    }

    fn tick(&mut self) -> InterpResult<()> {
        self.steps += 1;
        if self.steps > self.config.max_steps {
            return Err(InterpError::StepLimitExceeded {
                limit: self.config.max_steps,
            });
        }
        Ok(())
    }

    fn fresh_frame_id(&mut self) -> u64 {
        let id = self.next_frame_id;
        self.next_frame_id += 1;
        id
    }

    /// Run `entry` with `args` and return the value of its return place.
    pub fn run(&mut self, entry: &str, args: Vec<Value>) -> InterpResult<Value> {
        self.steps = 0;
        self.next_frame_id = 0;
        self.output.clear();
        let body = self
            .program
            .get(entry)
            .ok_or_else(|| self.missing_body_error(entry))?;
        if args.len() != body.arg_count {
            return Err(InterpError::type_mismatch(
                format!("call to `{entry}`"),
                format!("{} argument(s)", body.arg_count),
                format!("{} argument(s)", args.len()),
            ));
        }
        let mut locals = vec![Value::Uninit; body.local_count()];
        for (i, arg) in args.into_iter().enumerate() {
            locals[i + 1] = arg;
        }
        let entry_id = self.fresh_frame_id();
        let mut frames: Vec<Frame> = vec![Frame {
            id: entry_id,
            func: entry.to_string(),
            locals,
            block: 0,
            stmt: 0,
            return_dest: None,
            return_block: None,
        }];

        loop {
            // Fetch the current block's next unit of work, cloning the small
            // pieces we need so the frame stack can be mutated freely after.
            let (func_name, block_idx, stmt_idx) = {
                let frame = frames
                    .last()
                    .expect("frame stack is non-empty while running");
                (frame.func.clone(), frame.block, frame.stmt)
            };
            // Clone the body for this iteration: the oracle favours clarity
            // over speed, and owning the body lets the frame stack and the
            // step counter be mutated without borrow conflicts.
            let body = self.program.get(&func_name).cloned().ok_or_else(|| {
                InterpError::UnknownFunction {
                    name: func_name.clone(),
                }
            })?;
            let body = &body;
            let block = body
                .blocks
                .get(block_idx)
                .ok_or_else(|| InterpError::UnknownBlock {
                    func: func_name.clone(),
                    block: block_idx,
                })?;

            if stmt_idx < block.statements.len() {
                let statement = block.statements[stmt_idx].clone();
                // Advance the program counter before executing, so a call
                // (not possible in a statement) or an error leaves sane state.
                frames.last_mut().expect("frame").stmt += 1;
                self.tick()?;
                self.exec_statement(&mut frames, body, &statement)?;
                continue;
            }

            // Terminator.
            let terminator = block.terminator.clone();
            self.tick()?;
            match terminator {
                TerminatorKind::Goto { target } => {
                    Self::jump(&mut frames, body, target)?;
                }
                TerminatorKind::SwitchInt {
                    discr,
                    targets,
                    otherwise,
                } => {
                    let value = self.eval_operand(&mut frames, body, &discr)?;
                    let bits = value.switch_bits().ok_or_else(|| {
                        InterpError::type_mismatch(
                            "SwitchInt discriminant",
                            "integer or bool",
                            value.kind_name(),
                        )
                    })?;
                    let target = targets
                        .iter()
                        .find(|(pattern, _)| *pattern == bits)
                        .map(|(_, bb)| *bb)
                        .unwrap_or(otherwise);
                    Self::jump(&mut frames, body, target)?;
                }
                TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    target,
                } => {
                    let mut arg_values = Vec::with_capacity(args.len());
                    for arg in &args {
                        arg_values.push(self.eval_operand(&mut frames, body, arg)?);
                    }
                    self.start_call(&mut frames, &func, arg_values, &destination, target)?;
                }
                TerminatorKind::CallIndirect {
                    func,
                    args,
                    destination,
                    target,
                } => {
                    // The callee value comes from a place: resolve it
                    // together with the environment reference a
                    // closure body expects (see `eval_callable`). A
                    // body that takes its environment by value (an
                    // `FnOnce` closure's own body) gets the value
                    // itself instead (see `consumed_env_arg`).
                    let (callee, env_ref) = self.eval_callable(&frames, body, &func)?;
                    let env_ref = match self.consumed_env_arg(&mut frames, body, &func, &callee)? {
                        Some(by_value) => by_value,
                        None => env_ref,
                    };
                    let mut arg_values = Vec::with_capacity(args.len());
                    for arg in &args {
                        arg_values.push(self.eval_operand(&mut frames, body, arg)?);
                    }
                    match callee {
                        Value::FnPtr(name) => {
                            self.start_call(
                                &mut frames,
                                &name,
                                arg_values,
                                &destination,
                                target,
                            )?;
                        }
                        Value::Closure { def, .. } => {
                            let mut full = vec![env_ref];
                            full.extend(arg_values);
                            self.start_call(&mut frames, &def, full, &destination, target)?;
                        }
                        other => {
                            return Err(InterpError::type_mismatch(
                                "indirect call",
                                "callable value",
                                other.kind_name(),
                            ));
                        }
                    }
                }
                TerminatorKind::CallTrait {
                    kind,
                    receiver,
                    args,
                    destination,
                    target,
                } => {
                    let args_value = self.eval_operand(&mut frames, body, &args)?;
                    let arg_values = match args_value {
                        Value::Tuple(items) => items,
                        Value::Unit => Vec::new(),
                        other => {
                            return Err(InterpError::type_mismatch(
                                "Fn trait call arguments",
                                "argument tuple",
                                other.kind_name(),
                            ));
                        }
                    };
                    // The callable value, and the environment
                    // argument a closure body expects as its first
                    // argument. For `call` / `call_mut` the receiver
                    // already is the reference the body takes. For
                    // `call_once` the receiver is the value itself:
                    // a body that takes its environment by value (an
                    // `FnOnce` closure's own body — no shim exists in
                    // the program) receives the value, consuming the
                    // receiver place; any other body receives a
                    // reference to the value in place, which is what
                    // the real `call_once` shim passes when it
                    // consumes an `Fn` / `FnMut` closure (aliasing
                    // the moved-from place is equivalent because that
                    // place is dead after the call).
                    let (callable, env_ref) = match kind {
                        FnCallKind::CallOnce => {
                            let (value, env_ref) = self.eval_callable(&frames, body, &receiver)?;
                            let (value, env_ref) =
                                if matches!(value, Value::DynRef { .. }) {
                                    self.dyn_callable(&frames, &value)?
                                } else {
                                    (value, env_ref)
                                };
                            let env_ref = match self.consumed_env_arg(
                                &mut frames,
                                body,
                                &receiver,
                                &value,
                            )? {
                                Some(by_value) => by_value,
                                None => env_ref,
                            };
                            (value, env_ref)
                        }
                        FnCallKind::Call | FnCallKind::CallMut => {
                            let recv = self.eval_operand(&mut frames, body, &receiver)?;
                            match recv {
                                Value::Ref(_) | Value::FatRef { .. } => {
                                    let resolved = self.deref_value(&frames, &recv)?;
                                    let value = self.read_resolved(&frames, &resolved)?;
                                    if matches!(value, Value::DynRef { .. }) {
                                        self.dyn_callable(&frames, &value)?
                                    } else {
                                        (value, recv)
                                    }
                                }
                                Value::DynRef { .. } => {
                                    self.dyn_callable(&frames, &recv)?
                                }
                                value => {
                                    let env = self.callable_env_ref(
                                        &frames,
                                        body,
                                        &receiver,
                                        &value,
                                    )?;
                                    (value, env)
                                }
                            }
                        }
                    };
                    match callable {
                        Value::Closure { def, .. } => {
                            let mut full = vec![env_ref];
                            full.extend(arg_values);
                            self.start_call(&mut frames, &def, full, &destination, target)?;
                        }
                        Value::FnPtr(name) => {
                            self.start_call(
                                &mut frames,
                                &name,
                                arg_values,
                                &destination,
                                target,
                            )?;
                        }
                        other => {
                            return Err(InterpError::type_mismatch(
                                "Fn trait call",
                                "callable value",
                                other.kind_name(),
                            ));
                        }
                    }
                }
                TerminatorKind::CallBuiltin {
                    name,
                    generic_args,
                    args,
                    destination,
                    target,
                    drop_glue,
                } => {
                    if let Some(glue) = drop_glue {
                        // `std::mem::drop` with the dropped type's glue
                        // attached by conversion: run it against the
                        // argument place, exactly like a `Drop`
                        // terminator — a destructor call suspends the
                        // remaining glue on a `DropContinuation` and
                        // resumes it when the destructor returns. The
                        // call's own (unit) result is written first;
                        // the destination is a different local from
                        // the dropped argument place.
                        let Some(bb) = target else {
                            return Err(InterpError::MissingContinuation {
                                func: func_name.clone(),
                            });
                        };
                        let Some(operand) = args.first() else {
                            return Err(InterpError::type_mismatch(
                                "std::mem::drop",
                                "one argument",
                                "no arguments",
                            ));
                        };
                        let place = match operand {
                            Operand::Copy(place) | Operand::Move(place) => place,
                            _ => {
                                return Err(InterpError::type_mismatch(
                                    "std::mem::drop",
                                    "a place argument",
                                    "a constant argument",
                                ));
                            }
                        };
                        let resolved = self.resolve_place(&frames, body, place)?;
                        let root = match resolved {
                            Resolved::Place {
                                frame_pos,
                                local,
                                projection,
                            } => RefTarget::place(frames[frame_pos].id, local, projection),
                            Resolved::Value(_) => {
                                return Err(InterpError::type_mismatch(
                                    "std::mem::drop",
                                    "a storage place",
                                    "a constant aggregate",
                                ));
                            }
                        };
                        self.write_place(&mut frames, body, &destination, Value::Unit)?;
                        let work = vec![(Vec::new(), glue.root.clone())];
                        self.run_drop_glue(&mut frames, root, work, &glue.adts, bb)?;
                    } else {
                        let result =
                            self.exec_builtin(&mut frames, body, &name, &generic_args, &args)?;
                        match target {
                            Some(bb) => {
                                self.write_place(&mut frames, body, &destination, result)?;
                                Self::jump(&mut frames, body, bb)?;
                            }
                            None => {
                                return Err(InterpError::MissingContinuation {
                                    func: func_name.clone(),
                                });
                            }
                        }
                    }
                }
                TerminatorKind::Drop { place, target, glue } => {
                    // Run the dropped type's glue (conversion built
                    // it; see `mir_convert`). The glue work starts
                    // here and either finishes in `run_drop_glue`
                    // (which jumps to `target`) or suspends on a
                    // destructor call and resumes when that call
                    // returns.
                    let resolved = self.resolve_place(&frames, body, &place)?;
                    let root = match resolved {
                        Resolved::Place {
                            frame_pos,
                            local,
                            projection,
                        } => RefTarget::place(frames[frame_pos].id, local, projection),
                        Resolved::Value(_) => {
                            return Err(InterpError::type_mismatch(
                                "Drop",
                                "a storage place",
                                "a constant aggregate",
                            ));
                        }
                    };
                    let work = vec![(Vec::new(), glue.root.clone())];
                    self.run_drop_glue(&mut frames, root, work, &glue.adts, target)?;
                }
                TerminatorKind::Return => {
                    // A returning destructor frame resumes its
                    // suspended drop glue instead of delivering a
                    // value to a caller place.
                    let returning = frames.last().expect("frame").id;
                    if let Some(pos) = self
                        .drop_continuations
                        .iter()
                        .rposition(|c| c.callee_frame == returning)
                    {
                        let cont = self.drop_continuations.remove(pos);
                        frames.pop();
                        self.run_drop_glue(
                            &mut frames,
                            cont.root,
                            cont.work,
                            &cont.adts,
                            cont.target,
                        )?;
                        continue;
                    }
                    let frame = frames.pop().expect("frame");
                    let mut value = frame.locals.first().cloned().unwrap_or(Value::Uninit);
                    if value == Value::Uninit {
                        // rustc MIR never writes the return place of
                        // a zero-sized return type: the final block
                        // is a bare `return;`, and the type's single
                        // value is implied. An unwritten `_0`
                        // therefore reads as that value exactly when
                        // the declared return type is provably
                        // zero-sized. For every other type an
                        // unwritten `_0` is a genuine uninitialized
                        // read and stays an error.
                        match self.zst_local_value(&frame, 0) {
                            Some(v) => value = v,
                            None => {
                                return Err(InterpError::UninitializedRead {
                                    func: frame.func.clone(),
                                    place: "_0".to_string(),
                                });
                            }
                        }
                    }
                    if frames.is_empty() {
                        return Ok(value);
                    }
                    let dest = frame
                        .return_dest
                        .clone()
                        .expect("non-entry frame has a destination");
                    let resume = frame.return_block;
                    let caller_name = frames.last().expect("caller").func.clone();
                    let caller_body = self.program.get(&caller_name).cloned().ok_or_else(|| {
                        InterpError::UnknownFunction {
                            name: caller_name.clone(),
                        }
                    })?;
                    self.write_place(&mut frames, &caller_body, &dest, value)?;
                    match resume {
                        Some(bb) => Self::jump(&mut frames, &caller_body, bb)?,
                        None => {
                            return Err(InterpError::MissingContinuation { func: caller_name });
                        }
                    }
                }
                TerminatorKind::Unreachable => {
                    return Err(InterpError::panic("entered unreachable code"));
                }
                TerminatorKind::Assert {
                    cond,
                    expected,
                    target,
                    message,
                } => {
                    let value = self.eval_operand(&mut frames, body, &cond)?;
                    let actual = value.as_bool().ok_or_else(|| {
                        InterpError::type_mismatch("Assert condition", "bool", value.kind_name())
                    })?;
                    if actual != expected {
                        return Err(InterpError::panic(message));
                    }
                    Self::jump(&mut frames, body, target)?;
                }
                TerminatorKind::Unsupported(kind) => {
                    return Err(InterpError::UnsupportedTerminator { kind });
                }
            }
        }
    }

    /// The error for naming a function that has no body in the
    /// program: the recorded conversion failure when the function
    /// exists but could not be converted, a plain unknown-function
    /// error otherwise.
    fn missing_body_error(&self, name: &str) -> InterpError {
        match self.program.unconverted_reason(name) {
            Some(reason) => InterpError::UnconvertedFunction {
                name: name.to_string(),
                reason: reason.to_string(),
            },
            None => InterpError::UnknownFunction {
                name: name.to_string(),
            },
        }
    }

    /// Resolve a call that has no body under its own name against
    /// a `dyn` receiver: the vtable lookup, with the fat pointer's
    /// (concrete type, trait) pair standing in for the vtable.
    ///
    /// Returns `None` when the call is not a dynamic dispatch (the
    /// first argument is not a dyn fat pointer, or the callee name
    /// is not a trait method of the pointer's bounds and no impl
    /// body matches). Returns `Some(Ok((impl_name, receiver)))` to
    /// dispatch: the receiver is the fat pointer's data pointer.
    /// Returns `Some(Err(_))` when the call is a dispatch through
    /// the pointer but the concrete implementation body is not in
    /// the program — the explicit boundary of the dyn model, hit at
    /// the call, never at the value's declaration.
    fn virtual_call_target(
        &self,
        func: &str,
        args: &[Value],
    ) -> Option<Result<(String, Value), InterpError>> {
        let Value::DynRef {
            base,
            concrete,
            traits,
        } = args.first()?
        else {
            return None;
        };
        let (trait_path, method) = func.rsplit_once("::")?;
        if trait_path.is_empty() || method.is_empty() {
            return None;
        }
        let candidate = format!("<{concrete} as {trait_path}>::{method}");
        if self.program.get(&candidate).is_some()
            || self.program.unconverted_reason(&candidate).is_some()
        {
            return Some(Ok((candidate, (**base).clone())));
        }
        if traits.iter().any(|t| t == trait_path) {
            return Some(Err(InterpError::UnsupportedDynamicDispatch {
                method: func.to_string(),
                concrete: concrete.clone(),
            }));
        }
        None
    }

    /// Push a call frame for `func` with already-evaluated argument
    /// values (placed at locals 1..), returning into `destination` /
    /// `target` of the current top frame when the callee returns.
    /// Shared by direct calls, indirect calls and `Fn`-trait calls.
    fn start_call(
        &mut self,
        frames: &mut Vec<Frame>,
        func: &str,
        arg_values: Vec<Value>,
        destination: &Place,
        target: Option<BasicBlock>,
    ) -> InterpResult<()> {
        let callee = match self.program.get(func) {
            Some(c) => c.clone(),
            None => match self.virtual_call_target(func, &arg_values) {
                // A vtable call: dispatch to the concrete type's
                // impl body, with the fat pointer's data pointer as
                // the receiver (the box aggregate itself for a
                // by-value `Box<dyn>` receiver).
                Some(Ok((impl_func, receiver))) => {
                    // Shape the receiver the way the impl body's
                    // `self` type expects: a reference receiver gets
                    // a reference to the concrete value (through the
                    // box aggregate for a `Box<dyn>`), a by-value
                    // receiver gets the data pointer as it is.
                    let receiver = match self
                        .program
                        .get(&impl_func)
                        .and_then(|b| b.local_tys.get(1))
                    {
                        Some(Ty::Ref(..)) => match receiver {
                            Value::Enum { fields, .. } if fields.len() == 1 => {
                                Value::Ref(RefTarget::Value(Box::new(
                                    fields.into_iter().next().expect("one field"),
                                )))
                            }
                            other @ (Value::Ref(_)
                            | Value::FatRef { .. }
                            | Value::DynRef { .. }) => other,
                            other => {
                                Value::Ref(RefTarget::Value(Box::new(other)))
                            }
                        },
                        _ => receiver,
                    };
                    let mut args = arg_values;
                    args[0] = receiver;
                    return self.start_call(frames, &impl_func, args, destination, target);
                }
                Some(Err(e)) => return Err(e),
                None => return Err(self.missing_body_error(func)),
            },
        };
        if frames.len() >= self.config.max_call_depth {
            return Err(InterpError::CallDepthExceeded {
                limit: self.config.max_call_depth,
            });
        }
        if arg_values.len() != callee.arg_count {
            return Err(InterpError::type_mismatch(
                format!("call to `{func}`"),
                format!("{} argument(s)", callee.arg_count),
                format!("{} argument(s)", arg_values.len()),
            ));
        }
        let mut locals = vec![Value::Uninit; callee.local_count()];
        for (i, v) in arg_values.into_iter().enumerate() {
            locals[i + 1] = v;
        }
        let frame_id = self.fresh_frame_id();
        frames.push(Frame {
            id: frame_id,
            func: func.to_string(),
            locals,
            block: 0,
            stmt: 0,
            return_dest: Some(destination.clone()),
            return_block: target,
        });
        Ok(())
    }

    /// Evaluate a callable operand (a closure or function value being
    /// called by value) together with the environment reference a
    /// closure body expects as its first argument. The body's first
    /// parameter is a reference to the closure value in rustc MIR;
    /// when the operand is a place, the reference points at that
    /// place, so `FnMut` mutations of by-value captures persist
    /// across calls, exactly as through the real shim. A `Move`
    /// operand's place is deliberately not cleared here: the
    /// environment reference aliases it for the duration of the
    /// call, and the place is dead afterwards by MIR's move
    /// semantics. (A body that takes its environment by value is
    /// the exception: `consumed_env_arg` passes the value itself
    /// and does clear the place.)

    /// The callable value and environment reference behind a
    /// `dyn Fn`-family fat pointer: the concrete value it denotes
    /// (a closure value, for the payloads a program can contain),
    /// referenced through the fat pointer's own base when that
    /// base is a reference (a `&dyn Fn`), and as a value otherwise
    /// (an owned `Box<dyn Fn>` — the closure value inside the box
    /// aggregate; its captures are place references, so calls
    /// through the copy still reach them).
    fn dyn_callable(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<(Value, Value)> {
        let Value::DynRef { base, .. } = value else {
            unreachable!("dyn_callable on a non-dyn value")
        };
        let resolved = self.deref_value(frames, value)?;
        let callable = self.read_resolved(frames, &resolved)?;
        let env = match &**base {
            Value::Ref(_) | Value::FatRef { .. } => (**base).clone(),
            _ => Value::Ref(RefTarget::Value(Box::new(callable.clone()))),
        };
        Ok((callable, env))
    }

    fn eval_callable(
        &self,
        frames: &[Frame],
        body: &Body,
        operand: &Operand,
    ) -> InterpResult<(Value, Value)> {
        let value = match operand {
            Operand::Copy(place) | Operand::Move(place) => {
                let resolved = self.resolve_place(frames, body, place)?;
                self.read_resolved(frames, &resolved)?
            }
            Operand::Constant(c) => self.eval_constant(c)?,
        };
        let env_ref = self.callable_env_ref(frames, body, operand, &value)?;
        Ok((value, env_ref))
    }

    /// The environment reference for a callable value obtained from
    /// `operand`: a reference to the operand's place when it has one,
    /// otherwise a reference to the value itself (constant form).
    fn callable_env_ref(
        &self,
        frames: &[Frame],
        body: &Body,
        operand: &Operand,
        value: &Value,
    ) -> InterpResult<Value> {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => {
                match self.resolve_place(frames, body, place)? {
                    Resolved::Place {
                        frame_pos,
                        local,
                        projection,
                    } => Ok(Value::Ref(RefTarget::place(
                        frames[frame_pos].id,
                        local,
                        projection,
                    ))),
                    Resolved::Value(_) => {
                        Ok(Value::Ref(RefTarget::Value(Box::new(value.clone()))))
                    }
                }
            }
            Operand::Constant(_) => Ok(Value::Ref(RefTarget::Value(Box::new(value.clone())))),
        }
    }

    /// Whether the converted body of the closure `def` takes its
    /// environment by value: its first parameter is the closure
    /// value itself, not a reference to it. That is the body shape
    /// produced for a closure whose own kind is `FnOnce` — the
    /// closure body is resolved with the closure's own kind, so no
    /// `FnOnce` adapter shim exists in the program, and the body
    /// reads its captures with plain field projections on the value
    /// (and may move them out of it).
    fn closure_body_takes_env_by_value(&self, def: &str) -> bool {
        match self.program.get(def) {
            Some(body) => matches!(body.local_ty(Local(1)), Some(Ty::Closure(..))),
            None => false,
        }
    }

    /// The environment argument for a call that consumes its
    /// callable (`FnOnce::call_once`, or an indirect call through a
    /// callable place) when the callee is the closure `value`:
    /// `Some` holding the closure value itself when the closure's
    /// body takes its environment by value. A `Move` operand's
    /// place is then left uninitialized — the value has moved into
    /// the callee — so the closure is really consumed: a later read
    /// of the place is an uninitialized read and a later drop of it
    /// runs no glue a second time, exactly as with any other move.
    /// `None` when the body takes a reference to the environment
    /// instead; the caller passes the environment reference it
    /// already built, which is also what the `call_once` shim does
    /// when it consumes an `Fn` / `FnMut` closure.
    fn consumed_env_arg(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        operand: &Operand,
        value: &Value,
    ) -> InterpResult<Option<Value>> {
        match value {
            Value::Closure { def, .. } if self.closure_body_takes_env_by_value(def) => {}
            _ => return Ok(None),
        }
        if let Operand::Move(place) = operand {
            if place.is_direct_local() {
                let frame = frames.last_mut().expect("frame");
                if let Some(slot) = frame.locals.get_mut(place.local.0) {
                    *slot = Value::Uninit;
                }
            } else if let Ok(target) = self.resolve_place(frames, body, place) {
                if matches!(target, Resolved::Place { .. }) {
                    self.write_resolved(frames, &target, Value::Uninit)?;
                }
            }
        }
        Ok(Some(value.clone()))
    }

    /// Read the value at `root` + `path` for drop glue: `Ok(None)`
    /// when any part of the way is uninitialized (a moved-out or
    /// never-written part has no glue to run, matching rustc's drop
    /// flags); other read failures are errors.
    fn read_at(
        &self,
        frames: &[Frame],
        root: &RefTarget,
        path: &[RefProj],
    ) -> InterpResult<Option<Value>> {
        let (base, prefix): (Value, &[RefProj]) = match root {
            RefTarget::Place {
                frame,
                local,
                projection,
            } => {
                let frame_pos = self.frame_pos(frames, *frame)?;
                let frame = &frames[frame_pos];
                let base = frame.locals.get(*local).cloned().ok_or_else(|| {
                    InterpError::UnknownLocal {
                        func: frame.func.clone(),
                        local: *local,
                    }
                })?;
                (base, projection)
            }
            RefTarget::Value(v) => ((**v).clone(), &[]),
            // A null pointer denotes nothing; there is no drop
            // glue to read through it.
            RefTarget::Null => return Ok(None),
        };
        let mut value = base;
        for step in prefix.iter().chain(path.iter()) {
            if value == Value::Uninit {
                return Ok(None);
            }
            match project_value(value, std::slice::from_ref(step), "<drop>", "drop glue") {
                Ok(v) => value = v,
                Err(InterpError::UninitializedRead { .. }) => return Ok(None),
                Err(e) => return Err(e),
            }
        }
        if value == Value::Uninit {
            Ok(None)
        } else {
            Ok(Some(value))
        }
    }

    /// A reference to `root` + `path`, for a destructor call.
    fn extend_ref(&self, root: &RefTarget, path: &[RefProj]) -> InterpResult<RefTarget> {
        match root {
            RefTarget::Place {
                frame,
                local,
                projection,
            } => {
                let mut projection = projection.clone();
                projection.extend(path.iter().cloned());
                Ok(RefTarget::place(*frame, *local, projection))
            }
            RefTarget::Value(v) => {
                let projected = project_value((**v).clone(), path, "<drop>", "drop glue")?;
                Ok(RefTarget::Value(Box::new(projected)))
            }
            RefTarget::Null => Err(InterpError::type_mismatch(
                "drop glue reference",
                "a valid reference",
                "null pointer",
            )),
        }
    }

    /// Execute drop glue: process `work` until it is empty or a
    /// destructor call has to run. A destructor call pushes its
    /// frame plus a [`DropContinuation`] and returns; the `Return`
    /// terminator resumes the remaining work when the destructor
    /// finishes. When the work is done, the dropped place's own slot
    /// is cleared (for a direct local of the owner frame) and the
    /// owner frame jumps to `target`.
    fn run_drop_glue(
        &mut self,
        frames: &mut Vec<Frame>,
        root: RefTarget,
        mut work: Vec<(Vec<RefProj>, crate::ir::DropPlan)>,
        adts: &[crate::ir::DropPlan],
        target: BasicBlock,
    ) -> InterpResult<()> {
        use crate::ir::DropPlan as P;
        while let Some((path, plan)) = work.pop() {
            self.tick()?;
            match plan {
                P::Trivial => {}
                P::Untyped => {
                    // The type of this part is unrecorded, so judge
                    // by the value's own shape: kinds that never
                    // carry glue pass, tuples recurse (a tuple has
                    // no destructor of its own), and an ADT value
                    // might hide a destructor — an explicit error.
                    match self.read_at(frames, &root, &path)? {
                        None => {}
                        Some(Value::Tuple(items)) => {
                            for i in (0..items.len()).rev() {
                                let mut sub = path.clone();
                                sub.push(RefProj::Index(i));
                                work.push((sub, P::Untyped));
                            }
                        }
                        Some(Value::Enum { .. }) | Some(Value::Coroutine { .. }) => {
                            return Err(InterpError::UnsupportedTerminator {
                                kind: "Drop".to_string(),
                            });
                        }
                        Some(_) => {}
                    }
                }
                P::Unknown => {
                    return Err(InterpError::UnsupportedTerminator {
                        kind: "Drop".to_string(),
                    });
                }
                P::Seq(plans) => {
                    for plan in plans.into_iter().rev() {
                        work.push((path.clone(), plan));
                    }
                }
                P::At { path: extra, plan } => {
                    let mut full = path;
                    full.extend(extra);
                    work.push((full, *plan));
                }
                P::Adt { index } => {
                    let plan = adts.get(index).cloned().ok_or_else(|| {
                        InterpError::type_mismatch(
                            "drop glue",
                            "an ADT plan in the glue table",
                            "a glue index out of range",
                        )
                    })?;
                    work.push((path, plan));
                }
                P::Fields { fields, complete } => {
                    let Some(value) = self.read_at(frames, &root, &path)? else {
                        continue;
                    };
                    let is_coroutine = matches!(value, Value::Coroutine { .. });
                    let items = match value {
                        Value::Tuple(items) => items,
                        Value::Enum { fields, .. } => fields,
                        // A closure's fields are its captures.
                        Value::Closure { env, .. } => env,
                        Value::Coroutine { fields, .. } => fields,
                        other => {
                            return Err(InterpError::type_mismatch(
                                "drop glue",
                                "a tuple or ADT value",
                                other.kind_name(),
                            ));
                        }
                    };
                    if items.len() != fields.len()
                        && !(is_coroutine && items.len() < fields.len())
                    {
                        // With `complete`, conversion proved the
                        // field count, so a mismatch means the value
                        // is malformed; without it, the value holds
                        // fields whose glue conversion could not
                        // account for. Either way the drop cannot be
                        // carried out faithfully: explicit, never
                        // silent.
                        let _ = complete;
                        return Err(InterpError::UnsupportedTerminator {
                            kind: "Drop".to_string(),
                        });
                    }
                    for (i, field_plan) in fields.into_iter().enumerate().rev() {
                        if i >= items.len() {
                            // Only reachable for a coroutine value
                            // whose storage is shorter than the plan:
                            // the field was never written, so it
                            // holds nothing to drop.
                            continue;
                        }
                        if field_plan != P::Trivial {
                            let mut sub = path.clone();
                            sub.push(RefProj::Field(i));
                            work.push((sub, field_plan));
                        }
                    }
                }
                P::Each { elem } => {
                    let Some(value) = self.read_at(frames, &root, &path)? else {
                        continue;
                    };
                    let (count, is_enum) = match &value {
                        Value::Tuple(items) => (items.len(), false),
                        Value::Enum { fields, .. } => (fields.len(), true),
                        other => {
                            return Err(InterpError::type_mismatch(
                                "drop glue",
                                "an array, tuple or Vec value",
                                other.kind_name(),
                            ));
                        }
                    };
                    for i in (0..count).rev() {
                        let mut sub = path.clone();
                        sub.push(if is_enum {
                            RefProj::Field(i)
                        } else {
                            RefProj::Index(i)
                        });
                        work.push((sub, (*elem).clone()));
                    }
                }
                P::Variants { variants } => {
                    let Some(value) = self.read_at(frames, &root, &path)? else {
                        continue;
                    };
                    let variant = match value {
                        Value::Enum { variant, .. } => variant,
                        Value::Coroutine { variant, .. } => variant,
                        Value::Tuple(_) => 0,
                        other => {
                            return Err(InterpError::type_mismatch(
                                "drop glue",
                                "an ADT value",
                                other.kind_name(),
                            ));
                        }
                    };
                    let plan = variants
                        .get(variant as usize)
                        .cloned()
                        .unwrap_or(P::Unknown);
                    work.push((path, plan));
                }
                P::RefCellRelease { mutable } => {
                    // A moved-out / never-written guard has no
                    // borrow to release, exactly like drop flags in
                    // real MIR.
                    let Some(guard) = self.read_at(frames, &root, &path)? else {
                        continue;
                    };
                    self.release_refcell_borrow(frames, &guard, mutable)?;
                }
                P::CallDtor { func } => {
                    if self.read_at(frames, &root, &path)?.is_none() {
                        // Moved out of / never written: no glue, as
                        // with rustc's drop flags.
                        continue;
                    }
                    let callee = self.program.get(&func).cloned().ok_or_else(|| {
                        InterpError::UnknownFunction { name: func.clone() }
                    })?;
                    if frames.len() >= self.config.max_call_depth {
                        return Err(InterpError::CallDepthExceeded {
                            limit: self.config.max_call_depth,
                        });
                    }
                    if callee.arg_count != 1 || callee.local_count() < 2 {
                        return Err(InterpError::type_mismatch(
                            format!("destructor `{func}`"),
                            "one argument",
                            format!("{} argument(s)", callee.arg_count),
                        ));
                    }
                    let target_ref = self.extend_ref(&root, &path)?;
                    let mut locals = vec![Value::Uninit; callee.local_count()];
                    locals[1] = Value::Ref(target_ref);
                    let frame_id = self.fresh_frame_id();
                    self.drop_continuations.push(DropContinuation {
                        callee_frame: frame_id,
                        root,
                        work,
                        adts: adts.to_vec(),
                        target,
                    });
                    frames.push(Frame {
                        id: frame_id,
                        func,
                        locals,
                        block: 0,
                        stmt: 0,
                        return_dest: None,
                        return_block: None,
                    });
                    return Ok(());
                }
            }
        }
        // Glue finished: the dropped place is dead, as in MIR. A
        // direct local of the owner frame is cleared.
        if let RefTarget::Place {
            frame,
            local,
            projection,
        } = &root
        {
            if projection.is_empty() {
                let frame_pos = self.frame_pos(frames, *frame)?;
                if let Some(slot) = frames[frame_pos].locals.get_mut(*local) {
                    *slot = Value::Uninit;
                }
            }
        }
        let owner = frames.last().expect("owner frame");
        let owner_body = self
            .program
            .get(&owner.func)
            .cloned()
            .ok_or_else(|| InterpError::UnknownFunction {
                name: owner.func.clone(),
            })?;
        Self::jump(frames, &owner_body, target)
    }

    fn jump(frames: &mut Vec<Frame>, body: &Body, target: BasicBlock) -> InterpResult<()> {
        if target.0 >= body.blocks.len() {
            return Err(InterpError::UnknownBlock {
                func: body.name.clone(),
                block: target.0,
            });
        }
        let frame = frames.last_mut().expect("frame");
        frame.block = target.0;
        frame.stmt = 0;
        Ok(())
    }

    fn exec_statement(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        statement: &Statement,
    ) -> InterpResult<()> {
        match &statement.kind {
            StatementKind::Assign(place, rvalue) => {
                let dest_ty = body.local_ty(place.local).cloned();
                let value = self.eval_rvalue(frames, body, rvalue, dest_ty.as_ref())?;
                self.write_place(frames, body, place, value)
            }
            StatementKind::SetDiscriminant { place, variant } => {
                let target = self.resolve_place(frames, body, place)?;
                let Resolved::Place {
                    frame_pos,
                    local,
                    projection,
                } = target
                else {
                    return Err(InterpError::type_mismatch(
                        format!("SetDiscriminant on {place}"),
                        "mutable place",
                        "constant aggregate",
                    ));
                };
                let frame = &mut frames[frame_pos];
                let func = frame.func.clone();
                let slot = frame.locals.get_mut(local).ok_or_else(|| {
                    InterpError::UnknownLocal {
                        func: func.clone(),
                        local,
                    }
                })?;
                let place_desc = describe_resolved(local, &projection);
                let value = navigate_mut(slot, &projection, &func, &place_desc)?;
                match value {
                    Value::Coroutine {
                        variant: current, ..
                    } => {
                        *current = *variant;
                        // The state storage is shared between
                        // states: the fields stay in place. The
                        // transform sets the tag of the state it is
                        // entering only after (or before) storing
                        // that state's fields through downcast
                        // projections, and values saved for later
                        // states must survive this tag change.
                        Ok(())
                    }
                    Value::Enum {
                        variant: current,
                        fields,
                    } => {
                        *current = *variant;
                        // The old payload is invalidated, as in MIR.
                        fields.clear();
                        Ok(())
                    }
                    other => Err(InterpError::type_mismatch(
                        format!("SetDiscriminant on {place}"),
                        "ADT value",
                        other.kind_name(),
                    )),
                }
            }
            // StorageLive / StorageDead are liveness hints in rustc MIR;
            // they do not change values. Nop is a no-op by definition.
            StatementKind::StorageLive(_) | StatementKind::StorageDead(_) | StatementKind::Nop => {
                Ok(())
            }
            StatementKind::Unsupported(kind) => Err(InterpError::UnsupportedStatement {
                kind: kind.clone(),
            }),
        }
    }

    // ----- place resolution --------------------------------------------------
    //
    // A place is resolved against the frame stack. `Field`, `Index`,
    // `ConstantIndex` and `Downcast` steps accumulate as constant
    // `RefProj` steps on a (frame, local) pair; a `Deref` step reads
    // the reference value found there and continues at its target.
    // `Index` locals are read in the *executing* frame: a place is
    // syntactic to the function that contains it, so its index locals
    // always live there, even when earlier steps crossed into another
    // frame through a reference.

    pub(crate) fn frame_pos(&self, frames: &[Frame], id: u64) -> InterpResult<usize> {
        frames.iter().position(|f| f.id == id).ok_or_else(|| {
            InterpError::type_mismatch(
                "dereference",
                "reference into a live frame",
                format!("reference into finished frame #{id}"),
            )
        })
    }

    /// Classify a `BoxDerefTransmute` cast operand: when the
    /// operand is a place naming a `Box` or one of its pointer
    /// wrappers (`Unique` / `NonNull`), return the resolved operand
    /// place, the resolved place of the pointee, and the number of
    /// single-field wrapper layers an aggregate-built box value
    /// carries above its pointer (the box itself carries 3 — Box,
    /// Unique, NonNull — and each field already projected away
    /// removes one). Returns `None` for any other operand, whose
    /// cast keeps the strict wrapper-chain reading.
    fn box_cast_places(
        &self,
        frames: &[Frame],
        body: &Body,
        operand: &Operand,
    ) -> Option<(Resolved, Resolved, usize)> {
        let place = match operand {
            Operand::Copy(place) | Operand::Move(place) => place,
            Operand::Constant(_) => return None,
        };
        let Ty::Adt(name) = body.local_ty(place.local)? else {
            return None;
        };
        let segment = name.rsplit("::").next().unwrap_or(name.as_str());
        let is_box = matches!(name.as_str(), "std::boxed::Box" | "alloc::boxed::Box");
        let (depth, pointee_field) = match (segment, place.projection.as_slice()) {
            ("Box", []) if is_box => (3, true),
            ("Box", [ProjectionElem::Field(0)]) if is_box => (2, false),
            ("Unique", []) => (2, false),
            ("NonNull", []) => (1, false),
            _ => return None,
        };
        let operand_target = self.resolve_place(frames, body, place).ok()?;
        let Resolved::Place {
            frame_pos,
            local,
            projection,
        } = operand_target.clone()
        else {
            return None;
        };
        let mut pointee_projection = projection.clone();
        if pointee_field {
            pointee_projection.push(RefProj::Field(0));
        }
        let pointee_target = Resolved::Place {
            frame_pos,
            local,
            projection: pointee_projection,
        };
        Some((operand_target, pointee_target, depth))
    }

    /// The reference value naming a resolved place (the value a
    /// `BoxDerefTransmute` cast yields in the builtin box model).
    fn pointee_ref(frames: &[Frame], target: &Resolved) -> Value {
        match target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => Value::Ref(RefTarget::place(
                frames[*frame_pos].id,
                *local,
                projection.clone(),
            )),
            Resolved::Value(value) => {
                Value::Ref(RefTarget::Value(Box::new(value.clone())))
            }
        }
    }

    pub(crate) fn resolve_place(
        &self,
        frames: &[Frame],
        body: &Body,
        place: &Place,
    ) -> InterpResult<Resolved> {
        let mut cur = Resolved::Place {
            frame_pos: frames.len() - 1,
            local: place.local.0,
            projection: Vec::new(),
        };
        for elem in &place.projection {
            cur = match elem {
                ProjectionElem::Field(idx) => {
                    self.project_step(frames, cur, RefProj::Field(*idx), body, place)?
                }
                ProjectionElem::Index(local) => {
                    let idx = self.read_index_local(frames, body, *local, place)?;
                    self.project_step(frames, cur, RefProj::Index(idx), body, place)?
                }
                ProjectionElem::ConstantIndex { offset, from_end } => {
                    let len = self
                        .read_resolved(frames, &cur)?
                        .len()
                        .ok_or_else(|| {
                            InterpError::type_mismatch(
                                format!("constant index into {place}"),
                                "array, slice or string value",
                                "value without a length",
                            )
                        })?;
                    let idx = if *from_end {
                        len.checked_sub(*offset as u128).ok_or_else(|| {
                            index_oob(len as usize, *offset as usize)
                        })?
                    } else {
                        *offset as u128
                    };
                    let idx = usize::try_from(idx).map_err(|_| {
                        InterpError::panic(format!(
                            "index out of bounds: the len is {len} but the index is {idx}"
                        ))
                    })?;
                    self.project_step(frames, cur, RefProj::Index(idx), body, place)?
                }
                ProjectionElem::Downcast(variant) => {
                    self.project_step(frames, cur, RefProj::Downcast(*variant), body, place)?
                }
                ProjectionElem::Deref => {
                    let value = self.read_resolved(frames, &cur)?;
                    self.deref_value(frames, &value)?
                }
            };
        }
        Ok(cur)
    }

    fn project_step(
        &self,
        frames: &[Frame],
        cur: Resolved,
        step: RefProj,
        body: &Body,
        place: &Place,
    ) -> InterpResult<Resolved> {
        match cur {
            Resolved::Place {
                frame_pos,
                local,
                mut projection,
            } => {
                projection.push(step);
                Ok(Resolved::Place {
                    frame_pos,
                    local,
                    projection,
                })
            }
            Resolved::Value(value) => {
                let func = frames
                    .last()
                    .map(|f| f.func.clone())
                    .unwrap_or_else(|| body.name.clone());
                let projected = project_value(value, &[step], &func, &place.to_string())?;
                Ok(Resolved::Value(projected))
            }
        }
    }

    /// Read an `Index` projection's local in the executing frame.
    fn read_index_local(
        &self,
        frames: &[Frame],
        body: &Body,
        local: Local,
        place: &Place,
    ) -> InterpResult<usize> {
        let frame = frames.last().expect("frame");
        let value = frame
            .locals
            .get(local.0)
            .cloned()
            .ok_or_else(|| InterpError::UnknownLocal {
                func: body.name.clone(),
                local: local.0,
            })?;
        match value {
            Value::Uint(v) => usize::try_from(v).map_err(|_| {
                InterpError::panic(format!("index {v} does not fit in usize"))
            }),
            Value::Int(v) if v >= 0 => usize::try_from(v).map_err(|_| {
                InterpError::panic(format!("index {v} does not fit in usize"))
            }),
            Value::Int(v) => Err(InterpError::panic(format!(
                "index out of bounds: negative index {v} used for {place}"
            ))),
            Value::Uninit => Err(InterpError::UninitializedRead {
                func: body.name.clone(),
                place: place.to_string(),
            }),
            other => Err(InterpError::type_mismatch(
                format!("index of {place}"),
                "integer index",
                other.kind_name(),
            )),
        }
    }

    /// Follow a reference value to what it points at.
    pub(crate) fn deref_value(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Resolved> {
        match value {
            Value::Ref(RefTarget::Place {
                frame,
                local,
                projection,
            }) => Ok(Resolved::Place {
                frame_pos: self.frame_pos(frames, *frame)?,
                local: *local,
                projection: projection.clone(),
            }),
            Value::Ref(RefTarget::Value(v)) => Ok(Resolved::Value((**v).clone())),
            Value::Ref(RefTarget::Null) => Err(InterpError::type_mismatch(
                "dereference",
                "reference",
                "null pointer",
            )),
            // Dereferencing a fat pointer whose base is a reference
            // reaches through to the base place. A self-contained
            // fat pointer (a slice constant: the base is the element
            // aggregate itself) derefs to the slice value itself, so
            // `Len` and index bounds see the fat pointer's length.
            Value::FatRef { base, .. } => match &**base {
                Value::Tuple(_) => Ok(Resolved::Value(value.clone())),
                _ => self.deref_value(frames, base),
            },
            // Dereferencing a dyn fat pointer reaches the value it
            // was formed over: through the base pointer for a
            // reference or slice base, and to the held pointee for
            // a box base (the box aggregate itself — exactly what
            // `Deref::deref` for `Box` yields).
            Value::DynRef { base, .. } => match &**base {
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    Ok(Resolved::Value(fields[0].clone()))
                }
                _ => self.deref_value(frames, base),
            },
            other => Err(InterpError::type_mismatch(
                "dereference",
                "reference",
                other.kind_name(),
            )),
        }
    }

    pub(crate) fn read_resolved(
        &self,
        frames: &[Frame],
        target: &Resolved,
    ) -> InterpResult<Value> {
        match target {
            Resolved::Value(value) => {
                if *value == Value::Uninit {
                    return Err(InterpError::UninitializedRead {
                        func: "<constant>".to_string(),
                        place: "constant aggregate".to_string(),
                    });
                }
                Ok(value.clone())
            }
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                let frame = &frames[*frame_pos];
                let base = frame.locals.get(*local).cloned().ok_or_else(|| {
                    InterpError::UnknownLocal {
                        func: frame.func.clone(),
                        local: *local,
                    }
                })?;
                // A zero-sized local's place may never be written
                // by rustc MIR at all (a unit struct constructor's
                // body is a bare `return`): an unwritten read of
                // such a local denotes the type's single value.
                let base = if base == Value::Uninit {
                    self.zst_local_value(frame, *local).unwrap_or(base)
                } else {
                    base
                };
                let place_desc = describe_resolved(*local, projection);
                let value = project_value(base, projection, &frame.func, &place_desc)?;
                if value == Value::Uninit {
                    return Err(InterpError::UninitializedRead {
                        func: frame.func.clone(),
                        place: place_desc,
                    });
                }
                Ok(value)
            }
        }
    }

    /// The value of a never-written zero-sized local in `frame`
    /// (see the note in `read_resolved`); `None` for any other
    /// local. Conversion records the canonical value of every
    /// provably zero-sized local in the body (`Body::zst_locals`,
    /// computed from the driver's ADT shape tables); for
    /// hand-built bodies the shapes derivable from the IR type
    /// alone are synthesized by [`zst_value_of_ty`].
    fn zst_local_value(&self, frame: &Frame, local: usize) -> Option<Value> {
        let body = self.program.get(&frame.func)?;
        if let Some(Some(v)) = body.zst_locals.get(local) {
            return Some(v.clone());
        }
        zst_value_of_ty(body.local_ty(Local(local))?)
    }

    pub(crate) fn write_resolved(
        &self,
        frames: &mut Vec<Frame>,
        target: &Resolved,
        value: Value,
    ) -> InterpResult<()> {
        match target {
            Resolved::Value(_) => Err(InterpError::type_mismatch(
                "assignment",
                "mutable place",
                "constant aggregate",
            )),
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => {
                if projection.is_empty() {
                    // Direct local of the target frame: normalize
                    // against that frame's declared type.
                    let func = frames[*frame_pos].func.clone();
                    let target_body = self.program.get(&func).cloned().ok_or_else(|| {
                        InterpError::UnknownFunction { name: func.clone() }
                    })?;
                    let frame = &mut frames[*frame_pos];
                    let slot = frame.locals.get_mut(*local).ok_or_else(|| {
                        InterpError::UnknownLocal {
                            func: func.clone(),
                            local: *local,
                        }
                    })?;
                    *slot = Self::normalize_for_ty(value, target_body.local_ty(Local(*local)))?;
                    return Ok(());
                }
                let frame = &mut frames[*frame_pos];
                let func = frame.func.clone();
                let place_desc = describe_resolved(*local, projection);
                let slot = frame.locals.get_mut(*local).ok_or_else(|| {
                    InterpError::UnknownLocal {
                        func: func.clone(),
                        local: *local,
                    }
                })?;
                let slot = navigate_mut(slot, projection, &func, &place_desc)?;
                *slot = value;
                Ok(())
            }
        }
    }

    /// The live place a `&RefCell` / `&mut RefCell` argument points
    /// at. Borrowing needs a place: the flag lives inside the cell's
    /// storage and must be written back, which a constant aggregate
    /// (`RefTarget::Value`) cannot do — that case is an explicit
    /// unsupported error, never a silent no-op.
    pub(crate) fn refcell_place(&self, value: &Value, name: &str) -> InterpResult<RefTarget> {
        match value {
            Value::Ref(target @ RefTarget::Place { .. }) => Ok(target.clone()),
            Value::Ref(_) => Err(InterpError::UnsupportedTerminator {
                kind: format!(
                    "runtime support: {name} on a RefCell with no live place \
                     (constant aggregate) cannot track borrows"
                ),
            }),
            other => Err(InterpError::type_mismatch(
                name,
                "reference to RefCell",
                other.kind_name(),
            )),
        }
    }

    /// Read the borrow flag of the `Cell` aggregate `flag_cell`
    /// points at: 0 = unused, -1 = mutably borrowed, n > 0 = n
    /// shared borrows.
    pub(crate) fn refcell_flag(
        &self,
        frames: &[Frame],
        flag_cell: &RefTarget,
    ) -> InterpResult<i128> {
        let resolved = self.deref_value(frames, &Value::Ref(flag_cell.clone()))?;
        let cell = self.read_resolved(frames, &resolved)?;
        match cell {
            Value::Enum { fields, .. } if fields.len() == 1 => match &fields[0] {
                Value::Enum { fields: inner, .. } if inner.len() == 1 => match inner[0] {
                    Value::Int(flag) => Ok(flag),
                    // A flag materialised from a constant whose
                    // signed context was lost arrives as an
                    // unsigned word; the flag values (0, -1,
                    // small counts) reinterpret exactly.
                    Value::Uint(flag) => Ok(flag as i128),
                    ref other => Err(InterpError::type_mismatch(
                        "RefCell borrow flag",
                        "isize flag",
                        other.kind_name(),
                    )),
                },
                other => Err(InterpError::type_mismatch(
                    "RefCell borrow flag",
                    "UnsafeCell aggregate",
                    other.kind_name(),
                )),
            },
            ref other => Err(InterpError::type_mismatch(
                "RefCell borrow flag",
                "Cell aggregate",
                other.kind_name(),
            )),
        }
    }

    /// Write the borrow flag of the `Cell` aggregate `flag_cell`
    /// points at, rebuilding the `Cell { UnsafeCell { flag } }`
    /// aggregate in place.
    pub(crate) fn refcell_set_flag(
        &mut self,
        frames: &mut Vec<Frame>,
        flag_cell: &RefTarget,
        flag: i128,
    ) -> InterpResult<()> {
        let resolved = self.deref_value(frames, &Value::Ref(flag_cell.clone()))?;
        let cell = Value::Enum {
            variant: 0,
            fields: vec![Value::Enum {
                variant: 0,
                fields: vec![Value::Int(flag)],
            }],
        };
        self.write_resolved(frames, &resolved, cell)
    }

    /// Release the borrow held by a `Ref` / `RefMut` guard value:
    /// field 1 is the reference to the cell's borrow-flag `Cell`.
    /// A shared release decrements the count; a mutable release
    /// resets the flag to unused (0).
    pub(crate) fn release_refcell_borrow(
        &mut self,
        frames: &mut Vec<Frame>,
        guard: &Value,
        mutable: bool,
    ) -> InterpResult<()> {
        let flag_ref = match guard {
            Value::Enum { fields, .. } if fields.len() == 2 => &fields[1],
            other => {
                return Err(InterpError::type_mismatch(
                    "RefCell guard drop",
                    "Ref/RefMut guard aggregate",
                    other.kind_name(),
                ))
            }
        };
        let flag_target = match flag_ref {
            Value::Ref(target) => target.clone(),
            other => {
                return Err(InterpError::type_mismatch(
                    "RefCell guard drop",
                    "reference to the borrow-flag Cell",
                    other.kind_name(),
                ))
            }
        };
        let flag = self.refcell_flag(frames, &flag_target)?;
        let next = if mutable { 0 } else { flag - 1 };
        self.refcell_set_flag(frames, &flag_target, next)
    }

    pub(crate) fn read_place(
        &self,
        frames: &[Frame],
        body: &Body,
        place: &Place,
    ) -> InterpResult<Value> {
        let target = self.resolve_place(frames, body, place)?;
        self.read_resolved(frames, &target)
    }

    pub(crate) fn write_place(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        place: &Place,
        value: Value,
    ) -> InterpResult<()> {
        let target = self.resolve_place(frames, body, place)?;
        self.write_resolved(frames, &target, value)
    }

    /// Validate / wrap a value being stored into a typed direct local.
    fn normalize_for_ty(value: Value, ty: Option<&Ty>) -> InterpResult<Value> {
        match (ty, &value) {
            (Some(Ty::Bool), Value::Bool(_))
            | (Some(Ty::Unit), Value::Unit)
            | (Some(Ty::Tuple(_)), Value::Tuple(_))
            | (None, _) => Ok(value),
            (Some(Ty::Int(it)), Value::Int(v)) => Ok(Value::Int(wrap_signed(*v, it.bits()))),
            (Some(Ty::Uint(ut)), Value::Uint(v)) => Ok(Value::Uint(*v & ut.max())),
            // An `f32` place holds the `f32` rounding of the stored
            // (exactly widened) value; an `f64` place holds it as-is.
            (Some(Ty::Float(FloatTy::F32)), Value::Float(v)) => {
                Ok(Value::Float((*v as f32) as f64))
            }
            (Some(Ty::Float(FloatTy::F64)), Value::Float(_)) => Ok(value),
            // An integer payload whose kind tag differs from the
            // declared type: the bits are reinterpreted at the
            // declared width. This arises for scalar leaves of
            // structured constants, whose signedness is not stored
            // in the constant itself; MIR places and their values
            // always agree on width, so the declared type is the
            // exact interpretation.
            (Some(Ty::Int(it)), Value::Uint(v)) => {
                Ok(Value::Int(wrap_signed(*v as i128, it.bits())))
            }
            (Some(Ty::Uint(ut)), Value::Int(v)) => Ok(Value::Uint((*v as u128) & ut.max())),
            // Callable locals: a function-pointer place holds a
            // function item value or a (non-capturing) closure value;
            // a function-item place holds the item value; a closure
            // place holds the closure value.
            (Some(Ty::FnPtr), Value::FnPtr(_))
            | (Some(Ty::FnPtr), Value::Closure { .. })
            | (Some(Ty::FnDef(_)), Value::FnPtr(_))
            | (Some(Ty::Closure(..)), Value::Closure { .. }) => Ok(value),
            // ADT locals hold the ADT value forms: aggregates, string
            // data (`String`), and the fmt value family.
            (Some(Ty::Adt(_)), Value::Enum { .. })
            | (Some(Ty::Adt(_)), Value::Str(_))
            // A smart-pointer ADT over a `dyn` pointee (`Box<dyn
            // Trait>`, `Rc<dyn Trait>`) holds the fat pointer.
            | (Some(Ty::Adt(_)), Value::DynRef { .. })
            | (Some(Ty::Adt(_)), Value::FmtArg { .. })
            | (Some(Ty::Adt(_)), Value::FmtCount(_))
            | (Some(Ty::Adt(_)), Value::FmtArguments(_)) => Ok(value),
            // Reference locals hold resolved references, fat pointers,
            // and — for `&str` and byte-array references — the string
            // or byte data itself (a string constant *is* its value;
            // a reference to a string place is a `Ref`).
            (Some(Ty::Ref(_)), Value::Ref(_))
            | (Some(Ty::Ref(_)), Value::FatRef { .. })
            | (Some(Ty::Ref(_)), Value::DynRef { .. })
            | (Some(Ty::Ref(_)), Value::Str(_))
            | (Some(Ty::Ref(_)), Value::Bytes(_)) => Ok(value),
            (Some(Ty::Str), Value::Str(_)) => Ok(value),
            (Some(Ty::Array(_, n)), Value::Tuple(items)) if items.len() as u64 == *n => Ok(value),
            (Some(Ty::Array(elem, n)), Value::Bytes(b))
                if **elem == Ty::Uint(UintTy::U8) && b.len() as u64 == *n =>
            {
                Ok(value)
            }
            (Some(Ty::Slice(_)), Value::FatRef { .. }) => Ok(value),
            // Coroutine locals hold the state object itself. (A
            // coroutine value also reaches places typed as the ADT
            // wrappers around it - `Box`, `Pin` - as a field of the
            // wrapper's aggregate value, which is checked as the
            // wrapper, not here.)
            (Some(Ty::Coroutine(_)), Value::Coroutine { .. }) => Ok(value),
            (Some(expected), other) => Err(InterpError::type_mismatch(
                "assignment",
                format!("{expected:?}"),
                other.kind_name(),
            )),
        }
    }

    pub(crate) fn eval_operand(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        operand: &Operand,
    ) -> InterpResult<Value> {
        match operand {
            Operand::Copy(place) => self.read_place(frames, body, place),
            Operand::Move(place) => {
                let value = self.read_place(frames, body, place)?;
                if place.is_direct_local() {
                    let frame = frames.last_mut().expect("frame");
                    if let Some(slot) = frame.locals.get_mut(place.local.0) {
                        *slot = Value::Uninit;
                    }
                } else if let Ok(target) = self.resolve_place(frames, body, place) {
                    // A move out of a projected place leaves that
                    // part uninitialized too, so a later `Drop` of
                    // the whole value does not run the moved-out
                    // part's glue a second time (rustc's drop flags
                    // track exactly this).
                    if matches!(target, Resolved::Place { .. }) {
                        self.write_resolved(frames, &target, Value::Uninit)?;
                    }
                }
                Ok(value)
            }
            Operand::Constant(c) => self.eval_constant(c),
        }
    }

    /// Evaluate a constant to a value. Structured constants recurse:
    /// a reference constant becomes a reference to the constant value
    /// it points at, a slice constant becomes a fat reference over
    /// its element tuple.
    fn eval_constant(&self, c: &Constant) -> InterpResult<Value> {
        Ok(match c {
            Constant::Unit => Value::Unit,
            Constant::Bool(b) => Value::Bool(*b),
            Constant::Int(v) => Value::Int(*v),
            Constant::Uint(v) => Value::Uint(*v),
            Constant::Float(v) => Value::Float(*v),
            Constant::FnDef(name) => Value::FnPtr(name.clone()),
            Constant::Closure(def) => Value::Closure {
                def: def.clone(),
                env: Vec::new(),
            },
            Constant::Str(s) => Value::Str(s.clone()),
            Constant::Bytes(b) => Value::Bytes(b.clone()),
            Constant::Tuple(fields) => Value::Tuple(
                fields
                    .iter()
                    .map(|f| self.eval_constant(f))
                    .collect::<InterpResult<Vec<_>>>()?,
            ),
            Constant::Adt { variant, fields } => Value::Enum {
                variant: *variant,
                fields: fields
                    .iter()
                    .map(|f| self.eval_constant(f))
                    .collect::<InterpResult<Vec<_>>>()?,
            },
            Constant::Ref(inner) => {
                Value::Ref(RefTarget::Value(Box::new(self.eval_constant(inner)?)))
            }
            Constant::SliceVal(elems) => {
                let elems = elems
                    .iter()
                    .map(|e| self.eval_constant(e))
                    .collect::<InterpResult<Vec<_>>>()?;
                let len = elems.len() as u128;
                Value::FatRef {
                    base: Box::new(Value::Tuple(elems)),
                    len,
                }
            }
        })
    }

    // ----- rvalues -----------------------------------------------------------

    fn eval_rvalue(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        rvalue: &Rvalue,
        dest_ty: Option<&Ty>,
    ) -> InterpResult<Value> {
        match rvalue {
            Rvalue::Use(op) => self.eval_operand(frames, body, op),
            Rvalue::BinaryOp(op, pair) => {
                let lhs = self.eval_operand(frames, body, &pair.0)?;
                let rhs = self.eval_operand(frames, body, &pair.1)?;
                eval_binary(*op, &lhs, &rhs, dest_ty, false).map(|(v, _)| v)
            }
            Rvalue::CheckedBinaryOp(op, pair) => {
                let lhs = self.eval_operand(frames, body, &pair.0)?;
                let rhs = self.eval_operand(frames, body, &pair.1)?;
                let (v, overflowed) = eval_binary(*op, &lhs, &rhs, dest_ty, true)?;
                Ok(Value::Tuple(vec![v, Value::Bool(overflowed)]))
            }
            Rvalue::UnaryOp(UnOp::PtrMetadata, operand) => {
                let v = self.eval_operand(frames, body, operand)?;
                match &v {
                    Value::FatRef { len, .. } => Ok(Value::Uint(*len)),
                    Value::Str(s) => Ok(Value::Uint(s.len() as u128)),
                    other => Err(InterpError::type_mismatch(
                        "PtrMetadata",
                        "fat pointer",
                        other.kind_name(),
                    )),
                }
            }
            Rvalue::UnaryOp(op, operand) => {
                let v = self.eval_operand(frames, body, operand)?;
                eval_unary(*op, &v, dest_ty)
            }
            Rvalue::AggregateTuple(operands) => {
                let mut items = Vec::with_capacity(operands.len());
                for op in operands {
                    items.push(self.eval_operand(frames, body, op)?);
                }
                Ok(Value::Tuple(items))
            }
            Rvalue::AggregateAdt {
                variant,
                operands,
                field_index,
            } => {
                // A union aggregate's single operand initializes the
                // active field; the slots before it stay
                // uninitialized, exactly as a field write through a
                // projection leaves the slots it skipped over (see
                // `navigate_mut`'s coroutine arm). Reading the active
                // field then finds the stored value, and reading any
                // other field is an uninitialized read, as it should
                // be: the model does not reinterpret a union's bytes.
                let mut fields = match field_index {
                    Some(active) => vec![Value::Uninit; *active as usize],
                    None => Vec::with_capacity(operands.len()),
                };
                for op in operands {
                    fields.push(self.eval_operand(frames, body, op)?);
                }
                Ok(Value::Enum {
                    variant: *variant,
                    fields,
                })
            }
            Rvalue::ClosureAggregate { def, operands } => {
                let mut env = Vec::with_capacity(operands.len());
                for op in operands {
                    env.push(self.eval_operand(frames, body, op)?);
                }
                Ok(Value::Closure {
                    def: def.clone(),
                    env,
                })
            }
            Rvalue::CoroutineAggregate(operands) => {
                let mut fields = Vec::with_capacity(operands.len());
                for op in operands {
                    fields.push(self.eval_operand(frames, body, op)?);
                }
                Ok(Value::Coroutine {
                    variant: 0,
                    fields,
                })
            }
            Rvalue::Cast(operand, ty) => {
                let v = self.eval_operand(frames, body, operand)?;
                eval_cast(&v, ty)
            }
            Rvalue::Unsize(operand, meta) => {
                let v = self.eval_operand(frames, body, operand)?;
                match meta {
                    UnsizeMeta::Slice => match v {
                        Value::Ref(_) => {
                            let target = self.deref_value(frames, &v)?;
                            let pointee = self.read_resolved(frames, &target)?;
                            let len = pointee.len().ok_or_else(|| {
                                InterpError::type_mismatch(
                                    "unsizing cast",
                                    "array pointee",
                                    pointee.kind_name(),
                                )
                            })?;
                            Ok(Value::FatRef {
                                base: Box::new(v),
                                len,
                            })
                        }
                        Value::Bytes(b) => {
                            let len = b.len() as u128;
                            Ok(Value::FatRef {
                                base: Box::new(Value::Bytes(b)),
                                len,
                            })
                        }
                        other => Err(InterpError::type_mismatch(
                            "unsizing cast",
                            "reference to an array",
                            other.kind_name(),
                        )),
                    },
                    UnsizeMeta::Dyn { concrete, traits } => match v {
                        // A supertrait upcast: the data pointer and
                        // the concrete type carry over; only the
                        // bounds the pointer answers to change.
                        Value::DynRef {
                            base,
                            concrete: existing,
                            ..
                        } => Ok(Value::DynRef {
                            base,
                            concrete: existing,
                            traits: traits.clone(),
                        }),
                        Value::Ref(_) | Value::FatRef { .. } | Value::Enum { .. } => {
                            let concrete = concrete.clone().ok_or_else(|| {
                                InterpError::UnsupportedRvalue {
                                    kind: "Unsize to dyn (no concrete type recorded)"
                                        .to_string(),
                                }
                            })?;
                            Ok(Value::DynRef {
                                base: Box::new(v),
                                concrete,
                                traits: traits.clone(),
                            })
                        }
                        other => Err(InterpError::type_mismatch(
                            "unsizing cast to dyn",
                            "reference, fat pointer or box value",
                            other.kind_name(),
                        )),
                    },
                }
            }
            Rvalue::PtrToInt(operand) => {
                let value = self.eval_operand(frames, body, operand)?;
                match synthetic_address(&value) {
                    Some(addr) => Ok(Value::Uint(addr)),
                    None => Err(InterpError::TypeMismatch {
                        context: "pointer-to-integer transmute".to_string(),
                        expected: "a pointer value".to_string(),
                        found: value.kind_name().to_string(),
                    }),
                }
            }
            Rvalue::BoxDerefTransmute(operand) => {
                // The cast yields a pointer to the box's pointee.
                // Two box shapes exist in this value model. A box
                // built from its MIR aggregates is the wrapper chain
                // `Unique` around `NonNull` around the pointer, and
                // unwrapping the chain reaches the pointer value
                // itself. A box made by the `Box::new` builtin instead
                // holds its pointee directly as its single field, so
                // the chain bottoms out at the pointee value (or at a
                // reference that *is* the pointee, when the pointee
                // is itself a reference); the result is then a
                // reference to the pointee's place, which is what
                // reading or writing through the extracted pointer
                // must reach. `box_cast_places` tells the two apart:
                // it reports how many wrapper layers an
                // aggregate-built value carries above its pointer.
                let places = self.box_cast_places(frames, body, operand);
                let original = self.eval_operand(frames, body, operand)?;
                let mut v = original.clone();
                let mut layers = 0usize;
                loop {
                    match v {
                        Value::DynRef { .. } => {
                            break Ok(v)
                        }
                        Value::Ref(_) | Value::FatRef { .. } => {
                            match &places {
                                Some((_, pointee, depth)) if layers < *depth => {
                                    break Ok(Self::pointee_ref(
                                        frames, pointee,
                                    ));
                                }
                                _ => break Ok(v),
                            }
                        }
                        Value::Enum { mut fields, .. } if fields.len() == 1 => {
                            v = fields.pop().expect("one field");
                            layers += 1;
                        }
                        other => {
                            // Builtin-built boxes whose operand the
                            // classification covers: it names the
                            // pointee place. A move out of the operand
                            // place cleared the storage the pointer
                            // must name, so the value is restored
                            // first. (Lane BH)
                            if let Some((source, pointee, _)) = &places {
                                if matches!(operand, Operand::Move(_)) {
                                    self.write_resolved(
                                        frames,
                                        source,
                                        original.clone(),
                                    )?;
                                }
                                break Ok(Self::pointee_ref(frames, pointee));
                            }
                            // A box built by the `Box::new` /
                            // `Box::new_uninit` builtins stores its
                            // pointee as the field value itself, not
                            // as a pointer chain. When the cast's
                            // operand place projects into the box, the
                            // pointer it stands for is a reference to
                            // that place; a resolved constant value is
                            // referenced as a value. (Lane BO)
                            if let Operand::Copy(place) | Operand::Move(place) = operand {
                                match self.resolve_place(frames, body, place)? {
                                    Resolved::Place {
                                        frame_pos,
                                        local,
                                        projection,
                                    } if !projection.is_empty() => {
                                        break Ok(Value::Ref(RefTarget::place(
                                            frames[frame_pos].id,
                                            local,
                                            projection,
                                        )));
                                    }
                                    Resolved::Value(value) => {
                                        break Ok(Value::Ref(RefTarget::Value(Box::new(
                                            value,
                                        ))));
                                    }
                                    Resolved::Place { .. } => {}
                                }
                            }
                            break Err(InterpError::type_mismatch(
                                "BoxDerefTransmute cast",
                                "pointer wrapper (Unique/NonNull around a raw pointer)",
                                other.kind_name(),
                            ));
                        }
                    }
                }
            }
            Rvalue::Ref(place) => {
                // A borrow of exactly the place a dyn fat pointer
                // points at (`&*r`, `&*box`) is the fat pointer
                // itself: the result is again a `&dyn`, with the
                // same data pointer and the same vtable identity.
                if place.projection.len() == 1
                    && matches!(place.projection[0], ProjectionElem::Deref)
                {
                    let head = Place {
                        local: place.local,
                        projection: Vec::new(),
                    };
                    if let Ok(value @ Value::DynRef { .. }) =
                        self.read_place(frames, body, &head)
                    {
                        return Ok(value);
                    }
                }
                let target = self.resolve_place(frames, body, place)?;
                match target {
                    Resolved::Place {
                        frame_pos,
                        local,
                        projection,
                    } => Ok(Value::Ref(RefTarget::place(
                        frames[frame_pos].id,
                        local,
                        projection,
                    ))),
                    Resolved::Value(value) => Ok(Value::Ref(RefTarget::Value(Box::new(value)))),
                }
            }
            Rvalue::Len(place) => {
                let value = self.read_place(frames, body, place)?;
                let len = value.len().ok_or_else(|| {
                    InterpError::type_mismatch(
                        format!("Len of {place}"),
                        "array, slice or string value",
                        value.kind_name(),
                    )
                })?;
                Ok(Value::Uint(len))
            }
            Rvalue::Discriminant(place, discriminants) => {
                let value = self.read_place(frames, body, place)?;
                match value {
                    Value::Enum { variant, .. } | Value::Coroutine { variant, .. } => {
                        if discriminants.is_empty() {
                            // No table was recorded for this enum:
                            // its discriminants are the sequential
                            // defaults, so the index is the value.
                            Ok(Value::Int(variant as i128))
                        } else {
                            let discr = discriminants
                                .get(variant as usize)
                                .ok_or_else(|| {
                                    InterpError::type_mismatch(
                                        format!("Discriminant of {place}"),
                                        "variant covered by the discriminant table",
                                        format!("variant {variant}"),
                                    )
                                })?;
                            Ok(Value::Int(*discr))
                        }
                    }
                    other => Err(InterpError::type_mismatch(
                        format!("Discriminant of {place}"),
                        "ADT value",
                        other.kind_name(),
                    )),
                }
            }
            Rvalue::CopyForDeref(place) => {
                let value = self.read_place(frames, body, place)?;
                match &value {
                    Value::Ref(_) | Value::FatRef { .. } => {
                        let target = self.deref_value(frames, &value)?;
                        self.read_resolved(frames, &target)
                    }
                    _ => Ok(value),
                }
            }
            Rvalue::Unsupported(kind) => Err(InterpError::UnsupportedRvalue {
                kind: kind.clone(),
            }),
        }
    }
}

/// Human-readable form of a resolved place, for error messages.
pub(crate) fn describe_resolved(local: usize, projection: &[RefProj]) -> String {
    let mut out = format!("_{local}");
    for proj in projection {
        match proj {
            RefProj::Field(i) => out.push_str(&format!(".{i}")),
            RefProj::Index(i) => out.push_str(&format!("[{i}]")),
            RefProj::Downcast(v) => out.push_str(&format!(" as variant {v}")),
        }
    }
    out
}

/// The guest panic rustc raises for an out-of-bounds index.
fn index_oob(len: usize, index: usize) -> InterpError {
    InterpError::panic(format!(
        "index out of bounds: the len is {len} but the index is {index}"
    ))
}

/// Apply constant projection steps to a value (read direction).
/// The canonical value of a type that is zero-sized by its IR
/// shape alone (see `Interpreter::zst_local_value`): unit, tuples
/// of zero-sized types, arrays of zero-sized elements and empty
/// arrays, zero-capture closures, function items. ADT values need
/// the driver's shape tables and are only synthesized at
/// conversion time.
fn zst_value_of_ty(ty: &Ty) -> Option<Value> {
    match ty {
        Ty::Unit => Some(Value::Unit),
        Ty::Tuple(ts) => {
            let mut vals = Vec::with_capacity(ts.len());
            for t in ts {
                vals.push(zst_value_of_ty(t)?);
            }
            Some(Value::Tuple(vals))
        }
        Ty::Array(_, 0) => Some(Value::Tuple(Vec::new())),
        Ty::Array(elem, n) if *n <= 4096 => {
            let v = zst_value_of_ty(elem)?;
            Some(Value::Tuple(vec![v; *n as usize]))
        }
        Ty::Closure(def, 0) => Some(Value::Closure {
            def: def.clone(),
            env: Vec::new(),
        }),
        Ty::FnDef(name) => Some(Value::FnPtr(name.clone())),
        _ => None,
    }
}

pub(crate) fn project_value(
    value: Value,
    projection: &[RefProj],
    func: &str,
    place: &str,
) -> InterpResult<Value> {
    let mut value = value;
    for step in projection {
        if value == Value::Uninit {
            return Err(InterpError::UninitializedRead {
                func: func.to_string(),
                place: place.to_string(),
            });
        }
        value = match step {
            RefProj::Field(idx) => match value {
                Value::Tuple(ref items) => items.get(*idx).cloned().ok_or_else(|| {
                    InterpError::UninitializedRead {
                        func: func.to_string(),
                        place: place.to_string(),
                    }
                })?,
                Value::Enum { ref fields, .. } => {
                    fields.get(*idx).cloned().ok_or_else(|| {
                        InterpError::UninitializedRead {
                            func: func.to_string(),
                            place: place.to_string(),
                        }
                    })?
                }
                Value::Coroutine { ref fields, .. } => {
                    fields.get(*idx).cloned().ok_or_else(|| {
                        InterpError::UninitializedRead {
                            func: func.to_string(),
                            place: place.to_string(),
                        }
                    })?
                }
                Value::Closure { ref env, .. } => {
                    env.get(*idx).cloned().ok_or_else(|| {
                        InterpError::UninitializedRead {
                            func: func.to_string(),
                            place: place.to_string(),
                        }
                    })?
                }
                // Field 0 of a fat pointer value is the pointer
                // field of its smart-pointer wrapper (a `Box`'s
                // `Unique`): in this model the `DynRef` *is* that
                // pointer, metadata included.
                Value::DynRef { .. } if *idx == 0 => value,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        format!("field projection on {place}"),
                        "tuple or ADT value",
                        other.kind_name(),
                    ));
                }
            },
            RefProj::Index(idx) => match value {
                Value::FatRef {
                    ref base,
                    len,
                } => {
                    if *idx as u128 >= len {
                        return Err(index_oob(len as usize, *idx));
                    }
                    project_value((**base).clone(), &[RefProj::Index(*idx)], func, place)?
                }
                Value::Tuple(ref items) => items
                    .get(*idx)
                    .cloned()
                    .ok_or_else(|| index_oob(items.len(), *idx))?,
                Value::Bytes(ref b) => b
                    .get(*idx)
                    .map(|byte| Value::Uint(*byte as u128))
                    .ok_or_else(|| index_oob(b.len(), *idx))?,
                Value::Str(ref s) => s
                    .as_bytes()
                    .get(*idx)
                    .map(|byte| Value::Uint(*byte as u128))
                    .ok_or_else(|| index_oob(s.len(), *idx))?,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        format!("index projection on {place}"),
                        "array, byte or string value",
                        other.kind_name(),
                    ));
                }
            },
            RefProj::Downcast(variant) => match value {
                // A coroutine's downcast names the state whose field
                // layout the access uses; the storage is shared
                // between states, so the access is valid whatever
                // the current tag is (the body writes a state's
                // fields before setting the tag to that state).
                Value::Coroutine { .. } => value,
                Value::Enum {
                    variant: current, ..
                } => {
                    if current != *variant {
                        return Err(InterpError::type_mismatch(
                            format!("downcast of {place}"),
                            format!("variant {variant}"),
                            format!("variant {current}"),
                        ));
                    }
                    value
                }
                Value::Tuple(_) if *variant == 0 => value,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        format!("downcast of {place}"),
                        "ADT value",
                        other.kind_name(),
                    ));
                }
            },
        };
    }
    Ok(value)
}

/// Navigate a projection mutably inside a stored value and return the
/// final slot (write direction).
pub(crate) fn navigate_mut<'x>(
    slot: &'x mut Value,
    projection: &[RefProj],
    func: &str,
    place: &str,
) -> InterpResult<&'x mut Value> {
    if projection.is_empty() {
        return Ok(slot);
    }
    if *slot == Value::Uninit {
        return Err(InterpError::UninitializedRead {
            func: func.to_string(),
            place: place.to_string(),
        });
    }
    let (step, rest) = projection.split_first().expect("non-empty");
    match step {
        RefProj::Field(idx) => {
            if let Value::Coroutine { fields, .. } = slot {
                // The state storage grows to the written slot: a
                // state field stored for the first time extends the
                // storage, and slots skipped over are uninitialized,
                // like a field that has not been stored yet.
                if fields.len() <= *idx {
                    fields.resize(*idx + 1, Value::Uninit);
                }
                return navigate_mut(&mut fields[*idx], rest, func, place);
            }
            // A union value's field vector is sparse: it covers
            // only the fields written so far (see the union arm of
            // aggregate evaluation). Writing a further field extends
            // it, leaving the skipped slots uninitialized — the same
            // growth the coroutine arm above performs. Struct and
            // enum values are always fully populated, so the growth
            // never triggers for them.
            if let Value::Enum { fields, .. } = slot {
                if fields.len() <= *idx {
                    fields.resize(*idx + 1, Value::Uninit);
                }
            }
            let items = match slot {
                Value::Tuple(items) => items,
                Value::Enum { fields, .. } => fields,
                Value::Closure { env, .. } => env,
                other => {
                    return Err(InterpError::type_mismatch(
                        format!("field write on {place}"),
                        "tuple or ADT value",
                        other.kind_name(),
                    ));
                }
            };
            let item = items.get_mut(*idx).ok_or_else(|| InterpError::UninitializedRead {
                func: func.to_string(),
                place: place.to_string(),
            })?;
            navigate_mut(item, rest, func, place)
        }
        RefProj::Index(idx) => match slot {
            Value::Tuple(items) => {
                if *idx >= items.len() {
                    return Err(index_oob(items.len(), *idx));
                }
                navigate_mut(&mut items[*idx], rest, func, place)
            }
            Value::Bytes(b) => {
                if *idx >= b.len() {
                    return Err(index_oob(b.len(), *idx));
                }
                // Byte payloads come from constants and have no
                // `Value` slot to hand out; array locals built by
                // aggregates are tuples and take the arm above.
                Err(InterpError::type_mismatch(
                    format!("index write on {place}"),
                    "mutable array slot",
                    "constant bytes",
                ))
            }
            other => Err(InterpError::type_mismatch(
                format!("index write on {place}"),
                "array value",
                other.kind_name(),
            )),
        },
        RefProj::Downcast(variant) => match slot {
            // See the read direction: a coroutine's field writes go
            // through the downcast of the state being entered,
            // whatever the current tag is.
            Value::Coroutine { .. } => navigate_mut(slot, rest, func, place),
            Value::Enum {
                variant: current, ..
            } => {
                if current != variant {
                    return Err(InterpError::type_mismatch(
                        format!("downcast write on {place}"),
                        format!("variant {variant}"),
                        format!("variant {current}"),
                    ));
                }
                navigate_mut(slot, rest, func, place)
            }
            Value::Tuple(_) if *variant == 0 => navigate_mut(slot, rest, func, place),
            other => Err(InterpError::type_mismatch(
                format!("downcast write on {place}"),
                "ADT value",
                other.kind_name(),
            )),
        },
    }
}

// ----- arithmetic helpers ----------------------------------------------------

fn wrap_signed(v: i128, bits: u32) -> i128 {
    if bits >= 128 {
        return v;
    }
    let shift = 128 - bits;
    (v << shift) >> shift
}

fn wrap_unsigned(v: u128, bits: u32) -> u128 {
    if bits >= 128 {
        v
    } else {
        v & ((1u128 << bits) - 1)
    }
}

fn signed_bits(dest_ty: Option<&Ty>) -> u32 {
    match dest_ty {
        Some(Ty::Int(it)) => it.bits(),
        // A checked operation's destination is the `(T, bool)` tuple;
        // the operand width lives in its first element.
        Some(Ty::Tuple(ts)) => signed_bits(ts.first()),
        _ => 128,
    }
}

fn unsigned_bits(dest_ty: Option<&Ty>) -> u32 {
    match dest_ty {
        Some(Ty::Uint(ut)) => ut.bits(),
        Some(Ty::Tuple(ts)) => unsigned_bits(ts.first()),
        _ => 128,
    }
}

/// Evaluate a binary operation.
/// Returns `(result, overflowed)`. In non-checked mode an overflow (or a
/// division by zero / invalid shift) is a guest panic instead of a flag.
/// Comparison and bitwise-on-bool operations never overflow.
fn eval_binary(
    op: BinOp,
    lhs: &Value,
    rhs: &Value,
    dest_ty: Option<&Ty>,
    checked_form: bool,
) -> InterpResult<(Value, bool)> {
    use BinOp::*;
    // Boolean operations.
    if let (Value::Bool(a), Value::Bool(b)) = (lhs, rhs) {
        let v = match op {
            BitAnd => Value::Bool(*a && *b),
            BitOr => Value::Bool(*a || *b),
            BitXor => Value::Bool(*a ^ *b),
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            _ => {
                return Err(InterpError::type_mismatch(
                    format!("binary operation {op:?}"),
                    "integer operands",
                    "bool",
                ))
            }
        };
        return Ok((v, false));
    }
    // String equality (the only binary operations MIR performs on
    // string data directly; ordering goes through library calls).
    if let (Value::Str(a), Value::Str(b)) = (lhs, rhs) {
        let v = match op {
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            _ => {
                return Err(InterpError::type_mismatch(
                    format!("binary operation {op:?}"),
                    "integer operands",
                    "string",
                ))
            }
        };
        return Ok((v, false));
    }
    // Shifts: in MIR the amount of a `Shl` / `Shr` may have ANY
    // integer type, independent of the value's type, so the
    // same-kind pair dispatch below cannot see mixed pairs. Integer
    // shift pairs are evaluated by `eval_shift`, which reads the
    // amount as a `u32` and keeps the value's own type and
    // signedness.
    if matches!(op, Shl | Shr)
        && matches!(lhs, Value::Int(_) | Value::Uint(_))
        && matches!(rhs, Value::Int(_) | Value::Uint(_))
    {
        return eval_shift(op, lhs, rhs, dest_ty, checked_form);
    }
    // Signed arithmetic.
    if let (Value::Int(a), Value::Int(b)) = (lhs, rhs) {
        let (a, b) = (*a, *b);
        let bits = signed_bits(dest_ty);
        let (min, max) = if bits >= 128 {
            (i128::MIN, i128::MAX)
        } else {
            (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
        };
        let cmp = |f: &dyn Fn() -> bool| Ok((Value::Bool(f()), false));
        let wide: Option<i128> = match op {
            Eq => return cmp(&|| a == b),
            Ne => return cmp(&|| a != b),
            Lt => return cmp(&|| a < b),
            Le => return cmp(&|| a <= b),
            Gt => return cmp(&|| a > b),
            Ge => return cmp(&|| a >= b),
            Add => a
                .checked_add(b)
                .or(Some(wrap_signed(a.wrapping_add(b), bits))),
            Sub => a
                .checked_sub(b)
                .or(Some(wrap_signed(a.wrapping_sub(b), bits))),
            Mul => a
                .checked_mul(b)
                .or(Some(wrap_signed(a.wrapping_mul(b), bits))),
            Div | Rem => {
                if b == 0 {
                    return Err(InterpError::panic(if op == Div {
                        "attempt to divide by zero"
                    } else {
                        "attempt to calculate the remainder with a divisor of zero"
                    }));
                }
                if a == min && b == -1 {
                    // Overflow: checked form wraps (Div yields `min`,
                    // Rem yields 0), plain form panics via the flag below.
                    Some(if op == Div { a } else { 0 })
                } else if op == Div {
                    Some(a / b)
                } else {
                    Some(a % b)
                }
            }
            BitAnd => Some(a & b),
            BitOr => Some(a | b),
            BitXor => Some(a ^ b),
            Shl | Shr => unreachable!("shifts are evaluated by `eval_shift`"),
        };
        // Invariant: comparisons returned early above and every other
        // arm of the match yields `Some`, so `wide` is always present.
        let raw = wide.expect("all signed ops produce a value");
        // Determine overflow by recomputing in the declared width: a result
        // is overflowing if the mathematical result leaves [min, max].
        // Division overflow (MIN / -1) is covered because raw == MIN while
        // the true quotient is MAX+1; detect it explicitly.
        let overflowed = match op {
            Add => a.checked_add(b).map_or(true, |r| r < min || r > max),
            Sub => a.checked_sub(b).map_or(true, |r| r < min || r > max),
            Mul => a.checked_mul(b).map_or(true, |r| r < min || r > max),
            Div | Rem => a == min && b == -1,
            Shl => unreachable!("shifts are evaluated by `eval_shift`"),
            _ => false,
        };
        let result = Value::Int(wrap_signed(raw, bits));
        if overflowed && !checked_form {
            return Err(InterpError::panic(format!("arithmetic overflow in {op:?}")));
        }
        return Ok((result, overflowed));
    }
    // Unsigned arithmetic.
    if let (Value::Uint(a), Value::Uint(b)) = (lhs, rhs) {
        let (a, b) = (*a, *b);
        let bits = unsigned_bits(dest_ty);
        let max = if bits >= 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        };
        let cmp = |f: &dyn Fn() -> bool| Ok((Value::Bool(f()), false));
        match op {
            Eq => return cmp(&|| a == b),
            Ne => return cmp(&|| a != b),
            Lt => return cmp(&|| a < b),
            Le => return cmp(&|| a <= b),
            Gt => return cmp(&|| a > b),
            Ge => return cmp(&|| a >= b),
            _ => {}
        }
        if (op == Div || op == Rem) && b == 0 {
            return Err(InterpError::panic(if op == Div {
                "attempt to divide by zero"
            } else {
                "attempt to calculate the remainder with a divisor of zero"
            }));
        }
        // Comparisons returned above and shifts are evaluated by
        // `eval_shift` before this dispatch; the operators that
        // can reach this match are exactly the eight covered below,
        // which is what makes the closing `unreachable!` an internal
        // invariant rather than an input-dependent failure.
        let (raw, overflowed) = match op {
            Add => {
                let r = a.wrapping_add(b);
                (r, a.checked_add(b).map_or(true, |v| v > max))
            }
            Sub => (a.wrapping_sub(b), a < b),
            Mul => {
                let r = a.wrapping_mul(b);
                (r, a.checked_mul(b).map_or(true, |v| v > max))
            }
            Div => (a / b, false),
            Rem => (a % b, false),
            BitAnd => (a & b, false),
            BitOr => (a | b, false),
            BitXor => (a ^ b, false),
            _ => unreachable!("comparisons and shifts handled above"),
        };
        let result = Value::Uint(wrap_unsigned(raw, bits));
        if overflowed && !checked_form {
            return Err(InterpError::panic(format!("arithmetic overflow in {op:?}")));
        }
        return Ok((result, overflowed));
    }
    // Floating-point arithmetic, computed in `f64` (see `Value::Float`
    // for why that is exact at `f32` destinations too). Division by
    // zero and NaN follow IEEE 754, as in Rust: they are values, not
    // panics. Float operations never set the overflow flag.
    if let (Value::Float(a), Value::Float(b)) = (lhs, rhs) {
        let (a, b) = (*a, *b);
        let round = |r: f64| match float_ty_of(dest_ty) {
            Some(FloatTy::F32) => (r as f32) as f64,
            _ => r,
        };
        let v = match op {
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            Lt => Value::Bool(a < b),
            Le => Value::Bool(a <= b),
            Gt => Value::Bool(a > b),
            Ge => Value::Bool(a >= b),
            Add => Value::Float(round(a + b)),
            Sub => Value::Float(round(a - b)),
            Mul => Value::Float(round(a * b)),
            Div => Value::Float(round(a / b)),
            Rem => Value::Float(round(a % b)),
            _ => {
                return Err(InterpError::type_mismatch(
                    format!("binary operation {op:?}"),
                    "integer operands",
                    "float",
                ))
            }
        };
        return Ok((v, false));
    }
    Err(InterpError::type_mismatch(
        format!("binary operation {op:?}"),
        "matching integer/bool operands",
        format!("{} and {}", lhs.kind_name(), rhs.kind_name()),
    ))
}

/// The guest panic message for a shift whose amount meets or
/// exceeds the value's width: Rust 1.100.0's native wording, which
/// names only the direction (the amount is not part of the text).
fn shift_overflow_message(op: BinOp) -> &'static str {
    if op == BinOp::Shl {
        "attempt to shift left with overflow"
    } else {
        "attempt to shift right with overflow"
    }
}

/// Evaluate a `Shl` / `Shr` operation.
///
/// The value (left operand) keeps its own type and signedness: the
/// declared width comes from the destination type exactly as in the
/// same-kind paths of `eval_binary`, a signed value shifts
/// arithmetically and an unsigned value logically. The amount
/// (right operand) may have any integer type; only its numeric
/// value matters. It is read as a `u32`, as rustc interprets shift
/// amounts: a negative amount or one above `u32::MAX` does not fit,
/// and `u32::MAX` stands in for it, which is at or above every
/// type width and so reports an overflowing amount either way.
///
/// An amount at or above the value's width is the only overflow a
/// shift has: the checked form yields the wrapped result (the
/// amount taken modulo the width, the `wrapping_shl` /
/// `wrapping_shr` reading) with the overflow flag set, and the
/// plain form is a guest panic with Rust 1.100.0's native message
/// (`attempt to shift left with overflow` / `attempt to shift
/// right with overflow`). Bits shifted out of the declared width
/// are simply lost, as natively: they set neither the overflow
/// flag nor a panic, in either form.
fn eval_shift(
    op: BinOp,
    lhs: &Value,
    rhs: &Value,
    dest_ty: Option<&Ty>,
    checked_form: bool,
) -> InterpResult<(Value, bool)> {
    let shift = match rhs {
        Value::Int(b) => u32::try_from(*b).unwrap_or(u32::MAX),
        Value::Uint(b) => u32::try_from(*b).unwrap_or(u32::MAX),
        _ => unreachable!("shift amounts are integer values"),
    };
    match lhs {
        Value::Int(a) => {
            let a = *a;
            let bits = signed_bits(dest_ty);
            if shift >= bits {
                if checked_form {
                    let wrapped = if op == BinOp::Shl {
                        wrap_signed(a.wrapping_shl(shift % bits), bits)
                    } else {
                        wrap_signed(a.wrapping_shr(shift % bits), bits)
                    };
                    return Ok((Value::Int(wrapped), true));
                }
                return Err(InterpError::panic(shift_overflow_message(op)));
            }
            let raw = if op == BinOp::Shl {
                wrap_signed(a << shift, bits)
            } else {
                a >> shift
            };
            Ok((Value::Int(wrap_signed(raw, bits)), false))
        }
        Value::Uint(a) => {
            let a = *a;
            let bits = unsigned_bits(dest_ty);
            if shift >= bits {
                if checked_form {
                    let wrapped = if op == BinOp::Shl {
                        wrap_unsigned(a.wrapping_shl(shift % bits), bits)
                    } else {
                        wrap_unsigned(a.wrapping_shr(shift % bits), bits)
                    };
                    return Ok((Value::Uint(wrapped), true));
                }
                return Err(InterpError::panic(shift_overflow_message(op)));
            }
            let raw = if op == BinOp::Shl {
                a << shift
            } else {
                a >> shift
            };
            Ok((Value::Uint(wrap_unsigned(raw, bits)), false))
        }
        _ => unreachable!("shift values are integer values"),
    }
}

/// The float width a binary operation's destination declares, if any.
/// A checked operation's destination is the `(T, bool)` tuple; the
/// operand width lives in its first element.
fn float_ty_of(dest_ty: Option<&Ty>) -> Option<FloatTy> {
    match dest_ty {
        Some(Ty::Float(ft)) => Some(*ft),
        Some(Ty::Tuple(ts)) => float_ty_of(ts.first()),
        _ => None,
    }
}

fn eval_unary(op: UnOp, v: &Value, dest_ty: Option<&Ty>) -> InterpResult<Value> {
    match (op, v) {
        (UnOp::Neg, Value::Int(a)) => {
            let bits = signed_bits(dest_ty);
            match a.checked_neg() {
                Some(r)
                    if bits >= 128
                        || (r >= -(1i128 << (bits - 1)) && r <= (1i128 << (bits - 1)) - 1) =>
                {
                    Ok(Value::Int(wrap_signed(r, bits)))
                }
                _ => Err(InterpError::panic("arithmetic overflow in Neg")),
            }
        }
        (UnOp::Neg, Value::Float(a)) => Ok(Value::Float(-a)),
        (UnOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
        (UnOp::Not, Value::Int(a)) => Ok(Value::Int(wrap_signed(!a, signed_bits(dest_ty)))),
        (UnOp::Not, Value::Uint(a)) => Ok(Value::Uint(wrap_unsigned(!a, unsigned_bits(dest_ty)))),
        (UnOp::PtrMetadata, _) => Err(InterpError::type_mismatch(
            "PtrMetadata",
            "fat pointer",
            v.kind_name(),
        )),
        _ => Err(InterpError::type_mismatch(
            format!("unary operation {op:?}"),
            "integer or bool",
            v.kind_name(),
        )),
    }
}

/// Rust float-to-int cast semantics: saturate at the target type's
/// range, NaN becomes 0. The host `as` cast saturates at `i128`;
/// clamping to the declared width finishes the job.
fn float_to_int(v: f64, bits: u32) -> i128 {
    let wide = v as i128;
    if bits >= 128 {
        wide
    } else {
        let min = -(1i128 << (bits - 1));
        let max = (1i128 << (bits - 1)) - 1;
        wide.clamp(min, max)
    }
}

/// Rust float-to-uint cast semantics: saturate at the target type's
/// range (negative values become 0), NaN becomes 0.
fn float_to_uint(v: f64, bits: u32) -> u128 {
    let wide = v as u128;
    if bits >= 128 {
        wide
    } else {
        wide.min((1u128 << bits) - 1)
    }
}

/// A stand-in numeric address for a pointer value. Places in this
/// value model have no real addresses, but an address is still
/// observable through pointer-to-integer transmutes — most
/// importantly the alignment precondition checks rustc emits
/// around raw-pointer dereferences, which compute
/// `addr & (align - 1) == 0`. The synthetic address is a pure
/// function of the pointer's target: the same target always yields
/// the same address, distinct targets yield distinct addresses
/// (up to the mixing hash), and every address is a multiple of
/// 4096, so the alignment check holds for any alignment a place
/// in this model can carry. A reference to a constant aggregate
/// has no place identity; it shares one reserved address.
fn synthetic_addr(v: &Value) -> u128 {
    fn mix(h: &mut u128, x: u128) {
        // FNV-1a over the value's limbs.
        for limb in [x as u64, (x >> 64) as u64] {
            *h ^= limb as u128;
            *h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    let mut h: u128 = 0xcbf2_9ce4_8422_2325;
    match v {
        Value::Ref(RefTarget::Place {
            frame,
            local,
            projection,
        }) => {
            mix(&mut h, *frame as u128);
            mix(&mut h, *local as u128);
            for step in projection {
                let (tag, n) = match step {
                    RefProj::Field(i) => (1u128, *i as u128),
                    RefProj::Index(i) => (2u128, *i as u128),
                    RefProj::Downcast(i) => (3u128, *i as u128),
                };
                mix(&mut h, tag);
                mix(&mut h, n);
            }
        }
        Value::Ref(RefTarget::Value(_)) => {
            mix(&mut h, 0x5eed_5eed_5eed_5eed);
        }
        Value::FatRef { base, .. } => {
            return synthetic_addr(base);
        }
        Value::FnPtr(name) => {
            for b in name.bytes() {
                mix(&mut h, b as u128);
            }
        }
        _ => {}
    }
    const BASE: u128 = 0x1000_0000_0000;
    BASE + (h % (1 << 40)) * 4096
}

fn eval_cast(v: &Value, ty: &Ty) -> InterpResult<Value> {
    match ty {
        Ty::Int(it) => {
            // A float source saturates; it does not wrap like an
            // integer source.
            if let Value::Float(a) = v {
                return Ok(Value::Int(float_to_int(*a, it.bits())));
            }
            // A pointer source (a pointer-to-integer transmute)
            // yields the pointer's synthetic address.
            if matches!(v, Value::Ref(_) | Value::FatRef { .. } | Value::FnPtr(_)) {
                return Ok(Value::Int(wrap_signed(
                    synthetic_addr(v) as i128,
                    it.bits(),
                )));
            }
            let raw = match v {
                Value::Int(a) => *a,
                Value::Uint(a) => *a as i128,
                Value::Bool(b) => *b as i128,
                other => {
                    return Err(InterpError::type_mismatch(
                        "cast",
                        "numeric source",
                        other.kind_name(),
                    ))
                }
            };
            Ok(Value::Int(wrap_signed(raw, it.bits())))
        }
        Ty::Uint(ut) => {
            if let Value::Float(a) = v {
                return Ok(Value::Uint(float_to_uint(*a, ut.bits())));
            }
            // A pointer source (a pointer-to-integer transmute)
            // yields the pointer's synthetic address.
            if matches!(v, Value::Ref(_) | Value::FatRef { .. } | Value::FnPtr(_)) {
                return Ok(Value::Uint(wrap_unsigned(synthetic_addr(v), ut.bits())));
            }
            let raw = match v {
                Value::Int(a) => *a as u128,
                Value::Uint(a) => *a,
                Value::Bool(b) => *b as u128,
                other => {
                    return Err(InterpError::type_mismatch(
                        "cast",
                        "numeric source",
                        other.kind_name(),
                    ))
                }
            };
            Ok(Value::Uint(wrap_unsigned(raw, ut.bits())))
        }
        Ty::Float(ft) => {
            // Convert at the target width directly: going through
            // `f64` first would double-round wide integer sources.
            let out = match (v, ft) {
                (Value::Int(a), FloatTy::F32) => (*a as f32) as f64,
                (Value::Int(a), FloatTy::F64) => *a as f64,
                (Value::Uint(a), FloatTy::F32) => (*a as f32) as f64,
                (Value::Uint(a), FloatTy::F64) => *a as f64,
                (Value::Float(a), FloatTy::F32) => (*a as f32) as f64,
                (Value::Float(a), FloatTy::F64) => *a,
                (other, _) => {
                    return Err(InterpError::type_mismatch(
                        "cast",
                        "numeric source",
                        other.kind_name(),
                    ))
                }
            };
            Ok(Value::Float(out))
        }
        Ty::Bool => match v {
            Value::Bool(b) => Ok(Value::Bool(*b)),
            Value::Int(a) => Ok(Value::Bool(*a != 0)),
            Value::Uint(a) => Ok(Value::Bool(*a != 0)),
            other => Err(InterpError::type_mismatch(
                "cast",
                "numeric source",
                other.kind_name(),
            )),
        },
        _ => Err(InterpError::UnsupportedRvalue {
            kind: format!("Cast to {ty:?}"),
        }),
    }
}

/// Convenience one-shot entry point with the default configuration.
pub fn interpret(program: &Program, entry: &str, args: Vec<Value>) -> InterpResult<Value> {
    Interpreter::new(program).run(entry, args)
}

/// Extend a `RefCell` place target with a field path inside the
/// cell. Only place targets can be extended; guard construction
/// starts from `Interpreter::refcell_place`, which has already
/// rejected the rest.
pub(crate) fn refcell_sub_target(cell: &RefTarget, path: &[RefProj]) -> RefTarget {
    match cell {
        RefTarget::Place {
            frame,
            local,
            projection,
        } => {
            let mut projection = projection.clone();
            projection.extend_from_slice(path);
            RefTarget::Place {
                frame: *frame,
                local: *local,
                projection,
            }
        }
        // Unreachable through `refcell_place`; keep the target as-is
        // rather than panicking in the model.
        other => other.clone(),
    }
}
