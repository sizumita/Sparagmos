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
            .ok_or_else(|| InterpError::UnknownFunction {
                name: entry.to_string(),
            })?;
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
                    let callee = self
                        .program
                        .get(&func)
                        .cloned()
                        .ok_or_else(|| InterpError::UnknownFunction { name: func.clone() })?;
                    if frames.len() >= self.config.max_call_depth {
                        return Err(InterpError::CallDepthExceeded {
                            limit: self.config.max_call_depth,
                        });
                    }
                    let mut arg_values = Vec::with_capacity(args.len());
                    for arg in &args {
                        arg_values.push(self.eval_operand(&mut frames, body, arg)?);
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
                        func: func.clone(),
                        locals,
                        block: 0,
                        stmt: 0,
                        return_dest: Some(destination),
                        return_block: target,
                    });
                }
                TerminatorKind::CallBuiltin {
                    name,
                    generic_args,
                    args,
                    destination,
                    target,
                } => {
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
                TerminatorKind::Drop { place, target } => {
                    // Conversion only emits this for types whose drop
                    // glue has no guest-observable effect in this
                    // value model (deallocation and payload teardown
                    // of plain data). The place is dead afterwards, as
                    // in MIR, so a direct local is cleared.
                    if place.is_direct_local() {
                        let frame = frames.last_mut().expect("frame");
                        if let Some(slot) = frame.locals.get_mut(place.local.0) {
                            *slot = Value::Uninit;
                        }
                    }
                    Self::jump(&mut frames, body, target)?;
                }
                TerminatorKind::Return => {
                    let frame = frames.pop().expect("frame");
                    let mut value = frame.locals.first().cloned().unwrap_or(Value::Uninit);
                    if value == Value::Uninit {
                        // rustc MIR never writes the return place of a
                        // unit-returning function: its final block is a
                        // bare `return;`, and the single value of the
                        // unit type is implied. An unwritten `_0` therefore
                        // reads as `()` exactly when the declared return
                        // type is unit. For every other type an unwritten
                        // `_0` is a genuine uninitialized read and stays
                        // an error.
                        if body.local_ty(Local::RETURN) == Some(&Ty::Unit) {
                            value = Value::Unit;
                        } else {
                            return Err(InterpError::UninitializedRead {
                                func: frame.func.clone(),
                                place: "_0".to_string(),
                            });
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
            // Dereferencing a fat pointer reaches its base.
            Value::FatRef { base, .. } => self.deref_value(frames, base),
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
            // ADT locals hold the ADT value forms: aggregates, string
            // data (`String`), and the fmt value family.
            (Some(Ty::Adt(_)), Value::Enum { .. })
            | (Some(Ty::Adt(_)), Value::Str(_))
            | (Some(Ty::Adt(_)), Value::FmtArg { .. })
            | (Some(Ty::Adt(_)), Value::FmtCount(_))
            | (Some(Ty::Adt(_)), Value::FmtArguments(_)) => Ok(value),
            // Reference locals hold resolved references, fat pointers,
            // and — for `&str` and byte-array references — the string
            // or byte data itself (a string constant *is* its value;
            // a reference to a string place is a `Ref`).
            (Some(Ty::Ref(_)), Value::Ref(_))
            | (Some(Ty::Ref(_)), Value::FatRef { .. })
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
                }
                Ok(value)
            }
            Operand::Constant(c) => Ok(match c {
                Constant::Unit => Value::Unit,
                Constant::Bool(b) => Value::Bool(*b),
                Constant::Int(v) => Value::Int(*v),
                Constant::Uint(v) => Value::Uint(*v),
                Constant::Float(v) => Value::Float(*v),
                Constant::Str(s) => Value::Str(s.clone()),
                Constant::Bytes(b) => Value::Bytes(b.clone()),
            }),
        }
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
            Rvalue::AggregateAdt { variant, operands } => {
                let mut fields = Vec::with_capacity(operands.len());
                for op in operands {
                    fields.push(self.eval_operand(frames, body, op)?);
                }
                Ok(Value::Enum {
                    variant: *variant,
                    fields,
                })
            }
            Rvalue::Cast(operand, ty) => {
                let v = self.eval_operand(frames, body, operand)?;
                eval_cast(&v, ty)
            }
            Rvalue::Unsize(operand) => {
                let v = self.eval_operand(frames, body, operand)?;
                match v {
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
                }
            }
            Rvalue::Ref(place) => {
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
            Rvalue::Discriminant(place) => {
                let value = self.read_place(frames, body, place)?;
                match value {
                    Value::Enum { variant, .. } => Ok(Value::Int(variant as i128)),
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
                ref other => {
                    return Err(InterpError::type_mismatch(
                        format!("field projection on {place}"),
                        "tuple or ADT value",
                        other.kind_name(),
                    ));
                }
            },
            RefProj::Index(idx) => match value {
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
            let items = match slot {
                Value::Tuple(items) => items,
                Value::Enum { fields, .. } => fields,
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
            Shl | Shr => {
                let shift = u32::try_from(b).unwrap_or(u32::MAX);
                if shift >= bits {
                    if checked_form {
                        let wrapped = wrap_signed(a.wrapping_shl(shift % bits), bits);
                        return Ok((Value::Int(wrapped), true));
                    }
                    return Err(InterpError::panic(format!(
                        "attempt to shift by {shift} bits, which overflows the type width {bits}"
                    )));
                }
                Some(if op == Shl {
                    wrap_signed(a << shift, bits)
                } else {
                    a >> shift
                })
            }
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
            Shl => {
                // Overflow if significant bits are shifted out of the
                // declared width, i.e. shifting back does not restore `a`.
                let shift = u32::try_from(b).unwrap_or(u32::MAX);
                shift >= bits || (shift < 128 && (wrap_signed(a << shift, bits) >> shift) != a)
            }
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
        if op == Shl || op == Shr {
            let shift = u32::try_from(b).unwrap_or(u32::MAX);
            if shift >= bits {
                if checked_form {
                    let wrapped = wrap_unsigned(a.wrapping_shl(shift % bits), bits);
                    return Ok((Value::Uint(wrapped), true));
                }
                return Err(InterpError::panic(format!(
                    "attempt to shift by {shift} bits, which overflows the type width {bits}"
                )));
            }
            let raw = if op == Shl { a << shift } else { a >> shift };
            let overflowed = op == Shl && (wrap_unsigned(raw, bits) >> shift) != a;
            let result = Value::Uint(wrap_unsigned(raw, bits));
            if overflowed && !checked_form {
                return Err(InterpError::panic(format!("arithmetic overflow in {op:?}")));
            }
            return Ok((result, overflowed));
        }
        // Comparisons and shifts returned above; the operators that
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

fn eval_cast(v: &Value, ty: &Ty) -> InterpResult<Value> {
    match ty {
        Ty::Int(it) => {
            // A float source saturates; it does not wrap like an
            // integer source.
            if let Value::Float(a) = v {
                return Ok(Value::Int(float_to_int(*a, it.bits())));
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
