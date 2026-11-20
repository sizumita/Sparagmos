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
//! * Execution is bounded by a step limit and a call-depth limit so a
//!   looping program can never hang the oracle (or a test run).

use crate::error::{InterpError, InterpResult};
use crate::ir::*;
use crate::value::Value;

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
struct Frame {
    func: String,
    locals: Vec<Value>,
    block: usize,
    stmt: usize,
    /// Where (in the caller) the return value goes; `None` for the entry frame.
    return_dest: Option<Place>,
    /// Block to resume in the caller after return.
    return_block: Option<BasicBlock>,
}

/// The interpreter state for one program.
pub struct Interpreter<'a> {
    program: &'a Program,
    config: InterpConfig,
    steps: usize,
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
        }
    }

    /// Steps consumed by the most recent (or current) run.
    pub fn steps_used(&self) -> usize {
        self.steps
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

    /// Run `entry` with `args` and return the value of its return place.
    pub fn run(&mut self, entry: &str, args: Vec<Value>) -> InterpResult<Value> {
        self.steps = 0;
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
        let mut frames: Vec<Frame> = vec![Frame {
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
                    let value = {
                        let frame = frames.last_mut().expect("frame");
                        Self::eval_operand(frame, body, &discr)?
                    };
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
                    {
                        let frame = frames.last_mut().expect("frame");
                        for arg in &args {
                            arg_values.push(Self::eval_operand(frame, body, arg)?);
                        }
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
                    frames.push(Frame {
                        func: func.clone(),
                        locals,
                        block: 0,
                        stmt: 0,
                        return_dest: Some(destination),
                        return_block: target,
                    });
                }
                TerminatorKind::Return => {
                    let frame = frames.pop().expect("frame");
                    let value = frame.locals.first().cloned().unwrap_or(Value::Uninit);
                    if value == Value::Uninit {
                        return Err(InterpError::UninitializedRead {
                            func: frame.func.clone(),
                            place: "_0".to_string(),
                        });
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
                    {
                        let caller = frames.last_mut().expect("caller");
                        Self::write_place(caller, &caller_body, &dest, value)?;
                    }
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
                    let value = {
                        let frame = frames.last_mut().expect("frame");
                        Self::eval_operand(frame, body, &cond)?
                    };
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
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        statement: &Statement,
    ) -> InterpResult<()> {
        match &statement.kind {
            StatementKind::Assign(place, rvalue) => {
                let frame = frames.last_mut().expect("frame");
                let dest_ty = body.local_ty(place.local).cloned();
                let value = Self::eval_rvalue(frame, body, rvalue, dest_ty.as_ref())?;
                Self::write_place(frame, body, place, value)
            }
            // StorageLive / StorageDead are liveness hints in rustc MIR;
            // they do not change values. Nop is a no-op by definition.
            StatementKind::StorageLive(_) | StatementKind::StorageDead(_) | StatementKind::Nop => {
                Ok(())
            }
            StatementKind::Unsupported(kind) => {
                Err(InterpError::UnsupportedStatement { kind: kind.clone() })
            }
        }
    }

    // ----- places & operands -------------------------------------------------

    fn read_place(frame: &Frame, body: &Body, place: &Place) -> InterpResult<Value> {
        let mut value =
            frame
                .locals
                .get(place.local.0)
                .cloned()
                .ok_or_else(|| InterpError::UnknownLocal {
                    func: body.name.clone(),
                    local: place.local.0,
                })?;
        for elem in &place.projection {
            match elem {
                ProjectionElem::Field(idx) => match value {
                    Value::Tuple(ref items) => {
                        value = items.get(*idx).cloned().ok_or_else(|| {
                            InterpError::UninitializedRead {
                                func: body.name.clone(),
                                place: place.to_string(),
                            }
                        })?;
                    }
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            format!("field projection on {place}"),
                            "tuple",
                            other.kind_name(),
                        ));
                    }
                },
                ProjectionElem::Deref => {
                    return Err(InterpError::UnsupportedProjection {
                        kind: "Deref".into(),
                    });
                }
                ProjectionElem::Index(_) => {
                    return Err(InterpError::UnsupportedProjection {
                        kind: "Index".into(),
                    });
                }
            }
        }
        if value == Value::Uninit {
            return Err(InterpError::UninitializedRead {
                func: body.name.clone(),
                place: place.to_string(),
            });
        }
        Ok(value)
    }

    fn write_place(
        frame: &mut Frame,
        body: &Body,
        place: &Place,
        value: Value,
    ) -> InterpResult<()> {
        if place.projection.is_empty() {
            let slot =
                frame
                    .locals
                    .get_mut(place.local.0)
                    .ok_or_else(|| InterpError::UnknownLocal {
                        func: body.name.clone(),
                        local: place.local.0,
                    })?;
            *slot = Self::normalize_for_ty(value, body.local_ty(place.local))?;
            return Ok(());
        }
        // Projected write: walk to the parent value, then rebuild the field.
        // Only a single trailing Field projection is supported in this cut.
        if place.projection.len() == 1 {
            if let ProjectionElem::Field(idx) = place.projection[0] {
                let slot = frame.locals.get_mut(place.local.0).ok_or_else(|| {
                    InterpError::UnknownLocal {
                        func: body.name.clone(),
                        local: place.local.0,
                    }
                })?;
                return match slot {
                    Value::Tuple(items) => {
                        let field =
                            items
                                .get_mut(idx)
                                .ok_or_else(|| InterpError::UninitializedRead {
                                    func: body.name.clone(),
                                    place: place.to_string(),
                                })?;
                        *field = value;
                        Ok(())
                    }
                    Value::Uninit => Err(InterpError::UninitializedRead {
                        func: body.name.clone(),
                        place: place.to_string(),
                    }),
                    other => Err(InterpError::type_mismatch(
                        format!("field write on {place}"),
                        "tuple",
                        other.kind_name(),
                    )),
                };
            }
        }
        Err(InterpError::UnsupportedProjection {
            kind: format!("{place} ({} projections)", place.projection.len()),
        })
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
            (Some(expected), other) => Err(InterpError::type_mismatch(
                "assignment",
                format!("{expected:?}"),
                other.kind_name(),
            )),
        }
    }

    fn eval_operand(frame: &mut Frame, body: &Body, operand: &Operand) -> InterpResult<Value> {
        match operand {
            Operand::Copy(place) => Self::read_place(frame, body, place),
            Operand::Move(place) => {
                let value = Self::read_place(frame, body, place)?;
                if place.is_direct_local() {
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
            }),
        }
    }

    // ----- rvalues -----------------------------------------------------------

    fn eval_rvalue(
        frame: &mut Frame,
        body: &Body,
        rvalue: &Rvalue,
        dest_ty: Option<&Ty>,
    ) -> InterpResult<Value> {
        match rvalue {
            Rvalue::Use(op) => Self::eval_operand(frame, body, op),
            Rvalue::BinaryOp(op, pair) => {
                let lhs = Self::eval_operand(frame, body, &pair.0)?;
                let rhs = Self::eval_operand(frame, body, &pair.1)?;
                eval_binary(*op, &lhs, &rhs, dest_ty, false).map(|(v, _)| v)
            }
            Rvalue::CheckedBinaryOp(op, pair) => {
                let lhs = Self::eval_operand(frame, body, &pair.0)?;
                let rhs = Self::eval_operand(frame, body, &pair.1)?;
                let (v, overflowed) = eval_binary(*op, &lhs, &rhs, dest_ty, true)?;
                Ok(Value::Tuple(vec![v, Value::Bool(overflowed)]))
            }
            Rvalue::UnaryOp(op, operand) => {
                let v = Self::eval_operand(frame, body, operand)?;
                eval_unary(*op, &v, dest_ty)
            }
            Rvalue::AggregateTuple(operands) => {
                let mut items = Vec::with_capacity(operands.len());
                for op in operands {
                    items.push(Self::eval_operand(frame, body, op)?);
                }
                Ok(Value::Tuple(items))
            }
            Rvalue::Cast(operand, ty) => {
                let v = Self::eval_operand(frame, body, operand)?;
                eval_cast(&v, ty)
            }
            Rvalue::Ref(_) => Err(InterpError::UnsupportedRvalue { kind: "Ref".into() }),
            Rvalue::Unsupported(kind) => Err(InterpError::UnsupportedRvalue { kind: kind.clone() }),
        }
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
    Err(InterpError::type_mismatch(
        format!("binary operation {op:?}"),
        "matching integer/bool operands",
        format!("{} and {}", lhs.kind_name(), rhs.kind_name()),
    ))
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
        (UnOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
        (UnOp::Not, Value::Int(a)) => Ok(Value::Int(wrap_signed(!a, signed_bits(dest_ty)))),
        (UnOp::Not, Value::Uint(a)) => Ok(Value::Uint(wrap_unsigned(!a, unsigned_bits(dest_ty)))),
        _ => Err(InterpError::type_mismatch(
            format!("unary operation {op:?}"),
            "integer or bool",
            v.kind_name(),
        )),
    }
}

fn eval_cast(v: &Value, ty: &Ty) -> InterpResult<Value> {
    match ty {
        Ty::Int(it) => {
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
