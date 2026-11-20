//! Conversion from `sparagmos-mir` bodies into the interpreter's own IR,
//! plus a one-shot runner on top of it.
//!
//! This module is an internal adapter of the interpreter — it is *not* a
//! separate input path. On the project's main path, MIR reaches Sparagmos
//! exclusively through the Stable MIR driver (Lane I), which produces
//! [`sparagmos_mir::Body`] values; this layer only re-shapes those bodies
//! into the form [`crate::interp::Interpreter`] executes, inside this
//! crate. It is deliberately not a standalone public system.
//!
//! Conversion policy (the interpreter's honesty contract, extended):
//! * Kinds the interpreter's IR can mark as unsupported (rvalue /
//!   statement / terminator kinds) are converted into the corresponding
//!   `Unsupported` IR nodes, so execution fails loudly with context
//!   instead of conversion silently dropping them.
//! * Anything that cannot even be represented in the IR (local types,
//!   place projections, constant forms) fails conversion with a
//!   [`ConvertError`]. Nothing is ever silently skipped.
//!
//! Notable semantic choices:
//! * `mir` `BinaryOp` with a `*WithOverflow` operator and
//!   `CheckedBinaryOp` both become the IR's `CheckedBinaryOp` (they yield
//!   `(value, overflowed)` in rustc MIR as well).
//! * `*Unchecked` operators map to the plain IR operator: in valid MIR
//!   they only execute when no overflow occurs (overflow is UB there),
//!   so the oracle's checked behaviour is unobservable for them.
//! * Execution-only no-ops in rustc MIR (`FakeRead`, `PlaceMention`,
//!   `Retag`, `AscribeUserType`, `Coverage`, `ConstEvalCounter`) become
//!   IR `Nop`s, exactly as an executor is expected to treat them.
//! * `FalseEdge` / `FalseUnwind` go to their real target; the imaginary
//!   edge is never taken at run time.
//! * Unwind actions on `Call` / `Assert` are dropped: the interpreter has
//!   no unwinding — a guest panic aborts the whole run as an
//!   [`InterpError::Panic`], which is the oracle's model of a panic.
//! * `AggregateKind::Array` becomes a tuple value (the IR has no distinct
//!   array value; element reads through `Index` projections are still
//!   rejected at run time, as before).

use std::collections::HashMap;
use std::fmt;

use sparagmos_mir as mir;

use crate::error::InterpError;
use crate::interp::Interpreter;
use crate::ir;
use crate::value::Value;

/// One function of a MIR program, as handed over by the driver: a display
/// name (the interpreter keys functions by name), the `DefId` when known
/// (used to resolve `FnDef` call operands), and the body itself.
#[derive(Debug, Clone)]
pub struct MirFunction {
    pub name: String,
    pub def_id: Option<mir::DefId>,
    pub body: mir::Body,
}

/// A set of MIR functions forming one program (one crate's worth of
/// bodies, as produced by the driver on the main path).
#[derive(Debug, Clone, Default)]
pub struct MirProgram {
    pub functions: Vec<MirFunction>,
}

impl MirProgram {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a function known only by name.
    pub fn add(&mut self, name: impl Into<String>, body: mir::Body) -> &mut Self {
        self.functions.push(MirFunction {
            name: name.into(),
            def_id: None,
            body,
        });
        self
    }

    /// Add a function together with its `DefId`, so calls to it (whose
    /// func operand is a `FnDef` constant) can be resolved.
    pub fn add_with_def_id(
        &mut self,
        name: impl Into<String>,
        def_id: mir::DefId,
        body: mir::Body,
    ) -> &mut Self {
        self.functions.push(MirFunction {
            name: name.into(),
            def_id: Some(def_id),
            body,
        });
        self
    }

    pub fn get(&self, name: &str) -> Option<&MirFunction> {
        self.functions.iter().find(|f| f.name == name)
    }
}

/// Why a `sparagmos-mir` program could not be converted into IR at all.
/// (Kinds that *can* be represented as IR `Unsupported` nodes do not
/// appear here; they fail at execution time instead.)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvertError {
    /// A local (or cast target / tuple element) has a type the IR's
    /// scalar type model cannot represent (floats, ADTs, references, ...).
    UnsupportedType { context: String, ty: String },
    /// A constant form the IR cannot represent (strings, pointers,
    /// floats, unevaluated / type constants outside call position, ...).
    UnsupportedConstant { kind: String },
    /// A place projection the IR cannot represent.
    UnsupportedProjection { kind: String },
    /// Two functions with the same name in one program.
    DuplicateFunction { name: String },
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConvertError::UnsupportedType { context, ty } => {
                write!(f, "unsupported type in {context}: {ty}")
            }
            ConvertError::UnsupportedConstant { kind } => {
                write!(f, "unsupported constant: {kind}")
            }
            ConvertError::UnsupportedProjection { kind } => {
                write!(f, "unsupported projection: {kind}")
            }
            ConvertError::DuplicateFunction { name } => {
                write!(f, "duplicate function name `{name}` in MIR program")
            }
        }
    }
}

impl std::error::Error for ConvertError {}

/// Failure of a whole MIR run: either conversion or interpretation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MirRunError {
    Convert(ConvertError),
    Interp(InterpError),
}

impl fmt::Display for MirRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MirRunError::Convert(e) => write!(f, "MIR conversion failed: {e}"),
            MirRunError::Interp(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for MirRunError {}

impl From<ConvertError> for MirRunError {
    fn from(e: ConvertError) -> Self {
        MirRunError::Convert(e)
    }
}

impl From<InterpError> for MirRunError {
    fn from(e: InterpError) -> Self {
        MirRunError::Interp(e)
    }
}

/// Convert a whole MIR program into an interpreter [`ir::Program`].
pub fn convert_program(program: &MirProgram) -> Result<ir::Program, ConvertError> {
    let conv = Converter::new(program);
    let mut out = ir::Program::new();
    for func in &program.functions {
        if out.get(&func.name).is_some() {
            return Err(ConvertError::DuplicateFunction {
                name: func.name.clone(),
            });
        }
        let body = conv.convert_body(func)?;
        out.add_function(body);
    }
    Ok(out)
}

/// Convert `program` and run `entry` with `args` in the interpreter.
pub fn interpret_mir(
    program: &MirProgram,
    entry: &str,
    args: Vec<Value>,
) -> Result<Value, MirRunError> {
    let ir_program = convert_program(program)?;
    let mut interp = Interpreter::new(&ir_program);
    Ok(interp.run(entry, args)?)
}

// ---------------------------------------------------------------------------

struct Converter<'a> {
    /// Function name -> body, for callee argument typing.
    bodies: HashMap<String, &'a mir::Body>,
    /// `DefId` -> function name, for resolving `FnDef` call operands.
    def_names: HashMap<mir::DefId, String>,
}

impl<'a> Converter<'a> {
    fn new(program: &'a MirProgram) -> Self {
        let mut bodies = HashMap::new();
        let mut def_names = HashMap::new();
        for f in &program.functions {
            bodies.insert(f.name.clone(), &f.body);
            if let Some(def) = f.def_id {
                def_names.insert(def, f.name.clone());
            }
        }
        Converter { bodies, def_names }
    }

    fn convert_body(&self, func: &MirFunction) -> Result<ir::Body, ConvertError> {
        let body = &func.body;
        let mut local_tys = Vec::with_capacity(body.local_decls.len());
        for (i, decl) in body.local_decls.iter().enumerate() {
            local_tys.push(self.convert_ty(&decl.ty, &format!("local _{i} of `{}`", func.name))?);
        }
        let mut blocks = Vec::with_capacity(body.basic_blocks.len());
        for bb in &body.basic_blocks {
            blocks.push(self.convert_block(body, bb)?);
        }
        Ok(ir::Body::new(
            func.name.clone(),
            body.arg_count,
            local_tys,
            blocks,
        ))
    }

    fn convert_block(
        &self,
        body: &mir::Body,
        bb: &mir::BasicBlockData,
    ) -> Result<ir::BasicBlockData, ConvertError> {
        let mut statements = Vec::with_capacity(bb.statements.len());
        for stmt in &bb.statements {
            statements.push(self.convert_statement(body, stmt)?);
        }
        let terminator = match &bb.terminator {
            Some(term) => self.convert_terminator(body, &term.kind)?,
            None => ir::TerminatorKind::Unsupported("missing terminator".to_string()),
        };
        Ok(ir::BasicBlockData::new(statements, terminator))
    }

    // ----- types -----------------------------------------------------------

    fn convert_ty(&self, ty: &mir::Ty, context: &str) -> Result<ir::Ty, ConvertError> {
        let unsupported = || ConvertError::UnsupportedType {
            context: context.to_string(),
            ty: format!("{} ({ty:?})", ty.kind_name()),
        };
        Ok(match ty {
            mir::Ty::Bool => ir::Ty::Bool,
            mir::Ty::Int(it) => ir::Ty::Int(convert_int_ty(*it)),
            mir::Ty::Uint(ut) => ir::Ty::Uint(convert_uint_ty(*ut)),
            // `char` is a 32-bit scalar; the IR models it as `u32`.
            mir::Ty::Char => ir::Ty::Uint(ir::UintTy::U32),
            mir::Ty::Never => ir::Ty::Never,
            mir::Ty::Tuple(ts) if ts.is_empty() => ir::Ty::Unit,
            mir::Ty::Tuple(ts) => {
                let mut out = Vec::with_capacity(ts.len());
                for t in ts {
                    out.push(self.convert_ty(t, context)?);
                }
                ir::Ty::Tuple(out)
            }
            _ => return Err(unsupported()),
        })
    }

    // ----- places ----------------------------------------------------------

    fn convert_place(&self, place: &mir::Place) -> Result<ir::Place, ConvertError> {
        let mut out = ir::Place::from_index(place.local.index());
        for elem in &place.projection {
            out.projection.push(match elem {
                mir::PlaceElem::Field(idx, _) => ir::ProjectionElem::Field(*idx as usize),
                mir::PlaceElem::Deref => ir::ProjectionElem::Deref,
                mir::PlaceElem::Index(local) => ir::ProjectionElem::Index(ir::Local(local.index())),
                other => {
                    return Err(ConvertError::UnsupportedProjection {
                        kind: other.kind_name().to_string(),
                    })
                }
            });
        }
        Ok(out)
    }

    /// Best-effort type of a place (used only to type constants in
    /// operand position; `None` when it cannot be derived).
    fn place_ty(&self, body: &mir::Body, place: &mir::Place) -> Option<mir::Ty> {
        let mut ty = body.local_decls.get(place.local.index())?.ty.clone();
        for elem in &place.projection {
            ty = match elem {
                mir::PlaceElem::Field(_, t) => t.clone(),
                mir::PlaceElem::Deref => match ty {
                    mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => *inner,
                    _ => return None,
                },
                mir::PlaceElem::Index(_) | mir::PlaceElem::ConstantIndex { .. } => match ty {
                    mir::Ty::Array(elem, _) | mir::Ty::Slice(elem) => *elem,
                    _ => return None,
                },
                mir::PlaceElem::Downcast(..) => ty,
                mir::PlaceElem::Subslice { .. } => ty,
                mir::PlaceElem::OpaqueCast(t) | mir::PlaceElem::Subtype(t) => t.clone(),
            };
        }
        Some(ty)
    }

    fn operand_place_ty(&self, body: &mir::Body, op: &mir::Operand) -> Option<mir::Ty> {
        match op.place() {
            Some(place) => self.place_ty(body, place),
            None => None,
        }
    }

    // ----- constants & operands ---------------------------------------------

    fn convert_constant(
        &self,
        constant: &mir::Constant,
        ctx: Option<&mir::Ty>,
    ) -> Result<ir::Constant, ConvertError> {
        match &constant.literal {
            mir::ConstKind::Value(v) => self.convert_const_value(v, ctx),
            mir::ConstKind::Ty(t) => Err(ConvertError::UnsupportedConstant {
                kind: format!("type constant {} outside call position", t.kind_name()),
            }),
            mir::ConstKind::Unevaluated { def, .. } => Err(ConvertError::UnsupportedConstant {
                kind: format!("unevaluated constant {def}"),
            }),
        }
    }

    fn convert_const_value(
        &self,
        value: &mir::ConstValue,
        ctx: Option<&mir::Ty>,
    ) -> Result<ir::Constant, ConvertError> {
        Ok(match value {
            mir::ConstValue::Bool(b) => ir::Constant::Bool(*b),
            mir::ConstValue::Char(c) => ir::Constant::Uint(*c as u128),
            mir::ConstValue::ZeroSized => ir::Constant::Unit,
            mir::ConstValue::Scalar(s) => return self.convert_scalar(s, ctx),
            mir::ConstValue::Str(_) => {
                return Err(ConvertError::UnsupportedConstant {
                    kind: "string constant".to_string(),
                })
            }
            mir::ConstValue::Bytes(_) => {
                return Err(ConvertError::UnsupportedConstant {
                    kind: "byte-string constant".to_string(),
                })
            }
            mir::ConstValue::Slice { .. } => {
                return Err(ConvertError::UnsupportedConstant {
                    kind: "slice constant".to_string(),
                })
            }
            mir::ConstValue::ByRef { .. } => {
                return Err(ConvertError::UnsupportedConstant {
                    kind: "by-reference constant".to_string(),
                })
            }
        })
    }

    fn convert_scalar(
        &self,
        scalar: &mir::Scalar,
        ctx: Option<&mir::Ty>,
    ) -> Result<ir::Constant, ConvertError> {
        match scalar {
            mir::Scalar::Float { .. } => Err(ConvertError::UnsupportedConstant {
                kind: "float constant".to_string(),
            }),
            mir::Scalar::Ptr { .. } => Err(ConvertError::UnsupportedConstant {
                kind: "pointer constant".to_string(),
            }),
            mir::Scalar::Int { data, size } => {
                let data = *data;
                Ok(match ctx {
                    Some(mir::Ty::Int(it)) => ir::Constant::Int(sign_extend(data, it.bits())),
                    Some(mir::Ty::Uint(_)) | Some(mir::Ty::Char) => ir::Constant::Uint(data),
                    Some(mir::Ty::Bool) => ir::Constant::Bool(data != 0),
                    // No context: use the scalar's own storage width. A
                    // set sign bit means a negative signed value.
                    _ => {
                        let bits = (*size as u32) * 8;
                        if bits > 0 && bits < 128 && (data >> (bits - 1)) & 1 == 1 {
                            ir::Constant::Int(sign_extend(data, bits))
                        } else {
                            ir::Constant::Uint(data)
                        }
                    }
                })
            }
        }
    }

    fn convert_operand(
        &self,
        // Part of the uniform converter signature; operand typing is
        // done by callers through `ctx`, so the body is not needed here.
        _body: &mir::Body,
        op: &mir::Operand,
        ctx: Option<&mir::Ty>,
    ) -> Result<ir::Operand, ConvertError> {
        Ok(match op {
            mir::Operand::Copy(p) => ir::Operand::Copy(self.convert_place(p)?),
            mir::Operand::Move(p) => ir::Operand::Move(self.convert_place(p)?),
            mir::Operand::Constant(c) => ir::Operand::Constant(self.convert_constant(c, ctx)?),
        })
    }

    // ----- rvalues -----------------------------------------------------------

    fn convert_rvalue(
        &self,
        body: &mir::Body,
        rv: &mir::Rvalue,
        dest_ty: Option<&mir::Ty>,
    ) -> Result<ir::Rvalue, ConvertError> {
        use mir::Rvalue as R;
        Ok(match rv {
            R::Use(op) => ir::Rvalue::Use(self.convert_operand(body, op, dest_ty)?),
            R::BinaryOp(op, pair) => {
                let lhs_ctx = self
                    .operand_place_ty(body, &pair.1)
                    .or_else(|| dest_ty.cloned());
                let rhs_ctx = self
                    .operand_place_ty(body, &pair.0)
                    .or_else(|| dest_ty.cloned());
                let lhs = self.convert_operand(body, &pair.0, lhs_ctx.as_ref())?;
                let rhs = self.convert_operand(body, &pair.1, rhs_ctx.as_ref())?;
                match overflow_base(*op) {
                    // `BinaryOp` with a *WithOverflow operator yields the
                    // `(value, overflowed)` tuple in rustc MIR, exactly
                    // like `CheckedBinaryOp`. A plain operator in a
                    // `BinaryOp` yields just the value.
                    Some(base) => ir::Rvalue::CheckedBinaryOp(base, Box::new((lhs, rhs))),
                    None => match plain_binop(*op) {
                        Some(b) => ir::Rvalue::BinaryOp(b, Box::new((lhs, rhs))),
                        None => ir::Rvalue::Unsupported(format!("BinaryOp {}", op.name())),
                    },
                }
            }
            R::CheckedBinaryOp(op, pair) => {
                let lhs_ctx = self
                    .operand_place_ty(body, &pair.1)
                    .or_else(|| dest_ty.cloned());
                let rhs_ctx = self
                    .operand_place_ty(body, &pair.0)
                    .or_else(|| dest_ty.cloned());
                let lhs = self.convert_operand(body, &pair.0, lhs_ctx.as_ref())?;
                let rhs = self.convert_operand(body, &pair.1, rhs_ctx.as_ref())?;
                match overflow_base(*op).or_else(|| plain_binop(*op)) {
                    Some(base) => ir::Rvalue::CheckedBinaryOp(base, Box::new((lhs, rhs))),
                    None => ir::Rvalue::Unsupported(format!("CheckedBinaryOp {}", op.name())),
                }
            }
            R::UnaryOp(op, operand) => {
                let ctx = self
                    .operand_place_ty(body, operand)
                    .or_else(|| dest_ty.cloned());
                let v = self.convert_operand(body, operand, ctx.as_ref())?;
                match op {
                    mir::UnOp::Not => ir::Rvalue::UnaryOp(ir::UnOp::Not, v),
                    mir::UnOp::Neg => ir::Rvalue::UnaryOp(ir::UnOp::Neg, v),
                    mir::UnOp::PtrMetadata => {
                        ir::Rvalue::Unsupported("UnaryOp PtrMetadata".to_string())
                    }
                }
            }
            R::Cast(kind, operand, target) => {
                let ctx = self
                    .operand_place_ty(body, operand)
                    .or_else(|| dest_ty.cloned());
                let v = self.convert_operand(body, operand, ctx.as_ref())?;
                match kind {
                    mir::CastKind::IntToInt => match self.convert_ty(target, "cast target") {
                        Ok(t) => ir::Rvalue::Cast(v, t),
                        Err(_) => ir::Rvalue::Unsupported(format!(
                            "Cast IntToInt to {}",
                            target.kind_name()
                        )),
                    },
                    other => ir::Rvalue::Unsupported(format!("Cast {}", other.name())),
                }
            }
            R::Aggregate(kind, ops) => match &**kind {
                mir::AggregateKind::Tuple | mir::AggregateKind::Array(_) => {
                    let elem_tys: Vec<mir::Ty> = match dest_ty {
                        Some(mir::Ty::Tuple(ts)) => ts.clone(),
                        _ => Vec::new(),
                    };
                    let mut items = Vec::with_capacity(ops.len());
                    for (i, op) in ops.iter().enumerate() {
                        let ctx = elem_tys.get(i);
                        items.push(self.convert_operand(body, op, ctx)?);
                    }
                    ir::Rvalue::AggregateTuple(items)
                }
                other => ir::Rvalue::Unsupported(format!("Aggregate {}", other.kind_name())),
            },
            R::Repeat(op, count) => {
                let elem_ty = match dest_ty {
                    Some(mir::Ty::Tuple(ts)) => ts.first().cloned(),
                    Some(mir::Ty::Array(elem, _)) => Some((**elem).clone()),
                    _ => None,
                };
                let v = self.convert_operand(body, op, elem_ty.as_ref())?;
                let n = usize::try_from(*count).unwrap_or(usize::MAX);
                ir::Rvalue::AggregateTuple(vec![v; n])
            }
            R::Ref(_, _, place) => ir::Rvalue::Ref(self.convert_place(place)?),
            R::CopyForDeref(place) => {
                ir::Rvalue::Use(ir::Operand::Copy(self.convert_place(place)?))
            }
            R::RawPtr(_, _) => ir::Rvalue::Unsupported("RawPtr rvalue".to_string()),
            R::ThreadLocalRef(def) => ir::Rvalue::Unsupported(format!("ThreadLocalRef {def}")),
            R::Len(_) => ir::Rvalue::Unsupported("Len rvalue".to_string()),
            R::NullaryOp(op, _) => ir::Rvalue::Unsupported(format!("NullaryOp {}", op.kind_name())),
            R::Discriminant(_) => ir::Rvalue::Unsupported("Discriminant rvalue".to_string()),
            R::ShallowInitBox(_, _) => ir::Rvalue::Unsupported("ShallowInitBox rvalue".to_string()),
        })
    }

    // ----- statements & terminators ------------------------------------------

    fn convert_statement(
        &self,
        body: &mir::Body,
        stmt: &mir::Statement,
    ) -> Result<ir::Statement, ConvertError> {
        use mir::StatementKind as S;
        Ok(match &stmt.kind {
            S::Assign(pair) => {
                let (place, rv) = &**pair;
                let dest_ty = self.place_ty(body, place);
                let ir_place = self.convert_place(place)?;
                let ir_rv = self.convert_rvalue(body, rv, dest_ty.as_ref())?;
                ir::Statement::assign(ir_place, ir_rv)
            }
            S::StorageLive(l) => ir::Statement {
                kind: ir::StatementKind::StorageLive(ir::Local(l.index())),
            },
            S::StorageDead(l) => ir::Statement {
                kind: ir::StatementKind::StorageDead(ir::Local(l.index())),
            },
            // Execution no-ops in rustc MIR (borrow-check / bookkeeping
            // only): an executor treats them as `Nop`.
            S::Nop
            | S::FakeRead(..)
            | S::PlaceMention(_)
            | S::Retag(..)
            | S::AscribeUserType(..)
            | S::Coverage
            | S::ConstEvalCounter => ir::Statement::nop(),
            S::SetDiscriminant { .. } => ir::Statement {
                kind: ir::StatementKind::Unsupported("SetDiscriminant".to_string()),
            },
            S::Intrinsic(intr) => ir::Statement {
                kind: ir::StatementKind::Unsupported(format!("Intrinsic {}", intr.kind_name())),
            },
        })
    }

    fn resolve_callee(&self, func: &mir::Operand) -> Option<String> {
        let mir::Operand::Constant(c) = func else {
            return None;
        };
        match &c.literal {
            mir::ConstKind::Ty(mir::Ty::FnDef(def, _)) => self.def_names.get(def).cloned(),
            mir::ConstKind::Unevaluated { def, .. } => self.def_names.get(def).cloned(),
            _ => None,
        }
    }

    fn convert_terminator(
        &self,
        body: &mir::Body,
        kind: &mir::TerminatorKind,
    ) -> Result<ir::TerminatorKind, ConvertError> {
        use mir::TerminatorKind as T;
        Ok(match kind {
            T::Goto { target } => ir::TerminatorKind::Goto {
                target: ir::BasicBlock(target.index()),
            },
            T::SwitchInt { discr, targets } => {
                let ctx = self.operand_place_ty(body, discr);
                let discr = self.convert_operand(body, discr, ctx.as_ref())?;
                let pairs = targets
                    .values
                    .iter()
                    .copied()
                    .zip(targets.targets.iter().map(|b| ir::BasicBlock(b.index())))
                    .collect();
                ir::TerminatorKind::SwitchInt {
                    discr,
                    targets: pairs,
                    otherwise: ir::BasicBlock(targets.otherwise.index()),
                }
            }
            T::Return => ir::TerminatorKind::Return,
            T::Unreachable => ir::TerminatorKind::Unreachable,
            T::Call {
                func,
                args,
                destination,
                target,
                ..
            } => match self.resolve_callee(func) {
                Some(name) => {
                    let callee = self.bodies.get(&name);
                    let mut ir_args = Vec::with_capacity(args.len());
                    for (i, arg) in args.iter().enumerate() {
                        let ctx =
                            callee.and_then(|b| b.local_decls.get(i + 1).map(|d| d.ty.clone()));
                        ir_args.push(self.convert_operand(body, arg, ctx.as_ref())?);
                    }
                    ir::TerminatorKind::Call {
                        func: name,
                        args: ir_args,
                        destination: self.convert_place(destination)?,
                        target: target.map(|b| ir::BasicBlock(b.index())),
                    }
                }
                None => ir::TerminatorKind::Unsupported(
                    "Call to unresolved function operand".to_string(),
                ),
            },
            T::Assert {
                cond,
                expected,
                msg,
                target,
                ..
            } => ir::TerminatorKind::Assert {
                cond: self.convert_operand(body, cond, Some(&mir::Ty::Bool))?,
                expected: *expected,
                target: ir::BasicBlock(target.index()),
                message: assert_message_text(msg),
            },
            // The imaginary edge of a false edge is never taken at run
            // time; execution continues at the real target.
            T::FalseEdge { real_target, .. } | T::FalseUnwind { real_target, .. } => {
                ir::TerminatorKind::Goto {
                    target: ir::BasicBlock(real_target.index()),
                }
            }
            T::Drop { .. } => ir::TerminatorKind::Unsupported("Drop".to_string()),
            T::Resume => ir::TerminatorKind::Unsupported("Resume".to_string()),
            T::Terminate(k) => ir::TerminatorKind::Unsupported(format!("Terminate {k:?}")),
            T::Yield { .. } => ir::TerminatorKind::Unsupported("Yield".to_string()),
            T::CoroutineDrop => ir::TerminatorKind::Unsupported("CoroutineDrop".to_string()),
            T::InlineAsm { .. } => ir::TerminatorKind::Unsupported("InlineAsm".to_string()),
            T::TailCall { .. } => ir::TerminatorKind::Unsupported("TailCall".to_string()),
        })
    }
}

// ---------------------------------------------------------------------------

fn convert_int_ty(ty: mir::IntTy) -> ir::IntTy {
    match ty {
        mir::IntTy::I8 => ir::IntTy::I8,
        mir::IntTy::I16 => ir::IntTy::I16,
        mir::IntTy::I32 => ir::IntTy::I32,
        mir::IntTy::I64 => ir::IntTy::I64,
        mir::IntTy::I128 => ir::IntTy::I128,
        mir::IntTy::Isize => ir::IntTy::Isize,
    }
}

fn convert_uint_ty(ty: mir::UintTy) -> ir::UintTy {
    match ty {
        mir::UintTy::U8 => ir::UintTy::U8,
        mir::UintTy::U16 => ir::UintTy::U16,
        mir::UintTy::U32 => ir::UintTy::U32,
        mir::UintTy::U64 => ir::UintTy::U64,
        mir::UintTy::U128 => ir::UintTy::U128,
        mir::UintTy::Usize => ir::UintTy::Usize,
    }
}

/// Sign-extend the low `bits` of `data` into an `i128`.
fn sign_extend(data: u128, bits: u32) -> i128 {
    if bits == 0 || bits >= 128 {
        return data as i128;
    }
    let shift = 128 - bits;
    ((data << shift) as i128) >> shift
}

/// Human-readable text for a MIR `Assert` message: the custom string when
/// there is one, otherwise the message kind (the operands rustc embeds
/// are not rendered — the oracle only needs a stable panic message).
fn assert_message_text(msg: &mir::AssertMessage) -> String {
    match msg {
        mir::AssertMessage::Custom(s) => s.clone(),
        other => format!("assertion failed: {}", other.kind_name()),
    }
}

/// The IR operator for a `mir` `*WithOverflow` operator. These yield
/// `(value, overflowed)` both as a plain `BinaryOp` and inside
/// `CheckedBinaryOp`.
fn overflow_base(op: mir::BinOp) -> Option<ir::BinOp> {
    use mir::BinOp as M;
    Some(match op {
        M::AddWithOverflow => ir::BinOp::Add,
        M::SubWithOverflow => ir::BinOp::Sub,
        M::MulWithOverflow => ir::BinOp::Mul,
        _ => return None,
    })
}

/// The IR operator for a plain (non-checked) `mir` binary operator.
/// `Offset` (pointer arithmetic) and `Cmp` (three-way) have no IR
/// equivalent and yield `None`.
fn plain_binop(op: mir::BinOp) -> Option<ir::BinOp> {
    use mir::BinOp as M;
    Some(match op {
        M::Add | M::AddUnchecked => ir::BinOp::Add,
        M::Sub | M::SubUnchecked => ir::BinOp::Sub,
        M::Mul | M::MulUnchecked => ir::BinOp::Mul,
        M::Div => ir::BinOp::Div,
        M::Rem => ir::BinOp::Rem,
        M::BitXor => ir::BinOp::BitXor,
        M::BitAnd => ir::BinOp::BitAnd,
        M::BitOr => ir::BinOp::BitOr,
        M::Shl | M::ShlUnchecked => ir::BinOp::Shl,
        M::Shr | M::ShrUnchecked => ir::BinOp::Shr,
        M::Eq => ir::BinOp::Eq,
        M::Lt => ir::BinOp::Lt,
        M::Le => ir::BinOp::Le,
        M::Ne => ir::BinOp::Ne,
        M::Ge => ir::BinOp::Ge,
        M::Gt => ir::BinOp::Gt,
        M::Offset | M::Cmp => return None,
        // *WithOverflow in a plain position is handled by `overflow_base`
        // before this function is consulted.
        M::AddWithOverflow | M::SubWithOverflow | M::MulWithOverflow => return None,
    })
}
