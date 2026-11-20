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
//!   array value); `AggregateKind::Adt` becomes a variant + fields value.
//!   ADT aggregate operands are typed by the fields' declared types,
//!   recovered program-wide from `Field` projections (an aggregate
//!   rvalue itself carries no field types in this MIR form); without
//!   that, a non-negative literal in a signed field would be read as
//!   unsigned by the no-context scalar rule and the stored field would
//!   mismatch its declared type on first use.
//! * Calls to functions without converted bodies resolve through the
//!   driver's full name table: runtime-supported callees
//!   ([`crate::builtins`]) become `CallBuiltin`; anything else stays an
//!   unsupported call that names the function.
//! * `Drop` converts only for types whose drop glue provably has no
//!   guest-observable effect in this value model (plain data, `String`,
//!   modelled std containers/iterators, and compositions of those);
//!   other drops stay explicit errors.

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
    /// `DefId` -> display name for *every* named item the driver
    /// knows, including functions and types without converted bodies
    /// (std callees, ADT types). Used to name ADT locals and to
    /// resolve calls into runtime support.
    pub fn_names: HashMap<mir::DefId, String>,
    /// Discriminant values of the enum ADTs in the program, keyed by
    /// `DefId` (variant index -> discriminant value), as recorded by
    /// the driver. `Discriminant` rvalues are evaluated with these
    /// values so they agree with the `SwitchInt` branch keys, which
    /// carry discriminant values rather than variant indices.
    pub adt_discriminants: mir::AdtDiscriminants,
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
    /// type model cannot represent (trait objects, closures,
    /// coroutines, `f16` / `f128`, ...).
    UnsupportedType { context: String, ty: String },
    /// A constant form the IR cannot represent (pointers, `f16` /
    /// `f128` float constants, unevaluated / type constants outside
    /// call position, ...).
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

/// Resource bound on `Rvalue::Repeat` element counts, mirroring the
/// interpreter's step / call-depth limits: past this many elements the
/// aggregate is not materialised and the rvalue converts to an
/// explicit unsupported error instead of aborting on allocation.
const MAX_REPEAT_ELEMENTS: u64 = 1_000_000;

/// Reduce a raw `SwitchInt` target value to the canonical form the
/// interpreter compares against. MIR stores target values as the
/// discriminant type's bit pattern, zero-extended to 128 bits
/// (a `-2i32` branch key is stored as 4294967294), while the
/// interpreter holds signed values sign-extended (`switch_bits`),
/// so a signed type's keys are sign-extended here from the type's
/// width. Unsigned and non-integer discriminants already agree.
/// The JIT applies the same normalisation in its lowering.
fn normalize_switch_value(v: u128, ty: Option<&mir::Ty>) -> u128 {
    match ty {
        Some(mir::Ty::Int(it)) => {
            let bits = it.bits();
            if bits >= 128 {
                v
            } else {
                let shift = 128 - bits;
                (((v << shift) as i128) >> shift) as u128
            }
        }
        _ => v,
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

/// The outcome of a run that also captures the guest's stdout: the
/// run result plus everything the guest printed before it returned
/// or failed (a native program's output is likewise observable when
/// it panics).
#[derive(Debug, Clone, PartialEq)]
pub struct CapturedRun {
    pub result: Result<Value, MirRunError>,
    pub stdout: String,
}

/// Like [`interpret_mir`], but also returns the captured guest stdout.
pub fn interpret_mir_captured(
    program: &MirProgram,
    entry: &str,
    args: Vec<Value>,
) -> CapturedRun {
    match convert_program(program) {
        Ok(ir_program) => {
            let mut interp = Interpreter::new(&ir_program);
            let result = interp.run(entry, args).map_err(MirRunError::from);
            CapturedRun {
                result,
                stdout: interp.output().to_string(),
            }
        }
        Err(e) => CapturedRun {
            result: Err(MirRunError::from(e)),
            stdout: String::new(),
        },
    }
}

// ---------------------------------------------------------------------------

/// One ADT field slot whose declared type was recovered from a
/// `Field` projection somewhere in the program.
#[derive(PartialEq)]
struct AdtFieldKey {
    def: mir::DefId,
    /// The ADT instantiation's generic arguments (the projection's
    /// field type is already substituted for them).
    generic_args: Vec<mir::Ty>,
    variant: u32,
    field: u32,
}

/// Declared field types of ADTs, recovered from `Field` projections.
///
/// An `Aggregate` rvalue in this MIR form does not carry the types of
/// the fields it fills — the operands are bare values — but every
/// access to a field through a place does: `PlaceElem::Field` holds
/// the field's (substituted) declared type. Scanning every place in
/// the program before conversion therefore yields, for each
/// (ADT instantiation, variant, field) slot that is ever projected,
/// the type its stored value must carry. Aggregate conversion types
/// its operands by this table, so a field's value is stored with its
/// declared tag from the start instead of being re-derived — wrongly,
/// for signed types — from the literal's bit pattern.
#[derive(Default)]
struct AdtFieldTys {
    entries: Vec<(AdtFieldKey, mir::Ty)>,
}

impl AdtFieldTys {
    fn record(&mut self, key: AdtFieldKey, ty: mir::Ty) {
        // Repeated projections of one slot must agree (they are the
        // same declared field); the first sighting wins so a
        // malformed program cannot flip a field's type mid-conversion.
        if self.entries.iter().any(|(k, _)| *k == key) {
            return;
        }
        self.entries.push((key, ty));
    }

    fn get(
        &self,
        def: mir::DefId,
        generic_args: &[mir::Ty],
        variant: u32,
        field: u32,
    ) -> Option<&mir::Ty> {
        self.entries
            .iter()
            .find(|(k, _)| {
                k.def == def
                    && k.generic_args.as_slice() == generic_args
                    && k.variant == variant
                    && k.field == field
            })
            .map(|(_, ty)| ty)
    }
}

/// Walk one place's projection chain the way [`Converter::place_ty`]
/// does, recording the declared type of every ADT field stepped into.
fn scan_place_fields(table: &mut AdtFieldTys, body: &mir::Body, place: &mir::Place) {
    let Some(decl) = body.local_decls.get(place.local.index()) else {
        return;
    };
    let mut ty = decl.ty.clone();
    let mut variant = 0u32;
    for elem in &place.projection {
        match elem {
            mir::PlaceElem::Field(idx, field_ty) => {
                if let mir::Ty::Adt(def, args) = &ty {
                    table.record(
                        AdtFieldKey {
                            def: *def,
                            generic_args: args.clone(),
                            variant,
                            field: *idx,
                        },
                        field_ty.clone(),
                    );
                }
                ty = field_ty.clone();
                variant = 0;
            }
            mir::PlaceElem::Downcast(_, v) => variant = *v,
            mir::PlaceElem::Deref => match ty {
                mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => {
                    ty = *inner;
                    variant = 0;
                }
                _ => return,
            },
            mir::PlaceElem::Index(_) | mir::PlaceElem::ConstantIndex { .. } => match ty {
                mir::Ty::Array(elem, _) | mir::Ty::Slice(elem) => {
                    ty = *elem;
                    variant = 0;
                }
                _ => return,
            },
            mir::PlaceElem::Subslice { .. } => {}
            mir::PlaceElem::OpaqueCast(t) | mir::PlaceElem::Subtype(t) => {
                ty = t.clone();
                variant = 0;
            }
        }
    }
}

fn scan_operand_fields(table: &mut AdtFieldTys, body: &mir::Body, op: &mir::Operand) {
    if let Some(place) = op.place() {
        scan_place_fields(table, body, place);
    }
}

fn scan_rvalue_fields(table: &mut AdtFieldTys, body: &mir::Body, rv: &mir::Rvalue) {
    use mir::Rvalue as R;
    match rv {
        R::Use(op)
        | R::Repeat(op, _)
        | R::Cast(_, op, _)
        | R::UnaryOp(_, op)
        | R::ShallowInitBox(op, _) => scan_operand_fields(table, body, op),
        R::BinaryOp(_, pair) | R::CheckedBinaryOp(_, pair) => {
            scan_operand_fields(table, body, &pair.0);
            scan_operand_fields(table, body, &pair.1);
        }
        R::Ref(_, _, place)
        | R::RawPtr(_, place)
        | R::Len(place)
        | R::Discriminant(place)
        | R::CopyForDeref(place) => scan_place_fields(table, body, place),
        R::Aggregate(_, ops) => {
            for op in ops {
                scan_operand_fields(table, body, op);
            }
        }
        R::ThreadLocalRef(_) | R::NullaryOp(..) => {}
    }
}

fn scan_statement_fields(table: &mut AdtFieldTys, body: &mir::Body, stmt: &mir::Statement) {
    use mir::StatementKind as S;
    match &stmt.kind {
        S::Assign(pair) => {
            let (place, rv) = &**pair;
            scan_place_fields(table, body, place);
            scan_rvalue_fields(table, body, rv);
        }
        S::FakeRead(pair) => scan_place_fields(table, body, &pair.1),
        S::SetDiscriminant { place, .. } => scan_place_fields(table, body, place),
        S::Retag(_, place) | S::PlaceMention(place) => scan_place_fields(table, body, place),
        S::AscribeUserType(pair, _) => scan_place_fields(table, body, &pair.0),
        S::Intrinsic(intr) => match &**intr {
            mir::NonDivergingIntrinsic::CopyNonOverlapping { src, dst, count } => {
                scan_operand_fields(table, body, src);
                scan_operand_fields(table, body, dst);
                scan_operand_fields(table, body, count);
            }
            mir::NonDivergingIntrinsic::Assume(op) => scan_operand_fields(table, body, op),
        },
        S::StorageLive(_)
        | S::StorageDead(_)
        | S::Coverage
        | S::ConstEvalCounter
        | S::Nop => {}
    }
}

fn scan_terminator_fields(table: &mut AdtFieldTys, body: &mir::Body, kind: &mir::TerminatorKind) {
    use mir::TerminatorKind as T;
    match kind {
        T::SwitchInt { discr, .. } => scan_operand_fields(table, body, discr),
        T::Call {
            func,
            args,
            destination,
            ..
        } => {
            scan_operand_fields(table, body, func);
            for arg in args {
                scan_operand_fields(table, body, arg);
            }
            scan_place_fields(table, body, destination);
        }
        T::Assert { cond, .. } => scan_operand_fields(table, body, cond),
        T::Drop { place, .. } => scan_place_fields(table, body, place),
        T::Yield {
            value, resume_arg, ..
        } => {
            scan_operand_fields(table, body, value);
            scan_place_fields(table, body, resume_arg);
        }
        T::InlineAsm { operands, .. } => {
            for op in operands {
                scan_operand_fields(table, body, op);
            }
        }
        T::TailCall { func, args } => {
            scan_operand_fields(table, body, func);
            for arg in args {
                scan_operand_fields(table, body, arg);
            }
        }
        T::Goto { .. }
        | T::Resume
        | T::Terminate(_)
        | T::Return
        | T::Unreachable
        | T::CoroutineDrop
        | T::FalseEdge { .. }
        | T::FalseUnwind { .. } => {}
    }
}

fn scan_body_fields(table: &mut AdtFieldTys, body: &mir::Body) {
    for bb in &body.basic_blocks {
        for stmt in &bb.statements {
            scan_statement_fields(table, body, stmt);
        }
        if let Some(term) = &bb.terminator {
            scan_terminator_fields(table, body, &term.kind);
        }
    }
}

struct Converter<'a> {
    /// Function name -> body, for callee argument typing.
    bodies: HashMap<String, &'a mir::Body>,
    /// `DefId` -> function name, for resolving `FnDef` call operands.
    def_names: HashMap<mir::DefId, String>,
    /// `DefId` -> name for every named item (the driver's table,
    /// including body-less callees and ADT types).
    names: &'a HashMap<mir::DefId, String>,
    /// Declared ADT field types recovered from `Field` projections
    /// across the whole program (see [`AdtFieldTys`]).
    adt_fields: AdtFieldTys,
    /// Discriminant tables of the program (see `MirProgram`).
    discriminants: &'a mir::AdtDiscriminants,
}

impl<'a> Converter<'a> {
    fn new(program: &'a MirProgram) -> Self {
        let mut bodies = HashMap::new();
        let mut def_names = HashMap::new();
        let mut adt_fields = AdtFieldTys::default();
        for f in &program.functions {
            bodies.insert(f.name.clone(), &f.body);
            if let Some(def) = f.def_id {
                def_names.insert(def, f.name.clone());
            }
            scan_body_fields(&mut adt_fields, &f.body);
        }
        Converter {
            bodies,
            def_names,
            names: &program.fn_names,
            adt_fields,
            discriminants: &program.adt_discriminants,
        }
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

    /// Display name for a `DefId` (function items, closure bodies,
    /// ADT types): the driver's full name table first, then the
    /// table of converted bodies, then the raw id.
    fn def_display_name(&self, def: &mir::DefId) -> String {
        self.names
            .get(def)
            .or_else(|| self.def_names.get(def))
            .cloned()
            .unwrap_or_else(|| format!("{def:?}"))
    }

    fn convert_ty(&self, ty: &mir::Ty, context: &str) -> Result<ir::Ty, ConvertError> {
        let unsupported = || ConvertError::UnsupportedType {
            context: context.to_string(),
            ty: format!("{} ({ty:?})", ty.kind_name()),
        };
        Ok(match ty {
            mir::Ty::Bool => ir::Ty::Bool,
            mir::Ty::Int(it) => ir::Ty::Int(convert_int_ty(*it)),
            mir::Ty::Uint(ut) => ir::Ty::Uint(convert_uint_ty(*ut)),
            // `f16` / `f128` fall through to the unsupported arm.
            mir::Ty::Float(mir::FloatTy::F32) => ir::Ty::Float(ir::FloatTy::F32),
            mir::Ty::Float(mir::FloatTy::F64) => ir::Ty::Float(ir::FloatTy::F64),
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
            mir::Ty::Adt(def, _) => ir::Ty::Adt(self.def_display_name(def)),
            // Function pointers carry no signature in the IR; the
            // callee's arity is checked at the call site.
            mir::Ty::FnPtr(_) => ir::Ty::FnPtr,
            // Function items and closures are named by the body they
            // call, so a value of this type can be invoked.
            mir::Ty::FnDef(def, _) => ir::Ty::FnDef(self.def_display_name(def)),
            mir::Ty::Closure(def, args) => {
                let upvars = match args.get(2) {
                    Some(mir::Ty::Tuple(ts)) => ts.len(),
                    _ => 0,
                };
                ir::Ty::Closure(self.def_display_name(def), upvars)
            }
            mir::Ty::Ref(_, inner, _) => {
                ir::Ty::Ref(Box::new(self.convert_ty(inner, context)?))
            }
            // Raw pointers share the reference representation.
            mir::Ty::RawPtr(inner, _) => {
                ir::Ty::Ref(Box::new(self.convert_ty(inner, context)?))
            }
            mir::Ty::Str => ir::Ty::Str,
            mir::Ty::Array(elem, len) => {
                let n = match len {
                    mir::ArrayLen::Known(n) => *n as u64,
                    _ => return Err(unsupported()),
                };
                ir::Ty::Array(Box::new(self.convert_ty(elem, context)?), n)
            }
            mir::Ty::Slice(elem) => ir::Ty::Slice(Box::new(self.convert_ty(elem, context)?)),
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
                mir::PlaceElem::ConstantIndex {
                    offset, from_end, ..
                } => ir::ProjectionElem::ConstantIndex {
                    offset: *offset as u64,
                    from_end: *from_end,
                },
                mir::PlaceElem::Downcast(_, variant) => ir::ProjectionElem::Downcast(*variant),
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
            // A function item in operand position is a value (the
            // zero-sized fn item, reifiable to a function pointer):
            // it becomes a callable constant naming the function.
            mir::ConstKind::Ty(mir::Ty::FnDef(def, _)) => {
                Ok(ir::Constant::FnDef(self.def_display_name(def)))
            }
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
            mir::ConstValue::Str(s) => ir::Constant::Str(s.clone()),
            mir::ConstValue::Bytes(b) => {
                // The bytes are the elements themselves only for
                // byte-sized elements (the fmt pieces template is the
                // canonical case); wider elements are packed
                // little-endian encodings this layer does not decode.
                if let Some(size) = pointee_elem_size(ctx) {
                    if size != 1 {
                        return Err(ConvertError::UnsupportedConstant {
                            kind: format!("byte constant with {size}-byte elements"),
                        });
                    }
                }
                ir::Constant::Bytes(b.clone())
            }
            mir::ConstValue::Slice { data, meta } => {
                // A slice constant whose metadata equals its byte
                // length is a byte slice.
                if *meta as usize == data.len() {
                    ir::Constant::Bytes(data.clone())
                } else {
                    return Err(ConvertError::UnsupportedConstant {
                        kind: "slice constant with non-byte elements".to_string(),
                    });
                }
            }
            mir::ConstValue::Tuple(fields) => ir::Constant::Tuple(
                fields
                    .iter()
                    .map(|f| self.convert_const_value(f, None))
                    .collect::<Result<_, _>>()?,
            ),
            mir::ConstValue::Adt { variant, fields } => ir::Constant::Adt {
                variant: *variant,
                fields: fields
                    .iter()
                    .map(|f| self.convert_const_value(f, None))
                    .collect::<Result<_, _>>()?,
            },
            mir::ConstValue::Ref(inner) => {
                ir::Constant::Ref(Box::new(self.convert_const_value(inner, None)?))
            }
            mir::ConstValue::SliceVal(elems) => ir::Constant::SliceVal(
                elems
                    .iter()
                    .map(|e| self.convert_const_value(e, None))
                    .collect::<Result<_, _>>()?,
            ),
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
            mir::Scalar::Float { bits, ty } => match ty {
                mir::FloatTy::F32 => {
                    Ok(ir::Constant::Float(f32::from_bits(*bits as u32) as f64))
                }
                mir::FloatTy::F64 => Ok(ir::Constant::Float(f64::from_bits(*bits as u64))),
                other => Err(ConvertError::UnsupportedConstant {
                    kind: format!("{other:?} float constant"),
                }),
            },
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

    /// Convert the two operands of a binary operation under one shared
    /// typing context.
    ///
    /// In MIR both operands of a binary operation have the same type
    /// (shift amounts aside; a shift amount's numeric value does not
    /// depend on which integer type it is read as), and a literal
    /// operand carries no type of its own in `sparagmos-mir` — how its
    /// bits are interpreted must come from the operation's context:
    ///
    /// 1. A place operand pins the shared type: its declared type is
    ///    the operation's operand type.
    /// 2. Otherwise the result type gives it away. A checked operation
    ///    (a `CheckedBinaryOp`, or a `BinaryOp` with a `*WithOverflow`
    ///    operator) yields the `(T, bool)` pair, so the operand type is
    ///    the tuple's first element. A plain operation yields the
    ///    operand type directly — except comparisons, which yield
    ///    `bool` and say nothing about their operands. (Typing a
    ///    literal by the `bool` destination used to collapse every
    ///    nonzero literal to `true`.)
    /// 3. Otherwise — a comparison of two literals — no declared type
    ///    exists anywhere. The pair still needs ONE interpretation:
    ///    guessing per literal can yield an `Int`/`Uint` pair that
    ///    cannot even be compared. Both literals are read in the single
    ///    interpretation of [`literal_pair_ty`]. This is the shape of
    ///    rustc's expanded Div/Rem checks: `Eq(rhs, 0)`, `Eq(rhs, -1)`,
    ///    `Eq(lhs, MIN)` over two literals.
    fn convert_binop_operands(
        &self,
        body: &mir::Body,
        pair: &(mir::Operand, mir::Operand),
        dest_ty: Option<&mir::Ty>,
        checked_result: bool,
        comparison: bool,
    ) -> Result<(ir::Operand, ir::Operand), ConvertError> {
        let ctx = self
            .operand_place_ty(body, &pair.0)
            .or_else(|| self.operand_place_ty(body, &pair.1))
            .or_else(|| {
                if checked_result {
                    match dest_ty {
                        Some(mir::Ty::Tuple(ts)) => ts.first().cloned(),
                        _ => None,
                    }
                } else if comparison {
                    None
                } else {
                    dest_ty.cloned()
                }
            })
            .or_else(|| literal_pair_ty(&pair.0, &pair.1));
        let lhs = self.convert_operand(body, &pair.0, ctx.as_ref())?;
        let rhs = self.convert_operand(body, &pair.1, ctx.as_ref())?;
        Ok((lhs, rhs))
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
                let (lhs, rhs) = self.convert_binop_operands(
                    body,
                    pair,
                    dest_ty,
                    overflow_base(*op).is_some(),
                    is_comparison(*op),
                )?;
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
                let (lhs, rhs) = self.convert_binop_operands(
                    body,
                    pair,
                    dest_ty,
                    true,
                    is_comparison(*op),
                )?;
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
                    mir::UnOp::PtrMetadata => ir::Rvalue::UnaryOp(ir::UnOp::PtrMetadata, v),
                }
            }
            R::Cast(kind, operand, target) => {
                let ctx = self
                    .operand_place_ty(body, operand)
                    .or_else(|| dest_ty.cloned());
                let v = self.convert_operand(body, operand, ctx.as_ref())?;
                match kind {
                    mir::CastKind::IntToInt
                    | mir::CastKind::FloatToInt
                    | mir::CastKind::FloatToFloat
                    | mir::CastKind::IntToFloat => match self.convert_ty(target, "cast target") {
                        Ok(t) => ir::Rvalue::Cast(v, t),
                        Err(_) => ir::Rvalue::Unsupported(format!(
                            "Cast {} to {}",
                            kind.name(),
                            target.kind_name()
                        )),
                    },
                    mir::CastKind::PointerCoercion(mir::PointerCoercion::Unsize) => {
                        ir::Rvalue::Unsize(v)
                    }
                    // These coercions do not change the reference
                    // value in this model (a reference carries no
                    // mutability or pointee type at run time).
                    mir::CastKind::PointerCoercion(
                        mir::PointerCoercion::MutToConstPointer
                        | mir::PointerCoercion::ArrayToPointer,
                    ) => ir::Rvalue::Use(v),
                    mir::CastKind::PtrToPtr => ir::Rvalue::Use(v),
                    // Reifying a function item into a function
                    // pointer does not change the value in this
                    // model: both are the callable value naming the
                    // function.
                    mir::CastKind::PointerCoercion(
                        mir::PointerCoercion::ReifyFnPointer,
                    ) => ir::Rvalue::Use(v),
                    // A non-capturing closure coerced to a function
                    // pointer: with a place operand, the closure
                    // value itself is the callable. The constant
                    // form names no closure (the zero-sized constant
                    // does not preserve the closure type), so it
                    // stays an explicit unsupported.
                    mir::CastKind::PointerCoercion(
                        mir::PointerCoercion::ClosureFnPointer,
                    ) => match operand {
                        mir::Operand::Copy(_) | mir::Operand::Move(_) => ir::Rvalue::Use(v),
                        mir::Operand::Constant(_) => ir::Rvalue::Unsupported(
                            "Cast ClosureFnPointer of a zero-sized closure constant (the closure type is not preserved in the constant)"
                                .to_string(),
                        ),
                    },
                    mir::CastKind::BoxDerefTransmute => ir::Rvalue::BoxDerefTransmute(v),
                    other => ir::Rvalue::Unsupported(format!("Cast {}", other.name())),
                }
            }
            R::Aggregate(kind, ops) => match &**kind {
                // The empty tuple is the unit value: its type
                // converts to `Ty::Unit`, so the aggregate must
                // produce `Value::Unit`, not an empty tuple value.
                mir::AggregateKind::Tuple if ops.is_empty() => {
                    ir::Rvalue::Use(ir::Operand::Constant(ir::Constant::Unit))
                }
                mir::AggregateKind::Tuple | mir::AggregateKind::Array(_) => {
                    let elem_tys: Vec<mir::Ty> = match dest_ty {
                        Some(mir::Ty::Tuple(ts)) => ts.clone(),
                        Some(mir::Ty::Array(elem, _)) => {
                            vec![(**elem).clone(); ops.len()]
                        }
                        _ => Vec::new(),
                    };
                    let mut items = Vec::with_capacity(ops.len());
                    for (i, op) in ops.iter().enumerate() {
                        let ctx = elem_tys.get(i);
                        items.push(self.convert_operand(body, op, ctx)?);
                    }
                    ir::Rvalue::AggregateTuple(items)
                }
                mir::AggregateKind::Adt {
                    def,
                    variant_index,
                    generic_args,
                    ..
                } => {
                    // For `Range` / `RangeInclusive` the two bound
                    // fields are the ADT's single generic parameter;
                    // typing literal bounds by it keeps them from
                    // being read as unsigned by default.
                    let bound_ty = {
                        let name = self.names.get(def).map(String::as_str);
                        if matches!(
                            name,
                            Some("std::ops::Range" | "std::ops::RangeInclusive")
                        ) {
                            generic_args.first()
                        } else {
                            None
                        }
                    };
                    let mut fields = Vec::with_capacity(ops.len());
                    for (i, op) in ops.iter().enumerate() {
                        // The field's declared type, recovered from
                        // `Field` projections program-wide, is the
                        // primary context: it is what the stored
                        // value will be read as. The Range bound rule
                        // stays as the fallback for fields no place
                        // ever projects (their values are consumed by
                        // runtime support, not by MIR places).
                        let declared = self.adt_fields.get(
                            *def,
                            generic_args,
                            *variant_index,
                            i as u32,
                        );
                        let ctx = declared.or_else(|| if i < 2 { bound_ty } else { None });
                        fields.push(self.convert_operand(body, op, ctx)?);
                    }
                    ir::Rvalue::AggregateAdt {
                        variant: *variant_index,
                        operands: fields,
                    }
                }
                mir::AggregateKind::Closure { def, generic_args } => {
                    // The closure type's third generic argument is
                    // the upvars tuple; its element types type the
                    // capture operands.
                    let upvar_tys: Vec<mir::Ty> = match generic_args.get(2) {
                        Some(mir::Ty::Tuple(ts)) => ts.clone(),
                        _ => Vec::new(),
                    };
                    let mut fields = Vec::with_capacity(ops.len());
                    for (i, op) in ops.iter().enumerate() {
                        fields.push(self.convert_operand(body, op, upvar_tys.get(i))?);
                    }
                    ir::Rvalue::ClosureAggregate {
                        def: self.def_display_name(def),
                        operands: fields,
                    }
                }
                other => ir::Rvalue::Unsupported(format!("Aggregate {}", other.kind_name())),
            },
            R::Repeat(op, count) => {
                // A repeat count is guest-controlled input: a huge
                // count (e.g. `[0u8; 1 << 40]`) would make the
                // aggregate allocation abort the host process instead
                // of failing the run. Like the interpreter's step and
                // call-depth limits, the element count is a resource
                // bound; past it the rvalue stays an explicit
                // unsupported error.
                if *count > MAX_REPEAT_ELEMENTS {
                    ir::Rvalue::Unsupported(format!("Repeat of {count} elements"))
                } else {
                    let elem_ty = match dest_ty {
                        Some(mir::Ty::Tuple(ts)) => ts.first().cloned(),
                        Some(mir::Ty::Array(elem, _)) => Some((**elem).clone()),
                        _ => None,
                    };
                    let v = self.convert_operand(body, op, elem_ty.as_ref())?;
                    ir::Rvalue::AggregateTuple(vec![v; *count as usize])
                }
            }
            R::Ref(_, _, place) => ir::Rvalue::Ref(self.convert_place(place)?),
            R::CopyForDeref(place) => ir::Rvalue::CopyForDeref(self.convert_place(place)?),
            // Raw pointer creation shares the reference representation.
            R::RawPtr(_, place) => ir::Rvalue::Ref(self.convert_place(place)?),
            R::ThreadLocalRef(def) => ir::Rvalue::Unsupported(format!("ThreadLocalRef {def}")),
            R::Len(place) => ir::Rvalue::Len(self.convert_place(place)?),
            R::NullaryOp(op, _) => ir::Rvalue::Unsupported(format!("NullaryOp {}", op.kind_name())),
            R::Discriminant(place) => {
                let converted = self.convert_place(place)?;
                // Bake the discriminant table of the enum into the
                // rvalue: at execution time the place type is no
                // longer available, and the table is what separates
                // the discriminant value from the variant index.
                // No table (hand-built programs, non-ADT places)
                // leaves the table empty, which the interpreter
                // reads as "values are the variant indices".
                let discriminants = match self.place_ty(body, place) {
                    Some(mir::Ty::Adt(def, _)) => {
                        self.discriminants.get(&def).cloned().unwrap_or_default()
                    }
                    _ => Vec::new(),
                };
                ir::Rvalue::Discriminant(converted, discriminants)
            }
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
            S::SetDiscriminant {
                place,
                variant_index,
            } => ir::Statement {
                kind: ir::StatementKind::SetDiscriminant {
                    place: self.convert_place(place)?,
                    variant: *variant_index,
                },
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

    /// Resolve a callee that has no converted body through the
    /// driver's full name table (std/core functions): its display
    /// name, its rendered generic arguments, and the raw generic
    /// arguments (used to type literal arguments whose type the
    /// callee's signature pins to a generic parameter).
    fn resolve_external_callee(
        &self,
        func: &mir::Operand,
    ) -> Option<(String, Vec<String>, Vec<mir::Ty>)> {
        let mir::Operand::Constant(c) = func else {
            return None;
        };
        match &c.literal {
            mir::ConstKind::Ty(mir::Ty::FnDef(def, generic_args)) => {
                let name = self.names.get(def).cloned()?;
                let tags = generic_args
                    .iter()
                    .map(|t| crate::builtins::ty_tag(t, self.names))
                    .collect();
                Some((name, tags, generic_args.clone()))
            }
            _ => None,
        }
    }

    /// Whether this call's func operand names an `Fn`-family trait
    /// method (`Fn::call`, `FnMut::call_mut`, `FnOnce::call_once` in
    /// their `std::ops` / `core::ops` spellings): returns the call
    /// kind and the callee's raw generic arguments, whose first two
    /// entries are the Self type and the argument tuple type (used
    /// to type the receiver and tuple operands).
    fn resolve_fn_trait_call(
        &self,
        func: &mir::Operand,
    ) -> Option<(ir::FnCallKind, Vec<mir::Ty>)> {
        let (name, _, raw_generic_args) = self.resolve_external_callee(func)?;
        let canonical = crate::builtins::canonical_name(&name);
        let kind = match canonical.as_str() {
            "std::ops::Fn::call" | "core::ops::Fn::call" => ir::FnCallKind::Call,
            "std::ops::FnMut::call_mut" | "core::ops::FnMut::call_mut" => {
                ir::FnCallKind::CallMut
            }
            "std::ops::FnOnce::call_once" | "core::ops::FnOnce::call_once" => {
                ir::FnCallKind::CallOnce
            }
            _ => return None,
        };
        Some((kind, raw_generic_args))
    }

    /// Whether dropping a value of this type provably has no
    /// guest-observable effect in this value model: plain data drops
    /// nothing; references drop nothing; the listed std types only
    /// deallocate or drop their (recursively unobservable) contents.
    /// Anything else — user types that may implement `Drop`, closures,
    /// trait objects — stays an explicit error.
    fn drop_is_unobservable(&self, ty: &mir::Ty) -> bool {
        match ty {
            mir::Ty::Bool
            | mir::Ty::Char
            | mir::Ty::Int(_)
            | mir::Ty::Uint(_)
            | mir::Ty::Float(_)
            | mir::Ty::Never
            | mir::Ty::Str => true,
            mir::Ty::Tuple(ts) => ts.iter().all(|t| self.drop_is_unobservable(t)),
            // A closure's drop is the drop of its captures (the
            // upvars tuple is the closure type's third generic
            // argument).
            mir::Ty::Closure(_, args) => match args.get(2) {
                Some(mir::Ty::Tuple(ts)) => {
                    ts.iter().all(|t| self.drop_is_unobservable(t))
                }
                _ => false,
            },
            mir::Ty::Array(elem, _) | mir::Ty::Slice(elem) => self.drop_is_unobservable(elem),
            mir::Ty::Ref(..) | mir::Ty::RawPtr(..) | mir::Ty::FnDef(..) | mir::Ty::FnPtr(_) => {
                true
            }
            mir::Ty::Adt(def, args) => {
                let name = self
                    .names
                    .get(def)
                    .or_else(|| self.def_names.get(def))
                    .map(String::as_str);
                let known = matches!(
                    name,
                    Some(
                        "std::string::String"
                            | "std::vec::Vec"
                            | "std::boxed::Box"
                            | "std::array::IntoIter"
                            | "std::option::Option"
                            | "std::fmt::Arguments"
                            | "core::fmt::rt::Argument"
                            | "std::ops::Range"
                            | "std::ops::RangeInclusive"
                    )
                );
                known && args.iter().all(|t| self.drop_is_unobservable(t))
            }
            _ => false,
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
                    .map(|v| normalize_switch_value(*v, ctx.as_ref()))
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
            } => {
                // Indirect call through a callable value in a place
                // (a function pointer, function item, or closure
                // variable): the func operand is the callable itself.
                if matches!(func, mir::Operand::Copy(_) | mir::Operand::Move(_)) {
                    let func_ty = self.operand_place_ty(body, func);
                    let param_tys = callable_param_tys(func_ty.as_ref());
                    let ir_func = self.convert_operand(body, func, func_ty.as_ref())?;
                    let mut ir_args = Vec::with_capacity(args.len());
                    for (i, arg) in args.iter().enumerate() {
                        ir_args.push(self.convert_operand(body, arg, param_tys.get(i))?);
                    }
                    ir::TerminatorKind::CallIndirect {
                        func: ir_func,
                        args: ir_args,
                        destination: self.convert_place(destination)?,
                        target: target.map(|b| ir::BasicBlock(b.index())),
                    }
                } else if let Some(name) = self.resolve_callee(func) {
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
                } else if let Some((kind, raw_generic_args)) = self
                    .resolve_fn_trait_call(func)
                    .filter(|_| args.len() == 2)
                {
                    // A call through the `Fn` / `FnMut` / `FnOnce`
                    // trait methods — how rustc MIR calls a closure:
                    // receiver (the callable or a reference to it)
                    // plus the argument tuple.
                    let receiver =
                        self.convert_operand(body, &args[0], raw_generic_args.first())?;
                    let tuple =
                        self.convert_operand(body, &args[1], raw_generic_args.get(1))?;
                    ir::TerminatorKind::CallTrait {
                        kind,
                        receiver,
                        args: tuple,
                        destination: self.convert_place(destination)?,
                        target: target.map(|b| ir::BasicBlock(b.index())),
                    }
                } else if let Some((name, generic_args, raw_generic_args)) =
                    self.resolve_external_callee(func)
                {
                    let canonical = crate::builtins::canonical_name(&name);
                    if let Some(kind) = crate::builtins::lookup(&canonical) {
                        let mut ir_args = Vec::with_capacity(args.len());
                        for (i, arg) in args.iter().enumerate() {
                            let ctx = builtin_arg_ctx(kind, &raw_generic_args, i);
                            ir_args.push(self.convert_operand(body, arg, ctx.as_ref())?);
                        }
                        ir::TerminatorKind::CallBuiltin {
                            name: canonical,
                            generic_args,
                            args: ir_args,
                            destination: self.convert_place(destination)?,
                            target: target.map(|b| ir::BasicBlock(b.index())),
                        }
                    } else {
                        ir::TerminatorKind::Unsupported(format!(
                            "Call to {name} (no converted body)"
                        ))
                    }
                } else {
                    ir::TerminatorKind::Unsupported(
                        "Call to unresolved function operand".to_string(),
                    )
                }
            }
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
            T::Drop { place, target, .. }
                if self
                    .place_ty(body, place)
                    .is_some_and(|t| self.drop_is_unobservable(&t)) =>
            {
                ir::TerminatorKind::Drop {
                    place: self.convert_place(place)?,
                    target: ir::BasicBlock(target.index()),
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

/// Whether a `mir` binary operator is a comparison: its result is
/// `bool`, so its destination type is not its operand type.
fn is_comparison(op: mir::BinOp) -> bool {
    use mir::BinOp as M;
    matches!(op, M::Eq | M::Ne | M::Lt | M::Le | M::Gt | M::Ge)
}

/// The shared interpretation for two integer literals whose operation
/// has no declared operand type anywhere (a comparison of two
/// literals): the integer type of their common storage width — signed
/// when either bit pattern has the width's sign bit set (the patterns
/// rustc's expanded checks compare against, `-1` and the type
/// minimum, both do), unsigned otherwise. `None` when the two operands
/// are not integer literals of one standard width.
fn literal_pair_ty(lhs: &mir::Operand, rhs: &mir::Operand) -> Option<mir::Ty> {
    fn scalar_int(op: &mir::Operand) -> Option<(u128, u8)> {
        let mir::Operand::Constant(c) = op else {
            return None;
        };
        let mir::ConstKind::Value(mir::ConstValue::Scalar(mir::Scalar::Int { data, size })) =
            &c.literal
        else {
            return None;
        };
        Some((*data, *size))
    }
    let (ldata, lsize) = scalar_int(lhs)?;
    let (rdata, rsize) = scalar_int(rhs)?;
    if lsize != rsize {
        return None;
    }
    let bits = u32::from(lsize) * 8;
    let sign_bit = |data: u128| bits > 0 && (data >> (bits - 1)) & 1 == 1;
    if sign_bit(ldata) || sign_bit(rdata) {
        int_ty_of_bits(bits).map(mir::Ty::Int)
    } else {
        uint_ty_of_bits(bits).map(mir::Ty::Uint)
    }
}

fn int_ty_of_bits(bits: u32) -> Option<mir::IntTy> {
    Some(match bits {
        8 => mir::IntTy::I8,
        16 => mir::IntTy::I16,
        32 => mir::IntTy::I32,
        64 => mir::IntTy::I64,
        128 => mir::IntTy::I128,
        _ => return None,
    })
}

fn uint_ty_of_bits(bits: u32) -> Option<mir::UintTy> {
    Some(match bits {
        8 => mir::UintTy::U8,
        16 => mir::UintTy::U16,
        32 => mir::UintTy::U32,
        64 => mir::UintTy::U64,
        128 => mir::UintTy::U128,
        _ => return None,
    })
}

/// The typing context for one argument of a runtime-support call.
/// Most builtin arguments are places or string/byte constants, whose
/// conversion does not depend on a context; the exceptions are
/// literal arguments whose type the callee's signature pins to a
/// generic parameter (`RangeInclusive::new`'s bounds are its `Idx`)
/// or to a fixed type (`Argument::from_usize` takes `usize`).
/// The parameter types of a callable place type, for typing call
/// arguments whose literals carry no type of their own: a function
/// pointer's inputs directly; a closure's parameters through its
/// signature (whose single RustCall input is the parameter tuple).
/// Empty when the type carries no signature (function items).
fn callable_param_tys(ty: Option<&mir::Ty>) -> Vec<mir::Ty> {
    match ty {
        Some(mir::Ty::FnPtr(sig)) => sig.inputs.clone(),
        Some(mir::Ty::Closure(_, args)) => match args.get(1) {
            Some(mir::Ty::FnPtr(sig)) => match sig.inputs.first() {
                Some(mir::Ty::Tuple(ts)) => ts.clone(),
                _ => Vec::new(),
            },
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

fn builtin_arg_ctx(
    kind: crate::builtins::Builtin,
    raw_generic_args: &[mir::Ty],
    arg_index: usize,
) -> Option<mir::Ty> {
    use crate::builtins::Builtin;
    match kind {
        Builtin::RangeInclusiveNew => {
            if arg_index < 2 {
                raw_generic_args.first().cloned()
            } else {
                None
            }
        }
        Builtin::ArgumentFromUsize => Some(mir::Ty::Uint(mir::UintTy::Usize)),
        // Lane V: the value arguments of these builtins are the
        // callee's first generic parameter (the element / pointee /
        // cell type), so a literal argument takes that type.
        Builtin::BoxNew | Builtin::ManuallyDropNew | Builtin::CellNew => {
            if arg_index == 0 {
                raw_generic_args.first().cloned()
            } else {
                None
            }
        }
        Builtin::CellSet | Builtin::CellReplace | Builtin::VecPush => {
            if arg_index == 1 {
                raw_generic_args.first().cloned()
            } else {
                None
            }
        }
        // The index argument of `Index` / `IndexMut` is the trait's
        // `Idx` parameter, the callee's second generic argument.
        Builtin::IndexIndex | Builtin::IndexMut => {
            if arg_index == 1 {
                raw_generic_args.get(1).cloned()
            } else {
                None
            }
        }
        _ => None,
    }
}

/// The element size in bytes of the pointee of a constant's context
/// type, when it is a scalar-element array/slice behind any number of
/// references. `None` when it cannot be derived.
fn pointee_elem_size(ctx: Option<&mir::Ty>) -> Option<u64> {
    match ctx {
        Some(mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _)) => {
            pointee_elem_size(Some(inner))
        }
        Some(mir::Ty::Array(elem, _) | mir::Ty::Slice(elem)) => match &**elem {
            mir::Ty::Bool => Some(1),
            mir::Ty::Char => Some(4),
            mir::Ty::Int(it) => Some(u64::from(it.bits()) / 8),
            mir::Ty::Uint(ut) => Some(u64::from(ut.bits()) / 8),
            _ => None,
        },
        _ => None,
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
