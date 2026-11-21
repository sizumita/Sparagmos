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
//! * Conversion is isolated per body: a function whose body cannot
//!   be converted does not fail the whole program. It is recorded
//!   as unconverted with its reason; calling it fails at execution
//!   time with that reason, while the entry function's own failure
//!   stays a whole-run conversion failure.
//! * Functions are keyed by name. Bodies that share one name —
//!   anonymous `const _` items, which the driver names by a single
//!   shared item path — are disambiguated by their `DefId`, which
//!   is also how references to them resolve (see [`program_keys`]).
//! * `Drop` converts when conversion can account for the dropped
//!   type's whole drop glue: plain data needs none, modelled std
//!   containers drop their contents, and a user type's glue is its
//!   converted `Drop` impl body (when the program has one) followed
//!   by its fields' glue. A type whose glue cannot be fully
//!   accounted for (a `Drop` impl without a converted body, a field
//!   whose type no projection anywhere records, a type from another
//!   crate that might hide an impl) keeps the terminator an explicit
//!   error; nothing is dropped silently. An explicit
//!   `std::mem::drop` call is treated the same way: the dropped
//!   type's glue is built at conversion and attached to the call, so
//!   both paths run identical glue.

use std::collections::{HashMap, HashSet};
use std::fmt;

use sparagmos_mir as mir;

use crate::error::InterpError;
use crate::interp::Interpreter;
use crate::ir;
use crate::value::{RefProj, Value};

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
    /// Layouts of the monomorphic ADT instances in the program, as
    /// recorded by the driver from the compiler's layout query
    /// (see [`mir::AdtLayouts`]). Consulted by the `size_of` /
    /// `align_of` / `size_of_val` builtins for ADT type tags.
    pub adt_layouts: mir::AdtLayouts,
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
    /// Two functions in one program that cannot be told apart:
    /// they share a name, and no distinct `DefId` disambiguates
    /// them (see [`program_keys`]).
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
///
/// Conversion is isolated per body: a function whose body cannot be
/// converted is recorded in the program's unconverted table with
/// its failure reason instead of failing the whole conversion (see
/// the module docs). Only a structurally malformed program — two
/// functions that neither name nor `DefId` can tell apart — fails
/// conversion as a whole.
pub fn convert_program(program: &MirProgram) -> Result<ir::Program, ConvertError> {
    Ok(convert_all(program)?.0)
}

/// The key each function of `program` is stored under in the
/// converted program.
///
/// A function's key is its name — except when several functions
/// share one name. That is the case for anonymous `const _` items:
/// the driver names every one of them by the same item path (a
/// crate's `::_`), while their bodies are distinct items. Bodies
/// sharing a name are disambiguated by their `DefId`, which is
/// also how every reference to them resolves (call and constant
/// operands name a `DefId`, never the shared display name), so a
/// reference still finds exactly its own body and the name-keyed
/// tables stay coherent. A shared name whose bodies cannot be
/// told apart by `DefId` — a missing id, or the same id twice —
/// is a genuine duplicate and fails conversion as a whole.
fn program_keys(program: &MirProgram) -> Result<Vec<String>, ConvertError> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for f in &program.functions {
        *counts.entry(f.name.as_str()).or_default() += 1;
    }
    let mut used: HashSet<String> = HashSet::new();
    let mut keys = Vec::with_capacity(program.functions.len());
    for f in &program.functions {
        let key = if counts[f.name.as_str()] == 1 {
            f.name.clone()
        } else {
            match f.def_id {
                Some(def) => format!("{}#{def}", f.name),
                None => {
                    return Err(ConvertError::DuplicateFunction {
                        name: f.name.clone(),
                    });
                }
            }
        };
        if !used.insert(key.clone()) {
            return Err(ConvertError::DuplicateFunction {
                name: f.name.clone(),
            });
        }
        keys.push(key);
    }
    Ok(keys)
}

/// [`convert_program`] plus the per-function conversion failures,
/// for the run entry points: the entry function's own failure is a
/// whole-run conversion failure, exactly as before per-body
/// isolation, so callers need the failures beside the program.
fn convert_all(
    program: &MirProgram,
) -> Result<(ir::Program, HashMap<String, ConvertError>), ConvertError> {
    let keys = program_keys(program)?;
    // First phase: convert every body with trait-impl call
    // resolution off (the historical behaviour), learning which
    // bodies convert at all.
    let conv = Converter::new(program, &keys);
    let (mut out, mut failures) = convert_each(&conv, program, &keys);
    // Second phase: convert again with trait-impl call resolution
    // on, restricted to the first phase's *clean* bodies (see
    // `clean_targets`): bodies that converted and cannot, directly
    // or transitively, run into an unmodelled construct or an
    // unconverted body. Resolution only retargets calls, so the
    // converted set is unchanged in principle; a body whose
    // second-phase conversion unexpectedly fails keeps its
    // first-phase form, and the failure records are reconciled with
    // the merged program at the end.
    let converted = clean_targets(&out);
    if !converted.is_empty() {
        let conv = Converter::with_impl_targets(program, &keys, converted);
        let (resolved, _) = convert_each(&conv, program, &keys);
        for (key, body) in resolved.functions {
            out.unconverted.remove(&key);
            out.functions.insert(key, body);
        }
    }
    failures.retain(|key, _| !out.functions.contains_key(key));
    out.adt_layouts = program.adt_layouts.clone();
    Ok((out, failures))
}

/// The bodies of a first-phase program that trait-impl call
/// resolution may target in the second phase (see `convert_all`):
/// the *clean* bodies — running one can never hit an unmodelled
/// construct or a body that is not itself clean.
///
/// A body is unclean when
/// * it contains an unsupported statement or terminator (an
///   unsupported rvalue rides inside an assignment statement),
/// * a direct `Call` edge names a body that did not convert, or a
///   body that is itself unclean, or
/// * its drop glue (on a `Drop` terminator or a `mem::drop`
///   builtin call) calls an unclean destructor, or contains glue
///   conversion could not account for (`DropPlan::Unknown` /
///   `DropPlan::Untyped`: executing it judges or fails on the
///   value's own shape, which is exactly what resolution must not
///   newly expose a caller to).
///
/// `CallBuiltin` edges are neutral (builtins are the fallback
/// model itself), and so are `CallTrait` / `CallIndirect` edges
/// and `Call`s whose target name is no program body at all: a
/// vtable-dispatch call names the trait method, and the concrete
/// target is chosen from the receiver's value at run time.
/// Computed as a fixpoint: uncleanliness propagates backwards
/// along call and destructor edges until nothing changes.
fn clean_targets(program: &ir::Program) -> HashSet<String> {
    // Direct edges from each body to the program bodies it can
    // enter, plus the bodies that are unclean on their own.
    let mut edges: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut unclean: HashSet<&str> = HashSet::new();
    for (key, body) in &program.functions {
        let mut targets: Vec<&str> = Vec::new();
        for block in &body.blocks {
            for stmt in &block.statements {
                match &stmt.kind {
                    ir::StatementKind::Unsupported(_)
                    | ir::StatementKind::Assign(_, ir::Rvalue::Unsupported(_)) => {
                        unclean.insert(key.as_str());
                    }
                    _ => {}
                }
            }
            match &block.terminator {
                ir::TerminatorKind::Unsupported(_) => {
                    unclean.insert(key.as_str());
                }
                ir::TerminatorKind::Call { func, .. } => {
                    // An edge only when the name is a program body
                    // (converted, or recorded as unconverted); a
                    // name in neither table is a vtable-dispatch
                    // call, whose target the receiver's value
                    // chooses at run time.
                    if program.functions.contains_key(func)
                        || program.unconverted.contains_key(func)
                    {
                        targets.push(func.as_str());
                    }
                }
                ir::TerminatorKind::Drop { glue, .. } => {
                    if !drop_glue_edges(glue, &mut targets) {
                        unclean.insert(key.as_str());
                    }
                }
                ir::TerminatorKind::CallBuiltin {
                    drop_glue: Some(glue),
                    ..
                } => {
                    if !drop_glue_edges(glue, &mut targets) {
                        unclean.insert(key.as_str());
                    }
                }
                _ => {}
            }
        }
        edges.insert(key.as_str(), targets);
    }
    // Propagate: a body with an edge to an unconverted or unclean
    // body is itself unclean.
    loop {
        let mut changed = false;
        for (key, targets) in &edges {
            if unclean.contains(key) {
                continue;
            }
            if targets
                .iter()
                .any(|t| !program.functions.contains_key(*t) || unclean.contains(t))
            {
                unclean.insert(key);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    program
        .functions
        .keys()
        .filter(|k| !unclean.contains(k.as_str()))
        .cloned()
        .collect()
}

/// Walk one terminator's drop glue, pushing the destructor bodies
/// it can call onto `targets`. Returns `false` when the glue
/// contains a plan conversion could not account for
/// (`DropPlan::Unknown` / `DropPlan::Untyped`).
fn drop_glue_edges<'a>(glue: &'a ir::DropGlue, targets: &mut Vec<&'a str>) -> bool {
    fn walk<'a>(
        plan: &'a ir::DropPlan,
        adts: &'a [ir::DropPlan],
        targets: &mut Vec<&'a str>,
        seen: &mut HashSet<usize>,
    ) -> bool {
        match plan {
            ir::DropPlan::Trivial => true,
            // Releasing a RefCell borrow calls no destructor body.
            ir::DropPlan::RefCellRelease { .. } => true,
            ir::DropPlan::Unknown | ir::DropPlan::Untyped => false,
            ir::DropPlan::Seq(plans) => plans.iter().all(|p| walk(p, adts, targets, seen)),
            ir::DropPlan::At { plan, .. } => walk(plan, adts, targets, seen),
            ir::DropPlan::CallDtor { func } => {
                targets.push(func.as_str());
                true
            }
            ir::DropPlan::Each { elem } => walk(elem, adts, targets, seen),
            ir::DropPlan::Fields { fields, .. } => {
                fields.iter().all(|p| walk(p, adts, targets, seen))
            }
            ir::DropPlan::Variants { variants } => {
                variants.iter().all(|p| walk(p, adts, targets, seen))
            }
            // An ADT plan can refer to itself (recursive types);
            // `seen` keeps the walk finite — a revisited plan's
            // edges were collected by the first visit.
            ir::DropPlan::Adt { index } => {
                if !seen.insert(*index) {
                    return true;
                }
                match adts.get(*index) {
                    Some(plan) => walk(plan, adts, targets, seen),
                    None => false,
                }
            }
        }
    }
    let mut seen = HashSet::new();
    if !walk(&glue.root, &glue.adts, targets, &mut seen) {
        return false;
    }
    // Table entries the root never referenced are walked too, so
    // their destructor edges and unaccounted plans count as well.
    for (index, plan) in glue.adts.iter().enumerate() {
        if seen.insert(index) && !walk(plan, &glue.adts, targets, &mut seen) {
            return false;
        }
    }
    true
}

/// Convert every body of `program` with `conv`, isolating failures
/// per body (see [`convert_program`]).
fn convert_each(
    conv: &Converter,
    program: &MirProgram,
    keys: &[String],
) -> (ir::Program, HashMap<String, ConvertError>) {
    let mut out = ir::Program::new();
    let mut failures: HashMap<String, ConvertError> = HashMap::new();
    for (func, key) in program.functions.iter().zip(keys) {
        match conv.convert_body(func, key) {
            Ok(body) => {
                out.add_function(body);
            }
            Err(e) => {
                out.add_unconverted(key.clone(), e.to_string());
                failures.insert(key.clone(), e);
            }
        }
    }
    (out, failures)
}

/// Convert `program` and run `entry` with `args` in the interpreter.
///
/// Bodies are converted in isolation (see [`convert_program`]):
/// an unconvertible helper that is never called does not affect
/// the run, and calling one fails with its conversion reason. If
/// the entry function itself cannot be converted, the run fails
/// with that conversion error before anything executes.
pub fn interpret_mir(
    program: &MirProgram,
    entry: &str,
    args: Vec<Value>,
) -> Result<Value, MirRunError> {
    let (ir_program, failures) = convert_all(program)?;
    if let Some(e) = failures.get(entry) {
        return Err(MirRunError::Convert(e.clone()));
    }
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
    match convert_all(program) {
        Ok((ir_program, failures)) => {
            // The entry function's own conversion failure is a
            // whole-run failure with no guest output, as before.
            if let Some(e) = failures.get(entry) {
                return CapturedRun {
                    result: Err(MirRunError::from(e.clone())),
                    stdout: String::new(),
                };
            }
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
                // Coroutine places project the state object's fields
                // the same way ADT places do; the table is keyed by
                // the coroutine definition and its state (variant).
                let owner = match &ty {
                    mir::Ty::Adt(def, args) => Some((*def, args.clone())),
                    mir::Ty::Coroutine(def, args) => Some((*def, args.clone())),
                    _ => None,
                };
                if let Some((def, args)) = owner {
                    table.record(
                        AdtFieldKey {
                            def,
                            generic_args: args,
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

/// Operand counts of ADT aggregate constructions, per
/// (def, generic args, variant). An aggregate rvalue lists every
/// field of the variant it builds, so its operand count proves how
/// many fields that variant holds; drop-glue conversion uses it to
/// check that the field types recovered from projections cover the
/// whole variant.
#[derive(Default)]
struct AdtAggregateCounts {
    entries: Vec<(mir::DefId, Vec<mir::Ty>, u32, usize)>,
}

impl AdtAggregateCounts {
    fn record(
        &mut self,
        def: mir::DefId,
        generic_args: Vec<mir::Ty>,
        variant: u32,
        count: usize,
    ) {
        // Construction sites of one variant must agree on its field
        // count; the first sighting wins, as in `AdtFieldTys`.
        if self
            .entries
            .iter()
            .any(|(d, a, v, _)| *d == def && *a == generic_args && *v == variant)
        {
            return;
        }
        self.entries.push((def, generic_args, variant, count));
    }

    fn get(&self, def: mir::DefId, generic_args: &[mir::Ty], variant: u32) -> Option<usize> {
        self.entries
            .iter()
            .find(|(d, a, v, _)| *d == def && a.as_slice() == generic_args && *v == variant)
            .map(|e| e.3)
    }
}

/// The declared type of a place, walking its projections the way
/// `scan_place_fields` does. Used to type aggregate operands: an
/// operand filled into field `i` of an ADT aggregate carries that
/// field's declared type, so a construction site records field
/// types that no `Field` projection anywhere ever reveals (a field
/// that is only ever written, never read).
fn operand_place_ty(body: &mir::Body, place: &mir::Place) -> Option<mir::Ty> {
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
            mir::PlaceElem::Downcast(_, _) | mir::PlaceElem::Subslice { .. } => ty,
            mir::PlaceElem::OpaqueCast(t) | mir::PlaceElem::Subtype(t) => t.clone(),
        };
    }
    Some(ty)
}

fn scan_body_aggregates(
    counts: &mut AdtAggregateCounts,
    fields: &mut AdtFieldTys,
    body: &mir::Body,
) {
    for bb in &body.basic_blocks {
        for stmt in &bb.statements {
            if let mir::StatementKind::Assign(pair) = &stmt.kind {
                let (_, rv) = &**pair;
                if let mir::Rvalue::Aggregate(kind, ops) = rv {
                    // A coroutine creation aggregate is recorded as
                    // its unresumed state (variant 0), like an ADT
                    // aggregate of that variant.
                    let aggregate_key = match &**kind {
                        mir::AggregateKind::Adt {
                            def,
                            variant_index,
                            generic_args,
                            ..
                        } => Some((*def, generic_args.clone(), *variant_index)),
                        mir::AggregateKind::Coroutine { def, generic_args } => {
                            Some((*def, generic_args.clone(), 0))
                        }
                        _ => None,
                    };
                    if let Some((def, generic_args, variant_index)) = aggregate_key {
                        counts.record(def, generic_args.clone(), variant_index, ops.len());
                        for (i, op) in ops.iter().enumerate() {
                            if let Some(place) = op.place() {
                                if let Some(ty) = operand_place_ty(body, place) {
                                    fields.record(
                                        AdtFieldKey {
                                            def,
                                            generic_args: generic_args.clone(),
                                            variant: variant_index,
                                            field: i as u32,
                                        },
                                        ty,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

struct Converter<'a> {
    /// Program key -> body, for callee argument typing.
    bodies: HashMap<String, &'a mir::Body>,
    /// `DefId` -> program key, for resolving `FnDef` call operands.
    def_names: HashMap<mir::DefId, String>,
    /// `DefId` -> name for every named item (the driver's table,
    /// including body-less callees and ADT types).
    names: &'a HashMap<mir::DefId, String>,
    /// Declared ADT field types recovered from `Field` projections
    /// across the whole program (see [`AdtFieldTys`]).
    adt_fields: AdtFieldTys,
    /// Discriminant tables of the program (see `MirProgram`).
    discriminants: &'a mir::AdtDiscriminants,
    /// Operand counts of ADT aggregate constructions (see
    /// [`AdtAggregateCounts`]).
    adt_aggregates: AdtAggregateCounts,
    /// First path segment of the entry function (`{crate}::main`):
    /// the crate whose items the driver's name table enumerates
    /// completely, `Drop` impls included. `None` when the program
    /// has no `main` function.
    local_crate: Option<String>,
    /// The bodies trait-impl call resolution may target (see
    /// `resolve_impl_callee`): `None` disables resolution (the
    /// first conversion phase in `convert_all`), `Some` restricts
    /// it to the named bodies (the second phase).
    impl_targets: Option<HashSet<String>>,
}

impl<'a> Converter<'a> {
    fn new(program: &'a MirProgram, keys: &[String]) -> Self {
        let mut bodies = HashMap::new();
        let mut def_names = HashMap::new();
        let mut adt_fields = AdtFieldTys::default();
        let mut adt_aggregates = AdtAggregateCounts::default();
        for (f, key) in program.functions.iter().zip(keys) {
            bodies.insert(key.clone(), &f.body);
            if let Some(def) = f.def_id {
                def_names.insert(def, key.clone());
            }
            scan_body_fields(&mut adt_fields, &f.body);
            scan_body_aggregates(&mut adt_aggregates, &mut adt_fields, &f.body);
        }
        let local_crate = program
            .functions
            .iter()
            .find(|f| f.name == "main" || f.name.ends_with("::main"))
            .map(|f| f.name.split("::").next().unwrap_or("").to_string());
        Converter {
            bodies,
            def_names,
            names: &program.fn_names,
            adt_fields,
            discriminants: &program.adt_discriminants,
            adt_aggregates,
            local_crate,
            impl_targets: None,
        }
    }

    /// A converter with trait-impl call resolution enabled,
    /// restricted to `targets` (see `resolve_impl_callee`).
    fn with_impl_targets(
        program: &'a MirProgram,
        keys: &[String],
        targets: HashSet<String>,
    ) -> Self {
        let mut conv = Self::new(program, keys);
        conv.impl_targets = Some(targets);
        conv
    }

    fn convert_body(&self, func: &MirFunction, key: &str) -> Result<ir::Body, ConvertError> {
        let body = &func.body;
        let mut local_tys = Vec::with_capacity(body.local_decls.len());
        for (i, decl) in body.local_decls.iter().enumerate() {
            local_tys.push(self.convert_ty(&decl.ty, &format!("local _{i} of `{key}`"))?);
        }
        let mut blocks = Vec::with_capacity(body.basic_blocks.len());
        for bb in &body.basic_blocks {
            blocks.push(self.convert_block(body, bb)?);
        }
        Ok(ir::Body::new(
            key.to_string(),
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

    /// The program key of the body a `DefId` names, for references
    /// that must land on one specific converted body (closure
    /// values and closure types): the converted-body table first —
    /// it is keyed by the exact (definition, instance) id the
    /// driver assigned, so two instances of one closure definition
    /// resolve to their own bodies — and the display name only as
    /// a fallback for definitions with no body in the program.
    fn body_key(&self, def: &mir::DefId) -> String {
        self.def_names
            .get(def)
            .cloned()
            .unwrap_or_else(|| self.def_display_name(def))
    }

    /// The name of a concrete type as it appears in the `Self`
    /// position of its impl method names in the program
    /// (`<{name} as Trait>::method`): the definition's display name
    /// for a named type, the surface-syntax rendering otherwise.
    fn concrete_ty_name(&self, ty: &mir::Ty) -> String {
        match ty {
            mir::Ty::Adt(def, _)
            | mir::Ty::Closure(def, _)
            | mir::Ty::FnDef(def, _)
            | mir::Ty::Coroutine(def, _) => self.def_display_name(def),
            other => crate::builtins::ty_tag(other, self.names),
        }
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
                // The closure type's generic arguments are
                // `[parent args.., kind, signature, upvars tuple]`:
                // the upvars tuple is the LAST argument, not the
                // third — the third is the upvars tuple only for a
                // closure with no parent generics, and reading it
                // there misreads every closure defined inside a
                // generic function or trait method.
                let upvars = match args.last() {
                    Some(mir::Ty::Tuple(ts)) => ts.len(),
                    _ => 0,
                };
                ir::Ty::Closure(self.body_key(def), upvars)
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
            // `dyn Trait` is representable: behind a reference it is
            // a fat pointer whose value carries the concrete type;
            // as a bare place type it names the unsized pointee a
            // fat pointer's dereference reaches. Only the bounds'
            // names are kept.
            mir::Ty::Dynamic(traits, _, _) => ir::Ty::Dynamic(traits.clone()),
            // A coroutine type: the value is the state object the
            // coroutine transform produces (see `ir::Ty::Coroutine`).
            // Only the body's name is kept; the state layout is
            // carried by the value, and the generic arguments (upvars,
            // resume / yield / return types, witness) are not needed
            // to execute the transformed body.
            mir::Ty::Coroutine(def, _) => ir::Ty::Coroutine(self.def_display_name(def)),
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
            // A closure in constant position is a value too (the
            // zero-sized closure constant a `ClosureFnPointer` cast
            // coerces, or a stored closure value): with an empty
            // captured environment it becomes the closure value
            // naming the body. A constant whose environment is not
            // empty cannot be materialised here — the captured
            // values are not part of the type constant — and stays
            // an explicit unsupported naming the closure.
            mir::ConstKind::Ty(mir::Ty::Closure(def, args)) => {
                // The upvars tuple is the closure type's LAST
                // generic argument (see `convert_ty`); the third
                // argument is the upvars tuple only when the
                // closure has no parent generics, so a closure
                // constant inside a generic context (e.g. the
                // closure inside `Sum::sum`) was misread as
                // capturing an environment it does not have.
                let upvars = match args.last() {
                    Some(mir::Ty::Tuple(ts)) => ts.len(),
                    _ => usize::MAX,
                };
                if upvars == 0 {
                    Ok(ir::Constant::Closure(self.body_key(def)))
                } else {
                    Err(ConvertError::UnsupportedConstant {
                        kind: format!(
                            "closure constant {} with a captured environment",
                            self.def_display_name(def)
                        ),
                    })
                }
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
                    .enumerate()
                    .map(|(i, f)| {
                        // A promoted array constant is represented
                        // as a tuple of its elements (inside a
                        // by-reference constant); under an `Array`
                        // context every field takes the element type.
                        // Without it the fields decode context-free,
                        // which misreads the sign of 128-bit elements.
                        let elem_ctx = match ctx {
                            Some(mir::Ty::Tuple(ts)) => ts.get(i),
                            Some(mir::Ty::Array(elem, _)) => Some(&**elem),
                            _ => None,
                        };
                        self.convert_const_value(f, elem_ctx)
                    })
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
                // The pointee's declared type types the inner value:
                // a scalar leaf carries raw bits with no signedness of
                // its own, and a pointee is read back through a deref
                // with no typed store in between to re-tag it, so
                // dropping the context here would misread an unsigned
                // leaf with its sign bit set as a negative signed
                // value (a promoted `200.0f64 as u8` printing as -56).
                let inner_ctx = match ctx {
                    Some(mir::Ty::Ref(_, pointee, _) | mir::Ty::RawPtr(pointee, _)) => {
                        Some(&**pointee)
                    }
                    _ => None,
                };
                ir::Constant::Ref(Box::new(self.convert_const_value(inner, inner_ctx)?))
            }
            mir::ConstValue::SliceVal(elems) => ir::Constant::SliceVal(
                elems
                    .iter()
                    .map(|e| {
                        // The slice value may sit directly under a
                        // reference context (a `&[u8]` constant), so
                        // unwrap references before matching the
                        // array/slice element type.
                        let mut inner_ctx = ctx;
                        while let Some(
                            mir::Ty::Ref(_, pointee, _) | mir::Ty::RawPtr(pointee, _),
                        ) = inner_ctx
                        {
                            inner_ctx = Some(pointee);
                        }
                        let elem_ctx = match inner_ctx {
                            Some(mir::Ty::Array(elem, _) | mir::Ty::Slice(elem)) => {
                                Some(&**elem)
                            }
                            _ => None,
                        };
                        self.convert_const_value(e, elem_ctx)
                    })
                    .collect::<Result<_, _>>()?,
            ),
            mir::ConstValue::FnDef(def) => {
                ir::Constant::FnDef(self.def_display_name(def))
            }
            mir::ConstValue::Closure(def) => {
                ir::Constant::Closure(self.body_key(def))
            }
            mir::ConstValue::IntPtr(addr) => ir::Constant::IntPtr(*addr),
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
                    // set sign bit means a negative signed value, at
                    // every width: a 16-byte pattern with its top bit
                    // set is a negative `i128`, not `2^127` unsigned
                    // (the rule `literal_pair_ty` already applies to
                    // 128-bit literal pairs).
                    _ => {
                        let bits = (*size as u32) * 8;
                        if bits > 0 && (data >> (bits - 1)) & 1 == 1 {
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
    /// (shifts aside — their operands are typed separately by
    /// `convert_shift_operands`), and a literal
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

    /// Convert the two operands of a shift (`Shl` / `Shr`, plain,
    /// unchecked or checked).
    ///
    /// A shift is the one binary operation whose operands do not
    /// share a type in MIR: the amount may have any integer type.
    /// The shared-context conversion would mistype both sides — a
    /// constant value read under the amount's type (when the amount
    /// is a place), or a constant amount read under the value's
    /// type, truncating it to the value's width and sign — so the
    /// operands are typed separately:
    ///
    /// * The value takes the operation's own context: its place
    ///   type, else the result type (a checked shift's `(T, bool)`
    ///   result gives `T`).
    /// * The amount keeps its own type. A place amount's runtime
    ///   value already carries it, and a constant amount is
    ///   converted with no context, which reads it in its own
    ///   storage width (see `convert_scalar`). That reading is
    ///   exact whenever the amount fits the value's width; an
    ///   amount that does not fit — wider than the value, negative,
    ///   or an unsigned amount whose top bit is set, read as a
    ///   negative — is an overflowing shift under either reading,
    ///   which execution reports the same way (see `eval_shift`).
    fn convert_shift_operands(
        &self,
        body: &mir::Body,
        pair: &(mir::Operand, mir::Operand),
        dest_ty: Option<&mir::Ty>,
        checked_result: bool,
    ) -> Result<(ir::Operand, ir::Operand), ConvertError> {
        let value_ctx = self.operand_place_ty(body, &pair.0).or_else(|| {
            if checked_result {
                match dest_ty {
                    Some(mir::Ty::Tuple(ts)) => ts.first().cloned(),
                    _ => None,
                }
            } else {
                dest_ty.cloned()
            }
        });
        let lhs = self.convert_operand(body, &pair.0, value_ctx.as_ref())?;
        let rhs = self.convert_operand(body, &pair.1, None)?;
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
                let (lhs, rhs) = if is_shift(*op) {
                    self.convert_shift_operands(body, pair, dest_ty, false)?
                } else {
                    self.convert_binop_operands(
                        body,
                        pair,
                        dest_ty,
                        overflow_base(*op).is_some(),
                        is_comparison(*op),
                    )?
                };
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
                let (lhs, rhs) = if is_shift(*op) {
                    self.convert_shift_operands(body, pair, dest_ty, true)?
                } else {
                    self.convert_binop_operands(
                        body,
                        pair,
                        dest_ty,
                        true,
                        is_comparison(*op),
                    )?
                };
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
                        match dyn_traits_in(target) {
                            // Sized-to-`dyn`: the vtable identity is
                            // decided here, where both types are
                            // known — the concrete type from the
                            // operand's type, the bounds from the
                            // target.
                            Some(traits) => {
                                let source = self.operand_place_ty(body, operand);
                                match source {
                                    // A `dyn`-to-`dyn` coercion
                                    // (supertrait upcast): the
                                    // concrete type rides on the
                                    // value; only the bounds change.
                                    Some(src) if dyn_traits_in(&src).is_some() => {
                                        ir::Rvalue::Unsize(
                                            v,
                                            ir::UnsizeMeta::Dyn {
                                                concrete: None,
                                                traits,
                                            },
                                        )
                                    }
                                    Some(src) => {
                                        let pointee = unsize_pointee(&src);
                                        ir::Rvalue::Unsize(
                                            v,
                                            ir::UnsizeMeta::Dyn {
                                                concrete: Some(
                                                    self.concrete_ty_name(pointee),
                                                ),
                                                traits,
                                            },
                                        )
                                    }
                                    // A constant operand does not
                                    // preserve its type in this MIR
                                    // form, so the concrete type
                                    // cannot be named (the same
                                    // limit as the ClosureFnPointer
                                    // constant case below).
                                    None => ir::Rvalue::Unsupported(format!(
                                        "Cast Unsize to {} (the concrete type is not                                          preserved in the constant operand)",
                                        target.kind_name()
                                    )),
                                }
                            }
                            None => ir::Rvalue::Unsize(v, ir::UnsizeMeta::Slice),
                        }
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
                    // pointer: the closure value itself is the
                    // callable, for place operands and for constant
                    // operands alike — the driver preserves the
                    // closure type of zero-sized closure constants,
                    // and constant conversion resolves it to the
                    // closure value (a captured environment stays an
                    // explicit unsupported there).
                    mir::CastKind::PointerCoercion(
                        mir::PointerCoercion::ClosureFnPointer,
                    ) => ir::Rvalue::Use(v),
                    mir::CastKind::BoxDerefTransmute => ir::Rvalue::BoxDerefTransmute(v),
                    // `ptr as usize`: exposing a pointer's
                    // provenance yields its address as an integer,
                    // evaluated to the pointer's synthetic address
                    // (`Rvalue::PtrToInt`) — the same treatment as
                    // the pointer-to-integer transmute below. This
                    // is the cast rustc emits for `ptr as usize` in
                    // user code (e.g. alignment checks on raw
                    // pointers). The reverse direction (an integer
                    // cast back to a pointer,
                    // PointerWithExposedProvenance) stays an
                    // explicit unsupported: an address names no
                    // place in this model. (Lane BV)
                    mir::CastKind::PointerExposeProvenance => match (&ctx, target) {
                        (
                            Some(mir::Ty::RawPtr(..) | mir::Ty::Ref(..)),
                            mir::Ty::Int(_) | mir::Ty::Uint(_),
                        ) => ir::Rvalue::PtrToInt(v.clone()),
                        _ => ir::Rvalue::Unsupported(
                            "Cast PointerExposeProvenance".to_string(),
                        ),
                    },
                    // A pointer-to-integer `transmute` (what the
                    // compiler's UB precondition checks use to test a
                    // pointer for null / alignment) becomes the
                    // model's synthetic address. Every other
                    // transmute shape stays unsupported: the model
                    // cannot reinterpret a value's bytes, and
                    // inventing an integer-to-pointer target would
                    // fabricate a reference.
                    mir::CastKind::Transmute => match (&ctx, target) {
                        (
                            Some(mir::Ty::RawPtr(..) | mir::Ty::Ref(..)),
                            mir::Ty::Int(_) | mir::Ty::Uint(_),
                        ) => ir::Rvalue::PtrToInt(v),
                        _ => ir::Rvalue::Unsupported("Cast Transmute".to_string()),
                    },
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
                    // The closure type's LAST generic argument is
                    // the upvars tuple; its element types type the
                    // capture operands.
                    let upvar_tys: Vec<mir::Ty> = match generic_args.last() {
                        Some(mir::Ty::Tuple(ts)) => ts.clone(),
                        _ => Vec::new(),
                    };
                    let mut fields = Vec::with_capacity(ops.len());
                    for (i, op) in ops.iter().enumerate() {
                        fields.push(self.convert_operand(body, op, upvar_tys.get(i))?);
                    }
                    ir::Rvalue::ClosureAggregate {
                        def: self.body_key(def),
                        operands: fields,
                    }
                }
                mir::AggregateKind::Coroutine { def, generic_args } => {
                    // The creation aggregate fills the unresumed
                    // state's fields: the upvars. Their declared
                    // types come from the field table (the coroutine
                    // body's own projections of state 0), with the
                    // upvars tuple - the coroutine type's first
                    // generic argument - as the fallback.
                    let upvar_tys: Vec<mir::Ty> = match generic_args.first() {
                        Some(mir::Ty::Tuple(ts)) => ts.clone(),
                        _ => Vec::new(),
                    };
                    let mut fields = Vec::with_capacity(ops.len());
                    for (i, op) in ops.iter().enumerate() {
                        let declared =
                            self.adt_fields.get(*def, generic_args, 0, i as u32);
                        let ctx = declared.or_else(|| upvar_tys.get(i));
                        fields.push(self.convert_operand(body, op, ctx)?);
                    }
                    ir::Rvalue::CoroutineAggregate(fields)
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

    /// The argument types of the body a constant callable operand
    /// names (a decoded function-item or closure constant): the
    /// callee's own declared argument locals, skipping a closure
    /// body's environment local. Empty when the operand is not
    /// such a constant or the body is not converted.
    fn constant_callee_param_tys(&self, func: &mir::Operand) -> Vec<mir::Ty> {
        let mir::Operand::Constant(c) = func else {
            return Vec::new();
        };
        let (def, env_locals) = match &c.literal {
            mir::ConstKind::Value(mir::ConstValue::Closure(def)) => (def, 1),
            mir::ConstKind::Value(mir::ConstValue::FnDef(def)) => (def, 0),
            _ => return Vec::new(),
        };
        let Some(key) = self.def_names.get(def) else {
            return Vec::new();
        };
        let Some(callee) = self.bodies.get(key) else {
            return Vec::new();
        };
        (1 + env_locals..=callee.arg_count)
            .filter_map(|i| callee.local_decls.get(i).map(|d| d.ty.clone()))
            .collect()
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

    /// Resolve a trait-method call whose func operand names the
    /// trait item itself (which has no body) to the concrete impl
    /// body the driver collected for the call's `Self` type, if the
    /// program has one.
    ///
    /// MIR names such a call by the trait item (`std::ops::Index::
    /// index` with generic arguments `[Self, Idx]`), while the
    /// collected body is keyed by its instance name
    /// (`<S as std::ops::Index<usize>>::index`). Without this step
    /// the call falls through to the name-based builtin model of
    /// the trait method, which is only faithful for receivers whose
    /// impl bodies are not in the program (core's slice/array impls,
    /// `Vec`): for a user-defined impl it silently computes the
    /// builtin's semantics on the receiver instead — e.g. an
    /// `Index` impl returning `&str` was executed as a slice index
    /// into the receiver struct and panicked with a bogus
    /// out-of-bounds message. A collected body, when one matches,
    /// is the faithful semantics and wins over the builtin, exactly
    /// as a `DefId`-resolved body does in `resolve_callee`.
    fn resolve_impl_callee(&self, func: &mir::Operand) -> Option<String> {
        // Resolution runs only in the second conversion phase,
        // against the first phase's clean bodies (see
        // `clean_targets`): resolving to an impl body that cannot
        // itself convert — or that calls, however indirectly, a
        // body that cannot — would trade a working builtin model
        // of the trait method for an unconverted-body error. A
        // derived `PartialEq` on a packed struct is the shape
        // that pins this restriction: every eq body converts, but
        // the manual impl calls a helper whose pointer cast the
        // conversion leaves unsupported, while the builtin's
        // structural comparison is exact.
        let targets = self.impl_targets.as_ref()?;
        let key = self.find_impl_callee(func)?;
        targets.contains(&key).then_some(key)
    }

    /// Find the collected impl body for a trait-method call (see
    /// `resolve_impl_callee` for when a found body may be used).
    fn find_impl_callee(&self, func: &mir::Operand) -> Option<String> {
        let (name, _, raw_generic_args) = self.resolve_external_callee(func)?;
        let (trait_path, method) = name.rsplit_once("::")?;
        if trait_path.is_empty() || method.is_empty() {
            return None;
        }
        let self_ty = raw_generic_args.first()?;
        let self_name = self.concrete_ty_name(self_ty);
        // The impl's instance name spells the trait path as the
        // impl wrote it: the call may name the trait through `std::`
        // while the impl was recorded through `core::`, or the
        // other way round, so try both spellings.
        let mut paths = vec![trait_path.to_string()];
        if let Some(rest) = trait_path.strip_prefix("std::") {
            paths.push(format!("core::{rest}"));
        } else if let Some(rest) = trait_path.strip_prefix("core::") {
            paths.push(format!("std::{rest}"));
        }
        // The trait's own generic arguments (the func operand's
        // arguments after `Self`), rendered as they appear in the
        // impl name: `<S as std::ops::Index<usize>>::index`.
        let trait_args: Vec<String> = raw_generic_args[1..]
            .iter()
            .map(|t| crate::builtins::ty_tag(t, self.names))
            .collect();
        for path in &paths {
            if !trait_args.is_empty() {
                let candidate = format!(
                    "<{self_name} as {path}<{}>>::{method}",
                    trait_args.join(", ")
                );
                if let Some(key) = self.find_body_key(&candidate) {
                    return Some(key);
                }
            }
            let candidate = format!("<{self_name} as {path}>::{method}");
            if let Some(key) = self.find_body_key(&candidate) {
                return Some(key);
            }
        }
        // Fall back to a scan: exactly one impl body for this
        // `Self` type and trait, under any generic-argument
        // spelling (a defaulted trait argument, for one, is
        // elided from the instance name but present in the call's
        // generic arguments). An ambiguous scan resolves nothing.
        for path in &paths {
            if let Some(key) = self.scan_impl_key(&self_name, path, method) {
                return Some(key);
            }
        }
        None
    }

    /// The program key of a body stored under exactly `candidate`,
    /// or under its `DefId`-disambiguated form (`candidate#DefId`,
    /// see `program_keys`) when that form is unique.
    fn find_body_key(&self, candidate: &str) -> Option<String> {
        if self.bodies.contains_key(candidate) {
            return Some(candidate.to_string());
        }
        let mut matches = self.bodies.keys().filter(|k| {
            k.strip_prefix(candidate)
                .is_some_and(|rest| rest.starts_with('#'))
        });
        match (matches.next(), matches.next()) {
            (Some(key), None) => Some(key.clone()),
            _ => None,
        }
    }

    /// The unique body keyed as an impl of `trait_path` for
    /// `self_name` with method `method`, under any generic-argument
    /// spelling of the trait; `None` when there is not exactly one.
    fn scan_impl_key(&self, self_name: &str, trait_path: &str, method: &str) -> Option<String> {
        let prefix = format!("<{self_name} as {trait_path}");
        let suffix = format!(">::{method}");
        let mut found: Option<&String> = None;
        for key in self.bodies.keys() {
            let base = key.split('#').next().unwrap_or(key.as_str());
            if base.starts_with(&prefix) && base.ends_with(&suffix) {
                if found.is_some() {
                    return None;
                }
                found = Some(key);
            }
        }
        found.cloned()
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

    /// Build the drop glue of the dropped type, if every part of
    /// it can be accounted for (see [`DropGlueBuilder`]).
    fn drop_glue(&self, ty: &mir::Ty) -> Option<ir::DropGlue> {
        DropGlueBuilder::build(self, ty)
    }

    /// The drop glue of a `std::mem::drop` call's argument, mirroring
    /// the `Drop` terminator's treatment of the same type:
    /// `Ok(Some(glue))` when the argument is a place whose type's
    /// glue conversion can account for (the interpreter runs it
    /// against the argument place), `Ok(None)` when there is nothing
    /// to run (a constant argument whose glue is trivial, so
    /// discarding it is already faithful), and `Err(())` when the
    /// glue cannot be accounted for.
    fn mem_drop_glue(
        &self,
        body: &mir::Body,
        arg: &mir::Operand,
        ir_arg: &ir::Operand,
        raw_generic_args: &[mir::Ty],
    ) -> Result<Option<ir::DropGlue>, ()> {
        let dropped_ty = self
            .operand_place_ty(body, arg)
            .or_else(|| raw_generic_args.first().cloned());
        let glue = dropped_ty.and_then(|t| self.drop_glue(&t)).ok_or(())?;
        match ir_arg {
            ir::Operand::Copy(_) | ir::Operand::Move(_) => Ok(Some(glue)),
            _ if glue.root == ir::DropPlan::Trivial => Ok(None),
            _ => Err(()),
        }
    }

    /// The converted body of `adt_name`'s `Drop` impl, if the
    /// program has one. The driver's name for the impl is
    /// `<{adt_name} as std::ops::Drop>::drop` (the `core::` spelling
    /// of the trait path is accepted as well); a generic impl made
    /// concrete for one type (`impl Drop for G<i32>`) is found by
    /// the same pattern when it is the only candidate.
    fn dtor_body(&self, adt_name: &str, args: &[mir::Ty]) -> Option<String> {
        // The exact instance key, with the generic arguments
        // rendered as the driver records instance names: several
        // monomorphizations of one generic impl can coexist in the
        // program, and only the full key tells them apart.
        if !args.is_empty() {
            let parts: Vec<String> = args
                .iter()
                .map(|t| crate::builtins::ty_tag(t, self.names))
                .collect();
            let instantiated = format!("{adt_name}<{}>", parts.join(", "));
            for trait_path in ["std::ops::Drop", "core::ops::Drop"] {
                let key = format!("<{instantiated} as {trait_path}>::drop");
                if self.bodies.contains_key(&key) {
                    return Some(key);
                }
            }
        }
        for trait_path in ["std::ops::Drop", "core::ops::Drop"] {
            let key = format!("<{adt_name} as {trait_path}>::drop");
            if self.bodies.contains_key(&key) {
                return Some(key);
            }
        }
        let mut candidates = self
            .bodies
            .keys()
            .filter(|k| is_dtor_name(k, adt_name));
        match (candidates.next(), candidates.next()) {
            (Some(only), None) => Some(only.clone()),
            _ => None,
        }
    }

    /// Whether the driver's name table records a `Drop` impl for
    /// `adt_name` at all, under any generic spelling.
    fn dtor_impl_known(&self, adt_name: &str) -> bool {
        self.names.values().any(|n| is_dtor_name(n, adt_name))
    }

    /// Whether `name` (a type's driver name) belongs to the entry
    /// crate. For the entry crate's types the name table is
    /// complete, so a `Drop` impl that is absent from it does not
    /// exist; for any other crate's type an impl could exist
    /// unrecorded, and conversion must not assume there is none.
    fn is_local_type(&self, name: &str) -> bool {
        match &self.local_crate {
            Some(k) => name.split("::").next() == Some(k.as_str()),
            None => false,
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
                // A constant operand holding a decoded callable value
                // (a function pointer constant naming a function item
                // or a coerced closure) is the same shape: the call
                // goes through the value, so a closure target gets
                // its environment argument like any other indirect
                // call.
                let indirect = matches!(
                    func,
                    mir::Operand::Copy(_) | mir::Operand::Move(_)
                ) || matches!(
                    func,
                    mir::Operand::Constant(c) if matches!(
                        c.literal,
                        mir::ConstKind::Value(
                            mir::ConstValue::FnDef(_) | mir::ConstValue::Closure(_)
                        )
                    )
                );
                if indirect {
                    let func_ty = self.operand_place_ty(body, func);
                    // A constant callable carries no type of its own;
                    // type the arguments from the target body's
                    // signature instead, as the direct-call path does.
                    let param_tys = match func_ty.as_ref() {
                        Some(_) => callable_param_tys(func_ty.as_ref()),
                        None => self.constant_callee_param_tys(func),
                    };
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
                } else if let Some(name) = self
                    .resolve_callee(func)
                    .or_else(|| self.resolve_impl_callee(func))
                {
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
                        // An explicit `std::mem::drop` runs the
                        // dropped value's glue, exactly like a `Drop`
                        // terminator for the same type: build the
                        // glue now, while the type and the name
                        // tables are at hand (see `mem_drop_glue`).
                        // A call whose glue cannot be accounted for
                        // stays unsupported instead of silently
                        // discarding a value whose destructor has
                        // observable effects.
                        let mut unaccountable_drop = false;
                        let drop_glue = if kind == crate::builtins::Builtin::MemDrop
                            && args.len() == 1
                        {
                            match self.mem_drop_glue(
                                body,
                                &args[0],
                                &ir_args[0],
                                &raw_generic_args,
                            ) {
                                Ok(glue) => glue,
                                Err(()) => {
                                    unaccountable_drop = true;
                                    None
                                }
                            }
                        } else {
                            None
                        };
                        if unaccountable_drop {
                            ir::TerminatorKind::Unsupported(
                                "Call to std::mem::drop (drop glue cannot be accounted for)"
                                    .to_string(),
                            )
                        } else {
                            ir::TerminatorKind::CallBuiltin {
                                name: canonical,
                                generic_args,
                                args: ir_args,
                                destination: self.convert_place(destination)?,
                                target: target.map(|b| ir::BasicBlock(b.index())),
                                drop_glue,
                            }
                        }
                    } else if args.first().is_some_and(|a| {
                        self.operand_place_ty(body, a)
                            .is_some_and(|t| dyn_traits_in(&t).is_some())
                    }) {
                        // A method call on a `dyn Trait` value: the
                        // callee is the trait method, which has no
                        // body of its own. The call is a vtable
                        // dispatch, resolved at run time against the
                        // receiver's concrete type (see
                        // `virtual_call_target` in the interpreter).
                        let mut ir_args = Vec::with_capacity(args.len());
                        for arg in args.iter() {
                            ir_args.push(self.convert_operand(body, arg, None)?);
                        }
                        ir::TerminatorKind::Call {
                            func: name,
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
            T::Drop { place, target, .. } => {
                let glue = self
                    .place_ty(body, place)
                    .and_then(|t| self.drop_glue(&t));
                match glue {
                    Some(glue) => ir::TerminatorKind::Drop {
                        place: self.convert_place(place)?,
                        target: ir::BasicBlock(target.index()),
                        glue,
                    },
                    None => ir::TerminatorKind::Unsupported("Drop".to_string()),
                }
            }
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
        Builtin::BoxNew | Builtin::ManuallyDropNew | Builtin::CellNew | Builtin::RefCellNew => {
            if arg_index == 0 {
                raw_generic_args.first().cloned()
            } else {
                None
            }
        }
        Builtin::CellSet | Builtin::CellReplace | Builtin::VecPush | Builtin::RefCellReplace => {
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

/// Whether a `mir` binary operator is a shift (`Shl` / `Shr` or an
/// unchecked spelling): the one family whose operands do not share
/// a type, converted by `convert_shift_operands` instead of
/// `convert_binop_operands`.
fn is_shift(op: mir::BinOp) -> bool {
    use mir::BinOp as M;
    matches!(op, M::Shl | M::Shr | M::ShlUnchecked | M::ShrUnchecked)
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

// ---------------------------------------------------------------------------
// Drop glue construction
// ---------------------------------------------------------------------------

/// Whether `name` is the driver name of `adt_name`'s `Drop` impl:
/// `<{adt_name} as std::ops::Drop>::drop`, with any generic
/// argument list after the type name (the impl of a generic type is
/// recorded with its parameter spellings, e.g. `<probe::G<T> as
/// ...>::drop`). The `core::` spelling of the trait path is
/// accepted as well.
fn is_dtor_name(name: &str, adt_name: &str) -> bool {
    for trait_path in ["std::ops::Drop", "core::ops::Drop"] {
        let suffix = format!(" as {trait_path}>::drop");
        let Some(head) = name.strip_suffix(&suffix) else {
            continue;
        };
        if head == format!("<{adt_name}") {
            return true;
        }
        if let Some(args) = head.strip_prefix(&format!("<{adt_name}<")) {
            if args.ends_with('>') {
                return true;
            }
        }
    }
    false
}

/// Run `plan` against field 0 of the value it applies to (the
/// pointee of a `Box`, the payload of `Option::Some`).
fn at_field0(plan: ir::DropPlan) -> ir::DropPlan {
    match plan {
        ir::DropPlan::Trivial => ir::DropPlan::Trivial,
        other => ir::DropPlan::At {
            path: vec![RefProj::Field(0)],
            plan: Box::new(other),
        },
    }
}

/// A `Fields` plan, collapsed to `Trivial` when no field has glue.
fn collapse_fields(fields: Vec<ir::DropPlan>, complete: bool) -> ir::DropPlan {
    if fields.iter().all(|f| *f == ir::DropPlan::Trivial) {
        ir::DropPlan::Trivial
    } else {
        ir::DropPlan::Fields { fields, complete }
    }
}

/// A sequence of glue steps, collapsed when at most one remains.
fn collapse_seq(steps: Vec<ir::DropPlan>) -> ir::DropPlan {
    let mut steps: Vec<ir::DropPlan> = steps
        .into_iter()
        .filter(|p| *p != ir::DropPlan::Trivial)
        .collect();
    match steps.len() {
        0 => ir::DropPlan::Trivial,
        1 => steps.pop().expect("one step"),
        _ => ir::DropPlan::Seq(steps),
    }
}

/// Builds the drop glue of one dropped type. Glue is built only
/// when every part of the type is accounted for:
///
/// * scalars, references and function pointers have no glue;
/// * tuples, arrays and slices glue their elements;
/// * the modelled std types glue by their fixed shape (`Vec` its
///   elements, `Box` its pointee, `Option` its payload, `Range` its
///   bounds; `String`, `fmt` values and `ManuallyDrop` nothing);
/// * any other ADT glues its converted `Drop` impl body (called
///   first, as in rustc's drop glue) and then its fields, variant
///   by variant. An impl the name table records but the program
///   has no body for (a generic impl gets no monomorphic instance)
///   fails the build, as does an ADT from another crate whose impl
///   the table cannot rule out. A field whose type no projection
///   anywhere records (only ever written) gets an `Untyped` plan,
///   judged by its value's shape at run time.
struct DropGlueBuilder<'c, 'a> {
    conv: &'c Converter<'a>,
    adts: Vec<ir::DropPlan>,
    adt_index: HashMap<String, usize>,
}

impl<'c, 'a> DropGlueBuilder<'c, 'a> {
    fn build(conv: &'c Converter<'a>, ty: &mir::Ty) -> Option<ir::DropGlue> {
        let mut builder = DropGlueBuilder {
            conv,
            adts: Vec::new(),
            adt_index: HashMap::new(),
        };
        let root = builder.plan(ty)?;
        Some(ir::DropGlue {
            root,
            adts: builder.adts,
        })
    }

    fn plan(&mut self, ty: &mir::Ty) -> Option<ir::DropPlan> {
        use ir::DropPlan as P;
        match ty {
            mir::Ty::Bool
            | mir::Ty::Char
            | mir::Ty::Int(_)
            | mir::Ty::Uint(_)
            | mir::Ty::Float(_)
            | mir::Ty::Never
            | mir::Ty::Str => Some(P::Trivial),
            mir::Ty::Ref(..) | mir::Ty::RawPtr(..) | mir::Ty::FnDef(..) | mir::Ty::FnPtr(_) => {
                Some(P::Trivial)
            }
            mir::Ty::Tuple(ts) => {
                let mut fields = Vec::with_capacity(ts.len());
                for t in ts {
                    fields.push(self.plan(t)?);
                }
                Some(collapse_fields(fields, true))
            }
            mir::Ty::Array(elem, _) | mir::Ty::Slice(elem) => match self.plan(elem)? {
                P::Trivial => Some(P::Trivial),
                elem_plan => Some(P::Each {
                    elem: Box::new(elem_plan),
                }),
            },
            mir::Ty::Closure(_, args) => {
                // A closure's glue is its captures' glue: the last
                // generic argument is the capture tuple. Anything
                // else in that position is a shape this builder does
                // not model, and the drop stays refused.
                let mir::Ty::Tuple(captures) = args.last()? else {
                    return None;
                };
                let mut fields = Vec::with_capacity(captures.len());
                for t in captures {
                    fields.push(self.plan(t)?);
                }
                Some(collapse_fields(fields, true))
            }
            mir::Ty::Adt(def, args) => {
                let key = format!("{ty:?}");
                if let Some(&index) = self.adt_index.get(&key) {
                    return Some(P::Adt { index });
                }
                let index = self.adts.len();
                self.adt_index.insert(key, index);
                // Placeholder while this ADT's fields are planned, so
                // a recursive occurrence refers back to this entry.
                self.adts.push(P::Trivial);
                let plan = self.adt_plan(*def, args)?;
                self.adts[index] = plan;
                Some(P::Adt { index })
            }
            mir::Ty::Coroutine(def, args) => {
                let key = format!("{ty:?}");
                if let Some(&index) = self.adt_index.get(&key) {
                    return Some(P::Adt { index });
                }
                let index = self.adts.len();
                self.adt_index.insert(key, index);
                // Placeholder while this coroutine's fields are
                // planned, so a recursive occurrence (a future
                // awaiting another future of the same type) refers
                // back to this entry.
                self.adts.push(P::Trivial);
                let plan = self.coroutine_plan(*def, args)?;
                self.adts[index] = plan;
                Some(P::Adt { index })
            }
            _ => None,
        }
    }

    fn adt_plan(&mut self, def: mir::DefId, args: &[mir::Ty]) -> Option<ir::DropPlan> {
        use ir::DropPlan as P;
        let conv = self.conv;
        let name = conv
            .names
            .get(&def)
            .or_else(|| conv.def_names.get(&def))?
            .clone();
        // Modelled std types: their glue shape is fixed by the value
        // model, not by a converted body.
        match name.as_str() {
            "std::string::String" => return Some(P::Trivial),
            "std::fmt::Arguments" | "core::fmt::rt::Argument" => return Some(P::Trivial),
            // `ManuallyDrop` is defined to run no glue at all.
            "std::mem::ManuallyDrop" | "core::mem::ManuallyDrop" => return Some(P::Trivial),
            "std::vec::Vec" => {
                let elem = self.plan(args.first()?)?;
                return Some(match elem {
                    P::Trivial => P::Trivial,
                    e => P::Each { elem: Box::new(e) },
                });
            }
            "std::boxed::Box" => {
                let inner = self.plan(args.first()?)?;
                return Some(at_field0(inner));
            }
            "std::option::Option" => {
                let inner = self.plan(args.first()?)?;
                return Some(match inner {
                    P::Trivial => P::Trivial,
                    e => P::Variants {
                        variants: vec![P::Trivial, at_field0(e)],
                    },
                });
            }
            "std::cell::Ref" | "core::cell::Ref" => {
                return Some(P::RefCellRelease { mutable: false })
            }
            "std::cell::RefMut" | "core::cell::RefMut" => {
                return Some(P::RefCellRelease { mutable: true })
            }
            "std::cell::RefCell" | "core::cell::RefCell" => {
                // `RefCell` has no `Drop` impl of its own: the
                // flag `Cell` is plain data, and the value's glue
                // runs through the `UnsafeCell` field wrapper.
                let inner = self.plan(args.first()?)?;
                return Some(match inner {
                    P::Trivial => P::Trivial,
                    plan => P::At {
                        path: vec![RefProj::Field(1), RefProj::Field(0)],
                        plan: Box::new(plan),
                    },
                });
            }
            "std::result::Result" => {
                // Same value shape as `Option`: variant 0 is `Ok`,
                // variant 1 is `Err`, the payload sits at field 0.
                // (Needed so a `try_borrow` result holding a guard
                // can be dropped: the guard's release is the Ok
                // payload's glue.)
                let ok = self.plan(args.first()?)?;
                let err = self.plan(args.get(1)?)?;
                return Some(match (ok, err) {
                    (P::Trivial, P::Trivial) => P::Trivial,
                    (ok, err) => P::Variants {
                        variants: vec![at_field0(ok), at_field0(err)],
                    },
                });
            }
            // The `try_borrow` error types: unit structs with no
            // fields and no `Drop` impl.
            "std::cell::BorrowError" | "core::cell::BorrowError" => {
                return Some(P::Trivial)
            }
            "std::cell::BorrowMutError" | "core::cell::BorrowMutError" => {
                return Some(P::Trivial)
            }
            "std::ops::Range" | "std::ops::RangeInclusive" => {
                let bound = self.plan(args.first()?)?;
                return Some(collapse_fields(vec![bound.clone(), bound], true));
            }
            "std::array::IntoIter" => {
                // The modelled value does not expose the buffer's
                // elements as fields, so only a trivial element
                // type is accounted for.
                let elem = self.plan(args.first()?)?;
                return match elem {
                    P::Trivial => Some(P::Trivial),
                    _ => None,
                };
            }
            _ => {}
        }
        // Any other ADT: its glue is its `Drop` impl (called first,
        // as in rustc's drop glue) followed by its fields' glue.
        let mut steps = Vec::new();
        if let Some(func) = conv.dtor_body(&name, args) {
            steps.push(P::CallDtor { func });
        } else if conv.dtor_impl_known(&name) {
            // The impl exists but has no converted body (a generic
            // impl gets no monomorphic instance in the program):
            // its effects can neither be executed nor ruled out.
            return None;
        } else if !conv.is_local_type(&name) {
            // Not the entry crate's type: an impl could exist that
            // the name table never recorded.
            return None;
        }
        steps.push(self.variant_fields_plan(def, args)?);
        Some(collapse_seq(steps))
    }

    /// Variant indices of one ADT / coroutine instantiation with
    /// any evidence in the program: field projections, aggregate
    /// constructions, the discriminant table. Variant 0 always
    /// exists (every ADT and every coroutine has one).
    fn evidenced_variants(
        &self,
        def: mir::DefId,
        args: &[mir::Ty],
    ) -> std::collections::BTreeSet<u32> {
        let conv = self.conv;
        let mut variants: std::collections::BTreeSet<u32> = Default::default();
        variants.insert(0);
        for (key, _) in &conv.adt_fields.entries {
            if key.def == def && key.generic_args.as_slice() == args {
                variants.insert(key.variant);
            }
        }
        for (d, a, v, _) in &conv.adt_aggregates.entries {
            if *d == def && a.as_slice() == args {
                variants.insert(*v);
            }
        }
        if let Some(discs) = conv.discriminants.get(&def) {
            for v in 0..discs.len() {
                variants.insert(v as u32);
            }
        }
        variants
    }

    /// The drop glue of one coroutine instantiation: the live
    /// state's fields, state by state. Coroutine layouts are dense
    /// (unresumed, one state per suspension point, then the returned
    /// and panicked states), and a state's fields are all projected
    /// by the body that saves or resumes it - so a state with no
    /// evidence holds no fields to drop. The returned and panicked
    /// states, which follow the last suspension state and never
    /// hold fields (the unwinding path drops the live locals before
    /// the tag is set to panicked), are covered by extending the
    /// plan two past the last evidenced state.
    fn coroutine_plan(
        &mut self,
        def: mir::DefId,
        args: &[mir::Ty],
    ) -> Option<ir::DropPlan> {
        use ir::DropPlan as P;
        let variants = self.evidenced_variants(def, args);
        let max_v = *variants.iter().max().expect("variant 0 present") + 2;
        let mut plans = Vec::new();
        for v in 0..=max_v {
            if !variants.contains(&v) {
                plans.push(P::Trivial);
                continue;
            }
            plans.push(self.one_variant_fields(def, args, v)?);
        }
        Some(P::Variants { variants: plans })
    }

    /// The field glue of one ADT, across its variants: a single
    /// `Fields` plan for a struct, a `Variants` plan for an enum.
    fn variant_fields_plan(
        &mut self,
        def: mir::DefId,
        args: &[mir::Ty],
    ) -> Option<ir::DropPlan> {
        use ir::DropPlan as P;
        let variants = self.evidenced_variants(def, args);
        let max_v = *variants.iter().max().expect("variant 0 present");
        let mut plans = Vec::new();
        for v in 0..=max_v {
            if !variants.contains(&v) {
                // A variant index inside the discriminant range
                // that nothing in the program ever constructs or
                // projects: its fields are unaccounted for.
                plans.push(P::Unknown);
                continue;
            }
            plans.push(self.one_variant_fields(def, args, v)?);
        }
        if plans.len() == 1 {
            Some(plans.pop().expect("one variant"))
        } else {
            Some(P::Variants { variants: plans })
        }
    }

    /// The field glue of one variant of one ADT.
    fn one_variant_fields(
        &mut self,
        def: mir::DefId,
        args: &[mir::Ty],
        variant: u32,
    ) -> Option<ir::DropPlan> {
        let conv = self.conv;
        let count = conv.adt_aggregates.get(def, args, variant);
        let field_count = match count {
            Some(n) => n,
            None => conv
                .adt_fields
                .entries
                .iter()
                .filter(|(key, _)| {
                    key.def == def && key.generic_args.as_slice() == args && key.variant == variant
                })
                .map(|(key, _)| key.field as usize + 1)
                .max()
                .unwrap_or(0),
        };
        let mut fields = Vec::with_capacity(field_count);
        for f in 0..field_count as u32 {
            match conv.adt_fields.get(def, args, variant, f) {
                Some(ty) => {
                    let ty = ty.clone();
                    fields.push(self.plan(&ty)?);
                }
                None => {
                    // No projection anywhere records this field's
                    // type (it is only ever written, e.g. by an
                    // aggregate with a literal). Its glue is judged
                    // by the value's own shape at run time.
                    fields.push(ir::DropPlan::Untyped);
                }
            }
        }
        Some(collapse_fields(fields, count.is_some()))
    }
}

/// The trait bounds of the `dyn` type inside `ty`, if any: the
/// search goes through references and raw pointers and into ADT
/// generic arguments (a `Box<dyn Trait>` / `Rc<dyn Trait>` target
/// carries the `dyn` type in its argument).
fn dyn_traits_in(ty: &mir::Ty) -> Option<Vec<String>> {
    match ty {
        mir::Ty::Dynamic(traits, _, _) => Some(traits.clone()),
        mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => dyn_traits_in(inner),
        mir::Ty::Adt(_, args) => args.iter().find_map(dyn_traits_in),
        _ => None,
    }
}

/// The sized pointee an unsizing coercion starts from: the inner
/// type of a reference or raw pointer, the first generic argument
/// of a pointer-shaped ADT (`Box`, `Rc`, `Arc` share the shape), or
/// the type itself for a bare value operand.
fn unsize_pointee(ty: &mir::Ty) -> &mir::Ty {
    match ty {
        mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => inner,
        mir::Ty::Adt(_, args) => args.first().unwrap_or(ty),
        other => other,
    }
}
