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
//! * Unwind actions on `Call` / `Assert` / `Drop` / `InlineAsm` are
//!   preserved per block in the IR body (see [`ir::Body::unwind`]):
//!   a guest panic runs the cleanup blocks of each frame it unwinds
//!   through — the `Drop` terminators of the live locals, in MIR's
//!   own order — before the panic propagates, exactly as native
//!   unwinding does. A panic that escapes a cleanup aborts the run
//!   ([`InterpError::Abort`]).
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
    /// Declared shapes of the program's ADTs, as recorded by the
    /// driver (see `mir::AdtShape`). Used to decide which locals
    /// are provably zero-sized.
    pub adt_shapes: mir::AdtShapes,
    /// Derived-`Debug` shapes of the program's monomorphic ADT
    /// instances, as recorded by the driver (see
    /// `mir::AdtDebugShapes`). Consulted by the `Debug` formatter
    /// for ADT values.
    pub adt_debug_shapes: mir::AdtDebugShapes,
}

/// Widen an `f16` bit pattern to `f64`. The conversion is exact:
/// every f16 value (including subnormals) is representable in f64,
/// and scaling by a power of two never rounds.
pub(crate) fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as f64;
    let v = match exp {
        0 => frac * 2f64.powi(-24),
        31 => {
            if frac == 0.0 {
                return sign * f64::INFINITY;
            }
            return f64::NAN;
        }
        e => (1024.0 + frac) * 2f64.powi(e - 25),
    };
    sign * v
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
#[derive(Debug, Clone, PartialEq)]
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
/// crate's `::_`), while their bodies are distinct items. It is
/// also the case for a `--test` crate's two `main` functions —
/// the user's and the generated harness entry — which share the
/// definition name `<crate>::main`. Bodies
/// sharing a name are disambiguated by their `DefId`, which is
/// also how every reference to them resolves (call and constant
/// operands name a `DefId`, never the shared display name), so a
/// reference still finds exactly its own body and the name-keyed
/// tables stay coherent. A shared name whose bodies cannot be
/// told apart by `DefId` — a missing id, or the same id twice —
/// is a genuine duplicate and fails conversion as a whole.
/// If the recorded body `f` is a closure body that the driver
/// collected through an `Fn`-trait call instance — recognisable by
/// its recorded name being one of the `Fn` / `FnMut` / `FnOnce`
/// call methods (either `std::` or `core::` spelling, with or
/// without the driver's demand-context suffix) and its first
/// argument being the closure environment — return the `DefId` of
/// the closure it is the body of. The driver resolves such an
/// instance straight to the closure's body and records it under
/// the instance's own key, so without this registration the
/// closure's `DefId` names no body and every reference to the
/// closure value falls back to its display name.
///
/// A body that is the call *shim* itself is not the closure body:
/// it takes the environment plus the call's argument tuple as its
/// two arguments (the second argument's type is the closure
/// signature's input tuple) and forwards to the closure body, so
/// it is recognised by that shape and skipped.
/// Whether a body is a closure's own body (as opposed to the
/// `Fn`-trait method's shim or a genuine trait implementation):
/// its first parameter is the closure environment type, and it is
/// not shim-shaped — the shim takes the environment plus the whole
/// argument tuple, matching the closure signature's single tuple
/// input, while the closure's own body takes the environment and
/// the unpacked parameters.
fn body_is_closure_body(body: &mir::Body) -> bool {
    let Some(decl) = body.local_decls.get(1) else {
        return false;
    };
    let env_ty = match &decl.ty {
        mir::Ty::Ref(_, inner, _) => inner.as_ref(),
        other => other,
    };
    let mir::Ty::Closure(_, args) = env_ty else {
        return false;
    };
    if body.arg_count == 2 {
        let sig = args.iter().find_map(|t| match t {
            mir::Ty::FnPtr(sig) => Some(sig),
            _ => None,
        });
        if let (Some(sig), Some(d2)) = (sig, body.local_decls.get(2)) {
            if sig.inputs.len() == 1 && sig.inputs[0] == d2.ty {
                return false;
            }
        }
    }
    true
}

fn closure_body_alias(f: &MirFunction) -> Option<mir::DefId> {
    let base = f.name.split(" [").next().unwrap_or(&f.name);
    if !matches!(
        base,
        "std::ops::Fn::call"
            | "core::ops::Fn::call"
            | "std::ops::FnMut::call_mut"
            | "core::ops::FnMut::call_mut"
            | "std::ops::FnOnce::call_once"
            | "core::ops::FnOnce::call_once"
    ) {
        return None;
    }
    let env_ty = &f.body.local_decls.get(1)?.ty;
    let env_ty = match env_ty {
        mir::Ty::Ref(_, inner, _) => inner.as_ref(),
        other => other,
    };
    let mir::Ty::Closure(def, args) = env_ty else {
        return None;
    };
    if f.body.arg_count == 2 {
        // The shim shape: (environment, argument tuple).
        let sig = args.iter().find_map(|t| match t {
            mir::Ty::FnPtr(sig) => Some(sig),
            _ => None,
        })?;
        if sig.inputs.len() == 1 && sig.inputs[0] == f.body.local_decls.get(2)?.ty {
            return None;
        }
    }
    Some(*def)
}

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

/// The key [`program_keys`] stores one member of `program` under:
/// its name, or the `DefId`-disambiguated form when another member
/// shares the name. A caller that addresses a single known
/// function from outside the program — the driver naming the run
/// entry point — uses this, so the name it hands the interpreter
/// is the one conversion actually stored the body under. (With a
/// shared name and no `DefId` to disambiguate by, conversion as a
/// whole fails in [`program_keys`]; the bare name is returned here
/// and that failure surfaces first.)
pub fn program_key_of(program: &MirProgram, func: &MirFunction) -> String {
    match func.def_id {
        Some(def)
            if program
                .functions
                .iter()
                .filter(|f| f.name == func.name)
                .count()
                > 1 =>
        {
            format!("{}#{def}", func.name)
        }
        _ => func.name.clone(),
    }
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
    out.union_tables = conv.union_tables.clone();
    // Joint fixpoint of conversion, cleanliness and the
    // builtin-served set. Each round recomputes, from the merged
    // program so far:
    //
    // * the *clean* bodies (see `clean_targets`) — the bodies
    //   trait-impl call resolution may target, and
    // * the *unconverted* bodies — the bodies that failed
    //   conversion. A call naming one of them is served another
    //   way when it honestly can be (see `resolve_body_callee`):
    //   by a clean impl body for the same trait call, or by the
    //   builtin model of the callee.
    //
    // and converts every body again with those two sets, merging
    // the results in (a body whose re-conversion fails keeps its
    // previous form; a converted body is never lost). Iterating
    // matters because the sets feed each other: resolving a trait
    // call inside a body can make that body clean, which makes it
    // a resolution target for the next round, and a body withheld
    // only because it called an unconverted body becomes clean
    // once that call is builtin-served. The rounds are monotone —
    // the clean set only grows and the unconverted set only
    // shrinks, because a re-conversion only retargets calls to
    // bodies proven clean in the previous round or replaces an
    // edge to an unconverted body with a neutral builtin call —
    // so the loop reaches a fixpoint: in the final program, every
    // resolution target is clean in that same program, and every
    // diverted call names a body that is unconverted in that same
    // program. The round cap is a defensive bound only.
    let mut used: Option<(HashSet<String>, HashMap<String, String>)> = None;
    for _ in 0..MAX_RESOLUTION_ROUNDS {
        let targets = clean_targets(&out);
        let unconverted = unconverted_bodies(program, &keys, &out);
        if used
            .as_ref()
            .is_some_and(|(t, u)| *t == targets && *u == unconverted)
        {
            break;
        }
        if targets.is_empty() && unconverted.is_empty() {
            break;
        }
        let conv = Converter::with_resolution(
            program,
            &keys,
            targets.clone(),
            unconverted.clone(),
        );
        let (resolved, _) = convert_each(&conv, program, &keys);
        for (key, body) in resolved.functions {
            out.unconverted.remove(&key);
            out.functions.insert(key, body);
        }
        used = Some((targets, unconverted));
    }
    failures.retain(|key, _| !out.functions.contains_key(key));
    out.adt_layouts = program.adt_layouts.clone();
    out.adt_debug_shapes = program.adt_debug_shapes.clone();
    mark_caller_location_bodies(&mut out);
    Ok((out, failures))
}

/// Mark the `#[track_caller]` bodies of a converted program: the
/// driver appends the caller-location stamp as one extra
/// argument to every direct call of such a function, so a body
/// whose converted callers all pass `arg_count + 1` arguments is
/// a track-caller body. Each marked body gains a synthetic final
/// local (typed as the location reference) that the interpreter
/// binds the stamp to (see `Body::caller_location_local`).
fn mark_caller_location_bodies(program: &mut ir::Program) {
    let mut marked: Vec<String> = Vec::new();
    for body in program.functions.values() {
        for block in &body.blocks {
            if let ir::TerminatorKind::Call { func, args, .. } = &block.terminator {
                if let Some(callee) = program.functions.get(func) {
                    if args.len() == callee.arg_count + 1
                        && callee.caller_location_local.is_none()
                        && !marked.contains(func)
                    {
                        marked.push(func.clone());
                    }
                }
            }
        }
    }
    for name in marked {
        let Some(body) = program.functions.get_mut(&name) else {
            continue;
        };
        let idx = body.local_tys.len();
        body.local_tys.push(ir::Ty::Ref(Box::new(ir::Ty::Adt(
            "std::panic::Location".to_string(),
        ))));
        if !body.zst_locals.is_empty() {
            body.zst_locals.push(None);
        }
        body.caller_location_local = Some(idx);
    }
}

/// The defensive bound on the joint-fixpoint rounds in
/// `convert_all` (see there). The rounds are monotone, so the
/// bound is never the reason the loop stops in practice.
const MAX_RESOLUTION_ROUNDS: usize = 16;

/// The bodies of `out` that failed conversion, keyed by program
/// key with the function's plain name (the key itself can carry a
/// `DefId` disambiguation suffix; see `program_keys`). Later
/// fixpoint rounds use the table to serve calls naming an
/// unconverted body another way (see `resolve_body_callee`).
fn unconverted_bodies(
    program: &MirProgram,
    keys: &[String],
    out: &ir::Program,
) -> HashMap<String, String> {
    program
        .functions
        .iter()
        .zip(keys)
        .filter(|(_, key)| out.unconverted.contains_key(*key))
        .map(|(f, key)| (key.clone(), f.name.clone()))
        .collect()
}

/// The bodies of a converted program that trait-impl call
/// resolution may target in the next fixpoint round (see
/// `convert_all`):
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
            // A dyn value's glue is chosen from the value at run
            // time, like a vtable call's target: no static edges.
            ir::DropPlan::Dyn => true,
            // Releasing a RefCell borrow calls no destructor body.
            ir::DropPlan::RefCellRelease { .. } => true,
            ir::DropPlan::ChannelHandleDrop { .. } => true,
            // Releasing a weak reference calls no destructor body.
            ir::DropPlan::RcWeakRelease => true,
            ir::DropPlan::RcHandleDrop { value } => walk(value, adts, targets, seen),
            // Judged from the value at run time; calls no
            // destructor body.
            ir::DropPlan::Dynamic => true,
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
    pub stderr: String,
}

/// Like [`interpret_mir`], but also returns the captured guest stdout.
pub fn interpret_mir_captured(
    program: &MirProgram,
    entry: &str,
    args: Vec<Value>,
) -> CapturedRun {
    interpret_mir_captured_with_config(
        program,
        entry,
        args,
        crate::InterpConfig::default(),
    )
}

/// Like [`interpret_mir_captured`], with an explicit interpreter
/// configuration (the driver hands the guest program's
/// command-line arguments through it; see
/// [`crate::InterpConfig::guest_args`]).
pub fn interpret_mir_captured_with_config(
    program: &MirProgram,
    entry: &str,
    args: Vec<Value>,
    config: crate::InterpConfig,
) -> CapturedRun {
    match convert_all(program) {
        Ok((ir_program, failures)) => {
            // The entry function's own conversion failure is a
            // whole-run failure with no guest output, as before.
            if let Some(e) = failures.get(entry) {
                return CapturedRun {
                    result: Err(MirRunError::from(e.clone())),
                    stdout: String::new(),
                    stderr: String::new(),
                };
            }
            let mut interp = Interpreter::with_config(&ir_program, config);
            let result = interp.run(entry, args).map_err(MirRunError::from);
            CapturedRun {
                result,
                stdout: interp.output().to_string(),
                stderr: interp.error_output().to_string(),
            }
        }
        Err(e) => CapturedRun {
            result: Err(MirRunError::from(e)),
            stdout: String::new(),
            stderr: String::new(),
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

/// The cardinality of a type's value set, for the zero-sized
/// synthesis in `Converter::unique_value`.
enum UniqueValue {
    /// Exactly one value; here it is.
    One(Value),
    /// No values (an uninhabited type).
    Uninhabited,
    /// More than one value, or not provable here.
    Many,
}

/// Whether `ty` is a thin pointer type: a raw pointer or
/// reference whose pointee carries no metadata half (not a slice,
/// `str`, `dyn` or opaque alias pointee). A thin pointer's value
/// in this model is exactly its target, so transmutes between
/// thin pointer types — and between a thin pointer and a
/// transparent wrapper around one — reinterpret nothing.
fn is_thin_ptr_ty(ty: &mir::Ty) -> bool {
    match ty {
        mir::Ty::RawPtr(pointee, _) | mir::Ty::Ref(_, pointee, _) => !matches!(
            **pointee,
            mir::Ty::Slice(_)
                | mir::Ty::Str
                | mir::Ty::Dynamic(..)
                | mir::Ty::Alias(..)
        ),
        _ => false,
    }
}

/// Whether `ty` is a fat pointer type: a raw pointer or
/// reference whose pointee carries a metadata half (a slice,
/// `str` or `dyn` pointee). A fat pointer's value in this model
/// is its [`ir::Value::FatRef`] / [`ir::Value::DynRef`], so
/// transmutes between fat pointer types — and between a fat
/// pointer and a transparent wrapper around one — reinterpret
/// nothing either: the metadata is part of the value, and the
/// pointee type lives in the destination's declared type.
fn is_fat_ptr_ty(ty: &mir::Ty) -> bool {
    match ty {
        mir::Ty::RawPtr(pointee, _) | mir::Ty::Ref(_, pointee, _) => matches!(
            **pointee,
            mir::Ty::Slice(_) | mir::Ty::Str | mir::Ty::Dynamic(..)
        ),
        _ => false,
    }
}

/// The constant form of a uniquely determined zero-sized value
/// (the only values [`Converter::zst_value`] synthesizes), or
/// `None` for a value kind with no constant form.
fn value_to_constant(v: &Value) -> Option<ir::Constant> {
    match v {
        Value::Unit => Some(ir::Constant::Unit),
        Value::Tuple(items) => items
            .iter()
            .map(value_to_constant)
            .collect::<Option<Vec<_>>>()
            .map(ir::Constant::Tuple),
        Value::Enum { variant, fields } => fields
            .iter()
            .map(value_to_constant)
            .collect::<Option<Vec<_>>>()
            .map(|fields| ir::Constant::Adt {
                variant: *variant,
                fields,
            }),
        Value::Closure { def, env } if env.is_empty() => {
            Some(ir::Constant::Closure(def.clone()))
        }
        Value::FnPtr(name) => Some(ir::Constant::FnDef(name.clone())),
        _ => None,
    }
}

/// Whether `v` holds a callable value (a closure or a function
/// pointer) anywhere inside it.
fn value_contains_callable(v: &Value) -> bool {
    match v {
        Value::Closure { .. } | Value::FnPtr(_) => true,
        Value::Tuple(items) => items.iter().any(value_contains_callable),
        Value::Enum { fields, .. } => fields.iter().any(value_contains_callable),
        _ => false,
    }
}

/// The shape of a transparent wrapper struct for `transmute`
/// purposes: a struct whose fields are exactly one pointer
/// payload field (thin, or fat when matched by
/// [`Converter::fat_wrapper_shape`]) plus any number of
/// zero-sized fields whose values are uniquely determined by
/// their types (`NonNull`, `Pin`, `PtrAndZst` are the shapes
/// std transmutes through).
struct WrapperShape {
    /// Total number of fields of the struct.
    field_count: usize,
    /// Index of the payload (pointer) field.
    payload_index: usize,
    /// Instantiated type of the payload field.
    payload_ty: mir::Ty,
    /// Unique values of the zero-sized fields, by field index.
    fill: Vec<(usize, ir::Constant)>,
}

/// Substitute a definition's generic arguments for its parameters
/// in a declared field type (see `mir::AdtShape`): `Param(i)` is
/// replaced by the instantiation's `i`-th argument, recursively.
/// A parameter with no corresponding argument is left as-is (it
/// makes the type unprovable downstream, never unsound).
fn subst_ty(ty: &mir::Ty, args: &[mir::Ty]) -> mir::Ty {
    match ty {
        mir::Ty::Param(i, _) => args.get(*i as usize).cloned().unwrap_or_else(|| ty.clone()),
        mir::Ty::Tuple(ts) => {
            mir::Ty::Tuple(ts.iter().map(|t| subst_ty(t, args)).collect())
        }
        mir::Ty::Array(e, l) => mir::Ty::Array(Box::new(subst_ty(e, args)), l.clone()),
        mir::Ty::Adt(d, a) => {
            mir::Ty::Adt(*d, a.iter().map(|t| subst_ty(t, args)).collect())
        }
        mir::Ty::Ref(r, t, m) => mir::Ty::Ref(r.clone(), Box::new(subst_ty(t, args)), *m),
        mir::Ty::RawPtr(t, m) => mir::Ty::RawPtr(Box::new(subst_ty(t, args)), *m),
        mir::Ty::Slice(t) => mir::Ty::Slice(Box::new(subst_ty(t, args))),
        mir::Ty::Pat(t, pat) => mir::Ty::Pat(Box::new(subst_ty(t, args)), pat.clone()),
        mir::Ty::Alias(k, d, a) => {
            mir::Ty::Alias(*k, *d, a.iter().map(|t| subst_ty(t, args)).collect())
        }
        mir::Ty::Closure(d, a) => {
            mir::Ty::Closure(*d, a.iter().map(|t| subst_ty(t, args)).collect())
        }
        mir::Ty::Coroutine(d, a) => {
            mir::Ty::Coroutine(*d, a.iter().map(|t| subst_ty(t, args)).collect())
        }
        mir::Ty::CoroutineWitness(d, a) => {
            mir::Ty::CoroutineWitness(*d, a.iter().map(|t| subst_ty(t, args)).collect())
        }
        mir::Ty::FnDef(d, a) => {
            mir::Ty::FnDef(*d, a.iter().map(|t| subst_ty(t, args)).collect())
        }
        mir::Ty::Dynamic(bounds, t, r) => {
            mir::Ty::Dynamic(bounds.clone(), Box::new(subst_ty(t, args)), r.clone())
        }
        _ => ty.clone(),
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
        | R::Reborrow(_, _, place)
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
                if let Some(value) = &op.in_value {
                    scan_operand_fields(table, body, value);
                }
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
        // A pattern type is its base type for every
        // projection step.
        while let mir::Ty::Pat(inner, _) = ty {
            ty = *inner;
        }
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

/// The dispatch chosen for a call that names an `Fn` trait method
/// (see `Converter::fn_trait_or_direct`).
enum FnResolution {
    /// Evaluate the receiver as a callable value (`CallTrait`).
    Trait(ir::FnCallKind, Vec<mir::Ty>),
    /// Call the named body directly.
    Direct(String),
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
    /// ADT shape tables of the program (see `MirProgram`).
    adt_shapes: &'a mir::AdtShapes,
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
    /// it to the named bodies (later fixpoint rounds).
    impl_targets: Option<HashSet<String>>,
    /// The bodies that failed conversion in the previous fixpoint
    /// round, keyed by program key with the function's plain name:
    /// a call naming one of them is served another way when it
    /// honestly can be (see `resolve_body_callee`). Empty in the
    /// first conversion phase.
    unconverted: HashMap<String, String>,
    /// Scalar-shape tables of the program's union instances (see
    /// [`ir::Program::union_tables`]), interned in
    /// `Converter::new` by [`Self::compute_union_tables`]: every
    /// conversion round recomputes the same tables, so the
    /// indices baked into converted places agree across rounds.
    union_tables: Vec<Vec<Option<ir::ScalarShape>>>,
    /// Union instance -> index into `union_tables`.
    union_table_ids: Vec<(mir::AdtDefId, Vec<mir::Ty>, usize)>,
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
        // A closure body collected through an `Fn`-trait call
        // instance is recorded under the instance's key (the trait
        // method's name), not under the closure's own `DefId`:
        // register the closure `DefId` for that key so closure
        // values and closure types resolve to the body (see
        // `closure_body_alias`).
        for (f, key) in program.functions.iter().zip(keys) {
            if let Some(def) = closure_body_alias(f) {
                def_names.entry(def).or_insert_with(|| key.clone());
            }
        }
        let local_crate = program
            .functions
            .iter()
            .find(|f| f.name == "main" || f.name.ends_with("::main"))
            .map(|f| f.name.split("::").next().unwrap_or("").to_string());
        let mut conv = Converter {
            bodies,
            def_names,
            names: &program.fn_names,
            adt_fields,
            discriminants: &program.adt_discriminants,
            adt_shapes: &program.adt_shapes,
            adt_aggregates,
            local_crate,
            impl_targets: None,
            unconverted: HashMap::new(),
            union_tables: Vec::new(),
            union_table_ids: Vec::new(),
        };
        conv.compute_union_tables(program);
        conv
    }

    /// A converter for a joint-fixpoint round in `convert_all`:
    /// trait-impl call resolution restricted to `targets` (the
    /// previous round's clean bodies), and calls naming a body in
    /// `unconverted` (the previous round's conversion failures)
    /// served another way where that is honest (see
    /// `resolve_body_callee`).
    fn with_resolution(
        program: &'a MirProgram,
        keys: &[String],
        targets: HashSet<String>,
        unconverted: HashMap<String, String>,
    ) -> Self {
        let mut conv = Self::new(program, keys);
        conv.impl_targets = Some(targets);
        conv.unconverted = unconverted;
        conv
    }

    /// Whether the ADT is a union.
    fn is_union(&self, def: &mir::AdtDefId) -> bool {
        self.adt_shapes
            .get(def)
            .map(|shape| shape.kind == mir::AdtShapeKind::Union)
            .unwrap_or(false)
    }

    /// Intern the scalar-shape tables of every union instance a
    /// place in the program can project a field of. The instance
    /// set is the closure of the bodies' local types and the
    /// projection-recovered field types under field, pointee and
    /// element types: a place walk starts at a local, and each
    /// step's type is one of those (a `Field` step's type is the
    /// previous type's field type, a `Deref` step's the pointee,
    /// an index step's the element). The closure is a pure
    /// function of the program, so every conversion round
    /// interns the same tables in the same order.
    fn compute_union_tables(&mut self, program: &MirProgram) {
        let mut visited: Vec<mir::Ty> = Vec::new();
        let mut worklist: Vec<mir::Ty> = Vec::new();
        for f in &program.functions {
            for decl in &f.body.local_decls {
                worklist.push(decl.ty.clone());
            }
        }
        for (_, ty) in &self.adt_fields.entries {
            worklist.push(ty.clone());
        }
        while let Some(ty) = worklist.pop() {
            let mut ty = ty;
            while let mir::Ty::Pat(inner, _) = ty {
                ty = *inner;
            }
            if visited.contains(&ty) {
                continue;
            }
            visited.push(ty.clone());
            match &ty {
                mir::Ty::Adt(def, args) => {
                    if self.is_union(def) {
                        self.intern_union_table(def, args);
                    }
                    let variant_count = self
                        .adt_shapes
                        .get(def)
                        .map(|shape| shape.variants.len())
                        .unwrap_or(0);
                    for variant in 0..variant_count {
                        if let Some(ftys) =
                            self.instantiated_field_tys(def, args, variant as u32)
                        {
                            worklist.extend(ftys);
                        }
                    }
                }
                mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => {
                    worklist.push((**inner).clone());
                }
                mir::Ty::Array(elem, _) | mir::Ty::Slice(elem) => {
                    worklist.push((**elem).clone());
                }
                mir::Ty::Tuple(ts) => worklist.extend(ts.iter().cloned()),
                _ => {}
            }
        }
    }

    /// Intern one union instance's per-field scalar shapes.
    fn intern_union_table(&mut self, def: &mir::AdtDefId, args: &[mir::Ty]) {
        if self
            .union_table_ids
            .iter()
            .any(|(d, a, _)| d == def && a.as_slice() == args)
        {
            return;
        }
        let fields = self
            .instantiated_field_tys(def, args, 0)
            .unwrap_or_default();
        let shapes: Vec<Option<ir::ScalarShape>> = fields
            .iter()
            .map(|fty| self.union_field_shape(fty))
            .collect();
        let idx = self.union_tables.len();
        self.union_tables.push(shapes);
        self.union_table_ids.push((*def, args.to_vec(), idx));
    }

    /// The table index of a union instance interned by
    /// [`Self::compute_union_tables`]; `None` for an instance
    /// the closure did not reach (the place then converts as a
    /// plain field projection, the pre-union-fields behaviour).
    fn union_table_for(&self, def: &mir::AdtDefId, args: &[mir::Ty]) -> Option<usize> {
        self.union_table_ids
            .iter()
            .find(|(d, a, _)| d == def && a.as_slice() == args)
            .map(|(_, _, idx)| *idx)
    }

    /// The scalar shape of one union field's type: the transmute
    /// shape when the type has one, plus a tuple of equal scalar
    /// shapes — its elements occupy consecutive slots of one
    /// width in every field order, so the tuple's bit pattern is
    /// the array pattern over the shared element shape (and a
    /// tuple value and an array value are the same tuple of
    /// elements in this model).
    fn union_field_shape(&self, ty: &mir::Ty) -> Option<ir::ScalarShape> {
        if let Some(shape) = self.scalar_shape(ty, 0) {
            return Some(shape);
        }
        if let mir::Ty::Tuple(ts) = ty {
            if ts.is_empty() || ts.len() > 128 {
                return None;
            }
            let first = self.scalar_shape(&ts[0], 0)?;
            if first.bits() == 0 || first.bits().checked_mul(ts.len() as u32)? > 128 {
                return None;
            }
            for t in &ts[1..] {
                if self.scalar_shape(t, 0).as_ref() != Some(&first) {
                    return None;
                }
            }
            return Some(ir::ScalarShape::Array(
                Box::new(first),
                ts.len() as u64,
            ));
        }
        None
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
        let mut out = ir::Body::new(key.to_string(), body.arg_count, local_tys, blocks);
        out.unwind = body
            .basic_blocks
            .iter()
            .map(|bb| {
                use mir::TerminatorKind as T;
                match bb.terminator.as_ref().map(|t| &t.kind) {
                    Some(
                        T::Call { unwind, .. }
                        | T::Assert { unwind, .. }
                        | T::Drop { unwind, .. }
                        | T::InlineAsm { unwind, .. },
                    ) => convert_unwind_action(unwind),
                    _ => ir::UnwindAction::Continue,
                }
            })
            .collect();
        out.zst_locals = body
            .local_decls
            .iter()
            .map(|decl| self.zst_value(&decl.ty, 0))
            .collect();
        out.zst_fields = body
            .local_decls
            .iter()
            .map(|decl| {
                let mut map = std::collections::HashMap::new();
                // The walk budget bounds the conversion cost on
                // bodies whose types are enormous (iterator chains
                // with type lengths in the millions): the paths it
                // serves are shallow wrappers, collected long
                // before the budget runs out.
                let mut budget = 256usize;
                self.collect_zst_fields(&decl.ty, &mut Vec::new(), &mut map, 0, &mut budget);
                map
            })
            .collect();
        Ok(out)
    }

    /// The canonical value of a type that is provably zero-sized,
    /// or `None` when the type is not provably zero-sized (or its
    /// value is not uniquely determined by the type). rustc MIR
    /// never writes the place of a zero-sized local in many bodies
    /// — a unit struct constructor's body is a bare `return` — so
    /// a never-written read of such a place denotes this value.
    ///
    /// The proof is structural: unit, and tuples of zero-sized
    /// types; arrays of zero-sized elements and empty arrays;
    /// structs and enums with a single inhabited variant whose
    /// fields are all zero-sized (from the driver's shape table,
    /// with the instantiation's arguments substituted for the
    /// definition's parameters — a variant with an uninhabited
    /// field type can never be constructed, so `Option<Empty>`
    /// still has the single value `None`); zero-capture closures;
    /// function items. Enums with two or more inhabited variants
    /// are excluded (the type does not determine the variant), as
    /// are unions with more than one field (the type does not
    /// determine the active field; a single-field union's value
    /// is its one field's value) and opaque aliases (the hidden
    /// type is not visible here).
    fn zst_value(&self, ty: &mir::Ty, depth: u32) -> Option<Value> {
        match self.unique_value(ty, depth) {
            UniqueValue::One(v) => Some(v),
            _ => None,
        }
    }

    /// Collect the canonical values of the single-valued field
    /// paths of `ty` into `out`, keyed by field-index path (see
    /// `ir::Body::zst_fields`). Structs, unions and tuples expose
    /// their fields under bare field projections; an enum's fields
    /// sit behind a downcast step, which the runtime lookup does
    /// not follow, so enum fields are not collected. The walk is
    /// depth- and size-capped: it serves the shallow paths MIR
    /// actually projects (a wrapper or two around a zero-sized
    /// field), not arbitrary recursive types.
    fn collect_zst_fields(
        &self,
        ty: &mir::Ty,
        path: &mut Vec<usize>,
        out: &mut std::collections::HashMap<Vec<usize>, Value>,
        depth: u32,
        budget: &mut usize,
    ) {
        if depth > 8 || out.len() > 4096 || *budget == 0 {
            return;
        }
        *budget -= 1;
        let mut ty = ty.clone();
        while let mir::Ty::Pat(inner, _) = ty {
            ty = *inner;
        }
        let fields: Vec<mir::Ty> = match &ty {
            mir::Ty::Tuple(ts) => ts.clone(),
            mir::Ty::Adt(def, args) => {
                let Some(shape) = self.adt_shapes.get(def) else {
                    return;
                };
                if shape.kind == mir::AdtShapeKind::Enum {
                    return;
                }
                match self.instantiated_field_tys(def, args, 0) {
                    Some(fields) => fields,
                    None => return,
                }
            }
            _ => return,
        };
        for (i, field_ty) in fields.iter().enumerate() {
            path.push(i);
            if let Some(v) = self.zst_value(field_ty, 0) {
                out.insert(path.clone(), v);
            }
            self.collect_zst_fields(field_ty, path, out, depth + 1, budget);
            path.pop();
            if out.len() > 4096 {
                return;
            }
        }
    }

    /// Whether a type has exactly one value (and what it is), no
    /// value at all, or many. The one-value case is what
    /// [`Self::zst_value`] synthesizes: a type with a single value
    /// is zero-sized in every layout that matters here, and rustc
    /// MIR may leave its places unwritten. An enum contributes its
    /// unique inhabited variant: a variant with an uninhabited
    /// field type (`Empty`, `!`, a struct containing one) can
    /// never be constructed, so `Option<Empty>` has the single
    /// value `None` even though the enum declares two variants.
    fn unique_value(&self, ty: &mir::Ty, depth: u32) -> UniqueValue {
        if depth > 32 {
            return UniqueValue::Many;
        }
        match ty {
            mir::Ty::Never => UniqueValue::Uninhabited,
            mir::Ty::Tuple(ts) if ts.is_empty() => UniqueValue::One(Value::Unit),
            mir::Ty::Tuple(ts) => {
                let mut vals = Vec::with_capacity(ts.len());
                for t in ts {
                    match self.unique_value(t, depth + 1) {
                        UniqueValue::One(v) => vals.push(v),
                        UniqueValue::Uninhabited => return UniqueValue::Uninhabited,
                        UniqueValue::Many => return UniqueValue::Many,
                    }
                }
                UniqueValue::One(Value::Tuple(vals))
            }
            mir::Ty::Array(elem, len) => {
                let n = match len {
                    mir::ArrayLen::Known(n) => *n,
                    _ => return UniqueValue::Many,
                };
                if n == 0 {
                    return UniqueValue::One(Value::Tuple(Vec::new()));
                }
                if n > 4096 {
                    return UniqueValue::Many;
                }
                match self.unique_value(elem, depth + 1) {
                    UniqueValue::One(v) => {
                        UniqueValue::One(Value::Tuple(vec![v; n as usize]))
                    }
                    UniqueValue::Uninhabited => UniqueValue::Uninhabited,
                    UniqueValue::Many => UniqueValue::Many,
                }
            }
            mir::Ty::Adt(def, args) => {
                let Some(shape) = self.adt_shapes.get(def) else {
                    return UniqueValue::Many;
                };
                if shape.kind == mir::AdtShapeKind::Union
                    && shape.variants.first().map_or(0, |f| f.len()) != 1
                {
                    // A union's active field is not determined by
                    // its type — except a single-field union, whose
                    // one field's value the loop below determines
                    // like a struct's.
                    return UniqueValue::Many;
                }
                let mut single: Option<(u32, Vec<Value>)> = None;
                for (vi, variant_fields) in shape.variants.iter().enumerate() {
                    let mut fields = Vec::with_capacity(variant_fields.len());
                    let mut inhabited = true;
                    for (fi, fty) in variant_fields.iter().enumerate() {
                        // A field type recovered from a projection
                        // in some body is already instantiated for
                        // this `args`; the shape's declared type is
                        // in terms of the definition's parameters,
                        // whose indices also count lifetimes and
                        // consts that the converted argument list
                        // omits, so substitution can misalign.
                        // Prefer the recovered type when one exists.
                        let fty = self
                            .adt_fields
                            .get(*def, args, vi as u32, fi as u32)
                            .cloned()
                            .unwrap_or_else(|| subst_ty(fty, args));
                        match self.unique_value(&fty, depth + 1) {
                            UniqueValue::One(v) => fields.push(v),
                            UniqueValue::Uninhabited => {
                                inhabited = false;
                                break;
                            }
                            // An inhabited variant whose field
                            // has many values gives the type many
                            // values.
                            UniqueValue::Many => return UniqueValue::Many,
                        }
                    }
                    if !inhabited {
                        continue;
                    }
                    if single.is_some() {
                        return UniqueValue::Many;
                    }
                    single = Some((vi as u32, fields));
                }
                match single {
                    Some((variant, fields)) => {
                        UniqueValue::One(Value::Enum { variant, fields })
                    }
                    None => UniqueValue::Uninhabited,
                }
            }
            mir::Ty::Closure(def, args) => match args.get(2) {
                Some(mir::Ty::Tuple(ts)) if ts.is_empty() => {
                    UniqueValue::One(Value::Closure {
                        def: self.def_display_name(def),
                        env: Vec::new(),
                    })
                }
                _ => UniqueValue::Many,
            },
            mir::Ty::FnDef(def, _) => {
                UniqueValue::One(Value::FnPtr(self.body_key(def)))
            }
            _ => UniqueValue::Many,
        }
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
    /// The program key of the `static` item `def`: its display
    /// name, disambiguated with the `DefId` when several named
    /// items share that name (the [`program_keys`] rule applied to
    /// statics). Computed from the name table alone, so every
    /// conversion phase assigns the same key to the same static.
    fn static_key(&self, def: &mir::DefId) -> String {
        let name = self.def_display_name(def);
        if self.names.values().filter(|n| **n == name).count() > 1 {
            format!("{name}#{def}")
        } else {
            name
        }
    }

    fn def_display_name(&self, def: &mir::DefId) -> String {
        self.names
            .get(def)
            .or_else(|| self.def_names.get(def))
            .cloned()
            .unwrap_or_else(|| format!("{def:?}"))
    }

    /// The program key of the body a `DefId` names, for references
    /// that must land on one specific converted body (closure
    /// values and closure types, and `FnDef` constants — a function
    /// item materialized as a value names the instance its
    /// constant refers to, not the definition's first instance):
    /// the converted-body table first — it is keyed by the exact
    /// (definition, instance) id the driver assigned, so two
    /// instances of one definition resolve to their own bodies —
    /// and the display name only as a fallback for definitions
    /// with no body in the program.
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

    /// Whether a pointer cast from `source` to `target` goes from
    /// a transparent single-field wrapper (`UnsafeCell<T>` /
    /// `MaybeUninit<T>`) to the wrapped type itself. Decided from
    /// the cast's two types, both known at conversion time: the
    /// source pointee is the wrapper ADT and its single generic
    /// argument is exactly the target pointee.
    fn cast_enters_wrapper_field(&self, source: &mir::Ty, target: &mir::Ty) -> bool {
        fn pointee(ty: &mir::Ty) -> Option<&mir::Ty> {
            match ty {
                mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => Some(inner),
                _ => None,
            }
        }
        let (Some(src), Some(tgt)) = (pointee(source), pointee(target)) else {
            return false;
        };
        match src {
            mir::Ty::Adt(def, args) => {
                matches!(
                    self.def_display_name(def).as_str(),
                    "std::cell::UnsafeCell"
                        | "core::cell::UnsafeCell"
                        | "std::mem::MaybeUninit"
                        | "core::mem::MaybeUninit"
                ) && args.len() == 1
                    && args[0] == *tgt
            }
            _ => false,
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
            // `f128` falls through to the unsupported arm.
            mir::Ty::Float(mir::FloatTy::F16) => ir::Ty::Float(ir::FloatTy::F16),
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
            // A pattern type shares its base type's
            // representation: a value of the pattern type is a
            // base value, and no execution step needs the range
            // itself. Construction in a valid program guarantees
            // the range, and layout facts that depend on it
            // (niches) reach the interpreter through the driver's
            // rustc-derived tables, not through this type.
            mir::Ty::Pat(inner, _) => self.convert_ty(inner, context)?,
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

    fn convert_place(
        &self,
        body: &mir::Body,
        place: &mir::Place,
    ) -> Result<ir::Place, ConvertError> {
        let mut out = ir::Place::from_index(place.local.index());
        // The type before each projection step, tracked the way
        // `place_ty` tracks it: a `Field` step whose base type
        // is a union converts to a union field step (its read
        // reinterprets the held field's bits; see
        // `ir::ProjectionElem::UnionField`), every other step to
        // the plain step it always was. Once the walk loses the
        // type, later steps convert plainly too.
        let mut ty = body
            .local_decls
            .get(place.local.index())
            .map(|decl| decl.ty.clone());
        for elem in &place.projection {
            while let Some(mir::Ty::Pat(inner, _)) = &ty {
                ty = Some(inner.as_ref().clone());
            }
            out.projection.push(match elem {
                mir::PlaceElem::Field(idx, field_ty) => {
                    let step = match &ty {
                        Some(mir::Ty::Adt(def, args)) if self.is_union(def) => {
                            match self.union_table_for(def, args) {
                                Some(table) => ir::ProjectionElem::UnionField {
                                    idx: *idx as usize,
                                    table,
                                },
                                None => ir::ProjectionElem::Field(*idx as usize),
                            }
                        }
                        _ => ir::ProjectionElem::Field(*idx as usize),
                    };
                    ty = Some(field_ty.clone());
                    step
                }
                mir::PlaceElem::Deref => {
                    ty = match ty {
                        Some(
                            mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _),
                        ) => Some(*inner),
                        _ => None,
                    };
                    ir::ProjectionElem::Deref
                }
                mir::PlaceElem::Index(local) => {
                    ty = match ty {
                        Some(mir::Ty::Array(elem, _) | mir::Ty::Slice(elem)) => {
                            Some(*elem)
                        }
                        _ => None,
                    };
                    ir::ProjectionElem::Index(ir::Local(local.index()))
                }
                mir::PlaceElem::ConstantIndex {
                    offset, from_end, ..
                } => {
                    ty = match ty {
                        Some(mir::Ty::Array(elem, _) | mir::Ty::Slice(elem)) => {
                            Some(*elem)
                        }
                        _ => None,
                    };
                    ir::ProjectionElem::ConstantIndex {
                        offset: *offset as u64,
                        from_end: *from_end,
                    }
                }
                mir::PlaceElem::Downcast(_, variant) => {
                    ir::ProjectionElem::Downcast(*variant)
                }
                mir::PlaceElem::Subslice { from, to, from_end } => {
                    if let Some(mir::Ty::Array(elem, _)) = &ty {
                        ty = Some(mir::Ty::Slice(elem.clone()));
                    }
                    ir::ProjectionElem::Subslice {
                        from: *from,
                        to: *to,
                        from_end: *from_end,
                    }
                }
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
            // A pattern type is its base type for every
            // projection step.
            while let mir::Ty::Pat(inner, _) = ty {
                ty = *inner;
            }
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
                mir::PlaceElem::Subslice { .. } => match ty {
                    mir::Ty::Array(elem, _) => mir::Ty::Slice(elem),
                    t => t,
                },
                mir::PlaceElem::OpaqueCast(t) | mir::PlaceElem::Subtype(t) => t.clone(),
            };
        }
        Some(ty)
    }

    /// The discriminant table of the `Self` type of a comparison
    /// call: the callee's first generic argument, falling back to
    /// the receiver operand's type (a constant receiver carries no
    /// place type). `None` unless that type is an enum with a
    /// recorded table.
    fn comparison_discriminants(
        &self,
        body: &mir::Body,
        args: &[mir::Operand],
        raw_generic_args: &[mir::Ty],
    ) -> Option<Vec<i128>> {
        let self_ty = raw_generic_args.first().cloned().or_else(|| {
            args.first()
                .and_then(|a| self.operand_place_ty(body, a))
        })?;
        let self_ty = match self_ty {
            mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => *inner,
            t => t,
        };
        match self_ty {
            mir::Ty::Adt(def, _) => {
                let table = self.discriminants.get(&def)?;
                if table.is_empty() {
                    None
                } else {
                    Some(table.clone())
                }
            }
            _ => None,
        }
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
                Ok(ir::Constant::FnDef(self.body_key(def)))
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

    /// A projection step of a static reference, in the
    /// interpreter's reference-projection form.
    fn convert_ref_proj(&self, elem: &mir::PlaceElem) -> Result<RefProj, ConvertError> {
        Ok(match elem {
            mir::PlaceElem::Field(idx, _) => RefProj::Field(*idx as usize),
            mir::PlaceElem::ConstantIndex {
                offset,
                from_end: false,
                ..
            } => RefProj::Index(*offset as usize),
            mir::PlaceElem::Downcast(_, variant) => RefProj::Downcast(*variant),
            other => {
                return Err(ConvertError::UnsupportedConstant {
                    kind: format!("projection {other:?} in a static reference"),
                })
            }
        })
    }

    fn convert_const_value(
        &self,
        value: &mir::ConstValue,
        ctx: Option<&mir::Ty>,
    ) -> Result<ir::Constant, ConvertError> {
        Ok(match value {
            mir::ConstValue::Bool(b) => ir::Constant::Bool(*b),
            mir::ConstValue::Char(c) => ir::Constant::Uint(*c as u128),
            mir::ConstValue::ZeroSized => {
                // A zero-sized constant carries no value of its own:
                // its value is the one its context type determines.
                // When that value is a structure holding a callable
                // (a closure / function pointer inside an enum or
                // tuple - e.g. a `Fun<F>` wrapper whose only field is
                // a zero-capture closure passed as a constant
                // receiver), materialise it so projections and calls
                // through it can reach the callable. Every other
                // zero-sized constant keeps the unit reading: most
                // ZST types' unique values are plain aggregates whose
                // constant form would change existing handling.
                match ctx.and_then(|ty| self.zst_value(ty, 0)) {
                    Some(v) if value_contains_callable(&v) => {
                        match value_to_constant(&v) {
                            Some(k) => k,
                            None => ir::Constant::Unit,
                        }
                    }
                    _ => ir::Constant::Unit,
                }
            }
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
            mir::ConstValue::NullPtr => ir::Constant::NullPtr,
            mir::ConstValue::Uninit => ir::Constant::Uninit,
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
            mir::ConstValue::Adt { variant, fields } => {
                // The fields take the ADT's instantiated field
                // types as their context, exactly as the tuple arm
                // takes its element types: a scalar field's bits
                // carry no signedness of their own, so a `u128`
                // field holding 2^127 decoded context-free reads
                // back as a negative `i128` (the `Some(1 << 127)`
                // constant in checked_shl's test, among others).
                let field_ctxs = match ctx {
                    Some(mir::Ty::Adt(def, args)) => {
                        self.instantiated_field_tys(def, args, *variant)
                    }
                    _ => None,
                };
                ir::Constant::Adt {
                    variant: *variant,
                    fields: fields
                        .iter()
                        .enumerate()
                        .map(|(i, f)| {
                            let fctx = field_ctxs
                                .as_ref()
                                .and_then(|ts| ts.get(i))
                                .map(|t| t as &mir::Ty);
                            self.convert_const_value(f, fctx)
                        })
                        .collect::<Result<_, _>>()?,
                }
            }
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
            mir::ConstValue::StaticRef { def, value, projection } => {
                // The seed is the static's full initializer, typed
                // by the pointee context exactly as for a plain
                // reference constant; the reference itself names the
                // static's program-level storage, projected from the
                // static's base to the pointee.
                let inner_ctx = match ctx {
                    Some(mir::Ty::Ref(_, pointee, _) | mir::Ty::RawPtr(pointee, _)) => {
                        Some(&**pointee)
                    }
                    _ => None,
                };
                ir::Constant::StaticRef {
                    name: self.static_key(def),
                    value: Box::new(self.convert_const_value(value, inner_ctx)?),
                    projection: projection
                        .iter()
                        .map(|e| self.convert_ref_proj(e))
                        .collect::<Result<_, _>>()?,
                }
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
                ir::Constant::FnDef(self.body_key(def))
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
                mir::FloatTy::F16 => Ok(ir::Constant::Float(f16_to_f64(*bits as u16))),
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
        body: &mir::Body,
        op: &mir::Operand,
        ctx: Option<&mir::Ty>,
    ) -> Result<ir::Operand, ConvertError> {
        Ok(match op {
            mir::Operand::Copy(p) => ir::Operand::Copy(self.convert_place(body, p)?),
            mir::Operand::Move(p) => ir::Operand::Move(self.convert_place(body, p)?),
            mir::Operand::Constant(c) => {
                // With no context from the operation, a constant is
                // read under its own recorded type when it has one:
                // the bits of a scalar literal carry no signedness,
                // and the sign-bit guess in `convert_scalar` reads
                // `255u8` as `-1i8`.
                let ctx = ctx.or(c.ty.as_ref());
                ir::Operand::Constant(self.convert_constant(c, ctx)?)
            }
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
    /// 3. Otherwise — a comparison of two literals — the operation
    ///    itself declares no type. The pair still needs ONE
    ///    interpretation: guessing per literal can yield an
    ///    `Int`/`Uint` pair that cannot even be compared. A
    ///    literal's own recorded type (both literals share one type
    ///    in MIR) is that interpretation when either side carries
    ///    one; the sign-bit reading of [`literal_pair_ty`] remains
    ///    for literals with no recorded type (the shape of rustc's
    ///    expanded Div/Rem checks: `Eq(rhs, 0)`, `Eq(rhs, -1)`,
    ///    `Eq(lhs, MIN)` over two literals).
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
            .or_else(|| constant_recorded_ty(&pair.0).or_else(|| constant_recorded_ty(&pair.1)))
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

    // ----- transmute casts ---------------------------------------------------

    /// Convert a `CastKind::Transmute` rvalue: `operand` is the
    /// MIR operand, `v` its converted IR form, `target` the cast's
    /// target type. Only the shapes this value model can
    /// reinterpret exactly are converted:
    ///
    /// * pointer -> integer (the synthetic address) and the
    ///   reverse (an address-only reference);
    /// * thin pointer -> thin pointer, including reference ->
    ///   reference across pointee types (`&Waker` ->
    ///   `&LocalWaker`): the value is the pointer itself, the
    ///   pointee type lives in the destination's declared type;
    /// * fieldless enum <-> integer, through the baked
    ///   discriminant table;
    /// * thin pointer <-> transparent wrapper struct (one
    ///   thin-pointer payload field, the remaining fields
    ///   zero-sized with unique values): `NonNull`, `Pin` and the
    ///   coroutine machinery's `ResumeTy` chain transmute through
    ///   these;
    /// * zero-sized type -> zero-sized type whose value is
    ///   uniquely determined by its type;
    /// * scalar-shaped -> scalar-shaped of the same total width
    ///   ([`ir::Rvalue::Reinterpret`]): integers, floats, `bool`
    ///   and `char` in any combination, arrays of scalars, and
    ///   one-payload-field ADTs over those — the bit-level
    ///   reinterpretation `transmute` means there;
    /// * fat pointer <-> fat pointer, and fat pointer <->
///   transparent wrapper struct (one fat-pointer payload
    ///   field, the remaining fields zero-sized with unique
    ///   values): `NonNull<str>` and its kin;
    /// * one-field struct over a fieldless enum -> integer:
    ///   the discriminant of the field's value (`mem::Alignment`
    ///   -> `usize`).
    ///
    /// Every other shape — aggregate <-> aggregate byte
    /// reinterpretation beyond one payload field, fat pointers,
    /// enums with payloads — stays [`ir::Rvalue::Unsupported`]:
    /// the model has no flat byte memory to reinterpret through.
    fn convert_transmute(
        &self,
        body: &mir::Body,
        operand: &mir::Operand,
        v: &ir::Operand,
        target: &mir::Ty,
    ) -> ir::Rvalue {
        fn unsupported() -> ir::Rvalue {
            ir::Rvalue::Unsupported("Cast Transmute".to_string())
        }
        let src = self.operand_place_ty(body, operand);
        // A constant integer operand carries no place type; the
        // converted operand itself is the evidence for its kind.
        let src_is_int = matches!(src, Some(mir::Ty::Int(_) | mir::Ty::Uint(_)))
            || matches!(
                v,
                ir::Operand::Constant(ir::Constant::Int(_) | ir::Constant::Uint(_))
            );
        // A `ZeroSized` constant operand carries no value at
        // all: transmuting it into a zero-sized type whose value
        // is uniquely determined by the type yields that value,
        // whatever the (also zero-sized) source type was.
        if let mir::Operand::Constant(c) = operand {
            if matches!(
                c.literal,
                mir::ConstKind::Value(mir::ConstValue::ZeroSized)
            ) {
                if let Some(tv) = self.zst_value(target, 0) {
                    if let Some(k) = value_to_constant(&tv) {
                        return ir::Rvalue::Use(ir::Operand::Constant(k));
                    }
                }
            }
        }
        // The scalar-shape reinterpretation, decided from the
        // two types alone: when both sides flatten to the same
        // nonzero bit width, the cast is a `Reinterpret`. The
        // specialised arms below still take the shapes whose
        // result this does not reproduce (pointers and fieldless
        // enums carry no scalar shape, so those pairs never
        // produce a `generic`).
        let generic = {
            let to = self.scalar_shape(target, 0);
            let from = match &src {
                Some(s) => self.scalar_shape(s, 0),
                // A constant operand carries no place type; its
                // shape comes from the constant itself (see
                // `constant_scalar_shape`).
                None => to
                    .as_ref()
                    .and_then(|t| self.constant_scalar_shape(v, t.bits())),
            };
            match (from, to) {
                (Some(from), Some(to))
                    if from.bits() == to.bits() && from.bits() > 0 =>
                {
                    Some(ir::Rvalue::Reinterpret(v.clone(), from, to))
                }
                _ => None,
            }
        };
        match (&src, target) {
            // Pointer -> integer: the synthetic address (what the
            // compiler's UB precondition checks use to test a
            // pointer for null / alignment).
            (
                Some(mir::Ty::RawPtr(..) | mir::Ty::Ref(..)),
                mir::Ty::Int(_) | mir::Ty::Uint(_),
            ) => ir::Rvalue::PtrToInt(v.clone()),
            // Thin pointer -> thin pointer crossing an
            // `UnsafeCell` / `MaybeUninit` wrapper boundary in
            // the entering direction (the wrapper -> the type it
            // wraps): the pointer must come to denote the
            // wrapper's inner field, not the wrapper aggregate -
            // the same field-entering rvalue the cast conversion
            // emits for this shape (Lane DP's `PtrField`).
            (Some(s), t)
                if is_thin_ptr_ty(s)
                    && is_thin_ptr_ty(t)
                    && self.cast_enters_wrapper_field(s, t) =>
            {
                ir::Rvalue::PtrField(v.clone())
            }
            // Thin pointer -> thin pointer: the value itself.
            // The one crossing still excluded here is the
            // exiting direction (the wrapped type -> the
            // wrapper; see `pointee_crosses_cell_wrapper`): the
            // pointer would have to come to denote a wrapper
            // aggregate around the field it points into, and
            // there is no field-exiting counterpart to
            // `PtrField`, so those stay unsupported instead of
            // silently pointing at the field.
            (Some(s), t)
                if is_thin_ptr_ty(s)
                    && is_thin_ptr_ty(t)
                    && !self.pointee_crosses_cell_wrapper(s, t) =>
            {
                ir::Rvalue::Use(v.clone())
            }
            // Fat pointer -> fat pointer: the value itself,
            // exactly as for thin pointers (the metadata half
            // travels with the value; the pointee type lives in
            // the destination's declared type). The
            // UnsafeCell / MaybeUninit wrapper crossing stays
            // excluded, as for thin pointers: entering a wrapper
            // field is a place change, not a value identity.
            (Some(s), t)
                if is_fat_ptr_ty(s)
                    && is_fat_ptr_ty(t)
                    && !self.pointee_crosses_cell_wrapper(s, t) =>
            {
                ir::Rvalue::Use(v.clone())
            }
            // Integer -> thin pointer: an address-only reference.
            (_, t) if src_is_int && is_thin_ptr_ty(t) => {
                ir::Rvalue::IntToPtr(v.clone())
            }
            // Scalar-shaped -> scalar-shaped of the same width:
            // the bit-level reinterpretation (float <-> integer
            // bits, `bool` / `char` <-> integer, scalar arrays,
            // one-payload wrappers over scalars). For the one
            // overlap with the arms below (a one-field struct ->
            // the integer of exactly its field's type) the result
            // is the same value: the field itself.
            (_, _) if generic.is_some() => generic.clone().unwrap(),
            // Fieldless enum -> integer: the discriminant value.
            // Single-field struct -> integer of exactly the
            // field's type: the field itself (the struct is a
            // transparent wrapper around it).
            (Some(mir::Ty::Adt(def, args)), mir::Ty::Int(_) | mir::Ty::Uint(_)) => {
                if let Some(table) = self.fieldless_enum_discriminants(def) {
                    match self.convert_ty(target, "cast target") {
                        Ok(t) => ir::Rvalue::EnumToInt(v.clone(), table, t),
                        Err(_) => unsupported(),
                    }
                } else if let Some(field_ty) = self.single_field_ty(def, args)
                {
                    if field_ty == *target {
                        match self.field_operand(body, operand, v, 0, &field_ty) {
                            Some(op) => ir::Rvalue::Use(op),
                            None => unsupported(),
                        }
                    } else if let mir::Ty::Adt(fdef, _) = &field_ty {
                        // A transparent wrapper around a fieldless
                        // enum: the struct's value is the field's
                        // value, and the field transmutes to the
                        // integer as its discriminant (the
                        // `mem::Alignment` shape — a one-field
                        // struct over a fieldless enum whose
                        // discriminants are the alignment values,
                        // transmuted to `usize` by RawVec's
                        // allocation path).
                        match self.fieldless_enum_discriminants(fdef) {
                            Some(table) => match self
                                .field_operand(body, operand, v, 0, &field_ty)
                            {
                                Some(op) => match self
                                    .convert_ty(target, "cast target")
                                {
                                    Ok(t) => {
                                        ir::Rvalue::EnumToInt(op, table, t)
                                    }
                                    Err(_) => unsupported(),
                                },
                                None => unsupported(),
                            },
                            None => unsupported(),
                        }
                    } else {
                        unsupported()
                    }
                } else {
                    unsupported()
                }
            }
            // Integer -> fieldless enum: the variant whose
            // discriminant the integer's bit pattern equals.
            // Integer -> enum with a single variant carrying a
            // single integer field: that variant holding the
            // integer as its payload.
            (_, mir::Ty::Adt(def, args)) if src_is_int => {
                if let Some(table) = self.fieldless_enum_discriminants(def) {
                    ir::Rvalue::IntToEnum(v.clone(), table)
                } else if let Some(payload) =
                    self.single_variant_payload(def, args)
                {
                    if matches!(payload, mir::Ty::Int(_) | mir::Ty::Uint(_)) {
                        ir::Rvalue::AggregateAdt {
                            variant: 0,
                            operands: vec![v.clone()],
                            field_index: None,
                        }
                    } else {
                        unsupported()
                    }
                } else {
                    unsupported()
                }
            }
            // Pointer -> transparent wrapper struct: wrap the
            // pointer, filling the zero-sized fields with their
            // unique values. The payload may be a fat pointer
            // (`*const str` -> `NonNull<str>`), matched by the
            // fat wrapper shape.
            (Some(s), mir::Ty::Adt(def, args))
                if is_thin_ptr_ty(s) || is_fat_ptr_ty(s) =>
            {
                let shape = if is_fat_ptr_ty(s) {
                    self.fat_wrapper_shape(def, args)
                } else {
                    self.wrapper_shape(def, args)
                };
                match shape {
                    Some(shape) => {
                        let mut operands = Vec::with_capacity(shape.field_count);
                        for i in 0..shape.field_count {
                            if i == shape.payload_index {
                                operands.push(v.clone());
                            } else {
                                let c = shape
                                    .fill
                                    .iter()
                                    .find(|(j, _)| *j == i)
                                    .map(|(_, c)| c.clone());
                                match c {
                                    Some(c) => {
                                        operands.push(ir::Operand::Constant(c))
                                    }
                                    None => return unsupported(),
                                }
                            }
                        }
                        ir::Rvalue::AggregateAdt {
                            variant: 0,
                            operands,
                            field_index: None,
                        }
                    }
                    None => unsupported(),
                }
            }
            // Transparent wrapper struct -> pointer: read the
            // payload field back out. The payload may be a fat
            // pointer (`NonNull<str>` -> `*const str`, the
            // transmute `Location::file` performs), matched by
            // the fat wrapper shape.
            (Some(mir::Ty::Adt(def, args)), t)
                if is_thin_ptr_ty(t) || is_fat_ptr_ty(t) =>
            {
                let shape = if is_fat_ptr_ty(t) {
                    self.fat_wrapper_shape(def, args)
                } else {
                    self.wrapper_shape(def, args)
                };
                match shape {
                    Some(shape) => self.unwrap_wrapper(body, operand, v, &shape),
                    None => unsupported(),
                }
            }
            // Single-field struct wrapping an integer ->
            // fieldless enum: the field is the discriminant (the
            // `Bar { a }` -> `Foo` shape). Any other ADT -> ADT
            // pair falls through to the zero-sized rule below.
            (Some(mir::Ty::Adt(def, args)), mir::Ty::Adt(tdef, _)) => {
                match (
                    self.single_field_ty(def, args),
                    self.fieldless_enum_discriminants(tdef),
                ) {
                    (Some(field_ty), Some(table))
                        if matches!(
                            field_ty,
                            mir::Ty::Int(_) | mir::Ty::Uint(_)
                        ) =>
                    {
                        match self.field_operand(body, operand, v, 0, &field_ty) {
                            Some(op) => ir::Rvalue::IntToEnum(op, table),
                            None => unsupported(),
                        }
                    }
                    _ => match (self.zst_value(src.as_ref().unwrap(), 0), self.zst_value(target, 0)) {
                        (Some(_), Some(tv)) => match value_to_constant(&tv) {
                            Some(c) => {
                                ir::Rvalue::Use(ir::Operand::Constant(c))
                            }
                            None => unsupported(),
                        },
                        _ => unsupported(),
                    },
                }
            }
            // Zero-sized -> zero-sized with a uniquely determined
            // value: the value is the target type's unique value,
            // whatever the source value was (a zero-sized source
            // with a unique value carries no other information).
            (Some(s), t) => {
                match (self.zst_value(s, 0), self.zst_value(t, 0)) {
                    (Some(_), Some(tv)) => match value_to_constant(&tv) {
                        Some(c) => ir::Rvalue::Use(ir::Operand::Constant(c)),
                        None => unsupported(),
                    },
                    _ => unsupported(),
                }
            }
            _ => unsupported(),
        }
    }

    /// Read the payload field out of a transparent wrapper value
    /// (the reverse of the wrap in [`Self::convert_transmute`]).
    fn unwrap_wrapper(
        &self,
        body: &mir::Body,
        operand: &mir::Operand,
        v: &ir::Operand,
        shape: &WrapperShape,
    ) -> ir::Rvalue {
        match self.field_operand(
            body,
            operand,
            v,
            shape.payload_index as u32,
            &shape.payload_ty,
        ) {
            Some(op) => ir::Rvalue::Use(op),
            None => ir::Rvalue::Unsupported("Cast Transmute".to_string()),
        }
    }

    /// The operand denoting field `index` of the value `operand`
    /// denotes: for a place operand, the place extended with the
    /// field projection (preserving move-ness); for a constant
    /// ADT operand, the field constant. `None` for operand forms
    /// that carry no addressable field.
    fn field_operand(
        &self,
        body: &mir::Body,
        operand: &mir::Operand,
        v: &ir::Operand,
        index: u32,
        field_ty: &mir::Ty,
    ) -> Option<ir::Operand> {
        match operand.place() {
            Some(place) => {
                let place = place.clone().field(index, field_ty.clone());
                let p = self.convert_place(body, &place).ok()?;
                Some(match operand {
                    mir::Operand::Move(_) => ir::Operand::Move(p),
                    _ => ir::Operand::Copy(p),
                })
            }
            None => match v {
                ir::Operand::Constant(ir::Constant::Adt { fields, .. }) => fields
                    .get(index as usize)
                    .map(|c| ir::Operand::Constant(c.clone())),
                _ => None,
            },
        }
    }

    /// Whether a thin-pointer -> thin-pointer transmute crosses
    /// an `UnsafeCell` / `MaybeUninit` wrapper boundary: one
    /// side's pointee is the wrapper ADT and the other side's
    /// pointee is exactly the type the wrapper wraps. The
    /// entering direction (wrapper -> wrapped type) is composed
    /// with the cast conversion's field-entering treatment
    /// (`cast_enters_wrapper_field` -> `Rvalue::PtrField`) in
    /// `convert_transmute` before this guard is consulted, so
    /// the crossings this guard still excludes are the exiting
    /// direction (wrapped type -> wrapper), which has no
    /// field-exiting rvalue and stays explicit unsupported
    /// instead of silently pointing at the field.
    fn pointee_crosses_cell_wrapper(
        &self,
        src: &mir::Ty,
        target: &mir::Ty,
    ) -> bool {
        fn pointee(ty: &mir::Ty) -> Option<&mir::Ty> {
            match ty {
                mir::Ty::RawPtr(inner, _) | mir::Ty::Ref(_, inner, _) => {
                    Some(inner)
                }
                _ => None,
            }
        }
        let (Some(sp), Some(tp)) = (pointee(src), pointee(target)) else {
            return false;
        };
        let wrapped = |wrapper: &mir::Ty, plain: &mir::Ty| -> bool {
            match wrapper {
                mir::Ty::Adt(def, args) => {
                    args.len() == 1
                        && args[0] == *plain
                        && matches!(
                            self.def_display_name(def).as_str(),
                            "core::cell::UnsafeCell"
                                | "std::cell::UnsafeCell"
                                | "core::mem::MaybeUninit"
                                | "std::mem::MaybeUninit"
                        )
                }
                _ => false,
            }
        };
        wrapped(sp, tp) || wrapped(tp, sp)
    }

    /// The instantiated type of the only field of a struct with
    /// exactly one field, or `None` for any other ADT.
    fn single_field_ty(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
    ) -> Option<mir::Ty> {
        let shape = self.adt_shapes.get(def)?;
        if shape.kind != mir::AdtShapeKind::Struct || shape.variants.len() != 1
        {
            return None;
        }
        let fields = self.instantiated_field_tys(def, args, 0)?;
        if fields.len() == 1 {
            fields.into_iter().next()
        } else {
            None
        }
    }

    /// The instantiated type of the only field of an enum with
    /// exactly one variant carrying exactly one field, or `None`
    /// for any other ADT.
    fn single_variant_payload(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
    ) -> Option<mir::Ty> {
        let shape = self.adt_shapes.get(def)?;
        if shape.kind != mir::AdtShapeKind::Enum || shape.variants.len() != 1
        {
            return None;
        }
        let fields = self.instantiated_field_tys(def, args, 0)?;
        if fields.len() == 1 {
            fields.into_iter().next()
        } else {
            None
        }
    }

    /// The effective discriminant table of an enum whose
    /// variants are all fieldless (variant index -> discriminant
    /// value), or `None` when `def` is not such an enum. A missing
    /// recorded table means the discriminants are the sequential
    /// defaults, so the table is synthesized from the variant
    /// count — the same convention as the `Discriminant` rvalue.
    fn fieldless_enum_discriminants(
        &self,
        def: &mir::AdtDefId,
    ) -> Option<Vec<i128>> {
        let shape = self.adt_shapes.get(def)?;
        if shape.kind != mir::AdtShapeKind::Enum
            || shape.variants.iter().any(|fields| !fields.is_empty())
        {
            return None;
        }
        match self.discriminants.get(def) {
            Some(table) if !table.is_empty() => Some(table.clone()),
            _ => Some((0..shape.variants.len() as i128).collect()),
        }
    }

    /// The [`ir::ScalarShape`] of a type whose every bit is
    /// accounted for by scalar leaves (integers, floats, `bool`,
    /// `char`, arrays of those, and one-payload-field ADTs over
    /// those), or `None` for any other type. `depth` bounds the
    /// ADT nesting: a sized type cannot actually recurse without
    /// indirection (which has no scalar shape and ends the chain
    /// first), so the bound is only belt and braces.
    fn scalar_shape(
        &self,
        ty: &mir::Ty,
        depth: u32,
    ) -> Option<ir::ScalarShape> {
        use ir::ScalarShape as S;
        if depth > 8 {
            return None;
        }
        match ty {
            mir::Ty::Bool => Some(S::Bool),
            mir::Ty::Char => Some(S::Char),
            mir::Ty::Int(it) => Some(S::Int {
                bits: it.bits(),
                signed: true,
            }),
            mir::Ty::Uint(ut) => Some(S::Int {
                bits: ut.bits(),
                signed: false,
            }),
            mir::Ty::Float(ft) => match ft.bits() {
                16 | 32 | 64 => Some(S::Float { bits: ft.bits() }),
                _ => None,
            },
            mir::Ty::Array(elem, mir::ArrayLen::Known(n)) => {
                let elem = self.scalar_shape(elem, depth + 1)?;
                let total = elem.bits().checked_mul(*n as u32)?;
                if total == 0 || total > 128 {
                    return None;
                }
                Some(S::Array(Box::new(elem), *n))
            }
            mir::Ty::Adt(def, args) => {
                let shape = self.adt_shapes.get(def)?;
                if shape.variants.len() != 1 {
                    return None;
                }
                let fields = self.instantiated_field_tys(def, args, 0)?;
                let mut payload: Option<(u32, ir::ScalarShape)> = None;
                let mut fill = Vec::new();
                for (i, fty) in fields.iter().enumerate() {
                    match self.scalar_shape(fty, depth + 1) {
                        Some(fs) if fs.bits() > 0 => {
                            if payload.is_some() {
                                return None;
                            }
                            payload = Some((i as u32, fs));
                        }
                        _ => {
                            let zv = self.zst_value(fty, 0)?;
                            fill.push((i as u32, value_to_constant(&zv)?));
                        }
                    }
                }
                let (index, inner) = payload?;
                Some(S::AdtField {
                    index,
                    count: fields.len(),
                    fill,
                    inner: Box::new(inner),
                })
            }
            _ => None,
        }
    }

    /// The scalar shape of a constant operand, whose MIR form
    /// carries no type: the constant's own kind gives the leaf
    /// kind, and the width is the transmute target's width (a
    /// transmute's two sides agree on size, so the source's
    /// width is the target's). A tuple constant is read as an
    /// array of its first element's kind; a bytes constant as an
    /// array of `u8`. Anything else has no derivable shape.
    fn constant_scalar_shape(
        &self,
        v: &ir::Operand,
        width: u32,
    ) -> Option<ir::ScalarShape> {
        use ir::ScalarShape as S;
        let ir::Operand::Constant(c) = v else {
            return None;
        };
        match c {
            ir::Constant::Int(_) => Some(S::Int {
                bits: width,
                signed: true,
            }),
            ir::Constant::Uint(_) => Some(S::Int {
                bits: width,
                signed: false,
            }),
            ir::Constant::Float(_) => match width {
                16 | 32 | 64 => Some(S::Float { bits: width }),
                _ => None,
            },
            ir::Constant::Bool(_) => (width == 8).then_some(S::Bool),
            ir::Constant::Bytes(b) => {
                (width == 8 * b.len() as u32).then(|| {
                    S::Array(
                        Box::new(S::Int {
                            bits: 8,
                            signed: false,
                        }),
                        b.len() as u64,
                    )
                })
            }
            ir::Constant::Tuple(fields) if !fields.is_empty() => {
                let n = fields.len() as u32;
                if width == 0 || width % n != 0 {
                    return None;
                }
                let ew = width / n;
                let elem = match &fields[0] {
                    ir::Constant::Int(_) => S::Int {
                        bits: ew,
                        signed: true,
                    },
                    ir::Constant::Uint(_) => S::Int {
                        bits: ew,
                        signed: false,
                    },
                    ir::Constant::Float(_) => match ew {
                        16 | 32 | 64 => S::Float { bits: ew },
                        _ => return None,
                    },
                    ir::Constant::Bool(_) => {
                        if ew != 8 {
                            return None;
                        }
                        S::Bool
                    }
                    _ => return None,
                };
                let same = fields.iter().all(|f| {
                    matches!(
                        (f, &fields[0]),
                        (ir::Constant::Int(_), ir::Constant::Int(_))
                            | (
                                ir::Constant::Uint(_),
                                ir::Constant::Uint(_)
                            )
                            | (
                                ir::Constant::Float(_),
                                ir::Constant::Float(_)
                            )
                            | (ir::Constant::Bool(_), ir::Constant::Bool(_))
                    )
                });
                if same {
                    Some(S::Array(Box::new(elem), u64::from(n)))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// The [`WrapperShape`] of the struct `def` instantiated
    /// with `args`, or `None` when it is not a transparent
    /// wrapper (not a struct, more than one payload field, or a
    /// field that is neither the payload nor a uniquely valued
    /// zero-sized type).
    fn wrapper_shape(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
    ) -> Option<WrapperShape> {
        self.wrapper_shape_matching(def, args, is_thin_ptr_ty)
    }

    /// The wrapper shape whose payload field is a fat pointer
    /// (`NonNull<str>`, `NonNull<[T]>`, `NonNull<dyn Trait>`):
    /// the same transparent-wrapper rule as
    /// [`Self::wrapper_shape`], with the metadata-carrying
    /// pointer as the payload.
    fn fat_wrapper_shape(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
    ) -> Option<WrapperShape> {
        self.wrapper_shape_matching(def, args, is_fat_ptr_ty)
    }

    fn wrapper_shape_matching(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
        payload_ok: fn(&mir::Ty) -> bool,
    ) -> Option<WrapperShape> {
        let shape = self.adt_shapes.get(def)?;
        if shape.kind != mir::AdtShapeKind::Struct || shape.variants.len() != 1 {
            return None;
        }
        let fields = self.instantiated_field_tys(def, args, 0)?;
        let mut payload: Option<(usize, mir::Ty)> = None;
        let mut fill = Vec::new();
        for (i, fty) in fields.iter().enumerate() {
            if payload_ok(fty) {
                if payload.is_some() {
                    return None;
                }
                payload = Some((i, fty.clone()));
            } else if let Some(zv) = self.zst_value(fty, 0) {
                fill.push((i, value_to_constant(&zv)?));
            } else {
                return None;
            }
        }
        let (payload_index, payload_ty) = payload?;
        Some(WrapperShape {
            field_count: fields.len(),
            payload_index,
            payload_ty,
            fill,
        })
    }

    /// The field types of one variant of `def` instantiated with
    /// `args`: recovered field types (already instantiated for
    /// this `args`) win over the shape's declared types, whose
    /// parameter indices can misalign — the same resolution as
    /// `unique_value`.
    fn instantiated_field_tys(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
        variant: u32,
    ) -> Option<Vec<mir::Ty>> {
        let shape = self.adt_shapes.get(def)?;
        let declared = shape.variants.get(variant as usize)?;
        Some(
            declared
                .iter()
                .enumerate()
                .map(|(fi, fty)| {
                    self.adt_fields
                        .get(*def, args, variant, fi as u32)
                        .cloned()
                        .unwrap_or_else(|| subst_ty(fty, args))
                })
                .collect(),
        )
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
                                                glue: None,
                                            },
                                        )
                                    }
                                    Some(src) => {
                                        let pointee = unsize_pointee(&src);
                                        let concrete = self.concrete_ty_name(pointee);
                                        // The value carries the
                                        // pointee's drop glue from
                                        // here, like a vtable's
                                        // drop shim (see
                                        // `UnsizeMeta::Dyn`).
                                        let glue = DropGlueBuilder::build(self, pointee);
                                        ir::Rvalue::Unsize(
                                            v,
                                            ir::UnsizeMeta::Dyn {
                                                concrete: Some(concrete),
                                                traits,
                                                glue,
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
                    mir::CastKind::PtrToPtr => {
                        // A cast from a transparent single-field
                        // wrapper (`UnsafeCell<T>` / `MaybeUninit<T>`)
                        // to the wrapped type changes what the
                        // pointer denotes: the wrapper's only field.
                        // Every other pointer-to-pointer cast leaves
                        // the denoted place unchanged in this model.
                        if ctx
                            .as_ref()
                            .is_some_and(|src| self.cast_enters_wrapper_field(src, target))
                        {
                            ir::Rvalue::PtrField(v)
                        } else {
                            ir::Rvalue::Use(v)
                        }
                    }
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
                    // PointerWithExposedProvenance) is the next
                    // arm below. (Lane BV)
                    mir::CastKind::PointerExposeProvenance => match (&ctx, target) {
                        (
                            Some(
                                mir::Ty::RawPtr(..)
                                | mir::Ty::Ref(..)
                                | mir::Ty::FnPtr(..)
                                | mir::Ty::FnDef(..),
                            ),
                            mir::Ty::Int(_) | mir::Ty::Uint(_),
                        ) => ir::Rvalue::PtrToInt(v.clone()),
                        _ => ir::Rvalue::Unsupported(
                            "Cast PointerExposeProvenance".to_string(),
                        ),
                    },
                    // The reverse provenance cast (an integer
                    // address back to a pointer,
                    // `with_exposed_provenance` / `addr as *mut T`):
                    // the same address-only reference the
                    // integer -> pointer transmute produces. It
                    // compares, tests for null and casts back to
                    // an integer; reading or writing through it is
                    // an explicit error, as for any address-only
                    // pointer in this model.
                    mir::CastKind::PointerWithExposedProvenance => {
                        let src_is_int = matches!(
                            ctx,
                            Some(mir::Ty::Int(_) | mir::Ty::Uint(_))
                        ) || matches!(
                            v,
                            ir::Operand::Constant(
                                ir::Constant::Int(_) | ir::Constant::Uint(_)
                            )
                        );
                        if src_is_int && is_thin_ptr_ty(target) {
                            ir::Rvalue::IntToPtr(v.clone())
                        } else {
                            ir::Rvalue::Unsupported(
                                "Cast PointerWithExposedProvenance"
                                    .to_string(),
                            )
                        }
                    }
                    // A subtyping cast changes only lifetimes,
                    // which this model does not track (a reference
                    // value carries no lifetime): the value itself
                    // is unchanged.
                    mir::CastKind::Subtype => ir::Rvalue::Use(v),
                    // `transmute`: the shapes the value model
                    // can reinterpret exactly (pointers, integers,
                    // fieldless enums, transparent wrappers,
                    // zero-sized types) are decided in
                    // `convert_transmute`; every other shape
                    // stays an explicit unsupported.
                    mir::CastKind::Transmute => {
                        self.convert_transmute(body, operand, &v, target)
                    }
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
                    field_index,
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
                        // A union aggregate carries a single operand
                        // for its active field (`field_index`), so
                        // that field's declared type is the context,
                        // not field 0's.
                        let field = match field_index {
                            Some(active) => *active,
                            None => i as u32,
                        };
                        let declared = self.adt_fields.get(
                            *def,
                            generic_args,
                            *variant_index,
                            field,
                        );
                        let ctx = declared.or_else(|| if i < 2 { bound_ty } else { None });
                        fields.push(self.convert_operand(body, op, ctx)?);
                    }
                    ir::Rvalue::AggregateAdt {
                        variant: *variant_index,
                        operands: fields,
                        field_index: *field_index,
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
                    // A repeat evaluates its operand once and
                    // replicates the value; the operand is written
                    // as a move in MIR even for `Copy` element
                    // types (an array of `Copy` elements is itself
                    // `Copy`). Replicating a move operand would
                    // read the source place once per element and
                    // fail on the second read, the source having
                    // been moved out of. Read it as a copy instead:
                    // the value is replicated, not moved per
                    // element.
                    let v = match v {
                        ir::Operand::Move(place) => ir::Operand::Copy(place),
                        other => other,
                    };
                    ir::Rvalue::AggregateTuple(vec![v; *count as usize])
                }
            }
            R::Ref(_, _, place) => ir::Rvalue::Ref(self.convert_place(body, place)?),
            R::CopyForDeref(place) => ir::Rvalue::CopyForDeref(self.convert_place(body, place)?),
            // Raw pointer creation shares the reference representation.
            R::RawPtr(_, place) => ir::Rvalue::Ref(self.convert_place(body, place)?),
            R::Reborrow(_, _, place) => ir::Rvalue::Reborrow(self.convert_place(body, place)?),
            R::ThreadLocalRef(def) => ir::Rvalue::ThreadLocalRef(format!("{def}")),
            R::Len(place) => ir::Rvalue::Len(self.convert_place(body, place)?),
            R::NullaryOp(op, _) => ir::Rvalue::Unsupported(format!("NullaryOp {}", op.kind_name())),
            R::Discriminant(place) => {
                let converted = self.convert_place(body, place)?;
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
                let ir_place = self.convert_place(body, place)?;
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
                    place: self.convert_place(body, place)?,
                    variant: *variant_index,
                },
            },
            S::Intrinsic(intr) => ir::Statement {
                kind: match &**intr {
                    mir::NonDivergingIntrinsic::Assume(op) => {
                        ir::StatementKind::Assume(self.convert_operand(body, op, None)?)
                    }
                    mir::NonDivergingIntrinsic::CopyNonOverlapping { src, dst, count } => {
                        ir::StatementKind::CopyNonOverlapping {
                            src: self.convert_operand(body, src, None)?,
                            dst: self.convert_operand(body, dst, None)?,
                            count: self.convert_operand(body, count, None)?,
                        }
                    }
                },
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

    /// Resolve a direct call to the program body that serves it,
    /// if any (the caller falls through to the `Fn`-trait and
    /// external/builtin shapes when this returns `None`).
    ///
    /// The ordinary case is `resolve_callee`: the collected body
    /// the operand names. When that body failed conversion in the
    /// previous fixpoint round, calling it can only fail with its
    /// recorded reason, so the call is served another way when it
    /// honestly can be:
    ///
    /// * If the call names that very item — the shape of a trait
    ///   call whose collected body is the trait item's default
    ///   implementation — a clean impl body for the call's `Self`
    ///   type is the faithful target (it is what native execution
    ///   dispatches to), exactly as when the trait item has no
    ///   collected body at all.
    /// * Otherwise, when the builtin model serves the callee, the
    ///   call falls through to the external/builtin conversion
    ///   below — the same treatment a callee the driver never
    ///   collected receives. The interpreter's builtin is the
    ///   model of that very function, so the value computed is
    ///   the function's own semantics, not an approximation of a
    ///   different one.
    ///
    /// When neither applies, the unconverted body still serves
    /// the call, so running it raises the recorded conversion
    /// failure (and a `dyn` receiver still dispatches through the
    /// vtable at run time): an explicit error, never a silently
    /// different result.
    fn resolve_body_callee(&self, func: &mir::Operand) -> Option<String> {
        let Some(name) = self.resolve_callee(func) else {
            return self.resolve_impl_callee(func);
        };
        let Some(plain) = self.unconverted.get(&name) else {
            return Some(name);
        };
        if self
            .resolve_external_callee(func)
            .is_some_and(|(item, _, _)| item == *plain)
        {
            if let Some(key) = self.resolve_impl_callee(func) {
                return Some(key);
            }
        }
        if self.external_builtin(func) {
            return None;
        }
        Some(name)
    }

    /// Whether the external/builtin conversion below would serve
    /// this call with a builtin: the callee's canonical name has a
    /// builtin model. Checked exactly as that conversion checks
    /// it, so a call is only ever diverted to a builtin that the
    /// external path would really produce.
    fn external_builtin(&self, func: &mir::Operand) -> bool {
        self.resolve_external_callee(func).is_some_and(|(name, _, _)| {
            crate::builtins::lookup(&crate::builtins::canonical_name(&name)).is_some()
        })
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
        // Resolution runs only in the joint-fixpoint rounds of
        // `convert_all`, against the previous round's clean
        // bodies (see `clean_targets`): resolving to an impl body that cannot
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
        let mut found: Option<&String> = None;
        for key in self.bodies.keys() {
            let base = key.split('#').next().unwrap_or(key.as_str());
            let Some((self_part, trait_base, key_method)) = parse_impl_key(base) else {
                continue;
            };
            if key_method != method || trait_base != trait_path {
                continue;
            }
            if !impl_self_matches(&self_part, self_name) {
                continue;
            }
            if found.is_some() {
                return None;
            }
            found = Some(key);
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

    /// A direct `Add::add` / `Sub::sub` call whose Self type is
    /// one of the modelled time types converts to a builtin node:
    /// the `Instant` / `SystemTime` instances have no body at
    /// all, and the collected `Duration` instance body does not
    /// run in this model (it writes a raw nanos count into the
    /// niche-typed `Nanoseconds` field). Every other `Add` /
    /// `Sub` call keeps the normal paths (a collected body wins;
    /// the names are not in the builtin registry, so reach still
    /// collects the generic shim bodies).
    fn time_add_sub_terminator(
        &self,
        body: &mir::Body,
        func: &mir::Operand,
        args: &[mir::Operand],
        destination: &mir::Place,
        target: Option<mir::BasicBlock>,
    ) -> Result<Option<ir::TerminatorKind>, ConvertError> {
        let Some((name, generic_args, _raw)) = self.resolve_external_callee(func) else {
            return Ok(None);
        };
        let canonical = crate::builtins::canonical_name(&name);
        if canonical != "std::ops::Add::add" && canonical != "std::ops::Sub::sub" {
            return Ok(None);
        }
        let Some(self_ty) = generic_args.first() else {
            return Ok(None);
        };
        if !matches!(
            self_ty.as_str(),
            "std::time::Instant" | "std::time::SystemTime" | "std::time::Duration"
        ) {
            return Ok(None);
        }
        let mut ir_args = Vec::with_capacity(args.len());
        for arg in args.iter() {
            ir_args.push(self.convert_operand(body, arg, None)?);
        }
        Ok(Some(ir::TerminatorKind::CallBuiltin {
            name: canonical,
            generic_args: vec![self_ty.clone()],
            args: ir_args,
            destination: self.convert_place(body, destination)?,
            target: target.map(|b| ir::BasicBlock(b.index())),
            drop_glue: None,
            cmp_discriminants: None,
            result_unsize: None,
        }))
    }

    /// Whether this call's func operand names an `Fn`-family trait
    /// method (`Fn::call`, `FnMut::call_mut`, `FnOnce::call_once` in
    /// their `std::ops` / `core::ops` spellings): returns the call
    /// kind and the callee's raw generic arguments, whose first two
    /// entries are the Self type and the argument tuple type (used
    /// to type the receiver and tuple operands).
    /// How a non-indirect call is dispatched when both the `Fn`
    /// trait interpretation and name-based resolution are in play.
    ///
    /// The direct candidate is `resolve_body_callee`'s, so the
    /// joint-fixpoint honesty rules (a clean impl body serves a call
    /// naming an unconverted trait item; a builtin-modelled callee
    /// diverts to the builtin path) apply before the shape judgment
    /// below. Name-based resolution wins when it names a genuine body
    /// for the call: an inherent `Fn` implementation on a struct value,
    /// or the trait method's own shim (environment plus the whole
    /// argument tuple) — both take exactly the call's arguments.
    /// The trait interpretation wins when there is no direct body,
    /// and when the body recorded under the trait name is a
    /// closure's own body collected through the trait-call instance
    /// (see [`body_is_closure_body`]): a direct call would pass
    /// (receiver, tuple) to a body of the wrong arity and
    /// convention, while the trait call evaluates the receiver as a
    /// callable value, which is the convention that body expects.
    /// Convert a direct `Drop::drop` call into a `Drop` terminator
    /// against the receiver's pointee place, when the call's shape
    /// allows it (see the call conversion): the callee names the
    /// `Drop` trait item (`std::ops::Drop::drop` / `core::ops::
    /// Drop::drop`) or a `Drop` impl item (`<T as std::ops::Drop>::
    /// drop`), there is exactly one argument (the `&mut Self`
    /// receiver), and the dropped type's glue can be built. The
    /// receiver place plus a `Deref` projection is the pointee
    /// place the glue runs against; running it clears that place
    /// (the glue machinery clears a direct local), so a following
    /// `Drop` terminator for the same place reads it as
    /// uninitialised and runs no glue a second time.
    ///
    /// An impl-item call whose own converted body exists keeps
    /// that body: the inherent-body priority rule — a collected
    /// impl body is the faithful semantics of the call — wins over
    /// the glue conversion. A call whose glue cannot be built
    /// falls through to the ordinary call conversion.
    fn drop_call_terminator(
        &self,
        body: &mir::Body,
        func: &mir::Operand,
        args: &[mir::Operand],
        target: Option<mir::BasicBlock>,
    ) -> Option<ir::TerminatorKind> {
        if args.len() != 1 {
            return None;
        }
        let target = target?;
        let (name, _, raw_generic_args) = self.resolve_external_callee(func)?;
        let is_trait_item = matches!(
            name.as_str(),
            "std::ops::Drop::drop" | "core::ops::Drop::drop"
        );
        let is_impl_item = name.starts_with('<')
            && (name.ends_with(" as std::ops::Drop>::drop")
                || name.ends_with(" as core::ops::Drop>::drop"));
        if !is_trait_item && !is_impl_item {
            return None;
        }
        if is_impl_item && self.resolve_body_callee(func).is_some() {
            return None;
        }
        // Self type: the receiver argument's pointee type (the
        // argument is `&mut Self`), falling back to the call's
        // first generic argument.
        let self_ty = self
            .operand_place_ty(body, &args[0])
            .map(|t| match t {
                mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => *inner,
                other => other,
            })
            .or_else(|| raw_generic_args.first().cloned())?;
        let glue = self.drop_glue(&self_ty)?;
        let arg_place = match &args[0] {
            mir::Operand::Copy(place) | mir::Operand::Move(place) => place.clone(),
            _ => return None,
        };
        let mut pointee = arg_place;
        pointee.projection.push(mir::PlaceElem::Deref);
        Some(ir::TerminatorKind::Drop {
            place: self.convert_place(body, &pointee).ok()?,
            target: ir::BasicBlock(target.index()),
            glue,
        })
    }

    fn fn_trait_or_direct(
        &self,
        func: &mir::Operand,
        arg_count: usize,
    ) -> Option<FnResolution> {
        let trait_call = self
            .resolve_fn_trait_call(func)
            .filter(|_| arg_count == 2);
        let direct = self.resolve_body_callee(func);
        match (trait_call, direct) {
            (Some(trait_call), Some(name)) => {
                let mis_keyed = self
                    .bodies
                    .get(&name)
                    .is_some_and(|b| body_is_closure_body(b));
                if mis_keyed {
                    Some(FnResolution::Trait(trait_call.0, trait_call.1))
                } else {
                    Some(FnResolution::Direct(name))
                }
            }
            (Some(trait_call), None) => {
                Some(FnResolution::Trait(trait_call.0, trait_call.1))
            }
            (None, Some(name)) => Some(FnResolution::Direct(name)),
            (None, None) => None,
        }
    }

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

    /// The unsizing a `From` / `Into` conversion into a
    /// `Box<dyn Trait>` applies to its result, decided here where
    /// both types are known — exactly as for a sized-to-dyn
    /// coercion cast: the destination type names the bounds, the
    /// argument's type names the concrete pointee, and the
    /// pointee's drop glue is built now so the resulting `DynRef`
    /// carries it like a vtable's drop shim. The modelled `From`
    /// impls for a dyn box take the payload itself (or a reference
    /// to it), never an already-boxed value, so the pointee is
    /// the argument's type with any reference stripped. `None`
    /// when the destination is not a dyn box, when the argument's
    /// type is not preserved, or when the argument is itself a
    /// dyn value (no modelled conversion starts from one).
    fn from_result_unsize(
        &self,
        body: &mir::Body,
        destination: &mir::Place,
        arg: Option<&mir::Operand>,
        source_ty: Option<&mir::Ty>,
    ) -> Option<ir::UnsizeMeta> {
        let dest_ty = self.place_ty(body, destination)?;
        let mir::Ty::Adt(def, dest_args) = &dest_ty else {
            return None;
        };
        let name = self.names.get(def).or_else(|| self.def_names.get(def))?;
        if name != "std::boxed::Box" {
            return None;
        }
        let traits = dyn_traits_in(dest_args.first()?)?;
        // A zero-sized argument travels as a constant operand,
        // whose type the operand itself does not preserve; the
        // callee's generic arguments still name the source type
        // (the caller picks it out, as `mem_drop_glue` does for
        // the dropped type).
        let arg_ty = self
            .operand_place_ty(body, arg?)
            .or_else(|| source_ty.cloned())?;
        if dyn_traits_in(&arg_ty).is_some() {
            return None;
        }
        let pointee = match &arg_ty {
            mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => inner,
            other => other,
        };
        let concrete = self.concrete_ty_name(pointee);
        let glue = DropGlueBuilder::build(self, pointee);
        Some(ir::UnsizeMeta::Dyn {
            concrete: Some(concrete),
            traits,
            glue,
        })
    }

    /// The drop glue of a `std::mem::drop` call's argument, mirroring
    /// the `Drop` terminator's treatment of the same type:
    /// `Ok(Some(glue))` when the argument's glue can be accounted
    /// for and the interpreter has something to run it against —
    /// a place argument (run against the place), or a constant
    /// argument (run against the constant's value; a zero-sized
    /// constant carries no value, so the dropped type must
    /// determine one, and the operand is rewritten to carry it) —
    /// `Ok(None)` when there is nothing to run (the glue is
    /// trivial, so discarding the value is already faithful), and
    /// `Err(())` when the glue cannot be accounted for.
    fn mem_drop_glue(
        &self,
        body: &mir::Body,
        arg: &mir::Operand,
        ir_arg: &mut ir::Operand,
        raw_generic_args: &[mir::Ty],
    ) -> Result<Option<ir::DropGlue>, ()> {
        let dropped_ty = self
            .operand_place_ty(body, arg)
            .or_else(|| raw_generic_args.first().cloned());
        let glue = dropped_ty
            .as_ref()
            .and_then(|t| self.drop_glue(t))
            .ok_or(())?;
        if glue.root == ir::DropPlan::Trivial {
            return Ok(None);
        }
        match ir_arg {
            ir::Operand::Copy(_) | ir::Operand::Move(_) => Ok(Some(glue)),
            ir::Operand::Constant(ir::Constant::Unit) => {
                // A zero-sized constant: the value being dropped is
                // not carried by the operand. When the dropped type
                // determines it, rewrite the operand to carry that
                // value so the interpreter can run the glue against
                // it; otherwise the drop stays unaccounted for.
                let ty = dropped_ty.ok_or(())?;
                let value = self.zst_value(&ty, 0).ok_or(())?;
                let constant = value_as_constant(&value).ok_or(())?;
                *ir_arg = ir::Operand::Constant(constant);
                Ok(Some(glue))
            }
            ir::Operand::Constant(_) => Ok(Some(glue)),
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

    /// Convert the operands of an evaluable inline-asm block into
    /// register slots: inputs are evaluated at block entry, outputs
    /// written back at block exit. Only called for blocks the
    /// classifier accepted, so every operand is in the register
    /// class with a plain integer type.
    fn convert_asm_regs(
        &self,
        body: &mir::Body,
        operands: &[mir::InlineAsmOperand],
    ) -> Result<Vec<ir::AsmReg>, ConvertError> {
        let mut regs = Vec::with_capacity(operands.len());
        for op in operands {
            let init = match op.kind {
                mir::AsmOperandKind::Const => Some(ir::Operand::Constant(ir::Constant::Uint(
                    u128::from(op.const_bits.unwrap_or(0)),
                ))),
                mir::AsmOperandKind::In | mir::AsmOperandKind::InOut => {
                    let value = op
                        .in_value
                        .as_ref()
                        .expect("the driver sets in_value for in operands");
                    let ctx = self.operand_place_ty(body, value);
                    Some(self.convert_operand(body, value, ctx.as_ref())?)
                }
                mir::AsmOperandKind::Out | mir::AsmOperandKind::Sym => None,
            };
            let dest = match op.kind {
                mir::AsmOperandKind::Out | mir::AsmOperandKind::InOut => {
                    let place = op
                        .out_place
                        .as_ref()
                        .expect("the driver sets out_place for out operands");
                    Some(self.convert_place(place)?)
                }
                _ => None,
            };
            // The operand type: an inout operand's place type
            // (its in_value is the same place), else the in
            // operand's place type, else the out place's.
            let reg_ty = match op.kind {
                mir::AsmOperandKind::InOut | mir::AsmOperandKind::Out => op
                    .out_place
                    .as_ref()
                    .and_then(|p| self.place_ty(body, p))
                    .or_else(|| {
                        op.in_value
                            .as_ref()
                            .and_then(|v| self.operand_place_ty(body, v))
                    }),
                mir::AsmOperandKind::In => op
                    .in_value
                    .as_ref()
                    .and_then(|v| self.operand_place_ty(body, v)),
                _ => None,
            };
            let kind = match reg_ty {
                Some(mir::Ty::Int(_)) => ir::AsmRegKind::Signed,
                Some(mir::Ty::Bool) => ir::AsmRegKind::Bool,
                _ => ir::AsmRegKind::Unsigned,
            };
            regs.push(ir::AsmReg {
                init,
                dest,
                width: op.width,
                kind,
            });
        }
        Ok(regs)
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
                if let Some(drop_term) =
                    self.drop_call_terminator(body, func, args, *target)
                {
                    drop_term
                } else if indirect {
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
                        destination: self.convert_place(body, destination)?,
                        target: target.map(|b| ir::BasicBlock(b.index())),
                    }
                } else if let Some(term) = self.time_add_sub_terminator(
                    body,
                    func,
                    args,
                    destination,
                    *target,
                )? {
                    term
                } else if let Some(name) = self.fn_trait_or_direct(func, args.len()) {
                    match name {
                        FnResolution::Trait(kind, raw_generic_args) => {
                            let receiver = self.convert_operand(
                                body,
                                &args[0],
                                raw_generic_args.first(),
                            )?;
                            let tuple = self
                                .convert_operand(body, &args[1], raw_generic_args.get(1))?;
                            ir::TerminatorKind::CallTrait {
                                kind,
                                receiver,
                                args: tuple,
                                destination: self.convert_place(body, destination)?,
                                target: target.map(|b| ir::BasicBlock(b.index())),
                            }
                        }
                        FnResolution::Direct(name) => {
                            let callee = self.bodies.get(&name);
                            // The receiver (argument 0) of a call that
                            // also resolves as an `Fn`-trait call is
                            // typed by the call's own `Self` type, not
                            // by the callee body's declared receiver
                            // type: the body is the generic impl
                            // definition (its receiver mentions the
                            // definition's parameters), while the call
                            // names the concrete `Self` — a zero-sized
                            // constant receiver only materialises
                            // under the concrete type (see
                            // `convert_const_value`).
                            let self_ctx = self
                                .resolve_fn_trait_call(func)
                                .and_then(|(_, raw)| raw.first().cloned());
                            let mut ir_args = Vec::with_capacity(args.len());
                            for (i, arg) in args.iter().enumerate() {
                                let ctx = if i == 0 {
                                    self_ctx.clone().or_else(|| {
                                        callee.and_then(|b| {
                                            b.local_decls
                                                .get(i + 1)
                                                .map(|d| d.ty.clone())
                                        })
                                    })
                                } else {
                                    callee.and_then(|b| {
                                        b.local_decls.get(i + 1).map(|d| d.ty.clone())
                                    })
                                };
                                ir_args.push(self.convert_operand(body, arg, ctx.as_ref())?);
                            }
                            ir::TerminatorKind::Call {
                                func: name,
                                args: ir_args,
                                destination: self.convert_place(body, destination)?,
                                target: target.map(|b| ir::BasicBlock(b.index())),
                            }
                        }
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
                                &mut ir_args[0],
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
                        // The `PartialOrd` / `Ord` comparison
                        // builtins compare values structurally; for
                        // an enum `Self` type that comparison must
                        // order by discriminant value, so bake the
                        // type's table while it is at hand (the
                        // `Discriminant` rvalue does the same).
                        let cmp_discriminants = if matches!(
                            kind,
                            crate::builtins::Builtin::PartialCmp
                                | crate::builtins::Builtin::OrdCmp
                                | crate::builtins::Builtin::PartialOrdRel(_)
                        ) {
                            self.comparison_discriminants(body, args, &raw_generic_args)
                        } else {
                            None
                        };
                        // A `From` / `Into` conversion into a
                        // `Box<dyn Trait>`: the modelled conversion
                        // returns the plain box aggregate, so bake
                        // the unsizing the destination type implies
                        // (see `from_result_unsize`); the
                        // interpreter applies it to the result, as
                        // it would for a coercion cast.
                        let result_unsize = if matches!(
                            kind,
                            crate::builtins::Builtin::FromFrom
                                | crate::builtins::Builtin::IntoInto
                        ) {
                            // The conversion's source type: `From`
                            // renders the callee's generic arguments
                            // as [target, source], `Into` (whose
                            // `Self` is the source) as [source,
                            // target].
                            let source_ty = if matches!(
                                kind,
                                crate::builtins::Builtin::IntoInto
                            ) {
                                raw_generic_args.first()
                            } else {
                                raw_generic_args.get(1)
                            };
                            self.from_result_unsize(
                                body,
                                destination,
                                args.first(),
                                source_ty,
                            )
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
                                destination: self.convert_place(body, destination)?,
                                target: target.map(|b| ir::BasicBlock(b.index())),
                                drop_glue,
                                cmp_discriminants,
                                result_unsize,
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
                            destination: self.convert_place(body, destination)?,
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
                overflow: matches!(
                    &**msg,
                    mir::AssertMessage::Overflow(..) | mir::AssertMessage::OverflowNeg(..)
                ),
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
                        place: self.convert_place(body, place)?,
                        target: ir::BasicBlock(target.index()),
                        glue,
                    },
                    None => ir::TerminatorKind::Unsupported("Drop".to_string()),
                }
            }
            T::Resume => ir::TerminatorKind::Resume,
            T::Terminate(mir::TerminateKind::Abort) => ir::TerminatorKind::Abort,
            T::Terminate(mir::TerminateKind::Resume) => ir::TerminatorKind::Resume,
            T::Yield { .. } => ir::TerminatorKind::Unsupported("Yield".to_string()),
            T::CoroutineDrop => ir::TerminatorKind::Unsupported("CoroutineDrop".to_string()),
            T::InlineAsm {
                template,
                operands,
                options,
                destination,
                ..
            } => {
                if mir::inline_asm_is_noop(template, options) {
                    match destination {
                        Some(target) => ir::TerminatorKind::Goto {
                            target: ir::BasicBlock(target.index()),
                        },
                        None => ir::TerminatorKind::Unsupported(
                            "InlineAsm (no-op template without a destination)".to_string(),
                        ),
                    }
                } else if let Some(program) = mir::inline_asm_program(
                    template,
                    options,
                    operands,
                    destination.is_some(),
                ) {
                    if program.steps.is_empty() {
                        // The step-over class: templates with no
                        // real instruction, whose operands the
                        // classifier leaves inert (e.g. the
                        // explicit-register operands of a
                        // clobber-only block).
                        match destination {
                            Some(target) => ir::TerminatorKind::Goto {
                                target: ir::BasicBlock(target.index()),
                            },
                            None => ir::TerminatorKind::Unsupported(
                                "InlineAsm (no-op template without a destination)"
                                    .to_string(),
                            ),
                        }
                    } else {
                        ir::TerminatorKind::InlineAsm {
                            regs: self.convert_asm_regs(body, operands)?,
                            steps: program.steps,
                            target: if program.returns {
                                None
                            } else {
                                destination.map(|b| ir::BasicBlock(b.index()))
                            },
                        }
                    }
                } else {
                    ir::TerminatorKind::Unsupported(format!(
                        "InlineAsm (template {template:?} is not a no-op)"
                    ))
                }
            }
            T::TailCall { .. } => ir::TerminatorKind::Unsupported("TailCall".to_string()),
        })
    }
}

// ---------------------------------------------------------------------------

/// Preserve one MIR unwind action in the IR (see [`ir::UnwindAction`]).
fn convert_unwind_action(action: &mir::UnwindAction) -> ir::UnwindAction {
    match action {
        mir::UnwindAction::Cleanup(bb) => {
            ir::UnwindAction::Cleanup(ir::BasicBlock(bb.index()))
        }
        mir::UnwindAction::Continue => ir::UnwindAction::Continue,
        mir::UnwindAction::Terminate(_) => ir::UnwindAction::Terminate,
        mir::UnwindAction::Unreachable => ir::UnwindAction::Unreachable,
    }
}

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
/// The type a literal operand's producer recorded for it, if any.
fn constant_recorded_ty(op: &mir::Operand) -> Option<mir::Ty> {
    match op {
        mir::Operand::Constant(c) => c.ty.clone(),
        _ => None,
    }
}

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

/// Human-readable text for a MIR `Assert` message: the custom
/// string when there is one; for the compiler-generated kinds,
/// the exact message core panics with (these messages are what a
/// `catch_unwind` caller observes as the panic's `&str` payload,
/// so they must match core's text, not a paraphrase). Kinds whose
/// message embeds operand values (bounds checks, misaligned
/// dereferences) keep the stable kind text: the operands are not
/// rendered.
fn assert_message_text(msg: &mir::AssertMessage) -> String {
    use mir::AssertMessage as A;
    match msg {
        A::Custom(s) => s.clone(),
        A::Overflow(op, _, _) => match op {
            mir::BinOp::Add | mir::BinOp::AddWithOverflow => {
                "attempt to add with overflow".to_string()
            }
            mir::BinOp::Sub | mir::BinOp::SubWithOverflow => {
                "attempt to subtract with overflow".to_string()
            }
            mir::BinOp::Mul | mir::BinOp::MulWithOverflow => {
                "attempt to multiply with overflow".to_string()
            }
            mir::BinOp::Div => "attempt to divide with overflow".to_string(),
            mir::BinOp::Rem => {
                "attempt to calculate the remainder with overflow".to_string()
            }
            mir::BinOp::Shl | mir::BinOp::ShlUnchecked => {
                "attempt to shift left with overflow".to_string()
            }
            mir::BinOp::Shr | mir::BinOp::ShrUnchecked => {
                "attempt to shift right with overflow".to_string()
            }
            other => format!("assertion failed: {other:?}"),
        },
        A::OverflowNeg(_) => "attempt to negate with overflow".to_string(),
        A::DivisionByZero(_) => "attempt to divide by zero".to_string(),
        A::RemainderByZero(_) => {
            "attempt to calculate the remainder with a divisor of zero".to_string()
        }
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

/// Parse an impl-instance key `<Self as Trait>::method` (the
/// trait may carry a generic argument list, and `Self` may carry
/// one too) into `(Self section, trait path without arguments,
/// method)`. `None` for any other key shape.
fn parse_impl_key(base: &str) -> Option<(String, String, String)> {
    let (head, method) = base.rsplit_once(">::")?;
    if method.is_empty() || method.contains("::") {
        return None;
    }
    let inner = head.strip_prefix('<')?;
    let (self_part, trait_part) = inner.split_once(" as ")?;
    if self_part.is_empty() || trait_part.is_empty() {
        return None;
    }
    let trait_base = trait_part.split('<').next().unwrap_or(trait_part);
    Some((
        self_part.to_string(),
        trait_base.to_string(),
        method.to_string(),
    ))
}

/// Whether the `Self` section of an impl key denotes the type
/// named `self_name`: either the name itself, or the name with a
/// generic argument list appended (`YCombinator<F, A, R>` for the
/// type `YCombinator` — an impl body recorded under its generic
/// definition name, while a call's `Self` type renders without
/// the definition's parameter list). The crate prefix may be
/// present on either side or on neither.
fn impl_self_matches(self_part: &str, self_name: &str) -> bool {
    let self_base = self_part.split('<').next().unwrap_or(self_part);
    // A generic Self section must be exactly `name<...>` (the
    // argument list runs to the end of the section).
    if self_part != self_base && !self_part.ends_with('>') {
        return false;
    }
    if self_base == self_name {
        return true;
    }
    // When both sides carry a crate path they must agree exactly;
    // the short-name comparison is only for a side that omits it
    // (the generic-definition spelling above).
    if self_base.contains("::") && self_name.contains("::") {
        return false;
    }
    let short = self_name.rsplit("::").next().unwrap_or(self_name);
    let self_short = self_base.rsplit("::").next().unwrap_or(self_base);
    self_short == short
}

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

/// A synthesized value as an IR constant, for the shapes a
/// unique-value type can take (units, tuples and ADT values);
/// anything else has no constant form here.
fn value_as_constant(value: &Value) -> Option<ir::Constant> {
    Some(match value {
        Value::Unit => ir::Constant::Unit,
        Value::Enum { variant, fields } => ir::Constant::Adt {
            variant: *variant,
            fields: fields
                .iter()
                .map(value_as_constant)
                .collect::<Option<Vec<_>>>()?,
        },
        Value::Tuple(fields) => ir::Constant::Tuple(
            fields
                .iter()
                .map(value_as_constant)
                .collect::<Option<Vec<_>>>()?,
        ),
        _ => return None,
    })
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
    /// Table entries whose plan is still being built: they hold a
    /// `Trivial` placeholder that a recursive reference must keep
    /// pointing at (the finished plan replaces it), so the
    /// placeholder must never be read as a finished trivial plan.
    in_progress: std::collections::HashSet<usize>,
}

impl<'c, 'a> DropGlueBuilder<'c, 'a> {
    fn build(conv: &'c Converter<'a>, ty: &mir::Ty) -> Option<ir::DropGlue> {
        let mut builder = DropGlueBuilder {
            conv,
            adts: Vec::new(),
            adt_index: HashMap::new(),
            in_progress: Default::default(),
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
                    // A finished entry whose plan is trivial is the
                    // trivial plan itself: returning it lets the
                    // caller's field plans collapse, so a value
                    // holding fewer fields than the plan lists (a
                    // zero-sized field the value model elides) is
                    // not mistaken for an unaccounted drop.
                    if !self.in_progress.contains(&index)
                        && self.adts[index] == P::Trivial
                    {
                        return Some(P::Trivial);
                    }
                    return Some(P::Adt { index });
                }
                let index = self.adts.len();
                self.adt_index.insert(key, index);
                // Placeholder while this ADT's fields are planned, so
                // a recursive occurrence refers back to this entry.
                self.adts.push(P::Trivial);
                self.in_progress.insert(index);
                let plan = self.adt_plan(*def, args);
                self.in_progress.remove(&index);
                let plan = plan?;
                let trivial = plan == P::Trivial;
                self.adts[index] = plan;
                Some(if trivial {
                    P::Trivial
                } else {
                    P::Adt { index }
                })
            }
            mir::Ty::Dynamic(..) => {
                // A trait object's glue is its concrete type's,
                // resolved at run time from the value (see
                // `DropPlan::Dyn` and the registration at unsizing
                // casts). A value carrying no accounted glue is
                // judged by its shape at run time, the judgment
                // `DropPlan::Dynamic` names.
                Some(P::Dyn)
            }
            mir::Ty::Coroutine(def, args) => {
                let key = format!("{ty:?}");
                if let Some(&index) = self.adt_index.get(&key) {
                    if !self.in_progress.contains(&index)
                        && self.adts[index] == P::Trivial
                    {
                        return Some(P::Trivial);
                    }
                    return Some(P::Adt { index });
                }
                let index = self.adts.len();
                self.adt_index.insert(key, index);
                // Placeholder while this coroutine's fields are
                // planned, so a recursive occurrence (a future
                // awaiting another future of the same type) refers
                // back to this entry.
                self.adts.push(P::Trivial);
                self.in_progress.insert(index);
                let plan = self.coroutine_plan(*def, args);
                self.in_progress.remove(&index);
                let plan = plan?;
                let trivial = plan == P::Trivial;
                self.adts[index] = plan;
                Some(if trivial {
                    P::Trivial
                } else {
                    P::Adt { index }
                })
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
            "std::panic::AssertUnwindSafe" | "core::panic::AssertUnwindSafe" => {
                // A one-field transparent wrapper with no `Drop`
                // impl of its own: dropping it drops the held
                // value. `catch_unwind` arguments are dropped on
                // every path through the panic machinery, so this
                // glue is on the coroutine panic path as well.
                let inner = self.plan(args.first()?)?;
                return Some(at_field0(inner));
            }
            "std::pin::Pin" | "core::pin::Pin" => {
                // `Pin<P>` has no destructor of its own (it is a
                // transparent newtype with no `Drop` impl); dropping
                // it drops the pointer it wraps, exactly as dropping
                // the pointer itself would: `Pin<Box<T>>` runs the
                // box's pointee glue, `Pin<&mut T>` drops nothing
                // (`at_field0` collapses a trivial inner plan).
                let inner = self.plan(args.first()?)?;
                return Some(at_field0(inner));
            }
            "std::vec::Vec" => {
                let elem = self.plan(args.first()?)?;
                return Some(match elem {
                    P::Trivial => P::Trivial,
                    e => P::Each { elem: Box::new(e) },
                });
            }
            "std::boxed::Box" => {
                // A box is a one-field aggregate holding the
                // pointee in this value model (see the `Box::new`
                // builtin), and dropping the pointer drops the
                // pointee. A `dyn` pointee is the exception: the
                // pointer value *is* the fat pointer then, with no
                // field to project, so the plan is the dyn plan
                // itself.
                let inner = self.plan(args.first()?)?;
                return Some(match inner {
                    P::Dyn => P::Dyn,
                    other => at_field0(other),
                });
            }
            "std::rc::Rc" | "std::sync::Arc" => {
                // The handle names a reference-counted allocation
                // in the interpreter's store (see the `Rc::new` /
                // `Arc::new` builtins): dropping it decrements the
                // strong count, and the pointee's glue runs only
                // when the last strong handle goes away. A `dyn`
                // pointee keeps the dyn plan: the handle value *is*
                // the fat pointer then, and its baked glue applies
                // to the pointee in the allocation.
                let inner = self.plan(args.first()?)?;
                return Some(match inner {
                    P::Dyn => P::Dyn,
                    other => P::RcHandleDrop {
                        value: Box::new(other),
                    },
                });
            }
            "std::rc::Weak" | "std::sync::Weak" => {
                // A weak pointer owns no pointee; dropping it
                // releases its weak reference on the allocation,
                // which is observable through the counts and
                // through `upgrade` once the storage is freed.
                return Some(P::RcWeakRelease);
            }
            "std::borrow::Cow" if matches!(args.first(), Some(mir::Ty::Str)) => {
                // `Cow<str>`: the `Borrowed` variant holds a `&str`
                // and the `Owned` variant a `String`; both are
                // glue-free in this model. (Other borrowed types
                // own a `B::Owned` whose glue this builder cannot
                // derive from `B`, so they stay unmodelled.)
                return Some(P::Trivial);
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
            "std::ops::CoroutineState" | "core::ops::CoroutineState" => {
                // The resume result: an enum (`Yielded` / `Complete`)
                // with no `Drop` impl of its own, so its glue is its
                // variants' fields' glue. Every `resume` produces
                // one, and `catch_unwind` wraps it in a `Result`, so
                // this glue is on the coroutine runtime path itself.
                return Some(self.variant_fields_plan(def, args)?);
            }
            "std::task::Poll" | "core::task::Poll" => {
                // The future poll result: an enum (`Pending` /
                // `Ready`) with no `Drop` impl of its own; its glue
                // is its variants' fields' glue, as for
                // `CoroutineState`.
                return Some(self.variant_fields_plan(def, args)?);
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
            // mpsc handles: the value is the channel id, and the
            // destructor's only observable effect — the shared
            // handle count going down, which is what disconnects
            // the channel — is the interpreter's channel-handle
            // drop plan.
            "std::sync::mpsc::Sender" | "std::sync::mpsc::SyncSender" => {
                return Some(P::ChannelHandleDrop { sender: true })
            }
            "std::sync::mpsc::Receiver" => {
                return Some(P::ChannelHandleDrop { sender: false })
            }
            // `std::io::Error` (the payload of a `Builder::spawn`
            // result) has no `Drop` impl: its glue is its fields'
            // glue, planned from the same evidence the generic
            // path uses for the entry crate's types. An
            // unevidenced field stays unaccounted for and fails
            // the plan, as usual.
            "std::io::Error" => {
                return Some(self.variant_fields_plan(def, args)?)
            }
            // A thread `Builder`'s two fields are `Option<String>`
            // and `Option<usize>`, both trivial-glue; the value is
            // built by the `Builder` builtins in that shape.
            "std::thread::Builder" => {
                return Some(P::Fields {
                    fields: vec![P::Trivial, P::Trivial],
                    complete: true,
                })
            }
            // Join handles: the destructor detaches the thread.
            // In the cooperative thread model a detached thread
            // simply stays pending (it runs at the next scheduling
            // point), so there is no glue to run.
            "std::thread::JoinHandle" | "std::thread::ScopedJoinHandle" => {
                return Some(P::Trivial)
            }
            "std::cell::BorrowMutError" | "core::cell::BorrowMutError" => {
                return Some(P::Trivial)
            }
            "std::ops::Range" | "std::ops::RangeInclusive" => {
                let bound = self.plan(args.first()?)?;
                return Some(collapse_fields(vec![bound.clone(), bound], true));
            }
            "std::ops::RangeFrom" | "std::ops::RangeTo" | "std::ops::RangeToInclusive" => {
                // The one-bound ranges: a single field holding the
                // bound, whose glue is the field's.
                let bound = self.plan(args.first()?)?;
                return Some(collapse_fields(vec![bound], true));
            }
            "std::ops::RangeFull" => return Some(P::Trivial),
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
            "std::vec::IntoIter" | "alloc::vec::IntoIter" => {
                // Same modelled state as `array::IntoIter` (the
                // element snapshot plus the next index), so the
                // same accounting: only a trivial element type is
                // accounted for.
                let elem = self.plan(args.first()?)?;
                return match elem {
                    P::Trivial => Some(P::Trivial),
                    _ => None,
                };
            }
            "std::ffi::OsString" | "std::ffi::OsStr" => {
                // An `OsString` is the string value itself in this
                // model (see the `env::args_os` builtin): no owned
                // resource beyond it.
                return Some(P::Trivial);
            }
            "std::io::Stdout" | "std::io::Stderr" | "std::io::StdoutLock"
            | "std::io::StderrLock" => {
                // The modelled handle holds no owned resource:
                // writes through it take effect immediately in
                // this model, so the native drop-time flush of
                // the line writer is unobservable.
                return Some(P::Trivial);
            }
            "std::env::Args" | "std::env::ArgsOs" => {
                // The modelled value wraps the argument iterator
                // state ([element snapshot, next index]) whose
                // elements are the argument strings themselves:
                // dropping the iterator drops the not-yet-yielded
                // strings, which is unobservable for string values.
                return Some(P::Trivial);
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
    /// state's fields, state by state. The compiler numbers the
    /// states 0 (unresumed), 1 (returned), 2 (panicked) and one per
    /// suspension point from 3 on, and a state's fields are all
    /// projected by the body that saves or resumes it - so a state
    /// with no evidence holds no fields to drop. The plan must
    /// still *cover* every state the value can be dropped in: the
    /// three fixed states, plus every state the coroutine's own
    /// body names in a `SetDiscriminant` (the only way a state tag
    /// is ever written), even when that state holds no evidenced
    /// fields - a suspended coroutine with nothing live across the
    /// suspension point is dropped in exactly such a state.
    fn coroutine_plan(
        &mut self,
        def: mir::DefId,
        args: &[mir::Ty],
    ) -> Option<ir::DropPlan> {
        use ir::DropPlan as P;
        let conv = self.conv;
        let variants = self.evidenced_variants(def, args);
        let mut max_v = 2u32;
        if let Some(&top) = variants.iter().max() {
            max_v = max_v.max(top);
        }
        if let Some(body) = conv
            .def_names
            .get(&def)
            .and_then(|key| conv.bodies.get(key))
        {
            for bb in &body.basic_blocks {
                for stmt in &bb.statements {
                    if let mir::StatementKind::SetDiscriminant {
                        variant_index, ..
                    } = &stmt.kind
                    {
                        max_v = max_v.max(*variant_index);
                    }
                }
            }
        }
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
