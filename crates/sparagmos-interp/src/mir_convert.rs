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
    /// Per-variant field byte layouts of the monomorphic ADT
    /// instances in the program, as recorded by the driver
    /// (see [`mir::AdtFieldLayouts`]). Consulted by the
    /// interpreter's byte view of an ADT value's storage.
    pub adt_field_layouts: mir::AdtFieldLayouts,
    /// Direct discriminant tag layouts of the monomorphic
    /// enum instances in the program, as recorded by the
    /// driver (see [`mir::AdtTagLayouts`]). Consulted by the
    /// interpreter's byte view of an enum value's storage.
    pub adt_tag_layouts: mir::AdtTagLayouts,
    /// Storage-slot maps of the program's monomorphic coroutine
    /// instances, as recorded by the driver (see
    /// `mir::CoroutineLayouts`). Consulted when a place projects
    /// a state field of a coroutine value: the flat positional
    /// field vector does not by itself model the state-machine
    /// layout's sharing of one field across suspension states.
    pub coroutine_layouts: mir::CoroutineLayouts,
    /// Declared shapes of the program's ADTs, as recorded by the
    /// driver (see `mir::AdtShape`). Used to decide which locals
    /// are provably zero-sized.
    pub adt_shapes: mir::AdtShapes,
    /// Derived-`Debug` shapes of the program's monomorphic ADT
    /// instances, as recorded by the driver (see
    /// `mir::AdtDebugShapes`). Consulted by the `Debug` formatter
    /// for ADT values.
    pub adt_debug_shapes: mir::AdtDebugShapes,
    /// Debugger shapes of the program's monomorphic closure and
    /// coroutine instances, as recorded by the driver (see
    /// `mir::EnvDebugShapes`). Consulted by the debugger
    /// interface when it prints a closure or coroutine value.
    pub env_debug_shapes: mir::EnvDebugShapes,
    /// Thread-local statics of the program, as recorded by the
    /// driver (see `mir::ThreadLocalDef`): each entry's initializer
    /// seeds one thread's storage for that static.
    pub thread_locals: mir::ThreadLocals,
    /// Declared types of the program's statics, as recorded by
    /// the driver (see `mir::StaticTys`).
    pub static_tys: mir::StaticTys,
    /// Field types of monomorphic ADT instances whose generic
    /// shape mentions a type alias, as recorded by the driver
    /// (see `mir::AdtInstanceShapes`). Consulted for an ADT
    /// instance's field types before the generic shape, whose
    /// alias field types cannot be resolved here.
    pub adt_instance_shapes: mir::AdtInstanceShapes,
    /// Niche discriminant encodings of monomorphic enum
    /// instances, as recorded by the driver (see
    /// `mir::AdtNiches`). Consulted when a transmute involves
    /// a niche-encoded enum such as `Option<NonZero<T>>`.
    pub adt_niches: mir::AdtNiches,
    /// The source-file table the bodies' source lines index into
    /// (see `mir::Body::source_lines`), as recorded by the driver.
    /// Handed to the interpreter program for the debugger
    /// interface.
    pub source_files: Vec<String>,
    /// ADT instantiations whose drop glue the compiler certifies
    /// empty, as recorded by the driver (it resolves each dropped
    /// type's drop-in-place shim through Stable MIR and keeps the
    /// ones the compiler reports as empty shims). Keyed by the
    /// `Debug` rendering of the instantiation's type, the same
    /// rendering the drop-glue builder keys its ADT table with.
    /// Consulted for an ADT with no converted destructor body:
    /// certification covers the whole tree, fields included, so
    /// the trivial plan is exact for these types.
    pub drop_trivial_adts: std::collections::HashSet<String>,
    /// ADT instantiations the driver certifies have no `Drop`
    /// impl at all (its crate — the only one the orphan rule
    /// allows an impl in — defines none for the type). Keyed
    /// like `drop_trivial_adts`. A certified type's glue is
    /// exactly its fields' glue, so the glue builder may use
    /// its field-evidence plan for a foreign type it would
    /// otherwise have to refuse (an unrecorded impl cannot be
    /// ruled out for an uncertified foreign type).
    pub drop_no_impl_adts: std::collections::HashSet<String>,
    /// Destructor bodies the driver demanded for specific ADT
    /// instantiations, keyed by the `Debug` rendering of the
    /// instantiation's type (the certification tables' key) with
    /// the program name of the converted `Drop::drop` body as
    /// the value. The glue builder's own key construction
    /// renders generic arguments its way, which cannot reproduce
    /// the compiler's instance names for every argument shape
    /// (a closure argument is the case that forced this table);
    /// the driver knows the exact name it demanded.
    pub dtor_bodies: std::collections::HashMap<String, String>,

    /// Definition-location stamps of the program's
    /// `#[track_caller]` instances, keyed by the instance's
    /// `DefId`, as recorded by the driver (see
    /// `Converter::track_caller_defs` in the driver crate).
    pub track_caller_defs: HashMap<mir::DefId, mir::ConstValue>,
    /// The location stamps of the `Fn` / `FnMut` / `FnOnce`
    /// vtable shims (index 0 is `call`, 1 is `call_mut`, 2 is
    /// `call_once`), as recorded by the driver.
    pub fn_shim_locations: [Option<mir::ConstValue>; 3],
}

/// Widen an `f16` bit pattern to `f64`. The conversion is exact:
/// every f16 value (including subnormals) is representable in f64,
/// and scaling by a power of two never rounds.
/// The pointee type of a raw-pointer / reference type.
fn ptr_pointee(t: &mir::Ty) -> Option<&mir::Ty> {
    match t {
        mir::Ty::RawPtr(p, _) | mir::Ty::Ref(_, p, _) => Some(&**p),
        _ => None,
    }
}

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

/// Whether an `Fn`-trait call's `Self` type denotes a closure
/// or a trait object (possibly behind a reference or pointer):
/// callables the `CallTrait` dispatch resolves from the
/// receiver value. A concrete ADT `Self` instead names a
/// hand-written `Fn` impl, whose collected shim body the
/// direct resolution should keep.
/// The coroutine-closure aggregate that creates local `local`'s
/// value in `body`, if the body creates it: the closure's
/// definition and the types of its captures (the upvars tuple of
/// the aggregate's generic arguments). A coroutine-closure type
/// is recorded by the driver as the unit placeholder (its kind
/// is unaskable through Stable MIR), so the aggregate that
/// creates the value is the one place a body's conversion can
/// learn what the value is.
fn coroutine_closure_creation(
    body: &mir::Body,
    local: usize,
) -> Option<(mir::DefId, Vec<mir::Ty>)> {
    for bb in &body.basic_blocks {
        for stmt in &bb.statements {
            let mir::StatementKind::Assign(pair) = &stmt.kind else {
                continue;
            };
            let (place, rv) = &**pair;
            if place.local.index() != local || !place.projection.is_empty() {
                continue;
            }
            let mir::Rvalue::Aggregate(kind, _) = rv else {
                continue;
            };
            let mir::AggregateKind::CoroutineClosure {
                def,
                generic_args,
            } = &**kind
            else {
                continue;
            };
            let upvars = match generic_args.iter().rev().nth(1) {
                Some(mir::Ty::Tuple(ts)) => ts.clone(),
                _ => Vec::new(),
            };
            return Some((*def, upvars));
        }
    }
    None
}

/// The per-capture "taken by reference" flags of an async
/// closure, from its creation aggregate's generic arguments:
/// the upvars tuple (second-to-last) gives the environment's
/// field types, and the by-reference signature (last, a
/// fn-pointer type whose output is the tuple of the types the
/// closure's coroutine borrows for a call) gives each capture's
/// borrowed type. A capture whose borrowed type is a reference
/// TO its environment type was taken by value - the call borrows
/// the environment field itself. Any other capture was taken by
/// reference - the environment field is itself a reference, and
/// the call reborrows it. `None` when the arguments do not have
/// the expected shape.
fn async_capture_by_ref(generic_args: &[mir::Ty]) -> Option<Vec<bool>> {
    let upvars = match generic_args.iter().rev().nth(1) {
        Some(mir::Ty::Tuple(ts)) => ts,
        _ => return None,
    };
    let borrowed = match generic_args.last() {
        Some(mir::Ty::FnPtr(sig)) => match &*sig.output {
            mir::Ty::Tuple(ts) => ts,
            _ => return None,
        },
        _ => return None,
    };
    if upvars.len() != borrowed.len() {
        return None;
    }
    Some(
        upvars
            .iter()
            .zip(borrowed.iter())
            .map(|(env_ty, borrowed_ty)| {
                !matches!(borrowed_ty, mir::Ty::Ref(_, inner, _) if &**inner == env_ty)
            })
            .collect(),
    )
}

fn fn_self_is_closure_or_dyn(ty: &mir::Ty) -> bool {
    match ty {
        mir::Ty::Closure(..) | mir::Ty::Dynamic(..) => true,
        mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => {
            fn_self_is_closure_or_dyn(inner)
        }
        // An ADT `Self` wraps the callable in its first type
        // argument (`Box<dyn FnOnce>` — with its allocator
        // argument behind it — is the `Self` of its `call_once`
        // call, `Box<closure>` of a boxed closure's): look
        // through it.
        mir::Ty::Adt(_, args) => match args.first() {
            Some(first) => fn_self_is_closure_or_dyn(first),
            None => false,
        }
        _ => false,
    }
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
    out.async_closure_captures = conv.async_closure_captures.clone();
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
    out.adt_field_layouts = program.adt_field_layouts.clone();
    out.adt_tag_layouts = program.adt_tag_layouts.clone();
    out.adt_debug_shapes = program.adt_debug_shapes.clone();
    out.env_debug_shapes = program.env_debug_shapes.clone();
    out.alloc_shims = conv.alloc_shims.clone();
    // The thread-locals table: each recorded static's initializer,
    // converted with its declared type as context and keyed by the
    // same static key the `ThreadLocalRef` rvalues carry. A seed
    // that does not convert is dropped; evaluating that static's
    // `ThreadLocalRef` then fails with an explicit error.
    out.thread_locals = program
        .thread_locals
        .iter()
        .filter_map(|(def, defn)| {
            let seed = conv.convert_const_value(&defn.init, Some(&defn.ty)).ok()?;
            let seed = conv.expand_uninit_constant(&seed, &defn.ty, 0);
            Some((conv.static_key(def), seed))
        })
        .collect();
    // The statics' declared types, keyed by the same static key
    // the `StaticRef` values carry, for the interpreter's
    // pointer-offset denomination over statics.
    out.static_tys = program
        .static_tys
        .iter()
        .filter_map(|(def, ty)| {
            let ty = conv.convert_ty(ty, "a static's declared type").ok()?;
            Some((conv.static_key(def), ty))
        })
        .collect();
    // The definition-location stamps, keyed by the program key
    // of the body each belongs to (the `DefId` a function's
    // calls and constants resolve through), in the same
    // constant form the driver-appended call-site stamps take.
    for (f, key) in program.functions.iter().zip(keys.iter()) {
        let Some(def) = f.def_id else { continue };
        let Some(stamp) = program.track_caller_defs.get(&def) else {
            continue;
        };
        if let Ok(c) = conv.convert_const_value(stamp, None) {
            out.track_caller_defs.insert(key.clone(), c);
        }
    }
    out.fn_shim_locations = [
        program.fn_shim_locations[0]
            .as_ref()
            .and_then(|v| conv.convert_const_value(v, None).ok()),
        program.fn_shim_locations[1]
            .as_ref()
            .and_then(|v| conv.convert_const_value(v, None).ok()),
        program.fn_shim_locations[2]
            .as_ref()
            .and_then(|v| conv.convert_const_value(v, None).ok()),
    ];
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
    // Bodies the driver recorded a definition-location stamp
    // for are track-caller bodies by the compiler's own
    // resolution, whether or not any converted direct call
    // passes a stamp: a body reached only through a function
    // pointer or a vtable is never called directly at all.
    let mut marked: Vec<String> = program
        .track_caller_defs
        .keys()
        .filter(|k| program.functions.contains_key(*k))
        .cloned()
        .collect();
    for body in program.functions.values() {
        for block in &body.blocks {
            if let ir::TerminatorKind::Call { func, args, .. } = &block.terminator {
                if let Some(callee) = program.functions.get(func) {
                    if args.len() == callee.arg_count + 1
                        && callee.caller_location_local.is_none()
                        && !callee_takes_va_list(callee)
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

/// Whether the body's last declared parameter is a C variable
/// argument list (`std::ffi::VaList`, spelled through `core`
/// inside core's own bodies): the body is a C-variadic function,
/// so a caller passing `arg_count + 1` values is passing one
/// tail argument, not a caller-location stamp, and the
/// track-caller heuristic in [`mark_caller_location_bodies`]
/// must not claim it.
fn callee_takes_va_list(body: &ir::Body) -> bool {
    body.arg_count >= 1
        && matches!(
            body.local_tys.get(body.arg_count),
            Some(ir::Ty::Adt(name))
                if name == "std::ffi::VaList" || name == "core::ffi::VaList"
        )
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

/// Whether a drop glue's plans call the destructor body `key`
/// anywhere: through `Seq` / `At` / `Each` / `Fields` /
/// `Variants` / reference-count steps and the glue's ADT table.
fn glue_calls_dtor(glue: &ir::DropGlue, key: &str) -> bool {
    fn walk(
        plan: &ir::DropPlan,
        glue: &ir::DropGlue,
        key: &str,
        seen: &mut std::collections::HashSet<usize>,
    ) -> bool {
        use ir::DropPlan as P;
        match plan {
            P::CallDtor { func } => func == key,
            P::Seq(plans) => plans.iter().any(|p| walk(p, glue, key, seen)),
            P::At { plan, .. } => walk(plan, glue, key, seen),
            P::Each { elem } => walk(elem, glue, key, seen),
            P::Elided { plan, .. } => walk(plan, glue, key, seen),
            P::Fields { fields, .. } => {
                fields.iter().any(|p| walk(p, glue, key, seen))
            }
            P::Variants { variants } => {
                variants.iter().any(|p| walk(p, glue, key, seen))
            }
            P::RcHandleDrop { value } => walk(value, glue, key, seen),
            P::BoxPointee { pointee } => walk(pointee, glue, key, seen),
            P::DynOrPlain { plain } => walk(plain, glue, key, seen),
            P::Adt { index } => {
                seen.insert(*index)
                    && glue
                        .adts
                        .get(*index)
                        .is_some_and(|p| walk(p, glue, key, seen))
            }
            _ => false,
        }
    }
    let mut seen = std::collections::HashSet::new();
    walk(&glue.root, glue, key, &mut seen)
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
            // A waker's release is chosen from the value at run
            // time (a handle release, or the vtable entry the
            // value itself carries): no static edges.
            ir::DropPlan::WakerDrop => true,
            ir::DropPlan::RcHandleDrop { value } => walk(value, adts, targets, seen),
            // The dyn branch is chosen from the value at run time
            // (no static edges, like `Dyn` itself); the plain
            // branch's edges are the subplan's.
            ir::DropPlan::DynOrPlain { plain } => walk(plain, adts, targets, seen),
            // The pointee's storage is judged from the value at
            // run time; the glue that runs is the subplan's.
            ir::DropPlan::BoxPointee { pointee } => walk(pointee, adts, targets, seen),
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
            ir::DropPlan::Elided { plan, .. } => walk(plan, adts, targets, seen),
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
    out.source_files = program.source_files.clone();
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

/// Convert `program` and open a debugger session on `entry`
/// (see [`crate::debug::DebugSession`]): the caller sets its
/// breakpoints and source text, then runs a command script; the
/// session drives the interpreter itself, so nothing executes
/// until the script runs. Conversion failures are reported the
/// same way as for a run: a helper's failure is isolated, the
/// entry's own failure is returned here.
pub fn debug_mir(
    program: &MirProgram,
    entry: &str,
    args: Vec<Value>,
    config: crate::InterpConfig,
) -> Result<crate::debug::DebugSession, MirRunError> {
    let (ir_program, failures) = convert_all(program).map_err(MirRunError::from)?;
    if let Some(e) = failures.get(entry) {
        return Err(MirRunError::from(e.clone()));
    }
    Ok(crate::debug::DebugSession::new(
        ir_program,
        entry.to_string(),
        args,
        config,
    ))
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
/// Whether a scalar shape's leaf is an integer: the shape
/// flattens to an integer bit pattern (transparent wrapper
/// chains over an integer included).
fn scalar_leaf_is_int(shape: &ir::ScalarShape) -> bool {
    match shape {
        ir::ScalarShape::Int { .. } => true,
        ir::ScalarShape::AdtField { inner, .. } => {
            scalar_leaf_is_int(inner)
        }
        _ => false,
    }
}

/// Whether a scalar shape carries a thin pointer anywhere in
/// its payload: the pointer is a place identity in this model,
/// not a bit pattern, so a shape containing one cannot be
/// reinterpreted bit-wise without losing what the pointer
/// denotes.
fn scalar_shape_has_ptr_leaf(shape: &ir::ScalarShape) -> bool {
    match shape {
        ir::ScalarShape::Ptr => true,
        ir::ScalarShape::AdtField { inner, .. } => {
            scalar_shape_has_ptr_leaf(inner)
        }
        ir::ScalarShape::NicheEnum { inner, .. } => {
            scalar_shape_has_ptr_leaf(inner)
        }
        ir::ScalarShape::Array(elem, _) => scalar_shape_has_ptr_leaf(elem),
        _ => false,
    }
}

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

/// Whether a converted operand is a pointer constant: one of
/// the constant forms that evaluate to a pointer value (a
/// reference, an address-only pointer, a null pointer, a static
/// reference, a vtable pointer, a fat pointer, or a callable).
/// A cast of such a constant has no place type to judge from --
/// the constant's own form is the evidence that the source is a
/// pointer.
fn is_pointer_constant(v: &ir::Operand) -> bool {
    matches!(
        v,
        ir::Operand::Constant(
            ir::Constant::Ref(_)
                | ir::Constant::IntPtr(_)
                | ir::Constant::NullPtr
                | ir::Constant::StaticRef { .. }
                | ir::Constant::VTablePtr(_)
                | ir::Constant::FatPtr { .. }
                | ir::Constant::FnDef(_)
                | ir::Constant::Closure(..)
        )
    )
}

/// Whether `ty` is a fat pointer type: a raw pointer or
/// reference whose pointee carries a metadata half (a slice,
/// `str` or `dyn` pointee). A fat pointer's value in this model
/// is its [`ir::Value::FatRef`] / [`ir::Value::DynRef`], so
/// transmutes between fat pointer types — and between a fat
/// pointer and a transparent wrapper around one — reinterpret
/// nothing either: the metadata is part of the value, and the
/// pointee type lives in the destination's declared type.
/// Is `ty` the `str` type or a byte slice `[u8]`?
fn is_str_or_byte_slice(ty: &mir::Ty) -> bool {
    match ty {
        mir::Ty::Str => true,
        mir::Ty::Slice(elem) => matches!(**elem, mir::Ty::Uint(mir::UintTy::U8)),
        _ => false,
    }
}

fn is_fat_ptr_ty(ty: &mir::Ty) -> bool {
    match ty {
        mir::Ty::RawPtr(pointee, _) | mir::Ty::Ref(_, pointee, _) => matches!(
            **pointee,
            mir::Ty::Slice(_) | mir::Ty::Str | mir::Ty::Dynamic(..)
        ),
        _ => false,
    }
}

/// Whether a type mentions a trait object anywhere inside it
/// (a `dyn` type itself, a reference or pointer to one, a
/// `Box<dyn ...>`, a tuple holding one): such a value's method
/// calls dispatch through a vtable.
fn ty_mentions_dyn(ty: &mir::Ty) -> bool {
    match ty {
        mir::Ty::Dynamic(..) => true,
        mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => {
            ty_mentions_dyn(inner)
        }
        mir::Ty::Adt(_, args) | mir::Ty::Closure(_, args) => {
            args.iter().any(ty_mentions_dyn)
        }
        mir::Ty::Tuple(ts) => ts.iter().any(ty_mentions_dyn),
        mir::Ty::Slice(elem) | mir::Ty::Array(elem, _) => {
            ty_mentions_dyn(elem)
        }
        _ => false,
    }
}

/// The pointee type of a pointer or reference type.
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
        Value::Closure { def, env } => env
            .iter()
            .map(value_to_constant)
            .collect::<Option<Vec<_>>>()
            .map(|env| ir::Constant::Closure(def.clone(), env)),
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
                            field_index,
                            ..
                        } => Some((
                            *def,
                            generic_args.clone(),
                            *variant_index,
                            *field_index,
                        )),
                        mir::AggregateKind::Coroutine { def, generic_args } => {
                            Some((*def, generic_args.clone(), 0, None))
                        }
                        _ => None,
                    };
                    if let Some((def, generic_args, variant_index, field_index)) =
                        aggregate_key
                    {
                        // A union aggregate lists only its active
                        // field's operand: the operand belongs to
                        // that field (not to operand position 0),
                        // and the operand count says nothing about
                        // the variant's field count, so no count is
                        // recorded for it.
                        if field_index.is_none() {
                            counts.record(
                                def,
                                generic_args.clone(),
                                variant_index,
                                ops.len(),
                            );
                        }
                        for (i, op) in ops.iter().enumerate() {
                            if let Some(place) = op.place() {
                                if let Some(ty) = operand_place_ty(body, place) {
                                    fields.record(
                                        AdtFieldKey {
                                            def,
                                            generic_args: generic_args.clone(),
                                            variant: variant_index,
                                            field: field_index
                                                .unwrap_or(i as u32),
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
    /// The `bool` is whether the call carries a caller-location
    /// stamp as a third argument (the driver appends one when
    /// the resolved `Fn` shim instance is `#[track_caller]`).
    Trait(ir::FnCallKind, Vec<mir::Ty>, bool),
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
    /// Instance ADT shape tables of the program (see `MirProgram`).
    adt_instance_shapes: &'a mir::AdtInstanceShapes,
    adt_niches: &'a mir::AdtNiches,
    /// Coroutine storage-slot tables of the program (see `MirProgram`).
    coroutine_layouts: &'a mir::CoroutineLayouts,
    /// ADT instance layouts of the program (see `MirProgram`).
    adt_layouts: &'a mir::AdtLayouts,
    /// ADT instantiations with compiler-certified empty drop
    /// glue (see `MirProgram::drop_trivial_adts`).
    drop_trivial_adts: &'a std::collections::HashSet<String>,
    /// ADT instantiations certified to have no `Drop` impl
    /// (see `MirProgram::drop_no_impl_adts`).
    drop_no_impl_adts: &'a std::collections::HashSet<String>,
    /// Demanded destructor bodies by instantiation
    /// (see `MirProgram::dtor_bodies`).
    dtor_bodies: &'a std::collections::HashMap<String, String>,
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
    /// The guest's allocation shims, when the program defines a
    /// `#[global_allocator]`: symbol (`__rust_alloc`, ...) ->
    /// program key of the shim body (`<crate>::_::__rust_alloc`,
    /// ...). In a linked binary the std allocation entry points
    /// bind to these shims; the conversion binds them the same
    /// way (see `resolve_body_callee`).
    alloc_shims: HashMap<String, String>,
    /// Scalar-shape tables of the program's union instances (see
    /// [`ir::Program::union_tables`]), interned in
    /// `Converter::new` by [`Self::compute_union_tables`]: every
    /// conversion round recomputes the same tables, so the
    /// indices baked into converted places agree across rounds.
    union_tables: Vec<Vec<Option<ir::ScalarShape>>>,
    /// Union instance -> index into `union_tables`.
    union_table_ids: Vec<(mir::AdtDefId, Vec<mir::Ty>, usize)>,
    /// Per-capture "taken by reference" flags of the program's
    /// async closures (see [`ir::Program::async_closure_captures`]),
    /// computed in `Converter::new` from the creation aggregates:
    /// a pure function of the program, so every conversion round
    /// computes the same table.
    async_closure_captures: HashMap<String, Vec<bool>>,
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
        let mut alloc_shims: HashMap<String, String> = HashMap::new();
        for key in keys {
            for symbol in [
                "__rust_alloc",
                "__rust_dealloc",
                "__rust_realloc",
                "__rust_alloc_zeroed",
            ] {
                if key.contains("::_::") && key.ends_with(symbol) {
                    alloc_shims
                        .entry(symbol.to_string())
                        .or_insert_with(|| key.clone());
                }
            }
        }
        let mut conv = Converter {
            bodies,
            def_names,
            names: &program.fn_names,
            adt_fields,
            discriminants: &program.adt_discriminants,
            adt_shapes: &program.adt_shapes,
            adt_instance_shapes: &program.adt_instance_shapes,
            adt_niches: &program.adt_niches,
            coroutine_layouts: &program.coroutine_layouts,
            adt_layouts: &program.adt_layouts,
            drop_trivial_adts: &program.drop_trivial_adts,
            drop_no_impl_adts: &program.drop_no_impl_adts,
            dtor_bodies: &program.dtor_bodies,
            adt_aggregates,
            local_crate,
            alloc_shims,
            impl_targets: None,
            unconverted: HashMap::new(),
            union_tables: Vec::new(),
            union_table_ids: Vec::new(),
            async_closure_captures: HashMap::new(),
        };
        conv.compute_union_tables(program);
        conv.compute_async_closure_captures(program);
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

    /// Record the per-capture "taken by reference" flags of
    /// every async closure the program creates (see
    /// [`async_capture_by_ref`]), keyed by the closure's program
    /// key - the key the created `Value::Closure` carries.
    fn compute_async_closure_captures(&mut self, program: &MirProgram) {
        for func in &program.functions {
            for bb in &func.body.basic_blocks {
                for stmt in &bb.statements {
                    let mir::StatementKind::Assign(pair) = &stmt.kind else {
                        continue;
                    };
                    let (_, rv) = &**pair;
                    let mir::Rvalue::Aggregate(kind, _) = rv else {
                        continue;
                    };
                    let mir::AggregateKind::CoroutineClosure {
                        def,
                        generic_args,
                    } = &**kind
                    else {
                        continue;
                    };
                    if let Some(flags) = async_capture_by_ref(generic_args) {
                        self.async_closure_captures
                            .entry(self.body_key(def))
                            .or_insert(flags);
                    }
                }
            }
        }
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
        // A local whose type is a coroutine-closure type arrives
        // with the unit placeholder as its declared type (the
        // driver cannot ask that type's kind through Stable MIR;
        // see `guarded_ty_kind` in the driver). The one statement
        // that creates such a value - the coroutine-closure
        // aggregate assigned to the bare local - names the
        // closure's definition and captures, so the local's type
        // is the closure type the aggregate describes, exactly as
        // a plain closure local's declared type names its own.
        for bb in &body.basic_blocks {
            for stmt in &bb.statements {
                let mir::StatementKind::Assign(pair) = &stmt.kind else {
                    continue;
                };
                let (place, rv) = &**pair;
                if !place.projection.is_empty() {
                    continue;
                }
                let mir::Rvalue::Aggregate(kind, ops) = rv else {
                    continue;
                };
                let mir::AggregateKind::CoroutineClosure {
                    def,
                    generic_args,
                } = &**kind
                else {
                    continue;
                };
                let upvars = match generic_args.iter().rev().nth(1) {
                    Some(mir::Ty::Tuple(ts)) => ts.len(),
                    _ => ops.len(),
                };
                let idx = place.local.index();
                if idx < local_tys.len() {
                    local_tys[idx] = ir::Ty::Closure(self.body_key(def), upvars);
                }
            }
        }
        let mut blocks = Vec::with_capacity(body.basic_blocks.len());
        for bb in &body.basic_blocks {
            blocks.push(self.convert_block(body, bb)?);
        }
        let mut out = ir::Body::new(key.to_string(), body.arg_count, local_tys, blocks);
        out.source_lines = body
            .source_lines
            .iter()
            .map(|b| ir::BlockSourceLines {
                statements: b
                    .statements
                    .iter()
                    .map(|s| ir::SourceLine { file: s.file, line: s.line })
                    .collect(),
                terminator: ir::SourceLine {
                    file: b.terminator.file,
                    line: b.terminator.line,
                },
            })
            .collect();
        // The debugger interface's variable table, from the
        // body's var-debug-info entries (see `ir::DebugVar`).
        out.debug_vars = body
            .var_debug_info
            .iter()
            .filter_map(|info| {
                // Whether the variable is a `char` or a reference
                // to one at the source level, from the type its
                // debug info was recorded against.
                let ty_is_char = |ty: &mir::Ty| match ty {
                    mir::Ty::Char => true,
                    mir::Ty::Ref(_, inner, _) => matches!(&**inner, mir::Ty::Char),
                    _ => false,
                };
                let lines = (info.source_info.span.lo, info.source_info.span.hi);
                match &info.value {
                    mir::VarDebugInfoContents::Local(local) => Some(ir::DebugVarEntry {
                        name: info.name.clone(),
                        source: ir::DebugVar::Place(local.0 as usize, Vec::new()),
                        is_char: ty_is_char(&body.local_decls.get(local.0 as usize)?.ty),
                        ty: Some(crate::builtins::ty_tag(
                            &body.local_decls.get(local.0 as usize)?.ty,
                            self.names,
                        )),
                        lines,
                    }),
                    mir::VarDebugInfoContents::Place(place) => {
                        // The projection, in the interpreter's
                        // resolved form: only field/downcast steps
                        // carry over (the same restriction the
                        // decoder's local-projection reader works
                        // under); a variable reached through
                        // anything else is not carried.
                        let mut proj: Vec<crate::RefProj> = Vec::new();
                        for elem in &place.projection {
                            match elem {
                                mir::PlaceElem::Field(i, _) => {
                                    proj.push(crate::RefProj::Field(*i as usize))
                                }
                                mir::PlaceElem::Downcast(_, v) => {
                                    proj.push(crate::RefProj::Downcast(*v))
                                }
                                _ => return None,
                            }
                        }
                        // Only plain locals carry the char fact; a
                        // projected variable's element type is not
                        // tracked here.
                        let is_char = place.projection.is_empty()
                            && ty_is_char(&body.local_decls.get(place.local.0 as usize)?.ty);
                        // The projected type's tag, when the
                        // projection walk can recover it.
                        let ty = self
                            .debug_projected_ty(
                                &body.local_decls.get(place.local.0 as usize)?.ty,
                                &proj,
                            )
                            .map(|t| crate::builtins::ty_tag(&t, self.names));
                        Some(ir::DebugVarEntry {
                            name: info.name.clone(),
                            source: ir::DebugVar::Place(place.local.0 as usize, proj),
                            is_char,
                            ty,
                            lines,
                        })
                    }
                    mir::VarDebugInfoContents::Constant(c) => {
                        let constant = self.convert_constant(c, c.ty.as_ref()).ok()?;
                        let is_char = c.ty.as_ref().map(|t| ty_is_char(t)).unwrap_or(false);
                        Some(ir::DebugVarEntry {
                            name: info.name.clone(),
                            source: ir::DebugVar::Constant(constant),
                            is_char,
                            ty: c
                                .ty
                                .as_ref()
                                .map(|t| crate::builtins::ty_tag(t, self.names)),
                            lines,
                        })
                    }
                }
            })
            .collect();
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
    /// still has the single value `None`); closures whose
    /// captures are all single-valued; function items. Enums with two or more inhabited variants
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

    /// The one value of the element type of a
    /// `box_assume_init_into_vec_unsafe` argument
    /// (`Box<MaybeUninit<[T; N]>>`), when the element type has
    /// exactly one value: the element the source array's
    /// elided writes would have stored in each unwritten slot.
    /// `None` when the element type has many values (an
    /// unwritten slot is then a genuine uninitialized read, as
    /// before) or the argument's shape is not the expected
    /// box-of-array.
    fn into_vec_elem_value(&self, body: &mir::Body, arg: &mir::Operand) -> Option<Value> {
        fn array_elem(ty: &mir::Ty) -> Option<&mir::Ty> {
            match ty {
                mir::Ty::Array(elem, _) => Some(elem),
                mir::Ty::Adt(_, args) => args.iter().find_map(array_elem),
                _ => None,
            }
        }
        let ty = self.operand_place_ty(body, arg)?;
        self.zst_value(array_elem(&ty)?, 0)
    }

    /// Collect the canonical values of the single-valued
    /// projection paths of `ty` into `out`, keyed by path (see
    /// `ir::Body::zst_fields`). Structs, unions and tuples expose
    /// their fields under bare field projections; an enum's fields
    /// sit behind a downcast step, so each variant's fields are
    /// collected under that variant's downcast step. The runtime
    /// lookup only consults a recorded path after projection
    /// itself walked it and came back uninitialized — a downcast
    /// to a variant the stored value does not hold fails as a
    /// type mismatch, never as uninitialized — so a path recorded
    /// under one variant's downcast is only ever used for a value
    /// that actually holds that variant. A sequence — an array, a
    /// slice, or a `Vec`, whose modelled value holds its elements
    /// under bare index projections — exposes its element type
    /// under an index step: a zero-sized element's store may be
    /// elided like a ZST field's (a `vec!` of a zero-sized enum
    /// reaches execution with no element write in MIR at all), so
    /// the element's single value is recorded for the read that
    /// comes back uninitialized. The walk is depth- and
    /// size-capped: it serves the shallow paths MIR actually
    /// projects (a wrapper or two around a zero-sized field), not
    /// arbitrary recursive types.
    fn collect_zst_fields(
        &self,
        ty: &mir::Ty,
        path: &mut Vec<ir::ZstPathStep>,
        out: &mut std::collections::HashMap<Vec<ir::ZstPathStep>, Value>,
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
        match &ty {
            mir::Ty::Array(elem, _) | mir::Ty::Slice(elem) => {
                let elem = (**elem).clone();
                self.collect_zst_element(&elem, path, out, depth, budget);
                return;
            }
            mir::Ty::Adt(def, args)
                if self.names.get(def).map(String::as_str)
                    == Some("std::vec::Vec") =>
            {
                // A `Vec`'s modelled value holds the elements
                // themselves, so the element type sits under an
                // index step from the Vec place, in addition to
                // the struct fields the walk below records.
                if let Some(elem) = args.first().cloned() {
                    self.collect_zst_element(&elem, path, out, depth, budget);
                }
            }
            _ => {}
        }
        let fields: Vec<mir::Ty> = match &ty {
            mir::Ty::Tuple(ts) => ts.clone(),
            mir::Ty::Adt(def, args) => {
                let Some(shape) = self.adt_shapes.get(def) else {
                    return;
                };
                if shape.kind == mir::AdtShapeKind::Enum {
                    for vi in 0..shape.variants.len() {
                        let Some(fields) =
                            self.instantiated_field_tys(def, args, vi as u32)
                        else {
                            continue;
                        };
                        if *budget == 0 {
                            return;
                        }
                        *budget -= 1;
                        path.push(ir::ZstPathStep::Downcast(vi));
                        self.collect_zst_field_list(
                            &fields,
                            path,
                            out,
                            depth + 1,
                            budget,
                        );
                        path.pop();
                        if out.len() > 4096 {
                            return;
                        }
                    }
                    return;
                }
                match self.instantiated_field_tys(def, args, 0) {
                    Some(fields) => fields,
                    None => return,
                }
            }
            _ => return,
        };
        self.collect_zst_field_list(&fields, path, out, depth, budget);
    }

    /// Record the single-valued values of one field list (a
    /// struct's, a union's, a tuple's, or one enum variant's
    /// fields) under `path`, recursing into each field's own
    /// sub-paths.
    fn collect_zst_field_list(
        &self,
        fields: &[mir::Ty],
        path: &mut Vec<ir::ZstPathStep>,
        out: &mut std::collections::HashMap<Vec<ir::ZstPathStep>, Value>,
        depth: u32,
        budget: &mut usize,
    ) {
        for (i, field_ty) in fields.iter().enumerate() {
            path.push(ir::ZstPathStep::Field(i));
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

    /// Record the single value of a sequence's element type
    /// under an index step at `path`, and recurse into the
    /// element type's own sub-paths beneath that step (a
    /// single-valued field of the elements sits under
    /// `[Index, Field(..)]`, as a struct's sits under
    /// `[Field(..)]`).
    fn collect_zst_element(
        &self,
        elem: &mir::Ty,
        path: &mut Vec<ir::ZstPathStep>,
        out: &mut std::collections::HashMap<Vec<ir::ZstPathStep>, Value>,
        depth: u32,
        budget: &mut usize,
    ) {
        path.push(ir::ZstPathStep::Index);
        if let Some(v) = self.zst_value(elem, 0) {
            out.insert(path.clone(), v);
        }
        self.collect_zst_fields(elem, path, out, depth + 1, budget);
        path.pop();
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
            mir::Ty::Closure(def, args) => match args.last() {
                Some(mir::Ty::Tuple(ts)) => {
                    let mut env = Vec::with_capacity(ts.len());
                    for t in ts {
                        match self.unique_value(t, depth + 1) {
                            UniqueValue::One(v) => env.push(v),
                            UniqueValue::Uninhabited => {
                                return UniqueValue::Uninhabited
                            }
                            UniqueValue::Many => return UniqueValue::Many,
                        }
                    }
                    UniqueValue::One(Value::Closure {
                        def: self.def_display_name(def),
                        env,
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

    /// The program key a function-item *value* names (an `FnDef`
    /// type constant materialised as a callable value, e.g. a
    /// trait method coerced to a function pointer). The constant
    /// names the item's definition, whose own key is the body
    /// key only for a free function: for a trait method the body
    /// in the program is the impl instance for the constant's
    /// `Self` type, resolved exactly as a trait-method call
    /// resolves it (see `find_impl_key_for`). When neither
    /// resolution lands, the definition's key stands (the call
    /// then fails naming the item, as before).
    fn fn_def_value_key(&self, def: &mir::DefId, args: &[mir::Ty]) -> String {
        let key = self.body_key(def);
        if self.bodies.contains_key(&key) {
            return key;
        }
        let name = self.def_display_name(def);
        if let Some((trait_path, method)) = name.rsplit_once("::") {
            if !trait_path.is_empty() && !method.is_empty() {
                if let Some(impl_key) =
                    self.find_impl_key_for(trait_path, method, args)
                {
                    return impl_key;
                }
            }
        }
        key
    }

    /// The name of a concrete type as it appears in the `Self`
    /// position of its impl method names in the program
    /// (`<{name} as Trait>::method`): the definition's display name
    /// with its generic arguments rendered the way a call's generic
    /// arguments render them, for a named type; the surface-syntax
    /// rendering otherwise. The generic arguments are part of the
    /// name: a `dyn` value's concrete tag is compared against the
    /// rendered generic argument of a downcast, which carries them.
    fn concrete_ty_name(&self, ty: &mir::Ty) -> String {
        match ty {
            mir::Ty::Adt(def, args)
            | mir::Ty::Closure(def, args)
            | mir::Ty::Coroutine(def, args) => {
                let base = self.def_display_name(def);
                if args.is_empty() {
                    base
                } else {
                    let parts: Vec<String> = args
                        .iter()
                        .map(|t| crate::builtins::ty_tag(t, self.names))
                        .collect();
                    format!("{base}<{}>", parts.join(", "))
                }
            }
            mir::Ty::FnDef(def, _) => self.def_display_name(def),
            other => crate::builtins::ty_tag(other, self.names),
        }
    }

    /// The field-entry depth of a pointer cast from `source` to
    /// `target` that crosses single-field wrappers to reach the
    /// wrapped type itself, or `None` when the cast is not of
    /// that shape. The wrappers are the transparent cell types
    /// (`UnsafeCell<T>` / `MaybeUninit<T>`, recognized by name
    /// because their field convention is definitional) and any
    /// other struct with exactly one field whose type is the
    /// next layer (the `as_array` newtype idiom).
    /// Decided from the cast's two types, both known at conversion
    /// time: the source pointee is peeled wrapper by wrapper until
    /// it is exactly the target pointee.
    /// Whether two types are the same for cast classification:
    /// structural equality, except that an array length that is
    /// a generic parameter (or otherwise unevaluated) on either
    /// side matches any length -- the recorded field type of a
    /// generic definition carries the parameter, the cast's
    /// target in an instantiated caller carries the value, and
    /// the field-entry decision does not depend on the length.
    fn cast_ty_matches(a: &mir::Ty, b: &mir::Ty) -> bool {
        use mir::Ty as T;
        match (a, b) {
            (T::Array(ea, la), T::Array(eb, lb)) => {
                Self::cast_ty_matches(ea, eb)
                    && (la == lb
                        || matches!(la, mir::ArrayLen::Param(_) | mir::ArrayLen::Unevaluated(_))
                        || matches!(lb, mir::ArrayLen::Param(_) | mir::ArrayLen::Unevaluated(_)))
            }
            (T::Adt(da, aa), T::Adt(db, ab)) => {
                da == db
                    && aa.len() == ab.len()
                    && aa.iter().zip(ab.iter()).all(|(x, y)| Self::cast_ty_matches(x, y))
            }
            (T::Tuple(ta), T::Tuple(tb)) => {
                ta.len() == tb.len()
                    && ta.iter().zip(tb.iter()).all(|(x, y)| Self::cast_ty_matches(x, y))
            }
            (T::Ref(_, ta, ma), T::Ref(_, tb, mb)) => ma == mb && Self::cast_ty_matches(ta, tb),
            (T::RawPtr(ta, ma), T::RawPtr(tb, mb)) => ma == mb && Self::cast_ty_matches(ta, tb),
            (T::Slice(ea), T::Slice(eb)) => Self::cast_ty_matches(ea, eb),
            _ => a == b,
        }
    }

    fn cast_wrapper_entry_depth(
        &self,
        source: &mir::Ty,
        target: &mir::Ty,
    ) -> Option<u32> {
        match self.cast_entry_depth(source, target) {
            Some((depth, false)) => Some(depth),
            _ => None,
        }
    }

    /// The retargeting a pointer cast from `source` to `target`
    /// denotes when it crosses transparent wrappers (and possibly
    /// one array layer) to reach the target pointee: the number
    /// of field-0 entries the wrapper peel denotes, and whether
    /// the peel then continues into the first element of the
    /// array the innermost peeled layer holds (the cast's target
    /// is that array's element type, as in
    /// `buf.as_mut_ptr() as *mut T` over a nested `MaybeUninit`
    /// buffer). `None` when the cast is not of that shape.
    fn cast_entry_depth(
        &self,
        source: &mir::Ty,
        target: &mir::Ty,
    ) -> Option<(u32, bool)> {
        fn pointee(ty: &mir::Ty) -> Option<&mir::Ty> {
            match ty {
                mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => Some(inner),
                _ => None,
            }
        }
        let current = pointee(source)?.clone();
        let target = pointee(target)?;
        // A cast between slice pointers whose element types are
        // related by wrapper entry (`*mut [MaybeUninit<T>]` to
        // `*mut [T]`, the cast the array clone guard's drop
        // performs on the erased buffer): the fat pointer's base
        // denotes the first element's wrapper, so the entry depth
        // of the element types is a field-0 entry on the base.
        // The target is a slice here, so the peel must land
        // exactly on the target element type, never inside it.
        if let (mir::Ty::Slice(src_elem), mir::Ty::Slice(tgt_elem)) = (&current, target) {
            return match self.wrapper_peel((**src_elem).clone(), tgt_elem) {
                Some((depth, false)) => Some((depth, false)),
                _ => None,
            };
        }
        self.wrapper_peel(current, target)
    }

    /// Peel wrapper layers (`UnsafeCell` / `MaybeUninit`, and
    /// single-field array structs) off `current` until it matches
    /// `target`, returning the number of field-0 entries the peel
    /// denotes -- and, when the innermost peeled layer is an
    /// array (or slice) whose element type is exactly `target`,
    /// reporting that the peel continues into that array's first
    /// element. `None` when the peel lands nowhere the target
    /// names.
    fn wrapper_peel(
        &self,
        mut current: mir::Ty,
        target: &mir::Ty,
    ) -> Option<(u32, bool)> {
        let mut depth = 0u32;
        loop {
            let mir::Ty::Adt(def, args) = &current else {
                return None;
            };
            // Both wrapper kinds are one-field aggregates in the
            // value model -- `UnsafeCell` by its declared shape, and
            // `MaybeUninit` by the field-0 convention its pointer
            // accesses use (`MaybeUninit::new` constructions store
            // the payload at field 0 of the union value) -- so
            // entering either appends a field-0 projection. The
            // layers must be peeled to know whether the cast lands
            // exactly on the target pointee.
            let name = self.def_display_name(def);
            let is_wrapper = matches!(
                name.as_str(),
                "std::cell::UnsafeCell"
                    | "core::cell::UnsafeCell"
                    | "std::mem::MaybeUninit"
                    | "core::mem::MaybeUninit"
            );
            let next = if is_wrapper && args.len() == 1 {
                args[0].clone()
            } else if let Some(field) = self.single_field_ty(def, args)
                .filter(|field| matches!(field, mir::Ty::Array(..)))
            {
                // A single-field struct whose only field is an
                // array is entered the same way when the cast's
                // target is exactly the field's type: the
                // `as_array` idiom (`&*p.cast::<[T; N]>()` on a
                // `Simd([T; N])` or an array newtype) denotes the
                // field, not the aggregate. The restriction to
                // array fields is load-bearing: scalar-field
                // wrappers include representation wrappers over
                // values this model stores bare (the atomics'
                // alignment wrappers), where a field projection
                // would address storage that is not there.
                field
            } else {
                return None;
            };
            depth += 1;
            if Self::cast_ty_matches(&next, target) {
                return Some((depth, false));
            }
            // The peel crossed the wrappers and reached an array
            // whose element type is the target: the cast denotes
            // the array's first element, one index-0 step past
            // the field entries the peel denotes.
            if let mir::Ty::Array(elem, _) | mir::Ty::Slice(elem) = &next {
                if Self::cast_ty_matches(elem, target) {
                    return Some((depth, true));
                }
            }
            current = next;
        }
    }

    /// Whether a pointer cast from `source` to `target` goes from
    /// an aggregate pointee (array / slice / `str`) to the element
    /// type itself. Decided from the cast's two types, both known
    /// at conversion time: the source pointee is the aggregate and
    /// the target pointee is exactly its element type (`u8` for
    /// `str`). Such a cast retargets the pointer at the first
    /// element: an array or slice value rides as one aggregate in
    /// this model, so a pointer that denotes the aggregate as a
    /// whole and a pointer that denotes its first element differ
    /// by an index-0 projection, and only the cast's types say
    /// which of the two a given pointer value is.
    fn cast_enters_first_element(&self, source: &mir::Ty, target: &mir::Ty) -> bool {
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
            mir::Ty::Array(elem, _) | mir::Ty::Slice(elem) => **elem == *tgt,
            mir::Ty::Str => matches!(tgt, mir::Ty::Uint(mir::UintTy::U8)),
            _ => false,
        }
    }

    /// Whether a pointer cast from `source` to `target` is the
    /// reverse of [`Converter::cast_enters_first_element`]: the
    /// source pointee is an element type and the target pointee is
    /// an array of exactly that element type.
    fn cast_returns_to_array_base(&self, source: &mir::Ty, target: &mir::Ty) -> bool {
        fn pointee(ty: &mir::Ty) -> Option<&mir::Ty> {
            match ty {
                mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => Some(inner),
                _ => None,
            }
        }
        let (Some(src), Some(tgt)) = (pointee(source), pointee(target)) else {
            return false;
        };
        match tgt {
            mir::Ty::Array(elem, _) => **elem == *src,
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
            mir::Ty::Float(mir::FloatTy::F16) => ir::Ty::Float(ir::FloatTy::F16),
            mir::Ty::Float(mir::FloatTy::F32) => ir::Ty::Float(ir::FloatTy::F32),
            mir::Ty::Float(mir::FloatTy::F64) => ir::Ty::Float(ir::FloatTy::F64),
            mir::Ty::Float(mir::FloatTy::F128) => ir::Ty::Float(ir::FloatTy::F128),
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
        // A coroutine place's `Downcast(v)` step followed by a
        // `Field(i)` step names state field `i` of state `v` in
        // the compiler's state-machine layout. When the driver
        // recorded the instance's slot map, the pair converts to
        // the single layout slot of that field (see
        // `mir::CoroutineSlots`): a state field keeps one storage
        // cell in every state that contains it, which the
        // positional field index does not express. A downcast not
        // followed by a field step, and every projection of an
        // instance with no recorded map, converts exactly as
        // before.
        let mut coroutine_downcast: Option<(u32, &mir::CoroutineSlots)> = None;
        for elem in &place.projection {
            while let Some(mir::Ty::Pat(inner, _)) = &ty {
                ty = Some(inner.as_ref().clone());
            }
            if let Some((variant, slots)) = coroutine_downcast.take() {
                if let mir::PlaceElem::Field(idx, field_ty) = elem {
                    if let Some(slot) = slots
                        .variants
                        .get(variant as usize)
                        .and_then(|v| v.get(*idx as usize))
                    {
                        out.projection.push(ir::ProjectionElem::Field(*slot as usize));
                        ty = Some(field_ty.clone());
                        continue;
                    }
                }
                out.projection.push(ir::ProjectionElem::Downcast(variant));
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
                    if let Some(mir::Ty::Coroutine(def, args)) = &ty {
                        let key = crate::builtins::ty_tag(
                            &mir::Ty::Coroutine(*def, args.clone()),
                            self.names,
                        );
                        if let Some(slots) = self.coroutine_layouts.get(&key) {
                            coroutine_downcast = Some((*variant, slots));
                            // The downcast step is emitted (or
                            // folded into the following field step)
                            // at the top of the next iteration, so
                            // this arm never pushes.
                            continue;
                        }
                    }
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
        if let Some((variant, _)) = coroutine_downcast.take() {
            out.projection.push(ir::ProjectionElem::Downcast(variant));
        }
        Ok(out)
    }

    /// Whether a scalar shape has a pointer leaf (see the
    /// pointer exception in [`Self::deref_vector_shape`]).
    fn shape_contains_ptr(shape: &ir::ScalarShape) -> bool {
        match shape {
            ir::ScalarShape::Ptr => true,
            ir::ScalarShape::Array(elem, _) => Self::shape_contains_ptr(elem),
            ir::ScalarShape::AdtField { inner, .. } => Self::shape_contains_ptr(inner),
            ir::ScalarShape::NicheEnum { inner, .. } => Self::shape_contains_ptr(inner),
            _ => false,
        }
    }

    /// The scalar shape of a dereference through `place`:
    /// the shape a shaped load / store of the place runs
    /// through. The dereference may sit below field steps (a
    /// vector stored in a wrapper struct behind a pointer:
    /// `(*p).0` where the field is the vector or its storage
    /// array; the byte inside a tag wrapper: `(*p).0` over a
    /// control byte), because the pointee behind such a
    /// pointer can be stored as raw bytes the plain projection
    /// cannot enter. `None` for any other place: no
    /// dereference, or a place type with no scalar shape (its
    /// loads and stores are plain whole-value ones).
    fn deref_vector_shape(
        &self,
        body: &mir::Body,
        place: &mir::Place,
    ) -> Option<ir::ScalarShape> {
        if !place
            .projection
            .iter()
            .any(|e| matches!(e, mir::PlaceElem::Deref))
        {
            return None;
        }
        let ty = self.place_ty(body, place)?;
        let shape = self.scalar_shape(&ty, 0)?;
        // Any scalar shape qualifies, not only vector (array)
        // ones: a pointee stored as raw bytes — a control byte
        // in the static empty group is the case at hand — can
        // sit below a dereference under a field step of any
        // scalar-shaped type (the byte inside its tag wrapper),
        // and the plain projection cannot enter a byte string.
        // For a pointee stored whole the shaped load/store is
        // the canonicalising round trip, so the broader gate
        // changes nothing there.
        //
        // A shape with a pointer leaf is the exception: the
        // round trip through a pointer shape flattens a
        // reference to its synthetic address and constructs an
        // address-only pointer back, which is not the stored
        // value's identity (a place or allocation reference
        // denotes its target; the address-only form denotes
        // nothing readable). Pointers in this model are stored
        // whole, never as bytes, so there is nothing for a
        // shaped load/store to assemble — the plain projection
        // is the only honest path.
        if Self::shape_contains_ptr(&shape) {
            return None;
        }
        Some(shape)
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

    /// Whether a place's declared type is the coroutine
    /// transform's context cell type, `std::future::ResumeTy`.
    fn is_resume_ty_place(&self, body: &mir::Body, place: &mir::Place) -> bool {
        match self.place_ty(body, place) {
            Some(mir::Ty::Adt(def, _)) => {
                self.def_display_name(&def) == "std::future::ResumeTy"
            }
            _ => false,
        }
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

    /// The element size of a `BinaryOp::Offset`, from the pointer
    /// operand's type (its place type, else the result type — the
    /// operation's result is the pointer type itself), when the
    /// pointee's size is exactly computable from the type alone
    /// (see [`mir_ty_size`]). `None` leaves the size to execution,
    /// which derives it from the executing body's declared type of
    /// the pointer operand.
    fn offset_elem_size(
        &self,
        body: &mir::Body,
        pair: &(mir::Operand, mir::Operand),
        dest_ty: Option<&mir::Ty>,
    ) -> Option<u64> {
        let ty = self
            .operand_place_ty(body, &pair.0)
            .or_else(|| dest_ty.cloned())?;
        let pointee = match ty {
            mir::Ty::RawPtr(inner, _) => *inner,
            mir::Ty::Ref(_, inner, _) => *inner,
            _ => return None,
        };
        if let Some(size) = mir_ty_size(&pointee) {
            return Some(size);
        }
        // A composite pointee's size is its recorded layout: an
        // ADT's size lives in the layout table (the
        // `MaybeUninit<Chunk>` pointees of the byte-swap
        // machinery are the canonical case), and a tuple's size
        // is its fields' layouts composed — a hashbrown bucket
        // pointer's pointee is a `(K, V)` tuple whose fields may
        // themselves be ADTs.
        self.ty_layout(&pointee).map(|(size, _)| size)
    }

    /// The source pointee size for a pointer-to-pointer cast
    /// that needs its projection index rewritten in byte units
    /// (`Rvalue::PtrCastUnits`): both pointees' layouts are
    /// known, they differ, and the source is thin. `None` for
    /// every other cast, which keeps its existing form.
    fn cast_units_from_size(
        &self,
        ctx: &Option<mir::Ty>,
        target: &mir::Ty,
    ) -> Option<u64> {
        let src = ptr_pointee(ctx.as_ref()?)?;
        let dst = ptr_pointee(target)?;
        if matches!(
            src,
            mir::Ty::Slice(_) | mir::Ty::Str | mir::Ty::Dynamic(..)
        ) {
            return None;
        }
        let from = self.ty_layout(src).map(|(size, _)| size)?;
        let to = self.ty_layout(dst).map(|(size, _)| size)?;
        if from != to && from > 0 {
            Some(from)
        } else {
            None
        }
    }

    /// The native `(size, align)` of a type whose layout is
    /// structurally known or recorded in the layout table:
    /// scalars by their width, arrays and tuples by composing
    /// their elements' layouts (declaration order, each field
    /// aligned to its own alignment, the total rounded up to
    /// the widest), ADTs by their recorded layout. `None` when
    /// any component's layout is unknown.
    fn ty_layout(&self, ty: &mir::Ty) -> Option<(u64, u64)> {
        use mir::Ty as T;
        Some(match ty {
            T::Bool => (1, 1),
            T::Char => (4, 4),
            T::Int(t) => {
                let b = u64::from(t.bits() / 8);
                (b, b)
            }
            T::Uint(t) => {
                let b = u64::from(t.bits() / 8);
                (b, b)
            }
            T::Float(t) => {
                let b = u64::from(t.bits() / 8);
                (b, b)
            }
            T::Array(elem, mir::ArrayLen::Known(n)) => {
                let (es, ea) = self.ty_layout(elem)?;
                (es.checked_mul(*n)?, ea)
            }
            T::RawPtr(pointee, _) | T::Ref(_, pointee, _) => {
                if matches!(
                    &**pointee,
                    T::Slice(_) | T::Str | T::Dynamic(..)
                ) {
                    (16, 8)
                } else {
                    (8, 8)
                }
            }
            T::FnPtr(_) => (8, 8),
            T::Tuple(ts) => {
                let mut size = 0u64;
                let mut align = 1u64;
                for f in ts {
                    let (fs, fa) = self.ty_layout(f)?;
                    align = align.max(fa);
                    size = size.checked_add(fa - 1)? / fa * fa;
                    size = size.checked_add(fs)?;
                }
                (size.checked_add(align - 1)? / align * align, align)
            }
            T::Adt(def, args) => {
                let tag = crate::builtins::ty_tag(
                    &mir::Ty::Adt(*def, args.clone()),
                    self.names,
                );
                *self.adt_layouts.get(&tag)?
            }
            T::Pat(inner, _) => self.ty_layout(inner)?,
            T::Never => (0, 1),
            _ => return None,
        })
    }

    // ----- constants & operands ---------------------------------------------

    fn convert_constant(
        &self,
        constant: &mir::Constant,
        ctx: Option<&mir::Ty>,
    ) -> Result<ir::Constant, ConvertError> {
        match &constant.literal {
            mir::ConstKind::Value(v) => {
                let k = self.convert_const_value(v, ctx)?;
                // A constant whose bytes are (partly) uninitialized
                // decodes to a bare `Uninit` whatever its type; give
                // it the aggregate skeleton of its declared type so
                // element/field writes and reads see the structure
                // (see `expand_uninit_constant`). Without a context
                // type there is no skeleton to rebuild.
                Ok(match ctx {
                    Some(ty) => self.expand_uninit_constant(&k, ty, 0),
                    None => k,
                })
            }
            // A function item in operand position is a value (the
            // zero-sized fn item, reifiable to a function pointer):
            // it becomes a callable constant naming the function.
            mir::ConstKind::Ty(mir::Ty::FnDef(def, args)) => {
                Ok(ir::Constant::FnDef(self.fn_def_value_key(def, args)))
            }
            // A closure in constant position is a value too (the
            // zero-sized closure constant a `ClosureFnPointer` cast
            // coerces, or a stored closure value). A type constant
            // carries no stored bytes, so the value is materialised
            // from the type alone: the captured environment is
            // materialised exactly when every captured value is
            // the unique value of its type (a non-capturing closure
            // trivially qualifies). A capture whose type has
            // several values cannot be materialised here and stays
            // an explicit unsupported naming the closure.
            mir::ConstKind::Ty(mir::Ty::Closure(def, args)) => {
                // The upvars tuple is the closure type's LAST
                // generic argument (see `convert_ty`); the third
                // argument is the upvars tuple only when the
                // closure has no parent generics, so a closure
                // constant inside a generic context (e.g. the
                // closure inside `Sum::sum`) was misread as
                // capturing an environment it does not have.
                let env = match args.last() {
                    Some(mir::Ty::Tuple(ts)) => ts
                        .iter()
                        .map(|t| {
                            self.zst_value(t, 0)
                                .and_then(|v| value_to_constant(&v))
                        })
                        .collect::<Option<Vec<_>>>(),
                    _ => None,
                };
                match env {
                    Some(env) => Ok(ir::Constant::Closure(
                        self.body_key(def),
                        env,
                    )),
                    None => Err(ConvertError::UnsupportedConstant {
                        kind: format!(
                            "closure constant {} with a captured environment",
                            self.def_display_name(def)
                        ),
                    }),
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

    /// Expand the `Uninit` leaves of a seed constant (a
    /// thread-local's initializer, a static's initializer, an
    /// operand constant) into the aggregate skeleton of their
    /// declared type.
    ///
    /// The constant decoder collapses every all-uninitialized byte
    /// range into a bare `Uninit`, whatever its type. That loses
    /// the value's shape wherever the constant is used: a
    /// thread-local storage's `data` field, or — under optimized
    /// MIR, where `[MaybeUninit::uninit(); N]` const-propagates
    /// into a single `<uninit>` constant of the array type — an
    /// ordinary local whose elements are then written and read
    /// through dynamic indices. For a thread-local storage: the
    /// `data` field is an `UnsafeCell<MaybeUninit<T>>`, a real value
    /// of an aggregate type, and std's lazy initialization copies
    /// the `MaybeUninit` out by value (`ptr::replace` in
    /// `get_or_init_slow`) -- a read that must yield the uninitialized
    /// `MaybeUninit` itself, not fail for touching uninitialized
    /// memory. Guest code materializes that same value through the
    /// `MaybeUninit::uninit()` construction, whose conversion the
    /// aggregate arm of `convert_rvalue` rewrites to the model's
    /// field-0 payload form, so the expansion rebuilds exactly that
    /// form for a `MaybeUninit`: the payload slot carrying the
    /// payload type's own expansion. For the rest: structs and
    /// tuples get one slot per field, arrays get one slot per
    /// element, a union gets its unit-typed
    /// field (the `uninit: ()` pattern) with `Uninit` padding before
    /// it, and scalar leaves stay `Uninit`. Unions without a unit
    /// field have no canonical uninitialized form and keep the bare
    /// `Uninit`, as do enums (no variant is determined).
    fn expand_uninit_constant(
        &self,
        value: &ir::Constant,
        ty: &mir::Ty,
        depth: usize,
    ) -> ir::Constant {
        if depth > 16 {
            return value.clone();
        }
        match (value, ty) {
            (ir::Constant::Uninit, mir::Ty::Adt(def, args)) => {
                // A `MaybeUninit<T>` is held as a single field-0
                // payload (see the aggregate conversion in
                // `convert_rvalue`): an uninitialized one carries
                // the uninitialized skeleton of `T` in that slot,
                // not the union's unit-typed `uninit` field value,
                // so a constant of this type expands to the same
                // value the `MaybeUninit::uninit()` construction
                // materializes. (Bulk consumers are unaffected by
                // the shape: `write_bytes` counts uninitialised
                // leaves as occupying their slots when it decides
                // a fill is the target's whole value.)
                if self.is_maybe_uninit(def) {
                    if let Some(payload_ty) = args.first() {
                        let payload = self.expand_uninit_constant(
                            &ir::Constant::Uninit,
                            payload_ty,
                            depth + 1,
                        );
                        return ir::Constant::Adt {
                            variant: 0,
                            fields: vec![payload],
                        };
                    }
                }
                let Some(shape) = self.adt_shapes.get(def) else {
                    return value.clone();
                };
                match shape.kind {
                    mir::AdtShapeKind::Struct => {
                        let Some(fields) = shape.variants.first() else {
                            return value.clone();
                        };
                        let fields = fields
                            .iter()
                            .enumerate()
                            .map(|(fi, fty)| {
                                let fty = self
                                    .adt_fields
                                    .get(*def, args, 0, fi as u32)
                                    .cloned()
                                    .unwrap_or_else(|| subst_ty(fty, args));
                                self.expand_uninit_constant(&ir::Constant::Uninit, &fty, depth + 1)
                            })
                            .collect();
                        ir::Constant::Adt {
                            variant: 0,
                            fields,
                        }
                    }
                    mir::AdtShapeKind::Union => {
                        let Some(fields) = shape.variants.first() else {
                            return value.clone();
                        };
                        let unit_idx = fields
                            .iter()
                            .position(|f| subst_ty(f, args).is_unit() || f.is_unit());
                        match unit_idx {
                            Some(k) => {
                                let mut fields =
                                    vec![ir::Constant::Uninit; k];
                                fields.push(ir::Constant::Unit);
                                ir::Constant::Adt {
                                    variant: 0,
                                    fields,
                                }
                            }
                            None => value.clone(),
                        }
                    }
                    mir::AdtShapeKind::Enum => value.clone(),
                }
            }
            (ir::Constant::Uninit, mir::Ty::Tuple(ts)) => ir::Constant::Tuple(
                ts.iter()
                    .map(|t| self.expand_uninit_constant(&ir::Constant::Uninit, t, depth + 1))
                    .collect(),
            ),
            (ir::Constant::Adt { variant, fields }, mir::Ty::Adt(def, args)) => {
                let Some(shape) = self.adt_shapes.get(def) else {
                    return value.clone();
                };
                let Some(field_tys) = shape.variants.get(*variant as usize) else {
                    return value.clone();
                };
                let fields = fields
                    .iter()
                    .enumerate()
                    .map(|(fi, f)| match field_tys.get(fi) {
                        Some(fty) => {
                            let fty = self
                                .adt_fields
                                .get(*def, args, *variant, fi as u32)
                                .cloned()
                                .unwrap_or_else(|| subst_ty(fty, args));
                            self.expand_uninit_constant(f, &fty, depth + 1)
                        }
                        None => f.clone(),
                    })
                    .collect();
                ir::Constant::Adt {
                    variant: *variant,
                    fields,
                }
            }
            (ir::Constant::Tuple(fields), mir::Ty::Tuple(ts)) => ir::Constant::Tuple(
                fields
                    .iter()
                    .zip(ts.iter())
                    .map(|(f, t)| self.expand_uninit_constant(f, t, depth + 1))
                    .collect(),
            ),
            (ir::Constant::Tuple(fields), mir::Ty::Array(elem, _)) => ir::Constant::Tuple(
                fields
                    .iter()
                    .map(|f| self.expand_uninit_constant(f, elem, depth + 1))
                    .collect(),
            ),
            (ir::Constant::Uninit, mir::Ty::Array(elem, len)) => {
                // An uninitialized array constant is an array of
                // uninitialized elements: expand it element by
                // element so the value carries the array's shape
                // and element writes/reads through dynamic indices
                // see the same structure the unoptimized `Repeat`
                // form materializes. Lengths that are not known
                // constants, and arrays past the repeat
                // materialization limit, keep the bare `Uninit`.
                let mir::ArrayLen::Known(n) = len else {
                    return value.clone();
                };
                if *n > MAX_REPEAT_ELEMENTS {
                    return value.clone();
                }
                let elem =
                    self.expand_uninit_constant(&ir::Constant::Uninit, elem, depth + 1);
                ir::Constant::Tuple(vec![elem; *n as usize])
            }
            _ => value.clone(),
        }
    }

    /// The modelled-contents form of an empty `Vec` / `String`
    /// constant, when `value` is one's raw-layout constant tree
    /// (see the `Adt` arm of `convert_const_value`). The raw
    /// length is the second field of the raw `Vec` tree (the
    /// first is the buffer); a `String` wraps that tree in its
    /// single field.
    fn empty_modelled_container(
        &self,
        value: &mir::ConstValue,
        ctx: Option<&mir::Ty>,
    ) -> Option<ir::Constant> {
        let Some(mir::Ty::Adt(def, _)) = ctx else {
            return None;
        };
        let name = self.names.get(def).or_else(|| self.def_names.get(def))?;
        let mir::ConstValue::Adt { fields, .. } = value else {
            return None;
        };
        let raw_vec_len = |fields: &[mir::ConstValue]| -> Option<u128> {
            match fields.get(1) {
                Some(mir::ConstValue::Scalar(mir::Scalar::Int { data, .. })) => {
                    Some(*data)
                }
                _ => None,
            }
        };
        match name.as_str() {
            "std::vec::Vec" => (raw_vec_len(fields)? == 0).then_some(ir::Constant::Adt {
                variant: 0,
                fields: Vec::new(),
            }),
            "std::string::String" => {
                let mir::ConstValue::Adt {
                    fields: inner_fields,
                    ..
                } = fields.first()?
                else {
                    return None;
                };
                (raw_vec_len(inner_fields)? == 0)
                    .then_some(ir::Constant::Str(String::new()))
            }
            _ => None,
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
                // In pointer-to-slice position the bytes are not
                // the value: the value is the fat pointer to them
                // (a `&[u8]` static's initializer decodes to this
                // form), and a bare byte value where a slice
                // pointer belongs loses the length every consumer
                // of the pointer reads.
                if ctx_is_u8_slice_ptr(ctx) {
                    ir::Constant::SliceVal(
                        b.iter()
                            .map(|byte| ir::Constant::Uint(u128::from(*byte)))
                            .collect(),
                    )
                } else {
                    ir::Constant::Bytes(b.clone())
                }
            }
            mir::ConstValue::Slice { data, meta } => {
                // A slice constant whose metadata equals its byte
                // length is a byte slice.
                if *meta as usize == data.len() {
                    // As with the byte form above: in
                    // pointer-to-slice position the constant is
                    // the fat pointer, not the pointee data.
                    if ctx_is_u8_slice_ptr(ctx) {
                        ir::Constant::SliceVal(
                            data.iter()
                                .map(|byte| ir::Constant::Uint(u128::from(*byte)))
                                .collect(),
                        )
                    } else {
                        ir::Constant::Bytes(data.clone())
                    }
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
                // A constant of a modelled container type arrives
                // in the type's raw layout (a `Vec` constant is
                // the nested RawVec tree const evaluation
                // produced), but the value model represents the
                // container by its contents: a `Vec::new()`
                // default field value would otherwise materialise
                // as a raw-layout aggregate whose two fields the
                // `Vec::len` builtin reads as a length of 2. An
                // empty container's contents are empty in both
                // layouts, so it translates exactly; a non-empty
                // one's elements live in the buffer allocation the
                // raw tree points into, which this layer does not
                // decode, so it keeps the raw form.
                if let Some(k) = self.empty_modelled_container(value, ctx) {
                    return Ok(k);
                }
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
                let mut fields: Vec<ir::Constant> = fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let fctx = field_ctxs
                            .as_ref()
                            .and_then(|ts| ts.get(i))
                            .map(|t| t as &mir::Ty);
                        self.convert_const_value(f, fctx)
                    })
                    .collect::<Result<_, _>>()?;
                // A decoded `MaybeUninit<T>` constant carries one
                // slot per union field -- the `uninit: ()` slot
                // beside the `value: ManuallyDrop<T>` slot, both
                // decoded from the same bytes -- but the value
                // model holds a `MaybeUninit` as a single field-0
                // payload: the shape `MaybeUninit::new`
                // constructions store (see the aggregate
                // evaluation in the interpreter) and the shape its
                // pointer accesses read. Normalize the two-slot
                // decoded form to that shape: the payload is the
                // `value` slot's constant with the `ManuallyDrop`
                // wrapper unwrapped.
                if fields.len() == 2 {
                    if let Some(mir::Ty::Adt(def, _)) = ctx {
                        let name = self.def_display_name(def);
                        if matches!(
                            name.as_str(),
                            "std::mem::MaybeUninit" | "core::mem::MaybeUninit"
                        ) {
                            let value_slot = fields.pop().expect("two slots");
                            let payload = match value_slot {
                                ir::Constant::Adt { fields: inner, .. }
                                    if inner.len() == 1 =>
                                {
                                    inner.into_iter().next().expect("one field")
                                }
                                other => other,
                            };
                            fields = vec![payload];
                        }
                    }
                }
                ir::Constant::Adt {
                    variant: *variant,
                    fields,
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
            mir::ConstValue::StaticRef {
                def,
                value,
                projection,
                len,
            } => {
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
                let seed = self.convert_const_value(value, inner_ctx)?;
                // A static whose initializer is (partly)
                // uninitialized decodes to a bare `Uninit` seed
                // with no trace of the declared type's structure:
                // element writes into the storage then land in a
                // shapeless value and element reads fail as
                // uninitialized-place reads. When this reference
                // denotes the static whole (empty projection), the
                // pointee context IS the seed's type -- the
                // declared type for a thin reference, and for a fat
                // slice reference the array type the slice covers,
                // rebuilt from the element type and the carried
                // length -- so give the seed the declared type's
                // skeleton exactly as the thread-local table does
                // for its seeds. An interior reference's seed is
                // still the full initializer, whose type this
                // conversion does not know, so it keeps the bare
                // form.
                let seed = if projection.is_empty() {
                    let seed_ty = match (inner_ctx, len) {
                        (Some(ty), None) => Some(ty.clone()),
                        (Some(mir::Ty::Slice(elem)), Some(n)) => Some(mir::Ty::Array(
                            elem.clone(),
                            mir::ArrayLen::Known(*n),
                        )),
                        (Some(ty @ mir::Ty::Array(..)), Some(_)) => Some(ty.clone()),
                        _ => None,
                    };
                    match seed_ty {
                        Some(ty) => self.expand_uninit_constant(&seed, &ty, 0),
                        None => seed,
                    }
                } else {
                    seed
                };
                ir::Constant::StaticRef {
                    name: self.static_key(def),
                    value: Box::new(seed),
                    projection: projection
                        .iter()
                        .map(|e| self.convert_ref_proj(e))
                        .collect::<Result<_, _>>()?,
                    len: len.map(u128::from),
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
                ir::Constant::Closure(self.body_key(def), Vec::new())
            }
            mir::ConstValue::IntPtr(addr) => ir::Constant::IntPtr(*addr),
            mir::ConstValue::VTablePtr(id) => ir::Constant::VTablePtr(*id),
            mir::ConstValue::FatPtr { addr, meta } => ir::Constant::FatPtr {
                addr: *addr,
                meta: *meta,
            },
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
                mir::FloatTy::F128 => Ok(ir::Constant::Float128(*bits)),
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
            mir::Operand::Move(p) => {
                // The coroutine transform moves the `ResumeTy`
                // out of the state object's context cell at every
                // opaque-future poll site, then reads the cell
                // again for the next poll of the same activation:
                // the cell is state-machine storage (rewritten at
                // every state entry), not an owned value the move
                // vacates. `ResumeTy` is a pointer wrapper with no
                // drop glue of its own, so reading it without
                // vacating the cell is exact, and no other move in
                // the program has this shape.
                if self.is_resume_ty_place(body, p) {
                    ir::Operand::Copy(self.convert_place(body, p)?)
                } else {
                    ir::Operand::Move(self.convert_place(body, p)?)
                }
            }
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

    /// The field of a struct that carries its pointer chain:
    /// the unique field whose type is not a zero-sized type
    /// with a uniquely determined value. The remaining fields
    /// are fills in the sense of [`Self::wrapper_shape`]:
    /// they carry no information, so the chain field alone is
    /// the struct's payload. `Some((index, ty))`, or `None`
    /// when the ADT is not a single-variant struct or has
    /// zero / more than one non-fill field.
    fn chain_field(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
    ) -> Option<(usize, mir::Ty)> {
        let shape = self.adt_shapes.get(def)?;
        if shape.kind != mir::AdtShapeKind::Struct || shape.variants.len() != 1 {
            return None;
        }
        let fields = self.instantiated_field_tys(def, args, 0)?;
        let mut chain: Option<(usize, mir::Ty)> = None;
        for (i, fty) in fields.iter().enumerate() {
            if self.zst_value(fty, 0).is_some() {
                continue;
            }
            if chain.is_some() {
                return None;
            }
            chain = Some((i, fty.clone()));
        }
        chain
    }

    /// The operand denoting the thin pointer at the bottom of
    /// a chain of ADTs (the `NonNull` shape:
    /// `NonNull { pointer: NonZero<*const T> }` — the payload
    /// hides behind the `NonZero` newtype, one layer deeper
    /// than the transparent wrappers `wrapper_shape` matches,
    /// whose payload field must be the pointer itself). Each
    /// layer must be a struct with exactly one non-fill field
    /// (see [`Self::chain_field`]): the owning pointers are
    /// such chains carrying extra zero-sized fields — `Box`
    /// is a `Unique` plus an allocator, and `Unique` is a
    /// `NonNull` plus a `PhantomData`. The chain ends at the
    /// first field that is itself a thin pointer. `None` for
    /// any other shape.
    fn ptr_chain_operand(
        &self,
        body: &mir::Body,
        operand: &mir::Operand,
        v: &ir::Operand,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
    ) -> Option<ir::Operand> {
        let (index, field_ty) = self.chain_field(def, args)?;
        let field_v =
            self.field_operand(body, operand, v, index as u32, &field_ty)?;
        if is_thin_ptr_ty(&field_ty) {
            return Some(field_v);
        }
        let mir::Ty::Adt(fdef, fargs) = &field_ty else {
            return None;
        };
        // The next layer's operand: the same place extended by
        // the field projection (a constant operand stays a
        // constant: `field_operand` reads its fields from the
        // converted value instead).
        let next_operand = match operand.place() {
            Some(place) => {
                let place =
                    place.clone().field(index as u32, field_ty.clone());
                match operand {
                    mir::Operand::Move(_) => mir::Operand::Move(place),
                    _ => mir::Operand::Copy(place),
                }
            }
            None => operand.clone(),
        };
        self.ptr_chain_operand(body, &next_operand, &field_v, fdef, fargs)
    }

    /// Whether both types are pointers (a reference or a raw
    /// pointer) to byte-string DSTs: `str`, a `[u8]` slice,
    /// `std::ffi::OsStr` (or its sys representation type
    /// `std::sys::os_str::bytes::Slice`, the pointee the
    /// `OsStr` constructors actually transmute to) or
    /// `std::path::Path`. All of them are a
    /// byte string with a length metadata half, and in this
    /// value model a reference to any of them is the byte
    /// string itself (see the projection pun for the wrapper
    /// chain in `interp`), so a transmute between any two of
    /// these pointer types reinterprets nothing.
    fn is_byte_string_ptr_pair(&self, s: &mir::Ty, t: &mir::Ty) -> bool {
        fn pointee(ty: &mir::Ty) -> Option<&mir::Ty> {
            match ty {
                mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => Some(inner),
                _ => None,
            }
        }
        let byte_string = |ty: &mir::Ty| -> bool {
            match ty {
                mir::Ty::Str => true,
                mir::Ty::Slice(elem) => {
                    matches!(**elem, mir::Ty::Uint(mir::UintTy::U8))
                }
                mir::Ty::Adt(def, _) => matches!(
                    self.names
                        .get(def)
                        .or_else(|| self.def_names.get(def))
                        .map(String::as_str),
                    Some(
                        "std::ffi::OsStr"
                            | "std::sys::os_str::bytes::Slice"
                            | "std::path::Path"
                    )
                ),
                _ => false,
            }
        };
        match (pointee(s), pointee(t)) {
            (Some(a), Some(b)) => byte_string(a) && byte_string(b),
            _ => false,
        }
    }

    /// Whether a pointee type is one of the byte-string DST
    /// structs: `std::ffi::CStr`, `std::ffi::OsStr` (or its
    /// sys representation type), `std::path::Path`. Each is a
    /// byte string with a length metadata half, and its value
    /// in this model is the byte string itself (the same
    /// family `is_byte_string_ptr_pair` recognises), so a
    /// pointer to one is a fat pointer.
    fn pointee_is_byte_string_dst(&self, ty: &mir::Ty) -> bool {
        match ty {
            mir::Ty::Adt(def, _) => matches!(
                self.names
                    .get(def)
                    .or_else(|| self.def_names.get(def))
                    .map(String::as_str),
                Some(
                    "std::ffi::CStr"
                        | "core::ffi::CStr"
                        | "std::ffi::OsStr"
                        | "std::sys::os_str::bytes::Slice"
                        | "std::path::Path"
                )
            ),
            _ => false,
        }
    }

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
    /// * thin-pointer transparent wrapper <-> integer: the
    ///   payload pointer's synthetic address, wrapped back up
    ///   on the way back (`NonNull<u8>` -> `usize` is the
    ///   transmute `fmt`'s statically-known-string check
    ///   performs);
    /// * fieldless enum (or a one-field wrapper around one,
    ///   `mem::Alignment`) -> thin-pointer transparent wrapper:
    ///   the discriminant as the payload pointer's address
    ///   (`NonNull::dangling`);
    /// * zero-sized type -> zero-sized type whose value is
    ///   uniquely determined by its type;
    /// * scalar-shaped -> scalar-shaped of the same total width
    ///   ([`ir::Rvalue::Reinterpret`]): integers, floats, `bool`
    ///   and `char` in any combination, arrays of scalars, and
    ///   one-payload-field ADTs over those, and niche-encoded
    ///   enums over a scalar or thin-pointer payload
    ///   (`Option<NonNull<T>>`) — the bit-level
    ///   reinterpretation `transmute` means there;
    /// * pointer <-> pointer over byte-string DST pointees
    ///   (`str`, `[u8]`, `OsStr`, `Path` in any combination);
    /// * fat pointer <-> fat pointer, and fat pointer <->
///   transparent wrapper struct (one fat-pointer payload
    ///   field, the remaining fields zero-sized with unique
    ///   values): `NonNull<str>` and its kin;
    /// * one-field struct over a fieldless enum <-> integer:
    ///   the discriminant of the field's value (`mem::Alignment`
    ///   -> `usize`), and the reverse (integer -> the wrapper:
    ///   the integer is the wrapped enum's discriminant);
    /// * fieldless enum <-> the one-field transparent struct
    ///   wrapping it (`mem::Alignment` <-> its enum): the enum
    ///   value itself, wrapped into / read out of the struct's
    ///   single field.
    /// * one-payload wrapper <-> its payload, for a payload of
    ///   any type: a one-field struct whose only field has
    ///   exactly the other side's type, and `MaybeUninit<T>`
    ///   <-> `T` (the payload sits at field 0 of the union
    ///   value in this model) -- the `assume_init` transmute
    ///   an inlined `mem::zeroed` performs under optimization,
    ///   and its reverse.
    ///
    /// * function pointer -> function pointer: the value is
    ///   the function identity, independent of the signature it
    ///   is viewed through (an inlined `fmt::Argument`
    ///   construction rewrites the display fn pointer to the
    ///   erased unsafe signature the struct stores);
    /// * array -> array of the same length across an element
    ///   wrapper boundary (`[T; N]` <-> `[MaybeUninit<T>; N]`,
    ///   or the `UnsafeCell` twin): the cast maps the wrapper's
    ///   value representation over the elements
    ///   ([`ir::Rvalue::WrapArrayElems`] /
    ///   [`ir::Rvalue::UnwrapArrayElems`]). An inlined by-value
    ///   array iterator construction performs this cast on the
    ///   backing array;
    /// * a byte-string constant in pointer position (it
    ///   carries no place type) -> a thin pointer or a
    ///   transparent wrapper over one: const propagation
    ///   replaced the reference to the static byte string with
    ///   the allocation's bytes themselves (the `fmt::Arguments`
    ///   template), so the reference to those bytes is formed
    ///   first and the pointer decisions above run on it.
    ///
    /// Every other shape — aggregate <-> aggregate byte
    /// reinterpretation beyond one payload field, fat pointers,
    /// enums with payloads beyond the niche-encoded shape —
    /// stays [`ir::Rvalue::Unsupported`]:
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
        // A constant operand carries no place type. When the
        // constant is a decoded ADT value, the type its
        // producer recorded for it stands in for the place
        // type: a const-propagated `core` body holds
        // transmutes of folded constants whose reading
        // depends on the source type, which the value's
        // structure alone does not name (see the driver's
        // `constant_adt_ty`). A scalar constant keeps the
        // `None` reading: its shape comes from the converted
        // constant itself (`constant_scalar_shape`), which is
        // the form the value actually carries -- a `1u8`
        // transmuted to `bool` arrives as a bool constant,
        // not as the integer its literal spelled.
        let src = self
            .operand_place_ty(body, operand)
            .or_else(|| match constant_recorded_ty(operand) {
                Some(t @ mir::Ty::Adt(..)) => Some(t),
                _ => None,
            });
        // A byte-string constant in pointer position carries no
        // place type: const propagation replaced the reference to
        // the static byte string with the allocation's bytes
        // themselves (the shape an inlined `fmt::Arguments::new`
        // gives the template field: the bytes transmuted to the
        // `NonNull<u8>` the struct stores). The pointer the
        // transmute means is the reference to those bytes, so
        // when the target is a thin pointer or a transparent
        // wrapper over one, form that reference and record the
        // pointer type it denotes; the ordinary pointer decisions
        // below then run on the wrapped operand.
        let bytes_ref;
        let bytes_ty;
        let (src, v) = match v {
            ir::Operand::Constant(ir::Constant::Bytes(b))
                if is_thin_ptr_ty(target)
                    || matches!(
                        target,
                        mir::Ty::Adt(def, args)
                            if self.wrapper_shape(def, args).is_some()
                    ) =>
            {
                bytes_ref = ir::Operand::Constant(ir::Constant::Ref(
                    Box::new(ir::Constant::Bytes(b.clone())),
                ));
                bytes_ty = Some(mir::Ty::RawPtr(
                    Box::new(mir::Ty::Uint(mir::UintTy::U8)),
                    mir::Mutability::Not,
                ));
                (bytes_ty, &bytes_ref)
            }
            _ => (src, v),
        };
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
        // A string constant transmuted to a byte-slice pointer:
        // the constant's value is the string itself, which the
        // model's consumers read as bytes as readily as as text
        // (`PtrMetadata` reports its byte length, byte reads go
        // through the same storage), so the value carries over.
        if matches!(v, ir::Operand::Constant(ir::Constant::Str(_)))
            && is_fat_ptr_ty(target)
            && matches!(ptr_pointee(target), Some(t) if is_str_or_byte_slice(t))
        {
            return ir::Rvalue::Use(v.clone());
        }
        // The scalar-shape reinterpretation, decided from the
        // two types alone: when both sides flatten to the same
        // nonzero bit width, the cast is a `Reinterpret`. The
        // specialised arms below still take the shapes whose
        // result this does not reproduce (fieldless enums carry
        // no scalar shape, and the bare pointer pairs are claimed
        // by their own arms before this one is reached).
        let shape_to = self.scalar_shape(target, 0);
        let shape_from = match &src {
            Some(s) => self.scalar_shape(s, 0),
            // A constant operand carries no place type; its
            // shape comes from the constant itself (see
            // `constant_scalar_shape`).
            None => shape_to
                .as_ref()
                .and_then(|t| self.constant_scalar_shape(v, t.bits())),
        };
        // A transmute between two ADTs of identical scalar
        // shape whose shape carries a thin pointer: the value's
        // only payload is the pointer itself, whose place
        // identity a bit reinterpretation would reduce to an
        // address-only reference that no longer reads the
        // pointee (the same reason the bare-pointer pairs below
        // stay with their specialised arms). The two sides
        // interpret the stored value identically -- the wrapper
        // layers add no representation of their own in this
        // model, and the pointee type lives in the destination's
        // declared type -- so the value carries over unchanged.
        // `Box<MaybeUninit<T>>` -> `Box<T>`, the transmute
        // `Box::assume_init` performs, is the canonical case:
        // reinterpretation rebuilt the box aggregate around the
        // pointer's synthetic address, and every later read of
        // the pointee missed the allocation.
        let ptr_identity = matches!(
            (&src, target),
            (Some(mir::Ty::Adt(..)), mir::Ty::Adt(..))
        ) && match (&shape_from, &shape_to) {
            (Some(from), Some(to)) => {
                from == to && scalar_shape_has_ptr_leaf(from)
            }
            _ => false,
        };
        let generic = {
            let to = shape_to.clone();
            let from = shape_from.clone();
            match (from, to) {
                (Some(from), Some(to))
                    if from.bits() == to.bits() && from.bits() > 0 =>
                {
                    // A bare pointer on one side and a wrapper
                    // ADT on the other stays with the specialised
                    // arms below: they carry the pointer value
                    // itself (its place identity) across, where a
                    // bit reinterpretation would reduce it to an
                    // address-only reference that no longer reads
                    // the pointee. A niche-encoded ADT is the
                    // exception: no specialised arm reads a
                    // variant-dependent payload, so its pointer
                    // pairs go through the reinterpretation.
                    let src_bare_ptr = matches!(
                        src,
                        Some(mir::Ty::RawPtr(..) | mir::Ty::Ref(..))
                    );
                    let target_bare_ptr =
                        is_thin_ptr_ty(target) || is_fat_ptr_ty(target);
                    // Both sides bare pointers stay with the
                    // specialised arms too: they carry the value
                    // itself across (or refuse the wrapper-exiting
                    // crossing), where a bit reinterpretation
                    // would silently change what the pointer
                    // denotes.
                    let identity_pair = (src_bare_ptr && target_bare_ptr)
                        || (src_bare_ptr
                            && matches!(target, mir::Ty::Adt(..))
                            && !matches!(to, ir::ScalarShape::NicheEnum { .. }))
                        || (target_bare_ptr
                            && matches!(src, Some(mir::Ty::Adt(..)))
                            && !matches!(from, ir::ScalarShape::NicheEnum { .. }));
                    if identity_pair {
                        None
                    } else {
                        Some(ir::Rvalue::Reinterpret(v.clone(), from, to))
                    }
                }
                _ => None,
            }
        };
        // One-payload wrapper crossings, decided ahead of the
        // shape arms below: a one-field struct (or a
        // `MaybeUninit`) transmuted to exactly its payload type
        // reads the payload field out, and the reverse
        // transmute wraps the value into the wrapper. The arms
        // below keep the shapes whose payload is not the whole
        // other side -- pointer payloads behind newtype chains
        // and fieldless-enum payloads -- which these checks
        // cannot mistake for this one: they require the field
        // type to be exactly the other side's type.
        if let Some(mir::Ty::Adt(def, args)) = &src {
            if let Some(op) =
                self.payload_field_operand(body, operand, v, def, args, target)
            {
                return ir::Rvalue::Use(op);
            }
        }
        if let mir::Ty::Adt(tdef, targs) = target {
            if let Some(src_ty) = &src {
                if let Some(r) =
                    self.wrap_payload_rvalue(v, src_ty, tdef, targs)
                {
                    return r;
                }
            }
        }
        // An ADT carrying an integer scalar (a
        // `NonZero<usize>`, or a transparent wrapper chain
        // over an integer) transmuted into a transparent
        // pointer wrapper: the flattened integer is the
        // payload pointer's address -- the transmute
        // `NonNull::without_provenance` performs. A
        // fieldless-enum source (or a wrapper around one)
        // does not take this arm: its shape's leaf is the
        // enum itself, and the dedicated arms below serve it.
        if let (Some(mir::Ty::Adt(def, args)), mir::Ty::Adt(tdef, targs)) =
            (&src, target)
        {
            if let Some(shape) = self.wrapper_shape(tdef, targs) {
                if let Some(from) =
                    self.scalar_shape(&mir::Ty::Adt(*def, args.clone()), 0)
                {
                    if from.bits() == 64 && scalar_leaf_is_int(&from) {
                        return ir::Rvalue::ScalarToPtrWrapped {
                            operand: v.clone(),
                            from,
                            payload_index: shape.payload_index as u32,
                            field_count: shape.field_count,
                            fill: shape
                                .fill
                                .iter()
                                .map(|(i, c)| (*i as u32, c.clone()))
                                .collect(),
                        };
                    }
                }
            }
        }
let inner = match (&src, target) {
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
                    && self.cast_wrapper_entry_depth(s, t).is_some() =>
            {
                ir::Rvalue::PtrField(
                    v.clone(),
                    self.cast_wrapper_entry_depth(s, t)
                        .expect("guard checked"),
                )
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
            // Pointer <-> pointer where both pointees are
            // byte-string DSTs (`str`, `[u8]`, `OsStr`, `Path`):
            // all four are a byte string with a length metadata
            // half, and their values in this model are the byte
            // string itself (the projection pun reads an `OsStr`
            // / `Path` as its bytes), so the cast reinterprets
            // nothing — the `&[u8]` -> `&OsStr` step inside
            // `Component::as_os_str` is the shape that reaches
            // here, `is_fat_ptr_ty` not counting the DST structs
            // as fat pointees.
            (Some(s), t) if self.is_byte_string_ptr_pair(s, t) => {
                ir::Rvalue::Use(v.clone())
            }
            // Fat pointer between a `str` and a `[u8]` view of the
            // same bytes: both are stored over the same allocation
            // in this model (the `str::as_bytes` builtin relies on
            // the same sharing), so the pair carries over unchanged.
            (Some(s), t)
                if is_fat_ptr_ty(s)
                    && is_fat_ptr_ty(t)
                    && matches!(
                        (ptr_pointee(s), ptr_pointee(t)),
                        (Some(a), Some(b))
                            if is_str_or_byte_slice(a) && is_str_or_byte_slice(b)
                    ) =>
            {
                ir::Rvalue::Use(v.clone())
            }
            // Function pointer -> function pointer: the value
            // is the function identity, which does not depend on
            // the signature it is viewed through, so the value
            // carries over unchanged. This is the transmute the
            // `fmt` argument construction performs when it
            // rewrites a safe `fn(&T, &mut Formatter)` into the
            // type-erased `unsafe fn(NonNull<()>, &mut
            // Formatter)` stored in an `Argument` (inlined into
            // callers at higher opt levels).
            (Some(mir::Ty::FnPtr(..)), mir::Ty::FnPtr(..)) => {
                ir::Rvalue::Use(v.clone())
            }
            // Integer -> thin pointer: an address-only reference.
            (_, t) if src_is_int && is_thin_ptr_ty(t) => {
                ir::Rvalue::IntToPtr(v.clone())
            }
            // ADT -> ADT of identical pointer-carrying shape:
            // the value itself (see `ptr_identity` above).
            (Some(mir::Ty::Adt(..)), mir::Ty::Adt(..)) if ptr_identity => {
                ir::Rvalue::Use(v.clone())
            }
            // Scalar-shaped -> scalar-shaped of the same width:
            // the bit-level reinterpretation (float <-> integer
            // bits, `bool` / `char` <-> integer, scalar arrays,
            // one-payload wrappers over scalars). For the one
            // overlap with the arms below (a one-field struct ->
            // the integer of exactly its field's type) the result
            // is the same value: the field itself.
            (_, _) if generic.is_some() => generic.clone().unwrap(),
            // ADT -> integer through a newtype chain over a
            // thin pointer (the `NonNull` shape, whose payload
            // field is a `NonZero<*const T>`): the synthetic
            // address of the pointer at the bottom of the chain
            // (what `NonNull::as_ptr().addr()` and the packed
            // `io::Error` repr decode compute).
            (Some(mir::Ty::Adt(def, args)), mir::Ty::Int(_) | mir::Ty::Uint(_))
                if self
                    .ptr_chain_operand(body, operand, v, def, args)
                    .is_some() =>
            {
                let op = self
                    .ptr_chain_operand(body, operand, v, def, args)
                    .expect("guard checked");
                ir::Rvalue::PtrToInt(op)
            }
            // ADT -> thin pointer through the same chain: the
            // pointer at the bottom of the chain itself (what
            // `NonNull::as_ptr()` computes, and what a
            // `Box<T>` -> `*mut T` transmute reads out of the
            // box's `Unique` field).
            (Some(mir::Ty::Adt(def, args)), t)
                if is_thin_ptr_ty(t)
                    && self
                        .ptr_chain_operand(body, operand, v, def, args)
                        .is_some() =>
            {
                let op = self
                    .ptr_chain_operand(body, operand, v, def, args)
                    .expect("guard checked");
                ir::Rvalue::Use(op)
            }
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
                } else if let Some(shape) = self.wrapper_shape(def, args) {
                    // A transparent wrapper around a thin pointer
                    // (`NonNull`, `Unique`): transmuting it to an
                    // integer is the payload pointer's address —
                    // the transmute `fmt`'s statically-known-string
                    // check performs on its `NonNull<u8>`.
                    match self.field_operand(
                        body,
                        operand,
                        v,
                        shape.payload_index as u32,
                        &shape.payload_ty,
                    ) {
                        Some(op) => ir::Rvalue::PtrToInt(op),
                        None => unsupported(),
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
                } else if let Some(field_ty) = self.single_field_ty(def, args) {
                    // One-field struct wrapping a fieldless
                    // enum, from the discriminant integer (the
                    // `mem::Alignment` construction).
                    match &field_ty {
                        mir::Ty::Adt(fdef, _) => {
                            match self.fieldless_enum_discriminants(fdef) {
                                Some(table) => {
                                    ir::Rvalue::IntToEnumWrapped {
                                        operand: v.clone(),
                                        discriminants: table,
                                        payload_index: 0,
                                        field_count: 1,
                                        fill: Vec::new(),
                                    }
                                }
                                None => unsupported(),
                            }
                        }
                        _ => unsupported(),
                    }
                } else if let Some(payload) =
                    self.single_variant_payload(def, args)
                {
                    if matches!(payload, mir::Ty::Int(_) | mir::Ty::Uint(_)) {
                        ir::Rvalue::AggregateAdt {
                            variant: 0,
                            operands: vec![v.clone()],
                            field_index: None,
                            units_from: None,
                        }
                    } else {
                        unsupported()
                    }
                } else if let Some(shape) = self.wrapper_shape(def, args) {
                    // A transparent wrapper around a thin pointer
                    // (`NonNull`, `Unique`): the integer is the
                    // payload pointer's address, wrapped back up —
                    // the reverse of the wrapper-to-integer
                    // transmute above.
                    ir::Rvalue::IntToPtrWrapped {
                        operand: v.clone(),
                        payload_index: shape.payload_index as u32,
                        field_count: shape.field_count,
                        fill: shape
                            .fill
                            .iter()
                            .map(|(i, c)| (*i as u32, c.clone()))
                            .collect(),
                    }
                } else if let Some((table, shape)) =
                    self.enum_wrapper_shape(def, args)
                {
                    // A transparent wrapper around a fieldless
                    // enum (`mem::Alignment`): the integer is the
                    // discriminant of the enum value the wrapper
                    // holds — the reverse of the
                    // wrapper-to-integer transmute above.
                    ir::Rvalue::IntToEnumWrapped {
                        operand: v.clone(),
                        discriminants: table,
                        payload_index: shape.payload_index as u32,
                        field_count: shape.field_count,
                        fill: shape
                            .fill
                            .iter()
                            .map(|(i, c)| (*i as u32, c.clone()))
                            .collect(),
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
                            units_from: None,
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
            // Pointer wrapper -> pointer wrapper of another
            // pointee (`NonNull::cast` and its kin): the payload
            // pointer carries over, rewrapped in the target
            // wrapper's shape.
            (Some(mir::Ty::Adt(def, args)), mir::Ty::Adt(tdef, targs))
                if (self.wrapper_shape(def, args).is_some()
                    || self.fat_wrapper_shape(def, args).is_some())
                    && (self.wrapper_shape(tdef, targs).is_some()
                        || self.fat_wrapper_shape(tdef, targs).is_some()) =>
            {
                let src_shape = self
                    .wrapper_shape(def, args)
                    .or_else(|| self.fat_wrapper_shape(def, args))
                    .expect("guard checked");
                let dst_shape = self
                    .wrapper_shape(tdef, targs)
                    .or_else(|| self.fat_wrapper_shape(tdef, targs))
                    .expect("guard checked");
                match self.field_operand(
                    body,
                    operand,
                    v,
                    src_shape.payload_index as u32,
                    &src_shape.payload_ty,
                ) {
                    Some(op) => {
                        let mut operands =
                            Vec::with_capacity(dst_shape.field_count);
                        for i in 0..dst_shape.field_count {
                            if i == dst_shape.payload_index {
                                operands.push(op.clone());
                            } else {
                                let c = dst_shape
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
                        // A retarget at a differently sized
                        // pointee denominates the payload
                        // pointer's index in the source
                        // pointee's elements; rewrite it in
                        // byte units as the wrapper is built
                        // (see `Rvalue::PtrCastUnits`).
                        let units_from = match (
                            ptr_pointee(&src_shape.payload_ty)
                                .and_then(|t| self.ty_layout(t))
                                .map(|(size, _)| size),
                            ptr_pointee(&dst_shape.payload_ty)
                                .and_then(|t| self.ty_layout(t))
                                .map(|(size, _)| size),
                        ) {
                            (Some(f), Some(t)) if f != t && f > 0 => {
                                Some((f, dst_shape.payload_index))
                            }
                            _ => None,
                        };
                        ir::Rvalue::AggregateAdt {
                            variant: 0,
                            operands,
                            field_index: None,
                            units_from,
                        }
                    }
                    None => unsupported(),
                }
            }
            // Single-field struct wrapping an integer ->
            // fieldless enum: the field is the discriminant (the
            // `Bar { a }` -> `Foo` shape). Any other ADT -> ADT
            // pair falls through to the zero-sized rule below.
            (Some(mir::Ty::Adt(def, args)), mir::Ty::Adt(tdef, targs)) => {
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
                    // A one-field wrapper around a fieldless enum
                    // (`mem::Alignment`) transmuted to a
                    // transparent pointer wrapper (`NonNull`):
                    // the discriminant is the payload pointer's
                    // address — the transmute `NonNull::dangling`
                    // performs.
                    (Some(field_ty), None)
                        if self.wrapper_shape(tdef, targs).is_some() =>
                    {
                        let table = match &field_ty {
                            mir::Ty::Adt(fdef, _) => {
                                self.fieldless_enum_discriminants(fdef)
                            }
                            _ => None,
                        };
                        let shape = self
                            .wrapper_shape(tdef, targs)
                            .expect("guard checked");
                        match (
                            table,
                            self.field_operand(body, operand, v, 0, &field_ty),
                        ) {
                            (Some(table), Some(op)) => {
                                ir::Rvalue::EnumToPtrWrapped {
                                    operand: op,
                                    discriminants: table,
                                    payload_index: shape.payload_index as u32,
                                    field_count: shape.field_count,
                                    fill: shape
                                        .fill
                                        .iter()
                                        .map(|(i, c)| (*i as u32, c.clone()))
                                        .collect(),
                                }
                            }
                            _ => unsupported(),
                        }
                    }
                    // A fieldless enum itself transmuted to a
                    // transparent pointer wrapper: the same
                    // discriminant-as-address reading.
                    (None, None)
                        if self.fieldless_enum_discriminants(def).is_some()
                            && self.wrapper_shape(tdef, targs).is_some() =>
                    {
                        let table = self
                            .fieldless_enum_discriminants(def)
                            .expect("guard checked");
                        let shape = self
                            .wrapper_shape(tdef, targs)
                            .expect("guard checked");
                        ir::Rvalue::EnumToPtrWrapped {
                            operand: v.clone(),
                            discriminants: table,
                            payload_index: shape.payload_index as u32,
                            field_count: shape.field_count,
                            fill: shape
                                .fill
                                .iter()
                                .map(|(i, c)| (*i as u32, c.clone()))
                                .collect(),
                        }
                    }
                    // A fieldless enum transmuted to the
                    // one-field transparent struct wrapping it
                    // (the `mem::Alignment` shape: a struct whose
                    // single field is the enum itself): the value
                    // is the enum value, becoming the struct's
                    // single field.
                    (None, None)
                        if self.fieldless_enum_discriminants(def).is_some()
                            && self.single_field_ty(tdef, targs)
                                == Some(mir::Ty::Adt(def.clone(), args.to_vec())) =>
                    {
                        ir::Rvalue::AggregateAdt {
                            variant: 0,
                            operands: vec![v.clone()],
                            field_index: None,
                            units_from: None,
                        }
                    }
                    // The reverse direction: the one-field
                    // struct over a fieldless enum transmuted to
                    // the enum it wraps: the field itself.
                    (Some(field_ty), Some(_)) if field_ty == *target => {
                        match self.field_operand(body, operand, v, 0, &field_ty) {
                            Some(op) => ir::Rvalue::Use(op),
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
            // Array -> array of the same length across an
            // element wrapper boundary: `[T; N]` <->
            // `[MaybeUninit<T>; N]` (or the `UnsafeCell` twin).
            // The wrapper is transparent in this model -- its
            // payload is the element value at field 0 -- so the
            // array value, which stores its elements one by one,
            // carries over unchanged; only the element type the
            // destination declares changes. This is the transmute
            // an inlined array `IntoIter` construction performs
            // on the backing array.
            (Some(s), t) if self.array_elem_wrap_direction(s, t).is_some() => {
                match self.array_elem_wrap_direction(s, t) {
                    Some(true) => ir::Rvalue::WrapArrayElems(v.clone()),
                    _ => ir::Rvalue::UnwrapArrayElems(v.clone()),
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
        };
        // A transmute whose result the optimiser scalarised to a
        // bare pointer local (a `NonNull::cast` whose wrapper was
        // SROA'd away) reaches here as a plain `Use` of the
        // unwrapped pointer; the denomination rewrite the
        // wrapper-to-wrapper arm records on its aggregate applies
        // to this shape as well.
        if let ir::Rvalue::Use(op) = &inner {
            if let Some(from) = self.transmute_units_from(src.as_ref(), target) {
                return ir::Rvalue::PtrCastUnits(op.clone(), from);
            }
        }
        inner
    }

    /// The source element size for a pointer-valued transmute
    /// whose source and target pointees differ in size, when the
    /// source pointer is carried directly or inside a one-field
    /// pointer wrapper. Fat pointees keep their metadata path and
    /// never take part.
    fn transmute_units_from(
        &self,
        source: Option<&mir::Ty>,
        target: &mir::Ty,
    ) -> Option<u64> {
        let target_pointee = ptr_pointee(target)?;
        if matches!(
            target_pointee,
            mir::Ty::Slice(_) | mir::Ty::Str | mir::Ty::Dynamic(..)
        ) {
            return None;
        }
        let source_pointee = match source? {
            mir::Ty::RawPtr(p, _) | mir::Ty::Ref(_, p, _) => &**p,
            mir::Ty::Adt(def, args) => {
                let shape = self
                    .wrapper_shape(def, args)
                    .or_else(|| self.fat_wrapper_shape(def, args))?;
                match shape.payload_ty.clone() {
                    mir::Ty::RawPtr(p, _) | mir::Ty::Ref(_, p, _) => {
                        return self.transmute_units_from_pointees(&p, target);
                    }
                    _ => return None,
                }
            }
            _ => return None,
        };
        if matches!(
            source_pointee,
            mir::Ty::Slice(_) | mir::Ty::Str | mir::Ty::Dynamic(..)
        ) {
            return None;
        }
        self.transmute_units_from_pointees(source_pointee, target)
    }

    fn transmute_units_from_pointees(
        &self,
        source_pointee: &mir::Ty,
        target: &mir::Ty,
    ) -> Option<u64> {
        let target_pointee = ptr_pointee(target)?;
        if matches!(
            source_pointee,
            mir::Ty::Slice(_) | mir::Ty::Str | mir::Ty::Dynamic(..)
        ) {
            return None;
        }
        let (from, _) = self.ty_layout(source_pointee)?;
        let (to, _) = self.ty_layout(target_pointee)?;
        (from > 0 && from != to).then_some(from)
    }


    /// Whether `def` is `MaybeUninit` (std or core).
    fn is_maybe_uninit(&self, def: &mir::AdtDefId) -> bool {
        matches!(
            self.def_display_name(def).as_str(),
            "std::mem::MaybeUninit" | "core::mem::MaybeUninit"
        )
    }

    /// The operand denoting the payload of a one-payload
    /// wrapper value, when the wrapper's payload type is
    /// exactly `target`: a one-field struct whose only field
    /// has the target type, or a `MaybeUninit` of the target
    /// type. In this value model a `MaybeUninit<T>` holds its
    /// payload at field 0 of the union value -- the slot every
    /// pointer access to the union denotes -- so field 0 is
    /// the payload's slot for both shapes. `None` when the
    /// source is not such a wrapper, or when the operand
    /// carries no addressable payload field.
    fn payload_field_operand(
        &self,
        body: &mir::Body,
        operand: &mir::Operand,
        v: &ir::Operand,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
        target: &mir::Ty,
    ) -> Option<ir::Operand> {
        let payload_ty = if let Some(field_ty) = self.single_field_ty(def, args) {
            if field_ty != *target {
                return None;
            }
            field_ty
        } else if self.is_maybe_uninit(def) && args.len() == 1 && args[0] == *target
        {
            args[0].clone()
        } else {
            return None;
        };
        self.field_operand(body, operand, v, 0, &payload_ty)
    }

    /// The aggregate rvalue wrapping `v` into a one-payload
    /// wrapper of exactly its type: a one-field struct whose
    /// only field has the source type (the payload is the
    /// struct's only field), or a `MaybeUninit` of the source
    /// type (the payload goes to field 0 of the union value,
    /// the slot this model reads it back from). `None` when
    /// the target is not such a wrapper.
    fn wrap_payload_rvalue(
        &self,
        v: &ir::Operand,
        src: &mir::Ty,
        tdef: &mir::AdtDefId,
        targs: &[mir::Ty],
    ) -> Option<ir::Rvalue> {
        let field_index = if let Some(field_ty) = self.single_field_ty(tdef, targs) {
            if field_ty != *src {
                return None;
            }
            None
        } else if self.is_maybe_uninit(tdef)
            && targs.len() == 1
            && targs[0] == *src
        {
            Some(0)
        } else {
            return None;
        };
        Some(ir::Rvalue::AggregateAdt {
            variant: 0,
            operands: vec![v.clone()],
            field_index,
            units_from: None,
        })
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

    /// The direction in which an array -> array transmute of
    /// the same length crosses an element wrapper boundary:
    /// `Some(true)` when the target's element is the source's
    /// element wrapped in `MaybeUninit` or `UnsafeCell` (the two
    /// wrappers this model treats as transparent
    /// payload-at-field-0 carriers, as in
    /// [`Self::pointee_crosses_cell_wrapper`]), `Some(false)`
    /// for the reverse crossing, `None` when the pair does not
    /// cross the boundary.
    fn array_elem_wrap_direction(
        &self,
        src: &mir::Ty,
        target: &mir::Ty,
    ) -> Option<bool> {
        let (
            mir::Ty::Array(src_elem, src_len),
            mir::Ty::Array(target_elem, target_len),
        ) = (src, target)
        else {
            return None;
        };
        if !matches!(
            (src_len, target_len),
            (
                mir::ArrayLen::Known(a),
                mir::ArrayLen::Known(b)
            ) if a == b
        ) {
            return None;
        }
        let wraps = |wrapper: &mir::Ty, inner: &mir::Ty| -> bool {
            match wrapper {
                mir::Ty::Adt(def, args) => {
                    args.len() == 1
                        && args[0] == *inner
                        && matches!(
                            self.def_display_name(def).as_str(),
                            "core::mem::MaybeUninit"
                                | "std::mem::MaybeUninit"
                                | "core::cell::UnsafeCell"
                                | "std::cell::UnsafeCell"
                        )
                }
                _ => false,
            }
        };
        if wraps(target_elem, src_elem) {
            Some(true)
        } else if wraps(src_elem, target_elem) {
            Some(false)
        } else {
            None
        }
    }

    /// Whether a thin-pointer -> thin-pointer transmute crosses
    /// an `UnsafeCell` / `MaybeUninit` wrapper boundary: one
    /// side's pointee is the wrapper ADT and the other side's
    /// pointee is exactly the type the wrapper wraps. The
    /// entering direction (wrapper -> wrapped type) is composed
    /// with the cast conversion's field-entering treatment
    /// (`cast_wrapper_entry_depth` -> `Rvalue::PtrField`) in
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
            // A thin pointer flattens to its synthetic address.
            // This is what gives the pointer-carrying shapes a
            // scalar reading: a transparent wrapper over a thin
            // pointer (the `NonNull` chain) through the ADT arm
            // below, and a niche-encoded enum over one (the
            // `Option<NonNull<T>>` family) through
            // `niche_enum_shape`. Fat pointers carry a metadata
            // half this single-word shape cannot account for and
            // stay shapeless.
            mir::Ty::RawPtr(..) | mir::Ty::Ref(..) if is_thin_ptr_ty(ty) => {
                Some(S::Ptr)
            }
            mir::Ty::Array(elem, mir::ArrayLen::Known(n)) => {
                let elem = self.scalar_shape(elem, depth + 1)?;
                let total = elem.bits().checked_mul(*n as u32)?;
                if total == 0 || total > 128 {
                    return None;
                }
                Some(S::Array(Box::new(elem), *n))
            }
            mir::Ty::Adt(def, args) => {
                // A `MaybeUninit<T>` value holds its payload at
                // field 0 in this model -- the slot its pointer
                // accesses denote and its constructions store --
                // not at the declared `value` field's index, so a
                // transmute through the union flattens and builds
                // that slot: the payload's own shape behind a
                // one-field wrapper at index 0. Without a
                // scalar-shaped payload the union has no shape,
                // as before.
                if matches!(
                    self.def_display_name(def).as_str(),
                    "std::mem::MaybeUninit" | "core::mem::MaybeUninit"
                ) {
                    if args.len() != 1 {
                        return None;
                    }
                    let inner = self.scalar_shape(&args[0], depth + 1)?;
                    return Some(S::AdtField {
                        index: 0,
                        count: 1,
                        fill: Vec::new(),
                        inner: Box::new(inner),
                    });
                }
                let shape = self.adt_shapes.get(def)?;
                if shape.variants.len() != 1 {
                    if let Some(s) =
                        self.niche_enum_shape(def, args, shape, depth)
                    {
                        return Some(s);
                    }
                    return self.fieldless_enum_shape(def);
                }
                let fields = self.instantiated_field_tys(def, args, 0)?;
                // A single array field whose length is a const
                // parameter the type representation does not
                // carry (the `Simd<T, N>` storage field): the
                // instance's recorded layout size is the array's
                // byte size — a one-field struct's size is its
                // field's size, the field size already being a
                // multiple of the struct's alignment — so the
                // element count is the size over the element
                // width.
                if fields.len() == 1 {
                    if let mir::Ty::Array(elem, len) = &fields[0] {
                        if !matches!(len, mir::ArrayLen::Known(_)) {
                            if let Some(elem_shape) = self.scalar_shape(elem, depth + 1) {
                                let elem_bits = elem_shape.bits();
                                if elem_bits > 0 {
                                    let tag = crate::builtins::ty_tag(
                                        &mir::Ty::Adt(*def, args.to_vec()),
                                        self.names,
                                    );
                                    if let Some((size, _)) = self.adt_layouts.get(&tag) {
                                        let total = u128::from(*size) * 8;
                                        if total > 0
                                            && total <= 128
                                            && total % u128::from(elem_bits) == 0
                                        {
                                            return Some(S::AdtField {
                                                index: 0,
                                                count: 1,
                                                fill: Vec::new(),
                                                inner: Box::new(S::Array(
                                                    Box::new(elem_shape),
                                                    (total / u128::from(elem_bits)) as u64,
                                                )),
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
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

    /// The scalar shape of a niche-encoded enum instance: an
    /// enum whose layout stores the discriminant in the
    /// invalid values of its one payload field (see
    /// `mir::EnumNiche`), such as `Option<NonZero<T>>`. The
    /// shape applies only when the recorded encoding is
    /// exactly the one this value model reinterprets through:
    /// the niche scalar is the payload field itself (offset 0
    /// and the field's own width), the payload variant
    /// carries exactly that one field, and every other
    /// variant is fieldless and niche-encoded. Anything else
    /// — a niche inside a wider field, several payload
    /// variants — has no shape and stays unsupported.
    fn niche_enum_shape(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
        shape: &mir::AdtShape,
        depth: u32,
    ) -> Option<ir::ScalarShape> {
        let tag = crate::builtins::ty_tag(&mir::Ty::Adt(*def, args.to_vec()), self.names);
        let niche = self.adt_niches.get(&tag)?;
        if niche.niche_offset != 0 {
            return None;
        }
        let (first, last) = niche.niche_variants;
        let niche_count = last.checked_sub(first)? + 1;
        if shape.variants.len() as u32 != niche_count + 1 {
            return None;
        }
        let untagged = niche.untagged_variant as usize;
        if untagged >= shape.variants.len()
            || (first..=last).contains(&niche.untagged_variant)
        {
            return None;
        }
        for (i, fields) in shape.variants.iter().enumerate() {
            let expected = if i == untagged { 1 } else { 0 };
            if fields.len() != expected {
                return None;
            }
        }
        let fields = self.instantiated_field_tys(def, args, niche.untagged_variant)?;
        let [fty] = fields.as_slice() else {
            return None;
        };
        let inner = self.scalar_shape(fty, depth + 1)?;
        if inner.bits() == 0 || inner.bits() != niche.tag_bits {
            return None;
        }
        Some(ir::ScalarShape::NicheEnum {
            inner: Box::new(inner),
            untagged_variant: niche.untagged_variant,
            niche_variants: niche.niche_variants,
            niche_start: niche.niche_start,
        })
    }

    /// The scalar shape of a fieldless enum: the whole
    /// value is its discriminant, held in a tag whose width
    /// is the smallest standard integer width covering the
    /// discriminant range (the width rule the compiler's
    /// own tag selection follows for an enum without an
    /// explicit `repr`). An enum carrying an explicit wider
    /// `repr` is assigned the smaller inferred width, so a
    /// transmute pair against the wider integer does not
    /// take the generic path and stays with the dedicated
    /// discriminant arms.
    fn fieldless_enum_shape(
        &self,
        def: &mir::AdtDefId,
    ) -> Option<ir::ScalarShape> {
        let discriminants = self.fieldless_enum_discriminants(def)?;
        if discriminants.is_empty() {
            return None;
        }
        let lo = *discriminants.iter().min().expect("non-empty");
        let hi = *discriminants.iter().max().expect("non-empty");
        let bits = [8u32, 16, 32, 64, 128].into_iter().find(|w| {
            if *w == 128 {
                return true;
            }
            let w = *w;
            (lo >= 0 && (hi as u128) < (1u128 << w))
                || (lo >= -(1i128 << (w - 1)) && hi < (1i128 << (w - 1)))
        })?;
        Some(ir::ScalarShape::FieldlessEnum {
            bits,
            discriminants,
        })
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

    /// The transparent-wrapper shape whose payload field is a
    /// fieldless enum, together with that enum's discriminant
    /// table (`mem::Alignment` over its `AlignmentEnum` is the
    /// canonical case): the same one-payload rule as
    /// [`Self::wrapper_shape`], with the enum as the payload.
    fn enum_wrapper_shape(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
    ) -> Option<(Vec<i128>, WrapperShape)> {
        let shape = self.adt_shapes.get(def)?;
        if shape.kind != mir::AdtShapeKind::Struct || shape.variants.len() != 1 {
            return None;
        }
        let fields = self.instantiated_field_tys(def, args, 0)?;
        let mut payload: Option<(usize, mir::Ty, Vec<i128>)> = None;
        let mut fill = Vec::new();
        for (i, fty) in fields.iter().enumerate() {
            let table = match fty {
                mir::Ty::Adt(fdef, _) => self.fieldless_enum_discriminants(fdef),
                _ => None,
            };
            if let Some(table) = table {
                if payload.is_some() {
                    return None;
                }
                payload = Some((i, fty.clone(), table));
            } else if let Some(zv) = self.zst_value(fty, 0) {
                fill.push((i, value_to_constant(&zv)?));
            } else {
                return None;
            }
        }
        let (payload_index, payload_ty, table) = payload?;
        Some((
            table,
            WrapperShape {
                field_count: fields.len(),
                payload_index,
                payload_ty,
                fill,
            },
        ))
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
    /// The type a debugger variable's recorded projection reaches,
    /// starting from its local's declared type: the walk the
    /// debugger interface's variable table is built with, so an
    /// entry can carry the projected type's rendered tag. Only
    /// field and downcast steps are walked (the steps the table
    /// itself carries); anything else leaves the type unknown.
    fn debug_projected_ty(
        &self,
        base: &mir::Ty,
        proj: &[crate::RefProj],
    ) -> Option<mir::Ty> {
        let mut ty = base.clone();
        let mut variant = 0u32;
        for step in proj {
            match step {
                crate::RefProj::Downcast(v) => variant = *v as u32,
                crate::RefProj::Field(i) => {
                    ty = match &ty {
                        mir::Ty::Tuple(ts) => ts.get(*i)?.clone(),
                        mir::Ty::Array(elem, _) => (**elem).clone(),
                        mir::Ty::Adt(def, args) => self
                            .instantiated_field_tys(def, args, variant)?
                            .get(*i)?
                            .clone(),
                        _ => return None,
                    };
                }
                _ => return None,
            }
        }
        Some(ty)
    }

    fn instantiated_field_tys(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
        variant: u32,
    ) -> Option<Vec<mir::Ty>> {
        let shape = self.adt_shapes.get(def)?;
        let declared = shape.variants.get(variant as usize)?;
        let instance = self.instance_field_tys(def, args, variant);
        Some(
            declared
                .iter()
                .enumerate()
                .map(|(fi, fty)| {
                    self.adt_fields
                        .get(*def, args, variant, fi as u32)
                        .cloned()
                        .or_else(|| instance.and_then(|fs| fs.get(fi)).cloned())
                        .unwrap_or_else(|| subst_ty(fty, args))
                })
                .collect(),
        )
    }

    /// The driver-recorded field types of one variant of the ADT
    /// instance `def` instantiated with `args` (see
    /// `mir::AdtInstanceShapes`), or `None` when the driver
    /// recorded no entry for the instance — the common case, in
    /// which the generic shape's field types need no resolution
    /// beyond substitution.
    fn instance_field_tys(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
        variant: u32,
    ) -> Option<&[mir::Ty]> {
        let tag = crate::builtins::ty_tag(&mir::Ty::Adt(*def, args.to_vec()), self.names);
        self.adt_instance_shapes
            .get(&tag)?
            .variants
            .get(variant as usize)
            .map(Vec::as_slice)
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
            R::Use(op) => {
                // A whole-pointee load of a vector-shaped value
                // through a dereference: the shaped load, which
                // assembles the value from a raw block's elements
                // when the pointee is not stored whole.
                if let mir::Operand::Copy(place) | mir::Operand::Move(place) = op {
                    if let Some(shape) = self.deref_vector_shape(body, place) {
                        return Ok(ir::Rvalue::ShapedLoad(
                            self.convert_place(body, place)?,
                            shape,
                            matches!(op, mir::Operand::Move(_)),
                        ));
                    }
                }
                ir::Rvalue::Use(self.convert_operand(body, op, dest_ty)?)
            }
            R::BinaryOp(op, pair) if *op == mir::BinOp::Offset => {
                // `BinaryOp::Offset` is, like a shift, a binary
                // operation whose operands do not share a type: a
                // pointer and an element count. The operands are
                // typed separately the same way (the pointer takes
                // the operation's context — its place type, else the
                // result type, which is the pointer type itself; the
                // count keeps its own type).
                let (lhs, rhs) = self.convert_shift_operands(body, pair, dest_ty, false)?;
                let elem_size = self.offset_elem_size(body, pair, dest_ty);
                let count_width = match self.operand_place_ty(body, &pair.1) {
                    Some(mir::Ty::Uint(uty)) => uty.bits(),
                    _ => 0,
                };
                ir::Rvalue::PtrOffset(lhs, rhs, elem_size, count_width)
            }
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
                // The operand is read under its own (source) type,
                // never the cast's destination type: a constant
                // operand typed by the destination loses its
                // signedness (an `u128::MAX` operand of an
                // int-to-float cast was read as `-1i128` through
                // the float destination, which names neither).
                // `convert_operand` fills a constant's context
                // from the constant's own recorded type.
                let ctx = self.operand_place_ty(body, operand).or_else(|| {
                    match operand {
                        mir::Operand::Constant(_) => None,
                        _ => dest_ty.cloned(),
                    }
                });
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
                                        let concrete = self.concrete_ty_name(
                                            unsize_concrete_pointee(self, &src, target),
                                        );
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
                            None => {
                                let array_len = self
                                    .operand_place_ty(body, operand)
                                    .and_then(|src| {
                                        match unsize_pointee(&src) {
                                            mir::Ty::Array(
                                                _,
                                                mir::ArrayLen::Known(n),
                                            ) => Some(*n as u64),
                                            _ => None,
                                        }
                                    });
                                ir::Rvalue::Unsize(
                                    v,
                                    ir::UnsizeMeta::Slice { array_len },
                                )
                            }
                        }
                    }
                    // This coercion does not change the reference
                    // value in this model (a reference carries no
                    // mutability or pointee type at run time).
                    mir::CastKind::PointerCoercion(
                        mir::PointerCoercion::MutToConstPointer,
                    ) => ir::Rvalue::Use(v),
                    // An array-to-pointer coercion retargets a
                    // pointer to an array at its first element,
                    // exactly like the `PtrToPtr` cast below; the
                    // coercion kind itself guarantees the source is
                    // an array pointer and the target its element
                    // pointer, and the type check is a guard.
                    mir::CastKind::PointerCoercion(
                        mir::PointerCoercion::ArrayToPointer,
                    ) => {
                        if ctx
                            .as_ref()
                            .is_some_and(|src| self.cast_enters_first_element(src, target))
                        {
                            ir::Rvalue::PtrFirstElem(v)
                        } else {
                            ir::Rvalue::Use(v)
                        }
                    }
                    mir::CastKind::PtrToPtr => {
                        // A cast from a transparent single-field
                        // wrapper (`UnsafeCell<T>` / `MaybeUninit<T>`)
                        // to the wrapped type changes what the
                        // pointer denotes: the wrapper's only field.
                        // Every other pointer-to-pointer cast leaves
                        // the denoted place unchanged in this model.
                        let entry = ctx
                            .as_ref()
                            .and_then(|src| self.cast_entry_depth(src, target));
                        if let Some((depth, first_elem)) = entry {
                            if first_elem {
                                ir::Rvalue::PtrFieldFirstElem(v, depth)
                            } else {
                                ir::Rvalue::PtrField(v, depth)
                            }
                        } else if ctx.as_ref().is_some_and(|src| {
                            self.cast_enters_first_element(src, target)
                        }) {
                            // A cast from an array / slice / `str`
                            // pointer to an element pointer (what
                            // `as_ptr` returns and `&arr as *const T`
                            // writes) denotes the first element, not
                            // the aggregate as a whole. This precedes
                            // the `DataPtr` extraction below: dropping
                            // the metadata alone would leave the
                            // result denoting the whole aggregate.
                            ir::Rvalue::PtrFirstElem(v)
                        } else if ctx.as_ref().is_some_and(|src| {
                            self.cast_returns_to_array_base(src, target)
                        }) {
                            // The reverse cast: an element pointer
                            // cast back to a pointer to the whole
                            // array denotes the array as a whole.
                            ir::Rvalue::PtrArrayBase(v)
                        } else if let Some(f) = self.cast_units_from_size(&ctx, target) {
                            // A cast between differently sized
                            // pointees (see `cast_units_from_size`):
                            // the denoted position is unchanged,
                            // but its projection index is
                            // denominated in the source pointee's
                            // elements while later offsets use the
                            // target's, so the cast rewrites the
                            // index as a byte offset. For a thin
                            // source this is the cast's only
                            // effect; fat sources never reach
                            // here (the `DataPtr` path below owns
                            // them).
                            ir::Rvalue::PtrCastUnits(v, f)
                        } else if ptr_pointee(target).is_some_and(|p| {
                            // A pointer cast to a thin target drops
                            // the metadata word when the source value
                            // is fat (`*const [T]` to `*const T`, the
                            // shape of `<[T]>::as_ptr`): the result is
                            // the data pointer. `DataPtr` evaluates to
                            // the operand itself when the value is not
                            // fat, so it is correct for thin sources
                            // too -- including constant operands, whose
                            // source type this MIR form does not
                            // preserve (`ctx` is None). A byte-string
                            // DST struct target (`CStr` and its kin)
                            // is a fat target, not a thin one: its
                            // metadata is the byte length, and the
                            // cast keeps the value whole.
                            !matches!(
                                p,
                                mir::Ty::Slice(_) | mir::Ty::Str | mir::Ty::Dynamic(..)
                            ) && !self.pointee_is_byte_string_dst(p)
                        }) {
                            ir::Rvalue::DataPtr(v)
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
                    // closure value (an environment whose values
                    // the capture types do not determine stays an
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
                        // A constant operand carries no place type
                        // (the `ctx` above holds the cast's target
                        // type for it), so a pointer constant is
                        // judged from the converted constant itself:
                        // the constant forms that evaluate to a
                        // pointer value take the same `PtrToInt`
                        // reading, which yields the value's
                        // synthetic address exactly as for a place
                        // operand.
                        _ if matches!(target, mir::Ty::Int(_) | mir::Ty::Uint(_))
                            && is_pointer_constant(&v) =>
                        {
                            ir::Rvalue::PtrToInt(v.clone())
                        }
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
                    // A `MaybeUninit::uninit()` construction: the
                    // union aggregate activates the `uninit: ()`
                    // field, but in this value model a
                    // `MaybeUninit<T>` holds its payload at field
                    // 0 -- the slot its pointer accesses enter --
                    // and the payload of an uninitialized
                    // `MaybeUninit` is uninitialized memory of
                    // type `T`, not the unit value the field's
                    // declaration suggests. Storing the unit value
                    // there left every access that navigates
                    // inside the payload (an element write through
                    // `as_mut_ptr` into a `MaybeUninit<[T; N]>`,
                    // say) facing a unit where the payload's own
                    // shape belongs. Emit the value the model
                    // holds for an uninitialized `MaybeUninit`:
                    // the payload slot carrying the uninitialized
                    // skeleton of `T`, the same expansion an
                    // uninitialized constant of this type receives
                    // (see `expand_uninit_constant`).
                    if let Some(active) = field_index {
                        if self.is_maybe_uninit(def) {
                            // The active field's declared type, from
                            // the ADT's shape (the field table is
                            // recovered from projections, and the
                            // `uninit` field is never projected).
                            let active_ty = self
                                .adt_shapes
                                .get(def)
                                .and_then(|shape| {
                                    shape.variants.get(*variant_index as usize)
                                })
                                .and_then(|fields| fields.get(*active as usize))
                                .map(|fty| subst_ty(fty, generic_args));
                            if let (Some(payload_ty), Some(field_ty)) =
                                (generic_args.first(), active_ty)
                            {
                                if field_ty.is_unit() {
                                    let payload = self.expand_uninit_constant(
                                        &ir::Constant::Uninit,
                                        payload_ty,
                                        0,
                                    );
                                    return Ok(ir::Rvalue::Use(ir::Operand::Constant(
                                        ir::Constant::Adt {
                                            variant: 0,
                                            fields: vec![payload],
                                        },
                                    )));
                                }
                            }
                        }
                    }
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
                        units_from: None,
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
                mir::AggregateKind::CoroutineClosure { def, generic_args } => {
                    // An async closure's creation aggregate: the
                    // value is the closure environment, exactly as
                    // for a plain closure - what differs is only
                    // how a call on it is dispatched (the AsyncFn
                    // family creates the coroutine the closure's
                    // captures feed). The generic arguments pack
                    // the tail as [kind, signature, upvars tuple,
                    // by-reference captures fn-pointer], so the
                    // upvars tuple is the second-to-last argument
                    // (a plain closure's is the last).
                    let upvar_tys: Vec<mir::Ty> = match generic_args.iter().rev().nth(1) {
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
                mir::AggregateKind::RawPtr { data_ty, .. } => {
                    // `ptr::from_raw_parts` shape: one operand for a
                    // sized pointee (the aggregate is the pointer
                    // itself), two for an unsized one (data pointer
                    // plus metadata). Slice metadata is a length and
                    // becomes a fat reference; `dyn` metadata is a
                    // vtable pointer, whose concrete type is erased
                    // here, so it stays unsupported.
                    let data_ctx = mir::Ty::RawPtr(
                        Box::new(data_ty.clone()),
                        mir::Mutability::Not,
                    );
                    match (data_ty, ops.len()) {
                        (mir::Ty::Slice(_), 2) => {
                            let data =
                                self.convert_operand(body, &ops[0], Some(&data_ctx))?;
                            let meta = self.convert_operand(
                                body,
                                &ops[1],
                                Some(&mir::Ty::Uint(mir::UintTy::Usize)),
                            )?;
                            ir::Rvalue::RawPtrAggregate {
                                data,
                                meta: Some(meta),
                            }
                        }
                        (mir::Ty::Dynamic(..), _) => ir::Rvalue::Unsupported(
                            "Aggregate RawPtr (dyn metadata)".to_string(),
                        ),
                        (_, 1) => {
                            let data =
                                self.convert_operand(body, &ops[0], Some(&data_ctx))?;
                            ir::Rvalue::RawPtrAggregate { data, meta: None }
                        }
                        (_, 2) => {
                            // Any other two-operand shape (a sized
                            // pointee with unit metadata, a `str`
                            // pointer's length, ...): the metadata
                            // operand converts without a context and
                            // the evaluation decides its meaning
                            // (unit metadata is no metadata).
                            let data =
                                self.convert_operand(body, &ops[0], Some(&data_ctx))?;
                            let meta = self.convert_operand(body, &ops[1], None)?;
                            ir::Rvalue::RawPtrAggregate {
                                data,
                                meta: Some(meta),
                            }
                        }
                        _ => ir::Rvalue::Unsupported(format!(
                            "Aggregate RawPtr ({} operands)",
                            ops.len()
                        )),
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
            R::ThreadLocalRef(def) => ir::Rvalue::ThreadLocalRef(self.static_key(def)),
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
                // A whole-pointee store of a vector-shaped value
                // through a dereference: the shaped store, which
                // writes the value lane by lane when the pointee
                // lives in a raw element block.
                if let mir::Rvalue::Use(op) = rv {
                    if let Some(shape) = self.deref_vector_shape(body, place) {
                        return Ok(ir::Statement {
                            kind: ir::StatementKind::ShapedStore {
                                place: self.convert_place(body, place)?,
                                value: self.convert_operand(body, op, dest_ty.as_ref())?,
                                shape,
                            },
                        });
                    }
                }
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

    /// Whether the (unconverted) body's last declared parameter
    /// is a C variable argument list: the MIR-side form of
    /// [`callee_takes_va_list`], used while converting call
    /// arguments, before the callee's converted body exists.
    fn mir_body_takes_va_list(&self, body: &mir::Body) -> bool {
        if body.arg_count == 0 {
            return false;
        }
        match body.local_decls.get(body.arg_count).map(|d| &d.ty) {
            Some(mir::Ty::Adt(def, _)) => {
                let name = self.def_display_name(def);
                name == "std::ffi::VaList" || name == "core::ffi::VaList"
            }
            _ => false,
        }
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
            // The std allocation entry points have no collected
            // body (the builtin model pre-empts them), so they
            // never resolve here; a program with a
            // `#[global_allocator]` binds them to its own shims
            // instead, exactly as the linker binds the symbols
            // in a native binary.
            if let Some(key) = self.alloc_shim_route(func) {
                return Some(key);
            }
            return self.resolve_impl_callee(func);
        };
        // A `fmt` trait method call on a foreign `Self` type is
        // not bound to a body, even when one is stored under
        // the trait item's plain name: that body is a foreign
        // impl written against the real value layout, which
        // the value model does not reproduce. The call falls
        // through to the external/builtin conversion, whose
        // structural fmt engine is the faithful execution for
        // those types (see `fmt_call_self`).
        if self.fmt_foreign_self(&name, func) {
            return None;
        }
        // The in-place collect entry point (`std::vec::
        // in_place_collect::from_iter_in_place`) reaches the
        // program whenever a collected body materializes it as a
        // function-pointer constant — `SpecFromIter::from_iter`
        // calls it through exactly that shape, which the driver's
        // name-based collection never sees as a call target, so
        // the const-fn demand pass converts its body. That body
        // drives the raw allocation machinery the value model
        // does not serve and bottoms out in leaf specialization
        // calls (`SourceIter::as_inner` and friends) that have
        // no bodies at all. The builtin is the function's own
        // semantics in the model — drain the iterator into a
        // fresh `Vec`; the allocation reuse the real body
        // performs is unobservable — so it pre-empts the
        // collected body here, exactly as the registry pre-empts
        // collection for directly named calls. The same holds
        // for the nested specialization entry the builtin also
        // serves.
        if Self::vec_from_iter_nested_entry(&name) {
            return None;
        }
        let Some(plain) = self.unconverted.get(&name) else {
            // A name that is not itself a collected body denotes
            // a trait item the call names directly (the func
            // operand's `FnDef` is the trait method, with the
            // concrete `Self` among its generic arguments). A
            // clean collected impl body for that `Self` is the
            // faithful target — the same resolution
            // `resolve_impl_callee` performs for operand shapes
            // `resolve_callee` cannot name — and wins over the
            // name-based builtin model the run time would
            // otherwise serve the trait name with (a structural
            // `PartialEq` in place of a hand-written `eq`, say).
            // When no such body is clean, the name stands and
            // the existing fallbacks apply unchanged.
            if !self.bodies.contains_key(&name) {
                if let Some(key) = self.resolve_impl_callee(func) {
                    return Some(key);
                }
            }
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

    /// The guest allocation shim serving a call to one of the
    /// std allocation entry points, when the program defines a
    /// `#[global_allocator]` (see `alloc_shims`). A shim that
    /// itself failed conversion does not serve the call; the
    /// builtin model below does.
    fn alloc_shim_route(&self, func: &mir::Operand) -> Option<String> {
        let mir::Operand::Constant(c) = func else {
            return None;
        };
        let def = match &c.literal {
            mir::ConstKind::Ty(mir::Ty::FnDef(def, _)) => def,
            mir::ConstKind::Unevaluated { def, .. } => def,
            _ => return None,
        };
        let name = self.names.get(def)?;
        let symbol = match name.as_str() {
            "alloc::alloc::__rust_alloc" => "__rust_alloc",
            "alloc::alloc::__rust_dealloc" => "__rust_dealloc",
            "alloc::alloc::__rust_realloc" => "__rust_realloc",
            "alloc::alloc::__rust_alloc_zeroed" => "__rust_alloc_zeroed",
            _ => return None,
        };
        let shim = self.alloc_shims.get(symbol)?;
        if self.unconverted.contains_key(shim) {
            return None;
        }
        Some(shim.clone())
    }

    /// Whether the external/builtin conversion below would serve
    /// this call with a builtin: the callee's canonical name has a
    /// builtin model. Checked exactly as that conversion checks
    /// it, so a call is only ever diverted to a builtin that the
    /// external path would really produce.
    fn external_builtin(&self, func: &mir::Operand) -> bool {
        self.resolve_external_callee(func).is_some_and(|(name, _, _)| {
            let canonical = crate::builtins::canonical_name(&name);
            crate::builtins::lookup_qualified(&canonical).is_some()
                || crate::builtins::foreign_helper_lookup(&canonical).is_some()
        })
    }

    /// Whether a call to one of the iterator-consumer builtins
    /// (`Iterator::count` / `fold` / `for_each` / `collect`,
    /// `FromIterator::from_iter`) should be served by the
    /// collected instance body instead of the builtin: the
    /// builtins drive the modelled `next` state machine, which
    /// only the builtin-produced iterator states satisfy. For
    /// any other iterator type — adapters such as `Map` /
    /// `Filter`, whose values are genuine structs built by
    /// collected constructors — the instance body, including
    /// its specializations, is the faithful execution, so it
    /// wins whenever the program has one.
    fn consumer_prefers_body(
        &self,
        kind: crate::builtins::Builtin,
        func: &mir::Operand,
        raw_generic_args: &[mir::Ty],
    ) -> bool {
        let iter_ty = match kind {
            crate::builtins::Builtin::IteratorCount
            | crate::builtins::Builtin::IteratorFold
            | crate::builtins::Builtin::IteratorForEach
            | crate::builtins::Builtin::IteratorCollect => raw_generic_args.first(),
            crate::builtins::Builtin::FromIteratorFromIter => raw_generic_args.get(2),
            crate::builtins::Builtin::IteratorNextBack => raw_generic_args.first(),
            crate::builtins::Builtin::IteratorNext => raw_generic_args.first(),
            crate::builtins::Builtin::IntoIter => raw_generic_args.first(),
            _ => return false,
        };
        let Some(iter_ty) = iter_ty else {
            return false;
        };
        let tag = crate::builtins::ty_tag(iter_ty, self.names);
        // Hash-collection iteration is the inverse of the
        // consumer rule below: the builtin models many iterator
        // types, but a raw table's walk is not one of them, so
        // the body is preferred exactly for the hash shapes.
        match kind {
            crate::builtins::Builtin::IteratorNext => {
                return crate::builtins::hash_iterator_tag(&tag)
                    && self.resolve_body_callee(func).is_some();
            }
            crate::builtins::Builtin::IntoIter => {
                return crate::builtins::hash_collection_tag(&tag)
                    && self.resolve_body_callee(func).is_some();
            }
            _ => {}
        }
        let modelled = match kind {
            crate::builtins::Builtin::IteratorNextBack => {
                crate::builtins::iterator_next_back_modelled(&tag)
            }
            _ => crate::builtins::iterator_next_modelled(&tag),
        };
        if modelled {
            return false;
        }
        self.resolve_body_callee(func).is_some()
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
    /// When a callee name denotes a `fmt` trait method (the
    /// `Display`/`Debug`/radix family), the `Self` type the
    /// call is monomorphized at, in rendered form. The name
    /// comes in two shapes: the trait item
    /// (`std::fmt::Debug::fmt`, whose `Self` is the call's
    /// first generic argument) and the resolved impl instance
    /// (`<std::vec::Vec<i32> as std::fmt::Debug>::fmt`, whose
    /// `Self` is the key's subject). Foreign impl bodies of
    /// these methods are written against the real value layout
    /// (a `Vec`'s `Debug` reads its buffer pointer out of the
    /// `RawVec`), which the value model does not reproduce, so
    /// callers use this to keep such a call off any collected
    /// body: the structural fmt engine — the same engine that
    /// formats the type as an argument — serves it instead
    /// (see `builtins::fmt_trait_method`). A guest type's own
    /// `fmt` body stays the resolution, exactly as before.
    /// Whether a callee is a `fmt` trait method whose `Self`
    /// the structural fmt engine must serve: a foreign type,
    /// or a `dyn` trait object (whose concrete value the
    /// engine resolves through the fat reference, exactly as
    /// when the object is formatted as an argument). A guest
    /// type's own `fmt` body stays the resolution.
    fn fmt_foreign_self(&self, name: &str, func: &mir::Operand) -> bool {
        let Some(self_name) = self.fmt_call_self(name, func) else {
            return false;
        };
        let head = self_name.trim().trim_start_matches('&').trim_start();
        head.starts_with("dyn ") || !crate::builtins::guest_fmt_tag(&self_name)
    }

    fn fmt_call_self(&self, name: &str, func: &mir::Operand) -> Option<String> {
        if crate::builtins::fmt_trait_method(name).is_some() {
            let (_, _, raw_generic_args) = self.resolve_external_callee(func)?;
            return raw_generic_args
                .first()
                .map(|t| self.concrete_ty_name(t));
        }
        let rest = name.strip_prefix('<')?;
        let (self_part, trait_part) = rest.rsplit_once(" as ")?;
        let trait_path = trait_part.strip_suffix(">::fmt")?;
        if crate::builtins::fmt_trait_method(&format!("{trait_path}::fmt")).is_none() {
            return None;
        }
        Some(self_part.to_string())
    }

    fn find_impl_callee(&self, func: &mir::Operand) -> Option<String> {
        let (name, _, raw_generic_args) = self.resolve_external_callee(func)?;
        let (trait_path, method) = name.rsplit_once("::")?;
        if trait_path.is_empty() || method.is_empty() {
            return None;
        }
        // A `fmt` trait method on a foreign `Self` type is not
        // resolved to its collected impl body either, for the
        // reason `fmt_call_self` documents: the structural fmt
        // engine serves the call instead.
        if method == "fmt" && self.fmt_foreign_self(&name, func) {
            return None;
        }
        self.find_impl_key_for(trait_path, method, &raw_generic_args)
    }

    /// The collected impl body key for a trait method named by
    /// its trait path and method name, instantiated with
    /// `raw_generic_args` (whose first entry is the `Self`
    /// type): the resolution `find_impl_callee` performs for
    /// calls, shared with the `FnDef`-constant resolution in
    /// `fn_def_value_key`.
    fn find_impl_key_for(
        &self,
        trait_path: &str,
        method: &str,
        raw_generic_args: &[mir::Ty],
    ) -> Option<String> {
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
            let Some((self_part, trait_base, key_method)) =
                parse_impl_key(base).or_else(|| parse_impl_block_key(base))
            else {
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

    /// Whether `name` (a body key, possibly carrying an
    /// instance suffix) denotes one of the `Vec` collect
    /// specialization entries the `VecFromIterNested` builtin
    /// serves: the registry-named entries (the nested trait
    /// entry and the in-place entry, under either crate
    /// spelling), or the nested entry's impl-qualified form
    /// (`<std::vec::Vec<T> as std::vec::spec_from_iter_nested::
    /// SpecFromIterNested<T, I>>::from_iter`), which is how the
    /// collected impl body is keyed. The recognition of the
    /// impl-qualified form mirrors the interpreter's run-time
    /// name fallback for the same entry.
    fn vec_from_iter_nested_entry(name: &str) -> bool {
        Self::vec_from_iter_builtin_name(name).is_some()
    }

    /// The name a `VecFromIterNested` call for the entry
    /// `name` denotes is served under: the entry's own
    /// canonical name when the registry names it (the nested
    /// trait entry and the in-place entry), and the nested
    /// trait entry's name for the impl-qualified form, whose
    /// own spelling the registry does not carry — the
    /// interpreter resolves a builtin call by its name, so the
    /// call must carry a name the registry resolves.
    fn vec_from_iter_builtin_name(name: &str) -> Option<String> {
        let plain = name.split(" [").next().unwrap_or(name);
        let canonical = crate::builtins::canonical_name(plain);
        if matches!(
            crate::builtins::lookup_qualified(&canonical),
            Some(crate::builtins::Builtin::VecFromIterNested)
        ) {
            return Some(canonical);
        }
        if canonical.ends_with(">::from_iter")
            && canonical
                .contains("::spec_from_iter_nested::SpecFromIterNested")
            && crate::interp::impl_qualified_self_type(&canonical)
                .is_some_and(|target| {
                    target.starts_with("std::vec::Vec<")
                        || target.starts_with("alloc::vec::Vec<")
                })
        {
            return Some(
                "std::vec::spec_from_iter_nested::SpecFromIterNested::from_iter"
                    .to_string(),
            );
        }
        None
    }

    /// A call through a function-pointer constant naming one
    /// of the `Vec` collect specialization entries
    /// (`std::vec::in_place_collect::from_iter_in_place`, or
    /// the `SpecFromIterNested` impl): `SpecFromIter::from_iter`
    /// calls them through exactly this shape — the func operand
    /// is the decoded function-pointer value, not an
    /// `FnDef`-typed constant — so neither the driver's
    /// name-based collection nor the direct-call resolution
    /// ever sees the callee's name, and the call would
    /// otherwise dispatch through the value to the collected
    /// body, whose raw-allocation machinery the value model
    /// does not serve. The builtin is the entry's own semantics
    /// in the model (drain the iterator into a fresh `Vec`; the
    /// allocation management is unobservable), so the call
    /// converts to it directly. The stepping dispatch key is
    /// the iterator argument's own type, which is the entry's
    /// last generic argument.
    fn vec_from_iter_value_terminator(
        &self,
        body: &mir::Body,
        func: &mir::Operand,
        args: &[mir::Operand],
        destination: &mir::Place,
        target: Option<mir::BasicBlock>,
    ) -> Result<Option<ir::TerminatorKind>, ConvertError> {
        let mir::Operand::Constant(c) = func else {
            return Ok(None);
        };
        let mir::ConstKind::Value(mir::ConstValue::FnDef(def)) = &c.literal
        else {
            return Ok(None);
        };
        let Some(name) = self.def_names.get(def) else {
            return Ok(None);
        };
        let Some(canonical) = Self::vec_from_iter_builtin_name(name)
        else {
            return Ok(None);
        };
        if args.len() != 1 {
            return Ok(None);
        }
        let Some(iter_ty) = self.operand_place_ty(body, &args[0]) else {
            return Ok(None);
        };
        let iter_tag = crate::builtins::ty_tag(&iter_ty, self.names);
        let ir_arg = self.convert_operand(body, &args[0], Some(&iter_ty))?;
        Ok(Some(ir::TerminatorKind::CallBuiltin {
            name: canonical,
            generic_args: vec![iter_tag],
            args: vec![ir_arg],
            destination: self.convert_place(body, destination)?,
            target: target.map(|b| ir::BasicBlock(b.index())),
            drop_glue: None,
            cmp_discriminants: None,
            result_unsize: None,
        }))
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
            consumed_glue: None,
            cmp_discriminants: None,
            result_unsize: None,
            unique_elem: None,
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
    /// A `Box` call is the exception in the other direction: the
    /// box's method body is only the shell step (it releases the
    /// allocation; the pointee's destruction is the compiler's,
    /// done before the method runs), and drop elaboration emits
    /// this call form exactly when the pointee has been moved
    /// out of the box — the parts that remain are dropped by
    /// their own `Drop` terminators. The glue the conversion
    /// builds for a box call is therefore the trivial plan: the
    /// session rooted at the box place still releases the
    /// allocation and clears the place, and no pointee glue runs.
    ///
    /// One exception: when the dropped type's own glue routes
    /// its destructor step through the very body this call
    /// resolves to, the collected body IS that step and the call
    /// keeps it — the surrounding body runs the fields' glue
    /// itself (a compiler `drop_glue` shim does, through its own
    /// `Drop` terminators), which the full-glue conversion would
    /// run a second time.
    ///
    /// A second exception: a `Box` call that resolves to a
    /// collected body keeps it. The glue builder plans a box
    /// structurally (the pointee's plan at field 0), so the glue
    /// names no destructor step for the box itself — but the
    /// box's `Drop::drop` body is not the pointee's destruction:
    /// it only releases the allocation (the compiler destroys
    /// the pointee before the method runs). An async-destructor
    /// coroutine calls it as exactly that final shell step,
    /// after the pointee's async destruction has completed;
    /// expanding the call to the full glue would destroy the
    /// pointee a second time. (Modelled containers such as
    /// `Vec` keep the glue conversion: their collected bodies
    /// operate on the native buffer representation the value
    /// model replaces, so the glue is their only faithful
    /// route.) A call whose glue cannot be built falls through
    /// to the ordinary call conversion.
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
        // The `Box` shell-step exception (see the doc comment).
        if let mir::Ty::Adt(def, _) = &self_ty {
            let is_box = self
                .names
                .get(def)
                .or_else(|| self.def_names.get(def))
                .is_some_and(|n| n == "std::boxed::Box");
            if is_box && self.resolve_body_callee(func).is_some() {
                return None;
            }
        }
        let glue = self.drop_glue(&self_ty)?;
        // When the dropped type's own glue routes its destructor
        // step through a collected destructor body for this very
        // type, that body IS the step this call names (reached
        // directly, or through the trait item's dispatch body):
        // keep the ordinary call, whose execution runs the
        // destructor and nothing else. The surrounding body runs
        // the fields' glue itself (a compiler `drop_glue` shim
        // does, through its own `Drop` terminators), which the
        // full-glue conversion would run a second time. A type
        // whose glue the value model accounts for structurally
        // (a modelled container, a reference-counted handle)
        // names no such step, and the conversion below is its
        // faithful route.
        let dtor_key = match &self_ty {
            mir::Ty::Adt(def, args) => self
                .dtor_bodies
                .get(&format!("{self_ty:?}"))
                .cloned()
                .or_else(|| {
                    self.names
                        .get(def)
                        .or_else(|| self.def_names.get(def))
                        .and_then(|n| self.dtor_body(n, args))
                }),
            _ => None,
        };
        if let Some(key) = dtor_key {
            if glue_calls_dtor(&glue, &key) {
                return None;
            }
        }
        // The `Box` shell step (see the doc comment): the
        // method body only releases the allocation, so the glue
        // is the trivial plan — the session rooted at the box
        // place releases the allocation and clears the place,
        // and no pointee glue runs.
        let glue = if let mir::Ty::Adt(def, _) = &self_ty {
            let is_box = self
                .names
                .get(def)
                .or_else(|| self.def_names.get(def))
                .is_some_and(|n| n == "std::boxed::Box");
            if is_box {
                ir::DropGlue {
                    root: ir::DropPlan::Trivial,
                    adts: Vec::new(),
                }
            } else {
                glue
            }
        } else {
            glue
        };
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
        // A call to an `Fn`-trait method takes the receiver and
        // the argument tuple; a third argument is the
        // caller-location stamp the driver appended for a
        // `#[track_caller]` shim instance (a track-caller
        // closure), which the `CallTrait` conversion carries
        // along instead of dropping the call.
        let trait_call = self
            .resolve_fn_trait_call(func)
            .filter(|_| arg_count == 2 || arg_count == 3);
        let stamped = arg_count == 3;
        let direct = self.resolve_body_callee(func);
        match (trait_call, direct) {
            (Some(trait_call), Some(name)) => {
                let mis_keyed = self
                    .bodies
                    .get(&name)
                    .is_some_and(|b| body_is_closure_body(b));
                // When the callable is a closure or a trait
                // object, the receiver value carries the
                // callable and the `CallTrait` dispatch owns
                // the call: a body registered under the trait
                // method's name is one instance's body (a shim,
                // a forwarding impl), not the dispatch, and the
                // interpreter's vtable dispatch refuses to treat
                // it as one. A concrete ADT `Self` instead names
                // a hand-written `Fn` impl, whose collected body
                // the direct resolution keeps.
                let value_keyed = trait_call
                    .1
                    .first()
                    .is_some_and(fn_self_is_closure_or_dyn)
                    // The `AsyncFn` family is implemented only by
                    // async closure values, so the callable is the
                    // receiver value itself. The check above cannot
                    // see it: the call's `Self` type is a
                    // coroutine-closure type, which the driver
                    // records as the unit placeholder (its kind is
                    // unaskable through Stable MIR), so the kind of
                    // the call is the witness instead.
                    || matches!(
                        trait_call.0,
                        ir::FnCallKind::AsyncCall
                            | ir::FnCallKind::AsyncCallMut
                            | ir::FnCallKind::AsyncCallOnce
                    );
                if mis_keyed || value_keyed {
                    Some(FnResolution::Trait(
                        trait_call.0,
                        trait_call.1,
                        stamped,
                    ))
                } else {
                    Some(FnResolution::Direct(name))
                }
            }
            (Some(trait_call), None) => {
                Some(FnResolution::Trait(
                    trait_call.0,
                    trait_call.1,
                    stamped,
                ))
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
            "std::ops::AsyncFn::async_call" | "core::ops::AsyncFn::async_call" => {
                ir::FnCallKind::AsyncCall
            }
            "std::ops::AsyncFnMut::async_call_mut"
            | "core::ops::AsyncFnMut::async_call_mut" => ir::FnCallKind::AsyncCallMut,
            "std::ops::AsyncFnOnce::async_call_once"
            | "core::ops::AsyncFnOnce::async_call_once" => {
                ir::FnCallKind::AsyncCallOnce
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
                for method in ["drop", "pin_drop"] {
                    let key = format!("<{instantiated} as {trait_path}>::{method}");
                    if self.bodies.contains_key(&key) {
                        return Some(key);
                    }
                }
            }
        }
        for trait_path in ["std::ops::Drop", "core::ops::Drop"] {
            for method in ["drop", "pin_drop"] {
                let key = format!("<{adt_name} as {trait_path}>::{method}");
                if self.bodies.contains_key(&key) {
                    return Some(key);
                }
            }
        }
        let candidates: Vec<&String> = self
            .bodies
            .keys()
            .filter(|k| is_dtor_name(k, adt_name))
            .collect();
        match candidates.len() {
            1 => Some(candidates[0].clone()),
            0 => None,
            _ => {
                // Several monomorphizations of one generic impl
                // coexist: prefer the candidate whose rendered
                // arguments match the requested ones. The
                // driver's instance names also render lifetime
                // arguments (as `'_`), which the requested
                // argument list does not carry, so lifetimes are
                // skipped on the candidate side.
                let want: Vec<String> = args
                    .iter()
                    .map(|t| crate::builtins::ty_tag(t, self.names))
                    .collect();
                let matching: Vec<&&String> = candidates
                    .iter()
                    .filter(|k| {
                        let Some(rest) = k.strip_prefix('<') else {
                            return false;
                        };
                        let Some((self_part, _)) = rest.split_once(" as ") else {
                            return false;
                        };
                        let Some(open) = self_part.find('<') else {
                            return want.is_empty();
                        };
                        let Some(close) = self_part.rfind('>') else {
                            return false;
                        };
                        let inside = &self_part[open + 1..close];
                        let mut parts: Vec<String> = Vec::new();
                        let mut depth = 0i32;
                        let mut current = String::new();
                        for c in inside.chars() {
                            match c {
                                '<' | '(' | '[' => {
                                    depth += 1;
                                    current.push(c);
                                }
                                '>' | ')' | ']' => {
                                    depth -= 1;
                                    current.push(c);
                                }
                                ',' if depth == 0 => {
                                    parts.push(current.trim().to_string());
                                    current = String::new();
                                }
                                _ => current.push(c),
                            }
                        }
                        if !current.trim().is_empty() {
                            parts.push(current.trim().to_string());
                        }
                        let parts: Vec<&String> = parts
                            .iter()
                            .filter(|a| {
                                !a.starts_with('\'')
                            })
                            .collect();
                        parts.len() == want.len()
                            && parts
                                .iter()
                                .zip(want.iter())
                                .all(|(a, b)| a == &b)
                    })
                    .collect();
                match matching.len() {
                    1 => Some((*matching[0]).clone()),
                    _ => None,
                }
            }
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
                    Some(self.convert_place(body, place)?)
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
                } else if let Some(term) = self.vec_from_iter_value_terminator(
                    body,
                    func,
                    args,
                    destination,
                    *target,
                )? {
                    term
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
                        FnResolution::Trait(kind, raw_generic_args, stamped) => {
                            // An `AsyncFn` call's receiver is typed
                            // by nothing the conversion can use: its
                            // `Self` generic argument is the
                            // coroutine-closure type the driver
                            // records as the unit placeholder, and
                            // typing the receiver operand by it
                            // would misread the reference the
                            // operand names. The operand's own
                            // place type types it instead.
                            let receiver_ty = if matches!(
                                kind,
                                ir::FnCallKind::AsyncCall
                                    | ir::FnCallKind::AsyncCallMut
                                    | ir::FnCallKind::AsyncCallOnce
                            ) {
                                None
                            } else {
                                raw_generic_args.first()
                            };
                            let receiver =
                                self.convert_operand(body, &args[0], receiver_ty)?;
                            let tuple = self
                                .convert_operand(body, &args[1], raw_generic_args.get(1))?;
                            // The stamp is carried for a direct or
                            // generic call, whose shim forwards
                            // the call-site location. For a `dyn`
                            // receiver the call passes through the
                            // vtable shim, which presents its own
                            // location instead (see
                            // `ir::Program::fn_shim_locations`), so
                            // the appended stamp is dropped here.
                            let stamp = if stamped
                                && !raw_generic_args
                                    .first()
                                    .is_some_and(|t| ty_mentions_dyn(t))
                            {
                                Some(self.convert_operand(body, &args[2], None)?)
                            } else {
                                None
                            };
                            // The capture count of the coroutine
                            // an `AsyncFn` call creates: the
                            // future type's last generic argument
                            // is the coroutine's upvars tuple - the
                            // state-0 fields the creation aggregate
                            // writes - and the fields begin with
                            // the call arguments, so the captures
                            // are what remains.
                            let async_captures = if matches!(
                                kind,
                                ir::FnCallKind::AsyncCall
                                    | ir::FnCallKind::AsyncCallMut
                                    | ir::FnCallKind::AsyncCallOnce
                            ) {
                                let arg_count = match raw_generic_args.get(1) {
                                    Some(mir::Ty::Tuple(ts)) => ts.len(),
                                    _ => 0,
                                };
                                self.place_ty(body, destination)
                                    .and_then(|t| match t {
                                        mir::Ty::Coroutine(_, cargs) => {
                                            match cargs.last() {
                                                Some(mir::Ty::Tuple(ts)) => {
                                                    Some(ts.len())
                                                }
                                                _ => None,
                                            }
                                        }
                                        _ => None,
                                    })
                                    .and_then(|fields| {
                                        fields.checked_sub(arg_count)
                                    })
                            } else {
                                None
                            };
                            ir::TerminatorKind::CallTrait {
                                kind,
                                receiver,
                                args: tuple,
                                stamp,
                                async_captures,
                                receiver_glue: if matches!(
                                    kind,
                                    ir::FnCallKind::CallOnce
                                ) {
                                    raw_generic_args
                                        .first()
                                        .and_then(|t| self.drop_glue(t))
                                } else {
                                    None
                                },
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
                                // Only a declared parameter types a
                                // call argument. The parameters that
                                // name call arguments are locals
                                // 1..=arg_count, except a trailing
                                // `VaList` parameter of a C-variadic
                                // callee, which names no single
                                // argument (the whole tail is packed
                                // into it at the call boundary); and
                                // an argument past the last parameter
                                // (a `#[track_caller]` stamp) names
                                // none either. Typing such an
                                // argument by the local at its index
                                // — an unrelated temporary of the
                                // callee — corrupts literal arguments
                                // (an integer literal read under a
                                // `bool` temporary collapses to
                                // `true`), so it is typed by its own
                                // recorded type instead, the fallback
                                // `convert_operand` applies when no
                                // context is given.
                                let param_ctx = |b: &&mir::Body| {
                                    let named = if self.mir_body_takes_va_list(b) {
                                        b.arg_count - 1
                                    } else {
                                        b.arg_count
                                    };
                                    if i < named {
                                        b.local_decls.get(i + 1).map(|d| d.ty.clone())
                                    } else {
                                        None
                                    }
                                };
                                let ctx = if i == 0 {
                                    self_ctx.clone().or_else(|| callee.and_then(param_ctx))
                                } else {
                                    callee.and_then(param_ctx)
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
                    let kind = crate::builtins::lookup_qualified(&canonical)
                        .filter(|kind| {
                            !self.consumer_prefers_body(*kind, func, &raw_generic_args)
                        })
                        // A body-less foreign callee (an extern
                        // declaration) may be served by the
                        // foreign-helper models; they are not in
                        // `lookup`, so no collected body is ever
                        // displaced by them.
                        .or_else(|| crate::builtins::foreign_helper_lookup(&canonical))
                        // A UFCS `fmt` trait call whose `Self`
                        // instance has no converted body is served
                        // by the structural fmt engine (see
                        // `builtins::fmt_trait_method`); a collected
                        // impl body has already won above, exactly
                        // as for the foreign helpers. A `dyn`
                        // `Self` converts here too: the engine
                        // resolves the trait object to its
                        // concrete value, exactly as it does for
                        // a `dyn` argument.
                        .or_else(|| {
                            if raw_generic_args.first().is_some() {
                                crate::builtins::fmt_trait_method(&canonical)
                            } else {
                                None
                            }
                        });
                    if let Some(kind) = kind {
                        let mut ir_args = Vec::with_capacity(args.len());
                        for (i, arg) in args.iter().enumerate() {
                            let ctx = builtin_arg_ctx(kind, &raw_generic_args, i);
                            ir_args.push(self.convert_operand(body, arg, ctx.as_ref())?);
                        }
                        // The driver appends the caller-location
                        // stamp to a call of a `#[track_caller]`
                        // callee (a trait method like
                        // `Index::index`, whose resolved impl is
                        // track-caller through the trait). A body
                        // serving the call binds the stamp to its
                        // synthetic local; a builtin that does not
                        // consume the location keeps its declared
                        // argument list, so the stamp is dropped
                        // here.
                        if !crate::builtins::consumes_caller_location(kind)
                            && ir_args.len() > 1
                            && args.last().is_some_and(is_caller_location_stamp)
                        {
                            ir_args.pop();
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
                        // A modelled consumer of an owned
                        // iterator drops the iterator inside the
                        // callee's real body: build the iterator
                        // type's glue now (see `consumed_glue` in
                        // `ir`), so the interpreter can run it
                        // against the iterator's final state. The
                        // iterator type is the callee generic
                        // argument the interpreter's dispatch
                        // reads as the iterator type. A trivial
                        // glue has nothing to run; a glue that
                        // cannot be accounted for keeps the plain
                        // call, the treatment these calls have
                        // always had.
                        let consumed_glue = match kind {
                            crate::builtins::Builtin::IteratorCollect
                            | crate::builtins::Builtin::IteratorCount
                            | crate::builtins::Builtin::IteratorFold
                            | crate::builtins::Builtin::IteratorForEach => {
                                raw_generic_args.first()
                            }
                            crate::builtins::Builtin::FromIteratorFromIter => {
                                raw_generic_args.get(2)
                            }
                            crate::builtins::Builtin::VecFromIterNested => {
                                raw_generic_args.last()
                            }
                            _ => None,
                        }
                        .and_then(|ty| self.drop_glue(ty))
                        .filter(|glue| glue.root != ir::DropPlan::Trivial);
                        // A `box_assume_init_into_vec_unsafe` call
                        // asserts its source array is fully
                        // initialized; an element slot that still
                        // holds the uninitialized marker is one
                        // whose write was elided because the
                        // element type has exactly one value. Bake
                        // that value while the type is at hand
                        // (see `into_vec_elem_value`); the builtin
                        // materializes the unwritten slots from
                        // it.
                        let unique_elem = if kind
                            == crate::builtins::Builtin::BoxAssumeInitIntoVec
                        {
                            args.first()
                                .and_then(|arg| self.into_vec_elem_value(body, arg))
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
                                consumed_glue,
                                cmp_discriminants,
                                result_unsize,
                                unique_elem,
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
                let glue = self.place_ty(body, place).and_then(|t| {
                    // A coroutine-closure value's place type is
                    // the unit placeholder, whose glue is trivial;
                    // the value's own glue is its captures' glue.
                    // When the place is the bare local this body's
                    // coroutine-closure aggregate created, plan it
                    // from the aggregate, exactly as a plain
                    // closure type's glue is planned from its
                    // capture tuple.
                    if place.projection.is_empty()
                        && matches!(&t, mir::Ty::Tuple(ts) if ts.is_empty())
                    {
                        if let Some((def, upvars)) =
                            coroutine_closure_creation(body, place.local.index())
                        {
                            let closure_ty =
                                mir::Ty::Closure(def, vec![mir::Ty::Tuple(upvars)]);
                            if let Some(glue) = self.drop_glue(&closure_ty) {
                                return Some(glue);
                            }
                        }
                    }
                    self.drop_glue(&t)
                });
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
/// or to a fixed type (`Argument::from_usize`'s count is a
/// `usize`, passed by reference).
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

/// Whether an operand is the caller-location stamp the driver
/// appends to a `#[track_caller]` call: a constant reference to
/// the `Location` aggregate — its (file, line, column) fields,
/// with line and column 4-byte integers.
fn is_caller_location_stamp(op: &mir::Operand) -> bool {
    let mir::Operand::Constant(c) = op else {
        return false;
    };
    let mir::ConstKind::Value(mir::ConstValue::Ref(inner)) = &c.literal else {
        return false;
    };
    let mir::ConstValue::Adt { fields, .. } = &**inner else {
        return false;
    };
    let is_u32 = |v: &mir::ConstValue| {
        matches!(v, mir::ConstValue::Scalar(mir::Scalar::Int { size: 4, .. }))
    };
    fields.len() >= 3 && is_u32(&fields[1]) && is_u32(&fields[2])
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
        // The iterator consumers take the iterator itself, the
        // accumulator and the closure as their arguments: each is
        // typed by the callee's corresponding generic parameter
        // (Self, B, F for fold; Self, F for for_each; Self for
        // count), so a literal accumulator takes the accumulator
        // type instead of the default unsigned rendering.
        Builtin::IteratorCount
        | Builtin::IteratorFold
        | Builtin::IteratorForEach
        | Builtin::IteratorCollect => raw_generic_args.get(arg_index).cloned(),
        Builtin::FromIteratorFromIter => {
            if arg_index == 0 {
                raw_generic_args.get(2).cloned()
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

/// Whether the context type is a pointer (a reference or a raw
/// pointer) to a byte slice: a byte constant in that position
/// denotes the fat pointer itself, not the pointee data. A bare
/// slice context (a trailing unsized field inside an ADT
/// constant, the `CStr` shape) does not qualify — there the
/// bytes are the field's data.
fn ctx_is_u8_slice_ptr(ctx: Option<&mir::Ty>) -> bool {
    match ctx {
        Some(mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _)) => {
            matches!(
                &**inner,
                mir::Ty::Slice(elem)
                    if matches!(&**elem, mir::Ty::Uint(mir::UintTy::U8))
            )
        }
        _ => false,
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
        // The coroutine resume guards: rustc lowers these
        // asserts to the `panic_const_*` entry points whose
        // payloads are the exact strings below (the wording a
        // `catch_unwind` observer downcasts and compares), not
        // to a generic assertion failure.
        A::ResumedAfterReturn(kind) => match kind {
            mir::CoroutineKind::Async => {
                "`async fn` resumed after completion".to_string()
            }
            mir::CoroutineKind::Gen => {
                "`gen fn` should just keep returning `None` after completion".to_string()
            }
            mir::CoroutineKind::Coroutine => {
                "coroutine resumed after completion".to_string()
            }
        },
        A::ResumedAfterPanic(kind) => match kind {
            mir::CoroutineKind::Async => {
                "`async fn` resumed after panicking".to_string()
            }
            mir::CoroutineKind::Gen => {
                "`gen fn` should just keep returning `None` after panicking".to_string()
            }
            mir::CoroutineKind::Coroutine => {
                "coroutine resumed after panicking".to_string()
            }
        },
        A::ResumedAfterDrop(kind) => match kind {
            mir::CoroutineKind::Async => {
                "`async fn` resumed after async drop".to_string()
            }
            mir::CoroutineKind::Gen => {
                "`gen fn` resumed after async drop".to_string()
            }
            mir::CoroutineKind::Coroutine => {
                "coroutine resumed after async drop".to_string()
            }
        },
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
/// `Offset` (pointer arithmetic) is converted separately into
/// `ir::Rvalue::PtrOffset` (a `CheckedBinaryOp` with `Offset`, the
/// only way this function could still see it, is never emitted), so
/// it yields `None` here.
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
        M::Cmp => ir::BinOp::Cmp,
        M::Offset => return None,
        // *WithOverflow in a plain position is handled by `overflow_base`
        // before this function is consulted.
        M::AddWithOverflow | M::SubWithOverflow | M::MulWithOverflow => return None,
    })
}

/// The size in bytes of a `mir` type whose layout is fixed by the
/// type alone: scalars, arrays of such types with a known length,
/// and pointers (fat when the pointee is unsized). Anything whose
/// layout needs the compiler's layout computation — ADTs, tuples
/// with more than one element, slices, `str` — yields `None`.
fn mir_ty_size(ty: &mir::Ty) -> Option<u64> {
    use mir::Ty as T;
    Some(match ty {
        T::Bool => 1,
        T::Char => 4,
        T::Int(t) => u64::from(t.bits() / 8),
        T::Uint(t) => u64::from(t.bits() / 8),
        T::Float(t) => u64::from(t.bits() / 8),
        T::Array(elem, mir::ArrayLen::Known(n)) => n.checked_mul(mir_ty_size(elem)?)?,
        T::Array(_, _) => return None,
        T::RawPtr(pointee, _) | T::Ref(_, pointee, _) => {
            if matches!(&**pointee, T::Slice(_) | T::Str | T::Dynamic(..)) {
                16
            } else {
                8
            }
        }
        T::FnPtr(_) => 8,
        T::Tuple(ts) if ts.len() <= 1 => match ts.first() {
            Some(t) => mir_ty_size(t)?,
            None => 0,
        },
        // A multi-field tuple's size is its native layout:
        // fields in declaration order, each aligned to its own
        // alignment, the total rounded up to the widest field
        // alignment — the layout the compiler's `size_of` and
        // the allocation layouts the driver records both use,
        // and the stride pointer arithmetic over a tuple-typed
        // pointee (a hashbrown bucket pointer) is computed in.
        // A field whose layout is not structurally known (an
        // ADT, whose layout lives in the layout table the free
        // function cannot see) leaves the size unknown.
        T::Tuple(_) => {
            fn layout(ty: &mir::Ty) -> Option<(u64, u64)> {
                Some(match ty {
                    T::Bool => (1, 1),
                    T::Char => (4, 4),
                    T::Int(t) => {
                        let b = u64::from(t.bits() / 8);
                        (b, b)
                    }
                    T::Uint(t) => {
                        let b = u64::from(t.bits() / 8);
                        (b, b)
                    }
                    T::Float(t) => {
                        let b = u64::from(t.bits() / 8);
                        (b, b)
                    }
                    T::Array(elem, mir::ArrayLen::Known(n)) => {
                        let (es, ea) = layout(elem)?;
                        (es.checked_mul(*n)?, ea)
                    }
                    T::RawPtr(pointee, _) | T::Ref(_, pointee, _) => {
                        if matches!(
                            &**pointee,
                            T::Slice(_) | T::Str | T::Dynamic(..)
                        ) {
                            (16, 8)
                        } else {
                            (8, 8)
                        }
                    }
                    T::FnPtr(_) => (8, 8),
                    T::Tuple(ts) => {
                        let mut size = 0u64;
                        let mut align = 1u64;
                        for f in ts {
                            let (fs, fa) = layout(f)?;
                            align = align.max(fa);
                            size = size.checked_add(fa - 1)? / fa * fa;
                            size = size.checked_add(fs)?;
                        }
                        (size.checked_add(align - 1)? / align * align, align)
                    }
                    T::Pat(inner, _) => layout(inner)?,
                    T::Never => (0, 1),
                    _ => return None,
                })
            }
            layout(ty)?.0
        }
        T::Pat(inner, _) => mir_ty_size(inner)?,
        T::Never => 0,
        _ => return None,
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

/// Parse an impl body key in the impl-block path form the
/// driver records for impls inside a module (foreign crates in
/// particular): `some::module::<impl Trait<..> for Self>::method`.
/// Returns the `Self` section, the trait path without generic
/// arguments, and the method name, as [`parse_impl_key`] does
/// for the `<Self as Trait>::method` form.
fn parse_impl_block_key(base: &str) -> Option<(String, String, String)> {
    let (head, method) = base.rsplit_once(">::")?;
    if method.is_empty() || method.contains("::") {
        return None;
    }
    // The `>::` separator consumed the impl's closing `>`, so
    // `head` ends with the `Self` section itself.
    let marker = head.rfind("::<impl ")?;
    let inner = &head[marker + "::<impl ".len()..];
    let (trait_part, self_part) = inner.split_once(" for ")?;
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
    // The queried name may itself carry a generic argument list
    // (a fully rendered concrete type); only its base takes part
    // in the comparison, as the key side's list is stripped above.
    let self_name = self_name.split('<').next().unwrap_or(self_name);
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
        // `pin_drop` is the same trait's destructor entry point
        // under pin ergonomics: a type that provides it has no
        // `drop` method, and the compiler's glue for dropping
        // the type is a call to it.
        let suffix = ["drop", "pin_drop"]
            .iter()
            .find_map(|m| name.strip_suffix(&format!(" as {trait_path}>::{m}")));
        let Some(head) = suffix else {
            continue;
        };
        // The head must be exactly the ADT's own rendering:
        // `<{adt_name}` or `<{adt_name}<args>` with the argument
        // group balanced and closing at the head's very end. A
        // type defined inside the ADT's own items renders
        // `<{adt_name}<..>::method::Local<..>`: its head also
        // starts with `<{adt_name}<` and ends with `>`, but the
        // first group closes long before the end — and the
        // nested type's destructor is not the ADT's, so the
        // resemblance must not attribute it to the ADT.
        let Some(rest) = head.strip_prefix(&format!("<{adt_name}")) else {
            continue;
        };
        if rest.is_empty() {
            return true;
        }
        if !rest.starts_with('<') {
            continue;
        }
        let mut depth = 0usize;
        let mut closes_at_end = false;
        for (i, ch) in rest.char_indices() {
            match ch {
                '<' => depth += 1,
                '>' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        closes_at_end = i == rest.len() - 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        if closes_at_end {
            return true;
        }
    }
    false
}

/// The drop plan of a modelled owning iterator state
/// (`Enum { fields: [Tuple(elements), Uint(cursor)] }`, the shape
/// the `vec` / `array` / `option` `IntoIter` builtins build):
/// the elements tuple's glue, element by element. Slots the
/// iterator already yielded hold `Uninit`, which every plan
/// skips, so exactly the unyielded elements are dropped — the
/// real iterator's `Drop` impl semantics.
fn owned_iter_plan(elem: ir::DropPlan) -> ir::DropPlan {
    match elem {
        ir::DropPlan::Trivial => ir::DropPlan::Trivial,
        e => ir::DropPlan::At {
            path: vec![RefProj::Field(0)],
            plan: Box::new(ir::DropPlan::Each { elem: Box::new(e) }),
        },
    }
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
            mir::Ty::Array(elem, _) | mir::Ty::Slice(elem) => {
                let elem_plan = self.plan(elem)?;
                let elem_plan = self.elided_elem_plan(elem, elem_plan);
                match elem_plan {
                    P::Trivial => Some(P::Trivial),
                    elem_plan => Some(P::Each {
                        elem: Box::new(elem_plan),
                    }),
                }
            }
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

    /// The drop plan of one sequence element (of an array, a
    /// slice or a `Vec`), wrapped so the glue of an element whose
    /// store MIR elided still runs: when the element type has a
    /// single value, the plan becomes [`ir::DropPlan::Elided`]
    /// carrying that value. An element type with many values
    /// keeps its plain plan — an uninitialised slot of such a
    /// type in a live sequence is not a state this builder
    /// accounts for, and the read rules keep judging it.
    fn elided_elem_plan(&mut self, elem_ty: &mir::Ty, plan: ir::DropPlan) -> ir::DropPlan {
        use ir::DropPlan as P;
        match plan {
            P::Trivial => P::Trivial,
            plan => match self.conv.zst_value(elem_ty, 0) {
                Some(value) => P::Elided {
                    value,
                    plan: Box::new(plan),
                },
                None => plan,
            },
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
            // `MaybeUninit<T>` runs no glue either: its contents
            // are manually managed, so dropping the wrapper (e.g.
            // the `Box<MaybeUninit<[T; N]>>` a `vec!` construction
            // is assembled in before `assume_init`) drops nothing.
            "std::mem::MaybeUninit" | "core::mem::MaybeUninit" => return Some(P::Trivial),
            // The unit allocator marker: a zero-sized value with
            // no fields and no `Drop` impl.
            "std::alloc::Global" => return Some(P::Trivial),
            // The raw allocation buffer behind the modelled
            // containers (`Vec`, `String`, `VecDeque`): rustc's
            // generated drop glue for a container drops the
            // buffer field as a place of this type (and, inside
            // it, of `RawVecInner`). The buffer's real glue only
            // deallocates -- dropping the elements is the owner's
            // glue (the `Vec` plan, the slice glue), never the
            // buffer's -- and the value model does not represent
            // the allocation at all (a container value holds its
            // elements directly; capacity is not represented),
            // so there is no glue to run for the buffer field.
            "alloc::raw_vec::RawVec" | "alloc::raw_vec::RawVecInner" => {
                return Some(P::Trivial)
            }
            // A fieldless error struct (the payload of a failed
            // channel receive): nothing to drop.
            "std::sync::mpmc::RecvError" => return Some(P::Trivial),
            // A type identity: its payload is a hash value with
            // no glue of its own.
            "std::any::TypeId" => return Some(P::Trivial),
            "std::sync::mpmc::SendError" => {
                return Some(self.variant_fields_plan(def, args)?);
            }
            // `FlatMap` has no modelled iterator state in the
            // interpreter (its `next` is the converted body), so
            // its value keeps the real layout and the glue is the
            // fields': the source iterator, the closure and the
            // two in-flight inner iterators.
            "std::iter::FlatMap" | "std::iter::adapters::flatten::FlattenCompat"
            | "std::iter::Fuse" => {
                return Some(self.variant_fields_plan(def, args)?);
            }
            "std::panic::AssertUnwindSafe" | "core::panic::AssertUnwindSafe" => {
                // A one-field transparent wrapper with no `Drop`
                // impl of its own: dropping it drops the held
                // value. `catch_unwind` arguments are dropped on
                // every path through the panic machinery, so this
                // glue is on the coroutine panic path as well.
                let inner = self.plan(args.first()?)?;
                return Some(at_field0(inner));
            }
            "std::pin::Pin" | "core::pin::Pin" | "std::pin::PinMacroHelper" => {
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
                let elem_ty = args.first()?;
                let elem = self.plan(elem_ty)?;
                let elem = self.elided_elem_plan(elem_ty, elem);
                return Some(match elem {
                    P::Trivial => P::Trivial,
                    e => P::Each { elem: Box::new(e) },
                });
            }
            "std::boxed::Box" => {
                // Dropping a box drops its pointee. Which storage
                // the pointee lives in is a property of the box
                // value's own shape — the `Box::new` builtin's
                // one-field aggregate and the real `Unique` /
                // `NonNull` layout a converted constructor body
                // (`Box::from_raw`, `Box::new_in`, ...) builds
                // both occur — so the plan defers to the run-time
                // judgment in `DropPlan::BoxPointee`. A `dyn`
                // pointee is the exception: the pointer value *is*
                // the fat pointer then, with no field to project,
                // so the plan is the dyn plan itself. A pointee
                // that merely mentions a trait object (an ADT or
                // tuple whose unsized tail is a `dyn` field) may
                // likewise hold the trait-object form — such a box
                // is formed only by the unsizing coercion of a
                // concrete box — but the same type also spells
                // plain boxes (a `Vec<Box<dyn Trait>>` element type
                // inside the pointee, say), so the form is judged
                // on the value at run time.
                let pointee = args.first()?;
                let inner = self.plan(pointee)?;
                return Some(match inner {
                    P::Dyn => P::Dyn,
                    other => {
                        // A pointee with no glue leaves the box
                        // nothing to drop, exactly as the field-0
                        // addressing this replaces collapsed to
                        // `Trivial` (see `at_field0`).
                        let boxed = match other {
                            P::Trivial => P::Trivial,
                            o => P::BoxPointee {
                                pointee: Box::new(o),
                            },
                        };
                        if ty_mentions_dyn(pointee) {
                            P::DynOrPlain {
                                plain: Box::new(boxed),
                            }
                        } else {
                            boxed
                        }
                    }
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
                let pointee = args.first()?;
                let inner = self.plan(pointee)?;
                return Some(match inner {
                    P::Dyn => P::Dyn,
                    other if ty_mentions_dyn(pointee) => P::DynOrPlain {
                        plain: Box::new(P::RcHandleDrop {
                            value: Box::new(other),
                        }),
                    },
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
            "std::cell::Cell" | "core::cell::Cell" => {
                // A transparent wrapper with no `Drop` impl of
                // its own: the modelled value is the two-layer
                // aggregate `Cell { UnsafeCell { value } }` (see
                // `cell_value` in `builtins`), so the held value's
                // glue runs through both field-0 layers.
                let inner = self.plan(args.first()?)?;
                return Some(at_field0(at_field0(inner)));
            }
            "std::cell::UnsafeCell" | "core::cell::UnsafeCell" => {
                // The one-layer transparent wrapper the `Cell`
                // plan above peels: dropping it drops the held
                // value, as for `Cell` itself.
                let inner = self.plan(args.first()?)?;
                return Some(at_field0(inner));
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
            "std::collections::VecDeque" => {
                // A deque's glue is its elements' glue. The value
                // model has no deque shape of its own to run a
                // non-trivial element plan against, so only a
                // trivial element type is accounted for (the
                // `vec::IntoIter` rule).
                let elem = self.plan(args.first()?)?;
                return match elem {
                    P::Trivial => Some(P::Trivial),
                    _ => None,
                };
            }
            // Lock and atomic wrappers with no modelled lock
            // state: their glue is their fields' glue (the data
            // cell's, through the `UnsafeCell` plan), planned
            // from the same evidence the generic path uses for
            // the entry crate's types.
            "std::sync::Mutex" | "std::sync::RwLock" | "std::sync::atomic::Atomic" => {
                return Some(self.variant_fields_plan(def, args)?);
            }
            // The integer parse error types: a struct holding a
            // fieldless kind enum; neither has a `Drop` impl, so
            // the evidenced-fields plan is the whole glue.
            "std::num::ParseIntError" | "core::num::ParseIntError"
            | "std::num::IntErrorKind" | "core::num::IntErrorKind" => {
                return Some(self.variant_fields_plan(def, args)?);
            }
            "std::task::Waker" | "core::task::Waker" | "std::task::LocalWaker"
            | "core::task::LocalWaker" => {
                // Which release a waker's drop runs depends on
                // the shape of the stored value (the modelled
                // `From` conversion's handle aggregate, or a real
                // `RawWaker` from `Waker::from_raw` whose vtable
                // drop entry runs): the `WakerDrop` plan judges
                // the value at run time (see `ir::DropPlan`).
                return Some(P::WakerDrop);
            }
            // Marker types: no fields with glue, no `Drop` impl.
            "std::marker::PhantomPinned" | "std::marker::PhantomData" => {
                return Some(P::Trivial)
            }
            // The futex-backed sys synchronization primitives:
            // their state is atomic words only, with no glue.
            "std::sys::sync::mutex::futex::Mutex" | "std::sys::sync::rwlock::futex::RwLock" => {
                return Some(P::Trivial)
            }
            // The lock poisoning state: a flag wrapping an atomic
            // bool, and the guard token a lock guard holds. Neither
            // carries glue.
            "std::sync::poison::Flag" | "std::sync::poison::Guard" => {
                return Some(P::Trivial)
            }
            // The futex thread parker (atomic state words only)
            // and the atomic storage alignment wrappers (byte
            // blocks with no contents of their own): no glue.
            "std::sys::sync::thread_parking::futex::Parker"
            | "std::sync::atomic::private::Align1" | "std::sync::atomic::private::Align2"
            | "std::sync::atomic::private::Align4" | "std::sync::atomic::private::Align8"
            | "std::sync::atomic::private::Align16" | "std::sync::atomic::private::Align32"
            | "std::sync::atomic::private::Align64" | "std::sync::atomic::private::Align128" => {
                return Some(P::Trivial)
            }
            "std::async_iter::FromIter" => {
                // A one-field adapter holding the wrapped
                // iterator: dropping it drops the iterator.
                let inner = self.plan(args.first()?)?;
                return Some(at_field0(inner));
            }
            // Lock guards: the destructor releases the lock,
            // which the value model does not track as state, so
            // the glue is the fields' (a lock reference and a
            // poison flag - neither carries glue of its own).
            // The guarded value is borrowed, never dropped
            // through the guard. `Hook` is the panic hook holder:
            // its one field is the boxed hook, whose glue the
            // evidenced-fields plan reaches through the `Box`
            // plan. A thread handle's fields are its id and name.
            "std::sync::MutexGuard" | "std::sync::RwLockReadGuard"
            | "std::sync::RwLockWriteGuard" | "std::sync::nonpoison::MutexGuard"
            | "std::sync::nonpoison::RwLockReadGuard"
            | "std::sync::nonpoison::RwLockWriteGuard" | "std::panicking::Hook"
            | "std::thread::Thread" | "std::thread::thread::Inner" => {
                return Some(self.variant_fields_plan(def, args)?);
            }
            // `std::io::Error` (the payload of a `Builder::spawn`
            // result) has no `Drop` impl: its glue is its fields'
            // glue, planned from the same evidence the generic
            // path uses for the entry crate's types. An
            // unevidenced field stays unaccounted for and fails
            // the plan, as usual.
            // The bitpacked error representation word: for the
            // Os / Simple / SimpleMessage tags - the only
            // representations the runtime produces - the word
            // owns nothing (an error code, a kind discriminant,
            // or a `&'static` message). The Custom tag's heap box
            // is reachable only through `Error::new` construction,
            // which has no converted body in this model; if that
            // construction is ever modelled, this entry must be
            // revisited so the box's glue runs.
            "core::io::error::repr::Repr" => return Some(P::Trivial),
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
            // Process commands: the value is the process
            // model's handle (an id into the interpreter's
            // command table); the builder state it denotes
            // carries no guest-visible destruction, so there
            // is no glue to run. (A dropped `Child` likewise
            // has no glue in the model: no child is ever
            // started, so there is nothing to reap.)
            "std::process::Command"
            | "std::sys::process::unix::common::Command"
            | "std::process::Child" => {
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
            "std::array::IntoIter" | "core::array::IntoIter" => {
                // The modelled state holds the buffer's elements
                // in a tuple with yielded slots marked `Uninit`,
                // so the elements' glue is exact (see
                // `owned_iter_plan`).
                let elem = self.plan(args.first()?)?;
                return Some(owned_iter_plan(elem));
            }
            "std::vec::IntoIter" | "alloc::vec::IntoIter" => {
                let elem = self.plan(args.first()?)?;
                return Some(owned_iter_plan(elem));
            }
            "std::option::IntoIter" | "core::option::IntoIter" => {
                let elem = self.plan(args.first()?)?;
                return Some(owned_iter_plan(elem));
            }
            "std::collections::btree_set::IntoIter"
            | "alloc::collections::btree_set::IntoIter" => {
                // The modelled state is the element snapshot plus
                // the next index, like `vec::IntoIter` (see
                // `owned_iter_plan`).
                let elem = self.plan(args.first()?)?;
                return Some(owned_iter_plan(elem));
            }
            "std::collections::btree_map::IntoIter"
            | "alloc::collections::btree_map::IntoIter" => {
                // The snapshot's elements are `(key, value)`
                // tuples; their glue is the two components'.
                let key = self.plan(args.first()?)?;
                let value = self.plan(args.get(1)?)?;
                let elem = collapse_fields(vec![key, value], true);
                return Some(owned_iter_plan(elem));
            }
            // Borrowing iterators own nothing: the references (and,
            // for `IterMut`, the container snapshot) inside the
            // modelled state are not owned values, and the real
            // types have no glue of their own.
            "std::option::Iter" | "core::option::Iter"
            | "std::option::IterMut" | "core::option::IterMut"
            | "std::slice::Iter" | "core::slice::Iter"
            | "std::slice::IterMut" | "core::slice::IterMut"
            | "std::str::Chars" | "core::str::Chars"
            | "std::str::Bytes" | "core::str::Bytes"
            | "std::collections::btree_set::Iter"
            | "alloc::collections::btree_set::Iter"
            | "std::collections::btree_map::Iter"
            | "alloc::collections::btree_map::Iter"
            | "std::collections::btree_map::Keys"
            | "alloc::collections::btree_map::Keys"
            | "std::collections::btree_map::Values"
            | "alloc::collections::btree_map::Values"
            | "std::iter::Empty" | "core::iter::Empty" => return Some(P::Trivial),
            "std::path::Components" => {
                // The modelled state is the path's byte string
                // plus the two consumed counts (see
                // `Builtin::PathComponents` in `builtins`):
                // plain data with no glue of its own. The real
                // type owns nothing either (a borrowed path
                // slice plus parse state), so the plans agree.
                return Some(P::Trivial)
            }
            "std::path::Component" | "std::path::PrefixComponent"
            | "std::path::Prefix" => {
                // The component types hold only string slices
                // (`&OsStr` / `&str`) and plain discriminants,
                // and none of them has a `Drop` impl, so their
                // glue is empty. Naming them here matters
                // because library bodies are not drop-elaborated
                // the way entry-crate bodies are: a generic
                // library body (the closure inside
                // `Iterator::count`, say) still carries `Drop`
                // terminators for the items it consumes, and
                // the generic fallthrough below refuses a
                // non-local type it has no destructor record
                // for.
                return Some(P::Trivial)
            }
            "std::iter::Map" | "core::iter::Map" => {
                // State fields: the inner iterator's state and the
                // closure value; glue is the two fields' glue.
                let inner = self.plan(args.first()?)?;
                let func = self.plan(args.get(1)?)?;
                return Some(collapse_fields(vec![inner, func], true));
            }
            "std::iter::Enumerate" | "core::iter::Enumerate" => {
                let inner = self.plan(args.first()?)?;
                return Some(collapse_fields(vec![inner, P::Trivial], true));
            }
            "std::iter::Copied" | "core::iter::Copied" => {
                let inner = self.plan(args.first()?)?;
                return Some(at_field0(inner));
            }
            "std::iter::Take" | "core::iter::Take" => {
                let inner = self.plan(args.first()?)?;
                return Some(collapse_fields(vec![inner, P::Trivial], true));
            }
            "std::iter::Skip" | "core::iter::Skip" => {
                // Fields are the inner iterator and the
                // remaining skip count (a `usize`: no glue).
                let inner = self.plan(args.first()?)?;
                return Some(collapse_fields(vec![inner, P::Trivial], true));
            }
            "std::iter::Chain" | "core::iter::Chain" => {
                // Fields are the two iterators, each held
                // behind an `Option`: dropping the chain drops
                // whichever iterators are still present, each
                // through the `Option` plan for its type.
                let first = self.plan(args.first()?)?;
                let second = self.plan(args.get(1)?)?;
                let option_plan = |inner: P| match inner {
                    P::Trivial => P::Trivial,
                    e => P::Variants {
                        variants: vec![P::Trivial, at_field0(e)],
                    },
                };
                return Some(collapse_fields(
                    vec![option_plan(first), option_plan(second)],
                    true,
                ));
            }
            "std::iter::Once" | "core::iter::Once" => {
                // The state's only field is the wrapped
                // `option::IntoIter` state.
                let elem = self.plan(args.first()?)?;
                return Some(at_field0(owned_iter_plan(elem)));
            }
            "std::iter::SkipWhile" | "core::iter::SkipWhile" => {
                // Fields are the inner iterator, the `done` flag
                // (a bool: no glue) and the predicate.
                let inner = self.plan(args.first()?)?;
                let func = self.plan(args.get(1)?)?;
                return Some(collapse_fields(vec![inner, P::Trivial, func], true));
            }
            "std::iter::Repeat" | "core::iter::Repeat" => {
                // The state's only field is the element the
                // iterator clones from; dropping the iterator
                // drops it.
                let elem = self.plan(args.first()?)?;
                return Some(at_field0(elem));
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
            // The remaining field-only adapters: the iterator
            // adapters not listed above and the `str` escape
            // adapters. None of them has a `Drop` impl of its
            // own, so the glue is the fields' -- the inner
            // iterator, the closure / state / peeked slot the
            // adapter holds -- planned from the same evidence
            // the generic path uses for the entry crate's types.
            "std::iter::Rev" | "core::iter::Rev" | "std::iter::Cloned"
            | "core::iter::Cloned" | "std::iter::Cycle" | "core::iter::Cycle"
            | "std::iter::Zip" | "core::iter::Zip" | "std::iter::Filter"
            | "core::iter::Filter" | "std::iter::FilterMap"
            | "core::iter::FilterMap" | "std::iter::Peekable"
            | "core::iter::Peekable" | "std::iter::TakeWhile"
            | "core::iter::TakeWhile" | "std::iter::Scan" | "core::iter::Scan"
            | "std::iter::Inspect" | "core::iter::Inspect"
            | "std::iter::StepBy" | "core::iter::StepBy"
            | "std::str::EscapeDebug" | "core::str::EscapeDebug"
            | "std::str::EscapeDefault" | "core::str::EscapeDefault"
            | "std::str::EscapeUnicode" | "core::str::EscapeUnicode"
            | "std::char::EscapeDebug" | "core::str::CharEscapeDebug"
            | "std::char::EscapeDefault" | "core::str::CharEscapeDefault"
            | "std::char::EscapeUnicode" | "core::str::CharEscapeUnicode" => {
                return Some(self.variant_fields_plan(def, args)?);
            }
            "std::ffi::OsString" | "std::ffi::OsStr" => {
                // An `OsString` is the string value itself in this
                // model (see the `env::args_os` builtin): no owned
                // resource beyond it.
                return Some(P::Trivial);
            }
            "std::path::PathBuf" => {
                // A path buffer's only owned content is its
                // `OsString`, which is the string value itself in
                // this model (see the `OsString` entry above), so
                // the evidenced-fields plan is the whole glue.
                return Some(self.variant_fields_plan(def, args)?);
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
            "std::env::SplitPaths" => {
                // The modelled value is the `split_paths`
                // builtin's snapshot ([part snapshot, next
                // index]) whose elements are `PathBuf` values:
                // dropping the iterator drops the not-yet-yielded
                // parts, which is unobservable for path values
                // (the same reasoning as the `Args` entry above).
                // The type is opaque (`define_opaque`), so the
                // evidence-based planner cannot see its shape.
                return Some(P::Trivial);
            }
            "std::env::VarError" => {
                // A plain enum with no `Drop` impl. Its only
                // payload is the `NotUnicode` variant's
                // `OsString`, which is the string value itself in
                // this model (see the `OsString` entry above), so
                // the whole glue is trivial.
                return Some(P::Trivial);
            }
            "std::fs::Metadata" => {
                // A plain snapshot of the host `stat` data: no
                // `Drop` impl, and every field (the stat struct,
                // the file type, the times) is plain data, so the
                // whole glue is trivial.
                return Some(P::Trivial);
            }
            "std::fs::DirEntry" => {
                // No `Drop` impl; the modelled value is the
                // entry's path and name, both string data in this
                // model, so the whole glue is trivial.
                return Some(P::Trivial);
            }
            "std::fs::ReadDir" => {
                // The modelled value is the snapshot of the
                // directory's entries plus the read index:
                // dropping it drops the not-yet-yielded entries,
                // which are plain data in this model (the same
                // accounting as `std::env::Args` above).
                return Some(P::Trivial);
            }
            "build_helper::drop_bomb::DropBomb" => {
                // The bomb's own `Drop` impl panics when the
                // bomb is still armed. That check is a
                // recipe-authoring guard, not recipe semantics:
                // the harness's native oracle runs the recipe
                // first, so a recipe whose commands are dropped
                // unexecuted has already failed natively, and
                // the defuse itself is ordinary field data the
                // converted support code maintains. Its fields
                // (an `OsString`, a flag and a `Location`) are
                // all glue-free in this model, so the glue is
                // trivial.
                return Some(P::Trivial);
            }
            "run_make_support::command::Command"
            | "run_make_support::CompletedProcess"
            | "run_make_support::Rustc"
            | "run_make_support::Cargo"
            | "run_make_support::Rustdoc"
            | "run_make_support::Cc"
            | "run_make_support::LlvmAr"
            | "run_make_support::LlvmReadobj" => {
                // The run-make support library's own value
                // types: builders wrapping the modelled
                // `std::process::Command` (a handle whose glue
                // is trivial, above) plus option/string/path
                // data and the `DropBomb` accounted for above,
                // and the completed-process wrapper around the
                // modelled `Output`. None of them has a `Drop`
                // impl, and every field is glue-free in this
                // model, so the glue is trivial. (Their shapes
                // are not in the evidence tables -- the types
                // live in an auxiliary crate whose construction
                // sites are not converted -- so the generic
                // fields plan cannot be derived for them.)
                return Some(P::Trivial);
            }
            "run_make_support::similar::TextDiff"
            | "run_make_support::similar::udiff::UnifiedDiff" => {
                // Diff values are borrowed line data and owned
                // vectors of it: no `Drop` impl, and every
                // field is glue-free in this model.
                return Some(P::Trivial);
            }
            "std::process::Stdio" => {
                // Stdio configuration is not observed by the
                // process model (see the `ProcessCommandStdio`
                // builtin): a modelled `Stdio` value holds no
                // host resource, so its glue is trivial.
                return Some(P::Trivial);
            }
            "std::process::ChildStdin"
            | "std::process::ChildStdout"
            | "std::process::ChildStderr" => {
                // Child pipe handles exist only for a spawned
                // child, which the process model never produces
                // (spawn stays unsupported at its call site);
                // the values are unreachable in a running
                // program, and their glue is trivial.
                return Some(P::Trivial);
            }
            "std::string::FromUtf8Error" => {
                // A byte vector plus a plain validation error:
                // no `Drop` impl, nothing owned beyond the
                // buffer value itself.
                return Some(P::Trivial);
            }
            "std::process::Output" => {
                // No `Drop` impl; the fields are an exit status
                // (a plain wait-status integer) and two byte
                // vectors, whose buffers are values in this
                // model (as for `OsString` above), so the whole
                // glue is trivial.
                return Some(P::Trivial);
            }
            "std::process::ExitStatus" => {
                // A plain wait-status integer in a two-level
                // wrapper: no `Drop` impl, nothing owned.
                return Some(P::Trivial);
            }
            _ => {}
        }
        // Any other ADT: its glue is its `Drop` impl (called first,
        // as in rustc's drop glue) followed by its fields' glue.
        let mut steps = Vec::new();
        let ty_key = format!("{:?}", mir::Ty::Adt(def, args.to_vec()));
        let dtor = conv
            .dtor_bodies
            .get(&ty_key)
            .cloned()
            .or_else(|| conv.dtor_body(&name, args));
        if let Some(func) = dtor {
            steps.push(P::CallDtor { func });
        } else if conv.drop_trivial_adts.contains(&ty_key) {
            // The compiler certifies this instantiation's drop
            // glue empty: no destructor and no field glue exists
            // anywhere in its tree, so the trivial plan is exact
            // rather than an assumption about an unrecorded impl.
            return Some(P::Trivial);
        } else if conv.dtor_impl_known(&name) {
            // The impl exists but has no converted body (a generic
            // impl gets no monomorphic instance in the program):
            // its effects can neither be executed nor ruled out.
            return None;
        } else if !conv.is_local_type(&name) && !conv.drop_no_impl_adts.contains(&ty_key) {
            // Not the entry crate's type: an impl could exist that
            // the name table never recorded. (A type the driver
            // certifies has no `Drop` impl in its own crate — the
            // only crate the orphan rule allows one in — falls
            // through to its fields' glue below.)
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
            plans.push(self.one_variant_fields(def, args, v, false)?);
        }
        // When the driver recorded the instance's storage-slot
        // map, the conversion rewrote the body's state-field
        // projections to bare slot fields, so at run time the
        // state fields live in the value's flat field vector -
        // the storage a bare field step addresses - and not in
        // the per-state saved storage the glue's `Fields` arm
        // reads for a suspended coroutine. A slot-indexed
        // `Fields` plan would be judged against that saved
        // storage: its entries past the saved vector's length
        // read as never written and are silently skipped, and
        // the surviving indices name the wrong cells. Address
        // each state's field plans at their slots directly
        // instead: an `At` step carrying the bare slot field
        // resolves, through the ordinary projectors, to the
        // cell the body wrote. Variant 0 is the unresumed
        // state, whose fields are the upvars themselves; its
        // positional `Fields` plan already addresses the upvar
        // storage in exactly that way, so it is left alone.
        let key = crate::builtins::ty_tag(&mir::Ty::Coroutine(def, args.to_vec()), conv.names);
        if let Some(slots) = conv.coroutine_layouts.get(&key) {
            for (v, plan) in plans.iter_mut().enumerate().skip(1) {
                let Some(variant_slots) = slots.variants.get(v) else {
                    continue;
                };
                if let P::Fields { fields, .. } = plan {
                    let mut steps: Vec<P> = Vec::new();
                    for (f, field_plan) in fields.drain(..).enumerate() {
                        if field_plan == P::Trivial {
                            continue;
                        }
                        if let Some(slot) = variant_slots.get(f) {
                            steps.push(P::At {
                                path: vec![RefProj::Field(*slot as usize)],
                                plan: Box::new(field_plan),
                            });
                        }
                    }
                    *plan = P::Seq(steps);
                }
            }
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
            plans.push(self.one_variant_fields(def, args, v, true)?);
        }
        if plans.len() == 1 {
            Some(plans.pop().expect("one variant"))
        } else {
            Some(P::Variants { variants: plans })
        }
    }

    /// The field glue of one variant of one ADT. With
    /// `shape_fallback`, the driver's shape table — which declares
    /// every field of the variant with its type — fills in what
    /// the program's evidence misses. Evidence alone misses a
    /// variant that is only ever built as a constant: a
    /// zero-sized value's construction is a constant in MIR, so
    /// no aggregate statement records its field count and no
    /// projection records its fields' types, and the variant's
    /// fields' glue would be planned as nothing at all. Coroutine
    /// states pass `false`: their fields are evidence-based by
    /// design, since a saved field can be moved out of before the
    /// state is dropped.
    fn one_variant_fields(
        &mut self,
        def: mir::DefId,
        args: &[mir::Ty],
        variant: u32,
        shape_fallback: bool,
    ) -> Option<ir::DropPlan> {
        let conv = self.conv;
        let count = conv.adt_aggregates.get(def, args, variant);
        let evidence_count = match count {
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
        let shape_fields = if shape_fallback {
            conv.instantiated_field_tys(&def, args, variant)
        } else {
            None
        };
        let field_count =
            evidence_count.max(shape_fields.as_ref().map_or(0, |tys| tys.len()));
        let mut fields = Vec::with_capacity(field_count);
        for f in 0..field_count as u32 {
            match conv.adt_fields.get(def, args, variant, f) {
                Some(ty) => {
                    let ty = ty.clone();
                    fields.push(self.plan(&ty)?);
                }
                None => match shape_fields
                    .as_ref()
                    .and_then(|tys| tys.get(f as usize))
                {
                    Some(ty) => {
                        let ty = ty.clone();
                        // A field only the shape accounts for:
                        // plan its glue from the declared type. A
                        // part the builder cannot account for keeps
                        // the runtime judgment an unrecorded field
                        // gets, rather than failing the whole glue.
                        fields.push(
                            self.plan(&ty).unwrap_or(ir::DropPlan::Untyped),
                        );
                    }
                    None => {
                        // No projection anywhere records this
                        // field's type (it is only ever written,
                        // e.g. by an aggregate with a literal). Its
                        // glue is judged by the value's own shape
                        // at run time.
                        fields.push(ir::DropPlan::Untyped);
                    }
                },
            }
        }
        Some(collapse_fields(
            fields,
            count.is_some() || shape_fields.is_some(),
        ))
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

/// One wrapper layer on the way from a cast's target type to
/// the `dyn` type inside it: a reference layer, a raw-pointer
/// layer, or the generic-argument index of an ADT layer.
#[derive(Clone, Copy)]
enum DynStep {
    Ref,
    RawPtr,
    Arg(usize),
}

/// The wrapper path from a type to the `dyn` type inside it (the
/// traversal of `dyn_traits_in`, recording each layer).
fn dyn_path_in(ty: &mir::Ty) -> Option<Vec<DynStep>> {
    match ty {
        mir::Ty::Dynamic(..) => Some(Vec::new()),
        mir::Ty::Ref(_, inner, _) => {
            let mut path = dyn_path_in(inner)?;
            path.insert(0, DynStep::Ref);
            Some(path)
        }
        mir::Ty::RawPtr(inner, _) => {
            let mut path = dyn_path_in(inner)?;
            path.insert(0, DynStep::RawPtr);
            Some(path)
        }
        mir::Ty::Adt(_, args) => {
            for (i, arg) in args.iter().enumerate() {
                if let Some(mut path) = dyn_path_in(arg) {
                    path.insert(0, DynStep::Arg(i));
                    return Some(path);
                }
            }
            None
        }
        _ => None,
    }
}

/// The source type a `dyn` path names (see `dyn_path_in`): apply
/// each wrapper step of the target to the source. `None` when
/// the shapes diverge (the caller falls back to the heuristic
/// walk in `unsize_concrete_pointee`).
fn pointee_at_path<'t>(mut ty: &'t mir::Ty, path: &[DynStep]) -> Option<&'t mir::Ty> {
    for step in path {
        ty = match (step, ty) {
            (DynStep::Ref, mir::Ty::Ref(_, inner, _)) => inner,
            (DynStep::RawPtr, mir::Ty::RawPtr(inner, _)) => inner,
            (DynStep::Arg(i), mir::Ty::Adt(_, args)) => args.get(*i)?,
            _ => return None,
        };
    }
    Some(ty)
}

/// The concrete type an unsizing cast's *value* answers for: the
/// trait object the cast forms is a `dyn` of the type the
/// target's own wrapper path names in the source — the layers
/// above the `dyn` are the pointer the value travels through,
/// and everything at the path's end is the trait's `Self`,
/// wrapper layers included: `Box<Box<T>>` cast to `Box<dyn
/// Trait>` carries the concrete tag `Box<T>` (the trait is
/// implemented for `Box<T>`), while `Pin<Box<T>>` cast to
/// `Pin<Box<dyn Trait>>` names `T`. The tag must name that type
/// (see `Value::DynRef`): it is the vtable a dispatch looks up.
/// When the target's path cannot be followed in the source, the
/// heuristic walk below applies: the first step is the plain
/// pointee step (any ADT gives up its first argument) and
/// further steps only pass through the standard pointer wrappers
/// (`Pin`, `Box`, `Rc`, `Arc`, the `Weak`s) and references, so a
/// user type in the chain stops the walk.
fn unsize_concrete_pointee<'t>(
    conv: &Converter<'_>,
    src: &'t mir::Ty,
    target: &mir::Ty,
) -> &'t mir::Ty {
    if let Some(path) = dyn_path_in(target) {
        if let Some(ty) = pointee_at_path(src, &path) {
            return ty;
        }
    }
    let mut ty = unsize_pointee(src);
    loop {
        match ty {
            mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => ty = inner,
            mir::Ty::Adt(def, args) => {
                let name = conv
                    .names
                    .get(def)
                    .or_else(|| conv.def_names.get(def))
                    .map(|n| n.as_str());
                match name {
                    Some(
                        "std::pin::Pin"
                        | "core::pin::Pin"
                        | "std::boxed::Box"
                        | "alloc::boxed::Box"
                        | "std::rc::Rc"
                        | "alloc::rc::Rc"
                        | "std::sync::Arc"
                        | "alloc::sync::Arc"
                        | "std::rc::Weak"
                        | "alloc::rc::Weak"
                        | "std::sync::Weak"
                        | "alloc::sync::Weak",
                    ) => match args.first() {
                        Some(inner) => ty = inner,
                        None => return ty,
                    },
                    _ => return ty,
                }
            }
            other => return other,
        }
    }
}
