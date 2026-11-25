//! Panic-payload classification for Stable MIR queries, the captured-panic slot, and environment-shape helpers.

use super::*;

/// The cardinality of a type's value set, for the zero-sized
/// constant synthesis in `Converter::unique_const_value`.
pub(crate) enum UniqueConst {
    /// Exactly one value; here it is.
    One(sm::ConstValue),
    /// No values (an uninhabited type).
    Uninhabited,
    /// More than one value, or not provable here.
    Many,
}

/// The gap behind every coroutine-closure (async closure)
/// accounting in this crate: rustc_public 1.100.0's
/// internal-to-stable type conversion is unimplemented for
/// `ty::CoroutineClosure` ("FIXME(async_closures): Lower these
/// to SMIR"), and Stable MIR types are opaque, so the panic
/// fires lazily, the first time a conversion asks for the kind
/// of such a type — inside `convert_body`, or when the type of
/// an async closure item is asked for directly. Conversions
/// guard the question (see `guarded_ty_kind`): a body that
/// merely mentions the type still converts, with the type
/// recorded as a placeholder, and an async closure item — whose
/// type cannot be asked at all — is deferred and accounted
/// from converted bodies' demands. This reason names the gap:
/// it classifies the panic where one escapes a guard, and it
/// is the recorded reason of an async closure item that no
/// converted body demanded.
pub const COROUTINE_CLOSURE_TYPE_REASON: &str = "body mentions a coroutine-closure \
    (async closure) type, which Stable MIR 1.100.0 cannot convert \
    (rustc_public leaves ty::CoroutineClosure unimplemented)";

/// Whether a caught panic payload is rustc_public 1.100.0's
/// unimplemented coroutine-closure type conversion.
pub fn is_coroutine_closure_panic(payload: &(dyn std::any::Any + Send)) -> bool {
    let message = if let Some(s) = payload.downcast_ref::<&str>() {
        Some((*s).to_string())
    } else if let Some(s) = payload.downcast_ref::<String>() {
        Some(s.clone())
    } else {
        None
    };
    let r = message
        .as_deref()
        .is_some_and(|m| m.contains("FIXME(async_closures)"));
    r
}

/// The first panic's hook record (its message and location)
/// while the driver's panic-capture hook is installed; the hook
/// itself lives in the driver binary (its `begin_panic_capture`)
/// and records through [`record_captured_panic`]. The record is
/// the reliable witness of a panic whose payload the bridge made
/// opaque: a panic raised inside a Stable MIR query crosses the
/// bridge, which replaces the payload, so a `catch_unwind`
/// around the query cannot read the message from the payload —
/// but the hook saw the panic at its origin, location included.
/// Consumers clear the slot before the operation they guard and
/// take the record after it, so the record they read is that
/// operation's own panic (the discipline the driver's
/// location-based classification relies on as well).
static CAPTURED_PANIC: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Record a panic's hook text; only the first record stands
/// until it is taken or cleared.
pub fn record_captured_panic(text: String) {
    if let Ok(mut slot) = CAPTURED_PANIC.lock() {
        if slot.is_none() {
            *slot = Some(text);
        }
    }
}

/// Peek at the panic-capture slot without clearing it (see
/// [`take_captured_panic`]).
pub fn peek_captured_panic() -> Option<String> {
    CAPTURED_PANIC.lock().ok().and_then(|slot| slot.clone())
}

/// Take the recorded panic text, clearing the slot.
pub fn take_captured_panic() -> Option<String> {
    CAPTURED_PANIC.lock().ok().and_then(|mut slot| slot.take())
}

/// Clear the panic-capture slot without reading it.
pub fn clear_captured_panic() {
    if let Ok(mut slot) = CAPTURED_PANIC.lock() {
        *slot = None;
    }
}

/// The kind of a Stable MIR type, or `None` when the type is a
/// coroutine-closure type: asking for its kind materializes the
/// bridge's unimplemented coroutine-closure type conversion
/// (`FIXME(async_closures)`), which panics instead of answering.
/// The type's definition and arguments are not recoverable from
/// the opaque handle in that case. Any other panic is resumed —
/// only this one known gap is tolerated.
/// The definition and upvars-tuple type of a coroutine-closure
/// (async closure) type, read through the compiler context
/// installed for the conversion. The type's kind cannot be
/// asked through Stable MIR (the bridge's conversion of the
/// kind panics; see `guarded_ty_kind`), but a zero-sized
/// constant of the type is a closure value, and naming it
/// needs exactly these two facts. `None` when the type is not
/// a coroutine-closure type, when no compiler context is
/// installed, or when the trailing type arguments do not have
/// the expected shape: the closure signature (a function
/// pointer type), the upvars tuple, and the by-reference
/// captures signature (also a function pointer type).
pub(crate) fn coroutine_closure_const_parts(
    ty: &rustc_public::ty::Ty,
) -> Option<(
    rustc_public::DefId,
    rustc_public::ty::GenericArgs,
    rustc_public::ty::Ty,
)> {
    screen_question(|| {
        let tcx = compiler_tcx()?;
        let internal = rustc_public::rustc_internal::internal(tcx, ty.clone());
        let rustc_middle::ty::TyKind::CoroutineClosure(def, args) =
            internal.kind()
        else {
            return None;
        };
        let tys: Vec<rustc_middle::ty::Ty<'_>> = args
            .iter()
            .filter_map(|arg| match arg.kind() {
                rustc_middle::ty::GenericArgKind::Type(t) => Some(t),
                _ => None,
            })
            .collect();
        if tys.len() < 3 {
            return None;
        }
        let upvars = tys[tys.len() - 2];
        if !matches!(upvars.kind(), rustc_middle::ty::TyKind::Tuple(_)) {
            return None;
        }
        let def = rustc_public::rustc_internal::stable(def);
        let args = rustc_public::ty::GenericArgs(
            args.iter()
                .map(|arg| match arg.kind() {
                    rustc_middle::ty::GenericArgKind::Lifetime(l) => {
                        rustc_public::ty::GenericArgKind::Lifetime(
                            rustc_public::rustc_internal::stable(l),
                        )
                    }
                    rustc_middle::ty::GenericArgKind::Type(t) => {
                        rustc_public::ty::GenericArgKind::Type(
                            rustc_public::rustc_internal::stable(t),
                        )
                    }
                    rustc_middle::ty::GenericArgKind::Const(c) => {
                        rustc_public::ty::GenericArgKind::Const(
                            rustc_public::rustc_internal::stable(c),
                        )
                    }
                })
                .collect(),
        );
        let upvars = rustc_public::rustc_internal::stable(upvars);
        Some((def, args, upvars))
    })
}

pub fn guarded_ty_kind(ty: &rustc_public::ty::Ty) -> Option<rustc_public::ty::TyKind> {
    clear_captured_panic();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.kind())) {
        Ok(kind) => Some(kind),
        Err(payload) => {
            if is_coroutine_closure_panic(&payload) {
                // Consume this panic's hook record, so no outer
                // containment mistakes it for its own.
                let _ = take_captured_panic();
                return None;
            }
            // The payload of a panic that crossed the bridge is
            // opaque (neither `&str` nor `String`), so the hook's
            // record of the same panic — taken after the slot was
            // cleared above, hence this very panic's — is the
            // witness that identifies it.
            match take_captured_panic() {
                Some(record) if record.contains("FIXME(async_closures)") => None,
                other => {
                    if let Some(record) = other {
                        record_captured_panic(record);
                    }
                    std::panic::resume_unwind(payload)
                }
            }
        }
    }
}

/// Why a body containing a tail call cannot be converted:
/// rustc_public 1.100.0's internal-to-stable terminator
/// conversion is `unimplemented!()` for
/// `mir::TerminatorKind::TailCall` (convert/stable/mir.rs:775),
/// so a body using `become` cannot even be materialized as a
/// Stable MIR body. The affected body is skipped with this
/// reason instead of the panic aborting the whole crate's
/// conversion. (The panic's payload is the bare text
/// "not implemented"; the site is identified by its recorded
/// location — see `unimplemented_site_reason` in the driver.)
pub const TAIL_CALL_REASON: &str = "body contains a tail call \
    (mir::TerminatorKind::TailCall), which Stable MIR 1.100.0 does not \
    expose (rustc_public leaves TailCall conversion unimplemented)";

/// Why a body mentioning an unsafe binder cannot be converted:
/// rustc_public 1.100.0's type conversion is `unimplemented!()`
/// for `ty::UnsafeBinder` (convert/stable/ty.rs:471), and its
/// projection conversion for the wrap/unwrap projections is
/// `unimplemented!("FIXME(unsafe_binders):")`
/// (convert/stable/mir.rs). The affected body is skipped with
/// this reason instead of the panic aborting the conversion.
pub const UNSAFE_BINDER_REASON: &str = "body mentions an unsafe binder \
    (ty::UnsafeBinder / WrapUnsafeBinder), which Stable MIR 1.100.0 \
    cannot convert (rustc_public leaves unsafe binders unimplemented)";

/// Why a body whose instance constants cannot be evaluated
/// cannot be converted: while materializing an instance body,
/// rustc_public_bridge evaluates the body's constants; when a
/// constant's evaluation is too generic
/// (`ErrorHandled::TooGeneric` — an unevaluated constant whose
/// arguments still mention parameters, e.g. `type_name::<T>`
/// inside a body the demand collection materialized), the
/// bridge hits `unreachable!("Failed to evaluate instance
/// constant")`. The affected body is skipped with this reason
/// instead of the panic aborting the conversion.
pub const INSTANCE_CONST_REASON: &str = "body mentions an instance constant \
    that Stable MIR 1.100.0 cannot evaluate (rustc_public_bridge \
    evaluates an instance's constants while materializing its body \
    and hits unreachable! when the evaluation is too generic)";

/// Why a body mentioning a generic const expression in a type
/// cannot be converted: rustc_public 1.100.0's constant
/// conversion is `unimplemented!()` for `ty::ConstKind::Expr`
/// (convert/stable/ty.rs:577) — the const-tree expressions of
/// `generic_const_exprs` in type position. The affected body
/// is skipped with this reason instead of the panic aborting
/// the conversion. (Identified by panic location, like
/// `TAIL_CALL_REASON`.)
pub const CONST_EXPR_REASON: &str = "body mentions a generic const \
    expression in a type (ty::ConstKind::Expr), which Stable MIR \
    1.100.0 cannot convert (rustc_public leaves it unimplemented)";

/// The string text of a caught panic payload, when it is one.
fn payload_text(payload: &(dyn std::any::Any + Send)) -> Option<String> {
    if let Some(s) = payload.downcast_ref::<&str>() {
        Some((*s).to_string())
    } else if let Some(s) = payload.downcast_ref::<String>() {
        Some(s.clone())
    } else {
        None
    }
}

/// Whether a caught panic payload is rustc_public 1.100.0's
/// unimplemented unsafe-binder projection conversion.
pub fn is_unsafe_binder_panic(payload: &(dyn std::any::Any + Send)) -> bool {
    payload_text(payload).is_some_and(|m| m.contains("FIXME(unsafe_binders)"))
}

/// Whether a caught panic payload is rustc_public_bridge's
/// unreachable instance-constant evaluation.
pub fn is_instance_const_panic(payload: &(dyn std::any::Any + Send)) -> bool {
    payload_text(payload)
        .is_some_and(|m| m.contains("Failed to evaluate instance constant"))
}

/// The skip reason for a caught panic whose payload alone
/// identifies it as one of rustc_public 1.100.0's known
/// unimplemented conversions (see the `*_REASON` constants).
/// Panics whose payload is the bare text "not implemented"
/// carry no identifying text; those are identified by their
/// recorded location instead (`unimplemented_site_reason` in
/// the driver) and are resumed here.
pub fn stable_mir_panic_reason(
    payload: &(dyn std::any::Any + Send),
) -> Option<&'static str> {
    if is_coroutine_closure_panic(payload) {
        Some(COROUTINE_CLOSURE_TYPE_REASON)
    } else if is_unsafe_binder_panic(payload) {
        Some(UNSAFE_BINDER_REASON)
    } else if is_instance_const_panic(payload) {
        Some(INSTANCE_CONST_REASON)
    } else {
        None
    }
}

thread_local! {
    /// Memoized results of [`ty_mentions_param`], keyed by the
    /// compiler's type handle. The walk is a pure function of
    /// the interned type, and the demand passes re-walk the
    /// same shared subtrees for every candidate instance:
    /// without the memo, a deeply nested type is re-expanded
    /// once per occurrence (and every node pays the guarded
    /// kind query). Cleared whenever a fresh [`Converter`]
    /// starts a compilation, so handles from an earlier
    /// compilation in the same process are never read back.
    pub(crate) static TY_MENTIONS_PARAM_CACHE: std::cell::RefCell<
        std::collections::HashMap<rustc_public::ty::Ty, bool>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// The DWARF name of a closure/coroutine environment type: the
/// definition path with its `{closure#N}` segment in the
/// `{<label>_env#N}` form the compiler's debuginfo uses (see
/// `push_closure_or_coroutine_name` in the compiler's
/// `debuginfo/type_names.rs`). A definition with no closure
/// segment (an `async fn` item) takes the environment segment as
/// a new final segment, with disambiguator 0.
pub(crate) fn env_dwarf_name(def_name: &str, label: &str) -> String {
    if let Some(pos) = def_name.find("::{closure#") {
        let mut out = String::with_capacity(def_name.len() + 8);
        out.push_str(&def_name[..pos]);
        out.push_str("::{");
        out.push_str(label);
        out.push_str("_env#");
        out.push_str(&def_name[pos + "::{closure#".len()..]);
        out
    } else {
        format!("{def_name}::{{{label}_env#0}}")
    }
}

/// The definition name a local's declared type denotes when it
/// is (a reference to, or a `Pin` of a reference to) a closure or
/// coroutine environment: the environment argument of a closure
/// body and the state local of a coroutine body are declared
/// that way. Anything else is not an environment base.
pub(crate) fn env_def_name(ty: &rustc_public::ty::Ty) -> Option<String> {
    let mut cur = *ty;
    loop {
        match guarded_ty_kind(&cur)? {
            TyKind::RigidTy(RigidTy::Ref(_, inner, _)) => cur = inner,
            TyKind::RigidTy(RigidTy::Adt(def, args))
                if def.name().ends_with("pin::Pin") =>
            {
                match args.0.first() {
                    Some(GenericArgKind::Type(t)) => cur = *t,
                    _ => return None,
                }
            }
            TyKind::RigidTy(RigidTy::Closure(def, _)) => {
                return Some(def.name());
            }
            TyKind::RigidTy(RigidTy::Coroutine(def, _)) => {
                return Some(def.name());
            }
            _ => return None,
        }
    }
}

/// A coroutine state's DWARF variant name, by discriminant
/// index (the compiler's `CoroutineArgs::variant_name` rule).
pub(crate) fn coroutine_variant_name(variant: usize) -> String {
    match variant {
        0 => "Unresumed".to_string(),
        1 => "Returned".to_string(),
        2 => "Panicked".to_string(),
        n => format!("Suspend{}", n - 3),
    }
}

/// The upvar types of a closure/coroutine instance, in
/// environment field order: the elements of the upvars tuple,
/// the last type argument of the instance.
pub(crate) fn env_upvar_tys(args: &GenericArgs) -> Option<Vec<rustc_public::ty::Ty>> {
    args.0.iter().rev().find_map(|a| match a {
        GenericArgKind::Type(t) => match guarded_ty_kind(t) {
            Some(TyKind::RigidTy(RigidTy::Tuple(ts))) => Some(ts),
            _ => None,
        },
        _ => None,
    })
}

/// The body an environment shape is read from, for one resolved
/// closure/coroutine instance: the instance's own body when the
/// instance is a plain item, else the definition's own body.
///
/// A resolution that lands on a compiler shim (the async
/// destructor glue of `core::future::async_drop_in_place` is the
/// known case: resolving its coroutine yields a shim instance)
/// has no body of its own to read debug info from — the shim's
/// body is synthesized glue, not the environment the definition
/// declares — and forcing the shim builder out of band can
/// panic inside the compiler (its preconditions are the mono
/// collector's, not a bare query's). Such an instance gets no
/// shape: `None` here skips the recording instead of falling
/// back to the definition body, whose generic debug info does
/// not describe the monomorphic environment either.
pub(crate) fn instance_item_body(
    key: &str,
    inst: Option<Instance>,
    def_body: impl FnOnce() -> Option<mir::Body>,
) -> Option<mir::Body> {
    with_body_materialization(key, || match inst {
        Some(inst) if matches!(inst.kind, InstanceKind::Item) => {
            // An instance whose types mention a projection
            // that does not normalize: fetching its body
            // would pose that normalization and poison the
            // compilation (see
            // `unnormalizable_projection_in_instance`). Fall
            // back to the definition body, the way a missing
            // instance body does.
            if unnormalizable_projection_in_instance(&inst).is_some() {
                return def_body();
            }
            inst.body().or_else(def_body)
        }
        Some(_) => None,
        None => def_body(),
    })
}

/// The outcome of probing a reference constant's allocation for
/// the fat-pointer shape (see `Converter::read_fat_pointer`).
/// A resolved provenance target (see `Converter::resolve_alloc`):
/// a memory allocation, shared through the converter's cache,
/// or any other global allocation in its bare form.
pub(crate) enum ResolvedAlloc {
    Memory(std::rc::Rc<Allocation>),
    Other(mir::alloc::GlobalAlloc),
    /// The target is a memory allocation whose materialisation
    /// the work budget refused (see `MAX_CONST_WORK_BYTES`):
    /// the decode that needed it fails, so the constant stays
    /// undecoded rather than taking any fallback value.
    Refused,
}

pub(crate) enum FatPointerRead {
    /// Not a fat pointer: no provenance, a non-memory target, or
    /// no readable in-bounds length word.
    NotFat,
    /// A fat pointer whose payload the remaining expansion
    /// budget cannot pay for. The length metadata is known, so a
    /// fallback that rereads the pointee without it would
    /// silently change the constant's value.
    OverBudget,
    /// The pointee bytes (exactly the metadata's byte count)
    /// and the metadata itself.
    Data(Vec<u8>, u64),
}
