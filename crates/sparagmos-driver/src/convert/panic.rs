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

/// Whether a panic record's message part is a non-string
/// payload: either the plain form, or one of the named forms
/// the panic hook gives a known site
/// (`main::panic_guard::non_string_payload_for` renders a
/// named non-string payload as
/// `"<name> (panic with a non-string payload)"`). The
/// classifiers below key on the payload kind through this
/// predicate, never on one exact rendering, so naming a new
/// site in the hook cannot silently unclassify its panics.
fn message_is_non_string_payload(message: &str) -> bool {
    message == "panic with a non-string payload"
        || message.ends_with("(panic with a non-string payload)")
}

/// Whether a panic hook record (the text
/// [`record_captured_panic`] stores: the payload text plus the
/// panic location) is the compiler's own bug machinery
/// firing: a `bug!` in a `compiler/rustc_*` crate. Such a
/// panic is preceded by the compiler emitting its
/// internal-compiler-error diagnostic (`span_bug`), and its
/// payload is the non-string `ExplicitBug` marker, so the
/// record's message part is the non-string form. Ordinary
/// compiler panics with string payloads are not this class:
/// the `std::fmt::Write` dummy-binder panic
/// (`rustc_middle/src/ty/predicate.rs`) is the standing
/// witness that they occur on speculative probes the
/// reference compilation never runs, and the symbol
/// mangler's `unimplemented!()` for an unsafe-binder type
/// (`rustc_symbol_mangling/src/v0.rs`) shows a compiler-side
/// site is not by itself a bug either. The bridge's own
/// unimplemented conversions live under `rustc_public/` and
/// are conversion gaps with their own classification (see
/// `unimplemented_site_reason` in the driver), and the
/// compiler classes that already carry named verdicts — the
/// whole-body normalization overflow, the MIR validator
/// failure, the delayed-bug flush — keep those verdicts and
/// are excluded here.
pub fn captured_panic_is_compiler_bug(text: &str) -> bool {
    let Some((message, location)) = text.split_once(" (panicked at ") else {
        return false;
    };
    if !message_is_non_string_payload(message) {
        return false;
    }
    let Some(compiler) = location.split("/compiler/").nth(1) else {
        return false;
    };
    if !compiler.starts_with("rustc_") || compiler.starts_with("rustc_public/") {
        return false;
    }
    !captured_panic_is_normalize_overflow(text)
        && !captured_panic_is_mir_validator_fail(text)
        && !text.contains("rustc_errors/src/lib.rs:1499:")
}

/// Whether a panic hook record is the compiler's
/// unimplemented normalization of a `ty::ConstKind::Expr`
/// (`rustc_middle/src/mir/consts.rs:338`, a `bug!` with the
/// non-string `ExplicitBug` payload): `Const::eval` has no
/// evaluation for a generic const expression. The site is a
/// compiler bug when the compiler's own const-eval forces it
/// (the collector replay in `mono_check` keeps that crash
/// verdict through the bug-witness list), but Stable MIR's
/// body builder also calls `Const::eval` eagerly on every
/// constant of a body it materializes -- including const
/// expressions that appear only in types, which the native
/// pipeline never evaluates as values -- so inside a body
/// materialization the same panic is an artifact of the
/// fetch, not of the program (Issue #510). The two readings
/// are told apart by scope, not by site: this predicate only
/// names the site, and `with_body_materialization` decides
/// what a materialization-scope firing means.
pub fn captured_panic_is_const_expr_normalize(text: &str) -> bool {
    let Some((message, location)) = text.split_once(" (panicked at ") else {
        return false;
    };
    message_is_non_string_payload(message)
        && location.contains("rustc_middle/src/mir/consts.rs:338:")
}

/// Whether a panic hook record (the text
/// [`record_captured_panic`] stores) is rustc_public's
/// unimplemented unsafe-binder type conversion: the bare
/// "not implemented" payload of the `unimplemented!()` at
/// convert/stable/ty.rs:471, identified by its location (the
/// identification `unimplemented_site_reason` makes in the
/// driver for the same site). A type whose kind question hits
/// this site is an unsafe binder type (`ty::UnsafeBinder`):
/// the binder is transparent at the value level, so the kind
/// question can be answered by erasure instead (see
/// [`unsafe_binder_inner_ty`]).
pub fn captured_panic_is_unsafe_binder_ty(text: &str) -> bool {
    let Some((message, location)) = text.split_once(" (panicked at ") else {
        return false;
    };
    message == "not implemented"
        && location.contains("rustc_public/src/unstable/convert/stable/ty.rs:471:")
}

/// Every const-expr-normalize panic record of the run (see
/// [`captured_panic_is_const_expr_normalize`]), noted by the
/// panic hook alongside the bug-witness list and read only
/// through scoped diffs of the same shape
/// ([`const_expr_normalize_witness_since`]).
static CONST_EXPR_WITNESSES: std::sync::Mutex<Vec<String>> =
    std::sync::Mutex::new(Vec::new());

/// The number of const-expr-normalize witnesses recorded so
/// far: a scope's snapshot to diff with
/// [`const_expr_normalize_witness_since`].
pub fn const_expr_normalize_witness_len() -> usize {
    CONST_EXPR_WITNESSES
        .lock()
        .map(|list| list.len())
        .unwrap_or(0)
}

/// The first const-expr-normalize witness recorded at or
/// after the snapshot `len`, if any.
pub fn const_expr_normalize_witness_since(len: usize) -> Option<String> {
    CONST_EXPR_WITNESSES
        .lock()
        .ok()
        .and_then(|list| list.get(len).cloned())
}

/// Every compiler-bug panic record of the run, in the order
/// the panics fired (see [`captured_panic_is_compiler_bug`]).
/// The panic hook notes each panic here through
/// [`note_panic_observation`], independently of the first-wins
/// capture slot above: a bug panic must stay visible even
/// when an earlier, unrelated probe panic already occupies
/// that slot (121575's run contains a contained
/// `std::fmt::Write` probe panic before its pinned ICE).
/// The list is read only through scoped diffs — a verdict
/// authority snapshots the length before the work it decides
/// and asks whether a bug panic fired inside it
/// ([`compiler_bug_witness_since`]) — so a bug panic on a
/// speculative path no authority scopes is recorded and
/// never read.
static BUG_WITNESSES: std::sync::Mutex<Vec<String>> =
    std::sync::Mutex::new(Vec::new());

/// Note one panic's hook record in the bug-witness list when
/// it is a compiler bug (see
/// [`captured_panic_is_compiler_bug`]); called by the
/// driver's panic hook for every panic of the run, alongside
/// [`record_captured_panic`].
pub fn note_panic_observation(record: &str) {
    if captured_panic_is_const_expr_normalize(record) {
        if let Ok(mut list) = CONST_EXPR_WITNESSES.lock() {
            list.push(record.to_string());
        }
    }
    if !captured_panic_is_compiler_bug(record) {
        return;
    }
    if let Ok(mut list) = BUG_WITNESSES.lock() {
        list.push(record.to_string());
    }
}

/// The number of compiler-bug witnesses recorded so far: a
/// scope's snapshot to diff with
/// [`compiler_bug_witness_since`].
pub fn compiler_bug_witness_len() -> usize {
    BUG_WITNESSES.lock().map(|list| list.len()).unwrap_or(0)
}

/// The first compiler-bug witness recorded at or after the
/// snapshot `len` (see [`compiler_bug_witness_len`]), if any.
pub fn compiler_bug_witness_since(len: usize) -> Option<String> {
    BUG_WITNESSES
        .lock()
        .ok()
        .and_then(|list| list.get(len).cloned())
}

/// The run's compiler-crash verdict, once a verdict
/// authority has established one (see
/// [`note_compiler_crash`]): the record of the compiler-bug
/// panic that decides it. First-wins: the earliest authority
/// to establish the crash names it.
static COMPILER_CRASH: std::sync::Mutex<Option<String>> =
    std::sync::Mutex::new(None);

/// Establish the run's compiler-crash verdict from a
/// compiler-bug panic record a verdict authority scoped —
/// the collector replay in `mono_check`, over the instances
/// it collects and the bodies whose materialization it
/// replays. The driver's final report points consult the
/// slot before the clean-failure reports and report it as a
/// compiler crash (Issue #66).
pub fn note_compiler_crash(record: String) {
    if let Ok(mut slot) = COMPILER_CRASH.lock() {
        if slot.is_none() {
            *slot = Some(record);
        }
    }
}

/// Take the run's compiler-crash verdict, clearing the slot.
pub fn take_compiler_crash() -> Option<String> {
    COMPILER_CRASH.lock().ok().and_then(|mut slot| slot.take())
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

/// The type an unsafe binder type wraps, as it stands under
/// the binder: `unsafe<'a> &'a T` erases to `&'a T`, whose
/// region still mentions the binder's bound variable — a
/// region the converter treats opaquely, like any other.
/// An unsafe binder is transparent at the
/// value level — wrapping and unwrapping are identity
/// operations on the wrapped value — so a body that merely
/// mentions a binder type (a local's type, an argument or
/// return type, a field type) converts exactly as if the
/// binder were not there; only the wrap/unwrap constructs
/// themselves need the bridge conversions that stay
/// unimplemented (see [`UNSAFE_BINDER_REASON`]). The question
/// goes through the internal view, like
/// [`coroutine_closure_const_parts`]: the stable kind
/// question is itself the unimplemented conversion. Returns
/// `None` for a type that is not an unsafe binder, and when
/// the internal view is unavailable.
pub fn unsafe_binder_inner_ty(ty: &rustc_public::ty::Ty) -> Option<rustc_public::ty::Ty> {
    screen_question(|| {
        let tcx = compiler_tcx()?;
        let internal = rustc_public::rustc_internal::internal(tcx, ty.clone());
        let mut inner = internal;
        while let rustc_middle::ty::TyKind::UnsafeBinder(binder) = inner.kind() {
            inner = *binder.as_ref().skip_binder();
        }
        if inner == internal {
            return None;
        }
        Some(rustc_public::rustc_internal::stable(inner))
    })
}

/// Run a Stable MIR query under the coroutine-closure guard:
/// a query whose answer needs to render or inspect a
/// coroutine-closure (async closure) type panics inside
/// rustc_public (see [`COROUTINE_CLOSURE_TYPE_REASON`]); the
/// guard classifies that panic and answers `None`, and
/// re-raises any other panic untouched. The guard answers
/// `None` for the unsafe-binder type site as well (see
/// [`captured_panic_is_unsafe_binder_ty`]): a kind question
/// about a binder type has no stable answer, and the caller
/// erases the binder instead (see [`unsafe_binder_inner_ty`]).
pub fn guarded_query<T>(f: impl FnOnce() -> T) -> Option<T> {
    clear_captured_panic();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(value) => Some(value),
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
                Some(record)
                    if record.contains("FIXME(async_closures)")
                        || captured_panic_is_unsafe_binder_ty(&record) =>
                {
                    None
                }
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

pub fn guarded_ty_kind(ty: &rustc_public::ty::Ty) -> Option<rustc_public::ty::TyKind> {
    guarded_query(|| ty.kind())
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

/// The reason a body fetch reports when materializing the
/// instance's body fails inside the compiler itself: fetching
/// a body normalizes the body's types after erasing regions
/// (`TyCtxt::normalize_erasing_regions`), and for an instance
/// whose instantiation recursion is unbounded the trait solver
/// overflows inside that normalization, which the compiler
/// surfaces as a panic at the normalization entry point rather
/// than as a diagnostic. rustc's monomorphization collector
/// never materializes such an instance's whole body this way;
/// it detects the instantiation recursion itself and reports
/// "reached the recursion limit while instantiating" as a
/// compilation error. The body fetch sites therefore contain
/// this panic class (identified by its recorded location, see
/// [`captured_panic_is_normalize_overflow`]) and report it
/// under this name; their callers route it to the same
/// recursion-limit compilation failure as the demand passes'
/// own depth mirror.
pub const BODY_FETCH_NORMALIZE_OVERFLOW: &str = "body fetch: type normalization \
    overflowed the recursion limit";

/// Whether a recorded panic text (the hook text recorded by
/// `record_captured_panic`: the payload text plus the panic
/// location) is the trait solver's overflow inside whole-body
/// normalization after erasing regions. The panic's payload is
/// a non-string internal-error marker, so the class is
/// identified by its recorded location instead — the entry
/// point of `TyCtxt::normalize_erasing_regions` in the pinned
/// compiler (rustc_middle/src/ty/normalize_erasing_regions.rs:
/// 202).
pub fn captured_panic_is_normalize_overflow(text: &str) -> bool {
    text.contains("rustc_middle/src/ty/normalize_erasing_regions.rs:202:")
}

/// Fetch-failure reason recorded when fetching a body trips the
/// MIR validator's ban on projecting into a SIMD type (MCP#838).
/// Compiler-generated impl bodies over SIMD types (a derived
/// `PartialEq::eq`, say) project the ADT's fields and cannot be
/// materialized at all; the demand passes refuse them up front
/// under this same reason (`generated_simd_impl_method`), and a
/// body fetch that reaches the validator anyway is contained
/// and recorded with it. Hand-written bodies are unaffected:
/// their own field accesses compile to the pointer-cast idiom,
/// not to a projection the validator bans.
pub const BODY_FETCH_SIMD_VALIDATOR: &str =
    "compiler-generated impl body over a SIMD type";

/// Whether a captured panic record is a MIR validation
/// failure: every failure of the validator's `CfgChecker`
/// panics at the one `fail` site (`validate.rs:81` in the
/// pinned toolchain), and the panic payload is a non-string
/// bug marker, so the recorded location is all the record
/// carries. The location alone does not say WHICH check
/// failed; callers contain this class only for fetches of
/// compiler-generated impl bodies over SIMD types, where the
/// failing check is the MCP#838 ban on projecting into the
/// ADT (see `BODY_FETCH_SIMD_VALIDATOR`).
pub fn captured_panic_is_mir_validator_fail(text: &str) -> bool {
    text.contains("rustc_mir_transform/src/validate.rs:81:")
}

/// Whether this instance is a compiler-generated impl method
/// (a derived impl) over a `#[repr(simd)]` `Self` type. A
/// derived method carries the very span of its impl — both are
/// the derive attribute's span — while a hand-written method's
/// span is its own definition inside the impl block, never the
/// impl's. The distinction matters because a derived body is
/// synthesized by projecting the ADT's fields, and projecting
/// into a SIMD type is exactly what rustc's MIR validator
/// bans (MCP#838): asking for such a body can emit the ICE
/// diagnostic that poisons the whole compilation, even when
/// the panic itself is contained. A hand-written body chooses
/// its own accesses (the `as_array` pointer-cast idiom is the
/// known shape) and is safe to fetch.
pub fn generated_simd_impl_method(inst: &Instance) -> bool {
    let def = inst.def.def_id();
    let Some(parent) = def.parent() else {
        return false;
    };
    if def.span() != parent.span() {
        return false;
    }
    // The instance's own arguments are relative to the impl
    // method's definition (its `T`, `N`, ...), not the trait's,
    // so the `Self` type is read off the method's receiver: the
    // first input of the instance's signature, behind the
    // reference a `&self` receiver adds.
    let Some(sig) = guarded_ty_kind(&inst.ty())
        .and_then(|kind| kind.fn_sig())
        .map(|sig| sig.skip_binder())
    else {
        return false;
    };
    let Some(receiver) = sig.inputs_and_output.first() else {
        return false;
    };
    let self_ty = match guarded_ty_kind(receiver) {
        Some(rustc_public::ty::TyKind::RigidTy(
            rustc_public::ty::RigidTy::Ref(_, inner, _),
        )) => inner,
        _ => receiver.clone(),
    };
    guarded_ty_kind(&self_ty).is_some_and(|kind| kind.is_simd())
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

#[cfg(test)]
mod tests {
    use super::*;

    const DEV: &str =
        "/rustc-dev/0ef1bc386059596dada8fb65b50425aab2ea637a/compiler";

    fn non_string(location: &str) -> String {
        format!("panic with a non-string payload (panicked at {location})")
    }

    /// The three pinned ICEs of Issue #509's class (a):
    /// `bug!` records from `compiler/rustc_*` crates with the
    /// non-string `ExplicitBug` payload.
    #[test]
    fn pinned_ices_are_compiler_bugs() {
        for location in [
            format!("{DEV}/rustc_middle/src/mir/consts.rs:338:25"),
            format!("{DEV}/rustc_middle/src/ty/layout.rs:1411:17"),
            format!("{DEV}/rustc_middle/src/ty/instance.rs:640:21"),
        ] {
            assert!(
                captured_panic_is_compiler_bug(&non_string(&location)),
                "{location}"
            );
        }
    }

    /// String-payload compiler panics occur on speculative
    /// probes and inside the verdict pass's own checks
    /// without being the compiler's bug machinery: the
    /// `std::fmt::Write` dummy-binder panic and the symbol
    /// mangler's `unimplemented!()` are the standing
    /// witnesses (54888 and 154367 keep their verdicts).
    #[test]
    fn string_payload_compiler_panics_are_not_bugs() {
        let fmt = format!(
            "`<dyn std::fmt::Write as std::marker::Sized>` has escaping              bound vars, so it cannot be wrapped in a dummy binder.              (panicked at {DEV}/rustc_middle/src/ty/predicate.rs:490:9)"
        );
        assert!(!captured_panic_is_compiler_bug(&fmt));
        let mangling = format!(
            "not implemented (panicked at              {DEV}/rustc_symbol_mangling/src/v0.rs:646:37)"
        );
        assert!(!captured_panic_is_compiler_bug(&mangling));
    }

    /// The bridge's unimplemented conversions are conversion
    /// gaps with their own classification, not compiler bugs.
    #[test]
    fn bridge_sites_are_not_bugs() {
        let bridge = format!(
            "not implemented (panicked at              {DEV}/rustc_public/src/unstable/convert/stable/ty.rs:471:36)"
        );
        assert!(!captured_panic_is_compiler_bug(&bridge));
    }

    /// The compiler classes that already carry named
    /// verdicts keep them: the whole-body normalization
    /// overflow (a recursion-limit failure), the MIR
    /// validator failure (contained for SIMD-generated
    /// fetches), and the delayed-bug flush (reported by its
    /// own classification in the driver).
    #[test]
    fn named_verdict_classes_are_not_bugs() {
        for location in [
            format!(
                "{DEV}/rustc_middle/src/ty/normalize_erasing_regions.rs:202:17"
            ),
            format!("{DEV}/rustc_mir_transform/src/validate.rs:81:24"),
            format!("{DEV}/rustc_errors/src/lib.rs:1499:9"),
        ] {
            assert!(
                !captured_panic_is_compiler_bug(&non_string(&location)),
                "{location}"
            );
        }
    }

    /// Panics outside the compiler — the driver's own code,
    /// the standard library — are never compiler bugs,
    /// whatever their payload.
    #[test]
    fn non_compiler_panics_are_not_bugs() {
        assert!(!captured_panic_is_compiler_bug(&non_string(
            "crates/sparagmos-driver/src/convert/body.rs:100:5"
        )));
        assert!(!captured_panic_is_compiler_bug(
            "boom (panicked at library/std/src/io/mod.rs:10:5)"
        ));
        assert!(!captured_panic_is_compiler_bug("no location at all"));
    }

    /// The witness list is append-only and scoped reads see
    /// exactly the witnesses at or after their snapshot; a
    /// non-bug panic is never noted.
    #[test]
    fn witness_list_scopes_by_snapshot() {
        let before = compiler_bug_witness_len();
        assert_eq!(compiler_bug_witness_since(before), None);
        note_panic_observation("an ordinary string panic");
        assert_eq!(compiler_bug_witness_since(before), None);
        let record = non_string(&format!(
            "{DEV}/rustc_middle/src/ty/instance.rs:640:21"
        ));
        note_panic_observation(&record);
        assert_eq!(compiler_bug_witness_since(before), Some(record));
    }

    /// The const-expr-normalize site (Issue #510) is named
    /// by its location and non-string payload alone: a
    /// string-payload panic at the same site, and the
    /// non-string panics of the other pinned sites, are not
    /// this class.
    #[test]
    fn const_expr_normalize_site_is_named() {
        let record = non_string(&format!(
            "{DEV}/rustc_middle/src/mir/consts.rs:338:25"
        ));
        assert!(captured_panic_is_const_expr_normalize(&record));
        // It stays a compiler bug for the scopes that force
        // it themselves (the collector replay's crash
        // verdict); only the materialization scope discounts
        // its own firings.
        assert!(captured_panic_is_compiler_bug(&record));
        let string_at_site = format!(
            "boom (panicked at {DEV}/rustc_middle/src/mir/consts.rs:338:25)"
        );
        assert!(!captured_panic_is_const_expr_normalize(&string_at_site));
        let other_site = non_string(&format!(
            "{DEV}/rustc_middle/src/ty/layout.rs:1411:17"
        ));
        assert!(!captured_panic_is_const_expr_normalize(&other_site));
    }

    /// The panic hook names the non-string payloads of known
    /// sites (Issue #395's diagnosability work), so their
    /// records carry `"<name> (panic with a non-string
    /// payload)"` as the message part. The classifiers key on
    /// the payload kind, not the exact rendering, so a named
    /// record classifies exactly as the plain form did: the
    /// const-expr site is still the const-expr class and
    /// still a compiler bug, and the normalization overflow
    /// keeps its named verdict (not a bug).
    #[test]
    fn named_non_string_payloads_keep_their_classification() {
        let const_expr = format!("constant-expression normalization failure (panic with a non-string payload) (panicked at {DEV}/rustc_middle/src/mir/consts.rs:338:25)");
        assert!(captured_panic_is_const_expr_normalize(&const_expr));
        assert!(captured_panic_is_compiler_bug(&const_expr));
        let overflow = format!("type normalization overflow (panic with a non-string payload) (panicked at {DEV}/rustc_middle/src/ty/normalize_erasing_regions.rs:202:17)");
        assert!(!captured_panic_is_compiler_bug(&overflow));
        assert!(captured_panic_is_normalize_overflow(&overflow));
        let validator = format!("MIR validation failure (panic with a non-string payload) (panicked at {DEV}/rustc_mir_transform/src/validate.rs:81:24)");
        assert!(!captured_panic_is_compiler_bug(&validator));
        assert!(captured_panic_is_mir_validator_fail(&validator));
    }

    /// The unsafe-binder type site (Issue #510) is named by
    /// its bare "not implemented" payload and its location
    /// alone: the same payload at the const-expr site, and a
    /// different payload at the binder site, are not this
    /// class.
    #[test]
    fn unsafe_binder_ty_site_is_named() {
        let record = format!(
            "not implemented (panicked at /rustc/compiler/rustc_public/src/unstable/convert/stable/ty.rs:471:9)"
        );
        assert!(captured_panic_is_unsafe_binder_ty(&record));
        let const_expr_site = format!(
            "not implemented (panicked at /rustc/compiler/rustc_public/src/unstable/convert/stable/ty.rs:577:9)"
        );
        assert!(!captured_panic_is_unsafe_binder_ty(&const_expr_site));
        let other_payload = format!(
            "boom (panicked at /rustc/compiler/rustc_public/src/unstable/convert/stable/ty.rs:471:9)"
        );
        assert!(!captured_panic_is_unsafe_binder_ty(&other_payload));
    }

    /// The const-expr witness list is append-only and scoped
    /// reads see exactly the witnesses at or after their
    /// snapshot; an unrelated panic is never noted.
    #[test]
    fn const_expr_witness_list_scopes_by_snapshot() {
        let before = const_expr_normalize_witness_len();
        assert_eq!(const_expr_normalize_witness_since(before), None);
        note_panic_observation("an ordinary string panic");
        assert_eq!(const_expr_normalize_witness_since(before), None);
        let record = non_string(&format!(
            "{DEV}/rustc_middle/src/mir/consts.rs:338:25"
        ));
        note_panic_observation(&record);
        assert_eq!(
            const_expr_normalize_witness_since(before),
            Some(record)
        );
    }

    /// The crash slot is first-wins and clears on take.
    #[test]
    fn crash_slot_is_first_wins() {
        let _ = take_compiler_crash();
        note_compiler_crash("first".to_string());
        note_compiler_crash("second".to_string());
        assert_eq!(take_compiler_crash(), Some("first".to_string()));
        assert_eq!(take_compiler_crash(), None);
    }
}
