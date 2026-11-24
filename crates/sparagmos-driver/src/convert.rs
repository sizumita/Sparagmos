//! Conversion from `rustc_public` (Stable MIR, as of Rust 1.100.0) types
//! to `sparagmos_mir`.
//!
//! Policy: a function is converted completely or not at all. Any variant
//! we do not handle makes the whole function "skipped" and is counted in
//! the report with its variant name — constructs are never silently
//! dropped or approximated inside an otherwise-converted body.
//!
//! API notes for Rust 1.100.0's `rustc_public` (verified against the
//! `rustc-src` sources shipped with the toolchain):
//! - `TerminatorKind` exposes 10 variants (no Yield/FalseEdge/TailCall).
//! - `BinOp` has no `*WithOverflow` variants; overflow is expressed with
//!   plain ops plus `Assert` terminators in this MIR form.
//! - `BasicBlock` does not expose `is_cleanup`, so converted bodies
//!   always carry `is_cleanup: false` (a Stable MIR limitation, not a
//!   conversion choice).
//! - Constants are allocation-based (`ConstantKind::Allocated` /
//!   `TyConstKind::Value`); scalar values are read out of the allocation
//!   bytes using the constant's own type, so sizes are exact.

use std::collections::{HashMap, HashSet};

use rustc_public::abi::{FieldsShape, TagEncoding, VariantsShape};
use rustc_public::mir;
use rustc_public::mir::mono::{Instance, InstanceKind};
use rustc_public::ty::{
    AdtDef, AdtKind, AliasKind as SmirAliasKind, Allocation, AssocContainer, AssocKind, Binder,
    ClosureKind, ConstantKind, ExistentialPredicate, ExistentialProjection, ExistentialTraitRef,
    FloatTy as SmirFloatTy, FnDef, FnSig, GenericArgKind, GenericArgs, ImplDef,
    IntTy as SmirIntTy, Region, RegionKind, RigidTy, TermKind, Ty, TyConst,
    TyConstKind, TyKind, UintTy as SmirUintTy, VariantIdx,
};
use rustc_public_bridge::IndexedVal;
use rustc_public::{CrateDef, CrateDefType};

use sparagmos_mir as sm;

thread_local! {
    /// The running compilation's compiler context, handed to
    /// the library by the binary's compiler callback (which
    /// receives it from `rustc_public`'s `run_with_tcx` entry).
    /// [`Converter::collapsed_filename_lines`] consults it to
    /// resolve statement spans the way the compiler's own
    /// debuginfo does. The context is valid only for the length
    /// of the callback; the binary clears the slot when the
    /// callback returns. The lifetime is erased to 'static for
    /// storage: the stored value is the callback's own context,
    /// which outlives every conversion performed inside it, and
    /// it is never read outside the callback.
    static COMPILER_TCX: std::cell::Cell<Option<rustc_middle::ty::TyCtxt<'static>>> =
        const { std::cell::Cell::new(None) };
}

/// Install (or clear, with `None`) the compiler context the
/// span resolution consults. Called by the binary's compiler
/// callback around the conversion entry.
pub fn set_compiler_tcx(tcx: Option<rustc_middle::ty::TyCtxt<'_>>) {
    let erased = tcx.map(|t| unsafe {
        std::mem::transmute::<rustc_middle::ty::TyCtxt<'_>, rustc_middle::ty::TyCtxt<'static>>(t)
    });
    COMPILER_TCX.with(|slot| slot.set(erased));
}

pub(crate) fn compiler_tcx() -> Option<rustc_middle::ty::TyCtxt<'static>> {
    COMPILER_TCX.with(|slot| slot.get())
}

/// Run `f` — a Stable MIR body materialization — with the
/// diagnostics its eager constant evaluation emits discounted.
///
/// `Instance::body()` and its siblings hand back a body that is
/// already monomorphized with every constant operand evaluated
/// (rustc_public's body builder evaluates them as it converts).
/// That evaluation is speculative from this driver's point of
/// view: the driver materializes the bodies of items and
/// instances the compiler's own monomorphization collector
/// would never collect — unreachable local functions, instances
/// demanded only from dead code, bodies fetched to inspect a
/// shape — and a constant that fails evaluation there (a
/// panicking associated constant behind `if false` is the shape
/// of Issue #417) is reported by the const-eval query as a
/// session error even though the body itself is still produced,
/// with the constant left unevaluated. Left standing in the
/// session's error count, that one report fails the whole
/// compilation of a program rustc accepts.
///
/// The discount is scoped to body materialization on purpose.
/// Which constant failures are compilation errors is decided by
/// the driver's own authorities, never by a side effect of
/// fetching a body: the front end for the constants it evaluates
/// itself before collection begins (statics, constants in item
/// types), and `mono_check`'s replay of the monomorphization
/// collector for the required constants of the instances that
/// collector actually collects. Diagnostics emitted outside a
/// materialization keep their usual effect.
///
/// The discount is recorded, not forgotten, and it is scoped
/// by the class of diagnostic. The diagnostics a materialization
/// can emit fall in two classes with different verdicts:
///
/// - Lint errors (the known-panics reports are the known shape):
///   rustc lints every local body it builds, collected or not,
///   so a lint error from materializing a body is a compilation
///   error under rustc too. These are never discounted into a
///   conversion: the fetch records the body's key in the lint
///   registry, and the compile verdict fails on any entry.
/// - Evaluation errors (E0080, including the validity errors of
///   a body's constant operands): rustc produces these only for
///   the bodies its monomorphization collector actually
///   collects. These are discounted at the fetch, but the
///   body's key (the `def_path_str_with_args` form the Stable
///   MIR name of the instance already has) is recorded, and
///   `mono_check` — which replays the collector — turns a hit
///   on a collected instance into a finding. Bodies the
///   collector never reaches stay discounted for good.
///
/// The classes are told apart by whether the session gained a
/// non-lint error during the fetch. A fetch that gains both
/// classes at once is recorded as an evaluation error; for a
/// collected body the collector replay still fails it, so the
/// only residue is an uncollected body carrying both a lint
/// violation and a failing constant, which converts.
pub fn with_body_materialization<R>(key: &str, f: impl FnOnce() -> R) -> R {
    let Some(tcx) = compiler_tcx() else {
        return f();
    };
    let dcx = tcx.dcx();
    let errors_before = dcx.err_count();
    let non_lint_before = dcx.has_errors_excluding_lint_errors().is_some();
    // Contained, not handled: a panic out of the fetch keeps its
    // payload for the driver's own containment layers, but the
    // diagnostics emitted before it belong to the same
    // speculative materialization and are discounted either way.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    if dcx.err_count() > errors_before {
        let non_lint_after = dcx.has_errors_excluding_lint_errors().is_some();
        if non_lint_after && !non_lint_before {
            note_materialization_emission(key);
        } else {
            note_materialization_lint(key);
        }
        dcx.reset_err_count();
    }
    match result {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

thread_local! {
    /// Keys of the bodies whose materialization emitted
    /// evaluation diagnostics that `with_body_materialization`
    /// discounted, in the `def_path_str_with_args` form (a
    /// generic definition's own body is keyed by its plain def
    /// path). Written during collection, read by the compile
    /// verdict (`mono_check::check_required_consts`); one
    /// compilation runs per process, so nothing clears it.
    static MATERIALIZATION_EMISSIONS: std::cell::RefCell<std::collections::HashSet<String>> =
        std::cell::RefCell::new(std::collections::HashSet::new());

    /// Keys of the bodies whose materialization emitted a lint
    /// error. Same lifecycle as `MATERIALIZATION_EMISSIONS`.
    static MATERIALIZATION_LINTS: std::cell::RefCell<std::collections::HashSet<String>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

fn note_materialization_emission(key: &str) {
    if key.is_empty() {
        return;
    }
    MATERIALIZATION_EMISSIONS.with(|set| {
        set.borrow_mut().insert(key.to_string());
    });
}

fn note_materialization_lint(key: &str) {
    if key.is_empty() {
        return;
    }
    MATERIALIZATION_LINTS.with(|set| {
        set.borrow_mut().insert(key.to_string());
    });
}

/// Whether materializing the body named by `key` emitted
/// evaluation diagnostics that were discounted at the fetch
/// (see `with_body_materialization`).
pub fn materialization_emitted(key: &str) -> bool {
    MATERIALIZATION_EMISSIONS.with(|set| set.borrow().contains(key))
}

/// A body whose materialization emitted a lint error, if any:
/// rustc lints every local body, so each one is a compilation
/// error there (see `with_body_materialization`).
pub fn materialization_lint() -> Option<String> {
    MATERIALIZATION_LINTS.with(|set| set.borrow().iter().next().cloned())
}

/// Whether the running compilation's crate graph defines the
/// `drop_glue` lang item. A crate compiled against a minicore
/// (which defines only the lang items it names itself) has none:
/// asking the compiler to resolve `drop_in_place` there emits a
/// fatal "requires `drop_glue` lang_item" diagnostic that no
/// `catch_unwind` can contain, so drop-glue certification must
/// not ask (see `certify_drop_trivial` in the binary). Outside a
/// compilation callback there is no graph to consult; the item
/// is reported defined, preserving the caller's plain behavior.
pub fn drop_glue_lang_item_defined() -> bool {
    let Some(tcx) = compiler_tcx() else {
        return true;
    };
    tcx.lang_items()
        .get(rustc_attr_ir::lang_items::LangItem::DropGlue)
        .is_some()
}

/// Whether the static `def` is a foreign static (an item of an
/// `extern` block). A foreign static has no initializer by
/// construction, and asking the compiler to evaluate one ICEs
/// inside the typeck query ("can't type-check body of ..."):
/// evaluating a static initializer goes through the item's MIR
/// body, which a foreign item does not have. The ICE diagnostic
/// poisons the whole compilation even when the panic itself is
/// contained, so the evaluation must not be attempted — callers
/// handle the storage without it (a reference decoded against
/// an uninitialized seed, or no decode at all).
/// Outside a compilation callback there is no context to ask;
/// the static is reported non-foreign, preserving the caller's
/// plain behavior.
fn static_is_foreign(def: &mir::mono::StaticDef) -> bool {
    let Some(tcx) = compiler_tcx() else {
        return false;
    };
    let internal = rustc_public::rustc_internal::internal(tcx, def.def_id());
    tcx.is_foreign_item(internal)
}

/// Per-run converter state (DefId assignment) plus per-function error bag.
pub struct Converter {
    def_ids: HashMap<rustc_public::DefId, u32>,
    /// Sparagmos ids of resolved function *instances*, keyed by the
    /// referenced definition together with the instance it resolves
    /// to: two monomorphizations of one generic function are
    /// different call targets and must not share an id (the
    /// interpreter resolves calls through these ids), while the id's
    /// name stays the referenced definition's name — a trait method
    /// call keeps the trait method's name even though the resolved
    /// instance is an impl method, because builtin lookup and
    /// diagnostics key on the referenced name.
    instance_ids: HashMap<(rustc_public::DefId, Instance), u32>,
    pub def_names: Vec<String>,
    /// Discriminant values of every enum ADT converted so far,
    /// keyed by the converted `DefId` (variant index -> value).
    /// Handed to the interpreter with the converted program: its
    /// `Discriminant` rvalues must yield these values, because the
    /// `SwitchInt` branch keys carry values, not variant indices.
    pub adt_discriminants: sm::AdtDiscriminants,
    /// Layouts of the monomorphic ADT instances converted so far,
    /// keyed by the rendered instance name (the interpreter's type
    /// tag). Handed to the interpreter with the converted program:
    /// its `size_of` / `align_of` builtins consult the table for ADT
    /// tags. Filled from the compiler's own layout query, so repr
    /// attributes are already reflected; instances whose layout the
    /// compiler cannot give (generic parameters, unsized types) get
    /// no entry.
    pub adt_layouts: sm::AdtLayouts,
    /// Per-variant field byte layouts of the monomorphic ADT
    /// instances converted so far, keyed like `adt_layouts`
    /// (see `sm::AdtFieldLayouts`). Handed to the interpreter
    /// with the converted program: its byte view of an ADT
    /// value's storage resolves raw byte pointer positions
    /// through the field spans recorded here. Filled from the
    /// same layout query as `adt_layouts`.
    pub adt_field_layouts: sm::AdtFieldLayouts,
    /// Storage layouts of the monomorphic tuple types
    /// converted so far, keyed by the tuple type's rendered
    /// name (see `sm::TupleLayouts`). Handed to the
    /// interpreter with the converted program: its scalar
    /// shape of a tuple reads the fields at the recorded
    /// offsets. Filled from the same layout query as
    /// `adt_layouts`.
    pub tuple_layouts: sm::TupleLayouts,
    /// Direct discriminant tag layouts of the monomorphic
    /// enum instances converted so far, keyed like
    /// `adt_layouts` (see `sm::AdtTagLayouts`). Handed to the
    /// interpreter with the converted program: its byte view
    /// reads and writes an enum's tag span as the
    /// discriminant's own bytes. Filled from the same layout
    /// query as `adt_layouts`.
    pub adt_tag_layouts: sm::AdtTagLayouts,
    /// Materialisation plans for the pointee types of pointer
    /// casts converted so far, keyed by the converted pointee
    /// type's `Debug` rendering (the same rendering
    /// `mir_convert` looks up). Handed to the interpreter with
    /// the converted program: when a cast's operand is a
    /// type-erased constant allocation, the plan materialises
    /// the block into the cast target's type (see
    /// `decode_erased_alloc`).
    pub mat_plans: HashMap<String, sparagmos_interp::ir::MatPlan>,
    /// Storage-slot maps of the monomorphic coroutine instances
    /// converted so far, keyed like `adt_layouts` (see
    /// `sm::CoroutineLayouts`). Handed to the interpreter with the
    /// converted program: its place conversion resolves coroutine
    /// state-field projections through these maps.
    pub coroutine_layouts: sm::CoroutineLayouts,
    /// Compiler types whose coroutine slot-map recording was
    /// already attempted, keyed by the type handle itself (the
    /// rendered key is the expensive part, as in
    /// `layout_attempted_tys`).
    coroutine_slots_attempted: std::collections::HashSet<rustc_public::ty::Ty>,
    /// Rendered instance keys whose layout query was already
    /// attempted (successfully or not), so a type whose query
    /// cannot be answered is not re-queried at every conversion.
    layout_attempted: std::collections::HashSet<String>,
    /// Compiler types whose closure/coroutine layout recording
    /// was already attempted, keyed by the type handle itself:
    /// the rendered key is the whole instance rendering, so
    /// computing it for an already-attempted type is the
    /// expensive part (a deeply nested instance renders its
    /// entire type tree), and the handle check must come first.
    layout_attempted_tys: std::collections::HashSet<rustc_public::ty::Ty>,
    /// Name table for rendering ADT instance keys, kept in sync
    /// with `def_names` (which is indexed by the converted `DefId`).
    layout_names: std::collections::HashMap<sm::DefId, String>,
    /// Declared shapes of every ADT converted so far, keyed by the
    /// converted `DefId` (see `sm::AdtShape`). Handed to the
    /// interpreter with the converted program, which uses them to
    /// decide which types are zero-sized.
    pub adt_shapes: sm::AdtShapes,
    /// Derived-`Debug` shapes of the monomorphic ADT instances
    /// converted so far, keyed by the rendered instance name (see
    /// `sm::AdtDebugShapes`). Handed to the interpreter with the
    /// converted program: its `Debug` formatter prints ADT values
    /// from these shapes.
    pub adt_debug_shapes: sm::AdtDebugShapes,
    /// Debugger shapes of the monomorphic closure/coroutine
    /// instances converted so far, keyed by the rendered
    /// instance name (see `sm::EnvDebugShapes`). Handed to the
    /// interpreter with the converted program: its debugger
    /// interface prints closure and coroutine values from these
    /// shapes.
    pub env_debug_shapes: sm::EnvDebugShapes,
    /// Compiler types whose environment-shape recording was
    /// already attempted, keyed by the type handle itself (as in
    /// `layout_attempted_tys`).
    env_shape_attempted: std::collections::HashSet<rustc_public::ty::Ty>,
    /// Depth of nested `record_debug_shape` calls currently
    /// unfolding field types (see there).
    debug_shape_depth: usize,
    /// Thread-local statics converted so far, keyed by the
    /// converted `DefId` (see `sm::ThreadLocalDef`). Handed to the
    /// interpreter with the converted program, which seeds each
    /// thread's storage from these initializers. Filled when a
    /// `ThreadLocalRef` rvalue is converted; a static whose
    /// initializer cannot be decoded gets no entry, and its
    /// `ThreadLocalRef` stays an explicit error at run time.
    pub thread_locals: sm::ThreadLocals,
    /// Field types of monomorphic ADT instances whose generic
    /// shape mentions a type alias, keyed by the rendered instance
    /// name (see `sm::AdtInstanceShapes`). Handed to the
    /// interpreter with the converted program.
    pub adt_instance_shapes: sm::AdtInstanceShapes,
    pub adt_niches: sm::AdtNiches,
    /// Source-file table of the converted program: the file names
    /// the per-body source lines (see `sm::Body::source_lines`)
    /// index into, in first-use order. Handed to the interpreter
    /// with the converted program, for the debugger interface.
    pub source_files: Vec<String>,
    /// Interning map for `source_files`.
    source_file_ids: HashMap<String, u32>,
    /// Rendered instance keys whose instance-shape recording was
    /// already attempted (successfully or not), so an instance
    /// whose fields cannot be normalized is not retried at every
    /// conversion.
    instance_shape_attempted: std::collections::HashSet<String>,
    /// Declared types of the target crate's statics, recorded by
    /// the item scan before conversion (`note_static_ty`).
    static_tys: HashMap<sm::DefId, rustc_public::ty::Ty>,
    /// The same declared types converted to Sparagmos MIR types,
    /// handed to the interpreter with the program (see
    /// `sparagmos_mir::StaticTys`).
    pub static_tys_mir: sm::StaticTys,
    /// Definition-location stamps of the `#[track_caller]`
    /// instances registered so far, keyed by the instance's
    /// Sparagmos `DefId` (see `note_track_caller_instance`).
    /// Handed to the interpreter with the converted program: a
    /// call that reaches a track-caller body through a shim (a
    /// function pointer, a vtable slot) carries no call-site
    /// stamp, and the interpreter binds this definition stamp
    /// instead, which is the location the compiler's shim
    /// presents to the callee.
    pub track_caller_defs: HashMap<sm::DefId, sm::ConstValue>,
    /// Declared ABIs of the function instances registered so
    /// far, keyed by the instance's Sparagmos `DefId` (see
    /// `note_fn_abi`). Handed to the interpreter with the
    /// converted program: a naked function's incoming arguments
    /// arrive in the registers of its declared ABI, which the
    /// interpreter's seed plan for the function's asm block is
    /// computed from.
    pub fn_abis: HashMap<sm::DefId, sm::Abi>,
    /// The location stamps of the `Fn` / `FnMut` / `FnOnce`
    /// vtable shims — the trait method definitions in core,
    /// which compiler-generated `Fn` impls are attributed to —
    /// recorded by the dyn demand pass (see
    /// `note_fn_shim_location`). Index 0 is `call`, 1 is
    /// `call_mut`, 2 is `call_once`.
    pub fn_shim_locations: [Option<sm::ConstValue>; 3],
    /// Link symbols of the instances registered so far, keyed
    /// by the instance's Sparagmos `DefId` (see
    /// `note_link_symbol`). Handed to the interpreter with the
    /// converted program: an extern declaration and the
    /// in-program definition exporting its symbol (a
    /// `#[no_mangle]` / `#[export_name]` function, possibly of
    /// an auxiliary crate) share no `DefId`, but a linker binds
    /// them by this name, and the interpreter's call conversion
    /// does the same.
    pub link_symbols: HashMap<sm::DefId, String>,
    /// Reentrancy guard for `note_track_caller_instance`: the
    /// stamp decode itself resolves (and registers) instances.
    noting_track_caller: bool,
    /// Unsupported hits of the function currently being converted.
    errors: Vec<String>,
    /// Closures materialized as *values* out of constants during
    /// conversion so far: a function pointer stored in a constant
    /// allocation that names a closure (directly or through its
    /// `Fn` adapter shim), or a closure value nested inside a
    /// constant aggregate. No converted body's types mention these
    /// closures — the decoded value is the only mention — so the
    /// type-driven demand passes never see them; the driver binary
    /// collects their bodies from this record instead (see
    /// `collect_const_closure_instances` in the binary). A mention
    /// is recorded when a constant is decoded during a conversion
    /// attempt, mirroring how the reachability collection walks
    /// the targets of bodies whose conversion failed.
    pub const_closure_mentions:
        Vec<(rustc_public::ty::ClosureDef, rustc_public::ty::GenericArgs)>,
    /// Deduplication keys for `const_closure_mentions`
    /// (definition + rendered generic arguments, the key shape
    /// the demand passes use for unresolved mentions).
    const_closure_mentions_seen:
        std::collections::HashSet<(rustc_public::DefId, String)>,
    /// Function items materialized as values out of constants,
    /// the plain-function counterpart of
    /// `const_closure_mentions`: a function pointer stored
    /// inside a constant allocation (a `RawWakerVTable`'s
    /// entries are the canonical case) names a body no converted
    /// body's call operands mention, because the call goes
    /// through the decoded pointer value at run time. The driver
    /// binary collects these bodies from this record instead
    /// (see `collect_const_fn_instances` in the binary).
    pub const_fn_mentions: Vec<rustc_public::mir::mono::Instance>,
    /// Deduplication keys for `const_fn_mentions` (definition +
    /// rendered generic arguments).
    const_fn_mentions_seen:
        std::collections::HashSet<(rustc_public::DefId, String)>,
}

impl Default for Converter {
    fn default() -> Self {
        Self::new()
    }
}


/// Whether a converted type mentions a type alias anywhere
/// inside it (see `Converter::record_instance_shape`).
/// Whether an ADT instance's generic arguments mention a
/// generic parameter: a type parameter, a const parameter,
/// a bound variable, or a free region (early-param or
/// placeholder). A field's declared type can only mention
/// the ADT's own parameters, so the field types after
/// substitution are parameter-free exactly when the
/// instance's arguments are — which is the condition for
/// normalizing them in the driver's codegen typing
/// environment (see `Converter::record_instance_shape`).
fn instance_mentions_param(args: &GenericArgs) -> bool {
    use rustc_public::visitor::Visitable;
    struct MentionsParam(bool);
    impl rustc_public::visitor::Visitor for MentionsParam {
        type Break = ();
        fn visit_ty(
            &mut self,
            ty: &rustc_public::ty::Ty,
        ) -> std::ops::ControlFlow<()> {
            match ty.kind() {
                rustc_public::ty::TyKind::Param(_)
                | rustc_public::ty::TyKind::Bound(..) => {
                    self.0 = true;
                    std::ops::ControlFlow::Break(())
                }
                _ => ty.super_visit(self),
            }
        }
        fn visit_const(
            &mut self,
            c: &rustc_public::ty::TyConst,
        ) -> std::ops::ControlFlow<()> {
            match c.kind() {
                rustc_public::ty::TyConstKind::Param(_)
                | rustc_public::ty::TyConstKind::Bound(..) => {
                    self.0 = true;
                    std::ops::ControlFlow::Break(())
                }
                _ => c.super_visit(self),
            }
        }
        fn visit_reg(
            &mut self,
            reg: &rustc_public::ty::Region,
        ) -> std::ops::ControlFlow<()> {
            match reg.kind {
                rustc_public::ty::RegionKind::ReErased
                | rustc_public::ty::RegionKind::ReStatic
                | rustc_public::ty::RegionKind::ReBound(..) => {
                    std::ops::ControlFlow::Continue(())
                }
                _ => {
                    self.0 = true;
                    std::ops::ControlFlow::Break(())
                }
            }
        }
    }
    let mut screen = MentionsParam(false);
    for arg in &args.0 {
        match arg {
            GenericArgKind::Type(t) => {
                let _ = t.visit(&mut screen);
            }
            GenericArgKind::Const(c) => {
                let _ = c.visit(&mut screen);
            }
            GenericArgKind::Lifetime(r) => {
                let _ = r.visit(&mut screen);
            }
        }
        if screen.0 {
            return true;
        }
    }
    false
}

/// Whether a type mentions an array length the converter
/// cannot resolve by substitution: a const generic parameter
/// (or an unevaluated const) stands where a concrete length
/// belongs (see `Converter::record_instance_shape`).
fn ty_mentions_param_len(ty: &sm::Ty) -> bool {
    match ty {
        sm::Ty::Array(elem, len) => {
            !matches!(len, sm::ArrayLen::Known(_)) || ty_mentions_param_len(elem)
        }
        sm::Ty::Adt(_, args)
        | sm::Ty::FnDef(_, args)
        | sm::Ty::Closure(_, args)
        | sm::Ty::Coroutine(_, args)
        | sm::Ty::CoroutineWitness(_, args)
        | sm::Ty::Alias(_, _, args) => args.iter().any(ty_mentions_param_len),
        sm::Ty::Slice(elem)
        | sm::Ty::RawPtr(elem, _)
        | sm::Ty::Ref(_, elem, _)
        | sm::Ty::Pat(elem, _) => ty_mentions_param_len(elem),
        sm::Ty::FnPtr(sig) => {
            sig.inputs.iter().any(ty_mentions_param_len)
                || ty_mentions_param_len(&sig.output)
        }
        sm::Ty::Dynamic(_, self_ty, _) => ty_mentions_param_len(self_ty),
        sm::Ty::Tuple(ts) => ts.iter().any(ty_mentions_param_len),
        _ => false,
    }
}

fn ty_mentions_alias(ty: &sm::Ty) -> bool {
    match ty {
        sm::Ty::Alias(..) => true,
        sm::Ty::Adt(_, args)
        | sm::Ty::FnDef(_, args)
        | sm::Ty::Closure(_, args)
        | sm::Ty::Coroutine(_, args)
        | sm::Ty::CoroutineWitness(_, args) => args.iter().any(ty_mentions_alias),
        sm::Ty::Array(elem, _)
        | sm::Ty::Slice(elem)
        | sm::Ty::RawPtr(elem, _)
        | sm::Ty::Ref(_, elem, _)
        | sm::Ty::Pat(elem, _) => ty_mentions_alias(elem),
        sm::Ty::FnPtr(sig) => {
            sig.inputs.iter().any(ty_mentions_alias) || ty_mentions_alias(&sig.output)
        }
        sm::Ty::Dynamic(_, self_ty, _) => ty_mentions_alias(self_ty),
        sm::Ty::Tuple(ts) => ts.iter().any(ty_mentions_alias),
        _ => false,
    }
}

/// Whether an ADT instance may be unsized — or otherwise unsafe
/// to ask the compiler for its layout. The check walks the
/// *declared* field types (`FieldDef::ty`, which neither
/// substitutes nor normalizes; the substituting variant
/// `ty_with_args` normalizes projections and fails fatally on
/// ones that do not normalize in a codegen environment). A field
/// may be unsized when its type is a slice, `str`, a trait object,
/// an alias (whose sizedness is not decidable here), a parameter
/// bound to such a type by the instance's arguments, or a nested
/// ADT that may be unsized. Such types are only legal with the
/// unsized field in tail position — except under `trivial_bounds`,
/// where they can sit anywhere and the compiler's layout
/// computation itself ICEs on them (rustc never computes their
/// layouts). Screened instances get no layout entry and stay
/// explicitly unsupported. Parameter resolution goes one frame
/// deep at each ADT boundary, composing the instance arguments
/// with the ambient frame's (see `compose_args`): a parameter
/// the ambient frame leaves unresolved counts as suspect, which
/// only ever skips a query, never enables one.
/// Compose an ADT's instance arguments with the ambient
/// frame's arguments: a parameter of the ambient frame,
/// wherever it appears inside an argument -- directly, or
/// nested inside another ADT, reference, tuple, or the
/// like -- is replaced by the ambient frame's argument for
/// it. The substitution is structural and rebuilds only
/// the shapes `Ty::from_rigid_kind` constructs; a shape it
/// cannot rebuild (an alias type, a constant whose
/// substituted form has no interned identity of its own to
/// take) keeps its argument unsubstituted, exactly as a
/// non-parameter argument was always treated. The walkers
/// below resolve parameters one frame deep; composing at
/// every ADT boundary keeps a parameter chain resolvable
/// across nested generic ADTs (the `ManuallyDrop<T>` field
/// inside `MaybeUninit<T>` is the canonical case) without
/// ever enabling a query the one-frame rule would skip: an
/// argument the ambient frame itself leaves a parameter
/// stays a parameter, and stays suspect.
fn compose_args(
    args: &rustc_public::ty::GenericArgs,
    ambient: Option<&rustc_public::ty::GenericArgs>,
) -> rustc_public::ty::GenericArgs {
    let Some(ambient) = ambient else {
        return args.clone();
    };
    substitute_args(args, ambient, 0, false)
}

/// Compose arguments for the projection screen's field
/// walk (see `compose_args`), additionally resolving a
/// projection argument through the screen's own evidence
/// (see `projection_resolved_ty`) before it crosses the
/// ADT boundary. A projection argument's parameters belong
/// to the enclosing frame: carried into the ADT's frame
/// unresolved, they are read against that frame instead --
/// the alias's own self parameter resolving to the very
/// argument that carries it, until the depth cap gives
/// up. Resolving it here, with the enclosing frame still
/// in hand, substitutes the type the evidence assigns it;
/// a projection the evidence does not determine is left
/// as it was, exactly as `compose_args` leaves it.
fn compose_screen_args(
    args: &rustc_public::ty::GenericArgs,
    ambient: Option<&rustc_public::ty::GenericArgs>,
) -> rustc_public::ty::GenericArgs {
    let Some(ambient) = ambient else {
        return args.clone();
    };
    substitute_args(args, ambient, 0, true)
}

/// `args` with the ambient frame's parameters substituted
/// throughout, recursively (see `compose_args`).
fn substitute_args(
    args: &GenericArgs,
    ambient: &GenericArgs,
    depth: usize,
    resolve_aliases: bool,
) -> GenericArgs {
    GenericArgs(
        args.0
            .iter()
            .map(|a| match a {
                GenericArgKind::Type(t) => {
                    GenericArgKind::Type(substitute_ty(t, ambient, depth, resolve_aliases))
                }
                GenericArgKind::Const(c) => {
                    match substitute_const(c, ambient, depth, resolve_aliases) {
                        Some(c2) => GenericArgKind::Const(c2),
                        None => a.clone(),
                    }
                }
                _ => a.clone(),
            })
            .collect(),
    )
}

/// A type with the ambient frame's parameters substituted
/// throughout, recursively (see `compose_args`). A shape
/// that cannot be rebuilt -- an alias the resolution
/// leaves undetermined (see `compose_screen_args`), a
/// type whose kind cannot be asked, nesting past the
/// depth cap -- is returned unsubstituted.
fn substitute_ty(ty: &Ty, ambient: &GenericArgs, depth: usize, resolve_aliases: bool) -> Ty {
    if depth >= 8 {
        return *ty;
    }
    let Some(kind) = guarded_ty_kind(ty) else {
        return *ty;
    };
    match kind {
        TyKind::Param(p) => match ambient.0.get(p.index as usize) {
            Some(GenericArgKind::Type(t)) => *t,
            _ => *ty,
        },
        TyKind::RigidTy(r) => match r {
            RigidTy::Adt(def, a2) => Ty::from_rigid_kind(RigidTy::Adt(
                def,
                substitute_args(&a2, ambient, depth + 1, resolve_aliases),
            )),
            RigidTy::Ref(region, inner, mutability) => Ty::from_rigid_kind(RigidTy::Ref(
                region,
                substitute_ty(&inner, ambient, depth + 1, resolve_aliases),
                mutability,
            )),
            RigidTy::RawPtr(inner, mutability) => Ty::from_rigid_kind(RigidTy::RawPtr(
                substitute_ty(&inner, ambient, depth + 1, resolve_aliases),
                mutability,
            )),
            RigidTy::Slice(inner) => Ty::from_rigid_kind(RigidTy::Slice(substitute_ty(
                &inner,
                ambient,
                depth + 1,
                resolve_aliases,
            ))),
            RigidTy::Pat(inner, pat) => Ty::from_rigid_kind(RigidTy::Pat(
                substitute_ty(&inner, ambient, depth + 1, resolve_aliases),
                pat,
            )),
            RigidTy::Array(elem, len) => {
                match substitute_const(&len, ambient, depth + 1, resolve_aliases) {
                    Some(len) => Ty::from_rigid_kind(RigidTy::Array(
                        substitute_ty(&elem, ambient, depth + 1, resolve_aliases),
                        len,
                    )),
                    None => *ty,
                }
            }
            RigidTy::Tuple(ts) => Ty::from_rigid_kind(RigidTy::Tuple(
                ts.iter()
                    .map(|t| substitute_ty(t, ambient, depth + 1, resolve_aliases))
                    .collect(),
            )),
            RigidTy::FnDef(def, a2) => Ty::from_rigid_kind(RigidTy::FnDef(
                def,
                substitute_args(&a2, ambient, depth + 1, resolve_aliases),
            )),
            RigidTy::FnPtr(sig) => {
                let Binder {
                    value: fsig,
                    bound_vars,
                } = sig;
                Ty::from_rigid_kind(RigidTy::FnPtr(Binder {
                    value: FnSig {
                        inputs_and_output: fsig
                            .inputs_and_output
                            .iter()
                            .map(|t| substitute_ty(t, ambient, depth + 1, resolve_aliases))
                            .collect(),
                        c_variadic: fsig.c_variadic,
                        safety: fsig.safety,
                        abi: fsig.abi,
                    },
                    bound_vars,
                }))
            }
            RigidTy::Closure(def, a2) => Ty::from_rigid_kind(RigidTy::Closure(
                def,
                substitute_args(&a2, ambient, depth + 1, resolve_aliases),
            )),
            RigidTy::Coroutine(def, a2) => Ty::from_rigid_kind(RigidTy::Coroutine(
                def,
                substitute_args(&a2, ambient, depth + 1, resolve_aliases),
            )),
            RigidTy::CoroutineClosure(def, a2) => Ty::from_rigid_kind(RigidTy::CoroutineClosure(
                def,
                substitute_args(&a2, ambient, depth + 1, resolve_aliases),
            )),
            RigidTy::CoroutineWitness(def, a2) => Ty::from_rigid_kind(RigidTy::CoroutineWitness(
                def,
                substitute_args(&a2, ambient, depth + 1, resolve_aliases),
            )),
            RigidTy::Dynamic(preds, region) => Ty::from_rigid_kind(RigidTy::Dynamic(
                preds
                    .iter()
                    .map(|pred| substitute_pred(pred, ambient, depth + 1, resolve_aliases))
                    .collect(),
                region,
            )),
            _ => *ty,
        },
        TyKind::Alias(SmirAliasKind::Projection, alias) if resolve_aliases => {
            // A projection argument belongs to the
            // enclosing frame (see `compose_screen_args`):
            // substitute the type the screen's own
            // evidence assigns it, completed with the
            // same frame. A projection the evidence does
            // not determine is left unsubstituted.
            match projection_resolved_ty(&alias, Some(ambient), depth + 1) {
                Some(resolved) => substitute_ty(&resolved, ambient, depth + 1, resolve_aliases),
                None => *ty,
            }
        }
        _ => *ty,
    }
}

/// An existential predicate with the ambient frame's
/// parameters substituted throughout (see
/// `substitute_ty`). The binder's own bound variables are
/// the object type's, not the frame's, and are kept.
fn substitute_pred(
    pred: &Binder<ExistentialPredicate>,
    ambient: &GenericArgs,
    depth: usize,
    resolve_aliases: bool,
) -> Binder<ExistentialPredicate> {
    let value = match &pred.value {
        ExistentialPredicate::Trait(tref) => ExistentialPredicate::Trait(ExistentialTraitRef {
            def_id: tref.def_id,
            generic_args: substitute_args(&tref.generic_args, ambient, depth, resolve_aliases),
        }),
        ExistentialPredicate::Projection(p) => {
            let term = match &p.term {
                TermKind::Type(t) => {
                    TermKind::Type(substitute_ty(t, ambient, depth, resolve_aliases))
                }
                TermKind::Const(c) => match substitute_const(c, ambient, depth, resolve_aliases) {
                    Some(c2) => TermKind::Const(c2),
                    None => TermKind::Const(c.clone()),
                },
            };
            ExistentialPredicate::Projection(ExistentialProjection {
                def_id: p.def_id,
                generic_args: substitute_args(&p.generic_args, ambient, depth, resolve_aliases),
                term,
            })
        }
        ExistentialPredicate::AutoTrait(def) => ExistentialPredicate::AutoTrait(*def),
    };
    Binder {
        value,
        bound_vars: pred.bound_vars.clone(),
    }
}

/// A constant with the ambient frame's parameters
/// substituted (see `compose_args`), or `None` when the
/// substituted form cannot be represented: a substituted
/// constant needs an interned identity of its own, which
/// only a constant the frame itself supplies has. An
/// unevaluated constant whose arguments substitute to
/// themselves is returned unchanged.
fn substitute_const(
    c: &TyConst,
    ambient: &GenericArgs,
    depth: usize,
    resolve_aliases: bool,
) -> Option<TyConst> {
    match c.kind() {
        TyConstKind::Param(p) => match ambient.0.get(p.index as usize) {
            Some(GenericArgKind::Const(bound)) => Some(bound.clone()),
            _ => None,
        },
        TyConstKind::Unevaluated(_, args) => {
            if substitute_args(args, ambient, depth, resolve_aliases) == *args {
                Some(c.clone())
            } else {
                None
            }
        }
        _ => Some(c.clone()),
    }
}

fn adt_may_be_unsized(
    def: &AdtDef,
    args: &rustc_public::ty::GenericArgs,
    depth: usize,
) -> bool {
    if depth >= 8 {
        return true;
    }
    def.variants().iter().any(|variant| {
        variant
            .fields()
            .iter()
            .any(|field| ty_may_be_unsized(&field.ty(), Some(args), depth + 1))
    })
}

/// The sizedness of an ADT instance as far as the driver's
/// own walk can tell, refining `adt_may_be_unsized`'s bool:
/// `Unsized` only when a genuinely unsized type (a slice,
/// `str`, or a trait object) is reachable through the fields;
/// `Unknown` when the walk cannot prove either way — the
/// case of field types that mention a type alias, whose
/// sizedness only the compiler's layout computation knows.
#[derive(Clone, Copy, PartialEq)]
enum Sizedness {
    Sized,
    Unsized,
    Unknown,
}

fn adt_sizedness(
    def: &AdtDef,
    args: &rustc_public::ty::GenericArgs,
    depth: usize,
) -> Sizedness {
    if depth >= 8 {
        return Sizedness::Unknown;
    }
    let mut result = Sizedness::Sized;
    for variant in def.variants() {
        for field in variant.fields() {
            match ty_sizedness(&field.ty(), Some(args), depth + 1) {
                Sizedness::Unsized => return Sizedness::Unsized,
                Sizedness::Unknown => result = Sizedness::Unknown,
                Sizedness::Sized => {}
            }
        }
    }
    result
}

fn ty_sizedness(
    ty: &rustc_public::ty::Ty,
    args: Option<&rustc_public::ty::GenericArgs>,
    depth: usize,
) -> Sizedness {
    if depth >= 8 {
        return Sizedness::Unknown;
    }
    // The composed arguments can resolve a parameter to any
    // concrete type in the program, including one whose kind
    // the bridge cannot hand over (a coroutine-closure type):
    // the question goes through the guarded form, and an
    // unaskable type is `Unknown`, like an alias.
    let Some(kind) = guarded_ty_kind(ty) else {
        return Sizedness::Unknown;
    };
    match kind {
        TyKind::RigidTy(r) => match r {
            RigidTy::Slice(_) | RigidTy::Str | RigidTy::Dynamic(..) => {
                Sizedness::Unsized
            }
            RigidTy::Adt(def, a2) => adt_sizedness(&def, &compose_args(&a2, args), depth),
            RigidTy::Array(elem, _) => ty_sizedness(&elem, args, depth),
            RigidTy::Tuple(ts) => {
                let mut result = Sizedness::Sized;
                for t in ts.iter() {
                    match ty_sizedness(t, args, depth) {
                        Sizedness::Unsized => return Sizedness::Unsized,
                        Sizedness::Unknown => result = Sizedness::Unknown,
                        Sizedness::Sized => {}
                    }
                }
                result
            }
            _ => Sizedness::Sized,
        },
        TyKind::Param(p) => match args.and_then(|a| a.0.get(p.index as usize)) {
            Some(rustc_public::ty::GenericArgKind::Type(t)) => {
                ty_sizedness(t, None, depth + 1)
            }
            _ => Sizedness::Unknown,
        },
        _ => Sizedness::Unknown,
    }
}

fn ty_may_be_unsized(
    ty: &rustc_public::ty::Ty,
    args: Option<&rustc_public::ty::GenericArgs>,
    depth: usize,
) -> bool {
    if depth >= 8 {
        return true;
    }
    // See `ty_sizedness`: the composed arguments can carry a
    // type whose kind cannot be asked for; it counts as
    // possibly unsized, like every shape this walk cannot
    // resolve.
    let Some(kind) = guarded_ty_kind(ty) else {
        return true;
    };
    match kind {
        TyKind::RigidTy(r) => match r {
            RigidTy::Slice(_) | RigidTy::Str | RigidTy::Dynamic(..) => true,
            RigidTy::Adt(def, a2) => {
                adt_may_be_unsized(&def, &compose_args(&a2, args), depth)
            }
            RigidTy::Array(elem, _) => ty_may_be_unsized(&elem, args, depth),
            RigidTy::Tuple(ts) => ts.iter().any(|t| ty_may_be_unsized(t, args, depth)),
            _ => false,
        },
        TyKind::Param(p) => match args.and_then(|a| a.0.get(p.index as usize)) {
            Some(rustc_public::ty::GenericArgKind::Type(t)) => {
                ty_may_be_unsized(t, None, depth + 1)
            }
            _ => true,
        },
        _ => true,
    }
}

/// The recorded reason for a definition the conversion does
/// not pose the compiler's monomorphizing queries for,
/// because its types mention a trait projection that cannot
/// normalize in a fully-monomorphized typing environment
/// (see [`unnormalizable_projection_in_instance`]). The
/// constant-item accounting in the binary recognizes this
/// prefix when it re-accounts a screened constant definition
/// as not demanded, the way it does for a definition that
/// fails only on its own parameters.
pub const UNNORMALIZABLE_PROJECTION_REASON: &str =
    "mentions a projection that does not normalize in a \
     monomorphized typing environment";

thread_local! {
    /// The traits that have at least one implementation in
    /// the running compilation, built once (see
    /// `trait_has_impls`).
    static TRAITS_WITH_IMPLS: std::cell::RefCell<
        Option<HashSet<rustc_public::DefId>>,
    > = const { std::cell::RefCell::new(None) };
}

/// Whether `trait_def` has at least one implementation in
/// the running compilation. Building the table enumerates
/// `all_trait_impls` and instantiates each impl's trait
/// reference, which neither substitutes nor normalizes, so
/// it poses no query the projection screen exists to avoid.
fn trait_has_impls(trait_def: rustc_public::DefId) -> bool {
    TRAITS_WITH_IMPLS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let set = cache.get_or_insert_with(|| {
            let mut set: HashSet<rustc_public::DefId> = HashSet::new();
            for impl_def in rustc_public::all_trait_impls() {
                set.insert(impl_def.trait_impl().value.def_id.def_id());
            }
            set
        });
        set.contains(&trait_def)
    })
}

thread_local! {
    /// Per-trait memo for `trait_is_compiler_implemented`.
    static COMPILER_IMPLEMENTED_TRAITS: std::cell::RefCell<
        Option<HashMap<rustc_public::DefId, bool>>,
    > = const { std::cell::RefCell::new(None) };
}

/// Whether the compiler itself provides `trait_def`'s
/// implementations, so the implementation table is no evidence
/// about the trait at all. The compiler marks such traits
/// `#[rustc_deny_explicit_impl]`: user implementations are
/// forbidden outright, and the compiler supplies the
/// implementations from its own knowledge — every type
/// implements `Pointee` with the metadata its kind carries,
/// every enum implements `DiscriminantKind`, the sized
/// hierarchy and `Unsize` hold by construction. Such a
/// trait's projections normalize for the self types the
/// compiler assigns them to, table or no table, so the screen
/// must not reject them.
///
/// The mark is read two ways, because the attribute is not
/// encoded in dependency metadata: directly, for a trait
/// whose attributes the running compilation itself reads,
/// and by lang item for core's own marked traits, whose lang
/// items are the identity that survives into a dependent
/// compilation. Outside a compilation callback there is no
/// compiler context to consult (see `compiler_tcx`); the
/// trait is reported user-implemented, preserving the
/// screen's plain judgment.
fn trait_is_compiler_implemented(trait_def: rustc_public::DefId) -> bool {
    COMPILER_IMPLEMENTED_TRAITS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let memo = cache.get_or_insert_with(HashMap::new);
        if let Some(&known) = memo.get(&trait_def) {
            return known;
        }
        let implemented = compiler_tcx().is_some_and(|tcx| {
            let internal = rustc_public::rustc_internal::internal(tcx, trait_def);
            let marked = tcx
                .get_attrs(internal, rustc_span::sym::rustc_deny_explicit_impl)
                .into_iter()
                .next()
                .is_some();
            let lang_items = tcx.lang_items();
            let core_compiler_trait = [
                rustc_attr_ir::lang_items::LangItem::PointeeTrait,
                rustc_attr_ir::lang_items::LangItem::DiscriminantKind,
                rustc_attr_ir::lang_items::LangItem::Sized,
                rustc_attr_ir::lang_items::LangItem::MetaSized,
                rustc_attr_ir::lang_items::LangItem::PointeeSized,
                rustc_attr_ir::lang_items::LangItem::Unsize,
                rustc_attr_ir::lang_items::LangItem::Destruct,
                rustc_attr_ir::lang_items::LangItem::Tuple,
                rustc_attr_ir::lang_items::LangItem::BikeshedGuaranteedNoDrop,
            ]
            .iter()
            .any(|item| lang_items.get(*item) == Some(internal));
            marked || core_compiler_trait
        });
        memo.insert(trait_def, implemented);
        implemented
    })
}

/// Whether the projection `alias` cannot normalize in a
/// fully-monomorphized typing environment. The one judgment
/// the implementation table decides: the projection's trait
/// has no implementation at all, in any crate of the
/// compilation. No instantiation of a definition mentioning
/// the projection can satisfy its bounds then, so the
/// projection normalizes for no self type — parameter,
/// concrete, or otherwise — with two exceptions the screen
/// checks separately: a trait the compiler itself implements
/// (see `trait_is_compiler_implemented`), whose
/// implementations never reach the table, and a trait-object
/// self type, whose own existential projection predicate
/// binds the item and so normalizes the projection without
/// any implementation (see `dyn_binding_normalizes`).
///
/// Nothing else is judged. Whether an implementation
/// *covers* a concrete self type is not decidable from the
/// table alone: a `dyn` self type normalizes through its
/// own existential projection predicate, closures and
/// coroutines implement the `Fn` family and `Future`
/// through compiler-synthesized implementations with no
/// `impl` item, and a conditional implementation's bounds
/// are not the table's to check. A screen that guessed at
/// coverage would refuse queries rustc answers fine.
fn projection_cannot_normalize(alias: &rustc_public::ty::AliasTy) -> bool {
    let Some(trait_def) = alias.def_id.def_id().parent() else {
        return false;
    };
    !trait_has_impls(trait_def) && !trait_is_compiler_implemented(trait_def)
}

/// Whether the projection `alias` normalizes through an
/// existential associated-type binding of its self type:
/// the self type, resolved as far as the screen's own
/// evidence reaches (see `resolve_screen_self_ty`), is a
/// trait object whose predicates bind this very associated
/// item (`dyn Trait<Assoc = Term>`). Such a projection
/// normalizes to the binding's term in a fully-monomorphized
/// environment even though the trait has no implementation
/// at all — the object type itself carries the equality —
/// so the absence of implementations is no proof it cannot
/// normalize, and the screen must not reject it. Any other
/// shape — an unresolved parameter, a non-object self type,
/// an object binding a different item of the trait — leaves
/// the implementation-table judgment in force.
fn dyn_binding_normalizes(
    alias: &rustc_public::ty::AliasTy,
    ambient: Option<&GenericArgs>,
    depth: usize,
) -> bool {
    // A projection's generic arguments are its trait
    // reference's, whose first argument is the self type.
    let Some(GenericArgKind::Type(self_ty)) = alias.args.0.first() else {
        return false;
    };
    let Some(resolved) = resolve_screen_self_ty(self_ty, ambient, depth) else {
        return false;
    };
    let Some(TyKind::RigidTy(RigidTy::Dynamic(preds, _))) = guarded_ty_kind(&resolved) else {
        return false;
    };
    preds.iter().any(|pred| {
        matches!(
            &pred.value,
            ExistentialPredicate::Projection(p)
                if p.def_id.def_id() == alias.def_id.def_id()
        )
    })
}

thread_local! {
    /// Whether the screen's type resolution gave up at
    /// its depth cap during the judgment in progress:
    /// set by `resolve_screen_self_ty` when its budget
    /// runs out, and read -- and cleared -- by the one
    /// judgment that consults resolution (see
    /// `projection_screen_ty`), so exhaustion recorded
    /// anywhere else cannot leak into a verdict.
    static RESOLUTION_DEPTH_EXHAUSTED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Note that the resolution in progress exhausted its
/// depth budget (see `RESOLUTION_DEPTH_EXHAUSTED`).
fn note_resolution_depth_exhausted() {
    RESOLUTION_DEPTH_EXHAUSTED.with(|flag| flag.set(true));
}

/// Whether the resolution since the last take exhausted
/// its depth budget, clearing the record (see
/// `RESOLUTION_DEPTH_EXHAUSTED`).
fn take_resolution_depth_exhausted() -> bool {
    RESOLUTION_DEPTH_EXHAUSTED.with(|flag| flag.replace(false))
}

/// Resolve a type as far as the screen's own evidence
/// reaches: parameters through the ambient arguments the
/// way the screen's walk resolves them, and projections
/// through the values the implementation table and
/// trait-object bindings assign them (see
/// `projection_resolved_ty`). Returns the first type that
/// is neither a parameter nor a projection, or `None` when
/// a step has no answer in that evidence. No normalization
/// query is posed — the queries the screen exists to
/// avoid — so a projection whose value the evidence does
/// not determine simply stops the resolution.
fn resolve_screen_self_ty(ty: &Ty, ambient: Option<&GenericArgs>, depth: usize) -> Option<Ty> {
    let mut resolved: Ty = *ty;
    let mut depth = depth;
    loop {
        if depth >= 8 {
            // The budget is a termination guard, not
            // evidence: a resolution that runs out of it
            // has reached no answer, and the absence must
            // not be read as one (see
            // `RESOLUTION_DEPTH_EXHAUSTED`).
            note_resolution_depth_exhausted();
            return None;
        }
        let kind = guarded_ty_kind(&resolved)?;
        match kind {
            TyKind::Param(param) => match ambient.and_then(|a| a.0.get(param.index as usize)) {
                Some(GenericArgKind::Type(t)) => resolved = *t,
                _ => return None,
            },
            TyKind::Alias(SmirAliasKind::Projection, inner) => {
                resolved = projection_resolved_ty(&inner, ambient, depth + 1)?;
            }
            TyKind::Alias(..) => return None,
            _ => return Some(resolved),
        }
        depth += 1;
    }
}

/// The type a projection normalizes to, when the screen's
/// own evidence determines it. Two sources of evidence,
/// both read without posing a normalization query: the
/// projection's self type is itself a trait object binding
/// the item, in which case the binding's term is the
/// value; or an implementation of the projection's trait
/// applies to the resolved self type, in which case the
/// type that implementation assigns the item — read from
/// the implementation's own definition — is the value.
/// Anything else (no applicable implementation, several
/// disagreeing) determines nothing and returns `None`.
fn projection_resolved_ty(
    alias: &rustc_public::ty::AliasTy,
    ambient: Option<&GenericArgs>,
    depth: usize,
) -> Option<Ty> {
    let Some(GenericArgKind::Type(self_ty)) = alias.args.0.first() else {
        return None;
    };
    if let Some(resolved_self) = resolve_screen_self_ty(self_ty, ambient, depth + 1) {
        if let Some(TyKind::RigidTy(RigidTy::Dynamic(preds, _))) = guarded_ty_kind(&resolved_self) {
            for pred in preds.iter() {
                let ExistentialPredicate::Projection(p) = &pred.value else {
                    continue;
                };
                if p.def_id.def_id() == alias.def_id.def_id() {
                    if let TermKind::Type(term) = &p.term {
                        return Some(*term);
                    }
                }
            }
        }
    }
    projection_impl_assigned_ty(alias, ambient, depth)
}

thread_local! {
    /// The running compilation's implementations grouped
    /// by trait, built once (see `impls_of_trait`).
    static IMPLS_BY_TRAIT: std::cell::RefCell<
        Option<HashMap<rustc_public::DefId, Vec<ImplDef>>>,
    > = const { std::cell::RefCell::new(None) };
}

/// The implementations of `trait_def` in the running
/// compilation. Building the table enumerates
/// `all_trait_impls` and reads each impl's trait
/// reference, which neither substitutes nor normalizes,
/// so it poses no query the projection screen exists to
/// avoid (the same table `trait_has_impls` builds).
fn impls_of_trait(trait_def: rustc_public::DefId) -> Vec<ImplDef> {
    IMPLS_BY_TRAIT.with(|cache| {
        let mut cache = cache.borrow_mut();
        let map = cache.get_or_insert_with(|| {
            let mut map: HashMap<rustc_public::DefId, Vec<ImplDef>> = HashMap::new();
            for impl_def in rustc_public::all_trait_impls() {
                map.entry(impl_def.trait_impl().value.def_id.def_id())
                    .or_default()
                    .push(impl_def);
            }
            map
        });
        map.get(&trait_def).cloned().unwrap_or_default()
    })
}

/// The type the applicable implementation assigns a
/// projection's item, read from the implementation table:
/// the implementations of the projection's trait whose
/// whole trait reference matches the projection's — the
/// self type against the projection's resolved self type,
/// the remaining arguments against the projection's own
/// (see `match_impl_self_ty`) — must agree on one assigned
/// type, read from the matching implementation's own
/// definition through the compiler context. The query
/// posed is the definition query for the item's type,
/// which neither substitutes into nor normalizes the
/// screened definitions — the same class of question
/// `trait_has_impls`' table building already poses.
///
/// Agreement among the matching implementations decides
/// the value, with one refinement for disagreement (see
/// `decide_impl_assigned_ty`): a conditional
/// implementation — one declaring predicates the
/// structural match does not check — matches self types
/// it may not apply to, so its disagreement with an
/// unconditional implementation proves nothing.
fn projection_impl_assigned_ty(
    alias: &rustc_public::ty::AliasTy,
    ambient: Option<&GenericArgs>,
    depth: usize,
) -> Option<Ty> {
    let trait_def = alias.def_id.def_id().parent()?;
    let Some(GenericArgKind::Type(self_ty)) = alias.args.0.first() else {
        return None;
    };
    let resolved_self = resolve_screen_self_ty(self_ty, ambient, depth + 1)?;
    let mut candidates: Vec<(Ty, ImplDef)> = Vec::new();
    for impl_def in impls_of_trait(trait_def) {
        let tref = impl_def.trait_impl().value;
        // A projection carrying fewer arguments than the
        // trait reference cannot be this implementation's.
        // Arguments beyond the trait reference's are the
        // associated item's own -- a generic associated
        // type's -- and instantiate the item alongside the
        // implementation's arguments when its assigned type
        // is read, below.
        if alias.args.0.len() < tref.args().0.len() {
            continue;
        }
        let mut bindings: HashMap<u32, GenericArgKind> = HashMap::new();
        if !match_impl_self_ty(
            &tref.self_ty(),
            &resolved_self,
            ambient,
            &mut bindings,
            depth + 1,
        ) {
            continue;
        }
        // Matching the self type alone does not identify
        // the implementation: a trait can be implemented
        // for the same self type at several of its own
        // argument instantiations (`Carrier<u8> for ()`
        // beside `Carrier<bool> for ()`), each assigning
        // the item a different type. The trait reference's
        // remaining arguments must match the projection's
        // too, binding implementation parameters that
        // appear only in those positions.
        if !tref
            .args()
            .0
            .iter()
            .skip(1)
            .zip(alias.args.0.iter().skip(1))
            .all(|(i, t)| match_impl_arg(i, t, ambient, &mut bindings, depth + 1))
        {
            continue;
        }
        // An implementation matches the trait reference,
        // so it is the applicable one; if its assigned
        // type cannot be read, no other implementation's
        // can stand in for it.
        let mut args = impl_instantiation_args(&impl_def, &bindings)?;
        args.0
            .extend_from_slice(&alias.args.0[tref.args().0.len()..]);
        let ty = match impl_item_assigned_ty(&impl_def, alias, &args) {
            Some(ty) => ty,
            None => {
                // The implementation may inherit the item
                // from the trait's own default
                // (`associated_type_defaults`), declaring
                // no item of its own: the assigned type is
                // then the default, instantiated with the
                // trait reference this match established --
                // the resolved self type and the
                // projection's own remaining arguments,
                // composed with the ambient frame (the
                // item's own arguments travel with them,
                // past the trait reference's).
                let mut trait_args = vec![GenericArgKind::Type(resolved_self)];
                trait_args.extend_from_slice(
                    &compose_args(&GenericArgs(alias.args.0[1..].to_vec()), ambient).0,
                );
                trait_item_default_ty(alias, &GenericArgs(trait_args))?
            }
        };
        candidates.push((ty, impl_def));
    }
    decide_impl_assigned_ty(candidates)
}

/// The assigned type the matching implementations decide
/// on (see `projection_impl_assigned_ty`): the one type
/// they all assign, when they agree. On disagreement, the
/// unconditional implementations — ones declaring no
/// predicates — decide among themselves: a conditional
/// implementation matches structurally without its
/// predicates being checked, so it may not apply to this
/// self type at all, and in a program that compiles it
/// cannot apply alongside an unconditional one (coherence
/// forbids overlapping implementations), which makes the
/// unconditional agreement the applicable value and the
/// conditional disagreement no evidence against it. When
/// the unconditional implementations disagree among
/// themselves, or none is unconditional, the candidates
/// determine nothing.
fn decide_impl_assigned_ty(candidates: Vec<(Ty, ImplDef)>) -> Option<Ty> {
    if let Some((first, _)) = candidates.first() {
        if candidates.iter().all(|(ty, _)| ty == first) {
            return Some(*first);
        }
    }
    let mut unconditional: Option<Ty> = None;
    for (ty, impl_def) in &candidates {
        if impl_is_conditional(impl_def) {
            continue;
        }
        match unconditional {
            None => unconditional = Some(*ty),
            Some(prev) if prev == *ty => {}
            Some(_) => return None,
        }
    }
    unconditional
}

/// Whether an implementation's applicability is
/// conditional: it declares predicates — parameter bounds
/// or where-clauses — that structural matching does not
/// check, so it may not apply to a self type it matches.
/// The query posed is the definition query for the
/// implementation's explicit clauses (its predicates),
/// which neither substitutes into nor normalizes the
/// screened definitions — the same class of question
/// `impl_item_assigned_ty` poses.
fn impl_is_conditional(impl_def: &ImplDef) -> bool {
    screen_question(|| {
        let tcx = compiler_tcx()?;
        let internal = rustc_public::rustc_internal::internal(tcx, impl_def.def_id());
        Some(!tcx.explicit_clauses_of(internal).clauses.is_empty())
    })
    .unwrap_or(false)
}

/// The generic arguments instantiating an implementation
/// from the arguments its self-type match bound (see
/// `match_impl_self_ty`). A parameter the match never
/// bound has no value the table determines — except a
/// lifetime, which the assigned type's identity does not
/// turn on for the screen's judgment and which is read as
/// erased — so the instantiation fails rather than
/// guessing.
fn impl_instantiation_args(
    impl_def: &ImplDef,
    bindings: &HashMap<u32, GenericArgKind>,
) -> Option<GenericArgs> {
    let generics = impl_def.generics_of();
    let mut args = Vec::with_capacity(generics.params.len());
    for param in &generics.params {
        match bindings.get(&param.index) {
            Some(arg) => args.push(arg.clone()),
            None => match param.kind {
                rustc_public::ty::GenericParamDefKind::Lifetime => {
                    args.push(GenericArgKind::Lifetime(Region {
                        kind: RegionKind::ReErased,
                    }));
                }
                _ => return None,
            },
        }
    }
    Some(GenericArgs(args))
}

/// The type an implementation assigns the projection's
/// item, read from the implementation's own associated
/// item definition through the compiler context installed
/// for the conversion. Outside a compilation callback
/// there is no definition to read.
fn impl_item_assigned_ty(
    impl_def: &ImplDef,
    alias: &rustc_public::ty::AliasTy,
    args: &GenericArgs,
) -> Option<Ty> {
    let item = impl_def.associated_items().into_iter().find(|item| {
        matches!(item.kind, AssocKind::Type { .. })
            && matches!(
                &item.container,
                AssocContainer::TraitImpl(owner)
                    if owner.def_id() == alias.def_id.def_id()
            )
    })?;
    screen_question(|| {
        let tcx = compiler_tcx()?;
        let internal_item = rustc_public::rustc_internal::internal(tcx, item.def_id.def_id());
        let internal_args = rustc_public::rustc_internal::internal(tcx, args.clone());
        let internal_ty = tcx
            .type_of(internal_item)
            .instantiate(tcx, internal_args)
            .skip_normalization();
        Some(rustc_public::rustc_internal::stable(internal_ty))
    })
}

/// The type a projection's trait item defaults to, when
/// the trait declares a default for it
/// (`associated_type_defaults`) and an implementation
/// inherits the item instead of declaring it: the
/// default, read from the trait item's own definition
/// through the compiler context and instantiated with
/// `trait_args` -- the trait reference the implementation
/// match established, plus the item's own arguments. The
/// query posed is the definition query for the item's
/// type, the same class of question
/// `impl_item_assigned_ty` poses. An item the trait
/// declares without a default has nothing to inherit, so
/// nothing is read for it.
fn trait_item_default_ty(
    alias: &rustc_public::ty::AliasTy,
    trait_args: &GenericArgs,
) -> Option<Ty> {
    screen_question(|| {
        let tcx = compiler_tcx()?;
        let internal_item =
            rustc_public::rustc_internal::internal(tcx, alias.def_id.def_id());
        if !tcx
            .associated_item(internal_item)
            .defaultness(tcx)
            .is_default()
        {
            return None;
        }
        let internal_args =
            rustc_public::rustc_internal::internal(tcx, trait_args.clone());
        let internal_ty = tcx
            .type_of(internal_item)
            .instantiate(tcx, internal_args)
            .skip_normalization();
        Some(rustc_public::rustc_internal::stable(internal_ty))
    })
}

/// Whether a type mentions a generic parameter -- a type
/// or constant parameter, anywhere inside it. Lifetime
/// parameters do not count: the instantiation an
/// implementation match feeds reads an unbound lifetime as
/// erased (see `impl_instantiation_args`), so a lifetime
/// needs no binding the way a type or a constant does.
/// The walk poses no query beyond the kind questions the
/// guarded form answers.
fn ty_contains_param(ty: &Ty) -> bool {
    match guarded_ty_kind(ty) {
        Some(TyKind::Param(_)) => true,
        Some(TyKind::RigidTy(rigid)) => match rigid {
            RigidTy::Adt(_, args)
            | RigidTy::FnDef(_, args)
            | RigidTy::Closure(_, args)
            | RigidTy::Coroutine(_, args)
            | RigidTy::CoroutineClosure(_, args)
            | RigidTy::CoroutineWitness(_, args) => args.0.iter().any(generic_arg_contains_param),
            RigidTy::Ref(_, inner, _)
            | RigidTy::RawPtr(inner, _)
            | RigidTy::Slice(inner)
            | RigidTy::Pat(inner, _) => ty_contains_param(&inner),
            RigidTy::Array(elem, len) => ty_contains_param(&elem) || const_contains_param(&len),
            RigidTy::Tuple(ts) => ts.iter().any(ty_contains_param),
            RigidTy::FnPtr(sig) => sig
                .skip_binder()
                .inputs_and_output
                .iter()
                .any(ty_contains_param),
            RigidTy::Dynamic(preds, _) => preds.iter().any(|pred| match &pred.value {
                ExistentialPredicate::Trait(tref) => {
                    tref.generic_args.0.iter().any(generic_arg_contains_param)
                }
                ExistentialPredicate::Projection(p) => {
                    p.generic_args.0.iter().any(generic_arg_contains_param)
                        || match &p.term {
                            TermKind::Type(t) => ty_contains_param(t),
                            TermKind::Const(c) => const_contains_param(c),
                        }
                }
                ExistentialPredicate::AutoTrait(_) => false,
            }),
            _ => false,
        },
        Some(TyKind::Alias(_, alias)) => alias.args.0.iter().any(generic_arg_contains_param),
        _ => false,
    }
}

/// Whether a generic argument mentions a generic
/// parameter (see `ty_contains_param`).
fn generic_arg_contains_param(arg: &GenericArgKind) -> bool {
    match arg {
        GenericArgKind::Type(t) => ty_contains_param(t),
        GenericArgKind::Const(c) => const_contains_param(c),
        GenericArgKind::Lifetime(_) => false,
    }
}

/// Whether a constant mentions a generic parameter (see
/// `ty_contains_param`).
fn const_contains_param(c: &TyConst) -> bool {
    match c.kind() {
        TyConstKind::Param(_) => true,
        TyConstKind::Unevaluated(_, args) => args.0.iter().any(generic_arg_contains_param),
        _ => false,
    }
}

/// Whether an implementation with the trait-reference
/// self type `impl_self` applies to the resolved self
/// type `target`, recording the implementation's generic
/// arguments in `bindings`. The match is structural: an
/// implementation parameter binds the corresponding
/// target argument (and must agree with any earlier
/// binding of the same parameter), every other shape
/// must agree exactly, and a target shape the match
/// cannot judge — an unresolved parameter, a projection
/// the screen cannot resolve — does not match. The match
/// poses no query beyond the kind questions the guarded
/// form answers.
fn match_impl_self_ty(
    impl_self: &Ty,
    target: &Ty,
    ambient: Option<&GenericArgs>,
    bindings: &mut HashMap<u32, GenericArgKind>,
    depth: usize,
) -> bool {
    let Some(ikind) = guarded_ty_kind(impl_self) else {
        return false;
    };
    match ikind {
        TyKind::Param(param) => {
            // The target is written in the frame the
            // screen is walking, so the parameters inside
            // it are first resolved through the ambient
            // frame -- the type form of the resolution the
            // constant match performs for its target (see
            // `match_impl_const`) -- recursively (see
            // `substitute_ty`), so a parameter nested
            // inside a compound target is substituted the
            // way a bare parameter target is: walking
            // `Holder<u32>`, the self type `Wrap<(T,)>`
            // binds the implementation's `A` to `(u32,)`,
            // not to the unsubstituted `(T,)` the trait
            // argument is then compared against. A
            // parameter the ambient frame leaves
            // unresolved stays a parameter. Without it, an
            // implementation parameter appearing in both
            // the self type and another trait argument
            // binds the target's unsubstituted form
            // through the self type and is then compared
            // against the substituted one, rejecting an
            // applicable implementation.
            let substituted;
            let target = match ambient {
                Some(ambient) => {
                    substituted = substitute_ty(target, ambient, 0, false);
                    &substituted
                }
                None => target,
            };
            bind_impl_arg(bindings, param.index, GenericArgKind::Type(*target))
        }
        TyKind::RigidTy(irigid) => {
            // An interned-identity agreement settles the
            // match only when the implementation's self
            // type mentions no parameter. The two sides'
            // parameters belong to different items -- the
            // implementation's own and the walked frame's --
            // and identically indexed ones compare equal
            // without binding anything, leaving the
            // instantiation the assigned-type read performs
            // (see `impl_instantiation_args`) without the
            // argument it needs. A parameter-mentioning
            // self type is matched structurally instead, so
            // its parameters bind through the ambient frame
            // the way any other's do.
            let identity_decides = !ty_contains_param(impl_self);
            if identity_decides && impl_self == target {
                return true;
            }
            // A nested target position can still hold a
            // parameter or a projection, which the same
            // resolution settles before comparing.
            let resolved_target = match guarded_ty_kind(target) {
                Some(TyKind::Param(_)) | Some(TyKind::Alias(..)) => {
                    match resolve_screen_self_ty(target, ambient, depth + 1) {
                        Some(t) => t,
                        None => return false,
                    }
                }
                _ => *target,
            };
            if identity_decides && *impl_self == resolved_target {
                return true;
            }
            let Some(TyKind::RigidTy(trigid)) = guarded_ty_kind(&resolved_target) else {
                return false;
            };
            match (irigid, trigid) {
                (RigidTy::Adt(idef, iargs), RigidTy::Adt(tdef, targs)) => {
                    idef.def_id() == tdef.def_id()
                        && match_impl_args(&iargs, &targs, ambient, bindings, depth)
                }
                (RigidTy::Ref(_, ii, im), RigidTy::Ref(_, ti, tm)) => {
                    im == tm && match_impl_self_ty(&ii, &ti, ambient, bindings, depth)
                }
                (RigidTy::RawPtr(ii, im), RigidTy::RawPtr(ti, tm)) => {
                    im == tm && match_impl_self_ty(&ii, &ti, ambient, bindings, depth)
                }
                (RigidTy::Slice(ii), RigidTy::Slice(ti)) => {
                    match_impl_self_ty(&ii, &ti, ambient, bindings, depth)
                }
                (RigidTy::Pat(ii, _), RigidTy::Pat(ti, _)) => {
                    match_impl_self_ty(&ii, &ti, ambient, bindings, depth)
                }
                (RigidTy::Array(ie, ilen), RigidTy::Array(te, tlen)) => {
                    match_impl_self_ty(&ie, &te, ambient, bindings, depth)
                        && match_impl_const(&ilen, &tlen, ambient, bindings, depth)
                }
                (RigidTy::Tuple(its), RigidTy::Tuple(tts)) => {
                    its.len() == tts.len()
                        && its
                            .iter()
                            .zip(tts.iter())
                            .all(|(i, t)| match_impl_self_ty(i, t, ambient, bindings, depth))
                }
                (RigidTy::FnPtr(isig), RigidTy::FnPtr(tsig)) => {
                    // A generic implementation for a
                    // function pointer (`impl<A> Trait
                    // for fn(A)`) matches a concrete
                    // pointer type signature by
                    // signature: the calling convention
                    // must agree exactly, and each input
                    // and the output match the way self
                    // types match, binding the
                    // implementation's parameters. The
                    // signatures' own bound variables are
                    // their late-bound lifetimes, which
                    // the type match does not turn on.
                    let iv = isig.skip_binder();
                    let tv = tsig.skip_binder();
                    iv.c_variadic == tv.c_variadic
                        && iv.safety == tv.safety
                        && iv.abi == tv.abi
                        && iv.inputs_and_output.len() == tv.inputs_and_output.len()
                        && iv
                            .inputs_and_output
                            .iter()
                            .zip(tv.inputs_and_output.iter())
                            .all(|(i, t)| match_impl_self_ty(i, t, ambient, bindings, depth))
                }
                (RigidTy::Dynamic(ipreds, _), RigidTy::Dynamic(tpreds, _)) => {
                    // A generic implementation for a
                    // trait object (`impl<A> Trait for
                    // dyn Other<A>`) matches a concrete
                    // object type predicate by predicate
                    // (see `match_impl_dynamic_preds`):
                    // the object's own bound variables are
                    // what both lists' arguments are
                    // written against, and the region
                    // decides nothing the type match
                    // turns on, the way regions inside
                    // types are ignored throughout.
                    match_impl_dynamic_preds(&ipreds, &tpreds, ambient, bindings, depth)
                }
                _ => false,
            }
        }
        _ => false,
    }
}

/// Whether an implementation's trait-object self type
/// matches the target object type (see
/// `match_impl_self_ty`): every existential predicate the
/// implementation's object spells must have a counterpart
/// of the same trait in the target's predicate list, with
/// arguments matching the way trait arguments match and a
/// projection predicate's term matching the way its kind
/// matches, binding the implementation's parameters. The
/// target may spell further predicates the implementation
/// does not (an object type can carry bounds beyond the
/// ones an implementation's self type names); only the
/// implementation's own predicates constrain the match.
/// A failed counterpart attempt leaves no bindings
/// behind, so a later counterpart is tried against the
/// bindings as they stood.
fn match_impl_dynamic_preds(
    ipreds: &[rustc_public::ty::Binder<rustc_public::ty::ExistentialPredicate>],
    tpreds: &[rustc_public::ty::Binder<rustc_public::ty::ExistentialPredicate>],
    ambient: Option<&GenericArgs>,
    bindings: &mut HashMap<u32, GenericArgKind>,
    depth: usize,
) -> bool {
    ipreds.iter().all(|ipred| {
        tpreds.iter().any(|tpred| {
            let saved = bindings.clone();
            let matched = match (&ipred.value, &tpred.value) {
                (ExistentialPredicate::Trait(itref), ExistentialPredicate::Trait(ttref)) => {
                    itref.def_id.def_id() == ttref.def_id.def_id()
                        && match_impl_args(
                            &itref.generic_args,
                            &ttref.generic_args,
                            ambient,
                            bindings,
                            depth,
                        )
                }
                (ExistentialPredicate::Projection(ip), ExistentialPredicate::Projection(tp)) => {
                    ip.def_id.def_id() == tp.def_id.def_id()
                        && match_impl_args(
                            &ip.generic_args,
                            &tp.generic_args,
                            ambient,
                            bindings,
                            depth,
                        )
                        && match (&ip.term, &tp.term) {
                            (TermKind::Type(it), TermKind::Type(tt)) => {
                                match_impl_self_ty(it, tt, ambient, bindings, depth)
                            }
                            (TermKind::Const(ic), TermKind::Const(tc)) => {
                                match_impl_const(ic, tc, ambient, bindings, depth)
                            }
                            _ => false,
                        }
                }
                (ExistentialPredicate::AutoTrait(idef), ExistentialPredicate::AutoTrait(tdef)) => {
                    idef.def_id() == tdef.def_id()
                }
                _ => false,
            };
            if !matched {
                *bindings = saved;
            }
            matched
        })
    })
}

/// Whether the generic arguments of an implementation's
/// trait reference (or of an ADT the self type mentions)
/// match the target's, binding the implementation's
/// parameters (see `match_impl_self_ty`).
fn match_impl_args(
    iargs: &GenericArgs,
    targs: &GenericArgs,
    ambient: Option<&GenericArgs>,
    bindings: &mut HashMap<u32, GenericArgKind>,
    depth: usize,
) -> bool {
    iargs.0.len() == targs.0.len()
        && iargs
            .0
            .iter()
            .zip(targs.0.iter())
            .all(|(i, t)| match_impl_arg(i, t, ambient, bindings, depth))
}

/// Whether one generic argument of an implementation
/// matches the corresponding target argument (see
/// `match_impl_args`).
fn match_impl_arg(
    iarg: &GenericArgKind,
    targ: &GenericArgKind,
    ambient: Option<&GenericArgs>,
    bindings: &mut HashMap<u32, GenericArgKind>,
    depth: usize,
) -> bool {
    match (iarg, targ) {
        (GenericArgKind::Type(it), GenericArgKind::Type(tt)) => {
            match_impl_self_ty(it, tt, ambient, bindings, depth)
        }
        (GenericArgKind::Const(ic), GenericArgKind::Const(tc)) => {
            match_impl_const(ic, tc, ambient, bindings, depth)
        }
        (GenericArgKind::Lifetime(iregion), GenericArgKind::Lifetime(tregion)) => {
            // The target region is written in the frame
            // the screen is walking, so a lifetime
            // parameter there is first resolved through
            // the ambient frame -- the lifetime form of
            // the resolution the constant match performs
            // for its target (see `match_impl_const`) --
            // and a parameter the ambient frame leaves
            // unresolved stays a parameter.
            let resolved;
            let tregion = match &tregion.kind {
                RegionKind::ReEarlyParam(param) => {
                    match ambient.and_then(|a| a.0.get(param.index as usize)) {
                        Some(GenericArgKind::Lifetime(bound)) => {
                            resolved = bound.clone();
                            &resolved
                        }
                        _ => tregion,
                    }
                }
                _ => tregion,
            };
            match &iregion.kind {
                RegionKind::ReEarlyParam(param) => bind_impl_arg(
                    bindings,
                    param.index,
                    GenericArgKind::Lifetime(tregion.clone()),
                ),
                // A lifetime never decides which type an
                // implementation assigns the item -- the
                // instantiation reads lifetimes as erased
                // (see `impl_instantiation_args`), and
                // the type match ignores the regions
                // inside types the same way -- so a
                // comparison the regions leave
                // inconclusive (an erased region on
                // either side, a target parameter the
                // ambient frame does not resolve) is not
                // a mismatch: the implementation stays a
                // candidate and the assigned-type
                // agreement check decides. Two concrete
                // regions must still agree.
                _ => {
                    iregion == tregion
                        || matches!(iregion.kind, RegionKind::ReErased)
                        || matches!(
                            tregion.kind,
                            RegionKind::ReErased
                                | RegionKind::ReEarlyParam(_)
                                | RegionKind::ReBound(..)
                        )
                }
            }
        }
        _ => false,
    }
}

/// Whether an implementation's constant argument matches
/// the target's: an implementation constant parameter
/// binds the target constant, every other constant must
/// be the same constant. The target constant is written
/// in the frame the screen is walking, so a constant
/// parameter there is first resolved through the ambient
/// frame -- the constant form of the resolution the type
/// match performs for its target (see
/// `match_impl_self_ty`) -- and a parameter the ambient
/// frame leaves unresolved stays a parameter. An
/// unevaluated constant on each side matches another for
/// the same definition whose arguments match in turn, so
/// a parameter nested inside one is resolved (or bound)
/// the same way. Any other comparison an unevaluated
/// constant takes part in is inconclusive rather than a
/// mismatch: telling `{ N + 0 }` from the concrete `3`
/// takes evaluating it, which the screen never does, so
/// the implementation stays a candidate and the
/// assigned-type agreement check decides -- the lifetime
/// match in `match_impl_arg` treats its inconclusive
/// comparisons the same way.
fn match_impl_const(
    iconst: &TyConst,
    tconst: &TyConst,
    ambient: Option<&GenericArgs>,
    bindings: &mut HashMap<u32, GenericArgKind>,
    depth: usize,
) -> bool {
    let resolved;
    let tconst = match tconst.kind() {
        TyConstKind::Param(param) => {
            match ambient.and_then(|a| a.0.get(param.index as usize)) {
                Some(GenericArgKind::Const(bound)) => {
                    resolved = bound.clone();
                    &resolved
                }
                _ => tconst,
            }
        }
        _ => tconst,
    };
    match iconst.kind() {
        TyConstKind::Param(param) => {
            bind_impl_arg(bindings, param.index, GenericArgKind::Const(tconst.clone()))
        }
        TyConstKind::Unevaluated(idef, iargs) => match tconst.kind() {
            TyConstKind::Unevaluated(tdef, targs) if idef.def_id() == tdef.def_id() => {
                match_impl_args(&iargs, &targs, ambient, bindings, depth)
            }
            _ => true,
        },
        kind => match tconst.kind() {
            TyConstKind::Unevaluated(..) => true,
            _ => kind == tconst.kind(),
        },
    }
}

/// Record an implementation parameter's matched argument,
/// requiring agreement with any earlier binding of the
/// same parameter.
fn bind_impl_arg(
    bindings: &mut HashMap<u32, GenericArgKind>,
    index: u32,
    arg: GenericArgKind,
) -> bool {
    match bindings.get(&index) {
        Some(prev) => *prev == arg,
        None => {
            bindings.insert(index, arg);
            true
        }
    }
}
/// Walk a type for a projection that cannot normalize (see
/// [`projection_cannot_normalize`]), returning the recorded
/// reason for the first one found. The walk mirrors
/// `ty_sizedness`'s discipline: declared field types through
/// ADT boundaries, parameters resolved one frame deep
/// through the ambient arguments (see `compose_args`), a
/// depth cap, and no compiler query beyond the kind
/// questions the guarded form answers — the walk itself
/// never poses the normalization it screens for. A type the
/// walk cannot judge (an unaskable kind, an unresolved
/// parameter) counts as no finding.
fn projection_screen_ty(
    ty: &Ty,
    ambient: Option<&GenericArgs>,
    depth: usize,
) -> Option<String> {
    if depth >= 8 {
        return None;
    }
    let kind = guarded_ty_kind(ty)?;
    match kind {
        TyKind::Alias(alias_kind, alias) => {
            if alias_kind == SmirAliasKind::Projection && projection_cannot_normalize(&alias) {
                // The judgment rests on resolving the
                // projection's self type (see
                // `dyn_binding_normalizes`); a resolution
                // that exhausts its depth budget on the
                // way reaches no answer, so exhaustion
                // during this judgment leaves the screen
                // without an opinion rather than rejecting
                // a projection it could not finish
                // resolving. The record is cleared first,
                // so exhaustion from an earlier judgment
                // cannot leak into this one.
                let _ = take_resolution_depth_exhausted();
                if !dyn_binding_normalizes(&alias, ambient, depth)
                    && !take_resolution_depth_exhausted()
                {
                    return Some(format!(
                        "{UNNORMALIZABLE_PROJECTION_REASON}: \
                         {alias:?}'s trait has no implementation"
                    ));
                }
            }
            projection_screen_args(&alias.args, ambient, depth + 1)
        }
        TyKind::RigidTy(r) => match r {
            RigidTy::Adt(def, a2) => {
                let composed = compose_screen_args(&a2, ambient);
                projection_screen_args(&a2, ambient, depth + 1).or_else(|| {
                    def.variants().iter().find_map(|variant| {
                        variant.fields().iter().find_map(|field| {
                            projection_screen_ty(
                                &field.ty(),
                                Some(&composed),
                                depth + 1,
                            )
                        })
                    })
                })
            }
            RigidTy::Ref(_, inner, _)
            | RigidTy::RawPtr(inner, _)
            | RigidTy::Slice(inner)
            | RigidTy::Pat(inner, _) => {
                projection_screen_ty(&inner, ambient, depth + 1)
            }
            RigidTy::Array(elem, len) => {
                projection_screen_ty(&elem, ambient, depth + 1).or_else(|| {
                    projection_screen_const(&len, ambient, depth + 1)
                })
            }
            RigidTy::Tuple(ts) => ts
                .iter()
                .find_map(|t| projection_screen_ty(t, ambient, depth + 1)),
            RigidTy::FnDef(_, a2) => {
                projection_screen_args(&a2, ambient, depth + 1).or_else(|| {
                    ty.kind().fn_sig().and_then(|sig| {
                        sig.skip_binder()
                            .inputs_and_output
                            .iter()
                            .find_map(|t| {
                                projection_screen_ty(t, None, depth + 1)
                            })
                    })
                })
            }
            RigidTy::FnPtr(sig) => sig
                .skip_binder()
                .inputs_and_output
                .iter()
                .find_map(|t| projection_screen_ty(t, ambient, depth + 1)),
            RigidTy::Closure(_, a2)
            | RigidTy::Coroutine(_, a2)
            | RigidTy::CoroutineWitness(_, a2) => {
                projection_screen_args(&a2, ambient, depth + 1)
            }
            RigidTy::Dynamic(preds, _) => preds.iter().find_map(|pred| {
                match &pred.value {
                    ExistentialPredicate::Trait(tref) => {
                        projection_screen_args(
                            &tref.generic_args,
                            ambient,
                            depth + 1,
                        )
                    }
                    ExistentialPredicate::Projection(pred) => {
                        projection_screen_args(
                            &pred.generic_args,
                            ambient,
                            depth + 1,
                        )
                    }
                    ExistentialPredicate::AutoTrait(_) => None,
                }
            }),
            _ => None,
        },
        TyKind::Param(p) => {
            match ambient.and_then(|a| a.0.get(p.index as usize)) {
                Some(GenericArgKind::Type(t)) => {
                    projection_screen_ty(t, None, depth + 1)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Walk generic arguments for a projection that cannot
/// normalize (see `projection_screen_ty`).
fn projection_screen_args(
    args: &GenericArgs,
    ambient: Option<&GenericArgs>,
    depth: usize,
) -> Option<String> {
    if depth >= 8 {
        return None;
    }
    args.0.iter().find_map(|arg| match arg {
        GenericArgKind::Type(t) => projection_screen_ty(t, ambient, depth),
        GenericArgKind::Const(c) => projection_screen_const(c, ambient, depth),
        GenericArgKind::Lifetime(_) => None,
    })
}

/// Walk a constant for a projection that cannot normalize
/// (see `projection_screen_ty`): only an unevaluated
/// constant's arguments can mention one.
fn projection_screen_const(
    c: &rustc_public::ty::TyConst,
    ambient: Option<&GenericArgs>,
    depth: usize,
) -> Option<String> {
    if depth >= 8 {
        return None;
    }
    match c.kind() {
        TyConstKind::Unevaluated(_, args) => {
            projection_screen_args(&args, ambient, depth)
        }
        _ => None,
    }
}

/// Run a screen question under the containment
/// `guarded_ty_kind` applies to kind questions. The screen
/// asks questions some types cannot answer — an async
/// closure instance's type cannot be asked at all, and an
/// instance constant's evaluation can be too generic — and
/// those questions panic with one of rustc_public's known
/// unimplemented conversions. Such a panic means the
/// question has no answer the screen can use, so the screen
/// has no opinion; the panic hook's record of it is
/// consumed, so no outer containment mistakes it for its
/// own. Any other panic is resumed: the screen must never
/// paper over an unrelated compiler failure.
fn screen_question<T>(question: impl FnOnce() -> Option<T>) -> Option<T> {
    clear_captured_panic();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(question)) {
        Ok(answer) => answer,
        Err(payload) => {
            if stable_mir_panic_reason(&*payload).is_some() {
                let _ = take_captured_panic();
                return None;
            }
            // The payload of a panic that crossed the bridge
            // is opaque, so the hook's record of the same
            // panic — taken after the slot was cleared above,
            // hence this very panic's — is the witness that
            // identifies it.
            match take_captured_panic() {
                Some(record) if record.contains("FIXME(async_closures)") => {
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

/// Whether posing the compiler's monomorphizing queries
/// for this instance — materializing its body through
/// `Instance::body`, or resolving one of its associated
/// items during conversion — would ask the compiler to
/// normalize a trait projection that cannot normalize in a
/// fully-monomorphized typing environment, returning the
/// recorded reason if so.
///
/// Both queries normalize under
/// `TypingEnv::fully_monomorphized` (reveal-all, no caller
/// bounds) through the non-trying
/// `normalize_erasing_regions`, whose failure is a delayed
/// bug: the diagnostic is stored and flushed at the end of
/// the run, so the compilation is poisoned even where the
/// panic itself is contained. The query must not be posed.
/// rustc's own monomorphization collector never poses it
/// either: it instantiates only definitions demanded by
/// other instantiated code, and a projection no
/// implementation covers can never be demanded by a
/// compilable program — the definitions that mention one
/// (a function taking the projection as an argument, a
/// constructor whose field it is, a generic constant whose
/// body resolves a call through it) are exactly the
/// definitions that stay uninstantiated. The screen is
/// structural: it walks the instance's type (for a
/// function, its signature with the arguments applied), its
/// generic arguments, and the declared field types of the
/// ADTs those mention, and reports the first projection
/// whose trait has no implementation at all (see
/// `projection_cannot_normalize`).
pub fn unnormalizable_projection_in_instance(
    inst: &Instance,
) -> Option<String> {
    screen_question(|| {
        projection_screen_ty(&inst.ty(), None, 0)
            .or_else(|| projection_screen_args(&inst.args(), None, 0))
    })
}

/// Whether a body mentions a projection that cannot
/// normalize (see [`unnormalizable_projection_in_instance`]),
/// returning the recorded reason for the first one found.
/// The body is a definition body — fetched without
/// normalization — and the danger is inside its conversion:
/// resolving an associated call whose arguments mention
/// such a projection poses the normalization (an anonymous
/// constant inside a generic function, calling an
/// associated function through a projection over the
/// function's own parameter, is the known shape). The walk
/// covers the locals' types and the call terminators'
/// operand types; a call's callee constant carries the
/// call's generic arguments in its function type.
pub fn unnormalizable_projection_in_body(
    body: &mir::Body,
) -> Option<String> {
    screen_question(|| unnormalizable_projection_in_body_walk(body))
}

fn unnormalizable_projection_in_body_walk(
    body: &mir::Body,
) -> Option<String> {
    for local in body.locals() {
        if let Some(reason) = projection_screen_ty(&local.ty, None, 0) {
            return Some(reason);
        }
    }
    for block in &body.blocks {
        if let mir::TerminatorKind::Call { func, args, .. } =
            &block.terminator.kind
        {
            for op in std::iter::once(func).chain(args.iter()) {
                let reason = match op {
                    mir::Operand::Constant(c) => {
                        projection_screen_ty(&c.const_.ty(), None, 0)
                            .or_else(|| match c.const_.kind() {
                                ConstantKind::Ty(tc) => {
                                    projection_screen_const(&tc, None, 0)
                                }
                                ConstantKind::Unevaluated(uneval) => {
                                    projection_screen_args(
                                        &uneval.args,
                                        None,
                                        0,
                                    )
                                }
                                _ => None,
                            })
                    }
                    mir::Operand::RuntimeChecks(_) => None,
                    mir::Operand::Copy(place) | mir::Operand::Move(place) => {
                        place.ty(body.locals()).ok().and_then(|ty| {
                            projection_screen_ty(&ty, None, 0)
                        })
                    }
                };
                if reason.is_some() {
                    return reason;
                }
            }
        }
    }
    None
}

/// The cardinality of a type's value set, for the zero-sized
/// constant synthesis in `Converter::unique_const_value`.
enum UniqueConst {
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
    static TY_MENTIONS_PARAM_CACHE: std::cell::RefCell<
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
fn env_dwarf_name(def_name: &str, label: &str) -> String {
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
fn env_def_name(ty: &rustc_public::ty::Ty) -> Option<String> {
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
fn coroutine_variant_name(variant: usize) -> String {
    match variant {
        0 => "Unresumed".to_string(),
        1 => "Returned".to_string(),
        2 => "Panicked".to_string(),
        n => format!("Suspend{}", n - 3),
    }
}

/// The upvar count of a closure/coroutine instance: the length
/// of the upvars tuple, the last type argument of the instance.
fn env_upvar_count(args: &GenericArgs) -> Option<usize> {
    args.0.iter().rev().find_map(|a| match a {
        GenericArgKind::Type(t) => match guarded_ty_kind(t) {
            Some(TyKind::RigidTy(RigidTy::Tuple(ts))) => Some(ts.len()),
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
fn instance_item_body(
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

impl Converter {
    pub fn new() -> Self {
        TY_MENTIONS_PARAM_CACHE.with(|cache| cache.borrow_mut().clear());
        TRAITS_WITH_IMPLS.with(|cache| *cache.borrow_mut() = None);
        crate::reach::clear_expanded_size_cache();
        Self {
            def_ids: HashMap::new(),
            instance_ids: HashMap::new(),
            def_names: Vec::new(),
            adt_discriminants: sm::AdtDiscriminants::new(),
            adt_layouts: sm::AdtLayouts::new(),
            adt_field_layouts: sm::AdtFieldLayouts::new(),
            tuple_layouts: sm::TupleLayouts::new(),
            adt_tag_layouts: sm::AdtTagLayouts::new(),
            mat_plans: HashMap::new(),
            coroutine_layouts: sm::CoroutineLayouts::new(),
            coroutine_slots_attempted: std::collections::HashSet::new(),
            layout_attempted: std::collections::HashSet::new(),
            layout_attempted_tys: std::collections::HashSet::new(),
            layout_names: std::collections::HashMap::new(),
            adt_shapes: sm::AdtShapes::new(),
            adt_debug_shapes: sm::AdtDebugShapes::new(),
            env_debug_shapes: sm::EnvDebugShapes::new(),
            env_shape_attempted: std::collections::HashSet::new(),
            debug_shape_depth: 0,
            thread_locals: sm::ThreadLocals::new(),
            adt_instance_shapes: sm::AdtInstanceShapes::new(),
            adt_niches: sm::AdtNiches::new(),
            source_files: Vec::new(),
            source_file_ids: HashMap::new(),
            instance_shape_attempted: std::collections::HashSet::new(),
            static_tys: HashMap::new(),
            static_tys_mir: sm::StaticTys::new(),
            track_caller_defs: HashMap::new(),
            fn_abis: HashMap::new(),
            fn_shim_locations: [None, None, None],
            link_symbols: HashMap::new(),
            noting_track_caller: false,
            errors: Vec::new(),
            const_closure_mentions: Vec::new(),
            const_closure_mentions_seen: std::collections::HashSet::new(),
            const_fn_mentions: Vec::new(),
            const_fn_mentions_seen: std::collections::HashSet::new(),
        }
    }

    fn err(&mut self, what: impl Into<String>) {
        self.errors.push(what.into());
    }

    /// Map a Stable MIR `DefId` to a Sparagmos `DefId`. Stable MIR does
    /// not expose rustc's raw `(krate, index)` pair, so ids are assigned
    /// sequentially within one driver run (`krate` is always 0 in v0);
    /// `def_names` keeps the corresponding item name for each index.
    fn def_id(&mut self, id: rustc_public::DefId) -> sm::DefId {
        if let Some(&idx) = self.def_ids.get(&id) {
            return sm::DefId::new(0, idx);
        }
        let idx = self.def_names.len() as u32;
        self.def_ids.insert(id, idx);
        self.def_names.push(id.name());
        sm::DefId::new(0, idx)
    }

    pub fn crate_def_id<D: CrateDef>(&mut self, def: &D) -> sm::DefId {
        self.def_id(def.def_id())
    }

    /// Record the declared type of a static item. Called by the
    /// item scan before any conversion, and by the foreign-static
    /// reference decoder for the statics the scan does not
    /// enumerate: a reference into a static's interior is decoded
    /// against the static's own type (its seed is the full
    /// initializer and the reference carries a projection into
    /// it), and the type query is only safe for real items, so it
    /// is asked only at points that hold one.
    pub fn note_static_ty<D: CrateDef>(&mut self, def: &D, ty: rustc_public::ty::Ty) {
        let id = self.crate_def_id(def);
        let mir_ty = self.ty(&ty);
        self.static_tys_mir.insert(id, mir_ty);
        self.static_tys.insert(id, ty);
    }

    /// Record the definition of the thread-local static `item`
    /// (already assigned the Sparagmos id `id`) in the program's
    /// thread-locals table: its declared type and its initializer,
    /// decoded from the static's initializer allocation the way a
    /// plain static's seed is decoded in `decode_static_ref`. The
    /// declared type comes from the item scan's table when the
    /// static was recorded there; otherwise it is asked of the
    /// item itself, guarded the way the item scan guards it (this
    /// is rvalue conversion, not the constant decoder). A static
    /// whose type or initializer cannot be obtained gets no entry:
    /// the interpreter then reports its `ThreadLocalRef` as an
    /// explicit error instead of inventing storage.
    fn note_thread_local(&mut self, id: sm::DefId, item: &rustc_public::CrateItem) {
        if self.thread_locals.contains_key(&id) {
            return;
        }
        let Some(ty) = self.static_tys.get(&id).cloned().or_else(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| item.ty())).ok()
        }) else {
            return;
        };
        let Ok(def) = mir::mono::StaticDef::try_from(item.clone()) else {
            return;
        };
        if static_is_foreign(&def) {
            return;
        }
        let Ok(alloc) = def.eval_initializer() else {
            return;
        };
        let Some(init) = self.decode_const(&ty, &alloc, 0, 0) else {
            return;
        };
        let ty = self.ty(&ty);
        self.thread_locals.insert(id, sm::ThreadLocalDef { ty, init });
    }

    /// Map a resolved function instance, reached through the
    /// definition `through`, to a Sparagmos `DefId`. The `def_names`
    /// entry is `through`'s name (the spelling call sites and the
    /// builtin registry use), so several instances of one generic
    /// function share a name but never an id.
    pub fn instance_def_id(
        &mut self,
        through: rustc_public::DefId,
        inst: &Instance,
    ) -> sm::DefId {
        let key = (through, *inst);
        if let Some(&idx) = self.instance_ids.get(&key) {
            return sm::DefId::new(0, idx);
        }
        let idx = self.def_names.len() as u32;
        self.instance_ids.insert(key, idx);
        self.def_names.push(through.name());
        let id = sm::DefId::new(0, idx);
        self.note_link_symbol(id, inst);
        self.note_track_caller_instance(id, inst);
        self.note_fn_abi(id, inst);
        id
    }

    /// Record the declared ABI of a registered function
    /// instance (see [`Converter::fn_abis`]). Runs once per
    /// registered instance — every instance resolution in the
    /// driver flows through `instance_def_id`, so every body
    /// the program can enter has its ABI recorded under the
    /// very id the body is registered with.
    fn note_fn_abi(&mut self, id: sm::DefId, inst: &Instance) {
        if let Some(abi) = instance_fn_abi(inst) {
            self.fn_abis.insert(id, abi);
        }
    }

    /// Record the declared ABI of a function definition under
    /// its own (non-instance) id: the registration the
    /// auxiliary-crate item loop uses for bodies no instance
    /// demand has registered (see [`Converter::fn_abis`]).
    pub fn note_fn_abi_for_def(&mut self, id: sm::DefId, def: &FnDef) {
        if let Some(abi) = def_fn_abi(def) {
            self.fn_abis.insert(id, abi);
        }
    }

    /// If `inst` is a `#[track_caller]` instance with a body,
    /// record its definition-location stamp (see
    /// `track_caller_defs`). Runs once per registered instance —
    /// every instance resolution in the driver flows through
    /// `instance_def_id`, so every track-caller body the program
    /// can enter is recorded, whether it is ever called directly
    /// or only reached through a function pointer or a vtable.
    fn note_track_caller_instance(&mut self, id: sm::DefId, inst: &Instance) {
        if self.noting_track_caller || !inst.requires_caller_location() {
            return;
        }
        // A virtual instance has no body of its own to take a
        // definition span from (asking the compiler for one is
        // an ICE); the impl instances a dispatch can land on
        // are registered separately and carry the stamps.
        if matches!(inst.kind, InstanceKind::Virtual { .. }) {
            return;
        }
        self.noting_track_caller = true;
        let stamp = std::panic::catch_unwind(
            std::panic::AssertUnwindSafe(|| {
                with_body_materialization(&inst.name(), || inst.body())
                    .and_then(|body| self.def_location_stamp(&body))
            }),
        )
        .ok()
        .flatten();
        self.noting_track_caller = false;
        if let Some(stamp) = stamp {
            self.track_caller_defs.insert(id, stamp);
        }
    }

    /// Record the link symbol of the instance registered as
    /// `id` (see `link_symbols`). Runs once per registered
    /// instance — every instance resolution in the driver flows
    /// through `instance_def_id`, so every function the program
    /// can name is recorded, extern declarations included. The
    /// symbol is the compiler's own: `#[no_mangle]` and
    /// `#[export_name]` definitions carry their export name,
    /// extern declarations the name they import (a
    /// `#[link_name]` included), and everything else its
    /// mangled name, which no other instance shares. Asking
    /// the compiler can panic for instances it cannot name (a
    /// virtual instance, say); the panic is contained and the
    /// instance simply gets no symbol, so it never binds by
    /// name.
    fn note_link_symbol(&mut self, id: sm::DefId, inst: &Instance) {
        let symbol = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.link_symbol_of(inst)
        }))
        .ok()
        .flatten();
        if let Some(symbol) = symbol {
            self.link_symbols.insert(id, symbol);
        }
    }

    /// Record the link symbol of the function item `def`,
    /// registered under the Sparagmos `DefId` `id` without an
    /// instance resolution: the auxiliary-crate loop in the
    /// driver binary converts `FnDef` items directly, so their
    /// symbols are noted here instead of in `note_link_symbol`.
    /// The definition is resolved to its monomorphic instance
    /// for the naming; a definition that does not resolve (a
    /// generic one, say) gets no symbol.
    pub fn note_link_symbol_for_def(&mut self, id: sm::DefId, def: rustc_public::DefId) {
        if self.link_symbols.contains_key(&id) {
            return;
        }
        let symbol = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let inst = Instance::resolve(FnDef(def), &GenericArgs(Vec::new())).ok()?;
            self.link_symbol_of(&inst)
        }))
        .ok()
        .flatten();
        if let Some(symbol) = symbol {
            self.link_symbols.insert(id, symbol);
        }
    }

    /// The link symbol the compiler assigns to `inst`, when a
    /// compiler context is installed (outside one there is no
    /// compiler to ask, and no symbol is recorded).
    fn link_symbol_of(&self, inst: &Instance) -> Option<String> {
        let tcx = compiler_tcx()?;
        let internal = rustc_public::rustc_internal::internal(tcx, inst);
        Some(tcx.symbol_name(internal).name.to_string())
    }

    /// The `&'static Location` of a body's own definition: the
    /// stamp the compiler's call shims present to a
    /// `#[track_caller]` callee entered without a call-site
    /// stamp. Computed with the compiler's own resolution over
    /// a terminator of the body re-spanned to the body's
    /// definition span (the same fabrication
    /// `caller_location_stamp` performs for call sites, whose
    /// terminator already carries the call span).
    fn def_location_stamp(&mut self, body: &mir::Body) -> Option<sm::ConstValue> {
        let mut term = body.blocks.first()?.terminator.clone();
        term.source_info.span = body.span;
        let stamp = self.call_site_stamp(body, &term)?;
        Some(stamp)
    }

    /// Decode the caller-location stamp the compiler resolves
    /// for the call edge `term` of `body` (see
    /// `caller_location_stamp` for the call-site use).
    fn call_site_stamp(
        &mut self,
        body: &mir::Body,
        term: &mir::Terminator,
    ) -> Option<sm::ConstValue> {
        let mc = body.caller_location(term, None);
        let constant = self.constant(&mir::ConstOperand {
            span: term.source_info.span,
            user_ty: None,
            const_: mc,
        });
        match constant.literal {
            sm::ConstKind::Value(v) => Some(v),
            _ => None,
        }
    }

    /// Record the location stamp of one `Fn`-family vtable
    /// shim: the stamp of the shim body's own call into the
    /// callable body, which is the location a track-caller
    /// closure entered through a `dyn` receiver observes (the
    /// compiler attributes the generated `Fn` impl to the trait
    /// method's definition in core). `kind` is 0 for `call`, 1
    /// for `call_mut`, 2 for `call_once`. The first shim seen
    /// for a kind wins: the location is a property of the trait
    /// method, not of the callable.
    pub fn note_fn_shim_location(&mut self, kind: usize, shim_body: &mir::Body) {
        if self.fn_shim_locations[kind].is_some() || self.noting_track_caller {
            return;
        }
        self.noting_track_caller = true;
        let stamp = shim_body
            .blocks
            .iter()
            .filter_map(|b| match &b.terminator.kind {
                mir::TerminatorKind::Call { .. } => Some(&b.terminator),
                _ => None,
            })
            .next()
            .and_then(|term| self.call_site_stamp(shim_body, term));
        self.noting_track_caller = false;
        if let Some(stamp) = stamp {
            self.fn_shim_locations[kind] = Some(stamp);
        }
    }

    /// The Sparagmos `DefId` for an `FnDef` type as it appears in a
    /// body: the id of the monomorphic instance the call resolves to.
    /// When the instance cannot be resolved (the definition is not a
    /// resolvable function item), falls back to the definition's own
    /// id, which no converted body carries, so the call stays
    /// unresolved downstream.
    fn fn_def_id(&mut self, def: &FnDef, args: &GenericArgs) -> sm::DefId {
        match Instance::resolve(*def, args) {
            Ok(inst) => self.instance_def_id(def.def_id(), &inst),
            Err(_) => self.def_id(def.def_id()),
        }
    }


    /// The Sparagmos `DefId` for a closure type as it appears in a
    /// body: the id of the monomorphic closure instance the type
    /// denotes, resolved through `Instance::resolve_closure` with
    /// the closure's own kind — the closure counterpart of
    /// `fn_def_id` for function items. Two instances of one closure
    /// definition (the same closure demanded from different generic
    /// contexts) are different bodies and must not share an id.
    /// A closure type whose arguments still mention generic
    /// parameters (the closure seen from inside the generic body
    /// that defines it) or whose instance cannot be resolved keeps
    /// the definition's own id, which no converted body carries,
    /// so the reference stays unresolved downstream.
    pub fn closure_def_id(
        &mut self,
        def: &rustc_public::ty::ClosureDef,
        args: &GenericArgs,
    ) -> sm::DefId {
        let raw = def.def_id();
        if !args_mention_params(args) {
            if let Some(kind) = closure_kind(args) {
                if let Ok(inst) = Instance::resolve_closure(*def, args, kind) {
                    return self.instance_def_id(raw, &inst);
                }
            }
        }
        self.def_id(raw)
    }


    /// Convert one Stable MIR body, containing the panics of
    /// rustc_public 1.100.0's known unimplemented conversions
    /// whose payloads identify them (coroutine-closure types,
    /// unsafe-binder projections, instance-constant evaluation):
    /// a body hitting one fails with the corresponding
    /// `*_REASON` instead of the panic escaping. Any other panic
    /// is resumed — this containment must never paper over an
    /// unrelated compiler failure. (The bare "not implemented"
    /// payloads are identified by location one layer up, in the
    /// driver's `convert_body_isolated`.)
    pub fn convert_body_contained(
        &mut self,
        body: &mir::Body,
    ) -> Result<sm::Body, Vec<String>> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.convert_body(body)
        })) {
            Ok(result) => result,
            Err(payload) => {
                if let Some(reason) = stable_mir_panic_reason(&*payload) {
                    Err(vec![reason.to_string()])
                } else {
                    std::panic::resume_unwind(payload);
                }
            }
        }
    }

    /// Convert one Stable MIR body. On any unsupported construct, returns
    /// the list of reasons (variant names) instead of a body.
    pub fn convert_body(&mut self, body: &mir::Body) -> Result<sm::Body, Vec<String>> {
        self.errors.clear();
        let locals: Vec<sm::LocalDecl> = body
            .locals()
            .iter()
            .map(|d| sm::LocalDecl {
                ty: self.ty(&d.ty),
                mutability: mutability(d.mutability),
            })
            .collect();
        let mut source_lines: Vec<sm::BlockSourceLines> =
            Vec::with_capacity(body.blocks.len());
        let mut blocks: Vec<sm::BasicBlockData> = body
            .blocks
            .iter()
            .map(|b| {
                let (block, lines) = self.basic_block(body, b);
                source_lines.push(lines);
                block
            })
            .collect();
        // The line extent of each source scope in this body,
        // from the resolved lines of the statements and
        // terminators stated directly in it (same-file positions
        // only: the extent is a line range in the body's own
        // file). A debug entry's visibility is its scope, so the
        // entry's recorded extent is its scope's extent, which
        // is what the debugger's name resolution needs to tell a
        // live declaration from a shadowed one whose scope has
        // ended (see var_debug_info).
        let body_file = self.source_line(body.span, body.span).file;
        let mut scope_lines: HashMap<u32, (u32, u32)> = HashMap::new();
        let mut note_scope = |scope_lines: &mut HashMap<u32, (u32, u32)>,
                              scope: u32,
                              pos: sm::SourceLine| {
            if pos.is_known() && pos.file == body_file {
                let entry = scope_lines.entry(scope).or_insert((pos.line, pos.line));
                entry.0 = entry.0.min(pos.line);
                entry.1 = entry.1.max(pos.line);
            }
        };
        for bb in &body.blocks {
            for stmt in &bb.statements {
                let pos = self.source_line(stmt.source_info.span, body.span);
                note_scope(&mut scope_lines, stmt.source_info.scope, pos);
            }
            let pos = self.source_line(bb.terminator.source_info.span, body.span);
            note_scope(&mut scope_lines, bb.terminator.source_info.scope, pos);
        }
        let debug_info: Vec<sm::VarDebugInfo> = body
            .var_debug_info
            .iter()
            .map(|info| self.var_debug_info(info, &scope_lines))
            .collect();
        if self.errors.is_empty() {
            Self::initialize_const_bound_locals(body, &mut blocks, &debug_info);
            let mut b = sm::Body::new(locals, body.arg_locals().len(), blocks);
            b.var_debug_info = debug_info;
            b.source_lines = source_lines;
            Ok(b)
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    fn var_debug_info(
        &mut self,
        info: &mir::VarDebugInfo,
        scope_lines: &HashMap<u32, (u32, u32)>,
    ) -> sm::VarDebugInfo {
        let value = match &info.value {
            mir::VarDebugInfoContents::Place(p) => {
                sm::VarDebugInfoContents::Place(self.place(p))
            }
            mir::VarDebugInfoContents::Const(c) => {
                sm::VarDebugInfoContents::Constant(Box::new(self.constant(c)))
            }
        };
        // The entry's recorded (start, end) lines: the extent
        // of the lexical scope the variable is visible in,
        // which the debugger interface uses to pick the right
        // entry when one name is declared in several scopes.
        // The entry's own span is only the declaration point;
        // the extent is the line range of the entry's source
        // scope as stated in the body (computed by the caller),
        // widened to cover the declaration. A scope that states
        // nothing leaves the declaration point standing alone.
        // (This model's spans are otherwise dummy; the
        // debug-info span is the one place a resolved extent is
        // carried, as lines, not byte offsets.)
        let (lo, hi) = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            info.source_info.span.get_lines()
        })) {
            Ok(lines) => {
                let (mut lo, mut hi) = (lines.start_line as u32, lines.end_line as u32);
                if let Some(&(smin, smax)) = scope_lines.get(&info.source_info.scope) {
                    lo = lo.min(smin);
                    hi = hi.max(smax);
                }
                (lo, hi)
            }
            Err(_) => (0, 0),
        };
        sm::VarDebugInfo {
            name: info.name.clone(),
            source_info: sm::SourceInfo {
                span: sm::Span { lo, hi },
                scope: 0,
            },
            value,
        }
    }

    /// Optimized MIR can hold a local whose only definition was
    /// const-propagated away: the binding's value survives only as
    /// a `VarDebugInfoContents::Const` entry, while the local
    /// itself is still borrowed, dropped and otherwise used as a
    /// place (a match binding moved out of a constant scrutinee is
    /// the canonical shape). Such a local has no assignment
    /// anywhere in the body, so an executor meets it uninitialized.
    /// When a constant debug entry and a never-written local are
    /// the unique pair of their type, restore the definition as an
    /// assignment at the head of the entry block: the value is a
    /// constant, so it is the same at every point of the body.
    /// Ambiguous shapes (several never-written locals, or several
    /// constants, of one type) are left as they are: a wrong
    /// pairing would silently invent a value, which is worse than
    /// the explicit uninitialized failure.
    fn initialize_const_bound_locals(
        body: &mir::Body,
        blocks: &mut [sm::BasicBlockData],
        debug_info: &[sm::VarDebugInfo],
    ) {
        fn mark(written: &mut [bool], place: &sm::Place) {
            // Any write rooted at the local disqualifies it, not
            // just a whole-local assignment: a local the body
            // builds or mutates through projections (a union
            // overwritten field by field, say) is defined by the
            // body, whatever its debug info claims.
            written[place.local.0 as usize] = true;
        }
        let mut written = vec![false; body.locals().len()];
        for i in 1..=body.arg_locals().len() {
            written[i] = true;
        }
        for block in blocks.iter() {
            for stmt in &block.statements {
                match &stmt.kind {
                    sm::StatementKind::Assign(pair) => mark(&mut written, &pair.0),
                    sm::StatementKind::SetDiscriminant { place, .. } => {
                        mark(&mut written, place)
                    }
                    _ => {}
                }
            }
            if let Some(term) = &block.terminator {
                match &term.kind {
                    sm::TerminatorKind::Call { destination, .. } => {
                        mark(&mut written, destination)
                    }
                    sm::TerminatorKind::Yield { resume_arg, .. } => {
                        mark(&mut written, resume_arg)
                    }
                    sm::TerminatorKind::Drop {
                        place, replace, ..
                    } if *replace => mark(&mut written, place),
                    sm::TerminatorKind::InlineAsm { operands, .. } => {
                        for op in operands {
                            if matches!(
                                op.kind,
                                sm::AsmOperandKind::Out | sm::AsmOperandKind::InOut
                            ) {
                                if let Some(place) = &op.out_place {
                                    mark(&mut written, place);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        // Constant debug entries with their source types; the
        // converted debug info preserves the source order, so the
        // index names the same entry in both.
        let consts: Vec<(usize, Ty)> = body
            .var_debug_info
            .iter()
            .enumerate()
            .filter_map(|(i, info)| match &info.value {
                mir::VarDebugInfoContents::Const(c) => Some((i, c.const_.ty())),
                _ => None,
            })
            .collect();
        if consts.is_empty() || blocks.is_empty() {
            return;
        }
        let decl_tys: Vec<Ty> = body.locals().iter().map(|d| d.ty).collect();
        let mut init: Vec<sm::Statement> = Vec::new();
        for (local, ty) in decl_tys.iter().enumerate() {
            if written[local] {
                continue;
            }
            // The local pairs only when it is the sole never-written
            // local of its type and the constant is the sole
            // constant debug entry of that type.
            if decl_tys.iter().enumerate().any(|(other, t)| {
                other != local && t == ty && !written[other]
            }) {
                continue;
            }
            let mut matching = consts.iter().filter(|(_, t)| t == ty);
            let Some((info_idx, _)) = matching.next() else {
                continue;
            };
            if matching.next().is_some() {
                continue;
            }
            if let sm::VarDebugInfoContents::Constant(k) = &debug_info[*info_idx].value {
                // A zero-sized literal carries no value shape: the
                // definition it would restore cannot be rebuilt
                // from it (a union constant decodes to one, and
                // assigning the resulting unit to the union's local
                // is a type error, not a restoration).
                if matches!(k.literal, sm::ConstKind::Value(sm::ConstValue::ZeroSized)) {
                    continue;
                }
                init.push(sm::Statement::new(sm::StatementKind::Assign(Box::new((
                    sm::Place::from_local(sm::Local(local as u32)),
                    sm::Rvalue::Use(sm::Operand::Constant(k.clone())),
                )))));
            }
        }
        if init.is_empty() {
            return;
        }
        init.extend(blocks[0].statements.drain(..));
        blocks[0].statements = init;
    }

    fn basic_block(
        &mut self,
        body: &mir::Body,
        bb: &mir::BasicBlock,
    ) -> (sm::BasicBlockData, sm::BlockSourceLines) {
        let statements = bb
            .statements
            .iter()
            .map(|s| sm::Statement::new(self.statement_kind(&s.kind)))
            .collect();
        let lines = sm::BlockSourceLines {
            statements: bb
                .statements
                .iter()
                .map(|s| self.source_line(s.source_info.span, body.span))
                .collect(),
            terminator: self.source_line(bb.terminator.source_info.span, body.span),
        };
        let terminator = self.terminator_kind(body, &bb.terminator);
        let mut data = sm::BasicBlockData::new(sm::Terminator::new(terminator));
        data.statements = statements;
        // Stable MIR (1.100.0) does not expose `is_cleanup`; see module docs.
        data.is_cleanup = false;
        (data, lines)
    }

    /// Resolve one Stable MIR span to the (file, line) the debugger
    /// interface works in, interning the file name into
    /// `source_files`. The span is opaque outside the compiler, so
    /// the resolution must happen here, through the span's own
    /// stable queries; a span the compiler cannot place (a dummy or
    /// foreign-macro span) yields [`sm::SourceLine::UNKNOWN`]
    /// instead of failing the conversion — line information is an
    /// overlay on the body, never a conversion requirement.
    fn source_line(
        &mut self,
        span: rustc_public::ty::Span,
        body_span: rustc_public::ty::Span,
    ) -> sm::SourceLine {
        let resolved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.collapsed_filename_lines(span, body_span)
                .unwrap_or_else(|| (span.get_filename(), span.get_lines()))
        }));
        let Ok((file, lines)) = resolved else {
            return sm::SourceLine::UNKNOWN;
        };
        if lines.start_line == 0 {
            return sm::SourceLine::UNKNOWN;
        }
        let file_id = match self.source_file_ids.get(&file) {
            Some(&id) => id,
            None => {
                let id = self.source_files.len() as u32;
                self.source_files.push(file.clone());
                self.source_file_ids.insert(file, id);
                id
            }
        };
        sm::SourceLine {
            file: file_id,
            line: lines.start_line as u32,
        }
    }

    /// The (filename, lines) of a statement span as the
    /// compiler's own debuginfo reports them: the span is first
    /// walked outward through its macro expansion chain to the
    /// outermost call site whose expansion collapses debuginfo
    /// (`rustc_span::hygiene::walk_chain_collapsed` -- the exact
    /// resolution codegen applies to every statement's debug
    /// location in `adjusted_span_and_dbg_scope`), bounded by
    /// the function body's span. A statement written inside a
    /// collapsing macro (a `#[collapse_debuginfo]` macro, an
    /// external macro under the default flag, a builtin macro)
    /// is therefore attributed to the macro's invocation line,
    /// which is the line a debugger stops on. Without a compiler
    /// context (a conversion driven outside the callback) the
    /// span stands unresolved and the caller uses its raw
    /// position.
    fn collapsed_filename_lines(
        &self,
        span: rustc_public::ty::Span,
        body_span: rustc_public::ty::Span,
    ) -> Option<(rustc_public::Filename, rustc_public::ty::LineInfo)> {
        let tcx = compiler_tcx()?;
        let internal_span = rustc_public::rustc_internal::internal(tcx, span);
        let internal_bound = rustc_public::rustc_internal::internal(tcx, body_span);
        let collapsed = rustc_span::hygiene::walk_chain_collapsed(internal_span, internal_bound);
        let public: rustc_public::ty::Span = rustc_public::rustc_internal::stable(collapsed);
        Some((public.get_filename(), public.get_lines()))
    }

    fn statement_kind(&mut self, kind: &mir::StatementKind) -> sm::StatementKind {
        use mir::StatementKind as SK;
        match kind {
            SK::Assign(place, rvalue) => {
                let p = self.place(place);
                let r = self.rvalue(rvalue);
                sm::StatementKind::Assign(Box::new((p, r)))
            }
            SK::FakeRead(cause, place) => {
                let p = self.place(place);
                sm::StatementKind::FakeRead(Box::new((fake_read_cause(cause), p)))
            }
            SK::SetDiscriminant {
                place,
                variant_index,
            } => {
                let p = self.place(place);
                sm::StatementKind::SetDiscriminant {
                    place: Box::new(p),
                    variant_index: variant_index.to_index() as u32,
                }
            }
            SK::StorageLive(local) => sm::StatementKind::StorageLive(sm::Local(*local as u32)),
            SK::StorageDead(local) => sm::StatementKind::StorageDead(sm::Local(*local as u32)),
            SK::PlaceMention(place) => {
                let p = self.place(place);
                sm::StatementKind::PlaceMention(Box::new(p))
            }
            SK::AscribeUserType {
                place,
                projections,
                variance,
            } => {
                let p = self.place(place);
                let proj = sm::UserTypeProjection {
                    user_ty_index: projections.base as u32,
                    projections: vec![format!("{:?}", projections.projection)],
                };
                sm::StatementKind::AscribeUserType(Box::new((p, proj)), variance_of(*variance))
            }
            SK::Coverage(_) => sm::StatementKind::Coverage,
            SK::Intrinsic(intrinsic) => {
                sm::StatementKind::Intrinsic(Box::new(self.intrinsic(intrinsic)))
            }
            SK::ConstEvalCounter => sm::StatementKind::ConstEvalCounter,
            SK::Nop => sm::StatementKind::Nop,
        }
    }

    fn intrinsic(&mut self, intrinsic: &mir::NonDivergingIntrinsic) -> sm::NonDivergingIntrinsic {
        use mir::NonDivergingIntrinsic as I;
        match intrinsic {
            I::CopyNonOverlapping(cno) => sm::NonDivergingIntrinsic::CopyNonOverlapping {
                src: self.operand(&cno.src),
                dst: self.operand(&cno.dst),
                count: self.operand(&cno.count),
            },
            I::Assume(op) => sm::NonDivergingIntrinsic::Assume(self.operand(op)),
        }
    }

    /// The implicit caller-location argument for a call to a
    /// `#[track_caller]` callee, if this call is one: the
    /// constant `&'static Location` of this call site, computed
    /// by the compiler's own resolution (which accounts for MIR
    /// inlining). The interpreter binds it to the callee's
    /// synthetic location local; when the calling function is
    /// itself `#[track_caller]`, the interpreter forwards the
    /// caller's received location instead (see
    /// `Interpreter::start_call`).
    fn caller_location_stamp(
        &mut self,
        body: &mir::Body,
        term: &mir::Terminator,
        func: &mir::Operand,
    ) -> Option<sm::Operand> {
        let mir::Operand::Constant(c) = func else {
            return None;
        };
        let TyKind::RigidTy(RigidTy::FnDef(def, args)) = c.const_.ty().kind() else {
            return None;
        };
        let canonical = sparagmos_interp::builtins::canonical_name(&def.def_id().name());
        let builtin = sparagmos_interp::builtins::lookup_qualified(&canonical);
        // The location intrinsic is not itself `#[track_caller]`
        // (it produces the location rather than receiving one),
        // so the gate below would never stamp its calls. That is
        // fine while `Location::caller` stays a separate body —
        // the stamp reaches the intrinsic through the callee's
        // synthetic local — but once the compiler inlines
        // `Location::caller` into a body that received no stamp,
        // the intrinsic call carries no argument at all and the
        // call site's own location is the only correct stamp.
        if !matches!(
            builtin,
            Some(sparagmos_interp::builtins::Builtin::CallerLocation)
        ) {
            let inst = Instance::resolve(def, &args).ok()?;
            if !inst.requires_caller_location() {
                return None;
            }
            // A builtin-served callee has no body to bind the
            // stamp to; only the builtins that consume the
            // location (the intrinsic, the panic entry points)
            // are stamped, so the other builtins keep their
            // declared argument lists. A call the compiler
            // resolved to a DIFFERENT definition than the one it
            // names — a trait method call, which names the trait
            // but dispatches to the impl — is the exception: the
            // interpreter runs the resolved impl's body when one
            // was collected (registered under the trait name),
            // and that body binds the stamp like any direct
            // callee.
            if let Some(kind) = builtin {
                if !sparagmos_interp::builtins::consumes_caller_location(kind)
                    && inst.def.def_id() == def.def_id()
                {
                    return None;
                }
            }
        }
        // The location of a call is the span of its function
        // operand (the `fn_span` the compiler recorded for the
        // call): for a free function that is the call
        // expression's own start, and for a method call it is
        // the method name — the terminator's own span covers
        // the whole receiver expression and would name the
        // receiver instead.
        let mut term = term.clone();
        // The stamp is located by the function operand's span
        // (see below), but an optimized body can carry the
        // operand as a promoted constant whose span was erased
        // (it resolves to line 0): the terminator's own span
        // still names the call site then, so it takes over.
        // A span with a real position always wins, because for
        // a method call the terminator span covers the whole
        // receiver expression and would name the receiver.
        let operand_has_position = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.collapsed_filename_lines(c.span, body.span)
                .unwrap_or_else(|| (c.span.get_filename(), c.span.get_lines()))
                .1
                .start_line
                != 0
        }))
        .unwrap_or(false);
        if operand_has_position {
            term.source_info.span = c.span;
        }
        let mc = body.caller_location(&term, None);
        let constant = self.constant(&mir::ConstOperand {
            span: term.source_info.span,
            user_ty: None,
            const_: mc,
        });
        Some(sm::Operand::Constant(Box::new(constant)))
    }

    fn terminator_kind(&mut self, body: &mir::Body, term: &mir::Terminator) -> sm::TerminatorKind {
        use mir::TerminatorKind as TK;
        match &term.kind {
            TK::Goto { target } => sm::TerminatorKind::Goto { target: bb(*target) },
            TK::SwitchInt { discr, targets } => {
                let d = self.operand(discr);
                let (values, tgts, otherwise) = switch_targets(targets);
                sm::TerminatorKind::SwitchInt {
                    discr: d,
                    targets: sm::SwitchTargets::new(values, tgts, otherwise),
                }
            }
            TK::Resume => sm::TerminatorKind::Resume,
            TK::Abort => sm::TerminatorKind::Terminate(sm::TerminateKind::Abort),
            TK::Return => sm::TerminatorKind::Return,
            TK::Unreachable => sm::TerminatorKind::Unreachable,
            TK::Drop {
                place,
                target,
                unwind,
            } => {
                let p = self.place(place);
                sm::TerminatorKind::Drop {
                    place: p,
                    target: bb(*target),
                    unwind: self.unwind(unwind),
                    // Stable MIR does not expose `replace`.
                    replace: false,
                }
            }
            TK::Call {
                func,
                args,
                destination,
                target,
                unwind,
            } => {
                let f = self.operand(func);
                let mut a: Vec<sm::Operand> =
                    args.iter().map(|a| self.operand(a)).collect();
                // A call to a `#[track_caller]` callee carries the
                // implicit caller-location argument, appended here
                // (the callee body models it as a synthetic final
                // local).
                if let Some(stamp) = self.caller_location_stamp(body, term, func) {
                    a.push(stamp);
                }
                let d = self.place(destination);
                sm::TerminatorKind::Call {
                    func: f,
                    args: a,
                    destination: d,
                    target: target.map(bb),
                    unwind: self.unwind(unwind),
                    // Stable MIR does not expose the call source.
                    call_source: sm::CallSource::Normal,
                }
            }
            TK::Assert {
                cond,
                expected,
                msg,
                target,
                unwind,
            } => {
                let c = self.operand(cond);
                let m = self.assert_msg(msg);
                sm::TerminatorKind::Assert {
                    cond: c,
                    expected: *expected,
                    msg: Box::new(m),
                    target: bb(*target),
                    unwind: self.unwind(unwind),
                }
            }
            TK::InlineAsm {
                template,
                operands,
                options,
                destination,
                unwind,
                ..
            } => {
                // Conversion is faithful for every shape: the
                // structured form carries the template, the operands
                // and the options in full. Which blocks *execute* is
                // the executors' classification, applied where the
                // terminator is consumed (see
                // `sm::inline_asm_is_noop` and
                // `sm::inline_asm_program`): the no-op class and the
                // evaluable register-program class run, and anything
                // else is a named explicit error at execution time —
                // never a reason to drop the whole body here.
                let ops: Vec<sm::InlineAsmOperand> = operands
                    .iter()
                    .map(|op| self.asm_operand(body, op))
                    .collect();
                let targets = self.asm_targets(body, *destination, options, &ops);
                sm::TerminatorKind::InlineAsm {
                    template: template.clone(),
                    operands: ops,
                    options: options.clone(),
                    destination: destination.map(bb),
                    targets,
                    unwind: self.unwind(unwind),
                }
            }
        }
    }

    /// One Stable MIR inline-asm operand in the structured form
    /// the executors consume. The public operand struct keeps only
    /// the value/place pair, so the kind and the register are read
    /// from its raw representation (`In { reg: RegClass(X86(reg)),
    /// value: .. }`, `Const { value: const 5_usize }`, ...).
    fn asm_operand(
        &mut self,
        body: &mir::Body,
        op: &mir::InlineAsmOperand,
    ) -> sm::InlineAsmOperand {
        let raw = op.raw_rpr.as_str();
        let kind = if raw.starts_with("InOut {") {
            sm::AsmOperandKind::InOut
        } else if raw.starts_with("In {") {
            sm::AsmOperandKind::In
        } else if raw.starts_with("Out {") {
            sm::AsmOperandKind::Out
        } else if raw.starts_with("Const {") {
            sm::AsmOperandKind::Const
        } else if raw.starts_with("Label {") {
            sm::AsmOperandKind::Label
        } else {
            sm::AsmOperandKind::Sym
        };
        let reg_class = raw.contains("reg: RegClass(");
        let (width, const_bits) = match kind {
            sm::AsmOperandKind::Const => match asm_const_literal(raw) {
                Some((bits, width)) => (width, Some(bits)),
                None => (0, None),
            },
            sm::AsmOperandKind::Sym | sm::AsmOperandKind::Label => (0, None),
            sm::AsmOperandKind::In => (
                op.in_value
                    .as_ref()
                    .and_then(|v| v.ty(body.locals()).ok())
                    .map_or(0, |t| asm_type_width(&t)),
                None,
            ),
            _ => (
                op.out_place
                    .as_ref()
                    .and_then(|p| p.ty(body.locals()).ok())
                    .map_or(0, |t| asm_type_width(&t)),
                None,
            ),
        };
        let static_name = if kind == sm::AsmOperandKind::Sym {
            asm_raw_sym_name(raw)
        } else {
            None
        };
        let call_sig = if raw.starts_with("SymFn {") {
            static_name.as_deref().and_then(|name| self.asm_call_sig(name))
        } else {
            None
        };
        sm::InlineAsmOperand {
            kind,
            reg_class,
            width,
            const_bits,
            in_value: op.in_value.as_ref().map(|v| self.operand(v)),
            out_place: op.out_place.as_ref().map(|p| self.place(p)),
            reg_name: asm_raw_reg_name(raw),
            static_name,
            label_index: if kind == sm::AsmOperandKind::Label {
                asm_raw_label_index(raw)
            } else {
                None
            },
            call_sig,
        }
    }

    /// The successor list of an InlineAsm terminator: the
    /// destination followed by its label blocks, restored
    /// structurally by [`restore_asm_label_blocks`]. A terminator
    /// without label operands — or whose labels could not be
    /// restored — carries just its destination (or nothing).
    fn asm_targets(
        &self,
        body: &mir::Body,
        destination: Option<mir::BasicBlockIdx>,
        options: &str,
        ops: &[sm::InlineAsmOperand],
    ) -> Vec<sm::BasicBlock> {
        let dest = destination.map(bb);
        let label_count = ops
            .iter()
            .filter(|op| op.kind == sm::AsmOperandKind::Label)
            .count();
        if label_count == 0 {
            return dest.into_iter().collect();
        }
        let Some(restored) =
            restore_asm_label_blocks(body, label_count, options.contains("NORETURN"))
        else {
            return dest.into_iter().collect();
        };
        let mut targets: Vec<sm::BasicBlock> = dest.into_iter().collect();
        targets.extend(restored);
        targets
    }

    /// Resolve an asm `sym` operand naming a function to the
    /// register-passing call signature a `call` step executes
    /// through: the function must be the local crate's unique
    /// function the operand's path denotes (see
    /// [`asm_ref_matches_def`] — the complete path, never just
    /// its last segment), non-generic, with a fully scalar
    /// signature under the SysV64 or Win64 ABI families (the
    /// `C` and `System` ABIs are SysV64-shaped on this target),
    /// and register-passed throughout (no stack arguments).
    /// Anything else returns `None` and the call stays outside
    /// the evaluable class.
    fn asm_call_sig(&mut self, name: &str) -> Option<sm::AsmCallSig> {
        let mut candidates = rustc_public::local_crate()
            .fn_defs()
            .into_iter()
            .filter(|def| asm_ref_matches_def(&def.name(), name));
        let def = candidates.next()?;
        if candidates.next().is_some() {
            return None;
        }
        // Resolution can panic inside the compiler for shapes
        // it cannot monomorphize from empty arguments (generic
        // or predicate-carrying functions): such a function is
        // simply not a resolvable call target.
        let sig = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let inst = Instance::resolve(def, &GenericArgs(vec![])).ok()?;
            Some(guarded_ty_kind(&inst.ty())?.fn_sig()?.skip_binder())
        }))
        .ok()??;
        if sig.c_variadic {
            return None;
        }
        let abi = asm_abi_family(&sig.abi)?;
        let (output, inputs) = sig.inputs_and_output.split_last()?;
        let mut params = Vec::with_capacity(inputs.len());
        for input in inputs {
            params.push(asm_scalar(input)?);
        }
        let (mut ints, mut floats) = (0usize, 0usize);
        for param in &params {
            match param {
                sm::AsmScalar::Float { .. } => floats += 1,
                _ => ints += 1,
            }
        }
        let register_passed = match abi {
            sm::AsmAbi::SysV64 => ints <= 6 && floats <= 8,
            sm::AsmAbi::Win64 => params.len() <= 4,
        };
        if !register_passed {
            return None;
        }
        let ret = match guarded_ty_kind(output)? {
            TyKind::RigidTy(RigidTy::Tuple(ts)) if ts.is_empty() => None,
            _ => Some(asm_scalar(output)?),
        };
        Some(sm::AsmCallSig {
            name: def.name(),
            abi,
            params,
            ret,
        })
    }

    fn assert_msg(&mut self, msg: &mir::AssertMessage) -> sm::AssertMessage {
        use mir::AssertMessage as M;
        match msg {
            M::BoundsCheck { len, index } => sm::AssertMessage::BoundsCheck {
                len: Box::new(self.operand(len)),
                index: Box::new(self.operand(index)),
            },
            M::Overflow(op, l, r) => sm::AssertMessage::Overflow(
                binop(*op),
                Box::new(self.operand(l)),
                Box::new(self.operand(r)),
            ),
            M::OverflowNeg(op) => sm::AssertMessage::OverflowNeg(Box::new(self.operand(op))),
            M::DivisionByZero(op) => {
                sm::AssertMessage::DivisionByZero(Box::new(self.operand(op)))
            }
            M::RemainderByZero(op) => {
                sm::AssertMessage::RemainderByZero(Box::new(self.operand(op)))
            }
            M::ResumedAfterReturn(k) => sm::AssertMessage::ResumedAfterReturn(coroutine_kind(k)),
            M::ResumedAfterPanic(k) => sm::AssertMessage::ResumedAfterPanic(coroutine_kind(k)),
            M::ResumedAfterDrop(k) => sm::AssertMessage::ResumedAfterDrop(coroutine_kind(k)),
            M::MisalignedPointerDereference { required, found } => {
                sm::AssertMessage::MisalignedPointerDereference {
                    required: Box::new(self.operand(required)),
                    found: Box::new(self.operand(found)),
                }
            }
            M::NullPointerDereference => sm::AssertMessage::NullPointerDereference,
            M::NullReferenceConstructed => {
                sm::AssertMessage::Custom("NullReferenceConstructed".to_string())
            }
            M::InvalidEnumConstruction(op) => {
                sm::AssertMessage::InvalidEnumConstruction(Box::new(self.operand(op)))
            }
        }
    }

    fn unwind(&mut self, action: &mir::UnwindAction) -> sm::UnwindAction {
        use mir::UnwindAction as U;
        match action {
            U::Continue => sm::UnwindAction::Continue,
            U::Unreachable => sm::UnwindAction::Unreachable,
            U::Terminate => sm::UnwindAction::Terminate(sm::UnwindTerminateReason::Generic),
            U::Cleanup(t) => sm::UnwindAction::Cleanup(bb(*t)),
        }
    }

    fn rvalue(&mut self, rv: &mir::Rvalue) -> sm::Rvalue {
        use mir::Rvalue as R;
        match rv {
            R::Use(op, _) => sm::Rvalue::Use(self.operand(op)),
            R::Repeat(op, count) => {
                let o = self.operand(op);
                match count.eval_target_usize() {
                    Ok(n) => sm::Rvalue::Repeat(o, n),
                    Err(_) => {
                        self.err("Rvalue::Repeat (count not evaluable)");
                        sm::Rvalue::Repeat(o, 0)
                    }
                }
            }
            R::Ref(region, borrow, place) => {
                let p = self.place(place);
                sm::Rvalue::Ref(region_of(region), borrow_kind(borrow), p)
            }
            R::ThreadLocalRef(item) => {
                let id = self.crate_def_id(item);
                self.note_thread_local(id, item);
                sm::Rvalue::ThreadLocalRef(id)
            }
            R::AddressOf(kind, place) => {
                let p = self.place(place);
                sm::Rvalue::RawPtr(mutability(kind.to_mutable_lossy()), p)
            }
            R::Len(place) => {
                let p = self.place(place);
                sm::Rvalue::Len(p)
            }
            R::Cast(kind, op, ty) => {
                let o = self.operand(op);
                let t = self.ty(ty);
                if matches!(kind, mir::CastKind::PtrToPtr | mir::CastKind::Transmute) {
                    self.note_mat_plan(ty, &t);
                }
                sm::Rvalue::Cast(self.cast_kind(kind), o, t)
            }
            R::BinaryOp(op, l, r) => {
                let (l, r) = (self.operand(l), self.operand(r));
                sm::Rvalue::BinaryOp(binop(*op), Box::new((l, r)))
            }
            R::CheckedBinaryOp(op, l, r) => {
                let (l, r) = (self.operand(l), self.operand(r));
                sm::Rvalue::CheckedBinaryOp(binop(*op), Box::new((l, r)))
            }
            R::UnaryOp(op, o) => {
                let o = self.operand(o);
                sm::Rvalue::UnaryOp(unop(*op), o)
            }
            R::Discriminant(place) => {
                let p = self.place(place);
                sm::Rvalue::Discriminant(p)
            }
            R::Aggregate(kind, ops) => {
                let ops = ops.iter().map(|o| self.operand(o)).collect();
                let k = self.aggregate_kind(kind);
                sm::Rvalue::Aggregate(Box::new(k), ops)
            }
            R::CopyForDeref(place) => {
                let p = self.place(place);
                sm::Rvalue::CopyForDeref(p)
            }
            R::Reborrow(ty, m, place) => {
                let t = self.ty(ty);
                let p = self.place(place);
                sm::Rvalue::Reborrow(t, mutability(*m), p)
            }
        }
    }

    fn aggregate_kind(&mut self, kind: &mir::AggregateKind) -> sm::AggregateKind {
        use mir::AggregateKind as A;
        match kind {
            A::Array(ty) => sm::AggregateKind::Array(self.ty(ty)),
            A::Tuple => sm::AggregateKind::Tuple,
            A::Adt(def, variant, args, user_ty, field) => {
                let d = self.crate_def_id(def);
                sm::AggregateKind::Adt {
                    def: d,
                    variant_index: variant.to_index() as u32,
                    generic_args: self.type_args(args),
                    user_ty: user_ty.map(|u| u as u32),
                    field_index: field.map(|f| f as u32),
                }
            }
            A::Closure(def, args) => sm::AggregateKind::Closure {
                def: self.closure_def_id(def, args),
                generic_args: self.type_args(args),
            },
            A::Coroutine(def, args) => sm::AggregateKind::Coroutine {
                def: self.crate_def_id(def),
                generic_args: self.type_args(args),
            },
            A::CoroutineClosure(def, args) => sm::AggregateKind::CoroutineClosure {
                def: self.crate_def_id(def),
                generic_args: self.type_args(args),
            },
            A::RawPtr(ty, m) => sm::AggregateKind::RawPtr {
                data_ty: self.ty(ty),
                mutability: mutability(*m),
            },
        }
    }

    fn operand(&mut self, op: &mir::Operand) -> sm::Operand {
        use mir::Operand as O;
        match op {
            O::Copy(p) => sm::Operand::Copy(self.place(p)),
            O::Move(p) => sm::Operand::Move(self.place(p)),
            O::Constant(c) => sm::Operand::Constant(Box::new(self.constant(c))),
            O::RuntimeChecks(checks) => {
                // A codegen-time cfg flag (`cfg!(ub_checks)` and
                // its siblings), left symbolic in MIR and resolved
                // when the instance is code-generated. Sparagmos
                // compiles the whole program in one session with
                // the debug-profile defaults — debug assertions,
                // overflow checks and contract checks enabled, and
                // no optimisation flag is ever passed — which is
                // also how the native oracle for these programs is
                // built, so the flags resolve to `true` here. The
                // checks they guard (unsafe-precondition assertions
                // in core, contract ensures) then run like they do
                // in a native debug build.
                let _ = checks;
                sm::Operand::constant(sm::Constant::bool(true))
            }
        }
    }

    fn constant(&mut self, c: &mir::ConstOperand) -> sm::Constant {
        let mut k = self.constant_literal(c);
        if k.ty.is_none() {
            k.ty = self.constant_scalar_ty(c);
        }
        if k.ty.is_none() {
            k.ty = self.constant_adt_ty(c, &k);
        }
        k
    }

    /// The constant's own type when the constant is a decoded
    /// ADT value: a `transmute` of such a constant -- the form
    /// a const-propagated `core` body takes (the folded
    /// dangling-pointer construction inside
    /// `Unique::dangling` is a transmute of the folded
    /// `Alignment` constant) -- is decided from the source
    /// type, which the value's structure alone does not name
    /// (a one-field wrapper around a fieldless enum does not
    /// say which enum). Recorded only for ADT-typed constants
    /// whose literal decoded to an ADT value; every other
    /// constant keeps the narrower rules above.
    fn constant_adt_ty(
        &mut self,
        c: &mir::ConstOperand,
        k: &sm::Constant,
    ) -> Option<sm::Ty> {
        if !matches!(
            k.literal,
            sm::ConstKind::Value(sm::ConstValue::Adt { .. })
        ) {
            return None;
        }
        match guarded_ty_kind(&c.const_.ty())? {
            TyKind::RigidTy(RigidTy::Adt(..)) => {
                Some(self.ty(&c.const_.ty()))
            }
            _ => None,
        }
    }

    /// The constant's own type when it is a plain scalar type:
    /// the one reading of a scalar literal's bits that needs no
    /// context from the operation using it (see
    /// `sm::Constant::ty`). Other types are not recorded: their
    /// conversion is not needed for the literal reading, and
    /// asking for it could fail where the constant itself is fine.
    fn constant_scalar_ty(&mut self, c: &mir::ConstOperand) -> Option<sm::Ty> {
        match guarded_ty_kind(&c.const_.ty())? {
            TyKind::RigidTy(
                RigidTy::Int(_)
                | RigidTy::Uint(_)
                | RigidTy::Bool
                | RigidTy::Char
                | RigidTy::Float(_),
            ) => Some(self.ty(&c.const_.ty())),
            _ => None,
        }
    }

    fn constant_literal(&mut self, c: &mir::ConstOperand) -> sm::Constant {
        use ConstantKind as CK;
        let mc = &c.const_;
        match mc.kind() {
            CK::ZeroSized => {
                // Function items and closures are ZST constants whose
                // type names the callable (the FnDef, resp. the closure
                // definition); keep the type so call targets stay
                // identifiable.
                let ty = mc.ty();
                // A type whose kind cannot be asked (it passes
                // through a coroutine-closure type; see
                // `guarded_ty_kind`) is not treated as naming a
                // callable: it may only contain an async closure.
                // It falls through to the zero-sized value below,
                // whose walk is guarded the same way.
                let names_callable = match guarded_ty_kind(&ty) {
                    Some(kind) => matches!(
                        kind,
                        TyKind::RigidTy(RigidTy::FnDef(..) | RigidTy::Closure(..))
                    ),
                    None => false,
                };
                if names_callable {
                    let t = self.ty(&ty);
                    return sm::Constant {
                        literal: sm::ConstKind::Ty(t),
                        user_ty: None,
                        ty: None,
                    };
                }
                // Any other zero-sized constant still has a typed
                // value when its type determines one (a unit struct,
                // a single-variant enum, a tuple of zero-sized
                // types): keep it, so the interpreter sees the ADT
                // value rather than an untyped unit.
                if let Some(v) = self.zst_const_value(&ty, 0) {
                    return sm::Constant::from_value(v);
                }
                sm::Constant::from_value(sm::ConstValue::ZeroSized)
            }
            CK::Allocated(alloc) => {
                let ty = mc.ty();
                let v = self.const_from_alloc(&ty, alloc);
                sm::Constant::from_value(v)
            }
            CK::Ty(tc) => match &tc.kind() {
                TyConstKind::Value(ty, alloc) => {
                    let t = *ty;
                    let v = self.const_from_alloc(&t, alloc);
                    sm::Constant::from_value(v)
                }
                TyConstKind::ZSTValue(ty) => {
                    // A zero-sized type constant of function-item or
                    // closure type names a callable; keep the type,
                    // as for `CK::ZeroSized` above. A type whose
                    // kind cannot be asked (it passes through a
                    // coroutine-closure type; see
                    // `guarded_ty_kind`) is not treated as naming
                    // a callable either: it falls through to the
                    // zero-sized value below.
                    if match guarded_ty_kind(ty) {
                        Some(kind) => matches!(
                            kind,
                            TyKind::RigidTy(RigidTy::FnDef(..) | RigidTy::Closure(..))
                        ),
                        None => false,
                    } {
                        let t = self.ty(ty);
                        return sm::Constant {
                            literal: sm::ConstKind::Ty(t),
                            user_ty: None,
                            ty: None,
                        };
                    }
                    if let Some(v) = self.zst_const_value(ty, 0) {
                        return sm::Constant::from_value(v);
                    }
                    sm::Constant::from_value(sm::ConstValue::ZeroSized)
                }
                TyConstKind::Unevaluated(def, args) => {
                    let d = self.crate_def_id(def);
                    sm::Constant {
                        literal: sm::ConstKind::Unevaluated {
                            def: d,
                            args: self.type_args(args),
                        },
                        user_ty: None,
                        ty: None,
                    }
                }
                TyConstKind::Param(p) => {
                    self.err(format!("ConstantKind::Param({})", p.name));
                    sm::Constant::from_value(sm::ConstValue::ZeroSized)
                }
                TyConstKind::Bound(_, _) => {
                    self.err("ConstantKind::Bound");
                    sm::Constant::from_value(sm::ConstValue::ZeroSized)
                }
            },
            CK::Unevaluated(u) => {
                let d = self.crate_def_id(&u.def);
                sm::Constant {
                    literal: sm::ConstKind::Unevaluated {
                        def: d,
                        args: self.type_args(&u.args),
                    },
                    user_ty: None,
                    ty: None,
                }
            }
            CK::Param(p) => {
                self.err(format!("ConstantKind::Param({})", p.name));
                sm::Constant::from_value(sm::ConstValue::ZeroSized)
            }
        }
    }

    /// The typed constant value of a type that has exactly one
    /// value: unit, tuples and arrays of such types, structs all
    /// of whose fields are such types, and enums with a single
    /// inhabited variant likewise (a variant with an uninhabited
    /// field type — an empty enum, `!`, a struct containing one —
    /// can never be constructed, so `Option<Empty>` has the single
    /// value `None`). Types with no values or many values, and
    /// unions (whose active field the type does not determine),
    /// yield `None`.
    fn zst_const_value(
        &mut self,
        ty: &rustc_public::ty::Ty,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        match self.unique_const_value(ty, depth) {
            UniqueConst::One(v) => Some(v),
            _ => None,
        }
    }

    fn unique_const_value(
        &mut self,
        ty: &rustc_public::ty::Ty,
        depth: usize,
    ) -> UniqueConst {
        if depth > 32 {
            return UniqueConst::Many;
        }
        // A coroutine-closure type's kind cannot be asked (see
        // `guarded_ty_kind`); its value is the closure value
        // itself, which this walk does not construct, so the
        // type has no unique value here.
        let Some(kind) = guarded_ty_kind(ty) else {
            return UniqueConst::Many;
        };
        match kind {
            TyKind::RigidTy(RigidTy::Never) => UniqueConst::Uninhabited,
            TyKind::RigidTy(RigidTy::Tuple(elems)) => {
                let mut fields = Vec::with_capacity(elems.len());
                for e in &elems {
                    match self.unique_const_value(e, depth + 1) {
                        UniqueConst::One(v) => fields.push(v),
                        UniqueConst::Uninhabited => return UniqueConst::Uninhabited,
                        UniqueConst::Many => return UniqueConst::Many,
                    }
                }
                if fields.is_empty() {
                    UniqueConst::One(sm::ConstValue::ZeroSized)
                } else {
                    UniqueConst::One(sm::ConstValue::Tuple(fields))
                }
            }
            TyKind::RigidTy(RigidTy::Array(elem, len)) => {
                let Ok(n) = len.eval_target_usize() else {
                    return UniqueConst::Many;
                };
                if n == 0 {
                    return UniqueConst::One(sm::ConstValue::Tuple(Vec::new()));
                }
                if n > 4096 {
                    return UniqueConst::Many;
                }
                match self.unique_const_value(&elem, depth + 1) {
                    UniqueConst::One(v) => {
                        UniqueConst::One(sm::ConstValue::Tuple(vec![v; n as usize]))
                    }
                    UniqueConst::Uninhabited => UniqueConst::Uninhabited,
                    UniqueConst::Many => UniqueConst::Many,
                }
            }
            TyKind::RigidTy(RigidTy::Adt(def, args)) => {
                if def.kind() == AdtKind::Union {
                    return UniqueConst::Many;
                }
                let mut single: Option<(u32, Vec<sm::ConstValue>)> = None;
                for vi in 0..def.num_variants() {
                    let Some(variant) = def.variant(VariantIdx::to_val(vi)) else {
                        return UniqueConst::Many;
                    };
                    let mut fields = Vec::new();
                    let mut inhabited = true;
                    for field in variant.fields() {
                        let fty = field.ty_with_args(&args);
                        match self.unique_const_value(&fty, depth + 1) {
                            UniqueConst::One(v) => fields.push(v),
                            UniqueConst::Uninhabited => {
                                inhabited = false;
                                break;
                            }
                            UniqueConst::Many => return UniqueConst::Many,
                        }
                    }
                    if !inhabited {
                        continue;
                    }
                    if single.is_some() {
                        return UniqueConst::Many;
                    }
                    single = Some((vi as u32, fields));
                }
                match single {
                    Some((variant, fields)) => {
                        UniqueConst::One(sm::ConstValue::Adt { variant, fields })
                    }
                    None => UniqueConst::Uninhabited,
                }
            }
            _ => UniqueConst::Many,
        }
    }

    /// Interpret an allocation as a constant value, guided by its type.
    fn const_from_alloc(
        &mut self,
        ty: &rustc_public::ty::Ty,
        alloc: &Allocation,
    ) -> sm::ConstValue {
        // A constant whose type passes through a coroutine-
        // closure type cannot be asked its kind (see
        // `guarded_ty_kind`); the value is a zero-sized closure
        // value, or a composite containing one, and carries no
        // allocation bytes to decode.
        let Some(kind) = guarded_ty_kind(ty) else {
            return sm::ConstValue::ZeroSized;
        };
        match kind {
            TyKind::RigidTy(RigidTy::Bool) => match alloc.read_bool() {
                Ok(b) => sm::ConstValue::Bool(b),
                Err(_) => {
                    self.err("ConstantKind::Allocated (unreadable bool)");
                    sm::ConstValue::ZeroSized
                }
            },
            TyKind::RigidTy(RigidTy::Char) => match alloc.read_uint() {
                Ok(v) => match char::from_u32(v as u32) {
                    Some(ch) => sm::ConstValue::Char(ch),
                    None => {
                        self.err("ConstantKind::Allocated (invalid char)");
                        sm::ConstValue::ZeroSized
                    }
                },
                Err(_) => {
                    self.err("ConstantKind::Allocated (unreadable char)");
                    sm::ConstValue::ZeroSized
                }
            },
            TyKind::RigidTy(RigidTy::Int(_)) => match alloc.read_int() {
                Ok(v) => sm::ConstValue::Scalar(sm::Scalar::Int {
                    data: v as u128,
                    size: alloc.bytes.len() as u8,
                }),
                Err(_) => {
                    self.err("ConstantKind::Allocated (unreadable int)");
                    sm::ConstValue::ZeroSized
                }
            },
            TyKind::RigidTy(RigidTy::Uint(_)) => match alloc.read_uint() {
                Ok(v) => sm::ConstValue::Scalar(sm::Scalar::Int {
                    data: v,
                    size: alloc.bytes.len() as u8,
                }),
                Err(_) => {
                    self.err("ConstantKind::Allocated (unreadable uint)");
                    sm::ConstValue::ZeroSized
                }
            },
            TyKind::RigidTy(RigidTy::Float(f)) => {
                let bits = match f {
                    SmirFloatTy::F16 => alloc
                        .read_partial_uint(0..2)
                        .map(|v| v as u16 as u128)
                        .unwrap_or(0),
                    SmirFloatTy::F32 => alloc
                        .read_partial_uint(0..4)
                        .map(|v| v as u32 as u128)
                        .unwrap_or(0),
                    SmirFloatTy::F64 => alloc
                        .read_partial_uint(0..8)
                        .map(|v| v as u128)
                        .unwrap_or(0),
                    SmirFloatTy::F128 => alloc
                        .read_partial_uint(0..16)
                        .map(|v| v as u128)
                        .unwrap_or(0),
                };
                sm::ConstValue::Scalar(sm::Scalar::Float {
                    bits,
                    ty: float_ty(f),
                })
            }
            TyKind::RigidTy(RigidTy::Str) => match alloc.raw_bytes() {
                Ok(bytes) => match String::from_utf8(bytes) {
                    Ok(s) => sm::ConstValue::Str(s),
                    Err(_) => {
                        self.err("ConstantKind::Allocated (invalid UTF-8 str)");
                        sm::ConstValue::ZeroSized
                    }
                },
                Err(_) => {
                    self.err("ConstantKind::Allocated (unreadable str)");
                    sm::ConstValue::ZeroSized
                }
            },
            TyKind::RigidTy(RigidTy::Ref(_, inner, _)) => {
                self.ref_const(&inner, alloc)
            }
            TyKind::RigidTy(RigidTy::Tuple(elems)) if elems.is_empty() => {
                sm::ConstValue::ZeroSized
            }
            other => {
                let _ = &other;
                // Aggregate constants (tuples, arrays, ADTs) are
                // decoded field by field, guided by the layout.
                if let Some(v) = self.decode_const(ty, alloc, 0, 0) {
                    return v;
                }
                self.err(format!(
                    "ConstantKind::Allocated (type {})",
                    short_ty(ty)
                ));
                sm::ConstValue::ZeroSized
            }
        }
    }

    /// Read a fat-pointer allocation: provenance points at the data
    /// allocation, the second word is the length metadata.
    fn read_fat_pointer(&mut self, alloc: &Allocation) -> Option<(Vec<u8>, u64)> {
        let (_, prov) = alloc.provenance.ptrs.first()?;
        let target = mir::alloc::GlobalAlloc::from(prov.0);
        let mir::alloc::GlobalAlloc::Memory(target) = target else {
            return None;
        };
        let meta = alloc.read_partial_uint(8..16).ok()? as u64;
        let raw = target.raw_bytes().ok()?;
        // The data is exactly the metadata's byte count: the
        // backing allocation may extend past it (the file path
        // allocation behind a `Location` constant carries a
        // trailing NUL byte, for one).
        let data = raw.get(..meta as usize)?.to_vec();
        Some((data, meta))
    }

    /// Constants of reference type. The allocation of a reference
    /// constant holds the pointer itself, with provenance identifying
    /// the pointee allocation:
    /// - fat pointers (`&str`, `&[T]`): pointer + length word;
    /// - thin pointers (`&[i32; 3]`, `&&str` pieces, `&Location`): a
    ///   single provenance entry.
    /// Pointees made of plain scalar data are materialized; anything
    /// else becomes `ConstValue::ByRef` keyed by the Stable MIR
    /// allocation id, which preserves the constant's identity exactly
    /// (the runtime resolves the id against the allocation table).
    fn ref_const(
        &mut self,
        inner: &rustc_public::ty::Ty,
        alloc: &Allocation,
    ) -> sm::ConstValue {
        // The pointee kind, asked once: a type passing through
        // a coroutine-closure type cannot be asked at all (see
        // `guarded_ty_kind`), and none of the materialisation
        // shortcuts below apply to it.
        let inner_kind = guarded_ty_kind(inner);
        // Fat-pointer shape: `&str` and `&[u8]` literals take the
        // shortcut; every other shape falls through to the decoder
        // below (a raw-byte slice of wider or non-scalar elements
        // has no value form in the interpreter).
        if let Some((data, meta)) = self.read_fat_pointer(alloc) {
            match &inner_kind {
                Some(TyKind::RigidTy(RigidTy::Str)) => {
                    return match String::from_utf8(data) {
                        Ok(s) => sm::ConstValue::Str(s),
                        Err(_) => {
                            self.err("ConstantKind::Allocated (&str, invalid UTF-8)");
                            sm::ConstValue::ZeroSized
                        }
                    }
                }
                Some(TyKind::RigidTy(RigidTy::Slice(elem)))
                    if matches!(
                        elem.kind(),
                        TyKind::RigidTy(RigidTy::Uint(SmirUintTy::U8))
                    ) =>
                {
                    return sm::ConstValue::Slice { data, meta };
                }
                _ => {}
            }
        }
        // Thin-pointer shape: follow the provenance to the pointee.
        if let Some((_, prov)) = alloc.provenance.ptrs.first() {
            let alloc_id = prov.0.to_index() as u64;
            if let mir::alloc::GlobalAlloc::Memory(pointee) =
                mir::alloc::GlobalAlloc::from(prov.0)
            {
                let plain = pointee.provenance.ptrs.is_empty();
                match &inner_kind {
                    Some(TyKind::RigidTy(RigidTy::Str)) if plain => {
                        return match pointee.raw_bytes().map(String::from_utf8) {
                            Ok(Ok(s)) => sm::ConstValue::Str(s),
                            _ => {
                                self.err("ConstantKind::Allocated (&str, unreadable)");
                                sm::ConstValue::ZeroSized
                            }
                        };
                    }
                    Some(TyKind::RigidTy(RigidTy::Slice(elem))) if plain => {
                        if matches!(
                            elem.kind(),
                            TyKind::RigidTy(RigidTy::Uint(SmirUintTy::U8))
                        ) {
                            return match pointee.raw_bytes() {
                                Ok(data) => sm::ConstValue::Slice {
                                    meta: data.len() as u64,
                                    data,
                                },
                                Err(_) => {
                                    self.err("ConstantKind::Allocated (&[u8], unreadable)");
                                    sm::ConstValue::ZeroSized
                                }
                            };
                        }
                    }
                    Some(TyKind::RigidTy(RigidTy::Array(elem, _))) if plain => {
                        if matches!(
                            elem.kind(),
                            TyKind::RigidTy(RigidTy::Uint(SmirUintTy::U8))
                        ) {
                            return match pointee.raw_bytes() {
                                Ok(data) => sm::ConstValue::Bytes(data),
                                Err(_) => {
                                    self.err("ConstantKind::Allocated (&[u8; N], unreadable)");
                                    sm::ConstValue::ZeroSized
                                }
                            };
                        }
                    }
                    _ => {}
                }
            }
            // General shape: decode the pointee allocation by the
            // inner type, guided by the type's layout. Anything the
            // decoder cannot materialise keeps the exact `ByRef`
            // identity and stays unsupported downstream.
            if let Some(v) = self.decode_ref(inner, alloc, 0, 0) {
                return v;
            }
            return sm::ConstValue::ByRef {
                alloc_id,
                offset: 0,
            };
        }
        // No provenance at all. A fat pointer (a slice reference)
        // is an address word plus a metadata (length) word: a
        // dangling slice constant minted by
        // `slice::from_raw_parts(ptr, len)` in a const context has
        // no pointee, but its address identity and length are
        // exactly observable, so it decodes as a `FatPtr`.
        if matches!(&inner_kind, Some(TyKind::RigidTy(RigidTy::Slice(_)))) {
            if let (Ok(addr), Ok(meta)) = (
                alloc.read_partial_uint(0..PTR_SIZE),
                alloc.read_partial_uint(PTR_SIZE..2 * PTR_SIZE),
            ) {
                return sm::ConstValue::FatPtr {
                    addr: addr as u64,
                    meta: meta as u64,
                };
            }
        }
        // A stored zero word is a null pointer constant. Anything
        // else is an integer-derived address, which has no faithful
        // value form.
        if alloc.read_partial_uint(0..PTR_SIZE).ok() == Some(0) {
            return sm::ConstValue::NullPtr;
        }
        self.err(format!(
            "ConstantKind::Allocated (reference without provenance: {})",
            short_ty(inner)
        ));
        sm::ConstValue::ZeroSized
    }


    /// Whether a closure type's captured environment is empty: the
    /// environment tuple is the last of the closure's generic
    /// arguments (after the parent arguments, the kind and the
    /// signature). Anything not recognisably an empty tuple counts
    /// as non-empty.
    fn closure_env_is_empty(args: &GenericArgs) -> bool {
        match args.0.last() {
            Some(GenericArgKind::Type(t)) => match t.kind() {
                TyKind::RigidTy(RigidTy::Tuple(elems)) => elems.is_empty(),
                _ => false,
            },
            _ => false,
        }
    }

    /// Decode the value of type `ty` stored in `alloc` at `offset`
    /// into a structured constant. Aggregates are laid out with the
    /// type's own layout (`Ty::layout`), pointers are followed
    /// through provenance, and enums are resolved through their tag
    /// (direct or niche encoding). Returns `None` for anything that
    /// has no faithful value form: unions, function pointers and
    /// vtables, references to functions, null pointers,
    /// unsized values behind no pointer, and nesting deeper than
    /// the depth budget (cyclic statics exhaust it).
    fn decode_const(
        &mut self,
        ty: &rustc_public::ty::Ty,
        alloc: &Allocation,
        offset: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        if depth > MAX_CONST_DEPTH {
            return None;
        }
        // A range whose bytes are all uninitialized is the
        // uninitialized value of this type (the payload of a
        // `MaybeUninit::uninit()` constant, at whichever nesting
        // level it appears). A partially initialized range falls
        // through to the structural decode below, which reads the
        // initialized parts field by field.
        if let Ok(layout) = ty.layout() {
            let size = layout.shape().size.bytes() as usize;
            if size > 0
                && offset + size <= alloc.bytes.len()
                && alloc.bytes[offset..offset + size]
                    .iter()
                    .all(|b| b.is_none())
            {
                return Some(sm::ConstValue::Uninit);
            }
        }
        match ty.kind() {
            TyKind::RigidTy(RigidTy::Bool) => match read_uint_at(alloc, offset, 1)? {
                0 => Some(sm::ConstValue::Bool(false)),
                1 => Some(sm::ConstValue::Bool(true)),
                _ => None,
            },
            TyKind::RigidTy(RigidTy::Char) => {
                let v = read_uint_at(alloc, offset, 4)?;
                char::from_u32(v as u32).map(sm::ConstValue::Char)
            }
            TyKind::RigidTy(RigidTy::Int(t)) => {
                let size = int_ty_size(&t)?;
                let raw = read_uint_at(alloc, offset, size)?;
                Some(sm::ConstValue::Scalar(sm::Scalar::Int {
                    data: sign_extend(raw, size),
                    size: size as u8,
                }))
            }
            TyKind::RigidTy(RigidTy::Uint(t)) => {
                let size = uint_ty_size(&t)?;
                let raw = read_uint_at(alloc, offset, size)?;
                Some(sm::ConstValue::Scalar(sm::Scalar::Int {
                    data: raw,
                    size: size as u8,
                }))
            }
            TyKind::RigidTy(RigidTy::Float(f)) => {
                let size = match f {
                    SmirFloatTy::F16 => 2,
                    SmirFloatTy::F32 => 4,
                    SmirFloatTy::F64 => 8,
                    SmirFloatTy::F128 => 16,
                };
                let bits = read_uint_at(alloc, offset, size)?;
                Some(sm::ConstValue::Scalar(sm::Scalar::Float {
                    bits,
                    ty: float_ty(f),
                }))
            }
            TyKind::RigidTy(RigidTy::Str) => {
                // A `str` value in non-reference position is the
                // trailing unsized field of an ADT (a newtype over
                // `str`): its bytes run to the end of the allocation.
                let raw = alloc.raw_bytes().ok()?;
                let bytes = raw.get(offset..)?;
                String::from_utf8(bytes.to_vec()).ok().map(sm::ConstValue::Str)
            }
            TyKind::RigidTy(RigidTy::Slice(elem)) => {
                // Likewise a slice in non-reference position is a
                // trailing unsized field (the `[u8]` inside `CStr`
                // is the canonical case): its elements run to the
                // end of the allocation.
                let raw = alloc.raw_bytes().ok()?;
                let bytes = raw.get(offset..)?;
                if matches!(
                    elem.kind(),
                    TyKind::RigidTy(RigidTy::Uint(SmirUintTy::U8))
                ) {
                    return Some(sm::ConstValue::Bytes(bytes.to_vec()));
                }
                let stride = elem.layout().ok()?.shape().size.bytes();
                if stride == 0 || bytes.len() % stride != 0 {
                    return None;
                }
                let mut elems = Vec::new();
                for i in 0..bytes.len() / stride {
                    elems.push(self.decode_const(
                        &elem,
                        alloc,
                        offset + i * stride,
                        depth + 1,
                    )?);
                }
                Some(sm::ConstValue::SliceVal(elems))
            }
            TyKind::RigidTy(RigidTy::Ref(_, inner, _)) => {
                self.decode_ref(&inner, alloc, offset, depth)
            }
            TyKind::RigidTy(RigidTy::RawPtr(inner, _)) => {
                self.decode_ref(&inner, alloc, offset, depth)
            }
            TyKind::RigidTy(RigidTy::Array(elem, len)) => {
                let count = len.eval_target_usize().ok()?;
                let shape = ty.layout().ok()?.shape();
                let FieldsShape::Array { stride, count: layout_count } = shape.fields else {
                    return None;
                };
                if layout_count != count { return None; }
                if count > MAX_CONST_ARRAY_ELEMS {
                    // Past the element-wise bound, a byte array
                    // still materialises -- as the raw bytes
                    // themselves, the form byte arrays take in
                    // reference position (and byte slices take in
                    // value position above). The `Bytes` value's
                    // size is bounded by the allocation, which
                    // the host already holds, so the resource
                    // concern behind the bound -- a declared
                    // length decoupled from the allocation,
                    // expanded into one constant per element --
                    // does not apply. Any other element type
                    // keeps the bound and stays undecoded.
                    if stride.bytes() != 1
                        || !matches!(elem.kind(), TyKind::RigidTy(RigidTy::Uint(SmirUintTy::U8)))
                    {
                        return None;
                    }
                    // Read the array's own byte range, not the
                    // whole allocation: a static's storage is
                    // routinely padded past the value with
                    // uninitialised bytes, which must not fail
                    // the decode -- but an uninitialised byte
                    // inside the array does.
                    let end = offset.checked_add(count as usize)?;
                    let range = alloc.bytes.get(offset..end)?;
                    let mut bytes = Vec::with_capacity(count as usize);
                    for byte in range {
                        bytes.push((*byte)?);
                    }
                    return Some(sm::ConstValue::Bytes(bytes));
                }
                let mut elems = Vec::new();
                for i in 0..count {
                    elems.push(self.decode_const(
                        &elem,
                        alloc,
                        offset + i as usize * stride.bytes(),
                        depth + 1,
                    )?);
                }
                Some(sm::ConstValue::Tuple(elems))
            }
            TyKind::RigidTy(RigidTy::Tuple(elems)) => {
                if elems.is_empty() {
                    return Some(sm::ConstValue::ZeroSized);
                }
                let offsets = arbitrary_offsets(ty)?;
                if offsets.len() != elems.len() { return None; }
                let mut fields = Vec::new();
                for (elem, off) in elems.iter().zip(offsets) {
                    fields.push(self.decode_const(elem, alloc, offset + off, depth + 1)?);
                }
                Some(sm::ConstValue::Tuple(fields))
            }
            TyKind::RigidTy(RigidTy::FnDef(def, args)) => {
                // A function item value is zero-sized: the type
                // itself names the callable, no bytes are stored.
                if let Ok(inst) = rustc_public::mir::mono::Instance::resolve(def, &args) {
                    self.note_const_fn(&inst);
                }
                let d = self.fn_def_id(&def, &args);
                Some(sm::ConstValue::FnDef(d))
            }
            TyKind::RigidTy(RigidTy::FnPtr(_)) => {
                self.decode_fn_pointer(alloc, offset)
            }
            TyKind::RigidTy(RigidTy::Closure(def, args)) => {
                // A stored closure value with no captures is the
                // callable itself; a captured environment has no
                // constant form here and stays undecoded.
                if Self::closure_env_is_empty(&args) {
                    self.note_const_closure(&def, &args);
                    let d = self.closure_def_id(&def, &args);
                    Some(sm::ConstValue::Closure(d))
                } else {
                    None
                }
            }
            TyKind::RigidTy(RigidTy::Pat(inner, _)) => {
                // Pattern types (e.g. the `usize is 0..=N` field of
                // `core::num::niche_types` wrappers) share the inner
                // type's representation; the stored value is the
                // inner type's value.
                self.decode_const(&inner, alloc, offset, depth)
            }
            TyKind::RigidTy(RigidTy::Adt(def, args)) => {
                self.decode_adt(&def, &args, ty, alloc, offset, depth)
            }
            _ => None,
        }
    }

    /// Decode an ADT value (struct or enum) at `offset` in `alloc`.
    fn decode_adt(
        &mut self,
        def: &AdtDef,
        args: &GenericArgs,
        ty: &rustc_public::ty::Ty,
        alloc: &Allocation,
        offset: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        if matches!(def.name().as_str(), "core::ffi::CStr" | "std::ffi::CStr") {
            // The one field is a `[c_char]` byte string. `c_char`
            // is signed on this platform, so the generic decode
            // would produce signed-char elements, but a `CStr`'s
            // contents are bytes by definition — every public
            // view of it is a byte view (`to_bytes`, `to_str`,
            // the `&[u8]` casts its own methods perform) — so
            // decode them as the byte string they are, like the
            // `[u8]` trailing-slice decode below.
            let raw = alloc.raw_bytes().ok()?;
            let bytes = raw.get(offset..)?.to_vec();
            return Some(sm::ConstValue::Adt {
                variant: 0,
                fields: vec![sm::ConstValue::Bytes(bytes)],
            });
        }
        let shape = ty.layout().ok()?.shape();
        let (variant_idx, offsets): (usize, Vec<usize>) = match def.kind() {
            AdtKind::Struct => {
                let variant = def.variant(VariantIdx::to_val(0))?;
                if variant.fields().is_empty() {
                    return Some(sm::ConstValue::Adt { variant: 0, fields: Vec::new() });
                }
                (0, arbitrary_offsets(ty)?)
            }
            AdtKind::Enum => match &shape.variants {
                VariantsShape::Single { index } => {
                    let idx = index.to_index();
                    let variant = def.variant(VariantIdx::to_val(idx))?;
                    if variant.fields().is_empty() {
                        return Some(sm::ConstValue::Adt {
                            variant: idx as u32,
                            fields: Vec::new(),
                        });
                    }
                    (idx, arbitrary_offsets(ty)?)
                }
                VariantsShape::Multiple {
                    tag,
                    tag_encoding,
                    tag_field,
                    variants,
                } => {
                    let (idx, offsets) = match tag_encoding {
                        TagEncoding::Direct => {
                            // The stored tag's width and signedness
                            // come from the layout's tag scalar, not
                            // from the enum's discriminant type:
                            // rustc stores a small enum's tag in a
                            // scalar narrower than the type-level
                            // discriminant type (a fieldless enum
                            // with two variants gets a 1-byte tag
                            // while its discriminant type is isize).
                            let (size, signed) = tag_scalar_size_signed(tag)?;
                            let tag_off = arbitrary_offsets(ty)?.get(*tag_field).copied()?;
                            let raw = read_uint_at(alloc, offset + tag_off, size)?;
                            let val = if signed { sign_extend(raw, size) } else { raw };
                            // Compare in the value domain: the tag
                            // read above is sign-extended from the
                            // tag's width, while `Discr::val` is the
                            // bit pattern at the discriminant type's
                            // width (see `discr_value`); the raw
                            // patterns agree only when both widths
                            // are 128, so a negative discriminant
                            // never matched before this normalisation.
                            let tag_value = val as i128;
                            let mut found = None;
                            for i in 0..def.num_variants() {
                                let d = def.discriminant_for_variant(VariantIdx::to_val(i));
                                if discr_value(&d) == Some(tag_value) {
                                    found = Some(i);
                                    break;
                                }
                            }
                            let idx = found?;
                            let offsets = variants
                                .get(idx)?
                                .offsets
                                .iter()
                                .map(|o| o.bytes())
                                .collect();
                            (idx, offsets)
                        }
                        TagEncoding::Niche {
                            untagged_variant,
                            niche_variants,
                            niche_start,
                        } => {
                            let untagged = untagged_variant.to_index();
                            // The niche scalar's absolute offset
                            // within the enum is the layout's sole
                            // top-level field offset (the layout
                            // computation records the niche offset
                            // there), and its width is the tag
                            // scalar's width. The dataful variant's
                            // field that *contains* the niche may
                            // start earlier and be wider (a whole
                            // `Layout` around its alignment word),
                            // so neither its offset nor its size
                            // identifies the niche itself.
                            let tag_off = arbitrary_offsets(ty)?.get(*tag_field).copied()?;
                            let (size, _) = tag_scalar_size_signed(tag)?;
                            if size == 0 || size > 16 { return None; }
                            let raw = read_uint_at(alloc, offset + tag_off, size)?;
                            let start = niche_variants.start().to_index();
                            let end = niche_variants.end().to_index();
                            // A provenance entry at the niche field
                            // means the stored value is a real
                            // pointer, not a niche encoding of
                            // another variant (the `Option<&T>` case:
                            // a non-null pointer reads back as
                            // offset 0 in its target allocation).
                            let has_prov = alloc
                                .provenance
                                .ptrs
                                .iter()
                                .any(|(o, _)| *o as usize == offset + tag_off);
                            let idx = if !has_prov
                                && raw >= *niche_start
                                && raw - *niche_start <= (end - start) as u128
                            {
                                start + (raw - *niche_start) as usize
                            } else {
                                untagged
                            };
                            let offsets = variants
                                .get(idx)?
                                .offsets
                                .iter()
                                .map(|o| o.bytes())
                                .collect();
                            (idx, offsets)
                        }
                    };
                    (idx, offsets)
                }
                VariantsShape::Empty => return None,
            },
            AdtKind::Union => return self.decode_union(&def, &args, alloc, offset, depth),
        };
        let variant = def.variant(VariantIdx::to_val(variant_idx))?;
        let fields = self.decode_fields(&variant, args, &offsets, alloc, offset, depth)?;
        Some(sm::ConstValue::Adt {
            variant: variant_idx as u32,
            fields,
        })
    }

    /// Decode a union value. Which field is active is not recorded
    /// in an allocation, but a union *value* in this model carries
    /// every field, and reading a union field in Rust reinterprets
    /// the stored bytes as that field's type — so decoding every
    /// field from the same bytes at offset 0 (where all union
    /// fields live) is exactly the semantics a later field read
    /// performs. Scalar and aggregate decodes are total on bytes,
    /// so a multi-payload union decodes field by field; a field
    /// whose decode fails (a pointer without provenance is the
    /// usual case, handled by the disambiguation below when it is
    /// the stored field) fails the whole constant, honestly.
    fn decode_union(
        &mut self,
        def: &AdtDef,
        args: &GenericArgs,
        alloc: &Allocation,
        offset: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        let variant = def.variant(VariantIdx::to_val(0))?;
        let fields = variant.fields();
        let mut sizes = Vec::new();
        for field in fields.iter() {
            let fty = field.ty_with_args(args);
            let size = fty.layout().ok()?.shape().size.bytes();
            sizes.push(size);
        }
        // A provenance entry at the union's offset can only belong
        // to a pointer-shaped field stored there: pointer fields
        // are the ones that carry provenance, and the entry spans
        // exactly one pointer word. When exactly one field is a
        // pointer of that size, it is the stored field; the other
        // payload fields overlap its bytes and decode as
        // uninitialized (they are not the live field, and reading
        // them would reinterpret the pointer's representation).
        let prov_at_offset = alloc
            .provenance
            .ptrs
            .iter()
            .any(|(at, _)| *at as usize == offset);
        if prov_at_offset {
            let ptr_fields: Vec<usize> = fields
                .iter()
                .enumerate()
                .filter(|(i, field)| {
                    if sizes[*i] != PTR_SIZE {
                        return false;
                    }
                    // The pointer may sit under single-field
                    // transparent wrappers (`ManuallyDrop`,
                    // `UnsafeCell`, ...).
                    let mut inner = field.ty_with_args(args);
                    for _ in 0..8 {
                        match inner.kind() {
                            TyKind::RigidTy(RigidTy::Adt(def, inner_args)) => {
                                let variants = def.variants();
                                if variants.len() != 1 {
                                    return false;
                                }
                                let vfields = variants[0].fields();
                                if vfields.len() != 1 {
                                    return false;
                                }
                                inner = vfields[0].ty_with_args(&inner_args);
                            }
                            _ => break,
                        }
                    }
                    matches!(
                        inner.kind(),
                        TyKind::RigidTy(
                            RigidTy::Ref(..) | RigidTy::RawPtr(..) | RigidTy::FnPtr(_)
                        )
                    )
                })
                .map(|(i, _)| i)
                .collect();
            if ptr_fields.len() == 1 {
                let live = ptr_fields[0];
                let mut out = Vec::new();
                for (i, field) in fields.iter().enumerate() {
                    let fty = field.ty_with_args(args);
                    if i == live || sizes[i] == 0 {
                        out.push(self.decode_const(&fty, alloc, offset, depth + 1)?);
                    } else {
                        out.push(sm::ConstValue::Uninit);
                    }
                }
                return Some(sm::ConstValue::Adt {
                    variant: 0,
                    fields: out,
                });
            }
        }
        let mut out = Vec::new();
        for field in fields.iter() {
            let fty = field.ty_with_args(args);
            out.push(self.decode_const(&fty, alloc, offset, depth + 1)?);
        }
        Some(sm::ConstValue::Adt {
            variant: 0,
            fields: out,
        })
    }

    /// Decode the fields of one variant at the given offsets.
    fn decode_fields(
        &mut self,
        variant: &rustc_public::ty::VariantDef,
        args: &GenericArgs,
        offsets: &[usize],
        alloc: &Allocation,
        offset: usize,
        depth: usize,
    ) -> Option<Vec<sm::ConstValue>> {
        let fields = variant.fields();
        if fields.len() != offsets.len() { return None; }
        let mut out = Vec::new();
        for (field, off) in fields.iter().zip(offsets) {
            let fty = field.ty_with_args(args);
            out.push(self.decode_const(&fty, alloc, offset + off, depth + 1)?);
        }
        Some(out)
    }

    /// Record the materialisation plan of a pointer cast's
    /// target pointee type (see `mat_plan`), keyed by the
    /// converted pointee type's `Debug` rendering — the key
    /// `mir_convert` looks up when it converts the same cast.
    fn note_mat_plan(&mut self, ty: &rustc_public::ty::Ty, converted: &sm::Ty) {
        let pointee = match converted {
            sm::Ty::Ref(_, p, _) | sm::Ty::RawPtr(p, _) => (**p).clone(),
            _ => return,
        };
        let key = format!("{pointee:?}");
        if self.mat_plans.contains_key(&key) {
            return;
        }
        let rustc_pointee = match ty.kind() {
            TyKind::RigidTy(RigidTy::Ref(_, inner, _))
            | TyKind::RigidTy(RigidTy::RawPtr(inner, _)) => inner,
            _ => return,
        };
        let mut active = HashSet::new();
        let plan = self.mat_plan(&rustc_pointee, 0, &mut active);
        self.mat_plans.insert(key, plan);
    }

    /// The materialisation plan of a type (see
    /// `sparagmos_interp::ir::MatPlan`): the byte-layout
    /// description the interpreter needs to materialise a
    /// type-erased constant allocation into a value of this
    /// type. Built from the compiler's layout query, mirroring
    /// `decode_const`'s reading of the same layouts; a type
    /// whose layout cannot be described gets
    /// `MatPlan::Unsupported` (materialising it is an explicit
    /// error downstream, never a guess).
    fn mat_plan(
        &mut self,
        ty: &rustc_public::ty::Ty,
        depth: usize,
        active: &mut HashSet<rustc_public::ty::Ty>,
    ) -> sparagmos_interp::ir::MatPlan {
        use sparagmos_interp::ir::MatPlan;
        if depth > MAX_CONST_DEPTH {
            return MatPlan::Unsupported;
        }
        // A type already being expanded on the path from the
        // cast's target type: the type graph cycles here, and
        // expanding the type again would unroll the cycle into
        // a tree whose size grows like the branching factor
        // raised to the depth limit (a recursive struct with
        // three pointer fields alone would build tens of
        // millions of plans and exhaust memory during
        // conversion, for every cast that mentions the type).
        // The recursive edge gets no plan -- materialising
        // through it is an explicit error, never a guess --
        // and every other edge of the type plans as before.
        if !active.insert(*ty) {
            return MatPlan::Unsupported;
        }
        let plan = self.mat_plan_body(ty, depth, active);
        active.remove(ty);
        plan
    }

    /// The plan of one type, expanded one level (see
    /// `mat_plan`, which guards the expansion against type
    /// graph cycles).
    fn mat_plan_body(
        &mut self,
        ty: &rustc_public::ty::Ty,
        depth: usize,
        active: &mut HashSet<rustc_public::ty::Ty>,
    ) -> sparagmos_interp::ir::MatPlan {
        use sparagmos_interp::ir::MatPlan;
        match ty.kind() {
            TyKind::RigidTy(RigidTy::Bool) => MatPlan::Bool,
            TyKind::RigidTy(RigidTy::Char) => MatPlan::Char,
            TyKind::RigidTy(RigidTy::Int(t)) => match int_ty_size(&t) {
                Some(size) => MatPlan::Int {
                    size: size as u8,
                    signed: true,
                },
                None => MatPlan::Unsupported,
            },
            TyKind::RigidTy(RigidTy::Uint(t)) => match uint_ty_size(&t) {
                Some(size) => MatPlan::Int {
                    size: size as u8,
                    signed: false,
                },
                None => MatPlan::Unsupported,
            },
            TyKind::RigidTy(RigidTy::Float(f)) => MatPlan::Float {
                size: match f {
                    SmirFloatTy::F16 => 2,
                    SmirFloatTy::F32 => 4,
                    SmirFloatTy::F64 => 8,
                    SmirFloatTy::F128 => 16,
                },
            },
            TyKind::RigidTy(RigidTy::Str) => MatPlan::Str,
            TyKind::RigidTy(RigidTy::Slice(elem)) => {
                MatPlan::Slice(Box::new(self.mat_plan(&elem, depth + 1, active)))
            }
            TyKind::RigidTy(RigidTy::Ref(_, inner, _))
            | TyKind::RigidTy(RigidTy::RawPtr(inner, _)) => {
                // A `dyn` pointee's value needs its vtable,
                // which no plan can name; everything else
                // recurses into the pointee.
                if matches!(inner.kind(), TyKind::RigidTy(RigidTy::Dynamic(..))) {
                    MatPlan::Unsupported
                } else {
                    MatPlan::Ref(Box::new(self.mat_plan(&inner, depth + 1, active)))
                }
            }
            TyKind::RigidTy(RigidTy::FnPtr(_)) => MatPlan::FnPtr,
            TyKind::RigidTy(RigidTy::FnDef(..)) => MatPlan::Unit,
            TyKind::RigidTy(RigidTy::Tuple(elems)) => {
                if elems.is_empty() {
                    return MatPlan::Unit;
                }
                let Some(offsets) = arbitrary_offsets(ty) else {
                    return MatPlan::Unsupported;
                };
                if offsets.len() != elems.len() {
                    return MatPlan::Unsupported;
                }
                let Ok(layout) = ty.layout() else {
                    return MatPlan::Unsupported;
                };
                let fields = elems
                    .iter()
                    .zip(offsets)
                    .map(|(e, off)| sparagmos_interp::ir::MatField {
                        offset: off as u64,
                        plan: self.mat_plan(e, depth + 1, active),
                    })
                    .collect();
                MatPlan::Tuple {
                    size: layout.shape().size.bytes() as u64,
                    fields,
                }
            }
            TyKind::RigidTy(RigidTy::Array(elem, len)) => {
                let Ok(count) = len.eval_target_usize() else {
                    return MatPlan::Unsupported;
                };
                let Ok(layout) = ty.layout() else {
                    return MatPlan::Unsupported;
                };
                let FieldsShape::Array { stride, .. } = layout.shape().fields else {
                    return MatPlan::Unsupported;
                };
                MatPlan::Array {
                    elem: Box::new(self.mat_plan(&elem, depth + 1, active)),
                    stride: stride.bytes() as u64,
                    count: count as u64,
                }
            }
            TyKind::RigidTy(RigidTy::Adt(def, args)) => {
                self.mat_plan_adt(&def, &args, ty, depth, active)
            }
            _ => MatPlan::Unsupported,
        }
    }

    /// The materialisation plan of an ADT instance (see
    /// `mat_plan`), mirroring `decode_adt`'s reading of the
    /// layout: field offsets are the layout's, and the enum tag
    /// placement is the layout's tag encoding.
    fn mat_plan_adt(
        &mut self,
        def: &AdtDef,
        args: &GenericArgs,
        ty: &rustc_public::ty::Ty,
        depth: usize,
        active: &mut HashSet<rustc_public::ty::Ty>,
    ) -> sparagmos_interp::ir::MatPlan {
        use sparagmos_interp::ir::{MatField, MatPlan, MatTag};
        let Ok(layout) = ty.layout() else {
            return MatPlan::Unsupported;
        };
        let shape = layout.shape();
        let size = shape.size.bytes() as u64;
        match def.kind() {
            AdtKind::Struct => {
                let Some(variant) = def.variant(VariantIdx::to_val(0)) else {
                    return MatPlan::Unsupported;
                };
                if variant.fields().is_empty() {
                    return MatPlan::Struct {
                        size,
                        fields: Vec::new(),
                    };
                }
                let Some(offsets) = arbitrary_offsets(ty) else {
                    return MatPlan::Unsupported;
                };
                match self.mat_plan_fields(&variant, args, &offsets, depth, active) {
                    Some(fields) => MatPlan::Struct { size, fields },
                    None => MatPlan::Unsupported,
                }
            }
            AdtKind::Enum => match &shape.variants {
                VariantsShape::Single { index } => {
                    let idx = index.to_index();
                    let mut variants = Vec::new();
                    for i in 0..def.num_variants() {
                        if i == idx {
                            let variant = match def.variant(VariantIdx::to_val(i)) {
                                Some(v) => v,
                                None => return MatPlan::Unsupported,
                            };
                            if variant.fields().is_empty() {
                                variants.push(Vec::new());
                                continue;
                            }
                            let Some(offsets) = arbitrary_offsets(ty) else {
                                return MatPlan::Unsupported;
                            };
                            match self.mat_plan_fields(&variant, args, &offsets, depth, active) {
                                Some(fields) => variants.push(fields),
                                None => return MatPlan::Unsupported,
                            }
                        } else {
                            variants.push(Vec::new());
                        }
                    }
                    MatPlan::Enum {
                        size,
                        tag: MatTag::Single(idx as u32),
                        variants,
                    }
                }
                VariantsShape::Multiple {
                    tag,
                    tag_encoding,
                    tag_field,
                    variants,
                } => {
                    let mat_tag = match tag_encoding {
                        TagEncoding::Direct => {
                            let Some((tsize, signed)) = tag_scalar_size_signed(tag) else {
                                return MatPlan::Unsupported;
                            };
                            let Some(offsets) = arbitrary_offsets(ty) else {
                                return MatPlan::Unsupported;
                            };
                            let Some(tag_off) = offsets.get(*tag_field).copied() else {
                                return MatPlan::Unsupported;
                            };
                            let mut discriminants = Vec::new();
                            for i in 0..def.num_variants() {
                                let d = def.discriminant_for_variant(VariantIdx::to_val(i));
                                match discr_value(&d) {
                                    Some(v) => discriminants.push(v),
                                    None => return MatPlan::Unsupported,
                                }
                            }
                            MatTag::Direct {
                                offset: tag_off as u64,
                                size: tsize as u8,
                                signed,
                                discriminants,
                            }
                        }
                        TagEncoding::Niche {
                            untagged_variant,
                            niche_variants,
                            niche_start,
                        } => {
                            let Some((tsize, _)) = tag_scalar_size_signed(tag) else {
                                return MatPlan::Unsupported;
                            };
                            let Some(offsets) = arbitrary_offsets(ty) else {
                                return MatPlan::Unsupported;
                            };
                            let Some(tag_off) = offsets.get(*tag_field).copied() else {
                                return MatPlan::Unsupported;
                            };
                            MatTag::Niche {
                                offset: tag_off as u64,
                                size: tsize as u8,
                                untagged: untagged_variant.to_index() as u32,
                                niche_start: *niche_start,
                                start: niche_variants.start().to_index() as u32,
                                end: niche_variants.end().to_index() as u32,
                            }
                        }
                    };
                    let mut variant_fields = Vec::new();
                    for (i, vshape) in variants.iter().enumerate() {
                        let variant = match def.variant(VariantIdx::to_val(i)) {
                            Some(v) => v,
                            None => return MatPlan::Unsupported,
                        };
                        let offsets: Vec<usize> =
                            vshape.offsets.iter().map(|o| o.bytes()).collect();
                        match self.mat_plan_fields(&variant, args, &offsets, depth, active) {
                            Some(fields) => variant_fields.push(fields),
                            None => return MatPlan::Unsupported,
                        }
                    }
                    MatPlan::Enum {
                        size,
                        tag: mat_tag,
                        variants: variant_fields,
                    }
                }
                VariantsShape::Empty => MatPlan::Unsupported,
            },
            AdtKind::Union => {
                let Some(variant) = def.variant(VariantIdx::to_val(0)) else {
                    return MatPlan::Unsupported;
                };
                let fields = variant
                    .fields()
                    .iter()
                    .map(|f| {
                        let fty = f.ty_with_args(args);
                        self.mat_plan(&fty, depth + 1, active)
                    })
                    .collect();
                MatPlan::Union { size, fields }
            }
        }
    }

    /// The materialisation plans of one variant's fields at the
    /// given byte offsets (see `mat_plan`).
    fn mat_plan_fields(
        &mut self,
        variant: &rustc_public::ty::VariantDef,
        args: &GenericArgs,
        offsets: &[usize],
        depth: usize,
        active: &mut HashSet<rustc_public::ty::Ty>,
    ) -> Option<Vec<sparagmos_interp::ir::MatField>> {
        let fields = variant.fields();
        if fields.len() != offsets.len() {
            return None;
        }
        let mut out = Vec::new();
        for (field, off) in fields.iter().zip(offsets) {
            let fty = field.ty_with_args(args);
            out.push(sparagmos_interp::ir::MatField {
                offset: *off as u64,
                plan: self.mat_plan(&fty, depth + 1, active),
            });
        }
        Some(out)
    }

    /// Decode the allocation region at `offset` in `alloc` as a
    /// type-erased block (see `sm::ConstValue::ErasedAlloc`):
    /// the region's bytes verbatim, plus every provenance
    /// relocation inside it decoded type-free by
    /// `decode_erased_slot`. `None` when a relocation cannot be
    /// decoded type-free; the caller then keeps the declared
    /// type's decode.
    fn decode_erased_alloc(
        &mut self,
        alloc: &Allocation,
        offset: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        if depth > MAX_CONST_DEPTH {
            return None;
        }
        let bytes = alloc.bytes.get(offset..)?.to_vec();
        let mut relocs = Vec::new();
        for (o, _) in &alloc.provenance.ptrs {
            let o = *o as usize;
            if o < offset {
                continue;
            }
            let v = self.decode_erased_slot(alloc, o, depth + 1)?;
            relocs.push(((o - offset) as u32, v));
        }
        Some(sm::ConstValue::ErasedAlloc { bytes, relocs })
    }

    /// Decode the pointer slot at `offset` in `alloc` without a
    /// declared pointee type: the slot's value as a reference to
    /// whatever its provenance names, with a memory pointee
    /// preserved as an erased block. A static's pointee is its
    /// initializer allocation's block: reads through a later
    /// materialisation are exact, though the block is a
    /// snapshot, not the static's store. Mirrors `decode_ref`'s
    /// provenance handling.
    fn decode_erased_slot(
        &mut self,
        alloc: &Allocation,
        offset: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        if depth > MAX_CONST_DEPTH {
            return None;
        }
        let prov = alloc
            .provenance
            .ptrs
            .iter()
            .find(|(o, _)| *o as usize == offset);
        let Some((_, prov)) = prov else {
            let addr = read_uint_at(alloc, offset, PTR_SIZE)?;
            if addr == 0 {
                return Some(sm::ConstValue::NullPtr);
            }
            return Some(sm::ConstValue::IntPtr(addr as u64));
        };
        let target_off = read_uint_at(alloc, offset, PTR_SIZE)? as usize;
        match mir::alloc::GlobalAlloc::from(prov.0) {
            mir::alloc::GlobalAlloc::Memory(target) => {
                let block = self.decode_erased_alloc(&target, target_off, depth + 1)?;
                Some(sm::ConstValue::Ref(Box::new(block)))
            }
            mir::alloc::GlobalAlloc::Static(def) => {
                if static_is_foreign(&def) {
                    return None;
                }
                let init = def.eval_initializer().ok()?;
                let block = self.decode_erased_alloc(&init, target_off, depth + 1)?;
                Some(sm::ConstValue::Ref(Box::new(block)))
            }
            mir::alloc::GlobalAlloc::Function(inst) => self.fn_pointer_value(&inst),
            mir::alloc::GlobalAlloc::TypeId { ty: tid_ty } => {
                let (lo, _) = type_id_segments(&tid_ty);
                Some(sm::ConstValue::IntPtr(lo))
            }
            mir::alloc::GlobalAlloc::VTable(vty, trait_ref) => {
                Some(sm::ConstValue::VTablePtr(vtable_id(&vty, &trait_ref)))
            }
        }
    }

    /// Whether a reference decoded with declared pointee type
    /// `inner` is type-erased: the declared type is zero-sized
    /// while the region of `target` it denotes (from
    /// `target_off` on) is not empty, so the pointer was stored
    /// through an erased slot and its real type is named only
    /// by later casts.
    fn pointee_is_erased(
        &self,
        inner: &rustc_public::ty::Ty,
        target: &Allocation,
        target_off: usize,
    ) -> bool {
        if std::env::var_os("NT_DISABLE_ERASED").is_some() {
            return false;
        }
        if target_off >= target.bytes.len() {
            return false;
        }
        // A genuinely unsized pointee (a `str`-tailed struct is
        // the canonical case) is not erased: its typed decode
        // reads the trailing field to the end of the allocation,
        // and the guest projects the value's fields directly,
        // with no cast that could materialise a block. A
        // sizedness the walk cannot prove keeps the typed
        // decode too.
        if ty_sizedness(inner, None, 0) != Sizedness::Sized {
            return false;
        }
        match inner.layout() {
            Ok(layout) => layout.shape().size.bytes() == 0,
            Err(_) => false,
        }
    }

    /// Decode a reference (or raw pointer) stored in `alloc` at
    /// `offset`, pointing at a value of type `inner`. The result is
    /// the reference's value in the interpreter's model: string and
    /// byte data directly, a fat slice value for decoded slices, and
    /// `ConstValue::Ref` over the decoded pointee otherwise.
    /// Decode a function pointer stored in `alloc` at `offset`:
    /// its provenance names the target instance. A plain function
    /// item becomes the callable of that item. A closure coerced to
    /// a function pointer points at the closure's `Fn` trait adapter
    /// shim (const evaluation picks the `FnOnce` one), whose `Self`
    /// argument is the closure type; a direct closure instance's
    /// own type is the closure type. Both identify the closure
    /// body, which is the callable, with an empty environment (only
    /// non-capturing closures coerce to function pointers).
    /// Anything else (a vtable, a null pointer with no provenance)
    /// has no callable constant form and stays undecoded.
    fn decode_fn_pointer(
        &mut self,
        alloc: &Allocation,
        offset: usize,
    ) -> Option<sm::ConstValue> {
        let (_, prov) = alloc
            .provenance
            .ptrs
            .iter()
            .find(|(o, _)| *o as usize == offset)?;
        let mir::alloc::GlobalAlloc::Function(inst) =
            mir::alloc::GlobalAlloc::from(prov.0)
        else {
            return None;
        };
        self.fn_pointer_value(&inst)
    }

    /// The callable constant naming a function instance: a plain
    /// function item becomes the callable of that item; a closure
    /// (named directly, or through the `Fn` adapter shim const
    /// evaluation picks for a coercion) becomes the closure value
    /// with an empty environment.
    fn fn_pointer_value(
        &mut self,
        inst: &rustc_public::mir::mono::Instance,
    ) -> Option<sm::ConstValue> {
        if matches!(inst.kind, InstanceKind::Shim) {
            for arg in &inst.args().0 {
                if let GenericArgKind::Type(t) = arg {
                    if let TyKind::RigidTy(RigidTy::Closure(def, args)) = t.kind() {
                        self.note_const_closure(&def, &args);
                        let d = self.closure_def_id(&def, &args);
                        return Some(sm::ConstValue::Closure(d));
                    }
                }
            }
        }
        if let TyKind::RigidTy(RigidTy::Closure(def, args)) = inst.ty().kind() {
            self.note_const_closure(&def, &args);
            let d = self.closure_def_id(&def, &args);
            return Some(sm::ConstValue::Closure(d));
        }
        self.note_const_fn(inst);
        let d = self.instance_def_id(inst.def.def_id(), inst);
        Some(sm::ConstValue::FnDef(d))
    }

    /// Record a closure materialized as a value out of a constant
    /// (see the `const_closure_mentions` field): the demand passes
    /// collect the bodies of the closures recorded here, because
    /// no type in any converted body mentions them.
    fn note_const_closure(
        &mut self,
        def: &rustc_public::ty::ClosureDef,
        args: &rustc_public::ty::GenericArgs,
    ) {
        let key = (def.def_id(), format!("{args:?}"));
        if self.const_closure_mentions_seen.insert(key) {
            self.const_closure_mentions.push((*def, args.clone()));
        }
    }

    /// Record a function item materialized as a value out of a
    /// constant (see the `const_fn_mentions` field). The record
    /// keeps the resolved instance itself: constant evaluation
    /// has already monomorphized the pointer, so no resolution
    /// is left for the collection pass.
    fn note_const_fn(&mut self, inst: &rustc_public::mir::mono::Instance) {
        let key = (inst.def.def_id(), format!("{:?}", inst.args()));
        if self.const_fn_mentions_seen.insert(key) {
            self.const_fn_mentions.push(inst.clone());
        }
    }

    /// Decode a reference whose provenance names a foreign
    /// static (an item of an `extern` block). A foreign static
    /// has no initializer to decode — asking the compiler for
    /// one ICEs (see `static_is_foreign`) — so the reference
    /// denotes the static's program-level storage with an
    /// uninitialized seed: address-only uses (the common case:
    /// `addr_of!`, raw-pointer casts, comparisons) observe the
    /// storage's identity, writes through a mutable reference
    /// land in the store like any other static's, and reads of
    /// never-written storage meet the interpreter's
    /// uninitialized-read error instead of an invented value.
    /// A pointee whose decoded form IS the contents (a `str`)
    /// cannot be represented this way and stays undecodable,
    /// as does a slice reference whose backing array cannot be
    /// located in the static's declared type (see
    /// `static_slice_path`).
    fn decode_foreign_static_ref(
        &mut self,
        def: &mir::mono::StaticDef,
        inner: &rustc_public::ty::Ty,
        alloc: &Allocation,
        offset: usize,
        target_off: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        let def_id = self.crate_def_id(def);
        // The item scan does not enumerate foreign statics, so
        // the declared type is recorded here, at the first decode
        // that names the static: the interpreter expands a
        // whole-static reference's seed to the recorded type's
        // extent, and a fat slice reference's seed in particular
        // must not be sized by the slice's own (possibly prefix)
        // length. The query is the one the projection
        // computations below already make of this definition --
        // a foreign static is a real item, unlike the anonymous
        // promoted statics the type query is unsafe for -- and it
        // is contained the way the item scan contains it.
        if !self.static_tys.contains_key(&def_id) {
            if let Ok(ty) =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| def.ty()))
            {
                self.note_static_ty(def, ty);
            }
        }
        match inner.kind() {
            TyKind::RigidTy(RigidTy::Slice(elem)) => {
                if matches!(elem.kind(), TyKind::RigidTy(RigidTy::Uint(SmirUintTy::U8))) {
                    return None;
                }
                let count = read_uint_at(alloc, offset + PTR_SIZE, PTR_SIZE)? as usize;
                // The projection names the slice's backing array
                // inside the static's declared type (the whole
                // array, an array field, an array element): an
                // empty projection for a field's array would make
                // the slice view index the static's whole value
                // instead of the field. A slice whose backing
                // array cannot be located stays undecodable.
                let projection =
                    self.static_slice_path(&def.ty(), target_off, &elem, count, depth)?;
                Some(sm::ConstValue::StaticRef {
                    def: def_id,
                    value: Box::new(sm::ConstValue::Uninit),
                    projection,
                    len: Some(count as u64),
                })
            }
            TyKind::RigidTy(RigidTy::Str) => None,
            _ => {
                // The projection is computed exactly as for a
                // non-foreign static, at offset zero too: a
                // pointer to the first element or field of an
                // aggregate static denotes that sub-value, not
                // the whole static (an empty projection would
                // make a write through the pointer replace the
                // complete storage). `static_offset_path` yields
                // the empty path itself when the pointee IS the
                // whole static.
                let projection =
                    self.static_offset_path(&def.ty(), target_off, inner, depth)?;
                Some(sm::ConstValue::StaticRef {
                    def: def_id,
                    value: Box::new(sm::ConstValue::Uninit),
                    projection,
                    len: None,
                })
            }
        }
    }

    fn decode_ref(
        &mut self,
        inner: &rustc_public::ty::Ty,
        alloc: &Allocation,
        offset: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        if depth > MAX_CONST_DEPTH {
            return None;
        }
        // A pointee type passing through a coroutine-closure
        // type cannot be asked its kind (see `guarded_ty_kind`)
        // and has no decodable value form; the caller keeps the
        // `ByRef` identity instead.
        if guarded_ty_kind(inner).is_none() {
            return None;
        }
        let prov = alloc
            .provenance
            .ptrs
            .iter()
            .find(|(o, _)| *o as usize == offset);
        let Some((_, prov)) = prov else {
            // No provenance at the pointer slot: a stored
            // zero word is a null pointer (Lane CK). A
            // nonzero word is a dangling address from
            // `without_provenance` (an empty `Vec`'s buffer
            // pointer, `NonNull::dangling`): only the address
            // is observable, as an address-only reference
            // whose reads are explicit errors (Lane CE).
            let addr = read_uint_at(alloc, offset, PTR_SIZE)?;
            if addr == 0 {
                return Some(sm::ConstValue::NullPtr);
            }
            // A fat pointer (slice / str pointee) carries a second
            // word: the length. A dangling fat pointer constant
            // (from_raw_parts in a const context) has no
            // pointee, but its address identity and length are
            // exactly observable, so it decodes as a FatPtr
            // rather than dropping the metadata word.
            if matches!(
                inner.kind(),
                TyKind::RigidTy(RigidTy::Slice(_) | RigidTy::Str)
            ) {
                let meta = read_uint_at(alloc, offset + PTR_SIZE, PTR_SIZE)?;
                return Some(sm::ConstValue::FatPtr {
                    addr: addr as u64,
                    meta: meta as u64,
                });
            }
            return Some(sm::ConstValue::IntPtr(addr as u64));
        };
        let target_off = read_uint_at(alloc, offset, PTR_SIZE)? as usize;
        let (target, static_def) = match mir::alloc::GlobalAlloc::from(prov.0) {
            mir::alloc::GlobalAlloc::Memory(target) => (target, None),
            // A reference to a static: the pointee is the static's
            // own storage. The decoded value is the static's
            // initializer allocation (the value the storage holds
            // before the program runs); the reference keeps the
            // static's identity so the interpreter can route reads
            // and writes through the program-level statics store
            // instead of a per-constant snapshot.
            mir::alloc::GlobalAlloc::Static(def) => {
                if static_is_foreign(&def) {
                    return self
                        .decode_foreign_static_ref(&def, inner, alloc, offset, target_off, depth);
                }
                let init = def.eval_initializer().ok()?;
                (init, Some(def))
            }
            // A pointer at a function item: the item is zero-sized,
            // so the pointer's value is the callable itself. This
            // covers both references to function items reached
            // through pointer-typed fields and pointers to functions
            // stored under a cast (`&(foo as *const u8)`).
            mir::alloc::GlobalAlloc::Function(inst) => {
                return self.fn_pointer_value(&inst);
            }
            mir::alloc::GlobalAlloc::TypeId { ty: tid_ty } => {
                // A `TypeId` value is a 128-bit hash of the type,
                // stored as two pointer-sized segments whose first
                // segment's provenance is this symbolic allocation.
                // The concrete hash function is unobservable - only
                // equality between TypeIds matters - so synthesize
                // the segments from a stable hash of the type. A
                // reference to the allocation denotes the TypeId
                // value itself; a pointer field pointing at it (the
                // value's own first segment) denotes the segment's
                // bits.
                let (lo, hi) = type_id_segments(&tid_ty);
                if matches!(inner.kind(), TyKind::RigidTy(RigidTy::Adt(..))) {
                    return Some(sm::ConstValue::Ref(Box::new(sm::ConstValue::Adt {
                        variant: 0,
                        fields: vec![sm::ConstValue::Tuple(vec![
                            sm::ConstValue::IntPtr(lo),
                            sm::ConstValue::IntPtr(hi),
                        ])],
                    })));
                }
                return Some(sm::ConstValue::IntPtr(lo));
            }
            // A vtable pointer: vtables are not materialised, but
            // the pointer's identity is observable (raw-pointer
            // equality / ordering, `ptr::eq` on trait objects), so
            // it decodes to a stable identity number for the
            // (concrete type, trait) pair.
            mir::alloc::GlobalAlloc::VTable(vty, trait_ref) => {
                return Some(sm::ConstValue::VTablePtr(vtable_id(&vty, &trait_ref)));
            }
        };
        match inner.kind() {
            TyKind::RigidTy(RigidTy::Str) => {
                // The string's bytes are exactly the fat pointer's
                // length word (the second word at the pointer
                // slot), as for the slice arm below: the backing
                // allocation may extend past the string.
                let len = read_uint_at(alloc, offset + PTR_SIZE, PTR_SIZE)? as usize;
                let raw = target.raw_bytes().ok()?;
                let bytes = raw.get(target_off..target_off + len)?;
                String::from_utf8(bytes.to_vec()).ok().map(sm::ConstValue::Str)
            }
            TyKind::RigidTy(RigidTy::Slice(elem)) => {
                let count = read_uint_at(alloc, offset + PTR_SIZE, PTR_SIZE)? as usize;
                // A slice reference into a static's storage keeps
                // the static's identity, as a thin static reference
                // does: a detached snapshot of the elements would
                // make writes through a mutable slice static
                // (`static mut S: &mut [T] = &mut [..]`) land in a
                // copy no later read observes.
                if let Some(def) = &static_def {
                    if let Some(c) = self.decode_static_slice_ref(
                        def, &elem, &target, target_off, count, depth,
                    ) {
                        return Some(c);
                    }
                }
                if matches!(elem.kind(), TyKind::RigidTy(RigidTy::Uint(SmirUintTy::U8))) {
                    let raw = target.raw_bytes().ok()?;
                    let bytes = raw.get(target_off..target_off + count)?;
                    return Some(sm::ConstValue::Bytes(bytes.to_vec()));
                }
                let stride = elem.layout().ok()?.shape().size.bytes();
                if stride == 0 && count > 0 { return None; }
                let mut elems = Vec::new();
                for i in 0..count {
                    elems.push(self.decode_const(
                        &elem,
                        &target,
                        target_off + i * stride,
                        depth + 1,
                    )?);
                }
                Some(sm::ConstValue::SliceVal(elems))
            }
            TyKind::RigidTy(RigidTy::Array(elem, _))
                if matches!(elem.kind(), TyKind::RigidTy(RigidTy::Uint(SmirUintTy::U8))) =>
            {
                // Byte arrays keep the established `Bytes` form (the
                // fmt pieces template is the canonical case).
                let raw = target.raw_bytes().ok()?;
                let bytes = raw.get(target_off..)?;
                Some(sm::ConstValue::Bytes(bytes.to_vec()))
            }
            _ => {
                // A type-erased pointee: the declared type is
                // zero-sized but the allocation region it denotes
                // is not — the pointer was stored through an
                // erased slot (a `NonNull<()>` field is the
                // canonical case), and its real type is named
                // only by the casts the guest later performs.
                // Preserve the allocation exactly so those casts
                // can materialise it (see `decode_erased_alloc`);
                // when the block cannot be built, the declared
                // type's decode below stands.
                if self.pointee_is_erased(inner, &target, target_off) {
                    if let Some(block) =
                        self.decode_erased_alloc(&target, target_off, depth + 1)
                    {
                        return Some(sm::ConstValue::Ref(Box::new(block)));
                    }
                }
                if let Some(def) = &static_def {
                    if let Some(c) =
                        self.decode_static_ref(def, inner, &target, target_off, depth)
                    {
                        return Some(c);
                    }
                }
                // A byte-sized pointee whose target allocation
                // extends past it: the pointer was cast in a
                // const context from a wider pointer into the
                // same allocation (a byte pointer into a static
                // byte table is the canonical case), so it
                // denotes the allocation from the offset on, not
                // the single byte — the same snapshot the byte
                // array arm above takes. A pointer whose
                // allocation is exactly the pointee keeps the
                // scalar decode.
                if matches!(
                    inner.kind(),
                    TyKind::RigidTy(
                        RigidTy::Uint(SmirUintTy::U8)
                            | RigidTy::Int(SmirIntTy::I8)
                            | RigidTy::Bool
                    )
                ) {
                    if let Ok(raw) = target.raw_bytes() {
                        if raw.len() > target_off + 1 {
                            return Some(sm::ConstValue::Ref(Box::new(
                                sm::ConstValue::Bytes(raw[target_off..].to_vec()),
                            )));
                        }
                    }
                }
                let pointee = self.decode_const(inner, &target, target_off, depth + 1)?;
                Some(sm::ConstValue::Ref(Box::new(pointee)))
            }
        }
    }

    /// Decode a reference whose provenance names the static
    /// `def`: a [`sm::ConstValue::StaticRef`] carrying the static's
    /// full initializer as the store seed and the projection from
    /// the static's base to the pointee. `None` when the projection
    /// cannot be determined (the caller then falls back to a
    /// snapshot of the pointee: reads stay correct, writes are
    /// rejected like any other constant-aggregate write).
    fn decode_static_ref(
        &mut self,
        def: &mir::mono::StaticDef,
        inner: &rustc_public::ty::Ty,
        alloc: &Allocation,
        target_off: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        let def_id = self.crate_def_id(def);
        // The static's declared type, recorded by the item scan
        // (see `note_static_ty`). A static with no recorded type —
        // a foreign static, or an anonymous promoted one — is in
        // practice reached through references that denote it
        // whole, whose pointee type is the static's own type, so a
        // pointee-typed seed stays consistent for it.
        let Some(static_ty) = self.static_tys.get(&def_id).cloned() else {
            let seed = self.decode_const(inner, alloc, target_off, depth + 1)?;
            return Some(sm::ConstValue::StaticRef {
                def: def_id,
                value: Box::new(seed),
                projection: Vec::new(),
                len: None,
            });
        };
        let projection = self.static_offset_path(&static_ty, target_off, inner, depth)?;
        let seed = self.decode_const(&static_ty, alloc, 0, depth + 1)?;
        Some(sm::ConstValue::StaticRef {
            def: def_id,
            value: Box::new(seed),
            projection,
            len: None,
        })
    }

    /// Decode a fat slice reference whose provenance names the
    /// static `def`, as a [`sm::ConstValue::StaticRef`] carrying
    /// the slice length: the evaluated value is a fat reference
    /// over the static reference, in the shape the runtime
    /// unsizing cast builds, so writes through it reach the
    /// static's storage. The pointee must be the static's whole
    /// array storage — the shape a promoted array in a static
    /// initializer takes (`static mut S: &mut [T] = &mut [..]`),
    /// and a reference to a whole array static unsized to a
    /// slice. Anything else (an interior subslice, an array
    /// field inside a larger static) returns `None`, and the
    /// caller falls back to a snapshot of the elements: reads
    /// stay correct, writes are rejected like any other
    /// constant-aggregate write.
    fn decode_static_slice_ref(
        &mut self,
        def: &mir::mono::StaticDef,
        elem: &rustc_public::ty::Ty,
        alloc: &Allocation,
        target_off: usize,
        count: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        if target_off != 0 {
            return None;
        }
        let def_id = self.crate_def_id(def);
        let seed = match self.static_tys.get(&def_id).cloned() {
            Some(static_ty) => {
                // The static's declared type must be the array
                // the slice covers whole: an unsizing of the
                // static itself.
                let TyKind::RigidTy(RigidTy::Array(arr_elem, arr_len)) =
                    static_ty.kind()
                else {
                    return None;
                };
                if !Self::ty_eq_ignoring_regions(&arr_elem, elem)
                    || arr_len.eval_target_usize().ok()? != count as u64
                {
                    return None;
                }
                self.decode_const(&static_ty, alloc, 0, depth + 1)?
            }
            None => {
                // An anonymous promoted static: the target
                // allocation is the array itself, decoded element
                // by element exactly as the array arm of
                // `decode_const` decodes an array value.
                let stride = elem.layout().ok()?.shape().size.bytes();
                if stride == 0 && count > 0 {
                    return None;
                }
                let mut elems = Vec::new();
                for i in 0..count {
                    elems.push(self.decode_const(elem, alloc, i * stride, depth + 1)?);
                }
                sm::ConstValue::Tuple(elems)
            }
        };
        Some(sm::ConstValue::StaticRef {
            def: def_id,
            value: Box::new(seed),
            projection: Vec::new(),
            len: Some(count as u64),
        })
    }

    /// Whether two types denote the same type up to their
    /// region (lifetime) arguments. A static's declared type
    /// carries the `'static` region where the pointee type of a
    /// pointer at the static carries the erased region, so plain
    /// equality misses references that denote the static whole
    /// (a `*mut &i64` at a `static mut R: &i64`): the reference
    /// then degrades to a snapshot of the initializer and writes
    /// through it are rejected as constant-aggregate writes.
    /// Regions never affect a value's storage shape, so the
    /// projection paths computed for one spelling hold for the
    /// other.
    fn ty_eq_ignoring_regions(a: &Ty, b: &Ty) -> bool {
        if a == b {
            return true;
        }
        match (a.kind(), b.kind()) {
            (
                TyKind::RigidTy(RigidTy::Ref(_, ea, ma)),
                TyKind::RigidTy(RigidTy::Ref(_, eb, mb)),
            ) => ma == mb && Self::ty_eq_ignoring_regions(&ea, &eb),
            (
                TyKind::RigidTy(RigidTy::RawPtr(ea, ma)),
                TyKind::RigidTy(RigidTy::RawPtr(eb, mb)),
            ) => ma == mb && Self::ty_eq_ignoring_regions(&ea, &eb),
            (
                TyKind::RigidTy(RigidTy::Slice(ea)),
                TyKind::RigidTy(RigidTy::Slice(eb)),
            ) => Self::ty_eq_ignoring_regions(&ea, &eb),
            (
                TyKind::RigidTy(RigidTy::Array(ea, la)),
                TyKind::RigidTy(RigidTy::Array(eb, lb)),
            ) => {
                Self::ty_eq_ignoring_regions(&ea, &eb)
                    && format!("{la:?}") == format!("{lb:?}")
            }
            (
                TyKind::RigidTy(RigidTy::Tuple(ea)),
                TyKind::RigidTy(RigidTy::Tuple(eb)),
            ) => {
                ea.len() == eb.len()
                    && ea
                        .iter()
                        .zip(eb.iter())
                        .all(|(x, y)| Self::ty_eq_ignoring_regions(x, y))
            }
            (
                TyKind::RigidTy(RigidTy::Adt(da, aa)),
                TyKind::RigidTy(RigidTy::Adt(db, ab)),
            ) => da == db && Self::generic_args_eq_ignoring_regions(&aa, &ab),
            (
                TyKind::RigidTy(RigidTy::FnDef(da, aa)),
                TyKind::RigidTy(RigidTy::FnDef(db, ab)),
            ) => da == db && Self::generic_args_eq_ignoring_regions(&aa, &ab),
            _ => false,
        }
    }

    /// The argument-wise companion of
    /// [`Self::ty_eq_ignoring_regions`]: lifetime arguments are
    /// ignored, type arguments recurse, and constant arguments
    /// compare by their rendered form (a mismatch keeps the
    /// caller's fallback, as before).
    fn generic_args_eq_ignoring_regions(a: &GenericArgs, b: &GenericArgs) -> bool {
        a.0.len() == b.0.len()
            && a.0.iter().zip(b.0.iter()).all(|(x, y)| match (x, y) {
                (GenericArgKind::Type(ta), GenericArgKind::Type(tb)) => {
                    Self::ty_eq_ignoring_regions(ta, tb)
                }
                (GenericArgKind::Lifetime(_), GenericArgKind::Lifetime(_)) => true,
                (GenericArgKind::Const(ca), GenericArgKind::Const(cb)) => {
                    format!("{ca:?}") == format!("{cb:?}")
                }
                _ => false,
            })
    }

    /// The projection path from the base of a value of type `ty`
    /// to the sub-value at byte offset `off`, when that sub-value
    /// has type `inner`: layout-guided descent through struct and
    /// union fields, tuple elements and array elements. Enum
    /// payloads are not descended (the active variant is a fact of
    /// the stored tag this layer does not read for projections);
    /// a whole-value reference is the empty path.
    fn static_offset_path(
        &mut self,
        ty: &rustc_public::ty::Ty,
        off: usize,
        inner: &rustc_public::ty::Ty,
        depth: usize,
    ) -> Option<Vec<sm::PlaceElem>> {
        if depth > MAX_CONST_DEPTH {
            return None;
        }
        if off == 0 && Self::ty_eq_ignoring_regions(ty, inner) {
            return Some(Vec::new());
        }
        match ty.kind() {
            TyKind::RigidTy(RigidTy::Adt(def, args)) => match def.kind() {
                AdtKind::Struct => {
                    let variant = def.variant(VariantIdx::to_val(0))?;
                    let offsets = arbitrary_offsets(ty)?;
                    let fields: Vec<(rustc_public::ty::Ty, usize)> = variant
                        .fields()
                        .iter()
                        .map(|f| f.ty_with_args(&args))
                        .zip(offsets)
                        .collect();
                    let (i, fty, mut rest) =
                        self.record_offset_path(&fields, off, inner, depth)?;
                    let mut path = vec![sm::PlaceElem::Field(i as u32, self.ty(&fty))];
                    path.append(&mut rest);
                    Some(path)
                }
                AdtKind::Union => {
                    let variant = def.variant(VariantIdx::to_val(0))?;
                    let fields: Vec<(rustc_public::ty::Ty, usize)> = variant
                        .fields()
                        .iter()
                        .map(|f| (f.ty_with_args(&args), 0))
                        .collect();
                    let (i, fty, mut rest) =
                        self.record_offset_path(&fields, off, inner, depth)?;
                    let mut path = vec![sm::PlaceElem::Field(i as u32, self.ty(&fty))];
                    path.append(&mut rest);
                    Some(path)
                }
                AdtKind::Enum => None,
            },
            TyKind::RigidTy(RigidTy::Tuple(elems)) => {
                let offsets = arbitrary_offsets(ty)?;
                let fields: Vec<(rustc_public::ty::Ty, usize)> =
                    elems.iter().cloned().zip(offsets).collect();
                let (i, fty, mut rest) = self.record_offset_path(&fields, off, inner, depth)?;
                let mut path = vec![sm::PlaceElem::Field(i as u32, self.ty(&fty))];
                path.append(&mut rest);
                Some(path)
            }
            TyKind::RigidTy(RigidTy::Array(elem, _)) => {
                let stride = elem.layout().ok()?.shape().size.bytes() as usize;
                if stride == 0 || off % stride != 0 {
                    return None;
                }
                let idx = off / stride;
                let count =
                    ty.layout().ok()?.shape().size.bytes() as usize / stride;
                if idx >= count {
                    return None;
                }
                let mut rest = self.static_offset_path(&elem, 0, inner, depth + 1)?;
                let mut path = vec![sm::PlaceElem::ConstantIndex {
                    offset: idx as u64,
                    min_length: count as u64,
                    from_end: false,
                }];
                path.append(&mut rest);
                Some(path)
            }
            _ => None,
        }
    }

    /// The projection path from the base of a value of type
    /// `ty` to the array at byte offset `off` whose element type
    /// is `elem` and whose length is `count` \u2014 the backing
    /// array of a fat slice reference into a static's storage.
    /// The descent mirrors `static_offset_path`, with the array
    /// itself as the leaf: a slice starting at an array's base
    /// and fitting inside it denotes that array (the empty path;
    /// the reference's length word bounds the view), and an
    /// array element that is itself the backing array is
    /// reached through a constant index.
    fn static_slice_path(
        &mut self,
        ty: &rustc_public::ty::Ty,
        off: usize,
        elem: &rustc_public::ty::Ty,
        count: usize,
        depth: usize,
    ) -> Option<Vec<sm::PlaceElem>> {
        if depth > MAX_CONST_DEPTH {
            return None;
        }
        match ty.kind() {
            TyKind::RigidTy(RigidTy::Array(arr_elem, arr_len)) => {
                if arr_elem == *elem && off == 0 {
                    // The slice starts at this array's base and
                    // denotes the array itself when it fits.
                    if count as u64 <= arr_len.eval_target_usize().ok()? {
                        return Some(Vec::new());
                    }
                    return None;
                }
                // An array element that is itself the backing
                // array, reached at its own offset.
                let stride = arr_elem.layout().ok()?.shape().size.bytes() as usize;
                if stride == 0 || off % stride != 0 {
                    return None;
                }
                let idx = off / stride;
                let total =
                    ty.layout().ok()?.shape().size.bytes() as usize / stride;
                if idx >= total {
                    return None;
                }
                let mut rest =
                    self.static_slice_path(&arr_elem, 0, elem, count, depth + 1)?;
                let mut path = vec![sm::PlaceElem::ConstantIndex {
                    offset: idx as u64,
                    min_length: total as u64,
                    from_end: false,
                }];
                path.append(&mut rest);
                Some(path)
            }
            TyKind::RigidTy(RigidTy::Adt(def, args)) => match def.kind() {
                AdtKind::Struct => {
                    let variant = def.variant(VariantIdx::to_val(0))?;
                    let offsets = arbitrary_offsets(ty)?;
                    let fields: Vec<(rustc_public::ty::Ty, usize)> = variant
                        .fields()
                        .iter()
                        .map(|f| f.ty_with_args(&args))
                        .zip(offsets)
                        .collect();
                    let (i, fty, mut rest) =
                        self.record_slice_path(&fields, off, elem, count, depth)?;
                    let mut path = vec![sm::PlaceElem::Field(i as u32, self.ty(&fty))];
                    path.append(&mut rest);
                    Some(path)
                }
                AdtKind::Union => {
                    let variant = def.variant(VariantIdx::to_val(0))?;
                    let fields: Vec<(rustc_public::ty::Ty, usize)> = variant
                        .fields()
                        .iter()
                        .map(|f| (f.ty_with_args(&args), 0))
                        .collect();
                    let (i, fty, mut rest) =
                        self.record_slice_path(&fields, off, elem, count, depth)?;
                    let mut path = vec![sm::PlaceElem::Field(i as u32, self.ty(&fty))];
                    path.append(&mut rest);
                    Some(path)
                }
                AdtKind::Enum => None,
            },
            TyKind::RigidTy(RigidTy::Tuple(elems)) => {
                let offsets = arbitrary_offsets(ty)?;
                let fields: Vec<(rustc_public::ty::Ty, usize)> =
                    elems.iter().cloned().zip(offsets).collect();
                let (i, fty, mut rest) =
                    self.record_slice_path(&fields, off, elem, count, depth)?;
                let mut path = vec![sm::PlaceElem::Field(i as u32, self.ty(&fty))];
                path.append(&mut rest);
                Some(path)
            }
            _ => None,
        }
    }

    /// Find the record field (type and byte offset within the
    /// record) that contains `off` and from which the backing
    /// array of a slice reference (element type `elem`, length
    /// `count`) is reachable, returning the field index, the
    /// field's type, and the remaining path from that field.
    /// Mirrors `record_offset_path`, with `static_slice_path`
    /// as the reachability test.
    fn record_slice_path(
        &mut self,
        fields: &[(rustc_public::ty::Ty, usize)],
        off: usize,
        elem: &rustc_public::ty::Ty,
        count: usize,
        depth: usize,
    ) -> Option<(usize, rustc_public::ty::Ty, Vec<sm::PlaceElem>)> {
        for (i, (fty, foff)) in fields.iter().enumerate() {
            if *foff > off {
                continue;
            }
            let size = fty.layout().ok()?.shape().size.bytes() as usize;
            // A field contains `off` when it starts at or before
            // it and ends after it; a zero-sized field contains
            // only its own offset.
            let contains = if size == 0 { *foff == off } else { off < foff + size };
            if !contains {
                continue;
            }
            if let Some(rest) =
                self.static_slice_path(fty, off - foff, elem, count, depth + 1)
            {
                return Some((i, fty.clone(), rest));
            }
        }
        None
    }

    /// Find the record field (type and byte offset within the
    /// record) that contains `off` and from which a pointee of
    /// type `inner` is reachable, returning the field index, the
    /// field's type, and the remaining path from that field.
    fn record_offset_path(
        &mut self,
        fields: &[(rustc_public::ty::Ty, usize)],
        off: usize,
        inner: &rustc_public::ty::Ty,
        depth: usize,
    ) -> Option<(usize, rustc_public::ty::Ty, Vec<sm::PlaceElem>)> {
        for (i, (fty, foff)) in fields.iter().enumerate() {
            if *foff > off {
                continue;
            }
            let size = fty.layout().ok()?.shape().size.bytes() as usize;
            // A field contains `off` when it starts at or before it
            // and ends after it; a zero-sized field contains only
            // its own offset.
            let contains = if size == 0 { *foff == off } else { off < foff + size };
            if !contains {
                continue;
            }
            if let Some(rest) = self.static_offset_path(fty, off - foff, inner, depth + 1) {
                return Some((i, fty.clone(), rest));
            }
        }
        None
    }

    fn place(&mut self, p: &mir::Place) -> sm::Place {
        let mut place = sm::Place::from_local(sm::Local(p.local as u32));
        for elem in &p.projection {
            place.projection.push(self.projection_elem(elem));
        }
        place
    }

    fn projection_elem(&mut self, elem: &mir::ProjectionElem) -> sm::PlaceElem {
        use mir::ProjectionElem as P;
        match elem {
            P::Deref => sm::PlaceElem::Deref,
            P::Field(idx, ty) => sm::PlaceElem::Field(*idx as u32, self.ty(ty)),
            P::Index(local) => sm::PlaceElem::Index(sm::Local(*local as u32)),
            P::ConstantIndex {
                offset,
                min_length,
                from_end,
            } => sm::PlaceElem::ConstantIndex {
                offset: *offset,
                min_length: *min_length,
                from_end: *from_end,
            },
            P::Subslice { from, to, from_end } => sm::PlaceElem::Subslice {
                from: *from,
                to: *to,
                from_end: *from_end,
            },
            P::Downcast(variant) => sm::PlaceElem::Downcast(None, variant.to_index() as u32),
            P::OpaqueCast(ty) => sm::PlaceElem::OpaqueCast(self.ty(ty)),
        }
    }

    fn type_args(&mut self, args: &GenericArgs) -> Vec<sm::Ty> {
        args.0
            .iter()
            .filter_map(|a| match a {
                GenericArgKind::Type(t) => Some(self.ty(t)),
                GenericArgKind::Const(_) | GenericArgKind::Lifetime(_) => None,
            })
            .collect()
    }

    pub fn ty(&mut self, ty: &rustc_public::ty::Ty) -> sm::Ty {
        let Some(kind) = guarded_ty_kind(ty) else {
            // A coroutine-closure type, whose kind the bridge
            // cannot hand over (see `guarded_ty_kind`): neither
            // its structure nor a layout can be asked. The value
            // it types is created by a coroutine-closure
            // aggregate, which carries the closure's definition
            // and arguments itself, so the body still converts
            // with the type recorded as the unit placeholder.
            return sm::Ty::unit();
        };
        let converted = self.ty_inner(ty);
        match &kind {
            TyKind::RigidTy(RigidTy::Adt(def, args)) => {
                self.record_layout(ty, def, args, &converted);
                self.record_debug_shape(def, &converted);
                self.record_instance_shape(ty, def, args, &converted);
            }
            TyKind::RigidTy(RigidTy::Closure(..))
            | TyKind::RigidTy(RigidTy::Coroutine(..))
            | TyKind::RigidTy(RigidTy::CoroutineClosure(..)) => {
                self.record_coroutine_layout(ty, &converted);
            }
            _ => {}
        }
        if let TyKind::RigidTy(RigidTy::Coroutine(..)) = &kind {
            self.record_coroutine_slots(ty, &converted);
        }
        if matches!(
            kind,
            TyKind::RigidTy(RigidTy::Closure(..))
                | TyKind::RigidTy(RigidTy::Coroutine(..))
        ) {
            self.record_env_shape(ty, &kind, &converted);
        }
        converted
    }

    fn ty_inner(&mut self, ty: &rustc_public::ty::Ty) -> sm::Ty {
        match ty.kind() {
            TyKind::RigidTy(r) => self.rigid_ty(ty, &r),
            TyKind::Alias(kind, alias) => {
                let k = match kind {
                    SmirAliasKind::Projection => sm::AliasKind::Projection,
                    SmirAliasKind::Opaque => sm::AliasKind::Opaque,
                    SmirAliasKind::Free => sm::AliasKind::Free,
                    SmirAliasKind::Inherent => sm::AliasKind::Inherent,
                };
                let d = self.crate_def_id(&alias.def_id);
                sm::Ty::Alias(k, d, self.type_args(&alias.args))
            }
            TyKind::Param(p) => sm::Ty::Param(p.index, p.name.clone()),
            TyKind::Bound(idx, _) => sm::Ty::Bound(idx as u32),
        }
    }

    /// Record the declared shape of an ADT under its converted
    /// `DefId`, once per ADT (see `sm::AdtShape`). The entry is
    /// inserted before the field types are converted so that a
    /// recursive ADT terminates: the recursive reference observes
    /// the placeholder (no variants), which no consumer mistakes
    /// for a complete shape.
    ///
    /// Shape recording is auxiliary to the conversion of the body
    /// in progress: converting a field type can hit constructs the
    /// body itself never uses (pattern types on an unrelated std
    /// ADT's fields, say), so the current body's error list is
    /// saved and restored around the field conversions — a body
    /// that genuinely uses such a type still reports the error
    /// through its own conversion path.
    fn record_shape(&mut self, def: &AdtDef, id: sm::AdtDefId) {
        if self.adt_shapes.contains_key(&id) {
            return;
        }
        let kind = match def.kind() {
            AdtKind::Struct => sm::AdtShapeKind::Struct,
            AdtKind::Enum => sm::AdtShapeKind::Enum,
            AdtKind::Union => sm::AdtShapeKind::Union,
        };
        self.adt_shapes.insert(
            id,
            sm::AdtShape {
                kind,
                variants: Vec::new(),
            },
        );
        let saved_errors = std::mem::take(&mut self.errors);
        let mut variants = Vec::new();
        for variant in def.variants() {
            let mut fields = Vec::new();
            for field in variant.fields() {
                let fty = field.ty();
                fields.push(self.ty(&fty));
            }
            variants.push(fields);
        }
        self.errors = saved_errors;
        self.adt_shapes.insert(id, sm::AdtShape { kind, variants });
    }

    /// Record the discriminant values of an enum ADT under its
    /// converted `DefId`, once per ADT. Structs and unions have no
    /// `Discriminant` semantics to record (a struct value is its
    /// single variant; unions are not modelled as enum values).
    fn record_discriminants(&mut self, def: &AdtDef, id: sm::AdtDefId) {
        if self.adt_discriminants.contains_key(&id) {
            return;
        }
        if def.kind() != AdtKind::Enum {
            return;
        }
        let values: Vec<i128> = (0..def.num_variants())
            .map(|i| {
                let d = def.discriminant_for_variant(VariantIdx::to_val(i));
                discr_value(&d).unwrap_or(d.val as i128)
            })
            .collect();
        self.adt_discriminants.insert(id, values);
    }

    /// Record the derived-`Debug` shape of one monomorphic ADT
    /// instance under its rendered instance name, once per distinct
    /// instance (see `sm::AdtDebugShapes`). Only instances with
    /// no remaining generic parameters get an entry:
    /// a parameter would render into a field tag that names no real
    /// type, and the instance's own tag would collide across
    /// instantiations.
    ///
    /// The field types are taken from the definition (in terms of
    /// its generic parameters, exactly like `record_shape`) and the
    /// instance's converted arguments are substituted into them
    /// here, in converted form. Substituting through the compiler
    /// instead (`FieldDef::ty_with_args`) is not an option: it can
    /// ask the compiler to normalize a projection it cannot
    /// normalize, which panics inside the compiler and poisons the
    /// query it runs in.
    ///
    /// Shape recording is auxiliary to the conversion in progress,
    /// exactly like `record_shape`: the current body's error list
    /// is saved and restored around the field conversions, and a
    /// placeholder entry is inserted before the fields are
    /// converted so a recursive ADT terminates. The unfolding is
    /// also depth-capped as a belt-and-braces bound (instances past
    /// the cap stay unsupported, like any other unrecorded
    /// instance).
    fn record_debug_shape(&mut self, def: &AdtDef, converted: &sm::Ty) {
        let kind = match def.kind() {
            AdtKind::Struct => sm::AdtShapeKind::Struct,
            AdtKind::Enum => sm::AdtShapeKind::Enum,
            AdtKind::Union => sm::AdtShapeKind::Union,
        };
        if ty_has_param(converted) || self.debug_shape_depth >= 16 {
            return;
        }
        let key = self.layout_key(converted);
        if self.adt_debug_shapes.contains_key(&key) {
            return;
        }
        let sm::Ty::Adt(_, instance_args) = converted else {
            return;
        };
        let instance_args = instance_args.clone();
        self.adt_debug_shapes.insert(
            key.clone(),
            sm::AdtDebugShape {
                kind,
                variants: Vec::new(),
            },
        );
        self.debug_shape_depth += 1;
        let saved_errors = std::mem::take(&mut self.errors);
        let mut variants = Vec::new();
        for variant in def.variants() {
            let mut fields = Vec::new();
            for field in variant.fields() {
                let fty = field.ty();
                let converted_field = self.ty(&fty);
                let substituted = substitute_params(&converted_field, &instance_args);
                let tag = self.layout_key(&substituted);
                // A tuple field's recorded name is its position
                // written out as a number; anything else is a
                // declared field name.
                let name = if field.name.bytes().all(|b| b.is_ascii_digit())
                    && !field.name.is_empty()
                {
                    None
                } else {
                    Some(field.name.clone())
                };
                fields.push(sm::AdtDebugField { name, ty: tag });
            }
            variants.push(sm::AdtDebugVariant {
                name: variant.name(),
                fields,
            });
        }
        self.errors = saved_errors;
        self.debug_shape_depth -= 1;
        self.adt_debug_shapes
            .insert(key, sm::AdtDebugShape { kind, variants });
    }

    /// Record the compiler-computed layout of one monomorphic ADT
    /// instance under its rendered instance name, once per distinct
    /// instance. The layout query is the compiler's own, so
    /// `repr(C)` / `repr(packed)` / `repr(align)` and enum tag
    /// placement are reflected exactly. Instances the query cannot
    /// answer (generic parameters among the arguments, unsized
    /// types, trait objects) are skipped: they stay explicitly
    /// unsupported in the interpreter.
    fn record_layout(
        &mut self,
        ty: &rustc_public::ty::Ty,
        def: &AdtDef,
        args: &rustc_public::ty::GenericArgs,
        converted: &sm::Ty,
    ) {
        if !self.layout_attempted_tys.insert(*ty) {
            return;
        }
        let key = self.layout_key(converted);
        if !self.layout_attempted.insert(key.clone()) {
            return;
        }
        self.record_niche(ty, def, args, &key);
        // The maybe-unsized screen was this query's only guard,
        // but it answers conservatively for ADTs whose fields
        // mention a type alias — `Cow<str>`'s `Owned` field is
        // `<str as ToOwned>::Owned` — screening out instances
        // the compiler lays out without doubt (the niche
        // recorder above already relies on exactly that). The
        // query below is contained, so attempting it for a
        // screened instance whose unsizedness is only suspected
        // costs at most the contained failure the unscreened
        // path already tolerates: the instance simply gets no
        // entry and stays unsupported.
        // One class must not be asked at all: an instance the
        // sizedness walk proves genuinely unsized — a slice,
        // `str` or trait object reachable through its fields,
        // at any depth, with the instance's arguments composed
        // in — is ill-formed for layout, and asking does not
        // fail inside the query. The layout computation records
        // a delayed bug instead, which the compiler flushes as
        // an abort when the run ends, outside every containment
        // the driver has. The niche recorder's screen already
        // judges by exactly this walk; the layout table follows
        // it. (An earlier form of this guard looked only at the
        // instance's own declared field types, which let
        // wrappers through — an enum over ill-formed structs,
        // a generic ADT instantiated with `str` — and aborted
        // the whole compilation for them.)
        // The layout query is not total: for some types that can
        // appear in converted bodies (ill-formed bounds, types
        // the layout computation itself rejects) it panics
        // inside the compiler. The table is auxiliary — a
        // failed query must never fail the conversion — so the
        // query is contained: the instance simply gets no entry
        // and stays unsupported. (The panic is silent under the
        // driver's panic-capture hook, which is installed while
        // the compiler runs.)
        if adt_sizedness(def, args, 0) == Sizedness::Unsized {
            return;
        }
        let layout = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout()));
        let Ok(Ok(layout)) = layout else {
            return;
        };
        let shape = layout.shape();
        if shape.is_unsized() {
            return;
        }
        self.adt_layouts.insert(
            key.clone(),
            (shape.size.bytes() as u64, shape.abi_align),
        );
        self.record_field_layouts(def, args, &shape, &key);
        self.record_tag_layout(def, &shape, &key);
    }

    /// Record the per-variant field byte layout of one ADT
    /// instance (see `sm::AdtFieldLayouts`) from the layout
    /// `record_layout` already queried. Variant field offsets
    /// come from the layout shape itself; each field's size
    /// comes from the field type's own contained layout query.
    /// An instance whose fields cannot all be sized gets no
    /// entry, exactly as a failed whole-layout query leaves
    /// `adt_layouts` without one. Unions get no entry: their
    /// fields share one offset and the interpreter reads them
    /// through its own reinterpretation machinery.
    fn record_field_layouts(
        &mut self,
        def: &AdtDef,
        args: &rustc_public::ty::GenericArgs,
        shape: &rustc_public::abi::LayoutShape,
        key: &str,
    ) {
        use rustc_public::abi::{FieldsShape, VariantsShape};
        if def.kind() == AdtKind::Union {
            return;
        }
        // (variant index, field offsets in declaration order)
        let variant_offsets: Vec<(usize, Vec<u64>)> = match &shape.variants {
            VariantsShape::Empty => return,
            VariantsShape::Single { index } => {
                let offsets = match &shape.fields {
                    FieldsShape::Arbitrary { offsets } => {
                        offsets.iter().map(|o| o.bytes() as u64).collect()
                    }
                    FieldsShape::Array { stride, count } => (0..*count)
                        .map(|i| (i as u64) * (stride.bytes() as u64))
                        .collect(),
                    FieldsShape::Primitive | FieldsShape::Union(_) => return,
                };
                vec![(index.to_index(), offsets)]
            }
            VariantsShape::Multiple { variants, .. } => variants
                .iter()
                .enumerate()
                .map(|(v, variant)| {
                    (
                        v,
                        variant
                            .offsets
                            .iter()
                            .map(|o| o.bytes() as u64)
                            .collect::<Vec<u64>>(),
                    )
                })
                .collect(),
        };
        let width = variant_offsets
            .iter()
            .map(|(v, _)| v + 1)
            .max()
            .unwrap_or(0);
        let mut out: Vec<Vec<(u64, u64)>> = vec![Vec::new(); width];
        for (v, offsets) in &variant_offsets {
            let Some(variant) = def.variant(rustc_public::ty::VariantIdx::to_val(*v))
            else {
                return;
            };
            let fields = variant.fields();
            if fields.len() != offsets.len() {
                return;
            }
            let mut row = Vec::with_capacity(offsets.len());
            for (field, &off) in fields.iter().zip(offsets.iter()) {
                let fty = field.ty_with_args(args);
                // A tuple field type's own storage layout is
                // recorded as well, through the conversion
                // hook that records every tuple type a body
                // mentions: the interpreter's scalar shape of
                // the field reads the tuple's fields at the
                // recorded offsets, and the instantiated
                // tuple type may occur in no body.
                if matches!(fty.kind(), TyKind::RigidTy(RigidTy::Tuple(_))) {
                    let _ = self.ty(&fty);
                }
                // A field type mentioning a coroutine is never
                // queried: computing such a type's layout can
                // drive the compiler into the coroutine's own
                // destructor-shim construction, which is not
                // total — an `async_drop_in_place` coroutine's
                // layout query aborts the whole compilation
                // with an internal error the containment cannot
                // catch. The interpreter's byte view loses
                // nothing: a coroutine value is not an aggregate
                // of fields it could descend into.
                if ty_mentions_coroutine(&fty) {
                    return;
                }
                let queried = std::panic::catch_unwind(
                    std::panic::AssertUnwindSafe(|| fty.layout()),
                );
                let Ok(Ok(flayout)) = queried else { return };
                row.push((off, flayout.shape().size.bytes() as u64));
            }
            out[*v] = row;
        }
        self.adt_field_layouts.insert(key.to_string(), out);
    }

    /// Record the direct discriminant tag layout of one
    /// enum instance (see `sm::AdtTagLayouts`) from the
    /// layout `record_layout` already queried. Only a
    /// multi-variant enum with the direct tag encoding
    /// stores a tag: its offset is the tag field's offset in
    /// the layout's top-level field list, and its width is
    /// the tag scalar's own (the stored tag can be narrower
    /// than the enum's discriminant type). Each variant's
    /// discriminant value is recorded as the bit pattern
    /// the tag stores. An instance whose discriminant
    /// values cannot be read gets no entry.
    fn record_tag_layout(
        &mut self,
        def: &AdtDef,
        shape: &rustc_public::abi::LayoutShape,
        key: &str,
    ) {
        use rustc_public::abi::{FieldsShape, TagEncoding, VariantsShape};
        if def.kind() != AdtKind::Enum {
            return;
        }
        let VariantsShape::Multiple {
            tag,
            tag_encoding: TagEncoding::Direct,
            tag_field,
            ..
        } = &shape.variants
        else {
            return;
        };
        let FieldsShape::Arbitrary { offsets } = &shape.fields else {
            return;
        };
        let Some(tag_off) = offsets.get(*tag_field) else {
            return;
        };
        let Some((tag_size, _)) = tag_scalar_size_signed(tag) else {
            return;
        };
        if tag_size == 0 || tag_size > 16 {
            return;
        }
        let mut discriminants = Vec::with_capacity(def.num_variants());
        for i in 0..def.num_variants() {
            let d = def
                .discriminant_for_variant(rustc_public::ty::VariantIdx::to_val(i));
            let Some(v) = discr_value(&d) else { return };
            discriminants.push(v as u128);
        }
        self.adt_tag_layouts.insert(
            key.to_string(),
            sm::EnumTag {
                offset: tag_off.bytes() as u64,
                size: tag_size as u64,
                discriminants,
            },
        );
    }

    /// Record the niche discriminant encoding of one
    /// monomorphic enum instance (see `sm::AdtNiches`); the
    /// caller dedups through `layout_attempted`, so this runs
    /// once per distinct instance.
    ///
    /// The recording runs its own contained layout query
    /// instead of sharing `record_layout`'s result, because
    /// the instances that matter most never reach that
    /// result: the maybe-unsized screen that guards the
    /// layout table answers conservatively for ADTs whose
    /// fields mention a type alias, which is exactly the
    /// `Option<NonZero<T>>` family this table exists for.
    /// The compiler's layout query has no such doubt and
    /// answers for those instances directly (and is
    /// memoized, so asking again for enums that pass the
    /// screen costs nothing measurable). Only multi-variant
    /// enums can carry a niche encoding at all.
    fn record_niche(
        &mut self,
        ty: &rustc_public::ty::Ty,
        def: &AdtDef,
        args: &GenericArgs,
        key: &str,
    ) {
        if def.kind() != AdtKind::Enum || def.num_variants() < 2 {
            return;
        }
        // Genuinely unsized instances are skipped without
        // querying: forcing a layout for one (the
        // trivial-bounds test cases, whose fields are slices
        // under assumed `Sized` bounds) trips the compiler's
        // own delayed bugs. Instances whose sizedness the
        // walk cannot prove — the alias-carrying
        // `Option<NonZero<T>>` family — go through the
        // contained query below, which is the only way to
        // learn their encoding; an uncomputable layout
        // simply records nothing.
        if adt_sizedness(def, args, 0) == Sizedness::Unsized {
            return;
        }
        let layout = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout()));
        let Ok(Ok(layout)) = layout else {
            return;
        };
        let shape = layout.shape();
        if shape.is_unsized() {
            return;
        }
        let VariantsShape::Multiple {
            tag,
            tag_encoding:
                TagEncoding::Niche {
                    untagged_variant,
                    niche_variants,
                    niche_start,
                },
            tag_field,
            ..
        } = &shape.variants
        else {
            return;
        };
        let offsets = match &shape.fields {
            FieldsShape::Arbitrary { offsets } => {
                offsets.iter().map(|o| o.bytes()).collect::<Vec<_>>()
            }
            _ => return,
        };
        let Some((tag_bytes, _)) = tag_scalar_size_signed(tag) else {
            return;
        };
        let Some(offset) = offsets.get(*tag_field) else {
            return;
        };
        self.adt_niches.insert(
            key.to_string(),
            sm::EnumNiche {
                untagged_variant: untagged_variant.to_index() as u32,
                niche_variants: (
                    niche_variants.start().to_index() as u32,
                    niche_variants.end().to_index() as u32,
                ),
                niche_start: *niche_start,
                tag_field: *tag_field as u32,
                tag_bits: (tag_bytes * 8) as u32,
                niche_offset: *offset as u64,
            },
        );
    }

    /// Record the field types of one monomorphic ADT instance
    /// whose generic shape mentions a type alias or a const
    /// generic array length (see `sm::AdtInstanceShapes`), once
    /// per distinct instance.
    ///
    /// The generic shape's field types are the definition's own,
    /// so a projection alias among them reaches the interpreter
    /// unresolved — the `NonZero<T>` field is declared as
    /// `<T as ZeroablePrimitive>::NonZeroInner`, and the
    /// interpreter has no trait solver — and an array length
    /// that is a const generic parameter (`[T; N]`, the field
    /// of every `core::simd` vector) reaches it unresolvable by
    /// substitution: the converter's instantiated types drop
    /// const arguments, so no substitution can ever produce the
    /// concrete length. The compiler resolves both:
    /// `FieldDef::ty_with_args` instantiates a field's declared
    /// type with the instance's arguments and normalizes it in a
    /// monomorphic environment, which is exactly the resolution
    /// the shape lacks. The query is not total: on a projection
    /// that mentions a generic parameter it fails *fatally* —
    /// an internal compiler error that no catch_unwind
    /// contains — because the codegen typing environment's
    /// parameter environment is empty. Instances are therefore
    /// screened before any query: `instance_mentions_param`
    /// rejects instances whose arguments mention a parameter,
    /// and a contained layout query gates the rest — an
    /// instance whose layout the compiler cannot compute has
    /// field projections that only a where-clause on the
    /// definition justifies, which normalize just as fatally.
    /// Unrecorded instances keep the generic shape's
    /// behaviour downstream.
    ///
    /// The recording is auxiliary, like `record_shape`'s: the
    /// body's error list is saved and restored around the field
    /// conversions, and any error growth abandons the entry — a
    /// body that genuinely uses such a type still reports through
    /// its own conversion path.
    fn record_instance_shape(
        &mut self,
        ty: &rustc_public::ty::Ty,
        def: &AdtDef,
        args: &GenericArgs,
        converted: &sm::Ty,
    ) {
        let sm::Ty::Adt(id, _) = converted else {
            return;
        };
        let key = self.layout_key(converted);
        if !self.instance_shape_attempted.insert(key.clone()) {
            return;
        }
        let Some(shape) = self.adt_shapes.get(id) else {
            return;
        };
        if !shape
            .variants
            .iter()
            .flatten()
            .any(|t| ty_mentions_alias(t) || ty_mentions_param_len(t))
        {
            return;
        }
        // `ty_with_args` normalizes in a codegen typing
        // environment whose parameter environment is empty: a
        // projection that mentions a generic parameter — a
        // type or const parameter, or a free region — cannot
        // normalize there, and the failure is *fatal* to the
        // whole compilation, not a catchable panic (the
        // compiler's `Failed to normalize ... maybe try
        // try_normalize_erasing_regions` ICE). Only
        // parameter-free instances are recorded; the rest
        // keep the generic shape's behaviour downstream.
        if instance_mentions_param(args) {
            return;
        }
        // Gate on the compiler's own layout query for the
        // instance (contained, like every layout query in
        // this driver; the query is memoized, so asking again
        // here costs nothing measurable). Computing a layout
        // normalizes the field types in the same codegen
        // environment, so an instance with no computable
        // layout has field projections `ty_with_args` would
        // fail fatally on — typically projections justified
        // only by a where-clause on the definition, whose
        // trait has no unconditional implementation for the
        // substituted arguments at all.
        let layout = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout()));
        if !matches!(layout, Ok(Ok(_))) {
            return;
        }
        let kind = shape.kind;
        let saved_errors = std::mem::take(&mut self.errors);
        let mut variants = Vec::new();
        let mut complete = true;
        'variants: for variant in def.variants() {
            let mut fields = Vec::new();
            for field in variant.fields() {
                let resolved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    field.ty_with_args(args)
                }));
                let Ok(resolved) = resolved else {
                    complete = false;
                    break 'variants;
                };
                fields.push(self.ty(&resolved));
            }
            variants.push(fields);
        }
        if complete && self.errors.is_empty() {
            self.adt_instance_shapes
                .insert(key, sm::AdtShape { kind, variants });
        }
        self.errors = saved_errors;
    }

    /// The `adt_layouts` key for a converted type: its `ty_tag`
    /// rendering against the layout name table, which is kept in
    /// sync with `def_names` (the interpreter renders call-site
    /// tags against the same names).
    /// Render the term of an existential projection predicate
    /// — the `101` of `Trait<N = 101>`, the type of
    /// `Trait<Item = u8>` — as it appears in the type's name.
    fn existential_term(&mut self, term: &rustc_public::ty::TermKind) -> String {
        use rustc_public::ty::TermKind;
        match term {
            TermKind::Type(t) => {
                let converted = self.ty(t);
                self.layout_key(&converted)
            }
            TermKind::Const(c) => match c.kind() {
                rustc_public::ty::TyConstKind::Value(ty, alloc) => {
                    match self.const_from_alloc(&ty, &alloc) {
                        sm::ConstValue::Scalar(sm::Scalar::Int { data, size }) => {
                            if matches!(
                                guarded_ty_kind(&ty),
                                Some(TyKind::RigidTy(RigidTy::Int(_)))
                            ) {
                                let shift = 128 - u32::from(size) * 8;
                                format!("{}", ((data << shift) as i128) >> shift)
                            } else {
                                format!("{data}")
                            }
                        }
                        sm::ConstValue::Bool(b) => b.to_string(),
                        sm::ConstValue::Char(ch) => format!("'{ch}'"),
                        sm::ConstValue::Str(s) => format!("{s:?}"),
                        _ => "<const>".to_string(),
                    }
                }
                rustc_public::ty::TyConstKind::Param(p) => p.name.clone(),
                rustc_public::ty::TyConstKind::Unevaluated(def, _) => def.name(),
                _ => "<const>".to_string(),
            },
        }
    }

    fn layout_key(&mut self, converted: &sm::Ty) -> String {
        while self.layout_names.len() < self.def_names.len() {
            let index = self.layout_names.len();
            self.layout_names.insert(
                sm::DefId::new(0, index as u32),
                self.def_names[index].clone(),
            );
        }
        sparagmos_interp::builtins::ty_tag(converted, &self.layout_names)
    }

    /// Record the compiler-computed layout of one monomorphic
    /// closure or coroutine instance under its type tag, once per
    /// distinct instance. A coroutine's layout is the compiler's
    /// own state-machine layout (which captures live across each
    /// suspension point), and a closure's is its capture aggregate;
    /// neither can be derived from the converted MIR, so both come
    /// from the same layout query the ADT path uses. The entry
    /// lands in `adt_layouts`, the table the
    /// `size_of_val` / `align_of_val` builtins consult by tag.
    /// Instances the query cannot answer (generic parameters among
    /// the arguments) are skipped under the same containment as
    /// the ADT path: the query is caught, and the instance stays
    /// explicitly unsupported.
    fn record_coroutine_layout(&mut self, ty: &rustc_public::ty::Ty, converted: &sm::Ty) {
        if !self.layout_attempted_tys.insert(*ty) {
            return;
        }
        let key = self.layout_key(converted);
        if !self.layout_attempted.insert(key.clone()) {
            return;
        }
        let layout = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout()));
        let Ok(Ok(layout)) = layout else {
            return;
        };
        let shape = layout.shape();
        if shape.is_unsized() {
            return;
        }
        self.adt_layouts.insert(
            key,
            (shape.size.bytes() as u64, shape.abi_align),
        );
    }

    /// Record the compiler-computed storage layout of one
    /// monomorphic tuple type under its rendered name, once
    /// per distinct type (see `sm::TupleLayouts`): the
    /// tuple's total storage size and each field's (offset,
    /// size) in declaration order, from the same layout
    /// query the ADT path uses. A tuple's storage order is
    /// the compiler's choice — fields may be reordered and
    /// padding inserted — so the interpreter's scalar shape
    /// of a tuple is built from this record, never from
    /// declaration order. Types the query cannot answer
    /// (generic parameters among the elements, unsized
    /// elements) are skipped under the same containment as
    /// the ADT path: the query is caught, and the type
    /// simply gets no entry. A type mentioning a coroutine
    /// is not asked at all, exactly as in
    /// `record_field_layouts`: computing such a type's
    /// layout can drive the compiler into the coroutine's
    /// own destructor-shim construction, which is not
    /// total. The mention is judged on the converted type,
    /// never on the compiler's: asking the compiler's type
    /// for its kind is itself the question that panics for
    /// coroutine-closure types (see `guarded_ty_kind`), and
    /// this recorder runs inside type conversion, where
    /// such a panic would take the whole body with it.
    fn record_tuple_layout(
        &mut self,
        ty: &rustc_public::ty::Ty,
        elem_tys: &[rustc_public::ty::Ty],
        converted: &sm::Ty,
    ) {
        if !self.layout_attempted_tys.insert(*ty) {
            return;
        }
        let key = self.layout_key(converted);
        if !self.layout_attempted.insert(key.clone()) {
            return;
        }
        if converted_mentions_coroutine(converted) {
            return;
        }
        let layout =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout()));
        let Ok(Ok(layout)) = layout else {
            return;
        };
        let shape = layout.shape();
        if shape.is_unsized() {
            return;
        }
        let FieldsShape::Arbitrary { offsets } = &shape.fields else {
            return;
        };
        if offsets.len() != elem_tys.len() {
            return;
        }
        let mut spans = Vec::with_capacity(elem_tys.len());
        for (elem_ty, off) in elem_tys.iter().zip(offsets.iter()) {
            let queried = std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| elem_ty.layout()),
            );
            let Ok(Ok(elayout)) = queried else { return };
            spans.push((
                off.bytes() as u64,
                elayout.shape().size.bytes() as u64,
            ));
        }
        self.tuple_layouts
            .insert(key, (shape.size.bytes() as u64, spans));
    }

    /// Record the storage-slot map of one monomorphic coroutine
    /// instance (see `sm::CoroutineSlots`) from the compiler's
    /// layout of the instance. The layout's per-variant field
    /// offsets are absolute, and a state field keeps one offset
    /// in every variant that contains it, so slots are assigned
    /// per distinct offset across all variants, after the upvar
    /// slots — except the zero-sized fields that share an offset
    /// with the field after them inside one variant, which are
    /// distinct cells and get private slots (see below).
    /// Instances whose layout the query cannot answer,
    /// and shapes that do not match the state-machine form (the
    /// outer fields are the upvars followed by the discriminant
    /// tag), get no entry; their projections keep the positional
    /// conversion.
    fn record_coroutine_slots(
        &mut self,
        ty: &rustc_public::ty::Ty,
        converted: &sm::Ty,
    ) {
        use rustc_public::abi::{FieldsShape, VariantsShape};
        if !self.coroutine_slots_attempted.insert(*ty) {
            return;
        }
        let sm::Ty::Coroutine(..) = converted else {
            return;
        };
        let layout = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout()));
        let Ok(Ok(layout)) = layout else {
            return;
        };
        let shape = layout.shape();
        let FieldsShape::Arbitrary { offsets } = &shape.fields else {
            return;
        };
        let VariantsShape::Multiple { tag_field, variants, .. } = &shape.variants
        else {
            return;
        };
        // The outer fields of a coroutine layout are the upvars
        // followed by the discriminant tag, whose field index the
        // layout itself reports: the upvar count is the tag's
        // index. (The converted generic arguments do not reliably
        // start with the upvars tuple - a generic coroutine's
        // arguments begin with the definition's own parameters.)
        let n_upvars = *tag_field;
        if offsets.len() != n_upvars + 1 {
            return;
        }
        // One slot per distinct offset, shared by every variant
        // that has a field there — with one exception. Inside a
        // single variant, several fields can sit at one offset:
        // a zero-sized field consumes no bytes, so the field
        // after it starts at the same offset. Those fields are
        // distinct storage cells (the zero-sized value and the
        // value after it are both live in that state), and
        // merging them would let each overwrite the other. In
        // offset order, every field of such a group but the last
        // is zero-sized (the next field starts at its offset, so
        // it spans no bytes), so the last field owns the offset's
        // shared slot and each earlier field gets a private slot
        // of its own, used by no other variant.
        let mut state_offsets: Vec<u64> = Vec::new();
        for variant in variants {
            for offset in &variant.offsets {
                let offset = offset.bytes() as u64;
                if !state_offsets.contains(&offset) {
                    state_offsets.push(offset);
                }
            }
        }
        let mut next_slot = n_upvars + state_offsets.len();
        let variants: Vec<Vec<u32>> = variants
            .iter()
            .map(|variant| {
                let offsets: Vec<u64> =
                    variant.offsets.iter().map(|o| o.bytes() as u64).collect();
                // The owner of each offset in this variant: the
                // last field, in (offset, field index) order,
                // among the fields at that offset.
                let mut owner: std::collections::HashMap<u64, usize> =
                    std::collections::HashMap::new();
                let mut order: Vec<usize> = (0..offsets.len()).collect();
                order.sort_by_key(|&i| (offsets[i], i));
                for &i in &order {
                    owner.insert(offsets[i], i);
                }
                offsets
                    .iter()
                    .enumerate()
                    .map(|(i, offset)| {
                        if owner[&offset] == i {
                            let pos = state_offsets
                                .iter()
                                .position(|o| o == offset)
                                .expect("offset was collected above");
                            (n_upvars + pos) as u32
                        } else {
                            let slot = next_slot as u32;
                            next_slot += 1;
                            slot
                        }
                    })
                    .collect()
            })
            .collect();
        let key = self.layout_key(converted);
        self.coroutine_layouts.insert(key, sm::CoroutineSlots { variants });
    }

    /// Record the debugger shape of one monomorphic closure or
    /// coroutine instance under its rendered instance name, once
    /// per distinct instance (see `sm::EnvDebugShapes`).
    ///
    /// The shape is read off the instance body's own debug info.
    /// The compiler records each captured variable there with the
    /// name the debuginfo gives the environment field - the
    /// captured place's name, `__`-joined along the place's
    /// projections - addressed by a projection from the
    /// environment base local: a `Field` step for a capture, a
    /// `Downcast` + `Field` pair for one of a coroutine's saved
    /// locals. A `Deref` step after the field step marks a
    /// by-reference capture, whose debuginfo name carries the
    /// `_ref__` prefix. The base local is identified by its
    /// declared type, which strips (through references and a
    /// `Pin` wrapper) to this instance's own definition.
    ///
    /// Recording is auxiliary to the conversion in progress,
    /// exactly like `record_debug_shape`: the current body's
    /// error list is saved and restored around it. An instance
    /// whose environment fields are not all named by the debug
    /// info gets no entry - it stays unrenderable in the
    /// debugger rather than rendering with invented names.
    fn record_env_shape(
        &mut self,
        ty: &rustc_public::ty::Ty,
        kind: &TyKind,
        converted: &sm::Ty,
    ) {
        if !self.env_shape_attempted.insert(*ty) {
            return;
        }
        if ty_has_param(converted) {
            return;
        }
        let (def_name, label, args, body) = match kind {
            TyKind::RigidTy(RigidTy::Closure(def, args)) => {
                let Some(ck) = closure_kind(args) else {
                    return;
                };
                let resolved = Instance::resolve_closure(*def, args, ck).ok();
                let key = resolved
                    .as_ref()
                    .map(|inst| inst.name())
                    .unwrap_or_else(|| def.name());
                let body = std::panic::catch_unwind(
                    std::panic::AssertUnwindSafe(|| {
                        instance_item_body(&key, resolved, || def.body())
                    }),
                );
                let Ok(Some(body)) = body else {
                    return;
                };
                (def.name(), "closure", args, body)
            }
            TyKind::RigidTy(RigidTy::Coroutine(def, args)) => {
                let resolved = Instance::resolve(FnDef(def.def_id()), args).ok();
                let key = resolved
                    .as_ref()
                    .map(|inst| inst.name())
                    .unwrap_or_else(|| def.name());
                let body = std::panic::catch_unwind(
                    std::panic::AssertUnwindSafe(|| {
                        instance_item_body(&key, resolved, || def.body())
                    }),
                );
                let Ok(Some(body)) = body else {
                    return;
                };
                (def.name(), "coroutine", args, body)
            }
            _ => return,
        };
        let Some(upvar_count) = env_upvar_count(args) else {
            return;
        };
        let key = self.layout_key(converted);
        let saved_errors = std::mem::take(&mut self.errors);
        let shape = self.env_shape_from_body(
            &body,
            &def_name,
            env_dwarf_name(&def_name, label),
            upvar_count,
            &key,
            label == "coroutine",
        );
        self.errors = saved_errors;
        if let Some(shape) = shape {
            self.env_debug_shapes.insert(key, shape);
        }
    }

    /// Assemble one environment shape from an instance body's
    /// debug info (see `record_env_shape`). `key` is the
    /// instance's rendered name, under which its coroutine
    /// storage-slot map (if any) is also recorded.
    fn env_shape_from_body(
        &mut self,
        body: &mir::Body,
        def_name: &str,
        dwarf_name: String,
        upvar_count: usize,
        key: &str,
        is_coroutine: bool,
    ) -> Option<sm::EnvDebugShape> {
        // Parse the debug entries into capture and saved-local
        // candidates, grouped by their base local.
        struct Candidate {
            base: usize,
            variant: Option<usize>,
            field: usize,
            by_ref: bool,
            name: String,
            fty: rustc_public::ty::Ty,
        }
        let mut candidates: Vec<Candidate> = Vec::new();
        for info in &body.var_debug_info {
            let mir::VarDebugInfoContents::Place(place) = &info.value
            else {
                continue;
            };
            let proj = &place.projection;
            let mut i = 0;
            while matches!(proj.get(i), Some(mir::ProjectionElem::Deref)) {
                i += 1;
            }
            let (variant, field, fty, rest) = match (proj.get(i), proj.get(i + 1)) {
                (
                    Some(mir::ProjectionElem::Downcast(v)),
                    Some(mir::ProjectionElem::Field(f, fty)),
                ) => (Some(v.to_index()), *f, *fty, i + 2),
                (Some(mir::ProjectionElem::Field(f, fty)), _) => {
                    (None, *f, *fty, i + 1)
                }
                _ => continue,
            };
            // A capture's place ends at the field step or at the
            // dereference of the reference the capture holds; a
            // longer tail denotes a sub-place of a variable the
            // environment does not hold a field for.
            if variant.is_none()
                && rest != proj.len()
                && !(rest + 1 == proj.len()
                    && matches!(proj.get(rest), Some(mir::ProjectionElem::Deref)))
            {
                continue;
            }
            let by_ref = variant.is_none()
                && matches!(proj.get(rest), Some(mir::ProjectionElem::Deref));
            candidates.push(Candidate {
                base: place.local,
                variant,
                field,
                by_ref,
                name: info.name.clone(),
                fty,
            });
        }
        // The environment base: the one candidate base whose
        // declared type denotes this instance's definition.
        let locals = body.locals();
        let mut base: Option<usize> = None;
        for cand in &candidates {
            if base == Some(cand.base) {
                continue;
            }
            let Some(decl) = locals.get(cand.base) else {
                continue;
            };
            if env_def_name(&decl.ty).as_deref() == Some(def_name) {
                if base.is_some() {
                    return None;
                }
                base = Some(cand.base);
            }
        }
        let base = base?;
        // Field descriptors, in the shape's own terms: the
        // debuginfo name (with a by-reference capture's `_ref__`
        // prefix) and the field type's rendered tag.
        let mut upvars: HashMap<usize, sm::AdtDebugField> = HashMap::new();
        let mut saved: HashMap<(usize, usize), sm::AdtDebugField> = HashMap::new();
        for cand in candidates.into_iter().filter(|c| c.base == base) {
            let converted_field = self.ty(&cand.fty);
            let tag = self.layout_key(&converted_field);
            let name = if cand.by_ref {
                format!("_ref__{}", cand.name)
            } else {
                cand.name
            };
            let field = sm::AdtDebugField {
                name: Some(name),
                ty: tag,
            };
            match cand.variant {
                None => {
                    upvars.entry(cand.field).or_insert(field);
                }
                Some(v) => {
                    saved.entry((v, cand.field)).or_insert(field);
                }
            }
        }
        let upvars: Option<Vec<sm::AdtDebugField>> =
            (0..upvar_count).map(|i| upvars.get(&i).cloned()).collect();
        let upvars = upvars?;
        if !is_coroutine {
            return Some(sm::EnvDebugShape {
                dwarf_name,
                upvars,
                variants: Vec::new(),
                slotted: false,
            });
        }
        if let Some(slots) = self.coroutine_layouts.get(key).cloned() {
            // Name every slot from the saved-local candidates:
            // a candidate's (variant, field) address names the
            // slot the layout map assigns to it.
            let mut by_slot: HashMap<u32, sm::AdtDebugField> = HashMap::new();
            for ((v, f), field) in &saved {
                let slot = slots.variants.get(*v)?.get(*f)?;
                by_slot.insert(*slot, field.clone());
            }
            let mut variants = Vec::with_capacity(slots.variants.len());
            for (v, vslots) in slots.variants.iter().enumerate() {
                let mut fields = Vec::with_capacity(vslots.len());
                for slot in vslots {
                    fields.push(by_slot.get(slot)?.clone());
                }
                variants.push(sm::EnvDebugVariant {
                    name: coroutine_variant_name(v),
                    fields,
                    slots: vslots.clone(),
                });
            }
            return Some(sm::EnvDebugShape {
                dwarf_name,
                upvars,
                variants,
                slotted: true,
            });
        }
        // No storage-slot map: the saved storage is positional,
        // one vector per state in the body's own variant order.
        let n_variants = saved
            .keys()
            .map(|(v, _)| v + 1)
            .max()
            .unwrap_or(0)
            .max(3);
        let mut variants = Vec::with_capacity(n_variants);
        for v in 0..n_variants {
            let count = saved.keys().filter(|(sv, _)| *sv == v).count();
            let mut fields = Vec::with_capacity(count);
            for f in 0..count {
                fields.push(saved.get(&(v, f))?.clone());
            }
            variants.push(sm::EnvDebugVariant {
                name: coroutine_variant_name(v),
                fields,
                slots: (0..count as u32).collect(),
            });
        }
        Some(sm::EnvDebugShape {
            dwarf_name,
            upvars,
            variants,
            slotted: false,
        })
    }

    fn rigid_ty(&mut self, ty: &rustc_public::ty::Ty, r: &RigidTy) -> sm::Ty {
        match r {
            RigidTy::Bool => sm::Ty::Bool,
            RigidTy::Char => sm::Ty::Char,
            RigidTy::Int(t) => sm::Ty::Int(int_ty(*t)),
            RigidTy::Uint(t) => sm::Ty::Uint(uint_ty(*t)),
            RigidTy::Float(t) => sm::Ty::Float(float_ty(*t)),
            RigidTy::Never => sm::Ty::Never,
            RigidTy::Str => sm::Ty::Str,
            RigidTy::Adt(def, args) => {
                let d = self.crate_def_id(def);
                self.record_discriminants(def, d);
                self.record_shape(def, d);
                sm::Ty::Adt(d, self.type_args(args))
            }
            RigidTy::Foreign(def) => {
                let d = self.crate_def_id(def);
                sm::Ty::Foreign(d)
            }
            RigidTy::Array(elem, len) => {
                let e = self.ty(elem);
                let l = match len.eval_target_usize() {
                    Ok(n) => sm::ArrayLen::Known(n),
                    Err(_) => {
                        self.err("Ty::Array (length not evaluable)");
                        sm::ArrayLen::Param("?".to_string())
                    }
                };
                sm::Ty::Array(Box::new(e), l)
            }
            RigidTy::Slice(elem) => sm::Ty::Slice(Box::new(self.ty(elem))),
            RigidTy::RawPtr(elem, m) => sm::Ty::RawPtr(Box::new(self.ty(elem)), mutability(*m)),
            RigidTy::Ref(region, elem, m) => sm::Ty::Ref(
                region_of(region),
                Box::new(self.ty(elem)),
                mutability(*m),
            ),
            RigidTy::FnDef(def, args) => {
                let d = self.fn_def_id(def, args);
                sm::Ty::FnDef(d, self.type_args(args))
            }
            RigidTy::FnPtr(poly) => {
                let sig = &poly.value;
                let mut ios = sig.inputs_and_output.clone();
                let output = match ios.pop() {
                    Some(o) => self.ty(&o),
                    None => {
                        self.err("Ty::FnPtr (empty signature)");
                        sm::Ty::unit()
                    }
                };
                let inputs = ios.iter().map(|t| self.ty(t)).collect();
                let mut s = sm::FnSig::new(inputs, output);
                s.c_variadic = sig.c_variadic;
                s.safety = match sig.safety {
                    mir::Safety::Safe => sm::Safety::Safe,
                    mir::Safety::Unsafe => sm::Safety::Unsafe,
                };
                s.abi = abi_of(&sig.abi);
                sm::Ty::FnPtr(Box::new(s))
            }
            RigidTy::Dynamic(bounds, _) => {
                // The bounds as name strings. A projection
                // predicate is an associated-item binding of
                // one of the trait bounds (`Trait<N = 101>`):
                // it folds into that trait's name as a generic
                // argument rather than standing as a bound of
                // its own. Each entry keeps the trait path it
                // was rendered from beside the rendered name:
                // folding a binding rewrites the rendered name
                // (`Trait` becomes `Trait<A=u8>`), so matching
                // a later binding of the same trait against the
                // rendered names would miss the entry and stand
                // the binding up as a separate bound.
                let mut names: Vec<(String, String)> = Vec::new();
                let mut bindings: Vec<(String, String, String)> = Vec::new();
                for b in bounds.iter() {
                    match &b.value {
                        rustc_public::ty::ExistentialPredicate::Trait(t) => {
                            let name = t.def_id.name();
                            names.push((name.clone(), name));
                        }
                        rustc_public::ty::ExistentialPredicate::AutoTrait(d) => {
                            let name = d.name();
                            names.push((name.clone(), name));
                        }
                        rustc_public::ty::ExistentialPredicate::Projection(p) => {
                            let item = p.def_id.name();
                            let term = self.existential_term(&p.term);
                            match item.rsplit_once("::") {
                                Some((trait_path, assoc)) => bindings.push((
                                    trait_path.to_string(),
                                    assoc.to_string(),
                                    term,
                                )),
                                None => names.push((item.clone(), item)),
                            }
                        }
                    }
                }
                for (trait_path, assoc, term) in bindings {
                    let binding = format!("{assoc}={term}");
                    match names
                        .iter_mut()
                        .find(|(base, _)| base == &trait_path)
                    {
                        Some((_, slot)) => {
                            if slot.ends_with('>') {
                                slot.insert_str(
                                    slot.len() - 1,
                                    &format!(", {binding}"),
                                );
                            } else {
                                *slot = format!("{slot}<{binding}>");
                            }
                        }
                        None => names.push((
                            trait_path.clone(),
                            format!("{trait_path}<{binding}>"),
                        )),
                    }
                }
                let names = names
                    .into_iter()
                    .map(|(_, rendered)| rendered)
                    .collect();
                sm::Ty::Dynamic(names, Box::new(sm::Ty::unit()), sm::Region::Erased)
            }
            RigidTy::Closure(def, args) => {
                let d = self.closure_def_id(def, args);
                sm::Ty::Closure(d, self.type_args(args))
            }
            RigidTy::Coroutine(def, args) => {
                let d = self.crate_def_id(def);
                sm::Ty::Coroutine(d, self.type_args(args))
            }
            RigidTy::CoroutineClosure(def, args) => {
                let d = self.crate_def_id(def);
                sm::Ty::Coroutine(d, self.type_args(args))
            }
            RigidTy::CoroutineWitness(def, args) => {
                let d = self.crate_def_id(def);
                sm::Ty::CoroutineWitness(d, self.type_args(args))
            }
            RigidTy::Tuple(elems) => {
                let converted =
                    sm::Ty::Tuple(elems.iter().map(|t| self.ty(t)).collect());
                self.record_tuple_layout(ty, elems, &converted);
                converted
            }
            RigidTy::Pat(inner, _) => {
                // A pattern type is a refinement of its base
                // type with the same representation, so it
                // converts as the base type everywhere: body
                // locals, constants, and ADT shape fields alike
                // (std's own `NonNull` carries a pattern-typed
                // pointer field, and shape consumers judge fields
                // by their base types). Construction in a valid
                // program guarantees the range; layout facts that
                // depend on it (niches) come from the rustc
                // layout tables, not from this type.
                self.ty(inner)
            }
        }
    }

    fn cast_kind(&mut self, kind: &mir::CastKind) -> sm::CastKind {
        use mir::CastKind as C;
        match kind {
            C::PointerExposeAddress => sm::CastKind::PointerExposeProvenance,
            C::PointerWithExposedProvenance => sm::CastKind::PointerWithExposedProvenance,
            C::PointerCoercion(c) => sm::CastKind::PointerCoercion(pointer_coercion(*c)),
            C::IntToInt => sm::CastKind::IntToInt,
            C::FloatToInt => sm::CastKind::FloatToInt,
            C::FloatToFloat => sm::CastKind::FloatToFloat,
            C::IntToFloat => sm::CastKind::IntToFloat,
            C::PtrToPtr => sm::CastKind::PtrToPtr,
            C::FnPtrToPtr => sm::CastKind::FnPtrToPtr,
            C::Transmute => sm::CastKind::Transmute,
            C::Subtype => sm::CastKind::Subtype,
            C::BoxDerefTransmute => sm::CastKind::BoxDerefTransmute,
        }
    }
}

fn bb(idx: mir::BasicBlockIdx) -> sm::BasicBlock {
    sm::BasicBlock(idx as u32)
}

fn switch_targets(t: &mir::SwitchTargets) -> (Vec<u128>, Vec<sm::BasicBlock>, sm::BasicBlock) {
    let mut values = Vec::new();
    let mut targets = Vec::new();
    for (v, target) in t.branches() {
        values.push(v);
        targets.push(bb(target));
    }
    (values, targets, bb(t.otherwise()))
}

fn mutability(m: mir::Mutability) -> sm::Mutability {
    match m {
        mir::Mutability::Mut => sm::Mutability::Mut,
        mir::Mutability::Not => sm::Mutability::Not,
    }
}

fn region_of(r: &rustc_public::ty::Region) -> sm::Region {
    match r.kind {
        rustc_public::ty::RegionKind::ReStatic => sm::Region::Static,
        _ => sm::Region::Erased,
    }
}

/// Byte size of a scalar element type, for materializing array/slice
/// constants. `None` for anything non-scalar (aggregates, pointers...).
/// Depth budget for constant decoding: type nesting levels and
/// pointer hops share it, so cyclic statics exhaust it and stay
/// unsupported instead of recursing forever.
const MAX_CONST_DEPTH: usize = 32;

/// Resource bound on array constants materialised element by
/// element in `decode_const`, mirroring `unique_const_value`'s
/// bound: an array length is guest-controlled input (`[(); usize::MAX]`
/// is a legal constant whose allocation is empty), and past this
/// many elements the value is not materialised element by element
/// — the decode fails and reference constants keep their `ByRef`
/// identity instead of the host process aborting on allocation.
/// Byte arrays are the one exception: they materialise as the
/// raw `Bytes` value, whose size is the allocation's own.
const MAX_CONST_ARRAY_ELEMS: u64 = 4096;

/// Pointer width in bytes on the target this driver runs for
/// (x86_64); Stable MIR provenance entries are pointer-sized.
const PTR_SIZE: usize = 8;

/// Byte size and signedness of a layout tag scalar: the stored
/// discriminant of a directly tagged enum, or the niche-carrying
/// scalar of a niche-encoded one. Pointers count as unsigned
/// pointer-sized integers; floats cannot be tags.
fn tag_scalar_size_signed(tag: &rustc_public::abi::Scalar) -> Option<(usize, bool)> {
    use rustc_public::abi::{IntegerLength, Primitive, Scalar};
    let prim = match tag {
        Scalar::Initialized { value, .. } | Scalar::Union { value } => value,
    };
    match prim {
        Primitive::Int { length, signed } => {
            let size = match length {
                IntegerLength::I8 => 1,
                IntegerLength::I16 => 2,
                IntegerLength::I32 => 4,
                IntegerLength::I64 => 8,
                IntegerLength::I128 => 16,
            };
            Some((size, *signed))
        }
        Primitive::Pointer(_) => Some((PTR_SIZE, false)),
        _ => None,
    }
}

/// The two pointer-sized segments of a `TypeId` value for `ty`:
/// a 128-bit FNV-1a hash (two lanes) of the type's debug form,
/// which is stable within a run - all the equality semantics of
/// `TypeId` need.
fn type_id_segments(ty: &rustc_public::ty::Ty) -> (u64, u64) {
    fn lane(bytes: &[u8], seed: u64) -> u64 {
        let mut h = seed;
        for b in bytes {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }
    let text = format!("{ty:?}");
    let bytes = text.as_bytes();
    (
        lane(bytes, 0xcbf2_9ce4_8422_2325),
        lane(bytes, 0x8422_2325_cbf2_9ce4),
    )
}

/// The identity number of a vtable for the (concrete type, trait
/// reference) pair: an FNV-1a hash of the pair's debug form, with
/// the top bit set to namespace it away from the small integer
/// addresses dangling pointers carry. Only equality / ordering
/// between vtable identities is observable, which a stable hash
/// preserves.
fn vtable_id(
    ty: &rustc_public::ty::Ty,
    trait_ref: &Option<rustc_public::ty::Binder<rustc_public::ty::ExistentialTraitRef>>,
) -> u64 {
    let text = format!("{ty:?}|{trait_ref:?}");
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h | (1 << 63)
}

/// Read an unsigned integer of `size` bytes at `offset` in `alloc`.
fn read_uint_at(alloc: &Allocation, offset: usize, size: usize) -> Option<u128> {
    if size == 0 || size > 16 {
        return None;
    }
    alloc.read_partial_uint(offset..offset + size).ok()
}

/// Sign-extend a zero-extended value of `size` bytes to 128 bits
/// (the storage convention of `Scalar::Int` for signed types).
/// The true value of a variant discriminant. Stable MIR reports
/// `Discr::val` as the discriminant's bit pattern at the width of
/// the discriminant type, zero-extended to u128: an `i8` enum's
/// `-1` arrives as 255 and an `isize` enum's `-1` as 2^64 - 1, so
/// the bare pattern cast to `i128` is a large positive number.
/// Sign-extending the pattern from the discriminant type's width
/// recovers the value; unsigned discriminants are already exact.
/// Whether a converted type still mentions a generic parameter
/// anywhere inside it (such a type is not one monomorphic
/// instance, so per-instance tables keyed by its rendered tag
/// would be wrong).
/// Substitute an ADT instance's converted generic arguments for
/// the definition's parameters in a converted field type. A
/// parameter with no corresponding argument (which the instance's
/// own monomorphic check should already have ruled out) is left
/// as it is.
fn substitute_params(ty: &sm::Ty, args: &[sm::Ty]) -> sm::Ty {
    match ty {
        sm::Ty::Param(index, _) => args
            .get(*index as usize)
            .cloned()
            .unwrap_or_else(|| ty.clone()),
        sm::Ty::Adt(def, inner) => sm::Ty::Adt(
            *def,
            inner.iter().map(|t| substitute_params(t, args)).collect(),
        ),
        sm::Ty::FnDef(def, inner) => sm::Ty::FnDef(
            *def,
            inner.iter().map(|t| substitute_params(t, args)).collect(),
        ),
        sm::Ty::Closure(def, inner) => sm::Ty::Closure(
            *def,
            inner.iter().map(|t| substitute_params(t, args)).collect(),
        ),
        sm::Ty::Coroutine(def, inner) => sm::Ty::Coroutine(
            *def,
            inner.iter().map(|t| substitute_params(t, args)).collect(),
        ),
        sm::Ty::CoroutineWitness(def, inner) => sm::Ty::CoroutineWitness(
            *def,
            inner.iter().map(|t| substitute_params(t, args)).collect(),
        ),
        sm::Ty::Alias(kind, def, inner) => sm::Ty::Alias(
            *kind,
            *def,
            inner.iter().map(|t| substitute_params(t, args)).collect(),
        ),
        sm::Ty::Array(elem, len) => {
            sm::Ty::Array(Box::new(substitute_params(elem, args)), len.clone())
        }
        sm::Ty::Pat(elem, pat) => {
            sm::Ty::Pat(Box::new(substitute_params(elem, args)), pat.clone())
        }
        sm::Ty::Slice(elem) => sm::Ty::Slice(Box::new(substitute_params(elem, args))),
        sm::Ty::RawPtr(elem, mutability) => {
            sm::Ty::RawPtr(Box::new(substitute_params(elem, args)), *mutability)
        }
        sm::Ty::Ref(region, inner, mutability) => sm::Ty::Ref(
            region.clone(),
            Box::new(substitute_params(inner, args)),
            *mutability,
        ),
        sm::Ty::FnPtr(sig) => sm::Ty::FnPtr(Box::new(sm::FnSig {
            inputs: sig
                .inputs
                .iter()
                .map(|t| substitute_params(t, args))
                .collect(),
            output: Box::new(substitute_params(&sig.output, args)),
            ..(**sig).clone()
        })),
        sm::Ty::Dynamic(traits, inner, region) => sm::Ty::Dynamic(
            traits.clone(),
            Box::new(substitute_params(inner, args)),
            region.clone(),
        ),
        sm::Ty::Tuple(items) => sm::Ty::Tuple(
            items.iter().map(|t| substitute_params(t, args)).collect(),
        ),
        _ => ty.clone(),
    }
}

fn ty_has_param(ty: &sm::Ty) -> bool {
    match ty {
        sm::Ty::Param(..) => true,
        sm::Ty::Adt(_, args)
        | sm::Ty::FnDef(_, args)
        | sm::Ty::Closure(_, args)
        | sm::Ty::Coroutine(_, args)
        | sm::Ty::CoroutineWitness(_, args)
        | sm::Ty::Alias(_, _, args) => args.iter().any(ty_has_param),
        sm::Ty::Array(elem, _)
        | sm::Ty::Pat(elem, _)
        | sm::Ty::Slice(elem)
        | sm::Ty::RawPtr(elem, _) => ty_has_param(elem),
        sm::Ty::Ref(_, inner, _) => ty_has_param(inner),
        sm::Ty::FnPtr(sig) => {
            sig.inputs.iter().any(ty_has_param) || ty_has_param(&sig.output)
        }
        sm::Ty::Dynamic(_, inner, _) => ty_has_param(inner),
        sm::Ty::Tuple(items) => items.iter().any(ty_has_param),
        _ => false,
    }
}

fn discr_value(d: &rustc_public::ty::Discr) -> Option<i128> {
    match d.ty.kind() {
        TyKind::RigidTy(RigidTy::Int(t)) => {
            let size = int_ty_size(&t)?;
            Some(sign_extend(d.val, size) as i128)
        }
        TyKind::RigidTy(RigidTy::Uint(_)) => Some(d.val as i128),
        _ => None,
    }
}

fn sign_extend(raw: u128, size: usize) -> u128 {
    let bits = size * 8;
    if bits >= 128 {
        return raw;
    }
    if (raw >> (bits - 1)) & 1 == 1 {
        raw | (u128::MAX << bits)
    } else {
        raw
    }
}

fn int_ty_size(t: &SmirIntTy) -> Option<usize> {
    Some(match t {
        SmirIntTy::I8 => 1,
        SmirIntTy::I16 => 2,
        SmirIntTy::I32 => 4,
        SmirIntTy::I64 | SmirIntTy::Isize => 8,
        SmirIntTy::I128 => 16,
    })
}

fn uint_ty_size(t: &SmirUintTy) -> Option<usize> {
    Some(match t {
        SmirUintTy::U8 => 1,
        SmirUintTy::U16 => 2,
        SmirUintTy::U32 => 4,
        SmirUintTy::U64 | SmirUintTy::Usize => 8,
        SmirUintTy::U128 => 16,
    })
}

/// Whether a type mentions a coroutine anywhere in its
/// structure (through ADT generic arguments, tuples, arrays,
/// slices, references and pointer wrappers). See the guard in
/// `Converter::record_field_layouts`: such a type's layout is
/// never queried for a field size.
/// Whether a converted type mentions a coroutine (or a
/// coroutine closure, which converts to the same form)
/// anywhere inside it. The converted-side counterpart of
/// `ty_mentions_coroutine`, for screens that run inside
/// type conversion itself: asking the compiler's type for
/// its kind is the question that panics for
/// coroutine-closure types (see `guarded_ty_kind`), so a
/// recorder hooked into conversion must judge the mention
/// from the converted form alone.
fn converted_mentions_coroutine(ty: &sm::Ty) -> bool {
    match ty {
        sm::Ty::Coroutine(..) | sm::Ty::CoroutineWitness(..) => true,
        sm::Ty::Adt(_, args) | sm::Ty::Closure(_, args) | sm::Ty::FnDef(_, args) => {
            args.iter().any(converted_mentions_coroutine)
        }
        sm::Ty::Alias(_, _, args) => args.iter().any(converted_mentions_coroutine),
        sm::Ty::Array(elem, _) | sm::Ty::Slice(elem) | sm::Ty::Pat(elem, _) => {
            converted_mentions_coroutine(elem)
        }
        sm::Ty::Ref(_, inner, _) | sm::Ty::RawPtr(inner, _) => {
            converted_mentions_coroutine(inner)
        }
        sm::Ty::Tuple(items) => items.iter().any(converted_mentions_coroutine),
        sm::Ty::FnPtr(sig) => {
            sig.inputs.iter().any(converted_mentions_coroutine)
                || converted_mentions_coroutine(&sig.output)
        }
        sm::Ty::Dynamic(_, inner, _) => converted_mentions_coroutine(inner),
        _ => false,
    }
}

fn ty_mentions_coroutine(ty: &rustc_public::ty::Ty) -> bool {
    use rustc_public::ty::{GenericArgKind, RigidTy, TyKind};
    fn args_mention(args: &rustc_public::ty::GenericArgs) -> bool {
        args.0.iter().any(|arg| match arg {
            GenericArgKind::Type(t) => ty_mentions_coroutine(t),
            _ => false,
        })
    }
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Coroutine(..)) => true,
        TyKind::RigidTy(RigidTy::Adt(_, args)) => args_mention(&args),
        TyKind::RigidTy(RigidTy::Closure(_, args)) => args_mention(&args),
        TyKind::RigidTy(RigidTy::Tuple(tys)) => tys.iter().any(ty_mentions_coroutine),
        TyKind::RigidTy(RigidTy::Array(elem, _))
        | TyKind::RigidTy(RigidTy::Slice(elem))
        | TyKind::RigidTy(RigidTy::Pat(elem, _)) => ty_mentions_coroutine(&elem),
        TyKind::RigidTy(RigidTy::Ref(_, inner, _))
        | TyKind::RigidTy(RigidTy::RawPtr(inner, _)) => ty_mentions_coroutine(&inner),
        TyKind::Alias(_, alias) => args_mention(&alias.args),
        _ => false,
    }
}

/// Field offsets of a type's layout, in definition order.
fn arbitrary_offsets(ty: &rustc_public::ty::Ty) -> Option<Vec<usize>> {
    let shape = ty.layout().ok()?.shape();
    match shape.fields {
        FieldsShape::Arbitrary { offsets } => {
            Some(offsets.iter().map(|o| o.bytes()).collect())
        }
        _ => None,
    }
}

fn scalar_elem_layout(ty: &rustc_public::ty::Ty) -> Option<usize> {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Bool) => Some(1),
        TyKind::RigidTy(RigidTy::Char) => Some(4),
        TyKind::RigidTy(RigidTy::Int(t)) => Some(match t {
            SmirIntTy::I8 => 1,
            SmirIntTy::I16 => 2,
            SmirIntTy::I32 => 4,
            SmirIntTy::I64 | SmirIntTy::Isize => 8,
            SmirIntTy::I128 => 16,
        }),
        TyKind::RigidTy(RigidTy::Uint(t)) => Some(match t {
            SmirUintTy::U8 => 1,
            SmirUintTy::U16 => 2,
            SmirUintTy::U32 => 4,
            SmirUintTy::U64 | SmirUintTy::Usize => 8,
            SmirUintTy::U128 => 16,
        }),
        TyKind::RigidTy(RigidTy::Float(SmirFloatTy::F32)) => Some(4),
        TyKind::RigidTy(RigidTy::Float(SmirFloatTy::F64)) => Some(8),
        _ => None,
    }
}

/// Short human-readable rendering of a Stable MIR type for error reports.
fn short_ty(ty: &rustc_public::ty::Ty) -> String {
    let s = format!("{ty:?}");
    if s.len() > 90 {
        format!("{}...", &s[..90])
    } else {
        s
    }
}


fn int_ty(t: SmirIntTy) -> sm::IntTy {
    match t {
        SmirIntTy::I8 => sm::IntTy::I8,
        SmirIntTy::I16 => sm::IntTy::I16,
        SmirIntTy::I32 => sm::IntTy::I32,
        SmirIntTy::I64 => sm::IntTy::I64,
        SmirIntTy::I128 => sm::IntTy::I128,
        SmirIntTy::Isize => sm::IntTy::Isize,
    }
}

fn uint_ty(t: SmirUintTy) -> sm::UintTy {
    match t {
        SmirUintTy::U8 => sm::UintTy::U8,
        SmirUintTy::U16 => sm::UintTy::U16,
        SmirUintTy::U32 => sm::UintTy::U32,
        SmirUintTy::U64 => sm::UintTy::U64,
        SmirUintTy::U128 => sm::UintTy::U128,
        SmirUintTy::Usize => sm::UintTy::Usize,
    }
}

fn float_ty(t: SmirFloatTy) -> sm::FloatTy {
    match t {
        SmirFloatTy::F16 => sm::FloatTy::F16,
        SmirFloatTy::F32 => sm::FloatTy::F32,
        SmirFloatTy::F64 => sm::FloatTy::F64,
        SmirFloatTy::F128 => sm::FloatTy::F128,
    }
}

fn binop(op: mir::BinOp) -> sm::BinOp {
    use mir::BinOp as B;
    match op {
        B::Add => sm::BinOp::Add,
        B::AddUnchecked => sm::BinOp::AddUnchecked,
        B::Sub => sm::BinOp::Sub,
        B::SubUnchecked => sm::BinOp::SubUnchecked,
        B::Mul => sm::BinOp::Mul,
        B::MulUnchecked => sm::BinOp::MulUnchecked,
        B::Div => sm::BinOp::Div,
        B::Rem => sm::BinOp::Rem,
        B::BitXor => sm::BinOp::BitXor,
        B::BitAnd => sm::BinOp::BitAnd,
        B::BitOr => sm::BinOp::BitOr,
        B::Shl => sm::BinOp::Shl,
        B::ShlUnchecked => sm::BinOp::ShlUnchecked,
        B::Shr => sm::BinOp::Shr,
        B::ShrUnchecked => sm::BinOp::ShrUnchecked,
        B::Eq => sm::BinOp::Eq,
        B::Lt => sm::BinOp::Lt,
        B::Le => sm::BinOp::Le,
        B::Ne => sm::BinOp::Ne,
        B::Ge => sm::BinOp::Ge,
        B::Gt => sm::BinOp::Gt,
        B::Cmp => sm::BinOp::Cmp,
        B::Offset => sm::BinOp::Offset,
    }
}

fn unop(op: mir::UnOp) -> sm::UnOp {
    match op {
        mir::UnOp::Not => sm::UnOp::Not,
        mir::UnOp::Neg => sm::UnOp::Neg,
        mir::UnOp::PtrMetadata => sm::UnOp::PtrMetadata,
    }
}

fn pointer_coercion(c: mir::PointerCoercion) -> sm::PointerCoercion {
    use mir::PointerCoercion as P;
    match c {
        P::ReifyFnPointer(_) => sm::PointerCoercion::ReifyFnPointer,
        P::UnsafeFnPointer => sm::PointerCoercion::UnsafeFnPointer,
        P::ClosureFnPointer(_) => sm::PointerCoercion::ClosureFnPointer,
        P::Unsize => sm::PointerCoercion::Unsize,
        P::MutToConstPointer => sm::PointerCoercion::MutToConstPointer,
        P::ArrayToPointer => sm::PointerCoercion::ArrayToPointer,
    }
}

fn borrow_kind(b: &mir::BorrowKind) -> sm::BorrowKind {
    use mir::BorrowKind as B;
    match b {
        B::Shared => sm::BorrowKind::Shared,
        B::Fake(kind) => sm::BorrowKind::Fake(match kind {
            mir::FakeBorrowKind::Deep => sm::FakeBorrowKind::Deep,
            mir::FakeBorrowKind::Shallow => sm::FakeBorrowKind::Shallow,
        }),
        B::Mut { kind } => sm::BorrowKind::Mut {
            allow_two_phase_borrow: matches!(kind, mir::MutBorrowKind::TwoPhaseBorrow),
        },
    }
}

/// The width in bits of an inline-asm operand type: a plain
/// integer type, `bool` (8) or `char` (32). 0 for anything else
/// (floats, aggregates, pointers), which keeps the terminator
/// outside the evaluable class. `isize`/`usize` are 64 on the
/// x86_64 target Sparagmos aims at.
fn asm_type_width(ty: &Ty) -> u32 {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Int(t)) => match t {
            SmirIntTy::I8 => 8,
            SmirIntTy::I16 => 16,
            SmirIntTy::I32 => 32,
            SmirIntTy::I64 | SmirIntTy::Isize => 64,
            SmirIntTy::I128 => 128,
        },
        TyKind::RigidTy(RigidTy::Uint(t)) => match t {
            SmirUintTy::U8 => 8,
            SmirUintTy::U16 => 16,
            SmirUintTy::U32 => 32,
            SmirUintTy::U64 | SmirUintTy::Usize => 64,
            SmirUintTy::U128 => 128,
        },
        TyKind::RigidTy(RigidTy::Bool) => 8,
        TyKind::RigidTy(RigidTy::Char) => 32,
        _ => 0,
    }
}

/// The explicit register name of an asm operand's raw
/// representation (`reg: Reg(X86(ax))`), when the operand is
/// bound to one architectural register rather than a register
/// class.
fn asm_raw_reg_name(raw: &str) -> Option<String> {
    let rest = raw.split("reg: Reg(X86(").nth(1)?;
    let name = rest.split(')').next()?;
    Some(name.to_string())
}

/// The item name of a `Sym` asm operand's raw representation
/// (`SymFn { value: foo }`, `SymStatic { value: BAR }`).
fn asm_raw_sym_name(raw: &str) -> Option<String> {
    let rest = raw.split("value: ").nth(1)?;
    Some(rest.trim_end_matches('}').trim().to_string())
}

/// The target index of a `Label` asm operand's raw
/// representation (`Label { target_index: 1 }`): the index into
/// the terminator's internal successor list, where the
/// fallthrough destination (when the terminator has one) is
/// index 0 and the label blocks follow in operand order.
fn asm_raw_label_index(raw: &str) -> Option<usize> {
    let rest = raw.split("target_index: ").nth(1)?;
    rest.chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

/// The register scalar of a plain integer / bool / char / float
/// type, for an asm call signature. Anything else (aggregates,
/// pointers, 128-bit integers) is not register-passed in this
/// model and returns `None`.
fn asm_scalar(ty: &rustc_public::ty::Ty) -> Option<sm::AsmScalar> {
    use rustc_public::ty::{FloatTy, IntTy, UintTy};
    let TyKind::RigidTy(rigid) = guarded_ty_kind(ty)? else {
        return None;
    };
    Some(match rigid {
        RigidTy::Int(it) => sm::AsmScalar::Int {
            width: match it {
                IntTy::Isize => 64,
                IntTy::I8 => 8,
                IntTy::I16 => 16,
                IntTy::I32 => 32,
                IntTy::I64 => 64,
                IntTy::I128 => return None,
            },
            signed: true,
        },
        RigidTy::Uint(it) => sm::AsmScalar::Int {
            width: match it {
                UintTy::Usize => 64,
                UintTy::U8 => 8,
                UintTy::U16 => 16,
                UintTy::U32 => 32,
                UintTy::U64 => 64,
                UintTy::U128 => return None,
            },
            signed: false,
        },
        RigidTy::Bool => sm::AsmScalar::Bool,
        RigidTy::Char => sm::AsmScalar::Int {
            width: 32,
            signed: false,
        },
        RigidTy::Float(ft) => sm::AsmScalar::Float {
            width: match ft {
                FloatTy::F32 => 32,
                FloatTy::F64 => 64,
                _ => return None,
            },
        },
        _ => return None,
    })
}

/// Whether the local function whose full definition name is
/// `def_name` (crate segment included) is the very function an
/// asm `sym` operand references by `referenced`. The operand's
/// raw form names the resolved item by its complete path —
/// crate-relative for an item of the local crate (`local::foo`
/// for the definition `test::local::foo`), crate-qualified for
/// any other crate's item (`aux::foo`) — so a local definition
/// matches only when the referenced path is its whole
/// crate-relative path, or its full name. Matching only the
/// last segment would let an unrelated local function stand in
/// for an external target the operand actually names.
fn asm_ref_matches_def(def_name: &str, referenced: &str) -> bool {
    if def_name == referenced {
        return true;
    }
    let Some(prefix) = def_name.strip_suffix(referenced) else {
        return false;
    };
    // The elided prefix must be exactly the crate segment:
    // "<crate>::", with no further "::" inside it.
    prefix.len() > 2 && prefix.ends_with("::") && !prefix[..prefix.len() - 2].contains("::")
}

/// The declared ABI of a function type (the type of a function
/// definition or of a resolved instance), for recording with
/// the function's converted body. `None` when the type carries
/// no function signature.
fn fn_ty_abi(ty: &rustc_public::ty::Ty) -> Option<sm::Abi> {
    let sig = guarded_ty_kind(ty)?.fn_sig()?.skip_binder();
    Some(abi_of(&sig.abi))
}

/// Run an ABI query under the panic-capture discipline of
/// [`guarded_ty_kind`]. Asking a still-generic instance for its
/// type panics with "needs further instantiation" — a known
/// gap, not a crash: the instance's ABI is not needed, since a
/// monomorphized instance of the same function registers (and
/// is recorded) separately. That panic's hook record is
/// consumed, so no outer containment mistakes it for a
/// conversion panic; any other panic resumes.
fn guarded_abi_query(query: impl FnOnce() -> Option<sm::Abi>) -> Option<sm::Abi> {
    clear_captured_panic();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(query)) {
        Ok(abi) => abi,
        Err(payload) => match take_captured_panic() {
            Some(record) if record.contains("needs further instantiation") => None,
            other => {
                if let Some(record) = other {
                    record_captured_panic(record);
                }
                std::panic::resume_unwind(payload)
            }
        },
    }
}

/// The declared ABI of a resolved instance's function type,
/// for recording with the instance's converted body (see
/// [`guarded_abi_query`] for the containment).
pub fn instance_fn_abi(inst: &Instance) -> Option<sm::Abi> {
    guarded_abi_query(|| fn_ty_abi(&inst.ty()))
}

/// The declared ABI of a function definition's type, for
/// recording with the definition's converted body (see
/// [`guarded_abi_query`] for the containment).
pub fn def_fn_abi(def: &FnDef) -> Option<sm::Abi> {
    guarded_abi_query(|| fn_ty_abi(&def.ty()))
}

/// The ABI family of an asm call signature, from the signature's
/// rustc ABI. Only the two register conventions the interpreter
/// assembles are recognized; `C` and `System` are SysV64-shaped
/// on x86_64.
fn asm_abi_family(abi: &rustc_public::ty::Abi) -> Option<sm::AsmAbi> {
    let dbg = format!("{abi:?}");
    if dbg.starts_with("Win64") {
        Some(sm::AsmAbi::Win64)
    } else if dbg.starts_with("SysV64")
        || dbg.starts_with("System")
        || dbg == "C"
        || dbg.starts_with("C {")
    {
        Some(sm::AsmAbi::SysV64)
    } else {
        None
    }
}

/// Structurally restore the label blocks of a body's asm-goto
/// terminator. Stable MIR does not publish them: an InlineAsm
/// terminator's only reported successor is its fallthrough
/// destination. The label blocks are exactly the roots of the
/// region the body's control-flow graph cannot reach under that
/// reading — a label block's own continuation blocks may be
/// unreachable too, but only the label blocks themselves have
/// no predecessor inside the unreachable region. Restoration
/// applies only when the body has exactly one label-carrying
/// InlineAsm terminator (a single carrier) and the root count
/// matches the label count; for a `noreturn` terminator the
/// reported destination is itself the first label block, so one
/// fewer root is expected. Anything else returns `None`, and
/// the terminator's labels stay unrestored (which keeps the
/// terminator outside the evaluable class).
fn restore_asm_label_blocks(
    body: &mir::Body,
    label_count: usize,
    noreturn: bool,
) -> Option<Vec<sm::BasicBlock>> {
    let blocks = &body.blocks;
    let mut carriers = 0usize;
    for block in blocks.iter() {
        if let mir::TerminatorKind::InlineAsm { operands, .. } = &block.terminator.kind {
            if operands.iter().any(|op| op.raw_rpr.starts_with("Label")) {
                carriers += 1;
            }
        }
    }
    if carriers != 1 {
        return None;
    }
    let n = blocks.len();
    let succ: Vec<Vec<usize>> = blocks
        .iter()
        .map(|block| block.terminator.successors())
        .collect();
    let mut reachable = vec![false; n];
    reachable[0] = true;
    let mut stack = vec![0usize];
    while let Some(i) = stack.pop() {
        for &j in &succ[i] {
            if !reachable[j] {
                reachable[j] = true;
                stack.push(j);
            }
        }
    }
    let mut has_unreachable_pred = vec![false; n];
    for i in 0..n {
        if reachable[i] {
            continue;
        }
        for &j in &succ[i] {
            if !reachable[j] {
                has_unreachable_pred[j] = true;
            }
        }
    }
    let roots: Vec<usize> = (0..n)
        .filter(|&i| !reachable[i] && !has_unreachable_pred[i])
        .collect();
    let expected = if noreturn {
        label_count.checked_sub(1)?
    } else {
        label_count
    };
    if roots.len() != expected {
        return None;
    }
    Some(roots.iter().map(|&i| sm::BasicBlock(i as u32)).collect())
}

/// Read an integer literal out of a `Const` asm operand's raw
/// representation (`Const { value: const 5_usize }`): the bits
/// (masked to the type width) and the width. `None` for any other
/// shape (an unevaluated or non-integer constant).
fn asm_const_literal(raw: &str) -> Option<(u64, u32)> {
    let inner = raw
        .strip_prefix("Const { value: const ")?
        .strip_suffix(" }")?;
    let (num, width) = match inner.rsplit_once('_') {
        Some((num, suffix)) => {
            let width = match suffix {
                "i8" | "u8" => 8,
                "i16" | "u16" => 16,
                "i32" | "u32" => 32,
                "i64" | "u64" | "isize" | "usize" => 64,
                "i128" | "u128" => 128,
                _ => return None,
            };
            (num, width)
        }
        None => return None,
    };
    let value: i128 = num.parse().ok()?;
    let bits = if width >= 128 {
        value as u64
    } else {
        (value as u64) & ((1u64 << width) - 1)
    };
    Some((bits, width))
}

fn fake_read_cause(c: &mir::FakeReadCause) -> sm::FakeReadCause {
    use mir::FakeReadCause as F;
    match c {
        F::ForMatchGuard => sm::FakeReadCause::ForMatchedGuard,
        F::ForMatchedPlace(_) => sm::FakeReadCause::ForVarBinding,
        F::ForGuardBinding => sm::FakeReadCause::ForGuardBinding,
        F::ForLet(_) => sm::FakeReadCause::ForLet,
        F::ForIndex => sm::FakeReadCause::ForIndexOf,
    }
}

fn variance_of(v: mir::Variance) -> sm::Variance {
    match v {
        mir::Variance::Covariant => sm::Variance::Covariant,
        mir::Variance::Invariant => sm::Variance::Invariant,
        mir::Variance::Contravariant => sm::Variance::Contravariant,
        mir::Variance::Bivariant => sm::Variance::Bivariant,
    }
}

fn coroutine_kind(k: &mir::CoroutineKind) -> sm::CoroutineKind {
    use mir::CoroutineDesugaring as D;
    match k {
        mir::CoroutineKind::Desugared(D::Async, _) => sm::CoroutineKind::Async,
        mir::CoroutineKind::Desugared(D::Gen, _) => sm::CoroutineKind::Gen,
        mir::CoroutineKind::Desugared(D::AsyncGen, _) => sm::CoroutineKind::Coroutine,
        mir::CoroutineKind::Coroutine(_) => sm::CoroutineKind::Coroutine,
    }
}

fn abi_of(abi: &rustc_public::ty::Abi) -> sm::Abi {
    use rustc_public::ty::Abi as A;
    match abi {
        A::Rust => sm::Abi::Rust,
        A::C { .. } => sm::Abi::C,
        A::System { .. } => sm::Abi::System,
        A::Stdcall { .. } => sm::Abi::Stdcall,
        A::Fastcall { .. } => sm::Abi::Fastcall,
        A::RustCall => sm::Abi::RustCall,
        other => sm::Abi::Other(format!("{other:?}")),
    }
}

/// The closure kind (`Fn` / `FnMut` / `FnOnce`) recorded in a
/// closure type's generic arguments. rustc appends three synthetic
/// arguments to a closure's parent arguments — the kind, the
/// signature as a function-pointer type, and the tupled captures —
/// and encodes the kind itself as an integer type: `i8` for `Fn`,
/// `i16` for `FnMut`, `i32` for `FnOnce` (this is rustc's
/// `Ty::from_closure_kind`; the encoding is part of the Stable MIR
/// boundary this crate pins to Rust 1.100.0).
pub fn closure_kind(args: &GenericArgs) -> Option<ClosureKind> {
    let kind_arg = args.0.iter().rev().nth(2)?;
    let GenericArgKind::Type(ty) = kind_arg else {
        return None;
    };
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Int(SmirIntTy::I8)) => Some(ClosureKind::Fn),
        TyKind::RigidTy(RigidTy::Int(SmirIntTy::I16)) => Some(ClosureKind::FnMut),
        TyKind::RigidTy(RigidTy::Int(SmirIntTy::I32)) => Some(ClosureKind::FnOnce),
        _ => None,
    }
}

/// Whether a demanded closure's generic arguments are a
/// coroutine closure's (an async closure's). A plain closure
/// packs its tail as `[kind, sig, captures]`; a coroutine
/// closure packs `[kind, sig, captures, captures-by-ref]` —
/// an integer kind marker type, a fn-pointer signature, a
/// tuple of upvars, and a fn-pointer producing the
/// by-reference captures. The check is purely structural,
/// over the tail shapes; a plain closure's arguments never
/// match it (their last argument is the upvars tuple, not a
/// fn-pointer).
pub fn coroutine_closure_args(args: &GenericArgs) -> bool {
    let mut tail = args.0.iter().rev();
    let (Some(by_ref), Some(upvars), Some(sig), Some(kind)) =
        (tail.next(), tail.next(), tail.next(), tail.next())
    else {
        return false;
    };
    let is_fn_ptr = |arg: &GenericArgKind| {
        matches!(
            arg,
            GenericArgKind::Type(t)
                if matches!(
                    guarded_ty_kind(t),
                    Some(TyKind::RigidTy(RigidTy::FnPtr(..)))
                )
        )
    };
    let is_tuple = |arg: &GenericArgKind| {
        matches!(
            arg,
            GenericArgKind::Type(t)
                if matches!(
                    guarded_ty_kind(t),
                    Some(TyKind::RigidTy(RigidTy::Tuple(_)))
                )
        )
    };
    let is_kind = |arg: &GenericArgKind| {
        matches!(
            arg,
            GenericArgKind::Type(t)
                if matches!(
                    guarded_ty_kind(t),
                    Some(TyKind::RigidTy(RigidTy::Int(
                        SmirIntTy::I8 | SmirIntTy::I16 | SmirIntTy::I32
                    )))
                )
        )
    };
    is_fn_ptr(by_ref) && is_tuple(upvars) && is_fn_ptr(sig) && is_kind(kind)
}

/// Whether any generic argument still mentions a type or const
/// parameter, i.e. the item is not fully monomorphic.
pub fn args_mention_params(args: &GenericArgs) -> bool {
    args.0.iter().any(|arg| match arg {
        GenericArgKind::Type(ty) => ty_mentions_param(ty),
        GenericArgKind::Const(konst) => const_mentions_param(konst),
        GenericArgKind::Lifetime(_) => false,
    })
}

fn const_mentions_param(konst: &rustc_public::ty::TyConst) -> bool {
    match konst.kind() {
        TyConstKind::Param(_) => true,
        TyConstKind::Unevaluated(_, args) => args_mention_params(args),
        TyConstKind::Value(ty, _) | TyConstKind::ZSTValue(ty) => ty_mentions_param(ty),
        _ => false,
    }
}

fn ty_mentions_param(ty: &Ty) -> bool {
    if let Some(cached) =
        TY_MENTIONS_PARAM_CACHE.with(|cache| cache.borrow().get(ty).copied())
    {
        return cached;
    }
    let result = ty_mentions_param_walk(ty);
    TY_MENTIONS_PARAM_CACHE.with(|cache| {
        cache.borrow_mut().insert(*ty, result);
    });
    result
}

fn ty_mentions_param_walk(ty: &Ty) -> bool {
    // A type passing through a coroutine-closure type cannot
    // be asked its kind (see `guarded_ty_kind`); counting it
    // as mentioning a parameter is the conservative answer
    // for every caller (they all treat `true` as "not fully
    // monomorphic here" and defer or fall back).
    let Some(kind) = guarded_ty_kind(ty) else {
        return true;
    };
    match kind {
        TyKind::Param(_) => true,
        TyKind::Alias(_, alias) => args_mention_params(&alias.args),
        TyKind::RigidTy(rigid) => match rigid {
            RigidTy::Ref(_, inner, _) | RigidTy::RawPtr(inner, _) | RigidTy::Slice(inner) => {
                ty_mentions_param(&inner)
            }
            RigidTy::Array(inner, len) => ty_mentions_param(&inner) || const_mentions_param(&len),
            RigidTy::Pat(inner, _) => ty_mentions_param(&inner),
            RigidTy::Tuple(elems) => elems.iter().any(ty_mentions_param),
            RigidTy::FnPtr(sig) => sig.value.inputs_and_output.iter().any(ty_mentions_param),
            RigidTy::Adt(_, args)
            | RigidTy::FnDef(_, args)
            | RigidTy::Closure(_, args)
            | RigidTy::Coroutine(_, args)
            | RigidTy::CoroutineClosure(_, args)
            | RigidTy::CoroutineWitness(_, args) => args_mention_params(&args),
            RigidTy::Dynamic(bounds, _) => bounds.iter().any(|b| match &b.value {
                ExistentialPredicate::Trait(t) => args_mention_params(&t.generic_args),
                ExistentialPredicate::Projection(p) => args_mention_params(&p.generic_args),
                ExistentialPredicate::AutoTrait(_) => false,
            }),
            _ => false,
        },
        _ => false,
    }
}

