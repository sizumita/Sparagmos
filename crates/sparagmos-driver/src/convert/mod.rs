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
//! - `BasicBlock` does not expose `is_cleanup`, and `UnwindAction`
//!   states no `Terminate` reason; both are recovered structurally
//!   from the body's `Cleanup` edges after conversion (see
//!   `mark_cleanup_region` in convert/body.rs).
//! - Constants are allocation-based (`ConstantKind::Allocated` /
//!   `TyConstKind::Value`); scalar values are read out of the allocation
//!   bytes using the constant's own type, so sizes are exact.

#![allow(unused_imports)]

use std::collections::{HashMap, HashSet};

use rustc_public::abi::{FieldsShape, TagEncoding, VariantsShape};
use rustc_public::mir;
use rustc_public::mir::mono::{Instance, InstanceKind};
use rustc_public::ty::{
    AdtDef, AdtKind, AliasKind as SmirAliasKind, Allocation, AssocContainer, AssocKind, Binder,
    ClosureKind, ConstantKind, ExistentialPredicate, ExistentialProjection, ExistentialTraitRef,
    FloatTy as SmirFloatTy, FnDef, FnSig, GenericArgKind, GenericArgs, ImplDef,
    IntTy as SmirIntTy, Prov, Region, RegionKind, RigidTy, TermKind, Ty, TyConst,
    TyConstKind, TyKind, UintTy as SmirUintTy, VariantIdx,
};
use rustc_public_bridge::IndexedVal;
use rustc_public::{CrateDef, CrateDefType};

use sparagmos_mir as sm;

mod guard;
mod display_names;
mod statics;
mod subst;
mod screen;
mod panic;
mod body;
mod stmt;
mod rvalue;
mod const_decode;
mod ref_decode;
mod types;
mod helpers;

// The glob re-exports below keep every pre-split `crate::convert::*`
// path working. Some of them re-export nothing at `pub(crate)`
// visibility, and some are not referenced through this facade; both
// are intentional, so the unused-imports lint is allowed here.
pub(crate) use self::guard::*;
pub(crate) use self::display_names::*;
pub(crate) use self::statics::*;
pub(crate) use self::subst::*;
pub(crate) use self::screen::*;
pub(crate) use self::panic::*;
pub(crate) use self::body::*;
pub(crate) use self::stmt::*;
pub(crate) use self::rvalue::*;
pub(crate) use self::const_decode::*;
pub(crate) use self::ref_decode::*;
pub(crate) use self::types::*;
pub(crate) use self::helpers::*;

pub use self::guard::{
    compiler_tcx, drop_glue_lang_item_defined, materialization_emitted, materialization_lint,
    resolution_emitted, resolve_closure_guarded, resolve_instance_guarded, set_compiler_tcx,
    with_body_materialization,
};
pub use self::display_names::{drain_fn_display_names, set_debug_session};
pub use self::helpers::{
    args_mention_params, closure_kind, coroutine_closure_args, def_fn_abi, instance_fn_abi,
};
pub use self::panic::{
    captured_panic_is_mir_validator_fail, captured_panic_is_normalize_overflow,
    clear_captured_panic, generated_simd_impl_method, guarded_ty_kind,
    is_coroutine_closure_panic, is_instance_const_panic,
    is_unsafe_binder_panic, peek_captured_panic, record_captured_panic, stable_mir_panic_reason,
    take_captured_panic, CONST_EXPR_REASON, COROUTINE_CLOSURE_TYPE_REASON, INSTANCE_CONST_REASON,
    BODY_FETCH_NORMALIZE_OVERFLOW, BODY_FETCH_SIMD_VALIDATOR, TAIL_CALL_REASON,
    UNSAFE_BINDER_REASON,
};
pub use self::screen::{
    unnormalizable_projection_in_body, unnormalizable_projection_in_instance,
    UNNORMALIZABLE_PROJECTION_REASON,
};

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
    /// Sparagmos ids of the definitions the compiler classifies
    /// as compiler intrinsics (`InstanceKind::Intrinsic`), as
    /// recorded by [`Converter::note_intrinsic`]. Handed to the
    /// interpreter with the converted program: its name-only
    /// intrinsic fallback may serve a body-less callee only
    /// when the callee's definition is in this set, so an
    /// ordinary function that merely shares an intrinsic's name
    /// keeps its recorded conversion failure instead of being
    /// silently served by the builtin.
    pub intrinsic_defs: std::collections::HashSet<sm::DefId>,
    /// Sparagmos ids of the definitions of coroutine closures
    /// (async closures) whose types or values conversion has
    /// met, as recorded by [`Converter::note_async_closure`].
    /// Handed to the interpreter with the converted program:
    /// an async closure's own body is not convertible (it
    /// mentions the coroutine-closure type), so a call through
    /// the `Fn` trait family on one of these definitions is
    /// served by constructing the closure's coroutine, exactly
    /// as an `AsyncFn` call does.
    pub async_closure_defs: std::collections::HashSet<sm::DefId>,
    /// Vtable identities minted while decoding constants: for
    /// each `ConstValue::VTablePtr` identity number, the
    /// (concrete type, principal trait) pair it denotes, in the
    /// form the interpreter's `DynRef` values carry. Handed to
    /// the interpreter with the converted program: a vtable
    /// pointer that reaches it as a bare address (recovered
    /// from a metadata aggregate, or read back from a stored
    /// header word) resolves through this table.
    pub vtable_identities: HashMap<u64, (sm::Ty, Vec<String>)>,
    /// Vtable trait references named by the constants decoded
    /// so far, in the order their identities were first
    /// recorded: a vtable formed inside a constant (a
    /// `DynMetadata` in a trait-info table is the common shape)
    /// is a vtable formation no cast in any scanned body
    /// names, so the dyn demand pass takes its trait
    /// references from here as well (see
    /// `demand_dyn_instances` in the driver). Drained by that
    /// pass.
    pub vtable_demand: Vec<rustc_public::ty::TraitRef>,
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
    /// Runtime layout descriptors of the dynamically sized
    /// ADT instances converted so far, keyed like
    /// `adt_layouts` (see `sm::AdtDstLayout`). Handed to the
    /// interpreter with the converted program: its
    /// `size_of_val` / `align_of_val` builtins compose a DST
    /// value's layout from the descriptor and the tail's
    /// dynamic layout.
    pub adt_dst_layouts: sm::AdtDstLayouts,
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
    /// Expansion budget remaining for the constant currently
    /// being decoded (see `MAX_CONST_TOTAL_ELEMS`): reset at the
    /// root of every decoded constant (`const_from_alloc`, and
    /// at the root of every `decode_const` / `decode_ref` tree
    /// reached from one) and spent
    /// once per `decode_const` call, so one constant's total
    /// materialised size is bounded however its element counts
    /// nest. Arms that copy a byte payload spend the copied
    /// length as well: counting calls alone would let a repeated
    /// reference to one shared byte block copy that block once
    /// per element, unbounded by the call count.
    const_budget: usize,
    /// Set when a `spend_const_budget` charge does not fit.
    /// Read by the erased-pointer fallback in `decode_ref`,
    /// which clears it before attempting an erased decode and
    /// checks it afterwards: a block that failed to build
    /// because the budget ran out must fail the whole constant
    /// rather than fall back to the declared type's decode
    /// (see `decode_ref`).
    const_budget_failed: bool,
    /// Set when a `range_all_uninit` scan runs out of scan
    /// budget before its outcome is decided. Read by the same
    /// `decode_ref` fallbacks that read `const_budget_failed`,
    /// which clear it before attempting a decode and check it
    /// afterwards: a decode that failed after the scan budget
    /// denied it the `Uninit` shortcut may have failed for
    /// exactly that reason, so it must fail the whole constant
    /// rather than fall back to a different value (see
    /// `decode_ref`).
    const_scan_failed: bool,
    /// Scan budget remaining for the constant currently being
    /// decoded (see `MAX_CONST_SCAN_BYTES`): reset wherever
    /// `const_budget` resets, and spent by `range_all_uninit`
    /// once per byte examined, so the uninitialised-range scan
    /// at the head of `decode_const` -- the one decode step
    /// whose cost tracks a range's byte size rather than the
    /// materialised value -- stays bounded when one root
    /// constant re-decodes a large partially initialised
    /// pointee once per element. Exhaustion does not by itself
    /// fail a decode: the scan reports "undecided" and the
    /// structural decode proceeds, so a value the structural
    /// decode can still build is never lost. What exhaustion
    /// does record is `const_scan_failed`: when the structural
    /// decode then fails anyway, the `decode_ref` fallbacks
    /// cannot tell that failure apart from one the lost
    /// shortcut caused, so they keep the constant undecoded
    /// rather than substitute a snapshot (see `decode_ref`).
    const_scan_budget: usize,
    /// Set when a `spend_work_budget` charge does not fit.
    /// Read by the same `decode_ref` fallbacks that read
    /// `const_budget_failed`, and by the two materialisation
    /// points (`resolve_alloc`, `static_initializer`) as a stop
    /// signal: once the cumulative materialisation work for one
    /// root constant is over budget, later materialisations in
    /// the same decode tree are refused without being performed,
    /// so an eviction-thrashing constant costs the budget, not
    /// the budget times its element count. Unlike the other two
    /// exhaustion flags it is also cleared wherever the budgets
    /// refill: it is read outside the clear/check pairs, so a
    /// stale set flag would refuse materialisations for unrelated
    /// later constants (whether a constant decodes must not
    /// depend on the constants visited before it).
    const_work_failed: bool,
    /// Materialisation-work budget remaining for the constant
    /// currently being decoded (see `MAX_CONST_WORK_BYTES`):
    /// reset wherever `const_budget` resets, and spent by
    /// `resolve_alloc` / `static_initializer` once per backing
    /// allocation actually materialised, at its byte size.
    /// Cache hits spend nothing: the caches already bound how
    /// much is *retained*, this budget bounds how much work is
    /// *done*, which the retained-bytes cap cannot -- two
    /// allocations that together exceed the cap evict each
    /// other on every access, so each is re-materialised once
    /// per element while almost nothing is ever retained.
    const_work_budget: usize,
    /// Materialised memory allocations by provenance id,
    /// shared within one root constant's decode tree and
    /// cleared wherever `const_budget` resets. Resolving a
    /// pointer's target through rustc_public materialises the
    /// whole allocation; a table of references into one shared
    /// allocation reaches that resolution once per element, so
    /// without the cache the materialisation alone costs the
    /// backing size per element -- the same unbounded shape
    /// the expansion budget exists to bound, one layer down.
    /// Byte-capped (see `MAX_CACHED_ALLOC_BYTES`): past the
    /// cap the cache is cleared rather than grown, so one
    /// constant cannot pin every allocation it merely touches.
    alloc_cache: std::collections::HashMap<usize, std::rc::Rc<Allocation>>,
    /// Initialiser allocations of statics, keyed by the static's
    /// Sparagmos def id: the static arm of `decode_ref` otherwise
    /// re-evaluates (and thereby re-materialises) the initialiser for
    /// every reference into the static, so a table of many short
    /// slices into one large static pays the backing size per
    /// element. Shares the byte cap and the root-clear discipline of
    /// `alloc_cache` (see `clear_alloc_caches`).
    static_inits: std::collections::HashMap<sm::DefId, std::rc::Rc<Allocation>>,
    /// Total bytes held in `alloc_cache` and `static_inits`.
    alloc_cache_bytes: usize,
    /// Memoised decodes of allocated constants, keyed by the
    /// constant's type and its allocation's exact contents
    /// (bytes and provenance targets). `const_from_alloc`
    /// resets every budget at entry, so a root decode is a
    /// pure function of those two inputs, and a body that
    /// mentions the same constant many times -- one reference
    /// into a large static per element of a table the body's
    /// initializer builds -- otherwise re-materialises the
    /// pointee on every mention. Only decodes that finished
    /// with no recorded error and no budget-failure flag are
    /// memoised, so a memoised value is exactly what a fresh
    /// decode of the same constant produces. Allocations
    /// larger than `MAX_CONST_MEMO_BYTES` are not memoised:
    /// the key copies the bytes, and a large allocation's own
    /// decode is dominated by its content, paid once per
    /// mention as before.
    const_memo: std::collections::HashMap<
        (rustc_public::ty::Ty, Vec<Option<u8>>, Vec<(usize, ProvTargetKey)>),
        std::rc::Rc<sm::ConstValue>,
    >,
    /// Memoised span resolutions, keyed by the span and the
    /// body span bounding its macro-expansion walk. Resolving
    /// a span crosses into the compiler for every call, and a
    /// body's statements share spans heavily (every statement
    /// of one source expression carries that expression's
    /// span), so a large body otherwise pays the crossing
    /// once per statement.
    source_line_cache: std::collections::HashMap<
        (rustc_public::ty::Span, rustc_public::ty::Span),
        sm::SourceLine,
    >,
}

impl Default for Converter {
    fn default() -> Self {
        Self::new()
    }
}
