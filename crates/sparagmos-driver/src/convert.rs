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

use std::collections::HashMap;

use rustc_public::abi::{FieldsShape, TagEncoding, VariantsShape};
use rustc_public::mir;
use rustc_public::mir::mono::{Instance, InstanceKind};
use rustc_public::ty::{
    AdtDef, AdtKind, AliasKind as SmirAliasKind, Allocation, ConstantKind, FloatTy as SmirFloatTy,
    ClosureKind, ExistentialPredicate, FnDef, GenericArgKind, GenericArgs,
    IntTy as SmirIntTy, RigidTy,
    Ty, TyConstKind, TyKind,
    UintTy as SmirUintTy, VariantIdx,
};
use rustc_public_bridge::IndexedVal;
use rustc_public::{CrateDef, CrateDefType};

use sparagmos_mir as sm;

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
    /// Rendered instance keys whose layout query was already
    /// attempted (successfully or not), so a type whose query
    /// cannot be answered is not re-queried at every conversion.
    layout_attempted: std::collections::HashSet<String>,
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
    /// Rendered instance keys whose instance-shape recording was
    /// already attempted (successfully or not), so an instance
    /// whose fields cannot be normalized is not retried at every
    /// conversion.
    instance_shape_attempted: std::collections::HashSet<String>,
    /// Declared types of the target crate's statics, recorded by
    /// the item scan before conversion (`note_static_ty`).
    static_tys: HashMap<sm::DefId, rustc_public::ty::Ty>,
    /// Unsupported hits of the function currently being converted.
    errors: Vec<String>,
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
/// deep: a parameter that resolves to another parameter counts as
/// suspect, which only ever skips a query, never enables one.
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
    match ty.kind() {
        TyKind::RigidTy(r) => match r {
            RigidTy::Slice(_) | RigidTy::Str | RigidTy::Dynamic(..) => {
                Sizedness::Unsized
            }
            RigidTy::Adt(def, a2) => adt_sizedness(&def, &a2, depth),
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
    match ty.kind() {
        TyKind::RigidTy(r) => match r {
            RigidTy::Slice(_) | RigidTy::Str | RigidTy::Dynamic(..) => true,
            RigidTy::Adt(def, a2) => adt_may_be_unsized(&def, &a2, depth),
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

/// Why a body that mentions a coroutine-closure (async closure)
/// type cannot be converted: rustc_public 1.100.0's
/// internal-to-stable type conversion is unimplemented for
/// `ty::CoroutineClosure` ("FIXME(async_closures): Lower these
/// to SMIR"), and Stable MIR types are opaque, so the panic
/// fires lazily, the first time a conversion asks for the kind
/// of such a type — inside `convert_body`, or when the type of
/// an async closure item is asked for directly. The affected
/// body is skipped with this reason instead of the panic
/// aborting the conversion of the whole crate; every other body
/// still converts. Async closure bodies themselves stay
/// unconverted: Stable MIR 1.100.0 cannot hand them over at all.
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
    message.is_some_and(|m| m.contains("FIXME(async_closures)"))
}

impl Converter {
    pub fn new() -> Self {
        Self {
            def_ids: HashMap::new(),
            instance_ids: HashMap::new(),
            def_names: Vec::new(),
            adt_discriminants: sm::AdtDiscriminants::new(),
            adt_layouts: sm::AdtLayouts::new(),
            layout_attempted: std::collections::HashSet::new(),
            layout_names: std::collections::HashMap::new(),
            adt_shapes: sm::AdtShapes::new(),
            adt_debug_shapes: sm::AdtDebugShapes::new(),
            debug_shape_depth: 0,
            thread_locals: sm::ThreadLocals::new(),
            adt_instance_shapes: sm::AdtInstanceShapes::new(),
            adt_niches: sm::AdtNiches::new(),
            instance_shape_attempted: std::collections::HashSet::new(),
            static_tys: HashMap::new(),
            errors: Vec::new(),
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
    /// item scan before any conversion: a reference into a static's
    /// interior is decoded against the static's own type (its seed
    /// is the full initializer and the reference carries a
    /// projection into it), and the type query is only safe for
    /// real items, so it is asked there, never from the constant
    /// decoder.
    pub fn note_static_ty<D: CrateDef>(&mut self, def: &D, ty: rustc_public::ty::Ty) {
        let id = self.crate_def_id(def);
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
        sm::DefId::new(0, idx)
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


    /// Convert one Stable MIR body, containing rustc_public
    /// 1.100.0's unimplemented coroutine-closure type conversion:
    /// a body that mentions a coroutine-closure (async closure)
    /// type fails with `COROUTINE_CLOSURE_TYPE_REASON` instead of
    /// the panic escaping. Any other panic is resumed — this
    /// containment must never paper over an unrelated compiler
    /// failure.
    pub fn convert_body_contained(
        &mut self,
        body: &mir::Body,
    ) -> Result<sm::Body, Vec<String>> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.convert_body(body)
        })) {
            Ok(result) => result,
            Err(payload) => {
                if is_coroutine_closure_panic(&*payload) {
                    Err(vec![COROUTINE_CLOSURE_TYPE_REASON.to_string()])
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
        let blocks: Vec<sm::BasicBlockData> = body
            .blocks
            .iter()
            .map(|b| self.basic_block(body, b))
            .collect();
        let debug_info: Vec<sm::VarDebugInfo> = body
            .var_debug_info
            .iter()
            .map(|info| self.var_debug_info(info))
            .collect();
        if self.errors.is_empty() {
            let mut b = sm::Body::new(locals, body.arg_locals().len(), blocks);
            b.var_debug_info = debug_info;
            Ok(b)
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    fn var_debug_info(&mut self, info: &mir::VarDebugInfo) -> sm::VarDebugInfo {
        let value = match &info.value {
            mir::VarDebugInfoContents::Place(p) => {
                sm::VarDebugInfoContents::Place(self.place(p))
            }
            mir::VarDebugInfoContents::Const(c) => {
                sm::VarDebugInfoContents::Constant(Box::new(self.constant(c)))
            }
        };
        sm::VarDebugInfo {
            name: info.name.clone(),
            source_info: sm::SourceInfo::DUMMY,
            value,
        }
    }

    fn basic_block(&mut self, body: &mir::Body, bb: &mir::BasicBlock) -> sm::BasicBlockData {
        let statements = bb
            .statements
            .iter()
            .map(|s| sm::Statement::new(self.statement_kind(&s.kind)))
            .collect();
        let terminator = self.terminator_kind(body, &bb.terminator);
        let mut data = sm::BasicBlockData::new(sm::Terminator::new(terminator));
        data.statements = statements;
        // Stable MIR (1.100.0) does not expose `is_cleanup`; see module docs.
        data.is_cleanup = false;
        data
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
        let inst = Instance::resolve(def, &args).ok()?;
        if !inst.requires_caller_location() {
            return None;
        }
        // A builtin-served callee has no body to bind the stamp
        // to; only the builtins that consume the location (the
        // intrinsic, the panic entry points) are stamped, so the
        // other builtins keep their declared argument lists.
        let canonical = sparagmos_interp::builtins::canonical_name(&def.def_id().name());
        if let Some(kind) = sparagmos_interp::builtins::lookup(&canonical) {
            if !sparagmos_interp::builtins::consumes_caller_location(kind) {
                return None;
            }
        }
        let mc = body.caller_location(term, None);
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
                // Two classes convert for real (see
                // `sm::inline_asm_is_noop` and
                // `sm::inline_asm_program`): the no-op class
                // (empty / comment-only / nop-only template, no
                // NORETURN) when no operand place is written, and
                // the evaluable register-program class, whose
                // operands the structured form below carries in
                // full. Anything else keeps the named conversion
                // error.
                let has_out_place = operands.iter().any(|op| op.out_place.is_some());
                let noop = !has_out_place && sm::inline_asm_is_noop(template, options);
                let supported = noop
                    || sm::inline_asm_program(
                        template,
                        options,
                        &operands
                            .iter()
                            .map(|op| self.asm_operand(body, op, false))
                            .collect::<Vec<_>>(),
                        destination.is_some(),
                    )
                    .is_some();
                let ops = if supported {
                    operands
                        .iter()
                        .map(|op| self.asm_operand(body, op, true))
                        .collect()
                } else {
                    self.err("TerminatorKind::InlineAsm");
                    Vec::new()
                };
                sm::TerminatorKind::InlineAsm {
                    template: template.clone(),
                    operands: ops,
                    options: options.clone(),
                    destination: destination.map(bb),
                    unwind: self.unwind(unwind),
                }
            }
        }
    }

    /// One Stable MIR inline-asm operand in the structured form
    /// the executors consume. The public operand struct keeps only
    /// the value/place pair, so the kind and the register are read
    /// from its raw representation (`In { reg: RegClass(X86(reg)),
    /// value: .. }`, `Const { value: const 5_usize }`, ...). With
    /// `convert` false only the classification facts are computed
    /// (the value and place stay empty): classification must not
    /// itself convert operands of a terminator it may reject.
    fn asm_operand(
        &mut self,
        body: &mir::Body,
        op: &mir::InlineAsmOperand,
        convert: bool,
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
        } else {
            sm::AsmOperandKind::Sym
        };
        let reg_class = raw.contains("reg: RegClass(");
        let (width, const_bits) = match kind {
            sm::AsmOperandKind::Const => match asm_const_literal(raw) {
                Some((bits, width)) => (width, Some(bits)),
                None => (0, None),
            },
            sm::AsmOperandKind::Sym => (0, None),
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
        sm::InlineAsmOperand {
            kind,
            reg_class,
            width,
            const_bits,
            in_value: if convert {
                op.in_value.as_ref().map(|v| self.operand(v))
            } else {
                None
            },
            out_place: if convert {
                op.out_place.as_ref().map(|p| self.place(p))
            } else {
                None
            },
        }
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
        k
    }

    /// The constant's own type when it is a plain scalar type:
    /// the one reading of a scalar literal's bits that needs no
    /// context from the operation using it (see
    /// `sm::Constant::ty`). Other types are not recorded: their
    /// conversion is not needed for the literal reading, and
    /// asking for it could fail where the constant itself is fine.
    fn constant_scalar_ty(&mut self, c: &mir::ConstOperand) -> Option<sm::Ty> {
        match c.const_.ty().kind() {
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
                if matches!(
                    ty.kind(),
                    TyKind::RigidTy(RigidTy::FnDef(..) | RigidTy::Closure(..))
                ) {
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
                    // as for `CK::ZeroSized` above.
                    if matches!(
                        ty.kind(),
                        TyKind::RigidTy(RigidTy::FnDef(..) | RigidTy::Closure(..))
                    ) {
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
        match ty.kind() {
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
        match ty.kind() {
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
                    _ => {
                        // f128 has no exact f64 form; it stays an
                        // explicit unsupported constant. (f16 widens
                        // to f64 exactly and is decoded above.)
                        self.err("ConstantKind::Allocated (f128)");
                        0
                    }
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
        // Fat-pointer shape: `&str` and `&[u8]` literals take the
        // shortcut; every other shape falls through to the decoder
        // below (a raw-byte slice of wider or non-scalar elements
        // has no value form in the interpreter).
        if let Some((data, meta)) = self.read_fat_pointer(alloc) {
            match inner.kind() {
                TyKind::RigidTy(RigidTy::Str) => {
                    return match String::from_utf8(data) {
                        Ok(s) => sm::ConstValue::Str(s),
                        Err(_) => {
                            self.err("ConstantKind::Allocated (&str, invalid UTF-8)");
                            sm::ConstValue::ZeroSized
                        }
                    }
                }
                TyKind::RigidTy(RigidTy::Slice(elem))
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
                match inner.kind() {
                    TyKind::RigidTy(RigidTy::Str) if plain => {
                        return match pointee.raw_bytes().map(String::from_utf8) {
                            Ok(Ok(s)) => sm::ConstValue::Str(s),
                            _ => {
                                self.err("ConstantKind::Allocated (&str, unreadable)");
                                sm::ConstValue::ZeroSized
                            }
                        };
                    }
                    TyKind::RigidTy(RigidTy::Slice(elem)) if plain => {
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
                    TyKind::RigidTy(RigidTy::Array(elem, _)) if plain => {
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
        // No provenance at all: a stored zero word is a null
        // pointer constant. Anything else is an integer-derived
        // address, which has no faithful value form.
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
                    _ => return None,
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
    /// in an allocation, so only the shape with a single payload
    /// is faithful: at most one field is non-zero-sized (the
    /// `MaybeUninit<T>` shape — an `uninit: ()` field beside a
    /// `value: ManuallyDrop<T>` field — is the canonical case).
    /// Every field is read at offset 0, where all union fields
    /// live; zero-sized fields decode as their zero-sized values.
    /// A union with two or more payload-sized fields stays
    /// undecoded, since the bytes do not say which one is live.
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
        let mut payload_fields = 0;
        for field in fields.iter() {
            let fty = field.ty_with_args(args);
            let size = fty.layout().ok()?.shape().size.bytes();
            if size > 0 {
                payload_fields += 1;
            }
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
        if payload_fields > 1 {
            return None;
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
                        let d = self.closure_def_id(&def, &args);
                        return Some(sm::ConstValue::Closure(d));
                    }
                }
            }
        }
        if let TyKind::RigidTy(RigidTy::Closure(def, args)) = inst.ty().kind() {
            let d = self.closure_def_id(&def, &args);
            return Some(sm::ConstValue::Closure(d));
        }
        let d = self.instance_def_id(inst.def.def_id(), inst);
        Some(sm::ConstValue::FnDef(d))
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
            return Some(if addr == 0 {
                sm::ConstValue::NullPtr
            } else {
                sm::ConstValue::IntPtr(addr as u64)
            });
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
            // Vtables are not materialised.
            _ => return None,
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
                if let Some(def) = &static_def {
                    if let Some(c) =
                        self.decode_static_ref(def, inner, &target, target_off, depth)
                    {
                        return Some(c);
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
            });
        };
        let projection = self.static_offset_path(&static_ty, target_off, inner, depth)?;
        let seed = self.decode_const(&static_ty, alloc, 0, depth + 1)?;
        Some(sm::ConstValue::StaticRef {
            def: def_id,
            value: Box::new(seed),
            projection,
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
        if off == 0 && ty == inner {
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
        let converted = self.ty_inner(ty);
        match ty.kind() {
            TyKind::RigidTy(RigidTy::Adt(def, args)) => {
                self.record_layout(ty, &def, &args, &converted);
                self.record_debug_shape(&def, &converted);
                self.record_instance_shape(ty, &def, &args, &converted);
            }
            TyKind::RigidTy(RigidTy::Coroutine(..))
            | TyKind::RigidTy(RigidTy::CoroutineClosure(..)) => {
                self.record_coroutine_layout(ty, &converted);
            }
            _ => {}
        }
        converted
    }

    fn ty_inner(&mut self, ty: &rustc_public::ty::Ty) -> sm::Ty {
        match ty.kind() {
            TyKind::RigidTy(r) => self.rigid_ty(&r),
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
    /// instance (see `sm::AdtDebugShapes`). Only struct and enum
    /// instances with no remaining generic parameters get an entry:
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
            AdtKind::Union => return,
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
        let key = self.layout_key(converted);
        if !self.layout_attempted.insert(key.clone()) {
            return;
        }
        self.record_niche(ty, def, args, &key);
        if adt_may_be_unsized(def, args, 0) {
            return;
        }
        // The layout query is not total: for some types that can
        // appear in converted bodies (ill-formed bounds, types the
        // layout computation itself rejects) it panics inside the
        // compiler. The table is auxiliary — a failed query must
        // never fail the conversion — so the query is contained:
        // the instance simply gets no entry and stays unsupported.
        // (The panic is silent under the driver's panic-capture
        // hook, which is installed while the compiler runs.)
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
    /// whose generic shape mentions a type alias (see
    /// `sm::AdtInstanceShapes`), once per distinct instance.
    ///
    /// The generic shape's field types are the definition's own,
    /// so a projection alias among them reaches the interpreter
    /// unresolved — the `NonZero<T>` field is declared as
    /// `<T as ZeroablePrimitive>::NonZeroInner`, and the
    /// interpreter has no trait solver. The compiler resolves it:
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
        if !shape.variants.iter().flatten().any(|t| ty_mentions_alias(t)) {
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
    /// coroutine instance under its type tag, once per distinct
    /// instance. A coroutine's layout is the compiler's own
    /// state-machine layout (which captures live across each
    /// suspension point); it cannot be derived from the converted
    /// MIR, so it comes from the same layout query the ADT path
    /// uses. The entry lands in `adt_layouts`, the table the
    /// `size_of_val` / `align_of_val` builtins consult by tag.
    /// Instances the query cannot answer (generic parameters among
    /// the arguments) are skipped under the same containment as
    /// the ADT path: the query is caught, and the instance stays
    /// explicitly unsupported.
    fn record_coroutine_layout(&mut self, ty: &rustc_public::ty::Ty, converted: &sm::Ty) {
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

    fn rigid_ty(&mut self, r: &RigidTy) -> sm::Ty {
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
                let names = bounds
                    .iter()
                    .map(|b| match &b.value {
                        rustc_public::ty::ExistentialPredicate::Trait(t) => {
                            t.def_id.name()
                        }
                        rustc_public::ty::ExistentialPredicate::AutoTrait(d) => d.name(),
                        rustc_public::ty::ExistentialPredicate::Projection(p) => {
                            format!("{:?}", p.def_id.def_id())
                        }
                    })
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
            RigidTy::Tuple(elems) => sm::Ty::Tuple(elems.iter().map(|t| self.ty(t)).collect()),
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
    match ty.kind() {
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

