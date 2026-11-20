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

use rustc_public::mir;
use rustc_public::ty::{
    AliasKind as SmirAliasKind, Allocation, ConstantKind, FloatTy as SmirFloatTy, GenericArgKind,
    GenericArgs, IntTy as SmirIntTy, RigidTy, TyConstKind, TyKind, UintTy as SmirUintTy,
};
use rustc_public_bridge::IndexedVal;
use rustc_public::CrateDef;

use sparagmos_mir as sm;

/// Per-run converter state (DefId assignment) plus per-function error bag.
pub struct Converter {
    def_ids: HashMap<rustc_public::DefId, u32>,
    pub def_names: Vec<String>,
    /// Unsupported hits of the function currently being converted.
    errors: Vec<String>,
}

impl Default for Converter {
    fn default() -> Self {
        Self::new()
    }
}

impl Converter {
    pub fn new() -> Self {
        Self {
            def_ids: HashMap::new(),
            def_names: Vec::new(),
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

    fn crate_def_id<D: CrateDef>(&mut self, def: &D) -> sm::DefId {
        self.def_id(def.def_id())
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
        let blocks: Vec<sm::BasicBlockData> =
            body.blocks.iter().map(|b| self.basic_block(b)).collect();
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

    fn basic_block(&mut self, bb: &mir::BasicBlock) -> sm::BasicBlockData {
        let statements = bb
            .statements
            .iter()
            .map(|s| sm::Statement::new(self.statement_kind(&s.kind)))
            .collect();
        let terminator = self.terminator_kind(&bb.terminator.kind);
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

    fn terminator_kind(&mut self, kind: &mir::TerminatorKind) -> sm::TerminatorKind {
        use mir::TerminatorKind as TK;
        match kind {
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
                let a = args.iter().map(|a| self.operand(a)).collect();
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
                options,
                destination,
                unwind,
                ..
            } => {
                self.err("TerminatorKind::InlineAsm");
                sm::TerminatorKind::InlineAsm {
                    template: template.clone(),
                    operands: Vec::new(),
                    options: options.clone(),
                    destination: destination.map(bb),
                    unwind: self.unwind(unwind),
                }
            }
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
            R::Reborrow(..) => {
                self.err("Rvalue::Reborrow");
                sm::Rvalue::Use(sm::Operand::constant(sm::Constant::bool(false)))
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
                def: self.crate_def_id(def),
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
                self.err(format!("Operand::RuntimeChecks({checks:?})"));
                sm::Operand::constant(sm::Constant::bool(false))
            }
        }
    }

    fn constant(&mut self, c: &mir::ConstOperand) -> sm::Constant {
        use ConstantKind as CK;
        let mc = &c.const_;
        match mc.kind() {
            CK::ZeroSized => {
                // Function items are ZST constants whose type is the FnDef;
                // keep the type so call targets stay identifiable.
                let ty = mc.ty();
                if matches!(ty.kind(), TyKind::RigidTy(RigidTy::FnDef(..))) {
                    let t = self.ty(&ty);
                    return sm::Constant {
                        literal: sm::ConstKind::Ty(t),
                        user_ty: None,
                    };
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
                TyConstKind::ZSTValue(_) => {
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
                }
            }
            CK::Param(p) => {
                self.err(format!("ConstantKind::Param({})", p.name));
                sm::Constant::from_value(sm::ConstValue::ZeroSized)
            }
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
                    SmirFloatTy::F32 => alloc
                        .read_partial_uint(0..4)
                        .map(|v| v as u32 as u128)
                        .unwrap_or(0),
                    SmirFloatTy::F64 => alloc
                        .read_partial_uint(0..8)
                        .map(|v| v as u128)
                        .unwrap_or(0),
                    _ => {
                        self.err("ConstantKind::Allocated (f16/f128)");
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
        let data = target.raw_bytes().ok()?;
        let meta = alloc.read_partial_uint(8..16).ok()? as u64;
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
        // Fat-pointer shape: `&str` and `&[T]` literals.
        if let Some((data, meta)) = self.read_fat_pointer(alloc) {
            return match inner.kind() {
                TyKind::RigidTy(RigidTy::Str) => match String::from_utf8(data) {
                    Ok(s) => sm::ConstValue::Str(s),
                    Err(_) => {
                        self.err("ConstantKind::Allocated (&str, invalid UTF-8)");
                        sm::ConstValue::ZeroSized
                    }
                },
                TyKind::RigidTy(RigidTy::Slice(_)) => sm::ConstValue::Slice { data, meta },
                _ => {
                    self.err("ConstantKind::Allocated (fat pointer, unexpected type)");
                    sm::ConstValue::ZeroSized
                }
            };
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
                        if let Some(size) = scalar_elem_layout(&elem) {
                            return match pointee.raw_bytes() {
                                Ok(data) => sm::ConstValue::Slice {
                                    meta: (data.len() / size) as u64,
                                    data,
                                },
                                Err(_) => {
                                    self.err("ConstantKind::Allocated (&[T], unreadable)");
                                    sm::ConstValue::ZeroSized
                                }
                            };
                        }
                    }
                    TyKind::RigidTy(RigidTy::Array(elem, _)) if plain => {
                        if scalar_elem_layout(&elem).is_some() {
                            return match pointee.raw_bytes() {
                                Ok(data) => sm::ConstValue::Bytes(data),
                                Err(_) => {
                                    self.err("ConstantKind::Allocated (&[T; N], unreadable)");
                                    sm::ConstValue::ZeroSized
                                }
                            };
                        }
                    }
                    _ => {}
                }
            }
            return sm::ConstValue::ByRef {
                alloc_id,
                offset: 0,
            };
        }
        self.err(format!(
            "ConstantKind::Allocated (reference without provenance: {})",
            short_ty(inner)
        ));
        sm::ConstValue::ZeroSized
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
                let d = self.crate_def_id(def);
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
                let d = self.crate_def_id(def);
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
                self.err("Ty::Pat");
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
            C::BoxDerefTransmute => {
                self.err("CastKind::BoxDerefTransmute");
                sm::CastKind::Transmute
            }
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
