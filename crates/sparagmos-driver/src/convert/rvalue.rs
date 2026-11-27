//! Rvalue, aggregate, operand and constant-operand conversion.

use super::*;

impl Converter {
    pub(crate) fn rvalue(&mut self, rv: &mir::Rvalue) -> sm::Rvalue {
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
            A::CoroutineClosure(def, args) => {
                let d = self.crate_def_id(def);
                self.async_closure_defs.insert(d);
                sm::AggregateKind::CoroutineClosure {
                    def: d,
                    generic_args: self.type_args(args),
                }
            }
            A::RawPtr(ty, m) => sm::AggregateKind::RawPtr {
                data_ty: self.ty(ty),
                mutability: mutability(*m),
            },
        }
    }

    pub(crate) fn operand(&mut self, op: &mir::Operand) -> sm::Operand {
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

    /// The `TypeId` segments of a function-item constant's
    /// generic type arguments, parallel to the converted
    /// argument list (see `sm::Constant::generic_type_ids`):
    /// the compiler's generic arguments still spell the regions
    /// conversion erases, so the segments are computed from
    /// them directly, one per type argument, in order. A type
    /// argument Stable MIR cannot spell (a coroutine-closure
    /// type) voids the whole list rather than shifting the
    /// indices of the arguments after it.
    fn constant_fn_type_ids(
        &mut self,
        c: &mir::ConstOperand,
    ) -> Option<Vec<(u64, u64)>> {
        let kind = guarded_ty_kind(&c.const_.ty())?;
        let TyKind::RigidTy(RigidTy::FnDef(_, args)) = kind else {
            return None;
        };
        let mut ids = Vec::new();
        for a in &args.0 {
            match a {
                GenericArgKind::Type(t) => ids.push(type_id_segments(t)?),
                GenericArgKind::Const(_) | GenericArgKind::Lifetime(_) => {}
            }
        }
        Some(ids)
    }

    pub(crate) fn constant(&mut self, c: &mir::ConstOperand) -> sm::Constant {
        let mut k = self.constant_literal(c);
        if k.generic_type_ids.is_none() {
            k.generic_type_ids = self.constant_fn_type_ids(c);
        }
        if k.ty.is_none() {
            k.ty = self.constant_scalar_ty(c);
        }
        if k.ty.is_none() {
            k.ty = self.constant_adt_ty(c, &k);
        }
        if k.ty.is_none() {
            k.ty = self.constant_array_ty(c, &k);
        }
        k
    }

    /// The constant's own type when the constant is an
    /// array value: a `transmute` of such a constant (a
    /// folded `const` array regrouped into another array
    /// shape) is decided from the source type, which the
    /// decoded tuple value alone does not name -- neither
    /// the element kind nor whether the tuple's elements
    /// are the array's leaves or a regrouping of narrower
    /// leaves. Recorded only for array-typed constants
    /// whose literal decoded to a tuple of elements; every
    /// other constant keeps the narrower rules above.
    fn constant_array_ty(
        &mut self,
        c: &mir::ConstOperand,
        k: &sm::Constant,
    ) -> Option<sm::Ty> {
        if !matches!(
            k.literal,
            sm::ConstKind::Value(sm::ConstValue::Tuple(_))
        ) {
            return None;
        }
        match guarded_ty_kind(&c.const_.ty())? {
            TyKind::RigidTy(RigidTy::Array(..)) => {
                Some(self.ty(&c.const_.ty()))
            }
            _ => None,
        }
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
                        generic_type_ids: None,
                    };
                }
                // A zero-sized constant whose type is a
                // coroutine-closure (async closure) type is a
                // closure value too: every capture is zero-sized,
                // so the constant is the closure itself. The type's
                // kind cannot be asked through Stable MIR (see
                // `guarded_ty_kind`), so the definition and the
                // upvars tuple are read through the compiler
                // context; the recorded type is shaped like a
                // plain closure type whose last generic argument
                // is the upvars tuple, which is all the conversion
                // of a closure constant reads.
                if let Some((def, _, upvars)) =
                    coroutine_closure_const_parts(&ty)
                {
                    let d = self.def_id(def);
                    self.async_closure_defs.insert(d);
                    let upvars = self.ty(&upvars);
                    return sm::Constant {
                        literal: sm::ConstKind::Ty(sm::Ty::Closure(
                            d,
                            vec![upvars],
                        )),
                        user_ty: None,
                        ty: None,
                        generic_type_ids: None,
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
                            generic_type_ids: None,
                        };
                    }
                    // A zero-sized type constant of
                    // coroutine-closure type is a closure value,
                    // as for the `ZeroSized` literal kind above.
                    if let Some((def, _, upvars)) =
                        coroutine_closure_const_parts(ty)
                    {
                        let d = self.def_id(def);
                        self.async_closure_defs.insert(d);
                        let upvars = self.ty(&upvars);
                        return sm::Constant {
                            literal: sm::ConstKind::Ty(sm::Ty::Closure(
                                d,
                                vec![upvars],
                            )),
                            user_ty: None,
                            ty: None,
                            generic_type_ids: None,
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
                        generic_type_ids: None,
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
                    generic_type_ids: None,
                }
            }
            CK::Param(p) => {
                self.err(format!("ConstantKind::Param({})", p.name));
                sm::Constant::from_value(sm::ConstValue::ZeroSized)
            }
        }
    }
}
