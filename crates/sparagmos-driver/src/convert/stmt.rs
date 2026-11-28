//! Statement and terminator conversion, including inline-asm operands and targets.

use super::*;

impl Converter {
    pub(crate) fn statement_kind(&mut self, kind: &mir::StatementKind) -> sm::StatementKind {
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
            SK::Coverage(cov) => {
                // The Stable MIR coverage payload is opaque; its
                // rendered form is the internal `CoverageKind`
                // debug text, of which only the virtual-counter
                // form survives into the optimised bodies the
                // driver converts (the point/block-marker forms
                // are erased by the coverage pass itself). Any
                // other shape is a conversion error rather than a
                // silently dropped counter point: counts taken
                // from a body whose counter points were misread
                // would be fabricated data.
                let text = cov.to_string();
                match text
                    .strip_prefix("VirtualCounter(bcb")
                    .and_then(|rest| rest.strip_suffix(')'))
                    .and_then(|digits| digits.parse::<u32>().ok())
                {
                    Some(bcb) => sm::StatementKind::Coverage(bcb),
                    None => {
                        self.errors.push(format!(
                            "Coverage statement with unrecognised payload {text}"
                        ));
                        sm::StatementKind::Coverage(u32::MAX)
                    }
                }
            }
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
            let inst = resolve_instance_guarded(def, &args)?;
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
        // receiver instead. The exception is a call the
        // compiler itself rewrote around a moved argument —
        // the `const_eval_select` substitution calls the
        // runtime branch directly, with the branch operand in
        // the function position keeping its argument span —
        // where the operand names no callee of this call and
        // the terminator's own span, the original call
        // expression, is the location the compiler names.
        let mut term = term.clone();
        // The stamp is located by the function operand's span
        // (see below), but an optimized body can carry the
        // operand as a promoted constant whose span was erased
        // (it resolves to line 0): the terminator's own span
        // still names the call site then, so it takes over.
        // A span with a real position wins when the operand
        // opens the call (see `operand_opens_call`), because
        // for a method call the terminator span covers the
        // whole receiver expression and would name the
        // receiver; an operand the source text shows to be a
        // moved argument does not win, and a span the source
        // text cannot adjudicate keeps the standing rule.
        let operand_has_position = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.collapsed_filename_lines(c.span, body.span)
                .unwrap_or_else(|| (c.span.get_filename(), c.span.get_lines()))
                .1
                .start_line
                != 0
        }))
        .unwrap_or(false);
        if operand_has_position
            && self.operand_opens_call(c.span, term.source_info.span) != Some(false)
        {
            term.source_info.span = c.span;
        } else if !operand_has_position {
            if let Some(span) =
                self.method_call_name_span(term.source_info.span, &canonical)
            {
                // The operand's span was erased, and the terminator's
                // own span starts at the receiver: for a method call
                // the stamp must name the method name's own span,
                // recovered from the call's source text.
                term.source_info.span = span;
            }
        }
        let mc = body.caller_location(&term, None);
        let constant = self.constant(&mir::ConstOperand {
            span: term.source_info.span,
            user_ty: None,
            const_: mc,
        });
        Some(sm::Operand::Constant(Box::new(constant)))
    }

    pub(crate) fn terminator_kind(&mut self, body: &mir::Body, term: &mir::Terminator) -> sm::TerminatorKind {
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
        // A register operand holding a function pointer: the
        // signature shape a `call` step through it executes
        // through, resolved from the operand's own type.
        let fn_call_sig = if matches!(kind, sm::AsmOperandKind::In | sm::AsmOperandKind::InOut) {
            op.in_value
                .as_ref()
                .and_then(|v| v.ty(body.locals()).ok())
                .and_then(|t| self.asm_fn_ptr_sig(&t))
        } else {
            None
        };
        // A sym operand denoting a static: its display name,
        // for a `lea` step that materializes its address (the
        // interpreter resolves the name to its static key,
        // whose disambiguating suffix only conversion knows).
        let static_key = if raw.starts_with("SymStatic {") {
            static_name.clone()
        } else {
            None
        };
        let width = if fn_call_sig.is_some() && width == 0 {
            // A function pointer occupies a full register.
            64
        } else {
            width
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
            fn_call_sig,
            static_key,
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
            let inst = resolve_instance_guarded(def, &GenericArgs(vec![]))?;
            Some(guarded_ty_kind(&inst.ty())?.fn_sig()?.skip_binder())
        }))
        .ok()??;
        if sig.c_variadic {
            return None;
        }
        let abi = asm_abi_family(&sig.abi).or_else(|| {
            // An `extern "custom"` callee is by construction a
            // naked or `global_asm!` body: the call step forwards
            // the register file to it, and the SysV64 shape only
            // classifies its declared scalars.
            if format!("{:?}", sig.abi).starts_with("Custom") {
                Some(sm::AsmAbi::SysV64)
            } else {
                None
            }
        })?;
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

    /// Resolve the signature *shape* of a function-pointer (or
    /// function-item) type an asm register operand holds: the
    /// register-passing signature a `call` step through the
    /// register executes through, under the same scalar /
    /// register-passed discipline as [`Self::asm_call_sig`].
    /// The `custom` ABI family maps to the SysV64 register
    /// model: a `custom` callee is by construction a naked or
    /// `global_asm!` body whose register state the call step
    /// forwards, and the shape only classifies its declared
    /// scalars. The returned signature's `name` is empty: the
    /// callee is the operand's runtime value.
    fn asm_fn_ptr_sig(&mut self, ty: &Ty) -> Option<sm::AsmCallSig> {
        let sig = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Some(guarded_ty_kind(ty)?.fn_sig()?.skip_binder())
        }))
        .ok()??;
        if sig.c_variadic {
            return None;
        }
        let abi = asm_abi_family(&sig.abi).or_else(|| {
            if format!("{:?}", sig.abi).starts_with("Custom") {
                Some(sm::AsmAbi::SysV64)
            } else {
                None
            }
        })?;
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
            name: String::new(),
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
            // Provisional: `mark_cleanup_region` (convert/body.rs)
            // refines the reason of `Terminate` actions stated in
            // cleanup blocks to `InCleanup`; Stable MIR states none.
            U::Terminate => sm::UnwindAction::Terminate(sm::UnwindTerminateReason::Generic),
            U::Cleanup(t) => sm::UnwindAction::Cleanup(bb(*t)),
        }
    }
}
