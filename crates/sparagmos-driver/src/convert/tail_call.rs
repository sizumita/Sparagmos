//! Tail-call bodies, materialized through the compiler's
//! internal MIR (Issue #384).
//!
//! rustc_public 1.100.0's internal-to-stable terminator
//! conversion is `unimplemented!()` for
//! `mir::TerminatorKind::TailCall` (convert/stable/mir.rs:775),
//! so a body using `become` cannot be materialized as a Stable
//! MIR body at all: the fetch itself panics, and the body is
//! skipped with [`TAIL_CALL_REASON`]. The terminator is,
//! however, fully present in the compiler's internal MIR, and
//! its documented semantics are a call whose destination is
//! always the caller's return place, whose continuation is
//! always the current frame's return, and whose unwinding is
//! always the current frame's own (see the variant's docs in
//! rustc_middle). A tail call therefore has an exact rewriting
//! as an ordinary call into the return place followed by a
//! `return`:
//!
//! ```text
//! become f(args)   ==   _0 = f(args) -> [return: bb_ret, unwind: continue]
//! bb_ret: return
//! ```
//!
//! The one property the rewriting does not preserve is the
//! constant stack usage of a genuinely tail-recursive call:
//! the interpreter (and the JIT) will use one frame per call,
//! exactly as if the source had written the call in non-tail
//! position. The values computed are identical.
//!
//! The fetch mirrors rustc_public_bridge's own body builder
//! step for step -- clone the instance's MIR, instantiate it
//! into the fully-monomorphized environment, evaluate every
//! constant operand -- with the terminator rewriting applied
//! to the monomorphized body, and converts the result through
//! the bridge's internal-to-stable conversion, the same
//! conversion `Instance::body()` would have applied to the
//! unrewritten body. This is a Stable MIR gap filled at the
//! driver/conversion boundary, where this crate already reads
//! the internal view for the questions Stable MIR cannot
//! answer (see `unsafe_binder_inner_ty`); no other input path
//! is created.

use super::*;

/// The monomorphized Stable MIR body of `inst` with every
/// tail-call terminator rewritten as a call into the return
/// place followed by a return (see the module docs), or
/// `None` when the instance's internal body contains no tail
/// call, when no compiler context is installed, or when the
/// internal materialization cannot be carried out (a constant
/// whose evaluation stays too generic is the known case: the
/// ordinary Stable MIR fetch classifies that failure by name,
/// and this path must not paper over it).
pub fn tail_call_erased_body(inst: &Instance) -> Option<mir::Body> {
    screen_question(|| {
        let tcx = compiler_tcx()?;
        let internal = rustc_public::rustc_internal::internal(tcx, inst.clone());
        // The bridge asks an intrinsic instance for its
        // fallback body through the raw item instance; mirror
        // that, so the internal body read here is the body the
        // Stable MIR fetch would have converted.
        let internal = match internal.def {
            rustc_middle::ty::InstanceKind::Intrinsic(def_id) => {
                rustc_middle::ty::Instance::new_raw(def_id, internal.args)
            }
            _ => internal,
        };
        let body = tcx.instance_mir(internal.def);
        if !body.basic_blocks.iter().any(|block| {
            matches!(
                block.terminator().kind,
                rustc_middle::mir::TerminatorKind::TailCall { .. }
            )
        }) {
            return None;
        }
        // A body containing a tail call is a function body,
        // never an anonymous constant, so the bridge's body
        // builder always takes its instantiation branch for it
        // (the branch it skips only for argument-less anonymous
        // constants, whose bodies are already monomorphic and
        // cannot contain a call of any kind).
        let mut mono = internal.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            rustc_middle::ty::TypingEnv::fully_monomorphized(),
            rustc_middle::ty::EarlyBinder::bind(tcx, body.clone()),
        );
        let mut eval = ConstOperandEval { tcx, too_generic: false };
        rustc_middle::mir::visit::MutVisitor::visit_body(&mut eval, &mut mono);
        if eval.too_generic {
            return None;
        }
        rewrite_tail_calls(&mut mono);
        Some(rustc_public::rustc_internal::stable(mono))
    })
}

/// Rewrite every `TailCall` terminator of an internal body as
/// a call into the return place targeting a fresh block that
/// returns (see the module docs). The fresh blocks are
/// appended in the order the tail-call blocks appear, so the
/// rewriting is deterministic.
fn rewrite_tail_calls(body: &mut rustc_middle::mir::Body<'_>) {
    use rustc_middle::mir::{
        BasicBlock, BasicBlockData, Terminator, TerminatorKind,
    };
    let base = body.basic_blocks.len();
    let mut fresh: Vec<(usize, Terminator<'_>)> = Vec::new();
    for (index, block) in body.basic_blocks.iter().enumerate() {
        let Some(term) = &block.terminator else {
            continue;
        };
        if matches!(term.kind, TerminatorKind::TailCall { .. }) {
            fresh.push((index, term.clone()));
        }
    }
    for (n, (index, term)) in fresh.iter().enumerate() {
        let target = BasicBlock::from_usize(base + n);
        let TerminatorKind::TailCall { func, args, fn_span } = term.kind.clone() else {
            unreachable!()
        };
        let block = &mut body.basic_blocks.as_mut()[rustc_middle::mir::BasicBlock::from_usize(*index)];
        let term_mut = block.terminator.as_mut().expect("tail-call block has a terminator");
        term_mut.kind = TerminatorKind::Call {
            func,
            args,
            destination: rustc_middle::mir::Place::from(rustc_middle::mir::RETURN_PLACE),
            target: Some(target),
            unwind: rustc_middle::mir::UnwindAction::Continue,
            call_source: rustc_middle::mir::CallSource::Normal,
            fn_span,
        };
        let mut return_term = term.clone();
        return_term.kind = TerminatorKind::Return;
        body.basic_blocks
            .as_mut()
            .push(BasicBlockData::new(Some(return_term), false));
    }
}

/// The constant-operand evaluation rustc_public_bridge's body
/// builder applies to a monomorphized body: every constant
/// operand is replaced by its evaluated value. The one
/// divergence is the too-generic case, where the bridge's
/// `unreachable!()` aborts the fetch (and the driver classifies
/// it by name as an instance-constant failure): here it only
/// marks the visitor, and the caller abandons this path so the
/// ordinary fetch keeps that classification.
struct ConstOperandEval<'tcx> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    too_generic: bool,
}

impl<'tcx> rustc_middle::mir::visit::MutVisitor<'tcx> for ConstOperandEval<'tcx> {
    fn tcx(&self) -> rustc_middle::ty::TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_const_operand(
        &mut self,
        constant: &mut rustc_middle::mir::ConstOperand<'tcx>,
        location: rustc_middle::mir::Location,
    ) {
        let const_ = constant.const_;
        let val = match const_.eval(
            self.tcx,
            rustc_middle::ty::TypingEnv::fully_monomorphized(),
            constant.span,
        ) {
            Ok(v) => v,
            Err(rustc_middle::mir::interpret::ErrorHandled::Reported(..)) => return,
            Err(rustc_middle::mir::interpret::ErrorHandled::TooGeneric(..)) => {
                self.too_generic = true;
                return;
            }
        };
        let ty = constant.ty();
        constant.const_ = rustc_middle::mir::Const::Val(val, ty);
        self.super_const_operand(constant, location);
    }
}
