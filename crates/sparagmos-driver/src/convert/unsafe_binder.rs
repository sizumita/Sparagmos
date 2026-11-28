//! Unsafe-binder wrap/unwrap bodies, materialized through
//! the compiler's internal MIR (Issue #510).
//!
//! rustc_public 1.100.0's internal-to-stable conversion is
//! `unimplemented!("FIXME(unsafe_binders):")` for both
//! `mir::Rvalue::WrapUnsafeBinder`
//! (convert/stable/mir.rs:267) and
//! `mir::ProjectionElem::UnwrapUnsafeBinder`
//! (convert/stable/mir.rs:470), so a body using
//! `wrap_binder!` or `unwrap_binder!` cannot be materialized
//! as a Stable MIR body at all: the fetch itself panics, and
//! the body is skipped with [`UNSAFE_BINDER_REASON`]. Both
//! constructs are, however, fully present in the compiler's
//! internal MIR, and both are identity operations at the
//! value level — an unsafe binder is transparent, exactly
//! as the binder *type* erasure of PR #914 already assumes
//! for locals, arguments and return types:
//!
//! ```text
//! _d = WrapUnsafeBinder(op; binder_ty)  ==  _d = Use(op)
//! (place as UnwrapUnsafeBinder(inner))  ==  place
//! ```
//!
//! The destination local of a wrap keeps its binder type,
//! which the converter erases to the wrapped type (see
//! `unsafe_binder_inner_ty`), and the operand's type is
//! that same wrapped type, so the rewriting typechecks in
//! the erased view. An unwrap projection likewise only
//! changes the place's type from the binder type to the
//! wrapped type, which erasure already identifies.
//!
//! The fetch mirrors rustc_public_bridge's own body builder
//! step for step — clone the instance's MIR, instantiate it
//! into the fully-monomorphized environment, evaluate every
//! constant operand — with the erasure applied to the
//! monomorphized body, and converts the result through the
//! bridge's internal-to-stable conversion, the same
//! conversion `Instance::body()` would have applied to the
//! unerased body. This is a Stable MIR gap filled at the
//! driver/conversion boundary, where this crate already
//! reads the internal view for the questions Stable MIR
//! cannot answer (see `unsafe_binder_inner_ty` and
//! `tail_call_erased_body`); no other input path is created.

use super::*;

/// The monomorphized Stable MIR body of `inst` with every
/// unsafe-binder wrap/unwrap erased to the identity (see
/// the module docs), or `None` when the instance's internal
/// body contains neither construct, when no compiler
/// context is installed, or when the internal
/// materialization cannot be carried out (a constant whose
/// evaluation stays too generic is the known case: the
/// ordinary Stable MIR fetch classifies that failure by
/// name, and this path must not paper over it).
pub fn unsafe_binder_erased_body(inst: &Instance) -> Option<mir::Body> {
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
        if !body_contains_binder_construct(body) {
            return None;
        }
        // A body containing a wrap/unwrap is a function body,
        // never an anonymous constant, so the bridge's body
        // builder always takes its instantiation branch for
        // it (the branch it skips only for argument-less
        // anonymous constants, whose bodies are already
        // monomorphic and cannot contain a wrap of any kind).
        let mut mono = internal.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            rustc_middle::ty::TypingEnv::fully_monomorphized(),
            rustc_middle::ty::EarlyBinder::bind(tcx, body.clone()),
        );
        let mut eval = BinderConstOperandEval {
            tcx,
            too_generic: false,
        };
        rustc_middle::mir::visit::MutVisitor::visit_body(&mut eval, &mut mono);
        if eval.too_generic {
            return None;
        }
        let mut erase = BinderErase { tcx };
        rustc_middle::mir::visit::MutVisitor::visit_body(&mut erase, &mut mono);
        Some(rustc_public::rustc_internal::stable(mono))
    })
}

/// Whether an internal body mentions a wrap rvalue or an
/// unwrap projection anywhere: the two constructs whose
/// Stable MIR conversion is unimplemented.
fn body_contains_binder_construct(body: &rustc_middle::mir::Body<'_>) -> bool {
    let mut detect = BinderDetect { found: false };
    rustc_middle::mir::visit::Visitor::visit_body(&mut detect, body);
    detect.found
}

struct BinderDetect {
    found: bool,
}

impl<'tcx> rustc_middle::mir::visit::Visitor<'tcx> for BinderDetect {
    fn visit_rvalue(
        &mut self,
        rvalue: &rustc_middle::mir::Rvalue<'tcx>,
        location: rustc_middle::mir::Location,
    ) {
        if matches!(rvalue, rustc_middle::mir::Rvalue::WrapUnsafeBinder(..)) {
            self.found = true;
        }
        self.super_rvalue(rvalue, location);
    }

    fn visit_place(
        &mut self,
        place: &rustc_middle::mir::Place<'tcx>,
        context: rustc_middle::mir::visit::PlaceContext,
        location: rustc_middle::mir::Location,
    ) {
        if place.projection.iter().any(|elem| {
            matches!(
                elem,
                rustc_middle::mir::ProjectionElem::UnwrapUnsafeBinder(_)
            )
        }) {
            self.found = true;
        }
        self.super_place(place, context, location);
    }
}

fn place_has_unwrap(place: &rustc_middle::mir::Place<'_>) -> bool {
    place.projection.iter().any(|elem| {
        matches!(
            elem,
            rustc_middle::mir::ProjectionElem::UnwrapUnsafeBinder(_)
        )
    })
}

/// The erasure itself (see the module docs): a wrap rvalue
/// becomes a plain use of its operand, and an unwrap
/// projection is dropped from its place.
struct BinderErase<'tcx> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
}

impl<'tcx> rustc_middle::mir::visit::MutVisitor<'tcx> for BinderErase<'tcx> {
    fn tcx(&self) -> rustc_middle::ty::TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_rvalue(
        &mut self,
        rvalue: &mut rustc_middle::mir::Rvalue<'tcx>,
        location: rustc_middle::mir::Location,
    ) {
        if let rustc_middle::mir::Rvalue::WrapUnsafeBinder(op, _) = rvalue {
            let op = op.clone();
            *rvalue = rustc_middle::mir::Rvalue::Use(op, rustc_middle::mir::WithRetag::No);
        }
        self.super_rvalue(rvalue, location);
    }

    fn visit_place(
        &mut self,
        place: &mut rustc_middle::mir::Place<'tcx>,
        context: rustc_middle::mir::visit::PlaceContext,
        location: rustc_middle::mir::Location,
    ) {
        if place_has_unwrap(place) {
            let filtered: Vec<rustc_middle::mir::PlaceElem<'tcx>> = place
                .projection
                .iter()
                .filter(|elem| {
                    !matches!(
                        elem,
                        rustc_middle::mir::ProjectionElem::UnwrapUnsafeBinder(_)
                    )
                })
                .collect();
            *place =
                rustc_middle::mir::Place::from(place.local).project_deeper(&filtered, self.tcx);
        }
        self.super_place(place, context, location);
    }
}

/// The constant-operand evaluation rustc_public_bridge's
/// body builder applies to a monomorphized body: every
/// constant operand is replaced by its evaluated value. The
/// one divergence is the too-generic case, where the
/// bridge's `unreachable!()` aborts the fetch (and the
/// driver classifies it by name as an instance-constant
/// failure): here it only marks the visitor, and the caller
/// abandons this path so the ordinary fetch keeps that
/// classification. (A copy of the tail-call path's visitor:
/// the two erasures are independent.)
struct BinderConstOperandEval<'tcx> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    too_generic: bool,
}

impl<'tcx> rustc_middle::mir::visit::MutVisitor<'tcx> for BinderConstOperandEval<'tcx> {
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
