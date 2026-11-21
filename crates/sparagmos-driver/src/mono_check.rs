//! Monomorphization validity checks that rustc performs while
//! monomorphizing a crate for code generation and that MIR
//! conversion alone does not reproduce.
//!
//! MIR conversion is deliberately permissive: it translates the
//! bodies Stable MIR hands over, and a body that translates cleanly
//! is reported as converted. Some programs nevertheless fail to
//! compile under rustc because a check that only runs at
//! monomorphization time rejects them. When the driver also skipped
//! that check, its verdict disagreed with rustc's on
//! failure-expecting tests (the "expected failure but conversion
//! succeeded" bucket). The checks here mirror rustc's own, over the
//! same evidence rustc uses — the monomorphic bodies that were
//! converted, with the call targets' substituted types:
//!
//! - Invalid SIMD intrinsic monomorphizations (rustc's E0511, from
//!   `generic_simd_intrinsic` in rustc_codegen_llvm): every
//!   `simd_*` intrinsic requires its first argument's type to be a
//!   SIMD vector type, except `simd_splat` (whose *return* type
//!   must be a vector) and `simd_select_bitmask` (whose *second*
//!   argument must be a vector). A scalar where a vector is
//!   required is a hard compile error under rustc.
//! - Invalid SIMD layouts: a SIMD type with zero lanes or more
//!   lanes than the monomorphization lane limit has no layout;
//!   rustc rejects the crate when code generation asks for one.
//!   Every local of a converted body is materialized by code
//!   generation, so a SIMD-typed local whose layout does not exist
//!   is the same failure.
//!
//! Anything not provable from the converted bodies is left alone:
//! the checks only report a failure when the invalid shape is
//! positively present, mirroring the corresponding rustc error.

use rustc_public::mir::mono::{Instance, InstanceKind};
use rustc_public::mir::{Body, Operand, TerminatorKind};
use rustc_public::ty::{RigidTy, Ty, TyKind};
use rustc_public::CrateDef;

/// Scan the Stable MIR bodies that were converted for this program
/// and return the first monomorphization failure rustc would have
/// reported, if any.
pub fn check_bodies(bodies: &[Body]) -> Option<String> {
    for body in bodies {
        if let Some(err) = check_body(body) {
            return Some(err);
        }
    }
    None
}

fn check_body(body: &Body) -> Option<String> {
    for local in body.locals() {
        let ty = local.ty;
        if is_simd(ty) && !simd_layout_exists(ty) {
            return Some(format!(
                "the SIMD type `{ty}` has an invalid number of lanes \
                 (zero, or more than the monomorphization lane limit)"
            ));
        }
    }
    for bb in &body.blocks {
        let TerminatorKind::Call { func, .. } = &bb.terminator.kind else {
            continue;
        };
        if let Some(err) = check_call(func) {
            return Some(err);
        }
    }
    None
}

/// Whether the layout of a SIMD type exists. The layout query can
/// panic inside the compiler for shapes it cannot represent; a
/// panic means the layout could not be determined here, not that
/// it is invalid, so it is treated as "no finding".
fn simd_layout_exists(ty: Ty) -> bool {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout())) {
        Ok(Ok(_)) => true,
        Ok(Err(_)) => false,
        Err(_) => true,
    }
}

/// Whether a type is a SIMD vector type (a `#[repr(simd)]` ADT or
/// `core::simd::Simd`), the notion rustc's own check uses.
fn is_simd(ty: Ty) -> bool {
    ty.kind().is_simd()
}

fn check_call(func: &Operand) -> Option<String> {
    let Operand::Constant(c) = func else {
        return None;
    };
    let kind = c.const_.ty().kind();
    let TyKind::RigidTy(RigidTy::FnDef(def, args)) = &kind else {
        return None;
    };
    let (def, args) = (*def, args.clone());
    // Only compiler intrinsics are checked, and among them only the
    // `simd_*` family has monomorphization rules of this shape.
    let inst = Instance::resolve(def, &args).ok()?;
    if !matches!(inst.kind, InstanceKind::Intrinsic) {
        return None;
    }
    let full_name = def.name();
    let name = full_name.rsplit("::").next().unwrap_or(&full_name);
    if !name.starts_with("simd_") {
        return None;
    }
    let sig = kind.fn_sig()?;
    let sig = &sig.value;
    let inputs = sig.inputs();
    // Mirror `generic_simd_intrinsic`'s entry checks: the vector
    // argument whose type must be a SIMD type differs for the two
    // intrinsics handled before the general case.
    let (vector_ty, what) = match name {
        "simd_splat" => (Some(sig.output()), "return"),
        "simd_select_bitmask" => (inputs.get(1).copied(), "second argument"),
        _ => (inputs.first().copied(), "input"),
    };
    let vector_ty = vector_ty?;
    if is_simd(vector_ty) {
        return None;
    }
    Some(format!(
        "invalid monomorphization of `{name}` intrinsic: \
         expected SIMD {what} type, found non-SIMD `{vector_ty}`"
    ))
}
