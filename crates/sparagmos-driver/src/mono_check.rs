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
//! - Types too big for the target architecture: a type whose size
//!   exceeds the target's object-size bound has no layout; rustc's
//!   layout computation itself reports `values of the type ... are
//!   too big for the target architecture`, and monomorphization
//!   surfaces that failure for every type it must materialize.
//!   The check asks the compiler's own layout query (the same one
//!   the driver's layout table uses) about every local type of a
//!   converted body, and about the type arguments of calls whose
//!   lowering is defined by those layouts: intrinsic calls, which
//!   code generation lowers in the caller from the type arguments'
//!   layouts, and the `core::mem` size/alignment measuring
//!   functions (`size_of`, `align_of`, `size_of_val`, ...), whose
//!   instantiation evaluates the measured type's layout. Only the
//!   layout query's own too-big error counts; any other layout
//!   failure (unsized, parametric, uncomputable here) is left
//!   alone, exactly as for the SIMD layout check.
//!
//! Anything not provable from the converted bodies is left alone:
//! the checks only report a failure when the invalid shape is
//! positively present, mirroring the corresponding rustc error.

use rustc_public::mir::mono::{Instance, InstanceKind};
use rustc_public::mir::{Body, Operand, TerminatorKind};
use rustc_public::ty::{RigidTy, Ty, TyKind};
use rustc_public::{CrateDef, CrateDefType};

/// Scan the Stable MIR bodies that were converted for this program
/// and return the first monomorphization failure rustc would have
/// reported, if any.
pub fn check_bodies(bodies: &[Body]) -> Option<String> {
    let mut layout_memo = std::collections::HashMap::new();
    for body in bodies {
        if let Some(err) = check_body(body, &mut layout_memo) {
            return Some(err);
        }
    }
    None
}

fn check_body(
    body: &Body,
    layout_memo: &mut std::collections::HashMap<Ty, Option<String>>,
) -> Option<String> {
    for local in body.locals() {
        let ty = local.ty;
        if is_simd(ty) && !simd_layout_exists(ty) {
            return Some(format!(
                "the SIMD type `{ty}` has an invalid number of lanes \
                 (zero, or more than the monomorphization lane limit)"
            ));
        }
        if let Some(err) = layout_too_big(ty, layout_memo) {
            return Some(err);
        }
    }
    for bb in &body.blocks {
        let TerminatorKind::Call { func, .. } = &bb.terminator.kind else {
            continue;
        };
        if let Some(err) = check_call(func) {
            return Some(err);
        }
        if let Some(err) = check_measured_call(func, layout_memo) {
            return Some(err);
        }
    }
    None
}

/// The failure rustc reports when the layout of `ty` does not
/// exist because the type is too big for the target architecture,
/// or `None` when the layout exists or fails for any other
/// reason. The layout query can panic inside the compiler for
/// shapes it cannot represent; a panic means the layout could
/// not be determined here, not that it is invalid, so it is
/// treated as "no finding" (the same discipline as the SIMD
/// layout check). Results are memoized per scan: the walked
/// bodies repeat the same types heavily.
fn layout_too_big(
    ty: Ty,
    memo: &mut std::collections::HashMap<Ty, Option<String>>,
) -> Option<String> {
    if let Some(hit) = memo.get(&ty) {
        return hit.clone();
    }
    let found = raw_layout_too_big(ty);
    memo.insert(ty, found.clone());
    found
}

fn raw_layout_too_big(ty: Ty) -> Option<String> {
    if !layout_query_safe(ty) {
        return None;
    }
    let layout = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout()));
    let Err(err) = (match layout {
        Ok(r) => r,
        Err(_) => return None,
    }) else {
        return None;
    };
    // Stable MIR exposes the layout failure only as the
    // compiler's diagnostic text; the too-big failure is the one
    // whose text carries rustc's own wording for this error.
    let text = err.to_string();
    if !text.contains("too big for the target architecture") {
        return None;
    }
    // Report rustc's own diagnostic (the part after the failed
    // query's preamble) rather than a paraphrase.
    match text.find("values of the type") {
        Some(at) => Some(text[at..].to_string()),
        None => Some(text),
    }
}

/// Whether asking the compiler for `ty`'s layout is safe for
/// this scan. The layout query is not total: for a type with an
/// unsized field in a non-final position (accepted under
/// trivial bounds but never materializable) or a projection
/// alias in field position (whose normalization can be
/// ambiguous), the query does not return an error at all — it
/// creates a delayed internal compiler error that poisons the
/// rest of the compilation. The scan therefore walks the type
/// structurally first, asking the compiler nothing: a type is
/// safe when every component in field position is a concrete
/// sized shape, with unsized components only in tail position
/// (the last field of a variant, the last element of a tuple,
/// or nested under either). Pointer and reference types are
/// walked through as well: their layout needs the pointee's
/// metadata, which is computed from the pointee's tail. Type
/// parameters are resolved one frame at a time through the
/// enclosing ADT's arguments —
/// never through the compiler's instantiating-and-normalizing
/// accessors — and anything unresolvable is unsafe. Recursive
/// types are cut off by a visited set: a type already being
/// walked is assumed safe, as its own walk is in progress in
/// a context at least as strict (non-tail occurrences are
/// always walked before the tail occurrence).
fn layout_query_safe(ty: Ty) -> bool {
    let mut visited = std::collections::HashSet::new();
    walk_layout_type(ty, &[], true, &mut visited, 0)
}

/// Walk `ty` structurally. `frames` is the stack of generic
/// argument lists of the ADTs whose fields are being walked,
/// innermost last: a type parameter in `ty` refers to the
/// innermost frame, and the type it resolves to was written
/// one frame further out, so it is walked with the innermost
/// frame popped. `tail` marks tail position (see above).
fn walk_layout_type(
    ty: Ty,
    frames: &[rustc_public::ty::GenericArgs],
    tail: bool,
    visited: &mut std::collections::HashSet<Ty>,
    depth: usize,
) -> bool {
    if depth > 256 {
        return false;
    }
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Adt(def, args)) => {
            if !visited.insert(ty) {
                return true;
            }
            let mut frames = frames.to_vec();
            frames.push(args);
            def.variants().iter().all(|variant| {
                let fields = variant.fields();
                let last = fields.len().saturating_sub(1);
                fields.iter().enumerate().all(|(i, field)| {
                    walk_layout_type(
                        field.ty(),
                        &frames,
                        tail && i == last,
                        visited,
                        depth + 1,
                    )
                })
            })
        }
        TyKind::RigidTy(RigidTy::Tuple(elems)) => {
            let last = elems.len().saturating_sub(1);
            elems.iter().enumerate().all(|(i, elem)| {
                walk_layout_type(*elem, frames, tail && i == last, visited, depth + 1)
            })
        }
        TyKind::RigidTy(RigidTy::Array(elem, _)) => {
            walk_layout_type(elem, frames, false, visited, depth + 1)
        }
        TyKind::RigidTy(RigidTy::Pat(inner, _)) => {
            walk_layout_type(inner, frames, tail, visited, depth + 1)
        }
        TyKind::RigidTy(RigidTy::Ref(_, pointee, _))
        | TyKind::RigidTy(RigidTy::RawPtr(pointee, _)) => {
            walk_layout_type(pointee, frames, true, visited, depth + 1)
        }
        TyKind::RigidTy(RigidTy::Slice(_))
        | TyKind::RigidTy(RigidTy::Str)
        | TyKind::RigidTy(RigidTy::Dynamic(..)) => tail,
        TyKind::Param(param) => {
            let Some((innermost, outer)) = frames.split_last() else {
                return false;
            };
            match innermost.0.get(param.index as usize) {
                Some(rustc_public::ty::GenericArgKind::Type(arg)) => {
                    walk_layout_type(*arg, outer, tail, visited, depth + 1)
                }
                _ => false,
            }
        }
        TyKind::Alias(..) | TyKind::Bound(..) => false,
        TyKind::RigidTy(_) => true,
    }
}

/// Check the type arguments of a call whose lowering is defined
/// by those types' layouts — an intrinsic call, or one of the
/// `core::mem` size/alignment measuring functions — for a type
/// too big for the target architecture. Instantiating such a
/// call evaluates the measured type's layout in rustc, so a
/// too-big argument is the same monomorphization failure as a
/// too-big local. Type arguments of ordinary calls are *not*
/// checked: an ordinary generic function may never materialize
/// a type it merely names (a `PhantomData` parameter is the
/// standard shape), and rustc instantiates it without asking
/// for the type's layout.
fn check_measured_call(
    func: &Operand,
    layout_memo: &mut std::collections::HashMap<Ty, Option<String>>,
) -> Option<String> {
    let Operand::Constant(c) = func else {
        return None;
    };
    let kind = c.const_.ty().kind();
    let TyKind::RigidTy(RigidTy::FnDef(def, args)) = &kind else {
        return None;
    };
    let (def, args) = (*def, args.clone());
    let intrinsic = Instance::resolve(def, &args)
        .map(|inst| matches!(inst.kind, InstanceKind::Intrinsic))
        .unwrap_or(false);
    if !intrinsic && !is_mem_measure_fn(&def.name()) {
        return None;
    }
    for arg in &args.0 {
        if let rustc_public::ty::GenericArgKind::Type(ty) = arg {
            if let Some(err) = layout_too_big(*ty, layout_memo) {
                return Some(err);
            }
        }
    }
    None
}

/// Whether `name` is one of the `core::mem` functions whose
/// result *is* the measured type's layout (`size_of` and
/// `align_of` evaluate the layout as a constant; `size_of_val`
/// and the `min_align_of` family forward to the corresponding
/// intrinsics at instantiation).
fn is_mem_measure_fn(name: &str) -> bool {
    let mut segments = name.rsplit("::");
    let (Some(last), Some(parent)) = (segments.next(), segments.next()) else {
        return false;
    };
    parent == "mem"
        && matches!(
            last,
            "size_of"
                | "align_of"
                | "size_of_val"
                | "align_of_val"
                | "min_align_of"
                | "min_align_of_val"
        )
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

