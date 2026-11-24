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
//! - Invalid atomic intrinsic monomorphizations (also E0511,
//!   from `codegen_intrinsic_call` in rustc_codegen_ssa): the
//!   `atomic_*` intrinsics require the type of the value at the
//!   memory operand to be a basic integer or raw pointer type;
//!   the read-modify-write arithmetic group additionally
//!   requires the operand value to have that same type (or
//!   `usize` when the memory type is a raw pointer).
//! - The per-intrinsic SIMD argument rules of
//!   `generic_simd_intrinsic` beyond the entry check above:
//!   element-type requirements for the saturating arithmetic,
//!   bitmask, gather/scatter, masked load/store and reduction
//!   intrinsics (the ordered reductions' initial value included),
//!   length and element agreement for shuffles,
//!   selects and comparisons, insert/extract element agreement,
//!   and the constant index bounds of `simd_shuffle`,
//!   `simd_insert` and `simd_extract` (the index values are read
//!   from the call's constant operand, the same evidence code
//!   generation reads; named constants reach this check already
//!   evaluated by the compiler's own constant conversion).
//!
//! - Required constants of collected instances (rustc's E0080 at
//!   monomorphization time): the mono collector evaluates every
//!   required constant of every instance it collects — dead code
//!   included — and follows the items a body mentions (calls,
//!   drops, unsizing casts, closures recorded before optimization)
//!   to instances whose own constants it then evaluates the same
//!   way. Stable MIR bodies are post-optimization, so a mention
//!   that lives only in dead code is gone from them and the
//!   failing constant behind it is never demanded; conversion
//!   alone therefore succeeds where rustc fails. The check below
//!   (`check_required_consts`) replays the collector's walk over
//!   the compiler's internal bodies, which keep both the required
//!   constants and the mentioned items from before optimization.
//!
//! Anything not provable from the converted bodies is left alone:
//! the checks only report a failure when the invalid shape is
//! positively present, mirroring the corresponding rustc error.

use rustc_public::mir::mono::{Instance, InstanceKind};
use rustc_public::mir::{Body, Operand, TerminatorKind};
use rustc_public::ty::{AdtKind, RigidTy, Ty, TyKind};
use rustc_public::{CrateDef, CrateDefType};

use crate::convert::guarded_ty_kind;

/// Scan the Stable MIR bodies that were converted for this program
/// and return the first monomorphization failure rustc would have
/// reported, if any.
pub fn check_bodies(bodies: &[Body]) -> Option<String> {
    let mut layout_memo: std::collections::HashMap<(Ty, bool), Option<String>> =
        std::collections::HashMap::new();
    for body in bodies {
        if let Some(err) = check_body(body, &mut layout_memo) {
            return Some(err);
        }
    }
    None
}

fn check_body(
    body: &Body,
    layout_memo: &mut std::collections::HashMap<(Ty, bool), Option<String>>,
) -> Option<String> {
    for local in body.locals() {
        let ty = local.ty;
        if is_simd(ty) && simd_layout_invalid(ty) {
            return Some(format!(
                "the SIMD type `{ty}` has an invalid number of lanes \
                 (zero, or more than the monomorphization lane limit)"
            ));
        }
        if let Some(err) = layout_too_big(ty, layout_memo, false) {
            return Some(err);
        }
    }
    for bb in &body.blocks {
        let TerminatorKind::Call { func, args, .. } = &bb.terminator.kind else {
            continue;
        };
        if let Some(err) = check_call(func, args) {
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
    memo: &mut std::collections::HashMap<(Ty, bool), Option<String>>,
    bypass_screen: bool,
) -> Option<String> {
    if let Some(hit) = memo.get(&(ty, bypass_screen)) {
        return hit.clone();
    }
    let found = raw_layout_too_big(ty, bypass_screen);
    memo.insert((ty, bypass_screen), found.clone());
    found
}

fn raw_layout_too_big(ty: Ty, bypass_screen: bool) -> Option<String> {
    if !bypass_screen && !layout_query_safe(ty) {
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
    // compiler's diagnostic text; the failures this scan reports
    // are the ones whose text carries rustc's own wording for
    // the error — a type too big for the target architecture, or
    // a layout cycle (E0391), which monomorphization surfaces
    // for the same materialized types.
    let text = err.to_string();
    for needle in ["values of the type", "cycle detected when computing layout"] {
        if let Some(at) = text.find(needle) {
            return Some(text[at..].to_string());
        }
    }
    None
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
/// (the last field of a struct or single-variant enum, the
/// last element of a tuple, or nested under either — a
/// multi-variant ADT and a union have no tail position; see
/// the ADT arm of the walk). Pointer and reference types are
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
    // A type passing through a coroutine-closure type cannot be
    // asked its kind (see `guarded_ty_kind`); its layout safety
    // cannot be judged structurally, so it is not safe to query.
    let Some(kind) = guarded_ty_kind(&ty) else {
        return false;
    };
    match kind {
        TyKind::RigidTy(RigidTy::Adt(def, args)) => {
            if !visited.insert(ty) {
                return true;
            }
            // Tail position exists only in a struct-like type.
            // The layout calculator lays out a struct (and a
            // single-variant enum, and a tuple) with at most one
            // unsized field, in last position; but every variant
            // of a multi-variant ADT is laid out prefixed by the
            // discriminant, where an unsized field is always an
            // error, and a union's fields all overlap, so no
            // field is in tail position. Under the monomorphic
            // typing environment this query runs in, that error
            // is not returned — it is recorded as a delayed bug
            // that aborts the whole compilation when flushed
            // (the layout of `Poll<str>` in
            // tests/ui/mir/gvn-nonsensical-sized-str.rs). A
            // possibly unsized field of a multi-variant ADT or
            // a union therefore makes the query unsafe wherever
            // it sits.
            let has_tail = def.variants().len() <= 1 && def.kind() != AdtKind::Union;
            let mut frames = frames.to_vec();
            frames.push(args);
            def.variants().iter().all(|variant| {
                let fields = variant.fields();
                let last = fields.len().saturating_sub(1);
                fields.iter().enumerate().all(|(i, field)| {
                    walk_layout_type(
                        field.ty(),
                        &frames,
                        tail && i == last && has_tail,
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
/// too-big local. The layout query is asked without the
/// structural pre-screen here: rustc itself performs exactly
/// this query when it instantiates the call, so the query
/// cannot fail in a way rustc's own compilation of the same
/// program avoids. Type arguments of ordinary calls are *not*
/// checked: an ordinary generic function may never materialize
/// a type it merely names (a `PhantomData` parameter is the
/// standard shape), and rustc instantiates it without asking
/// for the type's layout.
fn check_measured_call(
    func: &Operand,
    layout_memo: &mut std::collections::HashMap<(Ty, bool), Option<String>>,
) -> Option<String> {
    let Operand::Constant(c) = func else {
        return None;
    };
    let Some(kind) = guarded_ty_kind(&c.const_.ty()) else {
        return None;
    };
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
            if let Some(err) = layout_too_big(*ty, layout_memo, true) {
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
/// Whether the layout query positively reports `ty`'s lane
/// count as invalid -- the failure rustc's own wording names
/// ("has zero elements", "has more elements than the limit").
/// Any other layout outcome is no finding: a layout can be
/// uncomputable here for reasons rustc accepts (a lane count
/// named by a constant the layout query cannot evaluate in
/// this context reports "unknown layout", and rustc's own
/// code generation evaluates it fine), and a panic means the
/// layout could not be determined, not that it is invalid --
/// the same discipline as the too-big check.
fn simd_layout_invalid(ty: Ty) -> bool {
    let layout = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout()));
    let Err(err) = (match layout {
        Ok(r) => r,
        Err(_) => return false,
    }) else {
        return false;
    };
    let text = err.to_string();
    text.contains("has zero elements") || text.contains("has more elements than the limit")
}

/// Whether a type is a SIMD vector type (a `#[repr(simd)]` ADT or
/// `core::simd::Simd`), the notion rustc's own check uses.
fn is_simd(ty: Ty) -> bool {
    // A type passing through a coroutine-closure type cannot be
    // asked its kind (see `guarded_ty_kind`) and is not a SIMD
    // type this check can judge.
    guarded_ty_kind(&ty).is_some_and(|kind| kind.is_simd())
}

fn check_call(func: &Operand, call_args: &[Operand]) -> Option<String> {
    let Operand::Constant(c) = func else {
        return None;
    };
    let Some(kind) = guarded_ty_kind(&c.const_.ty()) else {
        return None;
    };
    let TyKind::RigidTy(RigidTy::FnDef(def, args)) = &kind else {
        return None;
    };
    let (def, args) = (*def, args.clone());
    // Only compiler intrinsics are checked: among them the
    // `simd_*` and `atomic_*` families carry monomorphization
    // rules of this shape.
    let inst = Instance::resolve(def, &args).ok()?;
    if !matches!(inst.kind, InstanceKind::Intrinsic) {
        return None;
    }
    let full_name = def.name();
    let name = full_name.rsplit("::").next().unwrap_or(&full_name);
    let sig = kind.fn_sig()?;
    let sig = &sig.value;
    let inputs = sig.inputs();
    if name.starts_with("atomic_") {
        return check_atomic(name, inputs);
    }
    if !name.starts_with("simd_") {
        return None;
    }
    // Mirror `generic_simd_intrinsic`'s entry checks: the vector
    // argument whose type must be a SIMD type differs for the two
    // intrinsics handled before the general case.
    let (vector_ty, what) = match name {
        "simd_splat" => (Some(sig.output()), "return"),
        "simd_select_bitmask" => (inputs.get(1).copied(), "second argument"),
        _ => (inputs.first().copied(), "input"),
    };
    let vector_ty = vector_ty?;
    if !is_simd(vector_ty) {
        return Some(format!(
            "invalid monomorphization of `{name}` intrinsic: \
             expected SIMD {what} type, found non-SIMD `{vector_ty}`"
        ));
    }
    check_simd_family(name, inputs, sig.output(), call_args)
}

/// Mirror the E0511 checks rustc_codegen_ssa performs for the
/// `atomic_*` intrinsics (`codegen_intrinsic_call`): the type of
/// the value at the memory operand — the pointee of the first
/// argument, which is `*const T` for loads and `*mut T` for the
/// rest — must be a basic integer type (an integer of any width,
/// pointer-sized included; `bool` and `char` do not qualify) or
/// a raw pointer type. The read-modify-write arithmetic group
/// (`atomic_xadd` and its siblings, whose second argument is the
/// operand value) additionally requires the operand's type to
/// equal the memory type, or to be `usize` when the memory type
/// is a raw pointer. `atomic_fence` and
/// `atomic_singlethreadfence` carry no type rule and are left
/// alone, as is any shape not positively present.
fn check_atomic(name: &str, inputs: &[Ty]) -> Option<String> {
    let dst = inputs.first().copied()?;
    let Some(dst_kind) = guarded_ty_kind(&dst) else {
        return None;
    };
    let TyKind::RigidTy(RigidTy::RawPtr(pointee, _)) = dst_kind else {
        return None;
    };
    let invalid = || {
        Some(format!(
            "invalid monomorphization of `{name}` intrinsic: \
             expected basic integer or pointer type, found `{pointee}`"
        ))
    };
    match name {
        "atomic_load" | "atomic_store" | "atomic_xchg" | "atomic_cxchg"
        | "atomic_cxchgweak" | "atomic_max" | "atomic_min" | "atomic_umax"
        | "atomic_umin" => {
            if is_int_or_uint(pointee)
                || guarded_ty_kind(&pointee)
                    .is_some_and(|k| matches!(k, TyKind::RigidTy(RigidTy::RawPtr(..))))
            {
                None
            } else {
                invalid()
            }
        }
        "atomic_xadd" | "atomic_xsub" | "atomic_and" | "atomic_or" | "atomic_xor"
        | "atomic_nand" => {
            let val = inputs.get(1).copied()?;
            let ok = (is_int_or_uint(pointee) && val == pointee)
                || (guarded_ty_kind(&pointee)
                    .is_some_and(|k| matches!(k, TyKind::RigidTy(RigidTy::RawPtr(..))))
                    && guarded_ty_kind(&val).is_some_and(|k| {
                        matches!(
                            k,
                            TyKind::RigidTy(RigidTy::Uint(rustc_public::ty::UintTy::Usize))
                        )
                    }));
            if ok { None } else { invalid() }
        }
        _ => None,
    }
}

/// Whether a type is an integer type (signed or unsigned, any
/// width, pointer-sized included).
fn is_int_or_uint(ty: Ty) -> bool {
    guarded_ty_kind(&ty).is_some_and(|k| {
        matches!(k, TyKind::RigidTy(RigidTy::Int(_) | RigidTy::Uint(_)))
    })
}

/// The shape of a SIMD vector type as `(lane count, element
/// type)`, read from the single array field every `#[repr(simd)]`
/// type is built on, or from several fields all of the same type;
/// `None` when the type is not a SIMD type or its lane count is
/// not a concrete value here.
fn simd_shape(ty: Ty) -> Option<(u64, Ty)> {
    if !is_simd(ty) {
        return None;
    }
    let Some(ty_kind) = guarded_ty_kind(&ty) else {
        return None;
    };
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty_kind else {
        return None;
    };
    let variants = def.variants();
    let [variant] = variants.as_slice() else {
        return None;
    };
    let fields = variant.fields();
    let [field] = fields.as_slice() else {
        // A `#[repr(simd)]` type may instead be several fields
        // all of the same type `T`; its shape is then the field
        // count and `T`, read with the instance's arguments
        // substituted.
        if fields.len() < 2 {
            return None;
        }
        let elem = fields[0].ty_with_args(&args);
        if fields.iter().all(|f| f.ty_with_args(&args) == elem) {
            return Some((fields.len() as u64, elem));
        }
        return None;
    };
    // The field's declared type is written in terms of the
    // ADT's own generic parameters (`Simd<T, N>`'s field is
    // `[T; N]`); substitute the instance's arguments, the same
    // substitution code generation works with.
    let Some(field_kind) = guarded_ty_kind(&field.ty()) else {
        return None;
    };
    let TyKind::RigidTy(RigidTy::Array(elem, len)) = field_kind else {
        return None;
    };
    let elem = subst_param(elem, &args);
    // The lane count, resolving a count written as the ADT's
    // own const parameter through the instance's arguments.
    let lanes = match len.eval_target_usize() {
        Ok(n) => n,
        Err(_) => {
            let rustc_public::ty::TyConstKind::Param(p) = len.kind() else {
                return None;
            };
            match args.0.get(p.index as usize) {
                Some(rustc_public::ty::GenericArgKind::Const(inner)) => {
                    inner.eval_target_usize().ok()?
                }
                _ => return None,
            }
        }
    };
    Some((lanes, elem))
}

/// Resolve a type written as one of the enclosing ADT's type
/// parameters through the instance's generic arguments; any
/// other type is returned unchanged.
fn subst_param(ty: Ty, args: &rustc_public::ty::GenericArgs) -> Ty {
    match guarded_ty_kind(&ty) {
        Some(TyKind::Param(p)) => match args.0.get(p.index as usize) {
            Some(rustc_public::ty::GenericArgKind::Type(t)) => *t,
            _ => ty,
        },
        _ => ty,
    }
}

/// The bit width of an integer type, from its layout (the layout
/// of a primitive integer always exists); `None` when it cannot
/// be determined here.
fn int_bit_width(ty: Ty) -> Option<u64> {
    if !is_int_or_uint(ty) {
        return None;
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ty.layout())) {
        Ok(Ok(layout)) => Some(layout.shape().size.bytes() as u64 * 8),
        _ => None,
    }
}

/// The `u32` elements of a constant index operand, read from the
/// operand's allocation — the same values code generation reads
/// when it checks `simd_shuffle` indices. `None` when the
/// operand is not a constant whose allocation is exactly the
/// index vector's bytes.
fn const_u32_vector(op: &Operand, lanes: u64) -> Option<Vec<u32>> {
    use rustc_public::ty::ConstantKind as CK;
    let Operand::Constant(c) = op else {
        return None;
    };
    let read = |alloc: &rustc_public::ty::Allocation| -> Option<Vec<u32>> {
        if alloc.bytes.len() as u64 != 4 * lanes {
            return None;
        }
        (0..lanes)
            .map(|i| {
                let at = (4 * i) as usize;
                alloc.read_partial_uint(at..at + 4).ok().map(|v| v as u32)
            })
            .collect()
    };
    match c.const_.kind() {
        CK::Allocated(alloc) => read(&alloc),
        CK::Ty(tc) => match tc.kind() {
            rustc_public::ty::TyConstKind::Value(_, alloc) => read(&alloc),
            _ => None,
        },
        _ => None,
    }
}

/// The value of a constant scalar index operand (the `u32`
/// index of `simd_insert` / `simd_extract`), read from the
/// operand's allocation. `None` when the operand is not a
/// constant with a readable 4-byte value.
fn const_u32_scalar(op: &Operand) -> Option<u32> {
    use rustc_public::ty::ConstantKind as CK;
    let Operand::Constant(c) = op else {
        return None;
    };
    let read = |alloc: &rustc_public::ty::Allocation| -> Option<u32> {
        alloc.read_partial_uint(0..4).ok().map(|v| v as u32)
    };
    match c.const_.kind() {
        CK::Allocated(alloc) => read(&alloc),
        CK::Ty(tc) => match tc.kind() {
            rustc_public::ty::TyConstKind::Value(_, alloc) => read(&alloc),
            _ => None,
        },
        _ => None,
    }
}

/// The per-intrinsic argument rules of `generic_simd_intrinsic`
/// (rustc_codegen_llvm) beyond the entry check in `check_call`:
/// element-type requirements, length and element agreement
/// between arguments and the return type, and the constant
/// index bounds of `simd_shuffle`. Each rule fires only when
/// the invalid shape is positively present in the call's
/// substituted signature (or, for the shuffle bounds, in the
/// call's constant index operand).
fn check_simd_family(
    name: &str,
    inputs: &[Ty],
    ret_ty: Ty,
    call_args: &[Operand],
) -> Option<String> {
    let input = |i: usize| inputs.get(i).copied();
    let err = |msg: String| {
        Some(format!(
            "invalid monomorphization of `{name}` intrinsic: {msg}"
        ))
    };
    let mask_elem_err = |elem: Ty| {
        err(format!(
            "expected mask element type to be an integer, found `{elem}`"
        ))
    };
    match name {
        "simd_saturating_add" | "simd_saturating_sub" => {
            let in_ty = input(0)?;
            let (_, elem) = simd_shape(in_ty)?;
            if is_int_or_uint(elem) {
                None
            } else {
                err(format!(
                    "expected element type `{elem}` of vector type `{in_ty}` \
                     to be a signed or unsigned integer type"
                ))
            }
        }
        "simd_bitmask" => {
            let in_ty = input(0)?;
            let (len, elem) = simd_shape(in_ty)?;
            if !is_int_or_uint(elem) {
                return mask_elem_err(elem);
            }
            // The return must be an unsigned integer with
            // `max(len, 8).next_power_of_two()` bits, or a byte
            // array with `ceil(len / 8)` elements.
            let expected_bits = len.max(8).next_power_of_two();
            let ok = match guarded_ty_kind(&ret_ty) {
                Some(TyKind::RigidTy(RigidTy::Uint(_))) => {
                    int_bit_width(ret_ty) == Some(expected_bits)
                }
                Some(TyKind::RigidTy(RigidTy::Array(e, n))) => {
                    guarded_ty_kind(&e).is_some_and(|k| {
                        matches!(
                            k,
                            TyKind::RigidTy(RigidTy::Uint(rustc_public::ty::UintTy::U8))
                        )
                    }) && n.eval_target_usize().ok() == Some(len.div_ceil(8))
                }
                _ => false,
            };
            if ok {
                None
            } else {
                err(format!(
                    "cannot return `{ret_ty}`, expected `u{expected_bits}` \
                     or `[u8; {}]`",
                    len.div_ceil(8)
                ))
            }
        }
        "simd_shuffle" => {
            let in_ty = input(0)?;
            let (in_len, in_elem) = simd_shape(in_ty)?;
            let idx_ty = input(2)?;
            let Some((n, idx_elem)) = simd_shape(idx_ty) else {
                return err(
                    "simd_shuffle index must be a SIMD vector of `u32` elements"
                        .to_string(),
                );
            };
            if !guarded_ty_kind(&idx_elem).is_some_and(|k| {
                matches!(
                    k,
                    TyKind::RigidTy(RigidTy::Uint(rustc_public::ty::UintTy::U32))
                )
            }) {
                return err(
                    "simd_shuffle index must be a SIMD vector of `u32` elements"
                        .to_string(),
                );
            }
            if !is_simd(ret_ty) {
                return err(format!(
                    "expected SIMD return type, found non-SIMD `{ret_ty}`"
                ));
            }
            let (out_len, out_elem) = simd_shape(ret_ty)?;
            if out_len != n {
                return err(format!(
                    "expected return type with length {n} (same as the \
                     index type `{idx_ty}`), found `{ret_ty}` with \
                     length {out_len}"
                ));
            }
            if out_elem != in_elem {
                return err(format!(
                    "expected return element type `{in_elem}` (element of \
                     input `{in_ty}`), found `{out_elem}`"
                ));
            }
            if let Some(indices) = const_u32_vector(call_args.get(2)?, n) {
                let total = in_len * 2;
                for (i, v) in indices.iter().enumerate() {
                    if u64::from(*v) >= total {
                        return err(format!(
                            "expected index {i} to be less than {total}, \
                             found {v}"
                        ));
                    }
                }
            }
            None
        }
        "simd_gather" | "simd_scatter" => {
            let in_ty = input(0)?;
            let (in_len, in_elem) = simd_shape(in_ty)?;
            let ptr_ty = input(1)?;
            if !is_simd(ptr_ty) {
                return err(format!(
                    "expected SIMD second argument type, found \
                     non-SIMD `{ptr_ty}`"
                ));
            }
            let (ptr_len, ptr_elem) = simd_shape(ptr_ty)?;
            if ptr_len != in_len {
                return err(format!(
                    "expected second argument with length {in_len} (same \
                     as input type `{in_ty}`), found `{ptr_ty}` with \
                     length {ptr_len}"
                ));
            }
            let mask_ty = input(2)?;
            if !is_simd(mask_ty) {
                return err(format!(
                    "expected SIMD third argument type, found \
                     non-SIMD `{mask_ty}`"
                ));
            }
            let (mask_len, mask_elem) = simd_shape(mask_ty)?;
            if mask_len != in_len {
                return err(format!(
                    "expected third argument with length {in_len} (same \
                     as input type `{in_ty}`), found `{mask_ty}` with \
                     length {mask_len}"
                ));
            }
            if name == "simd_gather" {
                if !is_simd(ret_ty) {
                    return err(format!(
                        "expected SIMD return type, found non-SIMD `{ret_ty}`"
                    ));
                }
                if ret_ty != in_ty {
                    return err(format!(
                        "expected return type `{in_ty}`, found `{ret_ty}`"
                    ));
                }
            }
            // The pointer vector's elements must be pointers to
            // the input's element type. Code generation also
            // requires mutable pointers for scatter, but that
            // requirement is enforced only when a call is
            // lowered, and this scan also covers constant bodies
            // code generation never lowers (a masked store
            // through a `*const` in an unevaluated constant is
            // accepted by rustc), so mutability is not checked
            // here.
            let ptr_ok = match guarded_ty_kind(&ptr_elem) {
                Some(TyKind::RigidTy(RigidTy::RawPtr(p, _))) => p == in_elem,
                _ => false,
            };
            if !ptr_ok {
                return err(format!(
                    "expected element type of second argument `{ptr_ty}` \
                     to be a pointer to the element type `{in_elem}`"
                ));
            }
            if !is_int_or_uint(mask_elem) {
                return mask_elem_err(mask_elem);
            }
            None
        }
        "simd_masked_load" | "simd_masked_store" => {
            let mask_ty = input(0)?;
            let (mask_len, mask_elem) = simd_shape(mask_ty)?;
            let values_ty = input(2)?;
            if !is_simd(values_ty) {
                return err(format!(
                    "expected SIMD third argument type, found \
                     non-SIMD `{values_ty}`"
                ));
            }
            let (values_len, values_elem) = simd_shape(values_ty)?;
            if values_len != mask_len {
                return err(format!(
                    "expected third argument with length {mask_len} \
                     (same as input type `{mask_ty}`), found \
                     `{values_ty}` with length {values_len}"
                ));
            }
            if name == "simd_masked_load" && ret_ty != values_ty {
                return err(format!(
                    "expected return type `{values_ty}`, found `{ret_ty}`"
                ));
            }
            // The pointer must point at the values' element
            // type. (Pointer mutability is not checked, for the
            // reason given at the gather/scatter check above.)
            let pointer_ty = input(1)?;
            let ptr_ok = match guarded_ty_kind(&pointer_ty) {
                Some(TyKind::RigidTy(RigidTy::RawPtr(p, _))) => p == values_elem,
                _ => false,
            };
            if !ptr_ok {
                return err(format!(
                    "expected element type of second argument \
                     `{pointer_ty}` to be a pointer to the element \
                     type `{values_elem}`"
                ));
            }
            if !is_int_or_uint(mask_elem) {
                return mask_elem_err(mask_elem);
            }
            None
        }
        "simd_reduce_add_ordered" | "simd_reduce_mul_ordered"
        | "simd_reduce_add_unordered" | "simd_reduce_mul_unordered" => {
            let in_ty = input(0)?;
            let (_, elem) = simd_shape(in_ty)?;
            // The ordered pair takes the initial value as its
            // second argument, and it must have the element type.
            if matches!(
                name,
                "simd_reduce_add_ordered" | "simd_reduce_mul_ordered"
            ) {
                if let Some(initial) = input(1) {
                    if initial != elem {
                        return err(format!(
                            "expected initial value type `{elem}` \
                             (element of input `{in_ty}`), found \
                             `{initial}`"
                        ));
                    }
                }
            }
            if ret_ty != elem {
                return err(format!(
                    "expected return type `{elem}` (element of input \
                     `{in_ty}`), found `{ret_ty}`"
                ));
            }
            if is_int_or_uint(elem)
                || guarded_ty_kind(&elem)
                    .is_some_and(|k| matches!(k, TyKind::RigidTy(RigidTy::Float(_))))
            {
                None
            } else {
                err(format!(
                    "unsupported {name} from `{in_ty}` with element \
                     `{elem}` to `{ret_ty}`"
                ))
            }
        }
        "simd_reduce_min" | "simd_reduce_max" | "simd_reduce_and"
        | "simd_reduce_or" | "simd_reduce_xor" => {
            let in_ty = input(0)?;
            let (_, elem) = simd_shape(in_ty)?;
            if ret_ty != elem {
                return err(format!(
                    "expected return type `{elem}` (element of input \
                     `{in_ty}`), found `{ret_ty}`"
                ));
            }
            if is_int_or_uint(elem) {
                None
            } else {
                err(format!(
                    "unsupported {name} from `{in_ty}` with element \
                     `{elem}` to `{ret_ty}`"
                ))
            }
        }
        "simd_reduce_all" | "simd_reduce_any" => {
            let in_ty = input(0)?;
            let (_, elem) = simd_shape(in_ty)?;
            if is_int_or_uint(elem) {
                None
            } else {
                err(format!(
                    "unsupported {name} from `{in_ty}` with element \
                     `{elem}` to `{ret_ty}`"
                ))
            }
        }
        "simd_eq" | "simd_ne" | "simd_lt" | "simd_le" | "simd_gt" | "simd_ge" => {
            let in_ty = input(0)?;
            let (in_len, _) = simd_shape(in_ty)?;
            if !is_simd(ret_ty) {
                return err(format!(
                    "expected SIMD return type, found non-SIMD `{ret_ty}`"
                ));
            }
            let (out_len, out_elem) = simd_shape(ret_ty)?;
            if out_len != in_len {
                return err(format!(
                    "expected return type with length {in_len} (same as \
                     input type `{in_ty}`), found `{ret_ty}` with \
                     length {out_len}"
                ));
            }
            if !is_int_or_uint(out_elem) {
                return err(format!(
                    "expected return element type to be an integer, \
                     found `{out_elem}`"
                ));
            }
            None
        }
        "simd_select" => {
            let mask_ty = input(0)?;
            let (m_len, m_elem) = simd_shape(mask_ty)?;
            let v_ty = input(1)?;
            if !is_simd(v_ty) {
                return err(format!(
                    "expected SIMD second argument type, found \
                     non-SIMD `{v_ty}`"
                ));
            }
            let (v_len, _) = simd_shape(v_ty)?;
            if m_len != v_len {
                return err(format!(
                    "expected mask length {m_len} to match the vector \
                     length {v_len} of `{v_ty}`"
                ));
            }
            if !is_int_or_uint(m_elem) {
                return mask_elem_err(m_elem);
            }
            None
        }
        "simd_splat" => {
            let (_, out_elem) = simd_shape(ret_ty)?;
            let val_ty = input(0)?;
            if val_ty == out_elem {
                None
            } else {
                err(format!(
                    "expected element type `{out_elem}` of vector type \
                     `{ret_ty}`, found the splatted value's type \
                     `{val_ty}`"
                ))
            }
        }
        "simd_select_bitmask" => {
            let vec_ty = input(1)?;
            let (len, _) = simd_shape(vec_ty)?;
            let mask_ty = input(0)?;
            // Same accepted mask shapes as `simd_bitmask`'s
            // return: an unsigned integer with
            // `max(len, 8).next_power_of_two()` bits, or a byte
            // array with `ceil(len / 8)` elements.
            let expected_bits = len.max(8).next_power_of_two();
            let ok = match guarded_ty_kind(&mask_ty) {
                Some(TyKind::RigidTy(RigidTy::Int(_)))
                | Some(TyKind::RigidTy(RigidTy::Uint(_))) => {
                    int_bit_width(mask_ty) == Some(expected_bits)
                }
                Some(TyKind::RigidTy(RigidTy::Array(e, n))) => {
                    guarded_ty_kind(&e).is_some_and(|k| {
                        matches!(
                            k,
                            TyKind::RigidTy(RigidTy::Uint(rustc_public::ty::UintTy::U8))
                        )
                    }) && n.eval_target_usize().ok() == Some(len.div_ceil(8))
                }
                _ => false,
            };
            if ok {
                None
            } else {
                err(format!(
                    "invalid bitmask `{mask_ty}`, expected `u{expected_bits}` \
                     or `[u8; {}]`",
                    len.div_ceil(8)
                ))
            }
        }
        "simd_insert" | "simd_insert_dyn" => {
            let in_ty = input(0)?;
            let (lanes, elem) = simd_shape(in_ty)?;
            let val_ty = input(2)?;
            if val_ty != elem {
                return err(format!(
                    "expected inserted type `{elem}` (element of input \
                     `{in_ty}`), found `{val_ty}`"
                ));
            }
            if let Some(idx) = call_args.get(1).and_then(const_u32_scalar) {
                if u64::from(idx) >= lanes {
                    return err(format!(
                        "expected index to be less than {lanes}, \
                         found {idx}"
                    ));
                }
            }
            None
        }
        "simd_extract" | "simd_extract_dyn" => {
            let in_ty = input(0)?;
            let (lanes, elem) = simd_shape(in_ty)?;
            if ret_ty != elem {
                return err(format!(
                    "expected return type `{elem}` (element of input \
                     `{in_ty}`), found `{ret_ty}`"
                ));
            }
            if let Some(idx) = call_args.get(1).and_then(const_u32_scalar) {
                if u64::from(idx) >= lanes {
                    return err(format!(
                        "expected index to be less than {lanes}, \
                         found {idx}"
                    ));
                }
            }
            None
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------
// Required constants of collected instances
// ---------------------------------------------------------------------

/// Upper bound on the instances one required-constant walk visits.
/// Conversion itself collects at most a few thousand instance
/// bodies per program; the walk additionally follows mentions into
/// instances conversion never demanded (dead-code mentions), and
/// the cap keeps a pathological mention graph from turning the
/// check into a second full collection. Reaching the cap only
/// narrows the check, never widens it.
const MAX_REQUIRED_CONST_WALK: usize = 8192;

fn fully_monomorphized<'tcx>() -> rustc_middle::ty::TypingEnv<'tcx> {
    rustc_middle::ty::TypingEnv::fully_monomorphized()
}

/// Mirror of the constant evaluation rustc's monomorphization
/// collector performs (`MirUsedCollector` in rustc_monomorphize):
/// for every instance it collects, the collector evaluates the
/// body's required constants — every unevaluated constant operand
/// of the pre-optimization body, dead code included — and a
/// constant whose evaluation is already reported as an error
/// (E0080) fails the whole compilation. It reaches those bodies
/// not only through live calls: the items a body *mentions*
/// (calls, drops, unsizing casts and closure coercions recorded
/// before optimization, so dead code still mentions them) and the
/// function pointers inside evaluated constant values are all
/// collected and checked the same way.
///
/// The driver's own collection walks Stable MIR bodies, which are
/// post-optimization: a mention that lives only in dead code is
/// gone from them, and the failing constant it leads to is never
/// demanded, so conversion succeeds where rustc fails (the
/// `tests/ui/consts/required-consts` family). This check replays
/// the collector's walk over the compiler's internal bodies —
/// which keep both the required constants and the mentioned items
/// from before optimization — starting from the same roots
/// the collector itself starts from (see
/// `RequiredConstWalk::collect_roots`).
///
/// Only a positively reported evaluation error is a finding: a
/// constant that stays too generic after substitution, an instance
/// whose body the compiler cannot produce here, and any internal
/// panic while walking are all "could not determine", never a
/// failure — the same discipline as the layout check above.
/// (Evaluating a failing constant also emits rustc's own E0080
/// diagnostic into the session, which is how the failure surfaces
/// in practice — the same emission that fails the compilation
/// under rustc; the returned string is the driver's own record of
/// the finding.)
pub fn check_required_consts() -> Option<String> {
    let tcx = crate::convert::compiler_tcx()?;
    // A lint error recorded at a body materialization is a
    // finding of its own: rustc lints every local body, so the
    // body need not be one the collector collects for the
    // diagnostic to be a compilation error (see
    // `convert::with_body_materialization`).
    if let Some(key) = crate::convert::materialization_lint() {
        return Some(format!(
            "materialization of `{key}` emitted a lint error \
             (rustc lints every local body; see the diagnostics \
             above)"
        ));
    }
    let mut walk = RequiredConstWalk {
        tcx,
        visited: std::collections::HashSet::new(),
        visited_allocs: std::collections::HashSet::new(),
        queue: std::collections::VecDeque::new(),
        finding: None,
    };
    walk.collect_roots();
    walk.run();
    walk.finding
}

struct RequiredConstWalk<'tcx> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    visited: std::collections::HashSet<rustc_middle::ty::Instance<'tcx>>,
    visited_allocs: std::collections::HashSet<rustc_middle::mir::interpret::AllocId>,
    queue: std::collections::VecDeque<rustc_middle::ty::Instance<'tcx>>,
    finding: Option<String>,
}

impl<'tcx> RequiredConstWalk<'tcx> {
    fn run(&mut self) {
        while let Some(instance) = self.queue.pop_front() {
            if self.finding.is_some() {
                return;
            }
            // Instance-body access and constant evaluation go
            // through the compiler's internals and can panic on
            // exotic instances (foreign bodies with no MIR here,
            // shim shapes the queries do not expect). A panic
            // abandons that instance's walk, never the check.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.process(instance);
            }));
        }
    }

    fn enqueue(&mut self, instance: rustc_middle::ty::Instance<'tcx>) {
        use rustc_middle::ty::InstanceKind;
        match instance.def {
            // Intrinsics have no body to walk, and a virtual
            // instance's body is the trait definition's (its
            // constants are only evaluated for the concrete
            // instances the walk reaches separately).
            InstanceKind::Intrinsic(_)
            | InstanceKind::LlvmIntrinsic(_)
            | InstanceKind::Virtual(_, _) => return,
            _ => {}
        }
        if self.visited.len() >= MAX_REQUIRED_CONST_WALK {
            return;
        }
        if self.visited.insert(instance) {
            self.queue.push_back(instance);
        }
    }

    /// Seed the walk with the roots the monomorphization
    /// collector itself starts from (`RootCollector` in
    /// rustc_monomorphize, under the lazy collection strategy —
    /// the eager strategy is selected only by `-Zlink-dead-code`,
    /// which the driver does not pass):
    ///
    /// - the crate's entry point, when it has one;
    /// - every non-generic local function the reachability
    ///   analysis marks as a root — the entry point, exported and
    ///   `#[used]` functions, and the non-generic functions they
    ///   reference — plus functions carrying the internal-symbol
    ///   flags the collector also treats as roots;
    /// - every local static: collecting a static evaluates its
    ///   initializer, and a non-nested static's type also gets
    ///   its drop glue collected.
    ///
    /// The instances the driver's own demand passes resolved are
    /// NOT roots. Conversion translates every local body,
    /// reachable or not, so a demand originating in an
    /// unreferenced private function would seed the walk with an
    /// instance rustc never collects, and a failing constant
    /// inside it would fail a compilation rustc accepts.
    /// Everything the collector does collect is reached from
    /// these roots by the mention walk, exactly as in the
    /// compiler.
    fn collect_roots(&mut self) {
        let tcx = self.tcx;
        if let Some((entry_def_id, _)) = tcx.entry_fn(()) {
            self.enqueue(rustc_middle::ty::Instance::mono(tcx, entry_def_id));
        }
        for local_def_id in tcx.iter_local_def_id() {
            if self.finding.is_some() {
                return;
            }
            let def_id = local_def_id.to_def_id();
            // The root queries can panic on exotic items; a
            // panic abandons that root, never the check (the
            // same discipline as the per-instance walk in `run`).
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // Foreign items (the contents of `extern`
                // blocks) are never roots: the collector's item
                // visitor does not process them, an extern
                // function has no body to collect, and an extern
                // static has no initializer to evaluate (asking
                // for one is an internal compiler error).
                if tcx.is_foreign_item(def_id) {
                    return;
                }
                match tcx.def_kind(def_id) {
                    rustc_hir::def::DefKind::Fn | rustc_hir::def::DefKind::AssocFn => {
                        if tcx.generics_of(def_id).requires_monomorphization(tcx) {
                            return;
                        }
                        let flags = tcx.codegen_fn_attrs(def_id).flags;
                        if tcx.is_reachable_non_generic(def_id)
                            || flags.intersects(
                                rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL
                                    | rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags::EXTERNALLY_IMPLEMENTABLE_ITEM,
                            )
                        {
                            self.enqueue(rustc_middle::ty::Instance::mono(tcx, def_id));
                        }
                    }
                    rustc_hir::def::DefKind::Static { nested, .. } => {
                        self.collect_root_static(def_id, nested);
                    }
                    _ => {}
                }
            }));
        }
    }

    /// Collect one root static the way the collector visits a
    /// `MonoItem::Static`: evaluate its initializer and follow
    /// the allocations it points into (an evaluation error
    /// reported there is the same failure class as a required
    /// constant's), and collect the drop glue of a non-nested
    /// static's type.
    fn collect_root_static(&mut self, def_id: rustc_span::def_id::DefId, nested: bool) {
        let tcx = self.tcx;
        if !nested {
            let instance = rustc_middle::ty::Instance::mono(tcx, def_id);
            let ty = instance.ty(tcx, fully_monomorphized());
            let glue = rustc_middle::ty::Instance::resolve_drop_glue(tcx, ty);
            self.enqueue(glue);
        }
        match tcx.eval_static_initializer(def_id) {
            Ok(alloc) => self.chase_allocation(alloc),
            Err(rustc_middle::mir::interpret::ErrorHandled::Reported(..)) => {
                self.finding = Some(format!(
                    "initializer of static `{}` failed to evaluate \
                     (rustc reports the evaluation error during \
                     monomorphization collection)",
                    tcx.def_path_str(def_id)
                ));
            }
            Err(rustc_middle::mir::interpret::ErrorHandled::TooGeneric(_)) => {}
        }
    }

    /// Instantiate a value read from `instance`'s body with the
    /// instance's arguments, normalizing projections away. The
    /// collector does this with the `try_` variant and simply
    /// does not collect an item that fails to normalize (the
    /// panicking variant reports an internal error instead), so
    /// a failure here yields `None` and the item is skipped.
    fn monomorphize<T>(&self, instance: rustc_middle::ty::Instance<'tcx>, value: T) -> Option<T>
    where
        T: rustc_middle::ty::TypeFoldable<rustc_middle::ty::TyCtxt<'tcx>>,
    {
        instance
            .try_instantiate_mir_and_normalize_erasing_regions(
                self.tcx,
                fully_monomorphized(),
                rustc_middle::ty::EarlyBinder::bind(self.tcx, value),
            )
            .ok()
    }

    fn process(&mut self, instance: rustc_middle::ty::Instance<'tcx>) {
        let tcx = self.tcx;
        // The collector only walks instances it would codegen
        // locally: a foreign instance's constants were evaluated
        // when its home crate was compiled (or never, if it was
        // never collected there either), and its body may not be
        // available here at all.
        if !tcx.should_codegen_locally(instance) {
            return;
        }
        // A body the collector collects is one rustc itself
        // materializes in its own pipeline, so the evaluation
        // diagnostics of materializing it are compilation errors
        // there too: the validity errors of its constant
        // operands are the shape this check adds (evaluation
        // errors proper are also found by the required-constant
        // evaluation below, and lint errors are recorded
        // separately, for every local body). The driver's speculative
        // body fetches discount those diagnostics at the fetch
        // (a body the collector never reaches must not fail the
        // compilation — Issue #417) but record the body's key;
        // a hit here turns the record into the verdict. Both the
        // instance's own key and its definition's are consulted:
        // a generic definition's body is materialized once,
        // under the definition's key, and its diagnostics are
        // properties of every instance of it.
        {
            use rustc_middle::ty::print::{with_no_trimmed_paths, with_resolve_crate_name};
            let key = with_resolve_crate_name!(with_no_trimmed_paths!(
                tcx.def_path_str_with_args(instance.def.def_id(), instance.args)
            ));
            let def_key = with_resolve_crate_name!(with_no_trimmed_paths!(
                tcx.def_path_str(instance.def.def_id())
            ));
            if crate::convert::materialization_emitted(&key)
                || crate::convert::materialization_emitted(&def_key)
            {
                self.finding = Some(format!(
                    "materialization of `{key}` emitted diagnostics \
                     (rustc reports them for the bodies it collects; \
                     see the diagnostics above)"
                ));
                return;
            }
            // The same discipline for demand-path instance
            // resolution (see `convert::resolve_instance_guarded`),
            // with one difference: only the instance's own key
            // is consulted. A materialization diagnostic is a
            // property of the one generic body fetched per
            // definition, but a resolution's outcome depends on
            // the generic arguments attempted, so a record keyed
            // by the definition would condemn instantiations
            // that resolve cleanly.
            if crate::convert::resolution_emitted(&key) {
                self.finding = Some(format!(
                    "resolution of `{key}` emitted diagnostics \
                     (rustc reports them for the instances it collects; \
                     see the diagnostics above)"
                ));
                return;
            }
        }
        // Shim instances are handled by kind. Drop glue and
        // fn-pointer shim bodies are walked: a type's destructor
        // and field drops live in the former, and the latter's
        // body is what calls the pointee instance. The reify and
        // closure `call_once` shims are instead followed to the
        // instance they forward to — building a reify shim's
        // body for a closure is not even safe (it asks for the
        // closure's `fn_sig`, which the compiler refuses with
        // "use `args.as_closure().sig()`"), while the target
        // instance is the one whose constants the collector
        // evaluates. The remaining shims are synthetic thunks
        // whose targets the walk reaches by other edges (vtable
        // methods via unsizing casts) or not at all
        // (data-dependent callees).
        if let rustc_middle::ty::InstanceKind::Shim(shim) = instance.def {
            use rustc_middle::ty::ShimKind;
            match shim {
                ShimKind::DropGlue(..) | ShimKind::AsyncDropGlue(..) | ShimKind::FnPtr(..) => {}
                ShimKind::Reify(def_id, _) => {
                    self.enqueue_resolved(def_id, instance.args);
                    return;
                }
                ShimKind::ClosureOnce { closure, .. } => {
                    self.enqueue_resolved(closure, instance.args);
                    return;
                }
                _ => return,
            }
        }
        let body = tcx.instance_mir(instance.def);
        self.process_body(instance, body);
    }

    /// Resolve `(def_id, args)` the way the collector resolves a
    /// mentioned function item and queue the instance for the
    /// walk. Unresolvable or generic instances are not collected
    /// by the compiler either, so they are simply not queued.
    fn enqueue_resolved(
        &mut self,
        def_id: rustc_span::def_id::DefId,
        args: rustc_middle::ty::GenericArgsRef<'tcx>,
    ) {
        let tcx = self.tcx;
        let typing_env = rustc_middle::ty::TypingEnv::fully_monomorphized();
        let resolved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rustc_middle::ty::Instance::try_resolve(tcx, typing_env, def_id, args)
        }));
        if let Ok(Ok(Some(target))) = resolved {
            self.enqueue(target);
        }
    }

    fn process_body(
        &mut self,
        instance: rustc_middle::ty::Instance<'tcx>,
        body: &rustc_middle::mir::Body<'tcx>,
    ) {
        use rustc_middle::mir::visit::Visitor as _;
        let tcx = self.tcx;
        // The collector evaluates every required constant of the
        // body it collects (the list is taken from the
        // pre-optimization body, so dead code is included).
        for const_op in body.required_consts() {
            let Some(mono_const) = self.monomorphize(instance, const_op.const_) else {
                continue;
            };
            match mono_const.eval(tcx, fully_monomorphized(), const_op.span) {
                Ok(value) => self.chase_value(value),
                Err(rustc_middle::mir::interpret::ErrorHandled::Reported(..)) => {
                    self.finding = Some(required_const_finding(tcx, instance));
                    return;
                }
                Err(rustc_middle::mir::interpret::ErrorHandled::TooGeneric(_)) => {}
            }
            if self.finding.is_some() {
                return;
            }
        }
        // The items the body mentions: their instances' required
        // constants are evaluated by the collector exactly like
        // this body's. Item bodies carry the mentions the compiler
        // recorded before optimization; shim bodies may carry
        // none, so there the mentions are recomputed with the
        // same visitor the compiler uses (`MentionScan`).
        let mut mentions: Vec<rustc_middle::mir::MentionedItem<'tcx>> = body
            .mentioned_items()
            .iter()
            .map(|spanned| spanned.node)
            .collect();
        if mentions.is_empty() {
            let mut scan = MentionScan {
                tcx,
                body,
                mentioned: Vec::new(),
            };
            scan.visit_body(body);
            mentions = scan.mentioned;
        }
        for item in mentions {
            let Some(item) = self.monomorphize(instance, item) else {
                continue;
            };
            self.visit_mentioned(item);
            if self.finding.is_some() {
                return;
            }
        }
        // Constant operands of the reachable blocks: the scan
        // evaluates each one and chases the value into the
        // allocations it points at (which can hold further
        // instances and statics). A function item or closure
        // used as a plain value is not collected from here; it
        // is collected through the mentioned items above where
        // it is actually used as a function.
        let mut scan = OperandScan {
            walk: self,
            instance,
        };
        for (bb, data) in rustc_middle::mir::traversal::mono_reachable(body, tcx, instance) {
            scan.visit_basic_block_data(bb, data);
            if scan.walk.finding.is_some() {
                return;
            }
        }
    }
    fn visit_mentioned(&mut self, item: rustc_middle::mir::MentionedItem<'tcx>) {
        use rustc_middle::mir::MentionedItem;
        let tcx = self.tcx;
        match item {
            MentionedItem::Fn(ty) => {
                if let rustc_middle::ty::FnDef(def_id, args) = *ty.kind() {
                    if let Some(args) = args.no_bound_vars() {
                        if let Ok(Some(instance)) = rustc_middle::ty::Instance::try_resolve(
                            tcx,
                            fully_monomorphized(),
                            def_id,
                            args,
                        ) {
                            self.enqueue(instance);
                        }
                    }
                }
            }
            MentionedItem::Drop(ty) => {
                let instance = rustc_middle::ty::Instance::resolve_drop_glue(tcx, ty);
                self.enqueue(instance);
            }
            MentionedItem::Closure(ty) => {
                if let rustc_middle::ty::Closure(def_id, args) = *ty.kind() {
                    let instance = rustc_middle::ty::Instance::resolve_closure(
                        tcx,
                        def_id,
                        args,
                        rustc_middle::ty::ClosureKind::FnOnce,
                    );
                    self.enqueue(instance);
                }
            }
            MentionedItem::UnsizeCast {
                source_ty,
                target_ty,
            } => {
                // Only the direct shape is chased: a concrete type
                // unsized to a trait object, whose vtable's method
                // instances the collector collects. The cast's
                // operand types are the pointer types (`&Vec<T>`
                // unsizing to `&dyn Trait`); the tails the vtable
                // is formed for are found by traversing the two
                // operand types together, the way the collector's
                // tail search does (see `find_unsize_tails`). (A
                // vtable formed through the tail of a custom
                // unsizing struct is reached here through the
                // constant value carrying it whenever one was
                // evaluated, and the remaining shapes only narrow
                // the check.)
                let (source_ty, target_ty) = find_unsize_tails(tcx, source_ty, target_ty);
                if target_ty.is_trait() && !source_ty.is_trait() {
                    if let rustc_middle::ty::Dynamic(data, _) = target_ty.kind() {
                        if let Some(principal) = data.principal() {
                            let principal = tcx.instantiate_bound_regions_with_erased(principal);
                            let alloc_id = tcx.vtable_allocation((source_ty, Some(principal)));
                            self.chase_alloc(alloc_id);
                        }
                    }
                }
            }
        }
    }

    /// Follow a constant value into the allocation it points at:
    /// the collector collects the function instances and statics
    /// inside evaluated constants (a function pointer constant
    /// whose target body holds a failing required constant is the
    /// `collect-in-dead-fnptr-in-const` shape).
    fn chase_value(&mut self, value: rustc_middle::mir::ConstValue) {
        use rustc_middle::mir::interpret::Scalar;
        match value {
            rustc_middle::mir::ConstValue::Scalar(Scalar::Ptr(ptr, _)) => {
                self.chase_alloc(ptr.provenance.alloc_id());
            }
            rustc_middle::mir::ConstValue::Indirect { alloc_id, .. }
            | rustc_middle::mir::ConstValue::Slice { alloc_id, .. } => {
                self.chase_alloc(alloc_id);
            }
            _ => {}
        }
    }

    fn chase_alloc(&mut self, alloc_id: rustc_middle::mir::interpret::AllocId) {
        if self.finding.is_some() || !self.visited_allocs.insert(alloc_id) {
            return;
        }
        use rustc_middle::mir::interpret::GlobalAlloc;
        match self.tcx.global_alloc(alloc_id) {
            GlobalAlloc::Function { instance, .. } => self.enqueue(instance),
            GlobalAlloc::Static(def_id) => {
                // A static the collector reaches is code-generated,
                // which evaluates its initializer; an evaluation
                // error there is the same class of failure. A
                // foreign static (an `extern` block item) is never
                // collected and has no initializer to evaluate
                // (asking for one is an internal compiler error),
                // so it is left alone, as are thread-local statics,
                // whose initialization is lazy and likewise not
                // evaluated by the collector.
                if self.tcx.is_foreign_item(def_id) || self.tcx.is_thread_local_static(def_id) {
                    return;
                }
                match self.tcx.eval_static_initializer(def_id) {
                    Ok(alloc) => self.chase_allocation(alloc),
                    Err(rustc_middle::mir::interpret::ErrorHandled::Reported(..)) => {
                        self.finding = Some(format!(
                            "initializer of static `{}` failed to evaluate \
                             (rustc reports the evaluation error during \
                             monomorphization collection)",
                            self.tcx.def_path_str(def_id)
                        ));
                    }
                    Err(rustc_middle::mir::interpret::ErrorHandled::TooGeneric(_)) => {}
                }
            }
            // A type-id allocation holds no instances or statics.
            GlobalAlloc::TypeId { .. } => {}
            GlobalAlloc::Memory(alloc) => self.chase_allocation(alloc),
            GlobalAlloc::VTable(ty, dyn_ty) => {
                // Collect the vtable's method instances the way
                // the collector does for a vtable allocation
                // (`create_mono_items_for_vtable_methods`): the
                // entries of the principal trait, plus the drop
                // glue when the type needs dropping. (Chasing the
                // vtable allocation itself cannot reach them: the
                // allocation id is the one this very call is
                // visiting, so the visited set would turn the
                // chase away.)
                if let Some(principal) = dyn_ty.principal() {
                    let principal = self.tcx.instantiate_bound_regions_with_erased(principal);
                    let trait_ref = principal.with_self_ty(self.tcx, ty);
                    for entry in self.tcx.vtable_entries(trait_ref) {
                        if let rustc_middle::ty::VtblEntry::Method(instance) = entry {
                            self.enqueue(*instance);
                        }
                    }
                    if ty.needs_drop(self.tcx, fully_monomorphized()) {
                        let glue = rustc_middle::ty::Instance::resolve_drop_glue(self.tcx, ty);
                        self.enqueue(glue);
                    }
                }
            }
        }
    }

    fn chase_allocation(&mut self, alloc: rustc_middle::mir::interpret::ConstAllocation<'tcx>) {
        let ptrs = alloc.inner().provenance().ptrs();
        for &prov in ptrs.values() {
            self.chase_alloc(prov.alloc_id());
            if self.finding.is_some() {
                return;
            }
        }
    }
}

/// Find the tails of an unsizing cast's operand types the way the
/// collector's `find_tails_for_unsizing` does: the source and the
/// target are traversed together, never independently. One pointer
/// layer (a reference, a raw pointer or a box) is peeled from both
/// sides at once, and from there the two types descend in lockstep
/// (see `unsize_lockstep_tails`) for as long as they keep the same
/// shape, stopping when they diverge — in particular when the
/// target reaches the trait object. So `&&W<T>` unsizing to
/// `&dyn Trait` forms the vtable for `&W<T>`: the traversal stops
/// with the target at `dyn Trait` after peeling one layer.
/// Peeling each side on its own would strip both source
/// references and select the implementation for `W<T>` instead.
fn find_unsize_tails<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    source_ty: rustc_middle::ty::Ty<'tcx>,
    target_ty: rustc_middle::ty::Ty<'tcx>,
) -> (rustc_middle::ty::Ty<'tcx>, rustc_middle::ty::Ty<'tcx>) {
    let pointees = match (source_ty.kind(), target_ty.kind()) {
        (
            &rustc_middle::ty::Ref(_, source_pointee, _),
            &rustc_middle::ty::Ref(_, target_pointee, _)
            | &rustc_middle::ty::RawPtr(target_pointee, _),
        )
        | (
            &rustc_middle::ty::RawPtr(source_pointee, _),
            &rustc_middle::ty::RawPtr(target_pointee, _),
        ) => Some((source_pointee, target_pointee)),
        _ => match (source_ty.boxed_ty(), target_ty.boxed_ty()) {
            (Some(source_boxed), Some(target_boxed)) => Some((source_boxed, target_boxed)),
            _ => None,
        },
    };
    match pointees {
        Some((source, target)) => unsize_lockstep_tails(tcx, source, target),
        None => (source_ty, target_ty),
    }
}

/// The lockstep half of the tail search (the collector's
/// `struct_lockstep_tails_for_codegen`): descend both types
/// through the tail field of a struct both sides instantiate, or
/// through the last element of equal-length tuples, for as long
/// as they keep the same shape.
fn unsize_lockstep_tails<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    mut source: rustc_middle::ty::Ty<'tcx>,
    mut target: rustc_middle::ty::Ty<'tcx>,
) -> (rustc_middle::ty::Ty<'tcx>, rustc_middle::ty::Ty<'tcx>) {
    loop {
        match (source.kind(), target.kind()) {
            (
                &rustc_middle::ty::Adt(source_def, source_args),
                &rustc_middle::ty::Adt(target_def, target_args),
            ) if source_def == target_def && source_def.is_struct() => {
                let Some(field) = source_def.non_enum_variant().tail_opt() else {
                    break;
                };
                source = tcx
                    .normalize_erasing_regions(fully_monomorphized(), field.ty(tcx, source_args));
                target = tcx
                    .normalize_erasing_regions(fully_monomorphized(), field.ty(tcx, target_args));
            }
            (&rustc_middle::ty::Tuple(source_tys), &rustc_middle::ty::Tuple(target_tys))
                if source_tys.len() == target_tys.len() =>
            {
                let (Some(&source_last), Some(&target_last)) =
                    (source_tys.last(), target_tys.last())
                else {
                    break;
                };
                source = source_last;
                target = target_last;
            }
            _ => break,
        }
    }
    (source, target)
}

fn required_const_finding(
    tcx: rustc_middle::ty::TyCtxt<'_>,
    instance: rustc_middle::ty::Instance<'_>,
) -> String {
    format!(
        "required constant of `{}` failed to evaluate \
         (rustc reports the evaluation error during \
         monomorphization collection)",
        tcx.def_path_str(instance.def.def_id())
    )
}

/// Recompute a body's mentioned items the way the compiler's
/// `MentionedItems` pass does (rustc_mir_transform's
/// `MentionedItemsVisitor`, kept in sync with the mono collector's
/// own visitor): calls and tail calls, drops, inline-asm function
/// symbols, unsizing casts that may form a vtable, and closures or
/// function items coerced to function pointers. The pass stores
/// its result on item bodies; shim bodies may carry no recording,
/// so for those the recording is reproduced here from the same
/// input the pass would read.
struct MentionScan<'a, 'tcx> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    body: &'a rustc_middle::mir::Body<'tcx>,
    mentioned: Vec<rustc_middle::mir::MentionedItem<'tcx>>,
}

impl<'tcx> rustc_middle::mir::visit::Visitor<'tcx> for MentionScan<'_, 'tcx> {
    fn visit_terminator(
        &mut self,
        terminator: &rustc_middle::mir::Terminator<'tcx>,
        location: rustc_middle::mir::Location,
    ) {
        self.super_terminator(terminator, location);
        use rustc_middle::mir::MentionedItem;
        match &terminator.kind {
            rustc_middle::mir::TerminatorKind::Call { func, .. }
            | rustc_middle::mir::TerminatorKind::TailCall { func, .. } => {
                self.mentioned
                    .push(MentionedItem::Fn(func.ty(self.body, self.tcx)));
            }
            rustc_middle::mir::TerminatorKind::Drop { place, .. } => {
                self.mentioned
                    .push(MentionedItem::Drop(place.ty(self.body, self.tcx).ty));
            }
            rustc_middle::mir::TerminatorKind::InlineAsm { operands, .. } => {
                for op in operands {
                    if let rustc_middle::mir::InlineAsmOperand::SymFn { value } = op {
                        self.mentioned.push(MentionedItem::Fn(value.const_.ty()));
                    }
                }
            }
            _ => {}
        }
    }

    fn visit_rvalue(
        &mut self,
        rvalue: &rustc_middle::mir::Rvalue<'tcx>,
        location: rustc_middle::mir::Location,
    ) {
        self.super_rvalue(rvalue, location);
        use rustc_middle::mir::MentionedItem;
        use rustc_middle::ty::adjustment::PointerCoercion;
        match *rvalue {
            rustc_middle::mir::Rvalue::Cast(
                rustc_middle::mir::CastKind::PointerCoercion(PointerCoercion::Unsize, _),
                ref operand,
                target_ty,
            ) => {
                // Detect unsizing casts that require vtables
                // (mirrors the pass: array-to-str/slice unsizing
                // is the one shape that never does).
                let source_ty = operand.ty(self.body, self.tcx);
                let may_involve_vtable = match (
                    source_ty.builtin_deref(true).map(|t| t.kind()),
                    target_ty.builtin_deref(true).map(|t| t.kind()),
                ) {
                    (
                        Some(rustc_middle::ty::Array(..)),
                        Some(rustc_middle::ty::Str | rustc_middle::ty::Slice(..)),
                    ) => false,
                    _ => true,
                };
                if may_involve_vtable {
                    self.mentioned.push(MentionedItem::UnsizeCast {
                        source_ty,
                        target_ty,
                    });
                }
            }
            rustc_middle::mir::Rvalue::Cast(
                rustc_middle::mir::CastKind::PointerCoercion(
                    PointerCoercion::ClosureFnPointer(_),
                    _,
                ),
                ref operand,
                _,
            ) => {
                self.mentioned
                    .push(MentionedItem::Closure(operand.ty(self.body, self.tcx)));
            }
            rustc_middle::mir::Rvalue::Cast(
                rustc_middle::mir::CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer(_), _),
                ref operand,
                _,
            ) => {
                self.mentioned
                    .push(MentionedItem::Fn(operand.ty(self.body, self.tcx)));
            }
            _ => {}
        }
    }
}

/// The constant-operand scan of one visited instance's reachable
/// blocks (see `RequiredConstWalk::process_body`).
struct OperandScan<'a, 'tcx> {
    walk: &'a mut RequiredConstWalk<'tcx>,
    instance: rustc_middle::ty::Instance<'tcx>,
}

impl<'tcx> rustc_middle::mir::visit::Visitor<'tcx> for OperandScan<'_, 'tcx> {
    fn visit_const_operand(
        &mut self,
        const_op: &rustc_middle::mir::ConstOperand<'tcx>,
        _location: rustc_middle::mir::Location,
    ) {
        let tcx = self.walk.tcx;
        let Some(mono_const) = self.walk.monomorphize(self.instance, const_op.const_) else {
            return;
        };
        // A bare function-item or closure operand is NOT
        // collected: both are plain values (a function item is
        // zero-sized, and so is a captureless closure), and
        // merely using one as a value (passing it to a generic
        // function that never calls it, storing it, black-boxing
        // it) does not make the collector codegen its body, so
        // resolving it here would demand bodies rustc never
        // collects and fail compilations rustc accepts. The
        // instances behind them are collected where the value
        // is actually used as a function: a call is a mentioned
        // item (a closure call names the `Fn`/`FnMut`/`FnOnce`
        // trait method, whose resolution reaches the closure
        // instance), a coercion to a function pointer is a
        // mentioned item too (both handled in
        // `visit_mentioned`), and an evaluated function-pointer
        // value points into a function allocation, which the
        // evaluation below chases.
        if self.walk.finding.is_some() {
            return;
        }
        match mono_const.eval(tcx, fully_monomorphized(), const_op.span) {
            Ok(value) => self.walk.chase_value(value),
            Err(rustc_middle::mir::interpret::ErrorHandled::Reported(..)) => {
                self.walk.finding = Some(required_const_finding(tcx, self.instance));
            }
            Err(rustc_middle::mir::interpret::ErrorHandled::TooGeneric(_)) => {}
        }
    }
}
