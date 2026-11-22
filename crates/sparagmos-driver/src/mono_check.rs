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
//!   intrinsics, length and element agreement for shuffles,
//!   selects and comparisons, insert/extract element agreement,
//!   and the constant index bounds of `simd_shuffle` (the index
//!   values are read from the call's constant operand, the same
//!   evidence code generation reads).
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
        if is_simd(ty) && !simd_layout_exists(ty) {
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
/// type is built on; `None` when the type is not a SIMD type or
/// its lane count is not a concrete value here.
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
            let (_, elem) = simd_shape(in_ty)?;
            let val_ty = input(2)?;
            if val_ty == elem {
                None
            } else {
                err(format!(
                    "expected inserted type `{elem}` (element of input \
                     `{in_ty}`), found `{val_ty}`"
                ))
            }
        }
        "simd_extract" | "simd_extract_dyn" => {
            let in_ty = input(0)?;
            let (_, elem) = simd_shape(in_ty)?;
            if ret_ty == elem {
                None
            } else {
                err(format!(
                    "expected return type `{elem}` (element of input \
                     `{in_ty}`), found `{ret_ty}`"
                ))
            }
        }
        _ => None,
    }
}
