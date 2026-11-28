//! Free-standing conversion helpers: small kind mappings, allocation reads, constant budgets, and ABI queries.

use super::*;

pub(crate) fn bb(idx: mir::BasicBlockIdx) -> sm::BasicBlock {
    sm::BasicBlock(idx as u32)
}

pub(crate) fn switch_targets(t: &mir::SwitchTargets) -> (Vec<u128>, Vec<sm::BasicBlock>, sm::BasicBlock) {
    let mut values = Vec::new();
    let mut targets = Vec::new();
    for (v, target) in t.branches() {
        values.push(v);
        targets.push(bb(target));
    }
    (values, targets, bb(t.otherwise()))
}

pub(crate) fn mutability(m: mir::Mutability) -> sm::Mutability {
    match m {
        mir::Mutability::Mut => sm::Mutability::Mut,
        mir::Mutability::Not => sm::Mutability::Not,
    }
}

pub(crate) fn region_of(r: &rustc_public::ty::Region) -> sm::Region {
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
pub(crate) const MAX_CONST_DEPTH: usize = 32;

/// Resource bound on array constants materialised element by
/// element in `decode_const`, mirroring `unique_const_value`'s
/// bound: an array length is guest-controlled input (`[(); usize::MAX]`
/// is a legal constant whose allocation is empty), so past this
/// many elements the decode only proceeds when the elements are
/// provably present in the allocation — the expansion is then
/// bounded by bytes the host already holds, not by the declared
/// length. Arrays that fail that check (and zero-sized-element
/// arrays, whose bytes prove nothing) keep their `ByRef`
/// identity instead of the host process aborting on allocation.
/// Byte arrays materialise as the raw `Bytes` value, whose size
/// is the allocation's own.
pub(crate) const MAX_CONST_ARRAY_ELEMS: u64 = 4096;

/// Total number of values one constant's materialisation may
/// produce, summed over every nesting level. The per-array
/// bound and the over-bound presence check bound a single
/// array by its own declared length or by the bytes backing
/// it, but neither bounds the product over nested arrays: an
/// element type with a nonzero stride can contain zero-sized
/// arrays whose declared lengths multiply without occupying
/// allocation bytes -- a `[([[(); 4096]; 4096], bool); 5000]`
/// static is 5,000 bytes of storage whose element-wise decode
/// would produce tens of billions of values. The shared
/// budget spent by every recursive decode answers that: a
/// constant whose expansion exceeds it stays undecoded, the
/// same fallback as any other undecodable constant.
/// Byte-copying arms spend the copied length against it too,
/// so repeated references to one shared byte block cannot
/// multiply that block's copies past the budget either.
pub(crate) const MAX_CONST_TOTAL_ELEMS: usize = 1 << 20;

/// Work budget for the uninitialised-range scan at the head
/// of `decode_const` (see `Converter::range_all_uninit`):
/// bytes of allocation the scan may examine while decoding
/// one root constant. A single value's own scans total a
/// small multiple of its allocation size -- its ranges nest,
/// and each level's scan stops at the first initialised
/// byte -- so any one value fits easily; the budget exists
/// for repetition, where one root constant re-decodes the
/// same large partially initialised pointee once per element
/// and every re-decode scans it in full. One gibibyte of
/// examinations bounds that repetition to seconds while
/// leaving single values of any realistic size untouched.
pub(crate) const MAX_CONST_SCAN_BYTES: usize = 1 << 30;

/// Work budget for backing-allocation materialisation while
/// decoding one root constant (see
/// `Converter::spend_work_budget`): bytes of allocation the
/// converter may materialise -- through `eval_initializer`
/// for statics, through the provenance resolution for memory
/// allocations -- summed over the whole decode, however often
/// an allocation is re-materialised. The allocation caches
/// bound what is *retained* at any one time, but retention is
/// not work: two statics of 129 MiB each, referenced
/// alternately by one table of one-byte slices, together
/// exceed the 256 MiB cache cap, so every element evicts the
/// other static's initialiser and re-materialises it in full
/// -- a 5,000-element table copies over 600 GiB while
/// spending only a few thousand expansion-budget units.
/// Charging each materialisation at its byte size bounds the
/// cumulative work instead: cache hits are free, a single
/// large backing is paid for once, and only
/// re-materialisation spends again. Four gibibytes leaves the
/// largest measured legitimate shape -- a 21-element table
/// alternating ten 128 MiB statics and two 100 MiB promoted
/// arrays, about 2.4 GiB of cumulative materialisation under
/// the same eviction pattern -- comfortably inside, while a
/// thrashing root constant stays undecoded after about
/// thirty re-materialisations instead of five thousand.
pub(crate) const MAX_CONST_WORK_BYTES: usize = 1 << 32;

/// Byte-size bound on allocations whose decodes are memoised
/// across root constants (see `Converter::const_memo`): the
/// memo key copies the allocation's bytes, so it is only
/// worth building for the small allocations root constants
/// usually are -- a pointer slot, a scalar, a small
/// aggregate. A larger allocation's decode is dominated by
/// its own content and is paid once per mention, as before.
pub(crate) const MAX_CONST_MEMO_BYTES: usize = 512;

/// Bound on the number of memoised constant decodes (see
/// `Converter::const_memo`): past the bound the memo is
/// cleared rather than grown, so a conversion that touches
/// very many distinct constants cannot pin them all.
pub(crate) const MAX_CONST_MEMO_ENTRIES: usize = 8192;

/// Cap on the total bytes held in the converter's per-constant
/// allocation cache (see `Converter::alloc_cache`).
pub(crate) const MAX_CACHED_ALLOC_BYTES: usize = 1 << 28;

/// Pointer width in bytes on the host this driver runs for
/// (x86_64); Stable MIR provenance entries are pointer-sized.
pub(crate) const PTR_SIZE: usize = 8;

/// Pointer width in bytes on the session target, from the
/// compiler target data layout. Stable MIR allocations store
/// pointer words (and a fat pointer metadata word) at this
/// width, which is narrower than the host width on a 32-bit
/// or 16-bit target: a constant decode that reads a pointer
/// word at the host width fails on exactly those targets (a
/// `*mut u32` or `*mut fn(..)` constant naming a static, a
/// `*mut *mut ()` constant naming a foreign static). Falls
/// back to the host width when no compiler context is
/// installed, where the target is the host by construction.
pub(crate) fn target_ptr_size() -> usize {
    match compiler_tcx() {
        Some(tcx) => {
            let size = tcx.data_layout.pointer_size().bytes() as usize;
            if size == 0 {
                PTR_SIZE
            } else {
                size
            }
        }
        None => PTR_SIZE,
    }
}

/// Byte size and signedness of a layout tag scalar: the stored
/// discriminant of a directly tagged enum, or the niche-carrying
/// scalar of a niche-encoded one. Pointers count as unsigned
/// pointer-sized integers; floats cannot be tags.
pub(crate) fn tag_scalar_size_signed(tag: &rustc_public::abi::Scalar) -> Option<(usize, bool)> {
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
        Primitive::Pointer(_) => Some((target_ptr_size(), false)),
        _ => None,
    }
}

/// The two pointer-sized segments of a `TypeId` value for `ty`:
/// a 128-bit FNV-1a hash (two lanes) of the type's debug form,
/// which is stable within a run - all the equality semantics of
/// `TypeId` need. `None` when the type cannot be spelled at
/// all: rendering the debug form of a coroutine-closure (async
/// closure) type panics inside rustc_public, like asking its
/// kind, so the segments are computed under the same guard
/// (see `guarded_ty_kind`) and an unspellable type carries no
/// segments - its downcasts fall back to tag comparison.
/// Also `None` for a type past the conversion size limit: the
/// debug form is materialized in full, so an exponentially
/// large type (the nested-closure shape of Issue #481) would
/// render without bound - the guard there catches panics, not
/// work. Such a type's bodies are refused for size anyway, so
/// losing its segments loses nothing.
pub(crate) fn type_id_segments(ty: &rustc_public::ty::Ty) -> Option<(u64, u64)> {
    if !crate::reach::ty_within_expanded_size(ty, crate::reach::MAX_TY_EXPANDED_SIZE) {
        return None;
    }
    fn lane(bytes: &[u8], seed: u64) -> u64 {
        let mut h = seed;
        for b in bytes {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }
    guarded_query(|| {
        let text = format!("{ty:?}");
        let bytes = text.as_bytes();
        (
            lane(bytes, 0xcbf2_9ce4_8422_2325),
            lane(bytes, 0x8422_2325_cbf2_9ce4),
        )
    })
}

/// The identity number of a vtable for the (concrete type, trait
/// reference) pair: an FNV-1a hash of the pair's debug form, with
/// the top bit set to namespace it away from the small integer
/// addresses dangling pointers carry. Only equality / ordering
/// between vtable identities is observable, which a stable hash
/// preserves.
pub(crate) fn vtable_id(
    ty: &rustc_public::ty::Ty,
    trait_ref: &Option<rustc_public::ty::Binder<rustc_public::ty::ExistentialTraitRef>>,
) -> u64 {
    let text = format!("{ty:?}|{trait_ref:?}");
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h | (1 << 63)
}

/// What an allocation's provenance entry denotes, named in
/// terms that are stable across mentions of the same
/// constant, for the `const_memo` key: the bridge index of an
/// allocation id is assigned per materialisation, so it
/// cannot name the target -- two mentions of one reference
/// into a static carry different indices for the same
/// static. Statics and functions are named by their
/// definitions (a function's identity is the key the
/// const-fn mention table uses), vtables and type ids by the
/// rendered pair / type the codebase already hashes for
/// their identities, and a memory allocation by its exact
/// contents, recursively, within the memo bounds (see
/// `prov_target_key`).
#[derive(PartialEq, Eq, Hash)]
pub(crate) enum ProvTargetKey {
    Static(rustc_public::DefId),
    Function(rustc_public::DefId, String),
    VTable(String),
    TypeId(String),
    Memory(Vec<Option<u8>>, u64, Vec<(usize, ProvTargetKey)>),
}

/// The [`ProvTargetKey`] of the allocation `prov` denotes,
/// or `None` when the target cannot be named within the
/// memo bounds: a memory allocation larger than
/// `MAX_CONST_MEMO_BYTES`, or one whose own provenance
/// reaches deeper than two levels of memory allocations.
/// Such constants are simply not memoised; they decode on
/// every mention, as before.
pub(crate) fn prov_target_key(prov: &Prov, depth: usize) -> Option<ProvTargetKey> {
    use mir::alloc::GlobalAlloc;
    match GlobalAlloc::from(prov.0) {
        GlobalAlloc::Static(def) => {
            Some(ProvTargetKey::Static(rustc_public::CrateDef::def_id(&def)))
        }
        GlobalAlloc::Function(inst) => Some(ProvTargetKey::Function(
            inst.def.def_id(),
            format!("{:?}", inst.args()),
        )),
        GlobalAlloc::VTable(ty, trait_ref) => {
            Some(ProvTargetKey::VTable(format!("{ty:?}|{trait_ref:?}")))
        }
        GlobalAlloc::TypeId { ty } => Some(ProvTargetKey::TypeId(format!("{ty:?}"))),
        GlobalAlloc::Memory(alloc) => {
            if depth >= 2 || alloc.bytes.len() > MAX_CONST_MEMO_BYTES {
                return None;
            }
            let mut ptrs = Vec::with_capacity(alloc.provenance.ptrs.len());
            for (off, p) in &alloc.provenance.ptrs {
                ptrs.push((*off, prov_target_key(p, depth + 1)?));
            }
            Some(ProvTargetKey::Memory(alloc.bytes, alloc.align, ptrs))
        }
    }
}

/// Read an unsigned integer of `size` bytes at `offset` in `alloc`.
/// The provenance recorded for the pointer slot at
/// `offset` in `alloc`, if any. rustc_public builds
/// `Allocation::provenance.ptrs` from the compiler's
/// provenance map, which iterates in offset order, so the
/// vector is sorted by offset and the entry is found by
/// binary search. The linear scan this replaces made
/// decoding a large reference table quadratic in its
/// length: every element's lookup rescanned the table's
/// provenance from the start (a 400,000-element table
/// needed on the order of 80 billion comparisons).
pub(crate) fn prov_at(alloc: &Allocation, offset: usize) -> Option<&Prov> {
    let ptrs = &alloc.provenance.ptrs;
    let idx = ptrs.partition_point(|(o, _)| *o < offset);
    ptrs.get(idx)
        .filter(|(o, _)| *o == offset)
        .map(|(_, prov)| prov)
}

/// Collect the bytes of `alloc` in `range` into a fresh
/// vector, failing when the range is out of bounds or any
/// byte inside it is uninitialised. Unlike
/// `Allocation::raw_bytes`, which copies the whole
/// allocation and fails on any uninitialised byte anywhere
/// in it, this reads only the requested range: a fat pointer
/// into a large shared allocation -- one element of a big
/// table of short slices into one promoted array -- must
/// not copy the entire backing allocation per element (a
/// 400,000-element table of one-byte slices into a 16 MiB
/// array copied terabytes in aggregate while charging only
/// the one-byte results against the expansion budget).
pub(crate) fn alloc_bytes_in(alloc: &Allocation, range: std::ops::Range<usize>) -> Option<Vec<u8>> {
    let slice = alloc.bytes.get(range)?;
    // A pre-sized loop, not `iter().copied().collect()` into
    // an `Option<Vec>`: the collecting adapter cannot see the
    // final size, so the copy regrows the vector repeatedly
    // and costs an order of magnitude more per byte in an
    // unoptimised build (measured ~55 ns/byte against
    // ~7 ns/byte for this loop) -- the difference between a
    // second and a minute when a large backing allocation is
    // copied once per element of a table that references it.
    let mut bytes = Vec::with_capacity(slice.len());
    for byte in slice {
        bytes.push((*byte)?);
    }
    Some(bytes)
}

pub(crate) fn read_uint_at(alloc: &Allocation, offset: usize, size: usize) -> Option<u128> {
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
pub(crate) fn substitute_params(ty: &sm::Ty, args: &[sm::Ty]) -> sm::Ty {
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

/// Resolve the array lengths of a converted field type whose
/// length is one of the ADT's const parameters. Conversion
/// renders such a length in the nameless parameter form (the
/// value exists only at the instance), which renders into a
/// field tag that names no recorded shape; walking the
/// definition's field type and the converted type in parallel
/// puts the instance argument's value back. A length that does
/// not resolve — a const expression over parameters, a
/// non-usize const — keeps the parameter form, exactly as
/// conversion left it. The two types are structurally
/// parallel (conversion preserves the shape and only filters
/// the non-type arguments out of argument lists), so the walk
/// consumes converted arguments in the order the raw arguments
/// present their type arguments; a shape the walk does not
/// recognise is returned unchanged.
pub(crate) fn resolve_param_array_lens(
    raw: &rustc_public::ty::Ty,
    converted: &sm::Ty,
    args: &GenericArgs,
) -> sm::Ty {
    let Some(kind) = guarded_ty_kind(raw) else {
        return converted.clone();
    };
    match (&kind, converted) {
        (TyKind::RigidTy(RigidTy::Array(elem, len)), sm::Ty::Array(celem, clen)) => {
            let len = match (len.kind(), clen) {
                (TyConstKind::Param(p), sm::ArrayLen::Param(_)) => args
                    .0
                    .get(p.index as usize)
                    .and_then(|a| match a {
                        GenericArgKind::Const(c) => c.eval_target_usize().ok(),
                        _ => None,
                    })
                    .map(sm::ArrayLen::Known)
                    .unwrap_or_else(|| clen.clone()),
                _ => clen.clone(),
            };
            sm::Ty::Array(
                Box::new(resolve_param_array_lens(elem, celem, args)),
                len,
            )
        }
        (TyKind::RigidTy(RigidTy::Adt(_, raw_args)), sm::Ty::Adt(def, conv_args)) => {
            sm::Ty::Adt(*def, resolve_param_arg_list(&raw_args.0, conv_args, args))
        }
        (TyKind::RigidTy(RigidTy::FnDef(_, raw_args)), sm::Ty::FnDef(def, conv_args)) => {
            sm::Ty::FnDef(*def, resolve_param_arg_list(&raw_args.0, conv_args, args))
        }
        (
            TyKind::RigidTy(RigidTy::Closure(_, raw_args)),
            sm::Ty::Closure(def, conv_args),
        ) => sm::Ty::Closure(
            *def,
            resolve_param_arg_list(&raw_args.0, conv_args, args),
        ),
        (
            TyKind::RigidTy(RigidTy::Coroutine(_, raw_args)),
            sm::Ty::Coroutine(def, conv_args),
        ) => sm::Ty::Coroutine(
            *def,
            resolve_param_arg_list(&raw_args.0, conv_args, args),
        ),
        (TyKind::Alias(_, alias), sm::Ty::Alias(k, def, conv_args)) => sm::Ty::Alias(
            *k,
            *def,
            resolve_param_arg_list(&alias.args.0, conv_args, args),
        ),
        (TyKind::RigidTy(RigidTy::Ref(_, elem, _)), sm::Ty::Ref(region, celem, cm)) => {
            sm::Ty::Ref(
                region.clone(),
                Box::new(resolve_param_array_lens(elem, celem, args)),
                *cm,
            )
        }
        (TyKind::RigidTy(RigidTy::RawPtr(elem, _)), sm::Ty::RawPtr(celem, cm)) => {
            sm::Ty::RawPtr(
                Box::new(resolve_param_array_lens(elem, celem, args)),
                *cm,
            )
        }
        (TyKind::RigidTy(RigidTy::Slice(elem)), sm::Ty::Slice(celem)) => {
            sm::Ty::Slice(Box::new(resolve_param_array_lens(elem, celem, args)))
        }
        (TyKind::RigidTy(RigidTy::Pat(elem, _)), sm::Ty::Pat(celem, cpat)) => sm::Ty::Pat(
            Box::new(resolve_param_array_lens(elem, celem, args)),
            cpat.clone(),
        ),
        (TyKind::RigidTy(RigidTy::Tuple(raw_items)), sm::Ty::Tuple(conv_items)) => {
            sm::Ty::Tuple(
                raw_items
                    .iter()
                    .zip(conv_items.iter())
                    .map(|(r, c)| resolve_param_array_lens(r, c, args))
                    .collect(),
            )
        }
        (TyKind::RigidTy(RigidTy::FnPtr(poly)), sm::Ty::FnPtr(csig)) => {
            let raw_ios = &poly.value.inputs_and_output;
            let mut conv_ios: Vec<&sm::Ty> = csig.inputs.iter().collect();
            conv_ios.push(&csig.output);
            if raw_ios.len() != conv_ios.len() {
                return converted.clone();
            }
            let mut resolved: Vec<sm::Ty> = raw_ios
                .iter()
                .zip(conv_ios.iter())
                .map(|(r, c)| resolve_param_array_lens(r, c, args))
                .collect();
            let output = resolved.pop().unwrap_or_else(sm::Ty::unit);
            let mut sig = (**csig).clone();
            sig.inputs = resolved;
            sig.output = Box::new(output);
            sm::Ty::FnPtr(Box::new(sig))
        }
        _ => converted.clone(),
    }
}

fn resolve_param_arg_list(
    raw_args: &[GenericArgKind],
    conv_args: &[sm::Ty],
    args: &GenericArgs,
) -> Vec<sm::Ty> {
    let mut conv_iter = conv_args.iter();
    raw_args
        .iter()
        .filter_map(|a| match a {
            GenericArgKind::Type(t) => {
                conv_iter.next().map(|c| resolve_param_array_lens(t, c, args))
            }
            GenericArgKind::Const(_) | GenericArgKind::Lifetime(_) => None,
        })
        .collect()
}

pub(crate) fn ty_has_param(ty: &sm::Ty) -> bool {
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

pub(crate) fn discr_value(d: &rustc_public::ty::Discr) -> Option<i128> {
    match d.ty.kind() {
        TyKind::RigidTy(RigidTy::Int(t)) => {
            let size = int_ty_size(&t)?;
            Some(sign_extend(d.val, size) as i128)
        }
        TyKind::RigidTy(RigidTy::Uint(_)) => Some(d.val as i128),
        _ => None,
    }
}

pub(crate) fn sign_extend(raw: u128, size: usize) -> u128 {
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

pub(crate) fn int_ty_size(t: &SmirIntTy) -> Option<usize> {
    Some(match t {
        SmirIntTy::I8 => 1,
        SmirIntTy::I16 => 2,
        SmirIntTy::I32 => 4,
        SmirIntTy::I64 => 8,
        SmirIntTy::Isize => target_ptr_size(),
        SmirIntTy::I128 => 16,
    })
}

pub(crate) fn uint_ty_size(t: &SmirUintTy) -> Option<usize> {
    Some(match t {
        SmirUintTy::U8 => 1,
        SmirUintTy::U16 => 2,
        SmirUintTy::U32 => 4,
        SmirUintTy::U64 => 8,
        SmirUintTy::Usize => target_ptr_size(),
        SmirUintTy::U128 => 16,
    })
}

/// Whether a type mentions a coroutine anywhere in its
/// structure (through ADT generic arguments, tuples, arrays,
/// slices, references and pointer wrappers). See the guard in
/// `Converter::record_field_layouts`: such a type's layout is
/// never queried for a field size.
/// Whether a converted type mentions a coroutine (or a
/// coroutine closure, which converts to the same form)
/// anywhere inside it. The converted-side counterpart of
/// `ty_mentions_coroutine`, for screens that run inside
/// type conversion itself: asking the compiler's type for
/// its kind is the question that panics for
/// coroutine-closure types (see `guarded_ty_kind`), so a
/// recorder hooked into conversion must judge the mention
/// from the converted form alone.
pub(crate) fn converted_mentions_coroutine(ty: &sm::Ty) -> bool {
    match ty {
        sm::Ty::Coroutine(..) | sm::Ty::CoroutineWitness(..) => true,
        sm::Ty::Adt(_, args) | sm::Ty::Closure(_, args) | sm::Ty::FnDef(_, args) => {
            args.iter().any(converted_mentions_coroutine)
        }
        sm::Ty::Alias(_, _, args) => args.iter().any(converted_mentions_coroutine),
        sm::Ty::Array(elem, _) | sm::Ty::Slice(elem) | sm::Ty::Pat(elem, _) => {
            converted_mentions_coroutine(elem)
        }
        sm::Ty::Ref(_, inner, _) | sm::Ty::RawPtr(inner, _) => {
            converted_mentions_coroutine(inner)
        }
        sm::Ty::Tuple(items) => items.iter().any(converted_mentions_coroutine),
        sm::Ty::FnPtr(sig) => {
            sig.inputs.iter().any(converted_mentions_coroutine)
                || converted_mentions_coroutine(&sig.output)
        }
        sm::Ty::Dynamic(_, inner, _) => converted_mentions_coroutine(inner),
        _ => false,
    }
}

pub(crate) fn ty_mentions_coroutine(ty: &rustc_public::ty::Ty) -> bool {
    use rustc_public::ty::{GenericArgKind, RigidTy, TyKind};
    fn args_mention(args: &rustc_public::ty::GenericArgs) -> bool {
        args.0.iter().any(|arg| match arg {
            GenericArgKind::Type(t) => ty_mentions_coroutine(t),
            _ => false,
        })
    }
    // A type whose kind cannot be asked passes through a
    // coroutine-closure type (see `guarded_ty_kind`), so it
    // mentions a coroutine closure by construction. Every
    // caller treats `true` as "never query this type's
    // layout", which is the conservative answer for such a
    // type as well: its layout question is exactly the one
    // that is not total.
    let Some(kind) = guarded_ty_kind(ty) else {
        return true;
    };
    match kind {
        TyKind::RigidTy(RigidTy::Coroutine(..)) => true,
        TyKind::RigidTy(RigidTy::CoroutineClosure(..)) => true,
        TyKind::RigidTy(RigidTy::Adt(_, args)) => args_mention(&args),
        TyKind::RigidTy(RigidTy::Closure(_, args)) => args_mention(&args),
        TyKind::RigidTy(RigidTy::Tuple(tys)) => tys.iter().any(ty_mentions_coroutine),
        TyKind::RigidTy(RigidTy::Array(elem, _))
        | TyKind::RigidTy(RigidTy::Slice(elem))
        | TyKind::RigidTy(RigidTy::Pat(elem, _)) => ty_mentions_coroutine(&elem),
        TyKind::RigidTy(RigidTy::Ref(_, inner, _))
        | TyKind::RigidTy(RigidTy::RawPtr(inner, _)) => ty_mentions_coroutine(&inner),
        TyKind::Alias(_, alias) => args_mention(&alias.args),
        _ => false,
    }
}

/// Field offsets of a type's layout, in definition order.
pub(crate) fn arbitrary_offsets(ty: &rustc_public::ty::Ty) -> Option<Vec<usize>> {
    let shape = guarded_layout(ty)?.shape();
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
            SmirIntTy::I64 => 8,
        SmirIntTy::Isize => target_ptr_size(),
            SmirIntTy::I128 => 16,
        }),
        TyKind::RigidTy(RigidTy::Uint(t)) => Some(match t {
            SmirUintTy::U8 => 1,
            SmirUintTy::U16 => 2,
            SmirUintTy::U32 => 4,
            SmirUintTy::U64 => 8,
        SmirUintTy::Usize => target_ptr_size(),
            SmirUintTy::U128 => 16,
        }),
        TyKind::RigidTy(RigidTy::Float(SmirFloatTy::F32)) => Some(4),
        TyKind::RigidTy(RigidTy::Float(SmirFloatTy::F64)) => Some(8),
        _ => None,
    }
}

/// Short human-readable rendering of a Stable MIR type for error reports.
pub(crate) fn short_ty(ty: &rustc_public::ty::Ty) -> String {
    let s = format!("{ty:?}");
    if s.len() > 90 {
        format!("{}...", &s[..90])
    } else {
        s
    }
}


pub(crate) fn int_ty(t: SmirIntTy) -> sm::IntTy {
    match t {
        SmirIntTy::I8 => sm::IntTy::I8,
        SmirIntTy::I16 => sm::IntTy::I16,
        SmirIntTy::I32 => sm::IntTy::I32,
        SmirIntTy::I64 => sm::IntTy::I64,
        SmirIntTy::I128 => sm::IntTy::I128,
        SmirIntTy::Isize => sm::IntTy::Isize,
    }
}

pub(crate) fn uint_ty(t: SmirUintTy) -> sm::UintTy {
    match t {
        SmirUintTy::U8 => sm::UintTy::U8,
        SmirUintTy::U16 => sm::UintTy::U16,
        SmirUintTy::U32 => sm::UintTy::U32,
        SmirUintTy::U64 => sm::UintTy::U64,
        SmirUintTy::U128 => sm::UintTy::U128,
        SmirUintTy::Usize => sm::UintTy::Usize,
    }
}

pub(crate) fn float_ty(t: SmirFloatTy) -> sm::FloatTy {
    match t {
        SmirFloatTy::F16 => sm::FloatTy::F16,
        SmirFloatTy::F32 => sm::FloatTy::F32,
        SmirFloatTy::F64 => sm::FloatTy::F64,
        SmirFloatTy::F128 => sm::FloatTy::F128,
    }
}

pub(crate) fn binop(op: mir::BinOp) -> sm::BinOp {
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

pub(crate) fn unop(op: mir::UnOp) -> sm::UnOp {
    match op {
        mir::UnOp::Not => sm::UnOp::Not,
        mir::UnOp::Neg => sm::UnOp::Neg,
        mir::UnOp::PtrMetadata => sm::UnOp::PtrMetadata,
    }
}

pub(crate) fn pointer_coercion(c: mir::PointerCoercion) -> sm::PointerCoercion {
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

pub(crate) fn borrow_kind(b: &mir::BorrowKind) -> sm::BorrowKind {
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
/// integer type, `bool` (8), `char` (32) or a raw pointer (64).
/// 0 for anything else (floats, aggregates), which keeps the
/// terminator outside the evaluable class. `isize`/`usize` are 64 on the
/// x86_64 target Sparagmos aims at.
pub(crate) fn asm_type_width(ty: &Ty) -> u32 {
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
        // A raw pointer occupies a full register: an asm block
        // may materialize one (a `lea` of a static's address) or
        // pass one through a register it never dereferences.
        TyKind::RigidTy(RigidTy::RawPtr(..)) => 64,
        _ => 0,
    }
}

/// The explicit register name of an asm operand's raw
/// representation (`reg: Reg(X86(ax))`), when the operand is
/// bound to one architectural register rather than a register
/// class.
pub(crate) fn asm_raw_reg_name(raw: &str) -> Option<String> {
    let rest = raw.split("reg: Reg(X86(").nth(1)?;
    let name = rest.split(')').next()?;
    Some(name.to_string())
}

/// The item name of a `Sym` asm operand's raw representation
/// (`SymFn { value: foo }`, `SymStatic { value: BAR }`).
/// A static's raw form instead renders its definition id
/// (`SymStatic { def_id: DefId(0:4 ~ krate[51d6]::BAR) }`):
/// the name is the path after the `~`, with the crate
/// segment's `[hash]` disambiguator removed, matching the
/// path form `fn_defs` / `statics` report.
pub(crate) fn asm_raw_sym_name(raw: &str) -> Option<String> {
    if let Some(rest) = raw.split("value: ").nth(1) {
        return Some(rest.trim_end_matches('}').trim().to_string());
    }
    let inner = raw
        .split("~ ")
        .nth(1)?
        .trim()
        .trim_end_matches('}')
        .trim()
        .trim_end_matches(')')
        .trim();
    let (krate, rest) = inner.split_once("::")?;
    let krate = krate.split('[').next()?;
    Some(format!("{krate}::{rest}"))
}

/// The target index of a `Label` asm operand's raw
/// representation (`Label { target_index: 1 }`): the index into
/// the terminator's internal successor list, where the
/// fallthrough destination (when the terminator has one) is
/// index 0 and the label blocks follow in operand order.
pub(crate) fn asm_raw_label_index(raw: &str) -> Option<usize> {
    let rest = raw.split("target_index: ").nth(1)?;
    rest.chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

/// The register scalar of a plain integer / bool / char / float
/// type, for an asm call signature. Anything else (aggregates,
/// pointers, 128-bit integers) is not register-passed in this
/// model and returns `None`.
pub(crate) fn asm_scalar(ty: &rustc_public::ty::Ty) -> Option<sm::AsmScalar> {
    use rustc_public::ty::{FloatTy, IntTy, UintTy};
    let TyKind::RigidTy(rigid) = guarded_ty_kind(ty)? else {
        return None;
    };
    Some(match rigid {
        RigidTy::Int(it) => sm::AsmScalar::Int {
            width: match it {
                IntTy::Isize => 64,
                IntTy::I8 => 8,
                IntTy::I16 => 16,
                IntTy::I32 => 32,
                IntTy::I64 => 64,
                IntTy::I128 => return None,
            },
            signed: true,
        },
        RigidTy::Uint(it) => sm::AsmScalar::Int {
            width: match it {
                UintTy::Usize => 64,
                UintTy::U8 => 8,
                UintTy::U16 => 16,
                UintTy::U32 => 32,
                UintTy::U64 => 64,
                UintTy::U128 => return None,
            },
            signed: false,
        },
        RigidTy::Bool => sm::AsmScalar::Bool,
        RigidTy::Char => sm::AsmScalar::Int {
            width: 32,
            signed: false,
        },
        RigidTy::Float(ft) => sm::AsmScalar::Float {
            width: match ft {
                FloatTy::F32 => 32,
                FloatTy::F64 => 64,
                _ => return None,
            },
        },
        _ => return None,
    })
}

/// Whether the local function whose full definition name is
/// `def_name` (crate segment included) is the very function an
/// asm `sym` operand references by `referenced`. The operand's
/// raw form names the resolved item by its complete path —
/// crate-relative for an item of the local crate (`local::foo`
/// for the definition `test::local::foo`), crate-qualified for
/// any other crate's item (`aux::foo`) — so a local definition
/// matches only when the referenced path is its whole
/// crate-relative path, or its full name. Matching only the
/// last segment would let an unrelated local function stand in
/// for an external target the operand actually names.
pub(crate) fn asm_ref_matches_def(def_name: &str, referenced: &str) -> bool {
    if def_name == referenced {
        return true;
    }
    let Some(prefix) = def_name.strip_suffix(referenced) else {
        return false;
    };
    // The elided prefix must be exactly the crate segment:
    // "<crate>::", with no further "::" inside it.
    prefix.len() > 2 && prefix.ends_with("::") && !prefix[..prefix.len() - 2].contains("::")
}

/// The declared ABI of a function type (the type of a function
/// definition or of a resolved instance), for recording with
/// the function's converted body. `None` when the type carries
/// no function signature.
fn fn_ty_abi(ty: &rustc_public::ty::Ty) -> Option<sm::Abi> {
    let sig = guarded_ty_kind(ty)?.fn_sig()?.skip_binder();
    Some(abi_of(&sig.abi))
}

/// Run an ABI query under the panic-capture discipline of
/// [`guarded_ty_kind`]. Asking a still-generic instance for its
/// type panics with "needs further instantiation" — a known
/// gap, not a crash: the instance's ABI is not needed, since a
/// monomorphized instance of the same function registers (and
/// is recorded) separately. An instance the driver fabricated
/// without generic arguments for a definition that has
/// parameters (a `Drop` implementation generic over a
/// lifetime, asked for its drop function's ABI) fails the
/// same way inside the compiler's instantiation, with "no
/// args were provided in instantiate": the ABI is equally
/// unneeded, for the same reason. Those panics' hook records
/// are consumed, so no outer containment mistakes them for a
/// conversion panic; any other panic resumes.
pub(crate) fn guarded_abi_query(query: impl FnOnce() -> Option<sm::Abi>) -> Option<sm::Abi> {
    clear_captured_panic();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(query)) {
        Ok(abi) => abi,
        Err(payload) => match take_captured_panic() {
            Some(record)
                if record.contains("needs further instantiation")
                    || record.contains("no args were provided in instantiate") =>
            {
                None
            }
            other => {
                if let Some(record) = other {
                    record_captured_panic(record);
                }
                std::panic::resume_unwind(payload)
            }
        },
    }
}

/// The declared ABI of a resolved instance's function type,
/// for recording with the instance's converted body (see
/// [`guarded_abi_query`] for the containment).
pub fn instance_fn_abi(inst: &Instance) -> Option<sm::Abi> {
    guarded_abi_query(|| fn_ty_abi(&inst.ty()))
}

/// The declared ABI of a function definition's type, for
/// recording with the definition's converted body (see
/// [`guarded_abi_query`] for the containment).
pub fn def_fn_abi(def: &FnDef) -> Option<sm::Abi> {
    guarded_abi_query(|| fn_ty_abi(&def.ty()))
}

/// The ABI family of an asm call signature, from the signature's
/// rustc ABI. Only the two register conventions the interpreter
/// assembles are recognized; `C` and `System` are SysV64-shaped
/// on x86_64.
pub(crate) fn asm_abi_family(abi: &rustc_public::ty::Abi) -> Option<sm::AsmAbi> {
    let dbg = format!("{abi:?}");
    if dbg.starts_with("Win64") {
        Some(sm::AsmAbi::Win64)
    } else if dbg.starts_with("SysV64")
        || dbg.starts_with("System")
        || dbg == "C"
        || dbg.starts_with("C {")
    {
        Some(sm::AsmAbi::SysV64)
    } else {
        None
    }
}

/// Structurally restore the label blocks of a body's asm-goto
/// terminator. Stable MIR does not publish them: an InlineAsm
/// terminator's only reported successor is its fallthrough
/// destination. The label blocks are exactly the roots of the
/// region the body's control-flow graph cannot reach under that
/// reading — a label block's own continuation blocks may be
/// unreachable too, but only the label blocks themselves have
/// no predecessor inside the unreachable region. Restoration
/// applies only when the body has exactly one label-carrying
/// InlineAsm terminator (a single carrier) and the root count
/// matches the label count; for a `noreturn` terminator the
/// reported destination is itself the first label block, so one
/// fewer root is expected. Anything else returns `None`, and
/// the terminator's labels stay unrestored (which keeps the
/// terminator outside the evaluable class).
pub(crate) fn restore_asm_label_blocks(
    body: &mir::Body,
    label_count: usize,
    noreturn: bool,
) -> Option<Vec<sm::BasicBlock>> {
    let blocks = &body.blocks;
    let mut carriers = 0usize;
    for block in blocks.iter() {
        if let mir::TerminatorKind::InlineAsm { operands, .. } = &block.terminator.kind {
            if operands.iter().any(|op| op.raw_rpr.starts_with("Label")) {
                carriers += 1;
            }
        }
    }
    if carriers != 1 {
        return None;
    }
    let n = blocks.len();
    let succ: Vec<Vec<usize>> = blocks
        .iter()
        .map(|block| block.terminator.successors())
        .collect();
    let mut reachable = vec![false; n];
    reachable[0] = true;
    let mut stack = vec![0usize];
    while let Some(i) = stack.pop() {
        for &j in &succ[i] {
            if !reachable[j] {
                reachable[j] = true;
                stack.push(j);
            }
        }
    }
    let mut has_unreachable_pred = vec![false; n];
    for i in 0..n {
        if reachable[i] {
            continue;
        }
        for &j in &succ[i] {
            if !reachable[j] {
                has_unreachable_pred[j] = true;
            }
        }
    }
    let roots: Vec<usize> = (0..n)
        .filter(|&i| !reachable[i] && !has_unreachable_pred[i])
        .collect();
    let expected = if noreturn {
        label_count.checked_sub(1)?
    } else {
        label_count
    };
    if roots.len() != expected {
        return None;
    }
    Some(roots.iter().map(|&i| sm::BasicBlock(i as u32)).collect())
}

/// Read an integer literal out of a `Const` asm operand's raw
/// representation (`Const { value: const 5_usize }`): the bits
/// (masked to the type width) and the width. `None` for any other
/// shape (an unevaluated or non-integer constant).
pub(crate) fn asm_const_literal(raw: &str) -> Option<(u64, u32)> {
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
    // Masking to the type width must not shift by the width of
    // the mask type itself: a 64-bit (or wider) operand already
    // fills the returned `u64`, so its bits are the value's low
    // 64 bits as-is. Shifting `1u64 << 64` panics with "attempt
    // to shift left with overflow" under overflow checks, which
    // is what `usize`/`u64`/`i64` asm const operands (e.g.
    // tests/ui/asm/{aarch64,x86_64}/const.rs) used to hit.
    let bits = if width >= 64 {
        value as u64
    } else {
        (value as u64) & ((1u64 << width) - 1)
    };
    Some((bits, width))
}

pub(crate) fn fake_read_cause(c: &mir::FakeReadCause) -> sm::FakeReadCause {
    use mir::FakeReadCause as F;
    match c {
        F::ForMatchGuard => sm::FakeReadCause::ForMatchedGuard,
        F::ForMatchedPlace(_) => sm::FakeReadCause::ForVarBinding,
        F::ForGuardBinding => sm::FakeReadCause::ForGuardBinding,
        F::ForLet(_) => sm::FakeReadCause::ForLet,
        F::ForIndex => sm::FakeReadCause::ForIndexOf,
    }
}

pub(crate) fn variance_of(v: mir::Variance) -> sm::Variance {
    match v {
        mir::Variance::Covariant => sm::Variance::Covariant,
        mir::Variance::Invariant => sm::Variance::Invariant,
        mir::Variance::Contravariant => sm::Variance::Contravariant,
        mir::Variance::Bivariant => sm::Variance::Bivariant,
    }
}

pub(crate) fn coroutine_kind(k: &mir::CoroutineKind) -> sm::CoroutineKind {
    use mir::CoroutineDesugaring as D;
    match k {
        mir::CoroutineKind::Desugared(D::Async, _) => sm::CoroutineKind::Async,
        mir::CoroutineKind::Desugared(D::Gen, _) => sm::CoroutineKind::Gen,
        mir::CoroutineKind::Desugared(D::AsyncGen, _) => sm::CoroutineKind::Coroutine,
        mir::CoroutineKind::Coroutine(_) => sm::CoroutineKind::Coroutine,
    }
}

pub(crate) fn abi_of(abi: &rustc_public::ty::Abi) -> sm::Abi {
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

/// Whether a demanded closure's generic arguments are a
/// coroutine closure's (an async closure's). A plain closure
/// packs its tail as `[kind, sig, captures]`; a coroutine
/// closure packs `[kind, sig, captures, captures-by-ref]` —
/// an integer kind marker type, a fn-pointer signature, a
/// tuple of upvars, and a fn-pointer producing the
/// by-reference captures. The check is purely structural,
/// over the tail shapes; a plain closure's arguments never
/// match it (their last argument is the upvars tuple, not a
/// fn-pointer).
pub fn coroutine_closure_args(args: &GenericArgs) -> bool {
    let mut tail = args.0.iter().rev();
    let (Some(by_ref), Some(upvars), Some(sig), Some(kind)) =
        (tail.next(), tail.next(), tail.next(), tail.next())
    else {
        return false;
    };
    let is_fn_ptr = |arg: &GenericArgKind| {
        matches!(
            arg,
            GenericArgKind::Type(t)
                if matches!(
                    guarded_ty_kind(t),
                    Some(TyKind::RigidTy(RigidTy::FnPtr(..)))
                )
        )
    };
    let is_tuple = |arg: &GenericArgKind| {
        matches!(
            arg,
            GenericArgKind::Type(t)
                if matches!(
                    guarded_ty_kind(t),
                    Some(TyKind::RigidTy(RigidTy::Tuple(_)))
                )
        )
    };
    let is_kind = |arg: &GenericArgKind| {
        matches!(
            arg,
            GenericArgKind::Type(t)
                if matches!(
                    guarded_ty_kind(t),
                    Some(TyKind::RigidTy(RigidTy::Int(
                        SmirIntTy::I8 | SmirIntTy::I16 | SmirIntTy::I32
                    )))
                )
        )
    };
    is_fn_ptr(by_ref) && is_tuple(upvars) && is_fn_ptr(sig) && is_kind(kind)
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

/// Whether generic arguments genuinely mention a parameter
/// of the enclosing definition: a bare parameter, or an alias
/// (projection) over one, anywhere in the arguments. This is
/// the strict companion of [`args_mention_params`]: where
/// that walk counts a type whose kind cannot be asked (a
/// type passing through a coroutine-closure type) as
/// mentioning a parameter -- the conservative answer for
/// callers that defer or fall back -- this walk counts such
/// a type as monomorphic, because a coroutine-closure type
/// in a monomorphic position is a concrete type (an async
/// closure's type is the standing shape). A caller that must
/// not pose an argument to the compiler's normalization --
/// the instance demand scan is the standing one -- uses
/// this judgement: what it rejects is exactly what posing
/// cannot answer, a projection over a bare parameter.
pub fn args_mention_params_strict(args: &GenericArgs) -> bool {
    args.0.iter().any(|arg| match arg {
        GenericArgKind::Type(ty) => ty_mentions_param_strict(ty),
        GenericArgKind::Const(konst) => const_mentions_param_strict(konst),
        GenericArgKind::Lifetime(_) => false,
    })
}

fn const_mentions_param_strict(konst: &rustc_public::ty::TyConst) -> bool {
    match konst.kind() {
        TyConstKind::Param(_) => true,
        TyConstKind::Unevaluated(_, args) => args_mention_params_strict(args),
        TyConstKind::Value(ty, _) | TyConstKind::ZSTValue(ty) => {
            ty_mentions_param_strict(ty)
        }
        _ => false,
    }
}

fn ty_mentions_param_strict(ty: &Ty) -> bool {
    // A type whose kind cannot be asked passes through a
    // coroutine-closure type; see the note on
    // `args_mention_params_strict` for why this walk counts
    // it as monomorphic where `ty_mentions_param` does not.
    let Some(kind) = guarded_ty_kind(ty) else {
        return false;
    };
    match kind {
        TyKind::Param(_) => true,
        TyKind::Alias(_, alias) => args_mention_params_strict(&alias.args),
        TyKind::RigidTy(rigid) => match rigid {
            RigidTy::Ref(_, inner, _) | RigidTy::RawPtr(inner, _) | RigidTy::Slice(inner) => {
                ty_mentions_param_strict(&inner)
            }
            RigidTy::Array(inner, len) => {
                ty_mentions_param_strict(&inner) || const_mentions_param_strict(&len)
            }
            RigidTy::Pat(inner, _) => ty_mentions_param_strict(&inner),
            RigidTy::Tuple(elems) => elems.iter().any(ty_mentions_param_strict),
            RigidTy::FnPtr(sig) => sig
                .value
                .inputs_and_output
                .iter()
                .any(ty_mentions_param_strict),
            RigidTy::Adt(_, args)
            | RigidTy::FnDef(_, args)
            | RigidTy::Closure(_, args)
            | RigidTy::Coroutine(_, args)
            | RigidTy::CoroutineClosure(_, args)
            | RigidTy::CoroutineWitness(_, args) => args_mention_params_strict(&args),
            RigidTy::Dynamic(bounds, _) => bounds.iter().any(|b| match &b.value {
                ExistentialPredicate::Trait(t) => {
                    args_mention_params_strict(&t.generic_args)
                }
                ExistentialPredicate::Projection(p) => {
                    args_mention_params_strict(&p.generic_args)
                }
                ExistentialPredicate::AutoTrait(_) => false,
            }),
            _ => false,
        },
        _ => false,
    }
}

fn const_mentions_param(konst: &rustc_public::ty::TyConst) -> bool {
    match konst.kind() {
        TyConstKind::Param(_) => true,
        TyConstKind::Unevaluated(_, args) => args_mention_params(args),
        TyConstKind::Value(ty, _) | TyConstKind::ZSTValue(ty) => ty_mentions_param(ty),
        _ => false,
    }
}

pub(crate) fn ty_mentions_param(ty: &Ty) -> bool {
    if let Some(cached) =
        TY_MENTIONS_PARAM_CACHE.with(|cache| cache.borrow().get(ty).copied())
    {
        return cached;
    }
    let result = ty_mentions_param_walk(ty);
    TY_MENTIONS_PARAM_CACHE.with(|cache| {
        cache.borrow_mut().insert(*ty, result);
    });
    result
}

fn ty_mentions_param_walk(ty: &Ty) -> bool {
    // A type passing through a coroutine-closure type cannot
    // be asked its kind (see `guarded_ty_kind`); counting it
    // as mentioning a parameter is the conservative answer
    // for every caller (they all treat `true` as "not fully
    // monomorphic here" and defer or fall back).
    let Some(kind) = guarded_ty_kind(ty) else {
        return true;
    };
    match kind {
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

#[cfg(test)]
mod tests {
    use super::asm_const_literal;

    #[test]
    fn asm_const_literal_narrow_widths_mask_to_width() {
        assert_eq!(
            asm_const_literal("Const { value: const 5_u8 }"),
            Some((5, 8))
        );
        assert_eq!(
            asm_const_literal("Const { value: const -1_i8 }"),
            Some((0xFF, 8))
        );
        assert_eq!(
            asm_const_literal("Const { value: const -1_i16 }"),
            Some((0xFFFF, 16))
        );
        assert_eq!(
            asm_const_literal("Const { value: const -1_i32 }"),
            Some((0xFFFF_FFFF, 32))
        );
        assert_eq!(
            asm_const_literal("Const { value: const 7_u32 }"),
            Some((7, 32))
        );
    }

    #[test]
    fn asm_const_literal_full_width_operands_do_not_overflow() {
        // Regression: 64-bit asm const operands (the `usize`
        // constants of tests/ui/asm/aarch64/const.rs and
        // tests/ui/asm/x86_64/const.rs) used to panic in this
        // conversion with "attempt to shift left with overflow"
        // from masking with `1u64 << 64`.
        assert_eq!(
            asm_const_literal("Const { value: const 5_usize }"),
            Some((5, 64))
        );
        assert_eq!(
            asm_const_literal("Const { value: const 5_u64 }"),
            Some((5, 64))
        );
        assert_eq!(
            asm_const_literal("Const { value: const -1_i64 }"),
            Some((u64::MAX, 64))
        );
        assert_eq!(
            asm_const_literal("Const { value: const -1_isize }"),
            Some((u64::MAX, 64))
        );
        // Wider than the returned bits: the low 64 bits are kept.
        assert_eq!(
            asm_const_literal("Const { value: const -1_i128 }"),
            Some((u64::MAX, 128))
        );
    }

    #[test]
    fn asm_const_literal_rejects_non_integer_shapes() {
        assert_eq!(asm_const_literal("Const { value: const X }"), None);
        assert_eq!(asm_const_literal("In { reg: RegClass(X86(reg)) }"), None);
        assert_eq!(asm_const_literal("Const { value: const 5_f32 }"), None);
    }
}
