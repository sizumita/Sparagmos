//! Pointer arithmetic: synthetic addresses, layouts and slot pricing.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{RefProj, RefTarget, Value};

/// The synthetic address of a pointer value (see
/// [`crate::ir::Rvalue::PtrToInt`]): an FNV-1a hash of the reference
/// target, mapped into a non-zero, 65536-aligned range. The same
/// target always yields the same address within a run, so equality,
/// null and alignment checks on transmuted pointers behave as they
/// do natively; distinct targets collide only on a hash collision.
/// A target carrying a trailing byte displacement
/// ([`RefProj::ByteOffset`]) yields its base target's slot address
/// plus the displacement instead of a slot of its own, so the
/// transmuted address of a byte-displaced pointer moves exactly
/// as the native one does.
/// The slot size is 2^16, not 2^12, because a place in this model
/// can carry an alignment that large (`#[repr(align(0x10000))]`):
/// with 4096-byte slots the address of such a place is a multiple
/// of 4096 but not of its own alignment, and the compiler's
/// alignment precondition checks abort the guest.
/// The hash contributes 44 bits (2^44 aligned slots): truncating
/// it further makes collisions observable at realistic allocation
/// counts (a 20-bit truncation collided 4 times among 3,000
/// allocations), while the range still fits a guest `usize` with
/// headroom, so a pointer -> integer -> pointer round trip keeps
/// the address.
pub(crate) fn synthetic_address(value: &Value) -> Option<u128> {
    // An address-only pointer's address is the stored integer
    // itself (address 0 is the null address).
    if let Value::Ref(RefTarget::Address(addr)) = value {
        return Some(*addr as u128);
    }
    if matches!(value, Value::Ref(RefTarget::Null)) {
        // The null pointer's address is 0, as natively: `is_null`
        // and address comparisons depend on it.
        return Some(0);
    }
    // A dyn fat pointer over an address-only base denotes that
    // base address: over the null pointer it is the null
    // pointer (address 0, as above), and over a stored integer
    // address it is that address -- the vtable word does not
    // move the data pointer.
    if let Value::DynRef { base, .. } = value {
        if let Value::Ref(RefTarget::Address(addr)) = &**base {
            return Some(*addr as u128);
        }
        if matches!(&**base, Value::Ref(RefTarget::Null)) {
            return Some(0);
        }
    }
    let (stripped, offset) = strip_trailing_offsets_value(value);
    address_key(&stripped).map(|key| displaced_address(&key, offset))
}

/// The address a key's slot denotes under a byte displacement:
/// the slot address plus the displacement. A zero displacement
/// is the slot itself. The slot range starts at 2^32, far above
/// any displacement a guest's pointer arithmetic computes, so
/// the sum stays positive and distinct from the null address; a
/// displacement that would leave the positive range (not
/// reachable from a real pointer-arithmetic chain) keeps the
/// slot address.
fn displaced_address(key: &str, offset: i128) -> u128 {
    let slot = hash_address_key(key);
    if offset == 0 {
        return slot;
    }
    let moved = slot as i128 + offset;
    if moved > 0 {
        moved as u128
    } else {
        slot
    }
}

/// A reference target with its trailing byte displacements
/// removed, together with their sum. Pointer arithmetic whose
/// result does not coincide with a structural position carries
/// the displacement as a trailing [`RefProj::ByteOffset`] step,
/// and the position it denotes is the position the remaining
/// steps denote, moved by that many bytes -- so the target's
/// exposed address is its base target's address plus the
/// displacement, the relation the compiler's inserted alignment
/// checks (which key on the transmuted address) rely on. Only
/// trailing steps are priced this way: a displacement in the
/// middle of a projection (a field read through a byte-displaced
/// pointer) stays part of the target's identity, as it was
/// before addresses priced displacements at all.
fn split_trailing_byte_offsets(target: &RefTarget) -> (RefTarget, i128) {
    fn split(projection: &[RefProj]) -> (Vec<RefProj>, i128) {
        let mut end = projection.len();
        let mut total: i128 = 0;
        while end > 0 {
            match &projection[end - 1] {
                RefProj::ByteOffset(d) => {
                    total += d;
                    end -= 1;
                }
                _ => break,
            }
        }
        (projection[..end].to_vec(), total)
    }
    match target {
        RefTarget::Place {
            frame,
            local,
            projection,
        } => {
            let (projection, total) = split(projection);
            (
                RefTarget::Place {
                    frame: *frame,
                    local: *local,
                    projection,
                },
                total,
            )
        }
        RefTarget::Static { name, projection } => {
            let (projection, total) = split(projection);
            (
                RefTarget::Static {
                    name: name.clone(),
                    projection,
                },
                total,
            )
        }
        RefTarget::Alloc { id, projection } => {
            let (projection, total) = split(projection);
            (
                RefTarget::Alloc {
                    id: *id,
                    projection,
                },
                total,
            )
        }
        RefTarget::Value(base, projection) => {
            let (projection, total) = split(projection);
            (RefTarget::Value(base.clone(), projection), total)
        }
        RefTarget::Null | RefTarget::Address(_) => (target.clone(), 0),
    }
}

/// A pointer value with its trailing byte displacements split
/// off (see [`split_trailing_byte_offsets`]): the value the
/// remaining target denotes, and the summed displacement. Fat
/// and dyn pointers displace their base value.
fn strip_trailing_offsets_value(value: &Value) -> (Value, i128) {
    match value {
        Value::Ref(target) => {
            let (target, total) = split_trailing_byte_offsets(target);
            (Value::Ref(target), total)
        }
        Value::FatRef { base, len } => {
            let (base, total) = strip_trailing_offsets_value(base);
            (
                Value::FatRef {
                    base: Box::new(base),
                    len: *len,
                },
                total,
            )
        }
        Value::DynRef {
            base,
            concrete,
            traits,
            drop_glue,
            type_id,
        } => {
            let (base, total) = strip_trailing_offsets_value(base);
            (
                Value::DynRef {
                    base: Box::new(base),
                    concrete: concrete.clone(),
                    traits: traits.clone(),
                    drop_glue: drop_glue.clone(),
                    type_id: *type_id,
                },
                total,
            )
        }
        _ => (value.clone(), 0),
    }
}

/// The identity key a pointer value's synthetic address is
/// hashed from: a rendering of the target the value denotes.
/// Two values with the same key are the same pointer for every
/// address purpose in the model; [`Interpreter::pointer_address`]
/// hashes a normalised form of this key (see
/// [`Interpreter::canonical_target`]).
fn address_key(value: &Value) -> Option<String> {
    Some(match value {
        Value::Ref(target) => format!("ref:{target:?}"),
        Value::FatRef { base, len } => format!("fat:{base:?}:{len}"),
        Value::FnPtr(name) => format!("fn:{name}"),
        // A non-capturing closure value is a function pointer (it
        // coerces to one); its address identity is its body.
        Value::Closure { def, env } if env.is_empty() => format!("closure:{def}"),
        // A dyn fat pointer's data address is its base pointer's
        // identity together with the concrete type (two dyn values
        // over the same base but different concrete types are
        // different values' pointers in this model).
        Value::DynRef { base, concrete, .. } => format!("dyn:{base:?}:{concrete}"),
        _ => return None,
    })
}

/// Hash an address identity key into the synthetic address
/// range (see [`synthetic_address`] for the range's contract:
/// non-zero, 65536-aligned, 44 bits of hash).
fn hash_address_key(key: &str) -> u128 {
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
    let h = fnv1a(key.as_bytes());
    // The same 44 hash bits the 4096-slot mapping used, selecting
    // one of 2^44 slots of 2^16 bytes each: widening the slot
    // keeps every distinctness property of the field while making
    // each address a multiple of 2^16 (see [`synthetic_address`]).
    0x1_0000_0000u128 + u128::from((h >> 12) & 0xFFF_FFFF_FFFF) * 0x1_0000
}

/// Base of the synthetic stack region: frame bases descend
/// from here (see [`Interpreter::register_frame_base`]). The
/// region sits above the synthetic-hash range, and stack
/// addresses stay far below the statics and heap regions, so
/// the four address spaces never interleave.
const STACK_TOP: u128 = 1 << 56;

/// Base of the synthetic statics region: static storages are
/// priced upward from here in first-pricing order.
pub(crate) const STATICS_BASE: u128 = 1 << 61;

/// Base of the synthetic heap region: allocation-store
/// allocations are priced upward from here in allocation-id
/// order (allocation ids are issued in creation order and
/// never reused, so heap addresses order as the allocations
/// were created, as natively).
const HEAP_BASE: u128 = 1 << 62;

/// Base of the synthetic buffer region: the data buffers of
/// `Vec` values held in frame locals are priced upward from
/// here in first-pricing order (see
/// [`Interpreter::buffer_price`]). The region sits above the
/// synthetic-hash range and below the statics region, so
/// buffer addresses interleave with neither.
pub(crate) const BUFFER_BASE: u128 = (1 << 60) + (1 << 59);

/// The address a zero-sized root denotes: no storage backs
/// it, so — exactly as codegen prices such a pointer — the
/// address is the type's alignment, the canonical dangling
/// address (`NonNull::dangling`, an empty allocation's buffer
/// pointer, a reference to a zero-sized temporary).
fn dangling_address(align: u64) -> u128 {
    u128::from(align.max(1))
}

/// The dangling address of a zero-sized root, displaced by
/// the projection's byte displacements: every structural
/// position inside a zero-sized tree sits at the root's
/// address (the canonical zero-sized normalisation), but a
/// byte displacement the guest computed (`wrapping_byte_add`
/// on a dangling pointer) moves the address exactly as it
/// does natively. A displacement that would leave the
/// positive range keeps the dangling address, as
/// `displaced_address` keeps the slot.
fn dangling_displaced(align: u64, projection: &[RefProj]) -> u128 {
    let base = dangling_address(align);
    let mut sum: i128 = 0;
    for step in projection {
        if let RefProj::ByteOffset(d) = step {
            sum += d;
        }
    }
    base.checked_add_signed(sum).unwrap_or(base)
}

/// `value` rounded up to a multiple of `align` (a power of
/// two, at least one).
fn align_up_u128(value: u128, align: u64) -> u128 {
    let a = u128::from(align.max(1));
    (value + a - 1) / a * a
}

/// A root price plus a signed byte displacement, kept only
/// when the result is a guest-representable address.
fn displaced_price(base: u128, offset: i128) -> Option<u128> {
    let p = base as i128 + offset;
    if p >= 0 && p < (1i128 << 64) {
        Some(p as u128)
    } else {
        None
    }
}

/// One body's stack slot layout for the arithmetic address
/// model (see [`Interpreter::compute_slot_layout`]): the byte
/// offset of each local's slot within the frame — `None` for
/// a local the layout does not price (its declared size is
/// unknown) — and the frame's total extent in bytes.
pub(crate) struct SlotLayout {
    slots: Vec<Option<u64>>,
    extent: u64,
}

/// The block successors the storage-liveness dataflow runs
/// over: the terminator's ordinary targets (the body's
/// unwind cleanup target is added by the caller).
fn block_successors(block: &BasicBlockData) -> Vec<usize> {
    let mut out = Vec::new();
    match &block.terminator {
        TerminatorKind::Goto { target } => out.push(target.0),
        TerminatorKind::SwitchInt {
            targets, otherwise, ..
        } => {
            for (_, t) in targets {
                out.push(t.0);
            }
            out.push(otherwise.0);
        }
        TerminatorKind::Call { target, .. }
        | TerminatorKind::CallIndirect { target, .. }
        | TerminatorKind::CallTrait { target, .. }
        | TerminatorKind::CallBuiltin { target, .. }
        | TerminatorKind::InlineAsm { target, .. } => {
            if let Some(t) = target {
                out.push(t.0);
            }
        }
        TerminatorKind::Drop { target, .. } => out.push(target.0),
        TerminatorKind::Assert { target, .. } => out.push(target.0),
        _ => {}
    }
    out
}

/// Record the live set `cur` at statement boundary `boundary`
/// of block `bi` into the per-local live-range hulls.
fn note_live_hull(
    cur: &[u64],
    boundary: usize,
    bi: usize,
    hulls: &mut [Vec<Option<(usize, usize)>>],
) {
    for (l, hull) in hulls.iter_mut().enumerate() {
        if cur[l / 64] >> (l % 64) & 1 == 1 {
            let e = &mut hull[bi];
            *e = Some(match *e {
                None => (boundary, boundary),
                Some((f, l)) => (f.min(boundary), l.max(boundary)),
            });
        }
    }
}

/// Set the bit for local `l` in a liveness bitset.
fn liveness_set(set: &mut [u64], l: usize, n: usize) {
    if l < n {
        set[l / 64] |= 1u64 << (l % 64);
    }
}

/// The locals an operand reads: the root local of the place
/// it copies or moves (a projected place reads its root).
fn operand_liveness(op: &Operand, uses: &mut [u64], n: usize) {
    match op {
        Operand::Copy(p) | Operand::Move(p) => {
            liveness_set(uses, p.local.0, n);
        }
        Operand::Constant(_) => {}
    }
}

/// The locals an rvalue reads (see [`operand_liveness`]).
fn rvalue_liveness(rv: &Rvalue, uses: &mut [u64], n: usize) {
    match rv {
        Rvalue::Use(op)
        | Rvalue::UnaryOp(_, op)
        | Rvalue::Cast(op, _)
        | Rvalue::Unsize(op, _)
        | Rvalue::BoxDerefTransmute(op)
        | Rvalue::DataPtr(op)
        | Rvalue::Materialize(op, _)
        | Rvalue::PtrToInt(op)
        | Rvalue::FnPtrToPtr(op)
        | Rvalue::PtrField(op, _)
        | Rvalue::PtrFieldFirstElem(op, _)
        | Rvalue::PtrCastUnits(op, _)
        | Rvalue::PtrFirstElem(op)
        | Rvalue::PtrArrayBase(op)
        | Rvalue::PtrFieldBase(op, _)
        | Rvalue::WrapArrayElems(op)
        | Rvalue::UnwrapArrayElems(op)
        | Rvalue::UnwrapSliceElems(op)
        | Rvalue::WrapSliceElems(op)
        | Rvalue::BytesToStr(op)
        | Rvalue::IntToPtr(op)
        | Rvalue::EnumToInt(op, ..)
        | Rvalue::IntToEnum(op, ..)
        | Rvalue::SliceEnumToInt(op, ..)
        | Rvalue::SliceIntToEnum(op, ..)
        | Rvalue::Reinterpret(op, ..)
        | Rvalue::IntToPtrWrapped { operand: op, .. }
        | Rvalue::ScalarToPtrWrapped { operand: op, .. }
        | Rvalue::EnumToPtrWrapped { operand: op, .. }
        | Rvalue::IntToEnumWrapped { operand: op, .. } => {
            operand_liveness(op, uses, n);
        }
        Rvalue::BinaryOp(_, pair) | Rvalue::CheckedBinaryOp(_, pair) => {
            operand_liveness(&pair.0, uses, n);
            operand_liveness(&pair.1, uses, n);
        }
        Rvalue::AggregateTuple(ops) | Rvalue::CoroutineAggregate(ops) => {
            for op in ops {
                operand_liveness(op, uses, n);
            }
        }
        Rvalue::AggregateAdt { operands, .. }
        | Rvalue::ClosureAggregate { operands, .. } => {
            for op in operands {
                operand_liveness(op, uses, n);
            }
        }
        Rvalue::RawPtrAggregate { data, meta } => {
            operand_liveness(data, uses, n);
            if let Some(meta) = meta {
                operand_liveness(meta, uses, n);
            }
        }
        Rvalue::PtrOffset(a, b, ..) => {
            operand_liveness(a, uses, n);
            operand_liveness(b, uses, n);
        }
        Rvalue::Ref(p)
        | Rvalue::Len(p)
        | Rvalue::Discriminant(p, ..)
        | Rvalue::CopyForDeref(p)
        | Rvalue::Reborrow(p)
        | Rvalue::ShapedLoad(p, ..) => {
            liveness_set(uses, p.local.0, n);
        }
        Rvalue::ThreadLocalRef(_) | Rvalue::Unsupported(_) => {}
    }
}

/// The locals a statement reads (`uses`) and the local it
/// wholly writes (`defs`): a projected assignment reads its
/// root (a partial write), an unprojected one defines it.
fn statement_liveness(
    kind: &StatementKind,
    uses: &mut [u64],
    defs: &mut [u64],
    n: usize,
) {
    match kind {
        StatementKind::Assign(place, rv) => {
            rvalue_liveness(rv, uses, n);
            if place.projection.is_empty() {
                liveness_set(defs, place.local.0, n);
            } else {
                liveness_set(uses, place.local.0, n);
            }
        }
        StatementKind::ShapedStore { place, value, .. } => {
            operand_liveness(value, uses, n);
            liveness_set(uses, place.local.0, n);
        }
        StatementKind::SetDiscriminant { place, .. } => {
            liveness_set(uses, place.local.0, n);
        }
        StatementKind::Assume(op) => operand_liveness(op, uses, n),
        StatementKind::CopyNonOverlapping { src, dst, count } => {
            operand_liveness(src, uses, n);
            operand_liveness(dst, uses, n);
            operand_liveness(count, uses, n);
        }
        StatementKind::StorageLive(_)
        | StatementKind::StorageDead(_)
        | StatementKind::Nop
        | StatementKind::Coverage(..)
        | StatementKind::Unsupported(_) => {}
    }
}

/// The locals a terminator reads and the local its call
/// destination wholly writes (see [`statement_liveness`]).
fn terminator_liveness(
    kind: &TerminatorKind,
    uses: &mut [u64],
    defs: &mut [u64],
    n: usize,
) {
    fn dest_liveness(
        place: &Place,
        uses: &mut [u64],
        defs: &mut [u64],
        n: usize,
    ) {
        if place.projection.is_empty() {
            liveness_set(defs, place.local.0, n);
        } else {
            liveness_set(uses, place.local.0, n);
        }
    }
    match kind {
        TerminatorKind::SwitchInt { discr, .. } => {
            operand_liveness(discr, uses, n);
        }
        TerminatorKind::Call {
            args, destination, ..
        } => {
            for op in args {
                operand_liveness(op, uses, n);
            }
            dest_liveness(destination, uses, defs, n);
        }
        TerminatorKind::CallIndirect {
            func,
            args,
            destination,
            ..
        } => {
            operand_liveness(func, uses, n);
            for op in args {
                operand_liveness(op, uses, n);
            }
            dest_liveness(destination, uses, defs, n);
        }
        TerminatorKind::CallTrait {
            receiver,
            args,
            stamp,
            destination,
            ..
        } => {
            operand_liveness(receiver, uses, n);
            operand_liveness(args, uses, n);
            if let Some(stamp) = stamp {
                operand_liveness(stamp, uses, n);
            }
            dest_liveness(destination, uses, defs, n);
        }
        TerminatorKind::CallBuiltin {
            args, destination, ..
        } => {
            for op in args {
                operand_liveness(op, uses, n);
            }
            dest_liveness(destination, uses, defs, n);
        }
        TerminatorKind::Drop { place, .. } => {
            liveness_set(uses, place.local.0, n);
        }
        TerminatorKind::Assert { cond, .. } => {
            operand_liveness(cond, uses, n);
        }
        TerminatorKind::Return => {
            // The return terminator reads the return place.
            liveness_set(uses, 0, n);
        }
        TerminatorKind::Goto { .. }
        | TerminatorKind::Resume
        | TerminatorKind::Abort
        | TerminatorKind::Unreachable
        | TerminatorKind::InlineAsm { .. }
        | TerminatorKind::Unsupported(_) => {}
    }
}

/// The byte size of `value`'s storage, when it is known
/// from the value's shape alone: pointer-shaped leaves have
/// the model's pointer widths, aggregates sum their fields,
/// and byte/string data is its length. Scalar leaves carry
/// no width in the value and answer `None`.
fn value_storage_size(value: &Value) -> Option<usize> {
    match value {
        Value::Ref(_) => Some(8),
        Value::FatRef { .. } | Value::DynRef { .. } => Some(16),
        Value::Tuple(fields) => fields.iter().map(value_storage_size).sum(),
        Value::Enum { fields, .. } => fields.iter().map(value_storage_size).sum(),
        Value::Bytes(bytes) => Some(bytes.len()),
        Value::Str(text) => Some(text.len()),
        _ => None,
    }
}

/// The byte offset of each field of an aggregate constant
/// (plus the total size as the last entry), when every
/// field's byte size is known (see [`value_storage_size`]).
/// Byte data is one field per byte.
fn aggregate_field_offsets(value: &Value) -> Option<Vec<usize>> {
    let sizes: Vec<usize> = match value {
        Value::Tuple(fields) | Value::Enum { fields, .. } => fields
            .iter()
            .map(value_storage_size)
            .collect::<Option<Vec<_>>>()?,
        Value::Bytes(bytes) => vec![1; bytes.len()],
        _ => return None,
    };
    let mut offsets = Vec::with_capacity(sizes.len() + 1);
    let mut at = 0usize;
    for size in sizes {
        offsets.push(at);
        at += size;
    }
    offsets.push(at);
    Some(offsets)
}

/// A type in the middle of a projection walk (see
/// [`Interpreter::canonical_projection`]): either an IR type, or
/// a rendered type tag — the form an ADT field's type takes in
/// the program's debug-shape table, and the form a generic
/// argument of a rendered ADT name takes.
#[derive(Clone)]
enum WalkTy {
    Ty(Ty),
    Tag(String),
}

/// The order of two pointers into the same target, from the
/// positions they denote in it: `None` for pointers into
/// different targets, and for same-target positions whose order
/// the projections do not determine. The synthetic addresses are
/// unordered hashes, so an ordering the guest computes between
/// two positions of one allocation — the `offset_from`
/// precondition checks (`self >= origin`), the pointer pairs of
/// the slice iterators — must be answered here: natively those
/// positions have a definite order, and a hash order that
/// happens to invert it makes a well-formed program read as a
/// UB violation.
pub(crate) fn positional_order(a: &RefTarget, b: &RefTarget) -> Option<std::cmp::Ordering> {

    match (a, b) {
        (
            RefTarget::Place {
                frame: fa,
                local: la,
                projection: pa,
            },
            RefTarget::Place {
                frame: fb,
                local: lb,
                projection: pb,
            },
        ) if fa == fb && la == lb => compare_projections(pa, pb),
        (
            RefTarget::Static {
                name: na,
                projection: pa,
            },
            RefTarget::Static {
                name: nb,
                projection: pb,
            },
        ) if na == nb => compare_projections(pa, pb),
        (
            RefTarget::Alloc {
                id: ia,
                projection: pa,
            },
            RefTarget::Alloc {
                id: ib,
                projection: pb,
            },
        ) if ia == ib => compare_projections(pa, pb),
        (RefTarget::Value(va, pa), RefTarget::Value(vb, pb)) if va == vb => {
            compare_projections(pa, pb)
        }
        _ => None,
    }
}

/// The order of two positions in one target's storage, from
/// their projections: the first step where they differ decides
/// when both steps are positions of the same kind (`Index` into
/// a tuple-stored sequence, `Field` into an enum-stored one —
/// in both spellings the step number is the element order —
/// a byte displacement, or a sub-slice window's start). A
/// projection that is a prefix of the other denotes the base
/// position itself, which is the longer projection's position
/// exactly when every remaining step enters element 0. Any
/// other shape of difference is left undecided.
fn compare_projections(a: &[RefProj], b: &[RefProj]) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    let mut i = 0;
    loop {
        match (a.get(i), b.get(i)) {
            (None, None) => return Some(Ordering::Equal),
            (None, Some(_)) | (Some(_), None) => {
                let rest = if a.len() < b.len() { &b[i..] } else { &a[i..] };
                if rest
                    .iter()
                    .all(|step| matches!(step, RefProj::Index(0) | RefProj::Field(0)))
                {
                    return Some(Ordering::Equal);
                }
                return None;
            }
            (Some(x), Some(y)) => {
                if x == y {
                    i += 1;
                    continue;
                }
                return match (x, y) {
                    (RefProj::Index(m), RefProj::Index(n)) => Some(m.cmp(n)),
                    (RefProj::Field(m), RefProj::Field(n)) => Some(m.cmp(n)),
                    (RefProj::ByteOffset(m), RefProj::ByteOffset(n)) => Some(m.cmp(n)),
                    (
                        RefProj::Subslice { offset: m, .. },
                        RefProj::Subslice { offset: n, .. },
                    ) => {
                        if m != n {
                            Some(m.cmp(n))
                        } else {
                            i += 1;
                            continue;
                        }
                    }
                    _ => None,
                };
            }
        }
    }
}

/// The metadata word a `dyn` fat pointer flattens to: a
/// stable identity for the (concrete type, trait bounds) pair
/// the vtable serves, in the same synthetic-address space
/// [`synthetic_address`] assigns. The word names the vtable
/// rather than denoting storage; only the interpreter's
/// fat-value table turns a pattern carrying it back into the
/// pointer value it came from.
pub(crate) fn vtable_word(concrete: &str, traits: &[String]) -> u128 {
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
    let key = format!("vtable:{concrete}:{}", traits.join("+"));
    0x1_0000_0000u128 + u128::from(fnv1a(key.as_bytes()) & 0xFF_FFFF_FFFF_F000)
}

/// The data word a string value's fat pointer flattens to:
/// an identity for the string's bytes in the synthetic
/// address space. A string constant's storage has no other
/// address in this model, and equal contents pool to one
/// constant natively, so the bytes are the identity.
pub(crate) fn str_data_word(s: &str) -> u128 {
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
    let mut key = b"str:".to_vec();
    key.extend_from_slice(s.as_bytes());
    0x1_0000_0000u128 + u128::from(fnv1a(&key) & 0xFF_FFFF_FFFF_F000)
}

/// The data word a fat pointer flattens to: the base
/// pointer's synthetic address where it has one. A base the
/// address space does not cover (a byte string: a constant
/// slice's storage in this model) gets an identity word
/// keyed by the base value's own representation, the keying
/// `synthetic_address` itself uses for its targets.
pub(crate) fn data_word(base: &Value) -> u128 {
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
    if let Some(addr) = synthetic_address(base) {
        return addr;
    }
    let key = format!("fatbase:{base:?}");
    0x1_0000_0000u128 + u128::from(fnv1a(key.as_bytes()) & 0xFF_FFFF_FFFF_F000)
}

#[test]
fn synthetic_addresses_do_not_collide_at_scale() {
    // A few thousand distinct allocation targets (a realistic
    // live-allocation count for one program) must all receive
    // distinct synthetic addresses: a guest that keys a map by
    // pointer address or asserts address distinctness observes
    // any collision as two live objects sharing an address.
    let mut seen = std::collections::HashSet::new();
    for id in 0..3000u64 {
        let v = Value::Ref(RefTarget::Alloc {
            id,
            projection: Vec::new(),
        });
        let addr = synthetic_address(&v).expect("address");
        assert!(seen.insert(addr), "collision at allocation {id}");
    }
}

#[test]
fn synthetic_address_moves_with_trailing_byte_displacement() {
    let base = Value::Ref(RefTarget::Alloc {
        id: 7,
        projection: Vec::new(),
    });
    let base_addr = synthetic_address(&base).expect("address");
    let displaced = |d: i128| {
        Value::Ref(RefTarget::Alloc {
            id: 7,
            projection: vec![RefProj::ByteOffset(d)],
        })
    };
    // A byte-displaced pointer's address is the base target's
    // address plus the displacement -- the relation the
    // compiler's inserted alignment checks key on.
    assert_eq!(synthetic_address(&displaced(1)), Some(base_addr + 1));
    assert_eq!(synthetic_address(&displaced(-8)), Some(base_addr - 8));
    // Consecutive trailing displacements accumulate.
    let twice = Value::Ref(RefTarget::Alloc {
        id: 7,
        projection: vec![RefProj::ByteOffset(3), RefProj::ByteOffset(4)],
    });
    assert_eq!(synthetic_address(&twice), Some(base_addr + 7));
    // A displacement in the middle of a projection is part of
    // the target's identity, not a priced displacement: the
    // address is a slot of its own, as before.
    let middle = Value::Ref(RefTarget::Alloc {
        id: 7,
        projection: vec![RefProj::ByteOffset(1), RefProj::Field(0)],
    });
    let middle_addr = synthetic_address(&middle).expect("address");
    assert_eq!(middle_addr % 0x1_0000, 0);
    assert_ne!(middle_addr, base_addr + 1);
}

impl<'a> Interpreter<'a> {
    /// The size in bytes of an IR type whose layout is fixed by
    /// the type alone or by the program's ADT layout table:
    /// scalars, arrays of such types, pointers (fat when the
    /// pointee is unsized), and the ADT instances the driver
    /// recorded a layout for. Anything else — multi-element
    /// tuples (their field layout is the compiler's choice),
    /// slices, `str`, closures — yields `None`.
    pub(crate) fn ty_size(&self, ty: &Ty) -> Option<u64> {
        Some(match ty {
            Ty::Unit => 0,
            Ty::Bool => 1,
            Ty::Int(t) => u64::from(t.bits() / 8),
            Ty::Uint(t) => u64::from(t.bits() / 8),
            Ty::Float(t) => u64::from(t.bits() / 8),
            Ty::Adt(name) => self.program().adt_layouts.get(name).map(|&(size, _)| size)?,
            Ty::Ref(pointee) => {
                if self.ty_is_unsized(pointee) {
                    16
                } else {
                    8
                }
            }
            Ty::Array(elem, n) => n.checked_mul(self.ty_size(elem)?)?,
            Ty::Tuple(ts) if ts.len() <= 1 => match ts.first() {
                Some(t) => self.ty_size(t)?,
                None => 0,
            },
            Ty::FnPtr => 8,
            Ty::FnDef(_) => 0,
            _ => return None,
        })
    }

    /// The element size of a pointer offset whose size conversion
    /// could not bake: derived from the executing body's declared
    /// type of the pointer operand, when the operand is a plain
    /// local of pointer type.
    pub(crate) fn offset_elem_size_of(&self, body: &Body, op: &Operand) -> Option<u64> {
        let place = match op {
            Operand::Copy(place) | Operand::Move(place) => place,
            Operand::Constant(_) => return None,
        };
        if !place.projection.is_empty() {
            return None;
        }
        match body.local_tys.get(place.local.0)? {
            Ty::Ref(pointee) => self.ty_size(pointee),
            _ => None,
        }
    }

    /// The synthetic address of a pointer value with the
    /// size-aware normalisation [`synthetic_address`] cannot
    /// perform on the value alone: an element position in a
    /// zero-sized element sequence denotes the same address
    /// whatever its index — the position's byte displacement is
    /// zero — so the identity key is hashed from the target with
    /// such positions normalised to the sequence's base (see
    /// [`Self::canonical_target`]). Every other position keeps
    /// the plain synthetic identity: the address returned here
    /// differs from [`synthetic_address`]'s only where a
    /// zero-sized element step was actually normalised.
    /// `leaf_ty` is the declared pointee type of the operand the
    /// value was read from, when the caller knows it: it prices
    /// the normalisation of targets whose root type the
    /// interpreter does not record (allocations, constant
    /// aggregates).
    pub(crate) fn pointer_address(
        &self,
        frames: &[Frame],
        value: &Value,
        leaf_ty: Option<&Ty>,
    ) -> Option<u128> {
        if matches!(
            value,
            Value::Ref(RefTarget::Address(_)) | Value::Ref(RefTarget::Null)
        ) {
            return synthetic_address(value);
        }
        // A dyn fat pointer over an address-only base has the
        // base's address (see `synthetic_address`).
        if let Value::DynRef { base, .. } = value {
            if matches!(
                &**base,
                Value::Ref(RefTarget::Address(_)) | Value::Ref(RefTarget::Null)
            ) {
                return synthetic_address(value);
            }
        }
        // The arithmetic address model (see
        // [`Self::arithmetic_price`]): where it derives an
        // address, that address — not the synthetic hash —
        // is the pointer's identity, so two storages the
        // compiler's stack colouring would share a slot
        // compare equal, a packed field's address is its
        // base plus its layout offset, and a zero-sized
        // root's address is its alignment, as natively.
        if let Some(price) = self.arithmetic_price(value, leaf_ty, Some(frames)) {
            return Some(price);
        }
        match self.canonical_address_key(frames, value, leaf_ty) {
            Some((key, offset)) => Some(displaced_address(&key, offset)),
            None => synthetic_address(value),
        }
    }

    /// Record the stack base of a freshly pushed frame (see
    /// the `frame_info` field): the new frame's slot area sits
    /// one frame extent below the caller's base; the entry
    /// frame's sits one extent below the stack top. A frame
    /// id that already has a base keeps it. A body whose slot
    /// layout is unknown still advances the chain by a
    /// minimal extent, so its callees' frames price normally.
    pub(crate) fn register_frame_base(&self, frames: &[Frame], id: u64, func: &str) {
        let mut info = self.frame_info.borrow_mut();
        if info.contains_key(&id) {
            return;
        }
        let caller_base = frames
            .last()
            .and_then(|f| info.get(&f.id).map(|(_, b)| *b));
        let extent = self
            .slot_layout(func)
            .map(|l| u128::from(l.extent))
            .unwrap_or(16);
        let base = match caller_base {
            Some(cb) => cb - extent,
            None => STACK_TOP - extent,
        };
        info.insert(id, (func.to_string(), base));
    }

    /// The alignment of a type, mirroring [`Self::ty_size`].
    fn ty_align(&self, ty: &Ty) -> Option<u64> {
        Some(match ty {
            Ty::Unit => 1,
            Ty::Bool => 1,
            Ty::Int(t) => u64::from(t.bits() / 8),
            Ty::Uint(t) => u64::from(t.bits() / 8),
            Ty::Float(t) => u64::from(t.bits() / 8),
            Ty::Adt(name) => {
                let program = self.program();
                program
                    .adt_layouts
                    .get(name)
                    .map(|&(_, align)| align)
                    .or_else(|| {
                        let key = self.adt_instance_key(name)?;
                        program.adt_layouts.get(&key).map(|&(_, align)| align)
                    })?
            }
            Ty::Ref(_) => 8,
            Ty::Array(elem, _) => self.ty_align(elem)?,
            Ty::Slice(elem) => self.ty_align(elem)?,
            Ty::Str => 1,
            Ty::Tuple(ts) => ts
                .iter()
                .map(|t| self.ty_align(t))
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .max()
                .unwrap_or(1),
            Ty::FnPtr => 8,
            Ty::FnDef(_) => 1,
            _ => return None,
        })
    }

    /// [`Self::ty_size`], extended with the one composite the
    /// base table does not fold: a tuple's size from its
    /// fields' declaration-order layout.
    pub(crate) fn ty_size_full(&self, ty: &Ty) -> Option<u64> {
        if let Some(size) = self.ty_size(ty) {
            return Some(size);
        }
        match ty {
            Ty::Tuple(ts) => {
                let (_, size) = self.tuple_layout(ts)?;
                Some(size)
            }
            // An array of elements only the full walk can
            // size (tuples, instance-resolved ADTs): the
            // array is the element count times the element
            // size, as `ty_size` computes for the elements
            // it sizes directly.
            Ty::Array(elem, n) => n.checked_mul(self.ty_size_full(elem)?),
            // A bare ADT name (the IR types drop generic
            // arguments) resolves to the one recorded
            // instance, as the field-layout lookups resolve
            // it.
            Ty::Adt(name) => {
                let key = self.adt_instance_key(name)?;
                self.program()
                    .adt_layouts
                    .get(&key)
                    .map(|&(size, _)| size)
            }
            _ => None,
        }
    }

    /// The (offset, size) of each field of a tuple type under
    /// the declaration-order layout the value model stores
    /// tuples in, and the tuple's total size.
    fn tuple_layout(&self, ts: &[Ty]) -> Option<(Vec<(u64, u64)>, u64)> {
        let mut offsets = Vec::with_capacity(ts.len());
        let mut cursor: u128 = 0;
        let mut max_align = 1u64;
        for t in ts {
            let size = self.ty_size_full(t)?;
            let align = self.ty_align(t)?;
            max_align = max_align.max(align);
            cursor = align_up_u128(cursor, align);
            offsets.push((u64::try_from(cursor).ok()?, size));
            cursor += u128::from(size);
        }
        let total = u64::try_from(align_up_u128(cursor, max_align)).ok()?;
        Some((offsets, total))
    }

    /// The slot layout of the body named `func`, computed
    /// once and cached (see the `body_slots` field).
    fn slot_layout(&self, func: &str) -> Option<std::rc::Rc<SlotLayout>> {
        if let Some(hit) = self.body_slots.borrow().get(func) {
            return hit.clone();
        }
        let layout = self
            .program()
            .get(func)
            .and_then(|body| self.compute_slot_layout(body))
            .map(std::rc::Rc::new);
        self.body_slots
            .borrow_mut()
            .insert(func.to_string(), layout.clone());
        layout
    }

    /// Compute a body's stack slot layout from its storage
    /// statements: two locals may share a slot exactly when
    /// their storage lifetimes never overlap — the sharing
    /// the compiler's stack colouring performs, which is
    /// what makes two sequential scoped temporaries of one
    /// function share an address natively. Locals the body
    /// never names in a storage statement are live for the
    /// whole body (arguments, the return place, and every
    /// local of a body the compiler emitted no storage
    /// statements for: such a body shares nothing). A local
    /// whose declared size is unknown gets no slot.
    fn compute_slot_layout(&self, body: &Body) -> Option<SlotLayout> {
        let n = body.local_tys.len();
        let total_stmts: usize =
            body.blocks.iter().map(|b| b.statements.len()).sum();
        if n > 2048 || body.blocks.len() > 8192 || total_stmts > 65536 {
            return None;
        }
        let sizes: Vec<Option<u64>> = body
            .local_tys
            .iter()
            .enumerate()
            .map(|(i, t)| {
                self.ty_size_full(t).or_else(|| {
                    // An array local whose element the type
                    // walk cannot size (the element type
                    // mentions an ADT whose bare name denotes
                    // several recorded instances): conversion
                    // recorded the element size while the
                    // generic arguments were at hand (see
                    // `ir::Body::local_elem_sizes`), and the
                    // array's size is the count times it.
                    match t {
                        Ty::Array(_, n) => body
                            .local_elem_sizes
                            .get(i)
                            .copied()
                            .flatten()
                            .and_then(|elem| n.checked_mul(elem)),
                        _ => None,
                    }
                })
            })
            .collect();
        let aligns: Vec<u64> = body
            .local_tys
            .iter()
            .map(|t| self.ty_align(t).unwrap_or(1))
            .collect();
        let mut managed = vec![false; n];
        let mut any_storage = false;
        for block in &body.blocks {
            for stmt in &block.statements {
                match stmt.kind {
                    StatementKind::StorageLive(l)
                    | StatementKind::StorageDead(l) => {
                        any_storage = true;
                        if l.0 < n {
                            managed[l.0] = true;
                        }
                    }
                    _ => {}
                }
            }
        }
        // Live ranges of the managed locals, as per-block
        // hulls over statement boundaries (boundary `i` sits
        // before statement `i`; boundary `len` after the
        // last): `hulls[local][block]` is the first and last
        // boundary the local is live at in that block.
        let mut hulls: Vec<Vec<Option<(usize, usize)>>> =
            vec![vec![None; body.blocks.len()]; n];
        let succs: Vec<Vec<usize>> = body
            .blocks
            .iter()
            .enumerate()
            .map(|(bi, b)| {
                let mut s = block_successors(b);
                if let UnwindAction::Cleanup(t) = body.unwind_action(bi) {
                    s.push(t.0);
                }
                s
            })
            .collect();
        if any_storage {
            let words = n.div_ceil(64).max(1);
            let mut live_in: Vec<Vec<u64>> =
                vec![vec![0; words]; body.blocks.len()];
            loop {
                let mut changed = false;
                for bi in (0..body.blocks.len()).rev() {
                    let mut cur = vec![0u64; words];
                    for &s in &succs[bi] {
                        if s < body.blocks.len() {
                            for w in 0..words {
                                cur[w] |= live_in[s][w];
                            }
                        }
                    }
                    for stmt in body.blocks[bi].statements.iter().rev() {
                        match stmt.kind {
                            StatementKind::StorageLive(l) if l.0 < n => {
                                cur[l.0 / 64] &= !(1u64 << (l.0 % 64));
                            }
                            StatementKind::StorageDead(l) if l.0 < n => {
                                cur[l.0 / 64] |= 1u64 << (l.0 % 64);
                            }
                            _ => {}
                        }
                    }
                    if cur != live_in[bi] {
                        live_in[bi] = cur;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
            for (bi, block) in body.blocks.iter().enumerate() {
                let mut cur = live_in[bi].clone();
                note_live_hull(&cur, 0, bi, &mut hulls);
                for (si, stmt) in block.statements.iter().enumerate() {
                    match stmt.kind {
                        StatementKind::StorageLive(l) if l.0 < n => {
                            cur[l.0 / 64] |= 1u64 << (l.0 % 64);
                        }
                        StatementKind::StorageDead(l) if l.0 < n => {
                            cur[l.0 / 64] &= !(1u64 << (l.0 % 64));
                        }
                        _ => {}
                    }
                    note_live_hull(&cur, si + 1, bi, &mut hulls);
                }
            }
        }
        // Live ranges of the unmanaged locals (no storage
        // statement names them — optimised MIR drops the
        // markers wholesale), from a def/use liveness at the
        // same statement precision as the storage ranges:
        // the hulls of a backward walk per block, seeded
        // with the block-level live-out. These are the
        // ranges native stack colouring derives from an
        // alloca's uses when no lifetime markers remain.
        // The exact live boundaries of the unmanaged
        // locals, per (local, block): a hull would fill the
        // gap between two disjoint live segments in one
        // block (a local the optimiser reused for two
        // temporaries), and the filled gap is exactly what
        // would block their slot sharing.
        let mut live_points: std::collections::HashMap<
            (usize, usize),
            Vec<usize>,
        > = std::collections::HashMap::new();
        let mut liveness_computed = false;
        if (0..n).any(|l| !managed[l] && sizes[l].is_some()) {
            liveness_computed = true;
            let words = n.div_ceil(64).max(1);
            let mut uses: Vec<Vec<u64>> =
                vec![vec![0; words]; body.blocks.len()];
            let mut defs: Vec<Vec<u64>> =
                vec![vec![0; words]; body.blocks.len()];
            for (bi, block) in body.blocks.iter().enumerate() {
                for stmt in &block.statements {
                    statement_liveness(
                        &stmt.kind,
                        &mut uses[bi],
                        &mut defs[bi],
                        n,
                    );
                }
                terminator_liveness(
                    &block.terminator,
                    &mut uses[bi],
                    &mut defs[bi],
                    n,
                );
            }
            let mut live_in: Vec<Vec<u64>> =
                vec![vec![0; words]; body.blocks.len()];
            let mut live_out: Vec<Vec<u64>> =
                vec![vec![0; words]; body.blocks.len()];
            loop {
                let mut changed = false;
                for bi in (0..body.blocks.len()).rev() {
                    let mut out = vec![0u64; words];
                    for &s in &succs[bi] {
                        if s < body.blocks.len() {
                            for w in 0..words {
                                out[w] |= live_in[s][w];
                            }
                        }
                    }
                    let mut inn = vec![0u64; words];
                    for w in 0..words {
                        inn[w] = uses[bi][w] | (out[w] & !defs[bi][w]);
                    }
                    if inn != live_in[bi] || out != live_out[bi] {
                        live_in[bi] = inn;
                        live_out[bi] = out;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
            // Backward walk per block: at each statement
            // boundary, the set of locals live there; a
            // statement's uses join the set and its def
            // leaves it, walking against execution order.
            // Only unmanaged locals are noted: a managed
            // local's hull stays exactly its storage range.
            for (bi, block) in body.blocks.iter().enumerate() {
                let mut cur = live_out[bi].clone();
                let mut tu = vec![0u64; words];
                let mut td = vec![0u64; words];
                terminator_liveness(&block.terminator, &mut tu, &mut td, n);
                for w in 0..words {
                    cur[w] = tu[w] | (cur[w] & !td[w]);
                }
                for l in 0..n {
                    if cur[l / 64] >> (l % 64) & 1 == 1 && !managed[l] {
                        live_points
                            .entry((l, bi))
                            .or_default()
                            .push(block.statements.len());
                    }
                }
                for (si, stmt) in block.statements.iter().enumerate().rev() {
                    let mut su = vec![0u64; words];
                    let mut sd = vec![0u64; words];
                    statement_liveness(&stmt.kind, &mut su, &mut sd, n);
                    for w in 0..words {
                        cur[w] = su[w] | (cur[w] & !sd[w]);
                    }
                    for l in 0..n {
                        if cur[l / 64] >> (l % 64) & 1 == 1 && !managed[l] {
                            live_points.entry((l, bi)).or_default().push(si);
                        }
                    }
                }
            }
        }
        // Address-taken exposure: def/use liveness ends a
        // local's range at the statement that borrows it, but
        // the borrow keeps the storage alive for as long as
        // the reference lives -- the derefs flow through the
        // holder local's value, invisible to def/use. A
        // borrowed local (the cast temporary of `&(a as
        // i32)`) whose range ends at the borrow would share
        // its slot with the very local it must stay distinct
        // from. Extend each unmanaged address-taken local's
        // live points: from just after the take to the end of
        // its block, and -- when the reference escapes the
        // block's holder graph (stored through a
        // dereference) or a holder is live in another block
        // -- every boundary of every block.
        if liveness_computed {
            use std::collections::HashSet;
            // carry[h]: the locals whose storage a reference
            // held in `h` may denote.
            let mut carry: Vec<HashSet<usize>> =
                vec![HashSet::new(); n];
            let mut escapes: HashSet<usize> = HashSet::new();
            let mut takes: Vec<(usize, usize, usize)> = Vec::new();
            fn plain_src(op: &Operand) -> Option<usize> {
                match op {
                    Operand::Copy(p) | Operand::Move(p)
                        if p.projection.is_empty() =>
                    {
                        Some(p.local.0)
                    }
                    _ => None,
                }
            }
            for (bi, block) in body.blocks.iter().enumerate() {
                for (si, stmt) in block.statements.iter().enumerate() {
                    let StatementKind::Assign(dest, rv) = &stmt.kind
                    else {
                        continue;
                    };
                    let dest_deref = dest.projection.iter().any(|s| {
                        matches!(s, ProjectionElem::Deref)
                    });
                    if let Rvalue::Ref(place) = rv {
                        let taken = place.local.0;
                        if taken < n {
                            takes.push((taken, bi, si + 1));
                            if dest_deref {
                                escapes.insert(taken);
                            } else if dest.local.0 < n {
                                carry[dest.local.0].insert(taken);
                            }
                        }
                    }
                    if dest_deref {
                        // A carried reference stored through a
                        // dereference leaves the frame's
                        // holder graph.
                        if let Rvalue::Use(op) = rv {
                            if let Some(s) = plain_src(op) {
                                for &t in &carry[s] {
                                    escapes.insert(t);
                                }
                            }
                        }
                    }
                }
            }
            // Holder propagation to a fixpoint: copies,
            // aggregate fields and call returns carry the
            // references their sources hold.
            loop {
                let mut changed = false;
                for block in &body.blocks {
                    let mut absorb_pairs: Vec<(usize, usize)> =
                        Vec::new();
                    for stmt in &block.statements {
                        let StatementKind::Assign(dest, rv) =
                            &stmt.kind
                        else {
                            continue;
                        };
                        if !dest.projection.is_empty() {
                            continue;
                        }
                        let dest = dest.local.0;
                        match rv {
                            Rvalue::Use(op) => {
                                if let Some(s) = plain_src(op) {
                                    absorb_pairs.push((dest, s));
                                }
                            }
                            Rvalue::AggregateTuple(ops)
                            | Rvalue::ClosureAggregate {
                                operands: ops,
                                ..
                            } => {
                                for op in ops {
                                    if let Some(s) = plain_src(op) {
                                        absorb_pairs.push((dest, s));
                                    }
                                }
                            }
                            Rvalue::AggregateAdt {
                                operands, ..
                            } => {
                                for op in operands {
                                    if let Some(s) = plain_src(op) {
                                        absorb_pairs.push((dest, s));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    if let TerminatorKind::Call {
                        args,
                        destination,
                        ..
                    } = &block.terminator
                    {
                        if destination.projection.is_empty() {
                            let dest = destination.local.0;
                            for op in args {
                                if let Some(s) = plain_src(op) {
                                    absorb_pairs.push((dest, s));
                                }
                            }
                        }
                    }
                    for (dest, s) in absorb_pairs {
                        if dest < n && s < n && dest != s {
                            let adds: Vec<usize> =
                                carry[s].iter().copied().collect();
                            for t in adds {
                                changed |= carry[dest].insert(t);
                            }
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
            for (taken, bi, from) in takes {
                if managed[taken] {
                    continue;
                }
                let wide = escapes.contains(&taken)
                    || (0..n).any(|h| {
                        carry[h].contains(&taken)
                            && live_points
                                .keys()
                                .any(|&(l, b)| l == h && b != bi)
                    });
                if wide {
                    for (bj, block) in body.blocks.iter().enumerate() {
                        let pts = live_points
                            .entry((taken, bj))
                            .or_default();
                        for b in 0..=block.statements.len() {
                            if !pts.contains(&b) {
                                pts.push(b);
                            }
                        }
                    }
                } else {
                    let block_len = body.blocks[bi].statements.len();
                    let pts =
                        live_points.entry((taken, bi)).or_default();
                    for b in from..=block_len {
                        if !pts.contains(&b) {
                            pts.push(b);
                        }
                    }
                }
            }
        }
        // Two locals interfere when some block has both live
        // at a common boundary. An unmanaged local whose
        // liveness was not computed (no sized unmanaged
        // local exists to price) keeps the live-throughout
        // reading and interferes with every other.
        let interferes = |a: usize, b: usize| -> bool {
            if (!managed[a] && !liveness_computed)
                || (!managed[b] && !liveness_computed)
            {
                return true;
            }
            if managed[a] && managed[b] {
                for bi in 0..body.blocks.len() {
                    if let (Some((f1, l1)), Some((f2, l2))) =
                        (hulls[a][bi], hulls[b][bi])
                    {
                        if f1 <= l2 && f2 <= l1 {
                            return true;
                        }
                    }
                }
                return false;
            }
            // Any pair involving an unmanaged local compares
            // its exact live boundaries against the other
            // side: another unmanaged local's boundaries, or
            // a managed local's (continuous) storage hull.
            let (u, other) = if !managed[a] { (a, b) } else { (b, a) };
            for (&(l, bi), pts) in &live_points {
                if l != u {
                    continue;
                }
                if !managed[other] {
                    if let Some(opts) = live_points.get(&(other, bi)) {
                        if pts.iter().any(|p| opts.contains(p)) {
                            return true;
                        }
                    }
                } else if let Some((f, last)) = hulls[other][bi] {
                    if pts.iter().any(|p| f <= *p && *p <= last) {
                        return true;
                    }
                }
            }
            false
        };
        // Slot assignment in local order: the order the
        // compiler creates a function's allocas and lays out
        // its frame. A local takes the lowest existing slot
        // that holds it and whose occupants it does not
        // interfere with; failing that, it opens a slot of
        // its own size at the current extent. Stack colouring
        // then shows up only as this reuse rule: two
        // sequential temporaries of a size class land in the
        // same slot, the second finding the first's slot
        // free, so addresses the compiler makes coincide do
        // coincide here.
        // Two other orders misprice real shapes. Placing
        // largest-first with best fit let a small temporary
        // squat inside a larger local's slot -- one whose
        // range had not even begun -- and its same-sized
        // successor, which does interfere with the larger
        // local, was scattered to a different slot. Placing
        // in live-range start order let the swarm of
        // entry-block temporaries claim every low offset
        // first and pushed a large long-lived local to an
        // offset with an arbitrary alignment residue, which
        // address arithmetic on packed fields observes.
        let order: Vec<usize> =
            (0..n).filter(|&l| sizes[l].is_some()).collect();
        let mut slots: Vec<Option<u64>> = vec![None; n];
        let mut placed: Vec<(u64, u64, Vec<usize>)> = Vec::new();
        let mut extent: u64 = 0;
        for l in order {
            let size = sizes[l].unwrap_or(0);
            let align = aligns[l].max(1);
            // First fit in offset order: the lowest slot
            // that holds the local and whose occupants it
            // does not interfere with. Slots are created in
            // offset order, so the scan is in offset order.
            let mut chosen: Option<usize> = None;
            for (si, (off, ssize, occupants)) in placed.iter().enumerate() {
                if *ssize >= size
                    && *off % align == 0
                    && occupants.iter().all(|&o| !interferes(l, o))
                {
                    chosen = Some(si);
                    break;
                }
            }
            match chosen {
                Some(si) => {
                    slots[l] = Some(placed[si].0);
                    placed[si].2.push(l);
                }
                None => {
                    let off =
                        u64::try_from(align_up_u128(u128::from(extent), align))
                            .ok()?;
                    slots[l] = Some(off);
                    extent = off.checked_add(size)?;
                    placed.push((off, size, vec![l]));
                }
            }
        }
        // A frame always occupies some stack (the return
        // address and saved registers natively), so a callee's
        // slots never coincide with its caller's: the extent
        // is at least one slot of the frame alignment.
        let extent = u64::try_from(align_up_u128(u128::from(extent), 16).max(16))
            .ok()?;
        Some(SlotLayout { slots, extent })
    }

    /// The arithmetic address of a pointer value (see
    /// [`Self::pointer_address`]): the price codegen's
    /// address assignment implies — a place at its frame
    /// base plus its stack slot plus the projection's byte
    /// displacement, a static in the statics region, an
    /// allocation in the heap region, a zero-sized root at
    /// its alignment — or `None` where the model cannot
    /// derive it, in which case the caller keeps the
    /// synthetic-hash price.
    pub(crate) fn arithmetic_price(
        &self,
        value: &Value,
        leaf_ty: Option<&Ty>,
        frames: Option<&[Frame]>,
    ) -> Option<u128> {
        match value {
            Value::Ref(target) => self.price_target(target, leaf_ty, frames),
            // A fat pointer's data address is its base's: the
            // metadata word is not part of the address.
            Value::FatRef { base, .. } => {
                self.arithmetic_price(base, leaf_ty, frames)
            }
            Value::DynRef { base, .. } => {
                self.arithmetic_price(base, None, frames)
            }
            _ => None,
        }
    }

    /// The arithmetic address of the storage a target
    /// denotes (see [`Self::arithmetic_price`]).
    fn price_target(
        &self,
        target: &RefTarget,
        leaf_ty: Option<&Ty>,
        frames: Option<&[Frame]>,
    ) -> Option<u128> {
        match target {
            RefTarget::Place {
                frame,
                local,
                projection,
            } => {
                let (func, base) = self
                    .frame_info
                    .borrow()
                    .get(frame)
                    .map(|(f, b)| (f.clone(), *b))?;
                let body = self.program().get(&func)?;
                let root_ty = body.local_tys.get(*local)?;
                if self.ty_size_full(root_ty) == Some(0) {
                    let align = self.ty_align(root_ty)?;
                    return Some(dangling_displaced(align, projection));
                }
                let root_elem_size = body
                    .local_elem_sizes
                    .get(*local)
                    .copied()
                    .flatten();
                // An element position in a `Vec` value held
                // in this local denotes the vector's data
                // buffer — a heap allocation in the guest's
                // semantics, not the local's own storage — so
                // it prices from the buffer's own base (see
                // `vec_buffer_price`), not from the local's
                // slot: the slot is sized by the `Vec` value's
                // footprint, and walking a buffer-sized
                // element space from it prices the buffer's
                // positions over every neighbouring storage.
                // Only the frame-carrying entry points can
                // see the root's current value.
                if let Some(frames) = frames {
                    if let Some(price) = self.vec_buffer_price(
                        frames,
                        *frame,
                        *local,
                        projection,
                        root_ty,
                        root_elem_size,
                        leaf_ty,
                    ) {
                        return Some(price);
                    }
                }
                // A first-element position in an empty
                // sequence value denotes a buffer the value
                // does not have: its address is the element
                // type's dangling address (see
                // `empty_sequence_price`). Only the
                // frame-carrying entry points can see the
                // root's current value.
                if let Some(frames) = frames {
                    if let Some(price) = self.empty_sequence_price(
                        frames, *frame, *local, projection, root_ty,
                    ) {
                        return Some(price);
                    }
                }
                let layout = self.slot_layout(&func)?;
                let slot = layout.slots.get(*local).copied()??;
                let offset = self.projection_offset_with_root_elem(
                    Some(root_ty),
                    projection,
                    leaf_ty,
                    root_elem_size,
                )?;
                displaced_price(base + u128::from(slot), offset)
            }
            RefTarget::Static { name, projection } => {
                let root_ty = self.program().static_tys.get(name);
                let offset =
                    self.projection_offset(root_ty, projection, leaf_ty)?;
                displaced_price(self.static_price(name), offset)
            }
            RefTarget::Alloc { id, projection } => {
                if let Some((size, align)) = self.alloc_layout(*id) {
                    if size == 0 {
                        return Some(dangling_displaced(align, projection));
                    }
                } else if let Some(Ty::Slice(elem)) = self.alloc_ty(*id) {
                    // No layout on record because the pointee
                    // is unsized: a slice allocation with no
                    // recorded layout is a zero-length slice
                    // box, whose buffer is the element type's
                    // dangling address. (The stored value
                    // cannot decide this: a freshly allocated
                    // real buffer also stores a scalar-free
                    // placeholder until it is written.)
                    if let Some(align) = self.ty_align(&elem) {
                        return Some(dangling_displaced(align, projection));
                    }
                }
                let root_ty = self.alloc_ty(*id);
                let offset = self.projection_offset(
                    root_ty.as_ref(),
                    projection,
                    leaf_ty,
                )?;
                displaced_price(self.heap_price(*id), offset)
            }
            RefTarget::Value(base, projection) => {
                // A scalar-free value tree is zero-sized
                // storage: every position in it — the root
                // included — is the dangling address of the
                // pointee's alignment, as a reference to a
                // zero-sized constant temporary (a `&()`, a
                // zero-sized struct value, an empty array
                // constant) is natively.
                if !value_tree_has_scalar(base) {
                    if let Some(align) =
                        leaf_ty.and_then(|t| self.ty_align(t))
                    {
                        return Some(dangling_displaced(align, projection));
                    }
                }
                if projection.is_empty() {
                    return None;
                }
                // The root prices as the bare value target
                // does (its synthetic-hash slot); the
                // projection's displacement is arithmetic
                // where derivable.
                let root = Value::Ref(RefTarget::Value(base.clone(), Vec::new()));
                let key = address_key(&root)?;
                let offset =
                    self.projection_offset(None, projection, leaf_ty)?;
                displaced_price(hash_address_key(&key), offset)
            }
            RefTarget::Address(addr) => Some(u128::from(*addr)),
            RefTarget::Null => Some(0),
        }
    }

    /// The synthetic address of a static storage's base,
    /// assigned from the statics region on first use: each
    /// static occupies at least its alignment's worth of the
    /// region, so distinct statics — zero-sized ones
    /// included — never share an address, and none lands on
    /// the dangling address its alignment would name.
    fn static_price(&self, name: &str) -> u128 {
        if let Some(p) = self.static_prices.borrow().get(name) {
            return *p;
        }
        let (size, align) = match self.program().static_tys.get(name) {
            Some(ty) => (
                self.ty_size_full(ty).unwrap_or(0),
                self.ty_align(ty).unwrap_or(8),
            ),
            None => (0, 8),
        };
        let price = align_up_u128(self.static_cursor.get(), align);
        self.static_cursor
            .set(price + u128::from(size.max(align).max(1)));
        self.static_prices
            .borrow_mut()
            .insert(name.to_string(), price);
        price
    }

    /// The (size, align) an allocation was created with,
    /// from whichever record covers it: the raw allocator
    /// builtins' table, the box builtins' layout table, or
    /// the recorded pointee type.
    fn alloc_layout(&self, id: u64) -> Option<(u64, u64)> {
        if let Some(l) = self.alloc_layouts.borrow().get(&id) {
            return Some(*l);
        }
        if let Some((size, align)) = self.box_layouts.borrow().get(&id) {
            return Some((
                u64::try_from(*size).ok()?,
                u64::try_from(*align).ok()?,
            ));
        }
        let ty = self.alloc_ty(id)?;
        Some((self.ty_size_full(&ty)?, self.ty_align(&ty)?))
    }

    /// Record the layout a raw allocation was created with
    /// (see the `alloc_layouts` field).
    pub(crate) fn record_alloc_layout(&self, id: u64, size: u64, align: u64) {
        self.alloc_layouts.borrow_mut().insert(id, (size, align));
    }

    /// The synthetic address of an allocation's base (see
    /// the `heap_prices` field): prices follow allocation-id
    /// order, each allocation occupying its layout's extent,
    /// so two live allocations' addresses order as they were
    /// created and never overlap. An allocation whose layout
    /// is unknown reserves a nominal extent so later
    /// allocations still price; only an address comparison
    /// against an un-layouted allocation's interior can
    /// observe the approximation.
    fn heap_price(&self, id: u64) -> u128 {
        if let Some(p) = self.heap_prices.borrow().get(&id) {
            return *p;
        }
        let (mut next_id, mut cursor) = self.heap_cursor.get();
        if next_id == 0 {
            next_id = 1;
            cursor = HEAP_BASE;
        }
        let mut prices = self.heap_prices.borrow_mut();
        while next_id <= id {
            let (size, align) = self.alloc_layout(next_id).unwrap_or((16, 8));
            let price = align_up_u128(cursor, align);
            prices.insert(next_id, price);
            cursor = price + u128::from(size.max(align).max(1));
            next_id += 1;
        }
        self.heap_cursor.set((next_id, cursor));
        prices[&id]
    }

    /// The price of an element position in the data buffer
    /// of a `Vec` value held in a frame local (see
    /// [`Self::price_target`]): the buffer's base in the
    /// buffer region plus the projection's byte displacement,
    /// so one buffer has one price base and its positions sit
    /// at exact element steps from it. `None` when the root
    /// is not a `Vec` place holding a vector value, the
    /// projection does not step into the buffer (its first
    /// structural step is not an element step — a position
    /// in the `Vec` value's own fields denotes the local's
    /// storage, not the buffer), or the element size cannot
    /// be priced. A zero-capacity vector has no buffer: every
    /// element position prices from the element type's
    /// dangling address displaced by the projection's
    /// offset, so an empty buffer has the one price domain
    /// too.
    fn vec_buffer_price(
        &self,
        frames: &[Frame],
        frame_id: u64,
        local: usize,
        projection: &[RefProj],
        root_ty: &Ty,
        root_elem_size: Option<u64>,
        leaf_ty: Option<&Ty>,
    ) -> Option<u128> {
        let Ty::Adt(name) = root_ty else {
            return None;
        };
        if name.as_str() != "std::vec::Vec" && name.as_str() != "alloc::vec::Vec" {
            return None;
        }
        match projection.first() {
            Some(RefProj::Index(_)) | Some(RefProj::Subslice { .. }) => {}
            _ => return None,
        }
        let frame = frames.iter().find(|f| f.id == frame_id)?;
        // The storage extent the held vector value accounts
        // (its capacity), in either of the value model's two
        // vector forms — the dense aggregate form, whose
        // extent is its element list, and the sparse form,
        // which records it. This is the capacity every other
        // capacity question in the model answers with (see
        // `vec_stored_len`).
        let cap = crate::builtins::vec_stored_len(frame.locals.get(local)?)?;
        // The element denomination: the root type's own
        // walk first (it resolves through the one recorded
        // instance of the `Vec` type), then the element size
        // conversion recorded for the local (the record that
        // exists precisely because the IR root type drops
        // the generic arguments), then the operand's
        // declared pointee type. A program holding several
        // `Vec` instances denominates through the fallbacks:
        // the bare root name alone is ambiguous there.
        let seq = self.sequence_elem(&WalkTy::Ty(root_ty.clone()));
        let elem_size = seq
            .as_ref()
            .and_then(|(size, _)| *size)
            .or(root_elem_size)
            .or_else(|| leaf_ty.and_then(|t| self.ty_size_full(t)))?;
        let align = seq
            .as_ref()
            .and_then(|(_, elem_ty)| match elem_ty {
                WalkTy::Ty(t) => self.ty_align(t),
                WalkTy::Tag(tag) => {
                    u64::try_from(self.layout_of(tag)?.1).ok()
                }
            })
            .or_else(|| leaf_ty.and_then(|t| self.ty_align(t)))?;
        let offset = self.projection_offset_with_root_elem(
            Some(root_ty),
            projection,
            leaf_ty,
            root_elem_size,
        )?;
        if cap == 0 {
            return displaced_price(dangling_address(align), offset);
        }
        let base = self.buffer_price(frame_id, local, cap, elem_size, align);
        displaced_price(base, offset)
    }

    /// The synthetic address of the data buffer of the
    /// `Vec` value held in the (`frame`, `local`) place,
    /// assigned from the buffer region on first pricing:
    /// each buffer reserves its capacity's extent
    /// (`cap x elem_size` — the extent the guest can observe
    /// the buffer spanning), so two live buffers' priced
    /// ranges never overlap, and a buffer's positions price
    /// at exact element steps from the one base for as long
    /// as the value stays in its local. A buffer keeps the
    /// base and the extent it was first priced with; a
    /// growth past that extent after first pricing is the
    /// one approximation, of the same kind [`Self::heap_price`]
    /// documents for an allocation whose layout is unknown.
    fn buffer_price(
        &self,
        frame: u64,
        local: usize,
        cap: u128,
        elem_size: u64,
        elem_align: u64,
    ) -> u128 {
        if let Some(p) = self.buffer_prices.borrow().get(&(frame, local)) {
            return *p;
        }
        let align = elem_align.max(1);
        let extent = u128::from(elem_size)
            .saturating_mul(cap)
            .max(u128::from(align))
            .max(1);
        let price = align_up_u128(self.buffer_cursor.get(), align);
        self.buffer_cursor.set(price + extent);
        self.buffer_prices
            .borrow_mut()
            .insert((frame, local), price);
        price
    }

    /// The price of a first-element position in an empty
    /// sequence value: the position denotes the sequence's
    /// buffer, and an empty sequence has no buffer storage
    /// in this model — its buffer pointer is the element
    /// type's dangling address, as an empty `Vec`'s data
    /// pointer is natively. `None` when the projection is
    /// not the first-element shape (`[Index(0)]` plus any
    /// trailing byte displacements), the root's current
    /// value is not an empty sequence aggregate, or the
    /// root type does not denote a sequence.
    fn empty_sequence_price(
        &self,
        frames: &[Frame],
        frame_id: u64,
        local: usize,
        projection: &[RefProj],
        root_ty: &Ty,
    ) -> Option<u128> {
        let mut end = projection.len();
        let mut byte_offset: i128 = 0;
        while end > 0 {
            match &projection[end - 1] {
                RefProj::ByteOffset(d) => {
                    byte_offset += d;
                    end -= 1;
                }
                _ => break,
            }
        }
        if end != 1 || !matches!(projection[0], RefProj::Index(0)) {
            return None;
        }
        let frame = frames.iter().find(|f| f.id == frame_id)?;
        let value = frame.locals.get(local)?;
        let empty = match value {
            Value::SparseVec { len, .. } => *len == 0,
            Value::Tuple(items) => items.is_empty(),
            Value::Enum { fields, .. } => fields.is_empty(),
            _ => false,
        };
        if !empty {
            return None;
        }
        let (_, elem_ty) =
            self.sequence_elem(&WalkTy::Ty(root_ty.clone()))?;
        let align = match &elem_ty {
            WalkTy::Ty(t) => self.ty_align(t)?,
            WalkTy::Tag(tag) => {
                u64::try_from(self.layout_of(tag)?.1).ok()?
            }
        };
        displaced_price(dangling_address(align), byte_offset)
    }

    /// The byte displacement a projection denotes from its
    /// root's base, computed from the program's layout
    /// tables — the arithmetic the synthetic-hash model
    /// cannot do: a field sits at its layout offset, an
    /// element at index × element size. `root_ty` is the
    /// root's declared type when the interpreter records
    /// it; without one, only the sequence positions a leaf
    /// type denominates from the end (a heap buffer's
    /// element positions) are priced. `None` when any step's
    /// displacement is not derivable — the caller falls
    /// back to synthetic-hash pricing for the whole target.
    fn projection_offset(
        &self,
        root_ty: Option<&Ty>,
        projection: &[RefProj],
        leaf_ty: Option<&Ty>,
    ) -> Option<i128> {
        self.projection_offset_with_root_elem(root_ty, projection, leaf_ty, None)
    }

    /// [`Self::projection_offset`] with the element size
    /// conversion recorded for the root local (see
    /// `ir::Body::local_elem_sizes`): the IR root type drops
    /// the generic arguments the size is computed from, so
    /// where the walk cannot size the root sequence's element
    /// from the type alone (an element mentioning an ADT
    /// whose bare name denotes several recorded instances),
    /// the recorded size denominates the root step. Only the
    /// root step is denominated this way; deeper steps walk
    /// the types as before.
    fn projection_offset_with_root_elem(
        &self,
        root_ty: Option<&Ty>,
        projection: &[RefProj],
        leaf_ty: Option<&Ty>,
        root_elem_size: Option<u64>,
    ) -> Option<i128> {
        match root_ty {
            Some(root) => {
                self.projection_offset_forward(root, projection, root_elem_size)
            }
            None => self.projection_offset_backward(projection, leaf_ty),
        }
    }

    /// [`Self::projection_offset`] with a known root type:
    /// walk the projection with the walk-type cursor the
    /// canonicalisation walk uses.
    fn projection_offset_forward(
        &self,
        root: &Ty,
        projection: &[RefProj],
        root_elem_size: Option<u64>,
    ) -> Option<i128> {
        let mut offset: i128 = 0;
        let mut cur = Some(WalkTy::Ty(root.clone()));
        let mut variant: u32 = 0;
        for (step_idx, step) in projection.iter().enumerate() {
            match step {
                RefProj::Field(idx) => {
                    let c = cur.as_ref()?;
                    offset += i128::from(self.field_offset(c, variant, *idx)?);
                    cur = self.field_walk_ty(c, variant, *idx);
                }
                RefProj::Index(i) => {
                    let c = cur.as_ref()?;
                    let Some((elem_size, elem_ty)) = self.sequence_elem(c)
                    else {
                        // The walk cannot denominate the root
                        // step (a bare ADT name denoting
                        // several recorded instances): the
                        // recorded root element size prices
                        // this step, and the walk cannot
                        // continue past it — the same
                        // fall-back a byte displacement makes.
                        let size = if step_idx == 0 {
                            root_elem_size?
                        } else {
                            return None;
                        };
                        offset +=
                            i128::try_from(*i).ok()? * i128::from(size);
                        cur = None;
                        continue;
                    };
                    let elem_size = elem_size
                        .or(if step_idx == 0 { root_elem_size } else { None })?;
                    offset +=
                        i128::try_from(*i).ok()? * i128::from(elem_size);
                    cur = Some(elem_ty);
                }
                RefProj::Subslice { offset: o, .. } => {
                    let c = cur.as_ref()?;
                    let Some((elem_size, _)) = self.sequence_elem(c) else {
                        // See the `Index` arm: the recorded
                        // root element size prices the root
                        // step the walk cannot denominate.
                        let size = if step_idx == 0 {
                            root_elem_size?
                        } else {
                            return None;
                        };
                        offset +=
                            i128::try_from(*o).ok()? * i128::from(size);
                        cur = None;
                        continue;
                    };
                    let elem_size = elem_size
                        .or(if step_idx == 0 { root_elem_size } else { None })?;
                    offset +=
                        i128::try_from(*o).ok()? * i128::from(elem_size);
                }
                RefProj::Downcast(v) => {
                    variant = *v;
                }
                RefProj::UnionField { .. } => {
                    // Every field of a union sits at its base;
                    // the position's type is not tracked past
                    // it, so a further structural step falls
                    // back.
                    cur = None;
                }
                RefProj::ByteOffset(d) => {
                    offset += *d;
                    // The position a byte displacement denotes
                    // has no tracked type: a further structural
                    // step falls back.
                    cur = None;
                }
                RefProj::View(view) => {
                    // The recorded denomination (see
                    // `RefProj::View`) does not move the
                    // position; it denominates the steps
                    // after it.
                    cur = Some(WalkTy::Ty(view.clone()));
                    variant = 0;
                }
                RefProj::VecLen => return None,
            }
        }
        Some(offset)
    }

    /// The value of a reference (or raw pointer) formed over a
    /// place whose resolution crosses a dereference of a
    /// storage-less pointer -- a null pointer, or an address no
    /// allocation owns (the dangling address of a zero-sized
    /// allocation, an integer cast to a pointer). The pointee
    /// has no storage in the model, so no resolved place denotes
    /// it; but forming the pointer is still meaningful, and its
    /// value is the address the projection denotes: the
    /// pointer's own address displaced by the structural offset
    /// of the steps past the dereference, carrying the fat
    /// metadata the source pointer (or the destination type)
    /// gives it. The shapes are `&*p` for a `p` built from an
    /// integer, and `&raw const (*p).field` in address-of code
    /// that never reads through the result. `None` when the
    /// place does not have this shape -- no dereference, a
    /// pointer that denotes storage, a step the offset walk
    /// cannot price -- leaving the caller's original resolution
    /// error in place.
    pub(crate) fn resolve_place_as_address(
        &self,
        frames: &[Frame],
        body: &Body,
        place: &Place,
        dest_ty: Option<&Ty>,
    ) -> Option<Value> {
        let deref_pos = place
            .projection
            .iter()
            .position(|s| matches!(s, ProjectionElem::Deref))?;
        let prefix = Place {
            local: place.local,
            projection: place.projection[..deref_pos].to_vec(),
        };
        let pointer = self.read_place(frames, body, &prefix).ok()?;
        let (base, mut fat_len) = match &pointer {
            Value::Ref(RefTarget::Null) => (0i128, None),
            Value::Ref(RefTarget::Address(addr))
                if self.address_owner(*addr).is_none() =>
            {
                (i128::from(*addr), None)
            }
            Value::FatRef { base, len } => {
                let base = match &**base {
                    Value::Ref(RefTarget::Null) => 0i128,
                    Value::Ref(RefTarget::Address(addr))
                        if self.address_owner(*addr).is_none() =>
                    {
                        i128::from(*addr)
                    }
                    _ => return None,
                };
                (base, Some(*len))
            }
            _ => return None,
        };
        let pointee_ty = self.place_pointee_ty(body, &prefix)?;
        // The length of the sequence the dereference denotes,
        // where a step counts back from its end: the fat
        // pointer's own metadata, or the array pointee's length.
        let seq_len = fat_len.or(match &pointee_ty {
            Ty::Array(_, n) => Some(u128::from(*n)),
            _ => None,
        });
        let mut steps: Vec<RefProj> = Vec::new();
        for step in &place.projection[deref_pos + 1..] {
            match step {
                ProjectionElem::Field(idx) => {
                    steps.push(RefProj::Field(*idx))
                }
                ProjectionElem::UnionField { idx, table } => {
                    steps.push(RefProj::UnionField {
                        idx: *idx,
                        table: *table,
                    })
                }
                ProjectionElem::Downcast(v) => {
                    steps.push(RefProj::Downcast(*v))
                }
                ProjectionElem::Index(local) => {
                    let idx = self
                        .read_index_local(frames, body, *local, place)
                        .ok()?;
                    steps.push(RefProj::Index(idx));
                }
                ProjectionElem::ConstantIndex { offset, from_end } => {
                    if *from_end {
                        let len = seq_len?;
                        let idx = len.checked_sub(1 + u128::from(*offset))?;
                        steps.push(RefProj::Index(usize::try_from(idx).ok()?));
                    } else {
                        steps.push(RefProj::Index(
                            usize::try_from(*offset).ok()?,
                        ));
                    }
                }
                ProjectionElem::Subslice { from, to, from_end } => {
                    let (off, len) = if *from_end {
                        let total = seq_len?;
                        let end = total.checked_sub(u128::from(*to))?;
                        let start = u128::from(*from);
                        (start, end.checked_sub(start)?)
                    } else {
                        (
                            u128::from(*from),
                            u128::from(*to).checked_sub(u128::from(*from))?,
                        )
                    };
                    steps.push(RefProj::Subslice {
                        offset: usize::try_from(off).ok()?,
                        len: usize::try_from(len).ok()?,
                    });
                    fat_len = Some(len);
                }
                ProjectionElem::Deref => return None,
            }
        }
        let offset = self.projection_offset_forward(&pointee_ty, &steps, None)?;
        let addr = base.checked_add(offset)?;
        let thin = Value::Ref(RefTarget::Address(u64::try_from(addr).ok()?));
        // A pointer to an unsized pointee is fat: it carries the
        // length the source pointer (or the array pointee, or a
        // sub-slice step) determines.
        let dest_inner = match dest_ty {
            Some(Ty::Ref(inner)) => Some(&**inner),
            _ => None,
        };
        match dest_inner {
            Some(Ty::Slice(_)) | Some(Ty::Str) => {
                let len = fat_len.or(seq_len)?;
                Some(Value::FatRef {
                    base: Box::new(thin),
                    len,
                })
            }
            _ => Some(thin),
        }
    }

    /// The pointee type of the pointer a place holds: the
    /// place's declared type walked across its projection at
    /// the type level, then one reference step in. `None` when
    /// any step's type is not recorded, or the place does not
    /// hold a pointer.
    fn place_pointee_ty(&self, body: &Body, place: &Place) -> Option<Ty> {
        let mut ty = body.local_ty(place.local)?.clone();
        let mut variant = 0usize;
        for step in &place.projection {
            ty = match step {
                ProjectionElem::Field(idx) => {
                    let next = self.adt_field_ty(&ty, variant, *idx)?;
                    variant = 0;
                    next
                }
                ProjectionElem::UnionField { idx, .. } => {
                    let next = self.adt_field_ty(&ty, 0, *idx)?;
                    variant = 0;
                    next
                }
                ProjectionElem::Downcast(v) => {
                    variant = *v as usize;
                    ty
                }
                ProjectionElem::Deref => {
                    variant = 0;
                    match ty {
                        Ty::Ref(inner) => (*inner).clone(),
                        _ => return None,
                    }
                }
                ProjectionElem::Index(_)
                | ProjectionElem::ConstantIndex { .. } => match ty {
                    Ty::Array(elem, _) | Ty::Slice(elem) => (*elem).clone(),
                    _ => return None,
                },
                ProjectionElem::Subslice { .. } => ty,
            };
        }
        match ty {
            Ty::Ref(inner) => Some((*inner).clone()),
            _ => None,
        }
    }

    /// [`Self::projection_offset`] without a root type: only
    /// a single sequence step (denominated by the leaf type)
    /// among byte displacements is derivable.
    fn projection_offset_backward(
        &self,
        projection: &[RefProj],
        leaf_ty: Option<&Ty>,
    ) -> Option<i128> {
        let mut offset: i128 = 0;
        let mut sequence_step = false;
        for step in projection {
            match step {
                RefProj::ByteOffset(d) => offset += d,
                RefProj::Index(i) => {
                    if sequence_step {
                        return None;
                    }
                    sequence_step = true;
                    let elem = self.ty_size_full(leaf_ty?)?;
                    offset += i128::try_from(*i).ok()? * i128::from(elem);
                }
                RefProj::Subslice { offset: o, .. } => {
                    if sequence_step {
                        return None;
                    }
                    sequence_step = true;
                    let elem = self.backward_elem_size(leaf_ty?)?;
                    offset += i128::try_from(*o).ok()? * i128::from(elem);
                }
                _ => return None,
            }
        }
        Some(offset)
    }

    /// The element size of the sequence a leaf type denotes,
    /// for the backward offset walk: the leaf is the
    /// sub-sequence itself, so its element is one level in.
    fn backward_elem_size(&self, leaf: &Ty) -> Option<u64> {
        match leaf {
            Ty::Slice(elem) | Ty::Array(elem, _) => self.ty_size_full(elem),
            Ty::Ref(inner) => self.backward_elem_size(inner),
            _ => None,
        }
    }

    /// The byte offset of field `idx` of the value a walk
    /// type denotes, from the program's recorded field
    /// layouts (an ADT instance) or the computed tuple
    /// layout.
    fn field_offset(
        &self,
        cur: &WalkTy,
        variant: u32,
        idx: usize,
    ) -> Option<u64> {
        match cur {
            WalkTy::Ty(Ty::Tuple(ts)) => {
                self.tuple_layout(ts)?.0.get(idx).map(|&(off, _)| off)
            }
            WalkTy::Ty(Ty::Adt(name)) => {
                let key = self.adt_instance_key(name)?;
                self.program()
                    .adt_field_layouts
                    .get(&key)?
                    .get(variant as usize)?
                    .get(idx)
                    .map(|&(off, _)| off)
            }
            WalkTy::Tag(tag) => {
                let tag = tag.trim();
                let variants = self
                    .program()
                    .adt_field_layouts
                    .get(tag)
                    .or_else(|| {
                        let key = self.adt_instance_key(tag)?;
                        self.program().adt_field_layouts.get(&key)
                    });
                if let Some(variants) = variants {
                    return variants
                        .get(variant as usize)?
                        .get(idx)
                        .map(|&(off, _)| off);
                }
                // A tuple's rendered tag: compute the layout
                // from the parts' recorded layouts.
                if tag.starts_with('(') && tag.ends_with(')') {
                    let parts = split_top_level_str(&tag[1..tag.len() - 1]);
                    let mut cursor: u128 = 0;
                    for (i, part) in parts.iter().enumerate() {
                        let (size, align) = self.layout_of(part.trim())?;
                        cursor = align_up_u128(
                            cursor,
                            u64::try_from(align).ok()?,
                        );
                        if i == idx {
                            return u64::try_from(cursor).ok();
                        }
                        cursor += size;
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// The address identity key of a pointer value with
    /// zero-sized element positions normalised (see
    /// [`Self::pointer_address`]), together with the trailing
    /// byte displacement the key's slot is displaced by (see
    /// [`split_trailing_byte_offsets`]); `None` exactly when
    /// [`address_key`] has no key for the value.
    fn canonical_address_key(
        &self,
        frames: &[Frame],
        value: &Value,
        leaf_ty: Option<&Ty>,
    ) -> Option<(String, i128)> {
        match value {
            Value::Ref(target) => {
                let target = self.canonical_target(frames, target, leaf_ty);
                let (target, offset) = split_trailing_byte_offsets(&target);
                Some((format!("ref:{target:?}"), offset))
            }
            Value::FatRef { base, len } => {
                // The base target's element positions are
                // denominated in the slice's element type.
                let elem = match leaf_ty {
                    Some(Ty::Slice(elem)) => Some(&**elem),
                    _ => None,
                };
                let base = self.canonical_base_value(frames, base, elem);
                let (base, offset) = strip_trailing_offsets_value(&base);
                Some((format!("fat:{base:?}:{len}"), offset))
            }
            Value::DynRef { base, .. } => {
                // A trait-object pointer's address identity is
                // its data pointer's: the vtable it carries is
                // metadata, not part of the address -- the same
                // delegation `arithmetic_price` makes for the
                // arithmetic domain. Keying the wrapper itself
                // would give a borrow that recovered its `dyn`
                // identity at formation (see
                // `Interpreter::dyn_borrow_source`) a different
                // address from the thin borrow of the very same
                // place, splitting field-offset pointer
                // comparisons between the two forms. The vtable
                // distinction between two wide pointers is kept
                // in `eval_binary`'s comparison key, not here.
                self.canonical_address_key(frames, base, leaf_ty)
            }
            _ => {
                let (stripped, offset) = strip_trailing_offsets_value(value);
                address_key(&stripped).map(|key| (key, offset))
            }
        }
    }

    /// A fat pointer's base value with its target normalised
    /// (see [`Self::canonical_target`]); a non-reference base
    /// passes through unchanged.
    fn canonical_base_value(
        &self,
        frames: &[Frame],
        base: &Value,
        leaf_ty: Option<&Ty>,
    ) -> Value {
        match base {
            Value::Ref(target) => Value::Ref(self.canonical_target(frames, target, leaf_ty)),
            _ => base.clone(),
        }
    }

    /// The instance key (the rendered name the program's layout
    /// and debug-shape tables are keyed by) of the ADT instance
    /// a bare ADT name denotes in this program: the name itself
    /// when it is a key, else the one table key with that base
    /// name when exactly one exists. The IR's ADT types carry
    /// only the definition's name — the generic arguments live
    /// only in the instance tables — and the driver records a
    /// layout and a debug shape for every ADT instance a
    /// converted body mentions, so a bare name with exactly one
    /// recorded instance can only denote that instance. Two or
    /// more recorded instances make the name ambiguous here and
    /// resolve to `None`, as does an unrecorded name.
    pub(crate) fn adt_instance_key(&self, name: &str) -> Option<String> {
        if let Some(hit) = self.adt_instance_keys.borrow().get(name) {
            return hit.clone();
        }
        let program = self.program();
        let resolved = if program.adt_layouts.contains_key(name)
            || program.adt_debug_shapes.contains_key(name)
        {
            Some(name.to_string())
        } else {
            let mut found: Option<String> = None;
            let mut ambiguous = false;
            for key in program
                .adt_layouts
                .keys()
                .chain(program.adt_debug_shapes.keys())
            {
                let base = key.split('<').next().unwrap_or(key.as_str());
                if base == name {
                    match &found {
                        None => found = Some(key.clone()),
                        Some(prev) if *prev != *key => ambiguous = true,
                        _ => {}
                    }
                }
            }
            if ambiguous {
                None
            } else {
                found
            }
        };
        self.adt_instance_keys
            .borrow_mut()
            .insert(name.to_string(), resolved.clone());
        resolved
    }

    /// The element size a trailing element step is denominated
    /// in, from the type of the value at the step's position
    /// (the operand's declared pointee type): the element type
    /// itself for a thin pointer, the element of an array or
    /// slice pointee. An ADT element's size comes from the
    /// layout of the one instance the name denotes in this
    /// program (see [`Self::adt_instance_key`]).
    fn leaf_elem_size(&self, leaf: &Ty) -> Option<u64> {
        match leaf {
            Ty::Array(elem, _) | Ty::Slice(elem) => self.ty_size(elem),
            Ty::Adt(name) => self
                .adt_instance_key(name)
                .and_then(|key| self.program().adt_layouts.get(&key).map(|&(size, _)| size)),
            other => self.ty_size(other),
        }
    }

    /// A reference target with its zero-sized element positions
    /// normalised to the sequence base (see
    /// [`Self::pointer_address`]). The walk that prices each
    /// element step starts from the root's declared type where
    /// the interpreter records one — a place's local type, a
    /// static's declared type — and from the operand's pointee
    /// type (`leaf_ty`), walking the projection backwards from
    /// its trailing element step, where it does not (allocations
    /// and constant aggregates record no root type). A step the
    /// walk cannot price keeps its index: the normalisation only
    /// ever merges positions the types prove share one address.
    fn canonical_target(
        &self,
        frames: &[Frame],
        target: &RefTarget,
        leaf_ty: Option<&Ty>,
    ) -> RefTarget {
        match target {
            RefTarget::Place {
                frame,
                local,
                projection,
            } => {
                let root = frames
                    .iter()
                    .find(|f| f.id == *frame)
                    .and_then(|f| self.program().functions.get(&f.func))
                    .and_then(|body| body.local_tys.get(*local))
                    .cloned();
                // The stored root value vetoes a zero-sized
                // reading of its positions: a value tree with
                // scalar content belongs to a type of nonzero
                // size, whatever a comparing operand claims.
                let veto_zero = frames
                    .iter()
                    .find(|f| f.id == *frame)
                    .and_then(|f| f.locals.get(*local))
                    .map(|v| value_tree_has_scalar(v))
                    .unwrap_or(false);
                match root {
                    Some(root) => RefTarget::Place {
                        frame: *frame,
                        local: *local,
                        projection: self.canonical_projection(
                            &WalkTy::Ty(root),
                            projection,
                            leaf_ty,
                            veto_zero,
                        ),
                    },
                    None => target.clone(),
                }
            }
            RefTarget::Static { name, projection } => {
                match self.program().static_tys.get(name).cloned() {
                    Some(root) => RefTarget::Static {
                        name: name.clone(),
                        projection: self.canonical_projection(
                            &WalkTy::Ty(root),
                            projection,
                            leaf_ty,
                            false,
                        ),
                    },
                    None => target.clone(),
                }
            }
            RefTarget::Alloc { id, projection } => RefTarget::Alloc {
                id: *id,
                projection: match leaf_ty {
                    Some(leaf) => self.canonical_projection_bwd(projection, leaf),
                    None => projection.clone(),
                },
            },
            RefTarget::Value(base, projection) => {
                let projection = match leaf_ty {
                    Some(leaf) => self.canonical_projection_bwd(projection, leaf),
                    None => projection.clone(),
                };
                // A value materialised from a type-erased
                // constant block is a typed view of that one
                // allocation, not a new one: a pointer
                // denoting the view carries the block's
                // identity, so it compares and converts to
                // an address exactly as a pointer to the
                // block does (see [`Self::note_mat_origin`]).
                let base = match self.mat_origins.borrow().get(&format!("{base:?}")) {
                    Some(origin) => Box::new(origin.clone()),
                    None => base.clone(),
                };
                RefTarget::Value(base, projection)
            }
            RefTarget::Null | RefTarget::Address(_) => target.clone(),
        }
    }

    /// The canonical form of a projection for address purposes,
    /// walked from the root type: an `Index` or `Subslice` step
    /// into a sequence whose element size is zero denotes the
    /// sequence's base position (its byte displacement is zero
    /// whatever the index), so the step is dropped; every other
    /// step is kept verbatim. A step the walk cannot price keeps
    /// its index, and steps past a position whose type the walk
    /// cannot determine are kept as well: the walk prices a step
    /// only from the type at that step.
    fn canonical_projection(
        &self,
        root: &WalkTy,
        projection: &[RefProj],
        leaf_ty: Option<&Ty>,
        veto_zero: bool,
    ) -> Vec<RefProj> {
        let mut out = Vec::with_capacity(projection.len());
        let mut cur: Option<WalkTy> = Some(root.clone());
        let mut variant: u32 = 0;
        for (i, step) in projection.iter().enumerate() {
            match step {
                RefProj::Downcast(v) => {
                    variant = *v;
                    out.push(step.clone());
                }
                RefProj::Field(idx) => {
                    out.push(step.clone());
                    cur = match cur {
                        Some(c) => self.field_walk_ty(&c, variant, *idx),
                        None => None,
                    };
                }
                RefProj::Index(_) | RefProj::Subslice { .. } => {
                    let step_cur = cur.clone();
                    let elem = cur.as_ref().and_then(|c| self.sequence_elem(c));
                    let (mut size, next) = match elem {
                        Some((size, next)) => (size, Some(next)),
                        None => (None, None),
                    };
                    // The trailing step may still be priced
                    // from the operand's declared pointee type
                    // when the walk from the root cannot price
                    // it (a generic collection root whose
                    // element type the IR type erases) — but
                    // only under the guards of
                    // [`Self::trailing_zero_ok`]: generic
                    // machinery also compares positions through
                    // pointers cast to `*const ()`, whose
                    // pointee type denominates nothing.
                    if size.is_none() && i + 1 == projection.len() && !veto_zero {
                        if let Some(leaf) = leaf_ty {
                            if self.trailing_zero_ok(&step_cur, leaf) {
                                size = Some(0);
                            }
                        }
                    }
                    if size != Some(0) {
                        out.push(step.clone());
                    }
                    if matches!(step, RefProj::Index(_)) {
                        cur = next;
                    }
                }
                RefProj::View(view) => {
                    // The recorded denomination (see
                    // `RefProj::View`) is part of the
                    // target's identity, and denominates the
                    // steps after it for this walk as it
                    // does for the byte view.
                    out.push(step.clone());
                    cur = Some(WalkTy::Ty(view.clone()));
                    variant = 0;
                }
                RefProj::ByteOffset(_)
                | RefProj::UnionField { .. }
                | RefProj::VecLen => {
                    out.push(step.clone());
                    cur = None;
                }
            }
        }
        out
    }

    /// Whether a trailing element step the root walk could not
    /// price may be read as zero-sized from the comparing
    /// operand's declared pointee type (`leaf`): the leaf must
    /// itself be zero-sized, and where the step's sequence is a
    /// bare ADT name, the claim must be consistent with the
    /// instances of that ADT the program records — at least one
    /// recorded instance must have a zero-sized (or unpriceable)
    /// element. A claim no recorded instance supports is a cast
    /// artefact (generic sort machinery compares element
    /// positions through `*const ()` pointers), not evidence
    /// about this root.
    fn trailing_zero_ok(&self, cur: &Option<WalkTy>, leaf: &Ty) -> bool {
        if self.leaf_elem_size(leaf) != Some(0) {
            return false;
        }
        match cur {
            Some(WalkTy::Ty(Ty::Adt(name))) => {
                let program = self.program();
                let mut keys: Vec<&String> = Vec::new();
                for key in program
                    .adt_layouts
                    .keys()
                    .chain(program.adt_debug_shapes.keys())
                {
                    if key.split('<').next().unwrap_or(key.as_str()) == name.as_str()
                        && !keys.contains(&key)
                    {
                        keys.push(key);
                    }
                }
                if keys.is_empty() {
                    return true;
                }
                keys.iter().any(|key| {
                    match tag_sequence_elem(key)
                        .and_then(|elem| self.layout_of(elem))
                        .and_then(|(size, _)| u64::try_from(size).ok())
                    {
                        Some(0) => true,
                        Some(_) => false,
                        None => true,
                    }
                })
            }
            _ => true,
        }
    }

    /// The canonical form of a projection whose root type is
    /// not recorded (an allocation's, a constant aggregate's),
    /// walked backwards from the trailing element step with the
    /// operand's pointee type `leaf`: the trailing `Index` /
    /// `Subslice` step is denominated in `leaf`'s element size
    /// (the element type itself for a thin pointer, the element
    /// of an array or slice pointee), and is dropped when that
    /// size is zero. Steps before it are kept verbatim: walking
    /// backwards cannot price them.
    fn canonical_projection_bwd(&self, projection: &[RefProj], leaf: &Ty) -> Vec<RefProj> {
        let elem_size = self.leaf_elem_size(leaf);
        let mut out: Vec<RefProj> = Vec::with_capacity(projection.len());
        let mut priced = true;
        for step in projection.iter().rev() {
            match step {
                RefProj::Index(_) | RefProj::Subslice { .. } if priced => {
                    if elem_size != Some(0) {
                        out.push(step.clone());
                    }
                    priced = false;
                }
                _ => out.push(step.clone()),
            }
        }
        out.reverse();
        out
    }

    /// The element (size, type) of the sequence a walk type
    /// denotes, when it denotes one: arrays and slices directly,
    /// a pointer position through its pointee, and an ADT
    /// instance through its first generic argument (the element
    /// type of the modelled collections whose element positions
    /// a projection indexes directly). The size is `None` when
    /// the element type's layout is not one the interpreter can
    /// price; the element type is still reported, so the walk
    /// can continue past the step.
    fn sequence_elem(&self, cur: &WalkTy) -> Option<(Option<u64>, WalkTy)> {
        match cur {
            WalkTy::Ty(Ty::Array(elem, _)) | WalkTy::Ty(Ty::Slice(elem)) => {
                Some((self.ty_size_full(elem), WalkTy::Ty((**elem).clone())))
            }
            WalkTy::Ty(Ty::Ref(inner)) => match &**inner {
                Ty::Array(elem, _) | Ty::Slice(elem) => {
                    Some((self.ty_size_full(elem), WalkTy::Ty((**elem).clone())))
                }
                _ => Some((self.ty_size_full(inner), WalkTy::Ty((**inner).clone()))),
            },
            WalkTy::Ty(Ty::Adt(name)) => {
                let key = self.adt_instance_key(name)?;
                let elem = tag_sequence_elem(&key)?;
                let size = self
                    .layout_of(elem)
                    .and_then(|(size, _)| u64::try_from(size).ok());
                Some((size, WalkTy::Tag(elem.to_string())))
            }
            WalkTy::Tag(tag) => {
                let elem = tag_sequence_elem(tag)?;
                let size = self
                    .layout_of(elem)
                    .and_then(|(size, _)| u64::try_from(size).ok());
                Some((size, WalkTy::Tag(elem.to_string())))
            }
            _ => None,
        }
    }

    /// The type of field `idx` of the value a walk type denotes:
    /// a tuple's element, or an ADT variant's field from the
    /// program's debug-shape table (whose field types are
    /// rendered tags).
    fn field_walk_ty(&self, cur: &WalkTy, variant: u32, idx: usize) -> Option<WalkTy> {
        match cur {
            WalkTy::Ty(Ty::Tuple(ts)) => ts.get(idx).map(|t| WalkTy::Ty(t.clone())),
            WalkTy::Ty(Ty::Adt(name)) => self.adt_instance_field_tag(name, variant, idx).map(WalkTy::Tag),
            WalkTy::Tag(tag) => {
                let tag = tag.trim();
                if tag.starts_with('(') && tag.ends_with(')') {
                    let parts = split_top_level_str(&tag[1..tag.len() - 1]);
                    parts
                        .get(idx)
                        .map(|part| WalkTy::Tag((*part).to_string()))
                } else {
                    self.adt_instance_field_tag(tag, variant, idx).map(WalkTy::Tag)
                }
            }
            _ => None,
        }
    }

    /// The rendered type tag of field `idx` of variant
    /// `variant` of the ADT instance `name`, from the program's
    /// debug-shape table.
    fn adt_instance_field_tag(&self, name: &str, variant: u32, idx: usize) -> Option<String> {
        let key = self.adt_instance_key(name)?;
        let shape = self.program().adt_debug_shapes.get(&key)?;
        let variant = shape
            .variants
            .get(variant as usize)
            .or_else(|| shape.variants.first())?;
        variant.fields.get(idx).map(|field| field.ty.clone())
    }

    /// Absorb a pointer offset into an element index for an
    /// allocation whose declared type its creation point
    /// recorded (see the `alloc_tys` field): the recorded type
    /// is a sequence (an array or a slice), so a byte
    /// displacement over the allocation is denominated by the
    /// sequence's element size, exactly as a static array's
    /// offset is denominated by the program's recorded static
    /// type. The position must sit at the sequence itself: an
    /// optional trailing index or byte-offset step composes
    /// into the byte position first, and any other step means
    /// the position is inside a sub-value, whose denomination
    /// is not the sequence's. A displacement that lands
    /// between elements, or outside the recorded array's
    /// bounds (one past the end included, as a pointer there
    /// may be formed), answers `None` and the caller carries
    /// the displacement as before.
    fn alloc_typed_offset_absorb(
        &self,
        id: u64,
        projection: &[RefProj],
        disp: Option<i128>,
    ) -> Option<Vec<RefProj>> {
        let ty = self.alloc_ty(id)?;
        let (elem, len) = match &ty {
            Ty::Array(elem, n) => (self.ty_size(elem)?, Some(*n)),
            Ty::Slice(elem) => (self.ty_size(elem)?, None),
            _ => return None,
        };
        if elem == 0 {
            return None;
        }
        let elem = i128::from(elem);
        let disp = disp?;
        let (head, cur) = match projection.split_last() {
            Some((RefProj::Index(i), head)) => {
                (head, i128::try_from(*i).ok()?.checked_mul(elem)?)
            }
            Some((RefProj::ByteOffset(d), head)) => (head, *d),
            None => (projection, 0),
            _ => return None,
        };
        if !head.is_empty() {
            return None;
        }
        let total = cur.checked_add(disp)?;
        if total < 0 || total % elem != 0 {
            return None;
        }
        let idx = total / elem;
        if let Some(n) = len {
            if idx > i128::from(n) {
                return None;
            }
        }
        let idx = usize::try_from(idx).ok()?;
        Some(if idx == 0 && projection.is_empty() {
            Vec::new()
        } else {
            vec![RefProj::Index(idx)]
        })
    }

    /// The value of a pointer offset (see [`Rvalue::PtrOffset`]):
    /// `base` advanced by `count` elements of `elem_size` bytes
    /// each (the size may be unknown, in which case only the
    /// Absorb a pointer offset into an element index for a
    /// raw allocation block (see the `Alloc` arm of
    /// `offset_target`). Answers `None` when the block's
    /// contents are not element-shaped, the element size is
    /// unknown, or the resulting position is not
    /// element-aligned; the caller then carries the byte
    /// displacement as before.
    fn alloc_offset_absorb(
        &self,
        id: u64,
        projection: &[RefProj],
        elem_size: Option<u64>,
        disp: Option<i128>,
        count: i128,
    ) -> Option<InterpResult<Vec<RefProj>>> {
        let size = elem_size?;
        if !self.is_elem_block(id) {
            return None;
        }
        // Only the block's shape is judged here, so the store
        // is borrowed for the check: cloning the block's whole
        // contents (a raw table's bucket array, say) to learn
        // that it is a tuple would price every pointer offset
        // at the block's full size.
        {
            let store = self.allocs.borrow();
            match store.get(&id) {
                Some(Value::Tuple(_)) => {}
                // A zeroed block holds bytes; only byte elements
                // index it faithfully.
                Some(Value::Bytes(_)) if size == 1 => {}
                _ => return None,
            }
        }
        let disp = disp?;
        let (base, carried) = match projection.last() {
            Some(RefProj::ByteOffset(d)) => {
                (&projection[..projection.len() - 1], *d)
            }
            _ => (projection, 0),
        };
        let total = carried.checked_add(disp)?;
        // A pointer already denominated in bytes (its index was
        // rewritten by a differently sized pointer cast, see
        // `PtrCastUnits`) with nothing before the byte offset
        // stays in bytes: converting the total back into the
        // current element size's index would silently change
        // the position whenever the two denominations meet —
        // the byte-denominated bucket pointers of a raw table
        // versus its element-denominated ones. Byte offsets
        // resolve against the block's byte slots directly.
        if base.is_empty() && matches!(projection.last(), Some(RefProj::ByteOffset(_))) {
            return Some(Ok(vec![RefProj::ByteOffset(total)]));
        }
        let size_i = i128::from(size);
        // A zero-sized element type denominates its positions
        // in elements alone: the byte displacement is always
        // zero, so the offset's element count is the step
        // count (and a nonzero byte total is no element
        // position at all).
        let steps = if size_i == 0 {
            if total != 0 {
                return None;
            }
            count
        } else {
            if total % size_i != 0 {
                return None;
            }
            total / size_i
        };
        let idx = match base {
            [] => steps,
            [RefProj::Index(j)] => i128::try_from(*j).ok()?.checked_add(steps)?,
            _ => return None,
        };
        if idx < 0 {
            return Some(Err(InterpError::UnsupportedRvalue {
                kind: "pointer offset: negative element index into an allocation"
                    .to_string(),
            }));
        }
        let idx = match usize::try_from(idx) {
            Ok(i) => i,
            Err(_) => {
                return Some(Err(InterpError::UnsupportedRvalue {
                    kind: "pointer offset: element index out of range".to_string(),
                }))
            }
        };
        Some(Ok(vec![RefProj::Index(idx)]))
    }

    /// shapes that need no size — a zero count — evaluate).
    /// `wrapping` selects the `arith_offset` intrinsic's semantics
    /// over `BinaryOp::Offset`'s: the two compute the same denoted
    /// position wherever both are defined, and differ only for a
    /// null base, which wrapping turns into the bare address and
    /// the checked form rejects as the undefined behaviour it is.
    pub(crate) fn ptr_offset_value(
        &self,
        frames: &[Frame],
        base: Value,
        count: i128,
        elem_size: Option<u64>,
        wrapping: bool,
    ) -> InterpResult<Value> {
        if count == 0 {
            return Ok(base);
        }
        // The byte displacement, when the element size is known.
        // A displacement that overflows an i128 is not a position
        // in any allocation and has no honest rendering.
        let disp: Option<i128> = match elem_size {
            Some(size) => Some(count.checked_mul(i128::from(size)).ok_or_else(|| {
                InterpError::UnsupportedRvalue {
                    kind: "pointer offset: element displacement overflows".to_string(),
                }
            })?),
            None => None,
        };
        match base {
            Value::Ref(target) => Ok(Value::Ref(
                self.offset_target(target, frames, count, elem_size, disp, wrapping)?,
            )),
            Value::FatRef { base, len } => match *base {
                Value::Ref(_) => {
                    let base = self.ptr_offset_value(frames, *base, count, elem_size, wrapping)?;
                    Ok(Value::FatRef {
                        base: Box::new(base),
                        len,
                    })
                }
                Value::Tuple(elems) => {
                    // A self-contained slice constant: the offset
                    // data pointer denotes the suffix starting at
                    // the count, with the length metadata unchanged
                    // (exactly what offsetting a fat pointer does).
                    let n = usize::try_from(count).map_err(|_| {
                        InterpError::UnsupportedRvalue {
                            kind: "pointer offset: negative offset of a slice constant"
                                .to_string(),
                        }
                    })?;
                    if n > elems.len() {
                        return Err(InterpError::UnsupportedRvalue {
                            kind: "pointer offset: offset past the end of a slice constant"
                                .to_string(),
                        });
                    }
                    Ok(Value::FatRef {
                        base: Box::new(Value::Tuple(elems[n..].to_vec())),
                        len,
                    })
                }
                ref other => Err(InterpError::type_mismatch(
                    "pointer offset",
                    "fat pointer over a reference or a slice constant",
                    other.kind_name(),
                )),
            },
            Value::DynRef {
                base,
                concrete,
                traits,
                drop_glue,
                type_id,
            } => match *base {
                Value::Ref(_) => {
                    let base = self.ptr_offset_value(frames, *base, count, elem_size, wrapping)?;
                    Ok(Value::DynRef {
                        base: Box::new(base),
                        concrete,
                        traits,
                        drop_glue,
                        type_id,
                    })
                }
                ref other => Err(InterpError::type_mismatch(
                    "pointer offset",
                    "dyn pointer over a reference",
                    other.kind_name(),
                )),
            },
            other => Err(InterpError::type_mismatch(
                "pointer offset",
                "pointer value",
                other.kind_name(),
            )),
        }
    }

    /// A reference target advanced by a pointer offset (see
    /// [`Self::ptr_offset_value`]).
    pub(crate) fn offset_target(
        &self,
        target: RefTarget,
        frames: &[Frame],
        count: i128,
        elem_size: Option<u64>,
        disp: Option<i128>,
        wrapping: bool,
    ) -> InterpResult<RefTarget> {
        match target {
            RefTarget::Place {
                frame,
                local,
                projection,
            } => {
                let base = self
                    .frame_pos(frames, frame)
                    .ok()
                    .and_then(|pos| frames[pos].locals.get(local));
                let projection = self.offset_projection(
                    frames,
                    Some((frame, local)),
                    None,
                    projection,
                    disp,
                    elem_size,
                    base,
                    count,
                    false,
                )?;
                Ok(RefTarget::Place {
                    frame,
                    local,
                    projection,
                })
            }
            RefTarget::Static { name, projection } => {
                // The driver records the declared types of the
                // target crate's statics in the program, so an
                // offset over a static array is denominated by
                // that type, exactly as for a local; a static
                // without a recorded type (a foreign item) carries
                // the displacement unless the static value's own
                // shape absorbs it.
                let base = self.static_base(&name).ok();
                let projection = self.offset_projection(
                    frames,
                    None,
                    Some(&name),
                    projection,
                    disp,
                    elem_size,
                    base.as_ref(),
                    count,
                    false,
                )?;
                Ok(RefTarget::Static { name, projection })
            }
            RefTarget::Alloc { id, projection } => {
                // A raw block from the allocator builtins holds
                // its elements as a tuple (or, for a zeroed byte
                // block, as bytes): an offset in element units
                // is absorbed into an element index, the form
                // the block's contents are read and written
                // through. Any other pointee shape is not a
                // declared local type either, so no absorption
                // is attempted and the displacement is carried,
                // as for a static.
                if let Some(absorbed) =
                    self.alloc_offset_absorb(id, &projection, elem_size, disp, count)
                {
                    return absorbed.map(|projection| RefTarget::Alloc { id, projection });
                }
                if let Some(projection) =
                    self.alloc_typed_offset_absorb(id, &projection, disp)
                {
                    return Ok(RefTarget::Alloc { id, projection });
                }
                // The projection computation reads the block
                // through a reference; the store is borrowed for
                // the call rather than cloned (see
                // `alloc_offset_absorb`).
                let store = self.allocs.borrow();
                let base = store.get(&id);
                let projection = self.offset_projection(
                    frames,
                    None,
                    None,
                    projection,
                    disp,
                    elem_size,
                    base,
                    count,
                    true,
                )?;
                Ok(RefTarget::Alloc { id, projection })
            }
            RefTarget::Value(v, projection) => {
                if projection.is_empty() {
                    // An unprojected value target denotes the
                    // aggregate itself: for the element-shaped
                    // constants (tuples, byte strings) the offset
                    // is applied eagerly to the constant value.
                    // Any other aggregate's storage is a byte
                    // extent the offset denotes a position in --
                    // a one-past-the-end pointer into a constant
                    // allocation is the shape `ThinBox`'s ZST
                    // construction forms -- so the displacement
                    // is carried on the projection, as the
                    // projected arm below carries it.
                    match &*v {
                        Value::Tuple(_) | Value::Bytes(_) => Ok(RefTarget::value(
                            self.offset_value_target(*v, count, elem_size)?,
                        )),
                        _ => {
                            let projection = self.offset_projection(
                                frames,
                                None,
                                None,
                                Vec::new(),
                                disp,
                                elem_size,
                                Some(&v),
                                count,
                                false,
                            )?;
                            Ok(RefTarget::Value(v, projection))
                        }
                    }
                } else if let Some((&RefProj::Index(i), head)) = projection.split_last() {
                    // A byte-denominated offset (the pointer
                    // was cast to a byte pointer before the
                    // arithmetic, as `ThinBox`'s header
                    // back-step casts the data pointer to
                    // `*mut u8` and subtracts the header size)
                    // does not count in the elements the index
                    // denotes: translate it through the
                    // aggregate's field offsets when every
                    // field's byte size is known and the
                    // displacement lands on a field boundary.
                    if head.is_empty() {
                        if let Some(es) = elem_size {
                            if let Some(offsets) = aggregate_field_offsets(&v) {
                                if let Some(pos) = offsets.get(i).copied() {
                                    let landed = pos as i128
                                        + count * i128::from(es);
                                    if let Some(j) = offsets
                                        .iter()
                                        .position(|&o| o as i128 == landed)
                                    {
                                        return Ok(RefTarget::Value(
                                            v,
                                            vec![RefProj::Index(j)],
                                        ));
                                    }
                                }
                            }
                        }
                    }
                    // A value target projected to an element of
                    // the constant aggregate (the shape a
                    // first-element cast of a constant array or
                    // string reference produces): the offset is
                    // counted in elements of the type the index
                    // denotes, so it absorbs into the index
                    // directly, whatever the element size.
                    let idx = i128::try_from(i).map_err(|_| InterpError::UnsupportedRvalue {
                        kind: "pointer offset: element index out of range".to_string(),
                    })? + count;
                    if idx < 0 {
                        return Err(InterpError::UnsupportedRvalue {
                            kind: "pointer offset: negative offset of a constant array"
                                .to_string(),
                        });
                    }
                    let mut projection = head.to_vec();
                    projection.push(RefProj::Index(usize::try_from(idx).map_err(|_| {
                        InterpError::UnsupportedRvalue {
                            kind: "pointer offset: element index out of range".to_string(),
                        }
                    })?));
                    Ok(RefTarget::Value(v, projection))
                } else {
                    // A projected value target denotes a position
                    // inside the value: carry the displacement on
                    // the projection, as for statics and
                    // allocations, whose pointees are likewise not
                    // declared locals.
                    let projection = self.offset_projection(
                        frames,
                        None,
                        None,
                        projection,
                        disp,
                        elem_size,
                        Some(&v),
                        count,
                        false,
                    )?;
                    Ok(RefTarget::Value(v, projection))
                }
            }
            RefTarget::Null => {
                if wrapping {
                    let d = disp.ok_or_else(|| InterpError::UnsupportedRvalue {
                        kind: "pointer offset: element size is not known".to_string(),
                    })?;
                    Ok(RefTarget::Address(d as u64))
                } else {
                    Err(InterpError::UnsupportedRvalue {
                        kind: "pointer offset: offset of a null pointer".to_string(),
                    })
                }
            }
            RefTarget::Address(addr) => {
                // An address-only pointer has no allocation bounds
                // in this model for the checked form to check; the
                // address arithmetic is the observable behaviour.
                let d = disp.ok_or_else(|| InterpError::UnsupportedRvalue {
                    kind: "pointer offset: element size is not known".to_string(),
                })?;
                Ok(RefTarget::Address(addr.wrapping_add(d as u64)))
            }
        }
    }

    /// A constant aggregate advanced by a pointer offset. Such a
    /// target under an offset denotes its first element (the shape
    /// a cast of a constant array reference to an element pointer
    /// produces), so the offset denotes the count-th element,
    /// projected eagerly — constant aggregates are read-only in
    /// this model, so nothing observable is lost (the same
    /// reasoning as [`Self::ptr_field_value`]'s eager projection).
    /// A one-past-the-end position has no value to denote and
    /// stays an explicit error.
    fn offset_value_target(
        &self,
        value: Value,
        count: i128,
        elem_size: Option<u64>,
    ) -> InterpResult<Value> {
        match value {
            Value::Tuple(elems) => {
                let idx = usize::try_from(count).map_err(|_| InterpError::UnsupportedRvalue {
                    kind: "pointer offset: negative offset of a constant array".to_string(),
                })?;
                elems.get(idx).cloned().ok_or_else(|| InterpError::UnsupportedRvalue {
                    kind: "pointer offset: offset at or past the end of a constant array"
                        .to_string(),
                })
            }
            Value::Bytes(bytes) => {
                // A pointer into constant byte storage: only a
                // byte-sized element type denotes a single byte
                // of it.
                if elem_size != Some(1) {
                    return Err(InterpError::UnsupportedRvalue {
                        kind: "pointer offset: offset of a byte-string pointer with a                                non-byte element type"
                            .to_string(),
                    });
                }
                let idx = usize::try_from(count).map_err(|_| InterpError::UnsupportedRvalue {
                    kind: "pointer offset: negative offset of a byte string".to_string(),
                })?;
                bytes
                    .get(idx)
                    .map(|b| Value::Uint(u128::from(*b)))
                    .ok_or_else(|| InterpError::UnsupportedRvalue {
                        kind: "pointer offset: offset past the end of a byte string"
                            .to_string(),
                    })
            }
            other => Err(InterpError::UnsupportedRvalue {
                kind: format!(
                    "pointer offset: offset of a pointer to a constant {} value is not modelled",
                    other.kind_name()
                ),
            }),
        }
    }

    /// Advance a place/static target's projection by an offset:
    /// `disp` is the byte displacement (`None` when the offset's
    /// element size is unknown) and `count` the offset in
    /// elements, which denominates the advance for a zero-sized
    /// element type, whose byte displacement is always zero.
    /// When the target denotes an element
    /// of an array place (or the array place itself) and the total
    /// displacement lands exactly on an element boundary, it is
    /// absorbed into the index projection, so reads and writes
    /// through the result resolve structurally. The same absorption
    /// applies over any element aggregate the target's value
    /// denotes (see [`Self::absorb_displacement`]). One position
    /// is never absorbed: a position whose declared type is an
    /// ADT with a recorded field layout, where the displacement
    /// denotes a storage byte the byte view resolves (see
    /// [`Self::byte_view_covers`]). Otherwise the displacement
    /// is carried as a trailing [`RefProj::ByteOffset`] step:
    /// the position is faithfully denoted (and further offsets
    /// accumulate into it), and reading or writing through it
    /// resolves through the byte view where the target's layout
    /// is recorded, as an explicit error where it is not. A
    /// total displacement of zero never appears in the result:
    /// an offset that cancels a carried one restores the
    /// structural position.
    fn offset_projection(
        &self,
        frames: &[Frame],
        place: Option<(u64, usize)>,
        static_name: Option<&str>,
        mut projection: Vec<RefProj>,
        disp: Option<i128>,
        elem_size: Option<u64>,
        base: Option<&Value>,
        count: i128,
        storage_grows: bool,
    ) -> InterpResult<Vec<RefProj>> {
        // A displacement already carried on the target accumulates
        // into this one: the combination is what decides
        // absorption.
        let mut carried: i128 = 0;
        if let Some(RefProj::ByteOffset(prev)) = projection.last() {
            carried = *prev;
            projection.pop();
        }
        let total = match disp {
            Some(d) => carried.checked_add(d).ok_or_else(|| InterpError::UnsupportedRvalue {
                kind: "pointer offset: displacement overflows".to_string(),
            })?,
            None => {
                return Err(InterpError::UnsupportedRvalue {
                    kind: "pointer offset: element size is not known".to_string(),
                })
            }
        };
        if total == 0 && count == 0 {
            return Ok(projection);
        }
        let array_pos = match static_name {
            Some(name) => self.static_array_position(name, &projection),
            None => self.array_position(frames, place, &projection, base, elem_size),
        };
        let array_judged = array_pos.is_some();
        if let Some((prefix, cur, elem, tail)) = array_pos {
            if elem == 0 && total == 0 {
                // A zero-sized element type: the position is
                // denominated in elements alone (the byte
                // displacement is always zero), so the index
                // advances by the offset's element count, in
                // the same form the byte case below returns.
                let idx = cur + count;
                if idx >= 0 {
                    let mut out = projection[..prefix].to_vec();
                    if !(idx == 0 && projection.is_empty()) {
                        out.push(RefProj::Index(usize::try_from(idx).map_err(
                            |_| InterpError::UnsupportedRvalue {
                                kind: "pointer offset: element index out of range"
                                    .to_string(),
                            },
                        )?));
                    }
                    out.extend_from_slice(&tail);
                    return Ok(out);
                }
            }
            if elem > 0 {
                if let Some(byte_pos) = cur.checked_mul(elem).and_then(|b| b.checked_add(total)) {
                    if byte_pos >= 0 && byte_pos % elem == 0 {
                        let idx = byte_pos / elem;
                        // The result keeps the input's form: the
                        // steps that navigate to the sequence are
                        // preserved, and an index projection
                        // denotes the element itself, while an
                        // empty projection denotes the sequence
                        // place as a whole — different positions
                        // in this model, even at index 0.
                        let mut out = projection[..prefix].to_vec();
                        if !(idx == 0 && projection.is_empty()) {
                            out.push(RefProj::Index(usize::try_from(idx).map_err(
                                |_| InterpError::UnsupportedRvalue {
                                    kind: "pointer offset: element index out of range"
                                        .to_string(),
                                },
                            )?));
                        }
                        out.extend_from_slice(&tail);
                        return Ok(out);
                    }
                }
            }
        }
        // When the declared type cannot judge the position (the
        // target is not a sequence place), the value's own shape can:
        // a displacement over an element aggregate (a tuple, or an
        // ADT value whose fields are its elements, as a `Vec` value
        // is) in whole units of the offset's element size denotes
        // the element at that index, exactly as the array case
        // above. The declared-type verdict governs whenever it
        // applies, so a byte displacement into a typed sequence is
        // never reinterpreted this way.
        if !array_judged
            && !self.byte_view_covers(frames, place, static_name, &projection, base)
        {
            if let Some(base) = base {
                if let Some(absorbed) = self.absorb_displacement(
                    base,
                    &projection,
                    total,
                    elem_size,
                    count,
                    storage_grows,
                ) {
                    return Ok(absorbed);
                }
            }
        }
        if total == 0 {
            // Only an element-count offset over a zero-sized
            // element type reaches here (a zero byte total with
            // a nonzero count): no rule above could absorb it,
            // and the count has no byte denomination to carry,
            // so the structural position stands unchanged.
            return Ok(projection);
        }
        projection.push(RefProj::ByteOffset(total));
        Ok(projection)
    }

    /// The element index a pointer offset moves a sequence
    /// position to: the position `index` advanced by the
    /// offset. For a nonzero element size the offset is
    /// denominated in bytes (`total`, the accumulated byte
    /// displacement) and must land exactly on an element
    /// boundary. For a zero-sized element type the byte
    /// displacement is always zero and the positions are
    /// denominated in elements alone, so the index advances
    /// by the offset's element count (`off_count`) -- the
    /// same rule the `RefTarget::Value` arm of
    /// `offset_target` applies to constant aggregates
    /// ("whatever the element size"). A nonzero byte total
    /// over a zero-sized element type denominates no element
    /// position.
    fn offset_elem_index(
        index: i128,
        total: i128,
        size: i128,
        off_count: i128,
    ) -> Option<i128> {
        if size > 0 {
            let byte_pos = index.checked_mul(size)?.checked_add(total)?;
            if byte_pos < 0 || byte_pos % size != 0 {
                return None;
            }
            Some(byte_pos / size)
        } else {
            if total != 0 {
                return None;
            }
            let new_index = index.checked_add(off_count)?;
            if new_index < 0 {
                return None;
            }
            Some(new_index)
        }
    }

    /// Absorb a byte displacement into an element index over the
    /// element aggregate `base` that `projection` denotes a
    /// position in (see the call in
    /// [`Self::offset_projection`]). The projection must navigate
    /// structurally (fields, downcasts, sub-slices, and index
    /// steps into aggregate elements — a position inside a
    /// nested aggregate, such as one row of a matrix) to an
    /// element container — a tuple, a byte string, or an ADT
    /// value whose fields are its elements — and end either at
    /// the container itself (element 0) or at one index step
    /// into it. The displacement must move the position by a
    /// whole number of the offset's element size and land within
    /// the container (one past the end included: such a pointer
    /// may be formed, and reading through it stays an explicit
    /// out-of-bounds error). Anything else answers `None`, and
    /// the caller carries the displacement as before.
    ///
    /// One position shape is judged at a different level: when
    /// the steps after the last index step only enter a whole
    /// sub-value — field-0 steps into single-field aggregates
    /// and downcasts, the shape a transparent-wrapper entry
    /// (`PtrField`) leaves — the position denotes that sub-value
    /// as a whole, so the offset ranges over the repetitions of
    /// the sub-value: the elements of the container the last
    /// index step ranged over, whose stride is the entered
    /// value's size exactly because every entry step is
    /// size-preserving. The index step is adjusted and the
    /// entry steps are kept. Judging such a position at the
    /// entered value instead would index the offset into the
    /// value's own fields — a different position entirely (the
    /// `MaybeUninit` buffer walk behind array clone).
    fn absorb_displacement(
        &self,
        base: &Value,
        projection: &[RefProj],
        total: i128,
        elem_size: Option<u64>,
        off_count: i128,
        storage_grows: bool,
    ) -> Option<Vec<RefProj>> {
        let size = i128::from(elem_size?);
        // Navigate to the container the position sits in. A
        // sub-slice step descends into the sub-aggregate it
        // denotes (materialised on the side; the walk only reads
        // shapes and counts from it), so a position inside a
        // sub-slice of an element container — the back half of a
        // `split_at_mut` on a range slice, say — absorbs exactly
        // like a position in the container itself. An index step
        // that is not the last step descends into the element it
        // selects (which must be an aggregate the remaining
        // steps can navigate), so a position inside a nested
        // aggregate — an element of one row of a matrix, a field
        // of one element of an array — is judged at the
        // innermost container it sits in, the one the offset's
        // element size denominates.
        let mut temps: Vec<Box<Value>> = Vec::new();
        let mut cur: &Value = base;
        let mut prefix = projection.len();
        let mut index: i128 = 0;
        // The container the pending index ranges over, once an
        // index step has set the position (an owned snapshot: the
        // walk may materialise further sub-values into `temps`
        // after the step); `None` while the walk denotes the
        // container it has navigated to itself.
        let mut pending_container: Option<Value> = None;
        // The last index step the walk crossed — its position in
        // the projection, the container it ranges over and the
        // index it selects — and whether every step since then
        // only enters a whole sub-value (a field-0 step into a
        // single-field aggregate, or a downcast). A position of
        // that shape denotes the entered sub-value as a whole,
        // and an offset over it ranges over the container of
        // that index step, not over the entered value (see the
        // doc comment above).
        let mut index_level: Option<(usize, Value, i128)> = None;
        let mut trailing_entry = true;
        for (j, step) in projection.iter().enumerate() {
            match step {
                RefProj::Field(idx) => {
                    // A field step after an index step keeps the
                    // whole-value-entry shape only when it enters
                    // a single-field aggregate at field 0 (the
                    // transparent wrappers): any other field is a
                    // genuine sub-position whose stride the offset
                    // cannot denominate.
                    let single_field = match cur {
                        Value::Tuple(items) => items.len() == 1,
                        Value::Enum { fields, .. } => fields.len() == 1,
                        Value::Coroutine { fields, .. } => fields.len() == 1,
                        Value::Closure { env, .. } => env.len() == 1,
                        _ => false,
                    };
                    if *idx != 0 || !single_field {
                        trailing_entry = false;
                    }
                    // The judgment above is made at the step's
                    // own level: a step that retries through a
                    // one-field wrapper below is out of range at
                    // that level, so its index is non-zero and
                    // the judgment already answers for the
                    // retried levels too.
                    // The transparent-wrapper descent of
                    // `project_value`: a field step past the end
                    // of a one-field wrapper the storage acquired
                    // through a wrapper pointer (a `MaybeUninit`
                    // pointee) retries on the wrapper's only
                    // field — the projection itself carries no
                    // wrapper step, so the prefix accounting is
                    // unaffected.
                    loop {
                        match cur {
                            Value::Tuple(items) => {
                                cur = items.get(*idx)?;
                                break;
                            }
                            Value::Enum { fields, .. } => {
                                if let Some(v) = fields.get(*idx) {
                                    cur = v;
                                    break;
                                }
                                if fields.len() == 1
                                    && wrapper_payload(&fields[0])
                                {
                                    cur = &fields[0];
                                    continue;
                                }
                                return None;
                            }
                            Value::Coroutine { fields, .. } => {
                                cur = fields.get(*idx)?;
                                break;
                            }
                            Value::Closure { env, .. } => {
                                cur = env.get(*idx)?;
                                break;
                            }
                            _ => return None,
                        }
                    }
                }
                RefProj::Downcast(_) => {
                    if !matches!(cur, Value::Enum { .. } | Value::Coroutine { .. }) {
                        return None;
                    }
                }
                RefProj::Subslice { offset, len } => {
                    trailing_entry = false;
                    let sub = subslice_value(cur, *offset, *len)?;
                    temps.push(Box::new(sub));
                    cur = temps.last().expect("just pushed");
                }
                RefProj::Index(idx) => {
                    index_level =
                        Some((j, cur.clone(), i128::try_from(*idx).ok()?));
                    trailing_entry = true;
                    if j + 1 == projection.len() {
                        prefix = j;
                        index = i128::try_from(*idx).ok()?;
                        pending_container = Some(cur.clone());
                    } else {
                        // Descend into the selected element: the
                        // remaining steps navigate inside it, and
                        // a later index step sets the position
                        // afresh at that depth.
                        cur = match cur {
                            Value::Tuple(items) => items.get(*idx)?,
                            Value::Enum { fields, .. } => fields.get(*idx)?,
                            Value::SparseVec { elems, .. } => elems.get(*idx)?,
                            Value::Coroutine { fields, .. } => fields.get(*idx)?,
                            Value::Closure { env, .. } => env.get(*idx)?,
                            _ => return None,
                        };
                        prefix = projection.len();
                        index = 0;
                        pending_container = None;
                    }
                }
                _ => return None,
            }
        }
        // A position that ends in whole-value entry steps after
        // its last index step denotes the entered sub-value as a
        // whole: the offset ranges over the repetitions of that
        // sub-value in the container the index step ranged over
        // (the entry steps are size-preserving, so the offset's
        // element size is the stride there). Adjust that index
        // and keep the entry steps. A displacement that does not
        // land on a repetition boundary, or lands outside the
        // container, is not absorbed at this level — and is not
        // judged at the entered value either, whose fields the
        // offset does not denominate.
        if let Some((pos, snap_container, snap_index)) = &index_level {
            let trailing = &projection[pos + 1..];
            if !trailing.is_empty() && trailing_entry {
                let count: i128 = match snap_container {
                    Value::Tuple(items) => i128::try_from(items.len()).ok()?,
                    Value::Enum { fields, .. } => {
                        i128::try_from(fields.len()).ok()?
                    }
                    Value::SparseVec { elems, cap, .. } => {
                        i128::try_from((*cap).max(elems.len() as u128)).ok()?
                    }
                    Value::Bytes(b) => {
                        if size != 1 {
                            return None;
                        }
                        i128::try_from(b.len()).ok()?
                    }
                    Value::Str(text) => {
                        if size != 1 {
                            return None;
                        }
                        i128::try_from(text.len()).ok()?
                    }
                    _ => return None,
                };
                let new_index =
                    Self::offset_elem_index(*snap_index, total, size, off_count)?;
                if new_index > count
                    && !(storage_grows
                        && matches!(snap_container, Value::Tuple(_))
                        && new_index
                            <= crate::builtins::MAX_VEC_SET_LEN_ELEMENTS as i128)
                {
                    return None;
                }
                let new_index = usize::try_from(new_index).ok()?;
                let mut out: Vec<RefProj> = projection[..*pos].to_vec();
                out.push(RefProj::Index(new_index));
                out.extend_from_slice(trailing);
                return Some(out);
            }
        }
        // The container's element count, unwrapping the shapes a
        // pointer position can sit in: a fat reference denotes its
        // base's elements with the carried length, and a
        // single-field ADT whose only field is an element
        // aggregate is indexed through that field (the
        // `#[repr(simd)]` pun `project_value` also honours).
        let mut container: &Value = pending_container.as_ref().unwrap_or(cur);
        loop {
            match container {
                Value::FatRef { base, .. } => container = base,
                Value::Enum { fields, .. }
                    if fields.len() == 1
                        && matches!(fields[0], Value::Tuple(_) | Value::Bytes(_))
                        && prefix < projection.len() =>
                {
                    container = &fields[0];
                }
                _ => break,
            }
        }
        let count: i128 = match container {
            Value::Tuple(items) => i128::try_from(items.len()).ok()?,
            Value::Enum { fields, .. } => i128::try_from(fields.len()).ok()?,
            // A vector value in sparse form is an element
            // container whose extent is its storage (see
            // `Value::SparseVec`): a displacement in whole
            // element units denotes the element at that index,
            // spare-capacity slots included.
            Value::SparseVec { elems, cap, .. } => {
                i128::try_from((*cap).max(elems.len() as u128)).ok()?
            }
            Value::Bytes(b) => {
                if size != 1 {
                    return None;
                }
                i128::try_from(b.len()).ok()?
            }
            Value::Str(text) => {
                if size != 1 {
                    return None;
                }
                i128::try_from(text.len()).ok()?
            }
            _ => return None,
        };
        let new_index = Self::offset_elem_index(index, total, size, off_count)?;
        // The stored length is the container's extent only
        // for storage that cannot grow: a sequence tuple in
        // allocation storage grows past its stored prefix
        // with uninitialized slots (the write navigation
        // grows it on access, bounded by the same
        // materialisation limit), so an offset denominated
        // in whole elements over such a container absorbs
        // up to that limit — the position it denotes is the
        // element the access itself will materialise.
        if new_index > count
            && !(storage_grows
                && matches!(container, Value::Tuple(_))
                && new_index <= crate::builtins::MAX_VEC_SET_LEN_ELEMENTS as i128)
        {
            return None;
        }
        let new_index = usize::try_from(new_index).ok()?;
        let mut out: Vec<RefProj> = projection[..prefix].to_vec();
        // Keep the input's form when the position stays at the
        // container base itself (see `offset_projection`).
        if prefix == projection.len() && new_index == 0 {
            return Some(out);
        }
        out.push(RefProj::Index(new_index));
        Some(out)
    }

    /// The sequence position a place target denotes, when it is
    /// directly determinable from the target local's declared
    /// type: the projection ranges over a sequence the local
    /// holds — the local itself (an array, or a sequence ADT
    /// whose value is its element sequence), or a sequence-typed
    /// field of it — and ends at the sequence base (element 0)
    /// or at a single index into it, possibly entered through
    /// whole-value entry steps (see [`split_sequence_position`]).
    /// Returns the number of leading steps that navigate to the
    /// sequence (a field path), the current element index, the
    /// element size, and the entry steps that follow the index
    /// step, which the caller's rebuilt projection keeps.
    fn array_position(
        &self,
        frames: &[Frame],
        place: Option<(u64, usize)>,
        projection: &[RefProj],
        base: Option<&Value>,
        elem_size: Option<u64>,
    ) -> Option<(usize, i128, i128, Vec<RefProj>)> {
        let (frame_id, local) = place?;
        let frame = &frames[self.frame_pos(frames, frame_id).ok()?];
        let body = self.program().functions.get(&frame.func)?;
        // Split the projection into a leading field path and
        // the position within the sequence it navigates to: an
        // optional index step and the whole-value entry steps
        // that may follow it.
        let mut path: Vec<usize> = Vec::new();
        let mut steps = projection;
        while let Some((RefProj::Field(f), rest)) = steps.split_first() {
            path.push(*f);
            steps = rest;
        }
        let (cur, tail) = split_sequence_position(steps)?;
        if !path.is_empty() {
            // A sequence held in a field: the position ends at
            // an element of it, or at its base — a bare field
            // projection denotes element 0, exactly as a bare
            // local position does below. The one denomination
            // that must not be read as an element position is
            // the whole-field one: a pointer to the sequence
            // value itself, whose element size is the field
            // type's own size. When the offset's element size
            // is that size (and not also the sequence's
            // element size — the single-element case, where
            // the two readings coincide), the position stays
            // unjudged here and the displacement is carried.
            // A field type whose size is not derivable (a
            // debug shape can leave an array length
            // unresolved) gives no positive evidence of the
            // whole-field denomination, so the base reading
            // stands there too.
            // The element size is the one conversion recorded
            // for the field path, and the value held at the
            // path must actually be the element container, as
            // for the local case below.
            let elem = body
                .field_elem_sizes
                .get(local)
                .and_then(|sizes| sizes.get(&path))
                .copied()?;
            let cur = match cur {
                Some(c) => c,
                None if !tail.is_empty() => 0,
                None => {
                    let root_ty = body.local_tys.get(local)?;
                    let mut cur_ty = WalkTy::Ty(root_ty.clone());
                    for &f in &path {
                        cur_ty = self.field_walk_ty(&cur_ty, 0, f)?;
                    }
                    // The field type's size, where it is
                    // derivable: a debug shape can leave an
                    // array's length unresolved (`[T; _]`), in
                    // which case the whole-field denomination
                    // cannot be positively identified and the
                    // base reading stands.
                    let field_size = match &cur_ty {
                        WalkTy::Ty(t) => self.ty_size_full(t),
                        WalkTy::Tag(tag) => self
                            .layout_of(tag)
                            .and_then(|(size, _)| u64::try_from(size).ok()),
                    };
                    if let Some(field_size) = field_size {
                        if elem_size == Some(field_size) && field_size != elem {
                            return None;
                        }
                    }
                    0
                }
            };
            let field_steps: Vec<RefProj> =
                path.iter().map(|f| RefProj::Field(*f)).collect();
            let held = self.value_target_value(base?, &field_steps).ok()?;
            if !matches!(
                held,
                Value::Tuple(_) | Value::Enum { .. } | Value::SparseVec { .. }
            ) {
                return None;
            }
            return Some((path.len(), cur, i128::from(elem), tail.to_vec()));
        }
        let elem = match body.local_tys.get(local)? {
            Ty::Array(elem, _) => self.ty_size(elem)?,
            Ty::Adt(_) => {
                // A sequence ADT (a `Vec` above all): the value
                // the local holds is its element sequence, so a
                // position over it is denominated in the element
                // size conversion recorded for the local (see
                // `ir::Body::local_elem_sizes`; the IR type drops
                // the generic arguments the size is computed
                // from). The held value must actually be the
                // element container (a tuple of elements, an ADT
                // value whose fields are its elements, a sparse
                // vector): a sequence ADT in any other held shape
                // (a reference into it, an uninitialised slot)
                // denotes no element position.
                if !matches!(
                    base,
                    Some(
                        Value::Tuple(_)
                            | Value::Enum { .. }
                            | Value::SparseVec { .. }
                    )
                ) {
                    return None;
                }
                body.local_elem_sizes.get(local).copied().flatten()?
            }
            _ => return None,
        };
        Some((0, cur.unwrap_or(0), i128::from(elem), tail.to_vec()))
    }

    /// The array position a projection over the static `name`
    /// denotes, from the static's declared type as the driver
    /// recorded it in the program: the current element index and
    /// the element size in bytes, in the same form
    /// [`Self::array_position`] answers for a local. `None` when
    /// the static has no recorded type (a foreign item) or its
    /// declared type is not an array.
    fn static_array_position(
        &self,
        name: &str,
        projection: &[RefProj],
    ) -> Option<(usize, i128, i128, Vec<RefProj>)> {
        let elem = match self.program().static_tys.get(name)? {
            Ty::Array(elem, _) => self.ty_size(elem)?,
            _ => return None,
        };
        let (cur, tail) = split_sequence_position(projection)?;
        Some((0, cur.unwrap_or(0), i128::from(elem), tail.to_vec()))
    }

    /// The value a `Resolved::Value` target denotes: its base
    /// projected by the target's pending projection (the base
    /// itself when the projection is empty).
    pub(crate) fn value_target_value(
        &self,
        base: &Value,
        projection: &[RefProj],
    ) -> InterpResult<Value> {
        if projection.is_empty() {
            Ok(base.clone())
        } else {
            self.project_value_ref_t(
                base,
                projection,
                "<constant>",
                "constant aggregate",
                None,
            )
        }
    }
}

/// Split the steps that range over one sequence into the
/// element position they denote: an optional index step (the
/// current element; absent denotes the sequence base) and the
/// whole-value entry steps behind it — field-0 and downcast
/// steps, the shape a pointer cast into an element through a
/// transparent wrapper or a union payload leaves
/// (`[Index(k), Field(0)]` denotes element `k` entered through
/// the wrapper, and entry steps with no index step enter the
/// base element). The position is denominated in the
/// sequence's own elements, whatever type the pointer was cast
/// to: a cast changes the units a later offset counts in, not
/// the slots the stored sequence is laid out in, so an offset
/// over such a position must be priced by the sequence's
/// recorded element size and the entry steps kept, exactly as
/// the sequence accesses compose it (see
/// `Interpreter::seq_element_target`). Any other step shape
/// denotes no plain sequence position.
fn split_sequence_position(
    steps: &[RefProj],
) -> Option<(Option<i128>, &[RefProj])> {
    let entry = |s: &RefProj| matches!(s, RefProj::Field(0) | RefProj::Downcast(_));
    match steps {
        [RefProj::Index(i), rest @ ..] if rest.iter().all(entry) => {
            Some((Some(i128::try_from(*i).ok()?), rest))
        }
        rest if rest.iter().all(entry) => Some((None, rest)),
        _ => None,
    }
}

#[cfg(test)]
mod panic_report_tests {
    use super::*;

    fn location_stamp(file: &str, line: u128, col: u128) -> Value {
        Value::Ref(RefTarget::value(Value::Enum {
            variant: 0,
            fields: vec![
                Value::Str(file.to_string()),
                Value::Uint(line),
                Value::Uint(col),
            ],
        }))
    }

    fn report_frame(func: &str) -> Frame {
        Frame {
            id: 1,
            func: func.to_string(),
            locals: Vec::new(),
            stack_bytes: 0,
            block: 0,
            stmt: 0,
            return_dest: None,
            return_block: None,
            unwind_payload: None,
            in_cleanup: false,
            call_once_env_drop: None,
            incoming_asm: None,
        }
    }

    /// Note one `&str`-payload panic at `guest.rs:3:7` on the
    /// interpreter, as its entry point would.
    fn note_str_panic(interp: &Interpreter) {
        interp.note_panic_stamp(location_stamp("guest.rs", 3, 7));
        interp.note_panic_origin(PanicOrigin::PanicFn);
        interp.note_panic_no_backtrace(false);
    }

    /// The default report for an uncaught panic: the head with
    /// the thread's name and id and the panic's location, the
    /// message, and the backtrace note (the style is Off).
    #[test]
    fn default_report_prints_head_message_and_note() {
        let program = Program::new();
        let mut interp = Interpreter::new(&program);
        // `BacktraceStyle::Off` (variant 2 in std's order).
        interp.set_backtrace_style(Value::Enum {
            variant: 2,
            fields: Vec::new(),
        });
        note_str_panic(&interp);
        let mut frames = Vec::new();
        interp.fire_panic_hook(&mut frames, 0, "boom", &None);
        assert_eq!(
            interp.error_output(),
            "\nthread 'main' (1) panicked at guest.rs:3:7:\nboom\n\
             note: run with `RUST_BACKTRACE=1` environment variable \
             to display a backtrace\n"
        );
    }

    /// The note is once per process (std's FIRST_PANIC): a
    /// second Off-style report in the same run omits it.
    #[test]
    fn default_report_note_prints_once() {
        let program = Program::new();
        let mut interp = Interpreter::new(&program);
        interp.set_backtrace_style(Value::Enum {
            variant: 2,
            fields: Vec::new(),
        });
        let mut frames = Vec::new();
        note_str_panic(&interp);
        interp.fire_panic_hook(&mut frames, 0, "boom", &None);
        interp.note_panic_consumed();
        note_str_panic(&interp);
        interp.fire_panic_hook(&mut frames, 0, "boom", &None);
        assert_eq!(
            interp.error_output(),
            "\nthread 'main' (1) panicked at guest.rs:3:7:\nboom\n\
             note: run with `RUST_BACKTRACE=1` environment variable \
             to display a backtrace\n\
             \nthread 'main' (1) panicked at guest.rs:3:7:\nboom\n"
        );
    }

    /// A second panic while the first is still in flight (a
    /// destructor panicking during cleanup) forces the full
    /// backtrace shape even when the configured style is Short.
    #[test]
    fn second_in_flight_panic_forces_full_backtrace() {
        let program = Program::new();
        let mut interp = Interpreter::new(&program);
        // `BacktraceStyle::Short` (variant 0 in std's order).
        interp.set_backtrace_style(Value::Enum {
            variant: 0,
            fields: Vec::new(),
        });
        let mut frames = vec![report_frame("guest::main"), report_frame("guest::main")];
        note_str_panic(&interp);
        interp.fire_panic_hook(&mut frames, 0, "boom", &None);
        let first = interp.error_output().to_string();
        assert!(first.contains("   0: __rustc::rust_begin_unwind\n"), "{first}");
        assert!(first.contains("note: Some details are omitted"), "{first}");
        // No consumption: the first panic is still in flight
        // when the second fires (at the same stack depth, as a
        // cleanup panic fires below its raise point).
        note_str_panic(&interp);
        interp.fire_panic_hook(&mut frames, 2, "boom again", &None);
        let second = &interp.error_output()[first.len()..];
        assert!(
            second.contains("   0:     0x555555554000 - __rustc::rust_begin_unwind\n"),
            "{second}"
        );
        assert!(!second.contains("note: Some details are omitted"), "{second}");
    }

    /// A panic raised while the hook runs prints the
    /// processing-panic sequence — the second panic's location
    /// and (for a static `&str` payload) its message, then the
    /// abort line — and marks the run for abort.
    #[test]
    fn panic_inside_hook_prints_processing_sequence() {
        let program = Program::new();
        let mut interp = Interpreter::new(&program);
        interp.set_in_panic_hook(true);
        note_str_panic(&interp);
        let payload = Some(PanicPayload {
            ty: "&str".to_string(),
            value: Box::new(Value::Str("hook boom".to_string())),
            type_id: None,
        });
        let mut frames = Vec::new();
        interp.fire_panic_hook(&mut frames, 0, "hook boom", &payload);
        assert_eq!(
            interp.error_output(),
            "panicked at guest.rs:3:7:\nhook boom\n\
             thread panicked while processing panic. aborting.\n"
        );
        assert!(interp.hook_abort.get());
    }

    /// While an enclosing model produces the reports itself
    /// (the libtest harness model), the general printing is
    /// suppressed.
    #[test]
    fn suppressed_report_prints_nothing() {
        let program = Program::new();
        let mut interp = Interpreter::new(&program);
        interp.set_panic_report_suppressed(true);
        note_str_panic(&interp);
        let mut frames = Vec::new();
        interp.fire_panic_hook(&mut frames, 0, "boom", &None);
        assert_eq!(interp.error_output(), "");
    }

    /// The short shape follows std's printer across a
    /// short-backtrace region: the marker frames are not
    /// printed, the frames suspended between them are
    /// accounted by one omitted line, and the printed frames
    /// keep sequential indices. No shim frame is appended:
    /// the entry frame is not live in this rendering (no
    /// `run_with_report` entry is on the stack), as for a
    /// report rendered after `main` returned.
    #[test]
    fn short_backtrace_elides_the_region_between_the_markers() {
        let program = Program::new();
        let mut interp = Interpreter::new(&program);
        // `BacktraceStyle::Short` (variant 0 in std's order).
        interp.set_backtrace_style(Value::Enum {
            variant: 0,
            fields: Vec::new(),
        });
        note_str_panic(&interp);
        let mut frames = vec![
            report_frame("guest::main"),
            report_frame("guest::first"),
            report_frame("guest::__rust_end_short_backtrace"),
            report_frame("guest::first::{closure#0}"),
            report_frame("guest::second"),
            report_frame("guest::third"),
            report_frame("guest::__rust_begin_short_backtrace"),
            report_frame("guest::third::{closure#0}"),
            report_frame("guest::fourth"),
        ];
        interp.fire_panic_hook(&mut frames, 0, "boom", &None);
        assert_eq!(
            interp.error_output(),
            "\nthread 'main' (1) panicked at guest.rs:3:7:\nboom\n\
             stack backtrace:\n   0: __rustc::rust_begin_unwind\n   1: core::panicking::panic_fmt\n   2: core::panicking::panic\n   3: guest::fourth\n   4: guest::third::{closure#0}\n      [... omitted 3 frames ...]\n   5: guest::first\n   6: guest::main\n\
             note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.\n"
        );
    }

    /// The omitted line is singular for a one-frame region.
    #[test]
    fn short_backtrace_omitted_line_is_singular_for_one_frame() {
        let program = Program::new();
        let mut interp = Interpreter::new(&program);
        interp.set_backtrace_style(Value::Enum {
            variant: 0,
            fields: Vec::new(),
        });
        note_str_panic(&interp);
        let mut frames = vec![
            report_frame("guest::main"),
            report_frame("guest::outer"),
            report_frame("guest::__rust_end_short_backtrace"),
            report_frame("guest::hidden"),
            report_frame("guest::__rust_begin_short_backtrace"),
            report_frame("guest::visible"),
        ];
        interp.fire_panic_hook(&mut frames, 0, "boom", &None);
        assert_eq!(
            interp.error_output(),
            "\nthread 'main' (1) panicked at guest.rs:3:7:\nboom\n\
             stack backtrace:\n   0: __rustc::rust_begin_unwind\n   1: core::panicking::panic_fmt\n   2: core::panicking::panic\n   3: guest::visible\n      [... omitted 1 frame ...]\n   4: guest::outer\n   5: guest::main\n\
             note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.\n"
        );
    }
}


#[cfg(test)]
mod ptr_equality_tests {
    use super::*;

    fn probe_frame(func: &str, locals: Vec<Value>) -> Frame {
        Frame {
            id: 7,
            func: func.to_string(),
            locals,
            stack_bytes: 0,
            block: 0,
            stmt: 0,
            return_dest: None,
            return_block: None,
            unwind_payload: None,
            in_cleanup: false,
            call_once_env_drop: None,
            incoming_asm: None,
        }
    }

    fn place_price(
        interp: &Interpreter,
        frames: &[Frame],
        local: usize,
        projection: Vec<RefProj>,
        leaf: Option<&Ty>,
    ) -> Option<u128> {
        interp.arithmetic_price(
            &Value::Ref(RefTarget::Place {
                frame: 7,
                local,
                projection,
            }),
            leaf,
            Some(frames),
        )
    }

    fn slots_of(interp: &Interpreter, func: &str) -> Vec<Option<u64>> {
        interp.slot_layout(func).expect("layout").slots.clone()
    }

    /// Two temporaries whose storage ranges are disjoint
    /// share a stack slot (and so compare address-equal,
    /// as stack colouring makes them natively); a third
    /// whose range spans both does not.
    #[test]
    fn storage_ranges_share_slots() {
        let mut program = Program::new();
        let block = BasicBlockData::new(
            vec![
                Statement { kind: StatementKind::StorageLive(Local(3)) },
                Statement { kind: StatementKind::StorageLive(Local(1)) },
                Statement::assign(
                    Place::from_index(1),
                    Rvalue::Use(Operand::Constant(Constant::Uint(1))),
                ),
                Statement { kind: StatementKind::StorageDead(Local(1)) },
                Statement { kind: StatementKind::StorageLive(Local(2)) },
                Statement::assign(
                    Place::from_index(2),
                    Rvalue::Use(Operand::Constant(Constant::Uint(2))),
                ),
                Statement { kind: StatementKind::StorageDead(Local(2)) },
                Statement { kind: StatementKind::StorageDead(Local(3)) },
            ],
            TerminatorKind::Return,
        );
        program.add_function(Body::new(
            "probe::f",
            0,
            vec![
                Ty::Unit,
                Ty::Int(IntTy::I32),
                Ty::Int(IntTy::I32),
                Ty::Int(IntTy::I32),
            ],
            vec![block],
        ));
        let interp = Interpreter::new(&program);
        let slots = slots_of(&interp, "probe::f");
        assert_eq!(slots[1], slots[2], "disjoint ranges share a slot");
        assert_ne!(slots[1], slots[3], "a spanning range keeps its own slot");
    }

    /// A larger local created between two same-sized
    /// temporaries does not capture the first temporary's
    /// slot: slots are claimed in local (creation) order,
    /// so the first temporary owns a slot of its own size
    /// class and the second -- which interferes with the
    /// larger local but not with the first temporary --
    /// reuses it. (Largest-first best fit placed the first
    /// temporary inside the larger local's slot and
    /// scattered the second: the equal-pointers-unequal
    /// inline2 shape.)
    #[test]
    fn creation_order_keeps_size_class_slots() {
        let mut program = Program::new();
        let block = BasicBlockData::new(
            vec![
                Statement { kind: StatementKind::StorageLive(Local(1)) },
                Statement { kind: StatementKind::StorageLive(Local(2)) },
                Statement::assign(
                    Place::from_index(2),
                    Rvalue::Use(Operand::Constant(Constant::Uint(1))),
                ),
                Statement { kind: StatementKind::StorageDead(Local(2)) },
                Statement { kind: StatementKind::StorageLive(Local(3)) },
                Statement::assign(
                    Place::from_index(3),
                    Rvalue::Use(Operand::Constant(Constant::Uint(3))),
                ),
                Statement { kind: StatementKind::StorageLive(Local(4)) },
                Statement::assign(
                    Place::from_index(4),
                    Rvalue::Use(Operand::Constant(Constant::Uint(4))),
                ),
                Statement { kind: StatementKind::StorageDead(Local(4)) },
                Statement { kind: StatementKind::StorageDead(Local(3)) },
                Statement { kind: StatementKind::StorageDead(Local(1)) },
            ],
            TerminatorKind::Return,
        );
        program.add_function(Body::new(
            "probe::f",
            0,
            vec![
                Ty::Unit,
                Ty::Uint(UintTy::Usize),
                Ty::Int(IntTy::I32),
                Ty::Uint(UintTy::Usize),
                Ty::Int(IntTy::I32),
            ],
            vec![block],
        ));
        let interp = Interpreter::new(&program);
        let slots = slots_of(&interp, "probe::f");
        assert_eq!(
            slots[2], slots[4],
            "sequential same-sized temporaries share a slot"
        );
        assert_ne!(slots[2], slots[3], "the intervening larger local does not");
        assert_ne!(slots[2], slots[1], "the long-lived local does not");
    }

    /// Without storage markers (optimised MIR), def/use
    /// liveness gives the same sharing: two locals used in
    /// disjoint statement ranges share, one used across
    /// both does not.
    #[test]
    fn def_use_ranges_share_slots() {
        let mut program = Program::new();
        let block = BasicBlockData::new(
            vec![
                Statement::assign(
                    Place::from_index(3),
                    Rvalue::Use(Operand::Constant(Constant::Uint(7))),
                ),
                Statement::assign(
                    Place::from_index(1),
                    Rvalue::Use(Operand::Constant(Constant::Uint(1))),
                ),
                Statement::assign(
                    Place::from_index(0),
                    Rvalue::Use(Operand::Copy(Place::from_index(1))),
                ),
                Statement::assign(
                    Place::from_index(2),
                    Rvalue::Use(Operand::Constant(Constant::Uint(2))),
                ),
                Statement::assign(
                    Place::from_index(0),
                    Rvalue::Use(Operand::Copy(Place::from_index(2))),
                ),
                Statement::assign(
                    Place::from_index(0),
                    Rvalue::Use(Operand::Copy(Place::from_index(3))),
                ),
            ],
            TerminatorKind::Return,
        );
        program.add_function(Body::new(
            "probe::f",
            0,
            vec![
                Ty::Uint(UintTy::Usize),
                Ty::Int(IntTy::I32),
                Ty::Int(IntTy::I32),
                Ty::Int(IntTy::I32),
            ],
            vec![block],
        ));
        let interp = Interpreter::new(&program);
        let slots = slots_of(&interp, "probe::f");
        assert_eq!(slots[1], slots[2], "disjoint def/use ranges share");
        assert_ne!(slots[1], slots[3], "a spanning def/use range does not");
    }

    /// A zero-sized local's address is its alignment (the
    /// dangling address): 1 for a unit, 4 for an align-4
    /// zero-sized ADT.
    #[test]
    fn zero_sized_locals_price_at_their_alignment() {
        let mut program = Program::new();
        program
            .adt_layouts
            .insert("probe::Z".to_string(), (0, 4));
        program.add_function(Body::new(
            "probe::f",
            0,
            vec![Ty::Unit, Ty::Unit, Ty::Adt("probe::Z".to_string())],
            vec![],
        ));
        let interp = Interpreter::new(&program);
        let frames = vec![probe_frame(
            "probe::f",
            vec![Value::Unit, Value::Unit, Value::Enum { variant: 0, fields: vec![] }],
        )];
        interp.register_frame_base(&[], 7, "probe::f");
        assert_eq!(
            place_price(&interp, &frames, 1, vec![], Some(&Ty::Unit)),
            Some(1),
        );
        assert_eq!(
            place_price(
                &interp,
                &frames,
                2,
                vec![],
                Some(&Ty::Adt("probe::Z".to_string())),
            ),
            Some(4),
        );
    }

    /// A packed field's address is its root's address plus
    /// its layout offset, unaligned tail included.
    #[test]
    fn packed_field_price_is_root_plus_offset() {
        let mut program = Program::new();
        program
            .adt_layouts
            .insert("probe::P".to_string(), (9, 1));
        program.adt_field_layouts.insert(
            "probe::P".to_string(),
            vec![vec![(0, 1), (1, 8)]],
        );
        program.add_function(Body::new(
            "probe::f",
            0,
            vec![Ty::Unit, Ty::Adt("probe::P".to_string())],
            vec![],
        ));
        let interp = Interpreter::new(&program);
        let frames = vec![probe_frame(
            "probe::f",
            vec![
                Value::Unit,
                Value::Enum {
                    variant: 0,
                    fields: vec![Value::Uint(3), Value::Uint(4)],
                },
            ],
        )];
        interp.register_frame_base(&[], 7, "probe::f");
        let root = place_price(&interp, &frames, 1, vec![], None).expect("root");
        let tail = place_price(
            &interp,
            &frames,
            1,
            vec![RefProj::Field(1)],
            Some(&Ty::Uint(UintTy::U64)),
        )
        .expect("tail");
        assert_eq!(tail, root + 1);
    }

    /// Statics occupy the statics region in declaration
    /// order at their own alignments: a u64 static followed
    /// by a unit static puts the second exactly 8 past the
    /// first, and the first at the region base.
    #[test]
    fn statics_price_in_the_statics_region() {
        let mut program = Program::new();
        program
            .static_tys
            .insert("probe::A".to_string(), Ty::Uint(UintTy::U64));
        program
            .static_tys
            .insert("probe::B".to_string(), Ty::Unit);
        let interp = Interpreter::new(&program);
        let price = |name: &str| {
            interp
                .arithmetic_price(
                    &Value::Ref(RefTarget::Static {
                        name: name.to_string(),
                        projection: vec![],
                    }),
                    None,
                    None,
                )
                .expect("static price")
        };
        let a = price("probe::A");
        let b = price("probe::B");
        assert_eq!(a, 1u128 << 61);
        assert_eq!(b - a, 8);
    }

    /// Heap allocations price in creation order at their
    /// recorded alignments, and an element position is the
    /// base plus the element displacement.
    #[test]
    fn alloc_prices_follow_creation_order() {
        let program = Program::new();
        let interp = Interpreter::new(&program);
        let id1 = interp.alloc_insert(Value::Tuple(vec![]));
        interp.record_alloc_layout(id1, 16, 4);
        interp.record_alloc_ty(
            id1,
            Ty::Array(Box::new(Ty::Uint(UintTy::U32)), 4),
        );
        let id2 = interp.alloc_insert(Value::Tuple(vec![]));
        interp.record_alloc_layout(id2, 8, 8);
        let price = |id: u64, projection: Vec<RefProj>| {
            interp
                .arithmetic_price(
                    &Value::Ref(RefTarget::Alloc { id, projection }),
                    Some(&Ty::Uint(UintTy::U32)),
                    None,
                )
                .expect("alloc price")
        };
        let base1 = price(id1, vec![]);
        let base2 = price(id2, vec![]);
        assert_eq!(base1, 1u128 << 62);
        assert_eq!(base2 - base1, 16);
        assert_eq!(price(id1, vec![RefProj::Index(2)]) - base1, 8);
    }

    /// Fat-pointer order is (address, length): two slices
    /// of one buffer starting at the same position order
    /// by length, through the direct binary operation MIR
    /// lowers `<` to.
    #[test]
    fn fat_pointer_order_falls_through_to_length() {
        let program = Program::new();
        let interp = Interpreter::new(&program);
        let id = interp.alloc_insert(Value::Tuple(vec![]));
        interp.record_alloc_layout(id, 16, 4);
        interp.record_alloc_ty(
            id,
            Ty::Array(Box::new(Ty::Uint(UintTy::U32)), 4),
        );
        let base = Value::Ref(RefTarget::Alloc {
            id,
            projection: vec![],
        });
        let p0 = Value::FatRef { base: Box::new(base.clone()), len: 0 };
        let p1 = Value::FatRef { base: Box::new(base), len: 1 };
        let addr_of = |v: &Value| interp.pointer_address(&[], v, None);
        let lt = eval_binary(BinOp::Lt, &p0, &p1, None, false, &addr_of)
            .expect("lt");
        assert_eq!(lt.0, Value::Bool(true));
        let eq = eval_binary(BinOp::Eq, &p0, &p1, None, false, &addr_of)
            .expect("eq");
        assert_eq!(eq.0, Value::Bool(false));
        let cmp = eval_binary(BinOp::Cmp, &p0, &p1, None, false, &addr_of)
            .expect("cmp");
        assert_eq!(
            cmp.0,
            Value::Enum { variant: 0, fields: vec![] },
            "cmp is Less"
        );
    }

    /// Dyn-pointer order is (address, vtable word): two
    /// trait objects over one address with different
    /// vtables are unequal, and their order is the vtable
    /// words' order — the same words the pointers flatten
    /// to.
    #[test]
    fn dyn_pointer_order_uses_the_vtable_word() {
        let program = Program::new();
        let interp = Interpreter::new(&program);
        let id = interp.alloc_insert(Value::Tuple(vec![]));
        interp.record_alloc_layout(id, 16, 4);
        let base = Value::Ref(RefTarget::Alloc {
            id,
            projection: vec![],
        });
        let d1 = Value::DynRef {
            base: Box::new(base.clone()),
            concrete: "probe::C".to_string(),
            traits: vec!["probe::T1".to_string()],
            drop_glue: None,
            type_id: None,
        };
        let d2 = Value::DynRef {
            base: Box::new(base),
            concrete: "probe::C".to_string(),
            traits: vec!["probe::T2".to_string()],
            drop_glue: None,
            type_id: None,
        };
        let addr_of = |v: &Value| interp.pointer_address(&[], v, None);
        let eq = eval_binary(BinOp::Eq, &d1, &d2, None, false, &addr_of)
            .expect("eq");
        assert_eq!(eq.0, Value::Bool(false));
        let w1 = vtable_word("probe::C", &["probe::T1".to_string()]);
        let w2 = vtable_word("probe::C", &["probe::T2".to_string()]);
        assert_ne!(w1, w2);
        let cmp = eval_binary(BinOp::Cmp, &d1, &d2, None, false, &addr_of)
            .expect("cmp");
        let want = match w1.cmp(&w2) {
            std::cmp::Ordering::Less => 0,
            std::cmp::Ordering::Equal => 1,
            std::cmp::Ordering::Greater => 2,
        };
        assert_eq!(cmp.0, Value::Enum { variant: want, fields: vec![] });
    }

    /// A program with two `Vec<i32>` locals in one body,
    /// for the buffer-pricing tests: the `Vec` instance's
    /// layout is on record, so the root type denominates.
    fn vec_probe_program() -> Program {
        let mut program = Program::new();
        program
            .adt_layouts
            .insert("std::vec::Vec<i32>".to_string(), (24, 8));
        let block = BasicBlockData::new(vec![], TerminatorKind::Return);
        program.add_function(Body::new(
            "probe::f",
            0,
            vec![
                Ty::Unit,
                Ty::Adt("std::vec::Vec".to_string()),
                Ty::Adt("std::vec::Vec".to_string()),
            ],
            vec![block],
        ));
        program
    }

    fn sparse_vec(len: u128, cap: u128) -> Value {
        Value::SparseVec {
            elems: vec![Value::Int(0); len as usize],
            len,
            cap,
        }
    }

    /// Element positions in a `Vec` value held in a local
    /// price from the buffer's own base in the buffer
    /// region: exact element steps within the buffer, and
    /// two buffers' capacity ranges are disjoint, however
    /// close the holding locals' slots sit. The `Vec`
    /// value's own root still prices as the local's storage.
    #[test]
    fn vec_buffer_positions_price_from_their_own_base() {
        let program = vec_probe_program();
        let interp = Interpreter::new(&program);
        interp.register_frame_base(&[], 7, "probe::f");
        let frames = vec![probe_frame(
            "probe::f",
            vec![Value::Unit, sparse_vec(4, 4), sparse_vec(4, 8)],
        )];
        let a0 = place_price(&interp, &frames, 1, vec![RefProj::Index(0)], None)
            .expect("a0");
        let a3 = place_price(&interp, &frames, 1, vec![RefProj::Index(3)], None)
            .expect("a3");
        let b0 = place_price(&interp, &frames, 2, vec![RefProj::Index(0)], None)
            .expect("b0");
        assert!(a0 >= BUFFER_BASE, "buffer positions price in the buffer region");
        assert_eq!(a3 - a0, 12, "element steps are exact within a buffer");
        assert!(
            b0 >= a0 + 16 || a0 >= b0 + 32,
            "distinct buffers' capacity ranges are disjoint"
        );
        // The same buffer reached through a subslice step
        // prices from the same base.
        let a_sub = place_price(
            &interp,
            &frames,
            1,
            vec![RefProj::Subslice { offset: 0, len: 4 }],
            None,
        )
        .expect("a subslice base");
        assert_eq!(a_sub, a0, "subslice and index forms share the buffer base");
        // The Vec value itself (no element step) still
        // prices as the local's storage, in the stack region.
        let root = place_price(&interp, &frames, 1, vec![], None).expect("root");
        assert!(root < BUFFER_BASE, "the Vec place itself keeps slot pricing");
    }

    /// A zero-capacity `Vec` has no buffer: every element
    /// position — not just the first — prices from the
    /// element type's dangling address, displaced by the
    /// position's offset, so the empty buffer has the one
    /// price domain.
    #[test]
    fn empty_vec_positions_price_at_the_dangling_address() {
        let program = vec_probe_program();
        let interp = Interpreter::new(&program);
        interp.register_frame_base(&[], 7, "probe::f");
        let frames = vec![probe_frame(
            "probe::f",
            vec![Value::Unit, sparse_vec(0, 0), sparse_vec(0, 0)],
        )];
        let p0 = place_price(&interp, &frames, 1, vec![RefProj::Index(0)], None)
            .expect("p0");
        let p3 = place_price(&interp, &frames, 1, vec![RefProj::Index(3)], None)
            .expect("p3");
        assert_eq!(p0, 4, "an empty buffer's base is the element alignment");
        assert_eq!(p3, 4 + 12, "later positions displace from the same base");
    }
}
