use std::collections::HashMap;

use sparagmos_mir as mir;

use super::ty_helpers::{int_bits, uint_bits, MAX_ARRAY_LEAVES};
use super::types::MirFunctionInput;

// ---------------------------------------------------------------------------
// ADT layouts
// ---------------------------------------------------------------------------

/// One ADT instantiation whose layout the program determines: the
/// ADT definition and the generic arguments of one use. (A `Vec`
/// table with linear lookup, like the interpreter's field table —
/// `mir::Ty` has no `Hash`.)
#[derive(Clone, PartialEq)]
pub(super) struct AdtKey {
    def: mir::DefId,
    generic_args: Vec<mir::Ty>,
}

/// One field of one variant in an [`AdtLayout`]: its declared
/// type, its leaf offset from the value's first leaf, and its
/// leaf width.
#[derive(Clone, Debug)]
pub(super) struct AdtFieldSlot {
    pub(super) ty: mir::Ty,
    pub(super) offset: usize,
    pub(super) width: usize,
}

/// The slot layout of one ADT instantiation (see the module
/// docs): an optional tag leaf, then the payload area the
/// variants share.
#[derive(Clone, Debug)]
pub(super) struct AdtLayout {
    /// A tag leaf is present (the ADT has more than one variant).
    pub(super) tag: bool,
    /// Total leaves of a value of this ADT.
    pub(super) total: usize,
    /// Field slots per variant, in field order.
    pub(super) variants: Vec<Vec<AdtFieldSlot>>,
}

impl AdtLayout {
    pub(super) fn field(&self, variant: u32, field: u32) -> Option<&AdtFieldSlot> {
        self.variants
            .get(variant as usize)
            .and_then(|v| v.get(field as usize))
    }

    /// The layout of an ADT the scan says nothing about: the
    /// historical tag-only shape, one leaf holding the tag.
    pub(super) fn tag_only() -> Self {
        AdtLayout {
            tag: true,
            total: 1,
            variants: vec![Vec::new()],
        }
    }
}

/// Leaf bound for one ADT value. Every leaf is a physical stack
/// slot in the compiled frame, so an unbounded payload would let
/// a guest program demand an unbounded frame (the array bound
/// [`MAX_ARRAY_LEAVES`] is the same consideration).
pub(super) const MAX_ADT_LEAVES: usize = 64;

/// What the pre-lowering scan of all bodies recovers about ADT
/// shapes: declared field types from `Field` projections (which
/// carry the field's substituted declared type), variant field
/// counts from aggregate constructions, variant sightings, and
/// the ADT definitions built by union aggregates. This mirrors
/// the interpreter's recovery in `sparagmos-interp`'s conversion,
/// extended with the counts and sightings a slot layout needs.
#[derive(Default)]
pub(super) struct AdtScan {
    field_tys: Vec<(AdtKey, u32, u32, mir::Ty)>,
    agg_counts: Vec<(AdtKey, u32, usize)>,
    sightings: Vec<(AdtKey, u32)>,
    unions: Vec<AdtKey>,
}

impl AdtScan {
    fn record_field(&mut self, key: &AdtKey, variant: u32, field: u32, ty: mir::Ty) {
        // Repeated sightings of one slot must agree (they are the
        // same declared field); the first sighting wins.
        if self
            .field_tys
            .iter()
            .any(|(k, v, f, _)| k == key && *v == variant && *f == field)
        {
            return;
        }
        self.field_tys.push((key.clone(), variant, field, ty));
    }

    fn field_ty(&self, key: &AdtKey, variant: u32, field: u32) -> Option<&mir::Ty> {
        self.field_tys
            .iter()
            .find(|(k, v, f, _)| k == key && *v == variant && *f == field)
            .map(|(_, _, _, ty)| ty)
    }

    fn record_agg(&mut self, key: &AdtKey, variant: u32, count: usize) {
        if self
            .agg_counts
            .iter()
            .any(|(k, v, _)| k == key && *v == variant)
        {
            return;
        }
        self.agg_counts.push((key.clone(), variant, count));
    }

    fn agg_count(&self, key: &AdtKey, variant: u32) -> Option<usize> {
        self.agg_counts
            .iter()
            .find(|(k, v, _)| k == key && *v == variant)
            .map(|(_, _, c)| *c)
    }

    fn record_sighting(&mut self, key: &AdtKey, variant: u32) {
        if self.sightings.iter().any(|(k, v)| k == key && *v == variant) {
            return;
        }
        self.sightings.push((key.clone(), variant));
    }

    fn record_union(&mut self, key: &AdtKey) {
        if !self.unions.contains(key) {
            self.unions.push(key.clone());
        }
    }
}

/// Walk one place's projection chain, recording the declared
/// type of every ADT field stepped into and every variant
/// downcast to (the interpreter's `scan_place_fields`, narrowed
/// to ADT owners and extended with variant sightings).
pub(super) fn scan_place_fields(scan: &mut AdtScan, body: &mir::Body, place: &mir::Place) {
    let Some(decl) = body.local_decls.get(place.local.index()) else {
        return;
    };
    let mut ty = decl.ty.clone();
    let mut variant = 0u32;
    for elem in &place.projection {
        match elem {
            mir::PlaceElem::Field(idx, field_ty) => {
                if let mir::Ty::Adt(def, args) = &ty {
                    let key = AdtKey {
                        def: *def,
                        generic_args: args.clone(),
                    };
                    scan.record_field(&key, variant, *idx, field_ty.clone());
                }
                ty = field_ty.clone();
                variant = 0;
            }
            mir::PlaceElem::Downcast(_, v) => {
                if let mir::Ty::Adt(def, args) = &ty {
                    let key = AdtKey {
                        def: *def,
                        generic_args: args.clone(),
                    };
                    scan.record_sighting(&key, *v);
                }
                variant = *v;
            }
            mir::PlaceElem::Deref => match ty {
                mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => {
                    ty = *inner;
                    variant = 0;
                }
                _ => return,
            },
            mir::PlaceElem::Index(_) | mir::PlaceElem::ConstantIndex { .. } => match ty {
                mir::Ty::Array(elem, _) | mir::Ty::Slice(elem) => {
                    ty = *elem;
                    variant = 0;
                }
                _ => return,
            },
            mir::PlaceElem::Subslice { .. } => {}
            mir::PlaceElem::OpaqueCast(t) | mir::PlaceElem::Subtype(t) => {
                ty = t.clone();
                variant = 0;
            }
        }
    }
}

pub(super) fn scan_operand_fields(scan: &mut AdtScan, body: &mir::Body, op: &mir::Operand) {
    if let Some(place) = op.place() {
        scan_place_fields(scan, body, place);
    }
}

pub(super) fn scan_rvalue_fields(scan: &mut AdtScan, body: &mir::Body, rv: &mir::Rvalue) {
    use mir::Rvalue as R;
    match rv {
        R::Use(op)
        | R::Repeat(op, _)
        | R::Cast(_, op, _)
        | R::UnaryOp(_, op)
        | R::ShallowInitBox(op, _) => scan_operand_fields(scan, body, op),
        R::BinaryOp(_, pair) | R::CheckedBinaryOp(_, pair) => {
            scan_operand_fields(scan, body, &pair.0);
            scan_operand_fields(scan, body, &pair.1);
        }
        R::Ref(_, _, place)
        | R::Reborrow(_, _, place)
        | R::RawPtr(_, place)
        | R::Len(place)
        | R::Discriminant(place)
        | R::CopyForDeref(place) => scan_place_fields(scan, body, place),
        R::Aggregate(_, ops) => {
            for op in ops {
                scan_operand_fields(scan, body, op);
            }
        }
        R::ThreadLocalRef(_) | R::NullaryOp(..) => {}
    }
}

pub(super) fn scan_statement_fields(scan: &mut AdtScan, body: &mir::Body, stmt: &mir::Statement) {
    use mir::StatementKind as S;
    match &stmt.kind {
        S::Assign(pair) => {
            let (place, rv) = &**pair;
            scan_place_fields(scan, body, place);
            scan_rvalue_fields(scan, body, rv);
        }
        S::FakeRead(pair) => scan_place_fields(scan, body, &pair.1),
        S::SetDiscriminant {
            place,
            variant_index,
        } => {
            scan_place_fields(scan, body, place);
            if let Some(decl) = body.local_decls.get(place.local.index()) {
                if let mir::Ty::Adt(def, args) = &decl.ty {
                    let key = AdtKey {
                        def: *def,
                        generic_args: args.clone(),
                    };
                    scan.record_sighting(&key, *variant_index);
                }
            }
        }
        S::Retag(_, place) | S::PlaceMention(place) => scan_place_fields(scan, body, place),
        S::AscribeUserType(pair, _) => scan_place_fields(scan, body, &pair.0),
        S::Intrinsic(intr) => match &**intr {
            mir::NonDivergingIntrinsic::CopyNonOverlapping { src, dst, count } => {
                scan_operand_fields(scan, body, src);
                scan_operand_fields(scan, body, dst);
                scan_operand_fields(scan, body, count);
            }
            mir::NonDivergingIntrinsic::Assume(op) => scan_operand_fields(scan, body, op),
        },
        S::StorageLive(_)
        | S::StorageDead(_)
        | S::Coverage
        | S::ConstEvalCounter
        | S::Nop => {}
    }
}

pub(super) fn scan_terminator_fields(scan: &mut AdtScan, body: &mir::Body, kind: &mir::TerminatorKind) {
    use mir::TerminatorKind as T;
    match kind {
        T::SwitchInt { discr, .. } => scan_operand_fields(scan, body, discr),
        T::Call {
            func,
            args,
            destination,
            ..
        } => {
            scan_operand_fields(scan, body, func);
            for arg in args {
                scan_operand_fields(scan, body, arg);
            }
            scan_place_fields(scan, body, destination);
        }
        T::Assert { cond, .. } => scan_operand_fields(scan, body, cond),
        T::Drop { place, .. } => scan_place_fields(scan, body, place),
        T::Yield {
            value, resume_arg, ..
        } => {
            scan_operand_fields(scan, body, value);
            scan_place_fields(scan, body, resume_arg);
        }
        T::InlineAsm { operands, .. } => {
            for op in operands {
                if let Some(value) = &op.in_value {
                    scan_operand_fields(scan, body, value);
                }
            }
        }
        T::TailCall { func, args } => {
            scan_operand_fields(scan, body, func);
            for arg in args {
                scan_operand_fields(scan, body, arg);
            }
        }
        T::Goto { .. }
        | T::Resume
        | T::Terminate(_)
        | T::Return
        | T::Unreachable
        | T::CoroutineDrop
        | T::FalseEdge { .. }
        | T::FalseUnwind { .. } => {}
    }
}

pub(super) fn scan_body_fields(scan: &mut AdtScan, body: &mir::Body) {
    for bb in &body.basic_blocks {
        for stmt in &bb.statements {
            scan_statement_fields(scan, body, stmt);
        }
        if let Some(term) = &bb.terminator {
            scan_terminator_fields(scan, body, &term.kind);
        }
    }
}

/// The declared type of a place, walking its projections the way
/// [`scan_place_fields`] does. Used to type aggregate operands:
/// an operand filled into field `i` of an ADT aggregate carries
/// that field's declared type, so a construction site records
/// field types that no `Field` projection anywhere ever reveals
/// (a field that is only ever written, never read).
pub(super) fn place_decl_ty(body: &mir::Body, place: &mir::Place) -> Option<mir::Ty> {
    let mut ty = body.local_decls.get(place.local.index())?.ty.clone();
    for elem in &place.projection {
        ty = match elem {
            mir::PlaceElem::Field(_, t) => t.clone(),
            mir::PlaceElem::Deref => match ty {
                mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => *inner,
                _ => return None,
            },
            mir::PlaceElem::Index(_) | mir::PlaceElem::ConstantIndex { .. } => match ty {
                mir::Ty::Array(elem, _) | mir::Ty::Slice(elem) => *elem,
                _ => return None,
            },
            mir::PlaceElem::Downcast(_, _)
            | mir::PlaceElem::Subslice { .. } => ty,
            mir::PlaceElem::OpaqueCast(t) | mir::PlaceElem::Subtype(t) => t.clone(),
        };
    }
    Some(ty)
}

/// Record, for every ADT aggregate construction in `body`, the
/// variant's field count, the types of the fields its place
/// operands reveal, and the variant sighting itself; a union
/// aggregate (one with an active-field index) marks its ADT as a
/// union.
pub(super) fn scan_body_aggregates(scan: &mut AdtScan, body: &mir::Body) {
    for bb in &body.basic_blocks {
        for stmt in &bb.statements {
            if let mir::StatementKind::Assign(pair) = &stmt.kind {
                let (_, rv) = &**pair;
                if let mir::Rvalue::Aggregate(kind, ops) = rv {
                    if let mir::AggregateKind::Adt {
                        def,
                        variant_index,
                        generic_args,
                        field_index,
                        ..
                    } = &**kind
                    {
                        let key = AdtKey {
                            def: *def,
                            generic_args: generic_args.clone(),
                        };
                        if field_index.is_some() {
                            scan.record_union(&key);
                            continue;
                        }
                        scan.record_agg(&key, *variant_index, ops.len());
                        scan.record_sighting(&key, *variant_index);
                        for (i, op) in ops.iter().enumerate() {
                            if let Some(place) = op.place() {
                                if let Some(ty) = place_decl_ty(body, place) {
                                    scan.record_field(&key, *variant_index, i as u32, ty);
                                }
                            } else if let mir::Operand::Constant(c) = op {
                                if let Some(ty) = const_literal_ty(&c.literal) {
                                    scan.record_field(&key, *variant_index, i as u32, ty);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The JIT's shape for one `std::array::IntoIter<T, N>`
/// instantiation: the iterator value is the array's elements in
/// order followed by the two `alive` range words (`start`,
/// `end`) — the shape of the core type. The converted type of
/// the iterator carries only `T` (the const generic is not part
/// of it), so the element area is sized from the program's
/// `into_iter` call sites, whose self type is the array itself
/// (see [`scan_array_iters`]): `len` is the largest array length
/// sighted for the instantiation, and an iterator over a
/// shorter array leaves the tail of the element area unused
/// (its `end` word still holds its own length).
#[derive(Clone, Debug)]
pub(super) struct ArrayIterShape {
    /// The element type `T`.
    elem: mir::Ty,
    /// The layout's array length: the largest `N` sighted for
    /// this instantiation.
    len: usize,
    /// Leaves per element.
    pub(super) per_elem: usize,
}

impl ArrayIterShape {
    /// Leaves of the element area.
    pub(super) fn data(&self) -> usize {
        self.per_elem * self.len
    }

    /// Total leaves of the iterator value.
    pub(super) fn total(&self) -> usize {
        self.data() + 2
    }
}

/// Whether a definition name is one of the two spellings of the
/// array `IntoIterator::into_iter` entry point: the trait method
/// (the spelling converted call sites use) or the `[T; N]`
/// implementation itself.
pub(super) fn is_array_into_iter_fn(name: &str) -> bool {
    name == "std::iter::IntoIterator::into_iter"
        || (name.starts_with("std::array::iter::<impl std::iter::IntoIterator for [")
            && name.ends_with(">::into_iter"))
}

/// Whether a definition name is one of the two spellings of the
/// array iterator's `Iterator::next`: the trait method or the
/// `IntoIter` implementation itself.
pub(super) fn is_array_iter_next_fn(name: &str) -> bool {
    name == "std::iter::Iterator::next"
        || (name.starts_with("<std::array::IntoIter<")
            && name.ends_with("as std::iter::Iterator>::next"))
}

/// Sightings of `std::array::IntoIter` constructions: for every
/// call to an array `into_iter` entry point whose self type is
/// an array of known length and whose destination is a plain
/// `IntoIter` local, the instantiation key with the element type
/// and length the call fixes. The converted type does not carry
/// the length, so one key can be sighted with several lengths;
/// the recorded length is the largest (see [`ArrayIterShape`]).
pub(super) fn scan_array_iters(
    functions: &[MirFunctionInput],
    names: &HashMap<mir::DefId, String>,
) -> Vec<(AdtKey, Result<(mir::Ty, usize), String>)> {
    let intoiter_defs: Vec<mir::DefId> = names
        .iter()
        .filter(|(_, n)| n.as_str() == "std::array::IntoIter")
        .map(|(d, _)| *d)
        .collect();
    if intoiter_defs.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<(AdtKey, Result<(mir::Ty, usize), String>)> = Vec::new();
    for f in functions {
        let body = &f.body;
        for bb in &body.basic_blocks {
            let Some(term) = &bb.terminator else { continue };
            let mir::TerminatorKind::Call {
                func, destination, ..
            } = &term.kind
            else {
                continue;
            };
            let mir::Operand::Constant(c) = func else { continue };
            let mir::ConstKind::Ty(mir::Ty::FnDef(def, gargs)) = &c.literal else {
                continue;
            };
            let Some(name) = names.get(def) else { continue };
            if !is_array_into_iter_fn(name) {
                continue;
            }
            let Some(mir::Ty::Array(elem, mir::ArrayLen::Known(n))) = gargs.first() else {
                continue;
            };
            if !destination.projection.is_empty() {
                continue;
            }
            let Some(decl) = body.local_decls.get(destination.local.index()) else {
                continue;
            };
            let mir::Ty::Adt(ddef, dgargs) = &decl.ty else {
                continue;
            };
            if !intoiter_defs.contains(ddef) {
                continue;
            }
            let key = AdtKey {
                def: *ddef,
                generic_args: dgargs.clone(),
            };
            let sighting = ((**elem).clone(), *n as usize);
            match out.iter_mut().find(|(k, _)| *k == key) {
                None => out.push((key, Ok(sighting))),
                Some((_, Ok(prev))) => {
                    // The key fixes the element type; only the
                    // length can differ between sightings.
                    debug_assert_eq!(prev.0, sighting.0);
                    prev.1 = prev.1.max(sighting.1);
                }
                Some((_, Err(_))) => {}
            }
        }
    }
    out
}

/// The slot layout of one array-iterator instantiation under
/// its JIT shape (see [`ArrayIterShape`]): field 0 is the
/// element array, field 1 the `alive` range — the core type's
/// own fields, so a projection the program does reveal resolves
/// the way it does for any other ADT.
pub(super) fn array_iter_layout(
    shape: &ArrayIterShape,
    names: &HashMap<mir::DefId, String>,
) -> Result<AdtLayout, String> {
    let total = shape.total();
    if total > MAX_ADT_LEAVES {
        return Err(format!(
            "ADT value of {total} leaves exceeds the JIT layout bound"
        ));
    }
    let data = shape.data();
    let range_ty = match names
        .iter()
        .find(|(_, n)| n.as_str() == "std::ops::Range")
    {
        Some((def, _)) => {
            mir::Ty::Adt(*def, vec![mir::Ty::Uint(mir::UintTy::Usize)])
        }
        None => mir::Ty::Tuple(vec![
            mir::Ty::Uint(mir::UintTy::Usize),
            mir::Ty::Uint(mir::UintTy::Usize),
        ]),
    };
    Ok(AdtLayout {
        tag: false,
        total,
        variants: vec![vec![
            AdtFieldSlot {
                ty: mir::Ty::Array(
                    Box::new(shape.elem.clone()),
                    mir::ArrayLen::Known(shape.len as u64),
                ),
                offset: 0,
                width: data,
            },
            AdtFieldSlot {
                ty: range_ty,
                offset: data,
                width: 2,
            },
        ]],
    })
}

/// The slot layouts of a whole program's ADT instantiations,
/// computed once before lowering.
pub(super) struct AdtLayouts {
    entries: Vec<(AdtKey, Result<AdtLayout, String>)>,
    array_iters: Vec<(AdtKey, Result<ArrayIterShape, String>)>,
}

/// A width-exact type for a constant literal, when the literal
/// alone determines one: integer scalars by size, floats by
/// their type, tuples elementwise. Used to type aggregate
/// fields that are built directly from constants (no place
/// projection ever names their type). An integer literal's
/// signedness is not observable from the literal; the chosen
/// type only fixes the leaf width, which is all the layout
/// needs — reads go through projections, which carry their own
/// declared types.
pub(super) fn const_literal_ty(kind: &mir::ConstKind) -> Option<mir::Ty> {
    let mir::ConstKind::Value(v) = kind else {
        return None;
    };
    const_value_ty(v)
}

pub(super) fn const_value_ty(v: &mir::ConstValue) -> Option<mir::Ty> {
    use mir::IntTy;
    match v {
        mir::ConstValue::Scalar(mir::Scalar::Int { size, .. }) => Some(mir::Ty::Int(match size {
            1 => IntTy::I8,
            2 => IntTy::I16,
            4 => IntTy::I32,
            8 => IntTy::I64,
            16 => IntTy::I128,
            _ => return None,
        })),
        mir::ConstValue::Scalar(mir::Scalar::Float { ty, .. }) => Some(mir::Ty::Float(*ty)),
        mir::ConstValue::Bool(_) => Some(mir::Ty::Bool),
        mir::ConstValue::Char(_) => Some(mir::Ty::Char),
        mir::ConstValue::Tuple(fields) => {
            let tys: Option<Vec<mir::Ty>> = fields.iter().map(const_value_ty).collect();
            Some(mir::Ty::Tuple(tys?))
        }
        _ => None,
    }
}

impl AdtLayouts {
    pub(super) fn build(
        functions: &[MirFunctionInput],
        discriminants: &mir::AdtDiscriminants,
        names: &HashMap<mir::DefId, String>,
    ) -> AdtLayouts {
        let mut scan = AdtScan::default();
        for f in functions {
            scan_body_fields(&mut scan, &f.body);
            scan_body_aggregates(&mut scan, &f.body);
        }
        // The array by-value iterator's layouts are seeded
        // before the general computation: the scan of projections
        // and aggregates says nothing about the type (no converted
        // body ever touches its fields), and nested layout
        // computations must find the seeded entry the way they
        // find any other.
        let mut entries: Vec<(AdtKey, Result<AdtLayout, String>)> = Vec::new();
        let mut array_iters: Vec<(AdtKey, Result<ArrayIterShape, String>)> = Vec::new();
        for (key, sighting) in scan_array_iters(functions, names) {
            let layout = match &sighting {
                Ok((elem, len)) => {
                    let mut visiting = Vec::new();
                    match build_leaf_count(
                        elem,
                        &scan,
                        discriminants,
                        &mut entries,
                        &mut visiting,
                    ) {
                        Ok(per_elem) => {
                            let shape = ArrayIterShape {
                                elem: elem.clone(),
                                len: *len,
                                per_elem,
                            };
                            let layout = array_iter_layout(&shape, names);
                            array_iters.push((key.clone(), Ok(shape)));
                            layout
                        }
                        Err(reason) => {
                            array_iters.push((key.clone(), Err(reason.clone())));
                            Err(reason)
                        }
                    }
                }
                Err(reason) => {
                    array_iters.push((key.clone(), Err(reason.clone())));
                    Err(reason.clone())
                }
            };
            entries.push((key, layout));
        }
        let mut keys: Vec<AdtKey> = Vec::new();
        for (k, ..) in &scan.field_tys {
            if !keys.contains(k) {
                keys.push(k.clone());
            }
        }
        for (k, ..) in &scan.agg_counts {
            if !keys.contains(k) {
                keys.push(k.clone());
            }
        }
        for (k, _) in &scan.sightings {
            if !keys.contains(k) {
                keys.push(k.clone());
            }
        }
        for k in &scan.unions {
            if !keys.contains(k) {
                keys.push(k.clone());
            }
        }
        for key in &keys {
            let mut visiting = Vec::new();
            let layout = compute_layout(key, &scan, discriminants, &mut entries, &mut visiting);
            if !entries.iter().any(|(k, _)| k == key) {
                entries.push((key.clone(), layout));
            }
        }
        // The builtin-owned ADT layouts: the representation of
        // these types is defined by the builtin layer (see
        // `crate::builtins`), not recovered from the program — the
        // scan cannot see how a `fmt::Arguments` is built, because
        // building one *is* a builtin call. Every instantiation
        // the program mentions (in a local declaration, at any
        // nesting depth) gets the builtin definition, replacing
        // whatever the scan derived for it.
        let mut mentioned: Vec<AdtKey> = Vec::new();
        for f in functions {
            for decl in &f.body.local_decls {
                collect_adt_keys(&decl.ty, &mut mentioned);
            }
        }
        for key in mentioned {
            if let Some(layout) = builtin_adt_layout(&key, names) {
                if let Some(e) = entries.iter_mut().find(|(k, _)| *k == key) {
                    e.1 = Ok(layout);
                } else {
                    entries.push((key, Ok(layout)));
                }
            }
        }
        AdtLayouts {
            entries,
            array_iters,
        }
    }

    /// The array-iterator shape recorded for an instantiation,
    /// when the program constructs one.
    pub(super) fn array_iter(
        &self,
        def: mir::DefId,
        generic_args: &[mir::Ty],
    ) -> Option<Result<ArrayIterShape, String>> {
        let key = AdtKey {
            def,
            generic_args: generic_args.to_vec(),
        };
        self.array_iters
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, r)| r.clone())
    }

    /// The layout of one instantiation: the computed layout, the
    /// recorded reason it cannot be computed, or — for an ADT the
    /// scan says nothing about — the tag-only shape.
    pub(super) fn layout(&self, def: mir::DefId, generic_args: &[mir::Ty]) -> Result<AdtLayout, String> {
        let key = AdtKey {
            def,
            generic_args: generic_args.to_vec(),
        };
        match self.entries.iter().find(|(k, _)| *k == key) {
            Some((_, r)) => r.clone(),
            None => Ok(AdtLayout::tag_only()),
        }
    }

    /// Total leaves of a value of the ADT type `ty` (which must be
    /// an `Adt`), under the computed layouts.
    pub(super) fn adt_total(&self, ty: &mir::Ty) -> Result<usize, String> {
        let mir::Ty::Adt(def, args) = ty else {
            unreachable!("adt_total on a non-ADT type");
        };
        Ok(self.layout(*def, args)?.total)
    }
}

/// Collect every ADT instantiation a type mentions, at any
/// nesting depth (an `Argument` inside an array type, a range
/// inside a tuple, ...).
pub(super) fn collect_adt_keys(ty: &mir::Ty, out: &mut Vec<AdtKey>) {
    match ty {
        mir::Ty::Adt(def, args) => {
            let key = AdtKey {
                def: *def,
                generic_args: args.clone(),
            };
            if !out.contains(&key) {
                out.push(key);
            }
            for a in args {
                collect_adt_keys(a, out);
            }
        }
        mir::Ty::Tuple(ts) => {
            for t in ts {
                collect_adt_keys(t, out);
            }
        }
        mir::Ty::Array(e, _) | mir::Ty::Slice(e) => collect_adt_keys(e, out),
        mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => collect_adt_keys(inner, out),
        mir::Ty::Closure(_, args) | mir::Ty::Coroutine(_, args) => {
            for a in args {
                collect_adt_keys(a, out);
            }
        }
        _ => {}
    }
}

/// The layout the builtin layer defines for one of its own ADT
/// types, if `key` is one: a `fmt` value is the one-leaf store
/// handle; a range is its field layout (the shape the
/// interpreter's iterator state uses). `None` for any other ADT —
/// and for a range over a non-scalar element type, whose layout
/// stays the scan's business.
pub(super) fn builtin_adt_layout(
    key: &AdtKey,
    names: &HashMap<mir::DefId, String>,
) -> Option<AdtLayout> {
    let name = names.get(&key.def)?.as_str();
    let find_def = |want: &str| {
        names
            .iter()
            .find(|(_, n)| n.as_str() == want)
            .map(|(d, _)| *d)
    };
    match name {
        "std::string::String" => {
            // The heap model the builtin layer owns (see
            // `crate::builtins`): a `String` is its `Vec<u8>`
            // field, the (pointer, capacity, length) triple.
            // No converted body projects the field (construction
            // is the `From::from` builtin, the drop terminator
            // runs no code), so the scan cannot derive this
            // layout; the builtin layer defines it.
            let vec_def = find_def("std::vec::Vec")?;
            let global_def = find_def("std::alloc::Global")?;
            Some(AdtLayout {
                tag: false,
                total: 3,
                variants: vec![vec![AdtFieldSlot {
                    ty: mir::Ty::Adt(
                        vec_def,
                        vec![
                            mir::Ty::Uint(mir::UintTy::U8),
                            mir::Ty::Adt(global_def, Vec::new()),
                        ],
                    ),
                    offset: 0,
                    width: 3,
                }]],
            })
        }
        "std::vec::Vec" => {
            // A `Vec<T, A>` is its `RawVec<T, A>` field (the
            // (pointer, capacity) pair) followed by its length.
            // Same builtin-owned reasoning as `String`.
            let raw_def = find_def("alloc::raw_vec::RawVec")?;
            Some(AdtLayout {
                tag: false,
                total: 3,
                variants: vec![vec![
                    AdtFieldSlot {
                        ty: mir::Ty::Adt(raw_def, key.generic_args.clone()),
                        offset: 0,
                        width: 2,
                    },
                    AdtFieldSlot {
                        ty: mir::Ty::Uint(mir::UintTy::Usize),
                        offset: 2,
                        width: 1,
                    },
                ]],
            })
        }
        "std::mem::Alignment" => {
            // `Alignment` is its `AlignmentEnum` field alone.
            // The type flows through `Layout` values without any
            // converted body projecting it, so the scan cannot
            // derive this layout; the enum itself keeps the
            // tag-only shape (a unit-variant tag leaf), which is
            // its real one-word layout.
            let enum_def = find_def("std::mem::alignment::AlignmentEnum")?;
            Some(AdtLayout {
                tag: false,
                total: 1,
                variants: vec![vec![AdtFieldSlot {
                    ty: mir::Ty::Adt(enum_def, Vec::new()),
                    offset: 0,
                    width: 1,
                }]],
            })
        }
        "core::fmt::rt::Argument" | "std::fmt::Arguments" => Some(AdtLayout {
            tag: false,
            total: 1,
            variants: vec![Vec::new()],
        }),
        "std::option::Option" => {
            // The return value of the builtin `Iterator::next`:
            // the standard enum layout (tag leaf, then the
            // payload), for the scalar payloads `next` produces.
            // A payload of any other shape stays the scan's
            // business.
            let item = key.generic_args.first()?;
            if !matches!(
                item,
                mir::Ty::Int(_)
                    | mir::Ty::Uint(_)
                    | mir::Ty::Bool
                    | mir::Ty::Char
                    | mir::Ty::Float(_)
            ) {
                return None;
            }
            Some(AdtLayout {
                tag: true,
                total: 2,
                variants: vec![
                    Vec::new(),
                    vec![AdtFieldSlot {
                        ty: item.clone(),
                        offset: 1,
                        width: 1,
                    }],
                ],
            })
        }
        "std::ops::Range" | "std::ops::RangeInclusive" => {
            let idx = key.generic_args.first()?;
            if !matches!(
                idx,
                mir::Ty::Int(_) | mir::Ty::Uint(_) | mir::Ty::Char | mir::Ty::Bool
            ) {
                return None;
            }
            let mut fields = vec![
                AdtFieldSlot {
                    ty: idx.clone(),
                    offset: 0,
                    width: 1,
                },
                AdtFieldSlot {
                    ty: idx.clone(),
                    offset: 1,
                    width: 1,
                },
            ];
            if name == "std::ops::RangeInclusive" {
                fields.push(AdtFieldSlot {
                    ty: mir::Ty::Bool,
                    offset: 2,
                    width: 1,
                });
            }
            let total = fields.len();
            Some(AdtLayout {
                tag: false,
                total,
                variants: vec![fields],
            })
        }
        _ => None,
    }
}


/// Compute one instantiation's layout, memoising into `entries`
/// (nested ADT field types recurse into their own layouts;
/// `visiting` is the recursion stack, so a layout that mentions
/// itself without indirection is an explicit error).
pub(super) fn compute_layout(
    key: &AdtKey,
    scan: &AdtScan,
    discriminants: &mir::AdtDiscriminants,
    entries: &mut Vec<(AdtKey, Result<AdtLayout, String>)>,
    visiting: &mut Vec<AdtKey>,
) -> Result<AdtLayout, String> {
    if let Some((_, r)) = entries.iter().find(|(k, _)| k == key) {
        return r.clone();
    }
    if visiting.contains(key) {
        return Err("recursive ADT layout".to_string());
    }
    if scan.unions.contains(key) {
        let r = Err("union ADT".to_string());
        entries.push((key.clone(), r.clone()));
        return r;
    }
    visiting.push(key.clone());
    let result = compute_layout_inner(key, scan, discriminants, entries, visiting);
    visiting.pop();
    entries.push((key.clone(), result.clone()));
    result
}

pub(super) fn compute_layout_inner(
    key: &AdtKey,
    scan: &AdtScan,
    discriminants: &mir::AdtDiscriminants,
    entries: &mut Vec<(AdtKey, Result<AdtLayout, String>)>,
    visiting: &mut Vec<AdtKey>,
) -> Result<AdtLayout, String> {
    let mut variant_count = 1usize;
    if let Some(values) = discriminants.get(&key.def) {
        variant_count = variant_count.max(values.len());
    }
    for (k, v) in &scan.sightings {
        if k == key {
            variant_count = variant_count.max(*v as usize + 1);
        }
    }
    for (k, v, _) in &scan.agg_counts {
        if k == key {
            variant_count = variant_count.max(*v as usize + 1);
        }
    }
    let tag = variant_count > 1;
    let tag_leaves = tag as usize;
    let mut variants: Vec<Vec<AdtFieldSlot>> = Vec::with_capacity(variant_count);
    let mut payload = 0usize;
    for v in 0..variant_count as u32 {
        let mut field_count = scan.agg_count(key, v).unwrap_or(0);
        for (k, vv, f, _) in &scan.field_tys {
            if k == key && *vv == v {
                field_count = field_count.max(*f as usize + 1);
            }
        }
        let mut slots = Vec::with_capacity(field_count);
        let mut offset = tag_leaves;
        for f in 0..field_count as u32 {
            let Some(ty) = scan.field_ty(key, v, f) else {
                return Err(format!(
                    "ADT field type not recoverable (variant {v}, field {f})"
                ));
            };
            let width = build_leaf_count(ty, scan, discriminants, entries, visiting)?;
            slots.push(AdtFieldSlot {
                ty: ty.clone(),
                offset,
                width,
            });
            offset += width;
        }
        payload = payload.max(offset - tag_leaves);
        variants.push(slots);
    }
    let total = tag_leaves + payload;
    if total > MAX_ADT_LEAVES {
        return Err(format!(
            "ADT value of {total} leaves exceeds the JIT layout bound"
        ));
    }
    Ok(AdtLayout {
        tag,
        total,
        variants,
    })
}

/// Leaf count of a field type at layout-build time: the `flatten`
/// rules, with nested ADTs resolved through [`compute_layout`].
pub(super) fn build_leaf_count(
    ty: &mir::Ty,
    scan: &AdtScan,
    discriminants: &mir::AdtDiscriminants,
    entries: &mut Vec<(AdtKey, Result<AdtLayout, String>)>,
    visiting: &mut Vec<AdtKey>,
) -> Result<usize, String> {
    match ty {
        mir::Ty::Bool | mir::Ty::Char => Ok(1),
        mir::Ty::Int(it) => Ok(if int_bits(*it) <= 64 { 1 } else { 2 }),
        mir::Ty::Uint(ut) => Ok(if uint_bits(*ut) <= 64 { 1 } else { 2 }),
        mir::Ty::Tuple(ts) => {
            let mut n = 0;
            for t in ts {
                n += build_leaf_count(t, scan, discriminants, entries, visiting)?;
            }
            Ok(n)
        }
        mir::Ty::Never => Ok(0),
        mir::Ty::Ref(_, pointee, _) | mir::Ty::RawPtr(pointee, _) => match &**pointee {
            mir::Ty::Str | mir::Ty::Slice(_) => Ok(2),
            mir::Ty::Dynamic(..) => Err("fat reference / pointer type".to_string()),
            _ => Ok(1),
        },
        mir::Ty::Float(mir::FloatTy::F32) | mir::Ty::Float(mir::FloatTy::F64) => Ok(1),
        mir::Ty::Float(mir::FloatTy::F16) => Err("16-bit floating-point type".to_string()),
        mir::Ty::Float(mir::FloatTy::F128) => {
            Err("128-bit floating-point type".to_string())
        }
        mir::Ty::Array(elem, len) => {
            let mir::ArrayLen::Known(n) = len else {
                return Err("array type with unknown length".to_string());
            };
            let per = build_leaf_count(elem, scan, discriminants, entries, visiting)?;
            let total = (*n as usize).checked_mul(per).unwrap_or(usize::MAX);
            if total > MAX_ARRAY_LEAVES {
                return Err(
                    "array type with more scalar elements than the JIT frame holds"
                        .to_string(),
                );
            }
            Ok(total)
        }
        mir::Ty::Adt(def, args) => {
            let key = AdtKey {
                def: *def,
                generic_args: args.clone(),
            };
            // An ADT the scan says nothing about is the tag-only
            // shape (one leaf), as in `AdtLayouts::layout`.
            let known = scan.field_tys.iter().any(|(k, ..)| *k == key)
                || scan.agg_counts.iter().any(|(k, ..)| *k == key)
                || scan.sightings.iter().any(|(k, _)| *k == key)
                || scan.unions.contains(&key);
            if !known {
                return Ok(1);
            }
            Ok(compute_layout(&key, scan, discriminants, entries, visiting)?.total)
        }
        other => Err(format!("type {}", other.kind_name())),
    }
}
