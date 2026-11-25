//! ADT field-type recovery: program-wide scans that recover declared field types from projections.


use sparagmos_mir as mir;

use crate::ir;
use crate::value::{Value};



// ---------------------------------------------------------------------------

/// One ADT field slot whose declared type was recovered from a
/// `Field` projection somewhere in the program.
#[derive(PartialEq)]
pub(super) struct AdtFieldKey {
    pub(super) def: mir::DefId,
    /// The ADT instantiation's generic arguments (the projection's
    /// field type is already substituted for them).
    pub(super) generic_args: Vec<mir::Ty>,
    pub(super) variant: u32,
    pub(super) field: u32,
}

/// Declared field types of ADTs, recovered from `Field` projections.
///
/// An `Aggregate` rvalue in this MIR form does not carry the types of
/// the fields it fills — the operands are bare values — but every
/// access to a field through a place does: `PlaceElem::Field` holds
/// the field's (substituted) declared type. Scanning every place in
/// the program before conversion therefore yields, for each
/// (ADT instantiation, variant, field) slot that is ever projected,
/// the type its stored value must carry. Aggregate conversion types
/// its operands by this table, so a field's value is stored with its
/// declared tag from the start instead of being re-derived — wrongly,
/// for signed types — from the literal's bit pattern.
#[derive(Default)]
pub(super) struct AdtFieldTys {
    pub(super) entries: Vec<(AdtFieldKey, mir::Ty)>,
}

impl AdtFieldTys {
    fn record(&mut self, key: AdtFieldKey, ty: mir::Ty) {
        // Repeated projections of one slot must agree (they are the
        // same declared field); the first sighting wins so a
        // malformed program cannot flip a field's type mid-conversion.
        if self.entries.iter().any(|(k, _)| *k == key) {
            return;
        }
        self.entries.push((key, ty));
    }

    pub(super) fn get(
        &self,
        def: mir::DefId,
        generic_args: &[mir::Ty],
        variant: u32,
        field: u32,
    ) -> Option<&mir::Ty> {
        self.entries
            .iter()
            .find(|(k, _)| {
                k.def == def
                    && k.generic_args.as_slice() == generic_args
                    && k.variant == variant
                    && k.field == field
            })
            .map(|(_, ty)| ty)
    }
}

/// The cardinality of a type's value set, for the zero-sized
/// synthesis in `Converter::unique_value`.
pub(super) enum UniqueValue {
    /// Exactly one value; here it is.
    One(Value),
    /// No values (an uninhabited type).
    Uninhabited,
    /// More than one value, or not provable here.
    Many,
}

/// The most value nodes one [`Converter::zst_value`] synthesis may
/// materialise. The synthesis replicates an array's element value
/// once per element, so a per-level length cap alone does not bound
/// it: `[[(); 4096]; 4096]` passes a 4096 cap at each level and one
/// value of over sixteen million nodes. The budget counts nodes
/// across one whole synthesis (the same discipline as the driver's
/// `MAX_CONST_TOTAL_ELEMS` for constant decoding); a type whose one
/// value would exceed it is treated as having many values, which
/// only loses the unwritten-place synthesis for that type — the
/// read then takes the ordinary uninitialised path.
pub(super) const MAX_ZST_VALUE_NODES: u64 = 1 << 20;

/// Charge `nodes` against a `zst_value` synthesis budget; `false`
/// when the charge does not fit.
pub(super) fn spend_zst_nodes(budget: &mut u64, nodes: u64) -> bool {
    if nodes > *budget {
        return false;
    }
    *budget -= nodes;
    true
}

/// The number of nodes in a value tree (a node is one `Value`;
/// tuples, enums and closures add their children's counts).
pub(super) fn value_nodes(v: &Value) -> u64 {
    let children: &[Value] = match v {
        Value::Tuple(vals) => vals,
        Value::Enum { fields, .. } => fields,
        Value::Closure { env, .. } => env,
        _ => return 1,
    };
    children
        .iter()
        .fold(1u64, |acc, c| acc.saturating_add(value_nodes(c)))
}

/// Whether `ty` is a thin pointer type: a raw pointer or
/// reference whose pointee carries no metadata half (not a slice,
/// `str`, `dyn` or opaque alias pointee). A thin pointer's value
/// in this model is exactly its target, so transmutes between
/// thin pointer types — and between a thin pointer and a
/// transparent wrapper around one — reinterpret nothing.
/// Whether a scalar shape's leaf is an integer: the shape
/// flattens to an integer bit pattern (transparent wrapper
/// chains over an integer included).
pub(super) fn scalar_leaf_is_int(shape: &ir::ScalarShape) -> bool {
    match shape {
        ir::ScalarShape::Int { .. } => true,
        ir::ScalarShape::AdtField { inner, .. } => {
            scalar_leaf_is_int(inner)
        }
        _ => false,
    }
}

/// Whether a scalar shape is a pure bit pattern for the
/// bit-stream reinterpretation: integer, float, `bool` and
/// `char` leaves under arrays and one-payload wrappers.
/// Pointer-carrying leaves stay out — flattening one to its
/// synthetic address and constructing one back severs the
/// pointer from its pointee, which the aligned paths
/// (elementwise and whole-pattern) exist to avoid; niche
/// and multi-field shapes carry encodings a raw stream
/// does not reproduce.
pub(super) fn shape_is_bit_stream(shape: &ir::ScalarShape) -> bool {
    match shape {
        ir::ScalarShape::Int { .. }
        | ir::ScalarShape::Float { .. }
        | ir::ScalarShape::Bool
        | ir::ScalarShape::Char => true,
        ir::ScalarShape::Array(elem, _) => shape_is_bit_stream(elem),
        ir::ScalarShape::AdtField { inner, .. } => {
            shape_is_bit_stream(inner)
        }
        _ => false,
    }
}

/// Whether a multi-field shape is exactly a fat pointer's
/// two words: the data pointer at offset 0 and a 64-bit
/// length at offset 64 (the shape a `(ptr, len)` tuple
/// built for a DST transmute records).
pub(super) fn fat_words_fields(shape: &ir::ScalarShape) -> bool {
    match shape {
        ir::ScalarShape::Fields { fields, .. } => {
            fields.len() == 2
                && fields[0].0 == 0
                && fields[0].1 == ir::ScalarShape::Ptr
                && fields[1].0 == 64
                && matches!(
                    fields[1].1,
                    ir::ScalarShape::Int { bits: 64, .. }
                )
        }
        _ => false,
    }
}

/// Whether a scalar shape carries a thin pointer anywhere in
/// its payload: the pointer is a place identity in this model,
/// not a bit pattern, so a shape containing one cannot be
/// reinterpreted bit-wise without losing what the pointer
/// denotes.
pub(super) fn scalar_shape_has_ptr_leaf(shape: &ir::ScalarShape) -> bool {
    match shape {
        ir::ScalarShape::Ptr | ir::ScalarShape::FnPtr => true,
        ir::ScalarShape::AdtField { inner, .. } => {
            scalar_shape_has_ptr_leaf(inner)
        }
        ir::ScalarShape::NicheEnum { inner, .. } => {
            scalar_shape_has_ptr_leaf(inner)
        }
        ir::ScalarShape::AdtFields { fields, .. } => fields
            .iter()
            .any(|(_, _, s)| scalar_shape_has_ptr_leaf(s)),
        ir::ScalarShape::NicheEnumFields { fields, .. } => fields
            .iter()
            .any(|(_, _, s)| scalar_shape_has_ptr_leaf(s)),
        ir::ScalarShape::TaggedEnum { variants, .. } => variants
            .iter()
            .any(|v| v.fields.iter().any(|(_, _, s)| scalar_shape_has_ptr_leaf(s))),
        ir::ScalarShape::Array(elem, _) => scalar_shape_has_ptr_leaf(elem),
        _ => false,
    }
}

pub(super) fn is_thin_ptr_ty(ty: &mir::Ty) -> bool {
    match ty {
        mir::Ty::RawPtr(pointee, _) | mir::Ty::Ref(_, pointee, _) => !matches!(
            **pointee,
            mir::Ty::Slice(_)
                | mir::Ty::Str
                | mir::Ty::Dynamic(..)
                | mir::Ty::Alias(..)
        ),
        _ => false,
    }
}

/// Whether a converted operand is a pointer constant: one of
/// the constant forms that evaluate to a pointer value (a
/// reference, an address-only pointer, a null pointer, a static
/// reference, a vtable pointer, a fat pointer, or a callable).
/// A cast of such a constant has no place type to judge from --
/// the constant's own form is the evidence that the source is a
/// pointer.
pub(super) fn is_pointer_constant(v: &ir::Operand) -> bool {
    matches!(
        v,
        ir::Operand::Constant(
            ir::Constant::Ref(_)
                | ir::Constant::IntPtr(_)
                | ir::Constant::NullPtr
                | ir::Constant::StaticRef { .. }
                | ir::Constant::VTablePtr(_)
                | ir::Constant::FatPtr { .. }
                | ir::Constant::DynRef { .. }
                | ir::Constant::FnDef(_)
                | ir::Constant::Closure(..)
        )
    )
}

/// Whether `ty` is a fat pointer type: a raw pointer or
/// reference whose pointee carries a metadata half (a slice,
/// `str` or `dyn` pointee). A fat pointer's value in this model
/// is its [`ir::Value::FatRef`] / [`ir::Value::DynRef`], so
/// transmutes between fat pointer types — and between a fat
/// pointer and a transparent wrapper around one — reinterpret
/// nothing either: the metadata is part of the value, and the
/// pointee type lives in the destination's declared type.
/// Is `ty` the `str` type or a byte slice `[u8]`?
pub(super) fn is_str_or_byte_slice(ty: &mir::Ty) -> bool {
    match ty {
        mir::Ty::Str => true,
        mir::Ty::Slice(elem) => matches!(**elem, mir::Ty::Uint(mir::UintTy::U8)),
        _ => false,
    }
}

pub(super) fn is_fat_ptr_ty(ty: &mir::Ty) -> bool {
    match ty {
        mir::Ty::RawPtr(pointee, _) | mir::Ty::Ref(_, pointee, _) => matches!(
            **pointee,
            mir::Ty::Slice(_) | mir::Ty::Str | mir::Ty::Dynamic(..)
        ),
        _ => false,
    }
}

/// Whether a type mentions a trait object anywhere inside it
/// (a `dyn` type itself, a reference or pointer to one, a
/// `Box<dyn ...>`, a tuple holding one): such a value's method
/// calls dispatch through a vtable.
pub(super) fn ty_mentions_dyn(ty: &mir::Ty) -> bool {
    match ty {
        mir::Ty::Dynamic(..) => true,
        mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => {
            ty_mentions_dyn(inner)
        }
        mir::Ty::Adt(_, args) | mir::Ty::Closure(_, args) => {
            args.iter().any(ty_mentions_dyn)
        }
        mir::Ty::Tuple(ts) => ts.iter().any(ty_mentions_dyn),
        mir::Ty::Slice(elem) | mir::Ty::Array(elem, _) => {
            ty_mentions_dyn(elem)
        }
        _ => false,
    }
}

/// The pointee type of a pointer or reference type.
/// The constant form of a uniquely determined zero-sized value
/// (the only values [`Converter::zst_value`] synthesizes), or
/// `None` for a value kind with no constant form.
pub(super) fn value_to_constant(v: &Value) -> Option<ir::Constant> {
    match v {
        Value::Unit => Some(ir::Constant::Unit),
        Value::Tuple(items) => items
            .iter()
            .map(value_to_constant)
            .collect::<Option<Vec<_>>>()
            .map(ir::Constant::Tuple),
        Value::Enum { variant, fields } => fields
            .iter()
            .map(value_to_constant)
            .collect::<Option<Vec<_>>>()
            .map(|fields| ir::Constant::Adt {
                variant: *variant,
                fields,
            }),
        Value::Closure { def, env } => env
            .iter()
            .map(value_to_constant)
            .collect::<Option<Vec<_>>>()
            .map(|env| ir::Constant::Closure(def.clone(), env)),
        Value::FnPtr(name) => Some(ir::Constant::FnDef(name.clone())),
        _ => None,
    }
}

/// Whether `v` holds a callable value (a closure or a function
/// pointer) anywhere inside it.
pub(super) fn value_contains_callable(v: &Value) -> bool {
    match v {
        Value::Closure { .. } | Value::FnPtr(_) => true,
        Value::Tuple(items) => items.iter().any(value_contains_callable),
        Value::Enum { fields, .. } => fields.iter().any(value_contains_callable),
        _ => false,
    }
}

/// The shape of a transparent wrapper struct for `transmute`
/// purposes: a struct whose fields are exactly one pointer
/// payload field (thin, or fat when matched by
/// [`Converter::fat_wrapper_shape`]) plus any number of
/// zero-sized fields whose values are uniquely determined by
/// their types (`NonNull`, `Pin`, `PtrAndZst` are the shapes
/// std transmutes through).
pub(super) struct WrapperShape {
    /// Total number of fields of the struct.
    pub(super) field_count: usize,
    /// Index of the payload (pointer) field.
    pub(super) payload_index: usize,
    /// Instantiated type of the payload field.
    pub(super) payload_ty: mir::Ty,
    /// Unique values of the zero-sized fields, by field index.
    pub(super) fill: Vec<(usize, ir::Constant)>,
}

/// Substitute a definition's generic arguments for its parameters
/// in a declared field type (see `mir::AdtShape`): `Param(i)` is
/// replaced by the instantiation's `i`-th argument, recursively.
/// A parameter with no corresponding argument is left as-is (it
/// makes the type unprovable downstream, never unsound).
pub(super) fn subst_ty(ty: &mir::Ty, args: &[mir::Ty]) -> mir::Ty {
    match ty {
        mir::Ty::Param(i, _) => args.get(*i as usize).cloned().unwrap_or_else(|| ty.clone()),
        mir::Ty::Tuple(ts) => {
            mir::Ty::Tuple(ts.iter().map(|t| subst_ty(t, args)).collect())
        }
        mir::Ty::Array(e, l) => mir::Ty::Array(Box::new(subst_ty(e, args)), l.clone()),
        mir::Ty::Adt(d, a) => {
            mir::Ty::Adt(*d, a.iter().map(|t| subst_ty(t, args)).collect())
        }
        mir::Ty::Ref(r, t, m) => mir::Ty::Ref(r.clone(), Box::new(subst_ty(t, args)), *m),
        mir::Ty::RawPtr(t, m) => mir::Ty::RawPtr(Box::new(subst_ty(t, args)), *m),
        mir::Ty::Slice(t) => mir::Ty::Slice(Box::new(subst_ty(t, args))),
        mir::Ty::Pat(t, pat) => mir::Ty::Pat(Box::new(subst_ty(t, args)), pat.clone()),
        mir::Ty::Alias(k, d, a) => {
            mir::Ty::Alias(*k, *d, a.iter().map(|t| subst_ty(t, args)).collect())
        }
        mir::Ty::Closure(d, a) => {
            mir::Ty::Closure(*d, a.iter().map(|t| subst_ty(t, args)).collect())
        }
        mir::Ty::Coroutine(d, a) => {
            mir::Ty::Coroutine(*d, a.iter().map(|t| subst_ty(t, args)).collect())
        }
        mir::Ty::CoroutineWitness(d, a) => {
            mir::Ty::CoroutineWitness(*d, a.iter().map(|t| subst_ty(t, args)).collect())
        }
        mir::Ty::FnDef(d, a) => {
            mir::Ty::FnDef(*d, a.iter().map(|t| subst_ty(t, args)).collect())
        }
        mir::Ty::Dynamic(bounds, t, r) => {
            mir::Ty::Dynamic(bounds.clone(), Box::new(subst_ty(t, args)), r.clone())
        }
        _ => ty.clone(),
    }
}

/// Walk one place's projection chain the way [`Converter::place_ty`]
/// does, recording the declared type of every ADT field stepped into.
fn scan_place_fields(table: &mut AdtFieldTys, body: &mir::Body, place: &mir::Place) {
    let Some(decl) = body.local_decls.get(place.local.index()) else {
        return;
    };
    let mut ty = decl.ty.clone();
    let mut variant = 0u32;
    for elem in &place.projection {
        match elem {
            mir::PlaceElem::Field(idx, field_ty) => {
                // Coroutine places project the state object's fields
                // the same way ADT places do; the table is keyed by
                // the coroutine definition and its state (variant).
                let owner = match &ty {
                    mir::Ty::Adt(def, args) => Some((*def, args.clone())),
                    mir::Ty::Coroutine(def, args) => Some((*def, args.clone())),
                    _ => None,
                };
                if let Some((def, args)) = owner {
                    table.record(
                        AdtFieldKey {
                            def,
                            generic_args: args,
                            variant,
                            field: *idx,
                        },
                        field_ty.clone(),
                    );
                }
                ty = field_ty.clone();
                variant = 0;
            }
            mir::PlaceElem::Downcast(_, v) => variant = *v,
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

fn scan_operand_fields(table: &mut AdtFieldTys, body: &mir::Body, op: &mir::Operand) {
    if let Some(place) = op.place() {
        scan_place_fields(table, body, place);
    }
}

fn scan_rvalue_fields(table: &mut AdtFieldTys, body: &mir::Body, rv: &mir::Rvalue) {
    use mir::Rvalue as R;
    match rv {
        R::Use(op)
        | R::Repeat(op, _)
        | R::Cast(_, op, _)
        | R::UnaryOp(_, op)
        | R::ShallowInitBox(op, _) => scan_operand_fields(table, body, op),
        R::BinaryOp(_, pair) | R::CheckedBinaryOp(_, pair) => {
            scan_operand_fields(table, body, &pair.0);
            scan_operand_fields(table, body, &pair.1);
        }
        R::Ref(_, _, place)
        | R::RawPtr(_, place)
        | R::Len(place)
        | R::Discriminant(place)
        | R::Reborrow(_, _, place)
        | R::CopyForDeref(place) => scan_place_fields(table, body, place),
        R::Aggregate(_, ops) => {
            for op in ops {
                scan_operand_fields(table, body, op);
            }
        }
        R::ThreadLocalRef(_) | R::NullaryOp(..) => {}
    }
}

fn scan_statement_fields(table: &mut AdtFieldTys, body: &mir::Body, stmt: &mir::Statement) {
    use mir::StatementKind as S;
    match &stmt.kind {
        S::Assign(pair) => {
            let (place, rv) = &**pair;
            scan_place_fields(table, body, place);
            scan_rvalue_fields(table, body, rv);
        }
        S::FakeRead(pair) => scan_place_fields(table, body, &pair.1),
        S::SetDiscriminant { place, .. } => scan_place_fields(table, body, place),
        S::Retag(_, place) | S::PlaceMention(place) => scan_place_fields(table, body, place),
        S::AscribeUserType(pair, _) => scan_place_fields(table, body, &pair.0),
        S::Intrinsic(intr) => match &**intr {
            mir::NonDivergingIntrinsic::CopyNonOverlapping { src, dst, count } => {
                scan_operand_fields(table, body, src);
                scan_operand_fields(table, body, dst);
                scan_operand_fields(table, body, count);
            }
            mir::NonDivergingIntrinsic::Assume(op) => scan_operand_fields(table, body, op),
        },
        S::StorageLive(_)
        | S::StorageDead(_)
        | S::Coverage
        | S::ConstEvalCounter
        | S::Nop => {}
    }
}

fn scan_terminator_fields(table: &mut AdtFieldTys, body: &mir::Body, kind: &mir::TerminatorKind) {
    use mir::TerminatorKind as T;
    match kind {
        T::SwitchInt { discr, .. } => scan_operand_fields(table, body, discr),
        T::Call {
            func,
            args,
            destination,
            ..
        } => {
            scan_operand_fields(table, body, func);
            for arg in args {
                scan_operand_fields(table, body, arg);
            }
            scan_place_fields(table, body, destination);
        }
        T::Assert { cond, .. } => scan_operand_fields(table, body, cond),
        T::Drop { place, .. } => scan_place_fields(table, body, place),
        T::Yield {
            value, resume_arg, ..
        } => {
            scan_operand_fields(table, body, value);
            scan_place_fields(table, body, resume_arg);
        }
        T::InlineAsm { operands, .. } => {
            for op in operands {
                if let Some(value) = &op.in_value {
                    scan_operand_fields(table, body, value);
                }
            }
        }
        T::TailCall { func, args } => {
            scan_operand_fields(table, body, func);
            for arg in args {
                scan_operand_fields(table, body, arg);
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

pub(super) fn scan_body_fields(table: &mut AdtFieldTys, body: &mir::Body) {
    for bb in &body.basic_blocks {
        for stmt in &bb.statements {
            scan_statement_fields(table, body, stmt);
        }
        if let Some(term) = &bb.terminator {
            scan_terminator_fields(table, body, &term.kind);
        }
    }
}

/// Operand counts of ADT aggregate constructions, per
/// (def, generic args, variant). An aggregate rvalue lists every
/// field of the variant it builds, so its operand count proves how
/// many fields that variant holds; drop-glue conversion uses it to
/// check that the field types recovered from projections cover the
/// whole variant.
#[derive(Default)]
pub(super) struct AdtAggregateCounts {
    pub(super) entries: Vec<(mir::DefId, Vec<mir::Ty>, u32, usize)>,
}

impl AdtAggregateCounts {
    fn record(
        &mut self,
        def: mir::DefId,
        generic_args: Vec<mir::Ty>,
        variant: u32,
        count: usize,
    ) {
        // Construction sites of one variant must agree on its field
        // count; the first sighting wins, as in `AdtFieldTys`.
        if self
            .entries
            .iter()
            .any(|(d, a, v, _)| *d == def && *a == generic_args && *v == variant)
        {
            return;
        }
        self.entries.push((def, generic_args, variant, count));
    }

    pub(super) fn get(&self, def: mir::DefId, generic_args: &[mir::Ty], variant: u32) -> Option<usize> {
        self.entries
            .iter()
            .find(|(d, a, v, _)| *d == def && a.as_slice() == generic_args && *v == variant)
            .map(|e| e.3)
    }
}

/// The declared type of a place, walking its projections the way
/// `scan_place_fields` does. Used to type aggregate operands: an
/// operand filled into field `i` of an ADT aggregate carries that
/// field's declared type, so a construction site records field
/// types that no `Field` projection anywhere ever reveals (a field
/// that is only ever written, never read).
fn operand_place_ty(body: &mir::Body, place: &mir::Place) -> Option<mir::Ty> {
    let mut ty = body.local_decls.get(place.local.index())?.ty.clone();
    for elem in &place.projection {
        // A pattern type is its base type for every
        // projection step.
        while let mir::Ty::Pat(inner, _) = ty {
            ty = *inner;
        }
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
            mir::PlaceElem::Downcast(_, _) | mir::PlaceElem::Subslice { .. } => ty,
            mir::PlaceElem::OpaqueCast(t) | mir::PlaceElem::Subtype(t) => t.clone(),
        };
    }
    Some(ty)
}

pub(super) fn scan_body_aggregates(
    counts: &mut AdtAggregateCounts,
    fields: &mut AdtFieldTys,
    body: &mir::Body,
) {
    for bb in &body.basic_blocks {
        for stmt in &bb.statements {
            if let mir::StatementKind::Assign(pair) = &stmt.kind {
                let (_, rv) = &**pair;
                if let mir::Rvalue::Aggregate(kind, ops) = rv {
                    // A coroutine creation aggregate is recorded as
                    // its unresumed state (variant 0), like an ADT
                    // aggregate of that variant.
                    let aggregate_key = match &**kind {
                        mir::AggregateKind::Adt {
                            def,
                            variant_index,
                            generic_args,
                            field_index,
                            ..
                        } => Some((
                            *def,
                            generic_args.clone(),
                            *variant_index,
                            *field_index,
                        )),
                        mir::AggregateKind::Coroutine { def, generic_args } => {
                            Some((*def, generic_args.clone(), 0, None))
                        }
                        _ => None,
                    };
                    if let Some((def, generic_args, variant_index, field_index)) =
                        aggregate_key
                    {
                        // A union aggregate lists only its active
                        // field's operand: the operand belongs to
                        // that field (not to operand position 0),
                        // and the operand count says nothing about
                        // the variant's field count, so no count is
                        // recorded for it.
                        if field_index.is_none() {
                            counts.record(
                                def,
                                generic_args.clone(),
                                variant_index,
                                ops.len(),
                            );
                        }
                        for (i, op) in ops.iter().enumerate() {
                            if let Some(place) = op.place() {
                                if let Some(ty) = operand_place_ty(body, place) {
                                    fields.record(
                                        AdtFieldKey {
                                            def,
                                            generic_args: generic_args.clone(),
                                            variant: variant_index,
                                            field: field_index
                                                .unwrap_or(i as u32),
                                        },
                                        ty,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
