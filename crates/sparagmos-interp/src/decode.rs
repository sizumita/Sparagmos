//! Pre-decoded statement layer for the interpreter's run loop.
//!
//! The run loop used to execute the IR exactly as conversion
//! produced it: every statement execution re-resolved its places
//! (building a fresh resolved-place vector), re-read and re-wrote
//! values through the general resolved-place machinery, re-hashed
//! the function's name to find its body again on every plain-local
//! write, re-built the human-readable description of every
//! projected place it read, and re-evaluated every constant
//! operand from its [`Constant`] form. A profile of the scalar
//! bench kernels (see docs/PERFORMANCE.md) attributes the bulk of
//! execution time to exactly that repeated work — place
//! resolution and the read/write machinery around 30%, value
//! clone/drop around 16%, name hashing around 6%, description
//! formatting around 10% — not to the statement dispatch itself.
//!
//! This module decodes each [`Body`] once per interpreter into a
//! [`DecodedBody`] whose statements carry everything the loop
//! needs in execution-ready form:
//!
//! * operands over projection-less locals become direct slot
//!   accesses ([`DSrc::Local`] / [`DSrc::MoveLocal`]), skipping
//!   place resolution entirely;
//! * operands over places whose projection is only `Field` /
//!   `Downcast` steps become [`DSrc::LocalProj`]: the resolved
//!   projection and the place description are computed once, at
//!   decode time, and execution walks the value with the same
//!   `project_value` / `navigate_mut` the general path uses;
//! * constant operands that evaluate purely (no `StaticRef`
//!   anywhere inside — those seed the statics store and keep
//!   their first-touch timing) are evaluated once, at decode
//!   time, into [`DSrc::Const`];
//! * destinations get the same three-way split ([`DDst`]), so a
//!   plain-local write is a slot store plus the same
//!   `normalize_for_ty` validation the general write performs,
//!   without re-resolving the place or re-hashing the function
//!   name;
//! * a `SwitchInt` terminator whose patterns are all unique gets
//!   a sorted target table ([`DecodedBlock::switch`]) the loop
//!   binary-searches instead of scanning linearly.
//!
//! Everything the fast paths do not cover — places with `Deref`,
//! `Index`, `ConstantIndex` or `Subslice` steps, the rvalue forms
//! whose evaluation is entangled with place resolution
//! (`Ref`, `Unsize`, `BoxDerefTransmute`, ...), and the cold
//! statement kinds — decodes to a form that calls the original
//! execution functions on the original IR, so behaviour there is
//! the original behaviour by construction. The fast paths
//! themselves mirror the general paths step for step, including
//! their error cases: an uninitialized slot, for example, always
//! delegates back to the original read so the zero-sized-local
//! fallback and the exact error are the original ones.
//!
//! Decoding is a pure function of the body (plus constant
//! evaluation, which is itself pure for the constants admitted
//! here), and bodies never change during a run, so decoded bodies
//! are cached on the interpreter, keyed by body address.

use std::rc::Rc;

use crate::error::{InterpError, InterpResult};
use crate::interp::{
    cast_place_ty, cast_src_ty, constant_ty, describe_resolved, eval_binary, eval_cast, eval_unary,
    is_vec_adt_ty, navigate_mut_grow, operand_pointee_ty, project_value_ev,
    transparent_wrapper_ty, Frame, Interpreter,
};
use crate::ir::{
    BasicBlock, BinOp, Body, Constant, Local, Operand, Place, ProjectionElem, Rvalue, Statement,
    StatementKind, TerminatorKind, Ty, UnOp,
};
use crate::value::{RefProj, Value};

/// A function body decoded for execution (see the module docs).
pub(crate) struct DecodedBody {
    pub(crate) blocks: Vec<DecodedBlock>,
}

/// One basic block's decoded statements (mirroring the IR block's
/// statements one for one) plus its terminator's switch table.
pub(crate) struct DecodedBlock {
    pub(crate) stmts: Vec<DStmt>,
    /// `SwitchInt` targets sorted by pattern, present only when
    /// the block's terminator is a `SwitchInt` whose patterns are
    /// all unique (with a duplicate pattern the linear scan's
    /// first-match order is observable, so no table is built).
    pub(crate) switch: Option<Box<[(u128, BasicBlock)]>>,
}

/// A decoded statement.
pub(crate) enum DStmt {
    Assign {
        dst: DDst,
        rvalue: DRvalue,
    },
    /// `StorageLive` / `StorageDead` / `Nop`: no-ops (the loop's
    /// step tick still counts them, exactly as before).
    Nop,
    /// Any other statement kind, executed by the original
    /// statement interpreter on the original statement.
    Cold(Statement),
}

/// A decoded assignment destination.
pub(crate) enum DDst {
    /// A projection-less local of the executing frame.
    Local(u32),
    /// A place over a local of the executing frame whose
    /// projection is only `Field` / `Downcast` steps, with the
    /// resolved projection and the place description precomputed.
    LocalProj {
        local: u32,
        proj: Rc<[RefProj]>,
        desc: Rc<str>,
    },
    /// Any other place, written through the original path.
    General(Place),
}

impl DDst {
    fn local(&self) -> usize {
        match self {
            DDst::Local(idx) => *idx as usize,
            DDst::LocalProj { local, .. } => *local as usize,
            DDst::General(place) => place.local.0,
        }
    }
}

/// A decoded place used as a read source (the `Len` /
/// `Discriminant` / `Reborrow` rvalues).
pub(crate) enum DPlace {
    Local(u32),
    LocalProj {
        local: u32,
        proj: Rc<[RefProj]>,
        desc: Rc<str>,
        place: Place,
    },
    General(Place),
}

/// A decoded operand.
pub(crate) enum DSrc {
    /// `Copy` of a projection-less local.
    Local(u32),
    /// `Move` of a projection-less local.
    MoveLocal(u32),
    /// `Move` of a place over a local whose projection is only
    /// `Field` / `Downcast` steps: read through the projected
    /// fast path, then leave the moved-out part uninitialized,
    /// exactly as the general `Move` evaluation does.
    MoveLocalProj {
        local: u32,
        proj: Rc<[RefProj]>,
        desc: Rc<str>,
        place: Place,
    },
    /// `Copy` of a place over a local whose projection is only
    /// `Field` / `Downcast` steps, with the resolved projection
    /// and the place description precomputed. The original place
    /// is kept for the uninitialized-base delegation.
    LocalProj {
        local: u32,
        proj: Rc<[RefProj]>,
        desc: Rc<str>,
        place: Place,
    },
    /// A constant operand evaluated once at decode time, with
    /// the source type the constant records for itself (a float
    /// constant's width; see `constant_ty`): a cast consuming the
    /// predecoded constant still converts at that width.
    Const(Value, Option<Ty>),
    /// Any other operand, evaluated by the original path.
    Operand(Operand),
}

/// A decoded rvalue: the hot forms with decoded operands, the
/// rest as the original rvalue executed by the original
/// evaluator.
pub(crate) enum DRvalue {
    Use(DSrc),
    BinaryOp(BinOp, DSrc, DSrc),
    CheckedBinaryOp(BinOp, DSrc, DSrc),
    UnaryOp(UnOp, DSrc),
    AggregateTuple(Box<[DSrc]>),
    AggregateAdt {
        variant: u32,
        operands: Box<[DSrc]>,
        field_index: Option<u32>,
        units_from: Option<(u64, usize)>,
    },
    ClosureAggregate {
        def: String,
        operands: Box<[DSrc]>,
    },
    CoroutineAggregate(Box<[DSrc]>),
    Cast(DSrc, Ty),
    Len(DPlace, Place),
    Discriminant(DPlace, Rc<[i128]>, Place),
    Reborrow(DPlace),
    PtrToInt(DSrc),
    IntToPtr(DSrc),
    EnumToInt(DSrc, Rc<[i128]>, Ty),
    IntToEnum(DSrc, Rc<[i128]>),
    IntToEnumWrapped(DSrc, Rc<[i128]>, u32, usize, Rc<[(u32, Constant)]>),
    Orig(Box<Rvalue>),
}

/// How a place's projection classifies for decoding.
enum PlaceKind {
    Local,
    LocalProj(Rc<[RefProj]>, Rc<str>),
    General,
}

/// Classify a place: projection-less places and places whose
/// projection is only `Field` / `Downcast` steps get fast paths.
/// For those chains the general resolver's work is purely
/// syntactic — each step pushes the corresponding [`RefProj`]
/// onto the executing frame's (local, projection) pair — so the
/// resolved projection and the place description can be computed
/// once, here. Any other step (`Deref`, `Index`, `ConstantIndex`,
/// `Subslice`) makes resolution depend on run-time values, and
/// the place stays on the general path.
/// The `LocalProj` classification of `place`, for callers
/// outside this module (terminator operand evaluation in the
/// run loop): the precomputed resolved projection and place
/// description of a `Field` / `Downcast`-only place over a
/// local, or `None` for any other shape.
pub(crate) fn classify_local_proj(place: &Place) -> Option<(Rc<[RefProj]>, Rc<str>)> {
    match classify_place(place) {
        PlaceKind::LocalProj(proj, desc) => Some((proj, desc)),
        _ => None,
    }
}

fn classify_place(place: &Place) -> PlaceKind {
    if place.projection.is_empty() {
        return PlaceKind::Local;
    }
    if !place
        .projection
        .iter()
        .all(|e| matches!(e, ProjectionElem::Field(_) | ProjectionElem::Downcast(_)))
    {
        return PlaceKind::General;
    }
    let proj: Vec<RefProj> = place
        .projection
        .iter()
        .map(|e| match e {
            ProjectionElem::Field(i) => RefProj::Field(*i),
            ProjectionElem::Downcast(v) => RefProj::Downcast(*v),
            _ => unreachable!("classified as field/downcast only"),
        })
        .collect();
    let desc: Rc<str> = Rc::from(describe_resolved(place.local.0, &proj));
    PlaceKind::LocalProj(proj.into(), desc)
}

/// The original place a decoded `LocalProj` destination was
/// classified from, rebuilt from its precomputed projection
/// (only `Field` / `Downcast` steps occur there, by the
/// classification that produced it). `None` for any other
/// step, which the classification never admits.
fn local_proj_place(local: u32, proj: &[RefProj]) -> Option<Place> {
    let projection = proj
        .iter()
        .map(|step| match step {
            RefProj::Field(idx) => Some(ProjectionElem::Field(*idx)),
            RefProj::Downcast(v) => Some(ProjectionElem::Downcast(*v)),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Place {
        local: Local(local as usize),
        projection,
    })
}

/// Whether a constant evaluates without touching interpreter
/// state: everything except `StaticRef`, whose evaluation seeds
/// the statics store and so keeps its first-touch timing on the
/// general path.
fn constant_is_pure(c: &Constant) -> bool {
    match c {
        Constant::StaticRef { .. } => false,
        Constant::Tuple(fields) | Constant::SliceVal(fields) => {
            fields.iter().all(constant_is_pure)
        }
        Constant::Adt { fields, .. } => fields.iter().all(constant_is_pure),
        Constant::Ref(inner) => constant_is_pure(inner),
        _ => true,
    }
}

impl<'a> Interpreter<'a> {
    /// The decoded form of `body`, decoded on first use and
    /// cached for the rest of this interpreter's life. The cache
    /// is keyed by the body's address: the program outlives the
    /// interpreter and its bodies never move or change once the
    /// interpreter borrows the program, so an address names one
    /// body for the whole run.
    pub(crate) fn decoded_body(&self, body: &Body) -> Rc<DecodedBody> {
        let key = body as *const Body as usize;
        if let Some(found) = self.decoded_cache().borrow().get(&key) {
            return found.clone();
        }
        let decoded = Rc::new(self.decode_body(body));
        self.decoded_cache()
            .borrow_mut()
            .insert(key, decoded.clone());
        decoded
    }

    fn decode_body(&self, body: &Body) -> DecodedBody {
        let blocks = body
            .blocks
            .iter()
            .map(|block| DecodedBlock {
                stmts: block
                    .statements
                    .iter()
                    .map(|stmt| self.decode_stmt(stmt))
                    .collect(),
                switch: decode_switch(&block.terminator),
            })
            .collect();
        DecodedBody { blocks }
    }

    fn decode_stmt(&self, stmt: &Statement) -> DStmt {
        match &stmt.kind {
            StatementKind::Assign(place, rvalue) => DStmt::Assign {
                dst: self.decode_dst(place),
                rvalue: self.decode_rvalue(rvalue),
            },
            StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Nop => DStmt::Nop,
            _ => DStmt::Cold(stmt.clone()),
        }
    }

    fn decode_dst(&self, place: &Place) -> DDst {
        match classify_place(place) {
            PlaceKind::Local => DDst::Local(place.local.0 as u32),
            PlaceKind::LocalProj(proj, desc) => DDst::LocalProj {
                local: place.local.0 as u32,
                proj,
                desc,
            },
            PlaceKind::General => DDst::General(place.clone()),
        }
    }

    fn decode_dplace(&self, place: &Place) -> DPlace {
        match classify_place(place) {
            PlaceKind::Local => DPlace::Local(place.local.0 as u32),
            PlaceKind::LocalProj(proj, desc) => DPlace::LocalProj {
                local: place.local.0 as u32,
                proj,
                desc,
                place: place.clone(),
            },
            PlaceKind::General => DPlace::General(place.clone()),
        }
    }

    fn decode_src(&self, op: &Operand) -> DSrc {
        match op {
            Operand::Constant(c) => {
                if constant_is_pure(c) {
                    if let Ok(value) = self.eval_constant(c) {
                        return DSrc::Const(value, constant_ty(c));
                    }
                }
                DSrc::Operand(op.clone())
            }
            Operand::Copy(place) => match classify_place(place) {
                PlaceKind::Local => DSrc::Local(place.local.0 as u32),
                PlaceKind::LocalProj(proj, desc) => DSrc::LocalProj {
                    local: place.local.0 as u32,
                    proj,
                    desc,
                    place: place.clone(),
                },
                PlaceKind::General => DSrc::Operand(op.clone()),
            },
            Operand::Move(place) => match classify_place(place) {
                PlaceKind::Local => DSrc::MoveLocal(place.local.0 as u32),
                PlaceKind::LocalProj(proj, desc) => DSrc::MoveLocalProj {
                    local: place.local.0 as u32,
                    proj,
                    desc,
                    place: place.clone(),
                },
                PlaceKind::General => DSrc::Operand(op.clone()),
            },
        }
    }

    fn decode_rvalue(&self, rv: &Rvalue) -> DRvalue {
        match rv {
            Rvalue::Use(op) => DRvalue::Use(self.decode_src(op)),
            Rvalue::BinaryOp(op, pair) => {
                DRvalue::BinaryOp(*op, self.decode_src(&pair.0), self.decode_src(&pair.1))
            }
            Rvalue::CheckedBinaryOp(op, pair) => DRvalue::CheckedBinaryOp(
                *op,
                self.decode_src(&pair.0),
                self.decode_src(&pair.1),
            ),
            Rvalue::UnaryOp(UnOp::PtrMetadata, _) => DRvalue::Orig(Box::new(rv.clone())),
            Rvalue::UnaryOp(op, operand) => {
                DRvalue::UnaryOp(*op, self.decode_src(operand))
            }
            Rvalue::AggregateTuple(operands) => DRvalue::AggregateTuple(
                operands.iter().map(|o| self.decode_src(o)).collect(),
            ),
            Rvalue::AggregateAdt {
                variant,
                operands,
                field_index,
                units_from,
            } => DRvalue::AggregateAdt {
                variant: *variant,
                operands: operands.iter().map(|o| self.decode_src(o)).collect(),
                field_index: *field_index,
                units_from: *units_from,
            },
            Rvalue::ClosureAggregate { def, operands } => DRvalue::ClosureAggregate {
                def: def.clone(),
                operands: operands.iter().map(|o| self.decode_src(o)).collect(),
            },
            Rvalue::CoroutineAggregate(operands) => DRvalue::CoroutineAggregate(
                operands.iter().map(|o| self.decode_src(o)).collect(),
            ),
            Rvalue::Cast(operand, ty) => {
                DRvalue::Cast(self.decode_src(operand), ty.clone())
            }
            Rvalue::Len(place) => {
                DRvalue::Len(self.decode_dplace(place), place.clone())
            }
            Rvalue::Discriminant(place, discriminants) => DRvalue::Discriminant(
                self.decode_dplace(place),
                Rc::from(discriminants.as_slice()),
                place.clone(),
            ),
            Rvalue::Reborrow(place) => DRvalue::Reborrow(self.decode_dplace(place)),
            Rvalue::PtrToInt(operand) => DRvalue::PtrToInt(self.decode_src(operand)),
            Rvalue::IntToPtr(operand) => DRvalue::IntToPtr(self.decode_src(operand)),
            Rvalue::EnumToInt(operand, discriminants, ty) => DRvalue::EnumToInt(
                self.decode_src(operand),
                Rc::from(discriminants.as_slice()),
                ty.clone(),
            ),
            Rvalue::IntToEnum(operand, discriminants) => DRvalue::IntToEnum(
                self.decode_src(operand),
                Rc::from(discriminants.as_slice()),
            ),
            Rvalue::IntToEnumWrapped {
                operand,
                discriminants,
                payload_index,
                field_count,
                fill,
            } => DRvalue::IntToEnumWrapped(
                self.decode_src(operand),
                Rc::from(discriminants.as_slice()),
                *payload_index,
                *field_count,
                Rc::from(fill.as_slice()),
            ),
            // DataPtr, RawPtrAggregate, Unsize, BoxDerefTransmute,
            // PtrField, PtrFieldFirstElem, PtrFirstElem,
            // PtrArrayBase, Reinterpret,
            // ShapedLoad,
            // PtrOffset, Ref, CopyForDeref,
            // ThreadLocalRef, Unsupported: their evaluation is
            // entangled with place resolution or is an explicit
            // error; the original evaluator runs them.
            _ => DRvalue::Orig(Box::new(rv.clone())),
        }
    }

    /// Execute one decoded statement (the run loop's per-step
    /// entry, replacing the direct `exec_statement` call).
    pub(crate) fn exec_dstmt(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        stmt: &DStmt,
    ) -> InterpResult<()> {
        match stmt {
            DStmt::Nop => Ok(()),
            DStmt::Cold(stmt) => self.exec_statement(frames, body, stmt),
            DStmt::Assign { dst, rvalue } => {
                let dest_ty = body.local_ty(Local(dst.local()));
                let value = self.eval_drvalue(frames, body, rvalue, dest_ty)?;
                self.write_ddst(frames, body, dst, value)
            }
        }
    }

    /// Read a projection-less local of the executing frame. A
    /// slot holding a real value is cloned directly; anything
    /// else (an uninitialized slot, an out-of-range index)
    /// delegates to the original read, which produces the
    /// zero-sized fallback or the exact original error.
    fn read_local(
        &self,
        frames: &[Frame],
        body: &Body,
        idx: u32,
    ) -> InterpResult<Value> {
        let idx = idx as usize;
        let frame = frames.last().expect("frame");
        match frame.locals.get(idx) {
            Some(value) if *value != Value::Uninit => Ok(value.clone()),
            _ => self.read_place(frames, body, &Place::from_index(idx)),
        }
    }

    /// Whether `place`'s projection denotes a `Vec`'s
    /// structural fields at any step: the declared type is
    /// tracked from the place's local through field steps
    /// exactly as the general resolver tracks it (see
    /// `Interpreter::adt_field_ty`), and a `Field(0)` or
    /// `Field(1)` step whose base type is a `Vec` ADT names
    /// the buffer field or the length slot -- neither is
    /// storage in the element-list value the projected fast
    /// paths walk, so those paths must not serve the place.
    /// A `Field(0)` descent out of a named transparent
    /// wrapper (`ManuallyDrop<Vec>` above all) whose field
    /// type the tracking cannot name likewise delegates when
    /// a `Field(0)` / `Field(1)` step follows it: the payload
    /// may be a vector, a position the general resolver
    /// judges by the payload's shape.
    pub(crate) fn vec_structural_place(&self, body: &Body, place: &Place) -> bool {
        let mut cur_ty: Option<Ty> = body.local_ty(place.local).cloned();
        let mut variant = 0usize;
        // The suspicion the general resolver arms (see
        // `resolve_place_mode`): a `Field(0)` descent out of a
        // named transparent wrapper whose field type the
        // tracking cannot name leaves the payload's type
        // unknown, and a `Field(0)` / `Field(1)` step on the
        // payload may enter a `Vec`'s structural fields.
        let mut suspect = false;
        for elem in &place.projection {
            match elem {
                ProjectionElem::Field(idx) => {
                    if let Some(ty) = &cur_ty {
                        if is_vec_adt_ty(ty) && (*idx == 0 || *idx == 1) {
                            return true;
                        }
                    }
                    if suspect && (*idx == 0 || *idx == 1) {
                        return true;
                    }
                    let stepped = match &cur_ty {
                        Some(ty) => self.adt_field_ty(ty, variant, *idx),
                        None => None,
                    };
                    suspect = *idx == 0
                        && matches!(&cur_ty, Some(t) if transparent_wrapper_ty(t))
                        && stepped.is_none();
                    cur_ty = stepped;
                    variant = 0;
                }
                ProjectionElem::Downcast(v) => {
                    variant = *v as usize;
                    suspect = false;
                }
                _ => return false,
            }
        }
        false
    }

    /// Read a `Field` / `Downcast`-only projected place over a
    /// local of the executing frame, mirroring the general path
    /// (resolve, then `read_resolved`) step for step: the base
    /// slot is cloned, an uninitialized base delegates to the
    /// original read (zero-sized fallback and errors), the
    /// projection walks with `project_value` under the
    /// precomputed description, and a projected uninitialized
    /// read is rescued by the body's recorded zero-sized field
    /// value (`zst_field_value`) or reported as the same
    /// `UninitializedRead` the general path reports after
    /// projection.
    pub(crate) fn read_local_proj(
        &self,
        frames: &[Frame],
        body: &Body,
        local: u32,
        proj: &[RefProj],
        desc: &str,
        place: &Place,
    ) -> InterpResult<Value> {
        // A projection that walks a `Vec`'s structural fields
        // (its buffer-wrapper chain or its length slot) is not
        // the syntactic walk this fast path mirrors: the value
        // model holds a vector as its element list, so those
        // steps denote the vector's buffer bookkeeping, which
        // only the general resolver answers (see
        // `resolve_place_mode`'s buffer-chain walk). Delegate,
        // as for the uninitialized base below.
        if self.vec_structural_place(body, place) {
            return self.read_place(frames, body, place);
        }
        let local = local as usize;
        let frame = frames.last().expect("frame");
        let base = match frame.locals.get(local) {
            Some(value) => value.clone(),
            None => return self.read_place(frames, body, place),
        };
        if base == Value::Uninit {
            return self.read_place(frames, body, place);
        }
        // Mirror `read_resolved`: a projected read that comes
        // back uninitialized (as a value, or as the projection's
        // own uninitialized-read error) is rescued by the single
        // value of a zero-sized field whose store was elided,
        // recorded in the body's `zst_fields` table; any other
        // projection error is the original one.
        let value = match project_value_ev(
            Some(self.program()),
            body.local_ty(Local(local as usize)).cloned(),
            base,
            proj,
            &frame.func,
            desc,
        ) {
            Ok(v) if v != Value::Uninit => v,
            Ok(_) | Err(InterpError::UninitializedRead { .. }) => {
                match self.zst_field_value(frame, local, proj) {
                    Some(v) => v,
                    None => {
                        return Err(InterpError::UninitializedRead {
                            func: frame.func.clone(),
                            place: desc.to_string(),
                        });
                    }
                }
            }
            Err(e) => return Err(e),
        };
        Ok(value)
    }

    fn read_dplace(
        &self,
        frames: &[Frame],
        body: &Body,
        place: &DPlace,
    ) -> InterpResult<Value> {
        match place {
            DPlace::Local(idx) => self.read_local(frames, body, *idx),
            DPlace::LocalProj {
                local,
                proj,
                desc,
                place,
            } => self.read_local_proj(frames, body, *local, proj, desc, place),
            DPlace::General(place) => self.read_place(frames, body, place),
        }
    }

    fn eval_src(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        src: &DSrc,
    ) -> InterpResult<Value> {
        match src {
            DSrc::Const(value, _) => Ok(value.clone()),
            DSrc::Operand(op) => self.eval_operand(frames, body, op),
            DSrc::Local(idx) => self.read_local(frames, body, *idx),
            DSrc::MoveLocal(idx) => {
                let idx = *idx as usize;
                let taken = {
                    let frame = frames.last().expect("frame");
                    match frame.locals.get(idx) {
                        Some(value) if *value != Value::Uninit => value.clone(),
                        _ => {
                            return self.eval_operand(
                                frames,
                                body,
                                &Operand::Move(Place::from_index(idx)),
                            )
                        }
                    }
                };
                frames.last_mut().expect("frame").locals[idx] = Value::Uninit;
                Ok(taken)
            }
            DSrc::LocalProj {
                local,
                proj,
                desc,
                place,
            } => self.read_local_proj(frames, body, *local, proj, desc, place),
            DSrc::MoveLocalProj {
                local,
                proj,
                desc,
                place,
            } => {
                // A move out of a `Vec`'s structural fields
                // (see `read_local_proj`): the clearing below
                // navigates the element list, which is not the
                // storage the place denotes -- the general
                // operand evaluation owns the shape.
                if self.vec_structural_place(body, place) {
                    return self.eval_operand(
                        frames,
                        body,
                        &Operand::Move(place.clone()),
                    );
                }
                let value = self.read_local_proj(frames, body, *local, proj, desc, place)?;
                // The clearing `eval_operand` performs for a
                // move out of a projected place, through the
                // shared helper: the moved-out part is left
                // uninitialized, so a later `Drop` of the whole
                // value does not run its glue a second time.
                self.clear_local_proj(frames, *local as usize, proj, desc)?;
                Ok(value)
            }
        }
    }

    /// Write through a decoded destination, mirroring
    /// `write_place` + `write_resolved` for the shapes the fast
    /// paths cover.
    fn write_ddst(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        dst: &DDst,
        value: Value,
    ) -> InterpResult<()> {
        match dst {
            DDst::Local(idx) => {
                let idx = *idx as usize;
                let pos = frames.len() - 1;
                let slot = match frames[pos].locals.get_mut(idx) {
                    Some(slot) => slot,
                    None => {
                        return Err(InterpError::UnknownLocal {
                            func: frames[pos].func.clone(),
                            local: idx,
                        })
                    }
                };
                *slot = Self::normalize_for_ty(value, body.local_ty(Local(idx)))?;
                Ok(())
            }
            DDst::LocalProj { local, proj, desc } => {
                // A write to a `Vec`'s structural fields (see
                // `read_local_proj`): the slot navigation below
                // walks the element list, but the place denotes
                // the vector's buffer bookkeeping -- the
                // general write owns the shape (its `Vec`
                // buffer-slot and length-slot interceptions).
                if let Some(place) = local_proj_place(*local, proj) {
                    if self.vec_structural_place(body, &place) {
                        return self.write_place(frames, body, &place, value);
                    }
                }
                let local = *local as usize;
                let pos = frames.len() - 1;
                let func = frames[pos].func.clone();
                let slot = match frames[pos].locals.get_mut(local) {
                    Some(slot) => slot,
                    None => {
                        return Err(InterpError::UnknownLocal {
                            func,
                            local,
                        })
                    }
                };
                // The general path's projected write judges
                // the navigation's wrapper descent by the
                // local's declared type (see
                // `write_resolved_typed`); the fast path must
                // supply the same evidence, or a partially
                // materialised struct local's second field
                // write descends into its first field.
                let slot = navigate_mut_grow(
                    slot,
                    proj,
                    &func,
                    desc,
                    false,
                    Some(self.program()),
                    body.local_ty(Local(local)).cloned(),
                )?;
                *slot = value;
                Ok(())
            }
            DDst::General(place) => self.write_place(frames, body, place, value),
        }
    }

    /// The declared pointee type of the pointer value a decoded
    /// source evaluates to, when it is statically known (a plain
    /// local of pointer type, or an operand with one): the leaf
    /// type the address normalisation prices element steps from
    /// (see [`Interpreter::pointer_address`]).
    fn dsrc_pointee_ty(body: &Body, src: &DSrc) -> Option<Ty> {
        match src {
            DSrc::Local(idx) | DSrc::MoveLocal(idx) => {
                match body.local_ty(Local(*idx as usize)) {
                    Some(Ty::Ref(inner)) => Some((**inner).clone()),
                    _ => None,
                }
            }
            DSrc::Operand(op) => operand_pointee_ty(body, op),
            DSrc::LocalProj { .. } | DSrc::MoveLocalProj { .. } | DSrc::Const(..) => None,
        }
    }

    fn eval_drvalue(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        rv: &DRvalue,
        dest_ty: Option<&Ty>,
    ) -> InterpResult<Value> {
        match rv {
            DRvalue::Use(src) => self.eval_src(frames, body, src),
            DRvalue::BinaryOp(op, lhs, rhs) => {
                let leaf = Self::dsrc_pointee_ty(body, lhs)
                    .or_else(|| Self::dsrc_pointee_ty(body, rhs));
                let lhs = self.eval_src(frames, body, lhs)?;
                let rhs = self.eval_src(frames, body, rhs)?;
                eval_binary(*op, &lhs, &rhs, dest_ty, false, &|v| {
                    self.pointer_address(frames, v, leaf.as_ref())
                })
                .map(|(v, _)| v)
            }
            DRvalue::CheckedBinaryOp(op, lhs, rhs) => {
                let leaf = Self::dsrc_pointee_ty(body, lhs)
                    .or_else(|| Self::dsrc_pointee_ty(body, rhs));
                let lhs = self.eval_src(frames, body, lhs)?;
                let rhs = self.eval_src(frames, body, rhs)?;
                let (v, overflowed) = eval_binary(*op, &lhs, &rhs, dest_ty, true, &|v| {
                    self.pointer_address(frames, v, leaf.as_ref())
                })?;
                Ok(Value::Tuple(vec![v, Value::Bool(overflowed)]))
            }
            DRvalue::UnaryOp(op, src) => {
                let v = self.eval_src(frames, body, src)?;
                eval_unary(*op, &v, dest_ty)
            }
            DRvalue::AggregateTuple(srcs) => {
                let mut items = Vec::with_capacity(srcs.len());
                for src in srcs {
                    items.push(self.eval_src(frames, body, src)?);
                }
                Ok(Value::Tuple(items))
            }
            DRvalue::AggregateAdt {
                variant,
                operands,
                field_index,
                units_from,
            } => {
                let mut fields = match field_index {
                    Some(active) => vec![Value::Uninit; *active as usize],
                    None => Vec::with_capacity(operands.len()),
                };
                for src in operands {
                    fields.push(self.eval_src(frames, body, src)?);
                }
                if let Some((from_size, payload_pos)) = units_from {
                    if let Some(field) = fields.get_mut(*payload_pos) {
                        let taken = std::mem::replace(field, Value::Unit);
                        *field = crate::interp::ptr_cast_units_value(
                            taken,
                            *from_size,
                            &|id| self.alloc_index_is_typed(id),
                        );
                    }
                }
                // The `MaybeUninit::new(v)` normalization the
                // original evaluator performs (see `eval_rvalue`'s
                // AggregateAdt arm): the payload is stored at
                // field 0, not at the declared field index with
                // uninitialized padding before it. This arm must
                // mirror that exactly — a decoded aggregate that
                // padded the payload to field 1 left every reader
                // of the payload (which enters at field 0) looking
                // at an uninitialized slot.
                if *field_index == Some(1) && operands.len() == 1 {
                    let is_maybe_uninit = matches!(
                        dest_ty,
                        Some(Ty::Adt(name))
                            if name.starts_with("std::mem::MaybeUninit")
                                || name.starts_with("core::mem::MaybeUninit")
                    );
                    if is_maybe_uninit {
                        let operand = fields.pop().expect("one operand pushed");
                        let payload = match operand {
                            Value::Enum { fields: inner, .. } if inner.len() == 1 => {
                                inner.into_iter().next().expect("one field")
                            }
                            other => other,
                        };
                        return Ok(Value::Enum {
                            variant: 0,
                            fields: vec![payload],
                        });
                    }
                }
                Ok(Value::Enum {
                    variant: *variant,
                    fields,
                })
            }
            DRvalue::ClosureAggregate { def, operands } => {
                let mut env = Vec::with_capacity(operands.len());
                for src in operands {
                    env.push(self.eval_src(frames, body, src)?);
                }
                Ok(Value::Closure {
                    def: def.clone(),
                    env,
                })
            }
            DRvalue::CoroutineAggregate(srcs) => {
                let mut fields = Vec::with_capacity(srcs.len());
                for src in srcs {
                    fields.push(self.eval_src(frames, body, src)?);
                }
                Ok(Value::Coroutine {
                    variant: 0,
                    fields,
                    saved: Vec::new(),
                })
            }
            DRvalue::Cast(src, ty) => {
                let v = self.eval_src(frames, body, src)?;
                let src_ty = match src {
                    DSrc::Local(local) | DSrc::MoveLocal(local) => {
                        body.local_ty(Local(*local as usize)).cloned()
                    }
                    DSrc::LocalProj { place, .. } | DSrc::MoveLocalProj { place, .. } => {
                        cast_place_ty(body, place)
                    }
                    DSrc::Operand(operand) => cast_src_ty(body, operand),
                    DSrc::Const(_, ty) => ty.clone(),
                    _ => None,
                };
                eval_cast(&v, src_ty.as_ref(), ty)
            }
            DRvalue::Len(dplace, place) => {
                let value = self.read_dplace(frames, body, dplace)?;
                let len = self
                    .fat_deref_len(frames, body, place)
                    .or_else(|| value.len())
                    .or_else(|| self.fat_tail_field_len(frames, body, place));
                let len = len.ok_or_else(|| {
                    InterpError::type_mismatch(
                        format!("Len of {place}"),
                        "array, slice or string value",
                        value.kind_name(),
                    )
                })?;
                Ok(Value::Uint(len))
            }
            DRvalue::Discriminant(dplace, discriminants, place) => {
                let value = self.read_dplace(frames, body, dplace)?;
                match value {
                    Value::Enum { variant, .. } | Value::Coroutine { variant, .. } => {
                        if discriminants.is_empty() {
                            Ok(Value::Int(variant as i128))
                        } else {
                            let discr = discriminants.get(variant as usize).ok_or_else(|| {
                                InterpError::type_mismatch(
                                    format!("Discriminant of {place}"),
                                    "variant covered by the discriminant table",
                                    format!("variant {variant}"),
                                )
                            })?;
                            Ok(Value::Int(*discr))
                        }
                    }
                    _ => Ok(Value::Int(0)),
                }
            }
            DRvalue::Reborrow(dplace) => self.read_dplace(frames, body, dplace),
            DRvalue::PtrToInt(src) => {
                let leaf = Self::dsrc_pointee_ty(body, src);
                let value = self.eval_src(frames, body, src)?;
                match self.pointer_address(frames, &value, leaf.as_ref()) {
                    Some(addr) => {
                        self.note_exposed_target(addr as u64, &value);
                        Ok(Value::Uint(addr))
                    }
                    None => Err(InterpError::TypeMismatch {
                        context: "pointer-to-integer transmute".to_string(),
                        expected: "a pointer value".to_string(),
                        found: value.kind_name().to_string(),
                    }),
                }
            }
            DRvalue::IntToPtr(src) => {
                let value = self.eval_src(frames, body, src)?;
                match value {
                    Value::Int(i) => Ok(self.materialized_ptr(i as u64)),
                    Value::Uint(u) => Ok(self.materialized_ptr(u as u64)),
                    other => Err(InterpError::type_mismatch(
                        "integer-to-pointer transmute",
                        "an integer value",
                        other.kind_name(),
                    )),
                }
            }
            DRvalue::EnumToInt(src, discriminants, ty) => {
                let value = self.eval_src(frames, body, src)?;
                match value {
                    Value::Enum { variant, .. } => {
                        let discr = if discriminants.is_empty() {
                            variant as i128
                        } else {
                            *discriminants.get(variant as usize).ok_or_else(|| {
                                InterpError::type_mismatch(
                                    "enum-to-integer transmute",
                                    "variant covered by the discriminant table",
                                    format!("variant {variant}"),
                                )
                            })?
                        };
                        eval_cast(&Value::Int(discr), None, ty)
                    }
                    other => Err(InterpError::type_mismatch(
                        "enum-to-integer transmute",
                        "an enum value",
                        other.kind_name(),
                    )),
                }
            }
            DRvalue::IntToEnum(src, discriminants) => {
                let value = self.eval_src(frames, body, src)?;
                let bits = match value {
                    Value::Int(i) => i as u128,
                    Value::Uint(u) => u,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "integer-to-enum transmute",
                            "an integer value",
                            other.kind_name(),
                        ))
                    }
                };
                match discriminants.iter().position(|d| *d as u128 == bits) {
                    Some(variant) => Ok(Value::Enum {
                        variant: variant as u32,
                        fields: Vec::new(),
                    }),
                    None => Err(InterpError::type_mismatch(
                        "integer-to-enum transmute",
                        "a discriminant value of the target enum",
                        format!("integer {bits}"),
                    )),
                }
            }
            DRvalue::IntToEnumWrapped(src, discriminants, payload_index, field_count, fill) => {
                let value = self.eval_src(frames, body, src)?;
                let bits = match value {
                    Value::Int(i) => i as u128,
                    Value::Uint(u) => u,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "integer-to-enum-wrapper transmute",
                            "an integer value",
                            other.kind_name(),
                        ))
                    }
                };
                let variant = discriminants
                    .iter()
                    .position(|d| *d as u128 == bits)
                    .ok_or_else(|| {
                        InterpError::type_mismatch(
                            "integer-to-enum-wrapper transmute",
                            "a discriminant value of the wrapped enum",
                            format!("integer {bits}"),
                        )
                    })?;
                let mut fields = vec![Value::Uninit; *field_count];
                fields[*payload_index as usize] = Value::Enum {
                    variant: variant as u32,
                    fields: Vec::new(),
                };
                for (index, constant) in fill.iter() {
                    fields[*index as usize] = self.eval_constant(constant)?;
                }
                Ok(Value::Enum {
                    variant: 0,
                    fields,
                })
            }
            DRvalue::Orig(rv) => self.eval_rvalue(frames, body, rv, dest_ty),
        }
    }
}

/// The sorted `SwitchInt` target table for a block's terminator
/// (see [`DecodedBlock::switch`]).
fn decode_switch(term: &TerminatorKind) -> Option<Box<[(u128, BasicBlock)]>> {
    let TerminatorKind::SwitchInt { targets, .. } = term else {
        return None;
    };
    let mut sorted: Vec<(u128, BasicBlock)> = targets.clone();
    sorted.sort_by_key(|(pattern, _)| *pattern);
    if sorted.windows(2).any(|w| w[0].0 == w[1].0) {
        return None;
    }
    Some(sorted.into_boxed_slice())
}


#[cfg(test)]
mod dispatch_prototype {
    //! Dispatch-form comparison prototype (Lane GL, Issue #97).
    //!
    //! The user's suggestion that motivated this lane was that a
    //! `match`-dispatched loop is inherently slow and a threaded
    //! (indirect) dispatch should replace it. Rust has neither
    //! computed goto nor guaranteed tail calls, so the realistic
    //! "threaded" candidate is a function-pointer table. This
    //! prototype measures exactly that one variable: two executors
    //! run the *same* pre-decoded instruction stream with the
    //! *same* per-instruction work (the same slot reads/writes and
    //! the same real `eval_binary` calls); one dispatches with a
    //! `match` on a dense enum, the other through a table of
    //! function pointers. Run with:
    //!
    //! `cargo test -p sparagmos-interp --release dispatch_prototype -- --ignored --nocapture`
    use super::eval_binary;
    use crate::interp::synthetic_address;
    use crate::ir::BinOp;
    use crate::value::Value;
    use std::time::Instant;

    /// One pre-decoded instruction in the shape both executors
    /// share: add `slots[a]` and `slots[b]` (or the constant)
    /// into `slots[dst]`, in the mix a scalar loop body produces.
    #[derive(Clone, Copy)]
    enum Op {
        AddLL { dst: u32, a: u32, b: u32 },
        AddLC { dst: u32, a: u32, c: u128 },
        Move { dst: u32, a: u32 },
        LtLL { dst: u32, a: u32, b: u32 },
        CheckedAddLL { dst: u32, a: u32, b: u32 },
    }

    /// The same instruction as a tagged record for the
    /// function-pointer executor (tag indexes the handler table).
    #[derive(Clone, Copy)]
    struct Rec {
        tag: usize,
        dst: u32,
        a: u32,
        b: u32,
        c: u128,
    }

    struct Ctx {
        slots: Vec<Value>,
    }

    fn add_ll(ctx: &mut Ctx, dst: u32, a: u32, b: u32) {
        let lhs = ctx.slots[a as usize].clone();
        let rhs = ctx.slots[b as usize].clone();
        let (v, _) = eval_binary(BinOp::Add, &lhs, &rhs, None, false, &synthetic_address).unwrap();
        ctx.slots[dst as usize] = v;
    }
    fn add_lc(ctx: &mut Ctx, dst: u32, a: u32, c: u128) {
        let lhs = ctx.slots[a as usize].clone();
        let rhs = Value::Uint(c);
        let (v, _) = eval_binary(BinOp::Add, &lhs, &rhs, None, false, &synthetic_address).unwrap();
        ctx.slots[dst as usize] = v;
    }
    fn mov(ctx: &mut Ctx, dst: u32, a: u32) {
        let v = ctx.slots[a as usize].clone();
        ctx.slots[a as usize] = Value::Uninit;
        ctx.slots[dst as usize] = v;
    }
    fn lt_ll(ctx: &mut Ctx, dst: u32, a: u32, b: u32) {
        let lhs = ctx.slots[a as usize].clone();
        let rhs = ctx.slots[b as usize].clone();
        let (v, _) = eval_binary(BinOp::Lt, &lhs, &rhs, None, false, &synthetic_address).unwrap();
        ctx.slots[dst as usize] = v;
    }
    fn checked_add_ll(ctx: &mut Ctx, dst: u32, a: u32, b: u32) {
        let lhs = ctx.slots[a as usize].clone();
        let rhs = ctx.slots[b as usize].clone();
        let (v, overflowed) = eval_binary(BinOp::Add, &lhs, &rhs, None, true, &synthetic_address).unwrap();
        ctx.slots[dst as usize] = Value::Tuple(vec![v, Value::Bool(overflowed)]);
    }

    fn exec_match(ctx: &mut Ctx, stream: &[Op]) {
        for op in stream {
            match op {
                Op::AddLL { dst, a, b } => add_ll(ctx, *dst, *a, *b),
                Op::AddLC { dst, a, c } => add_lc(ctx, *dst, *a, *c),
                Op::Move { dst, a } => mov(ctx, *dst, *a),
                Op::LtLL { dst, a, b } => lt_ll(ctx, *dst, *a, *b),
                Op::CheckedAddLL { dst, a, b } => checked_add_ll(ctx, *dst, *a, *b),
            }
        }
    }

    type Handler = fn(&mut Ctx, &Rec);
    fn h_add_ll(ctx: &mut Ctx, r: &Rec) {
        add_ll(ctx, r.dst, r.a, r.b)
    }
    fn h_add_lc(ctx: &mut Ctx, r: &Rec) {
        add_lc(ctx, r.dst, r.a, r.c)
    }
    fn h_mov(ctx: &mut Ctx, r: &Rec) {
        mov(ctx, r.dst, r.a)
    }
    fn h_lt_ll(ctx: &mut Ctx, r: &Rec) {
        lt_ll(ctx, r.dst, r.a, r.b)
    }
    fn h_checked_add_ll(ctx: &mut Ctx, r: &Rec) {
        checked_add_ll(ctx, r.dst, r.a, r.b)
    }
    const HANDLERS: &[Handler] = &[
        h_add_ll,
        h_add_lc,
        h_mov,
        h_lt_ll,
        h_checked_add_ll,
    ];

    fn exec_fnptr(ctx: &mut Ctx, stream: &[Rec]) {
        for rec in stream {
            HANDLERS[rec.tag](ctx, rec);
        }
    }

    fn build_stream(iters: usize) -> (Vec<Op>, Vec<Rec>) {
        // One loop-body iteration in slot terms: tmp = i + 1;
        // i = tmp (move); acc = acc + i; flag = i < n;
        // chk = acc +checked i. Slots: 0 acc, 1 i, 2 n, 3 tmp,
        // 4 flag, 5 chk.
        let body_ops = [
            Op::AddLC { dst: 3, a: 1, c: 1 },
            Op::Move { dst: 1, a: 3 },
            Op::AddLL { dst: 0, a: 0, b: 1 },
            Op::LtLL { dst: 4, a: 1, b: 2 },
            Op::CheckedAddLL { dst: 5, a: 0, b: 1 },
        ];
        let body_recs = [
            Rec { tag: 1, dst: 3, a: 1, b: 0, c: 1 },
            Rec { tag: 2, dst: 1, a: 3, b: 0, c: 0 },
            Rec { tag: 0, dst: 0, a: 0, b: 1, c: 0 },
            Rec { tag: 3, dst: 4, a: 1, b: 2, c: 0 },
            Rec { tag: 4, dst: 5, a: 0, b: 1, c: 0 },
        ];
        let mut ops = Vec::with_capacity(iters * body_ops.len());
        let mut recs = Vec::with_capacity(iters * body_recs.len());
        for _ in 0..iters {
            ops.extend_from_slice(&body_ops);
            recs.extend_from_slice(&body_recs);
        }
        (ops, recs)
    }

    fn fresh_ctx() -> Ctx {
        Ctx {
            slots: vec![
                Value::Uint(0),
                Value::Uint(0),
                Value::Uint(u128::MAX),
                Value::Uninit,
                Value::Bool(false),
                Value::Uninit,
            ],
        }
    }

    #[test]
    #[ignore = "timing prototype; run explicitly with --release --nocapture"]
    fn dispatch_prototype() {
        let iters = 2_000_000usize;
        let (ops, recs) = build_stream(iters);
        let total = (iters * 5) as f64;
        let mut best = [f64::INFINITY; 2];
        for round in 0..5 {
            let mut ctx = fresh_ctx();
            let t = Instant::now();
            exec_match(&mut ctx, &ops);
            let dt = t.elapsed().as_secs_f64();
            best[0] = best[0].min(dt);
            // Keep the result observable so nothing is elided.
            assert!(matches!(ctx.slots[0], Value::Uint(_)));

            let mut ctx = fresh_ctx();
            let t = Instant::now();
            exec_fnptr(&mut ctx, &recs);
            let dt = t.elapsed().as_secs_f64();
            best[1] = best[1].min(dt);
            assert!(matches!(ctx.slots[0], Value::Uint(_)));
            println!(
                "round {round}: match {:.1} ns/op, fnptr {:.1} ns/op",
                best[0] * 1e9 / total,
                best[1] * 1e9 / total
            );
        }
        println!(
            "best of 5: match {:.2} ns/op, fnptr {:.2} ns/op ({} ops/run)",
            best[0] * 1e9 / total,
            best[1] * 1e9 / total,
            total as u64
        );
    }
}
