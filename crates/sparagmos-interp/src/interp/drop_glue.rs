//! Drop glue execution.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{RefProj, RefTarget, Value};

/// Drop glue suspended while a destructor body runs. A `Drop`
/// terminator whose glue calls a destructor pushes the destructor's
/// frame and one of these; when that frame returns, the interpreter
/// resumes the remaining glue work instead of delivering a return
/// value to a caller place.
pub(crate) struct DropContinuation {
    /// Frame id of the destructor call this work waits on.
    pub(crate) callee_frame: u64,
    /// The glue session the remaining work belongs to.
    pub(crate) session: GlueSession,
    /// Glue sessions suspended below `session` (dyn glue runs the
    /// concrete type's glue as a nested session; see
    /// [`GlueCompletion::Resume`]), innermost last.
    pub(crate) sessions: Vec<GlueSession>,
}

/// The outcome of running a suspended glue session as an
/// unwinding panic's cleanup.
pub(crate) enum GlueCleanupFlow {
    /// The session ran to completion synchronously; unwinding
    /// continues with the returned panic.
    Completed(InterpError),
    /// The session suspended on a field destructor's frame:
    /// executing that frame resumes the session, whose
    /// completion re-raises the stashed panic into the unwinder.
    Suspended,
}

/// One run of drop glue: a work stack of (path from `root`, plan)
/// items against one dropped place, with the ADT glue table the
/// plans index into.
pub(crate) struct GlueSession {
    /// The dropped place the work applies to.
    pub(crate) root: RefTarget,
    /// Whether the session's root plan is the glue of a box the
    /// root place itself owns (a `BoxPointee` plan, possibly
    /// under transparent field-0 wrappers such as `Pin<Box<T>>`
    /// or an ADT table entry): only such a session releases the
    /// root value's allocation when it finishes (see the release
    /// at the end of `run_glue_session`). A root plan of any
    /// other shape -- a `Pin<&mut T>` place, say, whose value is
    /// the same one-field aggregate around a base reference --
    /// owns no allocation, however box-shaped its value reads.
    pub(crate) owns_root_alloc: bool,
    /// Remaining work items, a stack: the last item runs next.
    pub(crate) work: Vec<(Vec<RefProj>, crate::ir::DropPlan)>,
    /// ADT glue table of the originating `Drop` terminator.
    pub(crate) adts: Vec<crate::ir::DropPlan>,
    /// What happens when the work runs out.
    pub(crate) completion: GlueCompletion,
}

/// How a glue session ends when its work runs out.
pub(crate) enum GlueCompletion {
    /// The session is a whole `Drop` terminator (or `mem::drop`
    /// call): the dropped place's own slot is cleared and the owner
    /// frame jumps to this block.
    Jump(BasicBlock),
    /// The session is the nested glue of a `dyn` value's concrete
    /// type: the session below it on the stack resumes.
    Resume,
    /// The session runs as the cleanup of a panic unwinding out of
    /// its destructor call: when the work runs out, the stashed
    /// panic (in `Interpreter::glue_unwind_panic`) continues
    /// unwinding instead of control returning to the owner frame.
    UnwindCleanup,
    /// The session runs synchronously inside a builtin (see
    /// `Interpreter::drop_value_sync`): there is no owner frame
    /// to jump to, so when the work runs out the session simply
    /// ends, and a destructor call runs to completion on the
    /// spot (through `call_to_completion`) instead of
    /// suspending the session.
    Sync,
}

/// Whether `plan`, applied at a glue session's root, is the
/// glue of a box the root place owns: a `BoxPointee` plan,
/// reached directly, through a `DynOrPlain`'s plain form,
/// through transparent field-0 wrappers (the `At` nodes
/// conversion builds for `Pin` and other one-field newtypes),
/// through an aligned-temporary or elided-element wrapper,
/// through one arm of a sequence, or through an ADT table
/// entry. Anything else owns no allocation at the root.
pub(crate) fn plan_owns_box(
    plan: &crate::ir::DropPlan,
    adts: &[crate::ir::DropPlan],
) -> bool {
    use crate::ir::DropPlan as P;
    fn inner(
        plan: &crate::ir::DropPlan,
        adts: &[crate::ir::DropPlan],
        depth: usize,
    ) -> bool {
        if depth > 32 {
            return false;
        }
        match plan {
            P::BoxPointee { .. } => true,
            P::DynOrPlain { plain } => inner(plain, adts, depth + 1),
            P::At { path, plan } => {
                path.iter()
                    .all(|step| matches!(step, RefProj::Field(0)))
                    && inner(plan, adts, depth + 1)
            }
            P::AlignedTemp { plan } => inner(plan, adts, depth + 1),
            P::Elided { plan, .. } => inner(plan, adts, depth + 1),
            P::Seq(items) => items.iter().any(|item| inner(item, adts, depth + 1)),
            P::Adt { index } => adts
                .get(*index)
                .is_some_and(|entry| inner(entry, adts, depth + 1)),
            _ => false,
        }
    }
    inner(plan, adts, 0)
}

/// Whether `value` is shaped like a lock guard the converted
/// lock bodies built: an aggregate whose first field is a
/// reference -- to the lock for `MutexGuard` /
/// `RwLockWriteGuard`, or, for `RwLockReadGuard` and the
/// mapped guards, the `NonNull` data pointer (a one-field
/// aggregate around a reference).
fn lock_guard_shaped(value: &Value) -> bool {
    fn ref_like(value: &Value) -> bool {
        match value {
            Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. } => true,
            Value::Enum { fields, .. } if fields.len() == 1 => ref_like(&fields[0]),
            _ => false,
        }
    }
    match value {
        Value::Enum { fields, .. } => fields.first().is_some_and(ref_like),
        _ => false,
    }
}

impl<'a> Interpreter<'a> {
    /// Read the value at `root` + `path` for drop glue: `Ok(None)`
    /// when any part of the way is uninitialized (a moved-out or
    /// never-written part has no glue to run, matching rustc's drop
    /// flags); other read failures are errors.
    fn read_at(
        &self,
        frames: &[Frame],
        root: &RefTarget,
        path: &[RefProj],
    ) -> InterpResult<Option<Value>> {
        let Some(value) = self.read_at_raw(frames, root, path)? else {
            return Ok(None);
        };
        // A path that ends on a box's field 0 denotes the pointee
        // (drop plans address a box's contents through its field
        // 0), and the value stored there is the box's allocation
        // pointer: follow it to the pointee itself.
        self.follow_alloc(value)
    }

    /// Follow a value that is a box's allocation pointer to the
    /// pointee it denotes; any other value passes through. A
    /// freed allocation reads as nothing (no drop glue is left
    /// to read in it), as a null pointer does in `read_at`.
    fn follow_alloc(&self, value: Value) -> InterpResult<Option<Value>> {
        match value {
            Value::Ref(RefTarget::Alloc { id, projection }) => {
                // The pointee was already dropped through a
                // pointer (see `glue_dropped_allocs`): nothing
                // is left to drop in it.
                if self.glue_dropped_allocs.borrow().contains(&id) {
                    return Ok(None);
                }
                let Ok(base) = self.alloc_base(id) else {
                    return Ok(None);
                };
                if base == Value::Uninit {
                    return Ok(None);
                }
                match self.project_value_t(
                    base,
                    &projection,
                    "<drop>",
                    "drop glue",
                    self.alloc_ty(id),
                ) {
                    Ok(v) if v == Value::Uninit => Ok(None),
                    Ok(v) => Ok(Some(v)),
                    Err(InterpError::UninitializedRead { .. }) => Ok(None),
                    Err(e) => Err(e),
                }
            }
            other => Ok(Some(other)),
        }
    }

    /// The value at `root` + `path` for drop glue, without
    /// following a box's allocation pointer the path ends on
    /// (see `read_at`): a path that crosses a box's field 0
    /// mid-way still applies its remaining steps inside the
    /// allocation, because the crossed step denotes the pointee,
    /// not the pointer slot.
    fn read_at_raw(
        &self,
        frames: &[Frame],
        root: &RefTarget,
        path: &[RefProj],
    ) -> InterpResult<Option<Value>> {
        // An allocation an earlier completed session already
        // dropped through a pointer holds nothing left to drop
        // (see `glue_dropped_allocs`): glue reaching it again,
        // at any projection, finds the storage empty, exactly
        // as the uninitialised storage the session-completion
        // marking used to leave behind read. The entry is only
        // ever recorded as a session completes, so a session's
        // own reads of its root are never affected.
        if let RefTarget::Alloc { id, .. } = root {
            if self.glue_dropped_allocs.borrow().contains(id) {
                return Ok(None);
            }
        }
        // The declared type of the root value, stepped alongside
        // the walk below as `read_place`'s reads step theirs: the
        // projectors' shape-only judgments (the single-array-field
        // pun above all) misread a one-element `Vec` of tuples
        // without it, and drop glue reads the same places the
        // ordinary reads do.
        let (base, prefix, root_ty): (Value, &[RefProj], Option<Ty>) = match root {
            RefTarget::Place {
                frame,
                local,
                projection,
            } => {
                let frame_pos = self.frame_pos(frames, *frame)?;
                let frame = &frames[frame_pos];
                let base = frame.locals.get(*local).cloned().ok_or_else(|| {
                    InterpError::UnknownLocal {
                        func: frame.func.clone(),
                        local: *local,
                    }
                })?;
                // A never-written place of an evidently zero-sized
                // type still denotes a value for drop glue: optimized
                // MIR deletes the (no-op) write of a ZST aggregate,
                // yet the place's destructor must run — it can have
                // observable effects, such as writing a static.
                //
                // A single-valued type determines that value exactly
                // (conversion records it in `zst_locals`), so read
                // through it: the local holds the recorded value
                // whenever it is live, and its full glue — field
                // destructors included — runs against it, as it
                // would against a stored value. This holds for a
                // root that *is* the local and for a root addressing
                // a sub-place of it alike (a sub-place's read has no
                // other way through: a fabricated ADT has no fields
                // to project through, so the read would report
                // nothing and the sub-place's destructor would be
                // skipped even though MIR emitted its drop).
                //
                // A type with many values keeps the synthesized
                // gate: a fieldless fabrication under which the
                // ADT's own destructor still runs, while field glue
                // beneath it reads as never-written and skips,
                // exactly as drop flags dictate.
                let base = if base == Value::Uninit {
                    self.zst_local_value(frame, *local)
                        .or_else(|| self.zst_glue_value(frame, *local))
                        .unwrap_or(base)
                } else {
                    base
                };
                let root_ty = self
                    .program()
                    .get(&frame.func)
                    .and_then(|body| body.local_ty(Local(*local)))
                    .cloned();
                (base, projection, root_ty)
            }
            RefTarget::Value(v, projection) => ((**v).clone(), projection, None),
            RefTarget::Static { name, projection } => {
                let root_ty = self.program().static_tys.get(name).cloned();
                (self.static_base(name)?, projection, root_ty)
            }
            RefTarget::Alloc { id, projection } => {
                // A freed allocation holds nothing to read glue
                // from, as with a null pointer below.
                let Ok(base) = self.alloc_base(*id) else {
                    return Ok(None);
                };
                (base, projection, self.alloc_ty(*id))
            }
            // A null pointer denotes nothing; there is no drop
            // glue to read through it.
            RefTarget::Null => return Ok(None),
            // An address-only pointer denotes nothing readable;
            // there is no drop glue to read through it.
            RefTarget::Address(_) => return Ok(None),
        };
        let mut value = base;
        let mut cur_ty = root_ty;
        let steps: Vec<RefProj> = prefix.iter().chain(path.iter()).cloned().collect();
        let mut step_idx = 0;
        while step_idx < steps.len() {
            if value == Value::Uninit {
                return Ok(None);
            }
            // A previous step crossed a box's field 0: the value
            // here is the box's allocation pointer and this step
            // applies inside the allocation. The evidence follows
            // the crossing: the followed value's type is the
            // allocation's recorded type stepped across the
            // pointer's own projection.
            let followed_ty = match &value {
                Value::Ref(RefTarget::Alloc { id, projection }) => Some(
                    step_ty_across(Some(self.program()), self.alloc_ty(*id), projection),
                ),
                _ => None,
            };
            match self.follow_alloc(value)? {
                Some(v) => value = v,
                None => return Ok(None),
            }
            if let Some(followed) = followed_ty {
                cur_ty = followed;
            }
            // A coroutine downcast only names the state whose saved
            // storage the following field step addresses, so the two
            // steps are projected together: projected on its own,
            // the field step would read the upvar storage instead.
            let take = if matches!(steps[step_idx], RefProj::Downcast(_))
                && matches!(value, Value::Coroutine { .. })
                && matches!(steps.get(step_idx + 1), Some(RefProj::Field(_)))
            {
                2
            } else {
                1
            };
            match self.project_value_t(
                value,
                &steps[step_idx..step_idx + take],
                "<drop>",
                "drop glue",
                cur_ty.clone(),
            ) {
                Ok(v) => value = v,
                Err(InterpError::UninitializedRead { .. }) => return Ok(None),
                Err(e) => return Err(e),
            }
            cur_ty = step_ty_across(
                Some(self.program()),
                cur_ty,
                &steps[step_idx..step_idx + take],
            );
            step_idx += take;
        }
        if value == Value::Uninit {
            Ok(None)
        } else {
            Ok(Some(value))
        }
    }

    /// A reference to `root` + `path`, for a destructor call.
    /// The path is walked one step at a time: a step that lands
    /// on a box's allocation pointer crosses into the allocation
    /// (the step denotes the pointee, as in `read_at`), so the
    /// destructor's `&mut self` names the pointee's storage, not
    /// the pointer slot inside a copy of the box. A step whose
    /// landing value cannot be read (a never-written place the
    /// glue gate let through) keeps the symbolic projection, as
    /// before.
    fn extend_ref(
        &self,
        frames: &[Frame],
        root: &RefTarget,
        path: &[RefProj],
    ) -> InterpResult<RefTarget> {
        let mut cur = root.clone();
        for step in path {
            let landed = self
                .read_at_raw(frames, &cur, std::slice::from_ref(step))
                .ok()
                .flatten();
            if let Some(Value::Ref(RefTarget::Alloc { id, projection })) = landed {
                cur = RefTarget::Alloc { id, projection };
                continue;
            }
            cur = match cur {
                RefTarget::Place {
                    frame,
                    local,
                    mut projection,
                } => {
                    projection.push(step.clone());
                    RefTarget::place(frame, local, projection)
                }
                RefTarget::Value(v, projection) => {
                    let mut projection = projection.clone();
                    projection.push(step.clone());
                    RefTarget::Value(v.clone(), projection)
                }
                RefTarget::Static {
                    name,
                    mut projection,
                } => {
                    projection.push(step.clone());
                    RefTarget::Static { name, projection }
                }
                RefTarget::Alloc {
                    id,
                    mut projection,
                } => {
                    projection.push(step.clone());
                    RefTarget::Alloc { id, projection }
                }
                RefTarget::Null => {
                    return Err(InterpError::type_mismatch(
                        "drop glue reference",
                        "a valid reference",
                        "null pointer",
                    ))
                }
                RefTarget::Address(_) => {
                    return Err(InterpError::type_mismatch(
                        "drop glue reference",
                        "a valid reference",
                        "address-only pointer",
                    ))
                }
            };
        }
        Ok(cur)
    }

    /// The (data pointer, drop callable) a genuine `RawWaker`
    /// aggregate value carries: field 0 is the data pointer the
    /// vtable functions act on, field 1 a reference to the
    /// vtable value, whose four function pointers are (clone,
    /// wake, wake_by_ref, drop). The drop entry is returned as
    /// the value it was stored as: a plain function item is a
    /// `FnPtr`, while a non-capturing closure literal in the
    /// vtable constant stays a `Closure` value (its coercion to
    /// a function pointer is a type-level fact the value model
    /// does not rewrite). `None` for any other shape.
    fn raw_waker_drop_fn(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<Option<(Value, Value)>> {
        let Value::Enum { fields, .. } = value else {
            return Ok(None);
        };
        let [data, vtable] = fields.as_slice() else {
            return Ok(None);
        };
        if !matches!(data, Value::Ref(_)) || !matches!(vtable, Value::Ref(_)) {
            return Ok(None);
        }
        let resolved = self.deref_value(frames, vtable)?;
        let table = self.read_resolved(frames, &resolved)?;
        let Value::Enum { fields, .. } = &table else {
            return Ok(None);
        };
        if fields.len() != 4 {
            return Ok(None);
        }
        if !matches!(fields[3], Value::FnPtr(_) | Value::Closure { .. }) {
            return Ok(None);
        }
        Ok(Some((data.clone(), fields[3].clone())))
    }

    /// Execute drop glue: process `work` until it is empty or a
    /// destructor call has to run. A destructor call pushes its
    /// frame plus a [`DropContinuation`] and returns; the `Return`
    /// terminator resumes the remaining work when the destructor
    /// finishes. When the work is done, the dropped place's own slot
    /// is cleared (for a direct local of the owner frame) and the
    /// owner frame jumps to `target`.
    /// Run a suspended drop-glue session as the cleanup of the
    /// panic unwinding out of its destructor call: the remaining
    /// glue (the fields the destructor itself did not reach) is
    /// the cleanup a native `drop_in_place` performs when the
    /// destructor panics, and the original panic then continues
    /// unwinding.
    pub(crate) fn run_glue_cleanup(
        &mut self,
        frames: &mut Vec<Frame>,
        cont: DropContinuation,
        panic: InterpError,
    ) -> InterpResult<GlueCleanupFlow> {
        let InterpError::Panic { message, payload } = panic else {
            return Ok(GlueCleanupFlow::Completed(panic));
        };
        self.glue_unwind_panic = Some((message, payload));
        let mut session = cont.session;
        session.completion = GlueCompletion::UnwindCleanup;
        match self.run_glue_session(frames, session, cont.sessions) {
            Ok(()) => Ok(GlueCleanupFlow::Suspended),
            Err(e @ InterpError::Panic { .. }) => {
                // The session's completion marked the stashed
                // panic as resumed (`resume_signaled`), but that
                // flag's only reader is the drive loop's take
                // when a panic surfaces from a block -- the
                // unwinding loop that receives the panic here
                // owns its continuation and never reads it.
                // Left set, it would leak into the next drive
                // region and suppress a fresh panic's hook
                // there (a panic raised by a later detached
                // drive, such as the entry payload disposal,
                // would print no report).
                self.resume_signaled = false;
                Ok(GlueCleanupFlow::Completed(e))
            }
            Err(e) => Err(e),
        }
    }

    pub(crate) fn run_drop_glue(
        &mut self,
        frames: &mut Vec<Frame>,
        root: RefTarget,
        work: Vec<(Vec<RefProj>, crate::ir::DropPlan)>,
        adts: &[crate::ir::DropPlan],
        target: BasicBlock,
    ) -> InterpResult<()> {
        let session = GlueSession {
            owns_root_alloc: work.iter().any(|(path, plan)| {
                path.is_empty() && plan_owns_box(plan, adts)
            }),
            root,
            work,
            adts: adts.to_vec(),
            completion: GlueCompletion::Jump(target),
        };
        self.run_glue_session(frames, session, Vec::new())
    }

    /// Run the drop glue of a detached value to completion,
    /// synchronously: a builtin that consumes a value the guest
    /// would have dropped (an item `Iterator::nth` skips, say)
    /// owes the value's destructor before it steps on. The
    /// value is parked in the allocation store first, as for a
    /// consumed iterator's final state: a destructor body
    /// writes back into its receiver, which a bare value root
    /// cannot accept. The session's `Sync` completion runs
    /// each destructor call to completion on the spot; a
    /// destructor that panics propagates as the builtin's own
    /// panic, exactly as a panicking destructor propagates
    /// out of the guest operation that consumed the value.
    pub(crate) fn drop_value_sync(
        &mut self,
        frames: &mut Vec<Frame>,
        value: Value,
        glue: &crate::ir::DropGlue,
    ) -> InterpResult<()> {
        let id = self.alloc_insert(value);
        let root = RefTarget::Alloc {
            id,
            projection: Vec::new(),
        };
        let session = GlueSession {
            owns_root_alloc: plan_owns_box(&glue.root, &glue.adts),
            root,
            work: vec![(Vec::new(), glue.root.clone())],
            adts: glue.adts.clone(),
            completion: GlueCompletion::Sync,
        };
        self.run_glue_session(frames, session, Vec::new())
    }

    /// Run one glue session (and, below it, any sessions suspended
    /// under it) until the outermost session completes or a
    /// destructor call suspends the run.
    pub(crate) fn run_glue_session(
        &mut self,
        frames: &mut Vec<Frame>,
        mut active: GlueSession,
        mut suspended: Vec<GlueSession>,
    ) -> InterpResult<()> {
        use crate::ir::DropPlan as P;
        // A session begun as `Sync` (a builtin consuming a
        // value) runs its destructor calls to completion
        // instead of suspending: the flag is captured here, at
        // the outermost session, and covers the nested
        // sessions this run stacks below it.
        let sync_dtors = matches!(active.completion, GlueCompletion::Sync);
        'sessions: loop {
            while let Some((path, plan)) = active.work.pop() {
                self.tick()?;
                match plan {
                    P::Trivial => {}
                    P::Untyped => {
                        // The type of this part is unrecorded, so judge
                        // by the value's own shape: kinds that never
                        // carry glue pass, tuples recurse (a tuple has
                        // no destructor of its own), and an ADT value
                        // might hide a destructor — an explicit error.
                        match self.read_at(frames, &active.root, &path)? {
                            None => {}
                            Some(Value::Tuple(items)) => {
                                for i in (0..items.len()).rev() {
                                    let mut sub = path.clone();
                                    sub.push(RefProj::Index(i));
                                    active.work.push((sub, P::Untyped));
                                }
                            }
                            // An enum with no fields has no contents
                            // whose glue could hide here (its own
                            // destructor, if any, is a separate plan
                            // node): nothing to judge, like an empty
                            // tuple. Typed zero-sized constants make
                            // this shape common — a fieldless variant
                            // written by an aggregate and never
                            // projected lands here with an untyped plan.
                            Some(Value::Enum { fields, .. }) if fields.is_empty() => {}
                            // A one-field aggregate is a wrapper
                            // whose only content is its field (a
                            // `PhantomData` newtype, a cell): judge
                            // the field by the same untyped plan,
                            // exactly as a tuple's elements are
                            // judged. Only a structural payload
                            // (an enum or a tuple) is entered; a
                            // scalar payload stays under the
                            // conservative error below.
                            Some(Value::Enum { fields, .. })
                                if fields.len() == 1
                                    && matches!(
                                        fields[0],
                                        Value::Enum { .. } | Value::Tuple(_)
                                    ) =>
                            {
                                let mut sub = path.clone();
                                sub.push(RefProj::Field(0));
                                active.work.push((sub, P::Untyped));
                            }
                            Some(Value::Enum { .. })
                            | Some(Value::SparseVec { .. })
                            | Some(Value::Coroutine { .. }) => {
                                    return Err(InterpError::UnsupportedTerminator {
                                    kind: "Drop".to_string(),
                                });
                            }
                            Some(_) => {}
                        }
                    }
                    P::Unknown => {
                        return Err(InterpError::UnsupportedTerminator {
                            kind: "Drop".to_string(),
                        });
                    }
                    P::Seq(plans) => {
                        for plan in plans.into_iter().rev() {
                            active.work.push((path.clone(), plan));
                        }
                    }
                    P::At { path: extra, plan } => {
                        let mut full = path;
                        full.extend(extra);
                        active.work.push((full, *plan));
                    }
                    P::Adt { index } => {
                        let plan = active.adts.get(index).cloned().ok_or_else(|| {
                            InterpError::type_mismatch(
                                "drop glue",
                                "an ADT plan in the glue table",
                                "a glue index out of range",
                            )
                        })?;
                        active.work.push((path, plan));
                    }
                    P::Fields { fields, complete } => {
                        let Some(value) = self.read_at(frames, &active.root, &path)? else {
                            continue;
                        };
                        let is_coroutine = matches!(value, Value::Coroutine { .. });
                        // For a coroutine in a saved state, the plan
                        // (selected by the `Variants` arm for exactly
                        // this state) covers that state's saved
                        // storage, and the field paths below must
                        // route through its downcast; in the
                        // unresumed state it covers the upvar
                        // storage the bare field paths address.
                        let mut coroutine_state: Option<u32> = None;
                        let items = match value {
                            Value::Tuple(items) => items,
                            Value::Enum { fields, .. } => fields,
                            // A closure's fields are its captures.
                            Value::Closure { env, .. } => env,
                            Value::Coroutine {
                                variant,
                                fields,
                                saved,
                            } => {
                                // A path addressed through a
                                // downcast to the unresumed state
                                // names the upvar storage explicitly
                                // (the drop of a suspended coroutine
                                // schedules its captures' glue under
                                // that address; see the `Variants`
                                // arm); any other path is read in
                                // the value's own state.
                                let effective = match path.last() {
                                    Some(RefProj::Downcast(0)) => 0,
                                    _ => variant,
                                };
                                coroutine_state = Some(effective);
                                if effective == 0 {
                                    fields
                                } else {
                                    saved
                                        .get(effective as usize)
                                        .cloned()
                                        .unwrap_or_default()
                                }
                            }
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "drop glue",
                                    "a tuple or ADT value",
                                    other.kind_name(),
                                ));
                            }
                        };
                        // A coroutine judged through a downcast to
                        // the unresumed state is walked by its upvar
                        // plan, but a slot-mapped coroutine's flat
                        // field vector also holds the later states'
                        // cells, written through bare slot fields
                        // while it ran: those cells belong to the
                        // current state's plan, not the upvars', so
                        // a longer vector is not a mismatch — the
                        // walk below only visits the plan's own
                        // indices.
                        let upvar_walk = is_coroutine
                            && matches!(path.last(), Some(RefProj::Downcast(0)));
                        if items.len() != fields.len()
                            && !(is_coroutine && items.len() < fields.len())
                            && !(upvar_walk && items.len() > fields.len())
                        {
                            // With `complete`, conversion proved the
                            // field count, so a mismatch means the value
                            // is malformed; without it, the value holds
                            // fields whose glue conversion could not
                            // account for. Either way the drop cannot be
                            // carried out faithfully: explicit, never
                            // silent.
                            let _ = complete;
                            return Err(InterpError::UnsupportedTerminator {
                                kind: "Drop".to_string(),
                            });
                        }
                        for (i, field_plan) in fields.into_iter().enumerate().rev() {
                            if i >= items.len() {
                                // Only reachable for a coroutine value
                                // whose storage is shorter than the plan:
                                // the field was never written, so it
                                // holds nothing to drop.
                                continue;
                            }
                            if field_plan != P::Trivial {
                                let mut sub = path.clone();
                                if let Some(state) = coroutine_state {
                                    if state != 0 {
                                        sub.push(RefProj::Downcast(state));
                                    }
                                }
                                sub.push(RefProj::Field(i));
                                active.work.push((sub, field_plan));
                            }
                        }
                    }
                    P::Each { elem } => {
                        let Some(value) = self.read_at(frames, &active.root, &path)? else {
                            continue;
                        };
                        if let Value::SparseVec { elems, len, .. } = &value {
                            // The declared elements drop one by
                            // one, as a materialised vector's do.
                            // The stored tail past the declared
                            // length is spare capacity: its slots
                            // hold values that were already moved
                            // out or un-declared (a `pop`'s
                            // element, a `set_len` shrink's tail),
                            // which must not be dropped a second
                            // time. The unmaterialised tail past
                            // the stored elements is all
                            // uninitialised slots, whose glue is a
                            // no-op under the read rules (a plan
                            // applied at a never-written slot finds
                            // nothing to run), so one representative
                            // tail index judges the plan for the
                            // whole tail instead of walking it slot
                            // by slot.
                            let declared =
                                usize::try_from(*len).unwrap_or(usize::MAX);
                            for i in (0..elems.len().min(declared)).rev() {
                                let mut sub = path.clone();
                                sub.push(RefProj::Field(i));
                                active.work.push((sub, (*elem).clone()));
                            }
                            if *len > elems.len() as u128
                                && !matches!(&*elem, P::Trivial)
                            {
                                // The representative only judges
                                // the plan for the tail: the slots
                                // past the stored prefix were never
                                // stored at all, so an elided-store
                                // rescue must not fire for them —
                                // judge with the element plan itself.
                                let judge = match &*elem {
                                    P::Elided { plan, .. } => (**plan).clone(),
                                    e => e.clone(),
                                };
                                let mut sub = path.clone();
                                sub.push(RefProj::Field(elems.len()));
                                active.work.push((sub, judge));
                            }
                            continue;
                        }
                        // A `Vec` value in its real aggregate form
                        // ([RawVec, len]): the elements live behind
                        // the buffer pointer, not in the aggregate's
                        // fields, so enumerating the fields would run
                        // the element glue against the buffer
                        // descriptor and the length slot. Translate
                        // the value to the element form and run this
                        // plan against the translation in a nested
                        // session, as the elided-element arm runs a
                        // plan against a standalone value. The
                        // translation of a zero-length vector never
                        // touches its buffer pointer (a dangling
                        // address-only pointer an allocator model
                        // handed out has no pointee to read). A value
                        // whose buffer the model cannot read back
                        // declines the translation and keeps the
                        // enumeration below.
                        if let Some(translated) =
                            self.vec_from_real_form(frames, &value)
                        {
                            let nested = GlueSession {
                                owns_root_alloc: false,
                                root: RefTarget::value(translated?),
                                work: vec![(
                                    Vec::new(),
                                    P::Each { elem: elem.clone() },
                                )],
                                adts: active.adts.clone(),
                                completion: GlueCompletion::Resume,
                            };
                            suspended.push(std::mem::replace(&mut active, nested));
                            continue 'sessions;
                        }
                        let (count, is_enum) = match &value {
                            Value::Tuple(items) => (items.len(), false),
                            Value::Enum { fields, .. } => (fields.len(), true),
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "drop glue",
                                    "an array, tuple or Vec value",
                                    other.kind_name(),
                                ));
                            }
                        };
                        for i in (0..count).rev() {
                            let mut sub = path.clone();
                            sub.push(if is_enum {
                                RefProj::Field(i)
                            } else {
                                RefProj::Index(i)
                            });
                            active.work.push((sub, (*elem).clone()));
                        }
                    }
                    P::Elided { value, plan } => {
                        match self.read_at(frames, &active.root, &path)? {
                            Some(_) => {
                                active.work.push((path, *plan));
                            }
                            None => {
                                // The slot reads uninitialised, but
                                // it is a live sequence element whose
                                // type has this one value: its store
                                // was elided, so the element is
                                // `value` itself. Run the plan
                                // against it in a nested session
                                // rooted at the value, the way a
                                // `mem::drop` of a constant argument
                                // runs its glue.
                                let nested = GlueSession {
                                    owns_root_alloc: plan_owns_box(&*plan, &active.adts),
                                    root: RefTarget::value(value),
                                    work: vec![(Vec::new(), *plan)],
                                    adts: active.adts.clone(),
                                    completion: GlueCompletion::Resume,
                                };
                                suspended.push(std::mem::replace(&mut active, nested));
                                continue 'sessions;
                            }
                        }
                    }
                    P::Variants { variants } => {
                        let Some(value) = self.read_at(frames, &active.root, &path)? else {
                            continue;
                        };
                        let (variant, stored_fields) = match &value {
                            Value::Enum { variant, .. } => (*variant, None),
                            Value::Coroutine {
                                variant,
                                fields,
                                saved,
                            } => {
                                let stored = if *variant == 0 {
                                    fields.len()
                                } else {
                                    saved.get(*variant as usize).map_or(0, Vec::len)
                                };
                                (*variant, Some(stored))
                            }
                            Value::Tuple(_) => (0, None),
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "drop glue",
                                    "an ADT value",
                                    other.kind_name(),
                                ));
                            }
                        };
                        let plan = match variants.get(variant as usize) {
                            Some(plan) => plan.clone(),
                            None => match stored_fields {
                                // A coroutine state the conversion
                                // collected no field evidence for: a
                                // state whose fields are saved or
                                // resumed is projected by the body and
                                // so evidenced, so an unevidenced
                                // state holds only what the value
                                // itself shows - judge each stored
                                // field by its own shape.
                                Some(count) => P::Fields {
                                    fields: vec![P::Untyped; count],
                                    complete: false,
                                },
                                None => P::Unknown,
                            },
                        };
                        active.work.push((path.clone(), plan));
                        // A suspended coroutine's captures live in
                        // the upvar storage, which the selected
                        // state's plan does not cover (it addresses
                        // the state's saved storage): schedule the
                        // unresumed state's plan against the upvar
                        // storage too, addressed through a downcast
                        // to the unresumed state. The work stack is
                        // LIFO, so pushing it after the state's own
                        // plan runs it first: the captures are the
                        // shared prefix of every state's field list
                        // and drop before the state's own locals,
                        // the compiler's layout order. Captures the
                        // state machine already moved out read back
                        // as uninitialised, so for a completed or
                        // panicked coroutine this is inert.
                        if let Value::Coroutine { variant, .. } = &value {
                            if *variant != 0 {
                                if let Some(upvar_plan) = variants.first() {
                                    if *upvar_plan != P::Trivial {
                                        let mut upath = path;
                                        upath.push(RefProj::Downcast(0));
                                        active.work.push((upath, upvar_plan.clone()));
                                    }
                                }
                            }
                        }
                    }
                    P::RefCellRelease { mutable } => {
                        // A moved-out / never-written guard has no
                        // borrow to release, exactly like drop flags in
                        // real MIR.
                        let Some(guard) = self.read_at(frames, &active.root, &path)? else {
                            continue;
                        };
                        self.release_refcell_borrow(frames, &guard, mutable)?;
                    }
                    P::ChannelHandleDrop { sender } => {
                        // A moved-out / never-written handle has no
                        // count to decrement, exactly like drop flags
                        // in real MIR.
                        let Some(value) = self.read_at(frames, &active.root, &path)? else {
                            continue;
                        };
                        // A scalar here is a modelled stand-in for
                        // the handle, not a handle value: it holds
                        // no count to decrement (see
                        // `is_scalar_value`).
                        if is_scalar_value(&value) {
                            continue;
                        }
                        let Some(id) = channel_handle_id(&value) else {
                            return Err(InterpError::type_mismatch(
                                "channel handle drop glue",
                                "an mpsc channel handle",
                                value.kind_name(),
                            ));
                        };
                        self.channel_handle_dropped(
                            id,
                            if sender { "sender" } else { "receiver" },
                        );
                    }
                    P::FileHandleDrop => {
                        // A moved-out / never-written handle has
                        // no file to close, exactly like drop
                        // flags in real MIR.
                        let Some(value) =
                            self.read_at(frames, &active.root, &path)?
                        else {
                            continue;
                        };
                        let Some(id) = file_handle_id(&value) else {
                            return Err(InterpError::type_mismatch(
                                "file handle drop glue",
                                "a std::fs::File handle",
                                value.kind_name(),
                            ));
                        };
                        self.file_handle_closed(id);
                    }
                    P::SocketHandleDrop => {
                        // A moved-out / never-written handle has
                        // no socket to close, exactly like drop
                        // flags in real MIR.
                        let Some(value) =
                            self.read_at(frames, &active.root, &path)?
                        else {
                            continue;
                        };
                        let Some(id) = file_handle_id(&value) else {
                            return Err(InterpError::type_mismatch(
                                "socket handle drop glue",
                                "a std::net handle",
                                value.kind_name(),
                            ));
                        };
                        self.socket_handle_closed(id);
                    }
                    P::StringDrainDrop => {
                        // A moved-out / never-written drain has
                        // no string to remove from, exactly like
                        // drop flags in real MIR.
                        let Some(value) =
                            self.read_at(frames, &active.root, &path)?
                        else {
                            continue;
                        };
                        self.drop_string_drain(frames, &value)?;
                    }
                    P::RcHandleDrop { value } => {
                        // A moved-out / never-written handle has no
                        // count to decrement, exactly like drop
                        // flags in real MIR.
                        let Some(handle) = self.read_at(frames, &active.root, &path)? else {
                            continue;
                        };
                        // A scalar here is a modelled stand-in for
                        // the handle, not a handle value: it holds
                        // no count to decrement (see
                        // `is_scalar_value`).
                        if is_scalar_value(&handle) {
                            continue;
                        }
                        let id = match rc_handle_id(&handle)
                            .or_else(|| raw_waker_alloc_id(&handle))
                        {
                            Some(id) => id,
                            None => {
                                // A hand-built waker whose data
                                // pointer does not denote an
                                // Rc/Arc allocation field (see
                                // `raw_waker_alloc_id`): its
                                // release is the vtable's drop
                                // function, run on the data
                                // pointer, exactly as the real
                                // `Waker` drop does. The call
                                // suspends this session like a
                                // destructor call does.
                                if let Some((data, callable)) =
                                    self.raw_waker_drop_fn(frames, &handle)?
                                {
                                    // A function item is called
                                    // with the data pointer alone; a
                                    // closure value takes its
                                    // environment argument first,
                                    // exactly as an indirect closure
                                    // call passes it.
                                    let (drop_fn, call_args) = match &callable {
                                        Value::FnPtr(name) => {
                                            (name.clone(), vec![data])
                                        }
                                        Value::Closure { def, .. } => {
                                            let env = if self
                                                .closure_body_takes_env_by_value(def)
                                            {
                                                callable.clone()
                                            } else {
                                                Value::Ref(RefTarget::value(
                                                    callable.clone(),
                                                ))
                                            };
                                            (def.clone(), vec![env, data])
                                        }
                                        _ => unreachable!(
                                            "raw_waker_drop_fn yields callables only"
                                        ),
                                    };
                                    let callee = self
                                        .program
                                        .get(&drop_fn)
                                        .ok_or_else(|| InterpError::UnknownFunction {
                                            name: drop_fn.clone(),
                                        })?;
                                    if frames.len() >= self.config.max_call_depth {
                                        return Err(InterpError::CallDepthExceeded {
                                            limit: self.config.max_call_depth,
                                        });
                                    }
                                    let (arg_count, local_count) =
                                        (callee.arg_count, callee.local_count());
                                    if arg_count != call_args.len()
                                        || local_count < 1 + call_args.len()
                                    {
                                        return Err(InterpError::type_mismatch(
                                            format!("waker vtable drop `{drop_fn}`"),
                                            format!("{} argument(s)", call_args.len()),
                                            format!("{arg_count} argument(s)"),
                                        ));
                                    }
                                    if sync_dtors {
                                        // A synchronous session has
                                        // no owner block to resume
                                        // at: run the drop function
                                        // to completion now, as the
                                        // `CallDtor` arm runs a
                                        // destructor in the same
                                        // session kind.
                                        self.call_to_completion(
                                            frames,
                                            &drop_fn,
                                            call_args,
                                        )?;
                                        continue;
                                    }
                                    let mut locals = vec![Value::Uninit; local_count];
                                    for (i, arg) in call_args.into_iter().enumerate() {
                                        locals[1 + i] = arg;
                                    }
                                    let frame_id = self.fresh_frame_id();
                                    self.register_frame_base(
                                        &frames, frame_id, &drop_fn,
                                    );
                                    self.drop_continuations.push(DropContinuation {
                                        callee_frame: frame_id,
                                        session: active,
                                        sessions: suspended,
                                    });
                                    let stack_bytes =
                                        self.charge_stack(frames, self.frame_stack_bytes(callee))?;
                                    frames.push(Frame {
                                        id: frame_id,
                                        func: drop_fn,
                                        locals,
                                        stack_bytes,
                                        block: 0,
                                        stmt: 0,
                                        return_dest: None,
                                        return_block: None,
                                        unwind_payload: None,
                                        in_cleanup: false,
            call_once_env_drop: None,
            incoming_asm: None,
                                    });
                                    return Ok(());
                                }
                                return Err(InterpError::type_mismatch(
                                    "reference-counted handle drop glue",
                                    "an Rc/Arc handle",
                                    handle.kind_name(),
                                ));
                            }
                        };
                        if !self.rc_dec_strong(id)? {
                            continue;
                        }
                        // The last strong handle is gone: the
                        // pointee's glue runs against the stored
                        // value in a nested session rooted at the
                        // allocation, and the weak bundle the
                        // strong handles held is released in a
                        // second nested session after it — so the
                        // storage, and its counts, stay observable
                        // to the destructor exactly as long as they
                        // are natively (a `Weak::upgrade` inside the
                        // destructor still reads strong = 0 and
                        // returns `None`, never a dangling error).
                        let release = GlueSession {
                            owns_root_alloc: false,
                            root: RefTarget::Alloc {
                                id,
                                projection: Vec::new(),
                            },
                            work: vec![(Vec::new(), P::RcWeakRelease)],
                            adts: active.adts.clone(),
                            completion: GlueCompletion::Resume,
                        };
                        // The pointee sits at the inner
                        // aggregate's value field; the inner
                        // itself may sit behind a raw
                        // allocation block's slot / wrapper
                        // steps (see `rc_inner_loc`).
                        let mut pointee_projection = self
                            .rc_inner_loc(&handle)
                            .map(|(_, prefix)| prefix)
                            .unwrap_or_default();
                        pointee_projection.push(RefProj::Field(2));
                        let nested = GlueSession {
                            owns_root_alloc: false,
                            root: RefTarget::Alloc {
                                id,
                                projection: pointee_projection,
                            },
                            work: vec![(Vec::new(), *value)],
                            adts: active.adts.clone(),
                            completion: GlueCompletion::Resume,
                        };
                        suspended.push(std::mem::replace(&mut active, release));
                        suspended.push(std::mem::replace(&mut active, nested));
                        continue 'sessions;
                    }
                    P::IoErrorRepr => {
                        // The packed `io::Error` representation:
                        // only the `Custom` tag owns anything (see
                        // `DropPlan::IoErrorRepr`). A moved-out /
                        // never-written representation has no glue
                        // to run, as with rustc's drop flags.
                        let Some(value) = self.read_at(frames, &active.root, &path)? else {
                            continue;
                        };
                        let Some(word) = crate::builtins::io_error_tagged_word(&value) else {
                            // Not the packed shape: the value owns
                            // nothing this plan can name.
                            continue;
                        };
                        if word & 0b11 != 0b01 {
                            continue;
                        }
                        // The word minus the tag is the
                        // registered synthetic address of the
                        // payload's allocation. Take the
                        // registration out before the payload's
                        // destructor runs (the address denotes
                        // nothing from here on), then drop the
                        // boxed trait object the payload's `Custom`
                        // value holds (its error field) in a
                        // nested session rooted at the allocation,
                        // as the real owner drop does.
                        let Some(id) = self.take_address_owner(word.wrapping_sub(1)) else {
                            continue;
                        };
                        let nested = GlueSession {
                            owns_root_alloc: false,
                            root: RefTarget::Alloc {
                                id,
                                projection: Vec::new(),
                            },
                            work: vec![(vec![RefProj::Field(1)], P::Dyn)],
                            adts: active.adts.clone(),
                            completion: GlueCompletion::Resume,
                        };
                        suspended.push(std::mem::replace(&mut active, nested));
                        continue 'sessions;
                    }
                    P::WakerDrop => {
                        // A moved-out / never-written waker has
                        // nothing to release, as with rustc's drop
                        // flags.
                        let Some(value) = self.read_at(frames, &active.root, &path)? else {
                            continue;
                        };
                        // The modelled `From` conversion's waker:
                        // a one-field aggregate carrying the
                        // `Arc` / `Rc` handle itself. Its release
                        // is the handle's, at the field's path.
                        if let Value::Enum { fields, .. } = &value {
                            if fields.len() == 1 && rc_handle_id(&fields[0]).is_some() {
                                let mut sub = path.clone();
                                sub.push(RefProj::Field(0));
                                active.work.push((
                                    sub,
                                    P::RcHandleDrop {
                                        value: Box::new(P::Dynamic),
                                    },
                                ));
                                continue;
                            }
                        }
                        // A real `RawWaker` (the `Waker::from_raw`
                        // shape): the waker value's field is the
                        // raw waker, whose fields are the data
                        // pointer and the vtable reference. Dropping
                        // the waker calls the vtable's drop entry
                        // with the data pointer, as `Waker`'s own
                        // `Drop` impl does; the call runs as a
                        // nested frame with this session as its
                        // continuation, exactly as a `CallDtor`
                        // call does.
                        let raw = match &value {
                            Value::Enum { fields, .. } if fields.len() == 1 => {
                                fields[0].clone()
                            }
                            _ => value.clone(),
                        };
                        let Value::Enum {
                            fields: raw_fields,
                            ..
                        } = &raw
                        else {
                            return Err(InterpError::type_mismatch(
                                "waker drop glue",
                                "a RawWaker value or a modelled waker handle",
                                value.kind_name(),
                            ));
                        };
                        if raw_fields.len() != 2 {
                            return Err(InterpError::type_mismatch(
                                "waker drop glue",
                                "a RawWaker value (data pointer and vtable)",
                                value.kind_name(),
                            ));
                        }
                        let data = raw_fields[0].clone();
                        let resolved = self.deref_value(frames, &raw_fields[1])?;
                        let vtable = self.read_resolved(frames, &resolved)?;
                        let Value::Enum {
                            fields: vtable_fields,
                            ..
                        } = &vtable
                        else {
                            return Err(InterpError::type_mismatch(
                                "waker drop glue",
                                "a RawWakerVTable value",
                                vtable.kind_name(),
                            ));
                        };
                        // The drop entry is taken as the value
                        // it was stored as: a plain function item
                        // is a `FnPtr`, while a non-capturing
                        // closure literal in the vtable constant
                        // stays a `Closure` value (its coercion to
                        // a function pointer is a type-level fact
                        // the value model does not rewrite) --
                        // exactly as `raw_waker_drop_fn` judges the
                        // same entry. A function item is called
                        // with the data pointer alone; a closure
                        // value takes its environment argument
                        // first, exactly as an indirect closure
                        // call passes it.
                        let drop_callable = match vtable_fields.get(3) {
                            Some(
                                callable @ (Value::FnPtr(_) | Value::Closure { .. }),
                            ) => callable.clone(),
                            _ => {
                                return Err(InterpError::type_mismatch(
                                    "waker drop glue",
                                    "a RawWakerVTable with a function-pointer drop entry",
                                    vtable.kind_name(),
                                ));
                            }
                        };
                        let (drop_name, call_args) = match &drop_callable {
                            Value::FnPtr(name) => (name.clone(), vec![data]),
                            Value::Closure { def, .. } => {
                                let env = if self
                                    .closure_body_takes_env_by_value(def)
                                {
                                    drop_callable.clone()
                                } else {
                                    Value::Ref(RefTarget::value(
                                        drop_callable.clone(),
                                    ))
                                };
                                (def.clone(), vec![env, data])
                            }
                            _ => unreachable!(
                                "waker vtable drop entry is a callable only"
                            ),
                        };
                        let callee = self.program.get(&drop_name).ok_or_else(|| {
                            InterpError::UnknownFunction {
                                name: drop_name.clone(),
                            }
                        })?;
                        if frames.len() >= self.config.max_call_depth {
                            return Err(InterpError::CallDepthExceeded {
                                limit: self.config.max_call_depth,
                            });
                        }
                        let (arg_count, local_count) =
                            (callee.arg_count, callee.local_count());
                        if arg_count != call_args.len()
                            || local_count < 1 + call_args.len()
                        {
                            return Err(InterpError::type_mismatch(
                                format!("waker vtable drop entry `{drop_name}`"),
                                format!("{} argument(s)", call_args.len()),
                                format!("{arg_count} argument(s)"),
                            ));
                        }
                        if sync_dtors {
                            // A synchronous session has no owner
                            // block to resume at: run the drop
                            // entry to completion now, as the
                            // `CallDtor` arm runs a destructor in
                            // the same session kind. A panic in it
                            // propagates as the consuming builtin's
                            // own panic.
                            self.call_to_completion(frames, &drop_name, call_args)?;
                            continue;
                        }
                        let mut locals = vec![Value::Uninit; local_count];
                        for (index, arg) in call_args.into_iter().enumerate() {
                            locals[1 + index] = arg;
                        }
                        let frame_id = self.fresh_frame_id();
                        self.register_frame_base(&frames, frame_id, &drop_name);
                        self.drop_continuations.push(DropContinuation {
                            callee_frame: frame_id,
                            session: active,
                            sessions: suspended,
                        });
                        let stack_bytes =
                            self.charge_stack(frames, self.frame_stack_bytes(callee))?;
                        frames.push(Frame {
                            id: frame_id,
                            func: drop_name,
                            locals,
                            stack_bytes,
                            block: 0,
                            stmt: 0,
                            return_dest: None,
                            return_block: None,
                            unwind_payload: None,
                            in_cleanup: false,
            call_once_env_drop: None,
            incoming_asm: None,
                        });
                        return Ok(());
                    }
                    P::RcWeakRelease => {
                        // The weak reference being released: the
                        // handle at the work path, or — for the
                        // bundle release the interpreter synthesizes
                        // after a pointee's glue — the allocation
                        // the session itself is rooted at.
                        let rooted = if path.is_empty() {
                            match &active.root {
                                RefTarget::Alloc { id, projection }
                                    if projection.is_empty() =>
                                {
                                    Some(*id)
                                }
                                _ => None,
                            }
                        } else {
                            None
                        };
                        let id = match rooted {
                            Some(id) => id,
                            None => {
                                let Some(handle) =
                                    self.read_at(frames, &active.root, &path)?
                                else {
                                    continue;
                                };
                                // A handle naming no allocation is
                                // a dangling `Weak::new()`: its count
                                // lives in a static fake allocation
                                // that is never freed, so releasing
                                // it is unobservable - a no-op.
                                let Some(id) = rc_handle_id(&handle) else {
                                    continue;
                                };
                                id
                            }
                        };
                        self.rc_dec_weak(id)?;
                    }
                    P::Dynamic => {
                        // A moved-out / never-written place holds
                        // nothing to drop, as with rustc's drop flags.
                        // The value is judged like an unaccounted
                        // `dyn` payload: the one-field aggregate
                        // wrappings the interpreter's own models
                        // build are stripped before the shape
                        // judgment, whose destructor-hiding shapes
                        // stay explicit errors.
                        if let Some(value) = self.read_at(frames, &active.root, &path)? {
                            drop_unaccounted_dyn(&value)?;
                        }
                    }
                    P::Dyn => {
                        // The dropped value is a trait object: run the
                        // concrete type's glue, which the value
                        // carries from its unsizing cast, against the
                        // pointee as a nested session.
                        let Some(value) = self.read_at(frames, &active.root, &path)? else {
                            continue;
                        };
                        // The trait object may sit at the bottom of
                        // a pointer wrapper chain rather than at the
                        // value's surface: a box a converted
                        // constructor body built (`Box::from_raw`)
                        // is the real `Unique` / `NonNull` layout
                        // over the fat pointer, with the allocator
                        // field beside it, where the builtin's box
                        // *is* the fat pointer itself. Descend the
                        // field-0 chain — the layers beside the
                        // pointer are not on the pointer path — and
                        // judge the value found there.
                        let (_, bottom) = wrapper_chain_bottom(&value);
                        let Value::DynRef {
                            base,
                            drop_glue,
                            ..
                        } = bottom.clone()
                        else {
                            // Not a trait-object value at all: judge
                            // it by its shape, as the `Dynamic` plan
                            // does.
                            drop_unaccounted_dyn(&value)?;
                            continue;
                        };
                        let Some(glue) = drop_glue.map(|g| *g) else {
                            // The value carries no accounted glue:
                            // it was fabricated by the interpreter
                            // itself (a caught panic's payload box, a
                            // joined thread's message box) rather than
                            // formed by an unsizing cast. Judge its
                            // contents by shape; shapes that can hide
                            // a destructor are explicit errors, never
                            // a silent skip.
                            drop_unaccounted_dyn(&base)?;
                            continue;
                        };
                        // A reference-counted handle in trait-object
                        // form: the pointee's glue is the last strong
                        // handle's business, exactly as for a plain
                        // handle (see `RcHandleDrop`) — decrement the
                        // count first, and only the drop that takes
                        // it to zero runs the glue the value carries,
                        // against the allocation's value field, with
                        // the weak bundle released after it.
                        let rc_loc = self.rc_inner_loc(&base);
                        let rc_base = rc_loc.as_ref().map(|(id, _)| *id);
                        if let Some(id) = rc_base {
                            if !self.rc_dec_strong(id)? {
                                continue;
                            }
                            let release = GlueSession {
                                owns_root_alloc: false,
                                root: RefTarget::Alloc {
                                    id,
                                    projection: Vec::new(),
                                },
                                work: vec![(Vec::new(), P::RcWeakRelease)],
                                adts: active.adts.clone(),
                                completion: GlueCompletion::Resume,
                            };
                            if glue.root == P::Trivial {
                                suspended.push(std::mem::replace(&mut active, release));
                                continue 'sessions;
                            }
                            let mut pointee_projection = rc_loc
                                .as_ref()
                                .map(|(_, prefix)| prefix.clone())
                                .unwrap_or_default();
                            pointee_projection.push(RefProj::Field(2));
                            let nested = GlueSession {
                                owns_root_alloc: false,
                                root: RefTarget::Alloc {
                                    id,
                                    projection: pointee_projection,
                                },
                                work: vec![(Vec::new(), glue.root)],
                                adts: glue.adts,
                                completion: GlueCompletion::Resume,
                            };
                            suspended.push(std::mem::replace(&mut active, release));
                            suspended.push(std::mem::replace(&mut active, nested));
                            continue 'sessions;
                        }
                        if glue.root == P::Trivial {
                            continue;
                        }
                        // The pointee the glue applies to: inside the
                        // one-field aggregate for a smart pointer
                        // built by the `new` builtins, or behind the
                        // data pointer itself for a pointer-shaped
                        // base.
                        let pointee_root = match &*base {
                            // A reference-counted handle base:
                            // the pointee is the allocation's
                            // value field in the store.
                            Value::Enum { .. } if rc_base.is_some() => RefTarget::Alloc {
                                id: rc_base.expect("checked"),
                                projection: {
                                    let mut projection = rc_loc
                                        .as_ref()
                                        .map(|(_, prefix)| prefix.clone())
                                        .unwrap_or_default();
                                    projection.push(RefProj::Field(2));
                                    projection
                                },
                            },
                            // A built box's single field is its
                            // allocation pointer: the pointee lives
                            // in the allocation store, behind that
                            // very reference.
                            Value::Enum { fields, .. }
                                if fields.len() == 1
                                    && matches!(fields[0], Value::Ref(_)) =>
                            {
                                match &fields[0] {
                                    Value::Ref(target) => target.clone(),
                                    _ => unreachable!("field shape checked"),
                                }
                            }
                            Value::Enum { fields, .. } if fields.len() == 1 => {
                                RefTarget::value(fields[0].clone())
                            }
                            Value::Ref(target) => target.clone(),
                            other => {
                                // An aggregate-built box base
                                // (`Box::from_raw`'s product, a
                                // by-value downcast's result): the
                                // data pointer sits at the end of
                                // the field-0 wrapper chain beside
                                // the allocator field.
                                match self.dyn_data_pointer(other) {
                                    Value::Ref(target) => target,
                                    _ => {
                                        return Err(InterpError::type_mismatch(
                                            "drop glue",
                                            "a box aggregate or reference dyn data pointer",
                                            other.kind_name(),
                                        ));
                                    }
                                }
                            }
                        };
                        let nested = GlueSession {
                            owns_root_alloc: plan_owns_box(&glue.root, &glue.adts),
                            root: pointee_root,
                            work: vec![(Vec::new(), glue.root)],
                            adts: glue.adts,
                            completion: GlueCompletion::Resume,
                        };
                        suspended.push(std::mem::replace(&mut active, nested));
                        continue 'sessions;
                    }
                    P::DynOrPlain { plain } => {
                        // Judge the value's own form (see
                        // `DropPlan::DynOrPlain`): a trait-object
                        // value runs the dyn plan against the glue
                        // it carries; a plain pointer aggregate
                        // runs the plan built for that form. The
                        // raw read keeps the stored form — following
                        // an allocation pointer here would read the
                        // pointee and judge it instead of the
                        // pointer.
                        let value = self.read_at_raw(frames, &active.root, &path)?;
                        if matches!(value, Some(Value::DynRef { .. })) {
                            active.work.push((path, P::Dyn));
                        } else {
                            active.work.push((path, *plain));
                        }
                    }
                    P::BoxPointee { pointee } => {
                        // Judge the box value's own shape (see
                        // `DropPlan::BoxPointee`). The raw read
                        // keeps the stored form: the pointer must
                        // not be followed before judging, or the
                        // pointee would be judged instead of the
                        // box.
                        let Some(value) =
                            self.read_at_raw(frames, &active.root, &path)?
                        else {
                            // Moved out of / never written: no
                            // glue, as with rustc's drop flags.
                            continue;
                        };
                        // A never-written zero-sized box reads as
                        // a fabricated field-less aggregate (see
                        // `read_at_raw`): there is no pointer and
                        // no pointee to drop.
                        if matches!(&value, Value::Enum { fields, .. } if fields.is_empty())
                        {
                            continue;
                        }
                        // Descend the pointer wrapper chain (see
                        // `wrapper_chain_bottom`): the value at
                        // its bottom is what the box carries — the
                        // pointer itself, or the pointee a
                        // fabricated box holds inline.
                        let (depth, bottom) = wrapper_chain_bottom(&value);
                        match bottom {
                            Value::Ref(target) => {
                                // The pointee lives behind the
                                // pointer: its glue runs in a
                                // nested session rooted at the
                                // pointer's target, exactly as the
                                // box's field 0 denoting the
                                // pointee did for the builtin form.
                                let nested = GlueSession {
                                    owns_root_alloc: plan_owns_box(&*pointee, &active.adts),
                                    root: target.clone(),
                                    work: vec![(Vec::new(), *pointee)],
                                    adts: active.adts.clone(),
                                    completion: GlueCompletion::Resume,
                                };
                                suspended.push(std::mem::replace(&mut active, nested));
                                continue 'sessions;
                            }
                            Value::FatRef { base, .. }
                                if matches!(
                                    &**base,
                                    Value::Ref(RefTarget::Alloc { .. })
                                        | Value::Ref(RefTarget::Place { .. })
                                        | Value::Ref(RefTarget::Static { .. })
                                        | Value::Ref(RefTarget::Value(..))
                                ) =>
                            {
                                // A fat pointer at the chain's
                                // bottom (a `Box<[T]>` a converted
                                // constructor body built): the
                                // pointee lives behind its data
                                // pointer, as for a thin pointer;
                                // the stored value behind it is the
                                // whole slice the plan addresses.
                                let Value::Ref(target) = &**base else {
                                    unreachable!("base shape checked");
                                };
                                let nested = GlueSession {
                                    owns_root_alloc: plan_owns_box(&*pointee, &active.adts),
                                    root: target.clone(),
                                    work: vec![(Vec::new(), *pointee)],
                                    adts: active.adts.clone(),
                                    completion: GlueCompletion::Resume,
                                };
                                suspended.push(std::mem::replace(&mut active, nested));
                                continue 'sessions;
                            }
                            Value::DynRef { .. } => {
                                // The chain carries a trait object
                                // (a pointee that mentions `dyn`
                                // without being one itself): its
                                // glue is the dyn plan's, run
                                // against the box value at this
                                // path, whose own chain descent
                                // finds the trait object.
                                active.work.push((path, P::Dyn));
                            }
                            _ => {
                                // The chain bottoms out at a value
                                // the box holds inline (a
                                // fabricated box, a fat pointer):
                                // the pointee glue runs against it
                                // in place, addressed through the
                                // chain's field-0 steps — at least
                                // the one step the plan always
                                // addressed a box's contents
                                // through.
                                let mut sub = path.clone();
                                for _ in 0..depth.max(1) {
                                    sub.push(RefProj::Field(0));
                                }
                                active.work.push((sub, *pointee));
                            }
                        }
                    }
                    P::GuardDtor { func } => {
                        // Run the guard's destructor only on a
                        // value shaped like a guard the converted
                        // lock bodies built (see the plan's doc):
                        // re-queue it as a plain destructor call.
                        // A moved-out or forged (stdio lock) value
                        // has no destructor to run.
                        let shaped = match self.read_at(frames, &active.root, &path)? {
                            Some(value) => lock_guard_shaped(&value),
                            None => false,
                        };
                        if shaped {
                            active.work.push((path, P::CallDtor { func }));
                        }
                    }
                    P::AlignedTemp { plan } => {
                        let Some(value) =
                            self.read_at(frames, &active.root, &path)?
                        else {
                            // Moved out of / never written: no glue,
                            // as with rustc's drop flags.
                            continue;
                        };
                        // The value sits at an unaligned address
                        // inside a packed parent: the compiler's
                        // glue drops such a field from an aligned
                        // temporary it moves the field into. Park
                        // the field's value in an allocation of its
                        // own -- as `drop_value_sync` parks a
                        // consumed value, so a destructor that
                        // writes back into its receiver has somewhere
                        // to write -- and run the field's plan
                        // against the allocation in a nested session.
                        let id = self.alloc_insert(value);
                        let nested = GlueSession {
                            owns_root_alloc: false,
                            root: RefTarget::Alloc {
                                id,
                                projection: Vec::new(),
                            },
                            work: vec![(Vec::new(), *plan)],
                            adts: active.adts.clone(),
                            completion: GlueCompletion::Resume,
                        };
                        suspended
                            .push(std::mem::replace(&mut active, nested));
                        continue 'sessions;
                    }
                    P::CallDtor { func } => {
                        if self.read_at(frames, &active.root, &path)?.is_none() {
                            // Moved out of / never written: no glue, as
                            // with rustc's drop flags.
                            continue;
                        }
                        let callee = self.program.get(&func).ok_or_else(|| {
                            InterpError::UnknownFunction { name: func.clone() }
                        })?;
                        if frames.len() >= self.config.max_call_depth {
                            return Err(InterpError::CallDepthExceeded {
                                limit: self.config.max_call_depth,
                            });
                        }
                        let (arg_count, local_count) =
                            (callee.arg_count, callee.local_count());
                        if arg_count != 1 || local_count < 2 {
                            return Err(InterpError::type_mismatch(
                                format!("destructor `{func}`"),
                                "one argument",
                                format!("{arg_count} argument(s)"),
                            ));
                        }
                        let target_ref = self.extend_ref(frames, &active.root, &path)?;
                        // A `pin_drop` destructor takes
                        // `Pin<&mut Self>`, and `Pin` is a one-field
                        // aggregate in this value model (the shape
                        // converted `Pin::new` bodies build): pass
                        // the wrapper, not the bare pointer, or the
                        // body's own `Deref` for `Pin` resolves
                        // through the pointee and unwraps the wrong
                        // field.
                        let dtor_arg = if func.ends_with(">::pin_drop") {
                            Value::Enum {
                                variant: 0,
                                fields: vec![Value::Ref(target_ref)],
                            }
                        } else {
                            Value::Ref(target_ref)
                        };
                        if sync_dtors {
                            // A synchronous session has no owner
                            // block to resume at: run the
                            // destructor to completion now, as
                            // the builtins run the guest code
                            // they drive. A panic in it
                            // propagates as the consuming
                            // builtin's own panic.
                            self.call_to_completion(
                                frames,
                                &func,
                                vec![dtor_arg],
                            )?;
                            continue;
                        }
                        let mut locals = vec![Value::Uninit; local_count];
                        locals[1] = dtor_arg;
                        let frame_id = self.fresh_frame_id();
                        self.register_frame_base(&frames, frame_id, &func);
                        self.drop_continuations.push(DropContinuation {
                            callee_frame: frame_id,
                            session: active,
                            sessions: suspended,
                        });
                        let stack_bytes =
                            self.charge_stack(frames, self.frame_stack_bytes(callee))?;
                        frames.push(Frame {
                            id: frame_id,
                            func,
                            locals,
                            stack_bytes,
                            block: 0,
                            stmt: 0,
                            return_dest: None,
                            return_block: None,
                            unwind_payload: None,
                            in_cleanup: false,
            call_once_env_drop: None,
            incoming_asm: None,
                        });
                        return Ok(());
                    }
                }
            }
            // The active session's work is done.
            let finishing_target = match active.completion {
                GlueCompletion::Resume => match suspended.pop() {
                    Some(session) => {
                        active = session;
                        continue 'sessions;
                    }
                    None => {
                        return Err(InterpError::type_mismatch(
                            "drop glue",
                            "an enclosing glue session to resume",
                            "none",
                        ));
                    }
                },
                GlueCompletion::Jump(target) => Some(target),
                GlueCompletion::UnwindCleanup => None,
                GlueCompletion::Sync => None,
            };
            // A finishing session releases the dropped place
            // exactly as a completed drop does: the place is dead,
            // as in MIR -- a direct local of the owner frame is
            // cleared, and a dropped box's allocation is freed with
            // it: the pointee's glue has run, and the storage the
            // box pointed at is gone, as a real free makes it. (A
            // box nested inside a larger dropped value is reached
            // as a work item of that value's session, not as a
            // session root; its allocation entry is left in the
            // store -- ids are never reused, so nothing can observe
            // the difference within a run.)
            if let Ok(Some(value)) = self.read_at(frames, &active.root, &[]) {
                // Only a box-shaped value whose pointer denotes
                // the allocation's base owns the allocation: a
                // one-field struct holding a reference into
                // another value's storage (a drop guard's `&mut`
                // field, say) has the same value shape but owns
                // nothing, and freeing its pointee would cut the
                // owner's storage from under it.
                if let Some((id, projection)) = box_alloc_pointer(&value) {
                    // Only a session whose root plan is a box's
                    // own glue owns the allocation: a place of
                    // any other type can hold the same box shape
                    // without owning it -- a `Pin<&mut T>`
                    // argument, say, is a one-field aggregate
                    // holding a base reference into an allocation
                    // the caller's `Box` owns, and dropping the
                    // argument must not free the caller's storage
                    // (a later poll of the same future would then
                    // read a freed allocation). The plan, not the
                    // value's shape, is the ownership evidence:
                    // it is what `GlueSession::owns_root_alloc`
                    // records.
                    if projection_is_base(&projection) && active.owns_root_alloc {
                        self.note_box_free(frames, &value);
                        self.free_alloc(id);
                    }
                }
            }
            // The place is cleared only when it is a local of the
            // frame executing the drop. A place that resolved
            // through a pointer into another frame (the
            // `drop_in_place` shim dropping the caller's local
            // through its `*mut` argument) is not killed: the
            // pointer operation is invisible to the owner body's
            // drop elaboration, which keeps treating the local as
            // live -- it may still be moved, reassigned or
            // forgotten afterwards. Only the pointee's glue ran,
            // as in real memory, where `drop_in_place` leaves the
            // bytes in place.
            self.note_string_buffer_free(frames, &active.root);
            // A session that dropped a whole allocation's value
            // through a pointer (a `drop_in_place` on a raw
            // pointer into a heap allocation -- the pointee drop
            // inside a compiler `drop_glue` shim, say) leaves the
            // storage dropped: a later step reaching the same
            // allocation (the shim's destructor step, converted
            // to the box's glue plan, reaching the pointee
            // through the box) must find nothing left to drop a
            // second time. The dropped state is recorded in
            // `glue_dropped_allocs`, which only glue reads
            // consult, instead of overwriting the storage: the
            // shim's other later step is the box destructor's
            // own converted body, whose `size_of_val` on the
            // pointee reads the storage through the ordinary
            // path to price the deallocation, exactly as a
            // native drop leaves the bytes in place until the
            // free. Sessions nested inside another (a
            // reference-counted pointee's glue, rooted at the
            // allocation's value field) keep the storage: their
            // enclosing session still owns it.
            if matches!(
                active.completion,
                GlueCompletion::Jump(_) | GlueCompletion::Sync
            ) {
                if let RefTarget::Alloc { id, projection } = &active.root {
                    if projection.is_empty() {
                        self.glue_dropped_allocs.borrow_mut().insert(*id);
                    } else if let Some(stored) = self.alloc_value_snapshot(id) {
                        // A projected cell is cleared only when it
                        // is part of a coroutine's state: a
                        // coroutine body drops its upvars in place
                        // through the state pointer when it
                        // completes (or as cleanup when it panics),
                        // and the future's own glue reaching the
                        // same cell afterwards must find nothing
                        // left to drop a second time. The coroutine
                        // glue is the only later reader of state
                        // cells and reads an uninitialised cell as
                        // already moved out. Any other projected
                        // cell keeps its value: an in-place drop
                        // through a pointer leaves the storage
                        // readable to the model's other readers,
                        // exactly as the whole-place treatment
                        // below has always assumed.
                        let func = frames.last().map(|f| f.func.as_str()).unwrap_or("");
                        if projection_enters_coroutine(&stored, projection, func) {
                            let resolved = Resolved::Alloc {
                                id: *id,
                                projection: projection.clone(),
                            };
                            self.write_resolved(frames, &resolved, Value::Uninit)?;
                        }
                    }
                }
            }
            if let RefTarget::Place {
                frame,
                local,
                projection,
            } = &active.root
            {
                if projection.is_empty() {
                    if frames.last().map(|f| f.id) == Some(*frame) {
                        let frame_pos = self.frame_pos(frames, *frame)?;
                        if let Some(slot) = frames[frame_pos].locals.get_mut(*local) {
                            *slot = Value::Uninit;
                        }
                    }
                } else if matches!(active.completion, GlueCompletion::Jump(_)) {
                    // The coroutine-state rule above, for a
                    // coroutine stored in a frame local (an
                    // unboxed future, polled through a mutable
                    // reference to the local): the body's in-place
                    // upvar drop leaves the state cell dead for
                    // the future's own glue. Unlike the
                    // whole-place treatment above, this applies
                    // from any frame: the session completing the
                    // drop usually runs inside the coroutine
                    // body's own frame, above the owner's, and a
                    // state cell's only later reader is the
                    // future's glue, which runs against the local
                    // wherever it lives. A root frame that has
                    // already left the stack has no cell to clear.
                    if let Ok(frame_pos) = self.frame_pos(frames, *frame) {
                        let stored = frames[frame_pos]
                            .locals
                            .get(*local)
                            .cloned()
                            .unwrap_or(Value::Uninit);
                        let func = frames.last().map(|f| f.func.as_str()).unwrap_or("");
                        if projection_enters_coroutine(&stored, projection, func) {
                            let resolved = Resolved::Place {
                                frame_pos,
                                local: *local,
                                projection: projection.clone(),
                            };
                            self.write_resolved(frames, &resolved, Value::Uninit)?;
                        }
                    }
                }
            }
            if matches!(active.completion, GlueCompletion::Sync) {
                // A synchronous session simply ends: no owner
                // frame to jump to, no stashed panic to resume.
                return Ok(());
            }
            match finishing_target {
                Some(target) => {
                    let owner = frames.last().expect("owner frame");
                    let owner_body = self
                        .program
                        .get(&owner.func)
                        .cloned()
                        .ok_or_else(|| InterpError::UnknownFunction {
                            name: owner.func.clone(),
                        })?;
                    return Self::jump(frames, &owner_body, target);
                }
                None => {
                    // The session ran as a panic's cleanup: the
                    // stashed panic continues unwinding from here,
                    // as a resume of the original unwinding (the
                    // panic hook does not fire again for it).
                    let Some((message, payload)) = self.glue_unwind_panic.take() else {
                        return Err(InterpError::type_mismatch(
                            "drop glue cleanup",
                            "a stashed panic to resume",
                            "none",
                        ));
                    };
                    self.resume_signaled = true;
                    return Err(InterpError::Panic { message, payload });
                }
            }
        }
    }
}

fn drop_unaccounted_dyn(value: &Value) -> InterpResult<()> {
    match value {
        Value::DynRef { base, .. } => drop_unaccounted_dyn(base),
        // One-field aggregates nest: a modelled smart-pointer or
        // handle value wraps its payload in one aggregate per
        // layer (a  around its counted handle around the
        // wake object, whose own one-field structs wrap theirs).
        // Strip the chain; the payload itself is still judged by
        // , whose shapes that can hide a
        // destructor stay explicit errors.
        Value::Enum { fields, .. } if fields.len() == 1 => {
            drop_unaccounted_dyn(&fields[0])
        }
        other => drop_dyn_value(other),
    }
}

fn drop_dyn_value(value: &Value) -> InterpResult<()> {
    match value {
        Value::Uninit
        | Value::Unit
        | Value::Bool(_)
        | Value::Int(_)
        | Value::Uint(_)
        | Value::Float(_)
        | Value::Float128(_)
        | Value::Str(_)
        | Value::Bytes(_)
        | Value::ErasedBlock { .. }
        | Value::UnionBytes { .. }
        | Value::Ref(_)
        | Value::FatRef { .. }
        | Value::FnPtr(_)
        | Value::FmtArg { .. }
        | Value::FmtCount(_)
        | Value::FmtArguments(_)
        | Value::Formatter(_)
        | Value::VaList(_)
        | Value::FmtBuilder(_) => Ok(()),
        Value::DynRef { base, .. } => drop_dyn_value(base),
        Value::Tuple(items) => items.iter().try_for_each(drop_dyn_value),
        Value::Closure { env, .. } => env.iter().try_for_each(drop_dyn_value),
        Value::Enum { fields, .. } if fields.is_empty() => Ok(()),
        Value::Enum { .. }
        | Value::SparseVec { .. }
        | Value::Coroutine { .. }
        | Value::PanicPayload(_) => {
            Err(InterpError::UnsupportedTerminator {
                kind: "Drop".to_string(),
            })
        }
    }
}
