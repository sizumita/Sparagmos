//! Converter: statements and terminators.


use sparagmos_mir as mir;

use crate::ir;

use super::adt_fields::*;
use super::asm::*;
use super::converter::*;
use super::drop_glue::*;
use super::dyn_unsize::*;
use super::errors::*;
use super::helpers::*;
use super::resolve::*;
use super::run::*;


impl<'a> Converter<'a> {
    // ----- statements & terminators ------------------------------------------

    /// The coverage owner the driver recorded for `stmt`:
    /// `Some(Some(symbol))` when the statement was inlined from
    /// the instance with that mangled symbol, `Some(None)` for
    /// the body's own statements, and `None` when the body
    /// carries no attribution data. The statement is located by
    /// pointer identity within its body (conversion always
    /// passes statements borrowed from that body), and its
    /// coverage index is its position among the body's coverage
    /// statements, matching the driver's table.
    fn coverage_owner(
        &self,
        body: &mir::Body,
        stmt: &mir::Statement,
    ) -> Option<Option<String>> {
        if body.coverage_owners.is_empty() {
            return None;
        }
        let mut idx = 0usize;
        for bb in &body.basic_blocks {
            for s in &bb.statements {
                if std::ptr::eq(s, stmt) {
                    return body.coverage_owners.get(idx).cloned();
                }
                if matches!(s.kind, mir::StatementKind::Coverage(_)) {
                    idx += 1;
                }
            }
        }
        None
    }

    pub(super) fn convert_statement(
        &self,
        body: &mir::Body,
        stmt: &mir::Statement,
    ) -> Result<ir::Statement, ConvertError> {
        use mir::StatementKind as S;
        Ok(match &stmt.kind {
            S::Assign(pair) => {
                let (place, rv) = &**pair;
                let dest_ty = self.place_ty(body, place);
                // A whole-pointee store of a vector-shaped value
                // through a dereference: the shaped store, which
                // writes the value lane by lane when the pointee
                // lives in a raw element block.
                if let mir::Rvalue::Use(op) = rv {
                    if let Some(shape) = self.deref_vector_shape(body, place) {
                        return Ok(ir::Statement {
                            kind: ir::StatementKind::ShapedStore {
                                place: self.convert_place(body, place)?,
                                value: self.convert_operand(body, op, dest_ty.as_ref())?,
                                shape,
                            },
                        });
                    }
                }
                let ir_place = self.convert_place(body, place)?;
                let ir_rv = self.convert_rvalue(body, rv, dest_ty.as_ref())?;
                ir::Statement::assign(ir_place, ir_rv)
            }
            S::StorageLive(l) => ir::Statement {
                kind: ir::StatementKind::StorageLive(ir::Local(l.index())),
            },
            S::StorageDead(l) => ir::Statement {
                kind: ir::StatementKind::StorageDead(ir::Local(l.index())),
            },
            // Execution no-ops in rustc MIR (borrow-check / bookkeeping
            // only): an executor treats them as `Nop`.
            S::Nop
            | S::FakeRead(..)
            | S::PlaceMention(_)
            | S::Retag(..)
            | S::AscribeUserType(..)
            | S::ConstEvalCounter => ir::Statement::nop(),
            // A coverage counter point. When the driver
            // attributed this statement to the instance it was
            // inlined from, count against that instance's record
            // (or nothing, when that instance has no record: no
            // counter was emitted for it natively either);
            // otherwise the interpreter resolves the record from
            // the executing body at run time.
            S::Coverage(bcb) => {
                let record = match self.coverage_owner(body, stmt) {
                    Some(Some(symbol)) => self
                        .coverage_by_symbol
                        .get(&symbol)
                        .copied()
                        .unwrap_or(u32::MAX - 1),
                    _ => u32::MAX,
                };
                ir::Statement::coverage(*bcb, record)
            }
            S::SetDiscriminant {
                place,
                variant_index,
            } => ir::Statement {
                kind: ir::StatementKind::SetDiscriminant {
                    place: self.convert_place(body, place)?,
                    variant: *variant_index,
                },
            },
            S::Intrinsic(intr) => ir::Statement {
                kind: match &**intr {
                    mir::NonDivergingIntrinsic::Assume(op) => {
                        ir::StatementKind::Assume(self.convert_operand(body, op, None)?)
                    }
                    mir::NonDivergingIntrinsic::CopyNonOverlapping { src, dst, count } => {
                        ir::StatementKind::CopyNonOverlapping {
                            src: self.convert_operand(body, src, None)?,
                            dst: self.convert_operand(body, dst, None)?,
                            count: self.convert_operand(body, count, None)?,
                        }
                    }
                },
            },
        })
    }

    /// The argument types of the body a constant callable operand
    /// names (a decoded function-item or closure constant): the
    /// callee's own declared argument locals, skipping a closure
    /// body's environment local. Empty when the operand is not
    /// such a constant or the body is not converted.
    fn constant_callee_param_tys(&self, func: &mir::Operand) -> Vec<mir::Ty> {
        let mir::Operand::Constant(c) = func else {
            return Vec::new();
        };
        let (def, env_locals) = match &c.literal {
            mir::ConstKind::Value(mir::ConstValue::Closure(def)) => (def, 1),
            mir::ConstKind::Value(mir::ConstValue::FnDef(def)) => (def, 0),
            _ => return Vec::new(),
        };
        let Some(key) = self.def_names.get(def) else {
            return Vec::new();
        };
        let Some(callee) = self.bodies.get(key) else {
            return Vec::new();
        };
        (1 + env_locals..=callee.arg_count)
            .filter_map(|i| callee.local_decls.get(i).map(|d| d.ty.clone()))
            .collect()
    }

    /// Whether the (unconverted) body's last declared parameter
    /// is a C variable argument list: the MIR-side form of
    /// [`callee_takes_va_list`], used while converting call
    /// arguments, before the callee's converted body exists.
    fn mir_body_takes_va_list(&self, body: &mir::Body) -> bool {
        if body.arg_count == 0 {
            return false;
        }
        match body.local_decls.get(body.arg_count).map(|d| &d.ty) {
            Some(mir::Ty::Adt(def, _)) => {
                let name = self.def_display_name(def);
                name == "std::ffi::VaList" || name == "core::ffi::VaList"
            }
            _ => false,
        }
    }

    pub(super) fn resolve_callee(&self, func: &mir::Operand) -> Option<String> {
        let mir::Operand::Constant(c) = func else {
            return None;
        };
        match &c.literal {
            mir::ConstKind::Ty(mir::Ty::FnDef(def, _)) => self.def_names.get(def).cloned(),
            mir::ConstKind::Unevaluated { def, .. } => self.def_names.get(def).cloned(),
            _ => None,
        }
    }

    /// Resolve a direct call to the program body that serves it,
    /// if any (the caller falls through to the `Fn`-trait and
    /// external/builtin shapes when this returns `None`).
    ///
    /// The ordinary case is `resolve_callee`: the collected body
    /// the operand names. When that body failed conversion in the
    /// previous fixpoint round, calling it can only fail with its
    /// recorded reason, so the call is served another way when it
    /// honestly can be:
    ///
    /// * If the call names that very item — the shape of a trait
    ///   call whose collected body is the trait item's default
    ///   implementation — a clean impl body for the call's `Self`
    ///   type is the faithful target (it is what native execution
    ///   dispatches to), exactly as when the trait item has no
    ///   collected body at all.
    /// * Otherwise, when the builtin model serves the callee, the
    ///   call falls through to the external/builtin conversion
    ///   below — the same treatment a callee the driver never
    ///   collected receives. The interpreter's builtin is the
    ///   model of that very function, so the value computed is
    ///   the function's own semantics, not an approximation of a
    ///   different one.
    ///
    /// When neither applies, the unconverted body still serves
    /// the call, so running it raises the recorded conversion
    /// failure (and a `dyn` receiver still dispatches through the
    /// vtable at run time): an explicit error, never a silently
    /// different result.
    fn resolve_body_callee(&self, func: &mir::Operand) -> Option<String> {
        let Some(name) = self.resolve_callee(func) else {
            // The std allocation entry points have no collected
            // body (the builtin model pre-empts them), so they
            // never resolve here; a program with a
            // `#[global_allocator]` binds them to its own shims
            // instead, exactly as the linker binds the symbols
            // in a native binary.
            if let Some(key) = self.alloc_shim_route(func) {
                return Some(key);
            }
            // An extern declaration whose symbol an in-program
            // definition exports binds to that definition's
            // body, exactly as the linker binds it in a native
            // binary (see `link_symbol_route`).
            if let Some(key) = self.link_symbol_route(func) {
                return Some(key);
            }
            return self.resolve_impl_callee(func);
        };
        // A `fmt` trait method call on a foreign `Self` type is
        // not bound to a body, even when one is stored under
        // the trait item's plain name: that body is a foreign
        // impl written against the real value layout, which
        // the value model does not reproduce. The call falls
        // through to the external/builtin conversion, whose
        // structural fmt engine is the faithful execution for
        // those types (see `fmt_call_self`).
        if self.fmt_foreign_self(&name, func) {
            return None;
        }
        // The in-place collect entry point (`std::vec::
        // in_place_collect::from_iter_in_place`) reaches the
        // program whenever a collected body materializes it as a
        // function-pointer constant — `SpecFromIter::from_iter`
        // calls it through exactly that shape, which the driver's
        // name-based collection never sees as a call target, so
        // the const-fn demand pass converts its body. That body
        // drives the raw allocation machinery the value model
        // does not serve and bottoms out in leaf specialization
        // calls (`SourceIter::as_inner` and friends) that have
        // no bodies at all. The builtin is the function's own
        // semantics in the model — drain the iterator into a
        // fresh `Vec`; the allocation reuse the real body
        // performs is unobservable — so it pre-empts the
        // collected body here, exactly as the registry pre-empts
        // collection for directly named calls. The same holds
        // for the nested specialization entry the builtin also
        // serves. The one exception is a body-served iterator
        // (see `body_served_iterator_arg`): the builtin's
        // drain drives the modelled `next` state machine,
        // which cannot step a hash table or a concrete
        // collection iterator, so for those sources the
        // collected body chain is the only faithful execution
        // and the pre-emption is lifted.
        if Self::vec_from_iter_nested_entry(&name)
            && !self.body_served_iterator_arg(func)
        {
            return None;
        }
        // An iterator consumer (`count` / `fold` / `for_each` /
        // `collect` / `nth`) over a type the modelled `next`
        // state machine serves is not bound to a body either,
        // even when one is stored under the trait item's plain
        // name: that body is core's default or specialised
        // implementation, whose pointer arithmetic over the
        // remaining items the value model does not reproduce
        // (a zero-sized element's remaining count is carried in
        // the end pointer's own bits), while the builtin drives
        // the same state machine `next` is served by, so the
        // two accountings cannot disagree about how many items
        // a sequence of calls consumed.
        if self.consumer_builtin_preempts(func) {
            return None;
        }
        let Some(plain) = self.unconverted.get(&name) else {
            // A name that is not itself a collected body denotes
            // a trait item the call names directly (the func
            // operand's `FnDef` is the trait method, with the
            // concrete `Self` among its generic arguments). A
            // clean collected impl body for that `Self` is the
            // faithful target — the same resolution
            // `resolve_impl_callee` performs for operand shapes
            // `resolve_callee` cannot name — and wins over the
            // name-based builtin model the run time would
            // otherwise serve the trait name with (a structural
            // `PartialEq` in place of a hand-written `eq`, say).
            // When no such body is clean, the name stands and
            // the existing fallbacks apply unchanged.
            if !self.bodies.contains_key(&name) {
                if let Some(key) = self.resolve_impl_callee(func) {
                    return Some(key);
                }
            }
            return Some(name);
        };
        if self
            .resolve_external_callee(func)
            .is_some_and(|(item, _, _)| item == *plain)
        {
            if let Some(key) = self.resolve_impl_callee(func) {
                return Some(key);
            }
        }
        if self.external_builtin(func) {
            return None;
        }
        Some(name)
    }

    /// Whether any of the call's generic arguments is an
    /// iterator type the modelled consumers cannot drain: a
    /// hash-collection iterator or one of the concrete
    /// collection iterators whose collected `next` bodies are
    /// the faithful execution (see
    /// `builtins::collection_iterator_tag`). The in-place /
    /// nested collect entries are pre-empted by the builtin
    /// only for iterators the state machine serves; for a
    /// body-served iterator the entry's own body chain --
    /// which steps the source through its collected `next`
    /// body -- is the faithful execution, and the builtin's
    /// drain would fail on the first step.
    fn body_served_iterator_arg(&self, func: &mir::Operand) -> bool {
        let Some((_, _, raw_generic_args)) = self.resolve_external_callee(func)
        else {
            return false;
        };
        raw_generic_args.iter().any(|ty| {
            let tag = crate::builtins::ty_tag(ty, self.names);
            crate::builtins::hash_iterator_tag(&tag)
                || crate::builtins::collection_iterator_tag(&tag)
        })
    }

    /// The guest allocation shim serving a call to one of the
    /// std allocation entry points, when the program defines a
    /// `#[global_allocator]` (see `alloc_shims`). A shim that
    /// itself failed conversion does not serve the call; the
    /// builtin model below does.
    fn alloc_shim_route(&self, func: &mir::Operand) -> Option<String> {
        let shim = self.alloc_shim_key(func)?;
        if self.unconverted.contains_key(&shim) {
            return None;
        }
        Some(shim)
    }

    /// The program key of the allocation shim a call to a std
    /// allocation entry point names, when the program defines a
    /// shim for it: the key whose membership in the round's
    /// unconverted set `alloc_shim_route` tests.
    pub(super) fn alloc_shim_key(&self, func: &mir::Operand) -> Option<String> {
        let mir::Operand::Constant(c) = func else {
            return None;
        };
        let def = match &c.literal {
            mir::ConstKind::Ty(mir::Ty::FnDef(def, _)) => def,
            mir::ConstKind::Unevaluated { def, .. } => def,
            _ => return None,
        };
        let name = self.names.get(def)?;
        let symbol = match name.as_str() {
            "alloc::alloc::__rust_alloc" => "__rust_alloc",
            "alloc::alloc::__rust_dealloc" => "__rust_dealloc",
            "alloc::alloc::__rust_realloc" => "__rust_realloc",
            "alloc::alloc::__rust_alloc_zeroed" => "__rust_alloc_zeroed",
            _ => return None,
        };
        self.alloc_shims.get(symbol).cloned()
    }

    /// The body serving a call to a body-less extern
    /// declaration, when exactly one converted body exports
    /// the declaration's link symbol: the in-program
    /// `#[no_mangle]` / `#[export_name]` definition a native
    /// link would bind the declaration to (its own crate's, or
    /// an auxiliary crate's). A body that failed conversion in
    /// the previous round does not serve the call, the same
    /// rule the allocation shims follow.
    fn link_symbol_route(&self, func: &mir::Operand) -> Option<String> {
        let key = self.link_symbol_key(func)?;
        if self.unconverted.contains_key(&key) {
            return None;
        }
        Some(key)
    }

    /// The program key of the in-program definition a call to a
    /// body-less extern declaration binds to, when one exports the
    /// declaration's link symbol: the key whose membership in the
    /// round's unconverted set `link_symbol_route` tests.
    pub(super) fn link_symbol_key(&self, func: &mir::Operand) -> Option<String> {
        let mir::Operand::Constant(c) = func else {
            return None;
        };
        let def = match &c.literal {
            mir::ConstKind::Ty(mir::Ty::FnDef(def, _)) => def,
            mir::ConstKind::Unevaluated { def, .. } => def,
            _ => return None,
        };
        let symbol = self.link_symbols.get(def)?;
        self.link_exports.get(symbol).cloned()
    }

    /// Whether the external/builtin conversion below would serve
    /// this call with a builtin: the callee's canonical name has a
    /// builtin model. Checked exactly as that conversion checks
    /// it, so a call is only ever diverted to a builtin that the
    /// external path would really produce.
    ///
    /// The intrinsic fallback (`builtins::intrinsic_lookup`) is
    /// deliberately not part of this predicate. This predicate is
    /// consulted only for a callee that HAS a body (one that
    /// failed conversion; see `resolve_body_callee`), and a
    /// compiler intrinsic never has a body, so such a callee is
    /// an ordinary function that merely shares an intrinsic's
    /// name: serving it with the intrinsic's builtin would
    /// silently replace its semantics instead of reporting its
    /// recorded conversion failure. The intrinsic fallback
    /// applies only where the callee is already known to have no
    /// body at all (the external conversion below and the run
    /// time's name dispatch).
    fn external_builtin(&self, func: &mir::Operand) -> bool {
        self.resolve_external_callee(func).is_some_and(|(name, _, _)| {
            let canonical = crate::builtins::canonical_name(&name);
            crate::builtins::lookup_qualified(&canonical).is_some()
                || crate::builtins::foreign_helper_lookup(&canonical).is_some()
        })
    }

    /// Whether the external/builtin conversion below would
    /// serve this call with one of the iterator-consumer
    /// builtins (`count` / `fold` / `for_each` / `collect` /
    /// `nth`) over an iterator type the modelled `next` state
    /// machine serves — or with the modelled
    /// `ExactSizeIterator::len` over a slice-state iterator:
    /// those calls are the builtin's, even when a body is
    /// stored under the trait item's name (see
    /// `resolve_body_callee`).
    fn consumer_builtin_preempts(&self, func: &mir::Operand) -> bool {
        let Some((name, _, raw_generic_args)) = self.resolve_external_callee(func)
        else {
            return false;
        };
        let canonical = crate::builtins::canonical_name(&name);
        let Some(kind) = crate::builtins::lookup_qualified(&canonical) else {
            return false;
        };
        let iter_ty = match kind {
            crate::builtins::Builtin::IteratorCount
            | crate::builtins::Builtin::IteratorFold
            | crate::builtins::Builtin::IteratorTryFold
            | crate::builtins::Builtin::IteratorForEach
            | crate::builtins::Builtin::IteratorCollect
            | crate::builtins::Builtin::IteratorNth
            | crate::builtins::Builtin::IteratorLen => raw_generic_args.first(),
            _ => return false,
        };
        let Some(iter_ty) = iter_ty else {
            return false;
        };
        // A `collect` into a hash or B-tree collection is not
        // pre-empted: the builtin can only build the target
        // shapes it models natively -- a hash table or a
        // sorted tree is not one of them -- while the
        // collected instance bodies (the trait default
        // delegating to the collection's own `FromIterator`
        // body: `with_hasher` + `extend` over the raw-table
        // memory model for a hash target, collect into a
        // `Vec` + sort + the bulk bottom-up build for a
        // B-tree target) execute faithfully for any iterator,
        // modelled or not. The target is the call's second
        // generic argument (the first is the iterator).
        if kind == crate::builtins::Builtin::IteratorCollect {
            if let Some(target_ty) = raw_generic_args.get(1) {
                let target_tag = crate::builtins::ty_tag(target_ty, self.names);
                if crate::builtins::hash_collection_tag(&target_tag)
                    || crate::builtins::btree_collection_tag(&target_tag)
                {
                    return false;
                }
            }
        }
        let tag = crate::builtins::ty_tag(iter_ty, self.names);
        match kind {
            crate::builtins::Builtin::IteratorLen => crate::builtins::iterator_len_modelled(&tag),
            _ => crate::builtins::iterator_consumer_modelled(&tag),
        }
    }

    /// Whether a call to one of the iterator-consumer builtins
    /// (`Iterator::count` / `fold` / `for_each` / `collect`,
    /// `FromIterator::from_iter`) should be served by the
    /// collected instance body instead of the builtin: the
    /// builtins drive the modelled `next` state machine, which
    /// only the builtin-produced iterator states satisfy. For
    /// any other iterator type — adapters such as `Map` /
    /// `Filter`, whose values are genuine structs built by
    /// collected constructors — the instance body, including
    /// its specializations, is the faithful execution, so it
    /// wins whenever the program has one.
    fn consumer_prefers_body(
        &self,
        kind: crate::builtins::Builtin,
        func: &mir::Operand,
        raw_generic_args: &[mir::Ty],
    ) -> bool {
        let iter_ty = match kind {
            crate::builtins::Builtin::IteratorCount
            | crate::builtins::Builtin::IteratorFold
            | crate::builtins::Builtin::IteratorTryFold
            | crate::builtins::Builtin::IteratorForEach
            | crate::builtins::Builtin::IteratorCollect => raw_generic_args.first(),
            crate::builtins::Builtin::FromIteratorFromIter => raw_generic_args.get(2),
            crate::builtins::Builtin::IteratorNextBack => raw_generic_args.first(),
            crate::builtins::Builtin::IteratorNext => raw_generic_args.first(),
            crate::builtins::Builtin::IteratorNth => raw_generic_args.first(),
            crate::builtins::Builtin::IteratorLen => raw_generic_args.first(),
            crate::builtins::Builtin::IntoIter => raw_generic_args.first(),
            _ => return false,
        };
        let Some(iter_ty) = iter_ty else {
            return false;
        };
        let tag = crate::builtins::ty_tag(iter_ty, self.names);
        // Hash-collection iteration is the inverse of the
        // consumer rule below: the builtin models many iterator
        // types, but a raw table's walk is not one of them, so
        // the body is preferred exactly for the hash shapes.
        match kind {
            crate::builtins::Builtin::IteratorNext => {
                return (crate::builtins::hash_iterator_tag(&tag)
                    || crate::builtins::collection_iterator_tag(&tag)
                    || crate::builtins::btree_iterator_tag(&tag))
                    && self.resolve_body_callee(func).is_some();
            }
            crate::builtins::Builtin::IntoIter => {
                return (crate::builtins::hash_collection_tag(&tag)
                    || crate::builtins::collection_container_tag(&tag))
                    && self.resolve_body_callee(func).is_some();
            }
            _ => {}
        }
        // Hash- and B-tree-collection targets of `collect` /
        // `from_iter` are the same inversion one level up: the
        // builtin drains the iterator itself and can only
        // build the target shapes it models natively -- a hash
        // table or a sorted tree is not one of them -- while
        // the instance bodies (for a hash target,
        // `with_hasher` + `extend` over the raw-table memory
        // model; for a B-tree target, collect into a `Vec` +
        // sort + the bulk bottom-up build) execute faithfully
        // for any iterator, modelled or not, so the body wins
        // whenever the program has one. The target is the
        // call's second generic argument for `collect` (whose
        // first is the iterator) and its first for `from_iter`
        // (whose first is the collection itself).
        if matches!(
            kind,
            crate::builtins::Builtin::IteratorCollect
                | crate::builtins::Builtin::FromIteratorFromIter
        ) {
            let target_ty = match kind {
                crate::builtins::Builtin::IteratorCollect => raw_generic_args.get(1),
                _ => raw_generic_args.first(),
            };
            if let Some(target_ty) = target_ty {
                let target_tag = crate::builtins::ty_tag(target_ty, self.names);
                if crate::builtins::hash_collection_tag(&target_tag)
                    || crate::builtins::btree_collection_tag(&target_tag)
                {
                    return self.resolve_body_callee(func).is_some();
                }
            }
        }
        let modelled = match kind {
            crate::builtins::Builtin::IteratorNextBack => {
                crate::builtins::iterator_next_back_modelled(&tag)
            }
            crate::builtins::Builtin::IteratorCount
            | crate::builtins::Builtin::IteratorFold
            | crate::builtins::Builtin::IteratorTryFold
            | crate::builtins::Builtin::IteratorForEach
            | crate::builtins::Builtin::IteratorCollect
            | crate::builtins::Builtin::IteratorNth => {
                crate::builtins::iterator_consumer_modelled(&tag)
            }
            crate::builtins::Builtin::IteratorLen => crate::builtins::iterator_len_modelled(&tag),
            _ => crate::builtins::iterator_next_modelled(&tag),
        };
        if modelled {
            return false;
        }
        self.resolve_body_callee(func).is_some()
    }

    /// Resolve a trait-method call whose func operand names the
    /// trait item itself (which has no body) to the concrete impl
    /// body the driver collected for the call's `Self` type, if the
    /// program has one.
    ///
    /// MIR names such a call by the trait item (`std::ops::Index::
    /// index` with generic arguments `[Self, Idx]`), while the
    /// collected body is keyed by its instance name
    /// (`<S as std::ops::Index<usize>>::index`). Without this step
    /// the call falls through to the name-based builtin model of
    /// the trait method, which is only faithful for receivers whose
    /// impl bodies are not in the program (core's slice/array impls,
    /// `Vec`): for a user-defined impl it silently computes the
    /// builtin's semantics on the receiver instead — e.g. an
    /// `Index` impl returning `&str` was executed as a slice index
    /// into the receiver struct and panicked with a bogus
    /// out-of-bounds message. A collected body, when one matches,
    /// is the faithful semantics and wins over the builtin, exactly
    /// as a `DefId`-resolved body does in `resolve_callee`.
    fn resolve_impl_callee(&self, func: &mir::Operand) -> Option<String> {
        // Resolution runs only in the joint-fixpoint rounds of
        // `convert_all`, against the previous round's clean
        // bodies (see `clean_targets`): resolving to an impl body that cannot
        // itself convert — or that calls, however indirectly, a
        // body that cannot — would trade a working builtin model
        // of the trait method for an unconverted-body error. A
        // derived `PartialEq` on a packed struct is the shape
        // that pins this restriction: every eq body converts, but
        // the manual impl calls a helper whose pointer cast the
        // conversion leaves unsupported, while the builtin's
        // structural comparison is exact.
        let targets = self.impl_targets.as_ref()?;
        let key = self.find_impl_callee(func)?;
        targets.contains(&key).then_some(key)
    }

    /// Find the collected impl body for a trait-method call (see
    /// `resolve_impl_callee` for when a found body may be used).
    /// When a callee name denotes a `fmt` trait method (the
    /// `Display`/`Debug`/radix family), the `Self` type the
    /// call is monomorphized at, in rendered form. The name
    /// comes in three shapes: the trait item
    /// (`std::fmt::Debug::fmt`, whose `Self` is the call's
    /// first generic argument), the resolved impl instance
    /// (`<std::vec::Vec<i32> as std::fmt::Debug>::fmt`, whose
    /// `Self` is the key's subject), and the module-qualified
    /// impl instance (`core::fmt::num::<impl
    /// std::fmt::Debug for u32>::fmt`, whose `Self` is the
    /// type the impl header's `for` introduces). Foreign impl bodies of
    /// these methods are written against the real value layout
    /// (a `Vec`'s `Debug` reads its buffer pointer out of the
    /// `RawVec`), which the value model does not reproduce, so
    /// callers use this to keep such a call off any collected
    /// body: the structural fmt engine — the same engine that
    /// formats the type as an argument — serves it instead
    /// (see `builtins::fmt_trait_method`). A guest type's own
    /// `fmt` body stays the resolution, exactly as before.
    /// Whether a callee is a `fmt` trait method whose `Self`
    /// the structural fmt engine must serve: a foreign type,
    /// or a `dyn` trait object (whose concrete value the
    /// engine resolves through the fat reference, exactly as
    /// when the object is formatted as an argument). A guest
    /// type's own `fmt` body stays the resolution.
    fn fmt_foreign_self(&self, name: &str, func: &mir::Operand) -> bool {
        let Some(self_name) = self.fmt_call_self(name, func) else {
            return false;
        };
        let head = self_name.trim().trim_start_matches('&').trim_start();
        head.starts_with("dyn ") || !crate::builtins::guest_fmt_tag(&self_name)
    }

    fn fmt_call_self(&self, name: &str, func: &mir::Operand) -> Option<String> {
        if crate::builtins::fmt_trait_method(name).is_some() {
            let (_, _, raw_generic_args) = self.resolve_external_callee(func)?;
            return raw_generic_args
                .first()
                .map(|t| self.concrete_ty_name(t));
        }
        if let Some(self_part) = Self::impl_header_self(name) {
            return Some(self_part);
        }
        let rest = name.strip_prefix('<')?;
        let (self_part, trait_part) = rest.rsplit_once(" as ")?;
        let trait_path = trait_part.strip_suffix(">::fmt")?;
        if crate::builtins::fmt_trait_method(&format!("{trait_path}::fmt")).is_none() {
            return None;
        }
        Some(self_part.to_string())
    }

    /// The `Self` type named by a callee in the module-qualified
    /// impl-instance shape (`core::fmt::num::<impl
    /// std::fmt::Debug for u32>::fmt`): the type the impl
    /// header's `for` introduces. An impl collected from a
    /// foreign crate's module is keyed this way rather than by
    /// the `<Self as Trait>` subject shape, and the integer
    /// `fmt` impls in `core::fmt::num` — whose bodies read the
    /// `Formatter`'s flag word directly — reach a guest body
    /// forwarding to a field's `fmt` under exactly this name.
    /// `None` when the name is not that shape or the header's
    /// trait is not a `fmt` trait.
    fn impl_header_self(name: &str) -> Option<String> {
        // The `>::fmt` strip consumes the impl bracket's own
        // closing `>` (it directly precedes `::fmt`), so the
        // header tail is exactly `<Trait> for <Type>` with the
        // type's own brackets intact.
        let inner = name.strip_suffix(">::fmt")?.rsplit_once("::<impl ")?.1;
        let (trait_path, self_part) = inner.rsplit_once(" for ")?;
        if crate::builtins::fmt_trait_method(&format!("{trait_path}::fmt")).is_none() {
            return None;
        }
        Some(self_part.to_string())
    }

    pub(super) fn find_impl_callee(&self, func: &mir::Operand) -> Option<String> {
        let (name, _, raw_generic_args) = self.resolve_external_callee(func)?;
        let (trait_path, method) = name.rsplit_once("::")?;
        if trait_path.is_empty() || method.is_empty() {
            return None;
        }
        // A `fmt` trait method on a foreign `Self` type is not
        // resolved to its collected impl body either, for the
        // reason `fmt_call_self` documents: the structural fmt
        // engine serves the call instead.
        if method == "fmt" && self.fmt_foreign_self(&name, func) {
            return None;
        }
        self.find_impl_key_for(trait_path, method, &raw_generic_args)
    }

    /// The collected impl body key for a trait method named by
    /// its trait path and method name, instantiated with
    /// `raw_generic_args` (whose first entry is the `Self`
    /// type): the resolution `find_impl_callee` performs for
    /// calls, shared with the `FnDef`-constant resolution in
    /// `fn_def_value_key`.
    pub(super) fn find_impl_key_for(
        &self,
        trait_path: &str,
        method: &str,
        raw_generic_args: &[mir::Ty],
    ) -> Option<String> {
        let self_ty = raw_generic_args.first()?;
        let self_name = self.concrete_ty_name(self_ty);
        // The impl's instance name spells the trait path as the
        // impl wrote it: the call may name the trait through `std::`
        // while the impl was recorded through `core::`, or the
        // other way round, so try both spellings.
        let mut paths = vec![trait_path.to_string()];
        if let Some(rest) = trait_path.strip_prefix("std::") {
            paths.push(format!("core::{rest}"));
        } else if let Some(rest) = trait_path.strip_prefix("core::") {
            paths.push(format!("std::{rest}"));
        }
        // The trait's own generic arguments (the func operand's
        // arguments after `Self`), rendered as they appear in the
        // impl name: `<S as std::ops::Index<usize>>::index`.
        let trait_args: Vec<String> = raw_generic_args[1..]
            .iter()
            .map(|t| crate::builtins::ty_tag(t, self.names))
            .collect();
        for path in &paths {
            if !trait_args.is_empty() {
                let candidate = format!(
                    "<{self_name} as {path}<{}>>::{method}",
                    trait_args.join(", ")
                );
                if let Some(key) = self.find_body_key(&candidate) {
                    return Some(key);
                }
            }
            let candidate = format!("<{self_name} as {path}>::{method}");
            if let Some(key) = self.find_body_key(&candidate) {
                return Some(key);
            }
        }
        // Fall back to a scan: exactly one impl body for this
        // `Self` type and trait, under any generic-argument
        // spelling (a defaulted trait argument, for one, is
        // elided from the instance name but present in the call's
        // generic arguments). An ambiguous scan resolves nothing.
        for path in &paths {
            if let Some(key) = self.scan_impl_key(&self_name, path, method) {
                return Some(key);
            }
        }
        None
    }

    /// The program key of a body stored under exactly `candidate`,
    /// or under its `DefId`-disambiguated form (`candidate#DefId`,
    /// see `program_keys`) when that form is unique.
    fn find_body_key(&self, candidate: &str) -> Option<String> {
        if self.bodies.contains_key(candidate) {
            return Some(candidate.to_string());
        }
        let mut matches = self.bodies.keys().filter(|k| {
            k.strip_prefix(candidate)
                .is_some_and(|rest| rest.starts_with('#'))
        });
        match (matches.next(), matches.next()) {
            (Some(key), None) => Some(key.clone()),
            _ => None,
        }
    }

    /// The unique body keyed as an impl of `trait_path` for
    /// `self_name` with method `method`, under any generic-argument
    /// spelling of the trait; `None` when there is not exactly one.
    fn scan_impl_key(&self, self_name: &str, trait_path: &str, method: &str) -> Option<String> {
        let mut found: Option<&String> = None;
        for key in self.bodies.keys() {
            let base = key.split('#').next().unwrap_or(key.as_str());
            let Some((self_part, trait_base, key_method)) =
                parse_impl_key(base).or_else(|| parse_impl_block_key(base))
            else {
                continue;
            };
            if key_method != method || trait_base != trait_path {
                continue;
            }
            if !impl_self_matches(&self_part, self_name) {
                continue;
            }
            if found.is_some() {
                return None;
            }
            found = Some(key);
        }
        found.cloned()
    }

    /// Resolve a callee that has no converted body through the
    /// driver's full name table (std/core functions): its display
    /// name, its rendered generic arguments, and the raw generic
    /// arguments (used to type literal arguments whose type the
    /// callee's signature pins to a generic parameter).
    /// The `DefId` a call's func operand names, when the operand
    /// is a direct function constant (the same shapes
    /// `resolve_callee` accepts).
    fn callee_def_id(&self, func: &mir::Operand) -> Option<mir::DefId> {
        let mir::Operand::Constant(c) = func else {
            return None;
        };
        match &c.literal {
            mir::ConstKind::Ty(mir::Ty::FnDef(def, _)) => Some(*def),
            mir::ConstKind::Unevaluated { def, .. } => Some(*def),
            _ => None,
        }
    }

    pub(super) fn resolve_external_callee(
        &self,
        func: &mir::Operand,
    ) -> Option<(String, Vec<String>, Vec<mir::Ty>)> {
        let mir::Operand::Constant(c) = func else {
            return None;
        };
        match &c.literal {
            mir::ConstKind::Ty(mir::Ty::FnDef(def, generic_args)) => {
                let name = self.names.get(def).cloned()?;
                let tags = generic_args
                    .iter()
                    .map(|t| crate::builtins::ty_tag(t, self.names))
                    .collect();
                Some((name, tags, generic_args.clone()))
            }
            _ => None,
        }
    }

    /// Whether `name` (a body key, possibly carrying an
    /// instance suffix) denotes one of the `Vec` collect
    /// specialization entries the `VecFromIterNested` builtin
    /// serves: the registry-named entries (the nested trait
    /// entry and the in-place entry, under either crate
    /// spelling), or the nested entry's impl-qualified form
    /// (`<std::vec::Vec<T> as std::vec::spec_from_iter_nested::
    /// SpecFromIterNested<T, I>>::from_iter`), which is how the
    /// collected impl body is keyed. The recognition of the
    /// impl-qualified form mirrors the interpreter's run-time
    /// name fallback for the same entry.
    fn vec_from_iter_nested_entry(name: &str) -> bool {
        Self::vec_from_iter_builtin_name(name).is_some()
    }

    /// The name a `VecFromIterNested` call for the entry
    /// `name` denotes is served under: the entry's own
    /// canonical name when the registry names it (the nested
    /// trait entry and the in-place entry), and the nested
    /// trait entry's name for the impl-qualified form, whose
    /// own spelling the registry does not carry — the
    /// interpreter resolves a builtin call by its name, so the
    /// call must carry a name the registry resolves.
    fn vec_from_iter_builtin_name(name: &str) -> Option<String> {
        let plain = name.split(" [").next().unwrap_or(name);
        let canonical = crate::builtins::canonical_name(plain);
        if matches!(
            crate::builtins::lookup_qualified(&canonical),
            Some(crate::builtins::Builtin::VecFromIterNested)
        ) {
            return Some(canonical);
        }
        if canonical.ends_with(">::from_iter")
            && canonical
                .contains("::spec_from_iter_nested::SpecFromIterNested")
            && crate::interp::impl_qualified_self_type(&canonical)
                .is_some_and(|target| {
                    target.starts_with("std::vec::Vec<")
                        || target.starts_with("alloc::vec::Vec<")
                })
        {
            return Some(
                "std::vec::spec_from_iter_nested::SpecFromIterNested::from_iter"
                    .to_string(),
            );
        }
        None
    }

    /// A call through a function-pointer constant naming one
    /// of the `Vec` collect specialization entries
    /// (`std::vec::in_place_collect::from_iter_in_place`, or
    /// the `SpecFromIterNested` impl): `SpecFromIter::from_iter`
    /// calls them through exactly this shape — the func operand
    /// is the decoded function-pointer value, not an
    /// `FnDef`-typed constant — so neither the driver's
    /// name-based collection nor the direct-call resolution
    /// ever sees the callee's name, and the call would
    /// otherwise dispatch through the value to the collected
    /// body, whose raw-allocation machinery the value model
    /// does not serve. The builtin is the entry's own semantics
    /// in the model (drain the iterator into a fresh `Vec`; the
    /// allocation management is unobservable), so the call
    /// converts to it directly. The stepping dispatch key is
    /// the iterator argument's own type, which is the entry's
    /// last generic argument.
    fn vec_from_iter_value_terminator(
        &self,
        body: &mir::Body,
        func: &mir::Operand,
        args: &[mir::Operand],
        destination: &mir::Place,
        target: Option<mir::BasicBlock>,
    ) -> Result<Option<ir::TerminatorKind>, ConvertError> {
        let mir::Operand::Constant(c) = func else {
            return Ok(None);
        };
        let mir::ConstKind::Value(mir::ConstValue::FnDef(def)) = &c.literal
        else {
            return Ok(None);
        };
        let Some(name) = self.def_names.get(def) else {
            return Ok(None);
        };
        let Some(canonical) = Self::vec_from_iter_builtin_name(name)
        else {
            return Ok(None);
        };
        if args.len() != 1 {
            return Ok(None);
        }
        let Some(iter_ty) = self.operand_place_ty(body, &args[0]) else {
            return Ok(None);
        };
        let iter_tag = crate::builtins::ty_tag(&iter_ty, self.names);
        let ir_arg = self.convert_operand(body, &args[0], Some(&iter_ty))?;
        Ok(Some(ir::TerminatorKind::CallBuiltin {
            name: canonical,
            generic_args: vec![iter_tag],
            args: vec![ir_arg],
            destination: self.convert_place(body, destination)?,
            target: target.map(|b| ir::BasicBlock(b.index())),
            drop_glue: None,
            consumed_glue: None,
            skipped_glue: None,
            elem_glue: None,
            receiver_glue: None,
            cmp_discriminants: None,
            result_unsize: None,
            unique_elem: None,
            zeroed_value: None,
        }))
    }

    /// A direct `Add::add` / `Sub::sub` call whose Self type is
    /// one of the modelled time types converts to a builtin node:
    /// the `Instant` / `SystemTime` instances have no body at
    /// all, and the collected `Duration` instance body does not
    /// run in this model (it writes a raw nanos count into the
    /// niche-typed `Nanoseconds` field). Every other `Add` /
    /// `Sub` call keeps the normal paths (a collected body wins;
    /// the names are not in the builtin registry, so reach still
    /// collects the generic shim bodies).
    fn time_add_sub_terminator(
        &self,
        body: &mir::Body,
        func: &mir::Operand,
        args: &[mir::Operand],
        destination: &mir::Place,
        target: Option<mir::BasicBlock>,
    ) -> Result<Option<ir::TerminatorKind>, ConvertError> {
        let Some((name, generic_args, _raw)) = self.resolve_external_callee(func) else {
            return Ok(None);
        };
        let canonical = crate::builtins::canonical_name(&name);
        if canonical != "std::ops::Add::add" && canonical != "std::ops::Sub::sub" {
            return Ok(None);
        }
        let Some(self_ty) = generic_args.first() else {
            return Ok(None);
        };
        if !matches!(
            self_ty.as_str(),
            "std::time::Instant" | "std::time::SystemTime" | "std::time::Duration"
        ) {
            return Ok(None);
        }
        let mut ir_args = Vec::with_capacity(args.len());
        for arg in args.iter() {
            ir_args.push(self.convert_operand(body, arg, None)?);
        }
        Ok(Some(ir::TerminatorKind::CallBuiltin {
            name: canonical,
            generic_args: vec![self_ty.clone()],
            args: ir_args,
            destination: self.convert_place(body, destination)?,
            target: target.map(|b| ir::BasicBlock(b.index())),
            drop_glue: None,
            consumed_glue: None,
            skipped_glue: None,
            elem_glue: None,
            receiver_glue: None,
            cmp_discriminants: None,
            result_unsize: None,
            unique_elem: None,
            zeroed_value: None,
        }))
    }

    /// A direct call to a `std::io::Read` / `Write` / `Seek`
    /// method whose Self type is `std::fs::File` (or a
    /// child-process pipe handle, whose value is a file-table
    /// id like `File`'s), or to one of `File`'s inherent handle
    /// methods, converts to a builtin node: those operations act
    /// on the interpreter's host file table (see `exec_file_io`
    /// in `builtins`). The trait names are deliberately absent
    /// from the builtin registry, so reach still collects the
    /// other receivers' bodies (a `&[u8]` reader's `read`, a
    /// `Cursor`'s `seek`); only a call whose Self is exactly
    /// `File` (or a reference to it) or a pipe handle is
    /// diverted here. The inherent names are `File`'s own, so
    /// no Self check is needed for them.
    fn file_io_terminator(
        &self,
        body: &mir::Body,
        func: &mir::Operand,
        args: &[mir::Operand],
        destination: &mir::Place,
        target: Option<mir::BasicBlock>,
    ) -> Result<Option<ir::TerminatorKind>, ConvertError> {
        let Some((name, generic_args, _raw)) = self.resolve_external_callee(func) else {
            return Ok(None);
        };
        let canonical = crate::builtins::canonical_name(&name);
        const TRAIT_METHODS: &[&str] = &[
            "std::io::Read::read",
            "std::io::Read::read_exact",
            "std::io::Read::read_to_end",
            "std::io::Read::read_to_string",
            "std::io::Read::read_buf",
            "std::io::Read::read_buf_exact",
            "std::io::Write::write",
            "std::io::Write::write_all",
            "std::io::Write::write_vectored",
            "std::io::Read::read_vectored",
            "std::io::Write::flush",
            "std::io::Seek::seek",
            "std::io::Seek::rewind",
            "std::io::Seek::stream_position",
            "std::io::BufRead::read_line",
            "std::io::BufRead::read_until",
        ];
        const INHERENT_METHODS: &[&str] = &[
            "std::fs::File::metadata",
            "std::fs::File::try_clone",
            "std::fs::File::sync_all",
            "std::fs::File::sync_data",
            "std::fs::File::set_len",
            "std::io::Stdin::read_line",
        ];
        let diverted = if TRAIT_METHODS.contains(&canonical.as_str()) {
            let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
            let self_ty = self_ty
                .strip_prefix("&mut ")
                .or_else(|| self_ty.strip_prefix("&"))
                .unwrap_or(self_ty);
            self_ty == "std::net::TcpStream"
                && matches!(
                    canonical.as_str(),
                    "std::io::Read::read"
                        | "std::io::Read::read_vectored"
                        | "std::io::Write::write"
                        | "std::io::Write::write_vectored"
                        | "std::io::Write::flush"
                )
                || self_ty == "std::fs::File"
                || (matches!(
                    canonical.as_str(),
                    "std::io::Write::write"
                        | "std::io::Write::write_all"
                        | "std::io::Write::flush"
                ) && (crate::builtins::stream_write_receiver(self_ty)
                    || self_ty == "std::process::ChildStdin"))
                || (matches!(
                    canonical.as_str(),
                    "std::io::Read::read"
                        | "std::io::Read::read_exact"
                        | "std::io::Read::read_to_end"
                        | "std::io::Read::read_to_string"
                        | "std::io::Read::read_buf"
                        | "std::io::Read::read_buf_exact"
                ) && matches!(
                    self_ty,
                    "std::process::ChildStdout"
                        | "std::process::ChildStderr"
                        | "std::io::Stdin"
                        | "std::io::StdinLock"
                ))
                || (matches!(
                    canonical.as_str(),
                    "std::io::BufRead::read_line"
                        | "std::io::BufRead::read_until"
                ) && matches!(
                    self_ty,
                    "std::io::Stdin" | "std::io::StdinLock"
                ))
        } else {
            INHERENT_METHODS.contains(&canonical.as_str())
        };
        if !diverted {
            return Ok(None);
        }
        let mut ir_args = Vec::with_capacity(args.len());
        for arg in args.iter() {
            ir_args.push(self.convert_operand(body, arg, None)?);
        }
        Ok(Some(ir::TerminatorKind::CallBuiltin {
            name: canonical,
            generic_args,
            args: ir_args,
            destination: self.convert_place(body, destination)?,
            target: target.map(|b| ir::BasicBlock(b.index())),
            drop_glue: None,
            consumed_glue: None,
            skipped_glue: None,
            elem_glue: None,
            receiver_glue: None,
            cmp_discriminants: None,
            result_unsize: None,
            unique_elem: None,
            zeroed_value: None,
        }))
    }

    /// Whether this call's func operand names an `Fn`-family trait
    /// method (`Fn::call`, `FnMut::call_mut`, `FnOnce::call_once` in
    /// their `std::ops` / `core::ops` spellings): returns the call
    /// kind and the callee's raw generic arguments, whose first two
    /// entries are the Self type and the argument tuple type (used
    /// to type the receiver and tuple operands).
    /// How a non-indirect call is dispatched when both the `Fn`
    /// trait interpretation and name-based resolution are in play.
    ///
    /// The direct candidate is `resolve_body_callee`'s, so the
    /// joint-fixpoint honesty rules (a clean impl body serves a call
    /// naming an unconverted trait item; a builtin-modelled callee
    /// diverts to the builtin path) apply before the shape judgment
    /// below. Name-based resolution wins when it names a genuine body
    /// for the call: an inherent `Fn` implementation on a struct value,
    /// or the trait method's own shim (environment plus the whole
    /// argument tuple) — both take exactly the call's arguments.
    /// The trait interpretation wins when there is no direct body,
    /// and when the body recorded under the trait name is a
    /// closure's own body collected through the trait-call instance
    /// (see [`body_is_closure_body`]): a direct call would pass
    /// (receiver, tuple) to a body of the wrong arity and
    /// convention, while the trait call evaluates the receiver as a
    /// callable value, which is the convention that body expects.
    /// Convert a direct `Drop::drop` call into a `Drop` terminator
    /// against the receiver's pointee place, when the call's shape
    /// allows it (see the call conversion): the callee names the
    /// `Drop` trait item (`std::ops::Drop::drop` / `core::ops::
    /// Drop::drop`) or a `Drop` impl item (`<T as std::ops::Drop>::
    /// drop`), there is exactly one argument (the `&mut Self`
    /// receiver), and the dropped type's glue can be built. The
    /// receiver place plus a `Deref` projection is the pointee
    /// place the glue runs against; running it clears that place
    /// (the glue machinery clears a direct local), so a following
    /// `Drop` terminator for the same place reads it as
    /// uninitialised and runs no glue a second time.
    ///
    /// A `Box` call is the exception in the other direction: the
    /// box's method body is only the shell step (it releases the
    /// allocation; the pointee's destruction is the compiler's,
    /// done before the method runs), and drop elaboration emits
    /// this call form exactly when the pointee has been moved
    /// out of the box — the parts that remain are dropped by
    /// their own `Drop` terminators. The glue the conversion
    /// builds for a box call is therefore the trivial plan: the
    /// session rooted at the box place still releases the
    /// allocation and clears the place, and no pointee glue runs.
    ///
    /// One exception: when the dropped type's own glue routes
    /// its destructor step through the very body this call
    /// resolves to, the collected body IS that step and the call
    /// keeps it — the surrounding body runs the fields' glue
    /// itself (a compiler `drop_glue` shim does, through its own
    /// `Drop` terminators), which the full-glue conversion would
    /// run a second time.
    ///
    /// A second exception: a `Box` call that resolves to a
    /// collected body keeps it. The glue builder plans a box
    /// structurally (the pointee's plan at field 0), so the glue
    /// names no destructor step for the box itself — but the
    /// box's `Drop::drop` body is not the pointee's destruction:
    /// it only releases the allocation (the compiler destroys
    /// the pointee before the method runs). An async-destructor
    /// coroutine calls it as exactly that final shell step,
    /// after the pointee's async destruction has completed;
    /// expanding the call to the full glue would destroy the
    /// pointee a second time. (Modelled containers such as
    /// `Vec` keep the glue conversion: their collected bodies
    /// operate on the native buffer representation the value
    /// model replaces, so the glue is their only faithful
    /// route.) A call whose glue cannot be built falls through
    /// to the ordinary call conversion.
    fn drop_call_terminator(
        &self,
        body: &mir::Body,
        func: &mir::Operand,
        args: &[mir::Operand],
        target: Option<mir::BasicBlock>,
    ) -> Option<ir::TerminatorKind> {
        if args.len() != 1 {
            return None;
        }
        let target = target?;
        let (name, _, raw_generic_args) = self.resolve_external_callee(func)?;
        let is_trait_item = matches!(
            name.as_str(),
            "std::ops::Drop::drop" | "core::ops::Drop::drop"
        );
        let is_impl_item = name.starts_with('<')
            && (name.ends_with(" as std::ops::Drop>::drop")
                || name.ends_with(" as core::ops::Drop>::drop"));
        if !is_trait_item && !is_impl_item {
            return None;
        }
        // Self type: the receiver argument's pointee type (the
        // argument is `&mut Self`), falling back to the call's
        // first generic argument.
        let self_ty = self
            .operand_place_ty(body, &args[0])
            .map(|t| match t {
                mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => *inner,
                other => other,
            })
            .or_else(|| raw_generic_args.first().cloned())?;
        // The `Box` shell-step exception (see the doc comment).
        if let mir::Ty::Adt(def, _) = &self_ty {
            let is_box = self
                .names
                .get(def)
                .or_else(|| self.def_names.get(def))
                .is_some_and(|n| n == "std::boxed::Box");
            if is_box && self.resolve_body_callee(func).is_some() {
                return None;
            }
        }
        let glue = self.drop_glue(&self_ty)?;
        // When the dropped type's own glue routes its destructor
        // step through a collected destructor body for this very
        // type, that body IS the step this call names (reached
        // directly, or through the trait item's dispatch body):
        // keep the ordinary call, whose execution runs the
        // destructor and nothing else. The surrounding body runs
        // the fields' glue itself (a compiler `drop_glue` shim
        // does, through its own `Drop` terminators), which the
        // full-glue conversion would run a second time. A type
        // whose glue the value model accounts for structurally
        // (a modelled container, a reference-counted handle)
        // names no such step, and the conversion below is its
        // faithful route.
        let dtor_key = match &self_ty {
            mir::Ty::Adt(def, args) => self
                .dtor_bodies
                .get(&format!("{self_ty:?}"))
                .cloned()
                .or_else(|| {
                    self.names
                        .get(def)
                        .or_else(|| self.def_names.get(def))
                        .and_then(|n| self.dtor_body(n, args))
                }),
            _ => None,
        };
        if let Some(key) = dtor_key {
            if glue_calls_dtor(&glue, &key) {
                return None;
            }
        }
        // The `Box` shell step (see the doc comment): the
        // method body only releases the allocation, so the glue
        // is the trivial plan — the session rooted at the box
        // place releases the allocation and clears the place,
        // and no pointee glue runs.
        let glue = if let mir::Ty::Adt(def, _) = &self_ty {
            let is_box = self
                .names
                .get(def)
                .or_else(|| self.def_names.get(def))
                .is_some_and(|n| n == "std::boxed::Box");
            if is_box {
                ir::DropGlue {
                    root: ir::DropPlan::Trivial,
                    adts: Vec::new(),
                }
            } else {
                glue
            }
        } else {
            glue
        };
        let arg_place = match &args[0] {
            mir::Operand::Copy(place) | mir::Operand::Move(place) => place.clone(),
            _ => return None,
        };
        let mut pointee = arg_place;
        pointee.projection.push(mir::PlaceElem::Deref);
        Some(ir::TerminatorKind::Drop {
            place: self.convert_place(body, &pointee).ok()?,
            target: ir::BasicBlock(target.index()),
            glue,
        })
    }

    fn fn_trait_or_direct(
        &self,
        func: &mir::Operand,
        arg_count: usize,
    ) -> Option<FnResolution> {
        // A call to an `Fn`-trait method takes the receiver and
        // the argument tuple; a third argument is the
        // caller-location stamp the driver appended for a
        // `#[track_caller]` shim instance (a track-caller
        // closure), which the `CallTrait` conversion carries
        // along instead of dropping the call.
        let trait_call = self
            .resolve_fn_trait_call(func)
            .filter(|_| arg_count == 2 || arg_count == 3);
        let stamped = arg_count == 3;
        let direct = self.resolve_body_callee(func);
        match (trait_call, direct) {
            (Some(trait_call), Some(name)) => {
                let mis_keyed = self
                    .bodies
                    .get(&name)
                    .is_some_and(|b| body_is_closure_body(b));
                // When the callable is a closure or a trait
                // object, the receiver value carries the
                // callable and the `CallTrait` dispatch owns
                // the call: a body registered under the trait
                // method's name is one instance's body (a shim,
                // a forwarding impl), not the dispatch, and the
                // interpreter's vtable dispatch refuses to treat
                // it as one. A concrete ADT `Self` instead names
                // a hand-written `Fn` impl, whose collected body
                // the direct resolution keeps.
                let value_keyed = trait_call
                    .1
                    .first()
                    .is_some_and(fn_self_is_closure_or_dyn)
                    // The `AsyncFn` family is implemented only by
                    // async closure values, so the callable is the
                    // receiver value itself. The check above cannot
                    // see it: the call's `Self` type is a
                    // coroutine-closure type, which the driver
                    // records as the unit placeholder (its kind is
                    // unaskable through Stable MIR), so the kind of
                    // the call is the witness instead.
                    || matches!(
                        trait_call.0,
                        ir::FnCallKind::AsyncCall
                            | ir::FnCallKind::AsyncCallMut
                            | ir::FnCallKind::AsyncCallOnce
                    );
                if mis_keyed || value_keyed {
                    Some(FnResolution::Trait(
                        trait_call.0,
                        trait_call.1,
                        stamped,
                    ))
                } else {
                    Some(FnResolution::Direct(name))
                }
            }
            (Some(trait_call), None) => {
                Some(FnResolution::Trait(
                    trait_call.0,
                    trait_call.1,
                    stamped,
                ))
            }
            (None, Some(name)) => Some(FnResolution::Direct(name)),
            (None, None) => None,
        }
    }

    fn resolve_fn_trait_call(
        &self,
        func: &mir::Operand,
    ) -> Option<(ir::FnCallKind, Vec<mir::Ty>)> {
        let (name, _, raw_generic_args) = self.resolve_external_callee(func)?;
        let canonical = crate::builtins::canonical_name(&name);
        let kind = match canonical.as_str() {
            "std::ops::Fn::call" | "core::ops::Fn::call" => ir::FnCallKind::Call,
            "std::ops::FnMut::call_mut" | "core::ops::FnMut::call_mut" => {
                ir::FnCallKind::CallMut
            }
            "std::ops::FnOnce::call_once" | "core::ops::FnOnce::call_once" => {
                ir::FnCallKind::CallOnce
            }
            "std::ops::AsyncFn::async_call" | "core::ops::AsyncFn::async_call" => {
                ir::FnCallKind::AsyncCall
            }
            "std::ops::AsyncFnMut::async_call_mut"
            | "core::ops::AsyncFnMut::async_call_mut" => ir::FnCallKind::AsyncCallMut,
            "std::ops::AsyncFnOnce::async_call_once"
            | "core::ops::AsyncFnOnce::async_call_once" => {
                ir::FnCallKind::AsyncCallOnce
            }
            _ => return None,
        };
        Some((kind, raw_generic_args))
    }

    /// Build the drop glue of the dropped type, if every part of
    /// it can be accounted for (see [`DropGlueBuilder`]).
    fn drop_glue(&self, ty: &mir::Ty) -> Option<ir::DropGlue> {
        DropGlueBuilder::build(self, ty)
    }

    /// The unsizing a `From` / `Into` conversion into a
    /// `Box<dyn Trait>` applies to its result, decided here where
    /// both types are known — exactly as for a sized-to-dyn
    /// coercion cast: the destination type names the bounds, the
    /// argument's type names the concrete pointee, and the
    /// pointee's drop glue is built now so the resulting `DynRef`
    /// carries it like a vtable's drop shim. The modelled `From`
    /// impls for a dyn box take the payload itself (or a reference
    /// to it), never an already-boxed value, so the pointee is
    /// the argument's type with any reference stripped. `None`
    /// when the destination is not a dyn box, when the argument's
    /// type is not preserved, or when the argument is itself a
    /// dyn value (no modelled conversion starts from one).
    fn from_result_unsize(
        &self,
        body: &mir::Body,
        destination: &mir::Place,
        arg: Option<&mir::Operand>,
        source_ty: Option<&mir::Ty>,
    ) -> Option<ir::UnsizeMeta> {
        let dest_ty = self.place_ty(body, destination)?;
        let mir::Ty::Adt(def, dest_args) = &dest_ty else {
            return None;
        };
        let name = self.names.get(def).or_else(|| self.def_names.get(def))?;
        if name != "std::boxed::Box" {
            return None;
        }
        let traits = dyn_traits_in(dest_args.first()?)?;
        // A zero-sized argument travels as a constant operand,
        // whose type the operand itself does not preserve; the
        // callee's generic arguments still name the source type
        // (the caller picks it out, as `mem_drop_glue` does for
        // the dropped type).
        let arg_ty = self
            .operand_place_ty(body, arg?)
            .or_else(|| source_ty.cloned())?;
        if dyn_traits_in(&arg_ty).is_some() {
            return None;
        }
        let pointee = match &arg_ty {
            mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => inner,
            other => other,
        };
        let concrete = self.concrete_ty_name(pointee);
        let glue = DropGlueBuilder::build(self, pointee);
        Some(ir::UnsizeMeta::Dyn {
            concrete: Some(concrete),
            traits,
            glue,
        })
    }

    /// The drop glue of a `std::mem::drop` call's argument, mirroring
    /// the `Drop` terminator's treatment of the same type:
    /// `Ok(Some(glue))` when the argument's glue can be accounted
    /// for and the interpreter has something to run it against —
    /// a place argument (run against the place), or a constant
    /// argument (run against the constant's value; a zero-sized
    /// constant carries no value, so the dropped type must
    /// determine one, and the operand is rewritten to carry it) —
    /// `Ok(None)` when there is nothing to run (the glue is
    /// trivial, so discarding the value is already faithful), and
    /// `Err(())` when the glue cannot be accounted for.
    fn mem_drop_glue(
        &self,
        body: &mir::Body,
        arg: &mir::Operand,
        ir_arg: &mut ir::Operand,
        raw_generic_args: &[mir::Ty],
    ) -> Result<Option<ir::DropGlue>, ()> {
        let dropped_ty = self
            .operand_place_ty(body, arg)
            .or_else(|| raw_generic_args.first().cloned());
        let glue = dropped_ty
            .as_ref()
            .and_then(|t| self.drop_glue(t))
            .ok_or(())?;
        if glue.root == ir::DropPlan::Trivial {
            return Ok(None);
        }
        match ir_arg {
            ir::Operand::Copy(_) | ir::Operand::Move(_) => Ok(Some(glue)),
            ir::Operand::Constant(ir::Constant::Unit) => {
                // A zero-sized constant: the value being dropped is
                // not carried by the operand. When the dropped type
                // determines it, rewrite the operand to carry that
                // value so the interpreter can run the glue against
                // it; otherwise the drop stays unaccounted for.
                let ty = dropped_ty.ok_or(())?;
                let value = self.zst_value(&ty, 0).ok_or(())?;
                let constant = value_as_constant(&value).ok_or(())?;
                *ir_arg = ir::Operand::Constant(constant);
                Ok(Some(glue))
            }
            ir::Operand::Constant(_) => Ok(Some(glue)),
            _ => Err(()),
        }
    }


    /// The converted body of `adt_name`'s `Drop` impl, if the
    /// program has one. The driver's name for the impl is
    /// `<{adt_name} as std::ops::Drop>::drop` (the `core::` spelling
    /// of the trait path is accepted as well); a generic impl made
    /// concrete for one type (`impl Drop for G<i32>`) is found by
    /// the same pattern when it is the only candidate.
    pub(super) fn dtor_body(&self, adt_name: &str, args: &[mir::Ty]) -> Option<String> {
        // The exact instance key, with the generic arguments
        // rendered as the driver records instance names: several
        // monomorphizations of one generic impl can coexist in the
        // program, and only the full key tells them apart.
        if !args.is_empty() {
            let parts: Vec<String> = args
                .iter()
                .map(|t| crate::builtins::ty_tag(t, self.names))
                .collect();
            let instantiated = format!("{adt_name}<{}>", parts.join(", "));
            for trait_path in ["std::ops::Drop", "core::ops::Drop"] {
                for method in ["drop", "pin_drop"] {
                    let key = format!("<{instantiated} as {trait_path}>::{method}");
                    if self.bodies.contains_key(&key) {
                        return Some(key);
                    }
                }
            }
        }
        for trait_path in ["std::ops::Drop", "core::ops::Drop"] {
            for method in ["drop", "pin_drop"] {
                let key = format!("<{adt_name} as {trait_path}>::{method}");
                if self.bodies.contains_key(&key) {
                    return Some(key);
                }
            }
        }
        let candidates: Vec<&String> = self
            .bodies
            .keys()
            .filter(|k| is_dtor_name(k, adt_name))
            .collect();
        match candidates.len() {
            1 => Some(candidates[0].clone()),
            0 => None,
            _ => {
                // Several monomorphizations of one generic impl
                // coexist: prefer the candidate whose rendered
                // arguments match the requested ones. The
                // driver's instance names also render lifetime
                // arguments (as `'_`), which the requested
                // argument list does not carry, so lifetimes are
                // skipped on the candidate side.
                let want: Vec<String> = args
                    .iter()
                    .map(|t| crate::builtins::ty_tag(t, self.names))
                    .collect();
                let matching: Vec<&&String> = candidates
                    .iter()
                    .filter(|k| {
                        let Some(rest) = k.strip_prefix('<') else {
                            return false;
                        };
                        let Some((self_part, _)) = rest.split_once(" as ") else {
                            return false;
                        };
                        let Some(open) = self_part.find('<') else {
                            return want.is_empty();
                        };
                        let Some(close) = self_part.rfind('>') else {
                            return false;
                        };
                        let inside = &self_part[open + 1..close];
                        let mut parts: Vec<String> = Vec::new();
                        let mut depth = 0i32;
                        let mut current = String::new();
                        for c in inside.chars() {
                            match c {
                                '<' | '(' | '[' => {
                                    depth += 1;
                                    current.push(c);
                                }
                                '>' | ')' | ']' => {
                                    depth -= 1;
                                    current.push(c);
                                }
                                ',' if depth == 0 => {
                                    parts.push(current.trim().to_string());
                                    current = String::new();
                                }
                                _ => current.push(c),
                            }
                        }
                        if !current.trim().is_empty() {
                            parts.push(current.trim().to_string());
                        }
                        let parts: Vec<&String> = parts
                            .iter()
                            .filter(|a| {
                                !a.starts_with('\'')
                            })
                            .collect();
                        parts.len() == want.len()
                            && parts
                                .iter()
                                .zip(want.iter())
                                .all(|(a, b)| a == &b)
                    })
                    .collect();
                match matching.len() {
                    1 => Some((*matching[0]).clone()),
                    _ => None,
                }
            }
        }
    }

    /// Whether the driver's name table records a `Drop` impl for
    /// `adt_name` at all, under any generic spelling.
    pub(super) fn dtor_impl_known(&self, adt_name: &str) -> bool {
        self.names.values().any(|n| is_dtor_name(n, adt_name))
    }

    /// Whether `name` (a type's driver name) belongs to the entry
    /// crate. For the entry crate's types the name table is
    /// complete, so a `Drop` impl that is absent from it does not
    /// exist; for any other crate's type an impl could exist
    /// unrecorded, and conversion must not assume there is none.
    pub(super) fn is_local_type(&self, name: &str) -> bool {
        match &self.local_crate {
            Some(k) => name.split("::").next() == Some(k.as_str()),
            None => false,
        }
    }

    /// Convert the operands of an evaluable inline-asm block into
    /// register slots: inputs are evaluated at block entry, outputs
    /// written back at block exit. Only called for blocks the
    /// classifier accepted, so every operand is in the register
    /// class with a plain integer type.
    fn convert_asm_regs(
        &self,
        body: &mir::Body,
        operands: &[mir::InlineAsmOperand],
    ) -> Result<Vec<ir::AsmReg>, ConvertError> {
        let mut regs = Vec::with_capacity(operands.len());
        for op in operands {
            let init = match op.kind {
                mir::AsmOperandKind::Const => Some(ir::Operand::Constant(ir::Constant::Uint(
                    u128::from(op.const_bits.unwrap_or(0)),
                ))),
                mir::AsmOperandKind::In | mir::AsmOperandKind::InOut => {
                    let value = op
                        .in_value
                        .as_ref()
                        .expect("the driver sets in_value for in operands");
                    let ctx = self.operand_place_ty(body, value);
                    Some(self.convert_operand(body, value, ctx.as_ref())?)
                }
                mir::AsmOperandKind::Out
                | mir::AsmOperandKind::Sym
                | mir::AsmOperandKind::Label => None,
            };
            let dest = match op.kind {
                mir::AsmOperandKind::Out | mir::AsmOperandKind::InOut => {
                    // A pure clobber (a `clobber_abi` expansion)
                    // has no place: its slot is inert.
                    match op.out_place.as_ref() {
                        Some(place) => Some(self.convert_place(body, place)?),
                        None => None,
                    }
                }
                _ => None,
            };
            // The operand type: an inout operand's place type
            // (its in_value is the same place), else the in
            // operand's place type, else the out place's.
            let reg_ty = match op.kind {
                mir::AsmOperandKind::InOut | mir::AsmOperandKind::Out => op
                    .out_place
                    .as_ref()
                    .and_then(|p| self.place_ty(body, p))
                    .or_else(|| {
                        op.in_value
                            .as_ref()
                            .and_then(|v| self.operand_place_ty(body, v))
                    }),
                mir::AsmOperandKind::In => op
                    .in_value
                    .as_ref()
                    .and_then(|v| self.operand_place_ty(body, v)),
                _ => None,
            };
            let kind = match reg_ty {
                Some(mir::Ty::Int(_)) => ir::AsmRegKind::Signed,
                Some(mir::Ty::Bool) => ir::AsmRegKind::Bool,
                _ => ir::AsmRegKind::Unsigned,
            };
            let arch = op
                .reg_name
                .as_deref()
                .and_then(mir::arch_reg_index);
            regs.push(ir::AsmReg {
                init,
                dest,
                width: op.width,
                kind,
                arch,
            });
        }
        Ok(regs)
    }

    pub(super) fn convert_terminator(
        &self,
        body: &mir::Body,
        kind: &mir::TerminatorKind,
        fn_abi: Option<&mir::Abi>,
        block_has_statements: bool,
    ) -> Result<ir::TerminatorKind, ConvertError> {
        use mir::TerminatorKind as T;
        Ok(match kind {
            T::Goto { target } => ir::TerminatorKind::Goto {
                target: ir::BasicBlock(target.index()),
            },
            T::SwitchInt { discr, targets } => {
                let ctx = self.operand_place_ty(body, discr);
                let discr = self.convert_operand(body, discr, ctx.as_ref())?;
                let pairs = targets
                    .values
                    .iter()
                    .map(|v| normalize_switch_value(*v, ctx.as_ref()))
                    .zip(targets.targets.iter().map(|b| ir::BasicBlock(b.index())))
                    .collect();
                ir::TerminatorKind::SwitchInt {
                    discr,
                    targets: pairs,
                    otherwise: ir::BasicBlock(targets.otherwise.index()),
                }
            }
            T::Return => ir::TerminatorKind::Return,
            T::Unreachable => ir::TerminatorKind::Unreachable,
            T::Call {
                func,
                args,
                destination,
                target,
                ..
            } => {
                // Indirect call through a callable value in a place
                // (a function pointer, function item, or closure
                // variable): the func operand is the callable itself.
                // A constant operand holding a decoded callable value
                // (a function pointer constant naming a function item
                // or a coerced closure) is the same shape: the call
                // goes through the value, so a closure target gets
                // its environment argument like any other indirect
                // call.
                let indirect = matches!(
                    func,
                    mir::Operand::Copy(_) | mir::Operand::Move(_)
                ) || matches!(
                    func,
                    mir::Operand::Constant(c) if matches!(
                        c.literal,
                        mir::ConstKind::Value(
                            mir::ConstValue::FnDef(_) | mir::ConstValue::Closure(_)
                        )
                    )
                );
                if let Some(drop_term) =
                    self.drop_call_terminator(body, func, args, *target)
                {
                    drop_term
                } else if let Some(term) = self.vec_from_iter_value_terminator(
                    body,
                    func,
                    args,
                    destination,
                    *target,
                )? {
                    term
                } else if indirect {
                    let func_ty = self.operand_place_ty(body, func);
                    // A constant callable carries no type of its own;
                    // type the arguments from the target body's
                    // signature instead, as the direct-call path does.
                    let param_tys = match func_ty.as_ref() {
                        Some(_) => callable_param_tys(func_ty.as_ref()),
                        None => self.constant_callee_param_tys(func),
                    };
                    let ir_func = self.convert_operand(body, func, func_ty.as_ref())?;
                    let mut ir_args = Vec::with_capacity(args.len());
                    for (i, arg) in args.iter().enumerate() {
                        ir_args.push(self.convert_operand(body, arg, param_tys.get(i))?);
                    }
                    ir::TerminatorKind::CallIndirect {
                        func: ir_func,
                        args: ir_args,
                        destination: self.convert_place(body, destination)?,
                        target: target.map(|b| ir::BasicBlock(b.index())),
                    }
                } else if let Some(term) = self.time_add_sub_terminator(
                    body,
                    func,
                    args,
                    destination,
                    *target,
                )? {
                    term
                } else if let Some(term) = self.file_io_terminator(
                    body,
                    func,
                    args,
                    destination,
                    *target,
                )? {
                    term
                } else if let Some(name) = self.fn_trait_or_direct(func, args.len()) {
                    match name {
                        FnResolution::Trait(kind, raw_generic_args, stamped) => {
                            // An `AsyncFn` call's receiver is typed
                            // by nothing the conversion can use: its
                            // `Self` generic argument is the
                            // coroutine-closure type the driver
                            // records as the unit placeholder, and
                            // typing the receiver operand by it
                            // would misread the reference the
                            // operand names. The operand's own
                            // place type types it instead.
                            let receiver_ty = if matches!(
                                kind,
                                ir::FnCallKind::AsyncCall
                                    | ir::FnCallKind::AsyncCallMut
                                    | ir::FnCallKind::AsyncCallOnce
                            ) {
                                None
                            } else {
                                raw_generic_args.first()
                            };
                            let receiver =
                                self.convert_operand(body, &args[0], receiver_ty)?;
                            let tuple = self
                                .convert_operand(body, &args[1], raw_generic_args.get(1))?;
                            // The stamp is carried for a direct or
                            // generic call, whose shim forwards
                            // the call-site location. For a `dyn`
                            // receiver the call passes through the
                            // vtable shim, which presents its own
                            // location instead (see
                            // `ir::Program::fn_shim_locations`), so
                            // the appended stamp is dropped here.
                            let stamp = if stamped
                                && !raw_generic_args
                                    .first()
                                    .is_some_and(|t| ty_mentions_dyn(t))
                            {
                                Some(self.convert_operand(body, &args[2], None)?)
                            } else {
                                None
                            };
                            // The capture count of the coroutine
                            // an `AsyncFn` call creates: the
                            // future type's last generic argument
                            // is the coroutine's upvars tuple - the
                            // state-0 fields the creation aggregate
                            // writes - and the fields begin with
                            // the call arguments, so the captures
                            // are what remains.
                            // For the `Fn` family the count is
                            // computed as well: a call whose
                            // receiver is an async closure value
                            // is served as the corresponding
                            // `AsyncFn` call (the closure's own
                            // body is not convertible), and a
                            // zero-sized such closure needs the
                            // count exactly as an `AsyncFn` call
                            // does. For any other receiver the
                            // value is never read.
                            let async_captures = if matches!(
                                kind,
                                ir::FnCallKind::AsyncCall
                                    | ir::FnCallKind::AsyncCallMut
                                    | ir::FnCallKind::AsyncCallOnce
                                    | ir::FnCallKind::Call
                                    | ir::FnCallKind::CallMut
                                    | ir::FnCallKind::CallOnce
                            ) {
                                let arg_count = match raw_generic_args.get(1) {
                                    Some(mir::Ty::Tuple(ts)) => ts.len(),
                                    _ => 0,
                                };
                                self.place_ty(body, destination)
                                    .and_then(|t| match t {
                                        mir::Ty::Coroutine(_, cargs) => {
                                            match cargs.last() {
                                                Some(mir::Ty::Tuple(ts)) => {
                                                    Some(ts.len())
                                                }
                                                _ => None,
                                            }
                                        }
                                        _ => None,
                                    })
                                    .and_then(|fields| {
                                        fields.checked_sub(arg_count)
                                    })
                            } else {
                                None
                            };
                            ir::TerminatorKind::CallTrait {
                                kind,
                                receiver,
                                args: tuple,
                                stamp,
                                async_captures,
                                receiver_glue: if matches!(
                                    kind,
                                    ir::FnCallKind::CallOnce
                                ) {
                                    raw_generic_args
                                        .first()
                                        .and_then(|t| self.drop_glue(t))
                                } else {
                                    None
                                },
                                destination: self.convert_place(body, destination)?,
                                target: target.map(|b| ir::BasicBlock(b.index())),
                            }
                        }
                        FnResolution::Direct(name) => {
                            let callee = self.bodies.get(&name);
                            // The receiver (argument 0) of a call that
                            // also resolves as an `Fn`-trait call is
                            // typed by the call's own `Self` type, not
                            // by the callee body's declared receiver
                            // type: the body is the generic impl
                            // definition (its receiver mentions the
                            // definition's parameters), while the call
                            // names the concrete `Self` — a zero-sized
                            // constant receiver only materialises
                            // under the concrete type (see
                            // `convert_const_value`).
                            let self_ctx = self
                                .resolve_fn_trait_call(func)
                                .and_then(|(_, raw)| raw.first().cloned());
                            let mut ir_args = Vec::with_capacity(args.len());
                            for (i, arg) in args.iter().enumerate() {
                                // Only a declared parameter types a
                                // call argument. The parameters that
                                // name call arguments are locals
                                // 1..=arg_count, except a trailing
                                // `VaList` parameter of a C-variadic
                                // callee, which names no single
                                // argument (the whole tail is packed
                                // into it at the call boundary); and
                                // an argument past the last parameter
                                // (a `#[track_caller]` stamp) names
                                // none either. Typing such an
                                // argument by the local at its index
                                // — an unrelated temporary of the
                                // callee — corrupts literal arguments
                                // (an integer literal read under a
                                // `bool` temporary collapses to
                                // `true`), so it is typed by its own
                                // recorded type instead, the fallback
                                // `convert_operand` applies when no
                                // context is given.
                                let param_ctx = |b: &&mir::Body| {
                                    let named = if self.mir_body_takes_va_list(b) {
                                        b.arg_count - 1
                                    } else {
                                        b.arg_count
                                    };
                                    if i < named {
                                        b.local_decls.get(i + 1).map(|d| d.ty.clone())
                                    } else {
                                        None
                                    }
                                };
                                let ctx = if i == 0 {
                                    self_ctx.clone().or_else(|| callee.and_then(param_ctx))
                                } else {
                                    callee.and_then(param_ctx)
                                };
                                ir_args.push(self.convert_operand(body, arg, ctx.as_ref())?);
                            }
                            ir::TerminatorKind::Call {
                                func: name,
                                args: ir_args,
                                destination: self.convert_place(body, destination)?,
                                target: target.map(|b| ir::BasicBlock(b.index())),
                            }
                        }
                    }
                } else if let Some((name, generic_args, raw_generic_args)) =
                    self.resolve_external_callee(func)
                {
                    let canonical = crate::builtins::canonical_name(&name);
                    let kind = crate::builtins::lookup_qualified(&canonical)
                        .filter(|kind| {
                            !self.consumer_prefers_body(*kind, func, &raw_generic_args)
                        })
                        // A body-less foreign callee (an extern
                        // declaration) may be served by the
                        // foreign-helper models; they are not in
                        // `lookup`, so no collected body is ever
                        // displaced by them.
                        .or_else(|| crate::builtins::foreign_helper_lookup(&canonical))
                        // A compiler intrinsic the guest crate
                        // declares itself (`#[rustc_intrinsic]`,
                        // the minicore shape) is named by its
                        // declaration site's path, which the
                        // qualified registry never matches; the
                        // intrinsic's own name is its identity
                        // (see `builtins::intrinsic_lookup`).
                        // The fallback is gated on the driver's
                        // own classification, recorded in the
                        // program: a body-less callee that Stable
                        // MIR does not classify as an intrinsic
                        // (an ordinary function whose body was
                        // skipped, or whose conversion failed)
                        // keeps that failure instead of being
                        // silently served by the builtin its name
                        // happens to match.
                        .or_else(|| {
                            let is_intrinsic = self
                                .callee_def_id(func)
                                .is_some_and(|def| self.intrinsic_defs.contains(&def));
                            if is_intrinsic {
                                crate::builtins::intrinsic_lookup(&canonical)
                            } else {
                                None
                            }
                        })
                        // A UFCS `fmt` trait call whose `Self`
                        // instance has no converted body is served
                        // by the structural fmt engine (see
                        // `builtins::fmt_trait_method`); a collected
                        // impl body has already won above, exactly
                        // as for the foreign helpers. A `dyn`
                        // `Self` converts here too: the engine
                        // resolves the trait object to its
                        // concrete value, exactly as it does for
                        // a `dyn` argument.
                        .or_else(|| {
                            if raw_generic_args.first().is_some() {
                                crate::builtins::fmt_trait_method(&canonical)
                            } else {
                                None
                            }
                        })
                        // A `ToSocketAddrs` call whose `Self`
                        // instance has no converted body is
                        // served by the net model (see
                        // `builtins::net_trait_method`); a
                        // collected impl body has already won
                        // above, exactly as for the fmt traits.
                        .or_else(|| {
                            if raw_generic_args.first().is_some() {
                                crate::builtins::net_trait_method(&canonical)
                            } else {
                                None
                            }
                        });
                    if let Some(kind) = kind {
                        let mut ir_args = Vec::with_capacity(args.len());
                        for (i, arg) in args.iter().enumerate() {
                            let ctx = builtin_arg_ctx(kind, &raw_generic_args, i);
                            ir_args.push(self.convert_operand(body, arg, ctx.as_ref())?);
                        }
                        // The driver appends the caller-location
                        // stamp to a call of a `#[track_caller]`
                        // callee (a trait method like
                        // `Index::index`, whose resolved impl is
                        // track-caller through the trait). A body
                        // serving the call binds the stamp to its
                        // synthetic local; a builtin that does not
                        // consume the location keeps its declared
                        // argument list, so the stamp is dropped
                        // here.
                        if !crate::builtins::consumes_caller_location(kind)
                            && ir_args.len() > 1
                            && args.last().is_some_and(is_caller_location_stamp)
                        {
                            ir_args.pop();
                        }
                        // An explicit `std::mem::drop` runs the
                        // dropped value's glue, exactly like a `Drop`
                        // terminator for the same type: build the
                        // glue now, while the type and the name
                        // tables are at hand (see `mem_drop_glue`).
                        // A call whose glue cannot be accounted for
                        // stays unsupported instead of silently
                        // discarding a value whose destructor has
                        // observable effects.
                        let mut unaccountable_drop = false;
                        let drop_glue = if kind == crate::builtins::Builtin::MemDrop
                            && args.len() == 1
                        {
                            match self.mem_drop_glue(
                                body,
                                &args[0],
                                &mut ir_args[0],
                                &raw_generic_args,
                            ) {
                                Ok(glue) => glue,
                                Err(()) => {
                                    unaccountable_drop = true;
                                    None
                                }
                            }
                        } else {
                            None
                        };
                        // The `PartialOrd` / `Ord` comparison
                        // builtins compare values structurally; for
                        // an enum `Self` type that comparison must
                        // order by discriminant value, so bake the
                        // type's table while it is at hand (the
                        // `Discriminant` rvalue does the same).
                        let cmp_discriminants = if matches!(
                            kind,
                            crate::builtins::Builtin::PartialCmp
                                | crate::builtins::Builtin::OrdCmp
                                | crate::builtins::Builtin::PartialOrdRel(_)
                        ) {
                            self.comparison_discriminants(body, args, &raw_generic_args)
                        } else {
                            None
                        };
                        // A `From` / `Into` conversion into a
                        // `Box<dyn Trait>`: the modelled conversion
                        // returns the plain box aggregate, so bake
                        // the unsizing the destination type implies
                        // (see `from_result_unsize`); the
                        // interpreter applies it to the result, as
                        // it would for a coercion cast.
                        let result_unsize = if matches!(
                            kind,
                            crate::builtins::Builtin::FromFrom
                                | crate::builtins::Builtin::IntoInto
                        ) {
                            // The conversion's source type: `From`
                            // renders the callee's generic arguments
                            // as [target, source], `Into` (whose
                            // `Self` is the source) as [source,
                            // target].
                            let source_ty = if matches!(
                                kind,
                                crate::builtins::Builtin::IntoInto
                            ) {
                                raw_generic_args.first()
                            } else {
                                raw_generic_args.get(1)
                            };
                            self.from_result_unsize(
                                body,
                                destination,
                                args.first(),
                                source_ty,
                            )
                        } else {
                            None
                        };
                        // A modelled consumer of an owned
                        // iterator drops the iterator inside the
                        // callee's real body: build the iterator
                        // type's glue now (see `consumed_glue` in
                        // `ir`), so the interpreter can run it
                        // against the iterator's final state. The
                        // iterator type is the callee generic
                        // argument the interpreter's dispatch
                        // reads as the iterator type. A trivial
                        // glue has nothing to run; a glue that
                        // cannot be accounted for keeps the plain
                        // call, the treatment these calls have
                        // always had.
                        let consumed_glue = match kind {
                            crate::builtins::Builtin::IteratorCollect
                            | crate::builtins::Builtin::IteratorCount
                            | crate::builtins::Builtin::IteratorFold
                            | crate::builtins::Builtin::IteratorForEach => {
                                raw_generic_args.first()
                            }
                            crate::builtins::Builtin::FromIteratorFromIter => {
                                raw_generic_args.get(2)
                            }
                            crate::builtins::Builtin::VecFromIterNested => {
                                raw_generic_args.last()
                            }
                            _ => None,
                        }
                        .and_then(|ty| self.drop_glue(ty))
                        .filter(|glue| glue.root != ir::DropPlan::Trivial);
                        // A str-pattern search likewise consumes
                        // its pattern inside the callee's real
                        // body: a by-value predicate (a closure
                        // passed to `str::contains` and its
                        // siblings) is dropped there, captures
                        // and all. Build the pattern argument
                        // type's glue the same way, from the
                        // argument operand's own type: a borrowed
                        // predicate's reference type, a `char`,
                        // a `&str` and a `&[char]` all have
                        // trivial glue and stay unadorned, as do
                        // constant arguments naming a function
                        // item (it carries no captures to drop).
                        // A zero-sized predicate travels as a
                        // constant operand too, whose type the
                        // operand itself does not preserve; the
                        // callee's generic arguments still name
                        // the pattern type then (the fallback
                        // `mem_drop_glue` uses for the dropped
                        // type), and a zero-sized closure does
                        // carry captures to drop, so its glue is
                        // built from that type the same way.
                        let consumed_glue = consumed_glue.or_else(|| {
                            if matches!(
                                kind,
                                crate::builtins::Builtin::StrContains
                                    | crate::builtins::Builtin::StrStartsWith
                                    | crate::builtins::Builtin::StrEndsWith
                            ) {
                                args.get(1)
                                    .and_then(|arg| {
                                        self.operand_place_ty(body, arg)
                                            .or_else(|| {
                                                raw_generic_args
                                                    .first()
                                                    .cloned()
                                            })
                                    })
                                    .and_then(|ty| self.drop_glue(&ty))
                                    .filter(|glue| {
                                        glue.root != ir::DropPlan::Trivial
                                    })
                            } else {
                                None
                            }
                        });
                        // An `Iterator::nth` call consumes the
                        // items its skip loop passes over: build
                        // the item type's glue now (see
                        // `skipped_glue` in `ir`), so the
                        // interpreter can run it against each
                        // skipped item. The item type is the
                        // destination type's `Option` argument.
                        // A trivial glue has nothing to run; a
                        // glue that cannot be accounted for
                        // keeps the plain call, the treatment
                        // these calls have always had.
                        let skipped_glue = if matches!(
                            kind,
                            crate::builtins::Builtin::IteratorNth
                        ) {
                            self.place_ty(body, destination)
                                .and_then(|dest_ty| match dest_ty {
                                    mir::Ty::Adt(def, args) => {
                                        let name = self
                                            .names
                                            .get(&def)
                                            .or_else(|| {
                                                self.def_names.get(&def)
                                            })?;
                                        if name == "std::option::Option" {
                                            args.into_iter().next()
                                        } else {
                                            None
                                        }
                                    }
                                    _ => None,
                                })
                                .and_then(|ty| self.drop_glue(&ty))
                                .filter(|glue| {
                                    glue.root != ir::DropPlan::Trivial
                                })
                        } else {
                            None
                        };
                        // A `From` / `Into` conversion into a
                        // `Box<[T]>` clones the source slice's
                        // elements one by one in the modelled
                        // conversion: build the element type's
                        // glue now (see `elem_glue` in `ir`), so
                        // the interpreter can drop the cloned
                        // prefix when a later element's clone
                        // panics. The conversion's target type:
                        // `From` renders the callee's generic
                        // arguments as [target, source], `Into`
                        // (whose `Self` is the source) as
                        // [source, target]. A trivial glue has
                        // nothing to run; a glue that cannot be
                        // accounted for keeps the plain call, the
                        // treatment these calls have always had.
                        let elem_glue = if matches!(
                            kind,
                            crate::builtins::Builtin::SliceToVec
                        ) {
                            // `slice::to_vec` clones the
                            // receiver slice's elements one by
                            // one in the modelled builtin: build
                            // the element type's glue from the
                            // receiver operand's type, so the
                            // cloned prefix drops when a later
                            // element's clone panics.
                            args.first()
                                .and_then(|arg| {
                                    self.operand_place_ty(body, arg)
                                })
                                .and_then(|ty| match ty {
                                    mir::Ty::Ref(_, inner, _) => {
                                        match *inner {
                                            mir::Ty::Slice(elem) => {
                                                Some(*elem)
                                            }
                                            mir::Ty::Array(elem, _) => {
                                                Some(*elem)
                                            }
                                            _ => None,
                                        }
                                    }
                                    mir::Ty::Slice(elem) => Some(*elem),
                                    _ => None,
                                })
                                .and_then(|elem| self.drop_glue(&elem))
                                .filter(|glue| {
                                    glue.root != ir::DropPlan::Trivial
                                })
                        } else if matches!(
                            kind,
                            crate::builtins::Builtin::FromFrom
                                | crate::builtins::Builtin::IntoInto
                        ) {
                            let target_ty = if matches!(
                                kind,
                                crate::builtins::Builtin::IntoInto
                            ) {
                                raw_generic_args.get(1)
                            } else {
                                raw_generic_args.first()
                            };
                            target_ty
                                .and_then(|ty| match ty {
                                    mir::Ty::Adt(def, targs)
                                        if self.names.get(def).map(String::as_str)
                                            == Some("std::boxed::Box") =>
                                    {
                                        match targs.first() {
                                            Some(mir::Ty::Slice(elem))
                                            | Some(mir::Ty::Array(elem, _)) => {
                                                Some(elem.clone())
                                            }
                                            _ => None,
                                        }
                                    }
                                    _ => None,
                                })
                                .and_then(|elem| self.drop_glue(&elem))
                                .filter(|glue| {
                                    glue.root != ir::DropPlan::Trivial
                                })
                        } else {
                            None
                        };
                        // A `Clone::clone` call on a closure
                        // clones the environment capture by
                        // capture in the modelled clone, and a
                        // call on a `Vec` / tuple / array / `Box`
                        // clones element by element (field by
                        // field, pointee) in the same model:
                        // build the receiver type's glue now
                        // (see `receiver_glue` in `ir`), so the
                        // interpreter can drop the parts cloned
                        // so far when a later part's clone
                        // panics. The `Self` type is the
                        // callee's first generic argument. A
                        // trivial glue has nothing to run; a
                        // glue that cannot be accounted for
                        // keeps the plain call, the treatment
                        // these calls have always had.
                        let receiver_glue = if matches!(
                            kind,
                            crate::builtins::Builtin::CloneClone
                        ) {
                            raw_generic_args
                                .first()
                                .and_then(|ty| match ty {
                                    mir::Ty::Closure(..)
                                    | mir::Ty::Tuple(_)
                                    | mir::Ty::Array(..) => {
                                        self.drop_glue(ty)
                                    }
                                    mir::Ty::Adt(def, _)
                                        if self
                                            .names
                                            .get(def)
                                            .or_else(|| {
                                                self.def_names.get(def)
                                            })
                                            .map(String::as_str)
                                            == Some("std::vec::Vec")
                                            || self
                                                .names
                                                .get(def)
                                                .or_else(|| {
                                                    self.def_names
                                                        .get(def)
                                                })
                                                .map(String::as_str)
                                                == Some("std::boxed::Box") =>
                                    {
                                        self.drop_glue(ty)
                                    }
                                    _ => None,
                                })
                                .filter(|glue| {
                                    glue.root != ir::DropPlan::Trivial
                                })
                        } else {
                            None
                        };
                        // A `box_assume_init_into_vec_unsafe` call
                        // asserts its source array is fully
                        // initialized; an element slot that still
                        // holds the uninitialized marker is one
                        // whose write was elided because the
                        // element type has exactly one value. Bake
                        // that value while the type is at hand
                        // (see `into_vec_elem_value`); the builtin
                        // materializes the unwritten slots from
                        // it.
                        let unique_elem = if kind
                            == crate::builtins::Builtin::BoxAssumeInitIntoVec
                        {
                            args.first()
                                .and_then(|arg| self.into_vec_elem_value(body, arg))
                        } else {
                            None
                        };
                        // A `mem::zeroed` call materializes the
                        // zero bit pattern of its type: for the
                        // types the builtin's own tag-based
                        // construction cannot describe (enums and
                        // ADT aggregates), build the denoted
                        // value while the type and the layout
                        // tables are at hand (see `zeroed_value`
                        // in `ir`).
                        let zeroed_value = if kind
                            == crate::builtins::Builtin::MemZeroed
                        {
                            raw_generic_args
                                .first()
                                .and_then(|ty| self.zeroed_model_value(ty, 0))
                        } else if kind == crate::builtins::Builtin::WriteBytes
                            && matches!(
                                args.get(1).and_then(|arg| {
                                    self.convert_operand(body, arg, None).ok()
                                }),
                                Some(ir::Operand::Constant(
                                    ir::Constant::Uint(0)
                                ))
                            )
                        {
                            // A `write_bytes` whose fill byte is
                            // the constant zero (`MaybeUninit::
                            // zeroed` is its canonical use): the
                            // filled region holds the zero
                            // pattern of the pointee type, whose
                            // denoted value is baked like
                            // `mem::zeroed`'s.
                            raw_generic_args
                                .first()
                                .and_then(|ty| self.zeroed_model_value(ty, 0))
                        } else {
                            None
                        };
                        if unaccountable_drop {
                            ir::TerminatorKind::Unsupported(
                                "Call to std::mem::drop (drop glue cannot be accounted for)"
                                    .to_string(),
                            )
                        } else {
                            ir::TerminatorKind::CallBuiltin {
                                name: canonical,
                                generic_args,
                                args: ir_args,
                                destination: self.convert_place(body, destination)?,
                                target: target.map(|b| ir::BasicBlock(b.index())),
                                drop_glue,
                                consumed_glue,
                                skipped_glue,
                                elem_glue,
                                receiver_glue,
                                cmp_discriminants,
                                result_unsize,
                                unique_elem,
                                zeroed_value,
                            }
                        }
                    } else if args.first().is_some_and(|a| {
                        self.operand_place_ty(body, a)
                            .is_some_and(|t| dyn_traits_in(&t).is_some())
                    }) {
                        // A method call on a `dyn Trait` value: the
                        // callee is the trait method, which has no
                        // body of its own. The call is a vtable
                        // dispatch, resolved at run time against the
                        // receiver's concrete type (see
                        // `virtual_call_target` in the interpreter).
                        let mut ir_args = Vec::with_capacity(args.len());
                        for arg in args.iter() {
                            ir_args.push(self.convert_operand(body, arg, None)?);
                        }
                        ir::TerminatorKind::Call {
                            func: name,
                            args: ir_args,
                            destination: self.convert_place(body, destination)?,
                            target: target.map(|b| ir::BasicBlock(b.index())),
                        }
                    } else {
                        ir::TerminatorKind::Unsupported(format!(
                            "Call to {name} (no converted body)"
                        ))
                    }
                } else {
                    ir::TerminatorKind::Unsupported(
                        "Call to unresolved function operand".to_string(),
                    )
                }
            }
            T::Assert {
                cond,
                expected,
                msg,
                target,
                ..
            } => {
                // The kinds whose message embeds the assertion's
                // operand values keep the operands for run-time
                // formatting (assert_message_text's kind text
                // stays as the fallback).
                let (format, operands) = match &**msg {
                    mir::AssertMessage::BoundsCheck { len, index } => (
                        Some(ir::AssertFormat::BoundsCheck),
                        vec![
                            self.convert_operand(body, len, None)?,
                            self.convert_operand(body, index, None)?,
                        ],
                    ),
                    mir::AssertMessage::MisalignedPointerDereference {
                        required,
                        found,
                    } => (
                        Some(ir::AssertFormat::MisalignedPointerDeref),
                        vec![
                            self.convert_operand(body, required, None)?,
                            self.convert_operand(body, found, None)?,
                        ],
                    ),
                    _ => (None, Vec::new()),
                };
                ir::TerminatorKind::Assert {
                    cond: self.convert_operand(body, cond, Some(&mir::Ty::Bool))?,
                    expected: *expected,
                    target: ir::BasicBlock(target.index()),
                    message: assert_message_text(msg),
                    format,
                    operands,
                    overflow: matches!(
                        &**msg,
                        mir::AssertMessage::Overflow(..)
                            | mir::AssertMessage::OverflowNeg(..)
                    ),
                }
            }
            // The imaginary edge of a false edge is never taken at run
            // time; execution continues at the real target.
            T::FalseEdge { real_target, .. } | T::FalseUnwind { real_target, .. } => {
                ir::TerminatorKind::Goto {
                    target: ir::BasicBlock(real_target.index()),
                }
            }
            T::Drop { place, target, .. } => {
                let glue = self.place_ty(body, place).and_then(|t| {
                    // A coroutine-closure value's place type is
                    // the unit placeholder, whose glue is trivial;
                    // the value's own glue is its captures' glue.
                    // When the place is the bare local this body's
                    // coroutine-closure aggregate created, plan it
                    // from the aggregate, exactly as a plain
                    // closure type's glue is planned from its
                    // capture tuple.
                    if place.projection.is_empty()
                        && matches!(&t, mir::Ty::Tuple(ts) if ts.is_empty())
                    {
                        if let Some((def, upvars)) =
                            coroutine_closure_creation(body, place.local.index())
                        {
                            let closure_ty =
                                mir::Ty::Closure(def, vec![mir::Ty::Tuple(upvars)]);
                            if let Some(glue) = self.drop_glue(&closure_ty) {
                                return Some(glue);
                            }
                        }
                    }
                    self.drop_glue(&t)
                });
                match glue {
                    Some(glue) => ir::TerminatorKind::Drop {
                        place: self.convert_place(body, place)?,
                        target: ir::BasicBlock(target.index()),
                        glue,
                    },
                    None => {
                        ir::TerminatorKind::Unsupported("Drop".to_string())
                    }
                }
            }
            T::Resume => ir::TerminatorKind::Resume,
            T::Terminate(mir::TerminateKind::Abort) => ir::TerminatorKind::Abort,
            T::Terminate(mir::TerminateKind::Resume) => ir::TerminatorKind::Resume,
            T::Yield { .. } => ir::TerminatorKind::Unsupported("Yield".to_string()),
            T::CoroutineDrop => ir::TerminatorKind::Unsupported("CoroutineDrop".to_string()),
            T::InlineAsm {
                template,
                operands,
                options,
                destination,
                targets,
                ..
            } => {
                if mir::inline_asm_is_noop(template, options) {
                    match destination {
                        Some(target) => ir::TerminatorKind::Goto {
                            target: ir::BasicBlock(target.index()),
                        },
                        None => ir::TerminatorKind::Unsupported(
                            "InlineAsm (no-op template without a destination)".to_string(),
                        ),
                    }
                } else if let Some(program) = mir::inline_asm_program(
                    template,
                    options,
                    operands,
                    targets,
                    destination.is_some(),
                ) {
                    if program.steps.is_empty() {
                        // The step-over class: templates with no
                        // real instruction, whose operands the
                        // classifier leaves inert (e.g. the
                        // explicit-register operands of a
                        // clobber-only block).
                        match destination {
                            Some(target) => ir::TerminatorKind::Goto {
                                target: ir::BasicBlock(target.index()),
                            },
                            None => ir::TerminatorKind::Unsupported(
                                "InlineAsm (no-op template without a destination)"
                                    .to_string(),
                            ),
                        }
                    } else {
                        // A naked block's return value is read
                        // out of the register file, so its return
                        // type must be a plain scalar shape.
                        let mut ret = None;
                        let mut shape_ok = true;
                        if program.returns {
                            let bare_ret = !program
                                .steps
                                .iter()
                                .any(|st| !matches!(st, mir::AsmStep::Ret));
                            match asm_return_shape(body) {
                                Some(shape) => {
                                    // A bare-`ret` block reads
                                    // nothing of its own: the
                                    // register the caller
                                    // established is the result, so
                                    // a scalar return takes the
                                    // shape — but only when the
                                    // block is the naked body
                                    // itself, which carries no
                                    // statements. A `ret` block
                                    // reached after statements ends
                                    // the function with the return
                                    // place as the body left it.
                                    if !bare_ret || !block_has_statements {
                                        ret = Some(shape);
                                    }
                                }
                                None => {
                                    // A body whose steps run before
                                    // the `ret` reads the return
                                    // register as its result, so a
                                    // return type with no scalar
                                    // shape keeps it outside the
                                    // class. A bare-`ret` body with
                                    // a unit return needs no shape.
                                    if !bare_ret {
                                        shape_ok = false;
                                    }
                                }
                            }
                        }
                        // A naked block's parameters arrive in
                        // the enclosing function's ABI registers,
                        // not through operands: they are seeded
                        // from the frame's argument locals. A
                        // naked body whose parameters cannot be
                        // seeded (an ABI family or an argument
                        // shape the register model does not
                        // cover) is a named error rather than a
                        // block that reads zeroed registers as if
                        // they were the arguments. A bare-`ret`
                        // body never reads the registers, so it
                        // needs no seeding.
                        let mut arg_seeds = Vec::new();
                        let mut seeds_ok = true;
                        if program.returns && body.arg_count > 0 {
                            match asm_arg_seed_plan(body, fn_abi) {
                                Some(plan) => arg_seeds = plan,
                                None => {
                                    if program
                                        .steps
                                        .iter()
                                        .any(|st| !matches!(st, mir::AsmStep::Ret))
                                    {
                                        seeds_ok = false;
                                    }
                                }
                            }
                        }
                        if shape_ok && seeds_ok {
                            ir::TerminatorKind::InlineAsm {
                                regs: self.convert_asm_regs(body, operands)?,
                                steps: program.steps,
                                target: if program.returns
                                    || options.contains("NORETURN")
                                {
                                    // A noreturn asm-goto leaves
                                    // through its label jump: no
                                    // fallthrough continuation.
                                    None
                                } else {
                                    destination.map(|b| ir::BasicBlock(b.index()))
                                },
                                ret,
                                arg_seeds,
                            }
                        } else if !seeds_ok {
                            ir::TerminatorKind::Unsupported(
                                "InlineAsm (naked function with parameters whose incoming ABI registers cannot be seeded)"
                                    .to_string(),
                            )
                        } else {
                            ir::TerminatorKind::Unsupported(format!(
                                "InlineAsm (template {template:?} is not a no-op)"
                            ))
                        }
                    }
                } else {
                    ir::TerminatorKind::Unsupported(format!(
                        "InlineAsm (template {template:?} is not a no-op)"
                    ))
                }
            }
            T::TailCall { .. } => ir::TerminatorKind::Unsupported("TailCall".to_string()),
        })
    }

}
