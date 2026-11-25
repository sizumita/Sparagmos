//! Iterator stepping: modelled next/next_back state machines and consumers.

use std::cmp::Ordering;


use crate::error::{InterpError, InterpResult};
use crate::interp::{
    channel_handle_id,
    Frame, Interpreter, Resolved,
};
use crate::ir::{Body, Operand, Ty};
use crate::value::{RefProj, RefTarget, Value};

use super::*;

impl<'a> Interpreter<'a> {
    pub(super) fn option_ref_into_iter(
        &mut self,
        frames: &mut Vec<Frame>,
        value: &Value,
        mutable: bool,
    ) -> InterpResult<Value> {
        let target = self.deref_value(frames, value)?;
        let option = self.read_resolved(frames, &target)?;
        let elems = match &option {
            Value::Enum { variant: 1, .. } => {
                let elem = match &target {
                    Resolved::Place {
                        frame_pos,
                        local,
                        projection,
                    } => {
                        let mut projection = projection.clone();
                        projection.push(RefProj::Field(0));
                        Value::Ref(RefTarget::place(
                            frames[*frame_pos].id,
                            *local,
                            projection,
                        ))
                    }
                    Resolved::Static { name, projection } => {
                        let mut projection = projection.clone();
                        projection.push(RefProj::Field(0));
                        Value::Ref(RefTarget::Static {
                            name: name.clone(),
                            projection,
                        })
                    }
                    Resolved::Alloc { id, projection } => {
                        let mut projection = projection.clone();
                        projection.push(RefProj::Field(0));
                        Value::Ref(RefTarget::Alloc {
                            id: *id,
                            projection,
                        })
                    }
                    Resolved::Value(base, projection) => {
                        let v = self.value_target_value(base, projection)?;
                        if mutable {
                            return Err(unsupported(
                                "IntoIterator::into_iter for &mut Option over a constant aggregate is not runtime-supported",
                            ));
                        }
                        match v {
                            Value::Enum { fields, .. } if !fields.is_empty() => {
                                Value::Ref(RefTarget::value(fields[0].clone()))
                            }
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "IntoIterator::into_iter for &Option",
                                    "option value",
                                    other.kind_name(),
                                ))
                            }
                        }
                    }
                };
                vec![elem]
            }
            Value::Enum { variant: 0, .. } => Vec::new(),
            other => {
                return Err(InterpError::type_mismatch(
                    "IntoIterator::into_iter for &Option",
                    "option value",
                    other.kind_name(),
                ))
            }
        };
        Ok(Value::Enum {
            variant: 0,
            fields: vec![Value::Tuple(elems), Value::Uint(0)],
        })
    }

    /// `Iterator::next`: mutate the iterator in place through the
    /// `&mut` argument and return the `Option` value.
    pub(super) fn iterator_next(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        ref_arg: &Value,
    ) -> InterpResult<Value> {
        let target = self.deref_value(frames, ref_arg)?;
        self.iterator_next_at(frames, self_ty, &target)
    }

    /// `Iterator::next` on the iterator stored at `target`:
    /// mutate it in place and return the `Option` value.
    pub(super) fn iterator_next_at(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &Resolved,
    ) -> InterpResult<Value> {
        // A `Map` on a place-based state is stepped in place
        // (its closure's environment reference must name the
        // closure field's own storage); `map_next` declines a
        // state that does not live in a place, and the
        // value-level step below then applies.
        if self_ty.starts_with("std::iter::Map<") {
            if let Some(result) = self.map_next(frames, self_ty, target)? {
                return Ok(result);
            }
        }
        if self_ty.starts_with("std::iter::SkipWhile<") {
            if let Some(result) = self.skip_while_next(frames, self_ty, target)? {
                return Ok(result);
            }
        }
        if self_ty.starts_with("std::iter::Filter<") {
            if let Some(result) = self.filter_next(frames, self_ty, target)? {
                return Ok(result);
            }
        }
        if self_ty.starts_with("std::iter::FilterMap<") {
            if let Some(result) = self.filter_map_next(frames, self_ty, target)? {
                return Ok(result);
            }
        }
        let state = self.read_resolved(frames, target)?;
        let (new_state, result) = self.iterator_next_state(frames, self_ty, state)?;
        self.write_resolved(frames, target, new_state)?;
        Ok(result)
    }

    /// Skip `n` items of an arithmetic range state at
    /// `target` in one arithmetic step — the advance the
    /// specialised `nth` / `advance_by` bodies of `Range` and
    /// `RangeInclusive` perform in core (for `char` ranges the
    /// distance and the advance both skip the surrogate gap,
    /// as `range_count`'s length already does), which the per-item
    /// loop cannot reproduce within the step limit for a
    /// large skip. `Ok(Some(true))` when the skip exhausted
    /// the range (the state is left in exactly the form the
    /// per-item step machine leaves an exhausted range in, so
    /// the two accountings cannot disagree), `Ok(Some(false))`
    /// when items remain (the caller's final step then serves
    /// the result), and `Ok(None)` when the state is not an
    /// arithmetic range's, leaving the per-item loop to the
    /// caller.
    pub(super) fn range_skip_at(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &Resolved,
        n: u128,
    ) -> InterpResult<Option<bool>> {
        let inclusive = if self_ty.starts_with("std::ops::RangeInclusive<") {
            true
        } else if self_ty.starts_with("std::ops::Range<") {
            false
        } else {
            return Ok(None);
        };
        let state = self.read_resolved(frames, target)?;
        let Ok(fields) = iterator_fields(&state, self_ty) else {
            return Ok(None);
        };
        // The bounds as bit patterns: with `start <= end` in
        // the value's own domain, the difference and the
        // advance are exact in wrapping arithmetic for both
        // the signed and the unsigned domain.
        let bits = |v: &Value| -> Option<u128> {
            match v {
                Value::Int(x) => Some(*x as u128),
                Value::Uint(x) => Some(*x),
                _ => None,
            }
        };
        let unbits = |start: &Value, bits: u128| -> Value {
            match start {
                Value::Int(_) => Value::Int(bits as i128),
                _ => Value::Uint(bits),
            }
        };
        if inclusive {
            if fields.len() != 3 {
                return Ok(None);
            }
            if fields[2] == Value::Bool(true) {
                // Already exhausted: the step machine leaves
                // the fields untouched and yields `None`.
                return Ok(Some(true));
            }
            if fields[2] != Value::Bool(false) {
                return Ok(None);
            }
            let (Some(start), Some(end)) =
                (bits(&fields[0]), bits(&fields[1]))
            else {
                return Ok(None);
            };
            if compare_values(&fields[0], &fields[1])? == Ordering::Greater {
                // Empty and unflagged: stepping yields `None`
                // without touching the fields.
                return Ok(Some(true));
            }
            if range_elem_ty(self_ty) == "char" {
                // A `char` range's bounds are valid chars, so
                // neither lies in the surrogate gap; a state
                // that does is not one this arithmetic serves.
                if (0xD800..0xE000).contains(&start)
                    || (0xD800..0xE000).contains(&end)
                {
                    return Ok(None);
                }
                let Some(count) =
                    range_count(&fields[0], &fields[1], true, "char")?
                else {
                    return Ok(None);
                };
                if n >= count {
                    let new_state = rebuild_state(
                        &state,
                        vec![
                            fields[1].clone(),
                            fields[1].clone(),
                            Value::Bool(true),
                        ],
                    );
                    self.write_resolved(frames, target, new_state)?;
                    return Ok(Some(true));
                }
                let new_start =
                    unbits(&fields[0], char_advance(start, n));
                let new_state = rebuild_state(
                    &state,
                    vec![new_start, fields[1].clone(), Value::Bool(false)],
                );
                self.write_resolved(frames, target, new_state)?;
                return Ok(Some(false));
            }
            // The item count is `diff + 1`.
            let diff = end.wrapping_sub(start);
            if n > diff {
                // The skip passes the last item, whose own
                // step leaves [end, end, true].
                let new_state = rebuild_state(
                    &state,
                    vec![
                        fields[1].clone(),
                        fields[1].clone(),
                        Value::Bool(true),
                    ],
                );
                self.write_resolved(frames, target, new_state)?;
                Ok(Some(true))
            } else {
                let new_start =
                    unbits(&fields[0], start.wrapping_add(n));
                let new_state = rebuild_state(
                    &state,
                    vec![new_start, fields[1].clone(), Value::Bool(false)],
                );
                self.write_resolved(frames, target, new_state)?;
                Ok(Some(false))
            }
        } else {
            if fields.len() != 2 {
                return Ok(None);
            }
            let (Some(start), Some(end)) =
                (bits(&fields[0]), bits(&fields[1]))
            else {
                return Ok(None);
            };
            if compare_values(&fields[0], &fields[1])? != Ordering::Less {
                // Empty: the specialised advance still
                // leaves the range at its end (a native
                // `Range { start: 5, end: 3 }` skipped past
                // reads back with `start == 3`), which the
                // write below performs.
                let new_state = rebuild_state(
                    &state,
                    vec![fields[1].clone(), fields[1].clone()],
                );
                self.write_resolved(frames, target, new_state)?;
                return Ok(Some(true));
            }
            if range_elem_ty(self_ty) == "char" {
                // See the inclusive arm: valid `char` bounds
                // never lie in the surrogate gap.
                if (0xD800..0xE000).contains(&start)
                    || (0xD800..0xE000).contains(&end)
                {
                    return Ok(None);
                }
                let Some(count) =
                    range_count(&fields[0], &fields[1], false, "char")?
                else {
                    return Ok(None);
                };
                if n >= count {
                    let new_state = rebuild_state(
                        &state,
                        vec![fields[1].clone(), fields[1].clone()],
                    );
                    self.write_resolved(frames, target, new_state)?;
                    return Ok(Some(true));
                }
                let new_start =
                    unbits(&fields[0], char_advance(start, n));
                let new_state = rebuild_state(
                    &state,
                    vec![new_start, fields[1].clone()],
                );
                self.write_resolved(frames, target, new_state)?;
                return Ok(Some(false));
            }
            // The item count is `diff`.
            let diff = end.wrapping_sub(start);
            if n >= diff {
                let new_state = rebuild_state(
                    &state,
                    vec![fields[1].clone(), fields[1].clone()],
                );
                self.write_resolved(frames, target, new_state)?;
                Ok(Some(true))
            } else {
                let new_start =
                    unbits(&fields[0], start.wrapping_add(n));
                let new_state = rebuild_state(
                    &state,
                    vec![new_start, fields[1].clone()],
                );
                self.write_resolved(frames, target, new_state)?;
                Ok(Some(false))
            }
        }
    }

    /// `DoubleEndedIterator::next_back`: mutate the iterator in
    /// place through the `&mut` argument and return the `Option`
    /// value — the mirror of [`Self::iterator_next`].
    pub(super) fn iterator_next_back(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        ref_arg: &Value,
    ) -> InterpResult<Value> {
        let target = self.deref_value(frames, ref_arg)?;
        self.iterator_next_back_at(frames, self_ty, &target)
    }

    /// `DoubleEndedIterator::next_back` on the iterator stored
    /// at `target`: mutate it in place and return the `Option`
    /// value.
    fn iterator_next_back_at(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &Resolved,
    ) -> InterpResult<Value> {
        // A `Map` on a place-based state is stepped in place,
        // exactly as for `next` (see `iterator_next_at`): its
        // closure's environment reference must name the closure
        // field's own storage.
        if self_ty.starts_with("std::iter::Map<") {
            if let Some(result) = self.map_next_back(frames, self_ty, target)? {
                return Ok(result);
            }
        }
        if self_ty.starts_with("std::iter::Filter<") {
            if let Some(result) = self.filter_next_back(frames, self_ty, target)? {
                return Ok(result);
            }
        }
        if self_ty.starts_with("std::iter::FilterMap<") {
            if let Some(result) = self.filter_map_next_back(frames, self_ty, target)? {
                return Ok(result);
            }
        }
        let state = self.read_resolved(frames, target)?;
        let (new_state, result) = self.next_back_on_state(frames, self_ty, state)?;
        self.write_resolved(frames, target, new_state)?;
        Ok(result)
    }

    /// The place-based front step of a `Filter` (see
    /// `skip_while_next` for the shape this mirrors): pull
    /// inner items in place until the predicate accepts one.
    /// The predicate is an `FnMut`, so it is called with a
    /// reference to its own field's storage, and it takes the
    /// pulled item by reference. Declines a state that does
    /// not live in a place, like `map_next` does.
    fn filter_next(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &Resolved,
    ) -> InterpResult<Option<Value>> {
        let Some(inner_target) = extend_target(target, RefProj::Field(0)) else {
            return Ok(None);
        };
        let Some(pred_target) = extend_target(target, RefProj::Field(1)) else {
            return Ok(None);
        };
        let inner_ty = first_generic_arg(self_ty).unwrap_or("");
        loop {
            let item = self.iterator_next_at(frames, inner_ty, &inner_target)?;
            let Some(item) = option_payload_opt(item) else {
                return Ok(Some(none_value()));
            };
            let callable = self.read_resolved(frames, &pred_target)?;
            let env_ref = self.target_ref_value(frames, &pred_target)?;
            let arg = Value::Ref(RefTarget::value(item.clone()));
            let keep = self.call_callable_sync(frames, &callable, Some(env_ref), vec![arg])?;
            match keep {
                Value::Bool(true) => return Ok(Some(some_value(item))),
                Value::Bool(false) => continue,
                other => {
                    return Err(InterpError::type_mismatch(
                        "Filter predicate result",
                        "a boolean",
                        other.kind_name(),
                    ))
                }
            }
        }
    }

    /// The place-based back step of a `Filter`: the mirror
    /// of `filter_next` over the inner iterator's back step.
    fn filter_next_back(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &Resolved,
    ) -> InterpResult<Option<Value>> {
        let Some(inner_target) = extend_target(target, RefProj::Field(0)) else {
            return Ok(None);
        };
        let Some(pred_target) = extend_target(target, RefProj::Field(1)) else {
            return Ok(None);
        };
        let inner_ty = first_generic_arg(self_ty).unwrap_or("");
        loop {
            let item = self.iterator_next_back_at(frames, inner_ty, &inner_target)?;
            let Some(item) = option_payload_opt(item) else {
                return Ok(Some(none_value()));
            };
            let callable = self.read_resolved(frames, &pred_target)?;
            let env_ref = self.target_ref_value(frames, &pred_target)?;
            let arg = Value::Ref(RefTarget::value(item.clone()));
            let keep = self.call_callable_sync(frames, &callable, Some(env_ref), vec![arg])?;
            match keep {
                Value::Bool(true) => return Ok(Some(some_value(item))),
                Value::Bool(false) => continue,
                other => {
                    return Err(InterpError::type_mismatch(
                        "Filter predicate result",
                        "a boolean",
                        other.kind_name(),
                    ))
                }
            }
        }
    }

    /// The place-based front step of a `FilterMap` (see
    /// `filter_next` for the shape this mirrors): pull inner
    /// items in place, apply the closure to each by value, and
    /// yield the first `Some` it returns; an item whose closure
    /// returns `None` is dropped as it is pulled. The closure
    /// is an `FnMut`, so it is called with a reference to its
    /// own field's storage. Declines a state that does not live
    /// in a place, like `map_next` does.
    fn filter_map_next(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &Resolved,
    ) -> InterpResult<Option<Value>> {
        let Some(inner_target) = extend_target(target, RefProj::Field(0)) else {
            return Ok(None);
        };
        let Some(closure_target) = extend_target(target, RefProj::Field(1)) else {
            return Ok(None);
        };
        let inner_ty = first_generic_arg(self_ty).unwrap_or("");
        loop {
            let item = self.iterator_next_at(frames, inner_ty, &inner_target)?;
            let Some(item) = option_payload_opt(item) else {
                return Ok(Some(none_value()));
            };
            let callable = self.read_resolved(frames, &closure_target)?;
            let env_ref = self.target_ref_value(frames, &closure_target)?;
            let mapped = self.call_callable_sync(frames, &callable, Some(env_ref), vec![item])?;
            match option_payload(mapped)? {
                Some(y) => return Ok(Some(some_value(y))),
                None => continue,
            }
        }
    }

    /// The place-based back step of a `FilterMap`: the mirror
    /// of `filter_map_next` over the inner iterator's back
    /// step.
    fn filter_map_next_back(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &Resolved,
    ) -> InterpResult<Option<Value>> {
        let Some(inner_target) = extend_target(target, RefProj::Field(0)) else {
            return Ok(None);
        };
        let Some(closure_target) = extend_target(target, RefProj::Field(1)) else {
            return Ok(None);
        };
        let inner_ty = first_generic_arg(self_ty).unwrap_or("");
        loop {
            let item = self.iterator_next_back_at(frames, inner_ty, &inner_target)?;
            let Some(item) = option_payload_opt(item) else {
                return Ok(Some(none_value()));
            };
            let callable = self.read_resolved(frames, &closure_target)?;
            let env_ref = self.target_ref_value(frames, &closure_target)?;
            let mapped = self.call_callable_sync(frames, &callable, Some(env_ref), vec![item])?;
            match option_payload(mapped)? {
                Some(y) => return Ok(Some(some_value(y))),
                None => continue,
            }
        }
    }

    /// One front or back step of a `str` split iterator (the
    /// `Split` family: `Split`, `RSplit`, `SplitTerminator`,
    /// `RSplitTerminator`, `SplitInclusive`), replaying core's
    /// `SplitInternal` state machine on the state the
    /// constructor's converted body built: the one-field
    /// wrapper holds `[start, end, matcher, allow_trailing_empty,
    /// finished]`, and the five types differ only in which
    /// internal step each direction runs (the terminators in
    /// the constructor's recorded `allow_trailing_empty`).
    /// Only the `char` pattern's searcher is modelled -- the
    /// search is a substring search for the needle's UTF-8
    /// encoding, which is exact because a char's encoding never
    /// occurs inside another char's -- and any other searcher
    /// shape is the generic unsupported error, as before.
    fn str_split_step(
        &mut self,
        state: &Value,
        self_ty: &str,
        back: bool,
    ) -> InterpResult<(Value, Value)> {
        let Some(mut parts) = StrSplitParts::decode(state) else {
            return Err(unsupported(format!(
                "Iterator::next for {self_ty} is not runtime-supported"
            )));
        };
        let reversed = self_ty.starts_with("std::str::RSplit<")
            || self_ty.starts_with("std::str::RSplitTerminator<");
        let inclusive = self_ty.starts_with("std::str::SplitInclusive<");
        let item = match (back == reversed, inclusive) {
            (true, false) => parts.next(),
            (true, true) => parts.next_inclusive(),
            (false, false) => parts.next_back(),
            (false, true) => parts.next_back_inclusive(),
        };
        let new_state = parts.encode(state);
        Ok(match item {
            Some(piece) => (new_state, some_value(Value::Str(piece))),
            None => (new_state, none_value()),
        })
    }

    /// The place-based back step of a `Map` (see `map_next`):
    /// step the inner iterator in place from the back, then apply
    /// the closure to the yielded item. Declines a state that
    /// does not live in a place, like `map_next` does.
    fn map_next_back(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &Resolved,
    ) -> InterpResult<Option<Value>> {
        let Some(inner_target) = extend_target(target, RefProj::Field(0)) else {
            return Ok(None);
        };
        let Some(closure_target) = extend_target(target, RefProj::Field(1)) else {
            return Ok(None);
        };
        let inner_ty = first_generic_arg(self_ty).unwrap_or("");
        let item = self.iterator_next_back_at(frames, inner_ty, &inner_target)?;
        let Some(item) = option_payload_opt(item) else {
            return Ok(Some(none_value()));
        };
        let callable = self.read_resolved(frames, &closure_target)?;
        let env_ref = self.target_ref_value(frames, &closure_target)?;
        let mapped = self.call_callable_sync(frames, &callable, Some(env_ref), vec![item])?;
        Ok(Some(some_value(mapped)))
    }

    /// One `DoubleEndedIterator::next_back` step on an iterator
    /// state value: the new state and the step's `Option` value.
    /// The leaf shapes are the pure state transition
    /// (`iterator_next_back_state`) that `Rev::next` already
    /// runs; the arms here are the wrappers whose back step is a
    /// composition of other steps: a `Rev`'s back step is its
    /// inner iterator's front step, an `Enumerate`'s is the
    /// inner back step indexed from the end, a `Copied`'s is the
    /// inner back step with the yielded reference followed, a
    /// `Map`'s is the inner back step with the closure applied,
    /// a `Filter`'s is the inner back step repeated until the
    /// predicate accepts an item, and an `IterMut`'s hands out
    /// the last remaining element of the container.
    fn next_back_on_state(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        state: Value,
    ) -> InterpResult<(Value, Value)> {
        if is_str_split_ty(self_ty) {
            return self.str_split_step(&state, self_ty, true);
        }
        if self_ty.starts_with("std::vec::Drain<")
            || self_ty.starts_with("alloc::vec::Drain<")
        {
            return self.drain_next_state(frames, state, true);
        }
        if self_ty.starts_with("std::path::Components") {
            // The back step of the state `Builtin::PathComponents`
            // builds, replaying the host's parsing (see
            // `components_step`).
            return self.components_step(state, true);
        }
        if self_ty.starts_with("std::iter::Rev<") {
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Rev state",
                    "[inner iterator]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = self_ty
                .strip_prefix("std::iter::Rev<")
                .and_then(|rest| rest.strip_suffix('>'))
                .unwrap_or("");
            let (new_inner, result) =
                self.iterator_next_state(frames, inner_ty, fields[0].clone())?;
            return Ok((rebuild_state(&state, vec![new_inner]), result));
        }
        if self_ty.starts_with("std::iter::Enumerate<") {
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Enumerate state",
                    "[iterator, count]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let count = match fields[1] {
                Value::Uint(v) => v,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "Enumerate count",
                        "unsigned integer",
                        other.kind_name(),
                    ))
                }
            };
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (new_inner, item) =
                self.next_back_on_state(frames, inner_ty, fields[0].clone())?;
            let Some(x) = option_payload_opt(item) else {
                return Ok((
                    rebuild_state(&state, vec![new_inner, Value::Uint(count)]),
                    none_value(),
                ));
            };
            // The back item's index is the front count plus the
            // items that remain after it; std computes it from
            // the inner iterator's exact length, so an inexact
            // size hint is an explicit unsupported error here
            // (the step itself already happened on the state
            // value, but the run fails before anything observes
            // the stepped state).
            let inner_ref = Value::Ref(RefTarget::value(fields[0].clone()));
            let hint = self.iterator_size_hint(frames, inner_ty, &inner_ref)?;
            let remaining = match &hint {
                Value::Tuple(parts) if parts.len() == 2 => match &parts[0] {
                    Value::Uint(lower) => match option_payload_opt(parts[1].clone()) {
                        Some(Value::Uint(upper)) if upper == *lower => Some(*lower),
                        _ => None,
                    },
                    _ => None,
                },
                _ => None,
            };
            let Some(remaining) = remaining else {
                return Err(unsupported(format!(
                    "DoubleEndedIterator::next_back for {self_ty} needs the inner \
                     iterator's exact length, which its size hint does not give"
                )));
            };
            if remaining == 0 {
                return Err(unsupported(format!(
                    "DoubleEndedIterator::next_back for {self_ty} yielded an item \
                     from an inner iterator reporting zero remaining items"
                )));
            }
            let index = count + remaining - 1;
            return Ok((
                rebuild_state(&state, vec![new_inner, Value::Uint(count)]),
                some_value(Value::Tuple(vec![Value::Uint(index), x])),
            ));
        }
        if self_ty.starts_with("std::iter::Copied<") {
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Copied state",
                    "[iterator]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (new_inner, item) =
                self.next_back_on_state(frames, inner_ty, fields[0].clone())?;
            return match option_payload_opt(item) {
                None => Ok((rebuild_state(&state, vec![new_inner]), none_value())),
                Some(x) => {
                    let copied = self.resolve_value(frames, &x)?;
                    Ok((rebuild_state(&state, vec![new_inner]), some_value(copied)))
                }
            };
        }
        if self_ty.starts_with("std::iter::Cloned<") {
            // The back step of a `Cloned` is the inner back
            // step with the yielded reference cloned, exactly
            // as the front step.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Cloned state",
                    "[iterator]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (new_inner, item) =
                self.next_back_on_state(frames, inner_ty, fields[0].clone())?;
            return match option_payload_opt(item) {
                None => Ok((rebuild_state(&state, vec![new_inner]), none_value())),
                Some(x) => {
                    let elem_ty = ref_item_element_ty(inner_ty);
                    let cloned = self.clone_modelled_value(
                        frames,
                        &elem_ty,
                        "std::clone::Clone::clone",
                        &x,
                        None,
                    )?;
                    Ok((rebuild_state(&state, vec![new_inner]), some_value(cloned)))
                }
            };
        }
        if self_ty.starts_with("std::iter::Filter<") {
            // The value-level back step, used when the Filter
            // does not live in a place (see `filter_next_back`):
            // pull inner items from the back until the
            // predicate accepts one. The predicate is applied
            // with a reference to the closure value itself, as
            // in the front step's value-level arm.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Filter state",
                    "[iterator, predicate]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let mut inner = fields[0].clone();
            let predicate = fields[1].clone();
            loop {
                let (new_inner, item) =
                    self.next_back_on_state(frames, inner_ty, inner)?;
                inner = new_inner;
                let Some(x) = option_payload_opt(item) else {
                    return Ok((
                        rebuild_state(&state, vec![inner, predicate]),
                        none_value(),
                    ));
                };
                let arg = Value::Ref(RefTarget::value(x.clone()));
                let keep = self.call_callable_sync(frames, &predicate, None, vec![arg])?;
                match keep {
                    Value::Bool(true) => {
                        return Ok((
                            rebuild_state(&state, vec![inner, predicate]),
                            some_value(x),
                        ));
                    }
                    Value::Bool(false) => continue,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "Filter predicate result",
                            "a boolean",
                            other.kind_name(),
                        ))
                    }
                }
            }
        }
        if self_ty.starts_with("std::iter::FilterMap<") {
            // The value-level back step, used when the
            // FilterMap does not live in a place (see
            // `filter_map_next_back`): pull inner items from
            // the back, apply the closure to each by value, and
            // yield the first `Some` it returns. The closure is
            // applied with a reference to the value itself, as
            // in the front step's value-level arm.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "FilterMap state",
                    "[iterator, closure]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let mut inner = fields[0].clone();
            let closure = fields[1].clone();
            loop {
                let (new_inner, item) =
                    self.next_back_on_state(frames, inner_ty, inner)?;
                inner = new_inner;
                let Some(x) = option_payload_opt(item) else {
                    return Ok((
                        rebuild_state(&state, vec![inner, closure]),
                        none_value(),
                    ));
                };
                let mapped =
                    self.call_callable_sync(frames, &closure, None, vec![x])?;
                if let Some(y) = option_payload(mapped)? {
                    return Ok((
                        rebuild_state(&state, vec![inner, closure]),
                        some_value(y),
                    ));
                }
            }
        }
        if self_ty.starts_with("std::iter::Map<") {
            // The value-level step, used when the Map does not
            // live in a place (see `map_next_back`): the closure
            // is applied with a reference to the value itself,
            // exactly as in the front step's fallback.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Map state",
                    "[iterator, closure]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (new_inner, item) =
                self.next_back_on_state(frames, inner_ty, fields[0].clone())?;
            return match option_payload_opt(item) {
                None => Ok((
                    rebuild_state(&state, vec![new_inner, fields[1].clone()]),
                    none_value(),
                )),
                Some(x) => {
                    let mapped =
                        self.call_callable_sync(frames, &fields[1], None, vec![x])?;
                    Ok((
                        rebuild_state(&state, vec![new_inner, fields[1].clone()]),
                        some_value(mapped),
                    ))
                }
            };
        }
        if self_ty.starts_with("std::option::Iter<")
            || self_ty.starts_with("std::option::IterMut<")
        {
            // The two state shapes of the front step (see
            // `next_on_state`): the [elements, index] state, whose
            // back step drops the last remaining element, and the
            // nested one-payload shape, whose only element is
            // taken exactly as the front step takes it.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() == 2 {
                if let (Value::Tuple(items), Value::Uint(idx)) = (&fields[0], &fields[1]) {
                    let idx = *idx as usize;
                    if idx < items.len() {
                        let item = items[items.len() - 1].clone();
                        let kept = items[..items.len() - 1].to_vec();
                        let new_state = rebuild_state(
                            &state,
                            vec![Value::Tuple(kept), Value::Uint(idx as u128)],
                        );
                        return Ok((new_state, some_value(item)));
                    }
                    return Ok((state, none_value()));
                }
            }
            return match take_option_payload(&state) {
                Ok(Some((new_state, item))) => Ok((new_state, some_value(item))),
                Ok(None) => Ok((state, none_value())),
                Err(()) => Err(InterpError::type_mismatch(
                    format!("iterator state of {self_ty}"),
                    "an Option value in one-field wrappers",
                    state.kind_name(),
                )),
            };
        }
        if self_ty.starts_with("std::iter::Once<") {
            // `Once`'s single payload is taken on the first step
            // from either end; the step is the front step's.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Once state",
                    "[option]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            return match take_option_payload(&fields[0]) {
                Ok(Some((new_inner, item))) => {
                    Ok((rebuild_state(&state, vec![new_inner]), some_value(item)))
                }
                Ok(None) => Ok((state, none_value())),
                Err(()) => Err(InterpError::type_mismatch(
                    "Once state",
                    "an Option value in one-field wrappers",
                    fields[0].kind_name(),
                )),
            };
        }
        if self_ty.starts_with("std::slice::IterMut<") {
            // The state `IntoIterator` builds for `&mut Vec` /
            // `&mut [T; N]` / `&mut [T]` (see the front step in
            // `next_on_state`): the back step hands out a
            // reference to the last remaining element in the
            // container's own storage and shrinks the length.
            // Any other state shape — the pointer-pair layout
            // the real `iter_mut` body builds, which the front
            // step does not serve either — falls through to the
            // pure transition below and is reported unsupported.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() == 3 {
                if let (Value::Uint(idx), Value::Uint(len)) = (&fields[1], &fields[2]) {
                    let (idx, len) = (*idx, *len);
                    if idx < len {
                        let elem =
                            self.iter_mut_element_ref(frames, &fields[0], (len - 1) as usize)?;
                        return Ok((
                            rebuild_state(
                                &state,
                                vec![
                                    fields[0].clone(),
                                    Value::Uint(idx),
                                    Value::Uint(len - 1),
                                ],
                            ),
                            some_value(elem),
                        ));
                    }
                    return Ok((state, none_value()));
                }
            }
        }
        if self_ty.starts_with("std::iter::Zip<") {
            // `Zip`'s back step pairs the two sides from the
            // back, front-aligned: the longer side first drops
            // its unpaired excess from the back, then both
            // sides step back once — core's general `next_back`,
            // whose contract needs both sides' exact lengths
            // (the `ExactSizeIterator` bound on the impl), so
            // an inexact size hint on either side is an
            // explicit unsupported error. The state's index /
            // len counters belong to the random-access
            // specialization and are carried untouched, as in
            // the front step.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 4 {
                return Err(InterpError::type_mismatch(
                    "Zip state",
                    "[iterator, iterator, index, len]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let type_args = generic_args_of(self_ty);
            let first_ty = type_args.first().map(String::as_str).unwrap_or("");
            let second_ty = type_args.get(1).map(String::as_str).unwrap_or("");
            let exact_len = |this: &mut Self,
                             frames: &mut Vec<Frame>,
                             ty: &str,
                             state: &Value|
             -> InterpResult<u128> {
                let (lower, upper) = this.size_hint_parts(frames, ty, state.clone())?;
                match upper {
                    Some(u) if u == lower => Ok(lower),
                    _ => Err(unsupported(format!(
                        "DoubleEndedIterator::next_back for {self_ty} needs both \
                         inner iterators' exact lengths, which a size hint does \
                         not give"
                    ))),
                }
            };
            let a_sz = exact_len(self, frames, first_ty, &fields[0])?;
            let b_sz = exact_len(self, frames, second_ty, &fields[1])?;
            let mut first = fields[0].clone();
            let mut second = fields[1].clone();
            for _ in 0..a_sz.saturating_sub(b_sz) {
                let (new_first, _) =
                    self.next_back_on_state(frames, first_ty, first)?;
                first = new_first;
            }
            for _ in 0..b_sz.saturating_sub(a_sz) {
                let (new_second, _) =
                    self.next_back_on_state(frames, second_ty, second)?;
                second = new_second;
            }
            let (new_first, first_item) =
                self.next_back_on_state(frames, first_ty, first)?;
            let (new_second, second_item) =
                self.next_back_on_state(frames, second_ty, second)?;
            let result = match (
                option_payload_opt(first_item),
                option_payload_opt(second_item),
            ) {
                (Some(x), Some(y)) => some_value(Value::Tuple(vec![x, y])),
                (None, None) => none_value(),
                _ => {
                    return Err(unsupported(format!(
                        "DoubleEndedIterator::next_back for {self_ty}: the two \
                         sides report equal exact lengths but stepped \
                         differently"
                    )))
                }
            };
            return Ok((
                rebuild_state(
                    &state,
                    vec![
                        new_first,
                        new_second,
                        fields[2].clone(),
                        fields[3].clone(),
                    ],
                ),
                result,
            ));
        }
        let (new_state, item) = match iterator_next_back_state(&state, self_ty) {
            Ok(step) => step,
            Err(e) => {
                let fields = iterator_fields(&state, self_ty)?;
                match slice_iter_step_real(self, frames, self_ty, &fields, true) {
                    Some(step) => {
                        let (new_fields, result) = step?;
                        (
                            rebuild_state(&state, new_fields),
                            option_payload_opt(result),
                        )
                    }
                    None => return Err(e),
                }
            }
        };
        Ok((
            new_state,
            match item {
                Some(value) => some_value(value),
                None => none_value(),
            },
        ))
    }

    /// The state transition of `Iterator::next` for the modelled
    /// iterator shapes, as a function of the iterator state value
    /// alone: returns the successor state and the `Option` result.
    /// `iterator_next_at` applies it to the state stored behind a
    /// reference; the consumer builtins (`count`, `fold`,
    /// `collect`, `for_each`) drive it directly over the owned
    /// iterator value they receive, performing exactly the
    /// sequence of `next` calls the core default bodies perform.
    /// One step of a `vec::Drain`: the state is the drain's
    /// real field layout — `[tail_start, tail_len, iter, vec]`,
    /// where `iter` is the slice iterator over the drained range
    /// in the source vector's storage (its pointers carry the
    /// range as trailing `Index` steps into that storage) and
    /// `vec` is the `NonNull` back to the source vector. A step
    /// reads the element at the iterator's current pointer (the
    /// `ptr::read` the drain's own `next` performs), marks the
    /// storage slot uninitialized — the element has moved out —
    /// and advances the pointer (or retreats the end, for
    /// `next_back`). The source vector's length stays at the
    /// range start the drain's creation set, and the drain's
    /// own destructor body performs the tail move when the
    /// drain value is dropped, exactly as it does natively.
    fn drain_next_state(
        &mut self,
        frames: &mut Vec<Frame>,
        state: Value,
        from_back: bool,
    ) -> InterpResult<(Value, Value)> {
        let Value::Enum {
            variant,
            fields: drain,
        } = &state
        else {
            return Err(InterpError::type_mismatch(
                "Drain state",
                "the drain's field layout",
                state.kind_name(),
            ));
        };
        if drain.len() != 4 {
            return Err(InterpError::type_mismatch(
                "Drain state",
                "[tail_start, tail_len, iter, vec]",
                format!("{} field(s)", drain.len()),
            ));
        }
        let Value::Enum {
            variant: iter_variant,
            fields: iter,
        } = &drain[2]
        else {
            return Err(InterpError::type_mismatch(
                "Drain state",
                "a slice iterator in the iter field",
                drain[2].kind_name(),
            ));
        };
        if iter.len() < 2 {
            return Err(InterpError::type_mismatch(
                "Drain state",
                "a slice iterator with pointer and end",
                format!("{} field(s)", iter.len()),
            ));
        }
        let (ptr_field, end_field) = (iter[0].clone(), iter[1].clone());
        let (Some(cur), Some(last)) =
            (iter_ptr_index(&ptr_field, None), iter_ptr_index(&end_field, None))
        else {
            return Err(InterpError::type_mismatch(
                "Drain state",
                "element pointers into the source vector's storage",
                format!(
                    "{} / {}",
                    ptr_field.kind_name(),
                    end_field.kind_name()
                ),
            ));
        };
        if cur >= last {
            return Ok((state, none_value()));
        }
        let (step_field, at) = if from_back {
            (end_field.clone(), last - 1)
        } else {
            (ptr_field.clone(), cur)
        };
        let Some(ptr_value) = iter_ptr_value(&step_field) else {
            return Err(InterpError::type_mismatch(
                "Drain state",
                "an element pointer in the stepped field",
                step_field.kind_name(),
            ));
        };
        let target = self.deref_value(frames, &ptr_value)?;
        let item = self.read_resolved(frames, &target)?;
        self.write_resolved(frames, &target, Value::Uninit)?;
        let stepped = iter_ptr_with_index(&step_field, if from_back { last - 1 } else { cur + 1 })
            .expect("index step checked");
        let mut iter = iter.clone();
        if from_back {
            iter[1] = stepped;
        } else {
            iter[0] = stepped;
        }
        let mut drain = drain.clone();
        drain[2] = Value::Enum {
            variant: *iter_variant,
            fields: iter,
        };
        let new_state = Value::Enum {
            variant: *variant,
            fields: drain,
        };
        let _ = at;
        Ok((new_state, some_value(item)))
    }

    /// One step of `std::io::Lines<B>`: the state is the
    /// one-field value the constructor's converted body
    /// built, holding the reader; the step is `read_line`
    /// over the reader's file (see `read_until_byte`) with
    /// std's own post-processing -- no item at end of stream,
    /// the line terminator (and a carriage return before it)
    /// stripped, an invalid-UTF-8 line the `InvalidData`
    /// error `read_line` itself reports. The state itself
    /// does not change: the reader's position lives in the
    /// host file.
    fn io_lines_next(
        &mut self,
        frames: &mut Vec<Frame>,
        state: Value,
    ) -> InterpResult<(Value, Value)> {
        let fields = aggregate_fields(&state).ok_or_else(|| {
            InterpError::type_mismatch(
                "io::Lines state",
                "an iterator holding a reader",
                state.kind_name(),
            )
        })?;
        let reader = fields.first().ok_or_else(|| {
            InterpError::type_mismatch(
                "io::Lines state",
                "an iterator holding a reader",
                state.kind_name(),
            )
        })?;
        let id = self.file_id_of(frames, reader)?;
        let mut bytes: Vec<u8> = Vec::new();
        let read = self.with_file(id, |f| {
            read_until_byte(f, &mut bytes, b'\n')
        });
        match read {
            None => Err(unsupported(
                "Iterator::next for io::Lines over a handle that is not open"
                    .to_string(),
            )),
            Some(Err(error)) => Ok((
                state,
                some_value(err_result(host_io_error(&error))),
            )),
            Some(Ok(())) => {
                if bytes.is_empty() {
                    return Ok((state, none_value()));
                }
                if std::str::from_utf8(&bytes).is_err() {
                    return Ok((
                        state,
                        some_value(err_result(io_simple_error_value(
                            std::io::ErrorKind::InvalidData as u32,
                        ))),
                    ));
                }
                if bytes.last() == Some(&b'\n') {
                    bytes.pop();
                    if bytes.last() == Some(&b'\r') {
                        bytes.pop();
                    }
                }
                let text =
                    String::from_utf8(bytes).expect("validated above");
                Ok((state, some_value(ok_result(Value::Str(text)))))
            }
        }
    }

    /// One step of `std::io::Bytes<B>`: a single byte read
    /// from the state's reader, in the same shape as
    /// `io_lines_next`.
    fn io_bytes_next(
        &mut self,
        frames: &mut Vec<Frame>,
        state: Value,
    ) -> InterpResult<(Value, Value)> {
        let fields = aggregate_fields(&state).ok_or_else(|| {
            InterpError::type_mismatch(
                "io::Bytes state",
                "an iterator holding a reader",
                state.kind_name(),
            )
        })?;
        let reader = fields.first().ok_or_else(|| {
            InterpError::type_mismatch(
                "io::Bytes state",
                "an iterator holding a reader",
                state.kind_name(),
            )
        })?;
        let id = self.file_id_of(frames, reader)?;
        let mut byte = [0u8; 1];
        let read = self.with_file(id, |f| {
            <std::fs::File as std::io::Read>::read(f, &mut byte)
        });
        match read {
            None => Err(unsupported(
                "Iterator::next for io::Bytes over a handle that is not open"
                    .to_string(),
            )),
            Some(Err(error)) => Ok((
                state,
                some_value(err_result(host_io_error(&error))),
            )),
            Some(Ok(0)) => Ok((state, none_value())),
            Some(Ok(_)) => Ok((
                state,
                some_value(ok_result(Value::Uint(u128::from(byte[0])))),
            )),
        }
    }

    pub(crate) fn iterator_next_state(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        state: Value,
    ) -> InterpResult<(Value, Value)> {
        // A trait-object iterator (`Box<dyn Iterator>`, or a
        // `&dyn Iterator` receiver): the state is the dyn
        // reference the unsizing cast formed, and stepping it
        // steps the concrete iterator it wraps - in place when
        // the wrapper holds a pointer to the concrete value,
        // functionally when it holds the value itself - and the
        // result is rewrapped in the same dyn reference.
        if let Value::DynRef {
            base,
            concrete,
            traits,
            drop_glue,
        } = &state
        {
            // The pointer to the concrete value: the base
            // itself for a reference coercion, or the single
            // field of the box aggregate for a `Box` coercion.
            let pointer = match &**base {
                Value::Ref(_) | Value::FatRef { .. } => Some((**base).clone()),
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    match &fields[0] {
                        Value::Ref(_) | Value::FatRef { .. } => {
                            Some(fields[0].clone())
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some(pointer) = pointer {
                let target = self.deref_value(frames, &pointer)?;
                let inner = self.read_resolved(frames, &target)?;
                let (new_inner, item) =
                    self.iterator_next_state(frames, concrete, inner)?;
                self.write_resolved(frames, &target, new_inner)?;
                return Ok((state, item));
            }
            let (new_base, item) =
                self.iterator_next_state(frames, concrete, (**base).clone())?;
            return Ok((
                Value::DynRef {
                    base: Box::new(new_base),
                    concrete: concrete.clone(),
                    traits: traits.clone(),
                    drop_glue: drop_glue.clone(),
                },
                item,
            ));
        }
        if self_ty.starts_with("std::io::Lines<") {
            return self.io_lines_next(frames, state);
        }
        if self_ty.starts_with("std::io::Bytes<") {
            return self.io_bytes_next(frames, state);
        }
        if self_ty.starts_with("std::vec::Drain<")
            || self_ty.starts_with("alloc::vec::Drain<")
        {
            return self.drain_next_state(frames, state, false);
        }
        if self_ty.starts_with("std::sync::mpsc::Iter<") {
            // The state holds the receiver handle; iteration is
            // `recv` until the channel disconnects.
            let handle = match &state {
                Value::Enum { fields, .. } if fields.len() == 1 => fields[0].clone(),
                other => {
                    return Err(InterpError::type_mismatch(
                        "mpsc Iter state",
                        "an iterator holding a receiver handle",
                        other.kind_name(),
                    ))
                }
            };
            let Some(id) = channel_handle_id(&handle) else {
                return Err(InterpError::type_mismatch(
                    "mpsc Iter state",
                    "a receiver handle",
                    handle.kind_name(),
                ));
            };
            return match self.mpsc_recv(frames, id, "Iterator::next for mpsc::Iter")? {
                Ok(value) => Ok((state, some_value(value))),
                Err(()) => Ok((state, none_value())),
            };
        }
        if self_ty.starts_with("std::sync::mpsc::IntoIter<") {
            // The state holds the receiver handle as its one field.
            let handle = match &state {
                Value::Enum { fields, .. } if fields.len() == 1 => fields[0].clone(),
                other => {
                    return Err(InterpError::type_mismatch(
                        "mpsc IntoIter state",
                        "an iterator holding a receiver handle",
                        other.kind_name(),
                    ))
                }
            };
            let Some(id) = channel_handle_id(&handle) else {
                return Err(InterpError::type_mismatch(
                    "mpsc IntoIter state",
                    "a receiver handle",
                    handle.kind_name(),
                ));
            };
            return match self.mpsc_recv(frames, id, "Iterator::next for mpsc::IntoIter")? {
                Ok(value) => Ok((state, some_value(value))),
                Err(()) => Ok((state, none_value())),
            };
        }
        if self_ty_is(self_ty, "std::env::Args") || self_ty_is(self_ty, "std::env::ArgsOs") {
            // `Args { inner: ArgsOs }` / `ArgsOs { inner:
            // sys::args::common::Args }` / the common `Args {
            // inner: vec::IntoIter }`: step the innermost owned-iterator
            // state and rebuild the wrappers. The stored
            // elements are the argument strings themselves;
            // `Args::next` yields them as-is, so its
            // OsString-to-String conversion is the identity
            // (the arguments are always valid text here, so
            // the native panic on invalid Unicode cannot
            // arise), while `ArgsOs::next` yields each element
            // in the standard `OsString` buffer form the
            // OsString consumers (deref, lossy conversion,
            // command construction) read.
            let depth = if self_ty_is(self_ty, "std::env::Args") { 3 } else { 2 };
            let mut inner = &state;
            for _ in 0..depth {
                inner = match inner {
                    Value::Enum { fields, .. } if fields.len() == 1 => &fields[0],
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("{self_ty} state"),
                            "nested one-field iterator wrappers",
                            other.kind_name(),
                        ))
                    }
                };
            }
            let (new_inner, item) = step_owned_iter(inner, self_ty)?;
            let mut rebuilt = new_inner;
            for _ in 0..depth {
                rebuilt = Value::Enum {
                    variant: 0,
                    fields: vec![rebuilt],
                };
            }
            let item = if self_ty_is(self_ty, "std::env::ArgsOs") {
                item.map(|value| match value {
                    Value::Str(text) => os_string_value(text.as_bytes()),
                    other => other,
                })
            } else {
                item
            };
            return Ok((
                rebuilt,
                match item {
                    Some(value) => some_value(value),
                    None => none_value(),
                },
            ));
        }
        if self_ty.starts_with("std::path::Components") {
            // The state is the path's byte string and the
            // numbers of components consumed from each end (see
            // `Builtin::PathComponents`); the step replays the
            // host's own parsing from it.
            return self.components_step(state, false);
        }
        if self_ty.starts_with("std::path::Ancestors") {
            // The state is the struct the converted
            // `Path::ancestors` body built: one field, the
            // `Option<&Path>` holding the next ancestor to
            // yield (see `ancestors_step`).
            return self.ancestors_step(frames, state);
        }
        if self_ty.starts_with("std::slice::Windows<") {
            // Two state shapes reach here: the native struct
            // {tail slice, window size} (built by a struct
            // literal in core's search machinery), stepped by
            // shrinking the tail exactly as the native `next`
            // does, and the `windows` builtin's [slice, size,
            // next index] snapshot.
            let fields = iterator_fields(&state, self_ty)?;
            let tail_slice = |elems: &[Value]| {
                Value::FatRef {
                    base: Box::new(Value::Ref(RefTarget::Value(
                        Box::new(Value::Tuple(elems.to_vec())),
                        vec![RefProj::Index(0)],
                    ))),
                    len: elems.len() as u128,
                }
            };
            let size_of = |f: &Value| -> InterpResult<usize> {
                // `NonZero<usize>` arrives as nested one-field
                // wrappers around the integer.
                let mut cur = f;
                loop {
                    match cur {
                        Value::Uint(v) => return Ok(*v as usize),
                        Value::Enum { fields, .. } if fields.len() == 1 => {
                            cur = &fields[0];
                        }
                        Value::Tuple(fields) if fields.len() == 1 => {
                            cur = &fields[0];
                        }
                        other => {
                            return Err(InterpError::type_mismatch(
                                "Windows size",
                                "non-zero integer",
                                other.kind_name(),
                            ))
                        }
                    }
                }
            };
            if fields.len() == 2 {
                let (elems, _) = self.slice_elements(frames, &fields[0])?;
                let size = size_of(&fields[1])?;
                if size == 0 || elems.len() < size {
                    return Ok((state, none_value()));
                }
                let item = tail_slice(&elems[..size]);
                let new_state = Value::Enum {
                    variant: 0,
                    fields: vec![tail_slice(&elems[1..]), fields[1].clone()],
                };
                return Ok((new_state, some_value(item)));
            }
            if fields.len() != 3 {
                return Err(InterpError::type_mismatch(
                    "Windows state",
                    "[slice, window size] or [slice, window size, next index]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let (elems, _) = self.slice_elements(frames, &fields[0])?;
            let size = size_of(&fields[1])?;
            let index = match fields[2] {
                Value::Uint(v) => v as usize,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "Windows index",
                        "integer",
                        other.kind_name(),
                    ))
                }
            };
            if size == 0 || index + size > elems.len() {
                return Ok((state, none_value()));
            }
            let item = tail_slice(&elems[index..index + size]);
            let new_state = Value::Enum {
                variant: 0,
                fields: vec![
                    fields[0].clone(),
                    fields[1].clone(),
                    Value::Uint((index + 1) as u128),
                ],
            };
            return Ok((new_state, some_value(item)));
        }
        if self_ty_is(self_ty, "std::env::SplitPaths") {
            // The state is the owned-iterator state the
            // `split_paths` builtin snapshots ([parts, next
            // index]); the items are the parts' `PathBuf`
            // values themselves.
            let (new_state, item) = step_owned_iter(&state, self_ty)?;
            return Ok((
                new_state,
                match item {
                    Some(part) => some_value(part),
                    None => none_value(),
                },
            ));
        }
        if self_ty_is(self_ty, "std::fs::ReadDir") {
            // The state is the one-field wrapper around the
            // owned-iterator state ([entries, next index]) the
            // fs sys layer's snapshot builds; the items are the
            // entries' `io::Result<DirEntry>` values themselves.
            let inner = match &state {
                Value::Enum { fields, .. } if fields.len() == 1 => {
                    fields[0].clone()
                }
                other => {
                    return Err(InterpError::type_mismatch(
                        "ReadDir state",
                        "a wrapper around [entries, next index]",
                        other.kind_name(),
                    ))
                }
            };
            let (new_inner, item) = step_owned_iter(&inner, self_ty)?;
            let rebuilt = Value::Enum {
                variant: 0,
                fields: vec![new_inner],
            };
            return Ok((
                rebuilt,
                match item {
                    Some(result) => some_value(result),
                    None => none_value(),
                },
            ));
        }
        let (new_state, result) = self.next_on_state(frames, self_ty, &state)?;
        Ok((new_state, result))
    }

    /// One step of a `std::path::Components` state, wrapping
    /// the item in its `Option` (see `components_step_state`).
    fn components_step(
        &self,
        state: Value,
        from_back: bool,
    ) -> InterpResult<(Value, Value)> {
        let (new_state, item) = components_step_state(&state, from_back)?;
        Ok((
            new_state,
            match item {
                Some(value) => some_value(value),
                None => none_value(),
            },
        ))
    }

    /// One step of a `std::path::Ancestors` state: the one
    /// field is the `Option<&Path>` the iterator yields
    /// next; the step returns it and stores its parent in
    /// its place (`None` once the parent chain ends) —
    /// exactly the native `Ancestors::next`. The parent is
    /// taken with the host's own `std::path`, as for the
    /// other path-parsing models, and the yielded value is
    /// a `&Path` in the model's usual form (`sub_path_value`).
    fn ancestors_step(
        &self,
        frames: &[Frame],
        state: Value,
    ) -> InterpResult<(Value, Value)> {
        let fields = iterator_fields(&state, "std::path::Ancestors")?;
        if fields.len() != 1 {
            return Err(InterpError::type_mismatch(
                "Ancestors state",
                "one field (the next ancestor)",
                format!("{} field(s)", fields.len()),
            ));
        }
        let field = fields.into_iter().next().unwrap();
        if matches!(&field, Value::Enum { variant: 0, .. }) {
            // The `next` field is `None`: the iterator is
            // exhausted and stays exhausted.
            return Ok((state, none_value()));
        }
        let Some(payload) = option_payload_opt(field) else {
            return Err(InterpError::type_mismatch(
                "Ancestors state",
                "an Option<&Path> field",
                "a non-Option field",
            ));
        };
        let bytes = self.path_bytes_of(frames, &payload)?.0;
        let next = {
            use std::os::unix::ffi::OsStrExt;
            let path =
                std::path::Path::new(std::ffi::OsStr::from_bytes(&bytes));
            match path.parent() {
                Some(parent) => {
                    some_value(sub_path_value(parent.as_os_str().as_bytes()))
                }
                None => none_value(),
            }
        };
        Ok((
            rebuild_state(&state, vec![next]),
            some_value(sub_path_value(&bytes)),
        ))
    }

    fn map_next(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &Resolved,
    ) -> InterpResult<Option<Value>> {
        let Some(inner_target) = extend_target(target, RefProj::Field(0)) else {
            return Ok(None);
        };
        let Some(closure_target) = extend_target(target, RefProj::Field(1)) else {
            return Ok(None);
        };
        let inner_ty = first_generic_arg(self_ty).unwrap_or("");
        let item = self.iterator_next_at(frames, inner_ty, &inner_target)?;
        let Some(item) = option_payload_opt(item) else {
            return Ok(Some(none_value()));
        };
        let callable = self.read_resolved(frames, &closure_target)?;
        let env_ref = match &closure_target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => Value::Ref(RefTarget::place(
                frames[*frame_pos].id,
                *local,
                projection.clone(),
            )),
            Resolved::Static { name, projection } => Value::Ref(RefTarget::Static {
                name: name.clone(),
                projection: projection.clone(),
            }),
            Resolved::Alloc { id, projection } => Value::Ref(RefTarget::Alloc {
                id: *id,
                projection: projection.clone(),
            }),
            Resolved::Value(..) => unreachable!("closure target is a place"),
        };
        let mapped = self.call_callable_sync(frames, &callable, Some(env_ref), vec![item])?;
        Ok(Some(some_value(mapped)))
    }

    fn target_ref_value(&self, frames: &[Frame], target: &Resolved) -> InterpResult<Value> {
        match target {
            Resolved::Place {
                frame_pos,
                local,
                projection,
            } => Ok(Value::Ref(RefTarget::place(
                frames[*frame_pos].id,
                *local,
                projection.clone(),
            ))),
            Resolved::Static { name, projection } => Ok(Value::Ref(RefTarget::Static {
                name: name.clone(),
                projection: projection.clone(),
            })),
            Resolved::Alloc { id, projection } => Ok(Value::Ref(RefTarget::Alloc {
                id: *id,
                projection: projection.clone(),
            })),
            Resolved::Value(..) => Err(unsupported(
                "a closure environment reference for a value with no storage".to_string(),
            )),
        }
    }

    fn skip_while_next(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &Resolved,
    ) -> InterpResult<Option<Value>> {
        let Some(inner_target) = extend_target(target, RefProj::Field(0)) else {
            return Ok(None);
        };
        let Some(flag_target) = extend_target(target, RefProj::Field(1)) else {
            return Ok(None);
        };
        let Some(closure_target) = extend_target(target, RefProj::Field(2)) else {
            return Ok(None);
        };
        let inner_ty = first_generic_arg(self_ty).unwrap_or("");
        loop {
            let state = self.read_resolved(frames, target)?;
            let fields = iterator_fields(&state, self_ty)?;
            let done = matches!(fields.get(1), Some(Value::Bool(true)));
            let item = self.iterator_next_at(frames, inner_ty, &inner_target)?;
            let Some(item) = option_payload_opt(item) else {
                return Ok(Some(none_value()));
            };
            if done {
                return Ok(Some(some_value(item)));
            }
            let callable = self.read_resolved(frames, &closure_target)?;
            let env_ref = self.target_ref_value(frames, &closure_target)?;
            let arg = Value::Ref(RefTarget::value(item.clone()));
            let keep = self.call_callable_sync(frames, &callable, Some(env_ref), vec![arg])?;
            match keep {
                Value::Bool(true) => continue,
                Value::Bool(false) => {
                    self.write_resolved(frames, &flag_target, Value::Bool(true))?;
                    return Ok(Some(some_value(item)));
                }
                other => {
                    return Err(InterpError::type_mismatch(
                        "SkipWhile predicate result",
                        "a boolean",
                        other.kind_name(),
                    ))
                }
            }
        }
    }

    pub(super) fn iter_mut_element_ref(
        &self,
        frames: &[Frame],
        container: &Value,
        idx: usize,
    ) -> InterpResult<Value> {
        match container {
            Value::Ref(RefTarget::Place {
                frame,
                local,
                projection,
            }) => {
                let target = Resolved::Place {
                    frame_pos: self.frame_pos(frames, *frame)?,
                    local: *local,
                    projection: projection.clone(),
                };
                let base = self.read_resolved(frames, &target)?;
                // An element-entry suffix (see
                // `element_entry_suffix`) whose entered element
                // is not itself a sequence: the element is the
                // entered position advanced by the index. (A
                // sequence at the entered position is a slice
                // over an element's own array field — the plain
                // reading below; without the fat length the two
                // shapes cannot be told apart here.)
                if !matches!(
                    base,
                    Value::Tuple(_)
                        | Value::SparseVec { .. }
                        | Value::Bytes(_)
                        | Value::Str(_)
                ) {
                    if let Some(projection) = advance_entry_projection(projection, idx) {
                        return Ok(Value::Ref(RefTarget::Place {
                            frame: *frame,
                            local: *local,
                            projection,
                        }));
                    }
                }
                let mut projection = projection.clone();
                match base {
                    Value::Tuple(_) => projection.push(RefProj::Index(idx)),
                    Value::Enum { .. } => projection.push(RefProj::Field(idx)),
                    ref other => {
                        return Err(unsupported(format!(
                            "Iterator::next for slice::IterMut over a {} is not runtime-supported",
                            other.kind_name()
                        )))
                    }
                }
                Ok(Value::Ref(RefTarget::Place {
                    frame: *frame,
                    local: *local,
                    projection,
                }))
            }
            Value::Ref(RefTarget::Static { name, projection }) => {
                let target = Resolved::Static {
                    name: name.clone(),
                    projection: projection.clone(),
                };
                let base = self.read_resolved(frames, &target)?;
                // See the place arm: an element-entry suffix
                // over a non-sequence entered element advances
                // the entered position.
                if !matches!(
                    base,
                    Value::Tuple(_)
                        | Value::SparseVec { .. }
                        | Value::Bytes(_)
                        | Value::Str(_)
                ) {
                    if let Some(projection) = advance_entry_projection(projection, idx) {
                        return Ok(Value::Ref(RefTarget::Static {
                            name: name.clone(),
                            projection,
                        }));
                    }
                }
                let mut projection = projection.clone();
                match base {
                    Value::Tuple(_) => projection.push(RefProj::Index(idx)),
                    Value::Enum { .. } => projection.push(RefProj::Field(idx)),
                    ref other => {
                        return Err(unsupported(format!(
                            "Iterator::next for slice::IterMut over a {} is not runtime-supported",
                            other.kind_name()
                        )))
                    }
                }
                Ok(Value::Ref(RefTarget::Static {
                    name: name.clone(),
                    projection,
                }))
            }
            Value::FatRef { base, .. } => self.iter_mut_element_ref(frames, base, idx),
            other => Err(unsupported(format!(
                "Iterator::next for slice::IterMut over {} is not runtime-supported",
                other.kind_name()
            ))),
        }
    }

    /// One `Iterator::next` step on an iterator state value:
    /// the new state and the step's `Option` value. Most
    /// iterators are stepped here, on their state as a value;
    /// a `Map` whose state lives in a place is stepped in
    /// `map_next` instead, and this value-level step is the
    /// fallback for a `Map` behind a constant aggregate.
    fn next_on_state(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        state: &Value,
    ) -> InterpResult<(Value, Value)> {
        if self_ty.is_empty() {
            // No type tag: the caller stayed in generic form (a
            // trait method taken as a function value, an
            // impl-qualified name conversion could not resolve),
            // so the state value is all there is. The one state
            // shape whose stepping does not depend on the tag is
            // the owned-sequence state [items, next index] the
            // model builds for the array / Vec / Option
            // into-iterators: all three step it identically in
            // their tagged arm below, including marking the taken
            // slot uninitialized. Any other shape falls through to
            // the tag dispatch, which names it unsupported.
            if let Ok(fields) = iterator_fields(state, self_ty) {
                if fields.len() == 2 {
                    if let (Value::Tuple(items), Value::Uint(idx)) =
                        (&fields[0], &fields[1])
                    {
                        let idx = *idx as usize;
                        if idx < items.len() {
                            let item = items[idx].clone();
                            let mut items = items.clone();
                            items[idx] = Value::Uninit;
                            let new_state = rebuild_state(
                                state,
                                vec![
                                    Value::Tuple(items),
                                    Value::Uint((idx + 1) as u128),
                                ],
                            );
                            return Ok((new_state, some_value(item)));
                        }
                        return Ok((state.clone(), none_value()));
                    }
                }
            }
        }
        if self_ty.starts_with("std::option::Iter<")
            || self_ty.starts_with("std::option::IterMut<")
        {
            // Two state shapes reach here: the [elements, index]
            // state `IntoIterator` builds for `&Option` /
            // `&mut Option` (the element is the payload
            // reference itself, yielded as it is), and the shape
            // `Option::iter`'s own converted body builds — the
            // option nested inside one-field wrappers
            // (`Iter { inner: Item { .. } }`).
            let fields = iterator_fields(state, self_ty)?;
            if fields.len() == 2 {
                if let (Value::Tuple(items), Value::Uint(idx)) = (&fields[0], &fields[1]) {
                    let idx = *idx as usize;
                    if idx < items.len() {
                        let item = items[idx].clone();
                        let new_state = rebuild_state(
                            state,
                            vec![
                                Value::Tuple(items.clone()),
                                Value::Uint((idx + 1) as u128),
                            ],
                        );
                        return Ok((new_state, some_value(item)));
                    }
                    return Ok((state.clone(), none_value()));
                }
            }
            return match take_option_payload(state) {
                Ok(Some((new_state, item))) => Ok((new_state, some_value(item))),
                Ok(None) => Ok((state.clone(), none_value())),
                Err(()) => Err(InterpError::type_mismatch(
                    format!("iterator state of {self_ty}"),
                    "an Option value in one-field wrappers",
                    state.kind_name(),
                )),
            };
        }
        if is_str_split_ty(self_ty) {
            return self.str_split_step(state, self_ty, false);
        }
        let fields = iterator_fields(state, self_ty)?;
        let (new_fields, result) = if self_ty.starts_with("std::ops::RangeInclusive<") {
            if fields.len() != 3 {
                return Err(InterpError::type_mismatch(
                    "RangeInclusive state",
                    "[start, end, exhausted]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let (start, end, exhausted) = (
                fields[0].clone(),
                fields[1].clone(),
                fields[2].clone(),
            );
            if exhausted == Value::Bool(true) {
                (fields, none_value())
            } else {
                match compare_values(&start, &end)? {
                    Ordering::Less => (
                        vec![
                            range_increment(&start, range_elem_ty(self_ty))?,
                            end,
                            Value::Bool(false),
                        ],
                        some_value(start),
                    ),
                    Ordering::Equal => (
                        vec![start.clone(), end, Value::Bool(true)],
                        some_value(start),
                    ),
                    Ordering::Greater => (fields, none_value()),
                }
            }
        } else if self_ty.starts_with("std::ops::Range<") {
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Range state",
                    "[start, end]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let (start, end) = (fields[0].clone(), fields[1].clone());
            match compare_values(&start, &end)? {
                Ordering::Less => (
                    vec![range_increment(&start, range_elem_ty(self_ty))?, end],
                    some_value(start),
                ),
                _ => (fields, none_value()),
            }
        } else if self_ty.starts_with("std::iter::Empty<") {
            // An empty iterator yields `None` forever.
            (fields, none_value())
        } else if self_ty.starts_with("std::slice::Iter<")
            || self_ty.starts_with("std::collections::btree_set::Iter<")
            || self_ty.starts_with("std::collections::btree_map::Iter<")
            || self_ty.starts_with("std::collections::btree_map::Keys<")
            || self_ty.starts_with("std::collections::btree_map::Values<")
        {
            // The state is the real field layout (see
            // `slice_iter_value`): (ptr, end, marker) with element
            // pointers into the element snapshot. `next` hands out
            // the current pointer itself as the `&T` item and
            // advances it by one element.
            if fields.len() != 3 {
                return Err(InterpError::type_mismatch(
                    "slice::Iter state",
                    "[ptr, end, marker]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let (ptr_field, end) = (fields[0].clone(), fields[1].clone());
            let elem_size = slice_iter_elem_size(self_ty);
            // The index-pair shortcut reads the pair's trailing
            // index steps only: a pair carrying a byte-offset
            // step (a real-pointer iterator over scalar-backed
            // storage) denotes positions that reading drops,
            // so it belongs to the real-pointer step below.
            let index_pair = if iter_ptr_has_byte_offset(&ptr_field)
                || iter_ptr_has_byte_offset(&end)
            {
                None
            } else {
                match (
                    iter_ptr_index(&ptr_field, elem_size),
                    iter_ptr_index(&end, elem_size),
                ) {
                    (Some(cur), Some(last)) => Some((cur, last)),
                    _ => None,
                }
            };
            if let Some((cur, last)) = index_pair {
                if cur < last {
                    let advanced = iter_ptr_with_index(&ptr_field, cur + 1)
                        .expect("index step checked");
                    let item =
                        iter_ptr_value(&ptr_field).expect("pointer field checked");
                    let item = btree_pair_item(self_ty, item);
                    (vec![advanced, end, fields[2].clone()], some_value(item))
                } else {
                    (fields, none_value())
                }
            } else if let Some(step) =
                slice_iter_step_real(self, frames, self_ty, &fields, false)
            {
                step?
            } else {
                return Err(InterpError::type_mismatch(
                    "slice::Iter state",
                    "element pointers into the element snapshot",
                    format!("{} / {}", ptr_field.kind_name(), end.kind_name()),
                ));
            }
        } else if self_ty_is(self_ty, "std::str::Chars")
            || self_ty_is(self_ty, "core::str::Chars")
            || self_ty_is(self_ty, "std::str::Bytes")
            || self_ty_is(self_ty, "core::str::Bytes")
        {
            // The state is the real wrapper shape the `chars` /
            // `bytes` builtins build (see `str_chars_value` /
            // `str_bytes_value`): the step is computed on the
            // wrapped byte-slice iterator (see `step_str_iter`).
            let (new_state, item) = step_str_iter(state, self_ty, false)?;
            (
                iterator_fields(&new_state, self_ty)?,
                match item {
                    Some(value) => some_value(value),
                    None => none_value(),
                },
            )
        } else if self_ty.starts_with("std::array::IntoIter<")
            || self_ty.starts_with("std::vec::IntoIter<")
            || self_ty.starts_with("alloc::vec::IntoIter<")
            || self_ty.starts_with("std::option::IntoIter<")
            || self_ty.starts_with("std::collections::btree_set::IntoIter<")
            || self_ty.starts_with("std::collections::btree_map::IntoIter<")
            || (self_ty.starts_with("dyn ")
                && fields.len() == 2
                && matches!(fields[0], Value::Tuple(_))
                && matches!(fields[1], Value::Uint(_)))
        {
            // The `dyn` case: a provided method of the trait
            // object (a `max` / `fold` shim instantiated for
            // `dyn Iterator`) calls `next` with the trait-object
            // tag, while the receiver place has already resolved
            // through the dyn reference to the concrete value.
            // When that value is the owned-iterator state the
            // `IntoIter` models build - the only state form this
            // arm steps - the step is the same; any other
            // trait-object state keeps the unsupported error
            // below.
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "array::IntoIter state",
                    "[items, next index]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let items = match &fields[0] {
                Value::Tuple(items) => items.clone(),
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "array::IntoIter items",
                        "array value",
                        other.kind_name(),
                    ))
                }
            };
            let idx = match fields[1] {
                Value::Uint(v) => v as usize,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "array::IntoIter index",
                        "unsigned integer",
                        other.kind_name(),
                    ))
                }
            };
            if idx < items.len() {
                let item = items[idx].clone();
                // An owning iterator's yielded element has moved
                // out of the iterator's storage: mark the slot
                // uninitialized in the new state, so the state's
                // drop glue (which walks the element tuple) does
                // not drop it a second time. (`slice::Iter` is
                // stepped by its own real-layout arm above.)
                let items = if self_ty.starts_with("std::array::IntoIter<")
                    || self_ty.starts_with("std::vec::IntoIter<")
                    || self_ty.starts_with("alloc::vec::IntoIter<")
                    || self_ty.starts_with("std::option::IntoIter<")
                    || self_ty.starts_with("std::collections::btree_set::IntoIter<")
                    || self_ty.starts_with("std::collections::btree_map::IntoIter<")
                {
                    let mut items = items;
                    items[idx] = Value::Uninit;
                    items
                } else {
                    items
                };
                (
                    vec![Value::Tuple(items), Value::Uint((idx + 1) as u128)],
                    some_value(item),
                )
            } else {
                (fields, none_value())
            }
        } else if self_ty.starts_with("std::iter::Rev<") {
            // `Rev::next` is the inner iterator's `next_back`,
            // computed by the modelled back step
            // (`next_back_on_state`) on the inner state value,
            // so an adapter whose back step applies a closure
            // (a `Map`, a `Filter`) composes under `Rev`
            // exactly as it does at the top level; the leaf
            // shapes fall through to the pure state transition
            // inside that step. The Rev state is the one-field
            // wrapper around the inner state.
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Rev state",
                    "[inner iterator]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = self_ty
                .strip_prefix("std::iter::Rev<")
                .and_then(|rest| rest.strip_suffix('>'))
                .unwrap_or("");
            let (new_inner, item) =
                self.next_back_on_state(frames, inner_ty, fields[0].clone())?;
            (vec![new_inner], item)
        } else if self_ty.starts_with("std::slice::IterMut<") {
            // Two state shapes: the [container, next index,
            // length] state `IntoIterator` builds for `&mut Vec`
            // / `&mut [T; N]` / `&mut [T]` (the element handed
            // out is a reference to the element's place in the
            // container's own storage), and the real field layout
            // (ptr, end, marker) — see `slice_iter_value` — the
            // collected `<[T]>::iter_mut` body builds over a
            // raw-parts slice and the state `IntoIterator` builds
            // for the `&mut` containers. `next` hands out the
            // current pointer itself as the `&mut T` item and
            // advances it by one element.
            if fields.len() != 3 {
                return Err(InterpError::type_mismatch(
                    "slice::IterMut state",
                    "[container, next index, length] or [ptr, end, marker]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            if let (Value::Uint(idx), Value::Uint(len)) = (&fields[1], &fields[2]) {
                let (idx, len) = (*idx, *len);
                if idx < len {
                    let elem =
                        self.iter_mut_element_ref(frames, &fields[0], idx as usize)?;
                    (
                        vec![
                            fields[0].clone(),
                            Value::Uint(idx + 1),
                            Value::Uint(len),
                        ],
                        some_value(elem),
                    )
                } else {
                    (fields, none_value())
                }
            } else {
                let (ptr_field, end) = (fields[0].clone(), fields[1].clone());
                let elem_size = slice_iter_elem_size(self_ty);
                // The index-pair shortcut, gated as in the
                // shared iterator's step above: a pair carrying
                // a byte-offset step belongs to the real-pointer
                // step.
                let index_pair = if iter_ptr_has_byte_offset(&ptr_field)
                    || iter_ptr_has_byte_offset(&end)
                {
                    None
                } else {
                    match (
                        iter_ptr_index(&ptr_field, elem_size),
                        iter_ptr_index(&end, elem_size),
                    ) {
                        (Some(cur), Some(last)) => Some((cur, last)),
                        _ => None,
                    }
                };
                if let Some((cur, last)) = index_pair {
                    if cur < last {
                        let advanced = iter_ptr_with_index(&ptr_field, cur + 1)
                            .expect("index step checked");
                        let item = iter_ptr_value(&ptr_field)
                            .expect("pointer field checked");
                        (
                            vec![advanced, end, fields[2].clone()],
                            some_value(item),
                        )
                    } else {
                        (fields, none_value())
                    }
                } else if let Some(step) =
                    slice_iter_step_real(self, frames, self_ty, &fields, false)
                {
                    step?
                } else if let Some(remaining) = iter_zst_remaining(&fields) {
                    // The zero-sized form the real-layout step
                    // above declines: the end field is the
                    // terminal length as a provenance-free
                    // address and the front pointer never moves;
                    // each step hands it out again and decrements
                    // the stored terminal index, the same
                    // representation the back step reads (see
                    // `iter_zst_bounds`).
                    if remaining == 0 {
                        return Ok((state.clone(), none_value()));
                    }
                    let item =
                        iter_ptr_value(&ptr_field).expect("front pointer checked");
                    let (_, stored_end) =
                        iter_zst_bounds(&fields).expect("remaining checked");
                    let new_end = Value::Ref(RefTarget::Address(stored_end - 1));
                    return Ok((
                        rebuild_state(
                            state,
                            vec![fields[0].clone(), new_end, fields[2].clone()],
                        ),
                        some_value(item),
                    ));
                } else {
                    return Err(InterpError::type_mismatch(
                        "slice::IterMut state",
                        "element pointers into the element snapshot",
                        format!("{} / {}", ptr_field.kind_name(), end.kind_name()),
                    ));
                }
            }
        } else if self_ty.starts_with("std::iter::Once<") {
            // `Once` is a one-field struct wrapping an `Option`:
            // the payload is taken on the first step.
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Once state",
                    "[option]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            // The field wraps the option in the same one-field
            // chain `option::IntoIter` is built from
            // (`Once { inner: IntoIter { inner: Item { .. } } }`).
            match take_option_payload(&fields[0]) {
                Ok(Some((new_inner, item))) => (vec![new_inner], some_value(item)),
                Ok(None) => (fields, none_value()),
                Err(()) => {
                    return Err(InterpError::type_mismatch(
                        "Once state",
                        "an Option value in one-field wrappers",
                        fields[0].kind_name(),
                    ))
                }
            }
        } else if self_ty.starts_with("std::iter::Repeat<") {
            // `Repeat` yields a clone of its one field forever;
            // in the value model the clone is the value itself.
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Repeat state",
                    "[element]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let elem = fields[0].clone();
            (fields, some_value(elem))
        } else if self_ty.starts_with("std::iter::Enumerate<") {
            // `Enumerate` is (inner iterator, count): the step
            // is the inner step paired with the current count.
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Enumerate state",
                    "[iterator, count]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let count = match fields[1] {
                Value::Uint(v) => v,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "Enumerate count",
                        "unsigned integer",
                        other.kind_name(),
                    ))
                }
            };
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (new_inner, item) = self.next_on_state(frames, inner_ty, &fields[0])?;
            match option_payload_opt(item) {
                None => (vec![new_inner, Value::Uint(count)], none_value()),
                Some(x) => (
                    vec![new_inner, Value::Uint(count + 1)],
                    some_value(Value::Tuple(vec![Value::Uint(count), x])),
                ),
            }
        } else if self_ty.starts_with("std::iter::Copied<") {
            // `Copied` is (inner iterator): the step is the
            // inner step with the yielded reference followed to
            // the element it denotes.
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Copied state",
                    "[iterator]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (new_inner, item) = self.next_on_state(frames, inner_ty, &fields[0])?;
            match option_payload_opt(item) {
                None => (vec![new_inner], none_value()),
                Some(x) => {
                    let copied = self.resolve_value(frames, &x)?;
                    (vec![new_inner], some_value(copied))
                }
            }
        } else if self_ty.starts_with("std::iter::Cloned<") {
            // `Cloned` is (inner iterator): the step is the
            // inner step with the yielded reference cloned --
            // core's `Cloned::next` is the inner step composed
            // with `Clone::clone`, so the item is the model's
            // clone of the element the reference denotes.
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Cloned state",
                    "[iterator]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (new_inner, item) = self.next_on_state(frames, inner_ty, &fields[0])?;
            match option_payload_opt(item) {
                None => (vec![new_inner], none_value()),
                Some(x) => {
                    let elem_ty = ref_item_element_ty(inner_ty);
                    let cloned = self.clone_modelled_value(
                        frames,
                        &elem_ty,
                        "std::clone::Clone::clone",
                        &x,
                        None,
                    )?;
                    (vec![new_inner], some_value(cloned))
                }
            }
        } else if self_ty.starts_with("std::iter::Filter<") {
            // `Filter` is (inner iterator, predicate): the
            // step pulls inner items until the predicate
            // accepts one; a rejected item is dropped as it
            // is pulled, exactly as core's loop drops it. The
            // predicate takes the item by reference and is
            // handed a reference to the pulled item. This is
            // the value-level step, used when the Filter does
            // not live in a place (see `filter_next`): the
            // predicate is applied with a reference to the
            // closure value itself, so capture reads resolve
            // and writes to by-value captures are the explicit
            // constant-aggregate error inside the call.
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Filter state",
                    "[iterator, predicate]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let mut inner = fields[0].clone();
            let predicate = fields[1].clone();
            let mut result = none_value();
            loop {
                let (new_inner, item) = self.next_on_state(frames, inner_ty, &inner)?;
                inner = new_inner;
                let Some(x) = option_payload_opt(item) else {
                    break;
                };
                let arg = Value::Ref(RefTarget::value(x.clone()));
                let keep = self.call_callable_sync(frames, &predicate, None, vec![arg])?;
                match keep {
                    Value::Bool(true) => {
                        result = some_value(x);
                        break;
                    }
                    Value::Bool(false) => continue,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "Filter predicate result",
                            "a boolean",
                            other.kind_name(),
                        ))
                    }
                }
            }
            (vec![inner, predicate], result)
        } else if self_ty.starts_with("std::iter::FilterMap<") {
            // `FilterMap` is (inner iterator, closure): the
            // step pulls inner items, applies the closure to
            // each by value, and yields the first `Some` it
            // returns; an item whose closure returns `None` is
            // dropped as it is pulled, exactly as core's loop
            // drops it. This is the value-level step, used when
            // the FilterMap does not live in a place (see
            // `filter_map_next`): the closure is applied with a
            // reference to the closure value itself, so capture
            // reads resolve and writes to by-value captures are
            // the explicit constant-aggregate error inside the
            // call.
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "FilterMap state",
                    "[iterator, closure]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let mut inner = fields[0].clone();
            let closure = fields[1].clone();
            let mut result = none_value();
            loop {
                let (new_inner, item) = self.next_on_state(frames, inner_ty, &inner)?;
                inner = new_inner;
                let Some(x) = option_payload_opt(item) else {
                    break;
                };
                let mapped = self.call_callable_sync(frames, &closure, None, vec![x])?;
                if let Some(y) = option_payload(mapped)? {
                    result = some_value(y);
                    break;
                }
            }
            (vec![inner, closure], result)
        } else if self_ty.starts_with("std::iter::SkipWhile<") {
            // `SkipWhile` is (inner iterator, done flag,
            // predicate) — core's own field layout: the
            // predicate is consulted only until its first
            // failure, which sets the flag. The predicate takes
            // the item by reference and is handed a reference
            // to the pulled item.
            if fields.len() != 3 {
                return Err(InterpError::type_mismatch(
                    "SkipWhile state",
                    "[iterator, done, predicate]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let mut inner = fields[0].clone();
            let predicate = fields[2].clone();
            let mut done = matches!(fields.get(1), Some(Value::Bool(true)));
            let mut result = none_value();
            loop {
                let (new_inner, item) = self.next_on_state(frames, inner_ty, &inner)?;
                inner = new_inner;
                let Some(x) = option_payload_opt(item) else {
                    break;
                };
                if done {
                    result = some_value(x);
                    break;
                }
                let arg = Value::Ref(RefTarget::value(x.clone()));
                let keep = self.call_callable_sync(frames, &predicate, None, vec![arg])?;
                match keep {
                    Value::Bool(true) => continue,
                    Value::Bool(false) => {
                        done = true;
                        result = some_value(x);
                        break;
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            "SkipWhile predicate result",
                            "a boolean",
                            other.kind_name(),
                        ))
                    }
                }
            }
            (vec![inner, Value::Bool(done), predicate], result)
        } else if self_ty.starts_with("std::iter::Take<") {
            // `Take` is (inner iterator, remaining count): at
            // zero it yields `None` without touching the inner
            // iterator.
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Take state",
                    "[iterator, remaining]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let remaining = match fields[1] {
                Value::Uint(v) => v,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "Take remaining count",
                        "unsigned integer",
                        other.kind_name(),
                    ))
                }
            };
            if remaining == 0 {
                (fields, none_value())
            } else {
                let inner_ty = first_generic_arg(self_ty).unwrap_or("");
                let (new_inner, item) = self.next_on_state(frames, inner_ty, &fields[0])?;
                match option_payload_opt(item) {
                    None => (vec![new_inner, Value::Uint(remaining)], none_value()),
                    Some(x) => (
                        vec![new_inner, Value::Uint(remaining - 1)],
                        some_value(x),
                    ),
                }
            }
        } else if self_ty.starts_with("std::iter::Map<") {
            // `Map` is (inner iterator, closure): the step is
            // the inner step with the closure applied to the
            // element. This is the value-level step, used when
            // the Map does not live in a place (see `map_next`):
            // the closure is applied with a reference to the
            // value itself, so capture reads resolve and writes
            // to by-value captures are the explicit
            // constant-aggregate error inside the call.
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Map state",
                    "[iterator, closure]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (new_inner, item) = self.next_on_state(frames, inner_ty, &fields[0])?;
            match option_payload_opt(item) {
                None => (vec![new_inner, fields[1].clone()], none_value()),
                Some(x) => {
                    let mapped =
                        self.call_callable_sync(frames, &fields[1], None, vec![x])?;
                    (
                        vec![new_inner, fields[1].clone()],
                        some_value(mapped),
                    )
                }
            }
        } else if self_ty.starts_with("std::iter::Skip<") {
            // `Skip` is (inner iterator, remaining skip count):
            // the first step discards up to `count` items and
            // yields the one after them, and the count is spent
            // by that step whether or not the inner iterator had
            // enough items -- core's `next` delegates the first
            // step to `nth`, which has exactly this contract --
            // so every later step is a plain inner step.
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Skip state",
                    "[iterator, remaining]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let remaining = match fields[1] {
                Value::Uint(v) => v,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "Skip remaining count",
                        "unsigned integer",
                        other.kind_name(),
                    ))
                }
            };
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let mut inner = fields[0].clone();
            let mut to_skip = remaining;
            let mut result = none_value();
            loop {
                let (new_inner, item) =
                    self.iterator_next_state(frames, inner_ty, inner)?;
                inner = new_inner;
                let Some(x) = option_payload_opt(item) else {
                    break;
                };
                if to_skip > 0 {
                    to_skip -= 1;
                    continue;
                }
                result = some_value(x);
                break;
            }
            (vec![inner, Value::Uint(0)], result)
        } else if self_ty.starts_with("std::iter::Chain<") {
            // `Chain` is (first as an `Option`, second as an
            // `Option`): the first iterator is stepped while it
            // is present; the step that exhausts it drops it to
            // `None` and pulls from the second in the same call.
            // An exhausted second stays in place -- its own
            // state keeps answering `None` -- exactly as core
            // leaves it.
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Chain state",
                    "[Option<iterator>, Option<iterator>]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let type_args = generic_args_of(self_ty);
            let first_ty = type_args.first().map(String::as_str).unwrap_or("");
            let second_ty = type_args.get(1).map(String::as_str).unwrap_or("");
            let mut first = fields[0].clone();
            let mut second = fields[1].clone();
            let mut result = none_value();
            let mut yielded = false;
            match first.clone() {
                Value::Enum {
                    variant: 1,
                    fields: payload,
                } if payload.len() == 1 => {
                    let (new_first, item) = self.iterator_next_state(
                        frames,
                        first_ty,
                        payload.into_iter().next().expect("length checked"),
                    )?;
                    match option_payload_opt(item) {
                        Some(x) => {
                            first = some_value(new_first);
                            result = some_value(x);
                            yielded = true;
                        }
                        None => first = none_value(),
                    }
                }
                Value::Enum { variant: 0, .. } => {}
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "Chain first state",
                        "an Option holding the first iterator",
                        other.kind_name(),
                    ))
                }
            }
            if !yielded {
                match second.clone() {
                    Value::Enum {
                        variant: 1,
                        fields: payload,
                    } if payload.len() == 1 => {
                        let (new_second, item) = self.iterator_next_state(
                            frames,
                            second_ty,
                            payload.into_iter().next().expect("length checked"),
                        )?;
                        second = some_value(new_second);
                        result = item;
                    }
                    Value::Enum { variant: 0, .. } => {}
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            "Chain second state",
                            "an Option holding the second iterator",
                            other.kind_name(),
                        ))
                    }
                }
            }
            (vec![first, second], result)
        } else if self_ty.starts_with("std::iter::Zip<") {
            // `Zip` is (first iterator, second iterator,
            // index, len): the two counters belong to core's
            // random-access specialization, which the modelled
            // step does not use — the general `next` this arm
            // mirrors leaves them untouched, stepping the first
            // iterator and, only when it yields, the second;
            // the step yields the pair when both yield. When
            // the first is exhausted the second is not stepped
            // at all; when the second is exhausted the first's
            // item is consumed and lost — core's `next` drops
            // it the same way.
            if fields.len() != 4 {
                return Err(InterpError::type_mismatch(
                    "Zip state",
                    "[iterator, iterator, index, len]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let type_args = generic_args_of(self_ty);
            let first_ty = type_args.first().map(String::as_str).unwrap_or("");
            let second_ty = type_args.get(1).map(String::as_str).unwrap_or("");
            let counters = vec![fields[2].clone(), fields[3].clone()];
            let (new_first, first_item) =
                self.next_on_state(frames, first_ty, &fields[0])?;
            match option_payload_opt(first_item) {
                None => (
                    [vec![new_first, fields[1].clone()], counters].concat(),
                    none_value(),
                ),
                Some(x) => {
                    let (new_second, second_item) =
                        self.next_on_state(frames, second_ty, &fields[1])?;
                    match option_payload_opt(second_item) {
                        None => (
                            [vec![new_first, new_second], counters].concat(),
                            none_value(),
                        ),
                        Some(y) => (
                            [vec![new_first, new_second], counters].concat(),
                            some_value(Value::Tuple(vec![x, y])),
                        ),
                    }
                }
            }
        } else {
            return Err(unsupported(format!(
                "Iterator::next for {self_ty} is not runtime-supported"
            )));
        };
        Ok((rebuild_state(state, new_fields), result))

    }

    /// Step an iterator value to exhaustion with the
    /// [`Self::iterator_next`] machinery, returning the items in
    /// order. The iterator arrives by value; it is parked in a
    /// fresh allocation so the place-based machinery can step
    /// it. The iterator's final state is read back from that
    /// allocation and returned with the items: a consumer whose
    /// modelled body drops the iterator when it is done hands
    /// the final state back for that drop (see
    /// `consumed_iter_final`).
    pub(super) fn drain_iterator(
        &mut self,
        frames: &mut Vec<Frame>,
        iter_ty: &str,
        iter_value: Value,
    ) -> InterpResult<(Vec<Value>, Value)> {
        let id = self.alloc_insert(iter_value);
        let iter_ref = Value::Ref(RefTarget::Alloc {
            id,
            projection: Vec::new(),
        });
        let mut items = Vec::new();
        loop {
            match self.iterator_next(frames, iter_ty, &iter_ref)? {
                Value::Enum {
                    variant: 1,
                    fields: item,
                } if item.len() == 1 => {
                    items.push(item.into_iter().next().expect("length checked"));
                }
                Value::Enum { variant: 0, .. } => break,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "iterator drain",
                        "an Option value from Iterator::next",
                        other.kind_name(),
                    ))
                }
            }
        }
        let final_state = self.alloc_value(id).unwrap_or(Value::Uninit);
        Ok((items, final_state))
    }

    /// The `(lower, upper)` bounds of an inner iterator
    /// state's `size_hint`, for the adapter arms of
    /// [`Self::iterator_size_hint`]: the state is parked as a
    /// value reference and the hint the modelled step reports
    /// for it is taken apart.
    fn size_hint_parts(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        state: Value,
    ) -> InterpResult<(u128, Option<u128>)> {
        let inner_ref = Value::Ref(RefTarget::value(state));
        match self.iterator_size_hint(frames, self_ty, &inner_ref)? {
            Value::Tuple(pair) if pair.len() == 2 => {
                let lower = match pair[0] {
                    Value::Uint(v) => v,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            format!("size_hint lower bound of {self_ty}"),
                            "unsigned integer",
                            other.kind_name(),
                        ))
                    }
                };
                let upper = match option_payload_opt(pair[1].clone()) {
                    Some(Value::Uint(v)) => Some(v),
                    Some(ref other) => {
                        return Err(InterpError::type_mismatch(
                            format!("size_hint upper bound of {self_ty}"),
                            "unsigned integer",
                            other.kind_name(),
                        ))
                    }
                    None => None,
                };
                Ok((lower, upper))
            }
            other => Err(InterpError::type_mismatch(
                format!("size_hint of {self_ty}"),
                "(usize, Option<usize>)",
                other.kind_name(),
            )),
        }
    }

    /// The `ExactSizeIterator::len` of a slice-state
    /// iterator: the remaining element count the state's
    /// unified bounds encode (see
    /// `slice_iter_front_and_remaining`). Core's own body for
    /// these types reads the count out of the end pointer's
    /// address, which is the terminal index rather than the
    /// remaining count once the front has stepped — and, for
    /// a snapshot state, the snapshot's address word rather
    /// than a count at all — so the modelled steps'
    /// accounting is the only faithful reading.
    pub(super) fn iterator_len(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        ref_arg: &Value,
    ) -> InterpResult<Value> {
        let target = self.deref_value(frames, ref_arg)?;
        let state = self.read_resolved(frames, &target)?;
        let elem_tag = first_generic_arg(self_ty).unwrap_or("");
        let (_, remaining) =
            self.slice_iter_front_and_remaining(frames, &state, self_ty, elem_tag)?;
        Ok(Value::Uint(u128::from(remaining)))
    }

    /// The remaining slice of a `slice::Iter` /
    /// `slice::IterMut` state: a fat reference over the
    /// state's current front pointer with the remaining
    /// count the unified bounds give — the value core's
    /// `make_slice` builds with `from_raw_parts`, whose
    /// length operand its body reads out of the end
    /// pointer's address (see `iterator_len`). The receiver
    /// is a reference for `as_slice` / `as_mut_slice` and
    /// the state itself for `into_slice`.
    pub(super) fn slice_iter_as_slice(
        &mut self,
        frames: &mut Vec<Frame>,
        elem_tag: &str,
        arg: &Value,
    ) -> InterpResult<Value> {
        let state = if matches!(arg, Value::Ref(_)) {
            let target = self.deref_value(frames, arg)?;
            self.read_resolved(frames, &target)?
        } else {
            arg.clone()
        };
        let (front, remaining) =
            self.slice_iter_front_and_remaining(frames, &state, "slice iterator", elem_tag)?;
        Ok(Value::FatRef {
            base: Box::new(front),
            len: u128::from(remaining),
        })
    }

    /// The (front pointer, remaining count) a slice-iterator
    /// state denotes, in the unified reading the modelled
    /// steps maintain (see `iter_zst_bounds`): the front
    /// pointer is the state's first field unwrapped, and the
    /// remaining count is the distance from the current
    /// index to the terminal index. A state whose two
    /// pointers are both address-only — the pair the step
    /// machine's own address form steps (see
    /// `slice_iter_step_real`) — reads the same way over the
    /// addresses themselves: their distance in element
    /// units (in address units for a zero-sized element,
    /// whose step advances the address itself). Any other
    /// unrecognised shape keeps the bounds reading's error.
    ///
    /// Before either reading, a pair denoting two positions
    /// in the same storage is measured directly with the
    /// `offset_from` distance (`pointer_distance`): that
    /// reading composes a trailing byte offset past an index
    /// step into the position it denotes, which the bounds
    /// reading's index extraction silently drops — the shape
    /// a real-pointer iterator over a scalar-backed slice
    /// takes (an `IterMut` over `slice::from_mut(&mut x)`,
    /// whose front denotes the scalar place itself and whose
    /// end denotes that same place one element further on).
    /// Wherever the distance reading applies to a pair the
    /// bounds reading also serves, the two agree.
    fn slice_iter_front_and_remaining(
        &self,
        frames: &[Frame],
        state: &Value,
        self_ty: &str,
        elem_tag: &str,
    ) -> InterpResult<(Value, u64)> {
        let fields = iterator_fields(state, self_ty)?;
        if fields.len() != 3 {
            return Err(InterpError::type_mismatch(
                "slice::Iter state",
                "[ptr, end, marker]",
                format!("{} field(s)", fields.len()),
            ));
        }
        let front = iter_ptr_value(&fields[0]).ok_or_else(|| {
            InterpError::type_mismatch(
                "slice::Iter state",
                "an element pointer in the first field",
                fields[0].kind_name(),
            )
        })?;
        if let Some(end_ptr) = iter_ptr_value(&fields[1]) {
            if let Some(remaining) =
                self.slice_iter_pointer_distance(frames, &front, &end_ptr, elem_tag)
            {
                return Ok((front, remaining));
            }
        }
        match slice_iter_bounds(state, self_ty) {
            Ok((cur, end)) => Ok((front, end.saturating_sub(cur) as u64)),
            Err(bounds_err) => {
                let (Some(end_ptr), Some((size, _))) = (
                    iter_ptr_value(&fields[1]),
                    self.layout_of(elem_tag)
                        .and_then(|(size, align)| u64::try_from(size).ok().map(|s| (s, align))),
                ) else {
                    return Err(bounds_err);
                };
                let (
                    Value::Ref(RefTarget::Address(cur)),
                    Value::Ref(RefTarget::Address(end)),
                ) = (&front, &end_ptr)
                else {
                    return Err(bounds_err);
                };
                let bytes = end.saturating_sub(*cur);
                let remaining = if size == 0 {
                    bytes
                } else if bytes % size == 0 {
                    bytes / size
                } else {
                    return Err(bounds_err);
                };
                Ok((front, remaining))
            }
        }
    }

    /// The remaining element count of a slice-iterator state
    /// whose front and end pointers denote two positions in
    /// the same storage, measured with the `offset_from`
    /// distance reading (`pointer_distance`) over the resolved
    /// pointers; `None` when either pointer denotes no storage
    /// position (an address-only or null pointer), when the
    /// two denote unrelated storage, or when the end does not
    /// lie at or past the front.
    fn slice_iter_pointer_distance(
        &self,
        frames: &[Frame],
        front: &Value,
        end: &Value,
        elem_tag: &str,
    ) -> Option<u64> {
        let front = self.deref_value(frames, front).ok()?;
        let end = self.deref_value(frames, end).ok()?;
        let args = [elem_tag.to_string()];
        let distance = self.pointer_distance(&end, &front, &args).ok()??;
        u64::try_from(distance).ok()
    }

    /// `Iterator::size_hint` for the modelled iterators: the
    /// `(lower, Option<upper>)` pair the standard library reports
    /// for each, computed from the iterator's state the same way
    /// [`Self::iterator_next`] steps it.
    ///
    /// Registering the name stops the driver collecting the real
    /// `size_hint` bodies. Those bodies cannot run against the
    /// modelled states: `env::Args::size_hint` — the caller that
    /// motivated this — delegates through `ArgsOs` to
    /// `vec::IntoIter::size_hint`, whose body reads the real
    /// struct layout (the buffer pointer and end fields) off the
    /// model's snapshot state and fails on an uninitialized deep
    /// field; the range and string iterators' bodies project
    /// their real layouts the same way. The values below are the
    /// exact std answers for each modelled state: an exact count
    /// where std knows one, `(0, None)` where std reports the
    /// trait default (the channel iterators do not override
    /// `size_hint`). Iterators the model does not serve fall
    /// back to that same trait default: `(0, None)` is valid
    /// for every iterator, so the call succeeds where the
    /// collected default body would have run before this
    /// builtin was registered — at the price of returning
    /// the default rather than the exact value for an
    /// unmodelled iterator that overrides `size_hint`.
    pub(super) fn iterator_size_hint(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        ref_arg: &Value,
    ) -> InterpResult<Value> {
        let target = self.deref_value(frames, ref_arg)?;
        let state = self.read_resolved(frames, &target)?;
        fn hint(lower: u128, upper: Option<u128>) -> Value {
            Value::Tuple(vec![
                Value::Uint(lower),
                match upper {
                    Some(v) => some_value(Value::Uint(v)),
                    None => none_value(),
                },
            ])
        }
        if self_ty.starts_with("std::sync::mpsc::Iter<")
            || self_ty.starts_with("std::sync::mpsc::IntoIter<")
        {
            // Not overridden in std: the trait default.
            return Ok(hint(0, None));
        }
        if self_ty_is(self_ty, "std::env::Args") || self_ty_is(self_ty, "std::env::ArgsOs") {
            // Delegates to the inner `vec::IntoIter`'s exact count.
            let depth = if self_ty_is(self_ty, "std::env::Args") { 3 } else { 2 };
            let mut inner = &state;
            for _ in 0..depth {
                inner = match inner {
                    Value::Enum { fields, .. } if fields.len() == 1 => &fields[0],
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("{self_ty} state"),
                            "nested one-field iterator wrappers",
                            other.kind_name(),
                        ))
                    }
                };
            }
            let (items, idx) = owned_iter_parts(inner, self_ty)?;
            let n = items.len().saturating_sub(idx) as u128;
            return Ok(hint(n, Some(n)));
        }
        if self_ty.starts_with("std::iter::Rev<") {
            // `Rev::size_hint` is its inner iterator's.
            let inner_ty = self_ty
                .strip_prefix("std::iter::Rev<")
                .and_then(|rest| rest.strip_suffix('>'))
                .unwrap_or("");
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Rev state",
                    "[inner iterator]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ref = Value::Ref(RefTarget::value(fields[0].clone()));
            return self.iterator_size_hint(frames, inner_ty, &inner_ref);
        }
        if self_ty.starts_with("std::iter::Copied<")
            || self_ty.starts_with("std::iter::Cloned<")
        {
            // `Copied::size_hint` and `Cloned::size_hint` are
            // the inner iterator's hint, unchanged: both
            // adapters yield exactly the inner iterator's
            // items, in order.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 1 {
                let label = if self_ty.starts_with("std::iter::Copied<") {
                    "Copied state"
                } else {
                    "Cloned state"
                };
                return Err(InterpError::type_mismatch(
                    label,
                    "[inner iterator]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let inner_ref = Value::Ref(RefTarget::value(fields[0].clone()));
            return self.iterator_size_hint(frames, inner_ty, &inner_ref);
        }
        if self_ty.starts_with("std::iter::Skip<") {
            // `Skip::size_hint` is the inner hint with the
            // remaining skip count saturated off both bounds.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Skip state",
                    "[iterator, remaining]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let remaining = match fields[1] {
                Value::Uint(v) => v,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "Skip remaining count",
                        "unsigned integer",
                        other.kind_name(),
                    ))
                }
            };
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (lower, upper) =
                self.size_hint_parts(frames, inner_ty, fields[0].clone())?;
            return Ok(hint(
                lower.saturating_sub(remaining),
                upper.map(|u| u.saturating_sub(remaining)),
            ));
        }
        if self_ty.starts_with("std::iter::Chain<") {
            // `Chain::size_hint` adds the two sides' hints; a
            // side already dropped to `None` counts as the
            // exact `(0, Some(0))`, and the upper bound is
            // known only when both sides' are.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Chain state",
                    "[Option<iterator>, Option<iterator>]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let type_args = generic_args_of(self_ty);
            let mut lower = 0u128;
            let mut upper = Some(0u128);
            for (i, side) in fields.iter().enumerate() {
                let side_ty = type_args.get(i).map(String::as_str).unwrap_or("");
                let (l, u) = match side {
                    Value::Enum {
                        variant: 1,
                        fields: payload,
                    } if payload.len() == 1 => {
                        self.size_hint_parts(frames, side_ty, payload[0].clone())?
                    }
                    Value::Enum { variant: 0, .. } => (0, Some(0)),
                    other => {
                        return Err(InterpError::type_mismatch(
                            "Chain state",
                            "an Option holding an iterator",
                            other.kind_name(),
                        ))
                    }
                };
                lower = lower.saturating_add(l);
                upper = match (upper, u) {
                    (Some(a), Some(b)) => a.checked_add(b),
                    _ => None,
                };
            }
            return Ok(hint(lower, upper));
        }
        if self_ty.starts_with("std::iter::Zip<") {
            // `Zip::size_hint` pairs the two sides' hints: the
            // lower bound is the smaller of the two lower
            // bounds; the upper bound is the smaller of the two
            // when both are known, the known one when only one
            // side knows its upper bound, and unknown only when
            // both are.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 4 {
                return Err(InterpError::type_mismatch(
                    "Zip state",
                    "[iterator, iterator, index, len]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let type_args = generic_args_of(self_ty);
            let first_ty = type_args.first().map(String::as_str).unwrap_or("");
            let second_ty = type_args.get(1).map(String::as_str).unwrap_or("");
            let (a_lower, a_upper) =
                self.size_hint_parts(frames, first_ty, fields[0].clone())?;
            let (b_lower, b_upper) =
                self.size_hint_parts(frames, second_ty, fields[1].clone())?;
            let upper = match (a_upper, b_upper) {
                (Some(x), Some(y)) => Some(x.min(y)),
                (Some(x), None) => Some(x),
                (None, Some(y)) => Some(y),
                (None, None) => None,
            };
            return Ok(hint(a_lower.min(b_lower), upper));
        }
        // The slice chunk / window family: `Chunks`, `ChunksMut`,
        // `RChunks`, `RChunksMut` (state [remaining slice, chunk
        // size]), their `Exact` variants (state [remaining exact
        // slice, remainder slice, chunk size]) and `Windows` (the
        // native [tail slice, window size] state, or the `windows`
        // builtin's [slice, window size, next index] snapshot).
        // None of these is served by the generic fallback below:
        // std's `size_hint` for each is exact, and
        // `ExactSizeIterator::len`'s collected default body
        // asserts `size_hint`'s upper bound equals its lower
        // bound, so the trait-default `(0, None)` fallback turned
        // every `len()` call into a spurious assertion failure
        // inside core's `assert_eq!`. The counts are std's own
        // formulas over the remaining slice: ceiling division by
        // the chunk size (floor division for the exact variants,
        // whose stored slice holds only whole chunks), and
        // `remaining - size + 1` for windows once the window fits.
        // A state shape this arm cannot read is an explicit
        // error (unsupported, exit 2), never the `(0, None)`
        // fallback that re-arms the assertion.
        {
            let bare_ty = self_ty
                .strip_prefix("&mut ")
                .or_else(|| self_ty.strip_prefix('&'))
                .unwrap_or(self_ty);
            let is_windows = bare_ty.starts_with("std::slice::Windows<");
            let is_chunk = [
                "std::slice::Chunks<",
                "std::slice::ChunksMut<",
                "std::slice::RChunks<",
                "std::slice::RChunksMut<",
            ]
            .iter()
            .any(|prefix| bare_ty.starts_with(prefix));
            let is_chunk_exact = [
                "std::slice::ChunksExact<",
                "std::slice::ChunksExactMut<",
                "std::slice::RChunksExact<",
                "std::slice::RChunksExactMut<",
            ]
            .iter()
            .any(|prefix| bare_ty.starts_with(prefix));
            if is_windows || is_chunk || is_chunk_exact {
                let fields = iterator_fields(&state, self_ty)?;
                // A chunk/window size: a plain integer, or a
                // `NonZero` wrapped in one-field aggregates.
                fn size_of(f: &Value, self_ty: &str) -> InterpResult<u128> {
                    let mut cur = f;
                    loop {
                        match cur {
                            Value::Uint(v) => return Ok(*v),
                            Value::Int(v) if *v >= 0 => return Ok(*v as u128),
                            Value::Enum { fields, .. } if fields.len() == 1 => {
                                cur = &fields[0];
                            }
                            Value::Tuple(fields) if fields.len() == 1 => {
                                cur = &fields[0];
                            }
                            other => {
                                return Err(InterpError::type_mismatch(
                                    format!("size field of {self_ty}"),
                                    "non-zero integer",
                                    other.kind_name(),
                                ))
                            }
                        }
                    }
                }
                if is_windows {
                    let (rem, size) = match fields.len() {
                        2 => {
                            let (_, len) =
                                self.slice_elements_ref(frames, &fields[0])?;
                            (len, size_of(&fields[1], self_ty)?)
                        }
                        3 => {
                            let (_, len) =
                                self.slice_elements_ref(frames, &fields[0])?;
                            let index = match fields[2] {
                                Value::Uint(v) => v,
                                ref other => {
                                    return Err(InterpError::type_mismatch(
                                        "Windows index",
                                        "integer",
                                        other.kind_name(),
                                    ))
                                }
                            };
                            (
                                len.saturating_sub(index),
                                size_of(&fields[1], self_ty)?,
                            )
                        }
                        _ => {
                            return Err(InterpError::type_mismatch(
                                "Windows state",
                                "[slice, window size] or [slice, window size, next index]",
                                format!("{} field(s)", fields.len()),
                            ))
                        }
                    };
                    let n = if size == 0 || rem < size {
                        0
                    } else {
                        rem - size + 1
                    };
                    return Ok(hint(n, Some(n)));
                }
                if is_chunk {
                    // The shared variants store [remaining slice,
                    // chunk size]; the `Mut` variants add a
                    // trailing marker field to that pair (see the
                    // three-field arm below).
                    let (rem, size) = match fields.len() {
                        2 => {
                            let (_, rem) =
                                self.slice_elements_ref(frames, &fields[0])?;
                            (rem, size_of(&fields[1], self_ty)?)
                        }
                        3 => {
                            // The `Mut` states reach here as
                            // [remaining fat slice, chunk size,
                            // marker]: the raw slice stays one fat
                            // value and the size is the middle
                            // field. (A flattened [pointer, length,
                            // size] triple would put a non-integer
                            // marker last instead of an integer
                            // size; the shape is told apart by the
                            // last field.)
                            let marker_last = match &fields[2] {
                                Value::Unit => true,
                                Value::Enum { fields, .. } => fields.is_empty(),
                                _ => false,
                            };
                            if marker_last {
                                let (_, rem) =
                                    self.slice_elements_ref(frames, &fields[0])?;
                                (rem, size_of(&fields[1], self_ty)?)
                            } else {
                                let rem = match fields[1] {
                                    Value::Uint(v) => v,
                                    _ => {
                                        let (_, rem) = self
                                            .slice_elements_ref(frames, &fields[0])?;
                                        rem
                                    }
                                };
                                (rem, size_of(&fields[2], self_ty)?)
                            }
                        }
                        _ => {
                            return Err(InterpError::type_mismatch(
                                format!("{self_ty} state"),
                                "[remaining slice, chunk size]",
                                format!("{} field(s)", fields.len()),
                            ))
                        }
                    };
                    if size == 0 {
                        return Err(unsupported(format!(
                            "size_hint of {self_ty} with a zero chunk size"
                        )));
                    }
                    let n = rem.div_ceil(size);
                    return Ok(hint(n, Some(n)));
                }
                // The exact variants: the stored slice holds only
                // whole chunks, so the count is floor division.
                // The shared variants store [remaining slice,
                // remainder slice, chunk size]; the `Mut` variants
                // add a trailing marker field to that triple (the
                // four-field arm), and a fully flattened raw
                // fat-pointer form would give five fields.
                let (rem, size) = match fields.len() {
                    3 => {
                        let (_, rem) =
                            self.slice_elements_ref(frames, &fields[0])?;
                        (rem, size_of(&fields[2], self_ty)?)
                    }
                    4 => {
                        // The `Mut` exact states: [remaining fat
                        // slice, remainder fat slice, chunk size,
                        // marker].
                        let (_, rem) =
                            self.slice_elements_ref(frames, &fields[0])?;
                        (rem, size_of(&fields[2], self_ty)?)
                    }
                    5 => {
                        let rem = match fields[1] {
                            Value::Uint(v) => v,
                            _ => {
                                let (_, rem) =
                                    self.slice_elements_ref(frames, &fields[0])?;
                                rem
                            }
                        };
                        (rem, size_of(&fields[4], self_ty)?)
                    }
                    _ => {
                        return Err(InterpError::type_mismatch(
                            format!("{self_ty} state"),
                            "[remaining slice, remainder slice, chunk size]",
                            format!("{} field(s)", fields.len()),
                        ))
                    }
                };
                if size == 0 {
                    return Err(unsupported(format!(
                        "size_hint of {self_ty} with a zero chunk size"
                    )));
                }
                let n = rem / size;
                return Ok(hint(n, Some(n)));
            }
        }
        let fields = iterator_fields(&state, self_ty)?;
        if self_ty.starts_with("std::ops::RangeInclusive<") {
            if fields.len() != 3 {
                return Err(InterpError::type_mismatch(
                    "RangeInclusive state",
                    "[start, end, exhausted]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            if fields[2] == Value::Bool(true) {
                return Ok(hint(0, Some(0)));
            }
            return Ok(match range_count(
                &fields[0],
                &fields[1],
                true,
                range_elem_ty(self_ty),
            )? {
                Some(n) => hint(n, Some(n)),
                None => hint(u128::from(u64::MAX), None),
            });
        }
        if self_ty.starts_with("std::ops::Range<") {
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Range state",
                    "[start, end]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            return Ok(match range_count(
                &fields[0],
                &fields[1],
                false,
                range_elem_ty(self_ty),
            )? {
                Some(n) => hint(n, Some(n)),
                None => hint(u128::from(u64::MAX), None),
            });
        }
        if self_ty.starts_with("std::iter::Empty<") {
            return Ok(hint(0, Some(0)));
        }
        if self_ty.starts_with("std::slice::Iter<")
            || self_ty.starts_with("std::collections::btree_set::Iter<")
            || self_ty.starts_with("std::collections::btree_map::Iter<")
            || self_ty.starts_with("std::collections::btree_map::Keys<")
            || self_ty.starts_with("std::collections::btree_map::Values<")
            || self_ty.starts_with("std::slice::IterMut<")
        {
            // The real layout's pointer pair: the remaining count
            // is the distance between the two element pointers.
            // A pair carrying a byte-offset step (a real-pointer
            // iterator over scalar-backed storage) denotes its
            // positions as distances from the storage, which the
            // index reading drops: its remaining count is the
            // unified distance reading's.
            let byte_offset_pair = iterator_fields(&state, self_ty)
                .map(|fields| {
                    fields.len() >= 2
                        && (iter_ptr_has_byte_offset(&fields[0])
                            || iter_ptr_has_byte_offset(&fields[1]))
                })
                .unwrap_or(false);
            let n = if byte_offset_pair {
                let elem_tag = first_generic_arg(self_ty).unwrap_or("");
                let (_, rem) = self.slice_iter_front_and_remaining(
                    frames,
                    &state,
                    self_ty,
                    elem_tag,
                )?;
                rem as u128
            } else {
                let (cur, end) = slice_iter_bounds(&state, self_ty)?;
                end.saturating_sub(cur) as u128
            };
            return Ok(hint(n, Some(n)));
        }
        if self_ty_is(self_ty, "std::str::Chars")
            || self_ty_is(self_ty, "core::str::Chars")
            || self_ty_is(self_ty, "std::str::Bytes")
            || self_ty_is(self_ty, "core::str::Bytes")
        {
            // The remaining byte count is the wrapped byte
            // iterator's remaining length, read off its pointer
            // pair. `Bytes::size_hint` is that count exactly;
            // std's `Chars::size_hint` bounds the char count by
            // it: a char is at most 4 bytes, so the lower bound
            // is the byte count divided by 4 rounded up.
            let inner = str_iter_inner(&state, self_ty)?;
            let (cur, end) = slice_iter_bounds(&inner, self_ty)?;
            let bytes = end.saturating_sub(cur) as u128;
            if self_ty_is(self_ty, "std::str::Chars")
                || self_ty_is(self_ty, "core::str::Chars")
            {
                return Ok(hint(bytes.div_ceil(4), Some(bytes)));
            }
            return Ok(hint(bytes, Some(bytes)));
        }
        if self_ty.starts_with("std::array::IntoIter<")
            || self_ty.starts_with("std::vec::IntoIter<")
            || self_ty.starts_with("alloc::vec::IntoIter<")
            || self_ty.starts_with("std::collections::btree_set::IntoIter<")
            || self_ty.starts_with("std::collections::btree_map::IntoIter<")
        {
            let (items, idx) = owned_iter_parts(&state, self_ty)?;
            let remaining = &items[idx.min(items.len())..];
            let n = remaining.len() as u128;
            return Ok(hint(n, Some(n)));
        }
        if self_ty.starts_with("std::iter::Filter<") {
            // `Filter::size_hint` is `(0, inner upper)`: the
            // predicate may reject every remaining item, so
            // the lower bound is zero, and no more items can
            // come out than go in, so the upper bound is the
            // inner iterator's.
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Filter state",
                    "[iterator, predicate]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (_, upper) =
                self.size_hint_parts(frames, inner_ty, fields[0].clone())?;
            return Ok(hint(0, upper));
        }
        if self_ty.starts_with("std::iter::FilterMap<") {
            // `FilterMap::size_hint` is `(0, inner upper)`, as
            // `Filter`'s is: the closure may drop every
            // remaining item, so the lower bound is zero, and
            // no more items can come out than go in, so the
            // upper bound is the inner iterator's.
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "FilterMap state",
                    "[iterator, closure]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (_, upper) =
                self.size_hint_parts(frames, inner_ty, fields[0].clone())?;
            return Ok(hint(0, upper));
        }
        // Any other iterator: the trait default, `(0, None)`.
        // That pair is a valid `size_hint` answer for every
        // iterator — a lower bound of zero and no upper bound
        // satisfy the contract unconditionally — and it is the
        // exact answer for iterators that do not override
        // `size_hint` (a plain user iterator behind adapters
        // such as `Fuse` is the common case). For an iterator
        // that DOES override `size_hint` but whose state the
        // model does not serve, this returns the contract-valid
        // default rather than the implementation's exact value;
        // callers may then under-reserve or check bounds less
        // tightly, but never observe an invalid bound. (Before
        // this fallback existed, registering the builtin name
        // removed the default body from collection, so such
        // calls failed outright even when the default was the
        // whole story.)
        Ok(hint(0, None))
    }
}

impl<'a> Interpreter<'a> {
    /// Execute `Iterator::collect` / `FromIterator::from_iter`:
    /// drain the modelled `next` state transition into the
    /// collection the call's target generic names. The value
    /// model has a native shape for each supported target: a
    /// `Vec` is its element list, a `String` is accumulated text
    /// (chars arrive as code-point integers, string items as
    /// string values or references to them), a boxed slice is a
    /// box over the element tuple and a boxed `str` a box over
    /// the accumulated text (the shapes `Vec::into_boxed_slice`
    /// and `String::into_boxed_str` build), an `Rc` / `Arc`
    /// slice is a fresh counted handle over the element tuple
    /// (the shape the counted-slice `From` model builds), and a
    /// `VecDeque` is the aggregate `vec_deque_from_value`
    /// assembles. Every other target is named in an explicit
    /// unsupported error rather than approximated.
    pub(super) fn exec_collect(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &str,
        mut state: Value,
    ) -> InterpResult<Value> {
        let mut items = Vec::new();
        loop {
            self.tick_step()?;
            let (next_state, item) = self.iterator_next_state(frames, self_ty, state)?;
            state = next_state;
            match option_payload(item)? {
                None => break,
                Some(item) => items.push(item),
            }
        }
        self.note_consumed_iter_final(state);
        if target.starts_with("std::vec::Vec<") || target.starts_with("alloc::vec::Vec<") {
            return Ok(Value::Enum {
                variant: 0,
                fields: items,
            });
        }
        if target == "std::string::String" || target == "alloc::string::String" {
            let out = self.collect_items_to_string(frames, items, "collect into String")?;
            return Ok(Value::Str(out));
        }
        if target.starts_with("std::boxed::Box<") || target.starts_with("alloc::boxed::Box<") {
            let Some(first) = first_generic_arg(target) else {
                return Err(unsupported(format!(
                    "collect into {target} is not runtime-supported"
                )));
            };
            if first == "str" {
                let out = self.collect_items_to_string(frames, items, "collect into Box<str>")?;
                let boxed = self.new_box(Value::Str(out));
                self.record_box_ty(&boxed, Ty::Str);
                return Ok(boxed);
            }
            if is_slice_tag(first) {
                // `FromIterator<T> for Box<[T]>`: the collected
                // items in a fresh boxed slice, built by the
                // shared boxed-slice construction.
                return self.boxed_slice_value(
                    Value::Tuple(items),
                    ty_from_tag(first),
                    "collect into Box<[T]>",
                );
            }
        }
        if (target.starts_with("std::rc::Rc<")
            || target.starts_with("alloc::rc::Rc<")
            || target.starts_with("std::sync::Arc<")
            || target.starts_with("alloc::sync::Arc<"))
            && first_generic_arg(target).is_some_and(|first| first.starts_with('['))
        {
            return Ok(self.rc_new_handle(Value::Tuple(items)));
        }
        if target.starts_with("std::collections::VecDeque<") {
            return self.vec_deque_from_value(
                frames,
                Value::Enum {
                    variant: 0,
                    fields: items,
                },
            );
        }
        Err(unsupported(format!(
            "collect into {target} is not runtime-supported"
        )))
    }

    /// Accumulate collected items into text, the way the
    /// `String` (and boxed-`str`) `FromIterator` impls do: char
    /// items arrive as code-point integers, string items as
    /// string values or references to them.
    fn collect_items_to_string(
        &mut self,
        frames: &mut Vec<Frame>,
        items: Vec<Value>,
        context: &str,
    ) -> InterpResult<String> {
        let mut out = String::new();
        for item in items {
            match item {
                Value::Uint(cp) => {
                    let c = char::from_u32(cp as u32).ok_or_else(|| {
                        InterpError::type_mismatch(
                            context,
                            "valid char code point",
                            format!("{cp}"),
                        )
                    })?;
                    out.push(c);
                }
                Value::Str(s) => out.push_str(&s),
                Value::Ref(_) => {
                    let s = self.resolve_string(frames, &item)?;
                    out.push_str(&s);
                }
                other => {
                    return Err(InterpError::type_mismatch(
                        context,
                        "char or string item",
                        other.kind_name(),
                    ));
                }
            }
        }
        Ok(out)
    }

    /// Execute `Iterator::fold` / `Iterator::for_each`: drive
    /// the modelled `next` state transition over the owned
    /// iterator value and invoke the closure operand per item —
    /// exactly the loop the core default bodies spell out. The
    /// closure is invoked through its operand (not a
    /// pre-evaluated value) so its environment aliases its
    /// place, as through the real `FnMut` shim, and mutations
    /// of its captures persist from one item to the next.
    pub(super) fn exec_iterator_fold_like(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        kind: Builtin,
        generic_args: &[String],
        args: &[Operand],
    ) -> InterpResult<Value> {
        let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
        let (mut state, mut accum, f) = match kind {
            Builtin::IteratorFold => {
                if args.len() != 3 {
                    return Err(InterpError::type_mismatch(
                        "builtin Iterator::fold",
                        "3 argument(s)",
                        format!("{} argument(s)", args.len()),
                    ));
                }
                let state = self.eval_operand(frames, body, &args[0])?;
                let accum = self.eval_operand(frames, body, &args[1])?;
                (state, accum, &args[2])
            }
            Builtin::IteratorForEach => {
                if args.len() != 2 {
                    return Err(InterpError::type_mismatch(
                        "builtin Iterator::for_each",
                        "2 argument(s)",
                        format!("{} argument(s)", args.len()),
                    ));
                }
                let state = self.eval_operand(frames, body, &args[0])?;
                (state, Value::Unit, &args[1])
            }
            _ => unreachable!("fold-like dispatch covers fold and for_each only"),
        };
        loop {
            self.tick_step()?;
            let (next_state, item) = self.iterator_next_state(frames, self_ty, state)?;
            state = next_state;
            let Some(item) = option_payload(item)? else {
                break;
            };
            let call_args = match kind {
                Builtin::IteratorFold => {
                    vec![std::mem::replace(&mut accum, Value::Unit), item]
                }
                _ => vec![item],
            };
            let result = self.call_callable_operand_to_completion(frames, body, f, call_args)?;
            if kind == Builtin::IteratorFold {
                accum = result;
            }
        }
        self.note_consumed_iter_final(state);
        Ok(accum)
    }

    /// Execute `Iterator::try_fold`: drive the modelled `next`
    /// state transition over the iterator and invoke `f` per
    /// item with the accumulator, exactly the loop the core
    /// default body spells out. The receiver arrives in the
    /// shape the call site passes it: the iterator value itself
    /// for a by-value operand, or a reference to it for the
    /// forwarded `&mut self` of another consumer's body - the
    /// reference form steps the iterator in place. `f` returns
    /// a `Try` value: its branch decides the loop, in the shape
    /// of the `Try` implementations the standard consumers use -
    /// `ControlFlow` (continue on variant 0, break on variant
    /// 1, the break carrying its payload into the result),
    /// `Result` (continue on `Ok`, return the `Err` value
    /// unchanged), `Option` (continue on `Some`, return
    /// `None`), and `NeverShortCircuit` (the single-field
    /// wrapper `count` and friends fold through, which never
    /// breaks). When the iterator runs out, the result is the
    /// `Try` type's output form of the final accumulator
    /// (`Continue` / `Ok` / `Some` / the wrapper itself). The
    /// result type is the call's fourth generic argument.
    pub(super) fn exec_iterator_try_fold(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        generic_args: &[String],
        args: &[Operand],
    ) -> InterpResult<Value> {
        if args.len() != 3 {
            return Err(InterpError::type_mismatch(
                "builtin Iterator::try_fold",
                "3 argument(s)",
                format!("{} argument(s)", args.len()),
            ));
        }
        let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
        let r_ty = generic_args.get(3).map(String::as_str).unwrap_or("");
        #[derive(Clone, Copy, PartialEq)]
        enum TryShape {
            ControlFlow,
            Result,
            Option,
            NeverShortCircuit,
        }
        let shape = if r_ty.contains("ControlFlow<") {
            TryShape::ControlFlow
        } else if r_ty.contains("Result<") {
            TryShape::Result
        } else if r_ty.contains("Option<") {
            TryShape::Option
        } else if r_ty.contains("NeverShortCircuit<") {
            TryShape::NeverShortCircuit
        } else {
            return Err(InterpError::type_mismatch(
                "builtin Iterator::try_fold",
                "a ControlFlow / Result / Option / NeverShortCircuit try type",
                r_ty,
            ));
        };
        let recv = self.eval_operand(frames, body, &args[0])?;
        // A reference receiver steps the iterator in place (the
        // `&mut self` a consumer body forwards); any other value
        // is the iterator state itself, stepped functionally and
        // handed back through the consumed-iterator channel.
        enum Drive {
            InPlace(Resolved),
            Owned(Value),
        }
        let mut drive = match self.deref_value(frames, &recv) {
            Ok(target) => Drive::InPlace(target),
            Err(_) => Drive::Owned(recv),
        };
        let mut accum = self.eval_operand(frames, body, &args[1])?;
        let f = &args[2];
        let break_value = loop {
            self.tick_step()?;
            let item = match &mut drive {
                Drive::InPlace(target) => {
                    self.iterator_next_at(frames, self_ty, target)?
                }
                Drive::Owned(state) => {
                    let (next_state, item) = self.iterator_next_state(
                        frames,
                        self_ty,
                        std::mem::replace(state, Value::Unit),
                    )?;
                    *state = next_state;
                    item
                }
            };
            let Some(item) = option_payload(item)? else {
                break None;
            };
            let out = self.call_callable_operand_to_completion(
                frames,
                body,
                f,
                vec![std::mem::replace(&mut accum, Value::Unit), item],
            )?;
            let Value::Enum { variant, fields } = out else {
                return Err(InterpError::type_mismatch(
                    "builtin Iterator::try_fold",
                    "try-typed closure result",
                    out.kind_name(),
                ));
            };
            match (shape, variant) {
                (TryShape::ControlFlow, 0)
                | (TryShape::Result, 0)
                | (TryShape::Option, 1)
                | (TryShape::NeverShortCircuit, 0) => {
                    let mut fields = fields.into_iter();
                    accum = fields.next().ok_or_else(|| {
                        InterpError::type_mismatch(
                            "builtin Iterator::try_fold",
                            "continue variant carrying the accumulator",
                            "empty variant",
                        )
                    })?;
                }
                (TryShape::ControlFlow, 1) | (TryShape::Result, 1) => {
                    break Some(Value::Enum { variant: 1, fields });
                }
                (TryShape::Option, 0) => {
                    break Some(Value::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    });
                }
                _ => {
                    return Err(InterpError::type_mismatch(
                        "builtin Iterator::try_fold",
                        "a two-variant try-typed closure result",
                        format!("variant {variant}"),
                    ));
                }
            }
        };
        if let Drive::Owned(state) = drive {
            self.note_consumed_iter_final(state);
        }
        if let Some(value) = break_value {
            return Ok(value);
        }
        Ok(match shape {
            TryShape::ControlFlow | TryShape::Result => Value::Enum {
                variant: 0,
                fields: vec![accum],
            },
            TryShape::Option => Value::Enum {
                variant: 1,
                fields: vec![accum],
            },
            TryShape::NeverShortCircuit => Value::Enum {
                variant: 0,
                fields: vec![accum],
            },
        })
    }
}

/// Whether the modelled `Iterator::next` state transition
/// (`iterator_next_state`) serves the iterator type the tag
/// names: the shapes whose values the builtin iterator
/// machinery itself produces — ranges, the `[items, index]`
/// states of `slice::Iter` / `array::IntoIter` / `Chars` /
/// `Bytes`, `Empty`, and the mpsc channel iterators.
/// Whether this type tag names a hash-collection iterator
/// (`hash_map` / `hash_set` view types, or hashbrown's raw
/// iterators): their `next` walks a raw table's control bytes,
/// which the modelled state machines do not reproduce — the
/// collected impl body is the faithful execution.
pub fn hash_iterator_tag(self_ty: &str) -> bool {
    // `Iterator` is also implemented for `&mut I`, whose
    // blanket body forwards to the inner iterator: the tag
    // may carry the reference.
    let self_ty = self_ty
        .strip_prefix("&mut ")
        .or_else(|| self_ty.strip_prefix('&'))
        .unwrap_or(self_ty);
    self_ty.starts_with("std::collections::hash_map::")
        || self_ty.starts_with("std::collections::hash_set::")
        || self_ty.starts_with("hashbrown::map::")
        || self_ty.starts_with("hashbrown::set::")
        || self_ty.starts_with("hashbrown::raw::RawIter")
        || self_ty.starts_with("hashbrown::raw::RawIntoIter")
        || self_ty.starts_with("hashbrown::raw::RawDrain")
        || self_ty.starts_with("hashbrown::raw::RawExtractIf")
}

/// Whether this `IntoIterator::into_iter` `Self` tag names a
/// hash collection (by value or by reference) or a reference
/// to a hash-collection iterator (the `&mut I` blanket impl):
/// the real body assembles the raw-table iterator state,
/// which the builtin's modelled constructions do not cover.
pub fn hash_collection_tag(self_ty: &str) -> bool {
    let inner = self_ty
        .strip_prefix("&mut ")
        .or_else(|| self_ty.strip_prefix('&'))
        .unwrap_or(self_ty);
    inner.starts_with("std::collections::HashMap<")
        || inner.starts_with("std::collections::HashSet<")
        || hash_iterator_tag(inner)
        || hash_iterator_tag(self_ty)
}

/// Whether the modelled `ExactSizeIterator::len` serves
/// this type: the slice-state iterators whose remaining
/// count the unified bounds read (see
/// `slice_iter_front_and_remaining`) — the family the
/// modelled `size_hint` reads through `slice_iter_bounds`.
/// Every other type's `len` keeps its collected body (the
/// trait default, which delegates to `size_hint`, or the
/// type's own override).
///
/// A reference-wrapper `Self` (`&mut I`, the one reference
/// form that implements the trait) is deliberately not
/// modelled here: core's blanket body for `&mut I` is
/// collected — the reach-side predicate likewise names
/// only the bare ADT — and forwards to the inner call,
/// which re-enters with the iterator type itself, the same
/// arrangement as the modelled `next` (see
/// `iterator_next_modelled`). Peeling the reference here
/// would preempt that body while the builtin's receiver
/// chase dereferences only once, leaving a reference where
/// the iterator state belongs.
pub fn iterator_len_modelled(self_ty: &str) -> bool {
    self_ty.starts_with("std::slice::Iter<")
        || self_ty.starts_with("std::slice::IterMut<")
        || self_ty.starts_with("std::collections::btree_set::Iter<")
        || self_ty.starts_with("std::collections::btree_map::Iter<")
        || self_ty.starts_with("std::collections::btree_map::Keys<")
        || self_ty.starts_with("std::collections::btree_map::Values<")
}

pub fn iterator_next_modelled(self_ty: &str) -> bool {
    self_ty.starts_with("std::ops::Range<")
        || self_ty.starts_with("std::ops::RangeInclusive<")
        || self_ty.starts_with("std::slice::Iter<")
        || self_ty.starts_with("std::slice::IterMut<")
        || self_ty.starts_with("std::collections::btree_set::Iter<")
        || self_ty.starts_with("std::collections::btree_map::Iter<")
        || self_ty.starts_with("std::collections::btree_map::Keys<")
        || self_ty.starts_with("std::collections::btree_map::Values<")
        || self_ty.starts_with("std::collections::btree_set::IntoIter<")
        || self_ty.starts_with("std::collections::btree_map::IntoIter<")
        || self_ty.starts_with("std::array::IntoIter<")
        || self_ty.starts_with("std::vec::IntoIter<")
        || self_ty.starts_with("alloc::vec::IntoIter<")
        || self_ty_is(self_ty, "std::env::Args")
        || self_ty_is(self_ty, "std::env::ArgsOs")
        || self_ty_is(self_ty, "std::fs::ReadDir")
        || self_ty_is(self_ty, "std::env::SplitPaths")
        || self_ty.starts_with("std::slice::Windows<")
        || self_ty.starts_with("std::iter::Empty<")
        || self_ty.starts_with("std::sync::mpsc::Iter<")
        || self_ty.starts_with("std::sync::mpsc::IntoIter<")
        || self_ty_is(self_ty, "std::str::Chars")
        || self_ty_is(self_ty, "core::str::Chars")
        || self_ty_is(self_ty, "std::str::Bytes")
        || self_ty_is(self_ty, "core::str::Bytes")
}

/// Whether the modelled iterator consumers (`nth`, `count`,
/// `fold`, `for_each`, `collect`) serve this type: exactly
/// the types the modelled `next` state machine serves. The
/// consumers must agree with the state machine's accounting:
/// core's consumer bodies are folds over `next` whose pointer
/// arithmetic the value model does not reproduce for the
/// real-layout state (a zero-sized element's remaining count
/// is carried in the end pointer's own bits), so a consumer
/// that ran its collected body after a modelled `nth` would
/// recount items the state machine already skipped.
pub fn iterator_consumer_modelled(self_ty: &str) -> bool {
    iterator_next_modelled(self_ty)
}

/// Whether the modelled `DoubleEndedIterator::next_back` step
/// transitions this iterator type's state directly: the leaf
/// shapes of `iterator_next_back_state` plus the `IterMut`
/// state the back step serves in `next_back_on_state`. Adapter
/// types are absent on purpose — their collected bodies
/// compose the modelled steps and stay the primary execution
/// (see `consumer_instance_needs_body` in the driver).
pub fn iterator_next_back_modelled(self_ty: &str) -> bool {
    if self_ty.starts_with("std::ops::Range<")
        || self_ty.starts_with("std::ops::RangeInclusive<")
        || self_ty.starts_with("std::slice::Iter<")
        || self_ty.starts_with("std::slice::IterMut<")
        || self_ty.starts_with("std::collections::btree_set::Iter<")
        || self_ty.starts_with("std::collections::btree_map::Iter<")
        || self_ty.starts_with("std::collections::btree_map::Keys<")
        || self_ty.starts_with("std::collections::btree_map::Values<")
        || self_ty.starts_with("std::collections::btree_set::IntoIter<")
        || self_ty.starts_with("std::collections::btree_map::IntoIter<")
        || self_ty.starts_with("std::array::IntoIter<")
        || self_ty.starts_with("std::vec::IntoIter<")
        || self_ty.starts_with("alloc::vec::IntoIter<")
        || self_ty.starts_with("std::option::IntoIter<")
        || self_ty.starts_with("std::option::Iter<")
        || self_ty.starts_with("std::option::IterMut<")
        || self_ty_is(self_ty, "std::env::Args")
        || self_ty_is(self_ty, "std::env::ArgsOs")
        || self_ty.starts_with("std::iter::Empty<")
        || self_ty.starts_with("std::path::Components")
        || self_ty.starts_with("std::iter::Once<")
        || self_ty.starts_with("std::str::Split<")
        || self_ty.starts_with("std::str::RSplit<")
        || self_ty.starts_with("std::str::SplitTerminator<")
        || self_ty.starts_with("std::str::RSplitTerminator<")
        || self_ty.starts_with("std::str::SplitInclusive<")
        || self_ty_is(self_ty, "std::str::Chars")
        || self_ty_is(self_ty, "core::str::Chars")
        || self_ty_is(self_ty, "std::str::Bytes")
        || self_ty_is(self_ty, "core::str::Bytes")
    {
        return true;
    }
    // A `Rev`'s back step is its inner iterator's front step.
    if self_ty.starts_with("std::iter::Rev<") {
        return first_generic_arg(self_ty).is_some_and(iterator_next_modelled);
    }
    // These adapters' back steps compose the inner back step.
    if self_ty.starts_with("std::iter::Enumerate<")
        || self_ty.starts_with("std::iter::Copied<")
        || self_ty.starts_with("std::iter::Cloned<")
        || self_ty.starts_with("std::iter::Map<")
        || self_ty.starts_with("std::iter::Filter<")
        || self_ty.starts_with("std::iter::FilterMap<")
    {
        return first_generic_arg(self_ty).is_some_and(iterator_next_back_modelled);
    }
    // A `Zip`'s back step composes the two sides' back steps,
    // so it is modelled exactly when both sides' are.
    if self_ty.starts_with("std::iter::Zip<") {
        let args = generic_args_of(self_ty);
        return args.len() >= 2
            && iterator_next_back_modelled(&args[0])
            && iterator_next_back_modelled(&args[1]);
    }
    false
}

pub(super) fn self_ty_is(self_ty: &str, name: &str) -> bool {
    self_ty == name
        || self_ty
            .strip_prefix(name)
            .is_some_and(|rest| rest.starts_with('<'))
}

/// The fields of an iterator state value (ADT aggregate or tuple).
/// One step of an owned-sequence iterator state (see
/// [`owned_iter_state`]): returns the new state and the yielded
/// item, or `None` when the sequence is exhausted (the state is
/// then returned unchanged by the caller).
fn step_owned_iter(state: &Value, self_ty: &str) -> InterpResult<(Value, Option<Value>)> {
    let fields = iterator_fields(state, self_ty)?;
    if fields.len() != 2 {
        return Err(InterpError::type_mismatch(
            format!("{self_ty} state"),
            "[items, next index]",
            format!("{} field(s)", fields.len()),
        ));
    }
    let items = match &fields[0] {
        Value::Tuple(items) => items.clone(),
        ref other => {
            return Err(InterpError::type_mismatch(
                format!("{self_ty} items"),
                "tuple of elements",
                other.kind_name(),
            ))
        }
    };
    let idx = match fields[1] {
        Value::Uint(v) => v as usize,
        ref other => {
            return Err(InterpError::type_mismatch(
                format!("{self_ty} index"),
                "unsigned integer",
                other.kind_name(),
            ))
        }
    };
    if idx < items.len() {
        let item = items[idx].clone();
        let new_state = match state {
            Value::Enum { variant, .. } => Value::Enum {
                variant: *variant,
                fields: vec![Value::Tuple(items), Value::Uint((idx + 1) as u128)],
            },
            _ => Value::Tuple(vec![Value::Tuple(items), Value::Uint((idx + 1) as u128)]),
        };
        Ok((new_state, Some(item)))
    } else {
        Ok((state.clone(), None))
    }
}

/// The iteration state of an owned-sequence iterator in this
/// model (`vec::IntoIter`, `array::IntoIter`, the env argument
/// iterators): the element snapshot plus the next index.
pub(super) fn owned_iter_state(items: Vec<Value>) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Tuple(items), Value::Uint(0)],
    }
}

/// The (items, next index) of an owned-sequence iterator state,
/// with the shape checks [`step_owned_iter`] performs.
fn owned_iter_parts(state: &Value, self_ty: &str) -> InterpResult<(Vec<Value>, usize)> {
    let fields = iterator_fields(state, self_ty)?;
    if fields.len() != 2 {
        return Err(InterpError::type_mismatch(
            format!("{self_ty} state"),
            "[items, next index]",
            format!("{} field(s)", fields.len()),
        ));
    }
    let items = match &fields[0] {
        Value::Tuple(items) => items.clone(),
        ref other => {
            return Err(InterpError::type_mismatch(
                format!("{self_ty} items"),
                "tuple of elements",
                other.kind_name(),
            ))
        }
    };
    let idx = match fields[1] {
        Value::Uint(v) => v as usize,
        ref other => {
            return Err(InterpError::type_mismatch(
                format!("{self_ty} index"),
                "unsigned integer",
                other.kind_name(),
            ))
        }
    };
    Ok((items, idx))
}

/// An iterator state value with `state`'s shape (ADT aggregate
/// or tuple) and new fields: the rebuild `iterator_next`
/// performs on a stepped state.
fn iterator_state_like(state: &Value, fields: Vec<Value>) -> Value {
    match state {
        Value::Enum { variant, .. } => Value::Enum {
            variant: *variant,
            fields,
        },
        _ => Value::Tuple(fields),
    }
}

/// One back step of an owned-sequence iterator state: yields
/// the last remaining item and drops it from the snapshot. The
/// front index is untouched, so front and back steps compose
/// exactly as the real double-ended iterators' do.
fn step_owned_iter_back(
    state: &Value,
    self_ty: &str,
) -> InterpResult<(Value, Option<Value>)> {
    let (items, idx) = owned_iter_parts(state, self_ty)?;
    if items.len() > idx {
        let item = items[items.len() - 1].clone();
        let kept = items[..items.len() - 1].to_vec();
        Ok((
            iterator_state_like(state, vec![Value::Tuple(kept), Value::Uint(idx as u128)]),
            Some(item),
        ))
    } else {
        Ok((state.clone(), None))
    }
}

/// One back step (`DoubleEndedIterator::next_back`) of an
/// iterator state, computed on the state value itself: returns
/// the new state and the yielded item. This is what `Rev<I>`'s
/// `next` runs. The shapes served are the ones `iterator_next`
/// serves whose back step is a state transition of the same
/// kind; anything else stays an explicit unsupported error.
fn iterator_next_back_state(
    state: &Value,
    self_ty: &str,
) -> InterpResult<(Value, Option<Value>)> {
    if self_ty.starts_with("std::path::Components") {
        return components_step_state(state, true);
    }
    if self_ty_is(self_ty, "std::env::Args") || self_ty_is(self_ty, "std::env::ArgsOs") {
        // Step the innermost owned-iterator state from the back
        // and rebuild the wrappers, mirroring the front step in
        // `iterator_next`.
        let depth = if self_ty_is(self_ty, "std::env::Args") { 3 } else { 2 };
        let mut inner = state;
        for _ in 0..depth {
            inner = match inner {
                Value::Enum { fields, .. } if fields.len() == 1 => &fields[0],
                other => {
                    return Err(InterpError::type_mismatch(
                        format!("{self_ty} state"),
                        "nested one-field iterator wrappers",
                        other.kind_name(),
                    ))
                }
            };
        }
        let (new_inner, item) = step_owned_iter_back(inner, self_ty)?;
        let mut rebuilt = new_inner;
        for _ in 0..depth {
            rebuilt = Value::Enum {
                variant: 0,
                fields: vec![rebuilt],
            };
        }
        let item = if self_ty_is(self_ty, "std::env::ArgsOs") {
            item.map(|value| match value {
                Value::Str(text) => os_string_value(text.as_bytes()),
                other => other,
            })
        } else {
            item
        };
        return Ok((rebuilt, item));
    }
    let fields = iterator_fields(state, self_ty)?;
    if self_ty.starts_with("std::ops::RangeInclusive<") {
        if fields.len() != 3 {
            return Err(InterpError::type_mismatch(
                "RangeInclusive state",
                "[start, end, exhausted]",
                format!("{} field(s)", fields.len()),
            ));
        }
        let (start, end, exhausted) = (
            fields[0].clone(),
            fields[1].clone(),
            fields[2].clone(),
        );
        if exhausted == Value::Bool(true) {
            return Ok((state.clone(), None));
        }
        return Ok(match compare_values(&start, &end)? {
            Ordering::Less => (
                iterator_state_like(
                    state,
                    vec![
                        start,
                        range_decrement(&end, range_elem_ty(self_ty))?,
                        Value::Bool(false),
                    ],
                ),
                Some(end),
            ),
            Ordering::Equal => (
                iterator_state_like(state, vec![start, end.clone(), Value::Bool(true)]),
                Some(end),
            ),
            Ordering::Greater => (state.clone(), None),
        });
    }
    if self_ty.starts_with("std::ops::Range<") {
        if fields.len() != 2 {
            return Err(InterpError::type_mismatch(
                "Range state",
                "[start, end]",
                format!("{} field(s)", fields.len()),
            ));
        }
        let (start, end) = (fields[0].clone(), fields[1].clone());
        return Ok(match compare_values(&start, &end)? {
            Ordering::Less => {
                let new_end = range_decrement(&end, range_elem_ty(self_ty))?;
                (
                    iterator_state_like(state, vec![start, new_end.clone()]),
                    Some(new_end),
                )
            }
            _ => (state.clone(), None),
        });
    }
    if self_ty.starts_with("std::slice::Iter<")
        || self_ty.starts_with("std::collections::btree_set::Iter<")
        || self_ty.starts_with("std::collections::btree_map::Iter<")
        || self_ty.starts_with("std::collections::btree_map::Keys<")
        || self_ty.starts_with("std::collections::btree_map::Values<")
        || self_ty.starts_with("std::slice::IterMut<")
    {
        return step_slice_iter_back(state, self_ty);
    }
    if self_ty_is(self_ty, "std::str::Chars")
        || self_ty_is(self_ty, "core::str::Chars")
        || self_ty_is(self_ty, "std::str::Bytes")
        || self_ty_is(self_ty, "core::str::Bytes")
    {
        return step_str_iter(state, self_ty, true);
    }
    if self_ty.starts_with("std::array::IntoIter<")
        || self_ty.starts_with("std::vec::IntoIter<")
        || self_ty.starts_with("alloc::vec::IntoIter<")
        || self_ty.starts_with("std::option::IntoIter<")
        || self_ty.starts_with("std::collections::btree_set::IntoIter<")
        || self_ty.starts_with("std::collections::btree_map::IntoIter<")
    {
        return step_owned_iter_back(state, self_ty);
    }
    if self_ty.starts_with("std::iter::Empty<") {
        return Ok((state.clone(), None));
    }
    if self_ty.starts_with("dyn ")
        && fields.len() == 2
        && matches!(fields[0], Value::Tuple(_))
        && matches!(fields[1], Value::Uint(_))
    {
        // A trait-object tag over an owned-iterator state, as
        // in the front step (see `iterator_next_state`).
        return step_owned_iter_back(state, self_ty);
    }
    Err(unsupported(format!(
        "DoubleEndedIterator::next_back for {self_ty} is not runtime-supported"
    )))
}

/// One step of a real-layout `slice::Iter` state whose
/// pointers are the model's real pointers rather than element
/// pointers into a snapshot: the state core's own construction
/// builds when library code forms the iterator itself (the
/// pointers then denote positions in the pointee's container,
/// in whatever projection form the pointer machinery produced
/// -- a field position plus a wrapper entry, a byte offset).
/// The step is the real iterator's own: the current pointer is
/// the item, pointer equality is the end, and advancing is the
/// pointer-offset machinery at the element type's size, which
/// renders the advanced pointer in the same form the state's
/// own construction produced for the end pointer. `None` when
/// the state is not of this shape (a zero-sized element type,
/// whose end field is a count rather than a pointer, included),
/// leaving the snapshot reading and its shape error in place.
fn slice_iter_step_real(
    interp: &Interpreter,
    frames: &mut Vec<Frame>,
    self_ty: &str,
    fields: &[Value],
    back: bool,
) -> Option<InterpResult<(Vec<Value>, Value)>> {
    if fields.len() != 3 {
        return None;
    }
    let ptr_ref = iter_ptr_value(&fields[0])?;
    let end_ref = iter_ptr_value(&fields[1])?;
    let (Value::Ref(ptr_target), Value::Ref(end_target)) = (&ptr_ref, &end_ref)
    else {
        return None;
    };
    if matches!(ptr_target, RefTarget::Null) || matches!(end_target, RefTarget::Null) {
        return None;
    }
    let elem_tag = first_generic_arg(self_ty)?;
    let (size, _) = interp.layout_of(elem_tag)?;
    let rewrap = |field: &Value, new_ref: Value| -> Value {
        match field {
            Value::Enum { variant, fields } if fields.len() == 1 => Value::Enum {
                variant: *variant,
                fields: vec![new_ref],
            },
            _ => new_ref,
        }
    };
    if ptr_target == end_target {
        return Some(Ok((fields.to_vec(), none_value())));
    }
    if size == 0 {
        // A zero-sized element type: core's own representation
        // encodes the remaining count in the pointer pair itself
        // (the end pointer is the start pointer plus the length,
        // in address units — for a ZST the pointer is
        // meaningless, so the count is what it carries). Only an
        // address-only pair carries the count that way in this
        // model: a storage-backed pair's positions are projection
        // indices, which `slice_iter_bounds` reads. Any other
        // pair keeps the shape error the caller reports for it.
        if let RefTarget::Address(end_addr) = end_target {
            // The mixed pair the collected `iter` / `iter_mut`
            // bodies build for a zero-sized element type: the
            // current pointer still projects into the element
            // storage (its trailing `Index` step is the current
            // position), while the end pointer is the terminal
            // element index carried as an address — core's
            // construction stores the length itself in the end
            // pointer for a ZST, the pointer being meaningless.
            // The remaining count is the distance from the
            // current index to that terminal index.
            if let Some(i) = iter_ptr_index(&ptr_ref, None) {
                let end = *end_addr as usize;
                if i >= end {
                    return Some(Ok((fields.to_vec(), none_value())));
                }
                if back {
                    let new_end = Value::Ref(RefTarget::Address((end - 1) as u64));
                    let item = iter_ptr_with_index(&ptr_ref, end - 1)?;
                    return Some(Ok((
                        vec![
                            fields[0].clone(),
                            rewrap(&fields[1], new_end),
                            fields[2].clone(),
                        ],
                        some_value(item),
                    )));
                }
                let advanced = iter_ptr_with_index(&ptr_ref, i + 1)?;
                return Some(Ok((
                    vec![
                        rewrap(&fields[0], advanced),
                        fields[1].clone(),
                        fields[2].clone(),
                    ],
                    some_value(ptr_ref),
                )));
            }
            return None;
        }
        let (RefTarget::Address(cur), RefTarget::Address(end)) = (ptr_target, end_target)
        else {
            return None;
        };
        let (cur, end) = (*cur, *end);
        if back {
            let new_end = Value::Ref(RefTarget::Address(end.wrapping_sub(1)));
            let item = new_end.clone();
            return Some(Ok((
                vec![
                    fields[0].clone(),
                    rewrap(&fields[1], new_end),
                    fields[2].clone(),
                ],
                some_value(item),
            )));
        }
        let advanced = Value::Ref(RefTarget::Address(cur.wrapping_add(1)));
        return Some(Ok((
            vec![
                rewrap(&fields[0], advanced),
                fields[1].clone(),
                fields[2].clone(),
            ],
            some_value(ptr_ref),
        )));
    }
    let size = u64::try_from(size).ok()?;
    if back {
        let new_end = match interp.ptr_offset_value(frames, end_ref, -1, Some(size), false) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let item = new_end.clone();
        Some(Ok((
            vec![
                fields[0].clone(),
                rewrap(&fields[1], new_end),
                fields[2].clone(),
            ],
            some_value(item),
        )))
    } else {
        let advanced = match interp.ptr_offset_value(frames, ptr_ref.clone(), 1, Some(size), false) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        Some(Ok((
            vec![
                rewrap(&fields[0], advanced),
                fields[1].clone(),
                fields[2].clone(),
            ],
            some_value(ptr_ref),
        )))
    }
}

pub(super) fn slice_iter_value(elems: Vec<Value>) -> Value {
    let len = elems.len();
    let snapshot = Value::Tuple(elems);
    let ptr = Value::Ref(RefTarget::Value(
        Box::new(snapshot.clone()),
        vec![RefProj::Index(0)],
    ));
    let end = Value::Ref(RefTarget::Value(
        Box::new(snapshot),
        vec![RefProj::Index(len)],
    ));
    Value::Enum {
        variant: 0,
        fields: vec![
            // `ptr: NonNull<T>` — the one-field wrapper around
            // the raw element pointer.
            Value::Enum {
                variant: 0,
                fields: vec![ptr],
            },
            end,
            // `_marker: PhantomData<&T>` — a zero-sized ADT
            // value.
            Value::Enum {
                variant: 0,
                fields: Vec::new(),
            },
        ],
    }
}

/// The element pointer a slice-iterator `ptr` field holds: the
/// field's declared type is `NonNull<T>`, a one-field wrapper
/// around the raw pointer, so the pointer is the wrapper's only
/// field when the value is wrapped, and the value itself when it
/// is not.
fn iter_ptr_value(field: &Value) -> Option<Value> {
    match field {
        Value::Enum { fields, .. } if fields.len() == 1 => Some(fields[0].clone()),
        Value::Ref(_) => Some(field.clone()),
        _ => None,
    }
}

/// The element type tag of a slice-iterable container type tag:
/// the `T` of `Vec<T>`, `[T; N]` or `[T]`.
pub(super) fn slice_container_element_tag(container_ty: &str) -> Option<&str> {
    let ty = container_ty.trim();
    if let Some(open) = ty.find('<') {
        let close = ty.rfind('>')?;
        return Some(ty[open + 1..close].trim());
    }
    if ty.starts_with('[') && ty.ends_with(']') {
        let inner = &ty[1..ty.len() - 1];
        return match inner.rsplit_once(';') {
            Some((elem, _)) => Some(elem.trim()),
            None => Some(inner.trim()),
        };
    }
    None
}


/// The (current, end) element indices a slice iterator state
/// encodes for a zero-sized element type: core stores the
/// length itself in the `end_or_len` field as a provenance-free
/// address (`without_provenance_mut(len)`), so that field
/// carries the terminal element index, not a remaining count.
/// The current index is the front pointer's own element index
/// when the pointer denotes a storage position (the mixed form
/// `slice_iter_step_real` steps: the front pointer advances by
/// index while the end field stays the terminal index); a front
/// pointer denoting no position stands at index zero and the
/// steps that serve that form keep it there, decrementing the
/// stored terminal index instead. Both forms therefore read
/// the same way — the remaining count is `end - current` —
/// which is what keeps front and back steps in agreement when
/// they alternate. Recognised by the address-only end field
/// together with a front pointer denoting a real place: an
/// address-only front pointer is a dangling empty slice
/// instead, whose end field is its data address, not a length.
fn iter_zst_bounds(fields: &[Value]) -> Option<(u64, u64)> {
    let end_field = fields.get(1)?;
    let Value::Ref(RefTarget::Address(end)) = iter_ptr_value(end_field)? else {
        return None;
    };
    let ptr = iter_ptr_value(fields.first()?)?;
    if matches!(ptr, Value::Ref(RefTarget::Address(_))) {
        return None;
    }
    let cur = iter_ptr_index(fields.first()?, None).unwrap_or(0) as u64;
    Some((cur, end))
}

/// The remaining element count a slice iterator state encodes
/// for a zero-sized element type (see `iter_zst_bounds`): the
/// distance from the front pointer's current index to the
/// terminal index the end field carries.
fn iter_zst_remaining(fields: &[Value]) -> Option<u64> {
    let (cur, end) = iter_zst_bounds(fields)?;
    Some(end.saturating_sub(cur))
}

/// The size of a `slice::Iter`'s element type, as the
/// iterator's type parameter names it, for the types whose
/// size the name alone fixes (the scalar and pointer types).
/// An ADT element's size is not derivable from the name here;
/// a byte displacement over such an element stays unjudged
/// (a displacement-free element pointer is judged by its
/// trailing element step alone).
fn slice_iter_elem_size(self_ty: &str) -> Option<u64> {
    let elem = first_generic_param(self_ty)?;
    let elem = elem.trim();
    Some(match elem {
        "bool" | "i8" | "u8" => 1,
        "i16" | "u16" => 2,
        "i32" | "u32" | "f32" | "char" => 4,
        "i64" | "u64" | "f64" | "isize" | "usize" => 8,
        "i128" | "u128" => 16,
        "()" => 0,
        _ => {
            let pointee = elem
                .strip_prefix("&mut ")
                .or_else(|| elem.strip_prefix('&'))
                .or_else(|| elem.strip_prefix("*const "))
                .or_else(|| elem.strip_prefix("*mut "));
            match pointee {
                Some(rest)
                    if rest.starts_with('[')
                        || rest == "str"
                        || rest.starts_with("dyn ") =>
                {
                    16
                }
                Some(_) => 8,
                None => return None,
            }
        }
    })
}

/// The position of the element step in a pointer projection:
/// the last `Index` or `Field` step, with only whole-value
/// entry steps (a transparent wrapper's field 0, a downcast)
/// after it. A `Field(0)` step counts as an entry step only
/// when an element step precedes it: a lone leading `Field(0)`
/// is the element step itself (element 0 of a field-stored
/// sequence), and a non-zero `Field` step after an `Index`
/// step denotes a field of the element, whose position is the
/// `Index` step's. Any other trailing step means the
/// projection does not end at an element position.
fn elem_step_pos(projection: &[RefProj]) -> Option<usize> {
    let mut pos = projection.len();
    while pos > 0 {
        match projection[pos - 1] {
            RefProj::Index(_) => return Some(pos - 1),
            RefProj::Field(f) => {
                let earlier_element = projection[..pos - 1]
                    .iter()
                    .any(|s| matches!(s, RefProj::Index(_) | RefProj::Field(_)));
                if f == 0 && earlier_element {
                    // A whole-value entry step into an element
                    // the earlier steps denote: keep scanning.
                    pos -= 1;
                } else if f != 0
                    && projection[..pos - 1]
                        .iter()
                        .any(|s| matches!(s, RefProj::Index(_)))
                {
                    // A field of the element the `Index` step
                    // denotes: the element step is that step.
                    break;
                } else {
                    return Some(pos - 1);
                }
            }
            RefProj::Downcast(_) => pos -= 1,
            _ => break,
        }
    }
    // Steps after the last `Index` step denoted a field of the
    // element (or the scan stopped at a non-entry step): the
    // last `Index` step is the element step.
    projection
        .iter()
        .rposition(|s| matches!(s, RefProj::Index(_)))
}


/// Whether a slice-iterator pointer's projection carries a
/// byte-offset step: the pointer then denotes a position in
/// element units past an index step (the shape a real-pointer
/// iterator over a scalar-backed slice takes — its end
/// pointer is the front place one element further on, spelled
/// as a trailing byte offset), which the index reading
/// `iter_ptr_index` silently drops. A pair carrying such a
/// step is not an index pair: the steps that would read it
/// as one must decline and let the distance reading
/// (`slice_iter_front_and_remaining`) or the real-pointer
/// step (`slice_iter_step_real`) judge it instead. (The
/// length readings keep their index reading of such a pair:
/// they only compare the two positions, and for a pair over
/// a constant value — which no distance reading can measure
/// — that comparison is the reading the modelled flow has
/// always given.)
fn iter_ptr_has_byte_offset(value: &Value) -> bool {
    if let Value::Enum { fields, .. } = value {
        if fields.len() == 1 {
            return iter_ptr_has_byte_offset(&fields[0]);
        }
    }
    let projection = match value {
        Value::Ref(RefTarget::Place { projection, .. }) => projection,
        Value::Ref(RefTarget::Static { projection, .. }) => projection,
        Value::Ref(RefTarget::Alloc { projection, .. }) => projection,
        Value::Ref(RefTarget::Value(_, projection)) => projection,
        _ => return false,
    };
    projection
        .iter()
        .any(|step| matches!(step, RefProj::ByteOffset(_)))
}

/// The element index a slice-iterator pointer denotes. Two
/// forms occur. The states the builtins build point into a
/// value snapshot, the projection a lone trailing `Index`
/// step. The states guest code builds (core's `Iter::new`,
/// inlined into its caller as an aggregate) point into the
/// sequence's own storage: the projection ends at an element
/// step -- an `Index` step into the element sequence or a
/// `Field` step onto an array element, possibly followed by
/// whole-value entry steps into that element (the wrapper
/// entry a `MaybeUninit` element cast leaves) -- optionally
/// followed by the byte displacement the caller's pointer
/// arithmetic added, which counts further elements in units
/// of the element size (a zero element size counts bytes
/// directly, the zero-sized element convention; an unknown
/// size leaves a displaced pointer unjudged).
fn iter_ptr_index(value: &Value, elem_size: Option<u64>) -> Option<usize> {
    if let Value::Enum { fields, .. } = value {
        if fields.len() == 1 {
            return iter_ptr_index(&fields[0], elem_size);
        }
    }
    let projection = match value {
        Value::Ref(RefTarget::Place { projection, .. }) => projection,
        Value::Ref(RefTarget::Static { projection, .. }) => projection,
        Value::Ref(RefTarget::Alloc { projection, .. }) => projection,
        Value::Ref(RefTarget::Value(_, projection)) => projection,
        _ => return None,
    };
    let (steps, bytes) = match projection.last() {
        Some(RefProj::ByteOffset(n)) if *n >= 0 => {
            (&projection[..projection.len() - 1], *n as u64)
        }
        _ => (projection.as_slice(), 0),
    };
    let base = match steps.get(elem_step_pos(steps)?) {
        Some(RefProj::Index(i)) => *i,
        Some(RefProj::Field(f)) => *f,
        _ => return None,
    };
    if bytes == 0 {
        return Some(base);
    }
    match elem_size {
        Some(0) => base.checked_add(usize::try_from(bytes).ok()?),
        Some(size) if bytes % size == 0 => {
            base.checked_add(usize::try_from(bytes / size).ok()?)
        }
        _ => None,
    }
}


/// The element position a projection denotes, as the
/// (step index, element index) pair: the last `Index` step, or
/// the trailing `Field` step when the projection carries no
/// `Index` step at all (see `element_position`).
pub(super) fn element_position_step(projection: &[RefProj]) -> Option<(usize, usize)> {
    if let Some((at, i)) = projection
        .iter()
        .enumerate()
        .rev()
        .find_map(|(at, step)| match step {
            RefProj::Index(i) => Some((at, *i)),
            _ => None,
        })
    {
        return Some((at, i));
    }
    match projection.last() {
        Some(RefProj::Field(i)) => Some((projection.len() - 1, *i)),
        _ => None,
    }
}

/// The same element pointer with its element step set to
/// `idx`: the element step (see `elem_step_pos`) takes the new
/// index, whole-value entry steps after it are kept, and a
/// trailing byte displacement (part of the old position, not
/// of the base) is dropped.
fn iter_ptr_with_index(value: &Value, idx: usize) -> Option<Value> {
    if let Value::Enum { variant, fields } = value {
        if fields.len() == 1 {
            let inner = iter_ptr_with_index(&fields[0], idx)?;
            return Some(Value::Enum {
                variant: *variant,
                fields: vec![inner],
            });
        }
    }
    fn with_index(projection: &[RefProj], idx: usize) -> Option<Vec<RefProj>> {
        let mut projection = projection.to_vec();
        if matches!(projection.last(), Some(RefProj::ByteOffset(_))) {
            projection.pop();
        }
        let pos = elem_step_pos(&projection)?;
        match &mut projection[pos] {
            RefProj::Index(i) => *i = idx,
            RefProj::Field(f) => *f = idx,
            _ => return None,
        }
        Some(projection)
    }
    match value {
        Value::Ref(RefTarget::Place {
            frame,
            local,
            projection,
        }) => Some(Value::Ref(RefTarget::place(
            *frame,
            *local,
            with_index(projection, idx)?,
        ))),
        Value::Ref(RefTarget::Static { name, projection }) => {
            Some(Value::Ref(RefTarget::Static {
                name: name.clone(),
                projection: with_index(projection, idx)?,
            }))
        }
        Value::Ref(RefTarget::Alloc { id, projection }) => Some(Value::Ref(RefTarget::Alloc {
            id: *id,
            projection: with_index(projection, idx)?,
        })),
        Value::Ref(RefTarget::Value(v, projection)) => Some(Value::Ref(RefTarget::Value(
            v.clone(),
            with_index(projection, idx)?,
        ))),
        _ => None,
    }
}

/// The (current, end) element indices of a real-layout
/// `slice::Iter` state value, with the shape checks
/// `iterator_next` performs for it.
fn slice_iter_bounds(state: &Value, self_ty: &str) -> InterpResult<(usize, usize)> {
    let fields = iterator_fields(state, self_ty)?;
    if fields.len() != 3 {
        return Err(InterpError::type_mismatch(
            "slice::Iter state",
            "[ptr, end, marker]",
            format!("{} field(s)", fields.len()),
        ));
    }
    if let Some((cur, end)) = iter_zst_bounds(&fields) {
        // The zero-sized form (see `iter_zst_bounds`): the end
        // field carries the terminal index and the front
        // pointer its current index, exactly as the steps read
        // them.
        return Ok((cur as usize, end as usize));
    }
    let elem_size = slice_iter_elem_size(self_ty);
    let (Some(cur), Some(end)) = (
        iter_ptr_index(&fields[0], elem_size),
        iter_ptr_index(&fields[1], elem_size),
    ) else {
        return Err(InterpError::type_mismatch(
            "slice::Iter state",
            "element pointers into the element snapshot",
            format!(
                "{} / {}",
                fields[0].kind_name(),
                fields[1].kind_name()
            ),
        ));
    };
    Ok((cur, end))
}

/// One back step of a real-layout `slice::Iter` state: yields
/// the last remaining element reference and pulls the end
/// pointer back by one element. The front pointer is untouched,
/// so front and back steps compose exactly as the real
/// double-ended iterator's do.
fn step_slice_iter_back(
    state: &Value,
    self_ty: &str,
) -> InterpResult<(Value, Option<Value>)> {
    let fields = iterator_fields(state, self_ty)?;
    if fields.len() != 3 {
        return Err(InterpError::type_mismatch(
            "slice::Iter state",
            "[ptr, end, marker]",
            format!("{} field(s)", fields.len()),
        ));
    }
    if let Some((cur, end)) = iter_zst_bounds(&fields) {
        if cur >= end {
            return Ok((state.clone(), None));
        }
        let new_end = |fields: &[Value]| -> Value {
            let raw = Value::Ref(RefTarget::Address(end - 1));
            match &fields[1] {
                Value::Enum { variant, .. } => Value::Enum {
                    variant: *variant,
                    fields: vec![raw],
                },
                _ => raw,
            }
        };
        if iter_ptr_index(&fields[0], None).is_some() {
            // The mixed zero-sized form (see `iter_zst_bounds`):
            // the end field is the terminal index the front
            // step's index advance is measured against, so the
            // back step yields the element at the index below
            // it and pulls the terminal index down — the front
            // pointer is untouched, exactly as
            // `slice_iter_step_real`'s back step does.
            let item_field = iter_ptr_with_index(&fields[0], (end - 1) as usize)
                .expect("index step checked");
            let item = iter_ptr_value(&item_field).expect("pointer field checked");
            let item = btree_pair_item(self_ty, item);
            return Ok((
                iterator_state_like(
                    state,
                    vec![fields[0].clone(), new_end(&fields), fields[2].clone()],
                ),
                Some(item),
            ));
        }
        // The zero-sized form whose front pointer denotes no
        // position (see `next_on_state`): the back step
        // decrements the stored length and hands out the front
        // pointer, exactly as the front step does.
        let item = iter_ptr_value(&fields[0]).expect("front pointer checked");
        return Ok((
            iterator_state_like(
                state,
                vec![fields[0].clone(), new_end(&fields), fields[2].clone()],
            ),
            Some(item),
        ));
    }
    if iter_ptr_has_byte_offset(&fields[0]) || iter_ptr_has_byte_offset(&fields[1]) {
        // A byte-offset pair (a real-pointer iterator over
        // scalar-backed storage) is not an index pair: decline
        // so the caller steps it through the real-pointer step.
        // (The decline lives here, not in `slice_iter_bounds`:
        // the length readings keep their index reading of such
        // a pair, whose positions they only compare.)
        return Err(InterpError::type_mismatch(
            "slice::Iter state",
            "element pointers into the element snapshot",
            "pointers displaced by a byte offset",
        ));
    }
    let (cur, end) = slice_iter_bounds(state, self_ty)?;
    if cur < end {
        let new_end =
            iter_ptr_with_index(&fields[1], end - 1).expect("index step checked");
        let item_field =
            iter_ptr_with_index(&fields[0], end - 1).expect("index step checked");
        let item = iter_ptr_value(&item_field).expect("pointer field checked");
        let item = btree_pair_item(self_ty, item);
        Ok((
            iterator_state_like(
                state,
                vec![fields[0].clone(), new_end, fields[2].clone()],
            ),
            Some(item),
        ))
    } else {
        Ok((state.clone(), None))
    }
}

/// The `core::str::Bytes` value over a string's bytes, in the
/// struct's real shape: `Bytes(Copied<slice::Iter<u8>>)` — the
/// one field is a `Copied` adapter holding the `slice::Iter<u8>`
/// over the byte snapshot, in the real field layout
/// `slice_iter_value` builds. The shape matters beyond the
/// builtins: core's converted bodies for the iterator's methods
/// project the fields and hand the innermost state to
/// `slice::Iter` methods, so a fabricated (snapshot, cursor)
/// state in its place hands the container value where an
/// iterator state is expected and the run dies at that seam.
pub(super) fn str_bytes_value(bytes: Vec<Value>) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![Value::Enum {
            variant: 0,
            fields: vec![slice_iter_value(bytes)],
        }],
    }
}

/// The `core::str::Chars` value over a string's bytes, in the
/// struct's real shape: `Chars { iter }`, where the field is
/// the `slice::Iter<u8>` over the byte snapshot directly, in
/// the real field layout `slice_iter_value` builds.
pub(super) fn str_chars_value(bytes: Vec<Value>) -> Value {
    Value::Enum {
        variant: 0,
        fields: vec![slice_iter_value(bytes)],
    }
}

fn self_ty_is_str_chars(self_ty: &str) -> bool {
    self_ty_is(self_ty, "std::str::Chars") || self_ty_is(self_ty, "core::str::Chars")
}

fn self_ty_is_str_bytes(self_ty: &str) -> bool {
    self_ty_is(self_ty, "std::str::Bytes") || self_ty_is(self_ty, "core::str::Bytes")
}

/// The byte-slice iterator state a `str::Chars` / `str::Bytes`
/// state wraps: one wrapper layer down for `Chars` (its field
/// is the byte-slice iterator itself), two for `Bytes` (through
/// the `Copied` adapter) — see `str_chars_value` /
/// `str_bytes_value`.
fn str_iter_inner(state: &Value, self_ty: &str) -> InterpResult<Value> {
    let fields = iterator_fields(state, self_ty)?;
    if fields.len() != 1 {
        return Err(InterpError::type_mismatch(
            format!("{self_ty} state"),
            "a one-field wrapper around a byte-slice iterator",
            format!("{} field(s)", fields.len()),
        ));
    }
    let inner = fields.into_iter().next().expect("one field checked");
    if self_ty_is_str_chars(self_ty) {
        return Ok(inner);
    }
    let inner_fields = iterator_fields(&inner, self_ty)?;
    if inner_fields.len() != 1 {
        return Err(InterpError::type_mismatch(
            format!("{self_ty} state"),
            "a one-field wrapper around a byte-slice iterator",
            format!("{} field(s)", inner_fields.len()),
        ));
    }
    Ok(inner_fields.into_iter().next().expect("one field checked"))
}

/// A `str::Chars` / `str::Bytes` state with `state`'s wrapper
/// shapes and `new_inner` as the wrapped byte-slice iterator
/// state: the rebuild the str-iterator steps perform.
fn str_iter_rewrap(state: &Value, self_ty: &str, new_inner: Value) -> InterpResult<Value> {
    let fields = iterator_fields(state, self_ty)?;
    if fields.len() != 1 {
        return Err(InterpError::type_mismatch(
            format!("{self_ty} state"),
            "a one-field wrapper around a byte-slice iterator",
            format!("{} field(s)", fields.len()),
        ));
    }
    if self_ty_is_str_chars(self_ty) {
        return Ok(iterator_state_like(state, vec![new_inner]));
    }
    let rebuilt_inner = iterator_state_like(&fields[0], vec![new_inner]);
    Ok(iterator_state_like(state, vec![rebuilt_inner]))
}

/// The byte elements of the snapshot a real-layout slice
/// iterator state points into (see `slice_iter_value`): the
/// producers' pointers denote a value snapshot whose elements
/// are the bytes themselves.
fn slice_iter_snapshot_bytes(state: &Value, self_ty: &str) -> InterpResult<Vec<u8>> {
    let fields = iterator_fields(state, self_ty)?;
    if fields.len() != 3 {
        return Err(InterpError::type_mismatch(
            "slice::Iter state",
            "[ptr, end, marker]",
            format!("{} field(s)", fields.len()),
        ));
    }
    let ptr = iter_ptr_value(&fields[0]).ok_or_else(|| {
        InterpError::type_mismatch(
            format!("{self_ty} state"),
            "an element pointer into a value snapshot",
            fields[0].kind_name(),
        )
    })?;
    let items = match &ptr {
        Value::Ref(RefTarget::Value(snapshot, _)) => match &**snapshot {
            Value::Tuple(items) => items.clone(),
            other => {
                return Err(InterpError::type_mismatch(
                    format!("{self_ty} state"),
                    "a value snapshot of bytes",
                    other.kind_name(),
                ))
            }
        },
        other => {
            return Err(InterpError::type_mismatch(
                format!("{self_ty} state"),
                "an element pointer into a value snapshot",
                other.kind_name(),
            ))
        }
    };
    let mut bytes = Vec::with_capacity(items.len());
    for item in &items {
        match item {
            Value::Uint(v) => bytes.push(u8::try_from(*v).map_err(|_| {
                InterpError::type_mismatch(
                    format!("{self_ty} state"),
                    "byte values",
                    format!("integer {v}"),
                )
            })?),
            other => {
                return Err(InterpError::type_mismatch(
                    format!("{self_ty} state"),
                    "byte values",
                    other.kind_name(),
                ))
            }
        }
    }
    Ok(bytes)
}

/// One step of a `str::Chars` / `str::Bytes` state, from the
/// front (`from_back == false`) or the back: the step runs on
/// the wrapped byte-slice iterator — a `Bytes` step moves one
/// byte, a `Chars` step moves one UTF-8 encoded char, decoded
/// with the host's own UTF-8 rules from the byte snapshot the
/// state's pointers denote. The string the iterator was built
/// from is valid UTF-8 and a `Chars` state's pointers only ever
/// rest on char boundaries, exactly as the real iterators
/// maintain, so the decode cannot fail on a state the producers
/// built.
fn step_str_iter(
    state: &Value,
    self_ty: &str,
    from_back: bool,
) -> InterpResult<(Value, Option<Value>)> {
    let inner = str_iter_inner(state, self_ty)?;
    let (cur, end) = slice_iter_bounds(&inner, self_ty)?;
    if cur >= end {
        return Ok((state.clone(), None));
    }
    let bytes = slice_iter_snapshot_bytes(&inner, self_ty)?;
    // Only a `Chars` step decodes: its pointers rest on char
    // boundaries, so the remaining range is valid UTF-8. A
    // `Bytes` step moves a single byte and never decodes — its
    // pointers routinely rest inside a char.
    let text = if self_ty_is_str_chars(self_ty) {
        Some(std::str::from_utf8(&bytes[cur..end]).map_err(|_| {
            InterpError::type_mismatch(
                format!("{self_ty} state"),
                "valid UTF-8 between the iterator's pointers",
                "invalid UTF-8",
            )
        })?)
    } else {
        None
    };
    let fields = iterator_fields(&inner, self_ty)?;
    let (item, new_inner) = if from_back {
        let (item, width) = if self_ty_is_str_chars(self_ty) {
            let c = text
                .expect("chars step decoded")
                .chars()
                .next_back()
                .expect("range checked non-empty");
            (Value::Uint(c as u32 as u128), c.len_utf8())
        } else {
            (Value::Uint(bytes[end - 1] as u128), 1)
        };
        let new_end =
            iter_ptr_with_index(&fields[1], end - width).expect("index step checked");
        (
            item,
            iterator_state_like(
                &inner,
                vec![fields[0].clone(), new_end, fields[2].clone()],
            ),
        )
    } else {
        let (item, width) = if self_ty_is_str_chars(self_ty) {
            let c = text
                .expect("chars step decoded")
                .chars()
                .next()
                .expect("range checked non-empty");
            (Value::Uint(c as u32 as u128), c.len_utf8())
        } else {
            (Value::Uint(bytes[cur] as u128), 1)
        };
        let new_ptr =
            iter_ptr_with_index(&fields[0], cur + width).expect("index step checked");
        (
            item,
            iterator_state_like(
                &inner,
                vec![new_ptr, fields[1].clone(), fields[2].clone()],
            ),
        )
    };
    Ok((str_iter_rewrap(state, self_ty, new_inner)?, Some(item)))
}

pub(super) fn iterator_fields(state: &Value, self_ty: &str) -> InterpResult<Vec<Value>> {
    match state {
        Value::Enum { fields, .. } => Ok(fields.clone()),
        Value::Tuple(items) => Ok(items.clone()),
        other => Err(InterpError::type_mismatch(
            format!("iterator state of {self_ty}"),
            "ADT value",
            other.kind_name(),
        )),
    }
}

/// Whether a rendered tag names one of the `str` split
/// iterators whose state machine `str_split_step` replays
/// (`Split`, `RSplit`, `SplitTerminator`, `RSplitTerminator`,
/// `SplitInclusive` -- the `<` keeps `Split` from also matching
/// `SplitInclusive` / `SplitTerminator`).
fn is_str_split_ty(self_ty: &str) -> bool {
    self_ty.starts_with("std::str::Split<")
        || self_ty.starts_with("std::str::RSplit<")
        || self_ty.starts_with("std::str::SplitTerminator<")
        || self_ty.starts_with("std::str::RSplitTerminator<")
        || self_ty.starts_with("std::str::SplitInclusive<")
}

/// The decoded state of a `str` split iterator: core's
/// `SplitInternal` fields plus the `CharSearcher` fields the
/// step machine reads. The searcher's needle is kept as its
/// UTF-8 encoding; the fields the machine does not read (the
/// needle char, its size, the encoding array) are carried as
/// their original values so the rebuilt state is field-exact.
struct StrSplitParts {
    start: usize,
    end: usize,
    haystack: String,
    finger: usize,
    finger_back: usize,
    needle: Vec<u8>,
    needle_value: Value,
    utf8_size_value: Value,
    utf8_encoded_value: Value,
    allow_trailing_empty: bool,
    finished: bool,
}

impl StrSplitParts {
    /// Decode the state value, or `None` when it is not the
    /// char-searcher split state (another pattern's searcher,
    /// whose step stays unsupported).
    fn decode(state: &Value) -> Option<Self> {
        let outer = iterator_fields(state, "").ok()?;
        if outer.len() != 1 {
            return None;
        }
        let internal = iterator_fields(&outer[0], "").ok()?;
        if internal.len() != 5 {
            return None;
        }
        let matcher = iterator_fields(&internal[2], "").ok()?;
        if matcher.len() != 6 {
            return None;
        }
        let Value::Str(haystack) = &matcher[0] else {
            return None;
        };
        let as_usize = |v: &Value| -> Option<usize> {
            match v {
                Value::Uint(x) => usize::try_from(*x).ok(),
                _ => None,
            }
        };
        let utf8_size = as_usize(&matcher[4])?;
        if utf8_size == 0 || utf8_size > 4 {
            return None;
        }
        let encoded = iterator_fields(&matcher[5], "").ok()?;
        if encoded.len() != 4 {
            return None;
        }
        let mut needle = Vec::with_capacity(utf8_size);
        for slot in &encoded[..utf8_size] {
            match slot {
                Value::Uint(b) if *b <= 255 => needle.push(*b as u8),
                _ => return None,
            }
        }
        // The recorded encoding must be the needle char's own:
        // the search below is a substring search for it.
        if std::str::from_utf8(&needle).is_err() {
            return None;
        }
        Some(StrSplitParts {
            start: as_usize(&internal[0])?,
            end: as_usize(&internal[1])?,
            haystack: haystack.clone(),
            finger: as_usize(&matcher[1])?,
            finger_back: as_usize(&matcher[2])?,
            needle,
            needle_value: matcher[3].clone(),
            utf8_size_value: matcher[4].clone(),
            utf8_encoded_value: matcher[5].clone(),
            allow_trailing_empty: internal[3] == Value::Bool(true),
            finished: internal[4] == Value::Bool(true),
        })
    }

    /// Rebuild the state value with the stepped fields, in the
    /// exact shape `decode` read.
    fn encode(&self, state: &Value) -> Value {
        let outer = iterator_fields(state, "").expect("state decoded");
        let internal = iterator_fields(&outer[0], "").expect("state decoded");
        let new_matcher = rebuild_state(
            &internal[2],
            vec![
                Value::Str(self.haystack.clone()),
                Value::Uint(self.finger as u128),
                Value::Uint(self.finger_back as u128),
                self.needle_value.clone(),
                self.utf8_size_value.clone(),
                self.utf8_encoded_value.clone(),
            ],
        );
        let new_internal = rebuild_state(
            &outer[0],
            vec![
                Value::Uint(self.start as u128),
                Value::Uint(self.end as u128),
                new_matcher,
                Value::Bool(self.allow_trailing_empty),
                Value::Bool(self.finished),
            ],
        );
        rebuild_state(state, vec![new_internal])
    }

    /// `CharSearcher::next_match`: the first needle occurrence
    /// in `haystack[finger..finger_back]`; the finger advances
    /// past it. A substring search for the needle's encoding
    /// is the same search (see `str_split_step`).
    fn next_match(&mut self) -> Option<(usize, usize)> {
        let bytes = self.haystack.as_bytes();
        let range = bytes.get(self.finger..self.finger_back)?;
        let at = range
            .windows(self.needle.len())
            .position(|w| w == self.needle.as_slice());
        match at {
            Some(at) => {
                let found = self.finger + at;
                self.finger = found + self.needle.len();
                Some((found, self.finger))
            }
            None => {
                self.finger = self.finger_back;
                None
            }
        }
    }

    /// `CharSearcher::next_match_back`: the last needle
    /// occurrence in `haystack[finger..finger_back]`; the back
    /// finger retreats to its start.
    fn next_match_back(&mut self) -> Option<(usize, usize)> {
        let bytes = self.haystack.as_bytes();
        let Some(range) = bytes.get(self.finger..self.finger_back) else {
            self.finger_back = self.finger;
            return None;
        };
        let at = range
            .windows(self.needle.len())
            .rposition(|w| w == self.needle.as_slice());
        match at {
            Some(at) => {
                let found = self.finger + at;
                self.finger_back = found;
                Some((found, found + self.needle.len()))
            }
            None => {
                self.finger_back = self.finger;
                None
            }
        }
    }

    /// `SplitInternal::get_end`.
    fn get_end(&mut self) -> Option<String> {
        if !self.finished {
            self.finished = true;
            if self.allow_trailing_empty || self.end > self.start {
                return Some(self.haystack[self.start..self.end].to_string());
            }
        }
        None
    }

    /// `SplitInternal::next`.
    fn next(&mut self) -> Option<String> {
        if self.finished {
            return None;
        }
        match self.next_match() {
            Some((a, _)) => {
                let elt = self.haystack[self.start..a].to_string();
                self.start = a + self.needle.len();
                Some(elt)
            }
            None => self.get_end(),
        }
    }

    /// `SplitInternal::next_inclusive`.
    fn next_inclusive(&mut self) -> Option<String> {
        if self.finished {
            return None;
        }
        match self.next_match() {
            Some((_, b)) => {
                let elt = self.haystack[self.start..b].to_string();
                self.start = b;
                Some(elt)
            }
            None => self.get_end(),
        }
    }

    /// The post-dance half of `SplitInternal::next_back` /
    /// `next_back_inclusive`: one back match, or the final
    /// piece (which the back steps yield unconditionally).
    fn back_match(&mut self, inclusive: bool) -> Option<String> {
        match self.next_match_back() {
            Some((a, b)) => {
                let elt = self.haystack[b..self.end].to_string();
                self.end = if inclusive { b } else { a };
                Some(elt)
            }
            None => {
                self.finished = true;
                Some(self.haystack[self.start..self.end].to_string())
            }
        }
    }

    /// `SplitInternal::next_back`, including its
    /// `allow_trailing_empty` dance (the recursive call in
    /// core is the one `back_match` call here).
    fn next_back(&mut self) -> Option<String> {
        if self.finished {
            return None;
        }
        if !self.allow_trailing_empty {
            self.allow_trailing_empty = true;
            match self.back_match(false) {
                Some(elt) if !elt.is_empty() => return Some(elt),
                _ => {
                    if self.finished {
                        return None;
                    }
                }
            }
        }
        self.back_match(false)
    }

    /// `SplitInternal::next_back_inclusive`: the mirror of
    /// `next_back` over `back_match`'s inclusive form.
    fn next_back_inclusive(&mut self) -> Option<String> {
        if self.finished {
            return None;
        }
        if !self.allow_trailing_empty {
            self.allow_trailing_empty = true;
            match self.back_match(true) {
                Some(elt) if !elt.is_empty() => return Some(elt),
                _ => {
                    if self.finished {
                        return None;
                    }
                }
            }
        }
        self.back_match(true)
    }
}

fn rebuild_state(state: &Value, new_fields: Vec<Value>) -> Value {
    match state {
        Value::Enum { variant, .. } => Value::Enum {
            variant: *variant,
            fields: new_fields,
        },
        _ => Value::Tuple(new_fields),
    }
}

fn option_payload_opt(value: Value) -> Option<Value> {
    match value {
        Value::Enum {
            variant: 1,
            mut fields,
        } if fields.len() == 1 => Some(fields.remove(0)),
        _ => None,
    }
}

fn extend_target(target: &Resolved, step: RefProj) -> Option<Resolved> {
    match target {
        Resolved::Place {
            frame_pos,
            local,
            projection,
        } => {
            let mut projection = projection.clone();
            projection.push(step);
            Some(Resolved::Place {
                frame_pos: *frame_pos,
                local: *local,
                projection,
            })
        }
        Resolved::Static { name, projection } => {
            let mut projection = projection.clone();
            projection.push(step);
            Some(Resolved::Static {
                name: name.clone(),
                projection,
            })
        }
        Resolved::Alloc { id, projection } => {
            let mut projection = projection.clone();
            projection.push(step);
            Some(Resolved::Alloc {
                id: *id,
                projection,
            })
        }
        Resolved::Value(..) => None,
    }
}

fn take_option_payload(state: &Value) -> Result<Option<(Value, Value)>, ()> {
    match state {
        Value::Enum {
            variant: 1,
            fields,
        } if fields.len() == 1 => Ok(Some((
            Value::Enum {
                variant: 0,
                fields: Vec::new(),
            },
            fields[0].clone(),
        ))),
        Value::Enum { variant: 0, fields } if fields.is_empty() => Ok(None),
        Value::Enum { variant: 0, fields } if fields.len() == 1 => {
            match take_option_payload(&fields[0])? {
                Some((new_inner, item)) => Ok(Some((
                    Value::Enum {
                        variant: 0,
                        fields: vec![new_inner],
                    },
                    item,
                ))),
                None => Ok(None),
            }
        }
        Value::Tuple(items) if items.len() == 1 => match take_option_payload(&items[0])? {
            Some((new_inner, item)) => {
                Ok(Some((Value::Tuple(vec![new_inner]), item)))
            }
            None => Ok(None),
        },
        _ => Err(()),
    }
}

/// The panic message `core::slice::index::slice_index_fail`
/// selects for `(start, end, len)`, in core's own branch order
/// (the runtime form of its `const_panic!` messages).
pub(super) fn slice_index_fail_message(start: usize, end: usize, len: usize) -> String {
    if start > len {
        format!("range start index {start} out of range for slice of length {len}")
    } else if end > len {
        format!("range end index {end} out of range for slice of length {len}")
    } else if start > end {
        format!("slice index starts at {start} but ends at {end}")
    } else {
        format!("range end index {end} out of range for slice of length {len}")
    }
}
