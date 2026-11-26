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

    /// The `Map` type a `str::Lines` value wraps: core defines
    /// `Lines` as a one-field wrapper over exactly this adapter
    /// (its `Iterator` impl forwards both steps to the field),
    /// so stepping a `Lines` state - in place or by value - is
    /// stepping the field under this tag. The callable the tag
    /// names is one of core's zero-sized `Fn` types, applied by
    /// the type-directed dispatch in the `Map` step itself.
    fn lines_map_ty(self_ty: &str) -> Option<&'static str> {
        if self_ty_is(self_ty, "std::str::Lines") {
            Some("std::iter::Map<std::str::SplitInclusive<char>, core::str::LinesMap>")
        } else {
            None
        }
    }

    /// The `Filter` type a `str::SplitWhitespace` value
    /// wraps: core defines `SplitWhitespace` as a one-field
    /// wrapper over exactly this adapter (its `Iterator`
    /// impl forwards both steps to the field), so stepping
    /// a `SplitWhitespace` state - in place or by value -
    /// is stepping the field under this tag. Both callables
    /// the tag names are zero-sized `Fn` types of core's
    /// `impl_fn_for_zst!` family, applied by the
    /// type-directed dispatch in the `Filter` and split
    /// steps themselves.
    fn split_whitespace_filter_ty(self_ty: &str) -> Option<&'static str> {
        if self_ty_is(self_ty, "std::str::SplitWhitespace") {
            Some("std::iter::Filter<std::str::Split<core::str::IsWhitespace>, core::str::IsNotEmpty>")
        } else {
            None
        }
    }

    /// The `FlatMap` type a `str` escape iterator value
    /// wraps: core defines `str::EscapeDebug` /
    /// `EscapeDefault` / `EscapeUnicode` as one-field
    /// wrappers over exactly this composition (a `FlatMap`
    /// of the string's `Chars` through the matching
    /// zero-sized `char` escape callable), and their
    /// `Iterator` impls forward the step to the field, so
    /// stepping a `str` escape state - in place or by
    /// value - is stepping the field under this tag. The
    /// callable the tag names is one of core's zero-sized
    /// `Fn` types, applied by the type-directed dispatch
    /// in the `Map` step the flatten machine composes.
    fn str_escape_flatmap_ty(self_ty: &str) -> Option<&'static str> {
        if self_ty_is(self_ty, "std::str::EscapeDebug")
            || self_ty_is(self_ty, "core::str::EscapeDebug")
        {
            Some("std::iter::FlatMap<std::str::Chars, std::char::EscapeDebug, core::str::CharEscapeDebug>")
        } else if self_ty_is(self_ty, "std::str::EscapeDefault")
            || self_ty_is(self_ty, "core::str::EscapeDefault")
        {
            Some("std::iter::FlatMap<std::str::Chars, std::char::EscapeDefault, core::str::CharEscapeDefault>")
        } else if self_ty_is(self_ty, "std::str::EscapeUnicode")
            || self_ty_is(self_ty, "core::str::EscapeUnicode")
        {
            Some("std::iter::FlatMap<std::str::Chars, std::char::EscapeUnicode, core::str::CharEscapeUnicode>")
        } else {
            None
        }
    }

    /// `Iterator::next` on the iterator stored at `target`:
    /// mutate it in place and return the `Option` value.
    pub(super) fn iterator_next_at(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &Resolved,
    ) -> InterpResult<Value> {
        if let Some(map_ty) = Self::lines_map_ty(self_ty) {
            if let Some(field) = extend_target(target, RefProj::Field(0)) {
                return self.iterator_next_at(frames, map_ty, &field);
            }
        }
        if let Some(filter_ty) = Self::split_whitespace_filter_ty(self_ty) {
            if let Some(field) = extend_target(target, RefProj::Field(0)) {
                return self.iterator_next_at(frames, filter_ty, &field);
            }
        }
        if let Some(flatmap_ty) = Self::str_escape_flatmap_ty(self_ty) {
            if let Some(field) = extend_target(target, RefProj::Field(0)) {
                return self.iterator_next_at(frames, flatmap_ty, &field);
            }
        }
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
        if self_ty.starts_with("std::iter::FlatMap<")
            || self_ty.starts_with("std::iter::Flatten<")
            || self_ty.starts_with("std::iter::adapters::flatten::FlattenCompat<")
        {
            if let Some(result) = self.flatten_next_at(frames, self_ty, target, false)? {
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
        if let Some(map_ty) = Self::lines_map_ty(self_ty) {
            if let Some(field) = extend_target(target, RefProj::Field(0)) {
                return self.iterator_next_back_at(frames, map_ty, &field);
            }
        }
        if let Some(filter_ty) = Self::split_whitespace_filter_ty(self_ty) {
            if let Some(field) = extend_target(target, RefProj::Field(0)) {
                return self.iterator_next_back_at(frames, filter_ty, &field);
            }
        }
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
        if self_ty.starts_with("std::iter::FlatMap<")
            || self_ty.starts_with("std::iter::Flatten<")
            || self_ty.starts_with("std::iter::adapters::flatten::FlattenCompat<")
        {
            if let Some(result) = self.flatten_next_at(frames, self_ty, target, true)? {
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
            let keep = self.call_callable_sync_typed(
                frames,
                &callable,
                second_generic_arg(self_ty).unwrap_or(""),
                Some(env_ref),
                vec![arg],
            )?;
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
            let keep = self.call_callable_sync_typed(
                frames,
                &callable,
                second_generic_arg(self_ty).unwrap_or(""),
                Some(env_ref),
                vec![arg],
            )?;
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
            let mapped = self.call_callable_sync_typed(
                frames,
                &callable,
                second_generic_arg(self_ty).unwrap_or(""),
                Some(env_ref),
                vec![item],
            )?;
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
            let mapped = self.call_callable_sync_typed(
                frames,
                &callable,
                second_generic_arg(self_ty).unwrap_or(""),
                Some(env_ref),
                vec![item],
            )?;
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
    /// Two searcher shapes are modelled: the `char` pattern's
    /// searcher -- the search is a substring search for the
    /// needle's UTF-8 encoding, which is exact because a char's
    /// encoding never occurs inside another char's -- and the
    /// `CharEqSearcher` family (`MultiCharEqSearcher`): a
    /// `char` predicate, whose scan applies the predicate to
    /// each char of the remaining window, or a set of chars (a
    /// `[char; N]` pattern's chars by value, or a `&[char]` /
    /// `&[char; N]` pattern's reference, resolved before the
    /// first step), whose scan tests membership. Any other
    /// searcher shape is the generic unsupported error, as
    /// before.
    /// One front or back step of a `str` matches iterator
    /// (`Matches` / `MatchIndices`), replaying core's searcher
    /// step on the state the constructor's converted body
    /// built: the two types are one-field wrappers over an
    /// internal struct whose only field is the pattern's
    /// searcher, and a step is exactly the searcher's
    /// `next_match` / `next_match_back` -- `Matches` yields the
    /// matched text, `MatchIndices` the pair of its byte index
    /// and the text. The searcher shapes modelled are the
    /// `char` pattern's (the search is a substring search for
    /// the needle's UTF-8 encoding, as in `str_split_step`)
    /// and the `CharEqSearcher` family -- a `char` predicate
    /// (closure or function pointer) or a set of chars (a
    /// `[char; N]` pattern's chars by value, or a `&[char]`
    /// pattern's slice) -- whose scan tests each char of the
    /// remaining window, and the `&str` pattern's `StrSearcher`
    /// (see `StrSearcherParts`). Any other searcher shape is
    /// the generic unsupported error, as before.
    fn str_matches_step(
        &mut self,
        frames: &mut Vec<Frame>,
        state: &Value,
        self_ty: &str,
        back: bool,
    ) -> InterpResult<(Value, Value)> {
        let indices = self_ty.starts_with("std::str::MatchIndices<");
        if let Some(mut parts) = StrMatchesParts::decode(state) {
            let found = if back {
                parts.next_match_back()
            } else {
                parts.next_match()
            };
            let new_state = parts.encode(state);
            let item = matches_item(&parts.haystack, indices, found);
            return Ok((new_state, item));
        }
        if let Some(mut parts) = StrMatchesEqParts::decode(state) {
            if !parts.resolve_set(self, frames) {
                return Err(unsupported(format!(
                    "Iterator::next for {self_ty} is not runtime-supported"
                )));
            }
            let found = if back {
                parts.next_match_back(self, frames)?
            } else {
                parts.next_match(self, frames)?
            };
            let new_state = parts.encode(state);
            let item = matches_item(&parts.haystack, indices, found);
            return Ok((new_state, item));
        }
        if let Some(mut parts) = StrMatchesStrParts::decode(state) {
            let found = if back {
                parts.next_match_back()
            } else {
                parts.next_match()
            };
            let new_state = parts.encode(state);
            let item = matches_item(&parts.searcher.haystack, indices, found);
            return Ok((new_state, item));
        }
        Err(unsupported(format!(
            "Iterator::next for {self_ty} is not runtime-supported"
        )))
    }

    fn str_split_step(
        &mut self,
        frames: &mut Vec<Frame>,
        state: &Value,
        self_ty: &str,
        back: bool,
    ) -> InterpResult<(Value, Value)> {
        let reversed = self_ty.starts_with("std::str::RSplit<")
            || self_ty.starts_with("std::str::RSplitTerminator<");
        let inclusive = self_ty.starts_with("std::str::SplitInclusive<");
        if let Some(mut parts) = StrSplitParts::decode(state) {
            let item = match (back == reversed, inclusive) {
                (true, false) => parts.next(),
                (true, true) => parts.next_inclusive(),
                (false, false) => parts.next_back(),
                (false, true) => parts.next_back_inclusive(),
            };
            let new_state = parts.encode(state);
            return Ok(match item {
                Some(piece) => (new_state, some_value(Value::Str(piece))),
                None => (new_state, none_value()),
            });
        }
        if let Some(mut parts) = StrSplitPredParts::decode(state, self_ty) {
            if !parts.resolve_set(self, frames) {
                return Err(unsupported(format!(
                    "Iterator::next for {self_ty} is not runtime-supported"
                )));
            }
            let item = match (back == reversed, inclusive) {
                (true, false) => parts.next(self, frames)?,
                (true, true) => parts.next_inclusive(self, frames)?,
                (false, false) => parts.next_back(self, frames)?,
                (false, true) => parts.next_back_inclusive(self, frames)?,
            };
            let new_state = parts.encode(state);
            return Ok(match item {
                Some(piece) => (new_state, some_value(Value::Str(piece))),
                None => (new_state, none_value()),
            });
        }
        if let Some(mut parts) = StrSplitStrParts::decode(state) {
            let item = match (back == reversed, inclusive) {
                (true, false) => parts.next(),
                (true, true) => parts.next_inclusive(),
                (false, false) => parts.next_back(),
                (false, true) => parts.next_back_inclusive(),
            };
            let new_state = parts.encode(state);
            return Ok(match item {
                Some(piece) => (new_state, some_value(Value::Str(piece))),
                None => (new_state, none_value()),
            });
        }
        Err(unsupported(format!(
            "Iterator::next for {self_ty} is not runtime-supported"
        )))
    }

    /// One front step of a `str` splitn iterator (`SplitN` /
    /// `RSplitN`), replaying core's `SplitNInternal` step on
    /// the state the constructor's converted body built: the
    /// type is a one-field wrapper over `SplitNInternal`,
    /// whose fields are the wrapped split state -- the very
    /// `SplitInternal` state `str_split_step`'s machines
    /// decode -- and the count of splits remaining. The step
    /// is core's own count logic, shared by both types: a
    /// count of zero is exhausted, a count of one yields the
    /// wrapped iterator's `get_end` (the whole remainder),
    /// and any larger count is spent on one wrapped step --
    /// the front step for `SplitN`, the back step for
    /// `RSplitN`, whose `Iterator::next` is the internal
    /// `next_back`. Neither type implements
    /// `DoubleEndedIterator`, so no back step is modelled.
    fn str_splitn_step(
        &mut self,
        frames: &mut Vec<Frame>,
        state: &Value,
        self_ty: &str,
    ) -> InterpResult<(Value, Value)> {
        let reversed = self_ty.starts_with("std::str::RSplitN<");
        if let Some(mut parts) = StrSplitNParts::decode(state, self_ty) {
            let item = parts.next(self, frames, reversed)?;
            let new_state = parts.encode(state);
            return Ok(match item {
                Some(piece) => (new_state, some_value(Value::Str(piece))),
                None => (new_state, none_value()),
            });
        }
        Err(unsupported(format!(
            "Iterator::next for {self_ty} is not runtime-supported"
        )))
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
        let mapped = self.call_callable_sync_typed(
            frames,
            &callable,
            second_generic_arg(self_ty).unwrap_or(""),
            Some(env_ref),
            vec![item],
        )?;
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
        // The reference-wrapped forwarding of `next_on_state`,
        // for the back step: core's blanket
        // `DoubleEndedIterator` impl for `&mut I` (and `&I`)
        // forwards to the pointee; the reference itself is
        // the state and is returned unchanged.
        if let Some(inner_ty) = self_ty
            .strip_prefix("&mut ")
            .or_else(|| self_ty.strip_prefix('&'))
        {
            match &state {
                Value::Ref(RefTarget::Value(pointee, projection))
                    if projection.is_empty() =>
                {
                    let (new_inner, item) = self.next_back_on_state(
                        frames,
                        inner_ty,
                        (**pointee).clone(),
                    )?;
                    return Ok((
                        Value::Ref(RefTarget::Value(
                            Box::new(new_inner),
                            projection.clone(),
                        )),
                        item,
                    ));
                }
                Value::Ref(_) => {
                    let target = self.deref_value(frames, &state)?;
                    let item =
                        self.iterator_next_back_at(frames, inner_ty, &target)?;
                    return Ok((state, item));
                }
                _ => {}
            }
        }
        if let Some(map_ty) = Self::lines_map_ty(self_ty) {
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Lines state",
                    "[map]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let (new_map, result) =
                self.next_back_on_state(frames, map_ty, fields[0].clone())?;
            return Ok((rebuild_state(&state, vec![new_map]), result));
        }
        if self_ty.starts_with("std::iter::FlatMap<")
            || self_ty.starts_with("std::iter::Flatten<")
            || self_ty.starts_with("std::iter::adapters::flatten::FlattenCompat<")
        {
            if let Some(step) = self.flatten_step_state(frames, self_ty, &state, true)? {
                return Ok(step);
            }
        }
        if let Some(filter_ty) = Self::split_whitespace_filter_ty(self_ty) {
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "SplitWhitespace state",
                    "[filter]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let (new_filter, result) =
                self.next_back_on_state(frames, filter_ty, fields[0].clone())?;
            return Ok((rebuild_state(&state, vec![new_filter]), result));
        }
        if self_ty.starts_with("std::slice::Chunks<")
            || self_ty.starts_with("std::slice::RChunks<")
            || self_ty.starts_with("std::slice::ChunksExact<")
            || self_ty.starts_with("std::slice::RChunksExact<")
            || self_ty.starts_with("std::slice::ChunksMut<")
            || self_ty.starts_with("std::slice::RChunksMut<")
            || self_ty.starts_with("std::slice::ChunksExactMut<")
            || self_ty.starts_with("std::slice::RChunksExactMut<")
        {
            return self.chunk_step_state(frames, self_ty, &state, true);
        }
        if is_str_split_ty(self_ty) {
            return self.str_split_step(frames, &state, self_ty, true);
        }
        if is_str_matches_ty(self_ty) {
            return self.str_matches_step(frames, &state, self_ty, true);
        }
        if self_ty.starts_with("std::vec::Drain<")
            || self_ty.starts_with("alloc::vec::Drain<")
        {
            return self.drain_next_state(frames, state, true);
        }
        if self_ty.starts_with("std::string::Drain")
            || self_ty.starts_with("alloc::string::Drain")
        {
            return self.string_drain_next_state(frames, state, true);
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
                let keep = self.call_callable_sync_typed(
                    frames,
                    &predicate,
                    second_generic_arg(self_ty).unwrap_or(""),
                    None,
                    vec![arg],
                )?;
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
                    self.call_callable_sync_typed(
                        frames,
                        &closure,
                        second_generic_arg(self_ty).unwrap_or(""),
                        None,
                        vec![x],
                    )?;
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
                        self.call_callable_sync_typed(
                            frames,
                            &fields[1],
                            second_generic_arg(self_ty).unwrap_or(""),
                            None,
                            vec![x],
                        )?;
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

    /// One step of a `std::string::Drain`: the state is the
    /// real layout the constructor's converted body built --
    /// `[string, start, end, iter]` -- and a step is exactly
    /// the wrapped `Chars` step std's own `next` / `next_back`
    /// forward to (the yielded chars are copies; the byte
    /// range itself leaves the string only when the drain
    /// value is dropped, by the `StringDrainDrop` plan). Any
    /// other field shape keeps the generic unsupported error.
    fn string_drain_next_state(
        &mut self,
        frames: &mut Vec<Frame>,
        state: Value,
        from_back: bool,
    ) -> InterpResult<(Value, Value)> {
        let fields = iterator_fields(&state, "std::string::Drain")?;
        if fields.len() != 4 {
            return Err(InterpError::type_mismatch(
                "string Drain state",
                "[string, start, end, iter]",
                format!("{} field(s)", fields.len()),
            ));
        }
        // A drain built over a string in its real aggregate
        // layout holds a `Chars` whose byte pointers denote
        // the string's own storage rather than a value
        // snapshot; snapshot the remaining bytes once, after
        // which the state has the builtin form and steps
        // normally.
        let chars = match self.string_drain_chars_snapshot(frames, &fields[3]) {
            Some(normalized) => normalized?,
            None => fields[3].clone(),
        };
        let (new_chars, item) = if from_back {
            self.next_back_on_state(frames, "std::str::Chars", chars)?
        } else {
            self.iterator_next_state(frames, "std::str::Chars", chars)?
        };
        let mut fields = fields;
        fields[3] = new_chars;
        Ok((rebuild_state(&state, fields), item))
    }

    /// The builtin snapshot form of a string drain's `Chars`
    /// state when its byte pointers are storage-backed (the
    /// drain was built over a string in its real aggregate
    /// layout): the remaining bytes read through the pointers,
    /// rebuilt with `str_chars_value`. `None` when the state
    /// already denotes a value snapshot (or anything else the
    /// generic `Chars` step itself judges).
    fn string_drain_chars_snapshot(
        &mut self,
        frames: &mut Vec<Frame>,
        chars: &Value,
    ) -> Option<InterpResult<Value>> {
        let inner = str_iter_inner(chars, "std::str::Chars").ok()?;
        let fields = iterator_fields(&inner, "std::slice::Iter").ok()?;
        if fields.len() != 3 {
            return None;
        }
        let ptr = iter_ptr_value(&fields[0])?;
        let end = iter_ptr_value(&fields[1])?;
        let (Value::Ref(ptr_target), Value::Ref(end_target)) = (&ptr, &end) else {
            return None;
        };
        if matches!(ptr_target, RefTarget::Value(..) | RefTarget::Null) {
            return None;
        }
        let end_target = end_target.clone();
        let mut bytes = Vec::new();
        let mut cur = ptr;
        loop {
            let Value::Ref(target) = &cur else {
                return None;
            };
            if *target == end_target {
                break;
            }
            let byte = match self
                .deref_value(frames, &cur)
                .and_then(|resolved| self.read_resolved(frames, &resolved))
            {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
            };
            bytes.push(byte);
            cur = match self.ptr_offset_value(frames, cur, 1, Some(1), false) {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
            };
        }
        Some(Ok(str_chars_value(bytes)))
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
        if self_ty.starts_with("std::string::Drain")
            || self_ty.starts_with("alloc::string::Drain")
        {
            return self.string_drain_next_state(frames, state, false);
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
        if self_ty.starts_with("std::sync::mpsc::TryIter<") {
            // The state is the value `Receiver::try_iter`
            // built: the receiver handle inside a one-field
            // aggregate (or the handle itself). `next` is
            // `try_recv`: a queued value yields; an
            // empty-but-live channel and a disconnected one
            // both end the iteration.
            let handle = match &state {
                Value::Enum { fields, .. } if fields.len() == 1 => fields[0].clone(),
                Value::Tuple(fields) if fields.len() == 1 => fields[0].clone(),
                other => other.clone(),
            };
            let id = match channel_handle_id(&handle) {
                Some(id) => id,
                None => self.handle_id_of_ref(
                    frames,
                    &handle,
                    "Iterator::next for mpsc::TryIter",
                )?,
            };
            return match self.with_channel(id, |st| st.queue.pop_front()) {
                Some(Some(value)) => Ok((state, some_value(value))),
                Some(None) => Ok((state, none_value())),
                None => Err(unsupported(
                    "Iterator::next for mpsc::TryIter on an unknown channel".to_string(),
                )),
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
        if self_ty_is(self_ty, "std::env::Vars") || self_ty_is(self_ty, "std::env::VarsOs") {
            // `Vars { inner: VarsOs }` / `VarsOs { inner:
            // sys::env::Env }` / the common `Env { iter:
            // vec::IntoIter<(OsString, OsString)> }`: step the
            // innermost owned-iterator state and rebuild the
            // wrappers, mirroring the `Args` step above. The
            // stored elements are the environment snapshot's
            // pairs in the standard `OsString` buffer form;
            // `VarsOs::next` yields them as-is, while
            // `Vars::next` converts each side through
            // `OsString::into_string`, whose unwrap panics on
            // invalid Unicode (see `vars_pair_to_strings`).
            let depth = if self_ty_is(self_ty, "std::env::Vars") { 3 } else { 2 };
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
            let item = match item {
                Some(value) if self_ty_is(self_ty, "std::env::Vars") => {
                    Some(vars_pair_to_strings(value)?)
                }
                other => other,
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
            // A yielded window is a snapshot: the step
            // materialised exactly the window's elements, so
            // the fat reference's base is that element tuple
            // itself -- the denomination `index_range`'s
            // materialised sub-slices use (see
            // `SliceElems::sub_snapshot`). Denominating the
            // base as an element position inside the tuple
            // (a reference to the tuple at index 0) instead
            // makes an index step through the window compose
            // onto that position and land inside the first
            // element whenever the element is itself a
            // sequence, so `window[0]` over a
            // `Vec<Vec<usize>>` read the inner vector's first
            // element rather than the vector.
            let tail_slice = |elems: &[Value]| {
                Value::FatRef {
                    base: Box::new(Value::Tuple(elems.to_vec())),
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
                let (elems, _) =
                    self.adapter_slice_elements(frames, self_ty, &fields[0])?;
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
            let (elems, _) =
                self.adapter_slice_elements(frames, self_ty, &fields[0])?;
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
        if let Some(map_ty) = Self::lines_map_ty(self_ty) {
            // The value-level step of a `Lines`: step the wrapped
            // `Map` state and rewrap it in the one-field state.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "Lines state",
                    "[map]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let (new_map, result) =
                self.next_on_state(frames, map_ty, &fields[0])?;
            return Ok((rebuild_state(&state, vec![new_map]), result));
        }
        if let Some(filter_ty) = Self::split_whitespace_filter_ty(self_ty) {
            // The value-level step of a `SplitWhitespace`:
            // step the wrapped `Filter` state and rewrap it
            // in the one-field state.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "SplitWhitespace state",
                    "[filter]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let (new_filter, result) =
                self.next_on_state(frames, filter_ty, &fields[0])?;
            return Ok((rebuild_state(&state, vec![new_filter]), result));
        }
        if let Some(flatmap_ty) = Self::str_escape_flatmap_ty(self_ty) {
            // The value-level step of a `str` escape
            // iterator: step the wrapped `FlatMap` state
            // and rewrap it in the one-field state.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "str escape iterator state",
                    "[flat map]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let (new_inner, result) =
                self.next_on_state(frames, flatmap_ty, &fields[0])?;
            return Ok((rebuild_state(&state, vec![new_inner]), result));
        }
        if self_ty.starts_with("std::slice::Chunks<")
            || self_ty.starts_with("std::slice::RChunks<")
            || self_ty.starts_with("std::slice::ChunksExact<")
            || self_ty.starts_with("std::slice::RChunksExact<")
            || self_ty.starts_with("std::slice::ChunksMut<")
            || self_ty.starts_with("std::slice::RChunksMut<")
            || self_ty.starts_with("std::slice::ChunksExactMut<")
            || self_ty.starts_with("std::slice::RChunksExactMut<")
        {
            return self.chunk_step_state(frames, self_ty, &state, false);
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

    /// One step of a slice chunk iterator's state (`Chunks`,
    /// `RChunks` and their `Exact` variants), in the front
    /// direction when `from_back` is false and in the back
    /// direction when it is true. The state is the one the
    /// collected `slice::chunks` family bodies build:
    /// [remaining slice, chunk size] for the plain variants
    /// and [remaining whole-chunk slice, remainder slice,
    /// chunk size] for the `Exact` variants, whose stored
    /// slice holds only whole chunks, so either end takes a
    /// full chunk or none. For the plain variants the short
    /// chunk sits at the end the primary direction yields
    /// last: `Chunks` splits anchored at the front (short
    /// chunk at the tail), `RChunks` anchored at the back
    /// (short chunk at the head), and a back step of either
    /// takes that short chunk first. The yielded
    /// sub-slices are snapshot fat references, the form the
    /// `Windows` step above yields: the chunk family
    /// modelled here yields shared slices, whose reads the
    /// snapshot serves exactly. A `Mut` variant's chunk must
    /// instead alias the storage the remaining slice
    /// denotes, or writes through it are lost: when the
    /// state's slice is backed by live storage, the chunk
    /// and the remaining slice are denominated as windows
    /// over that storage, exactly as a range index over the
    /// slice denominates its result (see `index_range`);
    /// only a slice with no live storage behind it keeps
    /// the snapshot form, whose elements are the only
    /// storage the model has for it.
    fn chunk_step_state(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        state: &Value,
        from_back: bool,
    ) -> InterpResult<(Value, Value)> {
        let mutable = self_ty.starts_with("std::slice::ChunksMut<")
            || self_ty.starts_with("std::slice::RChunksMut<")
            || self_ty.starts_with("std::slice::ChunksExactMut<")
            || self_ty.starts_with("std::slice::RChunksExactMut<");
        let reverse = self_ty.starts_with("std::slice::RChunks<")
            || self_ty.starts_with("std::slice::RChunksMut<")
            || self_ty.starts_with("std::slice::RChunksExact<")
            || self_ty.starts_with("std::slice::RChunksExactMut<");
        let exact = self_ty.starts_with("std::slice::ChunksExact<")
            || self_ty.starts_with("std::slice::ChunksExactMut<")
            || self_ty.starts_with("std::slice::RChunksExact<")
            || self_ty.starts_with("std::slice::RChunksExactMut<");
        let take_back = reverse != from_back;
        let fields = iterator_fields(state, self_ty)?;
        // The `Mut` variants carry one more field than
        // their shared counterparts: a trailing
        // `PhantomData` marker (present on `ChunksMut`,
        // `RChunksMut` and `ChunksExactMut`;
        // `RChunksExactMut` has none), the same shapes the
        // `size_hint` arm above reads. The marker is inert
        // state: it rides in `rest_fields` after the chunk
        // size and is carried through the step untouched.
        let (slice_field, rest_fields): (&Value, Vec<Value>) = if exact {
            if fields.len() != 3 && !(mutable && fields.len() == 4) {
                return Err(InterpError::type_mismatch(
                    format!("{self_ty} state"),
                    "[remaining slice, remainder slice, chunk size]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            (&fields[0], fields[1..].to_vec())
        } else {
            if fields.len() != 2 && !(mutable && fields.len() == 3) {
                return Err(InterpError::type_mismatch(
                    format!("{self_ty} state"),
                    "[remaining slice, chunk size]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            (&fields[0], fields[1..].to_vec())
        };
        let size = chunk_size_of(
            if exact {
                &rest_fields[1]
            } else {
                &rest_fields[0]
            },
            self_ty,
        )?;
        let (elems, _) =
            self.adapter_slice_elements(frames, self_ty, slice_field)?;
        let len = elems.len();
        if size == 0 || len == 0 {
            return Ok((state.clone(), none_value()));
        }
        // A yielded chunk is a snapshot, denominated
        // exactly as the `Windows` step's windows are (see
        // `tail_slice` there): the base is the chunk's
        // materialised element tuple itself. For a `Mut`
        // variant the denomination is the window over the
        // state's own slice base when that base is live
        // storage (see the doc comment above), so `start`
        // -- the part's offset within the remaining slice --
        // is carried as well.
        let sub_slice = |start: usize, part: &[Value]| -> InterpResult<Value> {
            if mutable {
                if let Some(base) = self.range_view_base(
                    frames,
                    slice_field,
                    start,
                    part.len(),
                    len,
                    "chunk step",
                )? {
                    return Ok(Value::FatRef {
                        base: Box::new(base),
                        len: part.len() as u128,
                    });
                }
            }
            Ok(Value::FatRef {
                base: Box::new(Value::Tuple(part.to_vec())),
                len: part.len() as u128,
            })
        };
        // The plain variants' short chunk sits at the end
        // the primary direction yields last: a front step
        // always takes a full chunk (or what remains), and a
        // back step takes the short chunk first -- the tail
        // chunk of a `Chunks` (whose splitting is anchored at
        // the front) and the head chunk of an `RChunks`
        // (whose splitting is anchored at the back) alike.
        // The item's range within the remaining slice; the
        // rest is the complementary range (the item sits at
        // the front and the rest follows it, or the item
        // sits at the back and the rest precedes it).
        let (item_start, item_len): (usize, usize) = if exact {
            if len < size {
                return Ok((state.clone(), none_value()));
            }
            if take_back {
                (len - size, size)
            } else {
                (0, size)
            }
        } else if from_back {
            let n = match len % size {
                0 => size,
                r => r,
            };
            if take_back {
                (len - n, n)
            } else {
                (0, n)
            }
        } else {
            let n = size.min(len);
            if take_back {
                (len - n, n)
            } else {
                (0, n)
            }
        };
        let item_elems = &elems[item_start..item_start + item_len];
        let (rest_start, rest_elems) = if item_start == 0 {
            (item_len, &elems[item_len..])
        } else {
            (0, &elems[..item_start])
        };
        let mut new_fields = vec![sub_slice(rest_start, rest_elems)?];
        new_fields.extend(rest_fields);
        Ok((
            rebuild_state(state, new_fields),
            some_value(sub_slice(item_start, item_elems)?),
        ))
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
        let mapped = self.call_callable_sync_typed(
            frames,
            &callable,
            second_generic_arg(self_ty).unwrap_or(""),
            Some(env_ref),
            vec![item],
        )?;
        Ok(Some(some_value(mapped)))
    }

    /// One front (`from_back == false`) or back step of a
    /// flatten composition on the iterator stored at `target`:
    /// `FlatMap` and `Flatten` are one-field wrappers over the
    /// `FlattenCompat` state core's step machine runs on, so
    /// the step is `flatten_compat_next_at` over that field (or
    /// over the target itself for a bare `FlattenCompat`).
    /// Declines (returns `Ok(None)`) when the composition's
    /// type triple cannot be derived from the rendered tag (see
    /// `flatten_step_types`), leaving the caller's state-level
    /// step — and its unsupported verdict — in place.
    fn flatten_next_at(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        target: &Resolved,
        from_back: bool,
    ) -> InterpResult<Option<Value>> {
        let Some((outer_ty, item_ty, inner_ty)) = flatten_step_types(self_ty) else {
            return Ok(None);
        };
        if self_ty.starts_with("std::iter::adapters::flatten::FlattenCompat<") {
            return self.flatten_compat_next_at(
                frames,
                &outer_ty,
                &item_ty,
                &inner_ty,
                target,
                from_back,
            );
        }
        let Some(inner_target) = extend_target(target, RefProj::Field(0)) else {
            return Ok(None);
        };
        self.flatten_compat_next_at(
            frames,
            &outer_ty,
            &item_ty,
            &inner_ty,
            &inner_target,
            from_back,
        )
    }

    /// The `FlattenCompat` step machine in place, replaying
    /// core's `next` / `next_back` on the three fields at
    /// `target`: (outer iterator in its `Fuse`, front inner
    /// iterator, back inner iterator). The leading inner
    /// iterator is drained first; when it runs out it is
    /// cleared and the outer iterator is pulled for the next
    /// inner iterator — the pulled item is converted with the
    /// same `IntoIterator::into_iter` transformation the
    /// guest's own call performs — and when the outer iterator
    /// is exhausted (its `Fuse` blown) the trailing inner
    /// iterator serves the rest. Every sub-step is taken in
    /// place, so an iterator whose state lives in the
    /// composition's own storage (a `Map`'s closure environment
    /// above all) steps exactly as it does at its own address.
    fn flatten_compat_next_at(
        &mut self,
        frames: &mut Vec<Frame>,
        outer_ty: &str,
        item_ty: &str,
        inner_ty: &str,
        target: &Resolved,
        from_back: bool,
    ) -> InterpResult<Option<Value>> {
        let Some(fuse_target) = extend_target(target, RefProj::Field(0)) else {
            return Ok(None);
        };
        let Some(fuse_iter_target) = extend_target(&fuse_target, RefProj::Field(0)) else {
            return Ok(None);
        };
        let (lead_idx, trail_idx) = if from_back { (2, 1) } else { (1, 2) };
        let Some(lead_target) = extend_target(target, RefProj::Field(lead_idx)) else {
            return Ok(None);
        };
        let Some(trail_target) = extend_target(target, RefProj::Field(trail_idx)) else {
            return Ok(None);
        };
        loop {
            match self.option_iter_step_at(frames, inner_ty, &lead_target, from_back)? {
                Some(SlotStep::Yielded(item)) => {
                    return Ok(Some(some_value(item)));
                }
                Some(SlotStep::Exhausted) | Some(SlotStep::Empty) => {}
                None => return Ok(None),
            }
            match self.option_iter_step_at(frames, outer_ty, &fuse_iter_target, from_back)? {
                Some(SlotStep::Yielded(iterable)) => {
                    let inner_state = self.into_iter_value(frames, item_ty, iterable)?;
                    self.write_resolved(frames, &lead_target, some_value(inner_state))?;
                }
                Some(SlotStep::Exhausted) | Some(SlotStep::Empty) => {
                    // The outer iterator is spent: the trailing
                    // inner iterator serves whatever remains,
                    // and its step's result is the step's result.
                    match self.option_iter_step_at(
                        frames,
                        inner_ty,
                        &trail_target,
                        from_back,
                    )? {
                        Some(SlotStep::Yielded(item)) => {
                            return Ok(Some(some_value(item)));
                        }
                        Some(SlotStep::Exhausted) | Some(SlotStep::Empty) => {
                            return Ok(Some(none_value()));
                        }
                        None => return Ok(None),
                    }
                }
                None => return Ok(None),
            }
        }
    }

    /// One step of an `Option<I>` iterator slot in place (a
    /// `FlattenCompat` inner-iterator field, or the payload
    /// slot of its `Fuse` field, which has the same shape):
    /// core's `and_then_or_clear` — a step that yields keeps
    /// the new state in the slot, a step that returns `None`
    /// clears the slot to `None`. Declines when the slot does
    /// not live in a place or does not hold an `Option` value.
    fn option_iter_step_at(
        &mut self,
        frames: &mut Vec<Frame>,
        iter_ty: &str,
        slot_target: &Resolved,
        from_back: bool,
    ) -> InterpResult<Option<SlotStep>> {
        let slot = self.read_resolved(frames, slot_target)?;
        match &slot {
            Value::Enum { variant: 0, .. } => return Ok(Some(SlotStep::Empty)),
            Value::Enum {
                variant: 1,
                fields,
            } if fields.len() == 1 => {}
            _ => return Ok(None),
        }
        let Some(payload_target) = extend_target(slot_target, RefProj::Downcast(1))
            .and_then(|t| extend_target(&t, RefProj::Field(0)))
        else {
            return Ok(None);
        };
        let item = if from_back {
            self.iterator_next_back_at(frames, iter_ty, &payload_target)?
        } else {
            self.iterator_next_at(frames, iter_ty, &payload_target)?
        };
        match option_payload_opt(item) {
            Some(item) => Ok(Some(SlotStep::Yielded(item))),
            None => {
                self.write_resolved(frames, slot_target, none_value())?;
                Ok(Some(SlotStep::Exhausted))
            }
        }
    }

    /// One front or back step of a flatten composition over a
    /// state value (the value-level counterpart of
    /// `flatten_next_at`, for a composition that does not live
    /// in a place): the same `FlattenCompat` machine, with the
    /// sub-steps taken functionally through `next_on_state` /
    /// `next_back_on_state`. Declines under the same condition
    /// as `flatten_next_at`.
    fn flatten_step_state(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        state: &Value,
        from_back: bool,
    ) -> InterpResult<Option<(Value, Value)>> {
        let Some((outer_ty, item_ty, inner_ty)) = flatten_step_types(self_ty) else {
            return Ok(None);
        };
        if self_ty.starts_with("std::iter::adapters::flatten::FlattenCompat<") {
            let step = self.flatten_compat_step_state(
                frames,
                &outer_ty,
                &item_ty,
                &inner_ty,
                state,
                from_back,
            )?;
            return Ok(Some(step));
        }
        let fields = iterator_fields(state, self_ty)?;
        if fields.len() != 1 {
            return Err(InterpError::type_mismatch(
                format!("{self_ty} state"),
                "a one-field wrapper around a FlattenCompat",
                format!("{} field(s)", fields.len()),
            ));
        }
        let (new_inner, item) = self.flatten_compat_step_state(
            frames,
            &outer_ty,
            &item_ty,
            &inner_ty,
            &fields[0],
            from_back,
        )?;
        Ok(Some((rebuild_state(state, vec![new_inner]), item)))
    }

    /// The value-level `FlattenCompat` step machine: the
    /// functional form of `flatten_compat_next_at`, over the
    /// state value's three fields.
    fn flatten_compat_step_state(
        &mut self,
        frames: &mut Vec<Frame>,
        outer_ty: &str,
        item_ty: &str,
        inner_ty: &str,
        state: &Value,
        from_back: bool,
    ) -> InterpResult<(Value, Value)> {
        let fields = iterator_fields(state, "FlattenCompat state")?;
        if fields.len() != 3 {
            return Err(InterpError::type_mismatch(
                "FlattenCompat state",
                "[fuse, front inner iterator, back inner iterator]",
                format!("{} field(s)", fields.len()),
            ));
        }
        let fuse_wrapper = fields[0].clone();
        let fuse_fields = iterator_fields(&fuse_wrapper, "Fuse state")?;
        if fuse_fields.len() != 1 {
            return Err(InterpError::type_mismatch(
                "Fuse state",
                "a one-field wrapper around an Option",
                format!("{} field(s)", fuse_fields.len()),
            ));
        }
        let mut fuse = fuse_fields.into_iter().next().expect("one field checked");
        let (mut lead, mut trail) = if from_back {
            (fields[2].clone(), fields[1].clone())
        } else {
            (fields[1].clone(), fields[2].clone())
        };
        // The state with the step's field values, in field
        // order, under `state`'s own wrapper shapes.
        let rebuild = |fuse: &Value, lead: &Value, trail: &Value| {
            let fuse_now = iterator_state_like(&fuse_wrapper, vec![fuse.clone()]);
            let (front, back) = if from_back {
                (trail.clone(), lead.clone())
            } else {
                (lead.clone(), trail.clone())
            };
            rebuild_state(state, vec![fuse_now, front, back])
        };
        loop {
            if let Some(inner_state) = option_payload_opt(lead.clone()) {
                let (new_inner, item) = if from_back {
                    self.next_back_on_state(frames, inner_ty, inner_state)?
                } else {
                    self.next_on_state(frames, inner_ty, &inner_state)?
                };
                match option_payload_opt(item) {
                    Some(item) => {
                        lead = some_value(new_inner);
                        return Ok((rebuild(&fuse, &lead, &trail), some_value(item)));
                    }
                    None => {
                        lead = none_value();
                        continue;
                    }
                }
            }
            match option_payload_opt(fuse.clone()) {
                None => {
                    // The outer iterator is spent: the trailing
                    // inner iterator serves whatever remains.
                    if let Some(trail_state) = option_payload_opt(trail.clone()) {
                        let (new_trail, item) = if from_back {
                            self.next_back_on_state(frames, inner_ty, trail_state)?
                        } else {
                            self.next_on_state(frames, inner_ty, &trail_state)?
                        };
                        if let Some(item) = option_payload_opt(item) {
                            trail = some_value(new_trail);
                            return Ok((rebuild(&fuse, &lead, &trail), some_value(item)));
                        }
                        trail = none_value();
                    }
                    return Ok((rebuild(&fuse, &lead, &trail), none_value()));
                }
                Some(outer_state) => {
                    let (new_outer, item) = if from_back {
                        self.next_back_on_state(frames, outer_ty, outer_state)?
                    } else {
                        self.next_on_state(frames, outer_ty, &outer_state)?
                    };
                    match option_payload_opt(item) {
                        Some(iterable) => {
                            fuse = some_value(new_outer);
                            let inner_state =
                                self.into_iter_value(frames, item_ty, iterable)?;
                            lead = some_value(inner_state);
                            continue;
                        }
                        None => {
                            fuse = none_value();
                            continue;
                        }
                    }
                }
            }
        }
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
            let keep = self.call_callable_sync_typed(
                frames,
                &callable,
                second_generic_arg(self_ty).unwrap_or(""),
                Some(env_ref),
                vec![arg],
            )?;
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
        // A reference-wrapped iterator (`&mut I`, and the
        // shared `&I` form): core's blanket `Iterator` impl
        // forwards the step to the pointee, and the modelled
        // consumers that compose steps hand the stored
        // reference value here with the reference tag (a `Zip`
        // steps its `&mut I` side as a state field), so the
        // forwarding belongs to the state machine as well as
        // to the collected blanket body. The step is in place
        // when the reference denotes a place, functional when
        // it denotes a value; the reference itself is the
        // state and is returned unchanged. A state that is not
        // a reference value (a fat trait-object reference, a
        // materialised aggregate) declines and falls through
        // to the tag dispatch below, exactly as before.
        if let Some(inner_ty) = self_ty
            .strip_prefix("&mut ")
            .or_else(|| self_ty.strip_prefix('&'))
        {
            match state {
                Value::Ref(RefTarget::Value(pointee, projection))
                    if projection.is_empty() =>
                {
                    let (new_inner, item) = self.iterator_next_state(
                        frames,
                        inner_ty,
                        (**pointee).clone(),
                    )?;
                    return Ok((
                        Value::Ref(RefTarget::Value(
                            Box::new(new_inner),
                            projection.clone(),
                        )),
                        item,
                    ));
                }
                Value::Ref(_) => {
                    let target = self.deref_value(frames, state)?;
                    let item = self.iterator_next_at(frames, inner_ty, &target)?;
                    return Ok((state.clone(), item));
                }
                _ => {}
            }
        }
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
        if self_ty.starts_with("std::slice::Chunks<")
            || self_ty.starts_with("std::slice::RChunks<")
            || self_ty.starts_with("std::slice::ChunksExact<")
            || self_ty.starts_with("std::slice::RChunksExact<")
            || self_ty.starts_with("std::slice::ChunksMut<")
            || self_ty.starts_with("std::slice::RChunksMut<")
            || self_ty.starts_with("std::slice::ChunksExactMut<")
            || self_ty.starts_with("std::slice::RChunksExactMut<")
        {
            // The chunk step also lives in
            // `iterator_next_state`; it belongs here too
            // because the adapter arms below step their inner
            // iterators through this entry (a `Map` over a
            // `Chunks` reaches the chunk state only from
            // here).
            return self.chunk_step_state(frames, self_ty, state, false);
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
            return self.str_split_step(frames, state, self_ty, false);
        }
        if is_str_splitn_ty(self_ty) {
            return self.str_splitn_step(frames, state, self_ty);
        }
        if is_str_matches_ty(self_ty) {
            return self.str_matches_step(frames, state, self_ty, false);
        }
        if self_ty.starts_with("std::sync::mpsc::Iter<")
            || self_ty.starts_with("std::sync::mpsc::IntoIter<")
            || self_ty.starts_with("std::sync::mpsc::TryIter<")
        {
            // The mpsc iterators' step lives in
            // `iterator_next_state` (a blocking `recv` for
            // `Iter` / `IntoIter`, a `try_recv` for
            // `TryIter`); it belongs here too because the
            // adapter arms below step their inner iterators
            // through this entry (a `Take` over an
            // `mpsc::Iter` reaches the channel only from
            // here). The delegation cannot recurse:
            // `iterator_next_state` serves these types
            // itself and never falls through to here for
            // them.
            return self.iterator_next_state(frames, self_ty, state.clone());
        }
        if self_ty.starts_with("std::iter::FlatMap<")
            || self_ty.starts_with("std::iter::Flatten<")
            || self_ty.starts_with("std::iter::adapters::flatten::FlattenCompat<")
        {
            if let Some(step) = self.flatten_step_state(frames, self_ty, state, false)? {
                return Ok(step);
            }
        }
        if is_str_escape_ty(self_ty) {
            // The `str` escape iterators' step is the
            // wrapped `FlatMap` composition's (see
            // `str_escape_flatmap_ty`); it belongs here
            // because the adapter arms below step their
            // inner iterators through this entry (a `Map`
            // over a `str::EscapeDebug` reaches the
            // composition only from here). The delegation
            // cannot recurse: `iterator_next_state` serves
            // these types by stepping the field under the
            // `FlatMap` tag, which the flatten arm above
            // serves directly.
            return self.iterator_next_state(frames, self_ty, state.clone());
        }
        if is_char_escape_ty(self_ty) {
            // The `char` escape iterators' step lives in
            // `step_escape_inner` (the `EscapeIterInner`
            // buffer state machine); it belongs here because
            // the adapter arms below step their inner
            // iterators through this entry (a `Map` over an
            // `EscapeDebug` reaches the buffer only from
            // here).
            let (new_state, item) = step_escape_inner(state, self_ty, false)?;
            return Ok((
                new_state,
                match item {
                    Some(value) => some_value(value),
                    None => none_value(),
                },
            ));
        }
        let fields = iterator_fields(state, self_ty)?;
        // A `vec::IntoIter` in its real field layout (the
        // aggregate an inlined `Vec::into_iter` construction
        // inside a collected body builds -- the iterator the
        // B-tree bulk sorted build steps, above all): the
        // modelled arm below steps only the [items, next
        // index] state the `IntoIterator` builtin builds, so
        // the real layout is stepped on its pointer pair
        // first, the real-layout step the slice iterators get
        // from `slice_iter_step_real`.
        if (self_ty.starts_with("std::vec::IntoIter<")
            || self_ty.starts_with("alloc::vec::IntoIter<"))
            && fields.len() == 6
        {
            if let Some(step) = vec_into_iter_step_real(self, frames, self_ty, &fields) {
                let (new_fields, item) = step?;
                return Ok((iterator_state_like(state, new_fields), item));
            }
        }
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
        } else if self_ty.starts_with("std::ops::RangeFrom<") {
            // An unbounded range: the state is the next bound
            // alone, and a step yields it and stores its
            // successor -- core's `Step::forward`, whose
            // overflow behaviour (the addition panic under
            // overflow checks, the wrap to the minimum
            // without) the session's own setting decides, as
            // the `#[rustc_inherit_overflow_checks]` on the
            // step prescribes. The successor is computed
            // first, exactly as core's `next` computes it
            // before yielding, so stepping at the maximum
            // panics without yielding.
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "RangeFrom state",
                    "[start]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let elem = range_elem_ty(self_ty);
            if !range_elem_step_served(elem) {
                return Err(unsupported(format!(
                    "Iterator::next for {self_ty} is not runtime-supported"
                )));
            }
            let start = fields[0].clone();
            let next = range_step_forward(&start, elem, self.overflow_checks())?;
            (vec![next], some_value(start))
        } else if self_ty.starts_with("std::range::RangeFromIter<") {
            // The new range API's by-value iterator: the
            // state is [start, exhausted], where the flag
            // records that the maximum has been yielded under
            // overflow checks. The step replays core's
            // implementation (library/core/src/range/iter.rs)
            // exactly: an exhausted iterator first re-steps
            // its bound -- `Step::forward`, which panics
            // under overflow checks and wraps without -- and
            // clears the flag; then the step itself is the
            // checked step when the session has overflow
            // checks (the maximum is yielded once, with the
            // flag set and the bound left in place) and the
            // wrapping step when it does not.
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "RangeFromIter state",
                    "[start, exhausted]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let elem = range_elem_ty(self_ty);
            if !range_elem_step_served(elem) {
                return Err(unsupported(format!(
                    "Iterator::next for {self_ty} is not runtime-supported"
                )));
            }
            let mut start = fields[0].clone();
            let exhausted = match &fields[1] {
                Value::Bool(b) => *b,
                other => {
                    return Err(InterpError::type_mismatch(
                        "RangeFromIter exhausted flag",
                        "boolean",
                        other.kind_name(),
                    ))
                }
            };
            if exhausted {
                start = range_step_forward(&start, elem, self.overflow_checks())?;
            }
            if self.overflow_checks() {
                match range_step_forward_checked(&start, elem)? {
                    Some(next) => (
                        vec![next, Value::Bool(false)],
                        some_value(start),
                    ),
                    None => (
                        vec![start.clone(), Value::Bool(true)],
                        some_value(start),
                    ),
                }
            } else {
                let next = range_step_forward(&start, elem, false)?;
                (vec![next, Value::Bool(false)], some_value(start))
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
                let keep = self.call_callable_sync_typed(
                    frames,
                    &predicate,
                    second_generic_arg(self_ty).unwrap_or(""),
                    None,
                    vec![arg],
                )?;
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
                let mapped = self.call_callable_sync_typed(
                    frames,
                    &closure,
                    second_generic_arg(self_ty).unwrap_or(""),
                    None,
                    vec![x],
                )?;
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
                let keep = self.call_callable_sync_typed(
                    frames,
                    &predicate,
                    second_generic_arg(self_ty).unwrap_or(""),
                    None,
                    vec![arg],
                )?;
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
                        self.call_callable_sync_typed(
                            frames,
                            &fields[1],
                            second_generic_arg(self_ty).unwrap_or(""),
                            None,
                            vec![x],
                        )?;
                    (
                        vec![new_inner, fields[1].clone()],
                        some_value(mapped),
                    )
                }
            }
        } else if self_ty.starts_with("std::iter::StepBy<") {
            // `StepBy` is (inner iterator, step - 1, first
            // take). The generic step (core's default
            // `StepByImpl::spec_next`) is the inner `nth` of
            // 0 on the first step and of `step - 1` after
            // that: skip that many items, then yield the
            // next; an inner iterator that runs out during
            // the skips yields `None` and stays exhausted.
            // For an integer `Range` inner (and the new
            // `RangeIter`) core specialises instead: the
            // constructor has rewritten the range's end into
            // the count of items left to yield, and the step
            // yields the bound while the count lasts, steps
            // it by the (element-clamped) step, and counts
            // down -- the `first take` flag plays no part.
            if fields.len() != 3 {
                return Err(InterpError::type_mismatch(
                    "StepBy state",
                    "[iterator, step - 1, first take]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let step_minus_one = match fields[1] {
                Value::Uint(v) => v,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "StepBy step",
                        "unsigned integer",
                        other.kind_name(),
                    ))
                }
            };
            let first_take = match fields[2] {
                Value::Bool(b) => b,
                ref other => {
                    return Err(InterpError::type_mismatch(
                        "StepBy first take flag",
                        "boolean",
                        other.kind_name(),
                    ))
                }
            };
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            if let Some(elem) = step_by_spec_range_elem(inner_ty) {
                let rfields = iterator_fields(&fields[0], inner_ty)?;
                if rfields.len() != 2 {
                    return Err(InterpError::type_mismatch(
                        "StepBy specialised range state",
                        "[start, remaining count]",
                        format!("{} field(s)", rfields.len()),
                    ));
                }
                let remaining = match rfields[1] {
                    Value::Uint(v) => v,
                    Value::Int(v) => v as u128,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            "StepBy specialised range count",
                            "integer",
                            other.kind_name(),
                        ))
                    }
                };
                if remaining == 0 {
                    (fields, none_value())
                } else {
                    let step = (step_minus_one + 1).min(
                        range_elem_max(elem).expect("specialised element"),
                    );
                    let new_start = range_wrapping_add(&rfields[0], elem, step)
                        .expect("specialised element steps");
                    let new_remaining = match rfields[1] {
                        Value::Uint(v) => Value::Uint(v - 1),
                        Value::Int(v) => Value::Int(v - 1),
                        _ => unreachable!("remaining checked above"),
                    };
                    let new_inner = rebuild_state(
                        &fields[0],
                        vec![new_start, new_remaining],
                    );
                    (
                        vec![new_inner, fields[1].clone(), fields[2].clone()],
                        some_value(rfields[0].clone()),
                    )
                }
            } else {
                let mut inner = fields[0].clone();
                let mut to_skip = if first_take { 0 } else { step_minus_one };
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
                (
                    vec![inner, fields[1].clone(), Value::Bool(false)],
                    result,
                )
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
            // No arm claims this tag: the type is not one the
            // modelled machine serves. When the program holds
            // the type's own collected `next` body -- a
            // user-defined iterator reached as the leaf of a
            // modelled composition (an adapter chain, a
            // consumer's drain, the `&mut I` blanket
            // forwarding) -- that body is the faithful step.
            if let Some(stepped) = self.impl_body_next_step(frames, self_ty, state)? {
                return Ok(stepped);
            }
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

    /// One front step of an iterator whose type the modelled
    /// state machine does not serve, executed by the type's
    /// own collected `Iterator::next` body. The machine's
    /// consumers and adapters compose steps by tag, so a
    /// user-defined iterator at the leaf of such a composition
    /// reaches the machine with a tag no arm claims, while the
    /// program can hold the very impl body the direct call
    /// sites run. The state is parked in a fresh allocation --
    /// the arrangement [`Self::drain_iterator`] uses -- so the
    /// body steps it in place through the allocation
    /// reference, and the final state is read back from the
    /// allocation. `Ok(None)` when the program holds no
    /// converted body for the type's `next`: the scan must
    /// name exactly one impl, and a body that was demanded but
    /// not converted is not a dispatch target, as in
    /// `call_callable_sync_typed`. The caller then reports the
    /// type as not runtime-supported, as before.
    fn impl_body_next_step(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        state: &Value,
    ) -> InterpResult<Option<(Value, Value)>> {
        let Some(impl_name) = self.scan_virtual_impl(self_ty, "std::iter::Iterator", "next")
        else {
            return Ok(None);
        };
        if self.program().get(&impl_name).is_none() {
            return Ok(None);
        }
        let id = self.alloc_insert(state.clone());
        let receiver = Value::Ref(RefTarget::Alloc {
            id,
            projection: Vec::new(),
        });
        let item = self.call_sync(frames, &impl_name, vec![receiver])?;
        let new_state = self.alloc_value(id).unwrap_or(Value::Uninit);
        Ok(Some((new_state, item)))
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

    /// `FlattenCompat::size_hint` replayed on a flatten
    /// composition's modelled state, for the `FlatMap` /
    /// `Flatten` / `FlattenCompat` arm of
    /// [`Self::iterator_size_hint`]. Core computes the hint
    /// from the front and back inner iterators' hints and
    /// the outer iterator's hint — read through the `Fuse`
    /// the outer iterator is parked in, where a blown fuse
    /// (the slot cleared to `None`) reports `(0, Some(0))`
    /// — scaled by the outer item's fixed inner size when
    /// it has one: an array is the one into-iterable whose
    /// length core knows statically. An inner iterator of
    /// core's one-shot family (it yields at most one item)
    /// selects the specialised impl instead, which ignores
    /// the front and back iterators entirely. `Ok(None)`
    /// when the composition's legs are not derivable from
    /// the tag, leaving the caller's fallback in place.
    fn flatten_size_hint(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        state: &Value,
    ) -> InterpResult<Option<(u128, Option<u128>)>> {
        let Some((outer_ty, item_ty, inner_ty)) = flatten_step_types(self_ty)
        else {
            return Ok(None);
        };
        let compat = if self_ty
            .starts_with("std::iter::adapters::flatten::FlattenCompat<")
        {
            state.clone()
        } else {
            let fields = iterator_fields(state, self_ty)?;
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    format!("{self_ty} state"),
                    "a one-field wrapper around a FlattenCompat",
                    format!("{} field(s)", fields.len()),
                ));
            }
            fields.into_iter().next().expect("one field checked")
        };
        let fields = iterator_fields(&compat, "FlattenCompat state")?;
        if fields.len() != 3 {
            return Err(InterpError::type_mismatch(
                "FlattenCompat state",
                "[fuse, front inner iterator, back inner iterator]",
                format!("{} field(s)", fields.len()),
            ));
        }
        let fuse_fields = iterator_fields(&fields[0], "Fuse state")?;
        if fuse_fields.len() != 1 {
            return Err(InterpError::type_mismatch(
                "Fuse state",
                "a one-field wrapper around an Option",
                format!("{} field(s)", fuse_fields.len()),
            ));
        }
        let outer_hint = match option_payload_opt(fuse_fields[0].clone()) {
            Some(outer_state) => {
                self.size_hint_parts(frames, &outer_ty, outer_state)?
            }
            None => (0, Some(0)),
        };
        if is_one_shot_ty(&inner_ty) {
            // The specialised impl: the outer hint, scaled
            // by the item's fixed size when it has one —
            // which for a one-shot inner iterator can only
            // be 0 or 1.
            return Ok(Some(match const_size_of_item(&item_ty) {
                Some(0) => (0, Some(0)),
                Some(1) => outer_hint,
                _ => (0, outer_hint.1),
            }));
        }
        let (flo, fhi) = match option_payload_opt(fields[1].clone()) {
            Some(inner_state) => {
                self.size_hint_parts(frames, &inner_ty, inner_state)?
            }
            None => (0, Some(0)),
        };
        let (blo, bhi) = match option_payload_opt(fields[2].clone()) {
            Some(inner_state) => {
                self.size_hint_parts(frames, &inner_ty, inner_state)?
            }
            None => (0, Some(0)),
        };
        let lo = flo.saturating_add(blo);
        if let Some(fixed) = const_size_of_item(&item_ty) {
            // Every remaining outer item expands to exactly
            // `fixed` inner items: the lower bound scales
            // outright, and the upper bound survives only
            // when every leg's is known — checked
            // arithmetic, as core's.
            let fixed = fixed as u128;
            let lower = outer_hint.0.saturating_mul(fixed).saturating_add(lo);
            let upper = match (fhi, bhi, outer_hint.1) {
                (Some(front), Some(back), Some(outer)) => {
                    match fixed.checked_mul(outer) {
                        Some(scaled) => front
                            .checked_add(back)
                            .and_then(|sum| sum.checked_add(scaled)),
                        None => None,
                    }
                }
                _ => None,
            };
            return Ok(Some((lower, upper)));
        }
        // The upper bound is known only when the outer
        // iterator is certainly spent: what remains is then
        // exactly the front and back iterators' contents.
        let pair = match (outer_hint, fhi, bhi) {
            ((0, Some(0)), Some(front), Some(back)) => {
                (lo, front.checked_add(back))
            }
            _ => (lo, None),
        };
        Ok(Some(pair))
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
        if is_char_escape_ty(self_ty) {
            // The escape iterators' `len` is the `alive`
            // range's length, the accounting their steps and
            // size hints share.
            let (start, end) = escape_inner_range(&state, self_ty)?;
            return Ok(Value::Uint(u128::from(end.saturating_sub(start))));
        }
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
    /// The elements of a slice adapter state's slice
    /// field under the element type the adapter's own type
    /// names, composed from the storage's bytes when the
    /// slice's formation regrouped them (see
    /// `regrouped_iter_elements`); the plain element
    /// reading otherwise.
    pub(super) fn adapter_slice_elements(
        &self,
        frames: &[Frame],
        self_ty: &str,
        slice_field: &Value,
    ) -> InterpResult<(Vec<Value>, u128)> {
        if let Some(elem_ty) = adapter_element_tag(self_ty).and_then(ty_from_tag) {
            if let Some(elems) =
                self.regrouped_iter_elements(frames, slice_field, &elem_ty)
            {
                let len = elems.len() as u128;
                return Ok((elems, len));
            }
        }
        self.slice_elements(frames, slice_field)
    }

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
        if is_char_escape_ty(self_ty) {
            // Exact size, as core's impls report it: the
            // remaining count is the `alive` range's length.
            let (start, end) = escape_inner_range(&state, self_ty)?;
            let n = u128::from(end.saturating_sub(start));
            return Ok(hint(n, Some(n)));
        }
        if is_str_escape_ty(self_ty) {
            // The `str` escape iterators forward `size_hint`
            // to the wrapped `FlatMap` composition, exactly
            // as their step forwards to it (see
            // `str_escape_flatmap_ty`): core's impls are
            // `self.inner.size_hint()`.
            if let Some(flatmap_ty) = Self::str_escape_flatmap_ty(self_ty) {
                let fields = iterator_fields(&state, self_ty)?;
                if fields.len() != 1 {
                    return Err(InterpError::type_mismatch(
                        "str escape iterator state",
                        "[flat map]",
                        format!("{} field(s)", fields.len()),
                    ));
                }
                let (lower, upper) =
                    self.size_hint_parts(frames, flatmap_ty, fields[0].clone())?;
                return Ok(hint(lower, upper));
            }
        }
        if self_ty.starts_with("std::sync::mpsc::Iter<")
            || self_ty.starts_with("std::sync::mpsc::IntoIter<")
            || self_ty.starts_with("std::sync::mpsc::TryIter<")
        {
            // Not overridden in std: the trait default.
            return Ok(hint(0, None));
        }
        if self_ty_is(self_ty, "std::env::Args")
            || self_ty_is(self_ty, "std::env::ArgsOs")
            || self_ty_is(self_ty, "std::env::Vars")
            || self_ty_is(self_ty, "std::env::VarsOs")
        {
            // Delegates to the inner `vec::IntoIter`'s exact count.
            let depth = if self_ty_is(self_ty, "std::env::Args")
                || self_ty_is(self_ty, "std::env::Vars")
            {
                3
            } else {
                2
            };
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
        if self_ty.starts_with("std::ops::RangeFrom<")
            || self_ty.starts_with("std::range::RangeFromIter<")
        {
            // Unbounded: core's `size_hint` is
            // `(usize::MAX, None)` for both the legacy
            // `RangeFrom` and the new `RangeFromIter`.
            return Ok(hint(u128::from(u64::MAX), None));
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
        if self_ty.starts_with("std::iter::Map<") {
            // `Map::size_hint` is its inner iterator's,
            // unchanged: the closure transforms each item
            // but neither adds nor drops any.
            let fields = iterator_fields(&state, self_ty)?;
            if fields.len() != 2 {
                return Err(InterpError::type_mismatch(
                    "Map state",
                    "[iterator, closure]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            let inner_ty = first_generic_arg(self_ty).unwrap_or("");
            let (lower, upper) =
                self.size_hint_parts(frames, inner_ty, fields[0].clone())?;
            return Ok(hint(lower, upper));
        }
        if self_ty.starts_with("std::iter::FlatMap<")
            || self_ty.starts_with("std::iter::Flatten<")
            || self_ty
                .starts_with("std::iter::adapters::flatten::FlattenCompat<")
        {
            // `FlatMap::size_hint` and `Flatten::size_hint`
            // forward to the wrapped `FlattenCompat`'s,
            // whose formula `flatten_size_hint` replays.
            if let Some((lower, upper)) =
                self.flatten_size_hint(frames, self_ty, &state)?
            {
                return Ok(hint(lower, upper));
            }
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

/// Whether this type tag names an iterator the modelled
/// consumers cannot drain -- a hash-collection iterator or
/// one of the concrete collection iterators (see
/// `hash_iterator_tag` / `collection_iterator_tag`) --
/// possibly wrapped in adapter types (`Map`, `Cloned`,
/// `Filter`, ...): an adapter's step composes its inner
/// iterator's step, so an adapter over a body-served iterator
/// is itself body-served, and the collected body chain is the
/// only faithful execution for the consumers that drain it.
/// Judged recursively over the tag's generic arguments; each
/// argument is a strictly smaller substring of the tag, so
/// the recursion terminates.
pub fn body_served_iterator_tag(self_ty: &str) -> bool {
    if hash_iterator_tag(self_ty) || collection_iterator_tag(self_ty) {
        return true;
    }
    generic_args_of(self_ty)
        .iter()
        .any(|arg| body_served_iterator_tag(arg))
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

/// Whether this type tag names a B-tree collection
/// (`BTreeMap` / `BTreeSet`): the `collect` / `from_iter`
/// consumer builtins can only build the target shapes they
/// model natively -- a sorted tree is not one of them --
/// while the collected instance bodies (collect into a
/// `Vec`, sort, then the bulk bottom-up build) execute
/// faithfully for any iterator, modelled or not, the
/// arrangement the hash collections use (see
/// `hash_collection_tag`).
pub fn btree_collection_tag(self_ty: &str) -> bool {
    let inner = self_ty
        .strip_prefix("&mut ")
        .or_else(|| self_ty.strip_prefix('&'))
        .unwrap_or(self_ty);
    inner.starts_with("std::collections::BTreeMap<")
        || inner.starts_with("std::collections::BTreeSet<")
        || inner.starts_with("alloc::collections::BTreeMap<")
        || inner.starts_with("alloc::collections::BTreeSet<")
}

/// Whether this type tag names a B-tree sorted-build
/// iterator (`DedupSortedIter`, the deduplicating wrapper
/// `BTreeMap::from_iter` feeds its bulk bottom-up build):
/// its `next` steps a one-element lookahead state the
/// modelled state machines do not reproduce -- the
/// collected impl body is the faithful execution, the
/// arrangement the hash iterators use (see
/// `hash_iterator_tag`).
pub fn btree_iterator_tag(self_ty: &str) -> bool {
    let bare = self_ty
        .strip_prefix("&mut ")
        .or_else(|| self_ty.strip_prefix('&'))
        .unwrap_or(self_ty);
    bare.starts_with("alloc::collections::btree::dedup_sorted_iter::DedupSortedIter<")
        || bare.starts_with("std::collections::btree::dedup_sorted_iter::DedupSortedIter<")
}

/// Whether this type is one of the concrete collection
/// iterators whose `next` the modelled state machine does
/// not step -- the `LinkedList` / `VecDeque` / `BinaryHeap`
/// iterator forms -- so their collected bodies are the
/// faithful execution, the arrangement the hash iterators
/// use (see `hash_iterator_tag`). A reference-wrapped tag
/// names the blanket `&mut I` / `&I` impl, whose step is
/// the pointee's, so the prefix is stripped before judging,
/// as for the hash tags.
pub fn collection_iterator_tag(self_ty: &str) -> bool {
    let bare = self_ty
        .strip_prefix("&mut ")
        .or_else(|| self_ty.strip_prefix('&'))
        .unwrap_or(self_ty);
    [
        "std::collections::linked_list::Iter<",
        "std::collections::linked_list::IterMut<",
        "std::collections::linked_list::IntoIter<",
        "alloc::collections::linked_list::Iter<",
        "alloc::collections::linked_list::IterMut<",
        "alloc::collections::linked_list::IntoIter<",
        "std::collections::vec_deque::Iter<",
        "std::collections::vec_deque::IterMut<",
        "std::collections::vec_deque::IntoIter<",
        "alloc::collections::vec_deque::Iter<",
        "alloc::collections::vec_deque::IterMut<",
        "alloc::collections::vec_deque::IntoIter<",
        "std::collections::binary_heap::Iter<",
        "std::collections::binary_heap::IntoIter<",
    ]
    .iter()
    .any(|prefix| bare.starts_with(prefix))
}

/// Whether this type is one of the concrete collection
/// containers (`LinkedList` / `VecDeque` / `BinaryHeap`)
/// whose by-value `IntoIterator` conversion builds a
/// real-layout owning iterator the builtin does not model,
/// so the collected body is the faithful execution (the
/// hash containers' arrangement, see `hash_collection_tag`).
pub fn collection_container_tag(self_ty: &str) -> bool {
    [
        "std::collections::LinkedList<",
        "alloc::collections::LinkedList<",
        "std::collections::VecDeque<",
        "alloc::collections::VecDeque<",
        "std::collections::BinaryHeap<",
    ]
    .iter()
    .any(|prefix| self_ty.starts_with(prefix))
}

/// Whether the pointee of a reference `Self` type is a
/// container whose own reference `IntoIterator` impl yields
/// a distinct iterator type (`&mut VecDeque` yields
/// `vec_deque::IterMut`, and so on), as opposed to an
/// iterator type itself, for which the blanket
/// `IntoIterator for &mut I` / `&I` impl is the identity.
pub(super) fn ref_pointee_is_container(inner: &str) -> bool {
    inner.starts_with('[')
        || [
            "std::vec::Vec<",
            "alloc::vec::Vec<",
            "std::collections::VecDeque<",
            "alloc::collections::VecDeque<",
            "std::collections::HashMap<",
            "std::collections::HashSet<",
            "std::collections::BTreeMap<",
            "std::collections::BTreeSet<",
            "std::collections::LinkedList<",
            "alloc::collections::LinkedList<",
            "std::collections::BinaryHeap<",
            "std::string::String",
            "alloc::string::String",
            "std::boxed::Box<",
            "alloc::boxed::Box<",
            "std::option::Option<",
            "core::option::Option<",
        ]
        .iter()
        .any(|prefix| inner.starts_with(prefix))
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
    is_char_escape_ty(self_ty)
        || self_ty.starts_with("std::slice::Iter<")
        || self_ty.starts_with("std::slice::IterMut<")
        || self_ty.starts_with("std::collections::btree_set::Iter<")
        || self_ty.starts_with("std::collections::btree_map::Iter<")
        || self_ty.starts_with("std::collections::btree_map::Keys<")
        || self_ty.starts_with("std::collections::btree_map::Values<")
}

pub fn iterator_next_modelled(self_ty: &str) -> bool {
    self_ty.starts_with("std::ops::Range<")
        || self_ty.starts_with("std::ops::RangeInclusive<")
        || self_ty.starts_with("std::ops::RangeFrom<")
        || self_ty.starts_with("std::range::RangeFromIter<")
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
        || self_ty_is(self_ty, "std::env::Vars")
        || self_ty_is(self_ty, "std::env::VarsOs")
        || self_ty_is(self_ty, "std::fs::ReadDir")
        || self_ty_is(self_ty, "std::env::SplitPaths")
        || self_ty.starts_with("std::slice::Windows<")
        || self_ty.starts_with("std::slice::Chunks<")
        || self_ty.starts_with("std::slice::RChunks<")
        || self_ty.starts_with("std::slice::ChunksExact<")
        || self_ty.starts_with("std::slice::RChunksExact<")
        || self_ty.starts_with("std::slice::ChunksMut<")
        || self_ty.starts_with("std::slice::RChunksMut<")
        || self_ty.starts_with("std::slice::ChunksExactMut<")
        || self_ty.starts_with("std::slice::RChunksExactMut<")
        || self_ty.starts_with("std::iter::Empty<")
        || self_ty.starts_with("std::sync::mpsc::Iter<")
        || self_ty.starts_with("std::sync::mpsc::IntoIter<")
        || self_ty.starts_with("std::sync::mpsc::TryIter<")
        || self_ty.starts_with("std::str::Split<")
        || self_ty.starts_with("std::str::RSplit<")
        || self_ty.starts_with("std::str::SplitTerminator<")
        || self_ty.starts_with("std::str::RSplitTerminator<")
        || self_ty.starts_with("std::str::SplitInclusive<")
        || self_ty.starts_with("std::str::SplitN<")
        || self_ty.starts_with("std::str::RSplitN<")
        || self_ty.starts_with("std::str::Matches<")
        || self_ty.starts_with("std::str::MatchIndices<")
        || self_ty_is(self_ty, "std::str::Chars")
        || self_ty_is(self_ty, "core::str::Chars")
        || self_ty_is(self_ty, "std::str::Bytes")
        || self_ty_is(self_ty, "core::str::Bytes")
        || is_char_escape_ty(self_ty)
        || is_str_escape_ty(self_ty)
}

/// The outcome of one in-place step of an `Option<I>`
/// iterator slot (see `Interpreter::option_iter_step_at`).
enum SlotStep {
    /// The slot held an iterator and the step yielded an item.
    Yielded(Value),
    /// The slot held an iterator, the step exhausted it, and
    /// the slot now stores `None`.
    Exhausted,
    /// The slot already stored `None`.
    Empty,
}

/// Whether a rendered tag names an iterator of core's
/// one-shot family — the iterators that yield at most one
/// item, for which `FlattenCompat` selects its specialised
/// `size_hint`: the one-item and empty sources (`Once`,
/// `OnceWith`, `Empty`, the `Option` and `Result`
/// iterators, a zero- or one-element array's `IntoIter`)
/// and the adapters that never increase the count
/// (`Cloned`, `Copied`, `Filter`, `FilterMap`, `Map`) over
/// one, which the check follows through the tag. A `&mut`
/// iterator passes the property through, as core's
/// blanket impl does.
fn is_one_shot_ty(ty: &str) -> bool {
    let ty = ty.strip_prefix("&mut ").unwrap_or(ty);
    if ty.starts_with("std::iter::Once<")
        || ty.starts_with("std::iter::OnceWith<")
        || ty.starts_with("std::iter::Empty<")
        || ty.starts_with("std::option::IntoIter<")
        || ty.starts_with("core::option::IntoIter<")
        || ty.starts_with("std::option::Iter<")
        || ty.starts_with("core::option::Iter<")
        || ty.starts_with("std::option::IterMut<")
        || ty.starts_with("core::option::IterMut<")
        || ty.starts_with("std::result::IntoIter<")
        || ty.starts_with("core::result::IntoIter<")
        || ty.starts_with("std::result::Iter<")
        || ty.starts_with("core::result::Iter<")
        || ty.starts_with("std::result::IterMut<")
        || ty.starts_with("core::result::IterMut<")
    {
        return true;
    }
    if ty.starts_with("std::array::IntoIter<") {
        // One-shot only for a zero- or one-element array.
        let args = generic_args_of(ty);
        return args.len() >= 2 && matches!(args[1].trim(), "0" | "1");
    }
    if ty.starts_with("std::iter::Cloned<")
        || ty.starts_with("std::iter::Copied<")
        || ty.starts_with("std::iter::Filter<")
        || ty.starts_with("std::iter::FilterMap<")
        || ty.starts_with("std::iter::Map<")
    {
        if let Some(inner) = first_generic_arg(ty) {
            return is_one_shot_ty(inner);
        }
    }
    false
}

/// The fixed length core's `ConstSizeIntoIterator`
/// reports for an into-iterable tag: an array's length,
/// the one statically known inner size (the impls cover
/// `[T; N]` and its reference forms). `None` for every
/// other type — and for an array whose length the tag
/// does not render as a plain number.
fn const_size_of_item(ty: &str) -> Option<usize> {
    let ty = ty.strip_prefix("&mut ").unwrap_or(ty);
    let ty = ty.strip_prefix('&').unwrap_or(ty);
    if !ty.starts_with('[') || !ty.ends_with(']') {
        return None;
    }
    let semi = ty.rfind(';')?;
    ty[semi + 1..ty.len() - 1].trim().parse().ok()
}

/// The type triple a flatten composition's step machine runs
/// on, derived from the composition's rendered tag: the outer
/// iterator's type, the outer item's type (the into-iterable
/// the step converts as the outer yields it), and the inner
/// iterator's type (the item's own `IntoIterator::IntoIter`).
/// `None` when any leg is not derivable from the tag — the
/// step then declines and the composition keeps the verdict
/// it had. The derivations are type-level only:
///
/// - `FlatMap<I, U, F>` wraps `FlattenCompat<Map<I, F>,
///   U::IntoIter>`: the outer and item legs are the tag's own
///   arguments, and the inner leg is `U`'s into-iterator tag.
/// - `Flatten<I>` wraps `FlattenCompat<I, I::Item::IntoIter>`:
///   the item leg is `I`'s item type (see `iterator_item_ty`).
/// - `FlattenCompat<I, U::IntoIter>` carries the outer and
///   inner legs as its arguments; the item leg is `I`'s item
///   type.
fn flatten_step_types(self_ty: &str) -> Option<(String, String, String)> {
    if self_ty.starts_with("std::iter::FlatMap<") {
        let args = generic_args_of(self_ty);
        if args.len() < 3 {
            return None;
        }
        let inner = into_iter_tag(&args[1])?;
        return Some((
            format!("std::iter::Map<{}, {}>", args[0], args[2]),
            args[1].clone(),
            inner,
        ));
    }
    if self_ty.starts_with("std::iter::Flatten<") {
        let outer = first_generic_arg(self_ty)?.to_string();
        let item = iterator_item_ty(&outer)?;
        let inner = into_iter_tag(&item)?;
        return Some((outer, item, inner));
    }
    if self_ty.starts_with("std::iter::adapters::flatten::FlattenCompat<") {
        let args = generic_args_of(self_ty);
        if args.len() < 2 {
            return None;
        }
        let item = iterator_item_ty(&args[0])?;
        return Some((args[0].clone(), item, args[1].clone()));
    }
    None
}

/// The `IntoIterator::IntoIter` type tag for a type tag, where
/// the mapping is a type-level identity or renaming — the tag
/// counterpart of the value transformation `into_iter_value`
/// performs (see dispatch.rs), covering the same arms: a type
/// that is its own iterator iterates under its own tag, the
/// by-value containers rename to their module's `IntoIter`,
/// and the reference forms name the reference iterator the
/// conversion builds. `None` for a type whose conversion the
/// value transformation does not serve.
fn into_iter_tag(ty: &str) -> Option<String> {
    if let Some(inner) = ty.strip_prefix("&mut ") {
        if inner.starts_with("std::option::Option<")
            || inner.starts_with("core::option::Option<")
        {
            let elem = first_generic_arg(inner)?;
            return Some(format!("std::option::IterMut<{elem}>"));
        }
        if ref_pointee_is_container(inner) {
            if inner.starts_with("std::vec::Vec<")
                || inner.starts_with("alloc::vec::Vec<")
                || inner.starts_with('[')
            {
                let elem = if inner.starts_with('[') {
                    slice_container_element_tag(inner)?
                } else {
                    first_generic_arg(inner)?
                };
                return Some(format!("std::slice::IterMut<{elem}>"));
            }
            return None;
        }
        // The blanket `IntoIterator for &mut I`: the reference
        // itself is the iterator.
        return Some(ty.to_string());
    }
    if let Some(inner) = ty.strip_prefix('&') {
        if inner.starts_with("std::option::Option<")
            || inner.starts_with("core::option::Option<")
        {
            let elem = first_generic_arg(inner)?;
            return Some(format!("std::option::Iter<{elem}>"));
        }
        if inner.starts_with("std::vec::Vec<")
            || inner.starts_with("alloc::vec::Vec<")
            || inner.starts_with('[')
        {
            let elem = if inner.starts_with('[') {
                slice_container_element_tag(inner)?
            } else {
                first_generic_arg(inner)?
            };
            return Some(format!("std::slice::Iter<{elem}>"));
        }
        if inner.starts_with("std::collections::BTreeSet<") {
            let elem = first_generic_arg(inner)?;
            return Some(format!("std::slice::Iter<{elem}>"));
        }
        if inner.starts_with("std::collections::BTreeMap<") {
            let args = generic_args_of(inner);
            if args.len() < 2 {
                return None;
            }
            return Some(format!("std::slice::Iter<({}, {})>", args[0], args[1]));
        }
        if ref_pointee_is_container(inner) {
            return None;
        }
        // The blanket `IntoIterator for &I`.
        return Some(ty.to_string());
    }
    if into_iter_identity(ty) {
        return Some(ty.to_string());
    }
    if ty.starts_with("std::vec::Vec<") || ty.starts_with("alloc::vec::Vec<") {
        return Some(ty.replacen("Vec<", "IntoIter<", 1));
    }
    if ty.starts_with("std::collections::BTreeSet<") {
        return Some(ty.replacen("BTreeSet<", "IntoIter<", 1));
    }
    if ty.starts_with("std::collections::BTreeMap<") {
        return Some(ty.replacen("BTreeMap<", "IntoIter<", 1));
    }
    if ty.starts_with("std::option::Option<") || ty.starts_with("core::option::Option<") {
        let elem = first_generic_arg(ty)?;
        return Some(format!("std::option::IntoIter<{elem}>"));
    }
    if ty.starts_with("std::sync::mpsc::Receiver<") {
        let elem = first_generic_arg(ty)?;
        return Some(format!("std::sync::mpsc::IntoIter<{elem}>"));
    }
    if ty.starts_with('[') && ty.ends_with(']') {
        let inner = &ty[1..ty.len() - 1];
        let (elem, len) = inner.rsplit_once(';')?;
        return Some(format!(
            "std::array::IntoIter<{}, {}>",
            elem.trim(),
            len.trim()
        ));
    }
    None
}

/// Whether `IntoIterator::into_iter` is the identity for this
/// type tag: the type is its own iterator (the blanket
/// `IntoIterator for I: Iterator` impl), so the value already
/// is the iterator state. The type-level mirror of the
/// identity condition in `into_iter_value` (dispatch.rs).
fn into_iter_identity(ty: &str) -> bool {
    ty.starts_with("std::ops::Range<")
        || ty.starts_with("std::ops::RangeInclusive<")
        || ty.starts_with("std::iter::Rev<")
        || ty.starts_with("std::slice::Iter<")
        || ty.starts_with("std::collections::btree_set::Iter<")
        || ty.starts_with("std::collections::btree_map::Iter<")
        || ty.starts_with("std::collections::btree_map::Keys<")
        || ty.starts_with("std::collections::btree_map::Values<")
        || ty.starts_with("std::collections::btree_set::IntoIter<")
        || ty.starts_with("std::collections::btree_map::IntoIter<")
        || ty.starts_with("std::slice::IterMut<")
        || ty.starts_with("std::array::IntoIter<")
        || ty.starts_with("std::vec::IntoIter<")
        || ty.starts_with("alloc::vec::IntoIter<")
        || ty.starts_with("std::vec::Drain<")
        || ty.starts_with("alloc::vec::Drain<")
        || ty.starts_with("std::option::IntoIter<")
        || ty.starts_with("std::option::Iter<")
        || ty.starts_with("std::option::IterMut<")
        || ty.starts_with("std::iter::Map<")
        || ty.starts_with("std::iter::Enumerate<")
        || ty.starts_with("std::iter::Copied<")
        || ty.starts_with("std::iter::Cloned<")
        || ty.starts_with("std::iter::Take<")
        || ty.starts_with("std::iter::Skip<")
        || ty.starts_with("std::iter::Chain<")
        || ty.starts_with("std::iter::Zip<")
        || ty.starts_with("std::iter::SkipWhile<")
        || ty.starts_with("std::iter::Filter<")
        || ty.starts_with("std::iter::FilterMap<")
        || ty.starts_with("std::iter::FlatMap<")
        || ty.starts_with("std::iter::Flatten<")
        || ty.starts_with("std::iter::adapters::flatten::FlattenCompat<")
        || ty.starts_with("std::io::Lines<")
        || ty.starts_with("std::io::Bytes<")
        || self_ty_is(ty, "std::str::Lines")
        || self_ty_is(ty, "std::str::LinesInclusive")
        || ty.starts_with("std::str::Split<")
        || ty.starts_with("std::str::RSplit<")
        || ty.starts_with("std::str::SplitTerminator<")
        || ty.starts_with("std::str::RSplitTerminator<")
        || ty.starts_with("std::str::SplitInclusive<")
        || ty.starts_with("std::iter::Once<")
        || ty.starts_with("std::path::Components")
        || ty.starts_with("std::path::Ancestors")
        || self_ty_is(ty, "std::fs::ReadDir")
        || ty.starts_with("std::iter::Empty<")
        || ty.starts_with("std::iter::Repeat<")
        || self_ty_is(ty, "std::str::Chars")
        || self_ty_is(ty, "core::str::Chars")
        || self_ty_is(ty, "std::str::Bytes")
        || self_ty_is(ty, "core::str::Bytes")
        || self_ty_is(ty, "std::char::EscapeDebug")
        || self_ty_is(ty, "core::char::EscapeDebug")
        || self_ty_is(ty, "std::char::EscapeDefault")
        || self_ty_is(ty, "core::char::EscapeDefault")
        || self_ty_is(ty, "std::char::EscapeUnicode")
        || self_ty_is(ty, "core::char::EscapeUnicode")
        || ty.starts_with("std::collections::linked_list::Iter<")
        || ty.starts_with("std::collections::linked_list::IterMut<")
        || ty.starts_with("std::collections::linked_list::IntoIter<")
        || ty.starts_with("alloc::collections::linked_list::Iter<")
        || ty.starts_with("alloc::collections::linked_list::IterMut<")
        || ty.starts_with("alloc::collections::linked_list::IntoIter<")
        || ty.starts_with("std::collections::vec_deque::Iter<")
        || ty.starts_with("std::collections::vec_deque::IterMut<")
        || ty.starts_with("std::collections::vec_deque::IntoIter<")
        || ty.starts_with("alloc::collections::vec_deque::Iter<")
        || ty.starts_with("alloc::collections::vec_deque::IterMut<")
        || ty.starts_with("alloc::collections::vec_deque::IntoIter<")
        || ty.starts_with("std::collections::binary_heap::Iter<")
        || ty.starts_with("std::collections::binary_heap::IntoIter<")
        || ty.starts_with("std::slice::Chunks<")
        || ty.starts_with("std::slice::ChunksExact<")
        || ty.starts_with("std::slice::RChunks<")
        || ty.starts_with("std::slice::RChunksExact<")
        || ty.starts_with("std::slice::ChunksMut<")
        || ty.starts_with("std::slice::ChunksExactMut<")
        || ty.starts_with("std::slice::RChunksMut<")
        || ty.starts_with("std::slice::RChunksExactMut<")
        || self_ty_is(ty, "std::env::Args")
        || self_ty_is(ty, "std::env::ArgsOs")
        || self_ty_is(ty, "std::env::Vars")
        || self_ty_is(ty, "std::env::VarsOs")
        || self_ty_is(ty, "std::env::SplitPaths")
        || ty.starts_with("std::slice::Windows<")
        || ty.starts_with("std::sync::mpsc::Iter<")
        || ty.starts_with("std::sync::mpsc::IntoIter<")
        || ty.starts_with("std::sync::mpsc::TryIter<")
}

/// The element type of a `StepBy` inner iterator core's
/// specialised range implementation serves (`Range` and the
/// new `RangeIter` over the unsigned integers), or `None`
/// for the generic `StepBy` step.
fn step_by_spec_range_elem(inner_ty: &str) -> Option<&str> {
    for prefix in ["std::ops::Range<", "std::range::RangeIter<"] {
        if let Some(rest) = inner_ty.strip_prefix(prefix) {
            let elem = rest.strip_suffix('>')?;
            if matches!(elem, "u8" | "u16" | "u32" | "u64" | "usize") {
                return Some(elem);
            }
        }
    }
    None
}

/// The `Iterator::Item` type tag of an iterator type tag,
/// where it is a type-level function of the tag: the first
/// type argument of the source iterators that carry their
/// item there, a reference to it for the reference-yielding
/// sources, and the inner iterator's item through the
/// transparent one-iterator adapters. `None` where the item
/// is not a function of the tag (a `Map`'s item is its
/// closure's return type, which the tag does not carry).
fn iterator_item_ty(ty: &str) -> Option<String> {
    // Transparent one-iterator adapters: the item is the
    // inner iterator's item. (`Chain`'s two sides share one
    // item type, so the first side names it.)
    for prefix in [
        "std::iter::Rev<",
        "std::iter::Fuse<",
        "std::iter::Take<",
        "std::iter::Skip<",
        "std::iter::StepBy<",
        "std::iter::Filter<",
        "std::iter::SkipWhile<",
        "std::iter::TakeWhile<",
        "std::iter::Inspect<",
        "std::iter::Cycle<",
        "std::iter::Peekable<",
        "std::iter::Chain<",
    ] {
        if ty.starts_with(prefix) {
            return iterator_item_ty(first_generic_arg(ty)?);
        }
    }
    // `Copied` / `Cloned` over a reference iterator: the item
    // is the pointee type.
    if ty.starts_with("std::iter::Copied<") || ty.starts_with("std::iter::Cloned<") {
        let item = iterator_item_ty(first_generic_arg(ty)?)?;
        if let Some(rest) = item.strip_prefix("&mut ") {
            return Some(rest.to_string());
        }
        if let Some(rest) = item.strip_prefix('&') {
            return Some(rest.to_string());
        }
        return None;
    }
    // Source iterators whose item is the first type argument.
    for prefix in [
        "std::vec::IntoIter<",
        "alloc::vec::IntoIter<",
        "std::array::IntoIter<",
        "std::collections::vec_deque::IntoIter<",
        "alloc::collections::vec_deque::IntoIter<",
        "std::collections::linked_list::IntoIter<",
        "alloc::collections::linked_list::IntoIter<",
        "std::collections::binary_heap::IntoIter<",
        "std::collections::btree_set::IntoIter<",
        "std::collections::hash_set::IntoIter<",
        "std::option::IntoIter<",
        "std::sync::mpsc::IntoIter<",
        "std::iter::Once<",
        "std::iter::Empty<",
        "std::iter::Repeat<",
        "std::ops::Range<",
        "std::ops::RangeInclusive<",
        "std::ops::RangeFrom<",
        "std::range::RangeFromIter<",
    ] {
        if ty.starts_with(prefix) {
            return Some(first_generic_arg(ty)?.to_string());
        }
    }
    // Reference-yielding sources: the item is a reference to
    // the first type argument.
    for prefix in [
        "std::slice::Iter<",
        "std::collections::btree_set::Iter<",
        "std::option::Iter<",
    ] {
        if ty.starts_with(prefix) {
            return Some(format!("&{}", first_generic_arg(ty)?));
        }
    }
    for prefix in ["std::slice::IterMut<", "std::option::IterMut<"] {
        if ty.starts_with(prefix) {
            return Some(format!("&mut {}", first_generic_arg(ty)?));
        }
    }
    // Mutable slice-yielding sources: the item is a mutable
    // slice reference.
    for prefix in [
        "std::slice::ChunksMut<",
        "std::slice::RChunksMut<",
        "std::slice::ChunksExactMut<",
        "std::slice::RChunksExactMut<",
    ] {
        if ty.starts_with(prefix) {
            return Some(format!("&mut [{}]", first_generic_arg(ty)?));
        }
    }
    // Slice-yielding sources: the item is a slice reference.
    for prefix in [
        "std::slice::Chunks<",
        "std::slice::RChunks<",
        "std::slice::ChunksExact<",
        "std::slice::RChunksExact<",
        "std::slice::Windows<",
    ] {
        if ty.starts_with(prefix) {
            return Some(format!("&[{}]", first_generic_arg(ty)?));
        }
    }
    if self_ty_is(ty, "std::str::Chars") || self_ty_is(ty, "core::str::Chars") {
        return Some("char".to_string());
    }
    if self_ty_is(ty, "std::str::Bytes") || self_ty_is(ty, "core::str::Bytes") {
        return Some("u8".to_string());
    }
    None
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
        || self_ty.starts_with("std::slice::Chunks<")
        || self_ty.starts_with("std::slice::RChunks<")
        || self_ty.starts_with("std::slice::ChunksExact<")
        || self_ty.starts_with("std::slice::RChunksExact<")
        || self_ty.starts_with("std::slice::ChunksMut<")
        || self_ty.starts_with("std::slice::RChunksMut<")
        || self_ty.starts_with("std::slice::ChunksExactMut<")
        || self_ty.starts_with("std::slice::RChunksExactMut<")
        || self_ty.starts_with("std::option::IntoIter<")
        || self_ty.starts_with("std::option::Iter<")
        || self_ty.starts_with("std::option::IterMut<")
        || self_ty_is(self_ty, "std::env::Args")
        || self_ty_is(self_ty, "std::env::ArgsOs")
        || self_ty_is(self_ty, "std::env::Vars")
        || self_ty_is(self_ty, "std::env::VarsOs")
        || self_ty.starts_with("std::iter::Empty<")
        || self_ty.starts_with("std::path::Components")
        || self_ty.starts_with("std::iter::Once<")
        || self_ty.starts_with("std::str::Split<")
        || self_ty.starts_with("std::str::RSplit<")
        || self_ty.starts_with("std::str::SplitTerminator<")
        || self_ty.starts_with("std::str::RSplitTerminator<")
        || self_ty.starts_with("std::str::SplitInclusive<")
        || self_ty.starts_with("std::str::Matches<")
        || self_ty.starts_with("std::str::MatchIndices<")
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
/// The chunk size a chunk iterator state stores: a plain
/// integer, or a `NonZero` size wrapped in one-field
/// aggregates (the shape core's `chunks` constructors store).
fn chunk_size_of(f: &Value, self_ty: &str) -> InterpResult<usize> {
    let mut cur = f;
    loop {
        match cur {
            Value::Uint(v) => return Ok(*v as usize),
            Value::Int(v) if *v >= 0 => return Ok(*v as usize),
            Value::Enum { fields, .. } if fields.len() == 1 => {
                cur = &fields[0];
            }
            Value::Tuple(fields) if fields.len() == 1 => {
                cur = &fields[0];
            }
            other => {
                return Err(InterpError::type_mismatch(
                    format!("chunk size of {self_ty}"),
                    "non-zero integer",
                    other.kind_name(),
                ))
            }
        }
    }
}

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

/// One `Vars::next` item conversion: the stored
/// `(OsString, OsString)` pair becomes the `(String, String)`
/// pair `Vars` yields, through the same
/// `OsString::into_string` the native body unwraps -- invalid
/// Unicode is the native unwrap panic.
fn vars_pair_to_strings(item: Value) -> InterpResult<Value> {
    let fields = match &item {
        Value::Tuple(fields) if fields.len() == 2 => fields.clone(),
        other => {
            return Err(InterpError::type_mismatch(
                "Vars item",
                "(OsString, OsString) pair",
                other.kind_name(),
            ))
        }
    };
    let mut strings = Vec::new();
    for side in &fields {
        let bytes = os_string_bytes(side).ok_or_else(|| {
            InterpError::type_mismatch("Vars item", "OsString value", side.kind_name())
        })?;
        match String::from_utf8(bytes) {
            Ok(text) => strings.push(Value::Str(text)),
            Err(error) => {
                return Err(InterpError::Panic {
                    message: format!(
                        "called `Result::unwrap()` on an `Err` value: {error:?}"
                    ),
                    payload: None,
                })
            }
        }
    }
    Ok(Value::Tuple(strings))
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

/// One `next` step of a `vec::IntoIter` in its real field
/// layout ([buf, phantom, cap, alloc, ptr, end], the fields
/// of `alloc::vec::into_iter::IntoIter`): the layout an
/// inlined `Vec::into_iter` construction inside a collected
/// body builds, where the modelled [items, next index]
/// state the arm in `next_on_state` steps is the form the
/// `IntoIterator` builtin builds instead. The step reads
/// the element at the current pointer and advances the
/// pointer by one element; the exhausted pair answers
/// `None` with the state unchanged. The consumed slot is
/// not cleared: every later reader of the iterator (the
/// remaining steps, its drop glue, `as_slice`) reads the
/// `ptr..end` range, never behind the current pointer.
/// `None` when the fields are not the real layout's
/// readable pointer pair (an address-only buffer, a
/// zero-sized element type), leaving the caller its shape
/// error.
fn vec_into_iter_step_real(
    interp: &Interpreter,
    frames: &mut Vec<Frame>,
    self_ty: &str,
    fields: &[Value],
) -> Option<InterpResult<(Vec<Value>, Value)>> {
    if fields.len() != 6 {
        return None;
    }
    let ptr_field = &fields[4];
    let ptr_ref = iter_ptr_value(ptr_field)?;
    let end_ref = iter_ptr_value(&fields[5])?;
    if ptr_ref == end_ref {
        return Some(Ok((fields.to_vec(), none_value())));
    }
    let elem_tag = first_generic_arg(self_ty)?;
    let (size, _) = interp.layout_of(elem_tag)?;
    if size == 0 {
        return None;
    }
    let size = u64::try_from(size).ok()?;
    let cur = iter_ptr_index(&ptr_ref, Some(size))?;
    let last = iter_ptr_index(&end_ref, Some(size))?;
    if cur >= last {
        return Some(Ok((fields.to_vec(), none_value())));
    }
    let target = interp.deref_value(frames, &ptr_ref).ok()?;
    let item = interp.read_resolved(frames, &target).ok()?;
    let advanced = iter_ptr_with_index(ptr_field, cur + 1)?;
    let mut new_fields = fields.to_vec();
    new_fields[4] = advanced;
    Some(Ok((new_fields, some_value(item))))
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
/// The element type tag a slice adapter's rendered type
/// names after its lifetime argument
/// (`std::slice::Windows<'_, [u8; 2]>` -> `[u8; 2]`): the
/// last top-level generic argument. `None` when the tag
/// names no generic arguments.
fn adapter_element_tag(self_ty: &str) -> Option<&str> {
    let open = self_ty.find('<')?;
    let close = self_ty.rfind('>')?;
    split_top_level(&self_ty[open + 1..close]).into_iter().last()
}

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
            // The string the iterator was built over, held as
            // the value itself: the form the converted
            // `str::chars` / `str::bytes` construction leaves
            // (the byte-slice iterator over the string's own
            // storage), where the builtin construction stores
            // a snapshot of the bytes instead. The bytes are
            // the same bytes either way; the step's pointers
            // index into them identically.
            Value::Str(text) => text
                .as_bytes()
                .iter()
                .map(|b| Value::Uint(*b as u128))
                .collect(),
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

/// Whether a rendered tag names one of the `char` escape
/// iterators: `char::EscapeDebug`, `char::EscapeDefault` and
/// `char::EscapeUnicode` are one-field wrappers over core's
/// `EscapeIterInner` buffer state, whose step
/// `step_escape_inner` replays.
fn is_char_escape_ty(self_ty: &str) -> bool {
    self_ty_is(self_ty, "std::char::EscapeDebug")
        || self_ty_is(self_ty, "core::char::EscapeDebug")
        || self_ty_is(self_ty, "std::char::EscapeDefault")
        || self_ty_is(self_ty, "core::char::EscapeDefault")
        || self_ty_is(self_ty, "std::char::EscapeUnicode")
        || self_ty_is(self_ty, "core::char::EscapeUnicode")
}

/// Whether a rendered tag names one of the `str` escape
/// iterators: `str::EscapeDebug`, `str::EscapeDefault`
/// and `str::EscapeUnicode` are one-field wrappers over
/// a `FlatMap` of the string's `Chars` through the
/// matching `char` escape callable, whose step the
/// `str_escape_flatmap_ty` delegation replays.
fn is_str_escape_ty(self_ty: &str) -> bool {
    self_ty_is(self_ty, "std::str::EscapeDebug")
        || self_ty_is(self_ty, "core::str::EscapeDebug")
        || self_ty_is(self_ty, "std::str::EscapeDefault")
        || self_ty_is(self_ty, "core::str::EscapeDefault")
        || self_ty_is(self_ty, "std::str::EscapeUnicode")
        || self_ty_is(self_ty, "core::str::EscapeUnicode")
}

/// The bound past which an `EscapeIterInner` `alive` range
/// denotes the union payload's literal char rather than a
/// span of its escape-sequence buffer: core's
/// `LITERAL_ESCAPE_START`. A literal payload's range is
/// exactly the one slot at this bound (128..129); a buffer
/// payload's range indexes the buffer (at most 10 slots).
const ESCAPE_LITERAL_START: u8 = 128;

/// The `alive` range stored in an `EscapeIterInner` value:
/// a `Range<u8>` in its real two-field shape.
fn escape_range_of(alive: &Value, self_ty: &str) -> InterpResult<(u8, u8)> {
    let fields = iterator_fields(alive, self_ty)?;
    if fields.len() != 2 {
        return Err(InterpError::type_mismatch(
            format!("{self_ty} alive range"),
            "[start, end]",
            format!("{} field(s)", fields.len()),
        ));
    }
    let as_u8 = |v: &Value| -> InterpResult<u8> {
        match v {
            Value::Uint(n) => Ok(*n as u8),
            other => Err(InterpError::type_mismatch(
                format!("{self_ty} alive range bound"),
                "unsigned byte",
                other.kind_name(),
            )),
        }
    };
    Ok((as_u8(&fields[0])?, as_u8(&fields[1])?))
}

/// The fields of the `EscapeIterInner` value a `char`
/// escape iterator state wraps: [data, alive, escaping].
fn escape_inner_fields(state: &Value, self_ty: &str) -> InterpResult<Vec<Value>> {
    let outer = iterator_fields(state, self_ty)?;
    if outer.len() != 1 {
        return Err(InterpError::type_mismatch(
            format!("{self_ty} state"),
            "[EscapeIterInner]",
            format!("{} field(s)", outer.len()),
        ));
    }
    let inner = iterator_fields(&outer[0], self_ty)?;
    if inner.len() != 3 {
        return Err(InterpError::type_mismatch(
            format!("{self_ty} EscapeIterInner state"),
            "[data, alive, PhantomData]",
            format!("{} field(s)", inner.len()),
        ));
    }
    Ok(inner)
}

/// The `alive` range of the `EscapeIterInner` state a
/// `char` escape iterator value wraps.
fn escape_inner_range(state: &Value, self_ty: &str) -> InterpResult<(u8, u8)> {
    let inner = escape_inner_fields(state, self_ty)?;
    escape_range_of(&inner[1], self_ty)
}

/// The escape-sequence bytes of an `EscapeIterInner`
/// payload in its buffer form: the union's `escape_seq`
/// field is an array of `ascii::Char`, a fieldless byte
/// enum whose value is its discriminant (a plain integer
/// element is the same byte in storage form).
fn escape_seq_bytes(data: &Value, self_ty: &str) -> InterpResult<Vec<u8>> {
    let fields = iterator_fields(data, self_ty)?;
    let array = fields.first().ok_or_else(|| {
        InterpError::type_mismatch(
            format!("{self_ty} EscapeIterInner data"),
            "a union payload holding the escape-sequence array",
            "no fields",
        )
    })?;
    let items = match array {
        Value::Tuple(items) => items,
        other => {
            return Err(InterpError::type_mismatch(
                format!("{self_ty} escape-sequence buffer"),
                "an array of ascii::Char",
                other.kind_name(),
            ))
        }
    };
    items
        .iter()
        .map(|item| match item {
            Value::Enum { variant, .. } => Ok(*variant as u8),
            Value::Uint(v) => Ok(*v as u8),
            other => Err(InterpError::type_mismatch(
                format!("{self_ty} escape-sequence element"),
                "ascii::Char",
                other.kind_name(),
            )),
        })
        .collect()
}

/// The literal char of an `EscapeIterInner` payload in its
/// `MaybeEscaped` literal form: the union's `literal` field
/// is the one scalar slot the construction filled, beside
/// the untouched buffer slot.
fn escape_literal_char(data: &Value, self_ty: &str) -> InterpResult<char> {
    let fields = iterator_fields(data, self_ty)?;
    for field in &fields {
        if let Value::Uint(v) = field {
            if let Some(c) = char::from_u32(*v as u32) {
                return Ok(c);
            }
        }
    }
    Err(InterpError::type_mismatch(
        format!("{self_ty} EscapeIterInner data"),
        "a union payload holding a literal char",
        data.kind_name(),
    ))
}

/// One step of a `char` escape iterator's state, from the
/// front (`from_back == false`) or the back: core's
/// `EscapeIterInner` serves the char at the `alive` range's
/// taken end — the literal char when the range is the
/// literal slot, the buffer byte there otherwise — and
/// shrinks the range past it. The payload is carried
/// unchanged; only the range moves, exactly as the real
/// state transitions. An exhausted range yields `None` and
/// leaves the state as it stands.
fn step_escape_inner(
    state: &Value,
    self_ty: &str,
    from_back: bool,
) -> InterpResult<(Value, Option<Value>)> {
    let outer = iterator_fields(state, self_ty)?;
    if outer.len() != 1 {
        return Err(InterpError::type_mismatch(
            format!("{self_ty} state"),
            "[EscapeIterInner]",
            format!("{} field(s)", outer.len()),
        ));
    }
    let inner = escape_inner_fields(state, self_ty)?;
    let (start, end) = escape_range_of(&inner[1], self_ty)?;
    if start >= end {
        return Ok((state.clone(), None));
    }
    let item = if end > ESCAPE_LITERAL_START {
        escape_literal_char(&inner[0], self_ty)?
    } else {
        let bytes = escape_seq_bytes(&inner[0], self_ty)?;
        let idx = usize::from(if from_back { end - 1 } else { start });
        let byte = bytes.get(idx).copied().ok_or_else(|| {
            InterpError::type_mismatch(
                format!("{self_ty} escape-sequence buffer"),
                "a byte at the alive range's taken end",
                format!("{} byte(s)", bytes.len()),
            )
        })?;
        // The buffer holds printable ASCII by construction.
        char::from(byte)
    };
    let (new_start, new_end) = if from_back {
        (start, end - 1)
    } else {
        (start + 1, end)
    };
    let new_alive = rebuild_state(
        &inner[1],
        vec![
            Value::Uint(u128::from(new_start)),
            Value::Uint(u128::from(new_end)),
        ],
    );
    let mut new_inner = inner;
    new_inner[1] = new_alive;
    let rebuilt_inner = rebuild_state(&outer[0], new_inner);
    Ok((
        rebuild_state(state, vec![rebuilt_inner]),
        Some(Value::Uint(u128::from(item as u32))),
    ))
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

/// Whether a rendered tag names one of the `str` matches
/// iterators whose state machine `str_matches_step` replays
/// (`Matches`, `MatchIndices`).
fn is_str_matches_ty(self_ty: &str) -> bool {
    self_ty.starts_with("std::str::Matches<")
        || self_ty.starts_with("std::str::MatchIndices<")
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

/// Whether a rendered tag names one of the `str` splitn
/// iterators whose state machine `str_splitn_step` replays
/// (`SplitN`, `RSplitN` -- the `<` keeps `SplitN` from also
/// matching the plain split family `is_str_split_ty` names).
fn is_str_splitn_ty(self_ty: &str) -> bool {
    self_ty.starts_with("std::str::SplitN<")
        || self_ty.starts_with("std::str::RSplitN<")
}

/// How a `CharEqSearcher`'s char test is served: a guest
/// predicate applied through the modelled call path, a set
/// of chars tested by membership (resolved from the state),
/// or a `&[char]` pattern's slice, resolved through the
/// interpreter before the first step.
enum MatchesEq {
    Predicate(Value),
    Set(Vec<char>),
    SetRef(Value),
}

/// The item a matches step yields for one searcher match:
/// the matched text for `Matches`, the pair of its byte
/// index and the text for `MatchIndices`.
fn matches_item(
    haystack: &str,
    indices: bool,
    found: Option<(usize, usize)>,
) -> Value {
    match found {
        Some((a, b)) => {
            let piece = Value::Str(haystack[a..b].to_string());
            if indices {
                some_value(Value::Tuple(vec![Value::Uint(a as u128), piece]))
            } else {
                some_value(piece)
            }
        }
        None => none_value(),
    }
}

/// The decoded state of a `str` matches iterator over a
/// `char` pattern: the pattern's `CharSearcher` fields, in
/// the two one-field wrappers (`Matches` over
/// `MatchesInternal`) the constructor built. The fields the
/// machine does not read (the needle char, its size, the
/// encoding array) are carried as their original values so
/// the rebuilt state is field-exact, as in `StrSplitParts`.
struct StrMatchesParts {
    haystack: String,
    finger: usize,
    finger_back: usize,
    needle: Vec<u8>,
    needle_value: Value,
    utf8_size_value: Value,
    utf8_encoded_value: Value,
}

impl StrMatchesParts {
    /// Decode the state value, or `None` when it is not the
    /// char-searcher matches state (another pattern's
    /// searcher, whose step stays unsupported).
    fn decode(state: &Value) -> Option<Self> {
        let outer = iterator_fields(state, "").ok()?;
        if outer.len() != 1 {
            return None;
        }
        let internal = iterator_fields(&outer[0], "").ok()?;
        if internal.len() != 1 {
            return None;
        }
        let searcher = iterator_fields(&internal[0], "").ok()?;
        if searcher.len() != 6 {
            return None;
        }
        let Value::Str(haystack) = &searcher[0] else {
            return None;
        };
        let as_usize = |v: &Value| -> Option<usize> {
            match v {
                Value::Uint(x) => usize::try_from(*x).ok(),
                _ => None,
            }
        };
        let utf8_size = as_usize(&searcher[4])?;
        if utf8_size == 0 || utf8_size > 4 {
            return None;
        }
        let encoded = iterator_fields(&searcher[5], "").ok()?;
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
        Some(StrMatchesParts {
            haystack: haystack.clone(),
            finger: as_usize(&searcher[1])?,
            finger_back: as_usize(&searcher[2])?,
            needle,
            needle_value: searcher[3].clone(),
            utf8_size_value: searcher[4].clone(),
            utf8_encoded_value: searcher[5].clone(),
        })
    }

    /// Rebuild the state value with the stepped fields, in
    /// the exact shape `decode` read.
    fn encode(&self, state: &Value) -> Value {
        let outer = iterator_fields(state, "").expect("state decoded");
        let internal = iterator_fields(&outer[0], "").expect("state decoded");
        let new_searcher = rebuild_state(
            &internal[0],
            vec![
                Value::Str(self.haystack.clone()),
                Value::Uint(self.finger as u128),
                Value::Uint(self.finger_back as u128),
                self.needle_value.clone(),
                self.utf8_size_value.clone(),
                self.utf8_encoded_value.clone(),
            ],
        );
        let new_internal = rebuild_state(&outer[0], vec![new_searcher]);
        rebuild_state(state, vec![new_internal])
    }

    /// `CharSearcher::next_match`: the first needle
    /// occurrence in `haystack[finger..finger_back]`; the
    /// finger advances past it (the same search
    /// `StrSplitParts` runs).
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
    /// occurrence in `haystack[finger..finger_back]`; the
    /// back finger retreats to its start.
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
}

/// The decoded state of a `str` matches iterator over a
/// `CharEqSearcher` pattern: the searcher is a one-field
/// wrapper (the `CharPredicateSearcher`) over the
/// three-field searcher -- the char test, the haystack and
/// its `CharIndices` -- and the window the scan runs over
/// is the front/back byte indices of the `CharIndices`'
/// byte-slice iterator, exactly as in `StrSplitPredParts`.
struct StrMatchesEqParts {
    haystack: String,
    front: usize,
    back: usize,
    char_eq: Value,
    test: MatchesEq,
}

impl StrMatchesEqParts {
    /// Decode the state value, or `None` when it is not the
    /// char-eq searcher matches state. A predicate the
    /// modelled call path cannot apply -- anything but a
    /// closure or a function pointer; a zero-sized `Fn`
    /// type's value carries no callable identity of its own
    /// -- also declines, as in `StrSplitPredParts`.
    fn decode(state: &Value) -> Option<Self> {
        let outer = iterator_fields(state, "").ok()?;
        if outer.len() != 1 {
            return None;
        }
        let internal = iterator_fields(&outer[0], "").ok()?;
        if internal.len() != 1 {
            return None;
        }
        let wrapper = iterator_fields(&internal[0], "").ok()?;
        if wrapper.len() != 1 {
            return None;
        }
        let searcher = iterator_fields(&wrapper[0], "").ok()?;
        if searcher.len() != 3 {
            return None;
        }
        let test = match &searcher[0] {
            Value::Closure { .. } | Value::FnPtr(_) => {
                MatchesEq::Predicate(searcher[0].clone())
            }
            Value::Tuple(items) => {
                MatchesEq::Set(char_set_of(items)?)
            }
            Value::FatRef { .. } | Value::Ref(_) => {
                MatchesEq::SetRef(searcher[0].clone())
            }
            _ => return None,
        };
        let Value::Str(haystack) = &searcher[1] else {
            return None;
        };
        // `CharIndices` pairs the front char index with its
        // `Chars`, which wraps the byte-slice iterator whose
        // bounds are the window.
        let char_indices = iterator_fields(&searcher[2], "").ok()?;
        if char_indices.len() != 2 {
            return None;
        }
        let chars = iterator_fields(&char_indices[1], "").ok()?;
        if chars.len() != 1 {
            return None;
        }
        let (front, back) = slice_iter_bounds(&chars[0], "").ok()?;
        Some(StrMatchesEqParts {
            haystack: haystack.clone(),
            front,
            back,
            char_eq: searcher[0].clone(),
            test,
        })
    }

    /// Resolve a char-set pattern's reference (a `&[char]`
    /// pattern's slice, a `&[char; N]` pattern's array) to
    /// its chars, so the scan tests membership directly.
    /// `false` when the set does not resolve, leaving the
    /// step to the generic unsupported error.
    fn resolve_set(
        &mut self,
        interp: &mut Interpreter<'_>,
        frames: &mut Vec<Frame>,
    ) -> bool {
        let MatchesEq::SetRef(reference) = &self.test else {
            return true;
        };
        match resolve_char_set(interp, frames, reference) {
            Some(chars) => {
                self.test = MatchesEq::Set(chars);
                true
            }
            None => false,
        }
    }

    /// Rebuild the state value with the stepped fields, in
    /// the exact shape `decode` read.
    fn encode(&self, state: &Value) -> Value {
        let outer = iterator_fields(state, "").expect("state decoded");
        let internal = iterator_fields(&outer[0], "").expect("state decoded");
        let wrapper = iterator_fields(&internal[0], "").expect("state decoded");
        let searcher = iterator_fields(&wrapper[0], "").expect("state decoded");
        let char_indices = iterator_fields(&searcher[2], "").expect("state decoded");
        let chars = iterator_fields(&char_indices[1], "").expect("state decoded");
        let iter_fields = iterator_fields(&chars[0], "").expect("state decoded");
        let new_front =
            iter_ptr_with_index(&iter_fields[0], self.front).expect("state decoded");
        let new_back =
            iter_ptr_with_index(&iter_fields[1], self.back).expect("state decoded");
        let new_iter = rebuild_state(
            &chars[0],
            vec![new_front, new_back, iter_fields[2].clone()],
        );
        let new_chars = rebuild_state(&char_indices[1], vec![new_iter]);
        // The `CharIndices` front field is the char index of
        // the window's front byte position.
        let front_index = self.haystack[..self.front].chars().count();
        let new_indices = rebuild_state(
            &searcher[2],
            vec![Value::Uint(front_index as u128), new_chars],
        );
        let new_searcher = rebuild_state(
            &wrapper[0],
            vec![
                self.char_eq.clone(),
                Value::Str(self.haystack.clone()),
                new_indices,
            ],
        );
        let new_wrapper = rebuild_state(&internal[0], vec![new_searcher]);
        let new_internal = rebuild_state(&outer[0], vec![new_wrapper]);
        rebuild_state(state, vec![new_internal])
    }

    /// The searcher's char test: apply the predicate through
    /// the modelled call path, or test set membership.
    fn matches(
        &self,
        interp: &mut Interpreter<'_>,
        frames: &mut Vec<Frame>,
        c: char,
    ) -> InterpResult<bool> {
        match &self.test {
            MatchesEq::Predicate(predicate) => {
                match interp.call_callable_sync(
                    frames,
                    predicate,
                    None,
                    vec![Value::Uint(c as u32 as u128)],
                )? {
                    Value::Bool(b) => Ok(b),
                    other => Err(InterpError::type_mismatch(
                        "matches predicate result",
                        "a boolean",
                        other.kind_name(),
                    )),
                }
            }
            MatchesEq::Set(chars) => Ok(chars.contains(&c)),
            MatchesEq::SetRef(_) => Err(unsupported(
                "matches char set was not resolved before the step",
            )),
        }
    }

    /// The searcher's `next_match`: the first char of the
    /// window the char test accepts. Rejected chars are
    /// consumed, as the searcher's own `next` consumes them,
    /// so the front ends just past the accepted char, or at
    /// the window's back when no char is accepted.
    fn next_match(
        &mut self,
        interp: &mut Interpreter<'_>,
        frames: &mut Vec<Frame>,
    ) -> InterpResult<Option<(usize, usize)>> {
        let window = self.haystack[self.front..self.back].to_string();
        for (rel, c) in window.char_indices() {
            let at = self.front + rel;
            if self.matches(interp, frames, c)? {
                self.front = at + c.len_utf8();
                return Ok(Some((at, self.front)));
            }
        }
        self.front = self.back;
        Ok(None)
    }

    /// The searcher's `next_match_back`: the last accepted
    /// char of the window; the back retreats to its start.
    fn next_match_back(
        &mut self,
        interp: &mut Interpreter<'_>,
        frames: &mut Vec<Frame>,
    ) -> InterpResult<Option<(usize, usize)>> {
        let window = self.haystack[self.front..self.back].to_string();
        for (rel, c) in window.char_indices().rev() {
            let at = self.front + rel;
            if self.matches(interp, frames, c)? {
                self.back = at;
                return Ok(Some((at, at + c.len_utf8())));
            }
        }
        self.back = self.front;
        Ok(None)
    }
}

/// The chars of a char-set pattern's recorded elements, or
/// `None` when an element is not a char value.
fn char_set_of(items: &[Value]) -> Option<Vec<char>> {
    let mut chars = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Value::Uint(x) => {
                chars.push(char::from_u32(u32::try_from(*x).ok()?)?)
            }
            _ => return None,
        }
    }
    Some(chars)
}

/// Resolve a char-set pattern's reference to the chars it
/// names. A fat reference whose base is the sequence value
/// itself names its elements directly; any other reference
/// is dereferenced and its pointee read, following further
/// references the pointee's own value is stored behind (a
/// promoted array's slice value, for one, sits in a place
/// holding the slice's own fat reference), until the
/// elements read as a sequence of chars. `None` when a hop
/// fails or the elements are not chars.
fn resolve_char_set(
    interp: &mut Interpreter<'_>,
    frames: &mut Vec<Frame>,
    reference: &Value,
) -> Option<Vec<char>> {
    let mut current = reference.clone();
    for _ in 0..8 {
        // A fat reference whose base is the sequence value
        // itself names its elements directly: the first
        // `len` of them are the set.
        if let Value::FatRef { base, len } = &current {
            let n = usize::try_from(*len).ok()?;
            match &**base {
                Value::Tuple(items) => {
                    return char_set_of(items.get(..n)?)
                }
                Value::Enum { fields, .. } => {
                    return char_set_of(fields.get(..n)?)
                }
                _ => {}
            }
        }
        let target = interp.deref_value(frames, &current).ok()?;
        let value = interp.read_resolved(frames, &target).ok()?;
        match &value {
            Value::Tuple(items) => return char_set_of(items),
            Value::Enum { fields, .. } => return char_set_of(fields),
            Value::FatRef { .. } | Value::Ref(_) => {
                current = value.clone()
            }
            _ => return None,
        }
    }
    None
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

/// How a split `CharEqSearcher`'s char test is served: a
/// guest predicate applied through the modelled call path,
/// a set of chars tested by membership (resolved from the
/// state), or a char set behind a reference (a `&[char]` or
/// `&[char; N]` pattern), resolved through the interpreter
/// before the first step. The matches family's `MatchesEq`
/// is the same trichotomy.
enum SplitEq {
    Predicate,
    Set(Vec<char>),
    SetRef(Value),
}

/// The decoded state of a `str` split iterator over a
/// `CharEq` pattern: core's `SplitInternal` fields plus the
/// `MultiCharEqSearcher` state the step machine drives. The
/// searcher is a one-field wrapper over the
/// `MultiCharEqSearcher` -- its char test, its haystack and
/// its `CharIndices` -- and the window the scan runs over is
/// the front/back byte indices of the `CharIndices`'
/// byte-slice iterator: the searcher's front step consumes
/// chars from the front of the one window, its back step
/// from the back, exactly as core's searcher shares the one
/// `CharIndices` between both directions.
struct StrSplitPredParts {
    start: usize,
    end: usize,
    haystack: String,
    front: usize,
    back: usize,
    char_eq: Value,
    test: SplitEq,
    predicate_ty: String,
    allow_trailing_empty: bool,
    finished: bool,
}

impl StrSplitPredParts {
    /// Decode the state value, or `None` when it is not the
    /// char-eq-searcher split state (another pattern's
    /// searcher, whose step stays unsupported). The char
    /// test the state carries decides the arm: a predicate
    /// the modelled call path cannot apply declines, leaving
    /// the state to the generic unsupported error -- the
    /// applicable shapes are a closure, a function pointer,
    /// and the empty value of a zero-sized `Fn` type (core's
    /// `impl_fn_for_zst!` family), which carries no callable
    /// identity of its own and is applied through the
    /// type-directed dispatch under the pattern type the
    /// split's tag names. A char set is the `[char; N]`
    /// pattern's chars by value, or a reference to the set
    /// (a `&[char]` / `&[char; N]` pattern).
    fn decode(state: &Value, self_ty: &str) -> Option<Self> {
        let outer = iterator_fields(state, "").ok()?;
        if outer.len() != 1 {
            return None;
        }
        let internal = iterator_fields(&outer[0], "").ok()?;
        if internal.len() != 5 {
            return None;
        }
        let matcher = iterator_fields(&internal[2], "").ok()?;
        if matcher.len() != 1 {
            return None;
        }
        let searcher = iterator_fields(&matcher[0], "").ok()?;
        if searcher.len() != 3 {
            return None;
        }
        let test = match &searcher[0] {
            Value::Closure { .. } | Value::FnPtr(_) => SplitEq::Predicate,
            Value::Enum { fields, .. } if fields.is_empty() => {
                SplitEq::Predicate
            }
            Value::Tuple(fields) if fields.is_empty() => {
                SplitEq::Predicate
            }
            Value::Tuple(items) => SplitEq::Set(char_set_of(items)?),
            Value::FatRef { .. } | Value::Ref(_) => {
                SplitEq::SetRef(searcher[0].clone())
            }
            _ => return None,
        };
        let Value::Str(haystack) = &searcher[1] else {
            return None;
        };
        // `CharIndices` pairs the front char index with its
        // `Chars`, which wraps the byte-slice iterator whose
        // bounds are the window.
        let char_indices = iterator_fields(&searcher[2], "").ok()?;
        if char_indices.len() != 2 {
            return None;
        }
        let chars = iterator_fields(&char_indices[1], "").ok()?;
        if chars.len() != 1 {
            return None;
        }
        let (front, back) = slice_iter_bounds(&chars[0], "").ok()?;
        let as_usize = |v: &Value| -> Option<usize> {
            match v {
                Value::Uint(x) => usize::try_from(*x).ok(),
                _ => None,
            }
        };
        Some(StrSplitPredParts {
            start: as_usize(&internal[0])?,
            end: as_usize(&internal[1])?,
            haystack: haystack.clone(),
            front,
            back,
            char_eq: searcher[0].clone(),
            test,
            predicate_ty: first_generic_arg(self_ty).unwrap_or("").to_string(),
            allow_trailing_empty: internal[3] == Value::Bool(true),
            finished: internal[4] == Value::Bool(true),
        })
    }

    /// Rebuild the state value with the stepped fields, in
    /// the exact shape `decode` read.
    fn encode(&self, state: &Value) -> Value {
        let outer = iterator_fields(state, "").expect("state decoded");
        let internal = iterator_fields(&outer[0], "").expect("state decoded");
        let matcher = iterator_fields(&internal[2], "").expect("state decoded");
        let searcher = iterator_fields(&matcher[0], "").expect("state decoded");
        let char_indices = iterator_fields(&searcher[2], "").expect("state decoded");
        let chars = iterator_fields(&char_indices[1], "").expect("state decoded");
        let iter_fields = iterator_fields(&chars[0], "").expect("state decoded");
        let new_front =
            iter_ptr_with_index(&iter_fields[0], self.front).expect("state decoded");
        let new_back =
            iter_ptr_with_index(&iter_fields[1], self.back).expect("state decoded");
        let new_iter = rebuild_state(
            &chars[0],
            vec![new_front, new_back, iter_fields[2].clone()],
        );
        let new_chars = rebuild_state(&char_indices[1], vec![new_iter]);
        // The `CharIndices` front field is the char index of
        // the window's front byte position.
        let front_index = self.haystack[..self.front].chars().count();
        let new_indices = rebuild_state(
            &searcher[2],
            vec![Value::Uint(front_index as u128), new_chars],
        );
        let new_searcher = rebuild_state(
            &matcher[0],
            vec![
                self.char_eq.clone(),
                Value::Str(self.haystack.clone()),
                new_indices,
            ],
        );
        let new_matcher = rebuild_state(&internal[2], vec![new_searcher]);
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

    /// The searcher's `char_eq.matches`: apply the predicate
    /// through the modelled call path, or test set membership.
    fn matches(
        &self,
        interp: &mut Interpreter<'_>,
        frames: &mut Vec<Frame>,
        c: char,
    ) -> InterpResult<bool> {
        match &self.test {
            SplitEq::Predicate => match interp.call_callable_sync_typed(
                frames,
                &self.char_eq,
                &self.predicate_ty,
                None,
                vec![Value::Uint(c as u32 as u128)],
            )? {
                Value::Bool(b) => Ok(b),
                other => Err(InterpError::type_mismatch(
                    "split predicate result",
                    "a boolean",
                    other.kind_name(),
                )),
            },
            SplitEq::Set(chars) => Ok(chars.contains(&c)),
            SplitEq::SetRef(_) => Err(unsupported(
                "split char set was not resolved before the step",
            )),
        }
    }

    /// Resolve a char-set pattern's reference to its chars,
    /// so the scan tests membership directly. `false` when
    /// the set does not resolve, leaving the step to the
    /// generic unsupported error. A by-value test needs no
    /// resolution.
    fn resolve_set(
        &mut self,
        interp: &mut Interpreter<'_>,
        frames: &mut Vec<Frame>,
    ) -> bool {
        let SplitEq::SetRef(reference) = &self.test else {
            return true;
        };
        match resolve_char_set(interp, frames, reference) {
            Some(chars) => {
                self.test = SplitEq::Set(chars);
                true
            }
            None => false,
        }
    }

    /// The searcher's `next_match`: the first char of the
    /// window the predicate accepts. Rejected chars are
    /// consumed, as the searcher's own `next` consumes them,
    /// so the front ends just past the accepted char, or at
    /// the window's back when no char is accepted.
    fn next_match(
        &mut self,
        interp: &mut Interpreter<'_>,
        frames: &mut Vec<Frame>,
    ) -> InterpResult<Option<(usize, usize)>> {
        let window = self.haystack[self.front..self.back].to_string();
        for (rel, c) in window.char_indices() {
            let at = self.front + rel;
            if self.matches(interp, frames, c)? {
                self.front = at + c.len_utf8();
                return Ok(Some((at, self.front)));
            }
        }
        self.front = self.back;
        Ok(None)
    }

    /// The searcher's `next_match_back`: the last accepted
    /// char of the window; the back retreats to its start.
    fn next_match_back(
        &mut self,
        interp: &mut Interpreter<'_>,
        frames: &mut Vec<Frame>,
    ) -> InterpResult<Option<(usize, usize)>> {
        let window = self.haystack[self.front..self.back].to_string();
        for (rel, c) in window.char_indices().rev() {
            let at = self.front + rel;
            if self.matches(interp, frames, c)? {
                self.back = at;
                return Ok(Some((at, at + c.len_utf8())));
            }
        }
        self.back = self.front;
        Ok(None)
    }

    /// `SplitInternal::get_end` (the same tail the `char`
    /// searcher's machine runs).
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
    fn next(
        &mut self,
        interp: &mut Interpreter<'_>,
        frames: &mut Vec<Frame>,
    ) -> InterpResult<Option<String>> {
        if self.finished {
            return Ok(None);
        }
        match self.next_match(interp, frames)? {
            Some((a, b)) => {
                let elt = self.haystack[self.start..a].to_string();
                self.start = b;
                Ok(Some(elt))
            }
            None => Ok(self.get_end()),
        }
    }

    /// `SplitInternal::next_inclusive`.
    fn next_inclusive(
        &mut self,
        interp: &mut Interpreter<'_>,
        frames: &mut Vec<Frame>,
    ) -> InterpResult<Option<String>> {
        if self.finished {
            return Ok(None);
        }
        match self.next_match(interp, frames)? {
            Some((_, b)) => {
                let elt = self.haystack[self.start..b].to_string();
                self.start = b;
                Ok(Some(elt))
            }
            None => Ok(self.get_end()),
        }
    }

    /// The post-dance half of `SplitInternal::next_back` /
    /// `next_back_inclusive`, mirroring the `char` searcher's
    /// machine.
    fn back_match(
        &mut self,
        interp: &mut Interpreter<'_>,
        frames: &mut Vec<Frame>,
        inclusive: bool,
    ) -> InterpResult<Option<String>> {
        match self.next_match_back(interp, frames)? {
            Some((a, b)) => {
                let elt = self.haystack[b..self.end].to_string();
                self.end = if inclusive { b } else { a };
                Ok(Some(elt))
            }
            None => {
                self.finished = true;
                Ok(Some(self.haystack[self.start..self.end].to_string()))
            }
        }
    }

    /// `SplitInternal::next_back`, including its
    /// `allow_trailing_empty` dance.
    fn next_back(
        &mut self,
        interp: &mut Interpreter<'_>,
        frames: &mut Vec<Frame>,
    ) -> InterpResult<Option<String>> {
        if self.finished {
            return Ok(None);
        }
        if !self.allow_trailing_empty {
            self.allow_trailing_empty = true;
            match self.back_match(interp, frames, false)? {
                Some(elt) if !elt.is_empty() => return Ok(Some(elt)),
                _ => {
                    if self.finished {
                        return Ok(None);
                    }
                }
            }
        }
        self.back_match(interp, frames, false)
    }

    /// `SplitInternal::next_back_inclusive`: the mirror of
    /// `next_back` over `back_match`'s inclusive form.
    fn next_back_inclusive(
        &mut self,
        interp: &mut Interpreter<'_>,
        frames: &mut Vec<Frame>,
    ) -> InterpResult<Option<String>> {
        if self.finished {
            return Ok(None);
        }
        if !self.allow_trailing_empty {
            self.allow_trailing_empty = true;
            match self.back_match(interp, frames, true)? {
                Some(elt) if !elt.is_empty() => return Ok(Some(elt)),
                _ => {
                    if self.finished {
                        return Ok(None);
                    }
                }
            }
        }
        self.back_match(interp, frames, true)
    }
}

/// The decoded state of an `&str` pattern's searcher
/// (`std::str::pattern::StrSearcher`, built by the
/// `StrSearcherNew` builtin): the haystack and the needle
/// the searcher stores, and the front / back cursors of its
/// `StrSearcherImpl` payload -- the `Empty` payload's
/// `position` / `end`, the `Byte` payload's, or the `TwoWay`
/// payload's. A step replays core's searcher semantics over
/// the stored strings: a substring scan of the remaining
/// window for the non-empty payloads (the two-way algorithm
/// computes the same match positions), and for the `Empty`
/// payload a transcription of core's steppers, whose match
/// flags and finished flag are carried and toggled exactly
/// as core's are. The payload fields the machine does not
/// read (the byte, the two-way constants and memory) are
/// carried as their original values so the rebuilt state is
/// field-exact, as in `StrMatchesParts`.
struct StrSearcherParts {
    haystack: String,
    needle: String,
    variant: u32,
    payload: Vec<Value>,
    finger: usize,
    finger_back: usize,
    // The `Empty` payload's flags, in its field order.
    match_fw: bool,
    match_bw: bool,
    finished: bool,
}

impl StrSearcherParts {
    /// The payload's `(position, end)` field indices, by
    /// `StrSearcherImpl` variant.
    fn cursor_indices(variant: u32) -> Option<(usize, usize, usize)> {
        match variant {
            // Empty { position, end, .. }
            0 => Some((0, 1, 5)),
            // Byte { b, position, end }
            1 => Some((1, 2, 3)),
            // TwoWay { .., position, end, .. }
            2 => Some((4, 5, 8)),
            _ => None,
        }
    }

    fn decode(searcher: &Value) -> Option<Self> {
        let fields = iterator_fields(searcher, "").ok()?;
        if fields.len() != 3 {
            return None;
        }
        let Value::Str(haystack) = &fields[0] else {
            return None;
        };
        let Value::Str(needle) = &fields[1] else {
            return None;
        };
        let Value::Enum {
            variant,
            fields: impl_fields,
        } = &fields[2]
        else {
            return None;
        };
        if impl_fields.len() != 1 {
            return None;
        }
        let (finger_idx, back_idx, payload_len) = Self::cursor_indices(*variant)?;
        let payload = iterator_fields(&impl_fields[0], "").ok()?;
        if payload.len() != payload_len {
            return None;
        }
        // The variant is a function of the needle's shape;
        // decline a state that disagrees with it rather than
        // step a shape the constructor never builds.
        let shape_ok = match variant {
            0 => needle.is_empty(),
            1 => needle.len() == 1,
            _ => needle.len() > 1,
        };
        if !shape_ok {
            return None;
        }
        let as_usize = |v: &Value| -> Option<usize> {
            match v {
                Value::Uint(n) => Some(*n as usize),
                Value::Int(n) => Some(*n as usize),
                _ => None,
            }
        };
        let as_bool = |v: &Value| -> Option<bool> {
            match v {
                Value::Bool(b) => Some(*b),
                _ => None,
            }
        };
        let finger = as_usize(&payload[finger_idx])?;
        let finger_back = as_usize(&payload[back_idx])?;
        let (match_fw, match_bw, finished) = if *variant == 0 {
            (
                as_bool(&payload[2])?,
                as_bool(&payload[3])?,
                as_bool(&payload[4])?,
            )
        } else {
            (false, false, false)
        };
        Some(Self {
            haystack: haystack.clone(),
            needle: needle.clone(),
            variant: *variant,
            payload,
            finger,
            finger_back,
            match_fw,
            match_bw,
            finished,
        })
    }

    fn encode(&self, searcher: &Value) -> Value {
        let fields = iterator_fields(searcher, "").expect("state decoded above");
        let Value::Enum {
            fields: impl_fields,
            ..
        } = &fields[2]
        else {
            unreachable!("state decoded above")
        };
        let (finger_idx, back_idx, _) =
            Self::cursor_indices(self.variant).expect("state decoded above");
        let mut payload = self.payload.clone();
        payload[finger_idx] = Value::Uint(self.finger as u128);
        payload[back_idx] = Value::Uint(self.finger_back as u128);
        if self.variant == 0 {
            payload[2] = Value::Bool(self.match_fw);
            payload[3] = Value::Bool(self.match_bw);
            payload[4] = Value::Bool(self.finished);
        }
        let payload_value = rebuild_state(&impl_fields[0], payload);
        let impl_value = rebuild_state(&fields[2], vec![payload_value]);
        rebuild_state(
            searcher,
            vec![
                Value::Str(self.haystack.clone()),
                Value::Str(self.needle.clone()),
                impl_value,
            ],
        )
    }

    /// One step of core's `Empty` searcher in the front
    /// direction: `Some(Some(range))` for a match step,
    /// `Some(None)` once finished, `None` for a reject step
    /// (the caller loops), transcribed from core's
    /// `Searcher::next`.
    fn empty_next(&mut self) -> Option<Option<(usize, usize)>> {
        if self.finished {
            return Some(None);
        }
        let is_match = self.match_fw;
        self.match_fw = !self.match_fw;
        let pos = self.finger;
        if is_match {
            return Some(Some((pos, pos)));
        }
        match self.haystack[pos..].chars().next() {
            None => {
                self.finished = true;
                Some(None)
            }
            Some(ch) => {
                self.finger = pos + ch.len_utf8();
                None
            }
        }
    }

    /// One step of core's `Empty` searcher in the back
    /// direction; see `empty_next`.
    fn empty_next_back(&mut self) -> Option<Option<(usize, usize)>> {
        if self.finished {
            return Some(None);
        }
        let is_match = self.match_bw;
        self.match_bw = !self.match_bw;
        let end = self.finger_back;
        if is_match {
            return Some(Some((end, end)));
        }
        match self.haystack[..end].chars().next_back() {
            None => {
                self.finished = true;
                Some(None)
            }
            Some(ch) => {
                self.finger_back = end - ch.len_utf8();
                None
            }
        }
    }

    /// Core `Searcher::next_match`: the next match's byte
    /// range, front first.
    fn next_match(&mut self) -> Option<(usize, usize)> {
        if self.variant == 0 {
            if self.finger > self.finger_back {
                return None;
            }
            loop {
                match self.empty_next() {
                    Some(found) => return found,
                    None => {}
                }
            }
        }
        let n = self.needle.len();
        let bytes = self.haystack.as_bytes();
        let Some(range) = bytes.get(self.finger..self.finger_back) else {
            self.finger = self.finger_back;
            return None;
        };
        let at = range
            .windows(n)
            .position(|w| w == self.needle.as_bytes());
        match at {
            Some(at) => {
                let found = self.finger + at;
                self.finger = found + n;
                Some((found, found + n))
            }
            None => {
                self.finger = self.finger_back;
                None
            }
        }
    }

    /// Core `Searcher::next_match_back`: the next match's
    /// byte range, back first.
    fn next_match_back(&mut self) -> Option<(usize, usize)> {
        if self.variant == 0 {
            if self.finger > self.finger_back {
                return None;
            }
            loop {
                match self.empty_next_back() {
                    Some(found) => return found,
                    None => {}
                }
            }
        }
        let n = self.needle.len();
        let bytes = self.haystack.as_bytes();
        let Some(range) = bytes.get(self.finger..self.finger_back) else {
            self.finger_back = self.finger;
            return None;
        };
        let at = range
            .windows(n)
            .rposition(|w| w == self.needle.as_bytes());
        match at {
            Some(at) => {
                let found = self.finger + at;
                self.finger_back = found;
                Some((found, found + n))
            }
            None => {
                self.finger_back = self.finger;
                None
            }
        }
    }
}

/// The decoded state of a `str` matches iterator over an
/// `&str` pattern: the two one-field wrappers over the
/// pattern's `StrSearcher` (see `StrSearcherParts`).
struct StrMatchesStrParts {
    searcher: StrSearcherParts,
}

impl StrMatchesStrParts {
    fn decode(state: &Value) -> Option<Self> {
        let outer = iterator_fields(state, "").ok()?;
        if outer.len() != 1 {
            return None;
        }
        let internal = iterator_fields(&outer[0], "").ok()?;
        if internal.len() != 1 {
            return None;
        }
        Some(Self {
            searcher: StrSearcherParts::decode(&internal[0])?,
        })
    }

    fn encode(&self, state: &Value) -> Value {
        let outer = iterator_fields(state, "").expect("state decoded above");
        let internal = iterator_fields(&outer[0], "").expect("state decoded above");
        let searcher = self.searcher.encode(&internal[0]);
        let internal = rebuild_state(&outer[0], vec![searcher]);
        rebuild_state(state, vec![internal])
    }

    fn next_match(&mut self) -> Option<(usize, usize)> {
        self.searcher.next_match()
    }

    fn next_match_back(&mut self) -> Option<(usize, usize)> {
        self.searcher.next_match_back()
    }
}

/// The decoded state of a `str` split iterator over an
/// `&str` pattern: the `SplitInternal` fields with the
/// matcher decoded as a `StrSearcher` (see
/// `StrSearcherParts`). The step replay is `StrSplitParts`'
/// with the matcher's `next_match` in place of the char
/// search.
struct StrSplitStrParts {
    start: usize,
    end: usize,
    searcher: StrSearcherParts,
    allow_trailing_empty: bool,
    finished: bool,
}

impl StrSplitStrParts {
    fn decode(state: &Value) -> Option<Self> {
        let outer = iterator_fields(state, "").ok()?;
        if outer.len() != 1 {
            return None;
        }
        let internal = iterator_fields(&outer[0], "").ok()?;
        if internal.len() != 5 {
            return None;
        }
        let as_usize = |v: &Value| -> Option<usize> {
            match v {
                Value::Uint(n) => Some(*n as usize),
                Value::Int(n) => Some(*n as usize),
                _ => None,
            }
        };
        let as_bool = |v: &Value| -> Option<bool> {
            match v {
                Value::Bool(b) => Some(*b),
                _ => None,
            }
        };
        Some(Self {
            start: as_usize(&internal[0])?,
            end: as_usize(&internal[1])?,
            searcher: StrSearcherParts::decode(&internal[2])?,
            allow_trailing_empty: as_bool(&internal[3])?,
            finished: as_bool(&internal[4])?,
        })
    }

    fn encode(&self, state: &Value) -> Value {
        let outer = iterator_fields(state, "").expect("state decoded above");
        let internal = iterator_fields(&outer[0], "").expect("state decoded above");
        let searcher = self.searcher.encode(&internal[2]);
        let internal = rebuild_state(
            &outer[0],
            vec![
                Value::Uint(self.start as u128),
                Value::Uint(self.end as u128),
                searcher,
                Value::Bool(self.allow_trailing_empty),
                Value::Bool(self.finished),
            ],
        );
        rebuild_state(state, vec![internal])
    }

    fn haystack(&self) -> &str {
        &self.searcher.haystack
    }

    fn needle_len(&self) -> usize {
        self.searcher.needle.len()
    }

    // The `SplitInternal` replay, as in `StrSplitParts`.

    /// `SplitInternal::get_end`.
    fn get_end(&mut self) -> Option<String> {
        if !self.finished {
            self.finished = true;
            if self.allow_trailing_empty || self.end > self.start {
                return Some(self.haystack()[self.start..self.end].to_string());
            }
        }
        None
    }

    /// `SplitInternal::next`.
    fn next(&mut self) -> Option<String> {
        if self.finished {
            return None;
        }
        match self.searcher.next_match() {
            Some((a, _)) => {
                let elt = self.haystack()[self.start..a].to_string();
                self.start = a + self.needle_len();
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
        match self.searcher.next_match() {
            Some((_, b)) => {
                let elt = self.haystack()[self.start..b].to_string();
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
        match self.searcher.next_match_back() {
            Some((a, b)) => {
                let elt = self.haystack()[b..self.end].to_string();
                self.end = if inclusive { b } else { a };
                Some(elt)
            }
            None => {
                self.finished = true;
                Some(self.haystack()[self.start..self.end].to_string())
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

/// The wrapped split state of a `SplitNInternal`, decoded
/// by the same machines `str_split_step` runs: the `char`
/// pattern's, the predicate pattern's, and the `&str`
/// pattern's. The step surface `SplitNInternal` needs is
/// the front step, the back step, and `get_end`.
enum StrSplitNInner {
    Char(StrSplitParts),
    Pred(StrSplitPredParts),
    Str(StrSplitStrParts),
}

impl StrSplitNInner {
    fn get_end(&mut self) -> Option<String> {
        match self {
            StrSplitNInner::Char(parts) => parts.get_end(),
            StrSplitNInner::Pred(parts) => parts.get_end(),
            StrSplitNInner::Str(parts) => parts.get_end(),
        }
    }

    fn next(
        &mut self,
        interp: &mut Interpreter<'_>,
        frames: &mut Vec<Frame>,
    ) -> InterpResult<Option<String>> {
        match self {
            StrSplitNInner::Char(parts) => Ok(parts.next()),
            StrSplitNInner::Pred(parts) => parts.next(interp, frames),
            StrSplitNInner::Str(parts) => Ok(parts.next()),
        }
    }

    fn next_back(
        &mut self,
        interp: &mut Interpreter<'_>,
        frames: &mut Vec<Frame>,
    ) -> InterpResult<Option<String>> {
        match self {
            StrSplitNInner::Char(parts) => Ok(parts.next_back()),
            StrSplitNInner::Pred(parts) => parts.next_back(interp, frames),
            StrSplitNInner::Str(parts) => Ok(parts.next_back()),
        }
    }

    fn encode(&self, wrapped: &Value) -> Value {
        match self {
            StrSplitNInner::Char(parts) => parts.encode(wrapped),
            StrSplitNInner::Pred(parts) => parts.encode(wrapped),
            StrSplitNInner::Str(parts) => parts.encode(wrapped),
        }
    }
}

/// The decoded state of a `str` splitn iterator: core's
/// `SplitNInternal` fields -- the wrapped `SplitInternal`
/// state (see `StrSplitNInner`) and the count of splits
/// remaining. The split machines decode a one-field wrapper
/// over the `SplitInternal` state, so decode and encode run
/// them over the wrapped field under that shape.
struct StrSplitNParts {
    inner: StrSplitNInner,
    count: usize,
}

impl StrSplitNParts {
    fn decode(state: &Value, self_ty: &str) -> Option<Self> {
        let outer = iterator_fields(state, "").ok()?;
        if outer.len() != 1 {
            return None;
        }
        let internal = iterator_fields(&outer[0], "").ok()?;
        if internal.len() != 2 {
            return None;
        }
        let count = match internal[1] {
            Value::Uint(n) => usize::try_from(n).ok()?,
            _ => return None,
        };
        let wrapped = Value::Tuple(vec![internal[0].clone()]);
        let inner = if let Some(parts) = StrSplitParts::decode(&wrapped) {
            StrSplitNInner::Char(parts)
        } else if let Some(parts) = StrSplitPredParts::decode(&wrapped, self_ty) {
            StrSplitNInner::Pred(parts)
        } else if let Some(parts) = StrSplitStrParts::decode(&wrapped) {
            StrSplitNInner::Str(parts)
        } else {
            return None;
        };
        Some(Self { inner, count })
    }

    fn encode(&self, state: &Value) -> Value {
        let outer = iterator_fields(state, "").expect("state decoded above");
        let internal = iterator_fields(&outer[0], "").expect("state decoded above");
        let wrapped = Value::Tuple(vec![internal[0].clone()]);
        let stepped = self.inner.encode(&wrapped);
        let new_iter = iterator_fields(&stepped, "")
            .expect("state decoded above")
            .into_iter()
            .next()
            .expect("state decoded above");
        let new_internal = rebuild_state(
            &outer[0],
            vec![new_iter, Value::Uint(self.count as u128)],
        );
        rebuild_state(state, vec![new_internal])
    }

    /// `SplitNInternal::next` / `next_back`: `back` selects
    /// the wrapped step's direction, as core's two entry
    /// points do; the count logic is shared.
    fn next(
        &mut self,
        interp: &mut Interpreter<'_>,
        frames: &mut Vec<Frame>,
        back: bool,
    ) -> InterpResult<Option<String>> {
        match self.count {
            0 => Ok(None),
            1 => {
                self.count = 0;
                Ok(self.inner.get_end())
            }
            _ => {
                self.count -= 1;
                if back {
                    self.inner.next_back(interp, frames)
                } else {
                    self.inner.next(interp, frames)
                }
            }
        }
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
