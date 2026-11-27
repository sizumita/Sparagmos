//! Virtual dispatch resolution: scanning the program for
//! the concrete implementation behind a trait-object call,
//! and evaluating callable operands through it.

use super::lookup::{
    impl_self_matches, normalize_impl_self, parse_impl_block_key, parse_impl_key,
};
use super::ref_shaped_receiver;
use crate::error::{InterpError, InterpResult};
use crate::interp::*;
use crate::value::{RefProj, RefTarget, Value};

impl<'a> Interpreter<'a> {
    /// Evaluate a callable operand (a closure or function value being
    /// called by value) together with the environment reference a
    /// closure body expects as its first argument. The body's first
    /// parameter is a reference to the closure value in rustc MIR;
    /// when the operand is a place, the reference points at that
    /// place, so `FnMut` mutations of by-value captures persist
    /// across calls, exactly as through the real shim. A `Move`
    /// operand's place is deliberately not cleared here: the
    /// environment reference aliases it for the duration of the
    /// call, and the place is dead afterwards by MIR's move
    /// semantics. (A body that takes its environment by value is
    /// the exception: `consumed_env_arg` passes the value itself
    /// and does clear the place.)

    /// The callable value and environment reference behind a
    /// `dyn Fn`-family fat pointer: the concrete value it denotes
    /// (a closure value, for the payloads a program can contain),
    /// referenced through the fat pointer's own base when that
    /// base is a reference (a `&dyn Fn`), and as a value otherwise
    /// (an owned `Box<dyn Fn>` — the closure value inside the box
    /// aggregate; its captures are place references, so calls
    /// through the copy still reach them).
    /// The unique program key of an impl of `trait_path` for the
    /// concrete type `concrete` with method `method`, under any
    /// generic-argument spelling of the trait (or of `Self`) and
    /// under either the `std::` or the `core::` spelling of the
    /// trait path. `None` when there is not exactly one candidate
    /// among the converted and unconverted bodies.
    pub(crate) fn scan_virtual_impl(
        &self,
        concrete: &str,
        trait_path: &str,
        method: &str,
    ) -> Option<String> {
        let mut paths = vec![trait_path.to_string()];
        if let Some(rest) = trait_path.strip_prefix("std::") {
            paths.push(format!("core::{rest}"));
        } else if let Some(rest) = trait_path.strip_prefix("core::") {
            paths.push(format!("std::{rest}"));
        }
        // A first pass over the bodies conversion keyed by
        // the tag spelling of their `Self` type (see
        // `Program::impl_tag_bodies`): the queried name is
        // a rendered type tag, and where a type inside the
        // generic arguments is spelled differently in a
        // tag than in an instance key -- a closure, above
        // all -- neither spelling pass below can tell two
        // instances of one generic impl apart, while the
        // table names the one body whose receiver's
        // declared type renders as exactly this tag.
        for path in &paths {
            if let Some(key) = self.program.impl_tag_bodies.get(&(
                concrete.to_string(),
                path.clone(),
                method.to_string(),
            )) {
                return Some(key.clone());
            }
        }
        let keys = || {
            self.program
                .functions
                .keys()
                .chain(self.program.unconverted.keys())
        };
        // A first pass for the candidate whose `Self` spelling
        // is the queried type itself, up to the spellings the
        // two sides legitimately differ in (see
        // `normalize_impl_self`): the base-name match below
        // cannot tell two impls of one generic type apart --
        // the `default_write_fmt` adapters over two different
        // writers, say -- and this pass can.
        let wanted = normalize_impl_self(concrete);
        let mut exact: Option<&String> = None;
        for key in keys() {
            let base = key.split('#').next().unwrap_or(key.as_str());
            let Some((self_part, trait_base, key_method)) = parse_impl_key(base) else {
                continue;
            };
            if key_method != method
                || !paths.iter().any(|p| p == &trait_base)
                || normalize_impl_self(&self_part) != wanted
            {
                continue;
            }
            if exact == Some(key) {
                continue;
            }
            if exact.is_some() {
                return None;
            }
            exact = Some(key);
        }
        if let Some(key) = exact {
            return Some(key.clone());
        }
        let mut found: Option<String> = None;
        for key in keys() {
            let base = key.split('#').next().unwrap_or(key.as_str());
            // An impl-block key (`...::<impl Trait for Self>::m`)
            // also parses, wrongly, as an impl key — its first
            // ` as ` is the inner impl's — so the marker, not
            // the parse order, chooses the parser.
            let Some((self_part, trait_base, key_method)) = (if base.contains("::<impl ") {
                parse_impl_block_key(base)
            } else {
                parse_impl_key(base)
            }) else {
                continue;
            };
            if key_method != method
                || !paths.iter().any(|p| p == &trait_base)
                || !impl_self_matches(&self_part, concrete)
            {
                continue;
            }
            if found.as_deref() == Some(key.as_str()) {
                continue;
            }
            if found.is_some() {
                return None;
            }
            found = Some(key.clone());
        }
        found
    }

    /// Dispatch an `Fn`-trait call whose dyn concrete value is an
    /// ADT value (a hand-written `Fn` / `FnMut` / `FnOnce` impl on
    /// a struct, reached through a `&dyn FnMut` and the like): scan
    /// for the concrete type's impl body for the call's kind and
    /// start it with `(receiver, argument tuple)`, the receiver
    /// shaped exactly as `start_call` shapes a vtable receiver —
    /// a reference when the impl body takes one (unwrapping a
    /// one-field aggregate data pointer, keeping an existing
    /// reference, otherwise referencing the value), the value
    /// itself when it takes the receiver by value. `None` when no
    /// impl body matches, leaving the caller to report the value
    /// as not callable.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dyn_adt_fn_call(
        &mut self,
        frames: &mut Vec<Frame>,
        kind: FnCallKind,
        concrete: &str,
        callable: Value,
        env_ref: Value,
        arg_values: Vec<Value>,
        destination: &Place,
        target: Option<BasicBlock>,
    ) -> Option<InterpResult<()>> {
        let (trait_path, method) = match kind {
            FnCallKind::Call => ("std::ops::Fn", "call"),
            FnCallKind::CallMut => ("std::ops::FnMut", "call_mut"),
            FnCallKind::CallOnce => ("std::ops::FnOnce", "call_once"),
            FnCallKind::AsyncCall => ("std::ops::AsyncFn", "async_call"),
            FnCallKind::AsyncCallMut => ("std::ops::AsyncFnMut", "async_call_mut"),
            FnCallKind::AsyncCallOnce => ("std::ops::AsyncFnOnce", "async_call_once"),
        };
        let impl_name = self.scan_virtual_impl(concrete, trait_path, method)?;
        let wants_ref = matches!(
            self.program
                .get(&impl_name)
                .and_then(|b| b.local_tys.get(1)),
            Some(Ty::Ref(..))
        );
        let receiver = if wants_ref {
            match env_ref {
                Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. } => env_ref,
                _ => ref_shaped_receiver(callable),
            }
        } else {
            callable
        };
        Some(self.start_call(
            frames,
            &impl_name,
            vec![receiver, Value::Tuple(arg_values)],
            destination,
            target,
        ))
    }

    pub(crate) fn dyn_callable(
        &self,
        frames: &[Frame],
        value: &Value,
    ) -> InterpResult<(Value, Value)> {
        let Value::DynRef { base, .. } = value else {
            unreachable!("dyn_callable on a non-dyn value")
        };
        let resolved = self.deref_value(frames, value)?;
        let callable = self.read_resolved(frames, &resolved)?;
        let env = match &**base {
            Value::Ref(_) | Value::FatRef { .. } => (**base).clone(),
            _ => Value::Ref(RefTarget::value(callable.clone())),
        };
        Ok((callable, env))
    }

    /// Whether the receiver of an `Fn`-family trait call is
    /// an async closure value: a closure whose definition the
    /// conversion recorded as a coroutine closure (see
    /// `Program::async_closure_captures`, which the recorded
    /// definitions are folded into). For `call_once` the
    /// receiver is the value itself; for `call` / `call_mut`
    /// it is a reference to the value. A receiver that cannot
    /// be read as a closure value is not one - the call's own
    /// dispatch produces its usual outcome for it.
    pub(crate) fn fn_call_receiver_is_async_closure(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        receiver: &Operand,
        by_value: bool,
    ) -> bool {
        // Read the receiver without consuming it: this is a
        // peek ahead of the real dispatch, which evaluates the
        // operand itself - evaluating a `Move` operand here
        // would move the closure out of its place and leave the
        // dispatch reading an uninitialised place. Place
        // operands are read in place (the same read
        // `eval_callable` performs); constants are pure.
        let recv = match receiver {
            Operand::Copy(place) | Operand::Move(place) => {
                let Ok(resolved) = self.resolve_place(frames, body, place) else {
                    return false;
                };
                let Ok(value) = self.read_resolved(frames, &resolved) else {
                    return false;
                };
                value
            }
            Operand::Constant(c) => {
                let Ok(value) = self.eval_constant(c) else {
                    return false;
                };
                value
            }
        };
        let value = if by_value {
            recv
        } else {
            let Ok(resolved) = self.deref_value(frames, &recv) else {
                return false;
            };
            let Ok(value) = self.read_resolved(frames, &resolved) else {
                return false;
            };
            value
        };
        match value {
            Value::Closure { def, .. } => {
                self.program.async_closure_captures.contains_key(&def)
            }
            _ => false,
        }
    }

    pub(crate) fn eval_callable(
        &self,
        frames: &[Frame],
        body: &Body,
        operand: &Operand,
    ) -> InterpResult<(Value, Value)> {
        let value = match operand {
            Operand::Copy(place) | Operand::Move(place) => {
                let resolved = self.resolve_place(frames, body, place)?;
                self.read_resolved(frames, &resolved)?
            }
            Operand::Constant(c) => self.eval_constant(c)?,
        };
        let env_ref = self.callable_env_ref(frames, body, operand, &value)?;
        Ok((value, env_ref))
    }

    /// The environment reference for a callable value obtained from
    /// `operand`: a reference to the operand's place when it has one,
    /// otherwise a reference to the value itself (constant form).
    pub(crate) fn callable_env_ref(
        &self,
        frames: &[Frame],
        body: &Body,
        operand: &Operand,
        value: &Value,
    ) -> InterpResult<Value> {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => {
                match self.resolve_place(frames, body, place)? {
                    Resolved::Place {
                        frame_pos,
                        local,
                        projection,
                    } => Ok(Value::Ref(RefTarget::place(
                        frames[frame_pos].id,
                        local,
                        projection,
                    ))),
                    Resolved::Value(..) => {
                        Ok(Value::Ref(RefTarget::value(value.clone())))
                    }
                    Resolved::Static { name, projection } => {
                        Ok(Value::Ref(RefTarget::Static { name, projection }))
                    }
                    Resolved::Alloc { id, projection } => {
                        Ok(Value::Ref(RefTarget::Alloc { id, projection }))
                    }
                }
            }
            Operand::Constant(_) => Ok(Value::Ref(RefTarget::value(value.clone()))),
        }
    }

    /// Whether the converted body of the closure `def` takes its
    /// environment by value: its first parameter is the closure
    /// value itself, not a reference to it. That is the body shape
    /// produced for a closure whose own kind is `FnOnce` — the
    /// closure body is resolved with the closure's own kind, so no
    /// `FnOnce` adapter shim exists in the program, and the body
    /// reads its captures with plain field projections on the value
    /// (and may move them out of it).
    pub(crate) fn closure_body_takes_env_by_value(&self, def: &str) -> bool {
        match self.program.get(def) {
            Some(body) => matches!(body.local_ty(Local(1)), Some(Ty::Closure(..))),
            None => false,
        }
    }

    /// The environment argument for a call that consumes its
    /// callable (`FnOnce::call_once`, or an indirect call through a
    /// callable place) when the callee is the closure `value`:
    /// `Some` holding the closure value itself when the closure's
    /// body takes its environment by value. A `Move` operand's
    /// place is then left uninitialized — the value has moved into
    /// the callee — so the closure is really consumed: a later read
    /// of the place is an uninitialized read and a later drop of it
    /// runs no glue a second time, exactly as with any other move.
    /// `None` when the body takes a reference to the environment
    /// instead; the caller passes the environment reference it
    /// already built, which is also what the `call_once` shim does
    /// when it consumes an `Fn` / `FnMut` closure.
    pub(crate) fn consumed_env_arg(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        operand: &Operand,
        value: &Value,
    ) -> InterpResult<Option<Value>> {
        match value {
            Value::Closure { def, .. } if self.closure_body_takes_env_by_value(def) => {}
            _ => return Ok(None),
        }
        if let Operand::Move(place) = operand {
            if place.is_direct_local() {
                let frame = frames.last_mut().expect("frame");
                if let Some(slot) = frame.locals.get_mut(place.local.0) {
                    *slot = Value::Uninit;
                }
            } else if let Ok(target) = self.resolve_place(frames, body, place) {
                if matches!(target, Resolved::Place { .. }) {
                    self.write_resolved(frames, &target, Value::Uninit)?;
                }
            }
        }
        Ok(Some(value.clone()))
    }
}
