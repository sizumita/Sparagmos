//! Virtual dispatch resolution: scanning the program for
//! the concrete implementation behind a trait-object call,
//! and evaluating callable operands through it.

use super::lookup::{
    impl_self_matches, impl_self_matches_erased, normalize_impl_self, parse_impl_block_key,
    parse_impl_key,
};
use super::ref_shaped_receiver;
use super::unwrap_callable_wrapper;
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
        let mut ambiguous = false;
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
                ambiguous = true;
                break;
            }
            found = Some(key.clone());
        }
        if ambiguous {
            return None;
        }
        if let Some(key) = found {
            return Some(key);
        }
        // A last pass for a receiver whose rendered concrete
        // type dropped the regions its impl key spells (a
        // function pointer's signature above all: the key
        // spells `fn(&'static u8)`, the tag `fn(&u8)`), which
        // neither pass above can equate. The match is up to
        // region erasure, with binder-carrying keys excluded
        // (see `impl_self_matches_erased`); as in every pass
        // here, two distinct candidates resolve nothing.
        let mut erased: Option<String> = None;
        for key in keys() {
            let base = key.split('#').next().unwrap_or(key.as_str());
            let Some((self_part, trait_base, key_method)) = (if base.contains("::<impl ") {
                parse_impl_block_key(base)
            } else {
                parse_impl_key(base)
            }) else {
                continue;
            };
            if key_method != method
                || !paths.iter().any(|p| p == &trait_base)
                || !impl_self_matches_erased(&self_part, concrete)
            {
                continue;
            }
            if erased.as_deref() == Some(key.as_str()) {
                continue;
            }
            if erased.is_some() {
                return None;
            }
            erased = Some(key.clone());
        }
        erased
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
        let impl_name = self
            .scan_virtual_impl(concrete, trait_path, method)
            .or_else(|| {
                self.scan_virtual_impl_by_arity(concrete, trait_path, method, arg_values.len())
            })
            .or_else(|| self.plain_trait_instance(concrete, trait_path, method))?;
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

    /// The body registered under the plain trait-method path
    /// itself (`std::ops::Fn::call` and siblings), when that
    /// one registered instance's declared receiver type names
    /// exactly `concrete`: static demand registers the first
    /// demanded instance of a trait method under the plain
    /// name, and when the instance is this concrete type's own
    /// impl, its body is the callable's body even though no
    /// impl key names it. An instance of any other type is not
    /// this type's impl and never serves.
    pub(crate) fn plain_trait_instance(
        &self,
        concrete: &str,
        trait_path: &str,
        method: &str,
    ) -> Option<String> {
        let mut plains = vec![format!("{trait_path}::{method}")];
        if let Some(rest) = trait_path.strip_prefix("std::") {
            plains.push(format!("core::{rest}::{method}"));
        } else if let Some(rest) = trait_path.strip_prefix("core::") {
            plains.push(format!("std::{rest}::{method}"));
        }
        for plain in plains {
            let Some(body) = self.program.get(&plain) else {
                continue;
            };
            let mut ty = body.local_tys.get(1)?;
            while let Ty::Ref(inner) = ty {
                ty = inner;
            }
            // The registered receiver type is the generic-free
            // `Ty::Adt` rendering, while a `dyn` receiver's
            // concrete tag spells the full generic arguments:
            // the base name is the identity the two share.
            let base = concrete.split('<').next().unwrap_or(concrete);
            if matches!(ty, Ty::Adt(name) if name == base) {
                return Some(plain);
            }
        }
        None
    }

    /// The candidates for an `Fn`-family impl of `trait_path`
    /// for `concrete` whose converted body's argument-tuple
    /// parameter has exactly `argc` elements, when exactly one
    /// such candidate exists. A type can carry several impls of
    /// one `Fn` trait, one per argument tuple (the hand-written
    /// overload sets `unboxed_closures` permits), which the
    /// uniqueness scan above cannot tell apart: the call's own
    /// argument count is what separates them, and the impl
    /// body's declared tuple parameter is where that count is
    /// recorded. Bodies that are not converted cannot be
    /// arity-checked (or run), so they never serve here.
    pub(crate) fn scan_virtual_impl_by_arity(
        &self,
        concrete: &str,
        trait_path: &str,
        method: &str,
        argc: usize,
    ) -> Option<String> {
        let mut paths = vec![trait_path.to_string()];
        if let Some(rest) = trait_path.strip_prefix("std::") {
            paths.push(format!("core::{rest}"));
        } else if let Some(rest) = trait_path.strip_prefix("core::") {
            paths.push(format!("std::{rest}"));
        }
        let mut found: Option<String> = None;
        for key in self
            .program
            .functions
            .keys()
            .chain(self.program.unconverted.keys())
        {
            let base = key.split('#').next().unwrap_or(key.as_str());
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
            let Some(impl_body) = self.program.get(key) else {
                continue;
            };
            let arity = match impl_body.local_tys.get(2) {
                Some(Ty::Tuple(items)) => items.len(),
                Some(Ty::Unit) => 0,
                _ => continue,
            };
            if arity != argc {
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

    /// The concrete ADT type a `Fn`-trait call's receiver
    /// operand declares, when it declares one: the receiver
    /// place's local type, peeled through the reference a
    /// `call` / `call_mut` receiver carries (a lone `Deref`
    /// projection names the same pointee). The value model
    /// carries no type on an ADT value itself, so the declared
    /// type is the only name a direct (non-`dyn`) receiver's
    /// hand-written `Fn` impl can be looked up by. `None` for
    /// every other receiver shape, whose dispatch keeps its
    /// usual outcome.
    pub(crate) fn fn_receiver_concrete(&self, body: &Body, receiver: &Operand) -> Option<String> {
        let (Operand::Copy(place) | Operand::Move(place)) = receiver else {
            return None;
        };
        if !(place.projection.is_empty()
            || matches!(place.projection.as_slice(), [ProjectionElem::Deref]))
        {
            return None;
        }
        let mut ty = body.local_tys.get(place.local.0)?;
        while let Ty::Ref(inner) = ty {
            ty = inner;
        }
        match ty {
            Ty::Adt(name) => Some(name.clone()),
            _ => None,
        }
    }

    /// Dispatch a call that names an `Fn` / `FnMut` / `FnOnce`
    /// trait method by its plain trait path (a function-item
    /// value such as `Fn::<()>::call` taken and called through
    /// a function pointer, or a generic body the converter kept
    /// in generic form). The body registered under the plain
    /// name is one monomorphic instance of the method, which
    /// need not be the receiver's: the call's own shape --
    /// (receiver, argument tuple) -- is exactly the `CallTrait`
    /// shape, so the call is served by the same receiver
    /// dispatch instead of by entering that instance. `None`
    /// when the receiver does not resolve to a callable this
    /// dispatch serves, leaving the caller to enter the
    /// registered body as before.
    pub(crate) fn try_plain_fn_trait_call(
        &mut self,
        frames: &mut Vec<Frame>,
        func: &str,
        arg_values: &[Value],
        destination: &Place,
        target: Option<BasicBlock>,
    ) -> Option<InterpResult<()>> {
        let kind = plain_fn_trait_kind(func)?;
        if arg_values.len() != 2 {
            return None;
        }
        let args = match &arg_values[1] {
            Value::Tuple(items) => items.clone(),
            Value::Unit => Vec::new(),
            _ => return None,
        };
        let recv = arg_values[0].clone();
        let mut dyn_concrete: Option<String> = None;
        let (callable, env_ref) = match kind {
            FnCallKind::CallOnce => match recv {
                Value::DynRef { .. } => {
                    if let Value::DynRef { concrete, .. } = &recv {
                        dyn_concrete = Some(concrete.clone());
                    }
                    self.dyn_callable(frames, &recv).ok()?
                }
                Value::Ref(_) | Value::FatRef { .. } => {
                    let env = Value::Ref(RefTarget::value(recv.clone()));
                    self.callable_through_refs(frames, recv, env).ok()?
                }
                value => {
                    let (value, unwrapped) = unwrap_callable_wrapper(value);
                    let env = if unwrapped {
                        Value::Ref(RefTarget::value(value.clone()))
                    } else if matches!(
                        &value,
                        Value::Closure { def, .. }
                            if self.closure_body_takes_env_by_value(def)
                    ) {
                        value.clone()
                    } else {
                        Value::Ref(RefTarget::value(value.clone()))
                    };
                    (value, env)
                }
            },
            FnCallKind::Call | FnCallKind::CallMut => match recv {
                Value::DynRef { .. } => {
                    if let Value::DynRef { concrete, .. } = &recv {
                        dyn_concrete = Some(concrete.clone());
                    }
                    self.dyn_callable(frames, &recv).ok()?
                }
                Value::Ref(_) | Value::FatRef { .. } => {
                    let resolved = self.deref_value(frames, &recv).ok()?;
                    let mut value = self.read_resolved(frames, &resolved).ok()?;
                    let mut direct_ref = recv.clone();
                    for _ in 0..8 {
                        match &value {
                            Value::Ref(_) | Value::FatRef { .. } => {
                                let t = self.deref_value(frames, &value).ok()?;
                                let v = self.read_resolved(frames, &t).ok()?;
                                direct_ref = value.clone();
                                value = v;
                            }
                            _ => break,
                        }
                    }
                    if matches!(value, Value::DynRef { .. }) {
                        if let Value::DynRef { concrete, .. } = &value {
                            dyn_concrete = Some(concrete.clone());
                        }
                        self.dyn_callable(frames, &value).ok()?
                    } else {
                        let (value, unwrapped) = unwrap_callable_wrapper(value);
                        if unwrapped {
                            let env = Value::Ref(RefTarget::value(value.clone()));
                            (value, env)
                        } else {
                            (value, direct_ref)
                        }
                    }
                }
                value => {
                    let (value, _unwrapped) = unwrap_callable_wrapper(value);
                    let env = Value::Ref(RefTarget::value(value.clone()));
                    (value, env)
                }
            },
            FnCallKind::AsyncCall | FnCallKind::AsyncCallMut | FnCallKind::AsyncCallOnce => {
                return None
            }
        };
        match callable {
            Value::Closure { def, .. } => {
                // An async closure's call belongs to the
                // `AsyncFn` machinery, not this dispatch.
                if self.program.async_closure_captures.contains_key(&def) {
                    return None;
                }
                let mut full = vec![env_ref];
                full.extend(args);
                Some(self.start_call(frames, &def, full, destination, target))
            }
            Value::FnPtr(name) => {
                if plain_fn_trait_kind(&name).is_some() {
                    None
                } else {
                    Some(self.start_call(frames, &name, args, destination, target))
                }
            }
            other => {
                if let Some(exposed) = self.exposed_callable(&other) {
                    return match exposed {
                        Value::Closure { def, .. } => {
                            let mut full = vec![env_ref];
                            full.extend(args);
                            Some(self.start_call(frames, &def, full, destination, target))
                        }
                        Value::FnPtr(name) => {
                            Some(self.start_call(frames, &name, args, destination, target))
                        }
                        _ => None,
                    };
                }
                let concrete = dyn_concrete?;
                self.dyn_adt_fn_call(
                    frames,
                    kind,
                    &concrete,
                    other,
                    env_ref,
                    args,
                    destination,
                    target,
                )
            }
        }
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

    /// Whether the receiver of an `AsyncFn`-family trait call
    /// is a function pointer value (a function item, or a
    /// non-capturing closure coerced to a function pointer):
    /// such a receiver has no closure environment and no
    /// coroutine of its own for the call to construct - the
    /// `AsyncFn` shim's meaning for it is the plain function
    /// call, whose return value is the future. The peek reads
    /// the receiver without consuming it, exactly as
    /// `fn_call_receiver_is_async_closure` does. For
    /// `async_call_once` the receiver is the value itself; for
    /// `async_call` / `async_call_mut` it is a reference to the
    /// value.
    pub(crate) fn fn_call_receiver_is_fn_ptr(
        &self,
        frames: &mut Vec<Frame>,
        body: &Body,
        receiver: &Operand,
        by_value: bool,
    ) -> bool {
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
        matches!(value, Value::FnPtr(_))
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

/// The `Fn`-family call kind a plain trait-method path names
/// (`std::ops::Fn::call` and its `core::` spelling, and the
/// `FnMut` / `FnOnce` siblings), when the path is exactly the
/// trait method's own -- a resolved instance key
/// (`<T as Trait>::call`) is not a plain path and returns
/// `None`.
pub(crate) fn plain_fn_trait_kind(func: &str) -> Option<FnCallKind> {
    let base = func.split('#').next().unwrap_or(func);
    match base {
        "std::ops::Fn::call" | "core::ops::Fn::call" => Some(FnCallKind::Call),
        "std::ops::FnMut::call_mut" | "core::ops::FnMut::call_mut" => Some(FnCallKind::CallMut),
        "std::ops::FnOnce::call_once" | "core::ops::FnOnce::call_once" => {
            Some(FnCallKind::CallOnce)
        }
        _ => None,
    }
}
