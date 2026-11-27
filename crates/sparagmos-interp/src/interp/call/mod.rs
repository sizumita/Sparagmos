//! Call resolution and invocation.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{RefProj, RefTarget, Value};

mod lookup;
mod virtual_scan;

pub(crate) use lookup::{
    impl_qualified_self_type, parse_coroutine_key, parse_coroutine_tag, split_generic_args,
    split_top_level_str, tag_sequence_elem,
};
use lookup::{elided_self_spellings, impl_path_primitive_op, impl_path_trait_method};

/// The value-semantics trait method an impl-path spelling of
/// a call denotes: the name ends in `>::METHOD` and names the
/// trait either qualified (`... as std::clone::Clone>::clone`)
/// or in an impl header (`<impl std::clone::Clone for i32>`).
/// `None` for any other shape.
/// Whether a value is a caller-location stamp: the `Location`
/// aggregate — (file, line, column) — or a reference to it, as
/// the driver-appended stamp of a `#[track_caller]` call
/// evaluates.
fn is_caller_location_value(v: &Value) -> bool {
    let inner = match v {
        Value::Ref(RefTarget::Value(inner, _)) => &**inner,
        other => other,
    };
    let Value::Enum { fields, .. } = inner else {
        return false;
    };
    if fields.len() < 3 {
        return false;
    }
    let file_is_str = match &fields[0] {
        Value::Str(_) => true,
        Value::Enum { fields: inner, .. } => {
            matches!(inner.first(), Some(Value::Str(_)))
        }
        _ => false,
    };
    file_is_str
        && matches!(fields[1], Value::Int(_) | Value::Uint(_))
        && matches!(fields[2], Value::Int(_) | Value::Uint(_))
}

impl<'a> Interpreter<'a> {
    /// The error for naming a function that has no body in the
    /// program: the recorded conversion failure when the function
    /// exists but could not be converted, a plain unknown-function
    /// error otherwise.
    pub(crate) fn missing_body_error(&self, name: &str) -> InterpError {
        match self.program.unconverted_reason(name) {
            Some(reason) => InterpError::UnconvertedFunction {
                name: name.to_string(),
                reason: reason.to_string(),
            },
            None => InterpError::UnknownFunction {
                name: name.to_string(),
            },
        }
    }

    /// The builtin a body-less call to `func` may be served by
    /// at run time, with the name's canonical form: `func` names
    /// a trait method by its plain trait path (a function value
    /// such as `.map(Clone::clone)`, or a call inside a body the
    /// converter kept in generic form), the program holds no body
    /// under that name — the driver collects none for a
    /// builtin-owned definition, and conversion had no impl
    /// instance to resolve the name to — and the builtin of the
    /// same name is one of the value-semantics models (see
    /// `builtins::runtime_name_fallback`). `None` when a body
    /// exists (a collected body wins, as everywhere) or no such
    /// model applies.
    pub(crate) fn builtin_name_fallback(
        &self,
        func: &str,
    ) -> Option<(String, crate::builtins::Builtin, Vec<String>)> {
        if self.program.get(func).is_some()
            || self.program.fn_link_aliases.contains_key(func)
        {
            return None;
        }
        let canonical = crate::builtins::canonical_name(func);
        if let Some(kind) = crate::builtins::lookup_qualified(&canonical) {
            if crate::builtins::runtime_name_fallback(kind) {
                return self.fallback_generic_args(kind, &canonical)
                    .map(|args| (canonical, kind, args));
            }
            return None;
        }
        // A body-less foreign callee named by a function value
        // (an extern declaration reified as a function pointer)
        // is served by the foreign-helper models: they compute
        // from the argument values alone, like the
        // value-semantics models above.
        if let Some(kind) = crate::builtins::foreign_helper_lookup(&canonical) {
            return Some((canonical, kind, Vec::new()));
        }
        // A name the converter kept in generic form carries the
        // turbofish spelling of its type arguments
        // (`std::vec::Vec::<T>::new`); the lookup's impl-path
        // rules are written against the instantiated spelling
        // (`std::vec::Vec<T>::new`). Retry with the turbofish
        // markers folded away.
        if canonical.contains("::<") {
            let folded = canonical.replace("::<", "<");
            if let Some(kind) = crate::builtins::lookup_qualified(&folded) {
                if crate::builtins::runtime_name_fallback(kind) {
                    return self
                        .fallback_generic_args(kind, &folded)
                        .map(|args| (folded, kind, args));
                }
            }
        }
        // The same trait methods spelled by an impl path: a
        // qualified name (`<i32 as std::clone::Clone>::clone`)
        // or an impl-block path (`std::clone::impls::<impl
        // std::clone::Clone for i32>::clone`). The lookup's
        // suffix rules are written against the trait-path
        // spelling only; the value-semantics models answer for
        // every spelling alike.
        if let Some(kind) = impl_path_trait_method(&canonical) {
            return Some((canonical, kind, Vec::new()));
        }
        // The same for the primitive arithmetic traits
        // (`<i32 as std::ops::Add>::add`): their core impls are
        // the operations themselves at the Self type the name
        // carries.
        if let Some(kind) = impl_path_primitive_op(&canonical) {
            return Some((canonical, kind, Vec::new()));
        }
        // An impl-qualified `Default::default` name (`<i32 as
        // std::default::Default>::default`). The global lookup
        // deliberately has no suffix rule for this shape —
        // conversion must keep collecting real impl bodies where
        // it can — but when no body was collected, the Self type
        // the name itself carries is everything the model needs.
        if canonical.ends_with(">::default")
            && canonical.contains("::Default>::default")
        {
            if let Some(self_ty) = impl_qualified_self_type(&canonical) {
                return Some((
                    canonical,
                    crate::builtins::Builtin::DefaultDefault,
                    vec![self_ty],
                ));
            }
        }
        // The same shape for the specialization entry point
        // every Vec collect delegates to
        // (`<std::vec::Vec<T> as std::vec::spec_from_iter_nested::
        // SpecFromIterNested>::from_iter`): the name carries the
        // Vec target, and the nested-collect model computes from
        // the iterator value alone (an empty iterator tag: the
        // generic transition reads the state value's shape).
        if canonical.ends_with(">::from_iter")
            && canonical.contains("::spec_from_iter_nested::SpecFromIterNested")
        {
            if let Some(target) = impl_qualified_self_type(&canonical) {
                if target.starts_with("std::vec::Vec<")
                    || target.starts_with("alloc::vec::Vec<")
                {
                    return Some((
                        canonical,
                        crate::builtins::Builtin::VecFromIterNested,
                        vec![String::new()],
                    ));
                }
            }
        }
        // An impl-qualified `FromIterator::from_iter` name
        // (`<std::vec::Vec<T> as std::iter::FromIterator<T>>::
        // from_iter`): the caller stayed in generic form, so no
        // impl instance was ever resolved and the plain trait-path
        // lookup never saw the name. The name itself carries the
        // target collection, and the collect model computes from
        // the iterator value alone (the iterator type tag only
        // selects special state machines; the generic transition
        // reads the state value), so serve it with the trait
        // path's builtin for the targets the model has a shape
        // for.
        if canonical.ends_with(">::from_iter")
            && (canonical.contains("::FromIterator<")
                || canonical.contains("::FromIterator>::"))
        {
            if let Some(target) = impl_qualified_self_type(&canonical) {
                if target.starts_with("std::vec::Vec<")
                    || target.starts_with("alloc::vec::Vec<")
                    || target == "std::string::String"
                    || target == "alloc::string::String"
                {
                    return Some((
                        canonical,
                        crate::builtins::Builtin::FromIteratorFromIter,
                        vec![target, String::new(), String::new()],
                    ));
                }
            }
        }
        None
    }

    /// The generic arguments a run-time name fallback executes
    /// with. The allowlisted models ignore them — all but
    /// `Default::default`, whose model needs the Self type: an
    /// impl-qualified name (`<i32 as std::default::Default>::
    /// default`) carries it, the plain trait path does not, and
    /// there the fallback does not apply (`None`).
    fn fallback_generic_args(
        &self,
        kind: crate::builtins::Builtin,
        canonical: &str,
    ) -> Option<Vec<String>> {
        if kind != crate::builtins::Builtin::DefaultDefault {
            return Some(Vec::new());
        }
        impl_qualified_self_type(canonical).map(|t| vec![t])
    }

    /// Complete a call served by the run-time name fallback:
    /// execute the builtin on the evaluated arguments and write
    /// its result to the call's destination, exactly as a
    /// `CallBuiltin` terminator completes.
    pub(crate) fn complete_builtin_name_fallback(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        canonical: &str,
        kind: crate::builtins::Builtin,
        generic_args: &[String],
        func: &str,
        arg_values: Vec<Value>,
        destination: &Place,
        target: Option<BasicBlock>,
    ) -> InterpResult<()> {
        // A call the conversion kept as a plain call can still
        // carry the driver-appended caller-location stamp (a
        // `#[track_caller]` trait method's call); a builtin that
        // does not consume the location is served the declared
        // arguments only, as in the `CallBuiltin` conversion.
        let mut arg_values = arg_values;
        if !crate::builtins::consumes_caller_location(kind)
            && arg_values.len() > 1
            && arg_values.last().is_some_and(is_caller_location_value)
        {
            arg_values.pop();
        }
        let result = self.exec_builtin_values(
            frames,
            body,
            &[],
            kind,
            canonical,
            generic_args,
            None,
            arg_values,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;
        match target {
            Some(bb) => {
                self.write_place(frames, body, destination, result)?;
                Self::jump(frames, body, bb)
            }
            None => Err(InterpError::MissingContinuation {
                func: func.to_string(),
            }),
        }
    }

    /// A vtable dispatch the program holds no impl body for,
    /// whose method the file-io model serves for the receiver's
    /// concrete type. The standard streams' `Write` impls are
    /// the case that arises: `<std::io::Stdout as std::io::Write>::
    /// write` has no MIR body in the prebuilt library at all, so
    /// no demand pass can put one in the program, while the same
    /// call with a statically known receiver is served by the
    /// captured-stream arms of `exec_file_io` (see
    /// `file_io_terminator` in `mir_convert`). The model is keyed
    /// on exactly the pair the fat pointer carries - the method's
    /// definition name and the concrete type - so the dispatch
    /// can be served the same way once the vtable lookup has
    /// established that no body exists. Returns the name, the
    /// generic arguments (the concrete type alone) and the
    /// receiver (the fat pointer's data pointer, as a vtable
    /// dispatch hands it over) when the dispatch is servable
    /// this way.
    pub(crate) fn dyn_file_io_target(
        &self,
        func: &str,
        arg_values: &[Value],
    ) -> Option<(String, Vec<String>, Value)> {
        let Value::DynRef {
            base,
            concrete,
            traits,
            ..
        } = arg_values.first()?
        else {
            return None;
        };
        let (trait_path, method) = func.rsplit_once("::")?;
        if trait_path.is_empty() || method.is_empty() {
            return None;
        }
        // The vtable's trait list and the callee name can
        // render the same I/O trait with either prefix
        // (`std::io::Write` / `core::io::Write`); compare
        // with the prefix normalised away.
        let spell = |p: &str| {
            p.strip_prefix("core::io::")
                .map(|rest| format!("std::io::{rest}"))
                .unwrap_or_else(|| p.to_string())
        };
        if !traits.iter().any(|t| spell(t) == spell(trait_path)) {
            return None;
        }
        let generic_args = vec![concrete.clone()];
        if !crate::builtins::is_file_io_call(func, &generic_args) {
            return None;
        }
        Some((func.to_string(), generic_args, (**base).clone()))
    }

    /// Complete a call served by the dyn file-io lookup: run
    /// the file-io model on the evaluated arguments, with the
    /// fat pointer's data pointer as the receiver, and write
    /// its result to the call's destination, exactly as a
    /// `CallBuiltin` terminator for the same call completes.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn complete_dyn_file_io(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        name: &str,
        generic_args: &[String],
        receiver: Value,
        mut arg_values: Vec<Value>,
        destination: &Place,
        target: Option<BasicBlock>,
    ) -> InterpResult<()> {
        if let Some(first) = arg_values.first_mut() {
            *first = receiver;
        }
        let result = self.exec_file_io(frames, name, generic_args, arg_values)?;
        match target {
            Some(bb) => {
                self.write_place(frames, body, destination, result)?;
                Self::jump(frames, body, bb)
            }
            None => Err(InterpError::MissingContinuation {
                func: name.to_string(),
            }),
        }
    }

    /// Recover the receiver of a trait-method call whose MIR
    /// operand is the dereference of a fat-pointer place
    /// (`move (*_9)`, `_9: *const dyn Trait`): the shape of a
    /// by-value `Self` vtable call, where the compiler's shim
    /// loads the value out of the data pointer before calling
    /// the method body, so the evaluated argument is the concrete
    /// value alone and the (concrete type, trait) pair the
    /// dispatch lookup needs stayed on the pointer the operand
    /// dereferenced. When the head place's value is a `dyn` fat
    /// pointer and the call does not already dispatch, rewrite
    /// the first argument as that fat pointer with the loaded
    /// value as its data pointer, provided the rewritten call
    /// resolves as a dispatch: the lookup reads the vtable
    /// identity from the fat pointer, and the receiver shaping
    /// in `start_call` then hands the impl body the value itself
    /// (a by-value `self`) or a reference to it (a `&self`).
    /// Any other shape leaves the arguments untouched, so the
    /// call keeps the outcome it had.
    pub(crate) fn recover_deref_dyn_receiver(
        &self,
        frames: &[Frame],
        body: &Body,
        func: &str,
        args: &[Operand],
        arg_values: &mut [Value],
    ) {
        let Some(Operand::Move(place) | Operand::Copy(place)) = args.first() else {
            return;
        };
        if !matches!(place.projection.last(), Some(ProjectionElem::Deref)) {
            return;
        }
        if self.virtual_call_target(func, arg_values).is_some() {
            return;
        }
        let head = Place {
            local: place.local,
            projection: place.projection[..place.projection.len() - 1].to_vec(),
        };
        let Ok(head_value) = self.read_place(frames, body, &head) else {
            return;
        };
        let Value::DynRef {
            concrete,
            traits,
            drop_glue,
            type_id,
            ..
        } = head_value
        else {
            return;
        };
        let Some(loaded) = arg_values.first().cloned() else {
            return;
        };
        let swapped = Value::DynRef {
            base: Box::new(loaded),
            concrete,
            traits,
            drop_glue,
            // The swap re-bases the same concrete type on the
            // loaded pointee: its TypeId segments carry over.
            type_id,
        };
        let mut probe = arg_values.to_vec();
        probe[0] = swapped.clone();
        if matches!(self.virtual_call_target(func, &probe), Some(Ok(_))) {
            arg_values[0] = swapped;
        }
    }

    /// Resolve a call that has no body under its own name against
    /// a `dyn` receiver: the vtable lookup, with the fat pointer's
    /// (concrete type, trait) pair standing in for the vtable.
    ///
    /// Returns `None` when the call is not a dynamic dispatch (the
    /// first argument is not a dyn fat pointer, or the callee name
    /// is not a trait method of the pointer's bounds and no impl
    /// body matches). Returns `Some(Ok((impl_name, receiver)))` to
    /// dispatch: the receiver is the fat pointer's data pointer.
    /// Returns `Some(Err(_))` when the call is a dispatch through
    /// the pointer but the concrete implementation body is not in
    /// the program — the explicit boundary of the dyn model, hit at
    /// the call, never at the value's declaration.
    pub(crate) fn virtual_call_target(
        &self,
        func: &str,
        args: &[Value],
    ) -> Option<Result<(String, Value), InterpError>> {
        // The receiver of a trait call is the trait object, but a
        // method whose self type is a wrapper reaches here with
        // the object still inside it: `Future::poll` takes
        // `Pin<&mut Self>`, so the argument is the one-field `Pin`
        // aggregate around the `DynRef`. See through one-field
        // aggregates (a bounded walk: the wrappers are one-field
        // newtypes) to the object; the layers are remembered so
        // the receiver handed to the impl body can be rebuilt in
        // the callee's own self-type shape below.
        let mut layers: Vec<u32> = Vec::new();
        let mut cursor = args.first()?;
        while layers.len() <= 4 {
            match cursor {
                Value::Enum { variant, fields } if fields.len() == 1 => {
                    layers.push(*variant);
                    cursor = &fields[0];
                }
                _ => break,
            }
        }
        let Value::DynRef {
            base,
            concrete,
            traits,
            ..
        } = cursor
        else {
            return None;
        };
        // The receiver the impl body is handed: with no wrapper
        // layers, the fat pointer's data pointer as it stands.
        // With layers, the callee's self type is those same
        // wrappers around the *concrete* data pointer, so the
        // box-shaped base is stripped down to the pointer it
        // carries and the wrappers are rebuilt around it.
        let receiver = {
            let mut inner = (**base).clone();
            if !layers.is_empty() {
                while let Value::Enum { fields, .. } = &inner {
                    if fields.len() == 1 {
                        inner = fields[0].clone();
                    } else {
                        break;
                    }
                }
                for variant in layers.iter().rev() {
                    inner = Value::Enum {
                        variant: *variant,
                        fields: vec![inner],
                    };
                }
            }
            inner
        };
        let (trait_path, method) = func.rsplit_once("::")?;
        if trait_path.is_empty() || method.is_empty() {
            return None;
        }
        let candidate = format!("<{concrete} as {trait_path}>::{method}");
        if self.program.get(&candidate).is_some()
            || self.program.unconverted_reason(&candidate).is_some()
        {
            return Some(Ok((candidate, receiver)));
        }
        // The instance name a body is stored under elides the
        // concrete type's trailing *defaulted* generic arguments,
        // while the rendered type the fat pointer carries keeps
        // them (`Layered<L, S, S>` for a `Layered<L, S, N = S>`
        // whose stored key is `Layered<L, S>`). Try the spellings
        // with trailing arguments dropped, longest first; the
        // dropped arguments are the declared defaults, which the
        // remaining arguments determine, so at most one spelling
        // can be a program key.
        for elided in elided_self_spellings(concrete) {
            let candidate = format!("<{elided} as {trait_path}>::{method}");
            if self.program.get(&candidate).is_some()
                || self.program.unconverted_reason(&candidate).is_some()
            {
                return Some(Ok((candidate, receiver)));
            }
        }
        // The pointer's bounds name the trait *instance* the
        // pointer was formed for, generic arguments included
        // (`Bar<i32>`), while the call names the trait
        // definition alone. When the concrete type implements
        // the trait under several instantiations -- `Bar<i32>`
        // and `Bar<bool>` for one `S2`, reached through
        // different supertrait upcasts of one `dyn Foo` -- the
        // scan below cannot choose between the instances, but
        // the bound can: it is the vtable's identity. Try the
        // candidates the bounds spell (under the same
        // elided-`Self` spellings as above) before the scan;
        // several distinct hits are genuinely ambiguous and
        // fall through to it.
        let mut trait_paths = vec![trait_path.to_string()];
        if let Some(rest) = trait_path.strip_prefix("std::") {
            trait_paths.push(format!("core::{rest}"));
        } else if let Some(rest) = trait_path.strip_prefix("core::") {
            trait_paths.push(format!("std::{rest}"));
        }
        let mut instanced: Vec<String> = Vec::new();
        for entry in traits {
            let entry_path = entry.split('<').next().unwrap_or(entry.as_str());
            if !trait_paths.iter().any(|p| p == entry_path) {
                continue;
            }
            let mut self_spellings = vec![concrete.clone()];
            self_spellings.extend(elided_self_spellings(concrete));
            for self_spelling in self_spellings {
                let candidate = format!("<{self_spelling} as {entry}>::{method}");
                if (self.program.get(&candidate).is_some()
                    || self.program.unconverted_reason(&candidate).is_some())
                    && !instanced.contains(&candidate)
                {
                    instanced.push(candidate);
                }
            }
        }
        if instanced.len() == 1 {
            return Some(Ok((instanced.pop().unwrap(), receiver)));
        }
        // The impl key may carry the trait's generic arguments
        // (`<S as std::ops::FnMut<(i32,)>>::call_mut`) or spell the
        // trait through the other of `std::` / `core::`: scan the
        // program's keys for the unique matching impl body.
        if let Some(found) = self.scan_virtual_impl(concrete, trait_path, method) {
            return Some(Ok((found, receiver)));
        }
        // A coroutine's `Future::poll` (and likewise a
        // generator's `Coroutine::resume` and an async
        // generator's `AsyncIterator::poll_next`) is the
        // coroutine body itself: the compiler synthesizes the
        // impl from the coroutine transform, so the vtable's
        // method instance *is* the coroutine definition, and its
        // body is in the program under the coroutine's own name
        // - the name the concrete tag the value carries denotes
        // (a tag only a coroutine definition produces: it ends
        // in a `{closure#N}` segment no user type can be named
        // with), though the tag's spelling and the body key's
        // differ; see `coroutine_body_key`.
        if concrete.contains("::{closure#")
            && ((method == "poll"
                && matches!(trait_path, "std::future::Future" | "core::future::Future"))
                || (method == "resume"
                    && matches!(
                        trait_path,
                        "std::ops::Coroutine" | "core::ops::Coroutine"
                    ))
                || (method == "poll_next"
                    && matches!(
                        trait_path,
                        "std::async_iter::AsyncIterator"
                            | "core::async_iter::AsyncIterator"
                    )))
        {
            if let Some(key) = self.coroutine_body_key(concrete) {
                return Some(Ok((key, receiver)));
            }
        }
        // A method the compiler itself resolved without the
        // vtable -- a `final` trait method is the case in
        // point: it cannot be overridden, so a call through
        // the trait object names the trait's own body with
        // `Self` set to the trait object type, and conversion
        // stored exactly that body under the callee's plain
        // name (its receiver type is the trait object itself,
        // naming this trait). That body is not one concrete
        // type's instance standing in for a dispatch -- it
        // is the very instance the call denotes -- so decline
        // here and let the caller take the direct path with
        // the fat pointer as the receiver.
        if let Some(body) = self.program.get(func) {
            if body_receiver_is_trait_object(body, trait_path) {
                return None;
            }
        }
        if traits.iter().any(|t| t == trait_path) {
            return Some(Err(InterpError::UnsupportedDynamicDispatch {
                method: func.to_string(),
                concrete: concrete.clone(),
            }));
        }
        None
    }

    /// The program key of the coroutine body a `dyn` value's
    /// concrete tag denotes. The tag renders the coroutine
    /// *type*: the definition path with the `{closure#N}` segment
    /// last and the full generic-argument list appended - the
    /// enclosing definition's arguments first, then the coroutine
    /// signature's own (resume) arguments. A body key instead
    /// spells an *instance*: the enclosing definition's arguments
    /// ride on the definition's own segments
    /// (`gen::<i32>::{closure#0}`), and a definition with no
    /// arguments keeps the plain path (`me::{closure#0}`). The
    /// matching key is therefore the one whose plain parent path
    /// and closure segment are the tag's and whose argument list
    /// is a prefix of the tag's; the longest such list wins (an
    /// argument-carrying instance beats the uninstantiated
    /// spelling), and a tie is ambiguous. `None` when no
    /// converted body matches.
    fn coroutine_body_key(&self, concrete: &str) -> Option<String> {
        if self.program.get(concrete).is_some() {
            return Some(concrete.to_string());
        }
        let (tag_parent, tag_seg, tag_args) = parse_coroutine_tag(concrete)?;
        let mut best: Option<(usize, String)> = None;
        for key in self.program.functions.keys() {
            let Some((key_parent, key_seg, key_args)) = parse_coroutine_key(key) else {
                continue;
            };
            if key_seg != tag_seg || key_parent != tag_parent {
                continue;
            }
            if key_args.len() > tag_args.len()
                || !key_args.iter().zip(tag_args.iter()).all(|(k, t)| k == t)
            {
                continue;
            }
            match &best {
                Some((len, _)) if *len > key_args.len() => {}
                Some((len, _)) if *len == key_args.len() => return None,
                _ => best = Some((key_args.len(), key.clone())),
            }
        }
        best.map(|(_, key)| key)
    }

    /// Push a call frame for `func` with already-evaluated argument
    /// values (placed at locals 1..), returning into `destination` /
    /// `target` of the current top frame when the callee returns.
    /// Shared by direct calls, indirect calls and `Fn`-trait calls.
    pub(crate) fn start_call(
        &mut self,
        frames: &mut Vec<Frame>,
        func: &str,
        arg_values: Vec<Value>,
        destination: &Place,
        target: Option<BasicBlock>,
    ) -> InterpResult<()> {
        self.start_call_stamped(
            frames,
            func,
            arg_values,
            None,
            None,
            destination,
            target,
        )
    }

    /// The definition-location stamp value recorded for the
    /// track-caller body `func`, if the driver recorded one.
    fn def_stamp_value(&self, func: &str) -> Option<Value> {
        let c = self.program.track_caller_defs.get(func)?;
        self.eval_constant(c).ok()
    }

    /// The body serving a call to `func`: the body registered
    /// under the name itself, or -- for a name that denotes a
    /// body-less extern declaration -- the in-program
    /// definition exporting the declaration's link symbol
    /// (see `Program::fn_link_aliases`). The declaration's
    /// own name still keys its definition-location stamp, so
    /// stamp lookups keep using the name the call arrived
    /// under; only the body comes through the alias.
    pub(crate) fn serving_body(&self, func: &str) -> Option<&Body> {
        self.program.get(func).or_else(|| {
            self.program
                .fn_link_aliases
                .get(func)
                .and_then(|target| self.program.get(target))
        })
    }

    /// The program key of the body serving a call to `func`
    /// (see `serving_body`): the name itself when a body is
    /// registered under it, the link alias's target for a
    /// body-less declaration, and the name unchanged when
    /// neither serves (the call then fails naming it).
    fn serving_name<'s>(&'s self, func: &'s str) -> &'s str {
        if self.program.get(func).is_some() {
            func
        } else {
            self.program
                .fn_link_aliases
                .get(func)
                .map(String::as_str)
                .unwrap_or(func)
        }
    }

    /// The `Coroutine::resume` shim's location stamp: the
    /// location a track-caller coroutine body entered through
    /// a `dyn` receiver observes (see
    /// `ir::Program::coroutine_shim_location`).
    fn coroutine_shim_stamp_value(&self) -> Option<Value> {
        let c = self.program.coroutine_shim_location.as_ref()?;
        self.eval_constant(c).ok()
    }

    /// [`Interpreter::start_call`] with the caller-location
    /// stamps a call edge can carry beyond its argument list:
    /// `edge_stamp` is a call-site stamp (the stamp a
    /// `CallTrait` terminator carries for a track-caller
    /// closure), which forwards through track-caller callers
    /// like the driver-appended argument of a direct call;
    /// `fixed_stamp` names the shim the call passed through (an
    /// `Fn` vtable shim's location) and is bound as-is, never
    /// forwarded — the shim presents its own location, not the
    /// location it received.
    ///
    /// A marked callee ends up with exactly one stamp, from the
    /// first of these sources that has one: the appended
    /// argument of the call, the edge stamp, the fixed stamp,
    /// and finally the callee's definition location (the stamp
    /// of last resort for entries through a shim that fabricates
    /// one: a function-pointer call, a vtable dispatch to an
    /// impl-only `#[track_caller]` method).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start_call_stamped(
        &mut self,
        frames: &mut Vec<Frame>,
        func: &str,
        arg_values: Vec<Value>,
        edge_stamp: Option<Value>,
        fixed_stamp: Option<Value>,
        destination: &Place,
        target: Option<BasicBlock>,
    ) -> InterpResult<()> {
        // A `#[track_caller]` callee (marked at conversion) may
        // arrive with one extra argument: the caller-location
        // stamp the driver appended to the call. Pop it before
        // anything else inspects the arguments; it is bound to
        // the callee's synthetic location local below. The pop
        // is keyed on the body registered under `func`, which
        // for a trait-method call is one instance of the method:
        // the popped stamp is threaded through the vtable
        // dispatch below, so it reaches the impl body that
        // actually runs (and is simply dropped when that body
        // turns out unmarked, whose location the model does not
        // track).
        let mut arg_values = arg_values;
        let mut stamp = match self.serving_body(func) {
            Some(callee)
                if callee.caller_location_local.is_some()
                    && arg_values.len() == callee.arg_count + 1 =>
            {
                arg_values.pop()
            }
            _ => None,
        };
        let mut forwardable = stamp.is_some();
        if stamp.is_none() {
            if let Some(e) = edge_stamp {
                stamp = Some(e);
                forwardable = true;
            } else if let Some(f) = fixed_stamp {
                stamp = Some(f);
            }
        }
        // A C-variadic callee receives its tail arguments packed
        // as one va list (see `pack_variadic_args`); the vtable
        // dispatch below re-enters this function with the impl
        // body, where the packed shape is already the declared
        // one and passes through unchanged.
        if let Some(callee) = self.serving_body(func) {
            arg_values = self.pack_variadic_args(func, callee, arg_values);
        }
        // A call whose receiver is a `dyn` fat pointer is a vtable
        // dispatch even when a body exists under the callee's own
        // name: that body is one monomorphic instance of the trait
        // method (static demand registers the first demanded
        // instance of a trait method under the plain trait name),
        // not a dispatch — calling it directly would run that one
        // concrete type's instance for every receiver. The vtable
        // lookup therefore goes first; it returns `None` for calls
        // that are not dispatches through the receiver (a free
        // function taking a `&dyn` argument is the common shape),
        // which keep the direct lookup below.
        let callee = match self.virtual_call_target(func, &arg_values) {
            Some(vt) => match vt {
                // A vtable call: dispatch to the concrete type's
                // impl body, with the fat pointer's data pointer as
                // the receiver (the box aggregate itself for a
                // by-value `Box<dyn>` receiver).
                Ok((impl_func, receiver)) => {
                    // Shape the receiver the way the impl body's
                    // `self` type expects: a reference receiver gets
                    // a reference to the concrete value (through the
                    // box aggregate for a `Box<dyn>`), a by-value
                    // receiver gets the data pointer as it is.
                    let receiver = match self
                        .program
                        .get(&impl_func)
                        .and_then(|b| b.local_tys.get(1))
                    {
                        Some(Ty::Ref(..)) => ref_shaped_receiver(receiver),
                        _ => receiver,
                    };
                    let mut args = arg_values;
                    args[0] = receiver;
                    // A coroutine body entered through the
                    // `Coroutine::resume` vtable slot observes
                    // the shim's own location (the trait
                    // method's definition in core): `resume`
                    // is not `#[track_caller]`, so no caller
                    // location forwards through it, and an
                    // entry carrying no stamp presents the
                    // shim stamp as a fixed stamp.
                    let stamp = if stamp.is_none()
                        && impl_func.contains("::{closure#")
                        && func.rsplit_once("::").is_some_and(|(tp, m)| {
                            m == "resume"
                                && matches!(
                                    tp,
                                    "std::ops::Coroutine" | "core::ops::Coroutine"
                                )
                        })
                    {
                        self.coroutine_shim_stamp_value().or(stamp)
                    } else {
                        stamp
                    };
                    let (edge, fixed) = if forwardable {
                        (stamp, None)
                    } else {
                        (None, stamp)
                    };
                    return self.start_call_stamped(
                        frames,
                        &impl_func,
                        args,
                        edge,
                        fixed,
                        destination,
                        target,
                    );
                }
                Err(e) => return Err(e),
            },
            None => match self.serving_body(func) {
                Some(c) => c,
                None => return Err(self.missing_body_error(func)),
            },
        };
        if frames.len() >= self.config.max_call_depth {
            return Err(InterpError::CallDepthExceeded {
                limit: self.config.max_call_depth,
            });
        }
        if arg_values.len() != callee.arg_count {
            return Err(InterpError::type_mismatch(
                format!("call to `{func}`"),
                format!("{} argument(s)", callee.arg_count),
                format!("{} argument(s)", arg_values.len()),
            ));
        }
        // No stamp on the edge at all: the call reached this
        // body through a shim that fabricates the callee's
        // definition location (a function-pointer call, a
        // vtable dispatch to an impl-only `#[track_caller]`
        // method), so bind the definition stamp the driver
        // recorded for the body.
        if callee.caller_location_local.is_some() && stamp.is_none() {
            stamp = self.def_stamp_value(func);
        }
        let mut locals = vec![Value::Uninit; callee.local_count()];
        for (i, v) in arg_values.into_iter().enumerate() {
            locals[i + 1] = v;
        }
        if let (Some(idx), Some(mut stamp)) = (callee.caller_location_local, stamp) {
            // When the calling frame is itself a track-caller
            // frame, a call-site stamp forwards — it is replaced
            // by the caller's own received location — matching
            // the compiler's propagation (a call written in a
            // plain function inlined into a track-caller body is
            // the one case this runtime forwarding cannot tell
            // apart; the stamp then still names a location in
            // the call chain, one frame further out). A shim
            // stamp (a fixed stamp, or a definition stamp bound
            // above) does not forward: the shim is a frame of
            // its own, and it presents its own location.
            if forwardable {
                if let Some(caller) = frames.last() {
                    if let Some(caller_idx) = self
                        .program
                        .get(&caller.func)
                        .and_then(|b| b.caller_location_local)
                    {
                        // A caller whose location local was
                        // never bound (a body entered with no
                        // stamp from any source) has no location
                        // to forward: the stamp stands.
                        let received = &caller.locals[caller_idx];
                        if !matches!(received, Value::Uninit) {
                            stamp = received.clone();
                        }
                    }
                }
            }
            locals[idx] = stamp;
        }
        let stack_bytes = self.charge_stack(frames, self.frame_stack_bytes(callee))?;
        let frame_id = self.fresh_frame_id();
        let frame_func = self.serving_name(func);
        self.register_frame_base(&frames, frame_id, frame_func);
        frames.push(Frame {
            id: frame_id,
            func: frame_func.to_string(),
            locals,
            stack_bytes,
            block: 0,
            stmt: 0,
            return_dest: Some(destination.clone()),
            return_block: target,
            unwind_payload: None,
            in_cleanup: false,
            call_once_env_drop: None,
            incoming_asm: None,
        });
        Ok(())
    }

    /// Push a frame for `func` with already-evaluated arguments and
    /// no return destination: the frame belongs to a nested
    /// [`Interpreter::drive`] region whose stop depth is the stack
    /// depth before the push, and its return value is that drive's
    /// result. Used by runtime support that calls back into guest
    /// code (the `catch_unwind` intrinsic's try/catch functions).
    fn start_detached_call(
        &mut self,
        frames: &mut Vec<Frame>,
        func: &str,
        arg_values: Vec<Value>,
    ) -> InterpResult<()> {
        // The vtable lookup goes first, exactly as in `start_call`.
        let body = match self.virtual_call_target(func, &arg_values) {
            Some(vt) => match vt {
                // A vtable call, shaped exactly as in `start_call`.
                Ok((impl_func, receiver)) => {
                    let receiver = match self
                        .program
                        .get(&impl_func)
                        .and_then(|b| b.local_tys.get(1))
                    {
                        Some(Ty::Ref(..)) => ref_shaped_receiver(receiver),
                        _ => receiver,
                    };
                    let mut args = arg_values;
                    args[0] = receiver;
                    return self.start_detached_call(frames, &impl_func, args);
                }
                Err(e) => return Err(e),
            },
            None => match self.serving_body(func) {
                Some(b) => b,
                None => return Err(self.missing_body_error(func)),
            },
        };
        let arg_values = self.pack_variadic_args(func, body, arg_values);
        if arg_values.len() != body.arg_count {
            return Err(InterpError::type_mismatch(
                format!("call to {func}"),
                format!("{} argument(s)", body.arg_count),
                format!("{} argument(s)", arg_values.len()),
            ));
        }
        let mut locals = vec![Value::Uninit; body.local_count()];
        for (i, arg) in arg_values.into_iter().enumerate() {
            locals[i + 1] = arg;
        }
        // A detached entry is a shim entry (runtime support
        // calling back into guest code): a track-caller body
        // receives its definition stamp, as in `start_call`.
        if let Some(idx) = body.caller_location_local {
            if let Some(stamp) = self.def_stamp_value(func) {
                locals[idx] = stamp;
            }
        }
        let stack_bytes = self.charge_stack(frames, self.frame_stack_bytes(body))?;
        let incoming_asm = self.pending_asm_incoming.take();
        let id = self.fresh_frame_id();
        let frame_func = self.serving_name(func);
        self.register_frame_base(&frames, id, frame_func);
        frames.push(Frame {
            id,
            func: frame_func.to_string(),
            locals,
            stack_bytes,
            block: 0,
            stmt: 0,
            return_dest: None,
            return_block: None,
            unwind_payload: None,
            in_cleanup: false,
            call_once_env_drop: None,
            incoming_asm,
        });
        Ok(())
    }

    /// Run `f` as a detached synchronous drive: a builtin is
    /// executing guest code synchronously and waiting on the
    /// result host-side. While it runs, `sync_nesting` is
    /// raised, so a blocking operation inside cannot suspend
    /// the thread — the host-side continuation waiting for the
    /// drive's result (a builtin's local state, a detached
    /// frame's missing return destination) cannot be saved and
    /// resumed the way a thread's frames can.
    pub(crate) fn detached_drive<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> InterpResult<T>,
    ) -> InterpResult<T> {
        self.sync_nesting.set(self.sync_nesting.get() + 1);
        let result = f(self);
        self.sync_nesting.set(self.sync_nesting.get() - 1);
        result
    }

    /// Call `func` with evaluated arguments and run the nested
    /// drive region until it returns, yielding its return value.
    pub(crate) fn call_to_completion(
        &mut self,
        frames: &mut Vec<Frame>,
        func: &str,
        arg_values: Vec<Value>,
    ) -> InterpResult<Value> {
        let depth = frames.len();
        self.start_detached_call(frames, func, arg_values)?;
        self.detached_drive(|s| s.call_to_completion_region(frames, depth))
    }

    /// The drive region behind [`Self::call_to_completion`],
    /// without the detached-drive marking.
    fn call_to_completion_region(
        &mut self,
        frames: &mut Vec<Frame>,
        depth: usize,
    ) -> InterpResult<Value> {
        // A detached call is a scheduled region like any other
        // (see `drive_scheduled`): the try function of a
        // `catch_unwind`, a destructor body, or another runtime
        // callback may itself spawn threads and spin, so a yield
        // raised inside is served by a scheduling pass here and
        // the region resumes; the caller only ever observes the
        // region's own outcome.
        loop {
            let prev_segment = self.segment.replace(SegmentMark {
                depth: self.drive_depth + 1,
                thread: self.current_thread_id(),
                steps0: self.steps,
            });
            let result = self.drive(frames, depth);
            self.segment = prev_segment;
            match result {
                Err(InterpError::Yield) => {
                    self.drain_threads(frames)?;
                    continue;
                }
                other => return other,
            }
        }
    }

    /// Call `func` synchronously from within the interpreter
    /// (a builtin that must apply a callable it holds as a value,
    /// such as an iterator adapter's closure): push a frame for it
    /// on the caller's stack, drive the stack until that frame
    /// returns, and yield its return value. The caller's frames
    /// below are left exactly as they were. A guest panic inside
    /// the callee propagates as the call's error, as a nested call
    /// from guest code would. On the current drive structure this
    /// is [`Self::call_to_completion`]: a detached frame driven by
    /// a nested [`Self::drive`] region, unwinding included.
    pub(crate) fn call_sync(
        &mut self,
        frames: &mut Vec<Frame>,
        func: &str,
        arg_values: Vec<Value>,
    ) -> InterpResult<Value> {
        self.detached_drive(|s| s.call_sync_inner(frames, func, arg_values))
    }

    /// The drive behind [`Self::call_sync`], without the
    /// detached-drive marking.
    pub(crate) fn call_sync_inner(
        &mut self,
        frames: &mut Vec<Frame>,
        func: &str,
        arg_values: Vec<Value>,
    ) -> InterpResult<Value> {
        // A callable value naming a trait method by its plain
        // trait path — `.map(Clone::clone)` handed to an adapter
        // the builtins drive — has no body under that name; the
        // builtin of the same name serves it (see
        // `builtin_name_fallback`). The body handed to the
        // dispatch is the caller's: the allowlisted models never
        // read it (only the caller-location model does, and it is
        // not allowlisted).
        if let Some((canonical, kind, generic_args)) = self.builtin_name_fallback(func) {
            let program: &Program = self.program;
            if let Some(body) = frames.last().and_then(|f| program.get(&f.func)) {
                return self.exec_builtin_values(
                    frames,
                    body,
                    &[],
                    kind,
                    &canonical,
                    &generic_args,
                    None,
                    arg_values,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                );
            }
        }
        self.call_to_completion(frames, func, arg_values)
    }

    /// Apply a callable value (a closure or a function pointer)
    /// to `args` synchronously, via [`Self::call_sync`]. For a
    /// closure whose body takes its environment by reference,
    /// `env_ref` is the reference the body receives as its first
    /// argument: a caller that knows where the closure value lives
    /// (an iterator adapter's state field, say) passes a reference
    /// to that storage so `FnMut` capture writes persist in place;
    /// Whether the closure body `def` itself drops its
    /// environment (a `Drop` terminator rooted at the environment
    /// parameter, local 1): some pipeline shapes inline the
    /// environment's drop into the body — the captures are dropped
    /// through the environment reference at the end of the body —
    /// and then the `call_once` shim drop (see
    /// [`Frame::call_once_env_drop`]) must not run as well.
    pub(crate) fn closure_body_drops_env(&self, def: &str) -> bool {
        let Some(body) = self.program.get(def) else {
            return false;
        };
        body.blocks.iter().any(|block| {
            matches!(
                &block.terminator,
                TerminatorKind::Drop { place, .. } if place.local.0 == 1
            )
        })
    }

    /// The drop root for a `call_once` receiver: the receiver
    /// place resolved in the caller (the glue session then also
    /// clears the place, as MIR's drop flags do after the shim
    /// consumes the value), or the evaluated value itself when
    /// the receiver is not a place.
    pub(crate) fn call_once_receiver_root(
        &self,
        frames: &[Frame],
        body: &Body,
        receiver: &Operand,
        value: &Value,
    ) -> InterpResult<RefTarget> {
        if let Operand::Copy(place) | Operand::Move(place) = receiver {
            if let Ok(resolved) = self.resolve_place(frames, body, place) {
                return Ok(match resolved {
                    Resolved::Place {
                        frame_pos,
                        local,
                        projection,
                    } => RefTarget::place(frames[frame_pos].id, local, projection),
                    Resolved::Static { name, projection } => RefTarget::Static {
                        name,
                        projection,
                    },
                    Resolved::Alloc { id, projection } => RefTarget::Alloc {
                        id,
                        projection,
                    },
                    Resolved::Value(base, projection) => {
                        RefTarget::Value(Box::new(base), projection)
                    }
                });
            }
        }
        Ok(RefTarget::value(value.clone()))
    }

    /// with `None` the environment is a reference to the value
    /// itself, through which capture reads resolve and capture
    /// writes are the explicit constant-aggregate error. A body
    /// that takes its environment by value (an `FnOnce` closure's
    /// own body) receives the closure value itself, as with a
    /// direct `FnOnce::call_once`.
    pub(crate) fn call_callable_sync(
        &mut self,
        frames: &mut Vec<Frame>,
        callable: &Value,
        env_ref: Option<Value>,
        args: Vec<Value>,
    ) -> InterpResult<Value> {
        if frames.len() >= self.config.max_call_depth {
            return Err(InterpError::CallDepthExceeded {
                limit: self.config.max_call_depth,
            });
        }
        match callable {
            Value::FnPtr(name) => self.call_sync(frames, name, args),
            Value::Closure { def, .. } => {
                let mut full = Vec::with_capacity(args.len() + 1);
                if self.closure_body_takes_env_by_value(def) {
                    full.push(callable.clone());
                } else {
                    full.push(env_ref.unwrap_or_else(|| {
                        Value::Ref(RefTarget::value(callable.clone()))
                    }));
                }
                full.extend(args);
                self.call_sync(frames, def, full)
            }
            other => Err(InterpError::type_mismatch(
                "synchronous call",
                "callable value",
                other.kind_name(),
            )),
        }
    }

    /// Apply a callable that the caller knows by type
    /// (`callable_ty` is the rendered type of the callable value),
    /// synchronously. A closure or function pointer is applied
    /// exactly as in [`Self::call_callable_sync`]. An *empty ADT*
    /// value - the value form of a zero-sized type - carries no
    /// type information of its own, so a callable of that shape is
    /// dispatched through its type instead: the unique `Fn` /
    /// `FnMut` / `FnOnce` impl body the program holds for
    /// `callable_ty` is called with `(receiver, argument tuple)`,
    /// the receiver shaped as the impl body expects (a reference
    /// when its first parameter is one - the caller's environment
    /// reference when it has one - and the value itself when it
    /// takes the receiver by value). This is the dispatch the
    /// iterator adapter steppers need for the `impl_fn_for_zst!`
    /// family in core (`str::LinesMap` behind `str::lines`, and
    /// its siblings): those callables only ever reach a call
    /// through an adapter's builtin step, never through a call
    /// edge that would name their type. When no impl body matches,
    /// the value falls through to the untyped dispatch and its
    /// usual outcome.
    pub(crate) fn call_callable_sync_typed(
        &mut self,
        frames: &mut Vec<Frame>,
        callable: &Value,
        callable_ty: &str,
        env_ref: Option<Value>,
        args: Vec<Value>,
    ) -> InterpResult<Value> {
        let empty_adt = match callable {
            Value::Enum { fields, .. } => fields.is_empty(),
            Value::Tuple(fields) => fields.is_empty(),
            _ => false,
        };
        if empty_adt && !callable_ty.is_empty() {
            for (trait_path, method) in [
                ("std::ops::Fn", "call"),
                ("std::ops::FnMut", "call_mut"),
                ("std::ops::FnOnce", "call_once"),
            ] {
                let Some(impl_name) =
                    self.scan_virtual_impl(callable_ty, trait_path, method)
                else {
                    continue;
                };
                // A body that was demanded but not converted is
                // not a dispatch target: `scan_virtual_impl`
                // scans the unconverted set too, and calling one
                // would only re-report its absence less clearly
                // than the untyped path below.
                if self.program.get(&impl_name).is_none() {
                    continue;
                }
                let wants_ref = matches!(
                    self.program
                        .get(&impl_name)
                        .and_then(|b| b.local_tys.get(1)),
                    Some(Ty::Ref(..))
                );
                let receiver = if wants_ref {
                    match env_ref {
                        Some(
                            r @ (Value::Ref(_)
                            | Value::FatRef { .. }
                            | Value::DynRef { .. }),
                        ) => r,
                        _ => ref_shaped_receiver(callable.clone()),
                    }
                } else {
                    callable.clone()
                };
                let args = self.fn_args_adapted(&impl_name, args);
                let result = self.call_sync(
                    frames,
                    &impl_name,
                    vec![receiver, Value::Tuple(args)],
                )
                .map(fn_result_materialized);
                return result;
            }
        }
        self.call_callable_sync(frames, callable, env_ref, args)
    }

    /// Shape the argument values of a type-directed `Fn`-family
    /// dispatch (see [`Self::call_callable_sync_typed`]) as the
    /// impl body's argument tuple declares them: the adapter
    /// steppers hold a `&str` item as the string data itself and
    /// a slice item as its element tuple, but the body's tuple
    /// element is the fat pointer those data denote, and the
    /// body's own pointer arithmetic on a bare data value loses
    /// the string it slices. An argument whose declared element
    /// type is a reference is therefore handed over as the
    /// reference value for the data (a fat pointer for a `str` or
    /// slice element); every other argument passes through
    /// unchanged.
    fn fn_args_adapted(&self, impl_name: &str, args: Vec<Value>) -> Vec<Value> {
        let Some(elem_tys) = self
            .program
            .get(impl_name)
            .and_then(|b| b.local_tys.get(2))
            .and_then(|t| match t {
                Ty::Tuple(ts) => Some(ts.clone()),
                _ => None,
            })
        else {
            return args;
        };
        args.into_iter()
            .enumerate()
            .map(|(i, arg)| {
                let Some(elem_ty) = elem_tys.get(i) else {
                    return arg;
                };
                let Ty::Ref(pointee) = elem_ty else {
                    return arg;
                };
                match (&**pointee, &arg) {
                    (Ty::Str, Value::Str(text)) => Value::FatRef {
                        base: Box::new(Value::Ref(RefTarget::value(arg.clone()))),
                        len: text.len() as u128,
                    },
                    (Ty::Slice(_), Value::Tuple(items)) => Value::FatRef {
                        base: Box::new(Value::Ref(RefTarget::value(arg.clone()))),
                        len: items.len() as u128,
                    },
                    (Ty::Slice(_), Value::Str(text)) => Value::FatRef {
                        base: Box::new(Value::Ref(RefTarget::value(arg.clone()))),
                        len: text.len() as u128,
                    },
                    _ => arg,
                }
            })
            .collect()
    }
}

impl<'a> Interpreter<'a> {
    /// Model std's lazy stdout buffer allocation on the first
    /// use of the print support: the stdout handle's line buffer
    /// (1024 bytes) is allocated through the global allocator
    /// the first time the program prints, and never freed. A
    /// guest allocator observes exactly that one allocation;
    /// with no guest allocator there is nothing to model.
    pub(crate) fn note_stdout_use(
        &mut self,
        frames: &mut Vec<Frame>,
    ) -> InterpResult<()> {
        if self.stdout_buffer_modelled.get() {
            return Ok(());
        }
        self.stdout_buffer_modelled.set(true);
        let _ = self.call_guest_alloc(frames, 1024, 8)?;
        Ok(())
    }

    /// Call the callable denoted by `operand` (a closure or
    /// function value in the current frame) with `args`, driving
    /// the frame stack until the call returns, and return its
    /// return value. The environment reference is built from the
    /// operand exactly as for an indirect call, so a closure
    /// place's captures are mutated in place and the mutations
    /// persist across repeated calls — the `FnMut` semantics the
    /// iterator consumer builtins (`fold`, `for_each`) rely on.
    pub(crate) fn call_callable_operand_to_completion(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        operand: &Operand,
        args: Vec<Value>,
    ) -> InterpResult<Value> {
        self.detached_drive(|s| {
            s.call_callable_operand_to_completion_inner(frames, body, operand, args)
        })
    }

    /// The drive behind
    /// [`Self::call_callable_operand_to_completion`], without
    /// the detached-drive marking.
    pub(crate) fn call_callable_operand_to_completion_inner(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        operand: &Operand,
        args: Vec<Value>,
    ) -> InterpResult<Value> {
        let base = frames.len();
        let (callee, env_ref) = self.eval_callable(&frames, body, operand)?;
        let (callee, env_ref) = self.callable_through_refs(frames, callee, env_ref)?;
        let env_ref = match self.consumed_env_arg(frames, body, operand, &callee)? {
            Some(by_value) => by_value,
            None => env_ref,
        };
        match callee {
            Value::Closure { def, .. } => {
                let mut full = vec![env_ref];
                full.extend(args);
                self.start_call(frames, &def, full, &Place::local(Local::RETURN), None)?;
            }
            Value::FnPtr(name) => {
                self.start_call(frames, &name, args, &Place::local(Local::RETURN), None)?;
            }
            other => {
                return Err(InterpError::type_mismatch(
                    "callable operand",
                    "a closure or function pointer",
                    other.kind_name(),
                ));
            }
        }
        self.drive_frames(frames, base)
    }

    /// Follow a callable operand value that is a reference to a
    /// callable (`&F` / `&mut F`): the blanket `Fn` / `FnMut`
    /// impls for references make the pointee the callable, the
    /// shape std's forwarding bodies pass (a
    /// `vec_deque::Iter::fold` calls the inner slice iterator's
    /// `fold` with `&mut f`, and the modelled consumer receives
    /// that reference as its closure operand). The innermost
    /// reference -- the one denoting the callable value itself
    /// -- is the environment reference its body expects, so
    /// `FnMut` capture mutations persist in the original
    /// closure place, exactly as through the real shim (the
    /// deref chain the `CallTrait` dispatch follows for its
    /// receiver). A value that is not a reference, and a
    /// reference whose deref chain does not end at a callable,
    /// come back unchanged with the environment reference
    /// `eval_callable` computed, so the caller's own judgement
    /// of the value (and its error) is preserved.
    pub(crate) fn callable_through_refs(
        &self,
        frames: &[Frame],
        callee: Value,
        env_ref: Value,
    ) -> InterpResult<(Value, Value)> {
        let mut value = callee.clone();
        while matches!(value, Value::Ref(_) | Value::FatRef { .. }) {
            let Ok(target) = self.deref_value(frames, &value) else {
                return Ok((callee, env_ref));
            };
            let Ok(pointee) = self.read_resolved(frames, &target) else {
                return Ok((callee, env_ref));
            };
            match pointee {
                Value::Closure { .. } | Value::FnPtr(_) => {
                    return Ok((pointee, value));
                }
                Value::Ref(_) | Value::FatRef { .. } => {
                    value = pointee;
                }
                _ => return Ok((callee, env_ref)),
            }
        }
        Ok((callee, env_ref))
    }
}

/// Whether `v` holds a callable value anywhere inside it (a
/// closure, a function pointer, or a dyn callable).
fn value_holds_callable(v: &Value) -> bool {
    match v {
        Value::Closure { .. } | Value::FnPtr(_) | Value::DynRef { .. } => true,
        Value::Tuple(items) => items.iter().any(value_holds_callable),
        Value::Enum { fields, .. } => fields.iter().any(value_holds_callable),
        _ => false,
    }
}

/// Shape a dyn call's data pointer into the reference an impl
/// body's reference-typed `self` expects. A reference, fat or dyn
/// pointer already denotes the pointee and passes through as it
/// is. A one-field smart-pointer wrapper (a built `Box`
/// aggregate) denotes the pointee through its only field: when
/// that field is itself a pointer — an allocation pointer for a
/// built box, or any reference-shaped value — the field *is* the
/// reference to the pointee, and wrapping it in a value reference
/// instead would make the callee's deref yield the pointer itself,
/// with every further field projection then failing on a
/// reference where an aggregate belongs. A plain field value, and
/// any other bare value, is referenced as a value.
/// The value standing for the default panic hook (what
/// `take_hook` returns when no guest hook is installed): a
/// `dyn Fn` marker the runtime recognises instead of calling.
pub(crate) fn default_hook_value() -> Value {
    Value::DynRef {
        base: Box::new(Value::Unit),
        concrete: "std::panic::default_hook".to_string(),
        traits: vec![
            "std::ops::Fn".to_string(),
            "std::marker::Send".to_string(),
            "std::marker::Sync".to_string(),
        ],
        drop_glue: None,
        type_id: None,
    }
}

/// Whether a hook value is the default-hook marker.
pub(crate) fn is_default_hook(value: &Value) -> bool {
    matches!(value, Value::DynRef { concrete, .. } if concrete == "std::panic::default_hook")
}

/// The concrete tag of a panic hook produced by
/// `std::panic::update_hook`: a `dyn Fn` marker whose base
/// is the pair `(update closure, previous hook)`. Like the
/// default hook's tag, it is a model-side name no guest type
/// carries.
pub(crate) const UPDATED_HOOK_CONCRETE: &str = "std::panic::updated_hook";

/// The hook value `update_hook` installs: a `dyn Fn` marker
/// carrying the update closure and the hook it wraps.
pub(crate) fn updated_hook_value(update: Value, prev: Value) -> Value {
    Value::DynRef {
        base: Box::new(Value::Tuple(vec![update, prev])),
        concrete: UPDATED_HOOK_CONCRETE.to_string(),
        traits: vec![
            "std::ops::Fn".to_string(),
            "std::marker::Send".to_string(),
            "std::marker::Sync".to_string(),
        ],
        drop_glue: None,
        type_id: None,
    }
}

/// The `(update closure, previous hook)` an updated-hook
/// marker carries, when `value` is one.
pub(crate) fn updated_hook_parts(value: &Value) -> Option<(Value, Value)> {
    let Value::DynRef { base, concrete, .. } = value else {
        return None;
    };
    if concrete != UPDATED_HOOK_CONCRETE {
        return None;
    }
    let Value::Tuple(items) = &**base else {
        return None;
    };
    let [update, prev] = items.as_slice() else {
        return None;
    };
    Some((update.clone(), prev.clone()))
}

/// Whether a hook value is a model-side panic-hook marker
/// (the default hook or an `update_hook` chain link): no
/// guest body serves a call on it; the hook machinery does
/// (see `Interpreter::invoke_panic_hook`).
pub(crate) fn is_panic_hook_marker(value: &Value) -> bool {
    is_default_hook(value) || updated_hook_parts(value).is_some()
}

/// A reference to field `idx` of the value `recv` references:
/// the same target with one more field projection. `None` when
/// `recv` is not a reference a field can be taken of. This is
/// the reference the `AsyncFn` coroutine creation forms for each
/// capture: the future's upvar for capture `i` is a reference to
/// the closure environment's field `i`, exactly as the
/// compiler's shim body forms it with `&((*env).i)`.
pub(crate) fn ref_to_field(recv: &Value, idx: usize) -> Option<Value> {
    let Value::Ref(target) = recv else {
        return None;
    };
    let mut target = target.clone();
    match &mut target {
        RefTarget::Place { projection, .. }
        | RefTarget::Value(_, projection)
        | RefTarget::Static { projection, .. }
        | RefTarget::Alloc { projection, .. } => {
            projection.push(RefProj::Field(idx));
        }
        RefTarget::Null | RefTarget::Address(_) => return None,
    }
    Some(Value::Ref(target))
}

/// A reference to field `idx` of the pointee of the reference
/// `recv`: the same target with one more field projection. This
/// is the reference an `AsyncFn` coroutine creation forms for a
/// deref capture: the closure environment holds one reference to
/// an aggregate, and the coroutine borrows the aggregate's fields
/// disjointly, each through that reference. `None` when `recv`
/// is not a reference a field can be taken of.
pub(crate) fn ref_projected(recv: &Value, idx: u32) -> Option<Value> {
    let Value::Ref(target) = recv else {
        return None;
    };
    target
        .projected(RefProj::Field(idx as usize))
        .map(Value::Ref)
}

/// Materialize the result of a type-directed `Fn`-family
/// dispatch (see `Interpreter::call_callable_sync_typed`) when
/// it is a fat pointer into value-form string data: the impl
/// body's own pointer arithmetic on a `&str` it was handed as
/// data produces a fat pointer whose base is a byte pointer
/// into the string value (an index-0 projection of it), a shape
/// whose pointee the value model cannot read back as the string
/// slice it denotes. The denoted slice is exactly the string
/// data the dispatch itself owns, so it is handed back in the
/// model's canonical string-data form - the same form the
/// argument arrived in. Any other result passes through
/// unchanged.
fn fn_result_materialized(value: Value) -> Value {
    let Value::FatRef { base, len } = &value else {
        return value;
    };
    let Value::Ref(RefTarget::Value(inner, projection)) = &**base else {
        return value;
    };
    let index0_only = projection.len() <= 1
        && projection
            .iter()
            .all(|p| matches!(p, RefProj::Index(0)));
    if !index0_only {
        return value;
    }
    match &**inner {
        Value::Str(text) => {
            let end = (*len as usize).min(text.len());
            if text.is_char_boundary(end) {
                Value::Str(text[..end].to_string())
            } else {
                value
            }
        }
        _ => value,
    }
}

/// Whether a body's receiver (its first argument) is the
/// trait object type naming `trait_path`: the receiver type,
/// with reference and raw-pointer layers peeled, is a `dyn`
/// type whose bounds name the trait (under either the
/// `std::` or the `core::` spelling). A body of this shape
/// stored under a trait method's plain name is the method's
/// own body monomorphised at the trait object -- the
/// instance a `final` method's call through the object
/// denotes -- not a concrete type's instance.
fn body_receiver_is_trait_object(body: &crate::ir::Body, trait_path: &str) -> bool {
    let mut ty = body.local_tys.get(1);
    while let Some(Ty::Ref(inner)) = ty {
        ty = Some(inner);
    }
    let Some(Ty::Dynamic(bounds)) = ty else {
        return false;
    };
    bounds.iter().any(|b| {
        b == trait_path
            || b.strip_prefix("std::").map_or(false, |rest| {
                format!("core::{rest}") == trait_path
            })
            || b.strip_prefix("core::").map_or(false, |rest| {
                format!("std::{rest}") == trait_path
            })
    })
}

fn ref_shaped_receiver(value: Value) -> Value {
    // A reference-counted handle names its pointee in the
    // allocation store, not inside the handle value: the
    // receiver is a reference to the stored value field, live
    // for every handle on the allocation. A trait-object
    // receiver is excluded: a `Box<dyn Trait>` handle has the
    // same value shape as an rc handle, and only the
    // allocation's contents tell them apart, which a receiver
    // reshape cannot consult -- the dyn dispatch downstream
    // rebuilds the receiver from the fat pointer itself.
    if !matches!(value, Value::DynRef { .. }) {
        if let Some(id) = rc_handle_id(&value) {
            return Value::Ref(RefTarget::Alloc {
                id,
                projection: vec![RefProj::Field(2)],
            });
        }
    }
    match value {
        Value::Enum { fields, .. } if fields.len() == 1 => {
            let field = fields.into_iter().next().expect("one field");
            match field {
                Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. } => field,
                field => Value::Ref(RefTarget::value(field)),
            }
        }
        other @ (Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. }) => other,
        other => Value::Ref(RefTarget::value(other)),
    }
}

/// Unwrap transparent one-field wrappers (`Enum` / `Tuple` values
/// with exactly one field) around a callable value, returning the
/// inner value and whether any unwrapping happened. A wrapper is
/// only entered when its sole field holds a callable, so an
/// ordinary one-field ADT value that is not a callable wrapper is
/// left untouched.
pub(crate) fn unwrap_callable_wrapper(v: Value) -> (Value, bool) {
    let mut value = v;
    let mut unwrapped = false;
    for _ in 0..8 {
        let next = match &value {
            Value::Enum { fields, .. } if fields.len() == 1 => {
                if value_holds_callable(&fields[0]) {
                    fields[0].clone()
                } else {
                    break;
                }
            }
            Value::Tuple(items) if items.len() == 1 => {
                if value_holds_callable(&items[0]) {
                    items[0].clone()
                } else {
                    break;
                }
            }
            _ => break,
        };
        value = next;
        unwrapped = true;
    }
    (value, unwrapped)
}
