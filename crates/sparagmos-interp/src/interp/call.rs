//! Call resolution and invocation.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{RefProj, RefTarget, Value};

/// The generic arguments of a rendered type name, split at the
/// top nesting level of the outermost angle brackets
/// (`std::vec::Vec<(), A>` gives `["()", "A"]`).
fn generic_args_of(name: &str) -> Option<Vec<&str>> {
    let open = name.find('<')?;
    let bytes = name.as_bytes();
    let mut depth = 0i32;
    let mut end = None;
    for i in open..bytes.len() {
        match bytes[i] {
            b'<' | b'(' | b'[' => depth += 1,
            b'>' | b')' | b']' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let inner = &name[open + 1..end?];
    Some(split_top_level_str(inner))
}

/// Split a comma-separated list at the top nesting level
/// (angle brackets, parentheses and brackets all nest).
pub(crate) fn split_top_level_str(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(s[start..].trim());
    parts
}

/// The element type tag of the sequence a rendered type tag
/// denotes: an array's element, a pointer or reference position's
/// pointee sequence (an array pointee's element, any other
/// pointee itself), or an ADT instance's first generic argument
/// (the element type of the modelled collections — `Vec<T>` and
/// its kin — whose element positions a projection indexes
/// directly).
pub(crate) fn tag_sequence_elem(tag: &str) -> Option<&str> {
    let tag = tag.trim();
    for prefix in ["*const ", "*mut "] {
        if let Some(rest) = tag.strip_prefix(prefix) {
            let rest = rest.trim();
            if rest.starts_with('[') {
                return tag_sequence_elem(rest);
            }
            return Some(rest);
        }
    }
    if let Some(rest) = tag.strip_prefix('&') {
        let mut rest = rest.trim_start();
        if let Some(stripped) = rest.strip_prefix('\'') {
            // A lifetime: skip its name and the following space.
            rest = stripped
                .split_once(' ')
                .map(|(_, tail)| tail)
                .unwrap_or(stripped)
                .trim_start();
        }
        let rest = rest.strip_prefix("mut ").unwrap_or(rest).trim();
        if rest.starts_with('[') {
            return tag_sequence_elem(rest);
        }
        return Some(rest);
    }
    if tag.starts_with('[') {
        let inner = tag.strip_prefix('[')?.strip_suffix(']')?;
        if let Some((elem, _)) = inner.rsplit_once(';') {
            return Some(elem.trim());
        }
        return Some(inner.trim());
    }
    generic_args_of(tag)?.into_iter().next()
}

/// The Self type of an impl-qualified method name
/// (`<i32 as std::default::Default>::default` gives `i32`):
/// the text between the leading `<` and the ` as ` separator at
/// the same bracket depth. `None` for any other name shape.
pub(crate) fn impl_qualified_self_type(name: &str) -> Option<String> {
    let inner = name.strip_prefix('<')?;
    let bytes = inner.as_bytes();
    let mut depth = 1usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => depth += 1,
            b'>' => {
                depth -= 1;
                if depth == 0 {
                    return None;
                }
            }
            b' ' if depth == 1 && inner[i..].starts_with(" as ") => {
                return Some(inner[..i].trim().to_string());
            }
            _ => {}
        }
        i += 1;
    }
    None
}

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

fn impl_path_trait_method(name: &str) -> Option<crate::builtins::Builtin> {
    use crate::builtins::Builtin;
    const METHODS: &[(&str, &str, Builtin)] = &[
        ("PartialEq", "eq", Builtin::PartialEqEq),
        ("PartialEq", "ne", Builtin::PartialEqNe),
        ("PartialOrd", "partial_cmp", Builtin::PartialCmp),
        ("PartialOrd", "lt", Builtin::PartialOrdRel(crate::builtins::RelOp::Lt)),
        ("PartialOrd", "le", Builtin::PartialOrdRel(crate::builtins::RelOp::Le)),
        ("PartialOrd", "gt", Builtin::PartialOrdRel(crate::builtins::RelOp::Gt)),
        ("PartialOrd", "ge", Builtin::PartialOrdRel(crate::builtins::RelOp::Ge)),
        ("Ord", "cmp", Builtin::OrdCmp),
        ("Clone", "clone", Builtin::CloneClone),
    ];
    for (trait_seg, method, kind) in METHODS {
        if !name.ends_with(&format!(">::{method}")) {
            continue;
        }
        let qualified = format!("::{trait_seg}>::{method}");
        let impl_header = format!("::{trait_seg} for ");
        if name.contains(&qualified) || name.contains(&impl_header) {
            return Some(*kind);
        }
    }
    None
}

/// The primitive arithmetic trait method an impl-qualified
/// spelling of a call denotes (`<i32 as std::ops::Add>::add`,
/// `<i32 as std::ops::Add<i32>>::add`): the operator traits on
/// the primitive types, whose core impls are the operations
/// themselves. The Self type must be a primitive the value
/// model computes on — for any other Self type the impl body
/// is the only honest source, so the name does not qualify.
/// The plain trait path (`std::ops::Add::add`) does not
/// qualify either: it names no Self type, and the values carry
/// no width to compute with. `None` for any other shape.
fn impl_path_primitive_op(name: &str) -> Option<crate::builtins::Builtin> {
    use crate::builtins::Builtin;
    use crate::ir::{BinOp, UnOp};
    const BINOPS: &[(&str, &str, BinOp)] = &[
        ("Add", "add", BinOp::Add),
        ("Sub", "sub", BinOp::Sub),
        ("Mul", "mul", BinOp::Mul),
        ("Div", "div", BinOp::Div),
        ("Rem", "rem", BinOp::Rem),
        ("BitAnd", "bitand", BinOp::BitAnd),
        ("BitOr", "bitor", BinOp::BitOr),
        ("BitXor", "bitxor", BinOp::BitXor),
        ("Shl", "shl", BinOp::Shl),
        ("Shr", "shr", BinOp::Shr),
    ];
    const UNOPS: &[(&str, &str, UnOp)] = &[
        ("Neg", "neg", UnOp::Neg),
        ("Not", "not", UnOp::Not),
    ];
    const PRIMITIVE_SELF: &[&str] = &[
        "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64",
        "u128", "usize", "f32", "f64", "bool",
    ];
    let self_ty = impl_qualified_self_type(name)?;
    if !PRIMITIVE_SELF.contains(&self_ty.as_str()) {
        return None;
    }
    // The trait segment: the last path segment before the
    // closing `>::`, generic arguments aside.
    let close = name.rfind(">::")?;
    let segment = name[..close].rsplit("::").next()?;
    let trait_name = segment.split('<').next()?;
    for (trait_seg, method, op) in BINOPS {
        if trait_name == *trait_seg && name.ends_with(&format!("::{method}")) {
            return Some(Builtin::PrimitiveBinOp(*op));
        }
    }
    for (trait_seg, method, op) in UNOPS {
        if trait_name == *trait_seg && name.ends_with(&format!("::{method}")) {
            return Some(Builtin::PrimitiveUnOp(*op));
        }
    }
    None
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
        if self.program.get(func).is_some() {
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
        let mut stamp = match self.program.get(func) {
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
        if let Some(callee) = self.program.get(func) {
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
            None => match self.program.get(func) {
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
                        stamp = caller.locals[caller_idx].clone();
                    }
                }
            }
            locals[idx] = stamp;
        }
        let frame_id = self.fresh_frame_id();
        self.register_frame_base(&frames, frame_id, func);
        frames.push(Frame {
            id: frame_id,
            func: func.to_string(),
            locals,
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
            None => match self.program.get(func) {
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
        let incoming_asm = self.pending_asm_incoming.take();
        let id = self.fresh_frame_id();
        self.register_frame_base(&frames, id, func);
        frames.push(Frame {
            id,
            func: func.to_string(),
            locals,
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
}

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
        let mut found: Option<String> = None;
        for key in self
            .program
            .functions
            .keys()
            .chain(self.program.unconverted.keys())
        {
            let base = key.split('#').next().unwrap_or(key.as_str());
            let Some((self_part, trait_base, key_method)) = parse_impl_key(base) else {
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
        let base = frames.len();
        let (callee, env_ref) = self.eval_callable(&frames, body, operand)?;
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
}

/// Parse an impl-instance key `<Self as Trait>::method` into
/// `(Self section, trait path without generic arguments, method)`,
/// mirroring `mir_convert`'s key parsing. `None` for other shapes.
/// Split a rendered generic-argument list's contents at its
/// top-level commas (commas inside nested `<...>`, `(...)`,
/// `[...]`, `{...}` groups do not split). Empty input yields an
/// empty list.
pub(crate) fn split_generic_args(inner: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let bytes = inner.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'-' if bytes.get(i + 1) == Some(&b'>') => {
                i += 1;
            }
            b'<' | b'(' | b'[' | b'{' => depth += 1,
            b'>' | b')' | b']' | b'}' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(inner[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    let tail = inner[start..].trim();
    if !tail.is_empty() {
        parts.push(tail.to_string());
    }
    parts
}

/// The `(plain path, arguments)` of a rendered instance path:
/// every top-level `::<...>` group is removed from the path and
/// its arguments collected in order. A path with no group yields
/// itself and an empty list; an unbalanced group yields `None`.
fn strip_generic_groups(path: &str) -> Option<(String, Vec<String>)> {
    let mut plain = String::new();
    let mut args = Vec::new();
    let bytes = path.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b':' && path[i..].starts_with("::<") {
            let mut depth = 0i32;
            let mut j = i + 2;
            let inner_start = j + 1;
            let mut inner_end = None;
            while j < bytes.len() {
                match bytes[j] {
                    b'-' if bytes.get(j + 1) == Some(&b'>') => {
                        j += 1;
                    }
                    b'<' => depth += 1,
                    b'>' => {
                        depth -= 1;
                        if depth == 0 {
                            inner_end = Some(j);
                            break;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            let inner_end = inner_end?;
            args.extend(split_generic_args(&path[inner_start..inner_end]));
            i = inner_end + 1;
        } else {
            let ch_len = path[i..].chars().next()?.len_utf8();
            plain.push_str(&path[i..i + ch_len]);
            i += ch_len;
        }
    }
    Some((plain, args))
}

/// The parts of a coroutine *type* tag as a `dyn` value carries
/// it: the enclosing definition's plain path, the closing
/// `{closure#N}` segment, and the tag's generic arguments (the
/// enclosing definition's arguments first, then the coroutine
/// signature's own). `None` when the tag is not of this shape.
fn parse_coroutine_tag(tag: &str) -> Option<(String, String, Vec<String>)> {
    let idx = tag.rfind("::{closure#")?;
    let parent = &tag[..idx];
    let rest = &tag[idx + 2..];
    let seg_end = rest.find('}')?;
    let seg = &rest[..=seg_end];
    let tail = &rest[seg_end + 1..];
    let args = if tail.is_empty() {
        Vec::new()
    } else {
        let inner = tail.strip_prefix('<')?.strip_suffix('>')?;
        split_generic_args(inner)
    };
    Some((parent.to_string(), seg.to_string(), args))
}

/// The parts of a coroutine body *key*: the enclosing
/// definition's plain path and its instance arguments (see
/// [`strip_generic_groups`]), plus the `{closure#N}` segment.
/// `None` when the key is not a coroutine body key of this
/// shape.
fn parse_coroutine_key(key: &str) -> Option<(String, String, Vec<String>)> {
    let idx = key.rfind("::{closure#")?;
    let rest = &key[idx + 2..];
    let seg_end = rest.find('}')?;
    if seg_end + 1 != rest.len() {
        return None;
    }
    let seg = &rest[..=seg_end];
    let (parent, args) = strip_generic_groups(&key[..idx])?;
    Some((parent, seg.to_string(), args))
}

/// The spellings of a rendered type with trailing generic
/// arguments dropped from its outermost argument list, longest
/// first: `a::W<X, Y, Z>` yields `a::W<X, Y>`, then `a::W<X>`,
/// then `a::W`. A name with no argument list (or an empty one)
/// yields nothing.
fn elided_self_spellings(name: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = name.to_string();
    while let Some(elided) = drop_last_generic_arg(&current) {
        out.push(elided.clone());
        current = elided;
    }
    out
}

/// `name` with the last argument of its outermost generic
/// argument list removed (the bare path once the list is
/// empty); `None` when the name does not end in an argument
/// list. Both scans track nesting depth, so commas inside
/// nested arguments do not count, and the `>` of a `->` is not
/// a bracket.
fn drop_last_generic_arg(name: &str) -> Option<String> {
    let bytes = name.as_bytes();
    if bytes.last() != Some(&b'>') {
        return None;
    }
    // The '<' matching the final '>', scanning backwards.
    let mut depth = 0i32;
    let mut open = None;
    for i in (0..bytes.len()).rev() {
        match bytes[i] {
            b'>' if i > 0 && bytes[i - 1] == b'-' => {}
            b'>' => depth += 1,
            b'<' => {
                depth -= 1;
                if depth == 0 {
                    open = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let open = open?;
    let inner = &name[open + 1..name.len() - 1];
    if inner.is_empty() {
        return None;
    }
    // The last comma at depth 0 within the list.
    let mut depth = 0i32;
    let mut last_comma = None;
    for (i, &b) in inner.as_bytes().iter().enumerate() {
        match b {
            b'<' | b'(' | b'[' => depth += 1,
            b'>' | b')' | b']' => depth -= 1,
            b',' if depth == 0 => last_comma = Some(i),
            _ => {}
        }
    }
    match last_comma {
        Some(i) => Some(format!("{}<{}>", &name[..open], &inner[..i])),
        None => Some(name[..open].to_string()),
    }
}

fn parse_impl_key(base: &str) -> Option<(String, String, String)> {
    let (head, method) = base.rsplit_once(">::")?;
    if method.is_empty() || method.contains("::") {
        return None;
    }
    let inner = head.strip_prefix('<')?;
    let (self_part, trait_part) = inner.split_once(" as ")?;
    if self_part.is_empty() || trait_part.is_empty() {
        return None;
    }
    let trait_base = trait_part.split('<').next().unwrap_or(trait_part);
    Some((
        self_part.to_string(),
        trait_base.to_string(),
        method.to_string(),
    ))
}

/// Whether an impl key's `Self` section denotes `name`: the name
/// itself, or the name with a generic argument list appended; the
/// crate prefix may be omitted on a side that carries none.
fn impl_self_matches(self_part: &str, name: &str) -> bool {
    let self_base = self_part.split('<').next().unwrap_or(self_part);
    // The queried name may itself carry a generic argument list
    // (a fully rendered concrete type); only its base takes part
    // in the comparison, as the key side's list is stripped above.
    let name = name.split('<').next().unwrap_or(name);
    if self_part != self_base && !self_part.ends_with('>') {
        return false;
    }
    if self_base == name {
        return true;
    }
    if self_base.contains("::") && name.contains("::") {
        return false;
    }
    self_base.rsplit("::").next().unwrap_or(self_base)
        == name.rsplit("::").next().unwrap_or(name)
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
    }
}

/// Whether a hook value is the default-hook marker.
pub(crate) fn is_default_hook(value: &Value) -> bool {
    matches!(value, Value::DynRef { concrete, .. } if concrete == "std::panic::default_hook")
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
