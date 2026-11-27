//! Default values, collection defaults and tag layouts.



use crate::interp::Interpreter;
use crate::value::{RefTarget, Value};

use super::*;

pub(super) fn generic_first_arg<'a>(tag: &'a str, prefix: &str) -> Option<&'a str> {
    let inner = tag.strip_prefix(prefix)?.strip_suffix('>')?;
    split_top_level(inner).into_iter().next()
}

/// All top-level generic arguments of a rendered tag with the
/// given prefix, in declaration order.
fn generic_args<'a>(tag: &'a str, prefix: &str) -> Option<Vec<&'a str>> {
    let inner = tag.strip_prefix(prefix)?.strip_suffix('>')?;
    Some(split_top_level(inner))
}

/// Two fresh SipHash keys for a new `std::hash::RandomState`,
/// drawn the way the modelled time builtins draw from the host:
/// the wall clock's nanoseconds, a per-process counter and the
/// process id, mixed through splitmix64. Successive calls in a
/// run differ, as successive `RandomState`s do natively.
pub(super) fn fresh_hash_keys() -> (u64, u64) {
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut state = nanos
        ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ u64::from(std::process::id()).rotate_left(32);
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    (next(), next())
}

/// The single value of a type with exactly one inhabitant,
/// from its rendered tag: the unit type, tuples and arrays of
/// single-value types, and `PhantomData`. Any other tag —
/// including zero-sized types with several values (an enum with
/// two unit variants, `Option<()>`), whose elements a
/// storageless slice could not faithfully invent — answers
/// `None`.
pub(super) fn single_value_for_tag(tag: &str) -> Option<Value> {
    let tag = tag.trim();
    match tag {
        "()" => return Some(Value::Unit),
        _ if tag.starts_with("std::marker::PhantomData<") => {
            return Some(Value::Enum {
                variant: 0,
                fields: Vec::new(),
            });
        }
        _ => {}
    }
    if tag.len() > 2 && tag.starts_with('(') && tag.ends_with(')') {
        let inner = &tag[1..tag.len() - 1];
        let mut vals = Vec::new();
        for part in split_top_level(inner) {
            vals.push(single_value_for_tag(part)?);
        }
        return Some(Value::Tuple(vals));
    }
    if tag.starts_with('[') && tag.ends_with(']') {
        let inner = &tag[1..tag.len() - 1];
        let (elem, n) = inner.rsplit_once(';')?;
        let n: usize = n.trim().parse().ok()?;
        if n > 4096 {
            return None;
        }
        let elem = single_value_for_tag(elem.trim())?;
        return Some(Value::Tuple(vec![elem; n]));
    }
    None
}

/// Whether `tag` renders `tracing_subscriber`'s
/// `DirectiveSet<T>` for some element type `T`: the bare
/// spelling, or the spelling with a naming crate's prefix
/// from an explicit `extern crate` item (see the tag-key
/// note on the `ErrorOutputType` arm of
/// `default_value_for_tag`).
fn is_directive_set_tag(tag: &str) -> bool {
    const HEAD: &str = "tracing_subscriber::filter::directive::DirectiveSet<";
    if !tag.ends_with('>') {
        return false;
    }
    tag.starts_with(HEAD)
        || tag
            .split_once("::")
            .is_some_and(|(_, rest)| rest.starts_with(HEAD))
}

pub(super) fn default_value_for_tag(tag: &str) -> Option<Value> {
    let tag = tag.trim();
    Some(match tag {
        "bool" => Value::Bool(false),
        "f32" | "f64" => Value::Float(0.0),
        "&str" => Value::Str(String::new()),
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => Value::Int(0),
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "char" => Value::Uint(0),
        "()" => Value::Unit,
        "std::string::String" => Value::Str(String::new()),
        // `rustc_hash::FxBuildHasher` is a unit struct; its
        // derived `Default` is the unit value. It is the hasher
        // the compiler's own hash collections are built with, so
        // the hash-collection defaults recurse into it.
        "rustc_hash::FxBuildHasher" => Value::Enum {
            variant: 0,
            fields: Vec::new(),
        },
        // `rustc_session::config::ErrorOutputType::default()` is
        // the `#[default]` variant with its field defaults:
        // `HumanReadable { kind: HumanReadableErrorType { short:
        // false, unicode: false }, color_config: ColorConfig::Auto }`
        // (rustc_errors' struct and enum, in declaration order).
        // This is the first runtime value the rustdoc entry program
        // asks for (session configuration), so it is modelled here
        // like the other hand-modelled defaults.
        // The rendered tag of a compiler-crate type named through
        // an explicit `extern crate` item carries the naming
        // crate's prefix (`<caller>::rustc_session::...`); from
        // inside `rustc_session` itself it is the bare path. Both
        // spellings denote this one ADT.
        _ if tag == "rustc_session::config::ErrorOutputType"
            || tag.ends_with("::rustc_session::config::ErrorOutputType") =>
        {
            Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Enum {
                        variant: 0,
                        fields: vec![Value::Bool(false), Value::Bool(false)],
                    },
                    Value::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    },
                ],
            }
        }
        // `rustc_errors::DiagCtxtFlags::default()` is the
        // derived `Default` of the diagnostics context flags
        // (rustc_errors' lib.rs): every field takes its own
        // default, in declaration order -- `can_emit_warnings:
        // false`, `treat_err_as_bug: None`,
        // `eagerly_emit_delayed_bugs: false`, `macro_backtrace:
        // false`, `deduplicate_diagnostics: false`,
        // `track_diagnostics: false`. This is the fourth runtime
        // value the rustdoc entry program asks for (diagnostics
        // context setup), so it is modelled here like the other
        // hand-modelled defaults. The rendered tag carries the
        // naming crate's prefix when the type is named through an
        // explicit `extern crate` item
        // (`<caller>::rustc_errors::DiagCtxtFlags`); from inside
        // the compiler crates it is the bare path. Both spellings
        // denote this one ADT.
        // `tracing_subscriber::filter::env::Builder::default()`
        // is the hand-written `Default` of the `EnvFilter`
        // builder (tracing-subscriber 0.3.23,
        // `src/filter/env/builder.rs`): `regex: true` (span
        // field values may be matched with regular expressions),
        // `env: None`, `default_directive: None`. The rustdoc
        // entry program builds its tracing filter through this
        // default, so it is modelled here like the other
        // hand-modelled defaults. The rendered tag carries the
        // naming crate's prefix when the type is named through
        // an explicit `extern crate` item; both spellings denote
        // this one ADT.
        _ if tag == "tracing_subscriber::filter::env::builder::Builder"
            || tag.ends_with("::tracing_subscriber::filter::env::builder::Builder") =>
        {
            let none = || Value::Enum {
                variant: 0,
                fields: Vec::new(),
            };
            Value::Enum {
                variant: 0,
                fields: vec![Value::Bool(true), none(), none()],
            }
        }
        // `tracing_subscriber::filter::directive::DirectiveSet::
        // default()` is the hand-written `Default` of the
        // directive set (tracing-subscriber 0.3.23,
        // `src/filter/directive.rs`): an empty `directives`
        // collection and `max_level: LevelFilter::OFF`. The
        // collection is the crate's `FilterVec`, a
        // `smallvec::SmallVec<[T; 8]>` under the `smallvec`
        // feature both compiler consumers (librustdoc and
        // rustc_log) enable; a fresh `SmallVec` is
        // `{ capacity: 0, data: <inline, uninitialised> }`
        // (smallvec 1.15.1: the capacity word alone selects
        // the inline variant at this size, and the `union`
        // feature the compiler workspace builds smallvec with
        // makes the data word the two-field union value,
        // inline field first). `LevelFilter` is tracing-core
        // 0.1.36's one-field struct over `Option<Level>`, and
        // `OFF` is its `None` case. The shape does not depend
        // on the element type, so the arm serves every
        // `DirectiveSet<T>` spelling: the rustdoc entry
        // program asks for the `StaticDirective` set, and the
        // `EnvFilter` dynamics set is the same impl. The
        // rendered tag carries the naming crate's prefix when
        // the type is named through an explicit `extern crate`
        // item; both spellings denote this one ADT.
        _ if is_directive_set_tag(tag) => {
            let none = Value::Enum {
                variant: 0,
                fields: Vec::new(),
            };
            Value::Enum {
                variant: 0,
                fields: vec![
                    // `directives`: the empty `SmallVec`.
                    Value::Enum {
                        variant: 0,
                        fields: vec![
                            Value::Uint(0),
                            // The union value, both fields
                            // present: the inline field holds
                            // the uninitialised buffer (a
                            // `ManuallyDrop` over a `MaybeUninit`
                            // in its uninitialised union form);
                            // the heap field is never read while
                            // the capacity word selects the
                            // inline variant.
                            Value::Enum {
                                variant: 0,
                                fields: vec![
                                    Value::Enum {
                                        variant: 0,
                                        fields: vec![Value::Enum {
                                            variant: 0,
                                            fields: vec![Value::Uninit],
                                        }],
                                    },
                                    Value::Tuple(vec![Value::Uninit, Value::Uint(0)]),
                                ],
                            },
                        ],
                    },
                    // `max_level`: `LevelFilter::OFF`.
                    Value::Enum {
                        variant: 0,
                        fields: vec![none],
                    },
                ],
            }
        }
        _ if tag == "rustc_errors::DiagCtxtFlags"
            || tag.ends_with("::rustc_errors::DiagCtxtFlags") =>
        {
            Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Bool(false),
                    Value::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    },
                    Value::Bool(false),
                    Value::Bool(false),
                    Value::Bool(false),
                    Value::Bool(false),
                ],
            }
        }
        // `anstream::adapter::strip::StripBytes::default()`:
        // the ANSI-escape stripping state machine the compiler's
        // error emitter carries (anstream 0.6.21,
        // `src/adapter/strip.rs`). Both fields are themselves
        // `Default` state machines: `state` is
        // `anstyle_parse::state::State::Ground` (the `#[default]`
        // variant, discriminant 12) and `utf8parser` wraps a
        // `utf8parse::Parser { point: 0, state: State::Ground }`
        // (utf8parse 0.2.2; its `State::Ground` is likewise the
        // `#[default]` variant, discriminant 0). This is the
        // third runtime value the rustdoc entry program asks for
        // (emitter setup), so it is modelled here like the other
        // hand-modelled defaults.
        // The rendered tag of a compiler-crate type named
        // through an explicit `extern crate` item carries the
        // naming crate's prefix (`<caller>::anstream::...`);
        // from inside the compiler crates it is the bare path.
        // The path itself is spelled through the defining
        // module (`adapter::strip::StripBytes`) when the call
        // originates inside anstream and through the public
        // re-export (`adapter::StripBytes`) when it is named
        // from outside. All spellings denote this one ADT.
        _ if tag == "anstream::adapter::strip::StripBytes"
            || tag == "anstream::adapter::StripBytes"
            || tag.ends_with("::anstream::adapter::strip::StripBytes")
            || tag.ends_with("::anstream::adapter::StripBytes") =>
        {
            Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Enum {
                        variant: 12,
                        fields: Vec::new(),
                    },
                    Value::Enum {
                        variant: 0,
                        fields: vec![Value::Enum {
                            variant: 0,
                            fields: vec![
                                Value::Uint(0),
                                Value::Enum {
                                    variant: 0,
                                    fields: Vec::new(),
                                },
                            ],
                        }],
                    },
                ],
            }
        }
        _ if tag.starts_with("std::vec::Vec<") => Value::Enum {
            variant: 0,
            fields: Vec::new(),
        },
        // The cell wrappers whose `Default` impl wraps the inner
        // type's default (`RefCell::new`, `Cell::new` of
        // `T::default()`): the shapes are the ones the
        // corresponding `new` builtins build. The allocating
        // wrappers (`Box`, `Rc`, `Arc`) are handled in
        // `Interpreter::default_value`, whose shapes only the
        // interpreter can mint.
        _ if generic_first_arg(tag, "std::cell::RefCell<").is_some() => {
            let inner =
                generic_first_arg(tag, "std::cell::RefCell<").expect("guard checked");
            refcell_value(default_value_for_tag(inner)?)
        }
        _ if generic_first_arg(tag, "std::cell::Cell<").is_some() => {
            let inner = generic_first_arg(tag, "std::cell::Cell<").expect("guard checked");
            cell_value(default_value_for_tag(inner)?)
        }
        _ if tag.starts_with("std::marker::PhantomData<") => Value::Enum {
            variant: 0,
            fields: Vec::new(),
        },
        _ if tag.contains("option::Option<") => Value::Enum {
            variant: 0,
            fields: Vec::new(),
        },
        _ if tag.starts_with('(') && tag.ends_with(')') => {
            let inner = &tag[1..tag.len() - 1];
            let mut vals = Vec::new();
            for part in split_top_level(inner) {
                vals.push(default_value_for_tag(part)?);
            }
            Value::Tuple(vals)
        }
        _ if tag.starts_with('[') && tag.ends_with(']') => {
            // `[T; N]`: an array of the element default.
            let inner = &tag[1..tag.len() - 1];
            let (elem, n) = inner.rsplit_once(';')?;
            let n: usize = n.trim().parse().ok()?;
            if n > 4096 {
                return None;
            }
            let elem_default = default_value_for_tag(elem.trim())?;
            Value::Tuple(vec![elem_default; n])
        }
        // Raw pointers: `Default` is the null pointer. The
        // impl covers only `Thin` pointees, so a fat pointee
        // (a slice, `str`, a `dyn` trait object) has no
        // `Default` to serve and stays unsupported.
        _ if tag.starts_with("*const ") || tag.starts_with("*mut ") => {
            let inner = tag
                .strip_prefix("*const ")
                .or_else(|| tag.strip_prefix("*mut "))
                .expect("guard checked")
                .trim();
            if inner == "str" || inner.starts_with("dyn ") || is_slice_tag(inner) {
                return None;
            }
            Value::Ref(RefTarget::Null)
        }
        // `&[T]`: `Default` is the empty slice, in the shape
        // a converted `&[]` constant takes -- a fat reference
        // over the empty aggregate -- so it prices (the
        // element type's dangling address) and compares as
        // that constant does.
        _ if tag.starts_with('&') && is_slice_tag(&tag[1..]) => Value::FatRef {
            base: Box::new(Value::Ref(RefTarget::Value(
                Box::new(Value::Tuple(Vec::new())),
                Vec::new(),
            ))),
            len: 0,
        },
        _ => return None,
    })
}

/// The deepest `Default::default()` fabrication recursion
/// (see `Interpreter::default_value_depth`). A type whose
/// derived `Default` recurses through a wrapper
/// (`struct S { b: Box<S> }`) diverges natively as well;
/// the cap keeps that divergence an explicit unsupported
/// error here instead of a host stack overflow.
const MAX_DEFAULT_DEPTH: u32 = 64;

/// The `(size, align)` of a rendered type tag on this platform
/// (x86_64), for the types whose layout is fixed and known.
impl<'a> Interpreter<'a> {
    /// The `(size, align)` of a rendered type tag: the static table
    /// for the primitive and abstract-model types, then the
    /// program's ADT layout table (compiler-computed layouts of the
    /// monomorphic ADT instances, recorded by the driver) for ADT
    /// tags. `None` stays an explicit unsupported error at the call
    /// site.
    /// `Default::default()` for a rendered type tag: the pure
    /// shapes come from `default_value_for_tag`; the allocating
    /// wrappers wrap the inner type's default in the value their
    /// `new` builtin builds (an allocation for `Box`, a counted
    /// handle for `Rc` / `Arc`), which only the interpreter can
    /// mint.
    pub(super) fn default_value(&self, tag: &str) -> Option<Value> {
        self.default_value_depth(tag, 0)
    }

    /// `Default::default()` for a rendered type tag, with
    /// the fabrication recursion depth (see
    /// [`MAX_DEFAULT_DEPTH`]).
    fn default_value_depth(&self, tag: &str, depth: u32) -> Option<Value> {
        if depth > MAX_DEFAULT_DEPTH {
            return None;
        }
        let tag = tag.trim();
        if let Some(inner) = generic_first_arg(tag, "std::boxed::Box<") {
            let pointee = if is_slice_tag(inner) {
                // `Box<[T]>`: its `Default` is an empty boxed
                // slice -- the pointee an empty
                // `Vec::into_boxed_slice` builds.
                Value::Tuple(Vec::new())
            } else {
                self.default_value_depth(inner, depth + 1)?
            };
            let boxed = self.new_box(pointee);
            if let Some(ty) = ty_from_tag(inner) {
                self.record_box_ty(&boxed, ty);
            }
            return Some(boxed);
        }
        if let Some(inner) = generic_first_arg(tag, "std::rc::Rc<")
            .or_else(|| generic_first_arg(tag, "std::sync::Arc<"))
        {
            let pointee = self.default_value_depth(inner, depth + 1)?;
            return Some(self.rc_new_handle(pointee));
        }
        if let Some(inner) = generic_first_arg(tag, "std::hash::BuildHasherDefault<") {
            let _ = inner;
            // `BuildHasherDefault<H>` is a tuple struct over a
            // single `PhantomData<H>` field; the hasher it builds
            // comes from `H::default()` at `build_hasher` time,
            // not from anything stored here.
            return Some(Value::Enum {
                variant: 0,
                fields: vec![Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                }],
            });
        }
        if tag.starts_with("std::collections::HashMap<")
            || tag.starts_with("std::collections::HashSet<")
        {
            // `HashMap::default()` is `with_hasher(S::default())`
            // over an empty raw table; `HashSet::default()` wraps
            // the same map. The shapes are the ones the collected
            // `new` bodies build: the table inner names the static
            // empty control group (sixteen `EMPTY` bytes) through
            // a value-target pointer, with zero mask, growth and
            // items.
            let inner = tag
                .trim_start_matches("std::collections::HashMap<")
                .trim_start_matches("std::collections::HashSet<");
            let inner = inner.strip_suffix('>')?;
            let args = split_top_level(inner);
            let hasher_idx = if tag.starts_with("std::collections::HashSet<") {
                1
            } else {
                2
            };
            let hasher_tag = args
                .get(hasher_idx)
                .map(|s| s.trim())
                .unwrap_or("std::hash::RandomState");
            let hasher = self.default_value_depth(hasher_tag, depth + 1)?;
            let ctrl = Value::Enum {
                variant: 0,
                fields: vec![Value::Ref(RefTarget::Value(
                    Box::new(Value::Bytes(vec![u8::MAX; 16])),
                    Vec::new(),
                ))],
            };
            let table_inner = Value::Enum {
                variant: 0,
                fields: vec![Value::Uint(0), ctrl, Value::Uint(0), Value::Uint(0)],
            };
            let table = Value::Enum {
                variant: 0,
                fields: vec![
                    table_inner,
                    Value::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    },
                    Value::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    },
                ],
            };
            let base = Value::Enum {
                variant: 0,
                fields: vec![hasher, table],
            };
            // `HashSet` wraps the map in its own one-field `base`
            // struct on top of the map's.
            let base = if tag.starts_with("std::collections::HashSet<") {
                Value::Enum {
                    variant: 0,
                    fields: vec![base],
                }
            } else {
                base
            };
            return Some(Value::Enum {
                variant: 0,
                fields: vec![base],
            });
        }
        if tag == "std::hash::RandomState" || tag == "core::hash::RandomState" {
            // `RandomState::default()` is `RandomState::new()`:
            // the struct value holding two fresh hash keys, the
            // shape the collected `new` body builds.
            let (k0, k1) = fresh_hash_keys();
            return Some(Value::Enum {
                variant: 0,
                fields: vec![Value::Uint(u128::from(k0)), Value::Uint(u128::from(k1))],
            });
        }
        if tag.starts_with("std::collections::HashMap<")
            || tag.starts_with("std::collections::HashSet<")
        {
            return self.hash_collection_default(tag, depth);
        }
        if tag.starts_with("indexmap::set::IndexSet<")
            || tag.starts_with("indexmap::map::IndexMap<")
            || tag.starts_with("indexmap::IndexMap<")
        {
            return self.index_collection_default(tag, depth);
        }
        if let Some(value) = default_value_for_tag(tag) {
            return Some(value);
        }
        self.default_value_composite(tag, depth)
    }

    /// The `Default::default()` cases the arms above and the
    /// pure tag function cannot serve: composites whose parts
    /// are themselves only fabricable with the program at
    /// hand, and a guest struct's derived `Default`.
    ///
    /// The composite arms mirror the pure function's, with
    /// the recursion running through the interpreter: an
    /// array or tuple over a guest ADT, which the pure
    /// recursion cannot fabricate, resolves here.
    ///
    /// A struct outside the `std` / `core` / `alloc`
    /// namespaces fabricates as the value its derived
    /// `Default` builds -- each field the field type's own
    /// default -- from the instance shape the driver
    /// recorded for the formatter. The model's std surface
    /// is enumerated by the arms above on purpose: a std
    /// struct the arms do not name (a `VecDeque`, say)
    /// keeps the explicit unsupported error rather than a
    /// field-wise guess at a hand-written impl. Enums stay
    /// unsupported in every namespace: which variant is
    /// `#[default]` is not part of the recorded shape.
    fn default_value_composite(&self, tag: &str, depth: u32) -> Option<Value> {
        let tag = tag.trim();
        if tag.len() > 2 && tag.starts_with('(') && tag.ends_with(')') {
            let inner = &tag[1..tag.len() - 1];
            let mut vals = Vec::new();
            for part in split_top_level(inner) {
                vals.push(self.default_value_depth(part, depth + 1)?);
            }
            return Some(Value::Tuple(vals));
        }
        if tag.starts_with('[') && tag.ends_with(']') {
            // `[T; N]`: an array of the element default.
            let inner = &tag[1..tag.len() - 1];
            let (elem, n) = inner.rsplit_once(';')?;
            let n: usize = n.trim().parse().ok()?;
            if n > 4096 {
                return None;
            }
            let elem_default = self.default_value_depth(elem.trim(), depth + 1)?;
            return Some(Value::Tuple(vec![elem_default; n]));
        }
        if tag.starts_with("std::")
            || tag.starts_with("core::")
            || tag.starts_with("alloc::")
        {
            return None;
        }
        let shape = self.program().adt_debug_shapes.get(tag)?;
        if shape.kind != sparagmos_mir::AdtShapeKind::Struct || shape.variants.len() != 1 {
            return None;
        }
        let mut fields = Vec::new();
        for field in &shape.variants[0].fields {
            fields.push(self.default_value_depth(&field.ty, depth + 1)?);
        }
        Some(Value::Enum {
            variant: 0,
            fields,
        })
    }

    /// `Default::default()` for the std hash collections: the
    /// empty table the collected `new` bodies build, assembled
    /// directly. `HashMap` / `HashSet` wrap hashbrown's map,
    /// whose table is a `RawTableInner` with `bucket_mask == 0`
    /// -- the empty singleton, which hashbrown identifies by the
    /// mask alone -- no items and no growth left, its control
    /// pointer aimed at a group of `EMPTY` control bytes. The
    /// collected bodies aim it at hashbrown's static empty
    /// group; a fresh block of `EMPTY` bytes serves the same
    /// reads (a mask-0 table's probes read only that group),
    /// and every growth and drop path keys off the mask, never
    /// the pointer's identity. The hasher field is the hasher
    /// type's own default, recursively; the allocator must be
    /// the global one, the only allocator whose value is the
    /// unit-like `Global`. Any other shape keeps the explicit
    /// unsupported error.
    fn hash_collection_default(&self, tag: &str, depth: u32) -> Option<Value> {
        let (is_set, args) =
            if let Some(args) = generic_args(tag, "std::collections::HashMap<") {
                (false, args)
            } else if let Some(args) = generic_args(tag, "std::collections::HashSet<") {
                (true, args)
            } else {
                return None;
            };
        let (hasher_tag, alloc_tag) = match (is_set, args.len()) {
            (false, 4) => (args[2], args[3]),
            (true, 3) => (args[1], args[2]),
            _ => return None,
        };
        if alloc_tag != "std::alloc::Global" && alloc_tag != "alloc::alloc::Global" {
            return None;
        }
        let hasher = self.default_value_depth(hasher_tag, depth + 1)?;
        let raw_table = self.empty_raw_table();
        let hashbrown_map = Value::Enum {
            variant: 0,
            fields: vec![hasher, raw_table],
        };
        if is_set {
            let hashbrown_set = Value::Enum {
                variant: 0,
                fields: vec![hashbrown_map],
            };
            Some(Value::Enum {
                variant: 0,
                fields: vec![hashbrown_set],
            })
        } else {
            Some(Value::Enum {
                variant: 0,
                fields: vec![hashbrown_map],
            })
        }
    }

    /// The empty hashbrown `RawTable` value: a `RawTableInner`
    /// with `bucket_mask == 0` (the empty singleton), no items
    /// and no growth left, its control pointer aimed at a fresh
    /// block of `EMPTY` control bytes, then the unit-like
    /// global allocator and the phantom marker. Shared by the
    /// std and indexmap collection defaults.
    fn empty_raw_table(&self) -> Value {
        let ctrl_id = self.alloc_insert(Value::Bytes(vec![0xFF; 64]));
        self.mark_elem_block(ctrl_id);
        let unit = || Value::Enum {
            variant: 0,
            fields: Vec::new(),
        };
        let raw_table_inner = Value::Enum {
            variant: 0,
            fields: vec![
                Value::Uint(0),
                // `ctrl`: the `NonNull<u8>` over the empty group.
                Value::Enum {
                    variant: 0,
                    fields: vec![Value::Ref(RefTarget::Alloc {
                        id: ctrl_id,
                        projection: Vec::new(),
                    })],
                },
                Value::Uint(0),
                Value::Uint(0),
            ],
        };
        Value::Enum {
            variant: 0,
            fields: vec![raw_table_inner, unit(), unit()],
        }
    }

    /// `Default::default()` for indexmap's collections: the
    /// empty value the collected constructors build.
    /// `IndexMap { core: Core { indices, entries },
    /// hash_builder }`, where `indices` is hashbrown's
    /// `hash_table::HashTable` (a single-field wrapper over the
    /// same empty `RawTable` as the std collections) and
    /// `entries` is an empty `Vec`; `IndexSet` wraps the map.
    /// The hasher field is the hasher type's own default,
    /// recursively; any other shape keeps the explicit
    /// unsupported error.
    fn index_collection_default(&self, tag: &str, depth: u32) -> Option<Value> {
        let (is_set, args) =
            if let Some(args) = generic_args(tag, "indexmap::set::IndexSet<") {
                (true, args)
            } else if let Some(args) = generic_args(tag, "indexmap::map::IndexMap<")
                .or_else(|| generic_args(tag, "indexmap::IndexMap<"))
            {
                (false, args)
            } else {
                return None;
            };
        let hasher_tag = match (is_set, args.len()) {
            (true, 2) => args[1],
            (false, 3) => args[2],
            _ => return None,
        };
        let hasher = self.default_value_depth(hasher_tag, depth + 1)?;
        let indices = Value::Enum {
            variant: 0,
            fields: vec![self.empty_raw_table()],
        };
        let entries = Value::Enum {
            variant: 0,
            fields: Vec::new(),
        };
        let core = Value::Enum {
            variant: 0,
            fields: vec![indices, entries],
        };
        let index_map = Value::Enum {
            variant: 0,
            fields: vec![core, hasher],
        };
        if is_set {
            Some(Value::Enum {
                variant: 0,
                fields: vec![index_map],
            })
        } else {
            Some(index_map)
        }
    }

    pub(crate) fn layout_of(&self, tag: &str) -> Option<(u128, u128)> {
        let tag = tag.trim();
        // A pointer-shaped tag -- a box, a reference, a raw
        // pointer, an `Rc`/`Arc` -- over an unsized pointee is
        // a fat pointer: two words, whatever the fixed table
        // below answers for the pointee's text. The table's
        // pointer arms answer the thin size for every pointee
        // but the few textual cases they special-case, so they
        // shadow the compiler-recorded layout of a fat box
        // (recorded as two words, like every fat pointer) for
        // every walk that denominates an element step by this
        // resolution: a `Vec<Box<str>>` element step was
        // denominated 8, halving the stride, and the sort's
        // own `copy_nonoverlapping` precondition priced
        // neighbouring elements as overlapping ranges. The
        // fat answer is stated here, where the program's
        // recorded DST layouts can take part in the unsized
        // judgement (the same judgement the converter makes;
        // see `ty_is_unsized` in `mir_convert`); the table
        // stays the fallback for every other shape, including
        // the thin pointers over sized pointees it prices
        // correctly.
        if let Some(pointee) = pointer_pointee_tag(tag) {
            if self.tag_is_unsized(&pointee) {
                return Some((16, 8));
            }
        }
        if let Some(layout) = type_layout(tag) {
            return Some(layout);
        }
        // Arrays: the layout is fixed by the element layout alone.
        // Elements sit at a stride of the element's size (a type's
        // size is always a multiple of its alignment, for the
        // primitive table, for the compiler-recorded ADT layouts
        // and for the composites computed here alike), so an array
        // is `n` element sizes at the element's alignment.
        if tag.starts_with('[') && tag.ends_with(']') {
            let inner = &tag[1..tag.len() - 1];
            if let Some((elem, n)) = inner.rsplit_once(';') {
                let n: u128 = n.trim().parse().ok()?;
                let (elem_size, elem_align) = self.layout_of(elem.trim())?;
                return Some((elem_size.checked_mul(n)?, elem_align));
            }
            // A slice is unsized: it has no static layout.
            return None;
        }
        // Tuples: rustc's `repr(Rust)` aggregate layout orders the
        // fields by decreasing alignment before assigning offsets.
        // Because every field's size is a multiple of its own
        // alignment, no padding is inserted between fields in that
        // order, so the size is the sum of the field sizes rounded
        // up to the largest field alignment -- a function of the
        // field layouts alone, independent of the written order.
        if tag.len() > 2 && tag.starts_with('(') && tag.ends_with(')') {
            let inner = &tag[1..tag.len() - 1];
            let mut size: u128 = 0;
            let mut align: u128 = 1;
            for part in split_top_level(inner) {
                let (field_size, field_align) = self.layout_of(part)?;
                size = size.checked_add(field_size)?;
                align = align.max(field_align);
            }
            let size = size.checked_add(align - 1)? & !(align - 1);
            return Some((size, align));
        }
        self.program()
            .adt_layouts
            .get(tag)
            .map(|&(size, align)| (u128::from(size), u128::from(align)))
    }

    /// Whether a rendered type tag denotes an unsized type:
    /// `str`, a slice or a trait object directly, a tuple by
    /// its tail element, and an ADT instance the program
    /// records a DST layout for (a struct or union whose tail
    /// field is unsized -- `OsStr`, `Path`, `CStr`, and guest
    /// DSTs alike). The rendered-tag mirror of the
    /// converter's `ty_is_unsized`.
    fn tag_is_unsized(&self, tag: &str) -> bool {
        let tag = tag.trim();
        if tag == "str" || tag.starts_with("dyn ") {
            return true;
        }
        if tag.starts_with('[') && tag.ends_with(']') {
            // `[T]` is a slice; `[T; n]` is an array, whose
            // length separator sits at the bracket's own depth.
            let inner = &tag[1..tag.len() - 1];
            let mut depth = 0i32;
            let mut prev = '\0';
            for c in inner.chars() {
                match c {
                    '<' | '(' | '[' => depth += 1,
                    '>' if prev == '-' => {}
                    '>' | ')' | ']' => depth -= 1,
                    ';' if depth == 0 => return false,
                    _ => {}
                }
                prev = c;
            }
            return true;
        }
        if tag.len() > 2 && tag.starts_with('(') && tag.ends_with(')') {
            let inner = &tag[1..tag.len() - 1];
            return split_top_level(inner)
                .last()
                .is_some_and(|tail| self.tag_is_unsized(tail));
        }
        self.program().adt_dst_layouts.contains_key(tag)
    }
}

/// The pointee tag of a pointer-shaped rendered tag: a box's
/// or smart pointer's first generic argument, a reference's
/// referent (past its lifetime and mutability), a raw
/// pointer's pointee. `None` for any other shape.
fn pointer_pointee_tag(tag: &str) -> Option<String> {
    let tag = tag.trim();
    for prefix in ["*const ", "*mut "] {
        if let Some(rest) = tag.strip_prefix(prefix) {
            return Some(rest.trim().to_string());
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
        return Some(rest.to_string());
    }
    for prefix in [
        "std::boxed::Box<",
        "alloc::boxed::Box<",
        "std::rc::Rc<",
        "alloc::rc::Rc<",
        "std::sync::Arc<",
        "alloc::sync::Arc<",
    ] {
        if tag.starts_with(prefix) {
            return generic_args_of(tag).into_iter().next();
        }
    }
    None
}

pub(super) fn type_layout(tag: &str) -> Option<(u128, u128)> {
    let tag = tag.trim();
    Some(match tag {
        "bool" | "u8" | "i8" => (1, 1),
        "u16" | "i16" => (2, 2),
        "u32" | "i32" | "f32" | "char" => (4, 4),
        "u64" | "i64" | "f64" | "usize" | "isize" => (8, 8),
        "u128" | "i128" => (16, 16),
        "()" => (0, 1),
        "std::string::String" => (24, 8),
        _ if tag.starts_with("std::vec::Vec<") => (24, 8),
        _ if tag.starts_with("std::boxed::Box<") => (8, 8),
        _ if tag.starts_with('&') => {
            let inner = tag[1..].trim_start();
            let inner = inner.strip_prefix("mut ").unwrap_or(inner);
            if inner == "str" || inner.starts_with('[') {
                (16, 8)
            } else {
                (8, 8)
            }
        }
        _ if tag.starts_with("*const ") || tag.starts_with("*mut ") => {
            let inner = tag.split_once(' ')?.1.trim();
            if inner == "str" || inner.starts_with('[') {
                (16, 8)
            } else {
                (8, 8)
            }
        }
        _ => return None,
    })
}

/// Whether values of a rendered type tag need drop glue, for the
/// types with a known answer: plain data never does; the standard
/// owning containers do.
pub(super) fn needs_drop_for_tag(tag: &str) -> Option<bool> {
    let tag = tag.trim();
    if is_integer_tag(tag)
        || matches!(tag, "bool" | "char" | "()" | "str")
        || tag.starts_with('&')
        || tag.starts_with("*const ")
        || tag.starts_with("*mut ")
    {
        return Some(false);
    }
    const OWNING: &[&str] = &[
        "std::string::String",
        "std::vec::Vec<",
        "std::boxed::Box<",
        "std::collections::HashMap<",
        "std::collections::BTreeMap<",
        "std::collections::VecDeque<",
        "std::cell::RefCell<",
        "std::sync::Mutex<",
        "std::sync::RwLock<",
        "std::sync::Arc<",
        "std::rc::Rc<",
        "std::ffi::CString",
    ];
    if OWNING.iter().any(|prefix| tag.starts_with(prefix)) {
        return Some(true);
    }
    None
}

/// Registration fragment of the builtin name registry: the `Default` surface.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn defaults_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::default::Default::default" | "core::default::Default::default" => {
            Builtin::DefaultDefault
        }
        _ => return None,
    })
}
