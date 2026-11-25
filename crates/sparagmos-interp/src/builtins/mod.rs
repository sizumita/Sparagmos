//! Runtime support for calls whose bodies exist only in std/core,
//! plus the `core::fmt` formatting engine.
//!
//! The interpreter executes guest functions from converted bodies.
//! Some callees have no body anywhere in the guest program — they
//! *are* the standard library (`std::io::_print`, the `fmt::Arguments`
//! constructors, iterator adapters over std types, ...). Those calls
//! are converted to [`crate::ir::TerminatorKind::CallBuiltin`] and
//! executed here, against the same value model as everything else.
//! This is the main path for printing, not a fallback: there is no
//! other `_print` to fall back to.
//!
//! Design notes:
//! * A `core::fmt::rt::Argument` is a value + formatter function
//!   pointer pair in core. The pointer's target is std code with no
//!   converted body, so it cannot be represented; the pair
//!   (formatting trait, pointee type tag) is stored instead
//!   ([`Value::FmtArg`]). That pair is exactly what core's formatter
//!   dispatch keys on, and the pointee is read through the stored
//!   reference when the arguments are formatted — as the real
//!   `format_args!` reads through it at print time.
//! * `fmt::Arguments` in its placeholder form is an encoded byte
//!   template plus an argument array (core `fmt::Arguments::new`).
//!   The template encoding and the write loop are mirrored from
//!   Rust 1.100.0 `core::fmt` (see `format_template`), and the
//!   padding rules mirror `Formatter::pad` / `pad_integral`.



use crate::error::InterpError;
use crate::value::Value;

mod atomics;
mod backtrace;
mod btree;
mod clone;
mod collections;
mod defaults;
mod dispatch;
mod equality;
mod ffi;
mod floats;
mod fmt;
mod fs;
mod harness;
mod io;
mod io_error;
mod iterators;
mod libc;
mod lookup;
mod names;
mod panic_report;
mod path;
mod pointer;
mod primitives;
mod process;
mod ranges;
mod registry;
mod resolve;
mod simd;
mod strings;
mod sync;
mod time;
mod values;
mod views;
mod waker;

pub(crate) use self::atomics::*;
pub(crate) use self::backtrace::*;
pub(crate) use self::btree::*;
pub(crate) use self::clone::*;
pub(crate) use self::collections::*;
pub(crate) use self::defaults::*;
pub(crate) use self::dispatch::*;
pub(crate) use self::equality::*;
pub(crate) use self::ffi::*;
pub(crate) use self::floats::*;
pub(crate) use self::fmt::*;
pub(crate) use self::fs::*;
pub(crate) use self::harness::*;
pub(crate) use self::io::*;
pub(crate) use self::io_error::*;
pub(crate) use self::iterators::*;
pub(crate) use self::libc::*;
pub(crate) use self::lookup::*;
pub(crate) use self::names::*;
pub(crate) use self::panic_report::*;
pub(crate) use self::path::*;
pub(crate) use self::pointer::*;
pub(crate) use self::primitives::*;
pub(crate) use self::process::*;
pub(crate) use self::ranges::*;
pub(crate) use self::registry::*;
pub(crate) use self::resolve::*;
pub(crate) use self::simd::*;
pub(crate) use self::strings::*;
pub(crate) use self::sync::*;
pub(crate) use self::time::*;
pub(crate) use self::values::*;
pub(crate) use self::views::*;
pub(crate) use self::waker::*;

pub use self::iterators::{
    hash_collection_tag, hash_iterator_tag, iterator_consumer_modelled, iterator_len_modelled,
    iterator_next_back_modelled, iterator_next_modelled,
};
pub use self::lookup::{
    consumes_caller_location, fmt_trait_method, foreign_helper_lookup, intrinsic_lookup, lookup,
    lookup_qualified,
};
pub use self::names::{canonical_name, ty_tag};
pub use self::registry::{
    BitOp, Builtin, FastFloatOp, FetchOp, FloatOp, FsSysOp, PathPart, RelOp, SatOp, SimdOp,
    TrimSide,
};

fn unsupported(what: impl Into<String>) -> InterpError {
    InterpError::UnsupportedTerminator {
        kind: format!("runtime support: {}", what.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The value `DirectiveSet::default()` is modelled with:
    /// the empty directive set -- an empty inline `SmallVec`
    /// (capacity word 0, the union data word holding the
    /// uninitialised inline buffer) and `max_level` at
    /// `LevelFilter::OFF` (the `None` case of its `Option`
    /// field). See the `DirectiveSet` arm of
    /// `default_value_for_tag`.
    fn expected_directive_set_default() -> Value {
        Value::Enum {
            variant: 0,
            fields: vec![
                Value::Enum {
                    variant: 0,
                    fields: vec![
                        Value::Uint(0),
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
                Value::Enum {
                    variant: 0,
                    fields: vec![Value::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    }],
                },
            ],
        }
    }

    /// The rustdoc entry program asks for
    /// `DirectiveSet<StaticDirective>::default()` (the
    /// eleventh stop on the rustdoc execution path). Without
    /// the `default_value_for_tag` arm the tag resolves to
    /// `None` here and the `Default::default` builtin raises
    /// the runtime-support error the chain stops on, so this
    /// test fails with the arm removed.
    #[test]
    fn directive_set_default_is_the_empty_set() {
        let expected = expected_directive_set_default();
        // The spelling the rustdoc chain stops on.
        assert_eq!(
            default_value_for_tag(
                "tracing_subscriber::filter::directive::DirectiveSet<tracing_subscriber::filter::directive::StaticDirective>"
            ),
            Some(expected.clone())
        );
        // The same tag named through an explicit `extern
        // crate` item (the naming crate's prefix).
        assert_eq!(
            default_value_for_tag(
                "rustc_log::tracing_subscriber::filter::directive::DirectiveSet<tracing_subscriber::filter::directive::StaticDirective>"
            ),
            Some(expected.clone())
        );
        // The `EnvFilter` dynamics set is the same generic
        // impl with the same empty shape.
        assert_eq!(
            default_value_for_tag(
                "tracing_subscriber::filter::directive::DirectiveSet<tracing_subscriber::filter::env::directive::Directive>"
            ),
            Some(expected)
        );
        // The public wrapper is a different `Default` impl
        // and stays unmodelled.
        assert_eq!(
            default_value_for_tag("tracing_subscriber::filter::targets::Targets"),
            None
        );
    }
}
