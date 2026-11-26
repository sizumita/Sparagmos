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

// The glob re-exports below keep every pre-split `crate::builtins::*`
// path working. Some of them re-export nothing at `pub(crate)`
// visibility, and some are not referenced through this facade; both
// are intentional, so the unused-imports lint is allowed here.
#![allow(unused_imports)]

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
mod libc_io;
mod lookup;
mod names;
mod net;
mod panic_report;
mod path;
mod pointer;
mod primitives;
mod process;
mod ranges;
mod registry;
mod resolve;
mod simd;
mod signals;
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
pub(crate) use self::libc_io::*;
pub(crate) use self::lookup::*;
pub(crate) use self::names::*;
pub(crate) use self::net::*;
pub(crate) use self::panic_report::*;
pub(crate) use self::path::*;
pub(crate) use self::pointer::*;
pub(crate) use self::primitives::*;
pub(crate) use self::process::*;
pub(crate) use self::ranges::*;
pub(crate) use self::registry::*;
pub(crate) use self::resolve::*;
pub(crate) use self::simd::*;
pub(crate) use self::signals::*;
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
    lookup_qualified, net_trait_method,
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

    /// The net model's name lookup answers only the one
    /// `ToSocketAddrs` method name; near-misses (another
    /// trait's method, the trait path without the method)
    /// stay unmodelled so a collected body is never shadowed.
    #[test]
    fn net_trait_method_matches_only_to_socket_addrs() {
        assert_eq!(
            net_trait_method("std::net::ToSocketAddrs::to_socket_addrs"),
            Some(Builtin::ToSocketAddrs)
        );
        assert_eq!(net_trait_method("std::net::ToSocketAddrs"), None);
        assert_eq!(
            net_trait_method("std::net::TcpListener::bind"),
            None
        );
    }

    /// Literal socket addresses resolve without any name
    /// lookup, in the canonical `SocketAddr` shape.
    #[test]
    fn socket_addr_str_parses_literals() {
        assert_eq!(
            parse_socket_addr_str("127.0.0.1:0"),
            Ok(vec![socket_addr_v4_value([127, 0, 0, 1], 0)])
        );
        assert_eq!(
            parse_socket_addr_str("192.0.2.7:65535"),
            Ok(vec![socket_addr_v4_value([192, 0, 2, 7], 65535)])
        );
        assert_eq!(
            parse_socket_addr_str("[::1]:80"),
            Ok(vec![socket_addr_v6_value(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                80,
                0,
                0
            )])
        );
    }

    /// `localhost` is the one name the model resolves (the
    /// loopback entry every host table carries); literal
    /// hosts with ports resolve; every other name is the
    /// resolver failure, never an invented address.
    #[test]
    fn socket_addr_str_resolves_localhost_only() {
        assert_eq!(
            parse_socket_addr_str("localhost:80"),
            Ok(vec![socket_addr_v4_value([127, 0, 0, 1], 80)])
        );
        assert_eq!(
            parse_socket_addr_str("10.0.0.9:22"),
            Ok(vec![socket_addr_v4_value([10, 0, 0, 9], 22)])
        );
        assert_eq!(parse_socket_addr_str("example.invalid:80"), Err(()));
        assert_eq!(parse_socket_addr_str("localhost"), Err(()));
        assert_eq!(parse_socket_addr_str("localhost:http"), Err(()));
        assert_eq!(parse_socket_addr_str("127.0.0.1:99999"), Err(()));
        assert_eq!(parse_socket_addr_str(""), Err(()));
    }

    /// A bare `SocketAddrV4` value normalizes into the
    /// wrapped `SocketAddr` form; a `SocketAddr` passes
    /// through unchanged; a non-address value is refused.
    #[test]
    fn socket_addr_value_normalizes() {
        let v4 = Value::Enum {
            variant: 0,
            fields: vec![ipv4_value([127, 0, 0, 1]), Value::Uint(443)],
        };
        assert_eq!(
            socket_addr_value(&v4),
            Some(socket_addr_v4_value([127, 0, 0, 1], 443))
        );
        let wrapped = socket_addr_v4_value([127, 0, 0, 1], 443);
        assert_eq!(socket_addr_value(&wrapped), Some(wrapped));
        assert_eq!(socket_addr_value(&Value::Uint(443)), None);
        assert_eq!(socket_addr_value(&Value::Str("127.0.0.1:443".into())), None);
    }

    /// The `(address, port)` pair forms: an `IpAddr` value
    /// and a bare `Ipv6Addr` value both combine with the
    /// port.
    #[test]
    fn ip_addr_with_port_combines() {
        let ip = Value::Enum {
            variant: 0,
            fields: vec![ipv4_value([127, 0, 0, 1])],
        };
        assert_eq!(
            ip_addr_with_port(&ip, 8080),
            Some(socket_addr_v4_value([127, 0, 0, 1], 8080))
        );
        let mut octets = [0u8; 16];
        octets[15] = 1;
        assert_eq!(
            ip_addr_with_port(&ipv6_value(octets), 8080),
            Some(socket_addr_v6_value(octets, 8080, 0, 0))
        );
        assert_eq!(ip_addr_with_port(&Value::Bool(true), 8080), None);
    }

    /// Static key resolution: an exact spelling wins; a
    /// reference spelled through an `extern crate .. as ..`
    /// alias resolves to the definition's key; a shared name
    /// resolves through its `#DefId` form; ambiguity and
    /// absence resolve nothing.
    #[test]
    fn resolve_static_key_handles_alias_spellings() {
        let keys: Vec<String> = vec![
            "cross_crate_def::EII1_IMPL".to_string(),
            "other::DECL".to_string(),
            "dup::ITEM#DefId(0:1)".to_string(),
            "dup::ITEM#DefId(0:2)".to_string(),
        ];
        let mut iter = || keys.iter();
        assert_eq!(
            resolve_static_key(iter(), "cross_crate_def::EII1_IMPL"),
            Some("cross_crate_def::EII1_IMPL")
        );
        assert_eq!(
            resolve_static_key(iter(), "cross_crate_def::codegen::EII1_IMPL"),
            Some("cross_crate_def::EII1_IMPL")
        );
        assert_eq!(
            resolve_static_key(iter(), "dup::ITEM"),
            None,
            "two disambiguated registrations are ambiguous"
        );
        assert_eq!(
            resolve_static_key(iter(), "dup::ITEM#DefId(0:2)"),
            Some("dup::ITEM#DefId(0:2)")
        );
        assert_eq!(resolve_static_key(iter(), "missing::ITEM"), None);
        assert_eq!(
            resolve_static_key(iter(), "cross_crate_def::codegen::DECL"),
            None
        );
    }

    /// The guard tag parser the `format_value` forwarding arm
    /// keys on: without it the arm never fires (the driver
    /// tests for guard formatting fail with it broken), and a
    /// loose match would forward for the cell itself.
    #[test]
    fn guard_elem_tag_names_the_guarded_type() {
        // Both crate spellings, the lifetime rendered or
        // elided, and both guard types.
        assert_eq!(
            guard_elem_tag("std::cell::Ref<'_, i32>"),
            Some("i32".to_string())
        );
        assert_eq!(
            guard_elem_tag("core::cell::Ref<'a, String>"),
            Some("String".to_string())
        );
        assert_eq!(guard_elem_tag("std::cell::Ref<i32>"), Some("i32".to_string()));
        assert_eq!(
            guard_elem_tag("std::cell::RefMut<'_, alloc::vec::Vec<i32>>"),
            Some("alloc::vec::Vec<i32>".to_string())
        );
        assert_eq!(
            guard_elem_tag("&std::cell::Ref<'_, bool>"),
            Some("bool".to_string())
        );
        // The cell itself and unrelated types are not guards:
        // the head must be followed directly by the argument
        // list.
        assert_eq!(guard_elem_tag("std::cell::RefCell<i32>"), None);
        assert_eq!(guard_elem_tag("std::cell::RefCell<'_, i32>"), None);
        assert_eq!(guard_elem_tag("my_crate::Ref<'_, i32>"), None);
        assert_eq!(guard_elem_tag("i32"), None);
    }
}
