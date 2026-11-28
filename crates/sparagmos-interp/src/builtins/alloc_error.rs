//! The allocation-error endpoints: `alloc::alloc::handle_alloc_error`
//! and `alloc::raw_vec::handle_error`.
//!
//! Neither function has a Stable MIR body in the shipped
//! libraries: `handle_alloc_error` is compiled behind the
//! `no_global_oom_handling` configuration Stable MIR does not
//! expose, and `handle_error`'s body is removed by the same
//! configuration. Every allocation failure in a guest program
//! funnels into one of them, so the runtime models the
//! dispatch the linked binary performs:
//!
//! * A program that defines a `#[alloc_error_handler]` has
//!   the generated `__rust_alloc_error_handler(size, align)`
//!   shim in its body set (the compiler's builtin macro emits
//!   it into the defining crate); `handle_alloc_error` calls
//!   that shim, which reconstructs the `Layout` and runs the
//!   guest's handler.
//! * A program that links std and defines no handler runs
//!   std's `rust_oom`: the installed hook when the guest set
//!   one with `set_alloc_error_hook` (unstable; modelled —
//!   the hook receives a fabricated `Layout` value, a hook
//!   that returns is followed by `process::abort`, and a
//!   hook that panics propagates its panic), otherwise the
//!   default hook's output (modelled by
//!   `Interpreter::print_alloc_error_default_hook`), then
//!   `process::abort`.
//! * A program with neither falls back to alloc's
//!   `__rdl_alloc_error_handler`: a nounwind panic with the
//!   message `memory allocation of {size} bytes failed`,
//!   raised at library/alloc/src/alloc.rs:699. In a `no_std`
//!   program the panic's endpoint is the guest's
//!   `#[panic_handler]`, so the model delivers a fabricated
//!   `PanicInfo` to it; with no handler recorded, the panic
//!   travels the ordinary machinery.
//!
//! `handle_error` matches on the `TryReserveErrorKind`:
//! `CapacityOverflow` is the `capacity overflow` panic raised
//! by alloc's `capacity_overflow` fn (its own location is
//! what std reports), and `AllocError { layout }` forwards
//! to `handle_alloc_error`.
//!
//! Two more body-less allocation-path functions are served
//! here. `Layout::from_size_alignment_unchecked` (and its
//! `usize`-alignment sibling `from_size_align_unchecked`)
//! are `#[rustc_const_stable]` constructors with no Stable
//! MIR body; the collected bodies of the checked `Layout`
//! constructors and of `for_value` / `array` / `pad_to_align`
//! bottom out in them, so the model is the construction
//! itself. And hashbrown's `Fallibility::capacity_overflow`
//! -- the method std's hash-table growth calls when the
//! bucket count overflows -- matches on the fallibility:
//! `Fallible` answers the `CapacityOverflow` error value,
//! `Infallible` panics with `Hash table capacity overflow`
//! (the `panic!` at hashbrown's src/raw.rs:36 in the pinned
//! build's vendored 0.17.1).

use crate::error::{InterpError, InterpResult};
use crate::interp::{Frame, Interpreter, PanicOrigin};
use crate::value::{RefTarget, Value};

use super::*;

impl<'a> Interpreter<'a> {
    /// `alloc::alloc::handle_alloc_error(layout) -> !`: run
    /// the allocation-failure dispatch for the layout's size
    /// and alignment. Never returns a value: every path ends
    /// in a guest call that diverges by contract, the std
    /// abort, or the `__rdl` panic.
    pub(crate) fn exec_handle_alloc_error(
        &mut self,
        frames: &mut Vec<Frame>,
        layout: &Value,
    ) -> InterpResult<Value> {
        let (size, align) = layout_size_align(layout);
        self.run_alloc_error(frames, size, align)
    }

    /// `alloc::raw_vec::handle_error(error) -> !`: the
    /// `TryReserveError` dispatch described in the module
    /// docs.
    pub(crate) fn exec_raw_vec_handle_error(
        &mut self,
        frames: &mut Vec<Frame>,
        error: &Value,
    ) -> InterpResult<Value> {
        let Some((variant, fields)) = try_reserve_error_kind(error) else {
            return Err(InterpError::type_mismatch(
                "alloc::raw_vec::handle_error",
                "a TryReserveError value",
                error.kind_name(),
            ));
        };
        match variant {
            // `CapacityOverflow => capacity_overflow()`: a
            // plain `panic!("capacity overflow")` inside the
            // `capacity_overflow` const fn, whose own location
            // std reports (the pinned 1.100.0 build places the
            // `panic!` at library/alloc/src/raw_vec/mod.rs:29:5
            // in the `/rustc/<hash>` remapped spelling, the same
            // fixed-stamp pattern as the `panic_in_cleanup`
            // site in `interp::exec`).
            0 => {
                if self.panic_location().is_none() {
                    self.note_panic_stamp(Value::Ref(RefTarget::value(
                        Value::Enum {
                            variant: 0,
                            fields: vec![
                                Value::Str(
                                    "/rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library/alloc/src/raw_vec/mod.rs"
                                        .to_string(),
                                ),
                                Value::Uint(29),
                                Value::Uint(5),
                            ],
                        },
                    )));
                }
                self.note_panic_origin(PanicOrigin::PanicFn);
                self.note_panic_no_backtrace(false);
                Err(InterpError::panic_str_payload("capacity overflow"))
            }
            // `AllocError { layout, .. }` forwards the layout
            // to `handle_alloc_error`.
            1 => {
                let Some(layout) = fields.first() else {
                    return Err(InterpError::type_mismatch(
                        "alloc::raw_vec::handle_error",
                        "an AllocError variant carrying a Layout",
                        error.kind_name(),
                    ));
                };
                let (size, align) = layout_size_align(layout);
                self.run_alloc_error(frames, size, align)
            }
            _ => Err(InterpError::type_mismatch(
                "alloc::raw_vec::handle_error",
                "a TryReserveErrorKind variant (0 or 1)",
                format!("variant {variant}"),
            )),
        }
    }

    /// `Layout::from_size_alignment_unchecked(size,
    /// alignment) -> Layout` and `Layout::
    /// from_size_align_unchecked(size, align) -> Layout`:
    /// the construction itself. The `Alignment` argument is
    /// already a `Layout` align field, so it passes through
    /// unchanged; a plain `usize` alignment is wrapped the
    /// way every fabricated `Layout` in this model wraps it
    /// (`alignment_value`).
    pub(crate) fn exec_layout_from_size_align_unchecked(
        &mut self,
        frames: &mut Vec<Frame>,
        size: &Value,
        align: &Value,
    ) -> InterpResult<Value> {
        let size = self.resolve_value(frames, size)?;
        let align = self.resolve_value(frames, align)?;
        let size = match size {
            Value::Uint(n) => n,
            Value::Int(n) if n >= 0 => n as u128,
            ref other => {
                return Err(InterpError::type_mismatch(
                    "Layout::from_size_align_unchecked",
                    "a size (usize)",
                    other.kind_name(),
                ));
            }
        };
        let align_field = match align {
            Value::Uint(n) => crate::interp::alignment_value(n),
            Value::Int(n) if n >= 0 => crate::interp::alignment_value(n as u128),
            other => other,
        };
        Ok(Value::Enum {
            variant: 0,
            fields: vec![Value::Uint(size), align_field],
        })
    }

    /// `hashbrown::raw::Fallibility::capacity_overflow(
    /// self) -> TryReserveError`: the match on the
    /// fallibility described in the module docs. The
    /// `Fallible` answer is the `TryReserveError` value in
    /// the form the other reserve models produce (see
    /// `VecTryReserve`'s error case).
    pub(crate) fn exec_hashbrown_capacity_overflow(
        &mut self,
        frames: &mut Vec<Frame>,
        fallibility: &Value,
    ) -> InterpResult<Value> {
        let fallibility = self.resolve_value(frames, fallibility)?;
        let variant = match fallibility {
            Value::Enum { variant, .. } => variant,
            Value::Uint(n) => n as u32,
            Value::Int(n) if n >= 0 => n as u32,
            ref other => {
                return Err(InterpError::type_mismatch(
                    "hashbrown::raw::Fallibility::capacity_overflow",
                    "a Fallibility value",
                    other.kind_name(),
                ));
            }
        };
        match variant {
            // `Fallible => TryReserveError::CapacityOverflow`.
            0 => Ok(Value::Enum {
                variant: 0,
                fields: Vec::new(),
            }),
            // `Infallible => panic!("Hash table capacity
            // overflow")`: the panic's own location is what
            // std reports (hashbrown 0.17.1, src/raw.rs:36:40
            // in the pinned build's vendored copy, in the
            // `/rustc/<hash>` remapped spelling).
            1 => {
                if self.panic_location().is_none() {
                    self.note_panic_stamp(Value::Ref(RefTarget::value(
                        Value::Enum {
                            variant: 0,
                            fields: vec![
                                Value::Str(
                                    "/rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library/vendor/hashbrown-0.17.1/src/raw.rs"
                                        .to_string(),
                                ),
                                Value::Uint(36),
                                Value::Uint(40),
                            ],
                        },
                    )));
                }
                self.note_panic_origin(PanicOrigin::PanicFn);
                self.note_panic_no_backtrace(false);
                Err(InterpError::panic_str_payload(
                    "Hash table capacity overflow",
                ))
            }
            _ => Err(InterpError::type_mismatch(
                "hashbrown::raw::Fallibility::capacity_overflow",
                "a Fallibility variant (0 or 1)",
                format!("variant {variant}"),
            )),
        }
    }

    /// `std::alloc::default_alloc_error_hook(layout)`: the
    /// default hook as a callable value (what
    /// `take_alloc_error_hook` hands back when no custom
    /// hook is installed): print the default report, then
    /// abort, exactly the endpoint `run_alloc_error` models.
    pub(crate) fn exec_default_alloc_error_hook(
        &mut self,
        frames: &mut Vec<Frame>,
        layout: &Value,
    ) -> InterpResult<Value> {
        let layout = self.resolve_value(frames, layout)?;
        let (size, _align) = layout_size_align(&layout);
        self.print_alloc_error_default_hook(frames, size);
        Err(InterpError::Abort {
            message: "memory allocation failed".to_string(),
        })
    }

    /// The shared tail of both endpoints: the
    /// allocation-failure dispatch for one (size, align).
    fn run_alloc_error(
        &mut self,
        frames: &mut Vec<Frame>,
        size: u128,
        align: u128,
    ) -> InterpResult<Value> {
        // A guest `#[alloc_error_handler]`: its generated shim
        // is part of the program (see the module docs). The
        // shim diverges by contract; whatever it does instead
        // — exit, abort, panic — is the program's behaviour
        // and propagates. Only a return breaks the contract,
        // and a handler that returns aborts natively.
        if let Some(shim) = self
            .program()
            .alloc_shims
            .get("__rust_alloc_error_handler")
            .cloned()
        {
            let result = self.call_callable_to_completion(
                frames,
                &Value::FnPtr(shim),
                vec![Value::Uint(size), Value::Uint(align)],
            );
            return match result {
                Ok(_) => Err(InterpError::Abort {
                    message: "allocation error handler returned".to_string(),
                }),
                Err(e) => Err(e),
            };
        }
        // A guest-installed custom hook: the handler shim a
        // program without a `#[alloc_error_handler]` links
        // calls the installed hook with the failed layout,
        // and the process aborts when the hook returns.
        if let Some(hook) = self.installed_alloc_error_hook() {
            let layout = Value::Enum {
                variant: 0,
                fields: vec![
                    Value::Uint(size),
                    crate::interp::alignment_value(align),
                ],
            };
            let result =
                self.call_callable_to_completion(frames, &hook, vec![layout]);
            return match result {
                Ok(_) => Err(InterpError::Abort {
                    message: "memory allocation failed".to_string(),
                }),
                Err(e) => Err(e),
            };
        }
        // std's `rust_oom` with the default hook, then abort.
        if self.program().links_std {
            self.print_alloc_error_default_hook(frames, size);
            return Err(InterpError::Abort {
                message: "memory allocation failed".to_string(),
            });
        }
        // alloc's `__rdl_alloc_error_handler`: the nounwind
        // panic, delivered to the guest's panic handler when
        // the program defines one (a `no_std` program's panic
        // endpoint), and through the ordinary panic machinery
        // otherwise.
        let message = format!("memory allocation of {size} bytes failed");
        if let Some(handler) = self.program().panic_handler.clone() {
            let info = alloc_error_panic_info(&message);
            let result = self.call_callable_to_completion(
                frames,
                &Value::FnPtr(handler),
                vec![info],
            );
            return match result {
                Ok(_) => Err(InterpError::Abort {
                    message: "panic handler returned".to_string(),
                }),
                Err(e) => Err(e),
            };
        }
        if self.panic_location().is_none() {
            self.note_panic_stamp(alloc_error_location());
        }
        self.note_panic_origin(PanicOrigin::PanicFmt);
        self.note_panic_no_backtrace(false);
        Err(InterpError::panic_str_payload(message))
    }
}

/// The caller-location stamp of the `__rdl` panic: a
/// reference to the `Location` value for the
/// `panic_nounwind_fmt` call inside
/// `__rdl_alloc_error_handler` (library/alloc/src/alloc.rs,
/// line 699).
fn alloc_error_location() -> Value {
    Value::Ref(RefTarget::value(Value::Enum {
        variant: 0,
        fields: vec![
            Value::Str("library/alloc/src/alloc.rs".to_string()),
            Value::Uint(699),
            Value::Uint(9),
        ],
    }))
}

/// The `&PanicInfo` argument for the guest panic handler on
/// the `__rdl` path, in `core::panic::PanicInfo`'s own field
/// order: the panic's message as an `fmt::Arguments`
/// reference (a plain message — the `__rdl` text is built by
/// `format_args!` from the one interpolation, and the plain
/// form formats to the same text), the location (the
/// `panic_nounwind_fmt` call inside
/// `__rdl_alloc_error_handler`, library/alloc/src/alloc.rs),
/// `can_unwind` off (the panic is nounwind) and
/// `force_no_backtrace` as the call passes it (off).
fn alloc_error_panic_info(message: &str) -> Value {
    let arguments = Value::Ref(RefTarget::value(Value::FmtArguments(
        crate::value::FmtArguments::Plain(message.to_string()),
    )));
    let location = alloc_error_location();
    let info = Value::Enum {
        variant: 0,
        fields: vec![
            arguments,
            location,
            Value::Bool(false),
            Value::Bool(false),
        ],
    };
    Value::Ref(RefTarget::value(info))
}

/// The (size, align) a `Layout` value carries, read through
/// the aggregate forms the value model uses: the two-field
/// struct, possibly inside single-field wrappers. The align
/// field is a `mem::Alignment`: either the plain integer or
/// the enum form whose variant is the alignment's exponent
/// (see `alignment_value`).
fn layout_size_align(layout: &Value) -> (u128, u128) {
    fn fields_of(v: &Value) -> Option<&Vec<Value>> {
        match v {
            Value::Tuple(fields) => Some(fields),
            Value::Enum { fields, .. } => Some(fields),
            Value::Ref(RefTarget::Value(inner, _)) => fields_of(inner),
            _ => None,
        }
    }
    fn scalar(v: &Value) -> Option<u128> {
        match v {
            Value::Uint(n) => Some(*n),
            Value::Int(n) if *n >= 0 => Some(*n as u128),
            Value::Ref(RefTarget::Value(inner, _)) => scalar(inner),
            _ => None,
        }
    }
    fn alignment(v: &Value) -> u128 {
        if let Some(n) = scalar(v) {
            return n;
        }
        match v {
            // The struct-wrapped variant form: one field
            // holding the payload-less variant whose index is
            // the exponent.
            Value::Enum { fields, .. } if fields.len() == 1 => alignment(&fields[0]),
            Value::Enum { variant, fields } if fields.is_empty() => {
                1u128.checked_shl(*variant).unwrap_or(1)
            }
            Value::Tuple(fields) if fields.len() == 1 => alignment(&fields[0]),
            Value::Ref(RefTarget::Value(inner, _)) => alignment(inner),
            _ => 1,
        }
    }
    let mut cur = layout;
    loop {
        let Some(fields) = fields_of(cur) else {
            return (scalar(layout).unwrap_or(0), 1);
        };
        if fields.len() == 2 {
            return (scalar(&fields[0]).unwrap_or(0), alignment(&fields[1]));
        }
        if fields.len() == 1 {
            cur = &fields[0];
            continue;
        }
        return (scalar(layout).unwrap_or(0), 1);
    }
}

/// The `TryReserveErrorKind` a `TryReserveError` value
/// carries: its variant index and fields. The error is a
/// one-field struct wrapping the kind enum, so single-field
/// aggregates unwrap until the enum appears.
fn try_reserve_error_kind(error: &Value) -> Option<(u32, Vec<Value>)> {
    match error {
        Value::Enum { variant, fields } => Some((*variant, fields.clone())),
        Value::Tuple(fields) if fields.len() == 1 => try_reserve_error_kind(&fields[0]),
        Value::Ref(RefTarget::Value(inner, _)) => try_reserve_error_kind(inner),
        _ => None,
    }
}

/// Registration fragment of the builtin name registry: the no-alloc shim marker.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn alloc_shim_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "alloc::alloc::__rust_no_alloc_shim_is_unstable_v2" => {
            Builtin::NoAllocShimMarker
        }
        _ => return None,
    })
}

/// Registration fragment of the builtin name registry: the allocator entry points.
///
/// One contiguous slice of the table that [`lookup`] chains in
/// registry order; the fragment lives with the family that serves
/// these builtins, so a new name is registered here, in the
/// family's own module, and nowhere else.
///
/// [`lookup`]: super::lookup
pub(crate) fn alloc_lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "alloc::alloc::__rust_alloc" => Builtin::RustAlloc { zeroed: false },
        "alloc::alloc::__rust_alloc_zeroed" => Builtin::RustAlloc { zeroed: true },
        "alloc::alloc::__rust_dealloc" => Builtin::RustDealloc,
        "alloc::alloc::__rust_realloc" => Builtin::RustRealloc,
        "alloc::alloc::handle_alloc_error" | "std::alloc::handle_alloc_error" => {
            Builtin::HandleAllocError
        }
        "alloc::raw_vec::handle_error" => Builtin::RawVecHandleError,
        "core::alloc::Layout::from_size_alignment_unchecked"
        | "alloc::alloc::Layout::from_size_alignment_unchecked"
        | "std::alloc::Layout::from_size_alignment_unchecked"
        | "core::alloc::Layout::from_size_align_unchecked"
        | "alloc::alloc::Layout::from_size_align_unchecked"
        | "std::alloc::Layout::from_size_align_unchecked" => {
            Builtin::LayoutFromSizeAlignUnchecked
        }
        "hashbrown::raw::Fallibility::capacity_overflow" => {
            Builtin::HashbrownCapacityOverflow
        }
        "std::alloc::set_alloc_error_hook" => Builtin::AllocErrorHookSet,
        "std::alloc::take_alloc_error_hook" => Builtin::AllocErrorHookTake,
        "std::alloc::default_alloc_error_hook" => Builtin::AllocErrorDefaultHook,
        _ => return None,
    })
}
