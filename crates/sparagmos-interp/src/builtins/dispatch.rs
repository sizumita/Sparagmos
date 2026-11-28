//! The builtin dispatch entry: exec_builtin and the exec_builtin_values match.

use std::cmp::Ordering;

use sparagmos_mir as mir;

use crate::error::{InterpError, InterpResult};
use crate::interp::{
    box_alloc_pointer, channel_handle_id, float_to_int, layered_wrapper_pointer,
    float_to_uint, rc_handle_id,
    round_to_f16, synthetic_address,
    FmtBuilderKind, FmtBuilderState,
    Frame, Interpreter, PanicOrigin, Resolved,
};
use crate::ir::{Body, Operand, Ty};
use crate::value::{FmtArguments, FmtTrait, PanicPayload, RefProj, RefTarget, Value};

use super::*;

impl<'a> Interpreter<'a> {
    /// Execute one `CallBuiltin` terminator. `name` is canonical;
    /// `generic_args` are the callee's rendered generic type tags.
    /// `unique_elem` is the conversion-baked element value of a
    /// `box_assume_init_into_vec_unsafe` call (see
    /// `TerminatorKind::CallBuiltin::unique_elem`),
    /// `skipped_glue` the conversion-baked item glue of an
    /// `Iterator::nth` call (see
    /// `TerminatorKind::CallBuiltin::skipped_glue`), and
    /// `elem_glue` the conversion-baked element glue of a
    /// boxed-slice `From` / `Into` conversion or a
    /// `slice::to_vec` call (see
    /// `TerminatorKind::CallBuiltin::elem_glue`), and
    /// `receiver_glue` the conversion-baked glue of the
    /// receiver of a `Clone::clone` call the modelled clone
    /// builds part by part (see
    /// `TerminatorKind::CallBuiltin::receiver_glue`).
    pub(crate) fn exec_builtin(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        name: &str,
        generic_args: &[String],
        generic_type_ids: Option<&[(u64, u64)]>,
        args: &[Operand],
        cmp_discriminants: Option<&[i128]>,
        unique_elem: Option<&Value>,
        skipped_glue: Option<&crate::ir::DropGlue>,
        elem_glue: Option<&crate::ir::DropGlue>,
        receiver_glue: Option<&crate::ir::DropGlue>,
        zeroed_value: Option<&Value>,
        validity_panic: Option<&str>,
    ) -> InterpResult<Value> {
        // `Add` / `Sub` on a modelled time type: conversion emits
        // a builtin node only for those calls (see
        // `time_add_sub_terminator` in mir_convert). The names are
        // deliberately absent from `lookup`, so the driver's reach
        // pass still collects the other instances' bodies (the
        // generic shims, guest impls) as usual.
        if name == "std::ops::Add::add" || name == "std::ops::Sub::sub" {
            let mut vals = Vec::with_capacity(args.len());
            for arg in args {
                vals.push(self.eval_operand(frames, body, arg)?);
            }
            // The time types' `Add` / `Sub` are `#[track_caller]`,
            // so conversion appended the caller-location stamp;
            // note it for the overflow panic's location, exactly
            // as the panic builtins do with theirs.
            if vals.len() == 3 {
                let stamp = vals.pop().expect("three values checked");
                self.note_forwarded_panic_stamp(frames, stamp);
            }
            let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
            if name == "std::ops::Add::add"
                && (self_ty == "std::string::String" || self_ty == "alloc::string::String")
            {
                // `String + &str` (and the `&String` / `String`
                // right-hand sides): the collected `add` body
                // has `push_str` inlined into it and walks the
                // vector's raw parts, which the string model
                // cannot serve; the concatenation itself is
                // exact on the model's strings, through the
                // readers' coercions for both operands (see
                // `string_add_terminator` in `mir_convert`).
                let mut out = self.resolve_string(frames, &vals[0])?;
                let rhs = self.resolve_string(frames, &vals[1])?;
                out.push_str(&rhs);
                return Ok(Value::Str(out));
            }
            return self.time_add_sub(name, self_ty, name.ends_with("Sub::sub"), &vals);
        }
        // `Read` / `Write` / `Seek` on `std::fs::File`, the
        // `Write` methods on the standard streams and
        // `Vec<u8>`, and `File`'s inherent handle methods:
        // conversion emits a builtin node only for those calls
        // (see `file_io_terminator` in mir_convert). The names
        // are deliberately absent from `lookup`, so the
        // driver's reach pass still collects the other
        // receivers' bodies (a `&[u8]` reader's `read` among
        // them).
        if is_file_io_call(name, generic_args) {
            let mut vals = Vec::with_capacity(args.len());
            for arg in args {
                vals.push(self.eval_operand(frames, body, arg)?);
            }
            return self.exec_file_io(frames, name, generic_args, vals);
        }
        // `Read` / `Write` leaf methods on `std::net::TcpStream`:
        // conversion emits a builtin node only for those calls
        // (see `file_io_terminator` in `mir_convert`).
        if is_stream_io_call(name, generic_args) {
            let mut vals = Vec::with_capacity(args.len());
            for arg in args {
                vals.push(self.eval_operand(frames, body, arg)?);
            }
            return self.exec_stream_io(frames, name, vals);
        }
        // The `core::ub_checks` runtime predicates (and the
        // raw-pointer alignment predicate they bottom out in):
        // conversion emits a builtin node only for those calls
        // (see `ub_check_terminator` in `mir_convert`), and the
        // node is served here, before the registry lookup, for
        // the same reason as the time arms above — the collected
        // bodies of the other instances stay reachable through
        // the driver's reach pass.
        if let Some(helper) = ub_check_helper(name) {
            let mut vals = Vec::with_capacity(args.len());
            for arg in args {
                vals.push(self.eval_operand(frames, body, arg)?);
            }
            return self.exec_ub_check_helper(frames, body, name, helper, args, vals);
        }
        // The baked name may carry the calling crate's
        // `extern crate` prefix (`krate::libc::sigaction`
        // for a callee the registry names `libc::sigaction`):
        // the qualified lookup resolves the same item under
        // either spelling, as it does for every other
        // builtin-name consumer.
        let kind = lookup_qualified(name)
            .or_else(|| foreign_helper_lookup(name))
            // A body-less declaration bound to a C symbol
            // by `#[link_name]` was identified at
            // conversion by that symbol; the baked name is
            // the declaration's Rust path, so resolve the
            // recorded symbol here too (the unversioned
            // prefix after the full symbol, as at
            // conversion).
            .or_else(|| {
                let symbol = self.program().link_symbol_names.get(name)?;
                foreign_helper_lookup(symbol).or_else(|| {
                    symbol
                        .split('@')
                        .next()
                        .filter(|base| base.len() != symbol.len())
                        .and_then(foreign_helper_lookup)
                })
            })
            .or_else(|| intrinsic_lookup(name))
            .or_else(|| fmt_trait_method(name))
            .or_else(|| net_trait_method(name))
            .ok_or_else(|| unsupported(format!("call to {name} is not runtime-supported")))?;
        // The closure-taking consumers evaluate their callable
        // operand themselves: the generic evaluation below moves
        // the closure out of its place, destroying the environment
        // reference the calls must alias for `FnMut` semantics
        // (see `call_callable_operand_to_completion`).
        if matches!(kind, Builtin::IteratorFold | Builtin::IteratorForEach) {
            return self.exec_iterator_fold_like(frames, body, kind, generic_args, args);
        }
        if matches!(kind, Builtin::IteratorTryFold) {
            return self.exec_iterator_try_fold(frames, body, generic_args, args);
        }
        // `String::retain` likewise evaluates its predicate
        // operand itself (see `exec_string_retain`).
        if matches!(kind, Builtin::StringRetain) {
            return self.exec_string_retain(frames, body, args);
        }
        let mut vals = Vec::with_capacity(args.len());
        for arg in args {
            vals.push(self.eval_operand(frames, body, arg)?);
        }
        // The panic entry points are `#[track_caller]`: the
        // driver appends the caller-location stamp as a trailing
        // argument, which the arms below do not consume; it is
        // the panic's location, recorded for the panic hook.
        if matches!(
            kind,
            Builtin::PanicStr | Builtin::PanicFmt | Builtin::PanicAny | Builtin::PanicAccessError
        ) && vals.len() == 2
        {
            let stamp = vals.pop().expect("two arguments");
            self.note_forwarded_panic_stamp(frames, stamp);
        }
        if matches!(kind, Builtin::PanicNounwindFmt) && vals.len() == 3 {
            let stamp = vals.pop().expect("three arguments");
            self.note_forwarded_panic_stamp(frames, stamp);
        }
        // The modelled `#[track_caller]` operations receive
        // the caller-location stamp as one trailing argument
        // past their declared arity (see
        // `consumes_caller_location`): record it, so a panic
        // they raise reports the baked location.
        if (matches!(kind, Builtin::OptionUnwrap | Builtin::ResultUnwrap)
            && vals.len() == 2)
            || (matches!(kind, Builtin::IndexIndex | Builtin::IndexMut)
                && vals.len() == 3)
        {
            let stamp = vals.pop().expect("the caller-location stamp");
            self.note_forwarded_panic_stamp(frames, stamp);
        }
        if matches!(kind, Builtin::AssertFailed) && vals.len() == 5 {
            let stamp = vals.pop().expect("five arguments");
            self.note_forwarded_panic_stamp(frames, stamp);
        }
        self.exec_builtin_values(frames, body, args, kind, name, generic_args, generic_type_ids, vals, cmp_discriminants, unique_elem, skipped_glue, elem_glue, receiver_glue, zeroed_value, validity_panic)
    }

    /// Execute a builtin whose arguments are already evaluated
    /// values: the dispatch half of [`Self::exec_builtin`], split
    /// out for the interpreter's run-time name fallback, whose
    /// call sites hold evaluated values, not operands.
    pub(crate) fn exec_builtin_values(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        args: &[Operand],
        kind: Builtin,
        name: &str,
        generic_args: &[String],
        generic_type_ids: Option<&[(u64, u64)]>,
        vals: Vec<Value>,
        cmp_discriminants: Option<&[i128]>,
        unique_elem: Option<&Value>,
        skipped_glue: Option<&crate::ir::DropGlue>,
        elem_glue: Option<&crate::ir::DropGlue>,
        receiver_glue: Option<&crate::ir::DropGlue>,
        zeroed_value: Option<&Value>,
        validity_panic: Option<&str>,
    ) -> InterpResult<Value> {
        let arity = |n: usize| -> InterpResult<()> {
            if vals.len() == n {
                Ok(())
            } else {
                Err(InterpError::type_mismatch(
                    format!("builtin {name}"),
                    format!("{n} argument(s)"),
                    format!("{} argument(s)", vals.len()),
                ))
            }
        };
        match kind {
            Builtin::Print => {
                arity(1)?;
        self.note_stdout_use(frames)?;
        let text = self.format_arguments(frames, &self.coerce_arguments(frames, &vals[0], name)?, 0)?;
                // std's `print_to` panics when the write
                // fails; a write-through sink's error is
                // that failure (see `write_stdout`). The
                // panic is std's own, raised at a fixed
                // position in `std::io::stdio` pinned to
                // the toolchain build (the `panic!` at
                // std/src/io/stdio.rs:1209:9 in the pinned
                // 1.100.0 build), not at the guest's print
                // call site: stamp it with std's location,
                // as the runtime's other fixed library
                // positions are stamped.
                if let Err(error) = self.write_stdout(text.as_bytes()) {
                    self.note_panic_stamp(Value::Ref(RefTarget::value(
                        Value::Enum {
                            variant: 0,
                            fields: vec![
                                Value::Str(
                                    "/rustc/0ef1bc386059596dada8fb65b50425aab2ea637a/library/std/src/io/stdio.rs"
                                        .to_string(),
                                ),
                                Value::Uint(1209),
                                Value::Uint(9),
                            ],
                        },
                    )));
                    return Err(InterpError::Panic {
                        message: format!("failed printing to stdout: {error}"),
                        payload: None,
                    });
                }
                Ok(Value::Unit)
            }
            Builtin::Eprint => {
                arity(1)?;
                let text = self.format_arguments(frames, &self.coerce_arguments(frames, &vals[0], name)?, 0)?;
                // std's `print_to` on the error stream
                // panics when the write fails, as on the
                // output stream (see `Builtin::Print`).
                if let Err(error) = self.write_stderr_bytes(text.as_bytes()) {
                    return Err(InterpError::Panic {
                        message: format!("failed printing to stderr: {error}"),
                        payload: None,
                    });
                }
                Ok(Value::Unit)
            }
            Builtin::FfiIdentity => {
                arity(1)?;
                Ok(vals[0].clone())
            }
            Builtin::FfiReturnTwo => {
                arity(0)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Uint(10), Value::Uint(20)],
                })
            }
            Builtin::FfiReturnFive => {
                arity(0)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![
                        Value::Uint(10),
                        Value::Uint(20),
                        Value::Uint(30),
                        Value::Uint(40),
                        Value::Uint(50),
                    ],
                })
            }
            Builtin::FfiDbgCall => {
                arity(2)?;
                match &vals[0] {
                    Value::FnPtr(target) => {
                        let target = target.clone();
                        let data = vals[1].clone();
                        self.call_sync(frames, &target, vec![data])
                    }
                    other => Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "function pointer callback",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::FfiEmptyStruct => {
                arity(3)?;
                let v1 = aggregate_fields(&vals[0]).ok_or_else(|| {
                    unsupported(format!("call to {name}: first argument is not a struct value"))
                })?;
                let v2 = aggregate_fields(&vals[2]).ok_or_else(|| {
                    unsupported(format!("call to {name}: third argument is not a struct value"))
                })?;
                if v1.len() != 6 || v2.len() != 6 {
                    return Err(unsupported(format!(
                        "call to {name}: ManyInts shape not recognised"
                    )));
                }
                // The C asserts: each scalar field of v1 is the
                // matching field of v2 plus one, and likewise
                // for the nested TwoU8s (field 5).
                let mut holds = true;
                for i in 0..5 {
                    holds &= as_i128(&v1[i], name)? == as_i128(&v2[i], name)? + 1;
                }
                let n1 = aggregate_fields(&v1[5]).ok_or_else(|| {
                    unsupported(format!("call to {name}: nested struct shape not recognised"))
                })?;
                let n2 = aggregate_fields(&v2[5]).ok_or_else(|| {
                    unsupported(format!("call to {name}: nested struct shape not recognised"))
                })?;
                if n1.len() != 2 || n2.len() != 2 {
                    return Err(unsupported(format!(
                        "call to {name}: nested struct shape not recognised"
                    )));
                }
                for i in 0..2 {
                    holds &= as_u128(&n1[i], name)? == as_u128(&n2[i], name)? + 1;
                }
                if !holds {
                    return Err(InterpError::Abort {
                        message: format!(
                            "{name}: assertion `v1 == v2 + 1` failed (C assert in rust_test_helpers)"
                        ),
                    });
                }
                Ok(Value::Unit)
            }
            Builtin::FfiGetTestInt => {
                arity(0)?;
                Ok(Value::Int(1))
            }
            Builtin::FfiGetNullPtr => {
                arity(0)?;
                Ok(Value::Ref(RefTarget::Null))
            }
            Builtin::FfiStaticMutCheckFour => {
                arity(0)?;
                let current = self
                    .static_by_item_name("rust_dbg_static_mut")
                    .ok_or_else(|| {
                        unsupported(format!(
                            "call to {name}: no seeded static named \
                             `rust_dbg_static_mut` in this run"
                        ))
                    })?;
                if matches!(current, Value::Uninit) {
                    return Err(InterpError::UninitializedRead {
                        func: name.to_string(),
                        place: "static `rust_dbg_static_mut`".to_string(),
                    });
                }
                if as_i128(&current, name)? != 4 {
                    return Err(InterpError::Abort {
                        message: format!(
                            "{name}: assertion `rust_dbg_static_mut == 4` failed \
                             (C assert in rust_test_helpers)"
                        ),
                    });
                }
                Ok(Value::Unit)
            }
            Builtin::LibcAtol => {
                arity(1)?;
                let arg = self.c_str_arg(frames, &vals[0])?;
                Ok(Value::Int(unsafe { host_atol(arg.as_ptr()) } as i128))
            }
            Builtin::LibcAtoll => {
                arity(1)?;
                let arg = self.c_str_arg(frames, &vals[0])?;
                Ok(Value::Int(unsafe { host_atoll(arg.as_ptr()) } as i128))
            }
            Builtin::LibcAtoi => {
                arity(1)?;
                let arg = self.c_str_arg(frames, &vals[0])?;
                Ok(Value::Int(unsafe { host_atoi(arg.as_ptr()) } as i128))
            }
            Builtin::LibcStrtol => {
                let (arg, base) = self.strto_args(frames, name, &vals)?;
                let v = unsafe { host_strtol(arg.as_ptr(), std::ptr::null_mut(), base) };
                Ok(Value::Int(v as i128))
            }
            Builtin::LibcStrtoll => {
                let (arg, base) = self.strto_args(frames, name, &vals)?;
                let v = unsafe { host_strtoll(arg.as_ptr(), std::ptr::null_mut(), base) };
                Ok(Value::Int(v as i128))
            }
            Builtin::LibcStrtoul => {
                let (arg, base) = self.strto_args(frames, name, &vals)?;
                let v = unsafe { host_strtoul(arg.as_ptr(), std::ptr::null_mut(), base) };
                Ok(Value::Uint(v as u128))
            }
            Builtin::LibcStrtoull => {
                let (arg, base) = self.strto_args(frames, name, &vals)?;
                let v = unsafe { host_strtoull(arg.as_ptr(), std::ptr::null_mut(), base) };
                Ok(Value::Uint(v as u128))
            }
            Builtin::LibcIsatty => {
                arity(1)?;
                let fd = as_i128(&vals[0], name)?;
                let tty = match fd {
                    0 => std::io::IsTerminal::is_terminal(&std::io::stdin()),
                    1 => std::io::IsTerminal::is_terminal(&std::io::stdout()),
                    2 => std::io::IsTerminal::is_terminal(&std::io::stderr()),
                    f if f >= 3 => self
                        .with_file(f as u64, |file| std::io::IsTerminal::is_terminal(&*file))
                        .unwrap_or(false),
                    _ => false,
                };
                Ok(Value::Int(i128::from(tty)))
            }
            Builtin::LgammaR => {
                arity(2)?;
                let x = as_f64(&vals[0], name)?;
                let mut sign: std::ffi::c_int = 0;
                let value = unsafe { host_lgamma_r(x, &mut sign) };
                let target = self.deref_value(frames, &vals[1])?;
                self.write_resolved(frames, &target, Value::Int(sign as i128))?;
                Ok(Value::Float(value))
            }
            Builtin::FfiDbgAbi1 => {
                arity(1)?;
                let f = aggregate_fields(&vals[0]).ok_or_else(|| {
                    unsupported(format!("call to {name}: argument is not a struct value"))
                })?;
                if f.len() != 4 {
                    return Err(unsupported(format!(
                        "call to {name}: quad shape not recognised"
                    )));
                }
                let a = as_u128(&f[0], name)? as u64;
                let b = as_u128(&f[1], name)? as u64;
                let c = as_u128(&f[2], name)? as u64;
                let d = as_u128(&f[3], name)? as u64;
                Ok(aggregate_like(
                    &vals[0],
                    vec![
                        Value::Uint(c.wrapping_add(1) as u128),
                        Value::Uint(d.wrapping_sub(1) as u128),
                        Value::Uint(a.wrapping_add(1) as u128),
                        Value::Uint(b.wrapping_sub(1) as u128),
                    ],
                ))
            }
            Builtin::FfiDbgAbi2 => {
                arity(1)?;
                let f = aggregate_fields(&vals[0]).ok_or_else(|| {
                    unsupported(format!("call to {name}: argument is not a struct value"))
                })?;
                if f.len() != 3 {
                    return Err(unsupported(format!(
                        "call to {name}: floats shape not recognised"
                    )));
                }
                let a = as_f64(&f[0], name)?;
                let c = as_f64(&f[2], name)?;
                Ok(aggregate_like(
                    &vals[0],
                    vec![
                        Value::Float(c + 1.0),
                        Value::Uint(0xff),
                        Value::Float(a - 1.0),
                    ],
                ))
            }
            Builtin::FfiDbgAbi3 | Builtin::FfiDbgAbi4 => {
                arity(1)?;
                let f = aggregate_fields(&vals[0]).ok_or_else(|| {
                    unsupported(format!("call to {name}: argument is not a struct value"))
                })?;
                if f.len() != 3 {
                    return Err(unsupported(format!(
                        "call to {name}: char_char shape not recognised"
                    )));
                }
                let a = as_u128(&f[0], name)? as u8;
                let b = as_u128(&f[1], name)? as u8;
                let c = as_f64(&f[2], name)?;
                // abi_4's third field is an f32: the C adds in
                // float precision; abi_3's is an f64.
                let c = if kind == Builtin::FfiDbgAbi4 {
                    (c as f32 + 1.0f32) as f64
                } else {
                    c + 1.0
                };
                Ok(aggregate_like(
                    &vals[0],
                    vec![
                        Value::Uint(a.wrapping_add(1) as u128),
                        Value::Uint(b.wrapping_sub(1) as u128),
                        Value::Float(c),
                    ],
                ))
            }
            Builtin::FfiArgField { arg, field } => {
                let Some(target) = vals.get(arg) else {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        format!("at least {} argument(s)", arg + 1),
                        format!("{} argument(s)", vals.len()),
                    ));
                };
                let f = aggregate_fields(target).ok_or_else(|| {
                    unsupported(format!(
                        "call to {name}: argument {arg} is not a struct value"
                    ))
                })?;
                let Some(value) = f.get(field) else {
                    return Err(unsupported(format!(
                        "call to {name}: argument {arg} has no field {field}"
                    )));
                };
                Ok(value.clone())
            }
            Builtin::FfiInterestingAverage => {
                if vals.is_empty() {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "at least 1 argument(s)",
                        "0 argument(s)",
                    ));
                }
                let n = as_u128(&vals[0], name)?;
                let rest = &vals[1..];
                if rest.len() as u128 != 2 * n {
                    return Err(unsupported(format!(
                        "call to {name}: expected {n} (i64, f64) pairs, got {} values",
                        rest.len()
                    )));
                }
                let mut sum = 0.0f64;
                for pair in rest.chunks(2) {
                    sum += as_i128(&pair[0], name)? as f64;
                    sum += as_f64(&pair[1], name)?;
                }
                Ok(Value::Float(sum / n as f64))
            }
            Builtin::FfiValistInterestingAverage => {
                arity(2)?;
                let n = as_u128(&vals[0], name)?;
                let id = self.va_list_handle(frames, &vals[1])?;
                let mut sum = 0.0f64;
                for _ in 0..n {
                    let int = self.va_list_next(id)?;
                    sum += as_i128(&int, name)? as f64;
                    let float = self.va_list_next(id)?;
                    sum += as_f64(&float, name)?;
                }
                Ok(Value::Float(sum / n as f64))
            }
            Builtin::FfiVaListNextI32 => {
                arity(1)?;
                let id = self.va_list_handle(frames, &vals[0])?;
                let value = self.va_list_next(id)?;
                Ok(Value::Int(as_i128(&value, name)? as i32 as i128))
            }
            Builtin::VaArg => {
                arity(1)?;
                let id = self.va_list_handle(frames, &vals[0])?;
                self.va_list_next(id)
            }
            Builtin::FfiSquare128 => {
                arity(1)?;
                let v = as_i128(&vals[0], name)?;
                Ok(Value::Int(v.wrapping_mul(v)))
            }
            Builtin::FfiSub128 => {
                arity(2)?;
                let a = as_i128(&vals[0], name)?;
                let b = as_i128(&vals[1], name)?;
                Ok(Value::Int(a.wrapping_sub(b)))
            }
            Builtin::FfiIncrementAllParts => {
                arity(1)?;
                let Some(bits) = union_u64_bits(&vals[0]) else {
                    return Err(unsupported(format!(
                        "call to {name}: argument is not a 64-bit union value"
                    )));
                };
                let lo = (bits as u32).wrapping_add(2) as u64;
                let hi = ((bits >> 32) as u32).wrapping_add(2) as u64;
                let combined = (hi << 32) | lo;
                Ok(union_with_u64_bits(&vals[0], combined.wrapping_add(1)))
            }
            Builtin::FfiLeakUninitData => {
                arity(3)?;
                let b = as_u128(&vals[1], name)?;
                Ok(Value::Uint(b & 0xFFFF))
            }
            Builtin::FfiNewSome => {
                if vals.is_empty() || vals.len() > 2 {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "1 or 2 argument(s)",
                        format!("{} argument(s)", vals.len()),
                    ));
                }
                Ok(Value::Enum {
                    variant: 1,
                    fields: vals.clone(),
                })
            }
            Builtin::FfiNewNone => {
                arity(0)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                })
            }
            Builtin::FfiUnpackOption => {
                if vals.len() < 2 || vals.len() > 3 {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "2 or 3 argument(s)",
                        format!("{} argument(s)", vals.len()),
                    ));
                }
                let (variant, fields) = match &vals[0] {
                    Value::Enum { variant, fields } => (*variant, fields.clone()),
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "tagged option enum value",
                            other.kind_name(),
                        ))
                    }
                };
                if variant == 0 {
                    return Ok(Value::Int(0));
                }
                if variant != 1 || fields.len() != vals.len() - 1 {
                    return Err(unsupported(format!(
                        "call to {name}: option shape not recognised"
                    )));
                }
                for (ptr, value) in vals[1..].iter().zip(fields) {
                    if matches!(ptr, Value::Ref(RefTarget::Null)) {
                        return Err(InterpError::Abort {
                            message: format!(
                                "{name}: assertion `into` failed (C assert in rust_test_helpers)"
                            ),
                        });
                    }
                    let target = self.deref_value(frames, ptr)?;
                    self.write_resolved(frames, &target, value)?;
                }
                Ok(Value::Int(1))
            }
            Builtin::ProcMacroIsAvailable => {
                arity(0)?;
                Ok(Value::Bool(false))
            }
            Builtin::EnvArgs | Builtin::EnvArgsOs => {
                arity(0)?;
                // argv[0] is the program path the run was invoked
                // with, then the configured guest arguments, in
                // order. The wrapper shape matches the real std
                // layout — `Args { inner: ArgsOs }`, `ArgsOs {
                // inner: sys::args::common::Args }`, and the common
                // `Args { inner: vec::IntoIter }` — because std
                // bodies that navigate the wrappers field by field
                // (the `ExactSizeIterator::len` one-liners) are
                // collected and run against this value: a shallower
                // shape makes their second field step land on the
                // iteration state's element snapshot. The innermost
                // value is the `vec::IntoIter` in this model's
                // iteration-state shape ([element snapshot, next
                // index]).
                let mut items: Vec<Value> = Vec::new();
                if !self.program_path().is_empty() {
                    items.push(Value::Str(self.program_path().to_string()));
                }
                for arg in self.guest_args() {
                    // An argument that is valid UTF-8 is the
                    // text element the argument iterators have
                    // always carried; one that is not is
                    // carried in the `OsString` form instead,
                    // so its bytes survive unchanged (the
                    // `ArgsOs` step yields it as-is, and the
                    // `Args` step panics on it, as std's own
                    // conversion does).
                    match String::from_utf8(arg.clone()) {
                        Ok(text) => items.push(Value::Str(text)),
                        Err(_) => items.push(os_string_value(arg)),
                    }
                }
                let common_args = Value::Enum {
                    variant: 0,
                    fields: vec![owned_iter_state(items)],
                };
                let args_os = Value::Enum {
                    variant: 0,
                    fields: vec![common_args],
                };
                if kind == Builtin::EnvArgsOs {
                    Ok(args_os)
                } else {
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![args_os],
                    })
                }
            }
            Builtin::EnvVars | Builtin::EnvVarsOs => {
                arity(0)?;
                // A snapshot of the host environment at the call,
                // in environ order. The guest runs in-process, so
                // the host environment is the guest's environment:
                // the same source `EnvGetenv` / `EnvVarInner`
                // read and `EnvSetenv` / `EnvUnsetenv` mutate, so
                // an enumeration always agrees with single-key
                // lookups. The wrapper shape matches the real std
                // layout -- `Vars { inner: VarsOs }`, `VarsOs {
                // inner: sys::env::Env }`, and the common `Env {
                // iter: vec::IntoIter<(OsString, OsString)> }` --
                // because std's own `Iterator::next` / `size_hint`
                // bodies for `Vars` / `VarsOs` / `Env` are
                // collected and run against this value. The
                // innermost value is the `vec::IntoIter` in this
                // model's iteration-state shape ([element
                // snapshot, next index]).
                let mut items: Vec<Value> = Vec::new();
                for (k, v) in std::env::vars_os() {
                    let key_bytes =
                        crate::platform::os_str_bytes(&k).ok_or_else(|| {
                            unsupported(
                                "env::vars_os: a host variable name has no byte form on this platform",
                            )
                        })?;
                    let value_bytes =
                        crate::platform::os_str_bytes(&v).ok_or_else(|| {
                            unsupported(
                                "env::vars_os: a host variable value has no byte form on this platform",
                            )
                        })?;
                    items.push(Value::Tuple(vec![
                        os_string_value(&key_bytes),
                        os_string_value(&value_bytes),
                    ]));
                }
                let common_env = Value::Enum {
                    variant: 0,
                    fields: vec![owned_iter_state(items)],
                };
                let vars_os = Value::Enum {
                    variant: 0,
                    fields: vec![common_env],
                };
                if kind == Builtin::EnvVarsOs {
                    Ok(vars_os)
                } else {
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![vars_os],
                    })
                }
            }
            Builtin::ProcessExit => {
                arity(1)?;
                let code = as_i128(&vals[0], name)? as i32;
                Err(InterpError::ProcessExit { code })
            }
            Builtin::IntrinsicsAbort => {
                arity(0)?;
                Err(InterpError::Abort {
                    message: "explicit abort".to_string(),
                })
            }
            Builtin::IntrinsicsUnreachable => {
                arity(0)?;
                Err(InterpError::Abort {
                    message: "entered unreachable code".to_string(),
                })
            }
            Builtin::Stdout => {
                arity(0)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Unit],
                })
            }
            Builtin::Stdin => {
                arity(0)?;
                // The handle names what descriptor 0
                // currently denotes: its file entry when
                // rerouted, the stdin entry by default, and
                // the null device while descriptor 0 is
                // closed (std reads a closed stdin as
                // end-of-file).
                let id = match self.desc_route(0) {
                    crate::interp::DescRoute::File(id) => id,
                    crate::interp::DescRoute::Stdin => {
                        self.stdin_file_id()
                    }
                    _ => self.null_file_id(),
                };
                Ok(file_value(id))
            }
            Builtin::StdStreamLock => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                self.read_resolved(frames, &target)
            }
            Builtin::WriteFmt => {
                arity(2)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                let self_ty = self_ty
                    .strip_prefix("&mut ")
                    .or_else(|| self_ty.strip_prefix("&"))
                    .unwrap_or(self_ty);
                let text = match self.format_arguments(
                    frames,
                    &self.coerce_arguments(frames, &vals[1], name)?,
                    0,
                ) {
                    Ok(text) => text,
                    // A formatting trait implementation returned
                    // `Err(fmt::Error)`: this builtin *is* core's
                    // `default_write_fmt` for its receivers, and
                    // its sinks have not failed before the text
                    // is composed, so core's outcome is the
                    // panic, not an `Err` value.
                    Err(InterpError::FmtReturnedError { .. }) => {
                        return Err(self.fmt_returned_error_panic());
                    }
                    Err(error) => return Err(error),
                };
                match self_ty {
                    "std::io::Stdout"
                    | "std::io::StdoutLock"
                    | "std::sys::stdio::unix::Stdout" => {
                        if let Err(error) = self.write_stdout(text.as_bytes()) {
                            return Ok(io_result_err(host_io_error(&error)));
                        }
                    }
                    "std::io::Stderr"
                    | "std::io::StderrLock"
                    | "std::sys::stdio::unix::Stderr" => {
                        self.push_error_output(&text);
                    }
                    "std::fs::File" => {
                        let id = self.file_id_of(frames, &vals[0])?;
                        let written = self.with_file(id, |f| {
                            <std::fs::File as std::io::Write>::write_all(
                                f,
                                text.as_bytes(),
                            )
                        });
                        match written {
                            Some(Ok(())) => {}
                            Some(Err(error)) => {
                                return Ok(io_result_err(host_io_error(&error)))
                            }
                            None => {
                                return Err(unsupported(
                                    "Write::write_fmt on a file handle that is not open"
                                        .to_string(),
                                ))
                            }
                        }
                    }
                    _ if self_ty.starts_with("std::vec::Vec<u8") => {
                        // `Write for Vec<u8>` appends the formatted
                        // bytes to the vector.
                        let target = self.deref_value(frames, &vals[0])?;
                        let mut state = self.read_resolved(frames, &target)?;
                        let items: Vec<Value> = text
                            .bytes()
                            .map(|b| Value::Uint(b as u128))
                            .collect();
                        if vec_real_aggregate_parts(&state).is_some() {
                            let uninit_elem = uninit_elem_value_for_tag("u8");
                            vec_ensure_capacity(
                                self,
                                &mut state,
                                items.len(),
                                false,
                                &uninit_elem,
                                1,
                            );
                            self.vec_append_real_form(
                                frames,
                                &mut state,
                                &items,
                                "Write::write_fmt for Vec<u8>",
                            )?;
                        } else {
                            vec_append_in_place(
                                &mut state,
                                &items,
                                "Write::write_fmt for Vec<u8>",
                            )?;
                        }
                        self.write_resolved(frames, &target, state)?;
                    }
                    _ => {
                        return Err(unsupported(format!(
                            "Write::write_fmt for {self_ty} is not runtime-supported"
                        )))
                    }
                }
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Unit],
                })
            }
            Builtin::FmtWrite => {
                arity(2)?;
                // The output is a `&mut dyn Write`: resolve it to
                // the dyn value, then deliver the formatted text
                // through the concrete type's own `write_str`. The
                // real `fmt::write` body walks the template and
                // calls `write_str` piece by piece, in template
                // order, stopping at the first failure: a writer
                // error ends the walk (later pieces are neither
                // written nor, in the body, composed), and a
                // formatting failure -- a trait implementation
                // returning `Err(fmt::Error)` -- is the call's own
                // `Err` result once the pieces before it are
                // delivered; the caller (core's
                // `default_write_fmt` checking its adapter,
                // `format!`'s expect, ...) decides what it means.
                // The parts are composed up front here, so a
                // piece's formatting runs even when an earlier
                // piece's write then fails, where the body would
                // not have composed it; the delivery order and
                // the failure precedence are the body's.
                let dyn_value = match &vals[0] {
                    Value::DynRef { .. } => vals[0].clone(),
                    Value::Ref(_) => {
                        let target = self.deref_value(frames, &vals[0])?;
                        self.read_resolved(frames, &target)?
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "&mut dyn Write value",
                            other.kind_name(),
                        ))
                    }
                };
                let Value::DynRef {
                    base, concrete, ..
                } = dyn_value
                else {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "&mut dyn Write value",
                        dyn_value.kind_name(),
                    ));
                };
                // A `Formatter`'s own `buf` trait object (the
                // trait object the `Formatter` field projection
                // in `project` synthesizes): the capture itself
                // is the sink in this model, so the pieces are
                // delivered straight into it below, as
                // `Formatter::write_str` forwards them to its
                // `buf` -- there is no collected `write_str`
                // impl to call, and none is needed.
                let formatter_sink = self.formatter_id_of(frames, &base).ok();
                let impl_name = match self.scan_virtual_impl(
                    &concrete,
                    "core::fmt::Write",
                    "write_str",
                ) {
                    Some(name) => name,
                    None if formatter_sink.is_some() => String::new(),
                    None => {
                        return Err(unsupported(format!(
                            "fmt::write output type {concrete} has no collected write_str impl"
                        )));
                    }
                };
                let receiver = match &*base {
                    Value::Ref(_) | Value::FatRef { .. } | Value::DynRef { .. } => {
                        (*base).clone()
                    }
                    _ => {
                        return Err(unsupported(format!(
                            "fmt::write output type {concrete} is not held behind a reference"
                        )))
                    }
                };
                let coerced = self.coerce_arguments(frames, &vals[1], name)?;
                let (parts, failure) = self.format_arguments_parts(frames, &coerced, 0);
                for part in parts {
                    if let Some(id) = formatter_sink {
                        self.formatter_push(id, &part);
                        continue;
                    }
                    let wrote = self.call_sync(
                        frames,
                        &impl_name,
                        vec![receiver.clone(), Value::Str(part)],
                    )?;
                    match wrote {
                        Value::Enum { variant: 0, .. } => {}
                        other => return Ok(other),
                    }
                }
                match failure {
                    None => Ok(Value::Enum {
                        variant: 0,
                        fields: vec![Value::Unit],
                    }),
                    Some(InterpError::FmtReturnedError { .. }) => Ok(Value::Enum {
                        variant: 1,
                        fields: vec![Value::Enum {
                            variant: 0,
                            fields: Vec::new(),
                        }],
                    }),
                    Some(error) => Err(error),
                }
            }
            Builtin::Format => {
                arity(1)?;
                let text = self.format_arguments(frames, &self.coerce_arguments(frames, &vals[0], name)?, 0)?;
                Ok(Value::Str(text))
            }
            Builtin::FmtNumFmt => {
                arity(2)?;
                // `core::fmt::num`'s `_fmt`: the decimal digits
                // of the unsigned value, written right-aligned
                // into the caller's buffer; the result is the
                // buffer's written suffix as a `&str` -- a fat
                // reference into the buffer itself, not a copy:
                // the signed `SpecToString` body hands it to
                // `String::push_str`, whose inlined form walks
                // the string's raw parts, so the digits must be
                // readable through the returned pointer.
                let value = as_u128(&vals[0], name)?;
                let digits = value.to_string();
                let bytes = digits.as_bytes();
                let Value::FatRef { base, len } = &vals[1] else {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "a byte slice pointer",
                        vals[1].kind_name(),
                    ));
                };
                let offset = (*len as usize)
                    .checked_sub(bytes.len())
                    .ok_or_else(|| {
                        unsupported(format!(
                            "builtin {name}: {} digits do not fit the {len}-byte buffer",
                            bytes.len()
                        ))
                    })?;
                self.write_bytes_into_slice(frames, &vals[1], offset, bytes)?;
                let step = RefProj::Subslice {
                    offset,
                    len: bytes.len(),
                };
                let base = match &**base {
                    Value::Ref(target) => target
                        .projected(step)
                        .map(|target| Value::Ref(target)),
                    other => RefTarget::value(other.clone())
                        .projected(step)
                        .map(|target| Value::Ref(target)),
                }
                .ok_or_else(|| {
                    unsupported(format!(
                        "builtin {name}: the buffer pointer denotes no storage"
                    ))
                })?;
                Ok(Value::FatRef {
                    base: Box::new(base),
                    len: bytes.len() as u128,
                })
            }
            Builtin::FmtNumFormatInto { signed, buf_len } => {
                arity(2)?;
                // `core::fmt::num`'s `format_into` (see
                // `Builtin::FmtNumFormatInto`): the decimal
                // digits of the value -- with a leading `-`
                // for a negative signed value -- written
                // right-aligned into the `NumBuffer`'s byte
                // array; the result is the buffer's written
                // suffix as a `&str`, a fat reference into
                // the buffer itself, not a copy.
                let text = if signed {
                    as_i128(&vals[0], name)?.to_string()
                } else {
                    as_u128(&vals[0], name)?.to_string()
                };
                let bytes = text.as_bytes();
                let Value::Ref(buf_target) = &vals[1] else {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "a NumBuffer reference",
                        vals[1].kind_name(),
                    ));
                };
                let target = self.deref_value(frames, &vals[1])?;
                let current = self.read_resolved(frames, &target)?;
                // The buffer value's first field is the byte
                // array; the second is the integer type's
                // `PhantomData` marker.
                let array = match &current {
                    Value::Enum { fields, .. } if !fields.is_empty() => fields[0].clone(),
                    Value::Tuple(items) if !items.is_empty() => items[0].clone(),
                    other => {
                        return Err(unsupported(format!(
                            "builtin {name}: a {} buffer value is not modelled",
                            other.kind_name()
                        )))
                    }
                };
                // The array's length: its own, once a write
                // has materialised it; before that the array
                // is the uninitialised value and the length
                // is the capacity the buffer type defines.
                let n = match &array {
                    Value::Tuple(items) => items.len(),
                    Value::Bytes(data) => data.len(),
                    Value::SparseVec { len, .. } => *len as usize,
                    Value::Enum { fields, .. } => fields.len(),
                    Value::Uninit => buf_len,
                    other => {
                        return Err(unsupported(format!(
                            "builtin {name}: a {} buffer array is not modelled",
                            other.kind_name()
                        )))
                    }
                };
                let offset = n.checked_sub(bytes.len()).ok_or_else(|| {
                    unsupported(format!(
                        "builtin {name}: {} digits do not fit the {n}-byte buffer",
                        bytes.len()
                    ))
                })?;
                let new_array = match replace_slice_bytes(&array, offset, bytes) {
                    Some(value) => value,
                    None if matches!(array, Value::Uninit) => {
                        let mut elems = vec![Value::Uninit; n];
                        for (i, b) in bytes.iter().enumerate() {
                            elems[offset + i] = Value::Uint(*b as u128);
                        }
                        Value::Tuple(elems)
                    }
                    None => {
                        return Err(unsupported(format!(
                            "builtin {name}: writing into a {} buffer array is not modelled",
                            array.kind_name()
                        )))
                    }
                };
                let new_buf = match current {
                    Value::Enum {
                        variant,
                        mut fields,
                    } => {
                        fields[0] = new_array;
                        Value::Enum { variant, fields }
                    }
                    Value::Tuple(mut items) => {
                        items[0] = new_array;
                        Value::Tuple(items)
                    }
                    _ => unreachable!("the buffer shape was checked above"),
                };
                self.write_resolved(frames, &target, new_buf)?;
                let step = RefProj::Subslice {
                    offset,
                    len: bytes.len(),
                };
                let base = buf_target
                    .projected(RefProj::Field(0))
                    .and_then(|target| target.projected(step))
                    .map(Value::Ref)
                    .ok_or_else(|| {
                        unsupported(format!(
                            "builtin {name}: the buffer pointer denotes no storage"
                        ))
                    })?;
                Ok(Value::FatRef {
                    base: Box::new(base),
                    len: bytes.len() as u128,
                })
            }
            Builtin::ArgumentsFromStr => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                Ok(Value::FmtArguments(FmtArguments::Plain(s)))
            }
            Builtin::ArgumentsNew => {
                arity(2)?;
                let template = self.resolve_bytes(frames, &vals[0])?;
                let args = self.resolve_arg_array(frames, &vals[1])?;
                Ok(Value::FmtArguments(FmtArguments::Template {
                    template,
                    args,
                }))
            }
            Builtin::ArgumentNew(trait_) => {
                arity(1)?;
                let ty = generic_args.first().cloned().unwrap_or_default();
                Ok(Value::FmtArg {
                    target: Box::new(vals.into_iter().next().expect("arity checked")),
                    trait_,
                    ty,
                })
            }
            Builtin::FmtMethod(trait_) => {
                // A UFCS `fmt` trait call whose `Self` instance
                // has no converted body: format the receiver
                // with the structural engine, under the captured
                // formatter's own options, and append the text
                // to its buffer — the same output the type's
                // `Argument` formatting produces for the same
                // value and options.
                arity(2)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                let pointee = self.resolve_value(frames, &vals[0])?;
                let id = self.formatter_id_of(frames, &vals[1])?;
                let capture = self.capture_of(id)?;
                let options = FmtOptions {
                    flags: capture.flags,
                    width: capture.width,
                    precision: capture.precision,
                };
                let text = self.format_value(
                    frames,
                    &pointee,
                    trait_,
                    self_ty,
                    &options,
                    capture.indent,
                )?;
                self.formatter_push(id, &text);
                Ok(fmt_result_ok())
            }
            Builtin::FormatterWriteStr => {
                arity(2)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                let s = self.resolve_string(frames, &vals[1])?;
                self.formatter_push(id, &s);
                Ok(fmt_result_ok())
            }
            Builtin::FormatterWriteChar => {
                arity(2)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                let c = match self.resolve_value(frames, &vals[1])? {
                    Value::Uint(v) => as_char(v)?,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "Formatter::write_char",
                            "char value",
                            other.kind_name(),
                        ))
                    }
                };
                self.formatter_push(id, &c.to_string());
                Ok(fmt_result_ok())
            }
            Builtin::FormatterPad => {
                arity(2)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                let s = self.resolve_string(frames, &vals[1])?;
                let options = self.capture_of(id)?.options();
                let text = pad_str(&s, &options);
                self.formatter_push(id, &text);
                Ok(fmt_result_ok())
            }
            Builtin::FormatterWriteFmt => {
                arity(2)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                let indent = self.capture_of(id)?.indent;
                let text = self.format_arguments(
                    frames,
                    &self.coerce_arguments(frames, &vals[1], name)?,
                    indent,
                )?;
                self.formatter_push(id, &text);
                Ok(fmt_result_ok())
            }
            Builtin::FormatterAlternate => {
                arity(1)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                Ok(Value::Bool(self.capture_of(id)?.options().alternate()))
            }
            Builtin::FormatterWidth => {
                arity(1)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                Ok(match self.capture_of(id)?.options().width() {
                    Some(w) => Value::Enum {
                        variant: 1,
                        fields: vec![Value::Uint(w as u128)],
                    },
                    None => Value::Enum {
                        variant: 0,
                        fields: vec![],
                    },
                })
            }
            Builtin::FormatterPrecision => {
                arity(1)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                Ok(match self.capture_of(id)?.options().precision() {
                    Some(p) => Value::Enum {
                        variant: 1,
                        fields: vec![Value::Uint(p as u128)],
                    },
                    None => Value::Enum {
                        variant: 0,
                        fields: vec![],
                    },
                })
            }
            Builtin::FormatterFill => {
                arity(1)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                Ok(Value::Uint(self.capture_of(id)?.options().fill() as u32 as u128))
            }
            Builtin::FormatterSignPlus => {
                arity(1)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                Ok(Value::Bool(self.capture_of(id)?.options().sign_plus()))
            }
            Builtin::FormatterSignAwareZeroPad => {
                arity(1)?;
                let id = self.formatter_id_of(frames, &vals[0])?;
                Ok(Value::Bool(self.capture_of(id)?.options().zero_pad()))
            }
            Builtin::DebugStructFinish { fields } => {
                arity(2 + 2 * fields)?;
                self.exec_debug_struct_finish(frames, &vals, fields)
            }
            Builtin::DebugStructFieldsFinish => {
                arity(4)?;
                self.exec_debug_struct_fields_finish(frames, &vals)
            }
            Builtin::DebugTupleFieldsFinish => {
                arity(3)?;
                self.exec_debug_tuple_fields_finish(frames, &vals)
            }
            Builtin::DebugTupleFinish { fields } => {
                arity(2 + fields)?;
                self.exec_debug_tuple_finish(frames, &vals, fields)
            }
            Builtin::FmtBuilderNew(kind) => {
                let (fmt_arg, builder_name) = match kind {
                    FmtBuilderKind::Struct | FmtBuilderKind::Tuple => {
                        arity(2)?;
                        let name = self.resolve_string(frames, &vals[1])?;
                        (vals[0].clone(), Some(name))
                    }
                    _ => {
                        arity(1)?;
                        (vals[0].clone(), None)
                    }
                };
                let formatter = self.formatter_id_of(frames, &fmt_arg)?;
                let id = self.new_fmt_builder(FmtBuilderState {
                    formatter,
                    kind,
                    name: builder_name,
                    has_fields: false,
                    result_ok: true,
                });
                Ok(Value::FmtBuilder(id))
            }
            Builtin::FmtBuilderEntry => {
                arity(2)?;
                let id = self.builder_id_of(frames, &vals[0])?;
                self.builder_entry(frames, id, &vals[1])?;
                Ok(Value::Ref(RefTarget::value(Value::FmtBuilder(id))))
            }
            Builtin::FmtBuilderMapEntry => {
                arity(3)?;
                let id = self.builder_id_of(frames, &vals[0])?;
                self.builder_map_entry(frames, id, &vals[1], &vals[2])?;
                Ok(Value::Ref(RefTarget::value(Value::FmtBuilder(id))))
            }
            Builtin::FmtBuilderStructField => {
                arity(3)?;
                let id = self.builder_id_of(frames, &vals[0])?;
                let field_name = self.resolve_string(frames, &vals[1])?;
                self.builder_struct_field(frames, id, &field_name, &vals[2])?;
                Ok(Value::Ref(RefTarget::value(Value::FmtBuilder(id))))
            }
            Builtin::FmtBuilderTupleField => {
                arity(2)?;
                let id = self.builder_id_of(frames, &vals[0])?;
                self.builder_tuple_field(frames, id, &vals[1])?;
                Ok(Value::Ref(RefTarget::value(Value::FmtBuilder(id))))
            }
            Builtin::FmtBuilderFinish { exhaustive } => {
                arity(1)?;
                let id = self.builder_id_of(frames, &vals[0])?;
                self.builder_finish(id, exhaustive)?;
                Ok(fmt_result_ok())
            }
            Builtin::ArgumentFromUsize => {
                arity(1)?;
                // Current core passes the count by reference
                // (`from_usize(&usize)`: the format-args capture
                // tuple holds a reference to the count), so read
                // through one reference level; a bare integer is
                // accepted as well. core truncates the count to
                // u16.
                let arg = self.read_ref_once(frames, &vals[0])?;
                let n = match arg {
                    Value::Uint(v) => v as u16,
                    Value::Int(v) => v as u16,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            "Argument::from_usize",
                            "integer",
                            other.kind_name(),
                        ))
                    }
                };
                Ok(Value::FmtCount(n))
            }
            Builtin::RangeInclusiveNew => {
                arity(2)?;
                let mut it = vals.into_iter();
                let start = it.next().expect("arity checked");
                let end = it.next().expect("arity checked");
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![start, end, Value::Bool(false)],
                })
            }
            Builtin::ThreadSpawn => {
                arity(1)?;
                let callable = vals.into_iter().next().expect("arity checked");
                self.spawn_callable(callable)
            }
            Builtin::ThreadBuilderSpawn => {
                // `(builder, f)`: the builder's name field is
                // registered for the new thread (so
                // `Thread::name` observes it); the callable is the
                // second argument. `Builder::spawn` returns
                // `io::Result<JoinHandle>`: the model's spawn
                // cannot fail, so the handle is the `Ok`.
                arity(2)?;
                let mut vals = vals.into_iter();
                let builder = vals.next().expect("arity checked");
                let callable = vals.next().expect("arity checked");
                let thread_name = builder_thread_name(&builder);
                let stack_limit = builder_thread_stack_size(&builder);
                let handle = self.spawn_callable(callable)?;
                if let Some(id) = join_handle_id(&handle) {
                    self.set_thread_name(id, thread_name);
                    self.set_thread_stack_limit(id, stack_limit);
                }
                Ok(ok_result(handle))
            }
            Builtin::ThreadScopedSpawn => {
                // `(&scope, f)`: the scope's identity is not
                // observed; the callable is the second argument.
                arity(2)?;
                let callable = vals.into_iter().nth(1).expect("arity checked");
                self.spawn_callable(callable)
            }
            Builtin::ThreadJoin | Builtin::ScopedThreadJoin => {
                arity(1)?;
                let Some(id) = join_handle_id(&vals[0]) else {
                    return Err(InterpError::type_mismatch(
                        "JoinHandle::join",
                        "a join handle value",
                        vals[0].kind_name(),
                    ));
                };
                match self.join_thread(frames, id)? {
                    Some(Ok(value)) => Ok(ok_result(value)),
                    // The error is the thread's panic payload as
                    // the `Box<dyn Any + Send>` the guest
                    // downcasts, in the same shape the
                    // `catch_unwind` cleanup produces.
                    Some(Err(payload)) => Ok(err_result(Value::DynRef {
                        base: Box::new(Value::Enum {
                            variant: 0,
                            fields: vec![(*payload.value).clone()],
                        }),
                        concrete: payload.ty.clone(),
                        traits: vec![
                            "std::any::Any".to_string(),
                            "std::marker::Send".to_string(),
                        ],
                        drop_glue: None,
                        type_id: payload.type_id,
                        vtable_methods: Vec::new(),
                    })),
                    // The joined thread is blocked and no
                    // scheduling round can unblock it: the join
                    // itself cannot complete (see
                    // `block_or_suspend`).
                    None => Err(self.block_or_suspend(
                        "JoinHandle::join",
                        "the joined thread is blocked and no pending thread can unblock it",
                    )),
                }
            }
            Builtin::ThreadScope => {
                arity(1)?;
                let callable = vals.into_iter().next().expect("arity checked");
                let scope_ref = Value::Ref(RefTarget::value(Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                }));
                // The scope body is an arbitrary guest program
                // and runs as a scheduled region, so the
                // threads it spawns interleave with it instead
                // of waiting for the scope-exit drain below.
                let result =
                    self.call_callable_scheduled(frames, &callable, vec![scope_ref])?;
                // Every thread spawned inside the scope must
                // finish before the scope returns; the cooperative
                // model runs all pending threads here, quantum by
                // quantum until none are left.
                self.drain_threads_to_completion(frames)?;
                Ok(result)
            }
            Builtin::ThreadCurrent => {
                arity(0)?;
                Ok(id_handle(self.current_thread_id()))
            }
            Builtin::ThreadIdOf => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                let thread = self.read_resolved(frames, &target)?;
                match channel_handle_id(&thread) {
                    Some(id) => Ok(id_handle(id)),
                    None => Err(InterpError::type_mismatch(
                        "Thread::id",
                        "a thread value",
                        thread.kind_name(),
                    )),
                }
            }
            Builtin::ThreadName => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                let thread = self.read_resolved(frames, &target)?;
                match channel_handle_id(&thread) {
                    Some(id) => Ok(match self.thread_name(id) {
                        Some(name) => some_value(Value::Str(name)),
                        None => none_value(),
                    }),
                    None => Err(InterpError::type_mismatch(
                        "Thread::name",
                        "a thread value",
                        thread.kind_name(),
                    )),
                }
            }
            Builtin::ThreadSleep => {
                arity(1)?;
                self.drain_threads(frames)?;
                Ok(Value::Unit)
            }
            Builtin::ThreadYield => {
                arity(0)?;
                self.drain_threads(frames)?;
                Ok(Value::Unit)
            }
            Builtin::ThreadPark => {
                arity(0)?;
                self.thread_park(frames, None, name)
            }
            Builtin::ThreadParkTimeout => {
                arity(1)?;
                let resolved = self.resolve_value(frames, &vals[0])?;
                // The model reads only whether the bound is
                // zero (a poll of the token) or not: every
                // non-zero bound ends the wait at the same
                // point (see `thread_park`), so a duration
                // whose shape the reader does not recognise
                // stands in as an arbitrary non-zero bound.
                let timeout = match duration_pair(&resolved) {
                    Some((secs, nanos)) => {
                        (secs.max(0) as u64, nanos.max(0) as u64)
                    }
                    None => (0, 1),
                };
                self.thread_park(frames, Some(timeout), name)
            }
            Builtin::ThreadUnpark => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                let thread = self.read_resolved(frames, &target)?;
                match channel_handle_id(&thread) {
                    Some(id) => {
                        self.set_park_token(id);
                        Ok(Value::Unit)
                    }
                    None => Err(InterpError::type_mismatch(
                        "Thread::unpark",
                        "a thread value",
                        thread.kind_name(),
                    )),
                }
            }
            Builtin::ThreadBuilderNew => {
                arity(0)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![none_value(), none_value()],
                })
            }
            Builtin::ThreadBuilderName | Builtin::ThreadBuilderStackSize => {
                arity(2)?;
                let field = if kind == Builtin::ThreadBuilderName { 0 } else { 1 };
                let mut vals = vals.into_iter();
                let builder = vals.next().expect("arity checked");
                let setting = vals.next().expect("arity checked");
                match builder {
                    Value::Enum { variant: 0, mut fields } if fields.len() == 2 => {
                        fields[field] = some_value(setting);
                        Ok(Value::Enum { variant: 0, fields })
                    }
                    other => Err(InterpError::type_mismatch(
                        "thread Builder",
                        "a builder value",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::DurationNew => {
                arity(2)?;
                let secs = uint_arg_bits(&vals[0], name)?;
                let nanos = uint_arg_bits(&vals[1], name)?;
                self.duration_value(secs, nanos)
            }
            Builtin::DurationFromSecs => {
                arity(1)?;
                self.duration_value(uint_arg_bits(&vals[0], name)?, 0)
            }
            Builtin::DurationFromMillis => {
                arity(1)?;
                let ms = uint_arg_bits(&vals[0], name)?;
                self.duration_value(ms / 1_000, (ms % 1_000) * 1_000_000)
            }
            Builtin::DurationFromMicros => {
                arity(1)?;
                let us = uint_arg_bits(&vals[0], name)?;
                self.duration_value(us / 1_000_000, (us % 1_000_000) * 1_000)
            }
            Builtin::DurationFromNanos => {
                arity(1)?;
                let ns = uint_arg_bits(&vals[0], name)?;
                self.duration_value(ns / 1_000_000_000, ns % 1_000_000_000)
            }
            Builtin::TestMainEnvArgs => {
                arity(1)?;
                let tests = self.read_test_descs(frames, &vals[0])?;
                // A `-Z` option is rejected before anything else
                // happens: the harness this models is a
                // stable-channel one, and libtest's own gate is
                // `is_nightly()` -- a nightly-built harness, or
                // `RUSTC_BOOTSTRAP` present in the environment.
                // The modelled harness is not nightly-built, so
                // the environment decides (the run's environment
                // is the guest's, as for `RUST_TEST_THREADS`
                // below): with no bootstrap, every `-Z` use is
                // refused, `-Z unstable-options` included. With
                // bootstrap, the one exception is `-Z
                // unstable-options` unlocking the json format, the
                // pair libtest's own JSON consumers pass; any
                // other `-Z` use, and unstable-options without a
                // json request, is refused as before. The json
                // format without the unlocking option is likewise
                // refused before listing or running.
                let raw_args: Vec<String> = self
            .guest_args()
            .iter()
            .map(|arg| String::from_utf8_lossy(arg).into_owned())
            .collect();
                let opts = HarnessOpts::parse(&raw_args);
                let bootstrap = std::env::var("RUSTC_BOOTSTRAP").is_ok();
                let mut unstable_options = false;
                let mut other_z = false;
                {
                    let mut j = 0;
                    while j < raw_args.len() {
                        let a = &raw_args[j];
                        if a == "-Zunstable-options" {
                            unstable_options = true;
                        } else if a == "-Z" {
                            if raw_args.get(j + 1).map(String::as_str)
                                == Some("unstable-options")
                            {
                                unstable_options = true;
                                j += 1;
                            } else {
                                other_z = true;
                            }
                        } else if a.starts_with("-Z") {
                            other_z = true;
                        }
                        j += 1;
                    }
                }
                if other_z || (unstable_options && (!bootstrap || !opts.json)) {
                    self.push_error_output(
                        "error: the option `Z` is only accepted on the nightly compiler\n",
                    );
                    return Ok(test_exit_code(101));
                }
                if opts.json && !unstable_options {
                    self.push_error_output(
                        "error: The \"json\" format is only accepted on the nightly \
                         compiler with -Z unstable-options\n",
                    );
                    return Ok(test_exit_code(101));
                }
                let mut selected: Vec<&TestDesc> = Vec::new();
                let mut filtered_out = 0usize;
                for test in &tests {
                    if opts.selected(test) {
                        selected.push(test);
                    } else {
                        filtered_out += 1;
                    }
                }
                if opts.list {
                    if opts.json {
                        // `--list` in the json format is the
                        // discovery event stream: a suite event,
                        // one `discovered` event per selected
                        // test carrying its descriptor's span,
                        // and the closing suite event with the
                        // counts.
                        self.push_output(
                            "{ \"type\": \"suite\", \"event\": \"discovery\" }\n",
                        );
                        let mut discovered_ignored = 0usize;
                        for test in &selected {
                            if test.ignore {
                                discovered_ignored += 1;
                            }
                            self.push_output(&format!(
                                "{{ \"type\": \"test\", \"event\": \"discovered\", \
                                 \"name\": \"{}\", \"ignore\": {}, \
                                 \"ignore_message\": \"{}\", \"source_path\": \"{}\", \
                                 \"start_line\": {}, \"start_col\": {}, \
                                 \"end_line\": {}, \"end_col\": {} }}\n",
                                json_escape(&test.name),
                                test.ignore,
                                json_escape(
                                    test.ignore_message.as_deref().unwrap_or("")
                                ),
                                json_escape(&test.source_file),
                                test.start_line,
                                test.start_col,
                                test.end_line,
                                test.end_col
                            ));
                        }
                        self.push_output(&format!(
                            "{{ \"type\": \"suite\", \"event\": \"completed\", \
                             \"tests\": {}, \"benchmarks\": 0, \"total\": {}, \
                             \"ignored\": {} }}\n",
                            selected.len(),
                            selected.len(),
                            discovered_ignored
                        ));
                        return Ok(test_exit_code(0));
                    }
                    // `--list`: one `{name}: test` line per
                    // selected test; the default format adds the
                    // count footer, the terse format does not.
                    for test in &selected {
                        self.push_output(&format!("{}: test\n", test.name));
                    }
                    if !opts.terse {
                        self.push_output(&format!(
                            "\n{} tests, 0 benchmarks\n",
                            selected.len()
                        ));
                    }
                    return Ok(test_exit_code(0));
                }
                let mut passed = 0usize;
                let mut failed = 0usize;
                let mut ignored = 0usize;
                // libtest's concurrency setup reads
                // RUST_TEST_THREADS before any test runs and
                // panics on a value that is not a positive
                // integer; the panic escapes the harness main.
                if let Ok(v) = std::env::var("RUST_TEST_THREADS") {
                    if !matches!(v.parse::<usize>(), Ok(n) if n > 0) {
                        return Err(InterpError::panic(format!(
                            "RUST_TEST_THREADS is `{v}`, should be a positive integer."
                        )));
                    }
                }
                let mut failure_names: Vec<String> = Vec::new();
                let mut failure_sections: Vec<String> = Vec::new();
                // The passing tests' `--show-output` accounts
                // (libtest's `ConsoleTestState::not_failures`):
                // each pass's name and captured output, in run
                // order, for the successes summary at the end
                // of the run.
                let mut success_accounts: Vec<(String, String)> = Vec::new();
                // TerseFormatter's progress state (libtest's
                // formatters/terse.rs): how many results have
                // been reported, and the column the short results
                // have filled on the current line. Only the
                // terse format reads them.
                let total_tests = selected.len();
                let mut terse_count = 0usize;
                let mut terse_column = 0usize;
                // The run's ignored test, for the terse finish
                // line a single ignored test with a message gets.
                let mut single_ignore: Option<(String, Option<String>)> = None;
                // std's panic output carries the backtrace note
                // on the process's first panic only (panicking's
                // FIRST_PANIC): a run whose first failure's
                // section includes it has spent it for every
                // later failure's section.
                let mut backtrace_note_pending = true;
                if opts.json {
                    self.push_output(&format!(
                        "{{ \"type\": \"suite\", \"event\": \"started\", \
                         \"test_count\": {} }}\n",
                        selected.len()
                    ));
                } else {
                    self.push_output(&format!(
                        "\nrunning {} {}\n",
                        selected.len(),
                        if selected.len() == 1 { "test" } else { "tests" }
                    ));
                }
                // The harness model synthesizes the panic
                // reports in its captured sections itself, so
                // the general default-report printing is
                // suppressed while the tests run: a test's panic
                // must not also print straight to the guest's
                // stderr.
                self.set_panic_report_suppressed(true);
                for test in selected {
                    let annotation = match test.should_panic {
                        ShouldPanic::No => "",
                        _ => " - should panic",
                    };
                    if test.ignore && !opts.runs_ignored() {
                        ignored += 1;
                        if opts.json {
                            self.push_output(&format!(
                                "{{ \"type\": \"test\", \"event\": \"started\", \
                                 \"name\": \"{}\" }}\n",
                                json_escape(&test.name)
                            ));
                            let message = match &test.ignore_message {
                                Some(msg) => format!(
                                    ", \"message\": \"{}\"",
                                    json_escape(msg)
                                ),
                                None => String::new(),
                            };
                            self.push_output(&format!(
                                "{{ \"type\": \"test\", \"name\": \"{}\", \
                                 \"event\": \"ignored\"{} }}\n",
                                json_escape(&test.name),
                                message
                            ));
                        } else if opts.terse {
                            // TerseFormatter::write_ignored: a
                            // short result, like a pass.
                            single_ignore = Some((
                                test.name.clone(),
                                test.ignore_message.clone(),
                            ));
                            self.push_output("i");
                            terse_count += 1;
                            terse_column += 1;
                            if terse_column % 88 == 87 {
                                self.push_output(&format!(
                                    " {}/{}\n",
                                    terse_count, total_tests
                                ));
                                terse_column = 0;
                            }
                        } else {
                            let note = match &test.ignore_message {
                                Some(msg) => format!(", {msg}"),
                                None => String::new(),
                            };
                            self.push_output(&format!(
                                "test {}{} ... ignored{}\n",
                                test.name, annotation, note
                            ));
                        }
                        continue;
                    }
                    if opts.json {
                        self.push_output(&format!(
                            "{{ \"type\": \"test\", \"event\": \"started\", \
                             \"name\": \"{}\" }}\n",
                            json_escape(&test.name)
                        ));
                    } else if opts.nocapture && !opts.terse {
                        self.push_output(&format!(
                            "test {}{} ... ",
                            test.name, annotation
                        ));
                    }
                    let out_mark = self.output_len();
                    let err_mark = self.error_output_len();
                    self.clear_panic_stamp();
                    // A test body is an arbitrary guest
                    // program: it runs as a scheduled region so
                    // its own threads interleave with it (see
                    // `Interpreter::drive_scheduled`).
                    let result = self.call_callable_scheduled(
                        frames,
                        &test.callable,
                        Vec::new(),
                    );
                    // The test's own output since the marks: in
                    // capture mode it is drained here (kept only
                    // for a failure's section); in nocapture mode
                    // it already flowed into the output live.
                    let (cap_out, cap_err) = if opts.nocapture {
                        (String::new(), String::new())
                    } else {
                        (
                            self.drain_output_since(out_mark),
                            self.drain_error_output_since(err_mark),
                        )
                    };
                    // `Some(body)` is the failure's section body:
                    // the captured output followed by the
                    // failure's account. A bare panic in
                    // nocapture mode has no section (its report
                    // went to the guest's stderr as it ran).
                    // For a test that panics as it was expected
                    // to, the panic report its captured output
                    // carries (the backtrace note included when
                    // this panic is the process's first): the
                    // `--show-output` accounts of a passing
                    // should-panic test, in both the json and
                    // the pretty formats, are its captured
                    // output plus this report.
                    let mut pass_note: Option<String> = None;
                    let failure_body: Option<Option<String>> = match result {
                        Ok(_) => match test.should_panic {
                            ShouldPanic::No => None,
                            _ => Some(Some(format!(
                                "{cap_out}{cap_err}note: test did not panic as expected at {}:{}:{}",
                                test.source_file, test.start_line, test.start_col
                            ))),
                        },
                        Err(InterpError::Panic { message, payload }) => {
                            self.note_panic_consumed();
                            if opts.nocapture
                                && !matches!(test.should_panic, ShouldPanic::No)
                            {
                                // In nocapture mode the panic's
                                // report goes to the guest's
                                // stderr as it runs (the default
                                // hook prints it at panic time):
                                // a should-panic test's panic is
                                // printed there too, whether the
                                // harness then counts the test as
                                // passed or failed. The backtrace
                                // note is spent on the process's
                                // first printed report.
                                let report = if backtrace_note_pending {
                                    backtrace_note_pending = false;
                                    panic_report_full(
                                        test,
                                        self.current_thread_id(),
                                        self.panic_location(),
                                        &message,
                                    )
                                } else {
                                    panic_report_head(
                                        test,
                                        self.current_thread_id(),
                                        self.panic_location(),
                                        &message,
                                    )
                                };
                                self.push_error_output(&report);
                            }
                            match &test.should_panic {
                                ShouldPanic::No => {
                                    if opts.nocapture {
                                        // The report the default
                                        // hook prints to the
                                        // guest's stderr as the
                                        // panic runs (see the
                                        // should-panic arms): in
                                        // nocapture mode there
                                        // is no captured section,
                                        // so this print is the
                                        // failure's only account.
                                        let report = if backtrace_note_pending {
                                            backtrace_note_pending = false;
                                            panic_report_full(
                                                test,
                                                self.current_thread_id(),
                                                self.panic_location(),
                                                &message,
                                            )
                                        } else {
                                            panic_report_head(
                                                test,
                                                self.current_thread_id(),
                                                self.panic_location(),
                                                &message,
                                            )
                                        };
                                        self.push_error_output(&report);
                                        Some(None)
                                    } else {
                                        let report = match self.panic_location() {
                                            Some((file, line, col)) => format!(
                                                "\nthread '{}' ({}) panicked at {}:{}:{}:\n{}\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n",
                                                test.name,
                                                self.current_thread_id(),
                                                file,
                                                line,
                                                col,
                                                message
                                            ),
                                            None => format!(
                                                "\nthread '{}' ({}) panicked:\n{}\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n",
                                                test.name,
                                                self.current_thread_id(),
                                                message
                                            ),
                                        };
                                        Some(Some(format!(
                                            "{cap_out}{cap_err}{report}"
                                        )))
                                    }
                                }
                                ShouldPanic::Yes => {
                                    pass_note = Some(panic_report_pass(
                                        test,
                                        self.current_thread_id(),
                                        self.panic_location(),
                                        &message,
                                        backtrace_note_pending,
                                    ));
                                    None
                                }
                                ShouldPanic::YesWithMessage(expected) => {
                                    let text = payload
                                        .as_ref()
                                        .and_then(|p| {
                                            self.resolve_string(frames, &p.value).ok()
                                        })
                                        .unwrap_or_else(|| message.clone());
                                    let is_string = match &payload {
                                        None => true,
                                        Some(p) => {
                                            p.ty == "&str"
                                                || p.ty == "std::string::String"
                                                || self
                                                    .resolve_string(frames, &p.value)
                                                    .is_ok()
                                        }
                                    };
                                    if !is_string {
                                        // libtest's note names the
                                        // payload by its `TypeId`
                                        // (the caught payload's
                                        // `TypeId`, formatted with
                                        // `{:?}`): when the raise
                                        // recorded the payload
                                        // type's TypeId segments,
                                        // render them in std's own
                                        // form - the segments are
                                        // this model's identity for
                                        // the type, standing in for
                                        // the compiler's hash.
                                        // Without recorded segments
                                        // the model has no identity
                                        // to name and keeps the
                                        // type's rendered name.
                                        let ty = match payload
                                            .as_ref()
                                            .and_then(|p| p.type_id)
                                        {
                                            Some((hi, lo)) => format!(
                                                "TypeId(0x{hi:016x}{lo:016x})"
                                            ),
                                            None => payload
                                                .as_ref()
                                                .map(|p| p.ty.clone())
                                                .unwrap_or_default(),
                                        };
                                        // The report the section
                                        // synthesises in capture
                                        // mode; in nocapture mode
                                        // it already went to the
                                        // guest's stderr as the
                                        // panic ran (the push
                                        // above), and the native
                                        // section carries the note
                                        // alone.
                                        let report = if opts.nocapture {
                                            String::new()
                                        } else {
                                            panic_report_full(
                                                test,
                                                self.current_thread_id(),
                                                self.panic_location(),
                                                &message,
                                            )
                                        };
                                        Some(Some(format!(
                                            "{cap_out}{cap_err}{report}note: expected panic with string value,\n found non-string value: `{ty}`\n     expected substring: \"{expected}\""
                                        )))
                                    } else if text.contains(expected.as_str()) {
                                        pass_note = Some(panic_report_pass(
                                            test,
                                            self.current_thread_id(),
                                            self.panic_location(),
                                            &message,
                                            backtrace_note_pending,
                                        ));
                                        None
                                    } else {
                                        // The report the section
                                        // synthesises in capture
                                        // mode; in nocapture mode
                                        // it already went to the
                                        // guest's stderr as the
                                        // panic ran (the push
                                        // above), and the native
                                        // section carries the note
                                        // alone.
                                        let report = if opts.nocapture {
                                            String::new()
                                        } else {
                                            panic_report_full(
                                                test,
                                                self.current_thread_id(),
                                                self.panic_location(),
                                                &message,
                                            )
                                        };
                                        Some(Some(format!(
                                            "{cap_out}{cap_err}{report}note: panic did not contain expected string\n      panic message: \"{text}\"\n expected substring: \"{expected}\""
                                        )))
                                    }
                                }
                            }
                        }
                        Err(e) => return Err(e),
                    };
                    if pass_note.is_some() {
                        // A should-panic test's expected panic
                        // is still a panic of the process: it
                        // spends the note's one appearance (its
                        // own account above carries the note
                        // when it was still unspent).
                        backtrace_note_pending = false;
                    }
                    // Spend the backtrace note on the first
                    // failure section that carries it (see
                    // backtrace_note_pending): later sections
                    // lose the note line, as the native run's do.
                    let failure_body = match failure_body {
                        Some(Some(body)) => {
                            const NOTE: &str = "note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n";
                            if body.contains(NOTE) {
                                if backtrace_note_pending {
                                    backtrace_note_pending = false;
                                    Some(Some(body))
                                } else {
                                    Some(Some(body.replace(NOTE, "")))
                                }
                            } else {
                                Some(Some(body))
                            }
                        }
                        other => other,
                    };
                    match failure_body {
                        None => {
                            passed += 1;
                            // The pass's `--show-output` account:
                            // its captured output plus the report
                            // of a panic it was expected to raise.
                            // Under `--nocapture` nothing was
                            // captured, so the account is empty
                            // and the pass lists by name alone.
                            success_accounts.push((
                                test.name.clone(),
                                if opts.nocapture {
                                    String::new()
                                } else {
                                    format!(
                                        "{cap_out}{cap_err}{}",
                                        pass_note.as_deref().unwrap_or("")
                                    )
                                },
                            ));
                            if opts.json {
                                // With `--show-output`, a passing
                                // test's event carries its
                                // captured output (plus the report
                                // of a panic it was expected to
                                // raise); without it, or under
                                // `--nocapture`, the field is
                                // omitted.
                                let captured = format!(
                                    "{cap_out}{cap_err}{}",
                                    pass_note.as_deref().unwrap_or("")
                                );
                                let stdout = if opts.show_output
                                    && !opts.nocapture
                                    && !captured.is_empty()
                                {
                                    format!(
                                        ", \"stdout\": \"{}\"",
                                        json_escape(&captured)
                                    )
                                } else {
                                    String::new()
                                };
                                self.push_output(&format!(
                                    "{{ \"type\": \"test\", \"name\": \"{}\", \
                                     \"event\": \"ok\"{} }}\n",
                                    json_escape(&test.name),
                                    stdout
                                ));
                            } else if opts.terse {
                                // TerseFormatter::write_ok.
                                self.push_output(".");
                                terse_count += 1;
                                terse_column += 1;
                                if terse_column % 88 == 87 {
                                    self.push_output(&format!(
                                        " {}/{}\n",
                                        terse_count, total_tests
                                    ));
                                    terse_column = 0;
                                }
                            } else if opts.nocapture {
                                self.push_output("ok\n");
                            } else {
                                self.push_output(&format!(
                                    "test {}{} ... ok\n",
                                    test.name, annotation
                                ));
                            }
                        }
                        Some(body) => {
                            failed += 1;
                            failure_names.push(test.name.clone());
                            if let Some(body) = &body {
                                failure_sections.push(format!(
                                    "\n---- {} stdout ----\n{}",
                                    test.name, body
                                ));
                            }
                            if opts.json {
                                // A failed test's event carries
                                // its failure account as `stdout`
                                // (the captured output followed by
                                // the account itself); under
                                // `--nocapture` nothing was
                                // captured, so the field is omitted.
                                let stdout = if opts.nocapture {
                                    String::new()
                                } else {
                                    format!(
                                        ", \"stdout\": \"{}\"",
                                        json_escape(
                                            body.as_deref().unwrap_or("")
                                        )
                                    )
                                };
                                self.push_output(&format!(
                                    "{{ \"type\": \"test\", \"name\": \"{}\", \
                                     \"event\": \"failed\"{} }}\n",
                                    json_escape(&test.name),
                                    stdout
                                ));
                            } else if opts.terse {
                                // TerseFormatter::write_failed:
                                // the progress line first (unless
                                // the column is already fresh),
                                // then the failure on its own
                                // line, without the should-panic
                                // annotation.
                                if terse_column != 0 {
                                    self.push_output(&format!(
                                        " {}/{}\n",
                                        terse_count, total_tests
                                    ));
                                    terse_column = 0;
                                }
                                terse_count += 1;
                                self.push_output(&format!(
                                    "{} --- FAILED\n",
                                    test.name
                                ));
                            } else if opts.nocapture {
                                self.push_output("FAILED\n");
                            } else {
                                self.push_output(&format!(
                                    "test {}{} ... FAILED\n",
                                    test.name, annotation
                                ));
                            }
                        }
                    }
                }
                self.set_panic_report_suppressed(false);
                if opts.json {
                    let event = if failed == 0 { "ok" } else { "failed" };
                    self.push_output(&format!(
                        "{{ \"type\": \"suite\", \"event\": \"{}\", \
                         \"passed\": {}, \"failed\": {}, \"ignored\": {}, \
                         \"measured\": 0, \"filtered_out\": {}, \
                         \"exec_time\": \"0.00\" }}\n",
                        event, passed, failed, ignored, filtered_out
                    ));
                } else {
                    if opts.show_output {
                        // PrettyFormatter::write_successes:
                        // libtest's `write_results` over the
                        // run's not-failures -- the header, one
                        // `---- name stdout ----` block per pass
                        // whose captured output is non-empty
                        // (in run order), the header again, and
                        // the passes' names, sorted. It prints
                        // whenever `--show-output` is set,
                        // whether or not anything failed, ahead
                        // of the failures account.
                        self.push_output("\nsuccesses:\n");
                        let mut success_stdouts = String::new();
                        for (name, output) in &success_accounts {
                            if !output.is_empty() {
                                success_stdouts.push_str(&format!(
                                    "---- {name} stdout ----\n{output}\n"
                                ));
                            }
                        }
                        if !success_stdouts.is_empty() {
                            self.push_output("\n");
                            self.push_output(&success_stdouts);
                        }
                        self.push_output("\nsuccesses:\n");
                        let mut success_names: Vec<&str> = success_accounts
                            .iter()
                            .map(|(name, _)| name.as_str())
                            .collect();
                        success_names.sort_unstable();
                        for name in success_names {
                            self.push_output(&format!("    {name}\n"));
                        }
                    }
                    if !failure_names.is_empty() {
                        self.push_output("\nfailures:\n");
                        for section in &failure_sections {
                            self.push_output(section);
                        }
                        if !failure_sections.is_empty() {
                            self.push_output("\n");
                        }
                        self.push_output("\nfailures:\n");
                        for name in &failure_names {
                            self.push_output(&format!("    {name}\n"));
                        }
                    }
                    let verdict = if failed == 0 { "ok" } else { "FAILED" };
                    self.push_output(&format!(
                        "\ntest result: {verdict}. {passed} passed; {failed} failed; \
                         {ignored} ignored; 0 measured; {filtered_out} filtered out; \
                         finished in 0.00s\n\n"
                    ));
                    // TerseFormatter::write_run_finish's special
                    // case: a run of exactly one test, ignored
                    // with a message, restates the reason after
                    // the summary.
                    if opts.terse && total_tests == 1 && ignored == 1 {
                        if let Some((name, Some(msg))) = &single_ignore {
                            self.push_output(&format!(
                                "test: {name}, ignore_message: {msg}\n\n"
                            ));
                        }
                    }
                }
                // `ExitCode`: success is 0, any failure is 101,
                // as in libtest.
                let byte = if failed == 0 { 0 } else { 101 };
                Ok(test_exit_code(byte))
            }
            Builtin::TestMainEnvArgsAbort => {
                arity(1)?;
                let tests = self.read_test_descs(frames, &vals[0])?;
                // A spawned test process of the harness
                // re-enters this entry with the invoker
                // variable set (libtest's
                // `SECONDARY_TEST_INVOKER_VAR`): the entry
                // checks it before anything else, removes it
                // (and the bench-mode companion) from the
                // environment the test then observes, runs the
                // named test in this process, and leaves by
                // the child's exit or abort — it never
                // returns. The run's environment is the
                // guest's, as elsewhere in the harness model.
                if let Ok(name) = std::env::var("__RUST_TEST_INVOKE") {
                    std::env::remove_var("__RUST_TEST_INVOKE");
                    if std::env::var("__RUST_TEST_BENCH_BENCHMARKS").is_ok() {
                        std::env::remove_var("__RUST_TEST_BENCH_BENCHMARKS");
                    }
                    let test = tests.iter().find(|t| t.name == name).ok_or_else(|| {
                        InterpError::panic(format!(
                            "couldn't find a test with the provided name '{name}'"
                        ))
                    })?;
                    return self.run_abort_secondary(frames, test);
                }
                self.run_test_main_abort(frames, tests)
            }
            Builtin::PanicSetHook => {
                arity(1)?;
                self.set_panic_hook(vals.into_iter().next().expect("one argument"));
                Ok(Value::Unit)
            }
            Builtin::PanicSetBacktraceStyle => {
                arity(1)?;
                self.set_backtrace_style(vals.into_iter().next().expect("one argument"));
                Ok(Value::Unit)
            }
            Builtin::PanicAlwaysAbort => {
                arity(0)?;
                self.note_always_abort();
                Ok(Value::Unit)
            }
            Builtin::PanicTakeHook => {
                arity(0)?;
                Ok(self.take_panic_hook())
            }
            Builtin::PanicUpdateHook => {
                arity(1)?;
                self.update_panic_hook(
                    vals.into_iter().next().expect("one argument"),
                );
                Ok(Value::Unit)
            }
            Builtin::CallerLocation => {
                // A `#[track_caller]` function's location is the
                // one it received (bound to its synthetic local);
                // any other function's is the stamp appended to
                // this call.
                if let Some(idx) = body.caller_location_local {
                    let frame = frames.last().expect("frame");
                    let received = frame.locals.get(idx).cloned().unwrap_or(Value::Uninit);
                    if !matches!(received, Value::Uninit) {
                        return Ok(received);
                    }
                }
                arity(1)?;
                Ok(vals.into_iter().next().expect("one argument"))
            }
            Builtin::InstantNow => {
                arity(0)?;
                Ok(instant_value(monotonic_now()))
            }
            Builtin::SystemTimeNow => {
                arity(0)?;
                Ok(instant_value(wall_now()))
            }
            Builtin::HashmapRandomKeys => {
                arity(0)?;
                let (k0, k1) = fresh_hash_keys();
                Ok(Value::Tuple(vec![
                    Value::Uint(u128::from(k0)),
                    Value::Uint(u128::from(k1)),
                ]))
            }
            Builtin::ProcessCommandNew => {
                arity(1)?;
                let program = self.os_str_bytes(frames, &vals[0])?;
                Ok(command_value(self.new_command(command_state_new(program))))
            }
            Builtin::ProcessCommandArg => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let arg = self.os_str_bytes(frames, &vals[1])?;
                self.with_command(id, |state| {
                    state.saw_nul |= arg.contains(&0);
                    state.args.push(arg);
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(Value::Unit)
            }
            Builtin::ProcessCommandUid => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let uid = as_u128(&vals[1], name)? as u32;
                self.with_command(id, |state| {
                    state.uid = Some(uid);
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::ProcessCommandGid => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let gid = as_u128(&vals[1], name)? as u32;
                self.with_command(id, |state| {
                    state.gid = Some(gid);
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::ProcessCommandGroups => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let (items, _) = self.slice_elements(frames, &vals[1])?;
                let mut groups = Vec::with_capacity(items.len());
                for item in &items {
                    groups.push(as_u128(item, name)? as u32);
                }
                self.with_command(id, |state| {
                    state.groups = Some(groups);
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::ProcessCommandArg0 => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let arg0 = self.os_str_bytes(frames, &vals[1])?;
                self.with_command(id, |state| {
                    state.argv0 = Some(arg0);
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                // The sys setter returns unit; the public
                // `CommandExt::arg0` returns the command
                // itself for chaining.
                if name.contains("set_arg_0") {
                    Ok(Value::Unit)
                } else {
                    Ok(vals.into_iter().next().expect("arity checked"))
                }
            }
            Builtin::FromRawFd => {
                arity(1)?;
                self.exec_from_raw_fd(frames, name, generic_args, &vals)
            }
            Builtin::ProcessCommandCwd => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let dir = self.os_str_bytes(frames, &vals[1])?;
                self.with_command(id, |state| {
                    state.saw_nul |= dir.contains(&0);
                    state.cwd = Some(dir);
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(Value::Unit)
            }
            Builtin::ProcessCommandEnvMut => {
                arity(1)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                // The environment editor is a handle value
                // carrying the command's id, in a reference the
                // `CommandEnv` builtins resolve back.
                Ok(Value::Ref(RefTarget::value(id_handle(id))))
            }
            Builtin::ProcessCommandEnvSet => {
                arity(3)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let key = self.os_str_bytes(frames, &vals[1])?;
                let val = self.os_str_bytes(frames, &vals[2])?;
                self.with_command(id, |state| {
                    state.saw_nul |= key.contains(&0) || val.contains(&0);
                    state.envs.push((key, Some(val)));
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(Value::Unit)
            }
            Builtin::ProcessCommandEnvRemove => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let key = self.os_str_bytes(frames, &vals[1])?;
                self.with_command(id, |state| {
                    state.envs.push((key, None));
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(Value::Unit)
            }
            Builtin::ProcessCommandEnvClear => {
                arity(1)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                self.with_command(id, |state| {
                    state.env_clear = true;
                    state.envs.clear();
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(Value::Unit)
            }
            Builtin::ProcessCommandStdio => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                self.with_command(id, |state| record_stdio_config(state, name, &vals[1]))
                    .ok_or_else(|| {
                        unsupported(format!("call to {name} on a stale command handle"))
                    })?;
                Ok(Value::Unit)
            }
            Builtin::ProcessSpawn => {
                // The public entry points (`Command::spawn`,
                // `Command::status`) pass only the receiver; the
                // sys entry point also passes the default stdio
                // and a flag. The outcome is the same.
                if vals.len() != 1 && vals.len() != 3 {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "1 or 3 argument(s)",
                        format!("{} argument(s)", vals.len()),
                    ));
                }
                let id = self.command_id_of(frames, &vals[0], name)?;
                let state = self
                    .command_snapshot(id)
                    .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                match spawn_gate_error(self.tool_interception(), &state) {
                    Some(error) => Ok(err_result(error)),
                    None => {
                        // `Command::status` runs the program and
                        // reports its exit status: a configured
                        // tool program through the tool runner,
                        // any other program through the process
                        // host when one is configured. A `Child`
                        // handle for a spawned tool cannot be
                        // fabricated, so tool interception never
                        // answers `spawn` itself.
                        if name == "std::process::Command::status" {
                            check_stdio_modelled(&state)?;
                            let outcome =
                                self.tool_or_host_outcome(&state, false)?;
                            match outcome {
                                Some(Ok(outcome)) => {
                                    // `status` inherits the
                                    // child's streams (it collects
                                    // no `Output`): what the
                                    // program printed joins the
                                    // guest's own streams here, in
                                    // invocation order, before the
                                    // status is reported -- for
                                    // the streams whose
                                    // configuration inherits them.
                                    // A stream the guest
                                    // redirected to null or a pipe
                                    // is not the guest's own, and
                                    // its bytes go nowhere
                                    // observable.
                                    match stdio_route(state.stdout_stdio, false) {
                                        StdioRoute::Relay => {
                                            self.push_output(
                                                &String::from_utf8_lossy(
                                                    &outcome.stdout,
                                                ),
                                            );
                                        }
                                        StdioRoute::File => {
                                            self.write_stdio_file(
                                                state.stdout_stdio,
                                                &outcome.stdout,
                                            );
                                        }
                                        _ => {}
                                    }
                                    match stdio_route(state.stderr_stdio, false) {
                                        StdioRoute::Relay => {
                                            self.push_error_output(
                                                &String::from_utf8_lossy(
                                                    &outcome.stderr,
                                                ),
                                            );
                                        }
                                        StdioRoute::File => {
                                            self.write_stdio_file(
                                                state.stderr_stdio,
                                                &outcome.stderr,
                                            );
                                        }
                                        _ => {}
                                    }
                                    return Ok(ok_result(exit_status_value(
                                        outcome.exit_code,
                                    )));
                                }
                                Some(Err(error)) => {
                                    return Ok(err_result(error));
                                }
                                None => {}
                            }
                        }
                        // `Command::spawn` through the process
                        // host starts a real child and hands the
                        // guest its `Child` handle.
                        if name == "std::process::Command::spawn" {
                            if let Some(result) = self.host_spawn(&state)? {
                                return Ok(result);
                            }
                        }
                        Err(unsupported(format!(
                            "spawning program `{}` is not runtime-supported (the process model does not start child processes)",
                            String::from_utf8_lossy(&state.program)
                        )))
                    }
                }
            }
            Builtin::ProcessCommandNewPublic => {
                arity(1)?;
                let program = self.os_str_bytes(frames, &vals[0])?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![command_value(self.new_command(command_state_new(program)))],
                })
            }
            Builtin::ProcessCommandArgPublic => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let arg = self.os_str_bytes(frames, &vals[1])?;
                self.with_command(id, |state| {
                    state.saw_nul |= arg.contains(&0);
                    state.args.push(arg);
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::ProcessCommandCwdPublic => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let dir = self.os_str_bytes(frames, &vals[1])?;
                self.with_command(id, |state| {
                    state.saw_nul |= dir.contains(&0);
                    state.cwd = Some(dir);
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::ProcessCommandEnvPublic => {
                arity(3)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let key = self.os_str_bytes(frames, &vals[1])?;
                let val = self.os_str_bytes(frames, &vals[2])?;
                self.with_command(id, |state| {
                    state.saw_nul |= key.contains(&0) || val.contains(&0);
                    state.envs.push((key, Some(val)));
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::ProcessCommandEnvRemovePublic => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let key = self.os_str_bytes(frames, &vals[1])?;
                self.with_command(id, |state| {
                    state.envs.push((key, None));
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::ProcessCommandEnvClearPublic => {
                arity(1)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                self.with_command(id, |state| {
                    state.env_clear = true;
                    state.envs.clear();
                })
                .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::ProcessCommandStdioPublic => {
                arity(2)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                self.with_command(id, |state| record_stdio_config(state, name, &vals[1]))
                    .ok_or_else(|| {
                        unsupported(format!("call to {name} on a stale command handle"))
                    })?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::ProcessOutput => {
                arity(1)?;
                let id = self.command_id_of(frames, &vals[0], name)?;
                let state = self
                    .command_snapshot(id)
                    .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                match spawn_gate_error(self.tool_interception(), &state) {
                    Some(error) => Ok(err_result(error)),
                    None => {
                        check_stdio_modelled(&state)?;
                        let outcome =
                            self.tool_or_host_outcome(&state, true)?;
                        match outcome {
                            Some(Ok(outcome)) => {
                                // `output` collects the streams
                                // it pipes -- by default and on
                                // an explicit pipe. A stream the
                                // guest redirected elsewhere is
                                // not collected: an inherited
                                // one joins the guest's own
                                // stream here, a nulled one is
                                // dropped, and the `Output`
                                // carries no bytes for either.
                                let stdout = match stdio_route(state.stdout_stdio, true) {
                                    StdioRoute::Collect => outcome.stdout.clone(),
                                    StdioRoute::Relay => {
                                        self.push_output(&String::from_utf8_lossy(
                                            &outcome.stdout,
                                        ));
                                        Vec::new()
                                    }
                                    StdioRoute::File => {
                                        self.write_stdio_file(
                                            state.stdout_stdio,
                                            &outcome.stdout,
                                        );
                                        Vec::new()
                                    }
                                    StdioRoute::Discard => Vec::new(),
                                };
                                let stderr = match stdio_route(state.stderr_stdio, true) {
                                    StdioRoute::Collect => outcome.stderr.clone(),
                                    StdioRoute::Relay => {
                                        self.push_error_output(&String::from_utf8_lossy(
                                            &outcome.stderr,
                                        ));
                                        Vec::new()
                                    }
                                    StdioRoute::File => {
                                        self.write_stdio_file(
                                            state.stderr_stdio,
                                            &outcome.stderr,
                                        );
                                        Vec::new()
                                    }
                                    StdioRoute::Discard => Vec::new(),
                                };
                                Ok(ok_result(process_output_value(
                                    outcome.exit_code,
                                    &stdout,
                                    &stderr,
                                )))
                            }
                            Some(Err(error)) => Ok(err_result(error)),
                            None => Err(unsupported(format!(
                                "running program `{}` to collect its output is not runtime-supported (the process model does not start child processes)",
                                String::from_utf8_lossy(&state.program)
                            ))),
                        }
                    }
                }
            }
            Builtin::ProcessCommandExec => {
                // The public entry passes only the receiver;
                // the sys entry also passes the default stdio
                // (`Stdio::Inherit` from `CommandExt::exec`),
                // which the model's routing already assumes.
                if vals.len() != 1 && vals.len() != 2 {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "1 or 2 argument(s)",
                        format!("{} argument(s)", vals.len()),
                    ));
                }
                let id = self.command_id_of(frames, &vals[0], name)?;
                let state = self
                    .command_snapshot(id)
                    .ok_or_else(|| unsupported(format!("call to {name} on a stale command handle")))?;
                match spawn_gate_error(self.tool_interception(), &state) {
                    // `exec` returns the `io::Error` itself,
                    // not a `Result`: a failure to start the
                    // program is the call's normal return.
                    Some(error) => Ok(error),
                    None => {
                        check_stdio_modelled(&state)?;
                        let outcome = match tool_outcome(
                            self.tool_interception(),
                            &state,
                            false,
                        )? {
                            Some(outcome) => Some(Ok(outcome)),
                            None => self.host_outcome(&state, false)?,
                        };
                        match outcome {
                            Some(Ok(outcome)) => {
                                // The program replaces the guest:
                                // what it prints joins the
                                // guest's own streams (the
                                // streams whose configuration
                                // inherits them), and the run
                                // ends with the program's exit
                                // code, as the replaced process's
                                // own end would.
                                match stdio_route(state.stdout_stdio, false) {
                                    // The relay goes through
                                    // the guest's own stdout
                                    // write path, not the
                                    // captured-output buffer:
                                    // the replaced process's
                                    // stdout IS the guest's
                                    // descriptor 1, routing and
                                    // write-through sink
                                    // included (a run the
                                    // process host re-ran as a
                                    // child streams it to the
                                    // parent's pipe). A write
                                    // failure there cannot
                                    // change the outcome: the
                                    // program has already run
                                    // to its end.
                                    StdioRoute::Relay => {
                                        let _ = self.write_stdout(&outcome.stdout);
                                    }
                                    StdioRoute::File => {
                                        self.write_stdio_file(
                                            state.stdout_stdio,
                                            &outcome.stdout,
                                        );
                                    }
                                    _ => {}
                                }
                                match stdio_route(state.stderr_stdio, false) {
                                    StdioRoute::Relay => {
                                        self.push_error_output(
                                            &String::from_utf8_lossy(
                                                &outcome.stderr,
                                            ),
                                        );
                                    }
                                    StdioRoute::File => {
                                        self.write_stdio_file(
                                            state.stderr_stdio,
                                            &outcome.stderr,
                                        );
                                    }
                                    _ => {}
                                }
                                Err(InterpError::ProcessExit {
                                    code: outcome.exit_code,
                                })
                            }
                            Some(Err(error)) => Ok(error),
                            None => Err(unsupported(format!(
                                "replacing the process with program `{}` is not runtime-supported (the process model does not start child processes)",
                                String::from_utf8_lossy(&state.program)
                            ))),
                        }
                    }
                }
            }
            Builtin::ProcessChildWait => {
                arity(1)?;
                let id = self.child_id_of(frames, &vals[0], name)?;
                let slots = self.child_slots_of(frames, &vals[0])?;
                match self.child_finish(id, true, slots)? {
                    ChildWait::Finished(code) => {
                        Ok(ok_result(exit_status_value(code)))
                    }
                    ChildWait::IoError(error) => Ok(io_result_err(error)),
                    ChildWait::Running => Err(unsupported(
                        "child wait did not observe the child's end"
                            .to_string(),
                    )),
                }
            }
            Builtin::ProcessChildTryWait => {
                arity(1)?;
                let id = self.child_id_of(frames, &vals[0], name)?;
                let slots = self.child_slots_of(frames, &vals[0])?;
                match self.child_finish(id, false, slots)? {
                    ChildWait::Finished(code) => Ok(ok_result(some_value(
                        exit_status_value(code),
                    ))),
                    ChildWait::Running => Ok(ok_result(none_value())),
                    ChildWait::IoError(error) => Ok(io_result_err(error)),
                }
            }
            Builtin::ProcessChildKill => {
                arity(1)?;
                let id = self.child_id_of(frames, &vals[0], name)?;
                let live = self
                    .with_child(id, |state| {
                        (state.host_id, state.finished.is_some())
                    })
                    .ok_or_else(|| {
                        unsupported(format!(
                            "call to {name} on a stale child handle"
                        ))
                    })?;
                // A child whose end was already observed is
                // gone; std's `kill` on an exited child reports
                // success, and the host entry is already reaped.
                if live.1 {
                    return Ok(io_result_ok(Value::Unit));
                }
                let host = self.process_host().cloned().ok_or_else(|| {
                    unsupported(
                        "child process without a process host".to_string(),
                    )
                })?;
                match host.0.kill(live.0) {
                    Ok(()) => Ok(io_result_ok(Value::Unit)),
                    Err(crate::interp::ProcessHostError::Io(e)) => {
                        Ok(io_result_err(host_io_error(&e)))
                    }
                    Err(crate::interp::ProcessHostError::Failed(reason)) => {
                        Err(unsupported(reason))
                    }
                }
            }
            Builtin::ProcessChildWaitWithOutput => {
                arity(1)?;
                let id = self.child_id_of(frames, &vals[0], name)?;
                let slots = self.child_slots_of(frames, &vals[0])?;
                match self.child_finish(id, true, slots)? {
                    ChildWait::Finished(code) => {
                        // The call consumes the `Child`: the
                        // entry leaves the table and the host
                        // forgets its side. The collected bytes
                        // are the piped streams the `Child` still
                        // held (a stream the guest took out is
                        // the guest's to read and contributes
                        // nothing; an inherited stream was
                        // relayed at the end's first observation
                        // and is not part of the `Output`, as in
                        // std).
                        let state = self.remove_child(id).ok_or_else(|| {
                            unsupported(format!(
                                "call to {name} on a stale child handle"
                            ))
                        })?;
                        if let Some(host) = self.process_host() {
                            host.0.release(state.host_id);
                        }
                        let empty: &[u8] = &[];
                        let stdout = if state.stdout_stdio
                            == Some(crate::interp::StdioConfig::Piped)
                            && slots.1
                        {
                            &state.stdout
                        } else {
                            empty
                        };
                        let stderr = if state.stderr_stdio
                            == Some(crate::interp::StdioConfig::Piped)
                            && slots.2
                        {
                            &state.stderr
                        } else {
                            empty
                        };
                        Ok(ok_result(process_output_value(
                            code, stdout, stderr,
                        )))
                    }
                    ChildWait::IoError(error) => Ok(io_result_err(error)),
                    ChildWait::Running => Err(unsupported(
                        "child wait did not observe the child's end"
                            .to_string(),
                    )),
                }
            }
            Builtin::BarrierNew => {
                arity(1)?;
                let parties = match self
                    .resolve_value(frames, &vals[0])?
                {
                    Value::Uint(n) => n as usize,
                    Value::Int(n) => n as usize,
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "a party count",
                            other.kind_name(),
                        ))
                    }
                };
                Ok(barrier_value(self.new_barrier(parties)))
            }
            Builtin::BarrierWait => {
                arity(1)?;
                let id = self.barrier_id_of(frames, &vals[0], name)?;
                let (generation, leader) =
                    self.barrier_arrive(id).ok_or_else(|| {
                        unsupported(format!(
                            "call to {name} on a stale barrier handle"
                        ))
                    })?;
                if leader {
                    return Ok(barrier_wait_result_value(true));
                }
                // This party waits for the generation to be
                // released: the remaining parties are threads
                // of this run, and this is a place the running
                // thread would block, so they get their turn
                // (the cooperative scheduling point).
                self.drain_threads(frames)?;
                match self.barrier_generation(id) {
                    Some(released) if released != generation => {
                        Ok(barrier_wait_result_value(false))
                    }
                    _ => Err(unsupported(
                        "Barrier::wait: the other parties never arrived (no runnable thread reached the barrier)"
                            .to_string(),
                    )),
                }
            }
            Builtin::SysMutexLockContended => {
                arity(1)?;
                self.sys_mutex_lock_contended(frames, &vals[0], name)
            }
            Builtin::SysMutexWake => {
                arity(1)?;
                // The caller's swap has already published the
                // unlocked word; a waiter in this model
                // observes the word at its next activation, so
                // there is no wait queue to notify.
                Ok(Value::Unit)
            }
            Builtin::SysRwLockReadContended => {
                arity(1)?;
                self.sys_rwlock_read_contended(frames, &vals[0], name)
            }
            Builtin::SysRwLockWriteContended => {
                arity(1)?;
                self.sys_rwlock_write_contended(frames, &vals[0], name)
            }
            Builtin::SysRwLockWakeWriterOrReaders => {
                arity(2)?;
                let state = match self.resolve_value(frames, &vals[1])? {
                    Value::Uint(n) => n as u32,
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "an rwlock state word",
                            other.kind_name(),
                        ))
                    }
                };
                self.sys_rwlock_wake_writer_or_readers(frames, &vals[0], state, name)
            }
            Builtin::FutexWait => {
                arity(3)?;
                let expected = match self.resolve_value(frames, &vals[1])? {
                    Value::Uint(n) => n as u32,
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "a futex expected word",
                            other.kind_name(),
                        ))
                    }
                };
                let has_timeout = matches!(
                    self.resolve_value(frames, &vals[2])?,
                    Value::Enum { variant: 1, .. }
                );
                self.futex_wait(frames, &vals[0], expected, has_timeout, name)
            }
            Builtin::FutexWake => {
                arity(1)?;
                // A waiter in this model observes the word at
                // its next activation, so there is no wait
                // queue to notify; some platforms always
                // report that none was woken, as this does.
                Ok(Value::Bool(false))
            }
            Builtin::FutexWakeAll => {
                arity(1)?;
                Ok(Value::Unit)
            }
            Builtin::OnceCall => {
                arity(3)?;
                self.sys_once_call(frames, &vals, name)
            }
            Builtin::BufGuardRemaining => {
                arity(1)?;
                self.buf_guard_remaining(frames, &vals[0], name)
            }
            Builtin::BacktraceCapture => {
                arity(0)?;
                Ok(backtrace_value(frames, false))
            }
            Builtin::BacktraceForceCapture => {
                arity(0)?;
                Ok(backtrace_value(frames, true))
            }
            Builtin::BacktraceStatus => {
                arity(1)?;
                let resolved = self.resolve_value(frames, &vals[0])?;
                match backtrace_status_of(&resolved) {
                    Some(status) => Ok(Value::Enum {
                        variant: status,
                        fields: Vec::new(),
                    }),
                    None => Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "a backtrace value",
                        resolved.kind_name(),
                    )),
                }
            }
            Builtin::ProcessChildId => {
                arity(1)?;
                let id = self.child_id_of(frames, &vals[0], name)?;
                let pid = self
                    .with_child(id, |state| state.pid)
                    .ok_or_else(|| {
                        unsupported(format!(
                            "call to {name} on a stale child handle"
                        ))
                    })?;
                Ok(Value::Uint(u128::from(pid)))
            }
            Builtin::CurrentExe => {
                arity(0)?;
                let path = std::env::current_exe().map_err(|e| {
                    unsupported(format!("current_exe is not available in this run: {e}"))
                })?;
                Ok(ok_result(pathbuf_value(&host_path_bytes(&path)?)))
            }
            Builtin::ProcessId => {
                arity(0)?;
                Ok(Value::Uint(u128::from(std::process::id())))
            }
            Builtin::IoGetOsFunctions => {
                arity(0)?;
                Ok(Value::Ref(RefTarget::value(os_functions_table())))
            }
            Builtin::IoSetOsFunctions => {
                arity(1)?;
                Ok(Value::Unit)
            }
            Builtin::ToSocketAddrs => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                self.exec_to_socket_addrs(frames, self_ty, &vals[0])
            }
            Builtin::IoDecodeErrorKind => {
                arity(1)?;
                let code = match vals[0] {
                    Value::Int(v) => v as i32,
                    Value::Uint(v) => v as i32,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "an errno integer",
                            other.kind_name(),
                        ))
                    }
                };
                Ok(Value::Enum {
                    variant: decode_error_kind_unix(code),
                    fields: Vec::new(),
                })
            }
            Builtin::IoFormatOsError => {
                arity(2)?;
                let code = match vals[0] {
                    Value::Int(v) => v as i32,
                    Value::Uint(v) => v as i32,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "an errno integer",
                            other.kind_name(),
                        ))
                    }
                };
                let id = self.formatter_id_of(frames, &vals[1])?;
                self.formatter_push(id, &os_error_message(code));
                Ok(fmt_result_ok())
            }
            Builtin::IoIsInterrupted => {
                arity(1)?;
                let code = match vals[0] {
                    Value::Int(v) => v as i32,
                    Value::Uint(v) => v as i32,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "an errno integer",
                            other.kind_name(),
                        ))
                    }
                };
                Ok(Value::Bool(code == 4))
            }
            Builtin::IoCustomOwnerFromBox => {
                arity(2)?;
                // The `Custom` payload std assembles: the kind,
                // the boxed trait object (the conversion's result
                // unsizing already formed the trait-object value),
                // and the two drop functions std stores beside
                // them (one name, two instantiations: dropping a
                // `Box<dyn Error>` and dropping a `Box<Custom>`).
                let payload = match &vals[1] {
                    Value::DynRef { .. } => vals[1].clone(),
                    Value::Enum { fields, .. } if fields.len() == 1 => {
                        match &fields[0] {
                            Value::DynRef { .. } => fields[0].clone(),
                            other => {
                                return Err(InterpError::type_mismatch(
                                    format!("builtin {name}"),
                                    "a boxed trait object",
                                    other.kind_name(),
                                ))
                            }
                        }
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "a boxed trait object",
                            other.kind_name(),
                        ))
                    }
                };
                let custom = Value::Enum {
                    variant: 0,
                    fields: vec![
                        vals[0].clone(),
                        payload,
                        Value::FnPtr(IO_ERROR_DROP_BOX_RAW_NAME.to_string()),
                        Value::FnPtr(IO_ERROR_DROP_BOX_RAW_NAME.to_string()),
                    ],
                };
                // The payload lives in its own allocation (the
                // box the packed word will name), and the owner is
                // the one-field aggregate around the allocation
                // pointer, the shape `Box::into_non_null` +
                // `CustomOwner::from_raw` produce.
                let id = self.alloc_insert(custom);
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Ref(RefTarget::Alloc {
                        id,
                        projection: Vec::new(),
                    })],
                })
            }
            Builtin::IoReprNewCustom => {
                arity(1)?;
                // The owner's allocation, named by the synthetic
                // address the packed word carries: register the
                // reverse lookup, then pack `address + tag` as
                // the representation word, in the same nesting
                // `io_error_value` builds below the `Error` level.
                let Some(id) = io_owner_alloc_id(&vals[0]).or_else(|| {
                    io_pointer_address(&vals[0])
                        .and_then(|addr| self.address_owner(addr))
                }) else {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "a custom error owner over an allocation",
                        vals[0].kind_name(),
                    ));
                };
                let owner_value = Value::Ref(RefTarget::Alloc {
                    id,
                    projection: Vec::new(),
                });
                let Some(address) = self
                    .arithmetic_price(&owner_value, None, Some(frames))
                    .or_else(|| synthetic_address(&owner_value))
                else {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "an allocation with a synthetic address",
                        vals[0].kind_name(),
                    ));
                };
                let address = address as u64;
                self.register_address_owner(address, id);
                Ok(io_repr_value(address | 0b01))
            }
            Builtin::TimeCheckedAdd(system) => {
                arity(2)?;
                let base = time_pair(&self.resolve_value(frames, &vals[0])?).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[0].kind_name())
                })?;
                let dur = duration_pair(&vals[1]).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a Duration value", vals[1].kind_name())
                })?;
                Ok(match checked_time_add(base, dur, system) {
                    Some(pair) => some_value(instant_value(pair)),
                    None => none_value(),
                })
            }
            Builtin::TimeCheckedSub(system) => {
                arity(2)?;
                let base = time_pair(&self.resolve_value(frames, &vals[0])?).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[0].kind_name())
                })?;
                let dur = duration_pair(&vals[1]).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a Duration value", vals[1].kind_name())
                })?;
                Ok(match checked_time_sub(base, dur, system) {
                    Some(pair) => some_value(instant_value(pair)),
                    None => none_value(),
                })
            }
            Builtin::InstantDurationSince => {
                arity(2)?;
                let a = time_pair(&self.resolve_value(frames, &vals[0])?).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[0].kind_name())
                })?;
                let b = time_pair(&vals[1]).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[1].kind_name())
                })?;
                match checked_time_sub(a, b, false) {
                    Some(pair) => self.duration_value(pair.0 as u64, pair.1 as u64),
                    None => Err(InterpError::panic_str_payload(
                        "supplied instant is later than self",
                    )),
                }
            }
            Builtin::InstantCheckedDurationSince => {
                arity(2)?;
                let a = time_pair(&self.resolve_value(frames, &vals[0])?).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[0].kind_name())
                })?;
                let b = time_pair(&vals[1]).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[1].kind_name())
                })?;
                match checked_time_sub(a, b, false) {
                    Some(pair) => Ok(some_value(
                        self.duration_value(pair.0 as u64, pair.1 as u64)?,
                    )),
                    None => Ok(none_value()),
                }
            }
            Builtin::InstantSaturatingDurationSince => {
                arity(2)?;
                let a = time_pair(&self.resolve_value(frames, &vals[0])?).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[0].kind_name())
                })?;
                let b = time_pair(&vals[1]).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[1].kind_name())
                })?;
                match checked_time_sub(a, b, false) {
                    Some(pair) => self.duration_value(pair.0 as u64, pair.1 as u64),
                    None => self.duration_value(0, 0),
                }
            }
            Builtin::InstantElapsed => {
                arity(1)?;
                let a = time_pair(&self.resolve_value(frames, &vals[0])?).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[0].kind_name())
                })?;
                match checked_time_sub(monotonic_now(), a, false) {
                    Some(pair) => self.duration_value(pair.0 as u64, pair.1 as u64),
                    None => Err(InterpError::panic_str_payload(
                        "supplied instant is later than self",
                    )),
                }
            }
            Builtin::SystemTimeDurationSince => {
                arity(2)?;
                let a = time_pair(&self.resolve_value(frames, &vals[0])?).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[0].kind_name())
                })?;
                let b = time_pair(&vals[1]).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[1].kind_name())
                })?;
                match checked_time_sub(a, b, true) {
                    Some(pair) => Ok(Value::Enum {
                        variant: 0,
                        fields: vec![self.duration_value(pair.0 as u64, pair.1 as u64)?],
                    }),
                    None => {
                        // `Err(SystemTimeError(duration))`: the
                        // error wraps the positive difference.
                        let back = checked_time_sub(b, a, true)
                            .expect("one direction is non-negative");
                        Ok(Value::Enum {
                            variant: 1,
                            fields: vec![Value::Enum {
                                variant: 0,
                                fields: vec![
                                    self.duration_value(back.0 as u64, back.1 as u64)?,
                                ],
                            }],
                        })
                    }
                }
            }
            Builtin::SystemTimeElapsed => {
                arity(1)?;
                let a = time_pair(&self.resolve_value(frames, &vals[0])?).ok_or_else(|| {
                    InterpError::type_mismatch(name, "a time value", vals[0].kind_name())
                })?;
                match checked_time_sub(wall_now(), a, true) {
                    Some(pair) => Ok(Value::Enum {
                        variant: 0,
                        fields: vec![self.duration_value(pair.0 as u64, pair.1 as u64)?],
                    }),
                    None => {
                        let back = checked_time_sub(a, wall_now(), true)
                            .expect("one direction is non-negative");
                        Ok(Value::Enum {
                            variant: 1,
                            fields: vec![Value::Enum {
                                variant: 0,
                                fields: vec![
                                    self.duration_value(back.0 as u64, back.1 as u64)?,
                                ],
                            }],
                        })
                    }
                }
            }
            Builtin::MpscChannel => {
                arity(0)?;
                let id = self.new_channel(None);
                Ok(Value::Tuple(vec![id_handle(id), id_handle(id)]))
            }
            Builtin::MpscSyncChannel => {
                arity(1)?;
                let bound = match &vals[0] {
                    Value::Uint(n) => *n as usize,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "sync_channel",
                            "a bound",
                            other.kind_name(),
                        ))
                    }
                };
                let id = self.new_channel(Some(bound));
                Ok(Value::Tuple(vec![id_handle(id), id_handle(id)]))
            }
            Builtin::MpscSend => {
                arity(2)?;
                let id = self.handle_id_of_ref(frames, &vals[0], name)?;
                let value = vals.into_iter().nth(1).expect("arity checked");
                self.mpsc_send(frames, id, value, name)
            }
            Builtin::MpscTrySend => {
                arity(2)?;
                let id = self.handle_id_of_ref(frames, &vals[0], name)?;
                let value = vals.into_iter().nth(1).expect("arity checked");
                match self.with_channel(id, |st| {
                    if st.receivers == 0 {
                        Err(1) // TrySendError::Disconnected(value)
                    } else if st.bound.is_some_and(|b| st.queue.len() >= b) {
                        Err(0) // TrySendError::Full(value)
                    } else {
                        st.queue.push_back(value.clone());
                        Ok(())
                    }
                }) {
                    Some(Ok(())) => Ok(ok_result(Value::Unit)),
                    Some(Err(variant)) => Ok(err_result(Value::Enum {
                        variant,
                        fields: vec![value],
                    })),
                    None => Err(unsupported(format!("{name} on an unknown channel"))),
                }
            }
            Builtin::MpscRecv => {
                arity(1)?;
                let id = self.handle_id_of_ref(frames, &vals[0], name)?;
                match self.mpsc_recv(frames, id, name)? {
                    Ok(value) => Ok(ok_result(value)),
                    // `RecvError` is a fieldless one-variant struct.
                    Err(()) => Ok(err_result(Value::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    })),
                }
            }
            Builtin::MpscTryRecv => {
                arity(1)?;
                let id = self.handle_id_of_ref(frames, &vals[0], name)?;
                // A native `try_recv` never blocks, but under
                // the cooperative scheduler a pending sender
                // only runs when the running thread reaches a
                // scheduling point; a caller spinning on
                // `try_recv` would starve it forever. An empty
                // result therefore first gives the other threads
                // one scheduling round — each runnable thread
                // gets one activation, the cooperative unit of
                // "meanwhile" — and re-checks the queue.
                let mut scheduled = false;
                loop {
                    match self.with_channel(id, |st| match st.queue.pop_front() {
                        Some(v) => Ok(v),
                        None => Err(if st.senders == 0 { 1 } else { 0 }),
                    }) {
                        Some(Ok(value)) => return Ok(ok_result(value)),
                        Some(Err(0)) if !scheduled => {
                            scheduled = true;
                            self.schedule_round(frames)?;
                        }
                        // `TryRecvError`: Empty = 0,
                        // Disconnected = 1, both fieldless.
                        Some(Err(variant)) => {
                            return Ok(err_result(Value::Enum {
                                variant,
                                fields: Vec::new(),
                            }))
                        }
                        None => {
                            return Err(unsupported(format!(
                                "{name} on an unknown channel"
                            )))
                        }
                    }
                }
            }
            Builtin::MpscRecvTimeout => {
                arity(2)?;
                let id = self.handle_id_of_ref(frames, &vals[0], name)?;
                // The model cannot wait out a duration: a value
                // that no pending thread produces is a timeout,
                // exactly what a native run whose producers have
                // all finished would observe.
                match self.mpsc_recv_or(frames, id, name)? {
                    Ok(value) => Ok(ok_result(value)),
                    Err(variant) => Ok(err_result(Value::Enum {
                        variant,
                        fields: Vec::new(),
                    })),
                }
            }
            Builtin::MpscIter => {
                arity(1)?;
                let id = self.handle_id_of_ref(frames, &vals[0], name)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![id_handle(id)],
                })
            }
            Builtin::IntoIter => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                let value = vals.into_iter().next().expect("arity checked");
                self.into_iter_value(frames, self_ty, value)
            }
            Builtin::IteratorNext => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                self.iterator_next(frames, self_ty, &vals[0])
            }
            Builtin::IteratorNextBack => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                self.iterator_next_back(frames, self_ty, &vals[0])
            }
            Builtin::IteratorSizeHint => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                self.iterator_size_hint(frames, self_ty, &vals[0])
            }
            Builtin::IteratorLen => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                self.iterator_len(frames, self_ty, &vals[0])
            }
            Builtin::SliceIterAsSlice => {
                arity(1)?;
                let elem_tag = generic_args.first().map(String::as_str).unwrap_or("");
                self.slice_iter_as_slice(frames, elem_tag, &vals[0])
            }
            Builtin::IteratorFold | Builtin::IteratorForEach | Builtin::IteratorTryFold => {
                unreachable!("fold-like builtins are dispatched before argument evaluation")
            }
            Builtin::IteratorCount => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                let mut state = vals.into_iter().next().expect("arity checked");
                let mut count: u128 = 0;
                loop {
                    self.tick_step()?;
                    let (next_state, item) =
                        self.iterator_next_state(frames, self_ty, state)?;
                    state = next_state;
                    match option_payload(item)? {
                        None => break,
                        Some(item) => {
                            // `count` consumes the items it
                            // pulls: each one drops as the
                            // loop's temporary would in core's
                            // own body.
                            self.drop_discarded_item(frames, self_ty, item)?;
                            count += 1;
                        }
                    }
                }
                self.note_consumed_iter_final(state);
                Ok(Value::Uint(count))
            }
            Builtin::IteratorNth => {
                arity(2)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                let n = match &vals[1] {
                    Value::Uint(v) => *v,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "Iterator::nth count",
                            "unsigned integer",
                            other.kind_name(),
                        ))
                    }
                };
                // `nth` is `advance_by(n)` followed by `next`.
                // An arithmetic range advances in one step, as
                // its specialised body in core does (see
                // `range_skip_at`); every other iterator drives
                // the modelled `next` transition `n` times, then
                // once more for the result. A skipped item is
                // consumed by the skip: its drop glue (baked by
                // conversion, see `skipped_glue` in `ir`) runs
                // before the next step, exactly as the core
                // default body drops the temporary each
                // `advance_by` iteration yields. An iterator
                // that runs out during the skips is left
                // exhausted and yields `None`, exactly as the
                // core default body leaves it.
                let target = self.deref_value(frames, &vals[0])?;
                if n > 0 {
                    if let Some(exhausted) =
                        self.range_skip_at(frames, self_ty, &target, n)?
                    {
                        if exhausted {
                            return Ok(none_value());
                        }
                        return self.iterator_next_at(frames, self_ty, &target);
                    }
                }
                for _ in 0..n {
                    self.tick_step()?;
                    let item = self.iterator_next_at(frames, self_ty, &target)?;
                    let Some(skipped) = option_payload(item)? else {
                        return Ok(none_value());
                    };
                    if let Some(glue) = skipped_glue {
                        self.drop_value_sync(frames, skipped, glue)?;
                    }
                }
                self.iterator_next_at(frames, self_ty, &target)
            }
            Builtin::IteratorCollect => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                let target = generic_args.get(1).map(String::as_str).unwrap_or("");
                let state = vals.into_iter().next().expect("arity checked");
                self.exec_collect(frames, self_ty, target, state)
            }
            Builtin::FromIteratorFromIter => {
                arity(1)?;
                let target = generic_args.first().map(String::as_str).unwrap_or("");
                let self_ty = generic_args.get(2).map(String::as_str).unwrap_or("");
                let state = vals.into_iter().next().expect("arity checked");
                self.exec_collect(frames, self_ty, target, state)
            }
            Builtin::PrimitiveBinOp(op) => {
                arity(2)?;
                let mut vals = vals.into_iter();
                let lhs = vals.next().expect("arity checked");
                let rhs = vals.next().expect("arity checked");
                self.exec_primitive_binop(frames, name, op, &lhs, &rhs)
            }
            Builtin::PrimitiveUnOp(op) => {
                arity(1)?;
                let value = vals.into_iter().next().expect("arity checked");
                self.exec_primitive_unop(frames, name, op, &value)
            }
            Builtin::EnvCurrentExe => {
                arity(0)?;
                Ok(match std::env::current_exe() {
                    Ok(path) => {
                        io_result_ok(pathbuf_value(&host_path_bytes(&path)?))
                    }
                    Err(e) => io_result_err(host_io_error(&e)),
                })
            }
            Builtin::FsOpen => {
                arity(2)?;
                let mut vals = vals.into_iter();
                let options = vals.next().expect("arity checked");
                let path = vals.next().expect("arity checked");
                let bytes = self.path_bytes(frames, &path)?;
                if bytes.contains(&0) {
                    return Ok(io_result_err(io_simple_error_value(ERROR_KIND_INVALID_INPUT)));
                }
                let options = self.resolve_value(frames, &options)?;
                let Some((
                    read,
                    write,
                    append,
                    truncate,
                    create,
                    create_new,
                    custom_flags,
                    mode,
                )) = open_options_of(&options)
                else {
                    return Err(InterpError::type_mismatch(
                        "OpenOptions::_open options",
                        "an OpenOptions value",
                        options.kind_name(),
                    ));
                };
                let mut host_options = std::fs::OpenOptions::new();
                host_options
                    .read(read)
                    .write(write)
                    .append(append)
                    .truncate(truncate)
                    .create(create)
                    .create_new(create_new);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt;
                    host_options.custom_flags(custom_flags);
                    host_options.mode(mode);
                }
                #[cfg(windows)]
                {
                    use std::os::windows::fs::OpenOptionsExt;
                    host_options.custom_flags(custom_flags as u32);
                    // std's default creation mode asks for
                    // nothing Windows cannot express; any
                    // other Unix mode has no equivalent.
                    if mode != 0o666 {
                        return Err(unsupported(format!(
                            "OpenOptions mode {mode:#o} has no equivalent on this host platform"
                        )));
                    }
                }
                #[cfg(not(any(unix, windows)))]
                {
                    let _ = (custom_flags, mode);
                    return Err(unsupported(
                        "OpenOptions custom flags and mode are not modelled on this host platform",
                    ));
                }
                match host_options.open(host_path(&bytes)?) {
                    Ok(file) => {
                        let id = self.register_file(file);
                        Ok(io_result_ok(file_value(id)))
                    }
                    Err(error) => Ok(io_result_err(host_io_error(&error))),
                }
            }
            Builtin::FsDirCreate => {
                arity(2)?;
                let mut vals = vals.into_iter();
                let builder = vals.next().expect("arity checked");
                let path = vals.next().expect("arity checked");
                let bytes = self.path_bytes(frames, &path)?;
                if bytes.contains(&0) {
                    return Ok(io_result_err(io_simple_error_value(ERROR_KIND_INVALID_INPUT)));
                }
                // The builder's `recursive` flag (its second
                // field) selects between std's two operations:
                // with it, `_create` builds the whole ancestor
                // chain; without it, one directory. The chain is
                // built on the host like the rest of the fs layer:
                // the converted `create_dir_all` body is never
                // reached (this builtin serves the `_create`
                // call), and its `Ancestors` iteration does not
                // carry its state faithfully in this model.
                let recursive = match self.resolve_value(frames, &builder) {
                    Ok(Value::Enum { fields, .. }) if fields.len() >= 2 => {
                        matches!(fields[1], Value::Bool(true))
                    }
                    _ => false,
                };
                let host_dir = host_path(&bytes)?;
                Ok(io_result_from_host(if recursive {
                    std::fs::create_dir_all(&host_dir).map(|()| Value::Unit)
                } else {
                    std::fs::create_dir(&host_dir).map(|()| Value::Unit)
                }))
            }
            Builtin::EnvJoinPaths => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                let value = vals.into_iter().next().expect("arity checked");
                // The argument is an `IntoIterator`; the forms
                // whose conversion is not the identity are the
                // by-value and by-reference containers, converted
                // here to their iterator states (the same states
                // the `IntoIterator::into_iter` builtin builds).
                let (state, iter_ty): (Value, String) = if iterator_next_modelled(self_ty) {
                    (value, self_ty.to_string())
                } else if self_ty.starts_with("std::vec::Vec<")
                    || self_ty.starts_with("alloc::vec::Vec<")
                {
                    match value {
                        Value::Enum { fields, .. } => (
                            owned_iter_state(fields),
                            self_ty.replacen("Vec<", "IntoIter<", 1),
                        ),
                        v @ Value::SparseVec { .. } => (
                            owned_iter_state(value_elements(&v)?),
                            self_ty.replacen("Vec<", "IntoIter<", 1),
                        ),
                        other => {
                            return Err(InterpError::type_mismatch(
                                "join_paths argument",
                                "vector value",
                                other.kind_name(),
                            ))
                        }
                    }
                } else if self_ty.starts_with("&std::vec::Vec<")
                    || self_ty.starts_with("&alloc::vec::Vec<")
                    || self_ty.starts_with("&[")
                {
                    let (elems, _) = self.slice_elements(frames, &value)?;
                    (slice_iter_value(elems), "std::slice::Iter<>".to_string())
                } else {
                    return Err(unsupported(format!(
                        "join_paths over {self_ty} is not runtime-supported"
                    )));
                };
                let mut state = state;
                let mut joined: Vec<u8> = Vec::new();
                let mut first = true;
                loop {
                    self.tick_step()?;
                    let (next_state, item) =
                        self.iterator_next_state(frames, &iter_ty, state)?;
                    state = next_state;
                    let Some(item) = option_payload(item)? else {
                        break;
                    };
                    let bytes = self.os_str_bytes(frames, &item)?;
                    if bytes.contains(&b':') {
                        // `JoinPathsError` is a zero-sized
                        // struct: the empty tuple shape.
                        return Ok(err_result(Value::Tuple(Vec::new())));
                    }
                    if !first {
                        joined.push(b':');
                    }
                    first = false;
                    joined.extend_from_slice(&bytes);
                }
                Ok(ok_result(os_string_value(&joined)))
            }
            Builtin::EnvSplitPaths => {
                arity(1)?;
                let bytes = self.os_str_bytes(frames, &vals[0])?;
                let parts: Vec<Value> = bytes
                    .split(|b| *b == b':')
                    .map(pathbuf_value)
                    .collect();
                Ok(owned_iter_state(parts))
            }
            Builtin::EnvVarInner => {
                arity(1)?;
                let key = self.os_str_bytes(frames, &vals[0])?;
                let value = if key.contains(&0) {
                    None
                } else {
                    crate::platform::os_str_from_bytes(&key)
                        .and_then(|name| std::env::var_os(name))
                };
                Ok(match value {
                    None => err_result(Value::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    }),
                    Some(v) => {
                        let bytes = crate::platform::os_str_bytes(&v)
                            .ok_or_else(|| {
                                unsupported(
                                    "env::var: the host value has no byte form on this platform",
                                )
                            })?;
                        match std::str::from_utf8(&bytes) {
                            Ok(s) => ok_result(Value::Str(s.to_string())),
                            Err(_) => err_result(Value::Enum {
                                variant: 1,
                                fields: vec![os_string_value(&bytes)],
                            }),
                        }
                    }
                })
            }
            Builtin::EnvGetenv => {
                arity(1)?;
                let key = self.os_str_bytes(frames, &vals[0])?;
                if key.contains(&0) {
                    return Ok(none_value());
                }
                let value = crate::platform::os_str_from_bytes(&key)
                    .and_then(|name| std::env::var_os(name));
                Ok(match value {
                    Some(v) => {
                        let bytes = crate::platform::os_str_bytes(&v)
                            .ok_or_else(|| {
                                unsupported(
                                    "getenv: the host value has no byte form on this platform",
                                )
                            })?;
                        some_value(os_string_value(&bytes))
                    }
                    None => none_value(),
                })
            }
            Builtin::EnvSetenv => {
                arity(2)?;
                let key = self.os_str_bytes(frames, &vals[0])?;
                let value = self.os_str_bytes(frames, &vals[1])?;
                if key.is_empty()
                    || key.contains(&b'=')
                    || key.contains(&0)
                    || value.contains(&0)
                {
                    return Ok(io_result_err(io_simple_error_value(20)));
                }
                let (Some(key_os), Some(value_os)) = (
                    crate::platform::os_str_from_bytes(&key),
                    crate::platform::os_str_from_bytes(&value),
                ) else {
                    return Err(unsupported(
                        "set_var: the key or value has no host string form on this platform",
                    ));
                };
                std::env::set_var(key_os, value_os);
                Ok(io_result_ok(Value::Unit))
            }
            Builtin::EnvUnsetenv => {
                arity(1)?;
                let key = self.os_str_bytes(frames, &vals[0])?;
                if key.is_empty() || key.contains(&b'=') || key.contains(&0) {
                    return Ok(io_result_err(io_simple_error_value(20)));
                }
                let Some(key_os) =
                    crate::platform::os_str_from_bytes(&key)
                else {
                    return Err(unsupported(
                        "remove_var: the key has no host string form on this platform",
                    ));
                };
                std::env::remove_var(key_os);
                Ok(io_result_ok(Value::Unit))
            }
            Builtin::EnvCurrentDir => {
                arity(0)?;
                Ok(match std::env::current_dir() {
                    Ok(dir) => {
                        io_result_ok(pathbuf_value(&host_path_bytes(&dir)?))
                    }
                    Err(e) => io_result_err(host_io_error(&e)),
                })
            }
            Builtin::EnvTempDir => {
                arity(0)?;
                Ok(pathbuf_value(&host_path_bytes(&std::env::temp_dir())?))
            }
            Builtin::EnvHomeDir => {
                arity(0)?;
                Ok(match std::env::home_dir() {
                    Some(dir) => {
                        some_value(pathbuf_value(&host_path_bytes(&dir)?))
                    }
                    None => none_value(),
                })
            }
            Builtin::SysChdir => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                if bytes.contains(&0) {
                    return Ok(io_result_err(io_simple_error_value(20)));
                }
                Ok(io_result_from_host(
                    std::env::set_current_dir(host_path(&bytes)?)
                        .map(|()| Value::Unit),
                ))
            }
            Builtin::StringFromUtf8 => {
                arity(1)?;
                let resolved = self.resolve_value(frames, &vals[0])?;
                let byte_elements = |elements: &[Value]| {
                    elements
                        .iter()
                        .map(|f| match f {
                            Value::Uint(v) => Ok(*v as u8),
                            Value::Int(v) => Ok(*v as u8),
                            other => Err(InterpError::type_mismatch(
                                "String::from_utf8 argument",
                                "byte elements",
                                other.kind_name(),
                            )),
                        })
                        .collect::<InterpResult<Vec<u8>>>()
                };
                let bytes = match &resolved {
                    Value::Enum { fields, .. } => byte_elements(fields)?,
                    // The sparse form: the declared length is
                    // the vector's content; the stored tail past
                    // it is spare capacity, not bytes.
                    Value::SparseVec { elems, len, .. } => {
                        let len = usize::try_from(*len).unwrap_or(usize::MAX);
                        if len > elems.len() {
                            return Err(InterpError::type_mismatch(
                                "String::from_utf8 argument",
                                "byte vector value",
                                resolved.kind_name(),
                            ));
                        }
                        byte_elements(&elems[..len])?
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            "String::from_utf8 argument",
                            "byte vector value",
                            other.kind_name(),
                        ))
                    }
                };
                Ok(match String::from_utf8(bytes.clone()) {
                    Ok(text) => ok_result(Value::Str(text)),
                    Err(e) => {
                        let ue = e.utf8_error();
                        let error_len = match ue.error_len() {
                            Some(n) => Value::Enum {
                                variant: 1,
                                fields: vec![Value::Uint(n as u128)],
                            },
                            None => Value::Enum {
                                variant: 0,
                                fields: Vec::new(),
                            },
                        };
                        let utf8_error = Value::Tuple(vec![
                            Value::Uint(ue.valid_up_to() as u128),
                            error_len,
                        ]);
                        err_result(Value::Enum {
                            variant: 0,
                            fields: vec![byte_vec_value(&bytes), utf8_error],
                        })
                    }
                })
            }
            Builtin::FsReadToStringInner => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                if bytes.contains(&0) {
                    return Ok(io_result_err(io_simple_error_value(20)));
                }
                Ok(match std::fs::read(host_path(&bytes)?) {
                    Ok(data) => match String::from_utf8(data) {
                        Ok(text) => io_result_ok(Value::Str(text)),
                        Err(_) => io_result_err(io_simple_error_value(std::io::ErrorKind::InvalidData as u32)),
                    },
                    Err(e) => io_result_from_host(Err::<Value, _>(e)),
                })
            }
            Builtin::FsReadInner => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                if bytes.contains(&0) {
                    return Ok(io_result_err(io_simple_error_value(20)));
                }
                Ok(io_result_from_host(
                    std::fs::read(host_path(&bytes)?)
                        .map(|data| byte_vec_value(&data)),
                ))
            }
            Builtin::FsWriteInner => {
                arity(2)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                if bytes.contains(&0) {
                    return Ok(io_result_err(io_simple_error_value(20)));
                }
                let contents = self.arg_bytes(frames, &vals[1])?;
                Ok(io_result_from_host(
                    std::fs::write(host_path(&bytes)?, &contents)
                        .map(|()| Value::Unit),
                ))
            }
            Builtin::PathToPathBuf => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                Ok(pathbuf_value(&bytes))
            }
            Builtin::PathIsAbsolute => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                Ok(Value::Bool(bytes.first() == Some(&b'/')))
            }
            Builtin::PathIsFile | Builtin::PathIsDir => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                let is_file = matches!(kind, Builtin::PathIsFile);
                Ok(Value::Bool(
                    match std::fs::metadata(host_path(&bytes)?) {
                        Ok(md) => {
                            if is_file {
                                md.is_file()
                            } else {
                                md.is_dir()
                            }
                        }
                        Err(_) => false,
                    },
                ))
            }
            Builtin::PathJoin => {
                arity(2)?;
                let base = self.path_bytes(frames, &vals[0])?;
                let path = self.path_bytes(frames, &vals[1])?;
                Ok(pathbuf_value(&path_push_bytes(&base, &path)))
            }
            Builtin::PathParent => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                Ok(match path_parent_len(&bytes) {
                    Some(end) => some_value(sub_path_value(&bytes[..end])),
                    None => none_value(),
                })
            }
            Builtin::PathFileName => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                Ok(match path_file_name_range(&bytes) {
                    Some((start, end)) => {
                        some_value(sub_path_value(&bytes[start..end]))
                    }
                    None => none_value(),
                })
            }
            Builtin::PathFileStem => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                Ok(match path_file_name_range(&bytes) {
                    Some((start, end)) => {
                        let (before, after) =
                            rsplit_file_at_dot(&bytes[start..end]);
                        let stem = before
                            .or(after)
                            .expect("a file name always has a stem side");
                        some_value(sub_path_value(stem))
                    }
                    None => none_value(),
                })
            }
            Builtin::PathExtension => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                Ok(match path_file_name_range(&bytes) {
                    Some((start, end)) => {
                        let (before, after) =
                            rsplit_file_at_dot(&bytes[start..end]);
                        match before.and(after) {
                            Some(ext) => some_value(sub_path_value(ext)),
                            None => none_value(),
                        }
                    }
                    None => none_value(),
                })
            }
            Builtin::StringFromUtf8Lossy => {
                arity(1)?;
                let bytes = self.arg_bytes(frames, &vals[0])?;
                Ok(cow_str_value(&bytes))
            }
            Builtin::StrFromUtf8 => {
                arity(1)?;
                let bytes = self.arg_bytes(frames, &vals[0])?;
                Ok(match std::str::from_utf8(&bytes) {
                    Ok(s) => ok_result(Value::Str(s.to_string())),
                    Err(e) => err_result(utf8_error_value(&e)),
                })
            }
            Builtin::LibcStrlen => {
                arity(1)?;
                let bytes = self.read_c_str(frames, &vals[0])?;
                Ok(Value::Uint(bytes.len() as u128))
            }
            Builtin::CStrFromBytesWithNul => {
                arity(1)?;
                let bytes = self.arg_bytes(frames, &vals[0])?;
                Ok(match bytes.iter().position(|b| *b == 0) {
                    Some(position) if position + 1 == bytes.len() => {
                        ok_result(vals[0].clone())
                    }
                    Some(position) => err_result(Value::Enum {
                        variant: 0,
                        fields: vec![Value::Uint(position as u128)],
                    }),
                    None => err_result(Value::Enum {
                        variant: 1,
                        fields: Vec::new(),
                    }),
                })
            }
            Builtin::CStrFromBytesUntilNul => {
                arity(1)?;
                let bytes = self.arg_bytes(frames, &vals[0])?;
                Ok(match bytes.iter().position(|b| *b == 0) {
                    Some(position) => {
                        ok_result(self.fat_ref_prefix(frames, &vals[0], position + 1)?)
                    }
                    None => err_result(Value::Enum {
                        variant: 0,
                        fields: vec![Value::Unit],
                    }),
                })
            }
            Builtin::CStrToStr => {
                arity(1)?;
                let bytes = self.c_str_bytes(frames, &vals[0])?;
                Ok(match std::str::from_utf8(&bytes) {
                    Ok(s) => ok_result(Value::Str(s.to_string())),
                    Err(e) => err_result(utf8_error_value(&e)),
                })
            }
            Builtin::CStrToStringLossy => {
                arity(1)?;
                let bytes = self.c_str_bytes(frames, &vals[0])?;
                Ok(cow_str_value(&bytes))
            }
            Builtin::CStringSpecNewImpl => {
                arity(1)?;
                let bytes = self.arg_bytes(frames, &vals[0])?;
                Ok(match bytes.iter().position(|b| *b == 0) {
                    Some(i) => err_result(Value::Enum {
                        variant: 0,
                        fields: vec![
                            Value::Uint(i as u128),
                            byte_vec_value(&bytes),
                        ],
                    }),
                    None => ok_result(self.c_string_value(bytes)),
                })
            }
            Builtin::CStringFromVecUnchecked => {
                arity(1)?;
                let bytes = self.arg_bytes(frames, &vals[0])?;
                Ok(self.c_string_value(bytes))
            }
            Builtin::CStringIntoString => {
                arity(1)?;
                let receiver = Value::Ref(RefTarget::Value(
                    Box::new(vals[0].clone()),
                    Vec::new(),
                ));
                let cstr = self.c_string_deref(frames, &receiver)?;
                let bytes = self.c_str_bytes(frames, &cstr)?;
                Ok(match std::str::from_utf8(&bytes) {
                    Ok(s) => ok_result(Value::Str(s.to_string())),
                    Err(e) => err_result(Value::Enum {
                        variant: 0,
                        fields: vec![
                            self.c_string_value(bytes),
                            utf8_error_value(&e),
                        ],
                    }),
                })
            }
            Builtin::CStringIntoBoxedCStr => {
                arity(1)?;
                let receiver = Value::Ref(RefTarget::Value(
                    Box::new(vals[0].clone()),
                    Vec::new(),
                ));
                let fat = self.c_string_deref(frames, &receiver)?;
                self.boxed_c_str(frames, &fat)
            }
            Builtin::CStringFromRaw => {
                arity(1)?;
                // The real body recomputes the length with
                // the runtime `strlen` before reconstituting
                // the box; the read itself is the model's
                // check that the pointer denotes a C string.
                let _bytes = self.read_c_str(frames, &vals[0])?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Enum {
                        variant: 0,
                        fields: vec![vals[0].clone()],
                    }],
                })
            }
            Builtin::ToStringLossy => {
                arity(1)?;
                let bytes = if name == "std::ffi::OsString::to_string_lossy" {
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    self.os_string_bytes_any(frames, &pointee)?.ok_or_else(|| {
                        InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "an OS string value",
                            pointee.kind_name(),
                        )
                    })?
                } else {
                    self.path_bytes(frames, &vals[0])?
                };
                Ok(cow_str_value(&bytes))
            }
            Builtin::OsStrToAsciiLowercase => {
                arity(1)?;
                let bytes = self.path_bytes(frames, &vals[0])?;
                let lowered: Vec<u8> =
                    bytes.iter().map(|b| b.to_ascii_lowercase()).collect();
                Ok(os_string_value(&lowered))
            }
            Builtin::OsStrMakeAsciiCase(upper) => {
                arity(1)?;
                // `OsStr::make_ascii_lowercase/uppercase(&mut
                // self)`: the receiver is the `&mut OsStr` the
                // `DerefMut` model hands out -- the `OsString`
                // place itself -- so the case folding rewrites
                // the string the place holds, as
                // `str::make_ascii_*` rewrites a `String` place
                // through `modify_string`. The rewritten value
                // takes the element form; every reader of an
                // `OsString` place accepts either shape (see
                // `os_string_bytes_any`).
                let target = self.deref_value(frames, &vals[0])?;
                let stored = self.read_resolved(frames, &target)?;
                // A boxed `OsStr`'s place holds the byte
                // string itself (the `Bytes` denomination a
                // `Box<OsStr>` allocation stores); its fold
                // rewrites it in that same denomination,
                // where an `OsString` place's fold rewrites
                // the string value.
                let boxed_bytes = matches!(stored, Value::Bytes(_));
                let mut bytes = match &stored {
                    Value::Bytes(raw) => raw.clone(),
                    _ => self
                        .os_string_bytes_any(frames, &stored)?
                        .ok_or_else(|| {
                            InterpError::type_mismatch(
                                "OsStr::make_ascii_uppercase/lowercase",
                                "an OS string value",
                                stored.kind_name(),
                            )
                        })?,
                };
                if upper {
                    bytes.make_ascii_uppercase();
                } else {
                    bytes.make_ascii_lowercase();
                }
                let rewritten = if boxed_bytes {
                    Value::Bytes(bytes)
                } else {
                    os_string_value(&bytes)
                };
                self.write_resolved(frames, &target, rewritten)?;
                Ok(Value::Unit)
            }
            Builtin::DirEntryPath | Builtin::DirEntryFileName => {
                arity(1)?;
                let entry = self.resolve_value(frames, &vals[0])?;
                let (path, name) = dir_entry_parts(&entry).ok_or_else(|| {
                    InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "a directory entry value",
                        entry.kind_name(),
                    )
                })?;
                Ok(if matches!(kind, Builtin::DirEntryPath) {
                    path
                } else {
                    name
                })
            }
            Builtin::DirEntryMetadata => {
                arity(1)?;
                let entry = self.resolve_value(frames, &vals[0])?;
                let (path, _) = dir_entry_parts(&entry).ok_or_else(|| {
                    InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "a directory entry value",
                        entry.kind_name(),
                    )
                })?;
                let bytes = self.path_buf_bytes_any(frames, &path)?.ok_or_else(|| {
                    InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "a directory entry holding a path buffer",
                        path.kind_name(),
                    )
                })?;
                if bytes.contains(&0) {
                    return Ok(io_result_err(io_simple_error_value(20)));
                }
                Ok(match std::fs::metadata(host_path(&bytes)?) {
                    Ok(md) => io_result_ok(Value::Enum {
                        variant: 0,
                        fields: vec![metadata_value(&md)?],
                    }),
                    Err(e) => io_result_err(host_io_error(&e)),
                })
            }
            Builtin::PathStartsWith => {
                arity(2)?;
                let path = self.path_bytes(frames, &vals[0])?;
                let child = self.path_bytes(frames, &vals[1])?;
                Ok(Value::Bool(path_starts_with(&path, &child)))
            }
            Builtin::PathEndsWith => {
                arity(2)?;
                let path = self.path_bytes(frames, &vals[0])?;
                let child = self.path_bytes(frames, &vals[1])?;
                Ok(Value::Bool(path_ends_with(&path, &child)))
            }
            Builtin::FsSys(op) => self.exec_fs_sys(frames, op, vals),
            Builtin::NetSys(op) => self.exec_net_sys(frames, op, generic_args, vals),
            Builtin::UnicodeLookupSlow(property) => {
                arity(1)?;
                self.exec_unicode_lookup_slow(property, &vals)
            }
            Builtin::FromFrom | Builtin::IntoInto => {
                arity(1)?;
                // `From::from` renders the callee's generic
                // arguments as [target, source]; `Into::into`
                // renders [source, target] (its `Self` is the
                // source). The blanket `Into` is `From` by
                // definition, so resolve both to the same
                // (target, source) pair before dispatching on
                // them.
                let (target, source) = if matches!(kind, Builtin::IntoInto) {
                    (
                        generic_args.get(1).map(String::as_str).unwrap_or(""),
                        generic_args.first().map(String::as_str).unwrap_or(""),
                    )
                } else {
                    (
                        generic_args.first().map(String::as_str).unwrap_or(""),
                        generic_args.get(1).map(String::as_str).unwrap_or(""),
                    )
                };
                let mut value = vals.into_iter().next().expect("arity checked");
                // A counted handle built from a `Cow<'_, B>`
                // (`Rc<B>: From<Cow<'_, B>>` and the `Arc` twin,
                // for the string-like and slice pointees): the
                // conversion is the borrowed form of `B` for the
                // `Borrowed` variant and the owned form for the
                // `Owned` variant, so unwrap the Cow to its held
                // field and re-name the source after the variant
                // before the arms below dispatch on it. Any other
                // Cow source keeps its shape and falls through to
                // the unsupported error below.
                let cow_source;
                let source = if let Some(pointee) = [
                    "std::rc::Rc<",
                    "alloc::rc::Rc<",
                    "std::sync::Arc<",
                    "alloc::sync::Arc<",
                ]
                .iter()
                .find_map(|prefix| generic_first_arg(target, prefix))
                {
                    if (source.starts_with("std::borrow::Cow<")
                        || source.starts_with("alloc::borrow::Cow<"))
                        && last_generic_param(source).as_deref() == Some(pointee)
                    {
                        match self.resolve_value(frames, &value) {
                            Ok(Value::Enum { variant, mut fields })
                                if fields.len() == 1 =>
                            {
                                let field = fields.pop().expect("len checked");
                                cow_source = if variant == 0 {
                                    format!("&{pointee}")
                                } else if pointee == "str" {
                                    "std::string::String".to_string()
                                } else if pointee == "std::ffi::OsStr" {
                                    "std::ffi::OsString".to_string()
                                } else if pointee == "std::path::Path" {
                                    "std::path::PathBuf".to_string()
                                } else if pointee == "std::ffi::CStr" {
                                    "std::ffi::CString".to_string()
                                } else if is_slice_tag(pointee) {
                                    format!(
                                        "std::vec::Vec<{}>",
                                        &pointee[1..pointee.len() - 1]
                                    )
                                } else {
                                    String::new()
                                };
                                if cow_source.is_empty() {
                                    source
                                } else {
                                    value = field;
                                    &cow_source
                                }
                            }
                            _ => source,
                        }
                    } else {
                        source
                    }
                } else {
                    source
                };
                // The reflexive impl (`impl<T> From<T> for T`) is
                // the identity by definition; collected code names
                // it whenever a generic conversion is instantiated
                // at equal types.
                if target == source && !target.is_empty() {
                    Ok(value)
                } else if legacy_new_range_conversion(source, target) {
                    // core's `From` impls between the legacy
                    // `std::ops::` range types and the new
                    // `std::range::` ones (`Range`, `RangeFrom`,
                    // `RangeToInclusive`, in both directions)
                    // copy the fields in order, so the value
                    // already is the target's shape.
                    Ok(value)
                } else if matches!(value, Value::Str(_))
                    && (target == "std::string::String" || target == "str" || target == "&str")
                {
                    Ok(value)
                } else if target == "std::string::String"
                    && (generic_first_arg(source, "std::boxed::Box<") == Some("str")
                        || generic_first_arg(source, "alloc::boxed::Box<") == Some("str"))
                {
                    // `String: From<Box<str>>`: the string moves
                    // out of the box, releasing its allocation.
                    match self.unbox_take(&value) {
                        Some(s @ Value::Str(_)) => Ok(s),
                        _ => Err(InterpError::type_mismatch(
                            "From::from to String",
                            "boxed string value",
                            value.kind_name(),
                        )),
                    }
                } else if target == "std::ffi::OsString"
                    && (generic_first_arg(source, "std::boxed::Box<")
                        == Some("std::ffi::OsStr")
                        || generic_first_arg(source, "alloc::boxed::Box<")
                            == Some("std::ffi::OsStr"))
                {
                    // `OsString: From<Box<OsStr>>`: the byte
                    // string moves out of the box, releasing
                    // its allocation; the `OsString` value is
                    // the buffer form over those bytes.
                    match self.unbox_take(&value) {
                        Some(pointee) => {
                            let bytes = self.os_str_bytes(frames, &pointee)?;
                            Ok(os_string_value(&bytes))
                        }
                        _ => Err(InterpError::type_mismatch(
                            "From::from to OsString",
                            "boxed OS string value",
                            value.kind_name(),
                        )),
                    }
                } else if target == "std::string::String"
                    && (source == "&str" || source == "&mut str")
                {
                    // `String: From<&str>` / `From<&mut str>`
                    // copies the text into a fresh string; in
                    // this value model the string value is the
                    // text itself, resolved through the
                    // reference (the borrowed text is not
                    // consumed: the resolved `Str` is the copy).
                    match self.resolve_value(frames, &value)? {
                        s @ Value::Str(_) => Ok(s),
                        _ => {
                            // A `str` place over byte storage
                            // (the `encode_utf8` window over a
                            // byte array) does not resolve to
                            // the text value; its bytes are the
                            // text, read as a byte slice.
                            let bytes =
                                self.slice_bytes(frames, &value, "From::from to String")?;
                            match String::from_utf8(bytes) {
                                Ok(text) => Ok(Value::Str(text)),
                                Err(_) => Err(InterpError::type_mismatch(
                                    "From::from to String",
                                    "valid UTF-8 text",
                                    "invalid UTF-8 bytes",
                                )),
                            }
                        }
                    }
                } else if matches!(value, Value::Int(_) | Value::Uint(_)) && is_integer_tag(target)
                {
                    // The `From` impls between integer types are the
                    // value-preserving ones; the stored value already
                    // denotes the result (width/sign are applied when
                    // the destination place is written).
                    Ok(value)
                } else if target == "std::process::ExitCode"
                    && matches!(value, Value::Uint(_) | Value::Int(_))
                {
                    // `ExitCode::from(u8)`: the code byte inside
                    // the platform exit-code wrapper, two one-field
                    // aggregates in this model (the shape the
                    // driver's entry-return handling reads back).
                    let byte = match value {
                        Value::Uint(v) => v,
                        Value::Int(v) => v as u128,
                        _ => unreachable!(),
                    };
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![Value::Enum {
                            variant: 0,
                            fields: vec![Value::Uint(byte)],
                        }],
                    })
                } else if target == "run_make_support::CompletedProcess"
                    && source == "std::process::Output"
                {
                    // run-make-support's `CompletedProcess` is a
                    // one-field wrapper around the process
                    // `Output`; the conversion moves it in.
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![value],
                    })
                } else if target == "std::path::PathBuf"
                    && (source == "&str" || source == "str")
                {
                    // `PathBuf: From<&str>` copies the text into
                    // the path's buffer.
                    match self.resolve_value(frames, &value)? {
                        Value::Str(s) => Ok(pathbuf_value(s.as_bytes())),
                        other => Err(InterpError::type_mismatch(
                            "From::from to PathBuf",
                            "string value",
                            other.kind_name(),
                        )),
                    }
                } else if target == "std::ffi::OsString"
                    && (source == "&str" || source == "str")
                {
                    // `OsString: From<&str>` likewise.
                    match self.resolve_value(frames, &value)? {
                        Value::Str(s) => Ok(os_string_value(s.as_bytes())),
                        other => Err(InterpError::type_mismatch(
                            "From::from to OsString",
                            "string value",
                            other.kind_name(),
                        )),
                    }
                } else if target == "std::path::PathBuf" && source == "std::string::String" {
                    // `PathBuf: From<String>` moves the string's
                    // bytes into the path. The string arrives in
                    // whichever form its producer built -- the
                    // string value itself, or the real aggregate
                    // a converted body hands back (`slice::join`
                    // and `concat` build one through
                    // `join_generic_copy`, its buffer elements in
                    // the `MaybeUninit` union form) -- so the
                    // bytes are read through the string readers,
                    // as at every other string consumer.
                    let text = self.resolve_string(frames, &value)?;
                    Ok(pathbuf_value(text.as_bytes()))
                } else if target == "std::ffi::OsString" && source == "std::string::String" {
                    // `OsString: From<String>` likewise.
                    let text = self.resolve_string(frames, &value)?;
                    Ok(os_string_value(text.as_bytes()))
                } else if target == "std::ffi::OsString"
                    && (source == "&std::ffi::OsStr" || source == "std::ffi::OsStr")
                {
                    // `OsString: From<&OsStr>` clones the byte
                    // string into an owned buffer.
                    let bytes = self.os_str_bytes(frames, &value)?;
                    Ok(os_string_value(&bytes))
                } else if target == "std::ffi::OsString" && source == "std::path::PathBuf" {
                    // `OsString: From<PathBuf>` is std's
                    // `PathBuf::into_os_string`: the path's
                    // bytes move into the OS string. The path
                    // value arrives in any of the model's
                    // shapes for it (the buffer form
                    // `pathbuf_value` builds, the element form,
                    // the real aggregate form a converted body
                    // filled through raw-pointer writes), so
                    // read it through the composition the
                    // `PathBuf` target arm reads its sources
                    // with, with the path-specific accessor in
                    // place of the OS-string one.
                    let bytes = match string_aggregate_bytes(&value) {
                        Some(bytes) => bytes,
                        None => match self.path_buf_bytes_any(frames, &value)? {
                            Some(bytes) => bytes,
                            None => self.path_bytes_of(frames, &value)?.0,
                        },
                    };
                    Ok(os_string_value(&bytes))
                } else if target == "std::ffi::OsString"
                    && (source == "&std::path::Path" || source == "std::path::Path")
                {
                    // `OsString: From<&Path>` clones the
                    // path's byte string; a path's bytes are
                    // an OS string's bytes, read by the
                    // `&OsStr` reader.
                    let bytes = self.os_str_bytes(frames, &value)?;
                    Ok(os_string_value(&bytes))
                } else if target == "std::ffi::OsString"
                    && (source == "&std::string::String" || source == "&alloc::string::String")
                {
                    // `OsString: From<&String>` clones the
                    // text into an owned buffer.
                    let s = self.resolve_string(frames, &value)?;
                    Ok(os_string_value(s.as_bytes()))
                } else if target == "std::ffi::OsString"
                    && (source.starts_with("std::borrow::Cow<")
                        || source.starts_with("alloc::borrow::Cow<"))
                    && last_generic_param(source).as_deref() == Some("std::ffi::OsStr")
                {
                    // `OsString: From<Cow<'_, OsStr>>` is
                    // `into_owned`: the borrowed variant's
                    // field is the `&OsStr`, the owned
                    // variant's field is the `OsString`
                    // itself, and the `&OsStr` reader reads
                    // the bytes out of either shape.
                    let cow = self.resolve_value(frames, &value)?;
                    match cow {
                        Value::Enum { mut fields, .. } if fields.len() == 1 => {
                            let held = fields.pop().expect("len checked");
                            let bytes = self.os_str_bytes(frames, &held)?;
                            Ok(os_string_value(&bytes))
                        }
                        other => Err(InterpError::type_mismatch(
                            "From::from to OsString",
                            "cow aggregate",
                            other.kind_name(),
                        )),
                    }
                } else if target == "std::path::PathBuf"
                    && (source == "&std::path::Path" || source == "std::path::Path")
                {
                    // `PathBuf: From<&Path>` clones the path.
                    let bytes = self.path_bytes(frames, &value)?;
                    Ok(pathbuf_value(&bytes))
                } else if target.starts_with("std::vec::Vec<")
                    && generic_first_arg(target, "std::vec::Vec<") == Some("u8")
                    && source == "std::string::String"
                {
                    // `Vec<u8>: From<String>` is
                    // `String::into_bytes` by definition: the
                    // string's bytes as the vector's element
                    // form, read through the string readers like
                    // the `into_bytes` model itself (the string
                    // may arrive in the real aggregate form a
                    // converted body built, as in the `PathBuf`
                    // arm above).
                    let text = self.resolve_string(frames, &value)?;
                    Ok(Value::Enum {
                        variant: 0,
                        fields: text
                            .into_bytes()
                            .into_iter()
                            .map(|b| Value::Uint(b as u128))
                            .collect(),
                    })
                } else if target.starts_with("std::vec::Vec<") {
                    // `Vec::from([T; N])`: the array's elements become
                    // the vector's.
                    match value {
                        Value::Tuple(items) => Ok(Value::Enum {
                            variant: 0,
                            fields: items,
                        }),
                        ref other => Err(InterpError::type_mismatch(
                            "From::from to Vec",
                            "array value",
                            other.kind_name(),
                        )),
                    }
                } else if target.starts_with("std::boxed::Box<")
                    && generic_first_arg(target, "std::boxed::Box<") == Some("str")
                    && (source.starts_with("std::borrow::Cow<")
                        || source.starts_with("alloc::borrow::Cow<"))
                {
                    // `From<Cow<'_, str>> for Box<str>`: the box
                    // holds the string data itself -- the owned
                    // String's text handed over, or a clone of
                    // the borrowed text -- never the Cow
                    // aggregate (boxing the aggregate would
                    // leave every str consumer of the box
                    // reading the Cow's own fields: Display and
                    // Debug see an ADT instead of the text).
                    // The result is the shape
                    // `String::into_boxed_str` builds. The Cow
                    // is consumed by the conversion, so the
                    // text moves into the box exactly once.
                    match value {
                        Value::Enum {
                            mut fields, ..
                        } if fields.len() == 1 => {
                            let held = fields.pop().expect("len checked");
                            // The borrowed variant's field is a
                            // reference to the source text, the
                            // owned variant's field is the String
                            // itself; in this value model both
                            // resolve to the text the box takes
                            // (a copy of it, so later writes to
                            // the source do not show through).
                            match self.resolve_value(frames, &held)? {
                                Value::Str(text) => {
                                    let boxed = self.new_box(Value::Str(text));
                                    self.record_box_ty(&boxed, Ty::Str);
                                    Ok(boxed)
                                }
                                other => Err(InterpError::type_mismatch(
                                    "From::from to Box<str>",
                                    "string data",
                                    other.kind_name(),
                                )),
                            }
                        }
                        other => Err(InterpError::type_mismatch(
                            "From::from to Box<str>",
                            "cow aggregate",
                            other.kind_name(),
                        )),
                    }
                } else if target.starts_with("std::boxed::Box<")
                    && matches!(
                        generic_first_arg(target, "std::boxed::Box<"),
                        Some(pt) if is_slice_tag(pt)
                    )
                    && (source.starts_with("std::vec::Vec<")
                        || source.starts_with("alloc::vec::Vec<")
                        || source.starts_with("std::borrow::Cow<")
                        || source.starts_with("alloc::borrow::Cow<")
                        || source.starts_with("&["))
                {
                    // `From<Vec<T>> for Box<[T]>` is
                    // `Vec::into_boxed_slice` by definition, and
                    // `From<Cow<'_, [T]>> for Box<[T]>` hands over
                    // the Cow's owned vector, or a clone of the
                    // borrowed slice's elements; `From<&[T]>`
                    // likewise clones the source slice's
                    // elements: the pointee is the element
                    // sequence, not the source value itself
                    // (boxing the aggregate -- or the reference --
                    // would leave every slice consumer of the box
                    // reading the source's own fields).
                    let pointee_ty =
                        generic_first_arg(target, "std::boxed::Box<").and_then(ty_from_tag);
                    // The element type tag, for the
                    // reference-counted judgment in the element
                    // clones below.
                    let elem_ty = generic_first_arg(target, "std::boxed::Box<")
                        .and_then(|pt| pt.strip_prefix('['))
                        .and_then(|pt| pt.strip_suffix(']'));
                    if source.starts_with("&[") {
                        // The argument is the slice reference
                        // itself. Snapshot its elements and
                        // re-own the ones that own allocations
                        // (the `Box::from` clone semantics, as
                        // in the borrowed-Cow arm below) before
                        // boxing the snapshot: boxing the
                        // reference would alias the source, and
                        // writes or drops on either side would
                        // show through the other.
                        let (elements, len) = self.slice_elements(frames, &value)?;
                        let elements = self.clone_source_elements(
                            frames,
                            &value,
                            elements,
                            len,
                            elem_ty,
                        );
                        let elements = self.clone_boxed_slice_elements(
                            &mut *frames,
                            &elements,
                            elem_ty,
                            elem_glue,
                        )?;
                        self.boxed_slice_value(
                            Value::Tuple(elements),
                            pointee_ty,
                            "From::from to Box<[T]>",
                        )
                    } else if source.starts_with("std::borrow::Cow<")
                        || source.starts_with("alloc::borrow::Cow<")
                    {
                        match value {
                            Value::Enum {
                                variant,
                                mut fields,
                                ..
                            } if fields.len() == 1 => {
                                let held = fields.pop().expect("len checked");
                                if variant == 0 {
                                    // Borrowed: the held value is
                                    // a reference to the source
                                    // slice. `Box::from` clones the
                                    // elements into the new
                                    // allocation (the
                                    // `From<&[T]> for Box<[T]>`
                                    // semantics), so resolve the
                                    // reference and box the element
                                    // snapshot: boxing the reference
                                    // itself would alias the
                                    // source, and later writes to
                                    // it would show through the box.
                                    let (elements, len) =
                                        self.slice_elements(frames, &held)?;
                                    // The snapshot copies the
                                    // elements' `Value`
                                    // representations: an element
                                    // that owns an allocation (a
                                    // `Box`, say) would keep
                                    // sharing it with the source,
                                    // and a write through the
                                    // source's element would show
                                    // through the box. `Box::from`
                                    // clones every element, so
                                    // re-own them the way the
                                    // clone models do (the
                                    // `slice::to_vec` treatment)
                                    // before boxing them.
                                    let elements = self.clone_source_elements(
                                        frames,
                                        &held,
                                        elements,
                                        len,
                                        elem_ty,
                                    );
                                    let elements = self.clone_boxed_slice_elements(
                                        &mut *frames,
                                        &elements,
                                        elem_ty,
                                        elem_glue,
                                    )?;
                                    self.boxed_slice_value(
                                        Value::Tuple(elements),
                                        pointee_ty,
                                        "From::from to Box<[T]>",
                                    )
                                } else {
                                    // Owned: the held value is
                                    // the vector.
                                    self.boxed_slice_value(
                                        held,
                                        pointee_ty,
                                        "From::from to Box<[T]>",
                                    )
                                }
                            }
                            other => Err(InterpError::type_mismatch(
                                "From::from to Box<[T]>",
                                "cow aggregate",
                                other.kind_name(),
                            )),
                        }
                    } else {
                        self.boxed_slice_value(value, pointee_ty, "From::from to Box<[T]>")
                    }
                } else if target == "std::ffi::CString"
                    && (generic_first_arg(source, "std::boxed::Box<")
                        == Some("std::ffi::CStr")
                        || generic_first_arg(
                            source,
                            "alloc::boxed::Box<",
                        ) == Some("std::ffi::CStr"))
                {
                    // `CString: From<Box<CStr>>` is
                    // `into_c_string`: the owned C string over
                    // the box's bytes, the terminator dropped
                    // here and rebuilt by `c_string_value`.
                    let pointee = self
                        .box_pointee(frames, &value)
                        .ok_or_else(|| {
                            InterpError::type_mismatch(
                                "From::from to CString",
                                "a boxed C string",
                                value.kind_name(),
                            )
                        })?;
                    let mut bytes =
                        self.arg_bytes(frames, &pointee)?;
                    if bytes.last() == Some(&0) {
                        bytes.pop();
                    }
                    Ok(self.c_string_value(bytes))
                } else if target == "std::ffi::CString"
                    && (source == "&std::ffi::CStr"
                        || source == "&core::ffi::CStr"
                        || source == "std::ffi::CStr"
                        || source == "core::ffi::CStr")
                {
                    // `CString: From<&CStr>` is the owned C
                    // string over the borrowed one's bytes:
                    // the terminator is dropped by the reader
                    // and rebuilt by `c_string_value`, as in
                    // the `Box<CStr>` arm above.
                    let bytes = self.c_str_bytes(frames, &value)?;
                    Ok(self.c_string_value(bytes))
                } else if (target.starts_with("std::boxed::Box<")
                    || target.starts_with("alloc::boxed::Box<"))
                    && (generic_first_arg(target, "std::boxed::Box<")
                        == Some("std::ffi::CStr")
                        || generic_first_arg(target, "alloc::boxed::Box<")
                            == Some("std::ffi::CStr"))
                    && source == "&std::ffi::CStr"
                {
                    // `From<&CStr> for Box<CStr>`: the box
                    // holds a copy of the borrowed string's
                    // bytes, terminator included (see
                    // `boxed_c_str`). Boxing the fat reference
                    // itself (the catch-all `Box` arm's
                    // verbatim move) leaves the allocation
                    // holding a reference value, which the
                    // projection engine cannot index through:
                    // every element read of the boxed string
                    // (`to_bytes()[i]`, subslicing, iteration)
                    // stopped on it.
                    self.boxed_c_str(frames, &value)
                } else if (target.starts_with("std::boxed::Box<")
                    || target.starts_with("alloc::boxed::Box<"))
                    && (generic_first_arg(target, "std::boxed::Box<")
                        == Some("std::ffi::CStr")
                        || generic_first_arg(target, "alloc::boxed::Box<")
                            == Some("std::ffi::CStr"))
                    && (source == "std::ffi::CString"
                        || source.starts_with("std::borrow::Cow<")
                        || source.starts_with("alloc::borrow::Cow<"))
                {
                    // `From<CString> for Box<CStr>` (and the
                    // owned Cow source) is `into_boxed_c_str`:
                    // the box holds the C string's bytes
                    // themselves (see `boxed_c_str`) -- not
                    // the `CString` aggregate.
                    // Boxing the aggregate (the catch-all `Box`
                    // arm's verbatim move) would leave the
                    // length-trusting consumers (`to_bytes`,
                    // `to_bytes_with_nul`) reading the
                    // aggregate's field count as the string's
                    // length, while the byte-scanning consumers
                    // (Debug, equality, ordering) read through
                    // it and hide the defect.
                    let owned = if source == "std::ffi::CString" {
                        value
                    } else {
                        // A Cow source: the borrowed variant's
                        // field is the `&CStr` itself, which
                        // already denotes the fat slice the box
                        // holds; the owned variant's field is
                        // the `CString` the arm below converts.
                        match self.resolve_value(frames, &value)? {
                            Value::Enum {
                                variant: 0,
                                mut fields,
                                ..
                            } if fields.len() == 1 => {
                                let held =
                                    fields.pop().expect("len checked");
                                let fat = self
                                    .resolve_value(frames, &held)?;
                                return self.boxed_c_str(frames, &fat);
                            }
                            Value::Enum {
                                variant: 1,
                                mut fields,
                                ..
                            } if fields.len() == 1 => {
                                fields.pop().expect("len checked")
                            }
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "From::from to Box<CStr>",
                                    "cow aggregate",
                                    other.kind_name(),
                                ))
                            }
                        }
                    };
                    let receiver = Value::Ref(RefTarget::Value(
                        Box::new(owned),
                        Vec::new(),
                    ));
                    let fat =
                        self.c_string_deref(frames, &receiver)?;
                    self.boxed_c_str(frames, &fat)
                } else if (target.starts_with("std::boxed::Box<")
                    || target.starts_with("alloc::boxed::Box<"))
                    && (generic_first_arg(target, "std::boxed::Box<")
                        == Some("std::path::Path")
                        || generic_first_arg(target, "alloc::boxed::Box<")
                            == Some("std::path::Path"))
                    && (source == "std::path::PathBuf"
                        || source.starts_with("std::borrow::Cow<")
                        || source.starts_with("alloc::borrow::Cow<"))
                {
                    // `From<PathBuf> for Box<Path>` is
                    // `into_boxed_path`, and `From<Cow<'_,
                    // Path>>` hands over a clone of the
                    // borrowed path or the owned buffer's
                    // box: in every case the box holds the
                    // path's bytes themselves, the pointee
                    // form every `Box<Path>` consumer reads
                    // (see the `PathBufIntoBoxedPath`
                    // builtin), with the `Path` type recorded
                    // on the allocation. Boxing the `PathBuf`
                    // aggregate (the catch-all `Box` arm's
                    // verbatim move) leaves the allocation
                    // holding an ADT value the path
                    // projection refuses ("expected path
                    // byte string, found ADT value"), so the
                    // `Deref` consumers of the box
                    // (`file_name`, `display`, the component
                    // machinery) all stopped, while the owned
                    // value's own readers accept the
                    // aggregate and hide the defect until
                    // the box is formed.
                    let owned = if source == "std::path::PathBuf" {
                        value
                    } else {
                        // A Cow source: the borrowed
                        // variant's field is the `&Path`
                        // itself, which already denotes the
                        // path's bytes; the owned variant's
                        // field is the `PathBuf` the byte
                        // reader below reads. Both are the
                        // single field of their variant.
                        match self.resolve_value(frames, &value)? {
                            Value::Enum { mut fields, .. }
                                if fields.len() == 1 =>
                            {
                                fields.pop().expect("len checked")
                            }
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "From::from to Box<Path>",
                                    "cow aggregate",
                                    other.kind_name(),
                                ))
                            }
                        }
                    };
                    let bytes = self.os_str_bytes(frames, &owned)?;
                    let boxed = self.new_box(Value::Bytes(bytes));
                    if let Some(ty) = ty_from_tag("std::path::Path") {
                        self.record_box_ty(&boxed, ty);
                    }
                    Ok(boxed)
                } else if target.starts_with("std::boxed::Box<") {
                    // `Box::from(value)` is `Box::new` by definition.
                    let boxed = self.new_box(value);
                    if let Some(ty) =
                        generic_first_arg(target, "std::boxed::Box<")
                            .and_then(ty_from_tag)
                    {
                        self.record_box_ty(&boxed, ty);
                    }
                    Ok(boxed)
                } else if (target.starts_with("std::pin::Pin<")
                    || target.starts_with("core::pin::Pin<"))
                    && (generic_first_arg(target, "std::pin::Pin<") == Some(source)
                        || generic_first_arg(target, "core::pin::Pin<") == Some(source))
                {
                    // `Pin<P>: From<P>` (the `Box` impl and the
                    // reference impls): the pin is the one-field
                    // wrapper around the pointer value itself,
                    // the shape `Pin::new` builds and the
                    // `Deref` / `DerefMut` models unwrap.
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![value],
                    })
                } else if (target.starts_with("std::ptr::NonNull<")
                    || target.starts_with("core::ptr::NonNull<"))
                    && source.starts_with('&')
                {
                    // `NonNull<T>: From<&T>` / `From<&mut T>`:
                    // the one-field wrapper around the
                    // reference's own pointer value; the
                    // pointee is not read.
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![value],
                    })
                } else if target.starts_with("std::cell::Cell<") {
                    // `From<T> for Cell<T>` is `Cell::new` by
                    // definition (core::cell), so the result is the
                    // natural cell shape.
                    Ok(cell_value(value))
                } else if target.starts_with("std::cell::RefCell<") {
                    // `From<T> for RefCell<T>` is `RefCell::new` by
                    // definition (core::cell).
                    Ok(refcell_value(value))
                } else if (target == "std::task::Waker"
                    && (source.starts_with("std::sync::Arc<")
                        || source.starts_with("alloc::sync::Arc<")))
                    || (target == "std::task::LocalWaker"
                        && (source.starts_with("std::rc::Rc<")
                            || source.starts_with("alloc::rc::Rc<")))
                {
                    // `Waker::from(Arc<T: Wake>)` (and the
                    // `LocalWaker` impl, which takes `Rc<T: LocalWake>`):
                    // std builds a `RawWaker` whose data pointer is
                    // the smart pointer's allocation and whose vtable
                    // forwards wake/clone/drop to that allocation.
                    // In this value model the waker is the source
                    // value carried in a one-field aggregate: the
                    // data the vtable would act on *is* the Arc/Rc
                    // value, and the reference count the vtable
                    // would maintain is unobservable (as it is for
                    // the modelled `Arc` itself). Carrying the
                    // waker, handing it to `Context::from_waker`
                    // and cloning it (a value copy, exactly how the
                    // modelled `Arc` clones) are faithful. Calls
                    // that would go through the vtable dispatch
                    // on this shape in their own builtins (see
                    // `Builtin::WakerWake`).
                    // `Waker::wake` / `wake_by_ref` are modelled
                    // as their own builtins, which dispatch on
                    // this shape; they find the handle's `Wake` /
                    // `LocalWake` impl methods in the dispatch
                    // recorded here, keyed by the handle's
                    // allocation (the waker value itself carries
                    // only the handle, so the pointee type - and
                    // with it the impl - is noted at this, the
                    // one place it is known).
                    if let Some(id) = rc_handle_id(&value) {
                        let (trait_path, pointee) = if target == "std::task::Waker" {
                            (
                                "std::task::Wake",
                                smart_ptr_pointee(
                                    source,
                                    &["std::sync::Arc<", "alloc::sync::Arc<"],
                                ),
                            )
                        } else {
                            (
                                "std::task::LocalWake",
                                smart_ptr_pointee(
                                    source,
                                    &["std::rc::Rc<", "alloc::rc::Rc<"],
                                ),
                            )
                        };
                        if let Some(pointee) = pointee {
                            self.note_waker_dispatch(id, &pointee, trait_path);
                        }
                    }
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![value],
                    })
                } else if target == "std::io::Error" && source == "std::io::ErrorKind" {
                    // `Error::from(kind)` is std's `const_error!`:
                    // the bare kind in the bitpacked error
                    // representation (the same shape the fs layer
                    // and the process model produce).
                    let disc = match value {
                        Value::Enum { variant, .. } => variant,
                        Value::Uint(v) => v as u32,
                        Value::Int(v) => v as u32,
                        ref other => {
                            return Err(InterpError::type_mismatch(
                                "From::from to io::Error",
                                "an ErrorKind value",
                                other.kind_name(),
                            ));
                        }
                    };
                    Ok(io_simple_error_value(disc))
                } else if target.starts_with("std::collections::VecDeque<") {
                    // `VecDeque: From<Vec<T>> / From<[T; N]> /
                    // From<&[T]>`: the deque has no modelled shape
                    // of its own in this value model (its value is
                    // the real aggregate, built and consumed by
                    // the converted std bodies), so the conversion
                    // assembles that aggregate from the source's
                    // elements (see `vec_deque_from_value`).
                    Ok(self.vec_deque_from_value(frames, value)?)
                } else if (target.starts_with("std::collections::BinaryHeap<")
                    || target.starts_with("alloc::collections::BinaryHeap<"))
                    && (source.starts_with("std::vec::Vec<")
                        || source.starts_with("alloc::vec::Vec<"))
                {
                    // `BinaryHeap<T>: From<Vec<T>>`: std moves
                    // the vector's buffer into the heap's
                    // single `data` field and rebuilds the
                    // heap order; the model builds that real
                    // aggregate over a fresh element block
                    // holding the elements in heap order (see
                    // `binary_heap_from_vec`).
                    self.binary_heap_from_vec(frames, value)
                } else if [
                    "std::rc::Rc<",
                    "alloc::rc::Rc<",
                    "std::sync::Arc<",
                    "alloc::sync::Arc<",
                ]
                .iter()
                .any(|prefix| generic_first_arg(target, prefix) == Some(source))
                    && !source.is_empty()
                {
                    // `Rc<T>: From<T>` / `Arc<T>: From<T>` is
                    // `Rc::new` / `Arc::new`: a fresh counted
                    // handle over the value itself. Judged
                    // before the counted-slice arm below: a
                    // `Vec` source whose target pointee is the
                    // vector itself is this conversion, not
                    // the slice one.
                    Ok(self.rc_new_handle(value))
                } else if (target.starts_with("std::rc::Rc<")
                    || target.starts_with("std::sync::Arc<"))
                    && (source.starts_with("std::vec::Vec<")
                        || source.starts_with("&[")
                        || source.starts_with("&mut [")
                        || source.starts_with('[')
                        || matches!(
                            generic_first_arg(source, "std::boxed::Box<"),
                            Some(pt) if is_slice_tag(pt)
                        )
                        || matches!(
                            generic_first_arg(source, "alloc::boxed::Box<"),
                            Some(pt) if is_slice_tag(pt)
                        ))
                {
                    // `Rc<[T]>: From<Vec<T>>` / `From<&[T]>` /
                    // `From<[T; N]>` / `From<Box<[T]>>` (and the
                    // `Arc` twins): a fresh counted handle
                    // over the element sequence. The slice
                    // pointee lives in the allocation store as
                    // the element tuple, the shape every slice
                    // consumer of a reference-counted handle
                    // derives the length from (see the handle
                    // arm of `unsize_to_slice`). A boxed source
                    // hands over its pointee (releasing the box's
                    // allocation); every other source resolves
                    // as it is. The take is gated on the source
                    // type naming a box: a vector's value has
                    // the same one-field shape as a box
                    // aggregate (a single allocation reference),
                    // so the shape alone cannot decide -- an
                    // ungated take unwraps a vector's buffer
                    // and mistakes its contents for the element
                    // sequence itself.
                    let boxed_source = matches!(
                        generic_first_arg(source, "std::boxed::Box<"),
                        Some(pt) if is_slice_tag(pt)
                    ) || matches!(
                        generic_first_arg(source, "alloc::boxed::Box<"),
                        Some(pt) if is_slice_tag(pt)
                    );
                    let resolved = match if boxed_source {
                        self.unbox_take(&value)
                    } else {
                        None
                    } {
                        Some(pointee) => pointee,
                        None => self.resolve_value(frames, &value)?,
                    };
                    let elements = match resolved {
                        Value::Enum { fields, .. } => fields,
                        Value::Tuple(items) => items,
                        v @ Value::SparseVec { .. } => value_elements(&v)?,
                        Value::FatRef { base, .. } => match *base {
                            Value::Tuple(items) => items,
                            other => {
                                return Err(InterpError::type_mismatch(
                                    "From::from to a counted slice",
                                    "slice value over an element tuple",
                                    other.kind_name(),
                                ));
                            }
                        },
                        ref other => {
                            return Err(InterpError::type_mismatch(
                                "From::from to a counted slice",
                                "vector or slice value",
                                other.kind_name(),
                            ));
                        }
                    };
                    Ok(self.rc_new_handle(Value::Tuple(elements)))
                } else if target == "std::net::Ipv4Addr"
                    && (source == "[u8; 4]" || source == "u32")
                {
                    // `Ipv4Addr: From<[u8; 4]>` is the octets
                    // themselves; `From<u32>` is the big-endian
                    // encoding of the octets. The result is the
                    // address value in its declared shape (a
                    // one-field aggregate over the octet array).
                    let octets = if source == "u32" {
                        match value {
                            Value::Uint(v) => Some((v as u32).to_be_bytes()),
                            Value::Int(v) => Some((v as u32).to_be_bytes()),
                            _ => None,
                        }
                    } else {
                        ipv4_octets(&value)
                    };
                    match octets {
                        Some(octets) => Ok(ipv4_value(octets)),
                        None => Err(InterpError::type_mismatch(
                            "From::from to Ipv4Addr",
                            "octet array or u32 value",
                            value.kind_name(),
                        )),
                    }
                } else if target == "u32"
                    && (source == "std::net::Ipv4Addr" || source == "core::net::Ipv4Addr")
                {
                    // The reverse conversion: `u32::from(addr)`
                    // is the big-endian encoding of the octets.
                    match ipv4_octets(&value) {
                        Some(octets) => {
                            Ok(Value::Uint(u128::from(u32::from_be_bytes(octets))))
                        }
                        None => Err(InterpError::type_mismatch(
                            "From::from from Ipv4Addr",
                            "an Ipv4Addr value",
                            value.kind_name(),
                        )),
                    }
                } else if target
                    == "tracing_subscriber::filter::env::directive::Directive"
                    && source == "tracing_core::metadata::LevelFilter"
                {
                    // `Directive: From<LevelFilter>` is
                    // `Self { level, ..Self::default() }`
                    // (tracing-subscriber 0.3.23,
                    // `src/filter/env/directive.rs`): the given
                    // level in the last field, `None` span and
                    // target and an empty field-match vector
                    // before it, in declaration order.
                    let none = || Value::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    };
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![
                            none(),
                            Value::Enum {
                                variant: 0,
                                fields: Vec::new(),
                            },
                            none(),
                            value,
                        ],
                    })
                } else if [
                    "std::rc::Rc<",
                    "alloc::rc::Rc<",
                    "std::sync::Arc<",
                    "alloc::sync::Arc<",
                ]
                .iter()
                .any(|prefix| generic_first_arg(target, prefix) == Some("str"))
                {
                    // `Rc<str>` / `Arc<str>`: `From<&str>` and
                    // `From<String>` hand over the string itself;
                    // `From<Box<str>>` hands over the box
                    // aggregate `new_box` builds around it, so
                    // the string moves out of the box (releasing
                    // its allocation) before the handle is built.
                    let text = if matches!(value, Value::Str(_)) {
                        value
                    } else if generic_first_arg(source, "std::boxed::Box<") == Some("str")
                        || generic_first_arg(source, "alloc::boxed::Box<") == Some("str")
                    {
                        match self.unbox_take(&value) {
                            Some(s @ Value::Str(_)) => s,
                            _ => {
                                return Err(InterpError::type_mismatch(
                                    "From::from to a counted string",
                                    "boxed string value",
                                    value.kind_name(),
                                ))
                            }
                        }
                    } else if let Ok(s @ Value::Str(_)) =
                        self.resolve_value(frames, &value)
                    {
                        // A borrowed `&str` whose value is a
                        // reference into string storage (the
                        // dereference of a `Box<str>`, an
                        // `Rc<str>` or a `String`): follow the
                        // reference to the string data and copy
                        // it into the new handle. The borrowed
                        // storage is not consumed.
                        s
                    } else {
                        return Err(InterpError::type_mismatch(
                            "From::from to a counted string",
                            "string value",
                            value.kind_name(),
                        ));
                    };
                    Ok(self.rc_new_handle(text))
                } else if [
                    "std::rc::Rc<",
                    "alloc::rc::Rc<",
                    "std::sync::Arc<",
                    "alloc::sync::Arc<",
                ]
                .iter()
                .any(|prefix| generic_first_arg(target, prefix).is_some())
                {
                    // The remaining `From` impls for a counted
                    // handle (see `rc_from_value`): the generic
                    // `From<Box<T>>` and the `OsStr` / `Path` /
                    // `CStr` families' borrowed and owned forms.
                    // The counted-slice and counted-string arms
                    // above serve their own pointees first, and
                    // `From<T>` is the arm before them.
                    let pointee = [
                        "std::rc::Rc<",
                        "alloc::rc::Rc<",
                        "std::sync::Arc<",
                        "alloc::sync::Arc<",
                    ]
                    .iter()
                    .find_map(|prefix| generic_first_arg(target, prefix))
                    .expect("the arm condition found a pointee");
                    match self.rc_from_value(frames, pointee, source, value)? {
                        Some(handle) => Ok(handle),
                        None => Err(unsupported(format!(
                            "From::from from {source} to {target} is not runtime-supported"
                        ))),
                    }
                } else if target == "std::path::PathBuf"
                    && (source.starts_with("std::borrow::Cow<")
                        || source.starts_with("alloc::borrow::Cow<"))
                    && last_generic_param(source).as_deref() == Some("std::path::Path")
                {
                    // `PathBuf: From<Cow<'_, Path>>` is
                    // `into_owned`: the borrowed variant's
                    // field is the `&Path`, the owned
                    // variant's field is the `PathBuf` itself,
                    // and the `&OsStr` reader reads the bytes
                    // out of either shape (the generic
                    // `PathBuf` arm below cannot: the Cow
                    // aggregate is neither a wrapper aggregate
                    // nor a byte string to it).
                    let cow = self.resolve_value(frames, &value)?;
                    match cow {
                        Value::Enum { mut fields, .. } if fields.len() == 1 => {
                            let held = fields.pop().expect("len checked");
                            let bytes = self.os_str_bytes(frames, &held)?;
                            Ok(pathbuf_value(&bytes))
                        }
                        other => Err(InterpError::type_mismatch(
                            "From::from to PathBuf",
                            "cow aggregate",
                            other.kind_name(),
                        )),
                    }
                } else if target.starts_with("std::borrow::Cow<")
                    || target.starts_with("alloc::borrow::Cow<")
                {
                    // std's `From` impls for a `Cow<'_, B>`
                    // itself (see `cow_from_value`): the owned
                    // form of `B` wraps in the `Owned` variant;
                    // a `&B`, or a reference to the owned form,
                    // wraps in the `Borrowed` variant. Any
                    // other source keeps the unsupported error
                    // below.
                    let borrowed = last_generic_param(target).unwrap_or_default();
                    match self.cow_from_value(frames, &borrowed, source, value)? {
                        Some(cow) => Ok(cow),
                        None => Err(unsupported(format!(
                            "From::from from {source} to {target} is not runtime-supported"
                        ))),
                    }
                } else if target == "std::path::PathBuf" {
                    // `PathBuf: From<&str> / From<&Path> /
                    // From<&OsStr> / From<String> / From<OsString>`:
                    // every one of them is the source's byte
                    // string in a fresh buffer (the shape
                    // `pathbuf_value` builds). An owned
                    // string-like source arrives as its wrapper
                    // aggregate; a borrowed one as the byte
                    // string itself or a reference to it.
                    let bytes = match string_aggregate_bytes(&value) {
                        Some(bytes) => bytes,
                        // An owned source in its real aggregate
                        // form (an `OsString` the converted
                        // bodies built through raw-pointer
                        // writes) reads through the accessor
                        // that follows its buffer pointer.
                        None => match self.os_string_bytes_any(frames, &value)? {
                            Some(bytes) => bytes,
                            None => self.path_bytes_of(frames, &value)?.0,
                        },
                    };
                    Ok(pathbuf_value(&bytes))
                } else if target == "std::process::Stdio"
                    && matches!(
                        source,
                        "std::fs::File"
                            | "std::process::ChildStdin"
                            | "std::process::ChildStdout"
                            | "std::process::ChildStderr"
                            | "std::os::fd::OwnedFd"
                    )
                {
                    // std's `From` impls for `Stdio` wrap the
                    // source handle in the sys enum's `Fd`
                    // variant under the public one-field
                    // wrapper -- the shape `stdio_config_of`
                    // decodes to the file the handle carries.
                    // A source value that does not bottom out
                    // at a file-table id is not a handle this
                    // model can hand to a child; it keeps the
                    // unsupported error below.
                    match crate::interp::file_handle_id(&value) {
                        Some(_) => Ok(Value::Enum {
                            variant: 0,
                            fields: vec![Value::Enum {
                                variant: 3,
                                fields: vec![value],
                            }],
                        }),
                        None => Err(unsupported(format!(
                            "From::from from {source} to {target} is not runtime-supported"
                        ))),
                    }
                } else if let Some(wrapped) = self.guest_struct_from(target, value) {
                    // A guest struct whose `From` is the
                    // struct construction itself
                    // (`#[derive(From)]` and the trivial
                    // manual impls): see `guest_struct_from`.
                    Ok(wrapped)
                } else {
                    Err(unsupported(format!(
                        "From::from from {source} to {target} is not runtime-supported"
                    )))
                }
            }
            Builtin::MustUse => {
                arity(1)?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::TlsRegisterDtor => {
                arity(2)?;
                let mut vals = vals.into_iter();
                let data = vals.next().expect("arity checked");
                let dtor = vals.next().expect("arity checked");
                self.register_tls_dtor(data, dtor);
                Ok(Value::Unit)
            }
            Builtin::PanicAccessError => {
                // `#[track_caller]`: the caller-location stamp was
                // popped before the dispatch, like the other panic
                // entry points.
                arity(1)?;
                Err(InterpError::panic_str_payload(
                    "cannot access a Thread Local Storage value during or after destruction: AccessError",
                ))
            }
            Builtin::Deref => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                // A `no_std` guest names the library types
                // through its own crate's extern prelude
                // (`test::alloc::string::String`), where a std
                // guest sees the `std::` re-export path; drop one
                // leading segment when what follows is a library
                // path so both spellings reach the models below.
                let self_ty = match self_ty.split_once("::") {
                    Some((_, rest))
                        if rest.starts_with("alloc::")
                            || rest.starts_with("core::")
                            || rest.starts_with("std::") =>
                    {
                        rest
                    }
                    _ => self_ty,
                };
                if self_ty == "std::string::String"
                    || self_ty == "alloc::string::String"
                {
                    // `String::deref -> &str`: the string data itself.
                    let s = self.resolve_string(frames, &vals[0])?;
                    Ok(Value::Str(s))
                } else if self_ty == "std::path::PathBuf" {
                    // `PathBuf::deref -> &Path`: the path the
                    // buffer holds, in the shape a path over the
                    // same bytes denotes (see `sub_path_value`).
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    let bytes = self.path_buf_bytes_any(frames, &pointee)?.ok_or_else(|| {
                        InterpError::type_mismatch(
                            "Deref::deref for PathBuf",
                            "a path buffer value",
                            pointee.kind_name(),
                        )
                    })?;
                    Ok(sub_path_value(&bytes))
                } else if self_ty == "std::ffi::OsString" {
                    // `OsString::deref -> &OsStr`: the string's
                    // bytes as the slice value, the same shape
                    // `PathBuf`'s deref produces.
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    let bytes = self.os_string_bytes_any(frames, &pointee)?.ok_or_else(|| {
                        InterpError::type_mismatch(
                            "Deref::deref for OsString",
                            "an OS string value",
                            pointee.kind_name(),
                        )
                    })?;
                    Ok(sub_path_value(&bytes))
                } else if self_ty == "std::ffi::CString" {
                    // `CString::deref -> &CStr`: the fat
                    // reference over the owned buffer.
                    self.c_string_deref(frames, &vals[0])
                } else if self_ty.starts_with("std::borrow::Cow<")
                    || self_ty.starts_with("alloc::borrow::Cow<")
                {
                    // `Cow<B>::deref -> &B::Target`: the borrowed
                    // variant's field already IS the target value
                    // (a `&str` / `&[u8]` reference value); the
                    // owned variant's field is the owned value
                    // (`String` / `Vec<_>`), whose deref is the
                    // same data read as the target — for a string
                    // target that is the string data itself. The
                    // string case is decided by the Cow's own
                    // borrowed-type parameter being exactly `str`:
                    // a borrowed type that merely mentions `str`
                    // (a slice of `str`, `OsStr`, `CStr`) is a
                    // different target, and its owned value
                    // derefs as itself.
                    let borrowed_ty = last_generic_param(self_ty);
                    let cow = self.resolve_value(frames, &vals[0])?;
                    match cow {
                        Value::Enum { variant, mut fields, .. }
                            if fields.len() == 1 =>
                        {
                            let field = fields.pop().expect("len checked");
                            if variant == 0 {
                                Ok(field)
                            } else if borrowed_ty.as_deref() == Some("str") {
                                let s = self.resolve_string(frames, &field)?;
                                Ok(Value::Str(s))
                            } else if borrowed_ty.as_deref().is_some_and(|b| {
                                b.starts_with('[') && b.ends_with(']')
                            }) {
                                // A slice target: the deref is
                                // the fat reference over the owned
                                // vector, the shape `Vec`'s own
                                // deref produces — a reference to
                                // the Cow's field when the Cow
                                // lives in a place (so writes land
                                // in the Cow), the field value
                                // itself for a detached Cow.
                                let len = field.len().ok_or_else(|| {
                                    InterpError::type_mismatch(
                                        "Deref::deref for Cow",
                                        "array value",
                                        field.kind_name(),
                                    )
                                })?;
                                let base = match self.place_projected_ref(
                                    frames,
                                    &vals[0],
                                    RefProj::Field(0),
                                )? {
                                    Some(reference) => reference,
                                    None => field,
                                };
                                Ok(Value::FatRef {
                                    base: Box::new(base),
                                    len,
                                })
                            } else if matches!(
                                borrowed_ty.as_deref(),
                                Some("std::ffi::OsStr") | Some("OsStr")
                            ) {
                                // An owned `Cow<OsStr>` holds an
                                // `OsString`; its deref is the
                                // byte slice over the string's
                                // bytes, the shape `OsString`'s
                                // own deref produces (see the
                                // `OsString` arm above).
                                let field = self.resolve_value(frames, &field)?;
                                let bytes = match self
                                    .os_string_bytes_any(frames, &field)?
                                {
                                    Some(bytes) => bytes,
                                    None => self.os_str_bytes(frames, &field)?,
                                };
                                Ok(sub_path_value(&bytes))
                            } else if matches!(
                                borrowed_ty.as_deref(),
                                Some("std::path::Path") | Some("Path")
                            ) {
                                // An owned `Cow<Path>` holds a
                                // `PathBuf`; its deref is the
                                // path over the buffer's bytes,
                                // the shape `PathBuf`'s own deref
                                // produces.
                                let field = self.resolve_value(frames, &field)?;
                                let bytes = match self
                                    .path_buf_bytes_any(frames, &field)?
                                {
                                    Some(bytes) => bytes,
                                    None => self.os_str_bytes(frames, &field)?,
                                };
                                Ok(sub_path_value(&bytes))
                            } else if matches!(
                                borrowed_ty.as_deref(),
                                Some("std::ffi::CStr") | Some("CStr")
                            ) {
                                // An owned `Cow<CStr>` holds a
                                // `CString`; its deref is the fat
                                // reference over the owned
                                // buffer, as `CString`'s own
                                // deref produces.
                                self.c_string_deref(frames, &field)
                            } else {
                                Ok(field)
                            }
                        }
                        ref other => Err(InterpError::type_mismatch(
                            "Deref::deref for Cow",
                            "cow aggregate",
                            other.kind_name(),
                        )),
                    }
                } else if self_ty.starts_with("std::vec::Vec<")
                    || self_ty.starts_with("alloc::vec::Vec<")
                {
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    let len = pointee.len().ok_or_else(|| {
                        InterpError::type_mismatch(
                            "Deref::deref for Vec",
                            "array value",
                            pointee.kind_name(),
                        )
                    })?;
                    Ok(Value::FatRef {
                        base: Box::new(vals.into_iter().next().expect("arity checked")),
                        len,
                    })
                } else if smallvec_array_arg(self_ty).is_some() {
                    self.smallvec_slice(
                        frames,
                        &vals[0],
                        self_ty,
                        "Deref::deref for SmallVec",
                    )
                } else if self_ty.starts_with("std::boxed::Box<")
                    || self_ty.starts_with("alloc::boxed::Box<")
                {
                    // A built box's single field is its allocation
                    // pointer, and `deref` yields that very
                    // reference: it names the pointee's storage in
                    // the allocation store, live for as long as the
                    // allocation is. (A box value of the older
                    // aggregate shape — one built by a converted
                    // body or a downcast — holds its pointee as the
                    // field value itself; `deref` then yields a
                    // reference to that value, as it does for
                    // constant aggregates.) A `Box<dyn Trait>` is
                    // the fat pointer itself, and dereferencing it
                    // yields that pointer — the `&dyn` the trait
                    // methods dispatch through.
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    if let Value::DynRef { .. } = pointee {
                        return Ok(pointee);
                    }
                    match pointee {
                        Value::Enum { fields, .. } if fields.len() == 1 => {
                            let inner = fields.into_iter().next().expect("len checked");
                            if let Value::Ref(RefTarget::Alloc { .. }) = inner {
                                return Ok(inner);
                            }
                            // An aggregate-shaped box holds its
                            // pointee as the field value itself.
                            // When the box lives in a place, the
                            // deref denotes that place's field 0,
                            // so writes through the result land in
                            // the box's storage; a box reached
                            // only as a value keeps the snapshot
                            // form, as for constant aggregates.
                            // (A field that is itself a one-field
                            // wrapper around a reference is another
                            // pointer layer, not the pointee: its
                            // snapshot is kept, as for the built
                            // box's allocation pointer above.)
                            let wrapped_pointer = matches!(
                                &inner,
                                Value::Enum { fields, .. }
                                    if fields.len() == 1
                                        && matches!(fields[0], Value::Ref(_))
                            );
                            if !wrapped_pointer {
                                if let Some(place_ref) = self.place_projected_ref(
                                    frames,
                                    &vals[0],
                                    RefProj::Field(0),
                                )? {
                                    return Ok(place_ref);
                                }
                            }
                            Ok(Value::Ref(RefTarget::value(inner)))
                        }
                        ref other => Err(InterpError::type_mismatch(
                            "Deref::deref for Box",
                            "boxed value",
                            other.kind_name(),
                        )),
                    }
                } else if self_ty.starts_with("std::rc::Rc<")
                    || self_ty.starts_with("std::sync::Arc<")
                {
                    // The handle names an allocation in the store:
                    // `deref` yields a reference to the stored
                    // value field — a live place, so reads and
                    // writes reach the storage every handle shares.
                    // An `Rc<dyn Trait>` / `Arc<dyn Trait>`
                    // dereferences to the fat pointer itself, as
                    // `Box<dyn Trait>` does.
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    if let Value::DynRef { .. } = pointee {
                        return Ok(pointee);
                    }
                    match rc_handle_id(&pointee) {
                        Some(id) => Ok(Value::Ref(RefTarget::Alloc {
                            id,
                            projection: self
                                .rc_pointee_projection(&pointee)
                                .expect("handle checked"),
                        })),
                        None => Err(InterpError::type_mismatch(
                            "Deref::deref for Rc/Arc",
                            "a reference-counted handle",
                            pointee.kind_name(),
                        )),
                    }
                } else if self_ty_is(self_ty, "std::cell::Ref")
                    || self_ty_is(self_ty, "std::cell::RefMut")
                {
                    // A borrow guard: field 0 is the reference to
                    // the cell's value. The borrow builtins store
                    // it bare; a guard a converted guest body built
                    // from core's real layout wraps it in a
                    // single-field aggregate (`NonNull`), which
                    // `refcell_guard_ref` peels. A field that does
                    // not resolve to a reference keeps the stored
                    // value itself, as before.
                    let guard = self.resolve_value(frames, &vals[0])?;
                    match crate::interp::refcell_guard_fields(&guard) {
                        Some(fields) => {
                            match crate::interp::refcell_guard_ref(&fields[0]) {
                                Some(target) => Ok(Value::Ref(target.clone())),
                                None => Ok(fields[0].clone()),
                            }
                        }
                        None => Err(InterpError::type_mismatch(
                            "RefCell borrow guard",
                            "Ref/RefMut guard aggregate",
                            guard.kind_name(),
                        )),
                    }
                } else if self_ty_is(self_ty, "std::sync::MutexGuard")
                    || self_ty_is(self_ty, "std::sync::RwLockWriteGuard")
                {
                    // A lock guard: field 0 is the reference to
                    // the lock the converted lock body stored;
                    // the guarded value is the lock's data field.
                    self.lock_guard_deref(
                        frames,
                        &vals[0],
                        "Deref::deref for lock guard",
                    )
                } else if self_ty_is(self_ty, "std::sync::RwLockReadGuard") {
                    // A read guard: field 0 is the `NonNull` data
                    // pointer its constructor took from the lock.
                    self.read_guard_deref(
                        frames,
                        &vals[0],
                        "Deref::deref for RwLockReadGuard",
                    )
                } else if self_ty_is(self_ty, "std::sync::MappedMutexGuard")
                    || self_ty_is(self_ty, "std::sync::MappedRwLockReadGuard")
                    || self_ty_is(self_ty, "std::sync::MappedRwLockWriteGuard")
                    || self_ty_is(self_ty, "std::sync::nonpoison::MappedMutexGuard")
                    || self_ty_is(self_ty, "std::sync::nonpoison::MappedRwLockReadGuard")
                    || self_ty_is(self_ty, "std::sync::nonpoison::MappedRwLockWriteGuard")
                {
                    // A mapped guard: field 0 is the `NonNull`
                    // data pointer `map` built from the mapping
                    // closure's returned reference, which denotes
                    // the mapped component itself.
                    self.mapped_guard_deref(
                        frames,
                        &vals[0],
                        "Deref::deref for mapped lock guard",
                    )
                } else if self_ty.starts_with('&') {
                    // A reference's own `Deref` returns the inner
                    // reference itself: `<&T as Deref>::deref` and
                    // `<&mut T as Deref>::deref` take `&self`, so
                    // the argument names the place holding the
                    // inner reference, and reading it exactly once
                    // yields that reference. Generic code over
                    // `P: Deref` instantiated with a reference (the
                    // `Pin<&mut T>` machinery, `ManuallyDrop`
                    // access, ...) reaches this arm.
                    Ok(self.read_ref_once(frames, &vals[0])?)
                } else if self_ty.starts_with("std::pin::Pin<") {
                    // `Pin<P>` derefs exactly as the pointer it
                    // wraps (see `pin_deref_value`).
                    let wrapper = self.resolve_value(frames, &vals[0])?;
                    pin_deref_value(wrapper, self_ty, "Deref::deref for Pin")
                } else if self_ty.starts_with("std::mem::ManuallyDrop<") {
                    // A one-field newtype wrapper: the result is a
                    // reference to the held value.
                    self.field_zero_ref(frames, &vals[0], "Deref::deref for ManuallyDrop")
                } else if self_ty.starts_with("std::boxed::ThinBox<") {
                    self.thin_box_deref(
                        frames,
                        &vals[0],
                        self_ty,
                        "Deref::deref for ThinBox",
                    )
                } else if self_ty.starts_with("std::panic::AssertUnwindSafe<") {
                    // A one-field newtype wrapper: the result is a
                    // reference to the held value.
                    self.field_zero_ref(frames, &vals[0], "Deref::deref for AssertUnwindSafe")
                } else if self_ty == "std::path::PathBuf" {
                    // `PathBuf::deref -> &Path`: the model's path
                    // values hold the byte string itself at the
                    // bottom of a chain of one-field wrappers (the
                    // `OsString`, its buffer, the byte vector's own
                    // wrapper), and a `Path` is that same byte
                    // string under the same kind of wrappers (the
                    // constructors reach it through raw pointer
                    // casts). Descend the chain and return a
                    // reference to the byte data: a live reference
                    // when the buffer lives in a place, and a
                    // self-contained fat pointer when the data is a
                    // byte vector's element aggregate.
                    let resolved = self.deref_value(frames, &vals[0])?;
                    let stored = self.read_resolved(frames, &resolved)?;
                    let mut depth = 0usize;
                    let mut bottom = &stored;
                    while let Value::Enum { fields, .. } = bottom {
                        // The byte vector itself is an aggregate
                        // of `Uint` elements: stop there, even
                        // when it holds a single byte (a one-byte
                        // path's vector is a one-field aggregate,
                        // indistinguishable in shape from a
                        // wrapper without this check).
                        if fields
                            .iter()
                            .all(|f| matches!(f, Value::Uint(_)))
                        {
                            break;
                        }
                        if fields.len() != 1 {
                            break;
                        }
                        depth += 1;
                        bottom = &fields[0];
                    }
                    match bottom {
                        Value::Str(_) | Value::Bytes(_) => {
                            let steps =
                                vec![RefProj::Field(0); depth];
                            match resolved {
                                Resolved::Place {
                                    frame_pos,
                                    local,
                                    mut projection,
                                } => {
                                    projection.extend(steps);
                                    Ok(Value::Ref(RefTarget::Place {
                                        frame: frames[frame_pos].id,
                                        local,
                                        projection,
                                    }))
                                }
                                Resolved::Static {
                                    name,
                                    mut projection,
                                } => {
                                    projection.extend(steps);
                                    Ok(Value::Ref(RefTarget::Static {
                                        name,
                                        projection,
                                    }))
                                }
                                Resolved::Alloc {
                                    id,
                                    mut projection,
                                } => {
                                    projection.extend(steps);
                                    Ok(Value::Ref(RefTarget::Alloc {
                                        id,
                                        projection,
                                    }))
                                }
                                Resolved::Value(base, mut projection) => {
                                    projection.extend(steps);
                                    Ok(Value::Ref(RefTarget::Value(
                                        Box::new(base),
                                        projection,
                                    )))
                                }
                            }
                        }
                        Value::FatRef { .. } => Ok(bottom.clone()),
                        Value::Enum { fields, .. }
                            if fields
                                .iter()
                                .all(|f| matches!(f, Value::Uint(_))) =>
                        {
                            let bytes: Vec<u8> = fields
                                .iter()
                                .map(|f| match f {
                                    Value::Uint(b) => *b as u8,
                                    _ => unreachable!("shape checked"),
                                })
                                .collect();
                            let len = bytes.len() as u128;
                            Ok(Value::FatRef {
                                base: Box::new(Value::Bytes(bytes)),
                                len,
                            })
                        }
                        other => Err(InterpError::type_mismatch(
                            "Deref::deref for PathBuf",
                            "path buffer holding byte data",
                            other.kind_name(),
                        )),
                    }
                } else {
                    Err(unsupported(format!(
                        "Deref::deref for {self_ty} is not runtime-supported"
                    )))
                }
            }
            Builtin::PathPart(part) => {
                arity(1)?;
                let (bytes, was_str) = self.path_bytes_of(frames, &vals[0])?;
                let part = {
                    let path = std::path::Path::new(
                        crate::platform::os_str_from_bytes(&bytes)
                            .ok_or_else(|| {
                                unsupported(
                                    "Path part: the path bytes have no host string form on this platform",
                                )
                            })?,
                    );
                    let part_os: Option<&std::ffi::OsStr> = match part {
                        PathPart::Parent => {
                            path.parent().map(|p| p.as_os_str())
                        }
                        PathPart::FileName => path.file_name(),
                        PathPart::FileStem => path.file_stem(),
                        PathPart::Extension => path.extension(),
                    };
                    match part_os {
                        Some(os) => Some(
                            crate::platform::os_str_bytes(os)
                                .ok_or_else(|| {
                                    unsupported(
                                        "Path part: the host component has no byte form on this platform",
                                    )
                                })?
                                .into_owned(),
                        ),
                        None => None,
                    }
                };
                Ok(match part {
                    Some(bytes) => some_value(if was_str {
                        // The parts of a text path are text: the
                        // parse only cuts at ASCII separators.
                        match String::from_utf8(bytes) {
                            Ok(s) => Value::Str(s),
                            Err(e) => Value::Bytes(e.into_bytes()),
                        }
                    } else {
                        Value::Bytes(bytes)
                    }),
                    None => none_value(),
                })
            }
            Builtin::PathComponents => {
                arity(1)?;
                let (bytes, _) = self.path_bytes_of(frames, &vals[0])?;
                Ok(components_state(&bytes, 0, 0))
            }
            Builtin::PathToPathBuf => {
                arity(1)?;
                let (bytes, _) = self.path_bytes_of(frames, &vals[0])?;
                Ok(pathbuf_value(&bytes))
            }
            Builtin::PathBufPush => {
                arity(2)?;
                let (addition, _) = self.path_bytes_of(frames, &vals[1])?;
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_resolved(frames, &target, |slot| {
                    let current = string_aggregate_bytes(slot).ok_or_else(|| {
                        InterpError::type_mismatch(
                            "PathBuf::push",
                            "path buffer holding byte data",
                            slot.kind_name(),
                        )
                    })?;
                    let mut buf = host_path(&current)?;
                    buf.push(host_path(&addition)?);
                    *slot = pathbuf_value(&host_path_bytes(&buf)?);
                    Ok(Value::Unit)
                })
            }
            Builtin::PathBufPop => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_resolved(frames, &target, |slot| {
                    let current = string_aggregate_bytes(slot).ok_or_else(|| {
                        InterpError::type_mismatch(
                            "PathBuf::pop",
                            "path buffer holding byte data",
                            slot.kind_name(),
                        )
                    })?;
                    let mut buf = host_path(&current)?;
                    let popped = buf.pop();
                    *slot = pathbuf_value(&host_path_bytes(&buf)?);
                    Ok(Value::Bool(popped))
                })
            }
            Builtin::PathBufSetExtension => {
                arity(2)?;
                let (extension, _) = self.path_bytes_of(frames, &vals[1])?;
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_resolved(frames, &target, |slot| {
                    let current = string_aggregate_bytes(slot).ok_or_else(|| {
                        InterpError::type_mismatch(
                            "PathBuf::set_extension",
                            "path buffer holding byte data",
                            slot.kind_name(),
                        )
                    })?;
                    let mut buf = host_path(&current)?;
                    let extension_os =
                        crate::platform::os_str_from_bytes(&extension)
                            .ok_or_else(|| {
                                unsupported(
                                    "PathBuf::set_extension: the extension has no host string form on this platform",
                                )
                            })?;
                    let set = buf.set_extension(extension_os);
                    *slot = pathbuf_value(&host_path_bytes(&buf)?);
                    Ok(Value::Bool(set))
                })
            }
            Builtin::PathBufSetFileName => {
                arity(2)?;
                let (file_name, _) = self.path_bytes_of(frames, &vals[1])?;
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_resolved(frames, &target, |slot| {
                    let current = string_aggregate_bytes(slot).ok_or_else(|| {
                        InterpError::type_mismatch(
                            "PathBuf::set_file_name",
                            "path buffer holding byte data",
                            slot.kind_name(),
                        )
                    })?;
                    let mut buf = host_path(&current)?;
                    // std's own definition: pop the current
                    // file name when there is one, then push
                    // the new one.
                    if buf.file_name().is_some() {
                        let popped = buf.pop();
                        debug_assert!(popped);
                    }
                    let name_os =
                        crate::platform::os_str_from_bytes(&file_name)
                            .ok_or_else(|| {
                                unsupported(
                                    "PathBuf::set_file_name: the file name has no host string form on this platform",
                                )
                            })?;
                    buf.push(name_os);
                    *slot = pathbuf_value(&host_path_bytes(&buf)?);
                    Ok(Value::Unit)
                })
            }
            Builtin::SysPathAbsolute => {
                arity(1)?;
                let (bytes, _) = self.path_bytes_of(frames, &vals[0])?;
                let path = host_path(&bytes)?;
                match std::path::absolute(&path) {
                    Ok(abs) => {
                        Ok(io_result_ok(pathbuf_value(&host_path_bytes(&abs)?)))
                    }
                    Err(error) => Ok(io_result_err(host_io_error(&error))),
                }
            }
            Builtin::PathWithExtension => {
                arity(2)?;
                let (bytes, _) = self.path_bytes_of(frames, &vals[0])?;
                let (extension, _) = self.path_bytes_of(frames, &vals[1])?;
                let extension_os =
                    crate::platform::os_str_from_bytes(&extension)
                        .ok_or_else(|| {
                            unsupported(
                                "Path::with_extension: the extension has no host string form on this platform",
                            )
                        })?;
                let buf = host_path(&bytes)?.with_extension(extension_os);
                Ok(pathbuf_value(&host_path_bytes(&buf)?))
            }
            Builtin::PathBufNew => Ok(pathbuf_value(&[])),
            Builtin::FromUtf8 => {
                arity(1)?;
                let (bytes, _) = self.path_bytes_of(frames, &vals[0])?;
                Ok(match std::str::from_utf8(&bytes) {
                    Ok(_) => Value::Enum {
                        variant: 0,
                        fields: vec![Value::Str(
                            // The bytes are valid UTF-8 (just
                            // checked) and the model's string is
                            // the same data.
                            String::from_utf8(bytes)
                                .expect("validity checked above"),
                        )],
                    },
                    Err(e) => Value::Enum {
                        variant: 1,
                        fields: vec![Value::Enum {
                            variant: 0,
                            fields: vec![
                                Value::Uint(e.valid_up_to() as u128),
                                match e.error_len() {
                                    Some(n) => some_value(Value::Uint(n as u128)),
                                    None => none_value(),
                                },
                            ],
                        }],
                    },
                })
            }
            Builtin::ToUppercase => {
                arity(1)?;
                // The argument is the `str` data itself or a
                // reference to it, depending on the MIR shape.
                let s = match &vals[0] {
                    Value::Str(s) => s.clone(),
                    _ => self.resolve_string(frames, &vals[0])?,
                };
                Ok(Value::Str(s.to_uppercase()))
            }
            Builtin::ToLowercase => {
                arity(1)?;
                let s = match &vals[0] {
                    Value::Str(s) => s.clone(),
                    _ => self.resolve_string(frames, &vals[0])?,
                };
                Ok(Value::Str(s.to_lowercase()))
            }
            Builtin::StringNew => {
                arity(0)?;
                Ok(Value::Str(String::new()))
            }
            Builtin::StringWithCapacity => {
                // The capacity argument is not represented, exactly
                // as for `Vec::with_capacity`: the result is the
                // empty string. A guest allocator still observes
                // the buffer allocation the native call makes
                // (the abstract string never reads the buffer;
                // the matching deallocation runs through the
                // string's drop glue as usual).
                arity(1)?;
                let cap = match &vals[0] {
                    Value::Uint(n) => *n,
                    Value::Int(n) if *n >= 0 => *n as u128,
                    _ => 0,
                };
                if cap > 0 {
                    if let Some(ptr) = self.call_guest_alloc(frames, cap, 1)? {
                        *self.pending_string_buffer.borrow_mut() =
                            Some((ptr, cap));
                    }
                }
                Ok(Value::Str(String::new()))
            }
            Builtin::StringLen | Builtin::StrLen => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                Ok(Value::Uint(s.len() as u128))
            }
            Builtin::StringIsEmpty | Builtin::StrIsEmpty => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                Ok(Value::Bool(s.is_empty()))
            }
            Builtin::StringCapacity => {
                arity(1)?;
                Err(unsupported(format!(
                    "{name}: allocation capacity is not tracked by \
                     the abstract string model"
                )))
            }
            Builtin::StringAsStr => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                Ok(Value::Str(s))
            }
            Builtin::StringAsBytes | Builtin::StrAsBytes => {
                arity(1)?;
                self.string_bytes_ref(frames, &vals[0])
            }
            Builtin::StrAsBytesMut => {
                arity(1)?;
                let view = vals[0].clone();
                let target = self.deref_value(frames, &view)?;
                let text = self.string_text_at(frames, &target, "str::as_bytes_mut")?;
                let len = text.len() as u128;
                let bytes = Value::Enum {
                    variant: 0,
                    fields: text
                        .into_bytes()
                        .into_iter()
                        .map(|b| Value::Uint(b as u128))
                        .collect(),
                };
                self.write_resolved(frames, &target, bytes)?;
                // The byte slice denotes the same storage the
                // `&mut str` denoted, at the string's byte
                // length.
                let base = match view {
                    Value::FatRef { base, .. } => *base,
                    other => other,
                };
                Ok(Value::FatRef {
                    base: Box::new(base),
                    len,
                })
            }
            Builtin::StringIntoBytes => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: s
                        .into_bytes()
                        .into_iter()
                        .map(|b| Value::Uint(b as u128))
                        .collect(),
                })
            }
            Builtin::StringAsMutVec => {
                arity(1)?;
                let view = vals[0].clone();
                let target = self.deref_value(frames, &view)?;
                let text = self.string_text_at(frames, &target, "String::as_mut_vec")?;
                // The string value in the byte vector's own
                // element form: the returned reference denotes
                // the receiver's storage itself (the vector
                // operations resolve it with no projection), so
                // the element form is what they mutate in
                // place, and the string readers coerce that
                // form back to text (see `resolve_string`).
                let bytes = Value::Enum {
                    variant: 0,
                    fields: text
                        .into_bytes()
                        .into_iter()
                        .map(|b| Value::Uint(b as u128))
                        .collect(),
                };
                self.write_resolved(frames, &target, bytes)?;
                Ok(view)
            }
            Builtin::StringLeak => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                let len = s.len() as u128;
                let id = self.alloc_insert(Value::Str(s));
                self.record_alloc_ty(id, Ty::Str);
                Ok(Value::FatRef {
                    base: Box::new(Value::Ref(RefTarget::Alloc {
                        id,
                        projection: Vec::new(),
                    })),
                    len,
                })
            }
            Builtin::StringIntoBoxedStr => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                let boxed = self.new_box(Value::Str(s));
                self.record_box_ty(&boxed, Ty::Str);
                Ok(boxed)
            }
            Builtin::OsStringIntoBoxedOsStr => {
                arity(1)?;
                let pointee = self.resolve_value(frames, &vals[0])?;
                let bytes = self.os_string_bytes_any(frames, &pointee)?.ok_or_else(|| {
                    InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "an OS string value",
                        pointee.kind_name(),
                    )
                })?;
                let boxed = self.new_box(Value::Bytes(bytes));
                self.record_box_ty(&boxed, Ty::Slice(Box::new(Ty::Uint(crate::ir::UintTy::U8))));
                Ok(boxed)
            }
            Builtin::PathBufIntoBoxedPath => {
                arity(1)?;
                // The box holds the path's bytes themselves,
                // the pointee form every `Box<Path>` consumer
                // reads (through `Deref`, the comparisons,
                // the counted-pointer conversions), as the
                // `OsString::into_boxed_os_str` box holds the
                // OS string's bytes. The `Path` pointee type
                // recorded on the allocation is the signal
                // the path-box recognition in tagless
                // comparisons consults (see
                // `operand_is_path_box`).
                let bytes = self.os_str_bytes(frames, &vals[0])?;
                let boxed = self.new_box(Value::Bytes(bytes));
                if let Some(ty) = ty_from_tag("std::path::Path") {
                    self.record_box_ty(&boxed, ty);
                }
                Ok(boxed)
            }
            Builtin::NoAllocShimMarker => {
                arity(0)?;
                Ok(Value::Unit)
            }
            Builtin::AsRef => {
                arity(1)?;
                self.as_ref_view(frames, generic_args, &vals[0], false)
            }
            Builtin::AsMut => {
                arity(1)?;
                self.as_ref_view(frames, generic_args, &vals[0], true)
            }
            Builtin::VecIntoBoxedSlice => {
                arity(1)?;
                // The generic argument is the element type; the
                // pointee is a slice of it.
                let pointee_ty = generic_args
                    .first()
                    .and_then(|tag| ty_from_tag(tag))
                    .map(|elem| Ty::Slice(Box::new(elem)));
                let value = vals.into_iter().next().expect("arity checked");
                self.boxed_slice_value(value, pointee_ty, "Vec::into_boxed_slice")
            }
            Builtin::StringPushStr => {
                arity(2)?;
                let suffix = self.resolve_string(frames, &vals[1])?;
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_string(frames, &target, "String::push_str", |text| {
                    text.push_str(&suffix);
                    Ok(Value::Unit)
                })
            }
            Builtin::StringPush => {
                arity(2)?;
                let c = match vals[1] {
                    Value::Uint(v) => u32::try_from(v)
                        .ok()
                        .and_then(char::from_u32)
                        .ok_or_else(|| {
                            InterpError::type_mismatch(
                                "String::push",
                                "char",
                                "invalid code point",
                            )
                        })?,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            "String::push",
                            "char",
                            other.kind_name(),
                        ))
                    }
                };
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_string(frames, &target, "String::push", |text| {
                    text.push(c);
                    Ok(Value::Unit)
                })
            }
            Builtin::StringInsert => {
                arity(3)?;
                let idx = expect_index(&vals[1], "String::insert")?;
                let c = match vals[2] {
                    Value::Uint(v) => u32::try_from(v)
                        .ok()
                        .and_then(char::from_u32)
                        .ok_or_else(|| {
                            InterpError::type_mismatch(
                                "String::insert",
                                "char",
                                "invalid code point",
                            )
                        })?,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            "String::insert",
                            "char",
                            other.kind_name(),
                        ))
                    }
                };
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_string(frames, &target, "String::insert", |text| {
                    // std's body opens with this assert (its
                    // message covers the out-of-range case too:
                    // `is_char_boundary` is false there).
                    if !text.is_char_boundary(idx) {
                        return Err(InterpError::panic(
                            "assertion failed: self.is_char_boundary(idx)".to_string(),
                        ));
                    }
                    text.insert(idx, c);
                    Ok(Value::Unit)
                })
            }
            Builtin::StringInsertStr => {
                arity(3)?;
                let idx = expect_index(&vals[1], "String::insert_str")?;
                let suffix = self.resolve_string(frames, &vals[2])?;
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_string(frames, &target, "String::insert_str", |text| {
                    if !text.is_char_boundary(idx) {
                        return Err(InterpError::panic(
                            "assertion failed: self.is_char_boundary(idx)".to_string(),
                        ));
                    }
                    text.insert_str(idx, &suffix);
                    Ok(Value::Unit)
                })
            }
            Builtin::StringRemove => {
                arity(2)?;
                let idx = expect_index(&vals[1], "String::remove")?;
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_string(frames, &target, "String::remove", |text| {
                    // std's body reads the char through
                    // `self[idx..]`: the three failure shapes
                    // are that indexing's and the end-of-string
                    // case's own panic messages.
                    if idx >= text.len() {
                        if idx == text.len() {
                            return Err(InterpError::panic(
                                "cannot remove a char from the end of a string"
                                    .to_string(),
                            ));
                        }
                        return Err(InterpError::panic(format!(
                            "start byte index {idx} is out of bounds for string of length {}",
                            text.len()
                        )));
                    }
                    if !text.is_char_boundary(idx) {
                        let lo = text
                            .char_indices()
                            .take_while(|(i, _)| *i < idx)
                            .last()
                            .map_or(0, |(i, _)| i);
                        let ch = text[lo..].chars().next().expect("char at boundary");
                        let hi = lo + ch.len_utf8();
                        return Err(InterpError::panic(format!(
                            "start byte index {idx} is not a char boundary; \
                             it is inside {ch:?} (bytes {lo}..{hi} of string)"
                        )));
                    }
                    Ok(Value::Uint(u128::from(text.remove(idx) as u32)))
                })
            }
            Builtin::StringTruncate => {
                arity(2)?;
                let new_len = expect_index(&vals[1], "String::truncate")?;
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_string(frames, &target, "String::truncate", |text| {
                    if new_len <= text.len() {
                        if !text.is_char_boundary(new_len) {
                            return Err(InterpError::panic(
                                "assertion failed: self.is_char_boundary(new_len)"
                                    .to_string(),
                            ));
                        }
                        text.truncate(new_len);
                    }
                    Ok(Value::Unit)
                })
            }
            Builtin::StringPop => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_string(frames, &target, "String::pop", |text| {
                    Ok(match text.pop() {
                        Some(c) => some_value(Value::Uint(u128::from(c as u32))),
                        None => none_value(),
                    })
                })
            }
            Builtin::StringClear => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_string(frames, &target, "String::clear", |text| {
                    text.clear();
                    Ok(Value::Unit)
                })
            }
            Builtin::StringRetain => {
                // Dispatched before the generic argument
                // evaluation (see `exec_string_retain`).
                Err(unsupported(
                    "String::retain reached the generic builtin dispatch",
                ))
            }
            Builtin::StringReplaceRange => {
                arity(3)?;
                let replacement = self.resolve_string(frames, &vals[2])?;
                // The range's bounds as stored in its value,
                // and its kind from the call's generic arguments
                // (the `R` the guest chose), decoded exactly as
                // the slice-index builtin decodes its range (see
                // `index_range` in `builtins::views`).
                let fields: Vec<Value> = match &vals[1] {
                    Value::Enum { fields, .. } => fields.clone(),
                    Value::Tuple(fields) => fields.clone(),
                    Value::Unit => Vec::new(),
                    other => {
                        return Err(InterpError::type_mismatch(
                            "builtin String::replace_range",
                            "range value",
                            other.kind_name(),
                        ))
                    }
                };
                let idx_tag = generic_args
                    .iter()
                    .find(|t| t.contains("Range"))
                    .map(String::as_str)
                    .unwrap_or("");
                let kind = if idx_tag.contains("RangeFull") {
                    RangeKindForIndex::Full
                } else if idx_tag.contains("RangeToInclusive") {
                    RangeKindForIndex::ToInclusive
                } else if idx_tag.contains("RangeInclusive") {
                    RangeKindForIndex::Inclusive
                } else if idx_tag.contains("RangeFrom") {
                    RangeKindForIndex::From
                } else if idx_tag.contains("RangeTo") {
                    RangeKindForIndex::To
                } else {
                    RangeKindForIndex::Range
                };
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_string(frames, &target, "String::replace_range", |text| {
                    let bound = |i: usize| -> InterpResult<usize> {
                        match fields.get(i) {
                            Some(v) => expect_index(v, "String::replace_range"),
                            None => Err(InterpError::type_mismatch(
                                "builtin String::replace_range",
                                "range value with bounds",
                                "boundless range value",
                            )),
                        }
                    };
                    // `slice::range`'s own checks, in its own
                    // messages (see `range_bounds`).
                    let (start, end) = range_bounds(&kind, &bound, text.len())?;
                    if !text.is_char_boundary(start) {
                        return Err(InterpError::panic(
                            "start of range should be a character boundary".to_string(),
                        ));
                    }
                    if !text.is_char_boundary(end) {
                        return Err(InterpError::panic(
                            "end of range should be a character boundary".to_string(),
                        ));
                    }
                    text.replace_range(start..end, &replacement);
                    Ok(Value::Unit)
                })
            }
            Builtin::StringSplitOff => {
                arity(2)?;
                let at = expect_index(&vals[1], "String::split_off")?;
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_string(frames, &target, "String::split_off", |text| {
                    if !text.is_char_boundary(at) {
                        return Err(InterpError::panic(
                            "assertion failed: self.is_char_boundary(at)".to_string(),
                        ));
                    }
                    Ok(Value::Str(text.split_off(at)))
                })
            }
            Builtin::StringReserve => {
                arity(2)?;
                // Capacity is not tracked by the abstract
                // string model (see `StringCapacity`), so
                // reserving has no observable effect, exactly
                // as a `VecReserve` no-op.
                Ok(Value::Unit)
            }
            Builtin::StringTryReserve => {
                arity(2)?;
                // With no capacity to allocate, the fallible
                // reserve always succeeds.
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Unit],
                })
            }
            Builtin::StrChars => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                let bytes = s
                    .as_bytes()
                    .iter()
                    .map(|b| Value::Uint(*b as u128))
                    .collect();
                Ok(str_chars_value(bytes))
            }
            Builtin::StrMakeAsciiCase(upper) => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_string(
                    frames,
                    &target,
                    "str::make_ascii_uppercase/lowercase",
                    |text| {
                        if upper {
                            text.make_ascii_uppercase();
                        } else {
                            text.make_ascii_lowercase();
                        }
                        Ok(Value::Unit)
                    },
                )
            }
            Builtin::StrBytes => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                let bytes = s
                    .as_bytes()
                    .iter()
                    .map(|b| Value::Uint(*b as u128))
                    .collect();
                Ok(str_bytes_value(bytes))
            }
            Builtin::StrTrim(side) => {
                arity(1)?;
                let s = self.resolve_string(frames, &vals[0])?;
                let out = match side {
                    TrimSide::Both => s.trim(),
                    TrimSide::Start => s.trim_start(),
                    TrimSide::End => s.trim_end(),
                };
                Ok(Value::Str(out.to_string()))
            }
            Builtin::PanicStr => {
                arity(1)?;
                // `expect_failed` is the one `&str`-message
                // entry whose trace does not head with
                // `begin_panic::<&str>` (see the origin's doc).
                if name.ends_with("expect_failed") {
                    self.note_panic_origin(PanicOrigin::ExpectFailed);
                } else {
                    self.note_panic_origin(PanicOrigin::PanicFn);
                }
                self.note_panic_no_backtrace(false);
                // `core::panicking::panic` takes the message as a
                // string; formatted-panic shapes hand over a built
                // `fmt::Arguments` instead. The payload mirrors
                // core: a literal message is a `&str` payload, a
                // formatted one a `String`.
                match &vals[0] {
                    Value::FmtArguments(FmtArguments::Plain(text)) => {
                        Err(InterpError::panic_str_payload(text.clone()))
                    }
                    Value::FmtArguments(_) => {
                        let args = self.coerce_arguments(frames, &vals[0], name)?;
                        let message = self.format_panic_message(frames, &args)?;
                        Err(InterpError::panic_string_payload(message))
                    }
                    _ => {
                        let message = self.resolve_string(frames, &vals[0])?;
                        Err(InterpError::panic_str_payload(message))
                    }
                }
            }
            Builtin::PanicFmt => {
                arity(1)?;
                self.note_panic_origin(PanicOrigin::PanicFmt);
                self.note_panic_no_backtrace(false);
                // The payload mirrors std's creation: a
                // verbatim `Arguments` (a literal-only panic,
                // whose `as_str` is the message itself) becomes
                // a `&str` payload; a formatted panic's payload
                // is the `String` its formatting produced. A
                // hook or a catch downcasting the payload
                // observes exactly that split natively.
                match self.coerce_arguments(frames, &vals[0], name)? {
                    FmtArguments::Plain(text) => {
                        Err(InterpError::panic_str_payload(text))
                    }
                    args => {
                        let message = self.format_panic_message(frames, &args)?;
                        Err(InterpError::panic_string_payload(message))
                    }
                }
            }
            Builtin::AssertFailed => {
                arity(4)?;
                let frame = if generic_args.len() >= 2 {
                    format!(
                        "core::panicking::assert_failed::<{}, {}>",
                        generic_args[0], generic_args[1]
                    )
                } else {
                    String::new()
                };
                self.note_panic_origin(PanicOrigin::AssertFailed(frame));
                self.note_panic_no_backtrace(false);
                // (kind, &left, &right, Option<fmt::Arguments>): the
                // message mirrors core's: the comparison, the custom
                // message when one was given, and both operands.
                let kind_variant = match self.resolve_value(frames, &vals[0])? {
                    Value::Enum { variant, .. } => variant,
                    _ => u32::MAX,
                };
                let mut message = match kind_variant {
                    0 => "assertion `left == right` failed".to_string(),
                    1 => "assertion `left != right` failed".to_string(),
                    2 => "assertion `left matches right` failed".to_string(),
                    _ => "assertion failed".to_string(),
                };
                if let Value::Enum {
                    variant: 1,
                    fields,
                } = self.resolve_value(frames, &vals[3])?
                {
                    if let Some(arg) = fields.into_iter().next() {
                        if let Ok(coerced) = self.coerce_arguments(frames, &arg, name) {
                            if let Ok(text) = self.format_arguments(frames, &coerced, 0)
                            {
                                message = format!("{message}: {text}");
                            }
                        }
                    }
                }
                // The operands are `&T` / `&U` with `T, U: Debug`:
                // core renders them with `{:?}`, so the rendering
                // is the fmt engine's `Debug` under each operand's
                // own generic-argument tag -- the same rendering
                // `ResultUnwrapFailed` uses for its error. The
                // value model's own `Display` is only the fallback
                // for a shape the engine cannot render, as the
                // custom message above is best-effort too.
                let left = self.resolve_value(frames, &vals[1])?;
                let right = self.resolve_value(frames, &vals[2])?;
                let left_tag = generic_args.first().map(String::as_str).unwrap_or("");
                let right_tag = generic_args.get(1).map(String::as_str).unwrap_or("");
                let left_text = self
                    .format_value(
                        frames,
                        &left,
                        FmtTrait::Debug,
                        left_tag,
                        &FmtOptions::new(),
                        0,
                    )
                    .unwrap_or_else(|_| left.to_string());
                let right_text = self
                    .format_value(
                        frames,
                        &right,
                        FmtTrait::Debug,
                        right_tag,
                        &FmtOptions::new(),
                        0,
                    )
                    .unwrap_or_else(|_| right.to_string());
                message = format!("{message}\n  left: {left_text}\n right: {right_text}");
                Err(InterpError::panic_string_payload(message))
            }
            Builtin::HandleAllocError => {
                arity(1)?;
                self.exec_handle_alloc_error(frames, &vals[0])
            }
            Builtin::RawVecHandleError => {
                arity(1)?;
                self.exec_raw_vec_handle_error(frames, &vals[0])
            }
            Builtin::AllocErrorHookSet => {
                arity(1)?;
                self.set_alloc_error_hook(vals[0].clone());
                Ok(Value::Unit)
            }
            Builtin::AllocErrorHookTake => {
                arity(0)?;
                Ok(match self.take_alloc_error_hook() {
                    Some(hook) => hook,
                    None => Value::FnPtr(
                        "std::alloc::default_alloc_error_hook".to_string(),
                    ),
                })
            }
            Builtin::AllocErrorDefaultHook => {
                arity(1)?;
                self.exec_default_alloc_error_hook(frames, &vals[0])
            }
            Builtin::PanicNounwindFmt => {
                // `panic_nounwind_fmt(args, force_no_backtrace)`:
                // the flag is the panic's force-no-backtrace
                // (std's `panic_in_cleanup` path passes `true`),
                // which the default report honours; the panic
                // carries the formatted message, exactly as
                // `panic_fmt`'s does.
                if vals.is_empty() || vals.len() > 2 {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "1 or 2 argument(s)",
                        format!("{} argument(s)", vals.len()),
                    ));
                }
                self.note_panic_origin(PanicOrigin::PanicFmt);
                let force_no_backtrace = matches!(vals.get(1), Some(Value::Bool(true)));
                self.note_panic_no_backtrace(force_no_backtrace);
                match self.coerce_arguments(frames, &vals[0], name)? {
                    FmtArguments::Plain(text) => {
                        Err(InterpError::panic_str_payload(text))
                    }
                    args => {
                        let message = self.format_panic_message(frames, &args)?;
                        Err(InterpError::panic_string_payload(message))
                    }
                }
            }
            Builtin::OptionUnwrapFailed => {
                arity(0)?;
                self.note_panic_origin(PanicOrigin::OptionUnwrap);
                self.note_panic_no_backtrace(false);
                Err(InterpError::panic_str_payload(
                    "called `Option::unwrap()` on a `None` value",
                ))
            }
            Builtin::ResultUnwrapFailed => {
                arity(2)?;
                self.note_panic_origin(PanicOrigin::ResultUnwrap);
                self.note_panic_no_backtrace(false);
                // `unwrap_failed(msg, error: &dyn Debug)` panics
                // with "{msg}: {error:?}"; the Debug rendering is
                // the fmt engine's, the same one `panic!` uses.
                let msg = self.resolve_string(frames, &vals[0])?;
                let error = match &vals[1] {
                    Value::DynRef { .. } => {
                        let target = self.deref_value(frames, &vals[1])?;
                        self.read_resolved(frames, &target)?
                    }
                    other => self.resolve_value(frames, other)?,
                };
                let rendered = self.format_value(
                    frames,
                    &error,
                    FmtTrait::Debug,
                    "",
                    &FmtOptions::new(),
                    0,
                )?;
                Err(InterpError::panic_string_payload(format!(
                    "{msg}: {rendered}"
                )))
            }
            Builtin::FromAsciiRadixPanic => {
                arity(1)?;
                let radix = as_u128(&vals[0], name)?;
                Err(InterpError::panic_string_payload(format!(
                    "from_ascii_bytes_radix: radix must lie in the range `[2, 36]` - found {radix}"
                )))
            }
            Builtin::StrSearcherNew => {
                arity(2)?;
                let haystack = self.resolve_string(frames, &vals[0])?;
                let needle = self.resolve_string(frames, &vals[1])?;
                Ok(str_searcher_value(&haystack, &needle))
            }
            Builtin::Memchr { reverse } => {
                arity(2)?;
                let needle = match self.resolve_value(frames, &vals[0])? {
                    Value::Uint(v) => v as u8,
                    Value::Int(v) => v as u8,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "memchr needle",
                            "byte value",
                            other.kind_name(),
                        ))
                    }
                };
                let (elems, _) = self.slice_elements(frames, &vals[1])?;
                let bytes: Vec<u8> = elems
                    .iter()
                    .map(|e| match e {
                        Value::Uint(v) => Ok(*v as u8),
                        Value::Int(v) => Ok(*v as u8),
                        other => Err(InterpError::type_mismatch(
                            "memchr haystack",
                            "byte elements",
                            other.kind_name(),
                        )),
                    })
                    .collect::<InterpResult<Vec<u8>>>()?;
                let found = if reverse {
                    bytes.iter().rposition(|b| *b == needle)
                } else {
                    bytes.iter().position(|b| *b == needle)
                };
                Ok(match found {
                    Some(i) => some_value(Value::Uint(i as u128)),
                    None => none_value(),
                })
            }
            Builtin::SliceIndexFail => {
                arity(3)?;
                let start = as_u128(&vals[0], name)? as usize;
                let end = as_u128(&vals[1], name)? as usize;
                let len = as_u128(&vals[2], name)? as usize;
                Err(InterpError::panic_string_payload(
                    slice_index_fail_message(start, end, len),
                ))
            }
            Builtin::LenMismatchFail => {
                arity(2)?;
                let dst_len = as_u128(&vals[0], name)?;
                let src_len = as_u128(&vals[1], name)?;
                Err(InterpError::panic_string_payload(format!(
                    "copy_from_slice: source slice length ({src_len}) does not match destination slice length ({dst_len})"
                )))
            }
            Builtin::RangeErrorReport => {
                arity(2)?;
                // `RangeError::report(self, len)` forwards to
                // `slice_index_fail`: with a zero start for
                // `EndPastLen`, with the stored pair for
                // `StartPastEnd`.
                let len = as_u128(&vals[1], name)? as usize;
                let (start, end) = match self.resolve_value(frames, &vals[0])? {
                    Value::Enum { variant: 0, fields } if fields.len() == 1 => {
                        (0, as_u128(&fields[0], name)? as usize)
                    }
                    Value::Enum { variant: 1, fields } if fields.len() == 2 => (
                        as_u128(&fields[0], name)? as usize,
                        as_u128(&fields[1], name)? as usize,
                    ),
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "a RangeError value",
                            other.kind_name(),
                        ))
                    }
                };
                Err(InterpError::panic_string_payload(
                    slice_index_fail_message(start, end, len),
                ))
            }
            Builtin::VecRemoveAssertFailed => {
                arity(2)?;
                let index = as_u128(&vals[0], name)?;
                let len = as_u128(&vals[1], name)?;
                Err(InterpError::panic_string_payload(format!(
                    "removal index (is {index}) should be < len (is {len})"
                )))
            }
            Builtin::VecInsertAssertFailed => {
                arity(2)?;
                let index = as_u128(&vals[0], name)?;
                let len = as_u128(&vals[1], name)?;
                Err(InterpError::panic_string_payload(format!(
                    "insertion index (is {index}) should be <= len (is {len})"
                )))
            }
            Builtin::PanicForNonpositiveArgument => {
                arity(0)?;
                Err(InterpError::panic_string_payload(
                    "argument of integer logarithm must be positive".to_string(),
                ))
            }
            Builtin::PanicOnOrdViolation => {
                arity(0)?;
                Err(InterpError::panic_string_payload(
                    "user-provided comparison function does not correctly implement a total order"
                        .to_string(),
                ))
            }
            Builtin::PanicArcOverflow => {
                arity(0)?;
                Err(InterpError::panic_string_payload(
                    "Arc counter overflow".to_string(),
                ))
            }
            Builtin::PanicAlreadyBorrowed => {
                arity(1)?;
                Err(InterpError::panic_string_payload(
                    "RefCell already borrowed".to_string(),
                ))
            }
            Builtin::SliceErrorFail => {
                arity(3)?;
                let text = self.resolve_string(frames, &vals[0])?;
                let begin = as_u128(&vals[1], name)? as usize;
                let end = as_u128(&vals[2], name)? as usize;
                Err(InterpError::panic_string_payload(slice_error_fail_message(
                    &text, begin, end,
                )))
            }
            Builtin::SqrtApprox => {
                arity(1)?;
                // core's formula, in the guest's own `usize`
                // arithmetic: with `ilog = (n | 1).ilog2()` and
                // `shift = ceil(ilog / 2)`, the approximation is
                // `((1 << shift) + (n >> shift)) / 2`.
                let n = as_u128(&vals[0], name)? as u64;
                let ilog = (n | 1).ilog2();
                let shift = ilog.div_ceil(2);
                let approx = ((1u64 << shift) + (n >> shift)) / 2;
                Ok(Value::Uint(approx as u128))
            }
            Builtin::SpinPause => {
                arity(0)?;
                Ok(Value::Unit)
            }
            Builtin::ExactDiv => {
                arity(2)?;
                match (&vals[0], &vals[1]) {
                    (Value::Int(a), Value::Int(b)) => {
                        if *b == 0 {
                            Err(InterpError::panic("attempt to divide by zero"))
                        } else {
                            Ok(Value::Int(a / b))
                        }
                    }
                    (Value::Uint(a), Value::Uint(b)) => {
                        if *b == 0 {
                            Err(InterpError::panic("attempt to divide by zero"))
                        } else {
                            Ok(Value::Uint(a / b))
                        }
                    }
                    other => Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "two integers of one signedness",
                        other.0.kind_name(),
                    )),
                }
            }
            Builtin::CatchUnwind => {
                arity(3)?;
                let try_fn = expect_fn_ptr(&vals[0], name)?;
                let catch_fn = expect_fn_ptr(&vals[2], name)?;
                let panicked =
                    self.exec_catch_unwind(frames, &try_fn, vals[1].clone(), &catch_fn)?;
                Ok(Value::Bool(panicked))
            }
            Builtin::PanicCleanup => {
                arity(1)?;
                match &vals[0] {
                    Value::PanicPayload(payload) => Ok(Value::DynRef {
                        // The payload travels in the aggregate
                        // box shape a sized box value has, so the
                        // data pointer has the shape `Value::DynRef`
                        // documents for a `Box` coercion.
                        base: Box::new(Value::Enum {
                            variant: 0,
                            fields: vec![payload.value.as_ref().clone()],
                        }),
                        concrete: payload.ty.clone(),
                        traits: vec![
                            "std::any::Any".to_string(),
                            "std::marker::Send".to_string(),
                        ],
                        // Fabricated by the interpreter, not
                        // formed by an unsizing cast: the glue
                        // it carries is the payload type's own,
                        // from the program's payload table (see
                        // `ir::Program::payload_drop_glues`); a
                        // payload type with no entry keeps the
                        // shape-judged drop (see `DropPlan::Dyn`).
                        drop_glue: self
                            .payload_drop_glue(&payload.ty)
                            .map(Box::new),
                        vtable_methods: Vec::new(),
                        type_id: payload.type_id,
                    }),
                    other => Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "a raw panic payload",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::Panicking => {
                arity(0)?;
                Ok(Value::Bool(self.panicking()))
            }
            Builtin::PanicCountIsZero => {
                arity(0)?;
                Ok(Value::Bool(!self.panicking()))
            }
            Builtin::PanicAny => {
                arity(1)?;
                let ty = generic_args.first().cloned().unwrap_or_default();
                // This kind serves `std::panic::panic_any` and
                // the `begin_panic` entry points alike; only an
                // explicit `panic_any` call shows a `panic_any`
                // frame under `begin_panic` in a native trace.
                if name == "std::panic::panic_any" {
                    self.note_panic_origin(PanicOrigin::PanicAny(ty.clone()));
                } else {
                    self.note_panic_origin(PanicOrigin::BeginPanic(ty.clone()));
                }
                self.note_panic_no_backtrace(false);
                let value = vals.into_iter().next().expect("arity checked");
                // The message the default hook reports for the
                // payload: a string payload reports as itself,
                // anything else as the opaque box (std's
                // `payload_as_str`).
                let message = match &value {
                    Value::Str(text) => text.clone(),
                    _ if ty == "&str" => self
                        .resolve_string(frames, &value)
                        .unwrap_or_else(|_| "Box<dyn Any>".to_string()),
                    _ => "Box<dyn Any>".to_string(),
                };
                Err(InterpError::panic_with_payload(
                    message,
                    PanicPayload {
                        ty,
                        value: Box::new(value),
                        // The payload type is this call's first
                        // generic argument: its segments are the
                        // call site's own recording of that type.
                        type_id: generic_type_ids
                            .and_then(|ids| ids.first().copied()),
                    },
                ))
            }
            Builtin::ResumeUnwind => {
                arity(1)?;
                // std's `resume_unwind` runs the same
                // must-abort checks as a fresh panic, in the
                // same order (the always-abort flag first, then
                // the in-hook state), each with its own line.
                if self.always_abort() {
                    self.push_error_output("aborting due to panic\n");
                    return Err(InterpError::Abort {
                        message:
                            "`resume_unwind` after `panic::always_abort`: the run aborts"
                                .to_string(),
                    });
                }
                if self.in_panic_hook() {
                    self.push_error_output(
                        "thread panicked while processing panic. aborting.\n",
                    );
                    return Err(InterpError::Abort {
                        message:
                            "`resume_unwind` while the panic hook was running: \
                             the run aborts"
                                .to_string(),
                    });
                }
                // A resumed panic is not a new panic: the hook
                // (already fired at the original raise) must not
                // fire again, which the unwinder's resumed flag
                // records.
                self.note_resumed_panic();
                let (concrete, type_id, value, _) =
                    resolve_dyn_box(self, frames, &vals[0], name)?;
                // The re-raised payload is the pointee the dyn
                // value's base denotes (the box's contents), not
                // the box itself: the cleanup that later turns a
                // payload back into a dyn value wraps it in a box
                // again, so handing it the box would double it.
                let value = dyn_box_pointee(self, frames, &value)?;
                let message = match &value {
                    Value::Str(text) => text.clone(),
                    _ => "Box<dyn Any>".to_string(),
                };
                Err(InterpError::panic_with_payload(
                    message,
                    PanicPayload {
                        ty: concrete,
                        value: Box::new(value),
                        type_id,
                    },
                ))
            }
            Builtin::DynAnyDowncastRef => {
                arity(1)?;
                let want = expect_generic_ty(generic_args, name)?;
                let (concrete, type_id, value, _) =
                    resolve_dyn_box(self, frames, &vals[0], name)?;
                if dyn_any_matches(
                    &concrete,
                    type_id,
                    &want,
                    generic_type_ids.and_then(|ids| ids.first().copied()),
                ) {
                    // The result denotes the payload the dyn
                    // value's base points at (the box's pointee),
                    // not the box itself.
                    let value = dyn_box_pointee(self, frames, &value)?;
                    Ok(Value::Enum {
                        variant: 1,
                        fields: vec![Value::Ref(RefTarget::value(value))],
                    })
                } else {
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![],
                    })
                }
            }
            Builtin::DynAnyDowncast(t_index) => {
                arity(1)?;
                let want = expect_generic_ty_at(generic_args, name, t_index)?;
                let (concrete, type_id, value, original) =
                    resolve_dyn_box(self, frames, &vals[0], name)?;
                if dyn_any_matches(
                    &concrete,
                    type_id,
                    &want,
                    generic_type_ids.and_then(|ids| ids.get(t_index).copied()),
                ) {
                    // `Ok` holds a `Box<T>`: the dyn value's base
                    // itself, which is the box in the shape the
                    // sized value had (see `Value::DynRef`) — an
                    // allocation-backed box from an unsizing cast,
                    // or the aggregate box a fabricated payload
                    // travels in. Wrapping it once more would hand
                    // back a box whose field is another box, and
                    // writes through the result's `Deref` could not
                    // reach the payload.
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![value],
                    })
                } else {
                    Ok(Value::Enum {
                        variant: 1,
                        fields: vec![original],
                    })
                }
            }
            Builtin::DynAnyIs => {
                arity(1)?;
                let want = expect_generic_ty(generic_args, name)?;
                let (concrete, type_id, _, _) =
                    resolve_dyn_box(self, frames, &vals[0], name)?;
                Ok(Value::Bool(dyn_any_matches(
                    &concrete,
                    type_id,
                    &want,
                    generic_type_ids.and_then(|ids| ids.first().copied()),
                )))
            }
            Builtin::PartialEqEq | Builtin::PartialEqNe => {
                arity(2)?;
                // std defines `Path` / `PathBuf` equality over
                // the parsed components (repeated and trailing
                // separators and interior `.` components are
                // normalised away), not over the raw bytes the
                // structural comparison below would use. The
                // callee name carries no impl subject in this
                // model, so the path types are recognised by the
                // call's own generic arguments (the `Self` and
                // `Rhs` type tags).
                let path_cmp = self.comparison_is_path(frames, generic_args, &vals);
                let eq = if path_cmp {
                    let lhs = self.path_cmp_bytes(frames, &vals[0])?;
                    let rhs = self.path_cmp_bytes(frames, &vals[1])?;
                    path_components(&lhs) == path_components(&rhs)
                } else if let Some(eq) =
                    self.raw_pointer_equal(frames, generic_args, &vals[0], &vals[1])
                {
                    eq
                } else if let Some(eq) = self.regrouped_slice_equal(
                    frames,
                    generic_args,
                    &vals[0],
                    &vals[1],
                ) {
                    eq?
                } else if let Some(eq) = self.punned_tuple_equal(
                    frames,
                    generic_args,
                    &vals[0],
                    &vals[1],
                ) {
                    eq?
                } else if let Some(eq) = self.punned_scalar_equal(
                    frames,
                    generic_args,
                    &vals[0],
                    &vals[1],
                ) {
                    eq?
                } else if let Some(eq) = self.box_pointee_equal(
                    frames,
                    generic_args,
                    &vals[0],
                    &vals[1],
                ) {
                    eq?
                } else {
                    self.values_equal(frames, &vals[0], &vals[1])?
                };
                Ok(Value::Bool(if kind == Builtin::PartialEqNe {
                    !eq
                } else {
                    eq
                }))
            }
            Builtin::PartialCmp => {
                arity(2)?;
                let ord = if self.comparison_is_path(frames, generic_args, &vals) {
                    let lhs = self.path_cmp_bytes(frames, &vals[0])?;
                    let rhs = self.path_cmp_bytes(frames, &vals[1])?;
                    Some(path_ordering(&lhs, &rhs))
                } else if let Some(ord) =
                    self.raw_pointer_cmp(frames, generic_args, &vals[0], &vals[1])
                {
                    ord
                } else if let Some(ord) = self.regrouped_slice_cmp(
                    frames,
                    generic_args,
                    &vals[0],
                    &vals[1],
                    cmp_discriminants,
                ) {
                    ord?
                } else if let Some(ord) = self.box_pointee_cmp(
                    frames,
                    generic_args,
                    &vals[0],
                    &vals[1],
                ) {
                    ord?
                } else if let Some(ord) = self.punned_tuple_cmp(
                    frames,
                    generic_args,
                    &vals[0],
                    &vals[1],
                    cmp_discriminants,
                ) {
                    ord?
                } else if let Some(ord) = self.punned_scalar_cmp(
                    frames,
                    generic_args,
                    &vals[0],
                    &vals[1],
                    cmp_discriminants,
                ) {
                    ord?
                } else {
                    self.values_cmp(frames, &vals[0], &vals[1], cmp_discriminants)?
                };
                Ok(match ord {
                    Some(ord) => Value::Enum {
                        variant: 1,
                        fields: vec![ordering_value(ord)],
                    },
                    None => Value::Enum {
                        variant: 0,
                        fields: vec![],
                    },
                })
            }
            Builtin::OrdCmp => {
                arity(2)?;
                let ord = if self.comparison_is_path(frames, generic_args, &vals) {
                    let lhs = self.path_cmp_bytes(frames, &vals[0])?;
                    let rhs = self.path_cmp_bytes(frames, &vals[1])?;
                    Some(path_ordering(&lhs, &rhs))
                } else if let Some(ord) =
                    self.raw_pointer_cmp(frames, generic_args, &vals[0], &vals[1])
                {
                    ord
                } else if let Some(ord) = self.regrouped_slice_cmp(
                    frames,
                    generic_args,
                    &vals[0],
                    &vals[1],
                    cmp_discriminants,
                ) {
                    ord?
                } else if let Some(ord) = self.box_pointee_cmp(
                    frames,
                    generic_args,
                    &vals[0],
                    &vals[1],
                ) {
                    ord?
                } else if let Some(ord) = self.punned_tuple_cmp(
                    frames,
                    generic_args,
                    &vals[0],
                    &vals[1],
                    cmp_discriminants,
                ) {
                    ord?
                } else if let Some(ord) = self.punned_scalar_cmp(
                    frames,
                    generic_args,
                    &vals[0],
                    &vals[1],
                    cmp_discriminants,
                ) {
                    ord?
                } else {
                    self.values_cmp(frames, &vals[0], &vals[1], cmp_discriminants)?
                };
                match ord {
                    Some(ord) => Ok(ordering_value(ord)),
                    // `Ord` is a total ordering: an unordered pair
                    // (a NaN-carrying value) cannot reach a genuine
                    // `Ord` implementation, so this stays explicit.
                    None => Err(InterpError::type_mismatch(
                        "Ord::cmp",
                        "a total ordering",
                        "an unordered value (NaN comparison)",
                    )),
                }
            }
            Builtin::PartialOrdRel(op) => {
                arity(2)?;
                let ord = if self.comparison_is_path(frames, generic_args, &vals) {
                    let lhs = self.path_cmp_bytes(frames, &vals[0])?;
                    let rhs = self.path_cmp_bytes(frames, &vals[1])?;
                    Some(path_ordering(&lhs, &rhs))
                } else if let Some(ord) =
                    self.raw_pointer_cmp(frames, generic_args, &vals[0], &vals[1])
                {
                    ord
                } else if let Some(ord) = self.regrouped_slice_cmp(
                    frames,
                    generic_args,
                    &vals[0],
                    &vals[1],
                    cmp_discriminants,
                ) {
                    ord?
                } else if let Some(ord) = self.box_pointee_cmp(
                    frames,
                    generic_args,
                    &vals[0],
                    &vals[1],
                ) {
                    ord?
                } else if let Some(ord) = self.punned_tuple_cmp(
                    frames,
                    generic_args,
                    &vals[0],
                    &vals[1],
                    cmp_discriminants,
                ) {
                    ord?
                } else if let Some(ord) = self.punned_scalar_cmp(
                    frames,
                    generic_args,
                    &vals[0],
                    &vals[1],
                    cmp_discriminants,
                ) {
                    ord?
                } else {
                    self.values_cmp(frames, &vals[0], &vals[1], cmp_discriminants)?
                };
                // An unordered pair makes every relational operator
                // false, as `partial_cmp` returning `None` does.
                Ok(Value::Bool(match (ord, op) {
                    (Some(Ordering::Less), RelOp::Lt) => true,
                    (Some(Ordering::Greater), RelOp::Gt) => true,
                    (Some(Ordering::Less | Ordering::Equal), RelOp::Le) => true,
                    (Some(Ordering::Greater | Ordering::Equal), RelOp::Ge) => true,
                    _ => false,
                }))
            }
            Builtin::WakerWake => {
                arity(1)?;
                let value = vals.into_iter().next().expect("arity checked");
                self.exec_waker_wake(frames, value, false, name)
            }
            Builtin::WakerWakeByRef => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                let value = self.read_resolved(frames, &target)?;
                self.exec_waker_wake(frames, value, true, name)
            }
            Builtin::CloneClone => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                self.clone_modelled_value(frames, self_ty, name, &vals[0], receiver_glue)
            }
            Builtin::DefaultDefault => {
                arity(0)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                match self.default_value(tag) {
                    Some(value) => Ok(value),
                    None => Err(unsupported(format!(
                        "Default::default for {tag} is not runtime-supported"
                    ))),
                }
            }
            Builtin::ToStringToString => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                match self.resolve_value(frames, &vals[0])? {
                    Value::Str(s) => Ok(Value::Str(s)),
                    Value::Bool(b) => Ok(Value::Str(b.to_string())),
                    Value::Int(v) => Ok(Value::Str(v.to_string())),
                    Value::Uint(v) if self_ty == "char" => {
                        match u32::try_from(v).ok().and_then(char::from_u32) {
                            Some(c) => Ok(Value::Str(c.to_string())),
                            None => Err(unsupported(
                                "ToString::to_string for an invalid char value",
                            )),
                        }
                    }
                    Value::Uint(v) => Ok(Value::Str(v.to_string())),
                    // `Arguments` implements `Display` (and so
                    // `ToString`) by formatting itself — exactly
                    // the `format_arguments` model.
                    Value::FmtArguments(ref args) => {
                        Ok(Value::Str(self.format_arguments(frames, args, 0)?))
                    }
                    ref other => {
                        // Every remaining shape: `to_string` is
                        // the receiver's `Display` formatting with
                        // default options, which the fmt engine
                        // serves for any value that has a `Display`
                        // impl — a trait object's is its concrete
                        // type's, a guest ADT's is its collected
                        // body, a float's is its decimal form.
                        let text = self.format_value(
                            frames,
                            other,
                            FmtTrait::Display,
                            self_ty,
                            &FmtOptions::new(),
                            0,
                        )?;
                        Ok(Value::Str(text))
                    }
                }
            }
            Builtin::BoxNew => {
                arity(1)?;
                // A box's pointee lives in the allocation store;
                // the box value is the allocation pointer (see
                // `Interpreter::new_box`). With a guest allocator
                // the storage comes from its shim, as the native
                // `__rust_alloc` call inside `Box::new` binds to
                // it; a zero-sized pointee allocates nothing in
                // either world.
                let v = vals.into_iter().next().expect("arity checked");
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let pointee_ty = ty_from_tag(tag);
                if let Some((size, align)) = self.layout_of(tag) {
                    if size > 0 {
                        if let Some(ptr) =
                            self.call_guest_alloc(frames, size, align)?
                        {
                            if matches!(ptr, Value::Ref(RefTarget::Null)) {
                                return Err(unsupported(
                                    "guest allocator returned null for Box::new",
                                ));
                            }
                            let target = self.deref_value(frames, &ptr)?;
                            self.write_resolved(frames, &target, v)?;
                            let boxed = Value::Enum {
                                variant: 0,
                                fields: vec![ptr],
                            };
                            self.record_box_layout(&boxed, size, align);
                            if let Some(ty) = &pointee_ty {
                                self.record_box_ty(&boxed, ty.clone());
                            }
                            return Ok(boxed);
                        }
                    }
                }
                let boxed = self.new_box(v);
                if let Some(ty) = pointee_ty {
                    self.record_box_ty(&boxed, ty);
                }
                Ok(boxed)
            }
            Builtin::ManuallyDropNew => {
                arity(1)?;
                // A one-field ADT aggregate holding its value.
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![vals.into_iter().next().expect("arity checked")],
                })
            }
            Builtin::RcNew | Builtin::ArcNew => {
                arity(1)?;
                // A fresh reference-counted allocation in the
                // store: the handle is the `NonNull` pointer to the
                // stored `RcInner` / `ArcInner`, so the converted
                // std bodies (`get_mut`, `make_mut`, the count
                // accessors, `Weak`) operate on the real layout.
                let v = vals.into_iter().next().expect("arity checked");
                Ok(self.rc_new_handle(v))
            }
            Builtin::RcFromRaw | Builtin::ArcFromRaw | Builtin::WeakFromRaw => {
                arity(1)?;
                // Rebuild the handle naming the allocation the
                // raw pointer denotes. The pointer `into_raw`
                // produces (and the only shape this reconstruction
                // accepts) is the stored inner aggregate's data
                // pointer: a reference to the allocation carrying
                // the data field's projection (`Field(2)`, the
                // position `Deref::deref` reads through), or the
                // bare allocation reference itself. The handle is
                // the allocation's canonical handle value; no
                // count moves, as `into_raw` forgot a live
                // handle rather than releasing one. Any other
                // pointer shape denotes no allocation this model
                // can name, and is an explicit error.
                // The argument is the pointer value itself (a
                // reference into the allocation store), not a
                // place denoting it: it is matched as it stands.
                let ptr = vals.into_iter().next().expect("arity checked");
                let id = match &ptr {
                    Value::Ref(RefTarget::Alloc { id, projection })
                        if projection.is_empty()
                            || (projection.last()
                                == Some(&RefProj::Field(2))
                                && projection[..projection.len() - 1]
                                    .iter()
                                    .all(|s| matches!(
                                        s,
                                        RefProj::Index(0)
                                            | RefProj::Field(0)
                                            | RefProj::ByteOffset(0)
                                    ))) =>
                    {
                        Some(*id)
                    }
                    _ => None,
                };
                match id {
                    Some(id) => Ok(self.rc_handle(id)),
                    None => Err(InterpError::type_mismatch(
                        "Rc::from_raw / Arc::from_raw / Weak::from_raw",
                        "a raw pointer to a reference-counted allocation's data field",
                        ptr.kind_name(),
                    )),
                }
            }
            Builtin::BoxNewUninit => {
                arity(0)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                // The tag names the *inner* type (the `T` of
                // `Box<MaybeUninit<T>>`); the allocation's
                // declared type is the union itself, which is
                // what the record must carry -- recording the
                // inner type would make the `MaybeUninit` entry
                // gate refuse the wrapper's own positions.
                let pointee_ty = Some(Ty::Adt(
                    "core::mem::MaybeUninit".to_string(),
                ));
                match uninit_value_for_tag(tag) {
                    Some(pointee) => {
                        if let Some((size, align)) = self.layout_of(tag) {
                            if size > 0 {
                                if let Some(ptr) = self
                                    .call_guest_alloc(frames, size, align)?
                                {
                                    if matches!(
                                        ptr,
                                        Value::Ref(RefTarget::Null)
                                    ) {
                                        return Err(unsupported(
                                            "guest allocator returned null \
                                             for Box::new_uninit",
                                        ));
                                    }
                                    let target =
                                        self.deref_value(frames, &ptr)?;
                                    self.write_resolved(
                                        frames,
                                        &target,
                                        pointee,
                                    )?;
                                    let boxed = Value::Enum {
                                        variant: 0,
                                        fields: vec![ptr],
                                    };
                                    self.record_box_layout(
                                        &boxed, size, align,
                                    );
                                    if let Some(ty) = &pointee_ty {
                                        self.record_box_ty(&boxed, ty.clone());
                                    }
                                    return Ok(boxed);
                                }
                            }
                        }
                        let boxed = self.new_box(pointee);
                        if let Some(ty) = pointee_ty {
                            self.record_box_ty(&boxed, ty);
                        }
                        Ok(boxed)
                    }
                    None => Err(unsupported(format!(
                        "Box::new_uninit for {tag} is not runtime-supported"
                    ))),
                }
            }
            Builtin::BoxAssumeInitIntoVec => {
                arity(1)?;
                let arg = vals.into_iter().next().expect("arity checked");
                let boxed = match &arg {
                    Value::Ref(_) | Value::FatRef { .. } => {
                        let target = self.deref_value(frames, &arg)?;
                        self.read_resolved(frames, &target)?
                    }
                    _ => arg,
                };
                match boxed {
                    Value::Enum { fields, .. } if fields.len() == 1 => {
                        let pointee = fields.into_iter().next().expect("one field");
                        // A built box's field is its allocation
                        // pointer: the array lives behind it in
                        // the allocation store.
                        let pointee = match &pointee {
                            Value::Ref(RefTarget::Alloc { .. }) => {
                                let target = self.deref_value(frames, &pointee)?;
                                self.read_resolved(frames, &target)?
                            }
                            _ => pointee,
                        };
                        match assume_init_elements(pointee) {
                            Some(elems) => Ok(Value::Enum {
                                variant: 0,
                                fields: match unique_elem {
                                    Some(canonical) => elems
                                        .into_iter()
                                        .map(|elem| fill_uninit(elem, canonical))
                                        .collect(),
                                    None => elems,
                                },
                            }),
                            None => Err(InterpError::type_mismatch(
                                "box_assume_init_into_vec_unsafe",
                                "box of an array value",
                                "box of a non-array value",
                            )),
                        }
                    }
                    ref other => Err(InterpError::type_mismatch(
                        "box_assume_init_into_vec_unsafe",
                        "box value",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::VecNew => {
                arity(0)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                })
            }
            Builtin::VecExtendFromSlice => {
                arity(2)?;
                let target = self.deref_value(frames, &vals[0])?;
                let (elems, _) = self.slice_elements(frames, &vals[1])?;
                let elem_tag = generic_args.first().map(String::as_str).unwrap_or("");
                let uninit_elem = uninit_elem_value_for_tag(elem_tag);
                let elem_size = self.layout_of(elem_tag).map_or(8, |(size, _)| size as usize);
                self.vec_append_resolved(
                    frames,
                    &target,
                    &elems,
                    "Vec::extend_from_slice",
                    &uninit_elem,
                    elem_size,
                )?;
                Ok(Value::Unit)
            }
            Builtin::VecPush => {
                arity(2)?;
                let target = self.deref_value(frames, &vals[0])?;
                let mut vals = vals.into_iter();
                let _ = vals.next();
                let elem = vals.next().expect("arity checked");
                let elem_tag = generic_args.first().map(String::as_str).unwrap_or("");
                let uninit_elem = uninit_elem_value_for_tag(elem_tag);
                let elem_size = self.layout_of(elem_tag).map_or(8, |(size, _)| size as usize);
                self.vec_append_resolved(
                    frames,
                    &target,
                    &[elem],
                    "Vec::push",
                    &uninit_elem,
                    elem_size,
                )?;
                Ok(Value::Unit)
            }
            Builtin::VecLen | Builtin::VecIsEmpty => {
                arity(1)?;
                // Only the length is read: resolve by borrowing,
                // so the query does not clone every element of the
                // vector the way the owned resolution did.
                let value = self.resolve_ref(frames, &vals[0])?;
                let len = value.len().ok_or_else(|| {
                    InterpError::type_mismatch(
                        "Vec length",
                        "vector value",
                        value.kind_name(),
                    )
                })?;
                if kind == Builtin::VecLen {
                    Ok(Value::Uint(len))
                } else {
                    Ok(Value::Bool(len == 0))
                }
            }
            Builtin::VecWithCapacity => {
                // One argument (`with_capacity`) or two
                // (`with_capacity_in`: capacity and the allocator,
                // a zero-sized handle here). The requested
                // capacity is the vector's initial storage extent
                // (see `vec_stored_len`): the result is the empty
                // vector holding that many uninitialised slots,
                // so growth within the capacity does not happen,
                // exactly as the real allocation behaves. A
                // capacity past the materialisation limit cannot
                // be held element by element: the sparse form
                // accounts it as its `cap` over an empty stored
                // prefix instead, so the capacity the vector
                // reports — and the spare-capacity room guest
                // code relies on — is the requested extent, not
                // zero.
                if vals.len() != 1 && vals.len() != 2 {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "1 or 2 argument(s)",
                        format!("{} argument(s)", vals.len()),
                    ));
                }
                let cap = match &vals[0] {
                    Value::Uint(n) => usize::try_from(*n).unwrap_or(usize::MAX),
                    Value::Int(n) if *n >= 0 => {
                        usize::try_from(*n).unwrap_or(usize::MAX)
                    }
                    _ => 0,
                };
                if cap == 0 {
                    return Ok(Value::Enum {
                        variant: 0,
                        fields: Vec::new(),
                    });
                }
                if cap > MAX_VEC_SET_LEN_ELEMENTS {
                    return Ok(Value::SparseVec {
                        elems: Vec::new(),
                        len: 0,
                        cap: cap as u128,
                    });
                }
                Ok(vec_with_len(
                    vec![
                        uninit_elem_value_for_tag(
                            generic_args.first().map(String::as_str).unwrap_or(""),
                        );
                        cap
                    ],
                    0,
                    &Value::Uninit,
                    cap as u128,
                ))
            }
            Builtin::VecFromElem => {
                arity(2)?;
                let mut vals = vals.into_iter();
                let elem = vals.next().expect("arity checked");
                let count = match vals.next().expect("arity checked") {
                    Value::Uint(n) => n,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            "std::vec::from_elem",
                            "element count (usize)",
                            other.kind_name(),
                        ));
                    }
                };
                // Mirrors the repeat-aggregate resource bound in
                // mir_convert: past it the vector is not
                // materialised, so a huge count fails explicitly
                // instead of exhausting the host.
                const MAX_FROM_ELEM: u128 = 1_000_000;
                if count > MAX_FROM_ELEM {
                    return Err(unsupported(format!(
                        "std::vec::from_elem with {count} elements exceeds the model's {MAX_FROM_ELEM}-element bound"
                    )));
                }
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![elem; count as usize],
                })
            }
            Builtin::VecReserve => {
                arity(2)?;
                let additional = match &vals[1] {
                    Value::Uint(n) => usize::try_from(*n).unwrap_or(usize::MAX),
                    // A `usize` argument whose high bit is set
                    // materialises as a sign-extended signed
                    // value; its bit pattern is the count.
                    Value::Int(v) => (*v as u64) as usize,
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "additional capacity (usize)",
                            other.kind_name(),
                        ));
                    }
                };
                let elem_tag = generic_args.first().map(String::as_str).unwrap_or("");
                let uninit_elem = uninit_elem_value_for_tag(elem_tag);
                let elem_size = self.layout_of(elem_tag).map_or(8, |(size, _)| size as usize);
                let target = self.deref_value(frames, &vals[0])?;
                let exact = name.ends_with("reserve_exact");
                // A reserve on a string's vector place is
                // the entry of a body-level raw-parts walk
                // over the string: prepare the form the walk
                // operates on at the target first (see
                // `materialise_resolved_string`), because the
                // modification below lends the slot under the
                // allocation store's borrow when the string
                // lives inside an allocation, where the
                // preparation's own allocation cannot run.
                self.materialise_resolved_string(frames, &target)?;
                // A target holding the real aggregate form
                // grows on a detached copy (see
                // `vec_reserve_real_form`): growth can
                // materialise a storage-less buffer, which
                // the modify path below cannot allocate
                // while it lends the slot.
                if self.vec_reserve_real_form(
                    frames,
                    &target,
                    additional,
                    exact,
                    &uninit_elem,
                    elem_size,
                )? {
                    return Ok(Value::Unit);
                }
                self.modify_resolved(frames, &target, |slot| {
                    // The walk operates on the vector slot
                    // the raw-parts entry prepared (see
                    // `raw_parts_vec_slot`).
                    let slot = self.raw_parts_vec_slot(slot);
                    vec_ensure_capacity(self, slot, additional, exact, &uninit_elem, elem_size);
                    Ok(Value::Unit)
                })
            }
            Builtin::VecTryReserve => {
                arity(2)?;
                let additional = match vals[1] {
                    Value::Uint(n) => n,
                    // A `usize` argument whose high bit is set (e.g.
                    // `usize::MAX`) materialises as a sign-extended
                    // signed value; its bit pattern is the count.
                    Value::Int(v) => (v as u64) as u128,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "additional capacity (usize)",
                            other.kind_name(),
                        ));
                    }
                };
                // Only the element count is read: borrow the
                // receiver instead of cloning every element.
                let resolved = self.resolve_ref(frames, &vals[0])?;
                let len = value_elements(&resolved)?.len() as u128;
                let elem_tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (elem_size, _) = type_layout(elem_tag).ok_or_else(|| {
                    unsupported(format!(
                        "{name}: element type {elem_tag} has no layout known to the                          value model, so the capacity-overflow check cannot be decided"
                    ))
                })?;
                // The std rule (RawVec's capacity-overflow check):
                // the byte size `required * elem_size` must not
                // exceed `isize::MAX`; a zero-sized element is
                // bounded only by `usize::MAX` elements. Allocation
                // failure, the other `TryReserveError` source, has
                // no abstract counterpart.
                let required = len + additional;
                let overflow = if elem_size == 0 {
                    required > usize::MAX as u128
                } else {
                    required > (isize::MAX as u128) / elem_size
                };
                if overflow {
                    // Err(TryReserveError::CapacityOverflow)
                    Ok(Value::Enum {
                        variant: 1,
                        fields: vec![Value::Enum {
                            variant: 0,
                            fields: Vec::new(),
                        }],
                    })
                } else {
                    // Success grows the storage as the
                    // corresponding `reserve` spelling does.
                    let uninit_elem = uninit_elem_value_for_tag(elem_tag);
                    let elem_size =
                        usize::try_from(elem_size).unwrap_or(usize::MAX);
                    let additional =
                        usize::try_from(additional).unwrap_or(usize::MAX);
                    let target = self.deref_value(frames, &vals[0])?;
                    let exact = name.ends_with("try_reserve_exact");
                    // As in `VecReserve` above: prepare the
                    // raw-parts form at the target first.
                    self.materialise_resolved_string(frames, &target)?;
                    // As in `VecReserve` above: a real-form
                    // target grows on a detached copy.
                    if !self.vec_reserve_real_form(
                        frames,
                        &target,
                        additional,
                        exact,
                        &uninit_elem,
                        elem_size,
                    )? {
                        self.modify_resolved(frames, &target, |slot| {
                            // As in `VecReserve` above: a reserve
                            // on a string's vector place opens a
                            // body-level raw-parts walk.
                            let slot = self.raw_parts_vec_slot(slot);
                            vec_ensure_capacity(
                                self,
                                slot,
                                additional,
                                exact,
                                &uninit_elem,
                                elem_size,
                            );
                            Ok(Value::Unit)
                        })?;
                    }
                    Ok(Value::Enum {
                        variant: 0,
                        fields: vec![Value::Unit],
                    })
                }
            }
            Builtin::VecShrink => {
                // `shrink_to_fit` passes one argument, `shrink_to`
                // two (receiver, minimum capacity). The shrink
                // drops the spare storage past the target capacity
                // — the storage a length shrink left behind (see
                // `vec_with_len`) — and leaves the length and its
                // elements untouched.
                if vals.len() != 1 && vals.len() != 2 {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "1 or 2 argument(s)",
                        format!("{} argument(s)", vals.len()),
                    ));
                }
                let min_cap = if vals.len() == 2 {
                    match vals[1] {
                        Value::Uint(n) => n,
                        Value::Int(n) if n >= 0 => n as u128,
                        ref other => {
                            return Err(InterpError::type_mismatch(
                                format!("builtin {name}"),
                                "minimum capacity (usize)",
                                other.kind_name(),
                            ));
                        }
                    }
                } else {
                    0
                };
                let target = self.deref_value(frames, &vals[0])?;
                self.modify_resolved(frames, &target, |slot| {
                    let mut rebuild = None;
                    if let Value::SparseVec { elems, len, cap } = slot {
                        // The shrink drops the spare storage past
                        // the target capacity: stored slots and
                        // the accounted extent alike (a shrink
                        // never grows the account, whatever
                        // minimum the caller names).
                        let keep = (*len).max(min_cap);
                        if (elems.len() as u128) > keep {
                            elems.truncate(keep as usize);
                        }
                        if keep == *len && (elems.len() as u128) == keep {
                            rebuild = Some(Value::Enum {
                                variant: 0,
                                fields: std::mem::take(elems),
                            });
                        } else {
                            *cap = keep.min(*cap).max(elems.len() as u128);
                        }
                    }
                    if let Some(value) = rebuild {
                        *slot = value;
                    }
                    Ok(Value::Unit)
                })
            }
            Builtin::VecSetLen => {
                arity(2)?;
                let new_len = match vals[1] {
                    Value::Uint(n) => n as usize,
                    Value::Int(n) if n >= 0 => n as usize,
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            "Vec::set_len",
                            "unsigned length",
                            other.kind_name(),
                        ));
                    }
                };
                let target = self.deref_value(frames, &vals[0])?;
                let uninit_elem = uninit_elem_value_for_tag(
                    generic_args.first().map(String::as_str).unwrap_or(""),
                );
                self.modify_resolved(frames, &target, |slot| {
                    // The vector's value at the new length (see
                    // `vec_with_len`): materialised up to the
                    // materialisation limit, sparse past it, so
                    // an astronomical length (the
                    // `Vec::with_capacity(1_000_000_000)` +
                    // `set_len` idiom) declares its length
                    // instead of exhausting the host. Shrinking
                    // keeps the tail elements in storage without
                    // dropping them, exactly as the real `set_len`
                    // contract specifies.
                    let prev_cap = vec_stored_len(slot).unwrap_or(0);
                    let elems = match std::mem::replace(slot, Value::Uninit) {
                        Value::Enum { fields, .. } => fields,
                        Value::SparseVec { elems, .. } => elems,
                        other => {
                            return Err(InterpError::type_mismatch(
                                "Vec::set_len",
                                "vector value",
                                other.kind_name(),
                            ));
                        }
                    };
                    *slot = vec_with_len(elems, new_len, &uninit_elem, prev_cap);
                    Ok(Value::Unit)
                })
            }
            Builtin::VecAsPtr => {
                arity(1)?;
                // The receiver reference retargeted at the first
                // element: the pointer every buffer consumer
                // (offsets, copies, reads) resolves against the
                // vector's element aggregate.
                self.ptr_first_elem_value(frames, body, vals[0].clone(), None, None)
            }
            Builtin::VecExtendTrusted => {
                arity(2)?;
                // The callee's generic arguments are the Vec's own
                // parameters followed by the iterator type, which
                // is the stepping dispatch key.
                let iter_ty = generic_args.last().map(String::as_str).unwrap_or("");
                let target = self.deref_value(frames, &vals[0])?;
                match self.read_resolved(frames, &target)? {
                    Value::Enum { .. } | Value::SparseVec { .. } => {}
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            "Vec::extend_trusted",
                            "vector value",
                            other.kind_name(),
                        ))
                    }
                }
                let (items, _) = self.drain_iterator(
                    frames,
                    iter_ty,
                    vals.into_iter().nth(1).expect("arity checked"),
                )?;
                let elem_tag = generic_args.first().map(String::as_str).unwrap_or("");
                let uninit_elem = uninit_elem_value_for_tag(elem_tag);
                let elem_size = self.layout_of(elem_tag).map_or(8, |(size, _)| size as usize);
                self.vec_append_resolved(
                    frames,
                    &target,
                    &items,
                    "Vec::extend_trusted",
                    &uninit_elem,
                    elem_size,
                )?;
                Ok(Value::Unit)
            }
            Builtin::VecSpecExtend => {
                arity(2)?;
                // The callee's generic arguments are the target
                // `Vec` type, the element type, and the iterator
                // type, which is the stepping dispatch key. Only
                // the `vec::IntoIter` instance reaches this arm;
                // conversion yields every other instance to its
                // collected body.
                let iter_ty = generic_args.get(2).map(String::as_str).unwrap_or("");
                let target = self.deref_value(frames, &vals[0])?;
                match self.read_resolved(frames, &target)? {
                    Value::Enum { .. } | Value::SparseVec { .. } => {}
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            "SpecExtend::spec_extend",
                            "vector value",
                            other.kind_name(),
                        ))
                    }
                }
                let (items, _) = self.drain_iterator(
                    frames,
                    iter_ty,
                    vals.into_iter().nth(1).expect("arity checked"),
                )?;
                let elem_tag = generic_args.get(1).map(String::as_str).unwrap_or("");
                let uninit_elem = uninit_elem_value_for_tag(elem_tag);
                let elem_size = self.layout_of(elem_tag).map_or(8, |(size, _)| size as usize);
                self.vec_append_resolved(
                    frames,
                    &target,
                    &items,
                    "SpecExtend::spec_extend",
                    &uninit_elem,
                    elem_size,
                )?;
                Ok(Value::Unit)
            }
            Builtin::VecFromIterNested => {
                arity(1)?;
                // The trait's generic arguments are the element
                // type followed by the iterator type, which is
                // the stepping dispatch key.
                let iter_ty = generic_args.last().map(String::as_str).unwrap_or("");
                let (items, final_state) = self.drain_iterator(
                    frames,
                    iter_ty,
                    vals.into_iter().next().expect("arity checked"),
                )?;
                self.note_consumed_iter_final(final_state);
                Ok(Value::Enum {
                    variant: 0,
                    fields: items,
                })
            }
            Builtin::VecCapacity => {
                arity(1)?;
                // A zero-sized element's capacity is `usize::MAX`
                // by definition (the `RawVecInner` rule), not the
                // storage extent.
                let elem_tag = generic_args.first().map(String::as_str).unwrap_or("");
                if self.layout_of(elem_tag).is_some_and(|(size, _)| size == 0) {
                    return Ok(Value::Uint(usize::MAX as u128));
                }
                // Only the storage extent is read: resolve by
                // borrowing, as `Vec::len` does.
                let value = self.resolve_ref(frames, &vals[0])?;
                let cap = vec_stored_len(&value).ok_or_else(|| {
                    InterpError::type_mismatch(
                        "Vec::capacity",
                        "vector value",
                        value.kind_name(),
                    )
                })?;
                Ok(Value::Uint(cap))
            }
            Builtin::SliceToVec => {
                // `to_vec` takes the slice alone; `to_vec_in`
                // takes the allocator handle after it (a
                // zero-sized value in this model).
                if vals.is_empty() || vals.len() > 2 {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "1 or 2 argument(s)",
                        format!("{} argument(s)", vals.len()),
                    ));
                }
                let (elems, len) = self.slice_elements(frames, &vals[0])?;
                // `to_vec` is the slice's clone into a vector:
                // every element clones with its own `Clone`
                // semantics, the way `Vec::clone` clones its
                // elements (see `clone_modelled_value`), not
                // the shape-only copy. The element tag is the
                // call's first generic argument.
                let elem_ty = generic_args.first().map(String::as_str);
                let elements =
                    self.clone_source_elements(frames, &vals[0], elems, len, elem_ty);
                let fields = self.clone_boxed_slice_elements(
                    frames,
                    &elements,
                    elem_ty,
                    elem_glue,
                )?;
                Ok(Value::Enum {
                    variant: 0,
                    fields,
                })
            }
            Builtin::SliceIter => {
                arity(1)?;
                // A fat pointer whose formation regrouped its
                // storage's elements iterates under the element
                // type the call names, not the stored shape (see
                // `regrouped_iter_elements`).
                let regrouped = generic_args
                    .first()
                    .and_then(|tag| ty_from_tag(tag))
                    .and_then(|elem_ty| {
                        self.regrouped_iter_elements(frames, &vals[0], &elem_ty)
                    })
                    .map(|elems| {
                        let len = elems.len() as u128;
                        (elems, len)
                    });
                let elem_tag =
                    generic_args.first().map(String::as_str).unwrap_or("");
                // A slice over a scalar's own storage (the
                // `slice::from_ref` shape, and a same-width
                // punned view over a scalar) reads its elements
                // from the storage itself, under the element
                // kind the call names: consult that reading
                // ahead of the plain element reading, whose
                // scalar case hands the storage back under its
                // own kind. Shapes the scalar reading does not
                // judge keep the plain reading below.
                let scalar_backed = self
                    .scalar_backed_slice_elements(frames, &vals[0], elem_tag)
                    .map(|elems| {
                        let len = elems.len() as u128;
                        (elems, len)
                    });
                let mut storage_state = None;
                let (elems, _) = match regrouped {
                    Some(elems) => elems,
                    None => match scalar_backed {
                        Some(elems) => elems,
                        None => match self.slice_elements(frames, &vals[0]) {
                            Ok(elems) => {
                                storage_state = self.slice_iter_storage_state(
                                    frames,
                                    &vals[0],
                                    &elems.0,
                                );
                                elems
                            }
                            Err(err) => {
                                // A slice over an address-only
                                // pointer has no elements in the
                                // value model — unless the element
                                // type has a single value, in which
                                // case every element is that value
                                // and no storage is needed to
                                // produce them.
                                match self.storageless_slice_elements(
                                    frames,
                                    &vals[0],
                                    elem_tag,
                                ) {
                                    Some(elems) => elems,
                                    None => return Err(err),
                                }
                            }
                        },
                    },
                };
                Ok(match storage_state {
                    Some(state) => state,
                    None => slice_iter_value(elems),
                })
            }
            Builtin::BTreeIter => {
                arity(1)?;
                if name.contains("BTreeSet") {
                    self.btree_set_iter_value(frames, &vals[0])
                } else {
                    self.btree_map_iter_value(frames, &vals[0])
                }
            }
            Builtin::BTreeKeys => {
                arity(1)?;
                self.btree_keys_value(frames, &vals[0])
            }
            Builtin::BTreeValues => {
                arity(1)?;
                self.btree_values_value(frames, &vals[0])
            }
            Builtin::StrContains
            | Builtin::StrStartsWith
            | Builtin::StrEndsWith
            | Builtin::StrTrimMatches(_) => {
                arity(2)?;
                let text_of = |v: &Value| -> InterpResult<String> {
                    // The text an argument denotes, in every
                    // form a string place can hold: the abstract
                    // form, the real aggregate form (the shape
                    // a body-built or raw-parts-materialised
                    // string takes) and the element form, read
                    // through the one string reader; a byte
                    // string denotes its lossy text.
                    match self.resolve_value(frames, v)? {
                        Value::Bytes(b) => Ok(String::from_utf8_lossy(&b).into_owned()),
                        _ => self.resolve_string(frames, v),
                    }
                };
                let haystack = text_of(&vals[0])?;
                // The `Pattern` impls beyond `&str`: a `char`
                // searches for that one character, a set of
                // `char`s for any of them, and a predicate
                // (`FnMut(char) -> bool`, arriving as a function
                // pointer or a closure) is consulted per
                // character -- the shapes collected std and
                // regex code actually pass.
                enum Needle {
                    Text(String),
                    One(char),
                    Set(Vec<char>),
                    Pred(Value),
                }
                let resolved = self.resolve_value(frames, &vals[1])?;
                // A string pattern held in an aggregate form
                // (the real aggregate a body-built or
                // raw-parts-materialised string takes, or the
                // element form): its text is the needle, read
                // through the string reader before the value
                // arms below judge the shape. Any other
                // aggregate (a boxed predicate, above all)
                // does not read as a string and keeps the
                // arms' own judgement.
                let aggregate_text = match &resolved {
                    Value::Enum { .. } => {
                        self.resolve_string(frames, &vals[1]).ok()
                    }
                    _ => None,
                };
                // A trait-object predicate is resolved through
                // the dyn callable machinery in the match below;
                // it records the environment its base denotes
                // here (a borrowed object's own reference, or the
                // reference into an owned object's box), and, for
                // an owned object, the object value to hand back
                // as the final state for the call's drop glue.
                // A concrete boxed predicate records the same
                // pair in the slice-fallback arm below: the
                // reference into its box, and the box itself.
                let mut dyn_pred_env: Option<Value> = None;
                let mut dyn_pred_final: Option<Value> = None;
                let needle = match resolved {
                    Value::Str(_) | Value::Bytes(_) => Needle::Text(text_of(&vals[1])?),
                    _ if aggregate_text.is_some() => {
                        Needle::Text(aggregate_text.clone().unwrap())
                    }
                    Value::Uint(v) => {
                        // A `&[char]` pattern whose fat base
                        // references an individual element (a
                        // slice built with
                        // `slice::from_raw_parts(&chars[0], 2)`,
                        // say) resolves to that element's scalar
                        // above. Detect the slice on the argument
                        // itself and read it through
                        // `slice_elements`, which preserves the
                        // length and the sibling elements, before
                        // accepting the scalar as the needle.
                        let char_set = if matches!(vals[1], Value::FatRef { .. }) {
                            self.char_set_of_slice(frames, &vals[1])
                        } else {
                            None
                        };
                        match char_set {
                            Some(set) => Needle::Set(set),
                            None => match char::from_u32(v as u32) {
                                Some(c) => Needle::One(c),
                                None => {
                                    return Err(InterpError::type_mismatch(
                                        "str pattern argument",
                                        "valid char pattern",
                                        "out-of-range char value",
                                    ))
                                }
                            },
                        }
                    }
                    Value::FnPtr(_) | Value::Closure { .. } => Needle::Pred(resolved),
                    Value::DynRef { .. } => {
                        // A trait-object predicate: `&dyn Fn`,
                        // `&mut dyn FnMut`, or an owned
                        // `Box<dyn FnMut>`. Resolve the object to
                        // the callable behind it and the
                        // environment its base denotes, the way
                        // the interpreter's own dyn method calls
                        // do.
                        let (callable, env) = self.dyn_callable(frames, &resolved)?;
                        match callable {
                            Value::FnPtr(_) | Value::Closure { .. } => {
                                match &env {
                                    // A borrowed object: the
                                    // environment is the reference
                                    // to the closure's own storage,
                                    // used directly on every
                                    // invocation (no allocation of
                                    // the predicate's own, and
                                    // nothing to drop).
                                    Value::Ref(target)
                                        if !matches!(target, RefTarget::Value(..)) =>
                                    {
                                        dyn_pred_env = Some(env);
                                    }
                                    _ => {
                                        // An owned object over a
                                        // box: the closure lives in
                                        // the box's allocation, so
                                        // invoke through that
                                        // allocation, and hand the
                                        // object itself back as the
                                        // final state for the call's
                                        // drop glue (its base denotes
                                        // the same storage the
                                        // invocations write through).
                                        if let Value::DynRef { base, .. } = &resolved {
                                            if let Some((id, projection)) =
                                                box_alloc_pointer(base)
                                            {
                                                if projection.is_empty() {
                                                    dyn_pred_env = Some(Value::Ref(
                                                        RefTarget::Alloc {
                                                            id,
                                                            projection: Vec::new(),
                                                        },
                                                    ));
                                                    dyn_pred_final =
                                                        Some(resolved.clone());
                                                }
                                            }
                                        }
                                    }
                                }
                                Needle::Pred(callable)
                            }
                            _ => {
                                return Err(InterpError::type_mismatch(
                                    "str pattern argument",
                                    "string, char, char set or predicate value",
                                    resolved.kind_name(),
                                ))
                            }
                        }
                    }
                    ref other => {
                        // A concrete boxed predicate
                        // (`Box<closure>` or `Box<fn ptr>`) arrives
                        // as the box aggregate over the allocation
                        // holding the callable, which `resolve_value`
                        // leaves in place. Resolve the callable
                        // before the character-slice fallback below:
                        // it is invoked through the box's
                        // allocation, so an `FnMut`'s state persists
                        // across the per-character invocations, and
                        // the box itself is handed back as the final
                        // state for the call's drop glue, exactly
                        // like an owned trait-object predicate
                        // above.
                        let boxed_callable = match box_alloc_pointer(other) {
                            Some((id, projection)) if projection.is_empty() => {
                                match self.alloc_value(id) {
                                    Some(pointee)
                                        if matches!(
                                            pointee,
                                            Value::FnPtr(_) | Value::Closure { .. }
                                        ) =>
                                    {
                                        Some((id, pointee))
                                    }
                                    _ => None,
                                }
                            }
                            _ => None,
                        };
                        match boxed_callable {
                            Some((id, pointee)) => {
                                dyn_pred_env = Some(Value::Ref(RefTarget::Alloc {
                                    id,
                                    projection: Vec::new(),
                                }));
                                dyn_pred_final = Some(other.clone());
                                Needle::Pred(pointee)
                            }
                            None => {
                                // A `&[char]` pattern arrives as a fat
                                // reference over the character tuple.
                                match self.slice_elements(frames, &vals[1]) {
                                    Ok((elems, _))
                                        if elems.iter().all(|e| matches!(e, Value::Uint(_))) =>
                                    {
                                        let mut set = Vec::with_capacity(elems.len());
                                        for e in &elems {
                                            let Value::Uint(v) = e else { unreachable!() };
                                            match char::from_u32(*v as u32) {
                                                Some(c) => set.push(c),
                                                None => {
                                                    return Err(InterpError::type_mismatch(
                                                        "str pattern argument",
                                                        "valid char pattern",
                                                        "out-of-range char value",
                                                    ))
                                                }
                                            }
                                        }
                                        Needle::Set(set)
                                    }
                                    _ => {
                                        return Err(InterpError::type_mismatch(
                                            "str pattern argument",
                                            "string, char, char set or predicate value",
                                            other.kind_name(),
                                        ))
                                    }
                                }
                            }
                        }
                    }
                };
                let result = match needle {
                    Needle::Text(t) => match kind {
                        Builtin::StrContains => {
                            Value::Bool(haystack.contains(&t))
                        }
                        Builtin::StrStartsWith => {
                            Value::Bool(haystack.starts_with(&t))
                        }
                        Builtin::StrEndsWith => {
                            Value::Bool(haystack.ends_with(&t))
                        }
                        Builtin::StrTrimMatches(side) => {
                            // Repeatedly strip the text pattern
                            // from the selected end(s): core's
                            // searcher semantics for a text
                            // pattern reach exactly the span
                            // between the leading and trailing
                            // runs of non-overlapping matches
                            // anchored at the ends. An empty
                            // pattern strips nothing.
                            let mut s = haystack.as_str();
                            if !t.is_empty() {
                                if matches!(
                                    side,
                                    TrimSide::Start | TrimSide::Both
                                ) {
                                    while let Some(rest) =
                                        s.strip_prefix(t.as_str())
                                    {
                                        s = rest;
                                    }
                                }
                                if matches!(
                                    side,
                                    TrimSide::End | TrimSide::Both
                                ) {
                                    while let Some(rest) =
                                        s.strip_suffix(t.as_str())
                                    {
                                        s = rest;
                                    }
                                }
                            }
                            Value::Str(s.to_string())
                        }
                        _ => unreachable!(),
                    },
                    needle => {
                        // Per-character needles: consult the
                        // test for the first (contains / starts
                        // with) or last (ends with) character the
                        // semantics reach.
                        //
                        // A predicate is an `FnMut`: it may
                        // mutate a by-value capture, and the
                        // mutation must be visible to the next
                        // character's invocation -- and, for a
                        // borrowed predicate, to the caller
                        // after the call. A borrowed predicate
                        // (`&pred` / `&mut pred`) arrives as a
                        // reference to the closure's own
                        // storage, which serves as the
                        // environment reference directly, so
                        // capture writes land in the caller's
                        // closure -- the arrangement the
                        // place-based iterator predicates use
                        // with their predicate field's storage.
                        // The reference may also sit one or
                        // more layers away from the closure: a
                        // pattern like `&mut borrowed`, where
                        // `borrowed: &mut pred`, hands the
                        // search a reference to the place
                        // holding the reference to the closure,
                        // and the chain is followed to the
                        // innermost reference, which denotes
                        // the closure's storage itself.
                        // (A reference to a value snapshot is
                        // not storage, and falls through to the
                        // by-value arrangement.) A predicate
                        // passed by value was moved into this
                        // call, so it gets storage of its own in
                        // the allocation store for the call's
                        // duration, and every invocation
                        // receives a reference to that storage.
                        // With no environment reference the call
                        // machinery would wrap a clone of the
                        // closure in a value target instead, and
                        // a write to a capture through it would
                        // fail as a write to a constant
                        // aggregate.
                        let borrowed_env = match &needle {
                            Needle::Pred(_) if dyn_pred_env.is_some() => dyn_pred_env.clone(),
                            Needle::Pred(_) => match &vals[1] {
                                Value::Ref(target)
                                    if !matches!(target, RefTarget::Value(..)) =>
                                {
                                    // Follow the chain one layer
                                    // at a time: each layer's
                                    // stored value, when it is
                                    // itself a reference to
                                    // storage, is the next
                                    // environment candidate, and
                                    // the last one denotes the
                                    // closure's own storage. A
                                    // layer that is a value
                                    // snapshot denotes no
                                    // storage, and the chain
                                    // then falls through to the
                                    // by-value arrangement.
                                    let mut env = vals[1].clone();
                                    let mut reaches_storage = true;
                                    loop {
                                        let stop = match &env {
                                            Value::Ref(RefTarget::Null)
                                            | Value::Ref(RefTarget::Address(_)) => true,
                                            Value::Ref(_) => false,
                                            _ => true,
                                        };
                                        if stop {
                                            break;
                                        }
                                        match self.read_ref_once(frames, &env) {
                                            Ok(Value::Ref(inner)) => {
                                                if matches!(inner, RefTarget::Value(..)) {
                                                    reaches_storage = false;
                                                    break;
                                                }
                                                env = Value::Ref(inner);
                                            }
                                            Ok(_) => break,
                                            Err(_) => break,
                                        }
                                    }
                                    if reaches_storage {
                                        Some(env)
                                    } else {
                                        None
                                    }
                                }
                                _ => None,
                            },
                            _ => None,
                        };
                        let pred_alloc = match &needle {
                            Needle::Pred(pred) if borrowed_env.is_none() => {
                                Some(self.alloc_insert(pred.clone()))
                            }
                            _ => None,
                        };
                        let pred_env = borrowed_env.or_else(|| {
                            pred_alloc.map(|id| {
                                Value::Ref(RefTarget::Alloc {
                                    id,
                                    projection: Vec::new(),
                                })
                            })
                        });
                        let mut test = |c: char| -> InterpResult<bool> {
                            Ok(match &needle {
                                Needle::One(n) => c == *n,
                                Needle::Set(set) => set.contains(&c),
                                Needle::Pred(pred) => match self.call_callable_sync(
                                    frames,
                                    pred,
                                    pred_env.clone(),
                                    vec![Value::Uint(u128::from(c as u32))],
                                )? {
                                    Value::Bool(b) => b,
                                    ref other => {
                                        return Err(InterpError::type_mismatch(
                                            "str pattern predicate",
                                            "boolean result",
                                            other.kind_name(),
                                        ))
                                    }
                                },
                                Needle::Text(_) => unreachable!(),
                            })
                        };
                        // The outcome stays a `Result` until the
                        // predicate's storage has been handed back
                        // below: a predicate that panics unwinds
                        // through the search, and the hand-back (and
                        // the drop glue it feeds) is owed on that
                        // path too -- the callee's cleanup drops a
                        // by-value predicate however the call ends.
                        let outcome: InterpResult<Value> = (|| match kind {
                            Builtin::StrContains => {
                                let mut found = false;
                                for c in haystack.chars() {
                                    if test(c)? {
                                        found = true;
                                        break;
                                    }
                                }
                                Ok(Value::Bool(found))
                            }
                            Builtin::StrStartsWith => match haystack.chars().next() {
                                Some(c) => Ok(Value::Bool(test(c)?)),
                                None => Ok(Value::Bool(false)),
                            },
                            Builtin::StrEndsWith => match haystack.chars().next_back() {
                                Some(c) => Ok(Value::Bool(test(c)?)),
                                None => Ok(Value::Bool(false)),
                            },
                            Builtin::StrTrimMatches(side) => {
                                // The trimmed span runs from the
                                // first character the test rejects
                                // (scanning from the selected
                                // start) to the end of the last
                                // character it rejects (scanning
                                // from the selected end): core's
                                // first/last reject spans. A side
                                // whose scan rejects nothing
                                // contributes the empty span at
                                // that end.
                                let mut start = 0usize;
                                let mut end = haystack.len();
                                if matches!(
                                    side,
                                    TrimSide::Start | TrimSide::Both
                                ) {
                                    start = haystack.len();
                                    for (i, c) in haystack.char_indices() {
                                        if !test(c)? {
                                            start = i;
                                            break;
                                        }
                                    }
                                }
                                if matches!(
                                    side,
                                    TrimSide::End | TrimSide::Both
                                ) {
                                    end = start;
                                    for (i, c) in
                                        haystack.char_indices().rev()
                                    {
                                        if i < start {
                                            break;
                                        }
                                        if !test(c)? {
                                            end = i + c.len_utf8();
                                            break;
                                        }
                                    }
                                }
                                Ok(Value::Str(
                                    haystack[start..end].to_string(),
                                ))
                            }
                            _ => unreachable!(),
                        })();
                        if let Some(id) = pred_alloc {
                            // The call consumed the predicate:
                            // the callee's real body drops it on
                            // the way out, captures and all, so
                            // hand its final state back for the
                            // drop conversion attached to the
                            // call (see `consumed_glue` in `ir`);
                            // the interpreter runs that glue
                            // once this builtin returns. This
                            // runs whether or not the predicate
                            // was ever invoked (an empty
                            // haystack still drops it). A
                            // borrowed predicate has no
                            // allocation here and keeps its
                            // owner, undropped. The hand-back
                            // also runs when the search unwound
                            // (a panicking predicate): the
                            // outcome is only propagated after
                            // the final state is recorded, and
                            // the `CallBuiltin` dispatch runs the
                            // attached glue on the error path as
                            // well, as the callee's cleanup
                            // would.
                            if let Some(final_pred) = self.alloc_value(id) {
                                self.note_consumed_iter_final(final_pred);
                            }
                            self.free_alloc(id);
                        }
                        if let Some(final_dyn) = dyn_pred_final {
                            // An owned trait-object predicate was
                            // invoked through its box's storage;
                            // the object value denotes that storage,
                            // so it is the final state the call's
                            // drop glue consumes. (Noting it here,
                            // before the outcome propagates, puts it
                            // on the unwind path as well, exactly
                            // like the by-value hand-back above.)
                            self.note_consumed_iter_final(final_dyn);
                        }
                        outcome?
                    }
                };
                Ok(result)
            }
            Builtin::SliceWindows => {
                arity(2)?;
                let size = match self.resolve_value(frames, &vals[1])? {
                    Value::Uint(v) => v,
                    Value::Enum { fields, .. } if fields.len() == 1 => match fields[0] {
                        Value::Uint(v) => v,
                        ref other => {
                            return Err(InterpError::type_mismatch(
                                "windows size",
                                "non-zero integer",
                                other.kind_name(),
                            ))
                        }
                    },
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            "windows size",
                            "non-zero integer",
                            other.kind_name(),
                        ))
                    }
                };
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![vals[0].clone(), Value::Uint(size), Value::Uint(0)],
                })
            }
            Builtin::IterEmpty => {
                arity(0)?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                })
            }
            Builtin::IndexIndex => {
                arity(2)?;
                // A range index (`slice[1..3]`, `slice[..]`, ...)
                // yields a sub-slice; the index type tag names the
                // range kind. Anything else is a plain integer index.
                let idx_tag = generic_args.get(1).map(String::as_str).unwrap_or("");
                if idx_tag.contains("ops::Range") {
                    return self.index_range(frames, name, &vals[0], &vals[1], idx_tag);
                }
                let idx = expect_index(&vals[1], name)?;
                // A receiver backed by a live place yields a live
                // element reference, exactly as `IndexMut` does:
                // interior-mutability writes through the shared
                // reference (`v[i].set(..)` on a `Vec<Cell<_>>`)
                // must land in the container's storage, which a
                // snapshot of the element cannot do.
                let fat_len = match &vals[0] {
                    Value::FatRef { len, .. } => Some(*len),
                    _ => None,
                };
                if let Some(r) = self.index_place_ref(frames, &vals[0], idx, fat_len)? {
                    return Ok(r);
                }
                let (elems, len) = self.slice_elements_ref(frames, &vals[0])?;
                if idx as u128 >= len {
                return Err(self.bounds_check_panic(frames, len as u128, idx as u128));
                }
                // A receiver with no live place behind it (a
                // constant aggregate, a detached value): reads
                // resolve against the element value itself, as for
                // constant aggregates. Only the one element is
                // cloned out of the borrowed element storage.
                let elem = elems.at(idx);
                Ok(Value::Ref(RefTarget::value(elem)))
            }
            Builtin::IndexMut => {
                arity(2)?;
                // A range index (`slice[1..3]`, `slice[..]`, ...)
                // yields a mutable sub-slice; the index type tag
                // names the range kind, as for the shared `Index`.
                let idx_tag = generic_args.get(1).map(String::as_str).unwrap_or("");
                if idx_tag.contains("ops::Range") {
                    return self.index_range_mut(frames, name, &vals[0], &vals[1], idx_tag);
                }
                let idx = expect_index(&vals[1], name)?;
                // A fat receiver's bounds are its carried length,
                // not the length of the base it points into (a
                // sub-slice of a longer array must panic at its
                // own length).
                let fat_len = match &vals[0] {
                    Value::FatRef { len, .. } => Some(*len),
                    _ => None,
                };
                let target = match &vals[0] {
                    Value::Ref(_) | Value::FatRef { .. } => {
                        self.deref_value(frames, &vals[0])?
                    }
                    other => {
                        return Err(InterpError::type_mismatch(
                            "IndexMut::index_mut",
                            "mutable reference",
                            other.kind_name(),
                        ))
                    }
                };
                match target {
                    Resolved::Place {
                        frame_pos,
                        local,
                        projection,
                    } => {
                        let target = Resolved::Place {
                            frame_pos,
                            local,
                            projection: projection.clone(),
                        };
                        let base = self.read_resolved(frames, &target)?;
                        let len = match fat_len {
                            Some(len) => len,
                            None => base.len().ok_or_else(|| {
                                InterpError::type_mismatch(
                                    "IndexMut::index_mut",
                                    "indexable value",
                                    base.kind_name(),
                                )
                            })?,
                        };
                        if idx as u128 >= len {
                return Err(self.bounds_check_panic(frames, len as u128, idx as u128));
                        }
                        // An element-entry suffix (see
                        // `element_entry_suffix`): the element is
                        // the entered position advanced by the
                        // index, under the fat length's bound
                        // (checked above) — pushing the index onto
                        // the entered element would address inside
                        // element zero instead.
                        if fat_len.map_or(false, |len| {
                            self.entry_suffix_denotes_elements(frames, &target, len)
                        }) {
                            if let Some(projection) =
                                advance_entry_projection(&projection, idx)
                            {
                                let frame_id = frames[frame_pos].id;
                                return Ok(Value::Ref(RefTarget::Place {
                                    frame: frame_id,
                                    local,
                                    projection,
                                }));
                            }
                        }
                        match base {
                            // Array (tuple) elements and vector
                            // (ADT aggregate) elements are both
                            // addressable through an index projection:
                            // the projection layer reads and writes
                            // `RefProj::Index` on both shapes. A
                            // vector in its sparse form (declared
                            // length below the stored list) is the
                            // same element aggregate and is
                            // addressable the same way.
                            Value::Tuple(_)
                            | Value::Enum { .. }
                            | Value::SparseVec { .. } => {
                                let mut projection = projection;
                                projection.push(RefProj::Index(idx));
                                let frame_id = frames[frame_pos].id;
                                Ok(Value::Ref(RefTarget::Place {
                                    frame: frame_id,
                                    local,
                                    projection,
                                }))
                            }
                            ref other => Err(unsupported(format!(
                                "IndexMut::index_mut on a {} is not runtime-supported",
                                other.kind_name()
                            ))),
                        }
                    }
                    Resolved::Static { name, projection } => {
                        let target = Resolved::Static {
                            name: name.clone(),
                            projection: projection.clone(),
                        };
                        let base = self.read_resolved(frames, &target)?;
                        let len = base.len().ok_or_else(|| {
                            InterpError::type_mismatch(
                                "IndexMut::index_mut",
                                "indexable value",
                                base.kind_name(),
                            )
                        })?;
                        if idx as u128 >= len {
                return Err(self.bounds_check_panic(frames, len as u128, idx as u128));
                        }
                        if fat_len.map_or(false, |len| {
                            self.entry_suffix_denotes_elements(frames, &target, len)
                        }) {
                            if let Some(projection) =
                                advance_entry_projection(&projection, idx)
                            {
                                return Ok(Value::Ref(RefTarget::Static {
                                    name,
                                    projection,
                                }));
                            }
                        }
                        match base {
                            Value::Tuple(_)
                            | Value::Enum { .. }
                            | Value::SparseVec { .. } => {
                                let mut projection = projection;
                                projection.push(RefProj::Index(idx));
                                Ok(Value::Ref(RefTarget::Static { name, projection }))
                            }
                            ref other => Err(unsupported(format!(
                                "IndexMut::index_mut on a {} is not runtime-supported",
                                other.kind_name()
                            ))),
                        }
                    }
                    Resolved::Alloc { id, projection } => {
                        let target = Resolved::Alloc {
                            id,
                            projection: projection.clone(),
                        };
                        let base = self.read_resolved(frames, &target)?;
                        let len = base.len().ok_or_else(|| {
                            InterpError::type_mismatch(
                                "IndexMut::index_mut",
                                "indexable value",
                                base.kind_name(),
                            )
                        })?;
                        if idx as u128 >= len {
                return Err(self.bounds_check_panic(frames, len as u128, idx as u128));
                        }
                        if fat_len.map_or(false, |len| {
                            self.entry_suffix_denotes_elements(frames, &target, len)
                        }) {
                            if let Some(projection) =
                                advance_entry_projection(&projection, idx)
                            {
                                return Ok(Value::Ref(RefTarget::Alloc { id, projection }));
                            }
                        }
                        match base {
                            // As for a place or static base above:
                            // ADT (vector aggregate) elements and
                            // sparse-vector elements are addressable
                            // through an index projection on an
                            // allocation too -- the projection layer
                            // reads and writes `RefProj::Index` on
                            // both shapes.
                            Value::Tuple(_)
                            | Value::Enum { .. }
                            | Value::SparseVec { .. } => {
                                let mut projection = projection;
                                projection.push(RefProj::Index(idx));
                                Ok(Value::Ref(RefTarget::Alloc { id, projection }))
                            }
                            ref other => Err(unsupported(format!(
                                "IndexMut::index_mut on a {} is not runtime-supported",
                                other.kind_name()
                            ))),
                        }
                    }
                    Resolved::Value(..) => Err(unsupported(
                        "IndexMut::index_mut on a constant aggregate is not runtime-supported",
                    )),
                }
            }
            Builtin::DerefMut => {
                arity(1)?;
                let self_ty = generic_args.first().map(String::as_str).unwrap_or("");
                // A `no_std` guest names the library types
                // through its own crate's extern prelude
                // (`test::alloc::string::String`), where a std
                // guest sees the `std::` re-export path; drop one
                // leading segment when what follows is a library
                // path so both spellings reach the models below.
                let self_ty = match self_ty.split_once("::") {
                    Some((_, rest))
                        if rest.starts_with("alloc::")
                            || rest.starts_with("core::")
                            || rest.starts_with("std::") =>
                    {
                        rest
                    }
                    _ => self_ty,
                };
                if self_ty == "std::string::String"
                    || self_ty == "alloc::string::String"
                {
                    // `&mut str` over the `String` place itself: reads
                    // see the string data, writes replace it.
                    Ok(vals.into_iter().next().expect("arity checked"))
                } else if self_ty.starts_with("std::vec::Vec<")
                    || self_ty.starts_with("alloc::vec::Vec<")
                {
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    let len = pointee.len().ok_or_else(|| {
                        InterpError::type_mismatch(
                            "DerefMut::deref_mut for Vec",
                            "array value",
                            pointee.kind_name(),
                        )
                    })?;
                    Ok(Value::FatRef {
                        base: Box::new(vals.into_iter().next().expect("arity checked")),
                        len,
                    })
                } else if smallvec_array_arg(self_ty).is_some() {
                    self.smallvec_slice(
                        frames,
                        &vals[0],
                        self_ty,
                        "DerefMut::deref_mut for SmallVec",
                    )
                } else if self_ty == "std::ffi::CString" {
                    // `CString::deref_mut -> &mut CStr`: the
                    // same fat reference `deref` produces.
                    self.c_string_deref(frames, &vals[0])
                } else if self_ty == "std::ffi::OsString" {
                    // `&mut OsStr` over the `OsString` place
                    // itself, as the `String` arm above hands
                    // out `&mut str` over the `String` place:
                    // reads see the string's bytes, and the
                    // `OsStr` mutation builtins rewrite the
                    // string the place holds.
                    Ok(vals.into_iter().next().expect("arity checked"))
                } else if self_ty.starts_with("std::boxed::Box<")
                    || self_ty.starts_with("alloc::boxed::Box<")
                {
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    // As in `Deref`: a `Box<dyn Trait>` dereferences
                    // to the fat pointer itself, and a built box
                    // dereferences to its allocation pointer — a
                    // live reference into the allocation store, so
                    // writes through it land in the pointee's
                    // storage and are observed through every copy
                    // of the box.
                    if let Value::DynRef { .. } = pointee {
                        return Ok(pointee);
                    }
                    match pointee {
                        Value::Enum { fields, .. } if fields.len() == 1 => {
                            let inner = fields.into_iter().next().expect("len checked");
                            if let Value::Ref(RefTarget::Alloc { .. }) = inner {
                                return Ok(inner);
                            }
                            // As in `Deref`: an aggregate-shaped
                            // box in a live place yields a
                            // reference to that place's field 0,
                            // so mutable writes through the
                            // result land in the box; a box
                            // reached only as a value keeps the
                            // snapshot form.
                            // (A field that is itself a one-field
                            // wrapper around a reference is another
                            // pointer layer, not the pointee: its
                            // snapshot is kept, as for the built
                            // box's allocation pointer above.)
                            let wrapped_pointer = matches!(
                                &inner,
                                Value::Enum { fields, .. }
                                    if fields.len() == 1
                                        && matches!(fields[0], Value::Ref(_))
                            );
                            if !wrapped_pointer {
                                if let Some(place_ref) = self.place_projected_ref(
                                    frames,
                                    &vals[0],
                                    RefProj::Field(0),
                                )? {
                                    return Ok(place_ref);
                                }
                            }
                            Ok(Value::Ref(RefTarget::value(inner)))
                        }
                        ref other => Err(InterpError::type_mismatch(
                            "DerefMut::deref_mut for Box",
                            "boxed value",
                            other.kind_name(),
                        )),
                    }
                } else if self_ty.starts_with("std::mem::ManuallyDrop<") {
                    // A one-field newtype wrapper: the result is a
                    // mutable reference to the held value. As for
                    // `AssertUnwindSafe` below, a place-backed
                    // argument yields a live place reference, so
                    // writes through the result land in the wrapper
                    // (a resolved-value snapshot would silently
                    // drop them).
                    self.field_zero_ref(
                        frames,
                        &vals[0],
                        "DerefMut::deref_mut for ManuallyDrop",
                    )
                } else if self_ty.starts_with("std::boxed::ThinBox<") {
                    self.thin_box_deref(
                        frames,
                        &vals[0],
                        self_ty,
                        "DerefMut::deref_mut for ThinBox",
                    )
                } else if self_ty.starts_with("std::rc::Rc<")
                    || self_ty.starts_with("std::sync::Arc<")
                {
                    // As in `Deref`: a reference to the stored
                    // value field, a live place in the allocation
                    // store, so a write through it is observed by
                    // every handle on the allocation. A dyn pointee
                    // yields the fat pointer itself.
                    let pointee = self.resolve_value(frames, &vals[0])?;
                    if let Value::DynRef { .. } = pointee {
                        return Ok(pointee);
                    }
                    match rc_handle_id(&pointee) {
                        Some(id) => Ok(Value::Ref(RefTarget::Alloc {
                            id,
                            projection: self
                                .rc_pointee_projection(&pointee)
                                .expect("handle checked"),
                        })),
                        None => Err(InterpError::type_mismatch(
                            "DerefMut::deref_mut for Rc/Arc",
                            "a reference-counted handle",
                            pointee.kind_name(),
                        )),
                    }
                } else if self_ty_is(self_ty, "std::cell::RefMut") {
                    // A mutable borrow guard: field 0 is the
                    // reference to the cell's value, in the same
                    // two shapes as the shared guard above.
                    let guard = self.resolve_value(frames, &vals[0])?;
                    match crate::interp::refcell_guard_fields(&guard) {
                        Some(fields) => {
                            match crate::interp::refcell_guard_ref(&fields[0]) {
                                Some(target) => Ok(Value::Ref(target.clone())),
                                None => Ok(fields[0].clone()),
                            }
                        }
                        None => Err(InterpError::type_mismatch(
                            "RefCell borrow guard",
                            "RefMut guard aggregate",
                            guard.kind_name(),
                        )),
                    }
                } else if self_ty_is(self_ty, "std::sync::MutexGuard")
                    || self_ty_is(self_ty, "std::sync::RwLockWriteGuard")
                {
                    // A lock guard: field 0 is the reference to
                    // the lock the converted lock body stored;
                    // the guarded value is the lock's data field.
                    self.lock_guard_deref(
                        frames,
                        &vals[0],
                        "DerefMut::deref_mut for lock guard",
                    )
                } else if self_ty_is(self_ty, "std::sync::MappedMutexGuard")
                    || self_ty_is(self_ty, "std::sync::MappedRwLockWriteGuard")
                    || self_ty_is(self_ty, "std::sync::nonpoison::MappedMutexGuard")
                    || self_ty_is(self_ty, "std::sync::nonpoison::MappedRwLockWriteGuard")
                {
                    // A mapped guard: as in `Deref`, field 0 is
                    // the `NonNull` pointer to the mapped
                    // component itself.
                    self.mapped_guard_deref(
                        frames,
                        &vals[0],
                        "DerefMut::deref_mut for mapped lock guard",
                    )
                } else if self_ty.starts_with("&mut ") {
                    // A mutable reference's own `DerefMut` returns
                    // the inner `&mut T` itself: the argument is
                    // `&mut self`, naming the place that holds the
                    // inner reference, so reading it exactly once
                    // yields that reference. The coroutine
                    // runtime paths reach this arm constantly:
                    // `Pin<&mut C>` is built by dereferencing the
                    // `&mut C` it wraps, and async drop glue
                    // dereferences `&mut ManuallyDrop<C>` /
                    // `&mut Pin<..>` the same way.
                    Ok(self.read_ref_once(frames, &vals[0])?)
                } else if self_ty.starts_with("std::pin::Pin<") {
                    // `Pin<P>` derefs exactly as the pointer it
                    // wraps (see `pin_deref_value`).
                    let wrapper = self.resolve_value(frames, &vals[0])?;
                    pin_deref_value(wrapper, self_ty, "DerefMut::deref_mut for Pin")
                } else if self_ty.starts_with("std::panic::AssertUnwindSafe<") {
                    // A one-field newtype wrapper: the result is a
                    // mutable reference to the held value.
                    self.field_zero_ref(
                        frames,
                        &vals[0],
                        "DerefMut::deref_mut for AssertUnwindSafe",
                    )
                } else {
                    Err(unsupported(format!(
                        "DerefMut::deref_mut for {self_ty} is not runtime-supported"
                    )))
                }
            }
            Builtin::CellNew => {
                arity(1)?;
                Ok(cell_value(vals.into_iter().next().expect("arity checked")))
            }
            Builtin::CellGet | Builtin::CellIntoInner => {
                arity(1)?;
                let state = self.resolve_value(frames, &vals[0])?;
                match cell_held(&state).cloned() {
                    Some(value) => Ok(value),
                    None => Err(InterpError::type_mismatch(
                        "Cell value",
                        "cell aggregate",
                        state.kind_name(),
                    )),
                }
            }
            Builtin::CellSet | Builtin::CellReplace => {
                arity(2)?;
                let target = self.deref_value(frames, &vals[0])?;
                let state = self.read_resolved(frames, &target)?;
                let old = match cell_held(&state).cloned() {
                    Some(value) => value,
                    None => {
                        return Err(InterpError::type_mismatch(
                            "Cell::set",
                            "cell aggregate",
                            state.kind_name(),
                        ))
                    }
                };
                self.write_resolved(
                    frames,
                    &target,
                    cell_value(vals.into_iter().nth(1).expect("arity checked")),
                )?;
                if kind == Builtin::CellReplace {
                    Ok(old)
                } else {
                    Ok(Value::Unit)
                }
            }
            Builtin::RefCellNew => {
                arity(1)?;
                let value = vals.into_iter().next().expect("arity checked");
                Ok(refcell_value(value))
            }
            Builtin::RefCellBorrow | Builtin::RefCellBorrowMut => {
                arity(1)?;
                let mutable = kind == Builtin::RefCellBorrowMut;
                let cell = self.refcell_place(&vals[0], name)?;
                let flag_target =
                    crate::interp::refcell_sub_target(&cell, &[RefProj::Field(0)]);
                let flag = self.refcell_flag(frames, &flag_target)?;
                if mutable {
                    if flag != 0 {
                        return Err(InterpError::panic("RefCell already borrowed"));
                    }
                    self.refcell_set_flag(frames, &flag_target, -1)?;
                } else {
                    if flag < 0 {
                        return Err(InterpError::panic(
                            "RefCell already mutably borrowed",
                        ));
                    }
                    self.refcell_set_flag(frames, &flag_target, flag + 1)?;
                }
                Ok(refcell_guard(&cell))
            }
            Builtin::RefCellTryBorrow | Builtin::RefCellTryBorrowMut => {
                arity(1)?;
                let mutable = kind == Builtin::RefCellTryBorrowMut;
                let cell = self.refcell_place(&vals[0], name)?;
                let flag_target =
                    crate::interp::refcell_sub_target(&cell, &[RefProj::Field(0)]);
                let flag = self.refcell_flag(frames, &flag_target)?;
                let conflict = if mutable { flag != 0 } else { flag < 0 };
                if conflict {
                    // Err(BorrowError / BorrowMutError): unit-struct
                    // aggregates in this value model.
                    return Ok(Value::Enum {
                        variant: 1,
                        fields: vec![Value::Enum {
                            variant: 0,
                            fields: Vec::new(),
                        }],
                    });
                }
                self.refcell_set_flag(
                    frames,
                    &flag_target,
                    if mutable { -1 } else { flag + 1 },
                )?;
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![refcell_guard(&cell)],
                })
            }
            Builtin::RefCellGetMut | Builtin::RefCellAsPtr => {
                arity(1)?;
                let cell = self.refcell_place(&vals[0], name)?;
                Ok(Value::Ref(crate::interp::refcell_sub_target(
                    &cell,
                    &[RefProj::Field(1), RefProj::Field(0)],
                )))
            }
            Builtin::RefCellReplace => {
                arity(2)?;
                let mut vals = vals.into_iter();
                let cell_arg = vals.next().expect("arity checked");
                let new_value = vals.next().expect("arity checked");
                let cell = self.refcell_place(&cell_arg, name)?;
                let flag_target =
                    crate::interp::refcell_sub_target(&cell, &[RefProj::Field(0)]);
                let flag = self.refcell_flag(frames, &flag_target)?;
                if flag != 0 {
                    return Err(InterpError::panic("RefCell already borrowed"));
                }
                let value_target = crate::interp::refcell_sub_target(
                    &cell,
                    &[RefProj::Field(1), RefProj::Field(0)],
                );
                let resolved = self.deref_value(frames, &Value::Ref(value_target))?;
                let old = self.read_resolved(frames, &resolved)?;
                self.write_resolved(frames, &resolved, new_value)?;
                Ok(old)
            }
            Builtin::OptionUnwrap => {
                arity(1)?;
                match self.resolve_value(frames, &vals[0])? {
                    Value::Enum { variant: 1, fields } if fields.len() == 1 => {
                        Ok(fields.into_iter().next().expect("len checked"))
                    }
                    Value::Enum { variant: 0, .. } => {
                        // The modelled operation raises the
                        // panic `unwrap_failed` raises natively:
                        // its location is the caller-location
                        // stamp the call carried (noted in
                        // `exec_builtin`), its entry path
                        // `unwrap_failed`'s.
                        self.note_panic_origin(PanicOrigin::OptionUnwrap);
                        self.note_panic_no_backtrace(false);
                        Err(InterpError::panic_str_payload(
                            "called `Option::unwrap()` on a `None` value",
                        ))
                    }
                    ref other => Err(InterpError::type_mismatch(
                        "Option::unwrap",
                        "Option value",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::ResultUnwrap => {
                arity(1)?;
                match self.resolve_value(frames, &vals[0])? {
                    Value::Enum { variant: 0, fields } if fields.len() == 1 => {
                        Ok(fields.into_iter().next().expect("len checked"))
                    }
                    Value::Enum { variant: 1, fields } => {
                        let err = fields
                            .into_iter()
                            .next()
                            .unwrap_or(Value::Unit);
                        // As for `Option::unwrap` above: the
                        // panic is `unwrap_failed`'s, located by
                        // the stamp the call carried.
                        self.note_panic_origin(PanicOrigin::ResultUnwrap);
                        self.note_panic_no_backtrace(false);
                        Err(InterpError::panic_string_payload(format!(
                            "called `Result::unwrap()` on an `Err` value: {err}"
                        )))
                    }
                    ref other => Err(InterpError::type_mismatch(
                        "Result::unwrap",
                        "Result value",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::OptionIsSome
            | Builtin::OptionIsNone
            | Builtin::ResultIsOk
            | Builtin::ResultIsErr => {
                arity(1)?;
                match self.resolve_value(frames, &vals[0])? {
                    Value::Enum { variant, .. } => Ok(Value::Bool(match kind {
                        Builtin::OptionIsSome => variant == 1,
                        Builtin::OptionIsNone => variant == 0,
                        Builtin::ResultIsOk => variant == 0,
                        _ => variant == 1,
                    })),
                    ref other => Err(InterpError::type_mismatch(
                        "Option/Result variant check",
                        "enum value",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::OptionTake => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                let old = self.read_resolved(frames, &target)?;
                match old {
                    Value::Enum { .. } => {
                        self.write_resolved(frames, &target, none_value())?;
                        Ok(old)
                    }
                    ref other => Err(InterpError::type_mismatch(
                        "Option::take",
                        "Option value",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::MemDrop => {
                arity(1)?;
                // A `mem::drop` whose argument's glue was trivial
                // runs no glue session, so a dropped box's storage
                // free is observed here instead (see
                // `Interpreter::note_box_free`).
                if self.program().alloc_shims.contains_key("__rust_dealloc") {
                    let arg = vals.into_iter().next().expect("arity checked");
                    if crate::interp::box_alloc_pointer(&arg).is_some() {
                        self.note_box_free(frames, &arg);
                        if let Some((id, _)) =
                            crate::interp::box_alloc_pointer(&arg)
                        {
                            self.free_alloc(id);
                        }
                    }
                }
                Ok(Value::Unit)
            }
            Builtin::MemSwap => {
                arity(2)?;
                let a = self.deref_value(frames, &vals[0])?;
                let b = self.deref_value(frames, &vals[1])?;
                let va = self.read_resolved(frames, &a)?;
                let vb = self.read_resolved(frames, &b)?;
                self.write_resolved(frames, &a, vb)?;
                self.write_resolved(frames, &b, va)?;
                Ok(Value::Unit)
            }
            Builtin::SwapNonoverlapping => {
                // `typed_swap_nonoverlapping` swaps a single pair
                // (two arguments); `swap_nonoverlapping` carries
                // the element count as a third.
                let n = match vals.len() {
                    2 => 1,
                    3 => match vals[2] {
                        Value::Uint(n) => usize::try_from(n).map_err(|_| {
                            InterpError::type_mismatch(
                                "swap_nonoverlapping count",
                                "count that fits in usize",
                                format!("{n}"),
                            )
                        })?,
                        Value::Int(n) if n >= 0 => n as usize,
                        ref other => {
                            return Err(InterpError::type_mismatch(
                                "swap_nonoverlapping count",
                                "unsigned integer",
                                other.kind_name(),
                            ))
                        }
                    },
                    _ => {
                        return Err(InterpError::type_mismatch(
                            "swap_nonoverlapping",
                            "two or three arguments",
                            format!("{} arguments", vals.len()),
                        ))
                    }
                };
                if n == 0 {
                    // A zero-count swap touches no memory: the
                    // pointers are never dereferenced.
                    return Ok(Value::Unit);
                }
                let a = self.deref_value(frames, &vals[0])?;
                let b = self.deref_value(frames, &vals[1])?;
                if n == 1 {
                    // A single element is the whole value each
                    // target denotes: the sequence access would
                    // index INTO a structured value (an index
                    // step descends into an aggregate's fields)
                    // instead of naming the element, so the
                    // exchange goes through the whole-value
                    // access, exactly like the one-element copy.
                    let va = self.read_resolved(frames, &a)?;
                    let vb = self.read_resolved(frames, &b)?;
                    self.write_resolved(frames, &a, vb)?;
                    self.write_resolved(frames, &b, va)?;
                    return Ok(Value::Unit);
                }
                // Snapshot both ranges before writing: the
                // ranges are promised non-overlapping, but the
                // snapshot keeps the exchange exact regardless
                // (the same discipline as the element copy).
                let mut va = Vec::with_capacity(n);
                let mut vb = Vec::with_capacity(n);
                for i in 0..n {
                    va.push(self.read_seq_at(frames, &a, i)?);
                    vb.push(self.read_seq_at(frames, &b, i)?);
                }
                for i in 0..n {
                    self.write_seq_at(frames, &a, i, vb[i].clone())?;
                    self.write_seq_at(frames, &b, i, va[i].clone())?;
                }
                Ok(Value::Unit)
            }
            Builtin::PtrRotate => {
                arity(3)?;
                // `ptr_rotate_gcd` / `ptr_rotate_swap` both take
                // `(left, mid, right)`: the range of `left +
                // right` elements ending and starting at `mid`
                // rotates left by `left`.
                let count_of = |v: &Value| -> InterpResult<usize> {
                    match v {
                        Value::Uint(n) => usize::try_from(*n).map_err(|_| {
                            InterpError::type_mismatch(
                                "ptr_rotate count",
                                "count that fits in usize",
                                format!("{n}"),
                            )
                        }),
                        Value::Int(n) if *n >= 0 => {
                            usize::try_from(*n).map_err(|_| {
                                InterpError::type_mismatch(
                                    "ptr_rotate count",
                                    "count that fits in usize",
                                    format!("{n}"),
                                )
                            })
                        }
                        other => Err(InterpError::type_mismatch(
                            "ptr_rotate count",
                            "unsigned integer",
                            other.kind_name(),
                        )),
                    }
                };
                let left = count_of(&vals[0])?;
                let right = count_of(&vals[2])?;
                if left == 0 || right == 0 {
                    // A rotation by nothing, or of a range one
                    // side of which is empty, moves no element.
                    return Ok(Value::Unit);
                }
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (size, _) = self.layout_of(tag).ok_or_else(|| {
                    unsupported(format!(
                        "ptr_rotate for {tag} is not runtime-supported"
                    ))
                })?;
                if size == 0 {
                    // A zero-sized element type: every element
                    // is the same value, so the rotation is
                    // unobservable.
                    return Ok(Value::Unit);
                }
                let size = u64::try_from(size).map_err(|_| {
                    unsupported(format!(
                        "ptr_rotate for {tag} is not runtime-supported"
                    ))
                })?;
                let n = left.checked_add(right).ok_or_else(|| {
                    unsupported("ptr_rotate: range length overflows".to_string())
                })?;
                // The range starts `left` elements before the
                // pointer the call hands over: rebase the
                // pointer to the range's first element, in the
                // element's own units.
                let back = i128::try_from(left).map_err(|_| {
                    unsupported("ptr_rotate: range length overflows".to_string())
                })?;
                let base_val = self.ptr_offset_value(
                    frames,
                    vals[1].clone(),
                    -back,
                    Some(size),
                    false,
                )?;
                let base = self.deref_value(frames, &base_val)?;
                // Snapshot the whole range before writing: the
                // rotation's writes land on the positions its
                // reads came from (the same discipline as the
                // element swap above).
                let mut elems = Vec::with_capacity(n);
                for i in 0..n {
                    elems.push(self.read_seq_at(frames, &base, i)?);
                }
                for i in 0..n {
                    let src = (i + left) % n;
                    self.write_seq_at(frames, &base, i, elems[src].clone())?;
                }
                Ok(Value::Unit)
            }
            Builtin::MemSizeOf | Builtin::MemAlignOf => {
                arity(0)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                match self.layout_of(tag) {
                    Some((size, align)) => Ok(Value::Uint(if kind == Builtin::MemSizeOf {
                        size
                    } else {
                        align
                    })),
                    None => Err(unsupported(format!(
                        "size_of/align_of for {tag} is not runtime-supported"
                    ))),
                }
            }
            Builtin::MemSizeOfVal => {
                arity(1)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                if tag == "str" {
                    let s = self.resolve_string(frames, &vals[0])?;
                    Ok(Value::Uint(s.len() as u128))
                } else if is_slice_tag(tag) {
                    // A slice: element count × element size.
                    let elem = &tag[1..tag.len() - 1];
                    let (elem_size, _) = self.layout_of(elem).ok_or_else(|| {
                        unsupported(format!(
                            "size_of_val for {tag} is not runtime-supported"
                        ))
                    })?;
                    let len = match &vals[0] {
                        Value::FatRef { len, .. } => *len,
                        _ => {
                            let value = self.resolve_value(frames, &vals[0])?;
                            value.len().ok_or_else(|| {
                                InterpError::type_mismatch(
                                    "size_of_val",
                                    "slice value",
                                    value.kind_name(),
                                )
                            })?
                        }
                    };
                    Ok(Value::Uint(len * elem_size))
                } else if let Some((size, _)) =
                    self.dyn_layout_of(frames, &vals[0], tag)
                {
                    Ok(Value::Uint(size))
                } else if let Some((size, _)) = self.layout_of(tag) {
                    Ok(Value::Uint(size))
                } else if let Some((size, _)) =
                    self.dst_size_align_of_val(frames, &vals[0], tag)
                {
                    Ok(Value::Uint(size))
                } else {
                    Err(unsupported(format!(
                        "size_of_val for {tag} is not runtime-supported"
                    )))
                }
            }
            Builtin::MemAlignOfVal => {
                arity(1)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                if tag == "str" {
                    Ok(Value::Uint(1))
                } else if is_slice_tag(tag) {
                    let elem = &tag[1..tag.len() - 1];
                    let (_, align) = self.layout_of(elem).ok_or_else(|| {
                        unsupported(format!(
                            "align_of_val for {tag} is not runtime-supported"
                        ))
                    })?;
                    Ok(Value::Uint(align))
                } else if let Some((_, align)) =
                    self.dyn_layout_of(frames, &vals[0], tag)
                {
                    Ok(Value::Uint(align))
                } else if let Some((_, align)) = self.layout_of(tag) {
                    Ok(Value::Uint(align))
                } else if let Some((_, align)) =
                    self.dst_size_align_of_val(frames, &vals[0], tag)
                {
                    Ok(Value::Uint(align))
                } else {
                    Err(unsupported(format!(
                        "align_of_val for {tag} is not runtime-supported"
                    )))
                }
            }
            Builtin::VtableSize | Builtin::VtableAlign => {
                arity(1)?;
                let want_align = kind == Builtin::VtableAlign;
                match self.vtable_stored_layout(frames, &vals[0]) {
                    Some((size, align)) => {
                        Ok(Value::Uint(if want_align { align } else { size }))
                    }
                    None => Err(unsupported(
                        "vtable_size/vtable_align for a pointer denoting no recorded vtable",
                    )),
                }
            }
            Builtin::FastFloat(op) => {
                arity(2)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (a, b) = match (&vals[0], &vals[1]) {
                    (Value::Float(a), Value::Float(b)) => (*a, *b),
                    _ => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "two float operands",
                            format!(
                                "{} and {}",
                                vals[0].kind_name(),
                                vals[1].kind_name()
                            ),
                        ));
                    }
                };
                let r = match op {
                    FastFloatOp::Add => a + b,
                    FastFloatOp::Sub => a - b,
                    FastFloatOp::Mul => a * b,
                    FastFloatOp::Div => a / b,
                    FastFloatOp::Rem => a % b,
                };
                // A narrower float operation's result rounds to
                // the operation's width, as in the interpreter's
                // own float arithmetic.
                Ok(Value::Float(match tag {
                    "f16" => round_to_f16(r),
                    "f32" => (r as f32) as f64,
                    _ => r,
                }))
            }
            Builtin::FloatAbs => {
                arity(1)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                match &vals[0] {
                    Value::Float(x) => Ok(Value::Float(match tag {
                        "f16" => round_to_f16((*x).abs()),
                        "f32" => ((*x) as f32).abs() as f64,
                        _ => (*x).abs(),
                    })),
                    Value::Float128(bits) => Ok(Value::Float128(crate::f128::abs(*bits))),
                    other => Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "float operand",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::FloatToIntUnchecked => {
                arity(1)?;
                let tag = generic_args.get(1).map(String::as_str).unwrap_or("");
                let (bits, signed) = int_tag(tag).ok_or_else(|| {
                    unsupported(format!(
                        "float_to_int_unchecked to {tag} is not runtime-supported"
                    ))
                })?;
                match vals[0] {
                    Value::Float(x) => Ok(if signed {
                        Value::Int(float_to_int(x, bits))
                    } else {
                        Value::Uint(float_to_uint(x, bits))
                    }),
                    ref other => Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "float operand",
                        other.kind_name(),
                    )),
                }
            }
            Builtin::MemNeedsDrop => {
                arity(0)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                // The compiler's verdict recorded at
                // conversion for the types the program's own
                // `needs_drop` calls ask about is exact — it
                // is the emptiness of the type's drop-in-place
                // shim, so it agrees with the glue the program
                // runs. The static tag table answers for the
                // types no recorded verdict names.
                let needs = self
                    .program()
                    .needs_drop_verdicts
                    .get(tag)
                    .copied()
                    .or_else(|| needs_drop_for_tag(tag));
                match needs {
                    Some(needs) => Ok(Value::Bool(needs)),
                    None => Err(unsupported(format!(
                        "needs_drop for {tag} is not runtime-supported"
                    ))),
                }
            }
            Builtin::ToOwned => {
                arity(1)?;
                // The owned type is named by the call's
                // generic arguments (the borrowed type). For
                // the borrowed types whose owned form is an
                // aggregate denominated by its bytes -- an
                // `OsStr` receiver owns an `OsString`, a
                // `Path` a `PathBuf`, a `CStr` a `CString` --
                // build that aggregate from the receiver's
                // bytes: the generic arms below would hand
                // back the byte sequence itself (a slice's
                // element list, or the string value), which
                // the owned type's consumers refuse as a
                // differently shaped ADT. The `CStr` case is
                // the sharpest: the C string aggregate's
                // single field is the byte string itself,
                // where the `CString`'s field is the owned
                // box, and the length-trusting consumers (the
                // `Deref` to `&CStr` above all) refuse the
                // byte string where the box belongs.
                match generic_args.first().map(String::as_str) {
                    Some("std::ffi::OsStr") => {
                        let bytes = self.os_str_bytes(frames, &vals[0])?;
                        return Ok(os_string_value(&bytes));
                    }
                    Some("std::path::Path") => {
                        let bytes = self.os_str_bytes(frames, &vals[0])?;
                        return Ok(pathbuf_value(&bytes));
                    }
                    Some("std::ffi::CStr") | Some("core::ffi::CStr") => {
                        let bytes = self.c_str_bytes(frames, &vals[0])?;
                        return Ok(self.c_string_value(bytes));
                    }
                    _ => {}
                }
                // A `str` receiver reached through a fat
                // reference: the owned form is a `String`, whose
                // value is the string itself -- not the byte
                // vector the slice arm below builds from the
                // same fat reference (that arm serves a slice
                // receiver, whose pointee is an element
                // sequence, never a string value).
                if let Value::FatRef { base, .. } = &vals[0] {
                    if let Ok(Value::Str(text)) = self.resolve_value(frames, base) {
                        return Ok(Value::Str(text));
                    }
                }
                match &vals[0] {
                    // A slice: the owned form is a vector of the
                    // elements.
                    Value::FatRef { .. } => {
                        let (elems, _) = self.slice_elements(frames, &vals[0])?;
                        Ok(Value::Enum {
                            variant: 0,
                            fields: elems,
                        })
                    }
                    Value::Ref(_) => {
                        let target = self.deref_value(frames, &vals[0])?;
                        // A reference denoting the element of a
                        // same-width punned scalar view (Issue
                        // #479) denotes the storage's bits under
                        // the element kind: the owned value is
                        // the punned reading, not the backing
                        // scalar the plain resolution returns.
                        if let Some(tag) = generic_args.first() {
                            if let Some(v) =
                                self.punned_scalar_element(frames, &target, tag)
                            {
                                return Ok(v);
                            }
                        }
                        match self.read_resolved(frames, &target)? {
                            Value::Tuple(items) => Ok(Value::Enum {
                                variant: 0,
                                fields: items,
                            }),
                            other => Ok(other),
                        }
                    }
                    other => Ok(other.clone()),
                }
            }
            Builtin::BlackBox => {
                arity(1)?;
                Ok(vals.into_iter().next().expect("arity checked"))
            }
            Builtin::IntPow => {
                // One call form in core's own `pow` bodies names
                // this function with no arguments and no return
                // target: it is the cold overflow exit of the
                // checked path, reached only when the
                // exponentiation has already overflowed.
                if vals.is_empty() {
                    return Err(InterpError::panic(
                        "attempt to exponentiate with overflow",
                    ));
                }
                arity(2)?;
                // The generic helper names its type in the
                // call's generic arguments; an inherent method
                // call carries none and names it in the
                // definition path's impl segment instead.
                let tag = generic_args
                    .first()
                    .map(String::as_str)
                    .filter(|t| !t.is_empty())
                    .or_else(|| {
                        name.split("<impl ")
                            .nth(1)
                            .and_then(|rest| rest.split('>').next())
                    })
                    .unwrap_or("");
                let (bits, signed) = int_tag(tag).ok_or_else(|| {
                    unsupported(format!("pow for type {tag} is not runtime-supported"))
                })?;
                let mask = bit_mask(bits);
                let base = (match &vals[0] {
                    Value::Int(i) => *i as u128,
                    Value::Uint(u) => *u,
                    _ => 0,
                }) & mask;
                let mut exp = match &vals[1] {
                    Value::Uint(u) => *u,
                    Value::Int(i) if *i >= 0 => *i as u128,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "pow exponent",
                            "unsigned integer",
                            other.kind_name(),
                        ))
                    }
                };
                // Checked multiplication at the type's width;
                // the wrapped product is kept for the
                // overflow-checks-off path.
                let mul = |x: u128, y: u128| -> (u128, bool) {
                    if signed {
                        let xi = sign_extend(x, bits);
                        let yi = sign_extend(y, bits);
                        let wrapped = (xi.wrapping_mul(yi) as u128) & mask;
                        let in_range = if bits == 128 {
                            xi.checked_mul(yi).is_some()
                        } else {
                            let max = (1i128 << (bits - 1)) - 1;
                            let min = -(1i128 << (bits - 1));
                            xi.checked_mul(yi)
                                .is_some_and(|v| (min..=max).contains(&v))
                        };
                        (wrapped, !in_range)
                    } else {
                        let wrapped = x.wrapping_mul(y) & mask;
                        let over = match x.checked_mul(y) {
                            Some(v) => v > mask,
                            None => true,
                        };
                        (wrapped, over)
                    }
                };
                let mut result: u128 = 1;
                let mut b = base;
                let mut overflowed = false;
                while exp > 0 {
                    if exp & 1 == 1 {
                        let (v, over) = mul(result, b);
                        result = v;
                        overflowed |= over;
                    }
                    exp >>= 1;
                    if exp > 0 {
                        let (v, over) = mul(b, b);
                        b = v;
                        overflowed |= over;
                    }
                }
                if overflowed && self.overflow_checks_at(frames) {
                    return Err(InterpError::panic(
                        "attempt to exponentiate with overflow",
                    ));
                }
                Ok(if signed {
                    Value::Int(sign_extend(result, bits))
                } else {
                    Value::Uint(result)
                })
            }
            Builtin::FunnelShift { left } => {
                arity(3)?;
                // An inherent method call carries no generic
                // arguments; the Self type is named in the
                // definition path's impl segment.
                let tag = generic_args
                    .first()
                    .map(String::as_str)
                    .filter(|t| !t.is_empty())
                    .or_else(|| {
                        name.split("<impl ")
                            .nth(1)
                            .and_then(|rest| rest.split('>').next())
                    })
                    .unwrap_or("");
                let (bits, signed) = int_tag(tag).ok_or_else(|| {
                    unsupported(format!(
                        "funnel shift for type {tag} is not runtime-supported"
                    ))
                })?;
                if signed {
                    return Err(unsupported(format!(
                        "funnel shift for signed type {tag}"
                    )));
                }
                let mask = bit_mask(bits);
                let operand = |v: &Value| -> Result<u128, InterpError> {
                    match v {
                        Value::Uint(u) => Ok(*u & mask),
                        Value::Int(i) => Ok((*i as u128) & mask),
                        other => Err(InterpError::type_mismatch(
                            "funnel shift operand",
                            "unsigned integer",
                            other.kind_name(),
                        )),
                    }
                };
                let hi = operand(&vals[0])?;
                let lo = operand(&vals[1])?;
                let n = operand(&vals[2])?;
                // The methods are `#[rustc_inherit_overflow_checks]`:
                // the width-reaching shift panics under the
                // instantiating crate's setting, resolved from
                // the stack, and wraps the amount otherwise.
                if n >= bits as u128 && self.overflow_checks_at(frames) {
                    return Err(InterpError::panic(if left {
                        "attempt to funnel shift left with overflow"
                    } else {
                        "attempt to funnel shift right with overflow"
                    }));
                }
                let n = (n as u32) & (bits - 1);
                let result = if n == 0 {
                    if left { hi } else { lo }
                } else if left {
                    ((hi << n) & mask) | (lo >> (bits - n))
                } else {
                    (hi >> n) | ((lo << (bits - n)) & mask)
                };
                Ok(Value::Uint(result))
            }
            Builtin::FromStrParse => {
                arity(1)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let text = self.resolve_string(frames, &vals[0])?;
                return self.exec_from_str(tag, &text, name);
            }
            Builtin::VolatileLoad => {
                arity(1)?;
                // A zero-sized pointee has a single value and no
                // storage: loading it through a storage-less
                // pointer (the null pointer a read_volatile of a
                // zero-sized type passes) fabricates that value
                // instead of dereferencing, mirroring the
                // zero-sized volatile store's early return below.
                if let Some(ty) = args
                    .first()
                    .and_then(|a| crate::interp::operand_pointee_ty(body, a))
                {
                    if let Some(resolved) =
                        self.storage_less_zst_deref(&vals[0], Some(&ty))
                    {
                        return self.read_resolved(frames, &resolved);
                    }
                }
                let target = self.deref_value(frames, &vals[0])?;
                // As with the atomic load: a wider-than-byte
                // access into byte storage composes the bytes.
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                if let Some((bits, signed)) = int_tag(tag) {
                    if let Some(value) =
                        self.read_punned_int(frames, &target, bits, signed)?
                    {
                        return Ok(value);
                    }
                }
                // The pointee value itself, composite types
                // included: no wrapper unwrapping (that is the
                // atomic operations' UnsafeCell model, not a
                // property of a plain load).
                Ok(self.read_resolved(frames, &target)?)
            }
            Builtin::VolatileStore => {
                arity(2)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                if self.layout_of(tag).is_some_and(|(size, _)| size == 0) {
                    return Ok(Value::Unit);
                }
                let target = self.deref_value(frames, &vals[0])?;
                // The store matching the punned load above.
                if let Some((bits, _)) = int_tag(tag) {
                    if self.write_punned_int(frames, &target, bits, &vals[1])? {
                        return Ok(Value::Unit);
                    }
                }
                self.write_resolved(frames, &target, vals[1].clone())?;
                Ok(Value::Unit)
            }
            Builtin::VolatileSet => {
                arity(3)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (size, _) = self.layout_of(tag).ok_or_else(|| {
                    unsupported(format!(
                        "volatile_set_memory element type {tag} has no recorded layout"
                    ))
                })?;
                let count = match &vals[2] {
                    Value::Uint(n) => *n,
                    Value::Int(n) if *n >= 0 => *n as u128,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "volatile_set_memory count",
                            "unsigned integer",
                            other.kind_name(),
                        ))
                    }
                };
                for i in 0..count {
                    let p = self.ptr_offset_value(
                        frames,
                        vals[0].clone(),
                        i as i128,
                        Some(size as u64),
                        true,
                    )?;
                    let target = self.deref_value(frames, &p)?;
                    self.write_resolved(frames, &target, vals[1].clone())?;
                }
                Ok(Value::Unit)
            }
            Builtin::VolatileCopy => {
                arity(3)?;
                let mut vals = vals.into_iter();
                let dst = vals.next().expect("arity checked");
                let src_v = vals.next().expect("arity checked");
                let count = vals.next().expect("arity checked");
                // The element type, from the caller's declared
                // pointer types, as on the intrinsic statement
                // path (this builtin's argument order is dst, src).
                let elem = args
                    .get(1)
                    .and_then(|a| crate::interp::operand_pointee_ty(body, a))
                    .or_else(|| {
                        args.first()
                            .and_then(|a| crate::interp::operand_pointee_ty(body, a))
                    });
                self.exec_copy_nonoverlapping(frames, src_v, dst, count, elem.as_ref())?;
                Ok(Value::Unit)
            }
            Builtin::PtrNull => {
                arity(0)?;
                // The null pointer: a dedicated reference target
                // denoting no place and no value. It is equal only
                // to itself (see `values_equal`); dereferencing it
                // is an explicit error, as a null dereference has
                // no meaning in the value model.
                Ok(Value::Ref(RefTarget::Null))
            }
            Builtin::FnPtrAddr => {
                arity(1)?;
                // The synthetic address every other exposure
                // of the same function value agrees on (the
                // `FnPtrToPtr` cast, a pointer-to-integer
                // transmute); recording it lets a later
                // `from_ptr` recover the callable.
                let addr = synthetic_address(&vals[0]).ok_or_else(|| {
                    InterpError::type_mismatch(
                        "FnPtr::addr",
                        "a function pointer value",
                        vals[0].kind_name(),
                    )
                })?;
                self.note_exposed_target(addr as u64, &vals[0]);
                Ok(Value::Uint(addr))
            }
            Builtin::FnPtrAsPtr => {
                arity(1)?;
                // The address-only reference the `FnPtrToPtr`
                // cast produces for the same value.
                let addr = synthetic_address(&vals[0]).ok_or_else(|| {
                    InterpError::type_mismatch(
                        "FnPtr::as_ptr",
                        "a function pointer value",
                        vals[0].kind_name(),
                    )
                })?;
                self.note_exposed_target(addr as u64, &vals[0]);
                Ok(Value::Ref(RefTarget::Address(addr as u64)))
            }
            Builtin::FnPtrFromPtr => {
                arity(1)?;
                // The argument is a `NonNull<Code>`: its scalar
                // shape is the bare pointer, but a value a
                // converted body built may still carry the
                // one-field wrapper, so unwrap to the pointer.
                let mut cursor = &vals[0];
                loop {
                    match cursor {
                        Value::Enum { fields, .. } if fields.len() == 1 => {
                            cursor = &fields[0];
                        }
                        Value::Tuple(items) if items.len() == 1 => {
                            cursor = &items[0];
                        }
                        _ => break,
                    }
                }
                let addr = synthetic_address(cursor).ok_or_else(|| {
                    InterpError::type_mismatch(
                        "FnPtr::from_ptr",
                        "a code pointer value",
                        vals[0].kind_name(),
                    )
                })?;
                self.exposed_callable(&Value::Uint(addr)).ok_or_else(|| {
                    InterpError::type_mismatch(
                        "FnPtr::from_ptr",
                        "a pointer obtained from FnPtr::as_ptr",
                        format!("address {addr:#x}"),
                    )
                })
            }
            Builtin::PtrOffsetFrom { unsigned } => {
                arity(2)?;
                // Two address-only pointers have no pointee to
                // dereference in this model; their distance is a
                // byte distance over the synthetic addresses.
                let addr_pair = match (&vals[0], &vals[1]) {
                    (
                        Value::Ref(RefTarget::Address(a)),
                        Value::Ref(RefTarget::Address(b)),
                    ) => Some((*a, *b)),
                    _ => None,
                };
                let d = match addr_pair {
                    Some((a, b)) => {
                        let tag = generic_args.first().map(String::as_str).unwrap_or("");
                        let size = self
                            .layout_of(tag)
                            .map(|(size, _)| size)
                            .filter(|size| *size > 0)
                            .ok_or_else(|| {
                                unsupported(format!(
                                    "ptr_offset_from with element type `{tag}` is not \
                                     runtime-supported"
                                ))
                            })? as i128;
                        let bytes = (a as i128).wrapping_sub(b as i128);
                        if bytes % size != 0 {
                            None
                        } else {
                            Some(bytes / size)
                        }
                    }
                    _ => {
                        let x = self.deref_value(frames, &vals[0])?;
                        let y = self.deref_value(frames, &vals[1])?;
                        self.pointer_distance(frames, &x, &y, generic_args)?
                    }
                };
                match d {
                    Some(d) if unsigned && d < 0 => Err(unsupported(
                        "ptr_offset_from_unsigned with a negative distance is not \
                         runtime-supported"
                            .to_string(),
                    )),
                    Some(d) => Ok(if unsigned {
                        Value::Uint(d as u128)
                    } else {
                        Value::Int(d)
                    }),
                    None => Err(unsupported(
                        "ptr_offset_from between unrelated pointers is not runtime-supported"
                            .to_string(),
                    )),
                }
            }
            Builtin::MemZeroed => {
                arity(0)?;
                // `mem::zeroed`'s own `assert_zero_valid` call
                // never runs -- this builtin preempts the body --
                // so a type the compiler's validity check rejects
                // raises its `panic_nounwind` here, before any
                // value is materialized (see
                // `TerminatorKind::CallBuiltin::validity_panic`).
                if let Some(message) = validity_panic {
                    return Err(self.validity_assertion_abort(frames, message));
                }
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                match zeroed_value_for_tag(tag) {
                    Some(value) => Ok(value),
                    // The tag-based construction names scalars,
                    // tuples and arrays; a type only the layout
                    // describes (an enum, an ADT aggregate)
                    // materializes as the value conversion
                    // built for this call from the recorded
                    // layout (see `zeroed_value` in `ir`).
                    None => match zeroed_value {
                        Some(value) => Ok(value.clone()),
                        None => Err(unsupported(format!(
                            "mem::zeroed for {tag} is not runtime-supported"
                        ))),
                    },
                }
            }
            Builtin::IntBitOp(op) => {
                arity(1)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (bits, signed) = int_tag(tag).ok_or_else(|| {
                    unsupported(format!("bit intrinsic for {tag} is not runtime-supported"))
                })?;
                let pattern = match vals[0] {
                    Value::Int(v) => (v as u128) & bit_mask(bits),
                    Value::Uint(v) => v & bit_mask(bits),
                    ref other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "integer",
                            other.kind_name(),
                        ));
                    }
                };
                let result = match op {
                    BitOp::CtPop => pattern.count_ones() as u128,
                    BitOp::Ctlz => (pattern.leading_zeros() - (128 - bits)) as u128,
                    BitOp::Cttz => {
                        if pattern == 0 {
                            bits as u128
                        } else {
                            pattern.trailing_zeros() as u128
                        }
                    }
                    BitOp::BitReverse => pattern.reverse_bits() >> (128 - bits),
                    BitOp::Bswap => pattern.swap_bytes() >> (128 - bits),
                };
                Ok(if signed {
                    Value::Int(sign_extend(result, bits))
                } else {
                    Value::Uint(result)
                })
            }
            Builtin::SaturatingArith(op) => {
                arity(2)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (bits, signed) = int_tag(tag).ok_or_else(|| {
                    unsupported(format!(
                        "saturating arithmetic for {tag} is not runtime-supported"
                    ))
                })?;
                if signed {
                    let (a, b) = (as_i128(&vals[0], name)?, as_i128(&vals[1], name)?);
                    let max = sign_extend(bit_mask(bits) >> 1, bits);
                    let min = -max - 1;
                    let r = match op {
                        SatOp::Add => {
                            if b > 0 && a > max - b {
                                max
                            } else if b < 0 && a < min - b {
                                min
                            } else {
                                a + b
                            }
                        }
                        SatOp::Sub => {
                            if b < 0 && a > max + b {
                                max
                            } else if b > 0 && a < min + b {
                                min
                            } else {
                                a - b
                            }
                        }
                    };
                    Ok(Value::Int(r))
                } else {
                    let (a, b) = (as_u128(&vals[0], name)?, as_u128(&vals[1], name)?);
                    let max = bit_mask(bits);
                    let r = match op {
                        SatOp::Add => a.checked_add(b).map_or(max, |sum| sum.min(max)),
                        SatOp::Sub => a.saturating_sub(b),
                    };
                    Ok(Value::Uint(r))
                }
            }
            Builtin::ArithOffset { wrapping } => {
                arity(2)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (size, _) = self.layout_of(tag).ok_or_else(|| {
                    unsupported(format!(
                        "pointer offset for {tag} is not runtime-supported"
                    ))
                })?;
                let count = match &vals[1] {
                    Value::Int(v) => *v,
                    Value::Uint(v) => i128::try_from(*v).map_err(|_| {
                        unsupported("pointer offset count does not fit in an isize")
                    })?,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "pointer offset count",
                            "integer count",
                            other.kind_name(),
                        ))
                    }
                };
                self.ptr_offset_value(frames, vals[0].clone(), count, Some(size as u64), wrapping)
            }
            Builtin::AtomicLoad => {
                arity(1)?;
                let target = self.deref_value(frames, &vals[0])?;
                // A wider-than-byte access through a pointer into
                // byte storage (an unaligned / type-punned load)
                // composes the stored bytes.
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                if let Some((bits, signed)) = int_tag(tag) {
                    if let Some(value) =
                        self.read_punned_int(frames, &target, bits, signed)?
                    {
                        return Ok(value);
                    }
                }
                let current = self.read_resolved(frames, &target)?;
                Ok(unwrap_wrapper(&current).clone())
            }
            Builtin::AtomicStore => {
                arity(2)?;
                let target = self.deref_value(frames, &vals[0])?;
                let value = vals.into_iter().nth(1).expect("arity checked");
                // The store matching the punned load above.
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                if let Some((bits, _)) = int_tag(tag) {
                    if self.write_punned_int(frames, &target, bits, &value)? {
                        return Ok(Value::Unit);
                    }
                }
                self.atomic_write(frames, &target, value)?;
                Ok(Value::Unit)
            }
            Builtin::AtomicXadd | Builtin::AtomicXsub => {
                arity(2)?;
                let sub = kind == Builtin::AtomicXsub;
                let target = self.deref_value(frames, &vals[0])?;
                let current = self.read_resolved(frames, &target)?;
                let old = unwrap_wrapper(&current).clone();
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (bits, signed) = int_tag(tag).ok_or_else(|| {
                    unsupported(format!("{name} on `{tag}` is not runtime-supported"))
                })?;
                let new = match (&old, &vals[1]) {
                    (Value::Uint(a), Value::Uint(b)) => {
                        let mask = if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 };
                        Value::Uint(if sub {
                            a.wrapping_sub(*b) & mask
                        } else {
                            a.wrapping_add(*b) & mask
                        })
                    }
                    (Value::Int(a), Value::Int(b)) => {
                        let raw = if sub {
                            a.wrapping_sub(*b)
                        } else {
                            a.wrapping_add(*b)
                        };
                        // Re-wrap at the operand's width.
                        let shift = 128 - bits;
                        Value::Int((raw << shift) >> shift)
                    }
                    _ => {
                        return Err(InterpError::type_mismatch(
                            name,
                            "matching integer operands",
                            format!(
                                "{} and {}",
                                old.kind_name(),
                                vals[1].kind_name()
                            ),
                        ))
                    }
                };
                let _ = signed;
                self.atomic_write(frames, &target, new)?;
                Ok(old)
            }
            Builtin::AtomicFetchOp(op) => {
                arity(2)?;
                let target = self.deref_value(frames, &vals[0])?;
                let current = self.read_resolved(frames, &target)?;
                let old = unwrap_wrapper(&current).clone();
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                let (bits, signed) = int_tag(tag).ok_or_else(|| {
                    unsupported(format!("{name} on `{tag}` is not runtime-supported"))
                })?;
                let mask = bit_mask(bits);
                let cur = as_u128(&old, name)? & mask;
                let operand = as_u128(&vals[1], name)? & mask;
                let new = match op {
                    FetchOp::Add => cur.wrapping_add(operand) & mask,
                    FetchOp::Sub => cur.wrapping_sub(operand) & mask,
                    FetchOp::And => cur & operand,
                    FetchOp::Or => cur | operand,
                    FetchOp::Xor => cur ^ operand,
                    FetchOp::Nand => !(cur & operand) & mask,
                    FetchOp::Max => {
                        if sign_extend(cur, bits) >= sign_extend(operand, bits) {
                            cur
                        } else {
                            operand
                        }
                    }
                    FetchOp::Min => {
                        if sign_extend(cur, bits) <= sign_extend(operand, bits) {
                            cur
                        } else {
                            operand
                        }
                    }
                    FetchOp::UMax => cur.max(operand),
                    FetchOp::UMin => cur.min(operand),
                };
                let new_value = if signed {
                    Value::Int(sign_extend(new, bits))
                } else {
                    Value::Uint(new)
                };
                self.atomic_write(frames, &target, new_value)?;
                Ok(old)
            }
            Builtin::PtrCopy => {
                arity(3)?;
                let mut vals = vals.into_iter();
                let src = vals.next().expect("arity checked");
                let dst = vals.next().expect("arity checked");
                let count = vals.next().expect("arity checked");
                // The element type, from the caller's declared
                // pointer types, as on the intrinsic statement
                // path.
                let elem = args
                    .first()
                    .and_then(|a| crate::interp::operand_pointee_ty(body, a))
                    .or_else(|| {
                        args.get(1)
                            .and_then(|a| crate::interp::operand_pointee_ty(body, a))
                    });
                self.exec_copy_nonoverlapping(frames, src, dst, count, elem.as_ref())?;
                Ok(Value::Unit)
            }
            Builtin::CompareBytes => {
                arity(3)?;
                let n = match &vals[2] {
                    Value::Uint(n) => usize::try_from(*n).map_err(|_| {
                        InterpError::type_mismatch(
                            "compare_bytes count",
                            "count that fits in usize",
                            format!("{n}"),
                        )
                    })?,
                    Value::Int(n) if *n >= 0 => *n as usize,
                    other => {
                        return Err(InterpError::type_mismatch(
                            "compare_bytes count",
                            "unsigned integer",
                            other.kind_name(),
                        ));
                    }
                };
                let (a, b) = self.read_compare_ranges(frames, &vals[0], &vals[1], n)?;
                let ord = a.cmp(&b);
                Ok(Value::Int(match ord {
                    std::cmp::Ordering::Less => -1,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                }))
            }
            Builtin::RawEq => {
                arity(2)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                if tag == "str" {
                    let a = self.resolve_string(frames, &vals[0])?;
                    let b = self.resolve_string(frames, &vals[1])?;
                    return Ok(Value::Bool(a == b));
                }
                // The byte count behind one reference, by the
                // same cascade `size_of_val` uses (a slice spans
                // its length × the element size).
                let mut byte_len = |interp: &mut Self, v: &Value| -> InterpResult<u128> {
                    if is_slice_tag(tag) {
                        let elem = &tag[1..tag.len() - 1];
                        let (elem_size, _) = interp.layout_of(elem).ok_or_else(|| {
                            unsupported(format!(
                                "raw_eq for {tag} is not runtime-supported"
                            ))
                        })?;
                        let len = match v {
                            Value::FatRef { len, .. } => *len,
                            _ => {
                                let value = interp.resolve_value(frames, v)?;
                                value.len().ok_or_else(|| {
                                    InterpError::type_mismatch(
                                        "raw_eq",
                                        "slice value",
                                        value.kind_name(),
                                    )
                                })?
                            }
                        };
                        Ok(len * elem_size)
                    } else if let Some((size, _)) = interp.dyn_layout_of(frames, v, tag) {
                        Ok(size)
                    } else if let Some((size, _)) = interp.layout_of(tag) {
                        Ok(size)
                    } else if let Some((size, _)) =
                        interp.dst_size_align_of_val(frames, v, tag)
                    {
                        Ok(size)
                    } else {
                        Err(unsupported(format!(
                            "raw_eq for {tag} is not runtime-supported"
                        )))
                    }
                };
                let n0 = byte_len(self, &vals[0])?;
                let n1 = byte_len(self, &vals[1])?;
                if n0 != n1 {
                    return Ok(Value::Bool(false));
                }
                let n = usize::try_from(n0).map_err(|_| {
                    unsupported(format!("raw_eq for {tag} is not runtime-supported"))
                })?;
                let (a, b) = self.read_compare_ranges(frames, &vals[0], &vals[1], n)?;
                Ok(Value::Bool(a == b))
            }
            Builtin::SimdOp(op) => self.exec_simd(op, frames, generic_args, &vals),
            Builtin::RustAlloc { zeroed } => {
                arity(2)?;
                let size = match &vals[0] {
                    Value::Uint(n) => *n,
                    Value::Int(n) if *n >= 0 => *n as u128,
                    other => {
                        return Err(InterpError::type_mismatch(
                            name,
                            "allocation size",
                            other.kind_name(),
                        ))
                    }
                };
                let align = u128::from(alloc_align_arg(&vals[1]));
                // The uninitialised form stores nothing per byte,
                // so only the zeroed form has a materialisation
                // limit; past it the shim's failure signal (the
                // null pointer) is the faithful answer.
                if zeroed && size > (1 << 28) {
                    return Ok(Value::Ref(RefTarget::Null));
                }
                let content = if zeroed {
                    Value::Bytes(vec![0u8; size as usize])
                } else {
                    // An element tuple that grows on write (see
                    // the allocation arm of write_resolved): the
                    // blocks elements have no value-level count
                    // until they are written, and unwritten slots
                    // read back as uninitialized.
                    Value::Tuple(Vec::new())
                };
                let id = self.alloc_insert(content);
                self.mark_elem_block(id);
                self.record_alloc_layout(id, size as u64, align as u64);
                Ok(Value::Ref(RefTarget::Alloc {
                    id,
                    projection: Vec::new(),
                }))
            }
            Builtin::RustDealloc => {
                arity(3)?;
                let ptr = unwrap_wrapper(&vals[0]);
                if let Value::Ref(RefTarget::Alloc { id, projection }) = ptr {
                    if self.alloc_ptr_is_base(*id, projection) {
                        self.free_alloc(*id);
                    }
                }
                Ok(Value::Unit)
            }
            Builtin::RustRealloc => {
                arity(4)?;
                let new_size = match &vals[3] {
                    Value::Uint(n) => *n,
                    Value::Int(n) if *n >= 0 => *n as u128,
                    other => {
                        return Err(InterpError::type_mismatch(
                            name,
                            "allocation size",
                            other.kind_name(),
                        ))
                    }
                };
                let realloc_align = u128::from(alloc_align_arg(&vals[2]));
                let ptr = unwrap_wrapper(&vals[0]).clone();
                // A block-base pointer: the empty projection, or
                // the element-0 form an elem block's base pointer
                // also travels in (a `RawVec`'s stored pointer).
                let content = match &ptr {
                    Value::Ref(RefTarget::Alloc { id, projection })
                        if self.alloc_ptr_is_base(*id, projection) =>
                    {
                        let old = self.alloc_base(*id)?;
                        self.free_alloc(*id);
                        old
                    }
                    _ => {
                        let _ = new_size;
                        Value::Uninit
                    }
                };
                let id = self.alloc_insert(content);
                self.mark_elem_block(id);
                self.record_alloc_layout(
                    id,
                    new_size as u64,
                    realloc_align as u64,
                );
                Ok(Value::Ref(RefTarget::Alloc {
                    id,
                    projection: Vec::new(),
                }))
            }
            Builtin::LibcMalloc => {
                arity(1)?;
                let size = alloc_size_arg(name, &vals[0])?;
                let id = self.alloc_insert(Value::Tuple(Vec::new()));
                self.mark_elem_block(id);
                self.record_alloc_layout(id, size as u64, 16);
                Ok(Value::Ref(RefTarget::Alloc {
                    id,
                    projection: Vec::new(),
                }))
            }
            Builtin::LibcCalloc => {
                arity(2)?;
                let count = alloc_size_arg(name, &vals[0])?;
                let size = alloc_size_arg(name, &vals[1])?;
                let total = count.saturating_mul(size);
                if total > (1 << 28) {
                    return Ok(Value::Ref(RefTarget::Null));
                }
                let id = self.alloc_insert(Value::Bytes(vec![0u8; total as usize]));
                self.mark_elem_block(id);
                self.record_alloc_layout(id, total as u64, 16);
                Ok(Value::Ref(RefTarget::Alloc {
                    id,
                    projection: Vec::new(),
                }))
            }
            Builtin::LibcFree => {
                arity(1)?;
                let ptr = unwrap_wrapper(&vals[0]);
                if let Value::Ref(RefTarget::Alloc { id, projection }) = ptr {
                    if self.alloc_ptr_is_base(*id, projection) {
                        self.free_alloc(*id);
                    }
                }
                Ok(Value::Unit)
            }
            Builtin::LibcRealloc => {
                arity(2)?;
                let new_size = alloc_size_arg(name, &vals[1])?;
                let ptr = unwrap_wrapper(&vals[0]).clone();
                // A block-base pointer: the empty projection, or
                // the element-0 form an elem block's base pointer
                // also travels in (a `RawVec`'s stored pointer).
                let content = match &ptr {
                    Value::Ref(RefTarget::Alloc { id, projection })
                        if self.alloc_ptr_is_base(*id, projection) =>
                    {
                        let old = self.alloc_base(*id)?;
                        self.free_alloc(*id);
                        old
                    }
                    _ => {
                        let _ = new_size;
                        Value::Uninit
                    }
                };
                let id = self.alloc_insert(content);
                self.mark_elem_block(id);
                self.record_alloc_layout(id, new_size as u64, 16);
                Ok(Value::Ref(RefTarget::Alloc {
                    id,
                    projection: Vec::new(),
                }))
            }
            Builtin::LibcPosixMemalign => {
                arity(3)?;
                let size = alloc_size_arg(name, &vals[2])?;
                let align = alloc_size_arg(name, &vals[1])?;
                let _ = size;
                let id = self.alloc_insert(Value::Tuple(Vec::new()));
                self.mark_elem_block(id);
                self.record_alloc_layout(id, size as u64, align as u64);
                let block = Value::Ref(RefTarget::Alloc {
                    id,
                    projection: Vec::new(),
                });
                let target = self.deref_value(frames, &vals[0])?;
                self.write_resolved(frames, &target, block)?;
                Ok(Value::Int(0))
            }
            Builtin::LibcSigaction => {
                arity(3)?;
                self.exec_libc_sigaction(frames, name, &vals)
            }
            Builtin::LibcSignal => {
                arity(2)?;
                self.exec_libc_signal(frames, name, &vals)
            }
            Builtin::LibcRaise => {
                arity(1)?;
                self.exec_libc_raise(frames, name, &vals)
            }
            Builtin::LibcAtexit => {
                arity(1)?;
                self.exec_libc_atexit(frames, name, &vals)
            }
            Builtin::LibcRead => {
                arity(3)?;
                self.exec_libc_read(frames, name, &vals)
            }
            Builtin::LibcWrite => {
                arity(3)?;
                self.exec_libc_write(frames, name, &vals)
            }
            Builtin::ErrnoLocation => {
                arity(0)?;
                self.exec_errno_location()
            }
            Builtin::LibcSysconf => {
                arity(1)?;
                self.exec_libc_sysconf(frames, name, &vals)
            }
            Builtin::LibcGetuid => {
                arity(0)?;
                self.exec_libc_getuid(name)
            }
            Builtin::LibcGetgid => {
                arity(0)?;
                self.exec_libc_getgid(name)
            }
            Builtin::LibcGetpid => {
                arity(0)?;
                self.exec_libc_getpid()
            }
            Builtin::LibcClearenv => {
                arity(0)?;
                self.exec_libc_clearenv(name)
            }
            Builtin::LibcPrctl => {
                if vals.is_empty() || vals.len() > 5 {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "1 to 5 argument(s)",
                        format!("{} argument(s)", vals.len()),
                    ));
                }
                self.exec_libc_prctl(frames, name, &vals)
            }
            Builtin::LibcDlopen => {
                arity(2)?;
                self.exec_libc_dlopen(frames, name, &vals)
            }
            Builtin::LibcDlsym => {
                arity(2)?;
                self.exec_libc_dlsym(frames, name, &vals)
            }
            Builtin::LibcDup => {
                arity(1)?;
                self.exec_libc_dup(frames, name, &vals)
            }
            Builtin::LibcDup2 => {
                arity(2)?;
                self.exec_libc_dup2(frames, name, &vals)
            }
            Builtin::LibcClose => {
                arity(1)?;
                self.exec_libc_close(frames, name, &vals)
            }
            Builtin::LibcPipe => {
                arity(1)?;
                self.exec_libc_pipe(frames, name, &vals)
            }
            Builtin::LibcFcntl => {
                if vals.len() < 2 || vals.len() > 3 {
                    return Err(InterpError::type_mismatch(
                        format!("builtin {name}"),
                        "2 to 3 argument(s)",
                        format!("{} argument(s)", vals.len()),
                    ));
                }
                self.exec_libc_fcntl(frames, name, &vals)
            }
            Builtin::LibcNftwLegacy => {
                arity(4)?;
                self.exec_libc_nftw(frames, name, &vals, true)
            }
            Builtin::LibcNftw => {
                arity(4)?;
                self.exec_libc_nftw(frames, name, &vals, false)
            }
            Builtin::AtomicCxchg => {
                arity(3)?;
                // Single-threaded: nothing can intervene between
                // the read and the conditional write, and a weak
                // exchange's spurious failure is not simulated, so
                // `atomic_cxchgweak` is the same operation.
                let target = self.deref_value(frames, &vals[0])?;
                let current = self.read_resolved(frames, &target)?;
                let success = self.atomic_value_eq(frames, &current, &vals[1])?;
                if success {
                    self.atomic_write(frames, &target, vals[2].clone())?;
                }
                // The exchanged-out value the caller receives is the
                // cell's scalar, not the wrapper aggregate the cell
                // place holds.
                let previous = unwrap_wrapper(&current).clone();
                Ok(Value::Tuple(vec![previous, Value::Bool(success)]))
            }
            Builtin::AtomicXchg => {
                arity(2)?;
                let target = self.deref_value(frames, &vals[0])?;
                let current = self.read_resolved(frames, &target)?;
                let previous = unwrap_wrapper(&current).clone();
                self.atomic_write(frames, &target, vals[1].clone())?;
                Ok(previous)
            }
            Builtin::AtomicFence => {
                arity(0)?;
                // A fence orders other threads' accesses; there
                // are none.
                Ok(Value::Unit)
            }
            Builtin::AssertValidity => {
                arity(0)?;
                // The asserted property is a fact about the
                // call's type the compiler itself settled (see
                // `TerminatorKind::CallBuiltin::validity_panic`):
                // for a type the compiler's validity check
                // accepts, code generation makes the assertion a
                // no-op; for one it rejects, code generation
                // calls `panic_nounwind` with the baked message,
                // whose report and abort the interpreter
                // reproduces here.
                if let Some(message) = validity_panic {
                    return Err(self.validity_assertion_abort(frames, message));
                }
                Ok(Value::Unit)
            }
            Builtin::OverflowChecks => {
                arity(0)?;
                // The setting of the crate that instantiated
                // the calling body, resolved from the stack:
                // the intrinsic's fallback body is a constant
                // of core's own build and is never collected.
                Ok(Value::Bool(self.overflow_checks_at(frames)))
            }
            Builtin::WriteBytes => {
                arity(3)?;
                let tag = generic_args.first().map(String::as_str).unwrap_or("");
                // The fill argument is a `u8`; take its low eight
                // bits, which are the byte either way it is stored.
                let byte = match &vals[1] {
                    Value::Uint(v) => *v as u8,
                    Value::Int(v) => *v as u8,
                    other => {
                        return Err(InterpError::type_mismatch(
                            format!("builtin {name}"),
                            "fill byte (u8)",
                            other.kind_name(),
                        ));
                    }
                };
                // A baked value was built at conversion for
                // exactly this call's constant fill byte (see
                // `zeroed_value` in `ir`); it covers the shapes
                // the tag-based constructions below cannot
                // describe (enums, pointers, tuples).
                let fill = zeroed_value
                    .cloned()
                    .or_else(|| byte_fill_value(tag, byte))
                    .or_else(|| self.wrapped_byte_fill(tag, byte, 0))
                    .ok_or_else(|| {
                        unsupported(format!(
                            "write_bytes for {tag} is not runtime-supported"
                        ))
                    })?;
                let n = element_count(&vals[2], name)?;
                if n == 0 {
                    // A zero-count fill touches no memory, so the
                    // pointer is never dereferenced.
                    return Ok(Value::Unit);
                }
                let target = self.deref_value(frames, &vals[0])?;
                // Filling not-yet-initialized memory is the
                // intrinsic's canonical use (a `MaybeUninit`
                // buffer being initialized): the region has no
                // prior value to patch element by element, so
                // materialize exactly the filled region in one
                // write. The pointer denotes the start of the
                // uninitialized region in that shape (a fresh
                // buffer's base pointer).
                // A target ending in an index projection denotes
                // one element of an existing aggregate (the shape
                // an array-to-element pointer cast produces), not
                // the start of an unmaterialized region: filling it
                // goes through the sequence access below, which
                // writes the elements at their own positions. (A
                // single-element fill writes the denoted element
                // itself either way, so it keeps this fast path.)
                let ends_with_index = match &target {
                    Resolved::Place { projection, .. }
                    | Resolved::Static { projection, .. }
                    | Resolved::Alloc { projection, .. } => {
                        matches!(projection.last(), Some(RefProj::Index(_)))
                    }
                    Resolved::Value(_, projection) => {
                        matches!(projection.last(), Some(RefProj::Index(_)))
                    }
                };
                if matches!(
                    self.read_resolved(frames, &target),
                    Err(InterpError::UninitializedRead { .. })
                ) && (n == 1 || !ends_with_index)
                {
                    if n == 1 {
                        self.write_resolved(frames, &target, fill)?;
                    } else {
                        self.write_resolved(frames, &target, Value::Tuple(vec![fill; n]))?;
                    }
                    return Ok(Value::Unit);
                }
                // As in the element copy: an unreadable destination
                // is a buffer being filled, which the indexed path
                // can create elements in.
                let dst = self.read_resolved(frames, &target);
                // A fill over a one-field aggregate's storage (a
                // `MaybeUninit` union place: the pointer denotes
                // the payload slot, which shares the place's own
                // address): when the fill has the slot's shape
                // rather than the aggregate's, it is the slot's
                // new content, and writing it whole would strip
                // the wrapper every later field read expects.
                if let Ok(v @ Value::Enum { fields, .. }) = &dst {
                    if fields.len() == 1
                        && !same_value_shape(&fill, v)
                        && same_value_shape(&fill, &fields[0])
                    {
                        let Value::Enum { variant, .. } = v else {
                            unreachable!("guard checked the variant")
                        };
                        let wrapped = Value::Enum {
                            variant: *variant,
                            fields: vec![fill],
                        };
                        self.write_resolved(frames, &target, wrapped)?;
                        return Ok(Value::Unit);
                    }
                }
                // A one-element fill denotes the target whole when
                // the target is not a sequence (a scalar or wrapper
                // place: the fill is its new value), and also when
                // the target is an array whose scalar-leaf count
                // the fill's type exactly matches — the intrinsic's
                // type is then the target's own type (an array
                // filled in one operation, the shape `mem::zeroed`
                // takes under optimisation), and writing the fill
                // as element 0 of a sequence would nest it inside
                // its own first element. Any other fill denotes one
                // element of the target sequence.
                let whole = n == 1
                    && match &dst {
                        Ok(v @ Value::Tuple(_)) => {
                            // The leaf counts agree for the
                            // ordinary shapes. They disagree
                            // for the unit/empty-tuple duality
                            // (the tag counts `()` as one leaf,
                            // the stored empty tuple as none),
                            // and the tag's count is
                            // unmeasurable for an element type
                            // with no scalar leaves (like
                            // `[!; 0]`): in both cases the fill
                            // denotes the target whole exactly
                            // when it has the target's own
                            // shape.
                            (value_leaf_count(v).is_some()
                                && value_leaf_count(v) == tag_leaf_count(tag))
                                || same_value_shape(v, &fill)
                        }
                        Ok(Value::FatRef { .. } | Value::Bytes(_) | Value::Str(_)) => {
                            false
                        }
                        Ok(_) => true,
                        Err(_) => false,
                    };
                if whole {
                    self.write_resolved(frames, &target, fill)?;
                } else {
                    for i in 0..n {
                        self.write_seq_at(frames, &target, i, fill.clone())?;
                    }
                }
                Ok(Value::Unit)
            }
            Builtin::FloatMethod { f32, op } => exec_float_method(name, f32, op, &vals),
            Builtin::Float128Method { op } => exec_float128_method(name, op, &vals),
            Builtin::Float16Method { op } => exec_float16_method(name, op, &vals),
        }
    }

    /// The fill value of a `write_bytes` over a pointee type
    /// [`byte_fill_value`] does not cover, resolved through the
    /// wrappers the byte pattern passes through unchanged:
    ///
    /// * `MaybeUninit<T>`: a byte fill over the union view writes
    ///   `T`'s bytes; the region's canonical value is `T`'s fill
    ///   (the readers of such a region read it as `T` — a raw
    ///   table's control bytes are allocated as
    ///   `MaybeUninit<Tag>` and read as `Tag`).
    /// * a one-field struct (per the program's recorded shape of
    ///   the instance): the struct value holding the field's
    ///   fill — the shape an aggregate construction of the
    ///   wrapper produces (`Tag` over its `u8`).
    ///
    /// Anything else stays unsupported at the call site.
    /// The allocation pointer a `Box` value carries: the
    /// box is the one-field aggregate holding it. A bare
    /// pointer value standing in for the box (the shape a
    /// raw-parts reconstruction produces) is its own
    /// pointer.
    fn box_pointer_of(&self, box_value: &Value) -> InterpResult<Value> {
        match box_value {
            Value::Enum { fields, .. } if fields.len() == 1 => {
                Ok(fields[0].clone())
            }
            ptr @ (Value::Ref(_) | Value::FatRef { .. }) => Ok(ptr.clone()),
            other => Err(InterpError::type_mismatch(
                "IntoIterator::into_iter for a Box",
                "box value",
                other.kind_name(),
            )),
        }
    }

    /// The pointee value a `Box` value denotes: the
    /// pointee lives in the allocation store under the
    /// box's pointer, so reading through the pointer
    /// yields it.
    fn box_pointee_of(
        &self,
        frames: &[Frame],
        box_value: &Value,
    ) -> InterpResult<Value> {
        let ptr = self.box_pointer_of(box_value)?;
        let target = self.deref_value(frames, &ptr)?;
        self.read_resolved(frames, &target)
    }

    /// The `slice::IterMut` state over a container's
    /// elements, in the iterator's real field layout (see
    /// `slice_iter_value`): the `NonNull` first element
    /// pointer, the past-the-end element pointer and the
    /// `PhantomData` marker, all denoting places in the
    /// container's own storage — the state the real
    /// `iter_mut` body builds, which the `slice::IterMut`
    /// steps consume. `container` is the value the
    /// element references are formed against and
    /// `container_tag` names the container type for the
    /// element-type lookup.
    fn slice_iter_mut_state(
        &mut self,
        frames: &mut Vec<Frame>,
        container: &Value,
        len: u128,
        container_tag: &str,
    ) -> InterpResult<Value> {
        // A zero-sized element type stores the iterator's
        // length in the end field instead of a
        // past-the-end pointer: core builds that state
        // with `without_provenance_mut(len)` there, the
        // converted `len` body reads the address back
        // directly, and the `slice::IterMut` steps decode
        // the same form (see `iter_zst_remaining`).
        let zero_sized = slice_container_element_tag(container_tag)
            .and_then(|tag| self.layout_of(tag))
            .is_some_and(|(size, _)| size == 0);
        let (first, end) = if zero_sized {
            // The front pointer still denotes the first
            // element when there is one. An empty
            // zero-sized container has no element to
            // denote and its steps never read the pointer
            // (the stored length is zero), so the
            // container value itself stands in.
            let first = if len == 0 {
                match container {
                    Value::FatRef { base, .. } => (**base).clone(),
                    _ => container.clone(),
                }
            } else {
                self.iter_mut_element_ref(frames, container, 0, Some(len))?
            };
            (first, Value::Ref(RefTarget::Address(len as u64)))
        } else {
            (
                self.iter_mut_element_ref(frames, container, 0, Some(len))?,
                self.iter_mut_element_ref(frames, container, len as usize, Some(len))?,
            )
        };
        Ok(Value::Enum {
            variant: 0,
            fields: vec![
                // `ptr: NonNull<T>` — the one-field
                // wrapper around the raw element pointer.
                Value::Enum {
                    variant: 0,
                    fields: vec![first],
                },
                end,
                // `_marker: PhantomData<&mut T>` — a
                // zero-sized ADT value.
                Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                },
            ],
        })
    }

    /// std's `From` impls for a `Cow<'_, B>` itself, as a
    /// value conversion: from `B`'s owned form (the `Owned`
    /// variant over the value itself), from a `&B` (the
    /// `Borrowed` variant over the reference value — the
    /// shape aggregate construction of `Cow::Borrowed`
    /// stores), and from a reference to the owned form (the
    /// `Borrowed` variant over the owned value's own deref
    /// shape). Any other (borrowed, source) pair is not a
    /// conversion std provides for a Cow and yields `None`.
    pub(super) fn cow_from_value(
        &self,
        frames: &[Frame],
        borrowed: &str,
        source: &str,
        value: Value,
    ) -> InterpResult<Option<Value>> {
        fn cow(variant: u32, field: Value) -> Option<Value> {
            Some(Value::Enum {
                variant,
                fields: vec![field],
            })
        }
        // Whether `source` names the slice type's element
        // vector (`Vec<T>` for a `[T]` borrowed type) under
        // exactly the given prefixes: the owned form's
        // prefixes carry no `&`, the reference form's do,
        // and the two must not be confused (a `&Vec<T>`
        // source wrapped as the owned variant would leave
        // the Cow holding a reference where the vector
        // itself belongs).
        let vec_of_borrowed = |source: &str, prefixes: &[&str]| {
            borrowed.starts_with('[')
                && borrowed.ends_with(']')
                && prefixes
                    .iter()
                    .filter_map(|prefix| generic_first_arg(source, prefix))
                    .next()
                    .is_some_and(|elem| borrowed[1..borrowed.len() - 1] == *elem)
        };
        let owned_form = match borrowed {
            "str" => matches!(source, "std::string::String" | "alloc::string::String"),
            "std::ffi::OsStr" => source == "std::ffi::OsString",
            "std::path::Path" => source == "std::path::PathBuf",
            "std::ffi::CStr" => source == "std::ffi::CString",
            _ => vec_of_borrowed(source, &["std::vec::Vec<", "alloc::vec::Vec<"]),
        };
        if owned_form {
            return Ok(cow(1, value));
        }
        if source == format!("&{borrowed}") {
            return Ok(cow(0, value));
        }
        // A reference to the owned form: the borrowed
        // variant holds the owned value's deref shape.
        let ref_owned_form = match borrowed {
            "str" => matches!(
                source,
                "&std::string::String" | "&alloc::string::String"
            ),
            "std::ffi::OsStr" => source == "&std::ffi::OsString",
            "std::path::Path" => source == "&std::path::PathBuf",
            "std::ffi::CStr" => source == "&std::ffi::CString",
            _ => vec_of_borrowed(source, &["&std::vec::Vec<", "&alloc::vec::Vec<"]),
        };
        if !ref_owned_form {
            return Ok(None);
        }
        let field = match borrowed {
            "str" => self.resolve_value(frames, &value)?,
            "std::ffi::OsStr" => {
                let pointee = self.resolve_value(frames, &value)?;
                let bytes = self
                    .os_string_bytes_any(frames, &pointee)?
                    .ok_or_else(|| {
                        InterpError::type_mismatch(
                            "From::from to Cow<OsStr>",
                            "an OS string value",
                            pointee.kind_name(),
                        )
                    })?;
                sub_path_value(&bytes)
            }
            "std::path::Path" => {
                let pointee = self.resolve_value(frames, &value)?;
                let bytes = self
                    .path_buf_bytes_any(frames, &pointee)?
                    .ok_or_else(|| {
                        InterpError::type_mismatch(
                            "From::from to Cow<Path>",
                            "a path buffer value",
                            pointee.kind_name(),
                        )
                    })?;
                sub_path_value(&bytes)
            }
            "std::ffi::CStr" => self.c_string_deref(frames, &value)?,
            _ => {
                // A slice target: the fat reference over the
                // vector, the shape `Vec`'s own deref
                // produces.
                let pointee = self.resolve_value(frames, &value)?;
                let len = pointee.len().ok_or_else(|| {
                    InterpError::type_mismatch(
                        "From::from to a borrowed Cow slice",
                        "array value",
                        pointee.kind_name(),
                    )
                })?;
                Value::FatRef {
                    base: Box::new(value),
                    len,
                }
            }
        };
        Ok(cow(0, field))
    }

    /// The pointee a box value holds, taken out of the box:
    /// a builtin-built box unboxes directly (its allocation is
    /// released); a box a converted std body built is the
    /// layered wrapper aggregate (a `Unique` over a `NonNull`
    /// over the pointer, the allocator as a sibling field), so
    /// its pointer chain is followed to the pointee instead
    /// \u2014 the counted-slice arm's fallback, generalised. A
    /// fat pointer over a self-contained element tuple resolves
    /// to itself; its base is the pointee.
    pub(super) fn box_pointee(&self, frames: &[Frame], value: &Value) -> Option<Value> {
        if let Some(pointee) = self.unbox_take(value) {
            return Some(pointee);
        }
        let pointer = layered_wrapper_pointer(value)?;
        match self.resolve_value(frames, &pointer) {
            Ok(Value::FatRef { base, .. }) if matches!(*base, Value::Tuple(_)) => Some(*base),
            Ok(pointee) => Some(pointee),
            Err(_) => None,
        }
    }

    /// std's remaining `From` impls for a reference-counted
    /// handle, whose pointee the counted string / counted slice
    /// arms do not serve (see the `Builtin::FromFrom` arm that
    /// calls this): the generic `From<Box<T>>` — the box's
    /// pointee moves into the handle uncloned, the box's
    /// allocation released — and the borrowed / owned forms
    /// of the `OsStr`, `Path` and `CStr` families, whose pointee
    /// is cloned into the handle in the family's stored form
    /// (the byte string for `OsStr` / `Path`, the shape the
    /// boxed OS string stores; the element tuple, terminator
    /// included, for `CStr`, the shape a `CString` buffer and a
    /// counted slice both store). Any other (pointee, source)
    /// pair is not a conversion std provides for a counted
    /// handle and yields `None`.
    pub(super) fn rc_from_value(
        &self,
        frames: &[Frame],
        pointee: &str,
        source: &str,
        value: Value,
    ) -> InterpResult<Option<Value>> {
        // The generic `From<Box<T>>`, gated on the source type
        // naming the box over exactly this pointee: a value's
        // shape alone cannot decide (a vector's value has the
        // same one-field shape as a box aggregate — see the
        // counted-slice arm's note).
        let boxed_source = generic_first_arg(source, "std::boxed::Box<") == Some(pointee)
            || generic_first_arg(source, "alloc::boxed::Box<") == Some(pointee);
        if boxed_source {
            // A `Box<dyn Trait>` source: the box's value is the
            // unsizing cast's `DynRef` over the box aggregate
            // (or carries it at the bottom of its wrapper
            // chain). The handle is built over the concrete
            // pointee the box held and the cast's vtable record
            // moves onto the handle \u2014 the shape the
            // handle-level coercion of an `Rc::new` value
            // produces, whose drop runs through that record.
            let dyn_source = match &value {
                d @ Value::DynRef { .. } => Some(d.clone()),
                _ => match layered_wrapper_pointer(&value) {
                    Some(d @ Value::DynRef { .. }) => Some(d),
                    _ => None,
                },
            };
            if let Some(Value::DynRef {
                base,
                concrete,
                traits,
                drop_glue,
                type_id,
                vtable_methods,
            }) = dyn_source
            {
                return Ok(match self.box_pointee(frames, &base) {
                    Some(pointee) => Some(Value::DynRef {
                        base: Box::new(self.rc_new_handle(pointee)),
                        concrete,
                        traits,
                        drop_glue,
                        type_id,
                        vtable_methods,
                    }),
                    None => None,
                });
            }
            return Ok(self
                .box_pointee(frames, &value)
                .map(|pointee| self.rc_new_handle(pointee)));
        }
        match pointee {
            "std::ffi::OsStr" => {
                if source == "&std::ffi::OsStr" {
                    // The borrowed form: the bytes are cloned
                    // into the handle; the source is not
                    // consumed.
                    let bytes = self.os_str_bytes(frames, &value)?;
                    Ok(Some(self.rc_new_handle(Value::Bytes(bytes))))
                } else if source == "std::ffi::OsString" {
                    let bytes = self.os_string_bytes_any(frames, &value)?.ok_or_else(|| {
                        InterpError::type_mismatch(
                            "From::from to a counted OsStr",
                            "an OS string value",
                            value.kind_name(),
                        )
                    })?;
                    Ok(Some(self.rc_new_handle(Value::Bytes(bytes))))
                } else {
                    Ok(None)
                }
            }
            "std::path::Path" => {
                if source == "&std::path::Path" {
                    let bytes = self.path_bytes(frames, &value)?;
                    Ok(Some(self.rc_new_handle(Value::Bytes(bytes))))
                } else if source == "std::path::PathBuf" {
                    let bytes = self.path_buf_bytes_any(frames, &value)?.ok_or_else(|| {
                        InterpError::type_mismatch(
                            "From::from to a counted Path",
                            "a path buffer value",
                            value.kind_name(),
                        )
                    })?;
                    Ok(Some(self.rc_new_handle(Value::Bytes(bytes))))
                } else {
                    Ok(None)
                }
            }
            "std::ffi::CStr" => {
                // `arg_bytes` reads the C string's slice,
                // terminator included, in both source shapes:
                // the borrowed fat reference directly, and the
                // owned `CString` through its deref shape.
                let bytes = if source == "&std::ffi::CStr" {
                    self.arg_bytes(frames, &value)?
                } else if source == "std::ffi::CString" {
                    let fat = self.c_string_deref(frames, &value)?;
                    self.arg_bytes(frames, &fat)?
                } else {
                    return Ok(None);
                };
                let elems: Vec<Value> = bytes
                    .iter()
                    .map(|b| Value::Uint(u128::from(*b)))
                    .collect();
                Ok(Some(self.rc_new_handle(Value::Tuple(elems))))
            }
            _ => Ok(None),
        }
    }

    /// The `IntoIterator::into_iter` value transformation: the
    /// iterator state the value iterates as, in the shape the
    /// modelled `Iterator::next` steps consume. Split out of
    /// the `Builtin::IntoIter` arm so the modelled iterator
    /// steps that pull an inner iterator out of an outer item
    /// (the flatten composition) perform the same conversion
    /// the guest's own `into_iter` call performs.
    pub(super) fn into_iter_value(
        &mut self,
        frames: &mut Vec<Frame>,
        self_ty: &str,
        value: Value,
    ) -> InterpResult<Value> {
        // A trait-object iterator value (a `Box<dyn
        // Iterator>` or `&dyn Iterator`) is already an
        // iterator, so the blanket identity below covers
        // it whatever its rendered Self type.
        if matches!(value, Value::DynRef { .. })
            || self_ty.starts_with("std::ops::Range<")
            || self_ty.starts_with("std::ops::RangeInclusive<")
            || self_ty.starts_with("std::ops::RangeFrom<")
            || self_ty.starts_with("std::range::RangeFromIter<")
            || self_ty.starts_with("std::iter::StepBy<")
            || self_ty.starts_with("std::iter::Rev<")
            || self_ty.starts_with("std::slice::Iter<")
            || self_ty.starts_with("std::collections::btree_set::Iter<")
            || self_ty.starts_with("std::collections::btree_map::Iter<")
            || self_ty.starts_with("std::collections::btree_map::Keys<")
            || self_ty.starts_with("std::collections::btree_map::Values<")
            || self_ty.starts_with("std::collections::btree_set::IntoIter<")
            || self_ty.starts_with("std::collections::btree_map::IntoIter<")
            || self_ty.starts_with("std::slice::IterMut<")
            || self_ty.starts_with("std::array::IntoIter<")
            || self_ty.starts_with("std::boxed::BoxedArrayIntoIter<")
            || self_ty.starts_with("std::vec::IntoIter<")
            || self_ty.starts_with("alloc::vec::IntoIter<")
            || self_ty.starts_with("std::vec::Drain<")
            || self_ty.starts_with("alloc::vec::Drain<")
            || self_ty.starts_with("std::option::IntoIter<")
            || self_ty.starts_with("std::option::Iter<")
            || self_ty.starts_with("std::option::IterMut<")
            || self_ty.starts_with("std::iter::Map<")
            || self_ty.starts_with("std::iter::Enumerate<")
            || self_ty.starts_with("std::iter::Copied<")
            || self_ty.starts_with("std::iter::Cloned<")
            || self_ty.starts_with("std::iter::Peekable<")
            || self_ty.starts_with("std::iter::Take<")
            || self_ty.starts_with("std::iter::Skip<")
            || self_ty.starts_with("std::iter::Chain<")
            || self_ty.starts_with("std::iter::Zip<")
            || self_ty.starts_with("std::iter::SkipWhile<")
            || self_ty.starts_with("std::iter::Filter<")
            || self_ty.starts_with("std::iter::FilterMap<")
            || self_ty.starts_with("std::iter::FlatMap<")
            || self_ty.starts_with("std::iter::Flatten<")
            || self_ty.starts_with("std::iter::adapters::flatten::FlattenCompat<")
            || self_ty.starts_with("std::io::Lines<")
            || self_ty.starts_with("std::io::Bytes<")
            || self_ty_is(self_ty, "std::str::Lines")
            || self_ty_is(self_ty, "std::str::LinesInclusive")
            || self_ty.starts_with("std::str::Split<")
            || self_ty.starts_with("std::str::RSplit<")
            || self_ty.starts_with("std::str::SplitTerminator<")
            || self_ty.starts_with("std::str::RSplitTerminator<")
            || self_ty.starts_with("std::str::SplitInclusive<")
            || self_ty.starts_with("std::iter::Once<")
            || self_ty.starts_with("std::path::Components")
            || self_ty.starts_with("std::path::Ancestors")
            || self_ty_is(self_ty, "std::fs::ReadDir")
            || self_ty.starts_with("std::iter::Empty<")
            || self_ty.starts_with("std::iter::Repeat<")
            || self_ty_is(self_ty, "std::str::Chars")
            || self_ty_is(self_ty, "core::str::Chars")
            || self_ty_is(self_ty, "std::str::Bytes")
            || self_ty_is(self_ty, "core::str::Bytes")
            || self_ty_is(self_ty, "std::str::EscapeDebug")
            || self_ty_is(self_ty, "core::str::EscapeDebug")
            || self_ty_is(self_ty, "std::str::EscapeDefault")
            || self_ty_is(self_ty, "core::str::EscapeDefault")
            || self_ty_is(self_ty, "std::str::EscapeUnicode")
            || self_ty_is(self_ty, "core::str::EscapeUnicode")
                    || self_ty_is(self_ty, "std::char::EscapeDebug")
            || self_ty_is(self_ty, "core::char::EscapeDebug")
            || self_ty_is(self_ty, "std::char::EscapeDefault")
            || self_ty_is(self_ty, "core::char::EscapeDefault")
            || self_ty_is(self_ty, "std::char::EscapeUnicode")
            || self_ty_is(self_ty, "core::char::EscapeUnicode")
            || self_ty.starts_with("std::collections::linked_list::Iter<")
            || self_ty.starts_with("std::collections::linked_list::IterMut<")
            || self_ty.starts_with("std::collections::linked_list::IntoIter<")
            || self_ty.starts_with("alloc::collections::linked_list::Iter<")
            || self_ty.starts_with("alloc::collections::linked_list::IterMut<")
            || self_ty.starts_with("alloc::collections::linked_list::IntoIter<")
            || self_ty.starts_with("std::collections::vec_deque::Iter<")
            || self_ty.starts_with("std::collections::vec_deque::IterMut<")
            || self_ty.starts_with("std::collections::vec_deque::IntoIter<")
            || self_ty.starts_with("alloc::collections::vec_deque::Iter<")
            || self_ty.starts_with("alloc::collections::vec_deque::IterMut<")
            || self_ty.starts_with("alloc::collections::vec_deque::IntoIter<")
            || self_ty.starts_with("std::collections::binary_heap::Iter<")
            || self_ty.starts_with("std::collections::binary_heap::IntoIter<")
            || self_ty.starts_with("std::slice::Chunks<")
            || self_ty.starts_with("std::slice::ChunksExact<")
            || self_ty.starts_with("std::slice::RChunks<")
            || self_ty.starts_with("std::slice::RChunksExact<")
            || self_ty.starts_with("std::slice::ChunksMut<")
            || self_ty.starts_with("std::slice::ChunksExactMut<")
            || self_ty.starts_with("std::slice::RChunksMut<")
            || self_ty.starts_with("std::slice::RChunksExactMut<")
            || self_ty.starts_with("std::slice::Split<")
            || self_ty.starts_with("core::slice::Split<")
            || self_ty.starts_with("std::slice::SplitInclusive<")
            || self_ty.starts_with("core::slice::SplitInclusive<")
            || self_ty.starts_with("std::slice::RSplit<")
            || self_ty.starts_with("core::slice::RSplit<")
            || self_ty_is(self_ty, "std::str::SplitAsciiWhitespace")
            || self_ty_is(self_ty, "core::slice::SplitAsciiWhitespace")
        {
            // These types are their own iterator: the
            // blanket `IntoIterator for I: Iterator` impl
            // is the identity, and the value is already in
            // the shape `Iterator::next` consumes for the
            // type — a range is its own state, a slice /
            // char / byte iterator was built by the
            // corresponding runtime support, an adapter or
            // source iterator by its constructor's
            // converted body, and the owning iterators by
            // the arms below.
            Ok(value)
        } else if self_ty.starts_with("std::range::RangeFrom<") {
            // The new range API's `RangeFrom` iterates as
            // `RangeFromIter { start, exhausted: false }` --
            // the two-field state core's own `into_iter`
            // builds and the modelled `RangeFromIter` step
            // (and the converted `remainder` body) consumes.
            let fields = iterator_fields(&value, self_ty)?;
            if fields.len() != 1 {
                return Err(InterpError::type_mismatch(
                    "IntoIterator::into_iter for a range::RangeFrom",
                    "[start]",
                    format!("{} field(s)", fields.len()),
                ));
            }
            Ok(Value::Enum {
                variant: 0,
                fields: vec![fields[0].clone(), Value::Bool(false)],
            })
        } else if self_ty.starts_with('[') {
            // Array by value: iteration state is the element
            // tuple plus the next index (our layout for
            // `std::array::IntoIter`, whose `next` below is
            // the only consumer).
            match value {
                Value::Tuple(items) => Ok(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Tuple(items), Value::Uint(0)],
                }),
                ref other => Err(InterpError::type_mismatch(
                    "IntoIterator::into_iter for an array",
                    "array value",
                    other.kind_name(),
                )),
            }
        } else if self_ty.starts_with("&std::vec::Vec<")
            || self_ty.starts_with("&alloc::vec::Vec<")
            || self_ty.starts_with("&[")
        {
            // `&Vec<T>` and `&[T]` iterate as `slice::Iter`,
            // in the real field layout `slice_iter_value`
            // builds and `Iterator::next` consumes. A
            // scalar-backed slice (the `slice::from_ref`
            // shape) reads its elements from the scalar's
            // own storage. A fat pointer whose formation
            // regrouped its storage's elements iterates
            // under the element type the slice type names,
            // not the stored shape (see
            // `regrouped_iter_elements`).
            let elem_ty = match ty_from_tag(super::fmt::pointee_tag(self_ty)) {
                Some(Ty::Slice(elem)) | Some(Ty::Array(elem, _)) => Some(*elem),
                _ => slice_container_element_tag(self_ty)
                    .or_else(|| {
                        slice_container_element_tag(
                            self_ty.strip_prefix('&').unwrap_or(self_ty),
                        )
                    })
                    .and_then(ty_from_tag),
            };
            let regrouped = elem_ty
                .as_ref()
                .and_then(|elem_ty| {
                    self.regrouped_iter_elements(frames, &value, elem_ty)
                })
                .map(|elems| {
                    let len = elems.len() as u128;
                    (elems, len)
                });
            let elem_tag = slice_container_element_tag(self_ty)
                .or_else(|| {
                    slice_container_element_tag(
                        self_ty.strip_prefix('&').unwrap_or(self_ty),
                    )
                })
                .unwrap_or("");
            // As in `Builtin::SliceIter`: the scalar-backed
            // reading, under the element kind the slice type
            // names, is consulted ahead of the plain element
            // reading.
            let scalar_backed = self
                .scalar_backed_slice_elements(frames, &value, elem_tag)
                .map(|elems| {
                    let len = elems.len() as u128;
                    (elems, len)
                });
            let mut storage_state = None;
            let (elems, _) = match regrouped {
                Some(elems) => elems,
                None => match scalar_backed {
                    Some(elems) => elems,
                    None => match self.slice_elements(frames, &value) {
                        Ok(elems) => {
                            storage_state = self
                                .slice_iter_storage_state(frames, &value, &elems.0);
                            elems
                        }
                        Err(err) => return Err(err),
                    },
                },
            };
            Ok(match storage_state {
                Some(state) => state,
                None => slice_iter_value(elems),
            })
        } else if self_ty.starts_with("&std::collections::BTreeSet<") {
            self.btree_set_iter_value(frames, &value)
        } else if self_ty.starts_with("&std::collections::BTreeMap<") {
            self.btree_map_iter_value(frames, &value)
        } else if self_ty_is(self_ty, "std::env::Args")
            || self_ty_is(self_ty, "std::env::ArgsOs")
            || self_ty_is(self_ty, "std::env::Vars")
            || self_ty_is(self_ty, "std::env::VarsOs")
            || self_ty_is(self_ty, "std::env::SplitPaths")
            || self_ty.starts_with("std::slice::Windows<")
        {
            // The env argument and environment iterators are
            // their own `IntoIterator` (the states `env::args`
            // and `env::vars` built).
            Ok(value)
        } else if self_ty.starts_with("std::boxed::Box<")
            || self_ty.starts_with("alloc::boxed::Box<")
        {
            // A by-value `Box<[T; N]>` iterates as
            // `array::IntoIter` over the pointee array and
            // a by-value `Box<[T]>` as `vec::IntoIter` over
            // the pointee elements: the states the array
            // and Vec arms build, from the sequence the
            // box's allocation holds.
            let Some(inner) = box_sequence_tag(self_ty) else {
                return Err(unsupported(format!(
                    "IntoIterator::into_iter for {self_ty} is not runtime-supported"
                )));
            };
            let pointee = self.box_pointee_of(frames, &value)?;
            if sequence_tag_is_array(inner) {
                let items = match &pointee {
                    Value::Tuple(items) => items.clone(),
                    _ => value_elements(&pointee)?,
                };
                Ok(Value::Enum {
                    variant: 0,
                    fields: vec![Value::Tuple(items), Value::Uint(0)],
                })
            } else {
                Ok(owned_iter_state(value_elements(&pointee)?))
            }
        } else if self_ty.starts_with("std::vec::Vec<")
            || self_ty.starts_with("alloc::vec::Vec<")
        {
            // A by-value `Vec` iterates as `vec::IntoIter`:
            // the element snapshot plus the next index (the
            // vector value's fields are its elements; the
            // sparse form a growth or a shrink leaves
            // behind contributes its declared elements).
            match value {
                Value::Enum { fields, .. } => Ok(owned_iter_state(fields)),
                v @ Value::SparseVec { .. } => {
                    Ok(owned_iter_state(value_elements(&v)?))
                }
                ref other => Err(InterpError::type_mismatch(
                    "IntoIterator::into_iter for a Vec",
                    "vector value",
                    other.kind_name(),
                )),
            }
        } else if self_ty.starts_with("std::collections::BTreeSet<") {
            let elems = self.btree_set_elements(frames, &value)?;
            Ok(owned_iter_state(elems))
        } else if self_ty.starts_with("std::collections::BTreeMap<") {
            let elems = self.btree_map_pair_elements(frames, &value)?;
            Ok(owned_iter_state(elems))
        } else if self_ty.starts_with("std::sync::mpsc::Iter<") {
            // `mpsc::Iter` is its own `IntoIterator`.
            Ok(value)
        } else if self_ty.starts_with("std::sync::mpsc::TryIter<") {
            // `mpsc::TryIter` is its own `IntoIterator`.
            Ok(value)
        } else if self_ty.starts_with("std::sync::mpsc::Receiver<") {
            // The channel iterator holds the receiver
            // handle as its one field (see `iterator_next`).
            Ok(Value::Enum {
                variant: 0,
                fields: vec![value],
            })
        } else if self_ty.starts_with("std::option::Option<")
            || self_ty.starts_with("core::option::Option<")
        {
            // An `Option<T>` by value iterates as
            // `option::IntoIter` over zero or one element,
            // in the same owning-iterator state shape.
            let elems = match &value {
                Value::Enum {
                    variant: 1,
                    fields,
                } => fields.clone(),
                Value::Enum { variant: 0, .. } => Vec::new(),
                other => {
                    return Err(InterpError::type_mismatch(
                        "IntoIterator::into_iter for an Option",
                        "option value",
                        other.kind_name(),
                    ))
                }
            };
            Ok(owned_iter_state(elems))
        } else if let Some(inner) = self_ty.strip_prefix("&mut ") {
            if inner.starts_with("std::vec::Vec<")
                || inner.starts_with("alloc::vec::Vec<")
                || inner.starts_with('[')
            {
                // `&mut Vec<T>`, `&mut [T; N]` and `&mut [T]`
                // iterate as `slice::IterMut`, in the
                // iterator's real field layout (see
                // `slice_iter_value`): the `NonNull` first
                // element pointer, the past-the-end element
                // pointer and the `PhantomData` marker, all
                // denoting places in the container's own
                // storage — the same state the real
                // `iter_mut` body builds, which the
                // `slice::IterMut` steps consume.
                let len = match &value {
                    Value::FatRef { len, .. } => *len,
                    _ => {
                        let target = self.deref_value(frames, &value)?;
                        let pointee = self.read_resolved(frames, &target)?;
                        value_elements(&pointee)?.len() as u128
                    }
                };
                self.slice_iter_mut_state(frames, &value, len, inner)
            } else if inner.starts_with("std::boxed::Box<")
                || inner.starts_with("alloc::boxed::Box<")
            {
                // `&mut Box<[T]>` and `&mut Box<[T; N]>`
                // iterate as `slice::IterMut` over the
                // pointee's elements: the construction is
                // the Vec / array one above with the box's
                // own pointer as the container, since the
                // element references denote positions in
                // the box's allocation.
                let Some(seq) = box_sequence_tag(inner) else {
                    return Err(unsupported(format!(
                        "IntoIterator::into_iter for {self_ty} is not runtime-supported"
                    )));
                };
                let target = self.deref_value(frames, &value)?;
                let box_value = self.read_resolved(frames, &target)?;
                let ptr = self.box_pointer_of(&box_value)?;
                let pointee = self.box_pointee_of(frames, &box_value)?;
                let len = value_elements(&pointee)?.len() as u128;
                self.slice_iter_mut_state(frames, &ptr, len, seq)
            } else if inner.starts_with("std::option::Option<")
                || inner.starts_with("core::option::Option<")
            {
                self.option_ref_into_iter(frames, &value, true)
            } else if ref_pointee_is_container(inner) {
                Err(unsupported(format!(
                    "IntoIterator::into_iter for {self_ty} is not runtime-supported"
                )))
            } else {
                // The blanket `IntoIterator for &mut I`
                // (I: Iterator): the reference itself is
                // the iterator and is returned unchanged,
                // the identity the blanket body performs.
                // A container pointee, whose own `&mut`
                // impl yields a distinct IterMut type,
                // keeps the unsupported error above.
                Ok(value)
            }
        } else if self_ty.starts_with("&std::boxed::Box<")
            || self_ty.starts_with("&alloc::boxed::Box<")
        {
            // `&Box<[T]>` and `&Box<[T; N]>` iterate as
            // `slice::Iter` over the pointee's elements:
            // the state the `&[T]` arm builds, from the
            // sequence the box's allocation holds.
            let bare = self_ty.strip_prefix('&').unwrap_or(self_ty);
            if box_sequence_tag(bare).is_none() {
                return Err(unsupported(format!(
                    "IntoIterator::into_iter for {self_ty} is not runtime-supported"
                )));
            }
            let target = self.deref_value(frames, &value)?;
            let box_value = self.read_resolved(frames, &target)?;
            let pointee = self.box_pointee_of(frames, &box_value)?;
            Ok(slice_iter_value(value_elements(&pointee)?))
        } else if self_ty.starts_with("&std::option::Option<")
            || self_ty.starts_with("&core::option::Option<")
        {
            self.option_ref_into_iter(frames, &value, false)
        } else if let Some(inner) = self_ty.strip_prefix('&') {
            if ref_pointee_is_container(inner) {
                Err(unsupported(format!(
                    "IntoIterator::into_iter for {self_ty} is not runtime-supported"
                )))
            } else {
                // The blanket `IntoIterator for &I`
                // (I: Iterator): the reference itself is
                // the iterator, returned unchanged.
                Ok(value)
            }
        } else {
            Err(unsupported(format!(
                "IntoIterator::into_iter for {self_ty} is not runtime-supported"
            )))
        }
    }

    /// The `From` conversion a guest struct's own impl
    /// performs when the impl is the struct construction
    /// itself (`#[derive(From)]`, and manual impls of the
    /// same shape): the converted value fills the struct's
    /// one real field, and every remaining field is a
    /// `PhantomData` marker, whose value is the empty
    /// aggregate. (The recorded shape renders a generic
    /// field's type with the parameter's own name, so the
    /// field is identified structurally -- the single
    /// non-marker field -- not by comparing type names.)
    /// `None` for any other target: std/core/alloc targets
    /// are the modelled conversions in the dispatch above,
    /// enum and union targets have no such impl, and a
    /// struct with a further real field could be doing
    /// anything with the value.
    fn guest_struct_from(&self, target: &str, value: Value) -> Option<Value> {
        if target.starts_with("std::")
            || target.starts_with("core::")
            || target.starts_with("alloc::")
        {
            return None;
        }
        let shape = self.program().adt_debug_shapes.get(target)?;
        if shape.kind != mir::AdtShapeKind::Struct || shape.variants.len() != 1 {
            return None;
        }
        let mut value = Some(value);
        let mut fields = Vec::with_capacity(shape.variants[0].fields.len());
        for field in &shape.variants[0].fields {
            let ty = field.ty.trim();
            if ty.starts_with("std::marker::PhantomData<")
                || ty.starts_with("core::marker::PhantomData<")
            {
                fields.push(Value::Enum {
                    variant: 0,
                    fields: Vec::new(),
                });
            } else if value.is_some() {
                fields.push(value.take().expect("checked above"));
            } else {
                return None;
            }
        }
        if value.is_some() {
            return None;
        }
        Some(Value::Enum {
            variant: 0,
            fields,
        })
    }

    fn wrapped_byte_fill(&self, tag: &str, byte: u8, depth: u32) -> Option<Value> {
        if depth > 4 {
            return None;
        }
        let tag = tag.trim();
        if let Some(inner) = generic_first_arg(tag, "std::mem::MaybeUninit<")
            .or_else(|| generic_first_arg(tag, "core::mem::MaybeUninit<"))
        {
            return byte_fill_value(inner, byte)
                .or_else(|| self.wrapped_byte_fill(inner, byte, depth + 1));
        }
        let shape = self.program().adt_debug_shapes.get(tag)?;
        if shape.kind != mir::AdtShapeKind::Struct || shape.variants.len() != 1 {
            return None;
        }
        let variant = &shape.variants[0];
        if variant.fields.len() != 1 {
            return None;
        }
        let field_ty = variant.fields[0].ty.as_str();
        let field_fill = byte_fill_value(field_ty, byte)
            .or_else(|| self.wrapped_byte_fill(field_ty, byte, depth + 1))?;
        Some(Value::Enum {
            variant: 0,
            fields: vec![field_fill],
        })
    }

    /// Write an atomic cell's new scalar, preserving the wrapper
    /// aggregates the cell place holds (`UnsafeCell` inside the
    /// atomic struct): the pointer names the wrapper place, so a
    /// bare scalar write would replace the aggregate and later
    /// whole-place reads of the cell would see the wrong shape.
    /// When the current contents cannot be read (an uninitialised
    /// cell), the value is written as-is.
    /// Evaluate `FromStr::from_str` for a primitive `Self`
    /// type. The result is the `Result` aggregate: variant 0
    /// (`Ok`) with the parsed value, variant 1 (`Err`) with
    /// the error value core constructs — a one-field struct
    /// holding the error-kind enum.
    fn exec_from_str(
        &self,
        tag: &str,
        text: &str,
        name: &str,
    ) -> InterpResult<Value> {
        fn ok(v: Value) -> Value {
            Value::Enum {
                variant: 0,
                fields: vec![v],
            }
        }
        fn err(kind: u32) -> Value {
            Value::Enum {
                variant: 1,
                fields: vec![Value::Enum {
                    variant: 0,
                    fields: vec![Value::Enum {
                        variant: kind,
                        fields: vec![],
                    }],
                }],
            }
        }
        if let Some((bits, signed)) = int_tag(tag) {
            use std::num::IntErrorKind as K;
            let parsed: Result<u128, u32> = if signed {
                macro_rules! go {
                    ($t:ty) => {
                        match text.parse::<$t>() {
                            Ok(v) => Ok(v as i128 as u128),
                            Err(e) => Err(match e.kind() {
                                K::Empty => 0,
                                K::InvalidDigit => 1,
                                K::PosOverflow => 2,
                                K::NegOverflow => 3,
                                K::Zero => 4,
                                _ => 1,
                            }),
                        }
                    };
                }
                match bits {
                    8 => go!(i8),
                    16 => go!(i16),
                    32 => go!(i32),
                    64 => go!(i64),
                    _ => go!(i128),
                }
            } else {
                macro_rules! go {
                    ($t:ty) => {
                        match text.parse::<$t>() {
                            Ok(v) => Ok(v as u128),
                            Err(e) => Err(match e.kind() {
                                K::Empty => 0,
                                K::InvalidDigit => 1,
                                K::PosOverflow => 2,
                                K::NegOverflow => 3,
                                K::Zero => 4,
                                _ => 1,
                            }),
                        }
                    };
                }
                match bits {
                    8 => go!(u8),
                    16 => go!(u16),
                    32 => go!(u32),
                    64 => go!(u64),
                    _ => go!(u128),
                }
            };
            return Ok(match parsed {
                Ok(p) => ok(if signed {
                    Value::Int(sign_extend(p, bits))
                } else {
                    Value::Uint(p)
                }),
                Err(kind) => err(kind),
            });
        }
        match tag {
            "f32" | "f64" => {
                let parsed = if tag == "f32" {
                    text.parse::<f32>().map(|v| Value::Float(f64::from(v)))
                } else {
                    text.parse::<f64>().map(Value::Float)
                };
                Ok(match parsed {
                    Ok(v) => ok(v),
                    // `ParseFloatError` distinguishes only the
                    // empty input from everything else.
                    Err(_) => err(if text.is_empty() { 0 } else { 1 }),
                })
            }
            "bool" => Ok(match text.parse::<bool>() {
                Ok(v) => ok(Value::Bool(v)),
                // `ParseBoolError` is a fieldless struct.
                Err(_) => Value::Enum {
                    variant: 1,
                    fields: vec![Value::Enum {
                        variant: 0,
                        fields: vec![],
                    }],
                },
            }),
            "char" => Ok(match text.parse::<char>() {
                Ok(v) => ok(Value::Uint(u128::from(v as u32))),
                // `ParseCharError` distinguishes only the empty
                // input from too-many-chars.
                Err(_) => err(if text.is_empty() { 0 } else { 1 }),
            }),
            _ => Err(unsupported(format!(
                "{name}: from_str for {tag} is not runtime-supported"
            ))),
        }
    }
}
