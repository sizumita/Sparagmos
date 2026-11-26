//! Free helpers shared by the converter: type/operand utilities and impl-key parsing.


use sparagmos_mir as mir;

use crate::ir;



// ---------------------------------------------------------------------------

/// Preserve one MIR unwind action in the IR (see [`ir::UnwindAction`]).
pub(super) fn convert_unwind_action(action: &mir::UnwindAction) -> ir::UnwindAction {
    match action {
        mir::UnwindAction::Cleanup(bb) => {
            ir::UnwindAction::Cleanup(ir::BasicBlock(bb.index()))
        }
        mir::UnwindAction::Continue => ir::UnwindAction::Continue,
        mir::UnwindAction::Terminate(reason) => {
            ir::UnwindAction::Terminate(match reason {
                mir::UnwindTerminateReason::InCleanup => {
                    ir::UnwindTerminateReason::InCleanup
                }
                mir::UnwindTerminateReason::Generic => {
                    ir::UnwindTerminateReason::CannotUnwind
                }
            })
        }
        mir::UnwindAction::Unreachable => ir::UnwindAction::Unreachable,
    }
}

pub(super) fn convert_int_ty(ty: mir::IntTy) -> ir::IntTy {
    match ty {
        mir::IntTy::I8 => ir::IntTy::I8,
        mir::IntTy::I16 => ir::IntTy::I16,
        mir::IntTy::I32 => ir::IntTy::I32,
        mir::IntTy::I64 => ir::IntTy::I64,
        mir::IntTy::I128 => ir::IntTy::I128,
        mir::IntTy::Isize => ir::IntTy::Isize,
    }
}

pub(super) fn convert_uint_ty(ty: mir::UintTy) -> ir::UintTy {
    match ty {
        mir::UintTy::U8 => ir::UintTy::U8,
        mir::UintTy::U16 => ir::UintTy::U16,
        mir::UintTy::U32 => ir::UintTy::U32,
        mir::UintTy::U64 => ir::UintTy::U64,
        mir::UintTy::U128 => ir::UintTy::U128,
        mir::UintTy::Usize => ir::UintTy::Usize,
    }
}

/// Whether a `mir` binary operator is a comparison: its result is
/// `bool`, so its destination type is not its operand type.
pub(super) fn is_comparison(op: mir::BinOp) -> bool {
    use mir::BinOp as M;
    matches!(op, M::Eq | M::Ne | M::Lt | M::Le | M::Gt | M::Ge)
}

/// The shared interpretation for two integer literals whose operation
/// has no declared operand type anywhere (a comparison of two
/// literals): the integer type of their common storage width — signed
/// when either bit pattern has the width's sign bit set (the patterns
/// rustc's expanded checks compare against, `-1` and the type
/// minimum, both do), unsigned otherwise. `None` when the two operands
/// are not integer literals of one standard width.
/// The type a literal operand's producer recorded for it, if any.
pub(super) fn constant_recorded_ty(op: &mir::Operand) -> Option<mir::Ty> {
    match op {
        mir::Operand::Constant(c) => c.ty.clone(),
        _ => None,
    }
}

pub(super) fn literal_pair_ty(lhs: &mir::Operand, rhs: &mir::Operand) -> Option<mir::Ty> {
    fn scalar_int(op: &mir::Operand) -> Option<(u128, u8)> {
        let mir::Operand::Constant(c) = op else {
            return None;
        };
        let mir::ConstKind::Value(mir::ConstValue::Scalar(mir::Scalar::Int { data, size })) =
            &c.literal
        else {
            return None;
        };
        Some((*data, *size))
    }
    let (ldata, lsize) = scalar_int(lhs)?;
    let (rdata, rsize) = scalar_int(rhs)?;
    if lsize != rsize {
        return None;
    }
    let bits = u32::from(lsize) * 8;
    let sign_bit = |data: u128| bits > 0 && (data >> (bits - 1)) & 1 == 1;
    if sign_bit(ldata) || sign_bit(rdata) {
        int_ty_of_bits(bits).map(mir::Ty::Int)
    } else {
        uint_ty_of_bits(bits).map(mir::Ty::Uint)
    }
}

fn int_ty_of_bits(bits: u32) -> Option<mir::IntTy> {
    Some(match bits {
        8 => mir::IntTy::I8,
        16 => mir::IntTy::I16,
        32 => mir::IntTy::I32,
        64 => mir::IntTy::I64,
        128 => mir::IntTy::I128,
        _ => return None,
    })
}

fn uint_ty_of_bits(bits: u32) -> Option<mir::UintTy> {
    Some(match bits {
        8 => mir::UintTy::U8,
        16 => mir::UintTy::U16,
        32 => mir::UintTy::U32,
        64 => mir::UintTy::U64,
        128 => mir::UintTy::U128,
        _ => return None,
    })
}

/// The typing context for one argument of a runtime-support call.
/// Most builtin arguments are places or string/byte constants, whose
/// conversion does not depend on a context; the exceptions are
/// literal arguments whose type the callee's signature pins to a
/// generic parameter (`RangeInclusive::new`'s bounds are its `Idx`)
/// or to a fixed type (`Argument::from_usize`'s count is a
/// `usize`, passed by reference).
/// The parameter types of a callable place type, for typing call
/// arguments whose literals carry no type of their own: a function
/// pointer's inputs directly; a closure's parameters through its
/// signature (whose single RustCall input is the parameter tuple).
/// Empty when the type carries no signature (function items).
pub(super) fn callable_param_tys(ty: Option<&mir::Ty>) -> Vec<mir::Ty> {
    match ty {
        Some(mir::Ty::FnPtr(sig)) => sig.inputs.clone(),
        Some(mir::Ty::Closure(_, args)) => match args.get(1) {
            Some(mir::Ty::FnPtr(sig)) => match sig.inputs.first() {
                Some(mir::Ty::Tuple(ts)) => ts.clone(),
                _ => Vec::new(),
            },
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

/// Whether an operand is the caller-location stamp the driver
/// appends to a `#[track_caller]` call: a constant reference to
/// the `Location` aggregate — its (file, line, column) fields,
/// with line and column 4-byte integers.
pub(super) fn is_caller_location_stamp(op: &mir::Operand) -> bool {
    let mir::Operand::Constant(c) = op else {
        return false;
    };
    let mir::ConstKind::Value(mir::ConstValue::Ref(inner)) = &c.literal else {
        return false;
    };
    let mir::ConstValue::Adt { fields, .. } = &**inner else {
        return false;
    };
    let is_u32 = |v: &mir::ConstValue| {
        matches!(v, mir::ConstValue::Scalar(mir::Scalar::Int { size: 4, .. }))
    };
    fields.len() >= 3 && is_u32(&fields[1]) && is_u32(&fields[2])
}

pub(super) fn builtin_arg_ctx(
    kind: crate::builtins::Builtin,
    raw_generic_args: &[mir::Ty],
    arg_index: usize,
) -> Option<mir::Ty> {
    use crate::builtins::Builtin;
    match kind {
        Builtin::RangeInclusiveNew => {
            if arg_index < 2 {
                raw_generic_args.first().cloned()
            } else {
                None
            }
        }
        Builtin::ArgumentFromUsize => Some(mir::Ty::Uint(mir::UintTy::Usize)),
        // Lane V: the value arguments of these builtins are the
        // callee's first generic parameter (the element / pointee /
        // cell type), so a literal argument takes that type.
        Builtin::BoxNew | Builtin::ManuallyDropNew | Builtin::CellNew | Builtin::RefCellNew => {
            if arg_index == 0 {
                raw_generic_args.first().cloned()
            } else {
                None
            }
        }
        Builtin::CellSet | Builtin::CellReplace | Builtin::VecPush | Builtin::RefCellReplace => {
            if arg_index == 1 {
                raw_generic_args.first().cloned()
            } else {
                None
            }
        }
        // The iterator consumers take the iterator itself, the
        // accumulator and the closure as their arguments: each is
        // typed by the callee's corresponding generic parameter
        // (Self, B, F for fold; Self, F for for_each; Self for
        // count), so a literal accumulator takes the accumulator
        // type instead of the default unsigned rendering.
        Builtin::IteratorCount
        | Builtin::IteratorFold
        | Builtin::IteratorTryFold
        | Builtin::IteratorForEach
        | Builtin::IteratorCollect => raw_generic_args.get(arg_index).cloned(),
        Builtin::FromIteratorFromIter => {
            if arg_index == 0 {
                raw_generic_args.get(2).cloned()
            } else {
                None
            }
        }
        // The index argument of `Index` / `IndexMut` is the trait's
        // `Idx` parameter, the callee's second generic argument.
        Builtin::IndexIndex | Builtin::IndexMut => {
            if arg_index == 1 {
                raw_generic_args.get(1).cloned()
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Whether the context type is a pointer (a reference or a raw
/// pointer) to a byte slice: a byte constant in that position
/// denotes the fat pointer itself, not the pointee data. A bare
/// slice context (a trailing unsized field inside an ADT
/// constant, the `CStr` shape) does not qualify — there the
/// bytes are the field's data.
pub(super) fn ctx_is_u8_slice_ptr(ctx: Option<&mir::Ty>) -> bool {
    match ctx {
        Some(mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _)) => {
            matches!(
                &**inner,
                mir::Ty::Slice(elem)
                    if matches!(&**elem, mir::Ty::Uint(mir::UintTy::U8))
            )
        }
        _ => false,
    }
}

/// The element size in bytes of the pointee of a constant's context
/// type, when it is a scalar-element array/slice behind any number of
/// references. `None` when it cannot be derived.
pub(super) fn pointee_elem_size(ctx: Option<&mir::Ty>) -> Option<u64> {
    match ctx {
        Some(mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _)) => {
            pointee_elem_size(Some(inner))
        }
        Some(mir::Ty::Array(elem, _) | mir::Ty::Slice(elem)) => match &**elem {
            mir::Ty::Bool => Some(1),
            mir::Ty::Char => Some(4),
            mir::Ty::Int(it) => Some(u64::from(it.bits()) / 8),
            mir::Ty::Uint(ut) => Some(u64::from(ut.bits()) / 8),
            _ => None,
        },
        _ => None,
    }
}

/// Sign-extend the low `bits` of `data` into an `i128`.
pub(super) fn sign_extend(data: u128, bits: u32) -> i128 {
    if bits == 0 || bits >= 128 {
        return data as i128;
    }
    let shift = 128 - bits;
    ((data << shift) as i128) >> shift
}

/// Human-readable text for a MIR `Assert` message: the custom
/// string when there is one; for the compiler-generated kinds,
/// the exact message core panics with (these messages are what a
/// `catch_unwind` caller observes as the panic's `&str` payload,
/// so they must match core's text, not a paraphrase). Kinds whose
/// message embeds operand values (bounds checks, misaligned
/// dereferences) keep the stable kind text: the operands are not
/// rendered.
pub(super) fn assert_message_text(msg: &mir::AssertMessage) -> String {
    use mir::AssertMessage as A;
    match msg {
        A::Custom(s) => s.clone(),
        A::Overflow(op, _, _) => match op {
            mir::BinOp::Add | mir::BinOp::AddWithOverflow => {
                "attempt to add with overflow".to_string()
            }
            mir::BinOp::Sub | mir::BinOp::SubWithOverflow => {
                "attempt to subtract with overflow".to_string()
            }
            mir::BinOp::Mul | mir::BinOp::MulWithOverflow => {
                "attempt to multiply with overflow".to_string()
            }
            mir::BinOp::Div => "attempt to divide with overflow".to_string(),
            mir::BinOp::Rem => {
                "attempt to calculate the remainder with overflow".to_string()
            }
            mir::BinOp::Shl | mir::BinOp::ShlUnchecked => {
                "attempt to shift left with overflow".to_string()
            }
            mir::BinOp::Shr | mir::BinOp::ShrUnchecked => {
                "attempt to shift right with overflow".to_string()
            }
            other => format!("assertion failed: {other:?}"),
        },
        A::OverflowNeg(_) => "attempt to negate with overflow".to_string(),
        A::DivisionByZero(_) => "attempt to divide by zero".to_string(),
        A::RemainderByZero(_) => {
            "attempt to calculate the remainder with a divisor of zero".to_string()
        }
        // The coroutine resume guards: rustc lowers these
        // asserts to the `panic_const_*` entry points whose
        // payloads are the exact strings below (the wording a
        // `catch_unwind` observer downcasts and compares), not
        // to a generic assertion failure.
        A::ResumedAfterReturn(kind) => match kind {
            mir::CoroutineKind::Async => {
                "`async fn` resumed after completion".to_string()
            }
            mir::CoroutineKind::Gen => {
                "`gen fn` should just keep returning `None` after completion".to_string()
            }
            mir::CoroutineKind::Coroutine => {
                "coroutine resumed after completion".to_string()
            }
        },
        A::ResumedAfterPanic(kind) => match kind {
            mir::CoroutineKind::Async => {
                "`async fn` resumed after panicking".to_string()
            }
            mir::CoroutineKind::Gen => {
                "`gen fn` should just keep returning `None` after panicking".to_string()
            }
            mir::CoroutineKind::Coroutine => {
                "coroutine resumed after panicking".to_string()
            }
        },
        A::ResumedAfterDrop(kind) => match kind {
            mir::CoroutineKind::Async => {
                "`async fn` resumed after async drop".to_string()
            }
            mir::CoroutineKind::Gen => {
                "`gen fn` resumed after async drop".to_string()
            }
            mir::CoroutineKind::Coroutine => {
                "coroutine resumed after async drop".to_string()
            }
        },
        other => format!("assertion failed: {}", other.kind_name()),
    }
}

/// The IR operator for a `mir` `*WithOverflow` operator. These yield
/// `(value, overflowed)` both as a plain `BinaryOp` and inside
/// `CheckedBinaryOp`.
pub(super) fn overflow_base(op: mir::BinOp) -> Option<ir::BinOp> {
    use mir::BinOp as M;
    Some(match op {
        M::AddWithOverflow => ir::BinOp::Add,
        M::SubWithOverflow => ir::BinOp::Sub,
        M::MulWithOverflow => ir::BinOp::Mul,
        _ => return None,
    })
}

/// Whether a `mir` binary operator is a shift (`Shl` / `Shr` or an
/// unchecked spelling): the one family whose operands do not share
/// a type, converted by `convert_shift_operands` instead of
/// `convert_binop_operands`.
pub(super) fn is_shift(op: mir::BinOp) -> bool {
    use mir::BinOp as M;
    matches!(op, M::Shl | M::Shr | M::ShlUnchecked | M::ShrUnchecked)
}

/// The IR operator for a plain (non-checked) `mir` binary operator.
/// `Offset` (pointer arithmetic) is converted separately into
/// `ir::Rvalue::PtrOffset` (a `CheckedBinaryOp` with `Offset`, the
/// only way this function could still see it, is never emitted), so
/// it yields `None` here.
pub(super) fn plain_binop(op: mir::BinOp) -> Option<ir::BinOp> {
    use mir::BinOp as M;
    Some(match op {
        M::Add | M::AddUnchecked => ir::BinOp::Add,
        M::Sub | M::SubUnchecked => ir::BinOp::Sub,
        M::Mul | M::MulUnchecked => ir::BinOp::Mul,
        M::Div => ir::BinOp::Div,
        M::Rem => ir::BinOp::Rem,
        M::BitXor => ir::BinOp::BitXor,
        M::BitAnd => ir::BinOp::BitAnd,
        M::BitOr => ir::BinOp::BitOr,
        M::Shl | M::ShlUnchecked => ir::BinOp::Shl,
        M::Shr | M::ShrUnchecked => ir::BinOp::Shr,
        M::Eq => ir::BinOp::Eq,
        M::Lt => ir::BinOp::Lt,
        M::Le => ir::BinOp::Le,
        M::Ne => ir::BinOp::Ne,
        M::Ge => ir::BinOp::Ge,
        M::Gt => ir::BinOp::Gt,
        M::Cmp => ir::BinOp::Cmp,
        M::Offset => return None,
        // *WithOverflow in a plain position is handled by `overflow_base`
        // before this function is consulted.
        M::AddWithOverflow | M::SubWithOverflow | M::MulWithOverflow => return None,
    })
}

/// The size in bytes of a `mir` type whose layout is fixed by the
/// type alone: scalars, arrays of such types with a known length,
/// and pointers (fat when the pointee is unsized). Anything whose
/// layout needs the compiler's layout computation — ADTs, tuples
/// with more than one element, slices, `str` — yields `None`.
pub(super) fn mir_ty_size(ty: &mir::Ty) -> Option<u64> {
    use mir::Ty as T;
    Some(match ty {
        T::Bool => 1,
        T::Char => 4,
        T::Int(t) => u64::from(t.bits() / 8),
        T::Uint(t) => u64::from(t.bits() / 8),
        T::Float(t) => u64::from(t.bits() / 8),
        T::Array(elem, mir::ArrayLen::Known(n)) => n.checked_mul(mir_ty_size(elem)?)?,
        T::Array(_, _) => return None,
        T::RawPtr(pointee, _) | T::Ref(_, pointee, _) => {
            if matches!(&**pointee, T::Slice(_) | T::Str | T::Dynamic(..)) {
                16
            } else {
                8
            }
        }
        T::FnPtr(_) => 8,
        T::Tuple(ts) if ts.len() <= 1 => match ts.first() {
            Some(t) => mir_ty_size(t)?,
            None => 0,
        },
        // A multi-field tuple's size is its native layout:
        // fields in declaration order, each aligned to its own
        // alignment, the total rounded up to the widest field
        // alignment — the layout the compiler's `size_of` and
        // the allocation layouts the driver records both use,
        // and the stride pointer arithmetic over a tuple-typed
        // pointee (a hashbrown bucket pointer) is computed in.
        // A field whose layout is not structurally known (an
        // ADT, whose layout lives in the layout table the free
        // function cannot see) leaves the size unknown.
        T::Tuple(_) => {
            fn layout(ty: &mir::Ty) -> Option<(u64, u64)> {
                Some(match ty {
                    T::Bool => (1, 1),
                    T::Char => (4, 4),
                    T::Int(t) => {
                        let b = u64::from(t.bits() / 8);
                        (b, b)
                    }
                    T::Uint(t) => {
                        let b = u64::from(t.bits() / 8);
                        (b, b)
                    }
                    T::Float(t) => {
                        let b = u64::from(t.bits() / 8);
                        (b, b)
                    }
                    T::Array(elem, mir::ArrayLen::Known(n)) => {
                        let (es, ea) = layout(elem)?;
                        (es.checked_mul(*n)?, ea)
                    }
                    T::RawPtr(pointee, _) | T::Ref(_, pointee, _) => {
                        if matches!(
                            &**pointee,
                            T::Slice(_) | T::Str | T::Dynamic(..)
                        ) {
                            (16, 8)
                        } else {
                            (8, 8)
                        }
                    }
                    T::FnPtr(_) => (8, 8),
                    T::Tuple(ts) => {
                        let mut size = 0u64;
                        let mut align = 1u64;
                        for f in ts {
                            let (fs, fa) = layout(f)?;
                            align = align.max(fa);
                            size = size.checked_add(fa - 1)? / fa * fa;
                            size = size.checked_add(fs)?;
                        }
                        (size.checked_add(align - 1)? / align * align, align)
                    }
                    T::Pat(inner, _) => layout(inner)?,
                    T::Never => (0, 1),
                    _ => return None,
                })
            }
            layout(ty)?.0
        }
        T::Pat(inner, _) => mir_ty_size(inner)?,
        T::Never => 0,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Drop glue construction
// ---------------------------------------------------------------------------

/// Parse an impl-instance key `<Self as Trait>::method` (the
/// trait may carry a generic argument list, and `Self` may carry
/// one too) into `(Self section, trait path without arguments,
/// method)`. `None` for any other key shape.
pub(super) fn parse_impl_key(base: &str) -> Option<(String, String, String)> {
    let (head, method) = base.rsplit_once(">::")?;
    // An instance key spells the method's own generic arguments
    // as a trailing `::<...>` segment (`bar::<'_>` for a method
    // with lifetime parameters); the method a call names is the
    // segment-free name.
    let method = method.split("::<").next().unwrap_or(method);
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

/// Parse an impl body key in the impl-block path form the
/// driver records for impls inside a module (foreign crates in
/// particular): `some::module::<impl Trait<..> for Self>::method`.
/// Returns the `Self` section, the trait path without generic
/// arguments, and the method name, as [`parse_impl_key`] does
/// for the `<Self as Trait>::method` form.
pub(super) fn parse_impl_block_key(base: &str) -> Option<(String, String, String)> {
    let (head, method) = base.rsplit_once(">::")?;
    // An instance key spells the method's own generic arguments
    // as a trailing `::<...>` segment (`bar::<'_>` for a method
    // with lifetime parameters); the method a call names is the
    // segment-free name.
    let method = method.split("::<").next().unwrap_or(method);
    if method.is_empty() || method.contains("::") {
        return None;
    }
    // The `>::` separator consumed the impl's closing `>`, so
    // `head` ends with the `Self` section itself.
    let marker = head.rfind("::<impl ")?;
    let inner = &head[marker + "::<impl ".len()..];
    let (trait_part, self_part) = inner.split_once(" for ")?;
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

/// Whether the `Self` section of an impl key denotes the type
/// named `self_name`: either the name itself, or the name with a
/// generic argument list appended (`YCombinator<F, A, R>` for the
/// type `YCombinator` — an impl body recorded under its generic
/// definition name, while a call's `Self` type renders without
/// the definition's parameter list). The crate prefix may be
/// present on either side or on neither.
pub(super) fn impl_self_matches(self_part: &str, self_name: &str) -> bool {
    // The section and the queried name can be the very same
    // rendering (a `Self` that is itself an impl-path type,
    // whose base the `<` split below cannot recover).
    if self_part == self_name {
        return true;
    }
    let self_base = self_part.split('<').next().unwrap_or(self_part);
    // The queried name may itself carry a generic argument list
    // (a fully rendered concrete type); only its base takes part
    // in the comparison, as the key side's list is stripped above.
    let self_name = self_name.split('<').next().unwrap_or(self_name);
    // A generic Self section must be exactly `name<...>` (the
    // argument list runs to the end of the section).
    if self_part != self_base && !self_part.ends_with('>') {
        return false;
    }
    if self_base == self_name {
        return true;
    }
    // When both sides carry a crate path they must agree exactly;
    // the short-name comparison is only for a side that omits it
    // (the generic-definition spelling above).
    if self_base.contains("::") && self_name.contains("::") {
        return false;
    }
    let short = self_name.rsplit("::").next().unwrap_or(self_name);
    let self_short = self_base.rsplit("::").next().unwrap_or(self_base);
    self_short == short
}

/// Whether `name` is the driver name of `adt_name`'s `Drop` impl:
/// `<{adt_name} as std::ops::Drop>::drop`, with any generic
/// argument list after the type name (the impl of a generic type is
/// recorded with its parameter spellings, e.g. `<probe::G<T> as
/// ...>::drop`). The `core::` spelling of the trait path is
/// accepted as well.
pub(super) fn is_dtor_name(name: &str, adt_name: &str) -> bool {
    for trait_path in ["std::ops::Drop", "core::ops::Drop"] {
        // `pin_drop` is the same trait's destructor entry point
        // under pin ergonomics: a type that provides it has no
        // `drop` method, and the compiler's glue for dropping
        // the type is a call to it.
        let suffix = ["drop", "pin_drop"]
            .iter()
            .find_map(|m| name.strip_suffix(&format!(" as {trait_path}>::{m}")));
        let Some(head) = suffix else {
            continue;
        };
        // The head must be exactly the ADT's own rendering:
        // `<{adt_name}` or `<{adt_name}<args>` with the argument
        // group balanced and closing at the head's very end. A
        // type defined inside the ADT's own items renders
        // `<{adt_name}<..>::method::Local<..>`: its head also
        // starts with `<{adt_name}<` and ends with `>`, but the
        // first group closes long before the end — and the
        // nested type's destructor is not the ADT's, so the
        // resemblance must not attribute it to the ADT.
        let Some(rest) = head.strip_prefix(&format!("<{adt_name}")) else {
            continue;
        };
        if rest.is_empty() {
            return true;
        }
        if !rest.starts_with('<') {
            continue;
        }
        let mut depth = 0usize;
        let mut closes_at_end = false;
        for (i, ch) in rest.char_indices() {
            match ch {
                '<' => depth += 1,
                '>' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        closes_at_end = i == rest.len() - 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        if closes_at_end {
            return true;
        }
    }
    false
}
