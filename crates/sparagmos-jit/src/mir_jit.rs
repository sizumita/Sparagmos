//! The MIR front end: compiles `sparagmos-mir` bodies to executable
//! code through the crate's existing backend IR.
//!
//! # Design: lowering, not a second code generator
//!
//! MIR bodies are *lowered* to the backend IR ([`crate::ir`]) and the
//! result is compiled by the existing baseline compiler as one
//! program ([`crate::program`]). The backend was always shaped for
//! this: it is a locals-in-stack-slots machine, and MIR locals are
//! mutable slots, so each MIR local maps onto one IR value slot per
//! scalar leaf (a tuple local occupies consecutive slots, one per
//! leaf). Writing a second, MIR-direct code generator instead would
//! duplicate the emitter, the W^X memory handling and the calling
//! convention for no semantic gain.
//!
//! # Value representation
//!
//! Every scalar leaf is one 64-bit word in *canonical form* for its
//! MIR type: sign-extended for signed integer types, zero-extended
//! (masked to the type's width) for unsigned types, `bool` and
//! `char`. The lowering re-establishes the invariant after every
//! operation whose raw 64-bit result can leave the type's range
//! (plain arithmetic, shifts, casts), so the backend's plain 64-bit
//! operations are exact for every integer type of up to 64 bits, and
//! comparisons only need the operand type's signedness.
//!
//! A float leaf holds its IEEE 754 bit pattern instead: all 64 bits
//! for `f64`, the low 32 bits (upper half zero) for `f32`. The
//! backend computes on the patterns in SSE registers, so float
//! arithmetic, comparisons and casts follow the hardware's IEEE
//! semantics — which are exactly Rust's — with no normalisation
//! step in the lowering.
//!
//! An ADT value is a single leaf holding its discriminant value,
//! taken from the program's discriminant table exactly as the
//! interpreter computes `Discriminant` results. Only the tag is
//! represented; payloads are outside the model, and every construct
//! that would create or touch one fails with an explicit
//! unsupported error at its own lowering.
//!
//! A thin reference or raw pointer is one word: the stack address
//! of the pointee's slot. Taking a reference emits the backend's
//! slot-address instruction, and reading or writing through
//! exactly one dereference loads or stores through that address,
//! so the access lands on the original place — the interpreter's
//! model of a reference, where reads and writes reach the place
//! the reference was taken of. This holds across a call, too: a
//! caller's frame outlives the call, so a callee writing through
//! a pointer argument writes the caller's place. Fat references
//! (to slices, `str` or trait objects), a projection *through* a
//! dereference (a field of a pointed-to value) and a second
//! dereference stay unsupported, as do pointees that are not a
//! single scalar leaf.
//!
//! # Panics
//!
//! A guest panic (failed `Assert`, checked-arithmetic overflow via
//! the `Assert` rustc emits, division by zero) is compiled to the
//! backend's panic instruction: the message is recorded through
//! `sparagmos-runtime`'s panic path and the program's panic flag is
//! set, after which the panic propagates out of the JIT call stack
//! and [`JitProgram::call`] returns it as a [`PanicRecord`]. Division
//! and remainder get explicit divisor guards at lowering time because
//! the hardware instructions trap on a zero divisor (and on
//! `MIN / -1` in the 64-bit signed case). Reaching a MIR `Unreachable`
//! block is a guest panic too, with the message `entered unreachable
//! code` — the interpreter's model — lowered to a jump into a panic
//! block like a failed `Assert`, never to a hardware trap.
//!
//! The interpreter's model of the *plain* (unchecked) operators is
//! that they panic on overflow — `arithmetic overflow in Add`, the
//! shift-amount message, and so on — so the lowering gives the plain
//! operators exactly those panics: plain `Add` / `Sub` / `Mul` are
//! compiled through the checked backend operation plus a branch on
//! its flag, plain division checks the `MIN / -1` case explicitly,
//! plain shifts check the amount only (bits shifted out are
//! lost, as natively), and plain `Neg` checks for the type's
//! minimum. The checked
//! forms (`CheckedBinaryOp`, `*WithOverflow`) yield the flag as a
//! value instead, as in the interpreter.
//!
//! # Scope
//!
//! Supported: integer / `bool` / `char` / `f32` / `f64` /
//! tuple-of-supported and array-of-supported locals (arrays are
//! their elements' leaves in order, read and written through
//! constant index projections), ADT locals holding only a
//! discriminant tag, thin references and raw pointers to scalar
//! places; `Use`, plain and checked `BinaryOp`, `UnaryOp`,
//! `IntToInt` / `FloatToInt` / `IntToFloat` / `FloatToFloat`
//! casts and the address-preserving pointer casts, tuple
//! aggregates, fieldless ADT aggregates, `Repeat`,
//! `Discriminant`, `SetDiscriminant`; `Goto`, `SwitchInt`, `Call`
//! (within the program, up to six arguments), `Assert`, `Drop` of
//! scalar values, `Return`, `Unreachable` (as a guest panic).
//! Anything else — `f16` / `f128`, 128-bit integers, fat
//! references, projections through a dereference, ADT payloads,
//! slices, aggregates other than tuples and fieldless ADTs,
//! dynamic index projections,
//! calls to functions outside the program, and so on — fails the
//! whole function's compilation with [`JitError::Unsupported`]
//! naming the construct. There is deliberately no partial or
//! fallback execution: a program either compiles completely on the
//! JIT path or not at all.

use std::collections::HashMap;
use std::fmt;

use sparagmos_mir as mir;
use sparagmos_runtime::error::PanicRecord;

use crate::compiler::CompileError;
use crate::ir::{
    BasicBlock as IrBlock, BinOp as IrBinOp, CheckedOp, FloatBinOp as IrFloatBinOp,
    FloatKind as IrFloatKind, Function as IrFunction, Instr, IntKind as IrIntKind,
    Operand as IrOperand, Terminator as IrTerminator, UnOp as IrUnOp,
};
use crate::program::{compile_program, CompiledProgram};

// ---------------------------------------------------------------------------
// Public input / output types
// ---------------------------------------------------------------------------

/// One function of a program to compile: a name (used for lookup and
/// diagnostics), the `DefId` other functions call it by (if any), and
/// its MIR body. This mirrors the shape the driver's conversion
/// produces (`MirProgram` in `sparagmos-interp`), kept local to this
/// crate so the JIT does not depend on the interpreter.
#[derive(Debug, Clone)]
pub struct MirFunctionInput {
    pub name: String,
    pub def_id: Option<mir::DefId>,
    pub body: mir::Body,
}

/// A whole program: the functions, the driver's definition-name
/// table (`DefId` -> path), used to name callees in diagnostics, and
/// the ADT discriminant table, which gives `Discriminant` reads and
/// `SetDiscriminant` writes the compiler's discriminant values.
#[derive(Debug, Clone, Default)]
pub struct MirProgramInput {
    pub functions: Vec<MirFunctionInput>,
    pub names: HashMap<mir::DefId, String>,
    pub adt_discriminants: mir::AdtDiscriminants,
}

/// Why a program did not compile on the JIT path.
#[derive(Debug)]
pub enum JitError {
    /// A construct outside the JIT's supported range. The string
    /// names the function and the construct.
    Unsupported(String),
    /// The backend rejected the lowered IR.
    Backend(CompileError),
}

impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JitError::Unsupported(what) => write!(f, "unsupported: {what}"),
            JitError::Backend(e) => write!(f, "backend error: {e}"),
        }
    }
}
impl std::error::Error for JitError {}
impl From<CompileError> for JitError {
    fn from(e: CompileError) -> Self {
        JitError::Backend(e)
    }
}

/// Why a call into a compiled program did not produce a value.
#[derive(Debug)]
pub enum JitCallError {
    /// The guest panicked; the record carries the panic message.
    Panic(PanicRecord),
    /// No function with this name is part of the program.
    UnknownFunction(String),
    /// The argument count differs from the function's parameter count.
    WrongArgCount {
        name: String,
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for JitCallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JitCallError::Panic(p) => write!(f, "guest panic: {}", p.message),
            JitCallError::UnknownFunction(n) => write!(f, "unknown function `{n}`"),
            JitCallError::WrongArgCount { name, expected, got } => {
                write!(f, "function `{name}` takes {expected} arguments, got {got}")
            }
        }
    }
}
impl std::error::Error for JitCallError {}

/// A MIR program compiled to executable code.
pub struct JitProgram {
    inner: CompiledProgram,
    index_by_name: HashMap<String, usize>,
    /// Parameter count (MIR argument locals) per function.
    param_counts: Vec<usize>,
    names: Vec<String>,
}

impl fmt::Debug for JitProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JitProgram")
            .field("functions", &self.names)
            .finish_non_exhaustive()
    }
}

impl JitProgram {
    /// Lower and compile the whole program. Fails with
    /// [`JitError::Unsupported`] if any function uses a construct
    /// outside the supported range — nothing is executed partially.
    pub fn compile(input: &MirProgramInput) -> Result<JitProgram, JitError> {
        let mut def_index: HashMap<mir::DefId, usize> = HashMap::new();
        for (i, f) in input.functions.iter().enumerate() {
            if let Some(def) = f.def_id {
                def_index.insert(def, i);
            }
        }
        // Parameter types of every function, for typing call arguments.
        let param_tys: Vec<Vec<mir::Ty>> = input
            .functions
            .iter()
            .map(|f| {
                f.body.local_decls[1..=f.body.arg_count]
                    .iter()
                    .map(|d| d.ty.clone())
                    .collect()
            })
            .collect();
        let mut messages = MsgTable::default();
        let mut ir_funcs: Vec<IrFunction> = Vec::with_capacity(input.functions.len());
        for f in &input.functions {
            let lower = FnLower::new(
                f,
                &def_index,
                &input.names,
                &param_tys,
                &input.adt_discriminants,
                &mut messages,
            );
            ir_funcs.push(lower.run()?);
        }
        let inner = compile_program(&ir_funcs, messages.into_vec())?;
        Ok(JitProgram {
            inner,
            index_by_name: input
                .functions
                .iter()
                .enumerate()
                .map(|(i, f)| (f.name.clone(), i))
                .collect(),
            param_counts: input.functions.iter().map(|f| f.body.arg_count).collect(),
            names: input.functions.iter().map(|f| f.name.clone()).collect(),
        })
    }

    /// Names of the program's functions, in input order.
    pub fn function_names(&self) -> Vec<&str> {
        self.names.iter().map(|s| s.as_str()).collect()
    }

    /// Call the function called `name` with one argument per MIR
    /// parameter (unit parameters are passed as 0 and ignored).
    /// Arguments are the parameters' canonical 64-bit representations
    /// (sign-extended for signed types). The result is the return
    /// value's canonical representation (0 for a unit return).
    pub fn call(&self, name: &str, args: &[i64]) -> Result<i64, JitCallError> {
        let idx = self
            .index_by_name
            .get(name)
            .copied()
            .ok_or_else(|| JitCallError::UnknownFunction(name.to_string()))?;
        self.call_index(idx, args)
    }

    /// Call the function at index `idx` (input order); see
    /// [`JitProgram::call`] for the argument contract.
    pub fn call_index(&self, idx: usize, args: &[i64]) -> Result<i64, JitCallError> {
        let expected = self.param_counts[idx];
        if args.len() != expected {
            return Err(JitCallError::WrongArgCount {
                name: self.names[idx].clone(),
                expected,
                got: args.len(),
            });
        }
        self.inner.call(idx, args).map_err(JitCallError::Panic)
    }
}

// ---------------------------------------------------------------------------
// Message table
// ---------------------------------------------------------------------------

#[derive(Default)]
struct MsgTable {
    msgs: Vec<String>,
    map: HashMap<String, u32>,
}

impl MsgTable {
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(i) = self.map.get(s) {
            return *i;
        }
        let i = self.msgs.len() as u32;
        self.msgs.push(s.to_string());
        self.map.insert(s.to_string(), i);
        i
    }
    fn into_vec(self) -> Vec<String> {
        self.msgs
    }
}

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

/// Bit width of an integer type on this 64-bit target.
fn int_bits(it: mir::IntTy) -> u32 {
    use mir::IntTy::*;
    match it {
        I8 => 8,
        I16 => 16,
        I32 => 32,
        I64 => 64,
        I128 => 128,
        Isize => 64,
    }
}

fn uint_bits(ut: mir::UintTy) -> u32 {
    use mir::UintTy::*;
    match ut {
        U8 => 8,
        U16 => 16,
        U32 => 32,
        U64 => 64,
        U128 => 128,
        Usize => 64,
    }
}

/// The integer shape of a scalar type, if it has one the JIT supports.
fn int_kind(ty: &mir::Ty) -> Option<IrIntKind> {
    match ty {
        mir::Ty::Int(it) => {
            let b = int_bits(*it);
            (b <= 64).then(|| IrIntKind::new(b as u8, true))
        }
        mir::Ty::Uint(ut) => {
            let b = uint_bits(*ut);
            (b <= 64).then(|| IrIntKind::new(b as u8, false))
        }
        mir::Ty::Bool => Some(IrIntKind::new(1, false)),
        mir::Ty::Char => Some(IrIntKind::new(32, false)),
        _ => None,
    }
}

/// Leaf bound for one array type. Every leaf is an 8-byte stack
/// slot in the compiled frame, so an unbounded array type would let
/// a guest program demand an unbounded frame; past this many scalar
/// elements the type is outside the JIT's range. (The interpreter
/// bounds `Repeat` element counts the same way, at 1,000,000; the
/// JIT's bound is in leaves because its frames are physical.)
const MAX_ARRAY_LEAVES: usize = 4096;

/// Number of scalar leaves a value of `ty` occupies, or the reason
/// the type is outside the JIT's range.
fn flatten(ty: &mir::Ty, fname: &str) -> Result<usize, JitError> {
    match ty {
        mir::Ty::Bool | mir::Ty::Char => Ok(1),
        mir::Ty::Int(it) => {
            if int_bits(*it) <= 64 {
                Ok(1)
            } else {
                Err(unsup(fname, "128-bit integer type"))
            }
        }
        mir::Ty::Uint(ut) => {
            if uint_bits(*ut) <= 64 {
                Ok(1)
            } else {
                Err(unsup(fname, "128-bit integer type"))
            }
        }
        mir::Ty::Tuple(ts) => {
            let mut n = 0;
            for t in ts {
                n += flatten(t, fname)?;
            }
            Ok(n)
        }
        // A never-typed value is never produced; it occupies no leaf.
        mir::Ty::Never => Ok(0),
        mir::Ty::Ref(_, pointee, _) | mir::Ty::RawPtr(pointee, _) => {
            // A thin reference / raw pointer is one word: the
            // pointee slot's address. A fat pointee (a slice, `str`
            // or a trait object) would need a second word for its
            // metadata and stays unsupported.
            match &**pointee {
                mir::Ty::Str | mir::Ty::Slice(_) | mir::Ty::Dynamic(..) => {
                    Err(unsup(fname, "fat reference / pointer type"))
                }
                _ => Ok(1),
            }
        }
        mir::Ty::Float(mir::FloatTy::F32) | mir::Ty::Float(mir::FloatTy::F64) => Ok(1),
        mir::Ty::Float(mir::FloatTy::F16) => {
            Err(unsup(fname, "16-bit floating-point type"))
        }
        mir::Ty::Float(mir::FloatTy::F128) => {
            Err(unsup(fname, "128-bit floating-point type"))
        }
        mir::Ty::Array(elem, len) => {
            // An array value is its elements' leaves in order, the
            // same shape the interpreter gives it (arrays ride as
            // tuples of their elements there).
            let mir::ArrayLen::Known(n) = len else {
                return Err(unsup(fname, "array type with unknown length"));
            };
            let per = flatten(elem, fname)?;
            let total = (*n as usize).checked_mul(per).unwrap_or(usize::MAX);
            if total > MAX_ARRAY_LEAVES {
                return Err(unsup(
                    fname,
                    "array type with more scalar elements than the JIT frame holds",
                ));
            }
            Ok(total)
        }
        mir::Ty::Adt(..) => {
            // An ADT value is a single leaf holding its discriminant
            // value. Payloads have no representation: every construct
            // that would create or touch one (an aggregate or a
            // constant with fields, a field or downcast projection)
            // fails with an explicit unsupported error at its own
            // lowering, so a program that only ever moves the tag —
            // `Discriminant` reads, `SetDiscriminant`, `SwitchInt`
            // on the discriminant — is exact.
            Ok(1)
        }
        other => Err(unsup(fname, &format!("type {}", other.kind_name()))),
    }
}

/// The backend float format of a scalar type, if it is one the JIT
/// supports.
fn float_kind(ty: &mir::Ty) -> Option<IrFloatKind> {
    match ty {
        mir::Ty::Float(mir::FloatTy::F32) => Some(IrFloatKind::F32),
        mir::Ty::Float(mir::FloatTy::F64) => Some(IrFloatKind::F64),
        _ => None,
    }
}

fn unsup(fname: &str, what: &str) -> JitError {
    JitError::Unsupported(format!("function `{fname}`: {what}"))
}

/// Reduce a raw `u128` target value of a `SwitchInt` to the canonical
/// `i64` for the discriminant type (target values are stored unsigned
/// in MIR; signed types sign-extend from their width).
fn normalize_u128(v: u128, ty: &mir::Ty) -> i64 {
    match int_kind(ty) {
        Some(k) => {
            let bits = k.bits as u32;
            if bits >= 64 {
                v as i64
            } else if k.signed {
                let shift = 64 - bits;
                (((v as u64) << shift) as i64) >> shift
            } else {
                (v as u64 & ((1u64 << bits) - 1)) as i64
            }
        }
        None => v as i64,
    }
}

/// The shared type of a pair of integer literals, for a comparison
/// neither of whose operands has a place type. Mirrors the
/// interpreter's rule: both literals must have the same byte size;
/// if either has the width's sign bit set the pair is read as the
/// signed integer of that width, otherwise as the unsigned one.
fn literal_pair_ty(lhs: &mir::Operand, rhs: &mir::Operand) -> Option<mir::Ty> {
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
    use mir::IntTy::*;
    Some(match bits {
        8 => I8,
        16 => I16,
        32 => I32,
        64 => I64,
        128 => I128,
        _ => return None,
    })
}

fn uint_ty_of_bits(bits: u32) -> Option<mir::UintTy> {
    use mir::UintTy::*;
    Some(match bits {
        8 => U8,
        16 => U16,
        32 => U32,
        64 => U64,
        128 => U128,
        _ => return None,
    })
}

/// The shared type of a pair of float literals, for a comparison
/// neither of whose operands has a place type (the float form of
/// [`literal_pair_ty`]).
fn literal_float_pair_ty(lhs: &mir::Operand, rhs: &mir::Operand) -> Option<mir::Ty> {
    fn scalar_float(op: &mir::Operand) -> Option<mir::FloatTy> {
        let mir::Operand::Constant(c) = op else {
            return None;
        };
        let mir::ConstKind::Value(mir::ConstValue::Scalar(mir::Scalar::Float { ty, .. })) =
            &c.literal
        else {
            return None;
        };
        Some(*ty)
    }
    let l = scalar_float(lhs)?;
    let r = scalar_float(rhs)?;
    if l == r && matches!(l, mir::FloatTy::F32 | mir::FloatTy::F64) {
        Some(mir::Ty::Float(l))
    } else {
        None
    }
}

/// The integer type of a lone integer literal, typed by the same
/// heuristic as [`literal_pair_ty`]: a set sign bit at the literal's
/// own width reads as the signed type of that width, otherwise the
/// unsigned one. Used for the source of an `IntToFloat` cast whose
/// operand is a constant.
fn literal_int_ty(op: &mir::Operand) -> Option<mir::Ty> {
    let mir::Operand::Constant(c) = op else {
        return None;
    };
    let mir::ConstKind::Value(mir::ConstValue::Scalar(mir::Scalar::Int { data, size })) =
        &c.literal
    else {
        return None;
    };
    let bits = u32::from(*size) * 8;
    if bits > 0 && bits < 128 && (data >> (bits - 1)) & 1 == 1 {
        int_ty_of_bits(bits).map(mir::Ty::Int)
    } else {
        uint_ty_of_bits(bits).map(mir::Ty::Uint)
    }
}

/// The panic message text for an `Assert`, mirroring the
/// interpreter's rule: a custom message verbatim, anything else as
/// `assertion failed: <kind>`.
fn assert_message_text(msg: &mir::AssertMessage) -> String {
    match msg {
        mir::AssertMessage::Custom(s) => s.clone(),
        other => format!("assertion failed: {}", other.kind_name()),
    }
}

// ---------------------------------------------------------------------------
// Per-function lowering
// ---------------------------------------------------------------------------

struct FnLower<'a> {
    func: &'a MirFunctionInput,
    body: &'a mir::Body,
    def_index: &'a HashMap<mir::DefId, usize>,
    def_names: &'a HashMap<mir::DefId, String>,
    param_tys: &'a [Vec<mir::Ty>],
    discriminants: &'a mir::AdtDiscriminants,
    messages: &'a mut MsgTable,
    /// Per local: (first leaf slot, leaf count).
    local_slots: Vec<(u32, usize)>,
    next_value: u32,
    /// IR blocks by id; MIR block `i` is IR block `i`, synthetic
    /// blocks are appended after them.
    blocks: Vec<Option<IrBlock>>,
    /// The segment currently being filled and its block id.
    cur_id: u32,
    cur: Vec<Instr>,
}

impl<'a> FnLower<'a> {
    fn new(
        func: &'a MirFunctionInput,
        def_index: &'a HashMap<mir::DefId, usize>,
        def_names: &'a HashMap<mir::DefId, String>,
        param_tys: &'a [Vec<mir::Ty>],
        discriminants: &'a mir::AdtDiscriminants,
        messages: &'a mut MsgTable,
    ) -> Self {
        Self {
            func,
            body: &func.body,
            def_index,
            def_names,
            param_tys,
            discriminants,
            messages,
            local_slots: Vec::new(),
            next_value: 0,
            blocks: Vec::new(),
            cur_id: 0,
            cur: Vec::new(),
        }
    }

    fn err(&self, what: &str) -> JitError {
        unsup(&self.func.name, what)
    }

    /// The discriminant value of `variant_index` of the ADT `def`,
    /// from the program's discriminant table. An ADT with no table
    /// entry has the sequential values, so the index itself is the
    /// value — the rule `sparagmos-mir` documents for the table's
    /// consumers and the one the interpreter applies. An entry that
    /// does not cover the variant is an error, as it is in the
    /// interpreter.
    fn discriminant_value(&self, def: mir::DefId, variant_index: u32) -> Result<i64, JitError> {
        match self.discriminants.get(&def) {
            None => Ok(variant_index as i64),
            Some(values) => values
                .get(variant_index as usize)
                .map(|v| *v as i64)
                .ok_or_else(|| {
                    self.err(&format!(
                        "discriminant table does not cover variant {variant_index}"
                    ))
                }),
        }
    }

    fn fresh(&mut self) -> u32 {
        let v = self.next_value;
        self.next_value += 1;
        v
    }

    fn emit(&mut self, i: Instr) {
        self.cur.push(i);
    }

    fn new_block(&mut self) -> u32 {
        let id = self.blocks.len() as u32;
        self.blocks.push(None);
        id
    }

    /// Seal the current segment with `term` and continue filling
    /// block `next` (which must not have been sealed yet).
    fn seal_and_continue(&mut self, term: IrTerminator, next: u32) {
        let instrs = std::mem::take(&mut self.cur);
        let id = self.cur_id as usize;
        debug_assert!(self.blocks[id].is_none());
        self.blocks[id] = Some(IrBlock::new(instrs, term));
        self.cur_id = next;
    }

    /// Seal a standalone block (used for synthetic panic blocks).
    fn seal_block(&mut self, id: u32, instrs: Vec<Instr>, term: IrTerminator) {
        debug_assert!(self.blocks[id as usize].is_none());
        self.blocks[id as usize] = Some(IrBlock::new(instrs, term));
    }

    /// A fresh block that panics with the given message and never
    /// returns.
    fn panic_block(&mut self, msg: u32) -> u32 {
        let id = self.new_block();
        self.seal_block(id, vec![Instr::Panic { msg }], IrTerminator::Unreachable);
        id
    }

    fn run(mut self) -> Result<IrFunction, JitError> {
        let body = self.body;
        // Lay out the locals. Argument locals are laid out first:
        // each one is one IR parameter, and IR parameters are by
        // definition the values 0..param_count (a unit argument gets
        // a parameter slot whose value is never read; a multi-leaf
        // argument is outside the baseline's register-argument
        // model). The return place and the remaining locals follow.
        let arg_count = body.arg_count;
        self.local_slots = vec![(0, 0); body.local_decls.len()];
        for idx in 1..=arg_count {
            let decl = &body.local_decls[idx];
            let count = flatten(&decl.ty, &self.func.name)?;
            if count > 1 {
                return Err(self.err("multi-word function argument"));
            }
            let slot = self.fresh();
            debug_assert_eq!(slot as usize, idx - 1);
            self.local_slots[idx] = (slot, count);
        }
        for idx in std::iter::once(0).chain(arg_count + 1..body.local_decls.len()) {
            let decl = &body.local_decls[idx];
            let count = flatten(&decl.ty, &self.func.name)?;
            let first = self.next_value;
            for _ in 0..count.max(1) {
                self.fresh();
            }
            self.local_slots[idx] = (first, count);
        }
        // Reserve IR block ids for the MIR blocks.
        for _ in 0..body.basic_blocks.len() {
            self.new_block();
        }
        for (i, bb) in body.basic_blocks.iter().enumerate() {
            self.cur_id = i as u32;
            self.cur = Vec::new();
            for stmt in &bb.statements {
                self.lower_statement(&stmt.kind)?;
            }
            match &bb.terminator {
                Some(term) => self.lower_terminator(&term.kind)?,
                None => {
                    let id = self.cur_id as usize;
                    let instrs = std::mem::take(&mut self.cur);
                    self.blocks[id] = Some(IrBlock::new(instrs, IrTerminator::Unreachable));
                }
            }
        }
        let blocks: Vec<IrBlock> = self
            .blocks
            .into_iter()
            .enumerate()
            .map(|(i, b)| b.unwrap_or_else(|| panic!("IR block {i} was never sealed")))
            .collect();
        Ok(IrFunction {
            name: self.func.name.clone(),
            param_count: arg_count,
            value_count: self.next_value as usize,
            blocks,
        })
    }

    // ----- places, operands, constants -------------------------------------

    /// Resolve a place to (first leaf slot, leaf count, type).
    fn place_leaves(&self, place: &mir::Place) -> Result<(u32, usize, mir::Ty), JitError> {
        let (first, _count) = self.local_slots[place.local.index()];
        let mut ty = self.body.local_decls[place.local.index()].ty.clone();
        let mut offset = 0usize;
        for elem in &place.projection {
            match elem {
                mir::PlaceElem::Field(idx, field_ty) => {
                    let mir::Ty::Tuple(fields) = &ty else {
                        return Err(self.err(&format!(
                            "Field projection on non-tuple type {}",
                            ty.kind_name()
                        )));
                    };
                    let idx = *idx as usize;
                    if idx >= fields.len() {
                        return Err(self.err("Field projection index out of range"));
                    }
                    for f in &fields[..idx] {
                        offset += flatten(f, &self.func.name)?;
                    }
                    ty = field_ty.clone();
                }
                mir::PlaceElem::ConstantIndex {
                    offset: idx,
                    min_length: _,
                    from_end,
                } => {
                    let mir::Ty::Array(elem, len) = &ty else {
                        return Err(self.err(&format!(
                            "ConstantIndex projection on non-array type {}",
                            ty.kind_name()
                        )));
                    };
                    let mir::ArrayLen::Known(len) = len else {
                        return Err(self.err(
                            "ConstantIndex projection on an array of unknown length",
                        ));
                    };
                    // `from_end` offsets count back from one past
                    // the last element (offset 1 is the last
                    // element), as in rustc and the interpreter.
                    let idx = if *from_end {
                        match len.checked_sub(*idx).filter(|i| *i < *len) {
                            Some(i) => i,
                            None => {
                                return Err(
                                    self.err("ConstantIndex projection out of range")
                                );
                            }
                        }
                    } else if idx < len {
                        *idx
                    } else {
                        return Err(self.err("ConstantIndex projection out of range"));
                    };
                    offset += idx as usize * flatten(elem, &self.func.name)?;
                    ty = (**elem).clone();
                }
                mir::PlaceElem::OpaqueCast(t) | mir::PlaceElem::Subtype(t) => {
                    ty = t.clone();
                }
                other => {
                    return Err(self.err(&format!("{} projection", other.kind_name())));
                }
            }
        }
        let count = flatten(&ty, &self.func.name)?;
        Ok((first + offset as u32, count, ty))
    }

    /// Resolve a place that goes through a dereference: local
    /// projections up to a single `Deref`, and nothing after it.
    /// Returns the slot holding the pointer and the pointee type,
    /// or `None` when the place has no `Deref` at all. Any other
    /// shape involving a `Deref` — a projection after it, a second
    /// `Deref` — is an explicit error: those need address
    /// arithmetic on the pointee's layout, which the slot model
    /// does not have.
    fn deref_place(&self, place: &mir::Place) -> Result<Option<(u32, mir::Ty)>, JitError> {
        let Some(pos) = place
            .projection
            .iter()
            .position(|e| matches!(e, mir::PlaceElem::Deref))
        else {
            return Ok(None);
        };
        if pos + 1 != place.projection.len() {
            return Err(self.err("projection through a dereference"));
        }
        let base = mir::Place {
            local: place.local,
            projection: place.projection[..pos].to_vec(),
        };
        let (slot, count, ty) = self.place_leaves(&base)?;
        if count != 1 {
            return Err(self.err("dereference of a multi-word pointer value"));
        }
        let pointee = match &ty {
            mir::Ty::Ref(_, pointee, _) | mir::Ty::RawPtr(pointee, _) => (**pointee).clone(),
            _ => {
                return Err(self.err(&format!(
                    "dereference of a non-pointer type {}",
                    ty.kind_name()
                )))
            }
        };
        Ok(Some((slot, pointee)))
    }

    fn place_ty(&self, place: &mir::Place) -> Result<mir::Ty, JitError> {
        if let Some((_, ty)) = self.deref_place(place)? {
            return Ok(ty);
        }
        Ok(self.place_leaves(place)?.2)
    }

    /// Best-effort type of an operand: from its place, when it has one.
    fn operand_place_ty(&self, op: &mir::Operand) -> Option<mir::Ty> {
        op.place().and_then(|p| self.place_ty(p).ok())
    }

    /// The float type of an operand, from its place or — for a float
    /// constant — from the constant itself. Casts need the source
    /// type even when the operand has no place.
    fn operand_float_ty(&self, op: &mir::Operand) -> Option<mir::Ty> {
        if let Some(t) = self.operand_place_ty(op) {
            if float_kind(&t).is_some() {
                return Some(t);
            }
        }
        if let mir::Operand::Constant(c) = op {
            if let mir::ConstKind::Value(mir::ConstValue::Scalar(mir::Scalar::Float {
                ty,
                ..
            })) = &c.literal
            {
                if matches!(ty, mir::FloatTy::F32 | mir::FloatTy::F64) {
                    return Some(mir::Ty::Float(*ty));
                }
            }
        }
        None
    }

    /// Materialise an operand as a single IR value in canonical form
    /// for `ctx`. Multi-leaf operands are rejected here; whole-value
    /// copies are handled by the `Use` path.
    fn operand_value(&mut self, op: &mir::Operand, ctx: &mir::Ty) -> Result<u32, JitError> {
        match op {
            mir::Operand::Copy(p) | mir::Operand::Move(p) => {
                if let Some((ptr_slot, pointee_ty)) = self.deref_place(p)? {
                    if flatten(&pointee_ty, &self.func.name)? != 1 {
                        return Err(
                            self.err("operand of aggregate type in scalar position")
                        );
                    }
                    // The word at the pointer is the pointee's
                    // canonical form, exactly as if it had been
                    // read from its own slot.
                    let dst = self.fresh();
                    self.emit(Instr::LoadPtr {
                        dst,
                        ptr: IrOperand::Value(ptr_slot),
                    });
                    return Ok(dst);
                }
                let (first, count, _) = self.place_leaves(p)?;
                if count != 1 {
                    return Err(self.err("operand of aggregate type in scalar position"));
                }
                Ok(first)
            }
            mir::Operand::Constant(c) => {
                let v = self.const_i64(c, ctx)?;
                let dst = self.fresh();
                self.emit(Instr::Const { dst, imm: v });
                Ok(dst)
            }
        }
    }

    /// Interpret a constant under the type context it is used in.
    fn const_i64(&self, c: &mir::Constant, ctx: &mir::Ty) -> Result<i64, JitError> {
        match &c.literal {
            mir::ConstKind::Value(v) => self.const_value_i64(v, ctx),
            mir::ConstKind::Ty(t) => Err(self.err(&format!(
                "type constant {} outside call position",
                t.kind_name()
            ))),
            mir::ConstKind::Unevaluated { def, .. } => {
                Err(self.err(&format!("unevaluated constant {def}")))
            }
        }
    }

    fn const_value_i64(&self, v: &mir::ConstValue, ctx: &mir::Ty) -> Result<i64, JitError> {
        Ok(match v {
            mir::ConstValue::Bool(b) => *b as i64,
            mir::ConstValue::Char(ch) => *ch as i64,
            mir::ConstValue::ZeroSized => 0,
            mir::ConstValue::Scalar(mir::Scalar::Int { data, size }) => {
                // `data` holds the value in `size` bytes; normalise it
                // to the context type's canonical form.
                match int_kind(ctx) {
                    Some(k) => {
                        let bits = k.bits as u32;
                        let masked = if bits >= 128 {
                            *data
                        } else {
                            data & ((1u128 << bits) - 1)
                        };
                        if k.signed && bits < 128 {
                            let shift = 128 - bits;
                            ((masked << shift) as i128 >> shift) as i64
                        } else {
                            masked as i64
                        }
                    }
                    None => {
                        // No integer context: fall back to the
                        // scalar's own size, read as unsigned.
                        let bits = (*size as u32) * 8;
                        if bits >= 128 {
                            *data as i64
                        } else {
                            (data & ((1u128 << bits) - 1)) as i64
                        }
                    }
                }
            }
            mir::ConstValue::Scalar(mir::Scalar::Float { bits, ty }) => {
                // A float value's slot word is its bit pattern: the
                // full 64 bits for f64, the low 32 (zero-extended)
                // for f32.
                match ty {
                    mir::FloatTy::F64 => *bits as u64 as i64,
                    mir::FloatTy::F32 => *bits as u32 as i64,
                    other => {
                        return Err(self.err(&format!("{} constant", other.name())));
                    }
                }
            }
            mir::ConstValue::Scalar(mir::Scalar::Ptr { .. }) => {
                return Err(self.err("pointer constant"));
            }
            mir::ConstValue::FnDef(_) => {
                return Err(self.err("function pointer constant"));
            }
            mir::ConstValue::Closure(_) => {
                return Err(self.err("closure constant"));
            }
            mir::ConstValue::Str(_) => return Err(self.err("string constant")),
            mir::ConstValue::Bytes(_) => return Err(self.err("byte-string constant")),
            mir::ConstValue::Slice { .. } => return Err(self.err("slice constant")),
            mir::ConstValue::ByRef { .. } => {
                return Err(self.err("by-reference constant"));
            }
            mir::ConstValue::Tuple(_) => {
                return Err(self.err("tuple constant"));
            }
            mir::ConstValue::Adt { variant, fields } => {
                if !fields.is_empty() {
                    return Err(self.err("ADT constant with fields"));
                }
                let mir::Ty::Adt(def, _) = ctx else {
                    return Err(self.err("ADT constant without an ADT type context"));
                };
                // A fieldless variant constant is its tag alone.
                self.discriminant_value(*def, *variant)?
            }
            mir::ConstValue::Ref(_) => {
                return Err(self.err("reference constant"));
            }
            mir::ConstValue::SliceVal(_) => {
                return Err(self.err("slice-value constant"));
            }
        })
    }

    /// Emit the normalisation that re-establishes canonical form for
    /// `ty` on the value in slot `slot` (in place).
    fn normalize(&mut self, slot: u32, ty: &mir::Ty) {
        let Some(k) = int_kind(ty) else { return };
        let bits = k.bits as u32;
        if bits >= 64 {
            return;
        }
        if k.signed {
            let shift = 64 - bits;
            self.emit(Instr::BinOp {
                dst: slot,
                op: IrBinOp::Shl,
                lhs: IrOperand::Value(slot),
                rhs: IrOperand::Const(shift as i64),
            });
            self.emit(Instr::BinOp {
                dst: slot,
                op: IrBinOp::Sar,
                lhs: IrOperand::Value(slot),
                rhs: IrOperand::Const(shift as i64),
            });
        } else {
            let mask = ((1u64 << bits) - 1) as i64;
            self.emit(Instr::BinOp {
                dst: slot,
                op: IrBinOp::And,
                lhs: IrOperand::Value(slot),
                rhs: IrOperand::Const(mask),
            });
        }
    }

    // ----- statements --------------------------------------------------------

    fn lower_statement(&mut self, stmt: &mir::StatementKind) -> Result<(), JitError> {
        use mir::StatementKind as S;
        match stmt {
            S::Assign(pair) => {
                let (place, rvalue) = &**pair;
                self.lower_assign(place, rvalue)
            }
            // Statements with no runtime effect on values.
            S::StorageLive(_)
            | S::StorageDead(_)
            | S::Nop
            | S::FakeRead(_)
            | S::PlaceMention(_)
            | S::Retag(..)
            | S::AscribeUserType(..)
            | S::Coverage
            | S::ConstEvalCounter => Ok(()),
            S::SetDiscriminant {
                place,
                variant_index,
            } => {
                let (dst, count, ty) = self.place_leaves(place)?;
                let mir::Ty::Adt(def, _) = &ty else {
                    return Err(self.err("SetDiscriminant on a non-ADT place"));
                };
                if count != 1 {
                    return Err(self.err("SetDiscriminant on an ADT with a payload shape"));
                }
                // The place's single leaf is its tag: write the
                // variant's discriminant value from the table.
                let tag = self.discriminant_value(*def, *variant_index)?;
                self.emit(Instr::Const { dst, imm: tag });
                Ok(())
            }
            S::Intrinsic(_) => Err(self.err("Intrinsic statement")),
        }
    }

    /// The value of a fresh reference to `target`: the stack
    /// address of the target place's first leaf slot, or the
    /// pointer's own value when the target is itself a dereference
    /// (`&*p` is `p`).
    fn ref_value(&mut self, target: &mir::Place) -> Result<u32, JitError> {
        if let Some((ptr_slot, _)) = self.deref_place(target)? {
            return Ok(ptr_slot);
        }
        let (first, _, _) = self.place_leaves(target)?;
        let dst = self.fresh();
        self.emit(Instr::SlotAddr { dst, slot: first });
        Ok(dst)
    }

    /// `place = &target` / `place = &raw target`: the reference
    /// value is the target's slot address (see [`Self::ref_value`]).
    fn lower_ref_creation(
        &mut self,
        place: &mir::Place,
        target: &mir::Place,
    ) -> Result<(), JitError> {
        let (dst, count, _) = self.place_leaves(place)?;
        if count != 1 {
            return Err(self.err("reference value does not fit in one slot"));
        }
        let v = self.ref_value(target)?;
        self.emit(Instr::Copy {
            dst,
            src: IrOperand::Value(v),
        });
        Ok(())
    }

    /// Lower `*ptr = rvalue`, where `ptr_slot` holds the pointer
    /// and the pointee is a single scalar leaf: compute the value
    /// into a scratch slot with the ordinary rvalue lowerings, then
    /// store it through the pointer so the write lands on the
    /// original place.
    fn lower_assign_through_ptr(
        &mut self,
        ptr_slot: u32,
        pointee_ty: &mir::Ty,
        rvalue: &mir::Rvalue,
    ) -> Result<(), JitError> {
        use mir::Rvalue as R;
        if flatten(pointee_ty, &self.func.name)? != 1 {
            return Err(self.err("assignment through a pointer to an aggregate value"));
        }
        let tmp = self.fresh();
        match rvalue {
            R::Use(op) => {
                let v = self.operand_value(op, pointee_ty)?;
                self.emit(Instr::Copy {
                    dst: tmp,
                    src: IrOperand::Value(v),
                });
            }
            R::BinaryOp(op, pair) => {
                let (lhs, rhs) = &**pair;
                self.lower_binop_to(tmp, pointee_ty, *op, lhs, rhs)?;
            }
            R::UnaryOp(op, operand) => {
                self.lower_unop_to(tmp, pointee_ty, *op, operand)?;
            }
            R::Cast(kind, operand, target) => {
                self.lower_cast_to(tmp, kind, operand, target)?;
            }
            R::Ref(_, _, target) | R::RawPtr(_, target) => {
                let v = self.ref_value(target)?;
                self.emit(Instr::Copy {
                    dst: tmp,
                    src: IrOperand::Value(v),
                });
            }
            other => {
                return Err(self.err(&format!(
                    "{} rvalue assigned through a dereference",
                    other.kind_name()
                )));
            }
        }
        self.emit(Instr::StorePtr {
            ptr: IrOperand::Value(ptr_slot),
            src: IrOperand::Value(tmp),
        });
        Ok(())
    }

    fn lower_assign(&mut self, place: &mir::Place, rvalue: &mir::Rvalue) -> Result<(), JitError> {
        use mir::Rvalue as R;
        // An assignment through a dereference (`*p = ...`) computes
        // the value into a scratch slot and stores it through the
        // pointer, so the write lands on the original place.
        if let Some((ptr_slot, pointee_ty)) = self.deref_place(place)? {
            return self.lower_assign_through_ptr(ptr_slot, &pointee_ty, rvalue);
        }
        match rvalue {
            R::Use(op) => {
                let (dst, dst_count, dst_ty) = self.place_leaves(place)?;
                match op {
                    mir::Operand::Copy(p) | mir::Operand::Move(p) => {
                        if let Some((ptr_slot, _)) = self.deref_place(p)? {
                            if dst_count != 1 {
                                return Err(
                                    self.err("Use copy with mismatched value shapes")
                                );
                            }
                            self.emit(Instr::LoadPtr {
                                dst,
                                ptr: IrOperand::Value(ptr_slot),
                            });
                            return Ok(());
                        }
                        let (src, src_count, _) = self.place_leaves(p)?;
                        if src_count != dst_count {
                            return Err(self.err("Use copy with mismatched value shapes"));
                        }
                        for k in 0..dst_count as u32 {
                            self.emit(Instr::Copy {
                                dst: dst + k,
                                src: IrOperand::Value(src + k),
                            });
                        }
                        Ok(())
                    }
                    mir::Operand::Constant(c) => {
                        if dst_count == 1 {
                            let v = self.const_i64(c, &dst_ty)?;
                            self.emit(Instr::Const { dst, imm: v });
                        } else if dst_count > 1 {
                            return Err(self.err("constant of aggregate type"));
                        }
                        // A zero-leaf (unit / never) destination has
                        // nothing to store.
                        Ok(())
                    }
                }
            }
            R::BinaryOp(op, pair) => {
                let (lhs, rhs) = &**pair;
                let (dst, _, dst_ty) = self.place_leaves(place)?;
                self.lower_binop_to(dst, &dst_ty, *op, lhs, rhs)
            }
            R::CheckedBinaryOp(op, pair) => {
                let (lhs, rhs) = &**pair;
                let (dst, _, dst_ty) = self.place_leaves(place)?;
                self.lower_checked_binop_to(dst, &dst_ty, *op, lhs, rhs)
            }
            R::UnaryOp(op, operand) => {
                let (dst, count, dst_ty) = self.place_leaves(place)?;
                if count != 1 {
                    return Err(self.err("unary operation on an aggregate value"));
                }
                self.lower_unop_to(dst, &dst_ty, *op, operand)
            }
            R::Cast(kind, operand, target) => {
                let (dst, count, _) = self.place_leaves(place)?;
                if count != 1 {
                    return Err(self.err("cast of an aggregate value"));
                }
                self.lower_cast_to(dst, kind, operand, target)
            }
            R::Aggregate(kind, ops) => self.lower_aggregate(place, kind, ops),
            R::Repeat(op, count) => self.lower_repeat(place, op, *count),
            R::Ref(_, _, target) => self.lower_ref_creation(place, target),
            R::ThreadLocalRef(_) => Err(self.err("ThreadLocalRef rvalue")),
            R::RawPtr(_, target) => self.lower_ref_creation(place, target),
            R::Len(_) => Err(self.err("Len rvalue")),
            R::Discriminant(src_place) => {
                let (dst, dst_count, dst_ty) = self.place_leaves(place)?;
                if dst_count != 1 {
                    return Err(self.err("Discriminant result is not a scalar"));
                }
                let (src, src_count, src_ty) = self.place_leaves(src_place)?;
                if !matches!(src_ty, mir::Ty::Adt(..)) {
                    return Err(self.err("Discriminant of a non-ADT place"));
                }
                if src_count != 1 {
                    return Err(self.err("Discriminant of an ADT with a payload shape"));
                }
                // The ADT's single leaf holds the discriminant value
                // itself; the read canonicalises it through the
                // destination's (discriminant) type, so `SwitchInt`
                // on the result compares against the same values the
                // interpreter produces.
                self.emit(Instr::Copy {
                    dst,
                    src: IrOperand::Value(src),
                });
                self.normalize(dst, &dst_ty);
                Ok(())
            }
            R::NullaryOp(..) => Err(self.err("NullaryOp rvalue")),
            R::ShallowInitBox(..) => Err(self.err("ShallowInitBox rvalue")),
            R::CopyForDeref(_) => Err(self.err("CopyForDeref rvalue")),
        }
    }

    /// `Rvalue::Repeat`: build the destination array by copying one
    /// element value into every element slot. The element is
    /// evaluated once, as in the interpreter (which materialises the
    /// same value `count` times). Element shapes the slot model
    /// cannot hold never reach here — `flatten` has already rejected
    /// the destination's type — and a constant element of aggregate
    /// type stays unsupported, as aggregate constants are elsewhere.
    fn lower_repeat(
        &mut self,
        place: &mir::Place,
        op: &mir::Operand,
        count: u64,
    ) -> Result<(), JitError> {
        // The interpreter's conversion bound on repeat counts,
        // mirrored so both engines accept the same programs.
        if count > 1_000_000 {
            return Err(self.err(&format!("Repeat of {count} elements")));
        }
        let (dst, dst_count, dst_ty) = self.place_leaves(place)?;
        let mir::Ty::Array(elem_ty, _) = &dst_ty else {
            return Err(self.err("Repeat rvalue with a non-array destination"));
        };
        let elem_ty = (**elem_ty).clone();
        let per = flatten(&elem_ty, &self.func.name)?;
        if per == 0 {
            // A zero-leaf element (unit / never) stores nothing; the
            // destination necessarily has zero leaves too.
            return Ok(());
        }
        let total = (count as usize).checked_mul(per).unwrap_or(usize::MAX);
        if total != dst_count {
            return Err(self.err("Repeat count does not match the array length"));
        }
        match op {
            mir::Operand::Copy(p) | mir::Operand::Move(p) => {
                let (src, src_count, _) = self.place_leaves(p)?;
                if src_count != per {
                    return Err(self.err("Repeat element shape mismatch"));
                }
                for i in 0..count as usize {
                    for k in 0..per as u32 {
                        self.emit(Instr::Copy {
                            dst: dst + (i * per) as u32 + k,
                            src: IrOperand::Value(src + k),
                        });
                    }
                }
                Ok(())
            }
            mir::Operand::Constant(_) => {
                if per != 1 {
                    return Err(self.err("Repeat of a constant aggregate element"));
                }
                let v = self.operand_value(op, &elem_ty)?;
                for i in 0..count as u32 {
                    self.emit(Instr::Copy {
                        dst: dst + i,
                        src: IrOperand::Value(v),
                    });
                }
                Ok(())
            }
        }
    }

    /// The base operation of a plain or `*WithOverflow` MIR operator,
    /// for the checked path.
    fn checked_base(op: mir::BinOp) -> Option<CheckedOp> {
        use mir::BinOp as M;
        match op {
            M::Add | M::AddWithOverflow => Some(CheckedOp::Add),
            M::Sub | M::SubWithOverflow => Some(CheckedOp::Sub),
            M::Mul | M::MulWithOverflow => Some(CheckedOp::Mul),
            _ => None,
        }
    }

    fn lower_checked_binop_to(
        &mut self,
        dst: u32,
        dst_ty: &mir::Ty,
        op: mir::BinOp,
        lhs: &mir::Operand,
        rhs: &mir::Operand,
    ) -> Result<(), JitError> {
        if matches!(op, mir::BinOp::Shl | mir::BinOp::Shr) {
            return self.lower_checked_shift_to(dst, dst_ty, op == mir::BinOp::Shl, lhs, rhs);
        }
        let Some(base) = Self::checked_base(op) else {
            return Err(self.err(&format!("checked operation {op:?}")));
        };
        let mir::Ty::Tuple(fields) = dst_ty else {
            return Err(self.err("checked operation result is not a (value, bool) tuple"));
        };
        if fields.len() != 2 {
            return Err(self.err("checked operation result is not a (value, bool) tuple"));
        }
        let val_ty = fields[0].clone();
        let Some(kind) = int_kind(&val_ty) else {
            return Err(self.err("checked operation on a non-integer type"));
        };
        let lv = self.operand_value(lhs, &val_ty)?;
        let rv = self.operand_value(rhs, &val_ty)?;
        self.emit(Instr::CheckedBinOp {
            dst_val: dst,
            dst_flag: dst + 1,
            op: base,
            lhs: IrOperand::Value(lv),
            rhs: IrOperand::Value(rv),
            kind,
        });
        Ok(())
    }

    fn lower_binop_to(
        &mut self,
        dst: u32,
        dst_ty: &mir::Ty,
        op: mir::BinOp,
        lhs: &mir::Operand,
        rhs: &mir::Operand,
    ) -> Result<(), JitError> {
        use mir::BinOp as M;
        // A plain BinaryOp with a *WithOverflow operator also yields
        // the (value, bool) pair.
        if Self::checked_base(op).is_some()
            && matches!(op, M::AddWithOverflow | M::SubWithOverflow | M::MulWithOverflow)
        {
            return self.lower_checked_binop_to(dst, dst_ty, op, lhs, rhs);
        }
        if flatten(dst_ty, &self.func.name)? != 1 {
            return Err(self.err("binary operation on an aggregate value"));
        }
        let is_compare = matches!(op, M::Eq | M::Ne | M::Lt | M::Le | M::Gt | M::Ge);
        // Arithmetic operands share the destination's type; comparison
        // operands share each other's type (the destination is bool,
        // which says nothing about them). A comparison of two literals
        // — the shape of rustc's expanded division checks, `Eq(4, 0)`
        // — has no place type anywhere, so the pair is typed from the
        // literals themselves (see literal_pair_ty); typing it by the
        // bool destination would collapse every nonzero literal to 1.
        let op_ty = if is_compare {
            self.operand_place_ty(lhs)
                .or_else(|| self.operand_place_ty(rhs))
                .or_else(|| literal_pair_ty(lhs, rhs))
                .or_else(|| literal_float_pair_ty(lhs, rhs))
                .unwrap_or_else(|| dst_ty.clone())
        } else {
            dst_ty.clone()
        };
        if let Some(fk) = float_kind(&op_ty) {
            return self.lower_float_binop(dst, op, lhs, rhs, fk, &op_ty);
        }
        let Some(kind) = int_kind(&op_ty) else {
            return Err(self.err(&format!(
                "binary operation {op:?} on type {}",
                op_ty.kind_name()
            )));
        };
        let lv = self.operand_value(lhs, &op_ty)?;
        let rv = self.operand_value(rhs, &op_ty)?;
        let signed = kind.signed;
        match op {
            M::Add | M::AddUnchecked | M::Sub | M::SubUnchecked | M::Mul | M::MulUnchecked => {
                // In the interpreter's model a plain arithmetic
                // operator panics on overflow; compile it as the
                // checked operation plus a branch on the flag.
                let (base, name) = match op {
                    M::Add | M::AddUnchecked => (CheckedOp::Add, "Add"),
                    M::Sub | M::SubUnchecked => (CheckedOp::Sub, "Sub"),
                    _ => (CheckedOp::Mul, "Mul"),
                };
                self.lower_plain_arith(dst, base, name, lv, rv, kind)?;
            }
            M::BitAnd | M::BitOr | M::BitXor => {
                let ir_op = match op {
                    M::BitAnd => IrBinOp::And,
                    M::BitOr => IrBinOp::Or,
                    _ => IrBinOp::Xor,
                };
                self.emit(Instr::BinOp {
                    dst,
                    op: ir_op,
                    lhs: IrOperand::Value(lv),
                    rhs: IrOperand::Value(rv),
                });
                self.normalize(dst, dst_ty);
            }
            M::Div | M::Rem => {
                self.lower_div_rem(dst, op == M::Rem, lv, rv, kind, dst_ty)?;
            }
            M::Shl | M::ShlUnchecked | M::Shr | M::ShrUnchecked => {
                self.lower_shift(
                    dst,
                    matches!(op, M::Shl | M::ShlUnchecked),
                    lv,
                    rv,
                    kind,
                    dst_ty,
                )?;
            }
            M::Eq | M::Ne | M::Lt | M::Le | M::Gt | M::Ge => {
                let ir_op = match (op, signed) {
                    (M::Eq, _) => IrBinOp::Eq,
                    (M::Ne, _) => IrBinOp::Ne,
                    (M::Lt, true) => IrBinOp::Lt,
                    (M::Lt, false) => IrBinOp::LtU,
                    (M::Le, true) => IrBinOp::Le,
                    (M::Le, false) => IrBinOp::LeU,
                    (M::Gt, true) => IrBinOp::Gt,
                    (M::Gt, false) => IrBinOp::GtU,
                    (M::Ge, true) => IrBinOp::Ge,
                    (M::Ge, false) => IrBinOp::GeU,
                    _ => unreachable!(),
                };
                self.emit(Instr::BinOp {
                    dst,
                    op: ir_op,
                    lhs: IrOperand::Value(lv),
                    rhs: IrOperand::Value(rv),
                });
            }
            M::Offset => return Err(self.err("Offset binary operation")),
            M::Cmp => return Err(self.err("Cmp binary operation")),
            M::AddWithOverflow | M::SubWithOverflow | M::MulWithOverflow => unreachable!(),
        }
        Ok(())
    }

    /// A plain `BinaryOp` on floats: arithmetic and comparisons map
    /// directly onto the backend's float operations (the hardware's
    /// IEEE semantics are Rust's, including the NaN comparison
    /// rules). Bitwise and shift operators have no float form.
    fn lower_float_binop(
        &mut self,
        dst: u32,
        op: mir::BinOp,
        lhs: &mir::Operand,
        rhs: &mir::Operand,
        kind: IrFloatKind,
        op_ty: &mir::Ty,
    ) -> Result<(), JitError> {
        use mir::BinOp as M;
        let fop = match op {
            M::Add | M::AddUnchecked => IrFloatBinOp::Add,
            M::Sub | M::SubUnchecked => IrFloatBinOp::Sub,
            M::Mul | M::MulUnchecked => IrFloatBinOp::Mul,
            M::Div => IrFloatBinOp::Div,
            M::Rem => IrFloatBinOp::Rem,
            M::Eq => IrFloatBinOp::Eq,
            M::Ne => IrFloatBinOp::Ne,
            M::Lt => IrFloatBinOp::Lt,
            M::Le => IrFloatBinOp::Le,
            M::Gt => IrFloatBinOp::Gt,
            M::Ge => IrFloatBinOp::Ge,
            other => {
                return Err(self.err(&format!(
                    "binary operation {other:?} on a floating-point type"
                )));
            }
        };
        let lv = self.operand_value(lhs, op_ty)?;
        let rv = self.operand_value(rhs, op_ty)?;
        self.emit(Instr::FloatBinOp {
            dst,
            op: fop,
            kind,
            lhs: IrOperand::Value(lv),
            rhs: IrOperand::Value(rv),
        });
        Ok(())
    }

    /// Plain `Add` / `Sub` / `Mul`: the checked backend operation,
    /// then a branch on its overflow flag to the interpreter's
    /// overflow panic. (The checked result is already normalised to
    /// the type by the backend, so no further fixup is needed.)
    fn lower_plain_arith(
        &mut self,
        dst: u32,
        base: CheckedOp,
        name: &str,
        lv: u32,
        rv: u32,
        kind: IrIntKind,
    ) -> Result<(), JitError> {
        let msg = self
            .messages
            .intern(&format!("arithmetic overflow in {name}"));
        let pblk = self.panic_block(msg);
        let flag = self.fresh();
        self.emit(Instr::CheckedBinOp {
            dst_val: dst,
            dst_flag: flag,
            op: base,
            lhs: IrOperand::Value(lv),
            rhs: IrOperand::Value(rv),
            kind,
        });
        let cont = self.new_block();
        self.seal_and_continue(
            IrTerminator::Branch {
                cond: IrOperand::Value(flag),
                then_blk: pblk,
                else_blk: cont,
            },
            cont,
        );
        Ok(())
    }

    /// Plain division / remainder with the guards the interpreter's
    /// semantics (and the hardware) need: a zero divisor panics with
    /// the divide-by-zero message, and the signed `MIN / -1` case —
    /// an overflow in the interpreter's model, and a hardware trap
    /// at 64 bits — panics with the arithmetic-overflow message.
    fn lower_div_rem(
        &mut self,
        dst: u32,
        is_rem: bool,
        lv: u32,
        rv: u32,
        kind: IrIntKind,
        dst_ty: &mir::Ty,
    ) -> Result<(), JitError> {
        let msg = self.messages.intern(if is_rem {
            "attempt to calculate the remainder with a divisor of zero"
        } else {
            "attempt to divide by zero"
        });
        let pblk = self.panic_block(msg);
        let nonzero = self.fresh();
        self.emit(Instr::BinOp {
            dst: nonzero,
            op: IrBinOp::Ne,
            lhs: IrOperand::Value(rv),
            rhs: IrOperand::Const(0),
        });
        let cont = self.new_block();
        self.seal_and_continue(
            IrTerminator::Branch {
                cond: IrOperand::Value(nonzero),
                then_blk: cont,
                else_blk: pblk,
            },
            cont,
        );
        if kind.signed {
            let name = if is_rem { "Rem" } else { "Div" };
            let msg = self
                .messages
                .intern(&format!("arithmetic overflow in {name}"));
            let pblk = self.panic_block(msg);
            let min = if kind.bits == 64 {
                i64::MIN
            } else {
                -(1i64 << (kind.bits - 1))
            };
            let is_min = self.fresh();
            self.emit(Instr::BinOp {
                dst: is_min,
                op: IrBinOp::Eq,
                lhs: IrOperand::Value(lv),
                rhs: IrOperand::Const(min),
            });
            let is_neg1 = self.fresh();
            self.emit(Instr::BinOp {
                dst: is_neg1,
                op: IrBinOp::Eq,
                lhs: IrOperand::Value(rv),
                rhs: IrOperand::Const(-1),
            });
            let both = self.fresh();
            self.emit(Instr::BinOp {
                dst: both,
                op: IrBinOp::And,
                lhs: IrOperand::Value(is_min),
                rhs: IrOperand::Value(is_neg1),
            });
            let cont = self.new_block();
            self.seal_and_continue(
                IrTerminator::Branch {
                    cond: IrOperand::Value(both),
                    then_blk: pblk,
                    else_blk: cont,
                },
                cont,
            );
        }
        let ir_op = match (is_rem, kind.signed) {
            (false, true) => IrBinOp::Div,
            (false, false) => IrBinOp::DivU,
            (true, true) => IrBinOp::Rem,
            (true, false) => IrBinOp::RemU,
        };
        self.emit(Instr::BinOp {
            dst,
            op: ir_op,
            lhs: IrOperand::Value(lv),
            rhs: IrOperand::Value(rv),
        });
        self.normalize(dst, dst_ty);
        Ok(())
    }

    /// Plain shift: an amount meeting or exceeding the type width
    /// panics with Rust 1.100.0's message for the direction
    /// (`attempt to shift left with overflow` / `attempt to shift
    /// right with overflow`), as the interpreter's plain form does.
    /// Bits shifted out of the width are simply lost, as natively.
    fn lower_shift(
        &mut self,
        dst: u32,
        is_shl: bool,
        lv: u32,
        rv: u32,
        kind: IrIntKind,
        dst_ty: &mir::Ty,
    ) -> Result<(), JitError> {
        let bits = kind.bits as u32;
        // Overshift check.
        let over = self.fresh();
        self.emit(Instr::BinOp {
            dst: over,
            op: IrBinOp::GeU,
            lhs: IrOperand::Value(rv),
            rhs: IrOperand::Const(bits as i64),
        });
        let msg = self.messages.intern(if is_shl {
            "attempt to shift left with overflow"
        } else {
            "attempt to shift right with overflow"
        });
        let pblk = self.panic_block(msg);
        let cont = self.new_block();
        self.seal_and_continue(
            IrTerminator::Branch {
                cond: IrOperand::Value(over),
                then_blk: pblk,
                else_blk: cont,
            },
            cont,
        );
        let shift_op = if is_shl {
            IrBinOp::Shl
        } else if kind.signed {
            IrBinOp::Sar
        } else {
            IrBinOp::Shr
        };
        self.emit(Instr::BinOp {
            dst,
            op: shift_op,
            lhs: IrOperand::Value(lv),
            rhs: IrOperand::Value(rv),
        });
        self.normalize(dst, dst_ty);
        Ok(())
    }

    /// Checked shift (`CheckedBinaryOp` with `Shl` / `Shr`): the
    /// value wraps (the amount is taken modulo the width) and the
    /// flag reports an overshift only; bits shifted out are lost
    /// and do not set the flag, as natively.
    fn lower_checked_shift_to(
        &mut self,
        dst: u32,
        dst_ty: &mir::Ty,
        is_shl: bool,
        lhs: &mir::Operand,
        rhs: &mir::Operand,
    ) -> Result<(), JitError> {
        let mir::Ty::Tuple(fields) = dst_ty else {
            return Err(self.err("checked operation result is not a (value, bool) tuple"));
        };
        if fields.len() != 2 {
            return Err(self.err("checked operation result is not a (value, bool) tuple"));
        }
        let val_ty = fields[0].clone();
        let Some(kind) = int_kind(&val_ty) else {
            return Err(self.err("checked operation on a non-integer type"));
        };
        let bits = kind.bits as u32;
        let lv = self.operand_value(lhs, &val_ty)?;
        let rv = self.operand_value(rhs, &val_ty)?;
        let over = self.fresh();
        self.emit(Instr::BinOp {
            dst: over,
            op: IrBinOp::GeU,
            lhs: IrOperand::Value(rv),
            rhs: IrOperand::Const(bits as i64),
        });
        let masked = self.fresh();
        self.emit(Instr::BinOp {
            dst: masked,
            op: IrBinOp::And,
            lhs: IrOperand::Value(rv),
            rhs: IrOperand::Const((bits - 1) as i64),
        });
        let shift_op = if is_shl {
            IrBinOp::Shl
        } else if kind.signed {
            IrBinOp::Sar
        } else {
            IrBinOp::Shr
        };
        self.emit(Instr::BinOp {
            dst,
            op: shift_op,
            lhs: IrOperand::Value(lv),
            rhs: IrOperand::Value(masked),
        });
        self.normalize(dst, &val_ty);
        self.emit(Instr::Copy {
            dst: dst + 1,
            src: IrOperand::Value(over),
        });
        Ok(())
    }

    fn lower_unop_to(
        &mut self,
        dst: u32,
        dst_ty: &mir::Ty,
        op: mir::UnOp,
        operand: &mir::Operand,
    ) -> Result<(), JitError> {
        match op {
            mir::UnOp::Neg => {
                if let Some(fk) = float_kind(dst_ty) {
                    // Float negation is an exact sign-bit flip (it
                    // applies to NaNs and zeroes alike), so it lowers
                    // to an integer xor on the value's bit pattern:
                    // the sign bit for f64, bit 31 for f32 (whose
                    // canonical form keeps the upper half zero).
                    let v = self.operand_value(operand, dst_ty)?;
                    let mask = match fk {
                        IrFloatKind::F64 => i64::MIN,
                        IrFloatKind::F32 => 0x8000_0000,
                    };
                    self.emit(Instr::BinOp {
                        dst,
                        op: IrBinOp::Xor,
                        lhs: IrOperand::Value(v),
                        rhs: IrOperand::Const(mask),
                    });
                    return Ok(());
                }
                let mir::Ty::Int(it) = dst_ty else {
                    return Err(self.err("Neg on a non-signed-integer type"));
                };
                let bits = int_bits(*it);
                let v = self.operand_value(operand, dst_ty)?;
                // Negating the type's minimum overflows; the
                // interpreter panics.
                let min = if bits == 64 {
                    i64::MIN
                } else {
                    -(1i64 << (bits - 1))
                };
                let is_min = self.fresh();
                self.emit(Instr::BinOp {
                    dst: is_min,
                    op: IrBinOp::Eq,
                    lhs: IrOperand::Value(v),
                    rhs: IrOperand::Const(min),
                });
                let msg = self.messages.intern("arithmetic overflow in Neg");
                let pblk = self.panic_block(msg);
                let cont = self.new_block();
                self.seal_and_continue(
                    IrTerminator::Branch {
                        cond: IrOperand::Value(is_min),
                        then_blk: pblk,
                        else_blk: cont,
                    },
                    cont,
                );
                self.emit(Instr::UnOp {
                    dst,
                    op: IrUnOp::Neg,
                    src: IrOperand::Value(v),
                });
                self.normalize(dst, dst_ty);
                Ok(())
            }
            mir::UnOp::Not => {
                let v = self.operand_value(operand, dst_ty)?;
                if matches!(dst_ty, mir::Ty::Bool) {
                    // Logical not on a canonical 0/1 value.
                    self.emit(Instr::BinOp {
                        dst,
                        op: IrBinOp::Xor,
                        lhs: IrOperand::Value(v),
                        rhs: IrOperand::Const(1),
                    });
                } else {
                    self.emit(Instr::UnOp {
                        dst,
                        op: IrUnOp::Not,
                        src: IrOperand::Value(v),
                    });
                    self.normalize(dst, dst_ty);
                }
                Ok(())
            }
            mir::UnOp::PtrMetadata => Err(self.err("PtrMetadata unary operation")),
        }
    }

    fn lower_cast_to(
        &mut self,
        dst: u32,
        kind: &mir::CastKind,
        operand: &mir::Operand,
        target: &mir::Ty,
    ) -> Result<(), JitError> {
        match kind {
            mir::CastKind::IntToInt => {
                if int_kind(target).is_none() {
                    return Err(self.err(&format!(
                        "cast to type {}",
                        target.kind_name()
                    )));
                }
                // The source value is canonical for its own type, so
                // casting is exactly normalising that 64-bit value to
                // the target type.
                let src_ty = self
                    .operand_place_ty(operand)
                    .unwrap_or_else(|| target.clone());
                let v = self.operand_value(operand, &src_ty)?;
                self.emit(Instr::Copy {
                    dst,
                    src: IrOperand::Value(v),
                });
                self.normalize(dst, target);
                Ok(())
            }
            mir::CastKind::FloatToInt => {
                let Some(src_ty) = self.operand_float_ty(operand) else {
                    return Err(self.err("FloatToInt cast with an unknown source type"));
                };
                let fk = float_kind(&src_ty).expect("operand_float_ty yields a float kind");
                let Some(int) = int_kind(target) else {
                    return Err(self.err(&format!(
                        "FloatToInt cast to type {}",
                        target.kind_name()
                    )));
                };
                let v = self.operand_value(operand, &src_ty)?;
                self.emit(Instr::FloatToInt {
                    dst,
                    kind: fk,
                    src: IrOperand::Value(v),
                    int,
                });
                Ok(())
            }
            mir::CastKind::IntToFloat => {
                let Some(fk) = float_kind(target) else {
                    return Err(self.err(&format!(
                        "IntToFloat cast to type {}",
                        target.kind_name()
                    )));
                };
                // The source type comes from the operand's place, or
                // from the literal itself for a constant operand.
                let Some(src_ty) = self
                    .operand_place_ty(operand)
                    .or_else(|| literal_int_ty(operand))
                else {
                    return Err(self.err("IntToFloat cast with an unknown source type"));
                };
                let Some(int) = int_kind(&src_ty) else {
                    return Err(self.err("IntToFloat cast from a non-integer type"));
                };
                let v = self.operand_value(operand, &src_ty)?;
                self.emit(Instr::IntToFloat {
                    dst,
                    kind: fk,
                    src: IrOperand::Value(v),
                    int,
                });
                Ok(())
            }
            mir::CastKind::FloatToFloat => {
                let Some(src_ty) = self.operand_float_ty(operand) else {
                    return Err(self.err("FloatToFloat cast with an unknown source type"));
                };
                let src_kind = float_kind(&src_ty).expect("operand_float_ty yields a float kind");
                let Some(fk) = float_kind(target) else {
                    return Err(self.err(&format!(
                        "FloatToFloat cast to type {}",
                        target.kind_name()
                    )));
                };
                let v = self.operand_value(operand, &src_ty)?;
                if src_kind == fk {
                    self.emit(Instr::Copy {
                        dst,
                        src: IrOperand::Value(v),
                    });
                } else {
                    self.emit(Instr::FloatToFloat {
                        dst,
                        kind: fk,
                        src: IrOperand::Value(v),
                        src_kind,
                    });
                }
                Ok(())
            }
            // Pointer casts that keep the address word unchanged:
            // the interpreter converts exactly these to a plain
            // `Use`, and a JIT pointer *is* the address word, so
            // the cast is a copy. Unsize coercions would have to
            // build a fat pointer and stay unsupported, as do the
            // provenance casts, which the interpreter also rejects.
            mir::CastKind::PtrToPtr
            | mir::CastKind::PointerCoercion(
                mir::PointerCoercion::MutToConstPointer
                | mir::PointerCoercion::ArrayToPointer,
            ) => {
                let src_ty = self
                    .operand_place_ty(operand)
                    .unwrap_or_else(|| target.clone());
                let v = self.operand_value(operand, &src_ty)?;
                self.emit(Instr::Copy {
                    dst,
                    src: IrOperand::Value(v),
                });
                Ok(())
            }
            other => Err(self.err(&format!("cast {other:?}"))),
        }
    }

    fn lower_aggregate(
        &mut self,
        place: &mir::Place,
        kind: &mir::AggregateKind,
        ops: &[mir::Operand],
    ) -> Result<(), JitError> {
        match kind {
            mir::AggregateKind::Tuple => {}
            mir::AggregateKind::Adt {
                def, variant_index, ..
            } => {
                if !ops.is_empty() {
                    return Err(self.err("ADT aggregate with fields"));
                }
                let (dst, count, _) = self.place_leaves(place)?;
                if count != 1 {
                    return Err(self.err("ADT aggregate destination shape mismatch"));
                }
                // A fieldless variant's value is its tag alone.
                let tag = self.discriminant_value(*def, *variant_index)?;
                self.emit(Instr::Const { dst, imm: tag });
                return Ok(());
            }
            other => {
                return Err(self.err(&format!("aggregate {}", other.kind_name())));
            }
        }
        let (dst, count, _) = self.place_leaves(place)?;
        if count != ops.len() {
            return Err(self.err("tuple aggregate with non-scalar fields"));
        }
        for (k, op) in ops.iter().enumerate() {
            // Field types come from the destination tuple's shape;
            // each field here is a single leaf.
            let field_ty = match self.place_leaves(place) {
                Ok((_, _, mir::Ty::Tuple(fields))) => fields[k].clone(),
                _ => return Err(self.err("tuple aggregate destination is not a tuple")),
            };
            let v = self.operand_value(op, &field_ty)?;
            self.emit(Instr::Copy {
                dst: dst + k as u32,
                src: IrOperand::Value(v),
            });
        }
        Ok(())
    }

    // ----- terminators -------------------------------------------------------

    fn lower_terminator(&mut self, term: &mir::TerminatorKind) -> Result<(), JitError> {
        use mir::TerminatorKind as T;
        match term {
            T::Goto { target } => {
                self.seal_terminator(IrTerminator::Jump(target.index() as u32));
                Ok(())
            }
            T::Return => {
                let (slot, count) = self.local_slots[0];
                let term = match count {
                    0 => IrTerminator::Return(IrOperand::Const(0)),
                    1 => IrTerminator::Return(IrOperand::Value(slot)),
                    _ => return Err(self.err("multi-word return value")),
                };
                self.seal_terminator(term);
                Ok(())
            }
            T::Unreachable => {
                // Reaching an `Unreachable` block is a guest panic,
                // exactly the interpreter's model: lower it to a jump
                // into a panic block, the same mechanism `Assert`
                // uses, so the panic is recorded through the runtime
                // path and propagates via the program's panic flag.
                // (Lowering it to the backend's trap terminator made
                // the generated code execute int3 and kill the host
                // process with SIGTRAP.)
                let msg_idx = self.messages.intern("entered unreachable code");
                let pblk = self.panic_block(msg_idx);
                self.seal_terminator(IrTerminator::Jump(pblk));
                Ok(())
            }
            T::SwitchInt { discr, targets } => self.lower_switch(discr, targets),
            T::Call {
                func,
                args,
                destination,
                target,
                ..
            } => self.lower_call(func, args, destination, *target),
            T::Assert {
                cond,
                expected,
                msg,
                target,
                ..
            } => {
                let text = assert_message_text(msg);
                let msg_idx = self.messages.intern(&text);
                let pblk = self.panic_block(msg_idx);
                let v = self.operand_value(cond, &mir::Ty::Bool)?;
                let (then_blk, else_blk) = if *expected {
                    (target.index() as u32, pblk)
                } else {
                    (pblk, target.index() as u32)
                };
                self.seal_terminator(IrTerminator::Branch {
                    cond: IrOperand::Value(v),
                    then_blk,
                    else_blk,
                });
                Ok(())
            }
            T::Drop { place, target, .. } => {
                // Dropping a scalar/tuple-of-scalars value runs no
                // code; anything else is outside the supported range.
                // Dropping through a dereference runs no code either.
                if self.deref_place(place)?.is_none() {
                    self.place_leaves(place)?;
                }
                self.seal_terminator(IrTerminator::Jump(target.index() as u32));
                Ok(())
            }
            T::FalseEdge { real_target, .. } | T::FalseUnwind { real_target, .. } => {
                self.seal_terminator(IrTerminator::Jump(real_target.index() as u32));
                Ok(())
            }
            T::Resume => Err(self.err("Resume terminator")),
            T::Terminate(_) => Err(self.err("Terminate terminator")),
            T::Yield { .. } => Err(self.err("Yield terminator")),
            T::CoroutineDrop => Err(self.err("CoroutineDrop terminator")),
            T::InlineAsm { .. } => Err(self.err("InlineAsm terminator")),
            T::TailCall { .. } => Err(self.err("TailCall terminator")),
        }
    }

    /// Seal the current MIR block with its terminator. The
    /// continuation id passed to `seal_and_continue` is the block
    /// itself: `run` re-initialises `cur`/`cur_id` for the next MIR
    /// block before anything else is emitted, so the stale
    /// continuation is never filled.
    fn seal_terminator(&mut self, term: IrTerminator) {
        let id = self.cur_id;
        self.seal_and_continue(term, id);
    }

    fn lower_switch(
        &mut self,
        discr: &mir::Operand,
        targets: &mir::SwitchTargets,
    ) -> Result<(), JitError> {
        let discr_ty = self
            .operand_place_ty(discr)
            .unwrap_or(mir::Ty::Uint(mir::UintTy::U64));
        let v = self.operand_value(discr, &discr_ty)?;
        if targets.values.is_empty() {
            self.seal_terminator(IrTerminator::Jump(targets.otherwise.index() as u32));
            return Ok(());
        }
        for (i, (value, target)) in targets
            .values
            .iter()
            .zip(targets.targets.iter())
            .enumerate()
        {
            let eq = self.fresh();
            self.emit(Instr::BinOp {
                dst: eq,
                op: IrBinOp::Eq,
                lhs: IrOperand::Value(v),
                rhs: IrOperand::Const(normalize_u128(*value, &discr_ty)),
            });
            let last = i + 1 == targets.values.len();
            let next = if last {
                targets.otherwise.index() as u32
            } else {
                self.new_block()
            };
            self.seal_and_continue(
                IrTerminator::Branch {
                    cond: IrOperand::Value(eq),
                    then_blk: target.index() as u32,
                    else_blk: next,
                },
                next,
            );
            // When `last`, `next` is the `otherwise` MIR block:
            // seal_and_continue only moves the fill cursor there, and
            // run() re-initialises the cursor before that block is
            // filled with its own contents. The block itself is not
            // sealed here.
        }
        Ok(())
    }

    fn lower_call(
        &mut self,
        func: &mir::Operand,
        args: &[mir::Operand],
        destination: &mir::Place,
        target: Option<mir::BasicBlock>,
    ) -> Result<(), JitError> {
        // Resolve the callee: only direct calls to functions of this
        // program (a FnDef constant naming a known DefId).
        let def = match func {
            mir::Operand::Constant(c) => match &c.literal {
                mir::ConstKind::Ty(mir::Ty::FnDef(def, _)) => Some(*def),
                mir::ConstKind::Unevaluated { def, .. } => Some(*def),
                _ => None,
            },
            _ => None,
        };
        let Some(def) = def else {
            return Err(self.err("call through a non-constant function value"));
        };
        let Some(&callee) = self.def_index.get(&def) else {
            let name = self
                .def_names
                .get(&def)
                .cloned()
                .unwrap_or_else(|| format!("{def}"));
            return Err(self.err(&format!(
                "call to function `{name}` which is not part of the program"
            )));
        };
        let callee_tys = &self.param_tys[callee];
        if args.len() != callee_tys.len() {
            return Err(self.err("call argument count does not match the callee"));
        }
        let mut ir_args: Vec<IrOperand> = Vec::new();
        for (arg, pty) in args.iter().zip(callee_tys.iter()) {
            match flatten(pty, &self.func.name)? {
                0 => { /* unit argument: passed as a dummy register, skipped here */ }
                1 => {
                    let v = self.operand_value(arg, pty)?;
                    ir_args.push(IrOperand::Value(v));
                }
                _ => return Err(self.err("call with a multi-word argument")),
            }
        }
        if ir_args.len() > 6 {
            return Err(self.err("call with more than 6 arguments"));
        }
        // A call whose destination is a dereference (`*p = f()`)
        // receives into a scratch slot and stores through the
        // pointer after the panic check, so a panicking callee
        // leaves the pointee untouched.
        let deref_dst = self.deref_place(destination)?;
        let (dst, count) = match &deref_dst {
            Some(_) => (self.fresh(), 1),
            None => {
                let (dst, count, _) = self.place_leaves(destination)?;
                (dst, count)
            }
        };
        if count > 1 {
            return Err(self.err("call returning a multi-word value"));
        }
        let dst = if count == 1 { dst } else { self.fresh() };
        self.emit(Instr::Call {
            dst,
            func: callee as u32,
            args: ir_args,
        });
        self.emit(Instr::CheckPanic);
        if let Some((ptr_slot, _)) = deref_dst {
            self.emit(Instr::StorePtr {
                ptr: IrOperand::Value(ptr_slot),
                src: IrOperand::Value(dst),
            });
        }
        match target {
            Some(t) => self.seal_terminator(IrTerminator::Jump(t.index() as u32)),
            None => self.seal_terminator(IrTerminator::Unreachable),
        }
        Ok(())
    }
}
