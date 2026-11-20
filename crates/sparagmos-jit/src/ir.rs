//! The JIT backend's function IR: the internal representation the
//! x86_64 compiler ([`crate::compiler`]) consumes.
//!
//! A function is a list of basic blocks, each block is a list of
//! instructions plus a terminator. Every value lives in a fixed stack
//! slot and is a raw 64-bit word; typed meaning (integer width and
//! signedness) is carried by the operations that produce and consume
//! the value, not by the value itself. The MIR front end
//! ([`crate::mir_jit`]) keeps every integer in canonical form for its
//! type (sign-extended for signed types, zero-extended / masked for
//! unsigned types and `bool`), so the plain 64-bit operations below
//! are exact for any integer type of up to 64 bits.
//!
//! A floating-point value occupies one slot too, holding its IEEE
//! 754 bit pattern: all 64 bits for `f64`, the low 32 bits for `f32`
//! with the upper half zero. The compiler moves the pattern into an
//! SSE register, computes there, and stores the pattern back, so the
//! slot invariant is the same canonical-form story as for integers.
//!
//! The IR is deliberately *not* SSA: an instruction may write into a
//! value slot that was written before (MIR locals are mutable, and the
//! lowering maps each MIR local leaf onto one value slot).

/// A virtual value (stack slot): defined by being a function parameter
/// (parameters are values `0..param_count`) or by an instruction.
pub type Value = u32;

/// Block identifier (index into `Function::blocks`).
pub type BlockId = u32;

/// An operand: either a previously defined value or an inline constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    Value(Value),
    Const(i64),
}

impl From<i64> for Operand {
    fn from(v: i64) -> Self {
        Operand::Const(v)
    }
}
impl From<Value> for Operand {
    fn from(v: Value) -> Self {
        Operand::Value(v)
    }
}

/// Binary operations. Comparisons yield 0 or 1.
///
/// The plain arithmetic operations work on raw 64-bit words. Division
/// and remainder come in signed (`Div` / `Rem`, truncating toward
/// zero, like Rust's `/` and `%` on signed integers) and unsigned
/// (`DivU` / `RemU`) forms; comparisons likewise come in signed and
/// unsigned (`U`-suffixed) forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    /// Signed division (truncates toward zero, like Rust `/` on i64).
    Div,
    /// Signed remainder (like Rust `%` on i64).
    Rem,
    /// Unsigned division.
    DivU,
    /// Unsigned remainder.
    RemU,
    And,
    Or,
    Xor,
    /// Shift left (shift amount taken modulo 64, as on x86_64).
    Shl,
    /// Logical shift right.
    Shr,
    /// Arithmetic shift right.
    Sar,
    Eq,
    Ne,
    /// Signed comparisons.
    Lt,
    Le,
    Gt,
    Ge,
    /// Unsigned comparisons.
    LtU,
    LeU,
    GtU,
    GeU,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

/// A floating-point format the backend computes in. (`f16` and
/// `f128` have no SSE scalar form and stay unsupported.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatKind {
    F32,
    F64,
}

impl FloatKind {
    pub const fn bits(self) -> u32 {
        match self {
            FloatKind::F32 => 32,
            FloatKind::F64 => 64,
        }
    }
}

/// Floating-point binary operations. The arithmetic forms follow
/// IEEE 754 as Rust exposes it: division by zero yields an infinity
/// (or NaN for 0/0), overflow yields an infinity, and `Rem` is
/// Rust's `%` (the exact `fmod` remainder, sign of the dividend).
/// The comparisons yield 0 or 1 with Rust's NaN rule: every
/// comparison involving a NaN is false, except `Ne`, which is true.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// The integer operations that have a checked (overflow-flagged) form
/// in MIR: `a op b` yields the wrapped value plus a flag telling
/// whether the true result fits the operation's type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckedOp {
    Add,
    Sub,
    Mul,
}

/// An integer type as the backend needs it for a checked operation:
/// the bit width of the MIR type and its signedness. The operands are
/// in canonical form for this type (see the module docs), which is
/// what makes the backend's overflow computation exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntKind {
    pub bits: u8,
    pub signed: bool,
}

impl IntKind {
    pub const fn new(bits: u8, signed: bool) -> Self {
        Self { bits, signed }
    }
}

/// One instruction. Every instruction that produces a value names its
/// destination value explicitly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instr {
    /// `dst = imm`
    Const { dst: Value, imm: i64 },
    /// `dst = src`
    Copy { dst: Value, src: Operand },
    /// `dst = lhs op rhs`
    BinOp {
        dst: Value,
        op: BinOp,
        lhs: Operand,
        rhs: Operand,
    },
    /// `dst = op src`
    UnOp { dst: Value, op: UnOp, src: Operand },
    /// Checked arithmetic: computes `lhs op rhs` at the given integer
    /// type, stores the wrapped value into `dst_val` and the overflow
    /// flag (0 or 1) into `dst_flag`. This is the backend form of
    /// MIR's `CheckedBinaryOp` and `*WithOverflow` operators, which
    /// yield a `(value, bool)` pair.
    CheckedBinOp {
        dst_val: Value,
        dst_flag: Value,
        op: CheckedOp,
        lhs: Operand,
        rhs: Operand,
        kind: IntKind,
    },
    /// Call function `func` (an index into the program's function
    /// table) with the given arguments (at most six, passed in the
    /// System V argument registers); the result lands in `dst`.
    /// Only meaningful in a program compile ([`crate::program`]);
    /// the single-function entry points reject it.
    Call {
        dst: Value,
        func: u32,
        args: Vec<Operand>,
    },
    /// Guest panic: record message `msg` (an index into the program's
    /// message table) through the runtime's panic path, mark the
    /// program's panic flag, and return from the current function
    /// immediately. Callers are expected to follow a call with
    /// [`Instr::CheckPanic`] so the panic propagates to the entry
    /// point instead of execution continuing with dummy values.
    /// Program compiles only, like [`Instr::Call`].
    Panic { msg: u32 },
    /// If the program's panic flag is set, return from the current
    /// function immediately (propagating a guest panic raised by a
    /// callee). Otherwise a no-op. Program compiles only.
    CheckPanic,
    /// Guest panic with the shift-overflow message, whose text
    /// depends on the runtime shift amount: "attempt to shift by
    /// {amount} bits, which overflows the type width {bits}". The
    /// message is formatted by a helper in the program layer into a
    /// scratch buffer and then recorded through the usual panic
    /// path. Program compiles only.
    PanicShiftOverflow { amount: Operand, bits: u32 },
    /// Guest panic with the index-out-of-bounds message, whose text
    /// depends on the runtime length and index: "index out of
    /// bounds: the len is {len} but the index is {index}". Like
    /// [`Instr::PanicShiftOverflow`], the message is formatted by a
    /// helper in the program layer into a scratch buffer and then
    /// recorded through the usual panic path. Program compiles only.
    PanicIndexOob { len: Operand, index: Operand },
    /// `dst = lhs op rhs` on floating-point values of format `kind`.
    /// Operands and the result are bit patterns in their slots (see
    /// the module docs); for the comparison forms the result is the
    /// integer 0 or 1.
    FloatBinOp {
        dst: Value,
        op: FloatBinOp,
        kind: FloatKind,
        lhs: Operand,
        rhs: Operand,
    },
    /// `dst = src as <int>`: Rust's float-to-integer cast, which
    /// saturates at the target type's bounds and maps NaN to 0.
    /// `kind` is the source float format, `int` the target type.
    FloatToInt {
        dst: Value,
        kind: FloatKind,
        src: Operand,
        int: IntKind,
    },
    /// `dst = src as <float>`: an integer-to-float cast with
    /// round-to-nearest-even, for a source of type `int` and a
    /// result of format `kind`.
    IntToFloat {
        dst: Value,
        kind: FloatKind,
        src: Operand,
        int: IntKind,
    },
    /// `dst = src as <float>`: a float-to-float conversion. Widening
    /// (`f32` to `f64`) is exact; narrowing rounds to nearest even
    /// and overflows to an infinity, as Rust's `as` does.
    FloatToFloat {
        dst: Value,
        kind: FloatKind,
        src: Operand,
        src_kind: FloatKind,
    },
    /// `dst = &slot`: the machine address of value `slot`'s stack
    /// slot. This is how the MIR front end represents a reference or
    /// raw pointer to a local place: the pointer *is* the slot's
    /// address, so loads and stores through it (below) read and
    /// write the place itself. The address is only meaningful while
    /// the current frame (or an ancestor frame, for a pointer passed
    /// down as a call argument) is live.
    SlotAddr { dst: Value, slot: Value },
    /// `dst = *ptr`: load the full 64-bit word at the address held
    /// in `ptr`. A scalar leaf's slot always holds the value's
    /// canonical 64-bit form, so a whole-word load through a pointer
    /// to a slot yields exactly the leaf's value.
    LoadPtr { dst: Value, ptr: Operand },
    /// `*ptr = src`: store the full 64-bit word `src` at the address
    /// held in `ptr` (the counterpart of [`Instr::LoadPtr`]).
    StorePtr { ptr: Operand, src: Operand },
}

/// How a basic block ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Terminator {
    /// Return a value from the function.
    Return(Operand),
    /// Unconditional jump.
    Jump(BlockId),
    /// Jump to `then_blk` if `cond != 0`, else to `else_blk`.
    Branch {
        cond: Operand,
        then_blk: BlockId,
        else_blk: BlockId,
    },
    Unreachable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    pub instrs: Vec<Instr>,
    pub terminator: Terminator,
}

impl BasicBlock {
    pub fn new(instrs: Vec<Instr>, terminator: Terminator) -> Self {
        Self { instrs, terminator }
    }
}

/// A whole function. Entry block is block 0. Parameters are the values
/// `0..param_count`; the highest value id used anywhere must be
/// `< value_count` so the compiler can give every value a stack slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    pub param_count: usize,
    pub value_count: usize,
    pub blocks: Vec<BasicBlock>,
}

/// A tiny builder that makes hand-writing IR in tests and examples bearable.
pub struct FunctionBuilder {
    name: String,
    param_count: usize,
    next_value: Value,
    blocks: Vec<BasicBlock>,
    cur_instrs: Vec<Instr>,
}

impl FunctionBuilder {
    pub fn new(name: &str, param_count: usize) -> Self {
        let mut b = Self {
            name: name.to_string(),
            param_count,
            next_value: 0,
            blocks: Vec::new(),
            cur_instrs: Vec::new(),
        };
        // Reserve value ids for the parameters.
        for _ in 0..param_count {
            b.fresh();
        }
        b
    }
    /// The value id of parameter `i`.
    pub fn param(&self, i: usize) -> Value {
        assert!(i < self.param_count);
        i as Value
    }
    pub fn fresh(&mut self) -> Value {
        let v = self.next_value;
        self.next_value += 1;
        v
    }
    pub fn const_(&mut self, imm: i64) -> Value {
        let dst = self.fresh();
        self.cur_instrs.push(Instr::Const { dst, imm });
        dst
    }
    pub fn binop(&mut self, op: BinOp, lhs: impl Into<Operand>, rhs: impl Into<Operand>) -> Value {
        let dst = self.fresh();
        self.cur_instrs.push(Instr::BinOp {
            dst,
            op,
            lhs: lhs.into(),
            rhs: rhs.into(),
        });
        dst
    }
    /// Push a raw instruction (allows writing back into an existing value's
    /// slot, which the loop examples use for mutable loop variables).
    pub fn push_raw(&mut self, instr: Instr) {
        self.cur_instrs.push(instr);
    }
    pub fn float_binop(
        &mut self,
        op: FloatBinOp,
        kind: FloatKind,
        lhs: impl Into<Operand>,
        rhs: impl Into<Operand>,
    ) -> Value {
        let dst = self.fresh();
        self.cur_instrs.push(Instr::FloatBinOp {
            dst,
            op,
            kind,
            lhs: lhs.into(),
            rhs: rhs.into(),
        });
        dst
    }
    pub fn unop(&mut self, op: UnOp, src: impl Into<Operand>) -> Value {
        let dst = self.fresh();
        self.cur_instrs.push(Instr::UnOp {
            dst,
            op,
            src: src.into(),
        });
        dst
    }
    /// End the current block with `term` and start a new one.
    /// Returns the new block's id.
    pub fn end_block(&mut self, term: Terminator) -> BlockId {
        let instrs = std::mem::take(&mut self.cur_instrs);
        self.blocks.push(BasicBlock::new(instrs, term));
        self.blocks.len() as BlockId // id the *next* block will have
    }
    /// Number of blocks finished so far (== id of the block being built).
    pub fn current_block(&self) -> BlockId {
        self.blocks.len() as BlockId
    }
    pub fn finish(mut self, term: Terminator) -> Function {
        self.end_block(term);
        Function {
            name: self.name,
            param_count: self.param_count,
            value_count: self.next_value as usize,
            blocks: self.blocks,
        }
    }
}
