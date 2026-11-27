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
//! are exact for any integer type of up to 64 bits. A 128-bit integer
//! occupies two consecutive slots — the low word first, then the high
//! word, the layout the interpreter's `i128` / `u128` values, the
//! constant materialisation and the System V pair convention all
//! share — and is computed on by the dedicated pair operations
//! ([`Instr::BinOp128`] and friends).
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

/// The operations on a 128-bit integer, held as a pair of 64-bit
/// words (low word first). Division and remainder come in signed
/// (`Div` / `Rem`, truncating toward zero, like Rust's `/` and `%`
/// on `i128`) and unsigned forms, as for the 64-bit operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp128 {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    DivU,
    RemU,
    And,
    Or,
    Xor,
}

/// The shift operations on a 128-bit integer: `Shl` (left), `Shr`
/// (logical right) and `Sar` (arithmetic right).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftOp128 {
    Shl,
    Shr,
    Sar,
}

/// One instruction. Every instruction that produces a value names its
/// destination value explicitly.
/// A panic site with no known source position: the `loc` field
/// of the panic instructions carries this instead of an index
/// into the program's panic-location table.
pub const NO_LOCATION: u32 = u32::MAX;

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
    /// A binary operation on 128-bit integers: the operands are
    /// the word pairs `(lhs_lo, lhs_hi)` and `(rhs_lo, rhs_hi)`,
    /// and the result's low word lands in `dst`, its high word in
    /// `dst2`. The operation wraps, as the plain MIR operators do.
    BinOp128 {
        dst: Value,
        dst2: Value,
        op: BinOp128,
        lhs_lo: Operand,
        lhs_hi: Operand,
        rhs_lo: Operand,
        rhs_hi: Operand,
    },
    /// A shift of the 128-bit pair `(lo, hi)` by `amount`, which
    /// the front end has already reduced to the range `0..=127`
    /// (the plain operators take the amount modulo the type width;
    /// a checked shift's overshift panic is a separate flag). The
    /// result's low word lands in `dst`, its high word in `dst2`.
    Shift128 {
        dst: Value,
        dst2: Value,
        op: ShiftOp128,
        lo: Operand,
        hi: Operand,
        amount: Operand,
    },
    /// Checked arithmetic on 128-bit integers: like
    /// [`Instr::CheckedBinOp`], but the wrapped value is the word
    /// pair stored into `dst_val` (low) and `dst_val2` (high), and
    /// the overflow flag — for the 128-bit type whose signedness
    /// is `signed` — lands in `dst_flag`.
    CheckedBinOp128 {
        dst_val: Value,
        dst_val2: Value,
        dst_flag: Value,
        op: CheckedOp,
        signed: bool,
        lhs_lo: Operand,
        lhs_hi: Operand,
        rhs_lo: Operand,
        rhs_hi: Operand,
    },
    /// Call function `func` (an index into the program's function
    /// table) with the given argument words: the first six in the
    /// System V argument registers, the rest on the stack in order
    /// (the seventh word at the lowest address), exactly as System V
    /// passes them; the result lands in `dst`, and its second word
    /// — when the callee returns two words — in `dst2`. Results
    /// travel the way System V returns a pair: the first word in
    /// RAX, the second in RDX.
    /// Only meaningful in a program compile ([`crate::program`]);
    /// the single-function entry points reject it.
    Call {
        dst: Value,
        dst2: Option<Value>,
        func: u32,
        args: Vec<Operand>,
    },
    /// Guest panic: record message `msg` (an index into the program's
    /// message table) through the runtime's panic path, mark the
    /// program's panic flag, and return from the current function
    /// immediately. Callers are expected to follow a call with
    /// [`Instr::CheckPanic`] so the panic propagates to the entry
    /// point instead of execution continuing with dummy values.
    /// `loc` indexes the program's panic-location table (the
    /// `file:line:col` strings the front end resolved from the
    /// panic site's source position), or is [`NO_LOCATION`] when
    /// the site has no known position; the recorded panic carries
    /// it so the default panic report can name the location, as
    /// std's does.
    /// Program compiles only, like [`Instr::Call`].
    Panic { msg: u32, loc: u32 },
    /// If the program's panic flag is set, return from the current
    /// function immediately (propagating a guest panic raised by a
    /// callee). Otherwise a no-op. Program compiles only.
    CheckPanic,
    /// Guest panic with the index-out-of-bounds message, whose text
    /// depends on the runtime length and index: "index out of
    /// bounds: the len is {len} but the index is {index}". The message is formatted
    /// by a helper in the program layer into a scratch buffer and
    /// then recorded through the usual panic path. Program compiles only.
    PanicIndexOob { len: Operand, index: Operand, loc: u32 },
    /// Guest panic with the negative-index message, whose text
    /// depends on the runtime index and on the place being indexed:
    /// "index out of bounds: negative index {index} used for
    /// {place}". `place` indexes the program's place-string table;
    /// the message is formatted by a helper in the program layer
    /// into a scratch buffer and then recorded through the usual
    /// panic path. Program compiles only.
    PanicNegativeIndex { index: Operand, place: u32, loc: u32 },
    /// Guest panic with the sub-slice range-end message: "range
    /// end index {end} out of range for slice of length {len}",
    /// formatted at run time like [`Instr::PanicIndexOob`].
    /// Program compiles only.
    PanicRangeEnd { end: Operand, len: Operand, loc: u32 },
    /// Guest panic with the sub-slice inverted-range message:
    /// "slice index starts at {from} but ends at {end}", formatted
    /// at run time like [`Instr::PanicIndexOob`]. Program compiles
    /// only.
    PanicSliceIndex { from: Operand, end: Operand, loc: u32 },
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
    /// `dst = src as <int>`: Rust's float-to-integer cast with a
    /// 128-bit target — saturating at the target type's bounds and
    /// mapping NaN to 0, as the 64-bit form does. The result's low
    /// word lands in `dst`, its high word in `dst2`; `signed` is
    /// the target type's signedness.
    FloatToInt128 {
        dst: Value,
        dst2: Value,
        kind: FloatKind,
        src: Operand,
        signed: bool,
    },
    /// `dst = src as <float>`: a 128-bit integer-to-float cast with
    /// round-to-nearest-even; the source is the word pair `(lo,
    /// hi)` of the type whose signedness is `signed`.
    Int128ToFloat {
        dst: Value,
        kind: FloatKind,
        lo: Operand,
        hi: Operand,
        signed: bool,
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
    /// `dst = &static[index]`: the machine address of the program's
    /// static blob `index` (a byte string or a constant value's word
    /// image, owned by the compiled program for its whole lifetime —
    /// see [`crate::program::StaticBlob`]). This is how the MIR front
    /// end materialises pointer-shaped constants (string constants,
    /// reference constants): the pointee lives in program storage,
    /// not in any frame, so the pointer stays valid after the
    /// current function returns, exactly like a native `&'static`.
    /// Program compiles only, like [`Instr::Call`].
    StaticAddr { dst: Value, index: u32 },
    /// `dst = *ptr`: load the full 64-bit word at the address held
    /// in `ptr`. A scalar leaf's slot always holds the value's
    /// canonical 64-bit form, so a whole-word load through a pointer
    /// to a slot yields exactly the leaf's value.
    LoadPtr { dst: Value, ptr: Operand },
    /// `*ptr = src`: store the full 64-bit word `src` at the address
    /// held in `ptr` (the counterpart of [`Instr::LoadPtr`]).
    StorePtr { ptr: Operand, src: Operand },
    /// Call host function `host` (an index into the program's host
    /// function table) with the given argument words: the backend
    /// passes the program's builtin context pointer as the first
    /// System V argument and the words in the registers after it,
    /// so a host call carries at most five words. The result lands
    /// in `dst`, and its second word — when the host function
    /// returns a pair — in `dst2` (RAX / RDX, the pair-return
    /// convention, as for [`Instr::Call`]). This is how the MIR
    /// front end lowers the builtin layer's calls (see
    /// [`crate::builtins`]): the semantics live in host functions
    /// because they are the host-side half of the std model (the
    /// fmt engine, the primitive float methods). Program compiles
    /// only, like [`Instr::Call`].
    HostCall {
        dst: Value,
        dst2: Option<Value>,
        host: u32,
        args: Vec<Operand>,
    },
}

/// How a basic block ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Terminator {
    /// Return a value from the function. A two-word result (a fat
    /// pointer, a two-scalar pair) travels in two registers:
    /// `value` in RAX and `second` in RDX — the System V
    /// pair-return convention, the mirror of how two-word
    /// arguments are passed. A result of at most one word has
    /// `second: None`.
    Return {
        value: Operand,
        second: Option<Operand>,
    },
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
