//! A small function IR used as the JIT's input for now.
//!
//! Lane A's MIR types (`sparagmos-mir`) are not final yet, so this crate
//! defines its own minimal SSA-flavoured IR: a function is a list of basic
//! blocks, each block is a list of instructions plus a terminator. All
//! values are signed 64-bit integers (`i64`), which keeps the first compiler
//! honest and simple. When the MIR types are fixed, a MIR -> this IR
//! lowering (or a direct MIR -> emitter backend) will replace this as the
//! front door; the emitter and executable-memory layers stay unchanged.

/// A virtual value (SSA-like): defined once by an instruction or by being a
/// function parameter (parameters are values `0..params`).
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    /// Signed division (truncates toward zero, like Rust `/` on i64).
    Div,
    /// Signed remainder (like Rust `%` on i64).
    Rem,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

/// One instruction. Every instruction that produces a value names its
/// destination value explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
