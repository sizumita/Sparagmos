//! Minimal MIR-like IR executed by the interpreter.
//!
//! This is Lane C's own model, shaped after rustc MIR as of Rust 1.100.0
//! (`rustc_middle::mir`): a [`Body`] is a list of basic blocks, each block
//! is statements plus exactly one terminator, statements are assignments,
//! and expressions are operands / binary / unary operations.
//!
//! Deliberate simplifications for the first cut (see the crate-level docs
//! for the assumed Lane A API and the expected differences):
//! * Functions are identified by name (`String`), not by `DefId`.
//! * A call's callee is a plain name, not a constant `FnDef` operand.
//! * Places only support tuple-field projection; deref / index /
//!   downcast are representable but rejected at run time.
//! * Types are reduced to the scalar [`Ty`] below (no ADTs, no references,
//!   no lifetimes, no generics).

use std::collections::HashMap;
use std::fmt;

/// Index of a local variable. Local 0 is always the return place,
/// locals `1..=arg_count` are the arguments (as in rustc MIR).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Local(pub usize);

impl Local {
    /// The return place, `_0`.
    pub const RETURN: Local = Local(0);
}

impl fmt::Display for Local {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_{}", self.0)
    }
}

/// Index of a basic block inside a [`Body`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BasicBlock(pub usize);

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// Signed integer types (bit width matters for overflow semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntTy {
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
}

impl IntTy {
    /// Width in bits. `isize` is 64 on the x86_64 target Sparagmos aims at.
    pub fn bits(self) -> u32 {
        match self {
            IntTy::I8 => 8,
            IntTy::I16 => 16,
            IntTy::I32 => 32,
            IntTy::I64 => 64,
            IntTy::I128 => 128,
            IntTy::Isize => 64,
        }
    }
    /// Inclusive `(min, max)` representable in this type.
    pub fn range(self) -> (i128, i128) {
        let bits = self.bits();
        if bits >= 128 {
            (i128::MIN, i128::MAX)
        } else {
            (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
        }
    }
}

/// Unsigned integer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UintTy {
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
}

impl UintTy {
    /// Width in bits. `usize` is 64 on x86_64.
    pub fn bits(self) -> u32 {
        match self {
            UintTy::U8 => 8,
            UintTy::U16 => 16,
            UintTy::U32 => 32,
            UintTy::U64 => 64,
            UintTy::U128 => 128,
            UintTy::Usize => 64,
        }
    }
    /// Largest representable value.
    pub fn max(self) -> u128 {
        let bits = self.bits();
        if bits >= 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        }
    }
}

/// The scalar type model used for width / overflow semantics.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Ty {
    Unit,
    Bool,
    Int(IntTy),
    Uint(UintTy),
    /// Tuple type, e.g. the `(T, bool)` produced by a checked operation.
    Tuple(Vec<Ty>),
    Never,
}

impl Ty {
    pub fn i8() -> Ty {
        Ty::Int(IntTy::I8)
    }
    pub fn i32() -> Ty {
        Ty::Int(IntTy::I32)
    }
    pub fn i64() -> Ty {
        Ty::Int(IntTy::I64)
    }
    pub fn i128() -> Ty {
        Ty::Int(IntTy::I128)
    }
    pub fn u32() -> Ty {
        Ty::Uint(UintTy::U32)
    }
    pub fn u64() -> Ty {
        Ty::Uint(UintTy::U64)
    }
    pub fn usize_() -> Ty {
        Ty::Uint(UintTy::Usize)
    }
}

/// One step of a place projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProjectionElem {
    /// Tuple / aggregate field access, e.g. `_1.0`.
    Field(usize),
    /// `*_1` — representable, not yet executable (explicit run-time error).
    Deref,
    /// `_1[_2]` — representable, not yet executable.
    Index(Local),
}

/// A place: a local plus a (possibly empty) projection chain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Place {
    pub local: Local,
    pub projection: Vec<ProjectionElem>,
}

impl Place {
    pub fn local(local: Local) -> Place {
        Place {
            local,
            projection: Vec::new(),
        }
    }
    pub fn from_index(index: usize) -> Place {
        Place::local(Local(index))
    }
    /// Append a tuple-field projection, e.g. `place.field(0)`.
    pub fn field(mut self, field: usize) -> Place {
        self.projection.push(ProjectionElem::Field(field));
        self
    }
    pub fn is_direct_local(&self) -> bool {
        self.projection.is_empty()
    }
}

impl fmt::Display for Place {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.local)?;
        for elem in &self.projection {
            match elem {
                ProjectionElem::Field(i) => write!(f, ".{i}")?,
                ProjectionElem::Deref => write!(f, ".*")?,
                ProjectionElem::Index(l) => write!(f, "[{l}]")?,
            }
        }
        Ok(())
    }
}

/// A compile-time constant.
#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Unit,
    Bool(bool),
    Int(i128),
    Uint(u128),
}

/// An operand: the right-hand side leaf of most rvalues.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Copy(Place),
    Move(Place),
    Constant(Constant),
}

impl Operand {
    pub fn copy_place(place: Place) -> Operand {
        Operand::Copy(place)
    }
    pub fn copy_local(index: usize) -> Operand {
        Operand::Copy(Place::from_index(index))
    }
    pub fn move_local(index: usize) -> Operand {
        Operand::Move(Place::from_index(index))
    }
    pub fn const_int(v: i128) -> Operand {
        Operand::Constant(Constant::Int(v))
    }
    pub fn const_uint(v: u128) -> Operand {
        Operand::Constant(Constant::Uint(v))
    }
    pub fn const_bool(v: bool) -> Operand {
        Operand::Constant(Constant::Bool(v))
    }
    pub fn const_unit() -> Operand {
        Operand::Constant(Constant::Unit)
    }
}

/// Binary operators (subset of rustc MIR `BinOp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Unary operators (rustc MIR `UnOp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnOp {
    Neg,
    Not,
}

/// A right-hand side expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Rvalue {
    Use(Operand),
    BinaryOp(BinOp, Box<(Operand, Operand)>),
    /// Like rustc MIR: yields a `(value, overflowed)` tuple, wrapping.
    CheckedBinaryOp(BinOp, Box<(Operand, Operand)>),
    UnaryOp(UnOp, Operand),
    /// Build a tuple value from operands.
    AggregateTuple(Vec<Operand>),
    /// An `as` cast to the given scalar type (wrapping, as in Rust).
    Cast(Operand, Ty),
    /// `&place` — representable, not yet executable.
    Ref(Place),
    /// Any rvalue kind not modelled yet; executing it is an explicit error.
    Unsupported(String),
}

impl Rvalue {
    pub fn binary(op: BinOp, lhs: Operand, rhs: Operand) -> Rvalue {
        Rvalue::BinaryOp(op, Box::new((lhs, rhs)))
    }
    pub fn checked_binary(op: BinOp, lhs: Operand, rhs: Operand) -> Rvalue {
        Rvalue::CheckedBinaryOp(op, Box::new((lhs, rhs)))
    }
}

/// Statement kinds (subset of rustc MIR `StatementKind`).
#[derive(Debug, Clone, PartialEq)]
pub enum StatementKind {
    Assign(Place, Rvalue),
    /// Hint only in rustc MIR; a no-op here.
    StorageLive(Local),
    /// Hint only in rustc MIR; a no-op here.
    StorageDead(Local),
    Nop,
    /// Any statement kind not modelled yet; executing it is an explicit error.
    Unsupported(String),
}

/// A single statement.
#[derive(Debug, Clone, PartialEq)]
pub struct Statement {
    pub kind: StatementKind,
}

impl Statement {
    pub fn assign(place: Place, rvalue: Rvalue) -> Statement {
        Statement {
            kind: StatementKind::Assign(place, rvalue),
        }
    }
    pub fn nop() -> Statement {
        Statement {
            kind: StatementKind::Nop,
        }
    }
}

/// Terminator kinds (subset of rustc MIR `TerminatorKind`).
#[derive(Debug, Clone, PartialEq)]
pub enum TerminatorKind {
    Goto {
        target: BasicBlock,
    },
    SwitchInt {
        discr: Operand,
        /// `(bit pattern of the discriminant, target block)` pairs.
        targets: Vec<(u128, BasicBlock)>,
        otherwise: BasicBlock,
    },
    Call {
        /// Callee name (simplification: rustc uses a constant FnDef operand).
        func: String,
        args: Vec<Operand>,
        destination: Place,
        /// Continuation block in the caller; `None` models a diverging call.
        target: Option<BasicBlock>,
    },
    Return,
    Unreachable,
    Assert {
        cond: Operand,
        expected: bool,
        target: BasicBlock,
        message: String,
    },
    /// Any terminator kind not modelled yet; executing it is an explicit error.
    Unsupported(String),
}

/// A basic block: statements followed by exactly one terminator.
#[derive(Debug, Clone, PartialEq)]
pub struct BasicBlockData {
    pub statements: Vec<Statement>,
    pub terminator: TerminatorKind,
}

impl BasicBlockData {
    pub fn new(statements: Vec<Statement>, terminator: TerminatorKind) -> Self {
        BasicBlockData {
            statements,
            terminator,
        }
    }
}

/// A function body: argument count, local types, and basic blocks.
/// `local_tys[0]` is the return type; `local_tys.len()` is the frame size.
#[derive(Debug, Clone, PartialEq)]
pub struct Body {
    pub name: String,
    pub arg_count: usize,
    pub local_tys: Vec<Ty>,
    pub blocks: Vec<BasicBlockData>,
}

impl Body {
    pub fn new(
        name: impl Into<String>,
        arg_count: usize,
        local_tys: Vec<Ty>,
        blocks: Vec<BasicBlockData>,
    ) -> Body {
        Body {
            name: name.into(),
            arg_count,
            local_tys,
            blocks,
        }
    }
    pub fn local_count(&self) -> usize {
        self.local_tys.len()
    }
    pub fn local_ty(&self, local: Local) -> Option<&Ty> {
        self.local_tys.get(local.0)
    }
}

/// A whole program: a set of named function bodies in one module.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Program {
    pub functions: HashMap<String, Body>,
}

impl Program {
    pub fn new() -> Program {
        Program::default()
    }
    pub fn add_function(&mut self, body: Body) -> &mut Self {
        self.functions.insert(body.name.clone(), body);
        self
    }
    pub fn get(&self, name: &str) -> Option<&Body> {
        self.functions.get(name)
    }
}
