//! AST for textual MIR, owned by `sparagmos-parse`.
//!
//! The shape deliberately mirrors the assumed `sparagmos-mir` API (see
//! [`crate::convert`]) so the conversion layer stays a thin, mechanical
//! field-by-field mapping: `Body`, `LocalDecl`, `BasicBlock`, `Statement`,
//! `Terminator`, `Rvalue`, `Operand`, `Place`, `Ty`, `Const`.
//!
//! Types are kept as a structured-but-permissive [`Ty`]: primitive / tuple /
//! reference / array forms are structured, anything else is preserved
//! verbatim as [`Ty::Named`] instead of being dropped.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct MirFile {
    pub bodies: Vec<Body>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Body {
    pub name: String,
    /// Argument locals in declaration order, e.g. `_1: i32`.
    pub args: Vec<LocalDecl>,
    pub return_ty: Ty,
    /// All locals declared with `let`, in source order (includes `_0`).
    pub locals: Vec<LocalDecl>,
    /// `debug name => _N;` bindings (also collected from inside `scope`s).
    pub debug_vars: Vec<DebugVar>,
    pub blocks: Vec<BasicBlock>,
    /// 1-based line of the `fn` keyword.
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebugVar {
    pub name: String,
    pub local: Local,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Local(pub u32);

impl fmt::Display for Local {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "_{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalDecl {
    pub local: Local,
    pub mutable: bool,
    pub ty: Ty,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    Bool,
    Char,
    Int(String),   // i8..i128, isize
    Uint(String),  // u8..u128, usize
    Float(String), // f16/f32/f64/f128
    Str,
    Unit,
    Tuple(Vec<Ty>),
    Ref {
        mutable: bool,
        inner: Box<Ty>,
    },
    RawPtr {
        mutable: bool,
        inner: Box<Ty>,
    },
    Array {
        elem: Box<Ty>,
        len: String,
    },
    Slice(Box<Ty>),
    /// Any other type, preserved verbatim (paths, generics, fn pointers…).
    Named(String),
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Bool => write!(f, "bool"),
            Ty::Char => write!(f, "char"),
            Ty::Int(s) | Ty::Uint(s) | Ty::Float(s) => write!(f, "{s}"),
            Ty::Str => write!(f, "str"),
            Ty::Unit => write!(f, "()"),
            Ty::Tuple(ts) => {
                write!(f, "(")?;
                for (i, t) in ts.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{t}")?;
                }
                write!(f, ")")
            }
            Ty::Ref { mutable, inner } => {
                write!(f, "&{}{}", if *mutable { "mut " } else { "" }, inner)
            }
            Ty::RawPtr { mutable, inner } => {
                write!(f, "*{} {}", if *mutable { "mut" } else { "const" }, inner)
            }
            Ty::Array { elem, len } => write!(f, "[{elem}; {len}]"),
            Ty::Slice(e) => write!(f, "[{e}]"),
            Ty::Named(s) => write!(f, "{s}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Projection {
    Deref,
    Field { index: u32, ty: Option<Ty> },
    Index(Local),
    ConstantIndex { offset: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Place {
    pub local: Local,
    pub projections: Vec<Projection>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Const {
    Bool(bool),
    Int {
        value: i128,
        ty: String,
    },
    Uint {
        value: u128,
        ty: String,
    },
    Float {
        text: String,
        ty: String,
    },
    Char(char),
    Str(String),
    Unit,
    /// Named / associated constant, e.g. `i32::MIN`, `usize::MAX`.
    Named(String),
    /// Function item used as a value, e.g. `answer` in a call position is
    /// handled by the call terminator; this covers `const { ... }` fallbacks
    /// preserved verbatim.
    Raw(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Copy(Place),
    Move(Place),
    Constant(Const),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitXor,
    BitAnd,
    BitOr,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Offset,
    AddWithOverflow,
    SubWithOverflow,
    MulWithOverflow,
    AddUnchecked,
    SubUnchecked,
    MulUnchecked,
    ShlUnchecked,
    ShrUnchecked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnOp {
    Not,
    Neg,
    PtrMetadata,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Rvalue {
    Use(Operand),
    BinaryOp(BinOp, Box<Operand>, Box<Operand>),
    UnaryOp(UnOp, Box<Operand>),
    /// `&place` / `&mut place` / `&raw const place` etc. (kind kept verbatim)
    Ref {
        kind: String,
        place: Place,
    },
    /// Any other `Name(args...)` rvalue (aggregate-like / intrinsic-like),
    /// preserved with its callee name and operands — never silently dropped,
    /// but not yet structured. See lib docs for the honest scope.
    AggregateCall {
        name: String,
        args: Vec<Operand>,
    },
    /// Verbatim fallback for a parenthesised / misc rvalue we could lex but
    /// not structure; carries source text so callers can inspect it.
    Raw(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Assign(Place, Rvalue),
    StorageLive(Local),
    StorageDead(Local),
    Nop,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnwindAction {
    Continue,
    Cleanup(BlockId),
    Terminate,
    Unreachable,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Terminator {
    Return,
    Unreachable,
    Goto {
        target: BlockId,
    },
    SwitchInt {
        discr: Operand,
        /// (value, target) pairs; value kept as u128 bit pattern.
        targets: Vec<(u128, BlockId)>,
        otherwise: Option<BlockId>,
    },
    Call {
        func: Operand,
        args: Vec<Operand>,
        destination: Option<Place>,
        target: Option<BlockId>,
        unwind: Option<UnwindAction>,
    },
    Assert {
        /// Condition as an rvalue so `assert(!move _2, ...)` keeps its `Not`.
        cond: Box<Rvalue>,
        expected: bool,
        msg: String,
        target: Option<BlockId>,
        unwind: Option<UnwindAction>,
    },
    Drop {
        place: Place,
        target: Option<BlockId>,
        unwind: Option<UnwindAction>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct BasicBlock {
    pub id: BlockId,
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
    pub line: usize,
}
