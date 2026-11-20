//! Thin conversion layer from this crate's AST ([`crate::ast`]) to the
//! assumed `sparagmos-mir` model.
//!
//! Lane A (`sparagmos-mir`) was still a placeholder when this lane started,
//! so the parser targets its own AST first. The AST was designed to be
//! structurally identical to the assumed `sparagmos-mir` API below, which
//! makes this conversion a mechanical field-by-field mapping. When Lane A's
//! real types land, only this file should need to change: delete the local
//! [`mir_model`] mirror and re-point the `From` impls at
//! `sparagmos_mir::*` (the names and field shapes are the contract).
//!
//! # Assumed `sparagmos-mir` API (the contract this lane codes against)
//!
//! ```text
//! pub struct Body { pub name: String, pub args: Vec<LocalDecl>,
//!                   pub return_ty: Ty, pub locals: Vec<LocalDecl>,
//!                   pub debug_vars: Vec<DebugVar>, pub blocks: Vec<BasicBlock> }
//! pub struct LocalDecl { pub local: Local, pub mutable: bool, pub ty: Ty }
//! pub struct Local(pub u32);  pub struct BlockId(pub u32);
//! pub enum Ty { Bool, Char, Int(String), Uint(String), Float(String), Str,
//!               Unit, Tuple(Vec<Ty>), Ref{mutable,inner}, RawPtr{mutable,inner},
//!               Array{elem,len}, Slice(..), Named(String) }
//! pub struct Place { pub local: Local, pub projections: Vec<Projection> }
//! pub enum Projection { Deref, Field{index,ty}, Index(Local), ConstantIndex{offset} }
//! pub enum Operand { Copy(Place), Move(Place), Constant(Const) }
//! pub enum Const { Bool(bool), Int{value:i128,ty}, Uint{value:u128,ty},
//!                  Float{text,ty}, Char(char), Str(String), Unit,
//!                  Named(String), Raw(String) }
//! pub enum Rvalue { Use(Operand), BinaryOp(BinOp,Box,Box), UnaryOp(UnOp,Box),
//!                   Ref{kind,place}, AggregateCall{name,args}, Raw(String) }
//! pub enum Statement { Assign(Place,Rvalue), StorageLive(Local),
//!                      StorageDead(Local), Nop }
//! pub enum Terminator { Return, Unreachable, Goto{target},
//!                       SwitchInt{discr,targets,otherwise},
//!                       Call{func,args,destination,target,unwind},
//!                       Assert{cond,expected,msg,target,unwind},
//!                       Drop{place,target,unwind} }
//! pub struct BasicBlock { pub id: BlockId, pub statements: Vec<Statement>,
//!                         pub terminator: Terminator }
//! ```

/// Mirror of the assumed `sparagmos-mir` types, kept inside this crate so
/// the conversion can be compiled and tested before Lane A lands.
pub mod mir_model {
    pub use crate::ast::{
        BasicBlock, BlockId, Body, Const, DebugVar, Local, LocalDecl, MirFile, Operand, Place,
        Projection, Rvalue, Statement, Terminator, Ty,
    };
}

/// Convert a parsed file into the (mirrored) `sparagmos-mir` model.
/// Today this is the identity mapping because the shapes are identical by
/// design; it exists so callers code against the conversion, not the AST,
/// and so the seam is in exactly one place when the real crate lands.
pub fn to_mir_model(file: crate::ast::MirFile) -> mir_model::MirFile {
    file
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_mir;

    #[test]
    fn conversion_round_trips_shape() {
        let src = "fn f() -> i32 { let mut _0: i32; bb0: { _0 = const 7_i32; return; } }";
        let file = parse_mir(src).unwrap();
        let model = to_mir_model(file.clone());
        assert_eq!(model, file);
    }
}
