//! Conversion and run error types.

use std::fmt;


use crate::error::InterpError;



/// Why a `sparagmos-mir` program could not be converted into IR at all.
/// (Kinds that *can* be represented as IR `Unsupported` nodes do not
/// appear here; they fail at execution time instead.)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvertError {
    /// A local (or cast target / tuple element) has a type the IR's
    /// type model cannot represent (trait objects, closures,
    /// coroutines, `f16` / `f128`, ...).
    UnsupportedType { context: String, ty: String },
    /// A constant form the IR cannot represent (pointers, `f16` /
    /// `f128` float constants, unevaluated / type constants outside
    /// call position, ...).
    UnsupportedConstant { kind: String },
    /// A place projection the IR cannot represent.
    UnsupportedProjection { kind: String },
    /// Two functions in one program that cannot be told apart:
    /// they share a name, and no distinct `DefId` disambiguates
    /// them (see [`program_keys`]).
    DuplicateFunction { name: String },
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConvertError::UnsupportedType { context, ty } => {
                write!(f, "unsupported type in {context}: {ty}")
            }
            ConvertError::UnsupportedConstant { kind } => {
                write!(f, "unsupported constant: {kind}")
            }
            ConvertError::UnsupportedProjection { kind } => {
                write!(f, "unsupported projection: {kind}")
            }
            ConvertError::DuplicateFunction { name } => {
                write!(f, "duplicate function name `{name}` in MIR program")
            }
        }
    }
}

impl std::error::Error for ConvertError {}

/// Failure of a whole MIR run: either conversion or interpretation.
#[derive(Debug, Clone, PartialEq)]
pub enum MirRunError {
    Convert(ConvertError),
    Interp(InterpError),
}

impl fmt::Display for MirRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MirRunError::Convert(e) => write!(f, "MIR conversion failed: {e}"),
            MirRunError::Interp(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for MirRunError {}

impl From<ConvertError> for MirRunError {
    fn from(e: ConvertError) -> Self {
        MirRunError::Convert(e)
    }
}

impl From<InterpError> for MirRunError {
    fn from(e: InterpError) -> Self {
        MirRunError::Interp(e)
    }
}
