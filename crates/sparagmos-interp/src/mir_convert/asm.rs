//! Inline-asm argument seed planning.


use sparagmos_mir as mir;

use crate::ir;



/// The register-passing ABI family of a naked function's
/// declared ABI: the families the interpreter's register file
/// models. The `C` and `System` ABIs are SysV64-shaped on this
/// target; the `sysv64` / `win64` spellings arrive as
/// [`mir::Abi::Other`]. Anything else (the Rust ABI included —
/// its register assignment is the compiler's own and is not
/// modelled) returns `None`.
fn naked_asm_abi_family(abi: &mir::Abi) -> Option<mir::AsmAbi> {
    match abi {
        mir::Abi::C | mir::Abi::System => Some(mir::AsmAbi::SysV64),
        mir::Abi::Other(name) if name.starts_with("SysV64") => Some(mir::AsmAbi::SysV64),
        mir::Abi::Other(name) if name.starts_with("Win64") => Some(mir::AsmAbi::Win64),
        _ => None,
    }
}

/// The register scalar of one naked-function parameter type:
/// the same classification the driver applies to an asm call
/// signature's parameters. Anything else (aggregates, pointers,
/// 128-bit integers, 16/128-bit floats) is not register-passed
/// in this model and returns `None`.
fn asm_arg_scalar(ty: &mir::Ty) -> Option<mir::AsmScalar> {
    Some(match ty {
        mir::Ty::Int(it) => mir::AsmScalar::Int {
            width: match it {
                mir::IntTy::I8 => 8,
                mir::IntTy::I16 => 16,
                mir::IntTy::I32 => 32,
                mir::IntTy::I64 | mir::IntTy::Isize => 64,
                mir::IntTy::I128 => return None,
            },
            signed: true,
        },
        mir::Ty::Uint(it) => mir::AsmScalar::Int {
            width: match it {
                mir::UintTy::U8 => 8,
                mir::UintTy::U16 => 16,
                mir::UintTy::U32 => 32,
                mir::UintTy::U64 | mir::UintTy::Usize => 64,
                mir::UintTy::U128 => return None,
            },
            signed: false,
        },
        mir::Ty::Bool => mir::AsmScalar::Bool,
        mir::Ty::Char => mir::AsmScalar::Int {
            width: 32,
            signed: false,
        },
        mir::Ty::Float(ft) => mir::AsmScalar::Float {
            width: match ft {
                mir::FloatTy::F32 => 32,
                mir::FloatTy::F64 => 64,
                _ => return None,
            },
        },
        _ => return None,
    })
}

/// The incoming-argument seedings of a naked-function body:
/// one entry per parameter, assigning the architectural
/// register the function's ABI passes it in. Returns `None`
/// when any parameter cannot be seeded — an unmodelled ABI
/// family, an argument shape outside the register scalars, or
/// more arguments than the ABI passes in registers (the rest
/// would arrive on the stack, which the register file does
/// not model).
pub(super) fn asm_arg_seed_plan(body: &mir::Body, abi: Option<&mir::Abi>) -> Option<Vec<ir::AsmArgSeed>> {
    // Register-file indices in argument order (see
    // `exec_asm_call`'s tables): SysV64 integer arguments in
    // rdi, rsi, rdx, rcx, r8, r9 and floats in xmm0..; Win64
    // arguments by position in rcx, rdx, r8, r9 / xmm0...
    const SYSV_INT: [u8; 6] = [7, 6, 2, 1, 8, 9];
    const WIN_INT: [u8; 4] = [1, 2, 8, 9];
    let family = naked_asm_abi_family(abi?)?;
    let mut seeds = Vec::with_capacity(body.arg_count);
    let (mut next_int, mut next_float) = (0usize, 0usize);
    for (pos, decl) in body
        .local_decls
        .iter()
        .skip(1)
        .take(body.arg_count)
        .enumerate()
    {
        let scalar = asm_arg_scalar(&decl.ty)?;
        let arch = match (family, scalar) {
            (mir::AsmAbi::SysV64, mir::AsmScalar::Float { .. }) => {
                if next_float >= 8 {
                    return None;
                }
                let a = 16 + next_float as u8;
                next_float += 1;
                a
            }
            (mir::AsmAbi::SysV64, _) => {
                if next_int >= SYSV_INT.len() {
                    return None;
                }
                let a = SYSV_INT[next_int];
                next_int += 1;
                a
            }
            (mir::AsmAbi::Win64, mir::AsmScalar::Float { .. }) => {
                if pos >= WIN_INT.len() {
                    return None;
                }
                16 + pos as u8
            }
            (mir::AsmAbi::Win64, _) => {
                if pos >= WIN_INT.len() {
                    return None;
                }
                WIN_INT[pos]
            }
        };
        seeds.push(ir::AsmArgSeed {
            local: (pos + 1) as u32,
            arch,
            scalar,
        });
    }
    Some(seeds)
}

/// The return shape of a body whose terminator is a naked asm
/// block: the interpreter reads the return value out of the
/// register file's `rax`, so the return type must be a plain
/// integer, bool or char. Anything else returns `None`.
pub(super) fn asm_return_shape(body: &mir::Body) -> Option<ir::AsmRet> {
    let ty = &body.local_decls.first()?.ty;
    let (width, kind) = match ty {
        mir::Ty::Int(it) => (
            match it {
                mir::IntTy::I8 => 8,
                mir::IntTy::I16 => 16,
                mir::IntTy::I32 => 32,
                mir::IntTy::I64 | mir::IntTy::Isize => 64,
                mir::IntTy::I128 => return None,
            },
            ir::AsmRegKind::Signed,
        ),
        mir::Ty::Uint(it) => (
            match it {
                mir::UintTy::U8 => 8,
                mir::UintTy::U16 => 16,
                mir::UintTy::U32 => 32,
                mir::UintTy::U64 | mir::UintTy::Usize => 64,
                mir::UintTy::U128 => return None,
            },
            ir::AsmRegKind::Unsigned,
        ),
        mir::Ty::Bool => (8, ir::AsmRegKind::Bool),
        mir::Ty::Char => (32, ir::AsmRegKind::Unsigned),
        _ => return None,
    };
    Some(ir::AsmRet { width, kind })
}
