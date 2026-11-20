//! Naive (repr(C)-like, declaration-order) layout computation.
use crate::error::{RuntimeError, RuntimeResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Layout {
    pub size: usize,
    pub align: usize,
}
impl Layout {
    pub fn new(size: usize, align: usize) -> RuntimeResult<Self> {
        if align == 0 || !align.is_power_of_two() {
            return Err(RuntimeError::Layout(format!("invalid align {align}")));
        }
        Ok(Self { size, align })
    }
    pub fn from_size_align(size: usize, align: usize) -> RuntimeResult<Self> {
        Self::new(size, align)
    }
    /// Size rounded up to align.
    pub fn pad_to_align(&self) -> Self {
        Self {
            size: align_up(self.size, self.align),
            align: self.align,
        }
    }
    pub fn extend(&self, next: Layout) -> RuntimeResult<(Layout, usize)> {
        let offset = align_up(self.size, next.align);
        let size = offset
            .checked_add(next.size)
            .ok_or(RuntimeError::Layout("size overflow".into()))?;
        Ok((
            Layout {
                size,
                align: self.align.max(next.align),
            },
            offset,
        ))
    }
}
pub fn align_up(v: usize, align: usize) -> usize {
    if align <= 1 {
        v
    } else {
        (v + align - 1) & !(align - 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntTy {
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UintTy {
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatTy {
    F16,
    F32,
    F64,
    F128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    Bool,
    Char,
    Never,
    Unit,
    Int(IntTy),
    Uint(UintTy),
    Float(FloatTy),
    Ptr,
    Ref,            // thin pointers: size_of::<usize>
    Str,            // fat pointer (ptr+len)
    Slice(Box<Ty>), // fat pointer
    Tuple(Vec<Ty>),
    Array(Box<Ty>, usize),
    Struct(Vec<Ty>),    // fields in declaration order
    Enum(Vec<Vec<Ty>>), // naive: discriminant (u32) + largest variant payload
    FnPtr,
    RawVec(Box<Ty>), // ptr+cap+len placeholder (3 words) — documented naive
}

pub fn layout_of(ty: &Ty) -> RuntimeResult<Layout> {
    let ptr = std::mem::size_of::<usize>();
    let l = match ty {
        Ty::Bool => Layout { size: 1, align: 1 },
        Ty::Char => Layout { size: 4, align: 4 },
        Ty::Never | Ty::Unit => Layout { size: 0, align: 1 },
        Ty::Int(t) => {
            let s = match t {
                IntTy::I8 => 1,
                IntTy::I16 => 2,
                IntTy::I32 => 4,
                IntTy::I64 => 8,
                IntTy::I128 => 16,
                IntTy::Isize => ptr,
            };
            Layout { size: s, align: s }
        }
        Ty::Uint(t) => {
            let s = match t {
                UintTy::U8 => 1,
                UintTy::U16 => 2,
                UintTy::U32 => 4,
                UintTy::U64 => 8,
                UintTy::U128 => 16,
                UintTy::Usize => ptr,
            };
            Layout { size: s, align: s }
        }
        Ty::Float(t) => {
            let s = match t {
                FloatTy::F16 => 2,
                FloatTy::F32 => 4,
                FloatTy::F64 => 8,
                FloatTy::F128 => 16,
            };
            Layout { size: s, align: s }
        }
        Ty::Ptr | Ty::Ref | Ty::FnPtr => Layout {
            size: ptr,
            align: ptr,
        },
        Ty::Str | Ty::Slice(_) => Layout {
            size: 2 * ptr,
            align: ptr,
        },
        Ty::RawVec(_) => Layout {
            size: 3 * ptr,
            align: ptr,
        },
        Ty::Tuple(fields) | Ty::Struct(fields) => {
            let mut cur = Layout { size: 0, align: 1 };
            for f in fields {
                let (next, _off) = cur.extend(layout_of(f)?)?;
                cur = next;
            }
            cur.pad_to_align()
        }
        Ty::Array(elem, n) => {
            let el = layout_of(elem)?;
            let stride = align_up(el.size, el.align);
            Layout {
                size: stride
                    .checked_mul(*n)
                    .ok_or(RuntimeError::Layout("array size overflow".into()))?,
                align: el.align,
            }
        }
        Ty::Enum(variants) => {
            let disc = Layout { size: 4, align: 4 };
            let mut payload = Layout { size: 0, align: 1 };
            for v in variants {
                let p = layout_of(&Ty::Struct(v.clone()))?;
                if p.size > payload.size || p.align > payload.align {
                    payload = Layout {
                        size: p.size.max(payload.size),
                        align: p.align.max(payload.align),
                    };
                }
            }
            let (combined, _) = disc.extend(payload)?;
            combined.pad_to_align()
        }
    };
    Ok(l)
}
pub fn size_of_ty(ty: &Ty) -> RuntimeResult<usize> {
    Ok(layout_of(ty)?.size)
}
pub fn align_of_ty(ty: &Ty) -> RuntimeResult<usize> {
    Ok(layout_of(ty)?.align)
}
/// Field offsets for Tuple/Struct.
pub fn field_offsets(ty: &Ty) -> RuntimeResult<Vec<usize>> {
    let fields = match ty {
        Ty::Tuple(f) | Ty::Struct(f) => f,
        _ => return Err(RuntimeError::Layout("not an aggregate".into())),
    };
    let mut cur = Layout { size: 0, align: 1 };
    let mut out = Vec::new();
    for f in fields {
        let (next, off) = cur.extend(layout_of(f)?)?;
        out.push(off);
        cur = next;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn primitives() {
        assert_eq!(size_of_ty(&Ty::Int(IntTy::I32)).unwrap(), 4);
        assert_eq!(align_of_ty(&Ty::Float(FloatTy::F64)).unwrap(), 8);
        assert_eq!(size_of_ty(&Ty::Bool).unwrap(), 1);
        assert_eq!(size_of_ty(&Ty::Char).unwrap(), 4);
    }
    #[test]
    fn struct_padding() {
        // struct { u8, u32, u8 } => 1+pad3+4+1+pad3=12 align4
        let t = Ty::Struct(vec![
            Ty::Uint(UintTy::U8),
            Ty::Uint(UintTy::U32),
            Ty::Uint(UintTy::U8),
        ]);
        assert_eq!(layout_of(&t).unwrap(), Layout { size: 12, align: 4 });
        assert_eq!(field_offsets(&t).unwrap(), vec![0, 4, 8]);
    }
    #[test]
    fn tuple_and_array() {
        assert_eq!(
            size_of_ty(&Ty::Tuple(vec![Ty::Int(IntTy::I64), Ty::Int(IntTy::I16)])).unwrap(),
            16
        );
        assert_eq!(
            size_of_ty(&Ty::Array(Box::new(Ty::Uint(UintTy::U16)), 5)).unwrap(),
            10
        );
        assert_eq!(
            size_of_ty(&Ty::Array(
                Box::new(Ty::Struct(vec![
                    Ty::Uint(UintTy::U8),
                    Ty::Uint(UintTy::U32)
                ])),
                3
            ))
            .unwrap(),
            24
        );
    }
    #[test]
    fn fat_pointers() {
        assert_eq!(size_of_ty(&Ty::Str).unwrap(), 16);
        assert_eq!(
            size_of_ty(&Ty::Slice(Box::new(Ty::Uint(UintTy::U8)))).unwrap(),
            16
        );
    }
    #[test]
    fn enum_layout() {
        let t = Ty::Enum(vec![vec![], vec![Ty::Uint(UintTy::U64)]]);
        let l = layout_of(&t).unwrap();
        assert_eq!(l.align, 8);
        assert_eq!(l.size, 16);
    }
}
