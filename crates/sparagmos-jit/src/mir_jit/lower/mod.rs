use std::collections::HashMap;

use sparagmos_mir as mir;

use crate::ir::{
    BasicBlock as IrBlock,
    Function as IrFunction, Instr,
    Terminator as IrTerminator,
};
use crate::builtins::{BuiltinTables};

use super::adt::{is_array_into_iter_fn, is_array_iter_next_fn, AdtLayout, AdtLayouts, ArrayIterShape};
use super::tables::{MsgTable, StaticsTable};
use super::ty_helpers::*;
use super::types::{JitError, MirFunctionInput};

mod binops;
mod calls;
mod operands;
mod places;
mod rvalues;

// ---------------------------------------------------------------------------
// Per-function lowering
// ---------------------------------------------------------------------------

/// Where the length word of a fat pointer lives: what a slice
/// place's `Len` reads and its index bounds checks compare against.
#[derive(Clone, Debug)]
enum LenSrc {
    /// A length known from the type (an array's declared length).
    Const(u64),
    /// The length word is this slot (a fat local's second word).
    Slot(u32),
    /// The length word is in memory, at this address (the second
    /// word of a fat pair stored in memory).
    Mem(Box<AddrExpr>),
    /// The length of a sub-slice taken from the end of a base of
    /// this length: the base's length minus `skip` (the range's
    /// `to` counted from the end, plus its `from`), computed when
    /// the length is read. The place's sub-slice check runs before
    /// any use of the place, so the base's length is known to be
    /// at least `skip` by then.
    Tail { base: Box<LenSrc>, skip: u64 },
}

/// How to compute the runtime address of a place in memory: a base
/// pointer source, a constant leaf offset from it, and any number
/// of dynamic index terms (each the index's value slot and the
/// leaves per element of that index step). Nested indices compose
/// additively: an element of a nested array lies at the sum of the
/// steps' displacements, because each array's elements are laid
/// out contiguously at the leaves-per-element stride. Every leaf
/// is one 8-byte slot, in memory exactly as in
/// a frame: a pointer in this model is always the address of a
/// place's first leaf, and leaf `k` lies `8 * k` bytes *below* it,
/// because the backend lays value slots out at descending addresses
/// (`slot_disp` in the compiler). Offsets below are in leaves and
/// are subtracted when an address is materialised.
#[derive(Clone, Debug)]
struct AddrExpr {
    source: PtrSrc,
    const_off: usize,
    dyn_terms: Vec<(IndexSrc, usize)>,
}

/// An index value for a dynamic index term or a bounds check:
/// a value slot, or a constant (a literal index into a slice,
/// whose length — and so whose bound — is a runtime value).
#[derive(Clone, Debug)]
enum IndexSrc {
    Slot(u32),
    Const(u64),
}

#[derive(Clone, Debug)]
enum PtrSrc {
    /// The base pointer is the value held in this slot.
    Slot(u32),
    /// The base pointer is the address of this slot itself.
    SlotAddr(u32),
    /// The base pointer is the word stored at this address (a
    /// pointer stored in memory: the second dereference).
    Loaded(Box<AddrExpr>),
}

/// A check a memory place needs before it is touched, in
/// projection order: the interpreter performs each one as it
/// resolves the corresponding projection, and a failure is the
/// interpreter's panic for that projection.
#[derive(Clone, Debug)]
enum PlaceCheck {
    Index(IndexCheck),
    Subslice(SubsliceCheck),
}

/// A bounds check an indexed place needs before it is touched:
/// the index value must be below the length, or the access is the
/// interpreter's index panic. `neg` is set for an index of a
/// signed integer type: the interpreter first panics on a negative
/// value with its negative-index message, naming the place (the
/// string is interned in the place table when the check is
/// emitted, classification itself being pure).
#[derive(Clone, Debug)]
struct IndexCheck {
    index: IndexSrc,
    len: LenSrc,
    neg: Option<String>,
}

/// The range check a sub-slice projection needs: the selected
/// range must fit the base's length (the source in `len`), or the
/// access is the interpreter's range panic for the violated end.
#[derive(Clone, Debug)]
struct SubsliceCheck {
    from: u64,
    to: u64,
    from_end: bool,
    len: LenSrc,
}

/// A place resolved against the slot model.
#[derive(Clone, Debug)]
enum PlaceLoc {
    /// Leaves of a local in the current frame.
    Slots { first: u32, count: usize, ty: mir::Ty },
    /// Leaves in memory, reached through at least one dereference
    /// or a dynamic index. `count` is `None` for a slice / `str`
    /// place itself, which has no word representation of its own;
    /// `len` is the length source when the type is a slice / `str`;
    /// `checks` are the checks to emit before touching it.
    Mem {
        addr: AddrExpr,
        count: Option<usize>,
        ty: mir::Ty,
        len: Option<LenSrc>,
        checks: Vec<PlaceCheck>,
    },
}

/// Whether a constant's value is pointer-shaped: materializing it
/// yields the address of a static blob holding its contents, plus
/// metadata words for the fat shapes. A reference constant delegates
/// to its pointee's shape when the pointee is itself pointer-shaped
/// (a `&str` constant *is* the fat pointer); otherwise it is a thin
/// pointer to a value-shaped pointee.
fn is_pointer_shaped(v: &mir::ConstValue) -> bool {
    matches!(
        v,
        mir::ConstValue::Str(_)
            | mir::ConstValue::Bytes(_)
            | mir::ConstValue::Slice { .. }
            | mir::ConstValue::SliceVal(_)
            | mir::ConstValue::Ref(_)
    )
}

pub(super) struct FnLower<'a> {
    func: &'a MirFunctionInput,
    body: &'a mir::Body,
    def_index: &'a HashMap<mir::DefId, usize>,
    def_names: &'a HashMap<mir::DefId, String>,
    param_tys: &'a [Vec<mir::Ty>],
    discriminants: &'a mir::AdtDiscriminants,
    layouts: &'a AdtLayouts,
    messages: &'a mut MsgTable,
    /// Place strings for the negative-index panic, interned like
    /// `messages`.
    places: &'a mut MsgTable,
    /// The program's static blobs, interned like `messages`.
    statics: &'a mut StaticsTable,
    /// The builtin layer's compile-time tables (fmt descriptors,
    /// literal strings), interned like `messages` while lowering
    /// builtin calls.
    tables: &'a mut BuiltinTables,
    /// The slot holding the hidden sret argument (the address of
    /// the caller's destination storage), for a function whose
    /// return value is wider than two words. `None` otherwise.
    sret_slot: Option<u32>,
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
    pub(super) fn new(
        func: &'a MirFunctionInput,
        def_index: &'a HashMap<mir::DefId, usize>,
        def_names: &'a HashMap<mir::DefId, String>,
        param_tys: &'a [Vec<mir::Ty>],
        discriminants: &'a mir::AdtDiscriminants,
        layouts: &'a AdtLayouts,
        messages: &'a mut MsgTable,
        places: &'a mut MsgTable,
        statics: &'a mut StaticsTable,
        tables: &'a mut BuiltinTables,
    ) -> Self {
        Self {
            func,
            body: &func.body,
            def_index,
            def_names,
            param_tys,
            discriminants,
            layouts,
            messages,
            places,
            statics,
            tables,
            sret_slot: None,
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

    /// The computed slot layout of the ADT type `ty`.
    fn adt_layout(&self, ty: &mir::Ty) -> Result<AdtLayout, JitError> {
        let mir::Ty::Adt(def, args) = ty else {
            return Err(self.err("layout of a non-ADT type"));
        };
        self.layouts
            .layout(*def, args)
            .map_err(|reason| self.err(&reason))
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

    pub(super) fn run(mut self) -> Result<IrFunction, JitError> {
        let body = self.body;
        // Lay out the locals. Argument locals are laid out first:
        // each one spans its leaves in the IR parameter sequence,
        // and IR parameters are by definition the values
        // 0..param_count (a unit argument gets a parameter slot
        // whose value is never read). The return place and the
        // remaining locals follow.
        let arg_count = body.arg_count;
        self.local_slots = vec![(0, 0); body.local_decls.len()];
        // A function whose return place flattens to more than two
        // words cannot return them in registers: it takes a hidden
        // leading argument word — the address of the caller's
        // destination storage — and stores the return words there
        // at `Return` (the System V sret convention, in the slot
        // model's terms). `lower_call` derives the same layout from
        // the destination's leaf count, so the two sides agree.
        let ret_count = flatten(&body.local_decls[0].ty, &self.func.name, self.layouts)?;
        let sret = ret_count > 2;
        // The IR parameter layout: one word per leaf of each
        // argument (the hidden sret word first, when present), laid
        // out by `arg_word_layout` (a unit argument still occupies
        // a dummy word; a fat argument occupies two; a pair that
        // would straddle the register/stack boundary is preceded by
        // a pad word, which gets a slot nobody reads).
        let mut counts = Vec::with_capacity(arg_count + 1);
        if sret {
            counts.push(1);
        }
        for idx in 1..=arg_count {
            let decl = &body.local_decls[idx];
            counts.push(flatten(&decl.ty, &self.func.name, self.layouts)?);
        }
        let (starts, ir_param_count) = arg_word_layout(&counts);
        let offset = usize::from(sret);
        for (i, &count) in counts.iter().enumerate() {
            let start = starts[i];
            while (self.next_value as usize) < start {
                self.fresh(); // the pad word's slot
            }
            let slot = self.fresh();
            for _ in 1..count.max(1) {
                self.fresh();
            }
            if sret && i == 0 {
                self.sret_slot = Some(slot);
            } else {
                self.local_slots[i + 1 - offset] = (slot, count);
            }
        }
        for idx in std::iter::once(0).chain(arg_count + 1..body.local_decls.len()) {
            let decl = &body.local_decls[idx];
            let count = flatten(&decl.ty, &self.func.name, self.layouts)?;
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
            param_count: ir_param_count,
            value_count: self.next_value as usize,
            blocks,
        })
    }

    // ----- places, operands, constants -------------------------------------

    /// One in-frame projection step (`Field` / `ConstantIndex` /
    /// `Downcast` / `OpaqueCast` / `Subtype`) applied to a running
    /// leaf offset, type and — for an ADT place — the variant the
    /// projections have downcast to (0 until one does). Shared by
    /// the frame segment of place resolution.
    fn frame_step(
        &self,
        offset: &mut usize,
        ty: &mut mir::Ty,
        variant: &mut u32,
        elem: &mir::PlaceElem,
    ) -> Result<(), JitError> {
        match elem {
            mir::PlaceElem::Field(idx, field_ty) => {
                if let mir::Ty::Adt(..) = &*ty {
                    let layout = self.adt_layout(ty)?;
                    let slot = layout.field(*variant, *idx).ok_or_else(|| {
                        self.err("Field projection outside the ADT layout")
                    })?;
                    *offset += slot.offset;
                    *ty = field_ty.clone();
                    *variant = 0;
                    return Ok(());
                }
                let mir::Ty::Tuple(fields) = &*ty else {
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
                    *offset += flatten(f, &self.func.name, self.layouts)?;
                }
                *ty = field_ty.clone();
                *variant = 0;
                Ok(())
            }
            mir::PlaceElem::Downcast(_, v) => {
                if !matches!(&*ty, mir::Ty::Adt(..)) {
                    return Err(self.err(&format!(
                        "Downcast projection on non-ADT type {}",
                        ty.kind_name()
                    )));
                }
                *variant = *v;
                Ok(())
            }
            mir::PlaceElem::ConstantIndex {
                offset: idx,
                min_length: _,
                from_end,
            } => {
                let mir::Ty::Array(elem, len) = &*ty else {
                    return Err(self.err(&format!(
                        "ConstantIndex projection on non-array type {}",
                        ty.kind_name()
                    )));
                };
                let mir::ArrayLen::Known(len) = len else {
                    return Err(
                        self.err("ConstantIndex projection on an array of unknown length")
                    );
                };
                // `from_end` offsets count back from one past the
                // last element (offset 1 is the last element), as in
                // rustc and the interpreter.
                let idx = if *from_end {
                    match len.checked_sub(*idx).filter(|i| *i < *len) {
                        Some(i) => i,
                        None => {
                            return Err(self.err("ConstantIndex projection out of range"));
                        }
                    }
                } else if idx < len {
                    *idx
                } else {
                    return Err(self.err("ConstantIndex projection out of range"));
                };
                *offset += idx as usize * flatten(elem, &self.func.name, self.layouts)?;
                *ty = (**elem).clone();
                *variant = 0;
                Ok(())
            }
            mir::PlaceElem::OpaqueCast(t) | mir::PlaceElem::Subtype(t) => {
                *ty = t.clone();
                *variant = 0;
                Ok(())
            }
            other => Err(self.err(&format!("{} projection", other.kind_name()))),
        }
    }

}
