
use sparagmos_mir as mir;

use crate::ir::{
    BinOp as IrBinOp,
    Instr,
    Operand as IrOperand,
    Terminator as IrTerminator,
};
use crate::builtins::{self, Builtin, FmtDescriptor, FmtTrait, FmtTy};
use crate::program::{StaticBlob};

use super::*;

impl<'a> FnLower<'a> {
    pub(super) fn lower_terminator(&mut self, term: &mir::TerminatorKind) -> Result<(), JitError> {
        use mir::TerminatorKind as T;
        match term {
            T::Goto { target } => {
                self.seal_terminator(IrTerminator::Jump(target.index() as u32));
                Ok(())
            }
            T::Return => {
                let (slot, count) = self.local_slots[0];
                let term = match count {
                    0 => IrTerminator::Return {
                        value: IrOperand::Const(0),
                        second: None,
                    },
                    1 => IrTerminator::Return {
                        value: IrOperand::Value(slot),
                        second: None,
                    },
                    // A two-word result (a fat pointer, a pair):
                    // the leaves are consecutive slots, returned
                    // in RAX/RDX (see the IR docs).
                    2 => IrTerminator::Return {
                        value: IrOperand::Value(slot),
                        second: Some(IrOperand::Value(slot + 1)),
                    },
                    // A wider result travels through the hidden
                    // sret argument: store each return word into
                    // the caller's destination storage (leaf `k`
                    // lies `8 * k` bytes below the base address, the
                    // slot model's addressing), then return a dummy
                    // word. The stores happen only here, at the
                    // return itself, so a panicking function leaves
                    // the caller's destination untouched.
                    _ => {
                        let base = self
                            .sret_slot
                            .expect("a wide return implies the sret layout");
                        for k in 0..count as u32 {
                            let a = if k == 0 {
                                base
                            } else {
                                let d = self.fresh();
                                self.emit(Instr::BinOp {
                                    dst: d,
                                    op: IrBinOp::Sub,
                                    lhs: IrOperand::Value(base),
                                    rhs: IrOperand::Const((k * 8) as i64),
                                });
                                d
                            };
                            self.emit(Instr::StorePtr {
                                ptr: IrOperand::Value(a),
                                src: IrOperand::Value(slot + k),
                            });
                        }
                        IrTerminator::Return {
                            value: IrOperand::Const(0),
                            second: None,
                        }
                    }
                };
                self.seal_terminator(term);
                Ok(())
            }
            T::Unreachable => {
                // Reaching an `Unreachable` block is a guest panic,
                // exactly the interpreter's model: lower it to a jump
                // into a panic block, the same mechanism `Assert`
                // uses, so the panic is recorded through the runtime
                // path and propagates via the program's panic flag.
                // (Lowering it to the backend's trap terminator made
                // the generated code execute int3 and kill the host
                // process with SIGTRAP.)
                let msg_idx = self.messages.intern("entered unreachable code");
                let pblk = self.panic_block(msg_idx);
                self.seal_terminator(IrTerminator::Jump(pblk));
                Ok(())
            }
            T::SwitchInt { discr, targets } => self.lower_switch(discr, targets),
            T::Call {
                func,
                args,
                destination,
                target,
                ..
            } => self.lower_call(func, args, destination, *target),
            T::Assert {
                cond,
                expected,
                msg,
                target,
                ..
            } => {
                // A bounds-check assert carries its message's
                // operands: the panic text is the index panic
                // formatted at run time from the actual length
                // and index, exactly as an index operation's own
                // check formats it (and as the interpreter
                // formats the same assert). Every other kind's
                // message is a static text.
                let pblk = match &**msg {
                    mir::AssertMessage::BoundsCheck { len, index } => {
                        let len_ty = self
                            .operand_place_ty(len)
                            .unwrap_or(mir::Ty::Uint(mir::UintTy::Usize));
                        let idx_ty = self
                            .operand_place_ty(index)
                            .unwrap_or(mir::Ty::Uint(mir::UintTy::Usize));
                        let len_v = self.operand_value(len, &len_ty)?;
                        let idx_v = self.operand_value(index, &idx_ty)?;
                        let id = self.new_block();
                        self.seal_block(
                            id,
                            vec![Instr::PanicIndexOob {
                                len: IrOperand::Value(len_v),
                                index: IrOperand::Value(idx_v),
                            }],
                            IrTerminator::Unreachable,
                        );
                        id
                    }
                    _ => {
                        let text = assert_message_text(msg);
                        let msg_idx = self.messages.intern(&text);
                        self.panic_block(msg_idx)
                    }
                };
                let v = self.operand_value(cond, &mir::Ty::Bool)?;
                let (then_blk, else_blk) = if *expected {
                    (target.index() as u32, pblk)
                } else {
                    (pblk, target.index() as u32)
                };
                self.seal_terminator(IrTerminator::Branch {
                    cond: IrOperand::Value(v),
                    then_blk,
                    else_blk,
                });
                Ok(())
            }
            T::Drop { place, target, .. } => {
                // Dropping a value in the JIT's model runs no code
                // (the model has no destructors); the place must
                // still resolve, in frame or in memory.
                self.classify_place(place)?;
                self.seal_terminator(IrTerminator::Jump(target.index() as u32));
                Ok(())
            }
            T::FalseEdge { real_target, .. } | T::FalseUnwind { real_target, .. } => {
                self.seal_terminator(IrTerminator::Jump(real_target.index() as u32));
                Ok(())
            }
            T::Resume => {
                // The JIT's panic model has no unwinding: a
                // guest panic propagates through the program's
                // panic flag and the `CheckPanic` after each
                // call, so no call ever transfers control to its
                // cleanup blocks, and a cleanup block's closing
                // `Resume` is never reached. Seal it the way a
                // MIR `Unreachable` is sealed — a jump into a
                // panic block — so that reaching one would be a
                // recorded guest panic, never a host trap.
                let msg_idx = self.messages.intern("unwind resume reached");
                let pblk = self.panic_block(msg_idx);
                self.seal_terminator(IrTerminator::Jump(pblk));
                Ok(())
            }
            T::Terminate(_) => Err(self.err("Terminate terminator")),
            T::Yield { .. } => Err(self.err("Yield terminator")),
            T::CoroutineDrop => Err(self.err("CoroutineDrop terminator")),
            T::InlineAsm {
                template,
                options,
                destination,
                ..
            } => {
                if mir::inline_asm_is_noop(template, options) {
                    match destination {
                        Some(target) => {
                            self.seal_terminator(IrTerminator::Jump(target.index() as u32));
                            Ok(())
                        }
                        None => Err(self.err("InlineAsm terminator without a destination")),
                    }
                } else {
                    Err(self.err("InlineAsm terminator"))
                }
            }
            T::TailCall { .. } => Err(self.err("TailCall terminator")),
        }
    }

    /// Seal the current MIR block with its terminator. The
    /// continuation id passed to `seal_and_continue` is the block
    /// itself: `run` re-initialises `cur`/`cur_id` for the next MIR
    /// block before anything else is emitted, so the stale
    /// continuation is never filled.
    fn seal_terminator(&mut self, term: IrTerminator) {
        let id = self.cur_id;
        self.seal_and_continue(term, id);
    }

    fn lower_switch(
        &mut self,
        discr: &mir::Operand,
        targets: &mir::SwitchTargets,
    ) -> Result<(), JitError> {
        let discr_ty = self
            .operand_place_ty(discr)
            .unwrap_or(mir::Ty::Uint(mir::UintTy::U64));
        if let Some(signed) = int128_signed(&discr_ty) {
            return self.lower_switch128(discr, &discr_ty, signed, targets);
        }
        let v = self.operand_value(discr, &discr_ty)?;
        if targets.values.is_empty() {
            self.seal_terminator(IrTerminator::Jump(targets.otherwise.index() as u32));
            return Ok(());
        }
        for (i, (value, target)) in targets
            .values
            .iter()
            .zip(targets.targets.iter())
            .enumerate()
        {
            let eq = self.fresh();
            self.emit(Instr::BinOp {
                dst: eq,
                op: IrBinOp::Eq,
                lhs: IrOperand::Value(v),
                rhs: IrOperand::Const(normalize_u128(*value, &discr_ty)),
            });
            let last = i + 1 == targets.values.len();
            let next = if last {
                targets.otherwise.index() as u32
            } else {
                self.new_block()
            };
            self.seal_and_continue(
                IrTerminator::Branch {
                    cond: IrOperand::Value(eq),
                    then_blk: target.index() as u32,
                    else_blk: next,
                },
                next,
            );
            // When `last`, `next` is the `otherwise` MIR block:
            // seal_and_continue only moves the fill cursor there, and
            // run() re-initialises the cursor before that block is
            // filled with its own contents. The block itself is not
            // sealed here.
        }
        Ok(())
    }

    /// `SwitchInt` on a 128-bit discriminant: each target value
    /// is a full-width constant, compared against the
    /// discriminant's word pair.
    fn lower_switch128(
        &mut self,
        discr: &mir::Operand,
        discr_ty: &mir::Ty,
        signed: bool,
        targets: &mir::SwitchTargets,
    ) -> Result<(), JitError> {
        let l = self.operand_leaves(discr, discr_ty, 2)?;
        if targets.values.is_empty() {
            self.seal_terminator(IrTerminator::Jump(targets.otherwise.index() as u32));
            return Ok(());
        }
        for (i, (value, target)) in targets
            .values
            .iter()
            .zip(targets.targets.iter())
            .enumerate()
        {
            let lo = self.const_slot(*value as u64);
            let hi = self.const_slot((*value >> 64) as u64);
            let eq = self.emit_cmp128(mir::BinOp::Eq, signed, &l, &[lo, hi]);
            let last = i + 1 == targets.values.len();
            let next = if last {
                targets.otherwise.index() as u32
            } else {
                self.new_block()
            };
            self.seal_and_continue(
                IrTerminator::Branch {
                    cond: IrOperand::Value(eq),
                    then_blk: target.index() as u32,
                    else_blk: next,
                },
                next,
            );
        }
        Ok(())
    }

    /// The recorded full name of a definition, for diagnostics
    /// and the builtin layer's name resolution.
    pub(super) fn callee_name(&self, def: mir::DefId) -> String {
        self.def_names
            .get(&def)
            .cloned()
            .unwrap_or_else(|| format!("{def}"))
    }

    // ── The builtin layer's call lowering ──────────────────────
    //
    // A call whose callee the program does not define is resolved
    // by name, exactly like the interpreter's builtin resolution
    // (mirrored in `crate::builtins`). The discipline is the guest
    // call's: operands are read with the same helpers, the
    // destination is classified the same way, and an operation
    // that can fail the guest program is followed by a panic
    // check, like a guest call.

    /// The destination of a builtin call: its location and word
    /// count, with scratch words reserved for a memory place (the
    /// results land there and are stored through by
    /// [`FnLower::finish_builtin_call`]). A zero-word destination
    /// gets a fresh dummy slot, like the guest call's.
    fn builtin_destination(
        &mut self,
        destination: &mir::Place,
    ) -> Result<(PlaceLoc, u32, usize), JitError> {
        let dst_loc = self.classify_place(destination)?;
        let (dst, count) = match &dst_loc {
            PlaceLoc::Mem { count, .. } => {
                let count = match count {
                    Some(c) => *c,
                    None => {
                        return Err(
                            self.err("builtin call destination is an unsized place")
                        )
                    }
                };
                let first = self.fresh();
                for _ in 1..count.max(1) {
                    self.fresh();
                }
                (first, count)
            }
            PlaceLoc::Slots { .. } => {
                let (dst, count, _) = self.place_leaves(destination)?;
                if count == 0 {
                    (self.fresh(), 0)
                } else {
                    (dst, count)
                }
            }
        };
        Ok((dst_loc, dst, count))
    }

    /// Emit a host call (arguments zero-padded to the host
    /// convention's five words).
    pub(super) fn emit_host_call(
        &mut self,
        host: u32,
        dst: u32,
        dst2: Option<u32>,
        mut args: Vec<IrOperand>,
    ) {
        while args.len() < 5 {
            args.push(IrOperand::Const(0));
        }
        self.emit(Instr::HostCall {
            dst,
            dst2,
            host,
            args,
        });
    }

    /// Finish a builtin call: the panic check (when the operation
    /// can fail), the store-through for a memory destination, and
    /// the terminator seal — the guest call's tail.
    fn finish_builtin_call(
        &mut self,
        dst_loc: &PlaceLoc,
        dst: u32,
        target: Option<mir::BasicBlock>,
        check_panic: bool,
    ) {
        if check_panic {
            self.emit(Instr::CheckPanic);
        }
        if let PlaceLoc::Mem { .. } = dst_loc {
            self.store_mem_from(dst_loc, dst);
        }
        match target {
            Some(t) => self.seal_terminator(IrTerminator::Jump(t.index() as u32)),
            None => self.seal_terminator(IrTerminator::Unreachable),
        }
    }

    /// The address a `String`-producing builtin call writes
    /// its result through: the (pointer, capacity, length)
    /// triple is three words, more than the host-call
    /// convention returns, so the host function writes it at
    /// the destination's own address (the sret shape the guest
    /// call uses for wide results) — the destination place's
    /// slot-run address, or its memory address with its checks
    /// emitted. The host call's own result word is a dummy (see
    /// [`Self::finish_string_call`]).
    fn string_destination(
        &mut self,
        destination: &mir::Place,
    ) -> Result<u32, JitError> {
        let dst_loc = self.classify_place(destination)?;
        match &dst_loc {
            PlaceLoc::Slots { first, count, .. } => {
                if *count != 3 {
                    return Err(self.err("String destination is not three words"));
                }
                let addr = self.fresh();
                self.emit(Instr::SlotAddr {
                    dst: addr,
                    slot: *first,
                });
                Ok(addr)
            }
            PlaceLoc::Mem {
                addr,
                count,
                checks,
                ..
            } => {
                if *count != Some(3) {
                    return Err(self.err("String destination is not three words"));
                }
                self.emit_checks(checks);
                Ok(self.materialize_addr(addr))
            }
        }
    }

    /// Finish a `String`-producing builtin call: the panic
    /// check and the terminator seal. Unlike
    /// [`Self::finish_builtin_call`] there is no store-through:
    /// the host function already wrote the result at the
    /// destination's address.
    fn finish_string_call(&mut self, target: Option<mir::BasicBlock>) {
        self.emit(Instr::CheckPanic);
        match target {
            Some(t) => self.seal_terminator(IrTerminator::Jump(t.index() as u32)),
            None => self.seal_terminator(IrTerminator::Unreachable),
        }
    }

    /// The (data pointer, length) word slots of the string-like
    /// value a `PartialEq` argument references, under the
    /// argument's value type: a `str` value is the fat reference
    /// itself; a `String` value's pair is read out of its triple
    /// by the host function; a reference to either first loads
    /// the inner reference's words.
    fn string_pair_of(
        &mut self,
        arg: &mir::Operand,
        value_ty: &mir::Ty,
    ) -> Result<(u32, u32), JitError> {
        match value_ty {
            mir::Ty::Str => self.operand_pair(arg),
            mir::Ty::Adt(def, _) if self.is_string_def(*def) => {
                let addr = self.arg_word(arg, "string comparison argument")?;
                let a = self.fresh();
                let b = self.fresh();
                debug_assert_eq!(b, a + 1);
                self.emit_host_call(
                    builtins::HOST_DEREF_BUFFER,
                    a,
                    Some(b),
                    vec![IrOperand::Value(addr)],
                );
                Ok((a, b))
            }
            mir::Ty::Ref(_, inner, _) => match &**inner {
                mir::Ty::Str => {
                    let base = self.arg_word(arg, "string comparison argument")?;
                    let ptr = self.fresh();
                    self.emit(Instr::LoadPtr {
                        dst: ptr,
                        ptr: IrOperand::Value(base),
                    });
                    let len_addr = self.word_addr(base, 1);
                    let len = self.fresh();
                    self.emit(Instr::LoadPtr {
                        dst: len,
                        ptr: IrOperand::Value(len_addr),
                    });
                    Ok((ptr, len))
                }
                mir::Ty::Adt(def, _) if self.is_string_def(*def) => {
                    let base = self.arg_word(arg, "string comparison argument")?;
                    let inner_addr = self.fresh();
                    self.emit(Instr::LoadPtr {
                        dst: inner_addr,
                        ptr: IrOperand::Value(base),
                    });
                    let a = self.fresh();
                    let b = self.fresh();
                    debug_assert_eq!(b, a + 1);
                    self.emit_host_call(
                        builtins::HOST_DEREF_BUFFER,
                        a,
                        Some(b),
                        vec![IrOperand::Value(inner_addr)],
                    );
                    Ok((a, b))
                }
                _ => Err(self.err("string comparison of an unsupported shape")),
            },
            _ => Err(self.err("string comparison of an unsupported shape")),
        }
    }

    /// Whether the ADT definition is `std::string::String`.
    fn is_string_def(&self, def: mir::DefId) -> bool {
        builtins::canonical_name(&self.callee_name(def)) == "std::string::String"
    }

    /// Lower a call whose semantics is moving the argument's
    /// own leaves into the destination (the identity
    /// conversions `From::from` / `Into::into` serve between
    /// equal types and between integer types, and
    /// `hint::must_use`).
    fn lower_leaf_copy(
        &mut self,
        arg: &mir::Operand,
        arg_ty: &mir::Ty,
        destination: &mir::Place,
        target: Option<mir::BasicBlock>,
    ) -> Result<(), JitError> {
        let (dst_loc, dst, count) = self.builtin_destination(destination)?;
        let want = flatten(arg_ty, &self.func.name, self.layouts)?;
        if count != want {
            return Err(self.err("identity conversion with a mismatched shape"));
        }
        let leaves = self.operand_leaves(arg, arg_ty, count)?;
        for (k, leaf) in leaves.iter().enumerate() {
            self.emit(Instr::Copy {
                dst: dst + k as u32,
                src: IrOperand::Value(*leaf),
            });
        }
        self.finish_builtin_call(&dst_loc, dst, target, false);
        Ok(())
    }

    /// One word of a place operand (the operand's own type taken
    /// from its place).
    fn arg_word(&mut self, arg: &mir::Operand, what: &str) -> Result<u32, JitError> {
        let ty = self
            .operand_place_ty(arg)
            .ok_or_else(|| self.err(&format!("{what} from a constant")))?;
        self.operand_value(arg, &ty)
    }

    /// Intern an fmt descriptor for (trait, pointee type),
    /// validating the type's shape at compile time.
    fn intern_descriptor(
        &mut self,
        trait_: FmtTrait,
        ty: &mir::Ty,
    ) -> Result<u32, JitError> {
        let desc = FmtDescriptor {
            trait_,
            ty: self.fmt_ty(trait_, ty)?,
        };
        if let Some(i) = self.tables.descriptors.iter().position(|d| *d == desc) {
            return Ok(i as u32);
        }
        let i = self.tables.descriptors.len() as u32;
        self.tables.descriptors.push(desc);
        Ok(i)
    }

    /// The [`FmtTy`] of a pointee type, with the trait's shape
    /// rules applied (radix traits format integers, exponent
    /// traits integers and floats; the Pointer trait has no JIT
    /// builtin, like the interpreter).
    fn fmt_ty(&self, trait_: FmtTrait, ty: &mir::Ty) -> Result<FmtTy, JitError> {
        if trait_ == FmtTrait::Pointer {
            return Err(self.err("fmt argument with the Pointer trait"));
        }
        if trait_ == FmtTrait::DebugNoop {
            return Ok(FmtTy::Ignored);
        }
        let v = self.fmt_ty_of(ty)?;
        let mut core = &v;
        while let FmtTy::Ref(inner) = core {
            core = inner;
        }
        match trait_ {
            FmtTrait::Octal
            | FmtTrait::LowerHex
            | FmtTrait::UpperHex
            | FmtTrait::Binary => {
                if !matches!(core, FmtTy::Int { .. } | FmtTy::Uint { .. }) {
                    return Err(self.err("fmt radix argument of a non-integer type"));
                }
            }
            FmtTrait::LowerExp | FmtTrait::UpperExp => {
                if !matches!(
                    core,
                    FmtTy::Int { .. } | FmtTy::Uint { .. } | FmtTy::F32 | FmtTy::F64
                ) {
                    return Err(self.err("fmt exponent argument of a non-numeric type"));
                }
            }
            _ => {}
        }
        Ok(v)
    }

    /// The runtime shape of a formatted value's type. Only the
    /// scalar and reference shapes the fmt engine reads are
    /// admitted; anything else is an explicit unsupported error.
    fn fmt_ty_of(&self, ty: &mir::Ty) -> Result<FmtTy, JitError> {
        Ok(match ty {
            mir::Ty::Int(it) => FmtTy::Int {
                bits: int_bits(*it),
            },
            mir::Ty::Uint(ut) => FmtTy::Uint {
                bits: uint_bits(*ut),
            },
            mir::Ty::Bool => FmtTy::Bool,
            mir::Ty::Char => FmtTy::Char,
            mir::Ty::Float(mir::FloatTy::F32) => FmtTy::F32,
            mir::Ty::Float(mir::FloatTy::F64) => FmtTy::F64,
            mir::Ty::Float(_) => {
                return Err(self.err("fmt argument of a 16/128-bit float type"))
            }
            mir::Ty::Str => FmtTy::Str,
            mir::Ty::Ref(_, inner, _) => FmtTy::Ref(Box::new(self.fmt_ty_of(inner)?)),
            mir::Ty::Adt(def, _) => {
                // A `String`'s formatted content is the byte
                // string its (pointer, capacity, length) triple
                // names — the same content its own `Display` /
                // `Debug` impls present.
                if builtins::canonical_name(&self.callee_name(*def))
                    == "std::string::String"
                {
                    return Ok(FmtTy::OwnedStr);
                }
                return Err(self.err(&format!(
                    "fmt argument of ADT type `{}`",
                    self.callee_name(*def)
                )))
            }
            other => {
                return Err(self.err(&format!(
                    "fmt argument of type {}",
                    other.kind_name()
                )))
            }
        })
    }

    /// Intern a literal string for the builtin layer.
    fn intern_string(&mut self, text: &str) -> u32 {
        if let Some(i) = self.tables.strings.iter().position(|t| t == text) {
            return i as u32;
        }
        let i = self.tables.strings.len() as u32;
        self.tables.strings.push(text.to_string());
        i
    }

    /// Intern a static byte blob — as a
    /// [`StaticBlob::GuestBytes`] in the program's statics table,
    /// the one interning mechanism for static data — and
    /// materialise its address in a fresh slot.
    fn intern_data_addr(&mut self, bytes: &[u8]) -> u32 {
        let index = self.statics.intern(StaticBlob::GuestBytes(bytes.to_vec()));
        let dst = self.fresh();
        self.emit(Instr::StaticAddr { dst, index });
        dst
    }

    /// The statics index of a byte array's
    /// [`StaticBlob::GuestBytes`] blob, when `ctx` is a reference
    /// (or raw pointer) to a byte array of exactly the bytes'
    /// length; `None` otherwise.
    pub(super) fn byte_array_blob_index(&mut self, b: &[u8], ctx: &mir::Ty) -> Option<u32> {
        let inner = match ctx {
            mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => inner,
            _ => return None,
        };
        let mir::Ty::Array(elem, mir::ArrayLen::Known(n)) = &**inner else {
            return None;
        };
        if !matches!(&**elem, mir::Ty::Uint(mir::UintTy::U8)) {
            return None;
        }
        if *n != b.len() as u64 {
            return None;
        }
        Some(self.statics.intern(StaticBlob::GuestBytes(b.to_vec())))
    }

    /// A byte-array constant (an fmt pieces template) read as a
    /// value: its address in the program's statics table, when the
    /// constant's type in context is a reference to a byte array
    /// of exactly the constant's length.
    pub(super) fn const_data_addr(
        &mut self,
        c: &mir::Constant,
        ctx: &mir::Ty,
    ) -> Option<u32> {
        let mir::ConstKind::Value(mir::ConstValue::Bytes(b)) = &c.literal else {
            return None;
        };
        let index = self.byte_array_blob_index(b, ctx)?;
        let dst = self.fresh();
        self.emit(Instr::StaticAddr { dst, index });
        Some(dst)
    }

    /// Lower a call the program does not define through the
    /// builtin layer. A name the layer does not know is not a
    /// lowering failure: the interpreter serves such a call
    /// only if it is ever executed (it raises its
    /// unknown-function error at the call, at run time), so the
    /// JIT lowers it to a panic with that same message, reached
    /// only if control flow ever reaches the call. Whole-program
    /// lowering must not fail for a body that merely *contains*
    /// an unservable call - the demand passes collect bodies
    /// for the interpreter (the fmt impl bodies, say) whose
    /// inner calls name instances no body was collected for,
    /// and those bodies are never entered on the fmt path in
    /// either engine.
    fn lower_builtin_call(
        &mut self,
        generic_args: &[mir::Ty],
        def: mir::DefId,
        args: &[mir::Operand],
        destination: &mir::Place,
        target: Option<mir::BasicBlock>,
    ) -> Result<(), JitError> {
        let name = self.callee_name(def);
        let canonical = builtins::canonical_name(&name);
        let Some(kind) = builtins::lookup(&canonical) else {
            let msg = self.messages.intern(&format!(
                "call to function `{name}` which is not part of the program"
            ));
            let pblk = self.panic_block(msg);
            self.seal_terminator(IrTerminator::Jump(pblk));
            return Ok(());
        };
        match kind {
            Builtin::ArgumentNew(trait_) => {
                if args.len() != 1 {
                    return Err(self.err("fmt argument constructor with wrong arity"));
                }
                let Some(pointee_ty) = generic_args.first() else {
                    return Err(
                        self.err("fmt argument constructor without a pointee type")
                    );
                };
                let desc = self.intern_descriptor(trait_, pointee_ty)?;
                // The argument is a reference to the pointee: one
                // word (thin), or two for a `str` pointee (fat).
                let arg_ty = self.operand_place_ty(&args[0]).ok_or_else(|| {
                    self.err("fmt argument constructor from a constant")
                })?;
                let words = flatten(&arg_ty, &self.func.name, self.layouts)?;
                let (ptr, second, fat) = match words {
                    1 => (
                        self.operand_value(&args[0], &arg_ty)?,
                        IrOperand::Const(0),
                        0,
                    ),
                    2 => {
                        if self.tables.descriptors[desc as usize].ty != FmtTy::Str {
                            return Err(
                                self.err("fat fmt argument of a non-str type")
                            );
                        }
                        let (a, b) = self.operand_pair(&args[0])?;
                        (a, IrOperand::Value(b), 1)
                    }
                    _ => return Err(self.err("fmt argument of an unsupported shape")),
                };
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 1 {
                    return Err(self.err("fmt argument destination is not one word"));
                }
                self.emit_host_call(
                    builtins::HOST_ARGUMENT_NEW,
                    dst,
                    None,
                    vec![
                        IrOperand::Value(ptr),
                        second,
                        IrOperand::Const(desc as i64),
                        IrOperand::Const(fat),
                    ],
                );
                self.finish_builtin_call(&dst_loc, dst, target, true);
                Ok(())
            }
            Builtin::ArgumentFromUsize => {
                if args.len() != 1 {
                    return Err(self.err("fmt count constructor with wrong arity"));
                }
                let ptr = self.arg_word(&args[0], "fmt count argument")?;
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 1 {
                    return Err(self.err("fmt count destination is not one word"));
                }
                self.emit_host_call(
                    builtins::HOST_ARGUMENT_FROM_USIZE,
                    dst,
                    None,
                    vec![IrOperand::Value(ptr)],
                );
                self.finish_builtin_call(&dst_loc, dst, target, true);
                Ok(())
            }
            Builtin::PartialEqEq | Builtin::PartialEqNe => {
                // Semantic (derived) value equality, the
                // interpreter's builtin of the same name: both
                // arguments are references to the compared values.
                // The JIT lowers it for the shapes whose derived
                // equality is leaf-wise word equality (see
                // `word_transparent`): the pointees' words are
                // loaded and compared word by word.
                if args.len() != 2 {
                    return Err(self.err("PartialEq call with wrong arity"));
                }
                let Some(self_ty) = generic_args.first() else {
                    return Err(self.err("PartialEq call without a Self type"));
                };
                let self_ty = self_ty.clone();
                // String equality: the `PartialEq` impls between
                // `String` and `str`, in any reference
                // combination, compare the contents — each side's
                // (pointer, length) pair goes to the host
                // comparison, whose result is the answer (negated
                // for `ne`, as below).
                let rhs_ty = generic_args
                    .get(1)
                    .cloned()
                    .unwrap_or_else(|| self_ty.clone());
                let string_like = |ty: &mir::Ty| match ty {
                    mir::Ty::Str => true,
                    mir::Ty::Adt(def, _) => self.is_string_def(*def),
                    mir::Ty::Ref(_, inner, _) => match &**inner {
                        mir::Ty::Str => true,
                        mir::Ty::Adt(def, _) => self.is_string_def(*def),
                        _ => false,
                    },
                    _ => false,
                };
                if string_like(&self_ty) || string_like(&rhs_ty) {
                    let (p1, l1) = self.string_pair_of(&args[0], &self_ty)?;
                    let (p2, l2) = self.string_pair_of(&args[1], &rhs_ty)?;
                    let raw = self.fresh();
                    self.emit_host_call(
                        builtins::HOST_STR_EQ,
                        raw,
                        None,
                        vec![
                            IrOperand::Value(p1),
                            IrOperand::Value(l1),
                            IrOperand::Value(p2),
                            IrOperand::Value(l2),
                        ],
                    );
                    let result = if kind == Builtin::PartialEqNe {
                        let d = self.fresh();
                        self.emit(Instr::BinOp {
                            dst: d,
                            op: IrBinOp::Xor,
                            lhs: IrOperand::Value(raw),
                            rhs: IrOperand::Const(1),
                        });
                        d
                    } else {
                        raw
                    };
                    let (dst_loc, dst, dcount) =
                        self.builtin_destination(destination)?;
                    if dcount != 1 {
                        return Err(
                            self.err("PartialEq destination is not one word")
                        );
                    }
                    self.emit(Instr::Copy {
                        dst,
                        src: IrOperand::Value(result),
                    });
                    self.finish_builtin_call(&dst_loc, dst, target, true);
                    return Ok(());
                }
                if !word_transparent(&self_ty) {
                    return Err(self.err(&format!(
                        "PartialEq on type {}",
                        self_ty.kind_name()
                    )));
                }
                let count = flatten(&self_ty, &self.func.name, self.layouts)?;
                let mut words: Vec<Vec<u32>> = Vec::with_capacity(2);
                for arg in args {
                    let (mir::Operand::Copy(p) | mir::Operand::Move(p)) = arg else {
                        return Err(self.err("PartialEq argument is not a place"));
                    };
                    // The pointee place: the argument's place
                    // with a dereference appended.
                    let mut projection = p.projection.clone();
                    projection.push(mir::PlaceElem::Deref);
                    let pointee = mir::Place {
                        local: p.local,
                        projection,
                    };
                    words.push(self.operand_leaves(
                        &mir::Operand::Copy(pointee),
                        &self_ty,
                        count,
                    )?);
                }
                let mut eq: Option<u32> = None;
                for k in 0..count {
                    let e = self.fresh();
                    self.emit(Instr::BinOp {
                        dst: e,
                        op: IrBinOp::Eq,
                        lhs: IrOperand::Value(words[0][k]),
                        rhs: IrOperand::Value(words[1][k]),
                    });
                    eq = Some(match eq {
                        None => e,
                        Some(acc) => {
                            let d = self.fresh();
                            self.emit(Instr::BinOp {
                                dst: d,
                                op: IrBinOp::And,
                                lhs: IrOperand::Value(acc),
                                rhs: IrOperand::Value(e),
                            });
                            d
                        }
                    });
                }
                // A zero-word Self (unit) is trivially equal.
                let result = eq.unwrap_or_else(|| self.const_slot(1));
                let result = if kind == Builtin::PartialEqNe {
                    let d = self.fresh();
                    self.emit(Instr::BinOp {
                        dst: d,
                        op: IrBinOp::Xor,
                        lhs: IrOperand::Value(result),
                        rhs: IrOperand::Const(1),
                    });
                    d
                } else {
                    result
                };
                let (dst_loc, dst, dcount) = self.builtin_destination(destination)?;
                if dcount != 1 {
                    return Err(self.err("PartialEq destination is not one word"));
                }
                self.emit(Instr::Copy {
                    dst,
                    src: IrOperand::Value(result),
                });
                self.finish_builtin_call(&dst_loc, dst, target, false);
                Ok(())
            }
            Builtin::ArgumentsNew => {
                if args.len() != 2 {
                    return Err(self.err("fmt::Arguments::new with wrong arity"));
                }
                let (tmpl_ptr, tmpl_len) = match &args[0] {
                    mir::Operand::Constant(c) => match &c.literal {
                        mir::ConstKind::Value(mir::ConstValue::Bytes(b)) => {
                            (self.intern_data_addr(b), b.len() as i64)
                        }
                        _ => {
                            return Err(self.err(
                                "fmt::Arguments::new with a non-literal pieces constant",
                            ))
                        }
                    },
                    _ => {
                        let ty = self.operand_place_ty(&args[0]).ok_or_else(|| {
                            self.err("fmt::Arguments::new pieces operand without a type")
                        })?;
                        let n = match &ty {
                            mir::Ty::Ref(_, inner, _)
                            | mir::Ty::RawPtr(inner, _) => match &**inner {
                                mir::Ty::Array(elem, mir::ArrayLen::Known(n))
                                    if matches!(
                                        &**elem,
                                        mir::Ty::Uint(mir::UintTy::U8)
                                    ) =>
                                {
                                    *n
                                }
                                _ => {
                                    return Err(self.err(
                                        "fmt::Arguments::new pieces is not a byte array",
                                    ))
                                }
                            },
                            _ => {
                                return Err(self.err(
                                    "fmt::Arguments::new pieces is not a reference",
                                ))
                            }
                        };
                        (self.operand_value(&args[0], &ty)?, n as i64)
                    }
                };
                let arr_ty = self.operand_place_ty(&args[1]).ok_or_else(|| {
                    self.err("fmt::Arguments::new arguments operand without a type")
                })?;
                let argc = match &arr_ty {
                    mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => {
                        match &**inner {
                            mir::Ty::Array(_, mir::ArrayLen::Known(n)) => *n,
                            _ => {
                                return Err(self.err(
                                    "fmt::Arguments::new arguments is not an array",
                                ))
                            }
                        }
                    }
                    _ => {
                        return Err(self.err(
                            "fmt::Arguments::new arguments is not a reference",
                        ))
                    }
                };
                let arr = self.operand_value(&args[1], &arr_ty)?;
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 1 {
                    return Err(self.err("fmt::Arguments destination is not one word"));
                }
                self.emit_host_call(
                    builtins::HOST_ARGUMENTS_NEW,
                    dst,
                    None,
                    vec![
                        IrOperand::Value(tmpl_ptr),
                        IrOperand::Const(tmpl_len),
                        IrOperand::Value(arr),
                        IrOperand::Const(argc as i64),
                    ],
                );
                self.finish_builtin_call(&dst_loc, dst, target, true);
                Ok(())
            }
            Builtin::ArgumentsFromStr => {
                if args.len() != 1 {
                    return Err(self.err("fmt::Arguments::from_str with wrong arity"));
                }
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 1 {
                    return Err(self.err("fmt::Arguments destination is not one word"));
                }
                match &args[0] {
                    mir::Operand::Constant(c) => match &c.literal {
                        mir::ConstKind::Value(mir::ConstValue::Str(text)) => {
                            let index = self.intern_string(text);
                            self.emit_host_call(
                                builtins::HOST_ARGUMENTS_FROM_STR_CONST,
                                dst,
                                None,
                                vec![IrOperand::Const(index as i64)],
                            );
                        }
                        _ => {
                            return Err(self.err(
                                "fmt::Arguments::from_str with a non-literal string",
                            ))
                        }
                    },
                    _ => {
                        let (a, b) = self.operand_pair(&args[0])?;
                        self.emit_host_call(
                            builtins::HOST_ARGUMENTS_FROM_STR_RT,
                            dst,
                            None,
                            vec![IrOperand::Value(a), IrOperand::Value(b)],
                        );
                    }
                }
                self.finish_builtin_call(&dst_loc, dst, target, true);
                Ok(())
            }
            Builtin::Print | Builtin::Eprint => {
                let to_stderr = matches!(kind, Builtin::Eprint);
                if args.len() != 1 {
                    return Err(self.err("print intrinsic with wrong arity"));
                }
                let handle = self.arg_word(&args[0], "print argument")?;
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 0 {
                    return Err(self.err("print destination is not unit"));
                }
                self.emit_host_call(
                    builtins::HOST_PRINT,
                    dst,
                    None,
                    vec![
                        IrOperand::Value(handle),
                        IrOperand::Const(to_stderr as i64),
                    ],
                );
                self.finish_builtin_call(&dst_loc, dst, target, true);
                Ok(())
            }
            Builtin::RangeInclusiveNew => {
                if args.len() != 2 {
                    return Err(self.err("RangeInclusive::new with wrong arity"));
                }
                let Some(idx_ty) = generic_args.first() else {
                    return Err(self.err("RangeInclusive::new without an element type"));
                };
                let start = self.operand_value(&args[0], idx_ty)?;
                let end = self.operand_value(&args[1], idx_ty)?;
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 3 {
                    return Err(
                        self.err("RangeInclusive destination is not three words")
                    );
                }
                self.emit(Instr::Copy {
                    dst,
                    src: IrOperand::Value(start),
                });
                self.emit(Instr::Copy {
                    dst: dst + 1,
                    src: IrOperand::Value(end),
                });
                self.emit(Instr::Copy {
                    dst: dst + 2,
                    src: IrOperand::Const(0),
                });
                self.finish_builtin_call(&dst_loc, dst, target, false);
                Ok(())
            }
            Builtin::IntoIter => {
                if args.len() != 1 {
                    return Err(self.err("IntoIterator::into_iter with wrong arity"));
                }
                let Some(iter_ty) = generic_args.first() else {
                    return Err(self.err("IntoIterator::into_iter without a type"));
                };
                // Only the ranges are their own iterator state;
                // anything else keeps the explicit error (the
                // interpreter's layer answers it the same way).
                let mir::Ty::Adt(def, _) = iter_ty else {
                    return Err(self.err(
                        "IntoIterator::into_iter for a non-ADT iterable",
                    ));
                };
                let def_name = self.callee_name(*def);
                if def_name != "std::ops::Range"
                    && def_name != "std::ops::RangeInclusive"
                {
                    return Err(self.err(&format!(
                        "IntoIterator::into_iter for `{def_name}`"
                    )));
                }
                let count = flatten(iter_ty, &self.func.name, self.layouts)?;
                let leaves = self.operand_leaves(&args[0], iter_ty, count)?;
                let (dst_loc, dst, dcount) = self.builtin_destination(destination)?;
                if dcount != count {
                    return Err(self.err("into_iter destination has the wrong shape"));
                }
                for (k, src) in leaves.iter().enumerate() {
                    if dst + k as u32 != *src {
                        self.emit(Instr::Copy {
                            dst: dst + k as u32,
                            src: IrOperand::Value(*src),
                        });
                    }
                }
                self.finish_builtin_call(&dst_loc, dst, target, false);
                Ok(())
            }
            Builtin::IteratorNext => {
                if args.len() != 1 {
                    return Err(self.err("Iterator::next with wrong arity"));
                }
                let Some(iter_ty) = generic_args.first() else {
                    return Err(self.err("Iterator::next without an iterator type"));
                };
                let mir::Ty::Adt(def, iargs) = iter_ty else {
                    return Err(self.err("Iterator::next for a non-ADT iterator"));
                };
                let def_name = self.callee_name(*def);
                let inclusive = match def_name.as_str() {
                    "std::ops::Range" => false,
                    "std::ops::RangeInclusive" => true,
                    _ => {
                        return Err(self.err(&format!(
                            "Iterator::next for `{def_name}`"
                        )))
                    }
                };
                let Some(idx_ty) = iargs.first() else {
                    return Err(self.err("Iterator::next without an element type"));
                };
                let signed = match idx_ty {
                    mir::Ty::Int(_) => 1,
                    mir::Ty::Uint(_) | mir::Ty::Char => 0,
                    _ => {
                        return Err(self.err(
                            "Iterator::next for a range of non-integer type",
                        ))
                    }
                };
                let dst_ty = self.place_ty(destination)?;
                let mir::Ty::Adt(option_def, _) = dst_ty else {
                    return Err(self.err("Iterator::next destination is not an Option"));
                };
                let none = self.discriminant_value(option_def, 0)?;
                let some = self.discriminant_value(option_def, 1)?;
                let ptr = self.arg_word(&args[0], "Iterator::next receiver")?;
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 2 {
                    return Err(self.err("Iterator::next destination is not two words"));
                }
                let host = if inclusive {
                    builtins::HOST_NEXT_RANGE_INCLUSIVE
                } else {
                    builtins::HOST_NEXT_RANGE
                };
                self.emit_host_call(
                    host,
                    dst,
                    Some(dst + 1),
                    vec![
                        IrOperand::Value(ptr),
                        IrOperand::Const(signed),
                        IrOperand::Const(none),
                        IrOperand::Const(some),
                    ],
                );
                self.finish_builtin_call(&dst_loc, dst, target, true);
                Ok(())
            }
            Builtin::FloatMethod { f32: is32, op } => {
                // The registry resolved the method from the
                // receiver type's name, so the float type is the
                // one the name carried.
                let float_ty = mir::Ty::Float(if is32 {
                    mir::FloatTy::F32
                } else {
                    mir::FloatTy::F64
                });
                let binary = matches!(
                    op,
                    builtins::FloatOp::Powi
                        | builtins::FloatOp::Powf
                        | builtins::FloatOp::Min
                        | builtins::FloatOp::Max
                        | builtins::FloatOp::Log
                        | builtins::FloatOp::Atan2
                        | builtins::FloatOp::Hypot
                        | builtins::FloatOp::DivEuclid
                        | builtins::FloatOp::RemEuclid
                        | builtins::FloatOp::CopySign
                );
                let arity = if op == builtins::FloatOp::MulAdd {
                    3
                } else if binary {
                    2
                } else {
                    1
                };
                if args.len() != arity {
                    return Err(self.err("float method with wrong arity"));
                }
                let z = if op == builtins::FloatOp::MulAdd {
                    IrOperand::Value(self.operand_value(&args[2], &float_ty)?)
                } else {
                    IrOperand::Const(0)
                };
                let (x, y) = if op == builtins::FloatOp::FromBits {
                    let bits_ty = if is32 {
                        mir::Ty::Uint(mir::UintTy::U32)
                    } else {
                        mir::Ty::Uint(mir::UintTy::U64)
                    };
                    (
                        self.operand_value(&args[0], &bits_ty)?,
                        IrOperand::Const(0),
                    )
                } else {
                    let recv = self.operand_value(&args[0], &float_ty)?;
                    let y = if binary || op == builtins::FloatOp::MulAdd {
                        let y_ty = if op == builtins::FloatOp::Powi {
                            mir::Ty::Int(mir::IntTy::I32)
                        } else {
                            float_ty.clone()
                        };
                        IrOperand::Value(self.operand_value(&args[1], &y_ty)?)
                    } else {
                        IrOperand::Const(0)
                    };
                    (recv, y)
                };
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                let second = if op == builtins::FloatOp::SinCos {
                    if count != 2 {
                        return Err(self.err("sin_cos destination is not two words"));
                    }
                    Some(dst + 1)
                } else {
                    if count != 1 {
                        return Err(self.err("float method destination is not one word"));
                    }
                    None
                };
                self.emit_host_call(
                    builtins::HOST_FLOAT_METHOD,
                    dst,
                    second,
                    vec![
                        IrOperand::Value(x),
                        y,
                        IrOperand::Const(op.code() as i64),
                        IrOperand::Const(is32 as i64),
                        z,
                    ],
                );
                self.finish_builtin_call(&dst_loc, dst, target, true);
                Ok(())
            }
            Builtin::RustAlloc | Builtin::RustAllocZeroed => {
                if args.len() != 2 {
                    return Err(self.err("allocation call with wrong arity"));
                }
                let size = self.arg_word(&args[0], "allocation size")?;
                let align = self.arg_word(&args[1], "allocation alignment")?;
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 1 {
                    return Err(self.err("allocation destination is not one word"));
                }
                let host = if kind == Builtin::RustAlloc {
                    builtins::HOST_RUST_ALLOC
                } else {
                    builtins::HOST_RUST_ALLOC_ZEROED
                };
                self.emit_host_call(
                    host,
                    dst,
                    None,
                    vec![IrOperand::Value(size), IrOperand::Value(align)],
                );
                self.finish_builtin_call(&dst_loc, dst, target, false);
                Ok(())
            }
            Builtin::RustDealloc => {
                if args.len() != 3 {
                    return Err(self.err("deallocation call with wrong arity"));
                }
                let ptr = self.arg_word(&args[0], "deallocation pointer")?;
                let size = self.arg_word(&args[1], "deallocation size")?;
                let align = self.arg_word(&args[2], "deallocation alignment")?;
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 0 {
                    return Err(self.err("deallocation destination is not unit"));
                }
                self.emit_host_call(
                    builtins::HOST_RUST_DEALLOC,
                    dst,
                    None,
                    vec![
                        IrOperand::Value(ptr),
                        IrOperand::Value(size),
                        IrOperand::Value(align),
                    ],
                );
                self.finish_builtin_call(&dst_loc, dst, target, false);
                Ok(())
            }
            Builtin::RustRealloc => {
                if args.len() != 4 {
                    return Err(self.err("reallocation call with wrong arity"));
                }
                let ptr = self.arg_word(&args[0], "reallocation pointer")?;
                let old = self.arg_word(&args[1], "reallocation old size")?;
                let align = self.arg_word(&args[2], "reallocation alignment")?;
                let new = self.arg_word(&args[3], "reallocation new size")?;
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 1 {
                    return Err(self.err("reallocation destination is not one word"));
                }
                self.emit_host_call(
                    builtins::HOST_RUST_REALLOC,
                    dst,
                    None,
                    vec![
                        IrOperand::Value(ptr),
                        IrOperand::Value(old),
                        IrOperand::Value(align),
                        IrOperand::Value(new),
                    ],
                );
                self.finish_builtin_call(&dst_loc, dst, target, false);
                Ok(())
            }
            Builtin::Format => {
                if args.len() != 1 {
                    return Err(self.err("fmt::format with wrong arity"));
                }
                let handle = self.arg_word(&args[0], "fmt::format argument")?;
                let addr = self.string_destination(destination)?;
                let dummy = self.fresh();
                self.emit_host_call(
                    builtins::HOST_STRING_FORMAT,
                    dummy,
                    None,
                    vec![IrOperand::Value(addr), IrOperand::Value(handle)],
                );
                self.finish_string_call(target);
                Ok(())
            }
            Builtin::StrToUppercase | Builtin::StrToLowercase => {
                if args.len() != 1 {
                    return Err(self.err("case conversion with wrong arity"));
                }
                let (ptr, len) = self.operand_pair(&args[0])?;
                let addr = self.string_destination(destination)?;
                let host = if kind == Builtin::StrToUppercase {
                    builtins::HOST_STR_TO_UPPERCASE
                } else {
                    builtins::HOST_STR_TO_LOWERCASE
                };
                let dummy = self.fresh();
                self.emit_host_call(
                    host,
                    dummy,
                    None,
                    vec![
                        IrOperand::Value(addr),
                        IrOperand::Value(ptr),
                        IrOperand::Value(len),
                    ],
                );
                self.finish_string_call(target);
                Ok(())
            }
            Builtin::FromFrom | Builtin::IntoInto => {
                if args.len() != 1 {
                    return Err(self.err("conversion call with wrong arity"));
                }
                // `From::from` renders its generic arguments as
                // [target, source]; `Into::into` as [source,
                // target] (the interpreter's builtin resolves
                // the same pair the same way).
                let (target_ty, source_ty) = if kind == Builtin::IntoInto {
                    (generic_args.get(1), generic_args.first())
                } else {
                    (generic_args.first(), generic_args.get(1))
                };
                let (Some(target_ty), Some(source_ty)) = (target_ty, source_ty)
                else {
                    return Err(self.err("conversion call without generic types"));
                };
                let target_ty = target_ty.clone();
                let source_ty = source_ty.clone();
                let is_string = |ty: &mir::Ty| match ty {
                    mir::Ty::Adt(def, _) => {
                        builtins::canonical_name(&self.callee_name(*def))
                            == "std::string::String"
                    }
                    _ => false,
                };
                let is_str_source = |ty: &mir::Ty| match ty {
                    mir::Ty::Str => true,
                    mir::Ty::Ref(_, inner, _) => matches!(&**inner, mir::Ty::Str),
                    _ => false,
                };
                if is_string(&target_ty) && is_str_source(&source_ty) {
                    // `String::from(&str)`: copy the source's
                    // bytes into a fresh buffer.
                    let (ptr, len) = self.operand_pair(&args[0])?;
                    let addr = self.string_destination(destination)?;
                    let dummy = self.fresh();
                    self.emit_host_call(
                        builtins::HOST_STRING_FROM_STR,
                        dummy,
                        None,
                        vec![
                            IrOperand::Value(addr),
                            IrOperand::Value(ptr),
                            IrOperand::Value(len),
                        ],
                    );
                    self.finish_string_call(target);
                    return Ok(());
                }
                // The identity conversions: the reflexive
                // `From<T> for T`, and the value-preserving
                // conversions between integer types of the same
                // word count (the interpreter's builtin serves
                // the same set; the destination place's own
                // reads apply the width).
                let int_like = |ty: &mir::Ty| {
                    matches!(ty, mir::Ty::Int(_) | mir::Ty::Uint(_))
                };
                if target_ty == source_ty
                    || (int_like(&target_ty) && int_like(&source_ty))
                {
                    return self.lower_leaf_copy(
                        &args[0],
                        &source_ty,
                        destination,
                        target,
                    );
                }
                Err(self.err(&format!(
                    "From::from conversion from {} to {}",
                    source_ty.kind_name(),
                    target_ty.kind_name()
                )))
            }
            Builtin::Deref => {
                if args.len() != 1 {
                    return Err(self.err("Deref::deref with wrong arity"));
                }
                let Some(self_ty) = generic_args.first() else {
                    return Err(self.err("Deref::deref without a Self type"));
                };
                let buffer = match self_ty {
                    mir::Ty::Adt(def, _) => {
                        let name =
                            builtins::canonical_name(&self.callee_name(*def));
                        name == "std::string::String"
                            || name.starts_with("std::vec::Vec")
                    }
                    _ => false,
                };
                if !buffer {
                    return Err(self.err(&format!(
                        "Deref::deref for {}",
                        self_ty.kind_name()
                    )));
                }
                // The argument is `&Self`: the buffer value's
                // address, whose triple the host function reads
                // the (pointer, length) pair out of.
                let addr = self.arg_word(&args[0], "Deref::deref argument")?;
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 2 {
                    return Err(self.err("Deref::deref destination is not a pair"));
                }
                self.emit_host_call(
                    builtins::HOST_DEREF_BUFFER,
                    dst,
                    Some(dst + 1),
                    vec![IrOperand::Value(addr)],
                );
                self.finish_builtin_call(&dst_loc, dst, target, false);
                Ok(())
            }
            Builtin::Panicking | Builtin::PanicCountIsZero => {
                if !args.is_empty() {
                    return Err(self.err("panic-count query with arguments"));
                }
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 1 {
                    return Err(
                        self.err("panic-count query destination is not one word")
                    );
                }
                let imm = if kind == Builtin::PanicCountIsZero { 1 } else { 0 };
                self.emit(Instr::Const { dst, imm });
                self.finish_builtin_call(&dst_loc, dst, target, false);
                Ok(())
            }
            Builtin::MustUse => {
                if args.len() != 1 {
                    return Err(self.err("must_use with wrong arity"));
                }
                let Some(ty) = generic_args.first() else {
                    return Err(self.err("must_use without a type"));
                };
                let ty = ty.clone();
                self.lower_leaf_copy(&args[0], &ty, destination, target)
            }
            Builtin::IntBitOp(op) => {
                if args.len() != 1 {
                    return Err(self.err("bit intrinsic with wrong arity"));
                }
                let Some(ty) = generic_args.first() else {
                    return Err(self.err("bit intrinsic without a type"));
                };
                let bits = match ty {
                    mir::Ty::Int(it) => int_bits(*it),
                    mir::Ty::Uint(ut) => uint_bits(*ut),
                    _ => {
                        return Err(self.err("bit intrinsic of a non-integer type"))
                    }
                };
                let ty = ty.clone();
                let count = flatten(&ty, &self.func.name, self.layouts)?;
                let leaves = self.operand_leaves(&args[0], &ty, count)?;
                let (dst_loc, dst, dcount) = self.builtin_destination(destination)?;
                if dcount != count {
                    return Err(self.err("bit intrinsic destination shape mismatch"));
                }
                let hi = if count == 2 {
                    IrOperand::Value(leaves[1])
                } else {
                    IrOperand::Const(0)
                };
                self.emit_host_call(
                    builtins::HOST_BIT_OP,
                    dst,
                    if count == 2 { Some(dst + 1) } else { None },
                    vec![
                        IrOperand::Const(op as i64),
                        IrOperand::Value(leaves[0]),
                        hi,
                        IrOperand::Const(bits as i64),
                    ],
                );
                self.finish_builtin_call(&dst_loc, dst, target, false);
                Ok(())
            }
            Builtin::PanicNounwindFmt => {
                if args.is_empty() {
                    return Err(self.err("panic_nounwind_fmt with no arguments"));
                }
                let handle = self.arg_word(&args[0], "panic_nounwind_fmt argument")?;
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 0 {
                    return Err(self.err("panic_nounwind_fmt destination is not never"));
                }
                self.emit_host_call(
                    builtins::HOST_PANIC_NOUNWIND_FMT,
                    dst,
                    None,
                    vec![IrOperand::Value(handle)],
                );
                self.finish_builtin_call(&dst_loc, dst, target, true);
                Ok(())
            }
            Builtin::PanicStr => {
                // The callee's trailing arguments (a `&Location`
                // in the current core) are not the message; the
                // message is the first argument.
                if args.is_empty() {
                    return Err(self.err("panic with no arguments"));
                }
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 0 {
                    return Err(self.err("panic destination is not never"));
                }
                // The argument is a `&str` (a fat pair) -- or, in
                // the formatted shape the interpreter's builtin
                // also serves, a `fmt::Arguments` handle, whose
                // panic is the formatted message. Only a place can
                // hold the handle shape; a constant argument is
                // the pair.
                let is_arguments = match self.operand_place_ty(&args[0]) {
                    Some(mir::Ty::Adt(def, _)) => {
                        builtins::canonical_name(&self.callee_name(def))
                            == "std::fmt::Arguments"
                    }
                    _ => false,
                };
                if is_arguments {
                    let handle = self.arg_word(&args[0], "panic argument")?;
                    self.emit_host_call(
                        builtins::HOST_PANIC_FMT,
                        dst,
                        None,
                        vec![IrOperand::Value(handle)],
                    );
                } else {
                    let (ptr, len) = self.operand_pair(&args[0])?;
                    self.emit_host_call(
                        builtins::HOST_PANIC_STR,
                        dst,
                        None,
                        vec![IrOperand::Value(ptr), IrOperand::Value(len)],
                    );
                }
                self.finish_builtin_call(&dst_loc, dst, target, true);
                Ok(())
            }
            Builtin::PanicFmt => {
                // As for `panic`: a trailing `&Location` argument
                // is not the message.
                if args.is_empty() {
                    return Err(self.err("panic_fmt with no arguments"));
                }
                let handle = self.arg_word(&args[0], "panic_fmt argument")?;
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 0 {
                    return Err(self.err("panic_fmt destination is not never"));
                }
                self.emit_host_call(
                    builtins::HOST_PANIC_FMT,
                    dst,
                    None,
                    vec![IrOperand::Value(handle)],
                );
                self.finish_builtin_call(&dst_loc, dst, target, true);
                Ok(())
            }
            Builtin::PanicAny => {
                // The payload is the first argument; a trailing
                // `&Location` (the callee is `#[track_caller]`)
                // is not part of it. The destination is `!`.
                if args.is_empty() {
                    return Err(self.err("begin_panic with no arguments"));
                }
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 0 {
                    return Err(self.err("begin_panic destination is not never"));
                }
                // std's `payload_as_str`: a string payload
                // reports as itself -- a `&str` payload is the
                // fat pair itself; a `String` payload is its
                // (pointer, capacity, length) triple, of which
                // the panic message is the (pointer, length).
                // Any other payload reports as the opaque box.
                // The payload type is the callee's first generic
                // argument; the operand's own place type is the
                // fallback (a literal `&str` payload is a
                // constant, which has no place type).
                enum Payload {
                    StrPair,
                    String,
                    Opaque,
                }
                let payload_ty = generic_args
                    .first()
                    .cloned()
                    .or_else(|| self.operand_place_ty(&args[0]));
                let shape = match &payload_ty {
                    Some(mir::Ty::Ref(_, inner, _))
                        if matches!(&**inner, mir::Ty::Str) =>
                    {
                        Payload::StrPair
                    }
                    Some(mir::Ty::Adt(def, _))
                        if builtins::canonical_name(&self.callee_name(*def))
                            == "std::string::String" =>
                    {
                        Payload::String
                    }
                    _ => Payload::Opaque,
                };
                match shape {
                    Payload::StrPair => {
                        let (ptr, len) = self.operand_pair(&args[0])?;
                        self.emit_host_call(
                            builtins::HOST_PANIC_STR,
                            dst,
                            None,
                            vec![IrOperand::Value(ptr), IrOperand::Value(len)],
                        );
                        self.finish_builtin_call(&dst_loc, dst, target, true);
                    }
                    Payload::String => {
                        let ty = self.operand_place_ty(&args[0]).ok_or_else(|| {
                            self.err("begin_panic String payload without a type")
                        })?;
                        let words = flatten(&ty, &self.func.name, self.layouts)?;
                        let leaves = self.operand_leaves(&args[0], &ty, words)?;
                        let (Some(&ptr), Some(&len)) =
                            (leaves.first(), leaves.get(2))
                        else {
                            return Err(self.err(
                                "begin_panic String payload of an unsupported shape",
                            ));
                        };
                        self.emit_host_call(
                            builtins::HOST_PANIC_STR,
                            dst,
                            None,
                            vec![IrOperand::Value(ptr), IrOperand::Value(len)],
                        );
                        self.finish_builtin_call(&dst_loc, dst, target, true);
                    }
                    Payload::Opaque => {
                        let msg = self.messages.intern("Box<dyn Any>");
                        let pblk = self.panic_block(msg);
                        self.seal_terminator(IrTerminator::Jump(pblk));
                    }
                }
                Ok(())
            }
            Builtin::PanicAssertFailed => {
                // The converter appends the caller-location
                // stamp to this `#[track_caller]` builtin (the
                // interpreter's registry counts it among the
                // location consumers); the stamp is not one of
                // the four declared arguments and the host
                // report does not use it, so a trailing fifth
                // argument is dropped, as the other panic entry
                // points drop theirs.
                let args = if args.len() == 5 { &args[..4] } else { args };
                if args.len() != 4 {
                    return Err(self.err("assert_failed with wrong arity"));
                }
                let Some(pointee_ty) = generic_args.first() else {
                    return Err(self.err("assert_failed without a compared type"));
                };
                let pointee_ty = pointee_ty.clone();
                let desc = self.intern_descriptor(FmtTrait::Debug, &pointee_ty)?;
                let kind = self.arg_word(&args[0], "assert_failed kind")?;
                // The operands go to the host function as `fmt`
                // argument handles over the `Debug` descriptor --
                // the arguments a `format_args!` over the operands
                // would build.
                let mut handles = Vec::new();
                for arg in &args[1..3] {
                    let arg_ty = self.operand_place_ty(arg).ok_or_else(|| {
                        self.err("assert_failed operand without a type")
                    })?;
                    let words = flatten(&arg_ty, &self.func.name, self.layouts)?;
                    let (ptr, second, fat) = match words {
                        1 => (
                            self.operand_value(arg, &arg_ty)?,
                            IrOperand::Const(0),
                            0,
                        ),
                        2 => {
                            let (a, b) = self.operand_pair(arg)?;
                            (a, IrOperand::Value(b), 1)
                        }
                        _ => {
                            return Err(
                                self.err("assert_failed operand of an unsupported shape")
                            )
                        }
                    };
                    let handle = self.fresh();
                    self.emit_host_call(
                        builtins::HOST_ARGUMENT_NEW,
                        handle,
                        None,
                        vec![
                            IrOperand::Value(ptr),
                            second,
                            IrOperand::Const(desc as i64),
                            IrOperand::Const(fat),
                        ],
                    );
                    self.emit(Instr::CheckPanic);
                    handles.push(handle);
                }
                // The custom message: an `Option<fmt::Arguments>`.
                // The payload's own layout is the one-word handle
                // shape, so the option flattens either to the
                // discriminant plus the handle, or -- under the
                // niche layout -- to the handle alone (0 for
                // `None`, as 0 is never a valid handle). The host
                // function takes the discriminant and the handle,
                // with -1 for the handle in the niche shape (the
                // handle is then the discriminant word itself).
                let opt_ty = self.operand_place_ty(&args[3]).ok_or_else(|| {
                    self.err("assert_failed message argument without a type")
                })?;
                let opt_count = flatten(&opt_ty, &self.func.name, self.layouts)?;
                let leaves = self.operand_leaves(&args[3], &opt_ty, opt_count)?;
                let (opt_disc, opt_args) = match leaves.as_slice() {
                    [disc, handle] => (IrOperand::Value(*disc), IrOperand::Value(*handle)),
                    [word] => (IrOperand::Value(*word), IrOperand::Const(-1)),
                    _ => {
                        return Err(self.err(
                            "assert_failed message option of an unsupported shape",
                        ))
                    }
                };
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 0 {
                    return Err(self.err("assert_failed destination is not never"));
                }
                self.emit_host_call(
                    builtins::HOST_PANIC_ASSERT_FAILED,
                    dst,
                    None,
                    vec![
                        IrOperand::Value(kind),
                        IrOperand::Value(handles[0]),
                        IrOperand::Value(handles[1]),
                        opt_disc,
                        opt_args,
                    ],
                );
                self.finish_builtin_call(&dst_loc, dst, target, true);
                Ok(())
            }
            Builtin::StringNew => {
                if !args.is_empty() {
                    return Err(self.err("String::new with arguments"));
                }
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 3 {
                    return Err(self.err("String::new destination is not three words"));
                }
                // The empty string: the dangling pointer of an
                // empty buffer (the alignment of `u8`), zero
                // capacity, zero length.
                self.emit(Instr::Const { dst, imm: 1 });
                self.emit(Instr::Const {
                    dst: dst + 1,
                    imm: 0,
                });
                self.emit(Instr::Const {
                    dst: dst + 2,
                    imm: 0,
                });
                self.finish_builtin_call(&dst_loc, dst, target, false);
                Ok(())
            }
            Builtin::StringLen | Builtin::StringCapacity => {
                if args.len() != 1 {
                    return Err(self.err("String accessor with wrong arity"));
                }
                let addr = self.arg_word(&args[0], "String accessor argument")?;
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 1 {
                    return Err(self.err("String accessor destination is not one word"));
                }
                let host = if kind == Builtin::StringLen {
                    builtins::HOST_STRING_LEN
                } else {
                    builtins::HOST_STRING_CAPACITY
                };
                self.emit_host_call(host, dst, None, vec![IrOperand::Value(addr)]);
                self.finish_builtin_call(&dst_loc, dst, target, false);
                Ok(())
            }
            Builtin::StringIsEmpty => {
                if args.len() != 1 {
                    return Err(self.err("String::is_empty with wrong arity"));
                }
                let addr = self.arg_word(&args[0], "String::is_empty argument")?;
                let len = self.fresh();
                self.emit_host_call(
                    builtins::HOST_STRING_LEN,
                    len,
                    None,
                    vec![IrOperand::Value(addr)],
                );
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 1 {
                    return Err(self.err("String::is_empty destination is not one word"));
                }
                self.emit(Instr::BinOp {
                    dst,
                    op: IrBinOp::Eq,
                    lhs: IrOperand::Value(len),
                    rhs: IrOperand::Const(0),
                });
                self.finish_builtin_call(&dst_loc, dst, target, false);
                Ok(())
            }
            Builtin::StrLen | Builtin::StrIsEmpty => {
                if args.len() != 1 {
                    return Err(self.err("str accessor with wrong arity"));
                }
                let (_ptr, len) = self.operand_pair(&args[0])?;
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 1 {
                    return Err(self.err("str accessor destination is not one word"));
                }
                if kind == Builtin::StrLen {
                    self.emit(Instr::Copy {
                        dst,
                        src: IrOperand::Value(len),
                    });
                } else {
                    self.emit(Instr::BinOp {
                        dst,
                        op: IrBinOp::Eq,
                        lhs: IrOperand::Value(len),
                        rhs: IrOperand::Const(0),
                    });
                }
                self.finish_builtin_call(&dst_loc, dst, target, false);
                Ok(())
            }
            Builtin::StringPushStr | Builtin::StringPush => {
                if args.len() != 2 {
                    return Err(self.err("String push with wrong arity"));
                }
                let addr = self.arg_word(&args[0], "String push argument")?;
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 0 {
                    return Err(self.err("String push destination is not unit"));
                }
                if kind == Builtin::StringPushStr {
                    let (ptr, len) = self.operand_pair(&args[1])?;
                    self.emit_host_call(
                        builtins::HOST_STRING_PUSH_STR,
                        dst,
                        None,
                        vec![
                            IrOperand::Value(addr),
                            IrOperand::Value(ptr),
                            IrOperand::Value(len),
                        ],
                    );
                } else {
                    let ch = self.operand_value(&args[1], &mir::Ty::Char)?;
                    self.emit_host_call(
                        builtins::HOST_STRING_PUSH,
                        dst,
                        None,
                        vec![IrOperand::Value(addr), IrOperand::Value(ch)],
                    );
                }
                self.finish_builtin_call(&dst_loc, dst, target, true);
                Ok(())
            }
            Builtin::StrOp(op) => {
                if args.len() != 2 {
                    return Err(self.err("str search with wrong arity"));
                }
                let (ptr, len) = self.operand_pair(&args[0])?;
                // The modelled pattern is a `&str` (the
                // interpreter's builtin serves the pattern types
                // by value shape; a one-word pattern here — a
                // `char` or a predicate — stays unsupported).
                if let Some(pat_ty) = self.operand_place_ty(&args[1]) {
                    if flatten(&pat_ty, &self.func.name, self.layouts)? != 2 {
                        return Err(self.err("str search with a non-str pattern"));
                    }
                }
                let (pp, pl) = self.operand_pair(&args[1])?;
                let (dst_loc, dst, count) = self.builtin_destination(destination)?;
                if count != 1 {
                    return Err(self.err("str search destination is not one word"));
                }
                self.emit_host_call(
                    builtins::HOST_STR_OP,
                    dst,
                    None,
                    vec![
                        IrOperand::Const(op as i64),
                        IrOperand::Value(ptr),
                        IrOperand::Value(len),
                        IrOperand::Value(pp),
                        IrOperand::Value(pl),
                    ],
                );
                self.finish_builtin_call(&dst_loc, dst, target, true);
                Ok(())
            }
        }
    }

    /// The address of word `off` of the value whose word 0 sits
    /// at `base` (word `k` lies `8 * k` bytes below the base, the
    /// slot model's addressing).
    fn word_addr(&mut self, base: u32, off: usize) -> u32 {
        if off == 0 {
            return base;
        }
        let d = self.fresh();
        self.emit(Instr::BinOp {
            dst: d,
            op: IrBinOp::Sub,
            lhs: IrOperand::Value(base),
            rhs: IrOperand::Const((off * 8) as i64),
        });
        d
    }

    /// The array-iterator shape of the instantiation a `next`
    /// call's self type names, when the program constructs that
    /// iterator (see [`AdtLayouts::array_iter`]).
    fn array_iter_shape_of(
        &self,
        func: &mir::Operand,
    ) -> Option<Result<ArrayIterShape, String>> {
        let mir::Operand::Constant(c) = func else {
            return None;
        };
        let mir::ConstKind::Ty(mir::Ty::FnDef(_, gargs)) = &c.literal else {
            return None;
        };
        let mir::Ty::Adt(def, args) = gargs.first()? else {
            return None;
        };
        self.layouts.array_iter(*def, args)
    }

    /// The array-iterator shape recorded for the instantiation
    /// a call destination's declared type names, when the
    /// destination is a plain local of that type.
    fn array_iter_shape_of_destination(
        &self,
        destination: &mir::Place,
    ) -> Option<Result<ArrayIterShape, String>> {
        if !destination.projection.is_empty() {
            return None;
        }
        let decl = self.body.local_decls.get(destination.local.index())?;
        let mir::Ty::Adt(def, args) = &decl.ty else {
            return None;
        };
        self.layouts.array_iter(*def, args)
    }

    /// `IntoIterator::into_iter` on an array, lowered inline:
    /// the destination takes the iterator's layout shape (see
    /// [`ArrayIterShape`]) — this call's array leaves copied in
    /// order, then the `alive` range words `0` and this call's
    /// own `N`, at the layout's offsets (a shorter array than
    /// the layout's leaves the element area's tail unused).
    fn lower_array_into_iter(
        &mut self,
        shape: &ArrayIterShape,
        self_ty: &mir::Ty,
        args: &[mir::Operand],
        destination: &mir::Place,
        target: Option<mir::BasicBlock>,
    ) -> Result<(), JitError> {
        let mir::Ty::Array(_, mir::ArrayLen::Known(n)) = self_ty else {
            return Err(self.err("array into_iter of an array of unknown length"));
        };
        let n = *n as usize;
        if args.len() != 1 {
            return Err(self.err("array into_iter with an unexpected argument count"));
        }
        let data = shape.data();
        let total = shape.total();
        let dst_loc = self.classify_place(destination)?;
        match &dst_loc {
            PlaceLoc::Slots { first, count, .. } => {
                if *count != total {
                    return Err(self.err("array iterator destination shape mismatch"));
                }
                let first = *first;
                self.store_operand_leaves(&args[0], self_ty, first)?;
                self.emit(Instr::Const {
                    dst: first + data as u32,
                    imm: 0,
                });
                self.emit(Instr::Const {
                    dst: first + data as u32 + 1,
                    imm: n as i64,
                });
            }
            PlaceLoc::Mem { count, .. } => {
                if *count != Some(total) {
                    return Err(self.err("array iterator destination shape mismatch"));
                }
                let base = self.fresh();
                for _ in 1..total {
                    self.fresh();
                }
                self.store_operand_leaves(&args[0], self_ty, base)?;
                self.emit(Instr::Const {
                    dst: base + data as u32,
                    imm: 0,
                });
                self.emit(Instr::Const {
                    dst: base + data as u32 + 1,
                    imm: n as i64,
                });
                self.store_mem_from(&dst_loc, base);
            }
        }
        match target {
            Some(t) => self.seal_terminator(IrTerminator::Jump(t.index() as u32)),
            None => self.seal_terminator(IrTerminator::Unreachable),
        }
        Ok(())
    }

    /// `Iterator::next` on an array by-value iterator, lowered
    /// inline: the iterator is reached through the `&mut`
    /// argument's address. While the `alive` range is non-empty
    /// the call reads the element at `start`, advances `start`,
    /// and writes `Some(element)` to the destination; otherwise
    /// it writes `None` (tag and zeroed payload). Neither outcome
    /// can panic.
    fn lower_array_iter_next(
        &mut self,
        shape: &ArrayIterShape,
        args: &[mir::Operand],
        destination: &mir::Place,
        target: Option<mir::BasicBlock>,
    ) -> Result<(), JitError> {
        if args.len() != 1 {
            return Err(self.err("array iterator next with an unexpected argument count"));
        }
        let dst_loc = self.classify_place(destination)?;
        let (dst_ty, dst_count) = match &dst_loc {
            PlaceLoc::Slots { count, ty, .. } => (ty.clone(), *count),
            PlaceLoc::Mem { count, ty, .. } => (
                ty.clone(),
                count.ok_or_else(|| {
                    self.err("array iterator next destination is an unsized place")
                })?,
            ),
        };
        let layout = self.adt_layout(&dst_ty)?;
        if !layout.tag || dst_count != layout.total {
            return Err(self.err("array iterator next destination has no Option shape"));
        }
        let mir::Ty::Adt(option_def, _) = &dst_ty else {
            return Err(self.err("array iterator next destination is not an enum"));
        };
        let option_def = *option_def;
        // The `Some` variant is the one with a single field (the
        // element); `None` is the one with none.
        let mut some_v = None;
        let mut none_v = None;
        for (v, fields) in layout.variants.iter().enumerate() {
            match fields.len() {
                1 => some_v = Some(v as u32),
                0 => none_v = Some(v as u32),
                _ => {}
            }
        }
        let (Some(some_v), Some(none_v)) = (some_v, none_v) else {
            return Err(self.err("array iterator next destination has no Option shape"));
        };
        let slot = layout
            .field(some_v, 0)
            .ok_or_else(|| self.err("array iterator next destination has no Option shape"))?;
        let (payload_off, payload_width) = (slot.offset, slot.width);
        if payload_width != shape.per_elem {
            return Err(self.err("array iterator element shape mismatch"));
        }
        // The destination's leaves: the place's own slots, or a
        // scratch run stored through after the join (the same
        // arrangement as a call destination in memory).
        let dst_first = match &dst_loc {
            PlaceLoc::Slots { first, .. } => *first,
            PlaceLoc::Mem { .. } => {
                let base = self.fresh();
                for _ in 1..dst_count {
                    self.fresh();
                }
                base
            }
        };
        let arg_ty = self
            .operand_place_ty(&args[0])
            .unwrap_or(mir::Ty::Uint(mir::UintTy::U64));
        let ptr = self.operand_value(&args[0], &arg_ty)?;
        let data = shape.data();
        let start_addr = self.word_addr(ptr, data);
        let start = self.fresh();
        self.emit(Instr::LoadPtr {
            dst: start,
            ptr: IrOperand::Value(start_addr),
        });
        let end_addr = self.word_addr(ptr, data + 1);
        let end = self.fresh();
        self.emit(Instr::LoadPtr {
            dst: end,
            ptr: IrOperand::Value(end_addr),
        });
        let cond = self.fresh();
        self.emit(Instr::BinOp {
            dst: cond,
            op: IrBinOp::LtU,
            lhs: IrOperand::Value(start),
            rhs: IrOperand::Value(end),
        });
        let some_blk = self.new_block();
        let none_blk = self.new_block();
        let join_blk = self.new_block();
        self.seal_and_continue(
            IrTerminator::Branch {
                cond: IrOperand::Value(cond),
                then_blk: some_blk,
                else_blk: none_blk,
            },
            some_blk,
        );
        // Some: read the element at `start` (its first leaf is
        // word `start * per_elem` of the value), advance `start`,
        // and write the tag and the payload leaves.
        let w0 = self.fresh();
        self.emit(Instr::BinOp {
            dst: w0,
            op: IrBinOp::Mul,
            lhs: IrOperand::Value(start),
            rhs: IrOperand::Const(shape.per_elem as i64),
        });
        let mut leaves = Vec::with_capacity(shape.per_elem);
        for j in 0..shape.per_elem {
            let w = if j == 0 {
                w0
            } else {
                let d = self.fresh();
                self.emit(Instr::BinOp {
                    dst: d,
                    op: IrBinOp::Add,
                    lhs: IrOperand::Value(w0),
                    rhs: IrOperand::Const(j as i64),
                });
                d
            };
            let byte = self.fresh();
            self.emit(Instr::BinOp {
                dst: byte,
                op: IrBinOp::Mul,
                lhs: IrOperand::Value(w),
                rhs: IrOperand::Const(8),
            });
            let a = self.fresh();
            self.emit(Instr::BinOp {
                dst: a,
                op: IrBinOp::Sub,
                lhs: IrOperand::Value(ptr),
                rhs: IrOperand::Value(byte),
            });
            let leaf = self.fresh();
            self.emit(Instr::LoadPtr {
                dst: leaf,
                ptr: IrOperand::Value(a),
            });
            leaves.push(leaf);
        }
        let bumped = self.fresh();
        self.emit(Instr::BinOp {
            dst: bumped,
            op: IrBinOp::Add,
            lhs: IrOperand::Value(start),
            rhs: IrOperand::Const(1),
        });
        self.emit(Instr::StorePtr {
            ptr: IrOperand::Value(start_addr),
            src: IrOperand::Value(bumped),
        });
        let some_tag = self.discriminant_value(option_def, some_v)?;
        self.emit(Instr::Const {
            dst: dst_first,
            imm: some_tag,
        });
        for (j, leaf) in leaves.iter().enumerate() {
            self.emit(Instr::Copy {
                dst: dst_first + (payload_off + j) as u32,
                src: IrOperand::Value(*leaf),
            });
        }
        self.seal_and_continue(IrTerminator::Jump(join_blk), none_blk);
        // None: the tag, and zeroed payload leaves.
        let none_tag = self.discriminant_value(option_def, none_v)?;
        self.emit(Instr::Const {
            dst: dst_first,
            imm: none_tag,
        });
        for j in 0..payload_width {
            self.emit(Instr::Const {
                dst: dst_first + (payload_off + j) as u32,
                imm: 0,
            });
        }
        self.seal_and_continue(IrTerminator::Jump(join_blk), join_blk);
        if let PlaceLoc::Mem { .. } = &dst_loc {
            self.store_mem_from(&dst_loc, dst_first);
        }
        match target {
            Some(t) => self.seal_terminator(IrTerminator::Jump(t.index() as u32)),
            None => self.seal_terminator(IrTerminator::Unreachable),
        }
        Ok(())
    }

    fn lower_call(
        &mut self,
        func: &mir::Operand,
        args: &[mir::Operand],
        destination: &mir::Place,
        target: Option<mir::BasicBlock>,
    ) -> Result<(), JitError> {
        // Resolve the callee: only direct calls to functions of this
        // program (a FnDef constant naming a known DefId).
        let callee = match func {
            mir::Operand::Constant(c) => match &c.literal {
                mir::ConstKind::Ty(mir::Ty::FnDef(def, gargs)) => {
                    Some((*def, gargs.to_vec()))
                }
                mir::ConstKind::Unevaluated { def, args: gargs } => {
                    Some((*def, gargs.to_vec()))
                }
                _ => None,
            },
            _ => None,
        };
        let Some((def, generic_args)) = callee else {
            return Err(self.err("call through a non-constant function value"));
        };
        // The array by-value iterator's two entry points are
        // never part of the program (the driver leaves them to
        // the interpreter's builtins); the JIT lowers them
        // inline. Any other absent callee keeps the path below.
        if !self.def_index.contains_key(&def) {
            if let Some(name) = self.def_names.get(&def) {
                if is_array_into_iter_fn(name) {
                    if let mir::Operand::Constant(c) = func {
                        if let mir::ConstKind::Ty(mir::Ty::FnDef(_, gargs)) = &c.literal {
                            if let Some(self_ty) = gargs.first() {
                                if matches!(self_ty, mir::Ty::Array(..)) {
                                    if let Some(shape) =
                                        self.array_iter_shape_of_destination(destination)
                                    {
                                        let shape =
                                            shape.map_err(|reason| self.err(&reason))?;
                                        return self.lower_array_into_iter(
                                            &shape,
                                            &self_ty.clone(),
                                            args,
                                            destination,
                                            target,
                                        );
                                    }
                                }
                            }
                        }
                    }
                } else if is_array_iter_next_fn(name) {
                    if let Some(shape) = self.array_iter_shape_of(func) {
                        let shape = shape.map_err(|reason| self.err(&reason))?;
                        return self.lower_array_iter_next(
                            &shape,
                            args,
                            destination,
                            target,
                        );
                    }
                }
            }
        }
        // A callee the program does not define is the builtin
        // layer's business: the interpreter resolves exactly these
        // calls by name, and so does the JIT (see
        // `lower_builtin_call`). A name neither layer knows stays
        // the same explicit error as before.
        let Some(&callee) = self.def_index.get(&def) else {
            return self.lower_builtin_call(&generic_args, def, args, destination, target);
        };
        let callee_tys = &self.param_tys[callee];
        // A `#[track_caller]` callee arrives with one extra
        // trailing argument: the caller-location stamp the
        // driver appended to the call (the interpreter pops it
        // before binding the declared arguments, the same way).
        // The callee's body declares no local for the stamp, so
        // it is simply dropped here.
        let args = if args.len() == callee_tys.len() + 1 {
            &args[..callee_tys.len()]
        } else {
            args
        };
        if args.len() != callee_tys.len() {
            return Err(self.err("call argument count does not match the callee"));
        }
        // The destination first: its leaf count is the callee's
        // return leaf count, which decides the call's shape. A
        // result of more than two words does not travel in
        // registers: the callee takes a hidden leading argument
        // word holding the address of the storage it stores the
        // result into (see `FnLower::run`) — the destination's own
        // slots for a frame destination, a scratch run for a memory
        // one.
        //
        // A call whose destination is in memory (`*p = f()`)
        // receives into the scratch run and stores through the
        // address after the panic check, so a panicking callee
        // leaves the pointee untouched (and the destination's
        // index is read after the call, as the interpreter reads
        // it when the call returns).
        let dst_loc = self.classify_place(destination)?;
        let (dst, count) = match &dst_loc {
            PlaceLoc::Mem { count, .. } => {
                let count = match count {
                    Some(c) => *c,
                    None => return Err(self.err("call destination is an unsized place")),
                };
                // The result lands in a consecutive scratch run;
                // `store_mem_from` stores it through the address
                // after the panic check.
                let first = self.fresh();
                let mut prev = first;
                for _ in 1..count.max(1) {
                    let next = self.fresh();
                    debug_assert_eq!(next, prev + 1);
                    prev = next;
                }
                (first, count)
            }
            PlaceLoc::Slots { .. } => {
                let (dst, count, _) = self.place_leaves(destination)?;
                (dst, count)
            }
        };
        let sret = count > 2;
        // Lay the argument words out by the shared contract
        // (`arg_word_layout`), the hidden sret word first when the
        // callee takes one: a unit argument contributes a dummy
        // zero word, and a pad zero word precedes a pair that would
        // straddle the register/stack boundary. Words past the
        // sixth travel on the stack; the backend call handles them.
        let mut counts = Vec::with_capacity(callee_tys.len());
        for pty in callee_tys.iter() {
            counts.push(flatten(pty, &self.func.name, self.layouts)?);
        }
        let mut layout_counts = Vec::with_capacity(counts.len() + 1);
        if sret {
            layout_counts.push(1);
        }
        layout_counts.extend_from_slice(&counts);
        let (starts, _) = arg_word_layout(&layout_counts);
        let offset = usize::from(sret);
        let mut ir_args: Vec<IrOperand> = Vec::new();
        if sret {
            let addr = self.fresh();
            self.emit(Instr::SlotAddr { dst: addr, slot: dst });
            ir_args.push(IrOperand::Value(addr));
        }
        for (i, (arg, pty)) in args.iter().zip(callee_tys.iter()).enumerate() {
            while ir_args.len() < starts[i + offset] {
                ir_args.push(IrOperand::Const(0)); // the pad word
            }
            match counts[i] {
                0 => ir_args.push(IrOperand::Const(0)), // unit: the dummy word
                1 => {
                    let v = self.operand_value(arg, pty)?;
                    ir_args.push(IrOperand::Value(v));
                }
                2 => {
                    if int128_signed(pty).is_some() {
                        // A 128-bit argument is the value's own
                        // leaf pair (a constant included), not a
                        // fat pointer.
                        let leaves = self.operand_leaves(arg, pty, 2)?;
                        ir_args.push(IrOperand::Value(leaves[0]));
                        ir_args.push(IrOperand::Value(leaves[1]));
                    } else {
                        // Any other two-word argument — a fat
                        // pointer, but equally a two-word
                        // aggregate (a `Layout` constant passed
                        // to an allocator function): its own two
                        // leaves, materialised the same way for
                        // places and constants.
                        let leaves = self.operand_leaves(arg, pty, 2)?;
                        ir_args.push(IrOperand::Value(leaves[0]));
                        ir_args.push(IrOperand::Value(leaves[1]));
                    }
                }
                n => {
                    let leaves = self.operand_leaves(arg, pty, n)?;
                    for leaf in leaves {
                        ir_args.push(IrOperand::Value(leaf));
                    }
                }
            }
        }
        let (dst, dst2) = if sret {
            // The result is already in the sret storage; the call's
            // nominal result word is a dummy.
            (self.fresh(), None)
        } else {
            match count {
                0 => (self.fresh(), None),
                1 => (dst, None),
                _ => (dst, Some(dst + 1)),
            }
        };
        self.emit(Instr::Call {
            dst,
            dst2,
            func: callee as u32,
            args: ir_args,
        });
        self.emit(Instr::CheckPanic);
        if let PlaceLoc::Mem { .. } = &dst_loc {
            self.store_mem_from(&dst_loc, dst);
        }
        match target {
            Some(t) => self.seal_terminator(IrTerminator::Jump(t.index() as u32)),
            None => self.seal_terminator(IrTerminator::Unreachable),
        }
        Ok(())
    }
}
