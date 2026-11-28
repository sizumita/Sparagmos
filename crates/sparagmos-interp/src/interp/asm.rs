//! Inline-asm register program execution.

use super::*;
use crate::error::{InterpError, InterpResult};

use crate::value::{RefTarget, Value};

impl<'a> Interpreter<'a> {
    /// Execute one evaluable inline-asm block: seed a 64-bit slot
    /// per operand from its input value, run the register program
    /// with x86-64 partial-register semantics, and write back the
    /// operands the program wrote. Inout operands always write
    /// back; a pure output the program never wrote keeps its
    /// destination untouched — its value is unspecified, and
    /// leaving the place alone keeps a later read an explicit
    /// uninitialized error rather than a fabricated zero.
    /// A naked block entered through an asm `call` step instead
    /// starts from the calling block's register file (see
    /// `Frame::incoming_asm`), and a naked block leaves its
    /// outgoing register file behind for that calling block
    /// (see `Interpreter::pending_asm_outgoing`).
    pub(crate) fn exec_inline_asm(
        &mut self,
        frames: &mut Vec<Frame>,
        body: &Body,
        regs: &[AsmReg],
        steps: &[AsmStep],
        ret: Option<AsmRet>,
        arg_seeds: &[AsmArgSeed],
        naked_shape: bool,
    ) -> InterpResult<AsmOutcome> {
        let mut slots: Vec<u64> = Vec::with_capacity(regs.len());
        // The callee names of function-pointer operands, by slot:
        // a function pointer's slot carries no bits (its value is
        // a name in the interpreter's value model), and a `call`
        // step through the register resolves the name here.
        let mut fn_ptrs: Vec<Option<String>> = Vec::with_capacity(regs.len());
        for reg in regs {
            let mut fn_ptr = None;
            let bits = match &reg.init {
                Some(op) => {
                    let value = self.eval_operand(frames, body, op)?;
                    match &value {
                        Value::FnPtr(name) => {
                            fn_ptr = Some(name.clone());
                            0
                        }
                        _ => asm_input_bits(&value, reg)?,
                    }
                }
                None => 0,
            };
            slots.push(bits);
            fn_ptrs.push(fn_ptr);
        }
        // The architectural register file: the 16 general-purpose
        // registers followed by 32 vector registers (only the low
        // 64 bits of each vector register are modelled — the part
        // `movd` / `movq` transfer). An explicit-register operand
        // aliases the register it names: it is seeded from its
        // input value, and the step executor keeps the register
        // and the operand's slot in step as the program runs.
        let mut arch = [0u64; 48];
        let mut arch_written = [false; 48];
        // A naked block entered through an asm `call` step starts
        // from the calling block's register file: the `call`
        // preserves the state the caller established, including
        // registers no argument seeding supplies.
        let mut forwarded_written: Option<[bool; 48]> = None;
        if naked_shape && body.blocks.len() == 1 {
            if let Some((incoming, incoming_written)) =
                frames.last().and_then(|frame| frame.incoming_asm)
            {
                arch = incoming;
                arch_written = incoming_written;
                forwarded_written = Some(incoming_written);
            }
        }
        // A naked block's incoming state: the enclosing function's
        // arguments, which `naked_asm!` receives through the
        // function's ABI rather than through operands, are seeded
        // into the registers the ABI passes them in. (A naked
        // block has no operands, so nothing below overwrites the
        // seedings; for other blocks the list is empty.) A
        // register the forwarded state actually established keeps
        // that raw content: the frame's locals hold the argument
        // narrowed to its declared scalar (the value the call
        // assembled), while the body reads the register itself,
        // whose bits above the scalar's width are the caller's
        // own and observable. Seeding therefore supplies only the
        // registers the forwarded state did not establish, and
        // every register when there is no forwarded state (a call
        // from Rust code, where the locals are the only source).
        for seed in arg_seeds {
            if forwarded_written.is_some_and(|w| w[seed.arch as usize]) {
                continue;
            }
            let value = frames
                .last()
                .expect("frame")
                .locals
                .get(seed.local as usize)
                .cloned()
                .unwrap_or(Value::Uninit);
            arch[seed.arch as usize] = asm_scalar_bits(&value, &seed.scalar)?;
            arch_written[seed.arch as usize] = true;
        }
        let mut bound: [Option<usize>; 48] = [None; 48];
        for (i, reg) in regs.iter().enumerate() {
            if let Some(a) = reg.arch {
                bound[a as usize] = Some(i);
                if reg.init.is_some() {
                    arch[a as usize] = slots[i];
                    arch_written[a as usize] = true;
                }
            }
        }
        // An output operand bound to a register the incoming
        // state already initialized — seeded by an aliased input
        // operand above, or by a naked block's argument seeds —
        // starts from that state: a step that reads the operand
        // observes the register's content, and the operand's
        // value is defined even if no step writes the register
        // (the `lateout` shape: with `in("rax") 42` and
        // `lateout("rax") y`, y holds 42 when the program never
        // writes rax).
        for (i, reg) in regs.iter().enumerate() {
            if reg.init.is_none() {
                if let Some(a) = reg.arch {
                    if arch_written[a as usize] {
                        slots[i] = arch[a as usize];
                    }
                }
            }
        }
        // The seeded register file, snapshotted before the
        // steps run: a naked block's outgoing state (below) is
        // measured against it, so only registers the program
        // itself changed are reported as established by the
        // body. The seedings above restate values the caller
        // already placed — an argument register's seeding may
        // even narrow what the caller established — and those
        // are the caller's state to keep, not the callee's to
        // impose on the way out.
        let seeded_arch = arch;
        let seeded_written = arch_written;
        let mut written = vec![false; regs.len()];
        // Slots whose current content a `lea` of a static's
        // address established, by slot: their write-back is the
        // static reference itself, not the priced bits (a raw
        // pointer place holds a reference value in this model,
        // and the bits alone do not type-check there).
        let mut sym_addrs: Vec<Option<String>> = vec![None; regs.len()];
        let mut flags = AsmFlags::default();
        let mut outcome = AsmOutcome::Fallthrough;
        for step in steps {
            match step {
                AsmStep::Jump { target } => {
                    outcome = AsmOutcome::JumpTo(BasicBlock(target.index()));
                    break;
                }
                AsmStep::Call { sig } => {
                    sym_addrs.iter_mut().for_each(|a| *a = None);
                    self.exec_asm_call(frames, &mut arch, &mut arch_written, sig)?;
                    // The call may have rewritten any register:
                    // resync the slots of bound operands with the
                    // register file's post-call state.
                    for (a, slot) in bound.iter().enumerate() {
                        if let Some(i) = slot {
                            slots[*i] = arch[a];
                            written[*i] = true;
                        }
                    }
                }
                AsmStep::CallReg { idx, sig } => {
                    sym_addrs.iter_mut().for_each(|a| *a = None);
                    // The callee is the operand's runtime value:
                    // a function pointer the seeding recorded by
                    // name. Anything else in the slot is not a
                    // callable the register model can name.
                    let Some(name) = fn_ptrs.get(*idx).cloned().flatten() else {
                        return Err(InterpError::UnsupportedTerminator {
                            kind: "InlineAsm (call through a register whose value is not a function pointer)"
                                .to_string(),
                        });
                    };
                    let mut sig = sig.clone();
                    sig.name = name;
                    self.exec_asm_call(frames, &mut arch, &mut arch_written, &sig)?;
                    for (a, slot) in bound.iter().enumerate() {
                        if let Some(i) = slot {
                            slots[*i] = arch[a];
                            written[*i] = true;
                        }
                    }
                }
                AsmStep::LeaSym { dst, static_name } => {
                    // The static's address under the same pricing
                    // a reference to the static carries, so the
                    // materialized pointer compares equal to one.
                    // The step names the static by its display
                    // name; the program's static key adds a
                    // disambiguating suffix when several
                    // definitions share it, so resolve the name
                    // against the program's static tables first.
                    let Some(static_key) = self.resolve_asm_static_key(static_name) else {
                        return Err(InterpError::UnsupportedTerminator {
                            kind: format!(
                                "InlineAsm (lea of the address of static `{static_name}`, which the program does not record)"
                            ),
                        });
                    };
                    // Seed the static's storage first: a reference
                    // the program never materialized through a
                    // constant has no store entry yet, and the
                    // write-back below hands out a reference to it.
                    self.ensure_static_seeded(&static_key)?;
                    let addr = self.static_price(&static_key) as u64;
                    match dst {
                        sparagmos_mir::AsmDst::Operand(idx, _) => {
                            slots[*idx] = addr;
                            written[*idx] = true;
                            sym_addrs[*idx] = Some(static_key.clone());
                            asm_sync_slot_to_arch(
                                &slots,
                                &mut arch,
                                &mut arch_written,
                                regs,
                                *idx,
                            );
                        }
                        sparagmos_mir::AsmDst::Arch(reg) => {
                            asm_arch_write(&mut arch, *reg, addr);
                            arch_written[reg.index as usize] = true;
                            asm_sync_arch_to_slot(
                                &arch,
                                &mut slots,
                                &mut written,
                                &bound,
                                *reg,
                            );
                        }
                    }
                }
                step => {
                    // A straight-line step that writes a slot (or
                    // a bound architectural register) supersedes
                    // any `lea`-of-static provenance that slot
                    // carried: its content is the step's result.
                    let dst_slot = match step {
                        AsmStep::Mov { dst, .. }
                        | AsmStep::Bin { dst, .. }
                        | AsmStep::Un { dst, .. } => Some(*dst),
                        AsmStep::SetCc {
                            dst: sparagmos_mir::AsmDst::Operand(idx, _),
                            ..
                        }
                        | AsmStep::Lea {
                            dst: sparagmos_mir::AsmDst::Operand(idx, _),
                            ..
                        } => Some(*idx),
                        AsmStep::MovArch { dst, .. }
                        | AsmStep::BinArch { dst, .. }
                        | AsmStep::UnArch { dst, .. } => bound[dst.index as usize],
                        AsmStep::SetCc {
                            dst: sparagmos_mir::AsmDst::Arch(reg),
                            ..
                        }
                        | AsmStep::Lea {
                            dst: sparagmos_mir::AsmDst::Arch(reg),
                            ..
                        } => bound[reg.index as usize],
                        _ => None,
                    };
                    if let Some(idx) = dst_slot {
                        sym_addrs[idx] = None;
                    }
                    exec_asm_step(
                    &mut slots,
                    &mut written,
                    &mut arch,
                    &mut arch_written,
                    &bound,
                        regs,
                        &mut flags,
                        step,
                    )?;
                }
            }
        }
        for (i, reg) in regs.iter().enumerate() {
            let Some(dest) = &reg.dest else { continue };
            // An explicit-register operand's value is the final
            // content of its register: the register file is kept
            // in step with the operand slots as the program runs,
            // and it also carries incoming state no step touched —
            // an aliased input's seed, which an output bound to
            // the same register observes. A register that was
            // never seeded or written leaves the destination
            // untouched, like an unwritten slot output below.
            let bits = match reg.arch {
                Some(a) if arch_written[a as usize] => arch[a as usize],
                _ => {
                    if !written[i] && reg.init.is_none() {
                        continue;
                    }
                    slots[i]
                }
            };
            let value = match &sym_addrs[i] {
                Some(key) => Value::Ref(RefTarget::Static {
                    name: key.clone(),
                    projection: Vec::new(),
                }),
                None => asm_output_value(bits, reg),
            };
            self.write_place(frames, body, dest, value)?;
        }
        if matches!(outcome, AsmOutcome::Fallthrough) {
            if let Some(shape) = ret {
            // A naked block: the return value is whatever the
            // program left in `rax`, read at the function's
            // return shape.
            let shape_reg = AsmReg {
                init: None,
                dest: None,
                width: shape.width,
                kind: shape.kind,
                arch: None,
            };
            let value = asm_output_value(arch[0], &shape_reg);
            frames.last_mut().expect("frame").locals[0] = value;
            }
        }
        if naked_shape {
            // A naked body leaves its register file behind for
            // the asm `call` step that entered it (see
            // `Interpreter::pending_asm_outgoing`): the calling
            // block continues with the architectural state this
            // body actually established. Only registers the
            // steps changed relative to the seeded file are
            // marked — a register the body never touched keeps
            // the calling block's own state.
            let mut out_written = [false; 48];
            for r in 0..48 {
                out_written[r] =
                    arch_written[r] && (!seeded_written[r] || arch[r] != seeded_arch[r]);
            }
            self.pending_asm_outgoing = Some((arch, out_written));
        }
        Ok(outcome)
    }

    /// Resolve an asm `lea` step's static display name to the
    /// program's static key: the key is the name itself unless
    /// conversion disambiguated a duplicated name with a
    /// `#DefId(..)` suffix, in which case the unique recorded
    /// key with that base name (among the seed and type tables)
    /// is the static the operand denotes. `None` when the
    /// program records no such static, or the name is ambiguous
    /// between several recorded keys.
    fn resolve_asm_static_key(&self, name: &str) -> Option<String> {
        let program = self.program();
        if program.static_seeds.contains_key(name) || program.static_tys.contains_key(name) {
            return Some(name.to_string());
        }
        let mut found: Option<&String> = None;
        for key in program
            .static_seeds
            .keys()
            .chain(program.static_tys.keys())
        {
            if key.split('#').next() == Some(name) {
                if found.is_some_and(|f| f != key) {
                    return None;
                }
                found = Some(key);
            }
        }
        found.cloned()
    }

    /// Execute one asm `call` step: assemble the arguments from
    /// the architectural register file per the signature's ABI,
    /// run the callee to completion, and scatter its return
    /// value back into the file. The registers the ABI lets the
    /// callee clobber are zeroed afterwards: their contents are
    /// unspecified, and zero keeps a later read from observing a
    /// stale value as if it were live. A naked callee instead
    /// hands back its outgoing register state (see
    /// `Interpreter::pending_asm_outgoing`), which replaces the
    /// zeroing: the registers its body changed take the values
    /// it left in them, and the registers it never touched keep
    /// this block's own state. A panic in the callee
    /// propagates: the enclosing frame's unwind table routes it
    /// exactly like a panic raised at a call site in the asm
    /// block.
    fn exec_asm_call(
        &mut self,
        frames: &mut Vec<Frame>,
        arch: &mut [u64; 48],
        arch_written: &mut [bool; 48],
        sig: &sparagmos_mir::AsmCallSig,
    ) -> InterpResult<()> {
        // Architectural indices: rax rcx rdx rbx rsp rbp rsi rdi
        // r8..r15 are 0..16; xmm0.. are 16...
        const SYSV_INT: [usize; 6] = [7, 6, 2, 1, 8, 9];
        const WIN_INT: [usize; 4] = [1, 2, 8, 9];
        let mut args: Vec<Value> = Vec::with_capacity(sig.params.len());
        let (mut next_int, mut next_float) = (0usize, 0usize);
        for (pos, param) in sig.params.iter().enumerate() {
            let bits = match (sig.abi, param) {
                (sparagmos_mir::AsmAbi::SysV64, sparagmos_mir::AsmScalar::Float { .. }) => {
                    let bits = arch[16 + next_float];
                    next_float += 1;
                    bits
                }
                (sparagmos_mir::AsmAbi::SysV64, _) => {
                    let bits = arch[SYSV_INT[next_int]];
                    next_int += 1;
                    bits
                }
                (sparagmos_mir::AsmAbi::Win64, sparagmos_mir::AsmScalar::Float { .. }) => {
                    arch[16 + pos]
                }
                (sparagmos_mir::AsmAbi::Win64, _) => arch[WIN_INT[pos]],
            };
            args.push(asm_scalar_value(bits, param));
        }
        // A naked callee's register file starts from this
        // block's architectural state: a native `call` preserves
        // every register, and the block may have established
        // values in registers the callee's argument seeding does
        // not supply (an explicit `in("r10")` the callee's body
        // reads). Stash the state for the frame the call creates
        // and restore the previous value afterwards, so nested
        // asm calls compose. Non-naked callees run their own
        // code before any asm block of theirs, so for them the
        // state is not forwarded.
        let callee_is_naked = self.program.get(&sig.name).is_some_and(|b| {
            b.blocks.len() == 1
                && matches!(
                    &b.blocks[0].terminator,
                    TerminatorKind::InlineAsm { target: None, .. }
                )
        });
        let saved_incoming = if callee_is_naked {
            self.pending_asm_incoming.replace((*arch, *arch_written))
        } else {
            None
        };
        // Likewise, clear any outgoing state an earlier naked
        // body left behind, so the state taken after the call
        // is this callee's alone; the saved value is restored
        // afterwards, so nested asm calls compose.
        let saved_outgoing = if callee_is_naked {
            self.pending_asm_outgoing.take()
        } else {
            None
        };
        let result = self.call_sync(frames, &sig.name, args);
        if callee_is_naked {
            self.pending_asm_incoming = saved_incoming;
        }
        let outgoing = if callee_is_naked {
            let taken = self.pending_asm_outgoing.take();
            self.pending_asm_outgoing = saved_outgoing;
            taken
        } else {
            None
        };
        let result = result?;
        let mut ret_reg: Option<usize> = None;
        if let Some((out_arch, out_written)) = outgoing {
            // A naked callee's outgoing architectural state
            // replaces the clobber model below: the registers
            // its steps changed take the values the body left
            // in them, and the registers it never touched keep
            // this block's own state — a native `call` preserves
            // whatever the callee does not overwrite, and
            // zeroing the clobber set would fabricate values
            // for registers the body demonstrably wrote. The
            // return-value scatter below still runs afterwards
            // for a return register the body did not establish;
            // one it did establish keeps its raw bits (see the
            // scatter).
            for r in 0..48 {
                if out_written[r] {
                    arch[r] = out_arch[r];
                    arch_written[r] = true;
                }
            }
        }
        if let Some(ret) = &sig.ret {
            let idx = match ret {
                sparagmos_mir::AsmScalar::Float { .. } => 16,
                _ => 0,
            };
            // A naked callee's outgoing state, when available,
            // is the return register's content: the merge above
            // has already placed the bits the body established,
            // and the registers it never touched keep this
            // block's own state — a native `call` to a naked body
            // preserves whatever the body does not overwrite,
            // including a bare-`ret` body, whose return register
            // still holds what the caller placed in it. An output
            // of the calling block bound to that register observes
            // the register itself, whose content may be wider
            // than the declared return scalar (a `u8` callee's
            // `rax` read back into a `u64` output). The narrowed
            // Rust value is the call's result, not the register's
            // content, so the scatter applies only when there is
            // no outgoing state (a non-naked callee).
            if outgoing.is_none() {
                arch[idx] = asm_scalar_bits(&result, ret)?;
                // The call established the return register's
                // content: mark it written, so a naked body that
                // returns through this block hands the register
                // on in its outgoing state (which is measured
                // against `arch_written`), and an output operand
                // bound to the register observes the result.
                arch_written[idx] = true;
            }
            ret_reg = Some(idx);
        }
        if outgoing.is_none() {
            let (gp_clobbers, xmm_clobbers): (&[usize], std::ops::Range<usize>) = match sig.abi {
                sparagmos_mir::AsmAbi::SysV64 => (&[0, 1, 2, 6, 7, 8, 9, 10, 11], 16..48),
                sparagmos_mir::AsmAbi::Win64 => (&[0, 1, 2, 8, 9, 10, 11], 16..22),
            };
            for &r in gp_clobbers {
                if Some(r) != ret_reg {
                    arch[r] = 0;
                }
            }
            for i in xmm_clobbers {
                if Some(i) != ret_reg {
                    arch[i] = 0;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Inline-asm register program execution (the evaluable class of
// sparagmos-mir's `inline_asm_program`): slots are 64-bit, reads
// and writes follow the x86-64 partial-register rules.
// ---------------------------------------------------------------------------

/// The bit mask of a `width`-bit operand.
fn asm_width_mask(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// Sign-extend the low `width` bits of `bits` to 64 bits.
fn asm_sign_extend(bits: u64, width: u32) -> u64 {
    if width >= 64 {
        bits
    } else {
        let shift = 64 - width;
        ((bits << shift) as i64 >> shift) as u64
    }
}

/// How an executed asm block leaves: through its continuation
/// or through a label jump.
pub(crate) enum AsmOutcome {
    Fallthrough,
    JumpTo(BasicBlock),
}

/// The interpreter value of one asm call argument: the register
/// bits interpreted at the parameter's scalar shape.
fn asm_scalar_value(bits: u64, scalar: &sparagmos_mir::AsmScalar) -> Value {
    match scalar {
        sparagmos_mir::AsmScalar::Int { width, signed } => {
            let masked = if *width >= 64 {
                bits
            } else {
                bits & ((1u64 << width) - 1)
            };
            if *signed {
                let shift = 64 - width;
                Value::Int(((masked << shift) as i64 >> shift) as i128)
            } else {
                Value::Uint(u128::from(masked))
            }
        }
        // A boolean argument occupies one byte of its register:
        // the callee reads only the low byte, and the upper bits
        // are unspecified incoming state. Testing the whole
        // register would turn a false whose upper bits happen to
        // be nonzero into true.
        sparagmos_mir::AsmScalar::Bool => Value::Bool(bits & 0xff != 0),
        sparagmos_mir::AsmScalar::Float { width } => {
            if *width == 32 {
                Value::Float(f64::from(f32::from_bits(bits as u32)))
            } else {
                Value::Float(f64::from_bits(bits))
            }
        }
    }
}

/// The register bits of an asm call's return value.
fn asm_scalar_bits(
    value: &Value,
    scalar: &sparagmos_mir::AsmScalar,
) -> InterpResult<u64> {
    Ok(match (value, scalar) {
        (Value::Int(v), _) => *v as u64,
        (Value::Uint(v), _) => *v as u64,
        (Value::Bool(b), _) => u64::from(*b),
        (Value::Float(f), sparagmos_mir::AsmScalar::Float { width: 32 }) => {
            f32::to_bits(*f as f32) as u64
        }
        (Value::Float(f), sparagmos_mir::AsmScalar::Float { .. }) => f.to_bits(),
        (other, _) => {
            return Err(InterpError::type_mismatch(
                "inline asm call return",
                "integer, bool or float",
                other.kind_name(),
            ));
        }
    })
}

/// The register-file index of the stack pointer (the layout
/// `ArchReg::index` documents: 0..16 are rax, rcx, rdx, rbx,
/// rsp, rbp, rsi, rdi, r8..=r15).
const ARCH_RSP: usize = 4;

/// Reject a read of the incoming stack pointer. A read of
/// `rsp` before the program itself has written it observes the
/// *incoming* value — the one the caller's frame layout fixes
/// (natively it points at the return address the call pushed).
/// The register model does not represent that state: the file
/// starts at zero and no seed supplies it, since arguments
/// never arrive in `rsp`. Returning the fabricated zero would
/// silently compute with a wrong stack pointer, so the read is
/// a named unsupported until the incoming stack state is
/// modelled. A program that establishes its own `rsp` first
/// (recorded in `arch_written`) reads that value normally.
fn check_incoming_rsp_read(
    arch_written: &[bool; 48],
    reg: sparagmos_mir::ArchReg,
) -> InterpResult<()> {
    if reg.index as usize == ARCH_RSP && !arch_written[ARCH_RSP] {
        return Err(InterpError::UnsupportedTerminator {
            kind: "InlineAsm (read of the incoming stack pointer `rsp`, which is not modelled)"
                .to_string(),
        });
    }
    Ok(())
}

/// Read an architectural register at its access width (`h` reads
/// bits 8..16).
fn asm_arch_read(arch: &[u64; 48], reg: sparagmos_mir::ArchReg) -> u64 {
    let bits = arch[reg.index as usize];
    match (reg.width, reg.high) {
        (32, _) => bits & 0xffff_ffff,
        (16, _) => bits & 0xffff,
        (8, true) => (bits >> 8) & 0xff,
        (8, false) => bits & 0xff,
        _ => bits,
    }
}

/// Write an architectural register with x86-64 partial-register
/// semantics: a 32-bit write zero-extends, 8/16-bit writes
/// preserve the rest, an `h` write lands in bits 8..16.
fn asm_arch_write(arch: &mut [u64; 48], reg: sparagmos_mir::ArchReg, value: u64) {
    let slot = &mut arch[reg.index as usize];
    match (reg.width, reg.high) {
        (32, _) => *slot = value & 0xffff_ffff,
        (16, _) => *slot = (*slot & !0xffff) | (value & 0xffff),
        (8, true) => *slot = (*slot & !0xff00) | ((value & 0xff) << 8),
        (8, false) => *slot = (*slot & !0xff) | (value & 0xff),
        _ => *slot = value,
    }
}

/// Keep the register file in step with a slot write: an operand
/// bound to an architectural register writes through to it.
fn asm_sync_slot_to_arch(
    slots: &[u64],
    arch: &mut [u64; 48],
    arch_written: &mut [bool; 48],
    regs: &[AsmReg],
    idx: usize,
) {
    if let Some(a) = regs[idx].arch {
        arch[a as usize] = slots[idx];
        arch_written[a as usize] = true;
    }
}

/// Keep a bound operand's slot in step with a register write.
fn asm_sync_arch_to_slot(
    arch: &[u64; 48],
    slots: &mut [u64],
    written: &mut [bool],
    bound: &[Option<usize>; 48],
    reg: sparagmos_mir::ArchReg,
) {
    if let Some(i) = bound[reg.index as usize] {
        slots[i] = arch[reg.index as usize];
        written[i] = true;
    }
}

/// The bits an asm input value enters its slot with: the value
/// zero-extended from the operand width. (The bits above the
/// width are unspecified on hardware — a wider register form
/// read there observes whatever the caller left; the model
/// fixes them to zero.)
fn asm_input_bits(value: &Value, reg: &AsmReg) -> InterpResult<u64> {
    let bits = match value {
        Value::Int(v) => *v as u64,
        Value::Uint(v) => *v as u64,
        Value::Bool(b) => u64::from(*b),
        other => {
            return Err(InterpError::type_mismatch(
                "inline asm operand",
                "integer or bool",
                other.kind_name(),
            ));
        }
    };
    Ok(bits & asm_width_mask(reg.width))
}

/// The value written back to an asm output place: the slot bits
/// interpreted at the operand type.
fn asm_output_value(bits: u64, reg: &AsmReg) -> Value {
    let masked = bits & asm_width_mask(reg.width);
    match reg.kind {
        // The sign-extended bits are the value's two's-complement
        // form: reinterpret them as a signed 64-bit integer before
        // widening to i128. Casting the u64 straight to i128
        // would zero-extend, turning -1 into 2^64 - 1 — a wrong
        // result wherever the value is consumed without a
        // further type normalization (a naked function's return
        // place is assigned directly, with no `write_place`).
        AsmRegKind::Signed => {
            Value::Int((asm_sign_extend(masked, reg.width) as i64) as i128)
        }
        AsmRegKind::Unsigned => Value::Uint(u128::from(masked)),
        AsmRegKind::Bool => Value::Bool(masked != 0),
    }
}

/// Read a register slot at an instruction operand: the modifier
/// picks the sub-register (`h` is bits 8..16); no modifier reads
/// the whole register — the full 64-bit register for the `reg`
/// class (a placeholder without a modifier names it, whatever
/// the operand width), the byte for the `reg_byte` class.
fn asm_read(slots: &[u64], regs: &[AsmReg], idx: usize, modifier: Option<char>) -> u64 {
    let slot = slots[idx];
    match modifier {
        Some('h') => (slot >> 8) & 0xff,
        Some('l') => slot & 0xff,
        Some('x') => slot & 0xffff,
        Some('e') => slot & 0xffff_ffff,
        _ if regs[idx].width == 8 => slot & 0xff,
        _ => slot,
    }
}

/// Write a register slot at an instruction operand, with x86-64
/// partial-register semantics: a 32-bit write zero-extends,
/// 8/16-bit writes preserve the rest of the slot, an `h` write
/// lands in bits 8..16, and a modifier-less write replaces the
/// whole register (the byte for the `reg_byte` class).
fn asm_write(
    slots: &mut [u64],
    written: &mut [bool],
    regs: &[AsmReg],
    idx: usize,
    modifier: Option<char>,
    value: u64,
) {
    written[idx] = true;
    let slot = &mut slots[idx];
    match modifier {
        Some('h') => *slot = (*slot & !0xff00) | ((value & 0xff) << 8),
        Some('l') => *slot = (*slot & !0xff) | (value & 0xff),
        Some('x') => *slot = (*slot & !0xffff) | (value & 0xffff),
        Some('e') => *slot = value & 0xffff_ffff,
        _ if regs[idx].width == 8 => *slot = (*slot & !0xff) | (value & 0xff),
        _ => *slot = value,
    }
}

/// The width an instruction operates at: the destination
/// modifier's width, else the full register (64 bits, or 8 for
/// the `reg_byte` class) — a modifier-less placeholder names the
/// whole register, so e.g. a shift count is masked at 64 bits,
/// not at the operand width.
fn asm_instr_width(regs: &[AsmReg], idx: usize, modifier: Option<char>) -> u32 {
    match modifier {
        Some('e') => 32,
        Some('x') => 16,
        Some('l') | Some('h') => 8,
        _ if regs[idx].width == 8 => 8,
        _ => 64,
    }
}

/// The value of an instruction source operand. An architectural
/// source is a register read: reading the incoming stack
/// pointer is rejected (see `check_incoming_rsp_read`).
fn asm_src_value(
    slots: &[u64],
    regs: &[AsmReg],
    arch: &[u64; 48],
    arch_written: &[bool; 48],
    src: &sparagmos_mir::AsmSrc,
) -> InterpResult<u64> {
    match src {
        sparagmos_mir::AsmSrc::Reg(idx, modifier) => {
            Ok(asm_read(slots, regs, *idx, *modifier))
        }
        sparagmos_mir::AsmSrc::Imm(bits) => Ok(*bits),
        sparagmos_mir::AsmSrc::Arch(reg) => {
            check_incoming_rsp_read(arch_written, *reg)?;
            Ok(asm_arch_read(arch, *reg))
        }
    }
}

/// Run one classified asm instruction against the slots and
/// the architectural register file. Jump and call steps are the
/// caller's (they transfer control / run a callee); every other
/// step is straight-line.
#[allow(clippy::too_many_arguments)]
fn exec_asm_step(
    slots: &mut [u64],
    written: &mut [bool],
    arch: &mut [u64; 48],
    arch_written: &mut [bool; 48],
    bound: &[Option<usize>; 48],
    regs: &[AsmReg],
    flags: &mut AsmFlags,
    step: &AsmStep,
) -> InterpResult<()> {
    use sparagmos_mir::AsmDst;
    match step {
        AsmStep::Nop
        | AsmStep::Ret
        | AsmStep::Jump { .. }
        | AsmStep::Call { .. }
        | AsmStep::CallReg { .. }
        | AsmStep::LeaSym { .. } => {}
        AsmStep::Test { left, src } => {
            let width = asm_src_width(slots, regs, arch, left);
            let mask = asm_width_mask(width);
            let a = asm_src_value(slots, regs, arch, arch_written, left)? & mask;
            let b = asm_src_value(slots, regs, arch, arch_written, src)? & mask;
            *flags = AsmFlags::from_logic(a & b, width);
        }
        AsmStep::Cmp { left, src } => {
            let width = asm_src_width(slots, regs, arch, left);
            let mask = asm_width_mask(width);
            let a = asm_src_value(slots, regs, arch, arch_written, left)? & mask;
            let b = asm_src_value(slots, regs, arch, arch_written, src)? & mask;
            *flags = AsmFlags::from_sub(a, b, width);
        }
        AsmStep::SetCc { cond, dst } => {
            let byte = u64::from(flags.test(*cond));
            match dst {
                AsmDst::Operand(idx, _) => {
                    // Only the low byte is written; the rest of
                    // the slot is preserved.
                    slots[*idx] = (slots[*idx] & !0xff) | byte;
                    written[*idx] = true;
                    asm_sync_slot_to_arch(slots, arch, arch_written, regs, *idx);
                }
                AsmDst::Arch(reg) => {
                    let low = sparagmos_mir::ArchReg {
                        width: 8,
                        high: reg.high,
                        ..*reg
                    };
                    asm_arch_write(arch, low, byte);
                    arch_written[reg.index as usize] = true;
                    asm_sync_arch_to_slot(arch, slots, written, bound, *reg);
                }
            }
        }
        AsmStep::Mov { dst, dst_mod, src } => {
            let value = asm_src_value(slots, regs, arch, arch_written, src)?;
            asm_write(slots, written, regs, *dst, *dst_mod, value);
            asm_sync_slot_to_arch(slots, arch, arch_written, regs, *dst);
        }
        AsmStep::Bin {
            op,
            dst,
            dst_mod,
            src,
        } => {
            let width = asm_instr_width(regs, *dst, *dst_mod);
            let mask = asm_width_mask(width);
            let a = asm_read(slots, regs, *dst, *dst_mod) & mask;
            let b = asm_src_value(slots, regs, arch, arch_written, src)? & mask;
            let result = asm_bin_result(*op, a, b, width);
            flags.update_bin(*op, a, b, result & mask, width);
            asm_write(slots, written, regs, *dst, *dst_mod, result & mask);
            asm_sync_slot_to_arch(slots, arch, arch_written, regs, *dst);
        }
        AsmStep::Un { op, dst, dst_mod } => {
            let width = asm_instr_width(regs, *dst, *dst_mod);
            let mask = asm_width_mask(width);
            let a = asm_read(slots, regs, *dst, *dst_mod) & mask;
            let result = asm_un_result(*op, a);
            flags.update_un(*op, a, result & mask, width);
            asm_write(slots, written, regs, *dst, *dst_mod, result & mask);
            asm_sync_slot_to_arch(slots, arch, arch_written, regs, *dst);
        }
        AsmStep::MovArch { dst, src } => {
            let value = asm_src_value(slots, regs, arch, arch_written, src)?;
            asm_arch_write(arch, *dst, value);
            arch_written[dst.index as usize] = true;
            asm_sync_arch_to_slot(arch, slots, written, bound, *dst);
        }
        AsmStep::BinArch { op, dst, src } => {
            let width = dst.width;
            let mask = asm_width_mask(width);
            check_incoming_rsp_read(arch_written, *dst)?;
            let a = asm_arch_read(arch, *dst) & mask;
            let b = asm_src_value(slots, regs, arch, arch_written, src)? & mask;
            let result = asm_bin_result(*op, a, b, width);
            flags.update_bin(*op, a, b, result & mask, width);
            asm_arch_write(arch, *dst, result & mask);
            arch_written[dst.index as usize] = true;
            asm_sync_arch_to_slot(arch, slots, written, bound, *dst);
        }
        AsmStep::UnArch { op, dst } => {
            let width = dst.width;
            let mask = asm_width_mask(width);
            check_incoming_rsp_read(arch_written, *dst)?;
            let a = asm_arch_read(arch, *dst) & mask;
            let result = asm_un_result(*op, a);
            flags.update_un(*op, a, result & mask, width);
            asm_arch_write(arch, *dst, result & mask);
            arch_written[dst.index as usize] = true;
            asm_sync_arch_to_slot(arch, slots, written, bound, *dst);
        }
        AsmStep::Lea { dst, base, disp } => {
            // The effective address wraps at the address size —
            // the width of the base register form — before the
            // destination write extends it: `lea rax, [ecx + 1]`
            // with ecx at 2^32 - 1 yields zero, not 2^32.
            let addr_width = match base {
                sparagmos_mir::AsmSrc::Reg(idx, modifier) => {
                    asm_instr_width(regs, *idx, *modifier)
                }
                sparagmos_mir::AsmSrc::Arch(reg) => reg.width,
                sparagmos_mir::AsmSrc::Imm(_) => 64,
            };
            let addr = asm_src_value(slots, regs, arch, arch_written, base)?
                .wrapping_add(*disp as u64)
                & asm_width_mask(addr_width);
            match dst {
                AsmDst::Operand(idx, modifier) => {
                    asm_write(slots, written, regs, *idx, *modifier, addr);
                    asm_sync_slot_to_arch(slots, arch, arch_written, regs, *idx);
                }
                AsmDst::Arch(reg) => {
                    asm_arch_write(arch, *reg, addr);
                    arch_written[reg.index as usize] = true;
                    asm_sync_arch_to_slot(arch, slots, written, bound, *reg);
                }
            }
        }
    }
    Ok(())
}

/// The flag state an asm program's `test` / `cmp` / arithmetic
/// steps leave for a later `setcc`: the five status flags the
/// condition codes read. A block starts with all flags clear;
/// a step that does not define flags (moves, shifts, `lea`)
/// leaves the state alone, as on hardware.
#[derive(Clone, Copy, Default)]
struct AsmFlags {
    zf: bool,
    sf: bool,
    cf: bool,
    of: bool,
    pf: bool,
}

impl AsmFlags {
    /// Flags of a logic result (`test`, `and`, `or`, `xor`):
    /// CF and OF clear, the rest from the result.
    fn from_logic(result: u64, width: u32) -> Self {
        let mask = asm_width_mask(width);
        let r = result & mask;
        Self {
            zf: r == 0,
            sf: width > 0 && (r >> (width - 1)) & 1 == 1,
            cf: false,
            of: false,
            pf: (r & 0xff).count_ones() % 2 == 0,
        }
    }

    /// Flags of `a - b` at `width` bits (`cmp`, `sub`).
    fn from_sub(a: u64, b: u64, width: u32) -> Self {
        let mask = asm_width_mask(width);
        let a = a & mask;
        let b = b & mask;
        let r = a.wrapping_sub(b) & mask;
        let sign = |v: u64| width > 0 && (v >> (width - 1)) & 1 == 1;
        Self {
            zf: r == 0,
            sf: sign(r),
            cf: a < b,
            of: sign(a) != sign(b) && sign(r) != sign(a),
            pf: (r & 0xff).count_ones() % 2 == 0,
        }
    }

    /// Flags of `a + b` at `width` bits (`add`).
    fn from_add(a: u64, b: u64, width: u32) -> Self {
        let mask = asm_width_mask(width);
        let a = a & mask;
        let b = b & mask;
        let full = u128::from(a) + u128::from(b);
        let r = (full as u64) & mask;
        let sign = |v: u64| width > 0 && (v >> (width - 1)) & 1 == 1;
        Self {
            zf: r == 0,
            sf: sign(r),
            cf: width < 128 && (full >> width) & 1 == 1,
            of: sign(a) == sign(b) && sign(r) != sign(a),
            pf: (r & 0xff).count_ones() % 2 == 0,
        }
    }

    /// Refresh the flags after a binary step, for the operation
    /// shapes whose flag semantics are plain: shifts leave the
    /// state alone (their CF/OF are defined but no step in the
    /// class reads them, and approximating them would be worse
    /// than leaving the prior state).
    fn update_bin(&mut self, op: sparagmos_mir::AsmBinOp, a: u64, b: u64, result: u64, width: u32) {
        use sparagmos_mir::AsmBinOp;
        match op {
            AsmBinOp::Add => *self = Self::from_add(a, b, width),
            AsmBinOp::Sub => *self = Self::from_sub(a, b, width),
            AsmBinOp::And | AsmBinOp::Or | AsmBinOp::Xor => {
                *self = Self::from_logic(result, width)
            }
            AsmBinOp::Shl | AsmBinOp::Shr | AsmBinOp::Sar => {}
        }
    }

    /// Refresh the flags after a unary step: `inc` / `dec`
    /// preserve CF, `neg` sets CF from the operand, `not`
    /// defines no flags.
    fn update_un(&mut self, op: sparagmos_mir::AsmUnOp, a: u64, result: u64, width: u32) {
        use sparagmos_mir::AsmUnOp;
        match op {
            AsmUnOp::Inc => {
                let cf = self.cf;
                *self = Self::from_add(a, 1, width);
                self.cf = cf;
            }
            AsmUnOp::Dec => {
                let cf = self.cf;
                *self = Self::from_sub(a, 1, width);
                self.cf = cf;
            }
            AsmUnOp::Neg => {
                let mut f = Self::from_sub(0, a, width);
                f.cf = a != 0;
                *self = f;
            }
            AsmUnOp::Not => {
                let _ = result;
            }
        }
    }

    /// Whether the condition holds over this flag state.
    fn test(&self, cond: sparagmos_mir::AsmCond) -> bool {
        use sparagmos_mir::AsmCond;
        match cond {
            AsmCond::E => self.zf,
            AsmCond::Ne => !self.zf,
            AsmCond::B => self.cf,
            AsmCond::Ae => !self.cf,
            AsmCond::L => self.sf != self.of,
            AsmCond::Ge => self.sf == self.of,
            AsmCond::Le => self.zf || self.sf != self.of,
            AsmCond::G => !self.zf && self.sf == self.of,
            AsmCond::A => !self.cf && !self.zf,
            AsmCond::Be => self.cf || self.zf,
            AsmCond::S => self.sf,
            AsmCond::Ns => !self.sf,
            AsmCond::O => self.of,
            AsmCond::No => !self.of,
            AsmCond::P => self.pf,
            AsmCond::Np => !self.pf,
        }
    }
}

/// The width a `test` / `cmp` left operand operates at: an
/// operand slot's instruction width, an architectural
/// register's access width, an immediate's full width (the
/// parser never admits an immediate on the left).
fn asm_src_width(
    _slots: &[u64],
    regs: &[AsmReg],
    _arch: &[u64; 48],
    src: &sparagmos_mir::AsmSrc,
) -> u32 {
    match src {
        sparagmos_mir::AsmSrc::Reg(idx, modifier) => asm_instr_width(regs, *idx, *modifier),
        sparagmos_mir::AsmSrc::Arch(reg) => reg.width,
        sparagmos_mir::AsmSrc::Imm(_) => 64,
    }
}

/// The result of one binary asm operation at `width` bits.
fn asm_bin_result(op: sparagmos_mir::AsmBinOp, a: u64, b: u64, width: u32) -> u64 {
    use sparagmos_mir::AsmBinOp;
    let mask = asm_width_mask(width);
    let a = a & mask;
    let b = b & mask;
    let count = (b & if width >= 64 { 63 } else { 31 }) as u32;
    match op {
        AsmBinOp::Add => a.wrapping_add(b),
        AsmBinOp::Sub => a.wrapping_sub(b),
        AsmBinOp::And => a & b,
        AsmBinOp::Or => a | b,
        AsmBinOp::Xor => a ^ b,
        AsmBinOp::Shl => a << count,
        AsmBinOp::Shr => a >> count,
        AsmBinOp::Sar => (asm_sign_extend(a, width) as i64 >> count) as u64,
    }
}

/// The result of one unary asm operation.
fn asm_un_result(op: sparagmos_mir::AsmUnOp, a: u64) -> u64 {
    use sparagmos_mir::AsmUnOp;
    match op {
        AsmUnOp::Neg => 0u64.wrapping_sub(a),
        AsmUnOp::Not => !a,
        AsmUnOp::Inc => a.wrapping_add(1),
        AsmUnOp::Dec => a.wrapping_sub(1),
    }
}
