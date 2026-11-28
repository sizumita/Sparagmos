//! Constant decoding from allocations: typed values, materialization plans, and erased/fat-pointer reads.

use super::*;

impl Converter {
    /// The typed constant value of a type that has exactly one
    /// value: unit, tuples and arrays of such types, structs all
    /// of whose fields are such types, and enums with a single
    /// inhabited variant likewise (a variant with an uninhabited
    /// field type — an empty enum, `!`, a struct containing one —
    /// can never be constructed, so `Option<Empty>` has the single
    /// value `None`). Types with no values or many values, and
    /// unions (whose active field the type does not determine),
    /// yield `None`.
    pub(crate) fn zst_const_value(
        &mut self,
        ty: &rustc_public::ty::Ty,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        match self.unique_const_value(ty, depth) {
            UniqueConst::One(v) => Some(v),
            _ => None,
        }
    }

    fn unique_const_value(
        &mut self,
        ty: &rustc_public::ty::Ty,
        depth: usize,
    ) -> UniqueConst {
        if depth > 32 {
            return UniqueConst::Many;
        }
        // A coroutine-closure type's kind cannot be asked (see
        // `guarded_ty_kind`); its value is the closure value
        // itself, which this walk does not construct, so the
        // type has no unique value here.
        let Some(kind) = guarded_ty_kind(ty) else {
            return UniqueConst::Many;
        };
        match kind {
            TyKind::RigidTy(RigidTy::Never) => UniqueConst::Uninhabited,
            TyKind::RigidTy(RigidTy::Tuple(elems)) => {
                let mut fields = Vec::with_capacity(elems.len());
                for e in &elems {
                    match self.unique_const_value(e, depth + 1) {
                        UniqueConst::One(v) => fields.push(v),
                        UniqueConst::Uninhabited => return UniqueConst::Uninhabited,
                        UniqueConst::Many => return UniqueConst::Many,
                    }
                }
                if fields.is_empty() {
                    UniqueConst::One(sm::ConstValue::ZeroSized)
                } else {
                    UniqueConst::One(sm::ConstValue::Tuple(fields))
                }
            }
            TyKind::RigidTy(RigidTy::Array(elem, len)) => {
                let Ok(n) = len.eval_target_usize() else {
                    return UniqueConst::Many;
                };
                if n == 0 {
                    return UniqueConst::One(sm::ConstValue::Tuple(Vec::new()));
                }
                if n > 4096 {
                    return UniqueConst::Many;
                }
                match self.unique_const_value(&elem, depth + 1) {
                    UniqueConst::One(v) => {
                        UniqueConst::One(sm::ConstValue::Tuple(vec![v; n as usize]))
                    }
                    UniqueConst::Uninhabited => UniqueConst::Uninhabited,
                    UniqueConst::Many => UniqueConst::Many,
                }
            }
            TyKind::RigidTy(RigidTy::Adt(def, args)) => {
                if def.kind() == AdtKind::Union {
                    return UniqueConst::Many;
                }
                let mut single: Option<(u32, Vec<sm::ConstValue>)> = None;
                for vi in 0..def.num_variants() {
                    let Some(variant) = def.variant(VariantIdx::to_val(vi)) else {
                        return UniqueConst::Many;
                    };
                    let mut fields = Vec::new();
                    let mut inhabited = true;
                    for field in variant.fields() {
                        let fty = field.ty_with_args(&args);
                        match self.unique_const_value(&fty, depth + 1) {
                            UniqueConst::One(v) => fields.push(v),
                            UniqueConst::Uninhabited => {
                                inhabited = false;
                                break;
                            }
                            UniqueConst::Many => return UniqueConst::Many,
                        }
                    }
                    if !inhabited {
                        continue;
                    }
                    if single.is_some() {
                        return UniqueConst::Many;
                    }
                    single = Some((vi as u32, fields));
                }
                match single {
                    Some((variant, fields)) => {
                        UniqueConst::One(sm::ConstValue::Adt { variant, fields })
                    }
                    None => UniqueConst::Uninhabited,
                }
            }
            _ => UniqueConst::Many,
        }
    }

    /// Interpret an allocation as a constant value, guided by its type.
    pub(crate) fn const_from_alloc(
        &mut self,
        ty: &rustc_public::ty::Ty,
        alloc: &Allocation,
    ) -> sm::ConstValue {
        // Every root constant starts with a full expansion
        // budget (see `MAX_CONST_TOTAL_ELEMS`). The aggregate
        // path below reaches a reset at the root of
        // `decode_const` / `decode_ref`, but the scalar and
        // reference paths do not: without a reset here, a
        // reference constant's fat-pointer charge would spend
        // whatever earlier constants in the same body left
        // behind, so whether a constant decodes would depend
        // on the constants visited before it.
        self.const_budget = MAX_CONST_TOTAL_ELEMS;
        self.const_scan_budget = MAX_CONST_SCAN_BYTES;
        self.const_work_budget = MAX_CONST_WORK_BYTES;
        self.const_work_failed = false;
        self.clear_alloc_caches();
        // The memoised decode of this exact constant, if a
        // previous mention already produced it (see the
        // `const_memo` field). The resets above have run, so
        // the converter state a hit leaves behind is the state
        // a fresh decode's entry would leave behind.
        let memo_key = if alloc.bytes.len() <= MAX_CONST_MEMO_BYTES {
            let mut ptrs = Vec::with_capacity(alloc.provenance.ptrs.len());
            let mut named = true;
            for (off, prov) in &alloc.provenance.ptrs {
                match prov_target_key(prov, 0) {
                    Some(k) => ptrs.push((*off, k)),
                    None => {
                        named = false;
                        break;
                    }
                }
            }
            named.then(|| (*ty, alloc.bytes.clone(), ptrs))
        } else {
            None
        };
        if let Some(key) = &memo_key {
            if let Some(v) = self.const_memo.get(key) {
                return (**v).clone();
            }
        }
        let errors_before = self.errors.len();
        // A constant whose type passes through a coroutine-
        // closure type cannot be asked its kind (see
        // `guarded_ty_kind`); the value is a zero-sized closure
        // value, or a composite containing one, and carries no
        // allocation bytes to decode.
        let Some(kind) = guarded_ty_kind(ty) else {
            return sm::ConstValue::ZeroSized;
        };
        let value = match kind {
            TyKind::RigidTy(RigidTy::Bool) => match alloc.bytes.as_slice() {
                [Some(0)] => sm::ConstValue::Bool(false),
                [Some(1)] => sm::ConstValue::Bool(true),
                // A byte outside {0, 1} is the product of a
                // transmute; its bits are the value, kept as a
                // scalar (the interpreter reads a scalar under a
                // bool context as `data != 0`, the meaning the
                // bits carry). A wholly uninitialised byte has
                // no bits to read and decodes as uninit, the
                // form the structural decode below already
                // gives an all-uninit range.
                [Some(b)] => sm::ConstValue::Scalar(sm::Scalar::Int {
                    data: *b as u128,
                    size: 1,
                }),
                [None] => sm::ConstValue::Uninit,
                _ => {
                    self.err("ConstantKind::Allocated (unreadable bool)");
                    sm::ConstValue::ZeroSized
                }
            },
            TyKind::RigidTy(RigidTy::Char) => match alloc.read_uint() {
                Ok(v) => match char::from_u32(v as u32) {
                    Some(ch) => sm::ConstValue::Char(ch),
                    // An invalid char is a transmuted value;
                    // keep its bits as a scalar, as the bool
                    // arm does (a scalar under a char context
                    // reads back as the same code unit value).
                    None => sm::ConstValue::Scalar(sm::Scalar::Int {
                        data: v,
                        size: alloc.bytes.len() as u8,
                    }),
                },
                Err(_) if alloc.bytes.iter().all(|b| b.is_none()) => {
                    sm::ConstValue::Uninit
                }
                Err(_) => {
                    self.err("ConstantKind::Allocated (unreadable char)");
                    sm::ConstValue::ZeroSized
                }
            },
            TyKind::RigidTy(RigidTy::Int(_)) => match alloc.read_int() {
                Ok(v) => sm::ConstValue::Scalar(sm::Scalar::Int {
                    data: v as u128,
                    size: alloc.bytes.len() as u8,
                }),
                Err(_) if alloc.bytes.iter().all(|b| b.is_none()) => {
                    sm::ConstValue::Uninit
                }
                Err(_) => {
                    self.err("ConstantKind::Allocated (unreadable int)");
                    sm::ConstValue::ZeroSized
                }
            },
            TyKind::RigidTy(RigidTy::Uint(_)) => match alloc.read_uint() {
                Ok(v) => sm::ConstValue::Scalar(sm::Scalar::Int {
                    data: v,
                    size: alloc.bytes.len() as u8,
                }),
                Err(_) if alloc.bytes.iter().all(|b| b.is_none()) => {
                    sm::ConstValue::Uninit
                }
                Err(_) => {
                    self.err("ConstantKind::Allocated (unreadable uint)");
                    sm::ConstValue::ZeroSized
                }
            },
            TyKind::RigidTy(RigidTy::Float(f)) => {
                let bits = match f {
                    SmirFloatTy::F16 => alloc
                        .read_partial_uint(0..2)
                        .map(|v| v as u16 as u128)
                        .unwrap_or(0),
                    SmirFloatTy::F32 => alloc
                        .read_partial_uint(0..4)
                        .map(|v| v as u32 as u128)
                        .unwrap_or(0),
                    SmirFloatTy::F64 => alloc
                        .read_partial_uint(0..8)
                        .map(|v| v as u128)
                        .unwrap_or(0),
                    SmirFloatTy::F128 => alloc
                        .read_partial_uint(0..16)
                        .map(|v| v as u128)
                        .unwrap_or(0),
                };
                sm::ConstValue::Scalar(sm::Scalar::Float {
                    bits,
                    ty: float_ty(f),
                })
            }
            TyKind::RigidTy(RigidTy::Str) => {
                // The copy below materialises the allocation's
                // bytes: charge them against the expansion
                // budget, like every other byte-copying decode.
                if !self.spend_const_budget(alloc.bytes.len()) {
                    self.err("ConstantKind::Allocated (str over the expansion budget)");
                    return sm::ConstValue::ZeroSized;
                }
                match alloc.raw_bytes() {
                    Ok(bytes) => match String::from_utf8(bytes) {
                        Ok(s) => sm::ConstValue::Str(s),
                        Err(_) => {
                            self.err("ConstantKind::Allocated (invalid UTF-8 str)");
                            sm::ConstValue::ZeroSized
                        }
                    },
                    Err(_) => {
                        self.err("ConstantKind::Allocated (unreadable str)");
                        sm::ConstValue::ZeroSized
                    }
                }
            }
            TyKind::RigidTy(RigidTy::Ref(_, inner, _)) => {
                self.ref_const(&inner, alloc)
            }
            TyKind::RigidTy(RigidTy::Tuple(elems)) if elems.is_empty() => {
                sm::ConstValue::ZeroSized
            }
            other => {
                let _ = &other;
                // Aggregate constants (tuples, arrays, ADTs) are
                // decoded field by field, guided by the layout.
                if let Some(v) = self.decode_const(ty, alloc, 0, 0) {
                    v
                } else {
                    self.err(format!(
                        "ConstantKind::Allocated (type {})",
                        short_ty(ty)
                    ));
                    sm::ConstValue::ZeroSized
                }
            }
        };
        // Record the decode for later mentions of the same
        // constant, but only when it is exactly reproducible:
        // a decode that recorded an error, or that a budget
        // failure shaped (a fallback taken because a charge
        // failed leaves the corresponding flag set), may have
        // depended on the state it ran in, so it is decoded
        // fresh on every mention, as before.
        if let Some(key) = memo_key {
            if self.errors.len() == errors_before
                && !self.const_budget_failed
                && !self.const_scan_failed
                && !self.const_work_failed
            {
                if self.const_memo.len() >= MAX_CONST_MEMO_ENTRIES {
                    self.const_memo.clear();
                }
                self.const_memo.insert(key, std::rc::Rc::new(value.clone()));
            }
        }
        value
    }

    /// Resolve the allocation provenance `prov` names. A
    /// memory allocation is served from `alloc_cache` when
    /// this decode tree has already materialised it, and is
    /// recorded there when it fits the cache's byte cap;
    /// anything else comes back as the bare `GlobalAlloc` for
    /// the caller to match on. A materialisation is charged
    /// against the materialisation-work budget (see
    /// `MAX_CONST_WORK_BYTES`); when the charge does not fit,
    /// the resolution is `Refused` and the decode that needed
    /// it fails.
    /// Drop both backing-allocation caches and reset their shared
    /// byte accounting. Called wherever the constant-expansion
    /// budget resets (the root of every decoded constant), so cached
    /// materialisations live for one root decode.
    pub(crate) fn clear_alloc_caches(&mut self) {
        self.alloc_cache.clear();
        self.static_inits.clear();
        self.alloc_cache_bytes = 0;
    }

    /// The initialiser allocation of the static `def`, evaluated at
    /// most once per root decode: `eval_initializer` materialises
    /// the whole allocation on every call, so resolving each element
    /// of a large table of references into one static through it
    /// would copy the backing once per element. Each evaluation is
    /// charged against the materialisation-work budget (see
    /// `MAX_CONST_WORK_BYTES`): an evaluation repeated because the
    /// cache evicted the initialiser is not free, so alternating
    /// references into statics that together exceed the cache cap
    /// stay bounded in cumulative work.
    pub(crate) fn static_initializer(
        &mut self,
        def: &mir::mono::StaticDef,
    ) -> Option<std::rc::Rc<Allocation>> {
        let key = self.crate_def_id(def);
        if let Some(alloc) = self.static_inits.get(&key) {
            return Some(alloc.clone());
        }
        // Once this root constant's materialisation work is
        // over budget, further evaluations are refused without
        // being performed (see `const_work_failed`).
        if self.const_work_failed {
            return None;
        }
        let alloc = std::rc::Rc::new(def.eval_initializer().ok()?);
        let len = alloc.bytes.len();
        if !self.spend_work_budget(len) {
            return None;
        }
        if self.alloc_cache_bytes + len > MAX_CACHED_ALLOC_BYTES {
            self.clear_alloc_caches();
        }
        if len <= MAX_CACHED_ALLOC_BYTES {
            self.alloc_cache_bytes += len;
            self.static_inits.insert(key, alloc.clone());
        }
        Some(alloc)
    }

    pub(crate) fn resolve_alloc(&mut self, prov: &Prov) -> ResolvedAlloc {
        let key = prov.0.to_index();
        if let Some(alloc) = self.alloc_cache.get(&key) {
            return ResolvedAlloc::Memory(alloc.clone());
        }
        // Once this root constant's materialisation work is
        // over budget, further materialisations are refused
        // without being performed (see `const_work_failed`).
        if self.const_work_failed {
            return ResolvedAlloc::Refused;
        }
        match mir::alloc::GlobalAlloc::from(prov.0) {
            mir::alloc::GlobalAlloc::Memory(alloc) => {
                let size = alloc.bytes.len();
                if !self.spend_work_budget(size) {
                    return ResolvedAlloc::Refused;
                }
                let alloc = std::rc::Rc::new(alloc);
                if size <= MAX_CACHED_ALLOC_BYTES {
                    if self.alloc_cache_bytes + size > MAX_CACHED_ALLOC_BYTES {
                        // The byte counter covers both caches, so
                        // resetting it must drop both: clearing only
                        // this map would leave cached static
                        // initialisers retained but unaccounted, and
                        // they could then accumulate far past the cap
                        // across one root decode.
                        self.clear_alloc_caches();
                    }
                    self.alloc_cache_bytes += size;
                    self.alloc_cache.insert(key, alloc.clone());
                }
                ResolvedAlloc::Memory(alloc)
            }
            other => ResolvedAlloc::Other(other),
        }
    }

    /// The memory allocation provenance `prov` names, if it
    /// names one (see `resolve_alloc`).
    fn memory_alloc(&mut self, prov: &Prov) -> Option<std::rc::Rc<Allocation>> {
        match self.resolve_alloc(prov) {
            ResolvedAlloc::Memory(alloc) => Some(alloc),
            ResolvedAlloc::Other(_) => None,
            ResolvedAlloc::Refused => None,
        }
    }

    /// Read a fat-pointer allocation: provenance points at the data
    /// allocation, the second word is the length metadata.
    fn read_fat_pointer(&mut self, alloc: &Allocation) -> FatPointerRead {
        let Some((_, prov)) = alloc.provenance.ptrs.first() else {
            return FatPointerRead::NotFat;
        };
        let Some(target) = self.memory_alloc(prov) else {
            return FatPointerRead::NotFat;
        };
        let Ok(meta) = alloc.read_partial_uint(8..16) else {
            return FatPointerRead::NotFat;
        };
        let meta = meta as u64;
        // The data is exactly the metadata's byte count, starting
        // at the offset the data pointer itself stores: the
        // backing allocation may extend past the denoted range on
        // both sides (the file path allocation behind a `Location`
        // constant carries a trailing NUL byte, and a subslice or
        // substring constant minted in a const context points into
        // the middle of its backing allocation), so read only the
        // denoted range rather than copying the whole backing
        // allocation or its prefix.
        let offset = read_uint_at(alloc, 0, PTR_SIZE).unwrap_or(0) as usize;
        let Some(range) = offset
            .checked_add(meta as usize)
            .map(|end| offset..end)
        else {
            return FatPointerRead::NotFat;
        };
        let Some(data) = alloc_bytes_in(&target, range) else {
            return FatPointerRead::NotFat;
        };
        if !self.spend_const_budget(data.len()) {
            return FatPointerRead::OverBudget;
        }
        FatPointerRead::Data(data, meta)
    }

    /// Constants of reference type. The allocation of a reference
    /// constant holds the pointer itself, with provenance identifying
    /// the pointee allocation:
    /// - fat pointers (`&str`, `&[T]`): pointer + length word;
    /// - thin pointers (`&[i32; 3]`, `&&str` pieces, `&Location`): a
    ///   single provenance entry.
    /// Pointees made of plain scalar data are materialized; anything
    /// else becomes `ConstValue::ByRef` keyed by the Stable MIR
    /// allocation id, which preserves the constant's identity exactly
    /// (the runtime resolves the id against the allocation table).
    fn ref_const(
        &mut self,
        inner: &rustc_public::ty::Ty,
        alloc: &Allocation,
    ) -> sm::ConstValue {
        // The pointee kind, asked once: a type passing through
        // a coroutine-closure type cannot be asked at all (see
        // `guarded_ty_kind`), and none of the materialisation
        // shortcuts below apply to it.
        let inner_kind = guarded_ty_kind(inner);
        // Fat-pointer shape: `&str` and `&[u8]` literals take the
        // shortcut; every other shape falls through to the decoder
        // below (a raw-byte slice of wider or non-scalar elements
        // has no value form in the interpreter).
        match self.read_fat_pointer(alloc) {
            FatPointerRead::Data(data, meta) => match &inner_kind {
                Some(TyKind::RigidTy(RigidTy::Str)) => {
                    return match String::from_utf8(data) {
                        Ok(s) => sm::ConstValue::Str(s),
                        Err(_) => {
                            self.err("ConstantKind::Allocated (&str, invalid UTF-8)");
                            sm::ConstValue::ZeroSized
                        }
                    }
                }
                Some(TyKind::RigidTy(RigidTy::Slice(elem)))
                    if matches!(
                        elem.kind(),
                        TyKind::RigidTy(RigidTy::Uint(SmirUintTy::U8))
                    ) =>
                {
                    return sm::ConstValue::Slice { data, meta };
                }
                _ => {}
            },
            FatPointerRead::OverBudget => {
                // The allocation is a fat pointer whose payload
                // the expansion budget refused. The plain-pointee
                // shortcuts below would read the whole backing
                // allocation and silently replace the recorded
                // length with the allocation's own, so a reference
                // of one of those shapes keeps its exact `ByRef`
                // identity (stays undecoded) instead. Other shapes
                // have no such shortcut and fall through as
                // before.
                let drops_meta = match &inner_kind {
                    Some(TyKind::RigidTy(RigidTy::Str)) => true,
                    Some(TyKind::RigidTy(RigidTy::Slice(elem))) => matches!(
                        elem.kind(),
                        TyKind::RigidTy(RigidTy::Uint(SmirUintTy::U8))
                    ),
                    _ => false,
                };
                if drops_meta {
                    if let Some((_, prov)) = alloc.provenance.ptrs.first() {
                        return sm::ConstValue::ByRef {
                            alloc_id: prov.0.to_index() as u64,
                            offset: read_uint_at(alloc, 0, PTR_SIZE)
                                .unwrap_or(0) as u64,
                        };
                    }
                }
            }
            FatPointerRead::NotFat => {}
        }
        // Thin-pointer shape: follow the provenance to the pointee.
        if let Some((_, prov)) = alloc.provenance.ptrs.first() {
            let alloc_id = prov.0.to_index() as u64;
            // The offset the pointer itself stores: a pointer into
            // the middle of its allocation (a byte-array window or
            // a substring minted in a const context) denotes the
            // bytes from that offset on, not the allocation's
            // prefix, so the plain-pointee shortcuts below read
            // the denoted range instead of the whole allocation.
            let target_off =
                read_uint_at(alloc, 0, PTR_SIZE).unwrap_or(0) as usize;
            if let Some(pointee) = self.memory_alloc(prov) {
                let plain = pointee.provenance.ptrs.is_empty();
                match &inner_kind {
                    Some(TyKind::RigidTy(RigidTy::Str)) if plain => {
                        return match alloc_bytes_in(
                            &pointee,
                            target_off..pointee.bytes.len(),
                        )
                        .map(String::from_utf8)
                        {
                            Some(Ok(s)) => sm::ConstValue::Str(s),
                            _ => {
                                self.err("ConstantKind::Allocated (&str, unreadable)");
                                sm::ConstValue::ZeroSized
                            }
                        };
                    }
                    Some(TyKind::RigidTy(RigidTy::Slice(elem))) if plain => {
                        if matches!(
                            elem.kind(),
                            TyKind::RigidTy(RigidTy::Uint(SmirUintTy::U8))
                        ) {
                            return match alloc_bytes_in(
                                &pointee,
                                target_off..pointee.bytes.len(),
                            ) {
                                Some(data) => sm::ConstValue::Slice {
                                    meta: data.len() as u64,
                                    data,
                                },
                                None => {
                                    self.err("ConstantKind::Allocated (&[u8], unreadable)");
                                    sm::ConstValue::ZeroSized
                                }
                            };
                        }
                    }
                    Some(TyKind::RigidTy(RigidTy::Array(elem, len))) if plain => {
                        if matches!(
                            elem.kind(),
                            TyKind::RigidTy(RigidTy::Uint(SmirUintTy::U8))
                        ) {
                            // The declared length of the array
                            // governs, not the extent of the backing
                            // allocation: the array may be one field
                            // of a larger allocation (a const item
                            // tuple field is the canonical case),
                            // whose following fields are not part of
                            // the array, and whose trailing padding
                            // bytes are uninitialised and would fail
                            // the read. A length that cannot be
                            // evaluated keeps the allocation-extent
                            // read.
                            let end = match len.eval_target_usize() {
                                Ok(n) => target_off
                                    .checked_add(n as usize)
                                    .unwrap_or(pointee.bytes.len()),
                                Err(_) => pointee.bytes.len(),
                            };
                            return match alloc_bytes_in(
                                &pointee,
                                target_off..end,
                            ) {
                                Some(data) => sm::ConstValue::Bytes(data),
                                None => {
                                    self.err("ConstantKind::Allocated (&[u8; N], unreadable)");
                                    sm::ConstValue::ZeroSized
                                }
                            };
                        }
                    }
                    _ => {}
                }
            }
            // General shape: decode the pointee allocation by the
            // inner type, guided by the type's layout. Anything the
            // decoder cannot materialise keeps the exact `ByRef`
            // identity and stays unsupported downstream.
            if let Some(v) = self.decode_ref(inner, alloc, 0, 0) {
                return v;
            }
            return sm::ConstValue::ByRef {
                alloc_id,
                offset: read_uint_at(alloc, 0, PTR_SIZE).unwrap_or(0) as u64,
            };
        }
        // No provenance at all. A fat pointer (a slice reference)
        // is an address word plus a metadata (length) word: a
        // dangling slice constant minted by
        // `slice::from_raw_parts(ptr, len)` in a const context has
        // no pointee, but its address identity and length are
        // exactly observable, so it decodes as a `FatPtr`.
        if matches!(&inner_kind, Some(TyKind::RigidTy(RigidTy::Slice(_)))) {
            if let (Ok(addr), Ok(meta)) = (
                alloc.read_partial_uint(0..PTR_SIZE),
                alloc.read_partial_uint(PTR_SIZE..2 * PTR_SIZE),
            ) {
                return sm::ConstValue::FatPtr {
                    addr: addr as u64,
                    meta: meta as u64,
                };
            }
        }
        // A stored zero word is a null pointer constant. Anything
        // else is an integer-derived address, which has no faithful
        // value form.
        if alloc.read_partial_uint(0..PTR_SIZE).ok() == Some(0) {
            return sm::ConstValue::NullPtr;
        }
        self.err(format!(
            "ConstantKind::Allocated (reference without provenance: {})",
            short_ty(inner)
        ));
        sm::ConstValue::ZeroSized
    }


    /// Whether a closure type's captured environment is empty: the
    /// environment tuple is the last of the closure's generic
    /// arguments (after the parent arguments, the kind and the
    /// signature). Anything not recognisably an empty tuple counts
    /// as non-empty.
    fn closure_env_is_empty(args: &GenericArgs) -> bool {
        match Self::closure_env_elems(args) {
            Some(elems) => elems.is_empty(),
            None => false,
        }
    }

    /// The element types of a closure type's captured
    /// environment: the closure generic arguments' last
    /// argument is the upvars tuple, whose elements are the
    /// captures in declaration order. `None` when the
    /// arguments do not have that shape.
    fn closure_env_elems(
        args: &GenericArgs,
    ) -> Option<Vec<rustc_public::ty::Ty>> {
        match args.0.last() {
            Some(GenericArgKind::Type(t)) => match t.kind() {
                TyKind::RigidTy(RigidTy::Tuple(elems)) => Some(elems),
                _ => None,
            },
            _ => None,
        }
    }

    /// Charge `units` against the shared constant-expansion
    /// budget (see `MAX_CONST_TOTAL_ELEMS`); `false` when the
    /// charge does not fit, in which case the constant being
    /// decoded stays undecoded. A failed charge also sets
    /// `const_budget_failed`, so a caller that would otherwise
    /// fall back to a different decode can tell budget
    /// exhaustion apart from ordinary undecodability.
    pub(crate) fn spend_const_budget(&mut self, units: usize) -> bool {
        if units > self.const_budget {
            self.const_budget_failed = true;
            return false;
        }
        self.const_budget -= units;
        true
    }

    /// Charge `units` against the materialisation-work budget
    /// (see `MAX_CONST_WORK_BYTES`); `false` when the charge
    /// does not fit, in which case the materialisation that
    /// prompted it is refused and the constant being decoded
    /// stays undecoded. A failed charge also sets
    /// `const_work_failed`, so a caller that would otherwise
    /// fall back to a different decode can tell work exhaustion
    /// apart from ordinary undecodability, and later
    /// materialisations in the same decode tree are refused
    /// without being performed at all.
    fn spend_work_budget(&mut self, units: usize) -> bool {
        if units > self.const_work_budget {
            self.const_work_failed = true;
            return false;
        }
        self.const_work_budget -= units;
        true
    }

    /// Whether every byte of `alloc` in `offset..offset + size`
    /// is uninitialised, spending one unit of the scan budget
    /// (see `MAX_CONST_SCAN_BYTES`) per byte examined. `None`
    /// when the budget runs out before the outcome is decided:
    /// the caller then decodes structurally, so exhaustion
    /// costs only the `Uninit` shortcut, never a value the
    /// structural decode could still build. The exhaustion is
    /// recorded in `const_scan_failed` for the fallbacks in
    /// `decode_ref`. The scan is the
    /// one decode step whose cost is proportional to a range's
    /// byte size rather than to the materialised value -- a
    /// partially initialised pointee is scanned in full at the
    /// aggregate (whose scan runs to its one initialised
    /// field) and again at its uninitialised field -- so
    /// without a budget of its own, a table repeating one such
    /// reference scans the same megabytes once per element
    /// while spending only a few expansion-budget units each
    /// (a 100,000-element table of references to one struct
    /// with a 16 MiB uninitialised payload scanned terabytes
    /// in aggregate). Charging the scan against the expansion
    /// budget instead would refuse single values outright:
    /// one 16 MiB uninitialised buffer is a small constant
    /// (an `Uninit` marker) whose one scan is unremarkable --
    /// only the repetition is unbounded, and only a budget on
    /// the examination itself bounds that.
    fn range_all_uninit(
        &mut self,
        alloc: &Allocation,
        offset: usize,
        size: usize,
    ) -> Option<bool> {
        let range = alloc.bytes.get(offset..offset + size)?;
        let mut examined = 0usize;
        let all = range.iter().all(|b| {
            examined += 1;
            b.is_none()
        });
        if examined > self.const_scan_budget {
            self.const_scan_budget = 0;
            self.const_scan_failed = true;
            return None;
        }
        self.const_scan_budget -= examined;
        Some(all)
    }

    /// Whether the byte range `offset..offset+size` of `alloc`
    /// is partially initialised: at least one initialised byte
    /// and at least one uninitialised byte. The scan is charged
    /// against the scan budget exactly as `range_all_uninit`'s
    /// is; a range the budget cannot pay for, and a range
    /// leaving the allocation, report `false` (the caller's
    /// fallback then does not fire). Unlike the decode's own
    /// scans this gate sets no failure flag: it is not part of
    /// a typed decode, only of the fallback decision after one.
    pub(crate) fn range_partially_uninit(
        &mut self,
        alloc: &Allocation,
        offset: usize,
        size: usize,
    ) -> bool {
        let Some(range) = alloc.bytes.get(offset..offset + size) else {
            return false;
        };
        let mut examined = 0usize;
        let mut any_init = false;
        let mut any_uninit = false;
        for b in range {
            examined += 1;
            if b.is_some() {
                any_init = true;
            } else {
                any_uninit = true;
            }
        }
        if examined > self.const_scan_budget {
            return false;
        }
        self.const_scan_budget -= examined;
        any_init && any_uninit
    }

    /// Decode the value of type `ty` stored in `alloc` at `offset`
    /// into a structured constant. Aggregates are laid out with the
    /// type's own layout (`Ty::layout`), pointers are followed
    /// through provenance, and enums are resolved through their tag
    /// (direct or niche encoding). Returns `None` for anything that
    /// has no faithful value form: unions, function pointers and
    /// vtables, references to functions, null pointers,
    /// unsized values behind no pointer, and nesting deeper than
    /// the depth budget.
    pub(crate) fn decode_const(
        &mut self,
        ty: &rustc_public::ty::Ty,
        alloc: &Allocation,
        offset: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        if depth > MAX_CONST_DEPTH {
            return None;
        }
        // One shared expansion budget per decoded constant
        // (see `MAX_CONST_TOTAL_ELEMS`): the root call refills
        // it and every call spends one unit, so the total
        // materialised size is bounded even when declared
        // element counts nest multiplicatively. Byte-copying
        // arms spend the copied length against the same budget.
        // A constant whose expansion exhausts it stays undecoded.
        if depth == 0 {
            self.const_budget = MAX_CONST_TOTAL_ELEMS;
            self.const_scan_budget = MAX_CONST_SCAN_BYTES;
            self.const_work_budget = MAX_CONST_WORK_BYTES;
            self.const_work_failed = false;
            self.clear_alloc_caches();
        }
        if !self.spend_const_budget(1) {
            return None;
        }
        // A range whose bytes are all uninitialized is the
        // uninitialized value of this type (the payload of a
        // `MaybeUninit::uninit()` constant, at whichever nesting
        // level it appears). A partially initialized range falls
        // through to the structural decode below, which reads the
        // initialized parts field by field. The scan itself is
        // charged against the scan budget (see
        // `range_all_uninit`): a scan the budget cannot pay for
        // is skipped, and the structural decode proceeds without
        // the shortcut.
        if let Ok(layout) = ty.layout() {
            let size = layout.shape().size.bytes() as usize;
            if size > 0 && offset + size <= alloc.bytes.len() {
                if let Some(true) = self.range_all_uninit(alloc, offset, size) {
                    return Some(sm::ConstValue::Uninit);
                }
            }
        }
        match ty.kind() {
            TyKind::RigidTy(RigidTy::Bool) => match read_uint_at(alloc, offset, 1)? {
                0 => Some(sm::ConstValue::Bool(false)),
                1 => Some(sm::ConstValue::Bool(true)),
                _ => None,
            },
            TyKind::RigidTy(RigidTy::Char) => {
                let v = read_uint_at(alloc, offset, 4)?;
                char::from_u32(v as u32).map(sm::ConstValue::Char)
            }
            TyKind::RigidTy(RigidTy::Int(t)) => {
                let size = int_ty_size(&t)?;
                let raw = read_uint_at(alloc, offset, size)?;
                Some(sm::ConstValue::Scalar(sm::Scalar::Int {
                    data: sign_extend(raw, size),
                    size: size as u8,
                }))
            }
            TyKind::RigidTy(RigidTy::Uint(t)) => {
                let size = uint_ty_size(&t)?;
                let raw = read_uint_at(alloc, offset, size)?;
                Some(sm::ConstValue::Scalar(sm::Scalar::Int {
                    data: raw,
                    size: size as u8,
                }))
            }
            TyKind::RigidTy(RigidTy::Float(f)) => {
                let size = match f {
                    SmirFloatTy::F16 => 2,
                    SmirFloatTy::F32 => 4,
                    SmirFloatTy::F64 => 8,
                    SmirFloatTy::F128 => 16,
                };
                let bits = read_uint_at(alloc, offset, size)?;
                Some(sm::ConstValue::Scalar(sm::Scalar::Float {
                    bits,
                    ty: float_ty(f),
                }))
            }
            TyKind::RigidTy(RigidTy::Str) => {
                // A `str` value in non-reference position is the
                // trailing unsized field of an ADT (a newtype over
                // `str`): its bytes run to the end of the allocation.
                // The copy is charged against the shared expansion
                // budget before it is made: a table of references
                // to one shared value of such a type reaches this
                // arm once per element, so uncharged copies would
                // add up past the budget.
                if !self.spend_const_budget(alloc.bytes.len().saturating_sub(offset)) {
                    return None;
                }
                let bytes = alloc_bytes_in(alloc, offset..alloc.bytes.len())?;
                String::from_utf8(bytes).ok().map(sm::ConstValue::Str)
            }
            TyKind::RigidTy(RigidTy::Slice(elem)) => {
                // Likewise a slice in non-reference position is a
                // trailing unsized field (the `[u8]` inside `CStr`
                // is the canonical case): its elements run to the
                // end of the allocation.
                let bytes = alloc_bytes_in(alloc, offset..alloc.bytes.len())?;
                if matches!(
                    elem.kind(),
                    TyKind::RigidTy(RigidTy::Uint(SmirUintTy::U8))
                ) {
                    // Charged like the `str` arm above: the tail
                    // copy spends its length against the shared
                    // expansion budget.
                    if !self.spend_const_budget(bytes.len()) {
                        return None;
                    }
                    return Some(sm::ConstValue::Bytes(bytes));
                }
                let stride = elem.layout().ok()?.shape().size.bytes();
                if stride == 0 || bytes.len() % stride != 0 {
                    return None;
                }
                let mut elems = Vec::new();
                for i in 0..bytes.len() / stride {
                    elems.push(self.decode_const(
                        &elem,
                        alloc,
                        offset + i * stride,
                        depth + 1,
                    )?);
                }
                Some(sm::ConstValue::SliceVal(elems))
            }
            TyKind::RigidTy(RigidTy::Ref(_, inner, _)) => {
                self.decode_ref(&inner, alloc, offset, depth)
            }
            TyKind::RigidTy(RigidTy::RawPtr(inner, _)) => {
                self.decode_ref(&inner, alloc, offset, depth)
            }
            TyKind::RigidTy(RigidTy::Array(elem, len)) => {
                let count = len.eval_target_usize().ok()?;
                let shape = ty.layout().ok()?.shape();
                let FieldsShape::Array { stride, count: layout_count } = shape.fields else {
                    return None;
                };
                if layout_count != count { return None; }
                if count > MAX_CONST_ARRAY_ELEMS {
                    // Past the element-wise bound, an array still
                    // materialises when its elements are provably
                    // present in the allocation: a byte array as
                    // the raw bytes themselves -- the form byte
                    // arrays take in reference position, and byte
                    // slices in value position above -- and any
                    // other element type element by element,
                    // exactly as under the bound. The resource
                    // concern behind the bound -- a declared
                    // length decoupled from the allocation,
                    // expanded into one constant per element --
                    // is answered by the presence check itself:
                    // the element-wise decode below only runs
                    // when `count * stride` bytes lie inside the
                    // allocation from `offset`, so the work is
                    // bounded by bytes the host already holds.
                    // A zero stride proves nothing about the
                    // count either, so a zero-sized element
                    // type takes the replication path below:
                    // one decode replicated to the declared
                    // count, charged to the shared expansion
                    // budget, which is what bounds a declared
                    // count decoupled from the allocation.
                    // Zero-sized descendants of a nonzero-stride
                    // element are not bounded by the byte count
                    // either; the shared expansion budget spent
                    // by every recursive decode bounds those
                    // (see `MAX_CONST_TOTAL_ELEMS`).
                    if stride.bytes() == 1
                        && matches!(elem.kind(), TyKind::RigidTy(RigidTy::Uint(SmirUintTy::U8)))
                    {
                        // The copy materialises `count` bytes:
                        // charge them against the shared expansion
                        // budget before allocating, like the
                        // byte-copying arms of `decode_ref` do.
                        // The presence check above bounds one
                        // decode by the allocation, but a table of
                        // references to one shared array reaches
                        // this arm once per element, so without
                        // the charge the copies add up past the
                        // budget.
                        if !self.spend_const_budget(count as usize) {
                            return None;
                        }
                        // Read the array's own byte range, not
                        // the whole allocation: a static's
                        // storage is routinely padded past the
                        // value with uninitialised bytes, which
                        // must not fail the decode -- but an
                        // uninitialised byte inside the array
                        // does.
                        let end = offset.checked_add(count as usize)?;
                        let range = alloc.bytes.get(offset..end)?;
                        let mut bytes = Vec::with_capacity(count as usize);
                        for byte in range {
                            bytes.push((*byte)?);
                        }
                        return Some(sm::ConstValue::Bytes(bytes));
                    }
                    let stride_bytes = stride.bytes();
                    if stride_bytes == 0 {
                        // A zero-sized element type stores
                        // nothing per element: every element
                        // reads the same zero bytes at the base,
                        // so one decode replicated to the
                        // declared count is the exact value --
                        // the same rule the slice decoder
                        // applies to a zero-sized element slice
                        // in `decode_ref`. The replication is
                        // charged to the shared expansion budget
                        // before it is built: a declared count
                        // the budget cannot cover (the resource
                        // concern behind the element-wise bound)
                        // stays undecoded, as does an element
                        // whose own decode fails.
                        if !self.spend_const_budget(count as usize) {
                            return None;
                        }
                        let elem_value =
                            self.decode_const(&elem, alloc, offset, depth + 1)?;
                        return Some(sm::ConstValue::Tuple(vec![
                            elem_value;
                            count as usize
                        ]));
                    }
                    let total = (count as u128).checked_mul(stride_bytes as u128)?;
                    if offset as u128 + total > alloc.bytes.len() as u128 {
                        return None;
                    }
                }
                let mut elems = Vec::new();
                for i in 0..count {
                    elems.push(self.decode_const(
                        &elem,
                        alloc,
                        offset + i as usize * stride.bytes(),
                        depth + 1,
                    )?);
                }
                Some(sm::ConstValue::Tuple(elems))
            }
            TyKind::RigidTy(RigidTy::Tuple(elems)) => {
                if elems.is_empty() {
                    return Some(sm::ConstValue::ZeroSized);
                }
                let offsets = arbitrary_offsets(ty)?;
                if offsets.len() != elems.len() { return None; }
                let mut fields = Vec::new();
                for (elem, off) in elems.iter().zip(offsets) {
                    fields.push(self.decode_const(elem, alloc, offset + off, depth + 1)?);
                }
                Some(sm::ConstValue::Tuple(fields))
            }
            TyKind::RigidTy(RigidTy::FnDef(def, args)) => {
                // A function item value is zero-sized: the type
                // itself names the callable, no bytes are stored.
                if let Some(inst) = resolve_instance_guarded(def, &args) {
                    self.note_const_fn(&inst);
                }
                let d = self.fn_def_id(&def, &args);
                Some(sm::ConstValue::FnDef(d))
            }
            TyKind::RigidTy(RigidTy::FnPtr(_)) => {
                self.decode_fn_pointer(alloc, offset)
            }
            TyKind::RigidTy(RigidTy::Closure(def, args)) => {
                // A stored closure value: the callable plus its
                // captured environment, which the allocation
                // holds as the closure type's fields. The
                // environment fields are decoded at the closure
                // layout's offsets, the same arbitrary-offsets
                // walk a tuple constant gets; a capture whose
                // own decode fails leaves the whole constant
                // undecoded.
                self.note_const_closure(&def, &args);
                let d = self.closure_def_id(&def, &args);
                if Self::closure_env_is_empty(&args) {
                    Some(sm::ConstValue::Closure(d, Vec::new()))
                } else {
                    let Some(elems) = Self::closure_env_elems(&args) else {
                        return None;
                    };
                    let offsets = arbitrary_offsets(ty)?;
                    if offsets.len() != elems.len() {
                        return None;
                    }
                    let mut env = Vec::new();
                    for (elem, off) in elems.iter().zip(offsets) {
                        env.push(self.decode_const(
                            elem,
                            alloc,
                            offset + off,
                            depth + 1,
                        )?);
                    }
                    Some(sm::ConstValue::Closure(d, env))
                }
            }
            TyKind::RigidTy(RigidTy::Pat(inner, _)) => {
                // Pattern types (e.g. the `usize is 0..=N` field of
                // `core::num::niche_types` wrappers) share the inner
                // type's representation; the stored value is the
                // inner type's value.
                self.decode_const(&inner, alloc, offset, depth)
            }
            TyKind::RigidTy(RigidTy::Adt(def, args)) => {
                self.decode_adt(&def, &args, ty, alloc, offset, depth)
            }
            _ => None,
        }
    }

    /// Decode an ADT value (struct or enum) at `offset` in `alloc`.
    fn decode_adt(
        &mut self,
        def: &AdtDef,
        args: &GenericArgs,
        ty: &rustc_public::ty::Ty,
        alloc: &Allocation,
        offset: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        if matches!(def.name().as_str(), "core::ffi::CStr" | "std::ffi::CStr") {
            // The one field is a `[c_char]` byte string. `c_char`
            // is signed on this platform, so the generic decode
            // would produce signed-char elements, but a `CStr`'s
            // contents are bytes by definition — every public
            // view of it is a byte view (`to_bytes`, `to_str`,
            // the `&[u8]` casts its own methods perform) — so
            // decode them as the byte string they are, like the
            // `[u8]` trailing-slice decode below.
            let bytes = alloc_bytes_in(alloc, offset..alloc.bytes.len())?;
            // The payload copy is charged against the shared
            // expansion budget: a table of references to one
            // shared `CStr` reaches this arm once per element,
            // so uncharged copies would add up past the budget.
            // Only the tail from `offset` is copied, not the
            // whole allocation before it.
            if !self.spend_const_budget(bytes.len()) {
                return None;
            }
            return Some(sm::ConstValue::Adt {
                variant: 0,
                fields: vec![sm::ConstValue::Bytes(bytes)],
            });
        }
        let shape = ty.layout().ok()?.shape();
        let (variant_idx, offsets): (usize, Vec<usize>) = match def.kind() {
            AdtKind::Struct => {
                let variant = def.variant(VariantIdx::to_val(0))?;
                if variant.fields().is_empty() {
                    return Some(sm::ConstValue::Adt { variant: 0, fields: Vec::new() });
                }
                (0, arbitrary_offsets(ty)?)
            }
            AdtKind::Enum => match &shape.variants {
                VariantsShape::Single { index } => {
                    let idx = index.to_index();
                    let variant = def.variant(VariantIdx::to_val(idx))?;
                    if variant.fields().is_empty() {
                        return Some(sm::ConstValue::Adt {
                            variant: idx as u32,
                            fields: Vec::new(),
                        });
                    }
                    (idx, arbitrary_offsets(ty)?)
                }
                VariantsShape::Multiple {
                    tag,
                    tag_encoding,
                    tag_field,
                    variants,
                } => {
                    let (idx, offsets) = match tag_encoding {
                        TagEncoding::Direct => {
                            // The stored tag's width and signedness
                            // come from the layout's tag scalar, not
                            // from the enum's discriminant type:
                            // rustc stores a small enum's tag in a
                            // scalar narrower than the type-level
                            // discriminant type (a fieldless enum
                            // with two variants gets a 1-byte tag
                            // while its discriminant type is isize).
                            let (size, signed) = tag_scalar_size_signed(tag)?;
                            let tag_off = arbitrary_offsets(ty)?.get(*tag_field).copied()?;
                            let raw = read_uint_at(alloc, offset + tag_off, size)?;
                            let val = if signed { sign_extend(raw, size) } else { raw };
                            // Compare in the value domain: the tag
                            // read above is sign-extended from the
                            // tag's width, while `Discr::val` is the
                            // bit pattern at the discriminant type's
                            // width (see `discr_value`); the raw
                            // patterns agree only when both widths
                            // are 128, so a negative discriminant
                            // never matched before this normalisation.
                            let tag_value = val as i128;
                            let mut found = None;
                            for i in 0..def.num_variants() {
                                let d = def.discriminant_for_variant(VariantIdx::to_val(i));
                                if discr_value(&d) == Some(tag_value) {
                                    found = Some(i);
                                    break;
                                }
                            }
                            let idx = found?;
                            let offsets = variants
                                .get(idx)?
                                .offsets
                                .iter()
                                .map(|o| o.bytes())
                                .collect();
                            (idx, offsets)
                        }
                        TagEncoding::Niche {
                            untagged_variant,
                            niche_variants,
                            niche_start,
                        } => {
                            let untagged = untagged_variant.to_index();
                            // The niche scalar's absolute offset
                            // within the enum is the layout's sole
                            // top-level field offset (the layout
                            // computation records the niche offset
                            // there), and its width is the tag
                            // scalar's width. The dataful variant's
                            // field that *contains* the niche may
                            // start earlier and be wider (a whole
                            // `Layout` around its alignment word),
                            // so neither its offset nor its size
                            // identifies the niche itself.
                            let tag_off = arbitrary_offsets(ty)?.get(*tag_field).copied()?;
                            let (size, _) = tag_scalar_size_signed(tag)?;
                            if size == 0 || size > 16 { return None; }
                            let raw = read_uint_at(alloc, offset + tag_off, size)?;
                            let start = niche_variants.start().to_index();
                            let end = niche_variants.end().to_index();
                            // A provenance entry at the niche field
                            // means the stored value is a real
                            // pointer, not a niche encoding of
                            // another variant (the `Option<&T>` case:
                            // a non-null pointer reads back as
                            // offset 0 in its target allocation).
                            let has_prov = prov_at(alloc, offset + tag_off).is_some();
                            let idx = if !has_prov
                                && raw >= *niche_start
                                && raw - *niche_start <= (end - start) as u128
                            {
                                start + (raw - *niche_start) as usize
                            } else {
                                untagged
                            };
                            let offsets = variants
                                .get(idx)?
                                .offsets
                                .iter()
                                .map(|o| o.bytes())
                                .collect();
                            (idx, offsets)
                        }
                    };
                    (idx, offsets)
                }
                VariantsShape::Empty => return None,
            },
            AdtKind::Union => return self.decode_union(&def, &args, alloc, offset, depth),
        };
        let variant = def.variant(VariantIdx::to_val(variant_idx))?;
        let fields = self.decode_fields(&variant, args, &offsets, alloc, offset, depth)?;
        Some(sm::ConstValue::Adt {
            variant: variant_idx as u32,
            fields,
        })
    }

    /// Decode a union value. Which field is active is not recorded
    /// in an allocation, but a union *value* in this model carries
    /// every field, and reading a union field in Rust reinterprets
    /// the stored bytes as that field's type — so decoding every
    /// field from the same bytes at offset 0 (where all union
    /// fields live) is exactly the semantics a later field read
    /// performs. Scalar and aggregate decodes are total on
    /// initialised bytes, so a multi-payload union decodes field
    /// by field; a field whose decode fails over a fully
    /// initialised span (a pointer without provenance is the
    /// usual case, handled by the disambiguation below when it
    /// is the stored field) fails the whole constant, honestly.
    /// A field whose span is only partially initialised -- the
    /// constant was written through a narrower field, so this
    /// reading spans bytes the constant never determined --
    /// decodes as uninitialised instead (see the field loop
    /// below), provided a decoded field's image covers every
    /// byte the span determines: the union's storage image is
    /// composed from the fields that decode, so a determined
    /// byte no decoded field carries would be lost, and that
    /// shape stays undecoded for the erased form.
    fn decode_union(
        &mut self,
        def: &AdtDef,
        args: &GenericArgs,
        alloc: &Allocation,
        offset: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        let variant = def.variant(VariantIdx::to_val(0))?;
        let fields = variant.fields();
        let mut sizes = Vec::new();
        for field in fields.iter() {
            let fty = field.ty_with_args(args);
            let size = fty.layout().ok()?.shape().size.bytes();
            sizes.push(size);
        }
        // A provenance entry at the union's offset can only belong
        // to a pointer-shaped field stored there: pointer fields
        // are the ones that carry provenance, and the entry spans
        // exactly one pointer word. When exactly one field is a
        // pointer of that size, it is the stored field; the other
        // payload fields overlap its bytes and decode as
        // uninitialized (they are not the live field, and reading
        // them would reinterpret the pointer's representation).
        let prov_at_offset = prov_at(alloc, offset).is_some();
        if prov_at_offset {
            let ptr_fields: Vec<usize> = fields
                .iter()
                .enumerate()
                .filter(|(i, field)| {
                    if sizes[*i] != PTR_SIZE {
                        return false;
                    }
                    // The pointer may sit under single-field
                    // transparent wrappers (`ManuallyDrop`,
                    // `UnsafeCell`, ...).
                    let mut inner = field.ty_with_args(args);
                    for _ in 0..8 {
                        match inner.kind() {
                            TyKind::RigidTy(RigidTy::Adt(def, inner_args)) => {
                                let variants = def.variants();
                                if variants.len() != 1 {
                                    return false;
                                }
                                let vfields = variants[0].fields();
                                if vfields.len() != 1 {
                                    return false;
                                }
                                inner = vfields[0].ty_with_args(&inner_args);
                            }
                            _ => break,
                        }
                    }
                    matches!(
                        inner.kind(),
                        TyKind::RigidTy(
                            RigidTy::Ref(..) | RigidTy::RawPtr(..) | RigidTy::FnPtr(_)
                        )
                    )
                })
                .map(|(i, _)| i)
                .collect();
            if ptr_fields.len() == 1 {
                let live = ptr_fields[0];
                let mut out = Vec::new();
                for (i, field) in fields.iter().enumerate() {
                    let fty = field.ty_with_args(args);
                    if i == live || sizes[i] == 0 {
                        out.push(self.decode_const(&fty, alloc, offset, depth + 1)?);
                    } else {
                        out.push(sm::ConstValue::Uninit);
                    }
                }
                return Some(sm::ConstValue::Adt {
                    variant: 0,
                    fields: out,
                });
            }
        }
        // Decode every field first: whether an undetermined
        // field may decode as uninitialised (below) depends on
        // the fields that do decode, in either order.
        let mut decoded: Vec<Option<sm::ConstValue>> = Vec::new();
        let mut budget_hits: Vec<bool> = Vec::new();
        for field in fields.iter() {
            let fty = field.ty_with_args(args);
            let budget_before = (
                self.const_budget_failed,
                self.const_scan_failed,
                self.const_work_failed,
            );
            decoded.push(self.decode_const(&fty, alloc, offset, depth + 1));
            budget_hits.push(
                (self.const_budget_failed && !budget_before.0)
                    || (self.const_scan_failed && !budget_before.1)
                    || (self.const_work_failed && !budget_before.2),
            );
        }
        // The widest span a genuinely decoded field covers with
        // fully initialised bytes: its storage image reproduces
        // the constant's bytes over that whole span.
        let full_init_cover = fields
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                decoded[*i].is_some()
                    && !self.range_partially_uninit(alloc, offset, sizes[*i] as usize)
            })
            .map(|(i, _)| sizes[i])
            .max()
            .unwrap_or(0);
        let mut out = Vec::new();
        let mut any_decoded = false;
        let mut any_undetermined = false;
        for (i, value) in decoded.into_iter().enumerate() {
            match value {
                Some(value) => {
                    any_decoded = true;
                    out.push(value);
                }
                None => {
                    // A field whose span the constant's bytes do
                    // not fully determine has no value of its
                    // type: the constant was written through a
                    // narrower field, so this wider reading spans
                    // initialised and uninitialised bytes, and a
                    // scalar leaf spanning both has no value form
                    // (the failure the erased-block fallback in
                    // `decode_ref` answers for a whole pointee).
                    // Inside a union the field is not necessarily
                    // lost: a union value's fields are readings of
                    // one byte span, and the interpreter composes
                    // the union's storage image from the fields
                    // that do decode -- so the undetermined field
                    // decodes as uninitialised, the same form the
                    // provenance disambiguation above gives a
                    // field that is not the stored one, PROVIDED
                    // every byte this field's span determines is
                    // covered by a decoded field's image: a
                    // determined byte inside the undetermined
                    // span itself (the constant's only write
                    // landed there) has no other carrier, and
                    // losing it would corrupt the constant, so
                    // that shape keeps the erased form, which
                    // preserves the bytes exactly. A failure the
                    // budgets caused, or one over a fully
                    // initialised span (an undecodable type, not
                    // an undetermined reading), likewise keeps
                    // the whole constant undecoded, honestly.
                    if sizes[i] == 0
                        || budget_hits[i]
                        || !self.range_partially_uninit(alloc, offset, sizes[i] as usize)
                    {
                        return None;
                    }
                    let span = alloc
                        .bytes
                        .get(offset..offset + sizes[i] as usize)
                        .unwrap_or(&[]);
                    if let Some(highest) = span.iter().rposition(|b| b.is_some()) {
                        if highest >= full_init_cover as usize {
                            return None;
                        }
                    }
                    any_undetermined = true;
                    out.push(sm::ConstValue::Uninit);
                }
            }
        }
        // When no field's reading is determined at all, the
        // structured form would hold nothing the constant
        // determines, where the erased form preserves its bytes
        // exactly: keep the constant undecoded for it.
        if any_undetermined && !any_decoded {
            return None;
        }
        Some(sm::ConstValue::Adt {
            variant: 0,
            fields: out,
        })
    }

    /// Decode the fields of one variant at the given offsets.
    fn decode_fields(
        &mut self,
        variant: &rustc_public::ty::VariantDef,
        args: &GenericArgs,
        offsets: &[usize],
        alloc: &Allocation,
        offset: usize,
        depth: usize,
    ) -> Option<Vec<sm::ConstValue>> {
        let fields = variant.fields();
        if fields.len() != offsets.len() { return None; }
        let mut out = Vec::new();
        for (field, off) in fields.iter().zip(offsets) {
            let fty = field.ty_with_args(args);
            out.push(self.decode_const(&fty, alloc, offset + off, depth + 1)?);
        }
        Some(out)
    }

    /// Record the materialisation plan of a pointer cast's
    /// target pointee type (see `mat_plan`), keyed by the
    /// converted pointee type's `Debug` rendering — the key
    /// `mir_convert` looks up when it converts the same cast.
    pub(crate) fn note_mat_plan(&mut self, ty: &rustc_public::ty::Ty, converted: &sm::Ty) {
        let pointee = match converted {
            sm::Ty::Ref(_, p, _) | sm::Ty::RawPtr(p, _) => (**p).clone(),
            _ => return,
        };
        let key = format!("{pointee:?}");
        if self.mat_plans.contains_key(&key) {
            return;
        }
        let rustc_pointee = match ty.kind() {
            TyKind::RigidTy(RigidTy::Ref(_, inner, _))
            | TyKind::RigidTy(RigidTy::RawPtr(inner, _)) => inner,
            _ => return,
        };
        let mut active = HashSet::new();
        let plan = self.mat_plan(&rustc_pointee, 0, &mut active);
        self.mat_plans.insert(key, plan);
    }

    /// The materialisation plan of a type (see
    /// `sparagmos_interp::ir::MatPlan`): the byte-layout
    /// description the interpreter needs to materialise a
    /// type-erased constant allocation into a value of this
    /// type. Built from the compiler's layout query, mirroring
    /// `decode_const`'s reading of the same layouts; a type
    /// whose layout cannot be described gets
    /// `MatPlan::Unsupported` (materialising it is an explicit
    /// error downstream, never a guess).
    fn mat_plan(
        &mut self,
        ty: &rustc_public::ty::Ty,
        depth: usize,
        active: &mut HashSet<rustc_public::ty::Ty>,
    ) -> sparagmos_interp::ir::MatPlan {
        use sparagmos_interp::ir::MatPlan;
        if depth > MAX_CONST_DEPTH {
            return MatPlan::Unsupported;
        }
        // A type already being expanded on the path from the
        // cast's target type: the type graph cycles here, and
        // expanding the type again would unroll the cycle into
        // a tree whose size grows like the branching factor
        // raised to the depth limit (a recursive struct with
        // three pointer fields alone would build tens of
        // millions of plans and exhaust memory during
        // conversion, for every cast that mentions the type).
        // The recursive edge gets no plan -- materialising
        // through it is an explicit error, never a guess --
        // and every other edge of the type plans as before.
        if !active.insert(*ty) {
            return MatPlan::Unsupported;
        }
        let plan = self.mat_plan_body(ty, depth, active);
        active.remove(ty);
        plan
    }

    /// The plan of one type, expanded one level (see
    /// `mat_plan`, which guards the expansion against type
    /// graph cycles).
    fn mat_plan_body(
        &mut self,
        ty: &rustc_public::ty::Ty,
        depth: usize,
        active: &mut HashSet<rustc_public::ty::Ty>,
    ) -> sparagmos_interp::ir::MatPlan {
        use sparagmos_interp::ir::MatPlan;
        match ty.kind() {
            TyKind::RigidTy(RigidTy::Bool) => MatPlan::Bool,
            TyKind::RigidTy(RigidTy::Char) => MatPlan::Char,
            TyKind::RigidTy(RigidTy::Int(t)) => match int_ty_size(&t) {
                Some(size) => MatPlan::Int {
                    size: size as u8,
                    signed: true,
                },
                None => MatPlan::Unsupported,
            },
            TyKind::RigidTy(RigidTy::Uint(t)) => match uint_ty_size(&t) {
                Some(size) => MatPlan::Int {
                    size: size as u8,
                    signed: false,
                },
                None => MatPlan::Unsupported,
            },
            TyKind::RigidTy(RigidTy::Float(f)) => MatPlan::Float {
                size: match f {
                    SmirFloatTy::F16 => 2,
                    SmirFloatTy::F32 => 4,
                    SmirFloatTy::F64 => 8,
                    SmirFloatTy::F128 => 16,
                },
            },
            TyKind::RigidTy(RigidTy::Str) => MatPlan::Str,
            TyKind::RigidTy(RigidTy::Slice(elem)) => {
                MatPlan::Slice(Box::new(self.mat_plan(&elem, depth + 1, active)))
            }
            TyKind::RigidTy(RigidTy::Ref(_, inner, _))
            | TyKind::RigidTy(RigidTy::RawPtr(inner, _)) => {
                // A `dyn` pointee's value needs its vtable,
                // which no plan can name; everything else
                // recurses into the pointee.
                if matches!(inner.kind(), TyKind::RigidTy(RigidTy::Dynamic(..))) {
                    MatPlan::Unsupported
                } else {
                    MatPlan::Ref(Box::new(self.mat_plan(&inner, depth + 1, active)))
                }
            }
            TyKind::RigidTy(RigidTy::FnPtr(_)) => MatPlan::FnPtr,
            TyKind::RigidTy(RigidTy::FnDef(..)) => MatPlan::Unit,
            TyKind::RigidTy(RigidTy::Tuple(elems)) => {
                if elems.is_empty() {
                    return MatPlan::Unit;
                }
                let Some(offsets) = arbitrary_offsets(ty) else {
                    return MatPlan::Unsupported;
                };
                if offsets.len() != elems.len() {
                    return MatPlan::Unsupported;
                }
                let Ok(layout) = ty.layout() else {
                    return MatPlan::Unsupported;
                };
                let fields = elems
                    .iter()
                    .zip(offsets)
                    .map(|(e, off)| sparagmos_interp::ir::MatField {
                        offset: off as u64,
                        plan: self.mat_plan(e, depth + 1, active),
                    })
                    .collect();
                MatPlan::Tuple {
                    size: layout.shape().size.bytes() as u64,
                    fields,
                }
            }
            TyKind::RigidTy(RigidTy::Array(elem, len)) => {
                let Ok(count) = len.eval_target_usize() else {
                    return MatPlan::Unsupported;
                };
                let Ok(layout) = ty.layout() else {
                    return MatPlan::Unsupported;
                };
                let FieldsShape::Array { stride, .. } = layout.shape().fields else {
                    return MatPlan::Unsupported;
                };
                MatPlan::Array {
                    elem: Box::new(self.mat_plan(&elem, depth + 1, active)),
                    stride: stride.bytes() as u64,
                    count: count as u64,
                }
            }
            TyKind::RigidTy(RigidTy::Adt(def, args)) => {
                self.mat_plan_adt(&def, &args, ty, depth, active)
            }
            _ => MatPlan::Unsupported,
        }
    }

    /// The materialisation plan of an ADT instance (see
    /// `mat_plan`), mirroring `decode_adt`'s reading of the
    /// layout: field offsets are the layout's, and the enum tag
    /// placement is the layout's tag encoding.
    fn mat_plan_adt(
        &mut self,
        def: &AdtDef,
        args: &GenericArgs,
        ty: &rustc_public::ty::Ty,
        depth: usize,
        active: &mut HashSet<rustc_public::ty::Ty>,
    ) -> sparagmos_interp::ir::MatPlan {
        use sparagmos_interp::ir::{MatField, MatPlan, MatTag};
        let Ok(layout) = ty.layout() else {
            return MatPlan::Unsupported;
        };
        let shape = layout.shape();
        let size = shape.size.bytes() as u64;
        match def.kind() {
            AdtKind::Struct => {
                let Some(variant) = def.variant(VariantIdx::to_val(0)) else {
                    return MatPlan::Unsupported;
                };
                if variant.fields().is_empty() {
                    return MatPlan::Struct {
                        size,
                        fields: Vec::new(),
                    };
                }
                let Some(offsets) = arbitrary_offsets(ty) else {
                    return MatPlan::Unsupported;
                };
                match self.mat_plan_fields(&variant, args, &offsets, depth, active) {
                    Some(fields) => MatPlan::Struct { size, fields },
                    None => MatPlan::Unsupported,
                }
            }
            AdtKind::Enum => match &shape.variants {
                VariantsShape::Single { index } => {
                    let idx = index.to_index();
                    let mut variants = Vec::new();
                    for i in 0..def.num_variants() {
                        if i == idx {
                            let variant = match def.variant(VariantIdx::to_val(i)) {
                                Some(v) => v,
                                None => return MatPlan::Unsupported,
                            };
                            if variant.fields().is_empty() {
                                variants.push(Vec::new());
                                continue;
                            }
                            let Some(offsets) = arbitrary_offsets(ty) else {
                                return MatPlan::Unsupported;
                            };
                            match self.mat_plan_fields(&variant, args, &offsets, depth, active) {
                                Some(fields) => variants.push(fields),
                                None => return MatPlan::Unsupported,
                            }
                        } else {
                            variants.push(Vec::new());
                        }
                    }
                    MatPlan::Enum {
                        size,
                        tag: MatTag::Single(idx as u32),
                        variants,
                    }
                }
                VariantsShape::Multiple {
                    tag,
                    tag_encoding,
                    tag_field,
                    variants,
                } => {
                    let mat_tag = match tag_encoding {
                        TagEncoding::Direct => {
                            let Some((tsize, signed)) = tag_scalar_size_signed(tag) else {
                                return MatPlan::Unsupported;
                            };
                            let Some(offsets) = arbitrary_offsets(ty) else {
                                return MatPlan::Unsupported;
                            };
                            let Some(tag_off) = offsets.get(*tag_field).copied() else {
                                return MatPlan::Unsupported;
                            };
                            let mut discriminants = Vec::new();
                            for i in 0..def.num_variants() {
                                let d = def.discriminant_for_variant(VariantIdx::to_val(i));
                                match discr_value(&d) {
                                    Some(v) => discriminants.push(v),
                                    None => return MatPlan::Unsupported,
                                }
                            }
                            MatTag::Direct {
                                offset: tag_off as u64,
                                size: tsize as u8,
                                signed,
                                discriminants,
                            }
                        }
                        TagEncoding::Niche {
                            untagged_variant,
                            niche_variants,
                            niche_start,
                        } => {
                            let Some((tsize, _)) = tag_scalar_size_signed(tag) else {
                                return MatPlan::Unsupported;
                            };
                            let Some(offsets) = arbitrary_offsets(ty) else {
                                return MatPlan::Unsupported;
                            };
                            let Some(tag_off) = offsets.get(*tag_field).copied() else {
                                return MatPlan::Unsupported;
                            };
                            MatTag::Niche {
                                offset: tag_off as u64,
                                size: tsize as u8,
                                untagged: untagged_variant.to_index() as u32,
                                niche_start: *niche_start,
                                start: niche_variants.start().to_index() as u32,
                                end: niche_variants.end().to_index() as u32,
                            }
                        }
                    };
                    let mut variant_fields = Vec::new();
                    for (i, vshape) in variants.iter().enumerate() {
                        let variant = match def.variant(VariantIdx::to_val(i)) {
                            Some(v) => v,
                            None => return MatPlan::Unsupported,
                        };
                        let offsets: Vec<usize> =
                            vshape.offsets.iter().map(|o| o.bytes()).collect();
                        match self.mat_plan_fields(&variant, args, &offsets, depth, active) {
                            Some(fields) => variant_fields.push(fields),
                            None => return MatPlan::Unsupported,
                        }
                    }
                    MatPlan::Enum {
                        size,
                        tag: mat_tag,
                        variants: variant_fields,
                    }
                }
                VariantsShape::Empty => MatPlan::Unsupported,
            },
            AdtKind::Union => {
                let Some(variant) = def.variant(VariantIdx::to_val(0)) else {
                    return MatPlan::Unsupported;
                };
                let fields = variant
                    .fields()
                    .iter()
                    .map(|f| {
                        let fty = f.ty_with_args(args);
                        self.mat_plan(&fty, depth + 1, active)
                    })
                    .collect();
                MatPlan::Union { size, fields }
            }
        }
    }

    /// The materialisation plans of one variant's fields at the
    /// given byte offsets (see `mat_plan`).
    fn mat_plan_fields(
        &mut self,
        variant: &rustc_public::ty::VariantDef,
        args: &GenericArgs,
        offsets: &[usize],
        depth: usize,
        active: &mut HashSet<rustc_public::ty::Ty>,
    ) -> Option<Vec<sparagmos_interp::ir::MatField>> {
        let fields = variant.fields();
        if fields.len() != offsets.len() {
            return None;
        }
        let mut out = Vec::new();
        for (field, off) in fields.iter().zip(offsets) {
            let fty = field.ty_with_args(args);
            out.push(sparagmos_interp::ir::MatField {
                offset: *off as u64,
                plan: self.mat_plan(&fty, depth + 1, active),
            });
        }
        Some(out)
    }

    /// Decode the allocation region at `offset` in `alloc` as a
    /// type-erased block (see `sm::ConstValue::ErasedAlloc`):
    /// the region's bytes verbatim, plus every provenance
    /// relocation inside it decoded type-free by
    /// `decode_erased_slot`. `None` when a relocation cannot be
    /// decoded type-free; the caller then keeps the declared
    /// type's decode.
    pub(crate) fn decode_erased_alloc(
        &mut self,
        alloc: &Allocation,
        offset: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        if depth > MAX_CONST_DEPTH {
            return None;
        }
        let tail = alloc.bytes.get(offset..)?;
        if !self.spend_const_budget(tail.len()) {
            return None;
        }
        let bytes = tail.to_vec();
        let mut relocs = Vec::new();
        for (o, _) in &alloc.provenance.ptrs {
            let o = *o as usize;
            if o < offset {
                continue;
            }
            let v = self.decode_erased_slot(alloc, o, depth + 1)?;
            relocs.push(((o - offset) as u32, v));
        }
        Some(sm::ConstValue::ErasedAlloc { bytes, relocs })
    }

    /// Decode the pointer slot at `offset` in `alloc` without a
    /// declared pointee type: the slot's value as a reference to
    /// whatever its provenance names, with a memory pointee
    /// preserved as an erased block. A static's pointee is its
    /// initializer allocation's block: reads through a later
    /// materialisation are exact, though the block is a
    /// snapshot, not the static's store. Mirrors `decode_ref`'s
    /// provenance handling.
    fn decode_erased_slot(
        &mut self,
        alloc: &Allocation,
        offset: usize,
        depth: usize,
    ) -> Option<sm::ConstValue> {
        if depth > MAX_CONST_DEPTH {
            return None;
        }
        let prov = prov_at(alloc, offset);
        let Some(prov) = prov else {
            let addr = read_uint_at(alloc, offset, PTR_SIZE)?;
            if addr == 0 {
                return Some(sm::ConstValue::NullPtr);
            }
            return Some(sm::ConstValue::IntPtr(addr as u64));
        };
        let target_off = read_uint_at(alloc, offset, PTR_SIZE)? as usize;
        match self.resolve_alloc(prov) {
            ResolvedAlloc::Memory(target) => {
                let block = self.decode_erased_alloc(&target, target_off, depth + 1)?;
                Some(sm::ConstValue::Ref(Box::new(block)))
            }
            ResolvedAlloc::Other(mir::alloc::GlobalAlloc::Static(def)) => {
                // As in `decode_ref`: an extern-item-impl
                // declaration's storage is its implementation's.
                let def = if static_is_foreign(&def) {
                    match eii_impl_static(&def) {
                        Some(impl_def) => impl_def,
                        None => return None,
                    }
                } else {
                    def
                };
                let init = self.static_initializer(&def)?;
                let block = self.decode_erased_alloc(&init, target_off, depth + 1)?;
                Some(sm::ConstValue::Ref(Box::new(block)))
            }
            ResolvedAlloc::Other(mir::alloc::GlobalAlloc::Function(inst)) => {
                self.fn_pointer_value(&inst)
            }
            ResolvedAlloc::Other(mir::alloc::GlobalAlloc::TypeId { ty: tid_ty }) => {
                let (lo, _) = type_id_segments(&tid_ty)?;
                Some(sm::ConstValue::IntPtr(lo))
            }
            ResolvedAlloc::Other(mir::alloc::GlobalAlloc::VTable(vty, trait_ref)) => {
                let id = vtable_id(&vty, &trait_ref);
                self.note_vtable_identity(id, &vty, &trait_ref);
                Some(sm::ConstValue::VTablePtr(id))
            }
            // `resolve_alloc` returns memory allocations in
            // the `Memory` variant, never wrapped in `Other`.
            ResolvedAlloc::Other(mir::alloc::GlobalAlloc::Memory(_)) => None,
            // The work budget refused the materialisation:
            // the constant stays undecoded.
            ResolvedAlloc::Refused => None,
        }
    }

    /// Whether a reference decoded with declared pointee type
    /// `inner` is type-erased: the declared type is zero-sized
    /// while the region of `target` it denotes (from
    /// `target_off` on) is not empty, so the pointer was stored
    /// through an erased slot and its real type is named only
    /// by later casts.
    pub(crate) fn pointee_is_erased(
        &self,
        inner: &rustc_public::ty::Ty,
        target: &Allocation,
        target_off: usize,
    ) -> bool {
        if std::env::var_os("NT_DISABLE_ERASED").is_some() {
            return false;
        }
        if target_off >= target.bytes.len() {
            return false;
        }
        // A genuinely unsized pointee (a `str`-tailed struct is
        // the canonical case) is not erased: its typed decode
        // reads the trailing field to the end of the allocation,
        // and the guest projects the value's fields directly,
        // with no cast that could materialise a block. A
        // sizedness the walk cannot prove keeps the typed
        // decode too.
        if ty_sizedness(inner, None, 0) != Sizedness::Sized {
            return false;
        }
        match inner.layout() {
            Ok(layout) => layout.shape().size.bytes() == 0,
            Err(_) => false,
        }
    }
}
