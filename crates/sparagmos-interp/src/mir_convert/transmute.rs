//! Converter: transmute casts.


use sparagmos_mir as mir;

use crate::ir;
use crate::value::{Value};

use super::adt_fields::*;
use super::converter::*;
use super::helpers::*;
use super::program::*;


impl<'a> Converter<'a> {
    // ----- transmute casts ---------------------------------------------------

    /// The field of a struct that carries its pointer chain:
    /// the unique field whose type is not a zero-sized type
    /// with a uniquely determined value. The remaining fields
    /// are fills in the sense of [`Self::wrapper_shape`]:
    /// they carry no information, so the chain field alone is
    /// the struct's payload. `Some((index, ty))`, or `None`
    /// when the ADT is not a single-variant struct or has
    /// zero / more than one non-fill field.
    fn chain_field(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
    ) -> Option<(usize, mir::Ty)> {
        let shape = self.adt_shapes.get(def)?;
        if shape.kind != mir::AdtShapeKind::Struct || shape.variants.len() != 1 {
            return None;
        }
        let fields = self.instantiated_field_tys(def, args, 0)?;
        let mut chain: Option<(usize, mir::Ty)> = None;
        for (i, fty) in fields.iter().enumerate() {
            if self.zst_value(fty, 0).is_some() {
                continue;
            }
            if chain.is_some() {
                return None;
            }
            chain = Some((i, fty.clone()));
        }
        chain
    }

    /// The operand denoting the thin pointer at the bottom of
    /// a chain of ADTs (the `NonNull` shape:
    /// `NonNull { pointer: NonZero<*const T> }` — the payload
    /// hides behind the `NonZero` newtype, one layer deeper
    /// than the transparent wrappers `wrapper_shape` matches,
    /// whose payload field must be the pointer itself). Each
    /// layer must be a struct with exactly one non-fill field
    /// (see [`Self::chain_field`]): the owning pointers are
    /// such chains carrying extra zero-sized fields — `Box`
    /// is a `Unique` plus an allocator, and `Unique` is a
    /// `NonNull` plus a `PhantomData`. The chain ends at the
    /// first field that is itself a thin pointer. `None` for
    /// any other shape.
    fn ptr_chain_operand(
        &self,
        body: &mir::Body,
        operand: &mir::Operand,
        v: &ir::Operand,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
    ) -> Option<ir::Operand> {
        let (index, field_ty) = self.chain_field(def, args)?;
        let field_v =
            self.field_operand(body, operand, v, index as u32, &field_ty)?;
        if is_thin_ptr_ty(&field_ty) {
            return Some(field_v);
        }
        let mir::Ty::Adt(fdef, fargs) = &field_ty else {
            return None;
        };
        // The next layer's operand: the same place extended by
        // the field projection (a constant operand stays a
        // constant: `field_operand` reads its fields from the
        // converted value instead).
        let next_operand = match operand.place() {
            Some(place) => {
                let place =
                    place.clone().field(index as u32, field_ty.clone());
                match operand {
                    mir::Operand::Move(_) => mir::Operand::Move(place),
                    _ => mir::Operand::Copy(place),
                }
            }
            None => operand.clone(),
        };
        self.ptr_chain_operand(body, &next_operand, &field_v, fdef, fargs)
    }

    /// The thin pointer type at the bottom of the ADT's pointer
    /// chain (see [`Self::ptr_chain_operand`]): the type of the
    /// pointer the chain's fields ultimately hold. `None` when
    /// the chain does not bottom out at a thin pointer.
    fn ptr_chain_bottom_ty(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
    ) -> Option<mir::Ty> {
        let (_, field_ty) = self.chain_field(def, args)?;
        if is_thin_ptr_ty(&field_ty) {
            return Some(field_ty);
        }
        let mir::Ty::Adt(fdef, fargs) = &field_ty else {
            return None;
        };
        self.ptr_chain_bottom_ty(fdef, fargs)
    }

    /// The field index at which the pointer chain of the
    /// ADT `def` (see [`Self::chain_field`]) enters its first
    /// layer, when a thin pointer transmuted into the ADT
    /// can be rebuilt faithfully: every layer is a
    /// single-payload struct whose chain field is either a
    /// thin pointer or the next layer of the chain — the
    /// owning-pointer shape (`Box` is a `Unique` plus an
    /// allocator, a `Unique` is a `NonNull` plus a
    /// `PhantomData`, a `NonNull` is a `NonZero` over the
    /// pointer itself) — and every layer below the first
    /// chains through its field 0. The first layer's index
    /// is the one the rebuilt value must place the pointer
    /// at (see the reverse-transmute arm in
    /// `convert_transmute`); the deeper layers are read
    /// through the stored pointer itself, the model's
    /// wrapper-chain convention (a field-0 step on a pointer
    /// value yields the pointer), so only field-0 layers may
    /// sit below the first. A chain entering a deeper layer
    /// at a non-zero field would need an aggregate nested
    /// inside an operand, which the IR does not express:
    /// `None` here keeps such a transmute an explicit
    /// unsupported instead of building a value whose fields
    /// were never constructed.
    fn ptr_chain_entry_index(&self, def: &mir::AdtDefId, args: &[mir::Ty]) -> Option<usize> {
        let (entry, mut field_ty) = self.chain_field(def, args)?;
        // A type-level walk over a finite chain of wrapper
        // types; the depth cap only guards against a
        // (nonsensical) self-nesting wrapper type.
        for _ in 0..32 {
            if is_thin_ptr_ty(&field_ty) {
                return Some(entry);
            }
            let mir::Ty::Adt(def, args) = &field_ty else {
                return None;
            };
            let (index, next_ty) = self.chain_field(def, args)?;
            if index != 0 {
                return None;
            }
            field_ty = next_ty;
        }
        None
    }

    /// Whether both types are pointers (a reference or a raw
    /// pointer) to byte-string DSTs: `str`, a `[u8]` slice,
    /// `std::ffi::OsStr` (or its sys representation type
    /// `std::sys::os_str::bytes::Slice`, the pointee the
    /// `OsStr` constructors actually transmute to) or
    /// `std::path::Path`. All of them are a
    /// byte string with a length metadata half, and in this
    /// value model a reference to any of them is the byte
    /// string itself (see the projection pun for the wrapper
    /// chain in `interp`), so a transmute between any two of
    /// these pointer types reinterprets nothing.
    fn is_byte_string_ptr_pair(&self, s: &mir::Ty, t: &mir::Ty) -> bool {
        fn pointee(ty: &mir::Ty) -> Option<&mir::Ty> {
            match ty {
                mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => Some(inner),
                _ => None,
            }
        }
        let byte_string = |ty: &mir::Ty| -> bool {
            match ty {
                mir::Ty::Str => true,
                mir::Ty::Slice(elem) => {
                    matches!(**elem, mir::Ty::Uint(mir::UintTy::U8))
                }
                mir::Ty::Adt(def, _) => matches!(
                    self.names
                        .get(def)
                        .or_else(|| self.def_names.get(def))
                        .map(String::as_str),
                    Some(
                        "std::ffi::OsStr"
                            | "std::sys::os_str::bytes::Slice"
                            | "std::path::Path"
                    )
                ),
                _ => false,
            }
        };
        match (pointee(s), pointee(t)) {
            (Some(a), Some(b)) => byte_string(a) && byte_string(b),
            _ => false,
        }
    }

    /// Whether a pointee type is one of the byte-string DST
    /// structs: `std::ffi::CStr`, `std::ffi::OsStr` (or its
    /// sys representation type), `std::path::Path`. Each is a
    /// byte string with a length metadata half, and its value
    /// in this model is the byte string itself (the same
    /// family `is_byte_string_ptr_pair` recognises), so a
    /// pointer to one is a fat pointer.
    pub(super) fn pointee_is_byte_string_dst(&self, ty: &mir::Ty) -> bool {
        match ty {
            mir::Ty::Adt(def, _) => matches!(
                self.names
                    .get(def)
                    .or_else(|| self.def_names.get(def))
                    .map(String::as_str),
                Some(
                    "std::ffi::CStr"
                        | "core::ffi::CStr"
                        | "std::ffi::OsStr"
                        | "std::sys::os_str::bytes::Slice"
                        | "std::path::Path"
                )
            ),
            _ => false,
        }
    }

    /// The rvalue for a `Transmute` cast from a byte-subslice
    /// place to an integer: the compiler's transmute validity
    /// checks (inserted under debug assertions) reinterpret
    /// the input through its byte-array form and read a chunk
    /// of it back as an integer — `copy _33[0..2] as u16`,
    /// where `_33` is the input transmuted to
    /// `[MaybeUninit<u8>; 4]`. The operand must be a place
    /// whose last projection is a literal subslice (`from_end`
    /// false) of exactly the integer's byte width, over a
    /// byte element (`u8` / `i8`, or `MaybeUninit` of one);
    /// the cast is then the array-of-bytes reinterpretation
    /// of the subslice's value, whose elements the place
    /// read yields in order.
    fn subslice_to_int_rvalue(
        &self,
        operand: &mir::Operand,
        v: &ir::Operand,
        src: &Option<mir::Ty>,
        target: &mir::Ty,
    ) -> Option<ir::Rvalue> {
        let (signed, bits) = match target {
            mir::Ty::Int(it) => (true, it.bits()),
            mir::Ty::Uint(ut) => (false, ut.bits()),
            _ => return None,
        };
        let mir::Ty::Slice(elem) = src.as_ref()? else {
            return None;
        };
        let byte_shape = ir::ScalarShape::Int {
            bits: 8,
            signed: false,
        };
        let elem_shape = match elem.as_ref() {
            mir::Ty::Uint(mir::UintTy::U8)
            | mir::Ty::Int(mir::IntTy::I8) => byte_shape,
            mir::Ty::Adt(def, args)
                if self.is_maybe_uninit(def)
                    && args.len() == 1
                    && matches!(
                        args[0],
                        mir::Ty::Uint(mir::UintTy::U8)
                            | mir::Ty::Int(mir::IntTy::I8)
                    ) =>
            {
                ir::ScalarShape::AdtField {
                    index: 0,
                    count: 1,
                    fill: Vec::new(),
                    inner: Box::new(byte_shape),
                }
            }
            _ => return None,
        };
        let place = match operand {
            mir::Operand::Copy(p) | mir::Operand::Move(p) => p,
            _ => return None,
        };
        let Some(mir::PlaceElem::Subslice {
            from,
            to,
            from_end,
        }) = place.projection.last()
        else {
            return None;
        };
        if *from_end {
            return None;
        }
        let len = to.checked_sub(*from)?;
        if len == 0 || len * 8 != u64::from(bits) {
            return None;
        }
        Some(ir::Rvalue::Reinterpret(
            v.clone(),
            ir::ScalarShape::Array(Box::new(elem_shape), len),
            ir::ScalarShape::Int { bits, signed },
        ))
    }

    /// The transmute between a pointer to a slice (or an
    /// array) of a fieldless enum and the same pointer shape
    /// over an integer, in either direction: the elements
    /// reinterpret through the enum discriminant table (see
    /// `ir::Rvalue::SliceEnumToInt`), which the pointer-identity
    /// arms of `convert_transmute` cannot express. `None` for
    /// every other pair of types.
    fn slice_enum_transmute(
        &self,
        v: &ir::Operand,
        src: &Option<mir::Ty>,
        target: &mir::Ty,
    ) -> Option<ir::Rvalue> {
        fn elem_of(ty: &mir::Ty) -> Option<&mir::Ty> {
            match ptr_pointee(ty)? {
                mir::Ty::Slice(elem) => Some(elem),
                mir::Ty::Array(elem, _) => Some(elem),
                _ => None,
            }
        }
        let src_elem = elem_of(src.as_ref()?)?;
        let dst_elem = elem_of(target)?;
        match (src_elem, dst_elem) {
            (mir::Ty::Adt(def, _), mir::Ty::Int(_) | mir::Ty::Uint(_)) => {
                let table = self.fieldless_enum_discriminants(def)?;
                let int_ty = self.convert_ty(dst_elem, "cast target").ok()?;
                Some(ir::Rvalue::SliceEnumToInt(v.clone(), table, int_ty))
            }
            (mir::Ty::Int(_) | mir::Ty::Uint(_), mir::Ty::Adt(def, _)) => {
                let table = self.fieldless_enum_discriminants(def)?;
                Some(ir::Rvalue::SliceIntToEnum(v.clone(), table))
            }
            _ => None,
        }
    }

    /// Convert a `CastKind::Transmute` rvalue: `operand` is the
    /// MIR operand, `v` its converted IR form, `target` the cast's
    /// target type. Only the shapes this value model can
    /// reinterpret exactly are converted:
    ///
    /// * pointer -> integer (the synthetic address) and the
    ///   reverse (an address-only reference);
    /// * thin pointer -> thin pointer, including reference ->
    ///   reference across pointee types (`&Waker` ->
    ///   `&LocalWaker`): the value is the pointer itself, the
    ///   pointee type lives in the destination's declared type;
    /// * fieldless enum <-> integer, through the baked
    ///   discriminant table;
    /// * thin pointer <-> transparent wrapper struct (one
    ///   thin-pointer payload field, the remaining fields
    ///   zero-sized with unique values): `NonNull`, `Pin` and the
    ///   coroutine machinery's `ResumeTy` chain transmute through
    ///   these;
    /// * thin-pointer transparent wrapper <-> integer: the
    ///   payload pointer's synthetic address, wrapped back up
    ///   on the way back (`NonNull<u8>` -> `usize` is the
    ///   transmute `fmt`'s statically-known-string check
    ///   performs);
    /// * fieldless enum (or a one-field wrapper around one,
    ///   `mem::Alignment`) -> thin-pointer transparent wrapper:
    ///   the discriminant as the payload pointer's address
    ///   (`NonNull::dangling`);
    /// * zero-sized type -> zero-sized type whose value is
    ///   uniquely determined by its type;
    /// * scalar-shaped -> scalar-shaped of the same total width
    ///   ([`ir::Rvalue::Reinterpret`]): integers, floats, `bool`
    ///   and `char` in any combination, arrays of scalars,
    ///   one-payload-field ADTs over those, multi-field structs
    ///   whose scalar fields sit at their recorded layout
    ///   offsets (a packed struct transmuted to a byte array),
    ///   direct-tag enums whose variants' fields sit at theirs,
    ///   and niche-encoded enums over a scalar or thin-pointer
    ///   payload (`Option<NonNull<T>>`), over a fieldless-enum
    ///   payload, or over a wider payload field holding the
    ///   niche scalar in a sub-window (an enum whose payload
    ///   is itself an enum), or over several payload fields
    ///   packed around the niche (the storage shape of
    ///   `DefKind`, which the compiler's query-result erasure
    ///   transmutes through a byte array) — the bit-level
    ///   reinterpretation `transmute` means there;
    /// * pointer <-> pointer over byte-string DST pointees
    ///   (`str`, `[u8]`, `OsStr`, `Path` in any combination);
    /// * fat pointer <-> fat pointer, and fat pointer <->
///   transparent wrapper struct (one fat-pointer payload
    ///   field, the remaining fields zero-sized with unique
    ///   values): `NonNull<str>` and its kin;
    /// * one-field struct over a fieldless enum <-> integer:
    ///   the discriminant of the field's value (`mem::Alignment`
    ///   -> `usize`), and the reverse (integer -> the wrapper:
    ///   the integer is the wrapped enum's discriminant);
    /// * one-field struct over an array <-> integer of the
    ///   same total width: the bit-stream reinterpretation
    ///   over the array's leaves (the `Simd<T, N>` -> `u16`
    ///   an inlined small-array integer comparison produces,
    ///   and the reverse construction); a storage length the
    ///   type representation does not carry takes its element
    ///   count from the integer's width;
    /// * fieldless enum <-> the one-field transparent struct
    ///   wrapping it (`mem::Alignment` <-> its enum): the enum
    ///   value itself, wrapped into / read out of the struct's
    ///   single field.
    /// * one-payload wrapper <-> its payload, for a payload of
    ///   any type: a one-field struct whose only field has
    ///   exactly the other side's type, and `MaybeUninit<T>`
    ///   <-> `T` (the payload sits at field 0 of the union
    ///   value in this model) -- the `assume_init` transmute
    ///   an inlined `mem::zeroed` performs under optimization,
    ///   and its reverse.
    ///
    /// * function pointer -> function pointer: the value is
    ///   the function identity, independent of the signature it
    ///   is viewed through (an inlined `fmt::Argument`
    ///   construction rewrites the display fn pointer to the
    ///   erased unsafe signature the struct stores);
    /// * array -> array of the same length across an element
    ///   wrapper boundary (`[T; N]` <-> `[MaybeUninit<T>; N]`,
    ///   or the `UnsafeCell` twin): the cast maps the wrapper's
    ///   value representation over the elements
    ///   ([`ir::Rvalue::WrapArrayElems`] /
    ///   [`ir::Rvalue::UnwrapArrayElems`]). An inlined by-value
    ///   array iterator construction performs this cast on the
    ///   backing array;
    /// * array -> array of the same length whose elements
    ///   are scalar-shaped of the same nonzero width, at
    ///   lengths whose total width exceeds the 128-bit
    ///   whole-array shape budget (so the generic path above
    ///   cannot see the pair): the reinterpretation the
    ///   generic path computes element by element for the
    ///   lengths that fit, carried by the elements' shapes
    ///   instead (`[ *mut T; 63 ]` -> `[Atomic<*mut T>; 63]`,
    ///   the `thread_local` crate's bucket table);
    /// * array <-> array (or one-payload wrapper over an
    ///   array) whose leaf streams have the same total width,
    ///   at any total width ([`ir::Rvalue::Reinterpret`] over
    ///   the stream shapes): the width-crossing regrouping
    ///   (`[i16; 50]` -> `[f32; 25]`) and the long wrapped
    ///   array (`Mu64([u64; 4])` -> `[u8; 32]`); a one-field
    ///   ADT over an array whose length the type
    ///   representation does not carry (a `Simd<T, N>`
    ///   storage field) takes its element count from the
    ///   other side, the pair being size-equal by the
    ///   transmute's own validity;
    /// * fat pointer <-> pointer to a DST-wrapper struct
    ///   (an ADT whose last field is the slice / `str` /
    ///   `dyn` tail the metadata describes): the fat value
    ///   itself under the other declared type;
    /// * (thin pointer, length) tuple <-> fat reference to
    ///   a DST-wrapper struct: the tuple's two recorded
    ///   words are the fat pointer's two words;
    /// * scalar or thin pointer <-> union: the bit
    ///   reinterpretation between the leaf shape and the
    ///   storage shape of the union field whose own shape
    ///   is the leaf's;
    /// * a byte-string constant in pointer position (it
    ///   carries no place type) -> a thin pointer or a
    ///   transparent wrapper over one: const propagation
    ///   replaced the reference to the static byte string with
    ///   the allocation's bytes themselves (the `fmt::Arguments`
    ///   template), so the reference to those bytes is formed
    ///   first and the pointer decisions above run on it.
    ///
    /// Every other shape — aggregate <-> aggregate byte
    /// reinterpretation whose fields the recorded layouts
    /// cannot account for scalar by scalar, fat pointers
    /// outside the fat-pointer shape, enums whose encoding
    /// is neither a recorded direct tag nor a recorded
    /// niche — stays [`ir::Rvalue::Unsupported`]:
    /// the model has no flat byte memory to reinterpret through.
    pub(super) fn convert_transmute(
        &self,
        body: &mir::Body,
        operand: &mir::Operand,
        v: &ir::Operand,
        target: &mir::Ty,
    ) -> ir::Rvalue {
        fn unsupported() -> ir::Rvalue {
            ir::Rvalue::Unsupported("Cast Transmute".to_string())
        }
        // A constant operand carries no place type. When the
        // constant is a decoded ADT value, the type its
        // producer recorded for it stands in for the place
        // type: a const-propagated `core` body holds
        // transmutes of folded constants whose reading
        // depends on the source type, which the value's
        // structure alone does not name (see the driver's
        // `constant_adt_ty`). A scalar constant keeps the
        // `None` reading: its shape comes from the converted
        // constant itself (`constant_scalar_shape`), which is
        // the form the value actually carries -- a `1u8`
        // transmuted to `bool` arrives as a bool constant,
        // not as the integer its literal spelled.
        let src = self
            .operand_place_ty(body, operand)
            .or_else(|| match constant_recorded_ty(operand) {
                Some(t @ mir::Ty::Adt(..)) => Some(t),
                _ => None,
            });
        // A byte-string constant in pointer position carries no
        // place type: const propagation replaced the reference to
        // the static byte string with the allocation's bytes
        // themselves (the shape an inlined `fmt::Arguments::new`
        // gives the template field: the bytes transmuted to the
        // `NonNull<u8>` the struct stores). The pointer the
        // transmute means is the reference to those bytes, so
        // when the target is a thin pointer or a transparent
        // wrapper over one, form that reference and record the
        // pointer type it denotes; the ordinary pointer decisions
        // below then run on the wrapped operand.
        let bytes_ref;
        let bytes_ty;
        let (src, v) = match v {
            ir::Operand::Constant(ir::Constant::Bytes(b))
                if is_thin_ptr_ty(target)
                    || matches!(
                        target,
                        mir::Ty::Adt(def, args)
                            if self.wrapper_shape(def, args).is_some()
                    ) =>
            {
                bytes_ref = ir::Operand::Constant(ir::Constant::Ref(
                    Box::new(ir::Constant::Bytes(b.clone())),
                ));
                bytes_ty = Some(mir::Ty::RawPtr(
                    Box::new(mir::Ty::Uint(mir::UintTy::U8)),
                    mir::Mutability::Not,
                ));
                (bytes_ty, &bytes_ref)
            }
            _ => (src, v),
        };
        // A constant integer operand carries no place type; the
        // converted operand itself is the evidence for its kind.
        let src_is_int = matches!(src, Some(mir::Ty::Int(_) | mir::Ty::Uint(_)))
            || matches!(
                v,
                ir::Operand::Constant(ir::Constant::Int(_) | ir::Constant::Uint(_))
            );
        // A `ZeroSized` constant operand carries no value at
        // all: transmuting it into a zero-sized type whose value
        // is uniquely determined by the type yields that value,
        // whatever the (also zero-sized) source type was.
        if let mir::Operand::Constant(c) = operand {
            if matches!(
                c.literal,
                mir::ConstKind::Value(mir::ConstValue::ZeroSized)
            ) {
                if let Some(tv) = self.zst_value(target, 0) {
                    if let Some(k) = value_to_constant(&tv) {
                        return ir::Rvalue::Use(ir::Operand::Constant(k));
                    }
                }
            }
        }
        // A string constant transmuted to a byte-slice pointer:
        // the constant's value is the string itself, which the
        // model's consumers read as bytes as readily as as text
        // (`PtrMetadata` reports its byte length, byte reads go
        // through the same storage), so the value carries over.
        if matches!(v, ir::Operand::Constant(ir::Constant::Str(_)))
            && is_fat_ptr_ty(target)
            && matches!(ptr_pointee(target), Some(t) if is_str_or_byte_slice(t))
        {
            return ir::Rvalue::Use(v.clone());
        }
        // The scalar-shape reinterpretation, decided from the
        // two types alone: when both sides flatten to the same
        // nonzero bit width, the cast is a `Reinterpret`. The
        // specialised arms below still take the shapes whose
        // result this does not reproduce (fieldless enums carry
        // no scalar shape, and the bare pointer pairs are claimed
        // by their own arms before this one is reached).
        let shape_to = self.scalar_shape(target, 0);
        let shape_from = match &src {
            Some(s) => self.scalar_shape(s, 0),
            // A constant operand carries no place type; its
            // shape comes from the constant itself (see
            // `constant_scalar_shape`).
            None => shape_to
                .as_ref()
                .and_then(|t| self.constant_scalar_shape(v, t.bits())),
        };
        // A transmute between two ADTs of identical scalar
        // shape whose shape carries a thin pointer: the value's
        // only payload is the pointer itself, whose place
        // identity a bit reinterpretation would reduce to an
        // address-only reference that no longer reads the
        // pointee (the same reason the bare-pointer pairs below
        // stay with their specialised arms). The two sides
        // interpret the stored value identically -- the wrapper
        // layers add no representation of their own in this
        // model, and the pointee type lives in the destination's
        // declared type -- so the value carries over unchanged.
        // `Box<MaybeUninit<T>>` -> `Box<T>`, the transmute
        // `Box::assume_init` performs, is the canonical case:
        // reinterpretation rebuilt the box aggregate around the
        // pointer's synthetic address, and every later read of
        // the pointee missed the allocation.
        let ptr_identity = matches!(
            (&src, target),
            (Some(mir::Ty::Adt(..)), mir::Ty::Adt(..))
        ) && match (&shape_from, &shape_to) {
            (Some(from), Some(to)) => {
                from == to && scalar_shape_has_ptr_leaf(from)
            }
            _ => false,
        } && !self.payload_prefix_crossing(&src, target);
        let generic = {
            let to = shape_to.clone();
            let from = shape_from.clone();
            match (from, to) {
                (Some(from), Some(to))
                    if from.bits() == to.bits() && from.bits() > 0 =>
                {
                    // A bare pointer on one side and a wrapper
                    // ADT on the other stays with the specialised
                    // arms below: they carry the pointer value
                    // itself (its place identity) across, where a
                    // bit reinterpretation would reduce it to an
                    // address-only reference that no longer reads
                    // the pointee. A niche-encoded ADT is the
                    // exception: no specialised arm reads a
                    // variant-dependent payload, so its pointer
                    // pairs go through the reinterpretation.
                    let src_bare_ptr = matches!(
                        src,
                        Some(mir::Ty::RawPtr(..) | mir::Ty::Ref(..))
                    );
                    let target_bare_ptr =
                        is_thin_ptr_ty(target) || is_fat_ptr_ty(target);
                    // Both sides bare pointers stay with the
                    // specialised arms too: they carry the value
                    // itself across (or refuse the wrapper-exiting
                    // crossing), where a bit reinterpretation
                    // would silently change what the pointer
                    // denotes.
                    let identity_pair = (src_bare_ptr && target_bare_ptr)
                        || (src_bare_ptr
                            && matches!(target, mir::Ty::Adt(..))
                            && !matches!(to, ir::ScalarShape::NicheEnum { .. }))
                        || (target_bare_ptr
                            && matches!(src, Some(mir::Ty::Adt(..)))
                            && !matches!(from, ir::ScalarShape::NicheEnum { .. }));
                    if identity_pair || self.payload_prefix_crossing(&src, target) {
                        // A wrapper pair whose payload pointer
                        // crosses a prefix struct's first field
                        // also stays with the specialised arms:
                        // a bit reinterpretation would carry the
                        // payload across verbatim, leaving it
                        // denoting the wrong sub-value (see
                        // `payload_prefix_crossing`).
                        None
                    } else {
                        Some(ir::Rvalue::Reinterpret(v.clone(), from, to))
                    }
                }
                _ => None,
            }
        };
        // A byte-subslice place transmuted to an integer (the
        // compiler's transmute validity checks read the
        // input's bytes this way): the slice source carries
        // no scalar shape, so the generic path above cannot
        // see it.
        if let Some(rv) =
            self.subslice_to_int_rvalue(operand, v, &src, target)
        {
            return rv;
        }
        // One-payload wrapper crossings, decided ahead of the
        // shape arms below: a one-field struct (or a
        // `MaybeUninit`) transmuted to exactly its payload type
        // reads the payload field out, and the reverse
        // transmute wraps the value into the wrapper. The arms
        // below keep the shapes whose payload is not the whole
        // other side -- pointer payloads behind newtype chains
        // and fieldless-enum payloads -- which these checks
        // cannot mistake for this one: they require the field
        // type to be exactly the other side's type.
        if let Some(mir::Ty::Adt(def, args)) = &src {
            if let Some(op) =
                self.payload_field_operand(body, operand, v, def, args, target)
            {
                return ir::Rvalue::Use(op);
            }
        }
        if let mir::Ty::Adt(tdef, targs) = target {
            if let Some(src_ty) = &src {
                if let Some(r) =
                    self.wrap_payload_rvalue(v, src_ty, tdef, targs)
                {
                    return r;
                }
            }
        }
        // The same one-payload wrapper crossings for a
        // one-element tuple: a tuple with a single field is a
        // one-field wrapper whose payload sits at field 0, so a
        // transmute to exactly the field's type reads the field
        // out, and the reverse transmute wraps the value into
        // the tuple. (Tuples carry no scalar shape in this
        // model, so the shape arms above never see these
        // crossings.)
        if let Some(mir::Ty::Tuple(ts)) = &src {
            if ts.len() == 1 && ts[0] == *target {
                if let Some(op) =
                    self.field_operand(body, operand, v, 0, &ts[0])
                {
                    return ir::Rvalue::Use(op);
                }
            }
        }
        if let mir::Ty::Tuple(ts) = target {
            if ts.len() == 1 {
                // A constant operand carries no place type; its
                // declared type stands in for the equality check
                // alone (the value itself is wrapped as it
                // stands, as the ADT wrap above wraps it).
                let src_ty = src
                    .clone()
                    .or_else(|| constant_recorded_ty(operand));
                if let Some(src_ty) = &src_ty {
                    if ts[0] == *src_ty {
                        return ir::Rvalue::AggregateTuple(vec![v.clone()]);
                    }
                }
            }
        }
        // An ADT carrying an integer scalar (a
        // `NonZero<usize>`, or a transparent wrapper chain
        // over an integer) transmuted into a transparent
        // pointer wrapper: the flattened integer is the
        // payload pointer's address -- the transmute
        // `NonNull::without_provenance` performs. A
        // fieldless-enum source (or a wrapper around one)
        // does not take this arm: its shape's leaf is the
        // enum itself, and the dedicated arms below serve it.
        if let (Some(mir::Ty::Adt(def, args)), mir::Ty::Adt(tdef, targs)) =
            (&src, target)
        {
            if let Some(shape) = self.wrapper_shape(tdef, targs) {
                if let Some(from) =
                    self.scalar_shape(&mir::Ty::Adt(*def, args.clone()), 0)
                {
                    if from.bits() == 64 && scalar_leaf_is_int(&from) {
                        return ir::Rvalue::ScalarToPtrWrapped {
                            operand: v.clone(),
                            from,
                            payload_index: shape.payload_index as u32,
                            field_count: shape.field_count,
                            fill: shape
                                .fill
                                .iter()
                                .map(|(i, c)| (*i as u32, c.clone()))
                                .collect(),
                        };
                    }
                }
            }
        }
        if let Some(rv) = self.slice_enum_transmute(&v, &src, target) {
            return rv;
        }
let inner = match (&src, target) {
            // Pointer -> integer: the synthetic address (what the
            // compiler's UB precondition checks use to test a
            // pointer for null / alignment).
            (
                Some(mir::Ty::RawPtr(..) | mir::Ty::Ref(..)),
                mir::Ty::Int(_) | mir::Ty::Uint(_),
            ) => ir::Rvalue::PtrToInt(v.clone()),
            // Thin pointer -> thin pointer crossing an
            // `UnsafeCell` / `MaybeUninit` wrapper boundary in
            // the entering direction (the wrapper -> the type it
            // wraps): the pointer must come to denote the
            // wrapper's inner field, not the wrapper aggregate -
            // the same field-entering rvalue the cast conversion
            // emits for this shape (Lane DP's `PtrField`).
            (Some(s), t)
                if is_thin_ptr_ty(s)
                    && is_thin_ptr_ty(t)
                    && self.cast_wrapper_entry_depth(s, t).is_some() =>
            {
                ir::Rvalue::PtrField(
                    v.clone(),
                    self.cast_wrapper_entry_depth(s, t)
                        .expect("guard checked"),
                )
            }
            // Thin pointer -> thin pointer crossing a prefix
            // boundary in the exiting direction (a struct's
            // first field -> the struct itself): the pointer
            // must come to denote the enclosing aggregate again
            // -- the field-exiting counterpart of the entry arm
            // above (see `cast_exit_depth`).
            (Some(s), t)
                if is_thin_ptr_ty(s)
                    && is_thin_ptr_ty(t)
                    && self.cast_crosses_prefix_field(s, t)
                    && self.cast_exit_depth(s, t).is_some() =>
            {
                ir::Rvalue::PtrFieldBase(
                    v.clone(),
                    self.cast_exit_depth(s, t).expect("guard checked"),
                )
            }
            // Thin pointer -> thin pointer: the value itself.
            // The one crossing still excluded here is the
            // cell-wrapper exiting direction (the wrapped type
            // -> the wrapper; see
            // `pointee_crosses_cell_wrapper`): the pointer would
            // have to come to denote a wrapper aggregate around
            // the field it points into, and a cell wrapper's
            // field is not storage this model can re-enter from
            // the field side, so those stay unsupported instead
            // of silently pointing at the field.
            (Some(s), t)
                if is_thin_ptr_ty(s)
                    && is_thin_ptr_ty(t)
                    && !self.pointee_crosses_cell_wrapper(s, t) =>
            {
                ir::Rvalue::Use(v.clone())
            }
            // Fat pointer -> fat pointer: the value itself,
            // exactly as for thin pointers (the metadata half
            // travels with the value; the pointee type lives in
            // the destination's declared type). The
            // UnsafeCell / MaybeUninit wrapper crossing stays
            // excluded, as for thin pointers: entering a wrapper
            // field is a place change, not a value identity.
            (Some(s), t)
                if is_fat_ptr_ty(s)
                    && is_fat_ptr_ty(t)
                    && !self.pointee_crosses_cell_wrapper(s, t) =>
            {
                ir::Rvalue::Use(v.clone())
            }
            // Fat pointer <-> pointer to a DST-wrapper
            // struct (an ADT whose last field is the
            // unsized tail the metadata describes): the
            // value is the same fat pointer — the data word
            // denotes the struct's storage, whose tail sits
            // at the end of it, and the metadata word counts
            // or names that tail under both readings
            // (`&[u8]` -> `&X` where `struct X([u8])`, and
            // `&Baz` whose tail is a `str`). The pointee
            // type lives in the destination's declared
            // type, as for the plain fat -> fat identity
            // above.
            (Some(s), t)
                if self.fat_tail_meta(s).is_some()
                    && self.fat_tail_meta(s) == self.fat_tail_meta(t)
                    && (matches!(
                        ptr_pointee(s),
                        Some(mir::Ty::Adt(..))
                    ) || matches!(
                        ptr_pointee(t),
                        Some(mir::Ty::Adt(..))
                    )) =>
            {
                ir::Rvalue::Use(v.clone())
            }
            // (thin pointer, length) tuple -> fat reference
            // to a DST-wrapper struct, and the reverse: the
            // tuple's recorded layout is the fat pointer's
            // two words — the data pointer at offset 0 and
            // the length at offset 64 — so the crossing is
            // the reinterpretation between the tuple's
            // Fields shape and the FatPtr shape (the
            // `PackedSized::unsize` transmute, which builds
            // a `&PackedUnsized` from `(self, len)`).
            (Some(mir::Ty::Tuple(_)), t)
                if self.fat_tail_meta(t).is_some()
                    && shape_from
                        .as_ref()
                        .is_some_and(fat_words_fields) =>
            {
                let from = shape_from.clone().expect("guard checked");
                let meta = self.fat_tail_meta(t).expect("guard checked");
                ir::Rvalue::Reinterpret(
                    v.clone(),
                    from,
                    ir::ScalarShape::FatPtr(meta),
                )
            }
            (Some(s), mir::Ty::Tuple(_))
                if self.fat_tail_meta(s).is_some()
                    && shape_to.as_ref().is_some_and(fat_words_fields) =>
            {
                let meta = self.fat_tail_meta(s).expect("guard checked");
                let to = shape_to.clone().expect("guard checked");
                ir::Rvalue::Reinterpret(
                    v.clone(),
                    ir::ScalarShape::FatPtr(meta),
                    to,
                )
            }
            // Pointer <-> pointer where both pointees are
            // byte-string DSTs (`str`, `[u8]`, `OsStr`, `Path`):
            // all four are a byte string with a length metadata
            // half, and their values in this model are the byte
            // string itself (the projection pun reads an `OsStr`
            // / `Path` as its bytes), so the cast reinterprets
            // nothing — the `&[u8]` -> `&OsStr` step inside
            // `Component::as_os_str` is the shape that reaches
            // here, `is_fat_ptr_ty` not counting the DST structs
            // as fat pointees.
            (Some(s), t) if self.is_byte_string_ptr_pair(s, t) => {
                ir::Rvalue::Use(v.clone())
            }
            // Fat pointer between a `str` and a `[u8]` view of the
            // same bytes: both are stored over the same allocation
            // in this model (the `str::as_bytes` builtin relies on
            // the same sharing), so the pair carries over unchanged.
            (Some(s), t)
                if is_fat_ptr_ty(s)
                    && is_fat_ptr_ty(t)
                    && matches!(
                        (ptr_pointee(s), ptr_pointee(t)),
                        (Some(a), Some(b))
                            if is_str_or_byte_slice(a) && is_str_or_byte_slice(b)
                    ) =>
            {
                ir::Rvalue::Use(v.clone())
            }
            // Function pointer -> function pointer: the value
            // is the function identity, which does not depend on
            // the signature it is viewed through, so the value
            // carries over unchanged. This is the transmute the
            // `fmt` argument construction performs when it
            // rewrites a safe `fn(&T, &mut Formatter)` into the
            // type-erased `unsafe fn(NonNull<()>, &mut
            // Formatter)` stored in an `Argument` (inlined into
            // callers at higher opt levels).
            (Some(mir::Ty::FnPtr(..)), mir::Ty::FnPtr(..)) => {
                ir::Rvalue::Use(v.clone())
            }
            // Integer -> thin pointer: an address-only reference.
            (_, t) if src_is_int && is_thin_ptr_ty(t) => {
                ir::Rvalue::IntToPtr(v.clone())
            }
            // ADT -> ADT of identical pointer-carrying shape:
            // the value itself (see `ptr_identity` above).
            (Some(mir::Ty::Adt(..)), mir::Ty::Adt(..)) if ptr_identity => {
                ir::Rvalue::Use(v.clone())
            }
            // Scalar-shaped -> scalar-shaped of the same width:
            // the bit-level reinterpretation (float <-> integer
            // bits, `bool` / `char` <-> integer, scalar arrays,
            // one-payload wrappers over scalars). For the one
            // overlap with the arms below (a one-field struct ->
            // the integer of exactly its field's type) the result
            // is the same value: the field itself.
            (_, _) if generic.is_some() => generic.clone().unwrap(),
            // Scalar or thin pointer -> union (and the
            // reverse): the union's storage shape is the
            // one-payload shape of the field whose own
            // shape is the other side's — a union's fields
            // all describe the same storage, and a union
            // field read reinterprets between them, so the
            // crossing is the bit reinterpretation between
            // the leaf shape and that AdtField shape. No
            // specialised arm carries a value into a union
            // (the wrapper arms serve structs), and the
            // generic path declined this pair precisely
            // because a bare leaf meets an ADT.
            (Some(s), mir::Ty::Adt(def, args))
                if self.adt_is_union(def)
                    && self.union_leaf_pair(def, args, s).is_some() =>
            {
                let (leaf, to) = self
                    .union_leaf_pair(def, args, s)
                    .expect("guard checked");
                ir::Rvalue::Reinterpret(v.clone(), leaf, to)
            }
            (Some(mir::Ty::Adt(def, args)), t)
                if self.adt_is_union(def)
                    && self.union_leaf_pair(def, args, t).is_some() =>
            {
                let (leaf, from) = self
                    .union_leaf_pair(def, args, t)
                    .expect("guard checked");
                ir::Rvalue::Reinterpret(v.clone(), from, leaf)
            }
            // ADT -> integer through a newtype chain over a
            // thin pointer (the `NonNull` shape, whose payload
            // field is a `NonZero<*const T>`): the synthetic
            // address of the pointer at the bottom of the chain
            // (what `NonNull::as_ptr().addr()` and the packed
            // `io::Error` repr decode compute).
            (Some(mir::Ty::Adt(def, args)), mir::Ty::Int(_) | mir::Ty::Uint(_))
                if self
                    .ptr_chain_operand(body, operand, v, def, args)
                    .is_some() =>
            {
                let op = self
                    .ptr_chain_operand(body, operand, v, def, args)
                    .expect("guard checked");
                ir::Rvalue::PtrToInt(op)
            }
            // ADT -> thin pointer through the same chain: the
            // pointer at the bottom of the chain itself (what
            // `NonNull::as_ptr()` computes, and what a
            // `Box<T>` -> `*mut T` transmute reads out of the
            // box's `Unique` field).
            (Some(mir::Ty::Adt(def, args)), t)
                if is_thin_ptr_ty(t)
                    && self
                        .ptr_chain_operand(body, operand, v, def, args)
                        .is_some() =>
            {
                let op = self
                    .ptr_chain_operand(body, operand, v, def, args)
                    .expect("guard checked");
                // The extracted pointer may be retargeted by
                // the transmute itself: a chain holding a
                // pointer to a prefix struct's first field, read
                // back out as a pointer to the struct (or the
                // reverse), denotes the field (or the whole
                // again), not the position it denoted inside
                // the wrapper (see `prefix_retarget`).
                match self
                    .ptr_chain_bottom_ty(def, args)
                    .and_then(|bottom| self.prefix_retarget(&bottom, t))
                {
                    Some((true, depth)) => ir::Rvalue::PtrField(op, depth),
                    Some((false, depth)) => {
                        ir::Rvalue::PtrFieldBase(op, depth)
                    }
                    None => ir::Rvalue::Use(op),
                }
            }
            // Fieldless enum -> integer: the discriminant value.
            // Single-field struct -> integer of exactly the
            // field's type: the field itself (the struct is a
            // transparent wrapper around it).
            (Some(mir::Ty::Adt(def, args)), mir::Ty::Int(_) | mir::Ty::Uint(_)) => {
                if let Some(table) = self.fieldless_enum_discriminants(def) {
                    match self.convert_ty(target, "cast target") {
                        Ok(t) => ir::Rvalue::EnumToInt(v.clone(), table, t),
                        Err(_) => unsupported(),
                    }
                } else if let Some(shape) = self.wrapper_shape(def, args) {
                    // A transparent wrapper around a thin pointer
                    // (`NonNull`, `Unique`): transmuting it to an
                    // integer is the payload pointer's address —
                    // the transmute `fmt`'s statically-known-string
                    // check performs on its `NonNull<u8>`.
                    match self.field_operand(
                        body,
                        operand,
                        v,
                        shape.payload_index as u32,
                        &shape.payload_ty,
                    ) {
                        Some(op) => ir::Rvalue::PtrToInt(op),
                        None => unsupported(),
                    }
                } else if let Some(field_ty) = self.single_field_ty(def, args)
                {
                    if field_ty == *target {
                        match self.field_operand(body, operand, v, 0, &field_ty) {
                            Some(op) => ir::Rvalue::Use(op),
                            None => unsupported(),
                        }
                    } else if let mir::Ty::Adt(fdef, _) = &field_ty {
                        // A transparent wrapper around a fieldless
                        // enum: the struct's value is the field's
                        // value, and the field transmutes to the
                        // integer as its discriminant (the
                        // `mem::Alignment` shape — a one-field
                        // struct over a fieldless enum whose
                        // discriminants are the alignment values,
                        // transmuted to `usize` by RawVec's
                        // allocation path).
                        match self.fieldless_enum_discriminants(fdef) {
                            Some(table) => match self
                                .field_operand(body, operand, v, 0, &field_ty)
                            {
                                Some(op) => match self
                                    .convert_ty(target, "cast target")
                                {
                                    Ok(t) => {
                                        ir::Rvalue::EnumToInt(op, table, t)
                                    }
                                    Err(_) => unsupported(),
                                },
                                None => unsupported(),
                            },
                            None => unsupported(),
                        }
                    } else {
                        // A one-field wrapper over a scalar
                        // stream the arms above do not read —
                        // an array field, the `Simd<T, N>`
                        // storage shape core's integer
                        // comparison of small arrays produces
                        // after inlining: the bit-stream
                        // reinterpretation, whose element
                        // count the wrapper derivation takes
                        // from the integer's width.
                        match self.stream_regroup_rvalue(
                            &mir::Ty::Adt(*def, args.clone()),
                            target,
                            v,
                        ) {
                            Some(rv) => rv,
                            None => unsupported(),
                        }
                    }
                } else {
                    match self.stream_regroup_rvalue(
                        &mir::Ty::Adt(*def, args.clone()),
                        target,
                        v,
                    ) {
                        Some(rv) => rv,
                        None => unsupported(),
                    }
                }
            }
            // Integer -> fieldless enum: the variant whose
            // discriminant the integer's bit pattern equals.
            // Integer -> enum with a single variant carrying a
            // single integer field: that variant holding the
            // integer as its payload.
            (Some(s), mir::Ty::Adt(def, args)) if src_is_int => {
                if let Some(table) = self.fieldless_enum_discriminants(def) {
                    ir::Rvalue::IntToEnum(v.clone(), table)
                } else if let Some(field_ty) = self.single_field_ty(def, args) {
                    // One-field struct wrapping a fieldless
                    // enum, from the discriminant integer (the
                    // `mem::Alignment` construction).
                    match &field_ty {
                        mir::Ty::Adt(fdef, _) => {
                            match self.fieldless_enum_discriminants(fdef) {
                                Some(table) => {
                                    ir::Rvalue::IntToEnumWrapped {
                                        operand: v.clone(),
                                        discriminants: table,
                                        payload_index: 0,
                                        field_count: 1,
                                        fill: Vec::new(),
                                    }
                                }
                                None => unsupported(),
                            }
                        }
                        // A field the arms above do not read
                        // — an array field, the reverse of
                        // the `Simd<T, N>` -> integer shape:
                        // the bit-stream reinterpretation,
                        // whose element count the wrapper
                        // derivation takes from the integer's
                        // width.
                        _ => match self.stream_regroup_rvalue(
                            s,
                            &mir::Ty::Adt(*def, args.clone()),
                            v,
                        ) {
                            Some(rv) => rv,
                            None => unsupported(),
                        },
                    }
                } else if let Some(payload) =
                    self.single_variant_payload(def, args)
                {
                    if matches!(payload, mir::Ty::Int(_) | mir::Ty::Uint(_)) {
                        ir::Rvalue::AggregateAdt {
                            variant: 0,
                            operands: vec![v.clone()],
                            field_index: None,
                            units_from: None,
                            payload_entry: None,
                        }
                    } else {
                        match self.stream_regroup_rvalue(
                            s,
                            &mir::Ty::Adt(*def, args.clone()),
                            v,
                        ) {
                            Some(rv) => rv,
                            None => unsupported(),
                        }
                    }
                } else if let Some(shape) = self.wrapper_shape(def, args) {
                    // A transparent wrapper around a thin pointer
                    // (`NonNull`, `Unique`): the integer is the
                    // payload pointer's address, wrapped back up —
                    // the reverse of the wrapper-to-integer
                    // transmute above.
                    ir::Rvalue::IntToPtrWrapped {
                        operand: v.clone(),
                        payload_index: shape.payload_index as u32,
                        field_count: shape.field_count,
                        fill: shape
                            .fill
                            .iter()
                            .map(|(i, c)| (*i as u32, c.clone()))
                            .collect(),
                    }
                } else if let Some((table, shape)) =
                    self.enum_wrapper_shape(def, args)
                {
                    // A transparent wrapper around a fieldless
                    // enum (`mem::Alignment`): the integer is the
                    // discriminant of the enum value the wrapper
                    // holds — the reverse of the
                    // wrapper-to-integer transmute above.
                    ir::Rvalue::IntToEnumWrapped {
                        operand: v.clone(),
                        discriminants: table,
                        payload_index: shape.payload_index as u32,
                        field_count: shape.field_count,
                        fill: shape
                            .fill
                            .iter()
                            .map(|(i, c)| (*i as u32, c.clone()))
                            .collect(),
                    }
                } else {
                    match self.stream_regroup_rvalue(
                        s,
                        &mir::Ty::Adt(*def, args.clone()),
                        v,
                    ) {
                        Some(rv) => rv,
                        None => unsupported(),
                    }
                }
            }
            // Thin pointer -> owning-pointer chain (the
            // reverse of the ADT -> thin pointer chain arm
            // above): the target's pointer chain bottoms out
            // at a thin pointer, so the pointer becomes the
            // owner's value. When the chain enters the target
            // at field 0, that value is the allocation pointer
            // in a one-field aggregate — the shape `Box::new`
            // itself produces, whatever the chain's depth
            // (`Box` over `Unique` over `NonNull` over
            // `NonZero` over the pointer) — and the consumers
            // of an owning pointer (the drop glue, the
            // chain-walking pointer extraction) are built
            // around that shape, so it is kept exactly. When
            // the chain enters at a later field (a zero-sized
            // field sits before the payload), no model
            // constructor produces the value, so the first
            // layer is built in full instead: the pointer at
            // the chain field's own index and the zero-sized
            // fields' unique values at theirs, the pointer
            // standing in for the field-0 layers below it, as
            // it does throughout the model's wrapper chains.
            // This is the `transmute::<*mut T, Box<T>>` shape
            // and its nested-wrapper kin. Targets with a
            // transparent wrapper shape keep the arm below:
            // their zero-sized fills are part of their value.
            (Some(s), mir::Ty::Adt(def, args))
                if is_thin_ptr_ty(s)
                    && self.wrapper_shape(def, args).is_none()
                    && self.ptr_chain_entry_index(def, args).is_some() =>
            {
                let entry = self
                    .ptr_chain_entry_index(def, args)
                    .expect("guard checked");
                if entry == 0 {
                    ir::Rvalue::AggregateAdt {
                        variant: 0,
                        operands: vec![v.clone()],
                        field_index: None,
                        units_from: None,
                        payload_entry: None,
                    }
                } else {
                    let fields = match self.instantiated_field_tys(def, args, 0) {
                        Some(fields) => fields,
                        None => return unsupported(),
                    };
                    let mut operands = Vec::with_capacity(fields.len());
                    for (i, fty) in fields.iter().enumerate() {
                        if i == entry {
                            operands.push(v.clone());
                        } else {
                            let fill = self.zst_value(fty, 0).and_then(|zv| value_to_constant(&zv));
                            match fill {
                                Some(c) => operands.push(ir::Operand::Constant(c)),
                                None => return unsupported(),
                            }
                        }
                    }
                    ir::Rvalue::AggregateAdt {
                        variant: 0,
                        operands,
                        field_index: None,
                        units_from: None,
                        payload_entry: None,
                    }
                }
            }
            // Pointer -> transparent wrapper struct: wrap the
            // pointer, filling the zero-sized fields with their
            // unique values. The payload may be a fat pointer
            // (`*const str` -> `NonNull<str>`), matched by the
            // fat wrapper shape.
            (Some(s), mir::Ty::Adt(def, args))
                if is_thin_ptr_ty(s) || is_fat_ptr_ty(s) =>
            {
                let shape = if is_fat_ptr_ty(s) {
                    self.fat_wrapper_shape(def, args)
                } else {
                    self.wrapper_shape(def, args)
                };
                match shape {
                    Some(shape) => {
                        let mut operands = Vec::with_capacity(shape.field_count);
                        for i in 0..shape.field_count {
                            if i == shape.payload_index {
                                operands.push(v.clone());
                            } else {
                                let c = shape
                                    .fill
                                    .iter()
                                    .find(|(j, _)| *j == i)
                                    .map(|(_, c)| c.clone());
                                match c {
                                    Some(c) => {
                                        operands.push(ir::Operand::Constant(c))
                                    }
                                    None => return unsupported(),
                                }
                            }
                        }
                        // The pointer being wrapped may itself
                        // be retargeted by the wrap: a bare
                        // pointer to a prefix struct wrapped as a
                        // pointer to its first field (the
                        // `NonNull::new_unchecked(ptr).cast()`
                        // sequence std's B-tree node code
                        // inlines to) denotes the field, not the
                        // whole (see `prefix_retarget`).
                        let payload_entry = self
                            .prefix_retarget(s, &shape.payload_ty)
                            .map(|(enter, depth)| {
                                (enter, depth, shape.payload_index)
                            });
                        ir::Rvalue::AggregateAdt {
                            variant: 0,
                            operands,
                            field_index: None,
                            units_from: None,
                            payload_entry,
                        }
                    }
                    None => unsupported(),
                }
            }
            // Transparent wrapper struct -> pointer: read the
            // payload field back out. The payload may be a fat
            // pointer (`NonNull<str>` -> `*const str`, the
            // transmute `Location::file` performs), matched by
            // the fat wrapper shape.
            (Some(mir::Ty::Adt(def, args)), t)
                if is_thin_ptr_ty(t) || is_fat_ptr_ty(t) =>
            {
                let shape = if is_fat_ptr_ty(t) {
                    self.fat_wrapper_shape(def, args)
                } else {
                    self.wrapper_shape(def, args)
                };
                match shape {
                    Some(shape) => {
                        let rv = self.unwrap_wrapper(body, operand, v, &shape);
                        // The reverse of the wrap above: a
                        // payload read back out under a pointer
                        // type across a prefix boundary denotes
                        // the field (or the whole again), not the
                        // position it denoted inside the wrapper.
                        match (
                            rv,
                            self.prefix_retarget(&shape.payload_ty, t),
                        ) {
                            (ir::Rvalue::Use(op), Some((true, depth))) => {
                                ir::Rvalue::PtrField(op, depth)
                            }
                            (ir::Rvalue::Use(op), Some((false, depth))) => {
                                ir::Rvalue::PtrFieldBase(op, depth)
                            }
                            (rv, _) => rv,
                        }
                    }
                    None => unsupported(),
                }
            }
            // Pointer wrapper -> pointer wrapper of another
            // pointee (`NonNull::cast` and its kin): the payload
            // pointer carries over, rewrapped in the target
            // wrapper's shape.
            (Some(mir::Ty::Adt(def, args)), mir::Ty::Adt(tdef, targs))
                if (self.wrapper_shape(def, args).is_some()
                    || self.fat_wrapper_shape(def, args).is_some())
                    && (self.wrapper_shape(tdef, targs).is_some()
                        || self.fat_wrapper_shape(tdef, targs).is_some()) =>
            {
                let src_shape = self
                    .wrapper_shape(def, args)
                    .or_else(|| self.fat_wrapper_shape(def, args))
                    .expect("guard checked");
                let dst_shape = self
                    .wrapper_shape(tdef, targs)
                    .or_else(|| self.fat_wrapper_shape(tdef, targs))
                    .expect("guard checked");
                match self.field_operand(
                    body,
                    operand,
                    v,
                    src_shape.payload_index as u32,
                    &src_shape.payload_ty,
                ) {
                    Some(op) => {
                        let mut operands =
                            Vec::with_capacity(dst_shape.field_count);
                        for i in 0..dst_shape.field_count {
                            if i == dst_shape.payload_index {
                                operands.push(op.clone());
                            } else {
                                let c = dst_shape
                                    .fill
                                    .iter()
                                    .find(|(j, _)| *j == i)
                                    .map(|(_, c)| c.clone());
                                match c {
                                    Some(c) => {
                                        operands.push(ir::Operand::Constant(c))
                                    }
                                    None => return unsupported(),
                                }
                            }
                        }
                        // A retarget at a differently sized
                        // pointee denominates the payload
                        // pointer's index in the source
                        // pointee's elements; rewrite it in
                        // byte units as the wrapper is built
                        // (see `Rvalue::PtrCastUnits`).
                        let units_from = match (
                            ptr_pointee(&src_shape.payload_ty)
                                .and_then(|t| self.ty_layout(t))
                                .map(|(size, _)| size),
                            ptr_pointee(&dst_shape.payload_ty)
                                .and_then(|t| self.ty_layout(t))
                                .map(|(size, _)| size),
                        ) {
                            (Some(f), Some(t)) if f != t && f > 0 => {
                                Some((f, dst_shape.payload_index))
                            }
                            _ => None,
                        };
                        // A retarget between pointees in a prefix
                        // relation (the target pointee is the
                        // source pointee's first field, or the
                        // reverse -- a B-tree internal node and the
                        // leaf node it starts with) changes what
                        // the payload pointer denotes, exactly as
                        // the same cast on a bare pointer does:
                        // the payload field has the entry's field-0
                        // steps appended (or the exit's removed)
                        // as the wrapper is built (see
                        // `cast_entry_depth` / `cast_exit_depth`).
                        // The array-element continuation of an
                        // entry has no payload counterpart and
                        // keeps the verbatim carry.
                        let payload_entry = self
                            .prefix_retarget(
                                &src_shape.payload_ty,
                                &dst_shape.payload_ty,
                            )
                            .map(|(enter, depth)| {
                                (enter, depth, dst_shape.payload_index)
                            });
                        ir::Rvalue::AggregateAdt {
                            variant: 0,
                            operands,
                            field_index: None,
                            units_from,
                            payload_entry,
                        }
                    }
                    None => unsupported(),
                }
            }
            // Single-field struct wrapping an integer ->
            // fieldless enum: the field is the discriminant (the
            // `Bar { a }` -> `Foo` shape). Any other ADT -> ADT
            // pair falls through to the zero-sized rule below.
            (Some(mir::Ty::Adt(def, args)), mir::Ty::Adt(tdef, targs)) => {
                match (
                    self.single_field_ty(def, args),
                    self.fieldless_enum_discriminants(tdef),
                ) {
                    (Some(field_ty), Some(table))
                        if matches!(
                            field_ty,
                            mir::Ty::Int(_) | mir::Ty::Uint(_)
                        ) =>
                    {
                        match self.field_operand(body, operand, v, 0, &field_ty) {
                            Some(op) => ir::Rvalue::IntToEnum(op, table),
                            None => unsupported(),
                        }
                    }
                    // A one-field wrapper around a fieldless enum
                    // (`mem::Alignment`) transmuted to a
                    // transparent pointer wrapper (`NonNull`):
                    // the discriminant is the payload pointer's
                    // address — the transmute `NonNull::dangling`
                    // performs.
                    (Some(field_ty), None)
                        if self.wrapper_shape(tdef, targs).is_some() =>
                    {
                        let table = match &field_ty {
                            mir::Ty::Adt(fdef, _) => {
                                self.fieldless_enum_discriminants(fdef)
                            }
                            _ => None,
                        };
                        let shape = self
                            .wrapper_shape(tdef, targs)
                            .expect("guard checked");
                        match (
                            table,
                            self.field_operand(body, operand, v, 0, &field_ty),
                        ) {
                            (Some(table), Some(op)) => {
                                ir::Rvalue::EnumToPtrWrapped {
                                    operand: op,
                                    discriminants: table,
                                    payload_index: shape.payload_index as u32,
                                    field_count: shape.field_count,
                                    fill: shape
                                        .fill
                                        .iter()
                                        .map(|(i, c)| (*i as u32, c.clone()))
                                        .collect(),
                                }
                            }
                            _ => unsupported(),
                        }
                    }
                    // A fieldless enum itself transmuted to a
                    // transparent pointer wrapper: the same
                    // discriminant-as-address reading.
                    (None, None)
                        if self.fieldless_enum_discriminants(def).is_some()
                            && self.wrapper_shape(tdef, targs).is_some() =>
                    {
                        let table = self
                            .fieldless_enum_discriminants(def)
                            .expect("guard checked");
                        let shape = self
                            .wrapper_shape(tdef, targs)
                            .expect("guard checked");
                        ir::Rvalue::EnumToPtrWrapped {
                            operand: v.clone(),
                            discriminants: table,
                            payload_index: shape.payload_index as u32,
                            field_count: shape.field_count,
                            fill: shape
                                .fill
                                .iter()
                                .map(|(i, c)| (*i as u32, c.clone()))
                                .collect(),
                        }
                    }
                    // A fieldless enum transmuted to the
                    // one-field transparent struct wrapping it
                    // (the `mem::Alignment` shape: a struct whose
                    // single field is the enum itself): the value
                    // is the enum value, becoming the struct's
                    // single field.
                    (None, None)
                        if self.fieldless_enum_discriminants(def).is_some()
                            && self.single_field_ty(tdef, targs)
                                == Some(mir::Ty::Adt(def.clone(), args.to_vec())) =>
                    {
                        ir::Rvalue::AggregateAdt {
                            variant: 0,
                            operands: vec![v.clone()],
                            field_index: None,
                            units_from: None,
                            payload_entry: None,
                        }
                    }
                    // The reverse direction: the one-field
                    // struct over a fieldless enum transmuted to
                    // the enum it wraps: the field itself.
                    (Some(field_ty), Some(_)) if field_ty == *target => {
                        match self.field_operand(body, operand, v, 0, &field_ty) {
                            Some(op) => ir::Rvalue::Use(op),
                            None => unsupported(),
                        }
                    }
                    _ => match (self.zst_value(src.as_ref().unwrap(), 0), self.zst_value(target, 0)) {
                        (Some(_), Some(tv)) => match value_to_constant(&tv) {
                            Some(c) => {
                                ir::Rvalue::Use(ir::Operand::Constant(c))
                            }
                            None => unsupported(),
                        },
                        _ => unsupported(),
                    },
                }
            }
            // Array -> array of the same length across an
            // element wrapper boundary: `[T; N]` <->
            // `[MaybeUninit<T>; N]` (or the `UnsafeCell` twin).
            // The wrapper is transparent in this model -- its
            // payload is the element value at field 0 -- so the
            // array value, which stores its elements one by one,
            // carries over unchanged; only the element type the
            // destination declares changes. This is the transmute
            // an inlined array `IntoIter` construction performs
            // on the backing array.
            (Some(s), t) if self.array_elem_wrap_direction(s, t).is_some() => {
                match self.array_elem_wrap_direction(s, t) {
                    Some(true) => ir::Rvalue::WrapArrayElems(v.clone()),
                    _ => ir::Rvalue::UnwrapArrayElems(v.clone()),
                }
            }
            // Array -> array of the same length whose
            // elements are scalar-shaped of the same nonzero
            // width, at a total width beyond the whole-array
            // shape budget: the elementwise reinterpretation
            // (see `array_elem_reinterpret_shapes`).
            (Some(s), t)
                if self.array_elem_reinterpret_shapes(s, t).is_some() =>
            {
                let (from, to) = self
                    .array_elem_reinterpret_shapes(s, t)
                    .expect("guard checked");
                ir::Rvalue::Reinterpret(v.clone(), from, to)
            }
            // Array <-> array (or one-payload wrapper
            // over an array) whose leaf streams have the
            // same total width, at any total width: the
            // bit-stream reinterpretation — the
            // width-crossing regrouping and the long
            // wrapped array the whole-shape and
            // elementwise paths above cannot see (see
            // `stream_regroup_shapes`).
            (Some(s), t)
                if self.stream_regroup_shapes(s, t).is_some() =>
            {
                let (from, to) = self
                    .stream_regroup_shapes(s, t)
                    .expect("guard checked");
                ir::Rvalue::Reinterpret(v.clone(), from, to)
            }
            // Zero-sized -> zero-sized with a uniquely determined
            // value: the value is the target type's unique value,
            // whatever the source value was (a zero-sized source
            // with a unique value carries no other information).
            (Some(s), t) => {
                match (self.zst_value(s, 0), self.zst_value(t, 0)) {
                    (Some(_), Some(tv)) => match value_to_constant(&tv) {
                        Some(c) => ir::Rvalue::Use(ir::Operand::Constant(c)),
                        None => unsupported(),
                    },
                    _ => unsupported(),
                }
            }
            _ => unsupported(),
        };
        // A transmute whose result the optimiser scalarised to a
        // bare pointer local (a `NonNull::cast` whose wrapper was
        // SROA'd away) reaches here as a plain `Use` of the
        // unwrapped pointer; the denomination rewrite the
        // wrapper-to-wrapper arm records on its aggregate applies
        // to this shape as well.
        if let ir::Rvalue::Use(op) = &inner {
            if let Some(from) = self.transmute_units_from(src.as_ref(), target) {
                return ir::Rvalue::PtrCastUnits(op.clone(), from);
            }
            // A pointer-carrying transmute whose target pointee
            // has a recorded materialisation plan (and whose
            // denomination the units rewrite above does not
            // change): a type-erased constant allocation operand
            // materialises into the target type, exactly as for
            // the `PtrToPtr` cast conversion. Every other operand
            // keeps the identity the arms above decided.
            if let Some(idx) = self.mat_plan_index(target) {
                return ir::Rvalue::Materialize(op.clone(), idx);
            }
        }
        inner
    }

    /// The source element size for a pointer-valued transmute
    /// whose source and target pointees differ in size, when the
    /// source pointer is carried directly or inside a one-field
    /// pointer wrapper. Fat pointees keep their metadata path and
    /// never take part.
    fn transmute_units_from(
        &self,
        source: Option<&mir::Ty>,
        target: &mir::Ty,
    ) -> Option<u64> {
        let target_pointee = ptr_pointee(target)?;
        if matches!(
            target_pointee,
            mir::Ty::Slice(_) | mir::Ty::Str | mir::Ty::Dynamic(..)
        ) {
            return None;
        }
        let source_pointee = match source? {
            mir::Ty::RawPtr(p, _) | mir::Ty::Ref(_, p, _) => &**p,
            mir::Ty::Adt(def, args) => {
                let shape = self
                    .wrapper_shape(def, args)
                    .or_else(|| self.fat_wrapper_shape(def, args))?;
                match shape.payload_ty.clone() {
                    mir::Ty::RawPtr(p, _) | mir::Ty::Ref(_, p, _) => {
                        return self.transmute_units_from_pointees(&p, target);
                    }
                    _ => return None,
                }
            }
            _ => return None,
        };
        if matches!(
            source_pointee,
            mir::Ty::Slice(_) | mir::Ty::Str | mir::Ty::Dynamic(..)
        ) {
            return None;
        }
        self.transmute_units_from_pointees(source_pointee, target)
    }

    fn transmute_units_from_pointees(
        &self,
        source_pointee: &mir::Ty,
        target: &mir::Ty,
    ) -> Option<u64> {
        let target_pointee = ptr_pointee(target)?;
        if matches!(
            source_pointee,
            mir::Ty::Slice(_) | mir::Ty::Str | mir::Ty::Dynamic(..)
        ) {
            return None;
        }
        let (from, _) = self.ty_layout(source_pointee)?;
        let (to, _) = self.ty_layout(target_pointee)?;
        (from > 0 && from != to).then_some(from)
    }


    /// Whether `def` is `MaybeUninit` (std or core).
    pub(super) fn is_maybe_uninit(&self, def: &mir::AdtDefId) -> bool {
        matches!(
            self.def_display_name(def).as_str(),
            "std::mem::MaybeUninit" | "core::mem::MaybeUninit"
        )
    }

    /// The operand denoting the payload of a one-payload
    /// wrapper value, when the wrapper's payload type is
    /// exactly `target`: a one-field struct whose only field
    /// has the target type, or a `MaybeUninit` of the target
    /// type. In this value model a `MaybeUninit<T>` holds its
    /// payload at field 0 of the union value -- the slot every
    /// pointer access to the union denotes -- so field 0 is
    /// the payload's slot for both shapes. `None` when the
    /// source is not such a wrapper, or when the operand
    /// carries no addressable payload field.
    fn payload_field_operand(
        &self,
        body: &mir::Body,
        operand: &mir::Operand,
        v: &ir::Operand,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
        target: &mir::Ty,
    ) -> Option<ir::Operand> {
        let payload_ty = if let Some(field_ty) = self.single_field_ty(def, args) {
            if field_ty != *target {
                return None;
            }
            field_ty
        } else if self.is_maybe_uninit(def) && args.len() == 1 && args[0] == *target
        {
            args[0].clone()
        } else {
            return None;
        };
        self.field_operand(body, operand, v, 0, &payload_ty)
    }

    /// The aggregate rvalue wrapping `v` into a one-payload
    /// wrapper of exactly its type: a one-field struct whose
    /// only field has the source type (the payload is the
    /// struct's only field), or a `MaybeUninit` of the source
    /// type (the payload goes to field 0 of the union value,
    /// the slot this model reads it back from). `None` when
    /// the target is not such a wrapper.
    fn wrap_payload_rvalue(
        &self,
        v: &ir::Operand,
        src: &mir::Ty,
        tdef: &mir::AdtDefId,
        targs: &[mir::Ty],
    ) -> Option<ir::Rvalue> {
        let field_index = if let Some(field_ty) = self.single_field_ty(tdef, targs) {
            if field_ty != *src {
                return None;
            }
            None
        } else if self.is_maybe_uninit(tdef)
            && targs.len() == 1
            && targs[0] == *src
        {
            Some(0)
        } else {
            return None;
        };
        Some(ir::Rvalue::AggregateAdt {
            variant: 0,
            operands: vec![v.clone()],
            field_index,
            units_from: None,
            payload_entry: None,
        })
    }

    /// Read the payload field out of a transparent wrapper value
    /// (the reverse of the wrap in [`Self::convert_transmute`]).
    fn unwrap_wrapper(
        &self,
        body: &mir::Body,
        operand: &mir::Operand,
        v: &ir::Operand,
        shape: &WrapperShape,
    ) -> ir::Rvalue {
        match self.field_operand(
            body,
            operand,
            v,
            shape.payload_index as u32,
            &shape.payload_ty,
        ) {
            Some(op) => ir::Rvalue::Use(op),
            None => ir::Rvalue::Unsupported("Cast Transmute".to_string()),
        }
    }

    /// The operand denoting field `index` of the value `operand`
    /// denotes: for a place operand, the place extended with the
    /// field projection (preserving move-ness); for a constant
    /// ADT operand, the field constant. `None` for operand forms
    /// that carry no addressable field.
    fn field_operand(
        &self,
        body: &mir::Body,
        operand: &mir::Operand,
        v: &ir::Operand,
        index: u32,
        field_ty: &mir::Ty,
    ) -> Option<ir::Operand> {
        match operand.place() {
            Some(place) => {
                let place = place.clone().field(index, field_ty.clone());
                let p = self.convert_place(body, &place).ok()?;
                Some(match operand {
                    mir::Operand::Move(_) => ir::Operand::Move(p),
                    _ => ir::Operand::Copy(p),
                })
            }
            None => match v {
                ir::Operand::Constant(ir::Constant::Adt { fields, .. }) => fields
                    .get(index as usize)
                    .map(|c| ir::Operand::Constant(c.clone())),
                _ => None,
            },
        }
    }

    /// The direction in which an array -> array transmute of
    /// the same length crosses an element wrapper boundary:
    /// `Some(true)` when the target's element is the source's
    /// element wrapped in `MaybeUninit` or `UnsafeCell` (the two
    /// wrappers this model treats as transparent
    /// payload-at-field-0 carriers, as in
    /// [`Self::pointee_crosses_cell_wrapper`]), `Some(false)`
    /// for the reverse crossing, `None` when the pair does not
    /// cross the boundary.
    /// The element-wrapper direction of a pointer cast
    /// between two slice pointees: `Some(true)` when the
    /// target's elements wrap the source's (`[T]` ->
    /// `[MaybeUninit<T>]`), `Some(false)` for the reverse,
    /// `None` when the element types are not related by
    /// exactly one `MaybeUninit` / `UnsafeCell` layer.
    pub(super) fn slice_elem_wrap_direction(
        &self,
        src: &mir::Ty,
        target: &mir::Ty,
    ) -> Option<bool> {
        let (Some(mir::Ty::Slice(src_elem)), Some(mir::Ty::Slice(target_elem))) =
            (ptr_pointee(src), ptr_pointee(target))
        else {
            return None;
        };
        let wraps = |wrapper: &mir::Ty, inner: &mir::Ty| -> bool {
            match wrapper {
                mir::Ty::Adt(def, args) => {
                    args.len() == 1
                        && args[0] == *inner
                        && matches!(
                            self.def_display_name(def).as_str(),
                            "core::mem::MaybeUninit"
                                | "std::mem::MaybeUninit"
                                | "core::cell::UnsafeCell"
                                | "std::cell::UnsafeCell"
                        )
                }
                _ => false,
            }
        };
        if wraps(target_elem, src_elem) {
            Some(true)
        } else if wraps(src_elem, target_elem) {
            Some(false)
        } else {
            None
        }
    }

    fn array_elem_wrap_direction(
        &self,
        src: &mir::Ty,
        target: &mir::Ty,
    ) -> Option<bool> {
        let (
            mir::Ty::Array(src_elem, src_len),
            mir::Ty::Array(target_elem, target_len),
        ) = (src, target)
        else {
            return None;
        };
        if !matches!(
            (src_len, target_len),
            (
                mir::ArrayLen::Known(a),
                mir::ArrayLen::Known(b)
            ) if a == b
        ) {
            return None;
        }
        let wraps = |wrapper: &mir::Ty, inner: &mir::Ty| -> bool {
            match wrapper {
                mir::Ty::Adt(def, args) => {
                    args.len() == 1
                        && args[0] == *inner
                        && matches!(
                            self.def_display_name(def).as_str(),
                            "core::mem::MaybeUninit"
                                | "std::mem::MaybeUninit"
                                | "core::cell::UnsafeCell"
                                | "std::cell::UnsafeCell"
                        )
                }
                _ => false,
            }
        };
        if wraps(target_elem, src_elem) {
            Some(true)
        } else if wraps(src_elem, target_elem) {
            Some(false)
        } else {
            None
        }
    }

    /// The elementwise reinterpretation shapes for an
    /// array -> array transmute of the same length whose
    /// elements are scalar-shaped of the same nonzero
    /// width: the pair of array shapes over the elements'
    /// own shapes, or `None` when the pair is not that
    /// shape. The whole-array scalar shapes cap the total
    /// width at 128 bits, so the generic scalar path
    /// cannot see a long array pair; the elements' shapes
    /// carry the same reinterpretation the generic path
    /// computes for the lengths that fit, and the
    /// evaluation walks the two array shapes element by
    /// element. `[ *mut T; 63 ]` -> `[Atomic<*mut T>; 63]`
    /// (the `thread_local` crate's bucket table) is the
    /// canonical case: each element is the pointer's bits
    /// behind the transparent atomic wrapper.
    fn array_elem_reinterpret_shapes(
        &self,
        src: &mir::Ty,
        target: &mir::Ty,
    ) -> Option<(ir::ScalarShape, ir::ScalarShape)> {
        let (
            mir::Ty::Array(src_elem, mir::ArrayLen::Known(src_len)),
            mir::Ty::Array(target_elem, mir::ArrayLen::Known(target_len)),
        ) = (src, target)
        else {
            return None;
        };
        if src_len != target_len || *src_len == 0 {
            return None;
        }
        let from = self.scalar_shape(src_elem, 0)?;
        let to = self.scalar_shape(target_elem, 0)?;
        if from.bits() == 0 || from.bits() != to.bits() {
            return None;
        }
        Some((
            ir::ScalarShape::Array(Box::new(from), *src_len),
            ir::ScalarShape::Array(Box::new(to), *src_len),
        ))
    }

    /// The bit-stream shape of a type: the scalar shape
    /// without the 128-bit total-width budget on arrays.
    /// Leaves are the scalar kinds; an array contributes
    /// its element's stream at any length; a one-variant
    /// ADT contributes its single nonzero-stream payload
    /// field (every other field a zero-sized constant, the
    /// discipline `scalar_shape`'s payload arm uses), which
    /// covers plain wrappers, `MaybeUninit<T>` and unions
    /// alike. Everything else — tuples, multi-field and
    /// niche/tagged shapes, pointers — has no stream shape:
    /// a pointer leaf is a place identity, not a bit
    /// pattern, and a multi-field layout's gaps are not
    /// stream positions.
    fn stream_shape(
        &self,
        ty: &mir::Ty,
        depth: u32,
    ) -> Option<ir::ScalarShape> {
        use ir::ScalarShape as S;
        if depth > 8 {
            return None;
        }
        match ty {
            mir::Ty::Bool => Some(S::Bool),
            mir::Ty::Char => Some(S::Char),
            mir::Ty::Int(it) => Some(S::Int {
                bits: it.bits(),
                signed: true,
            }),
            mir::Ty::Uint(ut) => Some(S::Int {
                bits: ut.bits(),
                signed: false,
            }),
            mir::Ty::Float(ft) => match ft.bits() {
                16 | 32 | 64 | 128 => Some(S::Float { bits: ft.bits() }),
                _ => None,
            },
            mir::Ty::Array(elem, mir::ArrayLen::Known(n)) => {
                if *n == 0 {
                    return None;
                }
                let elem = self.stream_shape(elem, depth + 1)?;
                if elem.bits() == 0 {
                    return None;
                }
                Some(S::Array(Box::new(elem), *n))
            }
            mir::Ty::Adt(def, args) => {
                if matches!(
                    self.def_display_name(def).as_str(),
                    "std::mem::MaybeUninit" | "core::mem::MaybeUninit"
                ) {
                    if args.len() != 1 {
                        return None;
                    }
                    let inner = self.stream_shape(&args[0], depth + 1)?;
                    return Some(S::AdtField {
                        index: 0,
                        count: 1,
                        fill: Vec::new(),
                        inner: Box::new(inner),
                    });
                }
                let shape = self.adt_shapes.get(def)?;
                if shape.variants.len() != 1 {
                    return None;
                }
                let fields = self.instantiated_field_tys(def, args, 0)?;
                let mut payload: Option<(u32, ir::ScalarShape)> = None;
                let mut fill = Vec::new();
                for (i, fty) in fields.iter().enumerate() {
                    match self.stream_shape(fty, depth + 1) {
                        Some(fs) if fs.bits() > 0 => {
                            if payload.is_some() {
                                return None;
                            }
                            payload = Some((i as u32, fs));
                        }
                        _ => {
                            let zv = self.zst_value(fty, 0)?;
                            fill.push((i as u32, value_to_constant(&zv)?));
                        }
                    }
                }
                let (index, inner) = payload?;
                Some(S::AdtField {
                    index,
                    count: fields.len(),
                    fill,
                    inner: Box::new(inner),
                })
            }
            _ => None,
        }
    }

    /// The stream shape of a one-field ADT whose only field
    /// is an array, with the element count taken from the
    /// transmute's other side (`total_bits` is its total
    /// width): the pair is size-equal by the transmute's
    /// own validity, so the other side's total width over
    /// this side's element width is the count. Deriving
    /// the count this way — rather than reading the
    /// field's recorded length — is what makes the
    /// `Simd<T, N>` storage field work: the length is a
    /// const parameter, const arguments are not part of
    /// the type here, and the recorded instance field
    /// types key on the type arguments only, so a sibling
    /// instance's length can be the recorded one.
    fn wrapped_array_stream_shape(
        &self,
        ty: &mir::Ty,
        total_bits: u32,
    ) -> Option<ir::ScalarShape> {
        let mir::Ty::Adt(def, args) = ty else {
            return None;
        };
        let shape = self.adt_shapes.get(def)?;
        if shape.variants.len() != 1 {
            return None;
        }
        let fields = self.instantiated_field_tys(def, args, 0)?;
        if fields.len() != 1 {
            return None;
        }
        let mir::Ty::Array(elem, _) = &fields[0] else {
            return None;
        };
        let elem = self.stream_shape(elem, 1)?;
        let eb = elem.bits();
        if eb == 0 || total_bits == 0 || total_bits % eb != 0 {
            return None;
        }
        Some(ir::ScalarShape::AdtField {
            index: 0,
            count: 1,
            fill: Vec::new(),
            inner: Box::new(ir::ScalarShape::Array(
                Box::new(elem),
                u64::from(total_bits / eb),
            )),
        })
    }

    /// The bit-stream reinterpretation of a pair as an
    /// rvalue, or `None` when the pair does not decompose
    /// into equal-width bit streams (see
    /// `stream_regroup_shapes`).
    fn stream_regroup_rvalue(
        &self,
        src: &mir::Ty,
        target: &mir::Ty,
        v: &ir::Operand,
    ) -> Option<ir::Rvalue> {
        let (from, to) = self.stream_regroup_shapes(src, target)?;
        Some(ir::Rvalue::Reinterpret(v.clone(), from, to))
    }

    /// The bit-stream reinterpretation shapes for a
    /// transmute pair: both sides decompose into scalar
    /// leaf streams of the same nonzero total width, and
    /// at least one side decomposes through an array (a
    /// bare leaf pair is the generic path's business, and
    /// a one-payload pair over a scalar is the payload
    /// arms'). Either side may instead be a one-field ADT
    /// over an array whose length the type representation
    /// does not carry; its count comes from the other side
    /// (see `wrapped_array_stream_shape`).
    fn stream_regroup_shapes(
        &self,
        src: &mir::Ty,
        target: &mir::Ty,
    ) -> Option<(ir::ScalarShape, ir::ScalarShape)> {
        let from = self.stream_shape(src, 0);
        let to = self.stream_shape(target, 0);
        // A one-field ADT over an array field takes its
        // element count from the other side whenever the
        // other side's stream shape is known (see
        // `wrapped_array_stream_shape`); when neither side
        // is such a wrapper — or both are, so neither total
        // is independently known — the stream shapes stand
        // as computed.
        let (from, to) = match (from, to) {
            (Some(f), Some(t)) => {
                let f = if self.is_array_wrapper(src)
                    && !self.is_array_wrapper(target)
                {
                    self.wrapped_array_stream_shape(src, t.bits())
                        .unwrap_or(f)
                } else {
                    f
                };
                let t = if self.is_array_wrapper(target)
                    && !self.is_array_wrapper(src)
                {
                    self.wrapped_array_stream_shape(target, f.bits())
                        .unwrap_or(t)
                } else {
                    t
                };
                (f, t)
            }
            (None, Some(t)) => (
                self.wrapped_array_stream_shape(src, t.bits())?,
                t,
            ),
            (Some(f), None) => {
                let total = f.bits();
                (
                    f,
                    self.wrapped_array_stream_shape(target, total)?,
                )
            }
            (None, None) => return None,
        };
        if from.bits() == 0 || from.bits() != to.bits() {
            return None;
        }
        if !shape_is_bit_stream(&from) || !shape_is_bit_stream(&to) {
            return None;
        }
        fn decomposes(shape: &ir::ScalarShape) -> bool {
            match shape {
                ir::ScalarShape::Array(..) => true,
                ir::ScalarShape::AdtField { inner, .. } => {
                    decomposes(inner)
                }
                _ => false,
            }
        }
        if !decomposes(&from) && !decomposes(&to) {
            return None;
        }
        Some((from, to))
    }

    /// The fat-pointer metadata kind a pointer type's value
    /// carries in this model, seeing through a DST-wrapper
    /// struct: a pointer to a slice / `str` / `dyn`
    /// pointee, or to an ADT whose last field is such a
    /// type (the unsized tail the metadata counts or
    /// names). `None` for a thin pointer type.
    fn fat_tail_meta(&self, ty: &mir::Ty) -> Option<ir::FatMeta> {
        self.tail_meta(ptr_pointee(ty)?, 0)
    }

    /// The metadata kind for an unsized tail type: a slice
    /// / `str` / `dyn` type itself, or a DST-wrapper ADT
    /// whose last field is (recursively) such a tail — the
    /// `Slice([u32])` wrapper inside a `PackedUnsized` is
    /// the same tail one level down. A tuple is a tail of
    /// the same kind when its last field is: only a tuple's
    /// last field may be unsized, and the fat metadata of a
    /// tuple-tailed pointee is that field's metadata (the
    /// `(u16, [u8])` tail of a `WT<(u16, [u8])>`, or a bare
    /// `(u8, [u32])` pointee).
    fn tail_meta(
        &self,
        tail: &mir::Ty,
        depth: u32,
    ) -> Option<ir::FatMeta> {
        if depth > 4 {
            return None;
        }
        match tail {
            mir::Ty::Slice(_) | mir::Ty::Str => Some(ir::FatMeta::Len),
            mir::Ty::Dynamic(..) => Some(ir::FatMeta::Vtable),
            mir::Ty::Adt(def, args) => {
                let shape = self.adt_shapes.get(def)?;
                if shape.variants.len() != 1 {
                    return None;
                }
                let fields = self.instantiated_field_tys(def, args, 0)?;
                self.tail_meta(fields.last()?, depth + 1)
            }
            mir::Ty::Tuple(fields) => {
                self.tail_meta(fields.last()?, depth + 1)
            }
            _ => None,
        }
    }

    /// Whether the type is a one-variant ADT with exactly
    /// one field, an array: the wrapper form whose stream
    /// shape's element count the transmute's other side
    /// fixes (see `wrapped_array_stream_shape`).
    fn is_array_wrapper(&self, ty: &mir::Ty) -> bool {
        let mir::Ty::Adt(def, args) = ty else {
            return false;
        };
        let Some(shape) = self.adt_shapes.get(def) else {
            return false;
        };
        if shape.variants.len() != 1 {
            return false;
        }
        let Some(fields) = self.instantiated_field_tys(def, args, 0)
        else {
            return false;
        };
        fields.len() == 1 && matches!(fields[0], mir::Ty::Array(..))
    }

    /// The (leaf, storage) shape pair for a transmute
    /// between a scalar-or-pointer leaf type and a union:
    /// the leaf is the other side's scalar shape when it
    /// is a leaf kind, and the storage shape is the
    /// union's read through the field whose shape is the
    /// leaf's (see `union_storage_shape`). `None` when the
    /// other side is not a leaf or no union field has its
    /// shape.
    fn union_leaf_pair(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
        other: &mir::Ty,
    ) -> Option<(ir::ScalarShape, ir::ScalarShape)> {
        let leaf = self.scalar_shape(other, 0)?;
        if !matches!(
            leaf,
            ir::ScalarShape::Ptr
                | ir::ScalarShape::FnPtr
                | ir::ScalarShape::Int { .. }
                | ir::ScalarShape::Float { .. }
                | ir::ScalarShape::Bool
                | ir::ScalarShape::Char
        ) {
            return None;
        }
        let storage = self.union_storage_shape(def, args, &leaf)?;
        Some((leaf, storage))
    }

    /// The storage shape of a union read through the
    /// field whose scalar shape is `want`: that field as
    /// the one-payload shape's payload, every zero-sized
    /// field as a fill constant, and every other field
    /// left uninit (a union's fields alias one storage;
    /// the value records the field it was written
    /// through). `None` when no field has shape `want`,
    /// or a zero-sized field has no constant.
    fn union_storage_shape(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
        want: &ir::ScalarShape,
    ) -> Option<ir::ScalarShape> {
        let fields = self.instantiated_field_tys(def, args, 0)?;
        let mut payload = None;
        let mut fill = Vec::new();
        for (i, fty) in fields.iter().enumerate() {
            match self.scalar_shape(fty, 0) {
                Some(fs) if &fs == want && payload.is_none() => {
                    payload = Some(i as u32);
                }
                Some(fs) if fs.bits() == 0 => {
                    let zv = self.zst_value(fty, 0)?;
                    fill.push((i as u32, value_to_constant(&zv)?));
                }
                _ => {}
            }
        }
        let index = payload?;
        Some(ir::ScalarShape::AdtField {
            index,
            count: fields.len(),
            fill,
            inner: Box::new(want.clone()),
        })
    }

    /// Whether the ADT is a union.
    fn adt_is_union(&self, def: &mir::AdtDefId) -> bool {
        self.adt_shapes
            .get(def)
            .is_some_and(|sh| sh.kind == mir::AdtShapeKind::Union)
    }

    /// Whether a thin-pointer -> thin-pointer transmute crosses
    /// an `UnsafeCell` / `MaybeUninit` wrapper boundary: one
    /// side's pointee is the wrapper ADT and the other side's
    /// pointee is exactly the type the wrapper wraps. The
    /// entering direction (wrapper -> wrapped type) is composed
    /// with the cast conversion's field-entering treatment
    /// (`cast_wrapper_entry_depth` -> `Rvalue::PtrField`) in
    /// `convert_transmute` before this guard is consulted, so
    /// the crossings this guard still excludes are the exiting
    /// direction (wrapped type -> wrapper), which has no
    /// field-exiting rvalue and stays explicit unsupported
    /// instead of silently pointing at the field.
    /// Whether a transmute between two pointer-wrapper ADTs
    /// retargets the payload pointer across a prefix struct's
    /// first field (see
    /// [`Self::cast_crosses_prefix_field`]): the retarget under
    /// which carrying the wrapper value over unchanged would
    /// leave the payload pointer denoting the wrong sub-value.
    /// Either side lacking a wrapper shape names no payload
    /// pointer to retarget, and the pair keeps its established
    /// reading.
    fn payload_prefix_crossing(
        &self,
        src: &Option<mir::Ty>,
        target: &mir::Ty,
    ) -> bool {
        let (Some(mir::Ty::Adt(def, args)), mir::Ty::Adt(tdef, targs)) =
            (src, target)
        else {
            return false;
        };
        let (Some(src_shape), Some(dst_shape)) = (
            self.wrapper_shape(def, args)
                .or_else(|| self.fat_wrapper_shape(def, args)),
            self.wrapper_shape(tdef, targs)
                .or_else(|| self.fat_wrapper_shape(tdef, targs)),
        ) else {
            return false;
        };
        self.cast_crosses_prefix_field(
            &src_shape.payload_ty,
            &dst_shape.payload_ty,
        )
    }

    fn pointee_crosses_cell_wrapper(
        &self,
        src: &mir::Ty,
        target: &mir::Ty,
    ) -> bool {
        fn pointee(ty: &mir::Ty) -> Option<&mir::Ty> {
            match ty {
                mir::Ty::RawPtr(inner, _) | mir::Ty::Ref(_, inner, _) => {
                    Some(inner)
                }
                _ => None,
            }
        }
        let (Some(sp), Some(tp)) = (pointee(src), pointee(target)) else {
            return false;
        };
        let wrapped = |wrapper: &mir::Ty, plain: &mir::Ty| -> bool {
            match wrapper {
                mir::Ty::Adt(def, args) => {
                    args.len() == 1
                        && args[0] == *plain
                        && matches!(
                            self.def_display_name(def).as_str(),
                            "core::cell::UnsafeCell"
                                | "std::cell::UnsafeCell"
                                | "core::mem::MaybeUninit"
                                | "std::mem::MaybeUninit"
                        )
                }
                _ => false,
            }
        };
        wrapped(sp, tp) || wrapped(tp, sp)
    }

    /// The instantiated type of the only field of a struct with
    /// exactly one field, or `None` for any other ADT.
    pub(super) fn single_field_ty(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
    ) -> Option<mir::Ty> {
        let shape = self.adt_shapes.get(def)?;
        if shape.kind != mir::AdtShapeKind::Struct || shape.variants.len() != 1
        {
            return None;
        }
        let fields = self.instantiated_field_tys(def, args, 0)?;
        if fields.len() == 1 {
            fields.into_iter().next()
        } else {
            None
        }
    }

    /// The instantiated type of the first field of a struct with
    /// at least two fields, when that field's type is an aggregate
    /// (an ADT, a tuple or an array): the prefix-struct shape,
    /// whose first field is stored as a real sub-aggregate at the
    /// head of the value (std's B-tree `InternalNode`, whose
    /// `data` field is the `LeafNode` the node is also addressed
    /// as). A scalar first field is excluded for the reason the
    /// single-field restriction in `wrapper_peel` states: values
    /// this model stores bare have no field storage for a field
    /// projection to address, while an aggregate first field of a
    /// multi-field struct is always stored as its own sub-value.
    /// `None` for any other ADT.
    pub(super) fn prefix_field_ty(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
    ) -> Option<mir::Ty> {
        let shape = self.adt_shapes.get(def)?;
        if shape.kind != mir::AdtShapeKind::Struct || shape.variants.len() != 1
        {
            return None;
        }
        let fields = self.instantiated_field_tys(def, args, 0)?;
        if fields.len() < 2 {
            return None;
        }
        let first = fields.into_iter().next()?;
        if matches!(
            first,
            mir::Ty::Adt(..) | mir::Ty::Tuple(_) | mir::Ty::Array(..)
        ) {
            Some(first)
        } else {
            None
        }
    }

    /// The instantiated type of the only field of an enum with
    /// exactly one variant carrying exactly one field, or `None`
    /// for any other ADT.
    fn single_variant_payload(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
    ) -> Option<mir::Ty> {
        let shape = self.adt_shapes.get(def)?;
        if shape.kind != mir::AdtShapeKind::Enum || shape.variants.len() != 1
        {
            return None;
        }
        let fields = self.instantiated_field_tys(def, args, 0)?;
        if fields.len() == 1 {
            fields.into_iter().next()
        } else {
            None
        }
    }

    /// The effective discriminant table of an enum whose
    /// variants are all fieldless (variant index -> discriminant
    /// value), or `None` when `def` is not such an enum. A missing
    /// recorded table means the discriminants are the sequential
    /// defaults, so the table is synthesized from the variant
    /// count — the same convention as the `Discriminant` rvalue.
    fn fieldless_enum_discriminants(
        &self,
        def: &mir::AdtDefId,
    ) -> Option<Vec<i128>> {
        let shape = self.adt_shapes.get(def)?;
        if shape.kind != mir::AdtShapeKind::Enum
            || shape.variants.iter().any(|fields| !fields.is_empty())
        {
            return None;
        }
        match self.discriminants.get(def) {
            Some(table) if !table.is_empty() => Some(table.clone()),
            _ => Some((0..shape.variants.len() as i128).collect()),
        }
    }

    /// The value the all-zeroes bit pattern of `ty` denotes, in
    /// the model's structural form -- what `std::mem::zeroed`
    /// materializes. Scalars are their zero; a tuple or array
    /// is built element by element; an ADT is built field by
    /// field, an enum taking the variant the recorded layout
    /// assigns to the zero pattern: the niche variant when
    /// zero lies in the recorded niche range, otherwise the
    /// variant whose discriminant is zero. `None` when the
    /// pattern denotes no value this construction can build:
    /// an enum with no zero-discriminant variant and no niche
    /// covering zero, a `dyn` fat pointer (a null vtable names
    /// no identity), a function pointer, an unrecorded ADT, a
    /// union other than `MaybeUninit`.
    pub(super) fn zeroed_model_value(&self, ty: &mir::Ty, depth: u32) -> Option<Value> {
        use crate::value::RefTarget;
        if depth > 8 {
            return None;
        }
        match ty {
            mir::Ty::Bool => Some(Value::Bool(false)),
            mir::Ty::Char => Some(Value::Uint(0)),
            mir::Ty::Int(_) => Some(Value::Int(0)),
            mir::Ty::Uint(_) => Some(Value::Uint(0)),
            mir::Ty::Float(ft) => Some(match ft.bits() {
                128 => Value::Float128(0),
                _ => Value::Float(0.0),
            }),
            mir::Ty::Tuple(ts) => {
                if ts.is_empty() {
                    return Some(Value::Unit);
                }
                let mut fields = Vec::with_capacity(ts.len());
                for t in ts {
                    fields.push(self.zeroed_model_value(t, depth + 1)?);
                }
                Some(Value::Tuple(fields))
            }
            mir::Ty::Array(elem, mir::ArrayLen::Known(n)) => {
                if *n > 4096 {
                    return None;
                }
                let elem = self.zeroed_model_value(elem, depth + 1)?;
                Some(Value::Tuple(vec![elem; *n as usize]))
            }
            // A zeroed thin pointer is the null pointer; a
            // zeroed slice or string pointer is the null fat
            // pointer of length zero.
            mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => {
                match &**inner {
                    mir::Ty::Slice(_) | mir::Ty::Str => Some(Value::FatRef {
                        base: Box::new(Value::Ref(RefTarget::Null)),
                        len: 0,
                    }),
                    mir::Ty::Dynamic(..) => None,
                    _ => Some(Value::Ref(RefTarget::Null)),
                }
            }
            mir::Ty::Adt(def, args) => {
                // `MaybeUninit<T>` holds its payload at field 0
                // in this model, and its `zeroed` constructor is
                // exactly the zeroed payload in that slot.
                if matches!(
                    self.def_display_name(def).as_str(),
                    "std::mem::MaybeUninit" | "core::mem::MaybeUninit"
                ) {
                    if args.len() != 1 {
                        return None;
                    }
                    let payload = self.zeroed_model_value(&args[0], depth + 1)?;
                    return Some(Value::Enum {
                        variant: 0,
                        fields: vec![payload],
                    });
                }
                let shape = self.adt_shapes.get(def)?;
                let zeroed_fields = |variant: u32| -> Option<Vec<Value>> {
                    let ftys = self.instantiated_field_tys(def, args, variant)?;
                    let mut fields = Vec::with_capacity(ftys.len());
                    for fty in &ftys {
                        fields.push(self.zeroed_model_value(fty, depth + 1)?);
                    }
                    Some(fields)
                };
                match shape.kind {
                    mir::AdtShapeKind::Union => None,
                    mir::AdtShapeKind::Struct => Some(Value::Enum {
                        variant: 0,
                        fields: zeroed_fields(0)?,
                    }),
                    mir::AdtShapeKind::Enum => {
                        // A niche-encoded instance: the recorded
                        // niche decides which variant the zero
                        // pattern denotes (the window the pattern
                        // reads in the niche scalar is zero).
                        let tag = crate::builtins::ty_tag(
                            &mir::Ty::Adt(*def, args.to_vec()),
                            self.names,
                        );
                        if let Some(niche) = self.adt_niches.get(&tag) {
                            let mask = if niche.tag_bits >= 128 {
                                u128::MAX
                            } else {
                                (1u128 << niche.tag_bits) - 1
                            };
                            let offset =
                                0u128.wrapping_sub(niche.niche_start) & mask;
                            let span = u128::from(
                                niche.niche_variants.1 - niche.niche_variants.0,
                            );
                            if offset <= span {
                                return Some(Value::Enum {
                                    variant: niche.niche_variants.0
                                        + offset as u32,
                                    fields: Vec::new(),
                                });
                            }
                            return Some(Value::Enum {
                                variant: niche.untagged_variant,
                                fields: zeroed_fields(niche.untagged_variant)?,
                            });
                        }
                        // A direct-tag instance: the variant
                        // whose discriminant is zero.
                        let table = self.discriminants.get(def)?;
                        let variant =
                            table.iter().position(|d| *d == 0)? as u32;
                        Some(Value::Enum {
                            variant,
                            fields: zeroed_fields(variant)?,
                        })
                    }
                }
            }
            _ => None,
        }
    }

    /// The [`ir::ScalarShape`] of a type whose every bit is
    /// accounted for by scalar leaves (integers, floats, `bool`,
    /// `char`, arrays and tuples of those, fat pointers, and
    /// one-payload-field ADTs over those), or `None` for any
    /// other type. `depth` bounds the
    /// ADT nesting: a sized type cannot actually recurse without
    /// indirection (which has no scalar shape and ends the chain
    /// first), so the bound is only belt and braces.
    pub(super) fn scalar_shape(
        &self,
        ty: &mir::Ty,
        depth: u32,
    ) -> Option<ir::ScalarShape> {
        use ir::ScalarShape as S;
        if depth > 8 {
            return None;
        }
        match ty {
            mir::Ty::Bool => Some(S::Bool),
            mir::Ty::Char => Some(S::Char),
            mir::Ty::Int(it) => Some(S::Int {
                bits: it.bits(),
                signed: true,
            }),
            mir::Ty::Uint(ut) => Some(S::Int {
                bits: ut.bits(),
                signed: false,
            }),
            mir::Ty::Float(ft) => match ft.bits() {
                // An `f128` value is stored as its exact bit
                // pattern (`Value::Float128`), so its scalar
                // shape is exact like the narrower widths'.
                16 | 32 | 64 | 128 => Some(S::Float { bits: ft.bits() }),
                _ => None,
            },
            // A thin pointer flattens to its synthetic address.
            // This is what gives the pointer-carrying shapes a
            // scalar reading: a transparent wrapper over a thin
            // pointer (the `NonNull` chain) through the ADT arm
            // below, and a niche-encoded enum over one (the
            // `Option<NonNull<T>>` family) through
            // `niche_enum_shape`.
            mir::Ty::RawPtr(..) | mir::Ty::Ref(..) if is_thin_ptr_ty(ty) => {
                Some(S::Ptr)
            }
            // A function pointer flattens to its synthetic
            // address, like a thin data pointer: the pointer's
            // own bit pattern is the payload, which is what a
            // niche-encoded enum over a function pointer
            // (`Option<fn()>`) stores. The leaf is its own
            // kind so construction recovers the callable value
            // only at function-pointer destinations.
            mir::Ty::FnPtr(..) => Some(S::FnPtr),
            // A fat pointer flattens to its two words (see
            // [`ir::ScalarShape::FatPtr`]): the data pointer's
            // synthetic address in the low word and the metadata
            // in the high word -- an element count for a slice
            // or string pointee, a vtable identity for a `dyn`
            // pointee.
            mir::Ty::RawPtr(..) | mir::Ty::Ref(..) if is_fat_ptr_ty(ty) => {
                let meta = match ptr_pointee(ty) {
                    Some(mir::Ty::Dynamic(..)) => ir::FatMeta::Vtable,
                    _ => ir::FatMeta::Len,
                };
                Some(S::FatPtr(meta))
            }
            mir::Ty::Array(elem, mir::ArrayLen::Known(n)) => {
                let elem = self.scalar_shape(elem, depth + 1)?;
                let total = elem.bits().checked_mul(*n as u32)?;
                if total == 0 || total > 128 {
                    return None;
                }
                Some(S::Array(Box::new(elem), *n))
            }
            // A tuple of scalar-shaped elements: the fields
            // sit at the offsets of the tuple's recorded
            // layout — the compiler may reorder a tuple's
            // fields and insert padding, so declaration
            // order is not the storage order and a shape is
            // sound only at the recorded offsets. Every
            // field must be scalar-shaped at exactly its
            // recorded span's width (see
            // [`Self::field_shape_at_span`]) and the spans
            // must not overlap; a tuple type with no
            // recorded layout has no shape.
            mir::Ty::Tuple(ts) => {
                if ts.is_empty() {
                    return None;
                }
                let tag = crate::builtins::ty_tag(ty, self.names);
                let (size, spans) = self.tuple_layouts.get(&tag)?;
                let size = *size;
                if size == 0 || size > 16 || spans.len() != ts.len() {
                    return None;
                }
                let mut fields = Vec::with_capacity(ts.len());
                let mut taken: Vec<(u64, u64)> = Vec::new();
                for (t, &(off, span)) in ts.iter().zip(spans.iter()) {
                    if span == 0 || off + span > size {
                        return None;
                    }
                    if taken.iter().any(|(o, s)| off < o + s && *o < off + span) {
                        return None;
                    }
                    taken.push((off, span));
                    let s = self.field_shape_at_span(t, span, depth)?;
                    fields.push(((off * 8) as u32, s));
                }
                Some(S::Fields {
                    total_bits: (size * 8) as u32,
                    fields,
                })
            }
            mir::Ty::Adt(def, args) => {
                // A `MaybeUninit<T>` value holds its payload at
                // field 0 in this model -- the slot its pointer
                // accesses denote and its constructions store --
                // not at the declared `value` field's index, so a
                // transmute through the union flattens and builds
                // that slot: the payload's own shape behind a
                // one-field wrapper at index 0. Without a
                // scalar-shaped payload the union has no shape,
                // as before.
                if matches!(
                    self.def_display_name(def).as_str(),
                    "std::mem::MaybeUninit" | "core::mem::MaybeUninit"
                ) {
                    if args.len() != 1 {
                        return None;
                    }
                    let inner = self.scalar_shape(&args[0], depth + 1)?;
                    return Some(S::AdtField {
                        index: 0,
                        count: 1,
                        fill: Vec::new(),
                        inner: Box::new(inner),
                    });
                }
                let shape = self.adt_shapes.get(def)?;
                if shape.variants.len() != 1 {
                    if let Some(s) =
                        self.niche_enum_shape(def, args, shape, depth)
                    {
                        return Some(s);
                    }
                    if let Some(s) = self.niche_enum_multi_shape(
                        def, args, shape, depth,
                    ) {
                        return Some(s);
                    }
                    if let Some(s) = self.fieldless_enum_shape(def) {
                        return Some(s);
                    }
                    return self.tagged_enum_shape(def, args, shape, depth);
                }
                let fields = self.instantiated_field_tys(def, args, 0)?;
                // A single array field whose length is a const
                // parameter the type representation does not
                // carry (the `Simd<T, N>` storage field): the
                // instance's recorded layout size is the array's
                // byte size — a one-field struct's size is its
                // field's size, the field size already being a
                // multiple of the struct's alignment — so the
                // element count is the size over the element
                // width.
                if fields.len() == 1 {
                    if let mir::Ty::Array(elem, len) = &fields[0] {
                        if !matches!(len, mir::ArrayLen::Known(_)) {
                            if let Some(elem_shape) = self.scalar_shape(elem, depth + 1) {
                                let elem_bits = elem_shape.bits();
                                if elem_bits > 0 {
                                    let tag = crate::builtins::ty_tag(
                                        &mir::Ty::Adt(*def, args.to_vec()),
                                        self.names,
                                    );
                                    if let Some((size, _)) = self.adt_layouts.get(&tag) {
                                        let total = u128::from(*size) * 8;
                                        if total > 0
                                            && total <= 128
                                            && total % u128::from(elem_bits) == 0
                                        {
                                            return Some(S::AdtField {
                                                index: 0,
                                                count: 1,
                                                fill: Vec::new(),
                                                inner: Box::new(S::Array(
                                                    Box::new(elem_shape),
                                                    (total / u128::from(elem_bits)) as u64,
                                                )),
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Several scalar-shaped fields: the
                // whole-storage shape at the recorded layout
                // offsets (a packed struct transmuted to a
                // byte array is the canonical case). The
                // one-payload loop below cannot describe it,
                // and where this fails no other arm can
                // either, so its result is final.
                if fields
                    .iter()
                    .filter(|fty| {
                        matches!(
                            self.scalar_shape(fty, depth + 1),
                            Some(fs) if fs.bits() > 0
                        )
                    })
                    .count()
                    >= 2
                {
                    return self.adt_fields_shape(def, args, &fields, depth);
                }
                let mut payload: Option<(u32, ir::ScalarShape)> = None;
                let mut fill = Vec::new();
                for (i, fty) in fields.iter().enumerate() {
                    match self.scalar_shape(fty, depth + 1) {
                        Some(fs) if fs.bits() > 0 => {
                            if payload.is_some() {
                                return None;
                            }
                            payload = Some((i as u32, fs));
                        }
                        _ => {
                            let zv = self.zst_value(fty, 0)?;
                            fill.push((i as u32, value_to_constant(&zv)?));
                        }
                    }
                }
                let (index, inner) = payload?;
                Some(S::AdtField {
                    index,
                    count: fields.len(),
                    fill,
                    inner: Box::new(inner),
                })
            }
            _ => None,
        }
    }

    /// The scalar shape of a niche-encoded enum instance: an
    /// enum whose layout stores the discriminant in the
    /// invalid values of its one payload field (see
    /// `mir::EnumNiche`), such as `Option<NonZero<T>>`. The
    /// payload variant must carry exactly that one field and
    /// every other variant must be fieldless and
    /// niche-encoded. The niche scalar is usually the
    /// payload field itself (the window is the field's whole
    /// pattern); it can also be a sub-window of a wider
    /// payload field — the payload is then itself a tagged
    /// enum whose tag is the niche — when the recorded
    /// layouts confirm the field spans the enum's whole
    /// storage at offset 0, so the shape's pattern is the
    /// field's pattern. A fieldless-enum payload takes its
    /// shape at the niche scalar's own width, the width its
    /// tag is stored at. Anything else — several payload
    /// variants, a niche scalar reaching outside the
    /// payload field's pattern — has no shape and stays
    /// unsupported.
    fn niche_enum_shape(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
        shape: &mir::AdtShape,
        depth: u32,
    ) -> Option<ir::ScalarShape> {
        let tag = crate::builtins::ty_tag(&mir::Ty::Adt(*def, args.to_vec()), self.names);
        let niche = self.adt_niches.get(&tag)?;
        let (first, last) = niche.niche_variants;
        let niche_count = last.checked_sub(first)? + 1;
        if shape.variants.len() as u32 != niche_count + 1 {
            return None;
        }
        let untagged = niche.untagged_variant as usize;
        if untagged >= shape.variants.len()
            || (first..=last).contains(&niche.untagged_variant)
        {
            return None;
        }
        for (i, fields) in shape.variants.iter().enumerate() {
            let expected = if i == untagged { 1 } else { 0 };
            if fields.len() != expected {
                return None;
            }
        }
        let fields = self.instantiated_field_tys(def, args, niche.untagged_variant)?;
        let [fty] = fields.as_slice() else {
            return None;
        };
        // The payload field's own shape: the field's shape
        // when its width is the niche scalar's, a wider
        // field whose pattern holds the scalar in a
        // sub-window (sound only when the field spans the
        // enum's whole storage at offset 0), or — for a
        // fieldless-enum field — that enum's shape at the
        // scalar's width (the width its tag is stored at,
        // not the width inferred from its discriminant
        // range), provided the discriminants' patterns at
        // that width stay distinct.
        let inner = match self.scalar_shape(fty, depth + 1) {
            Some(s) if s.bits() == niche.tag_bits => s,
            Some(s) if s.bits() > niche.tag_bits => {
                if !self.niche_field_spans_enum(
                    def,
                    args,
                    niche.untagged_variant,
                ) {
                    return None;
                }
                s
            }
            _ => {
                let mir::Ty::Adt(fdef, _) = fty else {
                    return None;
                };
                let discriminants =
                    self.fieldless_enum_discriminants(fdef)?;
                let mask = if niche.tag_bits >= 128 {
                    u128::MAX
                } else {
                    (1u128 << niche.tag_bits) - 1
                };
                let mut seen = std::collections::HashSet::new();
                if discriminants.is_empty()
                    || !discriminants
                        .iter()
                        .all(|d| seen.insert((*d as u128) & mask))
                {
                    return None;
                }
                ir::ScalarShape::FieldlessEnum {
                    bits: niche.tag_bits,
                    discriminants,
                }
            }
        };
        if inner.bits() == 0 {
            return None;
        }
        let niche_shift = (niche.niche_offset * 8) as u32;
        if niche_shift + niche.tag_bits > inner.bits() {
            return None;
        }
        if niche_shift > 0
            && !self.niche_field_spans_enum(
                def,
                args,
                niche.untagged_variant,
            )
        {
            return None;
        }
        Some(ir::ScalarShape::NicheEnum {
            inner: Box::new(inner),
            untagged_variant: niche.untagged_variant,
            niche_variants: niche.niche_variants,
            niche_start: niche.niche_start,
            niche_shift,
            niche_bits: niche.tag_bits,
        })
    }

    /// The scalar shape of a niche-encoded enum whose
    /// payload-carrying variant carries several fields (see
    /// [`ir::ScalarShape::NicheEnumFields`]): the variant's
    /// fields at their recorded layout offsets, the
    /// discriminant in the niche window inside the recorded
    /// niche field's span. [`Self::niche_enum_shape`] is the
    /// single-field case of the same encoding; the conditions
    /// mirror it — exactly one non-fieldless variant (the
    /// untagged one), every other variant fieldless, the
    /// variant count matching the recorded niche range, the
    /// niche window inside one payload field's span.
    /// Anything else — a
    /// payload field with no shape at its span, overlapping
    /// spans, a window reaching outside the niche field —
    /// has no shape and stays unsupported.
    fn niche_enum_multi_shape(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
        shape: &mir::AdtShape,
        depth: u32,
    ) -> Option<ir::ScalarShape> {
        let tag = crate::builtins::ty_tag(
            &mir::Ty::Adt(*def, args.to_vec()),
            self.names,
        );
        let niche = self.adt_niches.get(&tag)?;
        let (first, last) = niche.niche_variants;
        let niche_count = last.checked_sub(first)? + 1;
        // The recorded range spans every variant that needs a
        // discriminant; the untagged variant's own index sits
        // inside it as a dead slot when the payload variant is
        // not at either end of the declaration order (the
        // range then covers every variant), and outside it
        // otherwise (one more variant than the range covers).
        let untagged = niche.untagged_variant as usize;
        let in_range = (first..=last).contains(&niche.untagged_variant);
        if untagged >= shape.variants.len()
            || shape.variants.len() as u32
                != niche_count + u32::from(!in_range)
        {
            return None;
        }
        for (i, fields) in shape.variants.iter().enumerate() {
            if i == untagged {
                if fields.len() <= 1 {
                    return None;
                }
            } else if !fields.is_empty() {
                return None;
            }
        }
        let (size, _) = *self.adt_layouts.get(&tag)?;
        if size == 0 || size > 16 {
            return None;
        }
        let layouts = self.adt_field_layouts.get(&tag)?;
        let vlayout = layouts.get(untagged)?;
        let ftys =
            self.instantiated_field_tys(def, args, niche.untagged_variant)?;
        if vlayout.len() != ftys.len() {
            return None;
        }
        // The niche window (an absolute bit range of the
        // enum) must sit inside one payload field's span:
        // that field's pattern is what the window cuts into.
        // (The recorded niche field index names the field in
        // the layout's own field order, which is not the
        // declaration order the field layout rows use, so
        // the window is matched by span instead.)
        let win_lo = niche.niche_offset * 8;
        let win_hi = win_lo + u64::from(niche.tag_bits);
        if niche.tag_bits == 0 || win_hi > size * 8 {
            return None;
        }
        let mut out_fields = Vec::new();
        let mut fill = Vec::new();
        let mut spans: Vec<(u64, u64)> = Vec::new();
        for (i, fty) in ftys.iter().enumerate() {
            let (off, fsize) = vlayout[i];
            if fsize == 0 {
                let zv = self.zst_value(fty, 0)?;
                fill.push((i as u32, value_to_constant(&zv)?));
                continue;
            }
            if off + fsize > size {
                return None;
            }
            let fs = self.field_shape_at_span(fty, fsize, depth)?;
            if spans
                .iter()
                .any(|(o, s)| off < o + s && *o < off + fsize)
            {
                return None;
            }
            spans.push((off, fsize));
            out_fields.push((i as u32, (off * 8) as u32, fs));
        }
        if !out_fields.iter().any(|(_, off, s)| {
            u64::from(*off) <= win_lo
                && win_hi <= u64::from(*off) + u64::from(s.bits())
        }) {
            return None;
        }
        Some(ir::ScalarShape::NicheEnumFields {
            total_bits: (size * 8) as u32,
            count: ftys.len(),
            fill,
            fields: out_fields,
            untagged_variant: niche.untagged_variant,
            niche_variants: niche.niche_variants,
            niche_start: niche.niche_start,
            niche_shift: win_lo as u32,
            niche_bits: niche.tag_bits,
        })
    }

    /// Whether the payload field of the niche-encoded enum
    /// instance `def`/`args` spans the instance's whole
    /// storage at offset 0, per the recorded layouts: the
    /// condition under which the shape's bit pattern is the
    /// payload field's own pattern, so a niche scalar inside
    /// the field is a sub-window of the pattern.
    fn niche_field_spans_enum(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
        untagged_variant: u32,
    ) -> bool {
        let tag = crate::builtins::ty_tag(
            &mir::Ty::Adt(*def, args.to_vec()),
            self.names,
        );
        let Some((size, _)) = self.adt_layouts.get(&tag) else {
            return false;
        };
        let Some(layouts) = self.adt_field_layouts.get(&tag) else {
            return false;
        };
        let Some(v) = layouts.get(untagged_variant as usize) else {
            return false;
        };
        v.len() == 1 && v[0].0 == 0 && v[0].1 == *size
    }

    /// The scalar shape of a field whose recorded storage
    /// span is `span` bytes: the field's own shape when it
    /// accounts for exactly the span's bits, or — for a
    /// fieldless enum field, whose inferred shape width
    /// follows its discriminant range — that shape at the
    /// span's width, the width its tag is stored at,
    /// provided the discriminants' patterns at that width
    /// stay distinct.
    fn field_shape_at_span(
        &self,
        fty: &mir::Ty,
        span: u64,
        depth: u32,
    ) -> Option<ir::ScalarShape> {
        let want = (span * 8) as u32;
        if want == 0 || want > 128 {
            return None;
        }
        if let Some(s) = self.scalar_shape(fty, depth + 1) {
            if s.bits() == want {
                return Some(s);
            }
        }
        let mir::Ty::Adt(fdef, _) = fty else {
            return None;
        };
        let discriminants = self.fieldless_enum_discriminants(fdef)?;
        if discriminants.is_empty() {
            return None;
        }
        let mask = if want == 128 {
            u128::MAX
        } else {
            (1u128 << want) - 1
        };
        let mut seen = std::collections::HashSet::new();
        if !discriminants
            .iter()
            .all(|d| seen.insert((*d as u128) & mask))
        {
            return None;
        }
        Some(ir::ScalarShape::FieldlessEnum {
            bits: want,
            discriminants,
        })
    }

    /// The scalar shape of a single-variant ADT (a struct)
    /// with several scalar-shaped fields, at the fields'
    /// recorded layout offsets: every field must be
    /// scalar-shaped at exactly its storage span's width
    /// (see [`Self::field_shape_at_span`]) or zero-sized
    /// with a uniquely determined value, the fields' spans
    /// must not overlap, and the instance's layout must be
    /// recorded. Bits no field accounts for (padding)
    /// flatten as zero and are ignored on construction.
    fn adt_fields_shape(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
        fields: &[mir::Ty],
        depth: u32,
    ) -> Option<ir::ScalarShape> {
        let tag = crate::builtins::ty_tag(
            &mir::Ty::Adt(*def, args.to_vec()),
            self.names,
        );
        let (size, _) = *self.adt_layouts.get(&tag)?;
        if size == 0 || size > 16 {
            return None;
        }
        let layouts = self.adt_field_layouts.get(&tag)?;
        if layouts.len() != 1 {
            return None;
        }
        let vlayout = &layouts[0];
        if vlayout.len() != fields.len() {
            return None;
        }
        let mut out_fields = Vec::new();
        let mut fill = Vec::new();
        let mut spans: Vec<(u64, u64)> = Vec::new();
        for (i, fty) in fields.iter().enumerate() {
            let (off, fsize) = vlayout[i];
            if fsize == 0 {
                let zv = self.zst_value(fty, 0)?;
                fill.push((i as u32, value_to_constant(&zv)?));
                continue;
            }
            if off + fsize > size {
                return None;
            }
            let fs = self.field_shape_at_span(fty, fsize, depth)?;
            if spans
                .iter()
                .any(|(o, s)| off < o + s && *o < off + fsize)
            {
                return None;
            }
            spans.push((off, fsize));
            out_fields.push((i as u32, (off * 8) as u32, fs));
        }
        if out_fields.len() < 2 {
            return None;
        }
        Some(ir::ScalarShape::AdtFields {
            total_bits: (size * 8) as u32,
            count: fields.len(),
            fill,
            fields: out_fields,
        })
    }

    /// The scalar shape of a direct-tag enum instance: the
    /// recorded tag (offset, width, per-variant
    /// discriminants) plus each variant's scalar-shaped
    /// fields at their recorded layout offsets. Every
    /// variant's fields must be scalar-shaped at exactly
    /// their storage spans (see
    /// [`Self::field_shape_at_span`]) or zero-sized with a
    /// uniquely determined value, and no field span may
    /// overlap the tag's span or another field's.
    fn tagged_enum_shape(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
        shape: &mir::AdtShape,
        depth: u32,
    ) -> Option<ir::ScalarShape> {
        let tag = crate::builtins::ty_tag(
            &mir::Ty::Adt(*def, args.to_vec()),
            self.names,
        );
        let etag = self.adt_tag_layouts.get(&tag)?;
        let (size, _) = *self.adt_layouts.get(&tag)?;
        if size == 0
            || size > 16
            || etag.size == 0
            || etag.offset + etag.size > size
        {
            return None;
        }
        if etag.discriminants.len() != shape.variants.len() {
            return None;
        }
        let layouts = self.adt_field_layouts.get(&tag)?;
        if layouts.len() != shape.variants.len() {
            return None;
        }
        let mut variants = Vec::with_capacity(shape.variants.len());
        for v in 0..shape.variants.len() {
            let ftys = self.instantiated_field_tys(def, args, v as u32)?;
            let vlayout = &layouts[v];
            if vlayout.len() != ftys.len() {
                return None;
            }
            let mut out_fields = Vec::new();
            let mut fill = Vec::new();
            let mut spans: Vec<(u64, u64)> =
                vec![(etag.offset, etag.size)];
            for (i, fty) in ftys.iter().enumerate() {
                let (off, fsize) = vlayout[i];
                if fsize == 0 {
                    let zv = self.zst_value(fty, 0)?;
                    fill.push((i as u32, value_to_constant(&zv)?));
                    continue;
                }
                if off + fsize > size {
                    return None;
                }
                let fs = self.field_shape_at_span(fty, fsize, depth)?;
                if spans
                    .iter()
                    .any(|(o, s)| off < o + s && *o < off + fsize)
                {
                    return None;
                }
                spans.push((off, fsize));
                out_fields.push((i as u32, (off * 8) as u32, fs));
            }
            variants.push(ir::TaggedVariantShape {
                count: ftys.len(),
                fill,
                fields: out_fields,
            });
        }
        Some(ir::ScalarShape::TaggedEnum {
            total_bits: (size * 8) as u32,
            tag_offset: (etag.offset * 8) as u32,
            tag_bits: (etag.size * 8) as u32,
            discriminants: etag.discriminants.clone(),
            variants,
        })
    }

    /// The scalar shape of a fieldless enum: the whole
    /// value is its discriminant, held in a tag whose width
    /// is the smallest standard integer width covering the
    /// discriminant range (the width rule the compiler's
    /// own tag selection follows for an enum without an
    /// explicit `repr`). An enum carrying an explicit wider
    /// `repr` is assigned the smaller inferred width, so a
    /// transmute pair against the wider integer does not
    /// take the generic path and stays with the dedicated
    /// discriminant arms.
    fn fieldless_enum_shape(
        &self,
        def: &mir::AdtDefId,
    ) -> Option<ir::ScalarShape> {
        let discriminants = self.fieldless_enum_discriminants(def)?;
        if discriminants.is_empty() {
            return None;
        }
        let lo = *discriminants.iter().min().expect("non-empty");
        let hi = *discriminants.iter().max().expect("non-empty");
        let bits = [8u32, 16, 32, 64, 128].into_iter().find(|w| {
            if *w == 128 {
                return true;
            }
            let w = *w;
            (lo >= 0 && (hi as u128) < (1u128 << w))
                || (lo >= -(1i128 << (w - 1)) && hi < (1i128 << (w - 1)))
        })?;
        Some(ir::ScalarShape::FieldlessEnum {
            bits,
            discriminants,
        })
    }

    /// The scalar shape of a constant operand, whose MIR form
    /// carries no type: the constant's own kind gives the leaf
    /// kind, and the width is the transmute target's width (a
    /// transmute's two sides agree on size, so the source's
    /// width is the target's). A tuple constant is read as an
    /// array of its first element's kind; a bytes constant as an
    /// array of `u8`. Anything else has no derivable shape.
    fn constant_scalar_shape(
        &self,
        v: &ir::Operand,
        width: u32,
    ) -> Option<ir::ScalarShape> {
        use ir::ScalarShape as S;
        let ir::Operand::Constant(c) = v else {
            return None;
        };
        match c {
            ir::Constant::Int(_) => Some(S::Int {
                bits: width,
                signed: true,
            }),
            ir::Constant::Uint(_) => Some(S::Int {
                bits: width,
                signed: false,
            }),
            ir::Constant::Float(..) => match width {
                16 | 32 | 64 => Some(S::Float { bits: width }),
                _ => None,
            },
            ir::Constant::Float128(_) => {
                (width == 128).then_some(S::Float { bits: 128 })
            }
            ir::Constant::Bool(_) => (width == 8).then_some(S::Bool),
            ir::Constant::Bytes(b) => {
                (width == 8 * b.len() as u32).then(|| {
                    S::Array(
                        Box::new(S::Int {
                            bits: 8,
                            signed: false,
                        }),
                        b.len() as u64,
                    )
                })
            }
            ir::Constant::Tuple(fields) if !fields.is_empty() => {
                let n = fields.len() as u32;
                if width == 0 || width % n != 0 {
                    return None;
                }
                let ew = width / n;
                let elem = match &fields[0] {
                    ir::Constant::Int(_) => S::Int {
                        bits: ew,
                        signed: true,
                    },
                    ir::Constant::Uint(_) => S::Int {
                        bits: ew,
                        signed: false,
                    },
                    ir::Constant::Float(..) => match ew {
                        16 | 32 | 64 => S::Float { bits: ew },
                        _ => return None,
                    },
                    ir::Constant::Float128(_) => match ew {
                        128 => S::Float { bits: ew },
                        _ => return None,
                    },
                    ir::Constant::Bool(_) => {
                        if ew != 8 {
                            return None;
                        }
                        S::Bool
                    }
                    _ => return None,
                };
                let same = fields.iter().all(|f| {
                    matches!(
                        (f, &fields[0]),
                        (ir::Constant::Int(_), ir::Constant::Int(_))
                            | (
                                ir::Constant::Uint(_),
                                ir::Constant::Uint(_)
                            )
                            | (
                                ir::Constant::Float(..),
                                ir::Constant::Float(..)
                            )
                            | (
                                ir::Constant::Float128(_),
                                ir::Constant::Float128(_)
                            )
                            | (ir::Constant::Bool(_), ir::Constant::Bool(_))
                    )
                });
                if same {
                    Some(S::Array(Box::new(elem), u64::from(n)))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// The [`WrapperShape`] of the struct `def` instantiated
    /// with `args`, or `None` when it is not a transparent
    /// wrapper (not a struct, more than one payload field, or a
    /// field that is neither the payload nor a uniquely valued
    /// zero-sized type).
    fn wrapper_shape(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
    ) -> Option<WrapperShape> {
        self.wrapper_shape_matching(def, args, is_thin_ptr_ty)
    }

    /// The wrapper shape whose payload field is a fat pointer
    /// (`NonNull<str>`, `NonNull<[T]>`, `NonNull<dyn Trait>`):
    /// the same transparent-wrapper rule as
    /// [`Self::wrapper_shape`], with the metadata-carrying
    /// pointer as the payload.
    fn fat_wrapper_shape(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
    ) -> Option<WrapperShape> {
        self.wrapper_shape_matching(def, args, is_fat_ptr_ty)
    }

    /// The transparent-wrapper shape whose payload field is a
    /// fieldless enum, together with that enum's discriminant
    /// table (`mem::Alignment` over its `AlignmentEnum` is the
    /// canonical case): the same one-payload rule as
    /// [`Self::wrapper_shape`], with the enum as the payload.
    fn enum_wrapper_shape(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
    ) -> Option<(Vec<i128>, WrapperShape)> {
        let shape = self.adt_shapes.get(def)?;
        if shape.kind != mir::AdtShapeKind::Struct || shape.variants.len() != 1 {
            return None;
        }
        let fields = self.instantiated_field_tys(def, args, 0)?;
        let mut payload: Option<(usize, mir::Ty, Vec<i128>)> = None;
        let mut fill = Vec::new();
        for (i, fty) in fields.iter().enumerate() {
            let table = match fty {
                mir::Ty::Adt(fdef, _) => self.fieldless_enum_discriminants(fdef),
                _ => None,
            };
            if let Some(table) = table {
                if payload.is_some() {
                    return None;
                }
                payload = Some((i, fty.clone(), table));
            } else if let Some(zv) = self.zst_value(fty, 0) {
                fill.push((i, value_to_constant(&zv)?));
            } else {
                return None;
            }
        }
        let (payload_index, payload_ty, table) = payload?;
        Some((
            table,
            WrapperShape {
                field_count: fields.len(),
                payload_index,
                payload_ty,
                fill,
            },
        ))
    }

    fn wrapper_shape_matching(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
        payload_ok: fn(&mir::Ty) -> bool,
    ) -> Option<WrapperShape> {
        let shape = self.adt_shapes.get(def)?;
        if shape.kind != mir::AdtShapeKind::Struct || shape.variants.len() != 1 {
            return None;
        }
        let fields = self.instantiated_field_tys(def, args, 0)?;
        let mut payload: Option<(usize, mir::Ty)> = None;
        let mut fill = Vec::new();
        for (i, fty) in fields.iter().enumerate() {
            if payload_ok(fty) {
                if payload.is_some() {
                    return None;
                }
                payload = Some((i, fty.clone()));
            } else if let Some(zv) = self.zst_value(fty, 0) {
                fill.push((i, value_to_constant(&zv)?));
            } else {
                return None;
            }
        }
        let (payload_index, payload_ty) = payload?;
        Some(WrapperShape {
            field_count: fields.len(),
            payload_index,
            payload_ty,
            fill,
        })
    }

    /// The field types of one variant of `def` instantiated with
    /// `args`: recovered field types (already instantiated for
    /// this `args`) win over the shape's declared types, whose
    /// parameter indices can misalign — the same resolution as
    /// `unique_value`.
    /// The type a debugger variable's recorded projection reaches,
    /// starting from its local's declared type: the walk the
    /// debugger interface's variable table is built with, so an
    /// entry can carry the projected type's rendered tag. Only
    /// field and downcast steps are walked (the steps the table
    /// itself carries); anything else leaves the type unknown.
    pub(super) fn debug_projected_ty(
        &self,
        base: &mir::Ty,
        proj: &[crate::RefProj],
    ) -> Option<mir::Ty> {
        let mut ty = base.clone();
        let mut variant = 0u32;
        for step in proj {
            match step {
                crate::RefProj::Downcast(v) => variant = *v as u32,
                crate::RefProj::Field(i) => {
                    ty = match &ty {
                        mir::Ty::Tuple(ts) => ts.get(*i)?.clone(),
                        mir::Ty::Array(elem, _) => (**elem).clone(),
                        mir::Ty::Adt(def, args) => self
                            .instantiated_field_tys(def, args, variant)?
                            .get(*i)?
                            .clone(),
                        _ => return None,
                    };
                }
                _ => return None,
            }
        }
        Some(ty)
    }

    pub(super) fn instantiated_field_tys(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
        variant: u32,
    ) -> Option<Vec<mir::Ty>> {
        let shape = self.adt_shapes.get(def)?;
        let declared = shape.variants.get(variant as usize)?;
        let instance = self.instance_field_tys(def, args, variant);
        Some(
            declared
                .iter()
                .enumerate()
                .map(|(fi, fty)| {
                    self.adt_fields
                        .get(*def, args, variant, fi as u32)
                        .cloned()
                        .or_else(|| instance.and_then(|fs| fs.get(fi)).cloned())
                        .unwrap_or_else(|| subst_ty(fty, args))
                })
                .collect(),
        )
    }

    /// The driver-recorded field types of one variant of the ADT
    /// instance `def` instantiated with `args` (see
    /// `mir::AdtInstanceShapes`), or `None` when the driver
    /// recorded no entry for the instance — the common case, in
    /// which the generic shape's field types need no resolution
    /// beyond substitution.
    fn instance_field_tys(
        &self,
        def: &mir::AdtDefId,
        args: &[mir::Ty],
        variant: u32,
    ) -> Option<&[mir::Ty]> {
        let tag = crate::builtins::ty_tag(&mir::Ty::Adt(*def, args.to_vec()), self.names);
        self.adt_instance_shapes
            .get(&tag)?
            .variants
            .get(variant as usize)
            .map(Vec::as_slice)
    }

}
