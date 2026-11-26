//! Converter: type conversion.


use sparagmos_mir as mir;

use crate::ir;

use super::converter::*;
use super::errors::*;
use super::helpers::*;


impl<'a> Converter<'a> {
    // ----- types -----------------------------------------------------------

    /// Display name for a `DefId` (function items, closure bodies,
    /// ADT types): the driver's full name table first, then the
    /// table of converted bodies, then the raw id.
    /// The program key of the `static` item `def`: its display
    /// name, disambiguated with the `DefId` when several named
    /// items share that name (the [`program_keys`] rule applied to
    /// statics). Computed from the name table alone, so every
    /// conversion phase assigns the same key to the same static.
    pub(super) fn static_key(&self, def: &mir::DefId) -> String {
        let name = self.def_display_name(def);
        if self.names.values().filter(|n| **n == name).count() > 1 {
            format!("{name}#{def}")
        } else {
            name
        }
    }

    /// The skeleton of a declared ADT field type for the
    /// interpreter's place-walk type tracking (see
    /// [`ir::Program::adt_field_tys`]): the type reduced to the
    /// structure a walk can follow without knowing the instance —
    /// an ADT keeps its definition name (its generic arguments
    /// are erased, exactly as [`Self::convert_ty`] erases them),
    /// a reference or raw pointer keeps its pointee's skeleton, a
    /// tuple keeps its elements' skeletons (an untrackable
    /// element becomes the unit type, which no walk step can
    /// mistake for a modelled type), and every other type goes
    /// through the ordinary conversion. A field whose type
    /// mentions a generic parameter or a type alias at its head
    /// has no instance-independent skeleton: `None`, and tracking
    /// through that field ends.
    pub(super) fn field_ty_skeleton(&self, ty: &mir::Ty) -> Option<ir::Ty> {
        match ty {
            mir::Ty::Param(..) | mir::Ty::Alias(..) => None,
            mir::Ty::Adt(def, _) => Some(ir::Ty::Adt(self.def_display_name(def))),
            mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => Some(
                ir::Ty::Ref(Box::new(self.field_ty_skeleton(inner)?)),
            ),
            mir::Ty::Tuple(ts) => {
                let mut out = Vec::with_capacity(ts.len());
                for t in ts {
                    out.push(self.field_ty_skeleton(t).unwrap_or(ir::Ty::Unit));
                }
                Some(ir::Ty::Tuple(out))
            }
            other => self.convert_ty(other, "an ADT field type").ok(),
        }
    }

    pub(super) fn def_display_name(&self, def: &mir::DefId) -> String {
        self.names
            .get(def)
            .or_else(|| self.def_names.get(def))
            .cloned()
            .unwrap_or_else(|| format!("{def:?}"))
    }

    /// The program key of the body a `DefId` names, for references
    /// that must land on one specific converted body (closure
    /// values and closure types, and `FnDef` constants — a function
    /// item materialized as a value names the instance its
    /// constant refers to, not the definition's first instance):
    /// the converted-body table first — it is keyed by the exact
    /// (definition, instance) id the driver assigned, so two
    /// instances of one definition resolve to their own bodies —
    /// and the display name only as a fallback for definitions
    /// with no body in the program.
    pub(super) fn body_key(&self, def: &mir::DefId) -> String {
        self.def_names
            .get(def)
            .cloned()
            .unwrap_or_else(|| self.def_display_name(def))
    }

    /// The program key a function-item *value* names (an `FnDef`
    /// type constant materialised as a callable value, e.g. a
    /// trait method coerced to a function pointer). The constant
    /// names the item's definition, whose own key is the body
    /// key only for a free function: for a trait method the body
    /// in the program is the impl instance for the constant's
    /// `Self` type, resolved exactly as a trait-method call
    /// resolves it (see `find_impl_key_for`). When neither
    /// resolution lands, the definition's key stands (the call
    /// then fails naming the item, as before).
    pub(super) fn fn_def_value_key(&self, def: &mir::DefId, args: &[mir::Ty]) -> String {
        let key = self.body_key(def);
        if self.bodies.contains_key(&key) {
            return key;
        }
        let name = self.def_display_name(def);
        if let Some((trait_path, method)) = name.rsplit_once("::") {
            if !trait_path.is_empty() && !method.is_empty() {
                if let Some(impl_key) =
                    self.find_impl_key_for(trait_path, method, args)
                {
                    return impl_key;
                }
            }
        }
        key
    }

    /// The name of a concrete type as it appears in the `Self`
    /// position of its impl method names in the program
    /// (`<{name} as Trait>::method`): the definition's display name
    /// with its generic arguments rendered the way a call's generic
    /// arguments render them, for a named type; the surface-syntax
    /// rendering otherwise. The generic arguments are part of the
    /// name: a `dyn` value's concrete tag is compared against the
    /// rendered generic argument of a downcast, which carries them.
    pub(super) fn concrete_ty_name(&self, ty: &mir::Ty) -> String {
        match ty {
            mir::Ty::Adt(def, args)
            | mir::Ty::Closure(def, args)
            | mir::Ty::Coroutine(def, args) => {
                let base = self.def_display_name(def);
                if args.is_empty() {
                    base
                } else {
                    let parts: Vec<String> = args
                        .iter()
                        .map(|t| crate::builtins::ty_tag(t, self.names))
                        .collect();
                    format!("{base}<{}>", parts.join(", "))
                }
            }
            mir::Ty::FnDef(def, _) => self.def_display_name(def),
            other => crate::builtins::ty_tag(other, self.names),
        }
    }

    /// The field-entry depth of a pointer cast from `source` to
    /// `target` that crosses single-field wrappers to reach the
    /// wrapped type itself, or `None` when the cast is not of
    /// that shape. The wrappers are the transparent cell types
    /// (`UnsafeCell<T>` / `MaybeUninit<T>`, recognized by name
    /// because their field convention is definitional) and any
    /// other struct with exactly one field whose type is the
    /// next layer (the `as_array` newtype idiom).
    /// Decided from the cast's two types, both known at conversion
    /// time: the source pointee is peeled wrapper by wrapper until
    /// it is exactly the target pointee.
    /// Whether two types are the same for cast classification:
    /// structural equality, except that an array length that is
    /// a generic parameter (or otherwise unevaluated) on either
    /// side matches any length -- the recorded field type of a
    /// generic definition carries the parameter, the cast's
    /// target in an instantiated caller carries the value, and
    /// the field-entry decision does not depend on the length.
    fn cast_ty_matches(a: &mir::Ty, b: &mir::Ty) -> bool {
        use mir::Ty as T;
        match (a, b) {
            (T::Array(ea, la), T::Array(eb, lb)) => {
                Self::cast_ty_matches(ea, eb)
                    && (la == lb
                        || matches!(la, mir::ArrayLen::Param(_) | mir::ArrayLen::Unevaluated(_))
                        || matches!(lb, mir::ArrayLen::Param(_) | mir::ArrayLen::Unevaluated(_)))
            }
            (T::Adt(da, aa), T::Adt(db, ab)) => {
                da == db
                    && aa.len() == ab.len()
                    && aa.iter().zip(ab.iter()).all(|(x, y)| Self::cast_ty_matches(x, y))
            }
            (T::Tuple(ta), T::Tuple(tb)) => {
                ta.len() == tb.len()
                    && ta.iter().zip(tb.iter()).all(|(x, y)| Self::cast_ty_matches(x, y))
            }
            (T::Ref(_, ta, ma), T::Ref(_, tb, mb)) => ma == mb && Self::cast_ty_matches(ta, tb),
            (T::RawPtr(ta, ma), T::RawPtr(tb, mb)) => ma == mb && Self::cast_ty_matches(ta, tb),
            (T::Slice(ea), T::Slice(eb)) => Self::cast_ty_matches(ea, eb),
            _ => a == b,
        }
    }

    pub(super) fn cast_wrapper_entry_depth(
        &self,
        source: &mir::Ty,
        target: &mir::Ty,
    ) -> Option<u32> {
        match self.cast_entry_depth(source, target) {
            Some((depth, false)) => Some(depth),
            _ => None,
        }
    }

    /// The retargeting a pointer cast from `source` to `target`
    /// denotes when it crosses transparent wrappers (and possibly
    /// one array layer) to reach the target pointee: the number
    /// of field-0 entries the wrapper peel denotes, and whether
    /// the peel then continues into the first element of the
    /// array the innermost peeled layer holds (the cast's target
    /// is that array's element type, as in
    /// `buf.as_mut_ptr() as *mut T` over a nested `MaybeUninit`
    /// buffer). `None` when the cast is not of that shape.
    pub(super) fn cast_entry_depth(
        &self,
        source: &mir::Ty,
        target: &mir::Ty,
    ) -> Option<(u32, bool)> {
        fn pointee(ty: &mir::Ty) -> Option<&mir::Ty> {
            match ty {
                mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => Some(inner),
                _ => None,
            }
        }
        let current = pointee(source)?.clone();
        let target = pointee(target)?;
        // A cast between slice pointers whose element types are
        // related by wrapper entry (`*mut [MaybeUninit<T>]` to
        // `*mut [T]`, the cast the array clone guard's drop
        // performs on the erased buffer): the fat pointer's base
        // denotes the first element's wrapper, so the entry depth
        // of the element types is a field-0 entry on the base.
        // The target is a slice here, so the peel must land
        // exactly on the target element type, never inside it.
        if let (mir::Ty::Slice(src_elem), mir::Ty::Slice(tgt_elem)) = (&current, target) {
            return match self.wrapper_peel((**src_elem).clone(), tgt_elem) {
                Some((depth, false)) => Some((depth, false)),
                _ => None,
            };
        }
        self.wrapper_peel(current, target)
    }

    /// The retargeting a pointer cast from `source` to `target`
    /// denotes when it is the reverse of a field-0 entry (see
    /// [`Self::cast_entry_depth`]): the target pointee is an
    /// aggregate whose field-0 peel lands exactly on the source
    /// pointee, so the cast removes that many field-0 entries
    /// from the pointer's projection and the result denotes the
    /// enclosing aggregate again. This is the cast std's B-tree
    /// node code performs when it reads an internal node back out
    /// of the leaf-typed node pointer (`node.as_ptr() as *mut
    /// InternalNode`). `None` when the target's peel never lands
    /// on the source pointee.
    pub(super) fn cast_exit_depth(
        &self,
        source: &mir::Ty,
        target: &mir::Ty,
    ) -> Option<u32> {
        fn pointee(ty: &mir::Ty) -> Option<&mir::Ty> {
            match ty {
                mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => Some(inner),
                _ => None,
            }
        }
        let source = pointee(source)?;
        let mut current = pointee(target)?.clone();
        let mut depth = 0u32;
        loop {
            let (next, _) = self.peel_step(&current)?;
            depth += 1;
            if Self::cast_ty_matches(&next, source) {
                return Some(depth);
            }
            current = next;
        }
    }

    /// One step of the field-0 peel (see [`Self::wrapper_peel`]):
    /// the type a field-0 entry into a value of type `current`
    /// denotes, when `current` is a shape such an entry crosses.
    /// Both wrapper kinds are one-field aggregates in the value
    /// model -- `UnsafeCell` by its declared shape, and
    /// `MaybeUninit` by the field-0 convention its pointer
    /// accesses use (`MaybeUninit::new` constructions store the
    /// payload at field 0 of the union value) -- so entering
    /// either appends a field-0 projection. A single-field struct
    /// whose only field is an array is entered the same way (the
    /// `as_array` idiom). A struct with at least two fields whose
    /// first field is an aggregate is a prefix struct: its first
    /// field is stored as a real sub-aggregate at the head of the
    /// value, so a cast to the first field's type denotes that
    /// field exactly as a wrapper entry denotes a wrapper's only
    /// field. `None` when no field-0 entry is denoted.
    /// The step also reports whether it was a prefix-field step
    /// (crossing a prefix struct's first field, as opposed to a
    /// transparent wrapper or a single-field array struct): the
    /// crossing that changes what a pointer denotes beyond the
    /// wrapper conventions the access points already reconcile,
    /// which the transmute decisions below key on.
    fn peel_step(&self, current: &mir::Ty) -> Option<(mir::Ty, bool)> {
        let mir::Ty::Adt(def, args) = current else {
            return None;
        };
        let name = self.def_display_name(def);
        let is_wrapper = matches!(
            name.as_str(),
            "std::cell::UnsafeCell"
                | "core::cell::UnsafeCell"
                | "std::mem::MaybeUninit"
                | "core::mem::MaybeUninit"
        );
        if is_wrapper && args.len() == 1 {
            return Some((args[0].clone(), false));
        }
        if let Some(field) = self
            .single_field_ty(def, args)
            .filter(|field| matches!(field, mir::Ty::Array(..)))
        {
            // The restriction to array fields is load-bearing:
            // scalar-field wrappers include representation
            // wrappers over values this model stores bare (the
            // atomics' alignment wrappers), where a field
            // projection would address storage that is not there.
            return Some((field, false));
        }
        self.prefix_field_ty(def, args).map(|ty| (ty, true))
    }

    /// Peel `from`'s field-0 layers (see [`Self::peel_step`])
    /// looking for an exact landing on `to`: `Some(crossed)` --
    /// whether any step crossed a prefix struct's first field --
    /// when the peel lands, `None` when it does not.
    fn peel_lands_on(&self, from: &mir::Ty, to: &mir::Ty) -> Option<bool> {
        let mut current = from.clone();
        let mut crossed = false;
        loop {
            let (next, is_prefix) = self.peel_step(&current)?;
            crossed |= is_prefix;
            if Self::cast_ty_matches(&next, to) {
                return Some(crossed);
            }
            current = next;
        }
    }

    /// Whether a cast or transmute between the pointer types
    /// `source` and `target` retargets the pointer across a
    /// prefix struct's first field, in either direction (the
    /// target pointee is reached from the source pointee by a
    /// field-0 peel that crosses a prefix struct, or the
    /// reverse). A crossing of only transparent wrappers returns
    /// false: those denotations the model's access points
    /// already reconcile, and their established handling does
    /// not change.
    pub(super) fn cast_crosses_prefix_field(
        &self,
        source: &mir::Ty,
        target: &mir::Ty,
    ) -> bool {
        fn pointee(ty: &mir::Ty) -> Option<&mir::Ty> {
            match ty {
                mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => Some(inner),
                _ => None,
            }
        }
        let (Some(sp), Some(tp)) = (pointee(source), pointee(target)) else {
            return false;
        };
        matches!(self.peel_lands_on(sp, tp), Some(true))
            || matches!(self.peel_lands_on(tp, sp), Some(true))
    }

    /// The field-0 retarget a pointer of type `from` undergoes
    /// when it is reinterpreted as a pointer of type `to` across
    /// a prefix struct's first field (see
    /// [`Self::cast_crosses_prefix_field`]): `Some((true, depth))`
    /// for an entry (the `to` pointee is the `from` pointee's
    /// first field, reached by `depth` field-0 entries) and
    /// `Some((false, depth))` for the reverse exit. `None` when
    /// the two pointer types stand in no prefix relation (or in
    /// a wrapper-only one, whose denotations the access points
    /// already reconcile), so the pointer carries over with its
    /// denotation unchanged. The array-element continuation of
    /// an entry peel has no whole-pointee counterpart and does
    /// not count as a retarget.
    pub(super) fn prefix_retarget(
        &self,
        from: &mir::Ty,
        to: &mir::Ty,
    ) -> Option<(bool, u32)> {
        if !self.cast_crosses_prefix_field(from, to) {
            return None;
        }
        match (
            self.cast_entry_depth(from, to),
            self.cast_exit_depth(from, to),
        ) {
            (Some((depth, false)), _) => Some((true, depth)),
            (_, Some(depth)) => Some((false, depth)),
            _ => None,
        }
    }

    /// Peel field-0 layers (transparent wrappers, single-field
    /// array structs, and prefix structs; see
    /// [`Self::peel_step`]) off `current` until it matches
    /// `target`, returning the number of field-0 entries the peel
    /// denotes -- and, when the innermost peeled layer is an
    /// array (or slice) whose element type is exactly `target`,
    /// reporting that the peel continues into that array's first
    /// element. `None` when the peel lands nowhere the target
    /// names.
    fn wrapper_peel(
        &self,
        mut current: mir::Ty,
        target: &mir::Ty,
    ) -> Option<(u32, bool)> {
        let mut depth = 0u32;
        loop {
            // The layers must be peeled to know whether the cast
            // lands exactly on the target pointee.
            let (next, _) = self.peel_step(&current)?;
            depth += 1;
            if Self::cast_ty_matches(&next, target) {
                return Some((depth, false));
            }
            // The peel crossed the wrappers and reached an array
            // whose element type is the target: the cast denotes
            // the array's first element, one index-0 step past
            // the field entries the peel denotes.
            if let mir::Ty::Array(elem, _) | mir::Ty::Slice(elem) = &next {
                if Self::cast_ty_matches(elem, target) {
                    return Some((depth, true));
                }
            }
            current = next;
        }
    }

    /// Whether a pointer cast from `source` to `target` goes from
    /// an aggregate pointee (array / slice / `str`) to the element
    /// type itself. Decided from the cast's two types, both known
    /// at conversion time: the source pointee is the aggregate and
    /// the target pointee is exactly its element type (`u8` for
    /// `str`). Such a cast retargets the pointer at the first
    /// element: an array or slice value rides as one aggregate in
    /// this model, so a pointer that denotes the aggregate as a
    /// whole and a pointer that denotes its first element differ
    /// by an index-0 projection, and only the cast's types say
    /// which of the two a given pointer value is.
    pub(super) fn cast_enters_first_element(&self, source: &mir::Ty, target: &mir::Ty) -> bool {
        fn pointee(ty: &mir::Ty) -> Option<&mir::Ty> {
            match ty {
                mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => Some(inner),
                _ => None,
            }
        }
        let (Some(src), Some(tgt)) = (pointee(source), pointee(target)) else {
            return false;
        };
        match src {
            mir::Ty::Array(elem, _) | mir::Ty::Slice(elem) => **elem == *tgt,
            mir::Ty::Str => matches!(tgt, mir::Ty::Uint(mir::UintTy::U8)),
            _ => false,
        }
    }

    /// Whether a pointer cast from `source` to `target` is the
    /// reverse of [`Converter::cast_enters_first_element`]: the
    /// source pointee is an element type and the target pointee is
    /// an array of exactly that element type.
    pub(super) fn cast_returns_to_array_base(&self, source: &mir::Ty, target: &mir::Ty) -> bool {
        fn pointee(ty: &mir::Ty) -> Option<&mir::Ty> {
            match ty {
                mir::Ty::Ref(_, inner, _) | mir::Ty::RawPtr(inner, _) => Some(inner),
                _ => None,
            }
        }
        let (Some(src), Some(tgt)) = (pointee(source), pointee(target)) else {
            return false;
        };
        match tgt {
            mir::Ty::Array(elem, _) => **elem == *src,
            _ => false,
        }
    }

    pub(super) fn convert_ty(&self, ty: &mir::Ty, context: &str) -> Result<ir::Ty, ConvertError> {
        let unsupported = || ConvertError::UnsupportedType {
            context: context.to_string(),
            ty: format!("{} ({ty:?})", ty.kind_name()),
        };
        Ok(match ty {
            mir::Ty::Bool => ir::Ty::Bool,
            mir::Ty::Int(it) => ir::Ty::Int(convert_int_ty(*it)),
            mir::Ty::Uint(ut) => ir::Ty::Uint(convert_uint_ty(*ut)),
            mir::Ty::Float(mir::FloatTy::F16) => ir::Ty::Float(ir::FloatTy::F16),
            mir::Ty::Float(mir::FloatTy::F32) => ir::Ty::Float(ir::FloatTy::F32),
            mir::Ty::Float(mir::FloatTy::F64) => ir::Ty::Float(ir::FloatTy::F64),
            mir::Ty::Float(mir::FloatTy::F128) => ir::Ty::Float(ir::FloatTy::F128),
            // `char` is a 32-bit scalar; the IR models it as `u32`.
            mir::Ty::Char => ir::Ty::Uint(ir::UintTy::U32),
            mir::Ty::Never => ir::Ty::Never,
            mir::Ty::Tuple(ts) if ts.is_empty() => ir::Ty::Unit,
            mir::Ty::Tuple(ts) => {
                let mut out = Vec::with_capacity(ts.len());
                for t in ts {
                    out.push(self.convert_ty(t, context)?);
                }
                ir::Ty::Tuple(out)
            }
            mir::Ty::Adt(def, _) => ir::Ty::Adt(self.def_display_name(def)),
            // Function pointers carry no signature in the IR; the
            // callee's arity is checked at the call site.
            mir::Ty::FnPtr(_) => ir::Ty::FnPtr,
            // Function items and closures are named by the body they
            // call, so a value of this type can be invoked.
            mir::Ty::FnDef(def, _) => ir::Ty::FnDef(self.def_display_name(def)),
            mir::Ty::Closure(def, args) => {
                // The closure type's generic arguments are
                // `[parent args.., kind, signature, upvars tuple]`:
                // the upvars tuple is the LAST argument, not the
                // third — the third is the upvars tuple only for a
                // closure with no parent generics, and reading it
                // there misreads every closure defined inside a
                // generic function or trait method.
                let upvars = match args.last() {
                    Some(mir::Ty::Tuple(ts)) => ts.len(),
                    _ => 0,
                };
                ir::Ty::Closure(self.body_key(def), upvars)
            }
            mir::Ty::Ref(_, inner, _) => {
                ir::Ty::Ref(Box::new(self.convert_ty(inner, context)?))
            }
            // Raw pointers share the reference representation.
            mir::Ty::RawPtr(inner, _) => {
                ir::Ty::Ref(Box::new(self.convert_ty(inner, context)?))
            }
            mir::Ty::Str => ir::Ty::Str,
            mir::Ty::Array(elem, len) => {
                let n = match len {
                    mir::ArrayLen::Known(n) => *n as u64,
                    _ => return Err(unsupported()),
                };
                ir::Ty::Array(Box::new(self.convert_ty(elem, context)?), n)
            }
            mir::Ty::Slice(elem) => ir::Ty::Slice(Box::new(self.convert_ty(elem, context)?)),
            // A pattern type shares its base type's
            // representation: a value of the pattern type is a
            // base value, and no execution step needs the range
            // itself. Construction in a valid program guarantees
            // the range, and layout facts that depend on it
            // (niches) reach the interpreter through the driver's
            // rustc-derived tables, not through this type.
            mir::Ty::Pat(inner, _) => self.convert_ty(inner, context)?,
            // `dyn Trait` is representable: behind a reference it is
            // a fat pointer whose value carries the concrete type;
            // as a bare place type it names the unsized pointee a
            // fat pointer's dereference reaches. Only the bounds'
            // names are kept.
            mir::Ty::Dynamic(traits, _, _) => ir::Ty::Dynamic(traits.clone()),
            // A coroutine type: the value is the state object the
            // coroutine transform produces (see `ir::Ty::Coroutine`).
            // Only the body's name is kept; the state layout is
            // carried by the value, and the generic arguments (upvars,
            // resume / yield / return types, witness) are not needed
            // to execute the transformed body.
            mir::Ty::Coroutine(def, _) => ir::Ty::Coroutine(self.def_display_name(def)),
            _ => return Err(unsupported()),
        })
    }

}
