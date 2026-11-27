//! Drop-glue planning: drop plans for owned iterators and the drop-glue builder.

use std::collections::{HashMap};

use sparagmos_mir as mir;

use crate::ir;
use crate::value::{RefProj, Value};

use super::adt_fields::*;
use super::converter::*;


/// The drop plan of a modelled owning iterator state
/// (`Enum { fields: [Tuple(elements), Uint(cursor)] }`, the shape
/// the `vec` / `array` / `option` `IntoIter` builtins build):
/// the elements tuple's glue, element by element. Slots the
/// iterator already yielded hold `Uninit`, which every plan
/// skips, so exactly the unyielded elements are dropped — the
/// real iterator's `Drop` impl semantics.
fn owned_iter_plan(elem: ir::DropPlan) -> ir::DropPlan {
    match elem {
        ir::DropPlan::Trivial => ir::DropPlan::Trivial,
        e => ir::DropPlan::At {
            path: vec![RefProj::Field(0)],
            plan: Box::new(ir::DropPlan::Each { elem: Box::new(e) }),
        },
    }
}

/// Run `plan` against field 0 of the value it applies to (the
/// pointee of a `Box`, the payload of `Option::Some`).
fn at_field0(plan: ir::DropPlan) -> ir::DropPlan {
    match plan {
        ir::DropPlan::Trivial => ir::DropPlan::Trivial,
        other => ir::DropPlan::At {
            path: vec![RefProj::Field(0)],
            plan: Box::new(other),
        },
    }
}

/// A `Fields` plan, collapsed to `Trivial` when no field has glue.
fn collapse_fields(fields: Vec<ir::DropPlan>, complete: bool) -> ir::DropPlan {
    if fields.iter().all(|f| *f == ir::DropPlan::Trivial) {
        ir::DropPlan::Trivial
    } else {
        ir::DropPlan::Fields { fields, complete }
    }
}

/// A synthesized value as an IR constant, for the shapes a
/// unique-value type can take (units, tuples and ADT values);
/// anything else has no constant form here.
pub(super) fn value_as_constant(value: &Value) -> Option<ir::Constant> {
    Some(match value {
        Value::Unit => ir::Constant::Unit,
        Value::Enum { variant, fields } => ir::Constant::Adt {
            variant: *variant,
            fields: fields
                .iter()
                .map(value_as_constant)
                .collect::<Option<Vec<_>>>()?,
        },
        Value::Tuple(fields) => ir::Constant::Tuple(
            fields
                .iter()
                .map(value_as_constant)
                .collect::<Option<Vec<_>>>()?,
        ),
        _ => return None,
    })
}

/// A sequence of glue steps, collapsed when at most one remains.
fn collapse_seq(steps: Vec<ir::DropPlan>) -> ir::DropPlan {
    let mut steps: Vec<ir::DropPlan> = steps
        .into_iter()
        .filter(|p| *p != ir::DropPlan::Trivial)
        .collect();
    match steps.len() {
        0 => ir::DropPlan::Trivial,
        1 => steps.pop().expect("one step"),
        _ => ir::DropPlan::Seq(steps),
    }
}

/// Builds the drop glue of one dropped type. Glue is built only
/// when every part of the type is accounted for:
///
/// * scalars, references and function pointers have no glue;
/// * tuples, arrays and slices glue their elements;
/// * the modelled std types glue by their fixed shape (`Vec` its
///   elements, `Box` its pointee, `Option` its payload, `Range` its
///   bounds; `String`, `fmt` values and `ManuallyDrop` nothing);
/// * any other ADT glues its converted `Drop` impl body (called
///   first, as in rustc's drop glue) and then its fields, variant
///   by variant. An impl the name table records but the program
///   has no body for (a generic impl gets no monomorphic instance)
///   fails the build, as does an ADT from another crate whose impl
///   the table cannot rule out. A field whose type no projection
///   anywhere records (only ever written) gets an `Untyped` plan,
///   judged by its value's shape at run time.
pub(super) struct DropGlueBuilder<'c, 'a> {
    conv: &'c Converter<'a>,
    adts: Vec<ir::DropPlan>,
    adt_index: HashMap<String, usize>,
    /// Table entries whose plan is still being built: they hold a
    /// `Trivial` placeholder that a recursive reference must keep
    /// pointing at (the finished plan replaces it), so the
    /// placeholder must never be read as a finished trivial plan.
    in_progress: std::collections::HashSet<usize>,
}

impl<'c, 'a> DropGlueBuilder<'c, 'a> {
    pub(super) fn build(conv: &'c Converter<'a>, ty: &mir::Ty) -> Option<ir::DropGlue> {
        let mut builder = DropGlueBuilder {
            conv,
            adts: Vec::new(),
            adt_index: HashMap::new(),
            in_progress: Default::default(),
        };
        let root = builder.plan(ty)?;
        Some(ir::DropGlue {
            root,
            adts: builder.adts,
        })
    }

    fn plan(&mut self, ty: &mir::Ty) -> Option<ir::DropPlan> {
        use ir::DropPlan as P;
        match ty {
            mir::Ty::Bool
            | mir::Ty::Char
            | mir::Ty::Int(_)
            | mir::Ty::Uint(_)
            | mir::Ty::Float(_)
            | mir::Ty::Never
            | mir::Ty::Str => Some(P::Trivial),
            mir::Ty::Ref(..) | mir::Ty::RawPtr(..) | mir::Ty::FnDef(..) | mir::Ty::FnPtr(_) => {
                Some(P::Trivial)
            }
            mir::Ty::Tuple(ts) => {
                let mut fields = Vec::with_capacity(ts.len());
                for t in ts {
                    fields.push(self.plan(t)?);
                }
                Some(collapse_fields(fields, true))
            }
            mir::Ty::Array(elem, _) | mir::Ty::Slice(elem) => {
                let elem_plan = self.plan(elem)?;
                let elem_plan = self.elided_elem_plan(elem, elem_plan);
                match elem_plan {
                    P::Trivial => Some(P::Trivial),
                    elem_plan => Some(P::Each {
                        elem: Box::new(elem_plan),
                    }),
                }
            }
            mir::Ty::Closure(_, args) => {
                // A closure's glue is its captures' glue: the last
                // generic argument is the capture tuple. Anything
                // else in that position is a shape this builder does
                // not model, and the drop stays refused.
                let mir::Ty::Tuple(captures) = args.last()? else {
                    return None;
                };
                let mut fields = Vec::with_capacity(captures.len());
                for t in captures {
                    fields.push(self.plan(t)?);
                }
                Some(collapse_fields(fields, true))
            }
            mir::Ty::Adt(def, args) => {
                let key = format!("{ty:?}");
                if let Some(&index) = self.adt_index.get(&key) {
                    // A finished entry whose plan is trivial is the
                    // trivial plan itself: returning it lets the
                    // caller's field plans collapse, so a value
                    // holding fewer fields than the plan lists (a
                    // zero-sized field the value model elides) is
                    // not mistaken for an unaccounted drop.
                    if !self.in_progress.contains(&index)
                        && self.adts[index] == P::Trivial
                    {
                        return Some(P::Trivial);
                    }
                    return Some(P::Adt { index });
                }
                let index = self.adts.len();
                self.adt_index.insert(key, index);
                // Placeholder while this ADT's fields are planned, so
                // a recursive occurrence refers back to this entry.
                self.adts.push(P::Trivial);
                self.in_progress.insert(index);
                let plan = self.adt_plan(*def, args);
                self.in_progress.remove(&index);
                let plan = plan?;
                let trivial = plan == P::Trivial;
                self.adts[index] = plan;
                Some(if trivial {
                    P::Trivial
                } else {
                    P::Adt { index }
                })
            }
            mir::Ty::Dynamic(..) => {
                // A trait object's glue is its concrete type's,
                // resolved at run time from the value (see
                // `DropPlan::Dyn` and the registration at unsizing
                // casts). A value carrying no accounted glue is
                // judged by its shape at run time, the judgment
                // `DropPlan::Dynamic` names.
                Some(P::Dyn)
            }
            mir::Ty::Coroutine(def, args) => {
                let key = format!("{ty:?}");
                if let Some(&index) = self.adt_index.get(&key) {
                    if !self.in_progress.contains(&index)
                        && self.adts[index] == P::Trivial
                    {
                        return Some(P::Trivial);
                    }
                    return Some(P::Adt { index });
                }
                let index = self.adts.len();
                self.adt_index.insert(key, index);
                // Placeholder while this coroutine's fields are
                // planned, so a recursive occurrence (a future
                // awaiting another future of the same type) refers
                // back to this entry.
                self.adts.push(P::Trivial);
                self.in_progress.insert(index);
                let plan = self.coroutine_plan(*def, args);
                self.in_progress.remove(&index);
                let plan = plan?;
                let trivial = plan == P::Trivial;
                self.adts[index] = plan;
                Some(if trivial {
                    P::Trivial
                } else {
                    P::Adt { index }
                })
            }
            _ => None,
        }
    }

    /// The drop plan of one sequence element (of an array, a
    /// slice or a `Vec`), wrapped so the glue of an element whose
    /// store MIR elided still runs: when the element type has a
    /// single value, the plan becomes [`ir::DropPlan::Elided`]
    /// carrying that value. An element type with many values
    /// keeps its plain plan — an uninitialised slot of such a
    /// type in a live sequence is not a state this builder
    /// accounts for, and the read rules keep judging it.
    fn elided_elem_plan(&mut self, elem_ty: &mir::Ty, plan: ir::DropPlan) -> ir::DropPlan {
        use ir::DropPlan as P;
        match plan {
            P::Trivial => P::Trivial,
            plan => match self.conv.zst_value(elem_ty, 0) {
                Some(value) => P::Elided {
                    value,
                    plan: Box::new(plan),
                },
                None => plan,
            },
        }
    }

    fn adt_plan(&mut self, def: mir::DefId, args: &[mir::Ty]) -> Option<ir::DropPlan> {
        use ir::DropPlan as P;
        let conv = self.conv;
        let name = conv
            .names
            .get(&def)
            .or_else(|| conv.def_names.get(&def))?
            .clone();
        // A union's fields are never dropped: the language
        // requires them to be `Copy` or wrapped in
        // `ManuallyDrop`, so no field walk ever applies -- the
        // fields alias one storage, and the stored value holds
        // the one written field, which a struct-style walk
        // mismatches. The union type itself may still carry a
        // `Drop` impl, and then that destructor is the whole
        // glue: it runs on an explicit `mem::drop` and on
        // overwrite, exactly as for any other type.
        if conv
            .adt_shapes
            .get(&def)
            .is_some_and(|shape| shape.kind == mir::AdtShapeKind::Union)
        {
            let ty_key = format!("{:?}", mir::Ty::Adt(def, args.to_vec()));
            let dtor = conv
                .dtor_bodies
                .get(&ty_key)
                .cloned()
                .or_else(|| conv.dtor_body(&name, args));
            if let Some(func) = dtor {
                return Some(P::CallDtor { func });
            }
            if conv.drop_trivial_adts.contains(&ty_key)
                || conv.drop_no_impl_adts.contains(&ty_key)
            {
                // The compiler certifies this instantiation's
                // glue empty, or the driver certifies the type
                // has no `Drop` impl in its own crate; with no
                // field glue either, the plan is trivial.
                return Some(P::Trivial);
            }
            if conv.dtor_impl_known(&name) {
                // The impl exists but has no converted body for
                // this instantiation: its effects can neither be
                // executed nor ruled out.
                return None;
            }
            if conv.is_local_type(&name) {
                // The entry crate's name table is complete, so
                // no recorded impl means there is none.
                return Some(P::Trivial);
            }
            // A foreign union no certification covers: fall
            // through to the modelled-type table and the
            // generic path below, which refuse what they
            // cannot rule out.
        }
        // Modelled std types: their glue shape is fixed by the value
        // model, not by a converted body.
        match name.as_str() {
            "std::string::String" => return Some(P::Trivial),
            "std::fmt::Arguments" | "core::fmt::rt::Argument" => return Some(P::Trivial),
            // `ManuallyDrop` is defined to run no glue at all.
            "std::mem::ManuallyDrop" | "core::mem::ManuallyDrop" => return Some(P::Trivial),
            // `MaybeUninit<T>` runs no glue either: its contents
            // are manually managed, so dropping the wrapper (e.g.
            // the `Box<MaybeUninit<[T; N]>>` a `vec!` construction
            // is assembled in before `assume_init`) drops nothing.
            "std::mem::MaybeUninit" | "core::mem::MaybeUninit" => return Some(P::Trivial),
            // The unit allocator marker: a zero-sized value with
            // no fields and no `Drop` impl.
            "std::alloc::Global" => return Some(P::Trivial),
            // `NonZero<T>` wraps a primitive integer in a niche
            // wrapper (`T::NonZeroInner`, an associated type the
            // evidence tables cannot resolve, which is why the
            // generic path below plans its one field as
            // `Untyped` and the plan never collapses): it has no
            // `Drop` impl and its only content is the integer
            // itself, so its glue is empty whatever a value
            // holding one (a slice window / chunk iterator, say)
            // looks like in the model.
            "std::num::NonZero" | "core::num::NonZero" => return Some(P::Trivial),
            // The raw allocation buffer behind the modelled
            // containers (`Vec`, `String`, `VecDeque`): rustc's
            // generated drop glue for a container drops the
            // buffer field as a place of this type (and, inside
            // it, of `RawVecInner`). The buffer's real glue only
            // deallocates -- dropping the elements is the owner's
            // glue (the `Vec` plan, the slice glue), never the
            // buffer's -- and the value model does not represent
            // the allocation at all (a container value holds its
            // elements directly; capacity is not represented),
            // so there is no glue to run for the buffer field.
            "alloc::raw_vec::RawVec" | "alloc::raw_vec::RawVecInner" => {
                return Some(P::Trivial)
            }
            // A fieldless error struct (the payload of a failed
            // channel receive): nothing to drop.
            "std::sync::mpmc::RecvError" => return Some(P::Trivial),
            // A type identity: its payload is a hash value with
            // no glue of its own.
            "std::any::TypeId" => return Some(P::Trivial),
            "std::sync::mpmc::SendError" => {
                return Some(self.variant_fields_plan(def, args)?);
            }
            // A flatten composition's value keeps the real
            // layout (the modelled step runs on it in place), so
            // the glue is the fields': the wrapped compat state
            // for `FlatMap` / `Flatten`, and for `FlattenCompat`
            // itself the fused source iterator and the two
            // in-flight inner iterators.
            "std::iter::FlatMap" | "std::iter::adapters::flatten::FlattenCompat"
            | "std::iter::Fuse" => {
                return Some(self.variant_fields_plan(def, args)?);
            }
            "std::panic::AssertUnwindSafe" | "core::panic::AssertUnwindSafe" => {
                // A one-field transparent wrapper with no `Drop`
                // impl of its own: dropping it drops the held
                // value. `catch_unwind` arguments are dropped on
                // every path through the panic machinery, so this
                // glue is on the coroutine panic path as well.
                let inner = self.plan(args.first()?)?;
                return Some(at_field0(inner));
            }
            "std::cmp::Reverse"
            | "core::cmp::Reverse"
            | "std::num::Wrapping"
            | "core::num::Wrapping"
            | "std::num::Saturating"
            | "core::num::Saturating" => {
                // One-field transparent wrappers with no `Drop`
                // impl of their own: dropping the wrapper drops
                // the held value, exactly as dropping the held
                // value itself would (`at_field0` collapses a
                // trivial inner plan). These types are named here
                // because the generic path below refuses a foreign
                // ADT whose impl the name table cannot rule out,
                // and a `Termination` report payload is dropped by
                // glue the interpreter synthesizes for it -- no
                // body drops the payload, so the driver scan that
                // certifies dropped types never certifies them.
                let inner = self.plan(args.first()?)?;
                return Some(at_field0(inner));
            }
            "std::pin::Pin" | "core::pin::Pin" | "std::pin::PinMacroHelper" => {
                // `Pin<P>` has no destructor of its own (it is a
                // transparent newtype with no `Drop` impl); dropping
                // it drops the pointer it wraps, exactly as dropping
                // the pointer itself would: `Pin<Box<T>>` runs the
                // box's pointee glue, `Pin<&mut T>` drops nothing
                // (`at_field0` collapses a trivial inner plan).
                let inner = self.plan(args.first()?)?;
                return Some(at_field0(inner));
            }
            "std::vec::Vec" => {
                let elem_ty = args.first()?;
                let elem = self.plan(elem_ty)?;
                let elem = self.elided_elem_plan(elem_ty, elem);
                return Some(match elem {
                    P::Trivial => P::Trivial,
                    e => P::Each { elem: Box::new(e) },
                });
            }
            "std::boxed::Box" => {
                // Dropping a box drops its pointee. Which storage
                // the pointee lives in is a property of the box
                // value's own shape — the `Box::new` builtin's
                // one-field aggregate and the real `Unique` /
                // `NonNull` layout a converted constructor body
                // (`Box::from_raw`, `Box::new_in`, ...) builds
                // both occur — so the plan defers to the run-time
                // judgment in `DropPlan::BoxPointee`. A `dyn`
                // pointee is the exception: the pointer value *is*
                // the fat pointer then, with no field to project,
                // so the plan is the dyn plan itself. A pointee
                // that merely mentions a trait object (an ADT or
                // tuple whose unsized tail is a `dyn` field) may
                // likewise hold the trait-object form — such a box
                // is formed only by the unsizing coercion of a
                // concrete box — but the same type also spells
                // plain boxes (a `Vec<Box<dyn Trait>>` element type
                // inside the pointee, say), so the form is judged
                // on the value at run time.
                let pointee = args.first()?;
                let inner = self.plan(pointee)?;
                return Some(match inner {
                    P::Dyn => P::Dyn,
                    other => {
                        // A pointee with no glue leaves the box
                        // nothing to drop, exactly as the field-0
                        // addressing this replaces collapsed to
                        // `Trivial` (see `at_field0`).
                        let boxed = match other {
                            P::Trivial => P::Trivial,
                            o => P::BoxPointee {
                                pointee: Box::new(o),
                            },
                        };
                        if ty_mentions_dyn(pointee) {
                            P::DynOrPlain {
                                plain: Box::new(boxed),
                            }
                        } else {
                            boxed
                        }
                    }
                });
            }
            "std::rc::Rc" | "std::sync::Arc" => {
                // The handle names a reference-counted allocation
                // in the interpreter's store (see the `Rc::new` /
                // `Arc::new` builtins): dropping it decrements the
                // strong count, and the pointee's glue runs only
                // when the last strong handle goes away. A `dyn`
                // pointee keeps the dyn plan: the handle value *is*
                // the fat pointer then, and its baked glue applies
                // to the pointee in the allocation.
                let pointee = args.first()?;
                let inner = self.plan(pointee)?;
                return Some(match inner {
                    P::Dyn => P::Dyn,
                    other if ty_mentions_dyn(pointee) => P::DynOrPlain {
                        plain: Box::new(P::RcHandleDrop {
                            value: Box::new(other),
                        }),
                    },
                    other => P::RcHandleDrop {
                        value: Box::new(other),
                    },
                });
            }
            "std::rc::Weak" | "std::sync::Weak" => {
                // A weak pointer owns no pointee; dropping it
                // releases its weak reference on the allocation,
                // which is observable through the counts and
                // through `upgrade` once the storage is freed.
                return Some(P::RcWeakRelease);
            }
            "std::borrow::Cow" if matches!(args.first(), Some(mir::Ty::Str)) => {
                // `Cow<str>`: the `Borrowed` variant holds a `&str`
                // and the `Owned` variant a `String`; both are
                // glue-free in this model. (Other borrowed types
                // own a `B::Owned` whose glue this builder cannot
                // derive from `B`, so they stay unmodelled.)
                return Some(P::Trivial);
            }
            "std::option::Option" => {
                let inner = self.plan(args.first()?)?;
                return Some(match inner {
                    P::Trivial => P::Trivial,
                    e => P::Variants {
                        variants: vec![P::Trivial, at_field0(e)],
                    },
                });
            }
            "std::cell::Ref" | "core::cell::Ref" => {
                return Some(P::RefCellRelease { mutable: false })
            }
            "std::cell::RefMut" | "core::cell::RefMut" => {
                return Some(P::RefCellRelease { mutable: true })
            }
            "std::cell::Cell" | "core::cell::Cell" => {
                // A transparent wrapper with no `Drop` impl of
                // its own: the modelled value is the two-layer
                // aggregate `Cell { UnsafeCell { value } }` (see
                // `cell_value` in `builtins`), so the held value's
                // glue runs through both field-0 layers.
                let inner = self.plan(args.first()?)?;
                return Some(at_field0(at_field0(inner)));
            }
            "std::cell::UnsafeCell" | "core::cell::UnsafeCell" => {
                // The one-layer transparent wrapper the `Cell`
                // plan above peels: dropping it drops the held
                // value, as for `Cell` itself.
                let inner = self.plan(args.first()?)?;
                return Some(at_field0(inner));
            }
            "std::cell::RefCell" | "core::cell::RefCell" => {
                // `RefCell` has no `Drop` impl of its own: the
                // flag `Cell` is plain data, and the value's glue
                // runs through the `UnsafeCell` field wrapper.
                let inner = self.plan(args.first()?)?;
                return Some(match inner {
                    P::Trivial => P::Trivial,
                    plan => P::At {
                        path: vec![RefProj::Field(1), RefProj::Field(0)],
                        plan: Box::new(plan),
                    },
                });
            }
            "std::ops::CoroutineState" | "core::ops::CoroutineState" => {
                // The resume result: an enum (`Yielded` / `Complete`)
                // with no `Drop` impl of its own, so its glue is its
                // variants' fields' glue. Every `resume` produces
                // one, and `catch_unwind` wraps it in a `Result`, so
                // this glue is on the coroutine runtime path itself.
                return Some(self.variant_fields_plan(def, args)?);
            }
            "std::task::Poll" | "core::task::Poll" => {
                // The future poll result: an enum (`Pending` /
                // `Ready`) with no `Drop` impl of its own; its glue
                // is its variants' fields' glue, as for
                // `CoroutineState`.
                return Some(self.variant_fields_plan(def, args)?);
            }
            "std::result::Result" => {
                // Same value shape as `Option`: variant 0 is `Ok`,
                // variant 1 is `Err`, the payload sits at field 0.
                // (Needed so a `try_borrow` result holding a guard
                // can be dropped: the guard's release is the Ok
                // payload's glue.)
                let ok = self.plan(args.first()?)?;
                let err = self.plan(args.get(1)?)?;
                return Some(match (ok, err) {
                    (P::Trivial, P::Trivial) => P::Trivial,
                    (ok, err) => P::Variants {
                        variants: vec![at_field0(ok), at_field0(err)],
                    },
                });
            }
            // The `try_borrow` error types: unit structs with no
            // fields and no `Drop` impl.
            "std::cell::BorrowError" | "core::cell::BorrowError" => {
                return Some(P::Trivial)
            }
            // The `std::fs::File` wrapper chain (`File` -> sys
            // `File` -> `FileDesc` -> `OwnedFd`): the value is a
            // file-table id in the descriptor slot, and the
            // destructor's only observable effect — closing the
            // host file — is the interpreter's file-handle drop
            // plan. Every layer of the chain takes the same plan:
            // glue for a value of any layer (a `File` field of a
            // guest struct, a bare `OwnedFd`) closes its file.
            "std::fs::File" | "std::sys::fs::unix::File"
            | "std::sys::fd::unix::FileDesc" | "std::os::fd::OwnedFd" => {
                return Some(P::FileHandleDrop)
            }
            // The `std::net` wrapper chains (`TcpListener` ->
            // sys `TcpListener` -> `Socket` -> `FileDesc` ->
            // `OwnedFd`, and the `TcpStream` / `UdpSocket`
            // analogues): the value is a socket-table id in the
            // descriptor slot, and the destructor's only
            // observable effect — closing the host socket — is
            // the interpreter's socket-handle drop plan. Every
            // layer above `FileDesc` takes the same plan (the
            // two bottom layers are shared with the file model
            // and keep its plan: a socket value's own plan is
            // this leaf at its outermost layer, so the shared
            // layers' plans never run for a socket).
            "std::net::TcpListener" | "std::net::TcpStream"
            | "std::net::UdpSocket"
            | "std::sys::net::connection::socket::TcpListener"
            | "std::sys::net::connection::socket::TcpStream"
            | "std::sys::net::connection::socket::UdpSocket"
            | "std::sys::net::connection::socket::Socket" => {
                return Some(P::SocketHandleDrop)
            }
            // A `std::string::Drain`: the destructor's only
            // observable effect -- deleting its byte range
            // from the string it points at -- is the
            // interpreter's string-drain drop plan (the
            // collected destructor body reaches it through
            // `as_mut_vec` and a temporary `Vec::drain`,
            // whose raw-parts walk the string model cannot
            // serve). The remaining fields are a raw pointer,
            // two bounds and a `Chars` over a snapshot: none
            // carries glue of its own.
            "std::string::Drain" | "alloc::string::Drain" => {
                return Some(P::StringDrainDrop)
            }
            // The child-process pipe handles: the value is the
            // pipe end's file-table id under the same wrapper
            // chain as `File`'s, and the destructor's only
            // observable effect -- closing the end, which is
            // what delivers EOF / EPIPE to the other side -- is
            // the same file-handle drop plan.
            "std::process::ChildStdin" | "std::process::ChildStdout"
            | "std::process::ChildStderr" | "std::sys::pipe::Pipe" => {
                return Some(P::FileHandleDrop)
            }
            // mpsc handles: the value is the channel id, and the
            // destructor's only observable effect — the shared
            // handle count going down, which is what disconnects
            // the channel — is the interpreter's channel-handle
            // drop plan.
            "std::sync::mpsc::Sender" | "std::sync::mpsc::SyncSender" => {
                return Some(P::ChannelHandleDrop { sender: true })
            }
            "std::sync::mpsc::Receiver" => {
                return Some(P::ChannelHandleDrop { sender: false })
            }
            "std::collections::VecDeque" => {
                // A deque's glue is its elements' glue. The value
                // model has no deque shape of its own to run a
                // non-trivial element plan against, so only a
                // trivial element type is accounted for (the
                // `vec::IntoIter` rule).
                let elem = self.plan(args.first()?)?;
                return match elem {
                    P::Trivial => Some(P::Trivial),
                    _ => None,
                };
            }
            // Lock and atomic wrappers with no modelled lock
            // state: their glue is their fields' glue (the data
            // cell's, through the `UnsafeCell` plan), planned
            // from the same evidence the generic path uses for
            // the entry crate's types.
            "std::sync::Mutex" | "std::sync::RwLock" | "std::sync::atomic::Atomic" => {
                return Some(self.variant_fields_plan(def, args)?);
            }
            // The integer parse error types: a struct holding a
            // fieldless kind enum; neither has a `Drop` impl, so
            // the evidenced-fields plan is the whole glue.
            "std::num::ParseIntError" | "core::num::ParseIntError"
            | "std::num::IntErrorKind" | "core::num::IntErrorKind" => {
                return Some(self.variant_fields_plan(def, args)?);
            }
            "std::task::Waker" | "core::task::Waker" | "std::task::LocalWaker"
            | "core::task::LocalWaker" => {
                // Which release a waker's drop runs depends on
                // the shape of the stored value (the modelled
                // `From` conversion's handle aggregate, or a real
                // `RawWaker` from `Waker::from_raw` whose vtable
                // drop entry runs): the `WakerDrop` plan judges
                // the value at run time (see `ir::DropPlan`).
                return Some(P::WakerDrop);
            }
            // Marker types: no fields with glue, no `Drop` impl.
            "std::marker::PhantomPinned" | "std::marker::PhantomData" => {
                return Some(P::Trivial)
            }
            // A captured backtrace: the model's value is the
            // capture status and the rendered trace (see
            // `backtrace_value` in `builtins.rs`); it owns no
            // host resource, so the native destructor (which
            // frees the captured frames) has no observable
            // effect here.
            "std::backtrace::Backtrace" => {
                return Some(P::Trivial)
            }
            // The futex-backed sys synchronization primitives:
            // their state is atomic words only, with no glue.
            "std::sys::sync::mutex::futex::Mutex" | "std::sys::sync::rwlock::futex::RwLock" => {
                return Some(P::Trivial)
            }
            // The lock poisoning state: a flag wrapping an atomic
            // bool, and the guard token a lock guard holds. Neither
            // carries glue.
            "std::sync::poison::Flag" | "std::sync::poison::Guard" => {
                return Some(P::Trivial)
            }
            // The futex thread parker (atomic state words only)
            // and the atomic storage alignment wrappers (byte
            // blocks with no contents of their own): no glue.
            "std::sys::sync::thread_parking::futex::Parker"
            | "std::sync::atomic::private::Align1" | "std::sync::atomic::private::Align2"
            | "std::sync::atomic::private::Align4" | "std::sync::atomic::private::Align8"
            | "std::sync::atomic::private::Align16" | "std::sync::atomic::private::Align32"
            | "std::sync::atomic::private::Align64" | "std::sync::atomic::private::Align128" => {
                return Some(P::Trivial)
            }
            "std::async_iter::FromIter" => {
                // A one-field adapter holding the wrapped
                // iterator: dropping it drops the iterator.
                let inner = self.plan(args.first()?)?;
                return Some(at_field0(inner));
            }
            // Lock guards (including the mapped guards, whose
            // destructor releases the lock through the sys lock
            // reference they hold next to the data pointer): the destructor releases the lock
            // (and poisons it when the guard is dropped while
            // unwinding). The lock state is real state in this
            // model -- the converted lock bodies maintain it in
            // the lock aggregate's futex word -- so the release
            // must actually run: the guard's converted `Drop`
            // body is the glue (a `GuardDtor` plan, which runs
            // the body only on a value shaped like a guard the
            // lock bodies built -- the stdio lock types hold a
            // guard field whose value the io builtins forge),
            // followed by the fields' glue as in the generic
            // path below. When the program has no converted
            // body for the impl, the fields plan stands (the
            // guarded value is borrowed, never dropped through
            // the guard, and neither field carries glue of its
            // own).
            "std::sync::MutexGuard" | "std::sync::RwLockReadGuard"
            | "std::sync::RwLockWriteGuard" | "std::sync::MappedMutexGuard"
            | "std::sync::MappedRwLockReadGuard"
            | "std::sync::MappedRwLockWriteGuard"
            | "std::sync::nonpoison::MutexGuard"
            | "std::sync::nonpoison::RwLockReadGuard"
            | "std::sync::nonpoison::RwLockWriteGuard"
            | "std::sync::nonpoison::MappedMutexGuard"
            | "std::sync::nonpoison::MappedRwLockReadGuard"
            | "std::sync::nonpoison::MappedRwLockWriteGuard" => {
                let fields = self.variant_fields_plan(def, args)?;
                let ty_key = format!("{:?}", mir::Ty::Adt(def, args.to_vec()));
                let dtor = self
                    .conv
                    .dtor_bodies
                    .get(&ty_key)
                    .cloned()
                    .or_else(|| self.conv.dtor_body(&name, args));
                return Some(match dtor {
                    Some(func) => collapse_seq(vec![P::GuardDtor { func }, fields]),
                    None => fields,
                });
            }
            // `Hook` is the panic hook holder: its one field is
            // the boxed hook, whose glue the evidenced-fields
            // plan reaches through the `Box` plan. A thread
            // handle's fields are its id and name.
            "std::panicking::Hook" | "std::thread::Thread"
            | "std::thread::thread::Inner" => {
                return Some(self.variant_fields_plan(def, args)?);
            }
            // `std::io::Error` (the payload of a `Builder::spawn`
            // result) has no `Drop` impl: its glue is its fields'
            // glue, planned from the same evidence the generic
            // path uses for the entry crate's types. An
            // unevidenced field stays unaccounted for and fails
            // the plan, as usual.
            // The bitpacked error representation word: for the
            // Os / Simple / SimpleMessage tags the word owns
            // nothing (an error code, a kind discriminant, or a
            // `&'static` message). The Custom tag's word names the
            // payload's allocation by registered synthetic
            // address, and dropping the error must run the
            // payload's glue exactly once: a runtime-judged plan
            // (see `DropPlan::IoErrorRepr`), as the word's tag is
            // a property of the value, not of the type.
            "core::io::error::repr::Repr" => return Some(P::IoErrorRepr),
            "std::io::Error" => {
                return Some(self.variant_fields_plan(def, args)?)
            }
            // A thread `Builder`'s two fields are `Option<String>`
            // and `Option<usize>`, both trivial-glue; the value is
            // built by the `Builder` builtins in that shape.
            "std::thread::Builder" => {
                return Some(P::Fields {
                    fields: vec![P::Trivial, P::Trivial],
                    complete: true,
                })
            }
            // Join handles: the destructor detaches the thread.
            // In the cooperative thread model a detached thread
            // simply stays pending (it runs at the next scheduling
            // point), so there is no glue to run.
            "std::thread::JoinHandle" | "std::thread::ScopedJoinHandle" => {
                return Some(P::Trivial)
            }
            "std::cell::BorrowMutError" | "core::cell::BorrowMutError" => {
                return Some(P::Trivial)
            }
            // Process commands: the value is the process
            // model's handle (an id into the interpreter's
            // command table); the builder state it denotes
            // carries no guest-visible destruction, so there
            // is no glue to run. (A dropped `Child` likewise
            // has no glue in the model: no child is ever
            // started, so there is nothing to reap.)
            "std::process::Command"
            | "std::sys::process::unix::common::Command"
            | "std::process::Child" => {
                return Some(P::Trivial)
            }
            "std::ops::Range" | "std::ops::RangeInclusive" => {
                let bound = self.plan(args.first()?)?;
                return Some(collapse_fields(vec![bound.clone(), bound], true));
            }
            "std::ops::RangeFrom" | "std::ops::RangeTo" | "std::ops::RangeToInclusive" => {
                // The one-bound ranges: a single field holding the
                // bound, whose glue is the field's.
                let bound = self.plan(args.first()?)?;
                return Some(collapse_fields(vec![bound], true));
            }
            "std::ops::RangeFull" => return Some(P::Trivial),
            "std::array::IntoIter" | "core::array::IntoIter"
            | "std::boxed::BoxedArrayIntoIter" => {
                // The modelled state holds the buffer's elements
                // in a tuple with yielded slots marked `Uninit`,
                // so the elements' glue is exact (see
                // `owned_iter_plan`). (`BoxedArrayIntoIter`'s
                // modelled state is the same [items, next index]
                // pair its `into_iter` builds.)
                let elem = self.plan(args.first()?)?;
                return Some(owned_iter_plan(elem));
            }
            "std::vec::IntoIter" | "alloc::vec::IntoIter" => {
                let elem = self.plan(args.first()?)?;
                return Some(owned_iter_plan(elem));
            }
            "std::option::IntoIter" | "core::option::IntoIter" => {
                let elem = self.plan(args.first()?)?;
                return Some(owned_iter_plan(elem));
            }
            "std::collections::btree_set::IntoIter"
            | "alloc::collections::btree_set::IntoIter" => {
                // The modelled state is the element snapshot plus
                // the next index, like `vec::IntoIter` (see
                // `owned_iter_plan`).
                let elem = self.plan(args.first()?)?;
                return Some(owned_iter_plan(elem));
            }
            "std::collections::btree_map::IntoIter"
            | "alloc::collections::btree_map::IntoIter" => {
                // The snapshot's elements are `(key, value)`
                // tuples; their glue is the two components'.
                let key = self.plan(args.first()?)?;
                let value = self.plan(args.get(1)?)?;
                let elem = collapse_fields(vec![key, value], true);
                return Some(owned_iter_plan(elem));
            }
            // Borrowing iterators own nothing: the references (and,
            // for `IterMut`, the container snapshot) inside the
            // modelled state are not owned values, and the real
            // types have no glue of their own. The slice chunk /
            // window family belongs here too: each holds a borrowed
            // slice plus a plain size, and the modelled states do
            // not always keep the native field count (the `windows`
            // builtin's state is a [slice, size, next index]
            // snapshot of a two-field struct), so a fields plan
            // built from the native shape cannot be run against
            // them -- with no owned content, the trivial plan is
            // the exact one.
            "std::option::Iter" | "core::option::Iter"
            | "std::option::IterMut" | "core::option::IterMut"
            | "std::slice::Iter" | "core::slice::Iter"
            | "std::slice::IterMut" | "core::slice::IterMut"
            | "std::slice::Windows" | "core::slice::Windows"
            | "std::slice::Chunks" | "core::slice::Chunks"
            | "std::slice::ChunksMut" | "core::slice::ChunksMut"
            | "std::slice::ChunksExact" | "core::slice::ChunksExact"
            | "std::slice::ChunksExactMut" | "core::slice::ChunksExactMut"
            | "std::slice::RChunks" | "core::slice::RChunks"
            | "std::slice::RChunksMut" | "core::slice::RChunksMut"
            | "std::slice::RChunksExact" | "core::slice::RChunksExact"
            | "std::slice::RChunksExactMut" | "core::slice::RChunksExactMut"
            | "std::str::Chars" | "core::str::Chars"
            | "std::str::Bytes" | "core::str::Bytes"
            | "std::collections::btree_set::Iter"
            | "alloc::collections::btree_set::Iter"
            | "std::collections::btree_map::Iter"
            | "alloc::collections::btree_map::Iter"
            | "std::collections::btree_map::Keys"
            | "alloc::collections::btree_map::Keys"
            | "std::collections::btree_map::Values"
            | "alloc::collections::btree_map::Values"
            | "std::iter::Empty" | "core::iter::Empty" => return Some(P::Trivial),
            "std::path::Components" => {
                // The modelled state is the path's byte string
                // plus the two consumed counts (see
                // `Builtin::PathComponents` in `builtins`):
                // plain data with no glue of its own. The real
                // type owns nothing either (a borrowed path
                // slice plus parse state), so the plans agree.
                return Some(P::Trivial)
            }
            "std::path::Component" | "std::path::PrefixComponent"
            | "std::path::Prefix" => {
                // The component types hold only string slices
                // (`&OsStr` / `&str`) and plain discriminants,
                // and none of them has a `Drop` impl, so their
                // glue is empty. Naming them here matters
                // because library bodies are not drop-elaborated
                // the way entry-crate bodies are: a generic
                // library body (the closure inside
                // `Iterator::count`, say) still carries `Drop`
                // terminators for the items it consumes, and
                // the generic fallthrough below refuses a
                // non-local type it has no destructor record
                // for.
                return Some(P::Trivial)
            }
            "std::iter::Map" | "core::iter::Map" => {
                // State fields: the inner iterator's state and the
                // closure value; glue is the two fields' glue.
                let inner = self.plan(args.first()?)?;
                let func = self.plan(args.get(1)?)?;
                return Some(collapse_fields(vec![inner, func], true));
            }
            "std::iter::Enumerate" | "core::iter::Enumerate" => {
                let inner = self.plan(args.first()?)?;
                return Some(collapse_fields(vec![inner, P::Trivial], true));
            }
            "std::iter::Copied" | "core::iter::Copied" => {
                let inner = self.plan(args.first()?)?;
                return Some(at_field0(inner));
            }
            "std::iter::Take" | "core::iter::Take" => {
                let inner = self.plan(args.first()?)?;
                return Some(collapse_fields(vec![inner, P::Trivial], true));
            }
            "std::iter::Skip" | "core::iter::Skip" => {
                // Fields are the inner iterator and the
                // remaining skip count (a `usize`: no glue).
                let inner = self.plan(args.first()?)?;
                return Some(collapse_fields(vec![inner, P::Trivial], true));
            }
            "std::iter::Chain" | "core::iter::Chain" => {
                // Fields are the two iterators, each held
                // behind an `Option`: dropping the chain drops
                // whichever iterators are still present, each
                // through the `Option` plan for its type.
                let first = self.plan(args.first()?)?;
                let second = self.plan(args.get(1)?)?;
                let option_plan = |inner: P| match inner {
                    P::Trivial => P::Trivial,
                    e => P::Variants {
                        variants: vec![P::Trivial, at_field0(e)],
                    },
                };
                return Some(collapse_fields(
                    vec![option_plan(first), option_plan(second)],
                    true,
                ));
            }
            "std::iter::Once" | "core::iter::Once" => {
                // The state's only field is the wrapped
                // `option::IntoIter` state.
                let elem = self.plan(args.first()?)?;
                return Some(at_field0(owned_iter_plan(elem)));
            }
            "std::iter::SkipWhile" | "core::iter::SkipWhile" => {
                // Fields are the inner iterator, the `done` flag
                // (a bool: no glue) and the predicate.
                let inner = self.plan(args.first()?)?;
                let func = self.plan(args.get(1)?)?;
                return Some(collapse_fields(vec![inner, P::Trivial, func], true));
            }
            "std::iter::Repeat" | "core::iter::Repeat" => {
                // The state's only field is the element the
                // iterator clones from; dropping the iterator
                // drops it.
                let elem = self.plan(args.first()?)?;
                return Some(at_field0(elem));
            }
            "std::vec::IntoIter" | "alloc::vec::IntoIter" => {
                // Same modelled state as `array::IntoIter` (the
                // element snapshot plus the next index), so the
                // same accounting: only a trivial element type is
                // accounted for.
                let elem = self.plan(args.first()?)?;
                return match elem {
                    P::Trivial => Some(P::Trivial),
                    _ => None,
                };
            }
            // The remaining field-only adapters: the iterator
            // adapters not listed above and the `str` escape
            // adapters. None of them has a `Drop` impl of its
            // own, so the glue is the fields' -- the inner
            // iterator, the closure / state / peeked slot the
            // adapter holds -- planned from the same evidence
            // the generic path uses for the entry crate's types.
            "std::iter::Rev" | "core::iter::Rev" | "std::iter::Cloned"
            | "core::iter::Cloned" | "std::iter::Cycle" | "core::iter::Cycle"
            | "std::iter::Zip" | "core::iter::Zip" | "std::iter::Filter"
            | "core::iter::Filter" | "std::iter::FilterMap"
            | "core::iter::FilterMap" | "std::iter::Peekable"
            | "core::iter::Peekable" | "std::iter::TakeWhile"
            | "core::iter::TakeWhile" | "std::iter::Scan" | "core::iter::Scan"
            | "std::iter::Inspect" | "core::iter::Inspect"
            | "std::iter::StepBy" | "core::iter::StepBy"
            | "std::str::EscapeDebug" | "core::str::EscapeDebug"
            | "std::str::EscapeDefault" | "core::str::EscapeDefault"
            | "std::str::EscapeUnicode" | "core::str::EscapeUnicode"
            | "std::char::EscapeDebug" | "core::str::CharEscapeDebug"
            | "std::char::EscapeDefault" | "core::str::CharEscapeDefault"
            | "std::char::EscapeUnicode" | "core::str::CharEscapeUnicode" => {
                return Some(self.variant_fields_plan(def, args)?);
            }
            "std::ffi::OsString" | "std::ffi::OsStr" => {
                // An `OsString` is the string value itself in this
                // model (see the `env::args_os` builtin): no owned
                // resource beyond it.
                return Some(P::Trivial);
            }
            "std::path::PathBuf" => {
                // A path buffer's only owned content is its
                // `OsString`, which is the string value itself in
                // this model (see the `OsString` entry above), so
                // the evidenced-fields plan is the whole glue.
                return Some(self.variant_fields_plan(def, args)?);
            }
            "std::io::Stdout" | "std::io::Stderr" | "std::io::StdoutLock"
            | "std::io::StderrLock" => {
                // The modelled handle holds no owned resource:
                // writes through it take effect immediately in
                // this model, so the native drop-time flush of
                // the line writer is unobservable.
                return Some(P::Trivial);
            }
            "std::env::Args" | "std::env::ArgsOs" => {
                // The modelled value wraps the argument iterator
                // state ([element snapshot, next index]) whose
                // elements are the argument strings themselves:
                // dropping the iterator drops the not-yet-yielded
                // strings, which is unobservable for string values.
                return Some(P::Trivial);
            }
            "std::env::SplitPaths" => {
                // The modelled value is the `split_paths`
                // builtin's snapshot ([part snapshot, next
                // index]) whose elements are `PathBuf` values:
                // dropping the iterator drops the not-yet-yielded
                // parts, which is unobservable for path values
                // (the same reasoning as the `Args` entry above).
                // The type is opaque (`define_opaque`), so the
                // evidence-based planner cannot see its shape.
                return Some(P::Trivial);
            }
            "std::env::VarError" => {
                // A plain enum with no `Drop` impl. Its only
                // payload is the `NotUnicode` variant's
                // `OsString`, which is the string value itself in
                // this model (see the `OsString` entry above), so
                // the whole glue is trivial.
                return Some(P::Trivial);
            }
            "std::fs::Metadata" => {
                // A plain snapshot of the host `stat` data: no
                // `Drop` impl, and every field (the stat struct,
                // the file type, the times) is plain data, so the
                // whole glue is trivial.
                return Some(P::Trivial);
            }
            "std::fs::DirEntry" => {
                // No `Drop` impl; the modelled value is the
                // entry's path and name, both string data in this
                // model, so the whole glue is trivial.
                return Some(P::Trivial);
            }
            "std::fs::ReadDir" => {
                // The modelled value is the snapshot of the
                // directory's entries plus the read index:
                // dropping it drops the not-yet-yielded entries,
                // which are plain data in this model (the same
                // accounting as `std::env::Args` above).
                return Some(P::Trivial);
            }
            "build_helper::drop_bomb::DropBomb" => {
                // The bomb's own `Drop` impl panics when the
                // bomb is still armed. That check is a
                // recipe-authoring guard, not recipe semantics:
                // the harness's native oracle runs the recipe
                // first, so a recipe whose commands are dropped
                // unexecuted has already failed natively, and
                // the defuse itself is ordinary field data the
                // converted support code maintains. Its fields
                // (an `OsString`, a flag and a `Location`) are
                // all glue-free in this model, so the glue is
                // trivial.
                return Some(P::Trivial);
            }
            "run_make_support::command::Command"
            | "run_make_support::CompletedProcess"
            | "run_make_support::Rustc"
            | "run_make_support::Cargo"
            | "run_make_support::Rustdoc"
            | "run_make_support::Cc"
            | "run_make_support::LlvmAr"
            | "run_make_support::LlvmReadobj" => {
                // The run-make support library's own value
                // types: builders wrapping the modelled
                // `std::process::Command` (a handle whose glue
                // is trivial, above) plus option/string/path
                // data and the `DropBomb` accounted for above,
                // and the completed-process wrapper around the
                // modelled `Output`. None of them has a `Drop`
                // impl, and every field is glue-free in this
                // model, so the glue is trivial. (Their shapes
                // are not in the evidence tables -- the types
                // live in an auxiliary crate whose construction
                // sites are not converted -- so the generic
                // fields plan cannot be derived for them.)
                return Some(P::Trivial);
            }
            "run_make_support::similar::TextDiff"
            | "run_make_support::similar::udiff::UnifiedDiff" => {
                // Diff values are borrowed line data and owned
                // vectors of it: no `Drop` impl, and every
                // field is glue-free in this model.
                return Some(P::Trivial);
            }
            "std::process::Stdio" => {
                // Stdio configuration is not observed by the
                // process model (see the `ProcessCommandStdio`
                // builtin): a modelled `Stdio` value holds no
                // host resource, so its glue is trivial.
                return Some(P::Trivial);
            }
            "std::process::ChildStdin"
            | "std::process::ChildStdout"
            | "std::process::ChildStderr" => {
                // Child pipe handles exist only for a spawned
                // child, which the process model never produces
                // (spawn stays unsupported at its call site);
                // the values are unreachable in a running
                // program, and their glue is trivial.
                return Some(P::Trivial);
            }
            "std::string::FromUtf8Error" => {
                // A byte vector plus a plain validation error:
                // no `Drop` impl, nothing owned beyond the
                // buffer value itself.
                return Some(P::Trivial);
            }
            "std::process::Output" => {
                // No `Drop` impl; the fields are an exit status
                // (a plain wait-status integer) and two byte
                // vectors, whose buffers are values in this
                // model (as for `OsString` above), so the whole
                // glue is trivial.
                return Some(P::Trivial);
            }
            "std::process::ExitStatus" => {
                // A plain wait-status integer in a two-level
                // wrapper: no `Drop` impl, nothing owned.
                return Some(P::Trivial);
            }
            _ => {}
        }
        // Any other ADT: its glue is its `Drop` impl (called first,
        // as in rustc's drop glue) followed by its fields' glue.
        let mut steps = Vec::new();
        let ty_key = format!("{:?}", mir::Ty::Adt(def, args.to_vec()));
        let dtor = conv
            .dtor_bodies
            .get(&ty_key)
            .cloned()
            .or_else(|| conv.dtor_body(&name, args));
        if let Some(func) = dtor {
            steps.push(P::CallDtor { func });
        } else if conv.drop_trivial_adts.contains(&ty_key) {
            // The compiler certifies this instantiation's drop
            // glue empty: no destructor and no field glue exists
            // anywhere in its tree, so the trivial plan is exact
            // rather than an assumption about an unrecorded impl.
            return Some(P::Trivial);
        } else if conv.dtor_impl_known(&name) {
            // The impl exists but has no converted body (a generic
            // impl gets no monomorphic instance in the program):
            // its effects can neither be executed nor ruled out.
            return None;
        } else if !conv.is_local_type(&name) && !conv.drop_no_impl_adts.contains(&ty_key) {
            // Not the entry crate's type: an impl could exist that
            // the name table never recorded. (A type the driver
            // certifies has no `Drop` impl in its own crate — the
            // only crate the orphan rule allows one in — falls
            // through to its fields' glue below.)
            return None;
        }
        steps.push(self.variant_fields_plan(def, args)?);
        Some(collapse_seq(steps))
    }

    /// Variant indices of one ADT / coroutine instantiation with
    /// any evidence in the program: field projections, aggregate
    /// constructions, the discriminant table. Variant 0 always
    /// exists (every ADT and every coroutine has one).
    fn evidenced_variants(
        &self,
        def: mir::DefId,
        args: &[mir::Ty],
    ) -> std::collections::BTreeSet<u32> {
        let conv = self.conv;
        let mut variants: std::collections::BTreeSet<u32> = Default::default();
        variants.insert(0);
        for (key, _) in &conv.adt_fields.entries {
            if key.def == def && key.generic_args.as_slice() == args {
                variants.insert(key.variant);
            }
        }
        for (d, a, v, _) in &conv.adt_aggregates.entries {
            if *d == def && a.as_slice() == args {
                variants.insert(*v);
            }
        }
        if let Some(discs) = conv.discriminants.get(&def) {
            for v in 0..discs.len() {
                variants.insert(v as u32);
            }
        }
        variants
    }

    /// The drop glue of one coroutine instantiation: the live
    /// state's fields, state by state. The compiler numbers the
    /// states 0 (unresumed), 1 (returned), 2 (panicked) and one per
    /// suspension point from 3 on. The plan must *cover* every
    /// state the value can be dropped in: the three fixed states,
    /// plus every state the coroutine's own body names in a
    /// `SetDiscriminant` (the only way a state tag is ever
    /// written), even when that state holds no evidenced fields -
    /// a suspended coroutine with nothing live across the
    /// suspension point is dropped in exactly such a state. A
    /// state with no evidence can still hold live fields, though:
    /// the transformed body projects a saved field under the
    /// index of one state that holds it, not under every state's,
    /// so a field live in several states may be named only under
    /// the later ones. The slot-map pass below accounts for
    /// those fields from the other states' plans.
    fn coroutine_plan(
        &mut self,
        def: mir::DefId,
        args: &[mir::Ty],
    ) -> Option<ir::DropPlan> {
        use ir::DropPlan as P;
        let conv = self.conv;
        let variants = self.evidenced_variants(def, args);
        let mut max_v = 2u32;
        if let Some(&top) = variants.iter().max() {
            max_v = max_v.max(top);
        }
        if let Some(body) = conv
            .def_names
            .get(&def)
            .and_then(|key| conv.bodies.get(key))
        {
            for bb in &body.basic_blocks {
                for stmt in &bb.statements {
                    if let mir::StatementKind::SetDiscriminant {
                        variant_index, ..
                    } = &stmt.kind
                    {
                        max_v = max_v.max(*variant_index);
                    }
                }
            }
        }
        let mut plans = Vec::new();
        for v in 0..=max_v {
            if !variants.contains(&v) {
                plans.push(P::Trivial);
                continue;
            }
            plans.push(self.one_variant_fields(def, args, v, false)?);
        }
        // An async closure's future is a coroutine type carrying
        // the closure's own generic arguments: the closure kind
        // first, and the unresumed state's field types - the call
        // arguments followed by the captures, in the order the
        // call shim stores them - as the last argument's tuple.
        // The closure body is never converted (a body mentioning
        // the coroutine-closure type cannot be converted through
        // Stable MIR), so for the owning (`async_call_once`)
        // instantiation no projection evidence names the unresumed
        // state's fields, and the evidence plan above leaves the
        // state empty even though the value holds every field.
        // Plan the unresumed state from the type's own declaration
        // instead, when the evidence plan drops nothing: a value
        // in the unresumed state has not run, so every declared
        // field is still live in it. (For the borrowing
        // instantiations the declared fields are references, whose
        // plans are trivial, so this changes nothing there; where
        // evidence does plan the state, it agrees with the
        // declaration and stands.)
        if matches!(plans.first(), Some(P::Trivial)) {
            if let (Some(mir::Ty::Int(_)), Some(mir::Ty::Tuple(declared))) =
                (args.first(), args.last())
            {
                let declared = declared.clone();
                let mut field_plans = Vec::with_capacity(declared.len());
                let mut planned_all = true;
                for field_ty in &declared {
                    match self.plan(field_ty) {
                        Some(plan) => field_plans.push(plan),
                        None => {
                            planned_all = false;
                            break;
                        }
                    }
                }
                if planned_all
                    && field_plans.iter().any(|plan| *plan != P::Trivial)
                {
                    plans[0] = P::Fields {
                        fields: field_plans,
                        complete: true,
                    };
                }
            }
        }
        // When the driver recorded the instance's storage-slot
        // map, the conversion rewrote the body's state-field
        // projections to bare slot fields, so at run time the
        // state fields live in the value's flat field vector -
        // the storage a bare field step addresses - and not in
        // the per-state saved storage the glue's `Fields` arm
        // reads for a suspended coroutine. A slot-indexed
        // `Fields` plan would be judged against that saved
        // storage: its entries past the saved vector's length
        // read as never written and are silently skipped, and
        // the surviving indices name the wrong cells. Address
        // each state's field plans at their slots directly
        // instead: an `At` step carrying the bare slot field
        // resolves, through the ordinary projectors, to the
        // cell the body wrote. Variant 0 is the unresumed
        // state, whose fields are the upvars themselves; its
        // positional `Fields` plan already addresses the upvar
        // storage in exactly that way, so it is left alone.
        // The slot is the shared `coroutine_field_slot`
        // judgement, so a provably zero-sized field's plan
        // addresses the same private slot place conversion
        // wrote its value into, not the shared slot of the
        // sized field whose offset it happens to share.
        // The steps run in reverse field order: dropping a
        // suspended coroutine drops the locals live across the
        // suspension point the way the body's own drop
        // elaboration would drop them if the function returned
        // there - last declared, first dropped - and the
        // layout numbers a state's fields in declaration
        // order (parameters first, then locals), so the last
        // field is the last-declared local. The unresumed
        // state above is exempt: its upvars are the value's
        // own fields and drop in field order, like a struct's.
        let key = crate::builtins::ty_tag(&mir::Ty::Coroutine(def, args.to_vec()), conv.names);
        if let Some(slots) = conv.coroutine_layouts.get(&key) {
            for (v, plan) in plans.iter_mut().enumerate().skip(1) {
                if slots.variants.get(v).is_none() {
                    continue;
                }
                if let P::Fields { fields, .. } = plan {
                    let mut steps: Vec<P> = Vec::new();
                    for (f, field_plan) in fields.drain(..).enumerate() {
                        if field_plan == P::Trivial {
                            continue;
                        }
                        let field_ty =
                            conv.adt_fields.get(def, args, v as u32, f as u32);
                        if let Some(slot) =
                            conv.coroutine_field_slot(&key, slots, v, f, field_ty)
                        {
                            steps.push(P::At {
                                path: vec![RefProj::Field(slot as usize)],
                                plan: Box::new(field_plan),
                            });
                        }
                    }
                    steps.reverse();
                    *plan = P::Seq(steps);
                }
            }
            // A saved field is projected by the transformed body
            // under the index of one state that holds it, not
            // under every state's, so a state's own evidence can
            // leave a live field unplanned: an aggregate under
            // construction across two suspension points is
            // projected only under the later state's index, and
            // the earlier state's plan then never mentions the
            // storage cell the in-progress value sits in. A slot
            // is one cell in every state that contains it (the
            // slot map's own invariant), and the value a cell
            // holds in such a state is the same saved local the
            // other states' plans drop at that slot. Fill each
            // state's missing slot steps in from the other
            // states' rewritten plans - but only where every
            // state that plans the slot agrees on the plan:
            // different locals can be packed into one slot in
            // different states, and where the sightings disagree
            // the cell's content in this state is not provable,
            // so it keeps the unplanned treatment rather than a
            // guessed destructor. The state's own steps keep
            // their order - the reverse-declaration drop order
            // the rewrite above established - and a borrowed
            // step takes the place the donor state's plan
            // implies for it, relative to the slots both states
            // plan: two locals live in both states drop in the
            // same relative order in each.
            let snapshot = plans.clone();
            for (v, plan) in plans.iter_mut().enumerate().skip(1) {
                let Some(vslots) = slots.variants.get(v) else {
                    continue;
                };
                let mut steps: Vec<P> = match plan {
                    P::Seq(steps) => std::mem::take(steps),
                    P::Trivial => Vec::new(),
                    _ => continue,
                };
                let mut planned: std::collections::HashSet<usize> =
                    std::collections::HashSet::new();
                for step in steps.iter() {
                    if let P::At { path, .. } = step {
                        if let [RefProj::Field(slot)] = path.as_slice() {
                            planned.insert(*slot);
                        }
                    }
                }
                let mut missing: Vec<(usize, P, Vec<usize>)> = Vec::new();
                let mut seen: std::collections::HashSet<u32> =
                    std::collections::HashSet::new();
                for &slot in vslots.iter() {
                    if !seen.insert(slot) {
                        continue;
                    }
                    let slot = slot as usize;
                    if planned.contains(&slot) {
                        continue;
                    }
                    let mut found: Option<&P> = None;
                    let mut donor: Option<Vec<usize>> = None;
                    let mut ambiguous = false;
                    for (w, wplan) in snapshot.iter().enumerate().skip(1) {
                        if w == v {
                            continue;
                        }
                        let P::Seq(wsteps) = wplan else {
                            continue;
                        };
                        for step in wsteps.iter() {
                            if let P::At { path, plan: sub } = step {
                                if let [RefProj::Field(s)] = path.as_slice() {
                                    if *s == slot {
                                        match found {
                                            None => {
                                                found = Some(&**sub);
                                                donor = Some(
                                                    wsteps
                                                        .iter()
                                                        .filter_map(|st| match st {
                                                            P::At { path, .. } => {
                                                                match path.as_slice() {
                                                                    [RefProj::Field(
                                                                        ds,
                                                                    )] => Some(*ds),
                                                                    _ => None,
                                                                }
                                                            }
                                                            _ => None,
                                                        })
                                                        .collect(),
                                                );
                                            }
                                            Some(prev) if prev == &**sub => {}
                                            Some(_) => ambiguous = true,
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if !ambiguous {
                        if let (Some(sub), Some(donor)) = (found, donor) {
                            missing.push((slot, sub.clone(), donor));
                        }
                    }
                }
                if missing.is_empty() {
                    if let P::Seq(orig) = plan {
                        *orig = steps;
                    }
                    continue;
                }
                missing.sort_by_key(|(slot, _, donor)| {
                    donor.iter().position(|d| d == slot).unwrap_or(usize::MAX)
                });
                for (slot, sub, donor) in &missing {
                    let pos = donor
                        .iter()
                        .position(|d| d == slot)
                        .unwrap_or(usize::MAX);
                    let step_slots: Vec<Option<usize>> = steps
                        .iter()
                        .map(|st| match st {
                            P::At { path, .. } => match path.as_slice() {
                                [RefProj::Field(s)] => Some(*s),
                                _ => None,
                            },
                            _ => None,
                        })
                        .collect();
                    let mut insert_at = steps.len();
                    let mut anchored = false;
                    for j in (0..pos).rev() {
                        if let Some(idx) =
                            step_slots.iter().position(|s| *s == Some(donor[j]))
                        {
                            insert_at = idx + 1;
                            anchored = true;
                            break;
                        }
                    }
                    if !anchored {
                        for j in (pos + 1)..donor.len() {
                            if let Some(idx) =
                                step_slots.iter().position(|s| *s == Some(donor[j]))
                            {
                                insert_at = idx;
                                break;
                            }
                        }
                    }
                    steps.insert(
                        insert_at,
                        P::At {
                            path: vec![RefProj::Field(*slot)],
                            plan: Box::new(sub.clone()),
                        },
                    );
                }
                *plan = P::Seq(steps);
            }
        }
        Some(P::Variants { variants: plans })
    }

    /// The field glue of one ADT, across its variants: a single
    /// `Fields` plan for a struct, a `Variants` plan for an enum.
    fn variant_fields_plan(
        &mut self,
        def: mir::DefId,
        args: &[mir::Ty],
    ) -> Option<ir::DropPlan> {
        use ir::DropPlan as P;
        let variants = self.evidenced_variants(def, args);
        let max_v = *variants.iter().max().expect("variant 0 present");
        let mut plans = Vec::new();
        for v in 0..=max_v {
            if !variants.contains(&v) {
                // A variant index inside the discriminant range
                // that nothing in the program ever constructs or
                // projects: its fields are unaccounted for.
                plans.push(P::Unknown);
                continue;
            }
            plans.push(self.one_variant_fields(def, args, v, true)?);
        }
        if plans.len() == 1 {
            Some(plans.pop().expect("one variant"))
        } else {
            Some(P::Variants { variants: plans })
        }
    }

    /// The field glue of one variant of one ADT. With
    /// `shape_fallback`, the driver's shape table — which declares
    /// every field of the variant with its type — fills in what
    /// the program's evidence misses. Evidence alone misses a
    /// variant that is only ever built as a constant: a
    /// zero-sized value's construction is a constant in MIR, so
    /// no aggregate statement records its field count and no
    /// projection records its fields' types, and the variant's
    /// fields' glue would be planned as nothing at all. Coroutine
    /// states pass `false`: their fields are evidence-based by
    /// design, since a saved field can be moved out of before the
    /// state is dropped.
    fn one_variant_fields(
        &mut self,
        def: mir::DefId,
        args: &[mir::Ty],
        variant: u32,
        shape_fallback: bool,
    ) -> Option<ir::DropPlan> {
        let conv = self.conv;
        let count = conv.adt_aggregates.get(def, args, variant);
        let evidence_count = match count {
            Some(n) => n,
            None => conv
                .adt_fields
                .entries
                .iter()
                .filter(|(key, _)| {
                    key.def == def && key.generic_args.as_slice() == args && key.variant == variant
                })
                .map(|(key, _)| key.field as usize + 1)
                .max()
                .unwrap_or(0),
        };
        let shape_fields = if shape_fallback {
            conv.instantiated_field_tys(&def, args, variant)
        } else {
            None
        };
        let field_count =
            evidence_count.max(shape_fields.as_ref().map_or(0, |tys| tys.len()));
        let mut fields = Vec::with_capacity(field_count);
        for f in 0..field_count as u32 {
            match conv.adt_fields.get(def, args, variant, f) {
                Some(ty) => {
                    let ty = ty.clone();
                    fields.push(self.plan(&ty)?);
                }
                None => match shape_fields
                    .as_ref()
                    .and_then(|tys| tys.get(f as usize))
                {
                    Some(ty) => {
                        let ty = ty.clone();
                        // A field only the shape accounts for:
                        // plan its glue from the declared type. A
                        // part the builder cannot account for keeps
                        // the runtime judgment an unrecorded field
                        // gets, rather than failing the whole glue.
                        fields.push(
                            self.plan(&ty).unwrap_or(ir::DropPlan::Untyped),
                        );
                    }
                    None => {
                        // No projection anywhere records this
                        // field's type (it is only ever written,
                        // e.g. by an aggregate with a literal). Its
                        // glue is judged by the value's own shape
                        // at run time.
                        fields.push(ir::DropPlan::Untyped);
                    }
                },
            }
        }
        Some(collapse_fields(
            fields,
            count.is_some() || shape_fields.is_some(),
        ))
    }
}

impl<'a> Converter<'a> {
    /// The item type of an iterator type, when the type's
    /// shape names it directly: a pass-through adapter
    /// (`Filter`, `Skip`, ...) has its inner iterator's item,
    /// `Enumerate` pairs it with the count, `Zip` pairs the
    /// two sides' items, `Cloned` / `Copied` unwrap the inner
    /// item's reference, and the base iterators whose first
    /// generic argument is the item (`IntoIter`, the ranges,
    /// `Once`, ...) name it themselves. Anything else — a
    /// `Map`'s or `FilterMap`'s closure output, a reference
    /// iterator's borrowed item (its glue is trivial), a
    /// user-defined iterator — names no item here, and the
    /// caller accounts for no glue for it.
    pub(super) fn iterator_item_ty(&self, ty: &mir::Ty) -> Option<mir::Ty> {
        match ty {
            mir::Ty::Ref(_, inner, _) => self.iterator_item_ty(inner),
            mir::Ty::Adt(def, args) => {
                let name = self
                    .names
                    .get(def)
                    .or_else(|| self.def_names.get(def))?;
                if !(name.starts_with("std::")
                    || name.starts_with("core::")
                    || name.starts_with("alloc::"))
                {
                    return None;
                }
                let short = name.rsplit("::").next().unwrap_or(name);
                match short {
                    "Filter" | "Skip" | "SkipWhile" | "Take" | "StepBy" | "Rev"
                    | "Fuse" | "Peekable" | "Inspect" | "Chain" => {
                        self.iterator_item_ty(args.first()?)
                    }
                    "Enumerate" => {
                        let item = self.iterator_item_ty(args.first()?)?;
                        Some(mir::Ty::Tuple(vec![
                            mir::Ty::Uint(mir::UintTy::Usize),
                            item,
                        ]))
                    }
                    "Zip" => {
                        let first = self.iterator_item_ty(args.first()?)?;
                        let second = self.iterator_item_ty(args.get(1)?)?;
                        Some(mir::Ty::Tuple(vec![first, second]))
                    }
                    "Cloned" | "Copied" => {
                        match self.iterator_item_ty(args.first()?)? {
                            mir::Ty::Ref(_, elem, _) => Some((*elem).clone()),
                            _ => None,
                        }
                    }
                    "IntoIter"
                        if name.contains("::hash_map::")
                            || name.contains("::btree_map::") =>
                    {
                        Some(mir::Ty::Tuple(vec![
                            args.first()?.clone(),
                            args.get(1)?.clone(),
                        ]))
                    }
                    "IntoIter" | "Range" | "RangeInclusive" | "RangeFrom"
                    | "Empty" | "Once" | "Repeat" | "Drain" | "IntoKeys" => {
                        args.first().cloned()
                    }
                    "IntoValues" => args.get(1).cloned(),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// The program's iterator-item glue table (see
    /// [`ir::Program::iter_item_glues`]): every type that
    /// occurs as a local's type or as a call callee's generic
    /// argument, in any body of the program, is walked as a
    /// type tree, and each node that is an iterator type with
    /// an accountable, non-trivial item glue contributes its
    /// entry, keyed by the node's own tag — so an adapter and
    /// every iterator nested inside it are all recorded,
    /// whether the program steps them directly or only as
    /// another adapter's inner iterator.
    pub(super) fn iter_item_glues(
        &self,
        program: &super::program::MirProgram,
    ) -> HashMap<String, ir::DropGlue> {
        let mut out: HashMap<String, ir::DropGlue> = HashMap::new();
        let mut stack: Vec<mir::Ty> = Vec::new();
        for func in &program.functions {
            stack.extend(func.body.local_decls.iter().map(|decl| decl.ty.clone()));
            for block in &func.body.basic_blocks {
                let Some(term) = &block.terminator else {
                    continue;
                };
                let mir::TerminatorKind::Call { func, .. } = &term.kind else {
                    continue;
                };
                let mir::Operand::Constant(c) = func else {
                    continue;
                };
                if let mir::ConstKind::Ty(mir::Ty::FnDef(_, gargs)) = &c.literal {
                    stack.extend(gargs.iter().cloned());
                }
            }
        }
        while let Some(ty) = stack.pop() {
            if let Some(item) = self.iterator_item_ty(&ty) {
                if let Some(glue) = DropGlueBuilder::build(self, &item) {
                    if glue.root != ir::DropPlan::Trivial {
                        out.entry(crate::builtins::ty_tag(&ty, self.names))
                            .or_insert(glue);
                    }
                }
            }
            match &ty {
                mir::Ty::Tuple(ts) => stack.extend(ts.iter().cloned()),
                mir::Ty::Array(elem, _)
                | mir::Ty::Slice(elem)
                | mir::Ty::RawPtr(elem, _)
                | mir::Ty::Pat(elem, _) => stack.push((**elem).clone()),
                mir::Ty::Ref(_, elem, _) | mir::Ty::Dynamic(_, elem, _) => {
                    stack.push((**elem).clone())
                }
                mir::Ty::Adt(_, args)
                | mir::Ty::Closure(_, args)
                | mir::Ty::Coroutine(_, args)
                | mir::Ty::CoroutineWitness(_, args)
                | mir::Ty::Alias(_, _, args)
                | mir::Ty::FnDef(_, args) => stack.extend(args.iter().cloned()),
                mir::Ty::FnPtr(sig) => {
                    stack.extend(sig.inputs.iter().cloned());
                    stack.push((*sig.output).clone());
                }
                _ => {}
            }
        }
        out
    }
}
