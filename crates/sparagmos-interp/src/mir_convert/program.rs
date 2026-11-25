//! Program-level data: the MIR program handed over by the driver, its function keys and static seeds.

use std::collections::{HashMap, HashSet};

use sparagmos_mir as mir;

use crate::ir;

use super::converter::*;
use super::errors::*;


/// One function of a MIR program, as handed over by the driver: a display
/// name (the interpreter keys functions by name), the `DefId` when known
/// (used to resolve `FnDef` call operands), and the body itself.
#[derive(Debug, Clone)]
pub struct MirFunction {
    pub name: String,
    pub def_id: Option<mir::DefId>,
    pub body: mir::Body,
}

/// A set of MIR functions forming one program (one crate's worth of
/// bodies, as produced by the driver on the main path).
#[derive(Debug, Clone, Default)]
pub struct MirProgram {
    pub functions: Vec<MirFunction>,
    /// `DefId` -> display name for *every* named item the driver
    /// knows, including functions and types without converted bodies
    /// (std callees, ADT types). Used to name ADT locals and to
    /// resolve calls into runtime support.
    pub fn_names: HashMap<mir::DefId, String>,
    /// Declared ABIs of the program's function instances, keyed
    /// by the same `DefId` a converted body carries, as recorded
    /// by the driver when it registered each instance.
    /// Consulted for a naked function's asm block: its incoming
    /// arguments arrive in the registers of the function's ABI
    /// (see `asm_arg_seed_plan`).
    pub fn_abis: HashMap<mir::DefId, mir::Abi>,
    /// Discriminant values of the enum ADTs in the program, keyed by
    /// `DefId` (variant index -> discriminant value), as recorded by
    /// the driver. `Discriminant` rvalues are evaluated with these
    /// values so they agree with the `SwitchInt` branch keys, which
    /// carry discriminant values rather than variant indices.
    pub adt_discriminants: mir::AdtDiscriminants,
    /// Layouts of the monomorphic ADT instances in the program, as
    /// recorded by the driver from the compiler's layout query
    /// (see [`mir::AdtLayouts`]). Consulted by the `size_of` /
    /// `align_of` / `size_of_val` builtins for ADT type tags.
    pub adt_layouts: mir::AdtLayouts,
    /// Per-variant field byte layouts of the monomorphic ADT
    /// instances in the program, as recorded by the driver
    /// (see [`mir::AdtFieldLayouts`]). Consulted by the
    /// interpreter's byte view of an ADT value's storage.
    pub adt_field_layouts: mir::AdtFieldLayouts,
    /// Storage layouts of the monomorphic tuple types in
    /// the program, as recorded by the driver (see
    /// [`mir::TupleLayouts`]). Consulted when a tuple's
    /// scalar shape is built: a tuple's fields sit at the
    /// compiler's chosen offsets, so the shape is sound
    /// only at the recorded ones.
    pub tuple_layouts: mir::TupleLayouts,
    /// Direct discriminant tag layouts of the monomorphic
    /// enum instances in the program, as recorded by the
    /// driver (see [`mir::AdtTagLayouts`]). Consulted by the
    /// interpreter's byte view of an enum value's storage.
    pub adt_tag_layouts: mir::AdtTagLayouts,
    /// Storage-slot maps of the program's monomorphic coroutine
    /// instances, as recorded by the driver (see
    /// `mir::CoroutineLayouts`). Consulted when a place projects
    /// a state field of a coroutine value: the flat positional
    /// field vector does not by itself model the state-machine
    /// layout's sharing of one field across suspension states.
    pub coroutine_layouts: mir::CoroutineLayouts,
    /// Declared shapes of the program's ADTs, as recorded by the
    /// driver (see `mir::AdtShape`). Used to decide which locals
    /// are provably zero-sized.
    pub adt_shapes: mir::AdtShapes,
    /// Derived-`Debug` shapes of the program's monomorphic ADT
    /// instances, as recorded by the driver (see
    /// `mir::AdtDebugShapes`). Consulted by the `Debug` formatter
    /// for ADT values.
    pub adt_debug_shapes: mir::AdtDebugShapes,
    /// Debugger shapes of the program's monomorphic closure and
    /// coroutine instances, as recorded by the driver (see
    /// `mir::EnvDebugShapes`). Consulted by the debugger
    /// interface when it prints a closure or coroutine value.
    pub env_debug_shapes: mir::EnvDebugShapes,
    /// Thread-local statics of the program, as recorded by the
    /// driver (see `mir::ThreadLocalDef`): each entry's initializer
    /// seeds one thread's storage for that static.
    pub thread_locals: mir::ThreadLocals,
    /// Declared types of the program's statics, as recorded by
    /// the driver (see `mir::StaticTys`).
    pub static_tys: mir::StaticTys,
    /// Field types of monomorphic ADT instances whose generic
    /// shape mentions a type alias, as recorded by the driver
    /// (see `mir::AdtInstanceShapes`). Consulted for an ADT
    /// instance's field types before the generic shape, whose
    /// alias field types cannot be resolved here.
    pub adt_instance_shapes: mir::AdtInstanceShapes,
    /// Niche discriminant encodings of monomorphic enum
    /// instances, as recorded by the driver (see
    /// `mir::AdtNiches`). Consulted when a transmute involves
    /// a niche-encoded enum such as `Option<NonZero<T>>`.
    pub adt_niches: mir::AdtNiches,
    /// Materialisation plans for the pointee types of the
    /// program's pointer casts, as recorded by the driver (see
    /// `sparagmos_interp::ir::MatPlan`), keyed by the pointee
    /// type's `Debug` rendering. Consulted when a pointer cast
    /// is converted: a cast whose operand may be a type-erased
    /// constant allocation becomes a materialising cast.
    pub mat_plans: HashMap<String, ir::MatPlan>,
    /// The source-file table the bodies' source lines index into
    /// (see `mir::Body::source_lines`), as recorded by the driver.
    /// Handed to the interpreter program for the debugger
    /// interface.
    pub source_files: Vec<String>,
    /// The program's coverage mapping table, as recorded by the
    /// driver (see `mir::CoverageMapping`). Handed to the
    /// interpreter program, whose execution counts the records'
    /// physical counters.
    pub coverage: mir::CoverageMapping,
    /// ADT instantiations whose drop glue the compiler certifies
    /// empty, as recorded by the driver (it resolves each dropped
    /// type's drop-in-place shim through Stable MIR and keeps the
    /// ones the compiler reports as empty shims). Keyed by the
    /// `Debug` rendering of the instantiation's type, the same
    /// rendering the drop-glue builder keys its ADT table with.
    /// Consulted for an ADT with no converted destructor body:
    /// certification covers the whole tree, fields included, so
    /// the trivial plan is exact for these types.
    pub drop_trivial_adts: std::collections::HashSet<String>,
    /// ADT instantiations the driver certifies have no `Drop`
    /// impl at all (its crate — the only one the orphan rule
    /// allows an impl in — defines none for the type). Keyed
    /// like `drop_trivial_adts`. A certified type's glue is
    /// exactly its fields' glue, so the glue builder may use
    /// its field-evidence plan for a foreign type it would
    /// otherwise have to refuse (an unrecorded impl cannot be
    /// ruled out for an uncertified foreign type).
    pub drop_no_impl_adts: std::collections::HashSet<String>,
    /// Destructor bodies the driver demanded for specific ADT
    /// instantiations, keyed by the `Debug` rendering of the
    /// instantiation's type (the certification tables' key) with
    /// the program name of the converted `Drop::drop` body as
    /// the value. The glue builder's own key construction
    /// renders generic arguments its way, which cannot reproduce
    /// the compiler's instance names for every argument shape
    /// (a closure argument is the case that forced this table);
    /// the driver knows the exact name it demanded.
    pub dtor_bodies: std::collections::HashMap<String, String>,

    /// Definition-location stamps of the program's
    /// `#[track_caller]` instances, keyed by the instance's
    /// `DefId`, as recorded by the driver (see
    /// `Converter::track_caller_defs` in the driver crate).
    pub track_caller_defs: HashMap<mir::DefId, mir::ConstValue>,
    /// The location stamps of the `Fn` / `FnMut` / `FnOnce`
    /// vtable shims (index 0 is `call`, 1 is `call_mut`, 2 is
    /// `call_once`), as recorded by the driver.
    pub fn_shim_locations: [Option<mir::ConstValue>; 3],
    /// Link symbols of the program's function instances, keyed
    /// by the instance's `DefId`, as recorded by the driver (see
    /// `Converter::link_symbols` in the driver crate). A
    /// body-less extern declaration binds to the unique
    /// converted body exporting the same symbol (see
    /// `Converter::link_symbol_route`).
    pub link_symbols: HashMap<mir::DefId, String>,
    /// Sparagmos ids of the definitions the compiler classifies
    /// as compiler intrinsics, as recorded by the driver (see
    /// `Converter::note_intrinsic` in the driver crate). The
    /// converter's name-only intrinsic fallback serves a
    /// body-less callee only when its definition is in this set:
    /// an ordinary function whose body the driver skipped (or
    /// whose conversion failed) keeps that failure even when
    /// its name matches an intrinsic's.
    pub intrinsic_defs: std::collections::HashSet<mir::DefId>,
    /// Definitions of coroutine closures (async closures) the
    /// driver met while converting, keyed like the bodies.
    /// An async closure's own body cannot be converted (it
    /// mentions the coroutine-closure type, which Stable MIR
    /// cannot convert), so the conversion records the
    /// definition here instead: a definition in this set is
    /// served at call time by constructing the closure's
    /// coroutine, for the `Fn` trait family exactly as for
    /// the `AsyncFn` family. Populated from type mentions,
    /// creation aggregates and zero-sized closure constants.
    pub async_closure_defs: std::collections::HashSet<mir::DefId>,
}

/// Widen an `f16` bit pattern to `f64`. The conversion is exact:
/// every f16 value (including subnormals) is representable in f64,
/// and scaling by a power of two never rounds.
/// The pointee type of a raw-pointer / reference type.
pub(super) fn ptr_pointee(t: &mir::Ty) -> Option<&mir::Ty> {
    match t {
        mir::Ty::RawPtr(p, _) | mir::Ty::Ref(_, p, _) => Some(&**p),
        _ => None,
    }
}

pub(crate) fn f16_to_f64(bits: u16) -> f64 {
    crate::f128::f16_bits_to_f64(bits)
}

impl MirProgram {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a function known only by name.
    pub fn add(&mut self, name: impl Into<String>, body: mir::Body) -> &mut Self {
        self.functions.push(MirFunction {
            name: name.into(),
            def_id: None,
            body,
        });
        self
    }

    /// Add a function together with its `DefId`, so calls to it (whose
    /// func operand is a `FnDef` constant) can be resolved.
    pub fn add_with_def_id(
        &mut self,
        name: impl Into<String>,
        def_id: mir::DefId,
        body: mir::Body,
    ) -> &mut Self {
        self.functions.push(MirFunction {
            name: name.into(),
            def_id: Some(def_id),
            body,
        });
        self
    }

    pub fn get(&self, name: &str) -> Option<&MirFunction> {
        self.functions.iter().find(|f| f.name == name)
    }
}

pub(super) fn program_keys(program: &MirProgram) -> Result<Vec<String>, ConvertError> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for f in &program.functions {
        *counts.entry(f.name.as_str()).or_default() += 1;
    }
    let mut used: HashSet<String> = HashSet::new();
    let mut keys = Vec::with_capacity(program.functions.len());
    for f in &program.functions {
        let key = if counts[f.name.as_str()] == 1 {
            f.name.clone()
        } else {
            match f.def_id {
                Some(def) => format!("{}#{def}", f.name),
                None => {
                    return Err(ConvertError::DuplicateFunction {
                        name: f.name.clone(),
                    });
                }
            }
        };
        if !used.insert(key.clone()) {
            return Err(ConvertError::DuplicateFunction {
                name: f.name.clone(),
            });
        }
        keys.push(key);
    }
    Ok(keys)
}

/// The key [`program_keys`] stores one member of `program` under:
/// its name, or the `DefId`-disambiguated form when another member
/// shares the name. A caller that addresses a single known
/// function from outside the program — the driver naming the run
/// entry point — uses this, so the name it hands the interpreter
/// is the one conversion actually stored the body under. (With a
/// shared name and no `DefId` to disambiguate by, conversion as a
/// whole fails in [`program_keys`]; the bare name is returned here
/// and that failure surfaces first.)
pub fn program_key_of(program: &MirProgram, func: &MirFunction) -> String {
    match func.def_id {
        Some(def)
            if program
                .functions
                .iter()
                .filter(|f| f.name == func.name)
                .count()
                > 1 =>
        {
            format!("{}#{def}", func.name)
        }
        _ => func.name.clone(),
    }
}

/// Merge one conversion round's collected static seeds into
/// the program-wide table (see [`Converter::register_static_seed`]
/// for the priority rule). Re-converting a body in a later
/// round produces the same seeds as the round that first
/// converted it, so merging is idempotent, and a later round
/// can only upgrade a static's seed, never downgrade it.
pub(super) fn merge_static_seeds(
    dst: &mut HashMap<String, (u8, ir::Constant)>,
    conv: &Converter,
) {
    for (name, (priority, seed)) in conv.static_seeds.borrow().iter() {
        match dst.get(name) {
            Some((existing, _)) if *existing >= *priority => {}
            _ => {
                dst.insert(name.clone(), (*priority, seed.clone()));
            }
        }
    }
}
