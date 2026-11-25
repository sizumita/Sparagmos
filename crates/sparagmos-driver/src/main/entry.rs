//! Entry-point resolution and termination reporting for the collected program.

use super::*;

/// The Sparagmos `DefId` of the entry point rustc designated
/// for this crate, if it has one: the user's `main` in a binary
/// crate, the generated harness main in a `--test` crate. The
/// two share the definition name `<crate>::main` in a test
/// crate, so the entry cannot be recovered from names; it is
/// resolved here through the same instance the item loop
/// converted, to the id that body carries in
/// `Collected::body_def_ids`.
/// The entry point's Stable MIR body, fetched the way the item
/// loop fetched it, for the reachability collection's
/// execution-first order: the walk scans it ahead of the other
/// seeds, so the entry's own demand closure is recorded before
/// any other local body's. Empty when the crate has no entry
/// point (a library crate) or the body cannot be materialized.
pub(crate) fn entry_seed_bodies(
    items: &[rustc_public::CrateItem],
) -> Vec<rustc_public::mir::Body> {
    // The entry item: the one rustc designated when it did
    // (a binary compilation), and otherwise the first local
    // item named `main` -- the same resolution `build_program`
    // applies to the converted bodies, since run mode's lib
    // pin designates no entry point there.
    let entry = std::panic::catch_unwind(rustc_public::entry_fn)
        .ok()
        .flatten();
    let item = entry
        .and_then(|entry| {
            items.iter().find(|item| item.def_id() == entry.def_id())
        })
        .or_else(|| items.iter().find(|item| is_main(&item.name())))
        .or_else(|| {
            // An imported `main` (a crate-root re-export of a
            // local function): the root module's `main` child
            // (root_main_target) names the item by its def
            // path, which is the item's name in Stable MIR's
            // spelling.
            let tcx = sparagmos_driver::convert::compiler_tcx()?;
            let path = tcx.def_path_str(root_main_target()?);
            items.iter().find(|item| item.name() == path)
        });
    let Some(item) = item else {
        return Vec::new();
    };
    let Ok(inst) = Instance::try_from(item.clone()) else {
        return Vec::new();
    };
    sparagmos_driver::convert::with_body_materialization(&inst.name(), || {
        inst.body()
    })
    .into_iter()
    .collect()
}

pub(crate) fn entry_point_def_id(
    conv: &mut Converter,
    items: &[rustc_public::CrateItem],
    body_def_ids: &[sparagmos_mir::DefId],
) -> Option<sparagmos_mir::DefId> {
    let entry = std::panic::catch_unwind(rustc_public::entry_fn)
        .ok()
        .flatten();
    if let Some(entry) = entry {
        let item = items.iter().find(|item| item.def_id() == entry.def_id())?;
        return match Instance::try_from(item.clone()) {
            Ok(inst) => Some(conv.instance_def_id(item.def_id(), &inst)),
            Err(_) => None,
        };
    }
    resolver_main_def_id(conv, body_def_ids)
}

/// The target of the crate root's `main` child: the function a
/// crate-root item named `main` resolves to, following a
/// re-export (`use foo::bar as main;`,
/// `pub use main_functions::boilerplate as main;`). This is the
/// resolution rustc's own entry search performs in
/// rustc_resolve's `resolve_main` -- which, like the entry_fn
/// query, only runs for an executable crate type, so under run
/// mode's lib pin neither record exists and the resolution is
/// replayed here from the root module's children.
fn root_main_target() -> Option<rustc_span::def_id::DefId> {
    let tcx = sparagmos_driver::convert::compiler_tcx()?;
    let root = rustc_span::def_id::CRATE_DEF_ID;
    for child in tcx.module_children_local(root) {
        if child.ident.name == rustc_span::sym::main {
            if let rustc_hir::def::Res::Def(_, did) = child.res {
                return Some(did);
            }
        }
    }
    None
}

/// The entry point an imported `main` designates, for the
/// compilations where `rustc_public::entry_fn` designates none
/// (see root_main_target). The target's def-path string names
/// the converted instance: the converter's def table spells a
/// foreign item with the local crate's prefix
/// (`<crate>::main_functions::boilerplate`), so an exact match
/// or a `::{path}` suffix match identifies it. Only a def with
/// a converted body qualifies, and only when the match is
/// unique: naming a def conversion skipped, or one of several
/// same-named instances, would designate an entry execution
/// cannot run or cannot identify.
fn resolver_main_def_id(
    conv: &Converter,
    body_def_ids: &[sparagmos_mir::DefId],
) -> Option<sparagmos_mir::DefId> {
    let tcx = sparagmos_driver::convert::compiler_tcx()?;
    let path = tcx.def_path_str(root_main_target()?);
    let suffix = format!("::{path}");
    let mut found: Option<sparagmos_mir::DefId> = None;
    for (idx, name) in conv.def_names.iter().enumerate() {
        if name == &path || name.ends_with(&suffix) {
            let id = sparagmos_mir::DefId::new(0, idx as u32);
            if !body_def_ids.contains(&id) {
                continue;
            }
            if found.is_some() {
                return None;
            }
            found = Some(id);
        }
    }
    found
}

pub(crate) fn is_main(name: &str) -> bool {
    name == "main" || name.ends_with("::main")
}

/// Names of `FnDef` call targets that have no converted body in this
/// program (library / core functions, `println!` machinery, ...), in
/// the driver's `def_names` spelling. Used to make run-mode failure
/// reasons concrete about *which* call could not be resolved.
pub(crate) fn external_callees(c: &Collected) -> Vec<String> {
    let defined: BTreeSet<&str> = c.body_def_names.iter().map(|n| n.as_str()).collect();
    let mut out: BTreeSet<String> = BTreeSet::new();
    for (_, body) in &c.bodies {
        for bb in &body.basic_blocks {
            let Some(term) = &bb.terminator else { continue };
            let sparagmos_mir::TerminatorKind::Call { func, .. } = &term.kind else {
                continue;
            };
            let sparagmos_mir::Operand::Constant(konst) = func else {
                continue;
            };
            let sparagmos_mir::ConstKind::Ty(sparagmos_mir::Ty::FnDef(def, _)) = &konst.literal
            else {
                continue;
            };
            if def.krate != 0 {
                continue;
            }
            if let Some(name) = c.def_names.get(def.index as usize) {
                if !defined.contains(name.as_str()) {
                    out.insert(name.clone());
                }
            }
        }
    }
    out.into_iter().collect()
}

/// Execute the collected program's `main` in the interpreter and
/// return the run-mode process exit code (see the crate docs).
/// Whether the entry body at `entry_index` returns
/// `std::process::ExitCode` (its local 0 is that ADT).
pub(crate) fn entry_returns_exit_code(c: &Collected, entry_index: usize) -> bool {
    let Some((_, body)) = c.bodies.get(entry_index) else {
        return false;
    };
    let Some(decl) = body.local_decls.first() else {
        return false;
    };
    let sparagmos_mir::Ty::Adt(def, _) = &decl.ty else {
        return false;
    };
    c.def_names
        .get(def.index as usize)
        // The compiler-generated test harness main names the
        // type by the path visible in its generated module
        // (`<crate>::test::ExitCode`), not the defining path;
        // the value shape checked by `exit_code_value` keeps
        // the match honest.
        .map(|n| n == "std::process::ExitCode" || n.ends_with("::ExitCode"))
        .unwrap_or(false)
}

/// The code byte of an `ExitCode` value: the wrapper aggregate
/// around the platform aggregate around the byte, the shape the
/// `ExitCode::from` builtin builds and the collected
/// `Termination::report` body passes through.
pub(crate) fn exit_code_value(value: &sparagmos_interp::Value) -> Option<i32> {
    let sparagmos_interp::Value::Enum { fields, .. } = value else {
        return None;
    };
    let [inner] = fields.as_slice() else {
        return None;
    };
    let sparagmos_interp::Value::Enum { fields, .. } = inner else {
        return None;
    };
    let [byte] = fields.as_slice() else {
        return None;
    };
    match byte {
        sparagmos_interp::Value::Uint(v) => Some(*v as u8 as i32),
        sparagmos_interp::Value::Int(v) => Some(*v as u8 as i32),
        _ => None,
    }
}

/// Whether the entry body at `entry_index` returns
/// `core::result::Result` (its local 0 is that ADT): a
/// `fn main() -> Result<_, _>`, whose `Termination::report`
/// the driver applies itself (see `execute`).
pub(crate) fn entry_returns_result(c: &Collected, entry_index: usize) -> bool {
    let Some((_, body)) = c.bodies.get(entry_index) else {
        return false;
    };
    let Some(decl) = body.local_decls.first() else {
        return false;
    };
    let sparagmos_mir::Ty::Adt(def, _) = &decl.ty else {
        return false;
    };
    c.def_names
        .get(def.index as usize)
        .map(|n| n == "core::result::Result" || n == "std::result::Result")
        .unwrap_or(false)
}

/// The outcome of applying `Termination::report` to a returned
/// `Result` value, following the standard library's own
/// delegation: `Err(e)` prints `Error: {e:?}` and exits with
/// `ExitCode::FAILURE`; `Ok(v)` reports exactly as `v`'s own
/// `Termination` does -- `()` succeeds silently, an `ExitCode`
/// payload yields its code, a nested `Result` recurses.
pub(crate) enum TerminationReport<'a> {
    /// The report ends the process with this code, printing
    /// nothing.
    Code(i32),
    /// The report prints `Error: {payload:?}` -- the payload,
    /// rendered under the type tag of the `Err` type at the
    /// level that produced it -- drops the payload, and exits
    /// with `ExitCode::FAILURE`. The `Err` type itself rides
    /// along so the executing interpreter can build the
    /// payload's drop glue from it.
    Error(
        &'a sparagmos_interp::Value,
        String,
        &'a sparagmos_mir::Ty,
    ),
}

/// `Termination::report` for the entry's returned value, walked
/// along the entry's declared return type `ty` (a
/// `core::result::Result` ADT; see [`entry_result_ty`]): the
/// value alone does not say which of its payloads are themselves
/// `Termination` values, the type does. `None` when the value
/// does not have the one-field-per-variant shape its type
/// declares, or when an `Ok` payload's type has no `Termination`
/// reading the driver models -- an unhandled payload is
/// classified unsupported, never reported as a success with the
/// wrong exit code.
pub(crate) fn result_report<'a>(
    c: &Collected,
    names: &std::collections::HashMap<sparagmos_mir::DefId, String>,
    ty: &'a sparagmos_mir::Ty,
    value: &'a sparagmos_interp::Value,
) -> Option<TerminationReport<'a>> {
    let sparagmos_mir::Ty::Adt(def, args) = ty else {
        return None;
    };
    let is_result = c
        .def_names
        .get(def.index as usize)
        .map(|n| n == "core::result::Result" || n == "std::result::Result")
        .unwrap_or(false);
    if !is_result {
        return None;
    }
    let sparagmos_interp::Value::Enum { variant, fields } = value else {
        return None;
    };
    let [payload] = fields.as_slice() else {
        return None;
    };
    match variant {
        1 => {
            let err_ty = args.get(1)?;
            Some(TerminationReport::Error(
                payload,
                sparagmos_interp::builtins::ty_tag(err_ty, names),
                err_ty,
            ))
        }
        0 => ok_payload_report(c, names, args.first()?, payload),
        _ => None,
    }
}

/// `Termination::report` for an `Ok` payload, delegated to the
/// payload type's own `Termination` impl, as
/// `Result<T, E>::report` delegates `Ok(value)` to
/// `T::report()`.
fn ok_payload_report<'a>(
    c: &Collected,
    names: &std::collections::HashMap<sparagmos_mir::DefId, String>,
    ty: &'a sparagmos_mir::Ty,
    value: &'a sparagmos_interp::Value,
) -> Option<TerminationReport<'a>> {
    match ty {
        // `()` reports success and prints nothing.
        sparagmos_mir::Ty::Tuple(ts) if ts.is_empty() => {
            matches!(value, sparagmos_interp::Value::Unit).then_some(TerminationReport::Code(0))
        }
        sparagmos_mir::Ty::Adt(def, _) => {
            let name = c.def_names.get(def.index as usize)?;
            if name == "std::process::ExitCode" || name.ends_with("::ExitCode") {
                exit_code_value(value).map(TerminationReport::Code)
            } else if name == "core::result::Result" || name == "std::result::Result" {
                result_report(c, names, ty, value)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// The declared return type of the entry body at `entry_index`
/// (its local 0), when it is an ADT: the type the
/// [`result_report`] `Termination` walk follows.
pub(crate) fn entry_result_ty<'a>(c: &'a Collected, entry_index: usize) -> Option<&'a sparagmos_mir::Ty> {
    let (_, body) = c.bodies.get(entry_index)?;
    let decl = body.local_decls.first()?;
    let sparagmos_mir::Ty::Adt(..) = &decl.ty else {
        return None;
    };
    Some(&decl.ty)
}
