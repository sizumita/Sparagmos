//! Assembly of the collected program into the executable program handed to the engines.

use super::*;

/// A converted program ready to run, with its entry point
/// resolved: the shared front half of the run and debug entry
/// points (the two differ only in how the interpreter is driven).
pub(crate) struct BuiltProgram {
    pub(crate) program: sparagmos_interp::MirProgram,
    /// The program key of the entry body (see `execute`).
    pub(crate) entry: String,
    /// The display name of the entry body.
    pub(crate) entry_name: String,
    /// Index of the entry body in `Collected::bodies`.
    pub(crate) entry_index: usize,
    /// The arguments the entry invocation supplies.
    pub(crate) entry_args: Vec<sparagmos_interp::Value>,
    /// The entry body's declared argument count.
    pub(crate) entry_arg_count: usize,
}

/// Build the interpreter program from a collection and resolve
/// its entry point. `program_name` is the name the entry
/// invocation presents as argv[0] (the invocation path, which
/// a caller reconstructing a run may name separately from the
/// source it converted). On failure the run-mode error has
/// already been reported and the `Err` carries the process
/// exit code.
pub(crate) fn build_program(
    c: &Collected,
    program_name: &str,
    guest_args: &[String],
) -> Result<BuiltProgram, i32> {
    let mut program = sparagmos_interp::MirProgram::new();
    for ((name, body), def_id) in c.bodies.iter().zip(c.body_def_ids.iter()) {
        program.functions.push(sparagmos_interp::MirFunction {
            name: name.clone(),
            def_id: Some(*def_id),
            body: body.clone(),
        });
    }
    // The interpreter names ADT locals and resolves runtime-support
    // calls through the full name table (every named item, including
    // body-less std functions and types). `def_names` is indexed by
    // the DefId index within crate 0.
    for (index, name) in c.def_names.iter().enumerate() {
        program.fn_names.insert(
            sparagmos_mir::DefId::new(0, index as u32),
            name.clone(),
        );
    }
    program.intrinsic_defs = c.intrinsic_defs.clone();
    program.async_closure_defs = c.async_closure_defs.clone();
    program.vtable_identities = c.vtable_identities.clone();
    program.fn_abis = c.fn_abis.clone();
    program.adt_discriminants = c.adt_discriminants.clone();
    program.adt_layouts = c.adt_layouts.clone();
    program.adt_dst_layouts = c.adt_dst_layouts.clone();
    program.adt_field_layouts = c.adt_field_layouts.clone();
    program.tuple_layouts = c.tuple_layouts.clone();
    program.adt_tag_layouts = c.adt_tag_layouts.clone();
    program.coroutine_layouts = c.coroutine_layouts.clone();
    program.adt_shapes = c.adt_shapes.clone();
    program.adt_debug_shapes = c.adt_debug_shapes.clone();
    program.env_debug_shapes = c.env_debug_shapes.clone();
    program.thread_locals = c.thread_locals.clone();
    program.extra_static_seeds = c.extra_static_seeds.clone();
    program.static_tys = c.static_tys.clone();
    program.adt_instance_shapes = c.adt_instance_shapes.clone();
    program.adt_niches = c.adt_niches.clone();
    program.mat_plans = c.mat_plans.clone();
    program.drop_trivial_adts = c.drop_trivial_adts.clone();
    program.drop_no_impl_adts = c.drop_no_impl_adts.clone();
    program.dtor_bodies = c.dtor_bodies.clone();
    program.track_caller_defs = c.track_caller_defs.clone();
    program.fn_shim_locations = c.fn_shim_locations.clone();
    program.link_symbols = c.link_symbols.clone();
    program.source_files = c.source_files.clone();
    program.coverage = c.coverage.clone();
    program.fn_display_names = c.fn_display_names.clone();
    program.crate_overflow_checks = c.crate_overflow_checks.clone();
    // The entry body: the one rustc designated (see
    // `Collected::entry_def_id`) when collection recorded it, and
    // otherwise the first local body named `main`. A `--test`
    // crate has two bodies named `<crate>::main` — the user's and
    // the harness's — and only the designated one is the entry:
    // the user's `main` is an ordinary function there, neither
    // the entry nor run twice.
    let entry_index = c
        .entry_def_id
        .and_then(|def| {
            // The designated entry is authoritative wherever
            // its body sits: an imported `main` from a foreign
            // crate is a demanded body, past the local prefix.
            c.body_def_ids.iter().position(|id| *id == def)
        })
        .or_else(|| {
            c.bodies[..c.local_body_count]
                .iter()
                .position(|(n, _)| is_main(n))
        });
    let Some(entry_index) = entry_index else {
        for f in &c.report.functions {
            if is_main(&f.name) {
                if let FnOutcome::Skipped { reasons } = &f.outcome {
                    let rs: Vec<String> =
                        reasons.iter().map(|(k, v)| format!("{k} x{v}")).collect();
                    eprintln!(
                        "sparagmos-driver run: unsupported: `main` was not converted ({})",
                        rs.join(", ")
                    );
                    return Err(2);
                }
            }
        }
        eprintln!(
            "sparagmos-driver run: unsupported: no `main` function among the converted functions"
        );
        return Err(2);
    };
    // The interpreter keys a body whose name another body shares
    // under a `DefId`-disambiguated key; hand it the key
    // conversion stored this body under, not the bare name (the
    // bare name of a `--test` crate's harness main is also the
    // user's `main`, and names no stored body at all).
    let entry = sparagmos_interp::program_key_of(&program, &program.functions[entry_index]);
    let entry_name = program.functions[entry_index].name.clone();
    // A `#[no_main]` entry in the C convention
    // (`extern "C" fn main(argc, argv)`) declares two parameters;
    // the interpreter's call machinery checks the argument count,
    // so the entry invocation supplies the modelled `argc`/`argv`
    // a native `_start` would pass: the program path followed by
    // the guest arguments, each a NUL-terminated byte string, and
    // a null terminator for the array. A standard entry declares
    // no parameters and receives none, as before.
    let entry_arg_count = c.bodies[entry_index].1.arg_count;
    let entry_args: Vec<sparagmos_interp::Value> = if entry_arg_count == 2 {
        let mut argv: Vec<sparagmos_interp::Value> = Vec::new();
        let mut all_args = vec![program_name.to_string()];
        all_args.extend(guest_args.iter().cloned());
        for arg in &all_args {
            let mut bytes = arg.as_bytes().to_vec();
            bytes.push(0);
            argv.push(sparagmos_interp::Value::Ref(
                sparagmos_interp::RefTarget::value(sparagmos_interp::Value::Bytes(bytes)),
            ));
        }
        argv.push(sparagmos_interp::Value::Ref(
            sparagmos_interp::RefTarget::Null,
        ));
        vec![
            sparagmos_interp::Value::Int(all_args.len() as i128),
            // The array pointer denotes its first element, as a
            // native argv does: element reads and pointer
            // arithmetic over the array go through the index
            // projection.
            sparagmos_interp::Value::Ref(sparagmos_interp::RefTarget::Value(
                Box::new(sparagmos_interp::Value::Tuple(argv)),
                vec![sparagmos_interp::RefProj::Index(0)],
            )),
        ]
    } else {
        Vec::new()
    };
    Ok(BuiltProgram {
        program,
        entry,
        entry_name,
        entry_index,
        entry_args,
        entry_arg_count,
    })
}
