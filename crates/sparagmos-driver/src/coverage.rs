//! Coverage instrumentation support: capturing the coverage
//! mapping table of a crate compiled with
//! `-C instrument-coverage`, and rendering the two outputs
//! compiletest judges coverage tests by.
//!
//! The capture mirrors `rustc_codegen_llvm::coverageinfo::mapgen`
//! exactly: the used records come from the crate's mono items,
//! the unused records from the same eligibility/usage sweep
//! mapgen performs, mapping spans are resolved with the same
//! coordinate rules (`mapgen::spans::make_coords`), and the
//! dump rendering re-encodes each record with LLVM's
//! `CoverageMappingWriter` algorithm (region sort, expression
//! minimisation, ULEB128 payload) and prints it the way the
//! `coverage-dump` tool prints the decoded payload. Every value
//! in the table is the compiler's own; nothing is estimated.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use rustc_middle::mir::coverage as rcov;
use rustc_middle::mir::visit::Visitor as _;
use rustc_middle::mono::MonoItem;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::source_map::SourceMap;
use rustc_span::{FileName, Pos, RemapPathScopeComponents, SourceFile, Span};
use sparagmos_mir as sm;

/// Whether this compilation was asked to instrument coverage:
/// the gate that keeps the capture sweep (and its per-instance
/// MIR queries) out of every ordinary conversion.
pub fn instrument_coverage_enabled(tcx: TyCtxt) -> bool {
    tcx.sess.opts.cg.instrument_coverage
        != rustc_session::config::InstrumentCoverage::No
}

/// Collects every `FnDef` definition mentioned in a body
/// (call operands, function-pointer constants, ...): the
/// mention is evidence that the mentioning body's crate uses
/// the definition in its own code.
#[derive(Default)]
struct FnDefMentions {
    defs: Vec<rustc_span::def_id::DefId>,
}

impl<'tcx> rustc_middle::mir::visit::Visitor<'tcx> for FnDefMentions {
    fn visit_ty(
        &mut self,
        ty: ty::Ty<'tcx>,
        _: rustc_middle::mir::visit::TyContext,
    ) {
        if let ty::TyKind::FnDef(def_id, _) = ty.kind() {
            if !self.defs.contains(def_id) {
                self.defs.push(*def_id);
            }
        }
        self.super_ty(ty);
    }
}

/// Capture the coverage mapping table of the crate being
/// converted. `converted` maps a mono instance's mangled symbol
/// name to the program keys of the bodies it was converted
/// under (an instance can be converted under more than one key;
/// executing any of them feeds the same record's counters).
pub fn capture<'tcx>(
    tcx: TyCtxt<'tcx>,
    converted: &HashMap<String, Vec<String>>,
    instances: &HashMap<u32, rustc_public::mir::mono::Instance>,
    all_body_names: &HashSet<String>,
) -> sm::CoverageMapping {
    // `-Zcoverage-options=discard-all-spans-in-codegen`: this
    // crate's codegen discards every coverage span, so it
    // contributes no records of its own (the test-only flag
    // behind issue 133606). Records of foreign crates merged
    // at link time are unaffected.
    let discard_all =
        tcx.sess.coverage_options().discard_all_spans_in_codegen;

    let partitions = tcx.collect_and_partition_mono_items(());
    let mut mono_instances: Vec<ty::Instance<'tcx>> = Vec::new();
    for cgu in partitions.codegen_units {
        for item in cgu.items().keys() {
            if let MonoItem::Fn(instance) = item {
                mono_instances.push(*instance);
            }
        }
    }

    // The converted instances in internal form. The sweeps
    // below reach beyond this crate's mono items through them:
    // auxiliary-crate bodies are converted too, and the
    // coverage statements in them (and the instances inlined
    // into them) belong to records the linked binary carries.
    let converted_instances: Vec<ty::Instance<'tcx>> = instances
        .values()
        .filter_map(|public| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                rustc_public::rustc_internal::internal(tcx, public)
            }))
            .ok()
        })
        .collect();

    // The usage sets of mapgen's unused-function sweep.
    let mut used_via_inlining: HashSet<rustc_span::def_id::DefId> = HashSet::new();
    let mut missing_own_coverage: HashSet<rustc_span::def_id::DefId> =
        HashSet::new();
    if !discard_all {
        let mut defs_seen: HashSet<rustc_span::def_id::DefId> = HashSet::new();
        for instance in &mono_instances {
            if !defs_seen.insert(instance.def_id()) {
                continue;
            }
            let body = tcx.instance_mir(instance.def);
            let mut saw_own_coverage = false;
            for block in body.basic_blocks.iter() {
                for stmt in &block.statements {
                    if let rustc_middle::mir::StatementKind::Coverage(_) =
                        stmt.kind
                    {
                        if let Some(inlined) = stmt
                            .source_info
                            .scope
                            .inlined_instance(&body.source_scopes)
                        {
                            used_via_inlining.insert(inlined.def_id());
                        } else {
                            saw_own_coverage = true;
                        }
                    }
                }
            }
            if !saw_own_coverage && body.coverage_mir_info.is_some() {
                missing_own_coverage.insert(instance.def_id());
            }
        }
    }

    // (record, its source file's table string). All of a
    // record's mappings share the one source file (see
    // `prepare`), so the string travels with the record until
    // the global table order is fixed below.
    // The used set of the native mapping writer is not the
    // mono partition itself: during codegen, an instance is
    // marked as used when a counter increment is emitted for
    // it, and coverage statements that were inlined into a
    // codegenned body are attributed to the instance they were
    // inlined from (rustc_codegen_ssa's `codegen_coverage`).
    // An instance is therefore used exactly when some
    // codegenned body's body holds a VirtualCounter statement
    // attributed to it whose BCB was assigned a physical
    // counter. The bodies walked are this crate's mono items
    // and every converted body (auxiliary crates' bodies are
    // codegenned by their own compilations, whose coverage
    // statements mark their own instances the same way).
    // Home-crate usage evidence for the foreign unused sweep
    // below: (crate, def) pairs where some walked body of that
    // crate uses the definition -- a coverage statement
    // attributed to an instantiation of it through inlining, or
    // an `FnDef` mention (a call, a function pointer) in the
    // body. A foreign definition its home crate's own code uses
    // is a mono item of the home compilation, so the home
    // mapgen emits no unused record for it; the sweep must not
    // invent one either. Usage by another crate's bodies does
    // not count (the pair is keyed by the using body's crate),
    // and a body's mention of its own definition does not
    // count (a converted instantiation is not home usage).
    let mut home_used: HashSet<(
        rustc_span::def_id::CrateNum,
        rustc_span::def_id::DefId,
    )> = HashSet::new();
    let mut marked: Vec<ty::Instance<'tcx>> = Vec::new();
    {
        let mut marked_set: HashSet<ty::Instance<'tcx>> = HashSet::new();
        let mut bodies_seen: HashSet<ty::Instance<'tcx>> = HashSet::new();
        // Only bodies that native codegen processes are
        // walked: this crate's mono items, and converted
        // foreign bodies (an auxiliary crate's own codegen
        // processes the bodies the main compilation demanded
        // from it). Converted local bodies are NOT walked:
        // conversion converts every local body, including
        // never-codegenned ones (a dead function, an async
        // future that is created but never polled), and their
        // statements must not mark anything used.
        for instance in mono_instances.iter().chain(
            converted_instances
                .iter()
                .filter(|i| !i.def_id().is_local()),
        ) {
            if discard_all && instance.def_id().is_local() {
                continue;
            }
            if !bodies_seen.insert(*instance) {
                continue;
            }
            if !matches!(
                instance.def,
                ty::InstanceKind::Item(_) | ty::InstanceKind::Shim(..)
            ) {
                continue;
            }
            // A converted instance need not have a body at all
            // (an extern function, say); asking for its MIR
            // panics, so the walk is guarded per body, as in
            // `fill_coverage_owners`.
            let walked = std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| {
                    let mut found: Vec<ty::Instance<'tcx>> = Vec::new();
                    let mut attributed_all: Vec<ty::Instance<'tcx>> =
                        Vec::new();
                    let body = tcx.instance_mir(instance.def);
                    for block in body.basic_blocks.iter() {
                        for stmt in &block.statements {
                            let rustc_middle::mir::StatementKind::Coverage(
                                rustc_middle::mir::coverage::CoverageKind::VirtualCounter {
                                    bcb,
                                },
                            ) = stmt.kind
                            else {
                                continue;
                            };
                            let attributed = match stmt
                                .source_info
                                .scope
                                .inlined_instance(&body.source_scopes)
                            {
                                Some(inlined) => instance
                                    .instantiate_mir_and_normalize_erasing_regions(
                                        tcx,
                                        ty::TypingEnv::fully_monomorphized(),
                                        rustc_middle::ty::EarlyBinder::bind(tcx, inlined),
                                    ),
                                None => *instance,
                            };
                            if !matches!(
                                attributed.def,
                                ty::InstanceKind::Item(_)
                            ) {
                                continue;
                            }
                            attributed_all.push(attributed);
                            let has_counter = tcx
                                .coverage_codegen_info(attributed.def)
                                .map(|info| {
                                    info.phys_counter_for_node
                                        .contains_key(&bcb)
                                })
                                .unwrap_or(false);
                            if has_counter {
                                found.push(attributed);
                            }
                        }
                    }
                    let mut mentions = FnDefMentions::default();
                    mentions.visit_body(body);
                    (found, attributed_all, mentions.defs)
                }),
            );
            if let Ok((found, attributed_all, mentioned)) = walked {
                let walker_crate = instance.def_id().krate;
                for attributed in &attributed_all {
                    if *attributed != *instance {
                        home_used
                            .insert((walker_crate, attributed.def_id()));
                    }
                }
                for def in mentioned {
                    if def != instance.def_id() {
                        home_used.insert((walker_crate, def));
                    }
                }
                for attributed in found {
                    if marked_set.insert(attributed) {
                        marked.push(attributed);
                    }
                }
            }
        }
    }
    // A marked instance's record is part of this crate's own
    // mapping dump when the instance is monomorphised here or
    // is a local definition (mapgen of this crate emits it);
    // a foreign instance that is not monomorphised here has
    // its record only in the linked binary's mapping.
    let binary_only_of = |instance: &ty::Instance<'tcx>| {
        !instance.def_id().is_local() && !mono_instances.contains(instance)
    };
    let mut used: Vec<(sm::CovFunction, String)> = Vec::new();
    let mut binary_crates: HashSet<rustc_span::def_id::CrateNum> =
        HashSet::new();
    for instance in &marked {
        if let Some((mut record, file_string)) =
            prepare(tcx, *instance, true, converted)
        {
            record.binary_only = binary_only_of(instance);
            if record.binary_only {
                binary_crates.insert(instance.def_id().krate);
            }
            used.push((record, file_string));
        }
    }
    let mut unused: Vec<(sm::CovFunction, String)> = Vec::new();
    if !discard_all {
        for &local_def in tcx.mir_keys(()).iter() {
            if !tcx.is_eligible_for_coverage(local_def) {
                continue;
            }
            let def: rustc_span::def_id::DefId = local_def.to_def_id();
            let is_unused = (!partitions.all_mono_items.contains(&def)
                || missing_own_coverage.contains(&def))
                && !used_via_inlining.contains(&def);
            if !is_unused {
                continue;
            }
            let dummy = make_dummy_instance(tcx, local_def);
            if let Some(record) = prepare(tcx, dummy, false, converted) {
                unused.push(record);
            }
        }
    }

    // Foreign instances that were converted (demanded by the
    // main compilation) but are neither monomorphised here nor
    // marked above: their records were emitted by their home
    // crate's codegen and merged into the executable, so the
    // coverage-run report includes them (as binary-only
    // records). Instances whose home crate was compiled
    // without coverage instrumentation have no coverage info
    // and yield no record.
    let mut recorded: HashSet<String> =
        used.iter().chain(&unused).map(|(r, _)| r.symbol.clone()).collect();
    for instance in &converted_instances {
        if instance.def_id().is_local() {
            continue;
        }
        if !matches!(instance.def, ty::InstanceKind::Item(_)) {
            continue;
        }
        let symbol = tcx.symbol_name(*instance).name.to_string();
        if !recorded.insert(symbol) {
            continue;
        }
        let prepared = std::panic::catch_unwind(
            std::panic::AssertUnwindSafe(|| {
                prepare(tcx, *instance, true, converted)
            }),
        )
        .ok()
        .flatten();
        if let Some((mut record, file_string)) = prepared {
            record.binary_only = binary_only_of(instance);
            if record.binary_only {
                binary_crates.insert(instance.def_id().krate);
            }
            used.push((record, file_string));
        }
    }

    // A foreign crate that contributed binary-only records was
    // itself compiled with coverage instrumentation (an
    // auxiliary build of a coverage test), so its eligible
    // functions that were never demanded also have records in
    // the binary, as unused ones. Enumerate those crates'
    // function definitions through the metadata item tree.
    for krate in binary_crates {
        let root = rustc_span::def_id::DefId {
            krate,
            index: rustc_span::def_id::CRATE_DEF_INDEX,
        };
        let mut stack = vec![root];
        let mut visited: HashSet<rustc_span::def_id::DefId> = HashSet::new();
        let mut candidates: Vec<rustc_span::def_id::DefId> = Vec::new();
        while let Some(def) = stack.pop() {
            if !visited.insert(def) {
                continue;
            }
            for child in tcx.module_children(def).iter() {
                let rustc_hir::def::Res::Def(kind, child_def) = child.res
                else {
                    continue;
                };
                match kind {
                    rustc_hir::def::DefKind::Mod => stack.push(child_def),
                    rustc_hir::def::DefKind::Fn => {
                        candidates.push(child_def)
                    }
                    rustc_hir::def::DefKind::Impl { .. }
                    | rustc_hir::def::DefKind::Trait => {
                        for assoc in
                            tcx.associated_item_def_ids(child_def).iter()
                        {
                            if matches!(
                                tcx.def_kind(*assoc),
                                rustc_hir::def::DefKind::Fn
                            ) {
                                candidates.push(*assoc);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        for def in candidates {
            // The home crate's own code uses this definition,
            // so its home compilation monomorphised it and
            // emitted no unused record (see `home_used`).
            if home_used.contains(&(krate, def)) {
                continue;
            }
            // (Eligibility is local-only as a query; a foreign
            // definition without coverage info simply yields
            // no record from prepare.)
            let dummy = ty::Instance::new_raw(
                def,
                ty::GenericArgs::for_item(tcx, def, |param, _| {
                    if let ty::GenericParamDefKind::Lifetime = param.kind {
                        tcx.lifetimes.re_erased.into()
                    } else {
                        tcx.mk_param_from_def(param)
                    }
                }),
            );
            let symbol = tcx.symbol_name(dummy).name.to_string();
            if !recorded.insert(symbol) {
                continue;
            }
            let prepared = std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| {
                    prepare(tcx, dummy, false, converted)
                }),
            )
            .ok()
            .flatten();
            if let Some((mut record, file_string)) = prepared {
                record.binary_only = true;
                used.push((record, file_string));
            }
        }
    }

    // The global file table is built over this crate's used
    // records in symbol-name order, then the unused records in
    // symbol-name order (mapgen sorts each codegen unit's used
    // records by symbol name, and appends the dead-code unit's
    // unused records, also symbol-sorted; the coverage
    // compilations the harness performs are single-unit, so
    // this is the same order). Binary-only records come last:
    // their files join the table after this crate's own, so
    // the dump's file ids are this crate's alone.
    let (mut used_own, mut used_binary): (
        Vec<(sm::CovFunction, String)>,
        Vec<(sm::CovFunction, String)>,
    ) = (Vec::new(), Vec::new());
    for entry in used {
        if entry.0.binary_only {
            used_binary.push(entry);
        } else {
            used_own.push(entry);
        }
    }
    used_own.sort_by(|a, b| a.0.symbol.cmp(&b.0.symbol));
    used_binary.sort_by(|a, b| a.0.symbol.cmp(&b.0.symbol));
    unused.sort_by(|a, b| a.0.symbol.cmp(&b.0.symbol));
    let source_map = tcx.sess.source_map();
    let mut files: Vec<String> = vec![source_map
        .working_dir()
        .path(RemapPathScopeComponents::COVERAGE)
        .to_string_lossy()
        .into_owned()];
    let mut file_ids: HashMap<String, u32> = HashMap::new();
    let mut records: Vec<sm::CovFunction> = Vec::new();
    for (mut record, file_string) in
        used_own.into_iter().chain(unused).chain(used_binary)
    {
        let global = match file_ids.get(&file_string) {
            Some(&id) => id,
            None => {
                let id = files.len() as u32;
                files.push(file_string.clone());
                file_ids.insert(file_string, id);
                id
            }
        };
        for mapping in &mut record.mappings {
            mapping.file = global;
        }
        records.push(record);
    }

    // A record's statements may live in a converted body the
    // converter registered under a different symbol than the
    // record's: a coroutine's record is captured from its
    // `Future::poll` instance, but the body carrying the
    // statements is the coroutine body itself, whose converted
    // name is the record's own (demangled) name. Link such
    // bodies by name, so execution resolves their statements
    // against this record.
    {
        for record in &mut records {
            if record.is_used
                && all_body_names.contains(&record.name)
                && !record.body_names.contains(&record.name)
            {
                let name = record.name.clone();
                record.body_names.push(name);
            }
        }
    }

    sm::CoverageMapping { files, functions: records }
}

fn file_table_string(file: &SourceFile) -> String {
    if let FileName::Real(real) = &file.name {
        let (_work_dir, abs_name) =
            real.embeddable_name(RemapPathScopeComponents::COVERAGE);
        abs_name.to_string_lossy().into_owned()
    } else {
        file.name
            .display(RemapPathScopeComponents::COVERAGE)
            .to_string_lossy()
            .into_owned()
    }
}

/// One function's record, prepared the way mapgen's
/// `prepare_covfun_record` prepares it: `None` when the instance
/// carries no coverage info, has no codegen info, or loses all
/// of its mapping spans to the coordinate rules. On success,
/// also returns the record's source-file table string; the
/// mappings' `file` fields are placeholders that `capture`
/// replaces with global file ids.
fn prepare<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::Instance<'tcx>,
    is_used: bool,
    converted: &HashMap<String, Vec<String>>,
) -> Option<(sm::CovFunction, String)> {
    // Only real items get records: shim instances (the drop
    // glue of a coroutine is the observed case) can carry a
    // body whose coverage info is the underlying item's, but
    // the native mapping writer never emits records for them.
    if !matches!(instance.def, ty::InstanceKind::Item(_)) {
        return None;
    }
    let body = tcx.instance_mir(instance.def);
    let mir_info = body.coverage_mir_info.as_deref()?;
    let cg_info = tcx.coverage_codegen_info(instance.def)?;

    let source_map = tcx.sess.source_map();
    let first_span = mir_info.mappings.first()?.span;
    let source_file = source_map.lookup_source_file(first_span.lo());

    let mut mappings: Vec<sm::CovMapping> = Vec::new();
    for mapping in &mir_info.mappings {
        let Some(coords) = make_coords(source_map, &source_file, mapping.span)
        else {
            continue;
        };
        let kind = match mapping.kind {
            rcov::MappingKind::Code { bcb } => {
                sm::CovMappingKind::Code { bcb: bcb.as_u32() }
            }
            rcov::MappingKind::Branch { true_bcb, false_bcb } => {
                sm::CovMappingKind::Branch {
                    true_bcb: true_bcb.as_u32(),
                    false_bcb: false_bcb.as_u32(),
                }
            }
        };
        mappings.push(sm::CovMapping {
            kind,
            file: 0,
            start_line: coords.0,
            start_col: coords.1,
            end_line: coords.2,
            end_col: coords.3,
        });
    }
    if mappings.is_empty() {
        return None;
    }

    let expressions: Vec<sm::CovExpression> = cg_info
        .expressions
        .iter()
        .map(|expr| sm::CovExpression {
            lhs: cov_term(expr.lhs),
            subtract: matches!(expr.op, rcov::Op::Subtract),
            rhs: cov_term(expr.rhs),
        })
        .collect();
    let term_for_bcb: Vec<Option<sm::CovTerm>> =
        cg_info.term_for_bcb.iter().map(|t| t.map(cov_term)).collect();
    let mut counter_for_bcb: Vec<Option<u32>> =
        vec![None; cg_info.term_for_bcb.len()];
    for (bcb, counter) in &cg_info.phys_counter_for_node {
        if let Some(slot) = counter_for_bcb.get_mut(bcb.as_usize()) {
            *slot = Some(counter.as_u32());
        }
    }

    let symbol = tcx.symbol_name(instance).name.to_string();
    let name = match rustc_demangle::try_demangle(&symbol) {
        Ok(demangled) => format!("{demangled:#}"),
        Err(_) => format!("(couldn't demangle) {symbol}"),
    };
    let body_names = if is_used {
        converted.get(&symbol).cloned().unwrap_or_default()
    } else {
        Vec::new()
    };
    let file_string = file_table_string(&source_file);
    Some((
        sm::CovFunction {
            symbol,
            name,
            is_used,
            binary_only: false,
            body_names,
            expressions,
            mappings,
            term_for_bcb,
            counter_for_bcb,
            num_counters: cg_info.num_counters,
        },
        file_string,
    ))
}

fn cov_term(term: rcov::CovTerm) -> sm::CovTerm {
    match term {
        rcov::CovTerm::Zero => sm::CovTerm::Zero,
        rcov::CovTerm::Counter(id) => sm::CovTerm::Counter(id.as_u32()),
        rcov::CovTerm::Expression(id) => sm::CovTerm::Expression(id.as_u32()),
    }
}

/// The dummy instance mapgen builds for an unused function:
/// generic parameters filled with the parameters themselves.
fn make_dummy_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    local_def: rustc_span::def_id::LocalDefId,
) -> ty::Instance<'tcx> {
    let def_id = local_def.to_def_id();
    ty::Instance::new_raw(
        def_id,
        ty::GenericArgs::for_item(tcx, def_id, |param, _| {
            if let ty::GenericParamDefKind::Lifetime = param.kind {
                tcx.lifetimes.re_erased.into()
            } else {
                tcx.mk_param_from_def(param)
            }
        }),
    )
}

/// Line/byte-column coordinates of a span within its file,
/// computed exactly as mapgen's `make_coords` computes them
/// (byte columns, doctest line offset, and the validity screen
/// that discards regions llvm-cov would reject).
/// Returns (start_line, start_col, end_line, end_col), 1-based.
fn make_coords(
    source_map: &SourceMap,
    file: &Arc<SourceFile>,
    span: Span,
) -> Option<(u32, u32, u32, u32)> {
    if span.is_empty() {
        return None;
    }
    let line_and_byte_column = |pos: rustc_span::BytePos| -> Option<(usize, usize)> {
        let rpos = file.relative_position(pos);
        let line_index = file.lookup_line(rpos)?;
        let line_start = file.lines()[line_index];
        Some((line_index + 1, (rpos - line_start).to_usize() + 1))
    };
    let (start_line, start_col) = line_and_byte_column(span.lo())?;
    let (end_line, end_col) = line_and_byte_column(span.hi())?;
    let start_line = source_map.doctest_offset_line(&file.name, start_line);
    let end_line = source_map.doctest_offset_line(&file.name, end_line);
    let coords = (
        start_line as u32,
        start_col as u32,
        end_line as u32,
        end_col as u32,
    );
    let (start_line, start_col, end_line, end_col) = coords;
    let all_nonzero = [start_line, start_col, end_line, end_col]
        .into_iter()
        .all(|x| x != 0);
    let end_col_has_high_bit_unset = (end_col & (1 << 31)) == 0;
    let is_ordered = (start_line, start_col) <= (end_line, end_col);
    if all_nonzero && end_col_has_high_bit_unset && is_ordered {
        Some(coords)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// The coverage-mapping dump (compiletest's coverage-map mode)
// ---------------------------------------------------------------------------

/// A term in decoded-payload form: an expression term carries
/// the operator of the expression it names, as the mapping
/// format encodes it and the dump prints it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PTerm {
    Zero,
    Counter(u32),
    Expression(u32, bool), // (id, subtract)
}

/// One mapping region in writer form: the terms are the
/// record's BCB terms (zero for unused records), ready for the
/// writer's sort, minimisation and encoding.
#[derive(Clone, Copy, Debug)]
struct WRegion {
    is_branch: bool,
    count: sm::CovTerm,
    false_count: sm::CovTerm,
    start_line: u32,
    start_col: u32,
    end_line: u32,
    end_col: u32,
}

/// The decoded form of one function's encoded record: the
/// minimised expressions and the sorted regions, with all terms
/// adjusted to the minimised expression ids.
struct DecodedRecord {
    expressions: Vec<(PTerm, PTerm)>,
    /// (is_branch, count term, false term, start_line_offset,
    /// start_col, end_line_offset, end_col) per region, in
    /// payload order.
    regions: Vec<(bool, PTerm, PTerm, u32, u32, u32, u32)>,
    payload: Vec<u8>,
}

fn pterm(term: sm::CovTerm, expressions: &[sm::CovExpression]) -> PTerm {
    match term {
        sm::CovTerm::Zero => PTerm::Zero,
        sm::CovTerm::Counter(id) => PTerm::Counter(id),
        sm::CovTerm::Expression(id) => {
            let subtract = expressions
                .get(id as usize)
                .map(|e| e.subtract)
                .unwrap_or(false);
            PTerm::Expression(id, subtract)
        }
    }
}

fn encode_term(term: PTerm) -> u32 {
    match term {
        PTerm::Zero => 0,
        PTerm::Counter(id) => (id << 2) | 1,
        PTerm::Expression(id, subtract) => {
            (id << 2) | if subtract { 2 } else { 3 }
        }
    }
}

fn write_uleb(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Encode one function's record the way LLVM's
/// `CoverageMappingWriter` encodes it (the algorithm of
/// `CoverageMappingWriter::write`): regions are stable-sorted
/// by (file, start location, kind, end location descending),
/// expressions are minimised to the ones the sorted regions
/// use, and the payload is the ULEB128 sequence the format
/// specifies. All of a record's regions share local file 0, so
/// the sort keys reduce to location/kind/end.
fn encode_record(
    function: &sm::CovFunction,
    global_file_id: u32,
) -> DecodedRecord {
    let mut regions: Vec<WRegion> = Vec::new();
    for mapping in &function.mappings {
        match mapping.kind {
            sm::CovMappingKind::Code { bcb } => regions.push(WRegion {
                is_branch: false,
                count: function.term_of(bcb),
                false_count: sm::CovTerm::Zero,
                start_line: mapping.start_line,
                start_col: mapping.start_col,
                end_line: mapping.end_line,
                end_col: mapping.end_col,
            }),
            sm::CovMappingKind::Branch { true_bcb, false_bcb } => {
                regions.push(WRegion {
                    is_branch: true,
                    count: function.term_of(true_bcb),
                    false_count: function.term_of(false_bcb),
                    start_line: mapping.start_line,
                    start_col: mapping.start_col,
                    end_line: mapping.end_line,
                    end_col: mapping.end_col,
                })
            }
        }
    }
    // The writer's region sort: ascending start location; at
    // the same start, code regions before branch regions
    // (kind keys 0 and 8 in the writer's numbering); then
    // wider regions (later end) first. The sort is stable, so
    // full ties keep the input order (code regions in mapping
    // order, then branch regions in mapping order).
    regions.sort_by(|a, b| {
        (a.start_line, a.start_col)
            .cmp(&(b.start_line, b.start_col))
            .then_with(|| (a.is_branch as u32).cmp(&(b.is_branch as u32)))
            .then_with(|| (b.end_line, b.end_col).cmp(&(a.end_line, a.end_col)))
    });

    // CounterExpressionsMinimizer, transcribed literally: mark
    // the expressions reachable from the sorted regions' terms,
    // then gather them in region order (an expression is
    // gathered the first time the walk reaches it; its adjusted
    // id is its position in the gathered list).
    let expressions = &function.expressions;
    let mut adjusted: Vec<u32> = vec![0; expressions.len()];
    fn mark(
        term: sm::CovTerm,
        expressions: &[sm::CovExpression],
        adjusted: &mut [u32],
    ) {
        if let sm::CovTerm::Expression(id) = term {
            if adjusted[id as usize] != 0 {
                return;
            }
            adjusted[id as usize] = 1;
            let expr = expressions[id as usize];
            mark(expr.lhs, expressions, adjusted);
            mark(expr.rhs, expressions, adjusted);
        }
    }
    for region in &regions {
        mark(region.count, expressions, &mut adjusted);
        mark(region.false_count, expressions, &mut adjusted);
    }
    let mut used_expressions: Vec<sm::CovExpression> = Vec::new();
    fn gather(
        term: sm::CovTerm,
        expressions: &[sm::CovExpression],
        adjusted: &mut [u32],
        used: &mut Vec<sm::CovExpression>,
    ) {
        if let sm::CovTerm::Expression(id) = term {
            if adjusted[id as usize] == 0 {
                return;
            }
            adjusted[id as usize] = used.len() as u32;
            let expr = expressions[id as usize];
            used.push(expr);
            gather(expr.lhs, expressions, adjusted, used);
            gather(expr.rhs, expressions, adjusted, used);
        }
    }
    for region in &regions {
        gather(region.count, expressions, &mut adjusted, &mut used_expressions);
        gather(region.false_count, expressions, &mut adjusted, &mut used_expressions);
    }
    let adjust = |term: sm::CovTerm| -> sm::CovTerm {
        match term {
            sm::CovTerm::Expression(id) => {
                sm::CovTerm::Expression(adjusted[id as usize])
            }
            other => other,
        }
    };
    let min_expressions: Vec<(PTerm, PTerm)> = used_expressions
        .iter()
        .map(|expr| {
            (
                pterm(adjust(expr.lhs), &used_expressions),
                pterm(adjust(expr.rhs), &used_expressions),
            )
        })
        .collect();

    // The payload bytes.
    let mut payload: Vec<u8> = Vec::new();
    write_uleb(&mut payload, 1); // one virtual file
    write_uleb(&mut payload, global_file_id as u64);
    write_uleb(&mut payload, min_expressions.len() as u64);
    for (lhs, rhs) in &min_expressions {
        write_uleb(&mut payload, encode_term(*lhs) as u64);
        write_uleb(&mut payload, encode_term(*rhs) as u64);
    }
    write_uleb(&mut payload, regions.len() as u64);
    let mut prev_line = 0u32;
    let mut decoded_regions = Vec::new();
    for region in &regions {
        let count = pterm(adjust(region.count), &used_expressions);
        let false_count = pterm(adjust(region.false_count), &used_expressions);
        if region.is_branch {
            write_uleb(&mut payload, 4 << 3);
            write_uleb(&mut payload, encode_term(count) as u64);
            write_uleb(&mut payload, encode_term(false_count) as u64);
        } else {
            write_uleb(&mut payload, encode_term(count) as u64);
        }
        let start_offset = region.start_line - prev_line;
        let end_offset = region.end_line - region.start_line;
        write_uleb(&mut payload, start_offset as u64);
        write_uleb(&mut payload, region.start_col as u64);
        write_uleb(&mut payload, end_offset as u64);
        write_uleb(&mut payload, region.end_col as u64);
        prev_line = region.start_line;
        decoded_regions.push((
            region.is_branch,
            count,
            false_count,
            start_offset,
            region.start_col,
            end_offset,
            region.end_col,
        ));
    }

    DecodedRecord { expressions: min_expressions, regions: decoded_regions, payload }
}

fn format_pterm(term: PTerm) -> String {
    match term {
        PTerm::Zero => "Zero".to_string(),
        PTerm::Counter(id) => format!("Counter({id})"),
        PTerm::Expression(id, subtract) => {
            format!("Expression({id}, {})", if subtract { "Sub" } else { "Add" })
        }
    }
}

fn format_resolved(term: PTerm, expressions: &[(PTerm, PTerm)]) -> String {
    match term {
        PTerm::Zero => "Zero".to_string(),
        PTerm::Counter(id) => format!("c{id}"),
        PTerm::Expression(id, subtract) => {
            let (lhs, rhs) = expressions[id as usize];
            format!(
                "({} {} {})",
                format_resolved(lhs, expressions),
                if subtract { "-" } else { "+" },
                format_resolved(rhs, expressions)
            )
        }
    }
}

/// Render the coverage-mapping dump of a whole table, in the
/// `coverage-dump` tool's format: entries sorted by demangled
/// name (then used-before-unused in the tool's bool order --
/// unused entries first -- then payload bytes), each decoded
/// and printed in the tool's layout.
pub fn render_dump(table: &sm::CoverageMapping) -> String {
    struct Entry {
        name: String,
        is_used: bool,
        global_file_id: u32,
        decoded: DecodedRecord,
    }
    // Binary-only records are not in this crate's IR, so the
    // dump of it does not contain them.
    let mut entries: Vec<Entry> = table
        .functions
        .iter()
        .filter(|function| !function.binary_only)
        .map(|function| {
            let global_file_id =
                function.mappings.first().map(|m| m.file).unwrap_or(0);
            Entry {
                name: function.name.clone(),
                is_used: function.is_used,
                global_file_id,
                decoded: encode_record(function, global_file_id),
            }
        })
        .collect();
    entries.sort_by(|a, b| {
        a.name
            .cmp(&b.name)
            .then_with(|| a.is_used.cmp(&b.is_used))
            .then_with(|| a.decoded.payload.cmp(&b.decoded.payload))
    });

    let mut out = String::new();
    for entry in &entries {
        let decoded = &entry.decoded;
        out.push_str(&format!(
            "Function name: {}{}\n",
            entry.name,
            if entry.is_used { "" } else { " (unused)" }
        ));
        out.push_str(&format!(
            "Raw bytes ({}): 0x{:02x?}\n",
            decoded.payload.len(),
            decoded.payload.as_slice()
        ));
        out.push_str("Number of files: 1\n");
        let filename = table
            .files
            .get(entry.global_file_id as usize)
            .map(String::as_str)
            .unwrap_or("");
        out.push_str(&format!("- file 0 => {filename}\n"));
        out.push_str(&format!(
            "Number of expressions: {}\n",
            decoded.expressions.len()
        ));
        for (i, (lhs, rhs)) in decoded.expressions.iter().enumerate() {
            out.push_str(&format!(
                "- expression {i} operands: lhs = {}, rhs = {}\n",
                format_pterm(*lhs),
                format_pterm(*rhs)
            ));
        }
        out.push_str(&format!(
            "Number of file 0 mappings: {}\n",
            decoded.regions.len()
        ));
        let mut max_counter: Option<u32> = None;
        for (is_branch, count, false_count, slo, sc, elo, ec) in &decoded.regions {
            for term in [count, false_count] {
                if let PTerm::Counter(id) = term {
                    max_counter = max_counter.max(Some(*id));
                }
            }
            if *is_branch {
                out.push_str(&format!(
                    "- Branch {{ true: {}, false: {} }} at (prev + {slo}, {sc}) to (start + {elo}, {ec})\n",
                    format_pterm(*count),
                    format_pterm(*false_count)
                ));
                out.push_str(&format!(
                    "    true  = {}\n",
                    format_resolved(*count, &decoded.expressions)
                ));
                out.push_str(&format!(
                    "    false = {}\n",
                    format_resolved(*false_count, &decoded.expressions)
                ));
            } else {
                out.push_str(&format!(
                    "- Code({}) at (prev + {slo}, {sc}) to (start + {elo}, {ec})\n",
                    format_pterm(*count)
                ));
                if let PTerm::Expression(..) = count {
                    out.push_str(&format!(
                        "    = {}\n",
                        format_resolved(*count, &decoded.expressions)
                    ));
                }
            }
        }
        out.push_str(&format!(
            "Highest counter ID seen: {}\n",
            match max_counter {
                Some(id) => format!("c{id}"),
                None => "(none)".to_string(),
            }
        ));
        out.push('\n');
    }
    out
}
// ---------------------------------------------------------------------------
// The llvm-cov text report (compiletest's coverage-run mode)
// ---------------------------------------------------------------------------

/// One counted code region of one file, as llvm-cov's segment
/// builder consumes regions: endpoints are (line, column).
#[derive(Clone, Copy)]
struct CountedRegion {
    start: (u32, u32),
    end: (u32, u32),
    count: u64,
}

/// One counted branch region: the mapping span and both arms'
/// evaluated counts.
#[derive(Clone, Copy)]
struct CountedBranch {
    start: (u32, u32),
    true_count: u64,
    false_count: u64,
}

/// One coverage segment (llvm-cov's `CoverageSegment`): from
/// this (line, column) on, the given count applies.
#[derive(Clone, Copy)]
struct Segment {
    line: u32,
    col: u32,
    count: u64,
    has_count: bool,
    is_entry: bool,
}

/// llvm-cov's `SegmentBuilder::startSegment`, for the code
/// regions rustc emits (never gap regions; a segment is count-less
/// only when emitted as a skipped region).
fn start_segment(
    segments: &mut Vec<Segment>,
    count: u64,
    loc: (u32, u32),
    is_entry: bool,
    emit_skipped: bool,
) {
    let has_count = !emit_skipped;
    if !segments.is_empty() && !is_entry && !emit_skipped {
        let last = segments.last().unwrap();
        if last.has_count == has_count && last.count == count && !last.is_entry {
            return;
        }
    }
    segments.push(Segment {
        line: loc.0,
        col: loc.1,
        count: if has_count { count } else { 0 },
        has_count,
        is_entry,
    });
}

/// llvm-cov's `SegmentBuilder::completeRegionsUntil`: `active`
/// holds region indices, its suffix from `first` is the group of
/// regions that have just completed.
fn complete_regions_until(
    active: &mut Vec<usize>,
    regions: &[CountedRegion],
    segments: &mut Vec<Segment>,
    loc: Option<(u32, u32)>,
    first: usize,
) {
    active[first..].sort_by_key(|&i| regions[i].end);
    let mut i = first + 1;
    while i < active.len() {
        let mut completed = active[i];
        let completed_segment_loc = regions[active[i - 1]].end;
        if loc == Some(completed_segment_loc) {
            break;
        }
        if completed_segment_loc == regions[completed].end {
            i += 1;
            continue;
        }
        // Use the count of the last completed region that ends
        // where this one ends.
        for j in i + 1..active.len() {
            if regions[completed].end == regions[active[j]].end {
                completed = active[j];
            }
        }
        start_segment(segments, regions[completed].count, completed_segment_loc, false, false);
        i += 1;
    }
    let last = *active.last().unwrap();
    if first != 0 {
        if loc != Some(regions[last].end) {
            let parent = active[first - 1];
            start_segment(segments, regions[parent].count, regions[last].end, false, false);
        }
    } else if loc.is_none() || loc != Some(regions[last].end) {
        start_segment(segments, regions[last].count, regions[last].end, false, true);
    }
    active.drain(first..);
}

/// llvm-cov's `SegmentBuilder::buildSegments`: sort the file's
/// regions into nesting order, merge regions covering the same
/// area (their counts add up), and sweep them into segments.
fn build_segments(regions_in: &[CountedRegion]) -> Vec<Segment> {
    let mut regions: Vec<CountedRegion> = regions_in.to_vec();
    // sortNestedRegions: by start, with containing regions first.
    regions.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
    // combineRegions.
    let mut combined: Vec<CountedRegion> = Vec::new();
    for region in regions {
        if let Some(last) = combined.last_mut() {
            if last.start == region.start && last.end == region.end {
                last.count += region.count;
                continue;
            }
        }
        combined.push(region);
    }
    let regions = combined;

    let mut segments: Vec<Segment> = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    for (index, region) in regions.iter().enumerate() {
        let cur_start = region.start;
        // Regions that end at or before this start complete first
        // (a stable partition moves them behind the still-active
        // ones, preserving order within both parts).
        if let Some(first) = active.iter().position(|&i| regions[i].end <= cur_start) {
            let (kept, done): (Vec<usize>, Vec<usize>) = active
                .iter()
                .partition(|&&i| regions[i].end > cur_start);
            active = kept.into_iter().chain(done).collect();
            complete_regions_until(&mut active, &regions, &mut segments, Some(cur_start), first);
        }
        if cur_start == region.end {
            // A zero-length region never becomes active; as the
            // last region it leaves a skipped segment behind.
            let skipped = index + 1 == regions.len();
            let source = active.last().copied().unwrap_or(index);
            start_segment(&mut segments, regions[source].count, cur_start, true, skipped);
            if skipped && !active.is_empty() {
                let back = *active.last().unwrap();
                start_segment(&mut segments, regions[back].count, cur_start, false, false);
            }
            continue;
        }
        if index + 1 == regions.len() || cur_start != regions[index + 1].start {
            start_segment(&mut segments, region.count, cur_start, true, false);
        }
        active.push(index);
    }
    if !active.is_empty() {
        complete_regions_until(&mut active, &regions, &mut segments, None, 0);
    }
    segments
}

/// llvm-cov's `SourceCoverageView::formatCount`: engineering
/// notation with three significant digits (`1234` -> `1.23k`).
fn format_count(n: u64) -> String {
    let digits = n.to_string();
    let len = digits.len();
    if len <= 3 {
        return digits;
    }
    let int_len = if len % 3 == 0 { 3 } else { len % 3 };
    let mut out = digits[..int_len].to_string();
    if int_len != 3 {
        out.push('.');
        out += &digits[int_len..int_len + (3 - int_len)];
    }
    out.push(" kMGTPEZY".as_bytes()[(len - 1) / 3] as char);
    out
}

/// Render the llvm-cov style text report of a table under one
/// run's counted counter values (`counts` is parallel to
/// `table.functions`). The rendering follows llvm-cov's
/// `show --format=text --show-line-counts-or-regions` layout as
/// compiletest's normalisation expects it (real line numbers;
/// the harness anonymises them). Source text is read from the
/// files the mappings name, as llvm-cov does.
/// One instantiation subview of a file section: the per-record
/// view llvm-cov attaches under the line where a multiply
/// instantiated function ends.
struct InstantiationView {
    name: String,
    executed: bool,
    segments: Vec<Segment>,
    branch_groups: Vec<(u32, Vec<CountedBranch>)>,
    first_line: u32,
    last_line: u32,
}

/// Group counted branches by line the way llvm-cov's
/// `attachBranchSubViews` does: consecutive same-line branches
/// form one view, and the views print in line order.
fn branch_groups(branches: &[CountedBranch]) -> Vec<(u32, Vec<CountedBranch>)> {
    let mut groups: Vec<(u32, Vec<CountedBranch>)> = Vec::new();
    for branch in branches {
        match groups.last_mut() {
            Some((line, group)) if *line == branch.start.0 => group.push(*branch),
            _ => groups.push((branch.start.0, vec![*branch])),
        }
    }
    groups.sort_by_key(|(line, _)| *line);
    groups
}

/// A branch subview label, wrapped in the red-background
/// escape when colours are on and the count is zero (llvm-cov
/// colours the label of a never-taken branch direction).
fn branch_label(label: &str, count: u64, color: bool) -> String {
    if color && count == 0 {
        format!("\u{1b}[0;41m{label}\u{1b}[0m")
    } else {
        label.to_string()
    }
}

/// Render one source line's text with llvm-cov's zero-count
/// highlighting (the `--use-color` form): the spans between
/// segment boundaries are wrapped in the red-background escape
/// while the current highlight is red, which it becomes after a
/// segment whose count is zero (and starts as red when the
/// wrapped segment from the previous line has count zero).
/// Spans are byte ranges, matching LLVM's byte columns; an
/// escape pair is emitted even for an empty span, as LLVM's
/// coloured stream does.
fn render_line_text(
    out: &mut String,
    text: &str,
    line_segments: &[Segment],
    wrapped: Option<Segment>,
) {
    let bytes = text.as_bytes();
    let mut highlight =
        matches!(wrapped, Some(w) if w.has_count && w.count == 0);
    let mut push_span = |out: &mut String, span: &[u8], highlight: bool| {
        if highlight {
            *out += "\u{1b}[0;41m";
        }
        *out += &String::from_utf8_lossy(span);
        if highlight {
            *out += "\u{1b}[0m";
        }
    };
    let mut col = 1usize;
    for s in line_segments {
        let end = (s.col as usize).min(bytes.len() + 1);
        let span = if end >= col {
            &bytes[col - 1..end - 1]
        } else {
            &bytes[0..0]
        };
        push_span(out, span, highlight);
        col = end;
        highlight = s.has_count && s.count == 0;
    }
    let rest = if col <= bytes.len() {
        &bytes[col - 1..]
    } else {
        &bytes[0..0]
    };
    push_span(out, rest, highlight);
}

/// Render one view's lines (a whole file section, or one
/// instantiation subview) into `out`. `prefix` is the view's
/// line prefix (`"  |"` per nesting depth). Subviews of a line
/// print in llvm-cov's order -- instantiations, then branch
/// groups -- each led by a divider, with one closing divider
/// after the last.
#[allow(clippy::too_many_arguments)]
fn render_view_lines(
    out: &mut String,
    prefix: &str,
    lines: &[&str],
    first_line: u32,
    last_line: u32,
    segments: &[Segment],
    groups: &[(u32, Vec<CountedBranch>)],
    instantiations: &[(u32, InstantiationView)],
    color: bool,
) {
    let divider = format!("{prefix}  {}", "-".repeat(18));
    let mut wrapped: Option<Segment> = None;
    let mut seg_at = segments
        .iter()
        .position(|s| s.line >= first_line)
        .unwrap_or(segments.len());
    for line_no in first_line..=last_line {
        let mut line_segments: Vec<Segment> = Vec::new();
        while seg_at < segments.len() && segments[seg_at].line == line_no {
            line_segments.push(segments[seg_at]);
            seg_at += 1;
        }
        // LineCoverageStats for this line.
        let is_entry_counted = |s: &Segment| s.has_count && s.is_entry;
        let region_entries =
            line_segments.iter().filter(|s| is_entry_counted(s)).count();
        let start_of_skipped = line_segments
            .first()
            .map(|s| !s.has_count && s.is_entry)
            .unwrap_or(false);
        let mut mapped = !start_of_skipped
            && (wrapped.map(|w| w.has_count).unwrap_or(false) || region_entries > 0);
        mapped |= line_segments.iter().any(|s| s.is_entry && s.has_count);
        let mut execution = 0u64;
        if mapped {
            if let Some(w) = wrapped {
                execution = w.count;
            }
            if region_entries > 0 {
                for s in &line_segments {
                    if is_entry_counted(s) {
                        execution = execution.max(s.count);
                    }
                }
            }
        }

        *out += prefix;
        *out += &format!("{:>5}|", line_no);
        if mapped {
            let digits = format_count(execution);
            if color && region_entries >= 2 {
                // llvm-cov colours the count magenta when the
                // line has multiple regions; the padding stays
                // plain.
                *out += &" ".repeat(7 - digits.len());
                *out += "\u{1b}[0;35m";
                *out += &digits;
                *out += "\u{1b}[0m|";
            } else {
                *out += &format!("{:>7}|", digits);
            }
        } else {
            *out += "       |";
        }
        let text = lines[(line_no - 1) as usize];
        let text = text.strip_suffix('\r').unwrap_or(text);
        if color {
            render_line_text(out, text, &line_segments, wrapped);
        } else {
            *out += text;
        }
        *out += "\n";

        // Region markers for sub-line regions whose count
        // differs from the line's.
        // llvm-cov considers all but the last segment for
        // markers, and a line with a single segment never has
        // any (its shouldRenderRegionMarkers loop is empty).
        let marker_segments: &[Segment] = if line_segments.len() > 1 {
            &line_segments[..line_segments.len() - 1]
        } else {
            &line_segments[..0]
        };
        let show_markers = marker_segments
            .iter()
            .any(|s| s.is_entry && s.has_count && s.count != execution);
        if show_markers {
            *out += prefix;
            *out += &" ".repeat(14);
            let mut prev_col = 1u32;
            for s in marker_segments {
                if !s.is_entry || s.count == execution {
                    continue;
                }
                if s.col > prev_col {
                    *out += &" ".repeat((s.col - prev_col) as usize);
                }
                prev_col = s.col + 1;
                let c = format_count(s.count);
                prev_col += c.len() as u32;
                out.push('^');
                *out += &c;
            }
            *out += "\n";
        }

        // Subviews attached to this line: instantiations first,
        // then branch groups, each led by a divider, with one
        // closing divider after the last.
        let mut rendered_subview = false;
        for (line, view) in instantiations {
            if *line != line_no {
                continue;
            }
            *out += &divider;
            *out += "\n";
            if view.executed {
                *out += prefix;
                *out += "  | ";
                if color {
                    *out += "\u{1b}[0;36m";
                }
                *out += &view.name;
                *out += ":\n";
                if color {
                    *out += "\u{1b}[0m";
                }
                let sub_prefix = format!("{prefix}  |");
                render_view_lines(
                    out,
                    &sub_prefix,
                    lines,
                    view.first_line,
                    view.last_line,
                    &view.segments,
                    &view.branch_groups,
                    &[],
                    color,
                );
            } else {
                *out += prefix;
                *out += "  | ";
                if color {
                    *out += "\u{1b}[0;31m";
                }
                *out += "Unexecuted instantiation: ";
                *out += &view.name;
                *out += "\n";
                if color {
                    *out += "\u{1b}[0m";
                }
            }
            rendered_subview = true;
        }
        for (line, group) in groups {
            if *line != line_no {
                continue;
            }
            *out += &divider;
            *out += "\n";
            for branch in group {
                *out += prefix;
                *out += &format!(
                    "  |  Branch ({}:{}): [{}: {}, {}: {}]\n",
                    branch.start.0,
                    branch.start.1,
                    branch_label("True", branch.true_count, color),
                    format_count(branch.true_count),
                    branch_label("False", branch.false_count, color),
                    format_count(branch.false_count)
                );
            }
            rendered_subview = true;
        }
        if rendered_subview {
            *out += &divider;
            *out += "\n";
        }

        if !line_segments.is_empty() {
            wrapped = line_segments.last().copied();
        }
    }
}

pub fn render_report(
    table: &sm::CoverageMapping,
    counts: &[Vec<u64>],
    color: bool,
) -> String {
    // Counted regions and branches, per source file and per
    // record.
    let mut code_by_file: Vec<Vec<CountedRegion>> =
        vec![Vec::new(); table.files.len()];
    let mut branch_by_file: Vec<Vec<CountedBranch>> =
        vec![Vec::new(); table.files.len()];
    let mut code_by_record: Vec<Vec<(usize, CountedRegion)>> =
        vec![Vec::new(); table.functions.len()];
    let mut branch_by_record: Vec<Vec<(usize, CountedBranch)>> =
        vec![Vec::new(); table.functions.len()];
    for (index, function) in table.functions.iter().enumerate() {
        let empty: &[u64] = &[];
        let values = counts.get(index).map(Vec::as_slice).unwrap_or(empty);
        for mapping in &function.mappings {
            let file = mapping.file as usize;
            match mapping.kind {
                sm::CovMappingKind::Code { bcb } => {
                    let region = CountedRegion {
                        start: (mapping.start_line, mapping.start_col),
                        end: (mapping.end_line, mapping.end_col),
                        count: function.eval_term(function.term_of(bcb), values),
                    };
                    code_by_file[file].push(region);
                    code_by_record[index].push((file, region));
                }
                sm::CovMappingKind::Branch { true_bcb, false_bcb } => {
                    let branch = CountedBranch {
                        start: (mapping.start_line, mapping.start_col),
                        true_count: function
                            .eval_term(function.term_of(true_bcb), values),
                        false_count: function
                            .eval_term(function.term_of(false_bcb), values),
                    };
                    branch_by_file[file].push(branch);
                    branch_by_record[index].push((file, branch));
                }
            }
        }
    }

    // The files llvm-cov would show: every file named by some
    // record, in sorted order (its getUniqueSourceFiles).
    let mut files: Vec<usize> = (0..table.files.len())
        .filter(|&f| !code_by_file[f].is_empty() || !branch_by_file[f].is_empty())
        .collect();
    files.sort_by(|&a, &b| table.files[a].cmp(&table.files[b]));

    let show_names = files.len() != 1;
    let mut out = String::new();
    for file in files {
        let filename = &table.files[file];
        if show_names {
            if color {
                out += "\u{1b}[0;36m";
            }
            out += filename;
            out += ":\n";
            if color {
                out += "\u{1b}[0m";
            }
        }
        let segments = build_segments(&code_by_file[file]);
        let groups = branch_groups(&branch_by_file[file]);

        // llvm-cov skips a file whose source it cannot read
        // (e.g. a path rewritten by --remap-path-prefix): it
        // contributes no section at all.
        let Ok(source) = std::fs::read_to_string(filename) else {
            continue;
        };
        let lines: Vec<&str> = source.split('\n').collect();
        // split() leaves a phantom empty line after a trailing
        // newline; llvm-cov's line iterator ends at the newline.
        let line_count = if source.ends_with('\n') || source.is_empty() {
            lines.len().saturating_sub(1)
        } else {
            lines.len()
        };

        // Instantiation groups of this file: records grouped by
        // the start of their first counted region in the file
        // (llvm-cov's FunctionInstantiationSetCollector). A group
        // of two or more yields one subview per record, attached
        // at the record's last line in the file; a record that
        // never executed renders as an unexecuted-instantiation
        // line instead of a body.
        let mut by_start: std::collections::BTreeMap<(u32, u32), Vec<usize>> =
            std::collections::BTreeMap::new();
        for index in 0..table.functions.len() {
            let mut starts: Vec<(u32, u32)> = code_by_record[index]
                .iter()
                .filter(|(f, _)| *f == file)
                .map(|(_, r)| r.start)
                .chain(
                    branch_by_record[index]
                        .iter()
                        .filter(|(f, _)| *f == file)
                        .map(|(_, b)| b.start),
                )
                .collect();
            if starts.is_empty() {
                continue;
            }
            starts.sort();
            by_start.entry(starts[0]).or_default().push(index);
        }
        let mut instantiations: Vec<(u32, InstantiationView)> = Vec::new();
        for members in by_start.values() {
            if members.len() < 2 {
                continue;
            }
            for &index in members {
                let function = &table.functions[index];
                let regions: Vec<CountedRegion> = code_by_record[index]
                    .iter()
                    .filter(|(f, _)| *f == file)
                    .map(|(_, r)| *r)
                    .collect();
                let branches: Vec<CountedBranch> = branch_by_record[index]
                    .iter()
                    .filter(|(f, _)| *f == file)
                    .map(|(_, b)| *b)
                    .collect();
                let first_line =
                    regions.iter().map(|r| r.start.0).min().unwrap_or(0);
                let last_line = regions.iter().map(|r| r.end.0).max().unwrap_or(0);
                // The record's execution count is its entry
                // region's count: the first code region in
                // counted-region order.
                let mut ordered = regions.clone();
                ordered.sort_by(|a, b| {
                    a.start
                        .cmp(&b.start)
                        .then(b.end.0.cmp(&a.end.0))
                        .then(b.end.1.cmp(&a.end.1))
                });
                let executed =
                    ordered.first().map(|r| r.count > 0).unwrap_or(false);
                instantiations.push((
                    last_line,
                    InstantiationView {
                        name: function.name.clone(),
                        executed,
                        segments: build_segments(&regions),
                        branch_groups: branch_groups(&branches),
                        first_line,
                        last_line,
                    },
                ));
            }
        }
        instantiations.sort_by_key(|(line, _)| *line);

        render_view_lines(
            &mut out,
            "",
            &lines,
            1,
            line_count as u32,
            &segments,
            &groups,
            &instantiations,
            color,
        );
        out += "\n";
    }
    out
}

/// Attribute every converted body's coverage statements to the
/// instance they were inlined from, filling
/// `Body::coverage_owners`.
///
/// Stable MIR's source scopes do not name the inlined-from
/// instance, so the attribution is read from the internal MIR:
/// for the instance each body was converted under, the internal
/// body is walked in parallel with the converted body, and each
/// coverage statement's owner is its scope's inlined instance
/// (the same resolution codegen performs when it emits the
/// counter increment). The walk is only trusted when the two
/// bodies' coverage statements correspond exactly (same count,
/// same BCB sequence); a body that fails the check keeps an
/// empty table and its statements fall back to executing-body
/// attribution.
pub fn fill_coverage_owners<'tcx>(
    tcx: TyCtxt<'tcx>,
    instances: &HashMap<u32, rustc_public::mir::mono::Instance>,
    bodies: &mut [(String, sm::Body)],
    body_def_ids: &[sm::DefId],
) {
    for (pos, (_name, body)) in bodies.iter_mut().enumerate() {
        let Some(def_id) = body_def_ids.get(pos) else {
            continue;
        };
        let Some(public_instance) = instances.get(&def_id.index) else {
            continue;
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            attribute_body(tcx, public_instance, body)
        }));
        if let Ok(Some(owners)) = result {
            body.coverage_owners = owners;
        }
    }
}

/// The owner table of one converted body, or `None` when the
/// converted and internal coverage statements do not correspond
/// exactly.
fn attribute_body<'tcx>(
    tcx: TyCtxt<'tcx>,
    public_instance: &rustc_public::mir::mono::Instance,
    body: &sm::Body,
) -> Option<Vec<Option<String>>> {
    let instance = rustc_public::rustc_internal::internal(tcx, public_instance);
    // Only item and shim instances have instance MIR; asking
    // for an intrinsic's (or a vtable/reify shim's) is an
    // internal compiler error, and those bodies' statements
    // keep the executing-body fallback.
    if !matches!(
        instance.def,
        ty::InstanceKind::Item(_) | ty::InstanceKind::Shim(..)
    ) {
        return None;
    }
    let ibody = tcx.instance_mir(instance.def);
    let mut internal: Vec<(u32, Option<String>)> = Vec::new();
    for bb in ibody.basic_blocks.iter() {
        for stmt in &bb.statements {
            if let rustc_middle::mir::StatementKind::Coverage(
                rcov::CoverageKind::VirtualCounter { bcb },
            ) = &stmt.kind
            {
                // The scope's inlined instance is stated in
                // terms of this body's own generic parameters;
                // instantiate it with this instance, exactly as
                // the capture's marking walk instantiates it,
                // so the owner symbol is the record's symbol
                // (the raw instance's symbol names the
                // parameters instead of the arguments, and no
                // record carries it).
                let owner = stmt
                    .source_info
                    .scope
                    .inlined_instance(&ibody.source_scopes)
                    .map(|inlined| {
                        let attributed = instance
                            .instantiate_mir_and_normalize_erasing_regions(
                                tcx,
                                ty::TypingEnv::fully_monomorphized(),
                                rustc_middle::ty::EarlyBinder::bind(
                                    tcx, inlined,
                                ),
                            );
                        tcx.symbol_name(attributed).name.to_string()
                    });
                internal.push((bcb.as_u32(), owner));
            }
        }
    }
    let converted: Vec<u32> = body
        .basic_blocks
        .iter()
        .flat_map(|b| b.statements.iter())
        .filter_map(|s| match s.kind {
            sm::StatementKind::Coverage(bcb) => Some(bcb),
            _ => None,
        })
        .collect();
    if converted.len() != internal.len() {
        return None;
    }
    if converted
        .iter()
        .zip(&internal)
        .any(|(a, (b, _))| a != b)
    {
        return None;
    }
    Some(internal.into_iter().map(|(_, owner)| owner).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code_mapping(bcb: u32, sl: u32, sc: u32, el: u32, ec: u32) -> sm::CovMapping {
        sm::CovMapping {
            kind: sm::CovMappingKind::Code { bcb },
            file: 1,
            start_line: sl,
            start_col: sc,
            end_line: el,
            end_col: ec,
        }
    }

    fn function(symbol: &str, name: &str, mappings: Vec<sm::CovMapping>) -> sm::CovFunction {
        sm::CovFunction {
            symbol: symbol.to_string(),
            name: name.to_string(),
            is_used: true,
            binary_only: false,
            body_names: Vec::new(),
            expressions: Vec::new(),
            mappings,
            term_for_bcb: vec![Some(sm::CovTerm::Counter(0))],
            counter_for_bcb: vec![Some(0)],
            num_counters: 1,
        }
    }

    #[test]
    fn format_count_uses_engineering_notation() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(42), "42");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1000), "1.00k");
        assert_eq!(format_count(1234), "1.23k");
        assert_eq!(format_count(1_500_000), "1.50M");
    }

    /// Two instantiations of one generic function: the main
    /// view sums their counts, and each gets its own subview
    /// under the function's last line (llvm-cov's instantiation
    /// views).
    #[test]
    fn report_renders_instantiation_subviews() {
        let dir = std::env::temp_dir().join(format!(
            "sparagmos-cov-report-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("g.rs");
        std::fs::write(
            &src,
            "fn g<T>(x: T) -> T { x }\nfn main() { g::<u32>(1); g::<f32>(2.0); }\n",
        )
        .unwrap();
        let table = sm::CoverageMapping {
            files: vec![
                "/workdir".to_string(),
                src.to_string_lossy().into_owned(),
            ],
            functions: vec![
                function("g_u32", "test::g::<u32>", vec![code_mapping(0, 1, 1, 1, 25)]),
                function("g_f32", "test::g::<f32>", vec![code_mapping(0, 1, 1, 1, 25)]),
                function("main", "test::main", vec![code_mapping(0, 2, 1, 2, 40)]),
            ],
        };
        let counts = vec![vec![1], vec![2], vec![1]];
        let report = render_report(&table, &counts, false);
        // Main view: the generic's line sums both instantiations.
        assert!(
            report.contains("    1|      3|fn g<T>(x: T) -> T { x }"),
            "main view sums instantiations:\n{report}"
        );
        // Subviews for each instantiation, with its own count.
        assert!(
            report.contains("  | test::g::<u32>:"),
            "u32 subview header:\n{report}"
        );
        assert!(
            report.contains("  | test::g::<f32>:"),
            "f32 subview header:\n{report}"
        );
        assert!(
            report.contains("  |    1|      1|fn g<T>(x: T) -> T { x }"),
            "u32 subview body:\n{report}"
        );
        assert!(
            report.contains("  |    1|      2|fn g<T>(x: T) -> T { x }"),
            "f32 subview body:\n{report}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
