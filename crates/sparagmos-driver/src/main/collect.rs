//! The compiler run itself: the rustc callbacks and the top-level collection pass.

use super::*;

/// The callbacks driving the conversion's compiler run: the
/// `run_with_tcx!` callback shape, written out so the run can
/// also configure the session the way the real rustc entry point
/// does. rustc's own entry enters through `TimePassesCallbacks`,
/// whose `config` sets `trimmed_def_paths`; type printing
/// consults that flag when the compiler itself prints
/// (`-Zunpretty` dumps, `--print` answers), so without it a
/// printed Stable MIR body names `std::vec::Vec` where rustc
/// prints `Vec`. The flag is set exactly for those printing
/// invocations; every other run keeps the session the
/// rustc_public callbacks would have built.
struct ConversionCallbacks {
    result: Option<std::ops::ControlFlow<()>>,
}

impl ConversionCallbacks {
    fn run(&mut self, args: &[String]) -> Result<(), rustc_public::CompilerError<()>> {
        let compiler_result =
            rustc_driver::catch_fatal_errors(|| -> rustc_interface::interface::Result<()> {
                rustc_driver::run_compiler(args, self);
                Ok(())
            });
        match (compiler_result, self.result.take()) {
            (Ok(Ok(())), Some(std::ops::ControlFlow::Continue(value))) => Ok(value),
            // `collect` breaks the flow when the verdict chain
            // has rejected the compilation: the results are
            // stashed and the session was stopped before the
            // phases past analysis, so the run completed --
            // the caller reports the verdict from the stash.
            (Ok(Ok(())), Some(std::ops::ControlFlow::Break(value))) => Ok(value),
            (Ok(Ok(_)), None) => Err(rustc_public::CompilerError::Skipped),
            (Ok(Err(_)), _) | (Err(_), _) => Err(rustc_public::CompilerError::Failed),
        }
    }
}

impl rustc_driver::Callbacks for ConversionCallbacks {
    fn config(&mut self, config: &mut rustc_interface::interface::Config) {
        if config.opts.pretty.is_some() || !config.opts.prints.is_empty() {
            config.opts.trimmed_def_paths = true;
        }
    }

    fn after_analysis<'tcx>(
        &mut self,
        _compiler: &rustc_interface::interface::Compiler,
        tcx: rustc_middle::ty::TyCtxt<'tcx>,
    ) -> rustc_driver::Compilation {
        rustc_public::rustc_internal::run(tcx, || {
            self.result = Some(collect_with_tcx(tcx));
        })
        .unwrap();
        if self.result.as_ref().is_some_and(|val| val.is_continue()) {
            rustc_driver::Compilation::Continue
        } else {
            rustc_driver::Compilation::Stop
        }
    }
}

/// Run the compiler over `args` and convert what it analyses
/// (see `collect_with_tcx`), under the outcome contract
/// `rustc_public::run_with_tcx!` provided.
pub(crate) fn run_conversion_compiler(args: &[String]) -> Result<(), rustc_public::CompilerError<()>> {
    ConversionCallbacks { result: None }.run(args)
}

/// The compiler callback, in the `run_with_tcx` form: the
/// context is installed for the conversion's span resolution
/// (see `sparagmos_driver::convert::set_compiler_tcx`) and
/// cleared on the way out, including on a panic.
fn collect_with_tcx(tcx: rustc_middle::ty::TyCtxt) -> ControlFlow<()> {
    struct ClearTcx;
    impl Drop for ClearTcx {
        fn drop(&mut self) {
            sparagmos_driver::convert::set_compiler_tcx(None);
        }
    }
    sparagmos_driver::convert::set_compiler_tcx(Some(tcx));
    let _clear = ClearTcx;
    collect()
}

fn collect() -> ControlFlow<()> {
    let mut conv = Converter::new();
    let mut report = ConversionReport::default();
    let mut bodies: Vec<(String, Body)> = Vec::new();
    let mut body_def_ids: Vec<sparagmos_mir::DefId> = Vec::new();
    let mut body_def_names: Vec<String> = Vec::new();
    let local_defs: std::collections::HashSet<rustc_public::DefId> =
        rustc_public::all_local_items()
            .iter()
            .map(|i| i.def_id())
            .collect();
    // Stable MIR bodies of the converted local functions; the
    // reachability collection walks their call graph afterwards.
    let mut seed_bodies: Vec<rustc_public::mir::Body> = Vec::new();

    // Generic function definitions deferred by the item loop; the
    // instance collection after the loop accounts for them.
    let mut deferred: Vec<DeferredFn> = Vec::new();
    // Closures defined in generic contexts, deferred the same way;
    // the closure collection after the loop accounts for them.
    let mut deferred_closures: Vec<DeferredFn> = Vec::new();

    let items = rustc_public::all_local_items();
    // The first body fetch of the item loop or the
    // reachability collection whose whole-body normalization
    // overflowed the trait solver (the demand passes record
    // theirs on the demand state's own channel): the compiler
    // itself cannot materialize that instance, so the
    // compilation fails with the recursion-limit error rustc's
    // collector reports, folded into the compile verdict
    // below.
    let mut body_fetch_recursion_limit: Option<String> = None;
    // Record the local statics' declared types before converting
    // anything: a reference into a static's interior is decoded
    // against the static's own type (see Converter::decode_static_ref),
    // and the type query is only safe for real items — an anonymous
    // promoted static's type_of can ICE the compilation — so it is
    // asked here, for items, never from the constant decoder.
    for item in items.iter() {
        if matches!(item.kind(), ItemKind::Static) {
            if let Ok(ty) =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| item.ty()))
            {
                conv.note_static_ty(item, ty);
            }
        }
    }
    for item in items.iter() {
        let name = item.name();
        let item_instance = Instance::try_from(item.clone());
        // Assign (or fetch) this item's Sparagmos DefId up front, so a
        // converted body can be handed to the interpreter together with
        // the id that other bodies use to call it. Function items are
        // keyed by their resolved instance (the same key `FnDef` call
        // targets resolve through); other items keep the definition's
        // own id.
        let mut def_id = match &item_instance {
            Ok(inst) => conv.instance_def_id(item.def_id(), inst),
            Err(_) => conv.crate_def_id(item),
        };
        let instance = match item_instance {
            Ok(i) => i,
            Err(_) => match closure_instance(item) {
                ClosureAttempt::Instance(i) => {
                    // A closure body is keyed by its instance, like
                    // a function item: references to the closure
                    // type resolve through `closure_def_id` to the
                    // same key, and two instances of one closure
                    // definition stay distinct bodies.
                    def_id = conv.instance_def_id(item.def_id(), &i);
                    i
                }
                ClosureAttempt::GenericContext => {
                    // A closure defined inside a generic function,
                    // method or impl: no instance as an item, but
                    // monomorphic instances of the enclosing
                    // definition mention it with concrete arguments.
                    // Defer it like a generic function: a placeholder
                    // report entry keeps the target-crate accounting
                    // in place, and the closure collection after
                    // this loop fills in the outcome.
                    let mut reasons = BTreeMap::new();
                    reasons.insert(
                        "closure is defined in a generic context (its generic arguments \
                         still mention parameters); only monomorphic closures are converted"
                            .to_string(),
                        1,
                    );
                    deferred_closures.push(DeferredFn {
                        def: item.def_id(),
                        name: name.clone(),
                        report_idx: report.functions.len(),
                        ctor: false,
                    });
                    report.functions.push(FnReport {
                        name,
                        blocks: 0,
                        locals: 0,
                        arg_count: 0,
                        outcome: FnOutcome::Skipped { reasons },
                    });
                    continue;
                }
                ClosureAttempt::CoroutineClosure => {
                    // An async closure: its type cannot be asked
                    // (see `closure_instance`), so it has no
                    // instance as an item, but a converted body
                    // that creates it mentions it with concrete
                    // arguments in the coroutine-closure
                    // aggregate. Defer it like a generic-context
                    // closure: a placeholder report entry keeps
                    // the target-crate accounting in place, and
                    // the closure collection after this loop fills
                    // in the outcome.
                    let mut reasons = BTreeMap::new();
                    reasons.insert(COROUTINE_CLOSURE_TYPE_REASON.to_string(), 1);
                    deferred_closures.push(DeferredFn {
                        def: item.def_id(),
                        name: name.clone(),
                        report_idx: report.functions.len(),
                        ctor: false,
                    });
                    report.functions.push(FnReport {
                        name,
                        blocks: 0,
                        locals: 0,
                        arg_count: 0,
                        outcome: FnOutcome::Skipped { reasons },
                    });
                    continue;
                }
                ClosureAttempt::NotClosure => {
                    if matches!(item.kind(), ItemKind::Fn)
                        || matches!(
                            item.kind(),
                            ItemKind::Ctor(rustc_public::CtorKind::Fn)
                        )
                    {
                        // A generic function definition (or a
                        // generic data constructor): it has no
                        // monomorphic instance as an item, but its
                        // call sites may demand concrete instances.
                        // Defer it: a placeholder report entry keeps
                        // the target-crate accounting in place, and
                        // the instance collection after this loop
                        // fills in the outcome.
                        let mut reasons = BTreeMap::new();
                        reasons.insert(
                            format!(
                                "no monomorphic instance available (kind={:?})",
                                item.kind()
                            ),
                            1,
                        );
                        deferred.push(DeferredFn {
                            def: item.def_id(),
                            name: name.clone(),
                            report_idx: report.functions.len(),
                            ctor: matches!(item.kind(), ItemKind::Ctor(..)),
                        });
                        report.functions.push(FnReport {
                            name,
                            blocks: 0,
                            locals: 0,
                            arg_count: 0,
                            outcome: FnOutcome::Skipped { reasons },
                        });
                        continue;
                    }
                    if matches!(item.kind(), ItemKind::Const) {
                        // A constant that requires monomorphization
                        // as an item (an anonymous constant inside a
                        // generic function, an associated constant
                        // of a generic impl, ...): it has no
                        // instance, but the constant's own MIR body
                        // can still be converted, the way the
                        // demanded instances of a generic function
                        // are. Monomorphized bodies that use the
                        // constant carry its evaluated value, so the
                        // converted body is registered under the
                        // constant's own definition id — the key
                        // unevaluated constant references resolve
                        // through. A constant whose own body is
                        // unavailable too is a trivial constant
                        // (see `account_trivial_const`).
                        match fetch_query_normalize_aware(&item.name(), false, || item.body()) {
                            Ok(Some(body)) => {
                                let report_idx = report.functions.len();
                                // A constant whose body mentions
                                // a projection that does not
                                // normalize: resolving the
                                // associated calls inside its
                                // conversion would pose that
                                // normalization and poison the
                                // compilation (see
                                // `unnormalizable_projection_in_body`).
                                // Skip the conversion with the
                                // named reason; the accounting
                                // below re-accounts it not
                                // demanded, like a constant that
                                // fails only on its own
                                // parameters —
                                // monomorphized bodies that use
                                // the constant carry its
                                // evaluated value, so no
                                // converted body needs this
                                // definition body.
                                if let Some(reason) =
                                    sparagmos_driver::convert::unnormalizable_projection_in_body(&body)
                                {
                                    skip_fn(&mut report, &name, reason);
                                } else {
                                convert_local_body(
                                    &mut conv,
                                    &mut report,
                                    &mut bodies,
                                    &mut body_def_ids,
                                    &mut body_def_names,
                                    &mut seed_bodies,
                                    name,
                                    def_id,
                                    body,
                                );
                                }
                                // A conversion that fails only on
                                // the constant's own parameters has
                                // met the generic definition itself,
                                // not a monomorphic body: a
                                // `ConstantKind::Param` in the body
                                // means the body's constants still
                                // mention the parameters of the
                                // definition (or of its enclosing
                                // item), which only substitution by
                                // monomorphization removes — and this
                                // constant has no instance as an item
                                // (that is why it reached this arm).
                                // Monomorphized bodies that use the
                                // constant carry its evaluated value,
                                // so no converted body needs this
                                // definition body, the way a generic
                                // function definition nothing demands
                                // is not part of the program either.
                                // The projection screen's reason
                                // is the same shape: the
                                // definition's own types mention a
                                // projection no monomorphization
                                // could normalize, so nothing can
                                // demand the definition either.
                                // Account it `NotDemanded`, keeping
                                // the parameter record; a failure
                                // with any other reason mixed in
                                // stays skipped, as a genuine gap.
                                if let Some(entry) =
                                    report.functions.get_mut(report_idx)
                                {
                                    let param_reasons = match &entry.outcome {
                                        FnOutcome::Skipped { reasons }
                                            if !reasons.is_empty()
                                                && reasons.keys().all(|r| {
                                                    r.starts_with(
                                                        "ConstantKind::Param(",
                                                    ) || r
                                                        == "ConstantKind::Bound"
                                                        || r.starts_with(
                                                            sparagmos_driver::convert::UNNORMALIZABLE_PROJECTION_REASON,
                                                        )
                                                }) =>
                                        {
                                            Some(
                                                reasons
                                                    .iter()
                                                    .map(|(k, v)| {
                                                        format!("{k} x{v}")
                                                    })
                                                    .collect::<Vec<_>>()
                                                    .join(", "),
                                            )
                                        }
                                        _ => None,
                                    };
                                    if let Some(param_reasons) = param_reasons {
                                        entry.outcome =
                                            FnOutcome::NotDemanded {
                                                reason: format!(
                                                    "generic constant definition                                                      ({param_reasons}); only                                                      monomorphic instances are                                                      converted, and                                                      monomorphized bodies that                                                      use the constant carry                                                      its evaluated value"
                                                ),
                                            };
                                    }
                                }
                            }
                            Ok(None) => account_trivial_const(&mut report, name),
                            Err(reason) => skip_fn(&mut report, &name, reason),
                        }
                        continue;
                    }
                    skip_fn(
                        &mut report,
                        &name,
                        format!("no monomorphic instance available (kind={:?})", item.kind()),
                    );
                    continue;
                }
                ClosureAttempt::Unsupported(reason) => {
                    skip_fn(&mut report, &name, reason);
                    continue;
                }
            },
        };
        let screen_fetch = fetch_query_normalize_aware(
            &name,
            sparagmos_driver::convert::generated_simd_impl_method(&instance),
            || item.body(),
        );
        if let Err(reason) = &screen_fetch {
            if reason
                == sparagmos_driver::convert::BODY_FETCH_NORMALIZE_OVERFLOW
            {
                note_body_fetch_recursion_limit(
                    &mut body_fetch_recursion_limit,
                    &instance,
                );
                skip_fn(
                    &mut report,
                    &name,
                    sparagmos_driver::convert::BODY_FETCH_NORMALIZE_OVERFLOW
                        .to_string(),
                );
                continue;
            }
            if reason == sparagmos_driver::convert::BODY_FETCH_SIMD_VALIDATOR {
                // The fetch tripped the MIR validator's ban on
                // projecting into a SIMD type (MCP#838): the
                // body — a compiler-generated impl body whose
                // statements project the ADT's fields, like a
                // derived `PartialEq::eq` — cannot be
                // materialized at all. This is the class the
                // demand passes screen out up front
                // (`convert::generated_simd_impl_method`); here the fetch
                // itself is the precise test, so bodies without
                // a banned projection (a derived `Clone` that
                // copies the value whole) still convert above.
                skip_fn(
                    &mut report,
                    &name,
                    sparagmos_driver::convert::BODY_FETCH_SIMD_VALIDATOR
                        .to_string(),
                );
                continue;
            }
        }
        // A definition whose types mention a projection that
        // does not normalize in a monomorphized environment:
        // fetching its instance body would pose exactly that
        // normalization and poison the compilation (a delayed
        // bug, uncontainable — see
        // `unnormalizable_projection_in_instance`). rustc's
        // monomorphization collector never instantiates such
        // a definition either (no compilable program can
        // demand it), so defer it like a generic definition
        // and let the demand accounting below decide: an
        // undemanded constructor is accounted converted, any
        // other undemanded definition not demanded, and a
        // demanded one fails by name at the demand fetch.
        // The screen covers the instance's signature and
        // arguments; the definition body (fetched without
        // normalization) extends it to the body's locals and
        // call targets.
        let unnormalizable =
            sparagmos_driver::convert::unnormalizable_projection_in_instance(
                &instance,
            )
            .or_else(|| match &screen_fetch {
                Ok(Some(def_body)) => {
                    sparagmos_driver::convert::unnormalizable_projection_in_body(
                        def_body,
                    )
                }
                _ => None,
            });
        if let Some(reason) = unnormalizable {
            let mut reasons = BTreeMap::new();
            reasons.insert(reason, 1);
            deferred.push(DeferredFn {
                def: item.def_id(),
                name: name.clone(),
                report_idx: report.functions.len(),
                ctor: matches!(item.kind(), ItemKind::Ctor(..)),
            });
            report.functions.push(FnReport {
                name,
                blocks: 0,
                locals: 0,
                arg_count: 0,
                outcome: FnOutcome::Skipped { reasons },
            });
            continue;
        }
        let body = match fetch_query_normalize_aware(
            &instance.name(),
            sparagmos_driver::convert::generated_simd_impl_method(&instance),
            || instance.body(),
        ) {
            Ok(Some(body)) => body,
            Ok(None) => {
                if matches!(item.kind(), ItemKind::Const) {
                    // The constant has an instance but no body: a
                    // trivial constant (see `account_trivial_const`).
                    account_trivial_const(&mut report, name);
                    continue;
                }
                // An extern-item-impl (EII) function declaration
                // has no body of its own; the registered
                // implementation's body, instantiated with the
                // declaration's own arguments, is its body (see
                // `convert::eii_impl_instance`). Fetch that body
                // and convert it under the declaration's name,
                // the way the declaration itself would have been
                // converted; a body the projection screen would
                // reject is not fetched (the screen below judges
                // the declaration's own types, and the demand
                // passes judge the implementation separately).
                let eii_body = sparagmos_driver::convert::eii_impl_instance(&instance)
                    .and_then(|impl_inst| {
                        match fetch_query_normalize_aware(
                            &impl_inst.name(),
                            sparagmos_driver::convert::generated_simd_impl_method(
                                &impl_inst,
                            ),
                            || impl_inst.body(),
                        ) {
                            Ok(Some(body)) => Some(body),
                            _ => None,
                        }
                    })
                    .filter(|body| {
                        sparagmos_driver::convert::unnormalizable_projection_in_body(
                            body,
                        )
                        .is_none()
                    });
                match eii_body {
                    Some(body) => body,
                    None => {
                        let mut reasons = BTreeMap::new();
                        reasons.insert("no MIR body available".to_string(), 1);
                        report.functions.push(FnReport {
                            name,
                            blocks: 0,
                            locals: 0,
                            arg_count: 0,
                            outcome: FnOutcome::Skipped { reasons },
                        });
                        continue;
                    }
                }
            }
            Err(reason) => {
                if reason
                    == sparagmos_driver::convert::BODY_FETCH_NORMALIZE_OVERFLOW
                {
                    note_body_fetch_recursion_limit(
                        &mut body_fetch_recursion_limit,
                        &instance,
                    );
                    skip_fn(
                        &mut report,
                        &name,
                        sparagmos_driver::convert::BODY_FETCH_NORMALIZE_OVERFLOW
                            .to_string(),
                    );
                } else {
                    skip_fn(&mut report, &name, reason);
                }
                continue;
            }
        };
        convert_local_body(
            &mut conv,
            &mut report,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &mut seed_bodies,
            name,
            def_id,
            body,
        );
    }

    let local_body_count = bodies.len();
    let local_report_count = report.functions.len();

    // Demand monomorphic bodies for the `Drop` impls of the local
    // types the converted bodies drop. A generic `Drop` impl is
    // never instantiated by the item loop (it has no monomorphic
    // instance as an item) and never reached by the call-graph
    // collection (drop glue is not a call edge), so without this
    // pass the interpreter's glue builder finds the impl recorded
    // but bodiless and refuses the drop.
    let (mut drop_credited, mut drop_failures, mut dtor_bodies) = demand_drop_instances(
        &mut conv,
        &mut report,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &mut seed_bodies,
    );

    // Demand the `Iterator::next` instances of the
    // guest-stepped iterators the converted bodies hand to a
    // modelled consumer: the consumer's name is builtin-served,
    // so the call-graph collection stops there, and a generic
    // guest impl has no monomorphic instance as an item, while
    // the builtin drain's fallback step runs the impl body.
    // The credit and failure accounting folds into the drop
    // demand's maps, which are applied to the deferred
    // placeholders below.
    let (next_credited, next_failures) = demand_iterator_next_instances(
        &mut conv,
        &mut report,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &mut seed_bodies,
    );
    for (def, (count, counts)) in next_credited {
        let slot = drop_credited.entry(def).or_insert((0, counts));
        slot.0 += count;
    }
    for (def, reasons) in next_failures {
        let slot = drop_failures.entry(def).or_default();
        for (k, v) in reasons {
            *slot.entry(k.clone()).or_default() += v;
        }
    }

    // Demand the `core`-generated `Clone` bodies for tuple and
    // array types, for the same reason: the trait method's name
    // is builtin-served, so the call-graph collection stops
    // there and the elements' own `Clone` impls would never run.
    demand_clone_instances(
        &mut conv,
        &mut report,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &mut seed_bodies,
    );

    // Demand the `Wake` / `LocalWake` impl methods of the types
    // the converted bodies build wakers from: a waker's wake
    // dispatch is modelled in the interpreter, which calls the
    // impl methods by name, and `wake_by_ref`'s default body is
    // an instance nothing else ever demands (no call site names
    // it, and the vtable that would mention it is built by the
    // modelled `From` conversion, whose body is not collected).
    demand_wake_instances(
        &mut conv,
        &mut report,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &mut seed_bodies,
    );

    // Demand the `Fn`-family impl bodies of the zero-sized
    // callable types the converted bodies mention: an adapter
    // builtin consumes such a callable without any call edge
    // naming its body (see `demand_zst_fn_instances`). The
    // demanded bodies join the seed bodies like the wake
    // demand's, and the pass runs again over the collected
    // foreign bodies below, where library adapters name these
    // callables.
    let mut zst_fn_demanded = 0usize;
    demand_zst_fn_instances(
        &mut conv,
        &mut report,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &mut seed_bodies,
        &mut zst_fn_demanded,
    );

    // Convert the monomorphic instances of the generic local
    // function definitions that the converted bodies call; the
    // instance bodies join the reachability seeds below, so the
    // foreign functions they call are collected too.
    let mut demand_state = InstanceDemandState::default();
    let instance_seeds = collect_local_instances(
        &mut conv,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &deferred,
        &seed_bodies,
        &mut demand_state,
    );
    // The program DefIds and report names collected so far,
    // shared with every merge and demand pass below: the
    // reachability collection resolves some of the same closure
    // instances the closure demand passes do, and a body may
    // enter the program only once under its DefId (report
    // entries only once under their display name).
    let mut collected_ids: HashSet<sparagmos_mir::DefId> =
        body_def_ids.iter().copied().collect();
    let mut report_names: HashSet<String> =
        report.functions.iter().map(|f| f.name.clone()).collect();
    // Convert the demanded instances of closures defined in
    // generic contexts: the converted local bodies and the
    // demanded function instances mention them with concrete
    // arguments. The closure bodies join the reachability seeds,
    // so the foreign functions they call are collected too.
    let mut closure_scan = seed_bodies.clone();
    closure_scan.extend(instance_seeds.iter().cloned());
    let closure_seeds = collect_closure_instances(
        &mut conv,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &deferred_closures,
        &local_defs,
        &closure_scan,
        &collected_ids,
        &mut demand_state,
    );
    // Convert the demanded instances of the local coroutines
    // the converted bodies create or hold: a coroutine is
    // demanded by its creation aggregate and its coroutine-typed
    // locals, which neither the instance demand nor the closure
    // demand follows.
    let mut coroutine_scan = closure_scan;
    coroutine_scan.extend(closure_seeds.iter().cloned());
    let coroutine_seeds = collect_coroutine_instances(
        &mut conv,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &deferred,
        &coroutine_scan,
        &mut demand_state,
    );
    // Convert the closures the conversions so far materialized
    // as values out of constants (see
    // collect_const_closure_instances): their bodies join the
    // reachability seeds and the fixpoint scan set like the
    // closure pass's.
    let const_seeds = collect_const_closure_instances(
        &mut conv,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &deferred_closures,
        &local_defs,
        &collected_ids,
        &mut demand_state,
    );
    // And the plain functions the same conversions materialized
    // as values out of constants (see
    // collect_const_fn_instances): their bodies join the same
    // seed sets.
    let const_fn_seeds = collect_const_fn_instances(
        &mut conv,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &deferred,
        &local_defs,
        &collected_ids,
        &mut demand_state,
    );
    // Local bodies converted by the closure and coroutine passes
    // have not been scanned for the *function* instances they
    // demand (each demand pass scans only for its own kind of
    // target). The fixpoint loop below scans them with the
    // instance demand, and each round's newly converted closure
    // and coroutine bodies become the next round's scan set, so
    // every converted body is eventually scanned by every pass.
    // The vtable-demanded bodies converted below join the same
    // scan set for the same reason (see the note at the
    // `demand_dyn_instances` call).
    let mut local_scan: Vec<rustc_public::mir::Body> = closure_seeds.clone();
    local_scan.extend(coroutine_seeds.iter().cloned());
    local_scan.extend(const_seeds.iter().cloned());
    local_scan.extend(const_fn_seeds.iter().cloned());
    let mut reach_seeds = seed_bodies;
    reach_seeds.extend(instance_seeds);
    reach_seeds.extend(closure_seeds);
    reach_seeds.extend(coroutine_seeds);
    reach_seeds.extend(const_seeds);
    reach_seeds.extend(const_fn_seeds);

    // Demand the vtable method bodies of the `dyn` values the
    // bodies scanned so far form (trait default methods reached
    // only through a vtable are the case this serves; see
    // `demand_dyn_instances`). The demanded bodies join the
    // reachability seeds, so the functions they call are
    // collected too. The state is shared with the fixpoint rounds
    // below, which rescan the bodies those rounds convert.
    let deferred_def_set: HashSet<rustc_public::DefId> =
        deferred.iter().map(|d| d.def).collect();
    let mut dyn_credited: HashMap<
        rustc_public::DefId,
        (usize, (usize, usize, usize)),
    > = HashMap::new();
    let mut dyn_demanded = 0usize;
    // A recursion-limit failure recorded by the demand passes
    // seals the compilation verdict (the demand state's slot is
    // the verdict chain's first input and is never cleared):
    // the conversions and collections below can only build a
    // program the compilation has already failed, so they are
    // skipped wholesale when the slot is set.
    let dyn_seeds = if demand_state.recursion_limit.is_some() {
        Vec::new()
    } else {
        demand_dyn_instances(
            &mut conv,
            &mut report,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &reach_seeds.clone(),
            &deferred_def_set,
            &mut dyn_credited,
            &mut dyn_demanded,
        )
    };
    reach_seeds.extend(dyn_seeds.iter().cloned());
    // The vtable-demanded bodies also join the instance-demand
    // scan set: a demanded method body can itself demand generic
    // local instances concretely (a forwarding trait impl calling
    // a trait default method on its own fields is the common
    // shape), and no other pass scans a converted body for that
    // kind of demand -- the reachability walk collects the
    // foreign functions such bodies call but skips instances of
    // local definitions by design, and the rounds below only ever
    // rescan the bodies their own rounds converted. Without
    // this, a call inside a vtable-demanded body to a local
    // default-method instance is left with no converted body.
    local_scan.extend(dyn_seeds);

    // In the executing modes the reachability passes walk
    // execution-first (see `reach::CollectionOrder`): the entry
    // point's body is scanned ahead of the other seeds, so under
    // the collection cap the executable core is collected before
    // speculative periphery. Convert mode keeps the historical
    // first-in-first-out walk; its report is a measurement
    // contract.
    let collection_order = if EXECUTION_FIRST_MODE
        .load(std::sync::atomic::Ordering::Relaxed)
    {
        sparagmos_driver::reach::CollectionOrder::ExecutionFirst
    } else {
        sparagmos_driver::reach::CollectionOrder::Fifo
    };
    let entry_seeds =
        if collection_order == sparagmos_driver::reach::CollectionOrder::ExecutionFirst {
            entry_seed_bodies(&items)
        } else {
            Vec::new()
        };

    // Collect the bodies of non-local functions reachable from the
    // converted local bodies (std / core / alloc and other library
    // functions), so calls into them have bodies in the program too.
    let reached = if demand_state.recursion_limit.is_some() {
        sparagmos_driver::reach::CollectedFns {
            bodies: Vec::new(),
            body_def_ids: Vec::new(),
            body_def_names: Vec::new(),
            reports: Vec::new(),
            stable_bodies: Vec::new(),
            limit_reached: false,
            recursion_limit: None,
            walked_bodies: Vec::new(),
        }
    } else {
        sparagmos_driver::reach::collect_reachable(
            &mut conv,
            &reach_seeds,
            &local_defs,
            sparagmos_driver::reach::MAX_COLLECTED_BODIES,
            collection_order,
            &entry_seeds,
        )
    };
    if body_fetch_recursion_limit.is_none() {
        body_fetch_recursion_limit = reached.recursion_limit.clone();
    }
    collected_ids.extend(body_def_ids.iter().copied());
    let mut foreign_taken = 0usize;
    let mut frontier = reached.walked_bodies;
    for ((name, body), (def_id, def_name)) in reached
        .bodies
        .into_iter()
        .zip(reached.body_def_ids.into_iter().zip(reached.body_def_names))
    {
        if collected_ids.insert(def_id) {
            foreign_taken += 1;
            bodies.push((name, body));
            body_def_ids.push(def_id);
            body_def_names.push(def_name);
        }
    }
    for entry in reached.reports {
        if report_names.insert(entry.name.clone()) {
            report.functions.push(entry);
        }
    }

    // A generic local definition can also be demanded from inside
    // a collected foreign body (a std adapter calling a trait
    // method of a local generic type): the reachability collection
    // skips targets that resolve to local items, and the demand
    // pass above only walked local bodies, so such instances were
    // never converted. Alternate the two passes to a fixpoint:
    // scan the newly collected foreign bodies for demanded local
    // instances, then collect the foreign functions those instance
    // bodies call, whose bodies may in turn demand more local
    // instances. Later reachability passes re-walk (and
    // re-convert) targets earlier passes already collected; the
    // duplicates are dropped by DefId for bodies and by display
    // name for report entries (the sets above, which the initial
    // merge already used), and both collections keep their
    // per-program caps across passes (shared demand state, and a
    // remaining-budget argument for reachability).
    // Stable MIR sources for the trait-key alias pass below: the
    // local seeds plus every foreign body the reachability passes
    // walk, across all fixpoint rounds.
    let mut alias_sources = reach_seeds.clone();
    alias_sources.extend(frontier.iter().cloned());
    loop {
        let ids_before = body_def_ids.len();
        let mut fn_scan = frontier.clone();
        fn_scan.extend(local_scan.iter().cloned());
        let new_instances = collect_local_instances(
            &mut conv,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &deferred,
            &fn_scan,
            &mut demand_state,
        );
        // Newly collected foreign bodies and demanded instances
        // can also mention generic-context closures concretely.
        let mut closure_scan = frontier.clone();
        closure_scan.extend(new_instances.iter().cloned());
        let new_closure_bodies = collect_closure_instances(
            &mut conv,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &deferred_closures,
            &local_defs,
            &closure_scan,
            &collected_ids,
            &mut demand_state,
        );
        // Newly collected bodies can likewise create or hold
        // local coroutines concretely.
        let mut coroutine_scan = closure_scan;
        coroutine_scan.extend(new_closure_bodies.iter().cloned());
        let new_coroutine_bodies = collect_coroutine_instances(
            &mut conv,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &deferred,
            &coroutine_scan,
            &mut demand_state,
        );
        // Bodies converted or collected this round can form
        // `dyn` values whose vtables no earlier pass demanded
        // (a demanded generic instance unsizing its parameter's
        // concrete type is the common shape).
        let mut dyn_scan = fn_scan.clone();
        dyn_scan.extend(new_instances.iter().cloned());
        dyn_scan.extend(new_closure_bodies.iter().cloned());
        dyn_scan.extend(new_coroutine_bodies.iter().cloned());
        let new_dyn_bodies = if demand_state.recursion_limit.is_some() {
            Vec::new()
        } else {
            demand_dyn_instances(
                &mut conv,
                &mut report,
                &mut bodies,
                &mut body_def_ids,
                &mut body_def_names,
                &dyn_scan,
                &deferred_def_set,
                &mut dyn_credited,
                &mut dyn_demanded,
            )
        };
        // Newly converted bodies also decode constants, which
        // can materialize further closures as values (the
        // `RandomState::new::KEYS` shape: a foreign constant's
        // function pointer naming a closure no type mentions).
        let new_const_bodies = collect_const_closure_instances(
            &mut conv,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &deferred_closures,
            &local_defs,
            &collected_ids,
            &mut demand_state,
        );
        let new_const_fn_bodies = collect_const_fn_instances(
            &mut conv,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &deferred,
            &local_defs,
            &collected_ids,
            &mut demand_state,
        );
        if new_instances.is_empty()
            && new_closure_bodies.is_empty()
            && new_coroutine_bodies.is_empty()
            && new_dyn_bodies.is_empty()
            && new_const_bodies.is_empty()
            && new_const_fn_bodies.is_empty()
        {
            break;
        }
        local_scan = new_closure_bodies.clone();
        local_scan.extend(new_coroutine_bodies.iter().cloned());
        local_scan.extend(new_dyn_bodies.iter().cloned());
        local_scan.extend(new_const_bodies.iter().cloned());
        local_scan.extend(new_const_fn_bodies.iter().cloned());
        for id in &body_def_ids[ids_before..] {
            collected_ids.insert(*id);
        }
        if foreign_taken >= sparagmos_driver::reach::MAX_COLLECTED_BODIES {
            break;
        }
        let mut reach_input = new_instances;
        reach_input.extend(new_closure_bodies.iter().cloned());
        reach_input.extend(new_coroutine_bodies.iter().cloned());
        reach_input.extend(new_dyn_bodies.iter().cloned());
        reach_input.extend(new_const_bodies.iter().cloned());
        let reached = sparagmos_driver::reach::collect_reachable(
            &mut conv,
            &reach_input,
            &local_defs,
            sparagmos_driver::reach::MAX_COLLECTED_BODIES - foreign_taken,
            collection_order,
            &[],
        );
        if body_fetch_recursion_limit.is_none() {
            body_fetch_recursion_limit = reached.recursion_limit.clone();
        }
        frontier = reached.walked_bodies;
        alias_sources.extend(frontier.iter().cloned());
        alias_sources.extend(new_closure_bodies.iter().cloned());
        alias_sources.extend(new_coroutine_bodies.iter().cloned());
        alias_sources.extend(new_dyn_bodies.iter().cloned());
        alias_sources.extend(new_const_bodies.iter().cloned());
        for ((name, body), (def_id, def_name)) in reached
            .bodies
            .into_iter()
            .zip(reached.body_def_ids.into_iter().zip(reached.body_def_names))
        {
            if collected_ids.insert(def_id) {
                foreign_taken += 1;
                bodies.push((name, body));
                body_def_ids.push(def_id);
                body_def_names.push(def_name);
            }
        }
        for entry in reached.reports {
            if report_names.insert(entry.name.clone()) {
                report.functions.push(entry);
            }
        }
    }
    // The reachability conversions of the final round above can
    // record further constant-content closure mentions (their
    // conversions decode new constants), so drain that demand to
    // a fixpoint as well: demand what the mentions name, walk
    // what it converts for the foreign functions those closures
    // call, and repeat until a round converts nothing.
    loop {
        let ids_before = body_def_ids.len();
        let new_const_bodies = collect_const_closure_instances(
            &mut conv,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &deferred_closures,
            &local_defs,
            &collected_ids,
            &mut demand_state,
        );
        let new_const_fn_bodies = collect_const_fn_instances(
            &mut conv,
            &mut bodies,
            &mut body_def_ids,
            &mut body_def_names,
            &deferred,
            &local_defs,
            &collected_ids,
            &mut demand_state,
        );
        if new_const_bodies.is_empty() && new_const_fn_bodies.is_empty() {
            break;
        }
        for id in &body_def_ids[ids_before..] {
            collected_ids.insert(*id);
        }
        alias_sources.extend(new_const_bodies.iter().cloned());
        alias_sources.extend(new_const_fn_bodies.iter().cloned());
        if foreign_taken >= sparagmos_driver::reach::MAX_COLLECTED_BODIES {
            break;
        }
        let mut walk_seeds = new_const_bodies.clone();
        walk_seeds.extend(new_const_fn_bodies.iter().cloned());
        let reached = sparagmos_driver::reach::collect_reachable(
            &mut conv,
            &walk_seeds,
            &local_defs,
            sparagmos_driver::reach::MAX_COLLECTED_BODIES - foreign_taken,
            collection_order,
            &[],
        );
        if body_fetch_recursion_limit.is_none() {
            body_fetch_recursion_limit = reached.recursion_limit.clone();
        }
        alias_sources.extend(reached.walked_bodies.iter().cloned());
        for ((name, body), (def_id, def_name)) in reached
            .bodies
            .into_iter()
            .zip(reached.body_def_ids.into_iter().zip(reached.body_def_names))
        {
            if collected_ids.insert(def_id) {
                foreign_taken += 1;
                bodies.push((name, body));
                body_def_ids.push(def_id);
                body_def_names.push(def_name);
            }
        }
        for entry in reached.reports {
            if report_names.insert(entry.name.clone()) {
                report.functions.push(entry);
            }
        }
    }
    // Second drop-demand round over the full collected body
    // set: the first round ran before the reachability
    // collection, so it only saw the entry crate's bodies and
    // the type trees of the types they drop. Foreign bodies
    // collected by reachability drop foreign machinery of
    // their own (hashbrown's allocation guards are the
    // motivating case), and those drops need the same
    // destructor bodies. Instances the first round already
    // converted are skipped by identity inside the pass.
    if demand_state.recursion_limit.is_none() {
        let mut all_bodies = alias_sources.clone();
        // The destructor bodies this round converts call
        // functions of their own (the array clone guard's drop
        // calls a slice-index precondition check, for one), and
        // drop glue is not a call edge, so no earlier
        // reachability wave walked them: walk each newly
        // converted destructor body's call edges, merge what
        // they reach, and demand drops again over the enlarged
        // set, until a round converts no new destructor body.
        loop {
            let seeds_before = all_bodies.len();
            let bodies_before = bodies.len();
            let (credited2, failures2, dtor_bodies2) = demand_drop_instances(
                &mut conv,
                &mut report,
                &mut bodies,
                &mut body_def_ids,
                &mut body_def_names,
                &mut all_bodies,
            );
            for (def, (count, counts)) in credited2 {
                let slot = drop_credited.entry(def).or_insert((0, counts));
                slot.0 += count;
            }
            for (def, reasons) in failures2 {
                let slot = drop_failures.entry(def).or_default();
                for (k, v) in reasons {
                    *slot.entry(k).or_default() += v;
                }
            }
            dtor_bodies.extend(dtor_bodies2);
            if bodies.len() == bodies_before {
                break;
            }
            if foreign_taken >= sparagmos_driver::reach::MAX_COLLECTED_BODIES {
                break;
            }
            let walk_seeds: Vec<rustc_public::mir::Body> =
                all_bodies[seeds_before..].to_vec();
            let reached = sparagmos_driver::reach::collect_reachable(
                &mut conv,
                &walk_seeds,
                &local_defs,
                sparagmos_driver::reach::MAX_COLLECTED_BODIES - foreign_taken,
                collection_order,
                &[],
            );
            if body_fetch_recursion_limit.is_none() {
                body_fetch_recursion_limit = reached.recursion_limit.clone();
            }
            alias_sources.extend(reached.walked_bodies.iter().cloned());
            all_bodies.extend(reached.walked_bodies.iter().cloned());
            for ((name, body), (def_id, def_name)) in reached
                .bodies
                .into_iter()
                .zip(reached.body_def_ids.into_iter().zip(reached.body_def_names))
            {
                if collected_ids.insert(def_id) {
                    foreign_taken += 1;
                    bodies.push((name, body));
                    body_def_ids.push(def_id);
                    body_def_names.push(def_name);
                }
            }
            for entry in reached.reports {
                if report_names.insert(entry.name.clone()) {
                    report.functions.push(entry);
                }
            }
        }
    }
    // Second ZST-Fn demand round over the full collected body
    // set, mirroring the second drop round above: the first
    // round only saw the entry crate's bodies, but the callable
    // types at issue are named by foreign bodies (a library
    // adapter's state type). The bodies this round converts are
    // walked for their own callees and the enlarged set is
    // demanded again, until a round converts nothing new.
    if demand_state.recursion_limit.is_none() {
        let mut all_bodies = alias_sources.clone();
        loop {
            let seeds_before = all_bodies.len();
            let bodies_before = bodies.len();
            demand_zst_fn_instances(
                &mut conv,
                &mut report,
                &mut bodies,
                &mut body_def_ids,
                &mut body_def_names,
                &mut all_bodies,
                &mut zst_fn_demanded,
            );
            if bodies.len() == bodies_before {
                break;
            }
            if foreign_taken >= sparagmos_driver::reach::MAX_COLLECTED_BODIES {
                break;
            }
            let walk_seeds: Vec<rustc_public::mir::Body> =
                all_bodies[seeds_before..].to_vec();
            let reached = sparagmos_driver::reach::collect_reachable(
                &mut conv,
                &walk_seeds,
                &local_defs,
                sparagmos_driver::reach::MAX_COLLECTED_BODIES - foreign_taken,
                collection_order,
                &[],
            );
            if body_fetch_recursion_limit.is_none() {
                body_fetch_recursion_limit = reached.recursion_limit.clone();
            }
            alias_sources.extend(reached.walked_bodies.iter().cloned());
            all_bodies.extend(reached.walked_bodies.iter().cloned());
            for ((name, body), (def_id, def_name)) in reached
                .bodies
                .into_iter()
                .zip(reached.body_def_ids.into_iter().zip(reached.body_def_names))
            {
                if collected_ids.insert(def_id) {
                    foreign_taken += 1;
                    bodies.push((name, body));
                    body_def_ids.push(def_id);
                    body_def_names.push(def_name);
                }
            }
            for entry in reached.reports {
                if report_names.insert(entry.name.clone()) {
                    report.functions.push(entry);
                }
            }
        }
    }
    // Credit the vtable instances the dyn demand converted for
    // deferred definitions (a trait default method is the common
    // case): the "dynamic dispatch target (virtual instance)"
    // failure the demand passes recorded for such a definition is
    // served by those bodies, so it no longer stands against it.
    // Any other failure of the same definition is untouched, and
    // a definition no vtable instance was converted for keeps the
    // failure — its dyn calls stay unsupported at run time.
    for (def, (count, counts)) in &dyn_credited {
        let demand = demand_state.demands.entry(*def).or_default();
        demand.converted += count;
        if demand.first_counts.is_none() {
            demand.first_counts = Some(*counts);
        }
        demand
            .failures
            .remove("dynamic dispatch target (virtual instance)");
    }
    // A deferred trait method with a default body whose only
    // demand failure is the virtual-instance one was never
    // demanded as a monomorphic body: the demand came from
    // dynamic-dispatch sites in generic bodies, and no vtable
    // formation anywhere in the program names a concrete `Self`
    // for it (a formed vtable would have credited the definition
    // above). The default body is demandable in principle, so
    // nothing convertible was left unconverted; drop the demand
    // entry and the definition is accounted like an un-demanded
    // one. A definition with any other failure, one an instance
    // of which did convert, or a trait method without a default
    // body (no body of its own exists to demand) keeps its
    // accounting.
    for dfn in &deferred {
        let Some(demand) = demand_state.demands.get(&dfn.def) else {
            continue;
        };
        if demand.converted > 0
            || demand.failures.len() != 1
            || !demand
                .failures
                .contains_key("dynamic dispatch target (virtual instance)")
        {
            continue;
        }
        let has_default_body = guard_stable_mir_query("", || {
            rustc_public::ty::FnDef(dfn.def).has_body()
        })
        .unwrap_or(false);
        if has_default_body {
            demand_state.demands.remove(&dfn.def);
        }
    }
    // Credit the destructor instances the drop demand converted
    // to their impl definitions (and record its failures) the
    // same way: drop glue is not a call edge, so without this the
    // generic `Drop` impl's placeholder entry would stand against
    // the conversion even when the demanded instance converted.
    for (def, (count, counts)) in &drop_credited {
        let demand = demand_state.demands.entry(*def).or_default();
        demand.converted += count;
        if demand.first_counts.is_none() {
            demand.first_counts = Some(*counts);
        }
    }
    for (def, reasons) in &drop_failures {
        let demand = demand_state.demands.entry(*def).or_default();
        for (k, v) in reasons {
            *demand.failures.entry(k.clone()).or_default() += v;
        }
    }
    apply_instance_demands(&mut report, &deferred, &demand_state.demands);
    apply_instance_demands(&mut report, &deferred_closures, &demand_state.demands);
    apply_foreign_closure_demands(&mut report, &demand_state.foreign_closure_demands);
    apply_foreign_closure_demands(&mut report, &demand_state.foreign_fn_demands);

    // Register converted impl method bodies under the trait keys
    // their call sites resolve through (Issue #64). Demands can
    // come from any converted body: local items, demanded local
    // instances, and collected foreign bodies alike.
    register_trait_key_aliases(
        &mut conv,
        &mut bodies,
        &mut body_def_ids,
        &mut body_def_names,
        &alias_sources,
    );

    // Auxiliary crates (--aux): the main compilation links them as
    // dependencies, so their functions live in this same Stable MIR
    // session and share the converter's DefId assignment with the
    // local items. Convert them into the same program, after the
    // local functions. A generic auxiliary function has no
    // monomorphic body to convert and is recorded as skipped, the
    // same treatment local generic functions get. Genericity is
    // checked up front from the function's generic parameters (any
    // type or const parameter, its own or an enclosing
    // definition's): resolving a generic function to an instance
    // with empty generic arguments hands back its polymorphic body
    // and panics inside the compiler when it is instantiated.
    for crate_name in aux_crates() {
        for krate in rustc_public::find_crates(crate_name) {
            // An auxiliary crate can share the main crate's name
            // (see the E0519 note in `main`); `find_crates` then
            // also returns the local crate, whose functions the loop
            // above already collected.
            if krate.is_local {
                continue;
            }
            for fn_def in krate.fn_defs() {
                let name = fn_def.name();
                if bodies.iter().any(|(n, _)| n == &name) {
                    // Already converted (the reachability
                    // collection above picks up auxiliary
                    // functions called from the main crate).
                    continue;
                }
                let def_id = conv.crate_def_id(&fn_def);
                conv.note_fn_abi_for_def(def_id, &fn_def);
                conv.note_link_symbol_for_def(def_id, fn_def.def_id());
                let generic = fn_def.generics_of().params.iter().any(|param| {
                    matches!(
                        param.kind,
                        rustc_public::ty::GenericParamDefKind::Type { .. }
                            | rustc_public::ty::GenericParamDefKind::Const { .. }
                    )
                });
                let body = if generic {
                    Err("no monomorphic instance available \
                         (auxiliary crate function)"
                        .to_string())
                } else {
                    fetch_query_normalize_aware(&fn_def.name(), false, || fn_def.body())
                };
                convert_item(
                    &mut conv,
                    &mut report,
                    &mut bodies,
                    &mut body_def_ids, &mut body_def_names,
                    name,
                    def_id,
                    body,
                );
            }
        }
    }



    // Monomorphization failures rustc reports around MIR
    // conversion: the recursion limit the demand passes hit while
    // collecting instances (a mono-collection error, so it takes
    // precedence), the collector verdicts `mono_check` replays
    // over the instances it collects (the required-constant
    // evaluation, the per-instance `check_mono_item` checks, the
    // duplicate-symbol verdict, the common-linkage verdict and
    // the panic-runtime validation; also collection-time
    // errors, so they come next), the unshimmed-intrinsic
    // verdict over the converted bodies (a codegen-stage
    // rejection, gated like the collector verdicts so it cannot
    // pre-empt execution), and the
    // checks of `sparagmos_driver::mono_check` over the converted
    // bodies (invalid SIMD intrinsic instantiations and SIMD
    // layouts, types too big for the target architecture), and
    // finally the deferred `check_mono_item` verdict the
    // collector walk recorded, whose rejections the body checks
    // sometimes name more precisely. A
    // failure here fails the whole compilation, the way the
    // corresponding rustc error does. The body scan covers the
    // Stable MIR bodies conversion walked (local seeds, demanded
    // instances, and the reachability frontier); auxiliary-crate
    // bodies converted by the loop above are not retained in that
    // set and are not scanned.
    let compile_error = demand_state
        .recursion_limit
        .clone()
        .or_else(|| body_fetch_recursion_limit.clone())
        .or_else(|| {
            if !COMPILE_VERDICT_MODE.load(std::sync::atomic::Ordering::Relaxed) {
                return None;
            }
            sparagmos_driver::mono_check::check_required_consts()
        })
        .or_else(|| {
            if !COMPILE_VERDICT_MODE.load(std::sync::atomic::Ordering::Relaxed) {
                return None;
            }
            sparagmos_driver::mono_check::check_intrinsic_overrides(&alias_sources)
        })
        .or_else(|| sparagmos_driver::mono_check::check_bodies(&alias_sources))
        .or_else(|| sparagmos_driver::mono_check::take_mono_item_finding());

    // The coverage mapping table of an instrumented
    // compilation: captured from the compiler's own mono
    // collection and coverage queries, with each record linked
    // to the converted bodies of its instance (by mangled
    // symbol name, through the converter's instance registry).
    // Uninstrumented compilations skip the sweep entirely.
    let coverage = match sparagmos_driver::convert::compiler_tcx() {
        Some(tcx)
            if sparagmos_driver::coverage::instrument_coverage_enabled(tcx) =>
        {
            // The interpreter stores bodies under program keys:
            // the bare name, or `name#DefId` when several bodies
            // share a name (mir_convert's `program_keys`, over
            // exactly this body list). Coverage records link to
            // bodies by that key, so a statement's record is
            // resolved under the name the executing body carries.
            let mut name_counts: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::new();
            for (name, _) in bodies.iter() {
                *name_counts.entry(name.as_str()).or_default() += 1;
            }
            let body_keys: Vec<String> = bodies
                .iter()
                .zip(body_def_ids.iter())
                .map(|((name, _), def)| {
                    if name_counts[name.as_str()] == 1 {
                        name.clone()
                    } else {
                        format!("{name}#{def}")
                    }
                })
                .collect();
            let mut by_idx: std::collections::HashMap<u32, &str> =
                std::collections::HashMap::new();
            for (i, id) in body_def_ids.iter().enumerate() {
                by_idx.insert(id.index, body_keys[i].as_str());
            }
            let mut converted: std::collections::HashMap<
                String,
                Vec<String>,
            > = std::collections::HashMap::new();
            for ((_, inst), &idx) in conv.instance_ids() {
                if let Some(name) = by_idx.get(&idx) {
                    converted
                        .entry(inst.mangled_name())
                        .or_default()
                        .push(name.to_string());
                }
            }
            let instances: std::collections::HashMap<
                u32,
                rustc_public::mir::mono::Instance,
            > = conv
                .instance_ids()
                .iter()
                .map(|((_, inst), &idx)| (idx, inst.clone()))
                .collect();
            let all_body_names: std::collections::HashSet<String> =
                body_keys.iter().cloned().collect();
            sparagmos_driver::coverage::capture(
                tcx,
                &converted,
                &instances,
                &all_body_names,
            )
        }
        _ => sparagmos_mir::CoverageMapping::default(),
    };

    // Attribute each converted body's coverage statements to
    // the instance they were inlined from, so execution counts
    // a statement against its owner's record the way codegen
    // emits the increment (see `coverage::fill_coverage_owners`).
    if !coverage.functions.is_empty() {
        if let Some(tcx) = sparagmos_driver::convert::compiler_tcx() {
            let instances: std::collections::HashMap<
                u32,
                rustc_public::mir::mono::Instance,
            > = conv
                .instance_ids()
                .iter()
                .map(|((_, inst), &idx)| (idx, inst.clone()))
                .collect();
            sparagmos_driver::coverage::fill_coverage_owners(
                tcx,
                &instances,
                &mut bodies,
                &body_def_ids,
            );
        }
    }

    // The unicode skiplist tables' seeds, recorded only
    // when a `lookup_slow` callee is part of the program:
    // those statics' only reader has no converted body, so
    // nothing else ever names them (see
    // `Converter::note_unicode_data_tables`).
    if conv
        .def_names
        .iter()
        .any(|n| n.contains("::unicode_data::") && n.ends_with("::lookup_slow"))
    {
        conv.note_unicode_data_tables();
    }

    let verdict_failed = compile_error.is_some();

    let entry_def_id = entry_point_def_id(&mut conv, &items, &body_def_ids);

    // The crate graph's panic handler, named by its collected
    // body: the allocation-error model delivers alloc's
    // default (`__rdl`) panic to it in a program that links
    // no std (see `MirProgram::panic_handler`). A handler
    // whose body was not collected names nothing.
    let panic_handler = sparagmos_driver::convert::panic_handler_def().and_then(|def| {
        let item = items.iter().find(|item| item.def_id() == def)?;
        let name = item.name();
        bodies.iter().any(|(n, _)| n == &name).then_some(name)
    });
    // Whether the crate graph links std: the external crates
    // the local crate's compilation names (see
    // `MirProgram::links_std`).
    let links_std = rustc_public::external_crates()
        .iter()
        .any(|krate| krate.name == "std");

    // Certify the dropped types whose drop glue the compiler
    // knows is empty, over the same body set the monomorphization
    // checks scan: the interpreter's glue builder consults the
    // set for the foreign types its name table cannot rule a
    // destructor out for.
    let drop_trivial_adts = certify_drop_trivial(&mut conv, &alias_sources);
    let drop_no_impl_adts = certify_drop_no_impl(&mut conv, &alias_sources);
    let needs_drop_verdicts = certify_needs_drop(&mut conv, &alias_sources);

    stash(Collected {
        report,
        bodies,
        body_def_ids,
        body_def_names,
        fn_abis: conv.fn_abis.clone(),
        local_body_count,
        local_report_count,
        entry_def_id,
        panic_handler,
        links_std,
        def_names: conv.def_names.clone(),
        intrinsic_defs: conv.intrinsic_defs.clone(),
        async_closure_defs: conv.async_closure_defs.clone(),
        vtable_identities: conv.vtable_identities.clone(),
        adt_discriminants: conv.adt_discriminants.clone(),
        compile_error,
        adt_layouts: conv.adt_layouts.clone(),
        adt_dst_layouts: conv.adt_dst_layouts.clone(),
        adt_field_layouts: conv.adt_field_layouts.clone(),
        tuple_layouts: conv.tuple_layouts.clone(),
        adt_tag_layouts: conv.adt_tag_layouts.clone(),
        coroutine_layouts: conv.coroutine_layouts.clone(),
        adt_shapes: conv.adt_shapes.clone(),
        adt_debug_shapes: conv.adt_debug_shapes.clone(),
        env_debug_shapes: conv.env_debug_shapes.clone(),
        thread_locals: conv.thread_locals.clone(),
        extra_static_seeds: conv.extra_static_seeds.clone(),
        static_tys: conv.static_tys_mir.clone(),
        adt_instance_shapes: conv.adt_instance_shapes.clone(),
        adt_niches: conv.adt_niches.clone(),
        mat_plans: conv.mat_plans.clone(),
        drop_trivial_adts,
        drop_no_impl_adts,
        needs_drop_verdicts,
        dtor_bodies,
        track_caller_defs: conv.track_caller_defs.clone(),
        fn_shim_locations: conv.fn_shim_locations.clone(),
        coroutine_shim_location: conv.coroutine_shim_location.clone(),
        track_caller_coroutines: conv.track_caller_coroutines.clone(),
        link_symbols: conv.link_symbols.clone(),
        source_files: conv.source_files.clone(),
        coverage,
        fn_display_names: sparagmos_driver::convert::drain_fn_display_names(),
        crate_overflow_checks: aux_overflow_checks().to_vec(),
    });
    // A compilation the verdict chain has rejected stops the
    // session here, the way rustc's own start-of-codegen gate
    // refuses to codegen a session that has errors: the
    // diagnostics behind the verdict were discounted at their
    // fetches (their error count reset, so the compiler's gate
    // cannot see them), and the phases past analysis would run
    // on a program the compilation has already failed -- code
    // generation evaluates the very constants such a verdict
    // rests on and panics on them ("erroneous constant missed
    // by mono item collection"), masking the rejection as a
    // compiler crash. The verdict itself is reported by the
    // caller from the results stashed above.
    if verdict_failed {
        ControlFlow::Break(())
    } else {
        ControlFlow::Continue(())
    }
}
