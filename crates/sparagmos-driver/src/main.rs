//! `sparagmos-driver` — the Sparagmos rustc driver.
//!
//! Usage: `sparagmos-driver <file.rs> [--dump <path>] [--edition <ed>]`
//!
//! Compiles `<file.rs>` as a single crate with the pinned Rust 1.100.0
//! toolchain, obtains every local function's MIR through Stable MIR
//! (`rustc_public`), converts it to `sparagmos_mir::Body`, and prints a
//! per-function summary. Build/run instructions: see the crate README.

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_public;
extern crate rustc_span;

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::sync::{Mutex, OnceLock};

use rustc_public::mir::mono::Instance;
use rustc_public::CrateDef;

use sparagmos_driver::convert::Converter;
use sparagmos_driver::report::{ConversionReport, FnOutcome, FnReport};
use sparagmos_mir::Body;

struct Collected {
    report: ConversionReport,
    bodies: Vec<(String, Body)>,
    def_names: Vec<String>,
}

static COLLECTED: OnceLock<Mutex<Option<Collected>>> = OnceLock::new();

fn stash(c: Collected) {
    let cell = COLLECTED.get_or_init(|| Mutex::new(None));
    *cell.lock().unwrap() = Some(c);
}

fn take() -> Option<Collected> {
    COLLECTED.get()?.lock().unwrap().take()
}

fn collect() -> ControlFlow<()> {
    let mut conv = Converter::new();
    let mut report = ConversionReport::default();
    let mut bodies: Vec<(String, Body)> = Vec::new();

    let items = rustc_public::all_local_items();
    for item in items.iter() {
        let name = item.name();
        let instance = match Instance::try_from(item.clone()) {
            Ok(i) => i,
            Err(_) => {
                let mut reasons = BTreeMap::new();
                reasons.insert(
                    format!("no monomorphic instance available (kind={:?})", item.kind()),
                    1,
                );
                report.functions.push(FnReport {
                    name,
                    blocks: 0,
                    locals: 0,
                    arg_count: 0,
                    outcome: FnOutcome::Skipped { reasons },
                });
                continue;
            }
        };
        let Some(body) = instance.body() else {
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
        };
        let blocks = body.blocks.len();
        let locals = body.locals().len();
        let arg_count = body.arg_locals().len();
        match conv.convert_body(&body) {
            Ok(converted) => {
                bodies.push((name.clone(), converted));
                report.functions.push(FnReport {
                    name,
                    blocks,
                    locals,
                    arg_count,
                    outcome: FnOutcome::Converted,
                });
            }
            Err(errs) => {
                let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
                for e in errs {
                    *reasons.entry(e).or_default() += 1;
                }
                report.functions.push(FnReport {
                    name,
                    blocks,
                    locals,
                    arg_count,
                    outcome: FnOutcome::Skipped { reasons },
                });
            }
        }
    }

    stash(Collected {
        report,
        bodies,
        def_names: conv.def_names.clone(),
    });
    ControlFlow::Continue(())
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut input: Option<String> = None;
    let mut dump_path: Option<String> = None;
    let mut edition = "2021".to_string();
    let mut sysroot: Option<String> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--dump" => {
                i += 1;
                dump_path = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--dump needs a path");
                    std::process::exit(2);
                }));
            }
            "--edition" => {
                i += 1;
                edition = argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--edition needs a value");
                    std::process::exit(2);
                });
            }
            "--sysroot" => {
                i += 1;
                sysroot = Some(argv.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("--sysroot needs a path");
                    std::process::exit(2);
                }));
            }
            "-h" | "--help" => {
                println!("usage: sparagmos-driver <file.rs> [--dump <path>] [--edition <ed>] [--sysroot <path>]");
                return;
            }
            other if input.is_none() && !other.starts_with('-') => {
                input = Some(other.to_string());
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let Some(input) = input else {
        eprintln!("usage: sparagmos-driver <file.rs> [--dump <path>]");
        std::process::exit(2);
    };
    let sysroot = sysroot
        .or_else(|| std::env::var("SPARAGMOS_SYSROOT").ok())
        .unwrap_or_else(|| env!("SPARAGMOS_BUILD_SYSROOT").to_string());

    // Only the MIR is needed: emit metadata into a scratch directory so
    // running the driver never leaves build artifacts in the caller's
    // working directory.
    let out_dir = std::env::temp_dir().join(format!(
        "sparagmos-driver-out-{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&out_dir);
    let args: Vec<String> = vec![
        "sparagmos-driver".to_string(),
        input.clone(),
        format!("--edition={edition}"),
        "--crate-type=lib".to_string(),
        "--emit=metadata".to_string(),
        format!("--out-dir={}", out_dir.display()),
        format!("--sysroot={sysroot}"),
    ];
    let result = rustc_public::run!(args.as_slice(), collect);
    let collected = take();
    match (result, collected) {
        (Ok(()), Some(c)) => {
            println!(
                "sparagmos-driver: {input} (Stable MIR, {})",
                env!("SPARAGMOS_BUILD_RUSTC_VERSION")
            );
            for f in &c.report.functions {
                match &f.outcome {
                    FnOutcome::Converted => println!(
                        "fn {}: blocks={} locals={} args={} [converted]",
                        f.name, f.blocks, f.locals, f.arg_count
                    ),
                    FnOutcome::Skipped { reasons } => {
                        let rs: Vec<String> =
                            reasons.iter().map(|(k, v)| format!("{k} x{v}")).collect();
                        println!("fn {}: SKIPPED ({})", f.name, rs.join(", "));
                    }
                }
            }
            println!(
                "converted: {}, skipped: {}",
                c.report.converted_count(),
                c.report.skipped_count()
            );
            let totals = c.report.unsupported_totals();
            if !totals.is_empty() {
                println!("unsupported:");
                for (k, v) in &totals {
                    println!("  {k}: {v}");
                }
            }
            if let Some(path) = &dump_path {
                let text = sparagmos_driver::dump::render(&input, &c.report, &c.bodies, &c.def_names);
                if let Err(e) = std::fs::write(path, text) {
                    eprintln!("failed to write dump {path}: {e}");
                    std::process::exit(1);
                }
                println!("dump: {path}");
            }
        }
        (Ok(()), None) => {
            eprintln!("sparagmos-driver: internal error: no results collected");
            std::process::exit(1);
        }
        (Err(e), _) => {
            eprintln!("sparagmos-driver: compilation failed: {e:?}");
            std::process::exit(1);
        }
    }
}
