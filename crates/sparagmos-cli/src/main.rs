//! sparagmos CLI driver (first cut, Lane F).
//!
//! Subcommands:
//! - `sparagmos mir <file.rs> [--out <path>]` — emit textual MIR via the
//!   reference rustc (`rustup run 1.100.0 rustc --emit=mir`); prints to
//!   stdout, or saves to `--out` when given.
//! - `sparagmos check <file.rs>` — compile & run natively, show the
//!   expected result (stdout / stderr / exit code). This is the oracle the
//!   harness compares Sparagmos against; it does NOT run Sparagmos itself.
//! - `sparagmos corpus [--dir <path>] [--json]` — evaluate every program in
//!   `tests/corpus` and print the aggregate. Until the interpreter/JIT
//!   lanes are wired in, every case is honestly reported UNSUPPORTED.
//!
//! Exit codes: 0 = success, 1 = a check/corpus failure or native error,
//! 2 = usage error.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use sparagmos_harness::{
    compile_and_run_native, default_corpus_dir, emit_mir, evaluate_corpus, UnsupportedRunner,
};

fn usage() -> &'static str {
    "sparagmos — MIR-based JIT runtime (work in progress)\n\
     \n\
     USAGE:\n\
     \x20   sparagmos mir <file.rs> [--out <path>]\n\
     \x20   sparagmos check <file.rs>\n\
     \x20   sparagmos corpus [--dir <path>] [--json]\n\
     \x20   sparagmos --help\n"
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => code,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::from(2)
        }
    }
}

fn run(args: &[String]) -> Result<ExitCode, String> {
    let Some(cmd) = args.first() else {
        println!("{}", usage());
        return Ok(ExitCode::from(2));
    };
    match cmd.as_str() {
        "--help" | "-h" | "help" => {
            println!("{}", usage());
            Ok(ExitCode::SUCCESS)
        }
        "mir" => cmd_mir(&args[1..]),
        "check" => cmd_check(&args[1..]),
        "corpus" => cmd_corpus(&args[1..]),
        other => Err(format!("unknown subcommand '{other}'\n\n{}", usage())),
    }
}

/// Parse `--out <path>` / `--dir <path>` style options from the tail args.
fn opt_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn positional<'a>(args: &'a [String], what: &str) -> Result<&'a str, String> {
    args.iter()
        .find(|a| !a.starts_with("--"))
        .map(|s| s.as_str())
        .ok_or_else(|| format!("missing <{what}> argument"))
}

fn cmd_mir(args: &[String]) -> Result<ExitCode, String> {
    let file = positional(args, "file.rs")?;
    let mir = emit_mir(Path::new(file))?;
    if let Some(out) = opt_value(args, "--out") {
        std::fs::write(&out, &mir).map_err(|e| format!("failed to write {out}: {e}"))?;
        println!("wrote MIR for {file} to {out} ({} bytes)", mir.len());
    } else {
        print!("{mir}");
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_check(args: &[String]) -> Result<ExitCode, String> {
    let file = positional(args, "file.rs")?;
    match compile_and_run_native(Path::new(file)) {
        Ok(res) => {
            println!("file: {file}");
            println!("exit: {:?}", res.exit_code);
            println!("--- stdout ---\n{}", res.stdout);
            if !res.stderr.is_empty() {
                println!("--- stderr ---\n{}", res.stderr);
            }
            Ok(if res.success() {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            })
        }
        Err(detail) => {
            eprintln!("native check failed for {file}: {detail}");
            Ok(ExitCode::from(1))
        }
    }
}

fn cmd_corpus(args: &[String]) -> Result<ExitCode, String> {
    let dir: PathBuf = opt_value(args, "--dir")
        .map(PathBuf::from)
        .unwrap_or_else(default_corpus_dir);
    let json = args.iter().any(|a| a == "--json");
    // NOTE (connection point): replace `UnsupportedRunner` with the real
    // interpreter/JIT runner once Lane C/D expose one via `SparagmosRunner`.
    let report = evaluate_corpus(&dir, &UnsupportedRunner)
        .map_err(|e| format!("failed to read corpus dir {}: {e}", dir.display()))?;
    if json {
        print!("{}", report.to_json());
    } else {
        println!("corpus: {}", dir.display());
        for c in &report.cases {
            let native_exit = c
                .native
                .as_ref()
                .map(|r| format!("{:?}", r.exit_code))
                .unwrap_or_else(|| "-".into());
            println!(
                "  {:<20} {:<12} native_exit={}",
                c.case.name,
                c.status.label(),
                native_exit
            );
            if let sparagmos_harness::CaseStatus::Unsupported { reason } = &c.status {
                println!("  {:<20}   reason: {reason}", "");
            }
        }
        println!("{}", report.summary);
    }
    // Corpus "success" for the CLI means: harness itself worked and no case
    // *failed* / hit a native error. Unsupported is reported, not hidden,
    // but does not make the driver exit non-zero (nothing ran to fail yet).
    let bad = report.summary.failed + report.summary.native_error;
    Ok(if bad == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}
