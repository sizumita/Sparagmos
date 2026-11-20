#!/usr/bin/env python3
"""Classify and measure the exec-debugger category of metric 2(b).

Input is the per-test classification produced by
scripts/rustc-subset-survey.py (--json). The scope of this tool is the
170 tests the survey classified as nature="exec-debugger" (the
debuginfo suite) at the rust-lang/rust 1.100.0 tag.

Every debuginfo test's verdict is decided by a debugger session:
compiletest compiles the test file into a native executable carrying
debug info (the tests declare `//@ compile-flags: -g`), runs it under
GDB, LLDB or CDB (the Windows debugger) with a script assembled from
the file's `//@ <dbg>-command:` lines plus breakpoints at `#break`
lines, and requires every `//@ <dbg>-check:` line to appear, in
order, in the debugger's stdout (src/tools/compiletest/src/runtest/
debuginfo.rs and debugger.rs at the tag). The verdict material is
therefore the debugger's rendering of the program's state, which
depends on the native binary, the debug info rustc's codegen emitted
for it, the debugger itself, and rustc's pretty-printer /
debugger_tester scripts (src/etc). A `<dbg>-repr:` directive is a
pseudo-command intercepted by debugger_tester; it also implies a
check line ("<var>: Ok") in the debugger output.

The Sparagmos main path (driver -> Stable MIR -> interpreter/JIT)
produces none of those materials: it emits no native binary and no
debug info, and it has no debugger interface. No exec-debugger test
is therefore counted as a metric pass by this tool, and the
category's metric-2(b) pass count stays 0. What this tool measures
is:

1. A mechanical classification of all 170 tests: which debuggers
   each test scripts (gdb / lldb / cdb command sets), the verdict
   form (debugger-output comparison, debugger-session-only, or
   compile-and-load only), whether the verdict also exercises
   rustc's GDB pretty printers or the debugger_tester `repr`
   pseudo-commands, and the program shape (single file, revisions,
   FFI, compile-flags form).

2. For eligible tests, a *program-execution reference measurement*:
   the target program is run through the driver's run mode, and the
   outcome (runs to completion / panics / unsupported / driver
   error) is recorded. This decomposes the category into "the
   program itself executes on the main path" versus not. It is a
   diagnostic figure only, not a pass rate: even a program that
   runs to completion has no debugger session and no debugger
   output to compare, so the test does not pass.

Eligibility for the reference measurement follows the stage-1
runner's conventions (scripts/rustc-subset-run.py), reusing its
header parsing and host-applicability evaluation: single-file tests
with no revisions and no ignore-test, applicable to
x86_64-unknown-linux-gnu, whose compile-flags stay within the
debug-info flags the main path is structurally indifferent to
(`-g`, `-C debuginfo=*`, `-C collapse-macro-debuginfo=*`,
`--crate-name=*`): the main path produces no debug info at all, so
these flags cannot change what the driver does with the program.
Any other compile-flag makes the test ineligible, because the
driver run mode takes no extra rustc flags. Ineligible tests are
recorded as not attempted, with the first matching reason.

Usage:
    python3 scripts/rustc-subset-debugger.py \
        --survey-json subset-survey.json \
        --rust-dir ~/rust-1.100.0-survey \
        --driver-bin crates/sparagmos-driver/target/debug/sparagmos-driver \
        --out subset-debugger-results.json [--jobs 48] [--limit N]
"""

import argparse
import concurrent.futures
import importlib.util
import json
import sys
import time
from pathlib import Path

DEBUGGERS = ("gdb", "lldb", "cdb")

# ---------------------------------------------------------------------------
# Shared helpers from the stage-1 runner (same directory)
# ---------------------------------------------------------------------------


def load_stage1():
    path = Path(__file__).with_name("rustc-subset-run.py")
    spec = importlib.util.spec_from_file_location("rustc_subset_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Debugger-script classification
# ---------------------------------------------------------------------------


def debugger_scripts(header):
    """Per-debugger directive counts from a parsed header.

    Returns {dbg: {"command": n, "check": n, "repr": n}} for each
    debugger that has at least one directive. Both unscoped and
    revision-scoped directives count: a scoped directive still
    scripts that debugger, for one revision of the test. Only
    `//@` lines are directives at this tag (compiletest's
    COMPILETEST_DIRECTIVE_PREFIX), which is what the stage-1 Header
    parser recognises, so legacy `// gdb-command:` comments are
    ignored here exactly as compiletest ignores them.
    """
    out = {}
    for dbg in DEBUGGERS:
        counts = {"command": 0, "check": 0, "repr": 0}
        for kind in counts:
            name = f"{dbg}-{kind}"
            counts[kind] += len(header.unscoped.get(name, []))
            for (_rev, n), vals in header.scoped.items():
                if n == name:
                    counts[kind] += len(vals)
        if any(counts.values()):
            out[dbg] = counts
    return out


# verdict form -> the material the verdict is decided by
FORM_MATERIAL = {
    "debugger-output":
        "debugger stdout, matched in order against the test's "
        "*-check lines (the debugger's rendering of program state "
        "read from the binary's debug info)",
    "debugger-session-only":
        "the debugger session itself: the scripted commands must "
        "run without the debugger failing (the test asserts no "
        "output lines)",
    "compile-and-load-only":
        "compilation with debug info plus the debugger loading "
        "the binary (the test scripts no commands and asserts no "
        "output)",
}


def verdict_form(scripts):
    """Classify one debuginfo test by the material its verdict uses.

    A `-repr` directive counts as both a command and an implied
    check line, following DebuggerCommands::parse_from at the tag.
    """
    checks = sum(c["check"] + c["repr"] for c in scripts.values())
    commands = sum(c["command"] + c["repr"] for c in scripts.values())
    if checks:
        return "debugger-output"
    if commands:
        return "debugger-session-only"
    return "compile-and-load-only"


# ---------------------------------------------------------------------------
# Reference-measurement eligibility
# ---------------------------------------------------------------------------

# `-C` options that only shape debug info. The main path emits no
# debug info, so these cannot change what the driver does with the
# program; every other flag is a blocker.
DEBUGINFO_C_OPTS = ("debuginfo=", "collapse-macro-debuginfo=",
                    "collapse_macro_debuginfo=")


def flags_blocker(header):
    """Why the compile-flags leave the reference set, or None.

    Every compile-flags token must be `-g`, a debug-info `-C`
    option, or `--crate-name`; the first other token blocks.
    """
    tokens = []
    for v in header.values("compile-flags"):
        tokens.extend(v.split())
    bad = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-g":
            i += 1
        elif tok == "-C":
            opt = tokens[i + 1] if i + 1 < len(tokens) else ""
            if opt.startswith(DEBUGINFO_C_OPTS):
                i += 2
            else:
                bad.append(f"-C {opt}" if opt else "-C")
                i += 2 if opt else 1
        elif tok.startswith("-C") and tok[2:].startswith(DEBUGINFO_C_OPTS):
            i += 1
        elif tok.startswith("--crate-name="):
            i += 1
        elif tok == "--crate-name":
            i += 2
        else:
            bad.append(tok)
            i += 1
    if bad:
        return ("compile-flags beyond debug-info shaping ("
                + ", ".join(bad) + "); the driver run mode takes "
                "no extra rustc flags")
    return None


def ref_blocker(rec, header, stage1):
    """Why a debuginfo test is not attempted, or None.

    First matching reason wins, in the stage-1 runner's spirit.
    """
    if not rec["single_file"]:
        return "multi-file / auxiliary crates (driver runs a single file)"
    if header.has("revisions") or header.scoped:
        return ("revisions present (per-revision debugger scripts "
                "are not run)")
    if header.has("ignore-test"):
        return "ignore-test (compiletest itself never runs this test)"
    fb = flags_blocker(header)
    if fb:
        return fb
    status, why = stage1.applicability(header)
    if status in ("out-of-environment", "unevaluatable"):
        return why
    return None


# ---------------------------------------------------------------------------
# Per-test processing: classification first (cheap), driver run second
# ---------------------------------------------------------------------------


def classify(rec, rust_dir, stage1):
    """Build the result record for one exec-debugger test, without
    any driver run. Eligible tests come back with ref_status
    "eligible" and their edition filled in."""
    out = {
        "suite": rec["suite"],
        "path": rec["path"],
        "single_file": rec["single_file"],
        "ffi": rec["ffi"],
        "revisions": rec["revisions"],
        "debuggers": [],
        "scripts": {},
        "checks_total": 0,
        "commands_total": 0,
        "verdict_form": "",
        "verdict_material": "",
        "disable_gdb_pretty_printers": False,
        "repr_debuggers": [],
        "should_fail": False,
        "ignore_test": False,
        "only_cdb": False,
        "ignored_debuggers": [],
        "status": "not-attempted",
        "reason": "",
        "ref_status": "not-attempted",
        "ref_reason": "",
        "driver_exit": None,
        "seconds": 0.0,
    }
    path = rust_dir / rec["path"]
    text = path.read_text(errors="ignore")
    header = stage1.Header(text)

    scripts = debugger_scripts(header)
    out["scripts"] = scripts
    out["debuggers"] = sorted(scripts)
    out["checks_total"] = sum(c["check"] + c["repr"]
                              for c in scripts.values())
    out["commands_total"] = sum(c["command"] + c["repr"]
                                for c in scripts.values())
    out["repr_debuggers"] = sorted(d for d, c in scripts.items()
                                  if c["repr"])
    form = verdict_form(scripts)
    out["verdict_form"] = form
    out["verdict_material"] = FORM_MATERIAL[form]
    out["disable_gdb_pretty_printers"] = header.has(
        "disable-gdb-pretty-printers")
    out["should_fail"] = header.has("should-fail")
    out["ignore_test"] = header.has("ignore-test")
    out["only_cdb"] = header.has("only-cdb")
    out["ignored_debuggers"] = sorted(
        d for d in ("gdb", "lldb", "cdb") if header.has(f"ignore-{d}"))

    out["reason"] = (
        f"verdict material is {FORM_MATERIAL[form]}; the Sparagmos "
        "main path produces no native binary, no debug info and no "
        "debugger session")

    blocker = ref_blocker(rec, header, stage1)
    if blocker:
        out["ref_reason"] = blocker
        return out

    out["ref_status"] = "eligible"
    editions = header.values("edition")
    out["_edition"] = editions[0].strip() if editions else "2021"
    return out


def attempt(rec_out, rust_dir, driver_bin, stage1):
    """Run the driver for one eligible record and record how the
    target program itself fares on the main path."""
    t0 = time.time()
    try:
        res, kind, reason = stage1.sparagmos_side(
            rust_dir / rec_out["path"],
            rec_out.pop("_edition", "2021"), driver_bin)
        if kind == "result":
            rec_out["driver_exit"] = res["exit_code"]
            if res["exit_code"] == 0:
                rec_out["ref_status"] = "program-runs"
            else:
                rec_out["ref_status"] = "program-panics"
                rec_out["ref_reason"] = (
                    "guest program panicked (driver exit 101)")
        elif kind == "unsupported":
            rec_out["ref_status"] = "unsupported"
            rec_out["ref_reason"] = reason
        elif kind == "timeout":
            rec_out["ref_status"] = "timeout"
            rec_out["ref_reason"] = reason
        else:
            rec_out["ref_status"] = "driver-error"
            rec_out["ref_reason"] = reason
    except Exception as e:  # a runner bug must not lose the whole run
        rec_out["ref_status"] = "driver-error"
        rec_out["ref_reason"] = f"runner internal error: {e!r}"
    rec_out["seconds"] = round(time.time() - t0, 2)
    return rec_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--survey-json", type=Path, required=True,
                    help="JSON produced by rustc-subset-survey.py --json")
    ap.add_argument("--rust-dir", type=Path,
                    default=Path.home() / "rust-1.100.0-survey",
                    help="checkout of rust-lang/rust at the tag")
    ap.add_argument("--driver-bin", type=Path, required=True,
                    help="sparagmos-driver binary")
    ap.add_argument("--out", type=Path, required=True,
                    help="write per-test results and aggregates as JSON")
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0,
                    help="only attempt the first N eligible tests "
                         "(smoke runs); 0 = all")
    args = ap.parse_args()

    stage1 = load_stage1()
    survey = json.loads(args.survey_json.read_text())
    records = [r for r in survey["tests"]
               if r["nature"] == "exec-debugger"]
    denominator = survey["aggregates"]["denominator"]

    print(f"denominator: {denominator}; exec-debugger tests: "
          f"{len(records)}", flush=True)

    results = [classify(r, args.rust_dir, stage1) for r in records]
    eligible = [r for r in results if r["ref_status"] == "eligible"]
    if args.limit:
        for r in eligible[args.limit:]:
            r["ref_status"] = "not-attempted"
            r["ref_reason"] = "beyond --limit (smoke run)"
            r.pop("_edition", None)
        eligible = eligible[:args.limit]
    print(f"eligible tests to attempt: {len(eligible)}", flush=True)

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.jobs) as pool:
        futs = [pool.submit(attempt, r, args.rust_dir,
                            args.driver_bin, stage1) for r in eligible]
        for fut in concurrent.futures.as_completed(futs):
            fut.result()
    wall = time.time() - t0
    results.sort(key=lambda r: r["path"])

    def count(pred):
        return sum(1 for r in results if pred(r))

    by_form = {}
    for form in sorted({r["verdict_form"] for r in results}):
        by_form[form] = count(lambda r, f=form: r["verdict_form"] == f)
    by_debugger_set = {}
    for r in results:
        key = "+".join(r["debuggers"]) if r["debuggers"] else "none"
        by_debugger_set[key] = by_debugger_set.get(key, 0) + 1
    by_debugger_set = dict(sorted(by_debugger_set.items(),
                                  key=lambda kv: -kv[1]))

    attempted = [r for r in results
                 if r["ref_status"] in ("program-runs", "program-panics",
                                        "unsupported", "driver-error",
                                        "timeout")]
    runs = [r for r in attempted if r["ref_status"] == "program-runs"]

    def reason_top(rs, field):
        reasons = {}
        for r in rs:
            if r[field]:
                key = stage1.normalize_reason(r[field])
                reasons[key] = reasons.get(key, 0) + 1
        return sorted(reasons.items(), key=lambda kv: -kv[1])[:30]

    agg = {
        "denominator": denominator,
        "category_total": len(results),
        "metric_debugger_passed": 0,
        "metric_note":
            "No exec-debugger test is counted as a metric pass: "
            "every verdict in this category is decided by a debugger "
            "session over a native binary's debug info, which the "
            "Sparagmos main path does not produce. The "
            "program-execution figures below are diagnostics about "
            "the target programs, not pass rates.",
        "by_verdict_form": by_form,
        "by_debugger_set": by_debugger_set,
        "with_gdb_script": count(lambda r: "gdb" in r["scripts"]),
        "with_lldb_script": count(lambda r: "lldb" in r["scripts"]),
        "with_cdb_script": count(lambda r: "cdb" in r["scripts"]),
        "repr_users": count(lambda r: bool(r["repr_debuggers"])),
        "disable_gdb_pretty_printers": count(
            lambda r: r["disable_gdb_pretty_printers"]),
        "should_fail": count(lambda r: r["should_fail"]),
        "ignore_test": count(lambda r: r["ignore_test"]),
        "only_cdb": count(lambda r: r["only_cdb"]),
        "single_file": count(lambda r: r["single_file"]),
        "ffi": count(lambda r: r["ffi"]),
        "multi_revision": count(lambda r: r["revisions"] > 1),
        "reference": {
            "attempted": len(attempted),
            "program_runs": len(runs),
            "program_panics": count(
                lambda r: r["ref_status"] == "program-panics"),
            "unsupported": count(
                lambda r: r["ref_status"] == "unsupported"),
            "driver_error": count(
                lambda r: r["ref_status"] == "driver-error"),
            "timeout": count(lambda r: r["ref_status"] == "timeout"),
            "not_attempted": count(
                lambda r: r["ref_status"] == "not-attempted"),
            "runs_rate_percent":
                round(100.0 * len(runs) / len(attempted), 3)
                if attempted else 0.0,
            "top_unsupported_reasons": reason_top(
                [r for r in results if r["ref_status"] == "unsupported"],
                "ref_reason"),
        },
        "top_ref_not_attempted_reasons": reason_top(
            [r for r in results if r["ref_status"] == "not-attempted"],
            "ref_reason"),
        "wall_seconds": round(wall, 1),
        "jobs": args.jobs,
    }
    payload = {"aggregates": agg, "results": results}
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True))

    print(json.dumps(agg, indent=1, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
