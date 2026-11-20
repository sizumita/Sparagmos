#!/usr/bin/env python3
"""Classify and measure the rustdoc category of metric 2(b).

Input is the per-test classification produced by
scripts/rustc-subset-survey.py (--json). The scope of this tool is the
1,735 tests the survey classified as nature="rustdoc-check"
(rustdoc-html 802, rustdoc-ui 440, rustdoc-json 194, rustdoc-gui 152,
rustdoc-js 79, rustdoc-js-std 68) at the rust-lang/rust 1.100.0 tag.

Every rustdoc test's verdict is decided by comparing *rustdoc output*:
generated HTML or JSON content, the search index exercised from
JavaScript, browser GUI scripts run against the generated pages, or
rustdoc's own diagnostic/output stream. The Sparagmos main path
(driver -> Stable MIR -> interpreter/JIT) produces none of those
materials: the driver is a rustc driver and never runs rustdoc's
passes (doc generation, intra-doc link resolution, doctest extraction
and execution, search-index generation). No rustdoc test is therefore
counted as a metric pass by this tool, and the rustdoc category's
metric-2(b) pass count stays 0. What this tool measures is:

1. A mechanical classification of all 1,735 tests by verdict material
   (per suite, and per verdict form inside rustdoc-ui), so the
   category's composition is on record with counts.

2. For rustdoc-ui only -- the one suite whose input is a compilable
   .rs file and whose expected verdict includes a compilation
   success/failure bit -- a *compilation-outcome agreement*
   measurement: the driver's convert mode (rustc compilation through
   the main path's front end) is run on each eligible test, and its
   success/failure is compared with the outcome compiletest expects
   from rustdoc (a declared `*-pass` directive means success; a
   declared `*-fail` directive, or no declaration -- UI mode defaults
   to check-fail -- means failure). Agreement is reported as a
   diagnostic figure only. It is not a pass rate: a rustdoc-ui verdict
   also requires rustdoc's diagnostic text to match the test's
   snapshot, which the main path does not produce, so an agreeing
   outcome bit does not make the test pass.

Eligibility for the agreement measurement follows the stage-1
runner's conventions (scripts/rustc-subset-run.py), reusing its
header parsing and host-applicability evaluation: single-file tests
with no revisions, no compile-flags, no rustc-env, no run-rustfix,
no ignore-test, no add-core-stubs, and no environment condition that
excludes or cannot be evaluated for x86_64-unknown-linux-gnu.
Ineligible tests are recorded as not attempted, with the first
matching reason.

Usage:
    python3 scripts/rustc-subset-rustdoc.py \
        --survey-json subset-survey.json \
        --rust-dir ~/rust-1.100.0-survey \
        --driver-bin crates/sparagmos-driver/target/debug/sparagmos-driver \
        --out subset-rustdoc-results.json [--jobs 48] [--limit N]
"""

import argparse
import concurrent.futures
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

DRIVER_TIMEOUT = 60

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
# Verdict-material classification
# ---------------------------------------------------------------------------

# Directive names whose presence means the verdict inspects generated
# output content. The survey's directive scan does not know the
# negated `!has` spelling, so content detection uses its own regex.
CONTENT_DIRECTIVE_RE = re.compile(
    r"//@(?:\[[^\]]*\])?\s*!?([a-zA-Z][a-zA-Z0-9-]*)")

HTML_CONTENT_DIRECTIVES = {
    "has", "has-not", "matches", "matches-not", "count", "snapshot",
    "files", "hasraw", "has-dir", "matchesraw",
}
JSON_CONTENT_DIRECTIVES = {
    "is", "set", "has", "count", "jq", "ismany", "has-dir",
}

# verdict form -> the material the verdict is decided by
FORM_MATERIAL = {
    "html-content":
        "generated HTML content (has/matches/count/snapshot directives)",
    "html-generation":
        "generated HTML (auxiliary docs; content is checked from "
        "other tests or generation must simply succeed)",
    "json-content":
        "rustdoc JSON content (is/set/has/count/jq directives)",
    "search-index-js":
        "rustdoc search index, asserted by the paired JavaScript file",
    "search-index-js-std":
        "prebuilt std rustdoc search index, asserted from JavaScript",
    "browser-gui":
        "generated HTML driven by a browser GUI script (.goml)",
    "doctest-report":
        "rustdoc --test doctest run report (stdout snapshot)",
    "coverage-output":
        "rustdoc --show-coverage output (stdout snapshot)",
    "stdout-snapshot":
        "rustdoc stdout snapshot",
    "diagnostics-snapshot":
        "rustdoc diagnostics (stderr snapshot and/or error patterns)",
    "exit-status-only":
        "rustdoc exit status (no output snapshot)",
}


def content_directives(text):
    return {m.group(1) for m in CONTENT_DIRECTIVE_RE.finditer(text)}


def verdict_form(rec, path, header, text):
    """Classify one rustdoc test by the material its verdict uses.

    `path` is the test file; `header` is the stage-1 Header of its
    text. Purely mechanical: suite first, then (for rustdoc-ui) the
    verdict form in a fixed priority order.
    """
    suite = rec["suite"]
    if suite == "rustdoc-html":
        if content_directives(text) & HTML_CONTENT_DIRECTIVES:
            return "html-content"
        return "html-generation"
    if suite == "rustdoc-json":
        return "json-content"
    if suite == "rustdoc-js":
        return "search-index-js"
    if suite == "rustdoc-js-std":
        return "search-index-js-std"
    if suite == "rustdoc-gui":
        return "browser-gui"
    if suite == "rustdoc-ui":
        flags = header.values("compile-flags")
        tokens = [tok for v in flags for tok in v.split()]
        if "--test" in tokens:
            return "doctest-report"
        if "--show-coverage" in tokens:
            return "coverage-output"
        if path.with_suffix(".stdout").exists():
            return "stdout-snapshot"
        if path.with_suffix(".stderr").exists() \
                or header.has("error-pattern"):
            return "diagnostics-snapshot"
        return "exit-status-only"
    raise ValueError(f"not a rustdoc suite: {suite}")


# ---------------------------------------------------------------------------
# rustdoc-ui compilation-outcome agreement
# ---------------------------------------------------------------------------

PASS_MODES = ("check-pass", "build-pass", "run-pass")
FAIL_MODES = ("check-fail", "build-fail", "run-fail")


def expected_outcome(header):
    """The compilation outcome compiletest expects from rustdoc.

    A declared `*-pass` directive means success. A declared `*-fail`
    directive means failure, and so does no declaration at all: in UI
    mode compiletest defaults the pass/fail mode to check-fail when a
    test declares none (src/tools/compiletest/src/directives.rs).
    Scoped (per-revision) modes never reach here; revisions are an
    eligibility blocker.
    """
    if any(header.has(m) for m in PASS_MODES):
        return "success"
    return "failure"


def ui_blocker(rec, header, stage1):
    """Why a rustdoc-ui test is not attempted, or None.

    First matching reason wins, in the stage-1 runner's spirit:
    the driver compiles one file with fixed arguments, so anything
    that changes the compilation setup is out of the attempted set.
    """
    if not rec["single_file"]:
        return "multi-file / auxiliary crates (driver compiles a single file)"
    if header.has("revisions") or header.scoped:
        return "revisions present (per-revision expectations are not run)"
    if header.has("compile-flags"):
        return ("compile-flags present (driver takes no extra "
                "rustc/rustdoc flags; this also excludes the --test "
                "doctest and --show-coverage forms)")
    if header.has("rustc-env"):
        return "rustc-env present (process environment is not modelled)"
    if header.has("run-rustfix"):
        return "run-rustfix (verdict includes applying rustfix suggestions)"
    if header.has("ignore-test"):
        return "ignore-test (compiletest itself never runs this test)"
    if header.has("add-core-stubs"):
        return "add-core-stubs (test links the minicore stub crate)"
    status, why = stage1.applicability(header)
    if status in ("out-of-environment", "unevaluatable"):
        return why
    return None


def judge_agreement(expected, compiled):
    """Compare the expected outcome bit with the driver's outcome."""
    got = "success" if compiled else "failure"
    return "outcome-agree" if got == expected else "outcome-disagree"


def driver_compile(stage1, driver_bin, path, edition):
    """Run the driver's convert mode. Returns (compiled, exit_code,
    first_error_line): compiled is True/False for exit 0/1, or None
    when the driver produced no compilation outcome (timeout or an
    unexpected exit code)."""
    res = stage1.run_cmd(
        [str(driver_bin), str(path), "--edition", edition],
        timeout=DRIVER_TIMEOUT, env_extra=stage1.BOOTSTRAP_ENV)
    if res["timeout"]:
        return None, None, f"driver timed out after {DRIVER_TIMEOUT}s"
    if res["exit_code"] == 0:
        return True, 0, ""
    if res["exit_code"] == 1:
        return False, 1, stage1.first_error_line(res["stderr"])
    return None, res["exit_code"], \
        f"driver exit {res['exit_code']}: {res['stderr'].strip()[:300]}"


# ---------------------------------------------------------------------------
# Per-test processing: classification first (cheap), driver run second
# ---------------------------------------------------------------------------


def classify(rec, rust_dir, stage1):
    """Build the result record for one rustdoc test, without any
    driver run. Eligible rustdoc-ui tests come back with status
    "eligible" and their expected outcome filled in."""
    out = {
        "suite": rec["suite"],
        "path": rec["path"],
        "single_file": rec["single_file"],
        "ffi": rec["ffi"],
        "verdict_form": "",
        "verdict_material": "",
        "status": "not-attempted",
        "reason": "",
        "expected": None,
        "driver_exit": None,
        "seconds": 0.0,
    }
    path = rust_dir / rec["path"]
    text = path.read_text(errors="ignore") if path.suffix == ".rs" else ""
    header = stage1.Header(text)
    form = verdict_form(rec, path, header, text)
    out["verdict_form"] = form
    out["verdict_material"] = FORM_MATERIAL[form]

    if rec["suite"] != "rustdoc-ui":
        out["reason"] = (
            f"verdict material is {FORM_MATERIAL[form]}; the Sparagmos "
            "main path does not produce rustdoc output")
        return out

    blocker = ui_blocker(rec, header, stage1)
    if blocker:
        out["reason"] = blocker
        return out

    out["status"] = "eligible"
    out["expected"] = expected_outcome(header)
    editions = header.values("edition")
    out["_edition"] = editions[0].strip() if editions else "2021"
    return out


def attempt(rec_out, rust_dir, driver_bin, stage1):
    """Run the driver for one eligible record and judge agreement."""
    t0 = time.time()
    try:
        compiled, exit_code, detail = driver_compile(
            stage1, driver_bin, rust_dir / rec_out["path"],
            rec_out.pop("_edition", "2021"))
        rec_out["driver_exit"] = exit_code
        if compiled is None:
            rec_out["status"] = "driver-error"
            rec_out["reason"] = detail
        else:
            rec_out["status"] = judge_agreement(
                rec_out["expected"], compiled)
            if rec_out["status"] == "outcome-disagree":
                rec_out["reason"] = (
                    f"expected rustdoc compilation "
                    f"{rec_out['expected']}, driver compilation "
                    f"{'succeeded' if compiled else 'failed'}"
                    + (f": {detail}" if detail else ""))
    except Exception as e:  # a runner bug must not lose the whole run
        rec_out["status"] = "driver-error"
        rec_out["reason"] = f"runner internal error: {e!r}"
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
                    help="only attempt the first N eligible rustdoc-ui "
                         "tests (smoke runs); 0 = all")
    args = ap.parse_args()

    stage1 = load_stage1()
    survey = json.loads(args.survey_json.read_text())
    records = [r for r in survey["tests"]
               if r["nature"] == "rustdoc-check" and not r["env_excluded"]]
    denominator = survey["aggregates"]["denominator"]

    print(f"denominator: {denominator}; rustdoc tests: {len(records)}",
          flush=True)

    results = [classify(r, args.rust_dir, stage1) for r in records]
    eligible = [r for r in results if r["status"] == "eligible"]
    if args.limit:
        for r in eligible[args.limit:]:
            r["status"] = "not-attempted"
            r["reason"] = "beyond --limit (smoke run)"
            r.pop("_edition", None)
        eligible = eligible[:args.limit]
    print(f"eligible rustdoc-ui tests to attempt: {len(eligible)}",
          flush=True)

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

    by_suite = {}
    for suite in sorted({r["suite"] for r in results}):
        rs = [r for r in results if r["suite"] == suite]
        by_suite[suite] = {
            "total": len(rs),
            "attempted": sum(1 for r in rs
                             if r["status"] in ("outcome-agree",
                                                "outcome-disagree",
                                                "driver-error")),
            "outcome_agree": sum(1 for r in rs
                                 if r["status"] == "outcome-agree"),
            "outcome_disagree": sum(1 for r in rs
                                    if r["status"] == "outcome-disagree"),
        }
    by_form = {}
    for form in sorted({r["verdict_form"] for r in results}):
        by_form[form] = count(lambda r, f=form: r["verdict_form"] == f)

    attempted = [r for r in results
                 if r["status"] in ("outcome-agree", "outcome-disagree",
                                    "driver-error")]
    agree = [r for r in attempted if r["status"] == "outcome-agree"]
    disagree = [r for r in attempted if r["status"] == "outcome-disagree"]
    decided = len(agree) + len(disagree)

    def agreement_block(rs):
        a = sum(1 for r in rs if r["status"] == "outcome-agree")
        d = sum(1 for r in rs if r["status"] == "outcome-disagree")
        return {"attempted": a + d, "agree": a, "disagree": d,
                "agreement_rate_percent":
                    round(100.0 * a / (a + d), 3) if (a + d) else 0.0}

    reasons = {}
    for r in results:
        if r["status"] == "not-attempted" and r["reason"]:
            key = stage1.normalize_reason(r["reason"])
            reasons[key] = reasons.get(key, 0) + 1
    top_reasons = sorted(reasons.items(), key=lambda kv: -kv[1])[:30]

    agg = {
        "denominator": denominator,
        "rustdoc_total": len(results),
        "metric_rustdoc_passed": 0,
        "metric_note":
            "No rustdoc test is counted as a metric pass: every "
            "verdict in this category is decided by rustdoc output, "
            "which the Sparagmos main path does not produce. The "
            "outcome-agreement figures below are diagnostics about "
            "the driver's front end, not pass rates.",
        "by_suite": by_suite,
        "by_verdict_form": by_form,
        "not_attempted": count(lambda r: r["status"] == "not-attempted"),
        "agreement": {
            "attempted": len(attempted),
            "agree": len(agree),
            "disagree": len(disagree),
            "driver_error": count(
                lambda r: r["status"] == "driver-error"),
            "agreement_rate_percent":
                round(100.0 * len(agree) / decided, 3) if decided else 0.0,
            "expected_success": agreement_block(
                [r for r in attempted if r["expected"] == "success"]),
            "expected_failure": agreement_block(
                [r for r in attempted if r["expected"] == "failure"]),
        },
        "wall_seconds": round(wall, 1),
        "jobs": args.jobs,
        "top_not_attempted_reasons": top_reasons,
    }
    payload = {"aggregates": agg, "results": results}
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True))

    print(json.dumps(agg, indent=1, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
