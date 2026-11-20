#!/usr/bin/env python3
"""Run a stage of the rustc test-suite subset for Sparagmos metric 2(b).

This is a measurement tool for the project's main path only: every test
is executed natively with the reference rustc 1.100.0 and through
`sparagmos-driver run` (Stable MIR driver -> sparagmos-mir ->
sparagmos-interp), and the two results are compared under the pass
definition in docs/COVERAGE.md (stdout and exit code must agree; for
run-fail tests the exit code must also match the expected failure
status). Nothing here is a second execution path: the Sparagmos side is
always the driver binary's `run` mode.

Input is the per-test classification produced by
scripts/rustc-subset-survey.py (--json). Stage 1 (this tool's current
scope) covers tests the survey classified as nature="exec" that are
single-file and in the ui / ui-fulldeps suites (the run-pass /
run-fail / run-crash population). Everything else in the denominator
is reported as not attempted, with a reason; unattempted and failed
tests alike count against the overall metric-2(b) rate, whose
denominator is always the full subset (24,937 at the 1.100.0 tag).

A test is *attempted* only when both sides can be run faithfully:

- its header carries an unscoped run-pass / run-fail directive
  (revisions, run-crash, compile-flags, run-flags, exec-env, stdin and
  add-core-stubs are recorded as not-attempted reasons in stage 1);
- no `ignore-*` condition matches this host and every `only-*`
  condition is satisfied (otherwise the test is out of environment
  for x86_64-unknown-linux-gnu and is recorded as such);
- `needs-*` conditions are either known to hold here or recorded as
  unevaluatable (not attempted);
- the native compile succeeds and the native run's exit code matches
  the mode's expectation (otherwise the native oracle itself does not
  behave as the test expects, recorded with the compiler output).

Both sides are compiled/run with RUSTC_BOOTSTRAP=1 so that tests
using unstable features behave the same way they do under
compiletest (which runs against a development compiler). The native
compile uses `--edition` from the test's `//@ edition:` directive,
defaulting to 2021 (compiletest's default for the ui suite), and
opt-level 0, matching the harness's corpus oracle.

Usage:
    python3 scripts/rustc-subset-run.py \
        --survey-json subset-survey.json \
        --rust-dir ~/rust-1.100.0-survey \
        --driver-bin crates/sparagmos-driver/target/debug/sparagmos-driver \
        --out subset-run-results.json [--jobs 48] [--limit N]
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TOOLCHAIN = "1.100.0"
DENOMINATOR = 24937  # metric 2(b) denominator at tag 1.100.0 (COVERAGE.md)

NATIVE_COMPILE_TIMEOUT = 60
NATIVE_RUN_TIMEOUT = 30
DRIVER_TIMEOUT = 60

# ---------------------------------------------------------------------------
# Header directive parsing
# ---------------------------------------------------------------------------

DIRECTIVE_RE = re.compile(
    r"^\s*//@(?:\[([^\]]+)\])?\s*([a-zA-Z][a-zA-Z0-9-]*)"
    r"(?:\s*:\s*(.*?)|\s*\((.*)\))?\s*$")

RUN_MODES = ("run-pass", "run-fail", "run-crash")


class Header:
    """Parsed `//@` directives of one test file (unscoped + scoped)."""

    def __init__(self, text):
        self.unscoped = {}   # name -> [values]
        self.scoped = {}     # (revision, name) -> [values]
        self.names = set()   # every directive name seen, scoped or not
        for line in text.splitlines():
            m = DIRECTIVE_RE.match(line)
            if not m:
                continue
            rev, name = m.group(1), m.group(2)
            value = m.group(3) if m.group(3) is not None                 else (m.group(4) or "")
            self.names.add(name)
            if rev is None:
                self.unscoped.setdefault(name, []).append(value)
            else:
                self.scoped.setdefault((rev, name), []).append(value)

    def has(self, name):
        return name in self.unscoped

    def values(self, name):
        return self.unscoped.get(name, [])


def run_mode(header):
    """The unscoped run mode of a test, or None."""
    for mode in RUN_MODES:
        if header.has(mode):
            return mode
    return None


# ---------------------------------------------------------------------------
# Environment applicability (host: x86_64-unknown-linux-gnu)
# ---------------------------------------------------------------------------

# cfg atoms that are true for the measurement host. Directive suffixes
# are matched against this set the way compiletest matches them
# against the target's cfg list (an `ignore-<x>` skips the test when
# <x> matches; an `only-<x>` keeps the test only when <x> matches).
HOST_TRUE_CFGS = {
    "linux", "unix", "x86_64", "x86_64-unknown-linux-gnu",
    "target_os=\"linux\"", "target_arch=\"x86_64\"",
    "target_family=\"unix\"", "target_env=\"gnu\"",
    "target_vendor=\"unknown\"", "target_pointer_width=\"64\"",
    "target_endian=\"little\"", "64bit", "gnu",
}

# `needs-*` conditions known to hold on the measurement host.
NEEDS_SATISFIED = {
    "needs-run-enabled",   # we do run programs
    "needs-unwind",        # the reference toolchain is built with unwinding
}

# `ignore-*` suffixes that never match any host in compiletest's model
# (they gate on the pass mode rather than the environment); for the
# run modes measured here they do not apply.
IGNORE_NEVER_MATCH = {"ignore-pass", "ignore-check", "ignore-pretty"}


def eval_cfg_atom(atom):
    """Evaluate one cfg atom against the host. None = cannot evaluate."""
    atom = atom.strip()
    if atom in HOST_TRUE_CFGS:
        return True
    if atom in ("windows", "macos", "win32", "32bit"):
        return False
    m = re.match(r'^(target_[a-z_]+)\s*=\s*"([^"]*)"$', atom)
    if m:
        return f'{m.group(1)}="{m.group(2)}"' in HOST_TRUE_CFGS
    if re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", atom):
        # A bare cfg name that is not in the host set is false for this
        # host (other OS/arch/env names, stage/channel names handled
        # by callers where relevant).
        return False
    return None


def eval_condition(cond):
    """Evaluate an ignore-/only- condition suffix.

    Returns True (matches this host), False (does not match), or None
    (cannot be evaluated mechanically here).
    """
    cond = cond.strip()
    if not cond:
        return None
    if cond in IGNORE_NEVER_MATCH:
        return False
    if cond.startswith("cfg(") and cond.endswith(")"):
        return eval_cfg_expr(cond[4:-1])
    return eval_cfg_atom(cond)


def eval_cfg_expr(expr):
    """Evaluate a cfg(...) body with all/any/not over simple atoms."""
    expr = expr.strip()
    for fn, combine in (("all", all), ("any", any)):
        if expr.startswith(fn + "(") and expr.endswith(")"):
            parts = split_top_level(expr[len(fn) + 1:-1])
            vals = [eval_cfg_expr(p) for p in parts]
            if any(v is None for v in vals):
                return None
            return combine(vals)
    if expr.startswith("not(") and expr.endswith(")"):
        v = eval_cfg_expr(expr[4:-1])
        return None if v is None else not v
    return eval_cfg_atom(expr)


def split_top_level(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def applicability(header):
    """Decide whether a test applies to this host.

    Returns (status, reason): status is "applicable",
    "out-of-environment" (an ignore matched or an only did not), or
    "unevaluatable" (a needs-/condition could not be decided).
    """
    for value in header.values("ignore-cfg"):
        v = eval_cfg_expr(value)
        if v is True:
            return ("out-of-environment",
                    f"ignore-cfg({value}) matches this host")
        if v is None:
            return ("unevaluatable",
                    f"ignore-cfg({value}) cannot be evaluated")
    only_cfgs = header.values("only-cfg")
    if only_cfgs:
        vals = [eval_cfg_expr(v) for v in only_cfgs]
        if any(v is None for v in vals):
            return ("unevaluatable",
                    "only-cfg condition cannot be evaluated: "
                    + ", ".join(only_cfgs))
        if not any(vals):
            return ("out-of-environment",
                    "no only-cfg condition matches this host: "
                    + ", ".join(only_cfgs))
    for name in header.names:
        if name in ("ignore-cfg", "only-cfg"):
            continue
        if name.startswith("ignore-"):
            cond = name[len("ignore-"):]
            v = eval_condition(cond)
            if v is True:
                return ("out-of-environment", f"ignore-{cond} matches this host")
            if v is None:
                return ("unevaluatable", f"ignore-{cond} cannot be evaluated")
    onlys = [n[len("only-"):] for n in header.names if n.startswith("only-")]
    if onlys:
        vals = [eval_condition(c) for c in onlys]
        if any(v is None for v in vals):
            return ("unevaluatable",
                    "only-* condition cannot be evaluated: " + ", ".join(onlys))
        if not any(vals):
            return ("out-of-environment",
                    "no only-* condition matches this host: " + ", ".join(onlys))
    for name in sorted(header.names):
        if name.startswith("needs-"):
            if name in NEEDS_SATISFIED:
                continue
            return ("unevaluatable", f"{name} cannot be verified on this host")
    return ("applicable", "")


# ---------------------------------------------------------------------------
# Verdict logic (pure; unit-tested with fixtures)
# ---------------------------------------------------------------------------

def expected_exit(mode, header):
    """Exit code the mode expects from the program run."""
    if mode == "run-pass":
        return 0
    if mode == "run-fail":
        for v in header.values("failure-status"):
            try:
                return int(v.strip())
            except ValueError:
                pass
        return 101
    return None  # run-crash: no fixed expectation in stage 1


def judge(mode, header, native, sparagmos):
    """Compare native and Sparagmos results for one attempted test.

    `native` / `sparagmos` are dicts with exit_code (int or None) and
    stdout (str). Returns (status, detail): status is "pass" or
    "fail".
    """
    exp = expected_exit(mode, header)
    if native["exit_code"] != exp:
        return ("fail",
                f"native exit {native['exit_code']} != expected {exp}")
    if sparagmos["exit_code"] != native["exit_code"]:
        return ("fail",
                f"exit mismatch: native {native['exit_code']}, "
                f"sparagmos {sparagmos['exit_code']}")
    if sparagmos["stdout"] != native["stdout"]:
        return ("fail", "stdout mismatch")
    return ("pass", "")


def stage1_blocker(header, survey_rec):
    """Why a survey exec/single-file record is not attempted, or None.

    Checks that depend only on the header and the survey record
    (environment applicability is checked separately).
    """
    if header.has("revisions") or any(
            rev is not None for (rev, _n) in header.scoped):
        return "revisions present (per-revision cfgs not run in stage 1)"
    mode = run_mode(header)
    if mode is None:
        return "no unscoped run-pass/run-fail directive"
    if mode == "run-crash":
        return "run-crash mode (crash reproduction is out of stage-1 scope)"
    if header.has("compile-flags"):
        return "compile-flags present (driver run mode takes no extra rustc flags)"
    if header.has("run-flags"):
        return "run-flags present (guest program arguments are not supported)"
    if header.has("exec-env"):
        return "exec-env present (guest environment is not modelled)"
    if header.has("stdin"):
        return "stdin directive present (guest stdin is not modelled)"
    if header.has("add-core-stubs"):
        return "add-core-stubs (test links the minicore stub crate)"
    if header.has("ignore-test"):
        return "ignore-test (compiletest itself never runs this test)"
    return None


# ---------------------------------------------------------------------------
# Process execution
# ---------------------------------------------------------------------------

def run_cmd(argv, *, timeout, cwd=None, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    try:
        p = subprocess.run(argv, capture_output=True, timeout=timeout,
                           cwd=cwd, env=env, stdin=subprocess.DEVNULL)
        return {
            "exit_code": p.returncode,
            "stdout": p.stdout.decode("utf-8", errors="replace"),
            "stderr": p.stderr.decode("utf-8", errors="replace"),
            "timeout": False,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "exit_code": None,
            "stdout": (e.stdout or b"").decode("utf-8", errors="replace"),
            "stderr": (e.stderr or b"").decode("utf-8", errors="replace"),
            "timeout": True,
        }


def rustc_cmd():
    home = Path.home()
    rustup = home / ".cargo" / "bin" / "rustup"
    if rustup.exists():
        return [str(rustup), "run", TOOLCHAIN, "rustc"]
    return ["rustup", "run", TOOLCHAIN, "rustc"]


BOOTSTRAP_ENV = {"RUSTC_BOOTSTRAP": "1"}

_SYSROOT_LIB = None


def sysroot_lib():
    """The reference toolchain lib dir (libLLVM, librustc_driver).

    ui-fulldeps tests link librustc_driver and its LLVM library; the
    plain rustc invocation finds the crates but not -lLLVM, so the
    native compile gets this dir as -L and the native run gets it in
    LD_LIBRARY_PATH, the same setup compiletest arranges.
    """
    global _SYSROOT_LIB
    if _SYSROOT_LIB is None:
        try:
            out = subprocess.run(
                rustc_cmd() + ["--print", "sysroot"],
                capture_output=True, text=True, timeout=30)
            _SYSROOT_LIB = str(Path(out.stdout.strip()) / "lib")
        except Exception:
            _SYSROOT_LIB = ""
    return _SYSROOT_LIB


def native_side(path, edition, workdir):
    """Compile and run natively. Returns (result_dict, error_reason).

    error_reason is None when the native run completed (whatever its
    exit code); otherwise a not-attempted reason string.
    """
    binpath = Path(workdir) / "prog"
    lib = sysroot_lib()
    build = run_cmd(
        rustc_cmd() + ["--edition", edition, "-C", "opt-level=0"]
        + (["-L", lib] if lib else [])
        + ["-o", str(binpath), str(path)],
        timeout=NATIVE_COMPILE_TIMEOUT, env_extra=BOOTSTRAP_ENV)
    if build["timeout"]:
        return None, "native compile timed out"
    if build["exit_code"] != 0:
        first = first_error_line(build["stderr"])
        return None, f"native compile failed: {first}"
    run_env = {}
    if lib:
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        run_env["LD_LIBRARY_PATH"] = lib + (":" + prev if prev else "")
    run = run_cmd([str(binpath)], timeout=NATIVE_RUN_TIMEOUT,
                  cwd=str(workdir), env_extra=run_env)
    if run["timeout"]:
        return None, "native run timed out"
    return run, None


def sparagmos_side(path, edition, driver_bin, cwd=None):
    """Run through the driver. Returns (result_dict, kind, reason).

    kind is "result" (exit 0/101), "unsupported" (driver exit 2),
    "timeout", or "driver-error".
    """
    res = run_cmd([str(driver_bin), "run", str(path),
                   "--edition", edition],
                  timeout=DRIVER_TIMEOUT, cwd=cwd,
                  env_extra=BOOTSTRAP_ENV)
    if res["timeout"]:
        return None, "timeout", f"driver timed out after {DRIVER_TIMEOUT}s"
    if res["exit_code"] in (0, 101):
        return res, "result", ""
    if res["exit_code"] == 2:
        return None, "unsupported", driver_reason(res["stderr"])
    return None, "driver-error", \
        f"driver exit {res['exit_code']}: {res['stderr'].strip()[:300]}"


def first_error_line(stderr):
    for line in stderr.splitlines():
        line = line.strip()
        if line.startswith("error"):
            return line[:200]
    return (stderr.strip().splitlines() or ["<no output>"])[0][:200]


def driver_reason(stderr):
    lines = [l.strip() for l in stderr.splitlines()
             if "sparagmos-driver run:" in l]
    if lines:
        return "; ".join(lines)[:400]
    return ("driver reported unsupported (exit 2) without a reason line: "
            + stderr.strip()[:300])


# ---------------------------------------------------------------------------
# Reason categorisation for aggregates
# ---------------------------------------------------------------------------

def normalize_reason(reason):
    """Strip volatile details so equal causes group together."""
    r = reason
    r = re.sub(r"/[^\s:'\"]+", "<path>", r)
    r = re.sub(r"\b\d+\b", "<n>", r)
    r = re.sub(r"`[^`]*`", "`<name>`", r)
    return r


def bucket_reason(reason):
    """Coarse bucket for a driver/attempt reason string."""
    r = reason.lower()
    if "no converted body" in r or "calls to functions" in r:
        return "call to function with no converted body (std/core not modelled)"
    if "unsupported type in local" in r:
        return "unsupported type in local (interp type model)"
    if "no monomorphic instance" in r:
        return "no monomorphic instance available"
    if "conversion failed" in r or "mir conversion" in r:
        return "MIR conversion failed (other)"
    if "uninitialized" in r:
        return "read of uninitialized place"
    if "type mismatch" in r:
        return "type mismatch in interp conversion"
    if "timeout" in r or "timed out" in r:
        return "timeout"
    if "native compile failed" in r:
        return "native compile failed"
    return "other"


# ---------------------------------------------------------------------------
# Per-test driver
# ---------------------------------------------------------------------------

def process_test(rec, rust_dir, driver_bin):
    """Produce the result record for one survey record."""
    out = {
        "suite": rec["suite"],
        "path": rec["path"],
        "nature": rec["nature"],
        "single_file": rec["single_file"],
        "ffi": rec["ffi"],
        "status": "not-attempted",
        "reason": "",
        "bucket": "",
        "mode": None,
        "seconds": 0.0,
    }
    t0 = time.time()
    try:
        fill_result(rec, rust_dir, driver_bin, out)
    except Exception as e:  # a runner bug must not lose the whole run
        out["status"] = "not-attempted"
        out["reason"] = f"runner internal error: {e!r}"
        out["bucket"] = "runner internal error"
    out["seconds"] = round(time.time() - t0, 2)
    return out


def fill_result(rec, rust_dir, driver_bin, out):
    if rec["nature"] != "exec":
        out["reason"] = f"nature={rec['nature']} (not in stage-1 scope)"
        out["bucket"] = "out of stage-1 scope (nature)"
        return
    if rec["suite"] not in ("ui", "ui-fulldeps"):
        out["reason"] = (f"suite={rec['suite']} (exec outside ui/ui-fulldeps "
                         "needs suite-specific flags, e.g. coverage "
                         "instrumentation; not in stage-1 scope)")
        out["bucket"] = "out of stage-1 scope (suite)"
        return
    if not rec["single_file"]:
        out["reason"] = "multi-file / auxiliary crates (not in stage-1 scope)"
        out["bucket"] = "out of stage-1 scope (multi-file)"
        return

    path = rust_dir / rec["path"]
    text = path.read_text(errors="ignore")
    header = Header(text)
    mode = run_mode(header)
    out["mode"] = mode

    blocker = stage1_blocker(header, rec)
    if blocker:
        out["reason"] = blocker
        out["bucket"] = "not attempted (header feature)"
        return
    status, why = applicability(header)
    if status == "out-of-environment":
        out["reason"] = why
        out["bucket"] = "out of environment"
        return
    if status == "unevaluatable":
        out["reason"] = why
        out["bucket"] = "not attempted (unevaluatable condition)"
        return

    editions = header.values("edition")
    edition = editions[0].strip() if editions else "2021"

    with tempfile.TemporaryDirectory(prefix="subset-run-") as td:
        native, err = native_side(path, edition, td)
        if err:
            out["reason"] = err
            out["bucket"] = bucket_reason(err)
            return
        exp = expected_exit(mode, header)
        if native["exit_code"] != exp:
            out["reason"] = (f"native run exit {native['exit_code']} does "
                              f"not match the mode expectation {exp}")
            out["bucket"] = "native run does not match mode expectation"
            out["native"] = {"exit_code": native["exit_code"]}
            return
        got, kind, reason = sparagmos_side(path, edition, driver_bin, td)
        if kind == "unsupported":
            out["status"] = "unsupported"
            out["reason"] = reason
            out["bucket"] = bucket_reason(reason)
            return
        if kind in ("timeout", "driver-error"):
            out["status"] = "fail"
            out["reason"] = reason
            out["bucket"] = bucket_reason(reason)
            return
        verdict, detail = judge(mode, header, native, got)
        out["status"] = "pass" if verdict == "pass" else "fail"
        if verdict != "pass":
            out["reason"] = detail
            out["bucket"] = ("stdout mismatch"
                             if detail == "stdout mismatch"
                             else "exit-code mismatch")
            out["native"] = {"exit_code": native["exit_code"],
                             "stdout": native["stdout"][:500]}
            out["sparagmos"] = {"exit_code": got["exit_code"],
                                "stdout": got["stdout"][:500]}



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
                    help="only process the first N stage-1 candidates "
                         "(smoke runs); 0 = all")
    args = ap.parse_args()

    survey = json.loads(args.survey_json.read_text())
    records = survey["tests"]
    denominator = survey["aggregates"]["denominator"]

    # Stage-1 candidates are processed for real; every other record in
    # the denominator is carried through as not-attempted so the
    # aggregate arithmetic covers the whole subset.
    def is_candidate(r):
        return (r["nature"] == "exec" and r["single_file"]
                and r["suite"] in ("ui", "ui-fulldeps")
                and not r["env_excluded"])

    all_candidates = [r for r in records if is_candidate(r)]
    candidates = all_candidates[:args.limit] if args.limit         else all_candidates
    limited_out = all_candidates[len(candidates):]
    others = [r for r in records
              if not is_candidate(r) and not r["env_excluded"]]

    print(f"denominator: {denominator}; stage-1 candidates: "
          f"{len(candidates)}; other records carried as not-attempted: "
          f"{len(others)}", flush=True)

    results = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.jobs) as pool:
        futs = [pool.submit(process_test, r, args.rust_dir,
                            args.driver_bin) for r in candidates]
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(candidates)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    wall = time.time() - t0

    for r in others:
        results.append({
            "suite": r["suite"], "path": r["path"],
            "nature": r["nature"], "single_file": r["single_file"],
            "ffi": r["ffi"], "status": "not-attempted",
            "reason": "not in stage-1 scope (see bucket)",
            "bucket": ("out of stage-1 scope (nature)"
                       if r["nature"] != "exec"
                       else "out of stage-1 scope (multi-file)"),
            "mode": None, "seconds": 0.0,
        })

    for r in limited_out:
        results.append({
            "suite": r["suite"], "path": r["path"],
            "nature": r["nature"], "single_file": r["single_file"],
            "ffi": r["ffi"], "status": "not-attempted",
            "reason": "beyond --limit (smoke run)",
            "bucket": "not attempted (smoke-run limit)",
            "mode": None, "seconds": 0.0,
        })

    results.sort(key=lambda r: r["path"])
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    unsup = sum(1 for r in results if r["status"] == "unsupported")
    notatt = sum(1 for r in results if r["status"] == "not-attempted")

    buckets = {}
    for r in results:
        if r["status"] != "pass":
            buckets[r["bucket"]] = buckets.get(r["bucket"], 0) + 1
    reasons = {}
    for r in results:
        if r["status"] in ("unsupported", "not-attempted", "fail") \
                and r["reason"]:
            key = normalize_reason(r["reason"])
            reasons[key] = reasons.get(key, 0) + 1
    top_reasons = sorted(reasons.items(), key=lambda kv: -kv[1])[:30]

    cand_paths = {r["path"] for r in candidates}
    cand = [r for r in results if r["path"] in cand_paths]

    def rate_block(rs):
        tot = len(rs)
        ps = sum(1 for r in rs if r["status"] == "pass")
        return {"total": tot, "passed": ps,
                "pass_rate_percent":
                    round(100.0 * ps / tot, 3) if tot else 0.0}

    by_nature = {}
    for nature in sorted({r["nature"] for r in results}):
        by_nature[nature] = rate_block(
            [r for r in results if r["nature"] == nature])
    by_suite = {}
    for suite in sorted({r["suite"] for r in results}):
        by_suite[suite] = rate_block(
            [r for r in results if r["suite"] == suite])
    agg = {
        "denominator": denominator,
        "results_total": len(results),
        "passed": passed,
        "failed": failed,
        "unsupported": unsup,
        "not_attempted": notatt,
        "metric_2b_pass_rate_percent":
            round(100.0 * passed / denominator, 3) if denominator else 0.0,
        "stage1_candidates": len(candidates),
        "stage1_passed": sum(1 for r in cand if r["status"] == "pass"),
        "out_of_environment": sum(
            1 for r in results if r["bucket"] == "out of environment"),
        "exec_category": by_nature.get("exec",
                                       {"total": 0, "passed": 0,
                                        "pass_rate_percent": 0.0}),
        "by_nature": by_nature,
        "by_suite": by_suite,
        "wall_seconds": round(wall, 1),
        "jobs": args.jobs,
        "buckets": dict(sorted(buckets.items(), key=lambda kv: -kv[1])),
        "top_reasons": top_reasons,
    }
    payload = {"aggregates": agg, "results": results}
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True))

    print(json.dumps(agg, indent=1, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
