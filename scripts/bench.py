#!/usr/bin/env python3
"""Performance baseline for Sparagmos: native vs interpreter vs JIT.

For every program in tests/bench/ this measures, on this machine:

* native   — `rustc 1.100.0 -O` binary: compile time (once) and
             per-run wall time of the compiled binary.
* driver conversion — `sparagmos-driver <file> --dump <path>` wall
             time (process start + Stable MIR conversion + dump
             write), and `sparagmos-driver run <file>` wall time
             (the whole interpreter path end to end, default limits).
* stages   — in-process stage times from scripts/bench-helper on the
             dump-parsed program: dump parse, MIR->IR conversion,
             interpreter execution (raised step budget; see
             docs/PERFORMANCE.md for why), JIT compilation, JIT
             execution.

Every figure reported is the median of several runs; the JSON output
keeps all samples plus min/max. Nothing is measured once and
reported as-is.

Usage:
    python3 scripts/bench.py [--out bench-results.json]
                             [--skip-build]
                             [--native-reps N] [--dump-reps N]
                             [--run-reps N] [--interp-reps N]
                             [--jit-reps N]

The full run takes minutes (the interpreter is slow); run it
detached, e.g. `nohup python3 scripts/bench.py > bench.log 2>&1 &`.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "crates" / "sparagmos-driver" / "target" / "release" / "sparagmos-driver"
HELPER = REPO / "scripts" / "bench-helper" / "target" / "release" / "sparagmos-bench-helper"
BENCH_DIR = REPO / "tests" / "bench"
TIMEOUT = 1800  # seconds per subprocess; interpreter L-runs are slow


def env() -> dict:
    e = dict(os.environ)
    e["PATH"] = str(Path.home() / ".cargo" / "bin") + os.pathsep + e.get("PATH", "")
    return e


def run_timed(cmd: list[str], **kw) -> tuple[float, subprocess.CompletedProcess]:
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT,
                       env=env(), **kw)
    return time.perf_counter() - t, p


def stats(samples: list[float]) -> dict:
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "samples": samples,
    }


def build() -> None:
    print("building driver (release) ...", flush=True)
    subprocess.run(
        ["cargo", "+1.100.0", "build", "--release"],
        cwd=REPO / "crates" / "sparagmos-driver",
        env={**env(), "RUSTC_BOOTSTRAP": "1"}, check=True,
        capture_output=True, text=True)
    print("building bench helper (release) ...", flush=True)
    subprocess.run(["cargo", "build", "--release"],
                   cwd=REPO / "scripts" / "bench-helper",
                   env=env(), check=True, capture_output=True, text=True)


def helper_kv(args: list[str]) -> dict[str, str]:
    p = subprocess.run([str(HELPER), *args], capture_output=True, text=True,
                       timeout=TIMEOUT, env=env())
    out = {}
    for line in p.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    if p.returncode != 0:
        out["helper_error"] = (p.stderr.strip().splitlines() or [""])[-1]
    return out


def samples_of(kv: dict[str, str], prefix: str) -> list[float]:
    vals = []
    i = 0
    while f"{prefix}_{i}" in kv:
        vals.append(float(kv[f"{prefix}_{i}"]))
        i += 1
    return vals


def measure_program(src: Path, work: Path, reps: dict) -> dict:
    name = src.stem
    r: dict = {"program": src.name}
    print(f"== {name}", flush=True)

    # --- native ---------------------------------------------------
    native_bin = work / f"{name}.native"
    t, p = run_timed(["rustup", "run", "1.100.0", "rustc", "--edition", "2021",
                      "-O", str(src), "-o", str(native_bin)])
    r["native_compile_s"] = t
    if p.returncode != 0:
        r["native_error"] = p.stderr.strip().splitlines()[-1] if p.stderr else "?"
        return r
    times = []
    for _ in range(reps["native"]):
        t, p = run_timed([str(native_bin)])
        assert p.returncode == 0, f"native {name} exited {p.returncode}"
        times.append(t)
    r["native_exec_s"] = stats(times)

    # --- driver: conversion (dump) --------------------------------
    dump = work / f"{name}.dump"
    times = []
    for _ in range(reps["dump"]):
        t, p = run_timed([str(DRIVER), str(src), "--dump", str(dump)])
        assert p.returncode == 0, f"driver dump {name} exited {p.returncode}: {p.stderr}"
        times.append(t)
    r["driver_dump_s"] = stats(times)

    # --- driver: interpreter end to end (default limits) ----------
    times, codes, note = [], [], ""
    for _ in range(reps["run"]):
        t, p = run_timed([str(DRIVER), "run", str(src)])
        times.append(t)
        codes.append(p.returncode)
        if p.returncode != 0 and not note:
            note = (p.stderr.strip().splitlines() or [""])[0]
    r["driver_run_s"] = stats(times)
    r["driver_run_exit_codes"] = codes
    if note:
        r["driver_run_note"] = note

    # --- helper: per-path status (default limits) ------------------
    kv = helper_kv([str(dump), "check"])
    r["parse_ms"] = float(kv["parse_ms"]) if "parse_ms" in kv else None
    r["interp_status"] = kv.get("interp_status", kv.get("helper_error", "?"))
    r["jit_status"] = kv.get("jit_status", "?")

    # --- helper: interpreter stages (raised budget) ----------------
    if r["interp_status"] == "ok" or "step limit" in r["interp_status"]:
        kv = helper_kv([str(dump), "interp", str(reps["interp"])])
        if "helper_error" in kv:
            r["interp_stage_error"] = kv["helper_error"]
        else:
            r["convert_ms"] = stats(samples_of(kv, "convert_ms"))
            r["interp_exec_ms"] = stats(samples_of(kv, "interp_ms"))

    # --- helper: JIT stages ----------------------------------------
    if r["jit_status"] == "ok":
        kv = helper_kv([str(dump), "jit", str(reps["jit"])])
        if "helper_error" in kv:
            r["jit_stage_error"] = kv["helper_error"]
        else:
            r["jit_compile_ms"] = stats(samples_of(kv, "jitc_ms"))
            r["jit_exec_ms"] = stats(samples_of(kv, "jit_ms"))
    return r


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench-results.json")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--native-reps", type=int, default=7)
    ap.add_argument("--dump-reps", type=int, default=5)
    ap.add_argument("--run-reps", type=int, default=3)
    ap.add_argument("--interp-reps", type=int, default=2)
    ap.add_argument("--jit-reps", type=int, default=5)
    args = ap.parse_args()
    reps = {"native": args.native_reps, "dump": args.dump_reps,
            "run": args.run_reps, "interp": args.interp_reps,
            "jit": args.jit_reps}

    if not args.skip_build:
        build()
    for b in (DRIVER, HELPER):
        if not b.is_file():
            print(f"missing binary: {b} (build first or drop --skip-build)",
                  file=sys.stderr)
            return 1

    _, rustc_v = run_timed(["rustup", "run", "1.100.0", "rustc", "--version"])
    _, head = run_timed(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "os": platform.platform(),
        "cpu": cpu_model(),
        "nproc": os.cpu_count(),
        "loadavg_start": os.getloadavg(),
        "rustc": rustc_v.stdout.strip(),
        "git_head": head.stdout.strip(),
        "reps": reps,
        "host_binaries": "driver and bench helper built with cargo --release; "
                         "native guests with rustc -O",
    }

    results = []
    with tempfile.TemporaryDirectory(prefix="sparagmos-bench-") as td:
        work = Path(td)
        for src in sorted(BENCH_DIR.glob("*.rs")):
            results.append(measure_program(src, work, reps))
    meta["loadavg_end"] = os.getloadavg()

    out = {"meta": meta, "results": results}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")

    # Compact summary (medians; ratios against the native run time).
    print("\n| program | native exec | interp exec (pure) | JIT exec (pure) "
          "| JIT compile | driver run total |")
    print("|---|---|---|---|---|---|")
    for r in results:
        def fmt(v, unit="s"):
            return f"{v:.4f}{unit}" if v is not None else "—"
        nat = r.get("native_exec_s", {}).get("median")
        ie = r.get("interp_exec_ms", {}).get("median")
        je = r.get("jit_exec_ms", {}).get("median")
        jc = r.get("jit_compile_ms", {}).get("median")
        dr = r.get("driver_run_s", {}).get("median")
        def with_ratio(ms):
            if ms is None or nat is None:
                return "—"
            return f"{ms / 1000:.4f}s ({ms / 1000 / nat:.1f}x)"
        print(f"| {r['program']} | {fmt(nat)} | {with_ratio(ie)} | "
              f"{with_ratio(je)} | {fmt(jc / 1000 if jc is not None else None)} "
              f"| {fmt(dr)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
