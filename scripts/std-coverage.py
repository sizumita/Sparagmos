#!/usr/bin/env python3
"""Metric 3 (std feature coverage) extractor for Sparagmos.

The denominator of metric 3 is the set of `std` / `core` / `alloc`
function paths that actually appear as callees in the converted MIR of
the metric-2 programs: the own corpus (tests/corpus) plus the rustc
test-subset stage-1 candidates (exec, single-file, ui / ui-fulldeps,
3,699 files at tag 1.100.0). A path counts as covered when a call to it
is executable by the interpreter: the converted program contains a body
for it, or the interpreter resolves it as a builtin
(crates/sparagmos-interp/src/builtins.rs `lookup`).

For every input file the tool runs the driver in dump mode
(`sparagmos-driver <file.rs> --dump <path> --edition <ed>`, the edition
taken from the file's `//@ edition:` directive, default 2021 — the same
rule as scripts/rustc-subset-run.py) and parses the dump: the def-name
table gives every callee path, the `=== fn` sections give the set of
functions with converted bodies, and the `func` operand of each Call /
TailCall terminator gives the callees. Names are canonicalized exactly
like the interpreter does (lifetime-only generic groups removed), so
the measured set is the set of names the interpreter itself sees.

Files the driver cannot convert at the front end produce no dump;
their callees cannot be counted and they are reported separately.

Usage:
  python3 scripts/std-coverage.py \
      --results-json <subset-run-results.json> \
      --rust-dir <rust checkout> \
      --driver-bin <sparagmos-driver> \
      --out <results JSON>
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

DRIVER_TIMEOUT = 60
STD_PREFIXES = ("std", "core", "alloc")

DIRECTIVE_RE = re.compile(
    r"^\s*//@(?:\[([^\]]+)\])?\s*([a-zA-Z][a-zA-Z0-9-]*)"
    r"(?:\s*:\s*(.*?)|\s*\((.*)\))?\s*$")

DEF_NAME_RE = re.compile(r"^  DefId\(0:(\d+)\) = (.*)$", re.M)
FN_HEADER_RE = re.compile(r"^=== fn (.*) \(args=\d+, blocks=\d+, locals=\d+\)$",
                          re.M)
CALL_RE = re.compile(r"kind: (?:Call|TailCall) \{")
FNDEF_RE = re.compile(r"FnDef\(\s*DefId \{\s*krate: (\d+),\s*index: (\d+)")


# ---------------------------------------------------------------------------
# Name canonicalization (mirror of builtins::canonical_name)
# ---------------------------------------------------------------------------

def is_lifetime_list(inner):
    if not inner:
        return False
    for part in inner.split(","):
        part = part.strip()
        if part == "_":
            continue
        if part.startswith("'") and len(part) > 1:
            continue
        return False
    return True


def canonical_name(name):
    out = []
    rest = name
    while True:
        open_idx = rest.find("<")
        if open_idx == -1:
            break
        depth = 0
        close = None
        for i in range(open_idx, len(rest)):
            c = rest[i]
            if c == "<":
                depth += 1
            elif c == ">":
                depth -= 1
                if depth == 0:
                    close = i
                    break
        if close is None:
            break
        inner = rest[open_idx + 1:close]
        if is_lifetime_list(inner):
            out.append(rest[:open_idx])
            joined = "".join(out)
            if joined.endswith("::"):
                joined = joined[:-2]
            out = [joined]
            rest = rest[close + 1:]
        else:
            out.append(rest[:open_idx + 1])
            rest = rest[open_idx + 1:]
    out.append(rest)
    return "".join(out)


def is_std_path(canonical):
    first = canonical.split("::", 1)[0]
    return first in STD_PREFIXES


# ---------------------------------------------------------------------------
# Builtin registry (parsed from crates/sparagmos-interp/src/builtins.rs)
# ---------------------------------------------------------------------------

class BuiltinRegistry:
    """The interpreter's builtin support, read from builtins.rs.

    Exact names come from the match arms of `lookup`; the two guard
    rules (RangeInclusive::new, *::to_uppercase) are recognized by
    their condition text. Any other guard shape is a hard error so a
    future change to builtins.rs cannot silently drift this tool.
    """

    def __init__(self, source):
        start = source.find("pub fn lookup")
        if start == -1:
            raise ValueError("builtins.rs: pub fn lookup not found")
        end = source.find("\n}\n", start)
        if end == -1:
            raise ValueError("builtins.rs: end of lookup not found")
        body = source[start:end]
        self.exact = set(re.findall(r'"([^"\n]+)"\s*(?:\||=>)', body))
        if not self.exact:
            raise ValueError("builtins.rs: no exact builtin names found")
        guards = re.findall(r"^\s*_ if (.+?) =>", body, re.M)
        self.range_inclusive_new = False
        self.to_uppercase = False
        for g in guards:
            if 'canonical.contains("RangeInclusive")' in g \
                    and 'canonical.ends_with("::new")' in g:
                self.range_inclusive_new = True
            elif 'canonical.ends_with("::to_uppercase")' in g:
                self.to_uppercase = True
            else:
                raise ValueError(f"builtins.rs: unknown lookup guard: {g}")
        if len(guards) != 2:
            raise ValueError(
                f"builtins.rs: expected 2 lookup guards, found {len(guards)}")

    def lookup(self, canonical):
        if canonical in self.exact:
            return True
        if self.range_inclusive_new and "RangeInclusive" in canonical \
                and canonical.endswith("::new"):
            return True
        if self.to_uppercase and canonical.endswith("::to_uppercase"):
            return True
        return False


# ---------------------------------------------------------------------------
# Dump parsing
# ---------------------------------------------------------------------------

def parse_dump(text):
    """Parse one driver dump.

    Returns (def_names, body_names, callees, anomalies):
      def_names: {index: raw name}
      body_names: set of raw names with a converted body
      callees: list of raw callee names (one per Call/TailCall whose
               func operand is an FnDef constant)
      anomalies: list of strings (unresolvable DefIds etc.)
    """
    def_names = {int(m.group(1)): m.group(2) for m in DEF_NAME_RE.finditer(text)}
    body_names = set(FN_HEADER_RE.findall(text))
    callees = []
    anomalies = []
    for m in CALL_RE.finditer(text):
        seg = text[m.end():m.end() + 65536]
        args_pos = seg.find("args:")
        fndef = FNDEF_RE.search(seg)
        if fndef and (args_pos == -1 or fndef.start() < args_pos):
            krate, index = int(fndef.group(1)), int(fndef.group(2))
            if krate != 0 or index not in def_names:
                anomalies.append(
                    f"unresolvable callee DefId({krate}:{index})")
            else:
                callees.append(def_names[index])
    return def_names, body_names, callees, anomalies


def edition_of(text):
    """The file's unscoped `//@ edition:` value, default 2021
    (same rule as scripts/rustc-subset-run.py)."""
    for line in text.splitlines():
        m = DIRECTIVE_RE.match(line)
        if m and m.group(1) is None and m.group(2) == "edition":
            value = m.group(3) if m.group(3) is not None else (m.group(4) or "")
            return value.strip()
    return "2021"


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(task):
    path, source, driver_bin, timeout = task
    rec = {"path": path, "source": source, "status": "ok",
           "edition": None, "exit_code": None, "converted_fns": 0,
           "call_sites": 0, "std_paths": {}, "callees_without_body": [],
           "anomalies": [], "note": ""}
    p = Path(path)
    if not p.exists():
        rec["status"] = "missing"
        return rec
    try:
        text = p.read_text(errors="replace")
    except OSError as e:
        rec["status"] = "missing"
        rec["note"] = str(e)
        return rec
    rec["edition"] = edition_of(text)
    env = dict(os.environ, RUSTC_BOOTSTRAP="1")
    with tempfile.TemporaryDirectory(prefix="std-cov-") as td:
        dump = Path(td) / "out.dump"
        try:
            r = subprocess.run(
                [driver_bin, str(p), "--dump", str(dump),
                 "--edition", rec["edition"]],
                capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            rec["status"] = "timeout"
            return rec
        rec["exit_code"] = r.returncode
        if not dump.exists():
            rec["status"] = "no-dump"
            lines = (r.stderr or "").strip().splitlines()
            rec["note"] = lines[0][:200] if lines else ""
            return rec
        dump_text = dump.read_text(errors="replace")
    _defs, body_names, callees, anomalies = parse_dump(dump_text)
    rec["anomalies"] = anomalies
    rec["converted_fns"] = len(body_names)
    body_canonical = {canonical_name(n) for n in body_names}
    sites = Counter()
    with_body = {}
    no_body = set()
    for raw in callees:
        canon = canonical_name(raw)
        has_body = canon in body_canonical
        if is_std_path(canon):
            sites[canon] += 1
            with_body[canon] = with_body.get(canon, False) or has_body
        elif not has_body:
            no_body.add(canon)
        rec["call_sites"] += 1
    rec["std_paths"] = {name: {"sites": sites[name], "body": with_body[name]}
                        for name in sites}
    rec["callees_without_body"] = sorted(no_body)
    return rec


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(records, registry):
    paths = {}
    for rec in records:
        if rec["status"] != "ok":
            continue
        for name, info in rec["std_paths"].items():
            agg = paths.setdefault(name, {"path": name, "programs": 0,
                                          "call_sites": 0,
                                          "body_programs": 0,
                                          "builtin": registry.lookup(name)})
            agg["programs"] += 1
            agg["call_sites"] += info["sites"]
            if info["body"]:
                agg["body_programs"] += 1
    for agg in paths.values():
        agg["executable"] = agg["builtin"] or agg["body_programs"] >= 1
        agg["executable_strict"] = agg["builtin"] \
            or agg["body_programs"] == agg["programs"]
    ordered = sorted(paths.values(),
                     key=lambda a: (-a["programs"], a["path"]))
    denom = len(ordered)
    covered = sum(1 for a in ordered if a["executable"])
    covered_strict = sum(1 for a in ordered if a["executable_strict"])
    return {
        "denominator": denom,
        "executable": covered,
        "rate": (covered / denom) if denom else 0.0,
        "executable_strict": covered_strict,
        "rate_strict": (covered_strict / denom) if denom else 0.0,
        "paths": ordered,
    }


def stage1_candidates(results_json):
    """The stage-1 candidate paths from the subset runner results:
    exec, single-file, ui / ui-fulldeps (3,699 files at tag 1.100.0)."""
    d = json.loads(Path(results_json).read_text())
    return sorted(i["path"] for i in d["results"]
                  if i["nature"] == "exec" and i["single_file"]
                  and i["suite"] in ("ui", "ui-fulldeps"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-dir", type=Path, default=Path("tests/corpus"))
    ap.add_argument("--results-json", type=Path,
                    help="subset-run-results.json (stage-1 candidates)")
    ap.add_argument("--rust-dir", type=Path,
                    help="rust checkout the candidate paths are relative to")
    ap.add_argument("--driver-bin", required=True)
    ap.add_argument("--builtins-rs", type=Path,
                    default=Path("crates/sparagmos-interp/src/builtins.rs"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=DRIVER_TIMEOUT)
    ap.add_argument("--only", help="comma-separated path substrings to keep")
    args = ap.parse_args(argv)

    registry = BuiltinRegistry(args.builtins_rs.read_text())
    tasks = []
    for f in sorted(args.corpus_dir.glob("*.rs")):
        tasks.append((str(f), "corpus", str(args.driver_bin), args.timeout))
    if args.results_json:
        for rel in stage1_candidates(args.results_json):
            tasks.append((str(args.rust_dir / rel), "subset",
                          str(args.driver_bin), args.timeout))
    if args.only:
        keep = [s for s in args.only.split(",") if s]
        tasks = [t for t in tasks if any(s in t[0] for s in keep)]
    print(f"files: {len(tasks)} "
          f"(builtin exact names: {len(registry.exact)}, "
          f"guards: range_inclusive_new={registry.range_inclusive_new}, "
          f"to_uppercase={registry.to_uppercase})")

    records = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.jobs) as ex:
        for rec in ex.map(process_file, tasks):
            records.append(rec)
    records.sort(key=lambda r: (r["source"], r["path"]))

    agg = aggregate(records, registry)
    status_counts = Counter(r["status"] for r in records)
    summary = {
        "tool": "scripts/std-coverage.py",
        "files_total": len(records),
        "files_by_status": dict(status_counts),
        "files_without_dump": [r["path"] for r in records
                               if r["status"] != "ok"],
        "aggregate": agg,
        "files": records,
    }
    args.out.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"statuses: {dict(status_counts)}")
    print(f"denominator (unique std paths): {agg['denominator']}")
    print(f"executable: {agg['executable']} "
          f"({agg['rate'] * 100:.3f}%)")
    print(f"executable (strict, body in every program): "
          f"{agg['executable_strict']} ({agg['rate_strict'] * 100:.3f}%)")
    print("top unsupported paths (by programs):")
    shown = 0
    for a in agg["paths"]:
        if not a["executable"]:
            print(f"  {a['programs']:5d} programs  {a['path']}")
            shown += 1
            if shown >= 20:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
