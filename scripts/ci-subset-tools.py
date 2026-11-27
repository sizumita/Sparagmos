#!/usr/bin/env python3
"""Helpers for the CI subset-measurement workflows.

Subcommands:

- filter: write a survey JSON containing only the tests whose path
  contains at least one of the given comma-separated keywords. Used by
  the flip-diff workflow to build a topical population.
- shard: write shard <index> of <count> of a survey JSON. Tests are
  sorted by path and dealt round-robin so every shard carries a mix of
  suites and stages (a contiguous block would concentrate the heavy
  stage-1 records in a few shards).
- partition: split a survey JSON into one survey per host class for
  the host-partitioned official measurement (issue #517). Each record
  is assigned to the first class whose host model -- the runner's own
  applicability evaluation under that class's host profile -- admits
  it, or to the first (residual) class when no class admits it, so
  the per-class surveys are disjoint and cover the survey exactly
  once (the command refuses to write anything otherwise).
- merge: combine several subset result JSONs (one per shard) into a
  single result JSON whose aggregates are recomputed from the merged
  records, so the merged file is a drop-in replacement for a result
  file produced by one unsharded run. With --expect-total /
  --expect-nature (and optionally --survey) the merge additionally
  asserts that the denominator matches the expected population, and
  fails without writing output when it does not; the official
  measurement workflow always passes these options, while partial
  populations (flip-diff etc.) merge without them.
- check-candidates: assert that the stage-1 candidate set a metric-3
  result (scripts/std-coverage.py --survey-json) was measured over is
  the set the merged subset results define. The official workflow
  measures metric 3 from the survey, alongside the subset run, and
  this check fails the aggregate when the two sets ever diverge.
- diff: compare two result JSONs (base driver vs head driver) and
  report gains and losses. A record passes when its bucket is empty;
  the comparison key is path + nature, matching the flip-diff
  definition the lanes use on the dev host.

Only the Python standard library is used.
"""

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")


def rate(passed, total):
    if total == 0:
        return 0.0
    return round(100.0 * passed / total, 3)


# ---------------------------------------------------------------------------
# filter / shard
# ---------------------------------------------------------------------------

def cmd_filter(args):
    survey = load_json(args.survey)
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    if not keywords:
        print("error: no keywords given", file=sys.stderr)
        return 2
    kept = [t for t in survey["tests"]
            if any(k in t["path"] for k in keywords)]
    out = dict(survey)
    out["tests"] = kept
    write_json(args.out, out)
    print(f"filter: {len(kept)} of {len(survey['tests'])} tests kept "
          f"for keywords {keywords}")
    return 0


def cmd_shard(args):
    survey = load_json(args.survey)
    if not (0 <= args.index < args.count):
        print(f"error: shard index {args.index} out of range "
              f"for count {args.count}", file=sys.stderr)
        return 2
    tests = sorted(survey["tests"], key=lambda t: t["path"])
    kept = tests[args.index::args.count]
    out = dict(survey)
    out["tests"] = kept
    write_json(args.out, out)
    print(f"shard: {len(kept)} of {len(tests)} tests in shard "
          f"{args.index} of {args.count}")
    return 0


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def recompute_aggregates(results, wall_seconds, jit_mode):
    by_nature = {}
    by_suite = {}
    buckets = Counter()
    status_counts = Counter()
    passed = 0
    for r in results:
        status_counts[r["status"]] += 1
        is_pass = r["status"] == "pass"
        if is_pass:
            passed += 1
        if r.get("bucket"):
            buckets[r["bucket"]] += 1
        for table, key in ((by_nature, r["nature"]), (by_suite, r["suite"])):
            entry = table.setdefault(key, {"passed": 0, "total": 0})
            entry["total"] += 1
            if is_pass:
                entry["passed"] += 1
    for table in (by_nature, by_suite):
        for entry in table.values():
            entry["pass_rate_percent"] = rate(entry["passed"], entry["total"])
    total = len(results)
    return {
        "buckets": dict(buckets),
        "by_nature": by_nature,
        "by_suite": by_suite,
        "denominator": total,
        "failed": status_counts.get("fail", 0),
        "jit_mode": jit_mode,
        "merged_by": "scripts/ci-subset-tools.py merge",
        "metric_2b_pass_rate_percent": rate(passed, total),
        "not_attempted": status_counts.get("not-attempted", 0),
        "passed": passed,
        "results_total": total,
        "unsupported": status_counts.get("unsupported", 0),
        "wall_seconds": wall_seconds,
    }


def merge_files(paths):
    merged = {}
    wall = 0.0
    jit_mode = False
    for path in paths:
        data = load_json(path)
        if data.get("complete") is False:
            raise SystemExit(
                f"error: {path} is a partial results file from an "
                "interrupted rustc-subset-run.py run (marked "
                "\"complete\": false); refusing to merge it as a "
                "complete result")
        agg = data.get("aggregates", {})
        wall += float(agg.get("wall_seconds", 0.0))
        jit_mode = jit_mode or bool(agg.get("jit_mode", False))
        for r in data["results"]:
            key = (r["path"], r["nature"])
            if key in merged:
                raise SystemExit(f"error: duplicate record {key} in {path}")
            merged[key] = r
    results = [merged[k] for k in sorted(merged)]
    return {
        "aggregates": recompute_aggregates(results, wall, jit_mode),
        "results": results,
    }


def summary_markdown(agg, title):
    lines = [f"## {title}", ""]
    lines.append(f"**{agg['passed']}/{agg['denominator']} "
                 f"= {agg['metric_2b_pass_rate_percent']}%**")
    lines.append("")
    lines.append("| nature | passed | total | rate |")
    lines.append("| --- | ---: | ---: | ---: |")
    for nature in sorted(agg["by_nature"]):
        e = agg["by_nature"][nature]
        lines.append(f"| {nature} | {e['passed']} | {e['total']} "
                     f"| {e['pass_rate_percent']}% |")
    lines.append("")
    lines.append(f"failed: {agg['failed']}, "
                 f"unsupported: {agg['unsupported']}, "
                 f"not attempted: {agg['not_attempted']}")
    return "\n".join(lines) + "\n"


def parse_expect_nature(entries):
    """Parse repeated NAME=COUNT expectation arguments into a dict."""
    expected = {}
    for entry in entries or []:
        name, sep, count = entry.partition("=")
        if not sep or not name:
            raise ValueError(f"invalid --expect-nature {entry!r} "
                             f"(want NAME=COUNT)")
        try:
            expected[name] = int(count)
        except ValueError:
            raise ValueError(f"invalid --expect-nature {entry!r} "
                             f"(count is not an integer)") from None
    return expected


def denominator_errors(agg, expect_total, expect_nature, survey_counts=None):
    """Return a list of denominator mismatches (empty when all match).

    Checks, for every expectation that was given: the merged total,
    the merged per-nature totals, and -- when survey_counts (a Counter
    over the survey's tests) is given -- the survey's own total and
    per-nature counts, plus survey-vs-merged agreement per nature.
    """
    errors = []
    by_nature = agg["by_nature"]

    def check_total(label, actual, expected):
        if expected is not None and actual != expected:
            errors.append(f"{label}: expected {expected}, got {actual}")

    def check_natures(label, totals, expected):
        for nature in sorted(set(totals) | set(expected)):
            if totals.get(nature, 0) != expected.get(nature, 0):
                errors.append(
                    f"{label}: nature {nature!r} expected "
                    f"{expected.get(nature, 0)}, got {totals.get(nature, 0)}")

    check_total("merged total", agg["denominator"], expect_total)
    if expect_nature:
        check_natures("merged", {n: e["total"] for n, e in by_nature.items()},
                      expect_nature)
    if survey_counts is not None:
        check_total("survey total", sum(survey_counts.values()), expect_total)
        if expect_nature:
            check_natures("survey", dict(survey_counts), expect_nature)
        merged_totals = {n: e["total"] for n, e in by_nature.items()}
        for nature in sorted(set(merged_totals) | set(survey_counts)):
            if merged_totals.get(nature, 0) != survey_counts.get(nature, 0):
                errors.append(
                    f"survey vs merged: nature {nature!r} survey has "
                    f"{survey_counts.get(nature, 0)}, merged has "
                    f"{merged_totals.get(nature, 0)}")
    return errors


def cmd_merge(args):
    try:
        expect_nature = parse_expect_nature(args.expect_nature)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    merged = merge_files(args.inputs)
    agg = merged["aggregates"]
    survey_counts = None
    if args.survey is not None:
        survey = load_json(args.survey)
        # The survey's denominator excludes env_excluded tests (they
        # never reach the runner), matching the survey's own
        # aggregates and the merged results.
        survey_counts = Counter(t["nature"] for t in survey["tests"]
                                if not t.get("env_excluded"))
    errors = denominator_errors(agg, args.expect_total, expect_nature,
                                survey_counts)
    if errors:
        for e in errors:
            print(f"error: denominator assertion failed: {e}",
                  file=sys.stderr)
        return 1
    write_json(args.out, merged)
    print(f"merge: {agg['passed']}/{agg['denominator']} passed "
          f"({agg['metric_2b_pass_rate_percent']}%)")
    if args.summary:
        title = "Metric 4 (JIT)" if agg["jit_mode"] else "Metric 2(b)"
        Path(args.summary).write_text(
            summary_markdown(agg, title), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# check-candidates
# ---------------------------------------------------------------------------

def load_std_coverage():
    """scripts/std-coverage.py as a module, so the candidate predicate
    is the one metric 3 itself uses (not a copy that could drift)."""
    path = Path(__file__).resolve().parent / "std-coverage.py"
    spec = importlib.util.spec_from_file_location("std_coverage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cmd_check_candidates(args):
    std_coverage = load_std_coverage()
    expected = std_coverage.stage1_candidates(args.results)
    coverage = load_json(args.coverage)
    measured = coverage.get("subset_candidates")
    if measured is None:
        print(f"error: {args.coverage} records no subset_candidates",
              file=sys.stderr)
        return 1
    if measured != expected:
        missing = sorted(set(expected) - set(measured))
        extra = sorted(set(measured) - set(expected))
        print(f"error: metric-3 candidate set differs from the merged "
              f"results: {len(measured)} measured, {len(expected)} "
              f"expected; missing {missing[:10]}, extra {extra[:10]}",
              file=sys.stderr)
        return 1
    print(f"check-candidates: {len(measured)} stage-1 candidates agree")
    return 0


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def cmd_diff(args):
    def pass_map(path):
        data = load_json(path)
        return {(r["path"], r["nature"]): r.get("bucket", "") == ""
                for r in data["results"]}

    base = pass_map(args.base)
    head = pass_map(args.head)
    gains = sorted(k for k in head if head[k] and not base.get(k, False)
                   and k in base)
    losses = sorted(k for k in head if not head[k] and base.get(k, False))
    only_head = sorted(k for k in head if k not in base)
    lines = [f"gain: {len(gains)}, loss: {len(losses)}"]
    for k in gains:
        lines.append(f"GAIN {k[0]}|{k[1]}")
    for k in losses:
        lines.append(f"LOSS {k[0]}|{k[1]}")
    for k in only_head:
        lines.append(f"ONLY-HEAD {k[0]}|{k[1]}")
    report = "\n".join(lines) + "\n"
    print(report, end="")
    if args.summary:
        md = ["## Flip diff", "",
              f"**gain {len(gains)} / loss {len(losses)}** "
              f"(population {len(head)} records)", ""]
        for title, keys in (("Gains", gains), ("Losses", losses)):
            if keys:
                md.append(f"### {title}")
                for k in keys[:100]:
                    md.append(f"- `{k[0]}` ({k[1]})")
                if len(keys) > 100:
                    md.append(f"- ... and {len(keys) - 100} more")
                md.append("")
        Path(args.summary).write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# Host partition (issue #517)
# ---------------------------------------------------------------------------

#: Host classes for the partitioned measurement, by short name: the
#: runner target whose environment model judges the class. The
#: workflow enables linux-x86_64, linux-aarch64, darwin-aarch64,
#: darwin-x86_64 and windows-msvc; windows-gnu is defined so the
#: partition rule is complete, and its records stay in the
#: residual class until it gets jobs: it waits on a host
#: toolchain entry of its own -- its host model is the pinned
#: toolchain hosted on the gnu target, while the windows-latest
#: jobs install the runner's native MSVC-hosted toolchain (see
#: issue #517).
HOST_CLASSES = {
    "linux-x86_64": "x86_64-unknown-linux-gnu",
    "linux-aarch64": "aarch64-unknown-linux-gnu",
    "darwin-aarch64": "aarch64-apple-darwin",
    "windows-msvc": "x86_64-pc-windows-msvc",
    "windows-gnu": "x86_64-pc-windows-gnu",
    "darwin-x86_64": "x86_64-apple-darwin",
}

#: The classes the official workflow partitions into, in priority
#: order: a record goes to the first class that admits it, and the
#: first class is the residual class for records no class admits.
#: darwin-aarch64 precedes darwin-x86_64, so a record both macOS
#: models admit is judged on the aarch64 class and darwin-x86_64
#: keeps only the records its model alone admits.
DEFAULT_PARTITION_CLASSES = ["linux-x86_64", "linux-aarch64",
                             "darwin-aarch64", "darwin-x86_64",
                             "windows-msvc"]


def _load_runner_module():
    """The subset runner as a module (the same importlib pattern
    the runner's companion stages use), so the partition evaluates
    records with the runner's own applicability code and host
    profiles rather than a second implementation."""
    path = Path(__file__).with_name("rustc-subset-run.py")
    spec = importlib.util.spec_from_file_location(
        "rustc_subset_run_partition", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PartitionFacts:
    """ToolchainFacts stand-in for partitioning one host class.

    The target-spec facts (supported sanitizers, xray, dynamic
    linking, relocation model) are exact: they come from the pinned
    toolchain's own spec document for the class's target. The
    live-probe facts (LLVM components, profiler support, debugger
    presence) are set to their satisfied outcomes. That is safe for
    partitioning: a satisfied probe can only keep a record
    attemptable, never exclude it, so no record the Linux host
    attempts today is moved away, and a record assigned by a probe
    alone is re-evaluated for real by the judging host's runner,
    which records it out of environment there if the probe does not
    hold. The CI host jobs install the same rustc-dev component set
    as the Linux job, so the probes hold there by construction."""

    def __init__(self, spec):
        self._spec = spec or {}

    def sanitizers(self):
        return set(self._spec.get("supported-sanitizers") or [])

    def sanitizer_supported(self, name):
        return name in self.sanitizers()

    def xray(self):
        return bool(self._spec.get("supports-xray", False))

    def dynamic_linking(self):
        return bool(self._spec.get("dynamic-linking", False))

    def relocation_model(self):
        return self._spec.get("relocation-model") or "pic"

    # The remaining facts applicability() consults are the
    # live-probe ones discussed in the class docstring; they get
    # their satisfied outcomes (the values the official toolchain
    # installation the CI jobs use reports for them).
    def llvm_component_present(self, name):
        return True

    def has_profiler_support(self):
        return True

    def profiler_runtime(self):
        return True

    def debugger_present(self, kind):
        return True

    def rust_lld(self):
        return True

    def sanitizer_runtime_present(self):
        return True

    def git_hash(self):
        return True

    def llvm_zstd(self):
        return True

    def has_mnemonic(self, mnemonic):
        return True

    def supported_crate_types(self):
        return {"bin", "lib", "rlib", "dylib", "cdylib",
                "staticlib", "proc-macro"}

    def target_has_atomic(self):
        return {"8", "16", "32", "64", "128", "ptr"}

    def default_codegen_backend(self):
        return "llvm"


def _record_attemptable_on_host(runner, tree, record, facts):
    """Whether the runner on this host class would attempt the
    record: the same applicability witnesses the stages use --
    procedure and rustdoc records are gated on the whole header;
    the other natures are attempted when any revision view is
    applicable (debugger views through the debugger filter), and a
    revision-less exec record also counts its whole header, whose
    names stage 1 consults. A record whose file cannot be read is
    attemptable nowhere and lands in the residual class."""
    nature = record["nature"]
    if nature == "env_excluded":
        return False
    path = tree / record["path"]
    if path.is_dir():
        path = path / "rmake.rs"
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return False
    header = runner.Header(text)
    if nature in ("procedure", "rustdoc-check"):
        status, _why = runner.applicability(header, facts)
        return status != "out-of-environment"
    revisions = header.revisions()
    views = [header.view(rev) for rev in revisions] \
        if revisions else [header.view(None)]
    if nature == "exec-debugger":
        views = [runner._DebuggerFilteredView(view) for view in views]
    if nature == "exec" and not revisions:
        views.append(header)
    # A view counts when the host does not exclude it: "applicable",
    # or "unevaluatable" (a condition the canned facts cannot
    # decide; the judging host's runner evaluates it for real, so
    # the record belongs with the first host that does not exclude
    # it outright).
    return any(runner.applicability(view, facts)[0]
               != "out-of-environment" for view in views)


def partition_survey(runner, survey, tree, classes, specs):
    """Assign every survey record to exactly one host class.

    `classes` is a list of (name, target) in priority order; a
    record goes to the first class whose host model admits it, and
    records no class admits (including env_excluded records, which
    no host judges) go to the first, residual class, where they
    keep the verdict that class's runner gives them today. Raises
    ValueError when the assignment would not cover the survey
    exactly once -- a partition that drops or double-counts a
    record must never reach the workflow."""
    by_class = {name: [] for name, _target in classes}
    facts = {name: _PartitionFacts(specs.get(target))
             for name, target in classes}
    residual = classes[0][0]
    for record in survey["tests"]:
        dest = None
        if record.get("nature") != "env_excluded":
            for name, target in classes:
                runner.select_host_profile(target)
                if _record_attemptable_on_host(
                        runner, tree, record, facts[name]):
                    dest = name
                    break
        by_class[dest or residual].append(record)
    runner.select_host_profile(runner.DEFAULT_HOST_TARGET)
    seen = [record["path"] for records in by_class.values()
            for record in records]
    if len(seen) != len(set(seen)) or set(seen) != {
            record["path"] for record in survey["tests"]}:
        raise ValueError(
            "host partition does not cover the survey exactly once")
    return by_class


def _all_target_specs(runner):
    """The pinned toolchain's all-target-specs document, queried
    the way the runner queries it (its own rustc invocation and
    bootstrap env)."""
    res = runner.run_cmd(
        runner.rustc_cmd() + ["-Zunstable-options", "--print",
                              "all-target-specs-json"],
        timeout=120, env_extra=runner.BOOTSTRAP_ENV)
    if res["timeout"] or res["exit_code"] != 0:
        raise SystemExit(
            "partition: querying the pinned toolchain's target "
            "specs failed; cannot partition without them")
    return json.loads(res["stdout"])


def cmd_partition(args):
    survey = json.loads(Path(args.survey).read_text())
    classes = []
    for name in args.classes.split(","):
        name = name.strip()
        if name not in HOST_CLASSES:
            raise SystemExit(
                f"partition: unknown host class {name!r}; known: "
                + ", ".join(sorted(HOST_CLASSES)))
        classes.append((name, HOST_CLASSES[name]))
    runner = _load_runner_module()
    specs = _all_target_specs(runner)
    for _name, target in classes:
        if target not in specs:
            raise SystemExit(
                f"partition: the pinned toolchain has no target "
                f"spec for {target}; cannot partition for it")
    by_class = partition_survey(runner, survey, Path(args.tree),
                                classes, specs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, target in classes:
        doc = dict(survey)
        doc["tests"] = by_class[name]
        doc["partition"] = {"class": name, "target": target}
        out = out_dir / f"subset-survey-{name}.json"
        out.write_text(json.dumps(doc))
        total += len(by_class[name])
        print(f"partition: {name} ({target}): "
              f"{len(by_class[name])} records -> {out}")
    print(f"partition: {total} records in "
          f"{len(survey['tests'])} survey records, "
          f"{len(classes)} classes")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("filter")
    p.add_argument("--survey", type=Path, required=True)
    p.add_argument("--keywords", required=True,
                   help="comma-separated path substrings")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_filter)

    p = sub.add_parser("shard")
    p.add_argument("--survey", type=Path, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_shard)

    p = sub.add_parser("merge")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--summary", type=Path)
    p.add_argument("--expect-total", type=int,
                   help="fail unless the merged denominator equals this")
    p.add_argument("--expect-nature", action="append", default=[],
                   metavar="NAME=COUNT",
                   help="fail unless the merged total for NAME equals "
                        "COUNT (repeatable)")
    p.add_argument("--survey", type=Path,
                   help="survey JSON the shards were cut from; its total "
                        "and per-nature counts are asserted as well")
    p.add_argument("inputs", nargs="+", type=Path)
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("check-candidates")
    p.add_argument("--coverage", type=Path, required=True,
                   help="scripts/std-coverage.py result JSON")
    p.add_argument("--results", type=Path, required=True,
                   help="merged subset result JSON")
    p.set_defaults(func=cmd_check_candidates)

    p = sub.add_parser("partition")
    p.add_argument("--survey", type=Path, required=True)
    p.add_argument("--tree", type=Path, required=True,
                   help="the survey tree the records' headers are "
                        "read from")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--classes", default=",".join(DEFAULT_PARTITION_CLASSES),
        help="comma-separated host classes in priority order; "
             "the first is the residual class")
    p.set_defaults(func=cmd_partition)

    p = sub.add_parser("diff")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--summary", type=Path)
    p.set_defaults(func=cmd_diff)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
