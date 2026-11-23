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
- merge: combine several subset result JSONs (one per shard) into a
  single result JSON whose aggregates are recomputed from the merged
  records, so the merged file is a drop-in replacement for a result
  file produced by one unsharded run. With --expect-total /
  --expect-nature (and optionally --survey) the merge additionally
  asserts that the denominator matches the expected population, and
  fails without writing output when it does not; the official
  measurement workflow always passes these options, while partial
  populations (flip-diff etc.) merge without them.
- diff: compare two result JSONs (base driver vs head driver) and
  report gains and losses. A record passes when its bucket is empty;
  the comparison key is path + nature, matching the flip-diff
  definition the lanes use on the dev host.

Only the Python standard library is used.
"""

import argparse
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

    p = sub.add_parser("diff")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--summary", type=Path)
    p.set_defaults(func=cmd_diff)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
