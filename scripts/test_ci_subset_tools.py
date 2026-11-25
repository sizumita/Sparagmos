#!/usr/bin/env python3
"""Unit tests for scripts/ci-subset-tools.py (CI subset helpers).

Run with: python3 -m pytest scripts/test_ci_subset_tools.py
(or directly: python3 scripts/test_ci_subset_tools.py)

The merge subcommand is exercised end to end on small fixtures,
including the denominator assertion (--expect-total / --expect-nature
/ --survey) that the official measurement workflow relies on: merges
without expectations accept any population, merges with expectations
fail (and write no output) when the merged total, a per-nature total,
or the survey disagrees.
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location(
    "ci_subset_tools", Path(__file__).parent / "ci-subset-tools.py")
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)

SCRIPT = Path(__file__).parent / "ci-subset-tools.py"


def record(path, nature, status="pass"):
    return {"path": path, "nature": nature, "suite": "ui",
            "status": status, "bucket": "" if status == "pass" else "x"}


def result_file(records):
    return {"aggregates": {"wall_seconds": 1.0, "jit_mode": False},
            "results": records}


def survey_file(tests):
    return {"tests": [{"path": p, "nature": n} for p, n in tests]}


class ParseExpectNatureTests(unittest.TestCase):
    def test_parses_repeated_entries(self):
        self.assertEqual(
            tools.parse_expect_nature(["exec=3", "compile-check=10"]),
            {"exec": 3, "compile-check": 10})

    def test_rejects_missing_separator(self):
        with self.assertRaises(ValueError):
            tools.parse_expect_nature(["exec"])

    def test_rejects_non_integer_count(self):
        with self.assertRaises(ValueError):
            tools.parse_expect_nature(["exec=many"])


class DenominatorErrorsTests(unittest.TestCase):
    def agg(self):
        results = [record("a.rs", "exec"), record("b.rs", "exec"),
                   record("c.rs", "compile-check", "fail")]
        return tools.recompute_aggregates(results, 1.0, False)

    def test_matching_expectations_pass(self):
        self.assertEqual(
            tools.denominator_errors(
                self.agg(), 3, {"exec": 2, "compile-check": 1}),
            [])

    def test_no_expectations_pass_any_population(self):
        self.assertEqual(tools.denominator_errors(self.agg(), None, {}), [])

    def test_wrong_total_reported(self):
        errors = tools.denominator_errors(self.agg(), 4, {})
        self.assertEqual(len(errors), 1)
        self.assertIn("expected 4, got 3", errors[0])

    def test_wrong_nature_reported(self):
        errors = tools.denominator_errors(
            self.agg(), 3, {"exec": 1, "compile-check": 2})
        self.assertEqual(len(errors), 2)

    def test_unexpected_nature_reported(self):
        errors = tools.denominator_errors(self.agg(), 3, {"exec": 2})
        self.assertTrue(any("compile-check" in e for e in errors))

    def test_survey_mismatch_reported(self):
        from collections import Counter
        errors = tools.denominator_errors(
            self.agg(), 3, {"exec": 2, "compile-check": 1},
            Counter({"exec": 2, "compile-check": 2}))
        self.assertTrue(any("survey" in e for e in errors))


class MergeCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        records_a = [record("a.rs", "exec"), record("c.rs", "compile-check")]
        records_b = [record("b.rs", "exec"),
                     record("d.rs", "compile-check", "fail")]
        self.inputs = []
        for i, records in enumerate((records_a, records_b)):
            path = self.dir / f"shard-{i}.json"
            path.write_text(json.dumps(result_file(records)))
            self.inputs.append(str(path))
        self.survey = self.dir / "survey.json"
        self.survey.write_text(json.dumps(survey_file([
            ("a.rs", "exec"), ("b.rs", "exec"),
            ("c.rs", "compile-check"), ("d.rs", "compile-check")])))
        self.out = self.dir / "merged.json"

    def tearDown(self):
        self.tmp.cleanup()

    def run_merge(self, *extra):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "merge", "--out", str(self.out),
             *extra, *self.inputs],
            capture_output=True, text=True)

    def test_merge_without_expectations(self):
        res = self.run_merge()
        self.assertEqual(res.returncode, 0, res.stderr)
        merged = json.loads(self.out.read_text())
        self.assertEqual(merged["aggregates"]["denominator"], 4)
        self.assertEqual(merged["aggregates"]["passed"], 3)

    def test_merge_with_matching_expectations_and_survey(self):
        res = self.run_merge(
            "--expect-total", "4",
            "--expect-nature", "exec=2",
            "--expect-nature", "compile-check=2",
            "--survey", str(self.survey))
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(self.out.exists())

    def test_merge_with_wrong_total_fails_and_writes_nothing(self):
        res = self.run_merge("--expect-total", "24937")
        self.assertEqual(res.returncode, 1)
        self.assertIn("denominator assertion failed", res.stderr)
        self.assertFalse(self.out.exists())

    def test_merge_with_wrong_nature_fails(self):
        res = self.run_merge(
            "--expect-total", "4", "--expect-nature", "exec=3",
            "--expect-nature", "compile-check=1")
        self.assertEqual(res.returncode, 1)
        self.assertIn("denominator assertion failed", res.stderr)
        self.assertFalse(self.out.exists())

    def test_merge_ignores_env_excluded_survey_tests(self):
        survey = self.dir / "survey-env.json"
        data = survey_file([
            ("a.rs", "exec"), ("b.rs", "exec"),
            ("c.rs", "compile-check"), ("d.rs", "compile-check")])
        data["tests"].append({"path": "e.rs", "nature": "procedure",
                              "env_excluded": True})
        survey.write_text(json.dumps(data))
        res = self.run_merge(
            "--expect-total", "4",
            "--expect-nature", "exec=2",
            "--expect-nature", "compile-check=2",
            "--survey", str(survey))
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(self.out.exists())

    def test_merge_with_mismatched_survey_fails(self):
        bad_survey = self.dir / "bad-survey.json"
        bad_survey.write_text(json.dumps(survey_file([
            ("a.rs", "exec"), ("b.rs", "exec"), ("c.rs", "compile-check"),
            ("d.rs", "compile-check"), ("e.rs", "exec")])))
        res = self.run_merge(
            "--expect-total", "4",
            "--expect-nature", "exec=2",
            "--expect-nature", "compile-check=2",
            "--survey", str(bad_survey))
        self.assertEqual(res.returncode, 1)
        self.assertIn("survey", res.stderr)
        self.assertFalse(self.out.exists())


class CheckCandidatesTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.results = self.dir / "merged.json"
        self.results.write_text(json.dumps({"results": [
            {"path": "tests/ui/a.rs", "nature": "exec",
             "single_file": True, "suite": "ui"},
            {"path": "tests/ui/b.rs", "nature": "exec",
             "single_file": True, "suite": "ui"},
            {"path": "tests/ui/c.rs", "nature": "compile-check",
             "single_file": True, "suite": "ui"},
        ]}))

    def run_check(self, candidates):
        coverage = self.dir / "coverage.json"
        data = {"aggregate": {}}
        if candidates is not None:
            data["subset_candidates"] = candidates
        coverage.write_text(json.dumps(data))
        return subprocess.run(
            [sys.executable, str(SCRIPT), "check-candidates",
             "--coverage", str(coverage), "--results", str(self.results)],
            capture_output=True, text=True)

    def test_matching_set_passes(self):
        res = self.run_check(["tests/ui/a.rs", "tests/ui/b.rs"])
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_missing_candidate_fails(self):
        res = self.run_check(["tests/ui/a.rs"])
        self.assertEqual(res.returncode, 1)
        self.assertIn("tests/ui/b.rs", res.stderr)

    def test_coverage_without_candidate_list_fails(self):
        res = self.run_check(None)
        self.assertEqual(res.returncode, 1)


class PartitionTests(unittest.TestCase):
    """The host partition (issue #517): a record lands in the first
    class whose host model admits it, a record no class admits
    stays in the first (residual) class, and the classes cover the
    survey exactly once."""

    CLASSES = [("linux-x86_64", "x86_64-unknown-linux-gnu"),
               ("linux-aarch64", "aarch64-unknown-linux-gnu"),
               ("darwin-aarch64", "aarch64-apple-darwin")]
    SPECS = {
        "x86_64-unknown-linux-gnu":
            {"supported-sanitizers": ["address"]},
        "aarch64-unknown-linux-gnu":
            {"supported-sanitizers": ["address", "hwaddress"]},
        "aarch64-apple-darwin":
            {"supported-sanitizers": ["address"]},
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tree = Path(self.tmp.name)
        files = {
            "tests/ui/plain.rs": "fn main() {}\n",
            "tests/ui/arm-only.rs":
                "//@ only-aarch64\nfn main() {}\n",
            "tests/ui/apple-only.rs":
                "//@ only-apple\nfn main() {}\n",
            "tests/ui/win-only.rs":
                "//@ only-windows\nfn main() {}\n",
            "tests/ui/hwaddr.rs":
                "//@ needs-sanitizer-hwaddress\nfn main() {}\n",
        }
        for rel, text in files.items():
            path = self.tree / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.survey = {"tests": [
            {"path": rel, "nature": "compile-check"} for rel in files
        ] + [{"path": "tests/ui/excluded.rs",
              "nature": "env_excluded"}]}
        self.runner = tools._load_runner_module()

    def partition(self):
        return tools.partition_survey(
            self.runner, self.survey, self.tree, self.CLASSES,
            self.SPECS)

    def test_assignment(self):
        by_class = self.partition()
        paths = {name: {r["path"] for r in records}
                 for name, records in by_class.items()}
        self.assertEqual(paths["linux-aarch64"],
                         {"tests/ui/arm-only.rs",
                          "tests/ui/hwaddr.rs"})
        self.assertEqual(paths["darwin-aarch64"],
                         {"tests/ui/apple-only.rs"})
        # plain.rs is admitted everywhere and goes to the first
        # class; win-only.rs is admitted by no class in the list
        # and excluded.rs is env_excluded: both stay residual.
        self.assertEqual(paths["linux-x86_64"],
                         {"tests/ui/plain.rs", "tests/ui/win-only.rs",
                          "tests/ui/excluded.rs"})

    def test_coverage_is_exactly_once(self):
        by_class = self.partition()
        seen = [r["path"] for records in by_class.values()
                for r in records]
        self.assertEqual(len(seen), len(self.survey["tests"]))
        self.assertEqual(len(seen), len(set(seen)))

    def test_partition_restores_the_default_profile(self):
        self.partition()
        self.assertEqual(self.runner.HOST_TARGET,
                         self.runner.DEFAULT_HOST_TARGET)


if __name__ == "__main__":
    unittest.main()
