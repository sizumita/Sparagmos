#!/usr/bin/env python3
"""Unit tests for scripts/rustc-subset-rustdoc.py (rustdoc category).

Run with: python3 -m unittest scripts.test_rustc_subset_rustdoc
(or directly: python3 scripts/test_rustc_subset_rustdoc.py)

The classification, expectation and eligibility logic is exercised
on small fixture headers and fixture files in a temporary directory;
the agreement attempt is exercised with the driver invocation stubbed
out, so no toolchain or driver binary is needed.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location(
    "rustc_subset_rustdoc", Path(__file__).parent / "rustc-subset-rustdoc.py")
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)
stage1 = rd.load_stage1()


def header(text):
    return stage1.Header(text)


def rec(suite, single_file=True):
    return {"suite": suite, "path": f"tests/{suite}/x.rs",
            "nature": "rustdoc-check", "single_file": single_file,
            "ffi": False, "env_excluded": False}


class VerdictFormTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def form(self, suite, text, siblings=()):
        p = self.tmp / "x.rs"
        p.write_text(text)
        for s in siblings:
            (self.tmp / s).write_text("")
        return rd.verdict_form(rec(suite), p, header(text), text)

    def test_html_content_directives(self):
        self.assertEqual(
            self.form("rustdoc-html", "//@ has foo/fn.bar.html\n"),
            "html-content")

    def test_html_negated_directive_counts_as_content(self):
        self.assertEqual(
            self.form("rustdoc-html", "//@ !has foo/index.html\n"),
            "html-content")

    def test_html_without_content_directives_is_generation(self):
        self.assertEqual(
            self.form("rustdoc-html", "//@ aux-build: aux.rs\n"),
            "html-generation")

    def test_json_is_always_content(self):
        self.assertEqual(
            self.form("rustdoc-json", "pub fn f() {}\n"), "json-content")

    def test_script_suites(self):
        self.assertEqual(self.form("rustdoc-js", ""), "search-index-js")
        self.assertEqual(
            self.form("rustdoc-js-std", ""), "search-index-js-std")
        self.assertEqual(self.form("rustdoc-gui", ""), "browser-gui")

    def test_ui_doctest_form_wins_over_snapshots(self):
        self.assertEqual(
            self.form("rustdoc-ui",
                      "//@ compile-flags: --test\n//@ check-pass\n",
                      siblings=("x.stdout",)),
            "doctest-report")

    def test_ui_coverage_form(self):
        self.assertEqual(
            self.form("rustdoc-ui",
                      "//@ compile-flags: --show-coverage\n"),
            "coverage-output")

    def test_ui_stdout_snapshot_form(self):
        self.assertEqual(
            self.form("rustdoc-ui", "//@ check-pass\n",
                      siblings=("x.stdout",)),
            "stdout-snapshot")

    def test_ui_diagnostics_form_from_stderr_file(self):
        self.assertEqual(
            self.form("rustdoc-ui", "pub fn f() {}\n",
                      siblings=("x.stderr",)),
            "diagnostics-snapshot")

    def test_ui_diagnostics_form_from_error_pattern(self):
        self.assertEqual(
            self.form("rustdoc-ui", "//@ error-pattern: boom\n"),
            "diagnostics-snapshot")

    def test_ui_exit_status_only(self):
        self.assertEqual(
            self.form("rustdoc-ui", "//@ check-pass\n"),
            "exit-status-only")


class ExpectedOutcomeTests(unittest.TestCase):
    def test_check_pass_is_success(self):
        self.assertEqual(
            rd.expected_outcome(header("//@ check-pass\n")), "success")

    def test_build_pass_is_success(self):
        self.assertEqual(
            rd.expected_outcome(header("//@ build-pass\n")), "success")

    def test_check_fail_is_failure(self):
        self.assertEqual(
            rd.expected_outcome(header("//@ check-fail\n")), "failure")

    def test_no_declaration_defaults_to_failure(self):
        # compiletest UI mode defaults the pass/fail mode to
        # check-fail when a test declares none.
        self.assertEqual(
            rd.expected_outcome(header("pub fn f() {}\n")), "failure")


class BlockerTests(unittest.TestCase):
    def blocker(self, text, single_file=True):
        return rd.ui_blocker(rec("rustdoc-ui", single_file),
                             header(text), stage1)

    def test_plain_check_pass_is_eligible(self):
        self.assertIsNone(self.blocker("//@ check-pass\n"))

    def test_multi_file_blocks(self):
        self.assertIn("multi-file",
                      self.blocker("//@ check-pass\n", single_file=False))

    def test_revisions_block(self):
        self.assertIn("revisions",
                      self.blocker("//@ revisions: a b\n"))

    def test_scoped_directive_blocks(self):
        self.assertIn("revisions",
                      self.blocker("//@[a] check-pass\n"))

    def test_compile_flags_block(self):
        self.assertIn("compile-flags",
                      self.blocker("//@ compile-flags: --test\n"))

    def test_rustc_env_blocks(self):
        self.assertIn("rustc-env",
                      self.blocker("//@ rustc-env: FOO=1\n"))

    def test_run_rustfix_blocks(self):
        self.assertIn("run-rustfix", self.blocker("//@ run-rustfix\n"))

    def test_ignore_test_blocks(self):
        self.assertIn("ignore-test", self.blocker("//@ ignore-test\n"))

    def test_only_windows_is_out_of_environment(self):
        self.assertIn("only-*", self.blocker("//@ only-windows\n"))


class AgreementTests(unittest.TestCase):
    def test_agreement_matrix(self):
        self.assertEqual(rd.judge_agreement("success", True),
                         "outcome-agree")
        self.assertEqual(rd.judge_agreement("failure", False),
                         "outcome-agree")
        self.assertEqual(rd.judge_agreement("success", False),
                         "outcome-disagree")
        self.assertEqual(rd.judge_agreement("failure", True),
                         "outcome-disagree")


class ClassifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.tmp / "tests" / "rustdoc-ui").mkdir(parents=True)
        (self.tmp / "tests" / "rustdoc-html").mkdir(parents=True)

    def write(self, suite, name, text):
        p = self.tmp / "tests" / suite / name
        p.write_text(text)
        return {"suite": suite, "path": f"tests/{suite}/{name}",
                "nature": "rustdoc-check", "single_file": True,
                "ffi": False, "env_excluded": False}

    def test_html_is_never_eligible(self):
        r = self.write("rustdoc-html", "a.rs", "//@ has a/fn.f.html\n")
        out = rd.classify(r, self.tmp, stage1)
        self.assertEqual(out["status"], "not-attempted")
        self.assertEqual(out["verdict_form"], "html-content")
        self.assertIn("rustdoc output", out["reason"])

    def test_ui_check_pass_becomes_eligible(self):
        r = self.write("rustdoc-ui", "b.rs", "//@ check-pass\n")
        out = rd.classify(r, self.tmp, stage1)
        self.assertEqual(out["status"], "eligible")
        self.assertEqual(out["expected"], "success")

    def test_ui_blocked_keeps_reason(self):
        r = self.write("rustdoc-ui", "c.rs",
                       "//@ compile-flags: --test\n//@ check-pass\n")
        out = rd.classify(r, self.tmp, stage1)
        self.assertEqual(out["status"], "not-attempted")
        self.assertEqual(out["verdict_form"], "doctest-report")
        self.assertIn("compile-flags", out["reason"])


if __name__ == "__main__":
    unittest.main()
