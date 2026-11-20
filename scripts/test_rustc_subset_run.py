#!/usr/bin/env python3
"""Unit tests for scripts/rustc-subset-run.py (metric 2(b) runner).

Run with: python3 -m unittest scripts.test_rustc_subset_run
(or directly: python3 scripts/test_rustc_subset_run.py)

The verdict logic is exercised on small fixture headers/results;
process_test is exercised with the two process sides stubbed out, so
no toolchain or driver is needed.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location(
    "rustc_subset_run", Path(__file__).parent / "rustc-subset-run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def header(text):
    return run.Header(text)


class HeaderTests(unittest.TestCase):
    def test_unscoped_and_scoped(self):
        h = header("//@ run-pass\n//@[rev1] check-pass\nfn main() {}\n")
        self.assertTrue(h.has("run-pass"))
        self.assertIn(("rev1", "check-pass"), h.scoped)
        self.assertEqual(run.run_mode(h), "run-pass")

    def test_directive_without_space_and_value(self):
        h = header("//@edition: 2018\n//@ failure-status: 7\n")
        self.assertEqual(h.values("edition"), ["2018"])
        self.assertEqual(h.values("failure-status"), ["7"])

    def test_non_directive_lines_ignored(self):
        h = header("// run-pass\n/* //@ run-fail */\n")
        self.assertIsNone(run.run_mode(h))


class BlockerTests(unittest.TestCase):
    def rec(self):
        return {"suite": "ui", "path": "tests/ui/x.rs", "nature": "exec",
                "single_file": True, "ffi": False, "env_excluded": False}

    def test_plain_run_pass_has_no_blocker(self):
        self.assertIsNone(run.stage1_blocker(header("//@ run-pass\n"),
                                             self.rec()))

    def test_revisions_block(self):
        h = header("//@ revisions: a b\n//@[a] run-pass\n")
        self.assertIn("revisions", run.stage1_blocker(h, self.rec()))

    def test_compile_flags_block(self):
        h = header("//@ run-pass\n//@ compile-flags: -C opt-level=3\n")
        self.assertIn("compile-flags", run.stage1_blocker(h, self.rec()))

    def test_run_crash_blocks(self):
        h = header("//@ run-crash\n")
        self.assertIn("run-crash", run.stage1_blocker(h, self.rec()))


class ApplicabilityTests(unittest.TestCase):
    def test_ignore_windows_applicable_here(self):
        status, _ = run.applicability(header("//@ run-pass\n//@ ignore-windows\n"))
        self.assertEqual(status, "applicable")

    def test_ignore_linux_out_of_environment(self):
        status, why = run.applicability(header("//@ ignore-linux\n"))
        self.assertEqual(status, "out-of-environment")
        self.assertIn("ignore-linux", why)

    def test_only_windows_out_of_environment(self):
        status, _ = run.applicability(header("//@ only-windows\n"))
        self.assertEqual(status, "out-of-environment")

    def test_only_linux_applicable(self):
        status, _ = run.applicability(header("//@ only-linux\n"))
        self.assertEqual(status, "applicable")

    def test_cfg_expr(self):
        h = header('//@ ignore-[target_os = "linux"]\n'.replace(
            "ignore-[", "ignore-cfg(").replace("]", ")"))
        status, _ = run.applicability(h)
        self.assertEqual(status, "out-of-environment")

    def test_unknown_needs_unevaluatable(self):
        status, why = run.applicability(header("//@ needs-sanitizer-support\n"))
        self.assertEqual(status, "unevaluatable")
        self.assertIn("needs-sanitizer-support", why)


class JudgeTests(unittest.TestCase):
    def test_run_pass_match(self):
        h = header("//@ run-pass\n")
        native = {"exit_code": 0, "stdout": "hi\n"}
        got = {"exit_code": 0, "stdout": "hi\n"}
        self.assertEqual(run.judge("run-pass", h, native, got), ("pass", ""))

    def test_stdout_mismatch_fails(self):
        h = header("//@ run-pass\n")
        native = {"exit_code": 0, "stdout": "hi\n"}
        got = {"exit_code": 0, "stdout": "bye\n"}
        status, detail = run.judge("run-pass", h, native, got)
        self.assertEqual(status, "fail")
        self.assertIn("stdout", detail)

    def test_run_fail_default_status_101(self):
        h = header("//@ run-fail\n")
        native = {"exit_code": 101, "stdout": ""}
        got = {"exit_code": 101, "stdout": ""}
        self.assertEqual(run.judge("run-fail", h, native, got)[0], "pass")

    def test_run_fail_custom_failure_status(self):
        h = header("//@ run-fail\n//@ failure-status: 7\n")
        self.assertEqual(run.expected_exit("run-fail", h), 7)
        native = {"exit_code": 7, "stdout": ""}
        got = {"exit_code": 101, "stdout": ""}
        self.assertEqual(run.judge("run-fail", h, native, got)[0], "fail")


class BucketReasonTests(unittest.TestCase):
    def test_by_reference_constant_wins_over_callee_list(self):
        reason = ("sparagmos-driver run: unsupported: MIR conversion "
                  "failed: unsupported constant: by-reference constant; "
                  "sparagmos-driver run: calls to functions with no "
                  "converted body: std::io::_print")
        self.assertEqual(run.bucket_reason(reason),
                         "unsupported constant: by-reference constant")

    def test_box_deref_transmute(self):
        reason = ("sparagmos-driver run: unsupported: `main` was not "
                  "converted (CastKind::BoxDerefTransmute x1)")
        self.assertIn("BoxDerefTransmute", run.bucket_reason(reason))

    def test_unconverted_callee(self):
        reason = ("sparagmos-driver run: unsupported: call failed; "
                  "sparagmos-driver run: calls to functions with no "
                  "converted body: core::panicking::panic")
        self.assertIn("no converted body", run.bucket_reason(reason))

    def test_native_compile_failed(self):
        self.assertEqual(
            run.bucket_reason("native compile failed: error[E0308]"),
            "native compile failed")


class ProcessTestTests(unittest.TestCase):
    """process_test with both process sides stubbed."""

    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile")
                                         .TemporaryDirectory()))
        (self.tmp / "tests" / "ui").mkdir(parents=True)
        self.rust_dir = self.tmp

    def write_test(self, body):
        p = self.rust_dir / "tests" / "ui" / "t.rs"
        p.write_text(body)
        return {"suite": "ui", "path": "tests/ui/t.rs", "nature": "exec",
                "single_file": True, "ffi": False, "env_excluded": False}

    def run_with_sides(self, rec, native, sparagmos):
        orig_n, orig_s = run.native_side, run.sparagmos_side
        run.native_side = lambda *a: native
        run.sparagmos_side = lambda *a: sparagmos
        try:
            return run.process_test(rec, self.rust_dir, Path("/fake/driver"))
        finally:
            run.native_side, run.sparagmos_side = orig_n, orig_s

    def test_pass_end_to_end(self):
        rec = self.write_test("//@ run-pass\nfn main() { println!(\"x\"); }\n")
        res = self.run_with_sides(
            rec,
            ({"exit_code": 0, "stdout": "x\n", "stderr": ""}, None),
            ({"exit_code": 0, "stdout": "x\n", "stderr": ""}, "result", ""))
        self.assertEqual(res["status"], "pass")

    def test_driver_unsupported_maps_to_unsupported(self):
        rec = self.write_test("//@ run-pass\nfn main() {}\n")
        res = self.run_with_sides(
            rec,
            ({"exit_code": 0, "stdout": "", "stderr": ""}, None),
            (None, "unsupported",
             "sparagmos-driver run: unsupported type in local _2"))
        self.assertEqual(res["status"], "unsupported")
        self.assertIn("unsupported type", res["bucket"])

    def test_native_compile_failure_is_not_attempted(self):
        rec = self.write_test("//@ run-pass\nfn main() {}\n")
        res = self.run_with_sides(
            rec,
            (None, "native compile failed: error[E0308]: mismatch"),
            (None, "unsupported", "unused"))
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"], "native compile failed")

    def test_non_exec_is_out_of_scope(self):
        rec = self.write_test("fn main() {}\n")
        rec["nature"] = "compile-check"
        res = run.process_test(rec, self.rust_dir, Path("/fake/driver"))
        self.assertEqual(res["status"], "not-attempted")
        self.assertIn("nature", res["bucket"])


if __name__ == "__main__":
    unittest.main()
