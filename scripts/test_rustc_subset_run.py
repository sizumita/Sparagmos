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




class CompileExpectationTests(unittest.TestCase):
    def test_ui_default_is_check_fail(self):
        h = header("fn main() { let x: u8 = \"s\"; }\n")
        self.assertEqual(run.compile_expectation("ui", h), "failure")

    def test_ui_check_pass_and_build_pass(self):
        self.assertEqual(
            run.compile_expectation("ui", header("//@ check-pass\n")),
            "success")
        self.assertEqual(
            run.compile_expectation("ui", header("//@ build-pass\n")),
            "success")

    def test_ui_explicit_failures(self):
        self.assertEqual(
            run.compile_expectation("ui", header("//@ check-fail\n")),
            "failure")
        self.assertEqual(
            run.compile_expectation("ui", header("//@ build-fail\n")),
            "failure")

    def test_run_fail_or_crash_is_not_judgable(self):
        h = header("//@ run-fail-or-crash\n")
        self.assertIsNone(run.compile_expectation("ui", h))

    def test_crashes_suite_expects_crash(self):
        self.assertEqual(
            run.compile_expectation("crashes", header("//@ known-bug: #1\n")),
            "crash")

    def test_output_suites_expect_success(self):
        for suite in ("mir-opt", "codegen-llvm", "codegen-units",
                      "assembly-llvm", "pretty"):
            self.assertEqual(
                run.compile_expectation(suite, header("fn main() {}\n")),
                "success", suite)


class Stage2BlockerTests(unittest.TestCase):
    def rec(self):
        return {"suite": "ui", "path": "tests/ui/x.rs",
                "nature": "compile-check", "single_file": True,
                "ffi": False, "env_excluded": False}

    def test_plain_check_fail_has_no_blocker(self):
        self.assertIsNone(run.stage2_blocker(header("//@ check-fail\n"),
                                             self.rec()))

    def test_compile_flags_block(self):
        h = header("//@ compile-flags: -Z unstable-options\n")
        self.assertIn("compile-flags", run.stage2_blocker(h, self.rec()))

    def test_add_minicore_blocks(self):
        h = header("//@ add-minicore\n")
        self.assertIn("minicore", run.stage2_blocker(h, self.rec()))

    def test_rustc_env_blocks(self):
        h = header("//@ rustc-env: FOO=1\n")
        self.assertIn("rustc-env", run.stage2_blocker(h, self.rec()))

    def test_revisions_block(self):
        h = header("//@ revisions: a b\n//@[a] check-pass\n")
        self.assertIn("revisions", run.stage2_blocker(h, self.rec()))


def conv(**kw):
    base = {"exit_code": 0, "timeout": False, "converted": 3,
            "skipped": 0, "skip_totals": [], "compilation_failed": False,
            "ice": False, "stderr": "", "stdout": ""}
    base.update(kw)
    return base


class JudgeCompileTests(unittest.TestCase):
    def test_success_fully_converted_passes(self):
        self.assertEqual(run.judge_compile("success", conv()),
                         ("pass", "", ""))

    def test_success_with_skips_is_unsupported(self):
        status, reason, bucket = run.judge_compile(
            "success",
            conv(skipped=1,
                 skip_totals=[("TerminatorKind::InlineAsm", 1)]))
        self.assertEqual(status, "unsupported")
        self.assertIn("InlineAsm", reason)
        self.assertIn("InlineAsm", bucket)

    def test_success_but_compilation_failed_is_fail(self):
        status, _r, bucket = run.judge_compile(
            "success", conv(exit_code=1, compilation_failed=True,
                            converted=None, skipped=None,
                            stderr="error[E0308]: mismatched types"))
        self.assertEqual(status, "fail")
        self.assertIn("compilation failed", bucket)

    def test_success_but_driver_crash_is_fail(self):
        status, _r, bucket = run.judge_compile(
            "success", conv(exit_code=101, converted=None, skipped=None))
        self.assertEqual(status, "fail")
        self.assertEqual(bucket, "driver crash")

    def test_failure_clean_rejection_passes(self):
        status, _r, _b = run.judge_compile(
            "failure", conv(exit_code=1, compilation_failed=True,
                            converted=None, skipped=None))
        self.assertEqual(status, "pass")

    def test_failure_but_converted_is_fail(self):
        status, _r, bucket = run.judge_compile("failure", conv())
        self.assertEqual(status, "fail")
        self.assertIn("conversion succeeded", bucket)

    def test_failure_but_ice_is_fail(self):
        status, _r, bucket = run.judge_compile(
            "failure", conv(exit_code=101, ice=True, converted=None,
                            skipped=None))
        self.assertEqual(status, "fail")
        self.assertIn("crash", bucket)

    def test_crash_reproduced_passes(self):
        status, _r, _b = run.judge_compile(
            "crash", conv(exit_code=101, ice=True, converted=None,
                          skipped=None))
        self.assertEqual(status, "pass")

    def test_crash_no_longer_crashes(self):
        self.assertEqual(
            run.judge_compile("crash", conv())[0], "fail")
        self.assertEqual(
            run.judge_compile(
                "crash", conv(exit_code=1, compilation_failed=True,
                              converted=None, skipped=None))[0],
            "fail")

    def test_timeout_is_fail(self):
        status, _r, bucket = run.judge_compile(
            "success", conv(exit_code=None, timeout=True))
        self.assertEqual(status, "fail")
        self.assertEqual(bucket, "timeout")


class SnapshotFlagsTests(unittest.TestCase):
    def test_flags_from_siblings_and_text(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "t.rs").write_text("//~^ ERROR mismatched types\n")
            (d / "t.stderr").write_text("error: ...")
            (d / "t.fixed").write_text("fn main() {}\n")
            flags = run.snapshot_flags(d / "t.rs",
                                       "//~^ ERROR mismatched types\n")
            self.assertTrue(flags["stderr_snapshot"])
            self.assertTrue(flags["fixed_file"])
            self.assertTrue(flags["error_annotations"])
            self.assertFalse(flags["mir_snapshot"])
            self.assertFalse(flags["filecheck"])


class ProcessStage2Tests(unittest.TestCase):
    """process_test(stage2=True) with convert_side stubbed."""

    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile")
                                         .TemporaryDirectory()))
        (self.tmp / "tests" / "ui").mkdir(parents=True)
        self.rust_dir = self.tmp

    def write_test(self, body, suite="ui"):
        p = self.rust_dir / "tests" / "ui" / "t.rs"
        p.write_text(body)
        return {"suite": suite, "path": "tests/ui/t.rs",
                "nature": "compile-check", "single_file": True,
                "ffi": False, "env_excluded": False}

    def run_with_conv(self, rec, conv_result):
        orig = run.convert_side
        run.convert_side = lambda *a: conv_result
        try:
            return run.process_test(rec, self.rust_dir,
                                    Path("/fake/driver"), stage2=True)
        finally:
            run.convert_side = orig

    def test_check_fail_clean_rejection_passes(self):
        rec = self.write_test("//@ check-fail\nfn main() { let x: u8 = \"s\"; }\n")
        res = self.run_with_conv(
            rec, conv(exit_code=1, compilation_failed=True,
                      converted=None, skipped=None))
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["stage"], 2)
        self.assertEqual(res["expectation"], "failure")

    def test_check_pass_incomplete_conversion_is_unsupported(self):
        rec = self.write_test("//@ check-pass\nfn main() {}\n")
        res = self.run_with_conv(
            rec, conv(skipped=2,
                      skip_totals=[("Rvalue::BinaryOp", 3)]))
        self.assertEqual(res["status"], "unsupported")
        self.assertIn("skipped", res["reason"])

    def test_multi_file_not_attempted(self):
        rec = self.write_test("//@ check-pass\nfn main() {}\n")
        rec["single_file"] = False
        res = run.process_test(rec, self.rust_dir, Path("/fake/driver"),
                               stage2=True)
        self.assertEqual(res["status"], "not-attempted")
        self.assertIn("multi-file", res["bucket"])

    def test_incremental_suite_not_attempted(self):
        rec = self.write_test("fn main() {}\n", suite="incremental")
        res = run.process_test(rec, self.rust_dir, Path("/fake/driver"),
                               stage2=True)
        self.assertEqual(res["status"], "not-attempted")
        self.assertIn("incremental", res["bucket"])

    def test_compile_check_without_stage2_stays_out_of_scope(self):
        rec = self.write_test("//@ check-fail\nfn main() {}\n")
        res = run.process_test(rec, self.rust_dir, Path("/fake/driver"))
        self.assertEqual(res["status"], "not-attempted")
        self.assertIn("nature", res["bucket"])




class SkipBucketTests(unittest.TestCase):
    def test_allocated_constant_kind_is_shortened(self):
        totals = [("ConstantKind::Allocated (type Ty { id: 0, kind: "
                   "RigidTy(Adt(AdtDef(DefId { id: 2 })) })", 1)]
        self.assertEqual(run.skip_bucket(totals),
                         "conversion incomplete (ConstantKind::Allocated)")

    def test_by_reference_constant_bucket(self):
        totals = [("unsupported constant: by-reference constant", 2)]
        self.assertEqual(run.skip_bucket(totals),
                         "unsupported constant: by-reference constant")

    def test_no_totals(self):
        self.assertIn("no skip reason", run.skip_bucket([]))


if __name__ == "__main__":
    unittest.main()
