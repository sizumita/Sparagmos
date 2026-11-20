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

    def test_compile_flags_no_longer_block(self):
        # Stage 1 forwards compile-flags to both sides now.
        h = header("//@ run-pass\n//@ compile-flags: -C opt-level=3\n")
        self.assertIsNone(run.stage1_blocker(h, self.rec()))

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
        run.native_side = lambda *a, **kw: native
        run.sparagmos_side = lambda *a, **kw: sparagmos
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
        self.assertIsNone(run.stage2_blocker(
            header("//@ check-fail\n").view(None), self.rec()))

    def test_compile_flags_no_longer_block(self):
        # Flags are screened and forwarded, not blocked wholesale.
        h = header("//@ compile-flags: -Z unstable-options\n")
        self.assertIsNone(run.stage2_blocker(h.view(None), self.rec()))

    def test_add_minicore_blocks(self):
        h = header("//@ add-minicore\n")
        self.assertIn("minicore", run.stage2_blocker(h.view(None),
                                                     self.rec()))

    def test_rustc_env_blocks(self):
        h = header("//@ rustc-env: FOO=1\n")
        self.assertIn("rustc-env", run.stage2_blocker(h.view(None),
                                                      self.rec()))

    def test_scoped_blocker_only_blocks_its_revision(self):
        h = header("//@ revisions: a b\n//@[a] add-minicore\n")
        self.assertIn("minicore", run.stage2_blocker(h.view("a"),
                                                     self.rec()))
        self.assertIsNone(run.stage2_blocker(h.view("b"), self.rec()))

    def test_revisions_no_longer_block(self):
        # Revisions are judged per revision by fill_result_stage2.
        h = header("//@ revisions: a b\n//@[a] check-pass\n")
        self.assertIsNone(run.stage2_blocker(h.view("a"), self.rec()))
        self.assertIsNone(run.stage2_blocker(h.view("b"), self.rec()))


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
        run.convert_side = lambda *a, **kw: conv_result
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




class HeaderViewTests(unittest.TestCase):
    def test_revisions_parsed_in_order(self):
        h = header("//@ revisions: alpha beta-gamma\n")
        self.assertEqual(h.revisions(), ["alpha", "beta-gamma"])

    def test_unscoped_applies_to_every_revision(self):
        h = header("//@ revisions: a b\n//@ check-fail\n")
        self.assertTrue(h.view("a").has("check-fail"))
        self.assertTrue(h.view("b").has("check-fail"))
        self.assertTrue(h.view(None).has("check-fail"))

    def test_scoped_applies_only_to_its_revision(self):
        h = header("//@ revisions: a b\n//@[a] check-pass\n")
        self.assertTrue(h.view("a").has("check-pass"))
        self.assertFalse(h.view("b").has("check-pass"))
        self.assertFalse(h.view(None).has("check-pass"))

    def test_values_unscoped_then_scoped(self):
        h = header("//@ revisions: a\n//@ compile-flags: -C opt-level=1\n"
                   "//@[a] compile-flags: -C debuginfo=2\n")
        self.assertEqual(h.view("a").values("compile-flags"),
                         ["-C opt-level=1", "-C debuginfo=2"])
        self.assertEqual(h.view(None).values("compile-flags"),
                         ["-C opt-level=1"])

    def test_expectation_is_per_revision(self):
        h = header("//@ revisions: a b\n//@[a] check-pass\n"
                   "//@[b] check-fail\n")
        self.assertEqual(run.compile_expectation("ui", h.view("a")),
                         "success")
        self.assertEqual(run.compile_expectation("ui", h.view("b")),
                         "failure")


class SplitFlagsTests(unittest.TestCase):
    def test_plain_split(self):
        self.assertEqual(run.split_flags("-C opt-level=3 -Zfoo"),
                         ["-C", "opt-level=3", "-Zfoo"])

    def test_single_quoted_segment_keeps_spaces(self):
        self.assertEqual(
            run.split_flags("--cfg 'feature = \"x y\"' -O"),
            ["--cfg", 'feature = "x y"', "-O"])

    def test_compile_flags_accumulate_over_lines_and_scopes(self):
        h = header("//@ revisions: a\n//@ compile-flags: -A warnings\n"
                   "//@[a] compile-flags: -C opt-level=2\n")
        self.assertEqual(run.compile_flags(h.view("a")),
                         ["-A", "warnings", "-C", "opt-level=2"])
        self.assertEqual(run.compile_flags(h.view(None)),
                         ["-A", "warnings"])


class ScreenFlagsTests(unittest.TestCase):
    def test_ordinary_flags_pass(self):
        self.assertIsNone(run.screen_flags(
            ["-Znext-solver", "-C", "opt-level=3", "--crate-type", "rlib",
             "-Z", "mir-opt-level=2", "--cfg", "foo"]))

    def test_host_target_passes(self):
        self.assertIsNone(run.screen_flags(
            ["--target", "x86_64-unknown-linux-gnu"]))
        self.assertIsNone(run.screen_flags(
            ["--target=x86_64-unknown-linux-gnu"]))

    def test_other_target_blocked(self):
        for flags in (["--target", "aarch64-unknown-linux-gnu"],
                      ["--target=i686-pc-windows-msvc"]):
            reason = run.screen_flags(flags)
            self.assertIn("--target", reason)
            self.assertIn("not installed", reason)

    def test_print_blocked(self):
        self.assertIn("--print", run.screen_flags(["--print", "cfg"]))
        self.assertIn("--print", run.screen_flags(["--print=cfg"]))

    def test_sysroot_and_output_blocked(self):
        self.assertIn("--sysroot", run.screen_flags(["--sysroot=/x"]))
        self.assertIn("output", run.screen_flags(["--out-dir", "/x"]))
        self.assertIn("output", run.screen_flags(["-o", "prog"]))

    def test_unpretty_blocked_both_forms(self):
        self.assertIn("unpretty", run.screen_flags(["-Zunpretty=mir"]))
        self.assertIn("unpretty",
                      run.screen_flags(["-Z", "unpretty=mir"]))


class RevisionArgsTests(unittest.TestCase):
    def test_cfg_and_check_cfg(self):
        h = header("//@ revisions: a B-c\n")
        args = run.revision_args(h.view("a"), "a", ["a", "B-c"])
        self.assertEqual(args, ["--cfg", "a", "--check-cfg",
                                "cfg(test,FALSE,a,b_c)"])

    def test_no_auto_check_cfg_opts_out(self):
        h = header("//@ revisions: a\n//@ no-auto-check-cfg\n")
        self.assertEqual(run.revision_args(h.view("a"), "a", ["a"]),
                         ["--cfg", "a"])


class ProcessStage2RevisionTests(unittest.TestCase):
    """process_test(stage2=True) on revision tests, convert stubbed."""

    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile")
                                         .TemporaryDirectory()))
        (self.tmp / "tests" / "ui").mkdir(parents=True)
        self.rust_dir = self.tmp
        self.calls = []

    def write_test(self, body):
        p = self.rust_dir / "tests" / "ui" / "t.rs"
        p.write_text(body)
        return {"suite": "ui", "path": "tests/ui/t.rs",
                "nature": "compile-check", "single_file": True,
                "ffi": False, "env_excluded": False}

    def run_with_fake(self, rec, fake):
        orig = run.convert_side
        def stub(path, edition, driver_bin, rustc_args=()):
            self.calls.append(list(rustc_args))
            return fake(list(rustc_args))
        run.convert_side = stub
        try:
            return run.process_test(rec, self.rust_dir,
                                    Path("/fake/driver"), stage2=True)
        finally:
            run.convert_side = orig

    def test_all_revisions_pass(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@[a] check-pass\n//@[b] check-pass\n"
            "fn main() {}\n")
        res = self.run_with_fake(rec, lambda args: conv())
        self.assertEqual(res["status"], "pass")
        self.assertEqual(len(self.calls), 2)
        self.assertIn(["--cfg", "a"], [c[:2] for c in self.calls])
        self.assertIn(["--cfg", "b"], [c[:2] for c in self.calls])
        self.assertEqual(res["revisions"]["a"]["status"], "pass")
        self.assertEqual(res["revisions"]["b"]["status"], "pass")

    def test_one_revision_fails(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@[a] check-pass\n//@[b] check-pass\n"
            "fn main() {}\n")
        def fake(args):
            if args[:2] == ["--cfg", "b"]:
                return conv(exit_code=1, compilation_failed=True,
                            converted=None, skipped=None,
                            stderr="error[E0308]: x")
            return conv()
        res = self.run_with_fake(rec, fake)
        self.assertEqual(res["status"], "fail")
        self.assertIn("revision b", res["reason"])

    def test_out_of_environment_revision_is_skipped(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@[a] check-pass\n//@[b] check-pass\n"
            "//@[b] ignore-linux\nfn main() {}\n")
        res = self.run_with_fake(rec, lambda args: conv())
        self.assertEqual(res["status"], "pass")
        self.assertEqual(len(self.calls), 1)

    def test_all_revisions_out_of_environment(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@ ignore-linux\nfn main() {}\n")
        res = self.run_with_fake(rec, lambda args: conv())
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"], "out of environment")
        self.assertEqual(self.calls, [])

    def test_scoped_expectations_differ(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@[a] check-pass\n//@[b] check-fail\n"
            "fn main() {}\n")
        def fake(args):
            if args[:2] == ["--cfg", "b"]:
                return conv(exit_code=1, compilation_failed=True,
                            converted=None, skipped=None)
            return conv()
        res = self.run_with_fake(rec, fake)
        self.assertEqual(res["status"], "pass")

    def test_unhonourable_scoped_flags_block_the_record(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@[a] check-pass\n//@[b] check-pass\n"
            "//@[b] compile-flags: --target aarch64-unknown-linux-gnu\n"
            "fn main() {}\n")
        res = self.run_with_fake(rec, lambda args: conv())
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"], "not attempted (compile-flags)")
        self.assertIn("revision b", res["reason"])

    def test_unscoped_flags_reach_every_revision(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@ compile-flags: -C opt-level=2\n"
            "//@[a] check-pass\n//@[b] check-pass\nfn main() {}\n")
        res = self.run_with_fake(rec, lambda args: conv())
        self.assertEqual(res["status"], "pass")
        for call in self.calls:
            self.assertIn("opt-level=2", call)


class ProcessStage1FlagsTests(unittest.TestCase):
    """Stage 1 forwards compile-flags to both sides."""

    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile")
                                         .TemporaryDirectory()))
        (self.tmp / "tests" / "ui").mkdir(parents=True)
        self.rust_dir = self.tmp
        self.seen = {}

    def write_test(self, body):
        p = self.rust_dir / "tests" / "ui" / "t.rs"
        p.write_text(body)
        return {"suite": "ui", "path": "tests/ui/t.rs", "nature": "exec",
                "single_file": True, "ffi": False, "env_excluded": False}

    def run_with_capture(self, rec):
        orig_n, orig_s = run.native_side, run.sparagmos_side
        def fake_native(path, edition, workdir, extra_flags=()):
            self.seen["native_flags"] = list(extra_flags)
            return ({"exit_code": 0, "stdout": "", "stderr": ""}, None)
        def fake_spar(path, edition, driver_bin, cwd=None, rustc_args=()):
            self.seen["driver_args"] = list(rustc_args)
            return ({"exit_code": 0, "stdout": "", "stderr": ""},
                    "result", "")
        run.native_side, run.sparagmos_side = fake_native, fake_spar
        try:
            return run.process_test(rec, self.rust_dir,
                                    Path("/fake/driver"))
        finally:
            run.native_side, run.sparagmos_side = orig_n, orig_s

    def test_flags_forwarded_to_both_sides(self):
        rec = self.write_test(
            "//@ run-pass\n//@ compile-flags: -C opt-level=3\n"
            "fn main() {}\n")
        res = self.run_with_capture(rec)
        self.assertEqual(res["status"], "pass")
        self.assertEqual(self.seen["native_flags"],
                         ["-C", "opt-level=3"])
        self.assertEqual(self.seen["driver_args"], ["-C", "opt-level=3"])

    def test_screened_flags_not_attempted(self):
        rec = self.write_test(
            "//@ run-pass\n"
            "//@ compile-flags: --target aarch64-unknown-linux-gnu\n"
            "fn main() {}\n")
        res = self.run_with_capture(rec)
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"], "not attempted (compile-flags)")
        self.assertEqual(self.seen, {})


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
