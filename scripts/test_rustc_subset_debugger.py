#!/usr/bin/env python3
"""Unit tests for scripts/rustc-subset-debugger.py (exec-debugger
category).

Run with: python3 -m unittest scripts.test_rustc_subset_debugger
(or directly: python3 scripts/test_rustc_subset_debugger.py)

The script classification, verdict-form and eligibility logic is
exercised on small fixture headers and fixture files in a temporary
directory; the reference attempt is exercised with the driver
invocation stubbed out, so no toolchain or driver binary is needed.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location(
    "rustc_subset_debugger",
    Path(__file__).parent / "rustc-subset-debugger.py")
dbg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dbg)
stage1 = dbg.load_stage1()


def header(text):
    return stage1.Header(text)


def rec(single_file=True, revisions=1, ffi=False):
    return {"suite": "debuginfo", "path": "tests/debuginfo/x.rs",
            "nature": "exec-debugger", "single_file": single_file,
            "ffi": ffi, "revisions": revisions, "env_excluded": False}


class DebuggerScriptTests(unittest.TestCase):
    def test_counts_unscoped_directives(self):
        h = header("//@ gdb-command:run\n"
                   "//@ gdb-command:print x\n"
                   "//@ gdb-check:$1 = 5\n"
                   "//@ lldb-command:run\n")
        scripts = dbg.debugger_scripts(h)
        self.assertEqual(scripts["gdb"],
                         {"command": 2, "check": 1, "repr": 0})
        self.assertEqual(scripts["lldb"],
                         {"command": 1, "check": 0, "repr": 0})
        self.assertNotIn("cdb", scripts)

    def test_scoped_directives_count_for_their_debugger(self):
        h = header("//@[lto] lldb-check:[...]x = 4.5\n"
                   "//@[no-lto] lldb-check:[...]x = 4.5\n")
        scripts = dbg.debugger_scripts(h)
        self.assertEqual(scripts["lldb"]["check"], 2)

    def test_legacy_comment_without_at_is_not_a_directive(self):
        # compiletest at this tag only recognises the `//@` prefix,
        # so a plain `// gdb-command:` comment scripts nothing.
        h = header("// gdb-command:print x\nfn main() {}\n")
        self.assertEqual(dbg.debugger_scripts(h), {})

    def test_no_directives(self):
        self.assertEqual(dbg.debugger_scripts(header("fn main() {}\n")),
                         {})


class VerdictFormTests(unittest.TestCase):
    def test_check_lines_make_debugger_output(self):
        scripts = {"gdb": {"command": 1, "check": 1, "repr": 0}}
        self.assertEqual(dbg.verdict_form(scripts), "debugger-output")

    def test_repr_implies_a_check_line(self):
        scripts = {"lldb": {"command": 0, "check": 0, "repr": 1}}
        self.assertEqual(dbg.verdict_form(scripts), "debugger-output")

    def test_commands_without_checks_are_session_only(self):
        scripts = {"gdb": {"command": 3, "check": 0, "repr": 0}}
        self.assertEqual(dbg.verdict_form(scripts),
                         "debugger-session-only")

    def test_no_script_is_compile_and_load_only(self):
        self.assertEqual(dbg.verdict_form({}), "compile-and-load-only")


class FlagsBlockerTests(unittest.TestCase):
    def blocker(self, text):
        return dbg.flags_blocker(header(text))

    def test_no_flags(self):
        self.assertIsNone(self.blocker("fn main() {}\n"))

    def test_plain_g(self):
        self.assertIsNone(self.blocker("//@ compile-flags:-g\n"))
        self.assertIsNone(self.blocker("//@ compile-flags: -g\n"))

    def test_debuginfo_c_options(self):
        self.assertIsNone(
            self.blocker("//@ compile-flags:-C debuginfo=1\n"))
        self.assertIsNone(
            self.blocker("//@ compile-flags: -Cdebuginfo=2\n"))
        self.assertIsNone(self.blocker(
            "//@ compile-flags:-g -C collapse_macro_debuginfo=external\n"))

    def test_crate_name(self):
        self.assertIsNone(self.blocker(
            "//@ compile-flags:-g --crate-name=basic_types_globals\n"))

    def test_codegen_flags_block(self):
        self.assertIn("opt-level=3", self.blocker(
            "//@ compile-flags: -g -Copt-level=3\n"))
        self.assertIn("lto", self.blocker(
            "//@ compile-flags:-g -C lto\n"))
        self.assertIn("codegen-units=2", self.blocker(
            "//@ compile-flags:-g -Ccodegen-units=2\n"))

    def test_nightly_flags_block(self):
        self.assertIn("-Zmir-enable-passes", self.blocker(
            "//@ compile-flags: -g -Zmir-enable-passes=+SimplifyBounds\n"))


class RefBlockerTests(unittest.TestCase):
    def blocker(self, record, text):
        return dbg.ref_blocker(record, header(text), stage1)

    def test_plain_single_file_is_eligible(self):
        self.assertIsNone(self.blocker(rec(), "//@ compile-flags:-g\n"))

    def test_multi_file_blocks(self):
        self.assertIn("multi-file", self.blocker(
            rec(single_file=False), "//@ aux-build: aux.rs\n"))

    def test_revisions_block(self):
        self.assertIn("revisions", self.blocker(
            rec(revisions=2), "//@ revisions: lto no-lto\n"))

    def test_scoped_directive_blocks_as_revision(self):
        self.assertIn("revisions", self.blocker(
            rec(), "//@[lto] lldb-check:[...]x\n"))

    def test_ignore_test_blocks(self):
        self.assertIn("ignore-test", self.blocker(
            rec(), "//@ ignore-test (broken)\n"))

    def test_only_cdb_is_out_of_environment(self):
        reason = self.blocker(rec(), "//@ only-cdb\n")
        self.assertIn("only", reason)

    def test_ignore_gdb_does_not_block(self):
        # ignore-gdb gates one debugger variant, not the program.
        self.assertIsNone(self.blocker(rec(), "//@ ignore-gdb\n"))


class ClassifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.tmp / "tests" / "debuginfo").mkdir(parents=True)

    def classify(self, text, record=None):
        p = self.tmp / "tests" / "debuginfo" / "x.rs"
        p.write_text(text)
        return dbg.classify(record or rec(), self.tmp, stage1)

    def test_metric_status_is_never_pass(self):
        out = self.classify("//@ compile-flags:-g\n"
                            "//@ gdb-command:run\n"
                            "//@ gdb-check:$1 = 5\n"
                            "fn main() {}\n")
        self.assertEqual(out["status"], "not-attempted")
        self.assertIn("debugger stdout", out["reason"])
        self.assertEqual(out["verdict_form"], "debugger-output")
        self.assertEqual(out["debuggers"], ["gdb"])
        self.assertEqual(out["checks_total"], 1)
        self.assertEqual(out["ref_status"], "eligible")

    def test_compile_and_load_only_reason(self):
        out = self.classify("//@ compile-flags:-g\nfn main() {}\n")
        self.assertEqual(out["verdict_form"], "compile-and-load-only")
        self.assertEqual(out["debuggers"], [])

    def test_flags_recorded(self):
        out = self.classify("//@ compile-flags:-g\n"
                            "//@ disable-gdb-pretty-printers\n"
                            "//@ ignore-lldb\n"
                            "//@ gdb-repr: x\n"
                            "fn main() {}\n")
        self.assertTrue(out["disable_gdb_pretty_printers"])
        self.assertEqual(out["ignored_debuggers"], ["lldb"])
        self.assertEqual(out["repr_debuggers"], ["gdb"])
        # the repr pseudo-command implies a check line
        self.assertEqual(out["checks_total"], 1)
        self.assertEqual(out["verdict_form"], "debugger-output")

    def test_blocked_reference_keeps_reason(self):
        out = self.classify("//@ only-cdb\nfn main() {}\n")
        self.assertEqual(out["ref_status"], "not-attempted")
        self.assertIn("only", out["ref_reason"])
        self.assertTrue(out["only_cdb"])


class AttemptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def _attempt(self, side_result):
        out = {"path": "tests/debuginfo/x.rs", "ref_status": "eligible",
               "ref_reason": "", "driver_exit": None, "_edition": "2021"}

        class FakeStage1:
            @staticmethod
            def sparagmos_side(path, edition, driver_bin):
                return side_result

        return dbg.attempt(out, self.tmp, Path("driver"), FakeStage1)

    def test_program_runs(self):
        out = self._attempt(({"exit_code": 0}, "result", ""))
        self.assertEqual(out["ref_status"], "program-runs")
        self.assertEqual(out["driver_exit"], 0)

    def test_program_panics(self):
        out = self._attempt(({"exit_code": 101}, "result", ""))
        self.assertEqual(out["ref_status"], "program-panics")
        self.assertIn("panicked", out["ref_reason"])

    def test_unsupported_keeps_driver_reason(self):
        out = self._attempt((None, "unsupported", "no body for `main`"))
        self.assertEqual(out["ref_status"], "unsupported")
        self.assertEqual(out["ref_reason"], "no body for `main`")

    def test_timeout(self):
        out = self._attempt((None, "timeout", "driver timed out"))
        self.assertEqual(out["ref_status"], "timeout")

    def test_driver_error(self):
        out = self._attempt((None, "driver-error", "driver exit 3"))
        self.assertEqual(out["ref_status"], "driver-error")


if __name__ == "__main__":
    unittest.main()
