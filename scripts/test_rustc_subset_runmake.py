#!/usr/bin/env python3
"""Unit tests for scripts/rustc-subset-runmake.py (metric 2(b),
procedure category).

Run with: python3 -m unittest scripts.test_rustc_subset_runmake
(or directly: python3 scripts/test_rustc_subset_runmake.py)

Classification, header gating and assertion checking are exercised on
small fixture recipes; no toolchain or driver is needed.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location(
    "rustc_subset_runmake", Path(__file__).parent / "rustc-subset-runmake.py")
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)


class StubNeeds:
    """needs-* evaluator stub: everything holds unless listed."""

    def __init__(self, false_names=(), unknown_names=()):
        self.false_names = set(false_names)
        self.unknown_names = set(unknown_names)

    def evaluate(self, name, value):
        if name in self.unknown_names:
            return None
        if name in self.false_names:
            return False
        return True


def gate(text, needs=None):
    return rm.header_gate(rm.Header(text), needs or StubNeeds())


def classify(src, files=("foo.rs", "rmake.rs"), needs=None):
    return rm.classify_recipe(src, set(files), needs or StubNeeds())


BARE_OUTFILE = """
//@ ignore-cross-compile

use run_make_support::{run, rustc};

fn main() {
    rustc().output("foo").input("foo.rs").run();
    run("foo");
}
"""

STDOUT_ASSERT = """
//@ ignore-windows

use run_make_support::{run, rustc};

fn main() {
    rustc().input("foo.rs").run();
    run("foo").assert_stdout_not_contains("cannot unwind");
}
"""


class StripCommentsTests(unittest.TestCase):
    def test_line_and_block_comments_removed(self):
        src = 'let a = 1; // note\n/* multi\nline */ let b = 2;'
        out = rm.strip_comments(src)
        self.assertNotIn("note", out)
        self.assertNotIn("multi", out)
        self.assertIn("let b = 2;", out)

    def test_string_literal_preserved(self):
        src = 'let url = "https://example.com"; // trailing'
        out = rm.strip_comments(src)
        self.assertIn('"https://example.com"', out)
        self.assertNotIn("trailing", out)


class HeaderGateTests(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(gate("fn main() {}\n")[0], "ok")

    def test_ignore_other_platform(self):
        self.assertEqual(gate("//@ ignore-windows\n")[0], "ok")
        self.assertEqual(gate("//@ ignore-apple\n")[0], "ok")

    def test_ignore_this_platform(self):
        self.assertEqual(gate("//@ ignore-linux\n")[0], "out")
        self.assertEqual(gate("//@ ignore-x86_64\n")[0], "out")
        self.assertEqual(gate("//@ ignore-gnu\n")[0], "out")

    def test_only(self):
        self.assertEqual(gate("//@ only-linux\n")[0], "ok")
        self.assertEqual(gate("//@ only-x86_64\n")[0], "ok")
        self.assertEqual(gate("//@ only-windows\n")[0], "out")
        # compiletest arch matching is exact: x86 does not match x86_64
        self.assertEqual(gate("//@ only-x86\n")[0], "out")

    def test_ignore_cross_compile_does_not_exclude(self):
        self.assertEqual(gate("//@ ignore-cross-compile\n")[0], "ok")

    def test_backends(self):
        self.assertEqual(gate("//@ ignore-backends: gcc\n")[0], "ok")
        self.assertEqual(gate("//@ ignore-backends: llvm\n")[0], "out")
        self.assertEqual(gate("//@ only-backends: llvm\n")[0], "ok")
        self.assertEqual(gate("//@ only-backends: cranelift\n")[0], "out")

    def test_needs(self):
        self.assertEqual(gate("//@ needs-target-std\n")[0], "ok")
        self.assertEqual(
            gate("//@ needs-enzyme\n",
                 StubNeeds(false_names={"needs-enzyme"}))[0], "out")
        self.assertEqual(
            gate("//@ needs-something-new\n",
                 StubNeeds(unknown_names={"needs-something-new"}))[0],
            "uneval")

    def test_cfg_directives(self):
        self.assertEqual(
            gate('//@ ignore-cfg(target_os = "windows")\n')[0], "ok")
        self.assertEqual(
            gate('//@ only-cfg(target_os = "windows")\n')[0], "out")
        self.assertEqual(
            gate('//@ only-cfg(all(target_os = "linux", '
                 'target_arch = "x86_64"))\n')[0], "ok")
        self.assertEqual(
            gate('//@ ignore-cfg(target_feature = "avx")\n')[0],
            "uneval")

    def test_scoped_directive_is_unevaluatable(self):
        self.assertEqual(gate("//@[rev1] ignore-windows\n")[0], "uneval")


class ClassifyTests(unittest.TestCase):
    def test_bare_outfile_is_reducible(self):
        cls, reason, recipe = classify(BARE_OUTFILE)
        self.assertEqual(cls, "reducible", reason)
        self.assertEqual(recipe.input_file, "foo.rs")
        self.assertEqual(recipe.edition, "2015")
        self.assertEqual(recipe.output_name, "foo")
        self.assertEqual(recipe.assertions, [])

    def test_stdout_assertion_extracted_verbatim(self):
        cls, reason, recipe = classify(STDOUT_ASSERT)
        self.assertEqual(cls, "reducible", reason)
        self.assertEqual(recipe.assertions,
                         [("assert_stdout_not_contains", "cannot unwind")])

    def test_edition_and_bin_crate_type(self):
        src = BARE_OUTFILE.replace(
            'rustc().output("foo")',
            'rustc().edition("2024").crate_type("bin").output("foo")')
        cls, reason, recipe = classify(src)
        self.assertEqual(cls, "reducible", reason)
        self.assertEqual(recipe.edition, "2024")

    def test_builder_options(self):
        src = BARE_OUTFILE.replace(
            "rustc().output", 'rustc().arg("-O").output')
        cls, reason, _ = classify(src)
        self.assertEqual(cls, "builder-options", reason)
        src = BARE_OUTFILE.replace(
            "rustc().output", 'rustc().cfg("foo").output')
        cls, reason, _ = classify(src)
        self.assertEqual(cls, "builder-options", reason)

    def test_two_compilations(self):
        src = BARE_OUTFILE.replace(
            'run("foo");',
            'rustc().input("foo.rs").output("bar").run();\n    run("foo");')
        cls, reason, _ = classify(src)
        self.assertEqual(cls, "not-single-program", reason)

    def test_external_tools(self):
        src = BARE_OUTFILE.replace(
            "use run_make_support::{run, rustc};",
            "use run_make_support::{cc, run, rustc};").replace(
            'run("foo");', 'cc().input("foo.c").output("libfoo.a").run();\n'
            '    run("foo");')
        cls, reason, _ = classify(src)
        self.assertEqual(cls, "external-tools", reason)

    def test_compile_verdict_only(self):
        src = """
use run_make_support::rustc;

fn main() {
    rustc().input("foo.rs").crate_type("rlib").run();
}
"""
        cls, reason, _ = classify(src)
        self.assertEqual(cls, "compile-verdict-only", reason)

    def test_expected_compile_failure(self):
        src = """
use run_make_support::rustc;

fn main() {
    rustc().input("foo.rs").run_fail();
}
"""
        cls, reason, _ = classify(src)
        self.assertEqual(cls, "compile-verdict-only", reason)

    def test_input_not_local(self):
        src = """
use run_make_support::{run, rustc, source_root};

fn main() {
    rustc().input(source_root().join("foo.rs")).output("foo").run();
    run("foo");
}
"""
        cls, reason, _ = classify(src)
        self.assertEqual(cls, "input-not-local", reason)

    def test_run_with_args_is_run_shape(self):
        src = """
use run_make_support::{run_with_args, rustc};

fn main() {
    rustc().input("foo.rs").run();
    run_with_args("foo", &["x"]);
}
"""
        cls, reason, _ = classify(src)
        self.assertEqual(cls, "run-shape", reason)

    def test_diff_is_artifact_checks(self):
        src = """
use run_make_support::{diff, run, rustc};

fn main() {
    rustc().input("foo.rs").run();
    run("foo");
    diff().expected_file("expected.txt").actual_file("out.txt").run();
}
"""
        cls, reason, _ = classify(src)
        self.assertEqual(cls, "artifact-checks", reason)

    def test_header_excludes_despite_reducible_body(self):
        src = "//@ only-windows\n" + BARE_OUTFILE
        cls, reason, _ = classify(src)
        self.assertEqual(cls, "out-of-environment", reason)

    def test_extra_import_is_recipe_shape(self):
        src = BARE_OUTFILE.replace(
            "use run_make_support::{run, rustc};",
            "use run_make_support::{env_var, run, rustc};")
        cls, reason, _ = classify(src)
        self.assertEqual(cls, "recipe-shape", reason)


class AssertionTests(unittest.TestCase):
    def test_equals_contains_not_contains(self):
        self.assertIsNone(rm.check_assertions(
            [("assert_stdout_equals", "hi\\n")], b"hi\n"))
        self.assertIsNone(rm.check_assertions(
            [("assert_stdout_contains", "ell")], b"hello"))
        self.assertIsNone(rm.check_assertions(
            [("assert_stdout_not_contains", "xyz")], b"hello"))
        self.assertIsNotNone(rm.check_assertions(
            [("assert_stdout_equals", "bye")], b"hi"))
        self.assertIsNotNone(rm.check_assertions(
            [("assert_stdout_contains", "xyz")], b"hello"))
        self.assertIsNotNone(rm.check_assertions(
            [("assert_stdout_not_contains", "ell")], b"hello"))

    def test_unescape(self):
        self.assertEqual(rm.rust_unescape("a\\nb"), "a\nb")
        self.assertEqual(rm.rust_unescape("\\u{41}"), "A")
        self.assertIsNone(rm.rust_unescape("bad\\q"))


class CfgEvalTests(unittest.TestCase):
    def test_any_not(self):
        self.assertTrue(rm.eval_cfg('any(target_os = "windows", '
                                    'target_os = "linux")'))
        self.assertFalse(rm.eval_cfg('not(target_os = "linux")'))
        self.assertIsNone(rm.eval_cfg('target_feature = "avx"'))
        # three-valued: any(false, unknown) is unknown
        self.assertIsNone(rm.eval_cfg('any(target_os = "windows", '
                                      'target_feature = "avx")'))
        # all(false, unknown) is false
        self.assertFalse(rm.eval_cfg('all(target_os = "windows", '
                                     'target_feature = "avx")'))


if __name__ == "__main__":
    unittest.main()
