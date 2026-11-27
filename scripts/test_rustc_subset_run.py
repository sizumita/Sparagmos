#!/usr/bin/env python3
"""Unit tests for scripts/rustc-subset-run.py (metric 2(b) runner).

Run with: python3 -m unittest scripts.test_rustc_subset_run
(or directly: python3 scripts/test_rustc_subset_run.py)

The verdict logic is exercised on small fixture headers/results;
process_test is exercised with the two process sides stubbed out, so
no toolchain or driver is needed.
"""

import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location(
    "rustc_subset_run", Path(__file__).parent / "rustc-subset-run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def header(text):
    return run.Header(text)


def _process_terminated(pid):
    """Whether a process has terminated, zombie included.

    A terminated process whose parent has not reaped it stays in
    /proc as a zombie (state Z). Orphans reparented to an init
    that never reaps keep that entry forever, so disappearance
    from /proc is not a portable termination signal (Issue #486).
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    # Field 2 (comm) is parenthesised and may itself contain
    # spaces or parens; the state letter is the first field after
    # the closing paren.
    fields = stat.rpartition(")")[2].split()
    return bool(fields) and fields[0] == "Z"


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

    def test_directive_with_trailing_remark(self):
        # compiletest ends the directive name at the first space;
        # an unbalanced parenthesized remark must not drop the
        # directive (mirrors compiletest's line parser).
        h = header("//@ build-pass (FIXME(62277): could be check-pass?\n")
        self.assertTrue(h.has("build-pass"))
        self.assertEqual(
            run.compile_expectation("ui", h), "success")

    def test_directive_with_balanced_remark_keeps_paren_value(self):
        h = header("//@ check-pass (a remark)\n")
        self.assertTrue(h.has("check-pass"))
        self.assertEqual(h.values("check-pass"), ["a remark"])

    def test_non_directive_lines_ignored(self):
        h = header("// run-pass\n/* //@ run-fail */\n")
        self.assertIsNone(run.run_mode(h))

    def test_scoped_directive_with_space_after_prefix(self):
        # compiletest trims whitespace after `//@` before the
        # optional [revision] scope, so `//@ [lib] compile-flags:`
        # is the same directive as `//@[lib] compile-flags:`.
        h = header("//@ revisions: lib bin\n"
                   "//@ [lib] compile-flags: --crate-type lib\n"
                   "//@[bin] compile-flags: --crate-type bin\n")
        self.assertEqual(run.compile_flags(h.view("lib")),
                         ["--crate-type", "lib"])
        self.assertEqual(run.compile_flags(h.view("bin")),
                         ["--crate-type", "bin"])


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

    def test_run_crash_has_no_blocker(self):
        # run-crash is attempted and judged by crash class.
        h = header("//@ run-crash\n")
        self.assertIsNone(run.stage1_blocker(h, self.rec()))

    def test_ignore_test_blocks_without_run_directive(self):
        # compiletest decides ignore-test before mode dispatch, so a
        # file whose only mode directive is check-pass still reports
        # ignore-test rather than the missing-run-mode blocker.
        h = header("//@ check-pass\n//@ ignore-test (broken)\n")
        self.assertIn("ignore-test", run.stage1_blocker(h, self.rec()))

    def test_ignore_test_blocks_before_revisions(self):
        h = header("//@ revisions: a b\n//@ ignore-test\n")
        self.assertIn("ignore-test", run.stage1_blocker(h, self.rec()))


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

    def test_only_lines_are_conjunctive(self):
        # compiletest decides each only-* line on its own: one
        # non-matching line ignores the test even if another matches.
        h = header("//@ only-linux\n//@ only-windows\n")
        status, why = run.applicability(h)
        self.assertEqual(status, "out-of-environment")
        self.assertIn("windows", why)

    def test_only_elf_applicable(self):
        # The host target's binary format is ELF, and compiletest
        # registers an `elf` condition from it.
        status, _ = run.applicability(header("//@ only-elf\n"))
        self.assertEqual(status, "applicable")

    def test_only_host_triple_applicable(self):
        # Directive names include underscores (target triples);
        # the line must be parsed, not dropped.
        status, _ = run.applicability(
            header("//@ only-x86_64-unknown-linux-gnu\n"))
        self.assertEqual(status, "applicable")

    def test_only_x86_64_applicable(self):
        status, _ = run.applicability(header("//@ only-x86_64\n"))
        self.assertEqual(status, "applicable")

    def test_ignore_x86_64_out_of_environment(self):
        status, why = run.applicability(header("//@ ignore-x86_64\n"))
        self.assertEqual(status, "out-of-environment")
        self.assertIn("ignore-x86_64", why)

    def test_cfg_expr(self):
        h = header('//@ ignore-[target_os = "linux"]\n'.replace(
            "ignore-[", "ignore-cfg(").replace("]", ")"))
        status, _ = run.applicability(h)
        self.assertEqual(status, "out-of-environment")

    def test_channel_conditions_follow_the_release_channel(self):
        # The pinned toolchain is the official release build, whose
        # declared channel is stable, so exactly one channel
        # condition is true -- as in compiletest, never all false.
        self.assertTrue(run.eval_condition("stable"))
        self.assertFalse(run.eval_condition("beta"))
        self.assertFalse(run.eval_condition("nightly"))
        status, why = run.applicability(header("//@ ignore-stable\n"))
        self.assertEqual(status, "out-of-environment")
        self.assertIn("ignore-stable", why)
        status, _ = run.applicability(header("//@ only-stable\n"))
        self.assertEqual(status, "applicable")
        status, _ = run.applicability(header("//@ only-nightly\n"))
        self.assertEqual(status, "out-of-environment")
        status, _ = run.applicability(header("//@ ignore-nightly\n"))
        self.assertEqual(status, "applicable")

    def test_unknown_needs_unevaluatable(self):
        status, why = run.applicability(header("//@ needs-sanitizer-support\n"))
        self.assertEqual(status, "unevaluatable")
        self.assertIn("needs-sanitizer-support", why)

    def test_ignore_auxiliary_out_of_environment(self):
        # compiletest registers the `auxiliary` condition as true
        # unconditionally (prepare_conditions in directives/cfg.rs):
        # the file is "used by another main test file" and is never
        # run as a test itself, on any host.
        self.assertTrue(run.eval_condition("auxiliary"))
        status, why = run.applicability(header("//@ ignore-auxiliary\n"))
        self.assertEqual(status, "out-of-environment")
        self.assertIn("ignore-auxiliary", why)
        # An ordinary bare-name ignore still follows the host: it
        # does not match here.
        self.assertFalse(run.eval_condition("windows"))


class FakeFacts:
    """Canned ToolchainFacts answers for eval_needs tests."""

    def __init__(self, **kw):
        self.kw = kw

    def sanitizers(self):
        return self.kw.get("sanitizers")

    def sanitizer_runtime_present(self):
        return self.kw.get("sanitizer_runtime")

    def xray(self):
        return self.kw.get("xray")

    def dynamic_linking(self):
        return self.kw.get("dynamic_linking")

    def relocation_model(self):
        return self.kw.get("relocation_model")

    def profiler_runtime(self):
        return self.kw.get("profiler_runtime")

    def rust_lld(self):
        return self.kw.get("rust_lld")

    def git_hash(self):
        return self.kw.get("git_hash")

    def llvm_zstd(self):
        return self.kw.get("llvm_zstd")

    def has_mnemonic(self, mnemonic):
        return self.kw.get("mnemonics", {}).get(mnemonic)

    def supported_crate_types(self):
        return self.kw.get("crate_types")

    def target_has_atomic(self):
        return self.kw.get("atomic")

    def llvm_component_present(self, component):
        return self.kw.get("components", {}).get(component)

    def default_codegen_backend(self):
        return self.kw.get("default_backend")


def view(text):
    return header(text).view(None)


class NeedsEvalTests(unittest.TestCase):
    def test_host_rules_need_no_facts(self):
        v = view("//@ run-pass\n")
        self.assertIs(run.eval_needs("needs-threads", v, None), True)
        self.assertIs(run.eval_needs("needs-subprocess", v, None), True)
        self.assertIs(run.eval_needs("needs-asm-support", v, None), True)
        self.assertIs(run.eval_needs("needs-unwind", v, None), True)
        self.assertIs(run.eval_needs("needs-deterministic-layouts",
                                     v, None), True)
        self.assertIs(run.eval_needs("needs-dlltool", v, None), False)
        self.assertIs(run.eval_needs("needs-enzyme", v, None), False)
        self.assertIs(run.eval_needs("needs-offload", v, None), False)
        self.assertIs(run.eval_needs("needs-rustc-debug-assertions",
                                     v, None), False)

    def test_sanitizer_membership(self):
        v = view("//@ run-pass\n")
        f = FakeFacts(sanitizers={"address", "cfi"})
        self.assertIs(run.eval_needs("needs-sanitizer-address", v, f),
                      True)
        self.assertIs(run.eval_needs("needs-sanitizer-memory", v, f),
                      False)
        self.assertIs(run.eval_needs("needs-sanitizer-kasan", v, f),
                      False)
        self.assertIsNone(run.eval_needs("needs-sanitizer-address",
                                         v, None))
        self.assertIsNone(run.eval_needs("needs-sanitizer-address",
                                         v, FakeFacts()))

    def test_sanitizer_support_needs_list_and_runtime(self):
        v = view("//@ run-pass\n")
        self.assertIs(run.eval_needs(
            "needs-sanitizer-support", v,
            FakeFacts(sanitizers={"address"}, sanitizer_runtime=True)),
            True)
        self.assertIs(run.eval_needs(
            "needs-sanitizer-support", v,
            FakeFacts(sanitizers={"address"}, sanitizer_runtime=False)),
            False)
        self.assertIs(run.eval_needs(
            "needs-sanitizer-support", v,
            FakeFacts(sanitizers=set(), sanitizer_runtime=True)), False)
        self.assertIsNone(run.eval_needs(
            "needs-sanitizer-support", v,
            FakeFacts(sanitizers={"address"})))

    def test_spec_fact_passthrough(self):
        v = view("//@ run-pass\n")
        f = FakeFacts(xray=True, dynamic_linking=False,
                      relocation_model="pic", profiler_runtime=True,
                      rust_lld=False, git_hash=True, llvm_zstd=False,
                      mnemonics={"ret": True})
        self.assertIs(run.eval_needs("needs-xray", v, f), True)
        self.assertIs(run.eval_needs("needs-dynamic-linking", v, f),
                      False)
        self.assertIs(run.eval_needs("needs-relocation-model-pic",
                                     v, f), True)
        self.assertIs(run.eval_needs("needs-profiler-runtime", v, f),
                      True)
        self.assertIs(run.eval_needs("needs-rust-lld", v, f), False)
        self.assertIs(run.eval_needs("needs-git-hash", v, f), True)
        self.assertIs(run.eval_needs("needs-llvm-zstd", v, f), False)
        self.assertIs(run.eval_needs("needs-asm-ret", v, f), True)
        self.assertIsNone(run.eval_needs("needs-xray", v, FakeFacts()))

    def test_crate_type_list(self):
        v = view("//@ needs-crate-type: rlib, dylib\n")
        full = FakeFacts(crate_types={"bin", "cdylib", "dylib", "lib",
                                      "proc-macro", "rlib", "staticlib"})
        self.assertIs(run.eval_needs("needs-crate-type", v, full), True)
        self.assertIs(run.eval_needs(
            "needs-crate-type", v, FakeFacts(crate_types={"rlib"})),
            False)
        self.assertIsNone(run.eval_needs("needs-crate-type", v, None))
        bad = view("//@ needs-crate-type: nosuchtype\n")
        self.assertIsNone(run.eval_needs("needs-crate-type", bad, full))
        bare = view("//@ needs-crate-type\n")
        self.assertIsNone(run.eval_needs("needs-crate-type", bare, full))

    def test_target_has_atomic_list(self):
        f = FakeFacts(atomic={"8", "16", "32", "64", "ptr"})
        self.assertIs(run.eval_needs(
            "needs-target-has-atomic",
            view("//@ needs-target-has-atomic: 8, 64\n"), f), True)
        self.assertIs(run.eval_needs(
            "needs-target-has-atomic",
            view("//@ needs-target-has-atomic: 128\n"), f), False)
        self.assertIsNone(run.eval_needs(
            "needs-target-has-atomic",
            view("//@ needs-target-has-atomic: 7\n"), f))

    def test_llvm_components(self):
        v = view("//@ needs-llvm-components: x86 aarch64\n")
        f = FakeFacts(components={"x86": True, "aarch64": True})
        self.assertIs(run.eval_needs("needs-llvm-components", v, f),
                      True)
        f2 = FakeFacts(components={"x86": True, "aarch64": False})
        self.assertIs(run.eval_needs("needs-llvm-components", v, f2),
                      False)
        # An unknown component name is undecidable, not absent.
        v2 = view("//@ needs-llvm-components: x86 nosuchthing\n")
        self.assertIsNone(run.eval_needs("needs-llvm-components",
                                         v2, f))
        self.assertIsNone(run.eval_needs("needs-llvm-components",
                                         v, None))

    def test_backends(self):
        f = FakeFacts(default_backend="llvm")
        self.assertIs(run.eval_needs(
            "needs-backends", view("//@ needs-backends: llvm\n"), f),
            True)
        self.assertIs(run.eval_needs(
            "needs-backends", view("//@ needs-backends: gcc\n"), f),
            False)
        self.assertIs(run.eval_needs(
            "needs-backends",
            view("//@ needs-backends: gcc llvm\n"), f), True)
        self.assertIsNone(run.eval_needs(
            "needs-backends", view("//@ needs-backends: llvm\n"),
            FakeFacts()))
        self.assertIsNone(run.eval_needs(
            "needs-backends", view("//@ needs-backends: wat\n"), f))

    def test_unknown_needs_stays_undecidable(self):
        v = view("//@ run-pass\n")
        self.assertIsNone(run.eval_needs("needs-std-remap-debuginfo",
                                         v, FakeFacts()))
        self.assertIsNone(run.eval_needs("needs-abort", v, FakeFacts()))

    def test_applicability_outcomes(self):
        status, why = run.applicability(
            header("//@ run-pass\n//@ needs-enzyme\n"))
        self.assertEqual(status, "out-of-environment")
        self.assertIn("needs-enzyme", why)
        status, _ = run.applicability(
            header("//@ run-pass\n//@ needs-threads\n"))
        self.assertEqual(status, "applicable")
        f = FakeFacts(sanitizers={"address"})
        status, _ = run.applicability(
            header("//@ run-pass\n//@ needs-sanitizer-address\n"),
            facts=f)
        self.assertEqual(status, "applicable")
        status, why = run.applicability(
            header("//@ run-pass\n//@ needs-sanitizer-address\n"),
            facts=FakeFacts(sanitizers=set()))
        self.assertEqual(status, "out-of-environment")
        self.assertIn("needs-sanitizer-address", why)


class ToolchainFactsTests(unittest.TestCase):
    """ToolchainFacts over a fixture toolchain: run_cmd is stubbed
    with canned query answers and a fixture sysroot, so no real
    rustc runs."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.sysroot = self.tmp / "sysroot"
        lib = (self.sysroot / "lib" / "rustlib"
               / run.HOST_TARGET / "lib")
        lib.mkdir(parents=True)
        (lib / "libprofiler_builtins-deadbeef.rlib").write_bytes(b"x")
        (lib / "librustc-stable_rt.asan.a").write_bytes(b"x")
        bin_dir = (self.sysroot / "lib" / "rustlib"
                   / run.HOST_TARGET / "bin")
        bin_dir.mkdir(parents=True)
        (bin_dir / "rust-lld").write_bytes(b"x")
        self.calls = []
        orig = run.run_cmd
        run.run_cmd = self._fake_run_cmd
        self.addCleanup(setattr, run, "run_cmd", orig)

    def _fake_run_cmd(self, argv, **kw):
        self.calls.append(argv)
        ok = {"exit_code": 0, "stdout": "", "stderr": "",
              "timeout": False}
        def answer(stdout):
            return {**ok, "stdout": stdout}
        if "--print" in argv and "sysroot" in argv:
            return answer(str(self.sysroot) + "\n")
        if any("all-target-specs-json" in a for a in argv):
            spec = {run.HOST_TARGET: {
                "supported-sanitizers": ["address", "cfi"],
                "supports-xray": True,
                "dynamic-linking": True,
                # no relocation-model: compiletest defaults to pic
            }}
            return answer(json.dumps(spec))
        if any("supported-crate-types" in a for a in argv):
            return answer("bin\nrlib\nstaticlib\n")
        if "--print=backend-has-mnemonic:ret" in argv:
            return answer("true\n")
        if "--print=backend-has-zstd" in argv:
            return answer("false\n")
        if "--print" in argv and "cfg" in argv:
            return answer('target_has_atomic="8"\n'
                          'target_has_atomic="64"\n'
                          'target_has_atomic="ptr"\n')
        if "-vV" in argv:
            return answer("rustc 1.100.0 (0ef1bc386 2026-11-09)\n"
                          "commit-hash: 0ef1bc386\n")
        if "--emit" in argv:
            target = argv[argv.index("--target") + 1]
            out = Path(argv[argv.index("-o") + 1])
            if target == "m68k-unknown-linux-gnu":
                return {**ok, "exit_code": 1,
                        "stderr": "error: no m68k backend\n"}
            out.write_bytes(b"fake object")
            return ok
        raise AssertionError(f"unexpected argv: {argv}")

    def test_queried_facts(self):
        f = run.ToolchainFacts()
        self.assertEqual(f.sysroot(), self.sysroot)
        self.assertEqual(f.sanitizers(), {"address", "cfi"})
        self.assertIs(f.xray(), True)
        self.assertIs(f.dynamic_linking(), True)
        self.assertEqual(f.relocation_model(), "pic")
        self.assertEqual(f.supported_crate_types(),
                         {"bin", "rlib", "staticlib"})
        self.assertEqual(f.target_has_atomic(), {"8", "64", "ptr"})
        self.assertIs(f.has_mnemonic("ret"), True)
        self.assertIs(f.llvm_zstd(), False)
        self.assertIs(f.git_hash(), True)

    def test_sysroot_facts(self):
        f = run.ToolchainFacts()
        self.assertIs(f.profiler_runtime(), True)
        self.assertIs(f.sanitizer_runtime_present(), True)
        self.assertIs(f.rust_lld(), True)

    def test_component_probes(self):
        f = run.ToolchainFacts()
        self.assertIs(f.llvm_component_present("x86"), True)
        self.assertIs(f.llvm_component_present("m68k"), False)
        self.assertIsNone(f.llvm_component_present("nosuchthing"))
        self.assertEqual(f.default_codegen_backend(), "llvm")

    def test_facts_compose_into_applicability(self):
        f = run.ToolchainFacts()
        status, _ = run.applicability(
            header("//@ check-pass\n"
                   "//@ needs-llvm-components: x86\n"), facts=f)
        self.assertEqual(status, "applicable")
        status, why = run.applicability(
            header("//@ check-pass\n"
                   "//@ needs-llvm-components: m68k\n"), facts=f)
        self.assertEqual(status, "out-of-environment")
        self.assertIn("needs-llvm-components", why)

    def test_query_failure_is_undecidable_not_false(self):
        orig = run.run_cmd
        run.run_cmd = lambda argv, **kw: {
            "exit_code": 1, "stdout": "", "stderr": "boom",
            "timeout": False}
        self.addCleanup(setattr, run, "run_cmd", orig)
        f = run.ToolchainFacts()
        self.assertIsNone(f.sanitizers())
        self.assertIsNone(f.xray())
        self.assertIsNone(f.supported_crate_types())
        self.assertIsNone(f.llvm_component_present("x86"))
        self.assertIsNone(f.default_codegen_backend())


class JudgeTests(unittest.TestCase):
    def test_run_pass_match(self):
        h = header("//@ run-pass\n")
        native = {"exit_code": 0, "stdout": "hi\n"}
        got = {"exit_code": 0, "stdout": "hi\n"}
        self.assertEqual(run.judge("run-pass", h, native, got), ("pass", ""))

    def test_stdout_mismatch_fails(self):
        # check-run-results is the one directive under which
        # compiletest compares run output at all.
        h = header("//@ run-pass\n//@ check-run-results\n")
        native = {"exit_code": 0, "stdout": "hi\n", "stderr": ""}
        got = {"exit_code": 0, "stdout": "bye\n", "stderr": ""}
        status, detail = run.judge("run-pass", h, native, got)
        self.assertEqual(status, "fail")
        self.assertIn("stdout", detail)

    def test_stdout_not_compared_without_check_run_results(self):
        # Without the directive compiletest judges the run by
        # its exit class alone: libtest completion order, thread
        # interleaving and address-valued prints make the stream
        # non-reproducible, and the verdict must not see them.
        h = header("//@ run-pass\n")
        native = {"exit_code": 0, "stdout": "hi\n", "stderr": ""}
        got = {"exit_code": 0, "stdout": "bye\n", "stderr": ""}
        self.assertEqual(run.judge("run-pass", h, native, got),
                         ("pass", ""))

    def test_check_run_results_thread_id_normalized(self):
        # compiletest's normalize_output rewrites the numeric
        # thread id in panic lines to $TID on both sides: the
        # native run's OS thread id and the guest's modelled
        # one can never agree literally.
        h = header("//@ run-fail\n//@ check-stdout\n"
                   "//@ check-run-results\n")
        native = {"exit_code": 101, "stdout": "",
                  "stderr": "thread 'foo' (3584591) panicked at "
                            "src/main.rs:2:5:\nboom\n"}
        got = {"exit_code": 101, "stdout": "",
               "stderr": "thread 'foo' (1) panicked at "
                         "src/main.rs:2:5:\nboom\n",
               "stderr_clean": True}
        import pathlib
        status, detail = run.judge(
            "run-fail", h, native, got,
            path=pathlib.Path("/x/tests/ui/t.rs"),
            base_dir="/tmp/td")
        self.assertEqual((status, detail), ("pass", ""))

    def test_check_run_results_stderr_compared_when_clean(self):
        h = header("//@ run-pass\n//@ check-run-results\n")
        native = {"exit_code": 0, "stdout": "", "stderr": "a\n"}
        got = {"exit_code": 0, "stdout": "", "stderr": "b\n",
               "stderr_clean": True}
        import pathlib
        status, detail = run.judge(
            "run-pass", h, native, got,
            path=pathlib.Path("/x/tests/ui/t.rs"),
            base_dir="/tmp/td")
        self.assertEqual(status, "fail")
        self.assertEqual(detail, "stderr mismatch")
        self.assertEqual(run.failure_bucket(detail),
                         "stderr mismatch")

    def test_check_run_results_stderr_skipped_when_not_clean(self):
        # A guest result without the clean-stderr mark (the JIT
        # run's shared channel) is judged without the stderr
        # half: the driver process's own diagnostics ride that
        # stream and are not the guest's output.
        h = header("//@ run-pass\n//@ check-run-results\n")
        native = {"exit_code": 0, "stdout": "", "stderr": "a\n"}
        got = {"exit_code": 0, "stdout": "",
               "stderr": "driver diagnostics\n"}
        import pathlib
        status, _ = run.judge(
            "run-pass", h, native, got,
            path=pathlib.Path("/x/tests/ui/t.rs"),
            base_dir="/tmp/td")
        self.assertEqual(status, "pass")

    def test_run_fail_default_status_101(self):
        h = header("//@ run-fail\n")
        native = {"exit_code": 101, "stdout": ""}
        got = {"exit_code": 101, "stdout": ""}
        self.assertEqual(run.judge("run-fail", h, native, got)[0], "pass")

    def test_run_fail_custom_failure_status(self):
        # failure-status constrains the COMPILE phase's status
        # only (compiletest's check_correct_failure_status); the
        # run verdict is the RunResult class, so a native exit
        # of 7 and a guest panic exit of 101 -- both Fail --
        # satisfy run-fail together.
        h = header("//@ run-fail\n//@ failure-status: 7\n")
        native = {"exit_code": 7, "stdout": "", "stderr": ""}
        got = {"exit_code": 101, "stdout": "", "stderr": ""}
        self.assertEqual(run.judge("run-fail", h, native, got)[0],
                         "pass")

    def test_run_fail_any_fail_class_code_passes(self):
        # A sanitizer runtime's default exit (23) and a plain
        # process::exit(1) are the same RunResult::Fail class
        # as a panic's 101.
        h = header("//@ run-fail\n")
        for native_code, got_code in ((23, 1), (1, 101), (101, 42)):
            native = {"exit_code": native_code, "stdout": "",
                      "stderr": ""}
            got = {"exit_code": got_code, "stdout": "", "stderr": ""}
            self.assertEqual(
                run.judge("run-fail", h, native, got)[0], "pass",
                (native_code, got_code))

    def test_run_fail_guest_success_fails(self):
        h = header("//@ run-fail\n")
        native = {"exit_code": 101, "stdout": "", "stderr": ""}
        got = {"exit_code": 0, "stdout": "", "stderr": ""}
        status, detail = run.judge("run-fail", h, native, got)
        self.assertEqual(status, "fail")
        self.assertIn("exit mismatch", detail)

    def test_run_pass_guest_failure_fails(self):
        h = header("//@ run-pass\n")
        native = {"exit_code": 0, "stdout": "", "stderr": ""}
        got = {"exit_code": 101, "stdout": "", "stderr": ""}
        self.assertEqual(run.judge("run-pass", h, native, got)[0],
                         "fail")

    def test_error_pattern_checked_against_run_stderr(self):
        h = header("//@ run-fail\n"
                   "//@ error-pattern: LeakSanitizer: detected "
                   "memory leaks\n")
        native = {"exit_code": 23, "stdout": "",
                  "stderr": "==1==ERROR: LeakSanitizer: detected "
                            "memory leaks\n"}
        self.assertIsNone(run.native_mismatch("run-fail", h,
                                              native))
        got = {"exit_code": 101, "stdout": "",
               "stderr": "panicked: no leak here\n"}
        status, detail = run.judge("run-fail", h, native, got)
        self.assertEqual(status, "fail")
        self.assertIn("error pattern", detail)

    def test_error_pattern_missing_on_native_is_oracle_failure(self):
        h = header("//@ run-fail\n//@ error-pattern: boom\n")
        native = {"exit_code": 101, "stdout": "",
                  "stderr": "something else\n"}
        self.assertIn("error pattern",
                      run.native_mismatch("run-fail", h, native))

    def test_error_pattern_check_stdout_uses_combined_stream(self):
        h = header("//@ run-fail\n//@ check-stdout\n"
                   "//@ error-pattern: in stdout\n")
        native = {"exit_code": 101, "stdout": "in stdout\n",
                  "stderr": ""}
        self.assertIsNone(run.native_mismatch("run-fail", h,
                                              native))
        # Without check-stdout the same pattern in stdout does
        # not count: get_output is the stderr alone.
        h2 = header("//@ run-fail\n"
                    "//@ error-pattern: in stdout\n")
        self.assertIn("error pattern",
                      run.native_mismatch("run-fail", h2, native))

    def test_regex_error_pattern(self):
        h = header("//@ run-fail\n"
                   "//@ regex-error-pattern: thread '.+' panicked\n")
        native = {"exit_code": 101, "stdout": "",
                  "stderr": "thread 'foo' panicked at x\n"}
        self.assertIsNone(run.native_mismatch("run-fail", h,
                                              native))
        got = {"exit_code": 101, "stdout": "", "stderr": "quiet\n"}
        self.assertEqual(run.judge("run-fail", h, native, got)[0],
                         "fail")

    def test_run_result_class_matches_compiletest(self):
        # runtest/ui.rs's RunResult: success is Pass, 1..=127 is
        # Fail, signal death (negative here) or a code above 127
        # is Crash.
        self.assertEqual(run.run_result_class(0), "pass")
        for code in (1, 101, 127):
            self.assertEqual(run.run_result_class(code), "fail")
        for code in (-6, -11, 128, 134, 255):
            self.assertEqual(run.run_result_class(code), "crash")

    def test_run_crash_both_sides_crash_passes(self):
        h = header("//@ run-crash\n")
        native = {"exit_code": -6, "stdout": ""}
        got = {"exit_code": 134, "stdout": "",
               "driver_stderr":
                   "sparagmos-driver run: guest crashed: abort\n"}
        self.assertEqual(run.judge("run-crash", h, native, got),
                         ("pass", ""))

    def test_run_crash_plain_exit_134_fails(self):
        # A guest that terminates normally with exit code 134
        # (std::process::exit(134)) has not crashed: the code
        # is crash-class, but the driver's guest-crash marker
        # is absent, so the record must not pass.
        h = header("//@ run-crash\n")
        native = {"exit_code": -6, "stdout": ""}
        got = {"exit_code": 134, "stdout": "", "stderr": "",
               "driver_stderr": ""}
        status, detail = run.judge("run-crash", h, native, got)
        self.assertEqual(status, "fail")
        self.assertIn("not evidenced", detail)
        self.assertEqual(run.failure_bucket(detail),
                         "guest crash not evidenced")

    def test_run_crash_other_crash_class_codes_fail(self):
        # The driver produces 134 only for a guest crash, so
        # any other crash-class code (a plain exit above 127,
        # or a signal death that reached the judge) is not
        # evidenced as a guest crash either.
        h = header("//@ run-crash\n")
        native = {"exit_code": -6, "stdout": ""}
        for code in (200, -11):
            got = {"exit_code": code, "stdout": "",
                   "driver_stderr":
                       "sparagmos-driver run: guest crashed: abort\n"}
            status, detail = run.judge("run-crash", h, native, got)
            self.assertEqual(status, "fail")
            self.assertIn("not evidenced", detail)

    def test_run_crash_marker_on_shared_stderr_passes(self):
        # The JIT run shares one stderr channel: the marker
        # there is the same evidence.
        h = header("//@ run-crash\n")
        native = {"exit_code": -11, "stdout": ""}
        got = {"exit_code": 134, "stdout": "",
               "stderr":
                   "sparagmos-driver run: guest crashed: abort\n"}
        self.assertEqual(run.judge("run-crash", h, native, got),
                         ("pass", ""))

    def test_run_crash_guest_unwind_fails(self):
        # The guest panicked and unwound to the top (101): a
        # failure exit, not a crash.
        h = header("//@ run-crash\n")
        native = {"exit_code": -6, "stdout": ""}
        got = {"exit_code": 101, "stdout": ""}
        status, detail = run.judge("run-crash", h, native, got)
        self.assertEqual(status, "fail")
        self.assertIn("did not crash", detail)
        self.assertEqual(run.failure_bucket(detail),
                         "guest did not crash")

    def test_run_crash_guest_completion_fails(self):
        h = header("//@ run-crash\n")
        native = {"exit_code": -11, "stdout": ""}
        got = {"exit_code": 0, "stdout": ""}
        status, detail = run.judge("run-crash", h, native, got)
        self.assertEqual(status, "fail")
        self.assertIn("did not crash", detail)

    def test_run_crash_stdout_mismatch_fails(self):
        h = header("//@ run-crash\n")
        native = {"exit_code": -6, "stdout": "before\n"}
        got = {"exit_code": 134, "stdout": "",
               "driver_stderr":
                   "sparagmos-driver run: guest crashed: abort\n"}
        status, detail = run.judge("run-crash", h, native, got)
        self.assertEqual(status, "fail")
        self.assertEqual(detail, "stdout mismatch")

    def test_native_mismatch_run_crash(self):
        h = header("//@ run-crash\n")
        self.assertIsNone(run.native_mismatch(
            "run-crash", h, {"exit_code": -6}))
        self.assertIsNone(run.native_mismatch(
            "run-crash", h, {"exit_code": 134}))
        self.assertIn("does not crash", run.native_mismatch(
            "run-crash", h, {"exit_code": 101}))
        self.assertIn("does not crash", run.native_mismatch(
            "run-crash", h, {"exit_code": 0}))

    def test_native_mismatch_fixed_code_modes(self):
        h = header("//@ run-pass\n")
        self.assertIsNone(run.native_mismatch(
            "run-pass", h, {"exit_code": 0, "stderr": ""}))
        self.assertIn("does not match", run.native_mismatch(
            "run-pass", h, {"exit_code": 101, "stderr": ""}))
        # run-fail is a class, not the fixed code 101.
        hf = header("//@ run-fail\n")
        self.assertIsNone(run.native_mismatch(
            "run-fail", hf, {"exit_code": 23, "stderr": ""}))
        self.assertIsNone(run.native_mismatch(
            "run-fail", hf, {"exit_code": 1, "stderr": ""}))
        self.assertIn("does not match", run.native_mismatch(
            "run-fail", hf, {"exit_code": 0, "stderr": ""}))


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

    def test_ignore_auxiliary_compile_check_not_attempted(self):
        # A compile-check record for an auxiliary file is classified
        # out of environment by the stage-2 entry, before any side
        # runs: compiletest never runs the file as a test.
        rec = self.write_test("//@ ignore-auxiliary\npub fn helper() {}\n")
        rec["nature"] = "compile-check"
        res = run.process_test(rec, self.rust_dir, Path("/fake/driver"),
                               stage2=True)
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"], "out of environment")
        self.assertIn("ignore-auxiliary", res["reason"])




class RustdocHostToolScopeTests(unittest.TestCase):
    """rustdoc-check suites judged by host-side tooling (issue #365).

    The classification returns before any file or process side is
    touched, so the records need no fixture files.
    """

    def rec(self, suite):
        return {"suite": suite, "path": f"tests/{suite}/t.rs",
                "nature": "rustdoc-check", "single_file": True,
                "ffi": False, "env_excluded": False}

    def run_rec(self, suite, jit=False):
        return run.process_test(self.rec(suite), Path("/nonexistent"),
                                Path("/fake/driver"), jit=jit)

    def test_rustdoc_js_suites_need_a_host_javascript_engine(self):
        for suite in ("rustdoc-js", "rustdoc-js-std"):
            res = self.run_rec(suite)
            self.assertEqual(res["status"], "not-attempted")
            self.assertEqual(res["bucket"],
                             "not attempted (host JavaScript engine)")
            self.assertIn("tester.js", res["reason"])
            self.assertIn("issue #365", res["reason"])

    def test_rustdoc_gui_needs_a_host_browser(self):
        res = self.run_rec("rustdoc-gui")
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"], "not attempted (host browser)")
        self.assertIn("browser-ui-test", res["reason"])
        self.assertIn("issue #365", res["reason"])

    def test_classification_applies_to_the_jit_pass_too(self):
        res = self.run_rec("rustdoc-js", jit=True)
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"],
                         "not attempted (host JavaScript engine)")

    def test_helper_classifies_only_the_host_tool_suites(self):
        # main's carry-through for records no stage selects
        # consults the same helper, so pin its contract directly.
        self.assertIsNotNone(run.rustdoc_host_tool_scope(self.rec("rustdoc-js")))
        self.assertIsNotNone(run.rustdoc_host_tool_scope(self.rec("rustdoc-gui")))
        self.assertIsNone(run.rustdoc_host_tool_scope(self.rec("rustdoc-html")))
        exec_rec = self.rec("rustdoc-js")
        exec_rec["nature"] = "exec"
        self.assertIsNone(run.rustdoc_host_tool_scope(exec_rec))

    def test_rustdoc_html_keeps_the_generic_nature_bucket(self):
        # The html/ui/json suites are the rustdoc stage's own
        # territory (issue #364); until it dispatches them they
        # keep the generic out-of-scope bucket.
        for suite in ("rustdoc-html", "rustdoc-ui", "rustdoc-json"):
            res = self.run_rec(suite)
            self.assertEqual(res["status"], "not-attempted")
            self.assertEqual(res["bucket"], "out of stage-1 scope (nature)")


class CoverageScopeTests(unittest.TestCase):
    """exec records of the coverage suites (issue #514).

    The coverage-run-rustdoc classification, and the coverage
    classification on the JIT path, return before any file or
    process side is touched. The coverage suite itself is
    judged by stage 1 on the interpreter path; the first test
    drives that path with the process side stubbed.
    """

    def rec(self, suite, nature="exec"):
        return {"suite": suite, "path": f"tests/{suite}/t.rs",
                "nature": nature, "single_file": True,
                "ffi": False, "env_excluded": False}

    def run_rec(self, suite, jit=False):
        return run.process_test(self.rec(suite), Path("/nonexistent"),
                                Path("/fake/driver"), jit=jit)

    def test_coverage_suite_is_judged_by_stage_1(self):
        # On the interpreter path a coverage record is judged by
        # its two coverage outputs: with the driver side stubbed
        # to return outputs matching the snapshots, it passes.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            test_dir = root / "tests" / "coverage"
            test_dir.mkdir(parents=True)
            (test_dir / "t.rs").write_text("fn main() {}\n")
            (test_dir / "t.cov-map").write_text(
                "Function name: t::main\n"
                "- file 0 => $DIR/t.rs\n")
            (test_dir / "t.coverage").write_text(
                "   LL|      1|fn main() {}\n\n")

            def fake_run_cmd(argv, **kw):
                mode = argv[1]
                if mode == "coverage-map":
                    return {"exit_code": 0, "timeout": False,
                            "stdout": "Function name: t::main\n"
                                      f"- file 0 => {test_dir}/t.rs\n",
                            "stderr": ""}
                assert mode == "coverage-run", argv
                return {"exit_code": 0, "timeout": False,
                        "stdout": "    1|      1|fn main() {}\n\n",
                        "stderr": ""}

            with unittest.mock.patch.object(run, "run_cmd",
                                            fake_run_cmd):
                res = run.process_test(self.rec("coverage"), root,
                                       Path("/fake/driver"))
            self.assertEqual(res["status"], "pass", res)
            self.assertEqual(res["mode"], "coverage")

    def test_coverage_run_rustdoc_needs_coverage_instrumentation(self):
        res = self.run_rec("coverage-run-rustdoc")
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"],
                         "not attempted (coverage instrumentation)")
        self.assertIn("doctest", res["reason"])
        self.assertIn("issue #514", res["reason"])

    def test_classification_applies_to_the_jit_pass_too(self):
        res = self.run_rec("coverage", jit=True)
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"],
                         "not attempted (coverage instrumentation)")

    def test_helper_classifies_only_the_coverage_suites(self):
        # main's carry-through for records no stage selects
        # consults the same helper, so pin its contract directly.
        self.assertIsNotNone(run.coverage_scope(self.rec("coverage")))
        self.assertIsNotNone(
            run.coverage_scope(self.rec("coverage-run-rustdoc")))
        self.assertIsNone(run.coverage_scope(self.rec("ui")))
        self.assertIsNone(
            run.coverage_scope(self.rec("coverage",
                                        nature="compile-check")))


class KnownBugScopeTests(unittest.TestCase):
    """Exec views pinning a known codegen bug (issue #546).

    The helper classifies from the record and the directive
    view alone; the process fixtures stub both sides to a
    passing pair, so a verdict of not-attempted can only come
    from the classification.
    """

    def rec(self, path, nature="exec"):
        return {"suite": "ui", "path": path, "nature": nature,
                "single_file": True, "ffi": False,
                "env_excluded": False}

    def view(self, text):
        return header(text).view(None)

    def test_codegen_known_bug_run_view_is_classified(self):
        scope = run.known_bug_scope(
            self.rec("tests/ui/codegen/equal-pointers-unequal/"
                     "as-cast/inline1.rs"),
            self.view("//@ known-bug: #107975\n"
                      "//@ compile-flags: -Copt-level=2\n"
                      "//@ run-pass\nfn main() {}\n"))
        self.assertIsNotNone(scope)
        reason, bucket = scope
        self.assertEqual(bucket, "not attempted (known-bug)")
        self.assertIn("#107975", reason)
        self.assertIn("#529", reason)
        self.assertIn("#546", reason)

    def test_known_bug_outside_the_codegen_subtree_stays(self):
        # Front-end known bugs are reproduced by the driver's
        # embedded rustc by construction; both such records
        # pass at the 1.100.0 tag and stay attempted.
        for path in ("tests/ui/macros/"
                     "metavar_cross_edition_recursive_macros.rs",
                     "tests/ui/view-types/must-restrict.rs"):
            self.assertIsNone(run.known_bug_scope(
                self.rec(path),
                self.view("//@ known-bug: #1\n//@ run-pass\n")))

    def test_codegen_without_known_bug_stays(self):
        # The subtree's fixed-miscompile regression tests
        # assert the corrected behaviour and stay attempted.
        self.assertIsNone(run.known_bug_scope(
            self.rec("tests/ui/codegen/miscompile.rs"),
            self.view("//@ run-pass\n")))

    def test_compile_check_nature_stays(self):
        # Stage 2 judges the compile outcome; the directive's
        # only compiletest effect there (the annotation
        # relaxation) is not one stage 2 performs.
        self.assertIsNone(run.known_bug_scope(
            self.rec("tests/ui/codegen/t.rs",
                     nature="compile-check"),
            self.view("//@ known-bug: #1\n//@ run-pass\n")))

    def test_view_without_a_run_mode_stays(self):
        self.assertIsNone(run.known_bug_scope(
            self.rec("tests/ui/codegen/t.rs"),
            self.view("//@ known-bug: #1\n//@ check-pass\n")))

    def test_scoped_directive_classifies_only_its_revision(self):
        h = header("//@ revisions: rev1\n//@ run-pass\n"
                   "//@[rev1] known-bug: #1\nfn main() {}\n")
        rec = self.rec("tests/ui/codegen/t.rs")
        self.assertIsNone(run.known_bug_scope(rec, h.view(None)))
        self.assertIsNotNone(
            run.known_bug_scope(rec, h.view("rev1")))

    def test_carry_scope_reads_the_header(self):
        # main's carry-through for records no stage selects
        # consults the same helper through this reader, so
        # both paths record the same verdict.
        with __import__("tempfile").TemporaryDirectory() as td:
            rust_dir = Path(td)
            p = rust_dir / "tests" / "ui" / "codegen" / "t.rs"
            p.parent.mkdir(parents=True)
            p.write_text("//@ known-bug: #107975\n"
                         "//@ run-pass\nfn main() {}\n")
            scope = run.known_bug_carry_scope(
                self.rec("tests/ui/codegen/t.rs"), rust_dir)
            self.assertIsNotNone(scope)
            self.assertEqual(scope[1],
                             "not attempted (known-bug)")
            self.assertIsNone(run.known_bug_carry_scope(
                self.rec("tests/ui/codegen/missing.rs"), rust_dir))

    def test_carry_scope_sees_revision_scoped_directives(self):
        # view(None) alone cannot classify either file: in the
        # first the known-bug directive is scoped to rev1, in
        # the second the run mode is scoped as well. The
        # carry-through aggregates revisions the way
        # fill_result_revisions does, so it consults every
        # declared revision's view.
        with __import__("tempfile").TemporaryDirectory() as td:
            rust_dir = Path(td)
            p = rust_dir / "tests" / "ui" / "codegen" / "t.rs"
            p.parent.mkdir(parents=True)
            rec = self.rec("tests/ui/codegen/t.rs")
            p.write_text("//@ revisions: rev1\n"
                         "//@ run-pass\n"
                         "//@[rev1] known-bug: #107975\n"
                         "fn main() {}\n")
            scope = run.known_bug_carry_scope(rec, rust_dir)
            self.assertIsNotNone(scope)
            self.assertEqual(scope[1],
                             "not attempted (known-bug)")
            p.write_text("//@ revisions: rev1\n"
                         "//@[rev1] known-bug: #107975\n"
                         "//@[rev1] run-pass\n"
                         "fn main() {}\n")
            scope = run.known_bug_carry_scope(rec, rust_dir)
            self.assertIsNotNone(scope)
            self.assertEqual(scope[1],
                             "not attempted (known-bug)")

    def test_carry_scope_ignores_undeclared_revisions(self):
        # A directive scoped to a revision the file does not
        # declare never applies under compiletest, so it does
        # not classify the record either.
        with __import__("tempfile").TemporaryDirectory() as td:
            rust_dir = Path(td)
            p = rust_dir / "tests" / "ui" / "codegen" / "t.rs"
            p.parent.mkdir(parents=True)
            p.write_text("//@ revisions: rev1\n"
                         "//@ run-pass\n"
                         "//@[other] known-bug: #107975\n"
                         "fn main() {}\n")
            self.assertIsNone(run.known_bug_carry_scope(
                self.rec("tests/ui/codegen/t.rs"), rust_dir))

    def process_fixture(self, body):
        tmp = Path(self.enterContext(__import__("tempfile")
                                     .TemporaryDirectory()))
        (tmp / "tests" / "ui" / "codegen").mkdir(parents=True)
        (tmp / "tests" / "ui" / "codegen" / "t.rs").write_text(body)
        return tmp, self.rec("tests/ui/codegen/t.rs")

    def run_stubbed(self, rec, rust_dir, jit=False):
        orig_n, orig_s = run.native_side, run.sparagmos_side
        run.native_side = lambda *a, **kw: (
            {"exit_code": 0, "stdout": "", "stderr": ""}, None)
        run.sparagmos_side = lambda *a, **kw: (
            {"exit_code": 0, "stdout": "", "stderr": ""},
            "result", "")
        try:
            return run.process_test(rec, rust_dir,
                                    Path("/fake/driver"), jit=jit)
        finally:
            run.native_side, run.sparagmos_side = orig_n, orig_s

    def test_process_test_records_the_bucket(self):
        rust_dir, rec = self.process_fixture(
            "//@ known-bug: #107975\n//@ run-pass\n"
            "fn main() {}\n")
        res = self.run_stubbed(rec, rust_dir)
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"],
                         "not attempted (known-bug)")
        self.assertIn("#107975", res["reason"])

    def test_process_test_multi_file_record_is_classified(self):
        # The classification says what the verdict is, so it
        # is recorded before stage 1's multi-file scope
        # exclusion: a multi-file codegen known-bug record
        # keeps the known-bug bucket instead of the generic
        # out-of-scope one.
        rust_dir, rec = self.process_fixture(
            "//@ known-bug: #107975\n//@ run-pass\n"
            "fn main() {}\n")
        rec["single_file"] = False
        res = self.run_stubbed(rec, rust_dir)
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"],
                         "not attempted (known-bug)")

    def test_process_test_revision_view_is_classified_before_screening(self):
        # Per revision, the classification is consulted before
        # flag screening (as on the unrevisioned path): rev1's
        # flags would be screened (--print replaces the
        # compilation), but its verdict is the known-bug one.
        rust_dir, rec = self.process_fixture(
            "//@ revisions: rev1\n"
            "//@ compile-flags: --print cfg\n"
            "//@[rev1] known-bug: #107975\n"
            "//@[rev1] run-pass\n"
            "fn main() {}\n")
        res = self.run_stubbed(rec, rust_dir)
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"],
                         "not attempted (known-bug)")
        self.assertIn("rev1", res["reason"])

    def test_process_test_records_the_bucket_under_jit(self):
        # The classification is consulted on the shared
        # fill_result path, so metric 4 records it identically.
        rust_dir, rec = self.process_fixture(
            "//@ known-bug: #107975\n//@ run-pass\n"
            "fn main() {}\n")
        res = self.run_stubbed(rec, rust_dir, jit=True)
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"],
                         "not attempted (known-bug)")


class SanitizerVerdictScopeTests(unittest.TestCase):
    """Sanitizer-report verdicts (issues #532, #509).

    The exec run-fail family is classified by sanitizer_scope;
    the compile-check run-fail-or-crash family by
    sanitizer_run_fail_or_crash_scope. Both helpers classify
    from the directive view alone.
    """

    def rec(self, nature="exec"):
        return {"suite": "ui", "path": "tests/ui/sanitizer/x.rs",
                "nature": nature, "single_file": True, "ffi": False,
                "env_excluded": False}

    def view(self, text, revision=None):
        return header(text).view(revision)

    def test_run_fail_or_crash_with_sanitizer_is_classified(self):
        scope = run.sanitizer_run_fail_or_crash_scope(self.view(
            "//@ needs-sanitizer-address\n"
            "//@ compile-flags: -Z sanitizer=address -O -g\n"
            "//@ run-fail-or-crash\n"
            "//@ error-pattern: AddressSanitizer: stack-buffer-overflow\n"
            "fn main() {}\n"))
        self.assertIsNotNone(scope)
        reason, bucket = scope
        self.assertEqual(bucket,
                         "not attempted (sanitizer instrumentation)")
        self.assertIn("-Z sanitizer=address", reason)
        self.assertIn("run-fail-or-crash", reason)

    def test_joined_z_spelling_is_classified(self):
        scope = run.sanitizer_run_fail_or_crash_scope(self.view(
            "//@ compile-flags: -Zsanitizer=address -Clto=thin\n"
            "//@ run-fail-or-crash\nfn main() {}\n"))
        self.assertIsNotNone(scope)
        self.assertIn("-Z sanitizer=address", scope[0])

    def test_revision_scoped_sanitizer_flags(self):
        text = ("//@ revisions: cfi kcfi\n"
                "//@ [cfi] compile-flags: -Z sanitizer=cfi\n"
                "//@ [kcfi] compile-flags: -Z sanitizer=kcfi\n"
                "//@ run-fail-or-crash\nfn main() {}\n")
        cfi = run.sanitizer_run_fail_or_crash_scope(
            self.view(text, "cfi"))
        kcfi = run.sanitizer_run_fail_or_crash_scope(
            self.view(text, "kcfi"))
        self.assertIsNotNone(cfi)
        self.assertIsNotNone(kcfi)
        self.assertIn("-Z sanitizer=cfi", cfi[0])
        self.assertIn("-Z sanitizer=kcfi", kcfi[0])

    def test_run_fail_or_crash_without_sanitizer_stays(self):
        # The expected failure is the program's own; an
        # execution could in principle judge it, so the view
        # keeps the generic verdict-needs-execution bucket.
        self.assertIsNone(run.sanitizer_run_fail_or_crash_scope(
            self.view("//@ compile-flags: -O\n"
                      "//@ run-fail-or-crash\nfn main() {}\n")))

    def test_sanitizer_without_run_fail_or_crash_stays(self):
        self.assertIsNone(run.sanitizer_run_fail_or_crash_scope(
            self.view("//@ compile-flags: -Z sanitizer=address\n"
                      "//@ check-pass\nfn main() {}\n")))

    def test_run_fail_mode_is_the_exec_scopes_territory(self):
        self.assertIsNone(run.sanitizer_run_fail_or_crash_scope(
            self.view("//@ compile-flags: -Z sanitizer=leak\n"
                      "//@ run-fail\nfn main() {}\n")))

    def test_exec_sanitizer_scope_unchanged_by_the_helper(self):
        scope = run.sanitizer_scope(self.rec(), self.view(
            "//@ compile-flags: -Z sanitizer=leak\n"
            "//@ run-fail\nfn main() {}\n"))
        self.assertIsNotNone(scope)
        self.assertEqual(scope[1],
                         "not attempted (sanitizer instrumentation)")
        self.assertIn("-Z sanitizer=leak", scope[0])
        # A run-pass sanitizer view stays attempted, and the
        # exec scope never fires for a compile-check record.
        self.assertIsNone(run.sanitizer_scope(self.rec(), self.view(
            "//@ compile-flags: -Z sanitizer=cfi\n"
            "//@ run-pass\nfn main() {}\n")))
        self.assertIsNone(run.sanitizer_scope(
            self.rec(nature="compile-check"), self.view(
                "//@ compile-flags: -Z sanitizer=leak\n"
                "//@ run-fail\nfn main() {}\n")))


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

    def test_incremental_revision_kinds(self):
        # compiletest's IncrRevKind: the revision name's prefix
        # fixes the compile outcome (cpass/bpass/rpass succeed,
        # bfail fails).
        cases = [("rpass1", "success"), ("rpass2", "success"),
                 ("bpass1", "success"), ("cpass2", "success"),
                 ("bfail1", "failure"), ("bfail2", "failure")]
        for rev, want in cases:
            h = header(f"//@ revisions: {rev}\n")
            self.assertEqual(
                run.compile_expectation("incremental", h.view(rev)),
                want, rev)

    def test_incremental_unknown_revision_kind_not_judgable(self):
        h = header("//@ revisions: xpass1\n")
        self.assertIsNone(
            run.compile_expectation("incremental", h.view("xpass1")))

    def test_incremental_without_revision_not_judgable(self):
        h = header("fn main() {}\n")
        self.assertIsNone(
            run.compile_expectation("incremental", h.view(None)))


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

    def test_add_minicore_no_longer_blocks(self):
        # add-minicore tests are attempted with a prebuilt minicore
        # rlib (see MinicoreTests); add-core-stubs still blocks.
        h = header("//@ add-minicore\n")
        self.assertIsNone(run.stage2_blocker(h.view(None), self.rec()))
        h = header("//@ add-core-stubs\n")
        self.assertIn("core stub", run.stage2_blocker(h.view(None),
                                                      self.rec()))

    def test_rustc_env_no_longer_blocks(self):
        # rustc-env pairs are applied to the driver process
        # environment (see EnvDirectiveTests); they block neither
        # stage.
        h = header("//@ rustc-env: FOO=1\n")
        self.assertIsNone(run.stage2_blocker(h.view(None),
                                             self.rec()))

    def test_scoped_blocker_only_blocks_its_revision(self):
        h = header("//@ revisions: a b\n//@[a] add-core-stubs\n")
        self.assertIn("core stub", run.stage2_blocker(h.view("a"),
                                                      self.rec()))
        self.assertIsNone(run.stage2_blocker(h.view("b"), self.rec()))
        # A scoped add-minicore applies its rlib to that revision
        # only, via minicore_test_args; it blocks neither revision.
        h = header("//@ revisions: a b\n//@[a] add-minicore\n")
        self.assertIsNone(run.stage2_blocker(h.view("a"), self.rec()))
        self.assertIsNone(run.stage2_blocker(h.view("b"), self.rec()))

    def test_revisions_no_longer_block(self):
        # Revisions are judged per revision by fill_result_stage2.
        h = header("//@ revisions: a b\n//@[a] check-pass\n")
        self.assertIsNone(run.stage2_blocker(h.view("a"), self.rec()))
        self.assertIsNone(run.stage2_blocker(h.view("b"), self.rec()))

    def test_add_minicore_with_run_pass_blocks(self):
        # compiletest panics on the combination when parsing the
        # directives (`add-minicore` cannot be used to run the
        # test binary), so the file is never compiled; the one
        # suite file with both is a should-fail self-test of the
        # rejection.
        h = header("//@ add-minicore\n//@ run-pass\n//@ should-fail\n")
        blocker = run.stage2_blocker(h.view(None), self.rec())
        self.assertIn("cannot be used to run the test binary", blocker)
        self.assertEqual(run.header_bucket(blocker),
                         "not attempted (header feature)")

    def test_add_minicore_with_other_modes_does_not_block(self):
        # compiletest's rejection is specific to run-pass:
        # add-minicore with a compile-only mode or with
        # run-rustfix is attempted as usual.
        h = header("//@ add-minicore\n//@ check-pass\n")
        self.assertIsNone(run.stage2_blocker(h.view(None), self.rec()))
        h = header("//@ add-minicore\n//@ run-rustfix\n")
        self.assertIsNone(run.stage2_blocker(h.view(None), self.rec()))


def conv(**kw):
    base = {"exit_code": 0, "timeout": False, "converted": 3,
            "skipped": 0, "skip_totals": [], "compilation_failed": False,
            "ice": False, "stderr": "", "stdout": ""}
    base.update(kw)
    return base


class MinicoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile")
                                         .TemporaryDirectory()))
        (self.tmp / "tests" / "auxiliary").mkdir(parents=True)
        (self.tmp / "tests" / "auxiliary" / "minicore.rs").write_text(
            "#![no_core]\n")
        self.cache = self.tmp / "cache"
        os = __import__("os")
        self._old_env = os.environ.get("SPARGASMOS_MINICORE_CACHE")
        os.environ["SPARGASMOS_MINICORE_CACHE"] = str(self.cache)
        self.addCleanup(self._restore_env)
        run._MINICORE_RESULTS.clear()
        self.addCleanup(run._MINICORE_RESULTS.clear)

    def _restore_env(self):
        os = __import__("os")
        if self._old_env is None:
            os.environ.pop("SPARGASMOS_MINICORE_CACHE", None)
        else:
            os.environ["SPARGASMOS_MINICORE_CACHE"] = self._old_env

    def stub_rustc(self, ok=True):
        calls = []

        def fake_run_cmd(argv, **kw):
            calls.append(argv)
            if ok:
                out = Path(argv[argv.index("-o") + 1])
                out.write_bytes(b"!<arch>\n" + b"fake rlib")
                return {"exit_code": 0, "stdout": "", "stderr": "",
                        "timeout": False}
            return {"exit_code": 1, "stdout": "",
                    "stderr": "error: boom\n", "timeout": False}

        orig = run.run_cmd
        run.run_cmd = fake_run_cmd
        self.addCleanup(setattr, run, "run_cmd", orig)
        return calls

    def test_build_then_cache_hit(self):
        calls = self.stub_rustc()
        rlib, err = run.minicore_rlib(self.tmp, "2021", [], [])
        self.assertIsNone(err)
        self.assertTrue(run._is_rlib(rlib))
        self.assertEqual(len(calls), 1)
        argv = calls[0]
        self.assertIn("--crate-type", argv)
        self.assertIn("rlib", argv)
        self.assertIn("-Cpanic=abort", argv)
        # A second call for the same variant builds nothing, in this
        # process or (via the on-disk cache) a fresh one.
        rlib2, err2 = run.minicore_rlib(self.tmp, "2021", [], [])
        self.assertEqual((rlib2, err2), (rlib, None))
        run._MINICORE_RESULTS.clear()
        rlib3, err3 = run.minicore_rlib(self.tmp, "2021", [], [])
        self.assertEqual((rlib3, err3), (rlib, None))
        self.assertEqual(len(calls), 1)

    def test_flag_variant_builds_separately(self):
        calls = self.stub_rustc()
        a, _ = run.minicore_rlib(self.tmp, "2021", [], [])
        b, _ = run.minicore_rlib(self.tmp, "2021", ["-Copt-level=3"], [])
        self.assertNotEqual(a, b)
        self.assertEqual(len(calls), 2)

    def test_build_failure_is_a_reason_not_a_path(self):
        self.stub_rustc(ok=False)
        rlib, err = run.minicore_rlib(self.tmp, "2021", [], [])
        self.assertIsNone(rlib)
        self.assertIn("minicore build failed", err)
        self.assertIn("error: boom", err)

    def test_missing_source_is_a_reason(self):
        rlib, err = run.minicore_rlib(self.tmp / "nope", "2021", [], [])
        self.assertIsNone(rlib)
        self.assertIn("not found", err)

    def test_test_args_shape(self):
        orig = run.minicore_rlib
        run.minicore_rlib = lambda *a: (Path("/fake/libminicore.rlib"),
                                        None)
        self.addCleanup(setattr, run, "minicore_rlib", orig)
        h = header("//@ add-minicore\n")
        args, libs, err = run.minicore_test_args(
            h.view(None), self.tmp, "2021", [])
        self.assertIsNone(err)
        self.assertEqual(args, ["-Cpanic=abort",
                                "-Cforce-unwind-tables=yes",
                                "--extern",
                                "minicore=/fake/libminicore.rlib"])
        # The rlib also travels as a prebuilt auxiliary (not
        # collected), so an auxiliary build that declares
        # minicore binds it during its own build too.
        self.assertEqual(libs, [("minicore",
                                 Path("/fake/libminicore.rlib"), False)])

    def test_test_args_edition_in_minicore_flags_rejected(self):
        h = header("//@ add-minicore\n"
                   "//@ minicore-compile-flags: --edition 2021\n")
        args, libs, err = run.minicore_test_args(
            h.view(None), self.tmp, "2021", [])
        self.assertIsNone(args)
        self.assertIsNone(libs)
        self.assertIn("--edition", err)

    def test_test_args_build_failure_propagates(self):
        orig = run.minicore_rlib
        run.minicore_rlib = lambda *a: (None, "minicore build failed: x")
        self.addCleanup(setattr, run, "minicore_rlib", orig)
        h = header("//@ add-minicore\n")
        args, libs, err = run.minicore_test_args(
            h.view(None), self.tmp, "2021", [])
        self.assertIsNone(args)
        self.assertIsNone(libs)
        self.assertIn("minicore build failed", err)


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

    def test_failure_declared_crash_status_passes(self):
        status, _r, _b = run.judge_compile(
            "failure", conv(exit_code=101, ice=True, converted=None,
                            skipped=None),
            declared_status=101)
        self.assertEqual(status, "pass")

    def test_failure_declared_status_needs_ice_evidence(self):
        status, _r, bucket = run.judge_compile(
            "failure", conv(exit_code=101, converted=None,
                            skipped=None),
            declared_status=101)
        self.assertEqual(status, "fail")
        self.assertIn("crash", bucket)

    def test_failure_declared_status_mismatch_is_fail(self):
        status, _r, _b = run.judge_compile(
            "failure", conv(exit_code=101, ice=True, converted=None,
                            skipped=None),
            declared_status=102)
        self.assertEqual(status, "fail")

    def test_failure_declared_status_opt_out_is_fail(self):
        status, _r, _b = run.judge_compile(
            "failure", conv(exit_code=101, ice=True, converted=None,
                            skipped=None),
            declared_status=101, dont_check_status=True)
        self.assertEqual(status, "fail")

    def test_failure_clean_rejection_passes_with_declared_status(self):
        # The declared status adds the crash ground; it does not
        # revoke the clean-rejection ground (a test declaring
        # failure-status: 101 whose error the driver reports as
        # a clean rejection still passes, as before).
        status, _r, _b = run.judge_compile(
            "failure", conv(exit_code=1, compilation_failed=True,
                            converted=None, skipped=None),
            declared_status=101)
        self.assertEqual(status, "pass")

    def test_crash_reproduced_passes(self):
        status, _r, _b = run.judge_compile(
            "crash", conv(exit_code=101, ice=True, converted=None,
                          skipped=None))
        self.assertEqual(status, "pass")

    def test_crash_unevidenced_101_fails(self):
        # The driver also exits 101 when one of its own panics
        # escapes the compiler run (a Stable MIR conversion
        # panic, the driver thread panic): that abnormal exit
        # is not a reproduction of the reference compiler
        # crash, so without an internal compiler error in the
        # output the crash expectation is not met.
        status, reason, bucket = run.judge_compile(
            "crash", conv(exit_code=101, converted=None,
                          skipped=None))
        self.assertEqual(status, "fail")
        self.assertEqual(bucket, "compiler crash not evidenced")
        self.assertIn("internal compiler error", reason)

    def test_crash_signal_death_passes(self):
        # Death by signal is the compiler process crashing,
        # which is compiletest's crashes condition itself; it
        # needs no further evidence.
        status, _r, _b = run.judge_compile(
            "crash", conv(exit_code=-11, converted=None,
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

    def test_failure_conversion_panic_is_unsupported_not_driver_error(self):
        # Lane CI: a Stable MIR conversion panic on a
        # failure-expecting test is a named conversion limit.
        status, reason, bucket = run.judge_compile(
            "failure", conv(
                exit_code=1, converted=None, skipped=None,
                conversion_panicked=True,
                panic_reason="not implemented: FIXME(async_closures)",
                stderr="sparagmos-driver: Stable MIR conversion "
                       "panicked: not implemented"))
        self.assertEqual(status, "unsupported")
        self.assertIn("Stable MIR conversion panicked", bucket)
        self.assertIn("async_closures", reason)

    def test_success_conversion_panic_is_unsupported(self):
        status, _r, bucket = run.judge_compile(
            "success", conv(
                exit_code=1, converted=None, skipped=None,
                conversion_panicked=True,
                panic_reason="internal error: entered unreachable code",
                stderr="sparagmos-driver: Stable MIR conversion "
                       "panicked: internal error"))
        self.assertEqual(status, "unsupported")
        self.assertIn("Stable MIR conversion panicked", bucket)

    def test_panic_after_genuine_failure_stays_fail(self):
        # A panic raised while reporting a real compilation
        # failure (rustc's delayed-bug payloads) must not turn a
        # success-expecting failure into "unsupported".
        status, _r, bucket = run.judge_compile(
            "success", conv(
                exit_code=1, converted=None, skipped=None,
                compilation_failed=True, conversion_panicked=True,
                panic_reason="panic with a non-string payload",
                stderr="error: internal compiler error\n"
                       "sparagmos-driver: compilation failed"))
        self.assertEqual(status, "fail")
        self.assertIn("compilation failed", bucket)

    def test_aux_failure_is_named_unsupported(self):
        status, reason, bucket = run.judge_compile(
            "failure", conv(
                exit_code=1, converted=None, skipped=None,
                aux_failure="failed to compile: error[E0463]: "
                            "can't find crate for `minicore`"))
        self.assertEqual(status, "unsupported")
        self.assertEqual(bucket, "auxiliary crate build failure")
        self.assertIn("minicore", reason)


class RunCmdMemoryLimitTests(unittest.TestCase):
    """run_cmd's per-record memory bound (Issue #483): the
    watchdog kills a driver process tree whose RSS crosses the
    bound and reports mem_exceeded, so one ballooning record
    fails alone instead of taking the whole shard job (and its
    results) down with it. The children here are small Python
    balloons, not the driver."""

    BALLOON = ("x = bytearray(400 * 1024 * 1024)\n"
               "for i in range(0, len(x), 4096):\n"
               "    x[i] = 1\n"
               "import time\n"
               "time.sleep(60)\n")

    def test_unbounded_run_reports_no_mem_exceeded(self):
        res = run.run_cmd([sys.executable, "-c", "print('hi')"],
                          timeout=30)
        self.assertEqual(res["exit_code"], 0)
        self.assertEqual(res["stdout"], "hi\n")
        self.assertFalse(res["timeout"])
        self.assertFalse(res["mem_exceeded"])

    def test_bounded_run_inside_the_limit_is_unchanged(self):
        res = run.run_cmd([sys.executable, "-c", "print('hi')"],
                          timeout=30, mem_limit_mb=512)
        self.assertEqual(res["exit_code"], 0)
        self.assertEqual(res["stdout"], "hi\n")
        self.assertFalse(res["timeout"])
        self.assertFalse(res["mem_exceeded"])

    def test_ballooning_child_is_killed_by_the_bound(self):
        start = time.monotonic()
        res = run.run_cmd([sys.executable, "-c", self.BALLOON],
                          timeout=60, mem_limit_mb=96)
        self.assertLess(time.monotonic() - start, 30)
        self.assertTrue(res["mem_exceeded"])
        self.assertFalse(res["timeout"])
        self.assertEqual(res["exit_code"], -9)

    def test_ballooning_grandchild_is_killed_via_the_tree(self):
        code = ("import subprocess, sys, time\n"
                f"subprocess.Popen([sys.executable, '-c', "
                f"{self.BALLOON!r}])\n"
                "time.sleep(60)\n")
        start = time.monotonic()
        res = run.run_cmd([sys.executable, "-c", code],
                          timeout=60, mem_limit_mb=96)
        self.assertLess(time.monotonic() - start, 30)
        self.assertTrue(res["mem_exceeded"])
        self.assertFalse(res["timeout"])

    def test_pipe_holding_descendant_does_not_outlive_the_timeout(self):
        # The root exits at once; its child keeps the output pipes
        # and sleeps on. The bounded path must still return at the
        # record timeout: the child sits in the root's process
        # group, so the deadline kill reaches it and the post-kill
        # drain cannot wait out its sleep.
        code = ("import subprocess, sys\n"
                "subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(6)'])\n")
        start = time.monotonic()
        res = run.run_cmd([sys.executable, "-c", code],
                          timeout=1, mem_limit_mb=512)
        self.assertLess(time.monotonic() - start, 4)
        self.assertTrue(res["timeout"])
        self.assertFalse(res["mem_exceeded"])

    def test_ballooning_descendant_after_root_exit_is_killed(self):
        # The root exits at once; its child balloons. Once the root
        # is reaped the child drops out of the ppid tree, but it
        # keeps the root's process group, so the watchdog still
        # sees its RSS and the bound still kills it.
        code = ("import subprocess, sys\n"
                f"subprocess.Popen([sys.executable, '-c', "
                f"{self.BALLOON!r}])\n")
        start = time.monotonic()
        res = run.run_cmd([sys.executable, "-c", code],
                          timeout=30, mem_limit_mb=96)
        self.assertLess(time.monotonic() - start, 20)
        self.assertTrue(res["mem_exceeded"])
        self.assertFalse(res["timeout"])

    def test_own_session_descendant_is_killed_by_the_bound(self):
        # The child makes its own session, so it sits outside the
        # root's process group. While the root lives the watchdog
        # still sees it in the ppid tree, but the moment the group
        # kill lands, the root dies, the child is reparented, and a
        # fresh /proc scan can no longer find it. The kill must
        # spend the pids the polls remembered, not only what a
        # post-kill scan can still see.
        child = ("x = bytearray(40 * 1024 * 1024)\n"
                 "for i in range(0, len(x), 4096):\n"
                 "    x[i] = 1\n"
                 "import time\n"
                 "time.sleep(60)\n")
        code = ("import subprocess, sys, time\n"
                f"child = subprocess.Popen([sys.executable, '-c', "
                f"{child!r}], start_new_session=True, "
                "stdout=subprocess.DEVNULL, "
                "stderr=subprocess.DEVNULL)\n"
                "print(child.pid, flush=True)\n"
                "time.sleep(60)\n")
        start = time.monotonic()
        res = run.run_cmd([sys.executable, "-c", code],
                          timeout=60, mem_limit_mb=24)
        self.assertLess(time.monotonic() - start, 30)
        self.assertTrue(res["mem_exceeded"])
        self.assertFalse(res["timeout"])
        child_pid = int(res["stdout"].strip())
        deadline = time.monotonic() + 5
        while (not _process_terminated(child_pid)
               and time.monotonic() < deadline):
            time.sleep(0.1)
        # Terminated, not necessarily gone from /proc: once the
        # root dies the child is reparented to PID 1, and an init
        # that does not reap orphans leaves the killed child as a
        # zombie indefinitely (Issue #486).
        self.assertTrue(_process_terminated(child_pid))

    def test_process_terminated_covers_zombies(self):
        # A killed-but-unreaped child is a zombie: still in /proc,
        # but terminated. A live child is not terminated, and a
        # reaped (or never-existent) pid counts as terminated
        # because its /proc entry is gone (Issue #486).
        import subprocess
        live = subprocess.Popen([sys.executable, "-c",
                                 "import time; time.sleep(60)"])
        try:
            self.assertFalse(_process_terminated(live.pid))
            live.kill()
            deadline = time.monotonic() + 5
            while (not _process_terminated(live.pid)
                   and time.monotonic() < deadline):
                time.sleep(0.05)
            self.assertTrue(_process_terminated(live.pid))
            # The child is our own and still unreaped here, so the
            # True above can only have come from the zombie state.
            self.assertTrue(Path(f"/proc/{live.pid}").exists())
        finally:
            if live.poll() is None:
                live.kill()
            live.wait()
        self.assertTrue(_process_terminated(live.pid))
        self.assertTrue(_process_terminated(999_999_999))

    def test_timeout_still_reported_with_the_bound_set(self):
        res = run.run_cmd(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            timeout=1, mem_limit_mb=512)
        self.assertTrue(res["timeout"])
        self.assertFalse(res["mem_exceeded"])
        self.assertIsNone(res["exit_code"])

    def test_memory_limit_reason_buckets_as_memory_limit(self):
        self.assertEqual(
            run.bucket_reason("driver exceeded the per-record "
                              "memory limit of 5120 MB"),
            "memory limit")

    def test_judge_compile_memory_limit(self):
        conv = {"mem_exceeded": True, "timeout": False,
                "exit_code": -9}
        status, _reason, bucket = run.judge_compile("success", conv)
        self.assertEqual(status, "fail")
        self.assertEqual(bucket, "memory limit")

    def test_judge_jit_compile_memory_limit(self):
        conv = {"mem_exceeded": True, "timeout": False,
                "exit_code": -9}
        status, _reason, bucket = run.judge_jit_compile("success",
                                                        conv)
        self.assertEqual(status, "fail")
        self.assertEqual(bucket, "memory limit")


class WindowsProcessLayerTests(unittest.TestCase):
    """The runner process layer's Windows branches (Issue #517).

    Each branch is selected from sys.platform at call time;
    these tests patch it to "win32" and stub the process side,
    so they run on the POSIX hosts CI uses. The POSIX forms are
    pinned alongside: selecting the Windows branch must not
    move them.
    """

    def test_kill_process_tree_uses_taskkill_on_windows(self):
        calls = []

        def fake_run(argv, **kwargs):
            calls.append(argv)
            return unittest.mock.Mock(returncode=0)

        with unittest.mock.patch.object(sys, "platform", "win32"), \
                unittest.mock.patch.object(
                    run.subprocess, "run", fake_run), \
                unittest.mock.patch.object(
                    run.os, "killpg",
                    side_effect=AssertionError("killpg used")), \
                unittest.mock.patch.object(
                    run.os, "kill",
                    side_effect=AssertionError("kill used")):
            run._kill_process_tree(4242, pgid=4242,
                                   extra_pids=(4242, 7777))
        # The root's tree kill covers its descendants; a
        # remembered member is named individually, and the
        # root is not killed twice.
        self.assertEqual(
            calls,
            [["taskkill", "/PID", "4242", "/T", "/F"],
             ["taskkill", "/PID", "7777", "/T", "/F"]])

    def test_kill_process_tree_posix_form_unchanged(self):
        killed = []
        with unittest.mock.patch.object(sys, "platform", "linux"), \
                unittest.mock.patch.object(
                    run, "_proc_tree", lambda pid: ([4321], 0)), \
                unittest.mock.patch.object(
                    run.os, "killpg",
                    lambda pgid, sig: killed.append(("pg", pgid, sig))), \
                unittest.mock.patch.object(
                    run.os, "kill",
                    lambda pid, sig: killed.append(("pid", pid, sig))):
            run._kill_process_tree(4321, pgid=4321,
                                   extra_pids=(9999,))
        self.assertEqual(
            killed,
            [("pg", 4321, run.signal.SIGKILL),
             ("pid", 9999, run.signal.SIGKILL),
             ("pid", 4321, run.signal.SIGKILL)])

    def test_lib_path_entry_windows_uses_path_and_semicolons(self):
        with unittest.mock.patch.object(sys, "platform", "win32"), \
                unittest.mock.patch.dict(
                    os.environ, {"PATH": "C:\\Windows"}, clear=True):
            name, value = run._lib_path_entry(["C:\\a", "C:\\b"])
        self.assertEqual(name, "PATH")
        self.assertEqual(value, "C:\\a;C:\\b;C:\\Windows")

    def test_lib_path_entry_posix_form_unchanged(self):
        with unittest.mock.patch.object(sys, "platform", "linux"), \
                unittest.mock.patch.dict(
                    os.environ, {"LD_LIBRARY_PATH": "/prev"},
                    clear=True):
            self.assertEqual(
                run._lib_path_entry(["/a", "/b"]),
                ("LD_LIBRARY_PATH", "/a:/b:/prev"))
        with unittest.mock.patch.object(sys, "platform", "linux"), \
                unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(run._lib_path_entry(["/a"]),
                             ("LD_LIBRARY_PATH", "/a"))

    def test_find_artifact_windows_bare_names(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "foo.dll").touch()
            (Path(td) / "bar.lib").touch()
            (Path(td) / "libbaz.rlib").touch()
            (Path(td) / "baz.dll").touch()
            with unittest.mock.patch.object(sys, "platform", "win32"):
                # A dylib / proc-macro artifact is `{name}.dll`
                # with no lib prefix; a staticlib is `{name}.lib`.
                self.assertEqual(run.find_artifact(td, "foo"),
                                 Path(td) / "foo.dll")
                self.assertEqual(run.find_artifact(td, "bar"),
                                 Path(td) / "bar.lib")
                # The rlib still outranks the same crate's dll.
                self.assertEqual(run.find_artifact(td, "baz"),
                                 Path(td) / "libbaz.rlib")

    def test_find_artifact_posix_form_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "foo.dll").touch()
            (Path(td) / "libqux.so").touch()
            with unittest.mock.patch.object(sys, "platform", "linux"):
                # The bare-name glob is Windows-only: a prefixless
                # dll is not an artifact name on POSIX.
                self.assertIsNone(run.find_artifact(td, "foo"))
                self.assertEqual(run.find_artifact(td, "qux"),
                                 Path(td) / "libqux.so")

    def test_support_artifact_suffixes(self):
        with unittest.mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(run._support_artifact_suffixes(),
                             (".rlib", ".so", ".dylib"))
        with unittest.mock.patch.object(sys, "platform", "win32"):
            suffixes = run._support_artifact_suffixes()
        self.assertIn(".dll", suffixes)
        self.assertIn(".lib", suffixes)
        self.assertIn(".rlib", suffixes)

    def test_stage3_recipe_env_platform_forms(self):
        paths = {"rustc": "/tc/rustc", "rustdoc": "/tc/rustdoc",
                 "sysroot_lib": "/tc/lib",
                 "target_lib": "/tc/target/lib"}
        with unittest.mock.patch.object(
                run, "stage3_toolchain_paths", lambda: paths):
            with unittest.mock.patch.object(sys, "platform", "linux"):
                env = run.stage3_recipe_env(
                    Path("/rust"), Path("/build"), "run-make")
            self.assertEqual(env["LD_LIB_PATH_ENVVAR"],
                             "LD_LIBRARY_PATH")
            self.assertEqual(env["LD_LIBRARY_PATH"],
                             "/tc/lib:/tc/target/lib")
            self.assertNotIn("PATH", env)
            with unittest.mock.patch.object(sys, "platform", "win32"), \
                    unittest.mock.patch.dict(
                        os.environ, {"PATH": "C:\\Windows"},
                        clear=True):
                env = run.stage3_recipe_env(
                    Path("/rust"), Path("/build"), "run-make")
            self.assertEqual(env["LD_LIB_PATH_ENVVAR"], "PATH")
            self.assertEqual(env["PATH"],
                             "/tc/lib;/tc/target/lib;C:\\Windows")
            self.assertNotIn("LD_LIBRARY_PATH", env)


class ConcreteEditionTests(unittest.TestCase):
    def test_closed_range_takes_lower_bound(self):
        # compiletest's range is half-open and, with no requested
        # edition configured, edition_to_test falls to the lower
        # bound -- never the excluded upper end.
        self.assertEqual(run._concrete_edition("2015..2021"), "2015")

    def test_open_range_takes_first(self):
        self.assertEqual(run._concrete_edition("2018.."), "2018")

    def test_plain_edition_unchanged(self):
        self.assertEqual(run._concrete_edition("2024"), "2024")


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

    def run_with_conv_capture(self, rec, conv_result):
        captured = {}

        def stub(path, edition, driver_bin, rustc_args=(),
                 aux_specs=(), env_extra=None, env_remove=(),
                 jit=False, aux_env_scrub=(), capture_stdout=False,
                 aux_libs=(), check_mode=False):
            captured["rustc_args"] = list(rustc_args)
            captured["aux_libs"] = list(aux_libs)
            captured["capture_stdout"] = capture_stdout
            return conv_result

        orig = run.convert_side
        run.convert_side = stub
        try:
            res = run.process_test(rec, self.rust_dir,
                                   Path("/fake/driver"), stage2=True)
        finally:
            run.convert_side = orig
        return res, captured

    def test_add_minicore_forwards_extern_and_panic_args(self):
        rec = self.write_test("//@ add-minicore\n//@ check-pass\n"
                              "fn main() {}\n")
        orig = run.minicore_rlib
        run.minicore_rlib = lambda *a: (
            Path("/fake/libminicore.rlib"), None)
        try:
            res, captured = self.run_with_conv_capture(rec, conv())
        finally:
            run.minicore_rlib = orig
        self.assertEqual(res["status"], "pass")
        args = captured["rustc_args"]
        self.assertIn("-Cpanic=abort", args)
        self.assertIn("-Cforce-unwind-tables=yes", args)
        i = args.index("--extern")
        self.assertEqual(args[i + 1], "minicore=/fake/libminicore.rlib")
        # The rlib also travels as a prebuilt auxiliary (not
        # collected), so an auxiliary build that declares
        # minicore binds it during its own build too.
        self.assertEqual(
            captured["aux_libs"],
            [("minicore", Path("/fake/libminicore.rlib"), False)])

    def test_add_minicore_build_failure_is_not_attempted(self):
        rec = self.write_test("//@ add-minicore\n//@ check-fail\n"
                              "fn main() {}\n")
        orig = run.minicore_rlib
        run.minicore_rlib = lambda *a: (None, "minicore build failed: x")
        try:
            res, captured = self.run_with_conv_capture(rec, conv())
        finally:
            run.minicore_rlib = orig
        # Never judged: a missing extern would fabricate a pass on
        # this failure-expecting test.
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"], "not attempted (minicore build)")
        self.assertNotIn("rustc_args", captured)

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

    def test_driver_record_carries_not_demanded_counts(self):
        rec = self.write_test("//@ check-pass\nfn main() {}\n")
        res = self.run_with_conv(
            rec, conv(target_converted=6, target_skipped=0,
                      not_demanded=4, target_not_demanded=2))
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["driver"]["target_converted"], 6)
        self.assertEqual(res["driver"]["target_skipped"], 0)
        self.assertEqual(res["driver"]["not_demanded"], 4)
        self.assertEqual(res["driver"]["target_not_demanded"], 2)

    def test_multi_file_not_attempted(self):
        rec = self.write_test("//@ check-pass\nfn main() {}\n")
        rec["single_file"] = False
        res = run.process_test(rec, self.rust_dir, Path("/fake/driver"),
                               stage2=True)
        self.assertEqual(res["status"], "not-attempted")
        self.assertIn("multi-file", res["bucket"])

    def test_incremental_without_revisions_not_attempted(self):
        rec = self.write_test("fn main() {}\n", suite="incremental")
        res = run.process_test(rec, self.rust_dir, Path("/fake/driver"),
                               stage2=True)
        self.assertEqual(res["status"], "not-attempted")
        self.assertIn("revisions", res["reason"])

    def run_incremental_two_revisions(self, rec):
        """Process an incremental record with a capturing
        convert_side stub; returns the per-call rustc_args."""
        calls = []

        def stub(path, edition, driver_bin, rustc_args=(),
                 aux_specs=(), env_extra=None, env_remove=(),
                 jit=False, aux_env_scrub=(), capture_stdout=False,
                 aux_libs=(), check_mode=False):
            calls.append(list(rustc_args))
            return conv()

        orig = run.convert_side
        run.convert_side = stub
        try:
            res = run.process_test(rec, self.rust_dir,
                                   Path("/fake/driver"), stage2=True)
        finally:
            run.convert_side = orig
        self.assertEqual(res["status"], "pass")
        self.assertEqual(len(calls), 2)
        return calls

    def run_incremental_stubbed(self, rec, by_rev):
        """Process an incremental record with a convert_side stub
        returning the given result per revision; returns
        (result, per-call rustc_args)."""
        calls = []

        def stub(path, edition, driver_bin, rustc_args=(),
                 aux_specs=(), env_extra=None, env_remove=(),
                 jit=False, aux_env_scrub=(), capture_stdout=False,
                 aux_libs=(), check_mode=False):
            calls.append(list(rustc_args))
            for rev, res in by_rev.items():
                if rev in rustc_args or \
                        any(a.endswith("=" + rev) for a in rustc_args):
                    return res
            raise AssertionError(f"no stub for {rustc_args}")

        orig = run.convert_side
        run.convert_side = stub
        try:
            res = run.process_test(rec, self.rust_dir,
                                   Path("/fake/driver"), stage2=True)
        finally:
            run.convert_side = orig
        return res, calls

    def test_incremental_dep_graph_assertion_judged_sequentially(self):
        # A rustc_clean assertion in the test file is checked by
        # the driver's embedded rustc when the revisions run as
        # sequential sessions over one shared incremental
        # directory, so the record is judged -- with the same
        # -Cincremental directory forwarded to every revision.
        rec = self.write_test(
            "//@ revisions: rpass1 rpass2\n"
            "#![feature(rustc_attrs)]\n"
            "#[rustc_clean(cfg=\"rpass2\")]\n"
            "pub fn f() {}\nfn main() {}\n",
            suite="incremental")
        calls = self.run_incremental_two_revisions(rec)
        incrs = [[a for a in c if a.startswith("-Cincremental=")]
                 for c in calls]
        self.assertTrue(all(len(i) == 1 for i in incrs))
        self.assertEqual(incrs[0], incrs[1])

    def test_incremental_mir_only_assertion_failure_not_attempted(self):
        # A revision whose compilation fails solely on clean
        # assertions over MIR dep nodes reports the harness's
        # demand pattern -- conversion forces MIR bodies for every
        # definition in the crate, the reference compilation only
        # for codegen-reached items -- not the test, so the record
        # is not attempted for that reason.
        rec = self.write_test(
            "//@ revisions: rpass1 rpass2\n"
            "#![feature(rustc_attrs)]\n"
            "#[rustc_clean(cfg=\"rpass2\")]\n"
            "pub fn f() {}\nfn main() {}\n",
            suite="incremental")
        failed = conv(
            exit_code=1, compilation_failed=True, converted=None,
            stderr="error: `optimized_mir(f)` should be clean but "
                   "is not\n"
                   "error: aborting due to 1 previous error\n")
        res, _calls = self.run_incremental_stubbed(
            rec, {"rpass1": conv(), "rpass2": failed})
        self.assertEqual(res["status"], "not-attempted")
        self.assertIn("dep-graph", res["bucket"])
        self.assertIn("MIR dep nodes", res["reason"])

    def test_incremental_mixed_assertion_failure_is_judged(self):
        # A failure that also involves a non-MIR dep node is a
        # genuine assertion outcome and stays judged.
        rec = self.write_test(
            "//@ revisions: rpass1 rpass2\n"
            "#![feature(rustc_attrs)]\n"
            "#[rustc_clean(cfg=\"rpass2\")]\n"
            "pub fn f() {}\nfn main() {}\n",
            suite="incremental")
        failed = conv(
            exit_code=1, compilation_failed=True, converted=None,
            stderr="error: `optimized_mir(f)` should be clean but "
                   "is not\n"
                   "error: `hir_owner(f)` should be clean but is "
                   "not\n"
                   "error: aborting due to 2 previous errors\n")
        res, _calls = self.run_incremental_stubbed(
            rec, {"rpass1": conv(), "rpass2": failed})
        self.assertEqual(res["status"], "fail")

    def test_incremental_sequence_shares_lint_levels(self):
        # The whole sequence runs with one lint level set: with
        # per-revision levels (the adjustment pair only on
        # success-expecting revisions) the second session does not
        # load the first session's dep graph at all -- measured on
        # tests/incremental/clean.rs -- so a failure-expecting
        # revision in a sequence whose other revisions expect
        # success receives the pair too, and its expected failure
        # (here an ordinary compilation error) is judged.
        rec = self.write_test(
            "//@ revisions: rpass1 bfail2\n"
            "#![feature(rustc_attrs)]\n"
            "#[rustc_clean(cfg=\"bfail2\")]\n"
            "pub fn f() {}\nfn main() {}\n",
            suite="incremental")
        failed = conv(exit_code=1, compilation_failed=True,
                      converted=None,
                      stderr="error[E0308]: mismatched types\n")
        res, calls = self.run_incremental_stubbed(
            rec, {"rpass1": conv(), "bfail2": failed})
        self.assertEqual(res["status"], "pass")
        self.assertEqual(len(calls), 2)
        for args in calls:
            self.assertIn("unconditional_panic", args)
            self.assertIn("arithmetic_overflow", args)

    def test_incremental_commented_out_assertion_is_judged(self):
        # A commented-out annotation is not an assertion rustc
        # ever sees: it neither blocks the record (not even a
        # partition assertion) nor makes its sessions sequential.
        rec = self.write_test(
            "//@ revisions: rpass1 rpass2\n"
            "#![feature(rustc_attrs)]\n"
            "//? #[rustc_clean(cfg=\"rpass2\")]\n"
            "/* #[rustc_partition_reused(module=\"t\", cfg=\"rpass2\")] */\n"
            "pub fn f() {}\nfn main() {}\n",
            suite="incremental")
        calls = self.run_incremental_two_revisions(rec)
        for c in calls:
            self.assertFalse(any(a.startswith("-Cincremental=")
                                 for a in c))

    def test_incremental_partition_assertion_not_attempted(self):
        rec = self.write_test(
            "//@ revisions: rpass1 rpass2\n"
            "#![feature(rustc_attrs)]\n"
            "#![rustc_partition_reused(module=\"t\", cfg=\"rpass2\")]\n"
            "fn main() {}\n",
            suite="incremental")
        res = run.process_test(rec, self.rust_dir, Path("/fake/driver"),
                               stage2=True)
        self.assertEqual(res["status"], "not-attempted")
        self.assertIn("dep-graph", res["bucket"])
        self.assertIn("rustc_partition_reused", res["reason"])

    def test_incremental_aux_dep_graph_assertion_not_attempted(self):
        rec = self.write_test(
            "//@ revisions: rpass1 rpass2\n"
            "//@ aux-build: a.rs\n"
            "extern crate a;\nfn main() {}\n",
            suite="incremental")
        rec["single_file"] = False
        aux_dir = self.rust_dir / "tests" / "ui" / "auxiliary"
        aux_dir.mkdir()
        (aux_dir / "a.rs").write_text(
            "#![feature(rustc_attrs)]\n"
            "#[rustc_clean(cfg=\"rpass2\")]\n"
            "pub fn g() {}\n")
        res = run.process_test(rec, self.rust_dir, Path("/fake/driver"),
                               stage2=True)
        self.assertEqual(res["status"], "not-attempted")
        self.assertIn("dep-graph", res["bucket"])
        self.assertIn("a.rs", res["reason"])

    def test_incremental_aux_partition_assertion_not_attempted(self):
        # A partition assertion inside an auxiliary crate is
        # checked against that crate's own code generation work
        # products, which no session of this harness produces:
        # the record is not attempted, with the auxiliary file
        # named in the reason.
        rec = self.write_test(
            "//@ revisions: rpass1 rpass2\n"
            "//@ aux-build: a.rs\n"
            "extern crate a;\nfn main() {}\n",
            suite="incremental")
        rec["single_file"] = False
        aux_dir = self.rust_dir / "tests" / "ui" / "auxiliary"
        aux_dir.mkdir()
        (aux_dir / "a.rs").write_text(
            "#![feature(rustc_attrs)]\n"
            "#![rustc_partition_reused(module=\"a\", cfg=\"rpass2\")]\n"
            "pub fn g() {}\n")
        res = run.process_test(rec, self.rust_dir, Path("/fake/driver"),
                               stage2=True)
        self.assertEqual(res["status"], "not-attempted")
        self.assertIn("dep-graph", res["bucket"])
        self.assertIn("a.rs", res["reason"])
        self.assertIn("rustc_partition_reused", res["reason"])

    def test_incremental_partition_codegened_only_not_attempted(self):
        # A record asserting only that a module was codegened
        # stays out of scope too: a conversion session reuses
        # nothing, so the assertion would hold vacuously. The
        # verdict must come from real code generation work
        # products, not from their absence.
        rec = self.write_test(
            "//@ revisions: rpass1 rpass2\n"
            "#![feature(rustc_attrs)]\n"
            "#![rustc_partition_codegened(module=\"t\", cfg=\"rpass2\")]\n"
            "fn main() {}\n",
            suite="incremental")
        res = run.process_test(rec, self.rust_dir, Path("/fake/driver"),
                               stage2=True)
        self.assertEqual(res["status"], "not-attempted")
        self.assertIn("dep-graph", res["bucket"])
        self.assertIn("rustc_partition_codegened", res["reason"])

    def test_incremental_assert_incr_state_loaded_judged(self):
        # Sequential sessions share one incremental directory, so
        # the second session starts from the first session's cache
        # and -Zassert-incr-state=loaded holds by construction.
        rec = self.write_test(
            "//@ revisions: bpass1 bpass2\n"
            "//@ [bpass2] compile-flags: -Zassert-incr-state=loaded\n"
            "fn main() {}\n",
            suite="incremental")
        calls = self.run_incremental_two_revisions(rec)
        incrs = [[a for a in c if a.startswith("-Cincremental=")]
                 for c in calls]
        self.assertTrue(all(len(i) == 1 for i in incrs))
        self.assertEqual(incrs[0], incrs[1])

    def test_incremental_assert_incr_state_not_loaded_is_judged(self):
        # =not-loaded asserts a fresh start, which an independent
        # conversion satisfies by construction.
        rec = self.write_test(
            "//@ revisions: bpass1 bpass2\n"
            "//@ [bpass1] compile-flags: -Zassert-incr-state=not-loaded\n"
            "//@ [bpass2] compile-flags: -Zassert-incr-state=not-loaded\n"
            "fn main() {}\n",
            suite="incremental")
        res = self.run_with_conv(rec, conv())
        self.assertEqual(res["status"], "pass")

    def test_incremental_revisions_judged_per_revision(self):
        rec = self.write_test(
            "//@ revisions: rpass1 bfail2\nfn main() {}\n",
            suite="incremental")

        def stub(path, edition, driver_bin, rustc_args=(),
                 aux_specs=(), env_extra=None, env_remove=(),
                 jit=False, aux_env_scrub=(), capture_stdout=False,
                 aux_libs=(), check_mode=False):
            if "bfail2" in rustc_args:
                return conv(exit_code=1, compilation_failed=True,
                            converted=None, skipped=None)
            return conv()

        orig = run.convert_side
        run.convert_side = stub
        try:
            res = run.process_test(rec, self.rust_dir,
                                   Path("/fake/driver"), stage2=True)
        finally:
            run.convert_side = orig
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["revisions"]["rpass1"]["expectation"],
                         "success")
        self.assertEqual(res["revisions"]["bfail2"]["expectation"],
                         "failure")

    def test_compile_check_without_stage2_stays_out_of_scope(self):
        rec = self.write_test("//@ check-fail\nfn main() {}\n")
        res = run.process_test(rec, self.rust_dir, Path("/fake/driver"))
        self.assertEqual(res["status"], "not-attempted")
        self.assertIn("nature", res["bucket"])

    def test_mir_opt_forwards_rlib_crate_type(self):
        # compiletest compiles mir-opt tests as rlib crates; the
        # runner forwards the crate type it prescribes.
        rec = self.write_test("pub fn f() {}\n", suite="mir-opt")
        res, captured = self.run_with_conv_capture(rec, conv())
        self.assertEqual(res["status"], "pass")
        args = captured["rustc_args"]
        i = args.index("--crate-type")
        self.assertEqual(args[i + 1], "rlib")

    def test_ui_forwards_no_crate_type(self):
        # compiletest names no crate type for a UI test file: the
        # driver's convert mode must be left to the source-level
        # declaration or the rustc default (bin).
        rec = self.write_test("//@ check-pass\nfn main() {}\n")
        res, captured = self.run_with_conv_capture(rec, conv())
        self.assertEqual(res["status"], "pass")
        self.assertNotIn("--crate-type", captured["rustc_args"])
        self.assertFalse(any(a.startswith("--crate-type=")
                             for a in captured["rustc_args"]))

    def test_pretty_forwards_lib_crate_type(self):
        # A pretty test is never compiled as a binary by compiletest
        # (it stops at -Zunpretty output); the conversion proxy keeps
        # the lib shape.
        rec = self.write_test("pub fn f() {}\n", suite="pretty")
        res, captured = self.run_with_conv_capture(rec, conv())
        self.assertEqual(res["status"], "pass")
        args = captured["rustc_args"]
        i = args.index("--crate-type")
        self.assertEqual(args[i + 1], "lib")

    def test_pretty_declared_crate_type_is_left_alone(self):
        rec = self.write_test("#![crate_type = \"rlib\"]\n"
                              "pub fn f() {}\n", suite="pretty")
        res, captured = self.run_with_conv_capture(rec, conv())
        self.assertEqual(res["status"], "pass")
        self.assertNotIn("--crate-type", captured["rustc_args"])

    def test_own_crate_type_flag_wins_over_suite_default(self):
        rec = self.write_test(
            "//@ compile-flags: --crate-type=cdylib\n"
            "pub fn f() {}\n", suite="mir-opt")
        res, captured = self.run_with_conv_capture(rec, conv())
        self.assertEqual(res["status"], "pass")
        args = captured["rustc_args"]
        self.assertNotIn("--crate-type", args)
        self.assertIn("--crate-type=cdylib", args)

    def test_target_spec_file_is_forwarded_expanded(self):
        rec = self.write_test(
            "//@ check-pass\n"
            "//@ compile-flags: --target={{src-base}}/custom.json\n"
            "fn main() {}\n")
        res, captured = self.run_with_conv_capture(rec, conv())
        self.assertEqual(res["status"], "pass")
        tgt = [a for a in captured["rustc_args"]
               if a.startswith("--target")]
        self.assertEqual(len(tgt), 1)
        self.assertTrue(tgt[0].endswith("/custom.json"))
        self.assertNotIn("{{", tgt[0])

    def test_unknown_target_spec_rejection_is_attempted(self):
        rec = self.write_test(
            "//@ check-fail\n"
            "//@ compile-flags: --target x86_64-linux-gnu\n"
            "fn main() {}\n")
        orig = run.known_targets
        run.known_targets = lambda: {"x86_64-unknown-linux-gnu"}
        try:
            res, captured = self.run_with_conv_capture(
                rec, conv(exit_code=1, compilation_failed=True,
                          converted=None, skipped=None))
        finally:
            run.known_targets = orig
        self.assertEqual(res["status"], "pass")
        self.assertIn("--target", captured["rustc_args"])

    def test_unpretty_dropped_when_dump_unchecked(self):
        rec = self.write_test(
            "//@ check-pass\n//@ dont-check-compiler-stdout\n"
            "//@ compile-flags: -Zunpretty=mir\nfn main() {}\n")
        res, captured = self.run_with_conv_capture(rec, conv())
        self.assertEqual(res["status"], "pass")
        self.assertNotIn("-Zunpretty=mir", captured["rustc_args"])

    def test_unpretty_failure_without_snapshot_is_attempted(self):
        rec = self.write_test(
            "//@ check-fail\n//@ compile-flags: -Zunpretty=mir\n"
            "fn main() {}\n")
        res, captured = self.run_with_conv_capture(
            rec, conv(exit_code=1, compilation_failed=True,
                      converted=None, skipped=None))
        self.assertEqual(res["status"], "pass")
        self.assertNotIn("-Zunpretty=mir", captured["rustc_args"])

    def test_unpretty_checked_dump_is_attempted_and_compared(self):
        # The dump is the test's assertion: the view is attempted
        # as a print invocation, the flag kept, the compiler
        # stdout captured, and a matching dump passes.
        rec = self.write_test(
            "//@ check-fail\n//@ compile-flags: -Zunpretty=hir\n"
            "fn main() {}\n")
        (self.rust_dir / "tests" / "ui" / "t.stdout").write_text(
            "dump\n")
        res, captured = self.run_with_conv_capture(
            rec, conv(exit_code=1, compilation_failed=True,
                      converted=None, skipped=None,
                      compiler_stdout="dump\n",
                      compiler_stdout_dir=""))
        self.assertEqual(res["status"], "pass")
        self.assertTrue(captured["capture_stdout"])
        self.assertIn("-Zunpretty=hir", captured["rustc_args"])

    def test_unpretty_checked_dump_mismatch_fails(self):
        rec = self.write_test(
            "//@ check-fail\n//@ compile-flags: -Zunpretty=hir\n"
            "fn main() {}\n")
        (self.rust_dir / "tests" / "ui" / "t.stdout").write_text(
            "dump\n")
        res, _ = self.run_with_conv_capture(
            rec, conv(exit_code=1, compilation_failed=True,
                      converted=None, skipped=None,
                      compiler_stdout="other\n",
                      compiler_stdout_dir=""))
        self.assertEqual(res["status"], "fail")
        self.assertEqual(res["bucket"], "compiler stdout mismatch")

    def test_unpretty_success_dump_mismatch_fails(self):
        rec = self.write_test(
            "//@ check-pass\n//@ compile-flags: -Zunpretty=mir\n"
            "fn main() {}\n")
        (self.rust_dir / "tests" / "ui" / "t.stdout").write_text(
            "dump\n")
        res, _ = self.run_with_conv_capture(
            rec, conv(compiler_stdout="other\n",
                      compiler_stdout_dir=""))
        self.assertEqual(res["status"], "fail")
        self.assertEqual(res["bucket"], "compiler stdout mismatch")

    def test_print_request_is_compared(self):
        rec = self.write_test(
            "//@ check-pass\n//@ compile-flags: --print cfg\n"
            "fn main() {}\n")
        (self.rust_dir / "tests" / "ui" / "t.stdout").write_text(
            "answer\n")
        res, captured = self.run_with_conv_capture(
            rec, conv(compiler_stdout="answer\n",
                      compiler_stdout_dir=""))
        self.assertEqual(res["status"], "pass")
        self.assertTrue(captured["capture_stdout"])
        self.assertIn("--print", captured["rustc_args"])

    def test_print_missing_capture_fails(self):
        # A print invocation whose capture was never produced
        # cannot pass: the asserted output was never compared.
        rec = self.write_test(
            "//@ check-pass\n//@ compile-flags: --print cfg\n"
            "fn main() {}\n")
        res, _ = self.run_with_conv_capture(rec, conv())
        self.assertEqual(res["status"], "fail")
        self.assertEqual(res["bucket"], "compiler stdout mismatch")
        self.assertIn("capture", res["reason"])

    def test_print_dont_check_stdout_skips_comparison(self):
        rec = self.write_test(
            "//@ check-pass\n//@ dont-check-compiler-stdout\n"
            "//@ compile-flags: --print cfg\nfn main() {}\n")
        res, _ = self.run_with_conv_capture(
            rec, conv(compiler_stdout="anything\n",
                      compiler_stdout_dir=""))
        self.assertEqual(res["status"], "pass")




class PrintStdoutJudgeTests(unittest.TestCase):
    """judge_print_stdout's exemptions and capture discipline."""

    class View:
        def __init__(self, directives=()):
            self.directives = set(directives)

        def has(self, name):
            return name in self.directives

    def judge(self, suite, view, conv_result):
        return run.judge_print_stdout(suite, view, Path("t.rs"), None,
                                      conv_result)

    def test_crashes_suite_compares_no_stdout(self):
        # A crashing compiler leaves a partial dump behind; the
        # crash is the verdict, so any capture passes.
        out = self.judge("crashes", self.View(),
                         conv(compiler_stdout="partial dump",
                              compiler_stdout_dir=""))
        self.assertIsNone(out)

    def test_dont_check_stdout_compares_nothing(self):
        out = self.judge(
            "ui", self.View({"dont-check-compiler-stdout"}),
            conv(compiler_stdout="anything", compiler_stdout_dir=""))
        self.assertIsNone(out)

    def test_missing_capture_is_a_fail_not_a_pass(self):
        out = self.judge("ui", self.View(), conv())
        self.assertIsNotNone(out)
        self.assertEqual(out[0], "fail")
        self.assertEqual(out[2], "compiler stdout mismatch")


class ConvertSideTests(unittest.TestCase):
    """convert_side's driver argv (run_cmd stubbed)."""

    def run_argv(self, **kw):
        orig = run.run_cmd
        seen = []

        def stub(argv, *, timeout, cwd=None, env_extra=None,
                 env_remove=(), mem_limit_mb=None):
            seen.append(argv)
            return {"exit_code": 0, "timeout": False,
                    "stdout": "converted: 1, skipped: 0\n",
                    "stderr": ""}

        run.run_cmd = stub
        try:
            out = run.convert_side(Path("t.rs"), "2021",
                                   Path("/fake/driver"), **kw)
        finally:
            run.run_cmd = orig
        return seen[0], out

    def run_stdout(self, stdout):
        orig = run.run_cmd

        def stub(argv, *, timeout, cwd=None, env_extra=None,
                 env_remove=(), mem_limit_mb=None):
            return {"exit_code": 0, "timeout": False,
                    "stdout": stdout, "stderr": ""}

        run.run_cmd = stub
        try:
            return run.convert_side(Path("t.rs"), "2021",
                                    Path("/fake/driver"))
        finally:
            run.run_cmd = orig

    def test_check_mode_adds_driver_option(self):
        argv, out = self.run_argv(check_mode=True)
        self.assertIn("--compiletest-check-mode", argv)

    def test_check_mode_absent_by_default(self):
        argv, out = self.run_argv()
        self.assertNotIn("--compiletest-check-mode", argv)

    def test_not_demanded_lines_are_parsed(self):
        out = self.run_stdout(
            "sparagmos-driver: t.rs (Stable MIR, 1.100.0)\n"
            "fn main: blocks=1 locals=2 args=0 [converted]\n"
            "converted: 10, skipped: 0\n"
            "not demanded: 4\n"
            "target-crate converted: 6, skipped: 0\n"
            "target-crate not demanded: 2\n")
        self.assertEqual(out["converted"], 10)
        self.assertEqual(out["skipped"], 0)
        self.assertEqual(out["target_converted"], 6)
        self.assertEqual(out["target_skipped"], 0)
        self.assertEqual(out["not_demanded"], 4)
        self.assertEqual(out["target_not_demanded"], 2)

    def test_capture_stdout_adds_driver_option(self):
        argv, out = self.run_argv(capture_stdout=True)
        i = argv.index("--compiler-stdout")
        self.assertTrue(argv[i + 1].endswith("stdout.txt"))
        # compiletest's source-path simulation rides along: the
        # snapshots carry $SRC_DIR for libstd spans.
        flat = " ".join(argv)
        self.assertIn("-Zsimulate-remapped-rust-src-base="
                      "/rustc/FAKE_PREFIX", flat)
        self.assertIn("-Ztranslate-remapped-path-to-local-path=no",
                      flat)
        # The stub never creates the capture file: the result
        # reports the capture as absent, not as empty output.
        self.assertIsNone(out["compiler_stdout"])
        self.assertTrue(out["compiler_stdout_dir"])

    def test_no_capture_by_default(self):
        argv, out = self.run_argv()
        self.assertNotIn("--compiler-stdout", argv)
        self.assertNotIn("simulate-remapped", " ".join(argv))
        self.assertIsNone(out["compiler_stdout"])

    def test_capture_forces_compiletest_crate_version(self):
        # Bootstrap runs compiletest with
        # RUSTC_FORCE_RUSTC_VERSION=compiletest, which the
        # compiler folds into StableCrateId: a captured-stdout
        # run reproduces that environment so printed DefId
        # hashes match the snapshots.
        orig = run.run_cmd
        seen = {}

        def stub(argv, *, timeout, cwd=None, env_extra=None,
                 env_remove=(), mem_limit_mb=None):
            seen.update(env_extra or {})
            return {"exit_code": 0, "timeout": False,
                    "stdout": "converted: 1, skipped: 0\n",
                    "stderr": ""}

        run.run_cmd = stub
        try:
            run.convert_side(Path("t.rs"), "2021",
                             Path("/fake/driver"), capture_stdout=True)
            self.assertEqual(seen.get("RUSTC_FORCE_RUSTC_VERSION"),
                             "compiletest")
            seen.clear()
            run.convert_side(Path("t.rs"), "2021",
                             Path("/fake/driver"))
            self.assertNotIn("RUSTC_FORCE_RUSTC_VERSION", seen)
        finally:
            run.run_cmd = orig

    def test_counts_default_to_none_without_report_lines(self):
        out = self.run_stdout("converted: 1, skipped: 0\n")
        self.assertIsNone(out["not_demanded"])
        self.assertIsNone(out["target_not_demanded"])
        self.assertIsNone(out["target_converted"])
        self.assertIsNone(out["target_skipped"])

    def test_aux_specs_become_aux_args(self):
        argv, out = self.run_argv(
            aux_specs=[("helper", Path("/x/helper.rs"))])
        i = argv.index("--aux")
        self.assertEqual(argv[i + 1], "helper=/x/helper.rs")
        self.assertEqual(out["converted"], 1)

    def test_no_aux_args_by_default(self):
        argv, _ = self.run_argv()
        self.assertNotIn("--aux", argv)

    def test_aux_attributes_become_aux_arg_attributes(self):
        argv, _ = self.run_argv(aux_specs=[
            run.AuxSpec("pm", Path("/x/pm.rs"), "proc-macro", None),
            run.AuxSpec("old", Path("/x/old.rs"), None, "2015"),
        ])
        args = [argv[i + 1] for i, a in enumerate(argv) if a == "--aux"]
        self.assertEqual(args, ["pm=/x/pm.rs;crate-type=proc-macro",
                                "old=/x/old.rs;edition=2015"])

    def test_aux_cfg_becomes_aux_arg_attribute(self):
        spec = run.AuxSpec("a", Path("/x/a.rs"), None, None, None,
                           "rpass1")
        self.assertEqual(run.aux_spec_arg(spec),
                         "a=/x/a.rs;cfg=rpass1")

    def test_stamp_revision_cfg(self):
        specs = [run.AuxSpec("a", Path("/x/a.rs")),
                 ("b", Path("/x/b.rs"))]
        stamped = run.stamp_revision_cfg(specs, "rpass1")
        self.assertEqual([sp.cfg for sp in stamped],
                         ["rpass1", "rpass1"])
        # The resolved specs are not mutated, and a revisionless
        # attempt passes its specs through unchanged.
        self.assertIsNone(specs[0].cfg)
        self.assertIs(run.stamp_revision_cfg(specs, None), specs)
        # A hyphenated revision is stamped with the cfg name
        # compiletest derives from it, not the raw revision name:
        # rustc rejects the raw form as an invalid --cfg argument.
        stamped = run.stamp_revision_cfg(specs, "with-diag-in-deps")
        self.assertEqual([sp.cfg for sp in stamped],
                         ["with_diag_in_deps", "with_diag_in_deps"])


class ProcessStage2AuxTests(unittest.TestCase):
    """process_test(stage2=True) on tests with auxiliary crates."""

    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile")
                                         .TemporaryDirectory()))
        (self.tmp / "tests" / "ui" / "auxiliary").mkdir(parents=True)
        self.rust_dir = self.tmp
        self.seen = []

    def write_test(self, body, aux_files=()):
        p = self.rust_dir / "tests" / "ui" / "t.rs"
        p.write_text(body)
        for name, text in aux_files:
            (self.rust_dir / "tests" / "ui" / "auxiliary"
             / name).write_text(text)
        return {"suite": "ui", "path": "tests/ui/t.rs",
                "nature": "compile-check", "single_file": False,
                "ffi": False, "env_excluded": False}

    def run_with_conv(self, rec, conv_result):
        orig = run.convert_side

        def stub(path, edition, driver_bin, rustc_args=(),
                 aux_specs=(), env_extra=None, env_remove=(),
                 jit=False, aux_env_scrub=(), capture_stdout=False,
                 aux_libs=(), check_mode=False):
            self.seen.append([(spec.name, spec.path.name)
                               for spec in aux_specs])
            return conv_result

        run.convert_side = stub
        try:
            return run.process_test(rec, self.rust_dir,
                                    Path("/fake/driver"), stage2=True)
        finally:
            run.convert_side = orig

    def test_aux_test_is_attempted_and_forwards_specs(self):
        rec = self.write_test(
            "//@ check-pass\n//@ aux-build: helper.rs\nfn main() {}\n",
            [("helper.rs", "pub fn f() {}\n")])
        res = self.run_with_conv(rec, conv())
        self.assertEqual(res["status"], "pass")
        self.assertEqual(self.seen, [[("helper", "helper.rs")]])

    def test_transitive_aux_forwarded_deps_first(self):
        rec = self.write_test(
            "//@ check-pass\n//@ aux-build: mid.rs\nfn main() {}\n",
            [("mid.rs", "//@ aux-build: base.rs\npub fn g() {}\n"),
             ("base.rs", "pub fn f() {}\n")])
        res = self.run_with_conv(rec, conv())
        self.assertEqual(res["status"], "pass")
        self.assertEqual(self.seen,
                         [[("base", "base.rs"), ("mid", "mid.rs")]])

    def test_missing_aux_file_is_not_attempted(self):
        rec = self.write_test(
            "//@ check-pass\n//@ aux-build: nope.rs\nfn main() {}\n")
        res = run.process_test(rec, self.rust_dir, Path("/fake/driver"),
                               stage2=True)
        self.assertEqual(res["status"], "not-attempted")
        self.assertIn("multi-file", res["bucket"])
        self.assertIn("not found", res["reason"])

    def test_check_fail_aux_test_passes_on_clean_rejection(self):
        rec = self.write_test(
            "//@ check-fail\n//@ aux-build: helper.rs\nfn main() {}\n",
            [("helper.rs", "pub fn f() {}\n")])
        res = self.run_with_conv(
            rec, conv(exit_code=1, compilation_failed=True,
                      converted=None, skipped=None))
        self.assertEqual(res["status"], "pass")
        self.assertEqual(self.seen, [[("helper", "helper.rs")]])

    def test_scoped_aux_resolved_per_revision(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@[a] check-pass\n"
            "//@[b] check-pass\n//@[a] aux-build: helper.rs\n"
            "fn main() {}\n",
            [("helper.rs", "pub fn f() {}\n")])
        res = self.run_with_conv(rec, conv())
        self.assertEqual(res["status"], "pass")
        # Revision a builds the aux crate; revision b converts bare.
        self.assertEqual(self.seen, [[("helper", "helper.rs")], []])


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
        orig = run.installed_targets
        run.installed_targets = lambda: {run.HOST_TARGET}
        try:
            for flags in (["--target", "aarch64-unknown-linux-gnu"],
                          ["--target=i686-pc-windows-msvc"]):
                reason = run.screen_flags(flags)
                self.assertIn("--target", reason)
                self.assertIn("not installed", reason)
        finally:
            run.installed_targets = orig

    def test_installed_target_passes(self):
        targets = {"aarch64-unknown-linux-gnu", run.HOST_TARGET}
        self.assertIsNone(run.screen_flags(
            ["--target", "aarch64-unknown-linux-gnu"], targets=targets))
        self.assertIsNone(run.screen_flags(
            ["--target=aarch64-unknown-linux-gnu"], targets=targets))

    def test_default_targets_are_the_installed_ones(self):
        orig = run.installed_targets
        run.installed_targets = lambda: {"aarch64-unknown-linux-gnu",
                                         run.HOST_TARGET}
        try:
            self.assertIsNone(run.screen_flags(
                ["--target", "aarch64-unknown-linux-gnu"]))
            # The executing stage passes only the host target: an
            # installed foreign target is then blocked, with the
            # execution reason rather than the not-installed one.
            reason = run.screen_flags(
                ["--target", "aarch64-unknown-linux-gnu"],
                targets={run.HOST_TARGET})
            self.assertIn("--target", reason)
            self.assertIn("executes the program", reason)
        finally:
            run.installed_targets = orig

    def test_targets_in_sysroot(self):
        with tempfile.TemporaryDirectory() as td:
            rustlib = Path(td) / "lib" / "rustlib"
            for triple in ("x86_64-unknown-linux-gnu",
                           "aarch64-unknown-linux-gnu"):
                (rustlib / triple / "lib").mkdir(parents=True)
            (rustlib / "src").mkdir(parents=True)
            (rustlib / "components").write_text("")
            self.assertEqual(
                run._targets_in_sysroot(td),
                {"x86_64-unknown-linux-gnu",
                 "aarch64-unknown-linux-gnu"})
        self.assertEqual(run._targets_in_sysroot("/nonexistent"), set())

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

    def test_print_allowed_when_caller_judges_output(self):
        self.assertIsNone(run.screen_flags(["--print", "cfg"],
                                           allow_print=True))
        self.assertIsNone(run.screen_flags(["--print=cfg"],
                                           allow_print=True))
        self.assertIsNone(run.screen_flags(["-Zunpretty=mir"],
                                           allow_print=True))
        self.assertIsNone(run.screen_flags(["-Z", "unpretty=mir"],
                                           allow_print=True))
        # Other refusals still apply to a print invocation.
        self.assertIn("--sysroot", run.screen_flags(
            ["--print", "cfg", "--sysroot=/x"], allow_print=True))


class CompileFlagsExpansionTests(unittest.TestCase):
    def test_expands_variables_when_bases_supplied(self):
        h = header("//@ compile-flags: --target={{src-base}}/x.json"
                   " --remap-path-prefix={{src-base}}=/s\n")
        self.assertEqual(
            run.compile_flags(h.view(None), "/src/base", "/build/base"),
            ["--target=/src/base/x.json",
             "--remap-path-prefix=/src/base=/s"])

    def test_raw_tokens_without_bases(self):
        h = header("//@ compile-flags: --target={{src-base}}/x.json\n")
        self.assertEqual(run.compile_flags(h.view(None)),
                         ["--target={{src-base}}/x.json"])


class ScreenFlagsTargetSpecTests(unittest.TestCase):
    def setUp(self):
        self.orig = run.installed_targets
        run.installed_targets = lambda: {run.HOST_TARGET}

    def tearDown(self):
        run.installed_targets = self.orig

    def test_target_spec_file_is_not_refused(self):
        self.assertIsNone(run.screen_flags(
            ["--target", "/src/codegen/custom.json"]))
        self.assertIsNone(run.screen_flags(
            ["--target=/src/codegen/custom.json"]))

    def test_unknown_triple_forwarded_for_failure_expectation(self):
        known = {run.HOST_TARGET, "aarch64-unknown-linux-gnu"}
        self.assertIsNone(run.screen_flags(
            ["--target", "x86_64-linux-gnu"], known=known,
            expectation="failure"))

    def test_unknown_triple_refused_for_success_expectation(self):
        known = {run.HOST_TARGET, "aarch64-unknown-linux-gnu"}
        reason = run.screen_flags(
            ["--target", "x86_64-linux-gnu"], known=known,
            expectation="success")
        self.assertIn("not a target of this toolchain", reason)

    def test_known_triple_without_std_stays_refused_for_failure(self):
        # The only failure available there is the environment's
        # (no std component), not the test's; forwarding it would
        # manufacture a pass on a failure-expecting test.
        known = {run.HOST_TARGET, "amdgcn-amd-amdhsa"}
        reason = run.screen_flags(
            ["--target", "amdgcn-amd-amdhsa"], known=known,
            expectation="failure")
        self.assertIn("not installed", reason)

    def test_unknown_triple_without_known_set_stays_refused(self):
        reason = run.screen_flags(
            ["--target", "x86_64-linux-gnu"], expectation="failure")
        self.assertIn("not installed", reason)


class UnprettyTests(unittest.TestCase):
    def test_has_and_strip_in_both_spellings(self):
        flags = ["-Zunpretty=mir", "-C", "opt-level=0", "-Z",
                 "unpretty=expanded", "--crate-type", "lib"]
        self.assertTrue(run.has_unpretty(flags))
        self.assertEqual(run.strip_unpretty(flags),
                         ["-C", "opt-level=0", "--crate-type", "lib"])
        self.assertFalse(run.has_unpretty(["-Z", "mir-opt-level=0"]))

    def _path(self, td):
        path = Path(td) / "t.rs"
        path.write_text("fn main() {}\n")
        return path

    def test_unchecked_with_dont_check_compiler_stdout(self):
        with tempfile.TemporaryDirectory() as td:
            view = header("//@ dont-check-compiler-stdout\n").view(None)
            self.assertTrue(run.unpretty_unchecked(
                view, self._path(td), None, "success"))

    def test_failure_without_stdout_snapshot_is_unchecked(self):
        with tempfile.TemporaryDirectory() as td:
            view = header("//@ check-fail\n").view(None)
            self.assertTrue(run.unpretty_unchecked(
                view, self._path(td), None, "failure"))

    def test_failure_with_stdout_snapshot_stays_checked(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "t.stdout").write_text("dump\n")
            view = header("//@ check-fail\n").view(None)
            self.assertFalse(run.unpretty_unchecked(
                view, self._path(td), None, "failure"))

    def test_revision_stdout_snapshot_stays_checked(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "t.expanded.stdout").write_text("dump\n")
            view = header("//@ check-fail\n").view(None)
            self.assertFalse(run.unpretty_unchecked(
                view, self._path(td), "expanded", "failure"))

    def test_success_without_dont_check_stays_checked(self):
        with tempfile.TemporaryDirectory() as td:
            view = header("//@ check-pass\n").view(None)
            self.assertFalse(run.unpretty_unchecked(
                view, self._path(td), None, "success"))

    def test_crash_expectation_never_unchecked(self):
        with tempfile.TemporaryDirectory() as td:
            view = header("//@ dont-check-compiler-stdout\n").view(None)
            self.assertFalse(run.unpretty_unchecked(
                view, self._path(td), None, "crash"))


class HeaderBucketTests(unittest.TestCase):
    def test_ignore_test_has_its_own_bucket(self):
        self.assertEqual(
            run.header_bucket("ignore-test (compiletest itself "
                              "never runs this test)"),
            "not attempted (ignore-test)")

    def test_other_blockers_keep_the_header_feature_bucket(self):
        self.assertEqual(
            run.header_bucket("stdin directive present (guest stdin "
                              "is not modelled)"),
            "not attempted (header feature)")


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
        def stub(path, edition, driver_bin, rustc_args=(),
                 aux_specs=(), env_extra=None, env_remove=(),
                 jit=False, aux_env_scrub=(), capture_stdout=False,
                 aux_libs=(), check_mode=False):
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
        orig = run.installed_targets
        run.installed_targets = lambda: {run.HOST_TARGET}
        try:
            res = self.run_with_fake(rec, lambda args: conv())
        finally:
            run.installed_targets = orig
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"], "not attempted (compile-flags)")
        self.assertIn("revision b", res["reason"])

    def test_installed_target_flags_reach_the_driver(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@[a] check-pass\n//@[b] check-pass\n"
            "//@[b] compile-flags: --target aarch64-unknown-linux-gnu\n"
            "fn main() {}\n")
        orig = run.installed_targets
        run.installed_targets = lambda: {"aarch64-unknown-linux-gnu",
                                         run.HOST_TARGET}
        try:
            res = self.run_with_fake(rec, lambda args: conv())
        finally:
            run.installed_targets = orig
        self.assertEqual(res["status"], "pass")
        b_calls = [c for c in self.calls if c[:2] == ["--cfg", "b"]]
        self.assertEqual(len(b_calls), 1)
        self.assertIn("--target", b_calls[0])
        self.assertIn("aarch64-unknown-linux-gnu", b_calls[0])

    def test_unscoped_flags_reach_every_revision(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@ compile-flags: -C opt-level=2\n"
            "//@[a] check-pass\n//@[b] check-pass\nfn main() {}\n")
        res = self.run_with_fake(rec, lambda args: conv())
        self.assertEqual(res["status"], "pass")
        for call in self.calls:
            self.assertIn("opt-level=2", call)


class ProcessStage1FlagsTests(unittest.TestCase):
    """Stage 1 forwards compile-flags to both sides, preceded by
    the mode arguments compiletest itself adds for the suite
    (a ui test is compiled with -Ccodegen-units=1)."""

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
        def fake_native(path, edition, workdir, extra_flags=(),
                        aux_specs=(), **kw):
            self.seen["native_flags"] = list(extra_flags)
            return ({"exit_code": 0, "stdout": "", "stderr": ""}, None)
        def fake_spar(path, edition, driver_bin, cwd=None,
                      rustc_args=(), aux_specs=(), **kw):
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
                         ["-Ccodegen-units=1", "-C", "opt-level=3"])
        self.assertEqual(self.seen["driver_args"],
                         ["-Ccodegen-units=1", "-C", "opt-level=3"])

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


class HostProfileTests(unittest.TestCase):
    """The host profiles behind SPARAGMOS_HOST_TARGET (issue #517):
    the default profile is the environment model the runner has
    always used, and selecting another profile switches the whole
    model (target, cfg set, needs-* rules) at once."""

    # The x86_64-linux model as it stood before profiles existed.
    DEFAULT_CFGS = {
        "linux", "unix", "x86_64", "x86_64-unknown-linux-gnu",
        "target_os=\"linux\"", "target_arch=\"x86_64\"",
        "target_family=\"unix\"", "target_env=\"gnu\"",
        "target_vendor=\"unknown\"", "target_pointer_width=\"64\"",
        "target_endian=\"little\"", "64bit", "gnu", "elf",
    }
    DEFAULT_RULES = {
        "needs-run-enabled": True, "needs-unwind": True,
        "needs-threads": True, "needs-subprocess": True,
        "needs-asm-support": True, "needs-dlltool": False,
        "needs-deterministic-layouts": True,
        "needs-rustc-debug-assertions": False,
        "needs-std-debug-assertions": False, "needs-enzyme": False,
        "needs-offload": False, "needs-force-clang-based-tests": False,
        "needs-wasmtime": False, "needs-symlink": True,
        "needs-target-std": True,
    }

    def setUp(self):
        run.select_host_profile(run.DEFAULT_HOST_TARGET)

    def tearDown(self):
        run.select_host_profile(run.DEFAULT_HOST_TARGET)

    def test_default_profile_is_the_historical_model(self):
        self.assertEqual(run.HOST_TARGET, "x86_64-unknown-linux-gnu")
        self.assertEqual(run.HOST_TRUE_CFGS, self.DEFAULT_CFGS)
        self.assertEqual(run.NEEDS_HOST_RULES, self.DEFAULT_RULES)

    def test_windows_gnu_profile(self):
        run.select_host_profile("x86_64-pc-windows-gnu")
        self.assertEqual(run.HOST_TARGET, "x86_64-pc-windows-gnu")
        self.assertIn("windows", run.HOST_TRUE_CFGS)
        self.assertIn("windows-gnu", run.HOST_TRUE_CFGS)
        self.assertNotIn("unix", run.HOST_TRUE_CFGS)
        self.assertNotIn("elf", run.HOST_TRUE_CFGS)
        self.assertTrue(run.NEEDS_HOST_RULES["needs-dlltool"])

    def test_windows_msvc_profile_has_no_dlltool(self):
        run.select_host_profile("x86_64-pc-windows-msvc")
        self.assertIn("windows-msvc", run.HOST_TRUE_CFGS)
        self.assertFalse(run.NEEDS_HOST_RULES["needs-dlltool"])

    def test_darwin_profile(self):
        run.select_host_profile("aarch64-apple-darwin")
        self.assertIn("apple", run.HOST_TRUE_CFGS)
        self.assertIn("macos", run.HOST_TRUE_CFGS)
        self.assertIn("aarch64", run.HOST_TRUE_CFGS)
        self.assertNotIn("elf", run.HOST_TRUE_CFGS)

    def test_applicability_follows_the_selected_profile(self):
        h = header("//@ only-aarch64\nfn main() {}\n")
        status, _why = run.applicability(h.view(None), None)
        self.assertEqual(status, "out-of-environment")
        run.select_host_profile("aarch64-unknown-linux-gnu")
        status, _why = run.applicability(h.view(None), None)
        self.assertEqual(status, "applicable")

    def test_unknown_target_is_rejected(self):
        with self.assertRaises(ValueError):
            run.select_host_profile("riscv64gc-unknown-linux-gnu")



class DebuggerCommandTranslationTests(unittest.TestCase):
    """The session-script translation of debugger commands:
    history references and their comparisons are within the
    attempted scope; malformed comparisons are not."""

    def test_history_comparison_translates(self):
        self.assertEqual(
            run.translate_debugger_command("gdb", "print $1 == $2"),
            "print $1 == $2")
        self.assertEqual(
            run.translate_debugger_command("gdb", "print $1 != $2"),
            "print $1 != $2")
        self.assertEqual(
            run.translate_debugger_command("lldb", "print $0 == $1"),
            "v $0 == $1")

    def test_history_reference_translates(self):
        self.assertEqual(
            run.translate_debugger_command("gdb", "print $1"),
            "print $1")
        self.assertEqual(
            run.translate_debugger_command("gdb", "print $12.field"),
            "print $12.field")

    def test_parenthesised_comparison_stays_outside_scope(self):
        self.assertIsNone(
            run.translate_debugger_command("gdb", "print ($1 == $2)"))

    def test_dangling_comparison_stays_outside_scope(self):
        self.assertIsNone(
            run.translate_debugger_command("gdb", "print $1 =="))
        self.assertIsNone(
            run.translate_debugger_command("gdb", "print == $2"))


if __name__ == "__main__":
    unittest.main()


class AuxDirectivesTests(unittest.TestCase):
    def test_aux_build_name_derivation(self):
        header = run.Header("//@ aux-build: my-helper.rs\n")
        self.assertEqual(run.aux_directives(header),
                         [("my_helper", "my-helper.rs")])

    def test_aux_crate_explicit_name(self):
        header = run.Header("//@ aux-crate: parser=parser.rs\n")
        self.assertEqual(run.aux_directives(header),
                         [("parser", "parser.rs")])

    def test_proc_macro_entry_carries_crate_type(self):
        header = run.Header("//@ proc-macro: my-macro.rs\n")
        self.assertEqual(run.aux_entries(header),
                         [("my_macro", "my-macro.rs", "proc-macro",
                           None)])
        # The pair view keeps the crate (it is an aux dependency).
        self.assertEqual(run.aux_directives(header),
                         [("my_macro", "my-macro.rs")])


class ResolveAuxSpecsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile")
                                         .TemporaryDirectory()))
        self.dir = self.tmp / "tests" / "ui"
        (self.dir / "auxiliary").mkdir(parents=True)

    def write(self, rel, body):
        p = self.dir / rel
        p.write_text(body)
        return p

    def test_single_aux_in_auxiliary_dir(self):
        main = self.write("t.rs", "//@ run-pass\n//@ aux-build: helper.rs\n")
        self.write("auxiliary/helper.rs", "pub fn f() {}\n")
        specs, err = run.resolve_aux_specs(main)
        self.assertIsNone(err)
        self.assertEqual([spec.name for spec in specs], ["helper"])

    def test_transitive_aux_comes_first(self):
        main = self.write(
            "t.rs", "//@ run-pass\n//@ aux-build: mid.rs\n")
        self.write("auxiliary/mid.rs",
                   "//@ aux-build: base.rs\npub fn g() {}\n")
        self.write("auxiliary/base.rs", "pub fn f() {}\n")
        specs, err = run.resolve_aux_specs(main)
        self.assertIsNone(err)
        self.assertEqual([spec.name for spec in specs], ["base", "mid"])

    def test_missing_aux_file_is_an_error(self):
        main = self.write("t.rs", "//@ run-pass\n//@ aux-build: nope.rs\n")
        specs, err = run.resolve_aux_specs(main)
        self.assertEqual(specs, [])
        self.assertIn("not found", err)

    def test_no_directives_resolves_to_nothing(self):
        main = self.write("t.rs", "//@ run-pass\nfn main() {}\n")
        specs, err = run.resolve_aux_specs(main)
        self.assertIsNone(err)
        self.assertEqual(specs, [])

    def test_proc_macro_directive_marks_the_spec(self):
        main = self.write(
            "t.rs", "//@ run-pass\n//@ proc-macro: my-macro.rs\n")
        self.write("auxiliary/my-macro.rs", "pub fn f() {}\n")
        specs, err = run.resolve_aux_specs(main)
        self.assertIsNone(err)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].name, "my_macro")
        self.assertEqual(specs[0].crate_type, "proc-macro")

    def test_aux_proc_macro_dependency_comes_first(self):
        # An auxiliary file's own `//@ proc-macro:` names a crate
        # the auxiliary build needs first (dependency order).
        main = self.write(
            "t.rs", "//@ run-pass\n//@ aux-build: wrapper.rs\n")
        self.write("auxiliary/wrapper.rs",
                   "//@ proc-macro: inner-macro.rs\npub fn g() {}\n")
        self.write("auxiliary/inner-macro.rs", "pub fn f() {}\n")
        specs, err = run.resolve_aux_specs(main)
        self.assertIsNone(err)
        self.assertEqual([spec.name for spec in specs],
                         ["inner_macro", "wrapper"])
        self.assertEqual(specs[0].crate_type, "proc-macro")
        self.assertIsNone(specs[1].crate_type)

    def test_top_proc_macro_excluded_on_request(self):
        # Stage 2 builds the test's own proc-macro crates itself,
        # so they must not also become --aux specs; a proc-macro
        # crate an auxiliary file names is transitive and stays.
        main = self.write(
            "t.rs", "//@ run-pass\n//@ aux-build: wrapper.rs\n"
                    "//@ proc-macro: top-macro.rs\n")
        self.write("auxiliary/wrapper.rs",
                   "//@ proc-macro: inner-macro.rs\npub fn g() {}\n")
        self.write("auxiliary/inner-macro.rs", "pub fn f() {}\n")
        self.write("auxiliary/top-macro.rs", "pub fn h() {}\n")
        specs, err = run.resolve_aux_specs(main)
        self.assertIsNone(err)
        self.assertEqual([spec.name for spec in specs],
                         ["inner_macro", "wrapper", "top_macro"])
        specs, err = run.resolve_aux_specs(main, top_proc_macro=False)
        self.assertIsNone(err)
        self.assertEqual([spec.name for spec in specs],
                         ["inner_macro", "wrapper"])
        self.assertEqual(specs[0].crate_type, "proc-macro")

    def test_aux_edition_comes_from_the_aux_file(self):
        main = self.write(
            "t.rs", "//@ run-pass\n//@ edition: 2024\n"
                    "//@ aux-build: old.rs\n//@ aux-build: plain.rs\n")
        self.write("auxiliary/old.rs",
                   "//@ edition: 2015\npub fn f() {}\n")
        self.write("auxiliary/plain.rs", "pub fn g() {}\n")
        specs, err = run.resolve_aux_specs(main)
        self.assertIsNone(err)
        by_name = {spec.name: spec for spec in specs}
        self.assertEqual(by_name["old"].edition, "2015")
        # No directive of its own: the auxiliary is built with no
        # requested edition (rustc's default 2015), not the main
        # test's edition 2024.
        self.assertEqual(by_name["plain"].edition, "2015")


class ViewBlockerTests(unittest.TestCase):
    def test_unscoped_mode_applies_to_every_revision(self):
        h = header("//@ revisions: a b\n//@ run-pass\n")
        self.assertIsNone(run.stage1_view_blocker(h.view("a")))
        self.assertIsNone(run.stage1_view_blocker(h.view("b")))

    def test_revision_without_a_mode_is_not_blocked(self):
        # A revision view with no run mode is judged as a compile
        # check (compiletest's per-revision default), not blocked.
        h = header("//@ revisions: a b\n//@[a] run-pass\n")
        self.assertIsNone(run.stage1_view_blocker(h.view("a")))
        self.assertIsNone(run.stage1_view_blocker(h.view("b")))

    def test_scoped_blocker_only_blocks_its_revision(self):
        h = header("//@ revisions: a b\n//@[a] run-pass\n"
                   "//@[b] run-pass\n//@[a] add-core-stubs\n")
        self.assertIn("core stub", run.stage1_view_blocker(h.view("a")))
        self.assertIsNone(run.stage1_view_blocker(h.view("b")))

    def test_scoped_run_crash_blocks_no_revision(self):
        # run-crash is attempted and judged by crash class, in
        # every revision that declares it.
        h = header("//@ revisions: a b\n//@[a] run-pass\n"
                   "//@[b] run-crash\n")
        self.assertIsNone(run.stage1_view_blocker(h.view("a")))
        self.assertIsNone(run.stage1_view_blocker(h.view("b")))

    def test_scoped_only_directives_still_block_the_flat_path(self):
        # stage1_blocker is unchanged: scoped directives on a test
        # that declares no revisions keep the record unattempted.
        rec = {"suite": "ui", "path": "tests/ui/x.rs", "nature": "exec",
               "single_file": True, "ffi": False, "env_excluded": False}
        h = header("//@[a] run-pass\n")
        self.assertIn("revisions", run.stage1_blocker(h, rec))


class ProcessStage1RevisionTests(unittest.TestCase):
    """process_test on stage-1 revision tests, both sides stubbed."""

    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile")
                                         .TemporaryDirectory()))
        (self.tmp / "tests" / "ui").mkdir(parents=True)
        self.rust_dir = self.tmp
        self.native_seen = []   # {"rev", "edition", "flags", "aux"}
        self.driver_seen = []   # {"rev", "args", "aux"}

    def write_test(self, body):
        p = self.rust_dir / "tests" / "ui" / "t.rs"
        p.write_text(body)
        return {"suite": "ui", "path": "tests/ui/t.rs",
                "nature": "exec", "single_file": True,
                "ffi": False, "env_excluded": False}

    @staticmethod
    def revision_of(args):
        return args[args.index("--cfg") + 1] if "--cfg" in args else None

    def run_with_fakes(self, rec, native_fake=None, spar_fake=None):
        orig_n, orig_s = run.native_side, run.sparagmos_side

        def fake_native(path, edition, workdir, extra_flags=(),
                        aux_specs=(), **kw):
            seen = {"rev": self.revision_of(list(extra_flags)),
                    "edition": edition, "flags": list(extra_flags),
                    "aux": [spec.name for spec in aux_specs]}
            self.native_seen.append(seen)
            if native_fake:
                return native_fake(seen)
            return ({"exit_code": 0, "stdout": "", "stderr": ""}, None)

        def fake_spar(path, edition, driver_bin, cwd=None,
                      rustc_args=(), aux_specs=(), **kw):
            seen = {"rev": self.revision_of(list(rustc_args)),
                    "args": list(rustc_args),
                    "aux": [spec.name for spec in aux_specs]}
            self.driver_seen.append(seen)
            if spar_fake:
                return spar_fake(seen)
            return ({"exit_code": 0, "stdout": "", "stderr": ""},
                    "result", "")

        run.native_side, run.sparagmos_side = fake_native, fake_spar
        try:
            return run.process_test(rec, self.rust_dir,
                                    Path("/fake/driver"))
        finally:
            run.native_side, run.sparagmos_side = orig_n, orig_s

    def test_all_revisions_pass(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@[a] run-pass\n//@[b] run-pass\n"
            "fn main() {}\n")
        res = self.run_with_fakes(rec)
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["mode"], "run-pass")
        self.assertEqual([s["rev"] for s in self.native_seen],
                         ["a", "b"])
        self.assertEqual([s["rev"] for s in self.driver_seen],
                         ["a", "b"])
        for s in self.native_seen:
            self.assertIn("--check-cfg", s["flags"])
        self.assertEqual(res["revisions"]["a"]["status"], "pass")
        self.assertEqual(res["revisions"]["b"]["status"], "pass")

    def test_one_revision_stdout_mismatch_fails(self):
        # check-run-results puts the run's stdout under
        # comparison (compiletest compares it for no other
        # directive); revision b's guest prints the wrong line.
        rec = self.write_test(
            "//@ revisions: a b\n//@[a] run-pass\n//@[b] run-pass\n"
            "//@ check-run-results\n"
            "fn main() {}\n")

        def spar(seen):
            if seen["rev"] == "b":
                return ({"exit_code": 0, "stdout": "wrong\n",
                         "stderr": ""}, "result", "")
            return ({"exit_code": 0, "stdout": "", "stderr": ""},
                    "result", "")
        res = self.run_with_fakes(rec, spar_fake=spar)
        self.assertEqual(res["status"], "fail")
        self.assertEqual(res["bucket"], "stdout mismatch")
        self.assertIn("revision b", res["reason"])

    def test_one_revision_unsupported(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@[a] run-pass\n//@[b] run-pass\n"
            "fn main() {}\n")

        def spar(seen):
            if seen["rev"] == "b":
                return (None, "unsupported", "some conversion gap")
            return ({"exit_code": 0, "stdout": "", "stderr": ""},
                    "result", "")
        res = self.run_with_fakes(rec, spar_fake=spar)
        self.assertEqual(res["status"], "unsupported")
        self.assertIn("revision b", res["reason"])

    def test_out_of_environment_revision_is_skipped(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@[a] run-pass\n//@[b] run-pass\n"
            "//@[b] ignore-linux\nfn main() {}\n")
        res = self.run_with_fakes(rec)
        self.assertEqual(res["status"], "pass")
        self.assertEqual([s["rev"] for s in self.native_seen], ["a"])

    def test_all_revisions_out_of_environment(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@ run-pass\n//@ ignore-linux\n"
            "fn main() {}\n")
        res = self.run_with_fakes(rec)
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"], "out of environment")
        self.assertEqual(self.native_seen, [])

    def test_revision_without_a_run_mode_is_compile_checked(self):
        # Revision b's view carries no run mode: compiletest
        # judges it by its compile outcome, and a ui revision
        # with no mode directive defaults to check-fail -- so b
        # passes when the driver rejects the program cleanly,
        # while a still runs on both sides as before.
        rec = self.write_test(
            "//@ revisions: a b\n//@[a] run-pass\nfn main() {}\n")
        orig = run.convert_side
        run.convert_side = lambda *a, **kw: conv(
            exit_code=1, compilation_failed=True, converted=0)
        try:
            res = self.run_with_fakes(rec)
        finally:
            run.convert_side = orig
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["revisions"]["b"]["status"], "pass")
        self.assertEqual([s["rev"] for s in self.native_seen], ["a"])

    def test_scoped_flags_reach_only_their_revision(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@[a] run-pass\n//@[b] run-pass\n"
            "//@[a] compile-flags: -C opt-level=2\nfn main() {}\n")
        res = self.run_with_fakes(rec)
        self.assertEqual(res["status"], "pass")
        by_rev = {s["rev"]: s["args"] for s in self.driver_seen}
        self.assertIn("opt-level=2", by_rev["a"])
        self.assertNotIn("opt-level=2", by_rev["b"])
        # The revision arguments precede the compile-flags, the
        # order stage 2 uses.
        self.assertLess(by_rev["a"].index("--cfg"),
                        by_rev["a"].index("opt-level=2"))

    def test_unhonourable_scoped_flags_block_the_record(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@[a] run-pass\n//@[b] run-pass\n"
            "//@[b] compile-flags: --target aarch64-unknown-linux-gnu\n"
            "fn main() {}\n")
        res = self.run_with_fakes(rec)
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"], "not attempted (compile-flags)")
        self.assertIn("revision b", res["reason"])

    def test_scoped_failure_status_is_per_revision(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@ run-fail\n"
            "//@[b] failure-status: 1\nfn main() {}\n")

        def native(seen):
            code = 1 if seen["rev"] == "b" else 101
            return ({"exit_code": code, "stdout": "", "stderr": ""},
                    None)

        def spar(seen):
            code = 1 if seen["rev"] == "b" else 101
            return ({"exit_code": code, "stdout": "", "stderr": ""},
                    "result", "")
        res = self.run_with_fakes(rec, native_fake=native,
                                  spar_fake=spar)
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["mode"], "run-fail")

    def test_scoped_edition_is_used_for_its_revision(self):
        rec = self.write_test(
            "//@ revisions: a b\n//@[a] run-pass\n//@[b] run-pass\n"
            "//@[b] edition: 2018\nfn main() {}\n")
        res = self.run_with_fakes(rec)
        self.assertEqual(res["status"], "pass")
        editions = {s["rev"]: s["edition"] for s in self.native_seen}
        self.assertEqual(editions, {"a": "2015", "b": "2018"})

    def test_scoped_aux_resolves_for_its_revision(self):
        (self.tmp / "tests" / "ui" / "auxiliary").mkdir()
        (self.tmp / "tests" / "ui" / "auxiliary"
         / "helper.rs").write_text("pub fn f() {}\n")
        rec = self.write_test(
            "//@ revisions: a b\n//@[a] run-pass\n//@[b] run-pass\n"
            "//@[a] aux-build: helper.rs\nfn main() {}\n")
        rec["single_file"] = False
        res = self.run_with_fakes(rec)
        self.assertEqual(res["status"], "pass")
        aux = {s["rev"]: s["aux"] for s in self.native_seen}
        self.assertEqual(aux, {"a": ["helper"], "b": []})


class ResolveAuxSpecsRevisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile")
                                         .TemporaryDirectory()))
        self.dir = self.tmp / "tests" / "ui"
        (self.dir / "auxiliary").mkdir(parents=True)
class ProcMacroExternTests(unittest.TestCase):
    """Stage 2 reproduces compiletest's `//@ proc-macro:` handling:
    the named auxiliary file is built as a proc-macro crate and
    linked into the conversion with --extern."""

    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile")
                                          .TemporaryDirectory()))
        self.dir = self.tmp / "tests" / "ui"
        (self.dir / "auxiliary").mkdir(parents=True)
        self.orig_build = run.build_proc_macro

    def tearDown(self):
        run.build_proc_macro = self.orig_build

    def write(self, rel, body):
        p = self.dir / rel
        p.write_text(body)
        return p

    def test_scoped_aux_directive_resolves_per_revision(self):
        main = self.write(
            "t.rs",
            "//@ revisions: a b\n//@[a] aux-build: helper.rs\n")
        self.write("auxiliary/helper.rs", "pub fn f() {}\n")
        specs, err = run.resolve_aux_specs(main, "a")
        self.assertIsNone(err)
        self.assertEqual([spec.name for spec in specs], ["helper"])
        specs, err = run.resolve_aux_specs(main, "b")
        self.assertIsNone(err)
        self.assertEqual(specs, [])
        # Without a revision, scoped directives do not resolve.
        specs, err = run.resolve_aux_specs(main)
        self.assertIsNone(err)
        self.assertEqual(specs, [])
    def test_top_view_supplies_scoped_directives(self):
        main = self.write(
            "t.rs", "//@ revisions: a\n//@[a] aux-build: helper.rs\n")
        self.write("auxiliary/helper.rs", "pub fn f() {}\n")
        header = run.Header(main.read_text())
        specs, err = run.resolve_aux_specs(main)
        self.assertIsNone(err)
        self.assertEqual(specs, [])
        specs, err = run.resolve_aux_specs(main, top=header.view("a"))
        self.assertIsNone(err)
        self.assertEqual([spec.name for spec in specs], ["helper"])
    def stub_build(self, result):
        calls = []

        def fake(test_path, fname, edition, extra_externs=(),
                 modifiers=None):
            calls.append((test_path, fname, edition,
                          list(extra_externs)))
            return result

        run.build_proc_macro = fake
        return calls

    def test_no_directive_means_no_extern_args(self):
        run.build_proc_macro = lambda *a: (_ for _ in ()).throw(
            AssertionError("build must not be called"))
        view = run.Header("//@ check-pass\nfn main() {}\n").view(None)
        args, err = run.proc_macro_extern_args(
            self.dir / "t.rs", view, "2021")
        self.assertEqual(args, [])
        self.assertIsNone(err)

    def test_extern_args_from_built_crate(self):
        artifact = Path("/cache/libtest_macros.so")
        calls = self.stub_build(("test_macros", artifact, None))
        view = run.Header(
            "//@ check-pass\n//@ proc-macro: test-macros.rs\n").view(None)
        args, err = run.proc_macro_extern_args(
            self.dir / "t.rs", view, "2021")
        self.assertIsNone(err)
        self.assertEqual(args, ["--extern", f"test_macros={artifact}",
                                "-L", "dependency=/cache"])
        self.assertEqual([c[1] for c in calls], ["test-macros.rs"])

    def test_build_failure_is_a_not_attempted_reason(self):
        self.stub_build((None, None, "some rustc error"))
        view = run.Header(
            "//@ check-pass\n//@ proc-macro: test-macros.rs\n").view(None)
        args, err = run.proc_macro_extern_args(
            self.dir / "t.rs", view, "2021")
        self.assertEqual(args, [])
        self.assertIn("could not be built", err)
        self.assertIn("some rustc error", err)

    def test_revision_scoped_directive_only_applies_to_its_revision(self):
        self.stub_build(("test_macros", Path("/c/libtest_macros.so"),
                         None))
        h = run.Header("//@ revisions: a b\n"
                       "//@[a] proc-macro: test-macros.rs\n")
        args_b, err_b = run.proc_macro_extern_args(
            self.dir / "t.rs", h.view("b"), "2021")
        self.assertEqual(args_b, [])
        self.assertIsNone(err_b)
        args_a, err_a = run.proc_macro_extern_args(
            self.dir / "t.rs", h.view("a"), "2021")
        self.assertIsNone(err_a)
        self.assertIn("--extern", args_a)

    def test_chain_derives_crate_name_from_file_stem(self):
        main = self.write("t.rs", "//@ check-pass\n")
        self.write("auxiliary/test-macros.rs",
                   "extern crate proc_macro;\n")
        chain, err = run._proc_macro_chain(main, "test-macros.rs",
                                           "2021")
        self.assertIsNone(err)
        self.assertEqual([(n, f) for n, _, f, _mod in chain],
                         [("test_macros", "proc-macro")])

    def test_chain_uses_declared_crate_name(self):
        main = self.write("t.rs", "//@ check-pass\n")
        self.write("auxiliary/weird-file.rs",
                   "#![crate_name = \"custom_name\"]\n")
        chain, err = run._proc_macro_chain(main, "weird-file.rs",
                                           "2021")
        self.assertIsNone(err)
        self.assertEqual(chain[-1][0], "custom_name")

    def test_chain_missing_file_is_an_error(self):
        main = self.write("t.rs", "//@ check-pass\n")
        chain, err = run._proc_macro_chain(main, "nope.rs", "2021")
        self.assertIsNone(chain)
        self.assertIn("not found", err)

    def test_chain_builds_aux_deps_before_the_proc_macro(self):
        main = self.write("t.rs", "//@ check-pass\n")
        self.write("auxiliary/pm.rs",
                   "//@ aux-build: dep.rs\nextern crate proc_macro;\n")
        self.write("auxiliary/dep.rs", "pub fn f() {}\n")
        chain, err = run._proc_macro_chain(main, "pm.rs", "2021")
        self.assertIsNone(err)
        self.assertEqual([(n, f) for n, _, f, _mod in chain],
                         [("dep", None), ("pm", "proc-macro")])

    def test_chain_includes_the_tests_own_aux_builds(self):
        # compiletest builds a test's auxiliary crates before its
        # proc-macro crates into one shared directory, so a
        # proc-macro crate can name a crate only the TEST's header
        # declares (meta-macro-hygiene: meta-macro.rs's
        # `extern crate make_macro;` answers the test's
        # `//@ aux-build:make-macro.rs`, a directive meta-macro.rs
        # itself never carries). Without the test's specs in the
        # chain the proc-macro build fails with E0463.
        main = self.write(
            "t.rs",
            "//@ aux-build: make-macro.rs\n"
            "//@ proc-macro: meta-macro.rs\n")
        self.write("auxiliary/make-macro.rs", "pub fn f() {}\n")
        self.write("auxiliary/meta-macro.rs",
                   "extern crate make_macro;\n"
                   "extern crate proc_macro;\n")
        chain, err = run._proc_macro_chain(main, "meta-macro.rs",
                                           "2021")
        self.assertIsNone(err)
        self.assertEqual([(n, f) for n, _, f, _mod in chain],
                         [("make_macro", None),
                          ("meta_macro", "proc-macro")])

    def test_chain_excludes_aux_crates_that_consume_the_proc_macro(self):
        # A test auxiliary crate that names the proc-macro crate
        # is its consumer, not its input: it can only build after
        # the proc-macro exists (token-site-span in
        # mixed-site-span declares its own
        # `//@ proc-macro: mixed-site-span.rs` and
        # `extern crate mixed_site_span;`). It is no part of the
        # proc-macro's chain; the conversion builds it through
        # the ordinary --aux path.
        main = self.write(
            "t.rs",
            "//@ aux-build: token.rs\n"
            "//@ proc-macro: pm.rs\n")
        self.write("auxiliary/token.rs",
                   "extern crate pm;\npub fn f() {}\n")
        self.write("auxiliary/pm.rs",
                   "extern crate proc_macro;\n")
        chain, err = run._proc_macro_chain(main, "pm.rs", "2021")
        self.assertIsNone(err)
        self.assertEqual([(n, f) for n, _, f, _mod in chain],
                         [("pm", "proc-macro")])

    def test_chain_build_passes_unstable_options_for_renamed_extern(self):
        # A modifier-prefixed binding (`priv:priv_dep=...`) is
        # the renamed --extern syntax, which rustc only accepts
        # with -Zunstable-options; a chain build that receives
        # such a binding must carry the flag (pub-priv1's pm
        # build failed without it).
        dep = self.write("auxiliary/priv_dep.rs", "pub fn f() {}\n")
        pm = self.write("auxiliary/pm.rs",
                        "extern crate proc_macro;\n")
        chain = [("priv_dep", dep, None, "priv"),
                 ("pm", pm, "proc-macro", None)]
        outdir = self.tmp / "out"
        outdir.mkdir()
        seen = []

        def fake_run_cmd(argv, **kw):
            seen.append(list(argv))
            (outdir / "libpriv_dep.rlib").write_bytes(b"")
            (outdir / "libpm.so").write_bytes(b"")
            return {"timeout": False, "exit_code": 0, "stderr": ""}

        with unittest.mock.patch.object(run, "run_cmd",
                                        fake_run_cmd):
            name, artifact, err = run._build_proc_macro_chain(
                chain, "2021", outdir)
        self.assertIsNone(err)
        self.assertEqual(name, "pm")
        self.assertEqual(len(seen), 2)
        pm_argv = seen[1]
        self.assertIn("--extern", pm_argv)
        self.assertTrue(any(a.startswith("priv:priv_dep=")
                            for a in pm_argv))
        self.assertIn("-Zunstable-options", pm_argv)

    def test_later_directive_build_sees_earlier_crate(self):
        seen_extra = []

        def fake(test_path, fname, edition, extra_externs=(),
                 modifiers=None):
            seen_extra.append((fname, list(extra_externs)))
            stem = fname[:-3].replace("-", "_")
            return (stem, Path(f"/c/lib{stem}.so"), None)

        run.build_proc_macro = fake
        view = run.Header(
            "//@ check-pass\n//@ proc-macro: first-pm.rs\n"
            "//@ proc-macro: second-pm.rs\n").view(None)
        args, err = run.proc_macro_extern_args(
            self.dir / "t.rs", view, "2021")
        self.assertIsNone(err)
        self.assertEqual(seen_extra[0], ("first-pm.rs", []))
        self.assertEqual(seen_extra[1][0], "second-pm.rs")
        self.assertEqual(seen_extra[1][1],
                         [("first_pm", Path("/c/libfirst_pm.so"))])
        self.assertEqual(
            args,
            ["--extern", "first_pm=/c/libfirst_pm.so",
             "-L", "dependency=/c",
             "--extern", "second_pm=/c/libsecond_pm.so"])

    def test_concrete_edition_of_range_and_plain(self):
        self.assertEqual(run._concrete_edition("2015..2021"), "2015")
        self.assertEqual(run._concrete_edition("2021"), "2021")
        self.assertEqual(run._concrete_edition("2018"), "2018")

    def test_stage2_attempt_forwards_extern_to_convert_side(self):
        rec_path = self.write(
            "t.rs",
            "//@ check-pass\n//@ proc-macro: test-macros.rs\n"
            "fn main() {}\n")
        rec = {"suite": "ui", "path": "tests/ui/t.rs",
               "nature": "compile-check", "single_file": True,
               "ffi": False, "env_excluded": False}
        self.stub_build(("test_macros", Path("/c/libtest_macros.so"),
                         None))
        seen = {}
        orig = run.convert_side

        def capture(path, edition, driver_bin, rustc_args=(),
                    aux_specs=(), env_extra=None, env_remove=(),
                    jit=False, aux_env_scrub=(), capture_stdout=False,
                    aux_libs=(), check_mode=False):
            seen["args"] = list(rustc_args)
            return conv()

        run.convert_side = capture
        try:
            res = run.process_test(rec, self.tmp, Path("/fake/driver"),
                                   stage2=True)
        finally:
            run.convert_side = orig
        self.assertEqual(res["status"], "pass")
        self.assertIn("--extern", seen["args"])
        self.assertIn("test_macros=/c/libtest_macros.so", seen["args"])
        self.assertEqual(rec_path.name, "t.rs")

    def test_stage2_attempt_adds_auto_check_cfg_without_revision(self):
        # compiletest appends `--check-cfg cfg(test,FALSE)` to
        # every run, revision or not (compose_and_run_compiler);
        # it is what activates the session's unexpected-cfg
        # checking, which check-fail expectations like
        # cfg/suggest-alternative-name-on-target (a denied
        # unexpected_cfgs lint on well-known cfg values) rest on
        # (issue #509). A revision run receives the argument from
        # revision_args; a plain run must not lose it, and
        # no-auto-check-cfg opts out.
        rec = {"suite": "ui", "path": "tests/ui/t.rs",
               "nature": "compile-check", "single_file": True,
               "ffi": False, "env_excluded": False}
        seen = {}
        orig = run.convert_side

        def capture(path, edition, driver_bin, rustc_args=(),
                    aux_specs=(), env_extra=None, env_remove=(),
                    jit=False, aux_env_scrub=(), capture_stdout=False,
                    aux_libs=(), check_mode=False):
            seen["args"] = list(rustc_args)
            return conv(exit_code=1, compilation_failed=True)

        run.convert_side = capture
        try:
            self.write(
                "t.rs",
                "//@ check-fail\n#![deny(unexpected_cfgs)]\n"
                "#[cfg(target_abi = \"arm\")]\nstruct A;\n"
                "fn main() {}\n")
            res = run.process_test(rec, self.tmp, Path("/fake/driver"),
                                   stage2=True)
            self.assertEqual(res["status"], "pass")
            self.assertIn("--check-cfg", seen["args"])
            self.assertEqual(
                seen["args"][seen["args"].index("--check-cfg") + 1],
                "cfg(test,FALSE)")
            self.write(
                "t.rs",
                "//@ check-fail\n//@ no-auto-check-cfg\n"
                "fn main() {}\n")
            res = run.process_test(rec, self.tmp, Path("/fake/driver"),
                                   stage2=True)
            self.assertEqual(res["status"], "pass")
            self.assertNotIn("--check-cfg", seen["args"])
        finally:
            run.convert_side = orig

    def test_stage2_attempt_passes_check_mode_for_check_views(
            self):
        # A check-pass view is a compiletest check compilation:
        # stage 2 hands the driver --compiletest-check-mode via
        # convert_side's check_mode (issue #509). A build-fail
        # view is a full build and does not get it.
        rec = {"suite": "ui", "path": "tests/ui/t.rs",
               "nature": "compile-check", "single_file": True,
               "ffi": False, "env_excluded": False}
        seen = {}
        orig = run.convert_side

        def capture(path, edition, driver_bin, rustc_args=(),
                    aux_specs=(), env_extra=None, env_remove=(),
                    jit=False, aux_env_scrub=(), capture_stdout=False,
                    aux_libs=(), check_mode=False):
            seen["check_mode"] = check_mode
            if check_mode:
                return conv()
            return conv(exit_code=1, compilation_failed=True)

        run.convert_side = capture
        try:
            self.write("t.rs", "//@ check-pass\nfn main() {}\n")
            res = run.process_test(rec, self.tmp, Path("/fake/driver"),
                                   stage2=True)
            self.assertEqual(res["status"], "pass")
            self.assertTrue(seen["check_mode"])
            self.write("t.rs", "//@ build-fail\nfn main() {}\n")
            res = run.process_test(rec, self.tmp, Path("/fake/driver"),
                                   stage2=True)
            self.assertEqual(res["status"], "pass")
            self.assertFalse(seen["check_mode"])
        finally:
            run.convert_side = orig

    def test_stage2_attempt_not_attempted_when_proc_macro_unbuildable(
            self):
        self.write("t.rs",
                   "//@ check-pass\n//@ proc-macro: test-macros.rs\n"
                   "fn main() {}\n")
        rec = {"suite": "ui", "path": "tests/ui/t.rs",
               "nature": "compile-check", "single_file": True,
               "ffi": False, "env_excluded": False}
        self.stub_build((None, None, "kaboom"))
        res = run.process_test(rec, self.tmp, Path("/fake/driver"),
                               stage2=True)
        self.assertEqual(res["status"], "not-attempted")
        self.assertIn("auxiliary crate build failure", res["bucket"])
        self.assertIn("kaboom", res["reason"])


class EnvDirectiveTests(unittest.TestCase):
    """The env / run-flags header helpers (compiletest semantics)."""

    def test_parse_env_pairs_splits_at_first_equals(self):
        # compiletest's parse_env: the name is trimmed, the value
        # is not, and a missing `=` means an empty value.
        self.assertEqual(
            run.parse_env_pairs(["A=1", "B=x=y", "C", " D = 2"]),
            [("A", "1"), ("B", "x=y"), ("C", ""), ("D", " 2")])

    def test_expand_variables(self):
        out = run.expand_variables(
            "{{src-base}}/a {{cwd}} {{build-base}} {{target}} "
            "{{target-linker}} {{unknown}}", "/src", "/build")
        self.assertEqual(
            out, "/src/a " + os.getcwd() + " /build "
            + run.HOST_TARGET + "  {{unknown}}")

    def test_view_env_pairs_unsets_and_expansion(self):
        h = header("//@ exec-env: FOO={{src-base}}/x\n"
                   "//@ exec-env: BAR=1\n"
                   "//@ unset-exec-env: BAZ\n")
        pairs, unset = run.view_env(h.view(None), "exec-env",
                                    "unset-exec-env", "/src", "/b")
        self.assertEqual(pairs, {"FOO": "/src/x", "BAR": "1"})
        self.assertEqual(unset, ["BAZ"])

    def test_view_env_is_revision_scoped(self):
        h = header("//@ revisions: a b\n//@[a] rustc-env: X=1\n")
        pairs_a, _ = run.view_env(h.view("a"), "rustc-env",
                                  "unset-rustc-env", "/s", "/b")
        pairs_b, _ = run.view_env(h.view("b"), "rustc-env",
                                  "unset-rustc-env", "/s", "/b")
        self.assertEqual(pairs_a, {"X": "1"})
        self.assertEqual(pairs_b, {})

    def test_view_run_flags_split_and_expand(self):
        h = header("//@ run-flags: --test-threads=1 --opt "
                   "{{src-base}}/f\n")
        self.assertEqual(run.view_run_flags(h.view(None), "/src",
                                            "/b"),
                         ["--test-threads=1", "--opt", "/src/f"])

    def test_run_flags_and_exec_env_do_not_block_stage1(self):
        h = header("//@ run-pass\n//@ run-flags: --foo\n"
                   "//@ exec-env: A=1\n")
        self.assertIsNone(run.stage1_view_blocker(h.view(None)))


class AuxModifierTests(unittest.TestCase):
    """--extern modifiers (priv:, noprelude:) in aux directives."""

    def test_aux_entries_carry_modifiers(self):
        h = header("//@ aux-crate:priv:priv_dep=priv_dep.rs\n"
                   "//@ proc-macro:priv:pm.rs\n")
        entries = {(n, f): (t, m)
                   for n, f, t, m in run.aux_entries(h)}
        self.assertEqual(entries[("priv_dep", "priv_dep.rs")],
                         (None, "priv"))
        self.assertEqual(entries[("pm", "pm.rs")],
                         ("proc-macro", "priv"))

    def test_aux_spec_arg_includes_modifiers(self):
        spec = run.AuxSpec("priv_dep", "/x/priv_dep.rs", None, None,
                           "priv")
        self.assertEqual(run.aux_spec_arg(spec),
                         "priv:priv_dep=/x/priv_dep.rs")
        spec = run.AuxSpec("plain", "/x/plain.rs")
        self.assertEqual(run.aux_spec_arg(spec), "plain=/x/plain.rs")

    def test_parse_proc_macro_value(self):
        self.assertEqual(run.parse_proc_macro_value("priv:pm.rs"),
                         ("priv", "pm.rs"))
        self.assertEqual(run.parse_proc_macro_value("pm.rs"),
                         (None, "pm.rs"))


class GuestArgForwardingTests(unittest.TestCase):
    """sparagmos_side forwards run-flags and exec-env to the driver."""

    def test_guest_args_and_env_reach_run_cmd(self):
        captured = {}
        orig = run.run_cmd

        def stub(argv, *, timeout, cwd=None, env_extra=None,
                 env_remove=(), mem_limit_mb=None):
            captured["argv"] = list(argv)
            captured["env_extra"] = env_extra
            captured["env_remove"] = env_remove
            return {"exit_code": 0, "stdout": "", "stderr": "",
                    "timeout": False}

        run.run_cmd = stub
        try:
            got, kind, _ = run.sparagmos_side(
                Path("t.rs"), "2021", Path("/fake/driver"),
                guest_args=("--test-threads=1", "--list"),
                env_extra={"EXAMPLE_ENV": "1"},
                env_remove=("RUST_BACKTRACE",))
        finally:
            run.run_cmd = orig
        self.assertEqual(kind, "result")
        argv = captured["argv"]
        i = argv.index("--guest-arg")
        self.assertEqual(argv[i + 1:i + 4],
                         ["--test-threads=1", "--guest-arg", "--list"])
        self.assertEqual(captured["env_extra"]["EXAMPLE_ENV"], "1")
        self.assertEqual(captured["env_extra"]["RUSTC_BOOTSTRAP"],
                         "1")
        self.assertEqual(captured["env_remove"],
                         ("RUST_BACKTRACE",))


class AttemptEditionTests(unittest.TestCase):
    """attempt_edition: the one edition selection every attempt
    path uses, ranges collapsed before they can reach rustc."""

    def test_range_collapses_to_its_lower_bound(self):
        h = header("//@ edition:2015..2021\n")
        self.assertEqual(run.attempt_edition(h.view(None)), "2015")

    def test_open_range_collapses_to_its_first_edition(self):
        h = header("//@ edition:2018..\n")
        self.assertEqual(run.attempt_edition(h.view(None)), "2018")

    def test_plain_edition_unchanged(self):
        h = header("//@ edition:2015\n")
        self.assertEqual(run.attempt_edition(h.view(None)), "2015")

    def test_no_directive_defaults_to_2015(self):
        h = header("fn main() {}\n")
        self.assertEqual(run.attempt_edition(h.view(None)), "2015")

    def test_last_directive_wins(self):
        h = header("//@ edition:2015\n//@ edition:2018..2024\n")
        self.assertEqual(run.attempt_edition(h.view(None)), "2018")

    def test_stage1_flat_path_uses_the_collapsed_edition(self):
        # Regression: the flat stage-1 path used to hand the
        # verbatim range string (`2015..2021`) to both sides,
        # which rustc rejects as an edition value.
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (tmp / "tests" / "ui").mkdir(parents=True)
        p = tmp / "tests" / "ui" / "t.rs"
        p.write_text("//@ run-pass\n//@ edition:2015..2021\n"
                     "fn main() {}\n")
        rec = {"suite": "ui", "path": "tests/ui/t.rs",
               "nature": "exec", "single_file": True, "ffi": False,
               "env_excluded": False}
        seen = {}
        orig_n, orig_s = run.native_side, run.sparagmos_side

        def native(path, edition, *a, **kw):
            seen["native"] = edition
            return ({"exit_code": 0, "stdout": "", "stderr": ""}, None)

        def sparagmos(path, edition, *a, **kw):
            seen["sparagmos"] = edition
            return ({"exit_code": 0, "stdout": "", "stderr": ""},
                    "result", "")

        run.native_side, run.sparagmos_side = native, sparagmos
        try:
            res = run.process_test(rec, tmp, Path("/fake/driver"))
        finally:
            run.native_side, run.sparagmos_side = orig_n, orig_s
        self.assertEqual(res["status"], "pass")
        self.assertEqual(seen, {"native": "2015", "sparagmos": "2015"})


class CompiletestModeArgsTests(unittest.TestCase):
    """compiletest_mode_args: the arguments compiletest's mode
    layer adds, which the subset must supply itself."""

    def test_ui_tests_get_one_codegen_unit(self):
        self.assertEqual(run.compiletest_mode_args("ui", []),
                         ["-Ccodegen-units=1"])
        self.assertEqual(run.compiletest_mode_args("ui-fulldeps", []),
                         ["-Ccodegen-units=1"])

    def test_own_codegen_units_setting_wins(self):
        for flags in (["-Ccodegen-units=16"], ["-C", "codegen-units=16"],
                      ["-Ccodegen-units", "16"]):
            self.assertEqual(run.compiletest_mode_args("ui", flags), [])

    def test_cfi_with_lto_gets_one_codegen_unit(self):
        flags = ["-Clto", "-Zsanitizer=cfi", "-Copt-level=0"]
        self.assertEqual(
            run.compiletest_mode_args("codegen-llvm", flags),
            ["-Ccodegen-units=1"])

    def test_cfi_split_token_spelling(self):
        flags = ["-C", "lto=fat", "-Z", "sanitizer=cfi"]
        self.assertEqual(
            run.compiletest_mode_args("codegen-llvm", flags),
            ["-Ccodegen-units=1"])

    def test_cfi_without_lto_needs_nothing(self):
        self.assertEqual(run.compiletest_mode_args(
            "codegen-llvm", ["-Zsanitizer=cfi"]), [])

    def test_other_sanitizer_with_lto_needs_nothing(self):
        self.assertEqual(run.compiletest_mode_args(
            "codegen-llvm", ["-Clto", "-Zsanitizer=address"]), [])

    def test_unrelated_suite_and_flags(self):
        self.assertEqual(run.compiletest_mode_args("mir-opt", []), [])
        self.assertEqual(run.compiletest_mode_args("codegen-llvm", []),
                         [])


class SysrootExternArgsTests(unittest.TestCase):
    """sysroot_extern_args over a stubbed ambiguity map."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        orig = run.ambiguous_sysroot_crates
        run.ambiguous_sysroot_crates = lambda: {
            "libc": "/sysroot/lib/liblibc-7d5a06a231119d28.rmeta"}
        self.addCleanup(setattr, run, "ambiguous_sysroot_crates", orig)

    def write(self, text):
        p = self.tmp / "t.rs"
        p.write_text(text)
        return p

    def test_declared_ambiguous_crate_is_pinned(self):
        p = self.write("extern crate libc;\nfn main() {}\n")
        self.assertEqual(
            run.sysroot_extern_args(p, []),
            ["--extern",
             "libc=/sysroot/lib/liblibc-7d5a06a231119d28.rmeta"])

    def test_undeclared_crate_is_not_pinned(self):
        p = self.write("fn main() {}\n")
        self.assertEqual(run.sysroot_extern_args(p, []), [])

    def test_foreign_target_is_not_pinned(self):
        p = self.write("extern crate libc;\nfn main() {}\n")
        for flags in (["--target", "aarch64-unknown-linux-gnu"],
                      ["--target=aarch64-unknown-linux-gnu"]):
            self.assertEqual(run.sysroot_extern_args(p, flags), [])

    def test_host_target_still_pinned(self):
        p = self.write("extern crate libc;\nfn main() {}\n")
        args = run.sysroot_extern_args(
            p, ["--target", "x86_64-unknown-linux-gnu"])
        self.assertEqual(len(args), 2)

    def test_flag_bound_name_is_left_to_the_flags(self):
        p = self.write("extern crate libc;\nfn main() {}\n")
        for flags in (["--extern", "libc=/elsewhere.rmeta"],
                      ["--extern=libc=/elsewhere.rmeta"]):
            self.assertEqual(run.sysroot_extern_args(p, flags), [])

    def test_missing_source_reads_as_no_declarations(self):
        self.assertEqual(
            run.sysroot_extern_args(self.tmp / "absent.rs", []), [])


class ProcessStage2RunModeTests(unittest.TestCase):
    """process_test(stage2=True) on views whose mode is a run
    mode: judged by execution, with the process sides stubbed."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.tmp / "tests" / "ui").mkdir(parents=True)
        self.rust_dir = self.tmp

    def write_test(self, body):
        p = self.rust_dir / "tests" / "ui" / "t.rs"
        p.write_text(body)
        return {"suite": "ui", "path": "tests/ui/t.rs",
                "nature": "compile-check", "single_file": True,
                "ffi": False, "env_excluded": False}

    def run_with_sides(self, rec, native, sparagmos, conv_result):
        orig_n = run.native_side
        orig_s = run.sparagmos_side
        orig_c = run.convert_side
        run.native_side = lambda *a, **kw: native
        run.sparagmos_side = lambda *a, **kw: sparagmos
        run.convert_side = lambda *a, **kw: conv_result
        try:
            return run.process_test(rec, self.rust_dir,
                                    Path("/fake/driver"), stage2=True)
        finally:
            run.native_side = orig_n
            run.sparagmos_side = orig_s
            run.convert_side = orig_c

    def test_run_pass_revision_is_judged_by_execution(self):
        rec = self.write_test(
            "//@ revisions: runtime borrowck\n"
            "//@ [runtime] run-pass\n"
            "//@ [borrowck] check-fail\n"
            "fn main() { println!(\"x\"); }\n")
        res = self.run_with_sides(
            rec,
            ({"exit_code": 0, "stdout": "x\n", "stderr": ""}, None),
            ({"exit_code": 0, "stdout": "x\n", "stderr": ""},
             "result", ""),
            conv(exit_code=1, compilation_failed=True,
                 converted=None, skipped=None))
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["revisions"]["runtime"]["status"], "pass")
        self.assertEqual(res["revisions"]["runtime"]["expectation"],
                         "run-pass")
        self.assertEqual(res["revisions"]["borrowck"]["status"], "pass")

    def test_run_fail_revision_mismatch_fails(self):
        rec = self.write_test(
            "//@ revisions: runtime\n"
            "//@ [runtime] run-fail\n"
            "fn main() { panic!(\"boom\"); }\n")
        res = self.run_with_sides(
            rec,
            ({"exit_code": 101, "stdout": "", "stderr": ""}, None),
            ({"exit_code": 0, "stdout": "", "stderr": ""},
             "result", ""),
            conv())
        self.assertEqual(res["status"], "fail")
        self.assertEqual(res["bucket"], "exit-code mismatch")

    def test_run_crash_revision_is_judged_by_execution(self):
        rec = self.write_test(
            "//@ revisions: chk rt\n"
            "//@ [chk] check-pass\n"
            "//@ [rt] run-crash\n"
            "fn main() {}\n")
        res = self.run_with_sides(
            rec,
            ({"exit_code": -6, "stdout": "", "stderr": ""}, None),
            ({"exit_code": 134, "stdout": "", "stderr": "",
              "driver_stderr":
                  "sparagmos-driver run: guest crashed: abort\n"},
             "result", ""),
            conv())
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["revisions"]["rt"]["status"], "pass")
        self.assertEqual(res["revisions"]["rt"]["expectation"],
                         "run-crash")
        self.assertEqual(res["revisions"]["chk"]["status"], "pass")

    def test_run_crash_revision_guest_unwind_fails(self):
        rec = self.write_test(
            "//@ revisions: rt\n"
            "//@ [rt] run-crash\n"
            "fn main() {}\n")
        res = self.run_with_sides(
            rec,
            ({"exit_code": -6, "stdout": "", "stderr": ""}, None),
            ({"exit_code": 101, "stdout": "", "stderr": ""},
             "result", ""),
            conv())
        self.assertEqual(res["status"], "fail")
        self.assertEqual(res["bucket"], "guest did not crash")

    def test_run_crash_revision_native_not_crashing_unjudged(self):
        # The oracle itself does not reproduce the test's premise
        # (the native run does not crash), so the record cannot
        # be judged: it stays not-attempted under the native
        # expectation bucket, as a native mismatch does for the
        # fixed-code modes.
        rec = self.write_test(
            "//@ revisions: rt\n"
            "//@ [rt] run-crash\n"
            "fn main() {}\n")
        res = self.run_with_sides(
            rec,
            ({"exit_code": 0, "stdout": "", "stderr": ""}, None),
            ({"exit_code": 134, "stdout": "", "stderr": ""},
             "result", ""),
            conv())
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(
            res["bucket"],
            "native run does not match mode expectation")

    def test_unscoped_run_mode_is_judged_by_execution(self):
        rec = self.write_test("//@ run-pass\nfn main() {}\n")
        res = self.run_with_sides(
            rec,
            ({"exit_code": 0, "stdout": "", "stderr": ""}, None),
            ({"exit_code": 0, "stdout": "", "stderr": ""},
             "result", ""),
            conv())
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["expectation"], "run-pass")


class DebuggerHarnessTests(unittest.TestCase):
    """The exec-debugger harness: section parsing, the check
    matchers (ports of compiletest's debugger.rs), command
    translation, and per-variant skipping."""

    SAMPLE = (
        "//@ compile-flags:-g\n"
        "//@ gdb-command:run\n"
        "//@ gdb-command:print x\n"
        "//@ gdb-check:$1 = 5\n"
        "//@ lldb-command:run\n"
        "//@ lldb-command:v x\n"
        "//@ lldb-check:[...] 5\n"
        "fn main() {\n"
        "    let x = 5;\n"
        "    zzz(); // #break\n"
        "}\n"
        "fn zzz() {()}\n")

    def test_sections_parse_commands_checks_and_breaks(self):
        bp, cmds, checks = run.parse_debugger_sections(
            self.SAMPLE, "gdb", None)
        self.assertEqual(bp, [10])
        self.assertEqual(cmds, ["run", "print x"])
        self.assertEqual(checks, [(4, "$1 = 5")])
        bp, cmds, checks = run.parse_debugger_sections(
            self.SAMPLE, "lldb", None)
        self.assertEqual(bp, [10])
        self.assertEqual(cmds, ["run", "v x"])
        self.assertEqual(checks, [(7, "[...] 5")])

    def test_break_marker_wins_over_directive_shape(self):
        # A line containing #break is a breakpoint line even
        # when the rest of the line looks like a directive.
        text = "//@ gdb-command:run\nfn f() {} // #break\n"
        bp, cmds, _ = run.parse_debugger_sections(text, "gdb", None)
        self.assertEqual(bp, [2])
        self.assertEqual(cmds, ["run"])

    def test_repr_becomes_command_and_check(self):
        text = ("//@ gdb-command:run\n"
                "//@ gdb-repr:thing\n"
                "fn main() {}\n")
        _, cmds, checks = run.parse_debugger_sections(text, "gdb", None)
        self.assertEqual(cmds, ["run", "repr thing"])
        self.assertEqual(checks, [(2, "thing: Ok")])

    def test_revision_scoping(self):
        text = ("//@ revisions: a b\n"
                "//@ [a] gdb-command:run\n"
                "//@ [b] gdb-command:print y\n"
                "fn main() {}\n")
        _, a_cmds, _ = run.parse_debugger_sections(text, "gdb", "a")
        _, b_cmds, _ = run.parse_debugger_sections(text, "gdb", "b")
        self.assertEqual(a_cmds, ["run"])
        self.assertEqual(b_cmds, ["print y"])

    def test_check_single_line_fragment_order(self):
        self.assertTrue(run.debugger_check_single_line(
            "noise $1 = 5", "$1 = 5"))
        self.assertFalse(run.debugger_check_single_line(
            "$1 = 5 trailing", "$1 = 5"))

    def test_check_single_line_trailing_wildcard(self):
        self.assertTrue(run.debugger_check_single_line(
            "(f64) $0 = 4.5", "[...] 4.5"))
        self.assertFalse(run.debugger_check_single_line(
            "(f64) $0 = 4.5 trailing", "[...] 4.5"))
        self.assertTrue(run.debugger_check_single_line(
            "xx a yy b", "a [...] b"))
        self.assertFalse(run.debugger_check_single_line(
            "xx a yy", "a [...] b"))

    def test_check_output_reuses_the_matched_line(self):
        # compiletest's check_output does not advance past a
        # matched line: two checks may match the same output
        # line.
        ok, missing = run.debugger_check_output(
            [(1, "$1 = 5"), (2, "$1 = 5")], ["$1 = 5"])
        self.assertTrue(ok)
        self.assertEqual(missing, [])

    def test_check_output_reports_missing_in_order(self):
        ok, missing = run.debugger_check_output(
            [(1, "beta"), (2, "gamma")], ["alpha", "beta"])
        self.assertFalse(ok)
        self.assertEqual(missing, [(2, "gamma")])

    def test_translate_gdb(self):
        self.assertEqual(
            run.translate_debugger_command("gdb", "print/d x"),
            "print/d x")
        self.assertEqual(
            run.translate_debugger_command("gdb", "print x"), "print x")
        self.assertEqual(
            run.translate_debugger_command("gdb", "continue"), "continue")
        self.assertEqual(
            run.translate_debugger_command("gdb", "frame"), "show-line")
        self.assertEqual(
            run.translate_debugger_command("gdb", "frame 2"), "frame 2")
        self.assertEqual(
            run.translate_debugger_command("gdb", "set print pretty"),
            "# inert: set print pretty")
        # The elements limit shapes printed output, so it is
        # forwarded to the session instead of going inert.
        self.assertEqual(
            run.translate_debugger_command("gdb", "set print elements 5"),
            "print-elements 5")
        self.assertEqual(
            run.translate_debugger_command(
                "gdb", "set print elements unlimited"),
            "print-elements unlimited")
        self.assertIsNone(run.translate_debugger_command("gdb", "quit"))
        self.assertEqual(
            run.translate_debugger_command("gdb", "ptype x"), "ptype x")

    def test_translate_gdb_d2_scope(self):
        # Statics by path, named fields, and casts print.
        self.assertEqual(
            run.translate_debugger_command(
                "gdb", "print simple_struct::NO_PADDING_16"),
            "print simple_struct::NO_PADDING_16")
        self.assertEqual(
            run.translate_debugger_command("gdb", "print stack_unique.value"),
            "print stack_unique.value")
        self.assertEqual(
            run.translate_debugger_command(
                "gdb", "print *(singleton.data_ptr as *const [i64; 1])"),
            "print *(singleton.data_ptr as *const [i64; 1])")
        # Type queries and the repr pseudo-command.
        self.assertEqual(
            run.translate_debugger_command("gdb", "whatis local"),
            "whatis local")
        self.assertEqual(
            run.translate_debugger_command("gdb", "repr b"), "repr b")
        # Counted steps repeat the step line.
        self.assertEqual(
            run.translate_debugger_command("gdb", "next 2"), "next\nnext")
        self.assertEqual(
            run.translate_debugger_command("gdb", "step 3"),
            "step\nstep\nstep")
        # Frame navigation.
        self.assertEqual(
            run.translate_debugger_command("gdb", "up"), "up")
        self.assertEqual(
            run.translate_debugger_command("gdb", "down"), "down")
        # Breakpoints from the command section.
        self.assertEqual(
            run.translate_debugger_command("gdb", "break 12"), "break 12")
        self.assertEqual(
            run.translate_debugger_command(
                "gdb", "break cross_crate_spans.rs:12"),
            "break-at cross_crate_spans.rs:12")
        self.assertEqual(
            run.translate_debugger_command(
                "gdb", "break constant_ordering_prologue::binding"),
            "break-fn constant_ordering_prologue::binding")
        # info listings.
        self.assertEqual(
            run.translate_debugger_command("gdb", "info functions"),
            "info-functions -")
        self.assertEqual(
            run.translate_debugger_command(
                "gdb", "info functions -q function_names::main"),
            "info-functions-q function_names::main")
        self.assertEqual(
            run.translate_debugger_command("gdb", "info locals"),
            "info-locals")
        # A run with guest arguments still starts the session.
        self.assertEqual(
            run.translate_debugger_command("gdb", "run 7"), "run")
        # A formatted print written with a space (`print /x`).
        self.assertEqual(
            run.translate_debugger_command(
                "gdb", "print /x *(_ptr as *const u64)"),
            "print/x *(_ptr as *const u64)")
        # Call expressions stay outside the scope.
        self.assertIsNone(
            run.translate_debugger_command("gdb", "print fun(45, true)"))
        # gdb printf whose arguments stay within the grammar
        # is in scope and reaches the session verbatim; an
        # argument outside the grammar keeps it out.
        self.assertEqual(
            run.translate_debugger_command(
                "gdb", 'printf "s = \\"%s\\"\\n", s'),
            'printf "s = \\"%s\\"\\n", s')
        self.assertIsNone(
            run.translate_debugger_command("gdb", 'printf "%d", fun(1)'))
        # sizeof of a value expression is a print expression.
        self.assertEqual(
            run.translate_debugger_command("gdb", "print sizeof(packed)"),
            "print sizeof(packed)")
        self.assertIsNone(
            run.translate_debugger_command("gdb", "print sizeof(fun(1))"))

    def test_translate_lldb(self):
        self.assertEqual(
            run.translate_debugger_command("lldb", "v x"), "v x")
        self.assertEqual(
            run.translate_debugger_command("lldb", "expr x"), "v x")
        self.assertEqual(
            run.translate_debugger_command("lldb", "expr sizeof(packed)"),
            "v sizeof(packed)")
        self.assertEqual(
            run.translate_debugger_command("lldb", "settings set a b"),
            "# inert: settings set a b")
        self.assertEqual(
            run.translate_debugger_command("lldb", "frame select"),
            "show-line")
        self.assertIsNone(
            run.translate_debugger_command("lldb", "breakpoint set -l 3"))

    def test_translate_lldb_d2_scope(self):
        # Statics by path.
        self.assertEqual(
            run.translate_debugger_command(
                "lldb", "v basic_types_globals::B"),
            "v basic_types_globals::B")
        # The `fr v` abbreviation of `frame variable`.
        self.assertEqual(
            run.translate_debugger_command("lldb", "fr v empty_string"),
            "v empty_string")
        # A format registration for a type outside the char
        # family stays a session-shaping comment.
        self.assertEqual(
            run.translate_debugger_command(
                "lldb", "type format add -f decimal i32"),
            "# inert: type format add -f decimal i32")
        # The repr pseudo-command and frame navigation.
        self.assertEqual(
            run.translate_debugger_command("lldb", "repr slice"),
            "repr slice")
        self.assertEqual(
            run.translate_debugger_command("lldb", "up"), "up")
        # A formatted print keeps its format: the session
        # script's `print/<fmt>` lines carry it to the
        # renderer (an unsigned print of a negative value
        # differs observably from the plain print).
        self.assertEqual(
            run.translate_debugger_command("lldb", "v/d noPadding8"),
            "print/d noPadding8")
        self.assertEqual(
            run.translate_debugger_command("lldb", "p/u b"),
            "print/u b")
        self.assertIsNone(
            run.translate_debugger_command("lldb", "v/x x"))
        # The `b` breakpoint abbreviation, by file:line and by
        # (quoted) function path.
        self.assertEqual(
            run.translate_debugger_command("lldb", "b cross_crate_spans.rs:12"),
            "break-at cross_crate_spans.rs:12")
        self.assertEqual(
            run.translate_debugger_command(
                "lldb", 'b "constant_ordering_prologue::binding"'),
            "break-fn constant_ordering_prologue::binding")
        # Char-family format registrations become type-format
        # script lines, one per named type.
        self.assertEqual(
            run.translate_debugger_command(
                "lldb", "type format add --format hex char"),
            "type-format hex char")
        self.assertEqual(
            run.translate_debugger_command(
                "lldb", 'type format add -f d i8 char "signed char"'),
            "type-format dec i8\ntype-format dec char\n"
            "type-format dec signed char")
        self.assertEqual(
            run.translate_debugger_command(
                "lldb", "type format add -f decimal char"),
            "type-format dec char")

    def test_variant_skipping(self):
        h = run.Header("//@ ignore-gdb\nfn main() {}\n")
        view = h.view(None)
        self.assertTrue(run.debugger_variant_skipped(view, "gdb"))
        self.assertFalse(run.debugger_variant_skipped(view, "lldb"))
        h2 = run.Header("//@ only-lldb\nfn main() {}\n")
        self.assertTrue(run.debugger_variant_skipped(h2.view(None), "gdb"))
        self.assertFalse(run.debugger_variant_skipped(h2.view(None), "lldb"))

    def test_filtered_view_hides_variant_directives(self):
        h = run.Header("//@ only-lldb\n//@ run-pass\n")
        view = h.view(None)
        self.assertTrue(view.has("only-lldb"))
        filtered = run._DebuggerFilteredView(view)
        self.assertFalse(filtered.has("only-lldb"))
        self.assertTrue(filtered.has("run-pass"))


class JudgeJitCompileTests(unittest.TestCase):
    """judge_jit_compile: metric 4's compile-check verdict."""

    def test_success_and_jit_compiled_passes(self):
        c = conv(jit_compiled=True, jit_reason="")
        self.assertEqual(run.judge_jit_compile("success", c),
                         ("pass", "", ""))

    def test_success_but_jit_unsupported_is_unsupported(self):
        c = conv(exit_code=2, jit_compiled=False,
                 jit_reason="function `t::main`: unsupported "
                            "rvalue `InlineAsm`")
        status, reason, bucket = run.judge_jit_compile("success", c)
        self.assertEqual(status, "unsupported")
        self.assertIn("JIT lowering unsupported", reason)
        self.assertIn("JIT lowering unsupported", bucket)

    def test_success_but_jit_unsupported_callee_bucket(self):
        c = conv(exit_code=2, jit_compiled=False,
                 jit_reason="call to function `std::rt::panic_fmt` "
                            "which is not part of the program")
        _s, _r, bucket = run.judge_jit_compile("success", c)
        self.assertEqual(
            bucket, "JIT lowering unsupported (callee not in program)")

    def test_success_incomplete_conversion_stays_unsupported(self):
        c = conv(exit_code=2, jit_compiled=False,
                 jit_reason="unsupported rvalue",
                 skipped=2, skip_totals=[("Rvalue::BinaryOp", 3)])
        status, reason, bucket = run.judge_jit_compile("success", c)
        self.assertEqual(status, "unsupported")
        self.assertIn("conversion incomplete", reason)
        self.assertNotIn("JIT", bucket)

    def test_success_exit0_without_jit_report_is_fail(self):
        # A pass may never be claimed without evidence the JIT
        # lowering actually happened.
        c = conv(jit_compiled=None, jit_reason="")
        status, _r, bucket = run.judge_jit_compile("success", c)
        self.assertEqual(status, "fail")
        self.assertEqual(bucket, "driver error")

    def test_success_exit2_without_jit_report_is_fail(self):
        c = conv(exit_code=2, jit_compiled=None, jit_reason="")
        status, _r, bucket = run.judge_jit_compile("success", c)
        self.assertEqual(status, "fail")
        self.assertEqual(bucket, "driver error")

    def test_failure_clean_rejection_passes_without_jit(self):
        c = conv(exit_code=1, compilation_failed=True,
                 converted=None, skipped=None,
                 jit_compiled=None, jit_reason="")
        self.assertEqual(run.judge_jit_compile("failure", c)[0],
                         "pass")

    def test_failure_but_converted_and_jit_unsupported_is_fail(self):
        # A JIT lowering failure is not a compilation rejection:
        # it must not fabricate a pass on a failure-expecting test.
        c = conv(exit_code=2, jit_compiled=False,
                 jit_reason="unsupported rvalue")
        status, _r, bucket = run.judge_jit_compile("failure", c)
        self.assertEqual(status, "fail")
        self.assertEqual(bucket,
                         "expected failure but conversion succeeded")

    def test_crash_reproduced_passes(self):
        c = conv(exit_code=101, ice=True, jit_compiled=None,
                 jit_reason="")
        self.assertEqual(run.judge_jit_compile("crash", c)[0], "pass")

    def test_crash_unevidenced_101_fails(self):
        # Metric 4 judges crashes through judge_compile, so the
        # same evidence rule applies: an exit 101 with no
        # internal compiler error in the output is the driver
        # own escaped panic, not a reproduced crash.
        c = conv(exit_code=101, jit_compiled=None, jit_reason="")
        status, _r, bucket = run.judge_jit_compile("crash", c)
        self.assertEqual(status, "fail")
        self.assertEqual(bucket, "compiler crash not evidenced")

    def test_crash_exit2_is_not_a_crash(self):
        # In jit-compile mode exit 2 is the defined JIT-unsupported
        # outcome; it must not be read as a compiler crash.
        c = conv(exit_code=2, jit_compiled=False,
                 jit_reason="unsupported rvalue")
        status, _r, _b = run.judge_jit_compile("crash", c)
        self.assertEqual(status, "fail")

    def test_timeout_is_fail(self):
        c = conv(exit_code=None, timeout=True,
                 jit_compiled=None, jit_reason="")
        self.assertEqual(run.judge_jit_compile("success", c)[0],
                         "fail")


class JitSideTests(unittest.TestCase):
    """The JIT driver argv shapes (run_cmd stubbed)."""

    def stub_run_cmd(self, result):
        orig = run.run_cmd
        seen = []

        def stub(argv, *, timeout, cwd=None, env_extra=None,
                 env_remove=(), mem_limit_mb=None):
            seen.append(argv)
            return result

        run.run_cmd = stub
        self.addCleanup(lambda: setattr(run, "run_cmd", orig))
        return seen

    def test_sparagmos_side_jit_uses_run_jit(self):
        seen = self.stub_run_cmd(
            {"exit_code": 0, "timeout": False,
             "stdout": "x\n", "stderr": ""})
        got, kind, _ = run.sparagmos_side(
            Path("t.rs"), "2021", Path("/fake/driver"), jit=True)
        self.assertEqual(seen[0][1], "run-jit")
        self.assertEqual(kind, "result")
        self.assertEqual(got["stdout"], "x\n")

    def test_sparagmos_side_default_uses_run(self):
        seen = self.stub_run_cmd(
            {"exit_code": 0, "timeout": False,
             "stdout": "", "stderr": ""})
        run.sparagmos_side(Path("t.rs"), "2021", Path("/fake/driver"))
        self.assertEqual(seen[0][1], "run")

    def test_sparagmos_side_jit_unsupported_reason(self):
        self.stub_run_cmd(
            {"exit_code": 2, "timeout": False, "stdout": "",
             "stderr": "sparagmos-driver run-jit: unsupported: "
                       "guest arguments are not supported by JIT "
                       "execution\n"})
        _got, kind, reason = run.sparagmos_side(
            Path("t.rs"), "2021", Path("/fake/driver"), jit=True)
        self.assertEqual(kind, "unsupported")
        self.assertIn("run-jit", reason)
        self.assertEqual(
            run.bucket_reason(reason),
            "JIT execution: guest arguments unsupported")

    def test_convert_side_jit_uses_jit_compile_and_parses(self):
        seen = self.stub_run_cmd(
            {"exit_code": 2, "timeout": False,
             "stdout": "converted: 3, skipped: 0\njit-compiled: no\n",
             "stderr": "sparagmos-driver jit-compile: unsupported: "
                       "unsupported rvalue `InlineAsm`\n"})
        out = run.convert_side(Path("t.rs"), "2021",
                               Path("/fake/driver"), jit=True)
        self.assertEqual(seen[0][1], "jit-compile")
        self.assertEqual(out["jit_compiled"], False)
        self.assertIn("InlineAsm", out["jit_reason"])
        self.assertEqual(out["converted"], 3)

    def test_convert_side_default_has_no_jit_line(self):
        seen = self.stub_run_cmd(
            {"exit_code": 0, "timeout": False,
             "stdout": "converted: 1, skipped: 0\n", "stderr": ""})
        out = run.convert_side(Path("t.rs"), "2021",
                               Path("/fake/driver"))
        self.assertEqual(seen[0][1], "t.rs")
        self.assertIsNone(out["jit_compiled"])


class CompiletestCheckModeTests(unittest.TestCase):
    """compiletest_check_mode: which views compiletest
    compiles with --emit=metadata (issue #509)."""

    def view(self, text):
        return run.Header(text).view(None)

    def test_check_pass_and_check_fail_are_check_mode(self):
        self.assertTrue(run.compiletest_check_mode(
            "ui", self.view("//@ check-pass\nfn main() {}\n")))
        self.assertTrue(run.compiletest_check_mode(
            "ui", self.view("//@ check-fail\nfn main() {}\n")))

    def test_directive_less_ui_view_is_check_mode(self):
        # compiletest's UI default is check-fail.
        self.assertTrue(run.compiletest_check_mode(
            "ui", self.view("fn main() {}\n")))

    def test_build_modes_are_not_check_mode(self):
        self.assertFalse(run.compiletest_check_mode(
            "ui", self.view("//@ build-pass\nfn main() {}\n")))
        self.assertFalse(run.compiletest_check_mode(
            "ui", self.view("//@ build-fail\nfn main() {}\n")))

    def test_run_modes_are_not_check_mode(self):
        self.assertFalse(run.compiletest_check_mode(
            "ui", self.view("//@ run-pass\nfn main() {}\n")))

    def test_incremental_cpass_is_check_mode(self):
        h = run.Header("//@ revisions: cpass1 bfail1\n"
                       "fn main() {}\n")
        self.assertTrue(run.compiletest_check_mode(
            "incremental", h.view("cpass1")))
        self.assertFalse(run.compiletest_check_mode(
            "incremental", h.view("bfail1")))

    def test_other_suites_are_not_check_mode(self):
        self.assertFalse(run.compiletest_check_mode(
            "mir-opt", self.view("//@ check-pass\nfn main() {}\n")))
        self.assertFalse(run.compiletest_check_mode(
            "crashes", self.view("fn main() {}\n")))

    def test_emit_mir_flag_leaves_check_mode(self):
        # The shape of tests/ui/limits/vtable-try-as-dyn.rs:
        # the test's own --emit=mir accumulates with
        # compiletest's --emit=metadata, and the reference
        # compilation evaluates the constant whose error the
        # test asserts -- a metadata-only compilation stays
        # silent on it.
        self.assertFalse(run.compiletest_check_mode(
            "ui", self.view("//@ check-fail\n"
                            "//@ compile-flags: --crate-type=lib --emit=mir\n"
                            "fn main() {}\n")))
        self.assertFalse(run.compiletest_check_mode(
            "ui", self.view("//@ compile-flags: --emit mir\n"
                            "fn main() {}\n")))

    def test_analysis_level_emits_stay_check_mode(self):
        self.assertTrue(run.compiletest_check_mode(
            "ui", self.view("//@ check-fail\n"
                            "//@ compile-flags: --emit=metadata\n"
                            "fn main() {}\n")))
        self.assertTrue(run.compiletest_check_mode(
            "ui", self.view("//@ check-fail\n"
                            "//@ compile-flags: --emit=dep-info\n"
                            "fn main() {}\n")))
        # An emit list naming an output path keeps its kind.
        self.assertFalse(run.compiletest_check_mode(
            "ui", self.view("//@ check-fail\n"
                            "//@ compile-flags: --emit=mir=out.mir\n"
                            "fn main() {}\n")))


class SparagmosSideExitCodeTests(unittest.TestCase):
    """sparagmos_side's exit-code contract: the guest's own exit
    code passes through as the result; only the driver's
    reserved outcomes are classified away."""

    def side(self, result):
        orig = run.run_cmd
        run.run_cmd = lambda argv, **kw: result
        self.addCleanup(setattr, run, "run_cmd", orig)
        return run.sparagmos_side(
            Path("t.rs"), "2021", Path("/fake/driver"))

    def test_guest_exit_1_is_a_result(self):
        # A `Termination` failure (main -> Result Err) exits 1
        # natively; the driver's 1 is the guest's code, not a
        # driver failure.
        got, kind, _ = self.side(
            {"exit_code": 1, "timeout": False,
             "stdout": "", "stderr": "Error: \"boom\"\n"})
        self.assertEqual(kind, "result")
        self.assertEqual(got["exit_code"], 1)

    def test_guest_exit_3_is_a_result(self):
        got, kind, _ = self.side(
            {"exit_code": 3, "timeout": False,
             "stdout": "before\n", "stderr": ""})
        self.assertEqual(kind, "result")
        self.assertEqual(got["exit_code"], 3)

    def test_driver_unsupported_stays_unsupported(self):
        _got, kind, reason = self.side(
            {"exit_code": 2, "timeout": False, "stdout": "",
             "stderr": "sparagmos-driver run: unsupported: x\n"})
        self.assertEqual(kind, "unsupported")
        self.assertIn("unsupported", reason)

    def test_signal_death_is_a_driver_error(self):
        _got, kind, reason = self.side(
            {"exit_code": -6, "timeout": False,
             "stdout": "", "stderr": ""})
        self.assertEqual(kind, "driver-error")
        self.assertIn("signal 6", reason)


class ProcessJitTests(unittest.TestCase):
    """process_test(jit=True) with both process sides stubbed."""

    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile")
                                         .TemporaryDirectory()))
        (self.tmp / "tests" / "ui").mkdir(parents=True)
        self.rust_dir = self.tmp

    def write_test(self, body, nature="exec"):
        p = self.rust_dir / "tests" / "ui" / "t.rs"
        p.write_text(body)
        return {"suite": "ui", "path": "tests/ui/t.rs",
                "nature": nature, "single_file": True,
                "ffi": False, "env_excluded": False}

    def test_exec_pass_forwards_jit_to_driver_side(self):
        rec = self.write_test("//@ run-pass\nfn main() {}\n")
        seen = {}
        orig_n, orig_s = run.native_side, run.sparagmos_side
        run.native_side = lambda *a, **kw: (
            {"exit_code": 0, "stdout": "x\n", "stderr": ""}, None)

        def fake_side(*a, **kw):
            seen["jit"] = kw.get("jit", False)
            return ({"exit_code": 0, "stdout": "x\n",
                     "stderr": ""}, "result", "")

        run.sparagmos_side = fake_side
        try:
            res = run.process_test(rec, self.rust_dir,
                                   Path("/fake/driver"), jit=True)
        finally:
            run.native_side, run.sparagmos_side = orig_n, orig_s
        self.assertEqual(res["status"], "pass")
        self.assertTrue(seen["jit"])
        self.assertTrue(res["jit"])

    def test_exec_without_jit_does_not_flag_record(self):
        rec = self.write_test("//@ run-pass\nfn main() {}\n")
        orig_n, orig_s = run.native_side, run.sparagmos_side
        run.native_side = lambda *a, **kw: (
            {"exit_code": 0, "stdout": "", "stderr": ""}, None)
        run.sparagmos_side = lambda *a, **kw: (
            {"exit_code": 0, "stdout": "", "stderr": ""},
            "result", "")
        try:
            res = run.process_test(rec, self.rust_dir,
                                   Path("/fake/driver"))
        finally:
            run.native_side, run.sparagmos_side = orig_n, orig_s
        self.assertEqual(res["status"], "pass")
        self.assertNotIn("jit", res)

    def test_compile_check_jit_unsupported_is_not_a_pass(self):
        rec = self.write_test("//@ check-pass\nfn main() {}\n",
                              nature="compile-check")
        orig = run.convert_side

        def fake_conv(*a, **kw):
            self.assertTrue(kw.get("jit", False))
            return conv(exit_code=2, jit_compiled=False,
                        jit_reason="unsupported rvalue")

        run.convert_side = fake_conv
        try:
            res = run.process_test(rec, self.rust_dir,
                                   Path("/fake/driver"),
                                   stage2=True, jit=True)
        finally:
            run.convert_side = orig
        self.assertEqual(res["status"], "unsupported")
        self.assertIn("JIT lowering unsupported", res["bucket"])
        self.assertTrue(res["jit"])

    def test_compile_check_jit_compiled_passes(self):
        rec = self.write_test("//@ check-pass\nfn main() {}\n",
                              nature="compile-check")
        orig = run.convert_side
        run.convert_side = lambda *a, **kw: conv(
            jit_compiled=True, jit_reason="")
        try:
            res = run.process_test(rec, self.rust_dir,
                                   Path("/fake/driver"),
                                   stage2=True, jit=True)
        finally:
            run.convert_side = orig
        self.assertEqual(res["status"], "pass")

    def test_unattemptable_nature_counts_as_not_attempted(self):
        # procedure / exec-debugger / rustdoc records have no JIT
        # attempt in this harness; under --jit they stay
        # not-attempted (counted against the metric-4 rate, never
        # removed from the denominator).
        rec = self.write_test("fn main() {}\n", nature="procedure")
        res = run.process_test(rec, self.rust_dir,
                               Path("/fake/driver"), jit=True)
        self.assertEqual(res["status"], "not-attempted")

    def test_debugger_nature_stays_not_attempted_under_jit(self):
        # The debugger harness judges through the interpreter, so
        # under --jit an exec-debugger record must not reach it:
        # metric 4 counts it as not-attempted, exactly like the
        # other unattemptable natures. (The debugger candidates in
        # main() are submitted with the same jit flag; this pins
        # the dispatch they land on.)
        rec = self.write_test("fn main() {}\n", nature="exec-debugger")
        res = run.process_test(rec, self.rust_dir,
                               Path("/fake/driver"), stage2=True, jit=True)
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"], "out of stage-1 scope (nature)")


class Stage3ProcedureTests(unittest.TestCase):
    """Stage 3 (procedure / run-make): the native oracle runs
    first, the driver run second; both sides are stubbed."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.rust_dir = Path(self.td.name)
        self.test_dir = self.rust_dir / "tests" / "run-make" / "demo"
        self.test_dir.mkdir(parents=True)
        (self.test_dir / "input.txt").write_text("data")
        self.rec = {"suite": "run-make", "path": "tests/run-make/demo",
                    "nature": "procedure", "single_file": False,
                    "ffi": False, "env_excluded": False}

    def tearDown(self):
        self.td.cleanup()

    def write_recipe(self, text="fn main() {}\n"):
        (self.test_dir / "rmake.rs").write_text(text)

    def run_procedure(self, compile_res, native_res, guest_res):
        calls = []

        def fake_run_cmd(argv, **kw):
            calls.append(argv)
            if "-o" in argv:
                return compile_res
            if str(argv[0]).endswith("rmake"):
                return native_res
            return guest_res

        ok = {"exit_code": 0, "stdout": "", "stderr": "",
              "timeout": False}
        out = {"status": "not-attempted", "reason": "", "bucket": ""}
        with unittest.mock.patch.object(run, "run_cmd", fake_run_cmd),                 unittest.mock.patch.object(
                    run, "ensure_stage3_support",
                    lambda rd: (Path("/flat"),
                                Path("/flat/librun_make_support.rlib"))),                 unittest.mock.patch.object(
                    run, "stage3_toolchain_paths",
                    lambda: {"rustc": "/tc/rustc",
                             "rustdoc": "/tc/rustdoc",
                             "cargo": "/tc/cargo",
                             "sysroot_lib": "/tc/lib",
                             "target_lib": "/tc/target-lib"}),                 unittest.mock.patch.object(run, "rustc_cmd",
                                           lambda: ["rustc"]):
            run.fill_result_procedure(self.rec, self.rust_dir,
                                      Path("/driver"), out)
        return out, calls

    def test_missing_recipe_is_not_attempted(self):
        out, calls = self.run_procedure(None, None, None)
        self.assertEqual(out["status"], "not-attempted")
        self.assertEqual(out["bucket"], "out of stage-3 scope (no recipe)")
        self.assertEqual(calls, [])

    def test_revisions_are_out_of_scope(self):
        self.write_recipe("//@ revisions: a b\nfn main() {}\n")
        out, calls = self.run_procedure(None, None, None)
        self.assertEqual(out["status"], "not-attempted")
        self.assertEqual(out["bucket"], "out of stage-3 scope (revisions)")
        self.assertEqual(calls, [])

    def test_cargo_suites_are_out_of_scope(self):
        self.write_recipe()
        for suite in ("run-make-cargo", "build-std"):
            self.rec["suite"] = suite
            out, calls = self.run_procedure(None, None, None)
            self.assertEqual(out["status"], "not-attempted")
            self.assertEqual(out["bucket"],
                             "out of stage-3 scope (cargo)")
            self.assertIn("cargo", out["reason"])
            self.assertEqual(calls, [])

    def make_shim_rec(self):
        self.test_dir = (self.rust_dir / "tests" / "run-make-cargo"
                         / "same-crate-name-and-macro-name")
        self.test_dir.mkdir(parents=True)
        self.rec = {"suite": "run-make-cargo",
                    "path": "tests/run-make-cargo/"
                            "same-crate-name-and-macro-name",
                    "nature": "procedure", "single_file": False,
                    "ffi": False, "env_excluded": False}
        self.write_recipe()

    def run_shim_procedure(self, guest_res, verdicts_text):
        """fill_result_procedure for a cargo-shim record, with
        the guest run stubbed to leave `verdicts_text` in the
        side channel (None: the channel is never written)."""
        calls = []

        def fake_run_cmd(argv, **kw):
            calls.append((argv, kw))
            if "-o" in argv:
                return {"exit_code": 0, "stdout": "", "stderr": "",
                        "timeout": False}
            if str(argv[0]).endswith("rmake"):
                return {"exit_code": 0, "stdout": "", "stderr": "",
                        "timeout": False}
            if verdicts_text is not None:
                Path(kw["env_extra"]
                     ["SPARAGMOS_CARGO_SHIM_VERDICTS"]).write_text(
                         verdicts_text)
            return guest_res

        out = {"status": "not-attempted", "reason": "", "bucket": ""}
        with unittest.mock.patch.object(run, "run_cmd", fake_run_cmd), \
                unittest.mock.patch.object(
                    run, "ensure_stage3_support",
                    lambda rd: (Path("/flat"),
                                Path("/flat/librun_make_support.rlib"))), \
                unittest.mock.patch.object(
                    run, "stage3_toolchain_paths",
                    lambda: {"rustc": "/tc/rustc",
                             "rustdoc": "/tc/rustdoc",
                             "cargo": "/tc/cargo",
                             "sysroot_lib": "/tc/lib",
                             "target_lib": "/tc/target-lib"}), \
                unittest.mock.patch.object(run, "rustc_cmd",
                                           lambda: ["rustc"]):
            run.fill_result_procedure(self.rec, self.rust_dir,
                                      Path("/driver"), out)
        return out, calls

    def test_cargo_shim_record_passes_when_every_unit_converts(self):
        self.make_shim_rec()
        ok = {"exit_code": 0, "stdout": "", "stderr": "",
              "timeout": False}
        out, calls = self.run_shim_procedure(
            ok, "converted\tmylib\tmylib.rs\n"
                "converted\tconsumer\tconsumer.rs\n")
        self.assertEqual(out["status"], "pass")
        guest_argv, guest_kw = calls[-1]
        self.assertIn("--guest-process-host", guest_argv)
        self.assertIn("RUSTC_WRAPPER", guest_kw["env_extra"])
        self.assertIn("SPARAGMOS_CARGO_SHIM_VERDICTS",
                      guest_kw["env_extra"])

    def test_cargo_shim_record_fails_when_a_unit_does_not_convert(self):
        self.make_shim_rec()
        ok = {"exit_code": 0, "stdout": "", "stderr": "",
              "timeout": False}
        out, _ = self.run_shim_procedure(
            ok, "converted\tmylib\tmylib.rs\n"
                "failed\tconsumer\tconsumer.rs\n")
        self.assertEqual(out["status"], "fail")
        self.assertIn("did not convert", out["reason"])

    def test_cargo_shim_record_fails_without_any_verdict(self):
        self.make_shim_rec()
        ok = {"exit_code": 0, "stdout": "", "stderr": "",
              "timeout": False}
        out, _ = self.run_shim_procedure(ok, None)
        self.assertEqual(out["status"], "fail")
        self.assertIn("no compilation units", out["reason"])

    def test_cargo_shim_driver_unsupported_stays_unsupported(self):
        self.make_shim_rec()
        unsup = {"exit_code": 2, "stdout": "",
                 "stderr": "sparagmos-driver run: unsupported: no\n",
                 "timeout": False}
        out, _ = self.run_shim_procedure(unsup, None)
        self.assertEqual(out["status"], "unsupported")

    def test_native_failure_is_a_plain_failure(self):
        self.write_recipe()
        bad = {"exit_code": 1, "stdout": "",
               "stderr": "recipe blew up", "timeout": False}
        ok = {"exit_code": 0, "stdout": "", "stderr": "",
              "timeout": False}
        out, calls = self.run_procedure(ok, bad, ok)
        self.assertEqual(out["status"], "fail")
        self.assertEqual(out["bucket"], "native recipe failure")
        # compile + oracle run only; the driver is never invoked
        self.assertEqual(len(calls), 2)

    def test_native_compile_failure_is_a_plain_failure(self):
        self.write_recipe()
        bad = {"exit_code": 1, "stdout": "",
               "stderr": "error[E0308]: mismatched types",
               "timeout": False}
        ok = {"exit_code": 0, "stdout": "", "stderr": "",
              "timeout": False}
        out, calls = self.run_procedure(bad, ok, ok)
        self.assertEqual(out["status"], "fail")
        self.assertEqual(out["bucket"], "native recipe failure")
        self.assertEqual(len(calls), 1)

    def test_driver_success_passes(self):
        self.write_recipe()
        ok = {"exit_code": 0, "stdout": "", "stderr": "",
              "timeout": False}
        out, calls = self.run_procedure(ok, ok, ok)
        self.assertEqual(out["status"], "pass")
        self.assertEqual(out["stage"], 3)
        driver_argv = calls[-1]
        self.assertIn("--tool-program", driver_argv)
        self.assertIn("/tc/rustc", driver_argv)
        self.assertIn("--aux-lib", driver_argv)

    def test_driver_unsupported_is_unsupported(self):
        self.write_recipe()
        ok = {"exit_code": 0, "stdout": "", "stderr": "",
              "timeout": False}
        unsup = {"exit_code": 2, "stdout": "",
                 "stderr": "sparagmos-driver run: unsupported: no\n",
                 "timeout": False}
        out, _ = self.run_procedure(ok, ok, unsup)
        self.assertEqual(out["status"], "unsupported")
        self.assertIn("unsupported: no", out["reason"])

    def test_driver_failure_is_a_failure(self):
        self.write_recipe()
        ok = {"exit_code": 0, "stdout": "", "stderr": "",
              "timeout": False}
        guest_fail = {"exit_code": 101, "stdout": "",
                      "stderr": "thread main panicked",
                      "timeout": False}
        out, _ = self.run_procedure(ok, ok, guest_fail)
        self.assertEqual(out["status"], "fail")
        self.assertEqual(out["bucket"], "recipe failed under sparagmos")

    def test_recipe_env_shape(self):
        with unittest.mock.patch.object(
                run, "stage3_toolchain_paths",
                lambda: {"rustc": "/tc/rustc", "rustdoc": "/tc/rustdoc",
                         "cargo": "/tc/cargo", "sysroot_lib": "/tc/lib",
                             "target_lib": "/tc/target-lib"}):
            env = run.stage3_recipe_env(Path("/rust"), Path("/build"),
                                        "run-make")
            self.assertEqual(env["RUSTC"], "/tc/rustc")
            self.assertEqual(env["TARGET"],
                             "x86_64-unknown-linux-gnu")
            self.assertEqual(env["LD_LIB_PATH_ENVVAR"],
                             "LD_LIBRARY_PATH")
            # htmldocck.py refuses to start without the channel
            # prefix bootstrap exports for compiletest.
            self.assertEqual(env["DOC_RUST_LANG_ORG_CHANNEL"],
                             "https://doc.rust-lang.org/1.100.0")
            self.assertNotIn("CARGO", env)
            env = run.stage3_recipe_env(Path("/rust"), Path("/build"),
                                        "run-make-cargo")
            self.assertEqual(env["CARGO"], "/tc/cargo")


FAKE_TC_PATHS = {"rustc": "/tc/rustc", "rustdoc": "/tc/rustdoc",
                 "cargo": "/tc/cargo", "sysroot_lib": "/tc/lib",
                 "target_lib": "/tc/target-lib"}


class Stage4RustdocTests(unittest.TestCase):
    """Stage 4 (rustdoc-check): the pure verdict helpers are
    exercised directly; the record flow is exercised with both
    process sides stubbed."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.rust_dir = Path(self.td.name)
        tc_patch = unittest.mock.patch.object(
            run, "stage3_toolchain_paths", lambda: FAKE_TC_PATHS)
        tc_patch.start()
        self.addCleanup(tc_patch.stop)

    def tearDown(self):
        self.td.cleanup()

    def write_rec(self, suite, rel, text, single_file=True):
        path = self.rust_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return {"suite": suite, "path": rel, "nature": "rustdoc-check",
                "single_file": single_file, "ffi": False,
                "env_excluded": False}

    # -- pure helpers -------------------------------------------------

    def test_extract_rendered(self):
        diag = {"message": "m", "code": None, "level": "error",
                "spans": [], "children": [],
                "rendered": "error: m\n"}
        artifact = {"artifact": "x.rlib"}
        out = run.stage4_extract_rendered(
            json.dumps(diag) + "\nplain line\n"
            + json.dumps(artifact) + "\n")
        self.assertEqual(out, "error: m\nplain line\n")

    def test_load_annotations(self):
        p = self.rust_dir / "t.rs"
        p.write_text("fn main() {}\n"
                     "//~^ ERROR first\n"
                     "//~| NOTE second\n"
                     "//~? WARNING anywhere\n"
                     "let x; //~ ERROR inline\n")
        anns = run.stage4_load_annotations(p, None)
        self.assertEqual(
            [(a["line"], a["kind"], a["msg"]) for a in anns],
            [(1, "ERROR", "first"), (1, "NOTE", "second"),
             (None, "WARN", "anywhere"), (5, "ERROR", "inline")])

    def test_normalize_rules_and_apply(self):
        view = header('//@ normalize-stdout: "finished in '
                      '\\d+\\.\\d+s" -> "finished in $$TIME"\n')
        rules = run.stage4_normalize_rules(view, "normalize-stdout")
        self.assertEqual(rules, [("finished in \\d+\\.\\d+s",
                                  "finished in $$TIME")])
        out = run.stage4_normalize_output(
            "done finished in 1.25s ok", Path("/x/t.rs"),
            Path("/build"), rules, False)
        self.assertIn("finished in $TIME", out)

    def test_normalize_output_dir_and_annotations(self):
        p = Path("/tree/tests/rustdoc-ui/t.rs")
        out = run.stage4_normalize_output(
            "/tree/tests/rustdoc-ui/t.rs:1:1: error //~ ERROR x\n"
            "tab\there", p, Path("/build"), [], False)
        self.assertTrue(out.startswith("$DIR/t.rs:1:1: error"))
        self.assertIn("tab\\there", out)

    def test_pass_fail_mode(self):
        self.assertEqual(run.stage4_pass_fail_mode(header("")),
                         "check-fail")
        self.assertEqual(
            run.stage4_pass_fail_mode(header("//@ build-pass\n")),
            "build-pass")

    def test_check_expected_errors_match_and_miss(self):
        p = self.rust_dir / "t.rs"
        p.write_text("bad(); //~ ERROR unknown thing\n")
        view = header("bad(); //~ ERROR unknown thing\n")
        diag = {"message": "unknown thing happened", "code": None,
                "level": "error",
                "spans": [{"file_name": str(p), "line_start": 1,
                           "column_start": 1, "is_primary": True,
                           "label": None,
                           "suggested_replacement": None,
                           "expansion": None}],
                "children": [], "rendered": None}
        output = json.dumps(diag) + "\n"
        self.assertIsNone(run.stage4_check_expected_errors(
            view, p, output, False, None, Path("/build")))
        empty = ""
        err = run.stage4_check_expected_errors(
            view, p, empty, False, None, Path("/build"))
        self.assertIn("not found", err)

    def test_ui_verdict_modes(self):
        p = self.rust_dir / "t.rs"
        p.write_text("fn main() {}\n")
        proc = {"exit_code": 1, "stdout": "", "stderr": "",
                "timeout": False}
        ok, why = run.stage4_ui_verdict(
            header(""), p, proc, False, None, Path("/build"))
        self.assertTrue(ok)
        proc0 = dict(proc, exit_code=0)
        ok, why = run.stage4_ui_verdict(
            header(""), p, proc0, False, None, Path("/build"))
        self.assertFalse(ok)
        self.assertIn("did not emit an error", why)
        ok, why = run.stage4_ui_verdict(
            header("//@ check-pass\n"), p, proc, False, None,
            Path("/build"))
        self.assertFalse(ok)
        proc7 = dict(proc, exit_code=7)
        ok, why = run.stage4_ui_verdict(
            header("//@ failure-status: 7\n"), p, proc7, False,
            None, Path("/build"))
        self.assertTrue(ok)
        ok, why = run.stage4_ui_verdict(
            header("//@ failure-status: 7\n"), p, proc, False,
            None, Path("/build"))
        self.assertFalse(ok)
        self.assertIn("expected failure status (7)", why)

    def test_ui_verdict_snapshot(self):
        p = self.rust_dir / "t.rs"
        p.write_text("bad();\n//~^ ERROR boom\n")
        (self.rust_dir / "t.stderr").write_text("error: boom\n")
        diag = {"message": "boom", "code": None, "level": "error",
                "spans": [{"file_name": str(p), "line_start": 1,
                           "column_start": 1, "is_primary": True,
                           "label": None,
                           "suggested_replacement": None,
                           "expansion": None}],
                "children": [],
                "rendered": "error: boom\n"}
        proc = {"exit_code": 1, "stdout": "",
                "stderr": json.dumps(diag) + "\n", "timeout": False}
        ok, why = run.stage4_ui_verdict(
            header(""), p, proc, False, None, Path("/build"))
        self.assertTrue(ok, why)
        proc2 = dict(proc, stderr=json.dumps(
            dict(diag, rendered="error: bam\n")) + "\n")
        ok, why = run.stage4_ui_verdict(
            header(""), p, proc2, False, None, Path("/build"))
        self.assertFalse(ok)
        self.assertIn("snapshot", why)

    # -- record flow ---------------------------------------------------

    def test_browser_suite_is_a_scope_result(self):
        rec = self.write_rec("rustdoc-gui", "tests/rustdoc-gui/x.goml",
                             "")
        res = run.process_test(rec, self.rust_dir, Path("/driver"),
                               stage4=True)
        self.assertEqual(res["stage"], 4)
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(
            res["bucket"],
            "out of stage-4 scope (JavaScript/browser environment)")

    def test_rustdoc_record_without_stage4_is_carried(self):
        rec = self.write_rec("rustdoc-html", "tests/rustdoc-html/x.rs",
                             "/// hi\npub fn f() {}\n")
        res = run.process_test(rec, self.rust_dir, Path("/driver"))
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"],
                         "out of stage-1 scope (nature)")

    def test_stage4_posix_classes_translate(self):
        # Rust's regex crate accepts POSIX classes inside bracket
        # expressions; Python's re needs the class body spelled
        # out.
        pat = run._stage4_translate_posix("[[:word:]:-]+")
        self.assertIn(r"\w", pat)
        out = run._stage4_apply_rule("abc 123 def",
                                     "[[:digit:]]+", "N")
        self.assertEqual(out, "abc N def")
        # Unknown classes pass through untouched.
        self.assertEqual(run._stage4_translate_posix("[[:foo:]]"),
                         "[[:foo:]]")

    def test_stage4_ui_fail_bucket_kinds(self):
        self.assertEqual(
            run._stage4_ui_fail_bucket(
                "test compilation failed although it shouldn't"),
            "exit-code mismatch")
        self.assertEqual(
            run._stage4_ui_fail_bucket(
                "expected failure status (101) but received "
                "status 1"),
            "exit-code mismatch")
        self.assertEqual(
            run._stage4_ui_fail_bucket(
                "stdout differs from the snapshot"),
            "stdout mismatch")

    def test_stage4_ui_args_auto_check_cfg(self):
        # compose_and_run_compiler appends the auto --check-cfg
        # to every run (it activates the unexpected-cfg
        # warnings); a test opts out with no-auto-check-cfg.
        text = "pub fn f() {}\n"
        header = run.Header(text)
        view = header.view(None)
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            args, _explicit = run.stage4_ui_args(
                view, base / "x.rs", base, None, [], [], base,
                base, self.rust_dir)
        self.assertIn("--check-cfg", args)
        self.assertIn("cfg(test,FALSE)", args)
        header2 = run.Header("//@ no-auto-check-cfg\n" + text)
        view2 = header2.view(None)
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            args2, _e = run.stage4_ui_args(
                view2, base / "x.rs", base, None, [], [], base,
                base, self.rust_dir)
        self.assertNotIn("--check-cfg", args2)

    def test_html_native_failure_is_a_plain_failure(self):
        rec = self.write_rec("rustdoc-html", "tests/rustdoc-html/x.rs",
                             "/// hi\npub fn f() {}\n")
        bad = {"exit_code": 1, "stdout": "", "stderr": "error: nope",
               "timeout": False}
        with unittest.mock.patch.object(run, "run_cmd",
                                        lambda argv, **kw: bad):
            res = run.process_test(rec, self.rust_dir, Path("/driver"),
                                   stage4=True)
        self.assertEqual(res["status"], "fail")
        # A judged fail never carries the empty bucket: the
        # pass arithmetic counts bucket == "" as a pass.
        self.assertEqual(res["bucket"], "native rustdoc failure")
        self.assertTrue(res["reason"].startswith(
            "native rustdoc failure"))

    def test_html_probe_unsupported_makes_record_unsupported(self):
        rec = self.write_rec("rustdoc-html", "tests/rustdoc-html/x.rs",
                             "/// hi\npub fn f() {}\n")
        ok = {"exit_code": 0, "stdout": "", "stderr": "",
              "timeout": False}
        tools = {"htmldocck": Path("/tools/htmldocck.py"),
                 "jsondocck": None, "jsondoclint": None,
                 "error": None, "build_errors": {}}
        with unittest.mock.patch.object(run, "run_cmd",
                                        lambda argv, **kw: ok), \
                unittest.mock.patch.object(
                    run, "stage4_checker_tools", lambda rd: tools), \
                unittest.mock.patch.object(
                    run, "stage4_guest_probe",
                    lambda db, rd: {"state": "unsupported",
                                    "reason": "unsupported: frobnicate"}):
            res = run.process_test(rec, self.rust_dir, Path("/driver"),
                                   stage4=True)
        self.assertEqual(res["status"], "unsupported")
        self.assertEqual(res["reason"], "unsupported: frobnicate")

    def test_html_probe_unavailable_is_not_attempted(self):
        rec = self.write_rec("rustdoc-html", "tests/rustdoc-html/x.rs",
                             "/// hi\npub fn f() {}\n")
        ok = {"exit_code": 0, "stdout": "", "stderr": "",
              "timeout": False}
        tools = {"htmldocck": Path("/tools/htmldocck.py"),
                 "jsondocck": None, "jsondoclint": None,
                 "error": None, "build_errors": {}}
        with unittest.mock.patch.object(run, "run_cmd",
                                        lambda argv, **kw: ok), \
                unittest.mock.patch.object(
                    run, "stage4_checker_tools", lambda rd: tools), \
                unittest.mock.patch.object(
                    run, "stage4_guest_probe",
                    lambda db, rd: {"state": "unavailable",
                                    "reason": "no tree"}):
            res = run.process_test(rec, self.rust_dir, Path("/driver"),
                                   stage4=True)
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"],
                         "not attempted (rustdoc artifact tree)")

    def test_configured_paths_resolve_absolute(self):
        # --rustdoc-artifacts / --rustdoc-main (and their env
        # counterparts) may name relative paths: validation runs
        # in the runner's working directory, but the probe and the
        # guest runs execute from temporary directories, so the
        # resolvers must hand back absolute paths.
        tree = self.rust_dir / "artifacts"
        tree.mkdir()
        (tree / "librustdoc.rlib").write_bytes(b"")
        (tree / "librustc_driver.so").write_bytes(b"")
        (tree / "main.rs").write_text("fn main() {}\n")
        old_cwd = os.getcwd()
        self.addCleanup(os.chdir, old_cwd)
        self.addCleanup(setattr, run, "_STAGE4_ARTIFACTS_ARG",
                        run._STAGE4_ARTIFACTS_ARG)
        self.addCleanup(setattr, run, "_STAGE4_MAIN_RS_ARG",
                        run._STAGE4_MAIN_RS_ARG)
        run._STAGE4_ARTIFACTS_ARG = "artifacts"
        run._STAGE4_MAIN_RS_ARG = "artifacts/main.rs"
        os.chdir(self.rust_dir)
        artifacts = run.stage4_artifacts_dir()
        main_rs = run.stage4_main_rs(self.rust_dir)
        self.assertEqual(artifacts, tree.resolve())
        self.assertEqual(main_rs, (tree / "main.rs").resolve())
        # From any other working directory (the probe's and the
        # guests' temporary directories), the returned paths must
        # still name the validated files.
        os.chdir(self.rust_dir.parent)
        self.assertTrue((artifacts / "librustdoc.rlib").is_file())
        self.assertTrue(main_rs.is_file())

    # -- UI guest exit statuses --------------------------------------
    #
    # The guest rustdoc's own exit status belongs to the UI
    # verdict: a check-fail test is supposed to exit nonzero, so
    # anything but the driver's reserved exit 2 must reach
    # stage4_ui_verdict instead of failing the record up front.

    def ui_diag(self, line, rendered="error: boom\n"):
        p = self.rust_dir / "tests/rustdoc-ui/x.rs"
        diag = {"message": "boom", "code": None, "level": "error",
                "spans": [{"file_name": str(p), "line_start": line,
                           "column_start": 1, "is_primary": True,
                           "label": None,
                           "suggested_replacement": None,
                           "expansion": None}],
                "children": [], "rendered": rendered}
        return json.dumps(diag) + "\n"

    def run_ui_record(self, text, native_proc, guest_proc,
                      snapshot=None):
        rec = self.write_rec("rustdoc-ui", "tests/rustdoc-ui/x.rs",
                             text)
        if snapshot is not None:
            (self.rust_dir / "tests/rustdoc-ui/x.stderr").write_text(
                snapshot)

        def fake_run_cmd(argv, **kw):
            if argv[0] == FAKE_TC_PATHS["rustdoc"]:
                return native_proc
            return guest_proc

        with unittest.mock.patch.object(run, "run_cmd",
                                        fake_run_cmd), \
                unittest.mock.patch.object(
                    run, "stage4_guest_probe",
                    lambda db, rd: {"state": "ok", "reason": ""}), \
                unittest.mock.patch.object(
                    run, "stage4_artifacts_dir",
                    lambda: Path("/artifacts")), \
                unittest.mock.patch.object(
                    run, "stage4_main_rs",
                    lambda rd: Path("/artifacts/main.rs")):
            return run.process_test(rec, self.rust_dir,
                                    Path("/driver"), stage4=True)

    def test_ui_build_fail_mode_is_not_attempted(self):
        # compiletest's build-fail prerequisite is a check build
        # (--emit=metadata) that must pass while the judged build
        # fails; that distinction exists only for rustc (rustdoc
        # rejects --emit=metadata, and compiletest passes no
        # --emit to rustdoc at all), so stage 4 classifies the
        # mode instead of requiring one deterministic run to both
        # pass and fail. The native stub fails the way a real
        # build-fail judged run must; the previous flow failed
        # the record as a native failure on that prerequisite
        # before the guest was ever judged.
        calls = []

        def counting_run_cmd(argv, **kw):
            calls.append(argv)
            return {"exit_code": 1, "stdout": "",
                    "stderr": "error: nope\n", "timeout": False}

        rec = self.write_rec("rustdoc-ui", "tests/rustdoc-ui/x.rs",
                             "//@ build-fail\n/// [missing]\n"
                             "pub fn f() {}\n")
        with unittest.mock.patch.object(run, "run_cmd",
                                        counting_run_cmd):
            res = run.process_test(rec, self.rust_dir,
                                   Path("/driver"), stage4=True)
        self.assertEqual(res["status"], "not-attempted")
        self.assertEqual(res["bucket"],
                         "out of stage-4 scope (build-fail)")
        self.assertEqual(calls, [])

    def test_ui_check_fail_guest_reproducing_native_passes(self):
        proc = {"exit_code": 1, "stdout": "",
                "stderr": self.ui_diag(1), "timeout": False}
        res = self.run_ui_record("bad();\n//~^ ERROR boom\n",
                                 proc, dict(proc),
                                 snapshot="error: boom\n")
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["bucket"], "")

    def test_ui_check_fail_guest_silent_success_fails(self):
        native = {"exit_code": 1, "stdout": "",
                  "stderr": self.ui_diag(1), "timeout": False}
        guest = {"exit_code": 0, "stdout": "", "stderr": "",
                 "timeout": False}
        res = self.run_ui_record("bad();\n//~^ ERROR boom\n",
                                 native, guest,
                                 snapshot="error: boom\n")
        self.assertEqual(res["status"], "fail")
        self.assertIn("did not emit an error", res["reason"])
        self.assertEqual(res["bucket"], "exit-code mismatch")

    def test_ui_check_fail_guest_wrong_output_fails_on_output(self):
        native = {"exit_code": 1, "stdout": "",
                  "stderr": self.ui_diag(1), "timeout": False}
        guest = {"exit_code": 1, "stdout": "",
                 "stderr": self.ui_diag(1, "error: bam\n"),
                 "timeout": False}
        res = self.run_ui_record("bad();\n//~^ ERROR boom\n",
                                 native, guest,
                                 snapshot="error: boom\n")
        self.assertEqual(res["status"], "fail")
        self.assertTrue(res["reason"].startswith(
            "guest rustdoc failure:"))
        self.assertIn("snapshot", res["reason"])
        self.assertEqual(res["bucket"], "stdout mismatch")

    def test_ui_failure_status_guest_matching_status_passes(self):
        proc = {"exit_code": 7, "stdout": "",
                "stderr": self.ui_diag(2), "timeout": False}
        res = self.run_ui_record(
            "//@ failure-status: 7\nbad();\n//~^ ERROR boom\n",
            proc, dict(proc), snapshot="error: boom\n")
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["bucket"], "")

    def test_ui_check_pass_guest_failure_is_a_verdict_failure(self):
        native = {"exit_code": 0, "stdout": "", "stderr": "",
                  "timeout": False}
        guest = {"exit_code": 1, "stdout": "",
                 "stderr": "error: nope\n", "timeout": False}
        res = self.run_ui_record(
            "//@ check-pass\n/// hi\npub fn f() {}\n",
            native, guest)
        self.assertEqual(res["status"], "fail")
        self.assertIn("test compilation failed", res["reason"])
        self.assertEqual(res["bucket"], "exit-code mismatch")

    def test_ui_guest_driver_unsupported_stays_unsupported(self):
        native = {"exit_code": 1, "stdout": "",
                  "stderr": self.ui_diag(1), "timeout": False}
        guest = {"exit_code": 2, "stdout": "",
                 "stderr": "sparagmos-driver run: unsupported: "
                           "frobnicate\n",
                 "timeout": False}
        res = self.run_ui_record("bad();\n//~^ ERROR boom\n",
                                 native, guest,
                                 snapshot="error: boom\n")
        self.assertEqual(res["status"], "unsupported")
        self.assertIn("unsupported: frobnicate", res["reason"])

    def test_probe_timeout_default_matches_native_timeout(self):
        # The startup probe only asks the guest driver for its
        # version; its built-in default is the native timeout
        # (240 s), not the guest run's 1800 s (Issue #482). The
        # constants are read from the environment at import time,
        # so re-load the module under a controlled one.
        def fresh(probe_timeout):
            spec = importlib.util.spec_from_file_location(
                "rustc_subset_run_fresh",
                Path(__file__).parent / "rustc-subset-run.py")
            mod = importlib.util.module_from_spec(spec)
            with unittest.mock.patch.dict(os.environ):
                if probe_timeout is None:
                    os.environ.pop(
                        "SPARAGMOS_RUSTDOC_PROBE_TIMEOUT", None)
                else:
                    os.environ["SPARAGMOS_RUSTDOC_PROBE_TIMEOUT"] = \
                        probe_timeout
                spec.loader.exec_module(mod)
            return mod

        mod = fresh(None)
        self.assertEqual(mod.STAGE4_PROBE_TIMEOUT, 240)
        self.assertEqual(mod.STAGE4_PROBE_TIMEOUT,
                         mod.STAGE4_NATIVE_TIMEOUT)
        # The environment override still applies.
        mod = fresh("1800")
        self.assertEqual(mod.STAGE4_PROBE_TIMEOUT, 1800)


class CoverageNormalizeTests(unittest.TestCase):
    """compiletest's coverage-output normalisation, ported for
    the coverage suite's stage-1 path (issue #514)."""

    def test_sections_sorted_and_line_numbers_anonymized(self):
        text = (
            "/x/b.rs:\n"
            "    9|      3|fn b() {}\n"
            "\n"
            "/x/a.rs:\n"
            "    4|      2|fn a() {\n"
            "  ------------------\n"
            "  |  Branch (4:9): [True: 2, False: 1]\n"
            "  ------------------\n"
            "\n"
        )
        self.assertEqual(
            run.normalize_coverage_output(text, "/x"),
            "$DIR/a.rs:\n"
            "   LL|      2|fn a() {\n"
            "  ------------------\n"
            "  |  Branch (LL:9): [True: 2, False: 1]\n"
            "  ------------------\n"
            "\n"
            "$DIR/b.rs:\n"
            "   LL|      3|fn b() {}\n"
            "\n")

    def test_instantiation_subviews_sorted(self):
        text = (
            "    1|      2|fn g() {\n"
            "  ------------------\n"
            "  | z::g::<u32>:\n"
            "  |    1|      1|fn g() {\n"
            "  ------------------\n"
            "  | a::g::<f32>:\n"
            "  |    1|      1|fn g() {\n"
            "  ------------------\n"
            "\n"
        )
        self.assertEqual(
            run.normalize_coverage_output(text, "/x"),
            "   LL|      2|fn g() {\n"
            "  ------------------\n"
            "  | a::g::<f32>:\n"
            "  |   LL|      1|fn g() {\n"
            "  ------------------\n"
            "  | z::g::<u32>:\n"
            "  |   LL|      1|fn g() {\n"
            "  ------------------\n"
            "\n")

    def test_report_must_end_with_an_extra_blank_line(self):
        with self.assertRaises(ValueError):
            run.normalize_coverage_output(
                "    1|      1|fn a() {}\n", "/x")

    def test_cov_map_normalization_only_substitutes_the_dir(self):
        text = "Function name: f\n- file 0 => /x/tests/coverage/a.rs\n"
        self.assertEqual(
            run.normalize_cov_map_output(text, "/x/tests/coverage"),
            "Function name: f\n- file 0 => $DIR/a.rs\n")


class PartialResultsTests(unittest.TestCase):
    def record(self, path, status):
        return {"suite": "ui", "path": path, "nature": "exec",
                "single_file": True, "ffi": False, "status": status,
                "reason": "", "bucket": "", "mode": None,
                "seconds": 0.0}

    def test_snapshot_is_marked_incomplete_and_well_formed(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "results.json"
            results = [self.record("b.rs", "fail"),
                       self.record("a.rs", "pass"),
                       self.record("c.rs", "not-attempted")]
            run.write_partial_results(out, results, 24937, 12.34, 48,
                                      False)
            data = json.loads(out.read_text())
            self.assertIs(data["complete"], False)
            # records are stored in the final file's path order
            self.assertEqual([r["path"] for r in data["results"]],
                             ["a.rs", "b.rs", "c.rs"])
            agg = data["aggregates"]
            self.assertEqual(agg["denominator"], 24937)
            self.assertEqual(agg["results_total"], 3)
            self.assertEqual(agg["passed"], 1)
            self.assertEqual(agg["failed"], 1)
            self.assertEqual(agg["not_attempted"], 1)
            self.assertEqual(agg["unsupported"], 0)
            self.assertFalse(agg["jit_mode"])
            # the temp file is replaced, not left behind
            self.assertFalse((Path(d) / "results.json.tmp").exists())

    def test_later_snapshot_replaces_earlier(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "results.json"
            run.write_partial_results(
                out, [self.record("a.rs", "pass")], 10, 1.0, 2, True)
            run.write_partial_results(
                out, [self.record("a.rs", "pass"),
                      self.record("b.rs", "pass")], 10, 2.0, 2, True)
            data = json.loads(out.read_text())
            self.assertIs(data["complete"], False)
            self.assertEqual(data["aggregates"]["results_total"], 2)
            self.assertEqual(data["aggregates"]["passed"], 2)
            self.assertTrue(data["aggregates"]["jit_mode"])
