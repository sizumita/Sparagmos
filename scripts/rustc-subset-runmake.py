#!/usr/bin/env python3
"""Measure the run-make (procedure) category of the rustc test-suite
subset for Sparagmos metric 2(b).

Why this category needs its own runner
--------------------------------------
A run-make test is not a program; it is a *recipe* (`rmake.rs`, built on
the run-make-support library) that compiletest compiles and runs, and the
recipe's own assertions are the verdict: it may compile several crates,
link against C libraries, invoke cargo/rustdoc/LLVM tools, and diff
generated artifacts.  Sparagmos's main path (sparagmos-driver -> Stable
MIR conversion -> sparagmos-interp execution) can only judge the part of
a test that is the *execution of a Rust program*.  Counting a recipe as
passed because its rustc invocations succeed natively would credit
Sparagmos for work it never did, so this runner only attempts the
recipes whose verdict provably reduces to program execution, and
records a concrete reason for every other recipe.

Classification (mechanical, over the recipe source)
---------------------------------------------------
Every procedure test in the survey (554 records; 551 after the metric's
individual environmental exclusions) gets exactly one class:

- out-of-environment: a `//@` header condition excludes this host
  (x86_64-unknown-linux-gnu, host == target), or a `needs-*` condition
  is known not to hold for the installed rustc 1.100.0.
- unevaluatable-gate: a header condition the runner cannot evaluate.
- external-tools: the recipe drives cc/clang, cargo, rustdoc, LLVM
  tools, python, native-library builders or generic commands.
- not-single-program: the recipe compiles zero or several programs,
  uses bare_rustc / rustc_minicore, or links crates with `--extern`.
- compile-verdict-only: the verdict rests on compilation results or
  compiler output (including expected compile failure); no produced
  binary is ever run.
- artifact-checks: a produced binary is run, but the verdict also reads
  or diffs files/artifacts (diff(), --emit output, symbol tables, ...).
- builder-options: the single compilation uses rustc-builder options
  beyond input/output/edition/crate-type(bin), which the driver cannot
  reproduce (flags, --cfg, codegen options, ...).
- input-not-local: the compiled input is not a plain file sitting in
  the test directory (generated sources, source_root() paths, ...).
- run-shape: the binary is run with arguments, expected to fail, or
  run more than once, so the run does not reduce to one plain run.
- recipe-shape: anything else (extra statements, control flow, other
  run-make-support calls, non-enumerable imports, revisions).
- reducible: the recipe is exactly: one `rustc()` builder chain using
  only .input/.output/.edition/.crate_type("bin") ending in .run(),
  then one `run("<output>")`, optionally with literal stdout assertions
  (assert_stdout_equals / _contains / _not_contains) chained on it.

Judging (reducible recipes only)
--------------------------------
The test directory is copied to a temporary directory (compiletest
runs recipes in a scratch copy of the test directory).  The native
side compiles the program with the reference rustc 1.100.0
(RUSTC_BOOTSTRAP=1, the recipe's edition -- recipes default to edition
2015 -- host target, opt-level 0) and runs it.  The recipe's verdict
holds natively only if compilation succeeds, the run exits 0, and
every literal stdout assertion in the recipe holds for the native
stdout; otherwise the native oracle itself does not behave as the
recipe expects here and the test is recorded as not attempted with
that reason.  The Sparagmos side runs `sparagmos-driver run` on the
same file; the test passes iff the driver exit code equals the native
exit code and the captured guest stdout is byte-identical to the
native stdout.  Driver exit 2 is recorded as unsupported with the
driver's reason; any other mismatch is a failure.

The category pass rate is passes / 551.  Not-attempted and failed
tests alike count against it, and against the overall metric-2(b)
rate (denominator 24,937); this runner only reports the category
numbers and leaves the overall figure to integration.

Usage:
    python3 scripts/rustc-subset-survey.py --rust-dir DIR --json survey.json
    python3 scripts/rustc-subset-runmake.py \
        --survey-json survey.json --rust-dir ~/rust-1.100.0-survey \
        --driver-bin crates/sparagmos-driver/target/debug/sparagmos-driver \
        --out subset-runmake-results.json [--jobs 16] [--classify-only]
"""

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TOOLCHAIN = "1.100.0"
DENOMINATOR = 24937  # metric 2(b) denominator at tag 1.100.0 (COVERAGE.md)
HOST_TRIPLE = "x86_64-unknown-linux-gnu"

NATIVE_COMPILE_TIMEOUT = 60
NATIVE_RUN_TIMEOUT = 30
DRIVER_TIMEOUT = 60

BOOTSTRAP_ENV = {"RUSTC_BOOTSTRAP": "1"}

# ---------------------------------------------------------------------------
# Host facts (the environment this measurement runs in)
# ---------------------------------------------------------------------------

HOST = {
    "target": HOST_TRIPLE,
    "os": "linux",
    "env": "gnu",
    "abi": "",
    "arch": "x86_64",
    "family": "unix",
    "vendor": "unknown",
    "binary_format": "elf",
    "endian": "little",
    "pointer_width": "64",
}

KNOWN_TARGETS = {
    "x86_64-unknown-linux-gnu", "x86_64-unknown-linux-musl",
    "i686-unknown-linux-gnu", "i586-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu", "armv7-unknown-linux-gnueabihf",
    "x86_64-pc-windows-msvc", "x86_64-pc-windows-gnu",
    "i686-pc-windows-msvc", "i686-pc-windows-gnu",
    "x86_64-apple-darwin", "aarch64-apple-darwin",
    "aarch64-apple-ios", "x86_64-apple-ios",
    "wasm32-unknown-unknown", "wasm32-wasip1", "wasm32-wasip2",
    "wasm64-unknown-unknown", "aarch64-linux-android",
    "x86_64-linux-android", "riscv64gc-unknown-linux-gnu",
    "powerpc64le-unknown-linux-gnu", "s390x-unknown-linux-gnu",
    "loongarch64-unknown-linux-gnu", "aarch64-unknown-linux-musl",
    "thumbv7em-none-eabihf", "x86_64-unknown-none",
    "aarch64-unknown-none", "nvptx64-nvidia-cuda",
    "x86_64-fortanix-unknown-sgx", "aarch64-fuchsia",
    "x86_64-unknown-freebsd", "x86_64-unknown-netbsd",
    "x86_64-unknown-openbsd", "sparc64-unknown-linux-gnu",
    "mips64-unknown-linux-gnuabi64", "bpfeb-unknown-none",
    "bpfel-unknown-none",
}

KNOWN_OSES = {
    "linux", "windows", "macos", "ios", "tvos", "visionos", "watchos",
    "android", "freebsd", "netbsd", "openbsd", "dragonfly", "solaris",
    "illumos", "haiku", "emscripten", "wasi", "hermit", "uefi",
    "fuchsia", "redox", "vxworks", "aix", "hurd", "nto", "cygwin",
    "espidf", "none",
}

KNOWN_ENVS = {
    "gnu", "musl", "msvc", "uclibc", "newlib", "sim", "macabi", "llvm",
    "ohos", "hurd", "",
}

KNOWN_ARCHS = {
    "x86_64", "x86", "aarch64", "arm", "arm64ec", "riscv64", "riscv32",
    "wasm32", "wasm64", "powerpc", "powerpc64", "s390x", "sparc64",
    "sparc32", "loongarch64", "loongarch32", "mips", "mips64", "mips32",
    "m68k", "csky", "hexagon", "bpf", "avr", "thumb", "xtensa",
}


def cond_true(name):
    """Evaluate a bare compiletest condition name against the host.

    Returns True/False, or None when the name is not one this runner
    can evaluate.  Mirrors compiletest's cfg matching: names are the
    target triple, target_os, target_env, target_arch (exact: "x86"
    does not match an x86_64 target), and a fixed set of specials.
    """
    if name in KNOWN_TARGETS:
        return name == HOST["target"]
    if name in KNOWN_OSES:
        return name == HOST["os"]
    if name in KNOWN_ENVS and name != "":
        return name == HOST["env"]
    if name in KNOWN_ARCHS:
        return name == HOST["arch"]
    specials = {
        "unix": True,
        "windows": False,
        "apple": False,
        "elf": True,
        "cross-compile": False,
        "endian-big": False,
        "endian-little": True,
        "64bit": True,
        "32bit": False,
        "16bit": False,
        "test": True,
        "auxiliary": True,
        "enzyme": False,
        "offload": False,
        "remote": False,
        "i586": False,
        "thumb": False,
        "freestanding": False,
    }
    if name in specials:
        return specials[name]
    return None


# ---------------------------------------------------------------------------
# Header directives and gating
# ---------------------------------------------------------------------------

DIRECTIVE_RE = re.compile(
    r"^\s*//@(?:\[([^\]]+)\])?\s*([a-zA-Z][a-zA-Z0-9_-]*)"
    r"(?:\s*:\s*(.*?)\s*|\s*\((.*)\))?\s*$")


class Header:
    """Parsed `//@` directives of one rmake.rs file."""

    def __init__(self, text):
        self.entries = []  # (name, value_or_None, scoped_bool)
        for line in text.splitlines():
            m = DIRECTIVE_RE.match(line)
            if not m:
                continue
            scoped, name, colon_val, paren_val = m.groups()
            value = colon_val if colon_val is not None else paren_val
            self.entries.append((name, value, scoped is not None))

    def values(self, name):
        return [v for n, v, _ in self.entries if n == name]

    def has(self, name):
        return any(n == name for n, _, _ in self.entries)

    @property
    def has_scoped(self):
        return any(s for _, _, s in self.entries)


def split_value_list(value):
    if value is None:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


# --- cfg(...) expressions (for ignore-cfg / only-cfg) ----------------------

class _CfgParser:
    """Three-valued evaluator for cfg expressions over host atoms."""

    def __init__(self, text):
        self.tokens = re.findall(
            r"\s*(\(|\)|,|not\b|all\b|any\b|[A-Za-z_][A-Za-z0-9_]*"
            r"(?:\s*=\s*\"[^\"]*\")?)", text)
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def parse(self):
        val = self.expr()
        if self.peek() is not None:
            return None
        return val

    def expr(self):
        tok = self.next()
        if tok in ("all", "any"):
            if self.next() != "(":
                return None
            vals = []
            while True:
                vals.append(self.expr())
                sep = self.next()
                if sep == ")":
                    break
                if sep != ",":
                    return None
            if tok == "all":
                if any(v is False for v in vals):
                    return False
                return None if any(v is None for v in vals) else True
            if any(v is True for v in vals):
                return True
            return None if any(v is None for v in vals) else False
        if tok == "not":
            if self.next() != "(":
                return None
            val = self.expr()
            if self.next() != ")":
                return None
            return None if val is None else not val
        if tok is None or tok in ("(", ")", ","):
            return None
        return cfg_atom(tok)


def cfg_atom(atom):
    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\s*=\s*\"([^\"]*)\")?", atom)
    if not m:
        return None
    key, value = m.groups()
    if value is None:
        if key == "unix":
            return True
        if key == "windows":
            return False
        return None
    table = {
        "target_os": HOST["os"],
        "target_env": HOST["env"],
        "target_arch": HOST["arch"],
        "target_family": HOST["family"],
        "target_abi": HOST["abi"],
        "target_vendor": HOST["vendor"],
        "target_pointer_width": HOST["pointer_width"],
        "target_endian": HOST["endian"],
        "panic": "unwind",
    }
    if key in table:
        return table[key] == value
    return None


def eval_cfg(text):
    if text is None:
        return None
    return _CfgParser(text).parse()


# --- needs-* ---------------------------------------------------------------

class NeedsContext:
    """Mechanical facts about the installed toolchain, gathered once."""

    def __init__(self, sysroot):
        self.sysroot = sysroot
        target_lib = sysroot / "lib" / "rustlib" / HOST_TRIPLE / "lib"
        target_bin = sysroot / "lib" / "rustlib" / HOST_TRIPLE / "bin"
        self.std_present = target_lib.is_dir() and any(
            target_lib.glob("libstd-*.rlib"))
        self.profiler_present = target_lib.is_dir() and any(
            target_lib.glob("libprofiler_builtins*"))
        self.rust_lld_present = (target_bin / "rust-lld").exists()
        self.sanitizer_libs = set()
        if target_lib.is_dir():
            for f in target_lib.glob("libclang_rt.*"):
                self.sanitizer_libs.add(f.name)

    def evaluate(self, name, value):
        """True/False/None for one needs-* directive."""
        simple_true = {
            "needs-target-std": self.std_present,
            "needs-unwind": True,
            "needs-threads": True,
            "needs-subprocess": True,
            "needs-run-enabled": True,
            "needs-dynamic-linking": True,
            "needs-relocation-model-pic": True,
            "needs-symlink": True,
            "needs-deterministic-layouts": True,
            "needs-git-hash": True,
            "needs-asm-support": True,
            "needs-asm-ret": True,
            "needs-profiler-runtime": self.profiler_present,
            "needs-rust-lld": self.rust_lld_present,
        }
        if name in simple_true:
            return simple_true[name]
        simple_false = {
            "needs-dlltool": False,
            "needs-enzyme": False,
            "needs-offload": False,
            "needs-force-clang-based-tests": False,
            "needs-wasmtime": False,
        }
        if name in simple_false:
            return simple_false[name]
        if name == "needs-sanitizer-support":
            return bool(self.sanitizer_libs)
        if name.startswith("needs-sanitizer-"):
            suffix = {
                "address": "asan", "leak": "lsan", "thread": "tsan",
                "memory": "msan", "hwaddress": "hwasan",
            }.get(name[len("needs-sanitizer-"):])
            if suffix is None:
                return None
            return any(suffix in lib for lib in self.sanitizer_libs)
        if name == "needs-crate-type":
            known = {"bin", "rlib", "dylib", "cdylib", "staticlib",
                     "proc-macro", "lib"}
            vals = split_value_list(value)
            if not vals:
                return None
            if all(v in known for v in vals):
                return True
            return None  # a crate-type name this runner does not know
        if name == "needs-backends":
            vals = split_value_list(value)
            if not vals:
                return None
            return all(v == "llvm" for v in vals)
        if name == "needs-target-has-atomic":
            vals = split_value_list(value)
            if not vals:
                return None
            return all(v in {"8", "16", "32", "64", "128", "ptr"}
                       for v in vals)
        return None


def header_gate(header, needs):
    """Decide whether the test may run in this environment.

    Returns ("ok", "") / ("out", reason) / ("uneval", reason).
    """
    if header.has_scoped or header.has("revisions"):
        return "uneval", "recipe uses revisions (scoped directives)"
    for name, value, _ in header.entries:
        if name.startswith("ignore-") and name not in (
                "ignore-cfg", "ignore-backends"):
            cond = name[len("ignore-"):]
            got = cond_true(cond)
            if got is None:
                return "uneval", f"cannot evaluate header condition {name}"
            if got:
                return "out", f"header {name} matches this host"
        elif name.startswith("only-") and name not in (
                "only-cfg", "only-backends"):
            cond = name[len("only-"):]
            got = cond_true(cond)
            if got is None:
                return "uneval", f"cannot evaluate header condition {name}"
            if not got:
                return "out", f"header {name} does not match this host"
        elif name == "ignore-cfg":
            got = eval_cfg(value)
            if got is None:
                return "uneval", "cannot evaluate ignore-cfg expression"
            if got:
                return "out", "header ignore-cfg matches this host"
        elif name == "only-cfg":
            got = eval_cfg(value)
            if got is None:
                return "uneval", "cannot evaluate only-cfg expression"
            if not got:
                return "out", "header only-cfg does not match this host"
        elif name == "ignore-backends":
            if "llvm" in split_value_list(value):
                return "out", "header ignore-backends includes llvm"
        elif name == "only-backends":
            if "llvm" not in split_value_list(value):
                return "out", "header only-backends excludes llvm"
        elif name.startswith("needs-"):
            got = needs.evaluate(name, value)
            if got is None:
                return "uneval", f"cannot evaluate header condition {name}"
            if not got:
                return "out", f"header {name} does not hold here"
    return "ok", ""


# ---------------------------------------------------------------------------
# Recipe classification
# ---------------------------------------------------------------------------

def strip_comments(src):
    """Remove // and /* */ comments, preserving string literals."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth += 1
                    j += 2
                elif src.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


EXTERNAL_TOOL_RES = [
    (re.compile(r"(?<![\w.])cc\s*\("), "cc()"),
    (re.compile(r"(?<![\w.])clang\s*\("), "clang()"),
    (re.compile(r"(?<![\w.])cargo\s*\("), "cargo()"),
    (re.compile(r"(?<![\w.])rustdoc\s*\("), "rustdoc()"),
    (re.compile(r"(?<![\w.])llvm_\w+\s*\("), "llvm tools"),
    (re.compile(r"(?<![\w.])python\s*\("), "python()"),
    (re.compile(r"build_native_static_lib"), "build_native_static_lib"),
    (re.compile(r"build_native_dynamic_lib"), "build_native_dynamic_lib"),
    (re.compile(r"extra_rs_cxx_flags"), "C++ compilation flags"),
    (re.compile(r"(?<![\w.])ar\s*\("), "ar()"),
    (re.compile(r"(?<![\w.])cmd\s*\("), "generic cmd()"),
]

ARTIFACT_RES = [
    (re.compile(r"(?<![\w.])diff\s*\("), "diff() against an expected file"),
    (re.compile(r"\.emit\s*\("), "--emit output"),
    (re.compile(r"read_to_string|fs::read\b"), "reading generated files"),
    (re.compile(r"assert_dirs_are_equal"), "directory comparison"),
    (re.compile(r"get_symbol|symbols\s*\("), "symbol-table inspection"),
]

ALLOWED_BUILDER_METHODS = {"input", "output", "edition", "crate_type"}
STDOUT_ASSERT_METHODS = {
    "assert_stdout_equals", "assert_stdout_contains",
    "assert_stdout_not_contains",
}

USE_RE = re.compile(r"use\s+run_make_support::\{([^}]*)\}\s*;")
MAIN_RE = re.compile(r"fn\s+main\s*\(\s*\)\s*\{(?P<body>.*)\}\s*$", re.S)
FREE_RUN_RE = re.compile(r"(?<![\w.])run\s*\(")
FREE_RUN_FAIL_RE = re.compile(r"(?<![\w.])run_fail\s*\(")
STRING_ARG_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


class Recipe:
    """Extraction result for a reducible recipe."""

    def __init__(self, input_file, edition, output_name, assertions):
        self.input_file = input_file
        self.edition = edition
        self.output_name = output_name
        self.assertions = assertions  # [(method, literal)]


def _parse_chain_calls(text, start=0):
    """Parse a sequence of `.name(args)` calls starting at `start`.

    String-aware and whitespace-tolerant; argument text is returned
    raw (trimmed).  Returns ((calls, end_index), None) or (None, why).
    """
    calls = []
    i, n = start, len(text)
    while True:
        while i < n and text[i].isspace():
            i += 1
        if i >= n or text[i] != ".":
            break
        i += 1
        m = re.compile(r"[A-Za-z_][A-Za-z0-9_]*").match(text, i)
        if not m:
            return None, "malformed method call in chain"
        name = m.group(0)
        i = m.end()
        while i < n and text[i].isspace():
            i += 1
        if i >= n or text[i] != "(":
            return None, "malformed method call in chain"
        depth, j = 0, i
        while j < n:
            c = text[j]
            if c == '"':
                j += 1
                while j < n and text[j] != '"':
                    j += 2 if text[j] == "\\" else 1
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j >= n:
            return None, "unbalanced parentheses in chain"
        calls.append((name, text[i + 1:j].strip()))
        i = j + 1
    return (calls, i), None


def _parse_builder_chain(stmt):
    """Parse `rustc()....run()`; return (methods, None) or (None, why).

    The returned method list excludes the terminating run/run_fail.
    """
    s = stmt.strip()
    m = re.match(r"rustc\s*\(\s*\)", s)
    if not m:
        return None, "statement does not start with rustc()"
    parsed, why = _parse_chain_calls(s, m.end())
    if parsed is None:
        return None, why
    calls, end = parsed
    if s[end:].strip():
        return None, "builder chain has trailing content"
    if not calls or calls[-1][0] not in ("run", "run_fail") \
            or calls[-1][1] != "":
        return None, "builder chain does not end in .run()/.run_fail()"
    fail_expected = calls[-1][0] == "run_fail"
    return (calls[:-1], fail_expected), None


def _string_literal(arg):
    m = STRING_ARG_RE.fullmatch(arg.strip())
    if not m:
        return None
    return m.group(1)


def match_template(src, files_in_dir):
    """Strict reducible-template match.

    Returns (Recipe, None) on success or (None, (cls, reason)).
    """
    m = USE_RE.search(src)
    if not m:
        return None, ("recipe-shape",
                      "run_make_support is not imported with an "
                      "enumerable brace list")
    names = {n.strip().split("::")[-1] for n in m.group(1).split(",")
             if n.strip()}
    # The import set is checked last (just before a Recipe is
    # returned): the structural reasons below are more specific than
    # "imports something extra" when both apply.
    rest = USE_RE.sub("", src, count=1)
    if re.search(r"\buse\s", rest):
        return None, ("recipe-shape", "recipe has additional use items")
    mm = MAIN_RE.search(rest)
    if not mm or rest.replace(mm.group(0), "").strip():
        return None, ("recipe-shape",
                      "recipe has items outside fn main")
    stmts = [s.strip() for s in mm.group("body").split(";")]
    stmts = [s for s in stmts if s]
    if len(stmts) != 2:
        return None, ("recipe-shape",
                      f"recipe main has {len(stmts)} statements, "
                      "not the reducible two")
    parsed, why = _parse_builder_chain(stmts[0])
    if parsed is None:
        return None, ("recipe-shape", why)
    methods, fail_expected = parsed
    if fail_expected:
        return None, ("compile-verdict-only",
                      "the single compilation is expected to fail "
                      "(run_fail on the rustc builder)")
    extra = [n for n, _ in methods if n not in ALLOWED_BUILDER_METHODS]
    if extra:
        return None, ("builder-options",
                      "rustc builder uses options beyond "
                      f"input/output/edition/crate_type: {extra}")
    inputs = [a for n, a in methods if n == "input"]
    if len(inputs) != 1:
        return None, ("not-single-program",
                      f"builder chain has {len(inputs)} inputs")
    input_file = _string_literal(inputs[0])
    if input_file is None or "/" in input_file \
            or not input_file.endswith(".rs") \
            or input_file not in files_in_dir:
        return None, ("input-not-local",
                      "compiled input is not a plain .rs file in the "
                      f"test directory: {inputs[0].strip()}")
    edition = "2015"
    output_name = None
    for n, a in methods:
        if n == "edition":
            edition = _string_literal(a)
            if edition is None:
                return None, ("builder-options",
                              "edition is not a string literal")
        elif n == "output":
            output_name = _string_literal(a)
            if output_name is None:
                return None, ("recipe-shape",
                              "output is not a string literal")
        elif n == "crate_type":
            if _string_literal(a) != "bin":
                return None, ("not-single-program",
                              "compiled artifact is not a binary "
                              f"(crate_type {_string_literal(a)!r})")
    if output_name is None:
        output_name = input_file[:-len(".rs")]
    # Statement 2: run("name") with optional chained stdout assertions.
    s2 = stmts[1].strip()
    m2 = re.match(r"run\s*\(\s*(\"(?:[^\"\\]|\\.)*\")\s*\)", s2)
    if not m2:
        return None, ("run-shape",
                      "the produced binary is not run by a plain "
                      "run(\"...\") statement")
    run_target = _string_literal(m2.group(1))
    if run_target != output_name:
        return None, ("run-shape",
                      f"run() target {run_target!r} is not the "
                      f"compiled output {output_name!r}")
    assertions = []
    parsed, why = _parse_chain_calls(s2, m2.end())
    if parsed is None:
        return None, ("run-shape", why)
    calls, end = parsed
    if s2[end:].strip():
        return None, ("run-shape",
                      "run() statement has trailing content")
    for n, a in calls:
        if n not in STDOUT_ASSERT_METHODS:
            return None, ("run-shape",
                          f"run() result is checked with {n}, "
                          "which is not a literal stdout assertion")
        lit = _string_literal(a)
        if lit is None:
            return None, ("run-shape",
                          f"{n} argument is not a string literal")
        assertions.append((n, lit))
    if not names <= {"rustc", "run"}:
        return None, ("recipe-shape",
                      "recipe imports run_make_support items beyond "
                      f"rustc/run: {sorted(names - {'rustc', 'run'})}")
    return Recipe(input_file, edition, output_name, assertions), None


def classify_recipe(source, files_in_dir, needs):
    """Classify one recipe. Returns (cls, reason, recipe_or_None)."""
    header = Header(source)
    gate, why = header_gate(header, needs)
    if gate == "out":
        return "out-of-environment", why, None
    if gate == "uneval":
        return "unevaluatable-gate", why, None
    src = strip_comments(source)
    for rx, label in EXTERNAL_TOOL_RES:
        if rx.search(src):
            return "external-tools", f"recipe drives {label}", None
    n_rustc = len(re.findall(r"(?<![\w.])rustc\s*\(\s*\)", src))
    n_bare = len(re.findall(r"(?<![\w.])bare_rustc\s*\(", src))
    n_mini = len(re.findall(r"(?<![\w.])rustc_minicore\s*\(", src))
    if n_bare or n_mini:
        return "not-single-program", \
            "recipe uses bare_rustc/rustc_minicore", None
    if n_rustc != 1:
        return "not-single-program", \
            f"recipe invokes rustc() {n_rustc} times", None
    if re.search(r"\.extern_?\s*\(", src):
        return "not-single-program", \
            "recipe links crates with --extern", None
    runs_binary = bool(FREE_RUN_RE.search(src)) or \
        bool(re.search(r"(?<![\w.])run_with_args\s*\(", src)) or \
        bool(FREE_RUN_FAIL_RE.search(src))
    if not runs_binary:
        # A single compilation that is expected to fail is a
        # compile-time verdict; so is one whose binary never runs.
        if re.search(r"rustc\s*\(\s*\)[^;]*\.run_fail\s*\(\s*\)", src, re.S):
            return "compile-verdict-only", \
                "the compilation is expected to fail (run_fail)", None
        return "compile-verdict-only", \
            "no produced binary is ever run", None
    for rx, label in ARTIFACT_RES:
        if rx.search(src):
            return "artifact-checks", f"verdict also depends on {label}", \
                None
    recipe, miss = match_template(src, files_in_dir)
    if recipe is not None:
        return "reducible", "", recipe
    return miss[0], miss[1], None


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def rustc_cmd():
    home = Path.home()
    rustup = home / ".cargo" / "bin" / "rustup"
    if rustup.exists():
        return [str(rustup), "run", TOOLCHAIN, "rustc"]
    return ["rustup", "run", TOOLCHAIN, "rustc"]


def run_cmd(argv, timeout, cwd=None, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    try:
        p = subprocess.run(argv, capture_output=True, timeout=timeout,
                           cwd=cwd, env=env)
        return {"exit_code": p.returncode, "stdout": p.stdout,
                "stderr": p.stderr, "timeout": False}
    except subprocess.TimeoutExpired as e:
        return {"exit_code": None, "stdout": e.stdout or b"",
                "stderr": e.stderr or b"", "timeout": True}


def first_error_line(stderr_bytes):
    text = stderr_bytes.decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("error"):
            return line[:200]
    lines = text.strip().splitlines()
    return (lines[0] if lines else "<no output>")[:200]


def driver_reason(stderr_bytes):
    text = stderr_bytes.decode("utf-8", errors="replace")
    lines = [l.strip() for l in text.splitlines()
             if "sparagmos-driver run:" in l]
    if lines:
        return "; ".join(lines)[:400]
    return ("driver reported unsupported (exit 2) without a reason line: "
            + text.strip()[:300])


def rust_unescape(raw):
    """Unescape the contents of a Rust string literal (common escapes).

    Returns None for escapes this runner does not model."""
    out = []
    i, n = 0, len(raw)
    simple = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"',
              "'": "'", "0": "\0"}
    while i < n:
        c = raw[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        if i + 1 >= n:
            return None
        e = raw[i + 1]
        if e in simple:
            out.append(simple[e])
            i += 2
        elif e == "x":
            try:
                out.append(chr(int(raw[i + 2:i + 4], 16)))
            except ValueError:
                return None
            i += 4
        elif e == "u":
            m = re.match(r"u\{([0-9a-fA-F]{1,6})\}", raw[i + 1:])
            if not m:
                return None
            try:
                out.append(chr(int(m.group(1), 16)))
            except ValueError:
                return None
            i += 1 + m.end()
        else:
            return None
    return "".join(out)


def check_assertions(assertions, stdout_bytes):
    """Check the recipe's literal stdout assertions. Returns a reason
    string when one fails, else None."""
    try:
        stdout = stdout_bytes.decode("utf-8")
    except UnicodeDecodeError:
        if assertions:
            return "native stdout is not valid UTF-8"
        return None
    for method, literal in assertions:
        lit = rust_unescape(literal)
        if lit is None:
            return f"recipe assertion {method} uses escapes the " \
                   "runner cannot unescape"
        if method == "assert_stdout_equals" and stdout != lit:
            return f"native stdout does not equal the recipe's " \
                   f"assert_stdout_equals({literal!r})"
        if method == "assert_stdout_contains" and lit not in stdout:
            return f"native stdout does not contain the recipe's " \
                   f"assert_stdout_contains({literal!r})"
        if method == "assert_stdout_not_contains" and lit in stdout:
            return f"native stdout contains the recipe's " \
                   f"assert_stdout_not_contains({literal!r})"
    return None


def judge(rec, recipe, rust_dir, driver_bin):
    """Run both sides for one reducible recipe. Returns a result dict."""
    out = {"path": rec["path"], "suite": rec["suite"],
           "class": "reducible", "status": None, "reason": "",
           "input": recipe.input_file, "edition": recipe.edition}
    testdir = rust_dir / rec["path"]
    with tempfile.TemporaryDirectory(prefix="runmake-") as td:
        work = Path(td) / "work"
        try:
            shutil.copytree(testdir, work)
        except Exception as e:  # noqa: BLE001 - recorded, not raised
            out["status"] = "not-attempted"
            out["reason"] = f"runner internal error copying test dir: {e!r}"
            return out
        prog = work / recipe.input_file
        binpath = work / recipe.output_name
        build = run_cmd(
            rustc_cmd() + ["--edition", recipe.edition,
                           "--target", HOST_TRIPLE,
                           "-C", "opt-level=0",
                           "-L", str(work),
                           "-o", str(binpath), str(prog)],
            timeout=NATIVE_COMPILE_TIMEOUT, cwd=str(work),
            env_extra=BOOTSTRAP_ENV)
        if build["timeout"]:
            out["status"] = "not-attempted"
            out["reason"] = "native compile timed out"
            return out
        if build["exit_code"] != 0:
            out["status"] = "not-attempted"
            out["reason"] = ("native compile failed: "
                             + first_error_line(build["stderr"]))
            return out
        native = run_cmd([str(binpath)], timeout=NATIVE_RUN_TIMEOUT,
                         cwd=str(work))
        if native["timeout"]:
            out["status"] = "not-attempted"
            out["reason"] = "native run timed out"
            return out
        if native["exit_code"] != 0:
            out["status"] = "not-attempted"
            out["reason"] = (f"native run exit {native['exit_code']} does "
                             "not match the recipe's expectation (success)")
            return out
        bad = check_assertions(recipe.assertions, native["stdout"])
        if bad is not None:
            out["status"] = "not-attempted"
            out["reason"] = bad
            return out
        guest = run_cmd([str(driver_bin), "run", str(prog),
                         "--edition", recipe.edition],
                        timeout=DRIVER_TIMEOUT, cwd=str(work),
                        env_extra=BOOTSTRAP_ENV)
        if guest["timeout"]:
            out["status"] = "fail"
            out["reason"] = f"driver timed out after {DRIVER_TIMEOUT}s"
            return out
        if guest["exit_code"] == 2:
            out["status"] = "unsupported"
            out["reason"] = driver_reason(guest["stderr"])
            return out
        if guest["exit_code"] not in (0, 101):
            out["status"] = "fail"
            out["reason"] = (f"driver exit {guest['exit_code']}: "
                             + guest["stderr"].decode(
                                 "utf-8", errors="replace").strip()[:300])
            return out
        if guest["exit_code"] != native["exit_code"]:
            out["status"] = "fail"
            out["reason"] = (f"exit code differs: native "
                             f"{native['exit_code']}, sparagmos "
                             f"{guest['exit_code']}")
            return out
        if guest["stdout"] != native["stdout"]:
            out["status"] = "fail"
            out["reason"] = "stdout differs from the native run"
            return out
        out["status"] = "pass"
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def toolchain_sysroot():
    out = subprocess.run(rustc_cmd() + ["--print", "sysroot"],
                         capture_output=True, text=True, timeout=30)
    return Path(out.stdout.strip())


def process_record(rec, rust_dir, needs, driver_bin, classify_only):
    base = {"path": rec["path"], "suite": rec["suite"],
            "ffi": rec.get("ffi", False)}
    if rec.get("env_excluded"):
        base.update({"class": "env-excluded", "status": "env-excluded",
                     "reason": "; ".join(rec.get("env_reasons", []))})
        return base
    testdir = rust_dir / rec["path"]
    rmake = testdir / "rmake.rs"
    if not rmake.exists():
        base.update({"class": "recipe-shape", "status": "not-attempted",
                     "reason": "rmake.rs not found in the checkout"})
        return base
    source = rmake.read_text(errors="replace")
    files_in_dir = {p.name for p in testdir.iterdir() if p.is_file()}
    cls, reason, recipe = classify_recipe(source, files_in_dir, needs)
    base["class"] = cls
    if cls != "reducible":
        base.update({"status": "not-attempted", "reason": reason})
        return base
    if classify_only:
        base.update({"status": "not-attempted",
                     "reason": "classification only (not judged)",
                     "input": recipe.input_file,
                     "edition": recipe.edition})
        return base
    try:
        judged = judge(rec, recipe, rust_dir, driver_bin)
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        base.update({"status": "not-attempted",
                     "reason": f"runner internal error: {e!r}"})
        return base
    base.update({k: judged[k] for k in
                 ("status", "reason", "input", "edition")})
    return base


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--survey-json", required=True)
    ap.add_argument("--rust-dir", required=True)
    ap.add_argument("--driver-bin", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--classify-only", action="store_true")
    args = ap.parse_args(argv)

    rust_dir = Path(args.rust_dir)
    survey = json.loads(Path(args.survey_json).read_text())
    records = [r for r in survey["tests"] if r["nature"] == "procedure"]
    if args.limit:
        records = records[:args.limit]
    needs = NeedsContext(toolchain_sysroot())
    driver_bin = args.driver_bin
    if not args.classify_only:
        if not driver_bin:
            ap.error("--driver-bin is required unless --classify-only")
        driver_bin = str(Path(driver_bin).resolve())

    started = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.jobs) as pool:
        futures = [pool.submit(process_record, rec, rust_dir, needs,
                               driver_bin, args.classify_only)
                   for rec in records]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    results.sort(key=lambda r: r["path"])
    elapsed = time.time() - started

    in_denom = [r for r in results if r["status"] != "env-excluded"]
    by_status = {}
    by_class = {}
    for r in in_denom:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        by_class[r["class"]] = by_class.get(r["class"], 0) + 1
    n_pass = by_status.get("pass", 0)
    summary = {
        "denominator_procedure": len(in_denom),
        "records_total": len(results),
        "env_excluded": len(results) - len(in_denom),
        "by_status": by_status,
        "by_class": by_class,
        "attempted": sum(1 for r in in_denom if r["class"] == "reducible"),
        "category_pass_rate": (n_pass / len(in_denom)) if in_denom else 0.0,
        "elapsed_seconds": round(elapsed, 1),
        "classify_only": args.classify_only,
    }
    payload = {"summary": summary, "results": results}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=1,
                                             ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
