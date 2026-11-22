#!/usr/bin/env python3
"""Run a stage of the rustc test-suite subset for Sparagmos metric 2(b).

This is a measurement tool for the project's main path only: every test
is executed natively with the reference rustc 1.100.0 and through
`sparagmos-driver run` (Stable MIR driver -> sparagmos-mir ->
sparagmos-interp), and the two results are compared under the pass
definition in docs/COVERAGE.md (stdout and exit code must agree; for
run-fail tests the exit code must also match the expected failure
status). Nothing here is a second execution path: the Sparagmos side is
always the driver binary's `run` mode.

Input is the per-test classification produced by
scripts/rustc-subset-survey.py (--json). Stage 1 covers tests the
survey classified as nature="exec" in the ui / ui-fulldeps suites
(the run-pass / run-fail / run-crash population) that are either
single-file or use auxiliary crates: `//@ aux-build:` /
`//@ aux-crate:` directives are resolved transitively (an auxiliary
file's own directives included), and the resulting crates are built
by the native rustc for the oracle side and passed to the driver
with `--aux` for the Sparagmos side, in dependency order. A
`//@ proc-macro:` directive (in a test or an auxiliary file) names
a crate built as a proc-macro crate, and each auxiliary crate is
built with the edition its own `//@ edition:` declares when it
declares one, as compiletest builds them; both facts travel to the
driver as `--aux` attributes (`;crate-type=`, `;edition=`).
The native compilations receive the environment compiletest
gives them: `-C prefer-dynamic` unless the compilation's own
properties opt out with `//@ no-prefer-dynamic:`, the view's
`//@ rustc-env:` pairs on the test's compilation (each
auxiliary build takes its own file's), the `//@ proc-macro:`
crates as `--extern` arguments on both sides, and the
`-Lnative` directory holding the subset-built
librust_test_helpers.a. Stage 2 covers the nature="compile-check" population:
each test is run through the driver's convert-only mode and the
outcome is compared with the expectation compiletest derives from
the test's directives (see the stage-2 section below). Tests with
auxiliary crates are attempted in stage 2 the same way: the aux
crates are resolved and forwarded with `--aux`, and conversion
completeness is judged on the target crate's own functions, so
auxiliary bodies do not count against it. Everything
else in the denominator is reported as not attempted, with a reason;
unattempted and failed tests alike count against the overall
metric-2(b) rate, whose denominator is always the full subset
(24,937 at the 1.100.0 tag).

A test is *attempted* only when both sides can be run faithfully:

- its header carries a run-pass / run-fail directive
  (run-crash, stdin and add-core-stubs are recorded as
  not-attempted reasons in stage 1; `//@ run-flags:` are
  forwarded to both runs -- as the native run's argv and as the
  driver's `--guest-arg` options -- and `//@ exec-env:` pairs
  are applied to both runs' environments, the way compiletest
  applies them, with `//@ unset-exec-env:` names removed; a
  revision whose directive view carries no run mode is judged
  as a compile check under compiletest's per-revision default
  (ui tests default to check-fail), exactly as stage 2 judges
  compile-check views;
  a test's `compile-flags` are honoured in both stages by forwarding
  them (with compiletest's {{...}} variable expansion applied to
  the values) to the native rustc and, via the driver's
  `--rustc-arg`, to the driver's embedded rustc — the position
  compiletest gives them;
  revision tests are judged once per revision in both stages --
  with the revision's directive view, its compile-flags, and the
  `--cfg <revision>` and `--check-cfg` arguments compiletest
  adds -- and the record passes only when every revision that
  runs passes);
- no `ignore-*` condition matches this host and every `only-*`
  condition is satisfied (otherwise the test is out of environment
  for x86_64-unknown-linux-gnu and is recorded as such);
- every `needs-*` condition is evaluated the way compiletest
  evaluates it for the reference toolchain (compiletest's own
  target rules, queries to the pinned rustc, and the release
  build configuration); a condition that holds lets the test
  run, one that fails makes the test out of environment
  (compiletest would ignore it), and one that cannot be decided
  is recorded as unevaluatable (not attempted);
- the native compile succeeds and the native run's exit code matches
  the mode's expectation (otherwise the native oracle itself does not
  behave as the test expects, recorded with the compiler output).

Both sides are compiled/run with RUSTC_BOOTSTRAP=1 so that tests
using unstable features behave the same way they do under
compiletest (which runs against a development compiler). The native
compile uses `--edition` from the test's `//@ edition:` directive,
defaulting to 2015 (compiletest passes no requested edition and
bootstrap passes it none, so a test without a directive is compiled
with no `--edition` flag and rustc's own default applies), and
opt-level 0, matching the harness's corpus oracle.

With `--jit`, the same stages judge metric 4 — the same
denominator (24,937) on the JIT path — instead of metric 2(b):
stage 1 executes each exec record through the driver's `run-jit`
mode (the same conversion and collected program, executed by
`sparagmos-jit` instead of the interpreter) and compares stdout
and exit code with the native run under exactly the stage-1
rules; stage 2 runs the driver's `jit-compile` mode and requires,
in addition to stage 2's conversion verdict, that the JIT
lowering of the whole converted program succeed on a
success-expecting test (see judge_jit_compile). A named JIT
unsupported is a non-pass, never a pass. Records the JIT harness
has no attempt for at all (nature procedure / exec-debugger /
rustdoc, and everything else outside the two stages) are carried
through as not attempted and count against the metric-4 rate,
exactly as they count against metric 2(b).

Usage:
    python3 scripts/rustc-subset-run.py \
        --survey-json subset-survey.json \
        --rust-dir ~/rust-1.100.0-survey \
        --driver-bin crates/sparagmos-driver/target/debug/sparagmos-driver \
        --out subset-run-results.json [--jobs 48] [--limit N]
        [--stages 1|2|3|both|all] [--stage2-limit N] [--stage3-limit N] [--jit]
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import NamedTuple, Optional

TOOLCHAIN = "1.100.0"
DENOMINATOR = 24937  # metric 2(b) denominator at tag 1.100.0 (COVERAGE.md)

NATIVE_COMPILE_TIMEOUT = 60
NATIVE_RUN_TIMEOUT = 30
DRIVER_TIMEOUT = 60

# ---------------------------------------------------------------------------
# Header directive parsing
# ---------------------------------------------------------------------------

# compiletest's directive parser (directives/line.rs) trims
# whitespace after the `//@` prefix before looking for the optional
# `[revision]` scope, so `//@ [lib] compile-flags: ...` and
# `//@[lib] compile-flags: ...` are the same directive. The scope
# group must therefore allow whitespace before the bracket: without
# it, space-spelled scoped directives were dropped wholesale (their
# absence was long masked in stage 2 by the driver's crate-type pin,
# which supplied e.g. the `lib` shape a dropped
# `[lib] compile-flags: --crate-type lib` would have named).
DIRECTIVE_RE = re.compile(
    r"^\s*//@\s*(?:\[([^\]]+)\])?\s*([a-zA-Z][a-zA-Z0-9_-]*)"
    r"(?:\s*:\s*(.*?)|\s*\((.*)\)|\s+.*)?\s*$")

# The name class includes `_`: a directive name in compiletest
# includes its only-/ignore- suffix, and suffixes contain underscores
# (the x86_64 arch name, target triples such as
# x86_64-unknown-linux-gnu, rustc_abi-* conditions). Without it,
# `//@ only-x86_64-unknown-linux-gnu` and `//@ ignore-x86_64` lines
# failed to match and were silently dropped.
#
# The last alternative above is a trailing remark: compiletest
# ends the directive name at the first colon or space and treats
# the rest of the line as a remark, so a line such as
# `//@ build-pass (FIXME(62277): could be check-pass?` (unbalanced
# parenthesis) still declares the directive. Without it the whole
# line failed to match and the directive was silently dropped.

RUN_MODES = ("run-pass", "run-fail", "run-crash")


class Header:
    """Parsed `//@` directives of one test file (unscoped + scoped)."""

    def __init__(self, text):
        self.unscoped = {}   # name -> [values]
        self.scoped = {}     # (revision, name) -> [values]
        self.names = set()   # every directive name seen, scoped or not
        for line in text.splitlines():
            m = DIRECTIVE_RE.match(line)
            if not m:
                continue
            rev, name = m.group(1), m.group(2)
            value = m.group(3) if m.group(3) is not None                 else (m.group(4) or "")
            self.names.add(name)
            if rev is None:
                self.unscoped.setdefault(name, []).append(value)
            else:
                self.scoped.setdefault((rev, name), []).append(value)

    def has(self, name):
        return name in self.unscoped

    def values(self, name):
        return self.unscoped.get(name, [])

    def revisions(self):
        """Revision names declared by `//@ revisions:`, in order."""
        out = []
        for v in self.values("revisions"):
            out.extend(v.split())
        return out

    def view(self, revision):
        """The directives visible for one revision (None = unscoped)."""
        return HeaderView(self, revision)


class HeaderView:
    """The directive view of a Header for a single test revision.

    Mirrors compiletest's `applies_to_test_revision`: an unscoped
    directive applies to every revision, and a directive scoped with
    `//@[rev]` applies only to revision `rev`. With revision=None
    the view is the unscoped directives alone (scoped directives for
    undeclared revisions never apply, as in compiletest). `has`,
    `values` and `names` match the Header interface used by the
    verdict logic; `values` lists unscoped values first, then the
    revision's scoped values (compiletest processes lines in file
    order, so a scoped value also wins for last-one-wins flags).
    """

    def __init__(self, header, revision):
        self.header = header
        self.revision = revision
        names = set(header.unscoped)
        if revision is not None:
            for (rev, name) in header.scoped:
                if rev == revision:
                    names.add(name)
        self.names = names

    def has(self, name):
        if name in self.header.unscoped:
            return True
        return (self.revision is not None
                and (self.revision, name) in self.header.scoped)

    def values(self, name):
        vals = list(self.header.unscoped.get(name, []))
        if self.revision is not None:
            vals += self.header.scoped.get((self.revision, name), [])
        return vals


def run_mode(header):
    """The unscoped run mode of a test, or None."""
    for mode in RUN_MODES:
        if header.has(mode):
            return mode
    return None


_SYSROOT_BASE = None


def sysroot_base():
    """The reference toolchain's sysroot directory."""
    global _SYSROOT_BASE
    if _SYSROOT_BASE is None:
        try:
            out = subprocess.run(
                rustc_cmd() + ["--print", "sysroot"],
                capture_output=True, text=True, timeout=30)
            _SYSROOT_BASE = out.stdout.strip()
        except Exception:
            _SYSROOT_BASE = ""
    return _SYSROOT_BASE


def suite_root(path):
    """The suite root directory of a test file (compiletest's
    src-base for the test): the nearest ancestor directory whose
    parent is `tests`."""
    for parent in Path(path).parents:
        if parent.parent.name == "tests":
            return str(parent)
    return str(Path(path).parent)


def expand_variables(value, src_base, build_base):
    """compiletest's `{{...}}` substitutions in directive values.

    Mirrors expand_variables in compiletest's directives.rs (it
    expands every name-value directive value): {{cwd}} is the
    directory compiletest runs from -- here the runner's working
    directory; {{src-base}} the test's suite root; {{build-base}}
    the attempt's scratch directory; {{sysroot-base}} the
    reference toolchain's sysroot; {{target}} the host target;
    {{target-linker}} empty (no custom linker is configured on
    this host); {{rust-src-base}} the sysroot's rust source tree.
    """
    if "{{" not in value:
        return value
    value = value.replace("{{cwd}}", os.getcwd())
    value = value.replace("{{src-base}}", src_base)
    value = value.replace("{{build-base}}", build_base)
    value = value.replace("{{sysroot-base}}", sysroot_base())
    value = value.replace("{{target-linker}}", "")
    value = value.replace("{{target}}", HOST_TARGET)
    value = value.replace(
        "{{rust-src-base}}",
        str(Path(sysroot_base()) / "lib" / "rustlib" / "src" / "rust"))
    return value


def parse_env_pairs(values):
    """(name, value) pairs from env directive values, read the
    way compiletest's parse_env reads them: split at the first
    `=`, the name trimmed, a missing `=` means an empty value."""
    pairs = []
    for v in values:
        name, sep, val = v.partition("=")
        pairs.append((name.strip(), val if sep else ""))
    return pairs


def view_env(view, set_name, unset_name, src_base, build_base):
    """The environment a directive view prescribes: (pairs, unset).

    `pairs` is a dict of the view's `set_name` directives
    (`rustc-env` / `exec-env`) with compiletest's variable
    expansion applied to the values; `unset` is the list of
    variable names the view's `unset_name` directives remove
    (each value is one variable name, trimmed).
    """
    pairs = {}
    for name, val in parse_env_pairs(view.values(set_name)):
        pairs[name] = expand_variables(val, src_base, build_base)
    unset = [v.strip() for v in view.values(unset_name) if v.strip()]
    return pairs, unset


def view_run_flags(view, src_base, build_base):
    """The guest program arguments a directive view prescribes:
    its `run-flags` values, variable-expanded and split the way
    compiletest splits them."""
    flags = []
    for v in view.values("run-flags"):
        flags.extend(split_flags(expand_variables(v, src_base,
                                                  build_base)))
    return flags


# ---------------------------------------------------------------------------
# compile-flags handling
# ---------------------------------------------------------------------------

HOST_TARGET = "x86_64-unknown-linux-gnu"

_INSTALLED_TARGETS = None

_KNOWN_TARGETS = None


def _targets_in_sysroot(sysroot):
    """The triples with a std installed under a sysroot: the
    subdirectories of lib/rustlib that hold a lib directory (one per
    installed target, the host target included)."""
    rustlib = Path(sysroot) / "lib" / "rustlib"
    try:
        return {p.name for p in rustlib.iterdir()
                if p.is_dir() and (p / "lib").is_dir()}
    except OSError:
        return set()


def installed_targets():
    """The triples with a std installed in the pinned toolchain's
    sysroot. Computed once per process; an empty set when the
    sysroot cannot be queried, which keeps --target screening
    conservative."""
    global _INSTALLED_TARGETS
    if _INSTALLED_TARGETS is None:
        try:
            out = subprocess.run(
                rustc_cmd() + ["--print", "sysroot"],
                capture_output=True, text=True, timeout=30)
            _INSTALLED_TARGETS = _targets_in_sysroot(out.stdout.strip())
        except Exception:
            _INSTALLED_TARGETS = set()
    return _INSTALLED_TARGETS


def split_flags(flags):
    """Split a compile-flags value the way compiletest does.

    compiletest's split_flags: single-quoted segments keep their
    spaces; everything else is split on whitespace.
    """
    out = []
    for i, seg in enumerate(flags.split("'")):
        if i % 2 == 1:
            out.append(seg)
        else:
            out.extend(seg.split())
    return out


def compile_flags(view, src_base=None, build_base=None):
    """All compile-flags tokens visible in a directive view.

    compiletest expands {{...}} variables in every directive value
    before use (expand_variables in directives.rs), compile-flags
    included, so when the caller supplies the attempt's src-base
    and build-base the values are expanded the same way here.
    Callers that only compare flag sets may omit them and get the
    raw tokens.
    """
    flags = []
    for v in view.values("compile-flags"):
        if src_base is not None and build_base is not None:
            v = expand_variables(v, src_base, build_base)
        flags.extend(split_flags(v))
    return flags


def known_targets():
    """Every target triple the pinned toolchain knows (its
    `--print target-list`): the built-in targets, whether or not
    a std component is installed for them. Computed once per
    process; an empty set when the query fails, which callers
    treat as "unknown" so --target screening stays conservative.
    """
    global _KNOWN_TARGETS
    if _KNOWN_TARGETS is None:
        try:
            out = subprocess.run(
                rustc_cmd() + ["--print", "target-list"],
                capture_output=True, text=True, timeout=30)
            _KNOWN_TARGETS = {ln.strip() for ln in
                              out.stdout.splitlines() if ln.strip()}
        except Exception:
            _KNOWN_TARGETS = set()
    return _KNOWN_TARGETS


def screen_flags(flags, targets=None, known=None, expectation=None):
    """Why this flag set cannot be honoured here, or None.

    The driver forwards flags verbatim to its embedded rustc (and
    stage 1 to the native rustc), so almost everything compiletest
    accepts works unchanged. The exceptions are flags that would
    replace the compilation itself or contradict the driver's fixed
    invocation (its pinned sysroot and scratch output directory).

    `targets` is the set of triples a --target flag may name. The
    default (None) is every target installed in the pinned sysroot:
    conversion for an installed target is judged exactly like the
    host target's, and no foreign code is executed. Stage 1 passes
    just the host target instead, because it executes the program
    and a foreign binary cannot run on this host.

    Two --target values are not refused even when they name no
    installed target:

    - a target-spec file (a value ending in `.json`): rustc loads
      the spec itself, and no std component takes part in loading
      it, so the compilation the test asks for can be attempted
      as it stands;
    - when `known` (the toolchain's target list) is supplied and
      `expectation` is "failure", a triple that is not a target
      of the toolchain at all: rustc rejects such a specification
      while loading it, before any std is needed, and that
      rejection is exactly the failure the test asserts, so the
      driver's embedded rustc reproduces it in the attempt.

    A known triple with no installed std stays refused for every
    outcome: there the only failure available is the environment's
    (E0463, can't find crate for `std`), not the test's, and
    judging it would manufacture passes.
    """
    if targets is None:
        targets = installed_targets()
    i = 0
    while i < len(flags):
        t = flags[i]
        nxt = flags[i + 1] if i + 1 < len(flags) else None
        if t == "--target" or t.startswith("--target="):
            tgt = t[len("--target="):] if t.startswith("--target=") else nxt
            if tgt is not None and tgt.endswith(".json"):
                pass  # a target-spec file, not a triple
            elif tgt is not None and tgt not in targets:
                if known and tgt not in known:
                    if expectation != "failure":
                        return (f"compile-flags specify --target {tgt} "
                                "(that is not a target of this "
                                "toolchain)")
                    # expectation == "failure": the specification
                    # itself is rejected; fall through and attempt.
                elif tgt in installed_targets():
                    return (f"compile-flags specify --target {tgt} "
                            "(that target is installed, but this "
                            "stage executes the program and only the "
                            "host target can run here)")
                else:
                    return (f"compile-flags specify --target {tgt} "
                            "(that target is not installed on this host)")
        elif t == "--print" or t.startswith("--print="):
            return ("compile-flags request --print output (rustc "
                    "prints target information instead of compiling)")
        elif t == "--sysroot" or t.startswith("--sysroot="):
            return ("compile-flags specify --sysroot (the driver "
                    "pins the reference toolchain sysroot)")
        elif t in ("-o", "--out-dir") or t.startswith("--out-dir="):
            return ("compile-flags specify an output location (the "
                    "driver uses its own scratch output directory)")
        elif t.startswith("-Zunpretty") or (t == "-Z" and nxt is not None
                                             and nxt.startswith("unpretty")):
            return ("compile-flags request -Zunpretty output (rustc "
                    "stops after printing, so no conversion is "
                    "possible)")
        i += 1
    return None


def has_unpretty(flags):
    """Whether the flag set carries a -Zunpretty request, in
    either the joined (`-Zunpretty=...`) or split (`-Z`,
    `unpretty=...`) spelling."""
    i = 0
    while i < len(flags):
        t = flags[i]
        if t.startswith("-Zunpretty"):
            return True
        if t == "-Z" and i + 1 < len(flags) \
                and flags[i + 1].startswith("unpretty"):
            return True
        i += 1
    return False


def strip_unpretty(flags):
    """The flag tokens with every -Zunpretty request removed, in
    both spellings. Everything else is kept, in order."""
    out = []
    i = 0
    while i < len(flags):
        t = flags[i]
        if t == "-Z" and i + 1 < len(flags) \
                and flags[i + 1].startswith("unpretty"):
            i += 2
            continue
        if t.startswith("-Zunpretty"):
            i += 1
            continue
        out.append(t)
        i += 1
    return out


def stdout_snapshot_checked(path, revision):
    """Whether compiletest would compare the compiler's stdout
    for one directive view against a `.stdout` snapshot: the
    snapshot file for the view exists next to the test (the
    unscoped file, or the revision's own)."""
    stem = path.name[:-3] if path.name.endswith(".rs") else path.name
    parent = str(path.parent)
    if parent not in _DIR_LISTING_CACHE:
        try:
            _DIR_LISTING_CACHE[parent] = [p.name for p in
                                          path.parent.iterdir()]
        except OSError:
            _DIR_LISTING_CACHE[parent] = []
    names = _DIR_LISTING_CACHE[parent]
    if f"{stem}.stdout" in names:
        return True
    return (revision is not None
            and f"{stem}.{revision}.stdout" in names)


def unpretty_unchecked(view, path, revision, expectation):
    """Whether a view's -Zunpretty output cannot take part in its
    verdict, so the attempt may drop the printing request and
    convert the compilation itself.

    -Zunpretty makes rustc print a dump and stop, so no conversion
    is possible while the flag stands (screen_flags refuses it).
    The dump is the test's assertion whenever compiletest compares
    it, and those tests stay refused. It cannot take part in the
    verdict when the view declares `//@ dont-check-compiler-stdout`
    (the dump is compiler stdout, explicitly unchecked), or when
    the expected outcome is a compilation failure and no `.stdout`
    snapshot is checked for the view: the asserted failure is a
    diagnostic the compilation produces before -- or instead of --
    any dump, and an unchecked dump decides nothing. Crash
    expectations are excluded: their verdict is the crash itself,
    which a conversion does not reproduce.
    """
    if expectation not in ("success", "failure"):
        return False
    if view.has("dont-check-compiler-stdout"):
        return True
    if expectation != "failure":
        return False
    if view.has("check-stdout"):
        return False
    return not stdout_snapshot_checked(path, revision)


def normalize_revision(revision):
    """The cfg name compiletest derives from a revision name."""
    return revision.lower().replace("-", "_")


def revision_args(view, revision, all_revisions):
    """The rustc arguments compiletest adds for one revision run:
    `--cfg <revision>` plus, unless the test opts out with
    `no-auto-check-cfg`, a `--check-cfg` naming every revision."""
    args = ["--cfg", normalize_revision(revision)]
    if not view.has("no-auto-check-cfg"):
        check = "cfg(test,FALSE"
        for rev in all_revisions:
            check += "," + normalize_revision(rev)
        check += ")"
        args += ["--check-cfg", check]
    return args


# ---------------------------------------------------------------------------
# Environment applicability (host: x86_64-unknown-linux-gnu)
# ---------------------------------------------------------------------------

# cfg atoms that are true for the measurement host. Directive suffixes
# are matched against this set the way compiletest matches them
# against the target's cfg list (an `ignore-<x>` skips the test when
# <x> matches; an `only-<x>` keeps the test only when <x> matches).
HOST_TRUE_CFGS = {
    "linux", "unix", "x86_64", "x86_64-unknown-linux-gnu",
    "target_os=\"linux\"", "target_arch=\"x86_64\"",
    "target_family=\"unix\"", "target_env=\"gnu\"",
    "target_vendor=\"unknown\"", "target_pointer_width=\"64\"",
    "target_endian=\"little\"", "64bit", "gnu",
    # compiletest also registers the target's binary format as a
    # condition (directives/cfg.rs: `elf` when binary_format is
    # "elf"); x86_64-unknown-linux-gnu binaries are ELF.
    "elf",
}

# compiletest builds the channel conditions in the same condition set
# (directives/cfg.rs): exactly one of stable/beta/nightly is true, for
# the channel of the compiler under test, and a "dev" build counts as
# nightly. The channel is the compiler's own declaration -- the suffix
# of its version string. The pinned reference toolchain is the
# official 1.100.0 release build, whose version string
# ("rustc 1.100.0 (...)") carries no -nightly/-beta/-dev suffix, so
# its channel is stable. RUSTC_BOOTSTRAP=1, which both sides run
# under, unlocks unstable features but does not change the declared
# channel, so it plays no part here.
RELEASE_CHANNEL = "stable"

# ---------------------------------------------------------------------------
# needs-* conditions (compiletest's model, mirrored)
# ---------------------------------------------------------------------------
#
# compiletest decides `//@ needs-*` directives in
# src/tools/compiletest/src/directives/needs.rs from three kinds of
# sources, and the runner mirrors each kind the same way:
#
# * rules in compiletest's own code over the target identity
#   (has_threads, has_subprocess_support, has_asm_support,
#   find_dlltool) -- applied to the host target below;
# * queries to the compiler under test (its target-spec JSON, its
#   supported crate types, its cfg list, its --print=backend-*
#   answers, whether it can codegen for a target) and the contents
#   of its sysroot -- ToolchainFacts performs exactly those queries
#   against the pinned toolchain;
# * build-configuration flags bootstrap passes to compiletest
#   (--has-enzyme, --with-rustc-debug-assertions, ...). The
#   reference toolchain is the official 1.100.0 release build,
#   whose configuration is bootstrap's release configuration: the
#   opt-in options in NEEDS_RELEASE_CONFIG are off (llvm.enzyme /
#   llvm.offload are enabled only by dedicated experimental
#   builds; rust / std debug assertions and layout randomization
#   are opt-ins the release build does not take). They are
#   constants here, the same status HOST_TRUE_CFGS has for cfg
#   atoms.
#
# A condition that evaluates to False means compiletest would
# ignore the test, so the record is out of environment, exactly
# like an ignore-* match. A condition that cannot be decided
# stays unevaluatable; nothing is guessed.

_HOST_PARTS = HOST_TARGET.split("-")
_HOST_ARCH = _HOST_PARTS[0]
_HOST_OS = _HOST_PARTS[2] if len(_HOST_PARTS) > 2 else ""
_HOST_ENV = _HOST_PARTS[3] if len(_HOST_PARTS) > 3 else ""

# compiletest's has_asm_support arch list (common.rs).
ASM_SUPPORTED_ARCHS = {
    "x86", "x86_64", "arm", "aarch64", "arm64ec", "riscv32",
    "riscv64", "loongarch32", "loongarch64", "s390x",
}

# needs-* conditions decided by compiletest's target rules and by
# the release build configuration, for the host target. True means
# the condition holds here; False means compiletest would ignore
# the test on this host.
NEEDS_HOST_RULES = {
    "needs-run-enabled": True,    # we do run programs
    "needs-unwind": True,         # the reference toolchain unwinds
    # has_threads: only wasm targets without `-threads` lack them.
    "needs-threads": not HOST_TARGET.startswith("wasm"),
    # has_subprocess_support: denied only for sgx / wasm / emscripten.
    "needs-subprocess": not (_HOST_ENV == "sgx"
                             or _HOST_ARCH in ("wasm32", "wasm64")
                             or _HOST_OS == "emscripten"),
    "needs-asm-support": _HOST_ARCH in ASM_SUPPORTED_ARCHS,
    # find_dlltool: dlltool exists only for *-windows-gnu targets.
    "needs-dlltool": _HOST_OS == "windows" and _HOST_ENV == "gnu",
    # Release build configuration (see the section comment).
    "needs-deterministic-layouts": True,   # layouts are not randomized
    "needs-rustc-debug-assertions": False,
    "needs-std-debug-assertions": False,
    "needs-enzyme": False,
    "needs-offload": False,
    # compiletest runner configuration: this runner never runs
    # tests under wasmtime or clang, and always has symlinks and
    # std for the host target.
    "needs-force-clang-based-tests": False,
    "needs-wasmtime": False,
    "needs-symlink": True,
    "needs-target-std": True,
}

# needs-sanitizer-* directive -> the name in the target spec's
# `supported-sanitizers` list (compiletest's Sanitizer enum, in
# kebab-case as the spec JSON spells it).
SANITIZER_BY_DIRECTIVE = {
    "needs-sanitizer-address": "address",
    "needs-sanitizer-cfi": "cfi",
    "needs-sanitizer-dataflow": "dataflow",
    "needs-sanitizer-kcfi": "kcfi",
    "needs-sanitizer-kasan": "kernel-address",
    "needs-sanitizer-khwasan": "kernel-hwaddress",
    "needs-sanitizer-leak": "leak",
    "needs-sanitizer-memory": "memory",
    "needs-sanitizer-thread": "thread",
    "needs-sanitizer-hwaddress": "hwaddress",
    "needs-sanitizer-memtag": "memtag",
    "needs-sanitizer-realtime": "realtime",
    "needs-sanitizer-shadow-call-stack": "shadow-call-stack",
    "needs-sanitizer-safestack": "safestack",
}

KNOWN_CRATE_TYPES = {
    "bin", "cdylib", "dylib", "lib", "proc-macro", "rlib", "staticlib",
}
KNOWN_ATOMIC_WIDTHS = {"8", "16", "32", "64", "128", "ptr"}
KNOWN_CODEGEN_BACKENDS = {"llvm", "cranelift", "gcc"}

# A `#![no_core]` crate with the lang items a bare function needs.
# Compiling it to an object for a target exercises that target's
# LLVM backend end to end, which is what an LLVM component's
# presence in the build means; compiletest gets the component list
# from the LLVM build, the runner probes the built compiler.
_PROBE_SOURCE = """\
#![feature(no_core, lang_items)]
#![no_core]
#[lang = "pointee_sized"]
pub trait PointeeSized {}
#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}
#[lang = "sized"]
pub trait Sized: MetaSized {}
#[lang = "copy"]
pub trait Copy {}
#[lang = "freeze"]
pub trait Freeze {}
#[lang = "sync"]
pub trait Sync {}
#[no_mangle]
pub fn f(x: u32) -> u32 { x }
"""

# LLVM component -> (representative target, extra rustc args) for
# the codegen probe. avr and amdgpu targets require an explicit
# -C target-cpu before they codegen anything.
LLVM_COMPONENT_PROBES = {
    "x86": ("x86_64-unknown-linux-gnu", ()),
    "aarch64": ("aarch64-unknown-linux-gnu", ()),
    "arm": ("armv7-unknown-linux-gnueabihf", ()),
    "mips": ("mips64-unknown-linux-gnuabi64", ()),
    "riscv": ("riscv64gc-unknown-linux-gnu", ()),
    "powerpc": ("powerpc64le-unknown-linux-gnu", ()),
    "sparc": ("sparc64-unknown-linux-gnu", ()),
    "systemz": ("s390x-unknown-linux-gnu", ()),
    "loongarch": ("loongarch64-unknown-linux-gnu", ()),
    "webassembly": ("wasm32-unknown-unknown", ()),
    "bpf": ("bpfeb-unknown-none", ()),
    "nvptx": ("nvptx64-nvidia-cuda", ()),
    "avr": ("avr-none", ("-C", "target-cpu=atmega328p")),
    "amdgpu": ("amdgcn-amd-amdhsa", ("-C", "target-cpu=gfx900")),
    "m68k": ("m68k-unknown-linux-gnu", ()),
    "msp430": ("msp430-none-elf", ()),
    "csky": ("csky-unknown-linux-gnuabiv2", ()),
    "hexagon": ("hexagon-unknown-linux-musl", ()),
    "xtensa": ("xtensa-esp32-espidf", ()),
}

FACTS_QUERY_TIMEOUT = 120
PROBE_TIMEOUT = 60


class ToolchainFacts:
    """Facts about the pinned reference toolchain, obtained the way
    compiletest obtains them: by querying the compiler and
    inspecting its sysroot. Every accessor returns None when its
    query fails, so the caller can leave a condition unevaluatable
    instead of guessing. Results are computed lazily and cached;
    the runner builds one instance per process (host_facts()).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cache = {}

    def _once(self, key, compute):
        # Like minicore_rlib: the lock guards the cache only, never
        # the computation -- accessors call each other (sanitizers
        # reads target_spec, probes read sysroot), so holding the
        # lock across compute() would self-deadlock. A fact may be
        # computed twice under a race; the results are identical.
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        result = compute()
        with self._lock:
            self._cache.setdefault(key, result)
            return self._cache[key]

    def _query(self, args, timeout=FACTS_QUERY_TIMEOUT):
        """stdout of a rustc query against the pinned toolchain, or
        None when the query fails."""
        try:
            res = run_cmd(rustc_cmd() + list(args), timeout=timeout,
                          env_extra=BOOTSTRAP_ENV)
        except Exception:
            return None
        if res["timeout"] or res["exit_code"] != 0:
            return None
        return res["stdout"]

    def sysroot(self):
        def compute():
            out = self._query(["--print", "sysroot"], timeout=30)
            return Path(out.strip()) if out else None
        return self._once("sysroot", compute)

    def target_lib_dir(self):
        def compute():
            root = self.sysroot()
            if root is None:
                return None
            return root / "lib" / "rustlib" / HOST_TARGET / "lib"
        return self._once("target_lib_dir", compute)

    def _target_lib_listing(self):
        def compute():
            d = self.target_lib_dir()
            if d is None:
                return None
            try:
                return set(os.listdir(d))
            except OSError:
                return None
        return self._once("target_lib_listing", compute)

    def target_spec(self):
        """The host entry of `--print=all-target-specs-json`, the
        same document compiletest builds its TargetCfg from."""
        def compute():
            out = self._query(["-Zunstable-options",
                               "--print=all-target-specs-json"])
            if out is None:
                return None
            try:
                return json.loads(out).get(HOST_TARGET)
            except ValueError:
                return None
        return self._once("target_spec", compute)

    def _spec_bool(self, key):
        spec = self.target_spec()
        if spec is None:
            return None
        return bool(spec.get(key, False))

    def sanitizers(self):
        """The target's supported sanitizers (target-spec field,
        serde-defaulted to empty in compiletest)."""
        def compute():
            spec = self.target_spec()
            if spec is None:
                return None
            return set(spec.get("supported-sanitizers", []))
        return self._once("sanitizers", compute)

    def xray(self):
        return self._spec_bool("supports-xray")

    def dynamic_linking(self):
        return self._spec_bool("dynamic-linking")

    def relocation_model(self):
        def compute():
            spec = self.target_spec()
            if spec is None:
                return None
            # compiletest's serde default when the field is absent.
            return spec.get("relocation-model", "pic")
        return self._once("relocation_model", compute)

    def supported_crate_types(self):
        def compute():
            out = self._query(["--target", HOST_TARGET,
                               "--print=supported-crate-types",
                               "-Zunstable-options"])
            if out is None:
                return None
            return {ln.strip() for ln in out.splitlines() if ln.strip()}
        return self._once("supported_crate_types", compute)

    def target_has_atomic(self):
        def compute():
            out = self._query(["--target", HOST_TARGET,
                               "--print", "cfg"])
            if out is None:
                return None
            widths = set()
            for ln in out.splitlines():
                m = re.match(r'^target_has_atomic="([^"]+)"$', ln.strip())
                if m:
                    widths.add(m.group(1))
            return widths
        return self._once("target_has_atomic", compute)

    def _backend_print_bool(self, request):
        out = self._query(["--target", HOST_TARGET, "-Zunstable-options",
                           f"--print={request}"], timeout=60)
        if out is None:
            return None
        v = out.strip()
        if v == "true":
            return True
        if v == "false":
            return False
        return None

    def has_mnemonic(self, mnemonic):
        return self._backend_print_bool(f"backend-has-mnemonic:{mnemonic}")

    def llvm_zstd(self):
        return self._backend_print_bool("backend-has-zstd")

    def git_hash(self):
        def compute():
            try:
                res = run_cmd(rustc_cmd() + ["-vV"], timeout=30)
            except Exception:
                return None
            if res["timeout"] or res["exit_code"] != 0:
                return None
            return "commit-hash:" in res["stdout"]
        return self._once("git_hash", compute)

    def profiler_runtime(self):
        """Whether the profiler runtime is shipped for the host
        target (bootstrap's --profiler-runtime means the runtime
        was built into this sysroot)."""
        def compute():
            names = self._target_lib_listing()
            if names is None:
                return None
            return any(n.startswith("libprofiler_builtins")
                       and n.endswith(".rlib") for n in names)
        return self._once("profiler_runtime", compute)

    def sanitizer_runtime_present(self):
        """Whether sanitizer runtime archives are shipped for the
        host target (bootstrap sets RUSTC_SANITIZER_SUPPORT when
        sanitizers are enabled for the target, which is what puts
        these archives and the spec's sanitizer list in place)."""
        def compute():
            names = self._target_lib_listing()
            if names is None:
                return None
            return any("_rt." in n and n.endswith(".a") for n in names)
        return self._once("sanitizer_runtime_present", compute)

    def rust_lld(self):
        def compute():
            root = self.sysroot()
            if root is None:
                return None
            return (root / "lib" / "rustlib" / HOST_TARGET / "bin"
                    / "rust-lld").exists()
        return self._once("rust_lld", compute)

    def _probe_codegen(self, target, extra_args, emit):
        """Compile the no_core probe for `target`, asking for the
        given --emit kind. True when an output file was produced."""
        key = ("probe", target, tuple(extra_args), emit)
        def compute():
            try:
                with tempfile.TemporaryDirectory(
                        prefix="subset-probe-") as td:
                    src = Path(td) / "probe.rs"
                    src.write_text(_PROBE_SOURCE)
                    out = Path(td) / "probe.out"
                    res = run_cmd(
                        rustc_cmd() + [str(src), "--target", target,
                                       "--crate-type", "lib",
                                       "--emit", emit, "-o", str(out),
                                       *extra_args],
                        timeout=PROBE_TIMEOUT, env_extra=BOOTSTRAP_ENV)
                    return (not res["timeout"] and res["exit_code"] == 0
                            and out.exists())
            except Exception:
                return False
        return self._once(key, compute)

    def llvm_component_present(self, component):
        """Whether an LLVM target component is part of this build,
        probed by code-generating for a representative target.
        None when the component has no probe (unknown name) or the
        probe machinery itself fails on the host target."""
        spec = LLVM_COMPONENT_PROBES.get(component)
        if spec is None:
            return None
        host_target, host_extra = LLVM_COMPONENT_PROBES["x86"]
        if not self._probe_codegen(host_target, host_extra, "obj"):
            return None
        target, extra = spec
        return self._probe_codegen(target, extra, "obj")

    def default_codegen_backend(self):
        """The default codegen backend, probed: `--emit llvm-ir`
        only succeeds when the default backend is LLVM. None when
        the probe fails (the default is then some other backend,
        which this runner does not distinguish)."""
        def compute():
            if self._probe_codegen(HOST_TARGET, (), "llvm-ir"):
                return "llvm"
            return None
        return self._once("default_codegen_backend", compute)


_HOST_FACTS = None


def host_facts():
    """The process-wide ToolchainFacts, or None before main()
    initialises it (unit tests then see probed conditions as
    unevaluatable, never as guessed)."""
    return _HOST_FACTS


def eval_needs(name, view, facts):
    """Evaluate one `needs-*` directive the way compiletest does
    for the reference toolchain.

    Returns True (the condition holds; the test runs), False (the
    condition fails; compiletest would ignore the test), or None
    (the condition cannot be decided here). `view` supplies the
    directive values for the value-carrying directives; `facts` is
    a ToolchainFacts or None.
    """
    if name in NEEDS_HOST_RULES:
        return NEEDS_HOST_RULES[name]
    if name in SANITIZER_BY_DIRECTIVE:
        if facts is None:
            return None
        sanitizers = facts.sanitizers()
        if sanitizers is None:
            return None
        return SANITIZER_BY_DIRECTIVE[name] in sanitizers
    if name == "needs-sanitizer-support":
        if facts is None:
            return None
        sanitizers = facts.sanitizers()
        runtime = facts.sanitizer_runtime_present()
        if sanitizers is None or runtime is None:
            return None
        return bool(sanitizers) and runtime
    if name == "needs-xray":
        return None if facts is None else facts.xray()
    if name == "needs-dynamic-linking":
        return None if facts is None else facts.dynamic_linking()
    if name == "needs-relocation-model-pic":
        if facts is None:
            return None
        model = facts.relocation_model()
        return None if model is None else model == "pic"
    if name == "needs-profiler-runtime":
        return None if facts is None else facts.profiler_runtime()
    if name == "needs-rust-lld":
        return None if facts is None else facts.rust_lld()
    if name == "needs-git-hash":
        return None if facts is None else facts.git_hash()
    if name == "needs-llvm-zstd":
        return None if facts is None else facts.llvm_zstd()
    if name == "needs-asm-ret":
        return None if facts is None else facts.has_mnemonic("ret")
    if name == "needs-crate-type":
        return _eval_needs_list(
            view, name, KNOWN_CRATE_TYPES,
            None if facts is None else facts.supported_crate_types())
    if name == "needs-target-has-atomic":
        return _eval_needs_list(
            view, name, KNOWN_ATOMIC_WIDTHS,
            None if facts is None else facts.target_has_atomic())
    if name == "needs-llvm-components":
        if facts is None:
            return None
        tokens = []
        for value in view.values(name):
            tokens.extend(value.split())
        if not tokens:
            return None
        for token in tokens:
            present = facts.llvm_component_present(token)
            if present is None:
                return None
            if not present:
                return False
        return True
    if name == "needs-backends":
        tokens = []
        for value in view.values(name):
            tokens.extend(t.lower() for t in value.split())
        if not tokens or any(t not in KNOWN_CODEGEN_BACKENDS
                             for t in tokens):
            return None
        if facts is None:
            return None
        default = facts.default_codegen_backend()
        if default is None:
            return None
        return default in tokens
    return None


def _eval_needs_list(view, name, known, available):
    """Shared logic of the comma-separated needs-* directives
    (needs-crate-type, needs-target-has-atomic): every listed item
    must be a known value and must be available. An unknown item
    makes the directive undecidable here (compiletest rejects it
    outright); a missing value list is undecidable too."""
    wanted = []
    for value in view.values(name):
        wanted.extend(t.strip() for t in value.split(",") if t.strip())
    if not wanted or any(t not in known for t in wanted):
        return None
    if available is None:
        return None
    return all(t in available for t in wanted)

# `ignore-*` suffixes that never match any host in compiletest's model
# (they gate on the pass mode rather than the environment); for the
# run modes measured here they do not apply.
IGNORE_NEVER_MATCH = {"ignore-pass", "ignore-check", "ignore-pretty"}


def eval_cfg_atom(atom):
    """Evaluate one cfg atom against the host. None = cannot evaluate."""
    atom = atom.strip()
    if atom in HOST_TRUE_CFGS:
        return True
    if atom in ("stable", "beta", "nightly"):
        # Channel conditions (see RELEASE_CHANNEL): exactly one is
        # true, where the bare-name rule below would make all three
        # false, which no compiletest run ever sees.
        return atom == RELEASE_CHANNEL
    if atom in ("windows", "macos", "win32", "32bit"):
        return False
    m = re.match(r'^(target_[a-z_]+)\s*=\s*"([^"]*)"$', atom)
    if m:
        return f'{m.group(1)}="{m.group(2)}"' in HOST_TRUE_CFGS
    if re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", atom):
        # A bare cfg name that is not in the host set is false for this
        # host (other OS/arch/env names, stage/channel names handled
        # by callers where relevant).
        return False
    return None


def eval_condition(cond):
    """Evaluate an ignore-/only- condition suffix.

    Returns True (matches this host), False (does not match), or None
    (cannot be evaluated mechanically here).
    """
    cond = cond.strip()
    if not cond:
        return None
    if cond in IGNORE_NEVER_MATCH:
        return False
    if cond.startswith("cfg(") and cond.endswith(")"):
        return eval_cfg_expr(cond[4:-1])
    return eval_cfg_atom(cond)


def eval_cfg_expr(expr):
    """Evaluate a cfg(...) body with all/any/not over simple atoms."""
    expr = expr.strip()
    for fn, combine in (("all", all), ("any", any)):
        if expr.startswith(fn + "(") and expr.endswith(")"):
            parts = split_top_level(expr[len(fn) + 1:-1])
            vals = [eval_cfg_expr(p) for p in parts]
            if any(v is None for v in vals):
                return None
            return combine(vals)
    if expr.startswith("not(") and expr.endswith(")"):
        v = eval_cfg_expr(expr[4:-1])
        return None if v is None else not v
    return eval_cfg_atom(expr)


def split_top_level(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def applicability(header, facts=None):
    """Decide whether a test applies to this host.

    Returns (status, reason): status is "applicable",
    "out-of-environment" (an ignore matched, an only did not, or
    a needs-* condition does not hold, so compiletest would ignore
    the test), or "unevaluatable" (a needs-/condition could not
    be decided). `facts` defaults to the process-wide toolchain
    facts (host_facts()); unit tests pass canned facts or leave
    them uninitialised, in which case probed conditions stay
    unevaluatable.
    """
    if facts is None:
        facts = host_facts()
    for value in header.values("ignore-cfg"):
        v = eval_cfg_expr(value)
        if v is True:
            return ("out-of-environment",
                    f"ignore-cfg({value}) matches this host")
        if v is None:
            return ("unevaluatable",
                    f"ignore-cfg({value}) cannot be evaluated")
    only_cfgs = header.values("only-cfg")
    if only_cfgs:
        vals = [eval_cfg_expr(v) for v in only_cfgs]
        if any(v is None for v in vals):
            return ("unevaluatable",
                    "only-cfg condition cannot be evaluated: "
                    + ", ".join(only_cfgs))
        if not any(vals):
            return ("out-of-environment",
                    "no only-cfg condition matches this host: "
                    + ", ".join(only_cfgs))
    for name in header.names:
        if name in ("ignore-cfg", "only-cfg"):
            continue
        if name.startswith("ignore-"):
            cond = name[len("ignore-"):]
            v = eval_condition(cond)
            if v is True:
                return ("out-of-environment", f"ignore-{cond} matches this host")
            if v is None:
                return ("unevaluatable", f"ignore-{cond} cannot be evaluated")
    onlys = [n[len("only-"):] for n in header.names if n.startswith("only-")]
    if onlys:
        vals = [eval_condition(c) for c in onlys]
        # compiletest decides each only-* directive on its own
        # (handle_only): one non-matching condition already ignores
        # the test, so several only-* directives are conjunctive --
        # only-aarch64 + only-linux runs nowhere but aarch64-linux.
        if any(v is False for v in vals):
            failing = [c for c, v in zip(onlys, vals) if v is False]
            return ("out-of-environment",
                    "only-* condition does not match this host: "
                    + ", ".join(failing))
        if any(v is None for v in vals):
            return ("unevaluatable",
                    "only-* condition cannot be evaluated: " + ", ".join(onlys))
    for name in sorted(header.names):
        if name.startswith("needs-"):
            v = eval_needs(name, header, facts)
            if v is True:
                continue
            if v is False:
                return ("out-of-environment",
                        f"{name} is not satisfied on this host")
            return ("unevaluatable", f"{name} cannot be verified on this host")
    return ("applicable", "")


# ---------------------------------------------------------------------------
# Verdict logic (pure; unit-tested with fixtures)
# ---------------------------------------------------------------------------

def expected_exit(mode, header):
    """Exit code the mode expects from the program run."""
    if mode == "run-pass":
        return 0
    if mode == "run-fail":
        for v in header.values("failure-status"):
            try:
                return int(v.strip())
            except ValueError:
                pass
        return 101
    return None  # run-crash: no fixed expectation in stage 1


def judge(mode, header, native, sparagmos):
    """Compare native and Sparagmos results for one attempted test.

    `native` / `sparagmos` are dicts with exit_code (int or None) and
    stdout (str). Returns (status, detail): status is "pass" or
    "fail".
    """
    exp = expected_exit(mode, header)
    if native["exit_code"] != exp:
        return ("fail",
                f"native exit {native['exit_code']} != expected {exp}")
    if sparagmos["exit_code"] != native["exit_code"]:
        return ("fail",
                f"exit mismatch: native {native['exit_code']}, "
                f"sparagmos {sparagmos['exit_code']}")
    if sparagmos["stdout"] != native["stdout"]:
        return ("fail", "stdout mismatch")
    return ("pass", "")


def stage1_blocker(header, survey_rec):
    """Why a survey exec/single-file record is not attempted, or None.

    Checks that depend only on the header and the survey record
    (environment applicability is checked separately).
    """
    # compiletest decides ignore-test before any mode or revision
    # dispatch: the test never runs, whatever else its header says.
    # Checking it after the run-mode test would report an ignore-test
    # file without a run directive as a header-feature blocker
    # instead of under its own bucket.
    if header.has("ignore-test"):
        return "ignore-test (compiletest itself never runs this test)"
    if header.has("revisions") or any(
            rev is not None for (rev, _n) in header.scoped):
        return "revisions present (per-revision cfgs not run in stage 1)"
    mode = run_mode(header)
    if mode is None:
        return "no unscoped run-pass/run-fail directive"
    if mode == "run-crash":
        return "run-crash mode (crash reproduction is out of stage-1 scope)"
    if header.has("stdin"):
        return "stdin directive present (guest stdin is not modelled)"
    if header.has("add-core-stubs"):
        return "add-core-stubs (test links the minicore stub crate)"
    return None


def stage1_view_blocker(view):
    """Why one directive view of an exec record is not attempted
    in stage 1, or None.

    The per-revision counterpart of stage1_blocker, checked per
    revision view the way stage2_blocker is: a directive scoped to
    one revision only blocks that revision. Revisions themselves
    are not a blocker here; fill_result judges a revision test once
    per revision (see fill_result_revisions).
    """
    # ignore-test first, as in stage1_blocker: compiletest's ignore
    # decision precedes every per-revision mode dispatch.
    if view.has("ignore-test"):
        return "ignore-test (compiletest itself never runs this test)"
    mode = run_mode(view)
    if mode == "run-crash":
        return "run-crash mode (crash reproduction is out of stage-1 scope)"
    if view.has("stdin"):
        return "stdin directive present (guest stdin is not modelled)"
    if view.has("add-core-stubs"):
        return "add-core-stubs (test links the minicore stub crate)"
    return None


def header_bucket(reason):
    """The not-attempted bucket for a header blocker's reason.

    ignore-test is its own bucket: compiletest itself never runs
    the test, on any host, so the record is not a gap in what the
    subset reproduces at all. The other header features (run-crash,
    stdin, core stubs) name behaviour of the test that the subset
    declines to reproduce, and keep the shared bucket.
    """
    if reason.startswith("ignore-test"):
        return "not attempted (ignore-test)"
    return "not attempted (header feature)"


# ---------------------------------------------------------------------------
# Process execution
# ---------------------------------------------------------------------------

def run_cmd(argv, *, timeout, cwd=None, env_extra=None, env_remove=()):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    for key in env_remove:
        env.pop(key, None)
    try:
        p = subprocess.run(argv, capture_output=True, timeout=timeout,
                           cwd=cwd, env=env, stdin=subprocess.DEVNULL)
        return {
            "exit_code": p.returncode,
            "stdout": p.stdout.decode("utf-8", errors="replace"),
            "stderr": p.stderr.decode("utf-8", errors="replace"),
            "timeout": False,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "exit_code": None,
            "stdout": (e.stdout or b"").decode("utf-8", errors="replace"),
            "stderr": (e.stderr or b"").decode("utf-8", errors="replace"),
            "timeout": True,
        }


def rustc_cmd():
    home = Path.home()
    rustup = home / ".cargo" / "bin" / "rustup"
    if rustup.exists():
        return [str(rustup), "run", TOOLCHAIN, "rustc"]
    return ["rustup", "run", TOOLCHAIN, "rustc"]


BOOTSTRAP_ENV = {"RUSTC_BOOTSTRAP": "1"}

_SYSROOT_LIB = None


def sysroot_lib():
    """The reference toolchain lib dir (libLLVM, librustc_driver).

    ui-fulldeps tests link librustc_driver and its LLVM library; the
    plain rustc invocation finds the crates but not -lLLVM, so the
    native compile gets this dir as -L and the native run gets it in
    LD_LIBRARY_PATH, the same setup compiletest arranges.
    """
    global _SYSROOT_LIB
    if _SYSROOT_LIB is None:
        try:
            out = subprocess.run(
                rustc_cmd() + ["--print", "sysroot"],
                capture_output=True, text=True, timeout=30)
            _SYSROOT_LIB = str(Path(out.stdout.strip()) / "lib")
        except Exception:
            _SYSROOT_LIB = ""
    return _SYSROOT_LIB


CRATE_TYPE_RE = re.compile(r"crate_type\s*=\s*\"([^\"]+)\"")
CRATE_NAME_RE = re.compile(r"crate_name\s*=\s*\"([^\"]+)\"")


def declared_crate_type(text):
    """The `#![crate_type = "..."]` an auxiliary source declares."""
    m = CRATE_TYPE_RE.search(text)
    return m.group(1) if m else None


def declared_crate_types(text):
    """Every `#![crate_type = "..."]` an auxiliary source
    declares, in order. A source may declare several (rustc
    builds one artifact per declared type); the first is what
    declared_crate_type reports."""
    return CRATE_TYPE_RE.findall(text)


def declared_crate_name(text):
    """The `#![crate_name = "..."]` an auxiliary source declares.

    When a source declares one, the declaration is the crate's name
    (compiletest links the crate under it); a differing --crate-name
    flag is a rustc error, so none is passed.
    """
    m = CRATE_NAME_RE.search(text)
    return m.group(1) if m else None


class AuxSpec(NamedTuple):
    """One resolved auxiliary crate, in dependency order.

    `crate_type` is "proc-macro" for the crates a `//@ proc-macro:`
    directive names (compiletest builds those as proc-macro crates)
    and None otherwise. `edition` is the edition the auxiliary
    crate is built with: the auxiliary file's own `//@ edition:`
    directive when it declares one, and 2015 when it does not --
    compiletest builds each auxiliary crate from its own file's
    properties (TestProps::from_aux_file) with no requested
    edition, so a directive-less auxiliary reaches rustc with no
    `--edition` flag, never with the main test's edition. The
    distinction is observable: a proc-macro crate built under the
    test's edition 2024 gives its generated macro_rules that
    edition's token matching (macro_rules_edition_from_pm).
    """

    name: str
    path: Path
    crate_type: Optional[str] = None
    edition: Optional[str] = None
    modifiers: Optional[str] = None


def as_aux_spec(spec):
    """Normalize an aux spec: an AuxSpec passes through; a plain
    (name, path) pair gains empty attributes."""
    if isinstance(spec, AuxSpec):
        return spec
    name, path = spec
    return AuxSpec(name, Path(path))


def aux_spec_arg(spec):
    """The driver's `--aux` value for a spec: `name=path` plus the
    `;crate-type=` / `;edition=` attributes the spec carries. A
    spec with `--extern` modifiers binds under
    `modifiers:name`, the verbatim binding name the driver keeps
    for such specs."""
    spec = as_aux_spec(spec)
    binding = (f"{spec.modifiers}:{spec.name}" if spec.modifiers
               else spec.name)
    arg = f"{binding}={spec.path}"
    if spec.crate_type:
        arg += f";crate-type={spec.crate_type}"
    if spec.edition:
        arg += f";edition={spec.edition}"
    return arg


def parse_aux_crate_value(value):
    """(modifiers, crate_name, file_name) from an `aux-crate:`
    value, parsed the way compiletest's parse_aux_crate parses
    it: `name=path` or `modifiers:name=path`, where the modifiers
    are `--extern` modifiers (`priv`, `noprelude`, ...) the crate
    is bound with. Returns None for a value without `=`."""
    m = re.match(r"^(?:([^=]*?):)?([^=]*)=(.*)$", value.strip())
    if not m:
        return None
    modifiers, name, fname = m.groups()
    return (modifiers, name.strip(), fname.strip())


def parse_proc_macro_value(value):
    """(modifiers, file_name) from a `proc-macro:` value, parsed
    the way compiletest's parse_proc_macro parses it: `path` or
    `modifiers:path`."""
    m = re.match(r"^(?:([^=]*?):)?(.*)$", value.strip())
    modifiers, fname = m.groups()
    return (modifiers, fname.strip())


def aux_entries(header):
    """(crate_name, file_name, crate_type, modifiers) tuples from
    a header's aux directives.

    `aux-build: foo-bar.rs` builds crate `foo_bar` (the file stem
    with dashes as underscores, as compiletest derives it);
    `aux-crate: name=file.rs` names the crate explicitly and may
    carry `--extern` modifiers (`aux-crate: priv:name=file.rs`);
    `proc-macro: foo-bar.rs` builds crate `foo_bar` as a proc-macro
    crate (its crate_type is "proc-macro"; the others' is None)
    and may likewise carry modifiers (`proc-macro: priv:foo.rs`).
    """
    out = []
    for value in header.values("aux-build"):
        fname = value.strip()
        if fname:
            out.append((Path(fname).stem.replace("-", "_"), fname,
                        None, None))
    for value in header.values("aux-crate"):
        parsed = parse_aux_crate_value(value)
        if parsed:
            modifiers, name, fname = parsed
            out.append((name, fname, None, modifiers))
    for value in header.values("proc-macro"):
        modifiers, fname = parse_proc_macro_value(value)
        if fname:
            out.append((Path(fname).stem.replace("-", "_"), fname,
                        "proc-macro", modifiers))
    return out


def aux_directives(header):
    """(crate_name, file_name) pairs from a header's aux directives
    (see `aux_entries`)."""
    return [(name, fname) for name, fname, _, _ in aux_entries(header)]


def resolve_aux_specs(test_path, revision=None, top=None,
                      top_proc_macro=True):
    """Resolve a test's auxiliary crates, transitively, deps first.

    Returns (specs, error): specs is a list of AuxSpec in
    dependency order (a crate before the crates that use it), the
    order both process sides expect. Auxiliary files are looked up
    next to the file whose directive names them, in that directory's
    `auxiliary/` subdirectory, and next to the main test file (the
    places compiletest looks). error is a not-attempted reason when
    resolution fails. With `revision` set, directives are read
    through that revision's view (an auxiliary file's own
    directives are scoped by the same revision, as compiletest
    scopes them).
    The test's own directives are read from `top` when given (a
    Header, or a HeaderView when a revision's scoped aux directives
    apply); the transitive directives of auxiliary files always come
    from the files themselves, scoped by `revision` when one is set.
    With `top_proc_macro` false, `//@ proc-macro:` directives of the
    test itself do not become specs (the caller builds those crates
    itself); proc-macro directives of auxiliary files still do.
    """
    test_dir = test_path.parent
    specs = []
    by_name = {}
    visiting = set()

    def file_header(p):
        h = Header(p.read_text(errors="ignore"))
        if revision is not None:
            h = h.view(revision)
        return h

    def directives(p):
        return aux_entries(file_header(p))

    def find_file(fname, from_dir):
        # The `auxiliary/` subdirectory wins over the directory
        # itself: a test file and its auxiliary file can share a
        # name (tests/ui/associated-type-bounds has both fn-aux.rs
        # the test and auxiliary/fn-aux.rs the auxiliary crate), and
        # the auxiliary one is meant.
        for base in (from_dir / "auxiliary", from_dir,
                     test_dir / "auxiliary", test_dir):
            cand = base / fname
            if cand.is_file():
                return cand
        return None

    def visit(name, fname, from_dir, crate_type=None, modifiers=None):
        path = find_file(fname, from_dir)
        if path is None:
            return (f"auxiliary file {fname} for crate `{name}` "
                    f"not found")
        if name in by_name:
            if by_name[name] != path:
                return (f"auxiliary crate `{name}` is provided by "
                        f"two files: {by_name[name].name} and "
                        f"{path.name}")
            return None
        if name in visiting:
            return f"auxiliary crate dependency cycle at `{name}`"
        visiting.add(name)
        by_name[name] = path
        header = file_header(path)
        for dep_name, dep_file, dep_type, dep_mod in aux_entries(header):
            err = visit(dep_name, dep_file, path.parent, dep_type,
                        dep_mod)
            if err:
                return err
        visiting.discard(name)
        editions = header.values("edition")
        edition = _concrete_edition(editions[-1].strip()) \
            if editions else "2015"
        specs.append(AuxSpec(name, path, crate_type, edition,
                             modifiers))
        return None

    if top is not None:
        top_entries = aux_entries(top)
    else:
        top_entries = directives(test_path)
    if not top_proc_macro:
        # The caller builds the test's own `//@ proc-macro:` crates
        # itself (stage 2's proc_macro_extern_args); forwarding
        # those to the driver as --aux as well would build and link
        # them twice. Proc-macro crates an auxiliary file names are
        # transitive entries, not top entries, and stay.
        top_entries = [e for e in top_entries if e[2] is None]
    for name, fname, crate_type, modifiers in top_entries:
        err = visit(name, fname, test_dir, crate_type, modifiers)
        if err:
            return [], err
    return specs, None


def find_artifact(libdir, name):
    """The linkable artifact a library compilation produced."""
    best = None
    for cand in Path(libdir).glob(f"lib{name}.*"):
        rank = {"rlib": 0, "rmeta": 1, "so": 2, "dylib": 2,
                "dll": 2}.get(cand.suffix.lstrip("."))
        if rank is None:
            continue
        if best is None or rank < best[0]:
            best = (rank, cand)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Proc-macro auxiliary crates (stage 2)
# ---------------------------------------------------------------------------

_PROC_MACRO_CACHE = {}
_PROC_MACRO_KEY_LOCKS = {}
_PROC_MACRO_META_LOCK = threading.Lock()


def _proc_macro_key_lock(key):
    with _PROC_MACRO_META_LOCK:
        return _PROC_MACRO_KEY_LOCKS.setdefault(key, threading.Lock())


def _find_proc_macro_file(test_path, fname):
    """Locate a `//@ proc-macro:` file the way compiletest does:
    the `auxiliary/` subdirectory next to the test first, then the
    test's own directory."""
    test_dir = test_path.parent
    for base in (test_dir / "auxiliary", test_dir):
        cand = base / fname
        if cand.is_file():
            return cand
    return None


def _proc_macro_chain(test_path, fname, edition, modifiers=None):
    """Resolve one `//@ proc-macro:` directive to a build chain.

    Returns (chain, error): chain is a list of (crate_name,
    source_path, forced_crate_type, extern_modifiers) in
    dependency order -- the auxiliary crates the proc-macro crate
    itself depends on first (its own aux directives, resolved
    transitively), the proc-macro crate last with its type forced
    to proc-macro and the `//@ proc-macro:` directive's own
    modifiers attached, the treatment compiletest's directive
    names. Which edition each crate is built with is decided at
    build time from its own header.
    """
    src = _find_proc_macro_file(test_path, fname)
    if src is None:
        return None, f"proc-macro auxiliary file {fname} not found"
    deps, err = resolve_aux_specs(src)
    if err:
        return None, err
    text = src.read_text(errors="ignore")
    name = declared_crate_name(text) or Path(fname).stem.replace("-", "_")
    chain = [(spec.name, spec.path, spec.crate_type, spec.modifiers)
             for spec in deps]
    chain.append((name, src, "proc-macro", modifiers))
    return chain, None


def _concrete_edition(edition):
    """The single edition a range directive (`2015..2021`)
    collapses to for the subset's one attempt.

    compiletest's range is half-open, [lower, upper), and its
    edition_to_test picks the run's requested edition when it
    falls in the range and the lower bound otherwise. Bootstrap
    passes compiletest no requested edition, so the request
    defaults to the minimum (2015) and every range -- closed or
    open-ended (`2018..`) -- resolves to its lower bound: the
    edition the test's sources are written against (a
    `2015..2021` test may use syntax later editions reject).
    The subset's collapse takes the same lower bound.
    """
    if ".." in edition:
        start, _, _end = edition.partition("..")
        return start.strip()
    return edition


def attempt_edition(view):
    """The edition one attempt compiles under.

    The view's last `//@ edition:` directive, collapsed to the
    concrete edition the single attempt uses (see
    _concrete_edition); 2015 when the test declares none --
    compiletest's `--edition` is an unset option in the reference
    run (bootstrap passes it no value), so a directive-less test
    reaches rustc with no `--edition` flag at all and compiles
    under rustc's own default edition, 2015. Every attempt path
    (stage-1 flat, stage-1 per revision, stage 2) selects the
    edition through this one helper, so a range directive reaches
    rustc only in its collapsed form.
    """
    editions = view.values("edition")
    return _concrete_edition(editions[-1].strip()) if editions else "2015"


def _build_proc_macro_chain(chain, edition, outdir, extra_externs=()):

    """Build a resolved proc-macro chain into outdir.

    Mirrors native_side's auxiliary-crate builds: dependency
    order, each crate linked to the previous ones with --extern,
    the crate's own compile-flags honoured, each crate built
    with its own edition directive or 2015 when it declares none
    (never the test's edition). Returns (crate_name,
    artifact, error) for the chain's last crate, the proc-macro
    crate itself.
    """
    built = list(extra_externs)
    last_name, last_artifact = None, None
    for cname, csrc, forced, cmod in chain:
        text = csrc.read_text(errors="ignore")
        eff_name = declared_crate_name(text) or cname
        aux_view = Header(text).view(None)
        editions = aux_view.values("edition")
        ced = _concrete_edition(editions[-1].strip()) if editions \
            else "2015"
        declared = declared_crate_type(text)
        cmd = rustc_cmd() + ["--edition", ced, "--out-dir", str(outdir)]
        lib = sysroot_lib()
        if lib:
            cmd += ["-L", lib]
        cmd += ["-L", f"dependency={outdir}"]
        if declared is None:
            cmd += ["--crate-type", forced or "rlib"]
        if declared_crate_name(text) is None:
            cmd += ["--crate-name", cname]
        # A distinct -C metadata keeps an auxiliary crate whose
        # name matches the test crate's own name distinguishable
        # from it (E0519); compiletest arranges the same
        # distinction for same-named auxiliary crates.
        meta = hashlib.sha256(str(csrc).encode()).hexdigest()[:12]
        cmd += ["-C", f"metadata=sparagmos-aux-{meta}"]
        for prev_binding, prev_artifact in built:
            if prev_artifact is not None:
                cmd += ["--extern",
                        f"{prev_binding}={prev_artifact}"]
        if forced is not None:
            # compiletest links the proc_macro library crate into
            # proc-macro auxiliary builds with a bare --extern, so
            # sources may `use proc_macro::...` without declaring
            # it; without it such sources fail with E0432.
            cmd += ["--extern", "proc_macro"]
        cmd += compile_flags(aux_view, suite_root(csrc),
                             str(outdir))
        cmd += [str(csrc)]
        # The crate's own `//@ rustc-env:` pairs apply to its
        # build, as compiletest applies an auxiliary file's
        # rustc-env to the auxiliary build.
        cenv, cunset = view_env(aux_view, "rustc-env",
                                "unset-rustc-env",
                                suite_root(csrc),
                                str(outdir))
        env = dict(BOOTSTRAP_ENV)
        env.update(cenv)
        build = run_cmd(cmd, timeout=NATIVE_COMPILE_TIMEOUT,
                        env_extra=env, env_remove=cunset)
        if build["timeout"]:
            return None, None, (f"build of auxiliary crate "
                                f"`{eff_name}` timed out")
        if build["exit_code"] != 0:
            first = first_error_line(build["stderr"])
            if first.startswith("error: linking with `cc`"):
                # The generic cc line hides the cause; the
                # linker note names it (e.g. a bootstrap-built
                # test helper library the official sysroot does
                # not ship).
                for line in build["stderr"].splitlines():
                    if "unable to find library" in line:
                        first += ": " + line.strip().split(
                            "= note: ", 1)[-1]
                        break
            return None, None, (
                f"build of auxiliary crate `{eff_name}` failed: "
                f"{first}")
        effective = forced or declared or "rlib"
        artifact = (find_artifact(outdir, eff_name)
                    if effective in ("rlib", "lib", "dylib", "proc-macro")
                    else None)
        binding = f"{cmod}:{eff_name}" if cmod else eff_name
        built.append((binding, artifact))
        last_name, last_artifact = eff_name, artifact
    if last_artifact is None:
        return None, None, (f"build of proc-macro crate "
                            f"`{last_name}` produced no linkable "
                            f"artifact")
    return last_name, last_artifact, None


def build_proc_macro(test_path, fname, edition, extra_externs=(),
                     modifiers=None):
    """Build the proc-macro crate a `//@ proc-macro:` directive names.

    Returns (crate_name, artifact, error). Builds are cached: the
    same auxiliary crate backs many tests (test-macros.rs alone
    serves dozens), so a key over the chain's source contents, the
    test edition and any crates built earlier for the same test
    (extra_externs, which this crate is linked against) maps to
    one on-disk build shared by every caller, across revisions
    and repeated runs.
    """
    edition = _concrete_edition(edition)
    chain, err = _proc_macro_chain(test_path, fname, edition,
                                   modifiers)
    if err:
        return None, None, err
    h = hashlib.sha256()
    # Recipe version: bump when the build recipe changes (crate
    # metadata disambiguation, the bare proc_macro extern, ...),
    # so artifacts built by an older recipe are never reused.
    h.update(b"sparagmos-pm-recipe-4")  # 4: chain crates fall back to edition 2015, not the test edition
    h.update(edition.encode())
    for prev_name, prev_artifact in extra_externs:
        h.update(prev_name.encode())
        h.update(str(prev_artifact).encode())
    for cname, csrc, forced, cmod in chain:
        h.update(cname.encode())
        h.update(csrc.read_bytes())
        h.update((forced or "").encode())
        h.update((cmod or "").encode())
    key = h.hexdigest()[:24]
    if key in _PROC_MACRO_CACHE:
        return _PROC_MACRO_CACHE[key]
    cache_dir = (Path(tempfile.gettempdir())
                 / "sparagmos-proc-macro-cache" / key)
    with _proc_macro_key_lock(key):
        if key in _PROC_MACRO_CACHE:
            return _PROC_MACRO_CACHE[key]
        name = chain[-1][0]
        result = None
        if cache_dir.is_dir():
            artifact = find_artifact(cache_dir, name)
            if artifact is not None:
                result = (name, artifact, None)
        if result is None:
            tmpdir = Path(tempfile.mkdtemp(prefix="sparagmos-pm-build-"))
            name2, artifact2, err2 = _build_proc_macro_chain(
                chain, edition, tmpdir, extra_externs)
            if err2:
                result = (None, None, err2)
            else:
                cache_dir.mkdir(parents=True, exist_ok=True)
                for f in tmpdir.iterdir():
                    f.replace(cache_dir / f.name)
                result = (name2, cache_dir / artifact2.name, None)
        _PROC_MACRO_CACHE[key] = result
        return result


def proc_macro_extern_args(test_path, view, edition):
    """--extern arguments reproducing compiletest's `proc-macro:`
    handling for one directive view, or a not-attempted reason.

    compiletest compiles each file a `//@ proc-macro:` directive
    names as a proc-macro crate and links it into the test's
    compilation. Without that link the driver's rustc fails with
    E0463 (can't find crate) on tests native rustc accepts, so the
    conversion is attempted with the same --extern set compiletest
    arranges. Returns (args, error).
    """
    directives = [parse_proc_macro_value(v)
                  for v in view.values("proc-macro")]
    directives = [(mod, f) for mod, f in directives if f]
    if not directives:
        return [], None
    args = []
    seen_dirs = set()
    built = []
    for modifiers, fname in directives:
        # Directive order matters: a later proc-macro crate may
        # use an earlier one (compiletest links each auxiliary
        # build against the ones before it), so each build sees
        # the crates already built for this view as externs.
        name, artifact, err = build_proc_macro(
            test_path, fname, edition, extra_externs=built,
            modifiers=modifiers)
        if err:
            return [], (f"proc-macro auxiliary crate from {fname} "
                        f"could not be built: {err}")
        binding = f"{modifiers}:{name}" if modifiers else name
        built.append((binding, artifact))
        args += ["--extern", f"{binding}={artifact}"]
        dep_dir = str(artifact.parent)
        if dep_dir not in seen_dirs:
            seen_dirs.add(dep_dir)
            args += ["-L", f"dependency={dep_dir}"]
    return args, None


_SYSROOT_EXTERN_CACHE = None
_SYSROOT_RMETA_RE = re.compile(r"^lib(.+)-([0-9a-f]{16})\.(rmeta|rlib)$")
_EXTERN_CRATE_RE = re.compile(r"\bextern\s+crate\s+([A-Za-z_][A-Za-z0-9_]*)")


def ambiguous_sysroot_crates():
    """Sysroot crates with more than one rmeta candidate, mapped
    to the candidate a link can use.

    A rustup sysroot carrying the rustc-dev component holds two
    rmetas for crates such as libc in its target lib dir: the
    rust-std build's, paired with an rlib, and the compiler's own
    build's, metadata only. An `extern crate` lookup then fails
    with E0464 (multiple candidates), where the bootstrap sysroot
    compiletest runs against holds a single candidate per crate.
    The linkable candidate is the one whose hash also names an
    rlib; the metadata-only candidate cannot satisfy a link
    (rustc: "required to be available in rlib format"). Computed
    once from the pinned toolchain's target lib dir.
    """
    global _SYSROOT_EXTERN_CACHE
    if _SYSROOT_EXTERN_CACHE is None:
        facts = host_facts()
        if facts is None:
            return {}
        _SYSROOT_EXTERN_CACHE = {}
        libdir = facts.target_lib_dir()
        if libdir is not None:
            try:
                entries = os.listdir(libdir)
            except OSError:
                entries = []
            by_name = {}
            for entry in entries:
                m = _SYSROOT_RMETA_RE.match(entry)
                if m:
                    by_name.setdefault(m.group(1), {}) \
                        .setdefault(m.group(2), set()).add(m.group(3))
            for name, hashes in by_name.items():
                if len(hashes) < 2:
                    continue
                linkable = [h for h, kinds in hashes.items()
                            if kinds == {"rmeta", "rlib"}]
                if len(linkable) == 1:
                    _SYSROOT_EXTERN_CACHE[name] = str(
                        libdir / f"lib{name}-{linkable[0]}.rmeta")
    return _SYSROOT_EXTERN_CACHE


def sysroot_extern_args(path, flags):
    """`--extern` arguments pinning the `extern crate` names a
    source declares that the sysroot offers ambiguously (see
    ambiguous_sysroot_crates).

    Pinning reproduces the single-candidate lookup a bootstrap
    sysroot gives the same source. Only names the source itself
    declares are pinned, and only for a host-target compilation:
    the pinned paths are the host target's. A name the flags
    already bind with their own --extern is left to them.
    """
    ambiguous = ambiguous_sysroot_crates()
    if not ambiguous:
        return []
    bound = set()
    i = 0
    while i < len(flags):
        opt, eq, val = flags[i].partition("=")
        if opt in ("--target", "--extern"):
            if not eq:
                if i + 1 >= len(flags):
                    break
                i += 1
                val = flags[i]
            if opt == "--target" and val != HOST_TARGET:
                return []
            if opt == "--extern":
                bound.add(val.split("=", 1)[0].rsplit(":", 1)[-1])
        i += 1
    try:
        text = Path(path).read_text(errors="ignore")
    except OSError:
        return []
    args = []
    for name in sorted(set(_EXTERN_CRATE_RE.findall(text))):
        if name in ambiguous and name not in bound:
            args += ["--extern", f"{name}={ambiguous[name]}"]
    return args


_TEST_HELPERS_LOCK = threading.Lock()
_TEST_HELPERS_ARGS = None


def test_helpers_args(test_path):
    """`-L native=` arguments naming the directory that holds
    librust_test_helpers.a, or [] when it cannot be provided.

    bootstrap compiles tests/auxiliary/rust_test_helpers.c into
    that static library (host cc, opt level 0) and passes
    `-Lnative=<dir>` to every ui / incremental compilation
    through --target-rustcflags; tests reach the helpers through
    #[link(name = "rust_test_helpers")]. The subset builds the
    same library from the same source, once per process into a
    private cache directory, and hands the directory to the
    native compilations the same way. When the source is absent
    or the build fails the compilations proceed without it and
    fail exactly as they did before.
    """
    global _TEST_HELPERS_ARGS
    with _TEST_HELPERS_LOCK:
        if _TEST_HELPERS_ARGS is not None:
            return _TEST_HELPERS_ARGS
        args = []
        src = None
        for anc in Path(test_path).parents:
            if anc.name == "tests":
                cand = anc / "auxiliary" / "rust_test_helpers.c"
                if cand.is_file():
                    src = cand
                break
        if src is not None:
            outdir = (Path(tempfile.gettempdir())
                      / f"sparagmos-test-helpers-{os.getpid()}")
            lib = outdir / "librust_test_helpers.a"
            try:
                outdir.mkdir(parents=True, exist_ok=True)
                if not lib.is_file():
                    obj = outdir / "rust_test_helpers.o"
                    cc = run_cmd(["cc", "-O0", "-c", str(src),
                                  "-o", str(obj)], timeout=120)
                    if cc["exit_code"] == 0:
                        run_cmd(["ar", "rcs", str(lib), str(obj)],
                                timeout=60)
                if lib.is_file():
                    args = ["-L", f"native={outdir}"]
            except OSError:
                args = []
        _TEST_HELPERS_ARGS = args
        return args


def aux_header_view(text):
    """The unscoped directive view of an auxiliary file's own
    header. compiletest loads an auxiliary build's properties
    from the auxiliary file itself (TestProps::from_aux_file),
    so its compile-flags, rustc-env and no-prefer-dynamic are
    the file's own, never the test's."""
    return Header(text).view(None)


def native_side(path, edition, workdir, extra_flags=(), aux_specs=(),
                run_args=(), run_env_extra=None, run_env_remove=(),
                compile_env_extra=None, compile_env_remove=(),
                no_prefer_dynamic=False):
    """Compile and run natively. Returns (result_dict, error_reason).

    error_reason is None when the native run completed (whatever its
    exit code); otherwise a not-attempted reason string.
    `extra_flags` are the test's compile-flags, appended last as
    compiletest appends them. Auxiliary crates are compiled first,
    in dependency order, the way compiletest builds them for the
    test's own compilation. `run_args` are the test's run-flags,
    appended to the run's argv, and `run_env_extra` /
    `run_env_remove` its exec-env pairs and unsets, applied to
    the run only -- the positions compiletest gives them.
    `compile_env_extra` / `compile_env_remove` are the view's
    rustc-env pairs and unsets, applied to the compilations
    themselves (compiletest applies rustc-env to its rustc
    invocations, and a test may read the values back with env!);
    each auxiliary build takes its own file's rustc-env instead,
    as compiletest loads an auxiliary build's properties from
    the auxiliary file. `no_prefer_dynamic` is the view's
    `//@ no-prefer-dynamic:` directive: compiletest adds
    `-C prefer-dynamic` to a compilation unless its own
    properties opt out, the test's for the test's compilation
    and each auxiliary file's for its own, and builds an
    auxiliary crate as a dylib by default on dynamic-linking
    targets (an auxiliary file opting out keeps the default
    library type). Every compilation also receives the
    rust_test_helpers search path bootstrap provides (see
    test_helpers_args).
    """
    binpath = Path(workdir) / "prog"
    lib = sysroot_lib()
    helpers = test_helpers_args(path)
    compile_env = dict(BOOTSTRAP_ENV)
    if compile_env_extra:
        compile_env.update(compile_env_extra)
    externs = []
    auxdir = Path(workdir) / "auxlibs"
    if aux_specs:
        auxdir.mkdir()
        built = []
        for spec in aux_specs:
            spec = as_aux_spec(spec)
            name, src = spec.name, spec.path
            text = src.read_text(errors="ignore")
            declared = declared_crate_type(text)
            eff_name = declared_crate_name(text) or name
            aview = aux_header_view(text)
            aux_no_pd = aview.has("no-prefer-dynamic")
            aux_flags = compile_flags(aview, suite_root(src), workdir)
            if screen_flags(aux_flags, targets={HOST_TARGET}):
                aux_flags = []
            # A source declaring several crate types builds one
            # artifact per type; the one a dependent compilation
            # can link against is the first linkable declared
            # type (a cdylib-only auxiliary stays unbound, as
            # before). With no declared type, the directive's
            # forced type or compiletest's default applies.
            linkable = [t for t in declared_crate_types(text)
                        if t in ("rlib", "lib", "dylib", "proc-macro")]
            if linkable:
                effective = linkable[0]
            elif declared is not None:
                effective = declared
            else:
                effective = spec.crate_type or (
                    "rlib" if aux_no_pd else "dylib")
            cmd = rustc_cmd() + ["--edition", spec.edition or edition,
                                 "--out-dir", str(auxdir)]
            if lib:
                cmd += ["-L", lib]
            cmd += ["-L", f"dependency={auxdir}"]
            cmd += helpers
            if declared is None:
                cmd += ["--crate-type", effective]
            if declared_crate_name(text) is None:
                cmd += ["--crate-name", name]
            if not aux_no_pd and not any(
                    "prefer-dynamic" in f for f in aux_flags):
                cmd += ["-C", "prefer-dynamic"]
            for prev_binding, prev_artifact in built:
                if prev_artifact is not None:
                    cmd += ["--extern",
                            f"{prev_binding}={prev_artifact}"]
            cmd += sysroot_extern_args(src, ())
            cmd += aux_flags
            cmd += [str(src)]
            apairs, aunset = view_env(aview, "rustc-env",
                                      "unset-rustc-env",
                                      suite_root(src), workdir)
            aenv = dict(BOOTSTRAP_ENV)
            aenv.update(apairs)
            build = run_cmd(cmd, timeout=NATIVE_COMPILE_TIMEOUT,
                            env_extra=aenv, env_remove=aunset)
            if build["timeout"]:
                return None, "native compile timed out"
            if build["exit_code"] != 0:
                first = first_error_line(build["stderr"])
                return None, f"native compile failed: {first}"
            artifact = (find_artifact(auxdir, eff_name)
                        if effective in ("rlib", "lib", "dylib",
                                         "proc-macro")
                        else None)
            binding = (f"{spec.modifiers}:{eff_name}"
                       if spec.modifiers else eff_name)
            built.append((binding, artifact))
            if artifact is not None:
                externs += ["--extern", f"{binding}={artifact}"]
    main_cmd = (
        rustc_cmd() + ["--edition", edition, "-C", "opt-level=0"]
        + (["-L", lib] if lib else [])
        + (["-L", f"dependency={auxdir}"] if aux_specs else [])
        + helpers
        + externs
        + sysroot_extern_args(path, extra_flags))
    if not no_prefer_dynamic and not any(
            "prefer-dynamic" in f for f in extra_flags):
        main_cmd += ["-C", "prefer-dynamic"]
    main_cmd += ["-o", str(binpath), str(path)] + list(extra_flags)
    build = run_cmd(main_cmd, timeout=NATIVE_COMPILE_TIMEOUT,
                    env_extra=compile_env,
                    env_remove=compile_env_remove)
    if build["timeout"]:
        return None, "native compile timed out"
    if build["exit_code"] != 0:
        first = first_error_line(build["stderr"])
        return None, f"native compile failed: {first}"
    if extra_flags and not binpath.exists():
        return None, ("native compile produced no binary (the "
                      "compile-flags change what is emitted)")
    run_env = {}
    # The run's library path is compiletest's: the sysroot lib
    # dir (librustc_driver for ui-fulldeps binaries), the target
    # lib dir holding the shared std libraries a prefer-dynamic
    # binary loads, and the auxiliary directory holding any
    # dylib auxiliaries the binary links.
    lib_dirs = ([lib] if lib else [])
    facts = host_facts()
    if facts is not None and facts.target_lib_dir() is not None:
        lib_dirs.append(str(facts.target_lib_dir()))
    if aux_specs:
        lib_dirs.append(str(auxdir))
    if lib_dirs:
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        run_env["LD_LIBRARY_PATH"] = ":".join(
            lib_dirs + ([prev] if prev else []))
    if run_env_extra:
        run_env.update(run_env_extra)
    run = run_cmd([str(binpath)] + list(run_args),
                  timeout=NATIVE_RUN_TIMEOUT,
                  cwd=str(workdir), env_extra=run_env,
                  env_remove=run_env_remove)
    if run["timeout"]:
        return None, "native run timed out"
    return run, None


def sparagmos_side(path, edition, driver_bin, cwd=None, rustc_args=(),
                   aux_specs=(), guest_args=(), env_extra=None,
                   env_remove=(), jit=False):
    """Run through the driver. Returns (result_dict, kind, reason).

    kind is "result" (exit 0/101), "unsupported" (driver exit 2),
    "timeout", or "driver-error". `rustc_args` are the test's
    compile-flags, forwarded through the driver's `--rustc-arg`.
    Auxiliary crates are handed to the driver with `--aux`, in
    dependency order. `guest_args` are the test's run-flags,
    forwarded through the driver's `--guest-arg` (the guest
    program's argv after argv[0], whose policy is the
    interpreter's). `env_extra` / `env_remove` are the test's
    exec-env pairs and unsets, applied to the driver process --
    the guest runs in-process, so this is the environment
    compiletest's exec-env describes for it. With `jit=True` the
    driver's `run-jit` mode replaces `run`: the same conversion
    and the same collected program, executed by `sparagmos-jit`
    instead of the interpreter (metric 4).
    """
    rustc_args = list(rustc_args) + sysroot_extern_args(path, rustc_args)
    argv = [str(driver_bin), "run-jit" if jit else "run",
            str(path), "--edition", edition]
    for a in rustc_args:
        argv += ["--rustc-arg", a]
    for spec in aux_specs:
        argv += ["--aux", aux_spec_arg(spec)]
    for a in guest_args:
        argv += ["--guest-arg", a]
    env = dict(BOOTSTRAP_ENV)
    if env_extra:
        env.update(env_extra)
    res = run_cmd(argv,
                  timeout=DRIVER_TIMEOUT, cwd=cwd,
                  env_extra=env, env_remove=env_remove)
    if res["timeout"]:
        return None, "timeout", f"driver timed out after {DRIVER_TIMEOUT}s"
    if res["exit_code"] in (0, 101):
        return res, "result", ""
    if res["exit_code"] == 2:
        return None, "unsupported", driver_reason(res["stderr"], jit=jit)
    return None, "driver-error", \
        f"driver exit {res['exit_code']}: {res['stderr'].strip()[:300]}"


def first_error_line(stderr):
    for line in stderr.splitlines():
        line = line.strip()
        if line.startswith("error"):
            return line[:200]
    return (stderr.strip().splitlines() or ["<no output>"])[0][:200]


def driver_reason(stderr, jit=False):
    marker = "sparagmos-driver run-jit:" if jit \
        else "sparagmos-driver run:"
    lines = [l.strip() for l in stderr.splitlines()
             if marker in l]
    if lines:
        return "; ".join(lines)[:400]
    return ("driver reported unsupported (exit 2) without a reason line: "
            + stderr.strip()[:300])


# ---------------------------------------------------------------------------
# Reason categorisation for aggregates
# ---------------------------------------------------------------------------

def normalize_reason(reason):
    """Strip volatile details so equal causes group together."""
    r = reason
    r = re.sub(r"/[^\s:'\"]+", "<path>", r)
    r = re.sub(r"\b\d+\b", "<n>", r)
    r = re.sub(r"`[^`]*`", "`<name>`", r)
    return r


def bucket_reason(reason):
    """Coarse bucket for a driver/attempt reason string.

    The driver prints its primary reason first and supplementary
    lines (e.g. the unconverted-callees list) after it, so the first
    segment decides the bucket; the full text is only consulted for
    the patterns that have no primary line of their own.
    """
    first = re.split(r"; sparagmos-driver run(?:-jit)?:",
                     reason)[0].lower()
    r = reason.lower()
    if "guest arguments are not supported by jit" in first:
        return "JIT execution: guest arguments unsupported"
    if "c-convention entry" in first:
        return "JIT execution: C-convention entry unsupported"
    if "exitcode` is not supported by jit" in first:
        return "JIT execution: ExitCode entry unsupported"
    if "by-reference constant" in first:
        return "unsupported constant: by-reference constant"
    if "boxdereftransmute" in first:
        return "main not converted (BoxDerefTransmute cast)"
    if "inlineasm" in first:
        return "InlineAsm not supported"
    if "unsupported terminator: drop" in first:
        return "unsupported terminator: Drop"
    if "unsupported type in local" in first:
        return "unsupported type in local (interp type model)"
    if "no monomorphic instance" in first:
        return "no monomorphic instance available"
    if "was not converted" in first:
        return "main not converted (other conversion gap)"
    if "conversion failed" in first or "mir conversion" in first:
        return "MIR conversion failed (other)"
    if "uninitialized" in first:
        return "read of uninitialized place"
    if "type mismatch" in first:
        return "type mismatch in interp conversion"
    if "no converted body" in r or "calls to functions" in r:
        return "call to function with no converted body (std/core not modelled)"
    if "without a reason line" in r:
        return "driver exit 2 without a run-mode reason (front-end failure)"
    if "timeout" in r or "timed out" in r:
        return "timeout"
    if "native compile failed" in r:
        return "native compile failed"
    return "other"


# ---------------------------------------------------------------------------
# Per-test driver
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Debuginfo tests (nature "exec-debugger")
#
# compiletest runs these by compiling the test file unoptimized,
# setting a breakpoint on every line containing `#break`, feeding
# the file's `//@ <debugger>-command:` sections to a real
# debugger, and matching the `//@ <debugger>-check:` lines against
# its output in order. A real debugger reads DWARF out of the
# compiled binary; this runtime produces no binary, so the subset
# interprets the sections itself and drives the interpreter's
# debugger interface (the driver's `debug` entry point) instead:
# the same breakpoints, the same command flow, and the same check
# matching, with the session served by the execution engine.
#
# Only the scalar scope is attempted (the reachability census's
# D0/D1 categories): files whose applicable variants carry no
# session at all are judged by the conversion probe the
# compile-check category uses, and sessions are attempted only
# when every command and print expression stays within line
# stops and scalar prints. Anything beyond that (aggregate
# rendering, type queries, call expressions, pretty-printer
# forms, the `repr` pseudo-command's custom inspection) keeps a
# not-attempted bucket.
# ---------------------------------------------------------------------------

DEBUGGER_VARIANT_DIRECTIVES = frozenset({
    "ignore-gdb", "ignore-lldb", "ignore-cdb",
    "only-gdb", "only-lldb", "only-cdb",
})


class _DebuggerFilteredView:
    """A revision's directive view with the per-debugger variant
    directives removed. compiletest decides `ignore-gdb` /
    `only-lldb` and their kin per debugger variant, not per host,
    so the host applicability evaluation must not see them (the
    bare names would otherwise read as host conditions that do
    not hold); the variant decision itself is
    debugger_variant_skipped's."""

    def __init__(self, view):
        self._view = view
        self.names = view.names - DEBUGGER_VARIANT_DIRECTIVES

    def has(self, name):
        return (name not in DEBUGGER_VARIANT_DIRECTIVES
                and self._view.has(name))

    def values(self, name):
        return self._view.values(name)


def debugger_variant_skipped(view, debugger):
    """Whether compiletest would not run `debugger`'s variant of a
    test with these directives: an `ignore-<debugger>`, or an
    `only-<other>` naming another debugger."""
    if view.has(f"ignore-{debugger}"):
        return True
    return any(view.has(f"only-{other}")
               for other in ("gdb", "lldb", "cdb") if other != debugger)


def parse_debugger_sections(text, debugger, revision):
    """One debugger's sections of a debuginfo test file: a port of
    compiletest's DebuggerCommands::parse_from. Returns
    (breakpoint_lines, commands, check_lines): the 1-based lines
    containing `#break`, the debugger's commands in file order,
    and its (source line, pattern) check lines. The `-repr`
    pseudo-directive becomes a `repr <var>` command plus its
    artificial `<var>: Ok` check line, as in compiletest."""
    breakpoint_lines = []
    commands = []
    check_lines = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if "#break" in line:
            breakpoint_lines.append(lineno)
            continue
        m = DIRECTIVE_RE.match(line)
        if not m:
            continue
        rev, name, value = m.group(1), m.group(2), m.group(3)
        if rev is not None and rev != revision:
            continue
        if not name.startswith(debugger):
            continue
        kind = name[len(debugger):]
        if value is None:
            continue
        if kind == "-command":
            commands.append(value)
        elif kind == "-check":
            check_lines.append((lineno, value))
        elif kind == "-repr":
            commands.append(f"repr {value.strip()}")
            check_lines.append((lineno, f"{value.strip()}: Ok"))
    return breakpoint_lines, commands, check_lines


def debugger_check_single_line(line, check_line):
    """Whether one output line matches one check pattern: a port
    of compiletest's check_single_line. The pattern's `[...]`
    spans are wildcards; the fragments must appear in order, and
    unless the pattern ends with `[...]`, the line must end
    exactly at the last fragment."""
    line = line.strip()
    check_line = check_line.strip()
    can_start_anywhere = check_line.startswith("[...]")
    can_end_anywhere = check_line.endswith("[...]")
    fragments = [f for f in check_line.split("[...]") if f]
    if not fragments:
        return True
    if can_start_anywhere:
        pos = line.find(fragments[0])
        if pos < 0:
            return False
        rest = line[pos + len(fragments[0]):]
        fragments = fragments[1:]
    else:
        rest = line
    for fragment in fragments:
        pos = rest.find(fragment)
        if pos < 0:
            return False
        rest = rest[pos + len(fragment):]
    return can_end_anywhere or rest == ""


def debugger_check_output(check_lines, output_lines):
    """Match every check line in order against the session's
    output lines (extra lines may intervene; a matched line may
    match again for the next check, as in compiletest's
    check_output). Returns (ok, missing): the (source line,
    pattern) pairs that were not found."""
    missing = []
    last_idx = 0
    for src_lineno, pattern in check_lines:
        hit = None
        for idx in range(last_idx, len(output_lines)):
            if debugger_check_single_line(output_lines[idx], pattern):
                hit = idx
                break
        if hit is None:
            missing.append((src_lineno, pattern))
        else:
            last_idx = hit
    return (not missing), missing


# A print expression within the attempted scope: dereferences, a
# variable name, and numeric field/index steps — the grammar the
# debugger interface's expression evaluator speaks.
_DEBUGGER_EXPR_RE = re.compile(
    r"^\s*\**\s*[A-Za-z_][A-Za-z0-9_]*(\s*\.\d+|\s*\[\d+\])*\s*$")


def _translate_gdb_command(cmd):
    """One gdb command of a test file, as a line of the session
    script the driver's debug entry point speaks (see
    parse_debug_script there), or None when the command is
    outside the attempted scope. Session-shaping `set` commands
    that cannot change scalar output become script comments."""
    if cmd in ("run", "r"):
        return "run"
    if cmd in ("continue", "c"):
        return "continue"
    if cmd in ("next", "n"):
        return "next"
    if cmd in ("step", "s"):
        return "step"
    if cmd == "finish":
        return "finish"
    if cmd.startswith(("set print ", "set language ", "set pagination ",
                       "set confirm ", "set charset ")):
        return f"# inert: {cmd}"
    if cmd in ("frame", "f"):
        return "show-line"
    m = re.match(r"^(?:frame|f)\s+(\d+)$", cmd)
    if m:
        return f"frame {m.group(1)}"
    m = re.match(r"^(?:print|p)(?:/([a-z]))?\s+(.+)$", cmd)
    if m:
        fmt, expr = m.group(1), m.group(2).strip()
        if fmt is not None and fmt not in "dxtcu":
            return None
        if not _DEBUGGER_EXPR_RE.match(expr):
            return None
        return f"print/{fmt} {expr}" if fmt else f"print {expr}"
    return None


def _translate_lldb_command(cmd):
    """One lldb command of a test file, as a session script line,
    or None when outside the attempted scope."""
    if cmd in ("run", "r"):
        return "run"
    if cmd in ("continue", "c"):
        return "continue"
    if cmd in ("next", "n"):
        return "next"
    if cmd in ("step", "s"):
        return "step"
    if cmd == "finish":
        return "finish"
    if cmd.startswith("settings set "):
        return f"# inert: {cmd}"
    if cmd == "frame select":
        return "show-line"
    m = re.match(r"^frame\s+select\s+(\d+)$", cmd)
    if m:
        return f"frame {m.group(1)}"
    m = re.match(r"^(?:v|print|p|expr|expression)\s+(.+)$", cmd)
    if m:
        expr = m.group(1).strip()
        if expr.startswith("-"):
            return None
        if not _DEBUGGER_EXPR_RE.match(expr):
            return None
        return f"v {expr}"
    return None


def translate_debugger_command(debugger, command):
    """Translate one command of a test file's debugger section
    into the session script's vocabulary, or None when the
    command is outside the attempted scope."""
    cmd = command.strip()
    if debugger == "gdb":
        return _translate_gdb_command(cmd)
    return _translate_lldb_command(cmd)


def sparagmos_debug_side(path, edition, driver_bin, dialect, script_text,
                         rustc_args=(), aux_specs=(), env_extra=None,
                         env_remove=()):
    """Run one debugger session through the driver's `debug`
    entry point. Returns (result, kind, reason) in
    sparagmos_side's convention: kind "result" carries the
    session transcript in the result's stdout."""
    rustc_args = list(rustc_args) + sysroot_extern_args(path, rustc_args)
    script_path = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", suffix=".script", delete=False) as f:
            f.write(script_text)
            script_path = f.name
        argv = [str(driver_bin), "debug", str(path), "--edition", edition]
        for a in rustc_args:
            argv += ["--rustc-arg", a]
        for spec in aux_specs:
            argv += ["--aux", aux_spec_arg(spec)]
        argv += ["--debug-script", script_path, "--debug-dialect", dialect]
        env = dict(BOOTSTRAP_ENV)
        if env_extra:
            env.update(env_extra)
        res = run_cmd(argv, timeout=DRIVER_TIMEOUT, env_extra=env,
                      env_remove=env_remove)
    finally:
        if script_path is not None:
            os.unlink(script_path)
    if res["timeout"]:
        return None, "timeout", f"driver timed out after {DRIVER_TIMEOUT}s"
    if res["exit_code"] == 0:
        return res, "result", ""
    if res["exit_code"] == 2:
        return None, "unsupported", driver_reason(res["stderr"])
    return None, "driver-error", \
        f"driver exit {res['exit_code']}: {res['stderr'].strip()[:300]}"


def fill_result_debugger(rec, rust_dir, driver_bin, out):
    """Wrapper: the judgment runs under one scratch build
    directory (compiletest's build-base stand-in for flag and
    env expansion), removed when the record is judged. A
    `should-fail` test (one that exists to prove the checks are
    really compared: its expected output deliberately does not
    match) inverts the judgment, as compiletest's does."""
    path = rust_dir / rec["path"]
    try:
        header = Header(path.read_text(errors="ignore"))
    except OSError:
        header = None
    if header is not None and header.has("should-fail"):
        scratch = {"status": "not-attempted", "reason": "", "bucket": ""}
        with tempfile.TemporaryDirectory(prefix="subset-dbg-") as td:
            _fill_result_debugger(rec, rust_dir, driver_bin, scratch, td)
        if scratch["status"] == "pass":
            out["reason"] = ("should-fail test's session unexpectedly "
                              "matched all its check lines")
            out["bucket"] = "debugger output mismatch"
        else:
            out["status"] = "pass"
        return
    with tempfile.TemporaryDirectory(prefix="subset-dbg-") as td:
        _fill_result_debugger(rec, rust_dir, driver_bin, out, td)


def _fill_result_debugger(rec, rust_dir, driver_bin, out, td):
    """Judge one debuginfo (exec-debugger) record: run each
    applicable debugger variant's session through the
    interpreter's debugger interface and match its check lines
    against the transcript. A record whose applicable variants
    carry no session at all (compile-only files; cdb-only files,
    cdb having no session on this host) is judged by the
    conversion probe the compile-check category uses."""
    if rec["suite"] != "debuginfo":
        out["reason"] = (f"suite={rec['suite']} (the exec-debugger "
                         "harness serves the debuginfo suite)")
        out["bucket"] = "out of stage-1 scope (suite)"
        return
    path = rust_dir / rec["path"]
    text = path.read_text(errors="ignore")
    header = Header(text)
    if header.has("ignore-test"):
        out["reason"] = "ignore-test (compiletest itself never runs this test)"
        out["bucket"] = "not attempted (ignore-test)"
        return
    revisions = header.revisions() or [None]
    src_base = suite_root(path)
    # Per applicable revision: its compile assembly and its
    # debugger variants (debugger, breakpoints, commands, checks).
    applicable = []
    skip_reasons = []
    unevaluatable = False
    for revision in revisions:
        view = header.view(revision)
        status, why = applicability(_DebuggerFilteredView(view))
        if status == "out-of-environment":
            skip_reasons.append(why)
            continue
        if status == "unevaluatable":
            unevaluatable = True
            skip_reasons.append(why)
            continue
        variants = []
        for debugger in ("gdb", "lldb"):
            if debugger_variant_skipped(view, debugger):
                continue
            variants.append((debugger,)
                            + parse_debugger_sections(text, debugger,
                                                      revision))
        applicable.append((revision, view, variants))
    if not applicable:
        if unevaluatable:
            out["reason"] = "; ".join(skip_reasons)
            out["bucket"] = "not attempted (unevaluatable condition)"
        else:
            out["reason"] = "; ".join(skip_reasons)
            out["bucket"] = "out of environment"
        return

    def assembly(view, revision):
        """The revision's driver-invocation assembly, put
        together the way stage 2 assembles a view's conversion:
        (edition, args, aux_specs, compile_env, compile_unset,
        exec_env, exec_unset), or a (reason, bucket) refusal."""
        aux_specs = []
        if not rec["single_file"]:
            aux_specs, aux_err = resolve_aux_specs(path, revision)
            if aux_err:
                return None, (aux_err, "out of stage-1 scope (multi-file)")
        edition = attempt_edition(view)
        flags = compile_flags(view, src_base, td)
        screened = screen_flags(flags, targets={HOST_TARGET})
        if screened:
            return None, (screened, "not attempted (compile-flags)")
        rev_args = (revision_args(view, revision, revisions)
                    if revision is not None else [])
        args = (rev_args
                + compiletest_mode_args(rec["suite"], flags) + flags)
        env_pairs, env_unset = view_env(view, "exec-env",
                                        "unset-exec-env",
                                        src_base, td)
        cenv, cunset = view_env(view, "rustc-env", "unset-rustc-env",
                                src_base, td)
        return (edition, args, aux_specs, cenv, tuple(cunset),
                env_pairs, tuple(env_unset)), None

    has_session = any(
        commands or checks
        for (_rev, _view, variants) in applicable
        for (_dbg, _bps, commands, checks) in variants)

    if not has_session:
        # Compile-only: the conversion probe, judged as a
        # success-expecting compile-check.
        for revision, view, _variants in applicable:
            asm, refusal = assembly(view, revision)
            if refusal:
                out["reason"], out["bucket"] = refusal
                return
            edition, args, aux_specs, cenv, cunset, _ep, _eu = asm
            conv = convert_side(path, edition, driver_bin,
                                rustc_args=args, aux_specs=aux_specs,
                                env_extra=cenv, env_remove=cunset)
            if conv["timeout"]:
                out["reason"] = "driver timed out"
                out["bucket"] = "timeout"
                return
            if conv["compilation_failed"]:
                out["reason"] = ("compilation failed: "
                                  + first_error_line(conv["stderr"]))
                out["bucket"] = "compilation failed"
                return
            if conv["exit_code"] != 0 or conv.get("target_skipped"):
                totals = conv.get("target_skip_totals") or []
                detail = (": " + ", ".join(f"{k} x{v}" for k, v in totals)
                          if totals else "")
                out["reason"] = ("conversion incomplete (debugger "
                                  f"compile-only probe){detail}")
                out["bucket"] = "conversion incomplete"
                return
        out["status"] = "pass"
        return

    # Sessions: translate every variant's commands first; one
    # command outside the attempted scope holds the whole record
    # back (the file is not fully judged without it).
    sessions = []
    for revision, view, variants in applicable:
        for debugger, breakpoints, commands, checks in variants:
            if not commands and not checks:
                continue
            script_lines = [f"break {line}" for line in breakpoints]
            for command in commands:
                line = translate_debugger_command(debugger, command)
                if line is None:
                    out["reason"] = (f"{debugger} command outside the "
                                      f"attempted scope: {command}")
                    out["bucket"] = "not attempted (debugger feature)"
                    return
                script_lines.append(line)
            sessions.append((revision, view, debugger, commands, checks,
                             "\n".join(script_lines) + "\n"))
    for revision, view, debugger, commands, checks, script in sessions:
        if not commands:
            # compiletest would feed the debugger no commands at
            # all: the program never starts, no output is produced,
            # and every check line misses.
            detail = "; ".join(f"(line {ln}) `{pat}`"
                                for ln, pat in checks[:4])
            out["reason"] = (f"{debugger} check line(s) not found in the "
                              f"session output: {detail}")
            out["bucket"] = "debugger output mismatch"
            return
        asm, refusal = assembly(view, revision)
        if refusal:
            out["reason"], out["bucket"] = refusal
            return
        edition, args, aux_specs, cenv, cunset, env_pairs, env_unset = asm
        env = dict(cenv)
        env.update(env_pairs)
        env_remove = cunset + env_unset
        # compiletest compiles debuginfo tests unoptimized
        # ("debuginfo tests must be unoptimized", runtest.rs);
        # the MIR the session executes takes the same shape --
        # the optimizer's statement spans and constant-folded
        # debug info are not what the checks were written
        # against. (Measured both ways on the exec-debugger
        # population: the default pipeline passes strictly
        # fewer records.)
        args = args + ["-Zmir-opt-level=0"]
        res, kind, reason = sparagmos_debug_side(
            path, edition, driver_bin, debugger, script,
            rustc_args=args, aux_specs=aux_specs,
            env_extra=env, env_remove=env_remove)
        if kind == "timeout":
            out["reason"] = reason
            out["bucket"] = "timeout"
            return
        if kind == "unsupported":
            out["reason"] = reason
            out["bucket"] = "conversion incomplete (debugger session)"
            return
        if kind != "result":
            out["reason"] = reason
            out["bucket"] = "driver error"
            return
        ok, missing = debugger_check_output(
            checks, res["stdout"].splitlines())
        if not ok:
            detail = "; ".join(f"(line {ln}) `{pat}`" for ln, pat in missing[:4])
            out["reason"] = (f"{debugger} check line(s) not found in the "
                              f"session output: {detail}")
            out["bucket"] = "debugger output mismatch"
            return
    out["status"] = "pass"


def process_test(rec, rust_dir, driver_bin, stage2=False, stage3=False, jit=False):
    """Produce the result record for one survey record."""
    out = {
        "suite": rec["suite"],
        "path": rec["path"],
        "nature": rec["nature"],
        "single_file": rec["single_file"],
        "ffi": rec["ffi"],
        "status": "not-attempted",
        "reason": "",
        "bucket": "",
        "mode": None,
        "stage": 1,
        "seconds": 0.0,
    }
    if jit:
        out["jit"] = True
    t0 = time.time()
    try:
        if rec["nature"] == "exec-debugger" and not jit:
            fill_result_debugger(rec, rust_dir, driver_bin, out)
        elif stage2 and rec["nature"] == "compile-check":
            fill_result_stage2(rec, rust_dir, driver_bin, out, jit=jit)
        elif stage3 and rec["nature"] == "procedure":
            fill_result_procedure(rec, rust_dir, driver_bin, out)
        else:
            fill_result(rec, rust_dir, driver_bin, out, jit=jit)
    except Exception as e:  # a runner bug must not lose the whole run
        out["status"] = "not-attempted"
        out["reason"] = f"runner internal error: {e!r}"
        out["bucket"] = "runner internal error"
    out["seconds"] = round(time.time() - t0, 2)
    return out


def fill_result(rec, rust_dir, driver_bin, out, jit=False):
    if rec["nature"] != "exec":
        out["reason"] = f"nature={rec['nature']} (not in stage-1 scope)"
        out["bucket"] = "out of stage-1 scope (nature)"
        return
    if rec["suite"] not in ("ui", "ui-fulldeps"):
        out["reason"] = (f"suite={rec['suite']} (exec outside ui/ui-fulldeps "
                         "needs suite-specific flags, e.g. coverage "
                         "instrumentation; not in stage-1 scope)")
        out["bucket"] = "out of stage-1 scope (suite)"
        return
    path = rust_dir / rec["path"]
    text = path.read_text(errors="ignore")
    header = Header(text)
    mode = run_mode(header)
    out["mode"] = mode

    revs = header.revisions()
    if revs:
        fill_result_revisions(rec, header, revs, path, driver_bin, out,
                              rust_dir, jit=jit)
        return

    aux_specs = []
    if not rec["single_file"]:
        aux_specs, aux_err = resolve_aux_specs(path)
        if aux_err:
            out["reason"] = aux_err
            out["bucket"] = "out of stage-1 scope (multi-file)"
            return

    blocker = stage1_blocker(header, rec)
    if blocker:
        out["reason"] = blocker
        out["bucket"] = header_bucket(blocker)
        return
    status, why = applicability(header)
    if status == "out-of-environment":
        out["reason"] = why
        out["bucket"] = "out of environment"
        return
    if status == "unevaluatable":
        out["reason"] = why
        out["bucket"] = "not attempted (unevaluatable condition)"
        return

    if not rec["single_file"] and not aux_specs:
        out["reason"] = "multi-file / auxiliary crates (not in stage-1 scope)"
        out["bucket"] = "out of stage-1 scope (multi-file)"
        return

    edition = attempt_edition(header)

    src_base = suite_root(path)
    with tempfile.TemporaryDirectory(prefix="subset-run-") as td:
        view = header.view(None)
        flags = compile_flags(view, src_base, td)
        screened = screen_flags(flags, targets={HOST_TARGET})
        if screened:
            out["reason"] = screened
            out["bucket"] = "not attempted (compile-flags)"
            return
        pm_args, pm_err = proc_macro_extern_args(path, view, edition)
        if pm_err:
            out["reason"] = pm_err
            out["bucket"] = "auxiliary crate build failure"
            return
        flags = (compiletest_mode_args(rec["suite"], flags) + flags
                 + pm_args)
        env_pairs, env_unset = view_env(view, "exec-env",
                                        "unset-exec-env",
                                        src_base, td)
        cenv, cunset = view_env(view, "rustc-env", "unset-rustc-env",
                                src_base, td)
        run_flags = view_run_flags(view, src_base, td)
        native, err = native_side(path, edition, td, flags, aux_specs,
                                  run_args=run_flags,
                                  run_env_extra=env_pairs,
                                  run_env_remove=env_unset,
                                  compile_env_extra=cenv,
                                  compile_env_remove=cunset,
                                  no_prefer_dynamic=view.has(
                                      "no-prefer-dynamic"))
        if err:
            out["reason"] = err
            out["bucket"] = bucket_reason(err)
            return
        exp = expected_exit(mode, header)
        if native["exit_code"] != exp:
            out["reason"] = (f"native run exit {native['exit_code']} does "
                              f"not match the mode expectation {exp}")
            out["bucket"] = "native run does not match mode expectation"
            out["native"] = {"exit_code": native["exit_code"]}
            return
        denv = dict(cenv)
        denv.update(env_pairs)
        dunset = env_unset + [u for u in cunset if u not in env_unset]
        got, kind, reason = sparagmos_side(path, edition, driver_bin, td,
                                           rustc_args=flags,
                                           aux_specs=aux_specs,
                                           guest_args=run_flags,
                                           env_extra=denv,
                                           env_remove=dunset,
                                           jit=jit)
        if kind == "unsupported":
            out["status"] = "unsupported"
            out["reason"] = reason
            out["bucket"] = bucket_reason(reason)
            return
        if kind in ("timeout", "driver-error"):
            out["status"] = "fail"
            out["reason"] = reason
            out["bucket"] = bucket_reason(reason)
            return
        verdict, detail = judge(mode, header, native, got)
        out["status"] = "pass" if verdict == "pass" else "fail"
        if verdict != "pass":
            out["reason"] = detail
            out["bucket"] = ("stdout mismatch"
                             if detail == "stdout mismatch"
                             else "exit-code mismatch")
            out["native"] = {"exit_code": native["exit_code"],
                             "stdout": native["stdout"][:500]}
            out["sparagmos"] = {"exit_code": got["exit_code"],
                                "stdout": got["stdout"][:500]}





def _opt_pairs(flags):
    """(name, value) pairs of the -C / -Z / --name options in a
    flag list.

    compile-flags reach the runner already split into tokens, so
    an option appears joined (`-Ccodegen-units=1`) or as two
    tokens (`-C`, `codegen-units=1`); both spellings yield the
    same pair. Tokens that are not options are ignored.
    """
    pairs = []
    i = 0
    while i < len(flags):
        tok = flags[i]
        body = None
        if tok in ("-C", "-Z"):
            if i + 1 < len(flags):
                body = flags[i + 1]
                i += 1
        elif tok.startswith("-C") or tok.startswith("-Z"):
            body = tok[2:]
        elif tok.startswith("--") and "=" in tok:
            body = tok[2:]
        if body:
            name, _, value = body.partition("=")
            pairs.append((name, value))
        i += 1
    return pairs


def compiletest_mode_args(suite, flags):
    """Arguments compiletest adds for a suite's mode, which the
    subset must supply itself because it drives the same
    compilations outside compiletest.

    - Every UI test is compiled with `-Ccodegen-units=1`
      (make_compile_args, TestMode::Ui). The subset's stage-2 UI
      conversions, and stage 1's compile-checked UI revisions,
      take the same argument.
    - `-Zsanitizer=cfi` combined with LTO is accepted by rustc
      only with `-Ccodegen-units=1`. compiletest never trips
      that check for codegen tests: it compiles them with
      --emit=llvm-ir, under which no LTO validation runs. The
      driver's --emit=metadata conversion does trip it, so the
      value rustc itself requires is supplied; the codegen-unit
      count does not change the MIR a conversion sees.

    Nothing is added when the test sets codegen-units itself.
    """
    pairs = _opt_pairs(flags)
    if any(name == "codegen-units" for name, _ in pairs):
        return []
    if suite in ("ui", "ui-fulldeps"):
        return ["-Ccodegen-units=1"]
    sanitizers = [v for name, v in pairs if name == "sanitizer"]
    if any("cfi" in v.split(",") for v in sanitizers) and \
            any(name == "lto" for name, _ in pairs):
        return ["-Ccodegen-units=1"]
    return []


def check_mode_lint_args(expectation):
    """Lint levels mirroring compiletest's check compilation.

    compiletest compiles a success-expecting compile-check test
    in check mode (--emit=metadata), which never builds runtime
    MIR, so the two lints the runtime-MIR KnownPanicsLint pass
    evaluates (unconditional_panic, arithmetic_overflow -- both
    deny by default) never fire for it. Conversion demands
    runtime-MIR bodies, so without an adjustment they would
    fail compilations the reference accepts (a literal
    out-of-bounds index in code the check compilation never
    evaluates). A failure-expecting test gets no adjustment:
    compiletest's build-fail compilations are full builds, in
    which the pass runs and the lint is often the very failure
    asserted. The test's own compile-flags are appended after
    these, so an explicit level for either lint still wins.
    """
    if expectation != "success":
        return []
    return ["-A", "unconditional_panic", "-A", "arithmetic_overflow"]


def pretty_unpretty_args(suite, view):
    """The -Zunpretty argument compiletest runs a pretty test with.

    A pretty test is never compiled past printing: compiletest
    invokes rustc with `-Zunpretty=<mode>` (the view's pretty-mode
    directive, defaulting to "normal" -- TestProps' own default)
    and judges the print run. Supplying the same argument makes
    the subset judge the compilation compiletest performs: the
    compiler prints and stops, the driver reports the run as an
    empty successful conversion, and sources that are fragments
    for printing (an undefined name, a delegation whose attribute
    validation only runs in later passes) are not held to a
    whole-program conversion compiletest never asks of them. The
    printed-output comparison itself, and the typecheck of the
    printed source for tests without pretty-compare-only, remain
    outside what a conversion outcome can judge.
    """
    if suite != "pretty":
        return []
    modes = view.values("pretty-mode")
    mode = modes[-1].strip() if modes else "normal"
    return [f"-Zunpretty={mode}"]


def stage1_compile_check(rec, view, path, driver_bin, revision, res,
                         edition, rev_args, flags, aux_specs,
                         rust_dir, jit=False):
    """Judge a stage-1 revision view that carries no run mode.

    compiletest gives such a revision a compile-only verdict:
    the ui pass/fail directives visible in the view decide the
    expectation (check-pass / build-pass expect success,
    check-fail / build-fail expect failure), and a ui revision
    with no mode directive at all defaults to check-fail
    (TestProps::from_file in compiletest). The revision is
    judged exactly the way stage2_attempt judges a compile-check
    view -- the driver's convert-only mode under the revision's
    arguments, the view's rustc-env applied to the driver
    process, and judge_compile's outcome comparison -- so the
    same revision receives the same verdict shape in both
    stages.
    """
    expectation = compile_expectation(rec["suite"], view)
    res["expectation"] = expectation
    if expectation is None:
        res["reason"] = ("verdict involves executing the program "
                         "(run-fail-or-crash / run mode), which a "
                         "compile outcome alone cannot judge")
        res["bucket"] = "not attempted (verdict needs execution)"
        return res
    args = list(rev_args)
    if view.has("add-minicore"):
        mc_args, mc_err = minicore_test_args(view, rust_dir, edition,
                                             flags)
        if mc_err is not None:
            res["reason"] = mc_err
            res["bucket"] = "not attempted (minicore build)"
            return res
        args += mc_args
    args += compiletest_mode_args(rec["suite"], flags)
    args += pretty_unpretty_args(rec["suite"], view)
    args += check_mode_lint_args(expectation)
    args += flags
    pm_args, pm_err = proc_macro_extern_args(path, view, edition)
    if pm_err:
        res["reason"] = pm_err
        res["bucket"] = "auxiliary crate build failure"
        return res
    args += pm_args
    cenv, cunset = view_env(view, "rustc-env", "unset-rustc-env",
                            suite_root(path), os.getcwd())
    conv = convert_side(path, edition, driver_bin, rustc_args=args,
                        aux_specs=aux_specs, env_extra=cenv,
                        env_remove=cunset,
                        **({"jit": True} if jit else {}))
    if jit:
        verdict, reason, bucket = judge_jit_compile(expectation, conv)
    else:
        verdict, reason, bucket = judge_compile(expectation, conv)
    res["status"] = verdict
    res["reason"] = reason
    res["bucket"] = bucket
    res["driver"] = {
        "exit_code": conv["exit_code"],
        "converted": conv["converted"],
        "skipped": conv["skipped"],
        "skip_totals": conv["skip_totals"],
        "ice": conv["ice"],
    }
    if jit:
        res["driver"]["jit_compiled"] = conv.get("jit_compiled")
    return res


def stage1_attempt(rec, view, path, driver_bin, revision, all_revisions,
                   rust_dir, jit=False):
    """Judge one revision view of an exec record.

    The stage-1 counterpart of stage2_attempt: the revision is
    compiled and run on both sides under the revision's directive
    view -- its run mode, its compile-flags, and the `--cfg
    <revision>` / `--check-cfg` arguments compiletest adds -- and
    the execution is judged against the native run of the same
    revision. The view's run-flags and exec-env are forwarded to
    both runs the way compiletest applies them. A view with no
    run mode is not an execution at all under compiletest (its
    verdict is a compile outcome, defaulting to check-fail for
    ui tests); stage1_compile_check judges it the way stage 2
    judges compile-check views. Auxiliary crates are resolved
    through the same revision view. Returns a dict with
    status/reason/bucket/mode, plus the native and Sparagmos run
    details on a judged failure. A "not-attempted" result
    carries the bucket that says why; the "out of environment"
    bucket is kept distinct because the revision-combining
    caller treats skipped revisions specially.
    """
    res = {"status": "not-attempted", "reason": "", "bucket": "",
           "mode": None}
    blocker = stage1_view_blocker(view)
    if blocker:
        res["reason"] = blocker
        res["bucket"] = header_bucket(blocker)
        return res
    mode = run_mode(view)
    res["mode"] = mode
    status, why = applicability(view)
    if status == "out-of-environment":
        res["reason"] = why
        res["bucket"] = "out of environment"
        return res
    if status == "unevaluatable":
        res["reason"] = why
        res["bucket"] = "not attempted (unevaluatable condition)"
        return res
    aux_specs = []
    if not rec["single_file"]:
        aux_specs, aux_err = resolve_aux_specs(path, revision)
        if aux_err:
            res["reason"] = aux_err
            res["bucket"] = "out of stage-1 scope (multi-file)"
            return res
    edition = attempt_edition(view)
    src_base = suite_root(path)
    with tempfile.TemporaryDirectory(prefix="subset-run-") as td:
        flags = compile_flags(view, src_base, td)
        screened = screen_flags(flags, targets={HOST_TARGET})
        if screened:
            res["reason"] = screened
            res["bucket"] = "not attempted (compile-flags)"
            return res
        rev_args = revision_args(view, revision, all_revisions)
        args = rev_args + flags
        if mode is None:
            return stage1_compile_check(rec, view, path, driver_bin,
                                        revision, res, edition,
                                        rev_args, flags, aux_specs,
                                        rust_dir, jit=jit)
        pm_args, pm_err = proc_macro_extern_args(path, view, edition)
        if pm_err:
            res["reason"] = pm_err
            res["bucket"] = "auxiliary crate build failure"
            return res
        args = (rev_args + compiletest_mode_args(rec["suite"], flags)
                + flags + pm_args)
        env_pairs, env_unset = view_env(view, "exec-env",
                                        "unset-exec-env",
                                        src_base, td)
        cenv, cunset = view_env(view, "rustc-env", "unset-rustc-env",
                                src_base, td)
        run_flags = view_run_flags(view, src_base, td)
        native, err = native_side(path, edition, td, args, aux_specs,
                                  run_args=run_flags,
                                  run_env_extra=env_pairs,
                                  run_env_remove=env_unset,
                                  compile_env_extra=cenv,
                                  compile_env_remove=cunset,
                                  no_prefer_dynamic=view.has(
                                      "no-prefer-dynamic"))
        if err:
            res["reason"] = err
            res["bucket"] = bucket_reason(err)
            return res
        exp = expected_exit(mode, view)
        if native["exit_code"] != exp:
            res["reason"] = (f"native run exit {native['exit_code']} does "
                             f"not match the mode expectation {exp}")
            res["bucket"] = "native run does not match mode expectation"
            res["native"] = {"exit_code": native["exit_code"]}
            return res
        denv = dict(cenv)
        denv.update(env_pairs)
        dunset = env_unset + [u for u in cunset if u not in env_unset]
        got, kind, reason = sparagmos_side(path, edition, driver_bin,
                                           td, rustc_args=args,
                                           aux_specs=aux_specs,
                                           guest_args=run_flags,
                                           env_extra=denv,
                                           env_remove=dunset,
                                           jit=jit)
        if kind == "unsupported":
            res["status"] = "unsupported"
            res["reason"] = reason
            res["bucket"] = bucket_reason(reason)
            return res
        if kind in ("timeout", "driver-error"):
            res["status"] = "fail"
            res["reason"] = reason
            res["bucket"] = bucket_reason(reason)
            return res
        verdict, detail = judge(mode, view, native, got)
        res["status"] = "pass" if verdict == "pass" else "fail"
        if verdict != "pass":
            res["reason"] = detail
            res["bucket"] = ("stdout mismatch"
                             if detail == "stdout mismatch"
                             else "exit-code mismatch")
            res["native"] = {"exit_code": native["exit_code"],
                             "stdout": native["stdout"][:500]}
            res["sparagmos"] = {"exit_code": got["exit_code"],
                                "stdout": got["stdout"][:500]}
        return res


def fill_result_revisions(rec, header, revs, path, driver_bin, out,
                          rust_dir, jit=False):
    """Stage-1 judging for a test with `//@ revisions:`.

    The test is judged once per revision -- with that revision's
    directive view, its compile-flags, and the `--cfg` compiletest
    adds -- and the record passes only when every revision that
    runs passes (compiletest runs each revision as a separate test
    instance; the subset counts the file once, so the record
    aggregates its instances). A revision an ignore-/only-
    condition removes from this host does not run, as in
    compiletest; if any other revision cannot be attempted, the
    record is not attempted, with that revision's reason.
    """
    per_rev = {}
    for rev in revs:
        per_rev[rev] = stage1_attempt(rec, header.view(rev), path,
                                      driver_bin, rev, revs,
                                      rust_dir, jit=jit)
    out["revisions"] = {
        rev: {k: r[k] for k in ("status", "reason", "bucket", "mode")
              if r.get(k)}
        for rev, r in per_rev.items()
    }
    modes = {r["mode"] for r in per_rev.values() if r["mode"]}
    if len(modes) == 1:
        out["mode"] = modes.pop()
    ran = {rev: r for rev, r in per_rev.items()
           if r["bucket"] != "out of environment"}
    if not ran:
        first = next(iter(per_rev.values()))
        out["reason"] = first["reason"]
        out["bucket"] = "out of environment"
        return
    for rev, r in ran.items():
        if r["status"] == "not-attempted":
            out["reason"] = f"revision {rev}: {r['reason']}"
            out["bucket"] = r["bucket"]
            return
    for wanted in ("fail", "unsupported"):
        for rev, r in ran.items():
            if r["status"] == wanted:
                out["status"] = wanted
                out["reason"] = (f"revision {rev}: {r['reason']}"
                                 if r["reason"] else "")
                out["bucket"] = r["bucket"]
                for key in ("native", "sparagmos"):
                    if key in r:
                        out[key] = r[key]
                return
    out["status"] = "pass"


# ---------------------------------------------------------------------------
# Stage 2: compile-time verification tests (nature == "compile-check")
# ---------------------------------------------------------------------------
#
# Stage 2 judges the compile-check population (ui diagnostics, mir-opt,
# codegen, assembly, pretty, incremental, crashes) at the outcome level:
# the driver's convert-only mode is run on the test and its result is
# compared with the expectation compiletest itself derives from the
# test (see compile_expectation). The driver delegates diagnostics to
# the reference rustc, so bare compile success/failure would be close
# to tautological; the Sparagmos-specific signal is conversion
# completeness, so a success-expecting test only passes when the driver
# compiles it *and* converts every local function (skipped == 0, i.e. a
# complete MirProgram was obtained). A failure-expecting test passes
# when the driver reports a clean compilation failure. A crashes test
# passes when the compilation crashes the driver process the way it
# crashes rustc under compiletest (exit code other than 0 or 1).
#
# Per-test compile-flags are forwarded to the driver's embedded
# rustc, screened for flags that would replace the compilation
# itself or contradict its fixed invocation: a --target naming a
# triple with no installed std (a target-spec `.json` file is
# loaded by rustc itself and always forwarded, and a triple the
# toolchain does not know is forwarded on failure-expecting tests,
# whose asserted failure is rustc's rejection of the specification
# itself), --print, --sysroot, an output location, and -Zunpretty
# (dropped before screening only where the dump cannot take part
# in the verdict — see unpretty_unchecked), and revision tests are
# judged once per revision — with the revision's directive view,
# its compile-flags, and the `--cfg <revision>` / `--check-cfg`
# arguments compiletest adds. The record passes only when every revision that runs
# passes; a revision removed by an ignore-/only- condition does not
# run, as in compiletest.
# A view's `//@ rustc-env:` pairs are applied to the driver process
# environment (and `//@ unset-rustc-env:` names removed from it):
# the driver's embedded rustc is the compiler compiletest would
# set them for, and values a test reads at compile time with
# env! / option_env! take effect in it unchanged.
#
# Tests carrying `//@ add-minicore` are attempted the way compiletest
# runs them: the suite's minicore stub crate
# (tests/auxiliary/minicore.rs) is built with the pinned rustc under
# the test's own edition and compile-flags (plus the
# minicore-compile-flags), and the resulting rlib is forwarded to the
# driver's embedded rustc as `--extern minicore=...` together with
# the `-Cpanic=abort` / `-Cforce-unwind-tables=yes` pair compiletest
# adds for no_core tests. Builds are cached per flag variant; a
# variant that fails to build leaves its tests not attempted with
# that reason, never judged against a missing extern.
# Tests with auxiliary crates are attempted as in stage 1: the aux
# directives are resolved transitively and the crates are handed to
# the driver with `--aux` in dependency order (per revision, when
# the directives are revision-scoped). The driver's target-crate
# report counts only the test crate's own functions, so an
# auxiliary crate's unconverted bodies do not make the conversion
# incomplete; an auxiliary crate that fails to compile fails the
# conversion the way its build failure fails the test under
# compiletest.
#
# What stage 2 deliberately does not do: compare compiler output with
# the test's `.stderr` snapshot, check `//~` error annotations, apply
# rustfix suggestions (`.fixed`), or compare mir-opt `.mir` dumps and
# FileCheck patterns. compiletest's verdict includes those checks;
# reimplementing its diagnostic normalisation faithfully is a separate
# large subsystem, and the diagnostics themselves are produced by the
# reference rustc the driver embeds, so the marginal signal over the
# outcome-level result would measure the normaliser, not Sparagmos.
# Instead, every compile-check record carries `snapshot` flags marking
# which of those verifications its full compiletest verdict would
# require, and the aggregates report how many records (and how many
# stage-2 passes) carry them, so the unverified remainder is explicit.

CONVERT_TIMEOUT = 60

_UI_SUCCESS_DIRECTIVES = ("check-pass", "build-pass")
_UI_FAILURE_DIRECTIVES = ("check-fail", "build-fail")


# Revision kinds of the incremental suite, in the order of
# compiletest's IncrRevKind (src/tools/compiletest/src/runtest/
# incremental.rs): the kind is a prefix of the revision name and
# fixes the compile outcome that revision must produce.
_INCREMENTAL_KINDS = (
    ("cpass", "success"),
    ("bfail", "failure"),
    ("bpass", "success"),
    ("rpass", "success"),
)


def incremental_expectation(view):
    """The compile outcome one incremental revision must produce.

    Mirrors compiletest's run_incremental_test: the revision name's
    kind prefix decides whether that revision's compilation must
    succeed (cpass/bpass/rpass) or fail (bfail), and for bfail the
    failure must be a clean rejection. An rpass revision also runs
    the compiled program in compiletest; stage 2 judges the compile
    outcome only, the same reduction it applies to every suite
    (compilation-output comparison is likewise not performed). A
    revision name with no kind prefix is one compiletest itself
    rejects, and a test without revisions has no per-revision
    outcome at all; both return None (not judgable).
    """
    revision = getattr(view, "revision", None)
    if revision is None:
        return None
    for prefix, expectation in _INCREMENTAL_KINDS:
        if revision.startswith(prefix):
            return expectation
    return None


def compile_expectation(suite, header):
    """The compile outcome compiletest expects of a compile-check test.

    Returns "success" (compilation must succeed), "failure" (it must
    be rejected), "crash" (the compiler must crash, crashes suite), or
    None when the verdict cannot be reduced to a compile outcome
    (run-fail-or-crash involves executing the program).
    """
    if suite == "crashes":
        return "crash"
    if suite == "incremental":
        return incremental_expectation(header)
    if suite in ("ui", "ui-fulldeps"):
        if header.has("run-fail-or-crash"):
            return None
        for name in _UI_SUCCESS_DIRECTIVES:
            if header.has(name):
                return "success"
        for name in _UI_FAILURE_DIRECTIVES:
            if header.has(name):
                return "failure"
        if run_mode(header) is not None:
            return None  # judged by execution, not compilation
        # compiletest: UI tests default to check-fail when no
        # pass/fail directive is present (directives.rs from_file).
        return "failure"
    # mir-opt, codegen-llvm, codegen-units, assembly-llvm, assembly-gcc
    # and pretty tests are all verified against output produced by a
    # successful compilation.
    return "success"


def stage2_blocker(view, rec):
    """Why one revision view of a compile-check record is not
    attempted in stage 2, or None.

    Checked per revision view, so a directive scoped to one revision
    only blocks that revision. compile-flags and revisions are no
    longer blockers: flags are screened separately (screen_flags)
    and forwarded to the driver, and revisions are judged per
    revision by fill_result_stage2.
    """
    if view.has("add-core-stubs"):
        return "add-core-stubs (test links the core stub crate)"
    if view.has("ignore-test"):
        return "ignore-test (compiletest itself never runs this test)"
    return None


_DIR_LISTING_CACHE = {}


def snapshot_flags(path, text):
    """Which output verifications the full compiletest verdict needs.

    `path` is the test file; sibling snapshot files are looked up next
    to it. These flags are informational: stage 2 does not perform the
    checks (see the stage-2 section comment), but reports their counts.
    """
    stem = path.name[:-3] if path.name.endswith(".rs") else path.name
    parent = str(path.parent)
    if parent not in _DIR_LISTING_CACHE:
        try:
            _DIR_LISTING_CACHE[parent] = [p.name for p in path.parent.iterdir()]
        except OSError:
            _DIR_LISTING_CACHE[parent] = []
    sibs = [n for n in _DIR_LISTING_CACHE[parent]
            if n.startswith(stem + ".")]
    header = Header(text)
    return {
        "stderr_snapshot": any(n.endswith(".stderr") for n in sibs),
        "fixed_file": any(n.endswith(".fixed") for n in sibs),
        "mir_snapshot": any(n.endswith(".mir") for n in sibs),
        "error_annotations": "//~" in text,
        "filecheck": re.search(r"//\s*CHECK", text) is not None,
        "run_rustfix": bool(header.has("run-rustfix")
                            or header.has(
                                "run-rustfix-only-machine-applicable")),
    }


_CONVERTED_RE = re.compile(r"^converted: (\d+), skipped: (\d+)$", re.M)
_TARGET_RE = re.compile(r"^target-crate converted: (\d+), skipped: (\d+)$", re.M)


def convert_side(path, edition, driver_bin, rustc_args=(), aux_specs=(),
                 env_extra=None, env_remove=(), jit=False):
    """Run the driver in convert-only mode and parse its report.

    Returns a dict with exit_code (None on timeout), converted and
    skipped counts (None when no report was printed), skip_totals
    ([(kind, count)] from the report's `unsupported:` section, most
    frequent first), compilation_failed (the embedded rustc cleanly
    rejected the program: the driver's own marker, rustc's
    `error: aborting due to` line, or LLVM rejecting a forwarded
    `-Cllvm-args` value before compilation), conversion_panicked
    (the driver's catch_unwind reported a Stable MIR conversion
    panic, with panic_reason), aux_failure (an auxiliary crate's
    build failed, with the driver's reason), ice (stderr shows an
    internal compiler error), and stderr. `rustc_args` are
    forwarded to the driver's embedded rustc through its
    `--rustc-arg` option (a test's compile-flags and the
    per-revision arguments). Auxiliary crates are handed to the
    driver with `--aux`, in dependency order, as in run mode.

    With `jit=True` the driver's `jit-compile` mode replaces
    convert mode: the same conversion and report, plus a JIT
    lowering attempt over the whole converted program. The result
    then also carries jit_compiled (True / False / None when the
    driver printed no JIT line) and jit_reason (the JIT's named
    unsupported reason, when it printed one). In this mode exit
    code 2 means the compilation succeeded but JIT lowering is
    unsupported — it is not a compiler crash.
    """
    rustc_args = list(rustc_args) + sysroot_extern_args(path, rustc_args)
    argv = [str(driver_bin)]
    if jit:
        argv.append("jit-compile")
    argv += [str(path), "--edition", edition]
    for a in rustc_args:
        argv += ["--rustc-arg", a]
    for spec in aux_specs:
        argv += ["--aux", aux_spec_arg(spec)]
    env = dict(BOOTSTRAP_ENV)
    if env_extra:
        env.update(env_extra)
    res = run_cmd(argv,
                  timeout=CONVERT_TIMEOUT, env_extra=env,
                  env_remove=env_remove)
    out = {
        "exit_code": res["exit_code"],
        "timeout": res["timeout"],
        "converted": None,
        "skipped": None,
        "skip_totals": [],
        "compilation_failed": (
            "sparagmos-driver: compilation failed" in res["stderr"]
            or re.search(r"^error: aborting due to", res["stderr"],
                         re.M) is not None
            or "Unknown command line argument" in res["stderr"]),
        "ice": "internal compiler error" in res["stderr"],
        "stderr": res["stderr"],
        "stdout": res["stdout"],
    }
    if "jit-compiled: yes" in res["stdout"]:
        out["jit_compiled"] = True
    elif "jit-compiled: no" in res["stdout"]:
        out["jit_compiled"] = False
    else:
        out["jit_compiled"] = None
    jm = re.search(r"sparagmos-driver jit-compile: unsupported: (.+)",
                   res["stderr"])
    out["jit_reason"] = jm.group(1).strip() if jm else ""
    pm = re.search(r"sparagmos-driver: Stable MIR conversion "
                   r"panicked: (.+)", res["stderr"])
    out["conversion_panicked"] = pm is not None
    out["panic_reason"] = pm.group(1).strip() if pm else ""
    am = re.search(r"sparagmos-driver: auxiliary crate `[^`]+` "
                   r"(failed to compile: .+|produced no linkable "
                   r"artifact[^\n]*)", res["stderr"])
    out["aux_failure"] = am.group(1).strip() if am else ""
    m = _CONVERTED_RE.search(res["stdout"])
    if m:
        out["converted"] = int(m.group(1))
        out["skipped"] = int(m.group(2))
        tm = _TARGET_RE.search(res["stdout"])
        if tm:
            out["target_converted"] = int(tm.group(1))
            out["target_skipped"] = int(tm.group(2))
            ttotals = []
            in_tsection = False
            for line in res["stdout"].splitlines():
                if line == "target-crate unsupported:":
                    in_tsection = True
                    continue
                if in_tsection:
                    mm = re.match(r"^  (.+): (\d+)$", line)
                    if mm:
                        ttotals.append((mm.group(1), int(mm.group(2))))
                    else:
                        in_tsection = False
            out["target_skip_totals"] = sorted(ttotals, key=lambda kv: -kv[1])
        totals = []
        in_section = False
        for line in res["stdout"].splitlines():
            if line == "unsupported:":
                in_section = True
                continue
            if in_section:
                mm = re.match(r"^  (.+): (\d+)$", line)
                if mm:
                    totals.append((mm.group(1), int(mm.group(2))))
                else:
                    in_section = False
        out["skip_totals"] = sorted(totals, key=lambda kv: -kv[1])
    return out


def short_kind(kind):
    """A skip-reason kind without its parenthesised detail."""
    return kind.split(" (")[0]


def skip_bucket(skip_totals):
    """Aggregate bucket for a conversion-incomplete result."""
    if not skip_totals:
        return "conversion incomplete (no skip reason recorded)"
    kind = short_kind(skip_totals[0][0])
    low = kind.lower()
    if "by-reference" in low or "byref" in low:
        return "unsupported constant: by-reference constant"
    if "boxdereftransmute" in low:
        return "conversion incomplete (BoxDerefTransmute cast)"
    if "inlineasm" in low:
        return "conversion incomplete (InlineAsm)"
    return f"conversion incomplete ({kind})"


def judge_compile(expectation, conv):
    """Compare a convert_side result with the expected outcome.

    Returns (status, reason, bucket): status is "pass", "fail" or
    "unsupported" (conversion incomplete on a success-expecting test,
    the stage-1 analogue of the driver reporting exit 2).
    """
    if conv["timeout"] or conv["exit_code"] is None:
        return ("fail",
                f"driver timed out after {CONVERT_TIMEOUT}s",
                "timeout")
    if conv.get("aux_failure"):
        return ("unsupported",
                f"auxiliary crate build failed: "
                f"{conv['aux_failure']}",
                "auxiliary crate build failure")
    code = conv["exit_code"]
    if expectation == "success":
        if code == 0:
            if conv["skipped"] is None:
                return ("fail",
                        "driver exit 0 without a conversion report",
                        "driver error")
            # Conversion completeness is judged on the target
            # crate's own functions when the driver reports them:
            # collected library bodies (Lane AG) are precompiled
            # code, and their skips do not make the target crate's
            # compilation incomplete.
            skipped = conv.get("target_skipped")
            if skipped is None:
                skipped = conv["skipped"]
            if skipped == 0:
                return ("pass", "", "")
            totals_list = conv.get("target_skip_totals") or conv["skip_totals"]
            totals = ", ".join(f"{short_kind(k)} x{n}"
                               for k, n in totals_list[:5])
            return ("unsupported",
                    f"conversion incomplete: {skipped} "
                    f"function(s) skipped ({totals})",
                    skip_bucket(totals_list))
        if code == 1:
            if conv.get("conversion_panicked") \
                    and not conv["compilation_failed"]:
                return ("unsupported",
                        "conversion panicked in Stable MIR: "
                        f"{conv['panic_reason'][:160]}",
                        "conversion incomplete "
                        "(Stable MIR conversion panicked)")
            first = first_error_line(conv["stderr"])
            return ("fail",
                    "expected compile success, but the driver "
                    f"reported compilation failure: {first}",
                    "expected success but compilation failed")
        return ("fail",
                f"driver crashed (exit {code}) on a "
                "success-expecting test",
                "driver crash")
    if expectation == "failure":
        if code == 1 and conv["compilation_failed"]:
            return ("pass", "", "")
        if code == 1 and conv.get("conversion_panicked"):
            return ("unsupported",
                    "conversion panicked in Stable MIR: "
                    f"{conv['panic_reason'][:160]}",
                    "conversion incomplete "
                    "(Stable MIR conversion panicked)")
        if code == 1:
            return ("fail",
                    "driver exit 1 without a compilation-failure report",
                    "driver error")
        if code == 0:
            return ("fail",
                    "expected compile failure, but the driver "
                    "converted the test successfully",
                    "expected failure but conversion succeeded")
        return ("fail",
                f"compiler crashed (exit {code}) instead of a clean "
                "rejection on a failure-expecting test",
                "compiler crash on failure-expecting test")
    if expectation == "crash":
        if code not in (0, 1):
            return ("pass", "", "")
        if code == 0:
            return ("fail",
                    "no longer crashes: the driver converted the test",
                    "no crash (converted successfully)")
        return ("fail",
                "no longer crashes: clean compilation failure",
                "no crash (clean rejection)")
    raise ValueError(f"unknown expectation {expectation!r}")


def jit_bucket(reason):
    """Aggregate bucket for a JIT-lowering unsupported reason."""
    r = reason.lower()
    if "not part of the program" in r:
        return "JIT lowering unsupported (callee not in program)"
    if "backend error" in r:
        return "JIT lowering unsupported (backend error)"
    return "JIT lowering unsupported (other)"


def judge_jit_compile(expectation, conv):
    """Compare a jit-compile result with the expected outcome.

    Metric 4's compile-check verdict: the conversion outcome must
    match the expectation exactly as in stage 2 (judge_compile),
    and — on a success-expecting test whose conversion is complete
    — the JIT lowering of the whole converted program must also
    succeed. A named JIT unsupported is reported as unsupported
    (never a pass). On a failure- or crash-expecting test there is
    no successfully converted program to lower, so the conversion
    outcome alone decides, exactly as in stage 2. In jit-compile
    mode exit code 2 is the defined JIT-unsupported outcome, not
    a compiler crash, so it is intercepted before judge_compile
    (which reads any non-0/1 exit as a crash) can see it.
    """
    if conv["timeout"] or conv["exit_code"] is None:
        return ("fail",
                f"driver timed out after {CONVERT_TIMEOUT}s",
                "timeout")
    if conv.get("aux_failure"):
        return ("unsupported",
                f"auxiliary crate build failed: "
                f"{conv['aux_failure']}",
                "auxiliary crate build failure")
    if conv["exit_code"] == 2:
        if conv.get("jit_compiled") is not False:
            return ("fail",
                    "driver exit 2 without a JIT lowering report",
                    "driver error")
        reason = conv.get("jit_reason") or "JIT lowering unsupported"
        if expectation == "success":
            skipped = conv.get("target_skipped")
            if skipped is None:
                skipped = conv["skipped"]
            if skipped is None:
                return ("fail",
                        "driver exit 2 without a conversion report",
                        "driver error")
            if skipped != 0:
                totals_list = conv.get("target_skip_totals") \
                    or conv["skip_totals"]
                totals = ", ".join(f"{short_kind(k)} x{n}"
                                   for k, n in totals_list[:5])
                return ("unsupported",
                        f"conversion incomplete: {skipped} "
                        f"function(s) skipped ({totals})",
                        skip_bucket(totals_list))
            return ("unsupported",
                    f"JIT lowering unsupported: {reason[:300]}",
                    jit_bucket(reason))
        if expectation == "failure":
            return ("fail",
                    "expected compile failure, but the driver "
                    "converted the test successfully (a JIT "
                    "lowering failure is not a compilation "
                    "rejection)",
                    "expected failure but conversion succeeded")
        if expectation == "crash":
            return ("fail",
                    "no longer crashes: the compilation succeeded "
                    "and only the JIT lowering is unsupported",
                    "no crash (JIT lowering unsupported)")
        raise ValueError(f"unknown expectation {expectation!r}")
    verdict, reason, bucket = judge_compile(expectation, conv)
    if verdict == "pass" and expectation == "success":
        if conv.get("jit_compiled") is True:
            return (verdict, reason, bucket)
        if conv.get("jit_compiled") is False:
            jreason = conv.get("jit_reason") or "JIT lowering unsupported"
            return ("unsupported",
                    f"JIT lowering unsupported: {jreason[:300]}",
                    jit_bucket(jreason))
        return ("fail",
                "driver exit 0 without a JIT lowering report",
                "driver error")
    return (verdict, reason, bucket)


MINICORE_BUILD_TIMEOUT = 120

_MINICORE_LOCK = threading.Lock()
_MINICORE_RESULTS = {}


def minicore_cache_dir():
    """Where prebuilt minicore rlibs are cached, keyed by content and
    flag variant so a re-run reuses them across processes."""
    env = os.environ.get("SPARGASMOS_MINICORE_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "sparagmos-minicore"


def _is_rlib(path):
    try:
        with open(path, "rb") as f:
            return f.read(8) == b"!<arch>\n"
    except OSError:
        return False


def minicore_rlib(rust_dir, edition, flags, minicore_flags):
    """The prebuilt minicore rlib for one flag variant.

    Returns (path, None) on success, or (None, reason) when the build
    fails. compiletest builds minicore per test, with the test's own
    compile-flags; the runner caches per (edition, compile-flags,
    minicore-compile-flags) variant instead, which is the same set of
    distinct builds. A failed build is reported, never papered over:
    judging a test against a missing `--extern` would fabricate
    passes on failure-expecting tests.
    """
    key = (edition, tuple(flags), tuple(minicore_flags))
    with _MINICORE_LOCK:
        if key in _MINICORE_RESULTS:
            return _MINICORE_RESULTS[key]
    result = _build_minicore(rust_dir, edition, flags, minicore_flags)
    with _MINICORE_LOCK:
        _MINICORE_RESULTS.setdefault(key, result)
        return _MINICORE_RESULTS[key]


def _build_minicore(rust_dir, edition, flags, minicore_flags):
    src = rust_dir / "tests" / "auxiliary" / "minicore.rs"
    if not src.exists():
        return (None, f"minicore source not found at {src}")
    digest = hashlib.sha256()
    digest.update(src.read_bytes())
    digest.update(repr((edition, list(flags),
                        list(minicore_flags))).encode())
    out_dir = minicore_cache_dir() / digest.hexdigest()[:16]
    rlib = out_dir / "libminicore.rlib"
    if _is_rlib(rlib):
        return (rlib, None)
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=out_dir, suffix=".rlib")
    os.close(fd)
    tmp = Path(tmp_name)
    # The flag order mirrors compiletest's build_minicore: the
    # panic/unwind pair and the test's compile-flags come from
    # make_compile_args, then `--crate-type rlib`, a final
    # `-Cpanic=abort` and the minicore-compile-flags are appended.
    argv = rustc_cmd() + [
        str(src),
        "--edition", edition,
        "-Cpanic=abort",
        "-Cforce-unwind-tables=yes",
        *flags,
        "--crate-type", "rlib",
        "-Cpanic=abort",
        *minicore_flags,
        "-o", str(tmp),
    ]
    res = run_cmd(argv, timeout=MINICORE_BUILD_TIMEOUT,
                  env_extra=BOOTSTRAP_ENV)
    if res["timeout"] or res["exit_code"] != 0 or not _is_rlib(tmp):
        tmp.unlink(missing_ok=True)
        first = first_error_line(res["stderr"])
        return (None, "minicore build failed (edition "
                f"{edition}, compile-flags {list(flags)}, "
                f"minicore-compile-flags {list(minicore_flags)}): "
                f"{first or 'no rlib produced'}")
    os.replace(tmp, rlib)
    return (rlib, None)


def minicore_test_args(view, rust_dir, edition, flags):
    """The extra rustc arguments an add-minicore view's conversion
    needs: the panic/unwind pair and `--extern minicore=<rlib>`.

    Returns (args, None), or (None, reason) when the arguments cannot
    be produced (a minicore-compile-flags `--edition`, which
    compiletest itself rejects, or a failed minicore build); the
    caller reports the test as not attempted with that reason.
    """
    mc_flags = []
    for v in view.values("minicore-compile-flags"):
        mc_flags.extend(split_flags(v))
    for t in mc_flags:
        if t == "--edition" or t.startswith("--edition="):
            return (None, "minicore-compile-flags specify --edition "
                    "(compiletest rejects this; the edition directive "
                    "governs both compilations)")
    rlib, err = minicore_rlib(rust_dir, edition, flags, mc_flags)
    if err is not None:
        return (None, err)
    return (["-Cpanic=abort", "-Cforce-unwind-tables=yes",
             "--extern", f"minicore={rlib}"], None)


def stage2_attempt(rec, view, path, driver_bin, revision, all_revisions,
                   rust_dir, aux_specs=(), aux_per_revision=False,
                   jit=False):
    """Judge one directive view of a compile-check record.

    `revision` is the revision name this view belongs to (None for a
    test without revisions) and `all_revisions` the test's declared
    revisions. `aux_specs` are the test's auxiliary crates, resolved
    by the caller; when `aux_per_revision` is set the test's aux
    directives are revision-scoped and each view resolves its own
    set instead. Returns a dict with status/reason/bucket, plus the
    expectation and driver details when the conversion ran. A
    "not-attempted" result carries the bucket that says why; the
    "out of environment" bucket is kept distinct because the
    revision-combining caller treats skipped revisions specially.
    """
    res = {"status": "not-attempted", "reason": "", "bucket": ""}
    blocker = stage2_blocker(view, rec)
    if blocker:
        res["reason"] = blocker
        res["bucket"] = header_bucket(blocker)
        return res
    status, why = applicability(view)
    if status == "out-of-environment":
        res["reason"] = why
        res["bucket"] = "out of environment"
        return res
    if status == "unevaluatable":
        res["reason"] = why
        res["bucket"] = "not attempted (unevaluatable condition)"
        return res
    expectation = compile_expectation(rec["suite"], view)
    res["expectation"] = expectation
    if expectation is None and run_mode(view) not in ("run-pass",
                                                     "run-fail"):
        if rec["suite"] == "incremental":
            if revision is None:
                res["reason"] = ("incremental test declares no "
                                 "revisions, so there is no "
                                 "per-revision compile outcome to "
                                 "judge (compiletest requires a "
                                 "revision list for this suite)")
            else:
                res["reason"] = (f"revision name {revision!r} does "
                                 "not start with an incremental kind "
                                 "(cpass/bfail/bpass/rpass), which "
                                 "compiletest itself rejects")
        else:
            res["reason"] = ("verdict involves executing the program "
                             "(run-fail-or-crash / run mode), which a "
                             "compile outcome alone cannot judge")
        res["bucket"] = "not attempted (verdict needs execution)"
        return res
    edition = attempt_edition(view)
    flags = compile_flags(view, suite_root(path), os.getcwd())
    if has_unpretty(flags) and unpretty_unchecked(
            view, path, revision, expectation):
        flags = strip_unpretty(flags)
    screened = screen_flags(flags, known=known_targets() or None,
                            expectation=expectation)
    if screened:
        res["reason"] = screened
        res["bucket"] = "not attempted (compile-flags)"
        return res
    args = []
    if revision is not None:
        args += revision_args(view, revision, all_revisions)
    if view.has("add-minicore"):
        mc_args, mc_err = minicore_test_args(view, rust_dir, edition,
                                             flags)
        if mc_err is not None:
            res["reason"] = mc_err
            res["bucket"] = "not attempted (minicore build)"
            return res
        args += mc_args
    # Crate type: the driver's convert mode names none of its own
    # (compiletest parity: compiletest passes no --crate-type for a
    # test file, so a source-level #![crate_type] or the rustc
    # default (bin) governs). Two suites are prescribed by
    # compiletest itself and forwarded here. mir-opt is compiled as
    # an rlib (make_compile_args passes --crate-type=rlib for the
    # suite). A pretty test is never compiled as a binary at all --
    # compiletest stops at -Zunpretty output, so no entry-point or
    # reachability verdict exists to mirror; the conversion proxy
    # keeps the lib shape for it (unless the test declares its own
    # crate type, in flags or in the source). A test whose own
    # compile-flags name a crate type keeps them in both cases.
    if not any(a == "--crate-type" or a.startswith("--crate-type=")
               for a in flags):
        if rec["suite"] == "mir-opt":
            args += ["--crate-type", "rlib"]
        elif rec["suite"] == "pretty" and not declared_crate_type(
                path.read_text(errors="ignore")):
            args += ["--crate-type", "lib"]
    args += compiletest_mode_args(rec["suite"], flags)
    args += pretty_unpretty_args(rec["suite"], view)
    args += check_mode_lint_args(expectation)
    args += flags
    specs = aux_specs
    if aux_per_revision:
        # The test's aux directives are scoped to revisions: this
        # view builds the auxiliary crates its own directives name
        # (a view naming none converts without auxiliaries).
        if aux_directives(view):
            specs, aux_err = resolve_aux_specs(path, top=view,
                                               top_proc_macro=False)
            if aux_err:
                res["reason"] = aux_err
                res["bucket"] = "out of stage-2 scope (multi-file)"
                return res
        else:
            specs = []
    pm_args, pm_err = proc_macro_extern_args(path, view, edition)
    if pm_err:
        res["reason"] = pm_err
        res["bucket"] = "auxiliary crate build failure"
        return res
    args += pm_args
    if expectation is None:
        # A run-pass / run-fail view: compiletest's verdict for
        # it is the execution, judged under the same arguments
        # this conversion would have used.
        return stage2_run_attempt(rec, view, path, driver_bin, res,
                                  edition, args, flags, specs, jit=jit)
    # The view's `//@ rustc-env:` pairs are applied to the driver
    # process: the driver's embedded rustc reads them exactly as
    # compiletest's rustc invocation does (including values a
    # test reads back with env! / option_env!), and
    # `//@ unset-rustc-env:` names are removed from it.
    cenv, cunset = view_env(view, "rustc-env", "unset-rustc-env",
                            suite_root(path), os.getcwd())
    conv = convert_side(path, edition, driver_bin, rustc_args=args,
                        aux_specs=specs, env_extra=cenv,
                        env_remove=cunset,
                        **({"jit": True} if jit else {}))
    if jit:
        verdict, reason, bucket = judge_jit_compile(expectation, conv)
    else:
        verdict, reason, bucket = judge_compile(expectation, conv)
    res["status"] = verdict
    res["reason"] = reason
    res["bucket"] = bucket
    res["driver"] = {
        "exit_code": conv["exit_code"],
        "converted": conv["converted"],
        "skipped": conv["skipped"],
        "skip_totals": conv["skip_totals"],
        "ice": conv["ice"],
    }
    if jit:
        res["driver"]["jit_compiled"] = conv.get("jit_compiled")
    if verdict == "fail":
        res["driver"]["stderr_first"] = first_error_line(conv["stderr"])
    return res


def stage2_run_attempt(rec, view, path, driver_bin, res, edition,
                       args, flags, aux_specs, jit=False):
    """Judge a stage-2 view whose verdict is a program run.

    The view's mode is run-pass / run-fail (a scoped mode
    directive, or a revision name compiletest's ui mode reads as
    one), so compiletest compiles and *runs* the revision and
    judges the execution; no compile outcome decides it. The
    judging mirrors stage 1's execution of an exec record under
    the same view exactly -- native compile and run under the
    revision's arguments, the native exit checked against the
    mode expectation, the driver in run mode under the same
    arguments, exit codes and stdout compared -- so the same
    revision receives the same verdict in both stages. `args`
    are the arguments stage2_attempt assembled for the view's
    conversion; the native compile and the driver's run both
    take them, as compiletest compiles a run test with its
    mode arguments. Execution is host-only, so the flags are
    screened again against the host target alone. run-crash and
    run-fail-or-crash views never reach here: crash reproduction
    is out of the subset's scope in stage 1 as well, and they
    keep the not-attempted verdict compile_expectation's None
    produced for them.
    """
    mode = run_mode(view)
    res["expectation"] = mode
    screened = screen_flags(flags, targets={HOST_TARGET})
    if screened:
        res["reason"] = screened
        res["bucket"] = "not attempted (compile-flags)"
        return res
    src_base = suite_root(path)
    with tempfile.TemporaryDirectory(prefix="subset-run-") as td:
        env_pairs, env_unset = view_env(view, "exec-env",
                                        "unset-exec-env",
                                        src_base, td)
        cenv, cunset = view_env(view, "rustc-env", "unset-rustc-env",
                                src_base, td)
        run_flags = view_run_flags(view, src_base, td)
        native, err = native_side(path, edition, td, args, aux_specs,
                                  run_args=run_flags,
                                  run_env_extra=env_pairs,
                                  run_env_remove=env_unset,
                                  compile_env_extra=cenv,
                                  compile_env_remove=cunset,
                                  no_prefer_dynamic=view.has(
                                      "no-prefer-dynamic"))
        if err:
            res["reason"] = err
            res["bucket"] = bucket_reason(err)
            return res
        exp = expected_exit(mode, view)
        if native["exit_code"] != exp:
            res["reason"] = (f"native run exit {native['exit_code']} does "
                             f"not match the mode expectation {exp}")
            res["bucket"] = "native run does not match mode expectation"
            res["native"] = {"exit_code": native["exit_code"]}
            return res
        denv = dict(cenv)
        denv.update(env_pairs)
        dunset = env_unset + [u for u in cunset if u not in env_unset]
        got, kind, reason = sparagmos_side(path, edition, driver_bin,
                                           td, rustc_args=args,
                                           aux_specs=aux_specs,
                                           guest_args=run_flags,
                                           env_extra=denv,
                                           env_remove=dunset,
                                           jit=jit)
        if kind == "unsupported":
            res["status"] = "unsupported"
            res["reason"] = reason
            res["bucket"] = bucket_reason(reason)
            return res
        if kind in ("timeout", "driver-error"):
            res["status"] = "fail"
            res["reason"] = reason
            res["bucket"] = bucket_reason(reason)
            return res
        verdict, detail = judge(mode, view, native, got)
        res["status"] = "pass" if verdict == "pass" else "fail"
        if verdict != "pass":
            res["reason"] = detail
            res["bucket"] = ("stdout mismatch"
                             if detail == "stdout mismatch"
                             else "exit-code mismatch")
            res["native"] = {"exit_code": native["exit_code"],
                             "stdout": native["stdout"][:500]}
            res["sparagmos"] = {"exit_code": got["exit_code"],
                                "stdout": got["stdout"][:500]}
        return res


# ---------------------------------------------------------------------------
# Incremental suite: verdicts defined over cross-session state
# ---------------------------------------------------------------------------
#
# compiletest compiles an incremental test's revisions one after
# another into a single shared incremental directory, so a later
# revision starts from the earlier sessions' artifacts. Most
# revisions' verdicts still reduce to a per-revision compile outcome
# (see incremental_expectation): a warm cache does not change
# whether a revision's source compiles. Two families are the
# exception -- their verdict is *defined* over the state carried
# between sessions, so independent per-revision conversions cannot
# judge them:
#
# - Dep-graph assertions in the test's sources: #[rustc_clean] /
#   #[rustc_dirty] on items, #![rustc_partition_reused] /
#   #![rustc_partition_codegened] on the crate. Under
#   -Z query-dep-graph rustc checks them against the dep graph
#   loaded from the previous session: an item asserted clean whose
#   fingerprint a source change altered is a compile error in that
#   revision, and a codegen partition asserted reused that the
#   cache does not hold is one too. A fresh conversion has no
#   previous session, so the assertions cannot take their
#   compiletest meaning.
# - A revision whose compile-flags assert the cache state at
#   startup: -Zassert-incr-state=loaded requires the previous
#   session's cache to be present and loaded. (=not-loaded asserts
#   the opposite, which a fresh conversion satisfies by
#   construction, so it does not block.)

_DEP_GRAPH_ASSERTION_RE = re.compile(
    r"#!?\[\s*rustc_(clean|dirty|partition_reused|partition_codegened)\b")
_ASSERT_INCR_STATE_LOADED_RE = re.compile(
    r"assert-incr-state\s*=\s*loaded\b")


def _aux_source_texts(test_path):
    """(path, text) of the auxiliary crate sources a test builds.

    Best effort, for classification only: the transitively resolved
    specs when resolution succeeds, plus the files the test's own
    aux directives (scoped ones included) name, found where
    compiletest looks for them.
    """
    out = []
    seen = set()

    def add(path):
        if path not in seen and path.is_file():
            seen.add(path)
            try:
                out.append((path, path.read_text(errors="ignore")))
            except OSError:
                pass

    specs, _err = resolve_aux_specs(test_path, top_proc_macro=False)
    for spec in specs:
        add(spec.path)
    header = Header(test_path.read_text(errors="ignore"))
    fnames = []
    for name, values in ([(n, header.unscoped.get(n, []))
                          for n in ("aux-build", "aux-crate",
                                    "proc-macro")]
                         + [(n, v) for (r, n), v in
                            header.scoped.items()
                            if n in ("aux-build", "aux-crate",
                                     "proc-macro")]):
        for value in values:
            if name == "aux-crate":
                parsed = parse_aux_crate_value(value)
                fname = parsed[2] if parsed else ""
            elif name == "proc-macro":
                fname = parse_proc_macro_value(value)[1]
            else:
                fname = value.strip()
            if fname:
                fnames.append(fname)
    test_dir = test_path.parent
    for fname in fnames:
        for base in (test_dir / "auxiliary", test_dir):
            cand = base / fname
            if cand.is_file():
                add(cand)
                break
    return out


def incremental_cross_session_blocker(test_path, text):
    """Why an incremental test's verdict depends on state carried
    between compilation sessions, or None.

    Returns (reason, bucket). The check covers the test's own file
    and the auxiliary crates it builds: any dep-graph assertion, or
    a compile-flags assertion that the previous session's cache is
    loaded at startup, makes the verdict cross-session by
    definition.
    """
    sources = [(test_path, text)] + _aux_source_texts(test_path)
    for src_path, src_text in sources:
        m = _DEP_GRAPH_ASSERTION_RE.search(src_text)
        if m:
            attr = "rustc_" + m.group(1)
            where = ("the test file" if src_path == test_path
                     else f"auxiliary file {src_path.name}")
            return (f"dep-graph assertion #[{attr}] in {where}: "
                    "under -Z query-dep-graph the verdict compares "
                    "dep-graph state against the previous "
                    "compilation session, which independent "
                    "per-revision conversions cannot reproduce",
                    "out of stage-2 scope (incremental dep-graph "
                    "assertions)")
    for line in text.splitlines():
        if line.lstrip().startswith("//@") and \
                _ASSERT_INCR_STATE_LOADED_RE.search(line):
            return ("compile-flags assert the incremental cache "
                    "state at startup "
                    "(-Zassert-incr-state=loaded): the revision "
                    "must start from the previous session's cache, "
                    "which independent per-revision conversions do "
                    "not have",
                    "out of stage-2 scope (incremental cache-state "
                    "assertion)")
    return None


def fill_result_stage2(rec, rust_dir, driver_bin, out, jit=False):
    """Stage-2 counterpart of fill_result for compile-check records.

    A test without revisions is judged once, on its unscoped view.
    A revision test is judged once per revision — with that
    revision's directive view, its compile-flags, and the `--cfg`
    compiletest adds — and the record passes only when every
    revision that runs passes (compiletest runs each revision as a
    separate test instance; the subset counts the file once, so the
    record aggregates its instances). A revision an ignore-/only-
    condition removes from this host does not run, as in
    compiletest; if any other revision cannot be attempted, the
    record is not attempted, with that revision's reason.
    """
    out["stage"] = 2
    path = rust_dir / rec["path"]
    text = path.read_text(errors="ignore")
    out["snapshot"] = snapshot_flags(path, text)
    if rec["suite"] == "incremental":
        # Most incremental revisions reduce to a per-revision
        # compile outcome and are judged below like any other
        # revision test; only verdicts defined over state carried
        # between compilation sessions stay out of scope, each with
        # its own reason.
        blocker = incremental_cross_session_blocker(path, text)
        if blocker:
            out["reason"], out["bucket"] = blocker
            return
    header = Header(text)
    # Auxiliary crates, as in stage 1: the test's aux directives are
    # resolved transitively and the crates handed to the driver with
    # --aux, in dependency order. When every aux directive is scoped
    # to a revision, resolution happens per revision in
    # stage2_attempt instead. A non-single-file test whose files no
    # directive accounts for stays out of scope.
    aux_specs = []
    aux_per_revision = False
    if not rec["single_file"]:
        aux_specs, aux_err = resolve_aux_specs(path,
                                               top_proc_macro=False)
        if aux_err:
            out["reason"] = aux_err
            out["bucket"] = "out of stage-2 scope (multi-file)"
            return
        if not aux_specs:
            if any(name in ("aux-build", "aux-crate")
                   for (_rev, name) in header.scoped):
                aux_per_revision = True
            elif header.has("proc-macro") or any(
                    name == "proc-macro"
                    for (_rev, name) in header.scoped):
                # proc-macro auxiliaries are built by the runner
                # itself and forwarded as --extern inside
                # stage2_attempt; they never go through --aux.
                pass
            else:
                out["reason"] = ("multi-file / auxiliary crates "
                                 "(not in stage-2 scope)")
                out["bucket"] = "out of stage-2 scope (multi-file)"
                return
    revs = header.revisions()
    if not revs:
        res = stage2_attempt(rec, header.view(None), path, driver_bin,
                             None, [], rust_dir, aux_specs,
                             aux_per_revision, jit=jit)
        for key in ("status", "reason", "bucket", "expectation",
                    "driver", "native", "sparagmos"):
            if key in res:
                out[key] = res[key]
        return
    per_rev = {}
    for rev in revs:
        per_rev[rev] = stage2_attempt(rec, header.view(rev), path,
                                      driver_bin, rev, revs, rust_dir,
                                      aux_specs, aux_per_revision,
                                      jit=jit)
    out["revisions"] = {
        rev: {k: r[k] for k in ("status", "reason", "bucket",
                                "expectation") if r.get(k)}
        for rev, r in per_rev.items()
    }
    expectations = {r.get("expectation") for r in per_rev.values()}
    if len(expectations) == 1:
        out["expectation"] = expectations.pop()
    ran = {rev: r for rev, r in per_rev.items()
           if r["bucket"] != "out of environment"}
    if not ran:
        first = next(iter(per_rev.values()))
        out["reason"] = first["reason"]
        out["bucket"] = "out of environment"
        return
    for rev, r in ran.items():
        if r["status"] == "not-attempted":
            out["reason"] = f"revision {rev}: {r['reason']}"
            out["bucket"] = r["bucket"]
            return
    for wanted in ("fail", "unsupported"):
        for rev, r in ran.items():
            if r["status"] == wanted:
                out["status"] = wanted
                out["reason"] = (f"revision {rev}: {r['reason']}"
                                 if r["reason"] else "")
                out["bucket"] = r["bucket"]
                if "driver" in r:
                    out["driver"] = r["driver"]
                for key in ("native", "sparagmos"):
                    if key in r:
                        out[key] = r[key]
                return
    out["status"] = "pass"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stage 3: procedure tests (run-make recipes)
# ---------------------------------------------------------------------------

STAGE3_NATIVE_TIMEOUT = 300
STAGE3_TIMEOUT = 600
STAGE3_SUPPORT_TIMEOUT = 1800

_stage3_lock = threading.Lock()
_stage3_support = None       # (flat_dir, support_rlib) once built
_stage3_support_error = None
_stage3_paths = None         # pinned toolchain paths


def stage3_toolchain_paths():
    """The pinned 1.100.0 toolchain's rustc/rustdoc/cargo paths.

    Recipes read the compiler's path from the `RUSTC` environment
    variable, and the driver's tool interception matches the
    invoked program against `--tool-program`, so both sides must
    name the very same file: the toolchain's own rustc binary,
    not a rustup shim (a shim would be a different program path
    than the one the recipe was told about only if the two
    disagree -- here both are this path by construction).
    """
    global _stage3_paths
    if _stage3_paths is None:
        rustup = Path.home() / ".cargo" / "bin" / "rustup"
        base = [str(rustup), "which", "--toolchain", TOOLCHAIN] \
            if rustup.exists() else ["rustup", "which",
                                     "--toolchain", TOOLCHAIN]
        out = {}
        for tool in ("rustc", "rustdoc", "cargo"):
            res = run_cmd(base + [tool], timeout=30)
            out[tool] = res["stdout"].strip()
        # Two dylib dirs matter: the sysroot's own lib dir
        # (librustc_driver / libLLVM for the compiler binaries)
        # and the host target's lib dir (libstd.so for
        # prefer-dynamic recipe binaries and produced exes).
        root = Path(out["rustc"]).parent.parent
        out["sysroot_lib"] = str(root / "lib")
        out["target_lib"] = str(
            root / "lib" / "rustlib" / "x86_64-unknown-linux-gnu" / "lib")
        _stage3_paths = out
    return _stage3_paths


def ensure_stage3_support(rust_dir):
    """Build run-make-support (debug profile) once per process.

    Returns (flat_dir, support_rlib): a directory holding the
    support rlib plus every transitive dependency rlib/dylib in
    one flat `-L dependency` dir (the layout the driver's
    `--aux-lib` binding expects: the rlib's parent is the deps
    dir). The build tree is cached under the system temp dir,
    keyed by the rust tree's path, so later runs reuse it.
    Raises RuntimeError with the build output on failure.
    """
    global _stage3_support, _stage3_support_error
    with _stage3_lock:
        if _stage3_support is not None:
            return _stage3_support
        if _stage3_support_error is not None:
            raise RuntimeError(_stage3_support_error)
        try:
            _stage3_support = _build_stage3_support(rust_dir)
        except RuntimeError as e:
            _stage3_support_error = str(e)
            raise
        return _stage3_support


def _build_stage3_support(rust_dir):
    key = hashlib.sha256(str(rust_dir).encode()).hexdigest()[:16]
    root = Path(tempfile.gettempdir()) / f"sparagmos-stage3-support-{key}"
    flat = root / "flat"
    rlib = flat / "librun_make_support.rlib"
    if rlib.exists():
        return flat, rlib
    root.mkdir(parents=True, exist_ok=True)
    # Extract the support crate and its in-tree path deps (plus
    # src/stage0, which build_helper includes as text).
    archive = subprocess.run(
        ["git", "-C", str(rust_dir), "archive", "HEAD",
         "src/tools/run-make-support", "src/build_helper",
         "src/rustdoc-json-types", "src/stage0"],
        capture_output=True, timeout=120)
    if archive.returncode != 0:
        raise RuntimeError("git archive of run-make-support failed: "
                           + archive.stderr.decode("utf-8", "replace")[:300])
    subprocess.run(["tar", "-x", "-C", str(root)], input=archive.stdout,
                   check=True, timeout=120)
    crate_dir = root / "src" / "tools" / "run-make-support"
    env = dict(os.environ)
    env["RUSTC_BOOTSTRAP"] = "1"
    env["RUSTFLAGS"] = "-Zalways-encode-mir"
    rustup = Path.home() / ".cargo" / "bin" / "rustup"
    cargo = [str(rustup), "run", TOOLCHAIN, "cargo"] \
        if rustup.exists() else ["rustup", "run", TOOLCHAIN, "cargo"]
    build = subprocess.run(cargo + ["build"], cwd=crate_dir, env=env,
                           capture_output=True, text=True,
                           timeout=STAGE3_SUPPORT_TIMEOUT)
    if build.returncode != 0:
        raise RuntimeError("run-make-support build failed: "
                           + (build.stderr or "")[-500:])
    target = crate_dir / "target" / "debug"
    flat.mkdir(parents=True, exist_ok=True)
    for f in flat.iterdir():
        f.unlink()
    copied = 0
    for cand in [target / "librun_make_support.rlib"]:
        if cand.exists():
            shutil.copy2(cand, flat / cand.name)
            copied += 1
    build_dir = target / "build"
    if build_dir.exists():
        for out_dir in build_dir.glob("*/*/out"):
            for f in out_dir.iterdir():
                if f.suffix in (".rlib", ".so", ".dylib"):
                    shutil.copy2(f, flat / f.name)
                    copied += 1
    if not rlib.exists() or copied < 2:
        raise RuntimeError(
            f"run-make-support artifacts incomplete in {flat} "
            f"({copied} files gathered)")
    return flat, rlib


def stage3_recipe_env(rust_dir, build_root, suite):
    """The environment compiletest runs a recipe with.

    Mirrors compiletest's run-make setup (runtest/run_make.rs):
    the pinned toolchain's paths, the target, the source/build
    roots, the C toolchain names, and the dylib path variables
    run-make-support's helpers require.
    """
    paths = stage3_toolchain_paths()
    env = {
        "RUSTC": paths["rustc"],
        "RUSTDOC": paths["rustdoc"],
        "TARGET": "x86_64-unknown-linux-gnu",
        "SOURCE_ROOT": str(rust_dir),
        "BUILD_ROOT": str(build_root),
        "PYTHON": sys.executable or "python3",
        "CC": "cc",
        "CXX": "c++",
        "AR": "ar",
        "CC_DEFAULT_FLAGS": "",
        "CXX_DEFAULT_FLAGS": "",
        "LLVM_COMPONENTS": "",
        "LD_LIB_PATH_ENVVAR": "LD_LIBRARY_PATH",
        "LD_LIBRARY_PATH": paths["sysroot_lib"] + ":" + paths["target_lib"],
        "HOST_RUSTC_DYLIB_PATH": paths["sysroot_lib"],
        "TARGET_EXE_DYLIB_PATH": paths["target_lib"],
        "__BOOTSTRAP_JOBS": "8",
        # compiletest itself runs under the build's bootstrap
        # environment, and recipes routinely pass `-Z` flags to
        # the compiler they invoke; without this the pinned
        # release-channel rustc rejects them all.
        "RUSTC_BOOTSTRAP": "1",
    }
    if suite in ("run-make-cargo", "build-std"):
        env["CARGO"] = paths["cargo"]
    return env


def fill_result_procedure(rec, rust_dir, driver_bin, out):
    """Stage 3: run one run-make recipe natively (the oracle)
    and under the driver with tool interception.

    The recipe passes natively first -- a recipe that fails
    natively is a plain failure, exactly as compiletest would
    report it, and is not attempted under the driver. Under the
    driver, the recipe's compiler invocations are answered by
    the driver's own conversion machinery (the tool
    interception), never by real child processes: exit 0 is a
    pass, the driver's named-unsupported exit 2 is an
    unsupported result, and any other exit is a failure.
    """
    out["stage"] = 3
    test_dir = rust_dir / rec["path"]
    rmake_rs = test_dir / "rmake.rs"
    if not rmake_rs.exists():
        out["reason"] = "no rmake.rs in the test directory"
        out["bucket"] = "out of stage-3 scope (no recipe)"
        return
    header = Header(rmake_rs.read_text(errors="ignore"))
    status, why = applicability(header)
    if status == "out-of-environment":
        out["reason"] = why
        out["bucket"] = "out of environment"
        return
    if status == "unevaluatable":
        out["reason"] = why
        out["bucket"] = "not attempted (unevaluatable condition)"
        return
    if header.revisions():
        out["reason"] = "recipe declares revisions (stage 3 runs the plain recipe only)"
        out["bucket"] = "out of stage-3 scope (revisions)"
        return
    try:
        flat, rlib = ensure_stage3_support(rust_dir)
    except RuntimeError as e:
        out["reason"] = str(e)
        out["bucket"] = "stage-3 support build failure"
        return
    paths = stage3_toolchain_paths()
    with tempfile.TemporaryDirectory(prefix="subset-stage3-") as td:
        td = Path(td)
        # compiletest copies every test-dir entry except rmake.rs
        # itself into the recipe's working directory (rmake_out).
        oracle_out = td / "oracle" / "rmake_out"
        guest_out = td / "guest" / "rmake_out"
        for dest in (oracle_out, guest_out):
            dest.mkdir(parents=True)
            for entry in test_dir.iterdir():
                if entry.name == "rmake.rs":
                    continue
                if entry.is_dir():
                    shutil.copytree(entry, dest / entry.name)
                else:
                    shutil.copy2(entry, dest / entry.name)
        env = stage3_recipe_env(rust_dir, td, rec["suite"])
        compile_res = run_cmd(
            rustc_cmd() + ["--edition=2024",
                           "--extern", f"run_make_support={rlib}",
                           "-L", f"dependency={flat}",
                           "-Cprefer-dynamic", "-Dunused_must_use",
                           "-o", str(td / "oracle" / "rmake"),
                           str(rmake_rs)],
            timeout=STAGE3_NATIVE_TIMEOUT, cwd=td / "oracle",
            env_extra={"RUSTC_BOOTSTRAP": "-1"})
        if compile_res["timeout"] or compile_res["exit_code"] != 0:
            out["status"] = "fail"
            out["reason"] = ("native recipe compile failed: "
                             + first_error_line(compile_res["stderr"]))
            out["bucket"] = "native recipe failure"
            return
        native = run_cmd([str(td / "oracle" / "rmake")],
                         timeout=STAGE3_NATIVE_TIMEOUT, cwd=oracle_out,
                         env_extra=env)
        if native["timeout"] or native["exit_code"] != 0:
            out["status"] = "fail"
            out["reason"] = (
                f"native recipe exit {native['exit_code']}: "
                + first_error_line(native["stderr"] or native["stdout"]))
            out["bucket"] = "native recipe failure"
            return
        guest_env = dict(env)
        guest_env.update(BOOTSTRAP_ENV)
        got = run_cmd(
            [str(driver_bin), "run", str(rmake_rs),
             "--edition", "2024",
             "--aux-lib", f"run_make_support={rlib}",
             "--tool-program", paths["rustc"]],
            timeout=STAGE3_TIMEOUT, cwd=guest_out, env_extra=guest_env)
        if got["timeout"]:
            out["status"] = "fail"
            out["reason"] = f"driver timed out after {STAGE3_TIMEOUT}s"
            out["bucket"] = bucket_reason(out["reason"])
            return
        if got["exit_code"] == 0:
            out["status"] = "pass"
            return
        if got["exit_code"] == 2:
            out["status"] = "unsupported"
            out["reason"] = driver_reason(got["stderr"])
            out["bucket"] = bucket_reason(out["reason"])
            return
        out["status"] = "fail"
        out["reason"] = (f"recipe failed under sparagmos "
                          f"(exit {got['exit_code']}): "
                          + first_error_line(got["stderr"]))
        out["bucket"] = "recipe failed under sparagmos"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--survey-json", type=Path, required=True,
                    help="JSON produced by rustc-subset-survey.py --json")
    ap.add_argument("--rust-dir", type=Path,
                    default=Path.home() / "rust-1.100.0-survey",
                    help="checkout of rust-lang/rust at the tag")
    ap.add_argument("--driver-bin", type=Path, required=True,
                    help="sparagmos-driver binary")
    ap.add_argument("--out", type=Path, required=True,
                    help="write per-test results and aggregates as JSON")
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N stage-1 candidates "
                         "(smoke runs); 0 = all")
    ap.add_argument("--stages", choices=("1", "2", "3", "both", "all"),
                    default="both",
                    help="which stages to run (default: both; "
                         "'both' is stages 1+2, 'all' adds stage 3)")
    ap.add_argument("--stage2-limit", type=int, default=0,
                    help="only process the first N stage-2 candidates "
                         "(smoke runs); 0 = all")
    ap.add_argument("--stage3-limit", type=int, default=0,
                    help="only process the first N stage-3 candidates "
                         "(smoke runs); 0 = all")
    ap.add_argument("--jit", action="store_true",
                    help="metric 4: judge through the JIT path "
                         "instead of the interpreter/convert path "
                         "(exec records run via the driver's "
                         "run-jit mode; compile-check records are "
                         "judged by conversion plus JIT lowering "
                         "via the driver's jit-compile mode). The "
                         "denominator and the carried-through "
                         "records are unchanged, so records the JIT "
                         "harness cannot attempt (procedure, "
                         "exec-debugger, rustdoc, ...) count "
                         "against the rate, as in metric 2(b)")
    args = ap.parse_args()
    args.driver_bin = args.driver_bin.resolve()
    args.rust_dir = args.rust_dir.resolve()
    stage1_on = args.stages in ("1", "both", "all")
    stage2_on = args.stages in ("2", "both", "all")
    stage3_on = args.stages in ("3", "all")

    global _HOST_FACTS
    _HOST_FACTS = ToolchainFacts()

    survey = json.loads(args.survey_json.read_text())
    records = survey["tests"]
    denominator = survey["aggregates"]["denominator"]

    # Stage-1 candidates are processed for real; every other record in
    # the denominator is carried through as not-attempted so the
    # aggregate arithmetic covers the whole subset. A non-single-file
    # record is a candidate when its header carries auxiliary-crate
    # directives (scoped ones included: revision tests resolve them
    # per revision; scoped directives on a test that declares no
    # revisions stay blocked in stage 1).
    def has_aux_directives(r):
        try:
            text = (args.rust_dir / r["path"]).read_text(errors="ignore")
        except OSError:
            return False
        header = Header(text)
        if aux_directives(header):
            return True
        return any(name in ("aux-build", "aux-crate")
                   for (_rev, name) in header.scoped)

    def is_candidate(r):
        return (r["nature"] == "exec"
                and r["suite"] in ("ui", "ui-fulldeps")
                and not r["env_excluded"]
                and (r["single_file"] or has_aux_directives(r)))

    def is_candidate2(r):
        return (r["nature"] == "compile-check"
                and not r["env_excluded"])

    def is_candidate_debugger(r):
        return (r["nature"] == "exec-debugger"
                and r["suite"] == "debuginfo"
                and not r["env_excluded"])

    def is_candidate3(r):
        return (r["nature"] == "procedure"

                and not r["env_excluded"])

    all_candidates = ([r for r in records if is_candidate(r)]
                      if stage1_on else [])
    candidates = (all_candidates[:args.limit] if args.limit
                  else all_candidates)
    limited_out = all_candidates[len(candidates):]
    all_candidates2 = ([r for r in records if is_candidate2(r)]
                       if stage2_on else [])
    candidates2 = (all_candidates2[:args.stage2_limit]
                   if args.stage2_limit else all_candidates2)
    limited_out2 = all_candidates2[len(candidates2):]
    # The debugger harness runs with stage 2 (stage 1 is the
    # exec-only stage by definition); its records dispatch by
    # nature inside process_test.
    candidates_dbg = ([r for r in records if is_candidate_debugger(r)]
                      if stage2_on else [])
    all_candidates3 = ([r for r in records if is_candidate3(r)]
                       if stage3_on else [])
    candidates3 = (all_candidates3[:args.stage3_limit]
                   if args.stage3_limit else all_candidates3)
    limited_out3 = all_candidates3[len(candidates3):]
    others = [r for r in records
              if not is_candidate(r) and not r["env_excluded"]
              and not (stage2_on and is_candidate2(r))
              and not (stage2_on and is_candidate_debugger(r))
              and not (stage3_on and is_candidate3(r))]

    print(f"denominator: {denominator}; stage-1 candidates: "
          f"{len(candidates)}; stage-2 candidates: {len(candidates2)}; "
          f"debugger candidates: {len(candidates_dbg)}; "
          f"stage-3 candidates: {len(candidates3)}; "
          f"other records carried as not-attempted: {len(others)}",
          flush=True)

    results = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.jobs) as pool:
        futs = [pool.submit(process_test, r, args.rust_dir,
                            args.driver_bin, jit=args.jit)
                for r in candidates]
        futs += [pool.submit(process_test, r, args.rust_dir,
                             args.driver_bin, stage2=True, jit=args.jit)
                 for r in candidates2]
        futs += [pool.submit(process_test, r, args.rust_dir,
                             args.driver_bin, stage2=True, jit=args.jit)
                 for r in candidates_dbg]
        futs += [pool.submit(process_test, r, args.rust_dir,
                             args.driver_bin, stage3=True)
                 for r in candidates3]
        total_jobs = (len(candidates) + len(candidates2)
                      + len(candidates_dbg) + len(candidates3))

        done = 0
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{total_jobs} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    wall = time.time() - t0

    for r in others:
        results.append({
            "suite": r["suite"], "path": r["path"],
            "nature": r["nature"], "single_file": r["single_file"],
            "ffi": r["ffi"], "status": "not-attempted",
            "reason": "not in stage-1 scope (see bucket)",
            "bucket": ("out of stage-1 scope (nature)"
                       if r["nature"] != "exec"
                       else "out of stage-1 scope (multi-file)"),
            "mode": None, "seconds": 0.0,
        })

    for r in limited_out:
        results.append({
            "suite": r["suite"], "path": r["path"],
            "nature": r["nature"], "single_file": r["single_file"],
            "ffi": r["ffi"], "status": "not-attempted",
            "reason": "beyond --limit (smoke run)",
            "bucket": "not attempted (smoke-run limit)",
            "mode": None, "seconds": 0.0,
        })

    for r in limited_out2:
        results.append({
            "suite": r["suite"], "path": r["path"],
            "nature": r["nature"], "single_file": r["single_file"],
            "ffi": r["ffi"], "status": "not-attempted",
            "reason": "beyond --stage2-limit (smoke run)",
            "bucket": "not attempted (smoke-run limit)",
            "mode": None, "stage": 2, "seconds": 0.0,
        })

    for r in limited_out3:
        results.append({
            "suite": r["suite"], "path": r["path"],
            "nature": r["nature"], "single_file": r["single_file"],
            "ffi": r["ffi"], "status": "not-attempted",
            "reason": "beyond --stage3-limit (smoke run)",
            "bucket": "not attempted (smoke-run limit)",
            "mode": None, "stage": 3, "seconds": 0.0,
        })

    results.sort(key=lambda r: r["path"])
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    unsup = sum(1 for r in results if r["status"] == "unsupported")
    notatt = sum(1 for r in results if r["status"] == "not-attempted")

    buckets = {}
    for r in results:
        if r["status"] != "pass":
            buckets[r["bucket"]] = buckets.get(r["bucket"], 0) + 1
    reasons = {}
    for r in results:
        if r["status"] in ("unsupported", "not-attempted", "fail") \
                and r["reason"]:
            key = normalize_reason(r["reason"])
            reasons[key] = reasons.get(key, 0) + 1
    top_reasons = sorted(reasons.items(), key=lambda kv: -kv[1])[:30]

    cand_paths = {r["path"] for r in candidates}
    cand = [r for r in results if r["path"] in cand_paths]
    cand2_paths = {r["path"] for r in candidates2}
    cand2 = [r for r in results if r["path"] in cand2_paths]
    cand3_paths = {r["path"] for r in candidates3}
    cand3 = [r for r in results if r["path"] in cand3_paths]

    snap_keys = ("stderr_snapshot", "error_annotations", "fixed_file",
                 "mir_snapshot", "filecheck", "run_rustfix")

    def snapshot_counts(rs):
        counts = {k: sum(1 for r in rs if r.get("snapshot", {}).get(k))
                  for k in snap_keys}
        counts["any"] = sum(
            1 for r in rs
            if any(r.get("snapshot", {}).get(k) for k in snap_keys))
        counts["records"] = len(rs)
        return counts

    cc_results = [r for r in results if r["nature"] == "compile-check"]
    snapshot_verification = {
        "compile_check_all": snapshot_counts(cc_results),
        "stage2_passed": snapshot_counts(
            [r for r in cc_results if r["status"] == "pass"]),
    }

    def rate_block(rs):
        tot = len(rs)
        ps = sum(1 for r in rs if r["status"] == "pass")
        return {"total": tot, "passed": ps,
                "pass_rate_percent":
                    round(100.0 * ps / tot, 3) if tot else 0.0}

    by_nature = {}
    for nature in sorted({r["nature"] for r in results}):
        by_nature[nature] = rate_block(
            [r for r in results if r["nature"] == nature])
    by_suite = {}
    for suite in sorted({r["suite"] for r in results}):
        by_suite[suite] = rate_block(
            [r for r in results if r["suite"] == suite])
    agg = {
        "denominator": denominator,
        "results_total": len(results),
        "passed": passed,
        "failed": failed,
        "unsupported": unsup,
        "not_attempted": notatt,
        "jit_mode": bool(args.jit),
        "metric_2b_pass_rate_percent":
            round(100.0 * passed / denominator, 3) if denominator else 0.0,
        "stage1_candidates": len(candidates),
        "stage1_passed": sum(1 for r in cand if r["status"] == "pass"),
        "stage2_candidates": len(candidates2),
        "stage2_passed": sum(1 for r in cand2 if r["status"] == "pass"),
        "stage3_candidates": len(candidates3),
        "stage3_passed": sum(1 for r in cand3 if r["status"] == "pass"),
        "compile_check_category": by_nature.get(
            "compile-check",
            {"total": 0, "passed": 0, "pass_rate_percent": 0.0}),
        "snapshot_verification": snapshot_verification,
        "out_of_environment": sum(
            1 for r in results if r["bucket"] == "out of environment"),
        "exec_category": by_nature.get("exec",
                                       {"total": 0, "passed": 0,
                                        "pass_rate_percent": 0.0}),
        "by_nature": by_nature,
        "by_suite": by_suite,
        "wall_seconds": round(wall, 1),
        "jobs": args.jobs,
        "buckets": dict(sorted(buckets.items(), key=lambda kv: -kv[1])),
        "top_reasons": top_reasons,
    }
    if args.jit:
        # Metric 4 is metric 2(b)'s denominator judged on the JIT
        # path; the rate is the same arithmetic over the same
        # denominator, reported under its own key.
        agg["metric_4_jit_pass_rate_percent"] = \
            agg["metric_2b_pass_rate_percent"]
    payload = {"aggregates": agg, "results": results}
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True))

    print(json.dumps(agg, indent=1, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
