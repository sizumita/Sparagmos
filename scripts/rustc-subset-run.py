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
with `--aux` for the Sparagmos side, in dependency order. Stage 2 covers the nature="compile-check" population:
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

- its header carries an unscoped run-pass / run-fail directive
  (revisions, run-crash, run-flags, exec-env, stdin and
  add-core-stubs are recorded as not-attempted reasons in stage 1;
  a test's `compile-flags` are honoured in both stages by forwarding
  them to the native rustc and, via the driver's `--rustc-arg`, to
  the driver's embedded rustc — the position compiletest gives them;
  stage 2 additionally judges revision tests per revision, with the
  `--cfg <revision>` and `--check-cfg` arguments compiletest adds);
- no `ignore-*` condition matches this host and every `only-*`
  condition is satisfied (otherwise the test is out of environment
  for x86_64-unknown-linux-gnu and is recorded as such);
- `needs-*` conditions are either known to hold here or recorded as
  unevaluatable (not attempted);
- the native compile succeeds and the native run's exit code matches
  the mode's expectation (otherwise the native oracle itself does not
  behave as the test expects, recorded with the compiler output).

Both sides are compiled/run with RUSTC_BOOTSTRAP=1 so that tests
using unstable features behave the same way they do under
compiletest (which runs against a development compiler). The native
compile uses `--edition` from the test's `//@ edition:` directive,
defaulting to 2021 (compiletest's default for the ui suite), and
opt-level 0, matching the harness's corpus oracle.

Usage:
    python3 scripts/rustc-subset-run.py \
        --survey-json subset-survey.json \
        --rust-dir ~/rust-1.100.0-survey \
        --driver-bin crates/sparagmos-driver/target/debug/sparagmos-driver \
        --out subset-run-results.json [--jobs 48] [--limit N]
        [--stages 1|2|both] [--stage2-limit N]
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TOOLCHAIN = "1.100.0"
DENOMINATOR = 24937  # metric 2(b) denominator at tag 1.100.0 (COVERAGE.md)

NATIVE_COMPILE_TIMEOUT = 60
NATIVE_RUN_TIMEOUT = 30
DRIVER_TIMEOUT = 60

# ---------------------------------------------------------------------------
# Header directive parsing
# ---------------------------------------------------------------------------

DIRECTIVE_RE = re.compile(
    r"^\s*//@(?:\[([^\]]+)\])?\s*([a-zA-Z][a-zA-Z0-9-]*)"
    r"(?:\s*:\s*(.*?)|\s*\((.*)\))?\s*$")

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


# ---------------------------------------------------------------------------
# compile-flags handling
# ---------------------------------------------------------------------------

HOST_TARGET = "x86_64-unknown-linux-gnu"


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


def compile_flags(view):
    """All compile-flags tokens visible in a directive view."""
    flags = []
    for v in view.values("compile-flags"):
        flags.extend(split_flags(v))
    return flags


def screen_flags(flags):
    """Why this flag set cannot be honoured here, or None.

    The driver forwards flags verbatim to its embedded rustc (and
    stage 1 to the native rustc), so almost everything compiletest
    accepts works unchanged. The exceptions are flags that would
    replace the compilation itself or contradict the driver's fixed
    invocation (its pinned sysroot and scratch output directory, the
    one installed target).
    """
    i = 0
    while i < len(flags):
        t = flags[i]
        nxt = flags[i + 1] if i + 1 < len(flags) else None
        if t == "--target" or t.startswith("--target="):
            tgt = t[len("--target="):] if t.startswith("--target=") else nxt
            if tgt is not None and tgt != HOST_TARGET:
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
}

# `needs-*` conditions known to hold on the measurement host.
NEEDS_SATISFIED = {
    "needs-run-enabled",   # we do run programs
    "needs-unwind",        # the reference toolchain is built with unwinding
}

# `ignore-*` suffixes that never match any host in compiletest's model
# (they gate on the pass mode rather than the environment); for the
# run modes measured here they do not apply.
IGNORE_NEVER_MATCH = {"ignore-pass", "ignore-check", "ignore-pretty"}


def eval_cfg_atom(atom):
    """Evaluate one cfg atom against the host. None = cannot evaluate."""
    atom = atom.strip()
    if atom in HOST_TRUE_CFGS:
        return True
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


def applicability(header):
    """Decide whether a test applies to this host.

    Returns (status, reason): status is "applicable",
    "out-of-environment" (an ignore matched or an only did not), or
    "unevaluatable" (a needs-/condition could not be decided).
    """
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
        if any(v is None for v in vals):
            return ("unevaluatable",
                    "only-* condition cannot be evaluated: " + ", ".join(onlys))
        if not any(vals):
            return ("out-of-environment",
                    "no only-* condition matches this host: " + ", ".join(onlys))
    for name in sorted(header.names):
        if name.startswith("needs-"):
            if name in NEEDS_SATISFIED:
                continue
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
    if header.has("revisions") or any(
            rev is not None for (rev, _n) in header.scoped):
        return "revisions present (per-revision cfgs not run in stage 1)"
    mode = run_mode(header)
    if mode is None:
        return "no unscoped run-pass/run-fail directive"
    if mode == "run-crash":
        return "run-crash mode (crash reproduction is out of stage-1 scope)"
    if header.has("run-flags"):
        return "run-flags present (guest program arguments are not supported)"
    if header.has("exec-env"):
        return "exec-env present (guest environment is not modelled)"
    if header.has("stdin"):
        return "stdin directive present (guest stdin is not modelled)"
    if header.has("add-core-stubs"):
        return "add-core-stubs (test links the minicore stub crate)"
    if header.has("ignore-test"):
        return "ignore-test (compiletest itself never runs this test)"
    return None


# ---------------------------------------------------------------------------
# Process execution
# ---------------------------------------------------------------------------

def run_cmd(argv, *, timeout, cwd=None, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
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


def declared_crate_name(text):
    """The `#![crate_name = "..."]` an auxiliary source declares.

    When a source declares one, the declaration is the crate's name
    (compiletest links the crate under it); a differing --crate-name
    flag is a rustc error, so none is passed.
    """
    m = CRATE_NAME_RE.search(text)
    return m.group(1) if m else None


def aux_directives(header):
    """(crate_name, file_name) pairs from a header's aux directives.

    `aux-build: foo-bar.rs` builds crate `foo_bar` (the file stem
    with dashes as underscores, as compiletest derives it);
    `aux-crate: name=file.rs` names the crate explicitly.
    """
    out = []
    for value in header.values("aux-build"):
        fname = value.strip()
        if fname:
            out.append((Path(fname).stem.replace("-", "_"), fname))
    for value in header.values("aux-crate"):
        value = value.strip()
        if "=" in value:
            name, fname = value.split("=", 1)
            out.append((name.strip(), fname.strip()))
    return out


def resolve_aux_specs(test_path, top=None):
    """Resolve a test's auxiliary crates, transitively, deps first.

    Returns (specs, error): specs is a list of (crate_name, Path) in
    dependency order (a crate before the crates that use it), the
    order both process sides expect. Auxiliary files are looked up
    next to the file whose directive names them, in that directory's
    `auxiliary/` subdirectory, and next to the main test file (the
    places compiletest looks). error is a not-attempted reason when
    resolution fails.

    The test's own directives are read from `top` when given (a
    Header, or a HeaderView when a revision's scoped aux directives
    apply); the transitive directives of auxiliary files always come
    from the files themselves.
    """
    test_dir = test_path.parent
    specs = []
    by_name = {}
    visiting = set()

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

    def visit(name, fname, from_dir):
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
        header = Header(path.read_text(errors="ignore"))
        for dep_name, dep_file in aux_directives(header):
            err = visit(dep_name, dep_file, path.parent)
            if err:
                return err
        visiting.discard(name)
        specs.append((name, path))
        return None

    if top is None:
        top = Header(test_path.read_text(errors="ignore"))
    for name, fname in aux_directives(top):
        err = visit(name, fname, test_dir)
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


def native_side(path, edition, workdir, extra_flags=(), aux_specs=()):
    """Compile and run natively. Returns (result_dict, error_reason).

    error_reason is None when the native run completed (whatever its
    exit code); otherwise a not-attempted reason string.
    `extra_flags` are the test's compile-flags, appended last as
    compiletest appends them. Auxiliary crates are compiled first,
    in dependency order, the way compiletest builds them for the
    test's own compilation.
    """
    binpath = Path(workdir) / "prog"
    lib = sysroot_lib()
    externs = []
    if aux_specs:
        auxdir = Path(workdir) / "auxlibs"
        auxdir.mkdir()
        built = []
        for name, src in aux_specs:
            text = src.read_text(errors="ignore")
            declared = declared_crate_type(text)
            eff_name = declared_crate_name(text) or name
            cmd = rustc_cmd() + ["--edition", edition,
                                 "--out-dir", str(auxdir)]
            if lib:
                cmd += ["-L", lib]
            cmd += ["-L", f"dependency={auxdir}"]
            if declared is None:
                cmd += ["--crate-type", "rlib"]
            if declared_crate_name(text) is None:
                cmd += ["--crate-name", name]
            for prev_name, prev_artifact in built:
                if prev_artifact is not None:
                    cmd += ["--extern", f"{prev_name}={prev_artifact}"]
            cmd += [str(src)]
            build = run_cmd(cmd, timeout=NATIVE_COMPILE_TIMEOUT,
                            env_extra=BOOTSTRAP_ENV)
            if build["timeout"]:
                return None, "native compile timed out"
            if build["exit_code"] != 0:
                first = first_error_line(build["stderr"])
                return None, f"native compile failed: {first}"
            effective = declared or "rlib"
            artifact = (find_artifact(auxdir, eff_name)
                        if effective in ("rlib", "lib", "dylib",
                                         "proc-macro")
                        else None)
            built.append((eff_name, artifact))
            if artifact is not None:
                externs += ["--extern", f"{eff_name}={artifact}"]
    build = run_cmd(
        rustc_cmd() + ["--edition", edition, "-C", "opt-level=0"]
        + (["-L", lib] if lib else [])
        + (["-L", f"dependency={Path(workdir) / 'auxlibs'}"]
           if aux_specs else [])
        + externs
        + ["-o", str(binpath), str(path)] + list(extra_flags),
        timeout=NATIVE_COMPILE_TIMEOUT, env_extra=BOOTSTRAP_ENV)
    if build["timeout"]:
        return None, "native compile timed out"
    if build["exit_code"] != 0:
        first = first_error_line(build["stderr"])
        return None, f"native compile failed: {first}"
    if extra_flags and not binpath.exists():
        return None, ("native compile produced no binary (the "
                      "compile-flags change what is emitted)")
    run_env = {}
    if lib:
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        run_env["LD_LIBRARY_PATH"] = lib + (":" + prev if prev else "")
    run = run_cmd([str(binpath)], timeout=NATIVE_RUN_TIMEOUT,
                  cwd=str(workdir), env_extra=run_env)
    if run["timeout"]:
        return None, "native run timed out"
    return run, None


def sparagmos_side(path, edition, driver_bin, cwd=None, rustc_args=(), aux_specs=()):
    """Run through the driver. Returns (result_dict, kind, reason).

    kind is "result" (exit 0/101), "unsupported" (driver exit 2),
    "timeout", or "driver-error". `rustc_args` are the test's
    compile-flags, forwarded through the driver's `--rustc-arg`.
    Auxiliary crates are handed to the driver with `--aux`, in
    dependency order.
    """
    argv = [str(driver_bin), "run", str(path), "--edition", edition]
    for a in rustc_args:
        argv += ["--rustc-arg", a]
    for name, src in aux_specs:
        argv += ["--aux", f"{name}={src}"]
    res = run_cmd(argv,
                  timeout=DRIVER_TIMEOUT, cwd=cwd,
                  env_extra=BOOTSTRAP_ENV)
    if res["timeout"]:
        return None, "timeout", f"driver timed out after {DRIVER_TIMEOUT}s"
    if res["exit_code"] in (0, 101):
        return res, "result", ""
    if res["exit_code"] == 2:
        return None, "unsupported", driver_reason(res["stderr"])
    return None, "driver-error", \
        f"driver exit {res['exit_code']}: {res['stderr'].strip()[:300]}"


def first_error_line(stderr):
    for line in stderr.splitlines():
        line = line.strip()
        if line.startswith("error"):
            return line[:200]
    return (stderr.strip().splitlines() or ["<no output>"])[0][:200]


def driver_reason(stderr):
    lines = [l.strip() for l in stderr.splitlines()
             if "sparagmos-driver run:" in l]
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
    first = reason.split("; sparagmos-driver run:")[0].lower()
    r = reason.lower()
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

def process_test(rec, rust_dir, driver_bin, stage2=False):
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
    t0 = time.time()
    try:
        if stage2 and rec["nature"] == "compile-check":
            fill_result_stage2(rec, rust_dir, driver_bin, out)
        else:
            fill_result(rec, rust_dir, driver_bin, out)
    except Exception as e:  # a runner bug must not lose the whole run
        out["status"] = "not-attempted"
        out["reason"] = f"runner internal error: {e!r}"
        out["bucket"] = "runner internal error"
    out["seconds"] = round(time.time() - t0, 2)
    return out


def fill_result(rec, rust_dir, driver_bin, out):
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
        out["bucket"] = "not attempted (header feature)"
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

    editions = header.values("edition")
    edition = editions[0].strip() if editions else "2021"

    flags = compile_flags(header.view(None))
    screened = screen_flags(flags)
    if screened:
        out["reason"] = screened
        out["bucket"] = "not attempted (compile-flags)"
        return

    with tempfile.TemporaryDirectory(prefix="subset-run-") as td:
        native, err = native_side(path, edition, td, flags, aux_specs)
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
        got, kind, reason = sparagmos_side(path, edition, driver_bin, td,
                                           rustc_args=flags, aux_specs=aux_specs)
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
# rustc (screened only for flags that would replace the compilation
# itself: a non-host --target, --print, --sysroot, an output
# location, -Zunpretty), and revision tests are judged once per
# revision — with the revision's directive view, its compile-flags,
# and the `--cfg <revision>` / `--check-cfg` arguments compiletest
# adds. The record passes only when every revision that runs
# passes; a revision removed by an ignore-/only- condition does not
# run, as in compiletest.
#
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


def compile_expectation(suite, header):
    """The compile outcome compiletest expects of a compile-check test.

    Returns "success" (compilation must succeed), "failure" (it must
    be rejected), "crash" (the compiler must crash, crashes suite), or
    None when the verdict cannot be reduced to a compile outcome
    (run-fail-or-crash involves executing the program).
    """
    if suite == "crashes":
        return "crash"
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
            return None  # run modes are stage-1 territory
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
    if view.has("add-minicore") or view.has("add-core-stubs"):
        return "add-minicore (test links the minicore stub crate)"
    if view.has("ignore-test"):
        return "ignore-test (compiletest itself never runs this test)"
    if view.has("rustc-env"):
        return "rustc-env present (compiler environment is not modelled)"
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


def convert_side(path, edition, driver_bin, rustc_args=(), aux_specs=()):
    """Run the driver in convert-only mode and parse its report.

    Returns a dict with exit_code (None on timeout), converted and
    skipped counts (None when no report was printed), skip_totals
    ([(kind, count)] from the report's `unsupported:` section, most
    frequent first), compilation_failed (the driver reported a clean
    compilation failure), ice (stderr shows an internal compiler
    error), and stderr. `rustc_args` are forwarded to the driver's
    embedded rustc through its `--rustc-arg` option (a test's
    compile-flags and the per-revision arguments). Auxiliary crates
    are handed to the driver with `--aux`, in dependency order, as
    in run mode.
    """
    argv = [str(driver_bin), str(path), "--edition", edition]
    for a in rustc_args:
        argv += ["--rustc-arg", a]
    for name, src in aux_specs:
        argv += ["--aux", f"{name}={src}"]
    res = run_cmd(argv,
                  timeout=CONVERT_TIMEOUT, env_extra=BOOTSTRAP_ENV)
    out = {
        "exit_code": res["exit_code"],
        "timeout": res["timeout"],
        "converted": None,
        "skipped": None,
        "skip_totals": [],
        "compilation_failed": "sparagmos-driver: compilation failed"
                              in res["stderr"],
        "ice": "internal compiler error" in res["stderr"],
        "stderr": res["stderr"],
        "stdout": res["stdout"],
    }
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


def stage2_attempt(rec, view, path, driver_bin, revision, all_revisions,
                   aux_specs=(), aux_per_revision=False):
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
        res["bucket"] = "not attempted (header feature)"
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
    if expectation is None:
        res["reason"] = ("verdict involves executing the program "
                         "(run-fail-or-crash / run mode), which a "
                         "compile outcome alone cannot judge")
        res["bucket"] = "not attempted (verdict needs execution)"
        return res
    editions = view.values("edition")
    edition = editions[-1].strip() if editions else "2021"
    flags = compile_flags(view)
    screened = screen_flags(flags)
    if screened:
        res["reason"] = screened
        res["bucket"] = "not attempted (compile-flags)"
        return res
    args = []
    if revision is not None:
        args += revision_args(view, revision, all_revisions)
    args += flags
    specs = aux_specs
    if aux_per_revision:
        # The test's aux directives are scoped to revisions: this
        # view builds the auxiliary crates its own directives name
        # (a view naming none converts without auxiliaries).
        if aux_directives(view):
            specs, aux_err = resolve_aux_specs(path, view)
            if aux_err:
                res["reason"] = aux_err
                res["bucket"] = "out of stage-2 scope (multi-file)"
                return res
        else:
            specs = []
    conv = convert_side(path, edition, driver_bin, rustc_args=args,
                        aux_specs=specs)
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
    if verdict == "fail":
        res["driver"]["stderr_first"] = first_error_line(conv["stderr"])
    return res


def fill_result_stage2(rec, rust_dir, driver_bin, out):
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
        out["reason"] = ("incremental suite: the verdict depends on "
                         "recompilation across sessions, which a single "
                         "conversion cannot judge")
        out["bucket"] = "out of stage-2 scope (suite: incremental)"
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
        aux_specs, aux_err = resolve_aux_specs(path)
        if aux_err:
            out["reason"] = aux_err
            out["bucket"] = "out of stage-2 scope (multi-file)"
            return
        if not aux_specs:
            if any(name in ("aux-build", "aux-crate")
                   for (_rev, name) in header.scoped):
                aux_per_revision = True
            else:
                out["reason"] = ("multi-file / auxiliary crates "
                                 "(not in stage-2 scope)")
                out["bucket"] = "out of stage-2 scope (multi-file)"
                return
    revs = header.revisions()
    if not revs:
        res = stage2_attempt(rec, header.view(None), path, driver_bin,
                             None, [], aux_specs, aux_per_revision)
        for key in ("status", "reason", "bucket", "expectation",
                    "driver"):
            if key in res:
                out[key] = res[key]
        return
    per_rev = {}
    for rev in revs:
        per_rev[rev] = stage2_attempt(rec, header.view(rev), path,
                                      driver_bin, rev, revs,
                                      aux_specs, aux_per_revision)
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
                return
    out["status"] = "pass"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    ap.add_argument("--stages", choices=("1", "2", "both"),
                    default="both",
                    help="which stages to run (default: both)")
    ap.add_argument("--stage2-limit", type=int, default=0,
                    help="only process the first N stage-2 candidates "
                         "(smoke runs); 0 = all")
    args = ap.parse_args()
    args.driver_bin = args.driver_bin.resolve()
    args.rust_dir = args.rust_dir.resolve()
    stage1_on = args.stages in ("1", "both")
    stage2_on = args.stages in ("2", "both")

    survey = json.loads(args.survey_json.read_text())
    records = survey["tests"]
    denominator = survey["aggregates"]["denominator"]

    # Stage-1 candidates are processed for real; every other record in
    # the denominator is carried through as not-attempted so the
    # aggregate arithmetic covers the whole subset. A non-single-file
    # record is a candidate when its header carries auxiliary-crate
    # directives (scoped ones included: process_test records those
    # under the revisions blocker).
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
    others = [r for r in records
              if not is_candidate(r) and not r["env_excluded"]
              and not (stage2_on and is_candidate2(r))]

    print(f"denominator: {denominator}; stage-1 candidates: "
          f"{len(candidates)}; stage-2 candidates: {len(candidates2)}; "
          f"other records carried as not-attempted: {len(others)}",
          flush=True)

    results = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.jobs) as pool:
        futs = [pool.submit(process_test, r, args.rust_dir,
                            args.driver_bin) for r in candidates]
        futs += [pool.submit(process_test, r, args.rust_dir,
                             args.driver_bin, True)
                 for r in candidates2]
        total_jobs = len(candidates) + len(candidates2)
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
        "metric_2b_pass_rate_percent":
            round(100.0 * passed / denominator, 3) if denominator else 0.0,
        "stage1_candidates": len(candidates),
        "stage1_passed": sum(1 for r in cand if r["status"] == "pass"),
        "stage2_candidates": len(candidates2),
        "stage2_passed": sum(1 for r in cand2 if r["status"] == "pass"),
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
    payload = {"aggregates": agg, "results": results}
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True))

    print(json.dumps(agg, indent=1, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
