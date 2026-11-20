#!/usr/bin/env python3
"""Survey the rustc test-suite subset for Sparagmos metric 2(b).

Counts every test in the compiletest suites of rust-lang/rust at a given
tag, following the same discovery rules compiletest itself uses, and
classifies each test mechanically:

- nature: how pass/fail is decided
    - "exec":           the compiled program is executed and its behaviour
                        is part of the verdict (UI run-pass/run-fail/run-crash,
                        coverage modes)
    - "exec-debugger":  the compiled program is executed under a debugger
                        and debugger output is compared (debuginfo suite)
    - "procedure":      a recipe program (rmake.rs) is executed and drives
                        build/run steps (run-make family)
    - "compile-check":  the verdict comes from compilation output
                        (diagnostics, IR/asm/MIR dumps, pretty-printing)
    - "rustdoc-check":  the verdict comes from rustdoc output
                        (HTML/JSON/search index/GUI, rustdoc UI diagnostics)
- single_file: False when the test needs auxiliary crates/files
  (`//@ aux-build:` / `//@ aux-crate:`), or, for run-make tests, when the
  test directory holds files besides rmake.rs.
- build_procedure: True for the run-make family (recipe-driven builds,
  including Cargo projects in run-make-cargo and build-std).
- ffi: True when the test sources declare or use `extern "C"`/`"system"`
  functions, `#[link]` attributes, bare `extern {}` blocks, or ship C/C++
  sources to link against.
- env_excluded: environment-dependent tests that the metric definition
  excludes individually (network access or dedicated emulation hardware).
  Detected only in run-make-cargo: tests that need QEMU system emulation,
  and tests whose Cargo projects declare non-path (registry/git)
  dependencies that must be downloaded. Everything else stays in the
  denominator, including platform-gated tests and tests gated on compiler
  build configuration (`needs-*`).

Discovery rules mirror src/tools/compiletest/src/lib.rs at the tag:
- File suites: every `*.rs` file under the suite directory is one test,
  except files under a directory named `auxiliary`, files inside a
  directory containing a `compiletest-ignore-dir` marker, and files whose
  name starts with `.`, `#` or `~`.
- RunMake suites: every directory containing `rmake.rs` is one test;
  discovery does not descend into such a directory.
- rustdoc-gui is not a compiletest suite; its runner treats every
  `*.goml` file as one test. rustdoc-js-std likewise treats every
  `*.js` file as one test. rustdoc-js is a compiletest suite whose unit
  is the `*.rs` file (each paired with a same-named `.js` assertion file).

The suite list and modes mirror src/bootstrap/src/core/build_steps/test.rs
at the tag. The coverage suite runs in two modes (coverage-map and
coverage-run) over the same files; files are counted once.

Usage:
    python3 scripts/rustc-subset-survey.py [--rust-dir DIR] [--json FILE]
                                           [--fetch] [--fetch-dir DIR]

--fetch clones rust-lang/rust at the tag with a sparse checkout of
tests/ (plus compiletest/bootstrap sources used as rule references):

    git clone --depth 1 --branch 1.100.0 --filter=blob:none --sparse \
        https://github.com/rust-lang/rust.git DIR
    git -C DIR sparse-checkout set tests src/tools/compiletest src/bootstrap

Output is deterministic: re-running against the same checkout prints the
same numbers.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TAG = "1.100.0"
DEFAULT_FETCH_DIR = Path.home() / "rust-1.100.0-survey"

# suite name -> (relative dir, kind, mode note)
# kind: "rs" (one .rs file = one test), "rmake" (one rmake.rs dir = one
# test), "goml" / "js" (one script file = one test).
SUITES = [
    ("ui",                  "tests/ui",                  "rs"),
    ("ui-fulldeps",         "tests/ui-fulldeps",         "rs"),
    ("rustdoc-ui",          "tests/rustdoc-ui",          "rs"),
    ("crashes",             "tests/crashes",             "rs"),
    ("codegen-llvm",        "tests/codegen-llvm",        "rs"),
    ("codegen-units",       "tests/codegen-units",       "rs"),
    ("incremental",         "tests/incremental",         "rs"),
    ("debuginfo",           "tests/debuginfo",           "rs"),
    ("rustdoc-html",        "tests/rustdoc-html",        "rs"),
    ("rustdoc-json",        "tests/rustdoc-json",        "rs"),
    ("rustdoc-js",          "tests/rustdoc-js",          "rs"),
    ("pretty",              "tests/pretty",              "rs"),
    ("assembly-llvm",       "tests/assembly-llvm",       "rs"),
    ("assembly-gcc",        "tests/assembly-gcc",        "rs"),
    ("mir-opt",             "tests/mir-opt",             "rs"),
    ("coverage",            "tests/coverage",            "rs"),
    ("coverage-run-rustdoc", "tests/coverage-run-rustdoc", "rs"),
    ("run-make",            "tests/run-make",            "rmake"),
    ("run-make-cargo",      "tests/run-make-cargo",      "rmake"),
    ("build-std",           "tests/build-std",           "rmake"),
    ("rustdoc-gui",         "tests/rustdoc-gui",         "goml"),
    ("rustdoc-js-std",      "tests/rustdoc-js-std",      "js"),
]

RUSTDOC_SUITES = {"rustdoc-ui", "rustdoc-html", "rustdoc-json",
                  "rustdoc-js", "rustdoc-js-std", "rustdoc-gui"}
EXEC_SUITES = {"coverage", "coverage-run-rustdoc"}
RUN_DIRECTIVES = {"run-pass", "run-fail", "run-crash"}

DIRECTIVE_RE = re.compile(r"//@(?:\[[^\]]*\])?\s*([a-zA-Z][a-zA-Z0-9-]*)")
REVISIONS_RE = re.compile(r"//@(?:\[[^\]]*\])?\s*revisions:\s*([^\n]+)")
AUX_BUILD_RE = re.compile(r"//@(?:\[[^\]]*\])?\s*aux-build:\s*(\S+)")
AUX_CRATE_RE = re.compile(r"//@(?:\[[^\]]*\])?\s*aux-crate:\s*(\S+)")
FFI_RES = [
    re.compile(r"extern\s+\"(C|system)\""),
    re.compile(r"#\[\s*link\s*\("),
    re.compile(r"\bextern\s*\{"),
]
C_EXTS = {".c", ".cc", ".cpp", ".h", ".hpp"}


def read_text(path):
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return None


def directives_of(text):
    return set(m.group(1) for m in DIRECTIVE_RE.finditer(text))


def revision_count(text):
    m = REVISIONS_RE.search(text)
    if not m:
        return 1
    names = [n for n in re.split(r"[\s,]+", m.group(1).strip()) if n]
    return max(1, len(names))


def has_ffi_in_text(text):
    return any(r.search(text) for r in FFI_RES)


def discover_rs_tests(suite_dir):
    """Yield test .rs files using compiletest's discovery rules."""
    for root, dirs, files in os.walk(suite_dir):
        root = Path(root)
        if "compiletest-ignore-dir" in files:
            dirs[:] = []
            continue
        if "auxiliary" in dirs:
            dirs.remove("auxiliary")
        for name in sorted(files):
            if name.endswith(".rs") and not name.startswith((".", "#", "~")):
                yield root / name


def discover_rmake_tests(suite_dir):
    """Yield directories containing rmake.rs (one test each)."""
    for root, dirs, files in os.walk(suite_dir):
        if "rmake.rs" in files:
            dirs[:] = []
            yield Path(root)


def discover_script_tests(suite_dir, ext):
    for root, dirs, files in os.walk(suite_dir):
        if "auxiliary" in dirs:
            dirs.remove("auxiliary")
        for name in sorted(files):
            if name.endswith(ext):
                yield Path(root) / name


def aux_index(suite_dir):
    """Map basename -> [paths] for files under any auxiliary/ dir."""
    index = {}
    for root, dirs, files in os.walk(suite_dir):
        if Path(root).name == "auxiliary":
            for name in files:
                index.setdefault(name, []).append(Path(root) / name)
    return index


def ffi_for_rs_test(path, text, aux_idx):
    if has_ffi_in_text(text):
        return True
    for m in AUX_BUILD_RE.finditer(text):
        for aux_path in aux_idx.get(m.group(1), []):
            aux_text = read_text(aux_path)
            if aux_text and has_ffi_in_text(aux_text):
                return True
    # C sources shipped next to the test file also mean FFI.
    for sibling in path.parent.iterdir():
        if sibling.suffix in C_EXTS:
            return True
    return False


def ffi_for_rmake_test(test_dir):
    for root, _dirs, files in os.walk(test_dir):
        for name in files:
            p = Path(root) / name
            if p.suffix in C_EXTS:
                return True
            if p.suffix in (".rs", ".toml"):
                t = read_text(p)
                if t and has_ffi_in_text(t):
                    return True
    return False


def cargo_registry_deps(test_dir):
    """Non-path dependencies declared by Cargo projects in the test dir.

    Returns a list of dependency names that Cargo would have to download
    (plain version requirements or git sources). Path-only and workspace
    inherited entries are local and do not count.
    """
    found = []
    section_re = re.compile(
        r"\[(?:[\w.-]+\.)?(?:dev-|build-)?dependencies\][^\[]*", re.S)
    for toml in Path(test_dir).rglob("Cargo.toml"):
        text = read_text(toml)
        if not text:
            continue
        for sec in section_re.finditer(text):
            for line in sec.group(0).splitlines()[1:]:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r"([\w-]+)\s*=\s*(.+)", line)
                if not m:
                    continue
                name, value = m.group(1), m.group(2).strip()
                if value.startswith('"'):
                    found.append(name)
                elif value.startswith("{") and "git" in value:
                    found.append(name)
    return found


def classify(suite, kind, path, text, aux_idx):
    """Return a record dict for one test."""
    dirs = directives_of(text) if text else set()
    rec = {
        "suite": suite,
        "path": str(path),
        "revisions": revision_count(text) if text else 1,
        "env_excluded": False,
        "env_reasons": [],
    }
    if kind == "rmake":
        rec["nature"] = "procedure"
        rec["build_procedure"] = True
        all_files = [p for p in path.rglob("*") if p.is_file()]
        rec["single_file"] = len(all_files) == 1
        rec["ffi"] = ffi_for_rmake_test(path)
        if suite == "run-make-cargo":
            reasons = []
            qemu = "qemu" in path.name or (
                text is not None and "qemu" in text.lower())
            if qemu:
                reasons.append(
                    "QEMU system emulation and firmware are required")
            deps = cargo_registry_deps(path)
            if deps:
                reasons.append(
                    "registry dependencies must be downloaded: "
                    + ", ".join(sorted(set(deps))))
            if text is not None and "fortanix-unknown-sgx" in text:
                reasons.append(
                    "requires the Intel SGX "
                    "(x86_64-fortanix-unknown-sgx) target platform")
            if reasons:
                rec["env_excluded"] = True
                rec["env_reasons"] = reasons
    elif kind in ("goml", "js"):
        rec["nature"] = "rustdoc-check"
        rec["build_procedure"] = False
        rec["single_file"] = True
        rec["ffi"] = has_ffi_in_text(text) if text else False
    else:  # kind == "rs"
        rec["build_procedure"] = False
        rec["single_file"] = not (
            "aux-build" in dirs or "aux-crate" in dirs
            or AUX_BUILD_RE.search(text or "")
            or AUX_CRATE_RE.search(text or ""))
        rec["ffi"] = ffi_for_rs_test(path, text or "", aux_idx)
        if suite in RUSTDOC_SUITES:
            rec["nature"] = "rustdoc-check"
        elif suite == "debuginfo":
            rec["nature"] = "exec-debugger"
        elif suite in EXEC_SUITES:
            rec["nature"] = "exec"
        elif suite in ("ui", "ui-fulldeps"):
            rec["nature"] = "exec" if dirs & RUN_DIRECTIVES else "compile-check"
        else:
            rec["nature"] = "compile-check"
    return rec


def survey(rust_dir):
    tests_root = rust_dir / "tests"
    records = []
    unreadable = 0
    for suite, rel, kind in SUITES:
        suite_dir = rust_dir / rel
        if not suite_dir.is_dir():
            print(f"warning: suite dir missing: {suite_dir}", file=sys.stderr)
            continue
        aux_idx = aux_index(suite_dir) if kind == "rs" else {}
        if kind == "rs":
            paths = discover_rs_tests(suite_dir)
            for p in paths:
                text = read_text(p)
                if text is None:
                    unreadable += 1
                    continue
                records.append(classify(suite, kind, p, text, aux_idx))
        elif kind == "rmake":
            for d in discover_rmake_tests(suite_dir):
                text = read_text(d / "rmake.rs")
                if text is None:
                    unreadable += 1
                    continue
                records.append(classify(suite, kind, d, text, aux_idx))
        else:
            ext = ".goml" if kind == "goml" else ".js"
            for p in discover_script_tests(suite_dir, ext):
                text = read_text(p)
                if text is None:
                    unreadable += 1
                    continue
                records.append(classify(suite, kind, p, text, aux_idx))
    for r in records:
        r["path"] = str(Path(r["path"]).relative_to(rust_dir))
    return records, unreadable


def aggregate(records):
    def count(pred):
        return sum(1 for r in records if pred(r))

    by_suite = {}
    for suite, _rel, _kind in SUITES:
        rs = [r for r in records if r["suite"] == suite]
        if rs:
            by_suite[suite] = len(rs)
    agg = {
        "total": len(records),
        "by_suite": by_suite,
        "by_nature": {
            n: count(lambda r, n=n: r["nature"] == n)
            for n in ("exec", "exec-debugger", "procedure",
                      "compile-check", "rustdoc-check")
        },
        "single_file": count(lambda r: r["single_file"]),
        "multi_file": count(lambda r: not r["single_file"]),
        "build_procedure": count(lambda r: r["build_procedure"]),
        "ffi": count(lambda r: r["ffi"]),
        "env_excluded": count(lambda r: r["env_excluded"]),
        "denominator": count(lambda r: not r["env_excluded"]),
        "revision_expanded_total": sum(r["revisions"] for r in records),
        "excluded_tests": [
            {"suite": r["suite"], "path": r["path"], "reasons": r["env_reasons"]}
            for r in records if r["env_excluded"]
        ],
    }
    return agg


def fetch(dest):
    if dest.exists():
        print(f"fetch: {dest} already exists, skipping clone")
        return
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", TAG,
         "--filter=blob:none", "--sparse",
         "https://github.com/rust-lang/rust.git", str(dest)],
        check=True)
    subprocess.run(
        ["git", "-C", str(dest), "sparse-checkout", "set", "tests",
         "src/tools/compiletest", "src/bootstrap"],
        check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rust-dir", type=Path, default=DEFAULT_FETCH_DIR,
                    help="checkout of rust-lang/rust at the tag "
                         f"(default: {DEFAULT_FETCH_DIR})")
    ap.add_argument("--fetch", action="store_true",
                    help="clone the rust repo (sparse) if --rust-dir "
                         "does not exist")
    ap.add_argument("--json", type=Path,
                    help="write full per-test records and aggregates "
                         "to this JSON file")
    args = ap.parse_args()

    rust_dir = args.rust_dir
    if not (rust_dir / "tests").is_dir():
        if args.fetch:
            fetch(rust_dir)
        else:
            print(f"error: {rust_dir}/tests not found (use --fetch)",
                  file=sys.stderr)
            return 1

    commit = "unknown"
    try:
        commit = subprocess.run(
            ["git", "-C", str(rust_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        pass

    records, unreadable = survey(rust_dir)
    agg = aggregate(records)

    print(f"rust-lang/rust tag {TAG} (commit {commit})")
    print(f"tests surveyed: {agg['total']} "
          f"(unreadable files skipped: {unreadable})")
    print(f"denominator after environmental exclusions: "
          f"{agg['denominator']}")
    print()
    print("by suite:")
    for suite, n in agg["by_suite"].items():
        print(f"  {suite:22s} {n}")
    print("by nature:")
    for nature, n in agg["by_nature"].items():
        print(f"  {nature:22s} {n}")
    print(f"single-file: {agg['single_file']}  "
          f"multi-file/auxiliary: {agg['multi_file']}")
    print(f"build procedure (run-make family): {agg['build_procedure']}")
    print(f"FFI: {agg['ffi']}")
    print(f"revision-expanded total: {agg['revision_expanded_total']}")
    print(f"environmental exclusions: {agg['env_excluded']}")
    for e in agg["excluded_tests"]:
        print(f"  - {e['path']}: {'; '.join(e['reasons'])}")

    if args.json:
        payload = {
            "tag": TAG,
            "commit": commit,
            "aggregates": agg,
            "tests": sorted(records, key=lambda r: r["path"]),
        }
        args.json.write_text(json.dumps(payload, indent=1, sort_keys=True))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
