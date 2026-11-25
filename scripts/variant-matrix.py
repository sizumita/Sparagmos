#!/usr/bin/env python3
"""MIR variant implementation matrix for metric 1 in docs/COVERAGE.md.

For each of the 40 denominator variants (StatementKind 12 + TerminatorKind
15 + Rvalue 13, tabulated in crates/sparagmos-mir/src/coverage.rs
from a mechanical extraction of rustc 1.100.0's MIR enums), this
script mechanically cross-references three kinds of evidence:

  driver  -- crates/sparagmos-driver/src/convert.rs has a match arm for the
             variant that constructs the sparagmos-mir counterpart without
             unconditionally recording a conversion error;
  interp  -- a passing test constructs the sparagmos-mir variant in a body
             and executes it through the interpreter with a semantic
             assertion (direct evidence from the sparagmos-interp test
             suite, plus the parity tests in the sparagmos-jit suite whose
             `check` helper runs the interpreter as well);
  jit     -- a passing test constructs the sparagmos-mir variant and
             executes it through the JIT.

A variant is "implemented" only when all three hold (the definition in
docs/COVERAGE.md).  Exactly one engine having a test makes it "partial".
Nine denominator variants cannot occur in Stable MIR 1.100.0 output at
all; they are reported with the third status "unreachable" (see
UNREACHABLE below) and are never counted as implemented.

Method, and its limits:

  * Test files are scanned per function.  Variant references found in a
    helper function are attributed to the tests that call it (transitively,
    within one file).  A reference counts only when the test reaches an
    execution entry point of the engine and the test passes in a real
    `cargo test` run performed by this script.
  * Tests that expect an error (expect_err / is_err / should_panic /
    mentions of Unsupported) are flagged and contribute nothing unless a
    manual review entry below (MANUAL_REVIEWS) says the expectation is a
    guest-panic semantic assertion, which is a semantic test.  Every manual
    entry states its reason; everything not listed there is decided by the
    mechanical scan alone.
  * Tests written against the interpreter's internal IR
    (crates/sparagmos-interp/tests/interp_tests.rs) do not construct
    sparagmos-mir variants; they are reported separately as internal-IR
    evidence and never counted.
  * The corpus harness tests (sparagmos-harness) are not scanned: mapping
    corpus programs to variants would need driver dumps, which are not
    checked in.  The count is therefore strict.

Usage: python3 scripts/variant-matrix.py [--repo PATH] [--out matrix.json]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

ENUMS = ["StatementKind", "TerminatorKind", "Rvalue"]
MIR_SRC = {
    "StatementKind": "crates/sparagmos-mir/src/statement.rs",
    "TerminatorKind": "crates/sparagmos-mir/src/terminator.rs",
    "Rvalue": "crates/sparagmos-mir/src/rvalue.rs",
}
DRIVER_SRC = "crates/sparagmos-driver/src/convert.rs"
COVERAGE_RS = "crates/sparagmos-mir/src/coverage.rs"
# Metric-1 tables in coverage.rs: enum -> (denominator const,
# unreachable const).
METRIC1_TABLES = {
    "StatementKind": ("METRIC1_STATEMENT_KINDS",
                      "METRIC1_UNREACHABLE_STATEMENT_KINDS"),
    "TerminatorKind": ("METRIC1_TERMINATOR_KINDS",
                       "METRIC1_UNREACHABLE_TERMINATOR_KINDS"),
    "Rvalue": ("METRIC1_RVALUE_KINDS", "METRIC1_UNREACHABLE_RVALUE_KINDS"),
}

# Denominator name -> sparagmos-mir variant name, where they differ.
# Anything not listed maps to itself; a denominator name whose image is
# absent from the sparagmos-mir enum has no counterpart at all.
SM_NAME = {
    "TerminatorKind": {"UnwindResume": "Resume", "UnwindTerminate": "Terminate"},
}
# Denominator name -> rustc_public variant name matched on by the driver,
# where it differs from both the denominator and the sparagmos-mir name.
DRIVER_SRC_NAME = {
    "TerminatorKind": {"UnwindResume": "Resume", "UnwindTerminate": "Abort"},
    "Rvalue": {"RawPtr": "AddressOf"},
}
DRIVER_ALIAS = {"StatementKind": "SK", "TerminatorKind": "TK", "Rvalue": "R"}

# Manual review of mechanically flagged tests.  The mechanical scan flags
# every test that (transitively) expects an error; flagged tests count for
# nothing unless an entry here marks specific (test, engine, variant)
# combinations as genuine semantic tests.  Entries with "counts": False
# exclude a combination even when the scan did not flag it.
# Each entry: binary, test, engine, variant ("*" = all variants the test
# references), counts, reason.
MANUAL_REVIEWS = [
    # Guest-panic assertions (InterpError::Panic) are semantic tests: the
    # guest program's Assert/bounds semantics are exactly what is verified.
    {"binary": "mir_convert_tests", "test": "checked_add_overflow_is_a_guest_panic",
     "engine": "interp", "variant": "*", "counts": True,
     "reason": "asserts InterpError::Panic from the Assert terminator"},
    {"binary": "mir_convert_tests", "test": "literal_rem_min_by_neg_one_still_panics",
     "engine": "interp", "variant": "*", "counts": True,
     "reason": "asserts InterpError::Panic from the overflow Assert"},
    {"binary": "print_support_tests", "test": "array_index_out_of_bounds_is_a_guest_panic",
     "engine": "interp", "variant": "*", "counts": True,
     "reason": "asserts InterpError::Panic with the rustc bounds-check message"},
    # Host-side Unsupported/error assertions document non-support, not
    # semantics; the variants involved are covered by other tests.
    {"binary": "mir_convert_tests", "test": "drop_terminator_fails_at_execution_not_silently",
     "engine": "interp", "variant": "*", "counts": False,
     "reason": "asserts UnsupportedTerminator(Drop) for an unmodelled ADT place"},
    {"binary": "mir_convert_tests", "test": "unresolved_callee_fails_at_execution_not_silently",
     "engine": "interp", "variant": "*", "counts": False,
     "reason": "asserts UnsupportedTerminator for a Call to an unknown DefId"},
    {"binary": "mir_convert_tests", "test": "closure_aggregate_fails_at_execution_not_silently",
     "engine": "interp", "variant": "*", "counts": False,
     "reason": "asserts UnsupportedRvalue for AggregateKind::Closure"},
    {"binary": "mir_convert_tests", "test": "uninitialized_non_unit_return_place_is_still_an_error",
     "engine": "interp", "variant": "*", "counts": False,
     "reason": "asserts InterpError::UninitializedRead, a host-side error"},
    # The call attempts fail (UnknownFunction / WrongArgCount) before any
    # generated code runs, so no variant semantics are exercised.
    {"binary": "mir_jit", "test": "call_api_errors",
     "engine": "jit", "variant": "*", "counts": False,
     "reason": "jit.call fails before execution; semantics not exercised"},
]

# Driver arms whose self.err(...) call fires only on a sub-case, while the
# arm does construct the real variant on the main path.
CONDITIONAL_ERR_OK = {
    # convert.rs: R::Repeat records an error only when the repeat count is
    # not evaluable as a constant; otherwise it constructs Rvalue::Repeat.
    ("Rvalue", "Repeat"),
    # convert.rs: cast_kind() records an error only for
    # CastKind::BoxDerefTransmute; other CastKinds construct Rvalue::Cast.
    ("Rvalue", "Cast"),
    # convert.rs: TK::InlineAsm records an error only when the template
    # is outside the no-op class (see inline_asm_is_noop in
    # sparagmos-mir); no-op templates construct TerminatorKind::InlineAsm.
    ("TerminatorKind", "InlineAsm"),
}

# Denominator variants that cannot occur in Stable MIR 1.100.0 output at
# all: none of them exists in the rustc_public mir/body.rs enums the
# driver consumes (verified against the installed 1.100.0 sources; the
# same set is tabulated in crates/sparagmos-mir/src/coverage.rs as
# METRIC1_UNREACHABLE_*).  They stay in the denominator (40) and are
# reported with the third status "unreachable", never as "implemented".
UNREACHABLE = {
    ("StatementKind", "BackwardIncompatibleDropHint"),
    ("TerminatorKind", "UnwindResume"),
    ("TerminatorKind", "UnwindTerminate"),
    ("TerminatorKind", "TailCall"),
    ("TerminatorKind", "Yield"),
    ("TerminatorKind", "CoroutineDrop"),
    ("TerminatorKind", "FalseEdge"),
    ("TerminatorKind", "FalseUnwind"),
    ("Rvalue", "WrapUnsafeBinder"),
}


# --------------------------------------------------------------------------
# Rust source handling
# --------------------------------------------------------------------------

def strip_rust(src: str) -> str:
    """Remove comments and string/char literal contents (positions kept
    approximately: removed text becomes spaces, newlines preserved)."""
    out = []
    i, n = 0, len(src)
    state = "code"
    raw_hashes = 0
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "/":
                state = "line_comment"
                out.append("  ")
                i += 2
            elif c == "/" and nxt == "*":
                state = "block_comment"
                out.append("  ")
                i += 2
            elif c == "r" and (nxt == '"' or nxt == "#"):
                # raw string r"..." or r#"..."#
                j = i + 1
                h = 0
                while j < n and src[j] == "#":
                    h += 1
                    j += 1
                if j < n and src[j] == '"':
                    raw_hashes = h
                    state = "raw_string"
                    out.append(" " * (j - i + 1))
                    i = j + 1
                else:
                    out.append(c)
                    i += 1
            elif c == '"':
                state = "string"
                out.append(" ")
                i += 1
            elif c == "'" and nxt and (nxt == "\\" or (i + 2 < n and src[i + 2] == "'")):
                state = "char"
                out.append(" ")
                i += 1
            else:
                out.append(c)
                i += 1
        elif state == "line_comment":
            if c == "\n":
                state = "code"
                out.append(c)
            else:
                out.append(" ")
            i += 1
        elif state == "block_comment":
            if c == "*" and nxt == "/":
                state = "code"
                out.append("  ")
                i += 2
            else:
                out.append("\n" if c == "\n" else " ")
                i += 1
        elif state == "string":
            if c == "\\":
                out.append("  ")
                i += 2
            elif c == '"':
                state = "code"
                out.append(" ")
                i += 1
            else:
                out.append("\n" if c == "\n" else " ")
                i += 1
        elif state == "raw_string":
            if c == '"' and src[i + 1 : i + 1 + raw_hashes] == "#" * raw_hashes:
                state = "code"
                out.append(" " * (1 + raw_hashes))
                i += 1 + raw_hashes
            else:
                out.append("\n" if c == "\n" else " ")
                i += 1
        elif state == "char":
            if c == "\\":
                out.append("  ")
                i += 2
            elif c == "'":
                state = "code"
                out.append(" ")
                i += 1
            else:
                out.append(" ")
                i += 1
    return "".join(out)


def parse_str_table(rust_src: str, const_name: str) -> list:
    """String literals of `pub const <const_name>: &[&str] = &[...];`.

    Comments inside the array are skipped; the scan stops at the
    array's closing bracket."""
    m = re.search(r"pub const " + const_name + r"\b", rust_src)
    if not m:
        raise ValueError(f"const {const_name} not found")
    eq = rust_src.index("=", m.end())
    start = rust_src.index("[", eq)
    names = []
    i, n = start + 1, len(rust_src)
    while i < n:
        c = rust_src[i]
        if c == "]":
            return names
        if c == "/" and rust_src[i + 1 : i + 2] == "/":
            j = rust_src.find("\n", i)
            i = n if j == -1 else j + 1
        elif c == "/" and rust_src[i + 1 : i + 2] == "*":
            j = rust_src.find("*/", i + 2)
            if j == -1:
                break
            i = j + 2
        elif c == '"':
            j = i + 1
            buf = []
            while j < n and rust_src[j] != '"':
                if rust_src[j] == "\\" and j + 1 < n:
                    j += 1
                buf.append(rust_src[j])
                j += 1
            names.append("".join(buf))
            i = j + 1
        else:
            i += 1
    raise ValueError(f"unterminated string table {const_name}")


def parse_metric1_tables(coverage_rs: str) -> tuple:
    """The metric-1 denominator and unreachable set, parsed from the
    METRIC1_* tables in crates/sparagmos-mir/src/coverage.rs.

    Those tables are the mechanically extracted variant lists of
    rustc 1.100.0's MIR enums, pinned by unit tests in coverage.rs
    itself.  docs/COVERAGE.md defines the metric in prose but no
    longer tabulates the names, so the code tables are the parseable
    source of truth.

    Returns ({enum: [variant, ...]}, {(enum, variant), ...})."""
    denominator = {}
    unreachable = set()
    for enum in ENUMS:
        denom_const, unreach_const = METRIC1_TABLES[enum]
        denominator[enum] = parse_str_table(coverage_rs, denom_const)
        for name in parse_str_table(coverage_rs, unreach_const):
            unreachable.add((enum, name))
    return denominator, unreachable


def parse_enum_variants(rust_src: str, enum_name: str) -> list:
    """Top-level variant names of `pub enum <enum_name>` in Rust source."""
    text = strip_rust(rust_src)
    m = re.search(r"pub enum " + enum_name + r"\b", text)
    if not m:
        raise ValueError(f"enum {enum_name} not found")
    brace = text.index("{", m.end())
    depth = 0
    variants = []
    i = brace
    line_start = True
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
            i += 1
        elif c == "}":
            depth -= 1
            i += 1
            if depth == 0:
                break
        elif depth == 1 and line_start:
            mm = re.match(r"\s*([A-Z][A-Za-z0-9_]*)\s*[(,{]", text[i:])
            if mm:
                variants.append(mm.group(1))
                i += mm.end()
                if mm.group(0).rstrip().endswith("{"):
                    depth += 1  # struct variant: its '{' was consumed
            else:
                i += 1
            line_start = False
        else:
            if c == "\n":
                line_start = True
            i += 1
    return variants


# --------------------------------------------------------------------------
# Driver scan
# --------------------------------------------------------------------------

def scan_driver(convert_rs: str) -> dict:
    """Per denominator variant: does the driver convert it?

    Returns {enum: {name: {arm, constructs, err_strings, class}}}."""
    text = strip_rust(convert_rs)
    # String literals were blanked by strip_rust; err() messages must be
    # read from the original source instead.
    err_strings = re.findall(r'self\.err\("([^"]*)"\)', convert_rs)
    out = {}
    for enum in ENUMS:
        alias = DRIVER_ALIAS[enum]
        per = {}
        for name in DENOMINATOR[enum]:
            src_name = DRIVER_SRC_NAME.get(enum, {}).get(
                name, SM_NAME.get(enum, {}).get(name, name)
            )
            sm_name = SM_NAME.get(enum, {}).get(name, name)
            arm = re.search(r"\b" + alias + r"::" + src_name + r"\b", text) is not None
            constructs = (
                re.search(r"\bsm::" + enum + r"::" + sm_name + r"\b", text) is not None
            )
            # An err string counts against the variant only when it names
            # the variant's own path (e.g. "Rvalue::Reborrow"); strings that
            # merely contain the name as a substring of another path
            # (e.g. "CastKind::BoxDerefTransmute" for Rvalue::Cast) are
            # recorded as related evidence for manual classification.
            path_forms = {f"{enum}::{src_name}", f"{enum}::{sm_name}", f"{enum}::{name}"}
            errs = [s for s in err_strings if any(p in s for p in path_forms)]
            related = [
                s
                for s in err_strings
                if s not in errs and (name in s or src_name in s or sm_name in s)
            ]
            if sm_name not in SM_VARIANTS[enum]:
                cls = "no-type"
            elif not arm:
                cls = "no-arm"
            elif errs and (enum, name) not in CONDITIONAL_ERR_OK:
                cls = "err"
            elif errs or (related and (enum, name) in CONDITIONAL_ERR_OK):
                cls = "conditional"
            else:
                cls = "ok"
            per[name] = {
                "arm": arm,
                "constructs": constructs,
                "err_strings": errs,
                "related_err_strings": related,
                "class": cls,
            }
        out[enum] = per
    return out


# --------------------------------------------------------------------------
# Test-file scan
# --------------------------------------------------------------------------

INTERP_EXEC = [
    re.compile(r"interpret_mir\s*\("),
    re.compile(r"interpret_mir_captured\s*\("),
    re.compile(r"(?<![\w:])interpret\s*\("),
    re.compile(r"interp_call\s*\("),
]
JIT_EXEC = [
    re.compile(r"jit_call\s*\("),
    re.compile(r"\.call\s*\("),
]
NEGATIVE = [
    re.compile(r"expect_err"),
    re.compile(r"unwrap_err"),
    re.compile(r"is_err\s*\(\s*\)"),
    re.compile(r"Unsupported"),
]

# Internal-IR constructor methods that stand for a variant of the same
# denominator name (sparagmos-interp ir::Rvalue).
INTERNAL_CTOR = {"binary": "BinaryOp", "checked_binary": "BinaryOp"}


class FnInfo:
    def __init__(self, name, body, is_test, should_panic):
        self.name = name
        self.body = body
        self.is_test = is_test
        self.should_panic = should_panic
        self.direct = {e: set() for e in ENUMS}     # denominator names
        self.internal = {e: set() for e in ENUMS}   # denominator names
        self.calls = set()
        self.interp_exec = False
        self.jit_exec = False
        self.negative = should_panic


def extract_functions(text: str) -> dict:
    """{name: FnInfo} for every fn in comment/string-stripped source."""
    fns = {}
    prev_end = 0
    for m in re.finditer(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)", text):
        name = m.group(1)
        brace = text.find("{", m.end())
        if brace == -1:
            continue
        depth = 0
        i = brace
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = text[brace : i + 1]
        # Attributes of this fn live between the previous item's end and
        # this fn's start.
        segment = text[prev_end : m.start()]
        is_test = "#[test]" in segment
        should_panic = "#[should_panic]" in segment
        fns[name] = FnInfo(name, body, is_test, should_panic)
        prev_end = i + 1
    return fns


def file_resolver(text: str) -> dict:
    """Decide what bare enum names and the `mir::` prefix refer to.

    Returns {"mir_prefix": bool, "bare": {EnumName: "direct"|"internal"|None},
             "bare_statement": "direct"|"internal"|None}."""
    mir_prefix = re.search(r"use\s+sparagmos_mir\s+as\s+mir\s*;", text) is not None
    bare = {}
    stmt = None
    interp_imports = set()
    for m in re.finditer(r"use\s+sparagmos_interp(?:::ir)?::\{([^}]*)\}", text):
        interp_imports |= {x.strip().split(" as ")[-1] for x in m.group(1).split(",")}
    if re.search(r"use\s+sparagmos_interp::\{", text) and "ir" in text:
        pass
    glob_ir = re.search(r"use\s+sparagmos_interp::ir::\*", text) is not None
    mir_imports = set()
    for m in re.finditer(r"use\s+sparagmos_mir::\{([^}]*)\}", text):
        mir_imports |= {x.strip().split(" as ")[-1] for x in m.group(1).split(",")}
    for enum in ENUMS:
        if enum in interp_imports or glob_ir:
            bare[enum] = "internal"
        elif enum in mir_imports:
            bare[enum] = "direct"
        else:
            bare[enum] = None
    if "Statement" in interp_imports or glob_ir:
        stmt = "internal"
    elif "Statement" in mir_imports:
        stmt = "direct"
    return {"mir_prefix": mir_prefix, "bare": bare, "bare_statement": stmt}


def sm_to_denominator(enum: str, sm_name: str):
    inv = {v: k for k, v in SM_NAME.get(enum, {}).items()}
    if sm_name in inv:
        return inv[sm_name]
    if sm_name in DENOMINATOR[enum]:
        return sm_name
    return None  # sparagmos-mir-only variant (not in the denominator)


def scan_function_refs(fn: FnInfo, resolver: dict):
    body = fn.body
    if resolver["mir_prefix"]:
        for enum in ENUMS:
            for v in re.findall(r"mir::" + enum + r"::([A-Z][A-Za-z0-9_]*)", body):
                d = sm_to_denominator(enum, v)
                if d:
                    fn.direct[enum].add(d)
        if re.search(r"mir::Statement::assign\s*\(", body):
            fn.direct["StatementKind"].add("Assign")
        if re.search(r"mir::Statement::nop\s*\(", body):
            fn.direct["StatementKind"].add("Nop")
    for enum in ENUMS:
        cls = resolver["bare"].get(enum)
        if cls is None:
            continue
        target = fn.direct if cls == "direct" else fn.internal
        for v in re.findall(
            r"(?<![\w:])" + enum + r"::([A-Z][A-Za-z0-9_]*)", body
        ):
            if cls == "direct":
                d = sm_to_denominator(enum, v)
                if d:
                    target[enum].add(d)
            else:
                if v in DENOMINATOR[enum]:
                    target[enum].add(v)
        if cls == "internal" and enum == "Rvalue":
            for ctor, variant in INTERNAL_CTOR.items():
                if re.search(r"(?<![\w:])Rvalue::" + ctor + r"\s*\(", body):
                    target[enum].add(variant)
    if resolver["bare_statement"] in ("direct", "internal"):
        target = (
            fn.direct if resolver["bare_statement"] == "direct" else fn.internal
        )
        if re.search(r"(?<![\w:])Statement::assign\s*\(", body):
            target["StatementKind"].add("Assign")
        if re.search(r"(?<![\w:])Statement::nop\s*\(", body):
            target["StatementKind"].add("Nop")
    for pat in INTERP_EXEC:
        if pat.search(body):
            fn.interp_exec = True
    for pat in JIT_EXEC:
        if pat.search(body):
            fn.jit_exec = True
    for pat in NEGATIVE:
        if pat.search(body):
            fn.negative = True


def propagate(fns: dict):
    """Union refs/flags from callees into callers, transitively."""
    changed = True
    while changed:
        changed = False
        for fn in fns.values():
            for callee_name in list(fn.calls):
                callee = fns.get(callee_name)
                if callee is None or callee is fn:
                    continue
                for enum in ENUMS:
                    for bucket in ("direct", "internal"):
                        src = getattr(callee, bucket)[enum]
                        dst = getattr(fn, bucket)[enum]
                        if not src <= dst:
                            dst |= src
                            changed = True
                for flag in ("interp_exec", "jit_exec", "negative"):
                    if getattr(callee, flag) and not getattr(fn, flag):
                        setattr(fn, flag, True)
                        changed = True


def scan_test_file(path: str, engine_scope: str) -> dict:
    """Returns {test_name: FnInfo} with propagated refs.

    engine_scope: "interp" or "jit" -- JIT exec patterns are only trusted
    in sparagmos-jit files."""
    with open(path) as f:
        raw = f.read()
    text = strip_rust(raw)
    resolver = file_resolver(raw)
    fns = extract_functions(text)
    for fn in fns.values():
        fn.calls = {
            m
            for m in re.findall(r"(?<![\w:.])([a-z_][A-Za-z0-9_]*)\s*\(", fn.body)
            if m in fns and m != fn.name
        }
        scan_function_refs(fn, resolver)
        if engine_scope != "jit":
            fn.jit_exec = False
    propagate(fns)
    return {name: fn for name, fn in fns.items() if fn.is_test}


# --------------------------------------------------------------------------
# cargo test execution
# --------------------------------------------------------------------------

TEST_TARGETS = [
    # (package, extra args, binary label, files scanned for this binary)
    ("sparagmos-interp", ["--test", "interp_tests"], "interp_tests",
     ["crates/sparagmos-interp/tests/interp_tests.rs"]),
    ("sparagmos-interp", ["--test", "mir_convert_tests"], "mir_convert_tests",
     ["crates/sparagmos-interp/tests/mir_convert_tests.rs"]),
    ("sparagmos-interp", ["--test", "print_support_tests"], "print_support_tests",
     ["crates/sparagmos-interp/tests/print_support_tests.rs"]),
    ("sparagmos-interp", ["--lib"], "sparagmos_interp_lib",
     ["crates/sparagmos-interp/src/lib.rs", "crates/sparagmos-interp/src/interp.rs",
      "crates/sparagmos-interp/src/mir_convert/mod.rs", "crates/sparagmos-interp/src/ir.rs",
      "crates/sparagmos-interp/src/builtins/mod.rs", "crates/sparagmos-interp/src/value.rs",
      "crates/sparagmos-interp/src/error.rs"]),
    ("sparagmos-jit", ["--test", "mir_jit"], "mir_jit",
     ["crates/sparagmos-jit/tests/mir_jit.rs"]),
    ("sparagmos-jit", ["--test", "jit_exec"], "jit_exec",
     ["crates/sparagmos-jit/tests/jit_exec.rs"]),
    ("sparagmos-jit", ["--lib"], "sparagmos_jit_lib",
     ["crates/sparagmos-jit/src/lib.rs", "crates/sparagmos-jit/src/compiler.rs",
      "crates/sparagmos-jit/src/emitter.rs", "crates/sparagmos-jit/src/memory.rs",
      "crates/sparagmos-jit/src/program.rs", "crates/sparagmos-jit/src/mir_jit.rs",
      "crates/sparagmos-jit/src/ir.rs"]),
]


def run_cargo_tests(repo: str) -> dict:
    """{binary_label: {test_fn_name: True(passed)/False}} from real runs."""
    env = dict(os.environ)
    cargo_bin = os.path.expanduser("~/.cargo/bin")
    if os.path.isdir(cargo_bin):
        env["PATH"] = cargo_bin + os.pathsep + env.get("PATH", "")
    results = {}
    for package, args, label, _files in TEST_TARGETS:
        cmd = ["cargo", "test", "-p", package] + args
        proc = subprocess.run(
            cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=3600
        )
        per = {}
        for line in proc.stdout.splitlines():
            m = re.match(r"test (\S+) \.\.\. (ok|FAILED|ignored)", line)
            if m:
                per[m.group(1).split("::")[-1]] = m.group(2) == "ok"
        results[label] = per
    return results


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def review_for(binary, test, engine, variant):
    """Manual review verdict: True (counts), False (excluded), None."""
    verdict = None
    for r in MANUAL_REVIEWS:
        if (
            r["binary"] == binary
            and r["test"] == test
            and r["engine"] == engine
            and (r["variant"] == "*" or r["variant"] == variant)
        ):
            verdict = r["counts"]
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-tests", action="store_true",
                    help="do not run cargo test (all tests treated as not passed)")
    args = ap.parse_args()
    repo = args.repo

    global DENOMINATOR, SM_VARIANTS
    with open(os.path.join(repo, COVERAGE_RS)) as f:
        DENOMINATOR, unreachable_in_tables = parse_metric1_tables(f.read())
    total = sum(len(v) for v in DENOMINATOR.values())
    if total != 40:
        print(f"warning: denominator has {total} variants, expected 40", file=sys.stderr)
    if unreachable_in_tables != UNREACHABLE:
        print("warning: UNREACHABLE disagrees with the METRIC1_UNREACHABLE_* "
              "tables in coverage.rs", file=sys.stderr)
    SM_VARIANTS = {}
    for enum in ENUMS:
        with open(os.path.join(repo, MIR_SRC[enum])) as f:
            SM_VARIANTS[enum] = parse_enum_variants(f.read(), enum)

    with open(os.path.join(repo, DRIVER_SRC)) as f:
        driver = scan_driver(f.read())

    # Scan test files per binary target.
    scanned = {}  # label -> {test_name: FnInfo}
    for _pkg, _args, label, files in TEST_TARGETS:
        engine_scope = "jit" if label.startswith(("mir_jit", "jit_exec", "sparagmos_jit")) else "interp"
        merged = {}
        for rel in files:
            path = os.path.join(repo, rel)
            if not os.path.exists(path):
                continue
            if "/src/" in rel:
                # Only the #[cfg(test)] part of a source file is test code.
                with open(path) as f:
                    content = f.read()
                idx = content.find("#[cfg(test)]")
                if idx == -1:
                    continue
                fd, tmp = tempfile.mkstemp(suffix=".rs")
                try:
                    with os.fdopen(fd, "w") as f:
                        f.write(content[idx:])
                    merged.update(scan_test_file(tmp, engine_scope))
                finally:
                    os.unlink(tmp)
            else:
                merged.update(scan_test_file(path, engine_scope))
        scanned[label] = merged

    results = {} if args.skip_tests else run_cargo_tests(repo)

    variants_out = []
    summary = {"implemented": 0, "partial": 0, "none": 0, "unreachable": 0}
    flagged = []
    for enum in ENUMS:
        for name in DENOMINATOR[enum]:
            interp_tests, jit_tests, internal_tests = [], [], []
            for _pkg, _a, label, _f in TEST_TARGETS:
                passed_map = results.get(label, {})
                for tname, fn in scanned.get(label, {}).items():
                    if name not in fn.direct[enum] and name not in fn.internal[enum]:
                        continue
                    passed = passed_map.get(tname, False)
                    if name in fn.internal[enum] and name not in fn.direct[enum]:
                        if fn.interp_exec and passed:
                            internal_tests.append(f"{label}::{tname}")
                        continue
                    if not passed:
                        continue
                    for engine, exec_flag, acc in (
                        ("interp", fn.interp_exec, interp_tests),
                        ("jit", fn.jit_exec, jit_tests),
                    ):
                        if not exec_flag:
                            continue
                        verdict = review_for(label, tname, engine, name)
                        if fn.negative and verdict is not True:
                            flagged.append({
                                "binary": label, "test": tname, "engine": engine,
                                "variant": f"{enum}::{name}",
                                "review": verdict,
                            })
                            continue
                        if verdict is False:
                            continue
                        acc.append(f"{label}::{tname}")
            drv = driver[enum][name]
            driver_ok = drv["class"] in ("ok", "conditional")
            has_i, has_j = bool(interp_tests), bool(jit_tests)
            if (enum, name) in UNREACHABLE:
                status = "unreachable"
            elif driver_ok and has_i and has_j:
                status = "implemented"
            elif has_i or has_j:
                status = "partial"
            else:
                status = "none"
            summary[status] += 1
            variants_out.append({
                "enum": enum,
                "name": name,
                "sparagmos_mir_name": SM_NAME.get(enum, {}).get(name, name)
                if SM_NAME.get(enum, {}).get(name, name) in SM_VARIANTS[enum]
                else None,
                "driver": drv,
                "driver_ok": driver_ok,
                "interp_tests": sorted(set(interp_tests)),
                "jit_tests": sorted(set(jit_tests)),
                "internal_ir_tests": sorted(set(internal_tests)),
                "status": status,
            })

    extras = {
        enum: [v for v in SM_VARIANTS[enum] if sm_to_denominator(enum, v) is None]
        for enum in ENUMS
    }
    implemented = summary["implemented"]
    report = {
        "denominator": {e: DENOMINATOR[e] for e in ENUMS},
        "denominator_total": total,
        "sparagmos_mir_extras_not_in_denominator": extras,
        "summary": {
            **summary,
            "implemented_rate_percent": round(implemented / total * 100, 1)
            if total
            else 0.0,
        },
        "flagged_negative_candidates": flagged,
        "variants": variants_out,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
