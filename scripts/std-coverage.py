#!/usr/bin/env python3
"""Metric 3 (std feature coverage) extractor for Sparagmos.

The denominator of metric 3 is the set of `std` / `core` / `alloc`
function paths that actually appear as callees in the converted MIR of
the metric-2 programs: the own corpus (tests/corpus) plus the rustc
test-subset stage-1 candidates (exec, single-file, ui / ui-fulldeps,
3,699 files at tag 1.100.0). A path counts as covered when a call to it
is executable by the interpreter: the converted program contains a body
for it, or the interpreter resolves it as a builtin
(crates/sparagmos-interp/src/builtins/ `lookup_qualified`).

For every input file the tool runs the driver in dump mode
(`sparagmos-driver <file.rs> --dump <path> --edition <ed>`, the edition
taken from the file's `//@ edition:` directive, default 2021 — the same
rule as scripts/rustc-subset-run.py) and parses the dump: the def-name
table gives every callee path, the `=== fn` sections give the set of
functions with converted bodies, and the `func` operand of each Call /
TailCall terminator gives the callees. Names are canonicalized exactly
like the interpreter does (lifetime-only generic groups removed), so
the measured set is the set of names the interpreter itself sees.

Files the driver cannot convert at the front end produce no dump;
their callees cannot be counted and they are reported separately.

Usage:
  python3 scripts/std-coverage.py \
      --results-json <subset-run-results.json> | --survey-json <survey.json> \
      --rust-dir <rust checkout> \
      --driver-bin <sparagmos-driver> \
      --out <results JSON>
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

DRIVER_TIMEOUT = 60
STD_PREFIXES = ("std", "core", "alloc")

DIRECTIVE_RE = re.compile(
    r"^\s*//@(?:\[([^\]]+)\])?\s*([a-zA-Z][a-zA-Z0-9-]*)"
    r"(?:\s*:\s*(.*?)|\s*\((.*)\))?\s*$")

DEF_NAME_RE = re.compile(r"^  DefId\(0:(\d+)\) = (.*)$", re.M)
FN_HEADER_RE = re.compile(r"^=== fn (.*) \((?:def=DefId\(0:\d+\), )?args=\d+, blocks=\d+, locals=\d+\)$",
                          re.M)
CALL_RE = re.compile(r"kind: (?:Call|TailCall) \{")
FNDEF_RE = re.compile(r"FnDef\(\s*DefId \{\s*krate: (\d+),\s*index: (\d+)")


# ---------------------------------------------------------------------------
# Name canonicalization (mirror of builtins::canonical_name)
# ---------------------------------------------------------------------------

def is_lifetime_list(inner):
    if not inner:
        return False
    for part in inner.split(","):
        part = part.strip()
        if part == "_":
            continue
        if part.startswith("'") and len(part) > 1:
            continue
        return False
    return True


def canonical_name(name):
    out = []
    rest = name
    while True:
        open_idx = rest.find("<")
        if open_idx == -1:
            break
        depth = 0
        close = None
        for i in range(open_idx, len(rest)):
            c = rest[i]
            if c == "<":
                depth += 1
            elif c == ">":
                depth -= 1
                if depth == 0:
                    close = i
                    break
        if close is None:
            break
        inner = rest[open_idx + 1:close]
        if is_lifetime_list(inner):
            out.append(rest[:open_idx])
            joined = "".join(out)
            if joined.endswith("::"):
                joined = joined[:-2]
            out = [joined]
            rest = rest[close + 1:]
        else:
            out.append(rest[:open_idx + 1])
            rest = rest[open_idx + 1:]
    out.append(rest)
    return "".join(out)


def is_std_path(canonical):
    first = canonical.split("::", 1)[0]
    return first in STD_PREFIXES


# ---------------------------------------------------------------------------
# Builtin registry (parsed from crates/sparagmos-interp/src/builtins/)
# ---------------------------------------------------------------------------

def read_builtins_source(path):
    """Source text of the interpreter's builtin registry code.

    `path` may name the historical single `builtins.rs` file or the
    `builtins/` module directory it was split into; a directory is
    read as the concatenation of its `*.rs` files in name order.
    """
    if path.is_dir():
        return "".join(p.read_text() for p in sorted(path.glob("*.rs")))
    return path.read_text()


def _fn_body(source, sig):
    """The text of the function whose signature starts with `sig`,
    up to its closing brace at column 0 (the same convention the
    `lookup` extraction in `BuiltinRegistry` uses)."""
    start = source.find(sig)
    if start == -1:
        raise ValueError(f"builtins: {sig} not found")
    end = source.find("\n}\n", start)
    if end == -1:
        raise ValueError(f"builtins: end of {sig} not found")
    return source[start:end]


_FLOAT_PREFIXES = {
    "core::f128::<impl f128>::", "std::f128::<impl f128>::",
    "core::f16::<impl f16>::", "std::f16::<impl f16>::",
    "core::f32::<impl f32>::", "std::f32::<impl f32>::",
    "core::f64::<impl f64>::", "std::f64::<impl f64>::",
}


def _float_prefixes(body, named, f128):
    """The (prefix, method table) pairs of `float_method`, in source
    order. Each `strip_prefix` literal is served by the next resolver
    call that follows it in the body (`float128_method(` for the
    f128 prefixes, `float_method_named(` for the rest), and the
    literal set itself must be exactly the eight
    `{core,std}::fN::<impl fN>::` prefixes, so a new prefix shape is
    a hard error."""
    events = []
    for m in re.finditer(r'strip_prefix\("([^"]+)"\)', body):
        events.append((m.start(), "prefix", m.group(1)))
    for m in re.finditer(r"float128_method\(", body):
        events.append((m.start(), "table", f128))
    for m in re.finditer(r"float_method_named\(", body):
        events.append((m.start(), "table", named))
    events.sort()
    prefixes = []
    pending = []
    for _pos, kind, val in events:
        if kind == "prefix":
            pending.append(val)
        else:
            prefixes.extend((pfx, val) for pfx in pending)
            pending = []
    if pending:
        raise ValueError("builtins: float_method prefixes with no resolver")
    if {pfx for pfx, _t in prefixes} != _FLOAT_PREFIXES:
        raise ValueError("builtins: float_method prefix set changed")
    return prefixes


def _leading_segment_end(name):
    """Index just past the first `::` at angle-bracket depth 0, or
    None (mirrors `leading_segment_end` in the builtins module: a
    name starting with `<` has no strippable leading segment)."""
    if name.startswith("<"):
        return None
    depth = 0
    i = 0
    while i < len(name):
        ch = name[i]
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif ch == ":" and depth == 0 \
                and i + 1 < len(name) and name[i + 1] == ":":
            return i + 2
        i += 1
    return None


_GUARD_METHODS = ("contains", "ends_with", "starts_with")


def _tokenize_guard(text):
    """Tokens of one `lookup` guard condition: the boolean operators,
    parentheses, string literals and dotted `canonical...` atoms."""
    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
        elif text.startswith("&&", i) or text.startswith("||", i) \
                or text.startswith("==", i):
            tokens.append(text[i:i + 2])
            i += 2
        elif ch in "()":
            tokens.append(ch)
            i += 1
        elif ch == '"':
            j = i + 1
            buf = []
            while j < len(text) and text[j] != '"':
                if text[j] == "\\" and j + 1 < len(text):
                    buf.append(text[j + 1])
                    j += 2
                else:
                    buf.append(text[j])
                    j += 1
            if j >= len(text):
                raise ValueError(
                    f"builtins: unterminated string in lookup guard: {text}")
            tokens.append(("str", "".join(buf)))
            i = j + 1
        elif ch.isalpha() or ch == "_":
            j = i
            while j < len(text) and (text[j].isalnum()
                                     or text[j] in "_:."):
                j += 1
            tokens.append(("ident", text[i:j]))
            i = j
        else:
            raise ValueError(
                f"builtins: unknown lookup guard character: {ch!r} "
                f"in {text}")
    return tokens


class _GuardParser:
    """A `lookup` guard condition as a boolean expression over the
    term atoms, expanded to disjunctive normal form: a list of
    conjunctions, each a list of `(op, arg)` terms with op one of
    `eq`, `contains`, `ends_with`, `starts_with`."""

    def __init__(self, tokens, text):
        self.tokens = tokens
        self.text = text
        self.pos = 0

    def _fail(self):
        raise ValueError(f"builtins: unknown lookup guard: {self.text}")

    def _peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def parse(self):
        dnf = self._parse_or()
        if self.pos != len(self.tokens):
            self._fail()
        return dnf

    def _parse_or(self):
        dnf = self._parse_and()
        while self._peek() == "||":
            self.pos += 1
            dnf = dnf + self._parse_and()
        return dnf

    def _parse_and(self):
        dnf = self._parse_atom()
        while self._peek() == "&&":
            self.pos += 1
            rhs = self._parse_atom()
            dnf = [a + b for a in dnf for b in rhs]
        return dnf

    def _parse_atom(self):
        tok = self._peek()
        if tok == "(":
            self.pos += 1
            dnf = self._parse_or()
            if self._peek() != ")":
                self._fail()
            self.pos += 1
            return dnf
        if not (isinstance(tok, tuple) and tok[0] == "ident"):
            self._fail()
        self.pos += 1
        head = tok[1]
        if head == "canonical":
            if self._peek() != "==":
                self._fail()
            self.pos += 1
            lit = self._peek()
            if not (isinstance(lit, tuple) and lit[0] == "str"):
                self._fail()
            self.pos += 1
            return [[("eq", lit[1])]]
        prefix = "canonical."
        if not head.startswith(prefix):
            self._fail()
        method = head[len(prefix):]
        if method not in _GUARD_METHODS:
            self._fail()
        if self._peek() != "(":
            self._fail()
        self.pos += 1
        lit = self._peek()
        if not (isinstance(lit, tuple) and lit[0] == "str"):
            self._fail()
        self.pos += 1
        if self._peek() != ")":
            self._fail()
        self.pos += 1
        return [[(method, lit[1])]]


def _parse_lookup_guards(body):
    """Every guard arm of the registry fragments, each expanded to
    its DNF entries.
    An arm's condition may span several lines; it ends at the arm's
    `=>`. Any shape outside the modelled grammar is a hard error."""
    guards = []
    for text in re.findall(r"(?m)^\s*_ if (.*?)\s*=>", body, re.S):
        tokens = _tokenize_guard(text)
        guards.extend(_GuardParser(tokens, text).parse())
    return guards


class BuiltinRegistry:
    """The interpreter's builtin support, read from the builtins module.

    Exact names come from the match arms of the registration
    fragments `lookup` chains (one `<family>_lookup` function per
    builtin family module); every guard arm
    (`_ if <cond> =>`) is interpreted generically: the condition must
    be a boolean expression over `canonical.contains("..")`,
    `canonical.ends_with("..")`, `canonical.starts_with("..")` and
    `canonical == ".."` terms, combined with `&&`, `||` and
    parentheses, and is expanded to disjunctive normal form (each
    disjunct is one guard entry). Guard arms may span several lines.
    The fallback of `lookup` (the delegation its final `_ =>` arm
    always performed) hands names no fragment serves to two helper
    resolvers, modelled here from their own tables:
    `unicode_lookup_slow` (an exact set of `unicode_data` modules
    whose `::lookup_slow` callee is modelled) and `float_method`
    (the `{core,std}::f{16,32,64,128}::<impl fN>::` prefixes, each
    served by the method-name table of `float_method_named`, or of
    `float128_method` for f128). `lookup_qualified` additionally
    retries `lookup` with leading path segments stripped one at a
    time, exactly like the interpreter. Any other guard shape, or a
    helper whose shape no longer matches these expectations, is a
    hard error so a future change to the builtins module cannot
    silently drift this tool.
    """

    def __init__(self, source):
        start = source.find("pub fn lookup")
        if start == -1:
            raise ValueError("builtins: pub fn lookup not found")
        end = source.find("\n}\n", start)
        if end == -1:
            raise ValueError("builtins: end of lookup not found")
        body = source[start:end]
        # The registry is dispersed over the builtin family
        # modules: `lookup` itself is a thin chain over one
        # registration fragment per family (the `<family>_lookup`
        # functions the family modules own), so the name table is
        # the union of the fragments' bodies. The chain in `lookup`
        # is the index of fragments: a fragment the chain does not
        # name is not part of the registry, and a chain entry whose
        # function is missing is a hard error (via `_fn_body`).
        bodies = [body]
        for name in re.findall(r"([a-z][a-z0-9_]*_lookup)\(canonical\)", body):
            bodies.append(_fn_body(source, f"fn {name}(canonical"))
        table = "\n".join(bodies)
        self.exact = set(re.findall(r'"([^"\n]+)"\s*(?:\||=>)', table))
        if not self.exact:
            raise ValueError("builtins: no exact builtin names found")
        self.guards = _parse_lookup_guards(table)
        if not self.guards:
            raise ValueError("builtins: no lookup guards found")
        # Derived flags for the two historical special cases, kept for
        # the report line and the unit tests.
        self.range_inclusive_new = any(
            set(t) == {("contains", "RangeInclusive"),
                       ("ends_with", "::new")} for t in self.guards)
        self.to_uppercase = any(
            t == [("ends_with", "::to_uppercase")] for t in self.guards)
        # The `_ =>` fallback of `lookup` delegates to the two helper
        # resolvers; both call sites must be present in that shape.
        if "unicode_lookup_slow(canonical)" not in body \
                or "float_method(canonical)" not in body:
            raise ValueError(
                "builtins: lookup does not end in the modelled "
                "unicode/float fallback")
        self.unicode_modules = set(re.findall(
            r'"(core::unicode::unicode_data::[^"]+)"\s*=>',
            _fn_body(source, "fn unicode_lookup_slow(canonical")))
        if not self.unicode_modules:
            raise ValueError(
                "builtins: no unicode_lookup_slow modules found")
        self.float_methods = set(re.findall(
            r'"([^"\n]+)"\s*=>',
            _fn_body(source, "fn float_method_named(method")))
        if not self.float_methods:
            raise ValueError("builtins: no float_method_named names found")
        self.float128_methods = set(re.findall(
            r'"([^"\n]+)"\s*=>',
            _fn_body(source, "fn float128_method(method")))
        if not self.float128_methods:
            raise ValueError("builtins: no float128_method names found")
        self.float_prefixes = _float_prefixes(
            _fn_body(source, "fn float_method(canonical"),
            self.float_methods, self.float128_methods)

    def lookup(self, canonical):
        if canonical in self.exact:
            return True
        for terms in self.guards:
            ok = True
            for op, arg in terms:
                if op == "eq" and canonical != arg:
                    ok = False
                    break
                if op == "contains" and arg not in canonical:
                    ok = False
                    break
                if op == "ends_with" and not canonical.endswith(arg):
                    ok = False
                    break
                if op == "starts_with" and not canonical.startswith(arg):
                    ok = False
                    break
            if ok:
                return True
        return self._unicode_lookup_slow(canonical) \
            or self._float_method(canonical)

    def _unicode_lookup_slow(self, canonical):
        if not canonical.endswith("::lookup_slow"):
            return False
        return canonical[:-len("::lookup_slow")] in self.unicode_modules

    def _float_method(self, canonical):
        for prefix, table in self.float_prefixes:
            if canonical.startswith(prefix):
                return canonical[len(prefix):] in table
        return False

    def lookup_qualified(self, canonical):
        """`lookup` retried with leading path segments stripped one
        at a time (the interpreter's `lookup_qualified`)."""
        if self.lookup(canonical):
            return True
        rest = canonical
        while True:
            pos = _leading_segment_end(rest)
            if pos is None:
                return False
            rest = rest[pos:]
            if self.lookup(rest):
                return True


# ---------------------------------------------------------------------------
# Dump parsing
# ---------------------------------------------------------------------------

def parse_dump(text):
    """Parse one driver dump.

    Returns (def_names, body_names, callees, anomalies):
      def_names: {index: raw name}
      body_names: set of raw names with a converted body
      callees: list of raw callee names (one per Call/TailCall whose
               func operand is an FnDef constant)
      anomalies: list of strings (unresolvable DefIds etc.)
    """
    def_names = {int(m.group(1)): m.group(2) for m in DEF_NAME_RE.finditer(text)}
    body_names = set(FN_HEADER_RE.findall(text))
    callees = []
    anomalies = []
    for m in CALL_RE.finditer(text):
        seg = text[m.end():m.end() + 65536]
        args_pos = seg.find("args:")
        fndef = FNDEF_RE.search(seg)
        if fndef and (args_pos == -1 or fndef.start() < args_pos):
            krate, index = int(fndef.group(1)), int(fndef.group(2))
            if krate != 0 or index not in def_names:
                anomalies.append(
                    f"unresolvable callee DefId({krate}:{index})")
            else:
                callees.append(def_names[index])
    return def_names, body_names, callees, anomalies


def edition_of(text):
    """The file's unscoped `//@ edition:` value, default 2021
    (same rule as scripts/rustc-subset-run.py)."""
    for line in text.splitlines():
        m = DIRECTIVE_RE.match(line)
        if m and m.group(1) is None and m.group(2) == "edition":
            value = m.group(3) if m.group(3) is not None else (m.group(4) or "")
            return value.strip()
    return "2021"


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(task):
    path, source, driver_bin, timeout = task
    rec = {"path": path, "source": source, "status": "ok",
           "edition": None, "exit_code": None, "converted_fns": 0,
           "call_sites": 0, "std_paths": {}, "callees_without_body": [],
           "anomalies": [], "note": ""}
    p = Path(path)
    if not p.exists():
        rec["status"] = "missing"
        return rec
    try:
        text = p.read_text(errors="replace")
    except OSError as e:
        rec["status"] = "missing"
        rec["note"] = str(e)
        return rec
    rec["edition"] = edition_of(text)
    env = dict(os.environ, RUSTC_BOOTSTRAP="1")
    with tempfile.TemporaryDirectory(prefix="std-cov-") as td:
        dump = Path(td) / "out.dump"
        try:
            r = subprocess.run(
                [driver_bin, str(p), "--dump", str(dump),
                 "--edition", rec["edition"]],
                capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            rec["status"] = "timeout"
            return rec
        rec["exit_code"] = r.returncode
        if not dump.exists():
            rec["status"] = "no-dump"
            lines = (r.stderr or "").strip().splitlines()
            rec["note"] = lines[0][:200] if lines else ""
            return rec
        dump_text = dump.read_text(errors="replace")
    _defs, body_names, callees, anomalies = parse_dump(dump_text)
    rec["anomalies"] = anomalies
    rec["converted_fns"] = len(body_names)
    body_canonical = {canonical_name(n) for n in body_names}
    sites = Counter()
    with_body = {}
    no_body = set()
    for raw in callees:
        canon = canonical_name(raw)
        has_body = canon in body_canonical
        if is_std_path(canon):
            sites[canon] += 1
            with_body[canon] = with_body.get(canon, False) or has_body
        elif not has_body:
            no_body.add(canon)
        rec["call_sites"] += 1
    rec["std_paths"] = {name: {"sites": sites[name], "body": with_body[name]}
                        for name in sites}
    rec["callees_without_body"] = sorted(no_body)
    return rec


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(records, registry):
    paths = {}
    for rec in records:
        if rec["status"] != "ok":
            continue
        for name, info in rec["std_paths"].items():
            agg = paths.setdefault(name, {"path": name, "programs": 0,
                                          "call_sites": 0,
                                          "body_programs": 0,
                                          "builtin": registry.lookup_qualified(name)})
            agg["programs"] += 1
            agg["call_sites"] += info["sites"]
            if info["body"]:
                agg["body_programs"] += 1
    for agg in paths.values():
        agg["executable"] = agg["builtin"] or agg["body_programs"] >= 1
        agg["executable_strict"] = agg["builtin"] \
            or agg["body_programs"] == agg["programs"]
    ordered = sorted(paths.values(),
                     key=lambda a: (-a["programs"], a["path"]))
    denom = len(ordered)
    covered = sum(1 for a in ordered if a["executable"])
    covered_strict = sum(1 for a in ordered if a["executable_strict"])
    return {
        "denominator": denom,
        "executable": covered,
        "rate": (covered / denom) if denom else 0.0,
        "executable_strict": covered_strict,
        "rate_strict": (covered_strict / denom) if denom else 0.0,
        "paths": ordered,
    }


def is_stage1_candidate(rec):
    """exec, single-file, ui / ui-fulldeps: the stage-1 candidate
    predicate, over a subset-runner result record or a survey test
    entry (both carry nature, single_file, and suite)."""
    return (rec["nature"] == "exec" and rec["single_file"]
            and rec["suite"] in ("ui", "ui-fulldeps"))


def stage1_candidates(results_json):
    """The stage-1 candidate paths from the subset runner results
    (3,699 files at tag 1.100.0)."""
    d = json.loads(Path(results_json).read_text())
    return sorted(i["path"] for i in d["results"] if is_stage1_candidate(i))


def stage1_candidates_from_survey(survey_json):
    """The same candidate paths taken from the survey the runner is fed.

    The candidate predicate reads only properties the survey already
    classifies, so the set does not depend on the subset run's outcome
    and metric 3 can be measured alongside the subset run instead of
    after it. env_excluded tests never reach the runner, so they are
    not candidates. The official workflow cross-checks this set against
    the merged subset results (ci-subset-tools.py check-candidates)."""
    d = json.loads(Path(survey_json).read_text())
    return sorted(t["path"] for t in d["tests"]
                  if not t.get("env_excluded") and is_stage1_candidate(t))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-dir", type=Path, default=Path("tests/corpus"))
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--results-json", type=Path,
                     help="subset-run-results.json (stage-1 candidates)")
    src.add_argument("--survey-json", type=Path,
                     help="subset survey JSON (the same stage-1 candidates, "
                          "without waiting for the subset run)")
    ap.add_argument("--rust-dir", type=Path,
                    help="rust checkout the candidate paths are relative to")
    ap.add_argument("--driver-bin", required=True)
    ap.add_argument("--builtins-rs", type=Path,
                    default=Path("crates/sparagmos-interp/src/builtins"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=DRIVER_TIMEOUT)
    ap.add_argument("--only", help="comma-separated path substrings to keep")
    args = ap.parse_args(argv)

    registry = BuiltinRegistry(read_builtins_source(args.builtins_rs))
    tasks = []
    for f in sorted(args.corpus_dir.glob("*.rs")):
        tasks.append((str(f), "corpus", str(args.driver_bin), args.timeout))
    candidates = []
    if args.results_json:
        candidates = stage1_candidates(args.results_json)
    elif args.survey_json:
        candidates = stage1_candidates_from_survey(args.survey_json)
    for rel in candidates:
        tasks.append((str(args.rust_dir / rel), "subset",
                      str(args.driver_bin), args.timeout))
    if args.only:
        keep = [s for s in args.only.split(",") if s]
        tasks = [t for t in tasks if any(s in t[0] for s in keep)]
    print(f"files: {len(tasks)} "
          f"(builtin exact names: {len(registry.exact)}, "
          f"float methods: {len(registry.float_methods)}/"
          f"{len(registry.float128_methods)}, "
          f"unicode modules: {len(registry.unicode_modules)}, "
          f"guards: range_inclusive_new={registry.range_inclusive_new}, "
          f"to_uppercase={registry.to_uppercase})")

    records = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.jobs) as ex:
        for rec in ex.map(process_file, tasks):
            records.append(rec)
    records.sort(key=lambda r: (r["source"], r["path"]))

    agg = aggregate(records, registry)
    status_counts = Counter(r["status"] for r in records)
    summary = {
        "tool": "scripts/std-coverage.py",
        "subset_candidates": candidates,
        "files_total": len(records),
        "files_by_status": dict(status_counts),
        "files_without_dump": [r["path"] for r in records
                               if r["status"] != "ok"],
        "aggregate": agg,
        "files": records,
    }
    args.out.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"statuses: {dict(status_counts)}")
    print(f"denominator (unique std paths): {agg['denominator']}")
    print(f"executable: {agg['executable']} "
          f"({agg['rate'] * 100:.3f}%)")
    print(f"executable (strict, body in every program): "
          f"{agg['executable_strict']} ({agg['rate_strict'] * 100:.3f}%)")
    print("top unsupported paths (by programs):")
    shown = 0
    for a in agg["paths"]:
        if not a["executable"]:
            print(f"  {a['programs']:5d} programs  {a['path']}")
            shown += 1
            if shown >= 20:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
