#!/usr/bin/env python3
"""Unit tests for scripts/std-coverage.py (metric 3 extractor).

Run with: python3 -m unittest scripts.test_std_coverage
(or directly: python3 scripts/test_std_coverage.py)

Name canonicalization, the builtins registry parser, the dump
parser and the aggregation are exercised on small fixtures; the
registry is additionally checked against the real builtins module.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location(
    "std_coverage", Path(__file__).parent / "std-coverage.py")
cov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cov)

REPO_ROOT = Path(__file__).parent.parent


class CanonicalNameTests(unittest.TestCase):
    def test_lifetime_group_removed(self):
        self.assertEqual(
            cov.canonical_name("std::fmt::Arguments::<'a>::from_str"),
            "std::fmt::Arguments::from_str")

    def test_multi_lifetime_and_underscore(self):
        self.assertEqual(cov.canonical_name("foo::<'a, 'b>::bar"), "foo::bar")
        self.assertEqual(cov.canonical_name("foo::<'_>::bar"), "foo::bar")

    def test_type_group_kept(self):
        self.assertEqual(
            cov.canonical_name("std::ops::RangeInclusive::<Idx>::new"),
            "std::ops::RangeInclusive::<Idx>::new")
        self.assertEqual(
            cov.canonical_name("std::vec::Vec::<u8>::new"),
            "std::vec::Vec::<u8>::new")

    def test_no_generics_unchanged(self):
        self.assertEqual(cov.canonical_name("std::io::_print"), "std::io::_print")

    def test_unbalanced_unchanged(self):
        self.assertEqual(cov.canonical_name("a::<b"), "a::<b")


FIXTURE_BUILTINS = '''
pub fn lookup(canonical: &str) -> Option<Builtin> {
    Some(match canonical {
        "std::io::_print" => Builtin::Print,
        "std::fmt::Arguments::from_str" | "std::fmt::Arguments::from_str_nonconst" => {
            Builtin::ArgumentsFromStr
        }
        "core::fmt::rt::Argument::new_display" => Builtin::ArgumentNew(F::Display),
        _ if canonical.contains("RangeInclusive") && canonical.ends_with("::new") => {
            Builtin::RangeInclusiveNew
        }
        _ if canonical.ends_with("::to_uppercase") => Builtin::ToUppercase,
        _ => return unicode_lookup_slow(canonical).or_else(|| float_method(canonical)),
    })
}

fn unicode_lookup_slow(canonical: &str) -> Option<Builtin> {
    let module = canonical.strip_suffix("::lookup_slow")?;
    let property = match module {
        "core::unicode::unicode_data::alphabetic" => UnicodeProperty::Alphabetic,
        "core::unicode::unicode_data::n" => UnicodeProperty::N,
        _ => return None,
    };
    Some(Builtin::UnicodeLookupSlow(property))
}

fn float_method(canonical: &str) -> Option<Builtin> {
    if let Some(rest) = canonical
        .strip_prefix("core::f128::<impl f128>::")
        .or_else(|| canonical.strip_prefix("std::f128::<impl f128>::"))
    {
        return float128_method(rest);
    }
    if let Some(rest) = canonical
        .strip_prefix("core::f16::<impl f16>::")
        .or_else(|| canonical.strip_prefix("std::f16::<impl f16>::"))
    {
        return float_method_named(rest).map(|op| Builtin::Float16Method { op });
    }
    let (method, f32) = if let Some(rest) = canonical
        .strip_prefix("core::f32::<impl f32>::")
        .or_else(|| canonical.strip_prefix("std::f32::<impl f32>::"))
    {
        (rest, true)
    } else if let Some(rest) = canonical
        .strip_prefix("core::f64::<impl f64>::")
        .or_else(|| canonical.strip_prefix("std::f64::<impl f64>::"))
    {
        (rest, false)
    } else {
        return None;
    };
    let op = float_method_named(method)?;
    Some(Builtin::FloatMethod { f32, op })
}

fn float_method_named(method: &str) -> Option<FloatOp> {
    let op = match method {
        "is_nan" => O::IsNan,
        "powf" => O::Powf,
        "to_bits" => O::ToBits,
        _ => return None,
    };
    Some(op)
}

fn float128_method(method: &str) -> Option<Builtin> {
    let op = match method {
        "is_nan" => O::IsNan,
        "to_bits" => O::ToBits,
        _ => return None,
    };
    Some(Builtin::Float128Method { op })
}

fn is_file_io_call(name: &str, generic_args: &[String]) -> bool {
    match name {
        "std::fs::File::sync_all"
        | "std::fs::File::set_len" => true,
        "std::io::Read::read_to_end" => {
            generic_args.first().map(|tag| tag.as_str()).unwrap_or("")
                == "std::fs::File"
        }
        "std::io::Stdin::read_line" => true,
        _ => false,
    }
}
'''


class RegistryTests(unittest.TestCase):
    def test_fixture_registry(self):
        reg = cov.BuiltinRegistry(FIXTURE_BUILTINS)
        self.assertTrue(reg.lookup("std::io::_print"))
        self.assertTrue(reg.lookup("std::fmt::Arguments::from_str_nonconst"))
        self.assertTrue(reg.lookup("core::fmt::rt::Argument::new_display"))
        self.assertTrue(reg.lookup("std::ops::RangeInclusive::<Idx>::new"))
        self.assertTrue(reg.lookup("std::str::<impl str>::to_uppercase"))
        self.assertFalse(reg.lookup("std::vec::Vec::new"))
        self.assertFalse(reg.lookup("core::panicking::panic"))
        # The `_ =>` fallback resolvers of `lookup`: float methods
        # resolve under either the core or the std prefix, each with
        # its own method table (powf is modelled for f32/f64/f16 but
        # not for f128), and only the listed unicode_data modules
        # resolve their `::lookup_slow` callee.
        self.assertTrue(reg.lookup("core::f64::<impl f64>::to_bits"))
        self.assertTrue(reg.lookup("std::f32::<impl f32>::powf"))
        self.assertTrue(reg.lookup("core::f16::<impl f16>::is_nan"))
        self.assertTrue(reg.lookup("core::f128::<impl f128>::to_bits"))
        self.assertFalse(reg.lookup("core::f128::<impl f128>::powf"))
        self.assertFalse(reg.lookup("core::f16::<impl f16>::classify"))
        self.assertTrue(reg.lookup(
            "core::unicode::unicode_data::alphabetic::lookup_slow"))
        self.assertFalse(reg.lookup(
            "core::unicode::unicode_data::white_space::lookup_slow"))
        self.assertFalse(reg.lookup(
            "core::unicode::unicode_data::alphabetic::lookup"))
        # The file-I/O diversion's unconditional arms are inherent
        # names (the name fixes the receiver), served for every
        # call; its Self-gated trait arms stay unmodelled, so a
        # gated name does not resolve even though the gate's own
        # table names it.
        self.assertTrue(reg.lookup("std::fs::File::sync_all"))
        self.assertTrue(reg.lookup("std::fs::File::set_len"))
        self.assertTrue(reg.lookup("std::io::Stdin::read_line"))
        self.assertFalse(reg.lookup("std::io::Read::read_to_end"))
        self.assertFalse(reg.lookup("std::io::Write::flush"))
        # lookup_qualified strips leading segments; lookup does not.
        self.assertFalse(reg.lookup("vendor::std::io::_print"))
        self.assertTrue(reg.lookup_qualified("vendor::std::io::_print"))
        self.assertFalse(reg.lookup_qualified("std::vec::Vec::new"))

    def test_multiline_and_disjunctive_guards(self):
        src = FIXTURE_BUILTINS.replace(
            '        _ => return unicode_lookup_slow(canonical)',
            '''        _ if canonical == "std::vec::Vec::<T, A>::with_capacity"
            || canonical == "std::vec::Vec::<T, A>::with_capacity_in" => {
            Builtin::VecWithCapacity
        }
        _ if canonical.ends_with("::from_raw")
            && (canonical.starts_with("std::sync::Weak::<T>::")
                || canonical.starts_with("std::rc::Weak::<T>::")) => {
            Builtin::WeakFromRaw
        }
        _ => return unicode_lookup_slow(canonical)''')
        reg = cov.BuiltinRegistry(src)
        self.assertTrue(reg.lookup("std::vec::Vec::<T, A>::with_capacity"))
        self.assertTrue(
            reg.lookup("std::vec::Vec::<T, A>::with_capacity_in"))
        self.assertFalse(
            reg.lookup("std::vec::Vec::<T, A>::with_capacity_x"))
        self.assertTrue(reg.lookup("std::sync::Weak::<T>::from_raw"))
        self.assertTrue(reg.lookup("std::rc::Weak::<T>::from_raw"))
        self.assertFalse(reg.lookup("std::sync::Arc::<T>::from_raw"))
        self.assertFalse(reg.lookup("std::sync::Weak::<T>::into_raw"))

    def test_unknown_guard_is_an_error(self):
        bad = FIXTURE_BUILTINS.replace(
            'canonical.ends_with("::to_uppercase")',
            'canonical.matches("::to_uppercase")')
        with self.assertRaises(ValueError):
            cov.BuiltinRegistry(bad)

    def test_real_builtins_rs(self):
        src = cov.read_builtins_source(
            REPO_ROOT / "crates/sparagmos-interp/src/builtins")
        reg = cov.BuiltinRegistry(src)
        self.assertIn("std::io::_print", reg.exact)
        self.assertIn("core::fmt::rt::Argument::new_display", reg.exact)
        self.assertIn("std::iter::Iterator::next", reg.exact)
        self.assertTrue(reg.range_inclusive_new)
        self.assertTrue(reg.to_uppercase)
        # Multi-line and disjunctive guard arms are extracted
        # too: the BTree iterator entries and the Vec capacity
        # family live in arms a line-based reader drops.
        self.assertTrue(
            reg.lookup("std::collections::BTreeMap<K, V>::iter"))
        self.assertTrue(reg.lookup("std::vec::Vec::<T, A>::with_capacity"))
        self.assertTrue(
            reg.lookup("std::vec::Vec::<T, A>::extend_from_slice"))
        # Resolved only through the `lookup` fallback helpers: the
        # float inherent methods and the unicode skiplist lookups.
        self.assertTrue(reg.lookup("core::f64::<impl f64>::to_bits"))
        self.assertTrue(reg.lookup("std::f64::<impl f64>::sqrt"))
        self.assertTrue(reg.lookup("core::f32::<impl f32>::from_bits"))
        # The file-I/O diversion's unconditional arms: inherent
        # names only (the five `File` handle methods and
        # `Stdin::read_line`); the Self-gated trait arms of
        # `is_file_io_call` stay unmodelled.
        self.assertEqual(reg.file_io_names, {
            "std::fs::File::metadata",
            "std::fs::File::try_clone",
            "std::fs::File::sync_all",
            "std::fs::File::sync_data",
            "std::fs::File::set_len",
            "std::io::Stdin::read_line",
        })
        self.assertTrue(reg.lookup("std::io::Stdin::read_line"))
        self.assertFalse(reg.lookup("std::io::Write::flush"))
        self.assertFalse(reg.lookup("std::io::Read::read_to_end"))
        self.assertFalse(reg.lookup("std::io::BufRead::read_line"))
        self.assertTrue(reg.lookup("core::f16::<impl f16>::is_nan"))
        self.assertTrue(reg.lookup("core::f128::<impl f128>::from_bits"))
        self.assertTrue(reg.lookup(
            "core::unicode::unicode_data::grapheme_extend::lookup_slow"))
        # The f16 classify / byte-conversion family is served by
        # exact arms in `lookup`, registered for f16 only: they are
        # kept out of the shared float method table so the f32/f64
        # namesakes keep executing their collected bodies.
        self.assertTrue(reg.lookup("core::f16::<impl f16>::classify"))
        self.assertTrue(reg.lookup("core::f16::<impl f16>::is_normal"))
        self.assertTrue(reg.lookup("core::f16::<impl f16>::to_be_bytes"))
        self.assertTrue(reg.lookup("core::f16::<impl f16>::from_ne_bytes"))
        # The unmodelled remainder of the same families stays out:
        # the f32/f64 namesakes of the f16-only arms, f128 has no
        # powf, and white_space is not one of the modelled
        # unicode_data modules.
        self.assertFalse(reg.lookup("core::f32::<impl f32>::classify"))
        self.assertFalse(reg.lookup("core::f64::<impl f64>::to_be_bytes"))
        self.assertFalse(reg.lookup("core::f128::<impl f128>::powf"))
        self.assertFalse(reg.lookup(
            "core::unicode::unicode_data::white_space::lookup_slow"))


FIXTURE_DUMP = """sparagmos-mir-dump v0
source: fixture.rs
functions: 1
def-names:
  DefId(0:0) = fixture::main
  DefId(0:1) = std::io::_print
  DefId(0:2) = std::fmt::Arguments::<'a>::from_str
  DefId(0:3) = fixture::helper

=== fn fixture::main (args=0, blocks=3, locals=3)
Body {
    kind: Call {
        func: Constant(
            Constant {
                literal: Ty(
                    FnDef(
                        DefId {
                            krate: 0,
                            index: 2,
                        },
                        [],
                    ),
                ),
            },
        ),
        args: [],
    },
    kind: Call {
        func: Move(
            Place {
                local: Local(
                    1,
                ),
            },
        ),
        args: [
            Constant(
                Constant {
                    literal: Ty(
                        FnDef(
                            DefId {
                                krate: 0,
                                index: 3,
                            },
                            [],
                        ),
                    ),
                },
            ),
        ],
    },
    kind: TailCall {
        func: Constant(
            Constant {
                literal: Ty(
                    FnDef(
                        DefId {
                            krate: 0,
                            index: 1,
                        },
                        [],
                    ),
                ),
            },
        ),
        args: [],
    },
}
"""


class DumpParseTests(unittest.TestCase):
    def test_parse(self):
        defs, bodies, callees, anomalies = cov.parse_dump(FIXTURE_DUMP)
        self.assertEqual(defs[1], "std::io::_print")
        self.assertEqual(bodies, {"fixture::main"})
        # Direct Call and TailCall callees are found; the FnDef passed
        # as an argument of an indirect (Move) call is not a callee.
        self.assertEqual(
            callees,
            ["std::fmt::Arguments::<'a>::from_str", "std::io::_print"])
        self.assertEqual(anomalies, [])

    def test_std_path(self):
        self.assertTrue(cov.is_std_path("std::io::_print"))
        self.assertTrue(cov.is_std_path("core::panicking::panic"))
        self.assertTrue(cov.is_std_path("alloc::vec::Vec::new"))
        self.assertFalse(cov.is_std_path("fixture::main"))


class EditionTests(unittest.TestCase):
    def test_directive(self):
        self.assertEqual(
            cov.edition_of("//@ edition: 2024\nfn main() {}\n"), "2024")

    def test_default(self):
        self.assertEqual(cov.edition_of("fn main() {}\n"), "2021")

    def test_scoped_directive_ignored(self):
        self.assertEqual(
            cov.edition_of("//@[rev] edition: 2018\nfn main() {}\n"), "2021")


class AggregateTests(unittest.TestCase):
    def test_aggregate(self):
        reg = cov.BuiltinRegistry(FIXTURE_BUILTINS)
        records = [
            {"status": "ok", "std_paths": {
                "std::io::_print": {"sites": 2, "body": False},
                "std::vec::Vec::new": {"sites": 1, "body": True},
                "core::panicking::panic": {"sites": 3, "body": False}}},
            {"status": "ok", "std_paths": {
                "std::vec::Vec::new": {"sites": 1, "body": False},
                "core::panicking::panic": {"sites": 1, "body": False}}},
            {"status": "no-dump", "std_paths": {}},
        ]
        agg = cov.aggregate(records, reg)
        self.assertEqual(agg["denominator"], 3)
        # _print via builtin, Vec::new via a body in one program,
        # panic in neither.
        self.assertEqual(agg["executable"], 2)
        self.assertAlmostEqual(agg["rate"], 2 / 3)
        # Strict: Vec::new lacks a body in the second program.
        self.assertEqual(agg["executable_strict"], 1)


class CandidateTests(unittest.TestCase):
    """The survey and the subset results define one candidate set."""

    ENTRIES = [
        # (path, nature, single_file, suite, env_excluded)
        ("tests/ui/a.rs", "exec", True, "ui", False),
        ("tests/ui-fulldeps/b.rs", "exec", True, "ui-fulldeps", False),
        ("tests/ui/c/main.rs", "exec", False, "ui", False),
        ("tests/ui/d.rs", "compile-check", True, "ui", False),
        ("tests/codegen/e.rs", "exec", True, "codegen", False),
        ("tests/ui/f.rs", "exec", True, "ui", True),
    ]

    def write(self, name, data):
        import json
        import shutil
        import tempfile
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        path = d / name
        path.write_text(json.dumps(data))
        return path

    def test_survey_and_results_agree(self):
        survey = self.write("survey.json", {"tests": [
            {"path": p, "nature": n, "single_file": sf, "suite": su,
             "env_excluded": ex}
            for p, n, sf, su, ex in self.ENTRIES]})
        # env_excluded tests never reach the runner's results.
        results = self.write("results.json", {"results": [
            {"path": p, "nature": n, "single_file": sf, "suite": su}
            for p, n, sf, su, ex in self.ENTRIES if not ex]})
        expected = ["tests/ui-fulldeps/b.rs", "tests/ui/a.rs"]
        self.assertEqual(cov.stage1_candidates_from_survey(survey), expected)
        self.assertEqual(cov.stage1_candidates(results), expected)


if __name__ == "__main__":
    unittest.main()
