#!/usr/bin/env python3
"""Unit tests for scripts/variant-matrix.py.

Run: python3 -m unittest scripts.test_variant_matrix
(the parsing tests read the real METRIC1_* tables in
crates/sparagmos-mir/src/coverage.rs and the sparagmos-mir enum
sources, so they also guard the denominator against silent drift.)
"""

import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location(
    "variant_matrix", os.path.join(HERE, "variant-matrix.py")
)
vm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vm)


def load_metric1_tables():
    """(denominator, unreachable) parsed from the METRIC1_* tables
    in crates/sparagmos-mir/src/coverage.rs."""
    path = os.path.join(REPO, "crates", "sparagmos-mir", "src", "coverage.rs")
    with open(path) as f:
        return vm.parse_metric1_tables(f.read())


class DenominatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.denom, _unreach = load_metric1_tables()

    def test_total_is_40(self):
        self.assertEqual(sum(len(v) for v in self.denom.values()), 40)

    def test_per_enum_counts(self):
        self.assertEqual(len(self.denom["StatementKind"]), 12)
        self.assertEqual(len(self.denom["TerminatorKind"]), 15)
        self.assertEqual(len(self.denom["Rvalue"]), 13)

    def test_spot_members(self):
        self.assertIn("BackwardIncompatibleDropHint", self.denom["StatementKind"])
        self.assertIn("UnwindResume", self.denom["TerminatorKind"])
        self.assertIn("WrapUnsafeBinder", self.denom["Rvalue"])
        self.assertNotIn("Retag", self.denom["StatementKind"])


class EnumParseTest(unittest.TestCase):
    def parse(self, rel, enum):
        with open(os.path.join(REPO, rel)) as f:
            return vm.parse_enum_variants(f.read(), enum)

    def test_statement_kind(self):
        names = self.parse("crates/sparagmos-mir/src/statement.rs", "StatementKind")
        self.assertIn("Retag", names)  # sparagmos-mir-only variant
        self.assertNotIn("BackwardIncompatibleDropHint", names)
        self.assertEqual(len(names), 12)

    def test_terminator_kind_struct_variants(self):
        names = self.parse("crates/sparagmos-mir/src/terminator.rs", "TerminatorKind")
        self.assertEqual(len(names), 15)
        for n in ("Goto", "SwitchInt", "Resume", "Terminate", "TailCall", "InlineAsm"):
            self.assertIn(n, names)

    def test_rvalue(self):
        names = self.parse("crates/sparagmos-mir/src/rvalue.rs", "Rvalue")
        self.assertEqual(len(names), 16)
        self.assertIn("CheckedBinaryOp", names)  # sparagmos-mir-only
        self.assertIn("Reborrow", names)


class MappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vm.DENOMINATOR, _unreach = load_metric1_tables()

    def test_renamed_terminators(self):
        self.assertEqual(vm.sm_to_denominator("TerminatorKind", "Resume"), "UnwindResume")
        self.assertEqual(vm.sm_to_denominator("TerminatorKind", "Terminate"), "UnwindTerminate")

    def test_identity_and_extras(self):
        self.assertEqual(vm.sm_to_denominator("Rvalue", "BinaryOp"), "BinaryOp")
        self.assertIsNone(vm.sm_to_denominator("Rvalue", "CheckedBinaryOp"))
        self.assertIsNone(vm.sm_to_denominator("StatementKind", "Retag"))


class StripRustTest(unittest.TestCase):
    def test_comments_and_strings_removed(self):
        src = 'let a = "Rvalue::BinaryOp"; // TerminatorKind::Call\n/* x */ 1'
        out = vm.strip_rust(src)
        self.assertNotIn("BinaryOp", out)
        self.assertNotIn("Call", out)
        self.assertIn("1", out)

    def test_lifetimes_survive(self):
        out = vm.strip_rust("fn f<'a>(x: &'a u8) -> &'a u8 { x }")
        self.assertIn("&'a u8", out)


class UnreachableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.denom, cls.unreach = load_metric1_tables()

    def test_nine_unreachable(self):
        self.assertEqual(len(vm.UNREACHABLE), 9)

    def test_unreachable_matches_coverage_rs(self):
        # The tool's UNREACHABLE set and the METRIC1_UNREACHABLE_*
        # tables in coverage.rs are two tabulations of the same
        # fact; they must not drift apart.
        self.assertEqual(vm.UNREACHABLE, self.unreach)

    def test_unreachable_are_denominator_members(self):
        for enum, name in vm.UNREACHABLE:
            self.assertIn(name, self.denom[enum])

    def test_reachable_remainder_is_31(self):
        total = sum(len(v) for v in self.denom.values())
        self.assertEqual(total - len(vm.UNREACHABLE), 31)

    def test_spot_members(self):
        self.assertIn(("StatementKind", "BackwardIncompatibleDropHint"), vm.UNREACHABLE)
        self.assertIn(("TerminatorKind", "Yield"), vm.UNREACHABLE)
        self.assertIn(("TerminatorKind", "CoroutineDrop"), vm.UNREACHABLE)
        self.assertIn(("TerminatorKind", "TailCall"), vm.UNREACHABLE)
        self.assertIn(("TerminatorKind", "FalseEdge"), vm.UNREACHABLE)
        self.assertIn(("TerminatorKind", "FalseUnwind"), vm.UNREACHABLE)
        self.assertIn(("TerminatorKind", "UnwindResume"), vm.UNREACHABLE)
        self.assertIn(("TerminatorKind", "UnwindTerminate"), vm.UNREACHABLE)
        self.assertIn(("Rvalue", "WrapUnsafeBinder"), vm.UNREACHABLE)
        # InlineAsm IS reachable through Stable MIR (no-op class).
        self.assertNotIn(("TerminatorKind", "InlineAsm"), vm.UNREACHABLE)


if __name__ == "__main__":
    unittest.main()
