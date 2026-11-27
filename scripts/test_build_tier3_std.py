#!/usr/bin/env python3
"""Unit tests for scripts/build-tier3-std.py (Issue #516).

Run with: python3 -m unittest scripts.test_build_tier3_std
(or directly: python3 scripts/test_build_tier3_std.py)

The cargo build itself is not exercised (it needs the pinned
toolchain and minutes of CPU); the pure pieces are: the rlib
collection over cargo's build-std output layout, the probe
source selection per crate set, and the table's own sanity.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location(
    "build_tier3_std", Path(__file__).parent / "build-tier3-std.py")
bts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bts)


class CollectRlibsTest(unittest.TestCase):
    def _out(self, root, crate, files):
        out = root / "build" / crate / "0123abcd" / "out"
        out.mkdir(parents=True)
        for name in files:
            (out / name).write_bytes(b"")
        return out

    def test_collects_std_crates_and_skips_probe(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "aarch64v8r-unknown-none" / "release"
            self._out(root, "core", ["libcore-aaaa.rlib"])
            self._out(root, "alloc", ["liballoc-bbbb.rlib",
                                      "liballoc-bbbb.rmeta"])
            self._out(root, "compiler_builtins",
                      ["libcompiler_builtins-cccc.rlib"])
            self._out(root, bts.PROBE_CRATE, ["libprobe-dddd.rlib"])
            # A deps/ sibling is not part of the build-std layout.
            deps = root / "deps"
            deps.mkdir(parents=True)
            (deps / "libcore-eeee.rlib").write_bytes(b"")
            got = [p.name for p in
                   bts.collect_rlibs(Path(td), "aarch64v8r-unknown-none")]
            self.assertEqual(got, ["liballoc-bbbb.rlib",
                                   "libcompiler_builtins-cccc.rlib",
                                   "libcore-aaaa.rlib"])

    def test_missing_tree_collects_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(
                bts.collect_rlibs(Path(td), "avr-none"), [])


class ProbeSourceTest(unittest.TestCase):
    def test_std_set_uses_std(self):
        src = bts.probe_source(("core", "alloc", "std", "panic_unwind"))
        self.assertIn("std::", src)
        self.assertNotIn("no_std", src)

    def test_alloc_set_is_no_std_with_alloc(self):
        src = bts.probe_source(("core", "alloc"))
        self.assertIn("#![no_std]", src)
        self.assertIn("extern crate alloc;", src)

    def test_core_only_set_is_plain_no_std(self):
        src = bts.probe_source(("core",))
        self.assertIn("#![no_std]", src)
        self.assertNotIn("alloc", src)


class TableTest(unittest.TestCase):
    def test_table_is_sane(self):
        for triple, crates in bts.TIER3_STD.items():
            self.assertIn("-", triple)
            self.assertEqual(crates[0], "core")
            self.assertEqual(len(crates), len(set(crates)))

    def test_increment2_entries(self):
        self.assertEqual(bts.TIER3_STD["aarch64-unknown-linux-pauthtest"],
                         ("core", "std", "panic_unwind"))
        self.assertEqual(bts.TIER3_STD["amdgcn-amd-amdhsa"],
                         ("core", "alloc"))
        self.assertEqual(bts.TIER3_STD["avr-none"], ("core", "alloc"))
        self.assertEqual(bts.TIER3_STD["riscv32e-unknown-none-elf"],
                         ("core", "alloc"))

    def test_increment3_entries(self):
        full = ("core", "std", "panic_unwind")
        self.assertEqual(bts.TIER3_STD["hexagon-unknown-linux-musl"], full)
        self.assertEqual(bts.TIER3_STD["i686-apple-darwin"], full)
        self.assertEqual(bts.TIER3_STD["mips-unknown-linux-gnu"], full)
        self.assertEqual(
            bts.TIER3_STD["mips64-unknown-linux-gnuabi64"], full)
        self.assertEqual(bts.TIER3_STD["powerpc64-ibm-aix"], full)
        self.assertEqual(bts.TIER3_STD["sparc-unknown-linux-gnu"], full)
        self.assertEqual(bts.TIER3_STD["wasm64-unknown-unknown"], full)
        self.assertEqual(bts.TIER3_STD["bpfeb-unknown-none"],
                         ("core", "alloc"))
        self.assertEqual(bts.TIER3_STD["bpfel-unknown-none"],
                         ("core", "alloc"))
        self.assertEqual(bts.TIER3_STD["msp430-none-elf"],
                         ("core", "alloc"))
        self.assertEqual(bts.TIER3_STD["sparc-unknown-none-elf"],
                         ("core", "alloc"))

    def test_table_entry_set_is_the_union(self):
        self.assertEqual(set(bts.TIER3_STD), {
            "aarch64-unknown-linux-pauthtest",
            "aarch64v8r-unknown-none",
            "amdgcn-amd-amdhsa",
            "avr-none",
            "bpfeb-unknown-none",
            "bpfel-unknown-none",
            "hexagon-unknown-linux-musl",
            "i686-apple-darwin",
            "mips-unknown-linux-gnu",
            "mips64-unknown-linux-gnuabi64",
            "msp430-none-elf",
            "powerpc64-ibm-aix",
            "riscv32e-unknown-none-elf",
            "sparc-unknown-linux-gnu",
            "sparc-unknown-none-elf",
            "wasm64-unknown-unknown",
        })


class TargetCpuTest(unittest.TestCase):
    def test_cpu_table_only_names_table_triples(self):
        for triple in bts.TIER3_TARGET_CPU:
            self.assertIn(triple, bts.TIER3_STD)

    def test_env_scopes_rustflags_to_the_target(self):
        self.assertEqual(
            bts.target_cpu_env("amdgcn-amd-amdhsa"),
            {"CARGO_TARGET_AMDGCN_AMD_AMDHSA_RUSTFLAGS":
             "-C target-cpu=gfx900"})
        self.assertEqual(
            bts.target_cpu_env("avr-none"),
            {"CARGO_TARGET_AVR_NONE_RUSTFLAGS":
             "-C target-cpu=atmega328p"})

    def test_env_empty_without_a_pinned_cpu(self):
        self.assertEqual(bts.target_cpu_env("aarch64v8r-unknown-none"), {})
        self.assertEqual(
            bts.target_cpu_env("aarch64-unknown-linux-pauthtest"), {})
        self.assertEqual(bts.target_cpu_env("riscv32e-unknown-none-elf"),
                         {})


class InstalledCoreTest(unittest.TestCase):
    def test_installed_core_detects_libcore(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertFalse(
                bts.installed_core(root, "aarch64v8r-unknown-none"))
            lib = (root / "lib" / "rustlib" /
                   "aarch64v8r-unknown-none" / "lib")
            lib.mkdir(parents=True)
            (lib / "liballoc-aaaa.rlib").write_bytes(b"")
            self.assertFalse(
                bts.installed_core(root, "aarch64v8r-unknown-none"))
            (lib / "libcore-bbbb.rlib").write_bytes(b"")
            self.assertTrue(
                bts.installed_core(root, "aarch64v8r-unknown-none"))


if __name__ == "__main__":
    unittest.main()
