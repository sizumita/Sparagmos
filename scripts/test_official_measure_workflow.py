#!/usr/bin/env python3
"""Consistency tests for .github/workflows/official-measure.yml.

Run with: python3 -m pytest scripts/test_official_measure_workflow.py
(or directly: python3 scripts/test_official_measure_workflow.py)

The survey source tree is produced once by the prepare job and restored
by the measure, measure-host, measure-jit and metric3 jobs through
actions/cache.
actions/cache never re-saves on an exact key hit, so the cached tree's
shape (its sparse-checkout set) is part of the cache key's meaning:
every job must name one and the same key, and widening the checkout
set without bumping the key would leave the measure jobs restoring
the old tree. These tests pin the consistency half of that contract.
"""

import importlib.util
import re
import unittest
from pathlib import Path

WORKFLOW = (Path(__file__).parent.parent / ".github" / "workflows"
            / "official-measure.yml")

# The partition tool's class table and default class list: the
# workflow's host matrices must cover exactly the classes the
# prepare job partitions into (beyond the Linux residual class,
# which the build/measure jobs judge), or a partitioned class
# would have no jobs and its records would never reach the
# aggregate.
_spec = importlib.util.spec_from_file_location(
    "ci_subset_tools", Path(__file__).parent / "ci-subset-tools.py")
tools = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tools)


def workflow_text():
    return WORKFLOW.read_text()


class SurveyCacheKeyTest(unittest.TestCase):
    def test_all_survey_cache_keys_are_one_value(self):
        keys = re.findall(r"key:\s*(rust-survey-\S+)", workflow_text())
        # prepare (save) + measure, measure-host, measure-jit,
        # metric3 (restore).
        self.assertEqual(len(keys), 5, keys)
        self.assertEqual(len(set(keys)), 1, keys)

    def test_prepare_widens_checkout_for_stage3_oracles(self):
        # src/etc carries htmldocck.py, executed by run-make-support's
        # htmldocck() helper from the source root, and the CSS
        # directory is read directly by the rustdoc themes recipe.
        # Both must stay in the cached tree's sparse-checkout set.
        text = workflow_text()
        self.assertIn("src/etc", text)
        self.assertIn("src/librustdoc/html/static/css", text)


class DriverArtifactTest(unittest.TestCase):
    def test_jobs_run_the_shared_measure_profile_driver(self):
        # The driver is built once per host class (the build job
        # for Linux x86_64, the build-host job for the other
        # classes; both build with the measure profile) and every
        # judging job runs its class's artifact; no job builds its
        # own driver or points at a debug build.
        text = workflow_text()
        self.assertEqual(text.count("cargo build"), 2)
        self.assertIn("cargo build --profile measure", text)
        self.assertNotIn("target/debug/sparagmos-driver", text)
        self.assertEqual(text.count('--driver-bin "$DRIVER_BIN"'), 4)

    def test_measure_profile_keeps_dev_checks(self):
        cargo = (WORKFLOW.parent.parent.parent / "crates"
                 / "sparagmos-driver" / "Cargo.toml").read_text()
        profile = cargo.split("[profile.measure]", 1)[1]
        self.assertIn('inherits = "dev"', profile)
        self.assertNotIn("debug-assertions = false", profile)
        self.assertNotIn("overflow-checks = false", profile)


def matrix_includes():
    """The workflow's matrix include entries, in file order.

    Each entry is the mapping its `- class:` line opens; the
    target/runner/bin/shard lines that follow belong to it until the
    next entry or a non-entry line ends it. Entries without a
    target are the build-host matrix's; entries with one are the
    measure-host matrix's."""
    entries = []
    current = None
    for line in workflow_text().splitlines():
        m = re.match(r"\s*- class: (\S+)$", line)
        if m:
            current = {"class": m.group(1)}
            entries.append(current)
            continue
        m = re.match(r"\s+(target|runner|bin|shard): (\S+)$", line)
        if m and current is not None:
            current[m.group(1)] = m.group(2)
        elif line.strip() and not line.strip().startswith("#"):
            current = None
    return entries


class HostMatrixTest(unittest.TestCase):
    def test_host_matrices_cover_the_partition_classes(self):
        entries = matrix_includes()
        host_classes = [c for c in tools.DEFAULT_PARTITION_CLASSES
                        if c != "linux-x86_64"]
        build = [e for e in entries if "target" not in e]
        measure = [e for e in entries if "target" in e]
        # One build-host entry per non-residual class, in the
        # partition's priority order.
        self.assertEqual([e["class"] for e in build], host_classes)
        # Two measure-host shards per class, on the class's own
        # build runner, selecting the class's host target through
        # SPARAGMOS_HOST_TARGET.
        for cls in host_classes:
            shards = [e for e in measure if e["class"] == cls]
            self.assertEqual({e["shard"] for e in shards},
                             {"0", "1"}, cls)
            runner = next(e["runner"] for e in build
                          if e["class"] == cls)
            for e in shards:
                self.assertEqual(e["target"],
                                 tools.HOST_CLASSES[cls], cls)
                self.assertEqual(e["runner"], runner, cls)


if __name__ == "__main__":
    unittest.main()
