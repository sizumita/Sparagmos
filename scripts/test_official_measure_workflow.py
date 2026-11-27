#!/usr/bin/env python3
"""Consistency tests for .github/workflows/official-measure.yml.

Run with: python3 -m pytest scripts/test_official_measure_workflow.py
(or directly: python3 scripts/test_official_measure_workflow.py)

The survey source tree is produced once by the prepare job and restored
by the measure, measure-host-<class>, measure-jit and metric3 jobs
through actions/cache. actions/cache never re-saves on an exact key
hit, so the cached tree's shape (its sparse-checkout set) is part of
the cache key's meaning: every job must name one and the same key, and
widening the checkout set without bumping the key would leave the
measure jobs restoring the old tree. These tests pin the consistency
half of that contract.
"""

import importlib.util
import re
import unittest
from pathlib import Path

WORKFLOW = (Path(__file__).parent.parent / ".github" / "workflows"
            / "official-measure.yml")

# The partition tool's class table and default class list: the
# workflow's per-class host jobs must cover exactly the classes the
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


def job_blocks():
    """The workflow's jobs as an ordered {job id: block text} map.

    A job block runs from its two-space `  <id>:` header to the
    next job's header (or the end of the file). The comment lines
    preceding a header belong to the previous job's block; the
    tests below only read within one job's own block.
    """
    lines = workflow_text().splitlines()
    blocks = {}
    current = None
    for line in lines[lines.index("jobs:") + 1:]:
        m = re.match(r"^  ([a-zA-Z0-9_-]+):$", line)
        if m:
            current = m.group(1)
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line)
    return {jid: "\n".join(bl) for jid, bl in blocks.items()}


def host_classes():
    return [c for c in tools.DEFAULT_PARTITION_CLASSES
            if c != "linux-x86_64"]


def toolchain_descriptor(cls):
    """The toolchain descriptor a class's install steps name:
    the plain pinned release for the classes the runner's native
    toolchain hosts, and the gnu-hosted descriptor for
    windows-gnu, whose host model is the pinned toolchain hosted
    on the gnu target."""
    return ("1.100.0-x86_64-pc-windows-gnu"
            if cls == "windows-gnu" else "1.100.0")


class SurveyCacheKeyTest(unittest.TestCase):
    def test_all_survey_cache_keys_are_one_value(self):
        keys = re.findall(r"key:\s*(rust-survey-\S+)", workflow_text())
        # prepare (save) + measure, the five measure-host-<class>
        # jobs, measure-jit, metric3 (restore).
        self.assertEqual(len(keys), 9, keys)
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
        # for Linux x86_64, the build-host-<class> jobs for the
        # other classes; all build with the measure profile) and
        # every judging job runs its class's artifact; no job
        # builds its own driver or points at a debug build.
        text = workflow_text()
        self.assertEqual(text.count("cargo build"), 6)
        self.assertIn("cargo build --profile measure", text)
        self.assertNotIn("target/debug/sparagmos-driver", text)
        self.assertEqual(text.count('--driver-bin "$DRIVER_BIN"'), 8)

    def test_measure_profile_keeps_dev_checks(self):
        cargo = (WORKFLOW.parent.parent.parent / "crates"
                 / "sparagmos-driver" / "Cargo.toml").read_text()
        profile = cargo.split("[profile.measure]", 1)[1]
        self.assertIn('inherits = "dev"', profile)
        self.assertNotIn("debug-assertions = false", profile)
        self.assertNotIn("overflow-checks = false", profile)


class HostMatrixTest(unittest.TestCase):
    def test_host_jobs_cover_the_partition_classes(self):
        blocks = job_blocks()
        classes = host_classes()
        build_ids = [j for j in blocks if j.startswith("build-host-")]
        measure_ids = [j for j in blocks
                       if j.startswith("measure-host-")]
        # One build job and one measure job per non-residual
        # class, in the partition's priority order. The matrix
        # jobs they replace are gone: a dependent of a matrix job
        # waits for all of its legs, so one class's build failure
        # used to skip every class's measure jobs.
        self.assertEqual(build_ids,
                         [f"build-host-{c}" for c in classes])
        self.assertEqual(measure_ids,
                         [f"measure-host-{c}" for c in classes])
        text = workflow_text()
        for token in ("${{ matrix.class }}", "${{ matrix.target }}",
                      "${{ matrix.bin }}", "${{ matrix.host }}",
                      "${{ matrix.runner }}", "if: matrix.class"):
            self.assertNotIn(token, text)

    def test_each_measure_job_needs_only_its_own_build_job(self):
        blocks = job_blocks()
        for cls in host_classes():
            build = blocks[f"build-host-{cls}"]
            measure = blocks[f"measure-host-{cls}"]
            runner = re.search(r"^    runs-on: (\S+)$", build,
                               re.M).group(1)
            binname = re.search(r"target/measure/(\S+)$", build,
                                re.M).group(1)
            # The build job uploads the class's driver under the
            # artifact name the measure job downloads, and its
            # binary carries the class's name (the .exe suffix on
            # Windows).
            self.assertIn(f"name: sparagmos-driver-{cls}", build)
            self.assertIn(f"name: sparagmos-driver-{cls}", measure)
            self.assertIn(f"/.ci/driver/{binname}", measure)
            # Both jobs install the class's toolchain descriptor:
            # the gnu-hosted one for windows-gnu, the plain
            # pinned release for the rest.
            descriptor = toolchain_descriptor(cls)
            self.assertIn(f"toolchain: {descriptor}", build)
            self.assertIn(f"toolchain: {descriptor}", measure)
            # The gnu-hosted toolchain assertion step exists
            # exactly in the windows-gnu jobs: the toolchain
            # action sets a non-host default on a best-effort
            # basis, and only that class installs one.
            for block in (build, measure):
                self.assertEqual(
                    "Verify the gnu-hosted toolchain is the default"
                    in block, cls == "windows-gnu", cls)
            # The measure job runs on the class's own runner and
            # needs prepare and its own class's build job only,
            # so it runs whenever that build succeeded, whatever
            # happened to the other classes' builds.
            self.assertIn(f"    runs-on: {runner}", measure)
            self.assertIn(f"needs: [prepare, build-host-{cls}]",
                          measure)
            # Two shards over the class's own partition survey,
            # with the runner's host model selected by the
            # class's target, and result files in the shared
            # subset-shard-* namespace the aggregate merges.
            self.assertIn("shard: [0, 1]", measure)
            self.assertIn(
                f"SPARAGMOS_HOST_TARGET: {tools.HOST_CLASSES[cls]}",
                measure)
            self.assertIn(f"subset-survey-{cls}.json", measure)
            self.assertIn(f"subset-shard-{cls}-", measure)

    def test_aggregate_requires_every_host_class(self):
        # The aggregate publishes only when every class was
        # judged: a class whose build failed leaves its measure
        # job skipped, the condition below fails, and the run
        # reports incompleteness instead of silently dropping
        # the class from the denominator.
        aggregate = job_blocks()["aggregate"]
        for cls in host_classes():
            self.assertIn(f"      - measure-host-{cls}", aggregate)
            self.assertIn(
                f"needs.measure-host-{cls}.result == 'success'",
                aggregate)


if __name__ == "__main__":
    unittest.main()
