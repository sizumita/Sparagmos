#!/usr/bin/env python3
"""Consistency tests for .github/workflows/official-measure.yml.

Run with: python3 -m pytest scripts/test_official_measure_workflow.py
(or directly: python3 scripts/test_official_measure_workflow.py)

The survey source tree is produced once by the prepare job and restored
by the measure, measure-jit and aggregate jobs through actions/cache.
actions/cache never re-saves on an exact key hit, so the cached tree's
shape (its sparse-checkout set) is part of the cache key's meaning:
every job must name one and the same key, and widening the checkout
set without bumping the key would leave the measure jobs restoring
the old tree. These tests pin the consistency half of that contract.
"""

import re
import unittest
from pathlib import Path

WORKFLOW = (Path(__file__).parent.parent / ".github" / "workflows"
            / "official-measure.yml")


def workflow_text():
    return WORKFLOW.read_text()


class SurveyCacheKeyTest(unittest.TestCase):
    def test_all_survey_cache_keys_are_one_value(self):
        keys = re.findall(r"key:\s*(rust-survey-\S+)", workflow_text())
        # prepare (save) + measure, measure-jit, aggregate (restore).
        self.assertEqual(len(keys), 4, keys)
        self.assertEqual(len(set(keys)), 1, keys)

    def test_prepare_widens_checkout_for_stage3_oracles(self):
        # src/etc carries htmldocck.py, executed by run-make-support's
        # htmldocck() helper from the source root, and the CSS
        # directory is read directly by the rustdoc themes recipe.
        # Both must stay in the cached tree's sparse-checkout set.
        text = workflow_text()
        self.assertIn("src/etc", text)
        self.assertIn("src/librustdoc/html/static/css", text)


if __name__ == "__main__":
    unittest.main()
