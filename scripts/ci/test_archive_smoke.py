# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Offline assertions for the archive probe; synthetic vectors are not inference proof."""

import copy
import runpy
import unittest
from pathlib import Path

PROBE = runpy.run_path(str(Path(__file__).with_name("archive_smoke.py")))


def fixture():
    return {
        "model": PROBE["MODEL"],
        "data": [{"index": i, "embedding": [1.0] + [0.0] * 383} for i in range(2)],
    }


class VectorContract(unittest.TestCase):
    def test_valid_ordered_unit_vectors(self):
        self.assertEqual(len(PROBE["vectors"](fixture())), 2)

    def test_rejects_identity_count_and_order(self):
        original = fixture()
        cases = [copy.deepcopy(original) for _ in range(3)]
        cases[0]["model"] = "wrong"
        cases[1]["data"].pop()
        cases[2]["data"].reverse()
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(AssertionError):
                PROBE["vectors"](candidate)

    def test_rejects_dimensions_nonfinite_and_nonunit(self):
        for vector in ([1.0], [float("nan")] * 384, [float("inf")] * 384, [0.0] * 384):
            candidate = fixture()
            candidate["data"][0]["embedding"] = vector
            with self.subTest(vector=vector[0]), self.assertRaises(AssertionError):
                PROBE["vectors"](candidate)


if __name__ == "__main__":
    unittest.main()
