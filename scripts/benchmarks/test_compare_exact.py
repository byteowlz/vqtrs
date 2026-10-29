"""Offline regression tests for the fail-closed differential gate."""

import copy
import unittest

from compare_exact import compare


def report():
    return {"model": "test", "backend": "CPU", "cases": [
        {"id": "one", "rows": 1, "ms": [1.0, 2.0, 3.0],
         "output": {"tokens": 4, "bits": [[1065353216, 0]]}}]}


class ComparisonTests(unittest.TestCase):
    def test_exact_outputs_and_median(self):
        before = report()
        after = copy.deepcopy(before)
        after["cases"][0]["ms"] = [0.5, 1.0, 1.5]
        self.assertEqual(compare([before], [after])["speedup"], 2.0)

    def test_rejects_changed_bit_token_shape_identity_or_workload(self):
        for change in ("bit", "tokens", "rows", "model", "backend", "id", "missing", "duplicate"):
            with self.subTest(change=change):
                after = report()
                if change == "bit":
                    after["cases"][0]["output"]["bits"][0][0] += 1
                elif change == "tokens":
                    after["cases"][0]["output"]["tokens"] += 1
                elif change == "rows":
                    after["cases"][0]["rows"] += 1
                elif change in ("model", "backend"):
                    after[change] = "other"
                elif change == "id":
                    after["cases"][0]["id"] = "other"
                elif change == "missing":
                    after["cases"].clear()
                else:
                    after["cases"].append(copy.deepcopy(after["cases"][0]))
                with self.assertRaises(ValueError):
                    compare([report()], [after])

    def test_rejects_invalid_timing(self):
        for times in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.subTest(times=times):
                after = report()
                after["cases"][0]["ms"] = times
                with self.assertRaises(ValueError):
                    compare([report()], [after])

    def test_rejects_nondeterminism_between_repeats(self):
        changed = report()
        changed["cases"][0]["output"]["bits"][0][0] += 1
        with self.assertRaises(ValueError):
            compare([report(), changed], [report()])


if __name__ == "__main__":
    unittest.main()
