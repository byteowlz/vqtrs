"""Pure CPU asset/type checks, no model loading or network."""

import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

MODULE = runpy.run_path(str(Path(__file__).with_name("assets.py")))


class AssetsTests(unittest.TestCase):
    def test_checks_bytes_and_rejects_traversal(self):
        data = b"fixture"
        manifest = {
            "revision": "fixture-only",
            "files": {
                "file.bin": {
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "file.bin").write_bytes(data)
            with patch.object(Path, "read_text", return_value=json.dumps(manifest)):
                self.assertEqual(MODULE["verify"](root), "fixture-only")
                (root / "file.bin").write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    MODULE["verify"](root)
            manifest["files"] = {"../escape": {"size": 0, "sha256": "unused"}}
            with (
                patch.object(Path, "read_text", return_value=json.dumps(manifest)),
                self.assertRaises(ValueError),
            ):
                MODULE["verify"](root)

    def test_bf16_scale_and_explicit_fp16_rounding(self):
        table = np.full((2, 512), 0x3F80, dtype=np.uint16)
        table[1] = 0x4000
        values = MODULE["token_rows"](table, [0, 1])
        self.assertEqual(values.dtype, np.dtype("float16"))
        self.assertTrue(np.all(values[0] == np.float16(np.sqrt(np.float32(512)))))
        self.assertTrue(np.all(values[1] == np.float16(2 * np.sqrt(np.float32(512)))))
        for ids in [[], [-1], [262144], [0] * 513]:
            with self.assertRaises(AssertionError):
                MODULE["token_rows"](table, ids)


if __name__ == "__main__":
    unittest.main()
