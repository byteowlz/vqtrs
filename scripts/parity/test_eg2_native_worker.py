"""Offline regressions for native asset/tensor/output validation; no framework inference."""

import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "crates/vqtrs-core/src/embedding_gemma2/native_worker.py"
)
spec = importlib.util.spec_from_file_location("native_worker", SOURCE)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class Validation(unittest.TestCase):
    def test_hash_check_supports_hf_cache_symlinks_but_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blob = root / "blob"
            blob.write_bytes(b"pinned")
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "data").symlink_to(blob)
            manifest = {
                "files": {
                    "data": {"size": 6, "sha256": hashlib.sha256(b"pinned").hexdigest()}
                }
            }
            worker.verify(snapshot, manifest)
            blob.write_bytes(b"edited")
            with self.assertRaises(ValueError):
                worker.verify(snapshot, manifest)
            manifest["files"] = {"../blob": {"size": 6, "sha256": "wrong"}}
            with self.assertRaises(ValueError):
                worker.verify(snapshot, manifest)

    def test_tensor_shape_private_path_and_finiteness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = {"directory": str(root)}
            np.array([1, 2], dtype="<f4").tofile(root / "pixels")
            self.assertEqual(worker.tensor(item, "pixels", (2,)).tolist(), [1, 2])
            for shape in [(3,), (0,), (32 * 1260 * 768 + 1,)]:
                with self.assertRaises(ValueError):
                    worker.tensor(item, "pixels", shape)
            np.array([np.nan], dtype="<f4").tofile(root / "pixels")
            with self.assertRaises(ValueError):
                worker.tensor(item, "pixels", (1,))
            with self.assertRaises((ValueError, OSError)):
                worker.tensor(item, "../missing", (1,))

    def test_owner_pipe_closure_terminates_worker_without_framework_inference(self):
        code = f"import importlib.util;s=importlib.util.spec_from_file_location('native_worker',{str(SOURCE)!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.watch_owner()"
        process = subprocess.Popen(
            ["uv", "run", "--no-project", sys.executable, "-I", "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            process.stdin.close()
            self.assertEqual(process.wait(timeout=15), 1)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    def test_explicit_normalization_rejects_invalid_output(self):
        values = np.ones((2, 768), dtype=np.float32)
        actual = np.array(worker.normalize(values))
        self.assertTrue(np.allclose(np.linalg.norm(actual, axis=1), 1, atol=2e-6))
        for values in [
            np.zeros((1, 768)),
            np.ones((1, 767)),
            np.full((1, 768), np.nan),
            np.full((1, 768), np.inf),
        ]:
            with self.assertRaises(ValueError):
                worker.normalize(values)


if __name__ == "__main__":
    unittest.main()
