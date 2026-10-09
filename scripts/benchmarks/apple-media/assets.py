"""Verify the published data files; never import or execute checkpoint code."""

import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads(Path(__file__).with_name("assets-3f978ddf.json").read_text())
    for relative, expected in manifest["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe asset path")
        path = root / path
        if path.stat().st_size != expected["size"]:
            raise ValueError(f"asset size mismatch: {relative}")
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != expected["sha256"]:
            raise ValueError(f"asset hash mismatch: {relative}")
    return manifest["revision"]


def token_rows(table, ids):
    """Gather valid original BF16 rows, scale in F32 and explicitly round to FP16."""
    import numpy as np

    assert 0 < len(ids) <= 512 and all(0 <= x < 262144 for x in ids)
    values = np.asarray(table[ids], dtype=np.uint32) << 16
    return (values.view(np.float32) * np.float32(np.sqrt(np.float32(512)))).astype(
        np.float16
    )
