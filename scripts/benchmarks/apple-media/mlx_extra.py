import json
import os
import runpy
from pathlib import Path

import numpy as np

ns = runpy.run_path(str(Path(__file__).with_name("mlx_regular_probe.py")))
mx = ns["mx"]
model = ns["model"]
tensor = ns["tensor"]
root = Path(os.environ.get("EG2_BENCH_WORK_DIR", "/tmp/vqtrs-native-media"))


def encode(ids, mask, features=None, featuremask=None):
    kw = (
        {}
        if features is None
        else {
            "input_features": mx.array(features),
            "input_features_mask": mx.array(featuremask),
        }
    )
    v = model(mx.array(ids), attention_mask=mx.array(mask), **kw).text_embeds.astype(
        mx.float32
    )
    v = v / mx.linalg.norm(v, axis=-1, keepdims=True)
    mx.eval(v)
    return np.array(v)


rows = json.loads((root / "retrieval-inputs.json").read_text())["rows"][:4]
width = max(len(row["ids"]) for row in rows)
ids = np.zeros((4, width), dtype=np.int32)
mask = np.zeros_like(ids)
for i, row in enumerate(rows):
    ids[i, : len(row["ids"])] = row["ids"]
    mask[i, : len(row["ids"])] = 1
batch = encode(ids, mask)
reordered = encode(ids[::-1].copy(), mask[::-1].copy())[::-1]
singles = np.array(
    [
        encode([row["ids"]], np.ones((1, len(row["ids"])), dtype=np.int32))[0]
        for row in rows
    ]
)
report = {
    "batch_reorder_max_abs": float(abs(batch - reordered).max()),
    "ragged_batch_vs_single_max_abs": float(abs(batch - singles).max()),
}
assert report["batch_reorder_max_abs"] < 0.002
folder = (
    Path(os.environ.get("EG2_REFERENCE_DIR", "/tmp/eg2-parity")) / "ref-audio-final"
)
case = next(
    c
    for c in json.loads((folder / "manifest.json").read_text())["cases"]
    if c["name"] == "a_short_300ms"
)
x = tensor(folder, case, "input.input_features")
m = tensor(folder, case, "input.input_features_mask")
aids = tensor(folder, case, "input.input_ids")
amask = tensor(folder, case, "input.attention_mask")
original = encode(aids, amask, x, m)[0]
tests = []
for value in [0.0, float(np.log(0.001))]:
    pad = np.full((1, 1000, 128), value, dtype=np.float32)
    pad[:, : x.shape[1]] = x
    pm = np.zeros((1, 1000), dtype=np.int32)
    pm[:, : m.shape[1]] = m
    padded = encode(aids, amask, pad, pm)[0]
    a, b = original.astype(np.float64), padded.astype(np.float64)
    tests.append(
        {
            "pad_value": value,
            "input_frames": x.shape[1],
            "padded_frames": 1000,
            "max_abs_vs_unpadded": float(abs(original - padded).max()),
            "cosine_vs_unpadded": float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)),
        }
    )
report["audio_padding_diagnostic"] = tests
(root / "mlx-extra-checks.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
