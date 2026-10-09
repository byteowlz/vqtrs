"""Probe audited regular MLX definitions with original tensors; no processor/hub/code loading."""

import json
import os
import subprocess
import sys
import types
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

WORK = Path(os.environ.get("EG2_BENCH_WORK_DIR", "/tmp/vqtrs-native-media"))
CHECKOUT = Path(os.environ["EG2_MLX_VLM_DIR"])
assert (
    subprocess.check_output(
        ["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], text=True
    ).strip()
    == "4f4634bb813c0298cb1467bed2e957526c71d0b4"
)
subprocess.run(["git", "-C", str(CHECKOUT), "diff", "--quiet", "HEAD"], check=True)
SOURCE = CHECKOUT / "mlx_vlm"
MODEL = Path(os.environ["EG2_MODEL_DIR"])
REF = Path(os.environ.get("EG2_REFERENCE_DIR", "/tmp/eg2-parity"))
# Bypass package initializers (generation/processor imports); import only audited tensor definitions.
for name, path in [
    ("mlx_vlm", SOURCE),
    ("mlx_vlm.models", SOURCE / "models"),
    ("mlx_vlm.models.gemma4", SOURCE / "models/gemma4"),
    ("mlx_vlm.models.embedding_gemma2", SOURCE / "models/embedding_gemma2"),
    ("mlx_vlm.models.qwen3_5", SOURCE / "models/qwen3_5"),
]:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
from mlx_vlm.models.embedding_gemma2.config import ModelConfig
from mlx_vlm.models.embedding_gemma2.embedding_gemma2 import Model

mx.set_cache_limit(512 * 1024 * 1024)
mx.set_memory_limit(6 * 1024 * 1024 * 1024)
config = ModelConfig.from_dict(json.loads((MODEL / "config.json").read_text()))
model = Model(config)
weights = model.sanitize(mx.load(str(MODEL / "model.safetensors")))
expected = dict(tree_flatten(model.parameters()))
assert set(weights) == set(expected), (
    set(weights) - set(expected),
    set(expected) - set(weights),
)
assert all(
    v.shape == expected[k].shape and v.dtype == mx.bfloat16 for k, v in weights.items()
)
model.load_weights(list(weights.items()), strict=True)
mx.eval(model.parameters())
print("loaded audited original BF16 weights", flush=True)
results = []

if len(sys.argv) > 1 and sys.argv[1] == "text":
    fixture = json.loads((WORK / "retrieval-inputs.json").read_text())
    output = []
    for offset in range(0, len(fixture["rows"]), 16):
        batch = fixture["rows"][offset : offset + 16]
        width = max(len(row["ids"]) for row in batch)
        ids = np.zeros((len(batch), width), dtype=np.int32)
        mask = np.zeros_like(ids)
        for i, row in enumerate(batch):
            ids[i, : len(row["ids"])] = row["ids"]
            mask[i, : len(row["ids"])] = 1
        vector = model(mx.array(ids), attention_mask=mx.array(mask)).text_embeds.astype(
            mx.float32
        )
        vector = vector / mx.linalg.norm(vector, axis=-1, keepdims=True)
        mx.eval(vector)
        vectors = np.array(vector)
        assert vectors.shape == (len(batch), 768) and np.isfinite(vectors).all()
        output.extend(
            {"id": row["id"], "vector": value.tolist()}
            for row, value in zip(batch, vectors)
        )
    (WORK / "retrieval-mlx-regular.json").write_text(
        json.dumps({"runtime": "mlx-regular-bf16", "rows": output})
    )
    sys.exit(0)


def tensor(folder, case, key):
    t = case["tensors"][key]
    return np.fromfile(folder / t["file"], dtype=t["dtype"]).reshape(t["shape"])


def evaluate(name, inputs, expected):
    start = perf_counter()
    output = model(**inputs).text_embeds.astype(mx.float32)
    output = output / mx.linalg.norm(output, axis=-1, keepdims=True)
    mx.eval(output)
    actual = np.array(output).reshape(-1)
    assert actual.shape == (768,) and np.isfinite(actual).all()
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    result = {
        "name": name,
        "max_abs": float(np.max(np.abs(actual - expected))),
        "cosine": float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)),
        "norm": float(np.linalg.norm(a)),
        "pipeline_ms": (perf_counter() - start) * 1000,
        "vector": actual.tolist(),
    }
    results.append(result)
    print(json.dumps({k: v for k, v in result.items() if k != "vector"}), flush=True)


for kind in ["image", "audio"]:
    folder = REF / ("ref-image" if kind == "image" else "ref-audio-final")
    if not folder.exists():
        folder = folder.with_name("ref-audio")
    manifest = json.loads((folder / "manifest.json").read_text())
    keys = ["input_ids", "attention_mask"] + (
        ["pixel_values", "image_position_ids"]
        if kind == "image"
        else ["input_features", "input_features_mask"]
    )
    for case in manifest["cases"]:
        inputs = {key: mx.array(tensor(folder, case, "input." + key)) for key in keys}
        evaluate(case["name"], inputs, tensor(folder, case, "hf.embedding"))


# Keep vision frame activations bounded; regular MLX media path, not turbo/W8A8.
def sequential_video(pixels, positions):
    features = []
    for i in range(pixels.shape[0]):
        feature = model.get_image_features(pixels[i : i + 1], positions[i : i + 1])
        mx.eval(feature)
        features.append(feature)
    return mx.concatenate(features, axis=0)


model.get_video_features = sequential_video
folder = Path(os.environ.get("EG2_VIDEO_REFERENCE_DIR", "/tmp/eg2-video-ref-140"))
manifest = json.loads((folder / "manifest.json").read_text())
for case in manifest["cases"]:
    if "model_tensors" not in case:
        continue
    n = len(case["sampled_indices"])
    inputs = {
        "input_ids": mx.array([case["input_ids"]]),
        "attention_mask": mx.ones((1, len(case["input_ids"])), dtype=mx.int32),
        "pixel_values_videos": mx.array(
            np.fromfile(folder / case["pixels_file"], dtype="<f4").reshape(n, 1260, 768)
        ),
        "video_position_ids": mx.array(
            np.fromfile(folder / case["positions_file"], dtype="<i8").reshape(
                n, 1260, 2
            )
        ),
    }
    expected = np.fromfile(
        folder / case["model_tensors"]["embedding"]["file"], dtype="<f4"
    )
    evaluate(case["name"], inputs, expected)
(WORK / "mlx-regular-results.json").write_text(
    json.dumps(
        {
            "runtime": "MLX regular; BF16 weights, source-defined activation promotion, F32 output renormalization",
            "source_revision": "4f4634bb813c0298cb1467bed2e957526c71d0b4",
            "original_revision": MODEL.name,
            "preprocessing": "unchanged F32 oracle tensors; sequential video encoder",
            "allocator_peak_bytes": mx.get_peak_memory(),
            "cases": results,
        },
        indent=2,
    )
    + "\n"
)
