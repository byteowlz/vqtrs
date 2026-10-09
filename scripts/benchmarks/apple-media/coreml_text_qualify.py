import json
import os
import runpy
from pathlib import Path

import coremltools as ct
import numpy as np

WORK = Path(os.environ.get("EG2_BENCH_WORK_DIR", "/tmp/vqtrs-native-media"))
ROOT = Path(os.environ.get("EG2_COREML_DIR", str(WORK / "coreml")))
ASSETS = runpy.run_path(str(Path(__file__).with_name("assets.py")))
ASSETS["verify"](ROOT)
fixture = json.loads((WORK / "retrieval-inputs.json").read_text())
table = np.memmap(ROOT / "embeddings.bf16", dtype="<u2", mode="r", shape=(262144, 512))
models = {}


def embeds(ids):
    return ASSETS["token_rows"](table, ids)


def get(name):
    if name not in models:
        models[name] = ct.models.MLModel(
            str(ROOT / "EmbeddingGemma2Text.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            function_name=name,
        )
        print("loaded", name, flush=True)
    return models[name]


def single(ids):
    n = len(ids)
    s = next(s for s in [32, 48, 64, 128, 256, 512] if s >= n)
    x = np.zeros((1, s, 512), dtype=np.float16)
    x[0, :n] = embeds(ids)
    mask = np.zeros((1, s), dtype=np.float16)
    mask[0, :n] = 1
    return get(f"embed_{s}").predict({"inputs_embeds": x, "attention_mask": mask})[
        "embedding"
    ][0]


def packed(sequences):
    assert 0 < len(sequences) <= 8 and sum(map(len, sequences)) <= 256
    x = np.zeros((1, 256, 512), dtype=np.float16)
    bias = np.full((1, 1, 256, 256), -10000, dtype=np.float16)
    positions = np.zeros((256, 1), dtype=np.float16)
    pool = np.zeros((8, 256), dtype=np.float16)
    offset = 0
    for slot, ids in enumerate(sequences):
        n = len(ids)
        x[0, offset : offset + n] = embeds(ids)
        bias[0, 0, offset : offset + n, offset : offset + n] = 0
        positions[offset : offset + n, 0] = np.arange(n, dtype=np.float16)
        pool[slot, offset : offset + n] = np.float16(1 / n)
        offset += n
    for p in range(offset, 256):
        bias[0, 0, p, p] = 0
    return get("pack_256").predict(
        {
            "inputs_embeds": x,
            "attention_bias": bias,
            "positions": positions,
            "pool": pool,
        }
    )["embedding"][: len(sequences)]


for mode in ["single", "packed"]:
    output = []
    batch = []
    used = 0

    def save(batch, vectors, destination=output):
        vectors = np.asarray(vectors, dtype=np.float32)
        assert vectors.shape == (len(batch), 768) and np.isfinite(vectors).all()
        vectors = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
        destination.extend(
            {"id": row["id"], "vector": v.tolist()} for row, v in zip(batch, vectors)
        )

    for row in fixture["rows"]:
        if mode == "single" or len(row["ids"]) > 256:
            if batch:
                save(batch, packed([r["ids"] for r in batch]))
                batch = []
                used = 0
            save([row], [single(row["ids"])])
            continue
        if batch and (len(batch) == 8 or used + len(row["ids"]) > 256):
            save(batch, packed([r["ids"] for r in batch]))
            batch = []
            used = 0
        batch.append(row)
        used += len(row["ids"])
    if batch:
        save(batch, packed([r["ids"] for r in batch]))
    (WORK / f"retrieval-coreml-{mode}.json").write_text(
        json.dumps({"runtime": f"coreml-fp16-{mode}-f32-renorm", "rows": output})
    )
# Explicit bucket boundaries, neighbour/order checks on short variable-length query sequences.
checks = []
for n in [
    2,
    31,
    32,
    33,
    47,
    48,
    49,
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
]:
    vector = single([2] + [132] * (n - 2) + [1])
    assert np.isfinite(vector).all() and vector.shape == (768,)
    checks.append({"tokens": n, "status": "finite768"})
try:
    embeds([2] * 513)
except AssertionError:
    checks.append({"tokens": 513, "status": "explicit-overflow"})
else:
    raise AssertionError("accepted overflow")
sequences = [r["ids"] for r in fixture["rows"][:4]]
assert sum(map(len, sequences)) <= 256
singles = np.array([single(ids) for ids in sequences])
packs = np.array(packed(sequences))
reordered = np.array(packed(sequences[::-1]))[::-1]
checks.append(
    {"case": "pack-vs-single", "max_abs": float(np.max(abs(singles - packs)))}
)
checks.append(
    {"case": "reordered-neighbours", "max_abs": float(np.max(abs(packs - reordered)))}
)
assert np.max(abs(packs - reordered)) < 0.002
(WORK / "coreml-text-checks.json").write_text(json.dumps(checks, indent=2))
