import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

root = Path(os.environ.get("EG2_BENCH_WORK_DIR", "/tmp/vqtrs-native-media"))
manifest = json.loads((root / "retrieval-manifest.json").read_text())
fixture = json.loads((root / "retrieval-inputs.json").read_text())
ids = [row["id"] for row in fixture["rows"]]


def read(name):
    data = json.loads((root / name).read_text())
    assert [r["id"] for r in data["rows"]] == ids
    return {
        row["id"]: np.array(row["vector"], dtype=np.float64) for row in data["rows"]
    }


base = read("retrieval-metal.json")
spot = json.loads((root / "retrieval-cpu-spot-results.json").read_text())
max_abs = 0.0
min_cos = 1.0
for row in spot["rows"]:
    a, b = np.array(row["vector"], dtype=np.float64), base[row["id"]]
    max_abs = max(max_abs, float(abs(a - b).max()))
    min_cos = min(min_cos, float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
assert max_abs <= 2e-4 and min_cos >= 0.99999


def evaluate(vectors, dataset):
    name = dataset["dataset"]
    docs = dataset["documents"]
    qids = dataset["queries"]
    matrix = np.array([vectors[name + ":doc:" + d] for d in docs])
    matrix /= np.linalg.norm(matrix, axis=-1, keepdims=True)
    results = []
    rankings = {}
    for q in qids:
        v = vectors[name + ":query:" + q]
        scores = matrix @ (v / np.linalg.norm(v))
        order = sorted(range(len(docs)), key=lambda i: (-scores[i], docs[i]))
        ranked = [docs[i] for i in order]
        rankings[q] = ranked[:10]
        labels = dataset["labels"][q]
        gains = [
            (2 ** labels.get(d, 0) - 1) / math.log2(i + 2)
            for i, d in enumerate(ranked[:10])
        ]
        ideal = sorted(labels.values(), reverse=True)[:10]
        idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
        results.append(
            {
                "ndcg10": sum(gains) / idcg,
                "recall10": sum(d in labels for d in ranked[:10]) / len(labels),
                "mrr10": next(
                    (1 / (i + 1) for i, d in enumerate(ranked[:10]) if d in labels), 0
                ),
            }
        )
    return {
        key: float(np.mean([r[key] for r in results])) for key in results[0]
    }, rankings


report = {
    "scope": "bounded experimental gate; not full BEIR qualification",
    "rows": len(ids),
    "input_fixture_sha256": hashlib.sha256(
        (root / "retrieval-inputs.json").read_bytes()
    ).hexdigest(),
    "common_context": "explicitly prepared <=512 tokens, no native backend truncation",
    "dimensions": 768,
    "baseline": "Candle Metal F32 with16 dataset-derived CPU F32 spot checks",
    "f32_spot": {
        "rows": len(spot["rows"]),
        "max_abs": max_abs,
        "min_cosine": min_cos,
        "unchanged_gate_passed": True,
    },
    "datasets": [],
}
arms = {"candle-metal-f32": base}
for arm in ["mlx-regular", "coreml-single", "coreml-packed"]:
    arms[arm] = read("retrieval-" + arm + ".json")
for d in manifest["datasets"]:
    summary = {
        k: d[k]
        for k in [
            "dataset",
            "archive_sha256",
            "explicitly_capped_rows",
            "scope",
            "context",
        ]
    }
    summary.update(queries=len(d["queries"]), documents=len(d["documents"]), arms={})
    baseline_metrics, baseline_ranks = evaluate(base, d)
    for name, vectors in arms.items():
        metrics, ranks = evaluate(vectors, d)
        overlap = float(
            np.mean(
                [len(set(ranks[q]) & set(baseline_ranks[q])) / 10 for q in d["queries"]]
            )
        )
        keys = [d["dataset"] + ":query:" + q for q in d["queries"]] + [
            d["dataset"] + ":doc:" + x for x in d["documents"]
        ]
        cosine = []
        absdiff = []
        normerr = []
        for key in keys:
            a, b = vectors[key], base[key]
            cosine.append(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
            absdiff.append(float(abs(a - b).max()))
            normerr.append(abs(np.linalg.norm(a) - 1))
        summary["arms"][name] = {
            **metrics,
            "top10_overlap_vs_f32": overlap,
            "max_abs_vs_f32": max(absdiff),
            "min_cosine_vs_f32": min(cosine),
            "max_norm_error": float(max(normerr)),
            "metric_delta_vs_f32": {
                k: metrics[k] - baseline_metrics[k] for k in metrics
            },
        }
    report["datasets"].append(summary)
(root / "retrieval-summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
