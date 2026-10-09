import json, sys, time, os
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.environ["EG2_MLX_CANDIDATE_DIR"])
from embeddinggemma_mlx.turbo.model import build_state, resolve_preset
from embeddinggemma_mlx.turbo.engine import (
    TurboTextEncoder,
    state_path,
    CACHE_FORMAT,
    _checkpoint_identity,
)

root = Path(os.environ.get("EG2_BENCH_WORK_DIR", "/tmp/vqtrs-apple-bench"))
checkpoint = Path(os.environ["EG2_MODEL_DIR"])
cache = root / "mlx-cache"
precision = "bf16"
kinds = resolve_preset(precision)
start = time.perf_counter()
path = state_path(cache, precision)
if not path.exists():
    raw = mx.load(str(checkpoint / "model.safetensors"))
    weights = {
        k.removeprefix("language_model."): v
        for k, v in raw.items()
        if k.startswith("language_model.")
    }
    assert all(v.dtype == mx.bfloat16 for v in weights.values())
    tree = {}
    for k, v in weights.items():
        node = tree
        parts = k.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = v

    def namespace(tree):
        if isinstance(tree, dict):
            if all(k.isdigit() for k in tree):
                return [namespace(tree[str(i)]) for i in range(len(tree))]
            return NS(**{k: namespace(v) for k, v in tree.items()})
        return tree

    source = namespace(tree)
    source.config = NS(
        **json.loads((checkpoint / "config.json").read_text())["text_config"]
    )
    state = build_state(source, kinds)
    mx.eval(state)
    cache.mkdir(exist_ok=True)
    mx.save_safetensors(
        str(path),
        state,
        metadata={
            "identity": json.dumps(
                {
                    **_checkpoint_identity(checkpoint),
                    "format": CACHE_FORMAT,
                    "kinds": kinds,
                }
            )
        },
    )
    del source, state, weights, raw, tree
    mx.clear_cache()
conversion_ms = (time.perf_counter() - start) * 1000
start = time.perf_counter()
encoder = TurboTextEncoder(
    checkpoint=checkpoint, precision=precision, mode="edge", cache_dir=cache
)
load_ms = (time.perf_counter() - start) * 1000
mx.reset_peak_memory()
results = []
for case in json.loads((root / "fixtures.json").read_text())["cases"]:
    assert encoder.tokenize(case["texts"])[0] == case["ids"]
    for _ in range(3):
        encoder.encode(case["texts"], task="Raw", dimensions=768)
    times = []
    for _ in range(5):
        mx.synchronize()
        start = time.perf_counter()
        vectors = encoder.encode(case["texts"], task="Raw", dimensions=768)
        mx.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    assert vectors.shape == (len(case["texts"]), 768) and np.isfinite(vectors).all()
    print(case["id"], times, flush=True)
    results.append(dict(id=case["id"], ms=times, vectors=vectors.tolist()))
(root / "mlx-bf16.json").write_text(
    json.dumps(
        dict(
            runtime="mlx-turbo-bf16-edge",
            conversion_ms=conversion_ms,
            load_ms=load_ms,
            dimensions=768,
            warmups=3,
            trials=5,
            inference_cache=False,
            timing="prepared text -> tokenization + length-sorted padded batches + synchronized host copy",
            mlx_peak_allocator_bytes=mx.get_peak_memory(),
            cases=results,
        ),
        indent=2,
    )
)
