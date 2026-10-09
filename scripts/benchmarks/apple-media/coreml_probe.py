"""Developer probe of pinned native graphs, using unchanged public F32 oracle inputs."""

import json
import os
import runpy
import sys
from pathlib import Path
from time import perf_counter

import coremltools as ct
import numpy as np
import soundfile as sf

WORK = Path(os.environ.get("EG2_BENCH_WORK_DIR", "/tmp/vqtrs-native-media"))
ROOT = Path(os.environ.get("EG2_COREML_DIR", str(WORK / "coreml")))
ASSETS = runpy.run_path(str(Path(__file__).with_name("assets.py")))
ASSETS["verify"](ROOT)
REF = Path(os.environ.get("EG2_REFERENCE_DIR", "/tmp/eg2-parity"))
MEDIA = Path(
    os.environ.get(
        "EG2_MEDIA_DIR",
        str(Path(__file__).resolve().parents[3] / "scripts/parity/media"),
    )
)
OUT = WORK / "coreml-results.json"
TABLE = np.memmap(ROOT / "embeddings.bf16", dtype="<u2", mode="r", shape=(262144, 512))
POS = np.memmap(
    ROOT / "position_embeddings.f16", dtype="<f2", mode="r", shape=(2, 1024, 768)
)
text_models = {}


def load(package, function=None, units=ct.ComputeUnit.CPU_AND_NE):
    start = perf_counter()
    model = ct.models.MLModel(
        str(ROOT / package), compute_units=units, function_name=function
    )
    print("loaded", package, function, round(perf_counter() - start, 2), flush=True)
    return model


def table(ids):
    return ASSETS["token_rows"](TABLE, ids)


def text(embeds):
    n = len(embeds)
    if not 0 < n <= 512:
        raise ValueError("explicit CoreML context overflow")
    s = next(x for x in [32, 48, 64, 128, 256, 512] if x >= n)
    if s not in text_models:
        text_models[s] = load("EmbeddingGemma2Text.mlpackage", f"embed_{s}")
    x = np.zeros((1, s, 512), dtype=np.float16)
    x[0, :n] = embeds
    mask = np.zeros((1, s), dtype=np.float16)
    mask[0, :n] = 1
    return (
        text_models[s]
        .predict({"inputs_embeds": x, "attention_mask": mask})["embedding"]
        .reshape(-1)
        .astype(np.float32)
    )


def tensor(folder, case, key):
    t = case["tensors"][key]
    return np.fromfile(folder / t["file"], dtype=t["dtype"]).reshape(t["shape"])


def compare(name, result, expected, elapsed, **extra):
    assert result.shape == (768,) and np.isfinite(result).all()
    cosine = (
        np.dot(result.astype(np.float64), expected.astype(np.float64))
        / np.linalg.norm(result.astype(np.float64))
        / np.linalg.norm(expected.astype(np.float64))
    )
    info = {
        "name": name,
        "max_abs": float(np.max(np.abs(result - expected))),
        "cosine": float(cosine),
        "norm": float(np.linalg.norm(result.astype(np.float64))),
        "pipeline_ms": elapsed * 1000,
        **extra,
    }
    print(json.dumps(info), flush=True)
    return {**info, "vector": result.tolist()}


results = []
if sys.argv[1] == "audio":
    model = load("EmbeddingGemma2Audio.mlpackage", units=ct.ComputeUnit.CPU_AND_GPU)
    folder = REF / "ref-audio-final"
    if not folder.exists():
        folder = REF / "ref-audio"
    manifest = json.loads((folder / "manifest.json").read_text())
    for case in manifest["cases"]:
        name = case["name"]
        if "long" in name:
            continue
        path = {
            "a_chirp_2370ms": "chirp_2370ms.wav",
            "a_short_300ms": "short_300ms.wav",
            "a_silence_1000ms": "silence_1000ms.wav",
            "a_speech_7100ms_pcm16": "speech_7100ms_pcm16.wav",
        }[name]
        samples, sr = sf.read(MEDIA / path, dtype="float32")
        assert (
            sr == 16000
            and samples.ndim == 1
            and len(samples) <= 160000
            and np.isfinite(samples).all()
        )
        wave = np.zeros((1, 160160), dtype=np.float32)
        wave[0, 160 : 160 + len(samples)] = samples
        valid = max(0, min(1000, (160 + len(samples) - 321) // 160 + 1))
        mask = np.zeros((1, 1000), dtype=np.float16)
        mask[0, :valid] = 1
        count = (valid + 3) // 4
        start = perf_counter()
        tokens = model.predict({"waveform": wave, "frame_mask": mask})["audio_tokens"][
            0, :count
        ]
        embeds = np.concatenate(
            [table([2, 256000]), tokens.astype(np.float16), table([258883, 1])]
        )
        result = text(embeds)
        expected = tensor(folder, case, "hf.embedding")
        results.append(
            compare(
                name,
                result,
                expected,
                perf_counter() - start,
                valid_frames=valid,
                soft_tokens=count,
            )
        )
elif sys.argv[1] == "image":
    model = load(
        "EmbeddingGemma2Vision.mlpackage", "vision_280", ct.ComputeUnit.CPU_AND_GPU
    )
    folder = REF / "ref-image"
    manifest = json.loads((folder / "manifest.json").read_text())
    for case in manifest["cases"]:
        patches = tensor(folder, case, "input.pixel_values").astype(np.float16)
        positions = tensor(folder, case, "input.image_position_ids")[0]
        keep = (positions >= 0).all(axis=-1)
        coords = positions[keep]
        columns = (int(coords[:, 0].max()) + 1) // 3
        groups = (coords[:, 1] // 3) * columns + coords[:, 0] // 3
        count = int(groups.max()) + 1
        pool = np.zeros((280, 2520), dtype=np.float16)
        pool[groups, np.flatnonzero(keep)] = np.float16(1 / 9)
        embeds_pos = np.zeros((1, 2520, 768), dtype=np.float16)
        embeds_pos[0, keep] = POS[0, coords[:, 0]] + POS[1, coords[:, 1]]
        start = perf_counter()
        tokens = model.predict(
            {
                "patches": patches,
                "positions": positions[None].astype(np.float16),
                "pool": pool,
                "valid": keep[None].astype(np.float16),
                "position_embeddings": embeds_pos,
            }
        )["image_tokens"][0, :count]
        embeds = np.concatenate(
            [table([2, 255999]), tokens.astype(np.float16), table([258882, 1])]
        )
        expected = tensor(folder, case, "hf.embedding")
        result = text(embeds)
        results.append(
            compare(
                case["name"],
                result,
                expected,
                perf_counter() - start,
                soft_tokens=count,
                oracle_tokens=int(
                    np.count_nonzero(tensor(folder, case, "input.attention_mask"))
                ),
            )
        )
elif sys.argv[1] == "video":
    model = load(
        "EmbeddingGemma2Vision.mlpackage", "vision_140", ct.ComputeUnit.CPU_AND_GPU
    )
    folder = Path(os.environ.get("EG2_VIDEO_REFERENCE_DIR", "/tmp/eg2-video-ref-140"))
    manifest = json.loads((folder / "manifest.json").read_text())
    for case in manifest["cases"]:
        if "model_tensors" not in case:
            continue
        if len(case["input_ids"]) > 512:
            results.append(
                {
                    "name": case["name"],
                    "status": "explicitly-unsupported-context",
                    "tokens": len(case["input_ids"]),
                }
            )
            continue
        n = len(case["sampled_indices"])
        all_patches = np.fromfile(folder / case["pixels_file"], dtype="<f4").reshape(
            n, 1260, 768
        )
        all_positions = np.fromfile(
            folder / case["positions_file"], dtype="<i8"
        ).reshape(n, 1260, 2)
        features = []
        start = perf_counter()
        for patches, positions in zip(all_patches, all_positions):
            keep = (positions >= 0).all(axis=-1)
            coords = positions[keep]
            columns = (int(coords[:, 0].max()) + 1) // 3
            groups = (coords[:, 1] // 3) * columns + coords[:, 0] // 3
            count = int(groups.max()) + 1
            pool = np.zeros((140, 1260), dtype=np.float16)
            pool[groups, np.flatnonzero(keep)] = np.float16(1 / 9)
            embeds_pos = np.zeros((1, 1260, 768), dtype=np.float16)
            embeds_pos[0, keep] = POS[0, coords[:, 0]] + POS[1, coords[:, 1]]
            features.append(
                model.predict(
                    {
                        "patches": patches[None].astype(np.float16),
                        "positions": positions[None].astype(np.float16),
                        "pool": pool,
                        "valid": keep[None].astype(np.float16),
                        "position_embeddings": embeds_pos,
                    }
                )["image_tokens"][0, :count]
            )
        embeds = table(case["input_ids"])
        marker = np.array(case["input_ids"]) == 258884
        embeds[marker] = np.concatenate(features, axis=0)
        result = text(embeds)
        expected = np.fromfile(
            folder / case["model_tensors"]["embedding"]["file"], dtype="<f4"
        )
        results.append(
            compare(
                case["name"],
                result,
                expected,
                perf_counter() - start,
                frames=n,
                tokens=len(embeds),
            )
        )
else:
    raise ValueError("choose image, audio or video")
OUT = OUT.with_name(f"coreml-{sys.argv[1]}-results.json")
OUT.write_text(
    json.dumps(
        {
            "asset_revision": "3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4",
            "preprocessing": "shared F32 oracle patches/positions for image; original synthetic 16kHz WAV for audio",
            "arm": sys.argv[1],
            "cases": results,
        },
        indent=2,
    )
    + "\n"
)
