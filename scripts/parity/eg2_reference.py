# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers==5.19.0",
#   "sentence-transformers==6.1.0",
#   "torch==2.14.1",
#   "torchvision",
#   "pillow==12.3.0",
#   "numpy==2.4.6",
#   "librosa",
#   "soundfile",
#   "av",
# ]
# ///
"""Reference embeddings for google/embeddinggemma-2 from the Python implementation.

Writes, per case, the token ids, the processor tensors (pixel values, audio
features, ...), the per-layer hidden states and the final embeddings, as
little-endian f32 / i64 `.bin` files plus a `manifest.json`. The vqtrs parity
check (`scripts/parity/eg2_compare.py`) reads this directory.

Two reference paths are recorded:

* `st` -- `SentenceTransformer.encode`, the canonical public API
  (default attention implementation).
* `hf` -- `EmbeddingGemma2Model` with eager text/vision attention and boolean
  audio masks (`sdpa` mask configuration), mean pooling and L2 normalisation
  done here, plus the intermediates.

Everything runs in float32 on CPU, one input per forward pass (no padding).

Usage:
    uv run scripts/parity/eg2_reference.py --cases scripts/parity/eg2_cases.json \
        --out /tmp/eg2-parity/ref
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def save(out: pathlib.Path, name: str, array: np.ndarray) -> dict:
    array = np.ascontiguousarray(array)
    if array.dtype == np.float64:
        array = array.astype(np.float32)
    path = out / f"{name}.bin"
    path.write_bytes(array.tobytes())
    return {"file": path.name, "dtype": str(array.dtype), "shape": list(array.shape)}


def load_media(case: dict, base: pathlib.Path):
    from PIL import Image

    kind = case["modality"]
    if kind == "image":
        # Raw PIL image: the processor's own RGB conversion is part of the reference.
        image = Image.open(base / case["path"])
        image.load()
        return image
    if kind == "audio":
        import soundfile as sf

        audio, rate = sf.read(base / case["path"], dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != 16_000:
            raise ValueError(f"{case['path']}: expected 16 kHz audio, got {rate} Hz")
        return audio
    if kind == "video":
        return str(base / case["path"])
    return None


def attach_audio_hooks(model, out, name, entry):
    """Capture tensor outputs of each audio module for a focused parity probe."""
    def capture(label):
        def hook(_module, _args, output):
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, torch.Tensor):
                key = f"hf.audio.{label}"
                entry["tensors"][key] = save(out, f"{name}.{key}", output.detach().numpy())
        return hook
    return [module.register_forward_hook(capture(label))
            for label, module in model.audio_tower.named_modules() if label]


def processor_inputs(processor, modality, case, prompt, media):
    """Keep modality dispatch separate from model execution and dump writing."""
    if modality == "text":
        return processor(text=[prompt + case["text"]], return_tensors="pt")
    if modality == "image":
        return processor(images=[[media]], return_tensors="pt")
    if modality == "audio":
        return processor(audio=[media], return_tensors="pt")
    if modality == "video":
        return processor(videos=[media], return_tensors="pt")
    raise ValueError(f"unknown modality {modality}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--audio-trace", action="store_true", help="Capture audio module outputs for a focused case")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    import transformers
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoProcessor

    cases_path = pathlib.Path(args.cases)
    cases = json.loads(cases_path.read_text())
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    # Eager attention defines the text and vision numerics. The audio tower
    # always computes its chunked attention explicitly and only consumes the
    # mask; under "eager", transformers 5.19 hands it a float additive mask
    # that `masked_fill(mask.logical_not(), ...)` inverts (a bug), so audio
    # gets the boolean mask the "sdpa" setting produces -- the semantics
    # SentenceTransformer.encode runs with.
    hf_model = AutoModel.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        revision=MODEL_REVISION,
        attn_implementation={"text_config": "eager", "vision_config": "eager", "audio_config": "sdpa"},
    ).eval()
    st_model = SentenceTransformer(
        MODEL_ID, device="cpu", revision=MODEL_REVISION, model_kwargs={"dtype": torch.float32}
    )

    manifest = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "versions": {
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
        },
        "cases": [],
    }

    for case in cases:
        name = case["name"]
        modality = case["modality"]
        prompt = case.get("prompt", "")
        media = load_media(case, cases_path.parent)
        entry = {"name": name, "modality": modality, "prompt": prompt, "tensors": {}}

        # --- processor inputs (shared by both paths) ---
        inputs = processor_inputs(processor, modality, case, prompt, media)

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                entry["tensors"][f"input.{key}"] = save(out, f"{name}.input.{key}", value.numpy())

        # --- hf path: eager attention, intermediates captured ---
        hooks = []
        if args.audio_trace and modality == "audio":
            hooks = attach_audio_hooks(hf_model, out, name, entry)
        with torch.no_grad():
            model_inputs = {
                k: v for k, v in inputs.items() if k not in processor.unused_input_names
            }
            outputs = hf_model(**model_inputs, output_hidden_states=True)
            tokens = outputs.last_hidden_state[0]  # (seq, 768)
            mask = inputs["attention_mask"][0].to(tokens.dtype)
            pooled = (tokens * mask[:, None]).sum(0) / mask.sum()
            embedding = torch.nn.functional.normalize(pooled, p=2, dim=0)

        for hook in hooks:
            hook.remove()

        entry["tensors"]["hf.token_embeddings"] = save(out, f"{name}.hf.token_embeddings", tokens.numpy())
        entry["tensors"]["hf.embedding"] = save(out, f"{name}.hf.embedding", embedding.numpy())
        if outputs.hidden_states is not None:
            # Layer outputs are hidden_size wide; the trailing entry is the projected output.
            width = hf_model.config.text_config.hidden_size
            hidden = torch.stack([h[0] for h in outputs.hidden_states if h.shape[-1] == width]).numpy()
            entry["tensors"]["hf.hidden_states"] = save(out, f"{name}.hf.hidden_states", hidden)
        if outputs.image_hidden_states is not None:
            entry["tensors"]["hf.image_features"] = save(
                out, f"{name}.hf.image_features", outputs.image_hidden_states.numpy()
            )
        if outputs.audio_hidden_states is not None:
            entry["tensors"]["hf.audio_features"] = save(
                out, f"{name}.hf.audio_features", outputs.audio_hidden_states.numpy()
            )

        # --- st path: the public API ---
        if modality == "text":
            st_embedding = st_model.encode(
                [case["text"]], prompt=prompt or None, convert_to_numpy=True, batch_size=1
            )[0]
        else:
            st_embedding = st_model.encode([media], convert_to_numpy=True, batch_size=1)[0]
        entry["tensors"]["st.embedding"] = save(out, f"{name}.st.embedding", st_embedding)

        diff = float(np.abs(st_embedding - embedding.numpy()).max())
        entry["st_vs_hf_max_abs_diff"] = diff
        print(f"{name:28s} {modality:6s} seq={tokens.shape[0]:5d} st-vs-hf max|diff|={diff:.3e}")
        manifest["cases"].append(entry)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest['cases'])} cases to {out}")


if __name__ == "__main__":
    main()
