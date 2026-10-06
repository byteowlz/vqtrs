# /// script
# requires-python = ">=3.11"
# dependencies = ["transformers==5.19.0", "torch", "torchvision", "pillow", "numpy"]
# ///
"""Capture the inputs and outputs of every operation inside one text layer.

Used to find which kernel first breaks bit identity between vqtrs and the
PyTorch reference: `eg2_op_probe` (Rust) replays each operation on the
captured input and compares against the captured output.

Usage:
    uv run scripts/parity/eg2_op_probe.py --text "..." --out /tmp/eg2-parity/probe
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

MODEL_ID = "google/embeddinggemma-2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="task: search result | query: how do I bound in-flight inference?")
    parser.add_argument("--text-file", help="read the text from a cases JSON entry: <file>:<case name>")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)

    from transformers import AutoModel, AutoProcessor
    from transformers.models.embedding_gemma2 import modeling_embedding_gemma2 as m

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, dict] = {}

    current = {"layer": -1}

    def save(name: str, t: torch.Tensor) -> None:
        if current["layer"] != args.layer:
            return
        a = np.ascontiguousarray(t.detach().float().numpy())
        (out / f"{name}.bin").write_bytes(a.tobytes())
        tensors[name] = {"file": f"{name}.bin", "shape": list(a.shape)}

    # Wrap the free functions the attention module calls.
    orig_rope, orig_attn = m.apply_rotary_pos_emb, m.eager_attention_forward

    def rope(q, k, cos, sin, unsqueeze_dim=1):
        save("rope.q_in", q[0]); save("rope.k_in", k[0]); save("rope.cos", cos[0]); save("rope.sin", sin[0])
        q2, k2 = orig_rope(q, k, cos, sin, unsqueeze_dim)
        save("rope.q_out", q2[0]); save("rope.k_out", k2[0])
        return q2, k2

    def attn(module, query, key, value, attention_mask, **kw):
        save("attn.q", query[0]); save("attn.k", key[0]); save("attn.v", value[0])
        if attention_mask is not None:
            save("attn.mask", attention_mask[0, 0])
        k_rep = m.repeat_kv(key, module.num_key_value_groups)
        scores = torch.matmul(query, k_rep.transpose(2, 3)) * kw.get("scaling", 1.0)
        save("attn.scores", scores[0])
        if attention_mask is not None:
            scores = scores + attention_mask
        save("attn.scores_masked", scores[0])
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
        save("attn.probs", probs[0])
        o, w = orig_attn(module, query, key, value, attention_mask, **kw)
        save("attn.out", o[0])
        return o, w

    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32, attn_implementation="eager").eval()
    m.apply_rotary_pos_emb = rope
    layer = model.language_model.layers[args.layer]
    m.eager_attention_forward = attn
    layer.self_attn.__class__.forward.__globals__["eager_attention_forward"] = attn
    layer.self_attn.__class__.forward.__globals__["apply_rotary_pos_emb"] = rope

    def hook(name):
        def fn(_module, inputs, output):
            save(f"{name}.in", inputs[0][0])
            save(f"{name}.out", (output[0] if isinstance(output, tuple) else output)[0])
        return fn

    for idx, each in enumerate(model.language_model.layers):
        each.register_forward_pre_hook(lambda _m, _i, idx=idx: current.__setitem__("layer", idx))

    for name in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                 "post_feedforward_layernorm", "self_attn.q_proj", "self_attn.k_proj",
                 "self_attn.v_proj", "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm",
                 "self_attn.v_norm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "mlp.act_fn",
                 "ple_block"]:
        layer.get_submodule(name).register_forward_hook(hook(name))
    layer.register_forward_hook(hook("layer"))

    text = args.text
    if args.text_file:
        path, name = args.text_file.rsplit(":", 1)
        case = next(c for c in json.loads(pathlib.Path(path).read_text()) if c["name"] == name)
        text = case.get("prompt", "") + case["text"]
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    ids = processor(text=[text], return_tensors="pt")["input_ids"]
    with torch.no_grad():
        model(input_ids=ids)
    (out / "manifest.json").write_text(json.dumps({"layer": args.layer, "tensors": tensors}, indent=1))
    print(f"captured {len(tensors)} tensors from layer {args.layer} into {out}")


if __name__ == "__main__":
    main()
