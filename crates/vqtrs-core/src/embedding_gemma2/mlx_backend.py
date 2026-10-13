"""Audited regular mlx-vlm tensor definitions; no processor, hub, turbo or checkpoint code."""

import json
import os
import shutil
import sys
import types
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from native_worker import tensor, verify


class MLX:
    def __init__(self, root, directory):
        from importlib.metadata import version

        if version("mlx") != "0.32.3":
            raise ValueError("regular MLX worker requires mlx0.32.3")
        source = (
            Path(os.environ["VQTRS_EG2_MLX_SOURCE"]).resolve(strict=True) / "mlx_vlm"
        )
        manifest = json.loads((directory / "mlx-source.json").read_text())
        # Verify the entire Python tree, including extra/untracked modules, before any import.
        if {str(p.relative_to(source)) for p in source.rglob("*.py")} != set(
            manifest["files"]
        ):
            raise ValueError("MLX source file set mismatch")
        # Import only authenticated .py copies in Rust's private directory;
        # arbitrary .so/.pyc files in an external checkout cannot shadow them.
        checked = directory / "checked_mlx" / "mlx_vlm"
        for relative in manifest["files"]:
            destination = checked / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)
        verify(checked, manifest)
        source = checked
        sys.dont_write_bytecode = True
        for name, relative in [
            ("mlx_vlm", ""),
            ("mlx_vlm.models", "models"),
            ("mlx_vlm.models.gemma4", "models/gemma4"),
            ("mlx_vlm.models.embedding_gemma2", "models/embedding_gemma2"),
            ("mlx_vlm.models.qwen3_5", "models/qwen3_5"),
        ]:
            module = types.ModuleType(name)
            module.__path__ = [str(source / relative)]
            sys.modules[name] = module
        from mlx_vlm.models.embedding_gemma2.config import ModelConfig
        from mlx_vlm.models.embedding_gemma2.embedding_gemma2 import Model

        mx.set_cache_limit(512 * 1024 * 1024)
        mx.set_memory_limit(6 * 1024 * 1024 * 1024)
        self.model = Model(
            ModelConfig.from_dict(json.loads((root / "config.json").read_text()))
        )
        weights = self.model.sanitize(mx.load(str(root / "model.safetensors")))
        expected = dict(tree_flatten(self.model.parameters()))
        if set(weights) != set(expected) or any(
            v.shape != expected[k].shape or v.dtype != mx.bfloat16
            for k, v in weights.items()
        ):
            raise ValueError("original BF16 checkpoint shape/key/dtype mismatch")
        self.model.load_weights(list(weights.items()), strict=True)
        mx.eval(self.model.parameters())
        self.model.get_video_features = self.video

    def video(self, pixels, positions):
        features = []
        for i in range(pixels.shape[0]):
            feature = self.model.get_image_features(
                pixels[i : i + 1], positions[i : i + 1]
            )
            mx.eval(feature)
            features.append(feature)
        return mx.concatenate(features, axis=0)

    def media(self, item):
        inputs = {
            "input_ids": mx.array([item["ids"]]),
            "attention_mask": mx.ones((1, len(item["ids"])), dtype=mx.int32),
        }
        kind = item["kind"]
        if kind in ("image", "video"):
            frames, patches = item["frames"], item["patches"]
            if not 1 <= frames <= 32 or patches not in (1260, 2520):
                raise ValueError("invalid vision shape")
            pixels = mx.array(tensor(item, "pixels", (frames, patches, 768)))
            positions = mx.array(tensor(item, "positions", (frames, patches, 2), "<i8"))
            if kind == "image":
                inputs.update(pixel_values=pixels, image_position_ids=positions)
            else:
                inputs.update(pixel_values_videos=pixels, video_position_ids=positions)
        elif kind == "audio":
            frames = item["frames"]
            if not 1 <= frames <= 3200:
                raise ValueError("invalid audio frame count")
            inputs["input_features"] = mx.array(tensor(item, "mel", (1, frames, 128)))
            inputs["input_features_mask"] = mx.array(
                tensor(item, "mask", (1, frames), "u1"), dtype=mx.bool_
            )
        else:
            raise ValueError("invalid modality")
        return self.model(**inputs).text_embeds.astype(mx.float32)

    def texts(self, items):
        width = max(len(item["ids"]) for item in items)
        if len(items) > 1 and width > 512:
            raise ValueError("long MLX texts must be submitted individually")
        ids = np.zeros((len(items), width), dtype=np.int32)
        mask = np.zeros_like(ids)
        for i, item in enumerate(items):
            ids[i, : len(item["ids"])] = item["ids"]
            mask[i, : len(item["ids"])] = 1
        return self.model(
            mx.array(ids), attention_mask=mx.array(mask)
        ).text_embeds.astype(mx.float32)

    def embed(self, items):
        if all(item["kind"] == "text" for item in items):
            output = self.texts(items)
        else:
            if len(items) != 1:
                raise ValueError("media requests must be serialized")
            output = self.media(items[0])
        mx.eval(output)
        return np.array(output)
