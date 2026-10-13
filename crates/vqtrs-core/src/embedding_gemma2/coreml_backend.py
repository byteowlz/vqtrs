"""Pinned FP16 native CoreML graphs, with explicit GPU media / NE-preferred text."""

import coremltools as ct
import numpy as np
from native_worker import tensor


class CoreML:
    def __init__(self, root):
        if ct.__version__ != "9.0":
            raise ValueError("CoreML worker requires coremltools9.0")
        self.root = root
        self.table = np.memmap(
            root / "embeddings.bf16", dtype="<u2", mode="r", shape=(262144, 512)
        )
        self.positions = np.memmap(
            root / "position_embeddings.f16",
            dtype="<f2",
            mode="r",
            shape=(2, 1024, 768),
        )
        self.models = {}
        # Initialize a real function before the ready handshake, not just metadata.
        self.load("EmbeddingGemma2Text", "embed_32", ct.ComputeUnit.CPU_AND_NE)

    def load(self, package, function, units):
        key = (package, function)
        if key not in self.models:
            self.models[key] = ct.models.MLModel(
                str(self.root / (package + ".mlpackage")),
                compute_units=units,
                function_name=function,
            )
        return self.models[key]

    def rows(self, ids):
        bits = np.asarray(self.table[ids], dtype=np.uint32) << 16
        return (bits.view(np.float32) * np.float32(np.sqrt(np.float32(512)))).astype(
            np.float16
        )

    def vision(self, item):
        frames, patches = item["frames"], item["patches"]
        if frames < 1 or frames > 32 or patches not in (1260, 2520):
            raise ValueError("invalid vision shape")
        pixels = tensor(item, "pixels", (frames, patches, 768))
        positions = tensor(item, "positions", (frames, patches, 2), "<i8")
        slots = patches // 9
        model = self.load(
            "EmbeddingGemma2Vision", "vision_" + str(slots), ct.ComputeUnit.CPU_AND_GPU
        )
        output = []
        for values, coords in zip(pixels, positions):
            keep = (coords >= 0).all(axis=-1)
            valid = coords[keep]
            if not len(valid) or np.any(valid >= 1024):
                raise ValueError("invalid vision positions")
            columns = (int(valid[:, 0].max()) + 1) // 3
            groups = (valid[:, 1] // 3) * columns + valid[:, 0] // 3
            count = int(groups.max()) + 1
            if count != item["soft_tokens"] or count > slots:
                raise ValueError("vision placeholder mismatch")
            pool = np.zeros((slots, patches), dtype=np.float16)
            pool[groups, np.flatnonzero(keep)] = np.float16(1 / 9)
            pos = np.zeros((1, patches, 768), dtype=np.float16)
            pos[0, keep] = (
                self.positions[0, valid[:, 0]] + self.positions[1, valid[:, 1]]
            )
            result = model.predict(
                {
                    "patches": values[None].astype(np.float16),
                    "positions": coords[None].astype(np.float16),
                    "pool": pool,
                    "valid": keep[None].astype(np.float16),
                    "position_embeddings": pos,
                }
            )["image_tokens"][0, :count]
            output.append(result.astype(np.float16))
        return np.concatenate(output, axis=0)

    def audio(self, item):
        samples = item["samples"]
        if not 161 <= samples <= 160000:
            raise ValueError("CoreML audio requires >10ms and <=10s at16kHz")
        wave = np.zeros((1, 160160), dtype=np.float32)
        wave[0, 160 : 160 + samples] = tensor(item, "wave", (samples,))
        valid = min(1000, (160 + samples - 321) // 160 + 1)
        mask = np.zeros((1, 1000), dtype=np.float16)
        mask[0, :valid] = 1
        count = (valid + 3) // 4
        if count != item["soft_tokens"]:
            raise ValueError("audio placeholder mismatch")
        model = self.load("EmbeddingGemma2Audio", None, ct.ComputeUnit.CPU_AND_GPU)
        return model.predict({"waveform": wave, "frame_mask": mask})["audio_tokens"][
            0, :count
        ].astype(np.float16)

    def one(self, item):
        ids = np.asarray(item["ids"], dtype=np.int32)
        embeds = self.rows(ids.tolist())
        kind = item["kind"]
        marker = {"image": 258880, "video": 258884, "audio": 258881}.get(kind)
        if marker is not None:
            features = self.audio(item) if kind == "audio" else self.vision(item)
            mask = ids == marker
            if int(np.count_nonzero(mask)) != len(features):
                raise ValueError("media token count mismatch")
            embeds[mask] = features
        elif kind != "text":
            raise ValueError("invalid modality")
        width = next(x for x in (32, 48, 64, 128, 256, 512) if x >= len(ids))
        x = np.zeros((1, width, 512), dtype=np.float16)
        x[0, : len(ids)] = embeds
        mask = np.zeros((1, width), dtype=np.float16)
        mask[0, : len(ids)] = 1
        model = self.load(
            "EmbeddingGemma2Text", "embed_" + str(width), ct.ComputeUnit.CPU_AND_NE
        )
        return model.predict({"inputs_embeds": x, "attention_mask": mask})[
            "embedding"
        ].reshape(768)

    def embed(self, items):
        return np.stack([self.one(item) for item in items])
