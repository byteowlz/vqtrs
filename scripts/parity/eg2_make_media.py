# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow==12.3.0", "numpy==2.4.6", "soundfile"]
# ///
"""Deterministic synthetic media for the EmbeddingGemma 2 parity cases.

Writes into scripts/parity/media/ (gitignored); the cases files reference
these names. No downloads: everything is generated from a fixed seed.
"""

import pathlib

import numpy as np
from PIL import Image, ImageDraw

OUT = pathlib.Path(__file__).parent / "media"


def photo_like(rng: np.random.Generator, w: int, h: int) -> Image.Image:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 127 + 120 * np.sin(x / 37.0) * np.cos(y / 53.0)
    g = 127 + 120 * np.sin((x + y) / 71.0)
    b = 127 + 120 * np.cos(np.hypot(x - w / 2, y - h / 3) / 29.0)
    img = np.stack([r, g, b], -1) + rng.normal(0, 12, (h, w, 3))
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    for _ in range(12):
        x0, y0 = int(rng.integers(0, max(1, w - 40))), int(rng.integers(0, max(1, h - 40)))
        d.rectangle([x0, y0, x0 + int(rng.integers(10, 120)), y0 + int(rng.integers(10, 90))],
                    fill=tuple(int(v) for v in rng.integers(0, 256, 3)))
        d.line([0, int(rng.integers(0, h)), w, int(rng.integers(0, h))], fill=(255, 255, 255), width=3)
    return im


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(1234)
    photo_like(rng, 1037, 761).save(OUT / "photo_1037x761.png")
    photo_like(rng, 64, 48).save(OUT / "small_64x48.png")
    photo_like(rng, 1920, 90).save(OUT / "wide_1920x90.png")
    photo_like(rng, 816, 1224).save(OUT / "portrait_816x1224.jpg", quality=90)
    rgba = np.array(photo_like(rng, 500, 400).convert("RGBA"))
    rgba[..., 3] = np.linspace(0, 255, 500, dtype=np.uint8)[None, :]
    Image.fromarray(rgba, "RGBA").save(OUT / "alpha_500x400.png")
    photo_like(rng, 300, 300).convert("L").save(OUT / "gray_300x300.png")
    photo_like(rng, 912, 672).save(OUT / "exact_912x672.png")  # already a valid target size
    make_audio(rng)
    print(f"wrote media to {OUT}")


def make_audio(rng: np.random.Generator) -> None:
    import soundfile as sf

    sr = 16_000

    def t(seconds: float) -> np.ndarray:
        return np.arange(int(seconds * sr)) / sr

    chirp = np.sin(2 * np.pi * (200 + 900 * t(2.37)) * t(2.37)) * 0.4 + rng.normal(0, 0.03, len(t(2.37)))
    sf.write(OUT / "chirp_2370ms.wav", chirp.astype(np.float32), sr, subtype="FLOAT")
    # Speech-like: formant mixture with a syllable-rate envelope.
    x = t(7.1)
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 4 * x)) ** 2
    voiced = sum(np.sin(2 * np.pi * f * x + p) / (i + 1) for i, (f, p) in enumerate([(130, 0), (700, 1), (1200, 2), (2600, 3)]))
    speech = env * voiced * 0.2 + rng.normal(0, 0.01, len(x))
    sf.write(OUT / "speech_7100ms_pcm16.wav", np.clip(speech, -1, 1), sr, subtype="PCM_16")
    sf.write(OUT / "short_300ms.wav", (np.sin(2 * np.pi * 440 * t(0.3)) * 0.5).astype(np.float32), sr, subtype="FLOAT")
    sf.write(OUT / "silence_1000ms.wav", np.zeros(sr, dtype=np.float32), sr, subtype="FLOAT")
    long = np.sin(2 * np.pi * 330 * t(31.5)) * 0.3 * np.sin(2 * np.pi * 0.5 * t(31.5)) + rng.normal(0, 0.02, len(t(31.5)))
    sf.write(OUT / "long_31500ms_pcm16.wav", np.clip(long, -1, 1), sr, subtype="PCM_16")


if __name__ == "__main__":
    main()
