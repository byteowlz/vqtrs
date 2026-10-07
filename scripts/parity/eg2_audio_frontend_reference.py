# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.6", "transformers==5.19.0"]
# ///
"""Dump the fixed Gemma4 audio frontend tables and deterministic operation probes.

For the ignored Rust FFT/table tests. No checkpoint or audio fixtures needed.
Run with an isolated interpreter:
    uv run --with numpy==2.4.6 --with transformers==5.19.0 \
        python -I scripts/parity/eg2_audio_frontend_reference.py --out /tmp/eg2-parity/mel
"""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fe = Gemma4AudioFeatureExtractor()
    tensors = {}

    def save(name, array):
        array = np.ascontiguousarray(array)
        (args.out / name).write_bytes(array.tobytes())
        tensors[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}

    save("window_f32.bin", fe.window)
    save("filters_f64.bin", fe.mel_filters)
    frames = (np.random.default_rng(3).standard_normal((50, 320)) * 0.3).astype(np.float32)
    frames = frames * fe.window
    spec = np.fft.rfft(frames, n=512, axis=-1)
    mag = np.abs(spec)
    save("fft_frames_f32.bin", frames)
    save("fft_out_c64.bin", spec)
    save("fft_abs_f32.bin", mag)
    save("mel_f64.bin", np.matmul(mag[None], fe.mel_filters))
    assert spec.dtype == np.complex64
    (args.out / "manifest.json").write_text(json.dumps({"numpy": np.__version__, "tensors": tensors}, indent=2))
    print(f"wrote {len(tensors)} operation probes to {args.out}")


if __name__ == "__main__":
    main()
