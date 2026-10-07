# /// script
# requires-python = ">=3.11"
# dependencies = ["av==18.1.0", "numpy==2.4.6"]
# ///
"""Separate native codec RGB verification from the pinned CPU model oracle.

Use synthetic reference fixtures only. Export runs on the decoder host with
PyAV + NumPy. Rebase runs in the existing pinned torch/torchvision/transformers
CPU reference environment; it does not load downloaded code or fetch weights.
The original reference is never modified and the destination must not exist.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import runpy
import shutil
import subprocess

import av
import numpy as np


def export_rgb(args):
    assert av.__version__ == "18.1.0" and np.__version__ == "2.4.6"
    manifest = json.loads((args.reference / "manifest.json").read_text())
    case = next(c for c in manifest["cases"] if c["name"] == args.case)
    path = args.reference / case["container_file"]
    expected = case["count"] * case["height"] * case["width"] * 3
    assert 0 < expected <= 256 * 1024 * 1024, "synthetic fixture RGB limit"
    assert 0 < path.stat().st_size <= 16 * 1024 * 1024
    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        assert float(stream.average_rate) == case["fps"]
        for frame in container.decode(stream):
            assert len(frames) < case["count"], "unexpected extra source frame"
            assert (frame.width, frame.height) == (case["width"], case["height"])
            frames.append(frame.to_ndarray(format="rgb24").tobytes())
    rgb = b"".join(frames)
    assert len(rgb) == expected
    command = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1",
               "-noautorotate", "-max_pixels", "2097152", "-protocol_whitelist", "file",
               "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
               "-filter_threads", "1", "-frames:v", str(case["count"] + 1),
               "-threads", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    decoded = subprocess.run(command, capture_output=True, check=True, timeout=30).stdout
    assert decoded == rgb, "native FFmpeg RGB differs from independent native PyAV"
    old = (args.reference / case["frames_file"]).read_bytes()
    assert len(old) == len(rgb)
    differences = np.abs(np.frombuffer(old, np.uint8).astype(np.int16)
                         - np.frombuffer(rgb, np.uint8).astype(np.int16))
    evidence = {
        "case": args.case, "width": case["width"], "height": case["height"],
        "count": case["count"], "fps": case["fps"],
        "machine": platform.machine(), "system": platform.system(),
        "av": av.__version__, "av_libraries": av.library_versions,
        "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True,
                                 text=True, check=True, timeout=5).stdout.splitlines()[0],
        "container_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
        "native_ffmpeg_vs_pyav_unequal": 0,
        "original_rgb_unequal": int(np.count_nonzero(differences)),
        "original_rgb_max_abs_byte": int(differences.max()),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "frames.rgb").write_bytes(rgb)
    (args.out / "decoder.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


def rebase_reference(args):
    # Explicit local source import under -I, never a checkpoint-provided module.
    helper = runpy.run_path(str(Path(__file__).with_name("eg2_video_frontend_reference.py")))
    manifest = json.loads((args.reference / "manifest.json").read_text())
    for name, module in [("torch", helper["torch"]), ("torchvision", helper["torchvision"]),
                         ("transformers", helper["transformers"]), ("pillow", helper["PIL"]),
                         ("numpy", np), ("av", av)]:
        assert module.__version__ == manifest["versions"][name], (name, module.__version__)
    evidence = json.loads((args.rgb / "decoder.json").read_text())
    assert evidence["native_ffmpeg_vs_pyav_unequal"] == 0, "unverified decoder RGB"
    case = next(c for c in manifest["cases"] if c["name"] == evidence["case"])
    for key in ["width", "height", "count", "fps"]:
        assert case[key] == evidence[key], key
    assert hashlib.sha256((args.reference / case["container_file"]).read_bytes()).hexdigest() == evidence["container_sha256"]
    rgb = (args.rgb / "frames.rgb").read_bytes()
    assert hashlib.sha256(rgb).hexdigest() == evidence["rgb_sha256"]
    frames = np.frombuffer(rgb, np.uint8).reshape(case["count"], case["height"], case["width"], 3).copy()
    assert not args.out.resolve().is_relative_to(args.reference.resolve()), "output must be separate"
    shutil.copytree(args.reference, args.out)
    helper["torch"].set_num_threads(1)
    processor = helper["AutoProcessor"].from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    updated = helper["save_case"](args.out, processor, case["name"], frames, case["fps"], case["container_file"])
    assert updated["input_ids"] == case["input_ids"]
    assert updated["sampled_indices"] == case["sampled_indices"]
    helper["model_reference"](args.out, processor, args.model_dir, [updated], {case["name"]})
    manifest["cases"][manifest["cases"].index(case)] = updated
    manifest.setdefault("native_rgb_oracles", {})[case["name"]] = evidence
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    original = np.fromfile(args.reference / case["model_tensors"]["embedding"]["file"], dtype="<f4")
    native = np.fromfile(args.out / updated["model_tensors"]["embedding"]["file"], dtype="<f4")
    print("Decoder-only change in final CPU vector maxabs:", float(np.max(np.abs(original - native))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    export = modes.add_parser("export", help="prove native FFmpeg/PyAV RGB and export it")
    export.add_argument("--case", default="video_h264")
    rebase = modes.add_parser("rebase", help="generate CPU oracle for the exported native RGB")
    rebase.add_argument("--rgb", type=Path, required=True)
    rebase.add_argument("--model-dir", type=Path, required=True)
    for mode in [export, rebase]:
        mode.add_argument("--reference", type=Path, required=True)
        mode.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (export_rgb if args.mode == "export" else rebase_reference)(args)


if __name__ == "__main__":
    main()
