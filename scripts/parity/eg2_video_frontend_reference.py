# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.14.1", "torchvision==0.29.1", "transformers==5.19.0", "pillow==12.3.0", "numpy==2.4.6", "av==18.1.0"]
# ///
"""Pinned CPU HF video frontend fixtures; does not execute downloaded code.

Run via cached reference environment: uv run --no-project --python <cached-python>
python -I scripts/parity/eg2_video_frontend_reference.py --out /tmp/eg2-video-ref
With --model-forward, the default model cases include decoded_cap (70 source
frames uniformly sampled to 32), FFV1/MKV and H.264/MP4.
Then: EG2_VIDEO_REFERENCE=/tmp/eg2-video-ref cargo test -p vqtrs-core
--features embeddinggemma2 video::tests::reference_parity -- --ignored --nocapture

Synthetic decoded arrays exercise resize/patchification independent of codecs.
Small FFV1/MKV and H.264/MP4 exercise container decoding, RGB conversion and
sampling. PyAV decoded arrays + explicit actual count/rate metadata are the
oracle for MKV: HF's direct PyAV URL path trusts stream.frames, which is zero
for this MKV fixture. That broken URL path is NOT claimed as parity.
The oracle uses the checkpoint video_processor defaults unchanged, including
max_soft_tokens=140. Python class defaults are NOT checkpoint configuration.
HF 5.19 defaults URL loading to torchcodec (absent from the pinned environment),
so MP4 additionally tests the explicit HF PyAV loader.
All fixtures are synthetic. No remote code or URLs are used. --model-forward
loads only the local checkpoint weights; frontend-only runs do not load weights.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import av
import numpy as np
import PIL
from PIL import Image
import torch
import torchvision
import transformers
from torchvision.transforms.v2 import functional as tvf
from transformers import AutoModel, AutoProcessor
from transformers.video_utils import VideoMetadata, load_video

REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def synthetic(count, height, width):
    """Deterministic arithmetic pattern, consistent across NumPy versions."""
    i = np.arange(count * height * width * 3, dtype=np.uint64)
    return ((i * 37 + (i // 7) * 19 + (i // 97) * 11) % 256).astype(np.uint8).reshape(count, height, width, 3)


def save_case(out, processor, name, frames, fps=None, container=None):
    count, height, width, _ = frames.shape
    duration = count / fps if fps is not None else None
    metadata = VideoMetadata(total_num_frames=count, fps=fps, duration=duration)
    inputs = processor(
        videos=[frames], video_metadata=[metadata], return_tensors="pt", return_metadata=True
    )
    pixels = inputs["pixel_values_videos"].cpu().numpy().astype("<f4")
    positions = inputs["video_position_ids"].cpu().numpy().astype("<i8")
    frame_file = f"{name}.rgb"
    pathlib.Path(out / frame_file).write_bytes(frames.tobytes())
    pathlib.Path(out / f"{name}.pixels.bin").write_bytes(pixels.tobytes())
    pathlib.Path(out / f"{name}.positions.bin").write_bytes(positions.tobytes())
    indices = inputs["video_metadata"][0].frames_indices.tolist()
    tokens = int(np.count_nonzero(positions[0, :, 0] >= 0) // processor.video_processor.pooling_kernel_size**2)
    placeholder = processor.replace_video_token({"num_soft_tokens_per_video": [tokens],
                                                "num_frames_per_video": [len(indices)]}, 0)
    expected = (processor.boi_token + processor.video_token * tokens + processor.eoi_token) * len(indices)
    assert placeholder == expected, "default expansion must have no timestamps or separators"
    return {
        "name": name, "width": width, "height": height, "count": count,
        "fps": fps, "duration": duration, "frames_file": frame_file,
        "container_file": container, "sampled_indices": indices,
        "num_soft_tokens": tokens, "pixels_file": f"{name}.pixels.bin",
        "positions_file": f"{name}.positions.bin",
        "input_ids": inputs["input_ids"][0].tolist(),
        "placeholder": placeholder,
    }


def encode(out, name, frames, fps, codec):
    path = out / name
    count, height, width, _ = frames.shape
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "pipe:0", "-an", "-threads", "1"]
    if codec == "ffv1":
        command += ["-c:v", "ffv1", "-pix_fmt", "bgr0"]
    else:
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
    subprocess.run(command + [str(path)], input=frames.tobytes(), check=True, timeout=30)
    with av.open(path) as container:
        stream = container.streams.video[0]
        actual = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(stream)])
        actual_fps = float(stream.average_rate)
    assert actual.shape == frames.shape and actual_fps == fps
    return actual, actual_fps


def model_reference(out, processor, model_dir, cases, selected):
    """Batched HF forward, plus a direct batched-vs-perframe vision probe."""
    model = AutoModel.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, dtype=torch.float32,
        attn_implementation={"text_config": "eager", "vision_config": "eager", "audio_config": "sdpa"},
    ).eval()
    for case in cases:
        if case["name"] not in selected:
            continue
        frames = np.fromfile(out / case["frames_file"], dtype=np.uint8).reshape(
            case["count"], case["height"], case["width"], 3)
        metadata = VideoMetadata(total_num_frames=case["count"], fps=case["fps"], duration=case["duration"])
        inputs = processor(videos=[frames], video_metadata=[metadata], return_tensors="pt")
        captured = []
        hook = model.embed_vision.register_forward_hook(lambda _module, _args, output: captured.append(output.detach().clone()))
        with torch.no_grad():
            outputs = model(**{k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}, output_hidden_states=True)
        hook.remove()
        features = captured[0]
        tokens = outputs.last_hidden_state[0]
        mask = inputs["attention_mask"][0].to(tokens.dtype)
        pooled = torch.nn.functional.normalize((tokens * mask[:, None]).sum(0) / mask.sum(), p=2, dim=0)
        tensors = {"video_features": features, "token_embeddings": tokens, "embedding": pooled}
        width = model.config.text_config.hidden_size
        tensors["hidden_states"] = torch.stack([h[0] for h in outputs.hidden_states if h.shape[-1] == width])
        case["model_tensors"] = {}
        for name, value in tensors.items():
            file = f"{case['name']}.hf.{name}.bin"
            (out / file).write_bytes(value.cpu().numpy().astype("<f4").tobytes())
            case["model_tensors"][name] = {"file": file, "shape": list(value.shape)}
        perframe = []
        with torch.no_grad():
            for pixels, positions in zip(inputs["pixel_values_videos"], inputs["video_position_ids"]):
                result = model.get_video_features(pixels[None], positions[None], torch.tensor([1]))
                perframe.append(result.pooler_output[0])
        perframe = torch.cat(perframe, dim=0)
        case["hf_batched_vs_perframe"] = {
            "unequal_f32": int(torch.count_nonzero(features.view(torch.int32) != perframe.view(torch.int32))),
            "max_abs": float((features - perframe).abs().max()),
        }
        print(case["name"], "HF batched vs perframe:", case["hf_batched_vs_perframe"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--model-dir", type=pathlib.Path, default=pathlib.Path.home() / ".cache/huggingface/hub/models--google--embeddinggemma-2/snapshots" / REVISION)
    parser.add_argument("--model-forward", action="store_true", help="Also capture batched HF model forward and serial-frame vision comparison")
    parser.add_argument("--model-cases", default="decoded_cap,video_ffv1,video_h264",
                        help="Full-model cases; default includes the 70-source-frame -> 32-frame uniform cap")
    args = parser.parse_args()
    versions = {"torch": torch.__version__, "torchvision": torchvision.__version__,
                "transformers": transformers.__version__, "pillow": PIL.__version__, "numpy": np.__version__, "av": av.__version__}
    assert all(versions[k] == v for k, v in {"torch": "2.14.1", "torchvision": "0.29.1", "transformers": "5.19.0", "pillow": "12.3.0", "numpy": "2.4.6", "av": "18.1.0"}.items()), versions
    versions["av_libraries"] = {k: ".".join(map(str, v)) for k, v in av.library_versions.items()}
    for tool in ["ffmpeg", "ffprobe"]:
        versions[tool] = subprocess.run([tool, "-version"], capture_output=True,
                                        text=True, check=True, timeout=5).stdout.splitlines()[0]
    torch.set_num_threads(1)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    video = processor.video_processor
    assert video.max_soft_tokens == 140, "expected pinned checkpoint video settings, without overrides"
    video_settings = {k: getattr(video, k) for k in [
        "patch_size", "max_soft_tokens", "pooling_kernel_size", "rescale_factor"]}
    cases = []
    specs = [("decoded_up", 3, 48, 80, None), ("decoded_down", 2, 503, 701, None),
             ("decoded_exact", 1, 528, 528, None), ("decoded_wide", 2, 16, 997, None),
             ("decoded_cap", 70, 24, 32, None), ("decoded_fps_cap", 90, 24, 32, 2.0),
             ("decoded_slow", 2, 24, 32, 0.5)]
    for name, count, height, width, fps in specs:
        cases.append(save_case(out, processor, name, synthetic(count, height, width), fps))
    for name, codec in [("video_ffv1.mkv", "ffv1"), ("video_h264.mp4", "h264")]:
        frames, fps = encode(out, name, synthetic(24, 48, 80), 8.0, codec)
        case = save_case(out, processor, pathlib.Path(name).stem, frames, fps, name)
        if codec == "h264":
            sampled, metadata = load_video(str(out / name), backend="pyav",
                sample_indices_fn=lambda metadata: processor.video_processor.sample_frames(
                    metadata, fps=1, max_frames=32, overflow_strategy="uniform"))
            direct = processor(videos=[sampled], video_metadata=[metadata],
                               do_sample_frames=False, return_tensors="pt")
            assert np.array_equal(direct["pixel_values_videos"].numpy(), np.fromfile(out / case["pixels_file"], dtype="<f4").reshape(direct["pixel_values_videos"].shape))
            assert direct["input_ids"][0].tolist() == case["input_ids"]
        cases.append(case)
    # An HD fixture proves the container boundary accepts 1920x1080, not just tiny clips.
    frames, fps = encode(out, "video_hd.mkv", synthetic(1, 1080, 1920), 1.0, "ffv1")
    cases.append(save_case(out, processor, "video_hd", frames, fps, "video_hd.mkv"))
    # Real synthetic containers that violate each independent safety limit.
    rejected = []
    for file, count, height, width, fps in [
        ("reject_frame_count.mkv", 3601, 16, 16, 100.0),
        ("reject_frame_rate.mp4", 24, 16, 16, 240.0),
        ("reject_dimensions.mkv", 1, 2048, 2048, 1.0),
        ("reject_duration.mkv", 2, 16, 16, 0.01),
    ]:
        encode(out, file, synthetic(count, height, width), fps,
               "h264" if file.endswith(".mp4") else "ffv1")
        rejected.append(file)
    # Measure rather than assume PIL/torchvision equivalence.
    frame = synthetic(1, 503, 701)[0]
    target = processor.video_processor.aspect_ratio_preserving_resize(
        torch.from_numpy(frame).permute(2, 0, 1)[None], video.patch_size,
        video.max_soft_tokens * video.pooling_kernel_size**2, video.pooling_kernel_size,
        tvf.InterpolationMode.BICUBIC)[0].permute(1, 2, 0).numpy()
    pil = np.asarray(Image.fromarray(frame).resize((target.shape[1], target.shape[0]), Image.Resampling.BICUBIC))
    pil_comparison = {"unequal_bytes": int(np.count_nonzero(pil != target)),
                      "max_abs_byte": int(np.abs(pil.astype(int) - target.astype(int)).max()),
                      "target_shape": list(target.shape)}
    # Test all 256 uint8 intensities directly: f32 multiply vs f64 multiply then cast.
    values = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16)
    rescaled = processor.video_processor.rescale_and_normalize(
        values, True, video.rescale_factor, False, None, None).numpy().reshape(-1)
    float32_path = np.arange(256, dtype=np.float32) * np.float32(video.rescale_factor)
    float64_path = (np.arange(256, dtype=np.float64) * video.rescale_factor).astype(np.float32)
    rescale_comparison = {
        "f32_multiplier_unequal": int(np.count_nonzero(rescaled.view(np.uint32) != float32_path.view(np.uint32))),
        "f64_product_cast_unequal": int(np.count_nonzero(rescaled.view(np.uint32) != float64_path.view(np.uint32))),
    }
    assert rescale_comparison["f32_multiplier_unequal"] == 0
    if args.model_forward:
        model_reference(out, processor, args.model_dir, cases, set(args.model_cases.split(",")))
    manifest = {"revision": REVISION, "versions": versions, "max_soft_tokens": video.max_soft_tokens,
                "video_settings": video_settings,
                "rescale": rescale_comparison, "rejected_containers": rejected,
                "pil_vs_torchvision": pil_comparison, "cases": cases}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"versions": versions, "pil_vs_torchvision": pil_comparison, "rescale": rescale_comparison,
                      "cases": [{"name": c["name"], "sampled": c["sampled_indices"], "soft_tokens": c["num_soft_tokens"]} for c in cases]}, indent=2))


if __name__ == "__main__":
    main()
