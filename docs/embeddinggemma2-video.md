# EmbeddingGemma 2 video boundary and parity

Video uses the optional `embeddinggemma2` feature. CLI:

```sh
vqtrs embed-media --modality video clip.mp4 --no-daemon
```

The `/embeddings/multimodal` API accepts `{"modality":"video","data":"<base64>"}`.
Only inline base64 is accepted: no URLs, data URLs, or caller-local paths. The
existing 16 MiB HTTP body cap still applies (base64 overhead counts toward it).
CLI files are read with an independent 16 MiB encoded-container cap.

## Checkpoint processor contract

**Production and canonical references honor the checkpoint's `video_processor`
configuration: `max_soft_tokens=140`.** Python's unconfigured class default is
70, not the checkpoint default. Only an absent processor configuration file
uses the class fallback. Earlier forced-70 measurements are not canonical
checkpoint parity and are superseded by the results below. The reference oracle
loads `AutoProcessor` unchanged; it does not assign a different video budget.

`embedding_gemma2/video.rs` provides:

```text
decode_container(bytes: &[u8], settings: &ImageSettings)
preprocess_decoded(frames: &[RgbFrame], metadata: Option<VideoMetadata>,
                   settings: &ImageSettings)
```

Both return `Result<PreparedVideo, String>`. Main supplies `model.video_settings`
loaded from `processor_config.json` → `video_processor` (patch size 16, pooling
kernel 3, rescale factor 1/255, token budget 140). Image settings are separate.

- `frames`: frame-major `FramePatches`, with
  `settings.max_patches() × settings.patch_dim()` padded f32 pixels and
  `settings.max_patches() × 2` `(x,y)` positions. At the checkpoint default these
  are **1260×768 pixels and 1260×2 positions** per frame. Padding is zero pixels
  and `[-1,-1]` positions.
- `num_soft_tokens`: actual shared count per frame, at most the configured 140.
  Aspect-ratio rounding can leave unused budget; counts are not hardcoded.
- `sampled_indices`: original zero-based indices, including duplicates.

Default sampling is 1 fps followed by uniform overflow sampling to at most 32
frames, mirroring HF's floor/index/linspace arithmetic. Decoded arrays without
rate or duration skip rate-based sampling, rather than inventing timing. Frames
must have matching dimensions and exact interleaved RGB8 buffer lengths.
Container timing uses actual decoded frame count / average fps, matching HF's
PyAV convention; this is index sampling, not timestamp-based VFR selection.

Each frame expands to `<boi>` + `<|video|>` repeated `num_soft_tokens` times +
`<eoi>`. Blocks are directly concatenated without separators or timestamps.
The model supplies its normal prefix/suffix and replaces video placeholders
with the frame-major concatenation of projected vision features.

Resize reuses the measured torchvision uint8 bicubic antialias implementation
in `image.rs`, not PIL resize. The default-140 503×701 probe resized to 480×624
has 3,240 byte differences from PIL (maximum byte error 1). Rescale is f32
multiplication by f32(1/255): all 256 intensities agree with HF; f64 multiply
then cast differs for 126 intensities. Patch order is patch row, patch column,
pixel row, pixel column, RGB channel, then zero padding.

## Container safety and limits

Install maintained, trusted `ffmpeg` and `ffprobe` binaries on PATH. No shell is
used. The boundary writes inline bytes to a private temporary directory/file,
allows only MOV/MP4 and Matroska/WebM demuxers and the local-file input protocol,
and disables stdin. Automatic display-matrix rotation is disabled, matching
PyAV's decoded RGB orientation. Playlists, network fetches and arbitrary caller
paths are unsupported. FFmpeg's external MOV data references remain disabled
by default. Temporary files are removed on success and errors.

| Resource | Limit |
|---|---:|
| Encoded container | 16 MiB |
| Pixels per source frame | 2,097,152 (1920×1080 fits) |
| Actual source frame count | 3,600 |
| Native average frame rate | 120 fps, finite and positive |
| Declared and actual frame-count/rate duration | 120 seconds, finite and positive |
| Shared subprocess wall-time budget | 30 seconds |
| Metadata stdout / decoded-frame metadata stdout | 16 KiB / 512 KiB |
| Selected RGB stdout | At most 192 MiB |
| Individual FFmpeg decoder allocation | 64 MiB |

A metadata probe validates dimensions/timing/count claims; a bounded decoded
frame probe validates actual count and constant dimensions; decoding selects
only the resulting indices, with bounded raw RGB stdout. **Frame counting
decodes the full source.** Sampling 32 frames does not make a long clip cheap.
Missing duration/rate, changing dimensions, invalid buffers, unsupported formats,
nonfinite timing, oversized output, and subprocess failures are errors.

Settings are validated before decoding: 16-pixel patches, pooling kernel 3,
HF-supported token budgets {70,140,280,560,1120}, and a finite positive rescale
factor at most 1. The checkpoint uses 140; this is trusted model configuration,
not a caller-controlled HTTP override. Prepared pixel/position buffers are
below 119 MiB at 140 tokens (32 padded frames). Larger configured budgets scale
this cost, up to approximately 950 MiB at the supported maximum of 1120.

These are **not** an OS sandbox or hard total-process RSS limit. Raw RGB and
cloned selected frames can coexist, and decoder/library allocations add overhead.
Model attention/forward memory is additional. The wall budget limits subprocess
work, not Rust resize/model inference. Deployment isolation remains the caller's
responsibility.

## Reproducible verification

Use the cached pinned reference environment through `uv`, with isolated Python
imports (`-I`); downloaded model source is not executed:

```sh
uv run --no-project --python <cached-reference-python> python -I \
  scripts/parity/eg2_video_frontend_reference.py \
  --out /tmp/eg2-video-ref-140 --model-forward

EG2_VIDEO_REFERENCE=/tmp/eg2-video-ref-140 cargo test --release -p vqtrs-core \
  --features embeddinggemma2 video::tests -- --include-ignored --nocapture

cargo run --release -p vqtrs-core --features embeddinggemma2 \
  --example eg2_video_parity -- /tmp/eg2-video-ref-140 <checkpoint-directory> cpu

cargo run --release -p vqtrs-core --features embeddinggemma2-metal \
  --example eg2_video_parity -- /tmp/eg2-video-ref-140 <checkpoint-directory> metal
```

The optional third example argument is `cpu` (default), `metal`, or `cuda`.
Device constructors are explicit, with no fallback. CPU requires bit identity
for all tensors. GPU preserves strict IDs/indices/pixels/positions and requires
final embedding maximum absolute error ≤2e-4 and cosine ≥0.99999. Intermediate
GPU tensor differences are reported, not mislabeled as bit-identical. CUDA
requires `embeddinggemma2-cuda` and appropriate hardware; CUDA was not tested.

Oracle: checkpoint revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, CPU f32,
Torch 2.14.1, torchvision 0.29.1, transformers 5.19.0, Pillow 12.3.0, NumPy 2.4.6.
PyAV 18.1.0 is independently pinned. The manifest records the actual checkpoint
video settings, PyAV's linked library versions and both FFmpeg CLI versions.
The reference does **not** read RGB frames from the Rust/FFmpeg decoder: Python
uses PyAV `VideoFrame.to_ndarray`, and Rust uses the FFmpeg CLI. **Raw sampled RGB
bytes are compared before resize.** Current proof uses FFmpeg CLI 8.1.1 and
PyAV libavcodec 62.28.102 / libavformat 62.12.102 / libswscale 9.5.102.

The codec proof covers FFV1/bgr0 in MKV and H.264/yuv420p in MP4 (8 fps, 80×48),
plus FFV1 HD (1 fps, 1920×1080), under those recorded versions. It is not a
guarantee for arbitrary codecs, HDR/color metadata, or other library versions.
Synthetic decoded-array coverage includes upscale/downscale/exact resize,
extreme aspect ratio, uniform cap sampling, and fractional fps/duplicate indices.
Additional synthetic containers exceed actual frame count, rate, duration and
resolution limits. Raw RGB, model tensors and binary containers stay under
`/tmp`; none are checked into git.

HF 5.19's default container loader requires torchcodec, absent from this pinned
environment. MP4 additionally exercises HF's explicit `load_video(backend="pyav")`.
HF's direct PyAV MKV loader trusts `stream.frames`, zero for this generated MKV;
that broken path is not claimed as parity. MKV uses PyAV-decoded RGB arrays with
explicit actual count/rate metadata.

## Measured checkpoint-default results

`--model-forward` captures `decoded_cap`, `video_ffv1`, and `video_h264` by default.
The example checks every case containing model tensors, using manifest counts
rather than fixed budgets, on CPU or the explicitly selected GPU.

**All five CPU forwards** (the 32-frame decoded-array uniform-cap case and both
decoded/container forms of FFV1/MKV and H.264/MP4) match IDs, patches/positions,
video features, every text hidden state, token embeddings and final embeddings
**bit for bit**. The 32-frame case has 70 source frames without timing, **130
actual soft tokens/frame** within the 140 budget, and **4,226 joint tokens**.
Its 30,965,760 pixel floats, 2,129,920 video-feature floats, 51,929,088 hidden-state
floats, and 3,245,568 token-embedding floats have zero differing values. HF's own
batched vision vs independent per-frame calls also match exactly for all three
model-reference cases.

**All five Metal forwards** pass: final maximum errors 2.086e-7 (32-frame cap),
1.239e-7 (FFV1), and 2.138e-6 (H.264), cosine 1.000000000 at nine printed digits.
Intermediate GPU tensors differ, as reported by the runner. This is canonical
pinned checkpoint CPU parity and a measured Metal tolerance result, not a GPU
or cross-platform bit-identity promise.
