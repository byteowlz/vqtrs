# Experimental native Apple backends

Two independently opt-in runtimes are available on Apple Silicon macOS. They do
not replace `google/embeddinggemma-2` (Candle F32) or enable ONNX acceleration.
The default Cargo build remains ONNX-only.

| Feature | Exact model selection | Contract |
| --- | --- | --- |
| `embeddinggemma2-coreml` | `vqtrs/embeddinggemma-2-coreml-fp16-3f978ddf` | CoreML FP16 graphs/BF16 token table; macOS15+;512 joint tokens;<=10s audio |
| `embeddinggemma2-mlx` | `vqtrs/embeddinggemma-2-mlx-regular-bf16-4f4634bb` | Regular MLX/original BF16 weights, source-defined activation promotion;8192 joint tokens;<=30s audio |

Both return finite, explicitly host-normalized768-dimensional F32 vectors. The
normalization computes a stable F64 norm, rounds its denominator to F32, and
divides in F32. It does **not** make the underlying inference F32-equivalent.
No dimensionality reduction, quantization, automatic prefixes or backend retry
is performed. Text is embedded as supplied; query/document prefixes are the
caller's responsibility. Keep each model identity in a separate index; do not
silently reuse a Candle index.

## Capabilities and limits

Both support text, PNG/JPEG/WebP images,16kHz mono/stereo WAV, decoded RGB video
frames and the existing bounded FFmpeg container path. Inputs are independent,
not combined conversations. Image/video/audio preprocessing is the existing
Rust frontend; native workers receive only prepared private tensors.

- CoreML: GPU-preferred vision/audio followed by NE-preferred text, **not**
  all-ANE media. Compute-unit preferences are not actual placement traces.
  Text uses single-row fixed functions32/48/64/128/256/512, not packed inference.
  Video must fit512 tokens after canonical1fps/32-frame sampling. Overflow is
  rejected, not further downsampled or delegated to another backend.
- Regular MLX: all modalities use the regular backend, never turbo or W8A8.
  Video encodes frames sequentially before joint pooling. Text batches have
  at most16 rows when each row<=512 tokens; longer rows run individually.
  The allocator limit is6GiB with512MiB cache; an allocation failure is an error,
  not permission to change runtime/precision.
- Both reject audio longer than their window, rather than silently truncating
  it. Explicitly submit separate clips if windowed search is wanted; vqtrs
  does not invent an averaging/aggregation contract. Audio needs at least161
  samples (just over10ms) to produce a valid frame.
- Existing image-pixel and video decoder/output/time limits still apply. Native
  text is nonempty and<=256KiB and cannot contain untyped media placeholders.

**Known caveat:** CoreML's synthetic300ms sine clip differed substantially more
than other measured cases (cosine0.99237/maxabs0.01634 against F32). Its cause and
general audio task impact remain unestablished. Prefer Candle F32 or regular MLX
for precision-sensitive short/quiet audio. See the
[bounded qualification evidence](embeddinggemma2-apple-runtimes.md); small labelled
subsets and synthetic fixtures are not general quality or M5 qualification.

The user accepted these explicitly experimental choices with documented drift,
not a default replacement. The measured<=1percentage-point task loss on bounded
retrieval subsets is an experimental evaluation criterion, not a production
quality guarantee. Existing F32 gates remain unchanged.

## Explicit installation

Cargo embeds the worker adapter, not a Python environment, weights or external
MLX source. No automatic Python installation or checkpoint-code execution occurs.
The native runtime is reached through optional owned Python helper processes,
not an in-process Rust CoreML binding. `unsafe_code=forbid` remains in force.

```sh
cargo build --workspace --features embeddinggemma2-coreml,embeddinggemma2-mlx
uv venv --python 3.12 ~/.cache/vqtrs/native-env
uv pip install --python ~/.cache/vqtrs/native-env/bin/python \
  'numpy==2.5.3' 'coremltools==9.0' 'mlx==0.32.3' 'pillow==12.3.0'
export VQTRS_NATIVE_PYTHON="$HOME/.cache/vqtrs/native-env/bin/python"
```

Only install dependencies for the chosen runtime: CoreML does not need MLX, and
MLX does not need coremltools. `uv` must be available on the launching process's
PATH. The private Python executable must be absolute. Download **data only**:

- `VQTRS_EG2_COREML_DIR`: absolute directory containing the14 pinned files at
  [FluidInference revision3f978ddf](https://huggingface.co/FluidInference/embeddinggemma-2-coreml/tree/3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4).
  Include all three `.mlpackage` trees, token/position tables, config, tokenizer
  and model card. `coreml-assets.json` authenticates every file by size/SHA256.
- `VQTRS_EG2_GOOGLE_DIR`: absolute original Google snapshot at
  `914f7f89142e33e77833254d9c9b90c3cef7303b`, containing config, tokenizer,
  processor config and weights authenticated by `mlx-assets.json`.
  Normal Hugging Face blob-cache symlinks are supported.
- `VQTRS_EG2_MLX_SOURCE`: absolute audited mlx-vlm checkout at
  [4f4634bb](https://github.com/Blaizzy/mlx-vlm/tree/4f4634bb813c0298cb1467bed2e957526c71d0b4).
  No repository installer is run. All Python source files must match
  `mlx-source.json`, including the file set. Authenticated `.py` files are copied
  into the private worker directory so checkout `.so`/`.pyc` files cannot shadow
  them. Only tensor definitions are imported; processor/package initializers,
  hub loading, generation and checkpoint Python are bypassed.

Use the source's applicable license and Google's model terms; vqtrs' MIT license
is not a model relicense. Compile each feature independently if only one is
needed. Other platforms return an explicit unsupported-platform error.

## Library, CLI and API

```sh
vqtrs embed --no-daemon \
  -m vqtrs/embeddinggemma-2-coreml-fp16-3f978ddf 'task: search result | query: sunshine'
vqtrs embed-media --no-daemon --modality audio \
  -m vqtrs/embeddinggemma-2-mlx-regular-bf16-4f4634bb clip.wav
```

In the library, use `Engine::load(exact_model)` followed by `embed_batch` or
`embed_multimodal`. The existing HTTP `/v1/embeddings` and
`/embeddings/multimodal` and warm UDS use the same exact model selection and
return that identity. Catalog descriptions expose runtime/precision/limits.
An unavailable feature, wrong artifact or runtime failure cannot select Candle.
For native models, a responding daemon's error is not silently retried locally;
`--no-daemon` explicitly selects a fresh local load.

## Ownership and reproduction

One serialized worker/process group belongs to each loaded native engine. Rust
creates private tensor directories, bounds protocol lines to512KiB, validates
identity/request sequence/count/dimensions/finiteness/norm and retains media until
inference completes. Startup has a15-minute deadline (framework compilation);
requests have5minutes. Failures kill/reap the owned group and disable it until an
explicit model unload/reload. Drop reaps workers and removes private files.
A worker monitors owner-pipe closure even while native inference runs. These
limits are not an OS sandbox or total RSS/ANE-memory guarantee. Abrupt parent
SIGKILL can leave private temporary files; do not claim crash-proof cleanup.
Existing API permits remain held in blocking inference after client disconnect.

Resource admission still belongs to the serving/placement owner. Integration
checks used a normally admitted16GiB lease with a temporary authorized Sushi
pause, then restoration; never stop unrelated services to reproduce them.

- `eg2_native_smoke` Rust example: real library text/image/audio/decoded-video,
  repeated vectors, overflow and original Metal F32 fixture diagnostics.
- `scripts/parity/eg2_native_smoke.py`: real HTTP, warm UDS with local loading
  disabled, local CLI, mixed/reversed order and malformed/limit/error recovery.
- `scripts/parity/test_eg2_native_worker.py`: offline hash/cache-symlink,
  tensor/output validation and owner-pipe shutdown regressions.

The committed aggregate integration report contains synthetic metrics only;
weights, vectors and private process diagnostics remain outside Git. No registry
publication, installed-binary replacement, production endpoint rollout or original
MacBook-warning resolution is implied.
