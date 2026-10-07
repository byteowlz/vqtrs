# EmbeddingGemma 2 backend and parity

The optional `vqtrs-core/embeddinggemma2` feature exposes `EmbeddingGemma2`
and `Gemma2Input::{Text, Image, Audio}`. It loads the text, vision and audio
towers from `google/embeddinggemma-2` on CPU in f32 and returns a normalized
768-dimensional vector. No prompt is added to text; supply the checkpoint's
query/document prompt yourself.

```rust,ignore
use vqtrs_core::{EmbeddingGemma2, Gemma2Input};

let model = EmbeddingGemma2::from_hf("google/embeddinggemma-2")?;
let text = model.embed(Gemma2Input::Text("task: search result | query: a bird"))?;
let image = model.embed(Gemma2Input::Image(&std::fs::read("bird.png")?))?;
let audio = model.embed(Gemma2Input::Audio(&std::fs::read("bird.wav")?))?;
```

The model is in the normal catalog and `Engine::load("google/embeddinggemma-2")`
supports text plus `Engine::embed_multimodal`. Both binaries expose the
`embeddinggemma2` feature. Build/install with `--features embeddinggemma2` or
`just install-eg2`; it stays off by default. Video is tracked in `vqtrs-1a8h`
and is not implemented.

## CLI and HTTP

```bash
vqtrs embed --model google/embeddinggemma-2 "task: search result | query: a bird"
vqtrs embed-media --modality image bird.png
vqtrs embed-media --modality audio bird.wav other.wav --no-daemon
```

`embed-media` uses a warm daemon over its Unix socket when available; otherwise
it loads locally. Each file is a separate embedding, in input order.

`POST /v1/embeddings` accepts text as usual. The feature-gated
`POST /embeddings/multimodal` accepts a mixed input list:

```json
{
  "model": "google/embeddinggemma-2",
  "input": [
    {"modality": "text", "text": "task: search result | query: a bird"},
    {"modality": "image", "data": "<standard base64-encoded image bytes>"},
    {"modality": "audio", "data": "<standard base64-encoded WAV bytes>"}
  ]
}
```

The model defaults to EmbeddingGemma 2 on this route. It returns an embedding
list with `object`, `model`, `data[{index,object,embedding}]`; media token usage
is omitted, not estimated. URLs, file paths, data-URI prefixes and unknown
fields are not supported. The server never opens client paths or fetches media
URLs. Invalid media returns 400; schema errors return 422. Requests retain the
existing in-flight limit (429) and item cap (`max_batch_texts`, 413); this route
has a 16 MiB JSON body limit. A worker retains its permit even if its HTTP
client disconnects.

## Verified numerical contract

Bit identity is measured, not promised across platforms, BLAS versions, thread
settings or Python releases. The verified reference is:

- Checkpoint revision `914f7f89142e33e77833254d9c9b90c3cef7303b`.
- Apple Silicon macOS, CPU f32, Accelerate BLAS.
- torch 2.14.1, transformers 5.19.0, NumPy 2.4.6, Pillow 12.3.0,
  sentence-transformers 6.1.0; one PyTorch thread, one input at a time.
- HF text/vision eager attention, with ST mean pooling (including prompts)
  and L2 normalization. Audio uses boolean masks from the `sdpa` configuration;
  its tower computes attention explicitly, without an SDPA kernel.

The eager HF path is the hard parity gate. The public
`SentenceTransformer.encode` path uses default SDPA text attention and can
differ in the last bits even inside Python; its comparison is reported
separately and does not count as bit identity.

Verified fixtures:

| Modality | Cases | Compared tensors | Differing tensors |
| --- | ---: | ---: | ---: |
| Text | 7 | 189 | 0 |
| Image | 7 | 210 | 0 |
| Audio | 5 | 150 | 0 |

Audio covers a float32 chirp, silence, a short clip, PCM16 speech and a
31.5-second clip truncated to the reference's 30-second limit. A separate chirp
trace compares subsampling and all twelve audio layers: 43 tensors, zero
differences. The frontend probe compares 25,700 FFT components, 12,850
magnitudes, 320 window coefficients, 32,896 filter coefficients and 6,400
float64 GEMM results, all exactly equal.

## Input limits

- Joint sequences are capped at 8,192 tokens, including BOS/EOS and media
  placeholders. This CPU implementation materializes attention matrices;
  it does not implement the checkpoint's full 262,144-token context efficiently.
  Longer input returns an error rather than silently truncating text.
- Images are capped at 16,777,216 pixels. Dimensions are checked before
  allocating decoded pixel/MCU buffers.

## Audio limits and implementation details

- RIFF/WAVE only: PCM8/16/24/32 or IEEE float32/64, mono or stereo, 16 kHz.
  Stereo is averaged; resample/convert other formats before calling.
- Clips longer than 480,000 samples are truncated. Clips without a valid
  analysis frame, malformed headers/chunks and non-finite samples return
  errors. No decoder subprocess is used.
- The fixed frontend uses a 320-sample periodic Hann window, hop 160,
  512-point real FFT and 128 HTK mel bins. Mask multiplication preserves
  negative zero as in the reference.
- NumPy's tested float32 RFFT matches its double-precision transform narrowed
  to complex64. The Rust port computes in f64 then narrows. Complex magnitude
  follows NumPy's SIMD ratio/FMA algorithm; the mel GEMM stays in f64.
- LLVM's constant-base `pow(10, x)` specialization differs from NumPy's libm
  `pow` at one filter centre. A `black_box` base keeps the reference operation.
- CPU LayerNorm follows PyTorch's vectorized Welford/cascade arithmetic.
  The depthwise convolution's RMSNorm uses the channel-major outer reduction,
  not the contiguous row reduction.
- The biased output projection uses a safe ndarray beta=1 GEMM on macOS,
  accumulating the bias as PyTorch's `addmm` does. Separate GEMM plus bias
  addition changes rounding. ndarray is already a fastembed/ort dependency;
  its BLAS feature is enabled only for this optional backend on macOS.
- Other platforms use candle's CPU GEMM and have no bit-identity guarantee.

Ports retain their upstream notices/licenses beside the source; see
`crates/vqtrs-core/src/embedding_gemma2/NOTICE.md`.
This software is based in part on the work of the Independent JPEG Group.

## Reproduce

Generate deterministic media with `scripts/parity/eg2_make_media.py` in the
reference Python environment (its Pillow version matters). Media, weights
and raw dumps stay outside git. Run scripts with `python -I` via uv.

```bash
just check-eg2
just eg2-reference scripts/parity/eg2_cases_audio.json /tmp/eg2-parity/ref-audio
just eg2-parity /tmp/eg2-parity/ref-audio /path/to/checkpoint scripts/parity/eg2_cases_audio.json
just eg2-frontend-reference /tmp/eg2-parity/mel
EG2_MEL_DIR=/tmp/eg2-parity/mel cargo test --release -p vqtrs-core \
  --features embeddinggemma2 matches_numpy -- --ignored --nocapture
EG2_MEL_DIR=/tmp/eg2-parity/mel cargo test --release -p vqtrs-core \
  --features embeddinggemma2 tables_match_numpy -- --ignored --nocapture
EG2_AUDIO_REF_DIR=/tmp/eg2-parity/ref-audio \
  EG2_AUDIO_CASES="$PWD/scripts/parity/eg2_cases_audio.json" \
  cargo test --release -p vqtrs-core --features embeddinggemma2 \
  features_match_reference -- --ignored --nocapture
```

Use `--audio-trace` with `eg2_reference.py` on a one-case audio file for
module-level dumps. Set `EG2_AUDIO_TRACE_DIR` and `EG2_MODEL_DIR` to run the
ignored `audio_stage_probe` regression test. Normal workspace tests use no
network, model downloads or Python. `scripts/parity/eg2_api_smoke.py` checks a
disposable server's mixed-media/text HTTP and CLI/UDS output byte-for-byte
against these reference dumps, plus malformed input, unsupported URLs/paths,
batch limits and post-request health. Its server fixture needs
`max_batch_texts=3`; do not use the production instance.
