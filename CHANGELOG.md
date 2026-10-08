# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Fixed

- `VQTRS_ONNX_CPU_ONLY=1` explicitly bypasses compiled ONNX GPU providers and
  selects CPU for dense/sparse/M3/reranking. It offers a workaround for noisy
  macOS CoreML execution without hiding diagnostics or disabling Candle
  Qwen3/EmbeddingGemma Metal/CUDA. Defaults are unchanged; subprocess regression
  tests verify selection without mutating global process environment.
- `just install-all` now includes EmbeddingGemma 2 by default, with an explicit
  opt-out and CPU/CUDA/Metal features selected independently of Qwen3. Both
  binaries install with `--locked`. Shared candle CUDA compatibility handling
  also applies when only Gemma is selected. ONNX-only CPU installs no longer
  exit early under `set -e` or macOS Bash 3.2's empty-array `nounset` behavior.
  Offline installer regression tests run through `just test-installer` and
  `just test`; default Cargo builds and `just install` remain ONNX-only.

## [0.2.0] - 2026-10-07

### Added

- **core: optional EmbeddingGemma 2 text, image and audio backend (vqtrs-3jhq).**
  Direct `EmbeddingGemma2` / `Gemma2Input` library API, CPU f32, 768-dimensional
  normalized embeddings. Five audio fixtures compare 150 tensors bit-for-bit
  against the pinned Apple Silicon Python reference, with no differences;
  text and image parity remain covered. Includes a conformer audio tower,
  reference-matched WAV/mel preprocessing, offline regression tests and
  reproducible parity tooling. Catalog/Engine support, `vqtrs embed-media`
  and a bounded `/embeddings/multimodal` HTTP/UDS route are included; live
  integration checks also match the reference byte-for-byte. The feature stays
  off by default; see `docs/embeddinggemma2-parity.md` for the CPU numerical contract.
- **EmbeddingGemma 2 CUDA/Metal execution (vqtrs-v7nq).** Independent
  `embeddinggemma2-cuda` / `embeddinggemma2-metal` features, explicit device
  constructors, native GPU norms/attention/conformer operations and a warm
  benchmark harness. CPU bit identity is preserved; GPU parity is measured
  with tolerances, not claimed byte-identical. See `docs/embeddinggemma2-gpu.md`
  for verified full-model Metal/RTX 4090 CUDA results and decoder-platform qualifications.
- **EmbeddingGemma 2 video (vqtrs-1a8h).** Typed decoded RGB frames and bounded
  inline-container decoding through FFmpeg, exposed through Engine, CLI and
  HTTP/UDS. Sampling follows checkpoint defaults and preprocessing uses
  torchvision-compatible uint8 bicubic/rescaling, not the PIL image path.
  Independent PyAV codec and CPU/Metal/CUDA model parity tooling covers sampled
  frames, tokens, intermediate states and normalized output. See
  `docs/embeddinggemma2-video.md` for codec/platform and resource limits.

- Standard MIT `LICENSE` for vqtrs; upstream port/model licenses remain separate.
- `just install-mac-eg2` preserves CoreML/Qwen3 Metal while enabling the
  EmbeddingGemma 2 multimodal Metal backend in both binaries. Default builds
  and release archives remain ONNX-only; GPU/media support is opt-in.

### Fixed

- ONNX provider construction is strict-Clippy clean with acceleration enabled,
  including the combined CoreML/Qwen3 Metal/EmbeddingGemma Metal build;
  provider order and CPU fallback are unchanged.
- **api: the server no longer stops answering under concurrent or large embedding
  batches (vqtrs-76mr).** Inference tasks handed to `spawn_blocking` were
  unbounded, so a burst of requests queued on the engine mutex until tokio's
  blocking pool (512 threads) saturated; every later request then waited forever
  for a thread while the process sat idle, and recovery required a restart.
  Requests now pass an in-flight semaphore *before* the request body is read:
  beyond `max_inflight` (default 4) they are shed with `429` + `Retry-After` and
  `Connection: close`, so rejected work costs no body read or parse and the
  blocking pool can never saturate. Batches above `max_batch_texts` (default
  256) are rejected with `413`. `/health` reports `503` + `Retry-After` while
  every slot is busy, so an external probe can tell a saturated server from a
  healthy one. Both knobs are settable via config file (`max_inflight`,
  `max_batch_texts`) and CLI/env (`--max-inflight` / `VQTRS_MAX_INFLIGHT`,
  `--max-batch-texts` / `VQTRS_MAX_BATCH_TEXTS`). Resident memory is bounded by
  `max_inflight x max_batch_texts` (the ONNX runtime retains arena memory for
  the largest batch shape it has run).
- scripts/loadtest_embeddings.py reproduces the reported triggers (8 concurrent
  clients, back-to-back 512-text batches, a retry pile-up burst) and asserts the
  pass criteria: probes served throughout and after the run, overload rejected
  rather than hung, recovery inside a budget, RSS inside a budget.

