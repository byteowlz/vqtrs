# EmbeddingGemma 2 GPU execution

GPU is isolated from ONNX and Qwen3 acceleration. Both binaries forward these
optional features; all towers remain f32:

```bash
# NVIDIA
cargo build --release --workspace --features embeddinggemma2-cuda
# Apple Silicon (compose with existing backends when installing)
cargo build --release --workspace --features coreml,qwen3-metal,embeddinggemma2-metal
```

Generic `cuda`, `coreml`, and `qwen3-*` do not automatically select a GPU for
EmbeddingGemma. `from_hf` and normal Engine loading prefer its own enabled CUDA
feature, then its own Metal feature, then CPU if GPU initialization fails.
Allocation/inference errors are returned, not silently rerun on CPU.

`from_dir` intentionally defaults to CPU for reference reproducibility. Use
`EmbeddingGemma2::from_dir_on(dir, &Device::new_cuda(0)?)`, `from_hf_on`, or the
corresponding Metal device for explicit selection (no fallback). `device()`
reports the actual device. CPU selection remains available in GPU builds.

## Execution boundary

Weights and text/vision/conformer inference stay on the selected device.
GPU norms, GELU, softmax, pooling, subsampling/LayerNorm, attention, GLU,
causal depthwise convolution and biased output projection use native tensor
kernels. The depthwise convolution is a batched GEMM, not hundreds of separate
host/group calls. Input masks and small positional/scale tables are host-built;
there are no per-layer downloads of activation tensors in the normal embed path.

Tokenization, image/video decoding/resizing, and the NumPy-compatible f64
WAV/mel frontend remain CPU preprocessing. Metal has no f64 tensor support.
Only completed input features are uploaded and the final 768-float vector is
returned to the host. `forward` is a diagnostic API and intentionally captures
intermediate outputs; `embed` avoids those copies.

The 8,192-token safety cap still applies. Full checkpoint context, Flash
Attention, quantization and f16/bf16 inference are not implemented here.
F32 weights occupy roughly 3 GiB plus context, attention and decoder workspace;
reserve at least 5 GiB on a dedicated CUDA device for the parity workloads.

## Numerical verification

GPU reductions/GEMMs/transcendentals do not reproduce CPU evaluation order.
**GPU byte identity is not claimed.** The test gate preserves exact
preprocessing, token IDs, masks and frame selection; reports all intermediate
differences; and requires final maximum absolute error <=2e-4 and cosine
>=0.99999 against the pinned CPU oracle described in
[the reference contract](embeddinggemma2-parity.md).

Verified on Apple Silicon Metal, f32:

- All 19 original text/image/audio cases pass. Largest final-vector error
  9.555e-7; all printed cosines 1.000000000. CPU execution in the same GPU
  build still matches all 549 original reference tensors bit for bit.
- Five GPU operation/conformer synthetic tests pass. The same tests pass
  on NVIDIA RTX 4090, CUDA toolkit 12.8; they assert GPU residency rather than
  accepting a silent CPU fallback.
- CUDA workspace/all-target check and strict Clippy pass on Linux/NVIDIA.
  **Full-checkpoint CUDA vector parity is not yet verified:** the available
  GPU's existing workload leaves insufficient free memory. Do not conflate
  compilation/small-kernel tests with full-model validation.
- Video model/codec evidence is recorded separately in
  [video parity](embeddinggemma2-video.md).

A local warm five-iteration image benchmark (`photo_1037x761.png`) measured
mean CPU 11,660 ms versus Metal 534 ms, approximately 22x for that fixture.
This includes preprocessing and synchronized output copying, excludes model
loading and three warmups, and is not a general speedup guarantee.

## Reproduce

Use the same raw dumps/checkpoint as the CPU tests. Device selection is
explicit, so an unavailable GPU fails rather than yielding a CPU result:

```bash
cargo run --release -p vqtrs-core --features embeddinggemma2-metal \
  --example eg2_parity -- --device metal --reference /tmp/eg2-parity/ref-audio \
  --model-dir /path/to/checkpoint --cases scripts/parity/eg2_cases_audio.json
cargo test --release -p vqtrs-core --features embeddinggemma2-metal \
  gpu_ -- --ignored --nocapture
cargo run --release -p vqtrs-core --features embeddinggemma2-metal \
  --example eg2_bench -- /path/to/checkpoint metal image /path/to/image.png 10
```

Substitute `embeddinggemma2-cuda` / `cuda` on NVIDIA, or `cpu` as the example
argument to verify the original CPU contract. GPU tests require actual hardware;
normal workspace tests remain offline and hardware-independent.
