# Reusable native CPU mechanics

With `embeddinggemma2`, `vqtrs-core` exposes mechanics without loading an embedding model:

- `decode_rgb_image(bytes, max_pixels)`: the existing native RGB decoder, returned as
  `Gemma2VideoFrame`; validates the caller's pixel budget before pixel allocation
  (hard ceiling 16,777,216). Existing embedding calls keep their original budget.
- `resize_rgb_image`: validated native uint8 antialiased bicubic resize. Source, output
  and horizontal intermediate images are bounded to 16,777,216 pixels; malformed
  buffers and zero/overflowing dimensions reject before allocation.
- `torch_cpu::apply(F32Op, values)`: existing vector/scalar SLEEF sine/cosine, tanh GELU
  and scalar-base power operations, without changing operation boundaries.
- `torch_cpu::bf16_layer_norm_accumulation`: paired BF16-input/F32 Welford streams and
  affine arithmetic over lossless F32 representations. The caller supplies BF16-
  representable values/parameters and owns the final BF16 narrowing. Invalid inputs
  reject before modifying the buffer. This is not the F32 audio subsampler norm.

These seams support a separately pinned native decision-model vision port. They
provide no model weights, policy defaults, network access, GPU opening or inference
fallback. Existing embedding model selection and outputs are unchanged. Qualification
of another model, dtype policy, runtime or hardware is not inherited.

The original F32 norm golden tests remain unchanged. Additional paired-stream tests
cover biased norms and vector/tail widths 8, 9, 31, 32, 128 and 1152 against ARM CPU
Torch 2.13.0 `cf30153c4c131c8164ee7798e5022d810682e2cb`, one thread. Replays compare
all narrowed output bits using the same word-digest convention as the existing tests.
See the [Torch notice](../crates/vqtrs-core/src/embedding_gemma2/LICENSE-TORCH.txt).

```sh
cargo test --offline --locked -p vqtrs-core --no-default-features --features embeddinggemma2
cargo clippy --offline --locked --workspace --all-targets \
  --features vqtrs-api/embeddinggemma2,vqtrs-cli/embeddinggemma2
```
