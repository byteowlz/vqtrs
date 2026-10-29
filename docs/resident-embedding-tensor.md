# Resident EmbeddingGemma 2 text readout

`EmbeddingGemma2::embed_text_tensor(text)` is an additive library seam for a
same-device downstream learned head. It performs the same complete tokenization,
text tower, mean pooling and normalization as `embed_text`, returning the normalized
F32 `candle_core::Tensor` on the device explicitly supplied to `from_dir_on`.

There is no pooled-vector or intermediate-token-state download in this method.
It does not perform the vector API's host-side finite-value check: callers must
validate the final readout or reduce a validation status on device. This is not a
fallback, truncation option or independently qualified GPU numerical profile.

`EmbeddingGemma2::embed_text_batch(texts)` reuses this seam only for CUDA batches:
it queues independent complete forwards, then stacks and reads at most eight
final vectors together. It preserves input order and the vector API's finite-value
check. It does not pad, pack or change the inference math. Singletons and other
devices retain sequential vector readout; multimodal paths are unchanged.
`Engine::embed_batch` uses this method for the original Candle backend, so existing
CLI/HTTP/UDS text routes benefit without a transport change. Pinned GB10 exactness,
performance, ordering/error and memory-sanitizer evidence is recorded in
[the CUDA experiment summary](benchmarks/exact-speed-cuda-2026-10-29.json).

The tensor-returning API's caller-owned validation contract is unchanged. A complete real-checkpoint CPU tensor/vector bit-equivalence probe
is maintained by the local downstream dcsnmkr prototype; GPU fidelity requires its
own complete checkpoint/backend gate.
