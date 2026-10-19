# Resident EmbeddingGemma 2 text readout

`EmbeddingGemma2::embed_text_tensor(text)` is an additive library seam for a
same-device downstream learned head. It performs the same complete tokenization,
text tower, mean pooling and normalization as `embed_text`, returning the normalized
F32 `candle_core::Tensor` on the device explicitly supplied to `from_dir_on`.

There is no pooled-vector or intermediate-token-state download in this method.
It does not perform the vector API's host-side finite-value check: callers must
validate the final readout or reduce a validation status on device. This is not a
fallback, truncation option or independently qualified GPU numerical profile.

Existing vector-returning APIs, multimodal paths, finite-value checks and services
are unchanged. A complete real-checkpoint CPU tensor/vector bit-equivalence probe
is maintained by the local downstream dcsnmkr prototype; GPU fidelity requires its
own complete checkpoint/backend gate.
