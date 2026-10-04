# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Fixed

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

