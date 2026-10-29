# Vqtrs exact-speed autoresearch

## Objective and stages
Optimize existing engines, not their model identities or numerical contracts. Base: 6ae1b13ed59297d2f1540babf6abd93eb4e06c9d, Rust 1.97.0. Work is isolated because the user's main worktree has unrelated tracker migration changes.

Three DeepSeek research agents were dispatched for ONNX, native/Candle, and accuracy. Upstream HTTP 429 interrupted them; after recovery, all three wrote partial reports in the local research directory. Parent verified source and synthesized this plan. Some agents assumed Linux-style HF cache locations and missed the cached ONNX model in the macOS OS cache directory; original EG2 weights really are absent. Research hypotheses are not qualification.

1. ONNX CPU: benchmark cached all-MiniLM-L6-v2, test intra-op oversubscription first. Current fastembed defaults to every logical CPU. Do not change other engines' defaults based on one model's result. Dense batching/order/token counts must stay exact.
2. Candle EmbeddingGemma2 CPU: investigate repeated RoPE constants/tables and allocation work, never reassociate reductions or change GEMM/bias boundaries. Existing F32 reference bit-identity is mandatory. First require whole-output baseline comparison, then operation golden tests and full parity if assets exist.
3. Native CoreML/MLX: avoid redundant preprocessing/transport work, preserve precision, model identities, limits and fail-closed worker protocol. No packing/padding/batch-semantic changes without neighbor/order isolation and labelled retrieval tests.
4. Qwen3 and CUDA/TensorRT/DirectML: require actual checkpoints/hardware. Never infer qualification from CPU or another model. Ask the user for GPUs before remote runs. Local GPU inference also needs normal dpty resource admission.
5. Sparse/M3/rerank: preserve sparse support/weights, dense vectors, every rerank score and stable order/ties. Cached compatible model availability is a gate; a BGE checkpoint without the expected sparse outputs does not qualify M3.

## Metrics and protocol
Primary for each segment: sum of per-case median warm end-to-end library latency in ms. Track every case, load time and raw timings. No cross-engine aggregation that can hide regression. Unchanged benchmark inputs, binaries/compiler/feature flags, model hashes and runtime versions. Use existing scripts/parity/eg2_cases_text.json, not a newly generated corpus. First six cases form the bounded short suite; the existing long case is a separate gate/workload, not silently truncated by this harness.

Build immutable baseline and candidate binaries. Pair/interleave baseline and candidate repeats on the same host. Warm up each shape. Measure complete count_tokens + embed_batch + synchronized host vectors, not asynchronous enqueue. No result caches. Capture exact F32 bits plus token counts and indices. Compare each candidate to the SAME engine/model/dtype/device baseline. CPU exact candidates require zero changed output bits. Numerical/kernel changes need predeclared existing tolerances and labelled retrieval metrics (two datasets), never relaxed gates. Synthetic rankings are not retrieval-quality proof. Require repeatable gains outside observed noise and reject material regressions in any measured case. Record all crashes and rejects.

## Scope and constraints
Candidates: crates/vqtrs-core/src/{embed,sparse,rerank}.rs and bounded embedding_gemma2 internals after qualification coverage is available. Harness: crates/vqtrs-cli/examples/engine_bench.rs, scripts/benchmarks and justfile. No new production dependencies, quantization, precision/context/dimension/model/prompt changes, dropped work, edited golden values, unrelated files, services, registry publication or pushes. Keep strict lints/toolchain.

Run cargo fmt, cargo check, cargo clippy --workspace --all-targets, cargo test; optional backend changes owe their own feature matrix and numerical gates. Use just recipes. Run ripwire --quality-delta and --edit-check for touched symbols. Keep only scoped verified changes; otherwise restore exact changed hunks, never reset/clean the user's tree.

## Harness operation
The pi-autoresearch skill is installed but init_experiment/run_experiment/log_experiment are NOT registered in this session. Use explicit .auto/measure.sh and .auto/log.jsonl equivalents; do not claim the extension dashboard is running. .auto artifacts/logs and target stay local. Every attempt logs status, metrics and actionable side information. Each kept candidate gets its own commit after checks. Experiments are intentionally reversible and isolated, so an autoresearch branch is appropriate here; no PR/push is implied.

## Environment limits
Local Apple Silicon, 24 logical CPUs, 192 GiB memory; background CPU workloads observed. Initially ~12 GiB free disk. Existing model caches only. New compact no-debug/nonincremental target, no deletion of others' artifacts. GPU and missing checkpoints remain explicit coverage gaps.

## What's been tried
Eight baseline/experiment/verification cycles completed; see docs/benchmarks/exact-speed-cpu-2026-10-29.json and local .auto/log.jsonl. All production candidates were restored exactly. Threads=1 regressed batches ~4x; threads=4 improved aggregate but regressed batches 17%; threads=8 improved short suite 1.37x but failed long/ragged qualification up to 2.05x slower; threads=16 also failed long qualification. Do not retry blanket thread caps without a genuinely different workload-specific configuration strategy. encode_fast token accounting improved its phase ~4% in two runs, but total-time results were near host noise and broad tokenizer-family equivalence lacks coverage: do not claim a kept speedup.

The explicit local 8 GiB dpty lease for larger-engine qualification was REJECTED. Never lower the claim or evict services to get through. Asked user for an authorized GPU host and budget. Original EG2 model.safetensors symlink is broken. ONNX MiniLM is cached under the macOS OS-cache Hugging Face directory. No model downloads, GPU jobs, production restarts, pushes or production optimization commits occurred. Follow-up: vqtrs-9ay9. Loop is paused at a resource/hardware decision, not running unattended.

Harness baseline binary was rebuilt after adding token phase instrumentation at 7aecfaf; the prior thread-segment baseline is separately retained. Do not mix schemas or binary segments. Keep both immutable and record hashes before resuming.
