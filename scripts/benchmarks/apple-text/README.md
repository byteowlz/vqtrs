# Preliminary Apple text benchmark harness

Developer experiments only; no native backend, installer or default dependencies.
See [protocol, pins and results](../../../docs/embeddinggemma2-apple-runtimes.md).
Do not treat the synthetic rankings or sampled footprints as qualification.

Before execution, audit the pinned candidate source and obtain normal dpty
admission for the actual scope. The first pass reserved 8 GiB for sequential text
<=512 tokens; it did not qualify the rejected 16 GiB full suite. `guard.py` samples
owned task footprints and stops its process group above 8 GiB or below 14 GiB host
free memory. It is not a hard allocation limit; system CoreML compiler daemons are
not task descendants. `EG2_BENCH_TIMEOUT_SECS` sets a finite deadline up to900s
(default900); timeout terminates/reaps only the owned process group. No script
stops services, purges caches or obtains admission.

Set `EG2_MODEL_DIR` to the original pinned snapshot, `EG2_MLX_CANDIDATE_DIR` to the
pinned audited MLX checkout, and optionally `EG2_BENCH_WORK_DIR` (default
`/tmp/vqtrs-apple-bench`). Use a dedicated `uv` environment: Python3.12.13,
MLX0.32.3, NumPy2.5.3, tokenizers0.22.2 and safetensors0.7.0 were tested.

1. Run `prepare.py` through `uv` to produce the shared prepared-text/token-ID fixture.
2. Build `cargo build --release -p vqtrs-core --features embeddinggemma2-metal
   --example eg2_text_bench`. Run CPU and Metal separately through `guard.py`:
   `eg2_text_bench <snapshot> cpu|metal <fixtures.json> <output.json>`.
3. Run `mlx_probe.py` through `uv` and the guard. It uses unchanged candidate
   kernels/`build_state` with a direct audited checkpoint-to-object adapter;
   inference caching is off, output768, mode edge, precision BF16. Conversion
   is separate from loading. Do not import downloaded checkpoint Python.
4. Fetch the pinned CoreML package and BF16 table as data into `<work>/coreml`.
   Compile `swiftc -O -parse-as-library Manager.swift CoreMLMain.swift -o <binary>`
   and run through the guard. The modified MIT FluidUse manager retains its
   license here: downloads removed, concurrency1, prevalidated fixture token IDs
   substituted for the tokenizer. Thus Swift timing **excludes tokenization and
   transport**, unlike the Candle/MLX prepared-text timings. `--plans` enables
   optional placement diagnostics. The initial resource-constrained query stalled;
   a separate `PlanMain.swift` query succeeded after the authorized temporary
   service pause and16GiB admission. Compile it separately (without another `@main`)
   and run with a90s guard deadline. It reports static preferred-device counts for
   `embed_32`, not actual runtime execution or proof for every function. Do not
   infer actual ANE execution from `cpuAndNeuralEngine` alone.

Three warmups/five measured trials; final host vectors synchronize device work.
Run arms sequentially, but use paired/interleaved repeats for qualification.
Record actual rows/tokens, artifact/source hashes, versions, precision, complete
errors and cache state. Compare numerical error and labelled task quality before
approving any reduced-precision mode. Default Candle contracts remain unchanged.

Only synthetic fixtures/safe summaries belong in Git. Keep downloaded weights,
private inputs, compiled caches and raw process diagnostics outside the repo.
