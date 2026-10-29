# Accuracy-preserving performance experiments

Optimize each engine against its own immutable model/runtime/dtype/device baseline.
Never compare a faster reduced-precision engine with F32 and call it lossless.
The existing [CPU parity](embeddinggemma2-parity.md), [GPU tolerance](embeddinggemma2-gpu.md)
and [native runtime](embeddinggemma2-native.md) contracts remain unchanged.

## First experiments

1. Measure ONNX intra-op thread oversubscription with the cached MiniLM model.
   Do not generalize one model/CPU result to sparse, M3, reranking or other EPs.
2. Test repeated Candle RoPE/mask construction and allocation overhead without
   changing arithmetic order. CPU changes owe bit-exact outputs and golden tests.
3. Native workers: prioritize byte-identical preprocessing/IPC changes. Packing,
   padding width, sorting and batching can change numerical results: existing
   native measurements already demonstrate this. Do not accept these on latency
   alone or introduce a tolerated accuracy loss under a lossless objective.
4. Qualify Qwen3 and accelerator combinations on actual hardware/checkpoints.
   Missing coverage is a skip, not a pass. Obtain resource admission before GPU
   jobs; never stop unrelated services or purge their caches for a benchmark.

## Dense differential harness

```sh
just bench-engine Qdrant/all-MiniLM-L6-v2-onnx /tmp/baseline.json short
just test-bench-compare
uv run --no-project python scripts/benchmarks/compare_exact.py \
  --baseline /tmp/baseline.json --candidate /tmp/candidate.json
```

Use separate baseline/candidate binaries, identical build features and existing
cached assets. Record hashes, platform/runtime/compiler, thread settings, fixture
hash, background contention and actual iteration counts. The example does not
prevent the engine from downloading an uncached model: preflight assets first.
Do not point tests at production servers.

The workload reuses `scripts/parity/eg2_cases_text.json`. `short` explicitly skips
`t_long`; `all` includes it without harness-side truncation. Backend token/context
policies still apply. Three warmups per shape; five timed samples, each averaging
calls until at least 200 ms of measured time. Timings include token accounting,
inference, host-vector completion and finite/output-bit materialization, but not
model loading (reported separately). Token accounting has its own phase timing,
so savings there are not mistaken for a faster inference kernel.
Every repeated result must have identical
bits and token counts. Singleton, ragged batch and reversed batch outputs are
recorded separately: a batch need not equal singleton arithmetic, but each must
match its own baseline. This strict harness is intended for deterministic,
byte-preserving changes; it is not a substitute for tolerance-qualified GPU tests.

The comparator fails on changed model/backend identity, case set/order, row
counts, token counts, F32 output bits, nondeterministic repeats or invalid timing.
Its primary metric is the sum of per-case median warm latencies; it also reports
all case ratios. It does not decide whether a speedup is statistically real.
Pair/interleave repeats, inspect noise and reject material per-workload
regressions instead of hiding them in an aggregate. Run long/boundary and transport
qualification separately before broader claims. No result cache is permitted.

These are existing synthetic parity fixtures, not labelled relevance datasets.
Exact output matching is a strong compatibility gate on the measured workload,
not proof of general retrieval accuracy. Changed kernels/batching/precision need
broader differential and labelled retrieval tests; never loosen goldens to keep
an experiment. Preserve sparse support/weights and rerank scores/ties too.

## Isolated loop state

Use an isolated reversible worktree for experiments. `.auto/prompt.md` records
the playbook; `measure.sh` runs baseline/candidate/candidate/baseline suites;
`checks.sh` runs format/check/strict Clippy/tests and comparator regression tests.
The ignored `.auto/results`, `.auto/bin`, `.auto/target` and `.auto/log.jsonl`
contain local artifacts. Do not check in weights, machine paths or private inputs.
If pi's experiment tools are not registered, record explicit manual loop results
rather than claiming that the extension/dashboard is running. Only checked,
reproducible improvements are eligible to keep; no push is implied.
