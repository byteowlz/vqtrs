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

## Initial CPU results

The [2026-10-29 summary](benchmarks/exact-speed-cpu-2026-10-29.json) records eight
baseline/experiment/verification cycles. No production optimization was retained.
An eight-thread ONNX cap improved the short MiniLM suite **1.37x**, with identical
output bits, but failed long/ragged qualification (up to **2.05x slower**).
One/four/sixteen-thread caps also failed workload gates. Skipping token offsets
improved token-accounting time about 4%, but overall gains were near host noise
and tokenizer-family coverage was missing. Both ideas were restored, not shipped.

Default format/check/strict-Clippy/tests and comparator tests passed. Larger
Candle/native/GPU qualification could not proceed: normal local memory admission
was rejected and original EG2 checkpoint weights were absent. Hardware/resource
authorization is pending; untested engines are not qualified by these CPU results.
No model downloads, service changes, deployments or pushes were performed.

## CUDA readout result

The [2026-10-29 CUDA summary](benchmarks/exact-speed-cuda-2026-10-29.json) records
one kept original-Candle text optimization: defer final host readout in groups
of at most eight independent CUDA forwards. Per-text kernels, precision, inputs
and normalization are unchanged. It is not padded/arithmetic model batching.

On pinned GB10/F32, immutable-binary ABBA runs improved the fixture suite about
7%; two same-process paired repeats measured **1.094x** aggregate speedup and
**1.136x** on eight short queries, with **zero changed output bits**. Singletons
retain their original path; their timing variation is reported, not hidden.
CPU/Metal and native/other engines are intentionally unchanged. All seven existing
cases/55 rows passed token identity, library and HTTP/CLI/UDS exact-vector checks.
Boundaries, reversal, duplicate rows, long ragged inputs, oversize/recovery, finite
checks and Compute Sanitizer also passed. See the report for complete scope limits.

```sh
just bench-eg2-readout /path/to/pinned/checkpoint /tmp/readout-paired
just test-eg2-cuda-readout /path/to/pinned/checkpoint
```

Obtain resource admission first. The ARM GB10 build used the installed CUDA13
compiler explicitly and `RUSTFLAGS="-C target-feature=+fp16"` for an existing
ARM dependency's instruction requirements, identically for both arms; model
tensors remain F32. Do not silently change toolchains or infer other hardware's
qualification. The paired benchmark emits baseline/candidate JSON and enforces
exact output bits during every timed trial. Raw vectors/profiles stay local.

## Round two: bounded CUDA positional constants

The [round-two report](benchmarks/exact-speed-cuda-round2-2026-10-29.json) compares
against the **already optimized** readout-eight build. Readout groups16/32 were
rejected as noise. Reusing immutable CUDA positional-table prefixes measured
**1.051x additional speedup** across two ABBA repeats with zero changed output
bits. Medium/near-cap text improved about16%/13%; singleton short queries were
unchanged. A separate direct comparison measured **1.150x versus the original
serial build**, not an extrapolated multiplication.

This caches constants, never inference results. At most512 positional rows per
attention type are retained:3 MiB for the pinned checkpoint, maximum4 MiB. Both
head widths must be positive, <=512 and divisible by8, with finite positive
RoPE theta values. The width guard preserves the original vectorized sine/cosine
versus scalar-tail rounding behavior. Missing/incompatible parameters, larger
sequences and non-CUDA devices retain the original path. Readout size stays8.

Exact geometry/fallback tests, real-model boundary/order/error tests,7 synthetic
images,5 synthetic audio cases and a two-frame video passed. HTTP/CLI/UDS output
and token counts stayed exact. Geometry and full-model Compute Sanitizer runs
reported zero errors. Pinned synthetic generation/evaluation artifacts remain
outside Git; the report records hashes, bounds and qualification limitations.
The non-evicting12 GiB lease was released; no services were deployed or restarted.

## Round three: explicit long-context qualification

The new diagnostic preserves all seven original cases/55 rows and adds four
cases/five rows derived by whole-text repetition of the existing medium fixture.
Actual long lengths are 953, 2024 and 2262 tokens, plus a ragged pair. No
truncation, model-input policy or production context limit changes. Its new
sum-of-medians metric must not be compared with earlier short-only suite sums.

```sh
just bench-eg2-long-fixture /path/to/pinned/checkpoint /tmp/eg2-long.json
just bench-eg2-cuda-text /path/to/pinned/checkpoint /tmp/eg2-long.json /tmp/capture.json 4096
```

This benchmark cap is explicit; the default remains 512. Token IDs are checked
against the engine before timing. The generator pins the original fixture hash
and tokenizer dependency. Freeze both implementations before interleaved runs.

Increasing the CUDA positional bound from 512 to 4096 improves this extended
suite about 4% against the round-two implementation; short cases are unchanged.
The intermediate 2048-row candidate improved about 2.8% but left the 2262-token
case uncached. The larger bound uses 24 MiB for the pinned model, maximum 32 MiB,
versus the earlier 3/4 MiB. Width/scalar-tail guards, non-CUDA behavior and the
8192-token production cap are unchanged; sequences above 4096 use original
construction. This is a warm-inference optimization, not a free-memory claim.

The [final report](benchmarks/exact-speed-cuda-round3-2026-10-29.json) also retains
a smaller allocation-only change: fill the mask with the same `f32::MIN` values,
then write positive zero to each row's accepted interval. An independent scalar
cell oracle verifies exact bytes. For 2262 rows, host vector capacity falls from
9,265,152 to 5,116,644 floats; construction-only probes improve from about 7.7 ms
to 0.32 ms. This is not a whole-engine speedup: its isolated full-workload gain
is only about 0.3%, although that one case repeatedly improves about 3%.

Combined, two final immutable-baseline ABBA repeats measure **1.046x** across all
11 cases and **1.082x** on the four long cases. Original short cases remain
unchanged within variation; every observed case ratio is reported. All 60 text
rows, HTTP/CLI/UDS, 13 seeded media cases, boundary/error tests and geometry/full
model memchecks remain exact/clean. The host mask builder is shared, but no
CPU/Metal full-engine performance improvement is claimed.

Final median model initialization was **1682 ms baseline / 1739 ms candidate**:
about 57 ms more, with OS/compiler caches left intact. Keep this cold-load and
21 MiB canonical persistent-memory trade-off visible; the win is warm inference.
These gains cannot be multiplied into the earlier 15% short-suite result because
the workload changed. Frozen evaluation version `vqtrs-exact-readout-eval/v3`
records the derived fixture, paired captures and script/binary provenance.
The non-evicting lease is released; no service deployment or push occurred.

## Isolated loop state

Use an isolated reversible worktree for experiments. `.auto/prompt.md` records
the playbook; `measure.sh` runs baseline/candidate/candidate/baseline suites;
`checks.sh` runs format/check/strict Clippy/tests and comparator regression tests.
The ignored `.auto/results`, `.auto/bin`, `.auto/target` and `.auto/log.jsonl`
contain local artifacts. Do not check in weights, machine paths or private inputs.
If pi's experiment tools are not registered, record explicit manual loop results
rather than claiming that the extension/dashboard is running. Only checked,
reproducible improvements are eligible to keep; no push is implied.
