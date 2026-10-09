# Apple runtime extensions: benchmark before integration

Status: user-approved direction; preliminary M2 text measurements recorded,
**not implemented or benchmark-qualified**.
Tracker: `vqtrs-s7zm`; benchmark `vqtrs-2g84`; CoreML `vqtrs-8fe9`; MLX
`vqtrs-6dqt`. Both implementation issues are blocked by the benchmark.

## Boundaries

Keep the ONNX-only default build and the existing Candle F32 implementation.
The original `google/embeddinggemma-2` selection must not silently change runtime
when optional features are compiled. Existing CPU byte-identity and GPU tolerance
contracts in [parity](embeddinggemma2-parity.md) and
[GPU verification](embeddinggemma2-gpu.md) remain unchanged.

Proposed independent Cargo features, only to be wired after qualification:

- `embeddinggemma2-coreml`: native CoreML models, not the existing `coreml` feature
  which registers ONNX Runtime's CoreML execution provider.
- `embeddinggemma2-mlx`: MLX/Metal; BF16 and hardware-qualified W8A8 must be explicit,
  distinct precision modes, not automatic quantization.

A native CoreML text graph can target the dedicated Apple Neural Engine (ANE).
The M5 Neural Accelerators used by the MLX project are inside the **GPU**, not
that ANE. Neither is evidence of speed on a different chip. CoreML image/audio
encoders in the linked export use GPU execution; "100% ANE" is a text claim.
The ONNX CPU-only switch does not select either new runtime or disable Candle.

## Pinned candidates

These are audit/benchmark pins, not production trust or licensing approval:

| Candidate | Revision | Role |
| --- | --- | --- |
| [FluidInference/embeddinggemma-2-coreml](https://huggingface.co/FluidInference/embeddinggemma-2-coreml/tree/f9d567d5f36c0a98773cdba26034b8185a353834) | `f9d567d5f36c0a98773cdba26034b8185a353834` | Converted assets/config, BF16 token table and FP16 graphs |
| [FluidUse](https://github.com/FluidInference/FluidUse/tree/2c42578f119e52598cdf4eddaee7aca8be2757f0) | `2c42578f119e52598cdf4eddaee7aca8be2757f0` | Swift loader/reference; includes other decision-model runtimes |
| [octaviusp MLX CLI](https://github.com/octaviusp/embedding-gemma-2-cli-mlx-accelerated/tree/3575aef259be9e797a757a555c42528ecd049b58) | `3575aef259be9e797a757a555c42528ecd049b58` | BF16/W8A8 kernels, batching, memory controls and reference fallback |
| Original Google checkpoint | `914f7f89142e33e77833254d9c9b90c3cef7303b` | Existing F32 oracle; confirm converted assets' actual source revision |

Do not execute repository install scripts or downloaded checkpoint Python.
Audit dependencies/entry points, retain applicable notices, hash model files and
record tokenizer/config/weight provenance before evaluating. Source-model lineage
and exact checkpoint equivalence still need verification for the converted assets.

## Known differences to test, not conceal

- CoreML text functions have fixed buckets 32/48/64/128/256/512; packed inference
  has eight output slots. This differs from our 8192-token contract. Overflow must
  fail explicitly, never silently truncate or automatically split/average vectors.
- The published audio graph uses 10-second windows, versus our 30-second frontend.
  Longer audio and video token sequences need an explicit supported contract or
  a clear unsupported error. A vision encoder alone is not our full video path.
- CoreML FP16 has reported drift, especially quiet audio tokens. Author cosine
  results do not meet or replace our existing F32 numerical gate automatically.
- MLX turbo handles text and single-image rows; its source delegates audio/video
  and multi-image rows to the regular MLX backend. Do not advertise those as W8A8
  acceleration or perform an identity-invisible fallback in vqtrs.
- The MLX CLI defaults to 512-dimensional output; the primary comparison must use
  our 768 dimensions. Matryoshka experiments are separate, re-normalized modes.
- Upstream MLX uses maskless, length-sorted batches and pooling/projection changes.
  Test padding, order, prefixes, sliding-window boundaries and long contexts;
  algebraic equivalence is not floating-point byte identity.

## Benchmark protocol

1. Obtain normal resource admission on the actual Apple host. Run one model/arm
   at a time; do not purge, stop unrelated workloads or lower claims to get through.
2. Record chip, OS, runtime/compiler versions, source/asset hashes, device selection,
   precision, actual token lengths, output dimensions and background contention.
   Inspect CoreML placement plans separately from actual device execution evidence.
3. Compare Candle CPU/Metal F32, native CoreML FP16 and MLX BF16 edge/throughput.
   Run W8A8 only when its Metal/device capability is verified; otherwise mark skipped.
   M2 Ultra results cannot qualify M5-specific kernels, or vice versa.
4. Use identical public/synthetic fixtures and query/document prefixes. Include
   single requests and batches, short/medium/512-token inputs, ragged batches,
   mixed languages, code, empty text, overflow and modality-limit cases. Verify
   tokenization before attributing differences to kernels.
5. Report fresh-process loading and shader/CoreML compilation separately from
   warmed inference. Synchronize GPU completion, disable result caches, pair and
   interleave trials, report sample counts and actual measured rows. Do not relabel
   extrapolated throughput or author benchmarks as local measurements.
6. Separate preprocessing, model time and complete library/HTTP/UDS request time.
   Record process footprint, live tensor allocations, allocator cache and device
   allocation where available. RSS alone is not total unified-memory use.
7. Compare finite normalized vectors, dimensions, max absolute error and cosine.
   Also assess retrieval on at least two labelled datasets: nDCG, recall, MRR and
   top-k overlap. Keep raw public dataset/weights outside Git; commit safe summaries,
   protocols and hashes. Small fixture rankings do not establish general quality.
8. Review results before implementing each mode. Any reduced-precision acceptance
   criterion is separate from the unchanged F32 contract; do not silently relax
   max-error/cosine gates to get a new backend accepted.

## Integration gate

The existing seam is `catalog::{Backend,Resolved}` and `embed::{Inner,Engine}`,
with Candle internals behind `embedding_gemma2`. Extend the loaded-engine seam;
do not put native runtime details into API handlers or add retrieval/index logic.

Resolve a distinct explicit runtime/precision identity before registry lookup.
Include that identity and artifact/config revision in cache keys and response/run
provenance; model ID and vector width alone do not make indices interchangeable.
Decide the exact selection API after benchmark feasibility, not via a fake feature
flag today. No silent CPU/backend/precision retry after allocation or inference
failure. Unsupported OS/device/shape/modality must give an actionable error.

Audit safe Rust binding feasibility under `unsafe_code = forbid`; do not add
workspace exemptions. If a bounded native helper is necessary, review ownership,
framing, time/output limits, cancellation/reaping and dependency/install behavior
before choosing it. Optional backends must not introduce Swift/Python/MLX into the
default build or execute arbitrary checkpoint code.

Qualification must cover library, local CLI, warm UDS and HTTP; ordering, finite
outputs, registry limits, worker permits after client disconnect and malformed
input behavior. Run default, each optional feature and supported Apple feature
combinations through format/check/strict Clippy/tests. Keep current transports and
Candle numerical fixtures unchanged.

## Jev / decision-model relevance

Native CoreML fixed-shape encoder graphs are a promising route for small
Jev-style decision models. FluidUse also exposes Laya and other typed-decision
runtimes, but this does not demonstrate that our Kev/Jev checkpoint runs on ANE.
Jev benchmarking belongs to the decision-runtime owner, **not vqtrs**.

Preserve the exact decision head, question/candidate template, temperature and
Choice/Noul/Score postprocessing. Measure accuracy, Brier/ECE calibration, threshold
behavior, option-order invariance, cold/warm end-to-end latency and co-resident GPU
contention on the target Mac. Embedding cosine and texts/second do not prove
classifier calibration. ANE and GPU share memory/bandwidth, so ANE placement is not
proof of zero interference. Suggestions/probabilities never grant tool permissions.

## Initial 8 GiB text pass

2026-10-09: source/model revisions pinned and shared engine seams inspected.
The 16 GiB full-suite reservation remained rejected after the user deliberately
parked Kev. Investigation found native Metal allocations already included in host
use but not credited against the running recipe floor. `dpty-n0nc`, source
`f060b9c`, adds bounded native owned-graphics observations, excluding swapped and
reclaimable bytes; missing attribution retains the floor. No installed dpty binary
or daemon was replaced/restarted. The built CLI observed about 10.5 GiB admissible,
not enough for the original full-suite claim.

The scope was explicitly narrowed to **8 GiB, sequential text-only, <=512 tokens**,
then normally reserved and released. No other service was stopped or cache purged.
Candle CPU/Metal F32 and MLX turbo BF16 edge completed seven synthetic cases,
55 actual rows per trial, three warmups and five synchronized trials. macOS
26.6.2/25G83, M2 Ultra 192 GiB; an existing large MLX service stayed running.
These were sequential arm runs, **not interleaved trials** or transport benchmarks.

| Case | Rows / tokens per row | CPU F32 mean ms | Metal F32 mean ms | MLX BF16 edge mean ms |
| --- | --- | ---: | ---: | ---: |
| Short query | 1 / 16 | 62.05 | 34.32 | 22.52 |
| Short queries | 8 / 16 | 474.85 | 245.66 | 25.48 |
| Documents | 8 / 22–33 | 590.08 | 256.28 | 50.73 |
| Medium text | 1 / 120 | 203.63 | 35.01 | 27.38 |
| Near-cap text | 1 / 463 | 780.05 | 63.40 | 38.36 |
| Bulk documents | 32 / 27–38 | 2589.21 | 1005.58 | 74.57 |

The bulk case measured approximately 12.4 / 31.8 / 429.2 rows/s, respectively;
MLX's batching is part of that difference, not just kernel speed. No extrapolation.
Token IDs match the original tokenizer and output is 768-dimensional. Metal's
worst max-absolute difference versus CPU was `1.8533e-7`; MLX BF16's was
`0.00179115`, minimum cosine `0.99992868`. **MLX BF16 fails the existing F32 GPU
gate** (`2e-4` / `0.99999`); faster throughput is not approval to relax it.
All three ranked four authored queries correctly against eight authored documents;
that toy MRR/recall result does not establish general retrieval quality.

Sampled peak owned process footprints were 4.65 / 4.81 / 0.57 GiB, respectively.
These are 1-second samples, not guaranteed allocation peaks. MLX allocator peak is
reported separately. Its converted text state used the candidate's unchanged
`build_state`, with an audited direct original-checkpoint adapter to avoid loading
multimodal/Transformers dependencies. Process loading is separate from warm time;
OS/compiler cache state was not controlled.

**CoreML remains unmeasured:** the pinned package compiled and text32/48/64
functions loaded. `MLComputePlan` querying stalled; a subsequent inference-only
attempt was stopped while loading128 when host free memory fell below the
14 GiB safety floor. No completed vector, inference timing or verified ANE
placement is claimed. Its BF16 token table is byte-identical to the original
Google table, but that does not prove converted graph-weight lineage.
M5 W8A8, media, 8192-token contexts, HTTP/UDS, two real labelled datasets,
co-resident contention and reduced-precision acceptance remain unqualified.

Inspect the [safe machine-readable report](benchmarks/apple-text-2026-10-09.json),
[synthetic fixture](benchmarks/apple-text-fixtures.json) and
[developer harness](../scripts/benchmarks/apple-text/README.md). Integration issues
remain blocked. Resume CoreML/full qualification only with adequate admission;
M5-specific qualification additionally needs an M5 host.

## Admitted CoreML follow-up after temporary Sushi pause

The user subsequently authorized pausing another service for the tests. Sushi was
temporarily parked, a normal **16 GiB** reservation was admitted, and all text
arms were remeasured with Sushi stopped. EAVS/forum remained running. The lease
was released; Sushi was unparked/requested and `/v1/models` returned200 afterward.
Kev remains parked. No daemon replacement/restart or shared-cache purge occurred.
The standing queued vqtrs request also auto-started under normal scheduler
admission; after the tests its health/model-list endpoints returned200. This was
not a new API restart request or model-inference verification.

CoreML now completed both single and packed FP16 inference. Package compilation
was 2.32s; loading/initializing seven functions took124.12s, including device
compilation work. These are not guaranteed cold-start numbers: earlier attempts
and OS compiler caches were not cleared. A separate `embed_32` compute-plan query
completed and preferred ANE for all3,862 reported operations. That is **static
placement evidence for that function**, not a collected runtime execution trace
or proof for every bucket.

| Case | CPU F32 ms | Metal F32 ms | MLX BF16 edge ms | CoreML single FP16 ms | CoreML packed FP16 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| One16-token query | 62.81 | 33.41 | 21.57 | 5.67 | 18.93 |
| Eight16-token queries | 479.84 | 255.52 | 24.66 | 43.90 | 19.10 |
| Eight documents | 590.72 | 248.29 | 50.04 | 44.29 | 19.22 |
| One120-token text | 205.33 | 33.47 | 26.57 | 8.89 | 18.97 |
| One463-token text | 793.07 | 61.79 | 38.93 | 40.90 | 40.92 |
| 32 documents | 2611.60 | 993.33 | 73.29 | 179.31 | 76.58 |

Warm means,3 warmups/5 trials; actual row counts, no extrapolation. These are
**not directly interchangeable E2E timings**: Swift starts with pretokenized IDs,
including table gathering/prediction/host copy but excluding tokenizer/transport;
Candle and MLX include tokenizing prepared text. Runs remain sequential, not
interleaved. Single-query CoreML and bulk MLX look promising, not universally best.

CoreML single worst maxabs=`0.00157288`, min cosine=`0.99996441`; packed
maxabs=`0.00154997`, min cosine=`0.99996619`. Both **fail the unchanged F32 gate**.
FP16 output unit-norm error reached `0.00093748`; an explicit F32 renormalization
contract may be needed, not hidden postprocessing. Toy retrieval still ranks all
four labelled queries first, not general task-quality evidence. Model source/
graph lineage, reduced-precision approval, two labelled datasets, runtime traces,
media/long-context, transport and integration isolation remain pending.

The [follow-up report](benchmarks/apple-text-coreml-followup-2026-10-09.json)
preserves both precision/mode arms and all five trial timings. Its sampled worker
footprint does not include shared system CoreML compiler daemons and must not be
advertised as total CoreML memory. Native features remain unimplemented/blocked.
