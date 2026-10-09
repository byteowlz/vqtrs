# Targeted Apple runtime probes

Developer qualification only, not production backends. Source/model pins and
results are in `docs/embeddinggemma2-apple-runtimes.md`. Keep raw datasets, model
assets, inputs, vectors and process diagnostics outside Git.

## Prerequisites

- Normal resource admission for the complete scope; the recorded pass used16 GiB.
  Pausing a service requires authorization and subsequent restoration.
- The pinned Google snapshot and existing text/image/audio/video F32 oracle files.
- CoreML assets at `3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4`, including all three
  packages, `embeddings.bf16` and `position_embeddings.f16`.
- Audited, unmodified mlx-vlm checkout at
  `4f4634bb813c0298cb1467bed2e957526c71d0b4`. Do not run checkpoint Python or
  repository installers. The MLX probe deliberately skips package initializers
  and imports only the audited tensor-definition path; this is a developer probe,
  **not the proposed production binding/install design**.
- Private Python3.12 environment managed with `uv`: the recorded pass used
  MLX0.32.3, NumPy2.5.3, coremltools9.0, soundfile0.13.1, Pillow12.3.0 and
  tokenizers0.22.2.

Set `EG2_BENCH_WORK_DIR` (default `/tmp/vqtrs-native-media`), `EG2_MODEL_DIR` and
`EG2_MLX_VLM_DIR`. Optional `EG2_COREML_DIR`, `EG2_REFERENCE_DIR`,
`EG2_VIDEO_REFERENCE_DIR` and `EG2_MEDIA_DIR` select local assets/oracles.
Run Python through `uv run --no-project <private-python> -I`, never directly.

Run inference arms sequentially through `../apple-text/guard.py`, supplying a
unique run name followed by the private Python executable, `-I`, probe and mode.
The watchdog retains the8 GiB sampled ownership limit,14 GiB host-free floor and
900-second deadline. Its samples exclude shared compiler/ANE allocations; they
are not total memory or guaranteed allocation peaks.

## Media and isolation

- `coreml_probe.py image|audio|video`: unchanged image/video F32 oracle patches
  and positions, original synthetic16 kHz WAV for the in-graph audio frontend.
  Outputs native vectors and maxabs/cosine/norm diagnostics. The4226-token video
  explicitly reports unsupported context, not truncation or fallback.
- `mlx_regular_probe.py`: seven images, five audio and three video oracle cases.
  Strict original-checkpoint key/shape/BF16 audit; bounded allocator/cache;
  sequential video encoding, not turbo/W8A8. Media activation promotion follows
  the pinned source and is not claimed to be pure BF16 arithmetic.
- `mlx_extra.py`: ragged/reordered batches and a short-audio padding diagnostic.
- `coreml_audio_pipeline.py`: four synthetic clips, two warmups/three paired
  trials, one GPU producer and ordered text consumer with two-window lookahead.
  Do not infer M5 speedups or actual device traces from this M2 probe.

## Bounded labelled retrieval

Download public BEIR `scifact.zip` and `nfcorpus.zip` into the private work directory
from `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/`.
`prepare_retrieval.py` chooses the first16 sorted test queries, all their relevant
rows and96 seeded distractors per dataset. It explicitly prepares identical
<=512-token texts; this is **not backend truncation or a full BEIR benchmark**.
Selected archive entries are bounded; fixture bounds, token identity and source
hashes are checked. CoreML files must match `assets-3f978ddf.json`.

1. Run `prepare_retrieval.py`.
2. Build the `eg2_text_qualify` Rust example with `embeddinggemma2-metal`.
3. Through the watchdog, export one Metal F32 pass from `retrieval-inputs.json`
   into `retrieval-metal.json`, then CPU F32 from `retrieval-cpu-spot.json` into
   `retrieval-cpu-spot-results.json`. Arguments: snapshot, `metal|cpu`, fixture,
   output. The example refuses >2048 rows, >16 MiB fixtures, >16 KiB text,
   >512 tokens and token mismatches; no download or device fallback.
4. Run `mlx_regular_probe.py text` and `coreml_text_qualify.py`. The latter also
   checks fixed buckets,511/512/513 boundaries and packed ordering.
5. Run `summarize_retrieval.py`: nDCG/recall/MRR@10, top10 overlap, numerical
   diagnostics and the unchanged CPU/Metal F32 gate. Reduced-precision acceptance
   remains a separate decision; these scripts do not declare it approved.

JSON outputs include raw vectors/labels in the private work directory. Commit
only aggregate, public-safe reports. Source/code changes require rerunning the
relevant checks; prototype results do not establish library/CLI/HTTP/UDS readiness.
