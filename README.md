# vqtrs

A lean, fast, local **embeddings + reranking** engine. It does embeddings and
nothing else — one modular provider you can drop in front of anything.

- **`vqtrs-core`** — the engine. Wraps [`fastembed`](https://github.com/Anush008/fastembed-rs)
  with two backends: ONNX (via `ort`) for the catalog of small, fast
  sentence-transformers, and — behind the opt-in `qwen3` feature — the candle
  backend for the SOTA open-weight **Qwen3-Embedding** models (0.6B / 4B / 8B).
  Plus cross-encoder **reranking** (BGE / Jina).
- **`vqtrs`** (CLI) — one-shot `embed` / `rerank` / `models` over the engine.
- **`vqtrs-api`** (server) — OpenAI-compatible `/v1/embeddings` plus a `/rerank`
  endpoint.

The opt-in **EmbeddingGemma 2** backend (`embeddinggemma2`) embeds text,
images, 16 kHz WAV audio and video through the core library, CLI and HTTP API.
CPU reference parity, CUDA/Metal acceleration and bounded FFmpeg decoding are
optional. See [usage and numerical contract](docs/embeddinggemma2-parity.md),
[GPU tests](docs/embeddinggemma2-gpu.md) and [video limits](docs/embeddinggemma2-video.md).
Native [Apple runtime experiments](docs/embeddinggemma2-apple-runtimes.md) are
benchmark-first proposals, not supported backends.

## Quick start

```bash
cargo build
cargo test -p vqtrs-core        # engine tests (no network)
```

Install the `vqtrs` CLI + `vqtrs-api` server:

```bash
just install          # CPU / ONNX only
just install-all      # interactive: GPU + optional Qwen3 + Gemma 2 (default yes)
just install-cuda     # NVIDIA: ONNX + Qwen3 on GPU (CUDA <= 13.2)
just install-cuda13   # NVIDIA: same, for CUDA 13.3+
just install-mac      # Apple Silicon: ONNX CoreML + Qwen3 Metal
just install-mac-eg2  # CoreML + Qwen3 Metal + multimodal EmbeddingGemma Metal
just install-eg2      # CPU EmbeddingGemma 2: text, images, 16 kHz WAV, video
```

`install-all` enables EmbeddingGemma 2 by default (`[Y/n]`) in **both** binaries,
using Metal on Apple, CUDA on NVIDIA, or CPU without acceleration. Qwen3 remains
optional; on Apple with Qwen3 enabled the features are
`coreml,qwen3-metal,embeddinggemma2-metal`. Answer `n` to both backend prompts for
an ONNX-only install. Installs use `--locked`; model weights download on first
use, not during installation. Video containers require `ffmpeg` and `ffprobe`.
`just install` and default Cargo builds remain ONNX-only.

### CLI

```bash
# Embed (defaults to all-MiniLM-L6-v2; downloads on first use)
vqtrs embed "the cat sat on the mat" "quantum chromodynamics"

# Pick a model; pipe from stdin (one item per line)
cat docs.txt | vqtrs embed --model BAAI/bge-m3 --pretty

# Top open-weight model via the candle backend
vqtrs embed --model Qwen/Qwen3-Embedding-0.6B "hello"

# Sparse (SPLADE) embeddings — {indices, values} per text
vqtrs sparse "the cat sat on the mat"

# Joint dense + sparse in one BGE-M3 pass
vqtrs m3 "the cat sat on the mat" --pretty

# Rerank documents against a query
vqtrs rerank --query "how do I cook pasta?" --return-documents \
  "Boil water, add pasta, cook 10 minutes." \
  "The stock market fell today."

# List available models (add --json for machine output)
vqtrs models

# Pre-download a model: fuzzy-pick with fzf, or pass an exact code
vqtrs pull
vqtrs pull BAAI/bge-m3
```

Output is JSON on stdout (`--pretty` to indent), so it pipes cleanly into other
tools.

**Warm by default:** if a `vqtrs service` daemon is running, `embed` / `sparse` /
`m3` / `rerank` transparently route to it over its Unix socket — the model stays
loaded, so repeated calls skip the per-invocation load. Passing an explicit
`--model` (or `--no-daemon`) forces in-process loading instead.

### Server

Run it directly, or manage it as a daemon through the CLI:

```bash
vqtrs service run     --port 3000     # foreground
vqtrs service start   --port 3000     # background (detached, logged)
vqtrs service status                  # running pid/port (+ systemd state)
vqtrs service restart
vqtrs service stop
vqtrs service enable  --port 3000     # install + enable a systemd user unit (autostart)
vqtrs service disable                 # disable + remove the unit
```

`start`/`run`/`restart`/`enable` accept `--host`, `--port`, and `--model` /
`--rerank-model` / `--sparse-model` / `--m3-model` overrides. Background state
lives under `$XDG_STATE_HOME/vqtrs/` (pid, port, `vqtrs-api.log`). `enable`
writes `~/.config/systemd/user/vqtrs.service`. (`vqtrs-api …` still works as a
plain foreground binary.)

The server also serves every route over a **Unix domain socket** (default
`$XDG_RUNTIME_DIR/vqtrs.sock`, else the temp dir) alongside TCP — local, no port
management, filesystem-permission auth. The CLI uses it for the warm-routing
above; other local clients can too:

```bash
curl --unix-socket "$XDG_RUNTIME_DIR/vqtrs.sock" \
  http://localhost/v1/embeddings -H 'content-type: application/json' \
  -d '{"input":["hello"]}'
```

Override with `--socket <path>` / `VQTRS_SOCKET`, or disable with `--no-socket`.

**Multi-model gateway:** the server honours each request's `model` field —
models load on demand and stay warm, so a single server serves many models (and
the CLI routes any `--model` to it). Preload a set with `warm` and cap the rest
with `max_loaded` in the config. `usage.prompt_tokens` is a real tokenizer count.

OpenAI-compatible embeddings — works with any OpenAI client by pointing
`base_url` at it:

```bash
curl http://localhost:3000/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"model":"any","input":["hello","world"]}'
```

Sparse and joint dense+sparse (BGE-M3) share the same `{input}` body:

```bash
curl http://localhost:3000/embeddings/sparse -H 'content-type: application/json' \
  -d '{"input":["hello world"]}'          # -> {data:[{index, indices, values}], model}
curl http://localhost:3000/embeddings/m3 -H 'content-type: application/json' \
  -d '{"input":"hello world"}'            # -> {data:[{index, dense, sparse:{indices,values}}], model}
```

Reranking (Cohere/Jina shape):

```bash
curl http://localhost:3000/rerank \
  -H 'content-type: application/json' \
  -d '{"query":"pasta recipe","documents":["boil water and add pasta","tax law"],"return_documents":true}'
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/embeddings`,
`POST /embeddings/sparse`, `POST /embeddings/m3`, `POST /rerank`.

Configure the server three ways (CLI flag / env > config file > default): flags
and env (`VQTRS_MODEL`, `VQTRS_RERANK_MODEL`, `VQTRS_SPARSE_MODEL`,
`VQTRS_M3_MODEL`, `VQTRS_PORT`, …), or a TOML config at
`~/.config/vqtrs/config.toml` (or `--config <path>`). See
[`examples/config.toml`](examples/config.toml); `vqtrs-api --print-schema` emits
the JSON schema. The reranker, sparse, and BGE-M3 models load lazily on first
request.

## Library

```rust
use vqtrs_core::{Engine, Reranker};

let engine = Engine::load("Qwen/Qwen3-Embedding-0.6B")?;
let vectors = engine.embed_batch(&["hello".to_owned(), "world".to_owned()])?;

let reranker = Reranker::load("BAAI/bge-reranker-base")?;
let ranked = reranker.rerank("query", &docs, /* return_documents */ false, /* top_k */ Some(5))?;
# Ok::<(), vqtrs_core::VqtrsError>(())
```

`Engine` and `Reranker` take `&self` for inference and are `Send + Sync` — load
once, share behind an `Arc`. All models — ONNX and Qwen3 — download into the
standard Hugging Face hub cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else
`~/.cache/huggingface/hub`), shared with other HF tooling.

## Models

- **Embeddings** — ~30 ONNX models (MiniLM, BGE, BGE-M3, GTE, E5, Nomic,
  ModernBERT, Snowflake Arctic, Jina, …) plus Qwen3-Embedding 0.6B/4B/8B (candle).
- **Sparse** — SPLADE++ and BGE-M3 sparse; BGE-M3 also yields dense+sparse jointly.
- **Reranking** — BGE reranker base / v2-m3, Jina reranker v1-turbo / v2-base.

Run `vqtrs models` for the full catalog with dimensions and backends.

### Build features

The default build is **ONNX-only** — lean and fast to compile. The Qwen3 models
run on the candle backend, which is heavier, so it's gated behind the `qwen3`
feature (off by default):

```bash
cargo build                                    # ONNX only
cargo build --features qwen3                   # + Qwen3-Embedding
cargo install --path crates/vqtrs-cli --features qwen3
```

Without it, loading a Qwen3 model returns a clear error telling you to rebuild
with `--features qwen3`.

**GPU / acceleration** is feature-gated per backend (all off by default):

| feature | backend | platform |
|---|---|---|
| `cuda` | ONNX (ort CUDA EP) | NVIDIA |
| `tensorrt` | ONNX (ort TensorRT EP) | NVIDIA |
| `coreml` | ONNX (ort CoreML EP) | macOS |
| `directml` | ONNX (ort DirectML EP) | Windows |
| `qwen3-cuda` | Qwen3 (candle CUDA) | NVIDIA |
| `qwen3-metal` | Qwen3 (candle Metal) | macOS |
| `embeddinggemma2-cuda` | EmbeddingGemma 2 (candle CUDA, f32) | NVIDIA |
| `embeddinggemma2-metal` | EmbeddingGemma 2 (candle Metal, f32) | macOS |

ONNX, Qwen3 and EmbeddingGemma 2 acceleration are independent, so the installer
composes them per host:

```bash
# NVIDIA, all three backends on GPU
cargo install --path crates/vqtrs-cli --locked --features cuda,qwen3-cuda,embeddinggemma2-cuda
# Apple Silicon, all three backends on GPU
cargo install --path crates/vqtrs-cli --locked --features coreml,qwen3-metal,embeddinggemma2-metal
```

The candle features (`qwen3-*`, `embeddinggemma2-*`) compile GPU kernels directly
(need the CUDA toolkit / macOS at build time). The ort ONNX features register an execution
provider that needs a matching GPU-enabled `onnxruntime` at runtime and falls
back to CPU if it isn't present. `just install-all` auto-detects the host and
composes these.

> **CUDA 13.3+:** the pinned `cudarc 0.19.7` knows CUDA up to 13.2 and rejects
> 13.3 by version string only. Since all 13.x share one library ABI, pin its
> bindings to 13.2 and it builds + runs fine on a 13.3 toolkit:
> `CUDARC_CUDA_VERSION=13020 just install-cuda13` (or `just install-all`, which
> sets it automatically for either candle backend). CUDA 14+ isn't covered
> yet — the interactive installer keeps ONNX on the GPU and enabled candle
> backends on CPU (`cuda,qwen3,embeddinggemma2` when both are included).

### Noisy macOS CoreML execution

Some macOS ONNX/CoreML runs emit the native diagnostic
`Context leak detected, msgtracer returned -1`. It is not chat/prompt context
leakage, and the message alone does not establish a process memory leak or the
cause of an HTTP error. Avoid that execution path explicitly:

```bash
VQTRS_ONNX_CPU_ONLY=1 vqtrs service run --port 3010 --host 0.0.0.0
# Also works for in-process CLI inference:
VQTRS_ONNX_CPU_ONLY=1 vqtrs embed --no-daemon "a synthetic test sentence"
```

The variable is read when an ONNX model is constructed; restart an already-running
server to change it. `1` or case-insensitive `true` registers only the CPU provider
for dense/sparse/M3/reranking, even in a GPU-enabled build. It does **not** disable
Qwen3 or EmbeddingGemma Metal/CUDA. Unset it to retain compiled ONNX acceleration.
Changing providers can change floating-point vectors; do not mix provider modes
in an existing embedding index without checking consistency/re-embedding.
This is a provider-selection workaround, not stderr filtering or a patch to Apple
frameworks; confirm warning avoidance on the affected machine. HTTP 500 responses
still need their JSON error diagnosed separately. Install current `main` first;
the original `v0.2.0` tag predates this option.

## Workspace layout

```
crates/
  vqtrs-core/   # the embeddings + reranking engine (library)
  vqtrs-cli/    # `vqtrs` command-line binary
  vqtrs-api/    # OpenAI-compatible HTTP server
```

## License

[MIT](LICENSE) for original vqtrs code. Ported third-party code retains its
[upstream notices and licenses](crates/vqtrs-core/src/embedding_gemma2/NOTICE.md).
Downloaded model weights have their own licenses.

## Development

```bash
cargo fmt
cargo clippy --workspace --all-targets   # maximal lint preset, must be clean
cargo test
```
