# AGENTS.md

`vqtrs` is a lean local embeddings + reranking engine. It does embeddings and
nothing else.

## Layout

```
crates/
  vqtrs-core/   # the engine (library): fastembed ONNX + Qwen3 candle, reranking, sparse
  vqtrs-cli/    # `vqtrs` binary: embed / sparse / m3 / rerank / models / service
  vqtrs-api/    # `vqtrs-api` binary: OpenAI-compatible + /rerank HTTP/UDS server
```

Both binaries depend on `vqtrs-core`; nothing else. All deps are pinned in the
root `Cargo.toml`.

## Lints (the thing to know)

The workspace runs a **maximum-strictness Clippy preset** (`[workspace.lints]` in
root `Cargo.toml`). `cargo clippy --workspace --all-targets` must be clean.

- `unsafe_code = forbid`; `unwrap_used`/`expect_used`/`panic`/`todo` = deny — use
  `?`, `anyhow::Result`, `ok_or_else`, `map_or`. (unwrap/expect/panic are allowed
  in `#[cfg(test)]` only.)
- `allow_attributes = deny` — suppress a lint with `#[expect(lint, reason = "…")]`,
  never `#[allow(...)]`.
- pedantic/nursery/cargo all deny: public items need docs, `Result`-returning
  public fns need an `# Errors` section, doc prose needs backticks around
  identifiers (add domain terms to `clippy.toml` `doc-valid-idents`).

After any change: `cargo fmt`, then `cargo clippy --workspace --all-targets`, then
`cargo test` (engine tests are in `vqtrs-core`, no network).

## Build features (all off by default)

Default build is ONNX-only. `qwen3` adds the candle Qwen3 backend. GPU is
per-backend: `cuda`/`tensorrt`/`coreml`/`directml` (ONNX, ort EPs) and
`qwen3-cuda`/`qwen3-metal` (Qwen3, candle). See README for the matrix.

## Conventions

- Issue tracking is **trx** (`.trx/` in this repo). Use `trx ready` / `trx create`
  / `trx update --status in_progress` / `trx close`; `trx sync` commits `.trx/`.
- Clean refactors over back-compat; edit in place, no `FooV2`. Minimal file
  headers. Never publish to registries without explicit approval.
