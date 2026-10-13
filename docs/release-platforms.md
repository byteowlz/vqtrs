# Release platform verification

`Platform Archives` builds and tests all four native targets before producing
ONNX-only archives: Linux x86_64, Apple Silicon, Intel macOS, and Windows x86_64.
Main/PR/manual runs upload CI artifacts only; they do not create a release.
`Release` reuses those checks and publishes only an existing annotated version
matching `Cargo.toml`, after all four extracted-archive smokes and checksums pass.
Manual release dispatch defaults to **not publishing**. No registry publication.

## Transports

Unix platforms retain the warm UDS channel and the HTTP API. On Windows the API
starts TCP-only by default, and CLI embedding commands load locally; there is no
implicit TCP daemon discovery. Explicit API `--socket`, `VQTRS_SOCKET` or config
`socket` requests fail before loading models on unsupported platforms. CLI
`VQTRS_SOCKET` likewise returns a clear unsupported error rather than silently
performing local inference. Unix `--no-socket` behavior is unchanged.

Tests exercise both platform-selection policies without global environment
mutation, real Unix listener HTTP, and Windows-only client rejection. Native
Windows CI also builds/lints/tests the optional CPU `embeddinggemma2` feature.
Default archives never acquire it or the experimental Apple worker dependencies.

## Intel macOS runtime

`ort-sys` rc.12 needs ONNX Runtime API24 but provides no Intel macOS CPU prebuilt.
`scripts/ci/build-intel-ort.sh` builds the official **1.24.2** source at
`058787ceead760166e3c50a0a4cba8a833a6f53f` on a native Intel runner (macOS13+
runtime target). Its reviewed two-line upstream Clang repair
`c7c939ff1e961284dd3b6f720e7ec980f8d6d331` is backported: MLAS's existing narrow
pragma guard referenced a config macro it never included, so a dead constexpr
branch tripped `-Warray-bounds`. The patch changes only the pragma guards, not
arithmetic, and its SHA256/source identity is recorded in the bundle. No global
warning suppression, older runtime, different backend/feature set or GPU provider.
The shared CPU library, source/build identity and upstream license/notices are
bundled. Binaries use an executable-relative rpath. The extracted archive's
version, actual 384-dimensional normalized ONNX embeddings, CLI/HTTP parity,
UDS and limits (including pre-body429/503 shedding with `Retry-After`) are checked
with `DYLD_LIBRARY_PATH` removed. These checks must
pass; source inspection/architecture labels alone are not linking evidence.

Local offline checks:

```sh
just check
cargo clippy --workspace --all-targets
just test
just test-archive-contract
```

A real archive probe uses `uv run --no-project python -I
scripts/ci/archive_smoke.py --cli <extracted-vqtrs> --api <extracted-vqtrs-api>`
on an isolated, adequately provisioned runner, not a production service.
It uses synthetic texts, downloads a public ONNX model, and terminates/reaps its
own temporary API. Model vectors are not included in the public probe report.

## Immutable v0.2.0

`v0.2.0` remains at `a30f2f0c33818dcb5586404a146e21d7070731a2`; these portability
fixes are **post-tag source**, not a rewritten release. The original run
37670331088 packaged Linux/Apple Silicon but failed Windows/Intel, so no GitHub
release was created. A rerun against the old tag cannot magically include new
source/scripts. A corrective release needs a new version/tag after the native
matrix succeeds. Do not label main's artifacts as original v0.2.0 assets.
`vqtrs-s449` stays open until actual platform/archive and release evidence exists.
