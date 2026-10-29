#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_TARGET_DIR="$PWD/.auto/target" CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
cargo fmt --all -- --check
cargo check --quiet --locked --offline --workspace
cargo clippy --quiet --locked --offline --workspace --all-targets
cargo test --quiet --locked --offline --workspace
uv run --no-project python -m unittest discover -s scripts/benchmarks -p 'test_compare_exact.py'
