#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_TARGET_DIR="$PWD/.auto/target" CARGO_INCREMENTAL=0 CARGO_PROFILE_RELEASE_DEBUG=0
model=${VQTRS_BENCH_MODEL:-Qdrant/all-MiniLM-L6-v2-onnx}
suite=${VQTRS_BENCH_SUITE:-short}
label=${VQTRS_BENCH_LABEL:-candidate}
[[ "$label" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
mkdir -p .auto/results
cargo build --quiet --locked --offline --release -p vqtrs-cli --example engine_bench
cp .auto/target/release/examples/engine_bench .auto/bin/candidate
for arm in baseline candidate candidate baseline; do
    counter=${counter:-0}
    counter=$((counter+1))
    ".auto/bin/$arm" "$model" scripts/parity/eg2_cases_text.json ".auto/results/$label-$counter-$arm.json" "$suite"
done
uv run --no-project python scripts/benchmarks/compare_exact.py \
    --baseline ".auto/results/$label-1-baseline.json" ".auto/results/$label-4-baseline.json" \
    --candidate ".auto/results/$label-2-candidate.json" ".auto/results/$label-3-candidate.json"
