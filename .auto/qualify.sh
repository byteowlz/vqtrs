#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/.local/state/vqtrs-autoresearch/exact-speed-20261029"
out=.auto/results/qualification
F=scripts/parity/eg2_cases_text.json
EB=.auto/bin/engine_bench
AB=.auto/bin/engine_aux_bench
run() { # label binary args...
  local label=$1; shift
  for i in ${RUNS:-1 2}; do
    if "$@" "$out/$label-$i.json" "short" > "$out/$label-$i.log" 2>&1; then
      echo "ok $label run$i"
    else
      echo "FAIL $label run$i"; tail -3 "$out/$label-$i.log"; return 1
    fi
  done
}
run onnx-minilm "$EB" Qdrant/all-MiniLM-L6-v2-onnx "$F"
run qwen3-0.6b "$EB" Qwen/Qwen3-Embedding-0.6B "$F"
run sparse-splade "$AB" sparse Qdrant/Splade_PP_en_v1 "$F"
run m3-bgem3 "$AB" m3 BAAI/bge-m3 "$F"
run rerank-bge-base "$AB" rerank BAAI/bge-reranker-base "$F"
