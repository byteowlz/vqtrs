#!/usr/bin/env bash
set -euo pipefail

# Exercise the real installer without downloading dependencies or replacing binaries.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
mkdir "$TMP_DIR/bin"

printf '%s\n' '#!/usr/bin/env bash' 'case "$1" in -s) echo "$TEST_OS";; -m) echo "$TEST_ARCH";; *) exit 1;; esac' > "$TMP_DIR/bin/uname"
printf '%s\n' '#!/usr/bin/env bash' '[[ "$TEST_CUDA" != missing ]] || exit 127' 'echo "Cuda compilation tools, release ${TEST_CUDA}, V${TEST_CUDA}.0"' > "$TMP_DIR/bin/nvcc"
printf '%s\n' '#!/usr/bin/env bash' 'exit 0' > "$TMP_DIR/bin/nvidia-smi"
printf '%s\n' '#!/usr/bin/env bash' 'printf "%s|CUDARC=%s\n" "$*" "${CUDARC_CUDA_VERSION:-}" >> "$TEST_CARGO_LOG"' > "$TMP_DIR/bin/cargo"
chmod +x "$TMP_DIR"/bin/*

run_case() {
  local name="$1" os="$2" arch="$3" cuda="$4" answers="$5" features="$6" binding="${7:-}"
  local crate
  : > "$TMP_DIR/cargo.log"
  : > "$TMP_DIR/expected.log"
  if ! printf '%b' "$answers" | env -u CUDARC_CUDA_VERSION \
    PATH="$TMP_DIR/bin:$PATH" TEST_OS="$os" TEST_ARCH="$arch" TEST_CUDA="$cuda" \
    TEST_CARGO_LOG="$TMP_DIR/cargo.log" "$BASH" "$ROOT_DIR/scripts/install-vqtrs.sh" > "$TMP_DIR/output.log" 2>&1; then
    printf 'FAIL: %s (installer exited early)\n' "$name" >&2
    exit 1
  fi
  for crate in vqtrs-cli vqtrs-api; do
    if [[ -n "$features" ]]; then
      printf 'install --path crates/%s --locked --features %s --force|CUDARC=%s\n' "$crate" "$features" "$binding"
    else
      printf 'install --path crates/%s --locked --force|CUDARC=%s\n' "$crate" "$binding"
    fi
  done > "$TMP_DIR/expected.log"
  if ! diff -u "$TMP_DIR/expected.log" "$TMP_DIR/cargo.log"; then
    printf 'FAIL: %s\n' "$name" >&2
    exit 1
  fi
  printf 'PASS: %s\n' "$name"
}

# Host defaults and the user's Apple + Qwen3 scenario; blank Gemma answer opts in.
run_case apple-default Darwin arm64 12.8 '\n\n\n' 'coreml,embeddinggemma2-metal'
run_case apple-qwen-default-gemma Darwin arm64 12.8 '\ny\n\n' 'coreml,qwen3-metal,embeddinggemma2-metal'
run_case linux-default Linux x86_64 12.8 '\n\n\n' 'cuda,embeddinggemma2-cuda'
run_case generic-default FreeBSD x86_64 12.8 '\n\n\n' 'embeddinggemma2'

# Each backend can be independently enabled/disabled on every acceleration path.
run_case cpu-both Linux x86_64 12.8 '1\ny\ny\n' 'qwen3,embeddinggemma2'
run_case cpu-gemma Linux x86_64 12.8 '1\nn\ny\n' 'embeddinggemma2'
run_case cpu-qwen Linux x86_64 12.8 '1\ny\nn\n' 'qwen3'
run_case cpu-onnx Linux x86_64 12.8 '1\nn\nn\n' ''
run_case cuda-both Linux x86_64 12.8 '2\ny\ny\n' 'cuda,qwen3-cuda,embeddinggemma2-cuda'
run_case cuda-gemma Linux x86_64 12.8 '2\nn\ny\n' 'cuda,embeddinggemma2-cuda'
run_case cuda-qwen Linux x86_64 12.8 '2\ny\nn\n' 'cuda,qwen3-cuda'
run_case cuda-onnx Linux x86_64 12.8 '2\nn\nn\n' 'cuda'
run_case metal-both Darwin arm64 12.8 '3\ny\ny\n' 'coreml,qwen3-metal,embeddinggemma2-metal'
run_case metal-gemma Darwin arm64 12.8 '3\nn\ny\n' 'coreml,embeddinggemma2-metal'
run_case metal-qwen Darwin arm64 12.8 '3\ny\nn\n' 'coreml,qwen3-metal'
run_case metal-onnx Darwin arm64 12.8 '3\nn\nn\n' 'coreml'

# Shared candle toolkit compatibility must work without Qwen3 as well.
run_case cuda132-gemma Linux x86_64 13.2 '2\nn\ny\n' 'cuda,embeddinggemma2-cuda'
run_case cuda-missing-version Linux x86_64 missing '2\nn\ny\n' 'cuda,embeddinggemma2-cuda'
run_case cuda13-both Linux x86_64 13.3 '2\ny\ny\n' 'cuda,qwen3-cuda,embeddinggemma2-cuda' 13020
run_case cuda13-gemma Linux x86_64 13.4 '2\nn\ny\n' 'cuda,embeddinggemma2-cuda' 13020
run_case cuda14-both Linux x86_64 14.0 '2\ny\ny\n' 'cuda,qwen3,embeddinggemma2'
run_case cuda14-gemma Linux x86_64 14.0 '2\nn\ny\n' 'cuda,embeddinggemma2'
run_case cuda14-onnx Linux x86_64 14.0 '2\nn\nn\n' 'cuda'
run_case nonapple-metal Linux x86_64 12.8 '3\ny\ny\n' 'qwen3,embeddinggemma2'
run_case invalid-choice Linux x86_64 12.8 '9\nn\ny\n' 'embeddinggemma2'
run_case uppercase Darwin arm64 12.8 '3\nY\nN\n' 'coreml,qwen3-metal'
