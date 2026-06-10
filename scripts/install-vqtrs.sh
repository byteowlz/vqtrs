#!/usr/bin/env bash
set -euo pipefail

# Interactive installer for vqtrs (the `vqtrs` CLI + `vqtrs-api` server).
# Detects the host, lets you pick GPU acceleration and whether to include the
# (heavier) Qwen3 candle backend, then `cargo install`s both binaries.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OS="$(uname -s)"
ARCH="$(uname -m)"

echo "=== vqtrs installer ==="
echo "Host: ${OS} (${ARCH})"
echo

# --- GPU acceleration ----------------------------------------------------------
echo "Hardware acceleration:"
echo "  1) None (CPU only)         — works everywhere"
echo "  2) NVIDIA CUDA             — needs the CUDA toolkit"
echo "  3) Apple (CoreML + Metal)  — macOS / Apple Silicon"
echo

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
  DEFAULT_ACCEL=3
elif [[ "$OS" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  DEFAULT_ACCEL=2
else
  DEFAULT_ACCEL=1
fi
read -rp "Choice [${DEFAULT_ACCEL}]: " ACCEL
ACCEL="${ACCEL:-$DEFAULT_ACCEL}"

# --- Qwen3 backend -------------------------------------------------------------
echo
echo "Include the Qwen3 candle backend (SOTA 0.6B/4B/8B embedding models)?"
echo "It adds candle and is a much heavier build. ONNX models work without it."
read -rp "Include Qwen3? [y/N]: " QWEN3
QWEN3="${QWEN3:-n}"
WANT_QWEN3=false
[[ "$QWEN3" =~ ^[Yy]$ ]] && WANT_QWEN3=true

# --- compose feature string ----------------------------------------------------
FEATURES=""
case "$ACCEL" in
  1)
    $WANT_QWEN3 && FEATURES="qwen3"
    ;;
  2)
    FEATURES="cuda"
    $WANT_QWEN3 && FEATURES="cuda,qwen3-cuda"
    ;;
  3)
    if [[ "$OS" != "Darwin" ]]; then
      echo "Warning: CoreML/Metal are macOS-only; falling back to CPU."
      $WANT_QWEN3 && FEATURES="qwen3"
    else
      FEATURES="coreml"
      $WANT_QWEN3 && FEATURES="coreml,qwen3-metal"
    fi
    ;;
  *)
    echo "Invalid choice; using CPU only."
    $WANT_QWEN3 && FEATURES="qwen3"
    ;;
esac

FEATURE_ARGS=()
[[ -n "$FEATURES" ]] && FEATURE_ARGS=(--features "$FEATURES")

echo
echo "Installing with features: ${FEATURES:-<none, CPU/ONNX>}"
echo

for crate in vqtrs-cli vqtrs-api; do
  echo "+ cargo install --path crates/${crate} ${FEATURE_ARGS[*]} --force"
  cargo install --path "crates/${crate}" "${FEATURE_ARGS[@]}" --force
done

echo
echo "=== done ==="
echo "Binaries: vqtrs (CLI) and vqtrs-api (server) in ~/.cargo/bin"
echo "Try:  vqtrs models   |   vqtrs-api --help"
