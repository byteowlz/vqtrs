#!/usr/bin/env bash
set -euo pipefail

# Interactive installer for vqtrs (the `vqtrs` CLI + `vqtrs-api` server).
# Detects the host, selects acceleration and optional candle backends, then
# installs both binaries from the lockfile. EmbeddingGemma 2 defaults to enabled.

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

# --- EmbeddingGemma 2 backend ---------------------------------------------------
echo
echo "Include EmbeddingGemma 2 (text, image, audio and video embeddings)?"
echo "It adds candle. Video containers also require ffmpeg and ffprobe at runtime."
read -rp "Include EmbeddingGemma 2? [Y/n]: " GEMMA2
GEMMA2="${GEMMA2:-y}"
WANT_GEMMA2=false
[[ "$GEMMA2" =~ ^[Yy]$ ]] && WANT_GEMMA2=true

# --- compose feature string ----------------------------------------------------
FEATURES=""
CANDLE_SUFFIX=""
case "$ACCEL" in
  1) ;;
  2)
    FEATURES="cuda"
    CANDLE_SUFFIX="-cuda"
    if $WANT_QWEN3 || $WANT_GEMMA2; then
      # Both candle backends use cudarc 0.19.7, which knows CUDA up to 13.2.
      # Retain the 13.x binding pin; CUDA 14+ uses CPU candle backends.
      CUDA_OUTPUT="$(nvcc --version 2>/dev/null || true)"
      if [[ "$CUDA_OUTPUT" =~ release[[:space:]]([0-9]+)\.([0-9]+) ]]; then
        CUDA_MAJOR="${BASH_REMATCH[1]}"
        CUDA_MINOR="${BASH_REMATCH[2]}"
        CUDA_VER="${CUDA_MAJOR}.${CUDA_MINOR}"
        if [[ "$CUDA_MAJOR" == "13" && "$CUDA_MINOR" -gt 2 ]]; then
          echo "CUDA ${CUDA_VER}: pinning CUDARC_CUDA_VERSION=13020 (13.2 bindings, ABI-compatible)."
          export CUDARC_CUDA_VERSION=13020
        elif [[ "$CUDA_MAJOR" -ge 14 ]]; then
          echo "CUDA ${CUDA_VER} is newer than cudarc supports — candle backends on CPU (ONNX still on GPU)."
          CANDLE_SUFFIX=""
        fi
      else
        echo "Warning: CUDA toolkit version unavailable; candle CUDA builds require nvcc."
      fi
    fi
    ;;
  3)
    if [[ "$OS" != "Darwin" ]]; then
      echo "Warning: CoreML/Metal are macOS-only; falling back to CPU."
    else
      FEATURES="coreml"
      CANDLE_SUFFIX="-metal"
    fi
    ;;
  *)
    echo "Invalid choice; using CPU only."
    ;;
esac

if $WANT_QWEN3; then
  FEATURES="${FEATURES:+${FEATURES},}qwen3${CANDLE_SUFFIX}"
fi
if $WANT_GEMMA2; then
  FEATURES="${FEATURES:+${FEATURES},}embeddinggemma2${CANDLE_SUFFIX}"
fi

# Keep the array nonempty for macOS's stock Bash 3.2 with nounset enabled.
FEATURE_ARGS=(--locked)
if [[ -n "$FEATURES" ]]; then
  FEATURE_ARGS+=(--features "$FEATURES")
fi

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
