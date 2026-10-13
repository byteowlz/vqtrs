#!/usr/bin/env bash
# Native Intel macOS CPU ONNX Runtime for ort-sys rc.12 (API24).
set -euo pipefail

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != x86_64 ]]; then
    printf '%s\n' 'This build requires a native Intel macOS runner.' >&2
    exit 1
fi

ROOT=${1:?usage: build-intel-ort.sh ABSOLUTE_WORK_DIR}
[[ "$ROOT" == /* ]] || { printf '%s\n' 'Work directory must be absolute.' >&2; exit 1; }
PIN=058787ceead760166e3c50a0a4cba8a833a6f53f # official ONNX Runtime v1.24.2
mkdir -p "$ROOT"
git clone --filter=blob:none --no-checkout https://github.com/microsoft/onnxruntime.git "$ROOT/source"
git -C "$ROOT/source" checkout --detach "$PIN"
[[ "$(git -C "$ROOT/source" rev-parse HEAD)" == "$PIN" ]]
git -C "$ROOT/source" submodule update --init --recursive

# Isolate Python; only the pinned upstream build module's directory is added.
# No pybind/wheel/GPU builds or automatic upstream pip installs.
uv run --no-project --python 3.12 --with cmake==3.31.6 --with ninja==1.11.1.3 \
    python -I -c 'import runpy,sys; from pathlib import Path; p=Path(sys.argv[1]); sys.path.insert(0,str(p.parent)); sys.argv=sys.argv[1:]; runpy.run_path(str(p),run_name="__main__")' \
    "$ROOT/source/tools/ci_build/build.py" --build_dir "$ROOT/build" \
    --config Release --update --build --build_shared_lib --parallel 2 \
    --skip_tests --skip_pip_install --skip_submodule_sync --osx_arch x86_64 \
    --cmake_extra_defines CMAKE_OSX_DEPLOYMENT_TARGET=13.0 onnxruntime_BUILD_UNIT_TESTS=OFF

LIB="$ROOT/build/Release"
test -f "$LIB/libonnxruntime.1.24.2.dylib"
file "$LIB/libonnxruntime.1.24.2.dylib" | tee "$ROOT/architecture.txt"
file "$LIB/libonnxruntime.1.24.2.dylib" | rg 'x86_64' >/dev/null
install_name_tool -id '@rpath/libonnxruntime.1.24.2.dylib' "$LIB/libonnxruntime.1.24.2.dylib"
cp "$ROOT/source/LICENSE" "$LIB/ONNXRuntime-LICENSE"
cp "$ROOT/source/ThirdPartyNotices.txt" "$LIB/ONNXRuntime-ThirdPartyNotices.txt"
printf 'onnxruntime=1.24.2\nsource=%s\narch=x86_64\nprovider=CPU\n' "$PIN" > "$LIB/ONNXRuntime-build.txt"
shasum -a 256 "$LIB/libonnxruntime.1.24.2.dylib"

# Linking and tests use the matching shared library; archives must bundle it.
if [[ -n "${GITHUB_ENV:-}" ]]; then
    {
        printf 'ORT_LIB_PATH=%s\n' "$LIB"
        printf 'ORT_PREFER_DYNAMIC_LINK=1\n'
        printf 'DYLD_LIBRARY_PATH=%s\n' "$LIB"
        printf 'RUSTFLAGS=%s -C link-arg=-Wl,-rpath,@executable_path\n' "${RUSTFLAGS:-}"
    } >> "$GITHUB_ENV"
fi
