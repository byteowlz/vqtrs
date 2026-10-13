"""Private serialized native worker; only Rust-prepared tensors cross this boundary."""

import contextlib
import hashlib
import json
import os
import platform
import select
import sys
import threading
from pathlib import Path

import numpy as np

PROTOCOL = 1
COREML_ID = "vqtrs/embeddinggemma-2-coreml-fp16-3f978ddf"
MLX_ID = "vqtrs/embeddinggemma-2-mlx-regular-bf16-4f4634bb"
MAX_LINE = 512 * 1024


def verify(root, manifest):
    for relative, expected in manifest["files"].items():
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe asset path")
        # Pinned HF snapshots normally symlink to the shared blob store. Their
        # contents, not their spelling/location, are authenticated by SHA256.
        path = root / relative
        if not path.is_file():
            raise ValueError("missing asset file")
        if path.stat().st_size != expected["size"]:
            raise ValueError("asset size mismatch: " + str(relative))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected["sha256"]:
            raise ValueError("asset hash mismatch: " + str(relative))


def tensor(item, key, shape, dtype="<f4"):
    root = Path(item["directory"]).resolve(strict=True)
    path = (root / key).resolve(strict=True)
    if path.parent != root or not path.is_file():
        raise ValueError("unsafe tensor path")
    count = int(np.prod(shape))
    if count <= 0 or count > 32 * 1260 * 768:
        raise ValueError("tensor size limit")
    if path.stat().st_size != count * np.dtype(dtype).itemsize:
        raise ValueError("tensor shape mismatch")
    value = np.fromfile(path, dtype=dtype).reshape(shape)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite input tensor")
    return value


def normalize(vector):
    value = np.asarray(vector, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 768 or not np.isfinite(value).all():
        raise ValueError("invalid native output")
    norm = np.sqrt(np.sum(value.astype(np.float64) ** 2, axis=1, keepdims=True))
    if not np.isfinite(norm).all() or np.any(norm <= 0):
        raise ValueError("invalid native output norm")
    # Explicit public contract: normalize the returned native vector in host F32.
    return (value / norm.astype(np.float32)).tolist()


def watch_owner():
    """Exit even during prediction if the Rust owner closes/dies (uv is a descendant)."""
    poller = select.poll()
    poller.register(sys.stdin.fileno(), select.POLLHUP | select.POLLERR)
    while True:
        if poller.poll(500):
            os._exit(1)  # Private worker only; never terminate another process.


def run():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise ValueError("native Apple workers require Apple Silicon macOS")
    threading.Thread(target=watch_owner, daemon=True).start()
    kind, root = sys.argv[1], Path(sys.argv[2]).resolve(strict=True)
    directory = Path(__file__).parent
    manifest = json.loads((directory / (kind + "-assets.json")).read_text())
    verify(root, manifest)
    with contextlib.redirect_stdout(sys.stderr):
        if kind == "coreml":
            if int(platform.mac_ver()[0].split(".")[0]) < 15:
                raise ValueError("CoreML multifunction graphs require macOS15+")
            from coreml_backend import CoreML

            backend, identity = CoreML(root), COREML_ID
        elif kind == "mlx":
            from mlx_backend import MLX

            backend, identity = MLX(root, directory), MLX_ID
        else:
            raise ValueError("unknown native backend")
    print(json.dumps({"protocol": PROTOCOL, "model": identity}), flush=True)
    while True:
        line = sys.stdin.buffer.readline(MAX_LINE + 1)
        if not line:
            return
        if len(line) > MAX_LINE or not line.endswith(b"\n"):
            raise ValueError("native request size limit")
        request = json.loads(line)
        items = request["items"]
        if request["protocol"] != PROTOCOL or not 1 <= len(items) <= 16:
            raise ValueError("invalid native request")
        for item in items:
            ids = item["ids"]
            limit = 512 if kind == "coreml" else 8192
            if not 0 < len(ids) <= limit or any(
                type(x) is not int or not 0 <= x < 262144 for x in ids
            ):
                raise ValueError("invalid native sequence")
        with contextlib.redirect_stdout(sys.stderr):
            vectors = normalize(backend.embed(items))
        if len(vectors) != len(items):
            raise ValueError("native output count mismatch")
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "model": identity,
                    "id": request["id"],
                    "vectors": vectors,
                },
                allow_nan=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    try:
        # -I omits script-directory imports. Add only the Rust-owned private worker directory.
        sys.path.insert(0, str(Path(__file__).parent))
        run()
    except Exception as error:  # noqa: BLE001 -- fail-closed process boundary; never retry/fallback
        # No text/media values are logged. Fail closed, without restart/fallback.
        print(
            "native worker failed: " + type(error).__name__ + ": " + str(error),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
