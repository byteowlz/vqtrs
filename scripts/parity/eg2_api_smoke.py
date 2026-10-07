# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Verify live text/mixed-media HTTP and CLI embeddings against Python dumps.

Use a disposable vqtrs-api built with embeddinggemma2, max_batch_texts=3,
not the production service. No weights/Python model loading in this script.

uv run python -I scripts/parity/eg2_api_smoke.py --url http://127.0.0.1:3012 \
  --reference /tmp/eg2-parity --cli target/release/vqtrs --socket /tmp/eg2-test.sock
"""

import argparse
import base64
import json
import os
from pathlib import Path
import struct
import subprocess
import urllib.error
import urllib.request

MODEL = "google/embeddinggemma-2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--socket")
    args = parser.parse_args()

    def request(route, body, status=200):
        req = urllib.request.Request(args.url.rstrip("/") + route, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                code, raw = response.status, response.read()
        except urllib.error.HTTPError as error:
            code, raw = error.code, error.read()
        assert code == status, (route, code, status, raw[:400])
        return json.loads(raw) if status == 200 else None

    inputs, expected, cases = [], [], []
    for kind in ["text", "image", "audio"]:
        source = Path(__file__).parent / f"eg2_cases_{kind}.json"
        case = json.loads(source.read_text())[0]
        directory = args.reference / f"ref-{kind}"
        reference = json.loads((directory / "manifest.json").read_text())["cases"][0]
        assert case["name"] == reference["name"]
        expected.append((directory / reference["tensors"]["hf.embedding"]["file"]).read_bytes())
        if kind == "text":
            inputs.append({"modality": kind, "text": case.get("prompt", "") + case["text"]})
        else:
            path = source.parent / case["path"]
            inputs.append({"modality": kind, "data": base64.b64encode(path.read_bytes()).decode()})
            cases.append((kind, path))

    def exact(vector, reference):
        assert len(vector) == 768
        assert struct.pack("<768f", *vector) == reference, "embedding differs from pinned reference"

    mixed = request("/embeddings/multimodal", {"input": inputs})
    assert mixed["model"] == MODEL and len(mixed["data"]) == 3
    assert "usage" not in mixed
    for index, (item, reference) in enumerate(zip(mixed["data"], expected)):
        assert item["index"] == index
        exact(item["embedding"], reference)
    text = request("/v1/embeddings", {"model": MODEL, "input": inputs[0]["text"]})
    exact(text["data"][0]["embedding"], expected[0])

    request("/embeddings/multimodal", {"input": []}, 400)
    request("/embeddings/multimodal", {"input": [{"modality": "image", "data": "bad!"}]}, 400)
    request("/embeddings/multimodal", {"input": [{"modality": "audio", "data": "AQID"}]}, 400)
    request("/embeddings/multimodal", {"model": "not-a-model", "input": [inputs[0]]}, 400)
    request("/embeddings/multimodal", {"input": inputs + [inputs[0]]}, 413)
    request("/embeddings/multimodal", {"input": [{"modality": "text", "text": "a " * 9000}]}, 400)
    request("/embeddings/multimodal", {"input": [{"modality": "image", "data": "A" * (16 * 1024 * 1024)}]}, 413)
    request("/embeddings/multimodal", {"input": [{"modality": "image", "url": "https://example.invalid/image"}]}, 422)
    request("/embeddings/multimodal", {"input": [{"modality": "audio", "path": "/private/file"}]}, 422)
    request("/v1/embeddings", {"model": MODEL, "input": ["a", "b", "c", "d"]}, 413)
    with urllib.request.urlopen(args.url.rstrip("/") + "/health", timeout=10) as response:
        assert response.status == 200

    if args.cli:
        env = os.environ.copy()
        if args.socket:
            env["VQTRS_SOCKET"] = args.socket
        for index, (kind, path) in enumerate(cases, start=1):
            command = [str(args.cli.resolve()), "embed-media", "--modality", kind, str(path)]
            # Exercise both the warm UDS path (when provided) and local loading.
            modes = [[], ["--no-daemon"]] if args.socket else [["--no-daemon"]]
            for flags in modes:
                result = subprocess.run(command + flags, env=env, text=True, capture_output=True, check=True, timeout=180)
                output = json.loads(result.stdout)
                assert output["model"] == MODEL and output["dimensions"] == 768
                exact(output["data"][0]["embedding"], expected[index])
    print("PASS: mixed HTTP and text embeddings bit-identical; validation/batch caps/health verified; CLI checked when requested")


if __name__ == "__main__":
    main()
