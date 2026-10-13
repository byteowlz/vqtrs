"""Check an explicitly selected disposable native API against local CLI/UDS inference.

Run with uv run --no-project <private-python> -I this-script --help.
All inputs are repository-generated synthetic fixtures. Reports omit vectors/media.
"""

import argparse
import base64
import json
import math
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

TEXTS = [
    "task: search result | query: What powers the Sun?",
    "title: none | text: The Sun is powered by nuclear fusion.",
]


def vector(data):
    value = data["embedding"]
    if len(value) != 768 or not all(math.isfinite(x) for x in value):
        raise ValueError("invalid output shape/finiteness")
    if abs(math.sqrt(sum(x * x for x in value)) - 1) > 2e-6:
        raise ValueError("unnormalized native output")
    return value


def run(args):
    counts = {"http": 0, "uds": 0, "local_cli": 0, "rejected": 0}
    worst = 0.0

    def request(route, body, expected=200):
        req = urllib.request.Request(
            args.url + route,
            data=json.dumps({"model": args.model, **body}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=900) as response:
                status, raw = response.status, response.read(512 * 1024)
        except urllib.error.HTTPError as error:
            status, raw = error.code, error.read(4096)
        if status != expected:
            raise ValueError(
                f"unexpected HTTP status {status}, expected {expected}: {raw[:200]!r}"
            )
        if expected != 200:
            counts["rejected"] += 1
            return None
        value = json.loads(raw)
        if value["model"] != args.model:
            raise ValueError("model identity changed")
        counts["http"] += 1
        for i, data in enumerate(value["data"]):
            if data["index"] != i:
                raise ValueError("output order changed")
            vector(data)
        return value

    def compare(actual, expected):
        nonlocal worst
        if len(actual["data"]) != len(expected["data"]):
            raise ValueError("output count changed")
        for a, b in zip(actual["data"], expected["data"]):
            delta = max(abs(x - y) for x, y in zip(vector(a), vector(b)))
            worst = max(worst, delta)
            if delta > 2e-6:
                raise ValueError(f"same-runtime transport drift: {delta}")

    def cli(command, expected, mode):
        environment = {**os.environ, "VQTRS_SOCKET": args.socket}
        if mode == "uds":
            # Prove that a daemon result wasn't secretly recomputed locally.
            environment["VQTRS_NATIVE_PYTHON"] = "/nonexistent/vqtrs-native-disabled"
        output = subprocess.run(
            [
                str(args.cli),
                *command,
                "-m",
                args.model,
                *(["--no-daemon"] if mode == "local_cli" else []),
            ],
            env=environment,
            check=True,
            text=True,
            capture_output=True,
            timeout=900,
        )
        actual = json.loads(output.stdout)
        if actual["model"] != args.model or actual["dimensions"] != 768:
            raise ValueError("CLI model identity changed")
        compare(actual, expected)
        counts[mode] += 1

    text = request("/v1/embeddings", {"input": TEXTS})
    for mode in ("uds", "local_cli"):
        cli(["embed", *TEXTS], text, mode)
    cases = [
        ("image", args.media / "small_64x48.png"),
        ("audio", args.media / "short_300ms.wav"),
        ("video", args.video),
    ]
    inputs = [{"modality": "text", "text": TEXTS[0]}]
    expected = [text["data"][0]]
    for kind, path in cases:
        item = {"modality": kind, "data": base64.b64encode(path.read_bytes()).decode()}
        response = request("/embeddings/multimodal", {"input": [item]})
        expected.extend(response["data"])
        inputs.append(item)
        for mode in ("uds", "local_cli"):
            cli(["embed-media", "--modality", kind, str(path)], response, mode)
    mixed = request("/embeddings/multimodal", {"input": inputs})
    compare(mixed, {"data": expected})
    reversed_mixed = request(
        "/embeddings/multimodal", {"input": list(reversed(inputs))}
    )
    compare(reversed_mixed, {"data": list(reversed(expected))})
    # Text batch padding and reordering must not change vectors materially.
    reverse = request("/v1/embeddings", {"input": list(reversed(TEXTS))})
    compare(reverse, {"data": list(reversed(text["data"]))})
    request(
        "/embeddings/multimodal",
        {"input": [{"modality": "audio", "data": "AQID"}]},
        400,
    )
    request(
        "/embeddings/multimodal",
        {"input": [{"modality": "image", "data": "bad!"}]},
        400,
    )
    request(
        "/embeddings/multimodal",
        {"input": [{"modality": "video", "data": "AQID"}]},
        400,
    )
    request(
        "/embeddings/multimodal",
        {"input": [{"modality": "image", "url": "https://example.invalid/image"}]},
        422,
    )
    request("/embeddings/multimodal", {"input": []}, 400)
    long_audio = base64.b64encode(
        (args.media / "long_31500ms_pcm16.wav").read_bytes()
    ).decode()
    request(
        "/embeddings/multimodal",
        {"input": [{"modality": "audio", "data": long_audio}]},
        400,
    )
    request(
        "/embeddings/multimodal",
        {"input": [{"modality": "text", "text": "x " * 9000}]},
        400,
    )
    compare(request("/v1/embeddings", {"input": TEXTS}), text)
    invalid = args.report.parent / "invalid-native-audio.wav"
    invalid.write_bytes(b"bad")
    result = subprocess.run(
        [
            str(args.cli),
            "embed-media",
            "--modality",
            "audio",
            str(invalid),
            "-m",
            args.model,
        ],
        env={
            **os.environ,
            "VQTRS_SOCKET": args.socket,
            "VQTRS_NATIVE_PYTHON": "/nonexistent/vqtrs-native-disabled",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    invalid.unlink()
    if result.returncode == 0 or "no local retry" not in result.stderr:
        raise ValueError("native daemon failure was silently retried locally")
    counts["rejected"] += 1
    report = {
        "model": args.model,
        "synthetic_inputs_only": True,
        "checks": counts,
        "worst_same_runtime_transport_max_abs": worst,
        "native_vs_f32_equivalence_claimed": False,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--media", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    run(parser.parse_args())
