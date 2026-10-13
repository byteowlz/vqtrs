# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Smoke extracted ONNX-only archives on native CI hardware, never production."""

import argparse
import http.client
import json
import math
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "Qdrant/all-MiniLM-L6-v2-onnx"
TEXTS = ["A bicycle has two wheels.", "Fresh bread is baked in an oven."]


def vectors(response):
    assert response["model"] == MODEL, response.get("model")
    rows = response["data"]
    assert len(rows) == 2
    result = []
    for index, item in enumerate(rows):
        assert item["index"] == index
        vector = item["embedding"]
        assert len(vector) == 384 and all(math.isfinite(v) for v in vector)
        assert abs(math.sqrt(sum(v * v for v in vector)) - 1) < 1e-5
        result.append(vector)
    return result


def request(url, body=None, status=200):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            code, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        code, raw = error.code, error.read()
    assert code == status, (code, status, raw[:400])
    return json.loads(raw) if status == 200 else None


def reject_socket_override(command, env):
    rejected = subprocess.run(
        command, env=env, text=True, capture_output=True, timeout=15, check=False
    )
    assert (
        rejected.returncode != 0 and "Unix sockets are unavailable" in rejected.stderr
    )


def wait_ready(child, url):
    deadline = time.monotonic() + 120
    while True:
        assert child.poll() is None, "API exited; inspect api.log"
        try:
            health = request(url + "/health")
            assert health["max_in_flight"] == 1 and health["in_flight"] == 0
            return
        except urllib.error.URLError:
            assert time.monotonic() < deadline, "API startup deadline"
            time.sleep(0.2)


def http_checks(url, expected):
    actual = vectors(request(url + "/v1/embeddings", {"model": MODEL, "input": TEXTS}))
    assert (
        max(abs(a - b) for x, y in zip(expected, actual) for a, b in zip(x, y)) <= 1e-6
    )
    request(url + "/v1/embeddings", {"input": TEXTS + ["overflow"]}, 413)
    request(url + "/v1/embeddings", {"model": MODEL}, 422)
    assert request(url + "/health")["in_flight"] == 0
    return actual


def unix_checks(cli, work, env, actual):
    uds = work / "api.sock"
    warm = json.loads(
        subprocess.check_output(
            [str(cli), "embed", *TEXTS],
            env={**env, "VQTRS_SOCKET": str(uds)},
            cwd=work,
            text=True,
            timeout=120,
        )
    )
    assert vectors(warm) == actual
    connection = http.client.HTTPConnection("localhost", timeout=10)
    connection.sock = socket.socket(socket.AF_UNIX)
    try:
        connection.sock.connect(str(uds))
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 200
    finally:
        connection.close()


def stop(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=15)


def local_probe(cli, api, work, env):
    versions = [
        subprocess.check_output(
            [str(binary), "--version"], env=env, text=True, timeout=15
        )
        .strip()
        .split()[-1]
        for binary in (cli, api)
    ]
    assert versions[0] == versions[1], versions
    local = json.loads(
        subprocess.check_output(
            [str(cli), "embed", "--no-daemon", "--model", MODEL, *TEXTS],
            env=env,
            cwd=work,
            text=True,
            timeout=240,
        )
    )
    return versions[0], vectors(local)


def run(cli, api, work):
    env = os.environ.copy()
    env.pop("VQTRS_SOCKET", None)
    env.pop("DYLD_LIBRARY_PATH", None)  # Intel archive must find its bundled dylib.
    version, expected = local_probe(cli, api, work, env)
    config = work / "config.toml"
    config.write_text("max_loaded = 2\nmax_inflight = 1\nmax_batch_texts = 2\n")
    unix = os.name == "posix"
    uds = work / "api.sock"
    if not unix:
        reject_socket_override(
            [str(api), "--config", str(config), "--socket", str(uds)], env
        )
        reject_socket_override(
            [str(cli), "models", "--json"], {**env, "VQTRS_SOCKET": str(uds)}
        )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    args = [
        str(api),
        "--config",
        str(config),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if unix:
        args += ["--socket", str(uds)]
    url = f"http://127.0.0.1:{port}"
    with (work / "api.log").open("wb") as log:
        child = subprocess.Popen(args, env=env, cwd=work, stdout=log, stderr=log)
        try:
            wait_ready(child, url)
            actual = http_checks(url, expected)
            if unix:
                unix_checks(cli, work, env, actual)
        finally:
            stop(child)
    print(
        json.dumps(
            {
                "status": "passed",
                "version": version,
                "model": MODEL,
                "dimensions": 384,
                "finite_unit_norm": True,
                "http_cli_parity": True,
                "unix_transport": "passed" if unix else "explicitly unsupported",
                "batch_limit": 2,
                "inflight_limit": 1,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--api", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Preserve CI diagnostics in this private directory",
    )
    args = parser.parse_args()
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=False)
        run(args.cli.resolve(), args.api.resolve(), args.work_dir.resolve())
        return
    with tempfile.TemporaryDirectory(prefix="vqtrs-archive-smoke-") as temporary:
        run(args.cli.resolve(), args.api.resolve(), Path(temporary))


if __name__ == "__main__":
    main()
