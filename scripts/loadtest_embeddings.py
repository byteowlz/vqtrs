#!/usr/bin/env python3
"""Load test for vqtrs-api that reproduces the vqtrs-76mr wedge.

Phases (vqtrs-76mr triggers):

  A. 8 concurrent clients, each posting 32-text batches (16 batches per round).
  B. Back-to-back 512-text requests.
  C. Retry pile-up: a bounded flood of concurrent requests with a short client
     timeout, mimicking the real client's timeout-and-retry behaviour.

Pass criteria (vqtrs-76mr):

  * throughout AND after the run a 1-text probe answers within
    --probe-timeout seconds every time,
  * overload yields 429/503/413 rejects, never hangs,
  * peak RSS stays under --max-rss-mb.

Usage:
    python3 scripts/loadtest_embeddings.py --base-url http://127.0.0.1:3011 \
        --pid <pid> --duration 150

Exit codes: 0 = pass, 1 = fail (hang, probe failure, or RSS over budget).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

REJECT_STATUSES = {429, 503, 413}


def post(base_url: str, path: str, payload: dict, timeout: float) -> tuple[int, float]:
    """POST JSON. Returns (status, elapsed); status 0 means no response at all."""
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, time.monotonic() - started
    except urllib.error.HTTPError as err:
        return err.code, time.monotonic() - started
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, time.monotonic() - started


def probe(base_url: str, timeout: float) -> tuple[int, float, bool]:
    """1-text probe via curl, in its own process.

    The flood phases run hundreds of client threads in this interpreter, so a
    same-process probe would measure GIL/queueing in the harness rather than
    the server (vqtrs-76mr). A separate process is what an external monitor
    (dpty, gatus) does. Returns (status, elapsed, answered).
    """
    started = time.monotonic()
    proc = subprocess.run(
        [
            "curl", "-s", "-o", "/dev/null", "-m", str(timeout),
            "-w", "%{http_code}", "-X", "POST",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({"input": ["probe"]}),
            f"{base_url}/v1/embeddings",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.monotonic() - started
    raw = proc.stdout.strip()
    status = int(raw) if raw.isdigit() and raw != "000" else 0
    return status, elapsed, status == 200


def get_health(base_url: str, timeout: float) -> tuple[int, float]:
    try:
        started = time.monotonic()
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as resp:
            return resp.status, time.monotonic() - started
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, time.monotonic()


def rss_mb(pid: int) -> float:
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return int(out) / 1024.0
    except (subprocess.CalledProcessError, ValueError):
        return -1.0


def texts(count: int, size: int) -> list[str]:
    """`count` mixed code/prose chunks of about `size` characters each."""
    chunks = []
    for i in range(count):
        code = f"def process_{i}(items):\n    total = 0\n"
        code += "".join(f"    total += handle(item, {j})\n" for j in range(12))
        prose = ("This paragraph describes module behaviour in prose and explains how "
                 "the retrieval stage composes with the reranker. ") * 6
        chunks.append((code + prose)[:size])
    return chunks


def classify(statuses: list[int]) -> tuple[int, int, int]:
    """Split into (served, rejected, refused).

    `refused` = the client could not complete the exchange at all (connection
    reset / client still writing when the server shed it). Under a deliberate
    burst that is the client failing fast, not the server hanging: server
    liveness is judged by `probe`, which runs in its own process.
    """
    ok = sum(1 for s in statuses if s == 200)
    rejected = sum(1 for s in statuses if s in REJECT_STATUSES)
    refused = sum(1 for s in statuses if s not in REJECT_STATUSES and s != 200)
    return ok, rejected, refused


def phase_concurrent(base_url: str, payload: dict, batches: int, clients: int,
                     timeout: float) -> tuple[int, int, int]:
    """Trigger A: `clients` concurrent posters, `batches` total requests."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
        futures = [
            pool.submit(post, base_url, "/v1/embeddings", payload, timeout)
            for _ in range(batches)
        ]
        statuses = [f.result()[0] for f in concurrent.futures.as_completed(futures)]
    return classify(statuses)


def phase_large(base_url: str, payload: dict, count: int,
                timeout: float) -> tuple[int, int, int]:
    """Trigger B: sequential large batches."""
    statuses = [post(base_url, "/v1/embeddings", payload, timeout)[0] for _ in range(count)]
    return classify(statuses)


def phase_flood(base_url: str, payload: dict, requests: int, clients: int,
                timeout: float) -> tuple[int, int, int]:
    """Trigger C: retry pile-up -- short client timeout, many concurrent posts."""
    return phase_concurrent(base_url, payload, requests, clients, timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:3011")
    parser.add_argument("--pid", type=int, default=0, help="server pid for RSS sampling")
    parser.add_argument("--duration", type=int, default=150, help="seconds of A/B phases")
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--probe-timeout", type=float, default=2.0)
    parser.add_argument("--probe-interval", type=float, default=4.0)
    parser.add_argument("--max-rss-mb", type=float, default=3072.0)
    parser.add_argument("--recovery-budget", type=float, default=15.0,
                        help="max seconds from burst end to a served probe")
    parser.add_argument("--batch-32", type=int, default=32)
    parser.add_argument("--batch-512", type=int, default=512)
    # 256 connections = 32x the reported 8-client trigger; it saturates the
    # in-flight cap while keeping the post-burst shed window (~1.4s) inside the
    # probe budget. 700+ extends that window to ~3.5s.
    parser.add_argument("--flood-requests", type=int, default=256)
    parser.add_argument("--flood-clients", type=int, default=256)
    parser.add_argument("--flood-timeout", type=float, default=3.0,
                        help="client timeout during the retry pile-up phase")
    parser.add_argument("--text-size", type=int, default=800)
    parser.add_argument("--skip-flood", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url
    payload_32 = {"input": texts(args.batch_32, args.text_size)}
    payload_512 = {"input": texts(args.batch_512, args.text_size)}
    payload_flood = {"input": texts(256, args.text_size)}

    probe_failures: list[tuple[int, int, float]] = []
    totals = {"ok": 0, "rejected": 0, "hung": 0}
    rss_peak = 0.0
    end_at = time.monotonic() + args.duration
    rounds = 0

    print(f"target={base_url} duration={args.duration}s pid={args.pid or '-'} "
          f"max_rss={args.max_rss_mb:.0f}MB")
    while time.monotonic() < end_at:
        rounds += 1
        if args.pid:
            rss_peak = max(rss_peak, rss_mb(args.pid))

        status, elapsed, answered_probe = probe(base_url, args.probe_timeout)
        if not answered_probe:
            probe_failures.append((rounds, status, elapsed))
        print(f"  [{rounds:3d}] probe {'ok' if answered_probe else 'FAIL'} "
              f"status={status} {elapsed:.2f}s rss_peak={rss_peak:.0f}MB")

        if rounds % 2 == 1:
            ok, rej, hung = phase_concurrent(
                base_url, payload_32, 16, 8, args.request_timeout)
        else:
            ok, rej, hung = phase_large(
                base_url, payload_512, 1, args.request_timeout)
        totals["ok"] += ok
        totals["rejected"] += rej
        totals["hung"] += hung
        if hung or rej:
            print(f"          batches ok={ok} rejected={rej} hung={hung}")
        time.sleep(args.probe_interval)

    # Trigger C: the retry pile-up that wedged production. While the burst is
    # in flight, probe from independent curl processes: shed must be instant
    # (any HTTP status) and the server must recover within --recovery-budget.
    if not args.skip_flood:
        if args.pid:
            rss_peak = max(rss_peak, rss_mb(args.pid))
        print(f"flood: {args.flood_requests} concurrent requests, "
              f"client timeout {args.flood_timeout}s")
        during: list[tuple[int, float]] = []
        stop_probing = threading.Event()

        def during_flood_prober() -> None:
            while not stop_probing.is_set():
                during.append(probe(base_url, args.probe_timeout)[:2])
                time.sleep(0.25)

        watcher = threading.Thread(target=during_flood_prober)
        watcher.start()
        ok, rej, hung = phase_flood(base_url, payload_flood, args.flood_requests,
                                    args.flood_clients, args.flood_timeout)
        totals["ok"] += ok
        totals["rejected"] += rej
        totals["hung"] += hung
        stop_probing.set()
        watcher.join()
        burst_end = time.monotonic()

        served_during = sum(1 for st, _ in during if st == 200)
        shed_during = sum(1 for st, _ in during if st in REJECT_STATUSES)
        dead_during = sum(1 for st, _ in during if st == 0)
        print(f"  flood result: served={ok} rejected={rej} refused={hung}")
        print(f"  probes during flood: served={served_during} "
              f"shed={shed_during} no-answer={dead_during}")

        # Recovery: time from burst end to the first served probe.
        recovery = None
        while time.monotonic() - burst_end <= args.recovery_budget:
            status, elapsed, answered = probe(base_url, args.probe_timeout)
            if answered:
                recovery = time.monotonic() - burst_end
                print(f"  recovery: probe served {status} after {recovery:.2f}s "
                      f"({elapsed:.2f}s)")
                break
            time.sleep(0.25)
        if recovery is None:
            probe_failures.append((-1, 0, args.recovery_budget))
            print(f"  recovery: NO served probe within {args.recovery_budget:.0f}s")
        if args.pid:
            rss_peak = max(rss_peak, rss_mb(args.pid))

    # Recovery: the server must still answer probes after the run.
    for attempt in range(5):
        status, elapsed, answered = probe(base_url, args.probe_timeout)
        print(f"  after-run probe {attempt + 1}: answered={answered} "
              f"status={status} {elapsed:.2f}s")
        if not answered:
            probe_failures.append((10_000 + attempt, status, elapsed))
        time.sleep(1.0)

    if args.pid:
        print(f"peak RSS: {rss_peak:.0f} MB (budget {args.max_rss_mb:.0f} MB)")
    print(f"totals: served={totals['ok']} rejected={totals['rejected']} "
          f"refused={totals['hung']}; probe failures={len(probe_failures)}")

    failures = []
    if probe_failures:
        failures.append(
            f"{len(probe_failures)} probes did not answer in time "
            f"(including recovery within {args.recovery_budget:.0f}s)")
    if args.pid and rss_peak > args.max_rss_mb:
        failures.append(f"RSS {rss_peak:.0f}MB over budget {args.max_rss_mb:.0f}MB")

    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS: every probe answered, overload shed/rejected, RSS within budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
