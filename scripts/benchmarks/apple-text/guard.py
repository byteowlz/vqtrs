import json, math, os, re, signal, subprocess, sys, time
from pathlib import Path

name = sys.argv[1]
command = sys.argv[2:]
root = Path(os.environ.get("EG2_BENCH_WORK_DIR", "/tmp/vqtrs-apple-bench"))
env = {
    **os.environ,
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
peak = 0.0
started = time.monotonic()
max_seconds = float(os.environ.get("EG2_BENCH_TIMEOUT_SECS", "900"))
if not math.isfinite(max_seconds) or not 0 < max_seconds <= 900:
    raise ValueError("benchmark deadline must be finite and in (0,900] seconds")
samples = []
stopped = None
with (root / (name + ".log")).open("w") as log:
    p = subprocess.Popen(
        command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        while p.poll() is None:
            if time.monotonic() - started > max_seconds:
                raise RuntimeError("benchmark deadline exceeded")
            graph = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True)
            descendants = {p.pid}
            pairs = [
                tuple(map(int, l.split()))
                for l in graph.splitlines()
                if len(l.split()) == 2
            ]
            for _ in range(12):
                expanded = descendants | {
                    pid for pid, parent in pairs if parent in descendants
                }
                if expanded == descendants:
                    break
                descendants = expanded
            total = 0.0
            observed = 0
            for pid in descendants:
                try:
                    r = subprocess.run(
                        [
                            "/usr/bin/footprint",
                            "--pid",
                            str(pid),
                            "--format",
                            "bytes",
                            "--swapped",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    m = re.search(
                        r"\[" + str(pid) + r"\]:[^\n]*Footprint: ([0-9]+) B", r.stdout
                    )
                    if m:
                        total += int(m[1]) / (1 << 30)
                        observed += 1
                except subprocess.TimeoutExpired:
                    pass
            if not observed and p.poll() is None:
                raise RuntimeError("cannot monitor owned task footprint")
            peak = max(peak, total)
            samples.append([time.time(), total])
            if total > 8:
                raise RuntimeError("8 GiB sampled ownership footprint exceeded")
            vm = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
            page = re.search(r"page size of (\d+) bytes", vm)
            free = re.search(r"Pages free:\s*(\d+)\.", vm)
            if not page or not free:
                raise RuntimeError("cannot monitor host free memory")
            free_gib = int(page[1]) * int(free[1]) / (1 << 30)
            if free_gib < 14:
                raise RuntimeError(
                    "host free headroom below 14 GiB, retaining other workload floors"
                )
            time.sleep(1)
        code = p.returncode
    except BaseException as e:
        stopped = str(e)
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                # Darwin can report EPERM for a process group that just exited.
                # Never swallow a real inability to terminate a live owned task.
                if p.poll() is None:
                    raise
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        code = 1
(root / (name + "-memory.json")).write_text(
    json.dumps(
        {
            "sampled_peak_owned_footprint_gib": peak,
            "samples": samples,
            "stopped": stopped,
            "exit": code,
            "limitations": "1s sampling, not an allocation ceiling; excludes system CoreML compiler daemons",
        },
        indent=2,
    )
)
sys.exit(code)
