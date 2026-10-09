"""Bounded two-stage GPU -> CoreML text overlap; no splitting, tails, or averaging."""

import json
import os
import runpy
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

sys.argv = ["coreml_probe.py", "audio"]
ns = runpy.run_path(str(Path(__file__).with_name("coreml_probe.py")))
model = ns["model"]
text = ns["text"]
table = ns["table"]
media = ns["MEDIA"]
jobs = []
for name in [
    "chirp_2370ms.wav",
    "short_300ms.wav",
    "silence_1000ms.wav",
    "speech_7100ms_pcm16.wav",
]:
    samples, sr = sf.read(media / name, dtype="float32")
    assert sr == 16000 and len(samples) <= 160000 and np.isfinite(samples).all()
    wave = np.zeros((1, 160160), dtype=np.float32)
    wave[0, 160 : 160 + len(samples)] = samples
    frames = max(0, min(1000, (160 + len(samples) - 321) // 160 + 1))
    mask = np.zeros((1, 1000), dtype=np.float16)
    mask[0, :frames] = 1
    jobs.append((name, wave, mask, (frames + 3) // 4))


def gpu(job):
    _, wave, mask, count = job
    tokens = model.predict({"waveform": wave, "frame_mask": mask})["audio_tokens"][
        0, :count
    ]
    return np.concatenate(
        [table([2, 256000]), tokens.astype(np.float16), table([258883, 1])]
    )


def sequential():
    return np.array([text(gpu(job)) for job in jobs])


def overlap():
    # One producer; at most two windows resident/in flight. Consume in input order.
    output = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = deque([pool.submit(gpu, jobs[0])])
        for index in range(len(jobs)):
            if index + 1 < len(jobs):
                futures.append(pool.submit(gpu, jobs[index + 1]))
            output.append(text(futures.popleft().result()))
    return np.array(output)


for _ in range(2):
    sequential()
    overlap()
times = {"sequential": [], "overlap": []}
worst = 0.0
for _ in range(3):
    vectors = {}
    for mode, fn in [("sequential", sequential), ("overlap", overlap)]:
        start = perf_counter()
        vectors[mode] = fn()
        times[mode].append((perf_counter() - start) * 1000)
    worst = max(worst, float(abs(vectors["sequential"] - vectors["overlap"]).max()))
assert worst < 0.002 and np.isfinite(vectors["overlap"]).all()
report = {
    "rows_per_trial": len(jobs),
    "warmups": 2,
    "trials": 3,
    "ms": times,
    "max_abs_overlap_vs_sequential": worst,
    "ownership": "one GPU producer, ordered ANE-preferred text consumer, two-window lookahead",
    "placement": "explicit CPU_AND_GPU audio / CPU_AND_NE text, not a runtime device trace",
    "timing": "warm prepared waveform arrays -> prediction/table gather/text prediction/host copy; no file decode or transport",
    "scope": "four synthetic clips, exploratory M2 measurement, not the M5 author demo",
}
(
    Path(os.environ.get("EG2_BENCH_WORK_DIR", "/tmp/vqtrs-native-media"))
    / "coreml-audio-pipeline-results.json"
).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
