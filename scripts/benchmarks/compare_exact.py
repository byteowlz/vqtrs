"""Compare same-engine benchmark reports; fail closed on any changed output bit.

This checks exact differential compatibility, not general retrieval quality.
Run via uv; primary metric is the sum of per-case median latencies.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(reports):
    """Validate report identity and aggregate samples without hiding cases."""
    if not reports:
        raise ValueError("no reports")
    first = reports[0]
    cases = {}
    for report in reports:
        if (report["model"], report["backend"]) != (first["model"], first["backend"]):
            raise ValueError("engine/model identity changed")
        ids = [c["id"] for c in report["cases"]]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("duplicate or empty cases")
        if ids != [c["id"] for c in first["cases"]]:
            raise ValueError("case set/order changed")
        for case in report["cases"]:
            if not case["ms"] or any(not math.isfinite(v) or v <= 0 for v in case["ms"]):
                raise ValueError("invalid timing")
            output = case["output"]
            if case["rows"] <= 0 or len(output["bits"]) != case["rows"]:
                raise ValueError("invalid row count")
            row = cases.setdefault(case["id"], {"rows": case["rows"], "output": output, "ms": []})
            if row["rows"] != case["rows"] or row["output"] != output:
                raise ValueError("non-deterministic baseline/output")
            row["ms"].extend(case["ms"])
    return cases


def compare(baseline, candidate):
    """Return timings only after exact same-engine/shape/token/output checks."""
    if (baseline[0]["model"], baseline[0]["backend"]) != (candidate[0]["model"], candidate[0]["backend"]):
        raise ValueError("engine/model identity changed")
    before, after = summarize(baseline), summarize(candidate)
    if list(before) != list(after):
        raise ValueError("case set/order changed")
    details = {}
    for key, old in before.items():
        new = after[key]
        if (old["rows"], old["output"]) != (new["rows"], new["output"]):
            raise ValueError(f"accuracy gate failed: changed tokens, shape, ordering or F32 bits in {key}")
        b, c = statistics.median(old["ms"]), statistics.median(new["ms"])
        details[key] = {"baseline_ms": b, "candidate_ms": c, "speedup": b / c}
    b = sum(v["baseline_ms"] for v in details.values())
    c = sum(v["candidate_ms"] for v in details.values())
    return {"baseline_ms": b, "candidate_ms": c, "speedup": b / c,
            "changed_bits": 0, "worst_case_ratio": max(v["candidate_ms"] / v["baseline_ms"] for v in details.values()),
            "cases": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    result = compare([json.loads(p.read_text()) for p in args.baseline],
                     [json.loads(p.read_text()) for p in args.candidate])
    print(json.dumps(result, indent=2))
    print(f"METRIC suite_ms={result['candidate_ms']}")
    print(f"METRIC speedup={result['speedup']}")
    print(f"METRIC changed_bits={result['changed_bits']}")
    print(f"METRIC worst_case_ratio={result['worst_case_ratio']}")


if __name__ == "__main__":
    main()
