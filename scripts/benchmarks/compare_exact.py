"""Compare same-engine benchmark reports; fail closed on any changed output bit.

This checks exact differential compatibility, not general retrieval quality.
Run via uv; primary metric is the sum of per-case median latencies.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def check_report(first, report):
    """Require the same nonempty ordered workload and declared engine identity."""
    if (report["model"], report["backend"]) != (first["model"], first["backend"]):
        raise ValueError("engine/model identity changed")
    ids = [c["id"] for c in report["cases"]]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("duplicate or empty cases")
    if ids != [c["id"] for c in first["cases"]]:
        raise ValueError("case set/order changed")


def add_case(cases, case):
    """Validate one case before accumulating its two phase timing series."""
    for key in ("ms", "token_ms"):
        if not case[key] or any(not math.isfinite(v) or v <= 0 for v in case[key]):
            raise ValueError("invalid timing")
    output = case["output"]
    if case["rows"] <= 0 or len(output["bits"]) != case["rows"]:
        raise ValueError("invalid row count")
    row = cases.setdefault(case["id"], {"rows": case["rows"], "output": output, "ms": [], "token_ms": []})
    if row["rows"] != case["rows"] or row["output"] != output:
        raise ValueError("non-deterministic baseline/output")
    row["ms"].extend(case["ms"])
    row["token_ms"].extend(case["token_ms"])


def summarize(reports):
    """Validate report identity and aggregate samples without hiding cases."""
    if not reports:
        raise ValueError("no reports")
    cases = {}
    for report in reports:
        check_report(reports[0], report)
        for case in report["cases"]:
            add_case(cases, case)
    return cases


def compare(baseline, candidate):
    """Return timings only after exact same-engine/shape/token/output checks."""
    if not baseline or not candidate:
        raise ValueError("no reports")
    check_report(baseline[0], candidate[0])
    before, after = summarize(baseline), summarize(candidate)
    details = {}
    for key, old in before.items():
        new = after[key]
        if (old["rows"], old["output"]) != (new["rows"], new["output"]):
            raise ValueError(f"accuracy gate failed: changed tokens, shape, ordering or F32 bits in {key}")
        b, c = statistics.median(old["ms"]), statistics.median(new["ms"])
        tb, tc = statistics.median(old["token_ms"]), statistics.median(new["token_ms"])
        details[key] = {"baseline_ms": b, "candidate_ms": c, "speedup": b / c,
                        "baseline_token_ms": tb, "candidate_token_ms": tc, "token_speedup": tb / tc}
    b = sum(v["baseline_ms"] for v in details.values())
    c = sum(v["candidate_ms"] for v in details.values())
    tb = sum(v["baseline_token_ms"] for v in details.values())
    tc = sum(v["candidate_token_ms"] for v in details.values())
    return {"baseline_ms": b, "candidate_ms": c, "speedup": b / c,
            "baseline_token_ms": tb, "candidate_token_ms": tc, "token_speedup": tb / tc,
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
    for key in ("candidate_ms", "candidate_token_ms", "speedup", "token_speedup", "changed_bits", "worst_case_ratio"):
        print(f"METRIC {key}={result[key]}")


if __name__ == "__main__":
    main()
