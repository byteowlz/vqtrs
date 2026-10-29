# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.22.2"]
# ///
"""Derive the fixed long-context diagnostic fixture; no downloads or truncation."""
import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


def repeat_to_target(source, target, tokenizer):
    text = source
    for _ in range(127):
        following = text + "\n" + source
        if len(tokenizer.encode(following, add_special_tokens=True).ids) > target:
            ids = tokenizer.encode(text, add_special_tokens=True).ids
            if not target - 128 <= len(ids) <= target:
                raise ValueError("unexpected tokenizer/source geometry")
            return text, ids
        text = following
    raise ValueError("repetition bound exceeded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source_path = Path(__file__).resolve().parents[2] / "docs/benchmarks/apple-text-fixtures.json"
    raw = source_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "ddc81d848622dcedb109b73a80d7f1a75c799345d150599b688c5e210a53f2c5":
        raise ValueError("source fixture changed; qualify a new dataset version")
    fixture = json.loads(raw)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    source = next(c for c in fixture["cases"] if c["id"] == "text-medium")["texts"][0]
    rows = {target: repeat_to_target(source, target, tokenizer) for target in (1024, 2048, 2304)}
    for target, (text, ids) in rows.items():
        fixture["cases"].append({"id": f"long-{target}", "texts": [text], "ids": [ids]})
    fixture["cases"].append({"id": "long-ragged-pair", "texts": [rows[1024][0], rows[2048][0]],
                             "ids": [rows[1024][1], rows[2048][1]]})
    fixture["protocol"] = "Original seven short cases preserved; four explicit long cases derived by whole-text repetition, no tokenizer truncation."
    args.output.write_text(json.dumps(fixture, ensure_ascii=False, indent=2))
    print(json.dumps({"sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "cases": len(fixture["cases"]),
                      "rows": sum(len(c["texts"]) for c in fixture["cases"]),
                      "long_tokens": [len(ids) for _, ids in rows.values()]}))


if __name__ == "__main__":
    main()
