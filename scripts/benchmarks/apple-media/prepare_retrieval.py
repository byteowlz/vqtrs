import csv
import hashlib
import io
import json
import os
import random
import zipfile
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(os.environ.get("EG2_BENCH_WORK_DIR", "/tmp/vqtrs-native-media"))
MODEL = Path(os.environ["EG2_MODEL_DIR"])
tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
tok.no_padding()
tok.no_truncation()
rows = []
datasets = []
for name in ["scifact", "nfcorpus"]:
    archive = ROOT / (name + ".zip")
    with zipfile.ZipFile(archive) as z:

        def read(path, archive=z, dataset=name):
            info = archive.getinfo(dataset + "/" + path)
            assert info.file_size <= 64 * 1024 * 1024
            return archive.read(info).decode("utf-8")

        queries = {
            q["_id"]: q["text"]
            for q in map(json.loads, read("queries.jsonl").splitlines())
        }
        corpus = {
            d["_id"]: d for d in map(json.loads, read("corpus.jsonl").splitlines())
        }
        labels = {}
        for q in csv.DictReader(io.StringIO(read("qrels/test.tsv")), delimiter="\t"):
            if float(q["score"]) > 0:
                labels.setdefault(q["query-id"], {})[q["corpus-id"]] = float(q["score"])
    qids = sorted(labels)[:16]
    relevant = {d for q in qids for d in labels[q]}
    assert relevant <= set(corpus)
    other = sorted(set(corpus) - relevant)
    random.Random(20261009).shuffle(other)
    dids = sorted(relevant | set(other[:96]))
    capped = 0

    def add(identifier, text, prefix):
        global capped
        prepared = prefix + text
        ids = tok.encode(prepared).ids
        if len(ids) > 512 or len(prepared.encode()) > 16384:
            lo, hi = 0, len(text)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                p = prefix + text[:mid]
                if len(tok.encode(p).ids) <= 512 and len(p.encode()) <= 16384:
                    lo = mid
                else:
                    hi = mid - 1
            prepared = prefix + text[:lo]
            ids = tok.encode(prepared).ids
            capped += 1
        assert 2 <= len(ids) <= 512 and ids[0] == 2 and ids[-1] == 1
        rows.append({"id": identifier, "text": prepared, "ids": ids})

    for q in qids:
        add(name + ":query:" + q, queries[q], "task: search result | query: ")
    for d in dids:
        item = corpus[d]
        add(
            name + ":doc:" + d,
            (item.get("title", "") + "\n" + item["text"]).strip(),
            "title: none | text: ",
        )
    datasets.append(
        {
            "dataset": name,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "queries": qids,
            "documents": dids,
            "labels": {q: labels[q] for q in qids},
            "explicitly_capped_rows": capped,
            "scope": "first16 sorted test queries, all their relevant documents plus96 seeded distractors; NOT full BEIR",
            "context": "identical explicitly prepared <=512-token text, not backend truncation",
        }
    )
assert len(rows) <= 2048
(ROOT / "retrieval-inputs.json").write_text(json.dumps({"rows": rows}))
(ROOT / "retrieval-manifest.json").write_text(
    json.dumps({"datasets": datasets, "rows": len(rows)}, indent=2)
)
(ROOT / "retrieval-cpu-spot.json").write_text(
    json.dumps({"rows": rows[:: max(1, len(rows) // 16)][:16]})
)
print(
    [
        (
            d["dataset"],
            len(d["queries"]),
            len(d["documents"]),
            d["explicitly_capped_rows"],
        )
        for d in datasets
    ],
    len(rows),
)
