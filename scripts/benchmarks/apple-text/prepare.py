import json, os
from pathlib import Path
from tokenizers import Tokenizer

root = Path(os.environ.get("EG2_BENCH_WORK_DIR", "/tmp/vqtrs-apple-bench"))
root.mkdir(parents=True, exist_ok=True)
checkpoint = Path(os.environ["EG2_MODEL_DIR"])
tok = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
tok.no_truncation()
tok.no_padding()
docs = [
    "A bicycle chain needs regular cleaning and oil. Fix a puncture with a spare tube and tire levers.",
    "Bread dough rises when yeast ferments. Knead flour and water, rest the dough, then bake in a hot oven.",
    "Tomatoes grow well in sunny gardens. Water the roots and improve the soil with compost.",
    "Passenger trains connect cities. Check the departure platform and timetable before boarding.",
    "Astronomers observe stars and distant galaxies. A telescope gathers light from the night sky.",
    "A guitarist tunes the strings before playing music. Practice chords slowly with a metronome.",
    "Ocean currents carry warm water around the planet. Coral reefs shelter fish and other marine life.",
    "Software releases need tests and versioned changes. Review code, build the application, and publish release notes.",
]
queries = [
    "How do I repair my bike?",
    "How do I bake a loaf?",
    "What helps us see distant stars?",
    "How should software releases be prepared?",
]
q = lambda s: "task: search result | query: " + s
d = lambda s: "title: none | text: " + s
long = lambda n: d(
    " ".join(["Testing software changes improves reliable releases."] * n)
)
cases = [
    ("query-1", [q(queries[0])]),
    ("query-8", [q(x) for x in queries * 2]),
    ("docs-8", [d(x) for x in docs]),
    ("queries-4", [q(x) for x in queries]),
    ("text-medium", [long(16)]),
    ("text-near-cap", [long(65)]),
    ("bulk-32", [d(x + f" Review example {i}.") for i in range(4) for x in docs]),
]
items = []
for name, texts in cases:
    ids = [e.ids for e in tok.encode_batch(texts)]
    assert all(0 < len(x) <= 512 for x in ids), (name, [len(x) for x in ids])
    items.append(dict(id=name, texts=texts, ids=ids))
print([(x["id"], [len(i) for i in x["ids"]]) for x in items])
(root / "fixtures.json").write_text(
    json.dumps(
        dict(
            schema="vqtrs-apple-text-probe-v1",
            cases=items,
            labels={"queries-4": [[0], [1], [4], [7]]},
        ),
        indent=2,
    )
)
