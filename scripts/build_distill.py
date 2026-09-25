"""Distillation mix: gold targets blended with teacher distributions (scripts/teacher_label.py output).

A one-hot gold target becomes ALPHA * gold + (1 - ALPHA) * teacher, so the gold option always keeps at least ALPHA
of the mass and the teacher only spreads the rest; a gold target that is already a distribution (the probability
family) is kept as is. Rows without a teacher label keep their gold. Extra mixes (e.g. multilingual) are appended
with their gold, and their dev/calibration splits are appended to the base ones.

    python scripts/build_distill.py OUT_DIR TEACHER.jsonl BASE_DIR [EXTRA_DIR ...] [--alpha 0.5]
"""
import argparse
import json
import random
from pathlib import Path

from janus.data import state_hash
from janus.schema import Request

p = argparse.ArgumentParser()
p.add_argument("out"); p.add_argument("teacher"); p.add_argument("base"); p.add_argument("extra", nargs="*")
p.add_argument("--alpha", type=float, default=0.5)
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True)


def blend(row):
    qs = row["questions"]
    for q, t in zip(qs.values() if isinstance(qs, dict) else qs, row.pop("teacher")):
        g = q["target"]
        if sorted(g) == [0.0] * (len(g) - 1) + [1.0]:
            q["target"] = [a.alpha * x + (1 - a.alpha) * y for x, y in zip(g, t)]
    return row


rows = [blend(json.loads(l)) for l in open(a.teacher)]
blended = len(rows)
base_train = Path(a.base, "train.jsonl").read_text().splitlines()
rows += [json.loads(l) for l in base_train[blended:]]  # the teacher labelled a prefix of the base train split
for d in a.extra:
    rows += [json.loads(l) for l in open(Path(d, "train.jsonl"))]
random.Random(17).shuffle(rows)
with open(out / "train.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
counts = {"train": len(rows), "teacher_blended": blended}
# the same utterance can sit in two upstream splits (one MASSIVE fr-FR text does); held-out splits lose it, not train
seen = {state_hash(Request.from_dict(r).state) for r in rows} | {r.get("group_id") for r in rows} - {None, ""}
for split in ("dev", "calibration"):
    lines = [l for d in [a.base, *a.extra] for l in Path(d, f"{split}.jsonl").read_text().splitlines()]
    lines = [l for l in lines if not ({state_hash(Request.from_dict(json.loads(l)).state), json.loads(l).get("group_id")} & seen)]
    Path(out, f"{split}.jsonl").write_text("\n".join(lines) + "\n")
    counts[split] = len(lines)
json.dump({"dataset": "JANUS_DISTILL", "base": a.base, "extra": a.extra, "teacher": a.teacher, "alpha": a.alpha,
           "counts": counts}, open(out / "manifest.json", "w"), indent=2)
print(counts)
