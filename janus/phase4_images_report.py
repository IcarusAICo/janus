"""Paired text-versus-image accuracy and NLL by family for the Phase 4 image-state runs. No sentence about what Jev does."""

import json
from pathlib import Path
import sys

ARMS = ("qwen35_4b_images", "ref_qwen35_4b_tree")
PUBLIC = ("scienceqa", "ai2d", "mmmu")


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _correct(row):
    p, t = row["probabilities"], row["target"]
    return t[max(range(len(p)), key=p.__getitem__)]


def paired(text_predictions, image_predictions):
    """{family or 'all': {count, text_accuracy, image_accuracy, text_nll, image_nll, image_minus_text_accuracy}} over
    the (group_id, question_id) pairs present in both prediction files; unmatched rows are dropped, so the two
    columns always score the same questions."""
    image = {(r["group_id"], r["question_id"]): r for r in _rows(image_predictions)}
    sums = {}
    for row in _rows(text_predictions):
        twin = image.get((row["group_id"], row["question_id"]))
        if twin is None:
            continue
        for key in (row["family"], "all"):
            cell = sums.setdefault(key, {"count": 0, "text_correct": 0., "image_correct": 0., "text_nll": 0., "image_nll": 0.})
            cell["count"] += 1
            cell["text_correct"] += _correct(row)
            cell["image_correct"] += _correct(twin)
            cell["text_nll"] += row["nll"]
            cell["image_nll"] += twin["nll"]
    out = {}
    for key, cell in sums.items():
        n = cell["count"]
        out[key] = {"count": n, "text_accuracy": cell["text_correct"] / n, "image_accuracy": cell["image_correct"] / n,
                    "text_nll": cell["text_nll"] / n, "image_nll": cell["image_nll"] / n,
                    "image_minus_text_accuracy": (cell["image_correct"] - cell["text_correct"]) / n}
    return out


def report(root, arms=ARMS, public=PUBLIC):
    root = Path(root)
    lines = ["# Phase 4: image states", "",
             "One checkpoint per arm, scored on the text test (data/phase1-v1/test.jsonl) and on its image rendering "
             "(data/phase1-image-v1/test.jsonl); rows are paired by group id and question id. qwen35_4b_images trained on "
             "the image rendering; ref_qwen35_4b_tree is the text-trained Phase 3 checkpoint reading the images zero-shot. "
             "Entries: accuracy (target mass at the argmax) and raw NLL, text then image.", ""]
    for arm in arms:
        text, image = root / arm / "text" / "predictions.jsonl", root / arm / "image" / "predictions.jsonl"
        if not (text.exists() and image.exists()):
            continue
        table = paired(text, image)
        lines += [f"## {arm}", "", "| family | n | text acc | image acc | image - text | text NLL | image NLL |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for family in sorted(table, key=lambda k: (k == "all", k)):
            c = table[family]
            lines.append(f"| {family} | {c['count']} | {c['text_accuracy']:.3f} | {c['image_accuracy']:.3f} | "
                         f"{c['image_minus_text_accuracy']:+.3f} | {c['text_nll']:.3f} | {c['image_nll']:.3f} |")
        lines.append("")
    lines += ["## Public image sets (raw accuracy / raw NLL / calibrated NLL, own temperature per set)", "",
              "| arm | " + " | ".join(public) + " |", "| --- |" + " ---: |" * len(public)]
    for arm in arms:
        cells = []
        for name in public:
            path = root / arm / f"public_{name}" / "metrics.json"
            if not path.exists():
                cells.append("n/a")
                continue
            m = json.loads(path.read_text())
            cells.append(f"{m['raw']['accuracy']:.3f} / {m['raw']['nll']:.3f} / {m['calibrated']['nll']:.3f}")
        if any(c != "n/a" for c in cells):
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "One seed each; differences without paired intervals are screening signals. These numbers describe our "
              "graphs on our data; they say nothing about Jev's implementation.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(sys.argv[1] if len(sys.argv) > 1 else "runs/phase4/images"))
