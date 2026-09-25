"""Per-K and per-level accuracy and NLL for the Phase 4 cardinality sweep. No sentence about what Jev does."""

import json
import math
from pathlib import Path
import sys

ARMS = ("score_independent", "score_full")
REFERENCE = ("ref_g2",)


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def cells(predictions_path):
    """{(kind, cardinality): sums of count, correct, nll, calibrated_nll and (Score only) |expected level - gold|}."""
    out = {}
    for row in _rows(predictions_path):
        if row["kind"] not in ("choice", "score"):
            continue
        p, t = row["probabilities"], row["target"]
        cell = out.setdefault((row["kind"], row["cardinality"]),
                              {"count": 0, "correct": 0., "nll": 0., "calibrated_nll": 0., "level_error": 0.})
        cell["count"] += 1
        cell["correct"] += t[max(range(len(p)), key=p.__getitem__)]
        cell["nll"] += row["nll"]
        cell["calibrated_nll"] += row.get("calibrated_nll", row["nll"])
        if row["kind"] == "score":
            cell["level_error"] += abs(sum(i * v for i, v in enumerate(p)) - max(range(len(t)), key=t.__getitem__))
    return out


def _fmt(cell, kind):
    if not cell or not cell["count"]:
        return "n/a"
    n = cell["count"]
    text = f"{cell['correct'] / n:.3f} / {cell['nll'] / n:.3f} / {cell['calibrated_nll'] / n:.3f}"
    return text + (f" / {cell['level_error'] / n:.2f}" if kind == "score" else "")


def report(root, arms=ARMS, reference=REFERENCE):
    root = Path(root)
    labels = [*reference, *arms]
    tables = {label: cells(root / label / "cardinality" / "predictions.jsonl")
              for label in labels if (root / label / "cardinality" / "predictions.jsonl").exists()}
    lines = ["# Phase 4: Score semantics and cardinality sweep", "",
             "Arms: score_independent (tree; a Score level leaf sees the state, the instructions and only its own "
             "description) and score_full (tree; the block also lists every level with its index). Both trained on "
             "Phase 1 plus the cardinality training set; ref_g2 is the Phase 1 G2 checkpoint (Phase 1 data only). "
             "Cells are 200 states each (data/cardinality-v1/test.jsonl). Entries: accuracy / raw NLL / calibrated NLL"
             " (Score adds / mean |expected level - gold level|). Uniform NLL is log K.", ""]
    for kind, title, unit in (("choice", "Choice by option count K", "K"), ("score", "Score by level count", "levels")):
        sizes = sorted({c[1] for table in tables.values() for c in table if c[0] == kind})
        lines += [f"## {title}", "", f"| {unit} | uniform NLL | " + " | ".join(tables) + " |", "| --- | ---: |" + " ---: |" * len(tables)]
        for size in sizes:
            lines.append(f"| {size} | {math.log(size):.3f} | " + " | ".join(_fmt(t.get((kind, size)), kind) for t in tables.values()) + " |")
        lines.append("")
    lines += ["## Phase 1 test split (raw)", "", "| arm | accuracy | NLL | selected step | training minutes |", "| --- | ---: | ---: | ---: | ---: |"]
    for label in labels:
        m, s = _load(root / label / "phase1" / "metrics.json"), _load(root / label / "summary.json")
        if m or s:
            fmt = lambda value: "n/a" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
            lines.append(f"| {label} | {fmt(m and m['raw']['accuracy'])} | {fmt(m and m['raw']['nll'])} | "
                         f"{fmt(s and s['best_step'])} | {fmt(s and s['elapsed_seconds'] / 60)} |")
    lines += ["", "One seed each; differences without paired intervals are screening signals. Cardinality cells use compact "
              "option text and even K, so no cell asks for the median. These numbers describe our graphs on our data; "
              "they say nothing about Jev's implementation.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(sys.argv[1] if len(sys.argv) > 1 else "runs/phase4/cardinality"))
