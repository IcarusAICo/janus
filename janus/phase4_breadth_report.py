"""Tabulate the four-way breadth ablation: one table per axis, with the held-out family columns. No sentence about
what Jev does."""

import json
from pathlib import Path
import sys

from .synth.ablation import ARMS, AXES, HELD_OUT

SETS = ("families", "phase1", "panel", "unseen_clinc", "unseen_b77", "public", "cardinality", "mmlu_pro")
FAMILY_SET = {"retrieval": "families", "evidence": "families", "record_match": "families", "rel": "phase1"}


def _load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def _cell(block):
    return f"{block['raw']['accuracy']:.3f} / {block['raw']['nll']:.3f}" if block else "n/a"


def row(root, arm, arms=ARMS):
    """One table row: training summary, accuracy / NLL per set, then per held-out family (marked when held out)."""
    root = Path(root)
    summary = _load(root / arm / "summary.json")
    cells = [arm]
    if summary:
        cells += [str(summary["train_requests"]), str(arms.get(arm, {}).get("epochs", 1)),
                  f"{summary['best_step']} / {summary['steps']}", f"{summary['elapsed_seconds'] / 60:.0f}"]
    else:
        cells += ["n/a"] * 4
    metrics = {name: _load(root / arm / name / "metrics.json") for name in SETS}
    cells += [_cell(metrics[name]) for name in SETS]
    for family in HELD_OUT:
        block = ((metrics[FAMILY_SET[family]] or {}).get("by_family") or {}).get(family)
        cells.append(_cell(block) + (" (held out)" if arms.get(arm, {}).get("drop_family") == family else ""))
    return cells


def report(root, axes=AXES, arms=ARMS):
    header = ["arm", "train rows", "epochs", "selected / steps", "minutes", *SETS, *[f"{f} (family)" for f in HELD_OUT]]
    lines = ["# Phase 4 breadth ablation", "",
             "Every arm changes one axis against the 25k default mix (Qwen3.5-4B hybrid, LoRA, tree graph, seed 17, dev-NLL "
             "selection on the arm's own dev split). Cells are native accuracy / NLL. The last four columns are the by-family "
             "read-outs on the families test split (retrieval, evidence, record_match) and the Phase 1 test split (rel); an arm "
             "that trained without a family is marked there. One seed per arm; T3 labels are model consensus, not ground truth. "
             "These numbers describe our graphs on our data.", ""]
    for axis, names in axes.items():
        lines += [f"## {axis}", "", "| " + " | ".join(header) + " |", "| --- | ---: | ---: | ---: | ---: |" + " ---: |" * (len(header) - 5)]
        for arm in names:
            lines.append("| " + " | ".join(row(root, arm, arms)) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(sys.argv[1] if len(sys.argv) > 1 else "runs/phase4/breadth"))
