"""Tabulate the Phase 1 graph screen by scorecard. No sentence about what Jev does."""

import json
import math
from pathlib import Path
import sys


def discover_graphs(root):
    """Run directories under root (Phase 1: g0, g2, g4; Phase 2 adds the remaining candidates)."""
    root = Path(root)
    return tuple(sorted(p.name for p in root.iterdir()
                        if p.is_dir() and ((p / "summary.json").exists() or (p / "panel").is_dir())))


def _load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def _entropy(predictions_path):
    values = []
    for line in Path(predictions_path).read_text().splitlines():
        if line.strip():
            p = json.loads(line)["probabilities"]
            values.append(-sum(x * math.log(x) for x in p if x > 0) / math.log(len(p)))
    return sum(values) / len(values) if values else float("nan")


def _uniform_nll(predictions_path):
    values = [math.log(len(json.loads(line)["probabilities"])) for line in Path(predictions_path).read_text().splitlines() if line.strip()]
    return sum(values) / len(values) if values else float("nan")


def report(root):
    root = Path(root)
    GRAPHS = discover_graphs(root)
    lines = ["# Phase 1 graph screen", "", "## Predictive utility on the 800-state panel (native)", "",
             "| graph | accuracy | NLL | Brier | ECE | calibrated NLL |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for g in GRAPHS:
        m = _load(root / g / "panel" / "metrics.json")
        if m:
            lines.append(f"| {g} | {m['raw']['accuracy']:.4f} | {m['raw']['nll']:.4f} | {m['raw']['brier']:.4f} | {m['raw']['ece']:.4f} | {m['calibrated']['nll']:.4f} |")
    lines += ["", "## Per-family accuracy and NLL on the Phase 1 test split (native)", ""]
    families = sorted({f for g in GRAPHS for f in (_load(root / g / "test" / "metrics.json") or {}).get("by_family", {})})
    lines += ["| graph | " + " | ".join(f"{f} acc | {f} NLL" for f in families) + " |", "| --- |" + " ---: |" * (2 * len(families))]
    for g in GRAPHS:
        m = _load(root / g / "test" / "metrics.json")
        if m:
            cells = [f"{m['by_family'][f]['raw']['accuracy']:.4f} | {m['by_family'][f]['raw']['nll']:.4f}" if f in m["by_family"] else "n/a | n/a" for f in families]
            lines.append(f"| {g} | " + " | ".join(cells) + " |")
    lines += ["", "## Honest ignorance on held-out posterior families (uniform NLL is the honest-ignorance reference; modestly below it is reachable from base rates alone, far below it suggests leakage, above it is confident error)", "", "| graph | accuracy | NLL | uniform NLL | mean normalised entropy |", "| --- | ---: | ---: | ---: | ---: |"]
    for g in GRAPHS:
        m = _load(root / g / "test_post_unseen" / "metrics.json")
        if m:
            lines.append(f"| {g} | {m['raw']['accuracy']:.4f} | {m['raw']['nll']:.4f} | {_uniform_nll(root / g / 'test_post_unseen' / 'predictions.jsonl'):.4f} | {_entropy(root / g / 'test_post_unseen' / 'predictions.jsonl'):.4f} |")
    lines += ["", "## Order robustness and K-generalisation", "", "| graph | panel permutation TV | panel flip rate | K=16 accuracy | K=16 NLL |", "| --- | ---: | ---: | ---: | ---: |"]
    for g in GRAPHS:
        p, k = _load(root / g / "panel" / "metrics.json"), _load(root / g / "k16" / "metrics.json")
        if p and k:
            lines.append(f"| {g} | {p['choice_order']['mean_total_variation']:.4f} | {p['choice_order']['flip_rate']:.4f} | {k['raw']['accuracy']:.4f} | {k['raw']['nll']:.4f} |")
    latencies = {g: _load(root / g / "panel-latency.json") for g in GRAPHS}
    compute_keys = [k for k in ("backbone_tokens", "decoder_cross_attention_pairs")
                    if any(k in (l or {}).get("compute", {}) for l in latencies.values())]
    lines += ["", "## Efficiency", "", "| graph | training minutes | selected step | panel p50 ms | panel p95 ms |"
              + "".join(f" mean {k.replace('_', ' ')} |" for k in compute_keys),
              "| --- | ---: | ---: | ---: | ---: |" + " ---: |" * len(compute_keys)]
    for g in GRAPHS:
        s, l = _load(root / g / "summary.json"), latencies[g]
        if s and l:
            extra = "".join(f" {l['compute'][k]:.0f} |" if k in l.get("compute", {}) else " n/a |" for k in compute_keys)
            lines.append(f"| {g} | {s['elapsed_seconds'] / 60:.1f} | {s['best_step']} | {l['p50_ms']:.2f} | {l['p95_ms']:.2f} |" + extra)
    if compute_keys:
        lines += ["", "Backbone tokens count every token the backbone processes per request (state plus branches, "
                  "or state plus each question sequence in the decoder graph); decoder cross-attention pairs are "
                  "decoder layers times branch tokens times state tokens, the work the decoder graph moves out of the backbone."]
    lines += ["", "One seed each. Differences without paired intervals are screening signals, not confirmed results. "
              "These numbers describe our graphs on our data; they say nothing about Jev's implementation.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(sys.argv[1] if len(sys.argv) > 1 else "runs/phase1/screen"))
