"""Tabulate the WP3 breadth experiment with paired intervals. No sentence about what Jev does."""

import json
from pathlib import Path
import sys
import tempfile

from .evaluation import compare_accuracy, compare_runs

ARMS = ("mix-A", "mix-B", "mix-C")
PAIRS = (("mix-B", "mix-A"), ("mix-C", "mix-A"), ("mix-C", "mix-B"))
SETS = ("panel", "unseen_b77", "unseen_clinc", "phase1", "post_unseen", "public", "mmlu_pro", "k16")


def _load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def _interval(result):
    moved = result["lower"] > 0 or result["upper"] < 0
    return f"{result['mean_improvement']:+.4f} [{result['lower']:+.4f}, {result['upper']:+.4f}]{' *' if moved else ''}"


def _pair(root, first, second, name, family=None):
    a, b = root / first / name / "predictions.jsonl", root / second / name / "predictions.jsonl"
    if not (a.exists() and b.exists()):
        return None
    if family is None:
        return compare_accuracy(a, b), compare_runs(a, b)
    with tempfile.TemporaryDirectory() as scratch:
        paths = []
        for i, source in enumerate((a, b)):
            rows = [line.rstrip("\n") for line in source.open() if line.strip() and json.loads(line)["family"] == family]
            paths.append(Path(scratch) / f"{i}.jsonl")
            paths[-1].write_text("\n".join(rows) + "\n")
        return compare_accuracy(*paths), compare_runs(*paths)


def report(root):
    root = Path(root)
    lines = ["# Phase 2F breadth experiment", "",
             "Arms: mix-A = Phase 1 train only (the Phase 1 winner checkpoint); mix-B = Phase 1 + public (T1/T2); "
             "mix-C = mix-B + T3 pilot rows (model-consensus labels; T4 excluded). Native (uncalibrated) accuracy and NLL. "
             "Paired differences are first arm minus second arm for accuracy, and second-arm NLL minus first-arm NLL "
             "(positive favours the first arm in both columns); 95% intervals from the source-group bootstrap; "
             "`*` marks an interval that excludes zero.", ""]
    for name in SETS:
        lines += [f"## {name}", "", "| arm | requests | accuracy | NLL | calibrated NLL |", "| --- | ---: | ---: | ---: | ---: |"]
        for arm in ARMS:
            m = _load(root / arm / name / "metrics.json")
            if m:
                lines.append(f"| {arm} | {m['requests']} | {m['raw']['accuracy']:.4f} | {m['raw']['nll']:.4f} | {m['calibrated']['nll']:.4f} |")
        lines += ["", "| pair | groups | accuracy difference [95%] | NLL improvement [95%] |", "| --- | ---: | ---: | ---: |"]
        for first, second in PAIRS:
            result = _pair(root, first, second, name)
            if result:
                lines.append(f"| {first} vs {second} | {result[0]['groups']} | {_interval(result[0])} | {_interval(result[1])} |")
        lines.append("")
    families = sorted({f for arm in ARMS for f in (_load(root / arm / "public" / "metrics.json") or {}).get("by_family", {})})
    if families:
        lines += ["## public by source", "", "| source | " + " | ".join(f"{arm} acc | {arm} NLL" for arm in ARMS) + " |", "| --- |" + " ---: |" * (2 * len(ARMS))]
        for family in families:
            cells = []
            for arm in ARMS:
                m = _load(root / arm / "public" / "metrics.json")
                f = (m or {}).get("by_family", {}).get(family)
                cells.append(f"{f['raw']['accuracy']:.4f} | {f['raw']['nll']:.4f}" if f else "n/a | n/a")
            lines.append(f"| {family} | " + " | ".join(cells) + " |")
        lines += ["", "| source | pair | groups | accuracy difference [95%] | NLL improvement [95%] |", "| --- | --- | ---: | ---: | ---: |"]
        for family in families:
            for first, second in PAIRS:
                result = _pair(root, first, second, "public", family)
                if result:
                    lines.append(f"| {family} | {first} vs {second} | {result[0]['groups']} | {_interval(result[0])} | {_interval(result[1])} |")
        lines.append("")
    lines += ["## Training", "", "| arm | train requests | dev requests | training minutes | selected step | steps | budget limited |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for arm in ARMS:
        s = _load(root / arm / "summary.json")
        if s:
            lines.append(f"| {arm} | {s['train_requests']} | {s['dev_requests']} | {s['elapsed_seconds'] / 60:.1f} | {s['best_step']} | {s['steps']} | {s['budget_limited']} |")
    lines += ["", "## Gate", ""]
    moved = []
    for name in SETS:
        for first, second in PAIRS[:2]:
            result = _pair(root, first, second, name)
            if result:
                for label, r in (("accuracy", result[0]), ("NLL", result[1])):
                    if r["lower"] > 0 or r["upper"] < 0:
                        moved.append(f"{name} {label} ({first} vs {second}: {r['mean_improvement']:+.4f})")
    if moved:
        lines.append("Spec gate: the data investment moved a number only if the paired interval excludes zero. Intervals excluding zero: " + "; ".join(moved) + ".")
    else:
        lines.append("Spec gate: the data investment moved a number only if the paired interval excludes zero. No paired interval against mix-A excludes zero; the data investment moved nothing by this gate.")
    lines += ["", "Confound: mix-A was selected on Phase 1 dev alone while mix-B and mix-C were selected on Phase 1 dev plus public dev. "
              "T3 labels are model consensus, audited on 90 rows with 0 wrong (pooled 95% upper bound on the error rate about 3.3%). "
              "One seed each. These numbers describe our graphs on our data; they say nothing about Jev's implementation.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(sys.argv[1] if len(sys.argv) > 1 else "runs/phase2/breadth"))
