"""Tabulate the rank-capability screen by request kind. No sentence about what Jev does."""

import json
from pathlib import Path
import re
import sys

from .synth.rankworlds import RANK_KINDS, request_kind
from .synth.worlds import REL_KINDS

ARMS = ("g4_deep", "g4_long", "g4_deep_long", "pair", "g2_rank", "g4_rank", "pair_rank")
REFERENCE = {"g2": Path("runs/phase1/screen/g2"), "g4": Path("runs/phase1/screen/g4")}
PHASE1_TEST = Path("data/phase1-v1/test.jsonl")
RANK_TEST = Path("data/rank-v1/test.jsonl")
RANK_QUESTIONS = ("rank:pair", "rank:count", "rank:pick")


def _load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _correct(row):
    p = row["probabilities"]
    return float(row["target"][max(range(len(p)), key=p.__getitem__)])


def _requests(data_path):
    return {row["group_id"]: row for row in _rows(data_path)} if Path(data_path).exists() else {}


def _kind(row, requests):
    """The record's request kind; records written before the field existed are joined to the data file."""
    if "request_kind" in row:
        return row["request_kind"]
    raw = requests.get(row["group_id"])
    if raw is None:
        return None
    state = raw["state"] if isinstance(raw["state"], str) else json.dumps(raw["state"], sort_keys=True)
    return request_kind(state, row.get("family"))


def accuracy_by_kind(predictions_path, data_path, question_ids):
    """{request kind: {count, correct}} over the given question ids."""
    requests = _requests(data_path)
    out = {}
    for row in _rows(predictions_path):
        if row["question_id"] not in question_ids:
            continue
        kind = _kind(row, requests)
        if kind is None:
            continue
        cell = out.setdefault(kind, {"count": 0, "correct": 0})
        cell["count"] += 1
        cell["correct"] += int(_correct(row))
    return out


def _price(description):
    try:
        return json.loads(description)["price"]
    except (ValueError, KeyError, TypeError):
        match = re.search(r"\$(\d+)", description)
        return int(match.group(1)) if match else None


def cheapest_share(predictions_path, data_path):
    """On rel:second_cheapest, the share of picks that chose the cheapest option instead; None without data."""
    requests = _requests(data_path)
    picked, total = 0, 0
    for row in _rows(predictions_path):
        if row["question_id"] != "rel:pick" or _kind(row, requests) != "rel:second_cheapest":
            continue
        raw = requests.get(row["group_id"])
        if raw is None:
            continue
        criteria = raw["questions"]["rel:pick"]["criteria"]
        prices = [_price(criteria[key]) for key in row["keys"]]
        if any(p is None for p in prices):
            continue
        p = row["probabilities"]
        total += 1
        picked += int(max(range(len(p)), key=p.__getitem__) == prices.index(min(prices)))
    return picked / total if total else None


def _cell(table, kind):
    cell = table.get(kind)
    return f"{cell['correct'] / cell['count']:.3f}" if cell and cell["count"] else "n/a"


def report(root, arms=ARMS, reference=REFERENCE, phase1_test=PHASE1_TEST, rank_test=RANK_TEST):
    root = Path(root)
    # (label, phase1 predictions, rank predictions, panel metrics, summary)
    rows = []
    for name, run in reference.items():
        rows.append((name, Path(run) / "test" / "predictions.jsonl", root / f"ref_{name}" / "rank" / "predictions.jsonl",
                     Path(run) / "panel" / "metrics.json", Path(run) / "summary.json", Path(run) / "test" / "metrics.json"))
    for arm in arms:
        rows.append((arm, root / arm / "phase1" / "predictions.jsonl", root / arm / "rank" / "predictions.jsonl",
                     root / arm / "panel" / "metrics.json", root / arm / "summary.json", root / arm / "phase1" / "metrics.json"))
    rel_kinds = [f"rel:{k}" for k in REL_KINDS]
    rank_kinds = [f"rank:{k}" for k in RANK_KINDS]
    lines = ["# Phase 3E rank-capability screen", "",
             "Reference rows g2 and g4 are the Phase 1 screen checkpoints (Phase 1 data only). Arms g4_deep, g4_long and "
             "g4_deep_long are the set head deeper (4 layers, width 512, 8 heads), longer (3 epochs) or both; pair is the "
             "pairwise comparison head; the *_rank arms train on Phase 1 plus the rank family. Native (uncalibrated) "
             "accuracy; chance on rel:pick is about 0.18.", "",
             "## rel:pick accuracy by kind on the Phase 1 test split", ""]
    counts = {}
    for label, phase1, *_ in rows:
        if phase1.exists():
            counts = accuracy_by_kind(phase1, phase1_test, ("rel:pick",))
            break
    header = " | ".join(f"{k.split(':', 1)[1]} (n={counts.get(k, {}).get('count', 0)})" for k in rel_kinds)
    lines += [f"| arm | {header} | all rel:pick | second-cheapest picks that chose the cheapest |", "| --- |" + " ---: |" * (len(rel_kinds) + 2)]
    for label, phase1, *_ in rows:
        if not phase1.exists():
            continue
        table = accuracy_by_kind(phase1, phase1_test, ("rel:pick",))
        total = sum(v["count"] for v in table.values())
        overall = f"{sum(v['correct'] for v in table.values()) / total:.3f}" if total else "n/a"
        share = cheapest_share(phase1, phase1_test)
        lines.append(f"| {label} | " + " | ".join(_cell(table, k) for k in rel_kinds) + f" | {overall} | {share if share is None else f'{share:.2f}'} |")
    lines += ["", "## rank accuracy by kind on the rank test split (rank:pair, rank:count, rank:pick)", ""]
    counts = {}
    for _, _, rank, *__ in rows:
        if rank.exists():
            counts = accuracy_by_kind(rank, rank_test, RANK_QUESTIONS)
            break
    header = " | ".join(f"{k.split(':', 1)[1]} (n={counts.get(k, {}).get('count', 0)})" for k in rank_kinds)
    lines += [f"| arm | {header} | all |", "| --- |" + " ---: |" * (len(rank_kinds) + 1)]
    for label, _, rank, *__ in rows:
        if not rank.exists():
            continue
        table = accuracy_by_kind(rank, rank_test, RANK_QUESTIONS)
        total = sum(v["count"] for v in table.values())
        overall = f"{sum(v['correct'] for v in table.values()) / total:.3f}" if total else "n/a"
        lines.append(f"| {label} | " + " | ".join(_cell(table, k) for k in rank_kinds) + f" | {overall} |")
    lines += ["", "## Overall", "", "| arm | Phase 1 test accuracy | Phase 1 test NLL | panel accuracy | panel NLL | training minutes | selected step |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, _, _, panel, summary, phase1_metrics in rows:
        m, p, s = _load(phase1_metrics), _load(panel), _load(summary)
        if not (m or p or s):
            continue
        fmt = lambda value: "n/a" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"| {label} | {fmt(m and m['raw']['accuracy'])} | {fmt(m and m['raw']['nll'])} | {fmt(p and p['raw']['accuracy'])} | "
                     f"{fmt(p and p['raw']['nll'])} | {fmt(s and s['elapsed_seconds'] / 60)} | {fmt(s and s['best_step'])} |")
    lines += ["", "One seed each. Differences without paired intervals are screening signals, not confirmed results. "
              "The chance level on rank:pair is 0.5 and on rank:count 1/K. These numbers describe our graphs on our data; "
              "they say nothing about Jev's implementation.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(sys.argv[1] if len(sys.argv) > 1 else "runs/phase3/rank"))
