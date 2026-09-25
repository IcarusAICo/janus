"""JevBench public results files (runs/jevbench/<name>.jsonl) -> one table: accuracy per split and overall, and the
hard tier's 10-bin top-probability ECE. Report only: these items are never used to select or tune.

    python scripts/jevbench_table.py jevk5 laya janus-4b-v3i ...
"""
import json
import sys


def ece(conf, ok, bins=10):
    total = 0.
    for b in range(bins):
        idx = [i for i, c in enumerate(conf) if b / bins < c <= (b + 1) / bins or (b == 0 and c == 0)]
        if idx:
            total += abs(sum(ok[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx)) * len(idx)
    return total / len(conf)


print("| system | easy | standard | hard | all | hard ECE |\n|---|---:|---:|---:|---:|---:|")
for name in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"runs/jevbench/{name}.jsonl")]
    acc = lambda rs: sum(bool(r.get("correct")) for r in rs) / len(rs)
    split = {s: [r for r in rows if r["task_id"].split("-")[0] == s] for s in ("easy", "original", "hard")}
    hard = [r for r in split["hard"] if r.get("probs")]
    top = lambda r: max(r["probs"].values()) if isinstance(r["probs"], dict) else max(r["probs"])
    e = ece([top(r) for r in hard], [bool(r.get("correct")) for r in hard]) if hard else float("nan")
    print(f"| {name} | {acc(split['easy']):.3f} | {acc(split['original']):.3f} | {acc(split['hard']):.3f} | {acc(rows):.3f} | {e:.3f} |")
