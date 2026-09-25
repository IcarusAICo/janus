"""JevBench v1.3 axes on the 231 public items, and a v1.4 what-if, from our local results files.

NOT an official score. v1.4 needs sealed accuracy only the maintainers measure, and the judge tier is held out, so
Intelligence is re-weighted over easy/standard/hard (composite_v13.intelligence renormalises missing tiers).
Assumptions, the same for every system: sealed accuracy SEALED (the board's systems sit at 0.28-0.37), the v1.4
calibration equal to the v1.3 one (no sealed distributions), endpoint kind "gpu" (x2 + 0.15 s load adjustment),
and cost = GPU_USD_PER_H x mean latency. Latency is whatever card each results file was run on.

    python scripts/jevbench_v14_whatif.py jevk5 laya janus-08b-distill ... [--sealed 0.31] [--gpu-usd-per-h 0.69]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demos" / "jevbench"))
from jevbench import composite_v13 as v13, composite_v14 as v14  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("names", nargs="+"); p.add_argument("--sealed", type=float, default=0.31)
p.add_argument("--gpu-usd-per-h", type=float, default=0.69)
a = p.parse_args()
gold = {}
for line in open(Path(__file__).resolve().parent.parent / "demos/jevbench/datasets/public/hard.jsonl"):
    t = json.loads(line)
    if (t.get("provenance") or {}).get("gold_probs"):
        gold[t["id"]] = (t["provenance"]["gold_probs"], t["labels"])


def ece(rows, bins=10):
    conf = [max(r["probs"].values()) for r in rows]; ok = [bool(r["correct"]) for r in rows]
    total = 0.
    for b in range(bins):
        idx = [i for i, c in enumerate(conf) if b / bins < c <= (b + 1) / bins or (b == 0 and c == 0)]
        if idx:
            total += abs(sum(ok[i] for i in idx) - sum(conf[i] for i in idx))
    return total / len(conf)


print(f"sealed accuracy assumed {a.sealed}, GPU ${a.gpu_usd_per_h}/h, judge tier held out (not in public items)\n")
print("| system | public acc | I (v1.3) | C (v1.3) | S | cost | v1.3 score | I (v1.4 what-if) | **v1.4 what-if** |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for name in a.names:
    rows = [json.loads(l) for l in open(f"runs/jevbench/{name}.jsonl")]
    tier = lambda s: [r for r in rows if r["task_id"].split("-")[0] == s]
    acc = lambda rs: sum(bool(r["correct"]) for r in rs) / len(rs)
    tiers = {"easy": acc(tier("easy")), "standard": acc(tier("original")), "hard": acc(tier("hard"))}
    hard = [r for r in tier("hard") if r.get("probs")]
    tvds = [v13.tvd(r["probs"], gold[r["task_id"]][0], gold[r["task_id"]][1]) for r in rows if r["task_id"] in gold and r.get("probs")]
    lat = sorted(r["latency_s"] for r in rows if r.get("latency_s") is not None)
    p50, p95 = statistics.median(lat), lat[int(.95 * (len(lat) - 1))]
    axes = {"intelligence": v13.intelligence(tiers), "calibration": v13.calibration(ece(hard), statistics.mean(tvds) if tvds else None),
            "speed": v13.speed(p50, p95, "gpu"), "cost": v13.cost(a.gpu_usd_per_h * statistics.mean(lat) / 3600 * 1000)}
    public = acc(rows)
    new = dict(axes, intelligence=v14.intelligence(axes["intelligence"], a.sealed, public))
    print(f"| {name} | {public:.3f} | {axes['intelligence']:.1f} | {axes['calibration']:.1f} | {axes['speed']:.1f} | {axes['cost']:.1f} | "
          f"{v13.jevbench_score(axes):.1f} | {new['intelligence']:.1f} | **{v14.harmonic(new):.1f}** |")
