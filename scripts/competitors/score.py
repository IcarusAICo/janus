"""Score competitor predictions (scripts/competitors/run_competitor.py output) with janus.metrics.

    python scripts/competitors/score.py runs/competitors/laya/public.jsonl [more.jsonl ...] [--json out.json]

Per file: overall, per `family`, and per locale (group ids `massive:<locale>:...`, `xnli:<lang>:...`) accuracy, NLL
and ECE. Two views, because refusals are real: `answered` scores only the questions the competitor answered, and
`all` scores every question with a refusal counted as the uniform distribution (what a caller falls back to).
Latency is per request (median, p90, mean), over requests that ran.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from janus.metrics import metrics  # noqa: E402

FLOOR = 1e-12  # ponytail: log(0) is not finite and janus.metrics rejects it; a zero probability costs 27.6 nats


def locale(group_id):
    parts = group_id.split(":")
    return f"{parts[0]}:{parts[1]}" if parts[0] in ("massive", "xnli") and len(parts) > 2 else None


def summary(rows):
    out = {"questions": len(rows), "refused": sum(r["probabilities"] is None for r in rows)}
    out["coverage"] = 1 - out["refused"] / len(rows)
    for view, subset in (("answered", [r for r in rows if r["probabilities"] is not None]), ("all", rows)):
        if not subset:
            out[view] = None
            continue
        logits = [[math.log(max(p, FLOOR)) for p in (r["probabilities"] or [1 / r["cardinality"]] * r["cardinality"])]
                  for r in subset]
        m = metrics(logits, [r["target"] for r in subset])
        out[view] = {k: m[k] for k in ("count", "accuracy", "nll", "brier", "ece")}
    return out


def score(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows = [r for r in rows if r["target"] is not None]
    report = {"file": str(path), "overall": summary(rows)}
    for name, key in (("family", lambda r: r.get("family")), ("locale", lambda r: locale(r["group_id"]))):
        groups = {}
        for r in rows:
            if key(r) is not None:
                groups.setdefault(key(r), []).append(r)
        report[name] = {g: summary(v) for g, v in sorted(groups.items())}
    latency = sorted({r["request_index"]: r["latency_ms"] for r in rows if r.get("latency_ms") is not None}.values())
    if latency:
        report["latency_ms"] = {"requests": len(latency), "median": statistics.median(latency),
                                "p90": latency[min(len(latency) - 1, int(0.9 * len(latency)))],
                                "mean": statistics.fmean(latency)}
    truncated = [r for r in rows if "truncated" in r]
    if truncated:
        report["truncated_share"] = sum(r["truncated"] for r in truncated) / len(truncated)
    return report


def line(name, s):
    a, b = s["answered"] or {}, s["all"]
    return (f"  {name:<28} n={s['questions']:>6} cov={s['coverage']:.3f}  answered: acc={a.get('accuracy', float('nan')):.4f} "
            f"nll={a.get('nll', float('nan')):.4f} ece={a.get('ece', float('nan')):.4f}  all: acc={b['accuracy']:.4f} "
            f"nll={b['nll']:.4f} ece={b['ece']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("--json", help="write every report here")
    args = parser.parse_args()
    reports = [score(p) for p in args.predictions]
    for r in reports:
        print(r["file"])
        print(line("overall", r["overall"]))
        for section in ("family", "locale"):
            for name, s in r[section].items():
                print(line(f"{section}={name}", s))
        if "latency_ms" in r:
            lat = r["latency_ms"]
            print(f"  latency/request: median={lat['median']:.1f}ms p90={lat['p90']:.1f}ms mean={lat['mean']:.1f}ms "
                  f"over {lat['requests']} requests")
        if "truncated_share" in r:
            print(f"  truncated input: {r['truncated_share']:.3f} of answered questions")
    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
