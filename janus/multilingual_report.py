"""Per-locale MASSIVE breakdown and paired clean-versus-variant robustness tables (docs/phase4/robustness.md).

MASSIVE and XNLI rows carry their locale in the group id, `massive:<locale>:<hash>` / `xnli:<lang>:<hash>` (janus.synth.public), so any
predictions file keyed by group id joins to the locale without the source rows: our evaluation records
(`predictions.jsonl`: group_id, probabilities, target) and the benchmark answers under demos/artifacts/bench
(`id`, per-question `correct` and `confidence`, no distribution, so no NLL).

    python -m janus.multilingual_report massive runs/phase2/breadth/mix-C/public/predictions.jsonl [...]
    python -m janus.multilingual_report robustness runs/phase4/robustness/<checkpoint> [--data data/robustness-v1]
"""

import argparse
import json
import math
from pathlib import Path

BINS = 10


def locale_of(group_id):
    parts = str(group_id).split(":")
    return parts[1] if parts[0] in ("massive", "xnli") and len(parts) >= 3 else None


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def observations(path):
    """One dict per question: locale, kind, correct, confidence, nll (None when the file has no distribution)."""
    out = []
    for row in _rows(path):
        if "probabilities" in row:  # janus.evaluation record
            locale = locale_of(row["group_id"])
            if locale is None:
                continue
            p, y = row["probabilities"], row["target"]
            i = max(range(len(p)), key=p.__getitem__)
            nll = row.get("calibrated_nll", row.get("nll"))
            if nll is None:
                nll = -sum(t * math.log(max(q, 1e-12)) for t, q in zip(y, p) if t)
            out.append({"locale": locale, "kind": row["kind"], "correct": float(y[i]), "confidence": float(p[i]), "nll": float(nll)})
        elif "questions" in row:  # demos.bench answer
            locale = locale_of(row["id"])
            if locale is None or row.get("error"):
                continue
            for q in row["questions"]:
                out.append({"locale": locale, "kind": q["type"], "correct": float(bool(q["correct"])),
                            "confidence": float(q["confidence"]), "nll": None})
    return out


def summarise(obs):
    n = len(obs)
    if not n:
        return None
    bins = [[] for _ in range(BINS)]
    for o in obs:
        bins[min(int(o["confidence"] * BINS), BINS - 1)].append(o)
    ece = sum(len(b) / n * abs(sum(o["correct"] for o in b) / len(b) - sum(o["confidence"] for o in b) / len(b)) for b in bins if b)
    nlls = [o["nll"] for o in obs if o["nll"] is not None]
    return {"n": n, "accuracy": sum(o["correct"] for o in obs) / n, "nll": sum(nlls) / len(nlls) if len(nlls) == n else None, "ece": ece}


def by_locale(obs, kinds=("choice", "noul", "all")):
    table = {}
    for locale in sorted({o["locale"] for o in obs}) + ["all"]:
        for kind in kinds:
            rows = [o for o in obs if (locale == "all" or o["locale"] == locale) and (kind == "all" or o["kind"] == kind)]
            summary = summarise(rows)
            if summary:
                table[(locale, kind)] = summary
    return table


def _fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def massive_report(paths):
    lines = []
    for path in paths:
        table = by_locale(observations(path))
        if not table:
            lines += [f"## {path}", "", "no MASSIVE rows", ""]
            continue
        lines += [f"## {path}", "", "| locale | kind | n | accuracy | NLL | ECE |", "| --- | --- | ---: | ---: | ---: | ---: |"]
        for (locale, kind), s in table.items():
            lines.append(f"| {locale} | {kind} | {s['n']} | {_fmt(s['accuracy'])} | {_fmt(s['nll'])} | {_fmt(s['ece'])} |")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------------
# robustness: clean twin against variant, row for row

def _keyed(rows):
    out = {}
    for row in rows:
        p = row["probabilities"]
        i = max(range(len(p)), key=p.__getitem__)
        out[(row["group_id"].split("#")[0], row["question_id"])] = {
            "correct": float(row["target"][i]), "pick": row["keys"][i], "nll": row.get("calibrated_nll", row["nll"])}
    return out


def paired(clean_path, variant_path, injections=None):
    """Paired metrics; `injections` maps (pair_id, question_id) -> injected key for the followed-injection rate."""
    clean, variant = _keyed(_rows(clean_path)), _keyed(_rows(variant_path))
    keys = sorted(set(clean) & set(variant))
    if not keys:
        raise ValueError(f"No paired rows between {clean_path} and {variant_path}")
    c, v = [clean[k] for k in keys], [variant[k] for k in keys]
    n = len(keys)
    out = {"n": n, "clean_accuracy": sum(o["correct"] for o in c) / n, "variant_accuracy": sum(o["correct"] for o in v) / n,
           "clean_nll": sum(o["nll"] for o in c) / n, "variant_nll": sum(o["nll"] for o in v) / n,
           "broken": sum(a["correct"] > b["correct"] for a, b in zip(c, v)) / n,
           "fixed": sum(a["correct"] < b["correct"] for a, b in zip(c, v)) / n}
    if injections:
        hits = [k for k in keys if k in injections]
        out["followed_injection"] = sum(variant[k]["pick"] == injections[k] for k in hits) / len(hits) if hits else None
    return out


def injection_keys(data_dir):
    rows = _rows(Path(data_dir) / "instruction_injection.jsonl")
    return {(r["pair_id"], r["injection"]["question"]): r["injection"]["key"] for r in rows}


def robustness_report(run_dir, data_dir=None):
    from .synth.robustness import VARIANTS
    run_dir = Path(run_dir)
    injections = injection_keys(data_dir) if data_dir and (Path(data_dir) / "instruction_injection.jsonl").exists() else None
    lines = [f"## {run_dir}", "", "| variant | pairs | clean acc | variant acc | delta | clean NLL | variant NLL | broken | fixed | followed injection |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in VARIANTS:
        clean_path, variant_path = run_dir / f"clean_{variant}" / "predictions.jsonl", run_dir / variant / "predictions.jsonl"
        if not clean_path.exists() or not variant_path.exists():
            lines.append(f"| {variant} | missing | | | | | | | | |")
            continue
        s = paired(clean_path, variant_path, injections if variant == "instruction_injection" else None)
        lines.append(f"| {variant} | {s['n']} | {s['clean_accuracy']:.3f} | {s['variant_accuracy']:.3f} | "
                     f"{s['variant_accuracy'] - s['clean_accuracy']:+.3f} | {s['clean_nll']:.3f} | {s['variant_nll']:.3f} | "
                     f"{s['broken']:.3f} | {s['fixed']:.3f} | {_fmt(s.get('followed_injection'))} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    massive = sub.add_parser("massive")
    massive.add_argument("predictions", nargs="+")
    robustness = sub.add_parser("robustness")
    robustness.add_argument("run_dir")
    robustness.add_argument("--data", default="data/robustness-v1")
    args = parser.parse_args(argv)
    print(massive_report(args.predictions) if args.command == "massive" else robustness_report(args.run_dir, args.data))


if __name__ == "__main__":
    main()
