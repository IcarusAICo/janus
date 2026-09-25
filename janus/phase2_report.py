"""Tabulate the WP2b/WP2d objective and regulariser screen (Plan 2D).

Native metrics come first; the three post-hoc calibrators are reported beside them, never instead of them.
No sentence about what Jev does.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random

import numpy as np
import torch

from .data import load_requests, write_json

ARMS = ("S-CE", "S-Brier", "S-spherical", "S-CE+smooth", "S-CE+consistency", "S-CE+complement",
        "none-0.05", "none-0.1", "none-0.2", "none-0.4")
NONE_ARMS = tuple(a for a in ARMS if a.startswith("none-"))
SETS = (("test", "Phase 1 test", "data/phase1-v1/test.jsonl"),
        ("test_post_unseen", "Phase 1 held-out posterior families", "data/phase1-v1/test_post_unseen.jsonl"),
        ("panel", "study panel", "data/study-v1/benchmark.jsonl"),
        ("unseen_b77", "BANKING77 unseen intents (none arms only)", "data/study-v1/test_banking77_unseen.jsonl"))
VARIANTS = (("raw", "native"), ("calibrated", "temperature"),
            ("calibrated_by_cardinality", "per-K temperature"), ("histogram_binned", "histogram"))
NONE_DESCRIPTION = "None of these intents describes the message."
GAP_ARMS = ("S-CE", "S-CE+complement")


def _load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def _fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _variant_cell(section, key, variant):
    value = section.get(variant) if section else None
    return _fmt(value[key]) if value else "n/a"


def _predictions(root, arm, name):
    path = root / arm / name / "predictions.jsonl"
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@torch.inference_mode()
def complement_gap(checkpoint_path, data_path, device="cpu", seed=17, per_domain=50):
    """Noul complement gap p(yes | affirm) + p(yes | negate) - 1 on the D3 panel propositions.

    Selection matches janus.blocked_probes.plan_jobs: per domain, the Nouls with a negation template,
    sampled with random.Random(f"{seed}:D3:{domain}"); the model scores each question alone."""
    from .blocked_probes import DOMAINS, domain_of, negate_noul
    from .packing import pack_request
    from .training import load_checkpoint
    model, _ = load_checkpoint(checkpoint_path, device)
    requests = load_requests(data_path)
    rows = []
    for domain in DOMAINS:
        pool = [r for r in requests if domain_of(r) == domain and len(r.questions) > 1
                and r.questions[1].kind == "noul" and negate_noul(r.questions[1]) is not None]
        chosen = random.Random(f"{seed}:D3:{domain}").sample(pool, min(per_domain, len(pool)))
        for r in chosen:
            noul = r.questions[1]
            yes = []
            for q in (noul, negate_noul(noul)):
                packed = pack_request(replace(r, questions=(q,)), model.tokenizer, model.packing_mode,
                                      model.config.max_tokens, **model.packing_kwargs)
                yes.append(float(model(packed)[0].float().softmax(-1)[1]))
            rows.append({"domain": domain, "group_id": r.group_id, "question_id": noul.id,
                         "affirm_yes": yes[0], "negate_yes": yes[1], "gap": yes[0] + yes[1] - 1,
                         "target_yes": float(noul.target[1]) if noul.target else None})
    gaps = np.array([r["gap"] for r in rows])
    summary = {"count": len(rows)}
    if len(rows):
        q = np.quantile(gaps, [.05, .5, .95])
        summary.update({"gap_mean": float(gaps.mean()), "gap_sd": float(gaps.std()),
                        "abs_gap_mean": float(np.abs(gaps).mean()), "gap_q05": float(q[0]),
                        "gap_q50": float(q[1]), "gap_q95": float(q[2]),
                        "fraction_abs_gap_over_0.1": float((np.abs(gaps) > .1).mean()),
                        "by_domain": {d: {"count": int(sum(r["domain"] == d for r in rows)),
                                          "abs_gap_mean": float(np.mean([abs(r["gap"]) for r in rows if r["domain"] == d]))}
                                      for d in DOMAINS if any(r["domain"] == d for r in rows)}})
    return {"checkpoint": str(checkpoint_path), "data": str(data_path), "seed": seed,
            "per_domain": per_domain, "summary": summary, "rows": rows}


def _cached_gap(root, arm, data_path, device):
    path = root / arm / "complement_gap.json"
    cached = _load(path)
    if cached:
        return cached
    checkpoint_path = root / arm / "best.pt"
    if not checkpoint_path.exists() or not Path(data_path).exists():
        return None
    result = complement_gap(checkpoint_path, data_path, device)
    write_json(path, result)
    return result


def _none_option_table(root, arms, name, data_path):
    """Intent Choice questions: predicted none mass against the empirical none-target rate."""
    if not Path(data_path).exists():
        return []
    none_index = {}
    for request in load_requests(data_path):
        for q in request.questions:
            if q.kind == "choice":
                for i, o in enumerate(q.options):
                    if o.description == NONE_DESCRIPTION:
                        none_index[(request.group_id, q.id)] = i
    lines = []
    for arm in arms:
        rows = _predictions(root, arm, name)
        if rows is None:
            continue
        matched = [(r, none_index[(r["group_id"], r["question_id"])]) for r in rows
                   if (r["group_id"], r["question_id"]) in none_index]
        if not matched:
            continue
        # Native probabilities: the stored ones are temperature-scaled, so recompute from logits.
        natives = [(torch.tensor(r["logits"]).softmax(-1).tolist(), r["target"], i) for r, i in matched]
        is_none = [t[i] == 1. for _, t, i in natives]
        mean_p_none = float(np.mean([p[i] for p, _, i in natives]))
        rate = float(np.mean(is_none))
        none_nll = [r["nll"] for (r, _), flag in zip(matched, is_none) if flag]
        other_nll = [r["nll"] for (r, _), flag in zip(matched, is_none) if not flag]
        chosen_none = float(np.mean([max(range(len(p)), key=p.__getitem__) == i for p, _, i in natives]))
        lines.append(f"| {arm} | {len(matched)} | {rate:.4f} | {mean_p_none:.4f} | {mean_p_none - rate:+.4f} | "
                     f"{chosen_none:.4f} | {_fmt(np.mean(none_nll) if none_nll else None)} | "
                     f"{_fmt(np.mean(other_nll) if other_nll else None)} |")
    return lines


def report(root, device="cpu"):
    root = Path(root)
    lines = ["# Phase 2D objective and regulariser screen", "",
             "Arms (WP2 table): S-CE (the Phase 1 winner: CE, tree mode), S-Brier, S-spherical, S-CE+smooth "
             "(label smoothing 0.05), S-CE+consistency (symmetric KL to a permuted-option view, weight 0.1), "
             "S-CE+complement (each templated Noul also trained on its negation), and none-0.05/0.1/0.2/0.4 "
             "(none-option omission rate in the study menus; study data, train_limit 12,000). One seed, 3,600 s budget, "
             "dev-NLL selection. Native (uncalibrated) metrics are primary; temperature, per-cardinality temperature, "
             "and top-label histogram binning are fitted on the matching calibration split and reported beside native.", ""]
    for name, label, data_path in SETS:
        arms = [a for a in ARMS if (root / a / name / "metrics.json").exists()]
        if not arms:
            continue
        lines += [f"## {label} (`{name}`)", "", "### Native", "",
                  "| arm | questions | accuracy | NLL | Brier | ECE | choice permutation TV | choice flip rate |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for arm in arms:
            m = _load(root / arm / name / "metrics.json")
            r, order = m["raw"], m["choice_order"]
            lines.append(f"| {arm} | {r['count']} | {r['accuracy']:.4f} | {r['nll']:.4f} | {r['brier']:.4f} | "
                         f"{r['ece']:.4f} | {_fmt(order['mean_total_variation'])} | {_fmt(order['flip_rate'])} |")
        lines += ["", "### Post-hoc calibrators beside native", "",
                  "| arm | " + " | ".join(f"{v} NLL" for _, v in VARIANTS) + " | "
                  + " | ".join(f"{v} ECE" for _, v in VARIANTS) + " | "
                  + " | ".join(f"{v} Brier" for _, v in VARIANTS) + " |",
                  "| --- |" + " ---: |" * (3 * len(VARIANTS))]
        for arm in arms:
            m = _load(root / arm / name / "metrics.json")
            cells = [_variant_cell(m, key, variant) for key in ("nll", "ece", "brier") for variant, _ in VARIANTS]
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        cardinalities = sorted({k for arm in arms for k in _load(root / arm / name / "metrics.json")["by_cardinality"]}, key=int)
        lines += ["", "### ECE by cardinality (native / temperature / per-K temperature / histogram)", "",
                  "| arm | " + " | ".join(f"K={k}" for k in cardinalities) + " |", "| --- |" + " ---: |" * len(cardinalities)]
        for arm in arms:
            by_k = _load(root / arm / name / "metrics.json")["by_cardinality"]
            cells = []
            for k in cardinalities:
                section = by_k.get(k)
                cells.append("n/a" if not section else f"n={section['raw']['count']}: "
                             + " / ".join(_variant_cell(section, "ece", variant) for variant, _ in VARIANTS))
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        families = sorted({f for arm in arms for f in _load(root / arm / name / "metrics.json")["by_family"]})
        lines += ["", "### Native NLL by family", "",
                  "| arm | " + " | ".join(families) + " |", "| --- |" + " ---: |" * len(families)]
        for arm in arms:
            by_family = _load(root / arm / name / "metrics.json")["by_family"]
            lines.append(f"| {arm} | " + " | ".join(_fmt(by_family[f]["raw"]["nll"]) if f in by_family else "n/a"
                                                     for f in families) + " |")
        if name in {"panel", "unseen_b77"}:
            none_lines = _none_option_table(root, arms, name, data_path)
            if none_lines:
                lines += ["", "### None option on intent Choice questions (native probabilities)", "",
                          "| arm | questions | none target rate | mean p(none) | difference | none chosen rate | NLL on none-target questions | NLL on other questions |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"] + none_lines
        lines.append("")
    panel_data = next(path for n, _, path in SETS if n == "panel")
    gap_rows = []
    for arm in GAP_ARMS:
        result = _cached_gap(root, arm, panel_data, device)
        if result and result["summary"]["count"]:
            s = result["summary"]
            by_domain = ", ".join(f"{d} {v['abs_gap_mean']:.4f}" for d, v in s["by_domain"].items())
            gap_rows.append(f"| {arm} | {s['count']} | {s['gap_mean']:+.4f} | {s['gap_sd']:.4f} | {s['abs_gap_mean']:.4f} | "
                            f"{s['gap_q05']:+.4f} / {s['gap_q50']:+.4f} / {s['gap_q95']:+.4f} | "
                            f"{s['fraction_abs_gap_over_0.1']:.4f} | {by_domain} |")
    if gap_rows:
        lines += ["## Noul complement gap on the D3 panel propositions", "",
                  "Gap = p(yes | affirmed Noul) + p(yes | negated Noul) - 1, each question scored alone on the study panel "
                  "(same selection as the D3 probes: 50 propositions per domain, seed 17). Zero means the graph treats a "
                  "proposition and its negation as complements.", "",
                  "| arm | pairs | mean gap | sd | mean abs gap | q05 / q50 / q95 | fraction abs gap > 0.1 | mean abs gap by domain |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"] + gap_rows + [""]
    lines += ["## Training", "",
              "| arm | train requests | dev requests | training minutes | selected step | steps | budget limited | consistency forwards | complement questions added |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for arm in ARMS:
        s = _load(root / arm / "summary.json")
        if s:
            lines.append(f"| {arm} | {s['train_requests']} | {s['dev_requests']} | {s['elapsed_seconds'] / 60:.1f} | "
                         f"{s['best_step']} | {s['steps']} | {s['budget_limited']} | "
                         f"{s.get('consistency_forwards', 0)} | {s.get('complement_questions_added', 0)} |")
    lines += ["", "Caveats: one seed per arm; the none arms train on the study data and are comparable with each other, "
              "not with the Phase 1 arms; S-CE reuses the Phase 1 winner checkpoint (identical config and data) with a "
              "refitted calibration file. These numbers describe our graphs on our data.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="runs/phase2/objective")
    parser.add_argument("--device", default="cpu", help="Device for the complement-gap forwards (cached per arm)")
    args = parser.parse_args(argv)
    print(report(args.root, args.device))


if __name__ == "__main__":
    main()
