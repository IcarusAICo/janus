"""Phase 4 honesty arms: per-arm mean and spread over seeds, paired seed-by-seed deltas versus base, and the
held-out gate (held-out posterior NLL at or below uniform without losing in-distribution accuracy). Also builds
the withheld-family dev split that the dev_nll_plus_heldout selection rule scores. No sentence about what Jev does."""

import json
import math
from pathlib import Path
import random
import sys

from .data import assert_disjoint, file_hash, load_requests, state_hash, write_json, write_jsonl

ARMS = ("base", "complement", "consistency", "complement_consistency", "base_heldout_select")
SEEDS = (17, 23, 29)
# metric -> (run subdirectory, path inside metrics.json); summary.json fields are read separately
METRICS = {"test_acc": ("test", ("raw", "accuracy")), "test_nll": ("test", ("raw", "nll")),
           "panel_nll": ("panel", ("raw", "nll")), "panel_ece": ("panel", ("raw", "ece")),
           "heldout_acc": ("test_post_unseen", ("raw", "accuracy")), "heldout_nll": ("test_post_unseen", ("raw", "nll")),
           "heldout_uniform_nll": ("test_post_unseen", ("uniform", "nll")),
           "mmlu_acc": ("mmlu_pro", ("raw", "accuracy")), "mmlu_nll": ("mmlu_pro", ("raw", "nll"))}
COLUMNS = ("test_acc", "test_nll", "panel_nll", "panel_ece", "heldout_acc", "heldout_nll", "mmlu_acc", "mmlu_nll")


def build_heldout_dev(phase1_dir, count=300, seed=17):
    """`dev_post_unseen.jsonl`: `count` posterior states from the withheld families (8 and 9), drawn with the Phase 1
    builder under the seed pattern it uses (`f"{seed}:post:{split}"`), no paraphrase slice, rejected while they collide
    with any existing split, and recorded in the manifest. Selection only: never a test set."""
    from .synth.build import _generate, load_rows
    root = Path(phase1_dir)
    target = root / "dev_post_unseen.jsonl"
    if target.exists():
        raise FileExistsError(target)
    splits = {p.stem: load_requests(p) for p in sorted(root.glob("*.jsonl"))}
    seen = {("id", r.group_id) for rs in splits.values() for r in rs}
    seen |= {("state", state_hash(r.state)) for rs in splits.values() for r in rs}
    rng = random.Random(f"{seed}:post:dev_post_unseen")
    rows = [{k: v for k, v in row.items() if k != "paraphrased"}
            for row in _generate("post", count, rng, "test_post_unseen", None, 0., seen)]
    assert_disjoint({**splits, "dev_post_unseen": load_rows(rows)})
    write_jsonl(target, rows)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["counts"]["dev_post_unseen"] = {"post": len(rows)}
    manifest["tiers"]["T0"] = manifest["tiers"].get("T0", 0) + len(rows)
    manifest["holdouts"]["post"] += ("; dev_post_unseen also uses families 8-9 (checkpoint selection only, seed "
                                    f"'{seed}:post:dev_post_unseen', no paraphrase), disjoint from every other split")
    manifest["files"][target.name] = file_hash(target)
    write_json(root / "manifest.json", manifest)
    return manifest


def _load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def run_metrics(run_dir):
    """One run's numbers (None where the file or field is absent)."""
    out = {}
    for name, (subdir, keys) in METRICS.items():
        value = _load(Path(run_dir) / subdir / "metrics.json")
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        out[name] = value
    summary = _load(Path(run_dir) / "summary.json") or {}
    out["best_step"] = summary.get("best_step")
    out["minutes"] = summary.get("elapsed_seconds") and summary["elapsed_seconds"] / 60
    return out


def collect(root, arms=ARMS, seeds=SEEDS):
    """{arm: {seed: metrics}} over the run directories that exist under root/<arm>/s<seed>."""
    return {arm: {seed: run_metrics(Path(root) / arm / f"s{seed}") for seed in seeds
                  if (Path(root) / arm / f"s{seed}").exists()} for arm in arms}


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.
    return mean, sd, len(values)


def paired_deltas(arm_runs, base_runs, metric):
    """Seed-by-seed arm minus base, over the seeds where both have the metric."""
    return [arm_runs[s][metric] - base_runs[s][metric] for s in sorted(arm_runs)
            if s in base_runs and arm_runs[s].get(metric) is not None and base_runs[s].get(metric) is not None]


def gate(arm_runs, base_runs, tolerance=.01):
    """The Phase 3 summary gate. heldout_ok: mean held-out posterior NLL <= mean uniform NLL over the arm's seeds.
    accuracy_ok: mean paired Phase 1 test-accuracy delta versus base >= -tolerance (base itself passes this by
    definition). Both None until the files exist."""
    nll, _, n = mean_sd([r["heldout_nll"] for r in arm_runs.values()])
    uniform, _, _ = mean_sd([r["heldout_uniform_nll"] for r in arm_runs.values()])
    heldout_ok = None if not n or uniform is None else nll <= uniform
    deltas = paired_deltas(arm_runs, base_runs, "test_acc") if arm_runs is not base_runs else [0.]
    accuracy_ok = None if not deltas else sum(deltas) / len(deltas) >= -tolerance
    return {"heldout_nll": nll, "uniform_nll": uniform, "seeds": n, "heldout_ok": heldout_ok,
            "accuracy_delta": sum(deltas) / len(deltas) if deltas else None, "accuracy_ok": accuracy_ok,
            "passed": None if heldout_ok is None or accuracy_ok is None else heldout_ok and accuracy_ok}


def _cell(mean, sd, n, expected=len(SEEDS)):
    return "n/a" if mean is None else f"{mean:.3f} ± {sd:.3f}" + (f" (n={n})" if n != expected else "")


def report(root, arms=ARMS, seeds=SEEDS, tolerance=.01):
    runs = collect(root, arms, seeds)
    base = runs.get(arms[0], {})
    lines = ["# Phase 4: honest ignorance on withheld posterior tables (Qwen3.5-4B hybrid)", "",
             f"Arms: {', '.join(arms)}; seeds {', '.join(map(str, seeds))}; Phase 1 data. Native (uncalibrated) numbers, "
             "mean ± sample sd over the seeds that have finished. Held-out post is `test_post_unseen` (families 8 and 9); "
             "its uniform NLL is the honest-ignorance reference. MMLU-Pro is the 1,000-item sample with its own calibration split.", "",
             "## Per arm", "", "| arm | Phase 1 test acc | test NLL | panel NLL | panel ECE | held-out acc | held-out NLL | MMLU-Pro acc | MMLU-Pro NLL | selected step | train min |",
             "| --- |" + " ---: |" * 10]
    for arm in arms:
        cells = [_cell(*mean_sd([r[m] for r in runs[arm].values()]), len(seeds)) for m in COLUMNS]
        steps = ", ".join(str(r["best_step"]) for r in runs[arm].values() if r["best_step"] is not None) or "n/a"
        minutes, _, _ = mean_sd([r["minutes"] for r in runs[arm].values()])
        lines.append(f"| {arm} | " + " | ".join(cells) + f" | {steps} | {'n/a' if minutes is None else f'{minutes:.0f}'} |")
    uniform = [r["heldout_uniform_nll"] for a in runs.values() for r in a.values() if r["heldout_uniform_nll"] is not None]
    lines += ["", f"Uniform NLL on held-out post: {uniform[0]:.3f}." if uniform else "Uniform NLL on held-out post: n/a.", "",
              f"## Paired seed-by-seed delta versus {arms[0]} (arm minus base, mean ± sd over paired seeds)", "",
              "| arm | pairs | test acc | test NLL | panel NLL | panel ECE | held-out acc | held-out NLL | MMLU-Pro acc | MMLU-Pro NLL |",
              "| --- | ---: |" + " ---: |" * 8]
    for arm in arms[1:]:
        deltas = {m: paired_deltas(runs[arm], base, m) for m in COLUMNS}
        pairs = max((len(d) for d in deltas.values()), default=0)
        lines.append(f"| {arm} | {pairs} | " + " | ".join(_cell(*mean_sd(d), len(seeds)) if d else "n/a" for d in deltas.values()) + " |")
    lines += ["", f"## Gate: held-out posterior NLL at or below uniform, Phase 1 test accuracy within {tolerance:.3f} of base", "",
              "| arm | seeds | held-out NLL | uniform | held-out ok | test acc delta vs base | accuracy ok | gate |", "| --- | ---: | ---: | ---: | --- | ---: | --- | --- |"]
    for arm in arms:
        g = gate(runs[arm], base, tolerance)
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}" if isinstance(v, float) else ("pass" if v else "fail")
        lines.append(f"| {arm} | {g['seeds']} | {fmt(g['heldout_nll'])} | {fmt(g['uniform_nll'])} | {fmt(g['heldout_ok'])} | "
                     f"{fmt(g['accuracy_delta'])} | {fmt(g['accuracy_ok'])} | {fmt(g['passed'])} |")
    lines += ["", "Three seeds per arm; a paired delta whose sd exceeds its mean is seed noise. These numbers describe our graphs "
              "on our data.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(sys.argv[1] if len(sys.argv) > 1 else "runs/phase4/honesty"))
