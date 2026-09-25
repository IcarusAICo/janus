"""T3 volume run on the pilot cells with the highest first-check agreement, with stratified audit files.

`run_volume` is a thin wrapper over `generate.run_pilot` (same generator, checker, triage, pool, and cache); it
must use a seed other than the pilot's, otherwise the first `per_cell` specs of every cell would be the pilot's
own and would be served from the cache as duplicates. Acceptance here is model agreement, not verification."""

import json
from pathlib import Path
import random

from ..data import state_hash, write_json
from .generate import run_pilot

DATASET = "JEV_T3_VOLUME_V1"
PILOT_SEED = 17
# First-check agreement in data/t3-pilot/manifest.json: 0.990, 0.998, 0.996, 0.952, 0.950.
VOLUME_CELLS = ("routing_text", "argument_selection", "injection_state", "extract_candidates", "severity_rubric")
STRATA = ("T4", "T3_second", "T3")


def stratified_audit_sample(rows, size, rng):
    """Every T4 row first, then every T3_second row, then random T3 rows, truncated to `size`."""
    sample = []
    for outcome in STRATA:
        bucket = [r for r in rows if r["checks"]["outcome"] == outcome]
        rng.shuffle(bucket)
        sample.extend(bucket[:max(0, size - len(sample))])
    return sample


def _gold(question):
    pairs = zip(question["criteria"], question["target"]) if isinstance(question["criteria"], dict) else zip(range(len(question["criteria"])), question["target"])
    return [k for k, t in pairs if t == 1.]


def audit_markdown(cell, sample, available=None):
    """Same per-row format as the pilot audit files (`generate._audit_markdown`), with a stratification header."""
    counts = {o: sum(1 for r in sample if r["checks"]["outcome"] == o) for o in STRATA}
    lines = [f"# Audit sample for {cell} (volume, stratified)", "",
             "Mark each example as correct, wrong, or ambiguous. Gold is the intended label; checks are model answers, not truth.",
             "Stratified: every T4 row, then every T3_second row, then random T3 rows. "
             + f"Sample: {len(sample)} rows ({', '.join(f'{o} {counts[o]}' for o in STRATA)})"
             + (f"; available: {', '.join(f'{o} {available.get(o, 0)}' for o in STRATA)}." if available else "."), ""]
    for i, row in enumerate(sample, 1):
        q = next(iter(row["questions"].values()))
        lines += [f"## {i}. group {row['group_id']} (tier {row['tier']}, outcome {row['checks']['outcome']})", "",
                  "State:", "```", row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False, indent=1), "```",
                  f"Instructions: {q['instructions']}", "", "Options:", "```", json.dumps(q["criteria"], ensure_ascii=False, indent=1), "```",
                  f"Gold: {_gold(q)}", f"First check: {row['checks']['first']}", f"Second check: {row['checks']['second']}",
                  f"Rationale (generator, audit only): {row['rationale']}", "", "Verdict: [ ] correct [ ] wrong [ ] ambiguous", ""]
    return "\n".join(lines)


def _rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_volume(output, cells=VOLUME_CELLS, per_cell=2000, seed=18, luna=None, terra=None, cost_abort_usd=30., workers=8,
               audit_size=100, pilot="data/t3-pilot"):
    if seed == PILOT_SEED:
        raise ValueError(f"seed {PILOT_SEED} is the pilot seed; the volume run would regenerate the pilot's specs from the cache")
    output, pilot = Path(output), Path(pilot)
    manifest = run_pilot(output, cells=list(cells), per_cell=per_cell, seed=seed, luna=luna, terra=terra,
                         cost_abort_usd=cost_abort_usd, workers=workers)
    pilot_manifest = json.loads((pilot / "manifest.json").read_text()) if (pilot / "manifest.json").exists() else None
    audit, overlap = {}, {}
    for cell in cells:
        rows = _rows(output / f"{cell}.jsonl")
        available = {o: sum(1 for r in rows if r["checks"]["outcome"] == o) for o in STRATA}
        sample = stratified_audit_sample(rows, audit_size, random.Random(f"audit:{seed}:{cell}"))
        (output / f"audit-{cell}.md").write_text(audit_markdown(cell, sample, available))
        audit[cell] = {"size": len(sample), **{o: sum(1 for r in sample if r["checks"]["outcome"] == o) for o in STRATA}, "available": available}
        pilot_rows = _rows(pilot / f"{cell}.jsonl") if (pilot / f"{cell}.jsonl").exists() else []
        pilot_hashes = {state_hash(r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], sort_keys=True, ensure_ascii=False)) for r in pilot_rows}
        overlap[cell] = sum(state_hash(r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], sort_keys=True, ensure_ascii=False)) in pilot_hashes for r in rows)
    manifest.update({
        "dataset": DATASET, "pilot": {"path": str(pilot), "seed": PILOT_SEED,
                                      "selection": "the five pilot cells with the highest first-check agreement",
                                      "first_check_agreement": {c: pilot_manifest["cells"][c]["first_check_agreement_rate"] for c in cells}
                                      if pilot_manifest else None},
        "overlap_with_pilot_states": overlap, "audit": audit,
        "acceptance_note": "Acceptance is model agreement (generator gold recovered by an independent checker), not verification; "
                           "no cell is verified until its stratified audit file is marked."})
    write_json(output / "manifest.json", manifest)
    return manifest
