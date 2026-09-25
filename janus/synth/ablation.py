"""Four-way breadth ablation mixes (spec WP3 gate follow-up): size, coverage, label tier, optimisation, one axis at a time.

The pool is Phase 1 + public + T3 consensus rows + families-v1 + cardinality + the T3 top-up cells, deduplicated by
state, filtered to 2,048 tree-packed tokens and cleared of every evaluation state. It is shuffled once; every arm takes
the first N rows of that shuffle that satisfy its predicate, so arms share rows wherever the axis allows (the 10k arm is
a prefix of the default, the default a prefix of the 50k arm; a coverage arm is the default with one family's rows
replaced by the next rows in the shuffle). Dev and calibration follow the arm's predicate (a held-out family is absent
from selection too). `python -m janus.synth.ablation --output data/breadth-v2/mixes`.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from ..data import assert_disjoint, file_hash, load_requests, state_hash, write_json, write_jsonl
from .families import QWEN35
from .mix import EVALUATION_FILES as MIX_EVALUATION_FILES, _rows, _too_long

DATASET = "JEV_BREADTH_V2"
SPLIT_SOURCES = {"phase1": "data/phase1-v1", "public": "data/public-v1", "families": "data/families-v1", "cardinality": "data/cardinality-v1"}
T3_SOURCES = ("data/t3-pilot", "data/t3-volume-v1", "data/breadth-v2/t3-topup", "data/breadth-v2/t3-topup-2")
EVALUATION_FILES = MIX_EVALUATION_FILES + ("data/families-v1/test.jsonl", "data/cardinality-v1/test.jsonl",
                                           "data/public-v1/test_mmlu_pro_1000.jsonl", "data/rank-v1/test.jsonl")
DEFAULT_SIZE = 25000
HELD_OUT = ("retrieval", "evidence", "record_match", "rel")
# Arms: `data` names another arm whose files are reused; `config` names a file in configs/phase4_breadth/.
ARMS = {"default": {},
        "size_10k": {"size": 10000, "config": "size_10k"}, "size_50k": {"size": 50000, "config": "size_50k"},
        "no_retrieval": {"drop_family": "retrieval"}, "no_evidence": {"drop_family": "evidence"},
        "no_record_match": {"drop_family": "record_match"}, "no_rel": {"drop_family": "rel"},
        "tier_t01": {"tiers": ("T0", "T1")}, "tier_t012": {"tiers": ("T0", "T1", "T2")},
        "two_epochs": {"data": "default", "config": "two_epochs", "epochs": 2}}
AXES = {"size": ("size_10k", "default", "size_50k"),
        "coverage": ("default", "no_retrieval", "no_evidence", "no_record_match", "no_rel"),
        "tier": ("tier_t01", "tier_t012", "default"),
        "optimisation": ("default", "two_epochs")}
KEEP = ("state", "questions", "group_id", "tier", "family", "source", "pool", "cell", "style")
SECONDS_PER_ROW = 6149.2 / 12000  # runs/phase3/backbones/qwen35_4b_tree/summary.json: one epoch over the Phase 1 train split


def family_of(row):
    return row.get("family") or row["group_id"].split(":", 1)[0]


def _state_text(row):
    return row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], sort_keys=True, ensure_ascii=False)


def _keep(arm):
    def keep(row):
        return family_of(row) != arm.get("drop_family") and (not arm.get("tiers") or row.get("tier") in arm["tiers"])
    return keep


def load_pool(sources=SPLIT_SOURCES, t3=T3_SOURCES):
    """(train, dev, calibration) rows with `pool` provenance; T3 files contribute consensus (outcome T3) rows only."""
    train, dev, calibration = [], [], []
    for name, root in sources.items():
        for split, rows in (("train", train), ("dev", dev), ("calibration", calibration)):
            rows += [{**r, "pool": name} for r in _rows(Path(root) / f"{split}.jsonl", name)]
    for root in t3:
        for path in sorted(Path(root).glob("*.jsonl")) if Path(root).is_dir() else ():
            train += [{**r, "pool": f"t3:{path.stem}"} for r in _rows(path, f"t3:{path.stem}")
                      if r.get("tier") == "T3" and (r.get("checks") or {}).get("outcome", "T3") == "T3"]
    slim = lambda rows: [{k: r[k] for k in KEEP if k in r} for r in rows]
    return slim(train), slim(dev), slim(calibration)


def prepare_ablation(output, tokenizer, sources=SPLIT_SOURCES, t3=T3_SOURCES, seed=17, size=DEFAULT_SIZE, arms=ARMS,
                     evaluation_files=EVALUATION_FILES, max_tokens=2048, mode="tree"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train, dev, calibration = load_pool(sources, t3)
    evaluation_keys = {}
    for path in evaluation_files:
        if Path(path).exists():
            evaluation_keys[str(path)] = {key for r in load_requests(path) for key in (("state", state_hash(r.state)), ("id", r.group_id))}
    evaluation_states = {key[1]: path for path, keys in evaluation_keys.items() for key in keys if key[0] == "state"}
    # Same rules as mix.prepare_mix: training states that coincide with an evaluation state are dropped and recorded;
    # within the pool the first row with a given state wins; then the packed-length filter.
    dropped, duplicates, seen, pool = [], Counter(), set(), []
    for row in train:
        key = state_hash(_state_text(row))
        if key in evaluation_states:
            dropped.append({"group_id": row["group_id"], "pool": row["pool"], "evaluation_file": evaluation_states[key]})
        elif key in seen:
            duplicates[row["pool"]] += 1
        else:
            seen.add(key)
            pool.append(row)
    pool, too_long = _too_long(pool, max_tokens, tokenizer, mode)
    dev, too_long_dev = _too_long(dev, max_tokens, tokenizer, mode)
    calibration, too_long_calibration = _too_long(calibration, max_tokens, tokenizer, mode)
    random.Random(seed).shuffle(pool)
    summary = {}
    for name, arm in arms.items():
        epochs = arm.get("epochs", 1)
        if "data" in arm:
            rows = summary[arm["data"]]["counts"]["train"]
            summary[name] = {"data": arm["data"], "config": arm.get("config", "default"), "epochs": epochs, "counts": summary[arm["data"]]["counts"],
                             "gpu_hours_estimate": round(rows * epochs * SECONDS_PER_ROW / 3600, 2)}
            continue
        keep = _keep(arm)
        wanted = arm.get("size", size)
        rows = [r for r in pool if keep(r)][:wanted]
        if len(rows) < wanted:
            raise ValueError(f"arm {name}: the pool holds {len(rows)} eligible rows, fewer than {wanted}")
        files = {"train": rows, "dev": [r for r in dev if keep(r)], "calibration": [r for r in calibration if keep(r)]}
        for split, split_rows in files.items():
            write_jsonl(output / name / f"{split}.jsonl", split_rows)
        check = {split: load_requests(output / name / f"{split}.jsonl") for split in files}
        assert_disjoint(check)
        keys = {key for requests in check.values() for r in requests for key in (("state", state_hash(r.state)), ("id", r.group_id))}
        for path, path_keys in evaluation_keys.items():
            if keys & path_keys:
                raise ValueError(f"Data overlap between arm {name} and {path}")
        summary[name] = {"data": name, "config": arm.get("config", "default"), "epochs": epochs,
                         "held_out_family": arm.get("drop_family"), "tiers": list(arm["tiers"]) if arm.get("tiers") else None,
                         "counts": {split: len(v) for split, v in files.items()},
                         "tiers_in_train": dict(sorted(Counter(r.get("tier", "unknown") for r in rows).items())),
                         "pools_in_train": dict(sorted(Counter(r["pool"] for r in rows).items())),
                         "families_in_train": dict(sorted(Counter(family_of(r) for r in rows).items())),
                         "gpu_hours_estimate": round(len(rows) * epochs * SECONDS_PER_ROW / 3600, 2),
                         "files": {p.name: file_hash(p) for p in sorted((output / name).glob("*.jsonl"))}}
        write_json(output / name / "manifest.json", summary[name])
    manifest = {"dataset": DATASET, "seed": seed, "default_size": size, "axes": {k: list(v) for k, v in AXES.items()}, "arms": summary,
                "pool": {"rows_after_filters": len(pool), "tiers": dict(sorted(Counter(r.get("tier", "unknown") for r in pool).items())),
                         "pools": dict(sorted(Counter(r["pool"] for r in pool).items())),
                         "families": dict(sorted(Counter(family_of(r) for r in pool).items())), "dev": len(dev), "calibration": len(calibration)},
                "dropped_for_evaluation_overlap": dropped, "dropped_within_pool_duplicates": dict(duplicates),
                "dropped_over_max_tokens": {"max_tokens": max_tokens, "mode": mode, "tokenizer": getattr(tokenizer, "name_or_path", "bytes"),
                                            "train": too_long, "dev": too_long_dev, "calibration": too_long_calibration},
                "checked_against": sorted(evaluation_keys), "sources": {**{k: str(v) for k, v in sources.items()}, "t3": [str(p) for p in t3]},
                "t3_note": "T3 rows are model-consensus labels (spec WP3 3.1), not ground truth; T3_second and T4 rows are excluded as in mix.py.",
                "gpu_hours_estimate": {"seconds_per_row": round(SECONDS_PER_ROW, 3), "total": round(sum(a["gpu_hours_estimate"] for a in summary.values()), 1),
                                       "note": "training only, from the Phase 3 4B throughput on short Phase 1 rows; evaluation is extra"}}
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/breadth-v2/mixes")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN35[0], revision=QWEN35[1])
    manifest = prepare_ablation(args.output, tokenizer, seed=args.seed, size=args.size)
    print(json.dumps({"pool": manifest["pool"], "arms": {k: {kk: v[kk] for kk in ("data", "config", "epochs", "counts", "gpu_hours_estimate")}
                                                         for k, v in manifest["arms"].items()},
                      "dropped_over_max_tokens": manifest["dropped_over_max_tokens"], "gpu_hours_estimate": manifest["gpu_hours_estimate"]}, indent=2))


if __name__ == "__main__":
    main()
