"""Production mix (Phase 4): the breadth pool plus the benchmark tasks' train splits, capped per source.

The pool is `janus.synth.ablation.load_pool` (Phase 1, public, families, cardinality and the T3 consensus rows: the 65k
rows the 25k default mix was drawn from) plus the jevlike Wikispeedia next-click train split and the jevlike synthetic
badges train split, both converted to our request format exactly as `demos.bench.tasks.load_jevlike_jsonl` renders
them for the benchmark (state = context, keys "0".."n-1", one Choice with the same instructions). Rows are deduplicated
by state, cleared of every evaluation state the benchmark or the Phase 4 scripts read, filtered to `max_tokens`
tree-packed tokens, shuffled once, and then drawn per source by water-filling: every source gets an equal share of
`size` unless it has fewer rows, and no source exceeds `cap` of the mix. Dev and calibration are drawn per source
(Wikispeedia and badges from their target-disjoint validation splits, never from test). `python -m janus.synth.production
--output data/production-v1`.
"""

import argparse
from collections import Counter
import glob
import json
from pathlib import Path
import random

from ..data import assert_disjoint, file_hash, load_requests, state_hash, write_json, write_jsonl
from ..schema import Request
from .ablation import EVALUATION_FILES as ABLATION_EVALUATION_FILES, KEEP, SPLIT_SOURCES, T3_SOURCES, _state_text, family_of, load_pool
from .families import QWEN35
from .mix import _rows, _too_long

DATASET = "JEV_PRODUCTION_V1"
WIKISPEEDIA = "data/jevlike/wikispeedia/jsonl"
BADGES = "data/jevlike/synthetic"
# Every request file the benchmark (demos/bench/tasks.py) or a Phase 4 script evaluates on, plus the selection-only
# withheld dev file the trainer asserts disjointness against. Globs are resolved when the check runs.
EVALUATION_FILES = ABLATION_EVALUATION_FILES + (f"{WIKISPEEDIA}/test.jsonl", f"{BADGES}/test.jsonl",
                                                "data/study-v1/test_banking77.jsonl", "data/public-v1/test*.jsonl",
                                                "data/banking77-v1/test*.jsonl", "data/phase1-v1/dev_post_unseen.jsonl",
                                                "demos/jevbench/datasets/public/*.jsonl", "data/hardtier-v1/test.jsonl", "data/hardtier-judge-v2/test.jsonl",
                                                "data/hardtier-v2/test.jsonl")
INSTRUCTIONS = "Choose the option that continues this context."  # demos.bench.tasks.load_jevlike_jsonl
DEFAULT_SIZE, DEFAULT_CAP = 40000, .2
# Extra sources already in our request format (v2 mix): a directory with train.jsonl, dev.jsonl and, optionally,
# calibration.jsonl; the rows enter as one source named after the key. `--extra name=dir` on the command line.
EXTRA_SOURCES = {"hardtier": "data/hardtier-v1", "longcontext": "data/longcontext-v1"}


def jevlike_request(row, group_id, family, tier):
    """A jevlike {context, options, label} row as the request the benchmark builds from it."""
    options = list(row["options"])
    label = int(row["label"])
    if not 0 <= label < len(options):
        raise ValueError(f"{group_id}: label {label} outside {len(options)} options")
    keys = [str(i) for i in range(len(options))]
    return {"state": row["context"], "group_id": group_id, "family": family, "source": family, "pool": family, "tier": tier,
            "questions": {"answer": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": dict(zip(keys, options)),
                                     "target": [float(i == label) for i in range(len(options))]}}}


def load_jevlike(path, family, tier):
    path = Path(path)
    return [jevlike_request(json.loads(line), f"{family}:{path.stem}:{i}", family, tier)
            for i, line in enumerate(path.open(encoding="utf-8")) if line.strip()]


def load_any(path):
    """Requests from our JSONL, from jevlike JSONL (rows with `context`) or from JevBench JSONL (rows with `labels`),
    so all three can enter the leakage check."""
    first = json.loads(next((line for line in Path(path).open(encoding="utf-8") if line.strip()), "{}"))
    if "context" in first:
        return [Request.from_dict(r) for r in load_jevlike(path, Path(path).parent.name, "T1")]
    if "labels" in first:
        from .hardtier import jevbench_request
        with Path(path).open(encoding="utf-8") as handle:
            return [jevbench_request(json.loads(line)) for line in handle if line.strip()]
    return load_requests(path)


def evaluation_paths(patterns=EVALUATION_FILES):
    return sorted({p for pattern in patterns for p in glob.glob(str(pattern)) if Path(p).is_file()})


def source_of(row):
    return "t3" if row["pool"].startswith("t3:") else row["pool"]


def allocate(counts, total, cap):
    """Rows per source: water-filling. Sources with fewer rows than the current equal share take all of theirs (up to
    `cap`), the rest share what remains equally; no source exceeds `cap`."""
    allocation, pending, remaining = {}, dict(counts), total
    while pending:
        share = remaining // len(pending)
        small = {s: min(n, cap) for s, n in pending.items() if min(n, cap) <= share}
        if not small:
            allocation.update({s: share for s in pending})
            break
        allocation.update(small)
        remaining -= sum(small.values())
        pending = {s: n for s, n in pending.items() if s not in small}
    return allocation


def _sample(rows, count, rng):
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:count]


def prepare_production(output, tokenizer, sources=SPLIT_SOURCES, t3=T3_SOURCES, wikispeedia=WIKISPEEDIA, badges=BADGES,
                       evaluation_files=EVALUATION_FILES, size=DEFAULT_SIZE, cap=DEFAULT_CAP, seed=17,
                       dev_per_source=150, calibration_per_source=400, max_tokens=2048, mode="tree", extra=None, drop=()):
    """`drop`: (source name, family) pairs removed from an extra source's train split (v3: the v1 judge_hard rows,
    whose golds were 59% "correct"; the balanced judge v2 and v3 rows replace them)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train, dev, calibration = load_pool(sources, t3)
    for name, root in (extra or {}).items():
        for split, rows_ in (("train", train), ("dev", dev), ("calibration", calibration)):
            if Path(root, f"{split}.jsonl").exists():
                rows_ += [{**{k: r[k] for k in KEEP if k in r}, "pool": name, "source": name} for r in _rows(Path(root) / f"{split}.jsonl", name)
                          if not (split == "train" and (name, family_of(r)) in set(drop))]
    pool_states = {state_hash(_state_text(r)) for r in train}
    public_train = Path(sources["public"]) / "train.jsonl"
    public_outside_pool = [r for r in _rows(public_train, "public") if state_hash(_state_text(r)) not in pool_states] if public_train.exists() else []
    train += [{**r, "pool": "public"} for r in public_outside_pool]
    validation = {}
    for root, family, tier in ((wikispeedia, "wikispeedia", "T1"), (badges, "badges", "T0")):
        if Path(root, "train.jsonl").exists():
            train += load_jevlike(Path(root) / "train.jsonl", family, tier)
            validation[family] = load_jevlike(Path(root) / "validation.jsonl", family, tier)
    # Same rules as ablation.prepare_ablation: evaluation states are dropped and recorded, the first row with a given
    # state wins, then the packed-length filter.
    paths = evaluation_paths(evaluation_files)
    evaluation_keys = {path: {key for r in load_any(path) for key in (("state", state_hash(r.state)), ("id", r.group_id))} for path in paths}
    evaluation_states = {key[1]: path for path, keys in evaluation_keys.items() for key in keys if key[0] == "state"}
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
    rng = random.Random(seed)
    rng.shuffle(pool)
    counts = Counter(source_of(r) for r in pool)
    allocation = allocate(counts, size, int(size * cap))
    taken = Counter()
    rows = []
    for row in pool:
        source = source_of(row)
        if taken[source] < allocation[source]:
            taken[source] += 1
            rows.append(row)
    # Dev and calibration per source: the pool's own dev/calibration files (T3 has none), the jevlike validation splits.
    dev_rows, calibration_rows = [], []
    by_source = lambda rows_: {s: [r for r in rows_ if source_of(r) == s] for s in sorted({source_of(r) for r in rows_})}
    for source, source_rows in by_source(dev).items():
        dev_rows += _sample(source_rows, dev_per_source, rng)
    for source, source_rows in by_source(calibration).items():
        calibration_rows += _sample(source_rows, calibration_per_source, rng)
    for family, rows_ in validation.items():
        held = _sample(rows_, dev_per_source + calibration_per_source, rng)
        dev_rows += held[:dev_per_source]
        calibration_rows += held[dev_per_source:]
    # Wikispeedia repeats a state whenever two paths share (target, current); dev and calibration are deduplicated
    # by state against train and each other (first row wins, as in the pool).
    held_seen = {state_hash(_state_text(r)) for r in rows}
    def _unique(rows_):
        kept = []
        for row in rows_:
            key = state_hash(_state_text(row))
            if key not in held_seen:
                held_seen.add(key)
                kept.append(row)
        return kept
    dev_rows, calibration_rows = _unique(dev_rows), _unique(calibration_rows)
    dev_rows, too_long_dev = _too_long(dev_rows, max_tokens, tokenizer, mode)
    calibration_rows, too_long_calibration = _too_long(calibration_rows, max_tokens, tokenizer, mode)
    files = {"train": rows, "dev": dev_rows, "calibration": calibration_rows}
    for split, split_rows in files.items():
        write_jsonl(output / f"{split}.jsonl", split_rows)
    check = {split: load_requests(output / f"{split}.jsonl") for split in files}
    assert_disjoint(check)
    keys = {key for requests in check.values() for r in requests for key in (("state", state_hash(r.state)), ("id", r.group_id))}
    for path, path_keys in evaluation_keys.items():
        if keys & path_keys:
            raise ValueError(f"Data overlap between the production mix and {path}")
    per_source = lambda rows_: dict(sorted(Counter(source_of(r) for r in rows_).items()))
    manifest = {"dataset": DATASET, "seed": seed, "size": size, "cap": cap, "cap_rows": int(size * cap),
                "counts": {split: len(v) for split, v in files.items()},
                "sources": {split: per_source(v) for split, v in files.items()},
                "share_of_train": {s: round(n / len(rows), 4) for s, n in per_source(rows).items()},
                "pools_in_train": dict(sorted(Counter(r["pool"] for r in rows).items())),
                "families_in_train": dict(sorted(Counter(family_of(r) for r in rows).items())),
                "tiers_in_train": dict(sorted(Counter(r.get("tier", "unknown") for r in rows).items())),
                "pool": {"rows_after_filters": len(pool), "sources": dict(sorted(counts.items())), "allocation": allocation,
                         "public_train_rows_outside_breadth_pool": len(public_outside_pool)},
                "dropped_for_evaluation_overlap": dropped, "dropped_within_pool_duplicates": dict(duplicates),
                "dropped_over_max_tokens": {"max_tokens": max_tokens, "mode": mode, "tokenizer": getattr(tokenizer, "name_or_path", "bytes"),
                                            "train": too_long, "dev": too_long_dev, "calibration": too_long_calibration},
                "leakage_check": {"checked_against": paths, "keys": "normalised state hash and group_id", "overlap": 0},
                "inputs": {"pool": {**{k: str(v) for k, v in sources.items()}, "t3": [str(p) for p in t3]},
                           "wikispeedia": str(wikispeedia), "badges": str(badges), "extra": {k: str(v) for k, v in (extra or {}).items()},
                           "drop": [list(d) for d in drop]},
                "t3_note": "T3 rows are model-consensus labels (spec WP3 3.1), not ground truth; T3_second and T4 rows are excluded as in mix.py.",
                "files": {p.name: file_hash(p) for p in sorted(output.glob("*.jsonl"))}}
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/production-v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--cap", type=float, default=DEFAULT_CAP)
    parser.add_argument("--max-tokens", type=int, default=2048, help="packed-length filter (v1: 2048; v2: 8192)")
    parser.add_argument("--extra", action="append", default=[], metavar="NAME=DIR",
                        help=f"extra source directory in our format; repeatable; e.g. {' '.join(f'{k}={v}' for k, v in EXTRA_SOURCES.items())}")
    parser.add_argument("--drop", action="append", default=[], metavar="NAME:FAMILY", help="drop an extra source's train rows of one family")
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN35[0], revision=QWEN35[1])
    extra = dict(item.split("=", 1) for item in args.extra)
    manifest = prepare_production(args.output, tokenizer, seed=args.seed, size=args.size, cap=args.cap, max_tokens=args.max_tokens, extra=extra,
                                  drop=[tuple(d.split(':', 1)) for d in args.drop])
    print(json.dumps({k: manifest[k] for k in ("counts", "sources", "share_of_train", "pool", "dropped_over_max_tokens", "leakage_check")}, indent=2))


if __name__ == "__main__":
    main()
