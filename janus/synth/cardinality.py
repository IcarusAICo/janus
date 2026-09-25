"""Cardinality sweep (Phase 4): Choice K and Score level counts beyond the Phase 1 ranges, code-defined (T0).

Choice cells reuse the relative-menu world with a fixed menu size and compact option text; Score cells add three
caller-defined scales (delivery 2 levels, quality 7, urgency 10) to the 3- and 5-level incident severity scale.
`python -m janus.synth.cardinality --output data/cardinality-v1` writes train/dev/calibration/test.jsonl and a manifest.
"""

import argparse
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

from ..data import assert_disjoint, file_hash, load_requests, state_hash, write_json, write_jsonl
from ..packing import pack_request
from ..schema import Request
from .worlds import TEST_NAMES, TRAIN_NAMES, ordinal_world, relative_menu_world, world_id

DATASET = "JEV_CARDINALITY_V1"
# ponytail: the sweep stops at K=128, the ceiling at 4,096 tokens (docs/phase4/score-and-cardinality.md). K=255 is a
# separate cell (`prepare_k255`, test_k255.jsonl / train_k255.jsonl) that needs `--override max_tokens=8192` to run.
CHOICE_K = (2, 4, 8, 16, 32, 64, 128)
SCORE_LEVELS = (2, 3, 5, 7, 10)
MAX_TOKENS = 4096
QWEN = ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd")

# Scale: (family, question id, question, level descriptions in order, (lo, hi] bins of the hidden quantity per level).
SCALES = {
    2: ("delivery", "delivery:on_time", "Was this shipment delivered on time?",
        ["Late: the shipment arrived after the promised day",
         "On time: the shipment arrived on or before the promised day"], None),
    7: ("quality", "quality:grade", "What quality grade does this production batch earn?",
        ["Rejected: more than 100 defective units per 1,000 inspected",
         "Poor: 51 to 100 defective units per 1,000 inspected",
         "Below standard: 21 to 50 defective units per 1,000 inspected",
         "Standard: 11 to 20 defective units per 1,000 inspected",
         "Good: 6 to 10 defective units per 1,000 inspected",
         "Very good: 2 to 5 defective units per 1,000 inspected",
         "Flawless: at most 1 defective unit per 1,000 inspected"],
        [(101, 300), (51, 100), (21, 50), (11, 20), (6, 10), (2, 5), (0, 1)]),
    10: ("urgency", "urgency:level", "How urgent is this support ticket?",
         ["Parked: the deadline is more than a week away",
          "Low: the deadline is between three days and a week away",
          "Routine: the deadline is between two and three days away",
          "Scheduled: the deadline is between one and two days away",
          "Soon: the deadline is between twelve and twenty-four hours away",
          "Pressing: the deadline is between six and twelve hours away",
          "High: the deadline is between three and six hours away",
          "Very high: the deadline is between one and three hours away",
          "Immediate: less than one hour remains before the deadline",
          "Overdue: the deadline has already passed"],
         [(168, 400), (72, 168), (48, 72), (24, 48), (12, 24), (6, 12), (3, 6), (1, 3), (0, 1), (-48, 0)])}


def name_pool(names, copies=13):
    """Enough distinct item names for K=128 from the Phase 1 name lists: 'Atlas1', 'Borealis1', ..., 'Atlas13', ...
    (`copies` 26 gives the 260 test names the K=255 cell needs)."""
    return [f"{n}{i}" for i in range(1, copies + 1) for n in names]


def scale_world(rng, levels, style):
    """A record whose hidden quantity falls in a uniformly chosen level; the target is that level, one-hot."""
    family, qid, question, descriptions, bins = SCALES[levels]
    ident = rng.randrange(100000, 999999)
    if levels == 2:
        promised = rng.randint(1, 28)
        delivered = max(1, promised + rng.randint(-5, 5))
        level = int(delivered <= promised)
        record = {"shipment_id": ident, "promised_day": promised, "delivered_day": delivered}
        prose = f"Shipment {ident}: promised for day {promised} of the month, delivered on day {delivered}."
    elif levels == 7:
        level = rng.randrange(levels)
        lo, hi = bins[level]
        defects = rng.randint(lo, hi)
        record = {"batch_id": ident, "units_inspected": 1000, "defective_units": defects}
        prose = f"Batch {ident}: 1,000 units inspected, {defects} defective."
    else:
        level = rng.randrange(levels)
        lo, hi = bins[level]
        hours = rng.randrange(int(lo * 10) + 1, int(hi * 10)) / 10  # strictly inside the bin: no boundary ambiguity
        record = {"ticket_id": ident, "hours_until_deadline": hours}
        prose = (f"Ticket {ident}: {hours} hours until the deadline." if hours > 0
                 else f"Ticket {ident}: the deadline passed {-hours} hours ago.")
    world = {"family": family, "style": style, "levels": levels, "level": level, **record}
    raw = {"state": record if style == "json" else prose, "group_id": world_id(family, world), "questions": {
        qid: {"type": "score", "instructions": question, "criteria": descriptions,
              "target": [float(i == level) for i in range(levels)]}}}
    return world, Request.from_dict(raw)


def score_world(rng, levels, style):
    if levels in SCALES:
        return scale_world(rng, levels, style)
    return ordinal_world(rng, levels, False, style)


def log_uniform_k(rng, low=2, high=CHOICE_K[-1]):
    return int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))


def _rows(kind, count, size, rng, names, seen, tokenizer, max_tokens, stats):
    """`count` fresh rows; `size` is K, a level count, or None (K drawn log-uniformly per row)."""
    rows = []
    while len(rows) < count:
        style = "compact" if kind == "choice" else ("json" if len(rows) % 2 == 0 else "prose")
        k = size if size is not None else log_uniform_k(rng)
        if kind == "choice":
            _, request = relative_menu_world(rng, names, style, k=k)
        else:
            _, request = score_world(rng, k, style)
        keys = {("id", request.group_id), ("state", state_hash(request.state))}
        if keys & seen:
            continue
        seen |= keys
        tokens = pack_request(request, tokenizer, "tree", 10 ** 9, score_block="full").token_count
        if tokens > max_tokens:
            raise ValueError(f"{kind} with {k} options packs to {tokens} tree tokens, over {max_tokens}")
        cell = f"{kind}_{'k' if kind == 'choice' else 'l'}{k}"
        stats.setdefault(cell, []).append(tokens)
        rows.append({**request.to_dict(), "tier": "T0", "family": request.group_id.split(":")[0], "style": style,
                     "cell": cell})
    return rows


def _existing_keys(directory):
    keys = set()
    for path in sorted(Path(directory).glob("*.jsonl")):
        for r in load_requests(path):
            keys |= {("id", r.group_id), ("state", state_hash(r.state))}
    return keys


def prepare_cardinality(output, tokenizer, seed=17, per_cell=200, train=2000, dev=200, calibration=200,
                        phase1=None, max_tokens=MAX_TOKENS, choice_k=CHOICE_K, score_levels=SCORE_LEVELS):
    """Write the sweep. Test: `per_cell` states per Choice K and per Score level count. Train/dev/calibration:
    K log-uniform in [2, max K] plus every Score scale in equal shares. Rows never collide with `phase1`'s."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seen = _existing_keys(phase1) if phase1 else set()
    stats = {name: {} for name in ("train", "dev", "calibration", "test")}
    splits = {}
    train_pool, test_pool = name_pool(TRAIN_NAMES), name_pool(TEST_NAMES)
    for split, count in (("train", train), ("dev", dev), ("calibration", calibration)):
        rng = random.Random(f"{seed}:cardinality:{split}")
        rows = _rows("choice", count, None, rng, train_pool, seen, tokenizer, max_tokens, stats[split])
        for levels in score_levels:
            rows += _rows("score", max(1, count // len(score_levels) // 2), levels, rng, train_pool, seen, tokenizer,
                          max_tokens, stats[split])
        splits[split] = rows
    rng = random.Random(f"{seed}:cardinality:test")
    rows = []
    for k in choice_k:
        rows += _rows("choice", per_cell, k, rng, test_pool, seen, tokenizer, max_tokens, stats["test"])
    for levels in score_levels:
        rows += _rows("score", per_cell, levels, rng, test_pool, seen, tokenizer, max_tokens, stats["test"])
    splits["test"] = rows
    for name, rows in splits.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    check = {name: load_requests(output / f"{name}.jsonl") for name in splits}
    if phase1:
        check.update({f"phase1/{p.name}": load_requests(p) for p in sorted(Path(phase1).glob("*.jsonl"))})
    assert_disjoint(check)
    tokens = {split: {cell: {"count": len(v), "max": max(v), "mean": round(statistics.mean(v), 1)}
                      for cell, v in sorted(cells.items(), key=lambda kv: (kv[0].split("_")[0], int(kv[0].split("_")[1][1:])))}
              for split, cells in stats.items()}
    manifest = {"dataset": DATASET, "seed": seed, "choice_k": list(choice_k), "score_levels": list(score_levels),
                "max_tokens": max_tokens, "token_rule": "tree packing, score_block=full (the longest variant)",
                "counts": {name: dict(sorted(Counter(r["cell"].split("_")[0] for r in rows).items())) for name, rows in splits.items()},
                "test_cells": {cell: v["count"] for cell, v in tokens["test"].items()},
                "tokens": tokens, "phase1": str(phase1) if phase1 else None,
                "files": {p.name: file_hash(p) for p in sorted(output.glob("*.jsonl"))}}
    write_json(output / "manifest.json", manifest)
    return manifest


# K=255 cell (added after the sweep; docs/phase4/robustness.md). The runner now chunks and batches, so the cap can be
# raised at load with `--override max_tokens=8192`; the terse option line (no distance) keeps 255 leaves under it.
K255 = 255
K255_MAX_TOKENS = 8192
K255_PRICES = range(5, 400)  # 255 distinct prices; the Phase 1 pool has 195
QWEN35 = ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b")


def _k255_rows(count, k, rng, names, seen, tokenizer, stats):
    """Like `_rows("choice", ...)` with the terse line and the wide price pool; `k` None draws log-uniform in [129, 255]."""
    rows = []
    while len(rows) < count:
        size = k if k is not None else log_uniform_k(rng, CHOICE_K[-1] + 1, K255)
        _, request = relative_menu_world(rng, names, "terse", k=size, price_pool=K255_PRICES)
        keys = {("id", request.group_id), ("state", state_hash(request.state))}
        if keys & seen:
            continue
        seen |= keys
        tokens = pack_request(request, tokenizer, "tree", 10 ** 9, score_block="full").token_count
        if tokens > K255_MAX_TOKENS:
            raise ValueError(f"choice with {size} options packs to {tokens} tree tokens, over {K255_MAX_TOKENS}")
        cell = f"choice_k{size}"
        stats.setdefault(cell, []).append(tokens)
        rows.append({**request.to_dict(), "tier": "T0", "family": "rel", "style": "terse", "cell": cell})
    return rows


def prepare_k255(output, tokenizer, seed=17, per_cell=200, train=500, phase1=None):
    """Add `test_k255.jsonl` (`per_cell` states at K=255) and `train_k255.jsonl` (`train` states, K log-uniform in
    [129, 255]) to an existing sweep directory; the manifest gains a `k255` entry and nothing else changes."""
    output = Path(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ("test_k255.jsonl", "train_k255.jsonl"):
        if (output / name).exists():
            raise FileExistsError(f"Refusing to overwrite {output / name}")
    seen = _existing_keys(output) | (_existing_keys(phase1) if phase1 else set())
    stats = {"train": {}, "test": {}}
    rows = {"test": _k255_rows(per_cell, K255, random.Random(f"{seed}:cardinality:k255:test"), name_pool(TEST_NAMES, 26), seen,
                               tokenizer, stats["test"]),
            "train": _k255_rows(train, None, random.Random(f"{seed}:cardinality:k255:train"), name_pool(TRAIN_NAMES, 9), seen,
                                tokenizer, stats["train"])}
    for split, split_rows in rows.items():
        write_jsonl(output / f"{split}_k255.jsonl", split_rows)
    check = {p.name: load_requests(p) for p in sorted(output.glob("*.jsonl"))}
    if phase1:
        check.update({f"phase1/{p.name}": load_requests(p) for p in sorted(Path(phase1).glob("*.jsonl"))})
    assert_disjoint(check)
    all_train = [t for cell in stats["train"].values() for t in cell]
    manifest["k255"] = {
        "seed": seed, "max_tokens": K255_MAX_TOKENS, "style": "terse", "prices": [K255_PRICES.start, K255_PRICES.stop - 1],
        "tokenizer": getattr(tokenizer, "name_or_path", "bytes"), "token_rule": manifest["token_rule"],
        "counts": {"test": len(rows["test"]), "train": len(rows["train"])}, "train_k": [CHOICE_K[-1] + 1, K255],
        "tokens": {"test": {cell: {"count": len(v), "max": max(v), "mean": round(statistics.mean(v), 1)} for cell, v in stats["test"].items()},
                   "train": {"count": len(all_train), "max": max(all_train), "mean": round(statistics.mean(all_train), 1),
                             "min": min(all_train)}},
        "files": {name: file_hash(output / name) for name in ("test_k255.jsonl", "train_k255.jsonl")}}
    write_json(manifest_path, manifest)
    return manifest["k255"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--per-cell", type=int, default=200)
    parser.add_argument("--train", type=int, default=2000, help="Choice training rows; Score adds half as many")
    parser.add_argument("--phase1", default="data/phase1-v1", help="Rows are redrawn if they collide with this data")
    parser.add_argument("--k255", action="store_true", help="Add the K=255 cell to an existing --output (Qwen3.5 tokenizer)")
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    if args.k255:
        tokenizer = AutoTokenizer.from_pretrained(QWEN35[0], revision=QWEN35[1])
        entry = prepare_k255(args.output, tokenizer, seed=args.seed, per_cell=args.per_cell,
                             phase1=args.phase1 if Path(args.phase1).exists() else None)
        print(json.dumps(entry, indent=2))
        return
    tokenizer = AutoTokenizer.from_pretrained(QWEN[0], revision=QWEN[1])
    manifest = prepare_cardinality(args.output, tokenizer, seed=args.seed, per_cell=args.per_cell, train=args.train,
                                   phase1=args.phase1 if Path(args.phase1).exists() else None)
    print(json.dumps({k: manifest[k] for k in ("counts", "test_cells", "tokens")}, indent=2))


if __name__ == "__main__":
    main()
