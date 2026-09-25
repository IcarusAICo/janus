"""Rank-position worlds (tier T0): supervision for counting how many options beat a given option.

The relative-menu family fails on every kind that needs a rank position that is not an extreme (second
cheapest, closest, median; see docs/phase1/graph-screen.md, addendum). This family renders the same items
the same way and asks five kinds of question about rank positions on one attribute. Gold is computed by
code from the hidden world; the state carries only the request and a nuisance id. Phrases are shared with
the relative-menu family where the request is the same, and `request_kind` parses both."""

from collections import Counter
import json
from pathlib import Path
import random
import re

from ..data import assert_disjoint, file_hash, load_requests, state_hash, write_json, write_jsonl
from ..schema import Request
from .worlds import TEST_NAMES, TRAIN_NAMES, _item_text, world_id

DATASET = "JEV_RANK_V1"
MIX_DATASET = "JEV_PHASE1_RANK_V1"
RANK_KINDS = ("pair", "count", "kth", "closest", "median")
ATTRIBUTES = ("price", "rating", "distance_km")
# How each attribute is spoken about: (comparative, superlative, noun). Lower is "better" for price and
# distance, higher for rating; `_better` orders items so that index 0 is the extreme the superlative names.
WORDS = {"price": ("cheaper", "cheapest", "price"), "rating": ("rated higher", "highest rated", "rating"),
         "distance_km": ("closer", "closest", "distance")}
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")
CARDINALITIES = (4, 5, 6, 8)
MEDIAN_CARDINALITIES = (5, 7)
EVALUATION_FILES = ("data/phase1-v1/test.jsonl", "data/phase1-v1/test_post_unseen.jsonl", "data/rank-v1/test.jsonl",
                    "data/study-v1/benchmark.jsonl")


def _better(a, b, attribute):
    """True when `a` beats `b` on the attribute: cheaper, higher rated, or closer."""
    return a[attribute] > b[attribute] if attribute == "rating" else a[attribute] < b[attribute]


def _ranked(items, attribute):
    """Items from the extreme the superlative names (cheapest, highest rated, closest) downwards."""
    return sorted(items, key=lambda i: -i[attribute] if attribute == "rating" else i[attribute])


def _menu(items, style):
    return "; ".join(_item_text(i, style) for i in items)


def rank_world(rng, names, style):
    while True:
        kind = rng.choice(RANK_KINDS)
        attribute = rng.choice(ATTRIBUTES)
        k = rng.choice(MEDIAN_CARDINALITIES if kind == "median" else CARDINALITIES)
        chosen = rng.sample(names, k)
        prices = rng.sample(range(5, 200), k)
        ratings = [r / 10 for r in rng.sample(range(10, 51), k)]
        distances = rng.sample(range(1, 51), k)
        items = [{"name": n, "price": p, "rating": r, "distance_km": d} for n, p, r, d in zip(chosen, prices, ratings, distances)]
        comparative, superlative, noun = WORDS[attribute]
        reference = other = position = None
        if kind == "pair":
            reference, other = rng.sample([i["name"] for i in items], 2)
            a, b = (next(i for i in items if i["name"] == n) for n in (reference, other))
            gold = _better(a, b, attribute)
            ask = f"The customer asks: is the option named {reference} {comparative} than the option named {other}?"
        elif kind == "count":
            reference = rng.choice(items)["name"]
            ref = next(i for i in items if i["name"] == reference)
            gold = sum(_better(i, ref, attribute) for i in items)
            ask = f"The customer asks: how many options are {comparative} than the option named {reference}?"
        elif kind == "kth":
            position = rng.randint(2, k - 1)
            gold = _ranked(items, attribute)[position - 1]["name"]
            ask = f"The customer wants the {ORDINALS[position - 1]} {superlative} option."
        elif kind == "closest":
            reference = rng.choice(items)["name"]
            ref = next(i for i in items if i["name"] == reference)
            others = [i for i in items if i["name"] != reference]
            gaps = sorted(abs(i[attribute] - ref[attribute]) for i in others)
            if len(gaps) > 1 and abs(gaps[0] - gaps[1]) < 1e-9:
                continue
            gold = min(others, key=lambda i: abs(i[attribute] - ref[attribute]))["name"]
            ask = f"The customer wants the option whose {noun} is closest to the option named {reference}."
        else:
            gold = sorted(items, key=lambda i: i[attribute])[k // 2]["name"]
            ask = f"The customer wants the option whose {noun} is the median of all the options."
        break
    world = {"family": "rank", "kind": kind, "attribute": attribute, "style": style, "items": items,
             "reference": reference, "other": other, "position": position, "gold": gold}
    rng.shuffle(items)
    order_id = rng.randrange(100000, 999999)
    world["order_id"] = order_id
    state = {"order_id": order_id, "customer_request": ask} if style == "json" else f"Order {order_id}. {ask}"
    questions = {}
    if kind == "pair":
        questions["rank:pair"] = {"type": "noul", "instructions": f"The options are: {_menu(items, style)}. "
                                  f"Is the option named {reference} {comparative} than the option named {other}?",
                                  "target": [float(not gold), float(gold)]}
    elif kind == "count":
        questions["rank:count"] = {"type": "choice", "instructions": f"The options are: {_menu(items, style)}. "
                                   f"How many options are {comparative} than the option named {reference}?",
                                   "criteria": {str(c): str(c) for c in range(k)},
                                   "target": [float(c == gold) for c in range(k)]}
    else:
        probe = gold if rng.random() < .5 else rng.choice([i["name"] for i in items if i["name"] != gold])
        questions["rank:pick"] = {"type": "choice", "instructions": "Which option satisfies the customer's request?",
                                  "criteria": {f"o{i}": _item_text(item, style) for i, item in enumerate(items)},
                                  "target": [float(item["name"] == gold) for item in items]}
        questions["rank:is_gold"] = {"type": "noul",
                                     "instructions": f"Is the option named {probe} the correct pick for the customer's request?",
                                     "target": [float(probe != gold), float(probe == gold)]}
    raw = {"state": state, "group_id": world_id("rank", world), "questions": questions}
    return world, Request.from_dict(raw)


# Request phrases, in the order they are tried. Rank patterns are anchored on words the relative-menu family
# never uses ("asks:", ordinals other than second, "rating ... median" is rel, "price ... median" is rank).
REL_PATTERNS = (("second_cheapest", r"wants the second cheapest option"),
                ("median_rating", r"rating is the median of all the options"),
                ("closest_price_to_named", r"price is closest to the option named"),
                ("cheapest_above_named_rating", r"cheapest option that is rated higher than the option named"))
RANK_PATTERNS = (("pair", r"asks: is the option named \w+ (?:cheaper|rated higher|closer) than the option named"),
                 ("count", r"asks: how many options are (?:cheaper|rated higher|closer) than the option named"),
                 ("kth", r"wants the (?:" + "|".join(ORDINALS) + r") (?:cheapest|highest rated|closest) option"),
                 ("closest", r"(?:price|rating|distance) is closest to the option named"),
                 ("median", r"(?:price|rating|distance) is the median of all the options"))


def request_kind(state, family=None):
    """`<family>:<kind>` for a relative-menu or rank request rendered in the state (prose or JSON), else None."""
    tables = {"rel": REL_PATTERNS, "rank": RANK_PATTERNS}
    order = [family] if family in tables else ["rel", "rank"]
    for name in order:
        for kind, pattern in tables[name]:
            if re.search(pattern, state):
                return f"{name}:{kind}"
    return None


def _seen_from(directories):
    seen = set()
    for directory in directories:
        for path in sorted(Path(directory).glob("*.jsonl")) if Path(directory).is_dir() else ():
            for request in load_requests(path):
                seen.add(("id", request.group_id))
                seen.add(("state", state_hash(request.state)))
    return seen


def generate_rows(count, rng, split, seen):
    rows = []
    names = TEST_NAMES if split == "test" else TRAIN_NAMES
    while len(rows) < count:
        style = "json" if len(rows) % 2 == 0 else "prose"
        world, request = rank_world(rng, names, style)
        keys = {("id", request.group_id), ("state", state_hash(request.state))}
        if keys & seen:
            continue
        seen |= keys
        rows.append({**request.to_dict(), "tier": "T0", "family": "rank", "style": style, "kind": world["kind"],
                     "attribute": world["attribute"]})
    return rows


def prepare_rank(output, seed=17, train=6000, dev=300, calibration=300, test=1000, exclude=("data/phase1-v1",)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sizes = {"train": train, "dev": dev, "calibration": calibration, "test": test}
    # Identical phrases exist in the relative-menu family, so its states are redrawn here rather than collide.
    seen = _seen_from(exclude)
    splits = {}
    for split, count in sizes.items():
        rng = random.Random(f"{seed}:rank:{split}")
        splits[split] = generate_rows(count, rng, split, seen)
    assert_disjoint({name: [Request.from_dict(r) for r in rows] for name, rows in splits.items()})
    for name, rows in splits.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    manifest = {"dataset": DATASET, "seed": seed, "tier": "T0", "kinds": list(RANK_KINDS), "attributes": list(ATTRIBUTES),
                "counts": {name: dict(sorted(Counter(r["kind"] for r in rows).items())) for name, rows in splits.items()},
                "total": {name: len(rows) for name, rows in splits.items()},
                "tiers": {"T0": sum(len(rows) for rows in splits.values())},
                "holdouts": {"rank": "test uses item names never in train/dev/calibration (the Phase 1 relative-menu holdout)"},
                "excluded": [str(d) for d in exclude if Path(d).is_dir()],
                "gold_note": "T0: gold computed by code from the hidden world; the state carries only the request and a nuisance id; "
                             "for the pair and count kinds the menu is rendered in the question instructions.",
                "files": {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in sorted(output.glob("*.jsonl"))}}
    write_json(output / "manifest.json", manifest)
    return manifest


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare_phase1_rank(output, phase1="data/phase1-v1", rank="data/rank-v1", evaluation_files=EVALUATION_FILES):
    """Phase 1 train/dev/calibration concatenated with the rank family's, disjointness-checked against every evaluation file."""
    output, phase1, rank = Path(output), Path(phase1), Path(rank)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in ("train", "dev", "calibration"):
        files[name] = _rows(phase1 / f"{name}.jsonl") + _rows(rank / f"{name}.jsonl")
        write_jsonl(output / f"{name}.jsonl", files[name])
    check = {name: load_requests(output / f"{name}.jsonl") for name in files}
    for path in evaluation_files:
        if Path(path).exists():
            check[str(path)] = load_requests(path)
    assert_disjoint(check)
    manifest = {"dataset": MIX_DATASET, "sources": {"phase1": str(phase1), "rank": str(rank)},
                "counts": {name: len(rows) for name, rows in files.items()},
                "families": {name: dict(sorted(Counter(r.get("family", "unknown") for r in rows).items())) for name, rows in files.items()},
                "tiers": {name: dict(sorted(Counter(r.get("tier", "unknown") for r in rows).items())) for name, rows in files.items()},
                "checked_against": [str(p) for p in evaluation_files if Path(p).exists()],
                "inputs": {str(p): file_hash(p) for root in (phase1, rank) for p in sorted(root.glob("*.jsonl"))},
                "files": {p.name: file_hash(p) for p in sorted(output.glob("*.jsonl"))}}
    write_json(output / "manifest.json", manifest)
    return manifest
