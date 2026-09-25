"""Phase 1 dataset: three T0 world families plus a T1 study sample, with tiers and holdouts."""

from dataclasses import replace
import json
from pathlib import Path
import random

from ..data import assert_disjoint, file_hash, load_requests, state_hash, write_json, write_jsonl
from ..training import balanced_subset
from .openai_client import StructuredCompleter
from .render import Paraphraser
from .worlds import TEST_NAMES, TRAIN_NAMES, ordinal_world, posterior_family, posterior_world, relative_menu_world

DATASET = "JEV_PHASE1_V1"


def _generate(family, count, rng, split, paraphraser, fraction, seen=None):
    """Draw `count` fresh worlds. `seen` (group ids and normalised state hashes) is shared across every split
    so that nuisance-id collisions, which the birthday effect makes likely at 12k states, are redrawn."""
    rows = []
    seen = set() if seen is None else seen
    while len(rows) < count:
        style = "json" if len(rows) % 2 == 0 else "prose"
        if family == "rel":
            world, request = relative_menu_world(rng, TEST_NAMES if split == "test" else TRAIN_NAMES, style)
        elif family == "ord":
            levels = rng.choice((3, 4, 5)) if split == "test" else rng.choice((3, 4))
            reversed_order = split == "test" and rng.random() < .5
            world, request = ordinal_world(rng, levels, reversed_order, style)
        else:
            index = rng.choice((8, 9)) if split == "test_post_unseen" else rng.randrange(8)
            world, request = posterior_world(rng, posterior_family(index), index, style)
        keys = {("id", request.group_id), ("state", state_hash(request.state))}
        if keys & seen:
            continue
        seen |= keys
        tier = "T0"
        if paraphraser is not None and style == "prose" and family in ("ord", "post") and rng.random() < fraction:
            text = paraphraser.paraphrase(world, request.state)
            if text is None:
                continue
            key = ("state", state_hash(text))
            if key in seen:
                continue
            seen.add(key)
            request = replace(request, state=text)
            paraphrased = True
        else:
            paraphrased = False
        rows.append({**request.to_dict(), "tier": tier, "family": family, "style": style, "paraphrased": paraphrased})
    return rows


def prepare_phase1(output, seed=17, per_family=3000, dev=200, calibration=200, test=500, paraphrase_fraction=.1,
                   llm=True, study_dir="data/study-v1", completer=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    study_dir = Path(study_dir)
    paraphraser = None
    if llm:
        paraphraser = Paraphraser(completer or StructuredCompleter())
    sizes = {"train": per_family, "dev": dev, "calibration": calibration, "test": test}
    splits = {name: [] for name in list(sizes) + ["test_post_unseen"]}
    counts = {name: {} for name in splits}
    seen = set()
    for family in ("rel", "ord", "post"):
        for split, count in sizes.items():
            rng = random.Random(f"{seed}:{family}:{split}")
            rows = _generate(family, count, rng, split, paraphraser, paraphrase_fraction, seen)
            splits[split].extend(rows)
            counts[split][family] = len(rows)
    rng = random.Random(f"{seed}:post:unseen")
    rows = _generate("post", test, rng, "test_post_unseen", paraphraser, paraphrase_fraction, seen)
    splits["test_post_unseen"].extend(rows)
    counts["test_post_unseen"]["post"] = len(rows)
    for split, path, limit in (("train", study_dir / "train.jsonl", per_family), ("dev", study_dir / "dev.jsonl", dev)):
        sample = balanced_subset(load_requests(path), limit, seed)
        splits[split].extend({**r.to_dict(), "tier": "T1", "family": "study", "style": "text"} for r in sample)
        counts[split]["study"] = len(sample)
    assert_disjoint({name: load_rows(rows) for name, rows in splits.items()})
    for name, rows in splits.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    tiers = {}
    for rows in splits.values():
        for row in rows:
            tiers[row["tier"]] = tiers.get(row["tier"], 0) + 1
    manifest = {"dataset": DATASET, "seed": seed, "counts": counts, "tiers": tiers,
                "holdouts": {"rel": "test uses item names never in train/dev/calibration",
                             "ord": "train/dev/calibration use 3 or 4 levels in normal order; test adds 5 levels and reversed order",
                             "post": "train/dev/calibration/test use families 0-7; test_post_unseen uses families 8-9"},
                "paraphrase": {"requested": (paraphraser.accepted + paraphraser.rejected) if paraphraser else 0,
                               "accepted": paraphraser.accepted if paraphraser else 0,
                               "rejected": paraphraser.rejected if paraphraser else 0},
                "generation_model": paraphraser.completer.model if paraphraser else None,
                "cost_usd": paraphraser.completer.cost_usd() if paraphraser else 0.,
                "cost_usd_all_calls": paraphraser.completer.cost_usd(include_cached=True) if paraphraser else 0.,
                "llm_calls": dict(paraphraser.completer.calls) if paraphraser else {"new": 0, "cached": 0},
                "study_source": str(study_dir), "study_manifest": json.loads((study_dir / "manifest.json").read_text()),
                "files": {p.name: file_hash(p) for p in sorted(output.glob("*.jsonl"))}}
    write_json(output / "manifest.json", manifest)
    return manifest


def load_rows(rows):
    from ..schema import Request
    return [Request.from_dict({k: v for k, v in row.items() if k in ("state", "questions", "group_id")}) for row in rows]
