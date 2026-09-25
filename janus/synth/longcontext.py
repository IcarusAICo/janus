"""Long-context evaluation set (Phase 4): one decisive record at a controlled depth inside a long state, code-defined (T0).

Each state is a log of records, log lines and chat messages in which exactly one record (an order, an incident or a
message from the Phase 1 worlds) decides every question; the questions name its id. The filler is other records of the
same and neighbouring schemas with different ids, so the answer depends on finding that record and nothing else.
Cells: state-plus-longest-question budget L in {1k, ..., 32k} Qwen tokens times record depth in {0.1, 0.5, 0.9};
the state itself is L - RESERVE tokens (within 2%, at most 64 tokens) so that the longest question fits under L. Every state carries
4 to 8 isolated questions on the same record so fan-out is measured on the same states.
`python -m janus.synth.longcontext --output data/longcontext-v1` writes bench.jsonl (test cells), train.jsonl and
dev.jsonl (1k to 8k only, random depth) and a manifest with per-cell token statistics. Cells are sized with one
tokenizer (`--tokenizer`, default the pinned Qwen3 one; the manifest records which), so a set is exact only for
backbones that share it: `--tokenizer Qwen/Qwen3.5-4B-Base` for the qwen3_5 arms.
"""

import argparse
from collections import Counter
import json
import random
import statistics
from pathlib import Path

from ..data import assert_disjoint, file_hash, load_requests, state_hash, write_json, write_jsonl
from ..packing import pack_request
from ..schema import Request
from .cardinality import QWEN, _existing_keys
from .worlds import (FEATURES, ORD_LEVELS, ORD_MAPS, TEST_NAMES, TRAIN_NAMES, _item_text, ordinal_world, posterior_family,
                     posterior_world, relative_menu_world, world_id)

DATASET = "JEV_LONGCONTEXT_V1"
LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768)
TRAIN_LENGTHS = LENGTHS[:4]
DEPTHS = (.1, .5, .9)
# Tokens left under the cell budget for the longest question (a K=8 json relative-menu Choice packs to about 360
# tree tokens with score_block=full); the state targets L - RESERVE.
RESERVE = 512
TOLERANCE = .02
FAMILIES = ("rel", "ord", "post")
SERVICES = ("order-service", "incident-tracker", "mailroom", "auth", "billing", "scheduler")
REGIONS = ("eu-west-2", "us-east-1", "ap-south-1", "eu-central-1")
CHAT_NAMES = ("Dara", "Eli", "Farah", "Gus", "Hana", "Ivo", "Jun", "Kai")
FEATURE_QUESTIONS = {"uses_emoji": "Does message {id} use emoji?", "formal_greeting": "Does message {id} open with a formal greeting?",
                     "mentions_deadline": "Does message {id} mention a deadline?", "long_message": "Is message {id} long?"}


def _record(rng, family, style, names, family_index=None):
    """One world of `family` and its rendered state line; returns (world, request, record id)."""
    if family == "rel":
        world, request = relative_menu_world(rng, names, style, k=rng.choice((4, 5, 6, 8)))
        return world, request, world["order_id"]
    if family == "ord":
        world, request = ordinal_world(rng, 5, False, style)
        return world, request, world["incident_id"]
    index = rng.randrange(8) if family_index is None else family_index
    world, request = posterior_world(rng, posterior_family(index), index, style)
    return world, request, world["message_id"]


def _other_id(rng, record_id):
    while True:
        value = rng.randrange(100000, 999999)
        if value != record_id:
            return value


def filler_line(rng, family, style, names, record_id, clock):
    """A plausible line that never mentions `record_id`: a record of the same schema (half the lines), of a
    neighbouring schema, a service log line or a chat message. `clock` is a mutable [seconds] for log timestamps."""
    roll = rng.random()
    if roll < .7:
        other = family if roll < .5 else rng.choice([f for f in FAMILIES if f != family])
        while True:
            _, request, ident = _record(rng, other, style, names)
            if ident != record_id:
                return request.state
    clock[0] += rng.randint(1, 90)
    stamp = f"2026-03-14T{clock[0] // 3600 % 24:02d}:{clock[0] // 60 % 60:02d}:{clock[0] % 60:02d}Z"
    ident = _other_id(rng, record_id)
    if roll < .9:
        event = rng.choice((f"order {ident} queued (region {rng.choice(REGIONS)}, {rng.randint(1, 9)} items)",
                            f"incident {ident} acknowledged by on-call", f"message {ident} delivered in {rng.randint(3, 900)} ms",
                            f"retrying webhook for order {ident} (attempt {rng.randint(1, 5)})",
                            f"cache miss ratio {rng.randint(1, 40)}% over the last minute", "heartbeat ok",
                            f"incident {ident} escalated to tier {rng.randint(1, 3)}", f"session {ident} expired"))
        return f"{stamp} {rng.choice(('INFO', 'INFO', 'WARN', 'DEBUG'))} {rng.choice(SERVICES)}: {event}"
    line = rng.choice((f"can someone look at order {ident}? the customer asked twice", f"incident {ident} looks like a duplicate",
                       "taking lunch, back in 30", f"message {ident} bounced, wrong address", "deploy done, watching the dashboards",
                       f"who owns order {ident}?", "the report is due friday"))
    return f"[{stamp[11:16]}] {rng.choice(CHAT_NAMES)}: {line}"


def build_state(rng, family, style, names, record_line, record_id, target, depth, tokenizer):
    """Lines around `record_line` with about `target` tokens (within TOLERANCE) and the record `depth` of the way in.
    Returns (state text, token count, measured depth)."""
    encode = lambda value: tokenizer.encode(value, add_special_tokens=False)
    clock = [rng.randint(8 * 3600, 16 * 3600)]
    lines, total, inserted = [], 0, None
    record_tokens = len(encode(record_line))
    while total < target or inserted is None:
        line = filler_line(rng, family, style, names, record_id, clock)
        count = len(encode(line))
        # The record goes on whichever side of the line crossing depth * target lands closer to it.
        goal = depth * target
        if inserted is None and total + count >= goal and abs(total - goal) <= abs(total + count - goal):
            inserted = len(lines)
            lines.append(record_line)
            total += record_tokens
        lines.append(line)
        total += count
        if inserted is None and total >= goal:
            inserted = len(lines)
            lines.append(record_line)
            total += record_tokens
    tolerance = max(16, min(int(TOLERANCE * target), RESERVE // 8))
    for _ in range(40):
        actual = len(encode("\n".join(lines)))
        if actual > target + tolerance:
            drop = len(lines) - 1 if inserted != len(lines) - 1 else len(lines) - 2
            del lines[drop]
            inserted -= drop < inserted
        elif actual < target - tolerance:  # short lines (about 8 tokens, under the window) top up without moving the record
            clock[0] += rng.randint(1, 90)
            lines.append(f"{clock[0] // 3600 % 24:02d}:{clock[0] // 60 % 60:02d}:{clock[0] % 60:02d} heartbeat ok")
        else:
            break
    else:
        raise ValueError(f"could not reach {target} tokens within {tolerance}")
    before = len(encode("\n".join(lines[:inserted]))) if inserted else 0
    return "\n".join(lines), actual, before / actual


def questions_for(rng, family, world, request, style):
    """All isolated questions on the decisive record, the primary one first; every target is known from the world."""
    ident = world.get("order_id") or world.get("incident_id") or world.get("message_id")
    if family == "rel":
        items, gold = world["items"], world["gold"]
        menu = "; ".join(f"[o{i}] {_item_text(item, style)}" for i, item in enumerate(items))
        primary = ("rel:pick", {"type": "choice", "instructions": f"Which option satisfies the customer's request in order {ident}?",
                                "criteria": {f"o{i}": _item_text(item, style) for i, item in enumerate(items)},
                                "target": [float(item["name"] == gold) for item in items]})
        others = [(f"rel:is_gold:{item['name']}", {"type": "noul", "instructions":
                   f"The options for order {ident} are: {menu}. Is the option named {item['name']} the correct pick for the "
                   f"customer's request in order {ident}?", "target": [float(item["name"] != gold), float(item["name"] == gold)]})
                  for item in items]
    elif family == "ord":
        base = world["base"]
        primary, others = None, []
        for levels in (5, 4, 3):
            level = ORD_MAPS[levels][base]
            q = (f"ord:severity:{levels}", {"type": "score", "instructions": f"How severe is incident {ident}?",
                                            "criteria": list(ORD_LEVELS[levels]), "target": [float(i == level) for i in range(levels)]})
            major_level = ORD_MAPS[levels][3]
            threshold = min(b for b, l in ORD_MAPS[levels].items() if l == major_level)
            others.append((f"ord:at_least_major:{levels}", {"type": "noul", "instructions":
                           f"Is incident {ident} at least as severe as the level described as: {ORD_LEVELS[levels][major_level]}?",
                           "target": [float(base < threshold), float(base >= threshold)]}))
            if primary is None:
                primary = q
            else:
                others.append(q)
        for key, ask in (("data_loss", f"Did incident {ident} lose data?"), ("workaround", f"Was a workaround available for incident {ident}?")):
            value = world[key]
            others.append((f"ord:{key}", {"type": "noul", "instructions": ask, "target": [float(not value), float(value)]}))
    else:
        family_names, posterior = posterior_family(world["family_index"])["names"], world["posterior"]
        primary = ("post:sender", {"type": "choice", "instructions": f"Which sender most likely wrote message {ident}?",
                                   "criteria": {f"s{k}": family_names[k] for k in range(3)}, "target": posterior})
        others = [(f"post:is_named:{family_names[k]}", {"type": "noul", "instructions": f"Was message {ident} sent by {family_names[k]}?",
                   "target": [1 - posterior[k], posterior[k]]}) for k in range(3)]
        others += [(f"post:feature:{f}", {"type": "noul", "instructions": FEATURE_QUESTIONS[f].format(id=ident),
                    "target": [float(not world["features"][f]), float(world["features"][f])]}) for f in FEATURES]
    rng.shuffle(others)
    count = rng.randint(4, min(8, 1 + len(others)))
    return dict([primary] + others[:count - 1])


def make_state(rng, family, style, names, length, depth, tokenizer, split):
    """One row: the decisive record's world rendered in `style`, padded to the cell, with its question set."""
    world, request, ident = _record(rng, family, style, names)
    state, tokens, measured = build_state(rng, family, style, names, request.state, ident, length - RESERVE, depth, tokenizer)
    # Lines are atomic, so the record can miss the depth by half a line (a JSON record is about 55 tokens).
    if abs(measured - depth) > max(.05, 36 / (length - RESERVE)):
        raise ValueError(f"record landed at depth {measured:.3f}, wanted {depth}")
    questions = questions_for(rng, family, world, request, style)
    group_id = world_id(family, {"world": world, "length": length, "depth": depth, "split": split})
    row = Request.from_dict({"state": state, "group_id": group_id, "questions": questions}).to_dict()
    return {**row, "tier": "T0", "family": family, "style": style, "cell": f"{length // 1024}k:{depth}", "length": length,
            "depth": depth, "record_depth": round(measured, 4), "state_tokens": tokens, "question_count": len(questions)}


def _rows(count, cells, rng, names, seen, tokenizer, split, stats):
    """`count` fresh rows over `cells` (a list of (length, depth); cycled when the split has fixed cells, drawn
    when it is the random training distribution)."""
    rows = []
    while len(rows) < count:
        # Bench: cells are cycled and families/styles cycle within each cell, so every cell mixes all three families.
        turn = len(rows) // len(cells) if split == "bench" else len(rows)
        family = FAMILIES[turn % 3]
        style = "json" if (turn // 3) % 2 == 0 else "prose"
        length, depth = cells[len(rows) % len(cells)] if split == "bench" else (rng.choice(cells), round(rng.uniform(.05, .95), 3))
        row = make_state(rng, family, style, names, length, depth, tokenizer, split)
        request = Request.from_dict(row)
        keys = {("id", request.group_id), ("state", state_hash(request.state))}
        if keys & seen:
            continue
        seen |= keys
        # Both budgets with the real tokenizer: the whole request and the state plus the longest question (tree,
        # score_block=full is the longest variant); the packer raises when either is over.
        packed = pack_request(request, tokenizer, "tree", 2 * LENGTHS[-1], score_block="full", max_state_plus_question=length)
        cell = row["cell"] if split == "bench" else f"{length // 1024}k"
        stats.setdefault(cell, []).append({"state": packed.state_length, "packed": packed.token_count,
                                           "questions": len(request.questions), "depth": row["record_depth"]})
        rows.append(row)
    return rows


def prepare_longcontext(output, tokenizer, seed=17, per_cell=100, train=2000, dev=200, phase1=None, lengths=LENGTHS,
                        depths=DEPTHS, train_lengths=None, tokenizer_name=QWEN[0]):
    """`tokenizer` sizes the cells; `tokenizer_name` is recorded in the manifest (cells are only exact for that tokenizer)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train_lengths = tuple(train_lengths or [l for l in lengths if l <= TRAIN_LENGTHS[-1]] or lengths[:1])
    seen = _existing_keys(phase1) if phase1 else set()
    stats = {name: {} for name in ("train", "dev", "bench")}
    splits = {}
    for split, count in (("train", train), ("dev", dev)):
        rng = random.Random(f"{seed}:longcontext:{split}")
        splits[split] = _rows(count, list(train_lengths), rng, TRAIN_NAMES, seen, tokenizer, split, stats[split])
    cells = [(length, depth) for length in lengths for depth in depths]
    rng = random.Random(f"{seed}:longcontext:bench")
    splits["bench"] = _rows(per_cell * len(cells), cells, rng, TEST_NAMES, seen, tokenizer, "bench", stats["bench"])
    for name, rows in splits.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    check = {name: load_requests(output / f"{name}.jsonl") for name in splits}
    if phase1:
        check.update({f"phase1/{p.name}": load_requests(p) for p in sorted(Path(phase1).glob("*.jsonl"))})
    assert_disjoint(check)

    def summary(cell_rows):
        columns = {k: [r[k] for r in cell_rows] for k in ("state", "packed", "questions", "depth")}
        return {"count": len(cell_rows), "state_tokens": {"min": min(columns["state"]), "mean": round(statistics.mean(columns["state"]), 1),
                                                          "max": max(columns["state"])},
                "packed_tokens_max": max(columns["packed"]), "questions": {"min": min(columns["questions"]), "max": max(columns["questions"])},
                "record_depth_mean": round(statistics.mean(columns["depth"]), 3)}
    tokens = {split: {cell: summary(v) for cell, v in sorted(cells.items(), key=lambda kv: (int(kv[0].split("k")[0]), kv[0]))}
              for split, cells in stats.items()}
    manifest = {"dataset": DATASET, "seed": seed, "lengths": list(lengths), "depths": list(depths), "train_lengths": list(train_lengths),
                "reserve": RESERVE, "tolerance": f"{TOLERANCE} of the target, at most {RESERVE // 8} tokens",
                "tokenizer": tokenizer_name,
                "token_rule": f"{tokenizer_name} tokenizer; state targets length - reserve; packed = tree packing with score_block=full "
                              "and every question; state plus the longest question is asserted <= length",
                "counts": {name: len(rows) for name, rows in splits.items()},
                "families": {name: dict(sorted(Counter(r["family"] for r in rows).items())) for name, rows in splits.items()},
                "tokens": tokens, "phase1": str(phase1) if phase1 else None,
                "files": {p.name: file_hash(p) for p in sorted(output.glob("*.jsonl"))}}
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--per-cell", type=int, default=100)
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--dev", type=int, default=200)
    parser.add_argument("--lengths", default=",".join(str(l) for l in LENGTHS), help="Comma-separated cell budgets in tokens")
    parser.add_argument("--phase1", default="data/phase1-v1", help="Rows are redrawn if they collide with this data")
    parser.add_argument("--tokenizer", default=QWEN[0], help="Tokenizer that sizes the cells (the target backbone's; the Qwen3.5 tokenizer "
                        "renders Qwen3-sized 32k states to about 33k tokens); recorded in the manifest")
    parser.add_argument("--revision", default=None, help="Tokenizer revision (default: the pinned Qwen3 one for the default tokenizer)")
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    revision = args.revision or (QWEN[1] if args.tokenizer == QWEN[0] else None)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=revision)
    manifest = prepare_longcontext(args.output, tokenizer, seed=args.seed, per_cell=args.per_cell, train=args.train, dev=args.dev,
                                   phase1=args.phase1 if Path(args.phase1).exists() else None,
                                   lengths=tuple(int(l) for l in args.lengths.split(",")), tokenizer_name=args.tokenizer)
    print(json.dumps({k: manifest[k] for k in ("counts", "families", "tokens")}, indent=2))


if __name__ == "__main__":
    main()
