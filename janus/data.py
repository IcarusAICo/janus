"""Deterministic, source-grouped data preparation and training augmentation."""

from dataclasses import replace
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
from urllib.request import urlopen

from .schema import Option, Question, Request

BANKING_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BANKING_BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets"


def state_hash(text):
    digest = hashlib.sha256(" ".join(text.lower().split()).encode())
    for image in getattr(text, "images", ()):  # janus.schema.ImageState: the hash covers the image bytes
        digest.update(image.data)
    return digest.hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_requests(path):
    requests = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    request = Request.from_dict(json.loads(line), base_dir=Path(path).parent)
                    if not request.group_id:
                        request = replace(request, group_id=state_hash(request.state))
                    requests.append(request)
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    if not requests:
        raise ValueError(f"No requests in {path}")
    return requests


def assert_disjoint(splits):
    signatures = {}
    for name, requests in splits.items():
        keys = {("state", state_hash(r.state)) for r in requests}
        keys |= {("id", r.group_id) for r in requests if r.group_id}
        for previous, previous_keys in signatures.items():
            if keys & previous_keys:
                raise ValueError(f"Data overlap between {previous} and {name}")
        signatures[name] = keys


def split_banking(train_rows, test_rows, categories, seed=17, heldout_count=15):
    if not 0 < heldout_count < len(categories) - 1:
        raise ValueError("Holdout must leave at least two training intents")
    unseen = set(random.Random(seed).sample(sorted(categories), heldout_count))
    # Text, not class or source row index, is the grouping key. Test wins duplicates.
    groups = {}
    conflicts = set()
    for source, rows in (("train", train_rows), ("test", test_rows)):
        for row in rows:
            text, category = row["text"], row["category"]
            if category not in categories:
                raise ValueError(f"Unknown source category: {category}")
            key = state_hash(text)
            if key in groups and groups[key]["category"] != category:
                conflicts.add(key)
            groups[key] = {"text": text, "category": category, "group_id": key, "source": source}
    splits = {name: [] for name in ("train", "dev", "calibration", "test_seen", "test_unseen")}
    for key, row in sorted(groups.items()):
        if key in conflicts:
            continue
        if row["source"] == "test":
            split = "test_unseen" if row["category"] in unseen else "test_seen"
        elif row["category"] in unseen:
            continue
        else:
            bucket = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), 16) % 10
            split = "dev" if bucket == 8 else "calibration" if bucket == 9 else "train"
        splits[split].append(row)
    return splits, {"seed": seed, "seen_intents": sorted(set(categories) - unseen),
                    "unseen_intents": sorted(unseen), "conflicting_states_dropped": len(conflicts),
                    "counts": {k: len(v) for k, v in splits.items()}}


def banking_request(row, pool, seed=17, cardinalities=(2, 4, 8), omit=.2):
    if not 0 <= omit <= 1:
        raise ValueError("omit must be a probability")
    rng = random.Random(f"{seed}:{row['group_id']}")
    label = row["category"]
    if label not in pool:
        raise ValueError("Correct intent must be in the split's allowed label pool")
    k = rng.choice(cardinalities)
    if not 2 <= k <= len(pool):
        raise ValueError("Cardinality exceeds the available intent pool")
    omitted = rng.random() < omit
    candidates = [x for x in pool if x != label]
    rng.shuffle(candidates)
    words = set(re.findall(r"[a-z]+", label.lower()))
    # Half the distractors come from the most lexically related labels.
    hard = sorted(candidates, key=lambda c: len(words & set(c.lower().split("_"))), reverse=True)
    needed = k - 1 if omitted else k - 2
    chosen = hard[:needed // 2]
    chosen += [x for x in candidates if x not in chosen][:needed - len(chosen)]
    if not omitted:
        chosen.append(label)
    chosen.append(None)
    rng.shuffle(chosen)
    # Keys depend only on shuffled slots, never class IDs.
    options = tuple(Option(f"o{i}", c.replace("_", " ") if c else
                           "None of these intents describes the message.") for i, c in enumerate(chosen))
    target = tuple(float(c is None if omitted else c == label) for c in chosen)
    choice = Question("intent", "choice", "Which intent best describes the customer's message?",
                      options, target)
    positive = rng.random() < .5
    proposition = label if positive else rng.choice(candidates)
    noul = Question.from_dict("matches", {"type": "noul",
        "instructions": f"Does the customer's message express this intent: {proposition.replace('_', ' ')}?",
        "target": [float(not positive), float(positive)]})
    return Request(row["text"], (choice, noul), row["group_id"])


def shuffled_options(request, seed):
    rng = random.Random(seed)
    questions = []
    for q in request.questions:
        if q.kind != "choice":
            questions.append(q)
            continue
        indices = list(range(len(q.options)))
        rng.shuffle(indices)
        questions.append(replace(q, options=tuple(q.options[i] for i in indices),
                                 target=tuple(q.target[i] for i in indices) if q.target else None))
    return replace(request, questions=tuple(questions))


def training_epoch(requests, seed, epoch, sources=None, pool=None):
    epoch_seed = seed + epoch * 1_000_003
    if sources is None:
        return [shuffled_options(r, f"{epoch_seed}:{r.group_id}") for r in requests]
    by_id = {row["group_id"]: row for row in sources}
    if set(by_id) != {r.group_id for r in requests}:
        raise ValueError("Training source states do not match the prepared split")
    if any(row["category"] not in pool for row in sources):
        raise ValueError("Training source contains a forbidden intent")
    return [banking_request(by_id[r.group_id], pool, epoch_seed) for r in requests]


def training_sources(train_path):
    parent = Path(train_path).parent
    manifest_path = parent / "manifest.json"
    if not manifest_path.exists():
        return None, None, None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dataset") != "BANKING77":
        return None, None, manifest
    source_path = parent / "train_sources.jsonl"
    for path in (Path(train_path), source_path):
        if not path.exists() or manifest.get("files", {}).get(path.name) != file_hash(path):
            raise ValueError(f"Missing or changed prepared BANKING77 data: {path}; prepare a fresh directory")
    sources = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
    return sources, manifest["seen_intents"], manifest


def prepare_banking(output, seed=17, revision=BANKING_REVISION):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    raw = {}
    hashes = {}
    for name in ("train.csv", "test.csv", "categories.json"):
        url = f"{BANKING_BASE}/{revision}/banking_data/{name}"
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        raw[name] = payload.decode("utf-8")
        hashes[name] = hashlib.sha256(payload).hexdigest()
    categories = json.loads(raw["categories.json"])
    train_rows = list(csv.DictReader(io.StringIO(raw["train.csv"])))
    test_rows = list(csv.DictReader(io.StringIO(raw["test.csv"])))
    splits, manifest = split_banking(train_rows, test_rows, categories, seed)
    manifest.update({"dataset": "BANKING77", "source_revision": revision, "source_sha256": hashes,
                     "source": BANKING_BASE, "license": "CC-BY-4.0",
                     "citation": "Casanueva et al. (2020), Efficient Intent Detection with Dual Sentence Encoders"})
    for name, rows in splits.items():
        pool = categories if name == "test_unseen" else manifest["seen_intents"]
        requests = [banking_request(row, pool, seed) for row in rows]
        write_jsonl(output / f"{name}.jsonl", [r.to_dict() for r in requests])
        if name.startswith("test_"):
            write_jsonl(output / f"{name}_k16.jsonl",
                        [banking_request(row, pool, seed, (16,)).to_dict() for row in rows])
    write_jsonl(output / "train_sources.jsonl", splits["train"])
    manifest["files"] = {p.name: file_hash(p) for p in sorted(output.glob("*.jsonl"))}
    write_json(output / "manifest.json", manifest)
    return manifest


def synthetic_requests(count, seed=17):
    rng = random.Random(seed)
    colors = ["red", "blue", "green", "yellow"]
    requests = []
    for i in range(count):
        color = rng.choice(colors)
        levels = ["low", "medium", "high"]
        level = rng.randrange(3)
        order = rng.sample(colors, len(colors))
        raw = {"state": f"Record {seed}-{i}. Color: {color}. Intensity: {levels[level]}.",
               "questions": {
                   "color": {"type": "choice", "instructions": "What color is recorded?",
                             "criteria": {c: c for c in order}, "target": [float(c == color) for c in order]},
                   "red": {"type": "noul", "instructions": "Is the color red?",
                           "target": [float(color != "red"), float(color == "red")]},
                   "level": {"type": "score", "instructions": "What intensity is recorded?",
                             "criteria": levels, "target": [float(j == level) for j in range(3)]}}}
        raw["group_id"] = state_hash(raw["state"])
        requests.append(Request.from_dict(raw))
    return requests


def prepare_synthetic(output, count=32, seed=17):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, offset, n in (("train", 0, count), ("dev", 1, max(8, count // 4)),
                            ("calibration", 2, max(8, count // 4)), ("test", 3, max(8, count // 4))):
        write_jsonl(output / f"{name}.jsonl", [r.to_dict() for r in synthetic_requests(n, seed + offset)])
    write_json(output / "manifest.json", {"dataset": "synthetic plumbing only", "seed": seed,
               "files": {p.name: file_hash(p) for p in sorted(output.glob("*.jsonl"))}})
