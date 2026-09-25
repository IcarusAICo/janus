"""Choice examples in jevlike JSONL shape, plus this repo's MMLU-Pro System One files.

Synthetic menus copy the generator from https://github.com/vinnylarouge/jevlike
(MIT) so Jev/GPT can score the same distribution without training.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

COLOURS = ("amber", "azure", "bronze", "coral", "crimson", "gold", "green", "indigo")
ANIMALS = ("badger", "crane", "dolphin", "falcon", "gecko", "heron", "ibis", "jaguar")

REPO = Path(__file__).resolve().parents[2]
MMLU_PRO_1000 = REPO / "data" / "public-v1" / "test_mmlu_pro_1000.jsonl"
SYNTHETIC_DIR = REPO / "data" / "jevlike" / "synthetic"


@dataclass(frozen=True)
class BenchItem:
    example_id: str
    state: object
    questions: dict

    def choice_question(self):
        for qid, q in self.questions.items():
            if q.get("type") == "choice":
                return qid, q
        return None, None

    @property
    def criteria(self):
        _, q = self.choice_question()
        return dict(q["criteria"]) if q else {}

    @property
    def label(self):
        _, q = self.choice_question()
        if not q:
            return None
        keys = list(q["criteria"])
        target = q["target"]
        return keys[max(range(len(target)), key=lambda i: target[i])]

    @property
    def instructions(self):
        _, q = self.choice_question()
        return (q or {}).get("instructions") or ""


ChoiceExample = BenchItem


def synthetic_example(seed):
    rng = random.Random(seed)
    target = f"{rng.choice(COLOURS)} {rng.choice(ANIMALS)}"
    options = {target}
    while len(options) < rng.randint(2, 8):
        options.add(f"{rng.choice(COLOURS)} {rng.choice(ANIMALS)}")
    options = list(options)
    rng.shuffle(options)
    notes = " ".join(rng.choice(("north", "south", "east", "west")) for _ in range(8))
    context = f"Choose the exact badge {target}. Notes: {notes}. Badge: {target}."
    keys = [str(i) for i in range(len(options))]
    return BenchItem(
        example_id=f"synthetic:{seed}",
        state=context,
        questions={
            "answer": {
                "type": "choice",
                "instructions": "Pick the badge named in the context.",
                "criteria": dict(zip(keys, options)),
                "target": [float(i == options.index(target)) for i in range(len(options))],
            }
        },
    )


def write_synthetic(output, *, train=2000, validation=400, test=400, seed=17):
    """Same split sizes and seeding as jevlike-data synthetic defaults."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sizes = {"train": train, "validation": validation, "test": test}
    offset = 0
    for split, size in sizes.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(size):
                item = synthetic_example(seed + offset + index * 104729)
                handle.write(
                    json.dumps(
                        {
                            "context": item.state,
                            "options": list(item.criteria.values()),
                            "label": int(item.label),
                        }
                    )
                    + "\n"
                )
        offset += size * 104729
    return output


def load_jevlike_jsonl(path, *, limit=None, prefix="jevlike"):
    path = Path(path)
    out = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            if limit is not None and len(out) >= limit:
                break
            row = json.loads(line)
            options = list(row["options"])
            label = int(row["label"])
            keys = [str(i) for i in range(len(options))]
            item = BenchItem(
                example_id=f"{prefix}:{index}",
                state=row["context"],
                questions={
                    "answer": {
                        "type": "choice",
                        "instructions": "Choose the option that continues this context.",
                        "criteria": dict(zip(keys, options)),
                        "target": [float(i == label) for i in range(len(options))],
                    }
                },
            )
            out.append(item)
    return out


def load_systemone_jsonl(path, *, limit=None, family=None):
    """Load System One JSONL. Optionally keep one `family` (public-v1 mixed file)."""
    path = Path(path)
    out = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if family is not None and row.get("family") != family:
                continue
            if limit is not None and len(out) >= limit:
                break
            out.append(
                BenchItem(
                    example_id=row.get("group_id") or next(iter(row["questions"])),
                    state=row["state"],
                    questions=row["questions"],
                )
            )
    return out


TASK_PATHS = {
    "mmlu-pro": (MMLU_PRO_1000, None),
    "massive": (REPO / "data" / "public-v1" / "test.jsonl", "massive"),
    "civil": (REPO / "data" / "public-v1" / "test.jsonl", "civil"),
    "helpsteer": (REPO / "data" / "public-v1" / "test.jsonl", "helpsteer"),
    "injection": (REPO / "data" / "public-v1" / "test.jsonl", "injection"),
    "banking77": (REPO / "data" / "study-v1" / "test_banking77.jsonl", None),
    "banking77-k16": (REPO / "data" / "banking77-v1" / "test_seen_k16.jsonl", None),
    "clinc-unseen": (REPO / "data" / "study-v1" / "test_clinc150_unseen.jsonl", None),
}


def shuffled_context(examples):
    """Cyclic state swap. Labels stay with the original questions (jevlike control)."""
    if len(examples) < 2:
        raise ValueError("Need at least two examples for shuffled-context")
    out = []
    n = len(examples)
    for i, item in enumerate(examples):
        other = examples[(i + 1) % n]
        out.append(BenchItem(example_id=item.example_id + ":shuf-ctx", state=other.state, questions=item.questions))
    return out


def shuffled_options(examples, seed=17):
    """Permute Choice option order; Score/Noul questions are unchanged."""
    out = []
    for i, item in enumerate(examples):
        rng = random.Random(seed + i)
        questions = {}
        for qid, q in item.questions.items():
            if q.get("type") != "choice":
                questions[qid] = q
                continue
            keys = list(q["criteria"])
            order = keys[:]
            rng.shuffle(order)
            questions[qid] = {
                **q,
                "criteria": {k: q["criteria"][k] for k in order},
                "target": [q["target"][keys.index(k)] for k in order],
            }
        out.append(BenchItem(example_id=item.example_id + ":shuf-opt", state=item.state, questions=questions))
    return out


def apply_control(examples, control):
    if control in (None, "none"):
        return examples
    if control == "shuffled-context":
        return shuffled_context(examples)
    if control == "shuffled-options":
        return shuffled_options(examples)
    raise ValueError(f"unknown control {control}")


def load_task(name, *, limit=None, control="none"):
    if name == "synthetic":
        test = SYNTHETIC_DIR / "test.jsonl"
        if not test.exists():
            write_synthetic(SYNTHETIC_DIR)
        examples = load_jevlike_jsonl(test, limit=limit, prefix="synthetic")
    elif name == "wikispeedia":
        path = REPO / "data" / "jevlike" / "wikispeedia" / "jsonl" / "test.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run python -m demos.bench.tasks --wikispeedia")
        examples = load_jevlike_jsonl(path, limit=limit, prefix="wikispeedia")
    elif name == "wiki-hop":
        examples = []
    elif name in TASK_PATHS:
        path, family = TASK_PATHS[name]
        examples = load_systemone_jsonl(path, limit=limit, family=family)
    else:
        raise ValueError(f"unknown task {name}")
    return apply_control(examples, control)


def _stable(text):
    import hashlib

    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _title(text):
    from urllib.parse import unquote

    return unquote(text).replace("_", " ")


def build_wikispeedia(root, output, max_options=64):
    """Same next-click JSONL as jevlike (West & Leskovec, WWW 2012)."""
    root, output = Path(root), Path(output)
    graph_dir = root / "wikispeedia_paths-and-graph"
    outgoing = {}
    for line in (graph_dir / "links.tsv").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            source, target = line.split("\t")
            outgoing.setdefault(source, []).append(target)
    output.mkdir(parents=True, exist_ok=True)
    handles = {name: (output / f"{name}.jsonl").open("w", encoding="utf-8") for name in ("train", "validation", "test")}
    counts = {name: 0 for name in handles}
    try:
        lines = (graph_dir / "paths_finished.tsv").read_text(encoding="utf-8").splitlines()
        for row, line in enumerate(lines):
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            path = []
            for node in fields[3].split(";"):
                if node == "<":
                    if len(path) > 1:
                        path.pop()
                else:
                    path.append(node)
            if len(path) < 2:
                continue
            step = _stable(f"{fields[0]}:{fields[1]}:{row}") % (len(path) - 1)
            current, click, target = path[step], path[step + 1], path[-1]
            candidates = list(dict.fromkeys(outgoing.get(current, ())))
            if len(candidates) < 2 or click not in candidates:
                continue
            rng = random.Random(_stable(f"{row}:{target}:menu"))
            others = [item for item in candidates if item != click]
            rng.shuffle(others)
            menu = [click] + others[: max_options - 1]
            rng.shuffle(menu)
            article = root / "plaintext_articles" / f"{current}.txt"
            body = " ".join(article.read_text(encoding="utf-8", errors="replace").split())
            payload = {
                "context": f"Target article: {_title(target)}\nCurrent article: {_title(current)}\n{body[:2048]}",
                "options": [_title(item) for item in menu],
                "label": menu.index(click),
            }
            bucket = _stable(target + ":split") % 10
            split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
            handles[split].write(json.dumps(payload, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def download_wikispeedia(root):
    import tarfile
    import urllib.request

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    base = "https://snap.stanford.edu/data/wikispeedia"
    for archive in ("wikispeedia_paths-and-graph.tar.gz", "wikispeedia_articles_plaintext.tar.gz"):
        dest = root / archive
        if not dest.exists():
            urllib.request.urlretrieve(f"{base}/{archive}", dest)
        if archive.endswith("paths-and-graph.tar.gz") and not (root / "wikispeedia_paths-and-graph").exists():
            with tarfile.open(dest) as tar:
                tar.extractall(root, filter="data")
        if archive.endswith("plaintext.tar.gz") and not (root / "plaintext_articles").exists():
            with tarfile.open(dest) as tar:
                tar.extractall(root, filter="data")
    return root


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--wikispeedia", action="store_true")
    args = p.parse_args(argv)
    if args.synthetic:
        print(write_synthetic(SYNTHETIC_DIR))
    if args.wikispeedia:
        root = REPO / "data" / "jevlike" / "wikispeedia"
        download_wikispeedia(root)
        print(build_wikispeedia(root, root / "jsonl"))


if __name__ == "__main__":
    main()
