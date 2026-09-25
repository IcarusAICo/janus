"""Pinned, grouped multi-domain study data and deterministic menu augmentation.

Source annotations are kept in train_sources.jsonl, never appended to model state.
The model's packing code does not encode question IDs or source group IDs.
"""

from collections import Counter, defaultdict
from dataclasses import replace
import csv
import hashlib
import io
import json
from pathlib import Path
import random
from urllib.request import urlopen

from .data import (assert_disjoint, banking_request, file_hash,
                   shuffled_options, split_banking, state_hash, write_json, write_jsonl)
from .schema import Question, Request


DATASET = "JEV_MULTI_DOMAIN_V1"
DOMAINS = ("banking77", "clinc150", "sst5", "snli")
SOURCES = {
    "clinc150": {
        "repository": "clinc/clinc_oos", "configuration": "plus",
        "revision": "155b9c710419136e17307b80d0a13e68cd46b4ec",
        "license": "CC-BY-3.0",
        "license_source": "https://huggingface.co/datasets/clinc/clinc_oos",
        "citation": "Larson et al. (2019), An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction",
        "paper": "https://aclanthology.org/D19-1131/",
    },
    "sst5": {
        "repository": "SetFit/sst5", "configuration": None,
        "revision": "e51bdcd8cd3a30da231967c1a249ba59361279a3",
        "license": "Not specified in the SetFit/sst5 dataset card; consult original SST terms",
        "license_source": "https://nlp.stanford.edu/sentiment/",
        "citation": "Socher et al. (2013), Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank",
        "paper": "https://aclanthology.org/D13-1170/",
    },
    "snli": {
        "repository": "stanfordnlp/snli", "configuration": "plain_text",
        "revision": "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b",
        "license": "CC-BY-SA-4.0",
        "license_source": "https://nlp.stanford.edu/projects/snli/",
        "citation": "Bowman et al. (2015), A large annotated corpus for learning natural language inference",
        "paper": "https://aclanthology.org/D15-1075/",
    },
}


def row_state(row):
    if "premise" in row:
        return f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}"
    return row["text"]


def _row_label(row):
    return row["category"] if "category" in row else row["label"]


def _rank(seed, key):
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def balanced_rows(rows, limit, seed=17, stratum=None):
    """Round-robin deterministic label strata, excluding the unused remainder."""
    if limit is None or limit >= len(rows):
        return sorted(rows, key=lambda r: _rank(seed, state_hash(row_state(r))))
    if limit < 0:
        raise ValueError("Sample limit must be nonnegative")
    strata = defaultdict(list)
    for row in rows:
        strata[str(stratum(row) if stratum else _row_label(row))].append(row)
    for group in strata.values():
        group.sort(key=lambda r: _rank(seed, state_hash(row_state(r))))
    keys = sorted(strata, key=lambda key: _rank(seed, key))
    selected, offset = [], 0
    while len(selected) < limit:
        for key in keys:
            if offset < len(strata[key]):
                selected.append(strata[key][offset])
                if len(selected) == limit:
                    return selected
        offset += 1
    return selected


def split_domain_rows(domain, source_splits, seed=17, eval_limit=128,
                      train_limit=None, heldout_count=0):
    """Split by normalized source group before sampling or making questions.

    Official test groups take precedence over validation, which takes precedence
    over train. Within a winning group, only rows from that official split remain.
    All hypotheses sharing an SNLI premise belong to the same group.
    """
    if domain not in {"clinc150", "sst5", "snli"}:
        raise ValueError(f"Unsupported source domain: {domain}")
    priority = {"train": 0, "validation": 1, "test": 2}
    states, conflicts, group_priority = {}, set(), {}
    report = {"invalid_rows_dropped": 0, "duplicate_rows_dropped": 0,
              "lower_priority_group_rows_dropped": 0, "heldout_rows_dropped": 0,
              "raw_counts": {name: len(rows) for name, rows in source_splits.items()}}
    for source in priority:
        for raw in source_splits.get(source, []):
            if domain == "clinc150":
                category = raw.get("category")
                valid = isinstance(category, str) and category != "oos" and bool(category.strip())
                row = {"text": raw.get("text"), "category": category}
            elif domain == "sst5":
                label = raw.get("label")
                valid = isinstance(label, int) and label in range(5)
                row = {"text": raw.get("text"), "label": label}
            else:
                label = raw.get("label")
                valid = isinstance(label, int) and label in range(3)
                row = {"premise": raw.get("premise"), "hypothesis": raw.get("hypothesis"), "label": label}
            text_keys = ("premise", "hypothesis") if domain == "snli" else ("text",)
            if not valid or any(not isinstance(row[key], str) or not row[key].strip() for key in text_keys):
                report["invalid_rows_dropped"] += 1
                continue
            key = state_hash(row_state(row))
            group = state_hash(row["premise"]) if domain == "snli" else key
            row.update(domain=domain, source=source, group_id=f"{domain}:{group}")
            if key in states:
                report["duplicate_rows_dropped"] += 1
                if _row_label(states[key]) != _row_label(row):
                    conflicts.add(key)
            states[key] = row
            group_priority[group] = max(priority[source], group_priority.get(group, -1))
    report["conflicting_states_dropped"] = len(conflicts)
    categories = sorted({_row_label(r) for r in states.values()}) if domain == "clinc150" else []
    if heldout_count and not 0 < heldout_count < len(categories) - 1:
        raise ValueError("Holdout must leave at least two training intents")
    unseen = set(random.Random(seed).sample(categories, heldout_count))
    if categories:
        report.update(seen_intents=sorted(set(categories) - unseen), unseen_intents=sorted(unseen))
    splits = {name: [] for name in ("train", "dev", "calibration", "test")}
    for key, row in sorted(states.items()):
        if key in conflicts:
            continue
        group = row["group_id"].split(":", 1)[1]
        if priority[row["source"]] != group_priority[group]:
            report["lower_priority_group_rows_dropped"] += 1
            continue
        if row["source"] != "test" and _row_label(row) in unseen:
            report["heldout_rows_dropped"] += 1
            continue
        if row["source"] == "validation":
            name = "dev" if int(_rank(seed, group), 16) % 2 == 0 else "calibration"
        else:
            name = row["source"]
        splits[name].append(row)
    before = {name: len(rows) for name, rows in splits.items()}
    for name in ("dev", "calibration"):
        splits[name] = balanced_rows(splits[name], eval_limit, seed)
    splits["train"] = balanced_rows(splits["train"], train_limit, seed)
    report["sampling_excluded"] = {name: before[name] - len(rows) for name, rows in splits.items()}
    report["counts"] = {name: len(rows) for name, rows in splits.items()}
    return splits, report


def intent_request(row, pool, seed=17, cardinalities=(2, 4, 8), omit=.2):
    cardinalities = tuple(k for k in cardinalities if k <= len(pool))
    if not cardinalities:
        raise ValueError("At least two allowed intents are needed")
    request = banking_request(row, pool, seed, cardinalities, omit)
    domain = row.get("domain", "banking77")
    questions = tuple(replace(q, id=f"{domain}:{q.id}",
                              instructions=q.instructions.replace("customer's message", "user's request"))
                      for q in request.questions)
    return replace(request, questions=questions)


def sentiment_request(row):
    label = row["label"]
    if label not in range(5):
        raise ValueError("SST5 label must be 0 through 4")
    return Request.from_dict({"state": row["text"], "group_id": row["group_id"], "questions": {
        "sst5:sentiment": {"type": "score", "instructions": "Rate the overall sentiment of this movie review.",
                            "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"],
                            "target": [float(i == label) for i in range(5)]},
        "sst5:positive": {"type": "noul", "instructions": "Does this movie review express positive overall sentiment? Neutral sentiment does not count as positive.",
                          "target": [float(label <= 2), float(label >= 3)]}}})


def nli_request(row, seed=17):
    label = row["label"]
    if label not in range(3):
        raise ValueError("SNLI label must be entailment=0, neutral=1, contradiction=2")
    # A premise group can have several hypotheses: keep question identities unique.
    suffix = state_hash(row_state(row))[:16]
    request = Request.from_dict({"state": row_state(row), "group_id": row["group_id"], "questions": {
        f"snli:relation:{suffix}": {"type": "choice",
            "instructions": "Assume the premise is true. What relationship does the hypothesis have to the premise?",
            "criteria": {"o0": "Entailed: the premise establishes that the hypothesis is true.",
                         "o1": "Unknown: the premise establishes neither the hypothesis nor its negation.",
                         "o2": "Contradicted: the premise establishes that the hypothesis is false."},
            "target": [float(i == label) for i in range(3)]},
        f"snli:entailed:{suffix}": {"type": "noul",
            "instructions": "Assume the premise is true. Is the hypothesis logically entailed by the premise? Both contradiction and insufficient information mean it is not entailed.",
            "criteria": {"false": "No. The premise does not establish the hypothesis; it may contradict it or leave it unknown.",
                         "true": "Yes. The premise logically establishes the hypothesis."},
            "target": [float(label != 0), float(label == 0)]}}})
    return shuffled_options(request, f"{seed}:{state_hash(request.state)}")


def request_for_row(row, pools, seed=17, omit=.2):
    domain = row["domain"]
    if domain in {"banking77", "clinc150"}:
        return intent_request(row, pools[domain], seed, omit=omit)
    if domain == "sst5":
        return sentiment_request(row)
    if domain == "snli":
        return nli_request(row, seed)
    raise ValueError(f"Unknown source domain: {domain}")


def assert_study_disjoint(splits):
    """Check normalized states globally, plus source groups between splits."""
    assert_disjoint(splits)
    seen = {}
    pairs = set()
    for name, requests in splits.items():
        for request in requests:
            key = state_hash(request.state)
            if key in seen:
                raise ValueError(f"Normalized state duplicate in {seen[key]} and {name}")
            seen[key] = name
            for question in request.questions:
                identity = (request.group_id, question.id)
                if identity in pairs:
                    raise ValueError(f"Duplicate source group/question identity: {identity}")
                pairs.add(identity)


def study_training_sources(train_path):
    train_path = Path(train_path)
    manifest = json.loads((train_path.parent / "manifest.json").read_text())
    if manifest.get("dataset") != DATASET:
        raise ValueError("Expected a prepared multi-domain study manifest")
    source_path = train_path.parent / "train_sources.jsonl"
    for path in (train_path, source_path):
        if not path.exists() or manifest.get("files", {}).get(path.name) != file_hash(path):
            raise ValueError(f"Missing or changed prepared study data: {path}")
    sources = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
    return sources, manifest["training_intent_pools"], manifest


def study_training_epoch(requests, seed, epoch, sources, pools, omit=.2):
    """Regenerate every request's menu for this epoch; `omit` is the none-option omission rate for intent rows."""
    by_state = {state_hash(row_state(row)): row for row in sources}
    request_states = {state_hash(request.state) for request in requests}
    if (len(by_state) != len(sources) or len(request_states) != len(requests)
            or not request_states <= set(by_state)):
        raise ValueError("Training source states do not match prepared requests")
    for row in sources:
        if row["domain"] in {"banking77", "clinc150"} and row["category"] not in pools[row["domain"]]:
            raise ValueError("Training source contains a forbidden intent")
    result = []
    for request in requests:
        row = by_state[state_hash(request.state)]
        if row["group_id"] != request.group_id:
            raise ValueError("Training source grouping differs from prepared requests")
        result.append(request_for_row(row, pools, seed + epoch * 1_000_003, omit))
    return result


def augment_study(requests, train_path, seed, epoch):
    sources, pools, _ = study_training_sources(train_path)
    return study_training_epoch(requests, seed, epoch, sources, pools)


def download_study_sources(cache_dir=".cache/study"):
    """Fetch immutable public revisions, recording hashes of every source file."""
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    cache = Path(cache_dir)
    all_rows, provenance = {}, {}
    for domain, spec in SOURCES.items():
        rows, hashes = {}, {}
        for split in ("train", "validation", "test"):
            filename = (f"{'dev' if split == 'validation' else split}.jsonl" if domain == "sst5"
                        else f"{spec['configuration']}/{split}-00000-of-00001.parquet")
            path = hf_hub_download(spec["repository"], filename, repo_type="dataset", token=False,
                                   revision=spec["revision"], cache_dir=cache / "hub")
            hashes[filename] = file_hash(path)
            if domain == "sst5":
                values = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
                values = [{"text": r["text"], "label": r["label"]} for r in values]
            else:
                dataset = Dataset.from_parquet(path, cache_dir=str(cache / "datasets"))
                if domain == "clinc150":
                    labels = dataset.features["intent"].names
                    values = [{"text": r["text"], "category": labels[r["intent"]]} for r in dataset]
                else:
                    if dataset.features["label"].names != ["entailment", "neutral", "contradiction"]:
                        raise ValueError("Unexpected SNLI label mapping")
                    values = list(dataset)
            rows[split] = values
        card = hf_hub_download(spec["repository"], "README.md", repo_type="dataset", token=False,
                               revision=spec["revision"], cache_dir=cache / "hub")
        hashes["README.md"] = file_hash(card)
        provenance[domain] = {**spec, "source_sha256": hashes,
                              "source": f"https://huggingface.co/datasets/{spec['repository']}/tree/{spec['revision']}"}
        all_rows[domain] = rows
    return all_rows, provenance


def _banking_rows(banking_dir):
    directory = Path(banking_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("dataset") != "BANKING77":
        raise ValueError("Expected the source-pinned BANKING77 preparation")
    for name, expected in manifest["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Changed BANKING77 source artifact: {name}")
    raw = {}
    for name, expected in manifest["source_sha256"].items():
        url = f"{manifest['source']}/{manifest['source_revision']}/banking_data/{name}"
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"BANKING77 source hash mismatch: {name}")
        raw[name] = payload.decode("utf-8")
    categories = json.loads(raw["categories.json"])
    splits, recreated = split_banking(list(csv.DictReader(io.StringIO(raw["train.csv"]))),
                                      list(csv.DictReader(io.StringIO(raw["test.csv"]))), categories,
                                      seed=manifest["seed"], heldout_count=len(manifest["unseen_intents"]))
    if any(recreated[key] != manifest[key] for key in ("seen_intents", "unseen_intents", "counts")):
        raise ValueError("Recreated BANKING77 split disagrees with original manifest")
    for rows in splits.values():
        for row in rows:
            row.update(domain="banking77", group_id=f"banking77:{row['group_id']}")
    splits["test"] = splits.pop("test_seen") + splits.pop("test_unseen")
    return splits, manifest


def _exclude_cross_domain_states(domain_splits):
    owners = defaultdict(set)
    for domain, splits in domain_splits.items():
        for rows in splits.values():
            for row in rows:
                owners[state_hash(row_state(row))].add(domain)
    collisions = {key for key, domains in owners.items() if len(domains) > 1}
    dropped = Counter()
    for domain, splits in domain_splits.items():
        for name, rows in splits.items():
            kept = [r for r in rows if state_hash(row_state(r)) not in collisions]
            dropped[f"{domain}/{name}"] = len(rows) - len(kept)
            splits[name] = kept
    return {"normalized_states": len(collisions), "rows_dropped": dict(dropped)}


def prepare_study(output, banking_dir="data/banking77-v1", seed=17, snli_train=20_000,
                  eval_limit=128, benchmark_per_domain=200, clinc_heldout=30,
                  cache_dir=".cache/study"):
    """Create a fresh study directory; existing artifacts are never overwritten."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing study directory: {output}")
    if min(snli_train, eval_limit, benchmark_per_domain) < 1:
        raise ValueError("Training, evaluation, and benchmark limits must be positive")
    banking, bank_manifest = _banking_rows(banking_dir)
    for name in ("dev", "calibration"):
        banking[name] = balanced_rows(banking[name], eval_limit, seed)
    raw, provenance = download_study_sources(cache_dir)
    domain_splits = {"banking77": banking}
    reports = {"banking77": {"seen_intents": bank_manifest["seen_intents"],
                              "unseen_intents": bank_manifest["unseen_intents"],
                              "original_counts": bank_manifest["counts"],
                              "policy": "Original train and intent holdouts preserved; dev/calibration capped; unused rows excluded"}}
    for domain in ("clinc150", "sst5", "snli"):
        domain_splits[domain], reports[domain] = split_domain_rows(
            domain, raw[domain], seed, eval_limit,
            train_limit=snli_train if domain == "snli" else None,
            heldout_count=clinc_heldout if domain == "clinc150" else 0)
    cross_domain = _exclude_cross_domain_states(domain_splits)
    pools = {domain: reports[domain]["seen_intents"] for domain in ("banking77", "clinc150")}
    all_pools = {domain: sorted(reports[domain]["seen_intents"] + reports[domain]["unseen_intents"])
                 for domain in pools}
    requests = {name: [] for name in ("train", "dev", "calibration", "test")}
    test_domains, benchmarks, sources = {}, {}, []
    for domain in DOMAINS:
        splits = domain_splits[domain]
        reports[domain]["counts"] = {name: len(rows) for name, rows in splits.items()}
        for name, rows in splits.items():
            converted = []
            for row in rows:
                # Seen examples never show heldout intent descriptions as distractors.
                row_pools = all_pools if name == "test" and row.get("category") in reports[domain].get("unseen_intents", []) else pools
                converted.append(request_for_row(row, row_pools, seed))
            requests[name].extend(converted)
            if name == "test":
                test_domains[domain] = converted
        panel_rows = balanced_rows(splits["test"], benchmark_per_domain, seed)
        panel_states = {state_hash(row_state(row)) for row in panel_rows}
        benchmarks[domain] = [r for r in test_domains[domain] if state_hash(r.state) in panel_states]
        sources.extend(splits["train"])
    assert_study_disjoint(requests)
    output.mkdir(parents=True, exist_ok=False)
    for name, values in requests.items():
        write_jsonl(output / f"{name}.jsonl", [r.to_dict() for r in values])
    for domain, values in test_domains.items():
        write_jsonl(output / f"test_{domain}.jsonl", [r.to_dict() for r in values])
        write_jsonl(output / f"benchmark_{domain}.jsonl", [r.to_dict() for r in benchmarks[domain]])
        if domain in pools:
            by_state = {state_hash(row_state(r)): r for r in domain_splits[domain]["test"]}
            for split in ("seen", "unseen"):
                subset = [r for r in values if (by_state[state_hash(r.state)]["category"] in reports[domain]["unseen_intents"]) == (split == "unseen")]
                if subset:
                    write_jsonl(output / f"test_{domain}_{split}.jsonl", [r.to_dict() for r in subset])
    panel = [request for domain in DOMAINS for request in benchmarks[domain]]
    write_jsonl(output / "benchmark.jsonl", [r.to_dict() for r in panel])
    write_jsonl(output / "train_sources.jsonl", sources)
    provenance["banking77"] = {"source_manifest": str(Path(banking_dir) / "manifest.json"),
                               "source_manifest_sha256": file_hash(Path(banking_dir) / "manifest.json"),
                               **{k: bank_manifest[k] for k in ("source", "source_revision", "source_sha256", "license", "citation")}}
    manifest = {"dataset": DATASET, "seed": seed, "sources": provenance, "domains": reports,
                "training_intent_pools": pools, "counts": {name: len(rows) for name, rows in requests.items()},
                "benchmark_counts": {domain: len(rows) for domain, rows in benchmarks.items()},
                "cross_domain_deduplication": cross_domain,
                "policy": {"normalization": "Lowercase and collapse whitespace, then SHA256",
                           "snli_group": "Normalized premise, across every hypothesis and all official splits",
                           "priority": "Official test > official validation > official train",
                           "validation": "Group-hash divide validation into dev/calibration; label-balanced cap; remainder excluded",
                           "banking77": "Preserve original split and heldout intent membership",
                           "cross_domain_states": "Remove every occurrence of a state shared across domains",
                           "benchmark": "Deterministic class-balanced subset of test, shared by all compared systems",
                           "clinc_oos": "Excluded: study uses 150 annotated in-scope intents",
                           "sst5_noul": "Positive iff original label is 3 or 4; neutral is not positive",
                           "snli_noul": "Entailed iff original label is 0; unknown is not conflated with false"},
                "parameters": {"snli_train": snli_train, "eval_limit": eval_limit,
                               "benchmark_per_domain": benchmark_per_domain, "clinc_heldout": clinc_heldout},
                "files": {p.name: file_hash(p) for p in sorted(output.glob("*.jsonl"))}}
    write_json(output / "manifest.json", manifest)
    return manifest
