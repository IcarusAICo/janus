"""Training mixes for the WP3 breadth experiment, with provenance and disjointness checks."""

from collections import Counter
import json
from pathlib import Path

from ..data import assert_disjoint, file_hash, load_requests, write_json, write_jsonl

EVALUATION_FILES = ("data/study-v1/benchmark.jsonl", "data/study-v1/test_banking77_unseen.jsonl",
                    "data/study-v1/test_clinc150_unseen.jsonl", "data/phase1-v1/test.jsonl", "data/phase1-v1/test_post_unseen.jsonl",
                    "data/public-v1/test.jsonl", "data/public-v1/test_mmlu_pro.jsonl", "data/banking77-v1/test_seen_k16.jsonl")


def _rows(path, default_source):
    out = []
    # Iterate the handle: str.splitlines() also breaks on Unicode separators inside state strings.
    for line in Path(path).open():
        if line.strip():
            row = json.loads(line)
            row.setdefault("source", default_source)
            out.append(row)
    return out


def _too_long(rows, max_tokens, tokenizer, mode):
    """Drop rows whose packed request exceeds the training cap; return (kept, dropped_counts_by_source)."""
    from collections import Counter as _Counter
    from ..packing import pack_request
    from ..schema import Request
    kept, dropped = [], _Counter()
    for row in rows:
        request = Request.from_dict({k: v for k, v in row.items() if k in ("state", "questions", "group_id")})
        if pack_request(request, tokenizer, mode, 10 ** 9).token_count > max_tokens:
            dropped[row["source"]] += 1
        else:
            kept.append(row)
    return kept, dict(dropped)


def prepare_mix(output, phase1="data/phase1-v1", public="data/public-v1", t3="data/t3-pilot", seed=17, evaluation_files=EVALUATION_FILES,
                max_tokens=None, tokenizer=None, mode="tree"):
    output, phase1, public, t3 = Path(output), Path(phase1), Path(public), Path(t3)
    output.mkdir(parents=True, exist_ok=False)
    splits = {}
    for name in ("train", "dev", "calibration"):
        splits[name] = _rows(phase1 / f"{name}.jsonl", "phase1") + _rows(public / f"{name}.jsonl", "public")
    t3_rows = []
    for path in sorted(t3.glob("*.jsonl")):
        # T4 rows (checkers disagreed with gold) are excluded; so are T3_second rows (accepted only by the second
        # checker): the 2026-09-18 human audit marked 26 of the 30 sampled T3_second rows ambiguous.
        t3_rows += [r for r in _rows(path, f"t3:{path.stem}") if r.get("tier") == "T3"
                    and (r.get("checks") or {}).get("outcome", "T3") != "T3_second"]
    # Training rows whose normalised state text coincides with any evaluation state are dropped and recorded,
    # so the fixed evaluation sets stay intact and the leakage check below can pass honestly.
    from ..data import state_hash
    evaluation_hashes = {}
    for path in evaluation_files:
        if Path(path).exists():
            for request in load_requests(path):
                evaluation_hashes.setdefault(state_hash(request.state), path)
    dropped = []
    def _filter(rows):
        kept = []
        for row in rows:
            state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], sort_keys=True, ensure_ascii=False)
            collision = evaluation_hashes.get(state_hash(state))
            if collision:
                dropped.append({"group_id": row["group_id"], "source": row["source"], "evaluation_file": str(collision)})
            else:
                kept.append(row)
        return kept
    splits["train"] = _filter(splits["train"])
    t3_rows = _filter(t3_rows)
    # Within-split deduplication by normalised state text (the T3 volume repeats a few dozen states); first row wins.
    within_dropped = {}
    def _dedupe(rows, name):
        seen, kept = set(), []
        for row in rows:
            state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], sort_keys=True, ensure_ascii=False)
            key = state_hash(state)
            if key in seen:
                within_dropped[name] = within_dropped.get(name, 0) + 1
            else:
                seen.add(key); kept.append(row)
        return kept
    train_and_t3 = _dedupe(splits["train"] + t3_rows, "train_C")
    splits["train"] = _dedupe(splits["train"], "train_B")
    t3_rows = train_and_t3[len(splits["train"]):] if len(train_and_t3) >= len(splits["train"]) else []
    t3_rows = _dedupe(t3_rows, "t3")
    too_long = {}
    if max_tokens is not None:
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", revision="da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
        for name in ("train", "dev", "calibration"):
            splits[name], too_long[name] = _too_long(splits[name], max_tokens, tokenizer, mode)
        t3_rows, too_long["t3"] = _too_long(t3_rows, max_tokens, tokenizer, mode)
    files = {"train_B": splits["train"], "train_C": splits["train"] + t3_rows, "dev_BC": splits["dev"], "calibration_BC": splits["calibration"]}
    for name, rows in files.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    check = {name: load_requests(output / f"{name}.jsonl") for name in ("train_C", "dev_BC", "calibration_BC")}
    assert_disjoint(check)
    # Evaluation files may overlap one another by design (the panel is drawn from the test pool); each is checked against training only.
    for path in evaluation_files:
        if Path(path).exists():
            assert_disjoint({**check, path: load_requests(path)})
    manifest = {"dataset": "JEV_MIX_V1", "seed": seed,
                "counts": {name: len(rows) for name, rows in files.items()},
                "tiers": {name: dict(Counter(r.get("tier", "unknown") for r in rows)) for name, rows in files.items()},
                "sources": {name: dict(Counter(r["source"] for r in rows)) for name, rows in files.items()},
                "t3_note": "T3 rows are model-consensus labels (spec WP3 3.1), not ground truth; T4 and T3_second rows are excluded.",
                "dropped_for_evaluation_overlap": dropped,
                "dropped_within_split_duplicates": within_dropped,
                "dropped_over_max_tokens": {"max_tokens": max_tokens, "mode": mode, "by_split_and_source": too_long},
                "inputs": {str(p): file_hash(p) for root in (phase1, public, t3) for p in sorted(root.glob("*.jsonl")) if p.exists()},
                "files": {p.name: file_hash(p) for p in sorted(output.glob("*.jsonl"))}}
    write_json(output / "manifest.json", manifest)
    return manifest
