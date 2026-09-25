"""CPU-only descriptive and integrity audit of a prepared multi-domain study."""

from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

from .data import file_hash, state_hash, write_json
from .schema import Request


SPLITS = ("train", "dev", "calibration", "test")
INTENT_DOMAINS = ("banking77", "clinc150")


def _load(path):
    return [Request.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def _domain(request):
    return request.group_id.split(":", 1)[0]


def _target_name(question):
    if question.target is None:
        return "missing"
    index = max(range(len(question.target)), key=question.target.__getitem__)
    option = question.options[index]
    if question.kind == "noul":
        return option.key
    if question.kind == "score":
        return f"{index}:{option.description}"
    if question.id.startswith("snli:relation:"):
        return option.description.split(":", 1)[0].lower()
    return option.description


def _summary(requests):
    by_type, menu_sizes, questions_per_state = Counter(), Counter(), Counter()
    outcomes, slots = defaultdict(Counter), defaultdict(Counter)
    missing_targets = soft_targets = 0
    for request in requests:
        questions_per_state[str(len(request.questions))] += 1
        for question in request.questions:
            by_type[question.kind] += 1
            family = ":".join(question.id.split(":")[:2])
            outcomes[family][_target_name(question)] += 1
            if question.kind == "choice":
                menu_sizes[str(len(question.options))] += 1
            if question.target is None:
                missing_targets += 1
            else:
                index = max(range(len(question.target)), key=question.target.__getitem__)
                slots[family][str(index)] += 1
                soft_targets += int(question.target[index] != 1.)
    return {"states": len(requests), "decisions": sum(by_type.values()),
            "groups": len({r.group_id for r in requests}),
            "by_type": dict(sorted(by_type.items())),
            "questions_per_state": dict(sorted(questions_per_state.items())),
            "choice_menu_sizes": dict(sorted(menu_sizes.items())),
            "target_balance": {k: dict(sorted(v.items())) for k, v in sorted(outcomes.items())},
            "target_position_balance": {k: dict(sorted(v.items())) for k, v in sorted(slots.items())},
            "missing_targets": missing_targets, "soft_targets": soft_targets}


def _split_summary(requests):
    domains = defaultdict(list)
    for request in requests:
        domains[_domain(request)].append(request)
    return {**_summary(requests), "domains": {domain: _summary(rows) for domain, rows in sorted(domains.items())}}


def _duplicates(requests):
    states = Counter(state_hash(r.state) for r in requests)
    pairs = Counter((r.group_id, q.id) for r in requests for q in r.questions)
    return {"normalized_states": sum(count - 1 for count in states.values()),
            "question_pairs": sum(count - 1 for count in pairs.values())}


def _snli_groups(requests):
    counts = Counter(r.group_id for r in requests if _domain(r) == "snli")
    return {"states": sum(counts.values()), "groups": len(counts),
            "shared_groups": sum(size > 1 for size in counts.values()),
            "states_in_shared_groups": sum(size for size in counts.values() if size > 1),
            "maximum_size": max(counts.values(), default=0),
            "size_histogram": dict(sorted(Counter(str(size) for size in counts.values()).items()))}


def _snli_group_mismatches(requests):
    errors = 0
    for request in requests:
        if _domain(request) != "snli":
            continue
        premise, separator, _ = request.state.partition("\nHypothesis: ")
        if not premise.startswith("Premise: ") or not separator:
            errors += 1
        elif request.group_id != "snli:" + state_hash(premise[len("Premise: "):]):
            errors += 1
    return errors


def audit_study(data_dir, output):
    """Write counts, benchmark labels, and explicit pass/fail integrity checks.

    The source data is read only. Failed integrity checks remain in the report
    instead of stopping at the first failure, so the audit is useful diagnostically.
    Benchmark labels are keyed as labels[group_id][question_id] for paired analysis.
    """
    directory = Path(data_dir)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    requests = {name: _load(directory / f"{name}.jsonl") for name in (*SPLITS, "benchmark")}
    expected_files = manifest.get("files", {})
    mismatches = [name for name, expected in expected_files.items()
                  if not (directory / name).is_file() or file_hash(directory / name) != expected]
    unrecorded_files = [f"{name}.jsonl" for name in (*SPLITS, "benchmark") if f"{name}.jsonl" not in expected_files]
    within = {name: _duplicates(rows) for name, rows in requests.items()}
    overlap = {}
    state_sets = {name: {state_hash(r.state) for r in rows} for name, rows in requests.items()}
    group_sets = {name: {r.group_id for r in rows} for name, rows in requests.items()}
    for left, right in combinations(SPLITS, 2):
        overlap[f"{left}|{right}"] = {"normalized_states": len(state_sets[left] & state_sets[right]),
                                     "groups": len(group_sets[left] & group_sets[right])}
    test_by_state = {state_hash(r.state): r for r in requests["test"]}
    panel = requests["benchmark"]
    exact_members = sum(test_by_state.get(state_hash(r.state)) == r for r in panel)
    state_members = sum(state_hash(r.state) in test_by_state for r in panel)
    partitions, partition_overlap, partition_non_test, partition_non_exact = {}, {}, {}, {}
    for domain in INTENT_DOMAINS:
        partitions[domain] = {}
        partition_rows = {}
        for name in ("seen", "unseen"):
            path = directory / f"test_{domain}_{name}.jsonl"
            rows = _load(path) if path.is_file() else []
            partition_rows[name] = rows
            partitions[domain][name] = {state_hash(r.state) for r in rows}
        partition_overlap[domain] = len(partitions[domain]["seen"] & partitions[domain]["unseen"])
        partition_non_test[domain] = sum(len(values - state_sets["test"]) for values in partitions[domain].values())
        partition_non_exact[domain] = sum(test_by_state.get(state_hash(r.state)) != r
                                          for rows in partition_rows.values() for r in rows)
    partition_counts = {domain: {"seen": 0, "unseen": 0, "unclassified": 0} for domain in INTENT_DOMAINS}
    pair_labels = defaultdict(dict)
    for request in panel:
        domain = _domain(request)
        partition = None
        if domain in partitions:
            state = state_hash(request.state)
            membership = [name for name, states in partitions[domain].items() if state in states]
            partition = membership[0] if len(membership) == 1 else "unclassified"
            partition_counts[domain][partition] += 1
        for question in request.questions:
            pair_labels[request.group_id][question.id] = {"domain": domain, "kind": question.kind,
                                                        "intent_partition": partition}
    source_path = directory / "train_sources.jsonl"
    source_balance = defaultdict(Counter)
    source_states, source_groups = [], {}
    if source_path.is_file():
        for line in source_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            source_balance[row["domain"]][str(row["category"] if "category" in row else row["label"])] += 1
            text = f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}" if "premise" in row else row["text"]
            key = state_hash(text)
            source_states.append(key)
            source_groups[key] = row["group_id"]
    source_state_set = set(source_states)
    source_check = {"missing_states": len(state_sets["train"] - source_state_set),
                    "extra_states": len(source_state_set - state_sets["train"]),
                    "duplicate_states": len(source_states) - len(source_state_set),
                    "group_mismatches": sum(source_groups.get(state_hash(r.state)) != r.group_id for r in requests["train"])}
    summaries = {name: _split_summary(rows) for name, rows in requests.items()}
    count_mismatches = {name: {"manifest": manifest["counts"][name], "observed": len(requests[name])}
                        for name in SPLITS if name in manifest.get("counts", {})
                        and manifest["counts"][name] != len(requests[name])}
    snli_mismatches = {name: _snli_group_mismatches(rows) for name, rows in requests.items()}
    checks = {"manifest_dataset": manifest.get("dataset"),
              "manifest_count_mismatches": count_mismatches,
              "file_hashes_checked": len(expected_files), "file_hash_mismatches": mismatches,
              "unrecorded_required_files": unrecorded_files,
              "within_split_duplicates": within, "split_overlap": overlap,
              "snli_premise_group_mismatches": snli_mismatches,
              "intent_partition_overlap": partition_overlap,
              "intent_partition_non_test_states": partition_non_test,
              "intent_partition_non_exact_test_members": partition_non_exact,
              "training_sources": source_check}
    checks["passed"] = (
        manifest.get("dataset") == "JEV_MULTI_DOMAIN_V1"
        and not mismatches and not unrecorded_files and not count_mismatches
        and not any(any(values.values()) for values in within.values())
        and not any(any(values.values()) for values in overlap.values())
        and not any(snli_mismatches.values()) and not any(source_check.values())
        and not any(partition_overlap.values()) and not any(partition_non_test.values())
        and not any(partition_non_exact.values())
        and not any(values["unclassified"] for values in partition_counts.values())
        and exact_members == len(panel)
        and not any(summary["missing_targets"] for summary in summaries.values())
    )
    report = {"schema_version": 1, "data_dir": str(directory), "seed": manifest.get("seed"),
              "source_manifest_sha256": file_hash(manifest_path),
              "splits": {name: summaries[name] for name in SPLITS},
              "benchmark": {**summaries["benchmark"], "exact_test_members": exact_members,
                            "state_test_members": state_members, "intent_partitions": partition_counts},
              "benchmark_question_labels": dict(pair_labels),
              "training_source_class_balance": {domain: dict(sorted(counts.items())) for domain, counts in sorted(source_balance.items())},
              "snli_premise_groups": {name: _snli_groups(rows) for name, rows in requests.items()},
              "checks": checks,
              "definitions": {"target_balance": "Counts by the meaning of the argmax target in each question family; not shuffled option position",
                              "intent_target_balance": "Choice balance describes menu outcomes, including none-of-these; underlying intents can be absent from menus",
                              "training_source_class_balance": "Original labels in the training source sidecar, distinct from menu targets",
                              "benchmark_question_labels": "Nested lookup by group_id then question_id; intent_partition is null outside intent domains",
                              "normalized_state": "SHA256 of lowercase state with whitespace collapsed",
                              "exact_test_members": "Request equality includes state, group ID, all question instructions, ordered criteria, and targets",
                              "snli_premise_groups": "Group sizes count the retained hypotheses sharing a normalized premise",
                              "soft_targets": "Non-one-hot targets; target_balance still reports the highest-probability outcome"}}
    write_json(output, report)
    return report
