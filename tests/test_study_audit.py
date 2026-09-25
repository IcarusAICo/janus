from dataclasses import replace
import importlib.util
import json

from janus.data import file_hash, write_json, write_jsonl
from janus.schema import Question, Request
from janus.study_data import nli_request, sentiment_request


def audit_module():
    assert importlib.util.find_spec("janus.study_audit") is not None, "Study audit module is missing"
    from janus import study_audit
    return study_audit


def fixture_data(tmp_path):
    def sentiment(text, label):
        from janus.data import state_hash
        return sentiment_request({"text": text, "label": label, "group_id": "sst5:" + state_hash(text)})

    train = [sentiment("Training review", 4)]
    dev = [sentiment("Development review", 2)]
    calibration = [sentiment("Calibration review", 0)]
    from janus.data import state_hash
    premise = "A shared premise."
    snli = [nli_request({"premise": premise, "hypothesis": f"Claim {i}", "label": i,
                         "group_id": "snli:" + state_hash(premise)}) for i in range(3)]
    bank = Request.from_dict({"state": "Where is my new card?", "group_id": "banking77:card",
                             "questions": {"banking77:intent": {"type": "choice", "instructions": "Choose intent",
                                 "criteria": {"o0": "card arrival", "o1": "None of these intents describes the message."},
                                 "target": [1, 0]}, "banking77:matches": {"type": "noul", "instructions": "Card arrival?", "target": [0, 1]}}})
    test = [sentiment("Test review", 1), bank, *snli]
    splits = {"train": train, "dev": dev, "calibration": calibration, "test": test,
              "benchmark": [bank, *snli], "test_banking77_seen": [bank], "test_banking77_unseen": []}
    for name, values in splits.items():
        write_jsonl(tmp_path / f"{name}.jsonl", [r.to_dict() for r in values])
    write_jsonl(tmp_path / "train_sources.jsonl", [{"domain": "sst5", "text": train[0].state,
                                                   "label": 4, "group_id": train[0].group_id}])
    write_json(tmp_path / "manifest.json", {"dataset": "JEV_MULTI_DOMAIN_V1", "seed": 17,
        "counts": {name: len(splits[name]) for name in ("train", "dev", "calibration", "test")},
        "files": {path.name: file_hash(path) for path in tmp_path.glob("*.jsonl")}})
    return splits


def test_audit_counts_decisions_meaningful_targets_and_premise_groups(tmp_path):
    m = audit_module()
    splits = fixture_data(tmp_path)
    output = tmp_path / "report.json"
    report = m.audit_study(tmp_path, output)
    test = report["splits"]["test"]
    assert test["states"] == 5
    assert test["decisions"] == 10
    assert test["by_type"] == {"choice": 4, "noul": 5, "score": 1}
    assert test["domains"]["snli"]["choice_menu_sizes"] == {"3": 3}
    assert test["domains"]["snli"]["target_balance"]["snli:relation"] == {"entailed": 1, "unknown": 1, "contradicted": 1}
    assert test["domains"]["sst5"]["target_balance"]["sst5:sentiment"] == {"1:Negative": 1}
    assert report["snli_premise_groups"]["test"] == {"states": 3, "groups": 1, "shared_groups": 1,
                                                     "states_in_shared_groups": 3, "maximum_size": 3, "size_histogram": {"3": 1}}
    assert report["training_source_class_balance"]["sst5"] == {"4": 1}
    assert report["checks"]["passed"] is True
    assert json.loads(output.read_text()) == report


def test_panel_is_exact_subset_and_pair_labels_preserve_intent_partition(tmp_path):
    m = audit_module()
    splits = fixture_data(tmp_path)
    report = m.audit_study(tmp_path, tmp_path / "audit.json")
    assert report["benchmark"]["states"] == 4
    assert report["benchmark"]["exact_test_members"] == 4
    assert report["benchmark"]["intent_partitions"]["banking77"] == {"seen": 1, "unseen": 0, "unclassified": 0}
    bank = splits["test_banking77_seen"][0]
    labels = report["benchmark_question_labels"][bank.group_id]["banking77:intent"]
    assert labels == {"domain": "banking77", "kind": "choice", "intent_partition": "seen"}
    assert sum(len(items) for items in report["benchmark_question_labels"].values()) == 8


def test_audit_exposes_state_and_group_leakage_even_with_distinct_ids(tmp_path):
    m = audit_module()
    splits = fixture_data(tmp_path)
    leaked_state = replace(splits["train"][0], group_id="clinc150:different")
    leaked_group = replace(splits["train"][0], state="A different review")
    write_jsonl(tmp_path / "dev.jsonl", [r.to_dict() for r in [*splits["dev"], leaked_state, leaked_group]])
    report = m.audit_study(tmp_path, tmp_path / "audit.json")
    assert report["checks"]["passed"] is False
    overlap = report["checks"]["split_overlap"]["train|dev"]
    assert overlap == {"normalized_states": 1, "groups": 1}
    assert "dev.jsonl" in report["checks"]["file_hash_mismatches"]


def test_changed_panel_question_is_not_an_exact_test_member(tmp_path):
    m = audit_module()
    splits = fixture_data(tmp_path)
    original = splits["benchmark"][0]
    changed = replace(original, questions=(replace(original.questions[0], target=(0., 1.)), *original.questions[1:]))
    write_jsonl(tmp_path / "benchmark.jsonl", [changed.to_dict()])
    report = m.audit_study(tmp_path, tmp_path / "audit.json")
    assert report["benchmark"]["exact_test_members"] == 0
    assert report["benchmark"]["state_test_members"] == 1
    assert report["checks"]["passed"] is False


def test_duplicate_panel_pairs_and_wrong_premise_group_are_reported(tmp_path):
    m = audit_module()
    splits = fixture_data(tmp_path)
    snli = splits["benchmark"][1]
    write_jsonl(tmp_path / "benchmark.jsonl", [snli.to_dict(), snli.to_dict()])
    test = [replace(r, group_id="snli:wrong") if r.group_id.startswith("snli:") else r for r in splits["test"]]
    write_jsonl(tmp_path / "test.jsonl", [r.to_dict() for r in test])
    report = m.audit_study(tmp_path, tmp_path / "audit.json")
    assert report["checks"]["within_split_duplicates"]["benchmark"]["question_pairs"] == 2
    assert report["checks"]["snli_premise_group_mismatches"]["test"] == 3
    assert report["checks"]["passed"] is False
