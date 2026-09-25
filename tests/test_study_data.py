from dataclasses import replace
import importlib.util
import json

import pytest


def study():
    assert importlib.util.find_spec("janus.study_data") is not None, "Study data module is missing"
    from janus import study_data
    return study_data


def test_snli_premises_cannot_leak_across_source_splits():
    m = study()
    rows = {
        "train": [{"premise": "A person runs.", "hypothesis": "Someone moves.", "label": 0},
                  {"premise": "Unique training premise.", "hypothesis": "A training claim.", "label": 1}],
        "validation": [{"premise": "A PERSON  runs.", "hypothesis": "Someone is asleep.", "label": 2},
                       {"premise": "Only validation.", "hypothesis": "A claim.", "label": 1}],
        "test": [{"premise": "a person runs.", "hypothesis": "A person exercises.", "label": 0}],
    }
    split, report = m.split_domain_rows("snli", rows, seed=17, eval_limit=128)
    assert [r["premise"] for r in split["train"]] == ["Unique training premise."]
    assert [r["premise"] for r in split["test"]] == ["a person runs."]
    assert all(r["premise"] == "Only validation." for k in ("dev", "calibration") for r in split[k])
    assert report["lower_priority_group_rows_dropped"] == 2
    sets = [{r["group_id"] for r in items} for items in split.values()]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:])


def test_conflicting_and_invalid_source_labels_are_dropped():
    m = study()
    rows = {"train": [{"text": "A film", "label": 1}, {"text": "a  FILM", "label": 3},
                      {"text": "Invalid", "label": -1}, {"text": "Good", "label": 4}],
            "validation": [], "test": []}
    split, report = m.split_domain_rows("sst5", rows)
    assert [r["text"] for r in split["train"]] == ["Good"]
    assert report["conflicting_states_dropped"] == 1
    assert report["invalid_rows_dropped"] == 1


def test_sst5_uses_real_five_way_ordinal_targets_and_neutral_is_not_positive():
    m = study()
    for label in range(5):
        request = m.sentiment_request({"text": "A movie review", "label": label, "group_id": "sst5:x"})
        score, noul = request.questions
        assert score.kind == "score"
        assert score.target == tuple(float(i == label) for i in range(5))
        assert [o.description for o in score.options] == ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]
        assert noul.target == ((1., 0.) if label <= 2 else (0., 1.))


def test_snli_neutral_is_unknown_and_noul_means_entailed_not_true():
    m = study()
    expected = {0: "entailed", 1: "unknown", 2: "contradicted"}
    for label, category in expected.items():
        request = m.nli_request({"premise": "A runner is outside.", "hypothesis": "The runner won.",
                                 "label": label, "group_id": "snli:x"}, seed=21)
        choice, noul = request.questions
        winner = choice.options[choice.target.index(1.)]
        assert winner.description.startswith(category.capitalize())
        assert noul.target == ((0., 1.) if label == 0 else (1., 0.))
        assert "not establish" in noul.options[0].description.lower()
        assert "false" not in noul.options[0].description.lower()


def test_intent_augmentation_excludes_holdout_descriptions_and_tracks_targets():
    m = study()
    pool = [f"intent_{i}" for i in range(12)]
    rows = [{"domain": "clinc150", "text": f"request {i}", "category": pool[i % 12],
             "group_id": f"clinc150:g{i}"} for i in range(32)]
    initial = [m.intent_request(r, pool, 17) for r in rows]
    first = m.study_training_epoch(initial, 17, 0, rows, {"clinc150": pool})
    second = m.study_training_epoch(initial, 17, 1, rows, {"clinc150": pool})
    assert first == m.study_training_epoch(initial, 17, 0, rows, {"clinc150": pool})
    assert any(a.questions != b.questions for a, b in zip(first, second))
    for request, source in zip(second, rows):
        choice, noul = request.questions
        assert "customer" not in choice.instructions.lower()
        assert "bank" not in noul.instructions.lower()
        winner = choice.options[choice.target.index(1.)].description
        assert winner in {source["category"].replace("_", " "), "None of these intents describes the message."}
    with pytest.raises(ValueError, match="forbidden"):
        m.study_training_epoch(initial, 17, 0, rows, {"clinc150": pool[:-1]})


def test_nli_augmentation_supports_multiple_hypotheses_per_group():
    m = study()
    rows = [{"domain": "snli", "premise": "Shared premise", "hypothesis": f"Claim {label}",
             "label": label, "group_id": "snli:shared"} for label in range(3)]
    initial = [m.nli_request(row, 17) for row in rows]
    result = m.study_training_epoch(initial, 17, 2, rows, {})
    assert [r.state for r in result] == [r.state for r in initial]
    for request, label in zip(result, range(3)):
        choice = request.questions[0]
        expected = ["Entailed", "Unknown", "Contradicted"][label]
        assert choice.options[choice.target.index(1.)].description.startswith(expected)


def test_nli_subset_augmentation_accepts_extra_source_rows_in_same_premise_group():
    m = study()
    rows = [{"domain": "snli", "premise": "Shared premise", "hypothesis": f"Claim {label}",
             "label": label, "group_id": "snli:shared"} for label in range(3)]
    subset = [m.nli_request(rows[1], 17)]
    result = m.study_training_epoch(subset, 17, 2, rows, {})
    assert len(result) == 1
    assert result[0].state == subset[0].state
    with pytest.raises(ValueError, match="source states"):
        m.study_training_epoch(subset, 17, 2, rows[:1], {})


def test_hypotheses_have_unique_question_ids_but_share_the_premise_group():
    m = study()
    requests = [m.nli_request({"premise": "Shared premise", "hypothesis": f"Claim {i}",
                              "label": 0, "group_id": "snli:shared"}) for i in range(3)]
    assert len({r.group_id for r in requests}) == 1
    assert len({q.id for r in requests for q in r.questions}) == 6
    m.assert_study_disjoint({"train": requests})


def test_training_loader_detects_modified_requests_and_separates_source_annotations(tmp_path):
    m = study()
    from janus.data import file_hash, write_json, write_jsonl
    source = {"domain": "sst5", "text": "A good film", "label": 3, "group_id": "sst5:film"}
    request = m.sentiment_request(source)
    train_path = tmp_path / "train.jsonl"
    source_path = tmp_path / "train_sources.jsonl"
    write_jsonl(train_path, [request.to_dict()])
    write_jsonl(source_path, [source])
    write_json(tmp_path / "manifest.json", {"dataset": "JEV_MULTI_DOMAIN_V1", "training_intent_pools": {},
                                             "files": {p.name: file_hash(p) for p in (train_path, source_path)}})
    sources, pools, manifest = m.study_training_sources(train_path)
    assert sources == [source]
    assert "label" not in request.to_dict()
    assert m.augment_study([request], train_path, 17, 1)[0].state == "A good film"
    write_jsonl(train_path, [replace(request, state="Changed review").to_dict()])
    with pytest.raises(ValueError, match="changed"):
        m.study_training_sources(train_path)


def test_global_duplicate_checks_reject_different_domain_ids():
    m = study()
    a = m.sentiment_request({"text": "Shared state", "label": 4, "group_id": "sst5:a"})
    with pytest.raises(ValueError, match="duplicate|overlap"):
        m.assert_study_disjoint({"train": [a, replace(a, group_id="clinc150:b")]})


def test_clinc_heldout_intents_never_enter_training_or_validation():
    m = study()
    categories = [f"intent_{i}" for i in range(8)]
    raw = {name: [{"text": f"{name} {category} {i}", "category": category} for category in categories for i in range(8)]
           for name in ("train", "validation", "test")}
    splits, report = m.split_domain_rows("clinc150", raw, heldout_count=2, eval_limit=5)
    unseen = set(report["unseen_intents"])
    assert len(unseen) == 2
    assert all(r["category"] not in unseen for name in ("train", "dev", "calibration") for r in splits[name])
    assert {r["category"] for r in splits["test"]} == set(categories)
    assert len(splits["dev"]) == 5
    assert len(splits["calibration"]) == 5
    assert m.split_domain_rows("clinc150", raw, heldout_count=2, eval_limit=5) == (splits, report)


def test_benchmark_is_fixed_budget_and_stratified():
    m = study()
    rows = [{"text": f"review {label} {i}", "label": label, "group_id": f"sst5:{label}:{i}"}
            for label in range(5) for i in range(30)]
    panel = m.balanced_rows(rows, 25, seed=17)
    assert len(panel) == 25
    assert [sum(r["label"] == label for r in panel) for label in range(5)] == [5] * 5
    assert panel == m.balanced_rows(rows, 25, seed=17)


def test_study_epoch_threads_none_rate_into_intent_menus():
    m = study()
    pool = [f"intent_{i}" for i in range(12)]
    rows = [{"domain": "clinc150", "text": f"request {i}", "category": pool[i % 12],
             "group_id": f"clinc150:g{i}"} for i in range(64)]
    initial = [m.intent_request(r, pool, 17) for r in rows]

    def none_fraction(requests):
        hits = 0
        for request in requests:
            choice = request.questions[0]
            hits += choice.options[choice.target.index(1.)].description.startswith("None of these")
        return hits / len(requests)
    default = m.study_training_epoch(initial, 17, 0, rows, {"clinc150": pool})
    assert default == m.study_training_epoch(initial, 17, 0, rows, {"clinc150": pool}, omit=.2)
    assert none_fraction(m.study_training_epoch(initial, 17, 0, rows, {"clinc150": pool}, omit=0.)) == 0.
    assert none_fraction(m.study_training_epoch(initial, 17, 0, rows, {"clinc150": pool}, omit=1.)) == 1.
