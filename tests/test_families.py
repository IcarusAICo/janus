"""Phase 4 breadth: T0 families (gold recomputed from the world and the rendered state), the T3 top-up cells, the
four-way ablation mixes, and the report."""

import json
import random
import re

import pytest

from janus.data import write_jsonl
from janus.packing import ByteTokenizer
from janus.schema import Request
from janus.synth.families import (FAMILIES, IDENTITY, TEST_NAMES, TEST_SURNAMES, TRAIN_NAMES, TRAIN_SURNAMES, evidence_verdict,
                                evidence_world, prepare_families, record_match_world, retrieval_world, same_person)


def _passages_from_state(state):
    """Parse the numbered passages back out of every rendering."""
    if state.startswith("{"):
        return [r["text"] for r in json.loads(state)["results"]]
    if "| # | passage |" in state:
        return [line.split(" | ", 1)[1][:-2] for line in state.splitlines() if re.match(r"\| \d+ \| ", line)]
    return [line.split("] ", 1)[1] for line in state.splitlines() if re.match(r"\[\d+\] ", line)]


def test_retrieval_gold_is_the_unique_answering_passage_and_levels_drive_every_question():
    seen_none = seen_gold = 0
    for seed in range(300):
        rng = random.Random(seed)
        world, request = retrieval_world(rng, TRAIN_NAMES, TRAIN_SURNAMES, ("prose", "json", "table")[seed % 3])
        pick, rerank, relevance = request.questions
        passages, levels = world["passages"], world["levels"]
        assert _passages_from_state(request.state) == passages and 4 <= len(passages) <= 12
        assert [o.key for o in pick.options] == [f"p{n + 1}" for n in range(len(passages))] + ["none"]
        assert [o.description for o in pick.options][:-1] == passages
        entity, value = world["entity"], world["value"]
        has_value = lambda t: re.search(rf"(?<![\w:]){re.escape(value)}(?![\w:])", t)
        answering = [n for n, t in enumerate(passages) if entity in t and has_value(t)]
        if world["gold"] is None:
            seen_none += 1
            assert levels.count(3) == 0 and pick.target[-1] == 1. and answering == []
        else:
            seen_gold += 1
            assert levels.count(3) == 1 and pick.target[world["gold"]] == 1. and answering == [world["gold"]]
        near = [n for n, l in enumerate(levels) if l == 2]
        assert 1 <= len(near) <= 2 and all(entity in passages[n] and not has_value(passages[n]) for n in near)
        assert all(entity not in passages[n] for n, level in enumerate(levels) if level == 0)
        i, j = world["rerank"]
        assert levels[i] != levels[j] and rerank.target == (float(levels[i] < levels[j]), float(levels[i] > levels[j]))
        assert f"passage {i + 1} " in rerank.instructions and f"passage {j + 1}" in rerank.instructions
        assert relevance.kind == "score" and relevance.target.index(1.) == levels[world["probe"]]
        assert f"passage {world['probe'] + 1} " in relevance.instructions
    assert seen_none > 30 and seen_gold > 150


def test_evidence_label_recomputes_from_the_records_and_not_addressed_claims_are_absent_from_the_state():
    labels = {"supported": 0, "contradicted": 0, "not_addressed": 0}
    kinds = set()
    for seed in range(400):
        rng = random.Random(seed)
        world, request = evidence_world(rng, TRAIN_NAMES, TRAIN_SURNAMES, ("prose", "json", "table")[seed % 3])
        verdict, supported = request.questions
        label = world["label"]
        labels[label] += 1
        kinds.add((world["claim"]["kind"], world["claim"].get("mode")))
        assert evidence_verdict(world["claim"], world["records"]) == label
        assert [o.key for o in verdict.options] == ["supported", "contradicted", "not_addressed"]
        assert verdict.target[list(labels).index(label)] == 1. and supported.target[1] == float(label == "supported")
        assert world["claim_text"] in request.state
        for r in world["records"]:  # every record is rendered with its id, customer, quantity and status
            assert r["id"] in request.state and r["customer"] in request.state and str(r["quantity"]) in request.state
        claim = world["claim"]
        if claim.get("mode") in ("missing_id", "missing_customer"):
            assert claim["subject"] not in request.state.split("Claim")[0].split('"claim"')[0]
        if claim["kind"] == "absent_field":
            assert not any(word in request.state.lower().split("claim")[0] for word in ("driver", "carrier", "invoice", "insured"))
        if claim["kind"] == "equals" and claim.get("mode") is None:
            record = next(r for r in world["records"] if r["id"] == claim["subject"])
            assert (record[claim["field"]] == claim["value"]) == (label == "supported")
    assert min(labels.values()) > 80 and len(kinds) >= 6


def test_record_match_rule_holds_for_every_candidate_and_none_when_no_match():
    none = matches = 0
    modes = set()
    for seed in range(300):
        rng = random.Random(seed)
        world, request = record_match_world(rng, TRAIN_SURNAMES, ("prose", "json", "table")[seed % 3])
        which, same = request.questions
        k = len(world["candidates"])
        assert [o.key for o in which.options] == [f"c{n + 1}" for n in range(k)] + ["none"]
        for n, (candidate, view) in enumerate(zip(world["candidates"], world["views"])):
            is_match = same_person(world["probe"], view, world["entity"], candidate)
            assert is_match == (n == world["gold"]), (seed, n, world["kinds"][n])
            assert which.target[n] == float(is_match)
            shown = {f for f, _, _ in view}
            assert "name" in shown and (shown & set(IDENTITY))
            if n != world["gold"]:
                assert any(candidate[f] != world["entity"][f] for f in shown & {f for f, _, _ in world["probe"]} & set(IDENTITY))
            for _, label, value in view:
                assert label in request.state and value in request.state
        modes |= set(world["kinds"])
        if world["gold"] is None:
            none += 1
            assert which.target[-1] == 1. and same.target[1] == 0.
        else:
            matches += 1
            assert same.target[1] == float(world["probe_noul"] == world["gold"])
        assert f"candidate {world['probe_noul'] + 1} " in same.instructions
    assert none > 30 and matches > 150 and modes >= {"match", "namesake", "relative", "near", "random"}


def test_prepare_families_writes_splits_manifest_holdouts_and_is_deterministic(tmp_path):
    a = prepare_families(tmp_path / "a", ByteTokenizer(), train=9, dev=3, calibration=3, test=6, exclude=(), max_tokens=10 ** 6)
    b = prepare_families(tmp_path / "b", ByteTokenizer(), train=9, dev=3, calibration=3, test=6, exclude=(), max_tokens=10 ** 6)
    assert a["files"] == b["files"] and set(a["files"]) == {"train.jsonl", "dev.jsonl", "calibration.jsonl", "test.jsonl"}
    assert a["counts"]["train"] == {f: 9 for f in FAMILIES} and a["counts"]["test"] == {f: 6 for f in FAMILIES}
    rows = {name: [json.loads(l) for l in (tmp_path / "a" / f"{name}.jsonl").read_text().splitlines()] for name in ("train", "test")}
    assert all(r["tier"] == "T0" and r["family"] == r["group_id"].split(":")[0] and r["world"] for r in rows["train"])
    assert {r["style"] for r in rows["train"]} == {"prose", "json", "table"}
    train_text = " ".join(json.dumps(r) for r in rows["train"])
    test_text = " ".join(json.dumps(r) for r in rows["test"])
    words = lambda text: set(re.findall(r"[A-Za-z]+", text))
    assert not (words(test_text) & set(TRAIN_NAMES + TRAIN_SURNAMES)) and not (words(train_text) & set(TEST_NAMES + TEST_SURNAMES))
    assert all(Request.from_dict(r).questions for r in rows["train"])
    # States already in an excluded data set are redrawn, so the new build never shares a state with it.
    write_jsonl(tmp_path / "old" / "train.jsonl", rows["train"][:2])
    c = prepare_families(tmp_path / "c", ByteTokenizer(), train=9, dev=3, calibration=3, test=6, exclude=(tmp_path / "old",), max_tokens=10 ** 6)
    c_states = {json.dumps(json.loads(l)["state"], sort_keys=True) for l in (tmp_path / "c" / "train.jsonl").read_text().splitlines()}
    assert c["excluded"] == [str(tmp_path / "old")] and c["counts"] == a["counts"]
    assert not any(json.dumps(r["state"], sort_keys=True) in c_states for r in rows["train"][:2])
    with pytest.raises(ValueError, match="over 50"):
        prepare_families(tmp_path / "d", ByteTokenizer(), train=1, dev=1, calibration=1, test=1, exclude=(), max_tokens=50)


def _synthetic(n, seed, **extra):
    from janus.data import synthetic_requests
    return [dict(r.to_dict(), **extra) for r in synthetic_requests(n, seed=seed)]


def _split_dir(root, rows, dev=2, calibration=2):
    root.mkdir(parents=True)
    write_jsonl(root / "train.jsonl", rows[:-dev - calibration])
    write_jsonl(root / "dev.jsonl", rows[-dev - calibration:-calibration])
    write_jsonl(root / "calibration.jsonl", rows[-calibration:])


def test_prepare_ablation_varies_one_axis_at_a_time_and_keeps_evaluation_files_disjoint(tmp_path):
    from janus.synth.ablation import prepare_ablation
    phase1 = _synthetic(10, 1, tier="T0", family="rel") + _synthetic(6, 2, tier="T1", family="study")
    public = _synthetic(8, 3, tier="T1", family="massive", source="AmazonScience/massive") + _synthetic(8, 4, tier="T2", family="civil", source="google/civil_comments")
    families = _synthetic(6, 5, tier="T0", family="retrieval") + _synthetic(6, 6, tier="T0", family="evidence")
    cardinality = _synthetic(6, 7, tier="T0", family="quality")
    for rows in (phase1, public, families, cardinality):
        random.Random(0).shuffle(rows)
    sources = {"phase1": tmp_path / "p1", "public": tmp_path / "pub", "families": tmp_path / "fam", "cardinality": tmp_path / "card"}
    for name, rows in zip(sources, (phase1, public, families, cardinality)):
        _split_dir(sources[name], rows)
    t3 = _synthetic(6, 8, tier="T3", family="routing", cell="routing_text", checks={"outcome": "T3"})
    t3 += _synthetic(2, 9, tier="T3", family="routing", cell="routing_text", checks={"outcome": "T3_second"})
    t3 += _synthetic(2, 10, tier="T4", family="routing", cell="routing_text", checks={"outcome": "T4"})
    (tmp_path / "t3").mkdir()
    write_jsonl(tmp_path / "t3" / "routing_text.jsonl", t3)
    leaked = phase1[0]  # a training state that coincides with an evaluation state must be dropped and recorded
    write_jsonl(tmp_path / "eval.jsonl", [dict(leaked, group_id="other")] + _synthetic(3, 11))
    arms = {"default": {}, "size_6": {"size": 6, "config": "size_10k"}, "no_retrieval": {"drop_family": "retrieval"},
            "tier_t01": {"tiers": ("T0", "T1")}, "two_epochs": {"data": "default", "config": "two_epochs", "epochs": 2}}
    manifest = prepare_ablation(tmp_path / "mixes", ByteTokenizer(), sources=sources, t3=(tmp_path / "t3",), size=12, arms=arms,
                                evaluation_files=(str(tmp_path / "eval.jsonl"),), max_tokens=10 ** 6)
    rows = lambda arm, split="train": [json.loads(l) for l in (tmp_path / "mixes" / arm / f"{split}.jsonl").read_text().splitlines()]
    tiers = manifest["pool"]["tiers"]
    assert manifest["pool"]["rows_after_filters"] == 12 + 12 + 8 + 2 + 6 - 1 == sum(tiers.values()) and tiers["T3"] == 6 and "T4" not in tiers
    assert [d["group_id"] for d in manifest["dropped_for_evaluation_overlap"]] == [leaked["group_id"]]
    assert all(r["tier"] != "T4" and r.get("checks") is None and r["pool"] for r in rows("default"))
    assert len(rows("default")) == 12 and [r["group_id"] for r in rows("size_6")] == [r["group_id"] for r in rows("default")][:6]
    assert all(r["family"] != "retrieval" for r in rows("no_retrieval") + rows("no_retrieval", "dev") + rows("no_retrieval", "calibration"))
    assert manifest["arms"]["no_retrieval"]["held_out_family"] == "retrieval" and "retrieval" not in manifest["arms"]["no_retrieval"]["families_in_train"]
    assert {r["tier"] for r in rows("tier_t01")} <= {"T0", "T1"} and {r["tier"] for r in rows("tier_t01", "dev")} <= {"T0", "T1"}
    assert manifest["arms"]["two_epochs"] == {**manifest["arms"]["two_epochs"], "data": "default", "config": "two_epochs", "epochs": 2}
    assert manifest["arms"]["two_epochs"]["gpu_hours_estimate"] == pytest.approx(2 * manifest["arms"]["default"]["gpu_hours_estimate"])
    assert not (tmp_path / "mixes" / "two_epochs").exists() and manifest["arms"]["default"]["config"] == "default"
    assert set(manifest["axes"]) == {"size", "coverage", "tier", "optimisation"}
    with pytest.raises(ValueError, match="fewer than"):
        prepare_ablation(tmp_path / "big", ByteTokenizer(), sources=sources, t3=(tmp_path / "t3",), size=1000, arms={"default": {}},
                         evaluation_files=(), max_tokens=10 ** 6)


def test_breadth_report_tabulates_each_axis_with_held_out_family_columns(tmp_path):
    from janus.phase4_breadth_report import report
    from janus.data import write_json

    def metrics(acc, families):
        return {"raw": {"accuracy": acc, "nll": 1 - acc}, "by_family": {f: {"raw": {"accuracy": a, "nll": 1 - a}} for f, a in families.items()}}
    for arm, acc in (("default", .8), ("no_retrieval", .7), ("two_epochs", .9)):
        write_json(tmp_path / arm / "summary.json", {"train_requests": 25000, "best_step": 1200, "steps": 1563, "elapsed_seconds": 6000})
        write_json(tmp_path / arm / "families" / "metrics.json", metrics(acc, {"retrieval": acc, "evidence": .5, "record_match": .6}))
        write_json(tmp_path / arm / "phase1" / "metrics.json", metrics(.75, {"rel": .55, "ord": .9}))
    text = report(tmp_path)
    assert text.count("## ") == 4 and "| no_retrieval | 25000 | 1 | 1200 / 1563 | 100 | 0.700 / 0.300 | 0.750 / 0.250 | n/a |" in text
    assert "| 0.700 / 0.300 (held out) | 0.500 / 0.500 | 0.600 / 0.400 | 0.550 / 0.450 |" in text
    assert "| two_epochs | 25000 | 2 | " in text and "| size_10k | n/a | n/a | n/a | n/a | n/a |" in text
    assert "0.800 / 0.200 | 0.500 / 0.500 | 0.600 / 0.400 | 0.550 / 0.450 |" in text.split("## coverage")[1].split("\n")[4]


def test_t3_topup_registers_cells_for_the_run_only_and_writes_stratified_audits(tmp_path, monkeypatch):
    from janus.synth.families import T3_CELLS, run_t3_topup
    from janus.synth.generate import CELLS
    from janus.synth.openai_client import StructuredCompleter
    from test_synth import FakeOpenAI
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    for name, cell in T3_CELLS.items():
        assert name not in CELLS and cell["brief"] and cell["domain_hints"] and cell["difficulty"]
    assert T3_CELLS["rubric_grading"]["question_type"] == "score" and T3_CELLS["rubric_grading"]["levels"] == [3, 4, 5, 6, 7]
    assert T3_CELLS["tool_argument_none"]["none_option"] is True and T3_CELLS["tool_argument_none"]["state_format"] == "json"

    def luna_reply(user):
        if "Write the example" in user:
            if '"cell": "rubric_grading"' in user:
                return json.dumps({"instructions": "Which level does the ARTEFACT earn?", "gold_key": "L2", "state": "ARTEFACT: a short reply.",
                                   "options": [{"key": f"L{i}", "description": json.dumps({"summary": f"level {i}", "signals": "..."})} for i in range(1, 4)],
                                   "rationale": "audit only"})
            return json.dumps({"instructions": "Which value does the request state?", "gold_key": "none", "state": json.dumps({"request": "hi", "tool_schema": {}}),
                               "options": [{"key": "a", "description": "A"}, {"key": "b", "description": "B"}, {"key": "none", "description": "Not stated"}],
                               "rationale": "audit only"})
        return json.dumps({"answer_key": "1" if "ARTEFACT" in user else "none", "confidence": .9})
    luna = StructuredCompleter(cache_dir=tmp_path / "l", client=FakeOpenAI(luna_reply))
    terra = StructuredCompleter(model="gpt-5.6-terra", effort="medium", cache_dir=tmp_path / "t", client=FakeOpenAI(lambda u: json.dumps({"answer_key": "none", "confidence": .9})))
    manifest = run_t3_topup(tmp_path / "topup", per_cell=6, luna=luna, terra=terra, workers=2)
    assert len(CELLS) == 9 and manifest["consensus_rows"] == {"rubric_grading": 6, "tool_argument_none": 6}
    assert manifest["none_gold_rate"] == {"rubric_grading": None, "tool_argument_none": 1.}
    assert manifest["seed"] == 19 and manifest["models"]["second_check"] == "gpt-5.6-terra"
    for cell in T3_CELLS:
        rows = [json.loads(l) for l in (tmp_path / "topup" / f"{cell}.jsonl").read_text().splitlines()]
        assert len(rows) == 6 and all(r["tier"] == "T3" and r["cell"] == cell for r in rows)
        assert "stratified" in (tmp_path / "topup" / f"audit-{cell}.md").read_text()
        assert "uniform consensus rows" in (tmp_path / "topup" / f"audit2-{cell}-T3.md").read_text() and manifest["audit"][cell]["uniform_consensus_size"] == 6
    assert Request.from_dict(rows[0]).questions[0].kind == "choice"
    with pytest.raises(ValueError, match="pilot"):
        run_t3_topup(tmp_path / "again", seed=17, luna=luna, terra=terra)
