import datetime as dt
import json
import random

import pytest

from test_synth import FakeOpenAI  # sibling module; pytest prepends tests/ to sys.path


def _gold_key(question):
    return question.options[max(range(len(question.target)), key=question.target.__getitem__)].key


def _gold_description(question):
    return question.options[max(range(len(question.target)), key=question.target.__getitem__)].description


def test_long_context_gold_and_depth_placement():
    from janus.synth.gapcells import DEPTHS, LENGTHS, TRAIN_AGENTS, TRAIN_QUEUES, TRAIN_REGIONS, WORDS_PER_TOKEN, long_context_world
    vocab = {"agents": TRAIN_AGENTS, "queue": TRAIN_QUEUES, "region": TRAIN_REGIONS}
    for seed in range(36):
        rng = random.Random(seed)
        length, depth = LENGTHS[seed % 4], DEPTHS[seed % 3]
        difficulty, style = ("literal", "distractor")[seed % 2], ("prose", "json")[(seed // 2) % 2]
        world, request, evidence = long_context_world(rng, vocab, length, depth, 4 + seed % 5, difficulty, style)
        state, sentence = request.state, world["evidence_sentence"]
        assert state.count(sentence) == 1 and state.count(str(world["ticket"])) == 1
        assert evidence == {"ticket": world["ticket"], "attribute": world["attribute"], "value": world["gold"]}
        pick, probe = request.questions
        assert pick.id == "lc:pick" and _gold_description(pick) == world["gold"] and len(pick.options) == 4 + seed % 5
        assert probe.kind == "noul" and probe.target[1] == float(world["probe"] == world["gold"])
        # depth: locate the evidence in the rendered state and recompute the word fraction before it
        before = len(state[:state.index(sentence)].split()) / len(state.split())
        assert before == pytest.approx(world["depth_actual"], abs=1e-3) and abs(before - depth) < .05
        assert world["length_words"] == round(length * WORDS_PER_TOKEN)
        assert abs(len(state.split()) - world["length_words"]) / world["length_words"] < (.12 if style == "json" else .05)
        others = [c for c in world["candidates"] if c != world["gold"]]
        if difficulty == "literal":
            assert all(state.count(c) == 0 for c in others) and state.count(world["gold"]) == 1
        else:
            assert all(state.count(c) >= 1 for c in others)


def test_nested_record_gold_by_brute_force():
    from janus.synth.gapcells import BUCKETS, TRAIN_CITIES, TRAIN_FLAG_KINDS, nested_record_world
    vocab = {"cities": TRAIN_CITIES, "flag_kinds": TRAIN_FLAG_KINDS + ("exactly_one",)}
    for seed in range(120):
        rng = random.Random(seed)
        world, request, evidence = nested_record_world(rng, vocab, 2 + seed % 5, 4 + seed % 5, 3 + seed % 3, ("literal", "distractor")[seed % 2], "json")
        record = json.loads(request.state)
        field, flags, total = request.questions
        if world["field_kind"] == "address_city":
            assert _gold_description(field) == record["address"]["city"]
            assert "distractor" != world["difficulty"] or all(o["ship_to"]["city"] != record["address"]["city"] for o in record["orders"])
        else:
            assert _gold_description(field) == next(o["status"] for o in record["orders"] if o["order_id"] == world["named_order"])
        fa, fb = record["flags"][world["flag_a"]], record["flags"][world["flag_b"]]
        expected = {"and": fa and fb, "and_not": fa and not fb, "or": fa or fb, "exactly_one": fa != fb}[world["flag_kind"]]
        assert flags.target[1] == float(expected)
        amounts = [o["total"] for o in record["orders"] if world["total_kind"] == "literal" or o["status"] == "delivered"]
        level = sum(sum(amounts) >= edge for edge in BUCKETS[len(total.options)])
        assert total.kind == "score" and total.target[level] == 1.
        assert evidence["flags"] == record["flags"] and evidence["address_city"] == record["address"]["city"]
    _, prose, _ = nested_record_world(random.Random(1), vocab, 3, 4, 4, "distractor", "prose")
    assert "Flags:" in prose.state and "ships to" in prose.state


def _parse(text, fmt):
    return dt.datetime.strptime(text, {"iso": "%Y-%m-%d", "long": "%d %B %Y", "us_long": "%B %d, %Y"}[fmt]).date()


def test_dates_gold_by_brute_force_from_the_rendered_state():
    from janus.synth.gapcells import DATE_FORMATS, DATE_KINDS, dates_world
    vocab = {"formats": DATE_FORMATS, "kinds": tuple(DATE_KINDS)}
    for seed in range(120):
        rng = random.Random(seed)
        world, request, evidence = dates_world(rng, vocab, 2 + seed % 4, ("literal", "narrative")[seed % 2], ("json", "prose")[seed % 2])
        pick, above = request.questions
        fmt = world["format"]
        for option in pick.options:
            assert option.description in request.state
        invoices = [{"number": r["number"], "invoice": dt.date.fromisoformat(r["invoice_date"]), "due": dt.date.fromisoformat(r["due_date"]),
                     "amount": r["amount"]} for r in world["rows"]]
        for r in invoices:  # the world's dates are the ones rendered
            assert r["number"] in request.state
        kind = world["kind"]
        if kind == "earliest_due":
            expected = min(r["due"] for r in invoices)
        elif kind == "latest_invoice":
            expected = max(r["invoice"] for r in invoices)
        elif kind == "invoice_date_of":
            expected = next(r["invoice"] for r in invoices if r["number"] == world["named_number"])
        elif kind == "due_of_largest":
            expected = max(invoices, key=lambda r: r["amount"])["due"]
        else:
            expected = sorted(r["due"] for r in invoices)[1]
        assert _parse(_gold_description(pick), fmt) == expected
        assert len({_parse(o.description, fmt) for o in pick.options}) == len(pick.options)
        total = sum(r["amount"] for r in invoices)
        assert str(world["threshold"]) in above.instructions and above.target[1] == float(total > world["threshold"])
        if world["difficulty"] == "narrative":
            assert len(pick.options) == 2 * len(invoices) + 2
        assert evidence["invoices"] == world["rows"]


def test_dom_gold_matches_the_task_template():
    from janus.synth.gapcells import OPERATIONS, SYNONYMS, TRAIN_LABELS, dom_world
    vocab = {"labels": TRAIN_LABELS}
    seen_ops = set()
    for seed in range(150):
        rng = random.Random(seed)
        difficulty = ("literal", "paraphrase", "distractor")[seed % 3]
        world, request, evidence = dom_world(rng, vocab, 4 + seed % 9, difficulty, ("json", "prose")[seed % 2])
        element, operation = request.questions
        seen_ops.add(world["operation"])
        assert _gold_key(operation) == world["operation"] and tuple(o.key for o in operation.options) == OPERATIONS
        assert len(element.options) == world["cardinality"] + 1 and element.options[-1].key == "none"
        elements = world["elements"]
        assert len({e["index"] for e in elements}) == len(elements) == world["cardinality"]
        if world["operation"] in ("scroll", "done"):
            assert _gold_key(element) == "none"
        else:
            matches = [e for e in elements if e["text"] == world["target_label"] and e["tag"] == world["target_tag"]]
            assert len(matches) == 1 and _gold_key(element) == str(matches[0]["index"])
            twins = [e for e in elements if e["text"] == world["target_label"]]
            assert len(twins) == (2 if difficulty == "distractor" else 1)
            if difficulty == "paraphrase":
                assert SYNONYMS[world["target_label"]] in world["task"] and world["target_label"] not in world["task"]
            else:
                assert world["target_label"] in world["task"]
        assert world["task"] in request.state and evidence["task"]["operation"] == world["operation"]
    assert seen_ops == set(OPERATIONS)


def test_generators_are_deterministic():
    from janus.synth.gapcells import CELL_NAMES, sample_world
    for cell in CELL_NAMES:
        a = sample_world(cell, random.Random(5), "train", "json")
        b = sample_world(cell, random.Random(5), "train", "json")
        assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


def test_prepare_gap_cells_writes_disjoint_splits_with_holdouts(tmp_path):
    from janus.data import assert_disjoint, load_requests
    from janus.synth.gapcells import (TEST_AGENTS, TEST_CITIES, TEST_LABELS, TEST_QUEUES, TEST_REGIONS, TRAIN_CITIES, TRAIN_DATE_FORMATS,
                                    TRAIN_DATE_KINDS, TRAIN_FLAG_KINDS, TRAIN_LABELS, TRAIN_QUEUES, TRAIN_REGIONS, prepare_gap_cells)
    output = tmp_path / "gap"
    manifest = prepare_gap_cells(output, per_cell=12, dev=3, calibration=3, test=6)
    splits = {name: load_requests(output / f"{name}.jsonl") for name in ("train", "dev", "calibration", "test")}
    assert_disjoint(splits)
    assert manifest["dataset"] == "JEV_GAPCELLS_V1" and manifest["total"] == {"train": 48, "dev": 12, "calibration": 12, "test": 24}
    assert set(manifest["holdouts"]) == {"long_context", "nested_record", "dates", "dom"} and manifest["axes"]["long_context"]["max_abs_depth_error"] < .05
    rows = {name: [json.loads(l) for l in (output / f"{name}.jsonl").read_text().splitlines()] for name in splits}
    for name, split_rows in rows.items():
        for row in split_rows:
            assert row["tier"] == "T0" and row["cell"] in manifest["cells"] and row["difficulty"] and row["style"] in ("json", "prose")
            world, test = row["world"], name == "test"
            if row["cell"] == "long_context":
                pool = (TEST_QUEUES + TEST_REGIONS) if test else (TRAIN_QUEUES + TRAIN_REGIONS)
                assert set(world["candidates"]) <= set(pool) and (world["agent"] in TEST_AGENTS) == test
            elif row["cell"] == "nested_record":
                assert world["record"]["address"]["city"] in (TEST_CITIES if test else TRAIN_CITIES)
                assert test or world["flag_kind"] in TRAIN_FLAG_KINDS
            elif row["cell"] == "dates":
                assert test or (world["format"] in TRAIN_DATE_FORMATS and world["kind"] in TRAIN_DATE_KINDS)
            else:
                assert {e["text"] for e in world["elements"]} <= set(TEST_LABELS if test else TRAIN_LABELS)


def test_paraphrase_bank_keeps_only_back_checked_rewrites(tmp_path, monkeypatch):
    from janus.data import write_jsonl
    from janus.synth.openai_client import StructuredCompleter
    from janus.synth.paraphrase_bank import build_paraphrase_bank, distinct_instructions
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    q1, q2 = "Which team should handle this?", "Is the total above 40?"
    rows = [{"state": f"s{i}", "group_id": f"g{i}", "questions": {"q": {"type": "noul", "instructions": q if i % 2 else q1}}} for i, q in enumerate((q1, q2, q1, q2))]
    write_jsonl(tmp_path / "a.jsonl", rows[:2])
    write_jsonl(tmp_path / "b.jsonl", rows[2:])
    assert distinct_instructions((tmp_path / "a.jsonl", tmp_path / "b.jsonl")) == sorted((q1, q2))

    def reply(user):
        body = json.loads(user)
        if "original" in body:
            return json.dumps({"same_question": "WRONG" not in body["rewrite"]})
        original = body["instruction"]
        return json.dumps({"paraphrases": [f"Rewrite one of: {original}", f"Rewrite two of: {original}", original.upper(), f"WRONG {original}", "  "]})
    client = FakeOpenAI(reply)
    completer = StructuredCompleter(cache_dir=tmp_path / "cache", client=client)
    manifest = build_paraphrase_bank(tmp_path / "bank.json", sources=(tmp_path / "a.jsonl", tmp_path / "b.jsonl"), completer=completer)
    bank = json.loads((tmp_path / "bank.json").read_text())
    assert set(bank) == {q1, q2} and all(len(v) == 2 and all(p.startswith("Rewrite") for p in v) for v in bank.values())
    assert manifest["instructions"] == 2 and manifest["requested"] == 10 and manifest["accepted"] == 4
    assert manifest["rejected"] == {"blank": 2, "duplicate": 2, "not_same_question": 2} and manifest["aborted"] is False
    assert len(client.calls) == 2 + 2 * 3  # one generation per instruction, one check per non-blank non-duplicate rewrite
    assert (tmp_path / "bank.manifest.json").exists() and "not verification" in manifest["acceptance_note"]


def _synthetic_rows(counts):
    rows = []
    for outcome, n in counts.items():
        for i in range(n):
            rows.append({"state": f"state {outcome} {i}", "group_id": f"cell:{outcome}{i}", "tier": "T4" if outcome == "T4" else "T3",
                         "questions": {"cell:q": {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B"}, "target": [0., 1.]}},
                         "checks": {"first": {"answer_key": "b"}, "second": None, "outcome": outcome}, "rationale": "r"})
    return rows


def test_stratified_audit_sample_takes_doubted_rows_first():
    from janus.synth.volume import audit_markdown, stratified_audit_sample
    rows = _synthetic_rows({"T3": 300, "T4": 7, "T3_second": 5})
    sample = stratified_audit_sample(rows, 100, random.Random(0))
    outcomes = [r["checks"]["outcome"] for r in sample]
    assert len(sample) == 100 and outcomes[:7] == ["T4"] * 7 and outcomes[7:12] == ["T3_second"] * 5 and set(outcomes[12:]) == {"T3"}
    assert len({r["group_id"] for r in sample}) == 100
    small = stratified_audit_sample(rows, 9, random.Random(0))
    assert [r["checks"]["outcome"] for r in small] == ["T4"] * 7 + ["T3_second"] * 2
    text = audit_markdown("cell", sample, {"T4": 7, "T3_second": 5, "T3": 300})
    assert text.count("\n## ") == 100 and "Gold: ['b']" in text and "First check" in text and text.count("Verdict:") == 100


def test_run_volume_wraps_the_pilot_and_writes_stratified_audits(tmp_path, monkeypatch):
    from janus.synth.openai_client import StructuredCompleter
    from janus.synth.volume import run_volume
    from test_generate import _gen_reply
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    with pytest.raises(ValueError):
        run_volume(tmp_path / "x", cells=["routing_text"], per_cell=1, seed=17, luna=object(), terra=object())
    luna = StructuredCompleter(cache_dir=tmp_path / "l", client=FakeOpenAI(lambda u: _gen_reply("b") if "Write the example" in u else json.dumps({"answer_key": "b", "confidence": .8})))
    terra = StructuredCompleter(model="gpt-5.6-terra", effort="medium", cache_dir=tmp_path / "t", client=FakeOpenAI(lambda u: json.dumps({"answer_key": "b", "confidence": .9})))
    manifest = run_volume(tmp_path / "vol", cells=["routing_text"], per_cell=8, seed=18, luna=luna, terra=terra, audit_size=5, pilot=tmp_path / "no-pilot")
    assert manifest["dataset"] == "JEV_T3_VOLUME_V1" and manifest["cells"]["routing_text"]["accepted_T3"] == 8
    assert manifest["audit"]["routing_text"] == {"size": 5, "T4": 0, "T3_second": 0, "T3": 5, "available": {"T4": 0, "T3_second": 0, "T3": 8}}
    assert manifest["overlap_with_pilot_states"] == {"routing_text": 0} and "not verification" in manifest["acceptance_note"]
    audit = (tmp_path / "vol" / "audit-routing_text.md").read_text()
    assert audit.count("\n## ") == 5 and "stratified" in audit.lower()
