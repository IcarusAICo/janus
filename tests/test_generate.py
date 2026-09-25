import json
import random

import pytest

from test_synth import FakeOpenAI  # sibling module; tests/ has no __init__.py and pytest prepends it to sys.path


def test_cells_are_well_formed_and_sampler_is_deterministic():
    from janus.synth.generate import CELLS, sample_spec
    assert len(CELLS) == 9
    for name, cell in CELLS.items():
        assert cell["question_type"] in ("choice", "score", "noul") and cell["state_format"] in ("text", "json")
        assert cell["brief"] and cell["domain_hints"]
        spec = sample_spec(name, random.Random(3))
        assert spec == sample_spec(name, random.Random(3)) and spec["cell"] == name
        if cell["question_type"] == "choice":
            assert 2 <= spec["cardinality"] <= 255
        if cell["question_type"] == "score":
            assert 2 <= spec["levels"] <= 10
    assert CELLS["injection_state"]["adversarial"] is True


def _gen_reply(gold="b"):
    return json.dumps({"instructions": "Which team should handle this?",
                       "options": [{"key": "a", "description": "Payments"}, {"key": "b", "description": "Access"}, {"key": "none", "description": "None of these"}],
                       "gold_key": gold, "state": "I cannot log in to my account.", "rationale": "Login problems go to access."})


def test_generation_and_check_prompts_keep_the_gold_hidden_from_the_checker():
    from janus.synth.generate import check_prompt, generation_prompt, sample_spec, to_request
    spec = sample_spec("routing_text", random.Random(1))
    instructions, user = generation_prompt(spec)
    assert spec["domain"] in user and "gold_key" in instructions
    request = to_request(spec, json.loads(_gen_reply()))
    assert request.questions[0].target == (0., 1., 0.) and request.questions[0].id == "routing_text:q"
    _, check_user = check_prompt(request.state, request.questions[0].to_dict())
    assert "gold" not in check_user.lower() and "rationale" not in check_user.lower() and "Access" in check_user


def test_triage_rules():
    from janus.synth.generate import triage
    assert triage("b", {"answer_key": "b"}, None) == "T3"
    assert triage("b", {"answer_key": "a"}, {"answer_key": "b"}) == "T3_second"
    assert triage("b", {"answer_key": "a"}, {"answer_key": "a"}) == "T4"
    assert triage("b", {"answer_key": "a"}, {"answer_key": "none"}) == "discard"


def test_generate_example_runs_second_check_only_on_disagreement(tmp_path, monkeypatch):
    from janus.synth.generate import generate_example, sample_spec
    from janus.synth.openai_client import StructuredCompleter
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    def luna_reply(user):
        if "You are writing one example" in user or "Write the example" in user:
            return _gen_reply()
        return json.dumps({"answer_key": "a", "confidence": .6})  # checker disagrees

    def terra_reply(user):
        return json.dumps({"answer_key": "b", "confidence": .9})
    luna_client, terra_client = FakeOpenAI(luna_reply), FakeOpenAI(terra_reply)
    luna = StructuredCompleter(cache_dir=tmp_path / "l", client=luna_client)
    terra = StructuredCompleter(model="gpt-5.6-terra", effort="medium", cache_dir=tmp_path / "t", client=terra_client)
    row = generate_example(sample_spec("routing_text", random.Random(2)), luna, terra, random.Random(2))
    # tier is T3 or T4 only; the triage outcome (here T3_second) lives under checks.
    assert row["tier"] == "T3" and row["checks"]["outcome"] == "T3_second"
    assert row["checks"]["first"]["answer_key"] == "a" and row["checks"]["second"]["answer_key"] == "b"
    assert len(luna_client.calls) == 2 and len(terra_client.calls) == 1
    assert "rationale" in row and "gold_key" not in json.dumps(row["questions"])


def test_run_pilot_writes_rows_manifest_and_audit(tmp_path, monkeypatch):
    from janus.synth.generate import run_pilot
    from janus.synth.openai_client import StructuredCompleter
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    counter = {"n": 0}

    def luna_reply(user):
        if "Write the example" in user:
            counter["n"] += 1
            return _gen_reply(gold="b")
        return json.dumps({"answer_key": "b", "confidence": .8})
    luna = StructuredCompleter(cache_dir=tmp_path / "l", client=FakeOpenAI(luna_reply))
    terra = StructuredCompleter(model="gpt-5.6-terra", effort="medium", cache_dir=tmp_path / "t", client=FakeOpenAI(lambda u: json.dumps({"answer_key": "b", "confidence": .9})))
    manifest = run_pilot(tmp_path / "pilot", cells=["routing_text"], per_cell=12, luna=luna, terra=terra)
    assert manifest["cells"]["routing_text"]["accepted_T3"] == 12 and manifest["cells"]["routing_text"]["acceptance_rate"] == 1.
    assert (tmp_path / "pilot" / "routing_text.jsonl").exists() and (tmp_path / "pilot" / "audit-routing_text.md").exists()
    assert "gold" in (tmp_path / "pilot" / "audit-routing_text.md").read_text().lower()
